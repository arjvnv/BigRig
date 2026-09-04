"""Measure which layers can afford to be compressed, then spend the budget accordingly.

WHY THIS IS WORTH DOING
    Compressing every layer by the same amount treats them as interchangeable. They are not.
    Measured on OLMoE-1B-7B, dropping ONE layer to the floor precision costs between +0.009 and
    +0.075 nats depending on which layer -- an 8.4x spread, rising monotonically with depth.

    Spending the same memory in proportion to that measurement is worth 3.9% better perplexity
    (6.178 against 6.427 for uniform, at 2.78 GB against 2.82 GB).

WHY IT IS OPT-IN AND NOT THE DEFAULT
    The sensitivity has to be MEASURED, per model, and that costs one evaluation pass per layer:
    a few minutes for a 16-layer model, closer to an hour for a 48-layer one. That is a fine
    thing to ask for once and cache forever, and a terrible thing to impose on someone who just
    wants to run a model.

    An earlier version of the allocator did NOT measure -- it scored damage as linear in how far
    a layer dropped. The real damage is a cliff at 2 bits, a linear model cannot see a cliff, and
    the resulting plan was WORSE than uniform (7.228 against 6.427). Constrained above the cliff
    it wins. That failure is why the floor is enforced here rather than left to the caller.

HOW A LAYER IS TESTED WITHOUT REBUILDING ANYTHING
    Requantising a layer to a different bit width changes the packed size, so it no longer fits
    the pool slots that were allocated for it. Instead the weights are round-tripped THROUGH the
    target precision and back to the original format: dequantise, requantise low, dequantise,
    requantise back. The values then carry exactly the error the lower precision would introduce
    while the storage shape is untouched, so the pool never has to be rebuilt and peak memory is
    one projection of one layer.
"""
from __future__ import annotations

import json
import os
import time

import mlx.core as mx
import numpy as np

from .precision import CANDIDATES, allocate, bytes_per_param, plan_bytes
from .stream import PROJECTIONS

# Never probe below this. The cliff is measured, and a sensitivity number taken from the far side
# of it describes damage no shipped plan is allowed to cause anyway.
PROBE_BITS, PROBE_GROUP = 3, 128
FLOOR_CANDIDATES = [(3, 128), (3, 64), (4, 128), (4, 64), (6, 64), (8, 64)]


def _slots(pool, proj):
    s = pool._slots
    return (s[(proj, "weight")], s.get((proj, "scales")), s.get((proj, "biases")))


def simulate_precision(pool, bits: int, group: int) -> dict:
    """Make one layer's resident experts carry the error of `bits`, without changing their shape.

    Returns a snapshot the caller must pass to `restore`. The snapshot is one layer of weights;
    for a 128-expert Qwen3 layer that is ~264 MB, which is the peak cost of the whole scan.
    """
    q = pool.spec_quant
    snap = {}
    for proj in pool.projections:
        w, sc, bi = _slots(pool, proj)
        snap[proj] = (w, sc, bi)
        deq = mx.dequantize(w, sc, bi, group_size=q["group_size"], bits=q["bits"],
                            mode=q.get("mode", "affine"))
        lw, lsc, *lbi = mx.quantize(deq, group_size=group, bits=bits)
        del deq
        low = mx.dequantize(lw, lsc, lbi[0] if lbi else None, group_size=group, bits=bits)
        del lw, lsc
        nw, nsc, *nbi = mx.quantize(low, group_size=q["group_size"], bits=q["bits"])
        del low
        mx.eval(nw, nsc, *nbi)
        pool._slots[(proj, "weight")] = nw
        pool._slots[(proj, "scales")] = nsc
        if nbi and (proj, "biases") in pool._slots:
            pool._slots[(proj, "biases")] = nbi[0]
        mx.clear_cache()
    return snap


def restore(pool, snap: dict) -> None:
    for proj, (w, sc, bi) in snap.items():
        pool._slots[(proj, "weight")] = w
        if sc is not None:
            pool._slots[(proj, "scales")] = sc
        if bi is not None and (proj, "biases") in pool._slots:
            pool._slots[(proj, "biases")] = bi
    mx.clear_cache()


def scan(model, handle, ids, evaluate, window=512, stride=512, windows=3, guard=None,
         log=print) -> dict:
    """Per-layer sensitivity: nats added when that layer alone drops to the probe precision.

    Every layer is measured against the SAME baseline and restored before the next, so the
    numbers are independent of the order they were taken in.
    """
    base = evaluate.perplexity(model, ids, window=window, stride=stride,
                               max_windows=windows, guard=guard)
    log(f"  baseline nll {base['nll']:.5f} over {base['tokens']:,} tokens")
    out = {}
    for i, pool in enumerate(handle.pools):
        t0 = time.perf_counter()
        snap = simulate_precision(pool, PROBE_BITS, PROBE_GROUP)
        try:
            r = evaluate.perplexity(model, ids, window=window, stride=stride,
                                    max_windows=windows, guard=guard)
            d = r["nll"] - base["nll"]
        finally:
            restore(pool, snap)
            del snap
            mx.clear_cache()
        out[pool.layer] = d
        log(f"    layer {pool.layer:>3}: d_nll {d:+.5f}   [{time.perf_counter()-t0:.0f}s]")
    return {"baseline_nll": base["nll"], "probe": [PROBE_BITS, PROBE_GROUP],
            "sensitivity": {str(k): v for k, v in out.items()},
            "window": window, "windows": windows, "tokens": base["tokens"]}


# ------------------------------------------------------------------ caching
def profile_path(blob_or_model: str) -> str:
    return os.path.expanduser(blob_or_model) + ".tune.json"


def load_profile(path: str) -> dict | None:
    p = profile_path(path)
    if not os.path.exists(p):
        return None
    try:
        with open(p) as f:
            d = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
    return d if isinstance(d, dict) and d.get("sensitivity") else None


def save_profile(path: str, prof: dict) -> str:
    p = profile_path(path)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    tmp = p + ".tmp"
    with open(tmp, "w") as f:
        json.dump(prof, f, indent=1)
    os.replace(tmp, p)
    return p


# ------------------------------------------------------------------ planning
def plan_from_profile(prof: dict, manifest: dict, budget_bytes: float,
                      min_bits: int = 3) -> dict:
    """Turn measured sensitivity into a per-layer precision plan that fits `budget_bytes`."""
    layers = sorted(manifest["layers"], key=int)
    q = manifest["layers"][layers[0]]["quant"]
    per_layer_params = (manifest["layers"][layers[0]]["bytes_per_expert"]
                        * manifest["layers"][layers[0]]["n_experts"]
                        / bytes_per_param(q["bits"], q["group_size"]))
    counts = {int(k): per_layer_params for k in layers}
    sens = {int(k): max(float(v), 1e-6) for k, v in prof["sensitivity"].items()}
    missing = set(counts) - set(sens)
    if missing:
        raise ValueError(
            f"the tuning profile is missing {len(missing)} of {len(counts)} layers "
            f"(e.g. {sorted(missing)[:3]}). Re-run `rig compress --tune`; allocating from a "
            f"partial measurement would spend the budget on guesses.")
    cands = [c for c in FLOOR_CANDIDATES if c[0] >= min_bits] or [(min_bits, 128)]
    return allocate(sens, counts, budget_bytes, candidates=cands)


def describe_plan(plan: dict, manifest: dict) -> str:
    hist = {}
    for bg in plan.values():
        hist[bg] = hist.get(bg, 0) + 1
    return ", ".join(f"{n} layer{'s' if n != 1 else ''} at {b}-bit g{g}"
                     for (b, g), n in sorted(hist.items()))
