"""PRECISION BY LAYER: fit the whole model in RAM instead of streaming part of it.

THE MEASUREMENT THAT MOTIVATES THIS FILE
    Streaming works and is exact, but it is slow for one reason. To know which expert to fetch,
    the engine reads the router's output back from the GPU at every layer, and that read stops
    MLX pipelining. Measured on Qwen3-30B / M4 Air:

        the model's kernels alone            21.0 ms/token   47.1 tok/s
        the engine, 60% of experts resident 119.4 ms/token    7.8 tok/s
        the reads alone                      98.5 ms/token

    Nothing is fetched if nothing is ever missing. So: keep EVERY expert in RAM, and buy the room
    by storing them in fewer bits. There is then no fetch, no read-back, no stall -- and the cost
    moves from "wall-clock" to "quantisation error", which is a thing the meter can measure.

WHY PER LAYER AND NOT PER EXPERT
    Per-expert precision means two pool arrays per layer -- one for the hot experts, one for the
    cold -- because `mx.gather_qmm` needs a single array of a single bit width. Serving a token
    then reads from BOTH, which doubles the expert bytes per token: measured, the experts are
    793 MB of a 1331 MB token, so doubling them costs about 1.6x in wall-clock. Per-layer
    precision keeps one gather per layer and costs nothing at all.

    So per-layer is the default. Per-expert is only worth its 1.6x if it buys more than 1.6x of
    quality, which is an empirical question and not assumed here.

WHAT WAS BUILT AND THEN REMOVED
    An earlier version requantised the loaded model in memory and wrote a standard MLX model
    directory, which would have run with stock mlx_lm and no engine at all. It was deleted, not
    because it does not work, but because it costs the full model in RAM to produce -- 13.4 GB
    for the model this product exists to run on machines that do not have 13.4 GB. Compressing
    the blob on disk gets the same result at a few megabytes of peak memory. Exporting a plain
    model directory is still a reasonable feature; it just cannot be the mechanism.

THE OUTPUT IS A BLOB THE POOL LOADS AT FULL RESIDENCY
    A compressed blob is loaded into the expert pool at C == E. Every expert is resident, so the
    pool's sync-free path applies and the host is never read during a forward pass. Measured on
    OLMoE-1B-7B: 95-111 tok/s at every precision from 2 to 4 bits, against 48.3 tok/s for the
    same pool before the sync-free path existed. The speed comes from residency, not from the
    bits -- which is why the strategy compresses only as far as it must, and no further.
"""
from __future__ import annotations

import json
import os
import time

from . import home
import mlx.core as mx
import numpy as np

from .stream import PROJECTIONS

# bytes per parameter for an affine-quantised weight, including the fp16 scale and bias per group
def bytes_per_param(bits: int, group_size: int) -> float:
    return (bits + 32.0 / group_size) / 8.0


CANDIDATES = [(2, 128), (2, 64), (3, 128), (3, 64), (4, 128), (4, 64), (6, 64), (8, 64)]


# --------------------------------------------------------------------------- allocation
def allocate(sensitivity: dict, param_counts: dict, budget_bytes: float,
             candidates=None) -> dict:
    """Choose (bits, group_size) per layer to fit a byte budget at the least predicted damage.

    `sensitivity[layer]` is the measured cost of taking THAT layer to the reference low setting,
    in nats per token. It is used as a per-layer weight: a layer that hurt twice as much when
    compressed is assumed to hurt twice as much at any given precision. That is an assumption,
    not a measurement, and it is why the allocation is verified end to end afterwards rather
    than trusted.

    Greedy by marginal damage per byte saved, which is optimal for a separable objective and is
    what makes the result reproducible rather than a search artifact.
    """
    cands = sorted(candidates or CANDIDATES, key=lambda c: bytes_per_param(*c))
    if not cands:
        raise ValueError("no candidate precisions given")
    layers = sorted(param_counts)
    # start everyone at the most generous setting, then step down where it costs least
    cur = {l: len(cands) - 1 for l in layers}

    def total():
        return sum(param_counts[l] * bytes_per_param(*cands[cur[l]]) for l in layers)

    if total() <= budget_bytes:
        return {l: cands[cur[l]] for l in layers}

    lowest = 0
    while total() > budget_bytes:
        best, best_ratio = None, None
        for l in layers:
            if cur[l] <= lowest:
                continue
            b_now, g_now = cands[cur[l]]
            b_next, g_next = cands[cur[l] - 1]
            saved = param_counts[l] * (bytes_per_param(b_now, g_now) -
                                       bytes_per_param(b_next, g_next))
            if saved <= 0:
                continue
            # damage scales with the layer's measured sensitivity and with how far it drops
            damage = sensitivity.get(l, 1.0) * (bytes_per_param(b_now, g_now) /
                                                bytes_per_param(b_next, g_next) - 1.0)
            ratio = damage / saved
            if best_ratio is None or ratio < best_ratio:
                best, best_ratio = l, ratio
        if best is None:
            raise ValueError(
                f"cannot reach {budget_bytes/1e9:.2f} GB: every layer is already at the lowest "
                f"allowed precision {cands[lowest]}, which needs "
                f"{total()/1e9:.2f} GB.")
        cur[best] -= 1
    return {l: cands[cur[l]] for l in layers}


def plan_bytes(plan: dict, param_counts: dict) -> float:
    return sum(param_counts[l] * bytes_per_param(*plan[l]) for l in plan)


# --------------------------------------------------------------------------- on-disk requantise
def _read_seg(seg) -> bytes:
    fd = os.open(seg.path, os.O_RDONLY)
    try:
        os.lseek(fd, seg.offset, os.SEEK_SET)
        parts, got = [], 0
        while got < seg.length:
            b = os.read(fd, min(1 << 22, seg.length - got))
            if not b:
                raise IOError(f"short read at {seg.offset}+{got} in {seg.path}")
            parts.append(b)
            got += len(b)
        return b"".join(parts)
    finally:
        os.close(fd)


def requantize_blob(src_blob: str, dst_blob: str, plan: dict, progress=True,
                    manifest: dict | None = None) -> dict:
    """Rewrite an expert blob at new per-layer precisions, one expert at a time.

    WHY ON DISK AND NOT IN MEMORY
        Compressing a 13.4 GB model by loading it costs 13.4 GB, which is more than the machine
        this is built for has spare -- the entire premise of the product. The blob is already
        expert-contiguous, so an expert can be read, dequantised, requantised and written back
        while never holding more than one of them. Peak memory is a few megabytes regardless of
        model size.
    """
    from .stream import load_manifest
    src_blob, dst_blob = os.path.expanduser(src_blob), os.path.expanduser(dst_blob)
    os.makedirs(os.path.dirname(dst_blob), exist_ok=True)
    man = manifest if manifest is not None else load_manifest(src_blob)
    if man.get("version", 0) < 4:
        raise ValueError(
            f"{src_blob} was packed before blobs carried their quantisation parameters "
            f"(version {man.get('version')}). Re-run `bigrig prepare` on the model first; "
            f"guessing the source precision would decode every expert wrong.")
    from .stream import store_from_manifest
    store = store_from_manifest(src_blob, man)
    out = {"regions": {}, "layers": {}, "version": 4, "top_k": man.get("top_k")}
    off = 0
    try:
        with open(dst_blob, "wb") as fo:
            for lk in sorted(man["layers"], key=int):
                li = int(lk)
                info = man[
                    "layers"][lk]
                sq = info["quant"]
                bits, group = plan.get(li, (sq["bits"], sq["group_size"]))
                spec_out, first = {}, True
                E = info["n_experts"]
                for e in range(E):
                    # segments(), not region(): the source may be the model's own safetensors,
                    # where one expert is nine ranges across up to three shards.
                    buf = b"".join(_read_seg(sg) for sg in store.segments((li, e)))
                    pos, start = 0, off
                    # The projections this model carries, in the order its bytes are laid
                    # out -- gate/up/down for most, fc1/fc2 for Nemotron.
                    for proj in info["spec"]:
                        parts = {}
                        for comp, meta in info["spec"][proj].items():
                            raw = np.frombuffer(buf, dtype=np.uint8, count=meta["nbytes"],
                                                offset=pos)
                            pos += meta["nbytes"]
                            parts[comp] = (mx.array(raw)
                                           .view(getattr(mx, meta["dtype"]))
                                           .reshape(meta["shape"]))
                        deq = mx.dequantize(parts["weight"], parts["scales"],
                                            parts.get("biases"), group_size=sq["group_size"],
                                            bits=sq["bits"], mode=sq.get("mode", "affine"))
                        w, sc, *bi = mx.quantize(deq, group_size=group, bits=bits)
                        mx.eval(w, sc, *bi)
                        newp = {"weight": w, "scales": sc}
                        if bi:
                            newp["biases"] = bi[0]
                        if "bias" in parts:
                            newp["bias"] = parts["bias"]
                        if first:
                            spec_out[proj] = {}
                        for comp in ("weight", "scales", "biases", "bias"):
                            if comp not in newp:
                                continue
                            a = newp[comp]
                            b = np.array(mx.contiguous(a).view(mx.uint8), copy=False).reshape(-1)
                            if first:
                                spec_out[proj][comp] = {
                                    "shape": [int(x) for x in a.shape],
                                    "dtype": str(a.dtype).rsplit(".", 1)[-1],
                                    "nbytes": int(b.nbytes)}
                            fo.write(b.tobytes())
                            off += b.nbytes
                        del deq, w, sc, newp, parts
                    if pos != len(buf):
                        raise ValueError(
                            f"layer {li} expert {e}: consumed {pos} of {len(buf)} source bytes")
                    first = False
                    out["regions"][f"{li}:{e}"] = [start, off - start]
                    mx.clear_cache()
                per = sum(c["nbytes"] for p in spec_out.values() for c in p.values())
                out["layers"][lk] = {"n_experts": E, "spec": spec_out, "bytes_per_expert": per,
                                     "quant": {"group_size": group, "bits": bits,
                                               "mode": "affine"}}
                if progress:
                    print(f"    layer {li:>3}: {sq['bits']}b g{sq['group_size']} -> "
                          f"{bits}b g{group}   {E*per/1e9:.3f} GB", flush=True)
    finally:
        pass
    out["total_bytes"] = off
    with open(dst_blob + ".manifest.json", "w") as f:
        json.dump(out, f)
    return out


def compressed_path(blob: str, bits: int, group: int) -> str:
    return f"{os.path.expanduser(blob)}.q{bits}g{group}"


def ensure_compressed(blob: str, bits: int, group: int, verbose: bool = True,
                      manifest: dict | None = None, name: str = "") -> str:
    """Requantise experts to (bits, group) once and cache the result.

    The source may be a packed blob OR the model's own safetensors -- `manifest` says which, and
    the compressed output is always a blob of its own, because requantised weights exist nowhere
    else.
    """
    from .stream import load_manifest
    blob = os.path.expanduser(blob)
    if not blob:
        blob = os.path.join(home(), "data", "blobs",
                            (name or "model") + ".experts")
    dst = compressed_path(blob, bits, group)
    # A tuned copy at the same size is strictly better -- same memory, measured 3.9% better
    # perplexity -- so it wins if one exists. Without this the tuner writes a file nothing ever
    # opens, and `--tune` becomes an expensive no-op.
    tuned = dst + ".tuned"
    if os.path.exists(tuned) and os.path.exists(tuned + ".manifest.json"):
        try:
            tm = load_manifest(tuned)
            if os.path.getsize(tuned) == tm["total_bytes"]:
                if verbose:
                    print(f"  using the tuned {bits}-bit copy (per-layer precision)")
                return tuned
        except (ValueError, KeyError, OSError):
            pass                       # a damaged tuned copy is not a reason to fail
    if os.path.exists(dst) and os.path.exists(dst + ".manifest.json"):
        m = load_manifest(dst)
        if os.path.getsize(dst) == m["total_bytes"]:
            return dst
        if verbose:
            print(f"  cached {os.path.basename(dst)} is truncated; rebuilding")
    src = manifest if manifest is not None else load_manifest(blob)
    n_layers = len(src["layers"])
    if verbose:
        print(f"  compressing to {bits}-bit g{group} (one time, "
              f"{src['total_bytes']/1e9:.1f} GB to read) ...")
    t0 = time.perf_counter()
    m = requantize_blob(blob, dst, {int(k): (bits, group) for k in src["layers"]},
                        progress=False, manifest=src)
    if verbose:
        print(f"  compressed {src['total_bytes']/1e9:.2f} GB -> {m['total_bytes']/1e9:.2f} GB "
              f"in {time.perf_counter()-t0:.0f}s across {n_layers} layers")
    return dst


# WEIGHTS THAT ARE ON DISK AND NEVER REACH MEMORY.
#     A multimodal checkpoint ships a vision tower that mlx_lm's `sanitize` drops before the
#     model is built -- `Qwen3_5MoeForConditionalGeneration` is a text model here, and the
#     tower is dead bytes in a file. These are the exact prefixes mlx_lm skips; matching its
#     rule rather than inventing one is what stops this from disagreeing with the loader.
DISCARDED_PREFIXES = ("vision_tower", "model.visual")


def discarded_gb(model_dir: str) -> float:
    """Bytes in the checkpoint that the loader throws away, so the planner stops reserving them.

    Charging for them is not a rounding error. On Qwen3.6-35B-A3B-8bit the tower is 0.89 GB of
    a 3.50 GB resident estimate, and against a 9 GB ceiling that was the whole difference
    between a pool of 5 experts a layer and 11 -- the planner refused a model that fits, with
    the message that it needed 1.1 GB and had 0.7 GB, while 0.89 GB of what it had reserved was
    never going to be read.
    """
    import glob as _g
    from .direct import read_header
    model_dir = os.path.expanduser(model_dir)
    total = 0
    for f in _g.glob(os.path.join(model_dir, "*.safetensors")):
        try:
            meta, _ = read_header(f)
        except (OSError, ValueError):
            continue
        for name, spec in meta.items():
            if name == "__metadata__" or not isinstance(spec, dict):
                continue
            if not name.startswith(DISCARDED_PREFIXES):
                continue
            off = spec.get("data_offsets")
            if isinstance(off, (list, tuple)) and len(off) == 2:
                total += int(off[1]) - int(off[0])
    return total / 1e9


def non_expert_gb(model_dir: str, blob: str = "", manifest: dict | None = None) -> float:
    """Bytes of the model that are NOT experts -- attention, embeddings, the lm_head.

    They are always resident whatever the engine does, so every memory decision has to account
    for them. On Qwen3-30B they are 0.67 GB against 12.68 GB of experts; on a small model they
    are a much larger share and forgetting them is how a plan overruns.

    Weights the loader discards are NOT resident and are subtracted -- see `discarded_gb`.
    """
    from .stream import load_manifest
    import glob as _g
    model_dir = os.path.expanduser(model_dir)
    tot = sum(os.path.getsize(f) for f in _g.glob(os.path.join(model_dir, "*.safetensors")))
    if not tot:
        return 0.0
    m = manifest if manifest is not None else load_manifest(blob)
    return max(0.0, (tot - m["total_bytes"]) / 1e9 - discarded_gb(model_dir))
