"""What a model actually needs for scratch, measured on this machine, remembered per model.

WHY THIS EXISTS. Every model was charged a flat 3.0 GB of "working memory" -- the transient
scratch a forward pass allocates on top of the resident weights and the KV cache. That number
was measured on two large models and applied to all of them, and it is the largest single item
in the memory a small model reserves: on a 17 GB model at a ~6.7 GB floor, 3.0 of it is this,
and the model may use a third of that. Measured across seven models the real need runs from
0.68 GB to 2.90 GB. Charging the maximum to every model is what keeps a 17 GB model off an
8 GB Mac it would otherwise fit.

WHAT IS MEASURED, AND WHY IT IS THE SAFE NUMBER. `mx.get_peak_memory()` over a real forward
pass, minus the memory already resident before it -- MLX's own count of the anonymous and wired
scratch it allocated. That, not the process footprint, is what a reserve must cover: anonymous
and wired memory is what a Mac runs out of, while the page cache (which the footprint also
counts) is evicted first and harmlessly. The footprint is recorded too, as a cross-check, but
the reserve is sized from the MLX peak.

THE RULE IS ONE-DIRECTIONAL: IT CAN ONLY EVER LOWER THE RESERVE. `reserve_from` clamps the
measured peak to the flat default as a ceiling, so a model whose measurement meets or exceeds
3.0 (GLM and Nemotron both approach it) is charged exactly what it is today -- this never makes
a plan more aggressive than the shipped one. Below the ceiling it applies a margin and a floor.
The stored value is a running MAXIMUM across runs, so a heavier workload than the tune's probe
raises it for next time and never a lighter one lowers it. Safety only ratchets tighter.

WHERE IT COMES FROM. The first-run tune already builds the model at its chosen capacity and runs
real generations; the peak is captured there and written before the model is served, so the
reduction lands on the first serve. Served sessions also record passively, so a real workload
that exceeds the tune's synthetic one tightens the next run's reserve.
"""
from __future__ import annotations

import json
import os

from . import home

# The transient peak is bounded by the prefill pass width, which is a function of the model, not
# the prompt -- so this margin covers run-to-run variance and a little slack, not an unbounded
# tail. 1.3x over the measured peak plus 0.3 GB: at the largest sub-ceiling measurement
# (DeepSeek, 1.88 GB) that is 2.75 GB, comfortably under the 3.0 it replaces; at the smallest
# streamed one (Qwen3-30B, 0.68) it is 1.18, floored to 1.0.
MARGIN = 1.3
SLACK_GB = 0.3
FLOOR_GB = 1.0
# A run must push at least this many tokens through a prefill of at least this width before its
# peak is trusted as representative. A two-token "hi" never exercises the widest pass and would
# record a peak far below what a real prompt needs -- the one way this could become unsafe.
MIN_PREFILL_TOKENS = 64
MIN_DECODE_TOKENS = 16


def _dir() -> str:
    return os.path.join(home(), "data", "results")


def path(model_name: str, budget_gb: float | None = None) -> str:
    """One file per model AND budget. The prefill width, and so the scratch peak, can differ by
    budget, so a measurement at 9.7 GB is not silently reused at 5.6 GB -- the same reasoning as
    the knee's per-budget file."""
    safe = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in model_name)
    if budget_gb is None:
        return os.path.join(_dir(), f"workmem_{safe}.json")
    return os.path.join(_dir(), f"workmem_{safe}@{float(budget_gb):.1f}.json")


def load(model_name: str, budget_gb: float | None = None) -> dict | None:
    """This model's recorded scratch peak AT THIS BUDGET, or None.

    Deliberately does NOT fall back to a reading taken at another budget. A tighter budget holds
    a smaller pool, misses more, and so allocates MORE transient scratch -- reusing a looser
    budget's lower reading would under-reserve and is the one way this could crash a Mac. No
    reading at this exact budget means the flat default stands, which is always safe."""
    cands = [path(model_name, budget_gb)] if budget_gb is not None else []
    cands.append(path(model_name))          # the budget-less file, written only when budget is None
    for p in cands:
        if not os.path.exists(p):
            continue
        try:
            with open(p) as fh:
                d = json.load(fh)
        except (OSError, ValueError):
            continue
        if isinstance(d, dict) and isinstance(d.get("peak_gb"), (int, float)) and d["peak_gb"] > 0:
            return d
    return None


def record(model_name: str, budget_gb: float, peak_gb: float, footprint_gb: float = 0.0,
           prefill_tokens: int = 0, decode_tokens: int = 0) -> str | None:
    """Store this reading if it is representative, as a running maximum. Returns the file path
    when it wrote, None when the run was too small to trust or the reading did not exceed what
    is already recorded.

    Representativeness is the safety gate: a run that never drove a wide prefill or barely
    generated could peak far below a real workload, and trusting it would size the next pool too
    large. Such a run is dropped, not recorded low."""
    if peak_gb <= 0:
        return None
    if prefill_tokens < MIN_PREFILL_TOKENS or decode_tokens < MIN_DECODE_TOKENS:
        return None
    p = path(model_name, budget_gb)
    prev = None
    if os.path.exists(p):
        try:
            with open(p) as fh:
                prev = json.load(fh)
        except (OSError, ValueError):
            prev = None
    # A running maximum: only a HIGHER peak is written, so safety can only tighten. A lighter
    # workload never lowers the recorded need.
    if isinstance(prev, dict) and isinstance(prev.get("peak_gb"), (int, float)) \
            and prev["peak_gb"] >= peak_gb:
        return None
    out = {"model": model_name, "budget_gb": round(float(budget_gb), 2),
           "peak_gb": round(float(peak_gb), 3), "footprint_gb": round(float(footprint_gb), 3),
           "prefill_tokens": int(prefill_tokens), "decode_tokens": int(decode_tokens)}
    try:
        os.makedirs(_dir(), exist_ok=True)
        tmp = p + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(out, fh, indent=1)
        os.replace(tmp, p)
    except OSError:
        return None
    return p


def reserve_from(peak_gb: float, default_gb: float) -> float:
    """The working-memory figure to reserve, given a measured peak and the flat default.

    Clamped to the default as a hard ceiling -- this never asks for more scratch than the
    shipped constant, so it can only free memory, never take it. Below the ceiling: the peak
    plus a margin and a little slack, floored so an unusually small reading still leaves room."""
    if peak_gb <= 0:
        return float(default_gb)
    sized = peak_gb * MARGIN + SLACK_GB
    return round(max(FLOOR_GB, min(float(default_gb), sized)), 2)
