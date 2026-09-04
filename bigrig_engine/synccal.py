"""Measure what the host round-trip costs on THIS model, so the planner can stop guessing.

WHY A MEASUREMENT AND NOT A CONSTANT
    A streamed layer reads its router's output back to the host to decide what to fetch, and that
    read drains the GPU queue. A layer held at C == E can never miss, so it skips the round-trip
    entirely -- which means a slot budget can be spent either on holding more experts (fewer
    misses) or on making some layers sync-free (fewer stalls), and which of those wins depends on
    numbers that differ per model.

    `plan_capacity` has modelled that trade since the beginning and has shipped OFF the whole
    time, because the curve it was given was measured on Qwen3-30B and does not transfer:

        OLMoE at a 90% budget    predicted 4.87x    measured 1.19x
        Qwen3 at a 60% budget    predicted 1.06x    measured 0.97x

    The mechanism was never in doubt. The arithmetic was being fed another model's constants.

HOW IT IS MEASURED WITHOUT NEEDING THE MODEL TO FIT
    The obvious way -- hold every layer at C == E and take layers away -- needs the whole model
    resident, which is exactly what the models that most need this planner cannot do. So the
    round-trip is switched off per layer instead of engineered away: `_sync_free` is forced on,
    the layer reads slots straight off the device, and the answer for that layer is wrong.

    THE OUTPUT OF A CALIBRATION RUN IS DELIBERATELY WRONG AND IS NEVER SHOWN TO ANYONE.
    It is the same trick `bypass` uses, for the same reason: the only quantity being measured is
    time, and time does not care whether the expert was the right one. It does mean calibration
    must never run against a user's request -- `rig calibrate` is its own command.
"""
from __future__ import annotations


import json
import os
import time

from . import home

CURVE_DIR = os.path.join(home(), "data", "results")


def curve_path(model_name: str) -> str:
    safe = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in model_name)
    return os.path.join(CURVE_DIR, f"sync_{safe}.json")


def load_curve(model_name: str) -> dict | None:
    """This model's measured curve, or None. Never another model's -- that was the bug."""
    p = curve_path(model_name)
    if not os.path.exists(p):
        return None
    try:
        with open(p) as fh:
            d = json.load(fh)
    except (OSError, ValueError):
        return None
    if not isinstance(d, dict) or not d.get("ms_by_syncing_layers"):
        return None
    return d


def _median(xs):
    xs = sorted(xs)
    return xs[len(xs) // 2] if len(xs) % 2 else (xs[len(xs) // 2 - 1] + xs[len(xs) // 2]) / 2


def usable(curve: dict) -> tuple:
    """(is this curve fit to plan with, why not).

    A CURVE THAT DISAGREES WITH ITSELF MUST NOT REACH A PLANNER.
        Timing a decode step on a small model is noisy enough that repeated runs came out
        non-monotonic -- 11 syncing layers timing above 16, and on one run all 16 timing FASTER
        than none, which implies a negative cost for the round-trip. A planner handed that will
        confidently choose a configuration nobody has run. More syncing layers cannot be cheaper
        than fewer, so a curve that says otherwise is measuring noise and says so instead.
    """
    try:
        pts = sorted((int(k), float(v)) for k, v in curve["ms_by_syncing_layers"].items())
    except (KeyError, TypeError, ValueError):
        return False, "no curve"
    if len(pts) < 3:
        return False, "too few points"
    if curve.get("ms_per_sync_layer", 0) <= 0:
        return False, ("the round-trip measured as free or negative, which it is not -- "
                       "the timing is dominated by noise on this model")
    drops = [(a, ya, b, yb) for (a, ya), (b, yb) in zip(pts, pts[1:]) if yb < ya * 0.97]
    if drops:
        a, ya, b, yb = drops[0]
        return False, (f"{b} syncing layers timed faster than {a} ({yb:.1f} vs {ya:.1f} ms), "
                       f"which cannot be true -- the timing is dominated by noise")
    return True, ""


def measure(session, tokens: int = 24, points: int = 5, prompt: str = "Count to twenty.",
            verbose: bool = True, repeats: int = 3) -> dict:
    """Time one token against how many layers pay the host round-trip.

    Returns the curve, and `ms_per_miss` taken from the fetch the run actually did.
    """
    handle = getattr(session, "handle", None)
    if handle is None or not getattr(handle, "mods", None):
        raise ValueError(f"{session.name} is not streaming, so it pays no round-trips to measure")
    mods = handle.mods
    n_layers = len(mods)
    original = [m._sync_free for m in mods]

    ns = sorted({0, n_layers} | {round(n_layers * i / (points - 1)) for i in range(points)})
    ns = [n for n in ns if 0 <= n <= n_layers]
    msgs = [{"role": "user", "content": prompt}]
    out: dict = {}
    try:
        # One untimed pass so page cache, Metal kernels and the tokenizer are all warm; otherwise
        # the first point on the curve carries every one-off cost in the process.
        for _p, _i in session.stream_text(msgs, max_tokens=4, think=False):
            pass
        for n in ns:
            for i, m in enumerate(mods):
                m._sync_free = i >= n           # the first n layers keep the round-trip
            # WARM UNDER THIS EXACT CONFIGURATION BEFORE TIMING IT.
            #     A layer forced sync-free does not fetch, so it leaves the pool holding whatever
            #     the previous point left there. Timing straight after the switch charged one
            #     point for every fetch the point before it had skipped: the all-syncing point
            #     came out at 95.3 ms/token when running the same model at the same capacity
            #     normally measures 51.4. Every point is warmed in its own configuration, so each
            #     one is measuring itself.
            for _p, _i in session.stream_text(msgs, max_tokens=max(8, tokens // 2), think=False):
                pass
            handle.reset_stats()
            runs = []
            for _rep in range(max(1, repeats)):
            # DECODE ONLY. The round-trip is paid once per layer per STEP, so the quantity being
            # fitted is the cost of a decode step. Starting the clock before prefill folded a
            # fixed cost into a per-token number and divided it by however many tokens the run
            # happened to produce: the all-syncing point read 94.2 ms/token over 16 tokens where
            # the same configuration measures 51.4 over 40. The clock starts at the first token.
                # THE RATE COMES FROM THE ENGINE, NOT FROM COUNTING YIELDS. `stream_text`
                # holds text until a boundary and can deliver a whole reply in ONE yield, which
                # made `produced - first` zero and this line report 80,808 tok/s. `info["tok_s"]`
                # is already decode-only and is the number the product reports.
                rate = 0.0
                for _p, info in session.stream_text(msgs, max_tokens=tokens, think=False):
                    rate = info.get("tok_s") or rate
                runs.append((1000.0 / rate) if 0.01 < rate < 2000.0 else 0.0)
            # The median of several runs, because one run of a decode step on a small model is
            # noise with a number attached.
            runs = [r for r in runs if r > 0]
            ms = _median(runs) if runs else 0.0
            out[n] = round(ms, 3)
            if verbose:
                spread = (max(runs) - min(runs)) / max(ms, 1e-9)
                print(f"    {n:>3} of {n_layers} layers syncing: {ms:8.2f} ms/token"
                      f"   (spread {spread:5.0%} over {len(runs)} runs)", flush=True)
    finally:
        for m, was in zip(mods, original):
            m._sync_free = was

    st = handle.stats()
    misses = st.get("misses") or 0
    ms_per_miss = (st["fetch_seconds"] * 1000.0 / misses) if misses else None
    floor = out.get(0, min(out.values()))
    return {"model": session.name, "n_layers": n_layers,
            "capacity": handle.stats().get("capacity"),
            "n_experts": handle.stats().get("n_experts"),
            "floor_ms": floor,
            "ms_by_syncing_layers": {str(k): v for k, v in sorted(out.items())},
            "ms_per_sync_layer": round((out[n_layers] - floor) / max(1, n_layers), 4),
            "ms_per_miss": round(ms_per_miss, 4) if ms_per_miss else None,
            "tokens_per_point": tokens, "measured_at": int(time.time())}


def observe_miss(session, tokens: int = 24, prompt: str = "Count to twenty.") -> tuple:
    """(residency, miss rate) for the pool as it is currently loaded.

    MEASURING THE ROUND-TRIP WAS NOT ENOUGH, AND THIS IS THE HALF THAT WAS MISSING.
        With a per-model sync curve the planner still predicted 1.15x on OLMoE and delivered
        0.88x -- it made the model SLOWER. The sync half was measured; the other half, how fast
        the miss rate rises as residency falls, was still coming from a curve fitted to somebody
        else's traces. Measured on OLMoE: 32 of 64 experts missed 24.8% of the time and 21 of 64
        missed 45.2%, far steeper than the shared curve assumed, and steep enough to eat the
        whole saving. Both halves are model-specific or neither is worth having.
    """
    h = session.handle
    msgs = [{"role": "user", "content": prompt}]
    # WARM FIRST, THEN MEASURE. A pool starts empty, so every expert's first use is a miss
    # whatever the residency -- and a large pool has MORE slots to fill, so an unwarmed run makes
    # a roomy configuration look worse than a cramped one. Measured that way on OLMoE the curve
    # came out backwards: 10.6% at 50% residency against 27.5% at 75% and 39.7% at 91%, which
    # cannot be true of a cache. Filling the pool first and resetting the counters afterwards is
    # the difference between measuring steady state and measuring start-up.
    for _p, _i in session.stream_text(msgs, max_tokens=max(tokens, 24), think=False):
        pass
    h.reset_stats()
    for _p, _i in session.stream_text(msgs, max_tokens=tokens, think=False):
        pass
    st = h.stats()
    return st["capacity"] / st["n_experts"], st["miss_rate"]


def as_miss_fn(curve: dict):
    """f(residency) -> miss rate, from this model's own points. None if there are too few.

    Held flat outside the measured range rather than extrapolated: a straight line drawn past
    the last real point is how a planner talks itself into a configuration nobody has run.
    """
    pts = sorted((float(r), float(m)) for r, m in (curve or {}).get("miss_by_residency", {}).items())
    if len(pts) < 2:
        return None

    def f(r):
        if r >= 1.0:
            return 0.0
        if r <= pts[0][0]:
            return pts[0][1]
        if r >= pts[-1][0]:
            return pts[-1][1]
        for (a, ya), (b, yb) in zip(pts, pts[1:]):
            if a <= r <= b:
                return ya + (r - a) / (b - a) * (yb - ya)
        return pts[-1][1]
    return f


def save(curve: dict) -> str:
    os.makedirs(CURVE_DIR, exist_ok=True)
    p = curve_path(curve["model"])
    with open(p, "w") as fh:
        json.dump(curve, fh, indent=1)
    return p


def as_cost_fn(curve: dict):
    """(f(n_syncing) -> ms above the floor, ms_per_miss), or None if the curve is unusable.

    Interpolated between measured points and held flat outside them, because extrapolating a
    curve is how the old planner predicted 4.87x and measured 1.19x.
    """
    if not curve:
        return None
    # Without the miss half this planner has been shown to make a model slower. Refusing here is
    # what keeps that from shipping again.
    if not as_miss_fn(curve):
        return None
    try:
        raw = sorted((int(k), float(v)) for k, v in curve["ms_by_syncing_layers"].items())
    except (KeyError, TypeError, ValueError):
        return None
    # The floor is the cost with nothing syncing. Derived when it is not recorded, so a curve is
    # never discarded over a field that can be read straight off the points it already has.
    floor = float(curve.get("floor_ms", raw[0][1]))
    pts = [(n, v - floor) for n, v in raw]
    if len(pts) < 2:
        return None

    def f(n, _n_layers=None):
        if n <= pts[0][0]:
            return pts[0][1]
        if n >= pts[-1][0]:
            return pts[-1][1]
        for (a, ya), (b, yb) in zip(pts, pts[1:]):
            if a <= n <= b:
                return ya + (n - a) / (b - a) * (yb - ya)
        return pts[-1][1]
    return f, curve.get("ms_per_miss"), as_miss_fn(curve)
