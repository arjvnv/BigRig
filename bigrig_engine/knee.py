"""Find the point where holding more experts stops paying for itself.

WHAT THE KNEE IS, AND WHY A CEILING IS NOT ONE
    Measured on Qwen3-30B-A3B-3bit against a 9 GB budget, warmed at every point:

        experts   resident   tok/s    vs 12   miss    longest reply
             12    1.19 GB    3.84    1.00x  53.8%          40,960
             20    1.98 GB    4.33    1.13x  41.9%          34,012
             28    2.77 GB    4.83    1.26x  32.9%          25,948
             36    3.57 GB    5.34    1.39x  25.7%          17,884
             44    4.36 GB    6.24    1.63x  18.1%           9,820
             53    5.25 GB    6.50    1.69x  12.1%             748

    Memory is worth 1.69x, which is more than any change to the engine has been worth. But the
    last row is a trap: 36 -> 44 costs 0.79 GB and buys 0.90 tok/s, while 44 -> 53 costs 0.89 GB
    and buys 0.26 tok/s -- and takes the longest reply from 9,820 tokens to 748. Filling memory
    to the ceiling produces a fast model that cannot finish a paragraph.

    So the knee is the SMALLEST capacity whose speed is within `tolerance` of the best that
    fits. Everything above it is memory bought at a terrible exchange rate, and memory not taken
    is memory the rest of the machine can use.

WHY THIS IS NOT A SWEEP, AND WHY IT IS NOT PURE PREDICTION EITHER
    Timing six capacities took seven minutes. Nobody should wait seven minutes to start chatting.

    The shape is predictable, because a decode step is fixed work plus one read per missed
    expert: `ms = base + misses_per_token * ms_per_miss`. Both terms are already measured by the
    engine -- `ms_per_miss` falls straight out of the fetcher's own accounting -- so one timed
    run plus a few CHEAP miss probes fits the whole curve. Checked against the six points above,
    a two-point fit predicted five of them within 0.7%.

    It predicted the sixth within 5.3%, AND THAT IS THE WHOLE REASON THIS MODULE ALSO MEASURES.
    A 5% prediction error cannot resolve a knee defined by a 5% threshold: at that tolerance the
    predicted knee was 53 and the measured one was 44. Prediction narrows six candidates to two;
    timing those two decides. Every timed point is a median of repeats, because the ground truth
    above is single runs and one of its points is probably noise.
"""
from __future__ import annotations

import json
import os
import sys
import time

from . import home

KNEE_DIR = os.path.join(home(), "data", "results")

# How much speed may be given up to give memory back. 10% was chosen against the measured curve:
# at 10% the predicted and measured knees agree (44), at 5% they do not. It is a default, not a
# law -- a caller that wants the fastest possible configuration passes 0.
DEFAULT_TOLERANCE = 0.10

# WHAT IS ASKED DECIDES WHAT IS MEASURED, SO IT MUST NOT BE ONE REPETITIVE PROMPT.
#     Probing with "Count to twenty." measured a 19.1% miss rate at 36 experts. The same
#     capacity measured 25.7% against ordinary varied text, because counting routes to a narrow
#     set of experts over and over and almost never has to fetch. A knee fitted to that
#     under-values capacity and hands the user a pool that is too small for anything they
#     actually type. These span several domains on purpose.
PROBE_PROMPTS = [
    "Explain in about eighty words why mixture-of-experts models are cheaper to run than dense "
    "ones.",
    "Write a Python function that merges two sorted lists, and say what it costs.",
    "What were the main causes of the 1973 oil crisis?",
    "Translate into French: the weather tomorrow will be cold and clear.",
    "Summarise the difference between a mutex and a semaphore.",
    "A recipe calls for 250 g of flour for 12 biscuits. How much for 30?",
]

# Below this a pool cannot serve a step at all: a layer routing to top_k experts needs at least
# top_k slots, and _chunks would otherwise split every single step.
MIN_SLOTS_OVER_TOPK = 1


def knee_path(model_name: str, budget_gb: float | None = None) -> str:
    """One file per model AND budget. A knee is a fact about a (model, ceiling) pair, and a
    single file per model meant measuring at 9.0 GB silently threw away the 9.7 GB answer --
    the next run at 9.7 then spent two minutes measuring what it already knew."""
    safe = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in model_name)
    if budget_gb is None:
        return os.path.join(KNEE_DIR, f"knee_{safe}.json")
    return os.path.join(KNEE_DIR, f"knee_{safe}@{float(budget_gb):.1f}.json")


def _read(p: str) -> dict | None:
    if not os.path.exists(p):
        return None
    try:
        with open(p) as fh:
            d = json.load(fh)
    except (OSError, ValueError):
        return None
    if not isinstance(d, dict) or not d.get("capacity"):
        return None
    return d


def load(model_name: str, budget_gb: float | None = None) -> dict | None:
    """This model's knee at this budget, or None.

    A knee is only valid for the budget it was measured against -- the same model on a machine
    with half the memory has a different answer, and silently reusing one would be the same class
    of bug as reusing another model's sync curve. The per-budget file is tried first, then the
    older one-per-model file, which is still held to the same budget check.
    """
    import glob as _glob
    if budget_gb is not None:
        candidates = [knee_path(model_name, budget_gb), knee_path(model_name)]
    else:
        safe = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in model_name)
        candidates = [knee_path(model_name)] + sorted(
            _glob.glob(os.path.join(KNEE_DIR, f"knee_{safe}@*.json")),
            key=os.path.getmtime, reverse=True)
    for p in candidates:
        d = _read(p)
        if d is None:
            continue
        if budget_gb is not None and abs(float(d.get("budget_gb", -1)) - float(budget_gb)) > 0.05:
            continue
        return d
    return None


def save(result: dict) -> str:
    os.makedirs(KNEE_DIR, exist_ok=True)
    p = knee_path(result["model"], result.get("budget_gb"))
    with open(p, "w") as fh:
        json.dump(result, fh, indent=1)
    return p


def _median(xs):
    xs = sorted(xs)
    if not xs:
        return 0.0
    return xs[len(xs) // 2] if len(xs) % 2 else (xs[len(xs) // 2 - 1] + xs[len(xs) // 2]) / 2


def fit(anchor_ms: float, anchor_misses: float, ms_per_miss: float) -> float:
    """The fixed part of a decode step: everything that is not waiting for a missed expert.

    Negative would mean the reads cost more than the whole step, which cannot be true and means
    the anchor timing and the fetch accounting disagree. Clamped rather than propagated, so a
    caller never plans against a model that says a token takes less than no time.
    """
    return max(1.0, anchor_ms - anchor_misses * ms_per_miss)


def predict_ms(base_ms: float, miss_rate: float, top_k: int, n_layers: int,
               ms_per_miss: float) -> float:
    return base_ms + miss_rate * top_k * n_layers * ms_per_miss


def miss_at(curve: list, capacity: int) -> float:
    """Miss rate at a capacity, interpolated between probes and held flat outside them.

    Held rather than extrapolated for the reason `synccal` holds its curve: a straight line drawn
    past the last real point is how a planner talks itself into a configuration nobody has run.
    """
    pts = sorted((int(c), float(m)) for c, m in curve)
    if not pts:
        return 0.0
    if capacity <= pts[0][0]:
        return pts[0][1]
    if capacity >= pts[-1][0]:
        return pts[-1][1]
    for (a, ya), (b, yb) in zip(pts, pts[1:]):
        if a <= capacity <= b:
            return ya + (capacity - a) / (b - a) * (yb - ya)
    return pts[-1][1]


def shortlist(candidates: dict, tolerance: float = DEFAULT_TOLERANCE, keep: int = 2) -> list:
    """Capacities worth timing: the predicted knee and its neighbours.

    `candidates` maps capacity -> predicted tok/s. Returns the smallest capacity within tolerance
    of the predicted best, plus the next one up, because the prediction's own error is the same
    size as the tolerance and the true knee sits inside that band.
    """
    if not candidates:
        return []
    order = sorted(candidates)
    best = max(candidates.values())
    within = [c for c in order if candidates[c] >= best * (1 - tolerance)]
    if not within:
        return [order[-1]]
    i = order.index(within[0])
    out = order[max(0, i - 1):i + keep]
    return sorted(set(out))[:keep + 1]


def choose(measured: dict, tolerance: float = DEFAULT_TOLERANCE) -> tuple:
    """(knee, why) from TIMED points only. Never from predictions.

    The smallest capacity whose measured speed is within tolerance of the best measured. Ties go
    to the smaller capacity, which is the whole point: give back memory that buys nothing.
    """
    if not measured:
        return 0, "nothing was measured"
    # MORE EXPERTS CANNOT BE SLOWER. Same arithmetic, same kernels, strictly fewer reads from
    # disk. So a measured dip is noise -- and the noise here is the same size as the tolerance:
    # one run read 47 -> 7.96, 49 -> 7.47, 51 -> 8.10 tok/s, which is not a shape a cache can
    # have. A running maximum removes exactly that and nothing else; it can only ever raise a
    # point to a value already measured at a smaller capacity, never invent one.
    smoothed, run_max = {}, 0.0
    for c in sorted(measured):
        run_max = max(run_max, measured[c])
        smoothed[c] = run_max
    measured = smoothed
    best_c = max(measured, key=lambda c: measured[c])
    best = measured[best_c]
    within = sorted(c for c in measured if measured[c] >= best * (1 - tolerance))
    knee = within[0]
    if knee == best_c:
        return knee, (f"{knee} experts was also the fastest measured "
                      f"({best:.2f} tok/s)")
    return knee, (f"{knee} experts runs at {measured[knee]:.2f} tok/s, within "
                  f"{tolerance:.0%} of the best measured ({best:.2f} at {best_c}), "
                  f"and holds {best_c - knee} fewer experts per layer")


def probe_miss(session, tokens: int = 24, prompts=None) -> tuple:
    """(miss rate, ms per miss) for the pool as it is currently loaded. No timing needed.

    WARM FIRST, THEN RESET, THEN MEASURE. A pool starts empty, so every expert's first use is a
    miss whatever the residency -- and a LARGER pool has more slots to fill, so an unwarmed probe
    makes a roomy configuration look worse than a cramped one. Measured that way on OLMoE the
    curve came out backwards. This is the same discipline `synccal.observe_miss` uses and for the
    same reason.
    """
    h = session.handle
    if h is None:
        return 0.0, 0.0
    ps = list(prompts or PROBE_PROMPTS)
    for p in ps:                                   # warm across the whole spread, not one prompt
        for _c, _i in session.stream_text([{"role": "user", "content": p}],
                                          max_tokens=tokens, think=False):
            pass
    h.reset_stats()
    for p in ps:
        for _c, _i in session.stream_text([{"role": "user", "content": p}],
                                          max_tokens=tokens, think=False):
            pass
    st = h.stats()
    misses = st.get("misses") or 0
    # A MISS COSTS THE UPLOAD, NOT THE READ. With expert reads served from the page cache the
    # read is a view -- 1.7 ms a token -- and the copy of those bytes into a GPU slot is 22.5 ms.
    # Priced from fetch time alone, the knee believed misses were free, took the smallest
    # streamed capacity it could and spent the slots on whole layers: 4 whole + 13 of 256 against
    # a measured 1 whole + 31 that was 8% faster on real replies. The fit sees the true cost now.
    per = ((st["fetch_seconds"] + st.get("admit_seconds", 0.0)) * 1000.0 / misses) if misses else 0.0
    return st["miss_rate"], per


def time_at(session, tokens: int = 40, repeats: int = 3, prompts=None) -> float:
    """Median ms per decode step at the current capacity, or 0.0 if it could not be measured.

    THE RATE COMES FROM THE ENGINE, NOT FROM COUNTING YIELDS.
        The obvious timer starts a clock at the first streamed chunk and divides by how many
        chunks followed. `stream_text` does not stream one chunk per token -- it holds text until
        a boundary, and on this model a 32-token reply arrives as a SINGLE yield. A timer written
        that way divides by `max(1, produced - first)` == 1 and reports 80,808 tok/s, which is
        what it did, and the fit then clamped to a 1 ms token and chose a capacity from noise.

        `info["tok_s"]` is the engine's own decode-only rate -- the number the product reports
        and therefore the right one to optimise against. Wall-clock across the whole call is not
        a substitute: it includes prefill, which on a streamed model is seconds and would be
        divided by however many tokens the run happened to produce.
    """
    ps = list(prompts or PROBE_PROMPTS)
    for p in ps[:2]:
        for _c, _i in session.stream_text([{"role": "user", "content": p}],
                                          max_tokens=max(8, tokens // 2), think=False):
            pass
    runs = []
    for r in range(max(1, repeats)):
        rate = 0.0
        for _c, info in session.stream_text(
                [{"role": "user", "content": ps[r % len(ps)]}],
                max_tokens=tokens, think=False):
            rate = info.get("tok_s") or rate
        # NOTHING IMPLAUSIBLE REACHES A PLANNER. A streamed MoE on this hardware runs in single
        # or low double digits; anything past 2,000 tok/s is a broken measurement, and a broken
        # measurement that is merely improbable rather than impossible is the one that gets used.
        if 0.01 < rate < 2000.0:
            runs.append(1000.0 / rate)
    return _median(runs) if runs else 0.0


class Progress:
    """One line that keeps saying where the tune is, and how much longer.

    WHY THIS EXISTS
        The first run of a streamed model measures its own best capacity, and that takes minutes:
        it builds a pool at half a dozen capacities and generates through each. Before this it
        printed a line per finished step and nothing in between, so the longest stretch a new
        user ever sees -- four minutes on Qwen3.6 -- looked like a hang. The opening line even
        said "about a minute or two", which was wrong and made the wait worse.

    WHAT IT DOES NOT DO
        It does not write carriage returns into a log. A tune run under `nohup`, in CI, or piped
        to a file gets one plain line per step; only a real terminal gets the line that redraws.
        And it never invents a number: the estimate appears only once a step has actually been
        timed, and it is described as an estimate.
    """

    BAR = 12

    def __init__(self, total: int, verbose: bool = True, stream=None):
        self.total = max(1, int(total))
        self.done = 0
        self.t0 = time.perf_counter()
        # Per-step durations. THE FIRST STEP IS NOT LIKE THE OTHERS: it carries the model's
        # first load and fills the page cache, and on Qwen3-30B it took 1m49s against 25-28s for
        # every step after it. An estimate that averaged it in announced "about 9m06s left" on a
        # run that finished in 4m02s -- a number that wrong is worse than no number, so the
        # estimate is built from the steps after the first and does not appear until there are
        # two of them.
        self._marks: list = []
        self.stream = stream if stream is not None else sys.stdout
        self.verbose = bool(verbose)
        try:
            self.tty = self.verbose and self.stream.isatty()
        except Exception:                       # noqa: BLE001 -- a stream without isatty
            self.tty = False
        self._drawn = 0                          # characters of the live line, to erase it

    @staticmethod
    def _clock(sec: float) -> str:
        sec = int(max(0, sec))
        return f"{sec // 60}m{sec % 60:02d}s" if sec >= 60 else f"{sec}s"

    def _erase(self) -> None:
        if self._drawn:
            self.stream.write("\r" + " " * self._drawn + "\r")
            self._drawn = 0

    def line(self, text: str) -> None:
        """A permanent line. The live line is erased first and redrawn after, so the two never
        overwrite each other -- which is exactly what the first version of this did."""
        if not self.verbose:
            return
        self._erase()
        self.stream.write(text + "\n")
        self.stream.flush()
        self._draw()

    def _typical(self) -> float:
        """Seconds a step takes, from the ones that are representative. 0.0 if not enough yet."""
        if len(self._marks) < 2:
            return 0.0
        rest = self._marks[1:]
        return sum(rest) / len(rest)

    def _draw(self) -> None:
        if not self.tty or not getattr(self, "_label", ""):
            return
        el = time.perf_counter() - self.t0
        bar = "#" * int(self.BAR * self.done / self.total)
        left = ""
        per = self._typical()
        if per:
            left = f" · about {self._clock(per * (self.total - self.done))} left"
        msg = (f"  measuring [{bar:{'.'}<{self.BAR}}] {self.done}/{self.total} · {self._label}"
               f" · {self._clock(el)} elapsed{left}")
        self.stream.write("\r" + msg)
        self.stream.flush()
        self._drawn = len(msg)

    def step(self, label: str) -> None:
        """Announce the step that is ABOUT to run, before the silence it causes."""
        self._label = label
        self._step_t0 = time.perf_counter()
        if not self.verbose:
            return
        if self.tty:
            self._draw()
        else:
            self._erase()
            self.stream.write(f"    [{self.done + 1}/{self.total}] {label} "
                              f"({self._clock(time.perf_counter() - self.t0)} elapsed)\n")
            self.stream.flush()

    def done_step(self) -> None:
        now = time.perf_counter()
        self._marks.append(now - (self._step_t0 if getattr(self, "_step_t0", None) else self.t0))
        self.done += 1
        self._draw()

    def retotal(self, total: int) -> None:
        """The number of steps is only known exactly once the curve says how many capacities are
        worth timing. Better to correct it than to show a total that was a guess."""
        self.total = max(self.done, int(total), 1)
        self._draw()

    def close(self) -> None:
        self._erase()
        if self.verbose:
            self.stream.flush()


def measure(make_session, model_name: str, budget_gb: float, n_experts: int, top_k: int,
            n_layers: int, gb_per_slot: float, fits: int, probes: int = 3,
            tolerance: float = DEFAULT_TOLERANCE, verbose: bool = True) -> dict:
    """Find this model's knee on this machine.

    `make_session(capacity)` must return a fresh Session at that capacity AFTER releasing any
    previous one, and must build it the way the model will actually be SERVED -- page cache and
    all. Bypassing the cache here prices the disk honestly and chooses the wrong capacity:
    measured cold, 53 experts beats 43; measured as served, 43 wins 11.09 against 9.24, because
    the larger pool takes anonymous memory away from the page cache that is doing the caching.
    The capacity being chosen is the one the model will run at. Two pools alive at once is the one thing a machine chosen for not having enough
    memory cannot survive, so this function holds exactly one session at a time and closes it
    before asking for the next.

    THE ORDER MATTERS AND IT IS CHEAP-FIRST.
        Miss probes need no timing and settle in a couple of dozen tokens, so the curve is built
        from those. Only then is anything timed, and only at the two or three capacities the
        curve says are worth timing.
    """
    lo = max(top_k + MIN_SLOTS_OVER_TOPK, int(n_experts * 0.15))
    hi = max(lo + 1, min(fits, n_experts))
    if hi <= lo:
        return {"model": model_name, "capacity": hi, "budget_gb": round(budget_gb, 2),
                "why": f"only {hi} experts fit, so there is nothing to choose between",
                "measured": {}, "curve": [], "measured_at": int(time.time())}

    want = sorted({lo, hi} | {int(lo + (hi - lo) * i / max(1, probes - 1))
                              for i in range(probes)})
    curve, ms_per_miss = [], 0.0
    # len(want) miss probes, one timed anchor, then however many candidates the curve picks --
    # two, usually. `retotal` corrects it the moment that is known rather than guessing on.
    prog = Progress(len(want) + 1 + 2, verbose=verbose)
    prog.line(f"  probing miss rate at {len(want)} capacities (cheap -- no timing yet)")
    for c in want:
        prog.step(f"building a pool at {c} of {n_experts}")
        s = make_session(c)
        try:
            mr, per = probe_miss(s)
            curve.append((c, mr))
            if per > 0:
                ms_per_miss = max(ms_per_miss, per)
            prog.line(f"    {c:>4} of {n_experts}: {mr:6.1%} of expert lookups miss")
        finally:
            s.close()
            prog.done_step()

    anchor_c = want[len(want) // 2]
    prog.step(f"timing {anchor_c} of {n_experts}")
    s = make_session(anchor_c)
    try:
        anchor_ms = time_at(s)
    finally:
        s.close()
        prog.done_step()
    anchor_miss = miss_at(curve, anchor_c)
    prog.line(f"  timing one anchor at {anchor_c}: {anchor_ms:.1f} ms/token "
              f"({1000.0 / max(anchor_ms, 1e-9):.2f} tok/s)")

    base = fit(anchor_ms, anchor_miss * top_k * n_layers, ms_per_miss)
    grid = list(range(lo, hi + 1, max(1, (hi - lo) // 12)))
    if hi not in grid:
        grid.append(hi)
    pred = {c: 1000.0 / predict_ms(base, miss_at(curve, c), top_k, n_layers, ms_per_miss)
            for c in grid}
    cands = shortlist(pred, tolerance)
    prog.retotal(len(want) + 1 + len(cands))
    prog.line(f"  fitted {base:.1f} ms of fixed work + {ms_per_miss:.3f} ms per expert read; "
              f"timing {len(cands)} candidate(s): {cands}")

    measured = {}
    for i, c in enumerate(cands):
        prog.step(f"timing {c} of {n_experts} ({i + 1} of {len(cands)} candidates)")
        s = make_session(c)
        try:
            ms = time_at(s)
        finally:
            s.close()
            prog.done_step()
        measured[c] = 1000.0 / max(ms, 1e-9)
        prog.line(f"    {c:>4} of {n_experts}: {measured[c]:.2f} tok/s measured")
    if anchor_c not in measured:
        measured[anchor_c] = 1000.0 / max(anchor_ms, 1e-9)

    picked, why = choose(measured, tolerance)
    prog.close()
    return {"model": model_name, "capacity": int(picked), "why": why,
            "budget_gb": round(budget_gb, 2), "tolerance": tolerance,
            "gb_per_slot": round(gb_per_slot, 4), "n_experts": n_experts,
            "top_k": top_k, "n_layers": n_layers,
            "base_ms": round(base, 2), "ms_per_miss": round(ms_per_miss, 4),
            "curve": [[int(c), round(m, 4)] for c, m in curve],
            "predicted": {str(c): round(v, 3) for c, v in sorted(pred.items())},
            "measured": {str(c): round(v, 3) for c, v in sorted(measured.items())},
            "resident_gb": round(picked * gb_per_slot, 2),
            "measured_at": int(time.time())}
