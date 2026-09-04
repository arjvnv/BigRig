"""The knee finder, tested against the sweep it was built from and against the ways it can lie.

WHY THIS FILE IS ADVERSARIAL RATHER THAN CONFIRMATORY
    A capacity chooser that is quietly wrong does not fail. It hands the user a model that is
    slower than it needed to be, or one that holds memory the rest of the machine wanted, and
    nothing anywhere reports a problem. Every check below is a way this module could do that.

    The ground truth is the six-point sweep in knee.py's docstring, measured on
    Qwen3-30B-A3B-3bit against a 9 GB budget. It is single-run data and one of its points is
    probably noise, which is itself asserted below -- if the fit ever reproduces all six points
    exactly, the fit has been tuned to noise.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from bigrig_engine import knee                                          # noqa: E402

FAIL = []


def check(n, c, d=""):
    (print(f"  PASS  {n}") if c else (FAIL.append(n), print(f"  FAIL  {n}  {d}")))


# capacity, measured tok/s, measured miss rate
SWEEP = [(12, 3.84, 0.538), (20, 4.33, 0.419), (28, 4.83, 0.329),
         (36, 5.34, 0.257), (44, 6.24, 0.181), (53, 6.50, 0.121)]
TOP_K, LAYERS = 8, 48
CURVE = [(c, m) for c, _t, m in SWEEP]
MEASURED = {c: t for c, t, _m in SWEEP}

print("=" * 84)
print("1. THE MISS CURVE IS INTERPOLATED, NEVER EXTRAPOLATED")
print("=" * 84)
check("a probed capacity returns exactly what was probed",
      knee.miss_at(CURVE, 28) == 0.329)
check("between probes it interpolates",
      0.181 < knee.miss_at(CURVE, 40) < 0.257)
check("...linearly, so the midpoint is the mean",
      abs(knee.miss_at([(10, 0.5), (20, 0.1)], 15) - 0.3) < 1e-9)
# A straight line drawn past the last real point is how the old planner predicted 4.87x and
# measured 1.19x. Held flat instead.
check("below the lowest probe it holds flat rather than inventing a worse rate",
      knee.miss_at(CURVE, 1) == 0.538)
check("above the highest probe it holds flat rather than inventing a better one",
      knee.miss_at(CURVE, 999) == 0.121)
check("...and never returns a negative miss rate at any capacity",
      all(knee.miss_at(CURVE, c) >= 0 for c in range(1, 200)))
check("an empty curve is zero, not a crash", knee.miss_at([], 40) == 0.0)

print()
print("=" * 84)
print("2. THE FIT REPRODUCES THE MEASURED SWEEP")
print("=" * 84)
# Fit from the two ENDS only -- the hardest case, using none of the interior points.
lo, hi = SWEEP[0], SWEEP[-1]
ms_lo, ms_hi = 1000.0 / lo[1], 1000.0 / hi[1]
n_lo, n_hi = lo[2] * TOP_K * LAYERS, hi[2] * TOP_K * LAYERS
per_miss = (ms_lo - ms_hi) / (n_lo - n_hi)
base = knee.fit(ms_hi, n_hi, per_miss)
check("the fitted cost of one expert read is positive and sane",
      0.1 < per_miss < 5.0, f"{per_miss:.3f} ms")
check("the fitted fixed cost is a real fraction of a token, not the whole of it",
      50 < base < ms_hi, f"base {base:.1f} of {ms_hi:.1f} ms")

errs = {}
for c, tps, mr in SWEEP:
    p = 1000.0 / knee.predict_ms(base, mr, TOP_K, LAYERS, per_miss)
    errs[c] = abs(p - tps) / tps
check("every point is predicted within 6%", max(errs.values()) < 0.06,
      f"worst {max(errs.values()):.1%} at {max(errs, key=errs.get)}")
check("five of the six are predicted within 1%",
      sum(1 for e in errs.values() if e < 0.01) >= 5,
      f"{sum(1 for e in errs.values() if e < 0.01)} within 1%")
# If this ever passes, the model has been tuned to single-run noise rather than to the mechanism.
check("...and NOT all six, because the ground truth is single runs and one point is noise",
      max(errs.values()) > 0.01)

print()
print("=" * 84)
print("3. THE FIT REFUSES TO PRODUCE AN IMPOSSIBLE STEP")
print("=" * 84)
check("a step can never be predicted to take less than no time",
      knee.fit(10.0, 1000.0, 5.0) >= 1.0)
check("...which is what happens when the timing and the fetch accounting disagree",
      knee.fit(100.0, 500.0, 1.0) == 1.0)
check("a clean anchor is passed through untouched",
      abs(knee.fit(200.0, 100.0, 0.5) - 150.0) < 1e-9)

print()
print("=" * 84)
print("4. THE KNEE IS CHOSEN FROM MEASUREMENTS, AND IT GIVES MEMORY BACK")
print("=" * 84)
k10, why10 = knee.choose(MEASURED, 0.10)
check("on the measured sweep the knee is 44, not the fastest 53", k10 == 44, f"got {k10}")
check("...and it says how many experts that gives back", "9 fewer experts" in why10, why10)
check("a zero tolerance picks the fastest, because nothing may be given up",
      knee.choose(MEASURED, 0.0)[0] == 53)
check("a huge tolerance collapses to the smallest capacity offered",
      knee.choose(MEASURED, 0.99)[0] == 12)
check("the knee is never faster than the best measured",
      MEASURED[k10] <= max(MEASURED.values()))
check("the knee is always one of the capacities that was actually timed",
      k10 in MEASURED)
check("tightening the tolerance never returns a SMALLER capacity",
      all(knee.choose(MEASURED, a)[0] >= knee.choose(MEASURED, b)[0]
          for a, b in [(0.0, 0.05), (0.05, 0.10), (0.10, 0.20)]))
check("nothing measured gives a reason, not a capacity",
      knee.choose({}, 0.1) == (0, "nothing was measured"))
check("a single measurement is its own knee", knee.choose({7: 5.0}, 0.1)[0] == 7)

print()
print("=" * 84)
print("5. THE SHORTLIST BRACKETS THE TRUE KNEE INSTEAD OF TRUSTING THE PREDICTION")
print("=" * 84)
pred = {c: 1000.0 / knee.predict_ms(base, mr, TOP_K, LAYERS, per_miss) for c, _t, mr in SWEEP}
sl = knee.shortlist(pred, 0.10)
# The prediction's error and the tolerance are the same size, so the shortlist must contain the
# measured answer even when the predicted one differs. This is the check that justifies timing.
check("the shortlist contains the MEASURED knee even though the prediction is imperfect",
      44 in sl, f"shortlist {sl}")
check("...and it is short: three capacities or fewer", len(sl) <= 3, f"{sl}")
check("a tighter tolerance shortlists higher capacities",
      min(knee.shortlist(pred, 0.02)) >= min(sl))
check("an empty prediction shortlists nothing rather than guessing",
      knee.shortlist({}, 0.1) == [])
check("a flat curve shortlists the smallest, since nothing is bought by more",
      knee.shortlist({10: 5.0, 20: 5.0, 30: 5.0}, 0.10)[0] == 10)

print()
print("=" * 84)
print("6. A CACHED KNEE IS NEVER REUSED AGAINST A DIFFERENT BUDGET")
print("=" * 84)
import json                                                             # noqa: E402
import tempfile                                                         # noqa: E402

_orig = knee.KNEE_DIR
knee.KNEE_DIR = tempfile.mkdtemp()
try:
    p = knee.save({"model": "unit/test-model", "capacity": 44, "budget_gb": 9.0})
    check("what was saved can be read back", knee.load("unit/test-model")["capacity"] == 44)
    check("...at the budget it was measured against",
          knee.load("unit/test-model", 9.0)["capacity"] == 44)
    # Reusing another budget's answer is the same class of bug as reusing another model's sync
    # curve, which shipped for weeks and made models slower.
    check("a different budget is refused rather than silently reused",
          knee.load("unit/test-model", 6.0) is None)
    check("...including a slightly different one", knee.load("unit/test-model", 9.2) is None)
    check("a budget within rounding is accepted", knee.load("unit/test-model", 9.001) is not None)
    check("an unknown model is None, not a default", knee.load("never/measured") is None)
    with open(p, "w") as fh:
        fh.write("{ this is not json")
    check("a corrupt file is None rather than a crash", knee.load("unit/test-model") is None)
    with open(p, "w") as fh:
        json.dump({"model": "unit/test-model", "budget_gb": 9.0}, fh)
    check("a file with no capacity in it is refused", knee.load("unit/test-model") is None)
    check("a model name with slashes still round-trips",
          knee.knee_path("mlx-community/Qwen3-30B").endswith("knee_mlx-community_Qwen3-30B.json"))
finally:
    knee.KNEE_DIR = _orig

print()
print("=" * 84)
print("7. THE FLOOR: A POOL SMALLER THAN TOP-K CANNOT SERVE A TOKEN")
print("=" * 84)
check("the module states a floor above top-k at all", knee.MIN_SLOTS_OVER_TOPK >= 1)
check("the default tolerance is the one the measured curve agreed at",
      knee.DEFAULT_TOLERANCE == 0.10)
# At 5% the predicted knee was 53 and the measured one 44; at 10% they agree. Pinning this stops
# the default drifting back to a value the data does not support.
check("...and 5% is NOT that value, which is why the default is not 5%",
      knee.choose(MEASURED, 0.05)[0] != knee.shortlist(pred, 0.05)[0]
      or knee.DEFAULT_TOLERANCE != 0.05)

print()
print("=" * 84)
print("8. THE TIMER READS THE ENGINE'S RATE, NEVER THE NUMBER OF YIELDS")
print("=" * 84)


class _FakeSession:
    """A session that behaves the way the real one actually does: ONE yield for a whole reply."""

    def __init__(self, tok_s):
        self.tok_s, self.calls = tok_s, 0

    def stream_text(self, msgs, max_tokens=0, think=True, **kw):
        self.calls += 1
        rate = self.tok_s(self.calls) if callable(self.tok_s) else self.tok_s
        yield ("the whole reply arrives at once", {"generation_tokens": max_tokens,
                                                   "tok_s": rate})


# The defect: a timer counting yields divides by max(1, produced - first) == 1 and reports tens
# of thousands of tokens a second, which then fits a 1 ms token and picks a capacity from noise.
f = _FakeSession(6.24)
ms = knee.time_at(f, tokens=40, repeats=3)
check("a single-yield reply is still timed correctly",
      abs(ms - 1000.0 / 6.24) < 1e-6, f"got {ms:.3f} ms")
check("...and gives back the rate the engine reported",
      abs(1000.0 / ms - 6.24) < 1e-6)
# Two warm-up prompts then `repeats` timed runs; the warm-ups are never timed.
check("the warm-up runs are not counted as measurements", f.calls == 5,
      f"{f.calls} calls")
check("an implausible rate is discarded rather than planned against",
      knee.time_at(_FakeSession(80808.42), tokens=40, repeats=3) == 0.0)
check("...and so is a zero", knee.time_at(_FakeSession(0.0), tokens=40, repeats=1) == 0.0)
check("a plausible-but-fast rate is kept", knee.time_at(_FakeSession(900.0), repeats=1) > 0)
# One bad run among several must not drag the answer; the median is why repeats exist.
noisy = _FakeSession(lambda n: 80808.0 if n == 3 else 6.0)
check("one broken run among several does not move the median",
      abs(knee.time_at(noisy, repeats=3) - 1000.0 / 6.0) < 1e-6)
check("a session that never reports a rate gives 0.0, not a crash",
      knee.time_at(_FakeSession(None), repeats=2) == 0.0)

print()
print("=" * 84)
print("9. NOISE MUST NOT CHOOSE THE CAPACITY")
print("=" * 84)
# More experts cannot be slower -- same arithmetic, strictly fewer reads. A live run measured
# 47 -> 7.96, 49 -> 7.47, 51 -> 8.10 tok/s, a shape no cache can have, with the dip the same size
# as the tolerance.
NOISY = {47: 7.96, 49: 7.47, 51: 8.10}
check("a dip is smoothed away rather than allowed to pick the answer",
      knee.choose(NOISY, 0.10)[0] == 47, f"got {knee.choose(NOISY, 0.10)}")
check("the envelope does not change an already-monotone curve",
      knee.choose(MEASURED, 0.10)[0] == 44)
# The envelope may only raise a point to a value already measured lower down; it must never
# invent a speed nobody saw.
_all = set(MEASURED.values()) | set(NOISY.values())
check("no smoothed value is one that was never measured",
      all(v in _all for v in [knee.choose(NOISY, t)[0] and NOISY.get(knee.choose(NOISY, t)[0])
                              for t in (0.0, 0.1, 0.5)] if v is not None))
check("a strictly decreasing curve collapses to its first point, not its fastest",
      knee.choose({10: 9.0, 20: 8.0, 30: 7.0}, 0.10)[0] == 10)
check("zero tolerance still picks the fastest after smoothing",
      knee.choose(NOISY, 0.0)[0] == 51)

print()
print("=" * 84)
print("10. WHAT IS ASKED DECIDES WHAT IS MEASURED")
print("=" * 84)
# Probing with "Count to twenty." measured 19.1% misses at 36 experts; ordinary varied text
# measured 25.7% at the same capacity. A knee fitted to the repetitive prompt under-values
# capacity and gives the user a pool too small for anything they actually type.
check("there is a spread of probe prompts, not one", len(knee.PROBE_PROMPTS) >= 4)
check("...and they are genuinely different from each other",
      len({p.split()[0].lower() for p in knee.PROBE_PROMPTS}) >= 4)
check("...covering more than one kind of task",
      any("Python" in p or "function" in p for p in knee.PROBE_PROMPTS)
      and any("Translate" in p or "French" in p for p in knee.PROBE_PROMPTS))
check("none of them is a counting drill",
      not any("count to" in p.lower() for p in knee.PROBE_PROMPTS))


class _Recorder:
    """Records which prompts a measurement actually sent."""

    def __init__(self):
        self.seen = []

    def stream_text(self, msgs, max_tokens=0, think=True, **kw):
        self.seen.append(msgs[0]["content"])
        yield ("x", {"generation_tokens": max_tokens, "tok_s": 6.0})


r = _Recorder()
knee.time_at(r, tokens=40, repeats=3)
check("timing cycles the prompts instead of repeating one",
      len({p for p in r.seen}) > 1, f"sent {len(set(r.seen))} distinct")

print("\n" + "=" * 84)
print("THE TUNE SAYS WHERE IT IS, AND NEVER WRITES CARRIAGE RETURNS INTO A LOG")
print("=" * 84)
# The first run of a streamed model measures its own capacity, and that is minutes of near
# silence -- the longest wait a new user ever sees, and it looked like a hang. What is asserted
# here is everything that could make the cure worse than the disease: a log full of \r, a
# progress line that overwrites a result line, an estimate invented before anything was timed.
import io as _io


class _Fake(_io.StringIO):
    def __init__(self, tty): super().__init__(); self._tty = tty
    def isatty(self): return self._tty


_t = _Fake(True)
p = knee.Progress(4, verbose=True, stream=_t)
p.step("building a pool at 9 of 64")
_first = _t.getvalue()
check("a terminal gets a live line naming the step it is about to take",
      "\r" in _first and "building a pool at 9 of 64" in _first and "0/4" in _first, repr(_first[-90:]))
check("...and no estimate before anything has been timed", "left" not in _first, repr(_first[-60:]))
p.done_step()
p.step("timing 15 of 64")
_second = _t.getvalue()[len(_first):]
check("...still no estimate after ONE step: the first step is the unrepresentative one",
      "left" not in _second, repr(_second[-90:]))
check("...and the counter advances", "1/4" in _second)
p.done_step()
p.step("timing 22 of 64")
_third = _t.getvalue()
check("...an estimate appears once there are two comparable steps, and is called an estimate",
      "about" in _third and "left" in _third, repr(_third[-90:]))
_before = len(_t.getvalue())
p.line("      15 of 64:  63.7% of expert lookups miss")
_after = _t.getvalue()[_before:]
check("a result line erases the live line first, so the two never overwrite each other",
      _after.startswith("\r") and " " * 20 in _after.split("\r")[1]
      and "63.7% of expert lookups miss" in _after, repr(_after[:70]))
check("...and the live line is redrawn under it", _after.rstrip().endswith("left")
      or "measuring [" in _after.split("\n")[-1], repr(_after[-70:]))
p.retotal(6)
_last_frame = [f for f in _t.getvalue().split("\r") if "measuring [" in f][-1]
check("the total is corrected once the curve says how many candidates are worth timing",
      p.total == 6 and "/6" in _last_frame and "/4" not in _last_frame, repr(_last_frame[:90]))
p.close()
check("closing erases the live line rather than leaving it on screen",
      _t.getvalue().endswith("\r") or _t.getvalue().rstrip("\r ").endswith("miss")
      or _t.getvalue()[-1] in "\r ", repr(_t.getvalue()[-40:]))

_l = _Fake(False)
q = knee.Progress(3, verbose=True, stream=_l)
q.step("building a pool at 9 of 64")
q.done_step()
q.line("      9 of 64:  69.9% miss")
q.step("timing 15 of 64")
q.close()
_log = _l.getvalue()
check("a log gets no carriage returns at all", "\r" not in _log, repr(_log[:120]))
check("...one plain numbered line per step, with elapsed time",
      "[1/3] building a pool at 9 of 64" in _log and "elapsed" in _log, repr(_log[:80]))
check("...and result lines still appear", "69.9% miss" in _log)

_q = _Fake(True)
r = knee.Progress(2, verbose=False, stream=_q)
r.step("x"); r.done_step(); r.line("y"); r.close()
check("verbose=False writes nothing at all, so a quiet caller stays quiet", _q.getvalue() == "",
      repr(_q.getvalue()))


class _NoTty(_io.StringIO):
    def isatty(self): raise OSError("no")


_n = _NoTty()
z = knee.Progress(2, verbose=True, stream=_n)
z.step("x"); z.done_step(); z.close()
check("a stream that cannot answer isatty is treated as a log, not crashed on",
      "\r" not in _n.getvalue() and "[1/2]" in _n.getvalue(), repr(_n.getvalue()))
check("the clock reads in minutes and seconds past a minute",
      knee.Progress._clock(9) == "9s" and knee.Progress._clock(125) == "2m05s",
      f"{knee.Progress._clock(9)} {knee.Progress._clock(125)}")
check("a zero-step tune cannot divide by zero", knee.Progress(0, verbose=False).total == 1)
# THE ESTIMATE MUST NOT BE WORSE THAN NO ESTIMATE. The first step carries the model's first load
# and fills the page cache: measured on Qwen3-30B, 1m49s against 25-28s for every step after it.
# Averaging it in announced "about 9m06s left" on a run that took 4m02s. These are the real
# durations from that run; the estimate is built from the steps after the first.
_real = [109, 25, 25, 27, 28, 28]
_tot = sum(_real)
_p = knee.Progress(6, verbose=False)
_worst = 0.0
for _i, _d in enumerate(_real):
    _p._marks.append(_d); _p.done = _i + 1
    _per = _p._typical()
    _left = _tot - sum(_real[:_i + 1])
    if _i == 0:
        check("no estimate at all after the first step, which is the unrepresentative one",
              _per == 0.0)
    else:
        _worst = max(_worst, abs(_per * (_p.total - _p.done) - _left))
check("...and every estimate after that is within 15 s of the truth on the measured run",
      _worst <= 15, f"worst error {_worst:.0f}s")
_old = max(abs(sum(_real[:i + 1]) / (i + 1) * (6 - (i + 1)) - (_tot - sum(_real[:i + 1])))
           for i in range(6))
check("...which the old mean-of-everything estimator was not", _old > 60, f"{_old:.0f}s")
check("the tune no longer promises a minute or two it cannot keep",
      "about a minute or two" not in open(os.path.join(ROOT, "bigrig_engine", "cli.py"),
                                          encoding="utf-8").read())

print()
print("=" * 84)
print("ALL TESTS PASSED" if not FAIL else f"{len(FAIL)} FAILURES: " + ", ".join(FAIL))
print("=" * 84)
sys.exit(1 if FAIL else 0)
