"""Adversarial tests for AdaptiveMeter. Self-calibration has failure modes a fixed meter does
not, and each of these targets one of them."""
import os
import sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bigrig_layer.adaptive import AdaptiveMeter

FAIL = []
def check(n, c, d=""):
    (print(f"  PASS  {n}") if c else (FAIL.append(n), print(f"  FAIL  {n}  {d}")))

V = 1000
RNG = np.random.default_rng(0)
def conf(p1):
    p = np.full(V, (1 - p1) / (V - 1)); p[0] = p1; return p
def tok():
    return int(RNG.integers(0, 5000))
def feed(m, dist, n, loop=False):
    for _ in range(n):
        m.observe(dist() if callable(dist) else dist)
        m.observe_token(7 if loop else tok())

print("=" * 78); print("1. IT ADAPTS TO DIFFERENT MODEL PERSONALITIES"); print("=" * 78)
for name, p1 in [("very confident", 0.98), ("middling", 0.6), ("hesitant", 0.3)]:
    m = AdaptiveMeter(window=32, warmup=3)
    feed(m, lambda: conf(p1), 32 * 5)
    check(f"{name} model reads healthy against its OWN baseline",
          m.is_degraded() is False, f"z={m.z_score():.2f}")

print("\n" + "=" * 78); print("2. THE KILLER FAILURE — a long bad patch must not become 'normal'"); print("=" * 78)
m = AdaptiveMeter(window=32, warmup=3, ema=0.02)
feed(m, lambda: conf(0.9), 32 * 4)                 # healthy baseline
feed(m, lambda: np.full(V, 1.0 / V), 32 * 20)      # sustained degradation
check("still flagged after 20 windows of sustained degradation",
      m.is_degraded() is True, f"z={m.z_score():.2f} — baseline was absorbed")

print("\n" + "=" * 78); print("3. VARIANCE FLOOR — a perfectly consistent model must not explode"); print("=" * 78)
m = AdaptiveMeter(window=32, warmup=3)
feed(m, conf(0.9), 32 * 6)                          # identical input every step
z = m.z_score()
check("z stays finite and small on constant input", abs(z) < 10, f"z={z:.1f}")
m.observe(conf(0.899)); m.observe_token(tok())
check("a tiny deviation does not trigger a huge z", abs(m.z_score()) < 50, f"z={m.z_score():.1f}")

print("\n" + "=" * 78); print("4. LOOPING — caught regardless of confidence"); print("=" * 78)
m = AdaptiveMeter(window=32, warmup=3)
feed(m, lambda: conf(0.9), 32 * 4)
feed(m, lambda: conf(0.99), 32 * 2, loop=True)      # confident AND looping
check("confident looping is caught", m.is_degraded() is True and m.reason() == "looping",
      f"rep={m.repetition():.3f} reason={m.reason()}")

print("\n" + "=" * 78); print("5. COLD START"); print("=" * 78)
# THIS SECTION USED TO ASSERT THE METER SPOKE AT min_window, WHICH WAS SIXTEEN TOKENS.
#     That was chosen for responsiveness and it was the wrong trade, measured on the running
#     engine: short replies were the largest single source of false alarms. A sixteen-token reply
#     gives thirteen 4-grams, and a repetition rate over thirteen samples is noise. There is now
#     a second, higher floor -- `min_reply_tokens` -- below which the meter abstains outright.
#
#     `min_window` still does its old job, which is a different one: it governs when a PARTIAL
#     window may be scored at all, and `provisional()` still marks such a reading as noisier.
#     The tests below are written against a meter with the floor lowered to the old value, so
#     that the min_window mechanism itself is still exercised rather than shadowed by the floor.
m = AdaptiveMeter(window=64, min_window=16, warmup=2, min_reply_tokens=16)
feed(m, lambda: conf(0.9), 64 * 3)
m.reset(keep_baseline=True)
feed(m, lambda: np.full(V, 1.0 / V), 8)
check("silent below min_window", m.is_degraded() is None)
feed(m, lambda: np.full(V, 1.0 / V), 8)
check("speaks at min_window, once the reply floor allows it", m.is_degraded() is True)
check("marked provisional on a partial window", m.provisional() is True)
feed(m, lambda: np.full(V, 1.0 / V), 48)
check("no longer provisional once full", m.provisional() is False)
mm = AdaptiveMeter(window=64, min_window=16, warmup=2, min_reply_tokens=16)
feed(mm, lambda: conf(0.9), 64 * 3); mm.reset(keep_baseline=True)
feed(mm, lambda: conf(0.9), 16)
check("a partial window of HEALTHY output is not flagged", mm.is_degraded() is False)
# And the floor itself, at its shipped default, on the same sixteen tokens.
dd = AdaptiveMeter(window=64, min_window=16, warmup=2)
feed(dd, lambda: conf(0.9), 64 * 3); dd.reset(keep_baseline=True)
feed(dd, lambda: np.full(V, 1.0 / V), 16)
check("...but at the SHIPPED floor those same sixteen tokens are declined, not judged",
      dd.is_degraded() is None, repr(dd.is_degraded()))

print("\n" + "=" * 78); print("6. BASELINE LIFECYCLE"); print("=" * 78)
m = AdaptiveMeter(window=32, warmup=3)
feed(m, lambda: conf(0.9), 32 * 4)
n = m.n_windows
m.reset(keep_baseline=True)
check("reset(keep_baseline=True) keeps what it learned", m.n_windows == n and m._mu is not None)
m.reset(keep_baseline=False)
check("reset(keep_baseline=False) forgets", m._mu is None and m.n_windows == 0)
m2 = AdaptiveMeter(window=32, warmup=5)
feed(m2, lambda: conf(0.9), 32 * 2)
check("silent until warmup windows have been seen", m2.is_degraded() is None or m2.n_windows >= 5)

print("\n" + "=" * 78); print("7. HOSTILE INPUT"); print("=" * 78)
for nm, arr in [("NaN", np.where(np.arange(V) == 5, np.nan, conf(0.9))),
                ("inf", np.where(np.arange(V) == 5, np.inf, conf(0.9))),
                ("all zeros", np.zeros(V))]:
    m = AdaptiveMeter(window=8, warmup=2, min_window=4)
    try:
        m.observe(arr); check(f"{nm} raises in strict mode", False)
    except ValueError: check(f"{nm} raises in strict mode", True)
    m2 = AdaptiveMeter(window=8, warmup=2, strict=False, min_window=4)
    for _ in range(8): m2.observe(arr)
    check(f"{nm} skipped in lenient mode", m2.n_skipped == 8)

print("\n" + "=" * 78); print("8. DROP-IN PARITY WITH Meter"); print("=" * 78)
# An engine that switches to the fast path for speed must get the SAME meter, and code
# written against Meter must not break when swapped to AdaptiveMeter.
a, b = AdaptiveMeter(window=8, warmup=2, min_window=4), AdaptiveMeter(window=8, warmup=2, min_window=4)
rg = np.random.default_rng(7)
for _ in range(40):
    p = rg.dirichlet(np.ones(50) * 0.3)
    a.observe(probs=p)
    srt = np.sort(p)[::-1]; nz = p[p > 0]
    b.observe_stats(entropy=float(-(nz * np.log(nz)).sum()),
                    top1=float(srt[0]), margin=float(srt[0] - srt[1]))
check("observe_stats folds the same number of baseline windows as observe",
      a.n_windows == b.n_windows, f"{a.n_windows} vs {b.n_windows}")
za, zb = a.z_score(), b.z_score()
check("observe_stats gives a bit-identical z to observe",
      za is not None and zb is not None and abs(za - zb) < 1e-9, f"{za} vs {zb}")

m = AdaptiveMeter(window=4, strict=True, min_window=4)
try:
    m.observe_stats(entropy=float("nan"), top1=0.5, margin=0.2); ok = False
except ValueError:
    ok = True
check("observe_stats raises on a non-finite statistic when strict", ok)
lax = AdaptiveMeter(window=4, strict=False, min_window=4)
lax.observe_stats(entropy=float("inf"), top1=0.5, margin=0.2)
check("observe_stats skips instead of raising when not strict",
      lax.n_skipped == 1 and lax.n_observed == 0)

from bigrig_layer.meter import QualityMeter as Meter
# `calibrate` is deliberately absent: it is the supervised fitting routine that PRODUCES a
# fixed calibration, and a meter that calibrates itself has nothing to fit. Every method an
# instance is actually called with at runtime must be present.
EXEMPT = {"calibrate"}
missing = [n for n in dir(Meter)
           if not n.startswith("_") and n not in EXEMPT
           and callable(getattr(Meter, n)) and not hasattr(AdaptiveMeter, n)]
check("AdaptiveMeter exposes every public method QualityMeter does", not missing, f"missing: {missing}")

try:
    AdaptiveMeter(window=8, min_window=16); ok = False
except ValueError:
    ok = True
check("a window smaller than min_window is refused, not silently never-ready", ok)

print("\n" + "=" * 78); print("9. REPETITION IS LEARNED, NOT ASSUMED"); print("=" * 78)
# The shipped fixed threshold (0.15) fires on 21.6% of HEALTHY windows of a base model, whose
# normal repetition rate is simply higher than the instruct model it was fitted on. These test
# the learned replacement, including the two ways learning it can go wrong.
W = 64
def rep_toks(rho, base=0):
    """W tokens whose 4-gram repetition rate is rho by construction."""
    P = max(1, min(int(round((1.0 - rho) * (W - 3))), W - 3))
    return [base + (i % P) for i in range(W)]

def drive(m, rho, nwin, base=0):
    for j in range(nwin):
        for x in rep_toks(rho, base + 1000 * (j % 7)):
            m.observe_token(x)
            m.observe_stats(entropy=1.0, top1=0.6, margin=0.4)

m = AdaptiveMeter(window=W, warmup=2, min_window=8)
check("with no history the threshold is the absolute backstop, not a guess",
      m.rep_threshold() == m.rep_ceiling, f"{m.rep_threshold()}")

# a naturally repetitive model must teach the meter that its rate is normal
m = AdaptiveMeter(window=W, warmup=2, min_window=8)
drive(m, 0.25, 40)
check("a naturally repetitive model raises its own threshold above the floor",
      m.rep_threshold() > 0.15, f"threshold {m.rep_threshold():.3f}")
check("and then stops calling its own normal output looping",
      not m._looping(), f"rep {m.repetition():.3f} vs thr {m.rep_threshold():.3f}")

# THE LEARNING LOCK: the gate that keeps a loop out of the baseline must not also keep the
# model's normal rate out of it, or the threshold stays pinned at the floor forever.
m = AdaptiveMeter(window=W, warmup=2, min_window=8)
drive(m, 0.30, 40)
check("the healthy-only gate does not lock the threshold at its starting value",
      m.rep_threshold() > 0.16, f"threshold {m.rep_threshold():.3f} (pinned at the floor)")

# ...but genuine looping must still be caught, on ANY model, including the repetitive one
m = AdaptiveMeter(window=W, warmup=2, min_window=8)
drive(m, 0.25, 40)
before = m.rep_threshold()
drive(m, 0.95, 3)
check("hard looping is still caught on a model with a high learned threshold",
      m._looping() and m.reason() == "looping", f"thr {before:.3f}, rep {m.repetition():.3f}")

# a sustained loop must not be able to normalise itself past the ceiling
m = AdaptiveMeter(window=W, warmup=2, min_window=8)
drive(m, 0.92, 200)
check("a loop running from the very first token cannot push the threshold past the ceiling",
      m.rep_threshold() <= m.rep_ceiling, f"threshold {m.rep_threshold():.3f}")
check("and that loop is still reported as looping",
      m._looping(), f"rep {m.repetition():.3f} vs thr {m.rep_threshold():.3f}")

# one sample per window, not per token -- else the history is redundant and tracks the last
# window instead of the model
m = AdaptiveMeter(window=W, warmup=2, min_window=8)
drive(m, 0.10, 30)
check("the history holds one sample per window, not one per token",
      len(m._rbuf) <= 32, f"{len(m._rbuf)} samples after 30 windows")

# an explicit threshold must pin it, for the case where the rate is already measured
m = AdaptiveMeter(window=W, warmup=2, min_window=8, repetition_threshold=0.42)
drive(m, 0.25, 40)
check("an explicitly supplied threshold overrides learning", m.rep_threshold() == 0.42,
      f"{m.rep_threshold()}")

m = AdaptiveMeter(window=W, warmup=2, min_window=8)
drive(m, 0.25, 40)
m.reset(keep_baseline=False)
check("reset(keep_baseline=False) forgets the learned repetition rate too",
      m.rep_threshold() == m.rep_ceiling, f"{m.rep_threshold()}")

print("\n" + "=" * 78); print("10. FALSE-ALARM RATE ON HEALTHY OUTPUT"); print("=" * 78)
# THE GAP THAT LET A REAL BUG LIVE. Every earlier test asked whether the meter CORRELATES with
# quality; none asked how often it FIRES on output that is fine. The answer was 32% -- which
# would have made the auto-tuner give away nearly all of its speed on undamaged text.
rgz = np.random.default_rng(11)

def healthy_stream(m, n, jitter=0.08):
    """Ordinary output: a stable model with normal token-to-token variation."""
    fired = seen = 0
    for k in range(n):
        p1 = float(np.clip(0.55 + jitter * rgz.standard_normal(), 0.05, 0.98))
        m.observe(conf(p1))
        m.observe_token(int(rgz.integers(0, 5000)))
        d = m.is_degraded()
        if d is not None:
            seen += 1; fired += bool(d)
    return fired, seen

m = AdaptiveMeter(window=64, warmup=2, min_window=16)
healthy_stream(m, 3000)                      # learn this model
f, n = healthy_stream(m, 3000)               # then measure
rate = 100.0 * f / max(n, 1)
check("false-alarm rate on healthy output stays under the 10% design point",
      rate <= 10.0, f"{rate:.1f}% of healthy windows flagged")

# The mechanism behind that bug: the baseline folded on EVERY token, but adjacent windows differ
# by one token in sixty-four, so it measured its own jitter instead of the real spread.
m = AdaptiveMeter(window=64, warmup=2, min_window=16)
healthy_stream(m, 4000)
seen = []
mm = AdaptiveMeter(window=64, warmup=2, min_window=16)
for k in range(4000):
    p1 = float(np.clip(0.55 + 0.08 * rgz.standard_normal(), 0.05, 0.98))
    mm.observe(conf(p1)); mm.observe_token(int(rgz.integers(0, 5000)))
    if len(mm._buf) == mm.window: seen.append(mm._features().copy())
true_sd = np.array(seen).std(axis=0)
ratio = true_sd / mm._sd()
check("internal spread is within 2x of the true spread of healthy windows",
      float(np.max(ratio)) < 2.0, f"true/internal = {np.round(ratio, 2)}")

# the z history must NOT be gated on the meter's own verdict -- that circularity is what
# truncated the spread and inflated every z in the first place
m = AdaptiveMeter(window=32, warmup=2, min_window=8)
healthy_stream(m, 2000)
above = sum(1 for z in m._zbuf if z > m.z_bar())
check("the z history retains readings above the current bar, so it is not self-truncating",
      above > 0, f"history holds {len(m._zbuf)} values, none above the bar {m.z_bar():.2f}")

# and after all that, a genuine collapse must still be caught
m = AdaptiveMeter(window=32, warmup=2, min_window=8)
healthy_stream(m, 2000)
bar = m.z_bar()
caught = 0
for k in range(300):
    m.observe(vaguely := np.full(V, 1.0 / V))
    m.observe_token(int(rgz.integers(0, 5000)))
    if m.is_degraded(): caught += 1
check("a genuine collapse to maximum entropy is still caught after the bar is learned",
      caught > 150, f"flagged {caught}/300 windows, bar={bar:.2f}")

check("z_bar is bounded by the ceiling it was given",
      m.z_bar() <= m.z_ceiling, f"{m.z_bar()} > {m.z_ceiling}")

print("\n" + "=" * 78); print("11. THE SAME QUESTION, ASKED OF REAL OUTPUT"); print("=" * 78)
# Section 10's synthetic stream passes even with the bug reintroduced -- its jitter is too well
# behaved to reproduce what a real model does. Reintroducing the fixed 2.5 threshold leaves every
# synthetic test green while the false-alarm rate on REAL healthy generations is 32%. So the
# real stream is replayed here. This is the same lesson as the looping failure, which 53 passing
# tests missed and one run on real output caught.
import json as _json
_P = os.path.join(ROOT, "data/results/mon_pertoken.json")
if not os.path.exists(_P):
    print(f"  SKIPPED — {_P} absent. This test is the only one that reproduces the conditions")
    print( "  that caused a 32% false-alarm rate; the synthetic tests above do NOT. Regenerate")
    print( "  with src/mon_pertoken.py before trusting a green suite.")
    FAIL.append("real-output false-alarm test could not run")
else:
    _D = _json.load(open(_P))
    _H = [r for r in _D["runs"] if r["sig"] and r["lam"] == 0.0]      # toll OFF: every flag false
    _m = AdaptiveMeter(window=64, warmup=2, min_window=16)
    _fp = _tot = 0
    for _pass in (0, 1):                       # first pass learns the model, second measures
        for _r in _H:
            _m.reset(keep_baseline=True)
            for _i, (_e, _t1, _mg) in enumerate(_r["sig"]):
                _m.observe_stats(entropy=_e, top1=_t1, margin=_mg)
                if _i < len(_r["tokens"]): _m.observe_token(_r["tokens"][_i])
                _d = _m.is_degraded()
                if _pass and _d is not None:
                    _tot += 1; _fp += bool(_d)
    _rate = 100.0 * _fp / max(_tot, 1)
    check("false-alarm rate on REAL healthy generations is under the 10% design point",
          _rate <= 10.0, f"{_rate:.1f}% of {_tot} healthy windows flagged")
    print(f"        ({len(_H)} real generations, toll off, {_tot} scored windows, "
          f"learned bar {_m.z_bar():.2f})")

print("\n" + "=" * 78); print("12. THE REPETITION BAR MUST NOT LOCK"); print("=" * 78)
# A register that is genuinely more repetitive than the one the meter learned on (code, versus
# prose) must be able to raise the bar. Gating the history at the bar itself makes that
# impossible forever: every window above it is excluded, so the history never learns. Measured
# before the fix -- a coding session stayed pinned at 0.150 for as long as it ran.
def rep_stream(m, rho, nwin, base=0):
    for j in range(nwin):
        for x in rep_toks(rho, base + 1000 * (j % 7)):
            m.observe_token(x); m.observe_stats(entropy=1.0, top1=0.6, margin=0.4)

m = AdaptiveMeter(window=W, warmup=2, min_window=8)
rep_stream(m, 0.02, 40)                      # learn a low-repetition register
low = m.rep_threshold()
rep_stream(m, 0.22, 400)                     # then switch to a repetitive one, sustained
check("a sustained more-repetitive register can raise the learned bar",
      m.rep_threshold() > low + 0.02, f"{low:.3f} -> {m.rep_threshold():.3f}")

# ...but that adaptation must not swallow a real loop. Degenerate looping runs far above the
# band the gate admits, so it stays out of the history no matter how long it runs.
m = AdaptiveMeter(window=W, warmup=2, min_window=8)
rep_stream(m, 0.02, 40)
rep_stream(m, 0.95, 400)                     # a loop running for a very long time
check("a sustained degenerate loop cannot teach the meter that looping is normal",
      m.rep_threshold() < m.rep_ceiling, f"bar rose to {m.rep_threshold():.3f}")
check("and it is still flagged after running that long",
      m._looping(), f"rep {m.repetition():.3f} vs bar {m.rep_threshold():.3f}")

print("\n" + "=" * 78); print("13. THE FALSE-ALARM RATE IS THE DESIGN POINT"); print("=" * 78)
# z_quantile IS the false-alarm rate. If these ever come apart, a caller who asked for 1% is
# silently getting something else -- and the rate is the number the auto-tuner's usefulness
# rests on, not the correlation.
def steady(m, n, rg):
    fired = seen = 0
    for _ in range(n):
        p1 = float(np.clip(0.55 + 0.08 * rg.standard_normal(), 0.05, 0.98))
        m.observe(conf(p1)); m.observe_token(int(rg.integers(0, 5000)))
        d = m.is_degraded()
        if d is not None:
            seen += 1; fired += bool(d)
    return fired, seen

# On REAL output, because the synthetic stream above sits far inside the bar and reports ~0.5%
# for every quantile -- it cannot tell these settings apart at all.
_PT = os.path.join(ROOT, "data/results/mon_pertoken.json")
if not os.path.exists(_PT):
    print(f"  SKIPPED — {_PT} absent; the synthetic stream cannot distinguish design points.")
    FAIL.append("design-point test could not run")
else:
    import json as _j
    _H = [r for r in _j.load(open(_PT))["runs"] if r["sig"] and r["lam"] == 0.0]
    rates = {}
    for q in (0.90, 0.99):
        mq = AdaptiveMeter(window=64, warmup=2, min_window=16, z_quantile=q)
        fq = tq = 0
        for _pass in (0, 1):
            for r in _H:
                mq.reset(keep_baseline=True)
                for i, (e, t1, mg) in enumerate(r["sig"]):
                    mq.observe_stats(entropy=e, top1=t1, margin=mg)
                    if i < len(r["tokens"]): mq.observe_token(r["tokens"][i])
                    d = mq.is_degraded()
                    if _pass and d is not None:
                        tq += 1; fq += bool(d)
        rates[q] = 100.0 * fq / max(tq, 1)
    check("a tighter quantile gives a lower false-alarm rate on real output",
          rates[0.99] < rates[0.90], f"0.90 -> {rates[0.90]:.1f}%, 0.99 -> {rates[0.99]:.1f}%")
    check("the requested rate is roughly what is delivered",
          rates[0.99] <= 5.0 and rates[0.90] <= 15.0,
          f"0.90 -> {rates[0.90]:.1f}% (want <=15), 0.99 -> {rates[0.99]:.1f}% (want <=5)")

# and the control that a previous session skipped: a long run does not become twitchy because
# of a REGISTER CHANGE, it converges to the design point regardless of what it is watching
rg = np.random.default_rng(4)
m = AdaptiveMeter(window=64, warmup=2, min_window=16)
steady(m, 3000, rg)
f1, n1 = steady(m, 3000, rg)
steady(m, 12000, rg)                      # a long stretch of THE SAME register
f2, n2 = steady(m, 3000, rg)
early, late = 100.0 * f1 / max(n1, 1), 100.0 * f2 / max(n2, 1)
check("a long run on ONE register also drifts toward the design point (so a rise after a "
      "register change is not evidence of a register problem)",
      late >= early - 1.0, f"early {early:.1f}% -> late {late:.1f}%")

print("\n" + "=" * 78)
print("REPETITION THE PROMPT ASKED FOR IS NOT DAMAGE")
print("=" * 78)
# THE FALSE ALARM THIS REMOVES, MEASURED ON THE RUNNING ENGINE.
#     The repetition signal counts n-grams the window has seen before. That is the right test for
#     a model stuck in a loop and the wrong one for a model doing as it was told: asked to quote a
#     document, fill a table or continue a list, a HEALTHY model repeats. Observed at 7% of all
#     tokens flagged, every one of them from requested repetition or from replies too short to
#     judge. After this: 0 of 341 across the same three cases.
#
# The check that matters is the LAST one. Excusing repetition that appears in the prompt must not
# become a way to hide a genuine loop running alongside a quoted passage.
import numpy as _np                                                     # noqa: E402
_rng = _np.random.default_rng(0)


def _run(tokens, prompt=None, n=None):
    mm = AdaptiveMeter()
    if prompt is not None:
        mm.set_prompt(prompt)
    bad = 0
    for t in tokens:
        mm.observe_stats(2.0 + _rng.normal(0, .1), 0.5, 0.3)
        mm.observe_token(int(t))
        if mm.is_degraded():
            bad += 1
    return bad, len(tokens)


_healthy = list(_rng.integers(0, 30000, 400))
_loop = list(_rng.integers(0, 30000, 60)) + [7, 8, 9, 10] * 85
_quote = list(_rng.integers(0, 30000, 40)) * 10
_body = list(_rng.integers(0, 30000, 40)) * 2
_mixed = _body + [11, 12, 13, 14] * 80

check("healthy varied output is not flagged", _run(_healthy)[0] == 0, str(_run(_healthy)[0]))
check("a hard loop is still caught", _run(_loop)[0] > 20, f"{_run(_loop)[0]} of 400")
check("quoting a passage the prompt contains is not flagged",
      _run(_quote, prompt=_quote)[0] == 0, str(_run(_quote, prompt=_quote)[0]))
check("...and a loop running ALONGSIDE quoted text is still caught, so this is not an "
      "excuse-anything switch",
      _run(_mixed, prompt=_body)[0] > 20, f"{_run(_mixed, prompt=_body)[0]} of 400")
_m = AdaptiveMeter()
_m.set_prompt(list(range(500)))
check("the prompt's n-grams are stored as hashes, not tuples, so a long prompt is cheap",
      all(isinstance(x, int) for x in list(_m._prompt_grams)[:5]))
check("...and bounded, so an enormous prompt cannot dominate the process",
      AdaptiveMeter.PROMPT_GRAM_LIMIT <= 32768)

print("\n" + "=" * 78)
print("A REPLY TOO SHORT TO JUDGE MUST SAY SO, NOT PASS OR FAIL")
print("=" * 78)
# TWO BUGS, ONE SYMPTOM. `_looping()` and `_energy_degraded()` were both tested BEFORE the
# readiness gate, and the meter was never reset between replies -- so its 64-token window spanned
# the boundary between two answers. Four consecutive 16-token replies to the same question
# flagged 1, 6, 15 and 15 tokens; the model was not looping, the meter was reading the end of one
# reply and the start of the next as a single stretch of text.
_short = AdaptiveMeter()
_short.set_prompt([1, 2, 3])
for _t in [5] * 16:                       # a hard loop, but only sixteen tokens of it
    _short.observe_stats(2.0, .5, .3)
    _short.observe_token(_t)
check("a reply shorter than the floor returns None, not a verdict",
      _short.is_degraded() is None, repr(_short.is_degraded()))
check("...and gives no reason either, rather than a reason nobody may act on",
      _short.reason() is None)
check("None is distinguishable from 'fine', which is the whole point",
      _short.is_degraded() is not False)
for _t in [5] * 60:                       # keep going past the floor
    _short.observe_stats(2.0, .5, .3)
    _short.observe_token(_t)
check("...and once there IS enough output, the same loop is flagged",
      _short.is_degraded() is True, repr(_short.is_degraded()))
check("the floor is a real number of tokens, not zero",
      AdaptiveMeter().min_reply_tokens >= 16, str(AdaptiveMeter().min_reply_tokens))
# The engine must actually start each reply fresh, or the window keeps spanning answers.
import inspect as _i                                                    # noqa: E402
from bigrig_engine import session as _S                                 # noqa: E402
_st = _i.getsource(_S.Session.stream_text)
check("the engine resets the meter at the start of every reply",
      "self.meter.reset(keep_baseline=True)" in _st)
check("...keeping what it learned about the model, which is the point of self-calibration",
      "keep_baseline=True" in _st)
check("...and tells it what was asked, so requested repetition is excused",
      "self.meter.set_prompt(" in _st)

print("\n" + "=" * 78)
print("NO STATISTIC OF THE TOKEN STREAM DETECTS QUANTISATION, AND THAT IS NOT A TUNING FAILURE")
print("=" * 78)
# MEASURED TWICE, SIX STATISTICS, ON OLMoE-1B-7B-0125 with everything resident so streaming plays
# no part. Twelve prompts per condition, experts round-tripped through a lower precision so they
# carry exactly its error. AUROC against healthy output:
#
#     statistic            3-bit    2-bit
#     free energy          0.458    0.312     inverted -- quantised output looks MORE confident
#     repetition           0.625    0.708     weak
#     mean top-1 prob      0.486    0.444     chance
#     mean entropy         0.569    0.701     weak
#     mean max logit       0.569    0.125     nothing at 3 bits, strong (inverted) at 2
#     mean logsumexp       0.576    0.104     same
#
# At 2 bits several of them separate, which only says that visibly broken output is visibly
# broken. At 3 BITS -- what this product ships -- every one is chance.
#
# That is a definition, not a threshold that needs moving. "Damage" means "different from the
# model you would otherwise have run", and no statistic of a single token stream can see a
# difference from something it has never seen. The meter has no copy of the original.
#
# So the meter is not asked to do it, and `bigrig diff` supplies the missing half: the same
# prompt through the same weights, once as they ship and once carrying a lower precision's error.
_cli = open(os.path.join(ROOT, "bigrig_engine", "cli.py"), encoding="utf-8").read()
check("a command exists that supplies the reference the meter lacks", "def cmd_diff" in _cli)
check("...and it is registered, not dead code", 'sub.add_parser("diff"' in _cli)
check("...and it says why it is not the meter", "coin flip" in _cli and "0.500" in _cli)
check("...and records that six statistics were tried, not one", "logsumexp" in _cli)
check("it forces streaming, because the comparison needs the pools",
      "force_stream=True" in _cli)
check("...and says why that does not bias the result", "bit-exact" in _cli)

print("\n" + "=" * 78)
print(f"{'ALL TESTS PASSED' if not FAIL else str(len(FAIL)) + ' FAILURES: ' + ', '.join(FAIL)}")
print("=" * 78)
sys.exit(1 if FAIL else 0)
