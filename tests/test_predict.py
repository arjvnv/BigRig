"""Reading a layer early: what it predicts from, and why being wrong is safe.

THE MEASUREMENT THAT REVERSED AN EARLIER CONCLUSION
    This project measured, on a 2,271-token trace, that expert prefetching was impossible: the
    same layer's previous token named 0.00% of the misses and the previous layer 5.91%. That is
    true, and it is about the wrong signal. Predicting from the HIDDEN STATE instead, a linear
    ridge fit recovers the NEXT layer's routing at 64.5% recall@8 within a prompt, and 52.8%
    across held-out prompts. History cannot predict a miss -- an expert the recent past used is
    still resident. The hidden state can.

WHY A WRONG GUESS CANNOT MOVE A LOGIT
    The prediction decides ONE thing: what to start reading. `ensure()` is untouched -- it takes
    the real router's output and admits exactly the experts actually selected. Verified end to
    end at three prefetch widths: byte-identical replies in every case.
"""
import inspect
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bigrig_engine import predict, session, stream  # noqa: E402

FAIL = []


def check(n, c, d=""):
    (print(f"  PASS  {n}") if c else (FAIL.append(n), print(f"  FAIL  {n}  {d}")))


print("=" * 84); print("1. THE FIT"); print("=" * 84)
rng = np.random.default_rng(0)
# A signal that IS there: experts chosen by a linear function of the state, plus noise.
X = rng.normal(size=(400, 64)).astype(np.float32)
true_W = rng.normal(size=(64, 32)).astype(np.float32)
S = X @ true_W + 0.3 * rng.normal(size=(400, 32)).astype(np.float32)
Y = np.zeros((400, 32), dtype=np.float32)
for r in range(400):
    Y[r, np.argpartition(-S[r], 3)[:4]] = 1.0
W = predict.fit(X[:300], Y[:300])
truth = [np.argpartition(-S[r], 3)[:4] for r in range(300, 400)]
r8 = predict.recall_at(X[300:] @ W, truth, 8)
check("a linear fit recovers a linear signal it has not seen", r8 > 0.8, f"{r8:.1%}")
check("recall is bounded by one", predict.recall_at(X[300:] @ W, truth, 32) <= 1.0)
check("...and naming more experts never lowers it",
      predict.recall_at(X[300:] @ W, truth, 16) >= r8 - 1e-9)
# Pure noise must NOT look predictable, or the go/no-go gate is worthless.
Yn = np.zeros((400, 32), dtype=np.float32)
for r in range(400):
    Yn[r, rng.choice(32, 4, replace=False)] = 1.0
rn = predict.recall_at(X[300:] @ predict.fit(X[:300], Yn[:300]), 
                       [np.nonzero(Yn[r])[0] for r in range(300, 400)], 8)
check("noise does not look predictable", rn < 0.45, f"{rn:.1%}")
check("the fit is regularised, because the hidden dimensions are correlated",
      predict.RIDGE > 0 and "ridge" in inspect.getsource(predict.fit).lower())
check("an empty truth set scores zero rather than dividing by zero",
      predict.recall_at(np.zeros((0, 32)), [], 8) == 0.0)

print("\n" + "=" * 84); print("2. IT CANNOT CHANGE AN ANSWER"); print("=" * 84)
_call = inspect.getsource(stream.StreamingSwitchGLU.__call__)
_spec = inspect.getsource(stream.StreamingSwitchGLU._speculate)
_ens = inspect.getsource(stream.StreamingSwitchGLU.ensure)
check("ensure() knows nothing about predictions",
      "pred" not in _ens and "specul" not in _ens)
check("...so what is admitted is still exactly what the router selected",
      "self._pool.touch(uniq" in _ens)
check("the prediction is used only to NAME experts for the next layer to stage",
      "nxt.stage_pred = want or None" in _spec)      # that it never reads is the AST check below
# The docstring explains that it does NOT admit, so grepping the source matched its own
# explanation. Strip the prose and look at the code -- the rule in CONTRIBUTING.
import ast  # noqa: E402
_body = ast.parse(inspect.getsource(stream.StreamingSwitchGLU._speculate).lstrip()).body[0]
_calls = {n.func.attr for n in ast.walk(_body)
          if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
check("...and never to admit, evict or remap anything",
      not ({"admit", "touch", "_victim", "evict"} & _calls), str(sorted(_calls)))
check("...it only names and counts", _calls <= {"append"}, str(sorted(_calls)))
check("it is off unless staging is switched on, so an unmeasured machine runs the plain path",
      "if not STAGING:" in _spec)

print("\n" + "=" * 84); print("3. IT COSTS NO EXTRA ROUND-TRIP"); print("=" * 84)
# The host read is already a synchronisation point. Evaluating the prediction WITH it means the
# guess rides on a stall that was going to happen anyway.
_c = " ".join(_call.split())
check("the prediction is queued before the read that drains the queue",
      _c.index("self._next_gate(xf)") < _c.index("mx.eval(indices, pred)"))
check("the NEXT layer's own router is the predictor of first resort, the fitted map the fallback",
      _c.index("self._next_gate(xf)") < _c.index("xf @ self._pred_w"))
check("...and its measured recall on this model is written beside it", "83.6%" in _c and "47.3" in _c)
check("the next router is wired for every streamed layer without needing a fitted file",
      "mod._next_gate = gate" in inspect.getsource(stream.attach))
check("...and both are evaluated in a single drain", "mx.eval(indices, pred)" in _c)
check("with staging off, or nothing to predict for, the original path runs untouched",
      "if STAGING and self._next is not None and self._pred_m > 0 and (" in _c)

print("\n" + "=" * 84); print("4. WHAT IT ASKS FOR"); print("=" * 84)


class _Pool:
    def __init__(self, layer, resident):
        self.layer = layer
        self.g2s = np.full(64, -1, dtype=np.int64)
        for i, e in enumerate(resident):
            self.g2s[e] = i


class _Fetch:
    def __init__(self, boom=False):
        self.asked, self.dropped, self.boom = [], [], boom


_prev_staging = stream.STAGING
stream.STAGING = True                       # the naming is what is under test here
def named(m):
    return sorted(m._next.stage_pred or [])


def mod(resident, boom=False):
    o = type("M", (), {})()
    o._next = _Pool(7, resident)
    o._fetcher = _Fetch(boom)
    o._speculated = ()
    o.pred_issued = o.pred_dropped = 0
    o._speculate = stream.StreamingSwitchGLU._speculate.__get__(o)
    return o


m = mod(resident=[1, 2, 3])
m._speculate(np.array([1, 2, 9, 40]))
check("an expert already in the next layer's pool is not named again",
      named(m) == [9, 40], str(m._next.stage_pred))
check("...and it is named ON the next layer's pool, the one being predicted for",
      m._next.layer == 7 and m.pred_issued == 2)
m2 = mod(resident=[1, 2, 3])
m2._speculate(np.array([1, 2, 3]))
check("predicting only resident experts names nothing", m2._next.stage_pred is None)
m4 = mod(resident=[])
m4._speculate(np.array([5, 6]))
m4._speculate(np.array([7, 8]))
check("a new prediction replaces the last one rather than piling on", named(m4) == [7, 8])
m8 = mod(resident=[])
m8._speculate(np.array(list(range(20))))
check("naming is capped at the router's top-k so the copy fits inside the wait",
      len(named(m8)) == stream.STAGE_MAX == 8, str(len(named(m8))))

print("\n" + "=" * 84); print("5. SPECULATION IS CAPPED IN BYTES, NOT JUST IN COUNT"); print("=" * 84)
# THE BUG: naming 8 experts costs 8 x bytes-per-expert x layers of reads per token, and only the
# non-resident ones are actually read. On Qwen3 at 31% residency that is cheap. On gpt-oss at 3%
# residency almost every guess is a real read: 8 x 13.24 MB x 36 layers = 3.70 GB of speculation
# a token against 1.9 GB genuinely needed. Measured 0.65x throughput and 9.01 GB of RSS against a
# 9.00 GB ceiling -- the guard killed it. Capped, the same run peaks at 6.97 GB.


class _BigPool(_Pool):
    bytes_per_expert = 13_240_000          # gpt-oss: one expert at one layer


m5 = mod(resident=[])
m5._next = _BigPool(7, [])
m5._pred_budget = 268_435_456 // 36        # the shared budget, split over 36 layers
m5._speculate(np.array(list(range(8))))
check("a model with large experts is held to what the budget affords",
      len(named(m5)) == 1, f"{len(named(m5))} named")


class _SmallPool(_Pool):
    bytes_per_expert = 2_060_000           # Qwen3


m6 = mod(resident=[])
m6._next = _SmallPool(7, [])
m6._pred_budget = 268_435_456 // 48
m6._speculate(np.array(list(range(8))))
check("...and a model with small experts is allowed more", len(named(m6)) == 2)
m7 = mod(resident=[])
m7._next = _SmallPool(7, [])
m7._pred_budget = 0                        # no budget set: fall back to the count
m7._speculate(np.array([1, 2, 3]))
check("with no byte budget the count still applies", len(named(m7)) == 3)
check("a budget smaller than one expert still allows one, not zero", len(named(m5)) >= 1)
stream.STAGING = _prev_staging
check("the cap is a real number, not unlimited",
      0 < stream.SPECULATION_BUDGET_BYTES <= (1 << 30))

print("\n" + "=" * 84); print("6. A PREDICTOR IS NEVER BORROWED FROM ANOTHER MODEL"); print("=" * 84)
check("a model with no predictor gets None", predict.load("no-such-model-xyz") is None)
check("the file is named for the model", "predict_" in predict.predictor_path("m"))
check("...and a name with a slash cannot escape the results directory",
      "/" not in os.path.basename(predict.predictor_path("a/../b")))
_cli = open(os.path.join(ROOT, "bigrig_engine/cli.py")).read()
check("a predictor too weak to be worth the bandwidth is not saved",
      "too low to be worth the bandwidth" in _cli)
check("the flag says a wrong guess cannot change an answer",
      "never changes an answer" in _cli)

print("\n" + "=" * 84); print("7. CACHE-AWARE REROUTING CHANGES THE ANSWER, AND SAYS SO")
print("=" * 84)
# Everything else in this engine moves weights without touching them. This does not: a token
# whose expert is not in memory can be sent to a resident one that the router scored nearly as
# highly, and a different expert then runs. It is a speed-for-quality trade like compression, and
# like compression it is off unless asked for.
_rr = inspect.getsource(stream.StreamingSwitchGLU._reroute_to_resident)
check("it is off unless a tolerance is given",
      stream.StreamingSwitchGLU.__init__.__defaults__ is not None
      or "self._reroute = 0.0" in inspect.getsource(stream.StreamingSwitchGLU.__init__))
_c2 = " ".join(inspect.getsource(stream.StreamingSwitchGLU.__call__).split())
check("...and the check is explicit at the call site", "if self._reroute > 0.0" in _c2)
check("only an expert that is NOT resident is ever replaced",
      "if self._pool.g2s[e] >= 0: continue" in " ".join(_rr.split()))
check("...by one that IS resident", "np.nonzero(self._pool.g2s >= 0)" in _rr)
check("...and only when it scored within the tolerance",
      "row[best] >= row[e] * (1.0 - self._reroute)" in _rr)
check("a substitute is never chosen twice for the same token",
      "free = [c for c in free if c != best]" in _rr)
check("...nor one the token had already chosen", "if c not in chosen" in _rr)
check("the router's scores are softmaxed before being compared",
      "mx.softmax(self._gate(x)" in _rr)
check("every substitution is counted, and what it cost is recorded",
      "self.reroutes += 1" in _rr and "self.reroute_lost +=" in _rr)
check("...and reported, so the trade is visible rather than silent",
      '"reroutes"' in inspect.getsource(stream.StreamHandle.stats))
check("the session reports the tolerance it is running at",
      '"reroute": self.reroute_tol or None' in inspect.getsource(session.Session.stats))
check("the flag says plainly that the output changes",
      "THIS CHANGES THE OUTPUT" in open(
          os.path.join(ROOT, "bigrig_engine/cli.py")).read())
check("...and that it is a trade, not a free win",
      "not a free win" in open(
          os.path.join(ROOT, "bigrig_engine/cli.py")).read())
_att = inspect.getsource(stream.attach)
check("switching it on is announced at load, not buried",
      "THE OUTPUT CHANGES" in _att)

print("\n" + "=" * 84)
print(f"{'ALL TESTS PASSED' if not FAIL else str(len(FAIL)) + ' FAILURES: ' + ', '.join(FAIL)}")
print("=" * 84)
sys.exit(1 if FAIL else 0)
