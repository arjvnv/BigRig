"""Adversarial tests for QualityMeter. Written to BREAK it, not to confirm it works.

Every test here exists because it is a way the meter could silently produce a wrong number in
production -- and a quality meter that is silently wrong is worse than no meter at all.
"""
import os
import math
import sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from bigrig_layer import QualityMeter, CALIBRATION

FAIL = []
def check(name, cond, detail=""):
    (print(f"  PASS  {name}") if cond else (FAIL.append(name), print(f"  FAIL  {name}  {detail}")))

def confident(v=1000, p1=0.9):
    p = np.full(v, (1 - p1) / (v - 1)); p[0] = p1; return p

def uniform(v=1000):
    return np.full(v, 1.0 / v)

print("=" * 78); print("1. CORE BEHAVIOUR"); print("=" * 78)
m = QualityMeter()
check("score is None before the window fills", m.score() is None)
check("is_degraded is None before the window fills", m.is_degraded() is None)
for _ in range(CALIBRATION["window"] - 1):
    m.observe(confident())
check("still None one token short", m.score() is None)
m.observe(confident())
check("score appears exactly at the window size", m.score() is not None)
s_conf = m.score()

m2 = QualityMeter()
for _ in range(64): m2.observe(uniform())
s_unif = m2.score()
check("uniform (max uncertainty) scores WORSE than confident",
      s_unif > s_conf, f"{s_unif:.3f} vs {s_conf:.3f}")
check("confident is not flagged", m.is_degraded() is False, f"score {s_conf:.3f}")
check("uniform IS flagged", m2.is_degraded() is True, f"score {s_unif:.3f}")

print("\n" + "=" * 78); print("2. MONOTONICITY — the property a meter must have"); print("=" * 78)
scores = []
for p1 in [0.99, 0.9, 0.7, 0.5, 0.3, 0.1]:
    mm = QualityMeter()
    for _ in range(64): mm.observe(confident(p1=p1))
    scores.append(mm.score())
mono = all(scores[i] < scores[i + 1] for i in range(len(scores) - 1))
check("score rises monotonically as confidence falls", mono,
      " ".join(f"{s:.2f}" for s in scores))

print("\n" + "=" * 78); print("3. INPUT ROBUSTNESS — every way a caller can feed it garbage"); print("=" * 78)
m = QualityMeter()
try:
    m.observe(); check("no-arg observe raises", False)
except ValueError: check("no-arg observe raises ValueError", True)
except Exception as e: check("no-arg observe raises ValueError", False, type(e).__name__)

for name, arr in [("python list", list(confident())),
                  ("float32", confident().astype(np.float32)),
                  ("float16", confident().astype(np.float16)),
                  ("2-D (1,V) shape", confident().reshape(1, -1)),
                  ("unnormalised (sums to 7)", confident() * 7.0)]:
    try:
        mm = QualityMeter()
        for _ in range(64): mm.observe(arr)
        v = mm.score()
        check(f"accepts {name}", v is not None and math.isfinite(v), f"got {v}")
    except Exception as e:
        check(f"accepts {name}", False, f"{type(e).__name__}: {e}")

try:
    mm = QualityMeter()
    lg = np.log(confident() + 1e-12)
    for _ in range(64): mm.observe(logits=lg)
    a = mm.score()
    mm2 = QualityMeter()
    for _ in range(64): mm2.observe(confident())
    b = mm2.score()
    check("logits= and probs= agree", abs(a - b) < 1e-6, f"{a:.6f} vs {b:.6f}")
except Exception as e:
    check("logits= and probs= agree", False, f"{type(e).__name__}: {e}")

print("\n" + "=" * 78); print("4. DEGENERATE INPUTS — must not crash, must not return NaN"); print("=" * 78)
V = 1000
cases = {
    "one-hot (zero entropy)": np.eye(V)[0],
    "two-hot tie": (np.eye(V)[0] + np.eye(V)[1]) / 2,
    "tiny vocab (2)": np.array([0.6, 0.4]),
    "very large vocab (200k)": np.full(200000, 1 / 200000),
    "contains exact zeros": np.concatenate([[0.5, 0.5], np.zeros(V - 2)]),
}
for name, arr in cases.items():
    try:
        mm = QualityMeter()
        for _ in range(64): mm.observe(arr)
        v = mm.score()
        check(f"{name}", v is not None and math.isfinite(v), f"got {v}")
    except Exception as e:
        check(f"{name}", False, f"{type(e).__name__}: {e}")

print("\n" + "=" * 78); print("5. WINDOW SEMANTICS"); print("=" * 78)
mm = QualityMeter(window=16)
for _ in range(16): mm.observe(confident())
early = mm.score()
for _ in range(16): mm.observe(uniform())
late = mm.score()
check("window slides — old tokens fall out", late > early, f"{early:.3f} -> {late:.3f}")
mm.reset()
check("reset clears the window", mm.score() is None and mm.n_observed == 0)
check("custom window size honoured", QualityMeter(window=8).window == 8)

print("\n" + "=" * 78); print("6. CALIBRATION API"); print("=" * 78)
rng = np.random.default_rng(0)
X = rng.normal(size=(60, 3)); tgt = np.exp(X @ [0.4, -0.3, 0.2] + 1.0)
c = QualityMeter.calibrate(X, tgt)
check("calibrate returns all required keys",
      set(["mean", "std", "weight", "intercept", "threshold"]) <= set(c))
check("calibrate weights are finite", all(math.isfinite(v) for v in c["weight"]))
mm = QualityMeter(calibration=c)
for _ in range(64): mm.observe(confident())
check("a re-calibrated meter still scores", math.isfinite(mm.score()))
check("shipped calibration is unchanged by re-calibration",
      CALIBRATION["weight"] == [-0.737, -2.953, 1.629])

print("\n" + "=" * 78); print("7. HOSTILE INPUTS — must fail LOUDLY, never return a plausible number"); print("=" * 78)
VV = 1000
hostile = {"contains NaN": np.where(np.arange(VV) == 5, np.nan, confident()),
           "contains +inf": np.where(np.arange(VV) == 5, np.inf, confident()),
           "all zeros": np.zeros(VV),
           "negative values": confident() - 0.001,
           "single element": np.array([1.0])}
for hname, harr in hostile.items():
    mm = QualityMeter(window=4)
    try:
        mm.observe(harr); check(hname + " raises in strict mode", False, "silently accepted")
    except ValueError:
        check(hname + " raises in strict mode", True)
    m2 = QualityMeter(window=4, strict=False)
    for _ in range(4): m2.observe(harr)
    check(hname + " skipped in lenient mode", m2.n_skipped == 4 and m2.score() is None)

print("\n" + "=" * 78); print("8. NUMERICAL PRECISION"); print("=" * 78)
aa, bb = QualityMeter(), QualityMeter()
big = np.full(200000, 1 / 200000)
for _ in range(64):
    aa.observe(big.astype(np.float32)); bb.observe(big.astype(np.float64))
check("float32 vs float64 agree on a 200k vocab", abs(aa.score() - bb.score()) < 1e-3)
for nm, lg in [("huge +logits", np.full(VV, 1e4)), ("huge -logits", np.full(VV, -1e4))]:
    mm = QualityMeter()
    for _ in range(64): mm.observe(logits=lg)
    check(nm + " do not overflow", math.isfinite(mm.score()))

print("\n" + "=" * 78); print("9. THE REAL TEST — shipped constants reproduce the research"); print("=" * 78)
import json
DD = json.load(open(os.path.join(ROOT, "data", "results", "mon2_data.json")))
TSTP = set(range(DD["nprompts"] // 2, DD["nprompts"]))
def _rho(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    return float(np.corrcoef(np.argsort(np.argsort(x)), np.argsort(np.argsort(y)))[0, 1])
per = []
for lam in DD["lambdas"]:
    sc, jd = [], []
    for r in DD["runs"]:
        if r["prompt"] not in TSTP or r["lam"] != lam or not r["win"]: continue
        if r["judge_nll"] != r["judge_nll"]: continue
        f = np.array([np.nanmean([w[k] for w in r["win"]])
                      for k in ["S1_ent", "S2_top1", "S3_margin"]])
        if not np.all(np.isfinite(f)): continue
        z = (f - np.array(CALIBRATION["mean"])) / np.array(CALIBRATION["std"])
        sc.append(float(z @ np.array(CALIBRATION["weight"]) + CALIBRATION["intercept"]))
        jd.append(r["judge_nll"])
    if len(sc) >= 6: per.append(_rho(sc, jd))
rr = float(np.nanmean(per))
check("reproduces research rho (%.3f vs +0.893)" % rr, abs(rr - 0.893) < 0.05)

print("\n" + "=" * 78)
print(f"{'ALL TESTS PASSED' if not FAIL else str(len(FAIL)) + ' FAILURES: ' + ', '.join(FAIL)}")
print("=" * 78)
sys.exit(1 if FAIL else 0)
