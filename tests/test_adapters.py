"""Adversarial tests for the engine adapters.

The adapters are the whole of Phase 1: they are what lets the meter attach to an engine we did
not write. Every failure mode here is one where the meter would keep reporting numbers that
LOOK fine while being computed from the wrong thing -- which is worse than crashing.
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bigrig_layer.adapters import (entropy_from_topk, observe_ollama_entry,
                                     observe_openai_chunk, observe_topk, stats_from_topk)
from bigrig_layer.adaptive import AdaptiveMeter

FAIL = []
def check(n, c, d=""):
    (print(f"  PASS  {n}") if c else (FAIL.append(n), print(f"  FAIL  {n}  {d}")))

RNG = np.random.default_rng(0)
V = 151936


def full_dist(alpha=0.3, n=V):
    p = RNG.dirichlet(np.ones(200) * alpha)
    q = np.zeros(n); q[:200] = p
    return q


print("=" * 78); print("1. top1 AND margin MUST BE EXACT FROM TOP-K"); print("=" * 78)
# These depend only on the two largest probabilities, so a truncated list loses NOTHING.
worst_t1 = worst_mg = 0.0
for _ in range(300):
    q = full_dist()
    srt = np.sort(q)[::-1]
    true_t1, true_mg = float(srt[0]), float(srt[0] - srt[1])
    for K in (2, 5, 8, 20):
        lp = np.log(np.maximum(srt[:K], 1e-300))
        _, t1, mg = stats_from_topk(lp, V)
        worst_t1 = max(worst_t1, abs(t1 - true_t1))
        worst_mg = max(worst_mg, abs(mg - true_mg))
check("top-1 is exact at every K down to 2", worst_t1 < 1e-12, f"max error {worst_t1:.2e}")
check("margin is exact at every K down to 2", worst_mg < 1e-12, f"max error {worst_mg:.2e}")

print("\n" + "=" * 78); print("2. THE ENTROPY ESTIMATORS BEHAVE AS CLAIMED"); print("=" * 78)
below = above = 0
for _ in range(300):
    q = full_dist()
    nz = q[q > 0]
    h_true = float(-(nz * np.log(nz)).sum())
    srt = np.sort(q)[::-1]
    lp = np.log(np.maximum(srt[:8], 1e-300))
    h_tr = entropy_from_topk(lp, V, "truncated")
    h_co = entropy_from_topk(lp, V, "tail_uniform")
    below += h_tr <= h_true + 1e-9
    above += h_co >= h_tr - 1e-9
check("truncated entropy never exceeds the true entropy", below == 300, f"{300-below} violations")
check("the tail correction never reduces the estimate", above == 300, f"{300-above} violations")

# a DEGENERATE case the correction must not blow up on: all mass in the top-K
lp = np.log(np.array([0.9999999, 1e-7]))
h = entropy_from_topk(lp, V, "tail_uniform")
check("no residual mass leaves the correction finite and small", np.isfinite(h) and h < 0.01,
      f"{h}")

print("\n" + "=" * 78); print("3. HOSTILE INPUT — must raise, never silently mislead"); print("=" * 78)
for name, args in [("a single logprob (margin undefined)", ([math.log(1.0)],)),
                   ("an empty list", ([],))]:
    try:
        stats_from_topk(args[0], V); check(f"{name} raises", False)
    except ValueError: check(f"{name} raises", True)
for bad in (float("nan"), float("inf")):
    try:
        stats_from_topk([math.log(0.5), bad], V); check(f"{bad} raises", False)
    except ValueError: check(f"{bad} raises", True)
try:
    entropy_from_topk([math.log(0.5), math.log(0.3)], V, method="made_up")
    check("an unknown estimator name raises", False)
except ValueError: check("an unknown estimator name raises", True)

# an engine that returns an UNSORTED list must not corrupt top1/margin
srt = np.array([0.5, 0.3, 0.15, 0.05])
lp_sorted = np.log(srt)
lp_shuf = np.log(srt[[2, 0, 3, 1]])
a = stats_from_topk(lp_sorted, V)
b = stats_from_topk(lp_shuf, V)
check("an unsorted engine payload is sorted rather than trusted",
      all(abs(x - y) < 1e-12 for x, y in zip(a, b)), f"{a} vs {b}")

print("\n" + "=" * 78); print("4. THE ENGINE SHIMS PARSE REAL PAYLOAD SHAPES"); print("=" * 78)
# verified against ollama 0.32.1 output
ollama_entry = {"token": "Three", "logprob": -1.1963, "bytes": [84, 104],
                "top_logprobs": [{"token": "Three", "logprob": -1.1963},
                                 {"token": "One", "logprob": -1.2004},
                                 {"token": "I", "logprob": -1.8832}]}
m = AdaptiveMeter(window=8, warmup=2, min_window=4)
r = observe_ollama_entry(m, ollama_entry, V)
check("ollama native entry is parsed", r is not None and m.n_observed == 1, str(r))
check("...and its margin matches the two logprobs exactly",
      abs(r[2] - (math.exp(-1.1963) - math.exp(-1.2004))) < 1e-12, str(r))

oai = {"token": " the", "logprob": -0.5,
       "top_logprobs": [{"token": " the", "logprob": -0.5}, {"token": " a", "logprob": -1.5}]}
m2 = AdaptiveMeter(window=8, warmup=2, min_window=4)
check("OpenAI-shaped chunk is parsed", observe_openai_chunk(m2, oai, V) is not None)

# the case a caller WILL hit: forgetting to ask for top_logprobs
m3 = AdaptiveMeter(window=8, warmup=2, min_window=4)
check("a payload with no top_logprobs returns None instead of guessing",
      observe_openai_chunk(m3, {"token": "x", "logprob": -1.0}, V) is None and m3.n_observed == 0)
check("a payload with only ONE top_logprob also returns None",
      observe_ollama_entry(m3, {"token": "x", "top_logprobs": [{"token": "x", "logprob": -1.0}]},
                           V) is None)

print("\n" + "=" * 78); print("5. TOKEN KEYS — repetition must still work on text tokens"); print("=" * 78)
m4 = AdaptiveMeter(window=16, warmup=2, min_window=8)
for i in range(64):
    e = {"token": "loop", "top_logprobs": [{"token": "loop", "logprob": -0.01},
                                           {"token": "x", "logprob": -9.0}]}
    observe_ollama_entry(m4, e, V)
_r4 = m4.repetition()
check("a repeated TEXT token is seen as repetition", _r4 is not None and _r4 > 0.9,
      f"repetition {_r4}")
m5 = AdaptiveMeter(window=16, warmup=2, min_window=8)
for i in range(64):
    e = {"token": f"tok{i}", "top_logprobs": [{"token": "a", "logprob": -0.01},
                                              {"token": "b", "logprob": -9.0}]}
    observe_ollama_entry(m5, e, V)
_r5 = m5.repetition()
check("distinct text tokens are NOT seen as repetition", _r5 is not None and _r5 < 0.05,
      f"repetition {_r5}")

print("\n" + "=" * 78); print("6. THE ADAPTER PATH EQUALS THE DIRECT PATH"); print("=" * 78)
# Feeding through the adapter must leave the meter in the same state as feeding observe_stats
# directly. If these drift, every number the layer reports on a real engine is quietly wrong.
a1 = AdaptiveMeter(window=16, warmup=2, min_window=8)
a2 = AdaptiveMeter(window=16, warmup=2, min_window=8)
for i in range(80):
    q = full_dist()
    srt = np.sort(q)[::-1]
    lp = np.log(np.maximum(srt[:8], 1e-300))
    h, t1, mg = stats_from_topk(lp, V)
    observe_topk(a1, lp, V, token=i)
    a2.observe_stats(entropy=h, top1=t1, margin=mg); a2.observe_token(i)
z1, z2 = a1.z_score(), a2.z_score()
check("adapter and direct paths give an identical z", z1 is not None and z1 == z2, f"{z1} vs {z2}")
check("...and identical observation counts", a1.n_observed == a2.n_observed == 80)

print("\n" + "=" * 78); print("7. DROPPING ENTROPY DELIBERATELY (method='none')"); print("=" * 78)
# Some engines expose too few logprobs for entropy to mean anything. The layer must degrade to
# top-1 + margin + repetition rather than feed the meter a number it invented.
import warnings as _w
h, t1, mg = stats_from_topk([math.log(0.6), math.log(0.4)], V, method="none")
check("method='none' returns None for entropy and exact top1/margin",
      h is None and abs(t1 - 0.6) < 1e-12 and abs(mg - 0.2) < 1e-12, f"{(h, t1, mg)}")
try:
    entropy_from_topk([math.log(0.6), math.log(0.4)], V, method="none")
    check("entropy_from_topk refuses method='none' with a useful message", False)
except ValueError as e:
    check("entropy_from_topk refuses method='none' with a useful message", "observe_topk" in str(e))

mn = AdaptiveMeter(window=16, warmup=2, min_window=8)
with _w.catch_warnings():
    _w.simplefilter("ignore")
    for i in range(80):
        observe_topk(mn, [math.log(0.6), math.log(0.4)], V, token=i, method="none")
check("the meter still runs with entropy dropped", mn.n_observed == 80 and mn.z_score() is not None,
      f"observed {mn.n_observed}, z {mn.z_score()}")
check("...and the dropped entropy contributes nothing rather than a false signal",
      abs(mn.z_score()) < 1e-6, f"z {mn.z_score()}")

# looping must STILL be caught with entropy dropped -- it is the repetition channel that does it
ml = AdaptiveMeter(window=16, warmup=2, min_window=8)
with _w.catch_warnings():
    _w.simplefilter("ignore")
    for i in range(200):
        observe_topk(ml, [math.log(0.6), math.log(0.4)], V, token=i, method="none")
    for i in range(200):
        observe_topk(ml, [math.log(0.6), math.log(0.4)], V, token=7, method="none")
check("looping is still caught when entropy is dropped",
      ml.is_degraded() and ml.reason() == "looping", f"reason {ml.reason()}")

print("\n" + "=" * 78); print("8. THE K GUARD"); print("=" * 78)
with _w.catch_warnings(record=True) as rec:
    _w.simplefilter("always")
    stats_from_topk([math.log(0.6), math.log(0.4)], V)
check("a K below the smallest measured value warns", len(rec) == 1 and
      issubclass(rec[0].category, RuntimeWarning), f"{len(rec)} warnings")
with _w.catch_warnings(record=True) as rec:
    _w.simplefilter("always")
    stats_from_topk([math.log(x) for x in (0.4, 0.3, 0.15, 0.1, 0.05)], V)
check("K=5, the smallest measured, is silent", len(rec) == 0, f"{len(rec)} warnings")
with _w.catch_warnings(record=True) as rec:
    _w.simplefilter("always")
    stats_from_topk([math.log(0.6), math.log(0.4)], V, method="none")
check("...and dropping entropy on purpose does not warn about entropy", len(rec) == 0)

print("\n" + "=" * 78)
print(f"{'ALL TESTS PASSED' if not FAIL else str(len(FAIL)) + ' FAILURES: ' + ', '.join(FAIL)}")

print("\n" + "=" * 78)
print("FULL-VOCABULARY PATH (local engines hand back the whole distribution)")
print("=" * 78)
from bigrig_layer.adapters import stats_from_logprobs
import time as _t
V = 8192
_rng = np.random.default_rng(11)
_lg = _rng.normal(size=V).astype(np.float32) * 3
_lp = (_lg - _lg.max()) - np.log(np.exp(_lg - _lg.max()).sum())
h, t1, mg = stats_from_logprobs(_lp)
_p = np.exp(_lp.astype(np.float64))
exact = float(-(_p * np.log(_p)).sum())
srt = np.sort(_p)[::-1]
check("entropy matches the exact definition", abs(h - exact) < 1e-3, f"{h} vs {exact}")
check("top1 is the largest probability", abs(t1 - srt[0]) < 1e-6)
check("margin is the gap to the runner-up", abs(mg - (srt[0] - srt[1])) < 1e-6)
check("it agrees with the top-k path given the same full vector",
      abs(h - stats_from_topk(_lp, V)[0]) < 1e-3)
_u = np.full(V, -np.log(V), dtype=np.float32)
check("a uniform distribution gives log(V) nats",
      abs(stats_from_logprobs(_u)[0] - np.log(V)) < 1e-2)
_d = np.full(V, -60.0, dtype=np.float32); _d[3] = 0.0
check("a one-hot distribution gives ~0 nats", stats_from_logprobs(_d)[0] < 1e-3)
check("a margin of ~1 comes out of a one-hot distribution",
      abs(stats_from_logprobs(_d)[2] - 1.0) < 1e-3)
for bad, why in ((np.array([0.0]), "a single entry"),
                 (np.array([0.0, np.nan]), "a NaN"),
                 (np.array([0.0, np.inf]), "an infinity")):
    try:
        stats_from_logprobs(bad); check(f"rejects {why}", False, "accepted")
    except ValueError:
        check(f"rejects {why}", True)
_big = np.concatenate([_lp] * 18)[:151936].astype(np.float32)
_t0 = _t.perf_counter()
for _ in range(10): stats_from_logprobs(_big)
_fast = (_t.perf_counter() - _t0) / 10 * 1000
_t0 = _t.perf_counter()
for _ in range(10): stats_from_topk(_big, 151936)
_slow = (_t.perf_counter() - _t0) / 10 * 1000
check("the exact path is materially cheaper than the defensive sort",
      _fast < _slow * 0.5, f"{_fast:.2f} ms vs {_slow:.2f} ms")
print(f"        {_fast:.2f} ms/token vs {_slow:.2f} ms through the top-k path "
      f"({_slow/_fast:.1f}x) on a 151,936-token vocabulary")

print("=" * 78)
sys.exit(1 if FAIL else 0)