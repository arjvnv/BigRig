"""Adversarial tests for prefetch.

The failure mode this guards against is a prefetcher that LOOKS like it helps: it hides latency
in a microbenchmark while spending bandwidth the engine cannot spare, and the cost only shows up
as a lower token rate that nobody attributes to it. So the speculative path is tested for whether
it is honest about its own cost, not just for whether it runs.
"""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bigrig_engine.fetch import ParallelFetcher, Region, WeightStore
from bigrig_engine.prefetch import (CoOccurrencePredictor, PipelinedLoader, PrefetchController,
                                      speculation_pays)

FAIL = []
def check(n, c, d=""):
    (print(f"  PASS  {n}") if c else (FAIL.append(n), print(f"  FAIL  {n}  {d}")))


TMP = tempfile.NamedTemporaryFile(prefix="bigrig_pf_", suffix=".bin", delete=False)
BLK = 1 << 20
N = 32
rng = np.random.default_rng(0)
DATA = rng.integers(0, 255, BLK * N, dtype=np.uint8).tobytes()
TMP.write(DATA); TMP.close()
LAYOUT = {i: Region(i * BLK, BLK) for i in range(N)}
STORE = WeightStore(TMP.name, LAYOUT)

print("=" * 78); print("1. PIPELINING MUST NOT CHANGE THE BYTES"); print("=" * 78)
with ParallelFetcher(STORE, threads=8) as f:
    keys = list(range(8))
    streamed = dict(PipelinedLoader(f).stream(keys))
check("streaming returns every requested key", set(streamed) == set(keys))
check("streamed bytes are correct",
      all(streamed[k] == DATA[k * BLK:(k + 1) * BLK] for k in keys))
with ParallelFetcher(STORE, threads=8) as f:
    blocking = f.fetch(list(range(8)))
check("streamed bytes equal blocking-fetch bytes", streamed == blocking)

with ParallelFetcher(STORE, threads=4) as f:
    check("streaming an empty list yields nothing",
          list(PipelinedLoader(f).stream([])) == [])

# duplicates must not deadlock or double-yield
with ParallelFetcher(STORE, threads=4) as f:
    got = list(PipelinedLoader(f).stream([1, 1, 2, 2, 2]))
check("duplicate keys do not deadlock or duplicate yields", len(got) == 2, f"{len(got)} yielded")

print("\n" + "=" * 78); print("2. THE PREDICTOR MUST BE CAUSAL AND HELD OUT"); print("=" * 78)
L, T, K, E = 4, 400, 8, 32
idx = rng.integers(0, E, (L, T, K))
# plant real structure: layer l's experts depend on layer l-1's, so a predictor CAN learn it
for l in range(1, L):
    idx[l] = (idx[l - 1] * 3 + 7) % E
p = CoOccurrencePredictor(E)
try:
    p.predict(1, idx[0, 0], 8); check("an unfitted predictor refuses to predict", False)
except RuntimeError:
    check("an unfitted predictor refuses to predict", True)
p.fit(idx, upto=T // 2)
pred = p.predict(1, idx[0, 0], 8)
check("prediction returns exactly `budget` candidates", len(pred) == 8, str(len(pred)))
check("candidates are valid expert ids", all(0 <= int(x) < E for x in pred))
r_train_free = p.recall(idx, budget=16, frm=T // 2)
check("a predictor CAN learn planted structure", r_train_free > 0.5, f"recall {r_train_free:.3f}")

# the honesty check: on structureless data it must NOT claim skill
idx_rand = rng.integers(0, E, (L, T, K))
p2 = CoOccurrencePredictor(E).fit(idx_rand, upto=T // 2)
r_rand = p2.recall(idx_rand, budget=16, frm=T // 2)
chance = 16 / E
check("on random routing the predictor scores near chance, not above",
      r_rand < chance * 1.6, f"recall {r_rand:.3f} vs chance {chance:.3f}")
print(f"        (planted structure {r_train_free:.3f}, random {r_rand:.3f}, chance {chance:.3f})")

print("\n" + "=" * 78); print("3. SPECULATION MUST BE HONEST ABOUT ITS COST"); print("=" * 78)
# The real measured numbers. If speculation_pays() ever recommends these, it is broken.
for model, budget, recall in (("olmoe", 8, 0.497), ("olmoe", 16, 0.713), ("olmoe", 32, 0.894),
                              ("ling", 16, 0.506), ("ling", 32, 0.655)):
    r = speculation_pays(recall, budget, 8, idle_fraction=1.0)
    check(f"{model} B={budget} (recall {recall}) is NOT recommended even on an idle link",
          not r["recommended"], str(r))
check("a hypothetical high-recall, low-budget predictor WOULD be recommended",
      speculation_pays(0.95, 9, 8, idle_fraction=1.0)["recommended"],
      str(speculation_pays(0.95, 9, 8, 1.0)))
check("cost ratio is budget/k", abs(speculation_pays(0.5, 16, 8, 1.0)["cost_ratio"] - 2.0) < 1e-9)
check("more budget at fixed recall means more waste",
      speculation_pays(0.7, 32, 8, 1.0)["wasted"] > speculation_pays(0.7, 16, 8, 1.0)["wasted"])

print("\n" + "=" * 78); print("4. THE CONTROLLER DEFAULTS TO SAFE"); print("=" * 78)
with ParallelFetcher(STORE, threads=4) as f:
    c = PrefetchController(f, predictor=None)
    check("speculation is OFF by default", c.speculate is False)
    n = c.speculate_next(1, [0, 1], lambda l, e: e)
    check("speculate_next does nothing when disabled", n == 0)
    list(c.layer([0, 1, 2]))
    rep = c.report()
    check("certain fetches are counted", rep["certain"] == 3, str(rep))
    check("speculative overhead is zero when disabled", rep["speculative_overhead"] == 0.0)

with ParallelFetcher(STORE, threads=4) as f:
    pp = CoOccurrencePredictor(E).fit(idx, upto=T // 2)
    c = PrefetchController(f, predictor=pp, budget=4, speculate=True)
    n = c.speculate_next(1, [0, 1], lambda l, e: int(e) % N)
    check("speculation issues fetches when explicitly enabled", n == 4, str(n))
    check("...and the overhead is REPORTED rather than hidden",
          c.report()["speculative"] == 4, str(c.report()))

# a full prefetch queue must not stall the engine
with ParallelFetcher(STORE, threads=2) as f:
    pp = CoOccurrencePredictor(E).fit(idx, upto=T // 2)
    c = PrefetchController(f, predictor=pp, budget=4, speculate=True)
    f.prefetch(list(range(N)), max_pending=N + 8)
    n = c.speculate_next(1, [0, 1], lambda l, e: int(e) % N)
    check("a full prefetch queue is skipped, not raised into the caller", isinstance(n, int))

print("\n" + "=" * 78)
print("PREFETCHING MUST STAY OFF: WITH THE PAGE CACHE ON IT MAKES THE MODEL SLOWER")
print("=" * 78)
import inspect as _i
from bigrig_engine import stream as _stream
from bigrig_engine.session import Session as _S
_spec = _i.getsource(_stream.StreamingSwitchGLU._speculate)
# Worth 1.06x while expert reads bypassed the OS page cache. Once they were allowed into it the
# sign flipped: 8.05 / 7.86 tok/s off against 5.82 / 5.59 on, two interleaved A/B pairs. It still
# doubles the share of reads already in hand -- and that is now the problem, because a guess is
# right about 65% of the time and every wrong one pulls a page in by pushing a useful one out.
check("prefetch is off unless asked for",
      _i.signature(_S.__init__).parameters["prefetch_width"].default == 0)
check("...and the measurement that says to keep it off is written down",
      "5.82" in _spec and "8.05" in _spec)
check("...naming the mechanism, not just the number", "pushing a useful one out" in _spec)

os.remove(TMP.name)
print("\n" + "=" * 78)
print(f"{'ALL TESTS PASSED' if not FAIL else str(len(FAIL)) + ' FAILURES: ' + ', '.join(FAIL)}")
print("=" * 78)
sys.exit(1 if FAIL else 0)
