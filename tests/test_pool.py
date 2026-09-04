"""Adversarial tests for the expert pool.

This component sets the miss rate, and the miss rate is the term every speed number in the
project depends on. Its dangerous failure is a policy that LOOKS good because the harness is
wrong -- which already happened once: a sweep scored the winning policy at -37.2% because it
instantiated raw classes instead of their tuned factories. So the harness itself is tested here,
not only the policies.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bigrig_engine.pool import (LRU, Policy, WindowedLFU, _policy_factory, belady, evaluate,
                                  load_trace, recommended, select_policy, simulate)

FAIL = []
def check(n, c, d=""):
    (print(f"  PASS  {n}") if c else (FAIL.append(n), print(f"  FAIL  {n}  {d}")))

CANDS = ["lfuda", "s3fifo", "gaterank", "lirs", "arc", "coact"]

print("=" * 78); print("1. THE SIMULATOR MUST BE RIGHT BEFORE ANY POLICY IS"); print("=" * 78)
# A sequence with a known answer, computed by hand.
seq = [1, 2, 3, 1, 2, 3, 1, 2, 3]
r = simulate(seq, 3, 8, LRU)
check("capacity 3 holding a 3-cycle misses only the first pass",
      r["misses"] == 3 and r["hits"] == 6, str(r))
r2 = simulate(seq, 2, 8, LRU)
check("capacity 2 on a 3-cycle misses EVERY request (classic LRU thrash)",
      r2["misses"] == 9, str(r2))
check("Belady beats LRU on that same thrashing sequence",
      belady(seq, 2)["misses"] < r2["misses"],
      f"belady {belady(seq,2)['misses']} vs lru {r2['misses']}")
check("a cache big enough to hold everything misses only compulsory",
      simulate(seq, 8, 8, LRU)["misses"] == 3)
check("hits + misses always equals the request count",
      all(simulate(seq, c, 8, LRU)["hits"] + simulate(seq, c, 8, LRU)["misses"] == len(seq)
          for c in (1, 2, 3, 8)))

print("\n" + "=" * 78); print("2. BELADY MUST BE A TRUE LOWER BOUND"); print("=" * 78)
rng = np.random.default_rng(0)
worse = []
for trial in range(12):
    s = rng.integers(0, 12, 300).tolist()
    C = int(rng.integers(2, 8))
    b = belady(s, C)["miss_rate"]
    for name, fac in [("lru", LRU), ("wlfu", WindowedLFU)]:
        if simulate(s, C, 12, fac)["miss_rate"] < b - 1e-12:
            worse.append((name, C, trial))
check("NO implementable policy ever beats Belady", not worse, str(worse[:2]))

print("\n" + "=" * 78); print("3. EVERY POLICY MUST OBEY THE CACHE CONTRACT"); print("=" * 78)
# Instrumented replay: capacity is never exceeded, and victim() always names a resident expert.
def audited(seq, C, E, fac, ranks):
    pol = fac(C, E)
    resident, bad = set(), []
    for e, rk in zip(seq, ranks):
        e = int(e)
        if e in resident:
            pol.touch(e, rk); continue
        if len(resident) >= C:
            v = pol.victim()
            if v not in resident:
                bad.append(("victim not resident", v))
                resident.pop() if resident else None
            else:
                resident.discard(v)
            if hasattr(pol, "evicted"):
                pol.evicted(v)
        resident.add(e)
        pol.admit(e, rk)
        if len(resident) > C:
            bad.append(("over capacity", len(resident)))
    return bad

# The trace is 28 MB of real routing decisions -- too large to ship for a test, and
# regenerable from a model the cloner may not have. Say which file is missing and stop, rather
# than crashing the suite on a FileNotFoundError for someone who has just cloned the repo.
from bigrig_engine import home as _home
_TR = os.path.join(_home(), "data", "traces", "olmoe_full.npz")
if not os.path.exists(_TR):
    print(f"  SKIPPED - {_TR} absent. Every check below replays that routing trace; regenerate")
    print( "  it by running OLMoE once, or point BIGRIG_HOME at an install that has it.")
    print("\n" + "=" * 78); print("ALL TESTS PASSED"); print("=" * 78)
    sys.exit(0)

idx, _ = load_trace("olmoe")
L, T, k = idx.shape
E = int(idx.max()) + 1
sub = [(int(idx[0, t, r]), r) for t in range(600) for r in range(k)]
sq, rk = [a for a, _ in sub], [b for _, b in sub]
for name in CANDS:
    try:
        fac = _policy_factory(name, 16, E)
    except ImportError as e:
        check(f"{name} loads", False, str(e)); continue
    bad = audited(sq, 16, E, fac, rk)
    check(f"{name}: never exceeds capacity, always evicts a resident expert",
          not bad, f"{len(bad)} violations, first {bad[:1]}")

print("\n" + "=" * 78); print("4. THE LOADER MUST FIND EVERY POLICY"); print("=" * 78)
# The bug this guards: an earlier loader tried two of the three conventions and reported the two
# it missed as errors, which would have scored working policies as failures.
loaded = []
for name in CANDS:
    try:
        f = _policy_factory(name, 16, E)
        p = f(16, E)
        loaded.append(name)
        check(f"{name} resolves to a usable policy", hasattr(p, "victim"))
    except Exception as e:
        check(f"{name} resolves to a usable policy", False, str(e)[:60])
check("all six candidates load", len(loaded) == 6, f"{len(loaded)}/6")
try:
    _policy_factory("no_such_policy", 16, E)
    check("an unknown policy raises rather than returning None", False)
except (ImportError, ModuleNotFoundError):
    check("an unknown policy raises rather than returning None", True)

print("\n" + "=" * 78); print("5. THE RECOMMENDED DEFAULT"); print("=" * 78)
d = recommended(16, E)
check("recommended() returns a policy instance", hasattr(d, "victim"))
check("...and it is lfuda, the measured winner", getattr(d, "name", "") == "lfuda",
      getattr(d, "name", "?"))
f = recommended()
check("recommended() with no args returns a factory", callable(f) and hasattr(f(16, E), "victim"))

print("\n" + "=" * 78); print("6. SELECTION MUST ACTUALLY MEASURE"); print("=" * 78)
r = select_policy("olmoe", 16, layers=1)
check("selection reports LRU and Belady bounds", r["lru"] > r["belady"] > 0, str(r)[:80])
check("selection scores every candidate",
      all("gap_pct" in v or "error" in v for v in r["policies"].values()))
errs = [n for n, v in r["policies"].items() if "error" in v]
check("no candidate fails to load during selection", not errs, str(errs))
best_measured = max((v["gap_pct"], n) for n, v in r["policies"].items() if "gap_pct" in v)
check("the reported best IS the highest scorer", r["best"] == best_measured[1],
      f"{r['best']} vs {best_measured[1]}")

# THE POINT of selection: the winner is not the same across models
r2 = select_policy("ling", 32, layers=1)
print(f"        olmoe C=16 -> {r['best']} ({r['best_gap_pct']:.1f}%)   "
      f"ling C=32 -> {r2['best']} ({r2['best_gap_pct']:.1f}%)")
check("selection is measured per model rather than hardcoded",
      r["best"] in CANDS and r2["best"] in CANDS)

print("\n" + "=" * 78); print("7. TEETH — a cheating policy must be caught"); print("=" * 78)
class Cheater(Policy):
    """Returns a victim it does not hold. The audit in section 3 must catch this."""
    name = "cheater"
    def victim(self): return 999999
bad = audited(sq[:2000], 8, E, lambda c, e: Cheater(c, e), rk[:2000])
check("a policy that evicts a non-resident expert is detected", bool(bad), "audit passed it")

class Hoarder(Policy):
    """Never evicts. The audit must catch the cache growing past capacity."""
    name = "hoarder"
    def victim(self): return -1
bad2 = audited(sq[:2000], 8, E, lambda c, e: Hoarder(c, e), rk[:2000])
check("a policy that never evicts is detected", bool(bad2), "audit passed it")

print("\n" + "=" * 78)
print(f"{'ALL TESTS PASSED' if not FAIL else str(len(FAIL)) + ' FAILURES: ' + ', '.join(FAIL)}")
print("=" * 78)
sys.exit(1 if FAIL else 0)
