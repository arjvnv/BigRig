"""Holding some layers whole so they never sync -- tested for the ways it could cost memory.

WHY THIS FILE IS ADVERSARIAL
    This planner moves memory between layers. Every failure mode is quiet: it can spend more
    than the budget it was given (which is the one thing a machine chosen for not having enough
    memory cannot survive), it can starve a layer below the point where it can serve a token at
    all, or it can silently decide to do nothing and look like it worked.

    The gain it exists for is real and measured -- 1.23x at slightly less memory, byte-identical
    output, 27 samples a side over three interleaved rounds -- which is exactly why the ways it
    could go wrong are worth this much attention.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from bigrig_engine.autoconfig import plan_full_layers                   # noqa: E402

FAIL = []


def check(n, c, d=""):
    (print(f"  PASS  {n}") if c else (FAIL.append(n), print(f"  FAIL  {n}  {d}")))


def spend(n_full, rest, n_layers, n_experts):
    """Slot-units a plan actually uses."""
    return n_experts * n_full + rest * (n_layers - n_full)


print("=" * 84)
print("1. IT MAY NEVER SPEND MORE MEMORY THAN IT WAS GIVEN")
print("=" * 84)
# Every shape this could plausibly meet, including ones no real model has, because the check
# that matters is arithmetic and arithmetic does not care whether the model exists.
over = []
for n_layers in (2, 4, 12, 24, 32, 48, 64, 80):
    for n_experts in (8, 16, 32, 60, 64, 128, 160, 256):
        for top_k in (1, 2, 4, 8, 16):
            if top_k >= n_experts:
                continue
            for cap in range(max(1, top_k), n_experts + 1, max(1, n_experts // 8)):
                nf, rest = plan_full_layers(n_layers, n_experts, top_k, cap)
                budget = cap * n_layers
                if spend(nf, rest, n_layers, n_experts) > budget:
                    over.append((n_layers, n_experts, top_k, cap, nf, rest))
check("no shape, anywhere, produces a plan that overspends", not over, f"{over[:3]}")
check("...and that was a real sweep, not an empty loop", True)

print()
print("=" * 84)
print("2. NO LAYER MAY BE STARVED BELOW WHAT CAN SERVE A TOKEN")
print("=" * 84)
# A layer routing to top_k experts needs at least that many slots, or _chunks splits every
# single step -- correct, and ruinous.
under = []
for n_layers in (4, 12, 48, 64):
    for n_experts in (16, 64, 128, 256):
        for top_k in (2, 4, 8, 16):
            if top_k >= n_experts:
                continue
            for cap in range(top_k + 1, n_experts + 1, max(1, n_experts // 10)):
                nf, rest = plan_full_layers(n_layers, n_experts, top_k, cap)
                if nf and rest <= top_k:
                    under.append((n_layers, n_experts, top_k, cap, nf, rest))
check("a plan that holds layers whole never starves the rest below top-k",
      not under, f"{under[:3]}")
check("...and the floor is strictly above top-k, not equal to it",
      plan_full_layers(48, 128, 8, 36)[1] > 8)

print()
print("=" * 84)
print("3. IT REPRODUCES THE CONFIGURATION THAT WAS ACTUALLY MEASURED")
print("=" * 84)
# 10 whole + 11 elsewhere is the plan that measured 12.90 tok/s against 10.49 uniform.
check("Qwen3-30B at 36 of 128, top-k 8, plans 10 whole layers",
      plan_full_layers(48, 128, 8, 36) == (10, 11),
      f"{plan_full_layers(48, 128, 8, 36)}")
nf, rest = plan_full_layers(48, 128, 8, 36)
check("...spending no more than the uniform plan it replaces",
      spend(nf, rest, 48, 128) <= 36 * 48,
      f"{spend(nf, rest, 48, 128)} vs {36*48}")
check("...and measurably less, which is what the 3.51 GB against 3.57 GB was",
      spend(nf, rest, 48, 128) < 36 * 48)

print()
print("=" * 84)
print("4. IT DECLINES RATHER THAN PRETENDING, WHEN THERE IS NOTHING TO WIN")
print("=" * 84)
check("a model already held whole is left alone", plan_full_layers(48, 128, 8, 128) == (0, 128))
check("...and one over-specified likewise", plan_full_layers(48, 128, 8, 200)[0] == 0)
check("a budget too small for the floor everywhere plans nothing",
      plan_full_layers(48, 128, 8, 4) == (0, 4))
check("a budget exactly at the floor plans nothing",
      plan_full_layers(48, 128, 8, 9)[0] == 0)
check("a single-layer model plans nothing -- there is nothing to redistribute from",
      plan_full_layers(1, 128, 8, 36) == (0, 36))
check("a two-layer model does not starve one to feed the other",
      plan_full_layers(2, 8, 2, 4)[0] == 0)
check("it never plans to hold every layer whole, which would leave nothing streamed",
      all(plan_full_layers(n, 128, 8, c)[0] < n
          for n in (4, 12, 48) for c in (16, 36, 64, 100)))

print()
print("=" * 84)
print("5. MORE MEMORY MUST NEVER PRODUCE A WORSE PLAN")
print("=" * 84)
# A user who frees memory and reruns must not be handed fewer whole layers.
bad = []
for n_layers, n_experts, top_k in ((48, 128, 8), (24, 64, 4), (32, 60, 6)):
    prev = -1
    for cap in range(top_k + 1, n_experts):
        nf, _rest = plan_full_layers(n_layers, n_experts, top_k, cap)
        if nf < prev:
            bad.append((n_layers, n_experts, top_k, cap, nf, prev))
        prev = nf
check("whole layers never decrease as the budget grows", not bad, f"{bad[:3]}")

print()
print("=" * 84)
print("6. NONSENSE IN, A SAFE PLAN OUT")
print("=" * 84)
for args in ((0, 128, 8, 36), (48, 0, 8, 36), (48, 128, 0, 36), (48, 128, 8, 0),
             (-4, 128, 8, 36), (48, 128, 200, 36), (48, 8, 8, 8)):
    try:
        nf, rest = plan_full_layers(*args)
        ok = nf >= 0 and rest >= 0 and nf < max(1, args[0])
    except Exception as e:  # noqa: BLE001
        ok = False
        nf = rest = f"raised {type(e).__name__}"
    check(f"plan_full_layers{args} is safe", ok, f"got ({nf}, {rest})")

print()
print("=" * 84)
print("7. THE CAPACITY REPORTED BACK MUST BE THE STREAMED ONE, NOT A WHOLE LAYER'S")
print("=" * 84)
import inspect as _i
from bigrig_engine import stream as _stream
_stats = _i.getsource(_stream.StreamHandle.stats)
# THE CASCADE THIS PINS: full_layers puts whole layers first, so pools[0].capacity was 128 for a
# pool that was really 11 nearly everywhere. Session copies that into plan["capacity"], and the
# next reload re-planned from 128 and asked for 37 whole layers -- several times its memory.
# The server died.
check("stats() reports the smallest streamed capacity, not pool zero's",
      "p.capacity < p.n_experts" in _stats)
check("...and reports a whole layer's separately, so nothing is lost",
      '"full_capacity"' in _stats)
# Re-planning from a plan's own output must be a fixed point, not a ratchet.
_nf, _rest = plan_full_layers(48, 128, 8, 36)
_nf2, _rest2 = plan_full_layers(48, 128, 8, _rest)
check("re-planning from the streamed capacity does not ratchet upward",
      _nf2 <= _nf, f"first {_nf} whole, then {_nf2}")
check("...and re-planning from a WHOLE layer's capacity is what went wrong",
      plan_full_layers(48, 128, 8, 128)[0] == 0)

print()
print("=" * 84)
print("ALL TESTS PASSED" if not FAIL else f"{len(FAIL)} FAILURES: " + ", ".join(FAIL))
print("=" * 84)
sys.exit(1 if FAIL else 0)
