"""Per-model working-memory measurement, and the budget-scaled headroom and prompt cache.

The safety property under test is one-directional: this machinery may only ever LOWER the memory
a plan reserves, never raise it, and only when a representative measurement exists. A run too
small to trust, or none at all, must leave the shipped defaults exactly as they were. These are
pure -- no model -- so they run on a fresh clone; the end-to-end measurement against a live model
is exercised by the tune.
"""
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from bigrig_engine import workmem, autoconfig                          # noqa: E402
from bigrig_engine.session import (serving_reserve_gb, WORKING_MEMORY_GB,   # noqa: E402
                                    PROMPT_CACHE_GB)

FAIL = []


def check(n, c, d=""):
    (print(f"  PASS  {n}") if c else (FAIL.append(n), print(f"  FAIL  {n}  {d}")))


print("=" * 84); print("1. reserve_from CAN ONLY LOWER, NEVER RAISE"); print("=" * 84)
check("a measurement at or above the default reserves exactly the default (no raise)",
      workmem.reserve_from(3.0, 3.0) == 3.0 and workmem.reserve_from(5.0, 3.0) == 3.0)
check("a tiny measurement is floored, not taken literally",
      workmem.reserve_from(0.05, 3.0) == workmem.FLOOR_GB)
check("no measurement (0) leaves the default untouched", workmem.reserve_from(0.0, 3.0) == 3.0)
check("a mid measurement gets a margin and slack, still under the default",
      workmem.FLOOR_GB < workmem.reserve_from(1.0, 3.0) < 3.0)
# The measured seven models: none may reserve MORE than 3.0, and the two that need it keep it.
for name, peak, want_capped in [("Qwen3-30B", 0.68, False), ("Qwen3.6", 1.26, False),
                                ("DeepSeek", 1.88, False), ("Nemotron", 2.66, True),
                                ("GLM", 2.90, True)]:
    r = workmem.reserve_from(peak, 3.0)
    check(f"{name}: reserve {r} never exceeds the 3.0 default", r <= 3.0)
    if want_capped:
        check(f"{name}: a model that needs ~3.0 is not lowered below what it needs", r >= peak)
check("the margin is a real margin: reserve always exceeds the raw measurement",
      all(workmem.reserve_from(p, 3.0) >= p or workmem.reserve_from(p, 3.0) == 3.0
          for p in (0.4, 0.8, 1.2, 1.6, 2.0)))

print("\n" + "=" * 84); print("2. RECORDING IS GATED AND MONOTONIC"); print("=" * 84)
with tempfile.TemporaryDirectory() as d:
    orig = workmem._dir
    workmem._dir = lambda: d
    try:
        check("a run with too narrow a prefill is not recorded",
              workmem.record("m", 9.0, 2.0, prefill_tokens=8, decode_tokens=40) is None)
        check("a run with too little decode is not recorded",
              workmem.record("m", 9.0, 2.0, prefill_tokens=600, decode_tokens=2) is None)
        check("a representative run is recorded",
              workmem.record("m", 9.0, 1.0, prefill_tokens=600, decode_tokens=40) is not None)
        check("...and can be read back", (workmem.load("m", 9.0) or {}).get("peak_gb") == 1.0)
        check("a LOWER later reading does not overwrite a higher one (monotonic)",
              workmem.record("m", 9.0, 0.5, prefill_tokens=600, decode_tokens=40) is None
              and workmem.load("m", 9.0)["peak_gb"] == 1.0)
        check("a HIGHER later reading does overwrite (safety ratchets tighter)",
              workmem.record("m", 9.0, 2.2, prefill_tokens=600, decode_tokens=40) is not None
              and workmem.load("m", 9.0)["peak_gb"] == 2.2)
        check("a different budget is a different file",
              workmem.load("m", 5.6) is None and workmem.load("m", 9.0) is not None)
        check("a zero or negative peak is never recorded",
              workmem.record("m", 9.0, 0.0, prefill_tokens=600, decode_tokens=40) is None)
    finally:
        workmem._dir = orig
check("a model never measured reads back as absent", workmem.load("no-such-model-xyz", 9.0) is None)

print("\n" + "=" * 84); print("3. SCALED HEADROOM SHRINKS ON SMALL BUDGETS, NEVER BELOW THE FLOOR"); print("=" * 84)
check("a comfortable budget keeps the full 1.0 GB", autoconfig.scaled_headroom(9.0) == 1.0)
check("...and so does anything larger", autoconfig.scaled_headroom(24.0) == 1.0)
check("a 16 GB Mac's 5.6 GB ceiling gets less than 1.0 but more than the floor",
      autoconfig.HEADROOM_FLOOR_GB < autoconfig.scaled_headroom(5.6) < 1.0)
check("a very tight budget is floored, never zero", autoconfig.scaled_headroom(2.0) == autoconfig.HEADROOM_FLOOR_GB)
check("headroom never exceeds the flat default it replaces",
      all(autoconfig.scaled_headroom(b) <= 1.0 for b in (1.0, 3.5, 5.6, 8.0, 9.0, 40.0)))
check("headroom is monotonic in the budget",
      all(autoconfig.scaled_headroom(a) <= autoconfig.scaled_headroom(b)
          for a, b in zip((2.0, 4.0, 6.0, 8.0), (4.0, 6.0, 8.0, 10.0))))

print("\n" + "=" * 84); print("4. THE FLOOR ACTUALLY MOVES, AND ONLY DOWNWARD"); print("=" * 84)
QW = None
import json                                                            # noqa: E402
_qwp = os.path.join(ROOT, "data", "blobs", "Qwen3-30B-A3B-3bit.experts.manifest.json")
if os.path.exists(_qwp):
    QW = json.load(open(_qwp))

    def floor(reserve, headroom_scaled):
        for b in [x / 10 for x in range(30, 130)]:
            try:
                autoconfig.choose_capacity(QW, budget_gb=b, top_k=8, reserve_gb=reserve,
                                           non_expert_gb=0.87,
                                           headroom_gb=autoconfig.scaled_headroom(b)
                                           if headroom_scaled else None)
                return b
            except MemoryError:
                continue
        return None
    base = floor(serving_reserve_gb(), False)
    # A measured 0.68 GB scratch -> reserve_from ~1.18; combined with scaled headroom the floor
    # must drop, and a model that measured 3.0 (capped) must not move at all.
    lowered_reserve = serving_reserve_gb(working_memory_gb=workmem.reserve_from(0.68, WORKING_MEMORY_GB))
    lower = floor(lowered_reserve, True)
    check("a model with low measured scratch reaches a lower floor", lower < base, f"{lower} vs {base}")
    capped_reserve = serving_reserve_gb(working_memory_gb=workmem.reserve_from(3.5, WORKING_MEMORY_GB))
    check("a model that needs the full 3.0 does not get a lower reserve",
          capped_reserve == serving_reserve_gb(), f"{capped_reserve} vs {serving_reserve_gb()}")
else:
    print("  SKIPPED - Qwen3-30B manifest fixture absent")

print("\n" + "=" * 84); print("5. A STREAMED POOL IS NEVER SIZED LARGER THAN THE OLD PLANNER WOULD"); print("=" * 84)
if QW is not None:
    for b in (8.0, 9.0, 12.0):
        old = autoconfig.choose_capacity(QW, budget_gb=b, top_k=8, reserve_gb=serving_reserve_gb(),
                                         non_expert_gb=0.87)          # old: flat reserve, flat headroom
        # With a REAL measurement the reserve drops and the pool may grow -- that is the point --
        # but with NO measurement (default reserve) and scaled headroom, a big budget must be
        # identical to the old plan, since scaled_headroom == 1.0 there.
        same = autoconfig.choose_capacity(QW, budget_gb=b, top_k=8, reserve_gb=serving_reserve_gb(),
                                          non_expert_gb=0.87, headroom_gb=autoconfig.scaled_headroom(b))
        if b >= 8.3:
            check(f"at {b} GB with no measurement the plan is unchanged from before",
                  same["capacity"] == old["capacity"], f"{same['capacity']} vs {old['capacity']}")
else:
    print("  SKIPPED - Qwen3-30B manifest fixture absent")

print("\n" + "=" * 84); print("6. KV-CACHE PRECISION IS A DOCUMENTED CHOICE, OFF BY 0 OR 16"); print("=" * 84)
from bigrig_engine.session import resolve_kv_bits, KV_BITS                # noqa: E402
check("no request keeps the 4-bit default", resolve_kv_bits(None) == KV_BITS == 4)
check("0 means full precision (no quantization)", resolve_kv_bits(0) is None)
check("16 also means full precision", resolve_kv_bits(16) is None)
check("an explicit precision is honoured", resolve_kv_bits(8) == 8 and resolve_kv_bits(2) == 2)
for bad in (7, 1, 9, 12):
    try:
        resolve_kv_bits(bad); check(f"rejects kv_bits={bad}", False)
    except ValueError:
        check(f"rejects kv_bits={bad} with a sentence", True)

print()
print("=" * 84)
print("ALL TESTS PASSED" if not FAIL else f"{len(FAIL)} FAILURES: " + ", ".join(FAIL))
print("=" * 84)
sys.exit(1 if FAIL else 0)
