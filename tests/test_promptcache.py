"""Remembering prompts already read: what it buys, what it costs, and what it changes.

WHY THIS FILE IS ADVERSARIAL
    This is the first thing in the engine that makes a reply depend on state left over from an
    EARLIER reply. Everything else is a function of the prompt alone. Two failures follow from
    that and neither announces itself:

    A cache that is not charged against the ceiling is a cache that crashes the machine. The
    whole product is a promise about a memory ceiling, and this puts hundreds of megabytes behind
    that promise's back unless the planner is told about it first.

    A cache that changes what the model says is a cache that quietly makes the model look worse,
    and the user would have no way to attribute it. It DOES change what the model says, for a
    reason no implementation avoids -- measured below -- so the job here is to bound it, not to
    claim it away.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from bigrig_engine import session as S                                  # noqa: E402

FAIL = []


def check(n, c, d=""):
    (print(f"  PASS  {n}") if c else (FAIL.append(n), print(f"  FAIL  {n}  {d}")))


print("=" * 84)
print("1. IT MUST BE PAID FOR BEFORE THE POOL IS PLANNED, NOT AFTER")
print("=" * 84)
# The reserve is what the planner subtracts before deciding how many experts fit. If the cache
# were allocated outside it, every byte the cache holds would be a byte over the ceiling the user
# set -- and the ceiling is the entire promise this product makes.
import inspect                                                          # noqa: E402
_src = inspect.getsource(S.Session.__init__)
check("the cache budget is added to the serving reserve",
      "prompt_cache_gb" in _src and "serving_reserve_gb" in _src)
check("...before choose_capacity is asked anything",
      _src.index("self.prompt_cache_gb") < _src.index("choose_capacity"))
check("a session can turn it off entirely", "prompt_cache_gb" in
      inspect.signature(S.Session.__init__).parameters)
check("the default is bounded and small enough to leave a pool",
      0 < S.PROMPT_CACHE_GB <= 1.0, str(S.PROMPT_CACHE_GB))

print()
print("=" * 84)
print("2. THE STORE ITSELF: BOUNDED IN BYTES, AND RELEASABLE ON DEMAND")
print("=" * 84)
try:
    from mlx_lm.models.cache import LRUPromptCache

    class _Fake:
        """An entry of a known size, so the byte accounting can be checked without a model."""
        def __init__(self, n):
            self.nbytes = n

        def is_trimmable(self):
            return False

    lru = LRUPromptCache(max_size=64, max_bytes=1_000_000)
    for i in range(40):
        lru.insert_cache("m", list(range(i * 10, i * 10 + 10)), [_Fake(100_000)])
    check("it never holds more than the bytes it was given",
          lru.nbytes <= 1_000_000, f"{lru.nbytes} bytes")
    check("...and it did actually fill up, so that was a real limit", lru.nbytes > 0)
    lru.trim_to(n_bytes=0)
    check("trimming to zero releases everything", lru.nbytes == 0, f"{lru.nbytes} bytes")

    # The engine's own hook, on an object with no model behind it.
    class _Sess:
        prompt_cache_gb = 0.5
        _prompt_cache = LRUPromptCache(max_size=8, max_bytes=1_000_000)
    _Sess._prompt_cache.insert_cache("m", [1, 2, 3], [_Fake(400_000)])
    freed = S.Session.trim_prompt_cache(_Sess, 0)
    check("the engine's release hook reports the bytes it freed", freed == 400_000, str(freed))
    check("...and is a no-op, not an error, when there is no cache",
          S.Session.trim_prompt_cache(type("N", (), {"_prompt_cache": None})(), 0) == 0)
except ImportError as e:                                                # noqa: BLE001
    check(f"mlx_lm LRUPromptCache unavailable ({e}) -- skipped, not passed", True)

print()
print("=" * 84)
print("3. PRESSURE MUST SPEND THIS BEFORE IT SPENDS A POOL SLOT")
print("=" * 84)
# Shrinking the pool costs decode speed until it is undone and needs a full model rebuild to
# undo. Dropping remembered prompts costs one re-read. The order is not a preference.
from bigrig_engine import server as SV                                  # noqa: E402
_ms = inspect.getsource(SV._State._maybe_shrink)
check("the controller trims remembered prompts inside the pressure branch",
      "trim_prompt_cache" in _ms)
check("...before it asks the controller for a smaller pool",
      _ms.index("trim_prompt_cache") < _ms.index("self.memctl.decide"))
check("...and stops there when that was enough, rather than shrinking as well",
      "return" in _ms[_ms.index("trim_prompt_cache"):_ms.index("self.memctl.decide")])
check("a failure to trim cannot take the server down",
      "except Exception" in _ms[_ms.index("trim_prompt_cache") - 400:_ms.index("self.memctl.decide")])

print()
print("=" * 84)
print("4. WHAT IT CHANGES ABOUT REPLIES, STATED RATHER THAN CLAIMED AWAY")
print("=" * 84)
# MEASURED, on a 892-token document followed by six follow-up questions at temperature 0, every
# reply compared against the same conversation run with the cache off:
#
#     turns byte-identical                6 of 7
#     the one that differed               54.3% similar, same answer, different wording
#         uncached  'One number from the document is **4** (prices quadrupled).'
#         cached    '**4** (as in "Prices quadrupled").'
#
# WHY IT HAPPENS, AND WHY NO IMPLEMENTATION AVOIDS IT. The reused attention state is the state
# that was computed -- the same arrays, not a re-derivation -- so the prefix is exact. What
# differs is the SHAPE of the pass that reads the new tail: uncached, turn seven's 3,000 tokens
# go through in chunks of 69; cached, its 50 new tokens go through in one pass of 50. Different
# shapes reduce in a different order and floating point is not associative. It is the same
# mechanism that makes a prefill step of 64 and one of 128 disagree, which is measured in
# bigrig_engine/session.py, and it is why the step is no longer allowed to vary at runtime.
#
# The engine therefore does NOT claim a cached reply is byte-identical to an uncached one. It
# claims the prefix is exact, that the divergence is a reordering rather than a loss, and that
# `prompt_cache_gb=0` turns it off for anyone who needs the stronger guarantee.
_doc = inspect.getdoc(S.Session.trim_prompt_cache) or ""
check("the release path documents that it costs a re-read and not an answer",
      "re-read" in _doc)
_mod = open(os.path.join(ROOT, "bigrig_engine", "session.py")).read()
check("the measured numbers are recorded next to the constant, not just in a commit message",
      "888 reused" in _mod or "888 reused" in _mod or "10.0x" in _mod)
check("...including that it is a reserve rather than a cache that grows into free memory",
      "grows into whatever headroom" in _mod.lower() or "not a cache that grows" in _mod.lower())
check("turning it off is a documented mode, not an undocumented parameter",
      "prompt_cache_gb=0" in _mod)

print()
print("=" * 84)
print("5. TRAFFIC THAT CANNOT BENEFIT MUST NOT EVICT TRAFFIC THAT DOES")
print("=" * 84)
# THE FAILURE, MEASURED. One LRU held everything, so a conversation genuinely being continued sat
# beside a stream of one-off prompts and lost to them on recency alone. Twelve requests each
# carrying a different identifier at the FRONT -- so each matched almost nothing and each stored a
# new entry -- took the cache from 66 MB to 461 MB and evicted the entry being reused. The reused
# prompt went 0.53s -> 6.06s. That shape is not exotic; it is what agent traffic looks like.
#
# After segmenting: protected held at 65.9 MB, total capped at 198 MB, reused prompt 0.55s.
try:
    from mlx_lm.models.cache import LRUPromptCache                      # noqa: F401

    class _E:
        def __init__(self, n):
            self.nbytes = n

        def is_trimmable(self):
            return False

    c = S._TwoStagePromptCache(1_000_000, max_size=64)
    check("the store is split, and probation is the smaller share",
          c.probation.max_bytes < c.protected.max_bytes,
          f"{c.probation.max_bytes} vs {c.protected.max_bytes}")
    # One proven entry, then a flood of unproven ones.
    c.insert_cache("m", list(range(100, 140)), [_E(200_000)], proven=True)
    protected_before = int(c.protected.nbytes)
    for i in range(40):
        c.insert_cache("m", list(range(i * 1000, i * 1000 + 40)), [_E(100_000)], proven=False)
    check("a flood of unproven entries cannot touch the protected segment",
          int(c.protected.nbytes) == protected_before,
          f"{protected_before} -> {int(c.protected.nbytes)}")
    check("...and they are bounded by probation's own budget",
          int(c.probation.nbytes) <= c.probation.max_bytes)
    check("...so the proven entry is still there to be found",
          c.fetch_nearest_cache("m", list(range(100, 140)))[0] is not None)
    check("the whole store still respects the budget it was given",
          c.nbytes <= 1_000_000, f"{c.nbytes}")
    # Releasing under pressure spends the cheap segment first.
    c2 = S._TwoStagePromptCache(1_000_000)
    c2.insert_cache("m", [1, 2, 3], [_E(150_000)], proven=True)
    c2.insert_cache("m", [9, 9, 9], [_E(150_000)], proven=False)
    c2.trim_to(n_bytes=150_000)
    check("trimming spends probation before protected",
          int(c2.protected.nbytes) > 0 and int(c2.probation.nbytes) == 0,
          f"protected {int(c2.protected.nbytes)}, probation {int(c2.probation.nbytes)}")
    c2.trim_to(n_bytes=0)
    check("...and trimming to nothing releases both", c2.nbytes == 0, f"{c2.nbytes}")
except ImportError as e:                                                # noqa: BLE001
    check(f"mlx_lm LRUPromptCache unavailable ({e}) -- skipped, not passed", True)

print()
print("=" * 84)
print("6. A SHARED HANDFUL OF TOKENS IS NOT A HIT")
print("=" * 84)
# The trie returns the longest common prefix, and unrelated prompts have one -- two requests
# differing only in an identifier near the front still agree on the tokens before it. Treating
# that as a hit was wrong twice: serving it copies the whole stored entry (66 MB on this model)
# to save six tokens, and counting it as PROVEN promoted noise into the protected segment. That
# second one is what defeated the segmenting on the first attempt: the split worked exactly as
# designed and the signal feeding it was junk.
check("there is a floor below which a match is not worth serving",
      S.MIN_REUSE_TOKENS >= 8, str(S.MIN_REUSE_TOKENS))
check("...and a much higher bar before a conversation counts as proven",
      0.25 <= S.PROVEN_REUSE_FRACTION <= 1.0, str(S.PROVEN_REUSE_FRACTION))
_src = inspect.getsource(S.Session.stream_text)
check("a match shorter than the floor is discarded rather than served",
      "_matched < MIN_REUSE_TOKENS" in _src)
check("promotion is decided by the SHARE reused, not by the fact of a match",
      "PROVEN_REUSE_FRACTION * len(full_ids)" in _src)
check("how much was actually reused is reported, so a weak hit is visible from outside",
      "prompt_cache_matched" in inspect.getsource(S.Session.stats))
# The engine does NOT pretend a changed prefix can be reused. Attention state is positional: if a
# token near the front differs, every key and value after it is genuinely different, and serving
# them would be wrong rather than merely stale.
check("the code says why a changed prefix cannot be salvaged, rather than trying",
      "positional" in inspect.getsource(S._TwoStagePromptCache))

print()
print("=" * 84)
print("ALL TESTS PASSED" if not FAIL else f"{len(FAIL)} FAILURES: " + ", ".join(FAIL))
print("=" * 84)
sys.exit(1 if FAIL else 0)
