"""How long a reply the engine will allow, and the measurement that decides it.

WHY THIS FILE EXISTS
    The reply ceiling is one subtraction: budget, less what the model holds, less what one step
    needs, divided by the cost of a token of KV cache. Every term in it is a measurement, and one
    of them was being taken at the wrong moment -- so the subtraction went negative and every
    reply on this machine was cut at 256 tokens against room for eighteen thousand.

    It is a bad failure to catch by eye. Nothing errors, nothing is logged, and the model answers
    perfectly well for 256 tokens and then stops mid-sentence with `finish_reason: length`, which
    reads as a model that rambles rather than an engine that is wrong.
"""
import inspect
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from bigrig_engine import session as S                                  # noqa: E402

FAIL = []


def check(n, c, d=""):
    (print(f"  PASS  {n}") if c else (FAIL.append(n), print(f"  FAIL  {n}  {d}")))


print("=" * 84)
print("1. THE FOOTPRINT MUST BE READ AFTER THE ALLOCATOR HAS SETTLED, NOT BEFORE")
print("=" * 84)
# THE BUG. Filling the pool rebinds each slot tensor rather than writing into it -- MLX arrays
# are functional -- so warming leaves an abandoned (C, ...) tensor per write, still counted as
# active. On Qwen3-30B-A3B-3bit that was 2.64 GB of nothing, present at exactly the moment the
# footprint was read and gone the moment any real multi-token work began:
#
#     after load                          6.82 GB active
#     mx.clear_cache / gc / synchronize    6.82 GB   -- none of them touch it
#     one SINGLE-token pass                6.82 GB   -- which is why _measure_kv missed it
#     one 8-token pass                     4.18 GB
#
#     footprint 6.82 -> ceiling 9.00 - 6.82 - 3.00 = -0.82 GB -> the 256-token floor
#     footprint 4.18 -> ceiling                            -> 18,514 tokens
#
# Verified by generating against the new ceiling: 2,218 tokens peaked at 4.43 GB of a 9.00 GB
# budget, with active memory rising exactly as the KV arithmetic predicts.
src = inspect.getsource(S.Session.__init__)
check("the session settles the allocator before reading its footprint",
      "_settle" in src and src.index("self._settle()") < src.index("self.footprint_gb ="),
      "the footprint is read before anything forces the allocator to give back the warm fill")

st = inspect.getsource(S.Session._settle)
# A single token is not enough, and that is the whole subtlety: `_measure_kv` already ran one and
# the bug survived it. If this is ever "simplified" back to one token it silently returns.
check("the settling pass is more than one token, which is what makes it work",
      "max(2," in st, "a single-token pass leaves the abandoned tensors in place")
check("...and it is bounded, so a load does not pay for a long pass", "min(8," in st)
check("a model that cannot take the pass still loads",
      "except Exception" in st and "return" in st.split("except Exception")[1][:120])
check("the measurement that justifies it is recorded next to the code",
      "2.64 GB" in st and "256" in st)

print()
print("=" * 84)
print("2. THE CEILING ARITHMETIC ITSELF")
print("=" * 84)
tc = inspect.getsource(S.Session._token_ceiling)


class _S:
    """A session's numbers with no model behind them."""
    def __init__(self, budget=9.0, foot=4.18, work=3.0, kv=98304, ctx=40960, fixed=0,
                 kv_bits=None):
        self.budget_gb, self.footprint_gb, self.working_memory_gb = budget, foot, work
        self.kv_bytes_per_token, self.context_length, self.kv_fixed_bytes = kv, ctx, fixed
        # Compressing the cache changes what a token of reply COSTS, so the ceiling has to know.
        self.kv_bits, self.kv_quant_start = kv_bits, 4096


ceil = S.Session._token_ceiling
n, why = ceil(_S())
check("a model with room reports a ceiling in the thousands, not the floor",
      n > 10_000 and why == "memory", f"{n} ({why})")
check("...and it is the KV arithmetic, not a constant",
      ceil(_S(budget=12.0))[0] > ceil(_S(budget=9.0))[0])
n2, why2 = ceil(_S(budget=6.0))
check("a machine genuinely short of room still reports the floor, not a negative number",
      n2 == S.MIN_REPLY_TOKENS and why2 == "memory", f"{n2} ({why2})")
check("...and never a number below it", ceil(_S(budget=1.0))[0] >= S.MIN_REPLY_TOKENS)
check("the model's own context window is the other limit, and wins when it is smaller",
      ceil(_S(budget=64.0)) == (40960, "context"), str(ceil(_S(budget=64.0))))
check("a fixed per-reply cost is charged once rather than per token",
      ceil(_S(fixed=1_000_000_000))[0] < ceil(_S(fixed=0))[0])
check("a model that reports no KV cost falls back to the context window rather than dividing by zero",
      ceil(_S(kv=0)) == (40960, "context"))

print()
print("=" * 84)
print("3. THE CEILING MUST MOVE WHEN THE THING IT IS DERIVED FROM MOVES")
print("=" * 84)
# The server re-measures the footprint after a reload, for the same reason: `Session.__init__`
# reads active memory while MLX still holds the dropped pool, so a reload that did not re-measure
# shortened every reply that followed it.
srv = open(os.path.join(ROOT, "bigrig_engine", "server.py"), encoding="utf-8").read()
check("a reload re-measures the footprint rather than keeping the old one",
      "footprint_gb = _mx.get_active_memory()" in srv)
# Two different reasons to re-measure, and both must be handled. `clear_cache` returns what the
# OLD session freed; only a real multi-token pass returns what the NEW one abandoned filling its
# pool. Re-measuring with only the first would restore the 256-token ceiling after every reload.
check("...and settles the new pool first, or the over-estimate returns after every reload",
      "_settle()" in srv
      and srv.index("self.session._settle()") < srv.index("footprint_gb = _mx.get_active_memory()"))
check("...and recomputes the ceiling from it",
      "_token_ceiling()" in srv)
check("...in that order", srv.index("footprint_gb = _mx.get_active_memory()")
      < srv.index("_token_ceiling()"))

print()
print("=" * 84)
print("4. AND THE NUMBER MUST REACH THE USER, NOT JUST THE LOG")
print("=" * 84)
page = open(os.path.join(ROOT, "bigrig_engine", "webui.html"), encoding="utf-8").read()
check("the interface shows the longest reply it will accept", "max_completion_tokens" in page)
check("...and says which of the two limits produced it", "token_limit_reason" in page
      or "limited by memory" in page)
check("the session reports both, so a client can say so too",
      all(k in inspect.getsource(S.Session.stats)
          for k in ("max_completion_tokens", "token_limit_reason")))

print()
print("=" * 84)
print("5. STREAMING THE KV CACHE WAS THE PLAN. THE ARITHMETIC REFUSED IT.")
print("=" * 84)
# Experts stream because routing is SPARSE -- 8 of 128, so a pool of 11 serves a token and a miss
# reads 2 MB. Attention has no equivalent: every token attends to every previous key and value,
# so paging the cache means re-reading ALL of it every token. Measured on Qwen3-30B-A3B-3bit:
#
#     context      per token, dense    at 5 GB/s
#       8,192          805 MB            0.16 s
#      32,768         3221 MB            0.64 s
#     131,072        12885 MB            2.58 s
#
# against 793 MB per token for the experts at a FULL miss, which almost never happens. There is
# no residency policy that fixes a 100% miss rate, so the feature does not exist and this records
# why rather than leaving it to be proposed again.
#
# The problem it was meant to solve was real: 1.82 GB left for KV is 18,514 tokens of a 40,960
# window, so 55% of the model's own context was unreachable. Quantising the cache solves it:
#
#     fp16 (what shipped)      6.97 tok/s   147.3 MB of KV
#     8-bit                    7.39 tok/s    78.2 MB    1.88x less
#     4-bit                    7.78 tok/s    41.4 MB    3.56x less
#     4-bit after 1024 tokens  8.66 tok/s    41.4 MB    3.56x less
#
# FASTER, because the cache is most of what attention reads.
check("the refutation is recorded next to the constant that replaced it",
      "12.9 GB" in _mod_src if (_mod_src := open(
          os.path.join(ROOT, "bigrig_engine", "session.py"), encoding="utf-8").read()) else False)
check("...including that experts stream only because routing is sparse", "SPARSE" in _mod_src)
# ON BY DEFAULT, AND THE THRESHOLD IS WHAT MAKES THAT DEFENSIBLE. It is a compression and it
# does change the reply -- but below KV_QUANT_START nothing is quantised and the arithmetic is
# bit-for-bit what it was, and above it the alternative is not an exact reply but NO reply: the
# ceiling stopped at 18,514 tokens of a 40,960 window. Trading exactness for a conversation that
# can continue is the right way round.
check("compression is on by default, at a bit width that was measured",
      S.KV_BITS == 4, str(S.KV_BITS))
check("...but only past a threshold, so a short conversation is untouched",
      S.KV_QUANT_START >= 1024, str(S.KV_QUANT_START))
check("...and it can be turned off, for anyone who needs the stronger guarantee",
      "kv_bits" in __import__("inspect").signature(S.Session.__init__).parameters)
check("the ceiling arithmetic accounts for it rather than reporting the old number",
      "self.kv_bits" in inspect.getsource(S.Session._token_ceiling))
_c = S.Session._token_ceiling


check("with the cache compressed the ceiling rises",
      _c(_S(kv_bits=4))[0] > _c(_S())[0], f"{_c(_S())[0]} -> {_c(_S(kv_bits=4))[0]}")
check("...and stops at the model's own window rather than exceeding it",
      _c(_S(kv_bits=4))[0] <= 40960 and _c(_S(kv_bits=4))[1] == "context")
check("an impossible bit width is refused at construction",
      "kv_bits must be one of" in _mod_src)
check("the setting is reported, so a client can tell whether it is on",
      '"kv_bits": self.kv_bits' in _mod_src)

print()
print("=" * 84)
print("ALL TESTS PASSED" if not FAIL else f"{len(FAIL)} FAILURES: " + ", ".join(FAIL))
print("=" * 84)
sys.exit(1 if FAIL else 0)
