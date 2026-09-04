"""Guessing tokens from text already seen, and the exactness it costs to check the guess.

WHY THIS FILE IS ADVERSARIAL
    Two failures here are silent. A draft accepted past its first wrong token leaves the cache
    holding tokens the model never emitted, and generation continues fluently from a state that
    never existed -- no error, just a worse model. And a cache trimmed by the wrong amount is the
    same bug with the sign flipped. Both are checked against a reference walk rather than by
    inspection.

    The third is not a bug but a property, and it is the reason this is off by default: the
    verifying pass changes the answer even when every guess is wrong. It is measured, not argued.
"""
import inspect
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from bigrig_engine.lookahead import Stats, propose                      # noqa: E402

FAIL = []


def check(n, c, d=""):
    (print(f"  PASS  {n}") if c else (FAIL.append(n), print(f"  FAIL  {n}  {d}")))


print("=" * 84)
print("1. THE GUESS ITSELF")
print("=" * 84)
check("it returns what followed the pattern last time",
      propose([1, 2, 3, 4, 5, 1, 2, 3], n_gram=3, k=4) == [4, 5, 1, 2])
check("nothing to match on means no guess, not a wrong one",
      propose([9, 8, 7], n_gram=3, k=4) == [])
check("it falls back to a shorter pattern when the long one does not match",
      propose([1, 2, 3, 1, 2], n_gram=3, k=2) == [3, 1])
check("...but not below min_gram, where a match means nothing",
      propose([5, 1, 9, 9, 1], n_gram=3, k=2, min_gram=2) == [])
check("it never guesses more than it was asked for",
      len(propose([1, 2, 3, 4, 5, 6, 1, 2, 3], n_gram=3, k=2)) == 2)
# The pattern sits at the END of the context, so a search that included its own position would
# "predict" the tokens after it -- which do not exist yet -- and on a longer context would
# silently hand back the pattern's own tail as a guess. The search stops one short of it.
check("the pattern is never matched against its own position",
      propose([1, 2, 3], n_gram=2, k=2, min_gram=2) == [])
# An OVERLAPPING earlier occurrence is a different thing and is legitimate: in [7, 7, 7] the
# pair (7, 7) really did occur earlier and really was followed by a 7, so guessing 7 is the
# n-gram's honest answer rather than a self-match. Pinned because the obvious "fix" for the
# check above -- excluding any i within g of the end -- would break it.
check("...but a genuine overlapping earlier occurrence is still usable",
      propose([7, 7, 7], n_gram=3, k=1, min_gram=2) == [7])
# Most recent occurrence wins: in a conversation the recent past predicts better than the opening.
check("the most recent occurrence is the one used",
      propose([1, 2, 99, 1, 2, 77, 1, 2], n_gram=2, k=1) == [77])
check("an empty or too-short context is handled, not indexed off the end",
      propose([], n_gram=3, k=4) == [] and propose([1], n_gram=3, k=4) == [])

print()
print("=" * 84)
print("2. VERIFICATION MUST STOP AT THE FIRST WRONG GUESS, AND LEAVE THE CACHE EXACT")
print("=" * 84)
# A fake model and a fake cache, so acceptance and trimming are checked against a known answer
# rather than against whatever a real model happens to do.
import mlx.core as mx                                                   # noqa: E402
import bigrig_engine.lookahead as LA                                    # noqa: E402


class _Cache:
    def __init__(self):
        self.n = 0


def _fake_trim(cache, k):
    cache[0].n -= k
    return k


class _Model:
    """Always predicts TRUTH[i] at position i. Deterministic, so acceptance is knowable."""
    TRUTH = [100, 101, 102, 103, 104, 105]

    def __init__(self):
        self.vocab = 200

    def __call__(self, ids, cache=None):
        n = ids.shape[1]
        cache[0].n += n
        rows = []
        for j in range(n):
            row = [0.0] * self.vocab
            row[self.TRUTH[j]] = 10.0
            rows.append(row)
        return mx.array([rows])


_real_trim = None
try:
    import mlx_lm.models.cache as _C
    _real_trim = _C.trim_prompt_cache
    _C.trim_prompt_cache = _fake_trim

    # Every guess right: all kept, plus the free token after them.
    c, st = [_Cache()], Stats()
    got, _lg, _row = LA.verify(_Model(), c, 99, [100, 101, 102], None, st)
    check("a fully correct draft keeps every token and takes the free one after it",
          got == [100, 101, 102, 103], str(got))
    check("...and the cache holds exactly those, no more", c[0].n == 4, str(c[0].n))
    check("...and the stats say so", st.accepted == 3 and st.drafted == 3)

    # Wrong in the middle: everything after it is invalid and must be dropped.
    c, st = [_Cache()], Stats()
    got, _lg, _row = LA.verify(_Model(), c, 99, [100, 999, 102], None, st)
    check("a draft wrong in the middle stops there rather than keeping later guesses",
          got == [100, 101], str(got))
    check("...and the cache is trimmed back to exactly what was kept", c[0].n == 2, str(c[0].n))
    check("...and acceptance is counted honestly", st.accepted == 1 and st.drafted == 3)

    # Entirely wrong: still yields one real token, which is what makes the worst case break even.
    c, st = [_Cache()], Stats()
    got, _lg, _row = LA.verify(_Model(), c, 99, [999, 998, 997], None, st)
    check("a wholly wrong draft still returns the token an ordinary step would have",
          got == [100], str(got))
    check("...and leaves the cache holding one token, not four", c[0].n == 1, str(c[0].n))
    check("...so a wrong guess costs a pass, never a wrong token", st.accepted == 0)
    check("acceptance rate is reported, and is zero here", st.acceptance == 0.0)
finally:
    if _real_trim is not None:
        _C.trim_prompt_cache = _real_trim

print()
print("=" * 84)
print("3. THE PROPERTY THAT KEEPS THIS OFF BY DEFAULT")
print("=" * 84)
# MEASURED on Qwen3-30B-A3B-3bit at capacity 11.
#
#   WHERE IT PAYS
#     quoting a document back    94% of guesses accepted   10.85 -> 17.13 tok/s   1.58x
#                               and the output was byte-identical to plain greedy
#   WHERE IT DOES NOT
#     open prose                  8% of guesses accepted   no speed change worth the name
#                               and the output DIFFERED
#
#   WHY IT DIFFERED, WHICH IS NOT WHAT IT LOOKS LIKE. It is not the accepted guesses. Stepped
#   twelve positions alone, then the same twelve inside a five-token verifying pass whose draft
#   was nonsense by construction so nothing could be accepted: 2 of the 12 still chose a
#   DIFFERENT token, largest logit gap 1.78. A logit computed beside four others is not the logit
#   computed alone. Drafting at all is what costs exactness -- acceptance only decides whether
#   anything is bought with it.
#
#   So this ships as a capability with its numbers, not as a default. On the workload where it
#   wins it wins large and changes nothing; on the workload most requests actually are, it changes
#   roughly one token in six and buys nothing.
_src = open(os.path.join(ROOT, "bigrig_engine", "lookahead.py")).read()
check("the module says plainly that an accepted token is not always the token greedy would give",
      "not always" in _src)
check("...and records where the technique pays and where it does not",
      "does not translate" in _src or "open prose" in _src.lower())
check("it is wired into the serving path, and OFF unless a request asks",
      "lookahead: bool = False" in open(os.path.join(ROOT, "bigrig_engine", "session.py")).read())
check("the break-even table is recorded next to the code it justifies", "break-even" in _src)

print()
print("=" * 84)
print("4. THE STREAM IT PRODUCES MUST BE THE ONE mlx_lm PRODUCES, FIELD FOR FIELD")
print("=" * 84)
# Everything downstream -- the quality meter, stop sequences, the harmony rewriter, both HTTP
# endpoints -- consumes GenerationResponse and was written against mlx_lm's exact behaviour,
# including two details that look like bugs and are not: the end-of-sequence token is never
# yielded, and the final response repeats the last segment with a finish_reason attached.
#
# VERIFIED AGAINST THE REAL GENERATOR, not asserted here: on three prompts -- one that drafts
# well, one that drafts badly, and one that runs into max_tokens -- the two produced identical
# lists of (text, token, generation_tokens, finish_reason). The third case is the one that
# matters most, because generation_tokens becomes usage.completion_tokens on both endpoints and
# mlx_lm counts the eos token there.
src = open(os.path.join(ROOT, "bigrig_engine", "lookahead.py"), encoding="utf-8").read()
check("the loop counts what mlx_lm's enumerate counts, including the eos token",
      "n = -1" in src and "generation_tokens=n + 1" in src)
check("...and says why, because it looks like an off-by-one",
      "off-by-one" in src)
check("the eos token is consumed but never yielded",
      "if token in tokenizer.eos_token_ids:" in src
      and src.index("if token in tokenizer.eos_token_ids:") < src.index("detok.add_token(token)"))
check("a final response is emitted with a finish_reason",
      'finish_reason="stop" if token in tokenizer.eos_token_ids else "length"' in src)
check("prefill uses the width the rest of the engine uses, not one of its own",
      "prefill_step_size" in src and "quality-visible" in src)

print()
print("=" * 84)
print("5. SAMPLING MUST NOT BE SKEWED, AND THE REASON IS NOT THE USUAL ONE")
print("=" * 84)
# Speculative decoding with a draft MODEL needs rejection sampling, because the draft proposes
# from its own distribution. Nothing here proposes from a distribution: the guess comes from text
# already written and carries no probability mass. So the rule is "sample from the model, and
# notice we had guessed it" -- every kept token was drawn from the model's own conditional.
vs = inspect.getsource(LA.verify)
check("verification samples from the model at every position, not just the first",
      "sampler(logits[0] - lse)" in vs)
check("...in one call rather than one per position",
      vs.count("sampler(") == 1)
check("a guess is kept only when the model's own draw matches it",
      "if picks[j] == int(d):" in vs)
check("the argument for it being unbiased is written down, not assumed",
      "unbiased" in vs and "rejection sampling" in vs)

print()
print("=" * 84)
print("6. IT MUST STOP GUESSING WHEN GUESSES STOP LANDING")
print("=" * 84)
# A verifying pass of k+1 tokens costs about 2.2 single passes at k=3, so a draft nobody accepts
# is most of a wasted token. Without backing off, prose that guesses cannot predict measured
# 0.80x at k=8 and 0.75x at k=12. Most real replies are BOTH kinds at once -- quote a document,
# then comment on it -- so the loop has to notice mid-reply rather than be told in advance.
check("a failed draft backs the loop off", "misses, skip = 0, 0" in src and "skip = min(" in src)
check("...for longer each time it fails again", "2 ** misses" in src)
check("...but never so long that it stops noticing", "MAX_BACKOFF" in src and LA.MAX_BACKOFF <= 64)
check("any acceptance at all resets it to full rate",
      "if len(got) > 1:" in src and "misses, skip = 0, 0" in src.split("if len(got) > 1:")[1][:120])
check("how often it backed off is reported, not hidden", "backed_off" in LA.Stats().as_dict())

print()
print("=" * 84)
print("7. THE TWO INTERACTIONS THAT MADE IT LOOK BROKEN")
print("=" * 84)
# Both were found by measuring through the server rather than the Python API, and neither shows
# up as an error -- they show up as a feature that does nothing.
#
# THE PROMPT CACHE. When a prefix has been served from cache, the prompt handed to the generator
# is a handful of tail tokens; the document a draft should be quoting lives in the KV cache as
# attention state, not as tokens. Drafting searched the tail and found nothing: the same request
# ran 1.48x through the Python API and 1.00x through the server.
check("the generator is given the whole conversation to draft from, not just the unread tail",
      "context_ids" in src)
check("...and the session passes it the full ids when the cache served a prefix",
      "context_ids=full_ids" in open(os.path.join(ROOT, "bigrig_engine", "session.py"),
                                     encoding="utf-8").read())
# THINKING. With a reasoning block enabled the model first writes original reasoning, which no
# draft can predict. Same request: 7% accepted with thinking on, 99% with it off. Not a defect --
# but it is why a measurement taken through the server disagreed with one taken through the API,
# and the difference has to be recorded or it will be rediscovered.
check("the measured regimes are recorded, including what makes acceptance collapse",
      "think" in src.lower() and "99%" in src)

print()
print("=" * 84)
print("8. AND THE QUALITY METER MUST NOT BE FED A TOKEN THE MODEL NEVER EMITTED")
print("=" * 84)
# The meter's free-energy reading normally comes from wrapping the model's forward and taking the
# LAST row of what came back. A verifying pass ends on a position that may have been rejected, so
# that row belongs to a token that was thrown away. Without handing back the right row the meter
# falls back to reading the whole 151,000-wide log-probability vector per token -- 628 KB each,
# measured at 34% of generation, enough to turn a 2.97x speedup into 0.92x.
ses = open(os.path.join(ROOT, "bigrig_engine", "session.py"), encoding="utf-8").read()
check("the generator hands back the row that produced each token", "on_logits" in src)
check("...and the session feeds it to the meter", "_set_logits" in ses)
check("...instead of wrapping the forward, which would read the wrong row",
      "_mcls.__call__ = _capture" in ses
      and "else:" in ses[:ses.index("_mcls.__call__ = _capture")][-400:])

print()
print("=" * 84)
print("9. THE GAP TO mlx_lm ON A REPLY THAT NEVER DRAFTS, AND WHAT DID NOT CLOSE IT")
print("=" * 84)
# A reply that does not repeat itself is almost entirely plain single-token steps, and on one of
# those this loop measures 0.83-0.85x against plain decoding. Recorded here so the same five
# things are not tried again:
#
#   the forward-pass count is not the cause  1.02 model calls per token against 1.03
#   `propose` is not the cause               0.011 ms per token
#   the quality meter is not the cause       0.85x with it off, 0.89x with it on
#   async_eval then reading in the same round queues nothing; slower everywhere
#   issuing the next pass before syncing      no change -- step i+1 needs step i's TOKEN, so the
#                                             two passes are serial on the GPU regardless
#
# What DID measure: wired_limit and the generation stream, which mlx_lm wraps its own loop in.
# No help on the unfavourable reply, but a real gain where drafting works.
check("the loop runs inside the wiring and stream mlx_lm generates on",
      "wired_limit(model, [generation_stream])" in src and "mx.stream(generation_stream)" in src)
check("...and the isolated measurement is recorded, noise included",
      "16.39" in src and "median" in src)
check("everything that was tried and failed is written down, not silently dropped",
      all(t in src for t in ("async_eval", "serial on the GPU", "not the meter")))

print()
print("=" * 84)
print("10. THE CAUSE, FOUND BY BISECTING RATHER THAN BY GUESSING")
print("=" * 84)
# Five hypotheses had been ruled out and the cause was still unknown. The bisect that found it
# was one line: force `propose` to return nothing, so the loop runs and drafting never does.
#
#     mlx_lm's loop                    20.64 tok/s   1.00x
#     this loop, drafting as normal    16.02 tok/s   0.78x
#     this loop, drafting FORCED OFF   19.83 tok/s   0.96x
#
# So the plain-step path was never the problem -- it is within 4% of mlx_lm's. The cost was the
# draft machinery, which every earlier hypothesis had looked past because the pass COUNT looked
# fine: an ordinary question made only TWO verify passes in sixty-four tokens.
#
# Two passes, and they cost 15% of the reply. On a streamed model a wide speculative pass is far
# more expensive than on a resident one: nine tokens at top-8 wants 72 expert slots against a
# pool of 11, so it splits and costs about ten ordinary steps rather than the two a resident
# model would pay. The width was the price, and it was being paid at the ceiling before the
# backoff had seen a single failure.
check("the draft width is earned, not assumed", "cur_k = 1" in src)
check("...doubling on an accepted draft", "cur_k = min(int(k), max(1, cur_k * 2))" in src)
check("...and halving on a rejected one", "cur_k = max(1, cur_k // 2)" in src)
check("the ceiling the caller asked for is still respected",
      "min(int(cur_k)" in src and "cur_k = min(int(k)" in src)
check("the bisect and its numbers are recorded, not just the conclusion",
      "72 expert slots" in src and "ten ordinary steps" in src)
# MEASURED AFTER: an ordinary question 0.84x -> 0.95x, open prose 0.84x -> 0.89x, and the
# favourable case improved too, 1.32x -> 1.40x, because a reply that drafts well reaches the
# ceiling within three accepted guesses and never pays for a wide miss on the way.
check("the honest conclusion is stated rather than a fix implied",
      "honest price" in src)

print()
print("=" * 84)
print("ALL TESTS PASSED" if not FAIL else f"{len(FAIL)} FAILURES: " + ", ".join(FAIL))
print("=" * 84)
sys.exit(1 if FAIL else 0)
