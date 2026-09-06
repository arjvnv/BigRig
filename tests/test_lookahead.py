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
# Fake caches of the two kinds a real model has, and a fake model that records what it was fed,
# so acceptance and rollback are checked against a known answer through the REAL verify, the
# real mlx_lm trim and the real rollback -- nothing patched.
import mlx.core as mx                                                   # noqa: E402
import bigrig_engine.lookahead as LA                                    # noqa: E402


class _KV:
    """An attention cache: a write position that trims."""
    def __init__(self): self.offset = 0
    def is_trimmable(self): return True
    def trim(self, n):
        n = min(n, self.offset); self.offset -= n; return n


class _Recurrent:
    """A recurrent-state cache, the ArraysCache shape: `cache` is a list the model REBINDS. The
    state here is the tuple of every token id it has processed, so a phantom token is visible."""
    def __init__(self): self.cache = [()]
    def is_trimmable(self): return False
    def __getitem__(self, i): return self.cache[i]
    def __setitem__(self, i, v): self.cache[i] = v


class _Model:
    """Always predicts TRUTH[i] at position i. Deterministic, so acceptance is knowable."""
    TRUTH = [100, 101, 102, 103, 104, 105]

    def __init__(self):
        self.vocab = 200
        self.calls = []

    def __call__(self, ids, cache=None):
        toks = [int(t) for t in ids[0]]
        self.calls.append(toks)
        for c in cache:
            if isinstance(c, _KV):
                c.offset += len(toks)
            else:
                c[0] = tuple(c[0]) + tuple(toks)             # rebinding, as the real layer does
        rows = []
        for j in range(len(toks)):
            row = [0.0] * self.vocab
            row[self.TRUTH[j]] = 10.0
            rows.append(row)
        return mx.array([rows])


def _fresh(mixed):
    return [_KV(), _Recurrent(), _KV()] if mixed else [_KV(), _KV()]


for mixed in (False, True):
    kind = "a cache with recurrent state (Qwen3.6, Nemotron)" if mixed else "a cache that trims (KV only)"
    print(f"  -- {kind}")
    # Every guess right: all kept, plus the free token after them.
    c, st, m = _fresh(mixed), Stats(), _Model()
    got, _lg, _row = LA.verify(m, c, 99, [100, 101, 102], None, st)
    check("a fully correct draft keeps every token and takes the free one after it",
          got == [100, 101, 102, 103], str(got))
    check("...and the cache holds exactly those, no more",
          all(k.offset == 4 for k in c if isinstance(k, _KV))
          and all(r[0] == (99, 100, 101, 102) for r in c if isinstance(r, _Recurrent)),
          str([(k.offset if isinstance(k, _KV) else k[0]) for k in c]))
    check("...and the stats say so", st.accepted == 3 and st.drafted == 3 and st.rereads == 0)

    # Wrong in the middle: everything after it is invalid and must be dropped.
    c, st, m = _fresh(mixed), Stats(), _Model()
    got, _lg, _row = LA.verify(m, c, 99, [100, 999, 102], None, st)
    check("a draft wrong in the middle stops there rather than keeping later guesses",
          got == [100, 101], str(got))
    check("...and every layer is left holding exactly what was kept -- no phantom tokens",
          all(k.offset == 2 for k in c if isinstance(k, _KV))
          and all(r[0] == (99, 100) for r in c if isinstance(r, _Recurrent)),
          str([(k.offset if isinstance(k, _KV) else k[0]) for k in c]))
    check("...and acceptance is counted honestly", st.accepted == 1 and st.drafted == 3)
    if mixed:
        check("...a recurrent model re-reads the kept tokens in one extra pass, and says so",
              st.rereads == 1 and st.passes == 2 and m.calls[-1] == [99, 100], str(m.calls))
    else:
        check("...a trimmable model needs no extra pass", st.rereads == 0 and st.passes == 1 and len(m.calls) == 1)

    # Entirely wrong: still yields one real token, which is what makes the worst case break even.
    c, st, m = _fresh(mixed), Stats(), _Model()
    got, _lg, _row = LA.verify(m, c, 99, [999, 998, 997], None, st)
    check("a wholly wrong draft still returns the token an ordinary step would have",
          got == [100], str(got))
    check("...and leaves the cache holding one token, not four",
          all(k.offset == 1 for k in c if isinstance(k, _KV))
          and all(r[0] == (99,) for r in c if isinstance(r, _Recurrent)),
          str([(k.offset if isinstance(k, _KV) else k[0]) for k in c]))
    check("...so a wrong guess costs a pass, never a wrong token", st.accepted == 0)
    check("acceptance rate is reported, and is zero here", st.acceptance == 0.0)

# THE BUG, REPRODUCED ON THE FAKE: mlx_lm's trim alone leaves the rejected guess in every layer.
from mlx_lm.models.cache import trim_prompt_cache as _trim                # noqa: E402
c, m = _fresh(True), _Model()
m(mx.array([[99, 999, 998]]), cache=c)
_trim(c, 2)
check("mlx_lm's trim does nothing on a cache with a recurrent layer (the defect this guards)",
      c[0].offset == 3 and c[1][0] == (99, 999, 998), str([c[0].offset, c[1][0]]))


class _Opaque:
    def is_trimmable(self): return False
try:
    LA.verify(_Model(), [_KV(), _Opaque()], 99, [100], None, Stats())
    check("a layer that can be neither trimmed nor put back is refused, never corrupted", False)
except TypeError as e:
    check("a layer that can be neither trimmed nor put back is refused, never corrupted",
          "Opaque" in str(e) and "put back" in str(e), str(e))

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
sys.path.insert(0, os.path.join(ROOT, "tests"))
from _fakeserver import fake_server, post as _fpost                    # noqa: E402
import inspect as _insp                                                 # noqa: E402
from bigrig_engine import session as _sess                              # noqa: E402
with fake_server() as (_url, _state, _fs):
    _fpost(_url, "/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 2})
    _fpost(_url, "/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 2, "lookahead": True})
check("it is wired into the serving path, and OFF unless a request asks",
      _insp.signature(_sess.Session.stream_text).parameters["lookahead"].default is False
      and len(_fs.calls) == 2 and not _fs.calls[0].get("lookahead") and _fs.calls[1].get("lookahead") is True,
      str([c.get("lookahead") for c in _fs.calls]))
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
print("6. ON A REAL MODEL WITH RECURRENT STATE, A REJECTED GUESS LEAVES NO TRACE")
print("=" * 84)
_real = next((m for m in ("Qwen3.6-35B-A3B-4bit", "NVIDIA-Nemotron-3-Nano-30B-A3B-4bit")
              if os.path.isdir(os.path.join(ROOT, "models", m))), None)
if _real is None:
    print("  SKIPPED - no Qwen3.6 / Nemotron locally")
else:
    import json as _json
    import subprocess as _sp
    _p = _sp.run([sys.executable, os.path.join(ROOT, "tests", "_snapshot_child.py"), "lookahead_rollback", _real],
                 capture_output=True, text=True, timeout=900,
                 env=dict(os.environ, BIGRIG_MAX_GB=os.environ.get("BIGRIG_MAX_GB", "9")))
    _line = next((ln for ln in _p.stdout.splitlines() if ln.startswith("RESULT ")), None)
    if _line is None:
        check("the real-model scenario ran", False, (_p.stdout + _p.stderr)[-1200:])
    else:
        R = _json.loads(_line[7:])
        check("this model's cache really cannot be trimmed", not R["trimmable"])
        check("after a wholly rejected guess the token is the plain step's",
              R["rejected_token_same"])
        check("...and every layer -- attention offsets AND recurrent arrays -- is bit-identical to a plain step",
              R["rejected_offsets_same"] and R["rejected_arrays_same_count"] and R["rejected_bit_identical"])
        check("...at the cost of one re-read pass, reported", R["rejected_stats"]["rereads"] == 1
              and R["rejected_stats"]["verify_passes"] == 2, str(R["rejected_stats"]))
        check("mlx_lm's trim alone leaves the state different (the defect, reproduced live)",
              R["bug_offsets_differ"] and R["bug_arrays_differ"])
        check("an accepted guess advances every layer by exactly the tokens kept",
              R["accepted_got"] and R["accepted_offsets_same"] and R["accepted_stats"]["rereads"] == 0)
        check("...with the state within bfloat16 prefill-width noise of two plain steps (not exact, and not wrong)",
              R["accepted_max_rel_diff"] is not None and R["accepted_max_rel_diff"] < 0.10,
              f"max rel diff {R['accepted_max_rel_diff']:.3f}; worst {R['accepted_worst'][:1]}")

print()
print("=" * 84)
print("ALL TESTS PASSED" if not FAIL else f"{len(FAIL)} FAILURES: " + ", ".join(FAIL))
print("=" * 84)
sys.exit(1 if FAIL else 0)
