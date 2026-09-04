"""Guessing the next few tokens from text already seen, and checking the guess in one pass.

WHAT THIS IS
    Speculative decoding without a draft model. A second small model is the usual way to produce
    candidate tokens, and it costs memory this product does not have -- the whole premise here is
    a machine that could not fit the FIRST model. So the candidates come from the text instead:
    if the last few tokens have appeared before, whatever followed them last time is a reasonable
    guess at what follows them now.

WHY IT CAN WIN AT ALL
    A forward pass over k tokens costs much less than k passes over one token, because the weights
    are read once instead of k times and on this engine that read is most of the cost. Measured on
    Qwen3-30B-A3B-3bit at capacity 11, against 72.8 ms for a single token:

        k              2       3       4       6       8
        one pass    96.0   155.9   196.8   248.2   298.0 ms
        k passes   145.6   218.4   291.3   436.9   582.5 ms
        break-even   32%     57%     57%     48%     44%   of guesses must be right

    Those ratios are themselves a consequence of regrouping prefill by expert: a k-token pass
    routes to k * top_k experts and takes the split path, which used to cost 252 ms at k=4 and
    now costs 197 ms.

WHERE IT PAYS AND WHERE IT DOES NOT
    Acceptance is a property of the text, not of the model. Quoting a document, continuing a list,
    repeating a name or a code identifier -- the n-gram has been seen and the guess is usually
    right. Open prose invents its next token and the guess is usually wrong, and every wrong guess
    is a pass that bought nothing. It is switched on per request rather than globally for exactly
    that reason. Measured on Qwen3-30B-A3B-3bit at capacity 11, decode rate, k=8:

        reproducing a passage verbatim   20.7 -> 30.7 tok/s   1.48x   99% accepted, same text
        an ordinary question             20.8 -> 18.0 tok/s   0.87x    0% accepted, same text
        original prose                   21.4 -> 18.4 tok/s   0.86x    0% accepted, text differs

    and end to end over HTTP on the first of those, 8.35s -> 6.46s, 1.28x, 106 of 107 guesses
    accepted, 120 tokens in 16 verifying passes.

    HOW BIG A GUESS. The cost of a verifying pass grows more slowly than the tokens it can
    return, so on text that drafts well, larger is better: k=2 1.12x, k=4 1.19x, k=8 1.47x,
    k=12 1.59x. The waste when a guess is wrong grows too, which is what the backoff is for and
    why the default stops at 8.

    THE THING THAT LOOKS LIKE A BUG AND IS NOT. With a reasoning block enabled the model writes
    original reasoning before it writes the answer, and no draft can predict original reasoning.
    The same request measured 7% accepted with thinking on and 99% with it off. Anyone comparing
    a measurement taken through the server against one taken through the Python API will hit
    this, because the server leaves thinking on by default and the API test did not.

WHAT IT COSTS THAT IS NOT TIME
    The verifying pass computes k positions at once, and a logit computed beside k-1 others is not
    bit-identical to one computed alone -- measured elsewhere in this engine at up to 0.97 units on
    a 32.5 span, with 4 of 24 positions holding margins narrower than that. So an accepted token is
    the token the model would have produced ALMOST always, not always. This is not a way of getting
    the same answer faster; it is a different arithmetic path to an answer of the same quality.
"""
from __future__ import annotations

import mlx.core as mx

__all__ = ["propose", "verify", "generate", "stream", "Stats"]


# The longest run of plain single-token steps taken after drafts stop landing, before the loop
# tries a guess again. 32 keeps the wasted-pass cost near a percent while still noticing quickly
# when a reply goes back to quoting something.
MAX_BACKOFF = 32


class Stats:
    """What the loop did, so the trade can be reported instead of assumed."""

    def __init__(self):
        self.drafted = self.accepted = self.rounds = self.passes = 0
        self.backed_off = 0

    @property
    def acceptance(self) -> float:
        return self.accepted / self.drafted if self.drafted else 0.0

    def as_dict(self) -> dict:
        return {"drafted": self.drafted, "accepted": self.accepted,
                "acceptance": round(self.acceptance, 4), "rounds": self.rounds,
                "verify_passes": self.passes, "backed_off": self.backed_off}


def propose(context: list, n_gram: int = 3, k: int = 4, min_gram: int = 2) -> list:
    """Guess the next k tokens by finding the last n_gram tokens earlier in `context`.

    Searches from the END backwards, so the most recent occurrence wins -- in a conversation the
    recent past is a better guide to the next token than the opening paragraph.

    Falls back to shorter patterns down to `min_gram`. A long pattern that matches is a strong
    signal and a short one is a weak one, so the long ones are tried first and the short ones only
    when nothing longer is found; below `min_gram` the match means nothing at all and guessing on
    it would spend a verification pass to learn that.
    """
    n = len(context)
    for g in range(int(n_gram), int(min_gram) - 1, -1):
        if n <= g:
            continue
        pat = context[-g:]
        # -g so the pattern cannot match itself, -1 so there is at least one token after it.
        for i in range(n - g - 1, -1, -1):
            if context[i:i + g] == pat:
                got = context[i + g:i + g + int(k)]
                if got:
                    return list(got)
    return []


def verify(model, cache, last_token: int, draft: list, sampler=None,
           stats: Stats | None = None) -> list:
    """Check a draft in one pass. Returns the tokens to keep, always at least one.

    WHY THIS IS DISTRIBUTIONALLY CORRECT, AND NOT ONLY FOR GREEDY DECODING.
        Speculative decoding with a draft MODEL needs rejection sampling to avoid biasing the
        output, because the draft proposes from its own distribution and that distribution has to
        be corrected for. Nothing here proposes from a distribution: the guess comes from text
        already written and carries no probability mass of its own.

        So the rule is simply "sample the next token from the model, and notice that we had
        already guessed it". At position j the sampler draws from logits[j], which is conditioned
        on exactly the tokens accepted before it; if the draw equals the guess, the guess was
        right and the draw is kept. Every token kept was drawn from the model's own conditional
        distribution, which is the definition of unbiased. Raising the temperature lowers how
        often a guess is drawn, and lowers the speedup -- it does not skew the result.

    THE CONTRACT, AND THE ONE PLACE IT IS EASY TO GET WRONG
        The pass is fed [last_token] + draft, so position j predicts what follows draft[:j]. A
        draft token is kept only if every draft token before it was also kept -- one wrong guess
        invalidates everything after it, because those positions were computed on the assumption
        that the wrong token was there.

        The token after the last accepted one is FREE and is always taken. That is what makes the
        worst case break even rather than lose: a draft of k that is entirely wrong still yields
        the one token an ordinary step would have produced, for one pass instead of one pass.

        The cache is left holding exactly the accepted tokens. Anything drafted and rejected was
        written into it by this pass and is trimmed off, or the next step attends to tokens the
        model never emitted.
    """
    from mlx_lm.models.cache import trim_prompt_cache
    ids = [int(last_token)] + [int(t) for t in draft]
    logits = model(mx.array(ids)[None], cache=cache)
    if sampler is None:
        picks = [int(t) for t in mx.argmax(logits[0], axis=-1)]
    else:
        # Every position at once. Sampling them one at a time would be the same draw and several
        # more round-trips to the device, on a path whose entire purpose is to avoid those.
        lse = mx.logsumexp(logits[0], axis=-1, keepdims=True)
        picks = [int(t) for t in sampler(logits[0] - lse)]
    keep = []
    for j, d in enumerate(draft):
        if picks[j] == int(d):
            keep.append(int(d))
        else:
            break
    out = keep + [picks[len(keep)]]
    kept_row = len(keep)                      # the position whose logits produced the last token
    # Everything fed in beyond what was kept is now in the cache and must come out. The +1 is the
    # token that produced `picks[len(keep)]`; it stays, the rest go.
    extra = len(ids) - (len(keep) + 1)
    if extra > 0:
        trim_prompt_cache(cache, extra)
    if stats is not None:
        stats.rounds += 1
        stats.passes += 1
        stats.drafted += len(draft)
        stats.accepted += len(keep)
    return out, logits[0], kept_row


def generate(model, cache, prompt_ids: list, first_token: int, max_tokens: int,
             n_gram: int = 3, k: int = 4, stats: Stats | None = None) -> tuple:
    """Greedy generation with prompt-lookup drafting. Returns (tokens, stats).

    `cache` must already hold `prompt_ids`, and `first_token` is the token the prompt produced --
    that is, the caller has done the prefill and taken one step. Written this way so it composes
    with the engine's own prefill rather than owning it.

    A round with nothing worth guessing takes an ordinary single-token step, which is both faster
    than verifying an empty draft and exactly the arithmetic the non-speculative path runs.
    """
    st = stats if stats is not None else Stats()
    out = [int(first_token)]
    ctx = list(prompt_ids) + [int(first_token)]
    while len(out) < max_tokens:
        draft = propose(ctx, n_gram=n_gram, k=min(int(k), max_tokens - len(out)))
        if draft:
            got, _lg, _row = verify(model, cache, out[-1], draft, None, st)
        else:
            logits = model(mx.array([out[-1]])[None], cache=cache)
            got = [int(mx.argmax(logits[0, -1]))]
            st.rounds += 1
            st.passes += 1
        out += got
        ctx += got
    return out[:max_tokens], st


def stream(model, tokenizer, prompt, max_tokens: int = 256, prompt_cache=None, sampler=None,
           prefill_step_size: int = 64, n_gram: int = 3, k: int = 4,
           stats: Stats | None = None, prompt_progress_callback=None, on_logits=None,
           context_ids=None):
    """Generate with prompt-lookup drafting, yielding what mlx_lm's stream_generate yields.

    THE SHAPE OF WHAT COMES OUT IS NOT NEGOTIABLE, WHICH IS WHY THIS MIRRORS IT LINE FOR LINE.
        Everything downstream -- the quality meter, the stop-sequence handling, the harmony
        rewriter, both HTTP endpoints -- consumes `GenerationResponse` and was written against
        mlx_lm's exact behaviour, including the two details that look like bugs and are not: the
        end-of-sequence token is never yielded, and the final response repeats the last segment
        with a `finish_reason` attached. Producing a nearly-identical stream would have meant
        every one of those consumers being subtly wrong in a way only long replies would show.

        So this replaces only the token loop. Prefill, detokenising and the response shape are
        the same code paths, doing the same things in the same order.

    `stats` is filled in as it goes, so a caller can report what the drafting actually bought
    rather than what it was hoped to buy.

    `context_ids` is the WHOLE conversation, when `prompt` is only the part of it that still
    needs reading. Those differ whenever the prompt cache has served a prefix, and the difference
    matters more here than anywhere else in the engine: the cache holds the prefix's attention
    state, not its tokens, so a draft looking for a phrase to reuse would be searching a few
    tokens of tail rather than the document the user actually pasted. Measured before this
    existed: the same request that drafts at 1.48x through the Python API ran at 1.00x through
    the server, because the server has the prompt cache on and the Python test did not.

    `on_logits` is handed the RAW logits row that produced each token, just before it is yielded.
    The quality meter wants one scalar off that row -- free energy, `-logsumexp` -- and it
    normally gets it by wrapping the model's forward and taking the last row of whatever came
    back. That wrapper cannot be used here: a verifying pass ends on a position that may have
    been rejected, so the last row belongs to a token the model never emitted. Without this the
    meter falls back to reading the whole 151,000-wide log-probability vector per token, which
    costs 628 KB a token and measured 34% of generation -- enough to turn a 2.97x speedup into
    0.92x, which is exactly what it did before this existed.
    """
    import time as _t

    from mlx_lm.generate import GenerationResponse, generation_stream, wired_limit
    from mlx_lm.models.cache import make_prompt_cache
    from mlx_lm.tokenizer_utils import TokenizerWrapper

    if not isinstance(tokenizer, TokenizerWrapper):
        tokenizer = TokenizerWrapper(tokenizer)
    ids = list(prompt) if not isinstance(prompt, str) else tokenizer.encode(prompt)
    ids = [int(t) for t in ids]
    if prompt_cache is None:
        prompt_cache = make_prompt_cache(model)
    st = stats if stats is not None else Stats()
    detok = tokenizer.detokenizer
    detok.reset()

    # THE SAME WIRING AND THE SAME STREAM mlx_lm GENERATES ON.
    #     Two things its loop does that this one did not: it raises the wired-memory limit for the
    #     duration, so macOS does not reclaim the model's pages mid-reply, and it issues every
    #     forward pass on a dedicated thread-local stream. Both are properties of the CONTEXT the
    #     work runs in rather than of the work itself, which is why a loop issuing an identical
    #     number of identical forward passes was still slower.
    #
    #     Isolated by neutering both and re-running: on a reply that drafts well, four runs each,
    #     1.48x with them against 1.17x without -- but the "with" series is much noisier (20.74,
    #     16.39, 18.39, 16.56 against 15.94, 16.29, 16.37, 16.12), so best-of flatters it. By
    #     median it is nearer 1.08x. Somewhere between the two; kept because both summaries agree
    #     on the direction and neither shows a cost.
    with wired_limit(model, [generation_stream]), mx.stream(generation_stream):
        yield from _run(model, tokenizer, ids, max_tokens, prompt_cache, sampler,
                        prefill_step_size, n_gram, k, st, prompt_progress_callback, on_logits,
                        context_ids, detok, GenerationResponse, _t)


def _run(model, tokenizer, ids, max_tokens, prompt_cache, sampler, prefill_step_size, n_gram, k,
         st, prompt_progress_callback, on_logits, context_ids, detok, GenerationResponse, _t):
    """The loop itself. Split out only so `stream` can wrap it in one `with`."""
    tic = _t.perf_counter()
    # PREFILL IN THE SAME WIDTH THE REST OF THE ENGINE USES.
    #     The step is a quality-visible number here as everywhere else -- chunked prefill is not
    #     bit-exact across widths -- so this must not quietly pick its own. The last token is held
    #     back to be the one the first verifying pass is conditioned on.
    head, last = ids[:-1], ids[-1]
    for i in range(0, len(head), max(1, int(prefill_step_size))):
        model(mx.array(head[i:i + prefill_step_size])[None], cache=prompt_cache)
        mx.eval([c.state for c in prompt_cache if hasattr(c, "state")])
        if prompt_progress_callback is not None:
            try:
                prompt_progress_callback(min(i + prefill_step_size, len(head)), len(head))
            except Exception:                  # noqa: BLE001 -- progress is never worth a failure
                pass
    prompt_time = _t.perf_counter() - tic
    prompt_tps = len(ids) / prompt_time if prompt_time > 0 else 0.0
    tic = _t.perf_counter()

    ctx = [int(t) for t in context_ids] if context_ids else list(ids)
    # BACKING OFF WHEN THE GUESSES ARE WRONG, WHICH IS WHAT MAKES THIS SAFE TO TURN ON.
    #     A verifying pass of k+1 tokens costs about 2.2 single passes at k=3, so a draft nobody
    #     accepts is not free -- it is most of a wasted token. Measured without backoff on prose
    #     that guesses cannot predict: 0.80x at k=8 and 0.75x at k=12, against 1.47x and 1.59x on
    #     text being reproduced almost verbatim.
    #
    #     Most real replies are BOTH: a few sentences quoting a document, then commentary that
    #     invents every token. So rather than choose per request between a large gain and a
    #     material loss, this stops drafting when drafts stop landing and probes again after a
    #     gap that doubles each time. The cost of being wrong falls to one wasted pass per probe;
    #     the cost of being right is unchanged, because a run of accepted drafts never backs off.
    misses, skip = 0, 0
    # HOW MANY TOKENS TO GUESS RIGHT NOW. IT STARTS SMALL AND EARNS ITS WIDTH.
    #     `k` is the ceiling, not the opening bid. A rejected draft is a whole forward pass
    #     thrown away, and on a streamed model that pass is far worse than it looks: nine tokens
    #     at top-8 wants 72 expert slots against a pool of 11, so it splits and costs about ten
    #     ordinary steps. Opening at the ceiling meant a reply that never repeats itself paid
    #     that price before the backoff had seen a single failure -- measured, TWO such drafts
    #     were 15% of an entire reply with nothing else wrong.
    #
    #     Starting at one and doubling on every accepted draft reaches the ceiling in three
    #     rounds, which costs a genuinely repetitive reply almost nothing, and leaves a reply
    #     that guesses badly paying for two-token passes instead of nine-token ones.
    cur_k = 1
    # `n` COUNTS EXACTLY WHAT mlx_lm's ENUMERATE COUNTS, INCLUDING THE END-OF-SEQUENCE TOKEN.
    #     Their loop breaks on eos AFTER the index has advanced, so the final response reports one
    #     more than the number of tokens that carried text. It looks like an off-by-one and it is
    #     the contract: `generation_tokens` becomes `usage.completion_tokens` on both HTTP
    #     endpoints, and a generator that counted the honest number would report one fewer token
    #     than mlx_lm for the same reply. Mirrored rather than corrected.
    n = -1
    token, logprobs, from_draft = last, None, False
    pending: list = []
    while True:
        if not pending:
            if skip > 0:
                skip -= 1
                draft = []
            else:
                draft = propose(ctx, n_gram=n_gram,
                                k=min(int(cur_k), max(1, max_tokens - max(n, 0))))
            if draft:
                got, lg, row = verify(model, prompt_cache, token, draft, sampler, st)
                lse = mx.logsumexp(lg, axis=-1, keepdims=True)
                # One row of log-probabilities per token handed back, matching the position that
                # produced it, so the quality meter sees the same vector it would have seen had
                # each token been generated on its own.
                pending = [(int(t), (lg - lse)[min(j, row)], j < len(got) - 1, lg[min(j, row)])
                           for j, t in enumerate(got)][:max(1, max_tokens - max(n, 0))]
                if len(got) > 1:               # something was accepted: keep going at full rate
                    misses, skip = 0, 0
                    cur_k = min(int(k), max(1, cur_k * 2))
                else:
                    misses += 1
                    skip = min(2 ** misses, MAX_BACKOFF)
                    # AND GUESS LESS NEXT TIME, WHICH MATTERS MORE HERE THAN SKIPPING ROUNDS.
                    #     A rejected draft is a whole forward pass thrown away, and on a STREAMED
                    #     model that pass is far more expensive than it looks. Nine tokens at
                    #     top-8 wants 72 expert slots and the pool holds 11, so the pass splits
                    #     and costs about ten ordinary steps rather than the two a resident model
                    #     would pay. Measured on an ordinary question: TWO rejected 8-token
                    #     drafts, and nothing else wrong, cost 15% of the whole reply.
                    #
                    #     Skipping rounds alone could not fix that -- by the time the backoff
                    #     engages the expensive passes have already happened. Halving the draft
                    #     makes the next mistake a quarter the price, and doubling on success
                    #     climbs back to full width within two accepted guesses.
                    cur_k = max(1, cur_k // 2)
                    st.backed_off += 1
            else:
                # A PLAIN STEP, AND IT IS ABOUT 15% SLOWER THAN mlx_lm's. FIVE THINGS WERE TRIED.
                #     A reply that never repeats itself is almost entirely these steps, and on
                #     one this loop measures 0.83-0.85x against plain decoding. The forward-pass
                #     count is not the cause: 1.02 model calls per token here against 1.03 there.
                #     `propose` is not the cause either, at 0.011 ms a token. What was tried:
                #
                #       async_eval on the token, then reading it in the same iteration
                #           -- queues nothing ahead; slower everywhere
                #       issuing the NEXT forward pass before syncing the current token
                #           -- no change. Step i+1 needs step i's TOKEN, so the two passes are
                #              serial on the GPU whatever is asked of the scheduler
                #       keeping that pipeline engaged by treating "no n-gram found" as a miss
                #           -- no change, and it suppressed guesses that would have landed
                #       turning the quality meter off
                #           -- 0.85x with it off against 0.89x with it on, so not the meter
                #       wired_limit and the generation stream
                #           -- KEPT: no help HERE, but measurably helps where drafting works
                #
                #     So the gap is real, it is not any of the obvious causes, and it is the
                #     honest price of the option being on for work that does not suit it. The
                #     backoff limits how much of a reply pays it, and the option is off unless a
                #     request asks for it.
                out = model(mx.array([token])[None], cache=prompt_cache)
                lg = out[0, -1]
                lse = mx.logsumexp(lg, keepdims=True)
                nxt = int(mx.argmax(lg)) if sampler is None else int(sampler((lg - lse)[None])[0])
                st.rounds += 1
                st.passes += 1
                pending = [(nxt, lg - lse, False, lg)]
        token, logprobs, from_draft, raw = pending.pop(0)
        if on_logits is not None:
            on_logits(raw)
        ctx.append(token)
        n += 1
        if token in tokenizer.eos_token_ids:
            break
        detok.add_token(token)
        if (n + 1) == max_tokens:
            break
        yield GenerationResponse(
            text=detok.last_segment, token=token, logprobs=logprobs, from_draft=from_draft,
            prompt_tokens=len(ids), prompt_tps=prompt_tps, generation_tokens=n + 1,
            generation_tps=(n + 1) / max(_t.perf_counter() - tic, 1e-9),
            peak_memory=mx.get_peak_memory() / 1e9, finish_reason=None)

    detok.finalize()
    yield GenerationResponse(
        text=detok.last_segment, token=token, logprobs=logprobs, from_draft=from_draft,
        prompt_tokens=len(ids), prompt_tps=prompt_tps, generation_tokens=n + 1,
        generation_tps=(n + 1) / max(_t.perf_counter() - tic, 1e-9),
        peak_memory=mx.get_peak_memory() / 1e9,
        finish_reason="stop" if token in tokenizer.eos_token_ids else "length")
