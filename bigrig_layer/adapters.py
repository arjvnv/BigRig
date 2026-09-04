"""ENGINE ADAPTERS — feed the meter from whatever your inference engine already gives you.

THE WHOLE POINT OF THIS FILE
    The meter needs three numbers per token: entropy, top-1 probability, and the top1-top2
    margin. Different engines expose different things, and the gap between them is not cosmetic:

      full distribution        MLX, any in-process framework    all three EXACT
      top-K logprobs           ollama, llama.cpp, OpenAI API    top-1 and margin EXACT,
                                                                entropy must be ESTIMATED
      chosen token's logprob   some streaming APIs              NONE of the three; unusable

    top-1 and margin need only the top TWO entries, so any engine with `top_logprobs >= 2` gives
    them exactly. Entropy is a sum over the whole vocabulary and is the only casualty. Measured
    on Qwen2.5 via ollama, the top 8 entries carry 96.2% of the probability mass -- the missing
    3.8% is spread over ~152,000 tokens, and that tail carries real entropy.

    So `entropy_estimator` below is not a detail. It is the difference between a layer that works
    on every engine and one that quietly works only on the one we developed against.
"""
import math
import warnings

import numpy as np

# smallest top-K at which the entropy estimator was actually measured
MIN_K_FOR_ENTROPY = 5


def entropy_from_topk(logprobs, vocab_size, method="tail_uniform"):
    """Estimate full-vocabulary entropy from the top-K log-probabilities.

    `logprobs` is a sequence of natural-log probabilities, largest first (the OpenAI/ollama
    `top_logprobs` convention). `vocab_size` is needed by every method that models the tail.

      "truncated"     ignore the tail. Biased LOW, and the bias grows exactly when the model is
                      uncertain -- which is when the meter is trying to act.
      "tail_uniform"  spread the residual mass uniformly over the unseen tokens. Uniform is the
                      maximum-entropy arrangement for a given mass, so this is an UPPER bound on
                      the tail's contribution and therefore an upper bound on entropy.

    MEASURED, 48 generations of Ling-mini-2.0-3bit, scoring the METER rather than the entropy
    Bar, pre-registered: rho within the paired bootstrap interval
    of the full-vocabulary version, and a false-alarm rate <= 10%.

        variant          meter rho    vs full vocab     false alarms   verdict
        full vocabulary     0.938       (the ceiling)          2.68%     --
        tail_uniform K=20   0.942             +0.003           5.65%   SHIPS
        tail_uniform K=8    0.928             -0.010           8.10%   SHIPS
        tail_uniform K=5    0.928             -0.010           6.83%   SHIPS
        truncated   K=20    0.934             -0.004           2.73%   misses the bar
        truncated   K=8     0.934             -0.004           2.82%   misses the bar
        truncated   K=5     0.937             -0.001           2.87%   misses the bar

    STATED HONESTLY: all six are within noise of each other at the meter level, and the
    "truncated" variants miss the bar only on the interval's lower edge (-0.069 against a -0.05
    bar), not on their point estimates, which are the best of the six. They also produce FEWER
    false alarms, because under-estimating entropy makes the meter less trigger-happy. The
    pre-registered bar selects `tail_uniform`, so that is the default, but do not read the
    table as evidence that the tail correction is materially better -- it is not.

    PRACTICAL GUIDANCE: use K >= 20 if your engine allows it (lowest false-alarm rate of the
    passing variants). K = 5 is the smallest that was measured; below 5 nothing has been tested
    and `stats_from_topk` will say so.
    """
    p = np.exp(np.asarray(logprobs, dtype=np.float64))
    if p.size == 0:
        raise ValueError("entropy_from_topk(): empty logprobs")
    if not np.all(np.isfinite(p)):
        raise ValueError("entropy_from_topk(): non-finite log-probability")
    nz = p[p > 0]
    h = float(-(nz * np.log(nz)).sum())
    if method == "truncated":
        return h
    if method == "none":
        raise ValueError(
            "method='none' drops entropy; call stats_from_topk(..., method='none') and feed the "
            "meter through observe_topk, which handles it. entropy_from_topk cannot return "
            "'no entropy'.")
    if method != "tail_uniform":
        raise ValueError(f"unknown entropy method {method!r}")
    rest = max(0, int(vocab_size) - p.size)
    R = float(max(0.0, 1.0 - p.sum()))
    if R <= 1e-12 or rest <= 0:
        return h
    return h + float(-R * math.log(R / rest))


def stats_from_topk(logprobs, vocab_size, method="tail_uniform"):
    """(entropy, top1, margin) from top-K log-probabilities, largest first.

    top1 and margin are EXACT here regardless of K, provided K >= 2 -- they depend only on the
    two largest probabilities, which any engine that returns a top-K list has already given you.
    """
    lp = np.asarray(logprobs, dtype=np.float64)
    if lp.size < 2:
        raise ValueError(
            f"stats_from_topk() needs at least 2 log-probabilities, got {lp.size}. "
            "Request top_logprobs >= 2 from your engine; with fewer, the margin is undefined "
            "and the meter cannot see incoherence.")
    if not np.all(np.isfinite(lp)):
        raise ValueError("stats_from_topk(): non-finite log-probability")
    if np.any(np.diff(lp) > 1e-9):
        lp = np.sort(lp)[::-1]                 # tolerate an unsorted engine rather than trust it
    if lp.size < MIN_K_FOR_ENTROPY and method != "none":
        # top-1 and margin are still exact; only entropy is unvalidated down here.
        warnings.warn(
            f"top_logprobs={lp.size} is below the smallest K measured ({MIN_K_FOR_ENTROPY}). "
            f"top-1 and margin remain exact; the entropy estimate is UNVALIDATED at this K. "
            f"Request more, or pass method='none' to drop entropy deliberately.",
            RuntimeWarning, stacklevel=2)
    p = np.exp(lp)
    t1, mg = float(p[0]), float(p[0] - p[1])
    if method == "none":
        # Entropy deliberately dropped. The meter then runs on top-1, margin and repetition.
        # Returning NaN would poison the running baseline, and returning 0.0 would look like a
        # maximally confident model, so the caller is handed None and observe_topk substitutes
        # a NEUTRAL value -- see there.
        return None, t1, mg
    return entropy_from_topk(lp, vocab_size, method), t1, mg


def observe_topk(meter, logprobs, vocab_size, token=None, method="tail_uniform"):
    """Feed ONE generated token to the meter from a top-K logprob payload.

    `token` is the id actually emitted. Pass it: without it the meter cannot see looping, and
    looping is the failure mode the confidence signals are blind to.
    """
    h, t1, mg = stats_from_topk(logprobs, vocab_size, method)
    if h is None:
        # method="none": feed the meter's own running mean for entropy, so the feature is inert
        # rather than misleading. A constant would be learned as this model's normal and the
        # entropy channel would simply contribute nothing -- which is the intent.
        mu = getattr(meter, "_mu", None)
        h = float(mu[0]) if mu is not None else 0.0
    meter.observe_stats(entropy=h, top1=t1, margin=mg)
    if token is not None:
        meter.observe_token(token)
    return h, t1, mg


# ------------------------------------------------------------------ engine-specific shims
def observe_openai_chunk(meter, choice_logprobs, vocab_size, method="tail_uniform"):
    """One entry from an OpenAI-compatible `logprobs.content[i]` object.

    Works for anything that speaks that shape: OpenAI, llama.cpp's server in OAI mode, vLLM,
    LM Studio, and ollama's /v1 endpoint. Returns None if the entry carries no top_logprobs,
    which is the case when the caller forgot to request them.
    """
    top = choice_logprobs.get("top_logprobs") or []
    if len(top) < 2:
        return None
    lps = [t["logprob"] for t in top]
    tok = choice_logprobs.get("token")
    return observe_topk(meter, lps, vocab_size, token=_token_key(tok), method=method)


def observe_ollama_entry(meter, entry, vocab_size, method="tail_uniform"):
    """One entry from ollama's native /api/generate `logprobs` array.

    Shape, verified against ollama 0.32.1:
        {"token": "Three", "logprob": -1.196, "bytes": [...],
         "top_logprobs": [{"token": ..., "logprob": ...}, ...]}
    """
    top = entry.get("top_logprobs") or []
    if len(top) < 2:
        return None
    lps = [t["logprob"] for t in top]
    return observe_topk(meter, lps, vocab_size, token=_token_key(entry.get("token")),
                        method=method)


def _token_key(tok):
    """A stable integer id for a token given only its text.

    Engines that return text rather than ids force this. Repetition is counted over these keys,
    so it only has to be CONSISTENT, not meaningful -- two different strings must not collide,
    and the same string must always map to the same key. Python's hash is randomised per process
    but stable within one, which is exactly the required lifetime.
    """
    if tok is None:
        return None
    return hash(tok) & 0x7FFFFFFF if isinstance(tok, str) else int(tok)


def stats_from_logprobs(logprobs):
    """(entropy, top1, margin) from a FULL-vocabulary log-probability vector.

    WHEN TO USE THIS INSTEAD OF stats_from_topk
        `stats_from_topk` takes the top-K list an external API hands you, and it must estimate the
        tail it cannot see. A local engine has the whole distribution, so there is nothing to
        estimate -- this returns the exact entropy, which is the "full vocabulary" row the
        validation table calls the ceiling (meter rho 0.938).

    WHY IT IS NOT JUST stats_from_topk(full_vector)
        That works, because stats_from_topk defensively sorts an unsorted input -- but sorting
        152,936 float64s costs 4.36 ms per token. At 8 tokens/second that is 3.5% of the entire
        token budget spent putting a vocabulary in order to read two numbers off the front. This
        does the same work in O(n) with a partition, at 0.55 ms.

        It also avoids a real inaccuracy: passing the full vector through the top-K path leaves
        no residual mass for the tail term, which is fine, but slicing to top-20 first (the
        obvious "optimisation") is NOT -- measured 9.85 nats against a true 7.27, because the
        uniform tail assumption is an upper bound and a 20-token window leaves most of the mass
        in it.
    """
    lp = np.asarray(logprobs, dtype=np.float32).reshape(-1)
    if lp.size < 2:
        raise ValueError(f"stats_from_logprobs() needs a vocabulary, got {lp.size} entries")
    if not np.all(np.isfinite(lp)):
        raise ValueError("stats_from_logprobs(): non-finite log-probability")
    p = np.exp(lp, dtype=np.float32)
    # H = -sum p*log p, and log p IS the input -- no second log, no sort.
    h = float(-(p.astype(np.float64) * lp.astype(np.float64)).sum())
    top2 = np.partition(p, -2)[-2:]
    t1 = float(top2[1])
    return h, t1, float(t1 - float(top2[0]))
