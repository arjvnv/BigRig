"""bigrig — a runtime quality meter for local LLM inference.

WHAT IT DOES
    Tells you, while a model is generating, whether the output it is producing is poor --
    without a reference answer, without a second model, and without a second forward pass.

WHY IT EXISTS
    Every technique that makes a large model fit on small hardware -- quantisation, expert
    offloading, cache-aware routing -- trades quality for speed. None of the systems that
    implement them tell the user whether the trade went too far. FreeToken's 555-file codebase
    contains zero occurrences of "quality", "perplexity", "accuracy", "confidence" or "entropy".

WHAT IT NEEDS FROM YOUR ENGINE
    The next-token probability distribution. That is all. Three numbers are derived from it:
    entropy, top-1 probability, and the top1-top2 margin. No cache statistics, no router
    internals, no cooperation of any kind.

    This was measured, not assumed. Using all seven signals we tried scored rho=0.824; using
    only these three scored rho=0.893 -- the minimal set is also the BEST, because the extra
    features overfit.

WHAT IT DOES NOT DO
    It cannot tell you WHY quality is poor. A "damage alarm" -- attributing degradation to your
    compression setting -- was built, tested, and REFUTED: a placebo comparing two undamaged
    outputs scored HIGHER (0.75) than the claimed damage detection (0.55). Do not make that
    claim; this library will not help you make it.

    It is also SILENT on very short outputs: 16 tokens for a provisional reading, 64 for a full
    one. A generation that ends before that gets no reading at all.

USAGE — start here unless you have calibrated for your own model
    from bigrig_layer import AdaptiveMeter
    m = AdaptiveMeter()               # learns what THIS model's normal looks like
    for step in generation:
        m.observe(probs)              # numpy array of next-token probabilities
        m.observe_token(token)        # REQUIRED, or looping detection is off
        if m.is_degraded():
            ...                       # back off, warn, or log

    `QualityMeter` is the fixed-calibration alternative. Its constants were fitted on
    Ling-mini-2.0-3bit; on a base model its repetition threshold fires on 21.6% of HEALTHY
    output. Use it only once you have refit it for your model with
    `QualityMeter.calibrate(features, targets, repetitions=...)`.
ATTACHING TO AN ENGINE YOU DID NOT WRITE
    Most engines expose the top-K log-probabilities over their HTTP API, which is enough:

        from bigrig_layer import AdaptiveMeter, observe_ollama_entry
        m = AdaptiveMeter()
        for entry in response["logprobs"]:          # ollama /api/generate
            observe_ollama_entry(m, entry, vocab_size=151936)
            if m.is_degraded():
                ...

    `observe_openai_chunk` does the same for anything speaking the OpenAI logprobs shape --
    llama.cpp's server, vLLM, LM Studio, ollama's /v1 endpoint. top-1 and margin are EXACT from
    a top-K list; entropy is estimated, and how well is measured, not assumed. Request
    `top_logprobs >= 2` or the margin is undefined and the adapter will tell you so.

    Prime the meter on healthy output before trusting its verdicts. A meter whose baseline is
    learned from damaged output concludes that damage is normal.
"""
from .adaptive import AdaptiveMeter  # noqa: F401
from .adapters import (entropy_from_topk, observe_ollama_entry,  # noqa: F401
                       observe_openai_chunk, observe_topk, stats_from_topk)
from .controller import AutoTuner  # noqa: F401
from .meter import QualityMeter, CALIBRATION  # noqa: F401

__version__ = "0.3.0"
__all__ = ["AdaptiveMeter", "QualityMeter", "AutoTuner", "CALIBRATION",
           "observe_topk", "observe_ollama_entry", "observe_openai_chunk",
           "stats_from_topk", "entropy_from_topk"]
