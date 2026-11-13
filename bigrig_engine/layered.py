"""Layer-major prefill: push a long prompt through the model one LAYER at a time, not one chunk at a time.

WHY
    mlx_lm reads a prompt in chunks of `prefill_step` tokens, and each chunk runs through all 40
    layers before the next chunk starts. With streamed experts that order is expensive: every
    layer's experts are fetched once per chunk, and between two visits to the same layer the
    other 39 layers' experts have pushed it out of the page cache. A 1,771-token prompt is 7
    chunks at the session's 256-token step, and reading it chunk by chunk on a 16 GB Mac fetched
    55 GB of experts, ~50 GB of it from the SSD. Processing every chunk through layer L before any
    chunk reaches layer L+1 visits each layer once, back to back, so after the first chunk its
    experts are already in memory. Measured (climb iteration 2, 2026-11-13): time to first token
    22.1 -> 11.2 s on that prompt, every reply token-identical.

WHY THE OUTPUT IS BIT-IDENTICAL
    Only the ORDER of independent work changes. Chunk c at layer L needs two things: its own
    hidden state from layer L-1, and layer L's cache after chunks 0..c-1. Both are the same in
    either order. Every chunk keeps the same boundaries (`step`, counted from the start of the
    tokens given, exactly as mlx_lm's loop counts them), the same kernels and the same masks:
    each mask is built from the same cache object, at the moment that cache has absorbed exactly
    chunks 0..c-1, which is the state the chunked forward builds it from. The chunked forward's
    logits for prompt tokens are never evaluated (mlx_lm keeps only the cache), so they are not
    computed here either.

    This is proved for the architecture below and for nothing else: `text_model` returns None for
    any other model, and the caller falls back to mlx_lm's own loop. Prior art: Layered Prefill
    (arXiv 2510.08055) and Stream-Loading Prefill (arXiv 2606.10493) use the same order for the
    same reason on GPU servers; Edge0 (arXiv 2609.18063) loads whole layers for prefill on a Mac.
"""
import contextlib
import functools
import hashlib
import inspect

import mlx.core as mx
from mlx_lm.models.base import create_attention_mask, create_ssm_mask


def text_model(model):
    """The decoder stack this schedule is known to be exact for, or None.

    Qwen3.5 / Qwen3.6 (qwen3_5, qwen3_5_moe): linear-attention layers take a mask built from
    cache[ssm_idx], full-attention layers one built from cache[fa_idx], and both indices are the
    first layer of their kind, so each mask exists before any layer needs it.
    """
    try:
        from mlx_lm.models.qwen3_5 import Qwen3_5TextModel
    except ImportError:
        return None
    m = getattr(getattr(model, "language_model", None), "model", None)
    if not isinstance(m, Qwen3_5TextModel) or _forward_source(type(m)) != _KNOWN_FORWARD:
        return None
    kinds = [layer.is_linear for layer in m.layers]
    if True not in kinds or False not in kinds:
        return None
    if kinds.index(True) != m.ssm_idx or kinds.index(False) != m.fa_idx:
        return None
    return m


# The forward pass this schedule reproduces, as mlx_lm 0.31 writes it (embed, one fa mask from
# cache[fa_idx], one ssm mask from cache[ssm_idx], every layer in order, norm). If an mlx_lm update
# changes that body, layer-major prefill declines and mlx_lm's own loop reads the prompt, until
# someone checks the new body and updates this hash (tests/test_layered.py prints it).
_KNOWN_FORWARD = "99109386bc59da7e79abf9e21efc42200f44b58d394b8161aa24b55a4f08eb35"


@functools.lru_cache(maxsize=None)
def _forward_source(cls) -> str:
    """sha256 of the forward pass `cls` actually runs, whitespace normalised; "" if it cannot be read.

    The class's own `__call__` (a subclass that overrides it hashes differently), and "" for a
    wrapped one: getsource follows __wrapped__ and would hash the original, not the wrapper.
    Cached per class; the source does not change while the process runs.
    """
    fn = cls.__call__
    if hasattr(fn, "__wrapped__"):
        return ""
    try:
        src = inspect.getsource(fn)
    except (OSError, TypeError):
        return ""
    return hashlib.sha256(" ".join(src.split()).encode()).hexdigest()


def prefill(model, tokens, cache, step: int, progress=None, layer_scope=None, settle=None) -> bool:
    """Run `tokens` through `model` into `cache`, layer by layer. False, untouched, if unsupported.

    `tokens` is everything the chunked loop would have prefilled (the prompt minus its last
    token, which generation then reads on its own). `step` is the chunk size, counted from the
    first token as the chunked loop counts it.

    `progress(done, total)` is called after each layer with the share of the work finished,
    scaled to tokens.

    `layer_scope(i)`, if given, returns a context manager held while layer i reads every chunk;
    the engine uses it to keep that layer's expert arrays between chunks and to read the next
    layer ahead. It must not change what a layer computes.

    `settle(cache, i)`, if given, runs after each chunk of layer i and may replace cache[i]. The
    chunked loops quantize the KV cache between chunks, entry by entry, each on its own offset,
    so applying the same rule to layer i's entry after the same chunk reproduces it exactly.
    """
    m = text_model(model)
    if m is None or len(tokens) == 0 or step < 1 or len(cache) != len(m.layers):
        return False
    hidden = [m.embed_tokens(mx.array(tokens[i:i + step])[None]) for i in range(0, len(tokens), step)]
    masks = {}

    def run(li, j):
        """Chunk j through layer li, with the mask the chunked forward would build for it."""
        layer = m.layers[li]
        if li == m.ssm_idx:
            masks["ssm", j] = create_ssm_mask(hidden[j], cache[m.ssm_idx])
        if li == m.fa_idx:
            masks["fa", j] = create_attention_mask(hidden[j], cache[m.fa_idx])
        hidden[j] = layer(hidden[j], mask=masks["ssm" if layer.is_linear else "fa", j], cache=cache[li])
        if settle is not None:
            settle(cache, li)                    # may replace cache[li], as the chunked loop would
        # Materialise each chunk's output and this layer's cache before the next step, so the lazy
        # graph never spans more than one (layer, chunk) step.
        mx.eval(hidden[j], cache[li].state)

    n = len(m.layers)
    for li in range(n):
        with layer_scope(li) if layer_scope is not None else contextlib.nullcontext():
            for j in range(len(hidden)):
                run(li, j)
        mx.clear_cache()                          # as the chunked loop does after every chunk
        if progress is not None:
            progress(len(tokens) * (li + 1) // n, len(tokens))
    return True
