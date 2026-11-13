"""Layer-major prefill: push a long prompt through the model one LAYER at a time, not one chunk at a time.

WHY
    mlx_lm reads a prompt in chunks of `prefill_step` tokens, and each chunk runs through all 40
    layers before the next chunk starts. With streamed experts that order is expensive: every
    layer's experts are fetched once per chunk, and between two visits to the same layer the
    other 39 layers' experts have pushed it out of the page cache. A 1,771-token prompt is 14
    chunks, so a layer's experts can come off the SSD up to 14 times. Processing every chunk
    through layer L before any chunk reaches layer L+1 visits each layer once, back to back, so
    after the first chunk its experts are already in memory.

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
    if not isinstance(m, Qwen3_5TextModel):
        return None
    kinds = [layer.is_linear for layer in m.layers]
    if kinds.index(True) != m.ssm_idx or kinds.index(False) != m.fa_idx:
        return None
    return m


def prefill(model, tokens, cache, step: int, progress=None, settle=None) -> bool:
    """Run `tokens` through `model` into `cache`, layer by layer. False, untouched, if unsupported.

    `tokens` is everything the chunked loop would have prefilled (the prompt minus its last
    token, which generation then reads on its own). `progress(done, total)` is called after each
    layer with the share of the work finished, scaled to tokens. `settle(cache, i)`, if given, runs after each chunk of layer i and may replace cache[i]: the
    chunked loops quantize the KV cache between chunks, entry by entry, each on its own offset,
    so applying the same rule to layer i's entry after the same chunk reproduces it exactly.
    """
    m = text_model(model)
    if m is None or len(tokens) == 0 or step < 1 or len(cache) != len(m.layers):
        return False
    hidden = [m.embed_tokens(mx.array(tokens[i:i + step])[None]) for i in range(0, len(tokens), step)]
    masks = {}
    for li, layer in enumerate(m.layers):
        kind = "ssm" if layer.is_linear else "fa"
        for j, h in enumerate(hidden):
            # Built from the same cache, in the same state, as the chunked forward builds it.
            if li == m.ssm_idx:
                masks["ssm", j] = create_ssm_mask(h, cache[m.ssm_idx])
            if li == m.fa_idx:
                masks["fa", j] = create_attention_mask(h, cache[m.fa_idx])
            hidden[j] = layer(h, mask=masks[kind, j], cache=cache[li])
            if settle is not None:
                settle(cache, li)                # may replace cache[li], as the chunked loop would
            # Materialise each chunk's output and this layer's cache before the next chunk, so the
            # lazy graph never spans more than one (layer, chunk) step.
            mx.eval(hidden[j], cache[li].state)
        if progress is not None:
            progress(len(tokens) * (li + 1) // len(m.layers), len(tokens))
    mx.clear_cache()
    return True
