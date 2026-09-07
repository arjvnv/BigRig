"""How many bytes one generated token adds to the KV cache, from config.json alone.

Standard library only. No mlx, no model, no download. `verify.py` beside this file checks the
answer against what mlx_lm's own cache objects allocate, for every model config on the machine.

The usual formula is

    num_hidden_layers * num_key_value_heads * head_dim * 2 * 2

which is right for Llama shaped attention and wrong for five families now in wide use, always in
the direction that reserves memory nothing will ever use:

    Kimi-Linear-48B      30.9x too high
    DeepSeek-V3          24.9x too high
    Nemotron-3-Nano       8.7x too high
    Qwen3-Next-80B        4.0x too high
    Qwen3.6-35B           4.0x too high

Five shapes have to be told apart, and config.json says which:

    latent attention    `kv_lora_rank`, caches one compressed vector per layer, not per head
    sliding windows     `layer_types` holding "sliding_attention", those layers stop growing
    linear attention    `full_attention_interval`, `linear_attn_config.kda_layers`, or
                        `layer_types` holding "linear_attention", constant state, no growth
    Mamba blocks        `hybrid_override_pattern`, "*" attention, "M" a state, "-" no cache
    ordinary attention  everything else

A bare `sliding_window` field is NOT evidence of a windowed cache. Phi-3.5-MoE and Qwen1.5-MoE
both declare one and neither overrides `make_cache`, so every layer is an ordinary KVCache.
Trusting that field alone prices Phi-3.5-MoE at 17 GB per token.
"""


def kv_bytes_per_token(cfg: dict) -> tuple:
    """Return (growing bytes per token, fixed bytes) for a Hugging Face config dict.

    The second number is the ceiling cost of layers whose cache stops growing, charged once.
    Returns (0, 0) rather than raising when the config does not carry what is needed.
    """
    try:
        L = int(cfg["num_hidden_layers"])
    except (KeyError, TypeError, ValueError):
        return 0, 0

    # Per layer cost of one attention layer, in bytes per token, at two bytes an element.
    if cfg.get("kv_lora_rank"):
        # One latent and one rope key, one head each. No num_key_value_heads term: the point of
        # latent attention is that the per head keys and values are reconstructed from the latent.
        try:
            attn = (int(cfg["kv_lora_rank"]) + int(cfg.get("qk_rope_head_dim") or 0)) * 2
        except (TypeError, ValueError):
            return 0, 0
    else:
        try:
            heads = int(cfg.get("num_key_value_heads") or cfg["num_attention_heads"])
            hd = int(cfg.get("head_dim") or cfg["hidden_size"] // cfg["num_attention_heads"])
        except (KeyError, TypeError, ZeroDivisionError, ValueError):
            return 0, 0
        attn = heads * hd * 2 * 2                      # key and value, two bytes each

    types = cfg.get("layer_types")
    window = cfg.get("sliding_window") or cfg.get("sliding_window_size")
    interval = cfg.get("full_attention_interval")
    kda = (cfg.get("linear_attn_config") or {}).get("kda_layers")
    hybrid = cfg.get("hybrid_layer_pattern")
    blocks = cfg.get("hybrid_override_pattern") or cfg.get("layers_block_type")
    if isinstance(blocks, str):
        blocks = list(blocks)

    if isinstance(blocks, list) and blocks:
        norm = [str(b)[0] for b in blocks]
        full = sum(1 for b in norm if b in ("*", "a"))
        windowed = 0
    elif isinstance(types, list) and types:
        full = sum(1 for t in types if t == "full_attention")
        windowed = sum(1 for t in types if t == "sliding_attention")
    elif isinstance(hybrid, list) and hybrid:          # mimo: 1 marks a windowed layer
        windowed = sum(1 for v in hybrid if v == 1)
        full = len(hybrid) - windowed
    elif isinstance(kda, list):                        # kimi: listed layers are recurrent, 1 based
        full, windowed = L - sum(1 for i in kda if 1 <= int(i) <= L), 0
    elif interval:                                     # qwen3 next: linear unless (i+1) % iv == 0
        try:
            iv = int(interval)
            full = sum(1 for i in range(L) if (i + 1) % iv == 0)
        except (TypeError, ValueError, ZeroDivisionError):
            full = L
        windowed = 0
    else:
        full, windowed = L, 0

    per_token = full * attn
    fixed = windowed * attn * int(window) if (windowed and window) else 0
    return int(per_token), int(fixed)


def naive_bytes_per_token(cfg: dict) -> int:
    """The formula this replaces, for comparison."""
    try:
        return (int(cfg["num_hidden_layers"])
                * int(cfg.get("num_key_value_heads") or cfg["num_attention_heads"])
                * int(cfg.get("head_dim") or cfg["hidden_size"] // cfg["num_attention_heads"])
                * 2 * 2)
    except (KeyError, TypeError, ZeroDivisionError, ValueError):
        return 0
