"""What a token costs in KV cache, checked against every architecture family on this machine.

WHY THIS FILE EXISTS
    The engine sizes the expert pool and caps the reply length from one number: how many bytes
    each generated token adds to the KV cache. That number was `n_layers * num_key_value_heads *
    head_dim * 2 * 2` -- true of Llama-shaped attention, and wrong for four families this engine
    claims to serve, always in the direction that hurts:

        Kimi-Linear-48B     30.9x too high      2.04 GB reserved at 8k tokens, 0.07 GB needed
        DeepSeek-V3         24.9x too high     14.33 GB reserved at 8k tokens, 0.58 GB needed
        Ling 2.6 flash      14.2x too high
        Qwen3-Next-80B       4.0x too high
        Qwen3.6-35B          4.0x too high

    An overestimate is not a safe error here. DeepSeek-V3 was being charged 14.33 GB of KV cache
    for an 8,000-token reply against a 9 GB budget, so the reply ceiling collapsed and the pool
    was starved of memory that was never going to be used. The model would have run, badly, and
    nothing would have failed loudly enough to notice.

HOW THE NUMBERS ARE CHECKED
    Not by re-deriving the formula, which would only test that two copies of the same reasoning
    agree. mlx_lm's own cache objects are built for real and fed the shapes the model source
    demonstrably passes them, and their bytes are counted. A config on this machine is a real
    published config, not a fixture written to match.

    KVCache allocates in blocks of `step` (256), so every measurement is taken at a whole number
    of steps -- otherwise a 64-token probe reads 4x high and every family looks broken.
"""
import glob
import json
import os
import sys

import mlx.core as mx
from mlx_lm.models.cache import KVCache, RotatingKVCache

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bigrig_engine.session import kv_bytes          # noqa: E402

FAIL = []
N = 256                                               # a whole number of KVCache steps


def check(n, c, d=""):
    (print(f"  PASS  {n}") if c else (FAIL.append(n), print(f"  FAIL  {n}  {d}")))


def configs():
    for p in glob.glob(os.path.expanduser(
            "~/.cache/huggingface/hub/models--*/snapshots/*/config.json")):
        try:
            c = json.load(open(p))
        except (OSError, ValueError):
            continue
        yield p.split("models--")[1].split("/snapshots")[0].replace("--", "/"), \
            {**c.get("text_config", {}), **c}


def kinds_of(c, L):
    """Which cache mlx_lm builds for each layer, read off the same fields its models read."""
    types = c.get("layer_types")
    iv = c.get("full_attention_interval")
    kda = (c.get("linear_attn_config") or {}).get("kda_layers")
    hybrid = c.get("hybrid_layer_pattern")
    blocks = c.get("hybrid_override_pattern") or c.get("layers_block_type")
    if isinstance(blocks, str):
        blocks = list(blocks)
    if isinstance(blocks, list) and blocks:
        # mlx_lm's nemotron_h builds a cache ONLY for "M" and "*" blocks; "-" gets none.
        return [{"*": "kv", "a": "kv", "M": "lin", "m": "lin"}.get(str(b)[0], "skip")
                for b in blocks]
    if isinstance(types, list) and types:
        return ["kv" if t == "full_attention"
                else "win" if t == "sliding_attention" else "lin" for t in types]
    if isinstance(hybrid, list) and hybrid:
        return ["win" if v == 1 else "kv" for v in hybrid]
    if isinstance(kda, list):
        return ["lin" if (i + 1) in kda else "kv" for i in range(L)]
    if iv:
        return ["kv" if (i + 1) % int(iv) == 0 else "lin" for i in range(L)]
    return ["kv"] * L


def measured(c):
    """Bytes mlx_lm's own cache objects hold for N tokens of this config."""
    L = int(c["num_hidden_layers"])
    _b = c.get("hybrid_override_pattern") or c.get("layers_block_type")
    if _b:
        L = len(_b)
    if c.get("kv_lora_rank"):
        ks = (1, 1, N, int(c["kv_lora_rank"]))
        vs = (1, 1, N, int(c.get("qk_rope_head_dim") or 0))
    else:
        h = int(c.get("num_key_value_heads") or c["num_attention_heads"])
        hd = int(c.get("head_dim") or c["hidden_size"] // c["num_attention_heads"])
        ks = vs = (1, h, N, hd)
    win = int(c.get("sliding_window") or c.get("sliding_window_size") or 0)
    total = 0
    for kind in kinds_of(c, L):
        if kind in ("lin", "skip"):
            continue                                  # no cache, or a fixed state that never grows
        if kind == "kv":
            cache = KVCache()
            cache.update_and_fetch(mx.zeros(ks, dtype=mx.float16),
                                   mx.zeros(vs, dtype=mx.float16))
        else:
            # A WINDOWED LAYER MUST BE FED THE WAY GENERATION FEEDS IT.
            #     RotatingKVCache keeps everything handed to it in ONE call and only rotates on
            #     later ones, so a single N-token push reports N tokens however small the window
            #     is -- 256 tokens in a 128-token window, measured. Pushed one token at a time it
            #     settles at 128, which is what a reply of any length actually costs.
            #
            #     This mattered the moment a gpt-oss config entered the local cache: the helper
            #     priced its 18 windowed layers as though they were full, said 18,874,368 against
            #     the engine's 14,155,776, and the engine was right. mlx_lm's own gpt_oss
            #     make_cache builds RotatingKVCache(max_size=sliding_window) for exactly those
            #     layers.
            cache = RotatingKVCache(max_size=win)
            step_k = (ks[0], ks[1], 1, ks[3])
            step_v = (vs[0], vs[1], 1, vs[3])
            for _ in range(N):
                cache.update_and_fetch(mx.zeros(step_k, dtype=mx.float16),
                                       mx.zeros(step_v, dtype=mx.float16))
        total += cache.keys.nbytes + cache.values.nbytes
    return total


print("=" * 84)
print("EVERY ARCHITECTURE FAMILY ON THIS MACHINE IS PRICED THE WAY mlx_lm ALLOCATES")
print("=" * 84)

FAMILIES = ("deepseek_v3", "deepseek_v2", "qwen3_moe", "qwen3_next", "qwen3_5_moe", "qwen2_moe",
            "glm4_moe", "gpt_oss", "kimi_linear", "olmoe", "mixtral", "phimoe", "bailing_moe",
            "ernie4_5_moe", "qwen3_vl_moe", "nemotron_h")
seen = set()
for name, c in configs():
    mt = c.get("model_type", "")
    if mt not in FAMILIES or mt in seen or "num_hidden_layers" not in c:
        continue
    seen.add(mt)
    per, fixed = kv_bytes(c)
    try:
        truth = measured(c)
    except (KeyError, TypeError, ZeroDivisionError, ValueError) as e:
        check(f"{mt}: mlx_lm's caches can be built", False, str(e))
        continue
    check(f"{mt:14} priced as mlx_lm allocates ({name.split('/')[-1][:34]})",
          per * N + fixed == truth, f"ours {per * N + fixed:,} vs mlx {truth:,}")

check("at least eight distinct families were checked", len(seen) >= 8, f"only {sorted(seen)}")

print()
print("=" * 84)
print("THE SHAPES THAT USED TO BE PRICED AS ORDINARY ATTENTION")
print("=" * 84)
# Each of these is the arithmetic the old formula would have produced, against the new one.
CASES = [
    ("multi-head latent attention (DeepSeek-V3)",
     {"num_hidden_layers": 61, "kv_lora_rank": 512, "qk_rope_head_dim": 64,
      "num_key_value_heads": 128, "num_attention_heads": 128, "hidden_size": 7168}, 24.9),
    ("linear-attention hybrid (Qwen3-Next)",
     {"num_hidden_layers": 48, "full_attention_interval": 4, "num_key_value_heads": 2,
      "head_dim": 256, "num_attention_heads": 16}, 4.0),
]
for label, cfg, expect in CASES:
    per, _f = kv_bytes(cfg)
    old = (int(cfg["num_hidden_layers"])
           * int(cfg.get("num_key_value_heads") or cfg["num_attention_heads"])
           * int(cfg.get("head_dim") or cfg["hidden_size"] // cfg["num_attention_heads"]) * 2 * 2)
    check(f"{label} was {expect}x too expensive", round(old / per, 1) == expect,
          f"got {round(old / per, 1)}x")

check("a config with nothing usable in it returns zero rather than raising",
      kv_bytes({}) == (0, 0) and kv_bytes({"num_hidden_layers": 3}) == (0, 0))
check("a fixed cost is reported separately from a per-token one, never folded in",
      kv_bytes({"num_hidden_layers": 2, "num_key_value_heads": 1, "head_dim": 8,
                "num_attention_heads": 1, "sliding_window": 64,
                "layer_types": ["full_attention", "sliding_attention"]})
      == (1 * 8 * 2 * 2, 1 * 8 * 2 * 2 * 64))

print()
print("=" * 84)
print("ALL TESTS PASSED" if not FAIL else f"{len(FAIL)} FAILURES: " + ", ".join(FAIL))
print("=" * 84)
sys.exit(1 if FAIL else 0)
