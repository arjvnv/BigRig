"""Check kv_bytes_per_token against what mlx_lm's own cache objects allocate.

Not against a re-derivation of the same formula, which would only prove that two copies of one
piece of reasoning agree. mlx_lm's KVCache and RotatingKVCache are built for real, fed the shapes
the model source demonstrably passes them, and their bytes counted.

Configs are read from the local Hugging Face cache and from any directory given on the command
line. Nothing is downloaded. Families with no config on this machine are reported as UNCHECKED,
never silently counted as passing.

Run:  python verify.py [extra-model-dir ...]
"""
import glob
import json
import os
import sys

import mlx.core as mx
from mlx_lm.models.cache import KVCache, RotatingKVCache

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kv_bytes import kv_bytes_per_token, naive_bytes_per_token       # noqa: E402

N = 256                    # a whole number of KVCache steps; KVCache allocates in blocks of 256
FAIL, CHECKED = [], set()

FAMILIES = ("deepseek_v3", "deepseek_v2", "qwen3_moe", "qwen3_next", "qwen3_5_moe", "qwen2_moe",
            "glm4_moe", "glm4_moe_lite", "gpt_oss", "kimi_linear", "olmoe", "mixtral", "phimoe",
            "bailing_moe", "ernie4_5_moe", "qwen3_vl_moe", "nemotron_h", "minimax", "hunyuan",
            "llama", "qwen3", "mistral")


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + ("" if cond else f"   {detail}"))
    if not cond:
        FAIL.append(name)


def configs(extra_dirs):
    pats = [os.path.expanduser("~/.cache/huggingface/hub/models--*/snapshots/*/config.json")]
    for d in extra_dirs:
        pats += [os.path.join(d, "config.json"), os.path.join(d, "*", "config.json")]
    for p in sorted(set(sum((glob.glob(x) for x in pats), []))):
        try:
            c = json.load(open(p))
        except (OSError, ValueError):
            continue
        name = (p.split("models--")[1].split("/snapshots")[0].replace("--", "/")
                if "models--" in p else os.path.basename(os.path.dirname(p)))
        yield name, {**c.get("text_config", {}), **c}


def kinds_of(c, L):
    """Which cache mlx_lm builds per layer, read off the same fields its model files read."""
    types = c.get("layer_types")
    iv = c.get("full_attention_interval")
    kda = (c.get("linear_attn_config") or {}).get("kda_layers")
    hybrid = c.get("hybrid_layer_pattern")
    blocks = c.get("hybrid_override_pattern") or c.get("layers_block_type")
    if isinstance(blocks, str):
        blocks = list(blocks)
    if isinstance(blocks, list) and blocks:
        # nemotron_h builds a cache only for "M" and "*" blocks; "-" gets none at all.
        return [{"*": "kv", "a": "kv", "M": "lin", "m": "lin"}.get(str(b)[0], "skip") for b in blocks]
    if isinstance(types, list) and types:
        return ["kv" if t == "full_attention" else "win" if t == "sliding_attention" else "lin"
                for t in types]
    if isinstance(hybrid, list) and hybrid:
        return ["win" if v == 1 else "kv" for v in hybrid]
    if isinstance(kda, list):
        return ["lin" if (i + 1) in kda else "kv" for i in range(L)]
    if iv:
        return ["kv" if (i + 1) % int(iv) == 0 else "lin" for i in range(L)]
    return ["kv"] * L


def measured(c):
    """Bytes mlx_lm's own cache objects hold after N tokens of this config."""
    L = int(c["num_hidden_layers"])
    blocks = c.get("hybrid_override_pattern") or c.get("layers_block_type")
    if blocks:
        L = len(blocks)
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
            continue                          # no cache, or a fixed state that never grows
        if kind == "kv":
            cache = KVCache()
            cache.update_and_fetch(mx.zeros(ks, dtype=mx.float16), mx.zeros(vs, dtype=mx.float16))
        else:
            # A WINDOWED LAYER MUST BE FED THE WAY GENERATION FEEDS IT. RotatingKVCache keeps
            # everything handed to it in one call and only rotates on later ones, so a single
            # N token push reports N tokens however small the window is. One token at a time it
            # settles at the window, which is what a reply of any length actually costs.
            cache = RotatingKVCache(max_size=win)
            for _ in range(N):
                cache.update_and_fetch(mx.zeros((ks[0], ks[1], 1, ks[3]), dtype=mx.float16),
                                       mx.zeros((vs[0], vs[1], 1, vs[3]), dtype=mx.float16))
        total += cache.keys.nbytes + cache.values.nbytes
    return total


print("=" * 96)
print("kv_bytes_per_token vs what mlx_lm's own cache objects allocate, on this machine")
print("=" * 96)
print(f"  {'family':16s} {'model':38s} {'ours (B/tok)':>13s} {'naive':>12s} {'naive is':>9s}")
rows = []
for name, c in configs(sys.argv[1:]):
    mt = c.get("model_type", "")
    if mt not in FAMILIES or mt in CHECKED or "num_hidden_layers" not in c:
        continue
    try:
        truth = measured(c)
    except (KeyError, TypeError, ZeroDivisionError, ValueError):
        continue
    CHECKED.add(mt)
    per, fixed = kv_bytes_per_token(c)
    naive = naive_bytes_per_token(c)
    rows.append((mt, name.split("/")[-1][:38], per, naive, truth, fixed))
    over = f"{naive / per:.1f}x" if per else "n/a"
    print(f"  {mt:16s} {name.split('/')[-1][:38]:38s} {per:13,d} {naive:12,d} {over:>9s}")

print()
for mt, short, per, naive, truth, fixed in rows:
    check(f"{mt:16s} priced exactly as mlx_lm allocates ({short[:30]})",
          per * N + fixed == truth, f"ours {per * N + fixed:,} vs mlx_lm {truth:,}")

missing = [f for f in FAMILIES if f not in CHECKED]
print()
print(f"  checked {len(CHECKED)} families: {', '.join(sorted(CHECKED))}")
if missing:
    print(f"  UNCHECKED, no config on this machine: {', '.join(missing)}")

print()
print("=" * 96)
print("ALL PASSED" if not FAIL else f"{len(FAIL)} FAILURES: " + ", ".join(FAIL))
print("=" * 96)
raise SystemExit(1 if FAIL else 0)
