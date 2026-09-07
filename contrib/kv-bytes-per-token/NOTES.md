# KV bytes per token, by architecture family

## What this is

`kv_bytes.py` answers one question from `config.json` alone: how many bytes does one generated
token add to the KV cache, and how many bytes do the layers that stop growing cost once.

`verify.py` checks that answer against what `mlx_lm`'s own `KVCache` and `RotatingKVCache`
actually allocate. It is not a second copy of the same formula. Families with no config on the
machine are printed as UNCHECKED rather than quietly skipped.

## Current result on this machine

21 families priced exactly, zero mismatches: bailing_moe, deepseek_v2, deepseek_v3, ernie4_5_moe,
glm4_moe, glm4_moe_lite, gpt_oss, hunyuan, kimi_linear, llama, minimax, mixtral, nemotron_h,
olmoe, phimoe, qwen2_moe, qwen3, qwen3_5_moe, qwen3_moe, qwen3_next, qwen3_vl_moe.

What the usual formula gets wrong, measured:

| model | naive is | why |
|---|---|---|
| Kimi-Linear-48B | 30.9x too high | most layers are linear attention, constant state |
| DeepSeek-V3 | 24.9x too high | latent attention caches one vector, not one per head |
| Nemotron-3-Nano | 8.7x too high | 52 blocks, only 6 are attention |
| Qwen3-Next-80B | 4.0x too high | one attention layer in four |
| Qwen3.5-122B | 4.0x too high | same shape |
| gpt-oss-120b | 2.0x too high | half the layers are windowed and stop growing |

Overestimating is not the safe direction. It reserves memory that will never be used, which
shortens the reply ceiling and starves everything else, and nothing fails loudly.

## Where this should go

**mlx-lm, as a utility.** `mlx_lm/models/cache.py` has `make_prompt_cache`, `save`, `load`,
`trim`, and no way to ask what a cache will cost before building one. "Will this model fit"
is a constant question and there is nothing to answer it with. A `kv_bytes_per_token(config)`
beside `make_prompt_cache` is a small, self contained addition, and `verify.py` is the test that
holds it honest against the cache classes themselves.

**vllm-metal #644, for the Nemotron-H part.** That issue asks for Nemotron-H, a Mamba2 and MoE
hybrid, on the paged attention path. The relevant fact is that `hybrid_override_pattern` names
every block, `*` is attention, `M` is a recurrent state and `-` has no cache at all, so only 6 of
Nemotron-3-Nano's 52 blocks hold a growing KV cache. Anything that counts `num_hidden_layers`
prices it 8.7x high.

## Correction to an earlier reading of the issue list

vllm-metal #700 is about keeping up with vLLM's own KV cache refactors, not about per family byte
accounting. It is not the right home for this. The two above are.

## What is deliberately not here

The pool sizing and reply capping policy that consumes these numbers. This is the measurement,
not what to do with it.
