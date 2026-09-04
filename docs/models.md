# Models

[← BigRig](../README.md) · [Install](install.md) · [Quickstart](quickstart.md) · [Models](models.md) · [CLI](cli.md) · [How it works](HOW-IT-WORKS.md)

---

## What works

Any **Mixture-of-Experts** model in MLX format whose experts are stacked per layer, in either of
the two shapes checkpoints actually use: a three-projection gate/up/down SwiGLU MLP (Qwen,
DeepSeek, GLM, gpt-oss, Kimi) or a two-projection `fc1`/`fc2` MLP (Nemotron). Expert count,
top-k, quantisation and the layer path (`mlp.switch_mlp`, `mlp.experts`, `ffn.*`,
`mixer.switch_mlp`, `block_sparse_moe.*`, ...) are read from the checkpoint, never assumed. Multimodal checkpoints load text-only; the vision tower is
neither loaded nor charged to memory.

**Run on this machine** (M4, 24 GB, 9.7 GB ceiling), the model's own weights untouched (decode
bit-identical to a resident run; prompt processing matches to the last bit of a float or one
bit off it, the same variation mlx_lm shows between two prefill widths). Decode speed on a
streamed model depends on whether its experts are in the page cache: "warm" is a server that has
been running, "cold" is one that has just started or a cache churned by other work.

| model | experts | top-k | layers | measured |
|---|---|---|---|---|
| OLMoE-1B-7B | 64 | 8 | 16 | fits whole |
| Qwen3-30B-A3B-3bit | 128 | 8 | 48 | 10 tok/s cold (4 whole layers + 9 of 128); up to 14.5 measured with whole layers, warm |
| Qwen3.6-35B-A3B-4bit | 256 | 8 | 40 | **21 tok/s warm, 10.5 cold** (4 whole layers + 13 of 256, packed); first token in 5 s for a 450-token prompt, 12 s for 1,750 |
| Qwen3.6-35B-A3B-8bit | 256 | 8 | 40 | 1.6 tok/s at 4.7% on the old copy path (2026-09-01); not re-measured on the zero-copy path, predicted roughly 5-9 cold |
| Qwen3-MOE-4×0.6B | 4 | 2 | 28 | fits whole |
| DeepSeek-Coder-V2-Lite-4bit | 64 | 6 | 26 | **25.9 tok/s** (8 whole layers + 8 of 64, 44% resident); first token in 0.7 s for a 61-token prompt |
| Nemotron-3-Nano-30B-A3B-4bit | 128 | 6 | 23 | **11.5–13.2 tok/s** (1 whole layer + 8 of 128, 15% resident); first token in 3.2–3.4 s for a 69-token prompt |
| GLM-4.7-Flash-4bit | 64 | 4 | 46 | **6.2–6.6 tok/s** (2 whole layers + 5 of 64, 7.8% resident); first token in 3.7 / 9.2 / 23.8 s at 52 / 409 / 1,309 prompt tokens |

**Shape verified with `doctor`** (read from the hub, not downloaded or run here):

| model | download | doctor says on a 24 GB Mac |
|---|---|---|
| DeepSeek-V4-Flash-4bit | 151.5 GB | needs 12.1 GB; a choice, and SLOW at 2% |
| Kimi-K2.5 | 657.6 GB | IMPOSSIBLE — needs 41 GB at any setting |

**Recognised but not streamable yet:**

| model | why |
|---|---|
| Mixtral (all sizes) | its experts are stored one per expert as separate `w1`/`w2`/`w3` tensors rather than stacked per layer, so there is no single range to read an expert from. `doctor` says so rather than calling it "not an MoE model". It would have to fit whole. On the list. |

Dense (non-MoE) models are not the target: there is nothing to keep resident selectively.

## Where to find them

[mlx-community](https://huggingface.co/mlx-community) on Hugging Face. Pass the repo id straight
to `prepare`, `serve` or `run`.

## Choosing a size

`rig doctor <model>` tells you which of the three modes it would use, and for a streamed model a
speed word that is a prediction and says so. The prediction is the expert bytes one token must
read (an assumed 0.6 miss rate, measured 0.53-0.61 across 7-30% residency, times top-k, streamed
layers and bytes per expert) divided by how fast your Mac moves them: 0.65 of its calibrated disk
when the page cache is cold, about 6.4 GB/s of effective traffic when warm. The word is the cold
number's word; the sentence gives the range:

| predicted tok/s | says | means |
|---|---|---|
| ≥ 15 | FAST | fast enough for a coding agent |
| 8–15 | GOOD | comfortable for chat, workable for an agent |
| 3–8 | USABLE | fine for chat, slow for an agent |
| < 3 | SLOW | it runs, but you will be waiting on it |

What moves the number is bytes per token, so **a 4-bit build of the same model roughly halves
it** against 8-bit, and whole layers (which never miss) take layers out of the sum. Residency by
itself barely moves the miss rate in the range a small Mac reaches, which is why a slightly
higher ceiling often changes nothing. The first run measures the exact best setting for your
Mac, and the table above is what those predictions were checked against.

## Quantisation

Most MLX models ship at 4-bit. BigRig will shrink a model further **only with your agreement**,
and never below 3 bits by default — 2-bit measured at +83% to +145% perplexity, and produced
unusable output on a real model. `--min-bits 2` overrides that if you want to see it.

## Custom code

Some checkpoints ship their own Python. It is **not** downloaded unless you pass
`--trust-remote-code`, because running an unfamiliar script is not something a model runner
should do on your behalf.
