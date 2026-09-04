<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/logo-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="assets/logo-light.svg">
  <img alt="bigrig" src="assets/logo-light.svg" width="46%">
</picture>

<br>

![Apple Silicon](https://img.shields.io/badge/platform-Apple%20Silicon-1b1a18?style=flat-square)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-2f6f5e?style=flat-square)
![2092 assertions](https://img.shields.io/badge/tests-2092%20assertions-2f6f5e?style=flat-square)
![Decode bit-identical](https://img.shields.io/badge/decode-bit--identical-2f6f5e?style=flat-square)
![Apache 2.0](https://img.shields.io/badge/licence-Apache--2.0-2f6f5e?style=flat-square)

### Run the Mixture-of-Experts models your Mac cannot hold — unchanged, and fast.

**A 20.4 GB model, served from 9.7 GB of RAM, at 19–21 tokens a second, on a MacBook Air.**
Not a smaller model. Not a more aggressive quantisation. The same weights, byte for byte.

**[Quickstart](docs/quickstart.md)** ·
[Install](docs/install.md) ·
[Models](docs/models.md) ·
[CLI](docs/cli.md) ·
[How it works](docs/HOW-IT-WORKS.md) ·
[Contributing](CONTRIBUTING.md)

</div>

---

## About

A 35B Mixture-of-Experts model uses about 3B of its parameters for any given token — but every
runtime loads all 35B into memory, and then your Mac says no.

bigrig keeps the experts a model is actually using in RAM and reads the rest off the SSD as the
router asks for them. Apple Silicon shares one memory between CPU and GPU, so a cached expert is
handed to the GPU without being copied at all. The result is that **the memory a model needs
stops tracking its size**: it becomes attention plus a working set, which is a small fraction of
the whole.

If a model already fits, bigrig says so and gets out of the way. If it cannot be made to fit at
any setting, it says that too, before you download 40 GB to find out.

- **Your weights, untouched.** Streaming changes where a weight lives, never what it is. Decode
  is bit-identical to the same model held resident — checked layer by layer across 4,392 live
  layer computations. Prompt processing matches to the last bit of a float, or one bit off it:
  the same variation mlx_lm itself shows between two prefill widths.
- **It tells you before you download.** `bigrig doctor <model>` reads a model's shape off the
  hub and answers for *your* Mac — will it run, how much memory it needs, and roughly how fast,
  in tokens per second. Nothing is downloaded and nothing is written.
- **It measures itself on your machine.** The first run of a streamed model spends a minute or
  two finding how many experts to keep resident for the best speed on your hardware, and
  remembers the answer. Every later start is immediate.
- **It asks before it changes anything.** Shrinking a model is the only path that alters weights,
  so it is the only one that needs your agreement — and every run prints which mode it is
  serving, so the precision you are getting is never a surprise.
- **It speaks to the tools you already use.** OpenAI, Anthropic and Responses APIs on one port,
  a one-command hook into Claude Code and Codex, and a self-contained web interface that makes
  no external requests.

---

## Getting started

```bash
pip install bigrig
```

That is the whole install on an Apple Silicon Mac — it brings MLX with it. `rig` and `bigrig`
are the same command.

```bash
bigrig doctor mlx-community/Qwen3.6-35B-A3B-4bit   # will it run here, and how fast?
bigrig serve  mlx-community/Qwen3.6-35B-A3B-4bit   # download it, tune it once, serve it
```

`serve` prints where to find it:

```
  bigrig serving Qwen3.6-35B-A3B-4bit on http://127.0.0.1:8080
  running EXACT at 4-bit, 15% of experts in RAM, the rest streamed from disk
  (unmodified weights; decode bit-identical to the original)

  open  http://127.0.0.1:8080  in a browser to chat and watch quality
  api   127.0.0.1:8080/v1/chat/completions   (OpenAI)
        127.0.0.1:8080/v1/messages           (Anthropic — Claude Code)
        127.0.0.1:8080/v1/responses          (Responses — Codex)
  agent bigrig launch Qwen3.6-35B-A3B-4bit
```

Point a coding agent at it with one command — `bigrig launch <model>`, or
`bigrig launch --agent codex <model>`. Nothing on disk is changed; the environment variables are
set on the agent's process only.

---

## What runs, and how fast

Measured end to end on an M4 MacBook Air with 24 GB, at a 9.7 GB memory ceiling. "Warm" is a
server that has been running; "cold" is one that has just started.

| model | on disk | resident | decode | first token |
|---|---|---|---|---|
| DeepSeek-Coder-V2-Lite-4bit | 8.8 GB | 44% | **25.9 tok/s** | 0.7 s @ 61 tokens |
| Qwen3.6-35B-A3B-4bit | 20.4 GB | 15% | **19–21 warm, 10.5 cold** | 5.1 s @ 453, 12.4 s @ 1,749 |
| Nemotron-3-Nano-30B-A3B-4bit | 17.8 GB | 15% | **11.5–13.2 tok/s** | 3.2 s @ 69 tokens |
| Qwen3-30B-A3B-3bit | 17.2 GB | — | **10–14.5 tok/s** | — |
| GLM-4.7-Flash-4bit | 16.9 GB | 8% | **6.2–6.6 tok/s** | 9.2 s @ 409 tokens |
| OLMoE-1B-7B-4bit | 3.9 GB | 100% | fits whole, no streaming | — |

Conversations reuse what has already been read: a follow-up turn on a 900-token document
reached its first token in **1.1 s instead of 11.3 s**.

**Beyond what we have run here**, `doctor` computes the floor from the model's own shape. It puts
gpt-oss-120b (65.8 GB) at 7.2 GB and Qwen3-Next-80B-A3B (45 GB) at 6.2 GB — the same floor class
as the 20 GB models above, because a model's floor is set by its attention weights and one layer's
worth of experts, not by its total size. Those two are predictions from measured arithmetic, not
runs; `doctor` labels them as such, and so do we.

---

## Which models

Any **Mixture-of-Experts** model in MLX format whose experts are stacked per layer. That is a
layout, not a list of blessed names: eight expert paths and both MLP shapes are recognised — the
three-projection SwiGLU that Qwen, DeepSeek, GLM and gpt-oss use, and the two-projection
`fc1`/`fc2` that Nemotron uses. Expert count, top-k, quantisation and layer path are read from
the checkpoint, never assumed.

Six families are measured end to end (the table above). `doctor` has read and answered for
checkpoints from Qwen3, Qwen3.6, Qwen3-Next, Qwen3-Coder, DeepSeek-V2 and V3, GLM-4.5 through
4.7, gpt-oss, Kimi-Linear, MiniMax, Hunyuan, ERNIE, Granite, Phi-MoE and LongCat — from 4 GB up
to 378 GB. Mixtral is the one layout that is refused: it stores each expert as separate tensors
rather than stacked, so there is no single range to read an expert from, and `doctor` says so
plainly instead of calling it "not an MoE model".

Multimodal checkpoints load text-only; the vision tower is neither loaded nor charged to memory.
Dense (non-MoE) models are not the target — there is nothing to keep resident selectively.

Full list, and how the speed prediction works: **[docs/models.md](docs/models.md)**.

---

## What you need

- **An Apple Silicon Mac** (M1 or later) and macOS. MLX is Apple-only; this engine is built on it.
- **Python 3.10+.**
- **Memory.** By default bigrig plans to use 35% of installed RAM so the rest of your Mac keeps
  working, and it never exceeds that. Most 17–20 GB models need a 6–7 GB ceiling, which that
  default clears on a 24 GB Mac. On a 16 GB Mac the default leaves 5.6 GB, so raise it —
  `doctor` prints the exact figure and the command (`BIGRIG_MAX_GB=7 bigrig run <model>`).
- **Disk.** The model, plus one more copy of its experts for a streamed model's fast path. That
  copy doubles the model's footprint; the size is printed before it is made, it is skipped when
  the disk is tight, and `--no-pack` declines it. A model that fits in RAM writes nothing extra.

There is no GPU to configure, no build step, and nothing is sent anywhere — the engine, the
server and the web interface all run entirely on your machine.

---

## Licence

[Apache License 2.0](LICENSE). Use it, modify it, ship it commercially — keep the notice and
don't sue us over patents.
