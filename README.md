<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/logo-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="assets/logo-light.svg">
  <img alt="BigRig" src="assets/logo-light.svg" width="46%">
</picture>

<br>

![Apple Silicon](https://img.shields.io/badge/platform-Apple%20Silicon-1b1a18?style=flat-square)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-2f6f5e?style=flat-square)
![2092 assertions](https://img.shields.io/badge/tests-2092%20assertions-2f6f5e?style=flat-square)
![Decode bit-identical](https://img.shields.io/badge/decode-bit--identical-2f6f5e?style=flat-square)
![Apache 2.0](https://img.shields.io/badge/licence-Apache--2.0-2f6f5e?style=flat-square)

### Run the Mixture-of-Experts models your Mac cannot hold — unchanged, and fast.

Your weights are never modified, and BigRig tells you exactly what *your* machine
can do with a model before you download a single byte of it.

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

BigRig keeps the experts a model is actually using in RAM and reads the rest off the SSD as the
router asks for them. Apple Silicon shares one memory between CPU and GPU, so a cached expert is
handed to the GPU without being copied at all. The result is that **the memory a model needs
stops tracking its size.** What has to stay resident is the attention weights plus one layer's
worth of experts, and that is a small fraction of the whole.

In practice a 17–20 GB model runs in six or seven gigabytes, and the bigger the model the wider
that gap gets — `doctor` puts gpt-oss-120b, 65.8 GB on disk, at the same seven gigabytes.

If a model already fits, BigRig says so and gets out of the way. If it cannot be made to fit at
any setting, it says that too, before you download 40 GB to find out.

- **Your weights, untouched.** Streaming changes where a weight lives, never what it is. Decode
  is bit-identical to the same model held resident — checked layer by layer across 4,392 live
  layer computations. Prompt processing matches to the last bit of a float, or one bit off it:
  the same variation mlx_lm itself shows between two prefill widths.
- **It answers for your machine, not ours.** Speed depends on your chip and your SSD, so BigRig
  measures rather than guesses. `bigrig doctor <model>` reads the model's shape from the hub,
  profiles your Mac, and tells you whether it runs, what memory it needs and roughly how many
  tokens a second you will see. Nothing is downloaded and nothing is written.
- **It tunes itself where it is installed.** The first run of a streamed model spends a minute
  or two finding how many experts to keep resident for the best speed on your hardware, and
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

## How fast, on your Mac

An M1 Air and an M4 Max are not the same machine, and a model that is comfortable on one is not
on the other. So BigRig does not publish a number and invite you to hope it applies. `doctor`
profiles the memory and disk bandwidth of the Mac it is running on, works out how many bytes each
token has to move for the model you named, and answers in one of four words:

| verdict | tokens/s | what it means |
|---|---|---|
| **FAST** | ≥ 15 | fast enough for a coding agent |
| **GOOD** | 8–15 | comfortable for chat, workable for an agent |
| **USABLE** | 3–8 | fine for chat, slow for an agent |
| **SLOW** | < 3 | it runs, but you will be waiting on it |

You get that answer, and the memory figure behind it, before anything is downloaded.

Two things shift the verdict, and neither is obvious. **A model with more, smaller experts moves
fewer bytes a token than one with fewer, larger ones**, so the larger of two models is regularly
the faster one. And **conversations get cheaper as they go**: what has already been read is reused, so
a follow-up turn reaches its first token roughly ten times sooner than the one that opened the
document.

Six families have been run end to end during development, from 3.9 GB to 20.4 GB on disk, at
memory ceilings under 10 GB — the fastest of them at 25.9 tokens a second, the slowest at 6.2.
Per-model figures, and the machine they were taken on, are in
**[docs/models.md](docs/models.md)**.

---

## Which models

Any **Mixture-of-Experts** model in MLX format whose experts are stacked per layer. That is a
layout, not a list of blessed names: eight expert paths and both MLP shapes are recognised — the
three-projection SwiGLU that Qwen, DeepSeek, GLM and gpt-oss use, and the two-projection
`fc1`/`fc2` that Nemotron uses. Expert count, top-k, quantisation and layer path are read from
the checkpoint, never assumed.

`doctor` has read and answered for checkpoints from Qwen3, Qwen3.6, Qwen3-Next, Qwen3-Coder,
DeepSeek-V2 and V3, GLM-4.5 through 4.7, gpt-oss, Kimi-Linear, MiniMax, Hunyuan, ERNIE, Granite,
Phi-MoE and LongCat — from 4 GB up to 378 GB. Where a checkpoint stores its experts in a layout
that cannot be streamed, it says which layout and why, rather than calling it "not an MoE model".

Multimodal checkpoints load text-only; the vision tower is neither loaded nor charged to memory.
Dense (non-MoE) models are not the target — there is nothing to keep resident selectively.

Full list, and how the speed prediction works: **[docs/models.md](docs/models.md)**.

---

## What you need

- **An Apple Silicon Mac** (M1 or later) and macOS. MLX is Apple-only; this engine is built on it.
- **Python 3.10+.**
- **Memory.** By default BigRig plans to use 35% of installed RAM so the rest of your Mac keeps
  working, and it never exceeds that. Most 17–20 GB models need a six to seven gigabyte ceiling,
  which that default clears on a 24 GB Mac. On a 16 GB Mac it does not, so raise it — `doctor`
  prints the exact figure and the command (`BIGRIG_MAX_GB=7 bigrig run <model>`).
- **Disk.** The model, plus one more copy of its experts for a streamed model's fast path. That
  copy doubles the model's footprint; the size is printed before it is made, it is skipped when
  the disk is tight, and `--no-pack` declines it. A model that fits in RAM writes nothing extra.

There is no GPU to configure, no build step, and nothing is sent anywhere — the engine, the
server and the web interface all run entirely on your machine.

---

## Licence

[Apache License 2.0](LICENSE). Use it, modify it, ship it commercially — keep the notice and
don't sue us over patents.
