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

**[Quickstart](docs/quickstart.md)** ·
[Install](docs/install.md) ·
[Models](docs/models.md) ·
[CLI](docs/cli.md) ·
[How it works](docs/HOW-IT-WORKS.md) ·
[Contributing](CONTRIBUTING.md)

</div>

---

## About

A 30B MoE model uses only a handful of its experts for any given word, but every runtime keeps
all of them in memory. bigrig keeps a bounded set in RAM and reads the rest from SSD, or shrinks
the model to fit — **your choice, asked before anything changes** — while a quality meter watches
the output and tells you if the trade went too far.

- **Three strategies, chosen for you.** Run it untouched, shrink it to fit, or stream it from
  disk. Picked in increasing order of what each costs you: nothing, then accuracy, then speed.
- **It asks before changing your model.** Shrinking is the only path that alters weights, so it
  is the only one that needs your agreement — and every run afterwards still prints which mode
  it is serving.
- **The weights are never touched, and the arithmetic was measured against the stock runtime.**
  Decode is bit-identical to the same model run resident — checked layer by layer, 4,392 live
  layer computations with zero difference. Prompt processing matches to the last bit of a float
  or one bit off it, depending on how rows fall into kernel calls: the same variation mlx_lm
  itself shows between two prefill widths.
- **A quality meter, running live.** No reference answer, no second forward pass. It separates a
  healthy model from a damaged one at 0.0% vs 36.7% of tokens flagged.
- **One extra copy on disk, announced before it is made.** A streamed model's first run writes
  its experts out contiguously — the layout the GPU can read without the CPU copying it — which
  doubles that model's footprint. It prints the size first, skips itself when the disk is tight,
  and `--no-pack` reads from the downloaded safetensors instead. A model that fits in RAM writes
  nothing extra.
- **Speaks OpenAI and Anthropic.** Including a one-command hook into Claude Code, and a web
  interface for people who would rather not use a terminal.

---

## Getting started

### Install

```bash
pip install bigrig
```

That is the whole install on an Apple Silicon Mac — it brings MLX with it. `rig` and `bigrig`
are the same command. [Other ways to install](docs/install.md), including from source.

### Two commands to a running model

```bash
bigrig doctor mlx-community/Qwen3-30B-A3B-4bit    # will it run here, and how fast? nothing downloaded
bigrig serve  mlx-community/Qwen3-30B-A3B-4bit    # download it, pack it, tune it once, run it
```

`doctor` is the one worth running first: it says whether a model fits **before** you spend an
hour downloading one that does not, and roughly how fast it will be — `FAST`, `GOOD`, `USABLE`
or `SLOW`, predicted from the bytes each token must read and your Mac's measured disk, and
checked against the real runs in [Models](docs/models.md). When nothing would make it fit, it says `IMPOSSIBLE ON THIS
MAC` and stops there. Misspell a name and it offers the three closest.

`serve` downloads the model if needed, makes the packed copy of its experts the fast path needs
(it doubles the model's disk footprint; `--no-pack` skips it), and **the first run measures
itself**: a minute or two finding how many experts to keep in memory for the best speed on your
Mac, remembered from then on. Then it tells you exactly what it is about to serve, and where to find it:

```
  bigrig serving OLMoE-1B-7B-0125-Instruct-4bit on http://127.0.0.1:8080
  running EXACT at 4-bit, fully in RAM (untouched)
  quality monitor: on

  open  http://127.0.0.1:8080  in a browser to chat and watch quality
  api   127.0.0.1:8080/v1/chat/completions   (OpenAI)
        127.0.0.1:8080/v1/messages           (Anthropic -- Claude Code)
  agent bigrig launch OLMoE-1B-7B-0125-Instruct-4bit
```

That second line is printed on **every** run. Whatever mode you are in, the precision being
served is never a surprise.

### Then pick whichever suits you

**A coding agent** — the browser page has a **Code** tab with the one command that wires each
one up, and it says up front whether the model you loaded can call tools at all. An agent pointed
at a model that cannot does not fail loudly; it answers the question instead of doing the work.

```bash
bigrig launch Qwen3-30B-A3B-3bit                  # Claude Code
bigrig launch --agent codex Qwen3-30B-A3B-3bit    # Codex CLI
```

**A browser** — open `http://127.0.0.1:8080`. Chat with streaming replies, a **creativity**
slider, and **standing instructions** the model reads before every message — a role, a language,
rules — kept in your browser. Along the side: the model, the mode it is running in, memory held,
live tokens/second with the same **FAST / GOOD / USABLE / SLOW** verdict `doctor` gives (and
whether it is measured or still an estimate), and an **anomaly indicator that turns amber
mid-generation** if the model starts looping. A **Console** page shows where every gigabyte goes and lets you change how many
experts stay in memory; a **Code** page wires up coding agents. One self-contained page, no CDN,
nothing to install.

**Claude Code** — one command, which starts the server and points the agent at it:

```bash
rig launch OLMoE-1B-7B-0125-Instruct-4bit
```

Nothing on disk is changed: the environment variables are set on the agent's process only.

**Your own code** — both APIs are served on the same port, so any client library works by
changing its base URL:

```
POST /v1/chat/completions      OpenAI
POST /v1/messages              Anthropic
```

**A terminal** — `rig run <model>`.

**Models that think first** — Qwen3.5/3.6, GLM-4.x and Nemotron reason before answering, all
three measured here. The OpenAI API returns the answer as `content` and the reasoning as
`reasoning_content`; the Anthropic API gives the reasoning its own `thinking` block ahead of the
text, which is that protocol's own shape for it. Either way a coding agent gets the reply rather
than the scratchpad. The browser page folds the thinking away above the answer; the terminal
shows it inline.

**Guessing ahead** — Qwen3.5/3.6 ship a small head that predicts the next token from the model's
own state. `--mtp` loads it (0.56 GB at 4-bit) and has the model check every guess. Measured on
Qwen3.6-35B-A3B-4bit at the 9.7 GB ceiling: 85-89% of guesses right, 1.03-1.08x faster, and
replies no longer byte-identical to plain decoding (a two-token pass rounds differently). Off by
default; the numbers are in [the CLI reference](docs/cli.md).

**The file as the pool** — `--file-pool` keeps no copies of experts at all: a resident expert is a
live view of its cached pages, and prefill reads straight from the cache. Measured: 1.4-2.0x
faster to the first token, about 1.1x faster decode, half a gigabyte smaller footprint, and one
reply in three differs in a near-tie because the rows run through a different kernel. Off by
default for that reason.

---

## What happens when a model doesn't fit

bigrig stops and asks, because the two ways forward cost different things:

```
    [1] Shrink it to fit     full speed, but THE WEIGHTS CHANGE
    [2] Keep it exact        streamed from disk, the weights untouched, slower
```

Your answer is remembered, and every run still prints which one it is serving. In a script with
neither `--compress` nor `--exact`, it **refuses** rather than guessing — quietly serving a
degraded model is worse than stopping.

**It will also tell you not to use it.** If a model already fits, bigrig loads it normally and
says so; putting the engine in front of a model that doesn't need it only makes it slower.

**And it will tell you when nothing would help.** Some models need more memory than your Mac has
at any setting. `doctor` says so before a download — `IMPOSSIBLE ON THIS MAC`, with the number —
and nothing else in bigrig tries to talk you past it.

---

## Measurements

On an M4 MacBook Air (24 GB). Nothing here is projected.

**Shrinking vs streaming, at the same memory** (OLMoE-1B-7B, wikitext-2):

| memory | shrink (all in RAM) | stream (exact) | which wins |
|---|---|---|---|
| 2.82 GB | 3-bit, 92 tok/s, +17% perplexity | 50 tok/s, exact | shrink |
| 2.62 GB | 3-bit, **111 tok/s**, +18% perplexity | 25 tok/s, exact | **shrink, 4.4×** |
| 2.01 GB | 2-bit, 110 tok/s, **+83%** perplexity | 17 tok/s, exact | stream |

Below 3 bits is a cliff, not a bargain — so **3 bits is the floor**, and going lower takes an
explicit `--min-bits 2`.


---

## The quality meter on its own

`bigrig_layer` is independent of the engine. It watches any model's output distribution and
flags degradation with no reference answer and no second forward pass — and it works against
Ollama, llama.cpp and the OpenAI API too:

```python
from bigrig_layer import AdaptiveMeter

m = AdaptiveMeter()
for step in generation:
    m.observe(probs)
    m.observe_token(tok)
    if m.is_degraded():
        print("quality problem:", m.reason())
```

---

## Repository layout

```
bigrig_engine/     the engine — strategy, streaming, precision, serving, web UI, CLI
bigrig_layer/      the quality meter (standalone, engine-agnostic)
tests/             ./run-tests.sh runs every assertion
docs/              install, quickstart, models, CLI, how it works
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for how the tests are written and why.

## Requirements

Apple Silicon (M1 or later), macOS, Python 3.10+.

## Licence

[Apache License 2.0](LICENSE). Use it, modify it, ship it commercially — keep the notice and
don't sue us over patents.
