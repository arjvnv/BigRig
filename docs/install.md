# Install

[← BigRig](../README.md) · [Install](install.md) · [Quickstart](quickstart.md) · [Models](models.md) · [CLI](cli.md) · [How it works](HOW-IT-WORKS.md)

---

## Requirements

- Apple Silicon Mac (M1 or later), macOS
- Python 3.10+
- Disk space for the model — and **again as much** for a streamed model's packed copy of its
  experts, which the first run makes and `--no-pack` skips. A model that fits in RAM adds nothing.

## Install

```bash
pip install bigrig
```

That is all of it. On an Apple Silicon Mac this brings MLX and mlx-lm with it, so there is
nothing else to install and no build step.

If you would rather not install into your system Python — and on macOS you often cannot — make a
virtual environment first:

```bash
python3 -m venv ~/.bigrig-venv && source ~/.bigrig-venv/bin/activate
pip install bigrig
```

or use [pipx](https://pipx.pypa.io/), which handles that for you and puts `bigrig` on your PATH:

```bash
pipx install bigrig
```

### Check it worked

```bash
bigrig doctor
```

It prints what this machine can run. No model is downloaded and nothing is written.

## From source

For working on BigRig itself, or to get changes that are not in a release yet:

```bash
git clone https://github.com/arjvnv/bigrig.git && cd bigrig
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
```

[uv](https://docs.astral.sh/uv/) works too and is faster:

```bash
uv venv && source .venv/bin/activate
uv pip install -e .
```

## Verify

```bash
rig --version      # `rig` and `bigrig` are the same command
rig doctor         # measures this machine and says what it can run
```

`doctor` prints your free memory, the measured RAM and disk bandwidths, and — for any model you
already have — which of the three modes it would use and why.

## What gets installed

Two things, and the second works without the first:

- `bigrig_engine` — the engine, CLI and server. Needs MLX, so Apple Silicon only.
- `bigrig_layer` — the quality meter. Pure Python and numpy; works against Ollama, llama.cpp
  and the OpenAI API as well as this engine.

`pip install bigrig` brings the engine automatically on an Apple Silicon Mac. On any other
platform the same command installs only what the meter needs and does not fail, so `bigrig_layer`
is usable on Linux. The `[engine]` extra still exists for anyone following older instructions.

Next: [quickstart.md](quickstart.md).
