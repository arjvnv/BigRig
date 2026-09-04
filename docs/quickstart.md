# Quick start

[← BigRig](../README.md) · [Install](install.md) · [Quickstart](quickstart.md) · [Models](models.md) · [CLI](cli.md) · [How it works](HOW-IT-WORKS.md)

---

```bash
pip install bigrig
```

That is the install. More ways in [install.md](install.md).

## 0. Ask what this Mac can run

```bash
bigrig doctor
```

Worth doing first. It reads your machine and says which models fit **before** you download one,
and roughly how fast each would be — `FAST`, `GOOD`, `USABLE` or `SLOW`, predicted from the bytes each
token must read and your Mac's measured disk, and checked against the real runs in
[models.md](models.md).
A "no" comes with a number: either the memory it would need and the one flag that gives it, or
a plain `IMPOSSIBLE ON THIS MAC` when no flag would help. Misspell a name and it offers the three
closest. Nothing is downloaded and nothing is written.

```bash
bigrig doctor mlx-community/Qwen3-30B-A3B-4bit
```

## 1. Get a model

```bash
rig prepare mlx-community/OLMoE-1B-7B-0125-Instruct-4bit
```

Takes a Hugging Face repo id or a local directory. It downloads the model, and for one that will
be streamed it also writes a packed copy of the experts — the fast path — which **doubles that
model's footprint on disk**. The size is printed before it starts, and `--no-pack` skips it and
reads out of the downloaded files directly.

## 2. Start the server

```bash
rig serve OLMoE-1B-7B-0125-Instruct-4bit
```

**The first time** a streamed model runs at a given memory budget, BigRig spends a minute or two
measuring how many experts to keep in memory for the best speed on *your* Mac, says so while it
does, and remembers the answer. Every later start is immediate. (`--no-tune` skips it.)

It prints one line saying exactly what it is serving — the precision, and whether anything was
changed — then:

```
  open  http://127.0.0.1:8080  in a browser to chat and watch quality
  api   127.0.0.1:8080/v1/chat/completions   (OpenAI)
        127.0.0.1:8080/v1/messages           (Anthropic -- Claude Code)
        127.0.0.1:8080/v1/responses          (OpenAI Responses -- Codex CLI)
```

The browser page has a **Code** tab that shows these, says whether the loaded model can call
tools at all, and gives the one command that wires each coding agent to it.

## 3. Use it

**In a browser** — open the address. A chat box with streaming replies, a **creativity** slider
and a place for **standing instructions** (a role, a language, rules — read before every message,
kept in your browser), and a live anomaly meter that turns amber if the model starts looping. The
**Console** page shows where the memory goes and lets you change how many experts stay in memory
with a slider; the **Code** page gives the one command that wires each coding agent up. Nothing to
install.

**From code** — it speaks both APIs, so any client library works by changing its base URL:

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"What is a Mixture-of-Experts model?"}]}'
```

**With Claude Code** — one command, which starts the server and points the agent at it:

```bash
rig launch OLMoE-1B-7B-0125-Instruct-4bit
```

Nothing on disk is changed: the environment variables are set on the agent's process only, and
closing it restores exactly the setup you had.

**In the terminal** — `rig run <model>` for a plain chat loop. `/stats` for numbers, `/quit` to
leave.

## If the model does not fit

BigRig stops and asks, because the two ways forward cost different things:

```
    [1] Shrink it to fit     full speed, but the weights change
    [2] Keep it exact        streamed from disk, the weights untouched, slower
```

Your answer is remembered, and every run still prints which one it is serving. In a script with
neither `--compress` nor `--exact`, it refuses rather than guessing.

## If the model cannot fit at all

Some models need more memory than your Mac has, at any setting. `bigrig doctor` says so before
you download — `IMPOSSIBLE ON THIS MAC`, with the number it would need — and nothing else in
BigRig will try to talk you past it. That is not a limitation of the engine to work around; it is
the machine's answer.

Next: [models.md](models.md) · [cli.md](cli.md)
