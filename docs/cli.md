# CLI reference

[← BigRig](../README.md) · [Install](install.md) · [Quickstart](quickstart.md) · [Models](models.md) · [CLI](cli.md) · [How it works](HOW-IT-WORKS.md)

---

```
rig <command> [args]        # `bigrig` is the same command
```

**Start here**

| Command | Purpose |
|---|---|
| `rig doctor <model>` | Will it run here, and how fast? Nothing is downloaded |
| `rig run <model>` | Chat in the terminal |
| `rig serve <model>` | Start the server (browser page + OpenAI + Anthropic APIs) |
| `rig launch <model>` | Run Claude Code or Codex against it, in one command |

**More**

| Command | Purpose |
|---|---|
| `rig list` | Models already on this machine |
| `rig prepare <model>` | Download now and make the packed copy the fast path needs (`--no-pack` to skip) |
| `rig compress <model>` | Shrink a model so every expert fits in memory (faster, lossy) |
| `rig knee <model>` | Re-measure the fastest setting (`run` does this once for you) |
| `rig calibrate <model>` | Measure the host round-trip and the whole-layer split |
| `rig diff <model>` | The same prompt through shipped and compressed weights |

Every command supports `--help`.

**The first run tunes itself.** The first time a streamed model runs at a given memory budget,
BigRig spends a minute or two measuring how many experts to keep in memory for the best speed on
*this* Mac, then remembers the answer. Every later run uses it. Skip it with `--no-tune`, or
answer the question yourself with `--residency`.

## Shared flags

These apply to `run`, `serve` and `launch`:

| Flag | Default | Meaning |
|---|---|---|
| `--memory N` | what is actually free | GB the engine may use |
| `--exact` | — | Never shrink; stream instead. Weights untouched, decode bit-identical |
| `--compress` | — | Agree to shrink the model to fit. **The weights change** |
| `--min-bits N` | 3 | Never compress below this precision |
| `--forget-choice` | — | Discard the remembered shrink/exact decision and ask again |
| `--residency F` | measured on first run | Fraction of experts to keep resident. Giving one skips the first-run tune |
| `--no-tune` | — | Skip the one-time first-run speed measurement and use the safe estimate |
| `--prefetch N` | 0 (off) | Name N experts a layer ahead from the hidden state. Off: measured not to pay (0.84× prose, 0.98× code on Qwen3.6). Needs `bigrig predict <model>` first; `BIGRIG_STAGE=1` copies the named experts to the GPU during the layer's attention |
| `--no-monitor` | — | Turn off quality monitoring |
| `--threads N` | 8 | Reader threads for expert fetches |
| `--trust-remote-code` | — | Also download the model's custom Python |

`--exact` and `--compress` contradict each other and cannot both be given.

## rig serve

```bash
rig serve <model> [--host 127.0.0.1] [--port 8080] [--no-release-memory] [--no-reclaim-memory]
```

When the Mac runs short of memory, the server hands experts back between replies rather than
being killed, and takes them back in small steps once the machine has been quiet for three
minutes, never past the capacity it started from. Both halves are on by default:
`--no-release-memory` turns the whole controller off, `--no-reclaim-memory` keeps only the
giving-back half (a restart is then the only way to recover speed after a squeeze). The first
minute after start-up is ignored, because loading the model is itself a burst of memory
traffic and a server once shrank on its own start-up.

On start the server also reads the model's experts into the OS page cache in the background,
taking whatever memory is spare beyond a one-gigabyte margin, stopping the moment the machine is
short, and pausing while any reply is in flight. On this Mac that is the difference between
10.5 and 21 tok/s on Qwen3.6-35B-A3B-4bit, because a cold cache serves experts from disk one
page at a time. It reads the experts this model has used most first, from a record kept across
runs, so a cache too small for the whole file holds the right ones. `--no-warm` skips it.

Serves four things on one port:

- `GET  /` — the web interface
- `POST /v1/chat/completions`, `/v1/completions` — OpenAI
- `POST /v1/messages`, `/v1/messages/count_tokens` — Anthropic
- `GET  /health` — live residency, miss rate, mode, and whether weights were altered

Requests are served one at a time. One model, one expert pool: two generations at once would
evict each other's experts every step and both would finish later than if they had queued.
`/health` reports the queue depth.

## rig launch

```bash
rig launch <model> [--agent claude|codex|opencode] [--port N] [-- <args for the agent>]
```

Starts a server, points the agent at it, runs it, and stops the server when the agent exits.
Configuration is by environment variable on the child process only — nothing on disk changes.

If the agent is not installed, it prints the install command and stops rather than running an
installer for you.

## rig prepare

```bash
rig prepare <model> [--no-pack]
```

Downloads the model if needed, then makes a contiguous, page-aligned copy of its experts. That
copy is what the zero-copy path needs: the GPU reads an expert straight out of the page cache
instead of the CPU copying it in, and the model's own shards never lay experts out on page
boundaries (0 of 360 expert tensors in Qwen3.6-35B-A3B-4bit do). It doubles the model's disk
footprint; `--no-pack` keeps the disk and takes the slower copy path. `rig run` and `rig serve`
make the same copy on a streamed model's first run, before the one-time speed measurement, so
the measurement is of the path the model will actually run on.

## Guessing one token ahead: `--mtp`

```bash
rig serve <model> --mtp [PATH] [--mtp-bits 4|8|0]
rig run   <model> --mtp
```

Qwen3.5 and 3.6 ship a small "multi-token prediction" head: one transformer layer, trained with
the model, that guesses the token after the one just chosen. The MLX quantisations strip it;
mlx-community publishes it separately as `<model>-MTP-bf16` (1.69 GB). With `--mtp` the engine
loads it, guesses one token ahead, and has the model check the guess in the same pass that would
have produced the next token anyway. Every token that comes out is one the model chose. Measured
on Qwen3.6-35B-A3B-4bit at the 9.7 GB ceiling: 88.7% of guesses right in bf16, 88.4% with the
head's experts at 4-bit (the default, a third of the memory).

The head is charged to the memory ceiling like a draft model, and streamed layers are planned
with room for two tokens' experts, so a verify pass does not split. A request can turn it off
(`"mtp": false`) but not on, which is how the numbers above were taken: the
same warm server, on and off. Whether it helps on a given Mac is measured there, not promised.

One caveat, stated plainly: a rejected guess is redone on its own and is bit-identical to
ordinary decoding; an accepted guess and the token after it come from a two-token pass, the same
arithmetic through different kernels. How often that flips a near-tie is measured in
measurement. It is a choice, not a default.

## The file as the pool: `--file-pool`

```bash
rig serve <model> --file-pool
```

Normally an expert the model needs is copied from the file's cached pages into a pool slot on
the GPU, and the pool holds the hot experts beside the page cache that already has them. With
`--file-pool` there is no copy and no slot: a resident expert is a live view of its cached pages,
eviction is dropping the view, and the arithmetic runs on the view. Measured on
Qwen3.6-35B-A3B-4bit at the 9.7 GB ceiling: 1.4-2.0x faster to the first token (no copies during
prefill), about 1.1x faster decode, and a 0.5 GB smaller process footprint because the hot
experts exist once. The cost is the same one `--mtp` has: the rows run through
`quantized_matmul` rather than the gather a resident model uses, and about one reply in three
differs somewhere in a near-tie. Off by default for that reason. The numbers are in
measurement; this is the software form of the design that a native Metal kernel
would make exact.

## Structured output: `response_format`

The OpenAI field, on `/v1/chat/completions`:

```json
{"response_format": {"type": "json_object"}}
{"response_format": {"type": "json_schema", "json_schema": {"name": "x", "schema": {...}}}}
```

The sampler is constrained so the reply is **one complete JSON object and nothing else** — no
prose around it, no code fence, no trailing sentence. This is enforced token by token against a
JSON grammar, so it holds at any temperature and for any model, and it costs nothing per token the
model was not already paying: the model's own preference order among legal tokens is untouched,
and a token is only ever removed when it would have broken the document.

With a `json_schema`, the schema is placed in front of the model in words and every name in its
top-level `required` list must appear as a key before the object is allowed to close. Property
**types** are not enforced at the token level. If a model cannot produce a required key — a small
model that recites the schema instead of filling it, say — the object is still allowed to close
after it has stalled, so the reply always parses and the missing key is something the client can
check for, which broken JSON is not.

Thinking is turned off for a constrained request: the reply *is* the JSON, as in OpenAI's own
mode. `response_format` and `tools` cannot be combined — a tool call is not a JSON object — and
sending both is a `400` that says so. The two opt-in speed paths (`--mtp`, lookahead) yield to a
constrained request, which takes the standard path.

## Models that think before answering

Qwen3.5/3.6, GLM-4.x, Nemotron and their kin produce a block of reasoning before the answer; the
first three are measured here.

On the **OpenAI** endpoints the **answer** comes back as `content` and the reasoning separately,
as `reasoning_content` on a blocking reply and `delta.reasoning_content` while streaming — the
same shape vLLM and DeepSeek's own API use, so a client that does not know the field simply sees
the answer.

On the **Anthropic** endpoint the reasoning is its own `thinking` content block, emitted before
the text block, which is that protocol's own shape for it. Streaming opens the thinking block
first and closes it the moment the answer starts, so a reply with reasoning arrives as
`thinking` then `text`, and one that spends its whole budget reasoning arrives as a single
`thinking` block with `stop_reason: max_tokens`. No `signature_delta` is sent: only Anthropic's
own models can sign a reasoning block, and a forged signature would be worse than none.

The browser page shows the thinking folded away above the reply. `rig run` in the terminal shows
it inline, because watching a model think is the point there.

Send `"think": false` to turn thinking off where the model's template supports it (`rig doctor`
reports whether it does). A reply can legitimately have empty `content` and a long
`reasoning_content`: that is a model that spent its whole token budget thinking, and the page
says so on the footer.

## rig doctor

```bash
rig doctor [--calibrate]
```

`--calibrate` re-measures RAM and disk bandwidth (~30s) instead of using the stored profile.

The speed word in the verdict (FAST / GOOD / USABLE / SLOW) is a prediction, and says so. It is
the expert bytes one token moves -- an assumed 0.6 miss rate (measured 0.53-0.61 across 7-30%
residency) times top_k, streamed layers and bytes per expert -- divided by the rate this Mac
moves them: 0.65 of the calibrated disk when the page cache is cold, about 6.4 GB/s of effective
traffic when it is warm. Both numbers are printed as a range with the bytes behind them. Without
a calibration the disk is assumed at 3 GB/s, conservatively. The console shows the same four
words once it has measured a median, so the prediction and the measurement never disagree about
what a number means. [Models](models.md) lists the runs those predictions were checked against.
