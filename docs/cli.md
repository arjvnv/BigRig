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
| `--no-persist` | — | `run`/`serve`: keep the conversation cache in memory only; by default it is also written to disk so a restart resumes (see below) |
| `--vision` | — | `run`/`serve`: load the checkpoint's vision encoder and read images (see below) |
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
- `POST /v1/embeddings` — OpenAI embeddings, when started with `--embeddings` (see below)
- `GET  /health` — live residency, miss rate, mode, and whether weights were altered; `context_used` and `context_remaining` say how full the conversation is against the ceiling that binds on this Mac (`max_completion_tokens`), so a client can warn before the next turn has to start over

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

## Guessing a few tokens ahead from the text: `lookahead`

Per request (`"lookahead": true`; the page's "guess ahead"), no extra model: when the last few
tokens have appeared earlier in the conversation, whatever followed them last time is proposed as
the next few tokens and checked in one pass. It pays when the reply repeats something already
written -- quoting a document, continuing a list (measured 1.48x on a verbatim passage) -- and
costs a little when it does not (0.87x on an ordinary question on Qwen3-30B, 0.93x with thinking
on Qwen3.6). Every token that comes out is one the model chose; the rejection path is bit-identical
to ordinary decoding.

On a model with recurrent state (Qwen3.6, Nemotron) the state at an intermediate position is
never materialised, so after a rejected guess the engine puts every layer back and re-reads the
tokens it kept -- one extra pass, reported as `rereads`. That is why acceptance has to be high
for it to pay on those models. A build before this re-read left rejected guesses in the state,
which showed up as a reply repeating the user's own sentence; if you saw that, update.

## The file as the pool (the default): `--slot-pool` to turn it off

```bash
rig serve <model>                # experts run from the file's cached pages: no copy, no slot
rig serve <model> --slot-pool    # the copy path: each expert copied into a pool slot on the GPU
```

A resident expert is a live view of its cached pages, eviction is dropping the view, and the
arithmetic runs on the view; prefill reads every expert a chunk wants straight from the cache.
This became the default in 0.9 after the measurement it waited for -- five models, one process
per mode, the same prompts: faster to the first token on every streamed model (1.1-2.6x), faster
decode on most (up to 1.9x), a peak 2.5-6.5 GB lower, and on Qwen3.6 the same GSM8K score as the
copy path (49/50 both) with HumanEval 40/40 against 38/40. Every live suite passes under it.

What it changes: the rows run through `quantized_matmul` rather than the gather a resident model
uses, so about half of short greedy replies differ somewhere from the copy path's -- a near-tie
flipped by a different kernel path, the same class of difference chunk width already causes, and
the two benchmarks say it is not a quality loss. `--slot-pool` restores the copy path, for
comparing against earlier versions. (The copy path's prefill also had a defect the measurement
found -- a long prompt could grow its lazy graph to 20 GB and be killed by Metal -- fixed in
the same release; the views path never accumulated it.)

## The conversation cache: `--kv-bits`

Past `--kv-quant-start` tokens (4,096 by default) the conversation cache is compressed to 4 bits,
which is what lets a long context fit -- 3.56x less memory, on the flagship the difference between
reaching 45% of the context window and all of it. It is a compression, so replies past the
threshold differ from what full precision would produce.

Measured cost, DeepSeek-Coder-V2-Lite on a ~2,000-token needle-retrieval task: 7 of 12 answers
correct at 4-bit against 8 of 12 at full precision -- one answer, on one model. Small, real, and
now optional:

```bash
rig serve <model> --kv-bits 0     # keep the cache full precision; spend the memory instead
rig serve <model> --kv-bits 8     # halfway: 8-bit
rig serve <model> --kv-quant-start 16384   # stay full precision for longer before it engages
```

`--kv-bits 0` (or `16`) turns compression off. It is a serve-time setting, not per-request: the
cache is shared across a conversation, so its precision is fixed for the session.

## Seeing: `--vision`

```bash
rig serve Qwen3.6-35B-A3B-4bit --vision                  # read images on both chat APIs
rig run   Qwen3.6-35B-A3B-4bit --vision                  # then `/image <path>` before a question
```

Qwen3.5 and 3.6 ship a vision encoder inside the checkpoint -- 0.89 GB of bf16 weights the text
engine otherwise discards. `--vision` accepts images on `/v1/chat/completions` (OpenAI
`image_url` parts) and `/v1/messages` (Anthropic `image` blocks), as base64 data URLs only --
this server fetches nothing. The web page shows an attach button when the server can see.

The encoder is read from the checkpoint when a request carries an image and given back before
the prompt is read, so the pool is planned as if it did not exist -- on a 16 GB Mac that is the
difference between running and not: at a 7 GB budget, Qwen3.6 with the encoder charged to the
ceiling is refused by the planner, while per-request it plans the same pool as without vision,
reads the image, and peaks 1.1 GB above its footprint for the duration of the request (0.05 s
to load when the checkpoint is in the page cache; an SSD read of 0.89 GB when cold).
`--vision-resident` keeps it loaded instead, charged to the ceiling before the pool is planned
(`/health` shows it under `vision` and `reserved_gb`), for a Mac with memory to spare and many
image requests. Measured on Qwen3.6 at the 9 GB ceiling: a 640x400
screenshot is 240 image tokens, encodes in half a second, and its text and code came back
transcribed exactly; the first token arrived after 7 s, the whole reply in 17 s.

The encoder, the preprocessing and the multimodal positions are BigRig's own MLX
implementations, checked against transformers' on the same inputs: features to a relative error
of 1e-4 (cosine 1.000000), positions equal for one image and for three images across two turns.
Activations run in float32 over the bf16 weights because a bf16 residual stream ends far from
the reference (0.78 cosine) -- the stream reaches a scale where bf16 resolves to 64.

Images are scaled to at most 1.05 megapixels (about 1,000 image tokens); `--vision-pixels N`
raises it. At most 8 images per request. A request with images does not use or fill the
conversation cache -- two different images of the same size share identical placeholder
tokens, so a cache keyed on tokens would hand one picture's state to another -- and guess-ahead
and the MTP head sit out for it. Video is not read. A checkpoint without an encoder refuses
`--vision` with a sentence.

## Embeddings: `--embeddings`

```bash
rig serve <model> --embeddings                          # BAAI/bge-small-en-v1.5, fetched on first use
rig serve <model> --embeddings sentence-transformers/all-MiniLM-L6-v2
```

Retrieval and codebase-indexing tools, and most agent memories, call `/v1/embeddings`; without it
a server cannot be their backend. `--embeddings` loads a small sentence encoder beside the chat
model and serves the OpenAI endpoint: `input` as a string or list, `encoding_format` `float` or
`base64` (what the OpenAI SDK asks for), `dimensions` to cut and re-normalise. Inputs longer than
the encoder's window are cut to it and the response says how many (`bigrig.truncated_inputs`);
`usage` counts the tokens actually read.

The default encoder is BAAI/bge-small-en-v1.5 -- MIT, 133 MB, 384 dimensions, 512 tokens. Any
BERT-shaped sentence-transformers repo with a `tokenizer.json` loads the same way; CLS and mean
pooling are read from the repo's own pooling config. The encoder is BigRig's own MLX
implementation, checked against sentence-transformers (PyTorch) on the same texts: every vector
matches to float32 noise (max element difference 3.6e-7, cosine 1.000000). Its memory is charged to
the ceiling before the expert pool is planned, so the ceiling still holds; `/health` reports it
under `embeddings` and `reserved_gb`. Embedding requests share the model's queue with replies --
one thing on the device at a time.

## The quality meter acts

The live meter already told a healthy reply from a damaged one. It now ends a reply that is
degrading -- once flagged tokens run 16 in a row, or make up 35% of the reply after 64 tokens --
and says why (`looping`, `weights drifted`, `incoherent`) and what to try, ranked by what usually
causes it: turn off guess ahead if it was on, serve `--exact` if the weights are compressed,
lower the creativity if it is above 1.0, otherwise ask again. The thresholds were measured, not
guessed: sixteen healthy replies on four models had a longest run of 2 and a share of at most
6.1%; a reply from a corrupted state ran 65 flagged in a row and is now stopped 16 tokens in.

`finish_reason` stays `stop`; the verdict is in `bigrig.stopped_for`, `bigrig.quality_reason`,
`bigrig.quality_run`, `bigrig.remedy` on both OpenAI paths, and the page's reply line says
"STOPPED by the quality meter". Text already streamed stays -- a stream cannot retract. A request
can send `"quality_stop": false`; `--no-quality-stop` turns it off for the server.

## Conversations survive a restart

The conversation cache is what makes a follow-up fast: the state for everything already said is
kept, so the next turn reads only its own new tokens. It is also written to disk -- while a server
is idle, and between turns in the terminal, never during a reply -- so a restart, an update or a
pool rebuild resumes every conversation the last run held. Measured on Qwen3.6: a server started
fresh answered the third turn of a two-turn conversation from the previous process with all 164
prior tokens reused and a reply identical to one that never restarted.

What is on disk mirrors what is in memory, no more: the same half-gigabyte budget, the same
entries, and a conversation the cache lets go is removed from disk at the next flush. A saved
state is restored only into the configuration that computed it -- the same model files, KV
precision, serving mode and mlx_lm version; anything else and the files are discarded rather than
restored into numbers they do not belong to.

The files hold the tokens of your conversations, on your machine, under BigRig's data directory.

```bash
bigrig sessions                    # what is kept: model, conversations, size, age
bigrig sessions --clear            # delete all of it (or: --clear <model>)
rig serve <model> --no-persist     # memory only, nothing written
```

Models that reason before answering (Qwen3.6, Nemotron) carry state that cannot be rolled back,
and their templates render a past turn differently from a live one, so their cache used to miss
on every follow-up. The engine now also keeps the state at the point where the history ends and
the current turn begins -- exactly what the next turn starts with -- so those models reuse the
whole prior conversation too (measured: 0 tokens reused before, the entire history after).

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

## Capping how long a model thinks: `thinking_budget`

A reasoning model on a slow machine can spend its whole reply budget thinking. Measured on
Qwen3.6 with `max_tokens: 400` and no cap: 183 words of reasoning and **an empty answer**. A
thinking budget caps the reasoning tokens; when it runs out the block is closed and the model
answers with what it has.

```json
{"thinking_budget": 150}                                   // OpenAI-style bodies, ours
{"reasoning_effort": "low"}                                // OpenAI's dial: low 256, medium 1024, high 4096
{"thinking": {"type": "enabled", "budget_tokens": 150}}    // Anthropic's shape, on /v1/messages
```

Under budget the model is untouched -- same tokens, same order. At the budget, while still
thinking, the closing tag is forced and the model continues into its answer, exactly as if it had
chosen to stop there; after that the think tags are refused so a cut-off thought cannot leak one
into the reply. A model that finishes thinking under budget never notices the cap exists, and one
that is already answering is never interrupted. Measured at 150 tokens on the same question: 77
words of reasoning, then `17 × 23 = 391`.

Pick the budget for the task -- 40 tokens is too few to finish arithmetic, and a budget plus the
answer must fit inside `max_tokens`. The cap takes the standard generation path; `--mtp` and
lookahead yield to it.

The page has the same cap as a control -- "thinking limit", shown when the model thinks --
set to half the reply limit by default. Measured on Qwen3.6 with a 1,000-token reply limit and
an open question: uncapped, all 1,000 tokens went to thinking and no answer came; capped at
half, thinking stopped at 500 and the answer had the other 500. Both APIs report what the cap
did on the reply (`bigrig.thinking_cut`, `bigrig.reasoning_tokens`).

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
rig doctor                         # this Mac, your prepared models, and what on the hub fits
rig doctor --for coding            # the ranked list, kept to models suited to coding
rig doctor <model-or-repo>         # the full account for one model
rig doctor [--calibrate]           # re-measure RAM and disk bandwidth (~30s)
```

With no model named, after the machine and the prepared models, doctor reads the shapes of a
short curated list of Mixture-of-Experts checkpoints from the hub (metadata only; nothing is
downloaded, about 25 seconds) and ranks them: whether each runs at your budget and at which speed
tier, then by size. It is the same verdict `doctor <repo>` gives, side by side, with a plain note
of what each model is for -- not a quality ranking, because this engine has not measured that.
`--for chat|coding|reasoning|vision` filters; `--no-recommend` skips it.

`--calibrate` re-measures RAM and disk bandwidth (~30s) instead of using the stored profile.

The speed word in the verdict (FAST / GOOD / USABLE / SLOW) is a prediction, and says so. It is
the expert bytes one token moves -- an assumed 0.6 miss rate (measured 0.53-0.61 across 7-30%
residency) times top_k, streamed layers and bytes per expert -- divided by the rate this Mac
moves them: 0.65 of the calibrated disk when the page cache is cold, about 6.4 GB/s of effective
traffic when it is warm. Both numbers are printed as a range with the bytes behind them. Without
a calibration the disk is assumed at 3 GB/s, conservatively. The console shows the same four
words once it has measured a median, so the prediction and the measurement never disagree about
what a number means. [Models](models.md) lists the runs those predictions were checked against.
