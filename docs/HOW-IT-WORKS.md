# How it works

[← bigrig](../README.md) · [Install](install.md) · [Quickstart](quickstart.md) · [Models](models.md) · [CLI](cli.md) · [How it works](HOW-IT-WORKS.md)

---

## The problem

An MoE model routes each token to a handful of its experts — 8 of 128 in Qwen3-30B. The other 120
sit unused for that token. But every runtime loads all of them, so a 13.4 GB model needs 13.4 GB
of RAM even though any single word touches ~0.8 GB of it.

There are only two ways to run it on a machine with less: **keep fewer experts** (and fetch the
rest from disk when the router asks for them), or **make every expert smaller** (and keep them
all). They fail in opposite directions — one costs speed, the other costs accuracy — and which is
better depends on the numbers, not on taste.

## Why streaming is slower than it looks

To know which expert to fetch, the engine has to read the router's output back from the GPU. That
read makes MLX stop pipelining and drain everything it had queued. Measured on Qwen3-30B:

```
the model's kernels alone      21.0 ms/token    47.1 tok/s
with 48 per-layer read-backs  119.4 ms/token     8.3 tok/s
the read-backs alone           98.5 ms/token
```

**The stall is four times the model's entire compute.** It is not the disk — disk is 14–35% of a
token. It's the cost of asking a question mid-pipeline.

That cost is per *layer*, which is why it scales badly: a 48-layer model pays it 48 times a token.

**Where the time actually goes, measured.** On Qwen3.6-35B-A3B-4bit at a 9.7 GB ceiling a decode
token is about 58 ms. Roughly 63% of it is the host read of each streamed layer's routing -- not
the read itself, which is 0.2–0.35 ms on an M4, but the GPU work it forces to run one layer at a
time instead of overlapping. Another ~30% is copying the experts that missed the pool from the
page cache onto the GPU: about 150 a token at 1.77 MB each, and that copy runs at memory
bandwidth, so threads do not help. With no host reads at all the same token takes 23 ms. Every
bit-exact trick tried against those two costs -- issuing resident experts first, staging predicted
experts during the wait, compiling the expert block -- either added ops to the queue the next read
drains or contended for the bandwidth the GPU was already using, and measured slower. What did
move it was not copying at all: on Apple Silicon the page cache and the GPU share one memory, so a
missed expert's pages are now wrapped as a GPU buffer instead of copied -- bit-identical, and a
steady ~47 ms a token (21 tok/s) where the copy path swung between 53 and 95 depending on what was
cached.

## Why compressing is fast

Nothing is fetched if nothing is missing. An all-resident model never reads the router back, so
the stall disappears entirely.

The consequence worth internalising: **speed is flat across precisions.** 2-bit, 3-bit and 4-bit
all run at 92–111 tok/s, because the win comes from residency, not from the bit width. So the
engine compresses only as far as it must to become resident, and never further — extra shrinking
buys no speed and costs accuracy.

## The three strategies

`autoconfig.choose_strategy` picks in increasing order of what each costs the user:

1. **native** — it fits. No engine in the path.
2. **compress** — it fits at fewer bits. Weights change, so **this one asks.**
3. **stream** — it doesn't fit even at the floor precision. Exact, slower.

## Why 3 bits is the floor

Measured on OLMoE-1B-7B against wikitext-2 and man pages:

| precision | perplexity vs original |
|---|---|
| 3-bit | +18.5% |
| 2-bit | +83% to +145% |

2-bit is a cliff, not a cheaper point on a curve. A real 4×0.6B model compressed to 2-bit ran at
34.9 tok/s with every expert resident and emitted `.\n1\n1` — fast, resident, and useless. Hence
`DEFAULT_MIN_BITS = 3`.

## Where the bytes come from

Experts are read straight out of the model's own safetensors. A tensor of shape `(E, …)` stores
expert *e* contiguously at a computable offset, so nothing needs copying — a 13.4 GB model costs
13.4 GB, not 26.

The cost is that one expert is nine ranges (three projections × weight/scales/biases) instead of
one, and — the part that matters now — none of them starts on a page boundary. The zero-copy
admit (the GPU reading an expert in place from the page cache, no CPU copy) needs a page-aligned
base and a whole number of pages, so it is only possible from the packed copy. That is why
packing is the default again: `prepare` makes the copy, and `run`/`serve` make it on a streamed
model's first run. `--no-pack` keeps the disk and the copy path.

**The invariant:** reading expert *e* directly returns exactly the bytes the packed copy holds for
expert *e*. If that ever drifted, the pool would assemble a tensor from the wrong places and the
model would generate fluent nonsense. `tests/test_direct.py` asserts it byte for byte.

## The expert pool

A bounded set of slots per layer, with LFUDA eviction (chosen by measuring six policies against
Belady's optimum on real routing traces; it beat LRU by 23.7%).

Two things the pool has to get right:

- **Never evict an expert the current token still needs.** LFUDA has no notion of "in use", so
  the pool passes an exclusion set. An earlier version worked around this by telling the policy
  an eviction had happened when it hadn't — after 200 steps the policy could see 1 of 12 resident
  experts and had silently stopped working.
- **A layer that can't miss doesn't need the read-back.** When every expert is resident, the
  expert→slot lookup runs on the GPU and the pipeline is never broken.

## Prefilling is a choice

Filling the pool before generating reads *C* experts per layer — 7.6 GB on Qwen3. That's worth it
only when every expert will be resident anyway. When streaming, it's a guess about what the prompt
wants, and the first token needs just top-k per layer.

So: prefill when `C == E`, skip it otherwise. Measured on OLMoE at 50% residency, first token in
1.12 s instead of 2.23 s with identical output — and full prefill also drove MLX 2.5 GB higher.

On the zero-copy path the chunk is wider. The 128-token ceiling was measured on the copy path,
where every miss in a wide pass copied an expert into host memory and the peak grew with the
width. With experts read in place there is no such copy, and measured on Qwen3.6-35B-A3B-4bit
the peak *fell* as the pass widened (7.9 GB at 104 tokens, 5.7 GB at 512) while time to first
token fell 1.8-2.7x, because a narrow pass re-reads the whole expert set once per chunk. So a
packed model prefills in 512-token passes; the model's own files keep the narrow one. Either way
the width is fixed by the model and the install, never by the pool, so two runs of one install
give the same reply.

## The file as the pool

A Mac's GPU and its file cache share one memory, so an expert sitting in the page cache can be
handed to the GPU without a copy. The shipped pool uses that for the copy IN: an admitted expert
is read from the cache into a pool slot with no CPU copy. `--file-pool` goes one step further and
keeps no slots at all: a resident expert is a live view of its cached pages, eviction is dropping
the view, and prefill reads every expert a chunk wants straight from the cache. Measured, that is
1.4-2.0x faster to the first token, about 1.1x faster decode, and a 0.5 GB smaller process
footprint, because the hot experts no longer exist twice. It is a choice rather than the default
because the rows then run through a different kernel than a resident model's, and about one reply
in three flips a near-tie. A native Metal kernel that gathers across views would remove that
difference; it is the piece of this design that is not built.

## Guessing one token ahead

Qwen3.5/3.6 ship a small head, trained with the model, that guesses the token after the one just
chosen from the model's own hidden state. With `--mtp` the engine loads it (its experts
quantised to 4-bit, a third of the memory, at no measured cost in accuracy of its guesses) and
runs each step as a pass over two tokens: the one the model chose and the head's guess. Position
0 of that pass says what really follows; if it is the guess, the pass has also produced the token
after it, and two tokens came out of one pass. If not, the state is put back and the one
confirmed token is redone on its own, exactly as ordinary decoding would have -- which matters
because three of every four layers in this model carry a recurrent state that cannot be trimmed
the way a KV cache can. Measured on Qwen3.6-35B-A3B-4bit, the guess is right 88% of the time.

What that buys on a streamed model depends on the pool: two tokens want up to sixteen experts a
layer where one wants eight, so a streamed layer needs room for both or it splits the pass and
evicts between the halves. The plan keeps that room when the head is on. Every number for what
it costs and gains is in [the CLI reference](cli.md); it is a choice, not a default.

## Thinking and the answer

Most current models of this class open a reasoning block before they answer, and two shapes are
in the wild: the chat template opens `<think>` in the prompt and the reply closes it (Qwen3.6,
GLM-4.7), or the reply opens and closes it itself (Qwen3-30B). The engine tracks which, splits
the reasoning from the answer as the tokens arrive, and hands them to the caller separately. The
APIs return the answer as `content`; the terminal shows everything inline. Nothing is discarded
and nothing is rewritten — the split is where the model's own `</think>` falls.

## The quality meter

Independent of all of the above. It watches the output distribution and flags degradation with no
reference answer and no second forward pass. On the damage compression actually causes it flags
0.0% of tokens on a healthy model and 36.7% on one shrunk past the cliff.

It's what makes the speed/accuracy trade sellable rather than a guess — and it works against
Ollama, llama.cpp and the OpenAI API, not just this engine.
