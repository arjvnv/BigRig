# BigRig — a runtime quality meter for local LLM inference

Tells you, **while a model is generating**, whether the output it is producing is poor — with no
reference answer, no second model, and no second forward pass.

## Why

Every technique that makes a large model fit on small hardware — quantisation, expert offloading,
cache-aware routing — trades quality for speed. **None of the systems that implement them tell the
user whether the trade went too far.** FreeToken's 555-file codebase contains zero occurrences of
`quality`, `perplexity`, `accuracy`, `confidence` or `entropy`.

## Use

```python
from bigrig_layer import AdaptiveMeter

m = AdaptiveMeter()           # learns what THIS model's normal looks like
for step in generation:
    m.observe(probs)          # next-token distribution
    m.observe_token(tok)      # the id actually emitted -- REQUIRED, see below
    if m.is_degraded():
        print("quality problem:", m.reason())   # 'looping' or 'incoherent'
```

Closed loop:

```python
from bigrig_layer.controller import AutoTuner
t = AutoTuner(dial_max=0.4)
for step in generation:
    engine.set_dial(t.observe(probs, token=tok))
```

## Two checks, combined with OR — and why that matters

| check | catches | blind to | threshold |
|---|---|---|---|
| **confidence** (entropy, top-1, margin) | incoherent / rambling text | **looping** | learned per model |
| **repetition** (4-gram) | looping | incoherent text | learned per model |

They are **OR-ed, never averaged**, because each is blind to what the other catches.

**Both thresholds are learned.** An earlier version of this file said repetition's threshold was
absolute and transferred, because repetition is a property of the string. The *rate* is; what
rate counts as *abnormal* is not. Measured on healthy output, window-level:

| | mean | p99 | fires at a fixed 0.15 |
|---|---|---|---|
| Ling-mini-2.0-3bit (instruct) | 0.006 | 0.082 | 0.4% |
| OLMoE-1B-7B-0125 (base) | 0.085 | 0.410 | **21.6%** |

A base model with no chat template is simply more repetitive, so a threshold fitted on an
instruct model called one healthy window in five "looping". `AdaptiveMeter` learns it instead:
OLMoE's false-alarm rate drops to **3.4%** and Ling's is unchanged bit-for-bit. This is why
`AutoTuner` defaults to `AdaptiveMeter`.

**This was learned the hard way.** The confidence model alone scores ρ=0.893 against a reference
quality measure — and then failed completely on real broken output:

```
DEGRADED (toll 0.4)   score -1.167   "Virtual memory is is is is is is is..."
HEALTHY  (no toll)    score -0.983   "Virtual memory is a fundamental concept..."
```

It rated the broken text **better** than the healthy text. A looping model is maximally confident,
so every confidence signal reads healthy. The research missed it because the reference measure
(judge-NLL) is **also** blind to looping — validating a blind metric against a blind metric
produced 0.893 agreement, because they share the gap. Repetition had been dropped for *lowering*
that correlation, which is exactly what a check covering the reference's blind spot would do.

Both checks now run. **If you do not call `observe_token()`, looping detection is off.**

## Attaching to an engine you did not write

The meter needs three numbers per token. Where they come from depends on what your engine
exposes, and the difference is not cosmetic:

| what the engine gives you | entropy | top-1 | margin |
|---|---|---|---|
| the full distribution (MLX, in-process) | exact | exact | exact |
| top-K logprobs (ollama, llama.cpp, OpenAI, vLLM) | **estimated** | exact | exact |
| only the chosen token's logprob | — | — | — |

top-1 and margin need only the two largest probabilities, so any engine returning a top-K list
gives them exactly. Entropy is a sum over the whole vocabulary and is the only casualty:
measured on Qwen2.5 through ollama, the top 8 entries carry **96.2%** of the mass, and the
missing 3.8% is spread over ~152,000 tokens that still carry real entropy.

```python
from bigrig_layer import AdaptiveMeter, observe_ollama_entry

m = AdaptiveMeter()
for entry in response["logprobs"]:            # ollama /api/generate, logprobs: true
    observe_ollama_entry(m, entry, vocab_size=151936)
    if m.is_degraded():
        ...                                    # back off, warn, or log
```

`observe_openai_chunk` is the same for anything speaking the OpenAI logprobs shape.
**Request `top_logprobs >= 2`** — with fewer, the margin is undefined and the adapter returns
`None` rather than guessing.

**Prime the meter on healthy output first.** A meter whose baseline is learned from damaged
output concludes that damage is normal. This is not a subtlety — it is why the first version of
`example_ollama.py` reported 0% on both healthy and damaged text.

### Measured end to end, on ollama, through the HTTP API only

`python -m bigrig_layer.example_ollama --model qwen2.5:1.5b`

| output | windows flagged | first flag |
|---|---|---|
| normal, temperature 0.7 | **0.0%** | — |
| hot, temperature 2.5 | 25.5% | token 121 |
| incoherent, temperature 5.0 | **46.0%** | token 74 |

**A caveat that demo taught us.** At temperature 2.5 the model produced *"reverse-phase freezing
and melting of water"* — factually wrong and completely fluent. Nothing in the meter can see
that. It went undetected until the text became genuinely incoherent. This is the blind spot
below, reproduced live rather than argued.

## What it cannot do

**It reports THAT quality is poor, not WHY.** A "damage alarm" — attributing degradation to your
compression setting — was built, tested and **refuted**: a placebo comparing two undamaged outputs
scored *higher* (0.75) than the claimed damage detection (0.55). Do not make that claim.

Re-tested on a second model and **still refuted**: on OLMoE the placebo scores 0.890 against a
0.831 headline. Every correlation this layer reports is dominated by how hard the prompt is, not
by anything a compression setting did.

**It is silent on very short outputs.** It needs 16 tokens for a provisional reading and 64 for a
full one. Generations that end before that get no reading at all — on OLMoE that was 7 of 128,
three of them under 16 tokens. Short answers are unmonitored by construction.

For `AutoTuner` this is concrete: **on an intrinsically hard prompt it will back off
unnecessarily**, costing speed, not quality. The loop is asymmetric — retreat 5x faster than
advance — for that reason.

## Cost — measured, not asserted

| | ms/token | % of a token |
|---|---|---|
| baseline | 35.52 | — |
| meter, `observe(probs)` | **+2.44** | **+6.9%** |
| meter, on-device stats + one sync | +1.82 | +5.1% |
| measurement bias (null control) | ±0.33 | — |

**It is not free.** On Ling-mini-2.0-3bit it costs 5–7% of per-token time; the transfer is not the
cost (an on-device variant with three separate syncs was *slower*). `stride>1` would divide this
but is **explicitly unvalidated** — the subsampling test that appeared to justify it measured
something else.

## Calibration

Shipped constants are fitted on **one model** (Ling-mini-2.0-3bit), held out (ρ=0.893, robust
0.844–0.914 across four rotated splits). They are a starting point, not a universal constant.
Re-fit with `QualityMeter.calibrate(features, targets)`.

## Tests

`test_meter.py` (38) and `test_controller.py` (15). They include the hostile cases that found real
bugs: NaN and inf silently produced a plausible-looking score before validation was added.
