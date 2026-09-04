"""One model, loaded and ready to answer, with the quality meter already attached.

This is the layer everything user-facing sits on -- the CLI, the server, and any Python caller.
It exists so that "use the engine" is three lines rather than thirty, and so that the two things
that make this product worth installing are ON by default rather than opt-in:

    1. the model runs in less memory than it needs
    2. the engine tells you when the output has degraded

Point 2 matters most precisely when point 1 is working hardest. Streaming does not change the
math -- the round-trip test asserts bit-identical logits -- but a user who has just been told
their 13 GB model now runs in 6 GB has every right to want that claim checked continuously
rather than taken on faith.
"""
from __future__ import annotations

import json
import os
import re
import time

import mlx.core as mx

from . import autoconfig, calibrate, consent, precision, predict, stream, synccal
from . import knee as _knee

try:
    from bigrig_layer.adaptive import AdaptiveMeter
    from bigrig_layer.adapters import stats_from_logprobs
    _HAVE_METER = True
except Exception:                                   # the engine must run without the monitor
    _HAVE_METER = False


# No local model on Apple Silicon decodes faster than this. Anything above it is an artefact of
# dividing by a near-zero elapsed time, not a measurement.
# A model with no chat template gets a plain "user:/assistant:" prompt, and a BASE model will
# happily keep writing that pattern -- inventing the user's next question and answering it too.
# Observed on OLMoE-1B-7B-0125-4bit, which ships no template at all: one question produced four
# fabricated turns. These end a turn wherever the model tries to start a new one.
# What a forward pass needs beyond the resident weights and the KV cache.
#
# THIS WAS MEASURED DURING DECODE FIRST, AND DECODE IS THE WRONG PASS TO MEASURE.
#     Generating one token at a time peaks 0.72 GB above the weights, so 0.80 looked like a safe
#     reserve. Prefill pushes prefill_step_size prompt tokens through the model in ONE pass, and
#     that pass is hundreds of tokens wide. Measured on Qwen3-30B-A3B-3bit at 44/128 (M4 Air),
#     peak memory above the resident weights, KV subtracted:
#
#         prompt tokens      step 256      step 128
#             164             2.15 GB       1.29 GB
#             788             3.39 GB       2.40 GB
#           2,009             3.50 GB       2.44 GB
#           4,109             3.85 GB       2.36 GB
#
#     It saturates rather than growing without bound, but it saturates three times higher than
#     decode. At step 256 a 4,109-token prompt peaked at 9.29 GB against a 9.00 GB ceiling; at
#     step 128 the same prompt peaked at 7.80 GB. 2.6 leaves a margin over the largest step-128
#     figure.
#     A FORMULA THAT WAS WRONG, AND WHAT REPLACED IT
#     This was briefly `base + top_k * bytes_per_expert * n_layers`, on the reasoning that a step
#     can miss top-k experts in every layer. The arithmetic was right and the mechanism was not:
#     a layer's fetched bytes are released once they are in slots, so the peak carries ONE
#     layer's worth -- 53 MB on gpt-oss -- not the sum over 36 of them. The formula inflated the
#     reserve by 1.9 GB and made the planner refuse a model that runs.
#
#     What actually drives the peak is the width of the prefill pass, which is now chosen per
#     model (see PREFILL_ACT_BUDGET_GB). With that bounded, two measurements at a 9.00 GB
#     ceiling:
#
#         Qwen3-30B-A3B-3bit   44/128 resident, 2.06 MB experts   2.4 GB above resident
#         gpt-oss-120b-MXFP4    5/128 resident, 13.24 MB experts  2.96 GB above resident
#
#     3.0 clears both. Two points is not a law, so this stays a measured constant rather than a
#     formula pretending to more than it knows.
WORKING_MEMORY_GB = 3.0

# Below this the number has stopped being a limit and become a refusal. If memory is this tight
# the model should not have loaded at all.
MIN_REPLY_TOKENS = 256

# Used only when a checkpoint does not say what its context window is. Deliberately small: the
# cost of guessing low is a reply that stops early, the cost of guessing high is output generated
# past the positions the model was trained on, which degrades silently.
ASSUMED_CONTEXT = 4096

# Safety valve, and how the energy signal's cost is measured against the meter without it.
_NO_ENERGY = bool(os.environ.get("BIGRIG_NO_ENERGY"))

# See _prefill_step. Bounds the widest forward pass, and with it the largest memory spike the
# server can produce.
PREFILL_STEP = 128
# A run of this many consecutive flagged tokens counts as degradation; fewer is noise at z = 2.5.
FLAG_RUN = 3
# Below this share of flagged tokens, and with no run, a reply is reported clean.
FLAG_NOISE_SHARE = 0.02
# THE PACKED PATH IS A DIFFERENT MACHINE, AND ITS PEAK RUNS THE OTHER WAY.
#     The 128 above and the bytes-per-unit model below were measured on the copy path, where every
#     miss in a wide pass copied an expert into host memory and the peak grew with the width.
#     With zero-copy admits there is no such copy, and measured on Qwen3.6-35B-A3B-4bit at the
#     9.7 GB ceiling (2026-09-02, 1,749-token prompt) the peak FELL as the pass widened, because a
#     narrow pass re-reads the whole expert set once per chunk and churns the pool doing it:
#
#         step   104    256    512   1024      (tokens a pass)
#         ttft  41.2   27.9   22.7   22.5  s
#         peak  7.87   6.34   5.69   5.55  GB
#
#     and on a 453-token prompt 15.8 -> 5.8 s from 104 to 512. So a packed model on the zero-copy
#     path takes a 512-token pass, still bounded by the same activation budget for models with a
#     wider expert path. Still a function of the model and the install, never of the pool: two
#     runs of the same install give the same reply.
PREFILL_STEP_PACKED = 512
PREFILL_ACT_BYTES_PER_UNIT_PACKED = 320

# HOW MUCH MEMORY ONE PREFILL TOKEN COSTS, AND WHY THE STEP IS NOT A FUNCTION OF CAPACITY.
#     This used to be `MAX_PREFILL_SPANS * capacity // top_k` -- a step sized to hold the number
#     of _chunks splits constant, on the belief that the memory peak grew with the number of
#     splits. Measured on Qwen3-30B-A3B-3bit, that belief is wrong. Peak, at two capacities:
#
#         capacity 40:  step 18 -> 9.18 GB (15 chunks)   64 -> 9.58 (52)   128 -> 10.29 (103)
#         capacity 11:  step 18 -> 6.18 GB (14 chunks)   64 -> 6.64 (47)   128 ->  7.43 ( 94)
#
#     The chunk counts differ sevenfold between those rows and the step-18-to-128 delta is the
#     same either way, 1.11 GB against 1.25 GB. The peak tracks the STEP. Capacity moves the
#     baseline, because a bigger pool is bigger, and moves it by exactly the pool difference.
#     Splitting costs time, not memory, and step and capacity only looked linked because one was
#     computed from the other.
#
#     That matters for more than tidiness. Chunked prefill is not bit-exact across step sizes --
#     floating-point reductions run in a different order at a different shape -- so the step is a
#     quality-visible number. Measured on a 395-token prompt, steps 64 and 128 agree, and 16, 32,
#     256, 512 and 2048 each produce different text; the same step run twice is always identical,
#     so this is shape, not nondeterminism. A step derived from capacity is therefore a step the
#     memory controller can change underneath a conversation: shrinking 9.0 GB -> 6.5 GB moved it
#     80 -> 30, and a prompt longer than the step comes back worded differently. Deriving the
#     step from the MODEL instead makes it a constant for the life of the process, which is what
#     lets the controller reclaim memory without touching what the model says.
#
#     Bytes per prefill token scale with the expert intermediate a token is pushed through,
#     top_k * moe_intermediate_size. Two models, measured independently:
#
#         Qwen3-30B-A3B-3bit   top_k 8, moe_inter 768    6144 units    ~10.1 MB/token   1628 B
#         gpt-oss-120b         top_k 4, moe_inter 2880  11520 units    ~15.4 MB/token   1337 B
#
#     The higher of the two, so the estimate errs toward the smaller step.
PREFILL_ACT_BYTES_PER_UNIT = 1628

# What one prefill pass may spend on activations. It comes out of WORKING_MEMORY_GB, which is
# 3.00 and also covers everything else transient, so this takes under a quarter of it. At this
# budget Qwen3-30B gets step 69 and gpt-oss-120b gets 37 -- and 37 puts gpt-oss at roughly
# 8.06 GB against the 9.00 GB ceiling where step 128 measured 9.47 GB and would have crashed.
#
# EVERY NUMBER ABOVE WAS MEASURED BEFORE PREFILL WAS REGROUPED BY EXPERT, AND IS NOW
# CONSERVATIVE. Splitting in token order accumulated one output array per gather and there were
# 13,018 of them in a single prompt; holding them all until the concatenate was most of the peak,
# not the activations. Regrouped, the same prompt peaks at 5.11 GB where it used to peak at 9.10.
# The budget is deliberately not re-fitted to that. Widening the step would change what the model
# says -- that is the whole reason this constant exists rather than a capacity-derived formula --
# so it stays where it was measured, and the headroom is simply headroom.
PREFILL_ACT_BUDGET_GB = 0.70

# What the pool must leave behind for the reply. autoconfig held back a flat 3.0 GB covering the
# OS, the MLX runtime, the KV cache and activations -- tuned for this model, but a constant, so a
# model with a heavier KV cache would have silently overrun it. The same reserve is derived here
# instead: the two terms that do not depend on the model stay fixed, and the KV term is computed
# from the checkpoint. For Qwen3-30B that gives 1.3 + 0.8 + 0.81 = 2.91 GB, which is where the
# hand-tuned 3.0 came from -- the derivation reproduces it rather than replacing it.
#
#   MLX runtime                       Metal buffers, the command queue, the allocator's cache.
#                                     The OS itself is NOT counted here: autoconfig subtracts
#                                     MIN_HEADROOM_GB on top of whatever reserve it is given,
#                                     and counting the same slack twice cost about ten experts
#                                     per layer -- which, per the note on serving_reserve_gb,
#                                     made peak memory worse rather than better.
OS_AND_RUNTIME_GB = 0.3

# HOW MUCH MEMORY IS SET ASIDE TO REMEMBER PROMPTS ALREADY READ, AND WHY IT IS WORTH IT.
#     Every request re-read the whole conversation from the first token. In a chat that is the
#     same work over and over: turn three re-reads turn one's document, turn one's answer and
#     turn two's, to add one question. Keeping the attention state for a prefix that has already
#     been read turns that into reading only what is new.
#
#     Measured on a 892-token document followed by two follow-up questions, time to first token:
#
#         turn 1   892 tokens,   0 reused    10.30s      (nothing to reuse yet)
#         turn 2   939 tokens, 888 reused     1.13s      against 11.34s without    10.0x
#         turn 3   959 tokens, 935 reused     0.83s      against 11.31s without    13.6x
#
#     and all three replies were identical to the uncached run, which is the part that had to be
#     checked rather than assumed -- see tests/test_promptcache.py.
#
#     THE COST IS POOL SLOTS, WHICH IS WHY THIS IS A RESERVE AND NOT A CACHE THAT GROWS.
#     It is added to `serving_reserve_gb`, so the planner sizes the expert pool around it and the
#     ceiling still means what it says. Growing into whatever headroom looked free at the time
#     would put the machine's stability on a guess about a future prompt's length.
#
#     0.50 GB at 98,304 bytes a token on Qwen3-30B is about 5,000 tokens of conversation -- a
#     long document plus several turns. Against it, 0.50 GB is roughly five more experts a layer,
#     worth a few percent of decode. For anything with a second turn the cache is the better use
#     of the memory by an order of magnitude; for strictly single-turn use it is a loss, which is
#     what `prompt_cache_gb=0` is for.
# STREAMING THE KV CACHE WAS THE PLAN. THE ARITHMETIC REFUSED IT, SO THIS IS THE ANSWER INSTEAD.
#     Experts stream because routing is SPARSE: a token needs 8 of 128, so a pool of 11 serves it
#     and a miss reads 2 MB. Attention has no equivalent -- every token attends to every previous
#     key and value, so paging the cache means re-reading ALL of it every token. Measured on this
#     model: 805 MB per token at 8k of context, 3.2 GB at 32k, 12.9 GB at 131k. At a generous
#     5 GB/s that is 0.16s, 0.64s and 2.58s PER TOKEN before any arithmetic happens. There is no
#     residency policy that fixes a 100% miss rate.
#
#     The problem it was meant to solve is real. A 9 GB budget leaves 1.82 GB for KV after the
#     weights and the working set, which is 18,514 tokens of this model's 40,960 -- 45% of its own
#     context window unreachable.
#
#     Quantising the cache solves it outright. Measured on a 1,342-token prompt, 80 generated:
#
#         fp16 (what shipped)      6.97 tok/s   147.3 MB of KV
#         8-bit                    7.39 tok/s    78.2 MB    1.88x less
#         4-bit                    7.78 tok/s    41.4 MB    3.56x less
#         4-bit after 1024 tokens  8.66 tok/s    41.4 MB    3.56x less
#
#     FASTER, not slower, because the cache is most of what attention reads. 3.56x turns 18,514
#     tokens into more than the 40,960 the model was trained for, so the whole window is reachable.
#
#     ON BY DEFAULT, ABOVE KV_QUANT_START. It is a compression and it does change the reply, so
#     the threshold is what makes that acceptable: below 4,096 tokens nothing is quantised and the
#     arithmetic is bit-for-bit what it was, and above it the alternative is not an exact reply
#     but NO reply -- the ceiling used to stop at 18,514 and the model's window is 40,960.
#     Trading exactness for a conversation that can continue is the right way round; trading a
#     conversation that stops for exactness nobody asked for is not.
#
#     Reported in /health and named on the console, so it is never a silent change. Set to None to
#     turn it off, and a short conversation is unaffected either way.
KV_BITS = 4
KV_GROUP_SIZE = 64
KV_QUANT_START = 4096

PROMPT_CACHE_GB = 0.50

# The smallest prefix worth serving from the cache at all. Serving one copies the whole stored
# entry -- 66 MB on this model -- so a match shorter than this costs more to reuse than to redo.
MIN_REUSE_TOKENS = 32

# And the share of a prompt that must be reused before the conversation counts as PROVEN and its
# continuation earns the protected segment. Half is deliberately demanding: promotion is a claim
# that keeping this entry will pay again, and the cost of being wrong is evicting one that would.
PROVEN_REUSE_FRACTION = 0.50
# Kept for the warning, which needs a concrete reply length to price. Not part of the reserve --
# see the note where serving_reserve_gb is computed.
TARGET_REPLY_TOKENS = 8192

# gpt-oss does not answer in plain text. It emits a "harmony" transcript with named channels:
#
#   <|channel|>analysis<|message|>working it out...<|end|>
#   <|start|>assistant<|channel|>final<|message|>the answer
#
# Passed through untouched, a user is shown the control tokens and both channels, which reads as
# a broken model. The analysis channel is reasoning, so it is rewritten into the <think> markers
# every other reasoning model here already uses and the rest of the stack already understands --
# rather than teaching the UI, the API and the meter a second convention.
_HARMONY = (
    (re.compile(r"<\|channel\|>\s*analysis\s*<\|message\|>"), "<think>"),
    (re.compile(r"<\|end\|>\s*<\|start\|>\s*assistant\s*<\|channel\|>\s*final\s*"
                r"<\|message\|>"), "</think>"),
    (re.compile(r"<\|channel\|>\s*\w+\s*<\|message\|>"), ""),
    (re.compile(r"<\|[^|>]*\|>"), ""),
)
# A FIXED HOLD-BACK COSTS THE USER REAL SECONDS, SO IT IS NOT FIXED.
#     Holding 96 characters back meant nothing appeared on screen until 96 characters existed.
#     On gpt-oss at roughly one token a second that is about 26 seconds of blank window AFTER
#     generation has already started -- measured end to end, "hi" showed its first text at 66.0s
#     of which ~26s was this. The rewrite still runs over the whole reply, so the only thing that
#     genuinely cannot be shown yet is a marker still arriving: `<|chan` might become
#     `<|channel|>` and a stream cannot retract what it has printed. That is what is held, and
#     nothing else.
def _harmony_hold(raw: str) -> int:
    """Characters at the end of the RAW text that may still be part of an unfinished construct.

    ON THE RAW TEXT, NOT THE REWRITTEN TEXT, AND THE DIFFERENCE IS A CORRUPTED REPLY.
        Rewriting a prefix does not give a prefix of the rewrite. `<|channel|>analysis<|message|`
        is a pair that has not finished arriving: the pair rule does not match it, so the
        catch-all strips `<|channel|>` on its own and leaves the bare word "analysis", which then
        gets sent and cannot be taken back. Measured while building this, one character at a
        time: "<nalysis<er asks about hash ta". So the question is asked of the raw text -- is a
        construct still arriving? -- and the answer is held whole.
    """
    hold_from = len(raw)
    # A channel opener with no message marker after it is still arriving, and so is an <|end|>
    # whose following <|start|>...<|message|> has not landed -- that pair becomes `</think>`.
    for marker in ("<|channel|>", "<|end|>"):
        i = raw.rfind(marker)
        if i >= 0 and "<|message|>" not in raw[i + len(marker):]:
            hold_from = min(hold_from, i)
    # A marker cut mid-way by the chunk boundary.
    j = raw.rfind("<|")
    if j >= 0 and "|>" not in raw[j:]:
        hold_from = min(hold_from, j)
    return len(raw) - hold_from

TURN_STOPS = ("\nUser:", "\nuser:", "\nUSER:", "\nAssistant:", "\nassistant:",
              "\nHuman:", "\nQuestion:", "\n### ")
MAX_PLAUSIBLE_TOK_S = 2000.0
MIN_TOKENS_FOR_RATE = 4


def _sane_tps(tps, n_tokens):
    """A tokens/second figure, or None while it would still be nonsense.

    On the first token the elapsed time is ~0, so mlx_lm's running rate comes out at tens of
    thousands. Measured here: 31,128 tok/s on chunk one. Reporting that anywhere a person can
    see it -- a UI, an API response -- makes every other number on the screen untrustworthy.
    This project has already published one impossible throughput figure (2792 tok/s, from a
    regex that matched prompt evaluation instead of generation); once was enough.
    """
    if tps is None or n_tokens is None or n_tokens < MIN_TOKENS_FOR_RATE:
        return None
    try:
        t = float(tps)
    except (TypeError, ValueError):
        return None
    return t if 0.0 < t <= MAX_PLAUSIBLE_TOK_S else None


def _dir_gb(path: str) -> float:
    """Weights on disk, which is what they will take in memory."""
    import glob as _g
    return sum(os.path.getsize(f) for f in _g.glob(os.path.join(path, "*.safetensors"))) / 1e9


# THE CEILING THE ENGINE WILL NOT CROSS, WHATEVER IT IS ASKED FOR.
#     `available_gb()` reports what is free right now and the planner will spend all of it. On a
#     26 GB machine with 16.6 GB free that produced a 12.27 GB resident pool -- inside the free
#     memory and still far more than this machine can give up, because free memory is not spare
#     memory and the transient peak during a long prefill has been measured at 9 GB against a
#     5 GB plan. A budget with no ceiling is a budget that eventually takes the machine down.
#
#     Every path is clamped: the flag, the environment variable, and the automatic estimate. A
#     machine with room to spare raises it explicitly with BIGRIG_MAX_GB, which is a decision
#     someone made rather than a number that drifted upward on its own.
#
# WHY A SHARE OF TOTAL RAM AND NOT A SHARE OF WHAT IS FREE.
#     Free memory is not spare memory. This machine has 25.8 GB installed, reports 12.9 GB
#     "available", and cannot safely give the engine more than about 9 -- because most of what
#     is counted as available is file cache and inactive pages that belong to something. Total
#     RAM is the only memory number that does not move while we are looking at it.
#
#     THE SHARE IS CALIBRATED ON ONE MACHINE AND SHOULD BE REVISITED ON MORE. 0.35 of 25.8 GB
#     is 9.0, which is the ceiling this machine was validated at by hand after a crash at a
#     higher one. It is deliberately conservative: the transient peak during a long prefill has
#     measured between 1.55x and 2.79x the resident pool, so a ceiling set near what is free
#     leaves nothing for the spike.
SAFE_SHARE_OF_RAM = 0.35


def _default_ceiling() -> float:
    total = calibrate.total_gb()
    return round(total * SAFE_SHARE_OF_RAM, 1) if total else 9.0


MAX_ALLOWED_GB = float(os.environ.get("BIGRIG_MAX_GB") or _default_ceiling())


_DTYPE_BITS = {"float32": 32, "float16": 16, "bfloat16": 16, "uint32": 32,
               "int8": 8, "uint8": 8, "float64": 64}


def _dtype_label(spec: dict) -> str:
    """How to NAME the precision of weights that carry no quantisation parameters."""
    for comps in (spec or {}).values():
        w = comps.get("weight")
        if isinstance(w, dict) and w.get("dtype"):
            return str(w["dtype"])
    return "full precision"


def _dtype_bits(spec: dict) -> int:
    """Bits per weight for unquantised tensors, so arithmetic that expects a number gets one."""
    return _DTYPE_BITS.get(_dtype_label(spec), 16)


class ToolCallSplitter:
    """Separate tool calls from prose as the tokens arrive, one chunk at a time.

    THE PROBLEM STREAMING ADDS, WHICH THE BLOCKING PATH DOES NOT HAVE.
        With the whole reply in hand a call is easy to find: search for the delimiters and cut.
        Streaming has no whole reply. A chunk can end in the MIDDLE of `<tool_call>` -- the model
        emits it as several tokens -- so a splitter that simply looked for the tag would emit
        `<tool` as assistant text and then never recognise the tag when its second half arrived.
        The user would see machine syntax in the middle of a sentence and the client would see no
        call at all.

        So text is held back whenever its tail could still turn into an opening delimiter. `Sure
        <tool` emits `Sure ` and holds `<tool`; if the next chunk is `_call>` the held text was
        never prose, and if it is `box` the whole of `<toolbox` is released at once.

    WHAT IT GUARANTEES
        Every character of the reply comes out exactly once, in order, either as text or inside a
        call -- never both, never neither. `finish()` releases whatever is still held, including
        an unterminated call's body, which is dropped rather than shown: a half-written call is
        machine syntax the user never asked for and would corrupt the next turn's prompt.
    """

    def __init__(self, start: str, end: str, parse, tools=None, repair=None):
        self.start, self.end, self.parse, self.tools = start, end, parse, tools
        self.repair = repair
        self.buf = ""
        self.in_call = False

    def _holdback(self, text: str) -> int:
        """How many trailing characters could still become an opening delimiter."""
        n = min(len(text), len(self.start) - 1)
        for i in range(n, 0, -1):
            if self.start.startswith(text[-i:]):
                return i
        return 0

    def feed(self, chunk: str):
        """Returns (text safe to emit now, [calls completed by this chunk])."""
        self.buf += chunk or ""
        out, calls = [], []
        while True:
            if not self.in_call:
                i = self.buf.find(self.start)
                if i < 0:
                    keep = self._holdback(self.buf)
                    if keep:
                        out.append(self.buf[:-keep])
                        self.buf = self.buf[-keep:]
                    else:
                        out.append(self.buf)
                        self.buf = ""
                    break
                out.append(self.buf[:i])
                self.buf = self.buf[i + len(self.start):]
                self.in_call = True
            else:
                j = self.buf.find(self.end) if self.end else -1
                if j < 0:
                    break                      # still arriving; hold the whole body
                body, self.buf = self.buf[:j], self.buf[j + len(self.end):]
                self.in_call = False
                c = self._parse_one(body)
                if c:
                    calls.append(c)
        return "".join(out), calls

    def finish(self):
        """Flush. An unterminated call is dropped, not shown -- see the class docstring."""
        if self.in_call:
            c = self._parse_one(self.buf)
            self.buf, self.in_call = "", False
            return "", ([c] if c else [])
        text, self.buf = self.buf, ""
        return text, []

    def _parse_one(self, body: str):
        try:
            c = self.parse(body.strip(), self.tools)
        except Exception:                      # noqa: BLE001 -- malformed is normal at 3 bits
            c = None
        if not c and self.repair is not None:
            fixed = self.repair(body)
            if isinstance(fixed, dict) and fixed.get("name"):
                a = fixed.get("arguments")
                c = {"name": fixed["name"], "arguments": a if isinstance(a, dict) else {}}
        return c


class _TwoStagePromptCache:
    """Remembered prompts, split so that unrepeated ones cannot evict repeated ones.

    THE BUG THIS EXISTS FOR, MEASURED.
        One LRU held everything. A conversation that was genuinely being continued sat beside a
        stream of one-off prompts, and the one-offs won on recency simply by arriving later.
        Measured against a 500 MB budget: a prompt cached and reused answered in 0.53s; twelve
        requests that each carried a different identifier at the FRONT -- so each matched nothing
        and each stored a new entry -- took the cache from 66 MB to 461 MB; the reused prompt then
        answered in 6.06s. Evicted by traffic that could never benefit from being kept. 11.5x.

        That shape is not unusual, it is what agent traffic looks like: a long stable body with a
        request id or timestamp near the top, so every request misses and every request inserts.

    HOW IT IS SPLIT, AND WHY THE SIGNAL IS FREE.
        An entry earns the protected segment by being USED, not by being new. The signal costs
        nothing to compute: if the lookup that preceded this request HIT, the conversation is
        being continued and storing its continuation is likely to pay again. If it missed, this
        is a prompt nobody has asked for twice and it goes to probation, where it can only
        displace other unproven entries.

        Probation is deliberately the smaller share. It has to be big enough to hold one
        conversation long enough for a follow-up to arrive, and no bigger -- every byte it holds
        is a byte the proven entries cannot use.

    Deliberately NOT an attempt to make a changed prefix reusable. Attention state is positional:
    if a token near the front differs, every key and value after it is genuinely different and
    reusing them would be wrong, not merely stale. What can be fixed is the eviction, and that is
    what this fixes.
    """

    PROBATION_SHARE = 0.30

    def __init__(self, max_bytes: int, max_size: int = 64):
        from mlx_lm.models.cache import LRUPromptCache
        probation = int(max_bytes * self.PROBATION_SHARE)
        self.probation = LRUPromptCache(max_size=max_size, max_bytes=probation)
        self.protected = LRUPromptCache(max_size=max_size, max_bytes=max_bytes - probation)
        self.promotions = 0

    @property
    def nbytes(self) -> int:
        return int(self.probation.nbytes) + int(self.protected.nbytes)

    def fetch_nearest_cache(self, model, tokens):
        """Protected first: a proven entry is the better answer when both could serve."""
        cache, rest = self.protected.fetch_nearest_cache(model, tokens)
        if cache is not None and len(rest) < len(tokens):
            return cache, rest, True
        cache, rest = self.probation.fetch_nearest_cache(model, tokens)
        return cache, rest, False

    def insert_cache(self, model, tokens, prompt_cache, *, proven: bool = False):
        """`proven` means the lookup for this request hit. See the class docstring."""
        if proven:
            self.promotions += 1
            self.protected.insert_cache(model, tokens, prompt_cache)
        else:
            self.probation.insert_cache(model, tokens, prompt_cache)

    def trim_to(self, *, n_bytes: int | None = None, n_sequences: int | None = None):
        """Probation goes first, then protected, because that is the order they are worth."""
        if n_bytes is None:
            self.probation.trim_to(n_bytes=0)
            self.protected.trim_to(n_bytes=0)
            return
        self.probation.trim_to(n_bytes=max(0, n_bytes - int(self.protected.nbytes)))
        if self.nbytes > n_bytes:
            self.protected.trim_to(n_bytes=max(0, n_bytes))


def _config_dir(model_dir: str) -> str:
    """The directory holding this model's config.json, given a path OR a hub repo id.

    Returns model_dir unchanged when it is already a real directory, so the common case costs
    nothing and nothing about local models changes. For a repo id, asks the hub cache for the
    snapshot that is already on disk -- `local_files_only`, because this runs during startup and
    must never turn a config read into a download or a network stall.
    """
    d = os.path.expanduser(model_dir)
    if os.path.isdir(d):
        return d
    try:
        from huggingface_hub import snapshot_download
        return snapshot_download(repo_id=model_dir, local_files_only=True,
                                 allow_patterns=["config.json"])
    except Exception:                 # noqa: BLE001 -- not on disk, not cached, no network
        return d                      # callers already handle a config.json that is not there


def resolve_budget(budget_gb: float | None = None, quiet: bool = False) -> float:
    """How much memory this run may use: the flag, then the environment, then what is free.

    Shared so `rig doctor` and `rig serve` cannot answer it differently. They used to: doctor
    planned against whatever was free and reported 13 of 128 experts for gpt-oss, while serve
    planned against BIGRIG_MEM_GB and ran 4 of 128. Both were internally consistent and one of
    them was a lie to whoever read it first.
    """
    want = None
    if budget_gb is not None:
        want = float(budget_gb)
    else:
        env = os.environ.get("BIGRIG_MEM_GB")
        if env:
            try:
                want = float(env)
            except ValueError:
                want = None
    if want is None:
        want = calibrate.available_gb()
    if want > MAX_ALLOWED_GB:
        if not quiet:                     # doctor explains the cap itself, in its MACHINE block
            # Said from the user's side: nobody "asked for" 12.8 GB, that is what was free.
            print(f"  memory: {want:.1f} GB is free; using the {MAX_ALLOWED_GB:.1f} GB ceiling "
                  f"so the rest of the Mac keeps working (BIGRIG_MAX_GB raises it)", flush=True)
        want = MAX_ALLOWED_GB
    return want


def kv_bytes(cfg: dict) -> tuple:
    """(bytes each new token adds to the KV cache, bytes it costs regardless), from config.json.

    WHY THIS IS NOT ONE MULTIPLICATION
        It was, and the multiplication was `n_layers * num_key_value_heads * head_dim * 2 * 2` --
        every layer caching a key and a value per head. That is true of Llama-shaped attention and
        false of three families this engine claims to serve, in the direction that hurts most: the
        number is used to cap reply length and to size the expert pool, so an OVERestimate makes
        replies far shorter than they need to be and starves the pool of memory that was never
        going to be used.

        Measured against what mlx_lm actually allocates:

            DeepSeek-V3      1,748,992 B/token estimated   70,272 actual    24.9x too high
            GLM-4.5-Lite       must be checked per config, same mechanism
            Qwen3-Next-80B     estimated every layer; only 1 in 4 has a KV cache at all
            gpt-oss-120b       estimated every layer unbounded; half are capped at 128 tokens

        Three shapes have to be told apart, and config.json says which is which:

        MULTI-HEAD LATENT ATTENTION (`kv_lora_rank`) caches ONE compressed latent per layer, not
        one key and one value per head -- mlx_lm stores `kv_latent` of width kv_lora_rank and
        `k_pe` of width qk_rope_head_dim, each with a single head axis. Eleven architectures in
        mlx_lm use it, including every DeepSeek V2/V3/V3.2 and GLM's DSA and Lite variants.

        LINEAR-ATTENTION HYBRIDS (`full_attention_interval`, Qwen3-Next) give most layers a gated
        delta net whose state is a fixed pair of arrays. Only every nth layer holds a KV cache.

        SLIDING-WINDOW HYBRIDS (`layer_types`, gpt-oss) cap the windowed layers at
        `sliding_window` tokens: those layers stop growing, so they are a fixed cost, not a
        per-token one.

    The fixed term is deliberately charged at its ceiling rather than tracked as it fills. It is
    small -- gpt-oss's windowed half is 128 tokens deep -- and charging it up front means a reply
    can never run into a cost the plan did not price.
    """
    try:
        L = int(cfg["num_hidden_layers"])
    except (KeyError, TypeError, ValueError):
        return 0, 0

    # --- per-layer cost of an attention layer, in bytes per token -------------------------
    if cfg.get("kv_lora_rank"):
        # One latent and one rope key, one head each. No num_key_value_heads term: MLA's whole
        # point is that the per-head keys and values are reconstructed from the latent.
        try:
            attn = (int(cfg["kv_lora_rank"]) + int(cfg.get("qk_rope_head_dim") or 0)) * 2
        except (TypeError, ValueError):
            return 0, 0
    else:
        try:
            heads = int(cfg.get("num_key_value_heads") or cfg["num_attention_heads"])
            hd = int(cfg.get("head_dim") or cfg["hidden_size"] // cfg["num_attention_heads"])
        except (KeyError, TypeError, ZeroDivisionError, ValueError):
            return 0, 0
        attn = heads * hd * 2 * 2              # key and value, two bytes each

    # --- which layers actually hold a cache that GROWS with the reply --------------------
    #     Three kinds of layer, and mlx_lm's own make_cache is the authority on which is which:
    #       full_attention    KVCache        grows one key and one value per token
    #       sliding_attention RotatingKVCache grows to sliding_window and then stops
    #       linear_attention  ArraysCache    a fixed recurrent state, no growth at all
    #     `layer_types` names them directly and every windowed architecture in mlx_lm keys off
    #     it. A bare `sliding_window` field does NOT: Phi-3.5-MoE and Qwen1.5-MoE both declare
    #     one and neither overrides make_cache, so every layer is an ordinary KVCache. Trusting
    #     that field alone charged Phi-3.5-MoE 17 GB per token.
    types = cfg.get("layer_types")
    window = cfg.get("sliding_window") or cfg.get("sliding_window_size")
    interval = cfg.get("full_attention_interval")
    kda = (cfg.get("linear_attn_config") or {}).get("kda_layers")
    hybrid = cfg.get("hybrid_layer_pattern")
    # Mamba hybrids name every block: "*" attention, "M" a Mamba state, "-" an MLP with no cache
    # at all. Nemotron-3-Nano is 52 blocks and only 6 of them are attention, so charging by
    # num_hidden_layers priced it 8.7x too high.
    blocks = cfg.get("hybrid_override_pattern") or cfg.get("layers_block_type")
    if isinstance(blocks, str):
        blocks = list(blocks)

    if isinstance(blocks, list) and blocks:
        norm = [str(b)[0] for b in blocks]
        full = sum(1 for b in norm if b in ("*", "a"))       # "attention" normalises to "a"
        windowed = 0
    elif isinstance(types, list) and types:
        full = sum(1 for t in types if t == "full_attention")
        windowed = sum(1 for t in types if t == "sliding_attention")
    elif isinstance(hybrid, list) and hybrid:          # mimo: 1 marks a windowed layer
        windowed = sum(1 for v in hybrid if v == 1)
        full = len(hybrid) - windowed
    elif isinstance(kda, list):                        # kimi: listed layers are recurrent, 1-based
        full, windowed = L - sum(1 for i in kda if 1 <= int(i) <= L), 0
    elif interval:
        try:                                           # Qwen3-Next: linear unless (i+1) % iv == 0
            iv = int(interval)
            full = sum(1 for i in range(L) if (i + 1) % iv == 0)
        except (TypeError, ValueError, ZeroDivisionError):
            full = L
        windowed = 0
    else:
        full, windowed = L, 0

    per_token = full * attn
    fixed = windowed * attn * int(window) if (windowed and window) else 0
    return int(per_token), int(fixed)


class Session:
    """A loaded model plus its expert pool, its meter, and its counters."""

    def __init__(self, model_dir: str, capacity=None, policy: str = "lfuda", threads: int = 8,
                 monitor: bool = True, budget_gb: float | None = None, verbose: bool = True,
                 force_stream: bool = False, min_bits: int | None = None,
                 preference: str | None = None, interactive: bool = False,
                 remember: bool = True, warm="auto", full_layers=None,
                 no_full_layers: bool = False,
                 draft: str | None = None, draft_tokens: int = 3,
                 prefetch_width: int = 0, reroute: float = 0.0, nocache: bool = False,
                 prompt_cache_gb: float | None = None, kv_bits: int | None = None,
                 kv_quant_start: int | None = None,
                 mtp: str | None = None, mtp_bits: int | None = 4, announce: bool = True):
        # Everything this session was built from, so it can be rebuilt with one setting changed
        # without the caller having to remember the other twenty.
        self.init_kwargs = {
            "model_dir": model_dir, "capacity": capacity, "policy": policy, "threads": threads,
            "monitor": monitor, "budget_gb": budget_gb, "verbose": verbose,
            "force_stream": force_stream, "min_bits": min_bits, "preference": preference,
            "interactive": False, "remember": remember, "warm": warm,
            "full_layers": full_layers, "no_full_layers": no_full_layers,
            "draft": draft, "draft_tokens": draft_tokens,
            "prefetch_width": prefetch_width, "reroute": reroute, "nocache": nocache,
            "prompt_cache_gb": prompt_cache_gb, "kv_bits": kv_bits,
            "kv_quant_start": kv_quant_start, "mtp": mtp, "mtp_bits": mtp_bits,
            "announce": announce,
        }
        # WHAT THE PLAN IS, SAID ONCE. A first run builds a session, packs, builds again, tunes,
        # builds again -- and printed three plans, two of them about to be superseded. The lines
        # are collected here and printed by whoever knows which session is the one being served.
        self.plan_lines: list = []
        self.model_dir = os.path.expanduser(model_dir)
        self.name = os.path.basename(self.model_dir.rstrip("/"))
        # WHERE config.json ACTUALLY IS, WHICH IS NOT ALWAYS model_dir.
        #     A session started on a repo id ("mlx-community/Qwen3-30B-A3B-3bit") never needs
        #     that id to be a directory: the experts come from a packed blob keyed on the
        #     basename, and mlx_lm resolves the id against the hub cache to load the model. So
        #     model_dir stays a repo id, and everything that read config.json by joining onto it
        #     was silently landing in the OSError branch -- `_model_limits` returned (0, 0) for
        #     context length and bytes per KV token on every hub-cached model.
        #
        #     model_dir itself is deliberately NOT rewritten to the resolved path: `expert_source`
        #     keys the blob on its basename, and a snapshot directory's basename is a commit hash,
        #     so rewriting it would stop the blob from ever being found again.
        self.config_dir = _config_dir(self.model_dir)
        self.verbose = verbose
        self.handle = None
        self.plan = None
        t0 = time.perf_counter()

        # THE GAP THIS CLOSES
        #     BIGRIG_MEM_GB was read by the memory guard, which the bench harnesses arm and the
        #     engine never did. `BIGRIG_MEM_GB=9 rig serve` therefore planned against whatever
        #     the machine happened to have free -- 13.6 GB on a 24 GB Mac -- and the ceiling was
        #     honoured only by the accident of --residency landing under it. It is a real ceiling
        #     now, and everything downstream is planned against it.
        self.budget_gb = resolve_budget(budget_gb)
        budget_gb = self.budget_gb
        # Set before anything can fail, so a model that never streams still reports it honestly.
        self.reroute_tol = 0.0

        # A DRAFT MODEL IS PAID FOR BEFORE THE POOL IS SIZED, NOT AFTER.
        #     Speculative decoding proposes several tokens with a small model and checks them all
        #     in ONE pass of the big one -- which is worth more here than it is on a machine with
        #     room, because the per-layer host round-trip is paid per STEP and a step that
        #     settles n tokens amortises it n-fold. It costs memory for the draft, and memory is
        #     the resource this engine is short of, so its weights come out of the budget the
        #     expert pool is then planned against. Counting it afterwards would size the pool
        #     against memory the draft had already taken.
        self.draft_model = None
        self.draft_tokens = max(1, int(draft_tokens))
        self.draft_name = ""
        self.draft_gb = 0.0
        if draft:
            self.draft_gb = _dir_gb(os.path.expanduser(draft))
            budget_gb = max(budget_gb * 0.5, budget_gb - self.draft_gb)
            self.draft_name = os.path.basename(os.path.expanduser(draft).rstrip("/"))
        # THE MODEL'S OWN NEXT-TOKEN HEAD, CHARGED TO THE BUDGET BEFORE THE POOL IS PLANNED.
        #     Like a draft model, it lives beside the pool and the ceiling still has to hold.
        #     With its experts at 4-bit it occupies about a third of the file (measured 0.55 GB
        #     against a 1.69 GB file); at bf16 the whole file. See bigrig_engine/mtp.py.
        self.mtp_path = os.path.expanduser(mtp) if mtp else None
        self.mtp_bits = int(mtp_bits) if mtp_bits else None
        self.mtp_head, self.mtp_name, self.mtp_gb = None, "", 0.0
        self.mtp_stats = None
        if self.mtp_path:
            from . import mtp as _mtp
            self.mtp_name = os.path.basename(self.mtp_path.rstrip("/"))
            self.mtp_gb = round(_mtp.head_gb(self.mtp_path) * (0.33 if self.mtp_bits else 1.0), 2)
            budget_gb = max(budget_gb * 0.5, budget_gb - self.mtp_gb)
        # THE BUDGET THE POOL IS PLANNED FROM, WHICH IS THE CEILING LESS WHAT SITS BESIDE THE POOL.
        #     `budget_gb` stays the ceiling the user set, because that is what they asked for and
        #     what the page reports. The knee is keyed on THIS number: a knee measured for a bare
        #     model is a number about a different pool from the one a head or a draft leaves room
        #     for -- measured, the bare knee was found and skipped the tune, while the plan could
        #     not use it and ran on an estimate.
        self.pool_budget_gb = round(float(budget_gb), 2)

        # Read from config.json, which costs nothing and needs no model, because the pool cannot
        # be sized sensibly without knowing what a reply will cost.
        self.context_length, self.kv_bytes_per_token = self._model_limits()
        # The reply's own KV cache is deliberately NOT a term here. _token_ceiling enforces it
        # against the footprint that actually resulted, so reserving for it in the plan as well
        # counts it twice -- and the second count is not free. Shrinking the pool raises the miss
        # rate, and every miss is a buffer read into host memory on the way to a slot. Measured
        # on the same 4,109-token prompt, peak memory went the wrong way:
        #
        #     44/128 resident, 5.03 GB of weights   ->  7.80 GB peak
        #     26/128 resident, 3.25 GB of weights   ->  9.07 GB peak, over a 9.00 GB ceiling
        #
        # A smaller pool is not automatically a safer one. Below some point the transient cost of
        # missing exceeds what was saved by not holding the expert.
        # The manifest has to come first: the reserve depends on how big this model's experts
        # are, and the pool cannot be sized without knowing what is resident regardless.
        #
        # No implicit packing. A packed blob is used if one is already there; otherwise the
        # experts are read out of the model's own safetensors, which costs nothing on disk.
        man, blob = stream.expert_source(self.model_dir)
        self.packed = bool(blob)
        top_k = stream.model_top_k(self.model_dir, man)
        # config_dir, not model_dir. A session started on a hub repo id has no directory at
        # model_dir -- the experts come from a blob keyed on the basename and mlx_lm resolves the
        # id against the cache -- so this read failed and returned 0.0, silently telling the
        # planner that a model's attention weights, embeddings and norms were free. On this model
        # that is 0.67 GB unreserved against a 9 GB ceiling. The CLI happened to be safe because
        # it downloads to a real directory first; anything using the Python API was not.
        self.non_expert_gb = precision.non_expert_gb(self.config_dir, manifest=man)
        self.working_memory_gb = self._working_memory(man, top_k)
        # Charged to the reserve BEFORE the pool is planned, so the pool is one that leaves room
        # for it. Adding it afterwards would mean the ceiling the user set is not the ceiling.
        self.prompt_cache_gb = max(0.0, float(PROMPT_CACHE_GB if prompt_cache_gb is None
                                              else prompt_cache_gb))
        self.kv_bits = (int(kv_bits) if kv_bits else None) or KV_BITS
        if self.kv_bits not in (None, 2, 3, 4, 5, 6, 8):
            raise ValueError(f"kv_bits must be one of 2,3,4,5,6,8 or None, got {self.kv_bits}")
        self.kv_quant_start = int(KV_QUANT_START if kv_quant_start is None else kv_quant_start)
        self.serving_reserve_gb = round(OS_AND_RUNTIME_GB + self.working_memory_gb
                                        + self.prompt_cache_gb, 2)
        if not self.kv_bytes_per_token:      # nothing to derive from; keep the tuned constant
            self.serving_reserve_gb = autoconfig.RESERVE_GB
        try:
            self.plan = autoconfig.choose_capacity(man, budget_gb=budget_gb, top_k=top_k,
                                                   reserve_gb=self.serving_reserve_gb,
                                                   non_expert_gb=self.non_expert_gb)
        except MemoryError:
            # The planner refuses a model it cannot fit, which is right when it is the one
            # choosing. It is not right when the caller has already chosen: everywhere else in
            # this engine an explicit request wins, and refusing one here would mean a user who
            # knows their machine better than the estimate cannot say so. Honour it, and say
            # plainly what the estimate thought.
            if capacity is None:
                raise
            n_exp = max(int(l["n_experts"]) for l in man["layers"].values())
            per = max(int(l["bytes_per_expert"]) for l in man["layers"].values())
            cap_i = int(round(capacity * n_exp)) if capacity <= 1.0 else int(capacity)
            self.plan = {"capacity": cap_i, "n_experts": n_exp,
                         "residency": cap_i / n_exp, "top_k": top_k,
                         "pool_gb": cap_i * per * len(man["layers"]) / 1e9,
                         "n_layers": len(man["layers"]), "bytes_per_expert": per,
                         "fits_entirely": False, "available_gb": budget_gb,
                         "reserve_gb": self.serving_reserve_gb, "over_plan": True,
                         "model_expert_gb": sum(int(l["bytes_per_expert"]) * int(l["n_experts"])
                                                for l in man["layers"].values()) / 1e9}
            if verbose:
                print(f"  WARNING: the estimate says this model does not fit in "
                      f"{budget_gb:.1f} GB -- {self.non_expert_gb:.2f} GB is resident whatever "
                      f"the pool does, a step needs about {self.working_memory_gb:.2f} GB, and "
                      f"{top_k} experts a layer is the floor. Running at the capacity you asked "
                      f"for anyway. Watch memory.")

        # HOW to run it, not just how much of it to keep. Preference order is native, then
        # compress, then stream -- which is also increasing order of what it costs the user:
        # nothing, then accuracy, then speed.
        # Skipped entirely when the caller named a capacity: the answer is discarded two lines
        # below, and asking a planner that may refuse for an answer nobody uses turns an explicit
        # request into a crash.
        self.strategy = None if capacity is not None else autoconfig.choose_strategy(
            man, budget_gb=budget_gb, top_k=top_k, reserve_gb=self.serving_reserve_gb,
            non_expert_gb=self.non_expert_gb,
            min_bits=autoconfig.DEFAULT_MIN_BITS if min_bits is None else min_bits)
        if capacity is not None:                       # an explicit request always wins
            self.strategy = {"mode": "stream",
                             "capacity": (int(round(capacity * self.plan["n_experts"]))
                                          if capacity <= 1.0 else int(capacity)),
                             "n_experts": self.plan["n_experts"],
                             "reason": "residency was set explicitly"}
            self.strategy["residency"] = (self.strategy["capacity"] /
                                          self.strategy["n_experts"])
        elif force_stream and self.strategy["mode"] == "native":
            self.strategy = {"mode": "stream", "capacity": self.plan["capacity"],
                             "n_experts": self.plan["n_experts"],
                             "residency": self.plan["residency"],
                             "reason": "streaming was forced, for measurement"}

        # A MEASURED KNEE BEATS AN ARITHMETIC ONE, AND ONLY A MEASURED ONE IS USED.
        #     `autoconfig` sizes the pool from what FITS. `bigrig knee` sizes it from what was
        #     TIMED on this machine, and the gap is not small: on Qwen3-30B the last 0.89 GB of
        #     pool bought 4% speed and cost 9,000 tokens off the longest reply. Memory that buys
        #     nothing is memory the rest of the machine should have.
        #
        #     Three conditions, all necessary. The caller did not name a capacity, because an
        #     explicit request wins everywhere else in this engine and must here too. The model
        #     is being streamed, because a resident model has no capacity to choose. And the
        #     cached measurement was taken against THIS budget -- `knee.load` refuses it
        #     otherwise, which is what stops the same class of bug as reusing another model's
        #     sync curve, a mistake that shipped for weeks and made models slower.
        self.knee = None
        if capacity is None and self.strategy and self.strategy["mode"] == "stream":
            k = _knee.load(self.name, budget_gb)
            # A knee measured before the model changed shape would be a number about a different
            # model wearing the same name.
            if k and int(k.get("n_experts") or 0) == int(self.plan["n_experts"]):
                want = max(1, min(int(k["capacity"]), int(self.plan["n_experts"])))
                self.knee = k
                if want != self.strategy["capacity"]:
                    self.plan_lines.append(
                        f"  using the measured knee for this model: {want} experts a layer "
                          f"instead of {self.strategy['capacity']} -- {k.get('why', '')}")
                self.strategy["capacity"] = want
                self.strategy["residency"] = want / self.strategy["n_experts"]
                self.strategy["reason"] = "measured on this machine by `bigrig knee`"

        # CONSENT. `compress` is the only mode that alters the weights, so it is the only one
        # that needs agreement. Everything else passes straight through. With no flag, no
        # remembered choice and no terminal, this RAISES rather than choosing -- a script that
        # quietly serves a degraded model is worse than one that stops.
        _l0 = man["layers"][sorted(man["layers"], key=int)[0]]
        src_bits = int((_l0.get("quant") or {}).get("bits", 0))
        # AN UNQUANTISED MODEL HAS NO BIT COUNT, AND SAYING "0-bit" IS WORSE THAN SAYING NOTHING.
        #     `quant` is correctly absent for weights that were never quantised, so this read 0
        #     and the disclosure line -- the one sentence that always names what is being served
        #     -- came out as "running EXACT at 0-bit". For the one model whose whole point is
        #     that it is full precision. The dtype knows.
        self.source_precision = _dtype_label(_l0.get("spec") or {}) if not src_bits \
            else f"{src_bits}-bit"
        self.source_bits = src_bits or _dtype_bits(_l0.get("spec") or {})
        self.strategy, self.decision = consent.resolve(
            self.strategy, blob, src_bits, self.name, preference=preference,
            interactive=interactive, remember=remember)
        if self.strategy["mode"] == "stream" and "capacity" not in self.strategy:
            # Compression was declined, so fall back to whatever residency actually fits.
            self.strategy["capacity"] = self.plan["capacity"]
            self.strategy["n_experts"] = self.plan["n_experts"]
            self.strategy["residency"] = self.plan["residency"]

        mode = self.strategy["mode"]
        if mode == "native":
            self.model, self.tokenizer = stream.load_lenient(self.model_dir, lazy=False)
            self.streamed = False
        else:
            use_blob = blob
            if mode == "compress":
                use_blob = precision.ensure_compressed(
                    blob, self.strategy["bits"], self.strategy["group_size"], verbose=verbose,
                    manifest=man, name=self.name)
                man = stream.load_manifest(use_blob)
                cap = man["layers"][sorted(man["layers"], key=int)[0]]["n_experts"]
            else:
                cap = self.strategy["capacity"]
            # A NON-UNIFORM SPLIT IS ONLY ATTEMPTED WHEN THIS MODEL HAS BEEN MEASURED.
            #     Holding a layer at C == E makes it sync-free, which is worth more than the
            #     experts it costs on some models and less on others. The arithmetic has been
            #     right the whole time; what was wrong was feeding it Qwen3's constants for every
            #     model. With no measurement for THIS model, uniform capacity ships, exactly as
            #     before. `bigrig calibrate <model>` is what produces one.
            # A NON-UNIFORM SPLIT IS USED ONLY IF IT HAS BEEN MEASURED FASTER ON THIS MODEL.
            #     Not predicted faster. With both curves measured per-model the planner still
            #     predicted 1.63x and delivered 0.81x -- it made the model slower. The sync curve
            #     is noisy enough to come out non-monotonic between runs, and a planner fed a
            #     noisy curve picks a configuration nobody has run. `bigrig calibrate` runs the
            #     split against uniform and records the ratio it actually got; only that number
            #     is trusted here.
            self.sync_plan = None
            self.planned_from = None
            full = tuple(full_layers) if full_layers else ()
            if full_layers is None:
                curve = synccal.load_curve(self.name) or {}
                got = curve.get("verified_speedup")
                vp = curve.get("verified_plan") or {}
                if got and got > 1.02 and vp.get("full_layers"):
                    full = tuple(range(int(vp["full_layers"])))
                    cap = int(vp["capacity"])
                    self.sync_plan = {**vp, "verified_speedup": got}
                    if verbose:
                        print(f"  measured split: {vp['full_layers']} layers held whole (no host "
                              f"round-trip) + the rest at {cap}/{self.plan['n_experts']} -- "
                              f"{got:.2f}x, measured on this machine")
                elif not no_full_layers:
                    # NO VERIFIED CURVE, SO PLAN IT -- WHICH IS NEW, AND HERE IS WHY IT IS SAFE.
                    #     The paragraph above refuses to plan from a predicted speedup, because
                    #     the old planner did exactly that and shipped a configuration nobody
                    #     had run. This does not predict a speedup. It redistributes the SAME
                    #     slot budget so that as many layers as possible hold every expert and
                    #     therefore never read their routing back to the host -- which is 2.23 ms
                    #     a layer, 107 ms of a 127 ms token, the largest single cost in the
                    #     engine.
                    #
                    #     The mechanism is measured, not assumed: 27 samples a side over three
                    #     interleaved rounds gave 10.49 tok/s uniform against 12.90 split, 1.23x,
                    #     at 3.51 GB against 3.57 -- less memory, not more -- with a sample from
                    #     the split beating a uniform one 86% of the time. Output is unchanged,
                    #     6 of 6 replies byte-identical, because only which experts sit where has
                    #     moved; the weights and the arithmetic have not.
                    # THE BUDGET, KEPT SEPARATELY FROM WHAT THE PLAN PRODUCED.
                    #     The plan turns a uniform capacity into "n whole layers plus the rest
                    #     at a smaller number", and that smaller number is what gets reported
                    #     back. A memory controller that treats the reported number as home
                    #     will grow back to 11 and re-plan into 0 whole layers -- losing the
                    #     1.23x it started with, quietly. The budget is what home means.
                    self.planned_from = cap
                    # WITH THE HEAD ON, EVERY STREAMED LAYER NEEDS ROOM FOR TWO TOKENS' EXPERTS.
                    #     A verify pass carries two tokens, each routing to top_k experts, and a
                    #     layer with fewer than 2*top_k+1 slots splits that pass and evicts
                    #     between the halves. Measured at 10 slots a layer: the miss rate rose
                    #     from 0.61 to 0.73 and the head ran at 0.91x. So the whole-layer plan
                    #     may not take the streamed layers below that floor when the head is on.
                    nf, rest = autoconfig.plan_full_layers(
                        self.plan["n_layers"], self.plan["n_experts"], top_k, cap,
                        floor_over_topk=(top_k + 1 if self.mtp_path else 1))
                    if nf:
                        full = tuple(range(nf))
                        cap = rest
                        self.sync_plan = {"full_layers": nf, "capacity": rest,
                                          "source": "planned"}
                        self.plan_lines.append(
                            f"  {nf} layers held whole so they never pause to ask the CPU "
                            f"which experts to load;\n  the other "
                            f"{self.plan['n_layers'] - nf} hold {rest} of "
                            f"{self.plan['n_experts']}. Same memory, same answers.")
            # A predictor is used only to start disk reads early. It cannot change which
            # experts execute, so a model with no predictor behaves exactly as before.
            pred = predict.load(self.name)
            if pred and prefetch_width:
                pred["width"] = int(prefetch_width)
            self.predictor = pred if (pred and prefetch_width) else None
            self.model, self.tokenizer, self.handle = stream.load_streaming(
                self.model_dir, use_blob, capacity=cap, policy=policy, full_layers=full,
                threads=threads, verbose=verbose and announce, manifest=man, warm=warm,
                predictors=self.predictor, reroute=reroute, nocache=nocache)
            self.reroute_tol = float(reroute or 0.0)
            self.streamed = True
            hs = self.handle.stats()
            self.plan["capacity"] = hs["capacity"]
            self.plan["residency"] = (self.plan["capacity"] / hs["n_experts"])
            # The loader's own summary line, kept with the plan so a quietly built session can
            # still be announced by whoever serves it.
            self.plan_lines.append(
                f"  {int(hs.get('n_layers', 0) or len(self.handle.mods)) - int(hs.get('sync_free', 0))} "
                f"streamed layers at {hs['capacity']}/{hs['n_experts']}, "
                f"{hs.get('sync_free', 0)} sync-free, pool {hs.get('resident_gb', 0):.2f} GB of "
                f"{hs.get('disk_gb', 0):.2f} GB on disk")

        # Loaded after the target, so a draft that will not work is refused with the target
        # already in hand and the error can say what it was checked against.
        if draft:
            self._load_draft(os.path.expanduser(draft), verbose)
        if self.mtp_path:
            self._load_mtp(verbose)

        self.load_seconds = time.perf_counter() - t0
        if verbose and announce:
            for line in self.plan_lines:
                print(line, flush=True)
        self.meter = AdaptiveMeter() if (monitor and _HAVE_METER) else None
        self.vocab_size = int(getattr(self.tokenizer, "vocab_size", 0) or
                              len(getattr(self.tokenizer, "get_vocab", dict)()) or 0)
        self.flagged_tokens = 0
        self.total_tokens = 0
        # RUNS OF FLAGS, NOT SINGLE FLAGS, ARE WHAT DEGRADATION LOOKS LIKE.
        #     The meter judges each token against this model's own normal at z = 2.5, so about
        #     one healthy token in a hundred trips it by chance -- measured, 2 of 281 on a clean
        #     first reply. What compression damage and looping actually produce is a RUN of
        #     flagged tokens. So the count of runs of three or more is kept beside the raw flag
        #     count, and the words a person sees are keyed on the runs and the share, never on a
        #     lone flag.
        self.flag_runs = 0
        self._flag_run = 0
        self.draft_accepted = 0
        self._last_logits = None
        self._warned_template = False
        # Detected once, at load. The UI shows it, because a model that cannot be chatted with
        # will produce a fabricated dialogue and there is no way to tell that from a bad answer.
        self.has_chat_template = self._detect_chat_template()
        # Reasoning models spend most of a token budget thinking. Measured on Qwen3-30B: 296
        # words of reasoning against 92 words of answer -- 72% of the budget -- and the answer
        # was then cut off mid-sentence by the limit. Being able to turn it off is the
        # difference between "the model is broken" and "you chose to see the working".
        self.can_toggle_thinking = self._detect_thinking_toggle()
        # Some models answer in a channelled transcript rather than plain text. Detected once,
        # because the rewrite it needs is not free and every other model must not pay for it.
        self._harmony = self._detect_harmony()
        # WHETHER THIS MODEL'S PROMPT LEAVES IT MID-THOUGHT.
        #     Qwen3.6 and GLM-4.x end the generation prompt with an OPEN `<think>`, so the reply
        #     begins inside the reasoning block and closes it with `</think>` before answering.
        #     Qwen3-30B opens and closes the block itself. Either way the reasoning is not the
        #     answer, and an API that returns it as `content` hands a coding agent the model's
        #     scratchpad. Set per request in `_prompt`.
        self._starts_in_reasoning = False
        # What the model itself allows, and what a long reply actually costs. Both are read from
        # the checkpoint, never assumed: a reply limit picked out of the air is a restriction the
        # user cannot reason about, and on a machine chosen for having too little RAM the KV
        # cache is not a rounding error -- Qwen3-30B is 96 KB per token, 4 GB at full context.
        # Measured, not predicted: the pool reports what it planned to hold, but quantisation
        # block sizes and alignment mean what MLX actually holds is the only figure worth
        # subtracting from a ceiling.
        # MEASURED AFTER THE ALLOCATOR HAS SETTLED, WHICH IS NOT THE MOMENT THE LOAD FINISHES.
        self._settle()
        # MEASURED, NOT ASSUMED: in views mode MLX's counter already includes the imported
        # page-cache buffers (4.03 GB after decode in both modes), and the Mac's own footprint of
        # the process is LOWER by the pool's size (4.06 against 4.62 GB), because a file-backed
        # page serving the GPU is still the same page the cache holds. Charging the views on
        # top would count them twice and shorten every reply for nothing.
        self.footprint_gb = mx.get_active_memory() / 1e9
        # The config estimate above is a plan made before the model existed. Now it does exist,
        # and it can be asked directly -- which is the only answer that stays right for an
        # architecture nobody has written a rule for yet.
        self._measure_kv()
        self.max_completion_tokens, self.token_limit_reason = self._token_ceiling()
        self.prefill_step = self._prefill_step()
        # A hashable stand-in for the model, which is what the trie keys on -- an nn.Module is
        # not hashable. One session serves one model, so a constant would do; the name is used so
        # a shared cache would still be correct.
        self._cache_key = str(self.model_dir)
        self._generated_ids = []
        self.prompt_cache_hits = self.prompt_cache_misses = self.prompt_cache_reused = 0
        self.prompt_cache_matched = 0
        self._cache_proven = False
        self._prompt_cache = self._make_prompt_cache()
        # Named on the instance so the batch planner prices against the same numbers the pool was
        # planned with, rather than importing constants and drifting from them.
        self.capacity = int(self.plan.get("capacity") or 0)
        self.top_k = int(self.plan.get("top_k") or 0)
        # An explicit --residency overrides the plan, which means it can also override the room
        # the plan left for the reply. Saying so is the difference between a server that is
        # slow and one that is quietly unusable.
        if (verbose and self.token_limit_reason == "memory"
                and self.max_completion_tokens <= MIN_REPLY_TOKENS):
            want = self.footprint_gb + self.working_memory_gb + \
                TARGET_REPLY_TOKENS * self.kv_bytes_per_token / 1e9
            print(f"  WARNING: {self.footprint_gb:.2f} GB of experts leaves no room for a reply "
                  f"under a {self.budget_gb:.1f} GB budget. Replies are capped at "
                  f"{self.max_completion_tokens} tokens. An {TARGET_REPLY_TOKENS:,}-token reply "
                  f"would need {want:.2f} GB in total -- lower --residency, or raise the budget.")

    # ------------------------------------------------------------------ generation
    def _detect_chat_template(self) -> bool:
        try:
            out = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": "x"}], add_generation_prompt=True, tokenize=False)
            return bool(out) and out.strip() != "x"
        except Exception:
            return False

    def _model_limits(self) -> tuple:
        cfg = os.path.join(self.config_dir, "config.json")
        try:
            with open(cfg) as f:
                c = json.load(f)
        except (OSError, ValueError):
            return 0, 0
        c = {**c.get("text_config", {}), **c}
        ctx = c.get("max_position_embeddings") or c.get("max_seq_len") or 0
        per, fixed = kv_bytes(c)
        self.kv_fixed_bytes = fixed
        return int(ctx), per

    def _settle(self) -> None:
        """Run one small multi-token pass so the footprint below is the steady state, not a peak.

        THE BUG THIS FIXES, WHICH COST EVERY USER 93% OF THEIR REPLY LENGTH.
            Filling the pool rebinds each slot tensor rather than writing into it -- MLX arrays
            are functional, so `slots[k][i] = expert` produces a new (C, ...) tensor and abandons
            the old one. Warming does that once per expert, and the abandoned versions stay
            counted as active until something forces the allocator to reuse them. On this model
            that was 2.64 GB of nothing, sitting in `get_active_memory()` at exactly the moment
            the footprint was read.

            The footprint is then subtracted from the budget for the whole life of the session to
            decide how long a reply may be. So 9.00 GB less a 6.82 GB footprint less 3.00 GB of
            working memory came to MINUS 0.82 GB, the ceiling collapsed to the 256-token floor,
            and every reply was cut at 256 tokens on a machine with room for 18,000. Measured
            afterwards: a 598-token reply peaked at 6.83 GB, and active memory during generation
            sat at 4.21 GB -- the 2.64 GB was never real.

            `mx.clear_cache()`, `gc.collect()` and `mx.synchronize()` all leave it in place. A
            single-token pass leaves it too, which is why `_measure_kv` -- which runs one -- did
            not already fix this by accident. Two tokens is enough, because more than one token
            routes to more experts than a streamed pool holds and that forces the split path to
            allocate for real. Eight, for margin, at a measured cost of about 0.3 s.

        Never fatal. A model that cannot take this pass still loads; the footprint is then the
        old over-estimate, which is the behaviour that shipped before this existed.
        """
        try:
            from mlx_lm.models.cache import make_prompt_cache
            n = max(2, min(8, int(self.plan.get("top_k") or 8)))
            pad = self.tokenizer.eos_token_id or 0
            c = make_prompt_cache(self.model)
            y = self.model(mx.array([[pad] * n]), cache=c)
            mx.eval(y)
            del c, y
        except Exception:                       # noqa: BLE001 -- a probe must never cost a load
            return
        mx.clear_cache()

    def _measure_kv(self) -> None:
        """Replace the config estimate with what this model's own cache objects actually hold.

        WHY MEASURING BEATS ANY AMOUNT OF READING config.json
            `kv_bytes` reads the checkpoint and gets twelve architecture families right, which
            required a separate rule for MLA, for sliding windows, for linear-attention hybrids,
            for Mamba block patterns and for two differently-named layer-group fields. Every one
            of those rules was written by reading mlx_lm's `make_cache`, and every new
            architecture is a chance to need a thirteenth.

            The model is loaded by the time this runs, so it can just be asked. One forward pass
            of a single token fills the caches it really uses; a cache that grows with the reply
            reports its own width, and one that does not is counted once. `KVCache` allocates in
            blocks of `step`, so the per-token cost is the block divided by the block size --
            exact, not sampled.

        A FAILURE HERE MUST NOT COST A LOAD. If anything about this model's cache is shaped
        differently than expected, the config estimate stands and the model still runs.
        """
        from mlx_lm.models import cache as _cache
        try:
            caches = self.model.make_cache()
        except (AttributeError, TypeError, KeyError):
            return
        if not caches:
            return
        try:
            ids = mx.array([[self.tokenizer.eos_token_id or 0]])
            self.model(ids, cache=caches)
            mx.eval([getattr(c, "keys", None) for c in caches if hasattr(c, "keys")])
        except Exception:                       # noqa: BLE001 -- any failure keeps the estimate
            return

        per_token, fixed = 0, 0
        for c in caches:
            k, v = getattr(c, "keys", None), getattr(c, "values", None)
            if k is None:
                # A recurrent state: fixed in size however long the reply runs.
                for arr in (getattr(c, "state", None) or []):
                    fixed += getattr(arr, "nbytes", 0) or 0
                continue
            block = (k.nbytes or 0) + ((v.nbytes or 0) if v is not None else 0)
            step = max(1, int(getattr(c, "step", 1) or 1))
            if isinstance(c, _cache.RotatingKVCache):
                fixed += block              # capped at its window, so it stops growing
            else:
                per_token += block // step
        if per_token or fixed:
            self.kv_bytes_per_token = int(per_token)
            self.kv_fixed_bytes = int(fixed)
        # The pass above is a real forward pass, so it leaves the pool holding whatever it
        # touched and the counters showing a token nobody asked for. Both are reset.
        if self.handle:
            self.handle.reset_stats()

    @staticmethod
    def _working_memory(manifest, top_k: int) -> float:
        """What a step costs beyond the resident weights and the KV cache.

        Measured on two models that differ by six times in expert size and still landed within
        0.6 GB of each other once the prefill width was bounded -- see WORKING_MEMORY_GB. The
        signature keeps the manifest and top-k so a per-model term can be reintroduced if a third
        model shows one is needed, rather than inventing one from two points.
        """
        return WORKING_MEMORY_GB

    def _load_mtp(self, verbose: bool = True) -> None:
        """Load the model's own multi-token-prediction head, refusing a model it cannot drive.

        Refused rather than skipped: a user who asked for it and got a silent plain run would
        read the speed as the head's and draw the wrong conclusion.
        """
        from . import mtp as _mtp
        why = _mtp.supports(self.model)
        if why:
            raise ValueError(f"the MTP head {self.mtp_name} cannot drive {self.name}: {why}")
        tm = _mtp.text_model(self.model)
        self.mtp_head = _mtp.load_head(self.mtp_path, tm.args, quantize_bits=self.mtp_bits)
        self.mtp_stats = _mtp.Stats()
        if verbose:
            how = f"{self.mtp_bits}-bit experts" if self.mtp_bits else "bf16"
            print(f"  MTP head {self.mtp_name} ({how}, {self.mtp_gb:.2f} GB) guesses one token "
                  f"ahead; the model checks every guess", flush=True)

    def _load_draft(self, path: str, verbose: bool = True) -> None:
        """Load the model that proposes tokens, and refuse one that cannot check out.

        A draft with a different vocabulary produces token ids the target model reads as
        different words. Nothing raises -- the ids are all valid -- and the output is fluent and
        wrong, which is the worst failure this project has. So it is checked before anything runs.
        """
        # The engine's own loader, not mlx_lm's. Community checkpoints routinely ship a
        # quantised lm_head alongside tie_word_embeddings, which the strict load refuses outright
        # -- the exact draft used to measure this feature is one of them.
        model, tok = stream.load_lenient(path, lazy=False)

        # COMPARING TWO VOCABULARY SIZES IS NOT THE SAME AS COMPARING TWO VOCABULARIES.
        #     The first version of this check read `vocab_size` from the draft's config and
        #     compared it against the target's TOKENIZER, which are different quantities: a
        #     config's vocab_size is the embedding matrix, usually padded up, while a tokenizer's
        #     is how many tokens it actually has. It reported 151,936 against 151,643 for two
        #     models from the same family that agree on every token. The question that matters is
        #     whether the draft's ids mean the same words, so that is what is asked.
        probes = ["Hello, world!", "def f(x): return x * 2\n", "The quick brown fox.",
                  "\u4f60\u597d", "1234567890", "  leading and trailing  "]
        for text in probes:
            try:
                a, b = tok.encode(text), self.tokenizer.encode(text)
            except Exception as e:
                raise ValueError(f"the draft model's tokenizer could not be compared ({e})")
            if a != b:
                raise ValueError(
                    f"the draft model tokenises differently from {self.name}: {text!r} becomes "
                    f"{a[:8]} for the draft and {b[:8]} for the target. A draft proposes token "
                    f"IDS, so different tokenisation means it proposes different WORDS -- nothing "
                    f"would raise and the output would be fluent and wrong. Use a draft from the "
                    f"same family.")
        self.draft_model = model
        if verbose:
            print(f"  draft {self.draft_name} ({self.draft_gb:.2f} GB) proposing "
                  f"{self.draft_tokens} tokens per step")

    def _make_prompt_cache(self):
        """The store of attention state for prefixes already read. None when it is turned off.

        Bounded in BYTES, not entries. An entry is one conversation and conversations differ in
        length by orders of magnitude, so a limit counting entries would be a limit on nothing.
        `max_size` is set high enough to be inert and the byte budget does the work.
        """
        if self.prompt_cache_gb <= 0:
            return None
        try:
            import mlx_lm.models.cache                       # noqa: F401 -- availability check
        except ImportError:
            return None
        return _TwoStagePromptCache(int(self.prompt_cache_gb * 1e9))

    def trim_prompt_cache(self, to_bytes: int = 0) -> int:
        """Give back remembered prompts. Returns the bytes released.

        THE FIRST THING TO GO WHEN THE MACHINE IS SHORT. Everything else the engine holds is
        needed to answer at all -- weights, the pool, the live KV cache. This is the only thing
        that is purely an accelerator, so under pressure it is the only thing that can be dropped
        without changing what the engine can do, and dropping it costs a re-read, not an answer.
        """
        if self._prompt_cache is None:
            return 0
        before = int(self._prompt_cache.nbytes)
        self._prompt_cache.trim_to(n_bytes=int(max(0, to_bytes)))
        return before - int(self._prompt_cache.nbytes)

    def _prefill_step(self) -> int:
        """How many prompt tokens to push through the model in one forward pass.

        A FUNCTION OF THE MODEL, NEVER OF THE POOL. See PREFILL_ACT_BYTES_PER_UNIT for the
        measurements. The short version is that this number changes what the model says -- two
        step sizes give two different replies to the same long prompt -- so anything that can
        move it at runtime is a bug, and capacity moves at runtime.

        mlx_lm defaults to 2048, which is fine on a machine sized for the model and is not what
        this engine is for. The activations for a pass that wide are the largest single draw on
        memory here -- larger than the KV cache, and larger than anything decode ever allocates.
        512 already exceeded a 9 GB ceiling on a 1,208-token prompt and the guard killed the
        process; the shipped default of 2048 is four times that again.

        Clamped to PREFILL_STEP at the top because that is the widest pass measured safe on any
        model here, and to 16 at the bottom because below that the per-pass overhead dominates
        and time to first token gets worse without buying back meaningful memory.
        """
        top_k = int(self.plan.get("top_k") or 0)
        inter = 0
        try:
            with open(os.path.join(self.config_dir, "config.json")) as f:
                c = json.load(f)
            c = {**c.get("text_config", {}), **c}
            inter = int(c.get("moe_intermediate_size")
                        or c.get("intermediate_size") or 0)
        except (OSError, ValueError, TypeError):
            inter = 0
        unit = top_k * inter
        if unit <= 0:
            # Nothing to size against. The floor rather than the ceiling: an unrecognised
            # architecture is exactly the case where a wide pass has not been shown to be safe.
            return 16
        from . import stream as _st
        if getattr(self, "packed", False) and getattr(_st, "ZERO_COPY", False):
            step = int(PREFILL_ACT_BUDGET_GB * 1e9 / (unit * PREFILL_ACT_BYTES_PER_UNIT_PACKED))
            return max(16, min(PREFILL_STEP_PACKED, step))
        step = int(PREFILL_ACT_BUDGET_GB * 1e9 / (unit * PREFILL_ACT_BYTES_PER_UNIT))
        return max(16, min(PREFILL_STEP, step))

    def _token_ceiling(self) -> tuple[int, str]:
        """The longest reply this server will accept, and which of the two limits produced it.

        The model's context window is the hard limit: past `max_position_embeddings` the position
        encodings are outside anything the model saw in training, and quality falls off without
        anything failing. Memory is the other, and on a machine picked for having too little of
        it that is usually the one that binds first -- the KV cache grows by `kv_bytes_per_token`
        for every token generated, on top of weights that are already most of the budget.

        Reporting the smaller of the two, and saying which, is the whole point. A limit the user
        cannot account for reads as an arbitrary restriction; this one they can check against the
        numbers on the analytics page.
        """
        ctx = self.context_length or ASSUMED_CONTEXT
        if not self.kv_bytes_per_token or not self.budget_gb:
            return ctx, "context"
        # Windowed and recurrent layers cost the same whether the reply is 10 tokens or 10,000,
        # so they come off the top rather than being charged per token.
        spare = ((self.budget_gb - self.footprint_gb - self.working_memory_gb) * 1e9
                 - getattr(self, "kv_fixed_bytes", 0))
        per_token = self.kv_bytes_per_token
        if self.kv_bits:
            # Past `kv_quant_start` a token costs `kv_bits` of 16 plus the group's scale and bias.
            # Everything before it stays fp16, so this is what the cache costs in the limit rather
            # than from the first token -- which is the right term for a CEILING.
            per_token = max(1, int(per_token * (self.kv_bits / 16.0 + 2 * 2.0 / KV_GROUP_SIZE)))
        by_memory = int(spare // per_token)
        if by_memory < ctx:
            return max(by_memory, MIN_REPLY_TOKENS), "memory"
        return ctx, "context"

    def _detect_harmony(self) -> bool:
        """Does this model answer with channel markers rather than plain text?"""
        try:
            out = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": "x"}], add_generation_prompt=True, tokenize=False)
        except Exception:
            return False
        return "<|channel|>" in out or "<|start|>assistant" in out

    def _detect_thinking_toggle(self) -> bool:
        if not self.has_chat_template:
            return False
        try:
            m = [{"role": "user", "content": "x"}]
            a = self.tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
            b = self.tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False,
                                                   enable_thinking=False)
            return a != b
        except Exception:
            return False

    @staticmethod
    def _repair_json(text: str):
        """Parse a tool-call payload, closing brackets the model left open. None if hopeless.

        WHY A REPAIR STEP EXISTS AT ALL, AND WHAT IT IS ALLOWED TO DO
            Measured on this 3-bit model over eight deliberately awkward tool schemas, the two
            failures were not what quantisation was expected to produce. Neither was garbled:

                one was TRUNCATED -- the reply hit max_tokens in the middle of an argument
                one was SHORT ONE BRACE -- otherwise perfect, nested arrays and escapes intact

            Both are recoverable without guessing at meaning. This closes brackets that were left
            open and discards a trailing key whose value never arrived. It does NOT invent
            values, repair quoting, or fix a structure that parses into something different from
            what was written -- a call whose arguments are wrong is worse than no call, because
            the client will run it.

            Scanning has to respect strings, or a brace inside `"text": "{\"a\": 1}"` -- which is
            exactly what one of the failures contained -- is counted as structure.
        """
        t = (text or "").strip()
        if not t:
            return None
        try:
            return json.loads(t)
        except ValueError:
            pass
        stack, in_str, esc = [], False, False
        for ch in t:
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch in "{[":
                stack.append(ch)
            elif ch in "}]":
                if stack and stack[-1] == ("{" if ch == "}" else "["):
                    stack.pop()
        head = t + ('"' if in_str else "")
        for _ in range(3):
            cand = head + "".join("}" if c == "{" else "]" for c in reversed(stack))
            try:
                return json.loads(cand)
            except ValueError:
                # A trailing fragment -- `"backup": ` with nothing after it, or half a key --
                # cannot be closed into anything valid. Drop back to the last complete pair and
                # try again. Three attempts, because a truncation can leave at most a key, a
                # separator and a partial value.
                cut = max(head.rfind(","), head.rfind("{"), head.rfind("["))
                if cut <= 0:
                    return None
                head = head[:cut] if head[cut] == "," else head[:cut + 1]
        return None

    def supports_tools(self) -> bool:
        """Whether this model was trained to call tools at all.

        Read off the tokenizer rather than assumed: a model with no tool-call delimiters cannot
        signal a call however the prompt is written, and telling a caller it can is worse than
        telling them it cannot.
        """
        return bool(getattr(self.tokenizer, "has_tool_calling", False))

    def tool_splitter(self, tools=None):
        """A splitter for this model's delimiters, or None if it cannot call tools."""
        start = getattr(self.tokenizer, "tool_call_start", None)
        end = getattr(self.tokenizer, "tool_call_end", None)
        parser = getattr(self.tokenizer, "_tool_parser", None) or \
            getattr(self.tokenizer, "tool_parser", None)
        if not start or not parser:
            return None
        return ToolCallSplitter(start, end or "", parser, tools, self._repair_json)

    def extract_tool_calls(self, text: str, tools=None) -> tuple:
        """Split a reply into (visible text, [tool calls]). Never raises.

        WHY THE MODEL'S OWN DELIMITERS AND THE MODEL'S OWN PARSER
            Every family marks a call differently -- Qwen wraps it in `<tool_call>` tags holding
            JSON, others emit Python-ish calls or a bare object. mlx_lm ships eleven parsers and
            the tokenizer already knows which one belongs to this checkpoint, so nothing here
            needs to guess a format or maintain a list.

        A CALL THAT WILL NOT PARSE IS DROPPED FROM THE TEXT, NOT PASSED ON AS PROSE.
            Quantisation is hardest on structured output, and a 3-bit model can close a brace in
            the wrong place. Handing the raw `<tool_call>{"name": ...` back to a client as
            assistant text would put markup the user never wrote into the conversation and
            corrupt the next turn's prompt. The malformed region is removed and the reply is
            returned as text; the client sees a model that did not call a tool, which is the
            truthful reading of what happened.
        """
        start = getattr(self.tokenizer, "tool_call_start", None)
        end = getattr(self.tokenizer, "tool_call_end", None)
        parser = getattr(self.tokenizer, "_tool_parser", None) or \
            getattr(self.tokenizer, "tool_parser", None)
        if not start or not parser or start not in text:
            return text, []
        calls, out, i = [], [], 0
        while True:
            a = text.find(start, i)
            if a < 0:
                out.append(text[i:])
                break
            out.append(text[i:a])
            b = text.find(end, a) if end else -1
            body_end = b if b >= 0 else len(text)
            body = text[a + len(start):body_end]
            i = (body_end + len(end)) if b >= 0 else len(text)
            try:
                parsed = parser(body.strip(), tools)
            except Exception:                    # noqa: BLE001 -- malformed JSON is a normal
                parsed = None                    # outcome at 3 bits, not an error to raise
            if not parsed:
                # The model's own parser refused. Try once more on a repaired payload -- see
                # _repair_json for what that is allowed to mean.
                fixed = self._repair_json(body)
                if isinstance(fixed, dict) and fixed.get("name"):
                    args = fixed.get("arguments")
                    parsed = {"name": fixed["name"],
                              "arguments": args if isinstance(args, dict) else {}}
            if parsed:
                calls.append(parsed)
            if b < 0:
                # Unterminated: the reply was cut off mid-call. Everything from the opening tag
                # on is a fragment of machine syntax and must not reach the user as prose.
                break
        return "".join(out), calls

    THINK_OPEN, THINK_CLOSE = "<think>", "</think>"

    def _note_reasoning_start(self, text: str) -> str:
        """Remember whether a rendered prompt ends inside a reasoning block."""
        self._starts_in_reasoning = text.rstrip().endswith(self.THINK_OPEN)
        return text

    def _split_reasoning(self, text: str) -> tuple:
        """(reasoning so far, answer so far) for what the model has emitted.

        Two shapes, both measured on real templates: the prompt opens the block and the reply
        closes it (Qwen3.6, GLM-4.7), or the reply opens and closes it itself (Qwen3-30B). A
        reply that is still thinking has no answer yet, which is the honest thing to report --
        the model has not said anything to the user.
        """
        close = self.THINK_CLOSE
        if self._starts_in_reasoning:
            i = text.find(close)
            if i < 0:
                return text, ""
            return text[:i], text[i + len(close):].lstrip("\n")
        j = text.find(self.THINK_OPEN)
        if j >= 0 and not text[:j].strip():
            j += len(self.THINK_OPEN)
            i = text.find(close, j)
            if i < 0:
                return text[j:], ""
            return text[j:i], text[i + len(close):].lstrip("\n")
        return "", text

    def _prompt(self, messages=None, prompt: str = "", think: bool = True,
                continue_last: bool = False, tools=None) -> str:
        if not messages:
            return self._note_reasoning_start(prompt)
        kw = {}
        if not think and self.can_toggle_thinking:
            kw["enable_thinking"] = False
        # THE MODEL CANNOT CALL A TOOL IT WAS NEVER TOLD ABOUT.
        #     This is the whole of the "tools" half of tool calling: the chat template renders
        #     the signatures into the prompt, in whatever form this model was trained to expect.
        #     Without it the model is asked to read a file and writes an essay about the file --
        #     a fluent, confident, useless answer, and no error anywhere. Both endpoints returned
        #     HTTP 200 for exactly that before this existed.
        if tools:
            kw["tools"] = list(tools)
        # Continuing leaves the assistant's turn OPEN instead of starting a new one, so the model
        # carries on the sentence it was cut off in rather than restarting the answer.
        kw["continue_final_message" if continue_last else "add_generation_prompt"] = True
        try:
            return self._note_reasoning_start(
                self.tokenizer.apply_chat_template(messages, tokenize=False, **kw))
        except Exception as e:
            # Falling back silently is how a missing chat_template.jinja turns into "the model
            # is a bit dumb" instead of "a file did not download". Say it once, then carry on.
            if not self._warned_template:
                self._warned_template = True
                print(f"  WARNING: this model has no usable chat template ({e}). Falling back to "
                      f"a plain user/assistant format; replies will be worse than they should "
                      f"be. Check that chat_template.jinja downloaded.")
            return self._note_reasoning_start(
                "\n".join(f"{m.get('role','user')}: {m.get('content','')}"
                          for m in messages) + "\nassistant:")

    def stream_text(self, messages=None, prompt: str = "", max_tokens: int = 512,
                    temperature: float = 0.7, top_p: float = 0.95, seed: int | None = None,
                    stop=None, think: bool = True, continue_last: bool = False,
                    prefill_step_size: int | None = None, on_prefill=None,
                    lookahead: bool = False, lookahead_tokens: int = 8, tools=None,
                    mtp: bool | None = None, hide_reasoning: bool = False):
        """Yield (text_chunk, info) as the model generates. `info` carries live quality state.

        Stops at a turn boundary as well as at the model's own EOS. A model without a chat
        template has no reliable EOS for a turn, so without this it writes both sides of the
        conversation and the user sees a fabricated dialogue presented as an answer.
        """
        from mlx_lm.generate import stream_generate
        from mlx_lm.sample_utils import make_sampler
        if seed is not None:
            mx.random.seed(seed)
        sampler = make_sampler(temp=temperature, top_p=top_p)
        text = self._prompt(messages, prompt, think=think, continue_last=continue_last,
                            tools=tools)
        if self.handle:
            self.handle.reset_stats()
        stops = list(stop or ())
        if not self.has_chat_template:
            stops += list(TURN_STOPS)
        hold = max((len(x) for x in stops), default=1) - 1
        # `full` is what the user will see; `raw` is the tail the model has emitted that has not
        # been resolved yet, because it may end part-way through a control marker.
        full, sent, raw = "", 0, ""
        # CAPTURE THE RAW LOGITS, WHICH mlx_lm DOES NOT HAND BACK.
        #     GenerationResponse carries `logprobs`, and log-softmax has already normalised the
        #     logit scale away. The meter needs that scale (see _observe). The model's forward is
        #     wrapped for the duration of this generation and restored afterwards, so nothing
        #     outside it is affected and a crash cannot leave the class patched.
        _mcls = type(self.model)
        _orig_call = _mcls.__call__
        _self = self

        def _capture(mself, *a, **kw):
            out = _orig_call(mself, *a, **kw)
            try:
                _self._last_logits = out[0, -1] if out.ndim == 3 else out
            except Exception:
                _self._last_logits = None
            return out

        gen_kw = {}
        if on_prefill is not None:
            # Reading the prompt is not instant here and the user is looking at a blank window
            # while it happens. On gpt-oss at 3% residency a 68-token prompt -- which is what
            # "hi" becomes once the chat template is applied -- takes about 40 seconds, because
            # every prompt token needs its experts read off disk. Reporting it is the difference
            # between "slow" and "broken".
            gen_kw["prompt_progress_callback"] = on_prefill
        if self.draft_model is not None:
            gen_kw["draft_model"] = self.draft_model
            gen_kw["num_draft_tokens"] = self.draft_tokens
        if self.kv_bits:
            gen_kw["kv_bits"] = self.kv_bits
            gen_kw["kv_group_size"] = KV_GROUP_SIZE
            gen_kw["quantized_kv_start"] = self.kv_quant_start
        # REUSE THE ATTENTION STATE FOR WHATEVER PART OF THIS PROMPT HAS ALREADY BEEN READ.
        #     `text` is passed on unchanged when there is nothing to reuse, so a first turn takes
        #     exactly the path it took before. When there is, the ids are handed over instead --
        #     stream_generate accepts either -- with the cache and only the unread tail.
        #
        #     `full_ids` is kept for the insert AFTER generating: the entry is stored against the
        #     prompt plus the reply, because the next turn's prompt begins with exactly that and
        #     an entry keyed on the prompt alone would be a prefix of what is wanted rather than
        #     the whole of it, and would leave the reply to be read again.
        pc, full_ids, prompt_in = None, None, text
        self._generated_ids = []
        if self._prompt_cache is not None:
            try:
                full_ids = self.tokenizer.encode(text)
                pc, tail, _protected = self._prompt_cache.fetch_nearest_cache(
                    self._cache_key, full_ids)
                _matched = (len(full_ids) - len(tail)) if pc is not None else 0
                # A MATCH IS NOT A HIT UNTIL IT IS WORTH SOMETHING. TWO THRESHOLDS, TWO REASONS.
                #     The trie returns the longest common prefix, and unrelated prompts share
                #     one: two requests whose only difference is an identifier near the front
                #     still agree on the handful of tokens before it. Treating that as a hit was
                #     wrong twice over.
                #
                #     Using it is not free. Serving a match copies the whole stored cache -- 66 MB
                #     on this model -- so reusing six tokens costs a 66 MB copy to save six
                #     tokens of reading. MIN_REUSE_TOKENS is the floor where the copy pays.
                #
                #     Counting it as PROVEN was the worse of the two. Promotion to the protected
                #     segment is meant to mean "this conversation is really being continued", and
                #     a six-token agreement is not that. Measured: twelve requests that each
                #     matched about six tokens of 470 all promoted themselves, took protected from
                #     66 MB to 329 MB, and evicted the one entry that was genuinely being reused.
                #     The segmenting worked perfectly and the signal feeding it was noise.
                if pc is not None and _matched < MIN_REUSE_TOKENS:
                    pc, tail, _matched = None, full_ids, 0
                if pc is not None and len(tail) < len(full_ids):
                    self.prompt_cache_hits += 1
                    self.prompt_cache_reused += len(full_ids) - len(tail)
                    # How much of the prompt was reused, not just that something was. A request
                    # that matched 5 tokens of 500 is a miss wearing a hit's clothes, and without
                    # this number nobody can tell the two apart from outside.
                    self.prompt_cache_matched = _matched
                    self._cache_proven = _matched >= PROVEN_REUSE_FRACTION * len(full_ids)
                    gen_kw["prompt_cache"], prompt_in = pc, tail
                else:
                    from mlx_lm.models.cache import make_prompt_cache
                    self.prompt_cache_misses += 1
                    self.prompt_cache_matched = 0
                    self._cache_proven = False
                    pc = make_prompt_cache(self.model)
                    gen_kw["prompt_cache"], prompt_in = pc, full_ids
            except Exception:            # noqa: BLE001 -- an accelerator must never fail a reply
                pc, full_ids, prompt_in = None, None, text
                gen_kw.pop("prompt_cache", None)
        # GUESSING THE NEXT FEW TOKENS FROM TEXT ALREADY WRITTEN, WHEN THE CALLER ASKS FOR IT.
        #     Off unless asked, per request rather than per server, because whether it is worth
        #     anything is a property of the WORK and not of the machine. Quoting a document back:
        #     94% of guesses accepted, 10.85 -> 17.13 tok/s. Writing original prose: 8% accepted,
        #     nothing bought, and roughly one token in six comes out different -- not because a
        #     wrong guess was accepted, but because a logit computed beside four others is not the
        #     logit computed alone. See bigrig_engine/lookahead.py.
        #
        #     The quality meter is left on the log-probability path here rather than the free
        #     energy one. `_capture` reads the LAST row of whatever the model was called with, and
        #     a verifying pass ends on a position that may have been rejected -- so the energy
        #     reading would be taken from a token the model never emitted. The generator hands
        #     back the right row per token instead, which `_observe` uses when there are no
        #     captured logits. Slower meter, correct meter.
        # THE METER MUST START THIS REPLY, NOT CONTINUE THE LAST ONE.
        #     It was created once per session and never reset, so its 64-token repetition window
        #     spanned the boundary between answers: the opening of one reply sat in the same
        #     window as the closing of the one before. Several short replies to the same question
        #     therefore looked like a single looping one. Measured -- four consecutive 16-token
        #     replies flagged 1, 6, 15 and 15 tokens with nothing wrong.
        #
        #     `keep_baseline=True`: what it has learned about THIS MODEL's normal is the whole
        #     point of a self-calibrating meter and must survive. What must not survive is the
        #     previous answer's tokens.
        if self.meter is not None:
            try:
                self.meter.reset(keep_baseline=True)
                # And what was asked, so repetition the prompt requested is not read as damage.
                self.meter.set_prompt(full_ids if full_ids is not None
                                      else self.tokenizer.encode(text))
            except Exception:            # noqa: BLE001 -- monitoring never costs a reply
                pass
        self.lookahead_stats = None

        def _set_logits(row):
            """The meter's energy reading, taken from the row that actually produced the token."""
            self._last_logits = row

        # THE MODEL'S OWN HEAD, WHEN ONE WAS LOADED AND THE REQUEST DID NOT SAY NO.
        #     Loading the head is the opt-in; a request may still turn it off (`mtp: false`),
        #     which is what makes an A/B on one warm server possible. Lookahead wins if both
        #     are asked for, because it was asked for per request and the head per server.
        self.mtp_last = None
        # What the last real chunk reported, so the final flush of held-back text can carry the
        # same counts rather than zeros. Zeroes here would be a reply that generated nothing.
        counted = {"token": -1, "finish_reason": "stop", "tok_s": None, "from_draft": False,
                   "reasoning_delta": "", "prompt_tokens": 0, "generation_tokens": 0,
                   "degraded": False}
        rsent = 0                      # reasoning characters already handed to the caller
        if self.mtp_head is not None and mtp is not False and not lookahead:
            from . import mtp as _mtp
            self.mtp_last = _mtp.Stats()
            gen_kw.pop("draft_model", None)
            gen_kw.pop("num_draft_tokens", None)
            _produce = lambda: _mtp.stream(                                  # noqa: E731
                self.model, self.mtp_head, self.tokenizer, prompt_in, max_tokens=max_tokens,
                prompt_cache=gen_kw.get("prompt_cache"), sampler=sampler,
                prefill_step_size=(self.prefill_step if prefill_step_size is None
                                   else int(prefill_step_size)),
                stats=self.mtp_last,
                prompt_progress_callback=gen_kw.get("prompt_progress_callback"),
                on_logits=_set_logits)
        elif lookahead:
            from . import lookahead as _la
            self.lookahead_stats = _la.Stats()
            gen_kw.pop("draft_model", None)
            gen_kw.pop("num_draft_tokens", None)
            _produce = lambda: _la.stream(                                   # noqa: E731
                self.model, self.tokenizer, prompt_in, max_tokens=max_tokens,
                prompt_cache=gen_kw.get("prompt_cache"), sampler=sampler,
                prefill_step_size=(self.prefill_step if prefill_step_size is None
                                   else int(prefill_step_size)),
                k=max(1, int(lookahead_tokens)), stats=self.lookahead_stats,
                prompt_progress_callback=gen_kw.get("prompt_progress_callback"),
                on_logits=_set_logits,
                # The whole conversation, not just the part still to be read. When the prompt
                # cache has served a prefix, `prompt_in` is a handful of tail tokens and the
                # document a draft should be quoting from is not in it.
                context_ids=full_ids)
        else:
            _mcls.__call__ = _capture
            _produce = lambda: stream_generate(                              # noqa: E731
                self.model, self.tokenizer, prompt_in, max_tokens=max_tokens,
                sampler=sampler,
                prefill_step_size=(self.prefill_step if prefill_step_size is None
                                   else int(prefill_step_size)),
                **gen_kw)
        try:
            self._flag_run = 0
            for r in _produce():
                flagged = self._observe(r)
                self._flag_run = self._flag_run + 1 if flagged else 0
                if self._flag_run == FLAG_RUN:
                    self.flag_runs += 1
                if pc is not None:
                    # The ids, not the text: re-encoding the reply is not guaranteed to give
                    # back the tokens the model actually emitted, and a key that does not match
                    # what the next prompt encodes to is a cache that never hits.
                    self._generated_ids.append(int(r.token))
                self.total_tokens += 1
                self.flagged_tokens += int(bool(flagged))
                raw += r.text
                # A CONSTRUCT CANNOT BE REWRITTEN A PIECE AT A TIME.
                #     Rewriting only the newly-arrived tail splits `<|channel|>analysis` from its
                #     `<|message|>`, the pair rewrite then does not match, only the catch-all runs,
                #     and the user is shown the bare word "analysis" as though the model had said it.
                #     Measured on gpt-oss before this was fixed: "analysisUser asks: ...
                #     assistantfinalA hash table is ...". So the whole reply is rewritten each time,
                #     and `hold` keeps the last HARMONY_HOLD characters back until they are settled.
                #     Only models that need it pay for it, and those are the slow ones anyway.
                if self._harmony:
                    # Rewrite only what has settled; whatever might still be a construct waits.
                    settled = raw[:len(raw) - _harmony_hold(raw)]
                    full = self.visible(settled)
                else:
                    full = raw
                # THE REASONING IS NOT THE ANSWER. Held apart here rather than in each endpoint,
                # so every API and the web page agree about which is which. `full` becomes the
                # answer only, and every `sent` offset below counts answer characters -- while
                # the model is still thinking the answer is empty and nothing is yielded.
                rdelta = ""
                if hide_reasoning:
                    _reasoned, full = self._split_reasoning(full)
                    if len(_reasoned) > rsent:
                        rdelta, rsent = _reasoned[rsent:], len(_reasoned)
                # Whether the DRAFT proposed this token and the target accepted it. Without this
                # a draft that is simply wrong for the target is indistinguishable from a mechanism
                # that does not pay -- and they need opposite fixes.
                if getattr(r, "from_draft", False):
                    self.draft_accepted += 1
                info = {"token": r.token, "finish_reason": r.finish_reason,
                        "reasoning_delta": rdelta,
                        "from_draft": bool(getattr(r, "from_draft", False)),
                        "tok_s": _sane_tps(r.generation_tps, r.generation_tokens),
                        "prompt_tokens": r.prompt_tokens,
                        "generation_tokens": r.generation_tokens, "degraded": flagged}
                counted = info
                cut = min((full.find(x) for x in stops if full.find(x) >= 0), default=-1)
                if cut >= 0:
                    if cut > sent:
                        yield full[sent:cut], {**info, "finish_reason": "stop"}
                    elif not sent:
                        yield "", {**info, "finish_reason": "stop"}
                    return
                # Hold back anything that could still turn out to be the start of a stop sequence.
                # Emitting "\nUse" and then retracting it is not possible in a stream, so the only
                # way to avoid showing the beginning of a fabricated turn is not to send it yet.
                safe = len(full) - hold
                if safe > sent:
                    yield full[sent:safe], info
                    sent = safe
                elif r.finish_reason:
                    if len(full) > sent:
                        yield full[sent:], info
                        sent = len(full)
                    else:
                        yield "", info
                elif rdelta:
                    # STILL THINKING: NO ANSWER TEXT YET, BUT SOMETHING TO REPORT.
                    #     The reasoning rides on `info`, and `info` only reaches the caller when
                    #     a chunk is yielded -- so without this the whole of a model's thinking
                    #     was computed, split off, and then dropped on the floor. An empty chunk
                    #     is what every consumer here already skips.
                    yield "", info
        finally:
            # EVERYTHING HERE, AND NOTHING ELSE. This used to hold the whole per-token body --
            # all five yields included -- so the generator emitted nothing until generation had
            # finished and then delivered the entire reply in ONE chunk. Measured: a 32-token
            # reply arrived as a single yield. The OpenAI and Anthropic endpoints, the web
            # interface and any client wired to this server were therefore not streaming at all,
            # and at single-digit tokens a second that is forty seconds of blank screen followed
            # by a wall of text. Restoration belongs in `finally`; producing output does not.
            #
            # Restored however this generator ends -- exhausted, raised, or abandoned
            # mid-stream by a client that hung up. Leaving the class patched would
            # outlive the request.
            _mcls.__call__ = _orig_call
            self._last_logits = None
            # What the head bought on this reply, added to the running total the page shows.
            # In `finally` so an abandoned reply still counts what it did.
            if self.mtp_last is not None and self.mtp_stats is not None:
                for k in ("rounds", "drafted", "accepted", "recomputed"):
                    setattr(self.mtp_stats, k, getattr(self.mtp_stats, k) + getattr(self.mtp_last, k))
            # Stored against prompt PLUS reply, because that is exactly what the next turn's
            # prompt begins with. Keyed on the prompt alone it would be a prefix of what is
            # wanted rather than the whole of it, and the reply would be read again.
            #
            # In `finally`, so a reply the client abandoned half way is still kept -- the tokens
            # were read either way, and the next turn is just as likely to want them.
            if pc is not None and full_ids is not None:
                try:
                    self._prompt_cache.insert_cache(
                        self._cache_key, list(full_ids) + list(self._generated_ids), pc,
                        proven=getattr(self, "_cache_proven", False))
                except Exception:        # noqa: BLE001 -- never fail a reply over a cache write
                    pass

        # Generation has stopped, so nothing more is arriving and whatever was held back because
        # it MIGHT have been a construct was in fact the end of the reply.
        #
        # IT CARRIES THE COUNTS FORWARD, BECAUSE THE CALLER READS THEM OFF THE LAST CHUNK.
        #     This used to yield zeros. Both HTTP endpoints keep `last = info` for every chunk and
        #     build `usage` from whichever arrived last, so any reply that ended with held-back
        #     text -- which is most of them, since the tail is held until it cannot be the start
        #     of a stop sequence -- reported `prompt_tokens: 0, completion_tokens: 0` to the
        #     client and logged a 0-token reply with no rate to the analytics page. Caught in the
        #     readiness pass: two of three replies with real text reported zero usage.
        if self._harmony:
            full = self.visible(raw)
        _tail_r = ""
        if hide_reasoning:
            _reasoned, full = self._split_reasoning(full)
            if len(_reasoned) > rsent:
                _tail_r, rsent = _reasoned[rsent:], len(_reasoned)
        if len(full) > sent or _tail_r:
            yield full[sent:], {**counted, "token": -1, "finish_reason": "stop",
                                "reasoning_delta": _tail_r}

    def eos_ids(self) -> set:
        """Every id the tokenizer treats as the end of a turn."""
        ids = set()
        for attr in ("eos_token_id", "eos_token_ids"):
            v = getattr(self.tokenizer, attr, None)
            if isinstance(v, int):
                ids.add(int(v))
            elif v:
                ids.update(int(x) for x in v)
        return ids

    def stream_batch(self, specs: list[dict], emit) -> list[dict]:
        """Serve several requests in ONE forward pass. Calls emit(i, text, info) as text appears.

        WHAT THE MULTI-TOKEN TRADE IS WORTH, AND WHY IT IS NOT TAKEN
            One forward pass over n tokens is cheaper per token, because the 48 per-layer host
            round-trips are shared: 67.7 ms for one token, 123.4 ms for four -- 30.9 ms each,
            a 2.19x ceiling. Past four it collapses (236 ms for six), because six tokens at
            top-8 need 48 distinct experts and the pool holds 36, so the step splits.

            That ceiling is not reachable with what exists. Speculative decoding accepted 53%
            of 2 drafted tokens, 62% of 3 and 64% of 4, with a vocabulary-matched draft costing
            only 6.9 ms a token -- the arithmetic says 1.39x to 1.60x -- and measured 0.95-0.98x
            end to end. Batching measured 1.04x at four concurrent requests and 1.19x at eight.

            And both change the answer, for the same reason, which is not an implementation
            flaw. Verifying n tokens at once changes the shape of the matmul and so the order
            of the reduction:

                one token at a time, re-run against itself   max |difference| 0.000e+00
                four tokens in one pass, same positions      max |difference| 9.7e-01

            Characterised over 24 positions: no token flipped in that sample, the median
            disagreement was 0.742 against a logit span of 32.5 -- 2.28% of the scale -- but 4
            of the 24 positions had a top-1/top-2 margin SMALLER than the disagreement. Those
            are the coin-flips, and one flip early diverges everything after it, which is why
            2 of 5 longer replies came out differently worded.

            So the trade on offer is: at best 1.19x, in exchange for the model not always saying
            the same thing. It is not taken, and both mechanisms stay off by default.

        THIS DOES NOT PRODUCE THE SAME TOKENS AS SERVING THEM ONE AT A TIME, AND CANNOT.
            Streaming experts off disk is bit-exact because it changes where a weight lives and
            never what it is. Batching is a different thing: it changes the shape of the matmul,
            and with it the order the reduction happens in. Measured on this model, greedy, the
            same prompt at batch 1 against batch 2 with no padding involved at all:

                re-run at batch of 1        max |difference| 0.000e+00   exactly reproducible
                batch of 2, same prompt     max |difference| 1.500e+00
                batch of 4, same prompt     max |difference| 1.188e+00
                batch of 2, one padded      max |difference| 1.250e+00

            Padding is not the cause -- identical prompts in one batch give identical rows, so
            the mask is doing its job. bfloat16 carries eight bits of mantissa and this is
            forty-eight layers of accumulation. Where the top two logits are close, greedy then
            picks a different token and the reply diverges: 3 of 4 prompts came out identical,
            the fourth said "stores key-value pairs" where the unbatched run said "implements an
            associative array". Every batched server has this property. It is why batching is
            opt-in here rather than the default, and why the bit-exactness claim is stated for
            one request at a time.

        The quality meter is not run. It holds one adaptive state and B interleaved sequences
        would corrupt it, so `degraded` is reported as None -- not measured, rather than measured
        as fine.
        """
        from mlx_lm.generate import stream_generate                      # noqa: F401  (parity)
        from . import batch as _batch

        prompts, limits = [], []
        for sp in specs:
            text = self._prompt(sp.get("messages"), sp.get("prompt", ""),
                                think=sp.get("think", True),
                                continue_last=sp.get("continue_last", False))
            prompts.append(self.tokenizer.encode(text))
            limits.append(max(1, int(sp.get("max_tokens", 512))))
        stops = tuple(sp for sp in (specs[0].get("stop") or ()) if sp) or \
            (() if self.has_chat_template else TURN_STOPS)
        hold = max((len(x) for x in stops), default=1) - 1

        n = len(specs)
        acc = [[] for _ in range(n)]
        full = [""] * n
        sent = [0] * n
        stopped = [False] * n
        t0 = time.perf_counter()

        def info_for(i, finish=None):
            el = time.perf_counter() - t0
            return {"token": acc[i][-1] if acc[i] else None, "finish_reason": finish,
                    "tok_s": _sane_tps(len(acc[i]) / el if el > 0 else None, len(acc[i])),
                    "prompt_tokens": len(prompts[i]), "generation_tokens": len(acc[i]),
                    "degraded": None, "batched": n}

        def on_token(i, tok_id):
            if stopped[i]:
                return
            if tok_id is None:                      # the row hit an end-of-turn id
                stopped[i] = True
                if len(full[i]) > sent[i]:
                    emit(i, full[i][sent[i]:], info_for(i, "stop"))
                    sent[i] = len(full[i])
                return
            acc[i].append(int(tok_id))
            self.total_tokens += 1
            full[i] = self.tokenizer.decode(acc[i])
            cut = min((full[i].find(x) for x in stops if full[i].find(x) >= 0), default=-1)
            if cut >= 0:
                stopped[i] = True
                if cut > sent[i]:
                    emit(i, full[i][sent[i]:cut], info_for(i, "stop"))
                    sent[i] = cut
                return
            safe = len(full[i]) - hold
            if safe > sent[i]:
                emit(i, full[i][sent[i]:safe], info_for(i))
                sent[i] = safe

        _batch.generate_batch(self.model, prompts, limits, eos_ids=self.eos_ids(),
                              pad_id=(self.tokenizer.eos_token_id or 0),
                              prefill_step=self.prefill_step, on_token=on_token)

        out = []
        for i in range(n):
            fin = "stop" if stopped[i] else ("length" if len(acc[i]) >= limits[i] else "stop")
            if len(full[i]) > sent[i] and not stopped[i]:
                emit(i, full[i][sent[i]:], info_for(i, fin))
                sent[i] = len(full[i])
            out.append(info_for(i, fin))
        return out

    @staticmethod
    def visible(text: str) -> str:
        """What the user should actually see, from what the model actually emitted.

        A no-op for a model that emits plain text: none of the patterns match. Applied in the
        engine rather than the web page so the API returns the same clean text -- a client
        speaking OpenAI has no idea what a channel is either.
        """
        for pat, rep in _HARMONY:
            text = pat.sub(rep, text)
        return text

    def _observe(self, r) -> bool:
        """Feed one step to the meter. Never let monitoring break generation."""
        if self.meter is None or not self.vocab_size:
            return False
        try:
            lp = r.logprobs
            # THE WHOLE LOG-PROBABILITY VECTOR IS NOT NEEDED, AND IS NOT FREE.
            #     Reading it costs 628 KB a token and made the monitor 34% of generation. Free
            #     energy is ONE scalar, and measured on the same labelled set it detects strictly
            #     more: at 1 of 16 layers damaged, entropy scores 0.500 AUROC -- chance -- and
            #     energy 0.917. Dropping the vector read costs nothing in detection and gives the
            #     monitor back: 20.4 -> 35.4 tok/s with it on.
            #
            #     The vector path is kept for when the logits could not be captured, so a model
            #     whose forward this cannot wrap still gets a meter rather than none.
            use_energy = self._last_logits is not None and not _NO_ENERGY
            # FREE ENERGY, WHICH THE LOG-PROBABILITIES CANNOT CARRY.
            #     logprobs = logits - logsumexp(logits), so the softmax has already divided the
            #     scale out -- and the scale is exactly where light damage shows. Measured on
            #     OLMoE with layers deliberately scrambled, AUROC against healthy output: entropy
            #     0.625 at 2 damaged layers of 16 where energy is 1.000. Light damage is the kind
            #     this engine causes on purpose, so a meter blind to it cannot report on the
            #     trades the product makes. The raw logits are captured in stream_text.
            if use_energy:
                self.meter.observe_energy(float(-mx.logsumexp(self._last_logits).item()))
            # mlx_lm hands back the FULL log-probability vector, not a top-K list, so the exact
            # path applies. Routing it through stats_from_topk instead cost 4.36 ms per token in
            # a defensive sort of the whole vocabulary -- 3.5% of the token budget at 8 tok/s.
            if not use_energy:
                if lp is None:
                    return False
                self.meter.observe_stats(*stats_from_logprobs(lp))
            self.meter.observe_token(int(r.token))
            return bool(self.meter.is_degraded())
        except Exception:
            return False

    def generate(self, messages=None, prompt: str = "", **kw) -> dict:
        chunks, last = [], {}
        t0 = time.perf_counter()
        for c, info in self.stream_text(messages, prompt, **kw):
            chunks.append(c)
            last = info
        return {"text": "".join(chunks), "seconds": time.perf_counter() - t0, **last,
                "stats": self.stats()}

    # ------------------------------------------------------------------ reporting
    def close(self) -> None:
        """Give back everything this session holds. Safe to call more than once."""
        h = getattr(self, "handle", None)
        if h is not None and hasattr(h, "close"):
            try:
                if hasattr(h, "save_usage"):
                    h.save_usage(stream.usage_path(self.name))
            except Exception:                   # noqa: BLE001 -- a record, never a failure
                pass
            try:
                h.close()
            except Exception:                   # noqa: BLE001 -- teardown must not raise
                pass
        self.handle = None
        self.model = None
        self.draft_model = None
        self.mtp_head = None
        self._last_logits = None
        mx.clear_cache()

    def stats(self) -> dict:
        s = {"model": self.name, "streamed": self.streamed,
             # Whether this model can call tools at all, read off its tokenizer. The Code page
             # leads with it: an agent pointed at a model with no tool-call format does not fail
             # loudly, it answers the question instead of doing the work, and the user is left
             # guessing. Better to say so before they connect anything.
             "supports_tools": self.supports_tools(),
             "mode": self.strategy["mode"],
             "serving": self.describe_serving(),
             "source_precision": getattr(self, "source_precision", None),
             "reroute": self.reroute_tol or None,
             "weights_altered": self.strategy["mode"] == "compress",
             "decision": getattr(self, "decision", {}).get("source", "n/a"),
             # WHERE THE EXPERT COUNT CAME FROM, so the page can say "measured on this Mac"
             # rather than leave a person guessing whether a setting is a fact or a guess.
             "tuned": bool(getattr(self, "knee", None)),
             "capacity_source": ("measured on this Mac on first run"
                                 if getattr(self, "knee", None) else
                                 "an estimate; the first run at this budget will measure it"),
             "load_seconds": round(self.load_seconds, 2),
             "total_tokens": self.total_tokens,
             "flagged_tokens": self.flagged_tokens,
             "flagged_share": (self.flagged_tokens / self.total_tokens
                               if self.total_tokens else 0.0),
             "flag_runs": self.flag_runs,
             "monitor": self.meter is not None,
             "chat_template": self.has_chat_template,
             "can_toggle_thinking": self.can_toggle_thinking,
             "context_length": self.context_length,
             "max_completion_tokens": self.max_completion_tokens,
             "token_limit_reason": self.token_limit_reason,
             "budget_gb": round(self.budget_gb, 2),
             # Which read path the experts take. Packed is the page-aligned copy the GPU can
             # read in place; unpacked means every expert is copied in from the model's own
             # shards. The page says so, because the difference is the single largest speed
             # factor a user controls and nothing else on it would explain a slow server.
             "packed": bool(getattr(self, "packed", False)),
             "file_pool": bool(getattr(stream, "VIEWS_PREFILL", False)
                               or getattr(stream, "VIEWS_DECODE", False)),
             # What the Mac charges this process, wired file pages included. Beside the MLX
             # number so the two can be compared -- they differ by design in views mode.
             "phys_footprint_gb": round(calibrate.phys_footprint_gb(), 2),
             "mtp": self.mtp_name or None,
             "mtp_gb": self.mtp_gb if self.mtp_name else None,
             "mtp_bits": self.mtp_bits if self.mtp_name else None,
             **({"mtp_" + k: v for k, v in self.mtp_stats.as_dict().items()}
                if self.mtp_stats is not None else {}),
             "draft": self.draft_name or None,
             "draft_gb": round(self.draft_gb, 2) if self.draft_name else None,
             "draft_tokens": self.draft_tokens if self.draft_name else None,
             "draft_accepted": self.draft_accepted if self.draft_name else None,
             "draft_acceptance": (round(self.draft_accepted / self.total_tokens, 4)
                                  if self.draft_name and self.total_tokens else None),
             "serving_reserve_gb": self.serving_reserve_gb,
             "prompt_cache_gb": round(self.prompt_cache_gb, 2),
             "prompt_cache_bytes": (int(self._prompt_cache.nbytes)
                                    if self._prompt_cache is not None else 0),
             "prompt_cache_hits": self.prompt_cache_hits,
             "prompt_cache_misses": self.prompt_cache_misses,
             # The number that says whether the reserve was worth spending: prompt tokens that
             # did not have to be read again. Reported rather than a hit rate, because a hit on a
             # 2,000-token prefix and a hit on a 20-token one are not the same event.
             "prompt_tokens_reused": self.prompt_cache_reused,
             "prompt_cache_matched": self.prompt_cache_matched,
             "prompt_cache_protected_bytes": (int(self._prompt_cache.protected.nbytes)
                                              if self._prompt_cache is not None else 0),
             "footprint_gb": round(self.footprint_gb, 2),
             "kv_bytes_per_token": self.kv_bytes_per_token,
             "kv_bits": self.kv_bits, "kv_quant_start": self.kv_quant_start,
             "kv_fixed_bytes": getattr(self, "kv_fixed_bytes", 0)}
        if self.handle:
            h = self.handle.stats()
            s.update({"capacity": h["capacity"], "n_experts": h["n_experts"],
                      "full_layers": h.get("sync_free", 0),
                      "planned_from": getattr(self, "planned_from", None),
                      # For the pre-measurement speed verdict, which is a bytes-per-token sum.
                      "n_layers": int(self.plan.get("n_layers") or 0),
                      "top_k": int(self.plan.get("top_k") or self.top_k or 0),
                      "residency": h["capacity"] / h["n_experts"],
                      "resident_gb": round(h["resident_gb"], 2),
                      "model_gb": round(h["disk_gb"], 2),
                      "miss_rate": round(h["miss_rate"], 4),
                      "fetch_seconds": round(h["fetch_seconds"], 2),
                      "fetch_gb": round(h["fetch_gb"], 2),
                      "zero_copy_admits": int(h.get("zero_copy_admits", 0) or 0),
                      # What one more resident expert costs, so a caller can price the slider it
                      # is drawing rather than guessing at the shape of the trade.
                      "gb_per_slot": round(self.plan["bytes_per_expert"]
                                           * self.plan["n_layers"] / 1e9, 4),
                      "non_expert_gb": round(self.non_expert_gb, 2),
                      "working_memory_gb": round(self.working_memory_gb, 2)})
        return s

    def plan_summary(self) -> str:
        """The plan, in the lines the first run would otherwise have printed three times."""
        return "\n".join(self.plan_lines)

    def describe_serving(self) -> str:
        """One line, printed EVERY run, naming what is actually being served.

        The old summary said "100% of experts kept in RAM" for a compressed model. True, and
        misleading -- those experts were no longer the ones the user downloaded. Whatever else
        changes, this line must always name the precision.
        """
        m = self.strategy["mode"]
        if m == "compress":
            return (f"running at {self.strategy['bits']}-bit g{self.strategy['group_size']} "
                    f"(COMPRESSED to fit; it was {self.source_precision} -- output differs from "
                    f"the original)")
        if m == "stream":
            r = self.strategy.get("residency", 0.0)
            if r >= 0.999:
                # "100% in RAM, the rest streamed from disk" is a sentence that cannot be true.
                return (f"running EXACT at {self.source_precision}, every expert in RAM "
                        f"(unmodified weights; decode bit-identical to the original)")
            return (f"running EXACT at {self.source_precision}, {r*100:.0f}% of experts in RAM, "
                    f"the rest streamed from disk (unmodified weights; decode bit-identical "
                    f"to the original)")
        return f"running EXACT at {self.source_precision}, fully in RAM (untouched)"

