"""stream.reuse_detokenizer_tables: same text as mlx_lm's own detokenizer, table built once.

WHY THIS FILE EXISTS
    mlx_lm builds a streaming detokenizer per stream_generate call, and the BPE and SentencePiece
    ones rebuild their id -> text table from the whole vocabulary each time (228-236 ms a reply on
    Qwen3.6's 248,077 tokens, before the first token). reuse_detokenizer_tables makes later
    detokenizers share the first one's table. That is only acceptable if:
      - every streamed segment, and the final text, equals what the upstream class produces;
      - detokenizers made from one tokenizer do not share per-reply state (two replies in
        flight must not corrupt each other);
      - the vocabulary really is read once, and the switch and the fallbacks behave.
    Sections 1-3 use small stub tokenizers (no model). Section 4 uses the real tokenizers of the
    models on disk and is reported, loudly, as skipped when they are absent.

Run: .venv/bin/python tests/test_detokenizer_tables.py
"""
import functools
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from mlx_lm.tokenizer_utils import (BPEStreamingDetokenizer, NaiveStreamingDetokenizer,  # noqa: E402
                                    SPMStreamingDetokenizer)

from bigrig_engine import stream  # noqa: E402

FAIL = []
SKIPPED = []


def check(n, c, d=""):
    (print(f"  PASS  {n}") if c else (FAIL.append(n), print(f"  FAIL  {n}  {d}")))


class StubTokenizer:
    """What a detokenizer reads from a tokenizer: `vocab` (counted), and the clean-up flag."""

    def __init__(self, pieces, factory):
        self._vocab = {p: i for i, p in enumerate(pieces)}
        self.vocab_reads = 0
        self.clean_up_tokenization_spaces = False
        self._detokenizer_class = factory

    @property
    def vocab(self):
        self.vocab_reads += 1
        return dict(self._vocab)

    @property
    def detokenizer(self):
        return self._detokenizer_class(self)


# Byte-level BPE pieces: "Ġ" is a leading space, "Ċ" a newline, and the last three are the bytes
# of one multi-byte character split across tokens (the case the streaming code must hold back).
BPE_PIECES = ["Hello", "Ġworld", ",", "Ġthe", "Ġheat", "Ġpump", "Ċ", "def", "Ġfib", "(", "n", ")", ":",
              "Ġ", "ĠĠ", "n't", "Ġdo", ".", "!", "?", "ðŁ", "ĺ", "Ģ", "ĠcafÃ©", "<|im_end|>", "'s", "Ġit"]
SPM_PIECES = ["▁Hello", "▁world", ",", "▁the", "▁heat", "<0xF0>", "<0x9F>", "<0x98>", "<0x80>",
              "▁", "pump", ".", "▁▁", "n", "'t", "<s>", "</s>", "▁café"]
rng = np.random.default_rng(3)


def streams(detok, ids):
    """Every streamed segment and the final text for one reply, as stream_generate drives it."""
    detok.reset()
    segs = []
    for t in ids:
        detok.add_token(int(t))
        segs.append(detok.last_segment)
    detok.finalize()
    segs.append(detok.last_segment)
    return segs, detok.text


CASES = {"BPE": (BPE_PIECES, BPEStreamingDetokenizer),
         "SPM": (SPM_PIECES, SPMStreamingDetokenizer),
         "SPM (trim_space=False)": (SPM_PIECES, functools.partial(SPMStreamingDetokenizer, trim_space=False))}

print("=" * 78)
print("1. SAME SEGMENTS AND TEXT AS THE UPSTREAM CLASS; THE VOCABULARY IS READ ONCE")
print("=" * 78)
for name, (pieces, factory) in CASES.items():
    plain = StubTokenizer(pieces, factory)
    shared = stream.reuse_detokenizer_tables(StubTokenizer(pieces, factory))
    check(f"{name}: the detokenizer class was replaced", shared._detokenizer_class is not factory)
    seqs = [rng.integers(0, len(pieces), size=n) for n in (1, 2, 7, 40, 200)] + \
           [np.array([i for i, p in enumerate(pieces) if p in ("ðŁ", "ĺ", "Ģ", "<0xF0>", "<0x9F>", "<0x98>", "<0x80>")])]
    first = shared.detokenizer                                # builds the table
    reads_after_first = shared.vocab_reads
    ok, texts = True, []
    for ids in seqs:
        want = streams(plain.detokenizer, ids)
        got = streams(shared.detokenizer, ids)
        ok = ok and want == got
        texts.append(got[1])
    check(f"{name}: {len(seqs)} replies, every segment and the final text identical", ok)
    check(f"{name}: the replies are real text (non-empty; a character split over tokens decodes whole)",
          all(texts[:-1]) and "\U0001f600" in texts[-1], repr(texts[-1][:40]))
    check(f"{name}: the vocabulary was read for the first detokenizer only "
          f"({reads_after_first} reads, none after)", reads_after_first > 0 and shared.vocab_reads == reads_after_first,
          (reads_after_first, shared.vocab_reads))
    check(f"{name}: the upstream class reads it for every detokenizer (the cost is real)",
          plain.vocab_reads >= len(seqs))
    d1, d2 = shared.detokenizer, shared.detokenizer
    check(f"{name}: two detokenizers share the table object and nothing reset() owns",
          d1.tokenmap is d2.tokenmap and d1.tokens is not d2.tokens)
    check(f"{name}: an instance is still an upstream detokenizer", isinstance(d1, factory.func if isinstance(
        factory, functools.partial) else factory))

print("=" * 78)
print("2. TWO REPLIES IN FLIGHT DO NOT TOUCH EACH OTHER")
print("=" * 78)
for name, (pieces, factory) in CASES.items():
    plain = StubTokenizer(pieces, factory)
    shared = stream.reuse_detokenizer_tables(StubTokenizer(pieces, factory))
    a_ids, b_ids = rng.integers(0, len(pieces), size=60), rng.integers(0, len(pieces), size=60)
    want_a, want_b = streams(plain.detokenizer, a_ids), streams(plain.detokenizer, b_ids)
    da, db = shared.detokenizer, shared.detokenizer
    sa, sb = [], []
    for ta, tb in zip(a_ids, b_ids):                          # interleaved token by token
        da.add_token(int(ta)); sa.append(da.last_segment)
        db.add_token(int(tb)); sb.append(db.last_segment)
    da.finalize(); sa.append(da.last_segment)
    db.finalize(); sb.append(db.last_segment)
    check(f"{name}: interleaved replies each equal their own upstream stream",
          (sa, da.text) == want_a and (sb, db.text) == want_b)

print("=" * 78)
print("3. THE SWITCH, AND WHAT IT LEAVES ALONE")
print("=" * 78)
naive = StubTokenizer(BPE_PIECES, NaiveStreamingDetokenizer)
check("a tokenizer on the Naive detokenizer (no table) is left as it was",
      stream.reuse_detokenizer_tables(naive)._detokenizer_class is NaiveStreamingDetokenizer)


class _Sub(BPEStreamingDetokenizer):
    pass


sub = StubTokenizer(BPE_PIECES, _Sub)
check("an unknown detokenizer subclass is left as it was (exact upstream classes only)",
      stream.reuse_detokenizer_tables(sub)._detokenizer_class is _Sub)
check("an object with no detokenizer class is returned untouched, without raising",
      stream.reuse_detokenizer_tables(object) is object)
os.environ["BIGRIG_SHARE_DETOK"] = "0"
try:
    off = StubTokenizer(BPE_PIECES, BPEStreamingDetokenizer)
    check("BIGRIG_SHARE_DETOK=0 leaves the upstream class in place",
          stream.reuse_detokenizer_tables(off)._detokenizer_class is BPEStreamingDetokenizer)
finally:
    del os.environ["BIGRIG_SHARE_DETOK"]
twice = stream.reuse_detokenizer_tables(StubTokenizer(BPE_PIECES, BPEStreamingDetokenizer))
cls1 = twice._detokenizer_class
check("applying it twice changes nothing the second time",
      stream.reuse_detokenizer_tables(twice)._detokenizer_class is cls1)
other = StubTokenizer(BPE_PIECES[:10], BPEStreamingDetokenizer)
shared = stream.reuse_detokenizer_tables(StubTokenizer(BPE_PIECES, BPEStreamingDetokenizer))
shared.detokenizer
d_other = shared._detokenizer_class(other)
check("the shared class handed a DIFFERENT tokenizer builds that tokenizer's own table",
      len(d_other.tokenmap) == 10 and len(shared.detokenizer.tokenmap) == len(BPE_PIECES))

print("=" * 78)
print("4. THE REAL TOKENIZERS ON DISK: SAME TEXT, ONE VOCABULARY READ")
print("=" * 78)
from mlx_lm.utils import load_tokenizer  # noqa: E402
from pathlib import Path  # noqa: E402

TEXTS = ["To explain how a heat pump moves warmth from cold outdoor air into a house, start with the refrigerant.",
         "def fib(n: int) -> int:\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n    a, b = 0, 1\n    for _ in range(n):\n"
         "        a, b = b, a + b\n    return a\n",
         "Café naïve 日本語 \U0001f600 don't — 9:40 → 13:15 (3 h 35 min)."]
for model in ("Qwen3.6-35B-A3B-4bit", "OLMoE-1B-7B-0125-4bit"):
    path = Path(ROOT) / "models" / model
    if not (path / "tokenizer.json").exists():
        SKIPPED.append(model)
        print(f"  SKIPPED - {model}: no tokenizer at {path}")
        continue
    plain = load_tokenizer(path)
    shared = stream.reuse_detokenizer_tables(load_tokenizer(path))
    kind = getattr(plain._detokenizer_class, "__name__", str(plain._detokenizer_class))
    calls = {"n": 0}
    inner = shared._tokenizer
    orig = type(inner).get_vocab

    def counted(self, *a, _o=orig, **k):
        calls["n"] += 1
        return _o(self, *a, **k)

    type(inner).get_vocab = counted
    try:
        shared.detokenizer
        after_first = calls["n"]
        ok = True
        for text in TEXTS:
            ids = plain.encode(text)
            want, got = streams(plain.detokenizer, ids), streams(shared.detokenizer, ids)
            ok = ok and want == got and bool(got[1].strip())
        only_shared = calls["n"]
    finally:
        type(inner).get_vocab = orig
    if kind == "NaiveStreamingDetokenizer":
        check(f"{model}: Naive detokenizer, left as it was", shared._detokenizer_class is plain._detokenizer_class)
        continue
    check(f"{model} ({kind}): {len(TEXTS)} texts, every segment and the final text identical to upstream", ok)
    # `plain` shares the tokenizer class, so its detokenizers' reads are counted too: 3 texts x
    # the upstream reads per detokenizer. The shared one must add none after its first.
    per = after_first
    check(f"{model}: after the first detokenizer ({after_first} vocabulary reads) the shared one reads it "
          f"no more (total {only_shared} = first + upstream's {len(TEXTS)} x {per})",
          after_first > 0 and only_shared == after_first + len(TEXTS) * per, (after_first, only_shared))

print("\n" + "=" * 78)
if SKIPPED:
    print(f"SKIPPED: real-tokenizer checks for {', '.join(SKIPPED)} (models not on disk)")
print(f"{'ALL TESTS PASSED' if not FAIL else str(len(FAIL)) + ' FAILURES: ' + ', '.join(FAIL)}"
      + (" (with skips above)" if SKIPPED and not FAIL else ""))
print("=" * 78)
sys.exit(1 if FAIL else 0)
