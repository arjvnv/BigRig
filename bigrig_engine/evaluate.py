"""Perplexity, measured the same way every time, so two configurations can be compared.

WHY THIS EXISTS AS ITS OWN FILE
    Every quality claim this product makes is a comparison: "configuration X costs Y against the
    reference." A comparison is only as good as the guarantee that both sides were measured
    identically -- same text, same tokenisation, same window, same stride, same reduction. Every
    one of those is a place where two numbers can be produced that are individually correct and
    jointly meaningless.

    So the corpus is tokenised ONCE and the token ids are reused across configurations. Nothing
    downstream may re-tokenise, because two tokenisers that differ by a single space produce
    perplexities that differ by more than the effect being measured.
"""
from __future__ import annotations

import math
import os

import mlx.core as mx
import numpy as np

CHUNK = 256      # positions per float32 reduction slice
from . import home

CORPORA = os.path.join(home(), "data", "corpora")


def load_text(name: str, max_chars: int = 400_000) -> str:
    p = os.path.join(CORPORA, name)
    with open(p, encoding="utf-8", errors="ignore") as f:
        t = f.read()
    return t[:max_chars]


def tokenize_once(tokenizer, texts: dict, max_chars: int = 400_000) -> dict:
    """{domain: token ids}. Called ONCE per model; every configuration reuses the result."""
    out = {}
    for name, fname in texts.items():
        ids = tokenizer.encode(load_text(fname, max_chars))
        out[name] = np.asarray(ids, dtype=np.int64)
    return out


def perplexity(model, ids: np.ndarray, window: int = 1024, stride: int = 512,
               max_windows: int = 0, guard=None) -> dict:
    """Sliding-window perplexity with teacher forcing.

    `stride < window` means each window re-reads `window - stride` tokens of context that were
    already scored, and only the NEW tokens are counted. That is the standard way to avoid
    scoring the first tokens of a window with almost no context, which otherwise inflates
    perplexity for reasons that have nothing to do with the model being measured.
    """
    if window < 2:
        raise ValueError(f"window must be at least 2, got {window}")
    if not 1 <= stride <= window:
        raise ValueError(f"stride must be in [1, {window}], got {stride}")
    n = len(ids)
    if n < window + 1:
        raise ValueError(f"corpus has {n} tokens, need more than the {window}-token window")

    total_nll, total_tok = 0.0, 0
    starts = list(range(0, n - window - 1, stride))
    if max_windows:
        starts = starts[:max_windows]
    prev_end = 0
    for wi, s in enumerate(starts):
        chunk = ids[s:s + window + 1]
        x = mx.array(chunk[:-1])[None]
        y = mx.array(chunk[1:])
        logits = model(x)
        # Score only positions not already scored by an earlier window.
        first_new = max(0, prev_end - s)
        if first_new >= window:
            del logits
            mx.clear_cache()
            continue
        # Reduce in slices. Casting a whole (window, vocab) tensor to float32 is 622 MB at
        # window=1024 on a 152k vocabulary -- larger than the headroom this product is built to
        # operate in, and it would OOM exactly on the models the measurement is for.
        vals = []
        for a in range(first_new, window, CHUNK):
            b = min(a + CHUNK, window)
            lg = logits[0, a:b].astype(mx.float32)
            lse = mx.logsumexp(lg, axis=-1)
            tgt = mx.take_along_axis(lg, y[a:b, None], axis=-1)[:, 0]
            nll = lse - tgt                                # -log p(target)
            mx.eval(nll)
            v = np.asarray(nll)
            if not np.all(np.isfinite(v)):
                raise ValueError(
                    f"window {wi} produced a non-finite log-likelihood. A quantisation that "
                    f"overflows silently would otherwise be reported as excellent perplexity.")
            vals.append(v)
            del lg, lse, tgt, nll
        v = np.concatenate(vals)
        total_nll += float(v.sum())
        total_tok += v.size
        del logits, vals
        prev_end = s + window
        if guard is not None and wi % 4 == 0:
            guard(f"ppl w{wi}")
        mx.clear_cache()
    if total_tok == 0:
        raise ValueError("no tokens were scored -- check window and stride")
    mean_nll = total_nll / total_tok
    return {"nll": mean_nll, "ppl": math.exp(mean_nll), "tokens": total_tok,
            "windows": len(starts)}


def compare(ref: dict, cand: dict) -> dict:
    """The two numbers a user should be shown: how much worse, and by how much per token."""
    d_nll = cand["nll"] - ref["nll"]
    return {"d_nll": d_nll, "ppl_ratio": cand["ppl"] / ref["ppl"],
            "ppl_pct": (cand["ppl"] / ref["ppl"] - 1.0) * 100.0,
            "ref_ppl": ref["ppl"], "cand_ppl": cand["ppl"]}
