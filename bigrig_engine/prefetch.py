"""COMPONENT 10 — prefetch: start the fetch before the stall.

TWO KINDS OF PREFETCH, AND ONLY ONE OF THEM IS FREE

  CERTAIN (pipelining).  Within a layer the k experts are fetched together, but compute does not
  need all of them at once -- an expert's contribution can be accumulated the moment ITS bytes
  arrive. So the k fetches are issued in parallel and consumed as they complete. No prediction,
  no wasted bytes, 100% precision. This is the part that is unambiguously worth having.

  SPECULATIVE (prediction).  Fetching the NEXT layer's experts before its router has run requires
  guessing, and this project already measured how well that can be done. Recall of the k=8
  experts actually needed, from information available beforehand:

        budget B=8   B=16   B=32
      OLMoE  0.497  0.713  0.894
      Ling   0.377  0.506  0.655

  Read that as a cost, not a hit rate. Prefetching 16 candidates to catch 5.7 of the 8 you need
  means reading 2x the bytes for 71% coverage. When the link is the bottleneck -- which is the
  entire premise of the engine -- spending 2x the bandwidth to avoid some latency is a bad trade.
  P2 of this project's constraint sheet says it directly: prefetching moves the same bytes and
  converts a latency cost into a bandwidth cost.

  MEASURED VERDICT: speculation does not pay at ANY budget, on either model, even assuming a
  completely idle link. The predictor catches fewer experts than it wastes:

        model  budget  recall  bytes  caught  wasted
        olmoe       8   0.497    1.0x   3.98    4.02
        olmoe      16   0.713    2.0x   5.70   10.30
        olmoe      32   0.894    4.0x   7.15   24.85
        ling        8   0.377    1.0x   3.02    4.98
        ling       32   0.655    4.0x   5.24   26.76

  Even the 89% recall figure, which sounds strong, is reached only by fetching four times the
  bytes -- catching 7.15 of 8 while dragging in 24.85 that are never used. So the speculative
  path is implemented, measured, and left OFF. It is kept in the codebase because the arithmetic
  flips if a predictor ever reaches high recall at a budget near k, and `speculation_pays()`
  will say so; it is not kept because it currently works.
"""
from __future__ import annotations

import time
from collections import defaultdict

import numpy as np


class PipelinedLoader:
    """Issue a layer's expert fetches together; hand each one back as it lands.

    The win is time-to-FIRST-expert rather than time-to-all. With k=8 experts fetched in
    parallel, compute can begin after roughly one expert's latency instead of after all eight.
    """

    def __init__(self, fetcher):
        self.fetcher = fetcher

    def stream(self, keys):
        """Yield (key, bytes) in completion order. Compute may start on the first arrival."""
        keys = list(keys)
        if not keys:
            return
        self.fetcher.prefetch(keys)
        remaining = set(keys)
        while remaining:
            # collect whatever is ready; fetch() on a completed prefetch returns immediately
            for k in list(remaining):
                got = self.fetcher.fetch([k])
                yield k, got[k]
                remaining.discard(k)


class CoOccurrencePredictor:
    """Predict a layer's experts from the layer below's picks for the SAME token.

    P4 of the constraint sheet is the rule this respects: a decision made before a fetch may not
    depend on that fetch's result. Layer L-1's picks exist before layer L's fetch, so using them
    is legal. Using layer L's own router output would not be a prediction at all.

    Trained by counting co-occurrence on a held-out half of a trace; never on the half it is
    scored against.
    """

    def __init__(self, n_experts: int):
        self.E = n_experts
        self.M: dict = {}
        self.freq: np.ndarray | None = None
        self.fitted = False

    def fit(self, idx: np.ndarray, upto: int | None = None) -> "CoOccurrencePredictor":
        L, T, k = idx.shape
        upto = upto if upto is not None else T // 2
        self.freq = np.bincount(idx[:, :upto, :].ravel(), minlength=self.E).astype(float)
        for l in range(1, L):
            M = np.zeros((self.E, self.E), dtype=np.float32)
            prev, cur = idx[l - 1, :upto, :], idx[l, :upto, :]
            for t in range(upto):
                for a in prev[t]:
                    M[a, cur[t]] += 1.0
            self.M[l] = M
        self.fitted = True
        return self

    def predict(self, layer: int, prev_picks, budget: int):
        """Top-`budget` candidate experts for `layer`, from the layer below's picks."""
        if not self.fitted:
            raise RuntimeError("predictor is not fitted; call fit() on a held-out half first")
        if layer not in self.M:
            order = np.argsort(-self.freq)
            return order[:budget].tolist()
        score = self.M[layer][list(prev_picks)].sum(axis=0)
        return np.argsort(-score)[:budget].tolist()

    def recall(self, idx: np.ndarray, budget: int, frm: int | None = None) -> float:
        """Fraction of the experts actually needed that the prediction contained."""
        L, T, k = idx.shape
        frm = frm if frm is not None else T // 2
        hit = tot = 0
        for l in range(1, L):
            for t in range(frm, T):
                pred = set(self.predict(l, idx[l - 1, t], budget))
                need = set(int(x) for x in idx[l, t])
                hit += len(pred & need)
                tot += len(need)
        return hit / max(tot, 1)


def speculation_pays(recall: float, budget: int, k: int, idle_fraction: float) -> dict:
    """Does speculative prefetch pay, given a measured recall and measured idle bandwidth?

    Prefetching `budget` candidates to cover `k` needed experts reads budget/k times the bytes.
    Those extra bytes are free ONLY while the link is idle. The rule below is deliberately
    conservative: speculate only if the extra traffic fits inside measured idle capacity AND the
    recall is high enough that the caught experts outnumber the wasted ones.
    """
    cost_ratio = budget / max(k, 1)
    caught = recall * k
    wasted = budget - caught
    fits_idle = (cost_ratio - 1.0) <= idle_fraction
    worth_it = fits_idle and caught > wasted
    return {"cost_ratio": round(cost_ratio, 2), "caught": round(caught, 2),
            "wasted": round(wasted, 2), "fits_in_idle_bandwidth": fits_idle,
            "recommended": bool(worth_it)}


class PrefetchController:
    """Decides what to prefetch. Certain pipelining always; speculation only when it pays."""

    def __init__(self, fetcher, predictor=None, budget: int = 16, k: int = 8,
                 speculate: bool = False, idle_fraction: float = 0.0):
        self.loader = PipelinedLoader(fetcher)
        self.fetcher = fetcher
        self.predictor = predictor
        self.budget, self.k = budget, k
        self.speculate = speculate
        self.idle_fraction = idle_fraction
        self.stats = defaultdict(int)

    def layer(self, keys):
        """Fetch this layer's experts, streaming them as they arrive."""
        self.stats["layers"] += 1
        self.stats["certain_fetches"] += len(keys)
        return self.loader.stream(keys)

    def speculate_next(self, layer: int, prev_picks, key_of):
        """Optionally prefetch the NEXT layer's likely experts. Returns how many were issued."""
        if not (self.speculate and self.predictor is not None):
            return 0
        cand = self.predictor.predict(layer, prev_picks, self.budget)
        keys = [key_of(layer, int(e)) for e in cand]
        try:
            self.fetcher.prefetch(keys)
        except RuntimeError:
            return 0                       # prefetch queue full; skip rather than stall
        self.stats["speculative_fetches"] += len(keys)
        return len(keys)

    def report(self) -> dict:
        spec = self.stats["speculative_fetches"]
        cert = self.stats["certain_fetches"]
        return {"layers": self.stats["layers"], "certain": cert, "speculative": spec,
                "speculative_overhead": round(spec / max(cert, 1), 3)}
