"""LFUDA -- LFU with Dynamic Aging, for the MoE expert cache.

THE IDEA
    Plain LFU pins whatever was hot early. An expert that collected 400 hits during the prompt keeps
    a count no newly-hot expert can catch, so it sits in RAM long after the router stopped asking
    for it. That is classic LFU cache pollution, and on a routing trace it is real: expert
    popularity inside a layer drifts across the sequence, so "hot once" is not "hot now".

    LFUDA (Arlitt et al., the GreedyDual-Size-Frequency family) fixes this with one global scalar.
    Every expert carries a key

        K(e) = L + count(e)

    where L is the cache's running AGE: whenever an expert is evicted, L is raised to that victim's
    key, so L is monotone non-decreasing and equals the highest key ever thrown away. The key is
    re-anchored to the current L on every use, so an expert that stops being used does not decay in
    absolute terms -- L climbs past it instead. Numerically that is exponential decay of stale
    counts, but with no sweep over the table: O(1) per request.

WHAT ACTUALLY HAPPENED WHEN IT WAS MEASURED
    Textbook LFUDA -- which drops an expert's count when it is evicted -- LOSES TO LRU on both
    traces, badly (down to -199% of the gap on Ling at C=128). Two things break it here:

      1. Dropping the count on eviction destroys the only signal the policy has. Expert working
         sets churn, so a genuinely hot expert gets evicted once and comes back as a stranger.
      2. An uncapped count in a units-of-L key means one very hot expert accumulates a key that
         eviction-driven aging cannot catch up to within a sequence, because L only advances when
         something is evicted. At high capacity there are few evictions, so L barely moves and the
         policy degenerates to unaged LFU -- exactly the pollution it was meant to fix.

    Both are fixable without leaving the LFUDA frame, and the shipped defaults do:

      keep_ghost=True    counts survive eviction (bounded: <= n_experts counters, 64 or 256 here),
                         so frequency is not amnesiac across a single unlucky eviction.
      ghost_decay=0.5    but being evicted is itself evidence, so the surviving count is halved.
                         This is the second aging channel and it is the one that does the work:
                         with ghost counts kept but NOT decayed, the policy captures only 10.6% of
                         the LRU-to-Belady gap and is negative on Ling at C=128.
      fcap=12            counts saturate. Above the cap all experts tie on frequency and the
                         tie-break is recency, so the policy degrades gracefully into LRU in the
                         regime where LRU is already good (high capacity) instead of degenerating
                         into unaged LFU there.

    Uncapped textbook LFUDA is kept available as `TextbookLFUDA` because the negative result is the
    point: the aging term alone is not what makes LFU work on this workload.

CAUSALITY
    Uses only the current request, the router rank of the current request, this policy's own
    per-expert counters, and the global age L. No future request is consulted, no expert weights.

COST
    victim() is a linear scan of the RESIDENT set only -- O(capacity) per eviction, never
    O(n_experts). All other operations are O(1). State is two floats and an int per expert, so
    memory is O(n_experts) = 64 (OLMoE) or 256 (Ling) entries, independent of trace length.

MEASURED (first 4 layers, per the arm protocol; % of the LRU-to-Belady gap captured)
    OLMoE  C=8/16/24/32   : +16.4  +16.2  +26.5  +37.7
    Ling   C=16/32/64/128 : +20.3  +31.8  +33.0   +7.8      mean +23.7%
    Beats LRU at all 8 points, and at all 8 points again on held-out layers 4-7 (fcap and
    ghost_decay were chosen on layers 0-3, so layers 4-7 are the honest generalization check).
"""
from __future__ import annotations

from bigrig_engine.pool import Policy


class LFUDA(Policy):
    """LFU with dynamic aging. Defaults are the shipped configuration.

    Parameters
    ----------
    rank_weight
        Weight on the router's rank when crediting a use. 0.0 (default) means every request is
        worth exactly 1, as in textbook LFUDA. With w > 0 a request at rank r is worth
        1 + w / (1 + r), so a top-1 gate counts more than a rank-7 tail gate. Measured: worth
        roughly +2 points of gap at the smallest capacity and -2 at the largest, i.e. a wash, so
        it is off by default.
    keep_ghost
        Keep an expert's count after it is evicted (default True). Textbook LFUDA sets this False
        and is worse than LRU on both traces.
    ghost_decay
        Multiplier applied to a kept count at eviction time (default 0.5). Being evicted is
        evidence against the expert; this is the aging channel that carries the result.
    fcap
        Saturating cap on the count (default 12). Above it, experts tie on frequency and the
        recency tie-break decides, so the policy falls back to LRU rather than to unaged LFU in
        the high-capacity regime.
    """

    name = "lfuda"

    def __init__(self, capacity, n_experts, rank_weight: float = 0.0, keep_ghost: bool = True,
                 fcap: float | None = 12.0, ghost_decay: float = 0.5):
        super().__init__(capacity, n_experts)
        self.rank_weight = float(rank_weight)
        self.keep_ghost = bool(keep_ghost)
        self.fcap = fcap
        self.ghost_decay = float(ghost_decay)
        self.L = 0.0                 # global age: the highest key ever evicted
        self.cnt: dict = {}          # expert -> saturating frequency credit
        self.key: dict = {}          # expert -> L_at_last_use + cnt   (resident experts)
        self.last: dict = {}         # expert -> tick of last use, tie-breaks only
        self.resident: set = set()
        self.tick = 0

    # -- internal ---------------------------------------------------------
    def _credit(self, rank: int) -> float:
        if self.rank_weight == 0.0:
            return 1.0
        return 1.0 + self.rank_weight / (1.0 + rank)

    def _bump(self, e: int, rank: int) -> None:
        self.tick += 1
        self.last[e] = self.tick
        c = self.cnt.get(e, 0.0) + self._credit(rank)
        if self.fcap is not None and c > self.fcap:
            c = self.fcap
        self.cnt[e] = c
        # The aging step: re-anchor the key to the CURRENT age on every use. An expert that stops
        # being used keeps its key while L climbs underneath it, which is what retires it.
        self.key[e] = self.L + c

    # -- Policy interface -------------------------------------------------
    def admit(self, e: int, rank: int = 0) -> None:
        self.resident.add(e)
        if not self.keep_ghost:
            self.cnt.pop(e, None)
        self._bump(e, rank)

    def touch(self, e: int, rank: int = 0) -> None:
        self._bump(e, rank)

    def victim(self, exclude=()) -> int:
        """Smallest key wins; ties break least-recently-used.

        `exclude` names experts the CALLER cannot evict right now -- typically the ones the
        current token is still using. Honouring it here is what lets the pool avoid the
        alternative, which was to nominate a protected expert, call `evicted()` on it to move
        past it, and thereby tell this policy an eviction happened that did not. Measured
        consequence of that alternative: after 200 steps the policy could see 1 of 12 resident
        experts, the other 11 were unreachable by victim() forever, and L had been inflated by
        996 phantom evictions.
        """
        pool = self.resident if not exclude else (self.resident - set(exclude))
        if not pool:
            return -1
        return min(pool, key=lambda x: (self.key.get(x, 0.0), self.last.get(x, 0)))

    def evicted(self, e: int) -> None:
        # Raise the global age to the victim's key. max() keeps L monotone, which matters because
        # a recency tie-break can evict an expert whose key sits below an already-established L.
        k = self.key.pop(e, self.L)
        if k > self.L:
            self.L = k
        self.resident.discard(e)
        if not self.keep_ghost:
            self.cnt.pop(e, None)
            self.last.pop(e, None)
        elif self.ghost_decay != 1.0:
            self.cnt[e] = self.cnt.get(e, 0.0) * self.ghost_decay


class TextbookLFUDA(LFUDA):
    """Aging only: uncapped count, dropped on eviction. Measured WORSE than LRU on both traces."""
    name = "lfuda_textbook"

    def __init__(self, capacity, n_experts):
        super().__init__(capacity, n_experts, keep_ghost=False, fcap=None, ghost_decay=1.0)


def factory(**kw):
    """policy_factory(capacity, n_experts) with LFUDA hyperparameters bound."""
    return lambda C, E: LFUDA(C, E, **kw)
