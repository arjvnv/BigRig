"""gaterank -- gate-rank weighted recency for the MoE expert cache.

IDEA
    LRU orders residents by the wall-clock of their last request and nothing else. But the
    router hands us a second free signal on every single request: the expert's position in the
    top-k for this token. Rank 0 is the expert the gate wanted most. Component 9's own notes say
    that rank correlates with recency, so a rank-0 hit is evidence of a stronger, longer-lived
    affinity than a rank-7 hit, and the two should not buy the same amount of protection.

    So instead of ordering by `last_touch_tick`, gaterank orders by

        key(e) = last_touch_tick(e) + B * w(rank of that touch)

    where w(0) = 1 and w decays with rank. B is the boost budget measured in REQUESTS: a rank-0
    access is treated as if it happened B requests later than it really did, i.e. it takes B
    further requests of pressure before that expert looks as stale as an unboosted one. B = 0 is
    exactly LRU, which makes the policy a strict generalisation of the baseline and makes the
    "did the rank signal buy anything" question a one-parameter question.

MONOTONE KEYS
    A naive `key = tick + B*w(r)` can DEMOTE an expert: a rank-0 touch at tick 100 with B=256
    gets key 356, and its own next touch at tick 110 at rank 7 gets key 142. Being used again
    would make it a better eviction candidate, which is nonsense. gaterank therefore takes
    key = max(old_key, tick + B*w(r)), so an access can never lower an expert's standing. The
    flag is exposed (`monotone`) because it is a real design choice, and both settings were
    measured rather than assumed.

CAUSALITY
    Uses only: the id and router rank of the request being served, a monotone counter of requests
    this policy has seen, and one float per resident expert. No future requests, no weights, no
    trace-level statistics. Rank comes from the router's own top-k, which is computed before the
    fetch decision exists.

COST
    O(log capacity) amortised per admit/touch/victim. Keys live in a lazy min-heap; stale entries
    are skipped on pop, and the heap is rebuilt from the resident set whenever it grows past
    ~4x capacity, so both memory and per-op work stay bounded in capacity, never in n_experts.
    Nothing here scans all E experts.

MEASURED (layers 0-3 only, pool.evaluate, B=16 shape=linear(k=8) monotone=True)
    model  cap   gaterank    LRU    Belady   gap captured
    olmoe    8     0.7289  0.7883   0.5004    20.62%
    olmoe   16     0.5357  0.5401   0.3114     1.90%
    olmoe   24     0.4064  0.4095   0.2000     1.52%
    olmoe   32     0.2989  0.3014   0.1225     1.42%
    ling    16     0.6898  0.6902   0.4478     0.17%
    ling    32     0.5053  0.5202   0.2677     5.87%
    ling    64     0.2539  0.2573   0.1051     2.20%
    ling   128     0.0610  0.0612   0.0263     0.34%
    mean 4.25%; better than LRU in all 8 cells, but only barely in most of them.

WHAT THE SWEEP ACTUALLY SHOWED -- read this before believing the idea
    B < 7 is bit-for-bit identical to LRU. That is not a coincidence: requests arrive in rank
    order within a token, so consecutive requests are 1 tick apart and one rank step apart, and
    the key stays monotone in tick until B exceeds the intra-token spacing. So every gain above
    comes from B >= k reversing the ORDER OF A SINGLE TOKEN'S OWN k experts -- making the token's
    rank-0 pick the most-protected of the eight instead of the least. Everything from B=8 to
    B=16 is the same policy by that argument and measures the same (4.25%). Past B~32 rank-0
    experts from older tokens start outranking fresh rank-7 ones and it turns negative fast:
    B=512 linear = -6.9%, B=2048 = -23.0%.

    Consistent with that, the win concentrates exactly where a single token's k experts fight
    each other for room: olmoe at C=8 = k gets 20.6%, and the other seven cells average 1.9%.

    Direction control: flipping the sign of the boost (protect rank-7 hardest) gives -1.6% to
    -2.0% mean gap, i.e. worse than LRU. The rank signal is real and points the way expected;
    it is just worth very little once the cache is comfortably larger than k.
"""
from __future__ import annotations

import heapq

from bigrig_engine.pool import Policy


def _w_inv(r: int) -> float:
    """1, 1/2, 1/3, ... -- needs no knowledge of k."""
    return 1.0 / (1.0 + r)


def _w_geom(gamma: float):
    def f(r: int) -> float:
        return gamma ** r
    return f


def _w_linear(k: int):
    def f(r: int) -> float:
        if k <= 1:
            return 1.0
        return max(0.0, (k - 1 - r) / (k - 1))
    return f


SHAPES = {
    "inv": lambda p: _w_inv,
    "geom": lambda p: _w_geom(p),
    "linear": lambda p: _w_linear(int(p)),
}


class GateRank(Policy):
    """Recency, with each access credited a bonus that decays with its router rank."""

    name = "gaterank"

    def __init__(self, capacity, n_experts, boost: float = 16.0,
                 shape: str = "linear", param: float = 8, monotone: bool = True):
        super().__init__(capacity, n_experts)
        self.boost = float(boost)
        self.monotone = bool(monotone)
        self.w = SHAPES[shape](param)
        self.tick = 0
        self.key: dict = {}          # resident expert -> priority key
        self.heap: list = []         # (key, expert), lazily invalidated
        self._limit = 4 * max(1, capacity) + 16

    # ---- internals -------------------------------------------------------
    def _bump(self, e: int, rank: int) -> None:
        self.tick += 1
        k = self.tick + self.boost * self.w(rank)
        if self.monotone:
            old = self.key.get(e)
            if old is not None and old > k:
                k = old
        self.key[e] = k
        heapq.heappush(self.heap, (k, e))
        if len(self.heap) > self._limit:
            self._rebuild()

    def _rebuild(self) -> None:
        self.heap = [(v, e) for e, v in self.key.items()]
        heapq.heapify(self.heap)

    # ---- Policy interface ------------------------------------------------
    def admit(self, e, rank=0):
        self._bump(e, rank)

    def touch(self, e, rank=0):
        self._bump(e, rank)

    def victim(self):
        while self.heap:
            k, e = self.heap[0]
            cur = self.key.get(e)
            if cur is None or cur != k:   # evicted already, or superseded by a newer key
                heapq.heappop(self.heap)
                continue
            return e
        return -1

    def evicted(self, e):
        self.key.pop(e, None)


def make(boost: float = 16.0, shape: str = "linear", param: float = 8, monotone: bool = True):
    """Factory usable directly as `policy_factory` in pool.simulate/evaluate."""
    def factory(capacity, n_experts):
        return GateRank(capacity, n_experts, boost=boost, shape=shape,
                        param=param, monotone=monotone)
    return factory


# The configuration reported in the write-up, chosen by the sweep in this file's notes.
gaterank = make()
