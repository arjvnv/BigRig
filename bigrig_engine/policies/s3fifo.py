"""S3-FIFO for the MoE expert pool.

Three structures, all FIFO, no recency reordering on hit:

    S  -- small probationary FIFO (default 10% of capacity). Everything enters here unless it
          is a ghost hit. Its job is to flush one-hit-wonders fast.
    M  -- main FIFO (the remaining ~90%). Holds objects that proved themselves in S, plus
          anything that came back from the ghost queue.
    G  -- ghost FIFO of ids only (no payload). Records what S threw away, so an expert that
          returns after a short absence is admitted straight into M instead of paying probation
          again.

Each resident expert carries a small saturating counter `freq` (cap 3). A hit only bumps the
counter -- position in the queue never changes, which is the whole point: no per-hit list
surgery, unlike LRU.

Eviction (the cache is full, we must free exactly one slot):
    if |S| >= s_target:  take S's oldest.
        freq > 1  -> it earned a place: move it to M, reset freq, KEEP LOOKING (no slot freed).
        else      -> it is a one-hit-wonder: drop it, remember the id in G. Done.
    else:                take M's oldest.
        freq > 0  -> second chance: decrement and requeue at M's tail, KEEP LOOKING.
        else      -> drop it. Done (no ghost entry -- M evictions are not one-hit-wonders).

CAUSALITY: uses only the current request, its own admission/hit history, and per-expert
counters. `rank` is accepted and deliberately ignored -- this arm measures S3-FIFO as published,
so that any win or loss is attributable to the queue structure and not to a rank heuristic
layered on top.

COST: O(1) per hit and per admit. Eviction is amortized O(1); the worst case for a single
victim() call is bounded by the queue length because every loop iteration either strictly
decrements a saturating counter (cap 3) or moves an object S->M once. No scan over the full
expert set ever happens.
"""
from __future__ import annotations

from collections import OrderedDict, deque

from bigrig_engine.pool import Policy


class S3FIFO(Policy):
    name = "s3fifo"

    #: fraction of capacity given to the probationary queue
    S_RATIO = 0.10
    #: saturating cap on the per-expert hit counter
    FREQ_MAX = 3
    #: freq strictly greater than this in S promotes to M rather than being dropped
    PROMOTE_THRESHOLD = 1
    #: ghost queue length, as a multiple of capacity
    GHOST_RATIO = 1.0

    def __init__(self, capacity, n_experts):
        super().__init__(capacity, n_experts)
        self.s_target = max(1, int(round(self.S_RATIO * capacity)))
        self.ghost_max = max(1, int(round(self.GHOST_RATIO * capacity)))
        self.S: deque = deque()          # oldest at left
        self.M: deque = deque()          # oldest at left
        self.in_s: set = set()
        self.in_m: set = set()
        self.freq: dict = {}
        self.ghost: OrderedDict = OrderedDict()   # id -> True, oldest first

    # ---------------------------------------------------------------- requests

    def admit(self, e, rank=0):
        if e in self.in_s or e in self.in_m:      # defensive; simulate() never does this
            return
        self.freq[e] = 0
        if e in self.ghost:
            del self.ghost[e]
            self.M.append(e)
            self.in_m.add(e)
        else:
            self.S.append(e)
            self.in_s.add(e)

    def touch(self, e, rank=0):
        f = self.freq.get(e, 0)
        if f < self.FREQ_MAX:
            self.freq[e] = f + 1

    # ---------------------------------------------------------------- eviction

    def victim(self):
        while self.S or self.M:
            if self.S and len(self.S) >= self.s_target:
                v = self._evict_s()
            elif self.M:
                v = self._evict_m()
            else:
                v = self._evict_s()
            if v is not None:
                return v
        return -1

    def _evict_s(self):
        e = self.S.popleft()
        self.in_s.discard(e)
        if self.freq.get(e, 0) > self.PROMOTE_THRESHOLD:
            self.freq[e] = 0
            self.M.append(e)
            self.in_m.add(e)
            return None                      # promoted, no slot freed yet
        self.freq.pop(e, None)
        self._ghost_push(e)
        return e

    def _evict_m(self):
        e = self.M.popleft()
        f = self.freq.get(e, 0)
        if f > 0:
            self.freq[e] = f - 1
            self.M.append(e)                 # second chance, no slot freed yet
            return None
        self.in_m.discard(e)
        self.freq.pop(e, None)
        return e

    def _ghost_push(self, e):
        self.ghost[e] = True
        self.ghost.move_to_end(e)
        while len(self.ghost) > self.ghost_max:
            self.ghost.popitem(last=False)

    def evicted(self, e):
        # victim() already unlinked it; this only keeps state consistent if the harness ever
        # evicts something we did not choose.
        self.in_s.discard(e)
        self.in_m.discard(e)
        self.freq.pop(e, None)
        try:
            self.S.remove(e)
        except ValueError:
            pass
        try:
            self.M.remove(e)
        except ValueError:
            pass


def factory(capacity, n_experts):
    return S3FIFO(capacity, n_experts)
