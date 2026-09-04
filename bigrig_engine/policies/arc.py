"""ARC -- Adaptive Replacement Cache (Megiddo & Modha, FAST '03) for the MoE expert pool.

THE IDEA
    A single LRU list cannot tell "this expert was touched once, recently" apart from "this
    expert is touched constantly". ARC splits residency into two LRU lists:

        T1  experts seen exactly once since they entered  -> the RECENCY half
        T2  experts seen at least twice                   -> the FREQUENCY half

    and keeps two GHOST lists of ids only (no weights, no data):

        B1  recently evicted out of T1
        B2  recently evicted out of T2

    A request that hits a ghost is evidence about which half was starved. A hit in B1 means we
    threw away a recency entry too early, so the target size p of T1 grows. A hit in B2 means we
    threw away a frequency entry too early, so p shrinks. Eviction then takes from whichever of
    T1/T2 is currently over its share. The split is learned per layer, online, from the trace
    itself -- nothing is tuned by hand.

WHY IT SHOULD SUIT THIS WORKLOAD
    Expert traces are a mixture: a small set of hot experts fired by nearly every token, plus a
    long tail of one-off experts dragged in by a single token's top-k. Plain LRU lets the tail
    evict the hot set (scan pollution). ARC quarantines every newcomer in T1, so a burst of
    one-off experts mostly evicts other one-off experts, while the hot set lives in T2.

CAUSALITY
    Uses only: the current request, ids of previously evicted experts, and its own list order.
    No future requests, no weights. `rank` is accepted for interface compatibility and is
    deliberately UNUSED -- this is textbook ARC, so that the number it produces is attributable
    to the ARC mechanism and not to a rank heuristic bolted on top.

COST
    O(1) per admit / touch / victim / evicted. Four OrderedDicts; |T1|+|T2| = capacity and
    |T1|+|T2|+|B1|+|B2| <= 2*capacity, so memory is bounded by 2*capacity ints regardless of
    how many experts the model has. Nothing ever scans the full expert set.

ONE DEVIATION FROM THE PAPER, STATED PLAINLY
    In the paper, a ghost hit adjusts p and only THEN calls REPLACE to pick a victim. This
    harness calls victim() before admit(), and victim() is not told which expert is arriving,
    so the arrival cannot be inspected until after the eviction has happened. Therefore the p
    update from a ghost hit lands one eviction late. p is a slow-moving control variable, so
    this is a one-step lag in a smoothed signal, not a change of algorithm.
"""
from __future__ import annotations

from collections import OrderedDict

from bigrig_engine.pool import Policy


class ARC(Policy):
    """Adaptive Replacement Cache."""

    name = "arc"

    def __init__(self, capacity: int, n_experts: int):
        super().__init__(capacity, n_experts)
        self.c = capacity
        self.p = 0.0            # learned target size for T1
        self.t1: OrderedDict = OrderedDict()   # resident, seen once   (LRU at front)
        self.t2: OrderedDict = OrderedDict()   # resident, seen 2+     (LRU at front)
        self.b1: OrderedDict = OrderedDict()   # ghosts evicted from T1
        self.b2: OrderedDict = OrderedDict()   # ghosts evicted from T2

    # ---------------------------------------------------------------- helpers
    def _trim_ghosts(self) -> None:
        """Keep the paper's invariants: |T1|+|B1| <= c and |L1|+|L2| <= 2c."""
        c = self.c
        while len(self.t1) + len(self.b1) > c and self.b1:
            self.b1.popitem(last=False)
        while (len(self.t1) + len(self.t2) + len(self.b1) + len(self.b2)) > 2 * c:
            if self.b2:
                self.b2.popitem(last=False)
            elif self.b1:
                self.b1.popitem(last=False)
            else:
                break

    # ---------------------------------------------------------------- contract
    def admit(self, e: int, rank: int = 0) -> None:
        """`e` missed and has just been inserted. Ghost membership decides where it lands."""
        if e in self.b1:
            # We evicted a recency entry too soon -> favour T1.
            delta = max(1.0, len(self.b2) / max(1, len(self.b1)))
            self.p = min(float(self.c), self.p + delta)
            del self.b1[e]
            self.t2[e] = True            # a second sighting: it is now frequency
        elif e in self.b2:
            # We evicted a frequency entry too soon -> favour T2.
            delta = max(1.0, len(self.b1) / max(1, len(self.b2)))
            self.p = max(0.0, self.p - delta)
            del self.b2[e]
            self.t2[e] = True
        else:
            self.t1.pop(e, None)
            self.t2.pop(e, None)
            self.t1[e] = True            # brand new: quarantine in the recency half
        self._trim_ghosts()

    def touch(self, e: int, rank: int = 0) -> None:
        """`e` hit while resident. One sighting promotes T1 -> T2; T2 just refreshes."""
        if e in self.t1:
            del self.t1[e]
            self.t2[e] = True
        elif e in self.t2:
            self.t2.move_to_end(e)
        else:
            self.t2[e] = True            # defensive; should not occur

    def victim(self) -> int:
        """Evict the LRU of whichever half is over its learned share."""
        if self.t1 and (len(self.t1) > self.p or not self.t2):
            return next(iter(self.t1))
        if self.t2:
            return next(iter(self.t2))
        if self.t1:
            return next(iter(self.t1))
        return -1

    def evicted(self, e: int) -> None:
        """The victim left RAM but its id is kept as a ghost -- that is what makes p adapt."""
        if e in self.t1:
            del self.t1[e]
            self.b1[e] = True
        elif e in self.t2:
            del self.t2[e]
            self.b2[e] = True
        self._trim_ghosts()
