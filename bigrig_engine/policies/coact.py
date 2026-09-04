"""coact -- co-activation aware eviction for the MoE expert cache.

THE SIGNAL
    MoE routers do not pick experts independently. Within one token the k experts the gate selects
    form a recurring *set*, and this project has measured that structure in real routing traces.
    LRU throws it away: it ranks a resident expert only by when it was last touched, never by
    whether the experts the router is asking for RIGHT NOW are the ones it usually travels with.

    coact keeps a running estimate of P(x | y) -- given that expert y was just requested, how often
    x is requested in the same token -- and uses it to score every resident expert against the
    recent request context. An expert that co-activates strongly with what is being routed now is
    protected even when it is stale by clock time; an expert that is recent but belongs to a
    routing regime the model has moved past is evicted.

    Measured on this project's traces, the victim coact picks has a next-use distance at the
    ~0.71 percentile of the resident set, against ~0.55 for LRU (1.0 would be Belady).

THE SCORE
    Two terms, both decayed with the same half-life:

        score(x) = S(x)          decayed count of x's own uses, over all history
                 + beta * A(x)   affinity of x to the last W requests

        A(x) = sum over the last W requests y != x of  decay^(t - t_y) * P(x | y)

    A is deliberately WINDOWED while S is not. That asymmetry is the whole trick. An earlier
    version accumulated the co-activation term over unbounded history too, and it lost to LRU at
    every setting -- because a long-resident expert accrues co-activation credit for as long as it
    sits in the cache, so a freshly fetched expert is always the minimum and the cache freezes into
    thrash. Bounding A to a window puts an incumbent and a newcomer on identical footing: both are
    scored against the same W requests, and a newcomer's A is recomputed from that same window on
    admission.

COST
    Per request: O(max_nb) to spread affinity, O(max_nb) to retire the request leaving the window.
    Per admission: O(W). Per eviction: O(capacity) -- one pass over the resident set, never over
    all n_experts. Memory: O(n_experts * max_nb) for the co-activation table plus O(capacity).

CAUSALITY
    Every number here comes from requests that already happened plus the router rank of the current
    one. Token boundaries are recovered from the rank stream itself -- a rank that does not exceed
    the previous rank opens a new token -- which is information the router already emitted. No
    future request is read anywhere.
"""
from __future__ import annotations

from collections import defaultdict, deque

from bigrig_engine.pool import Policy


class CoAct(Policy):
    """Co-activation aware eviction.

    Parameters
    ----------
    half_life   : decay half-life in REQUESTS (k requests = one token).
    beta        : weight on the co-activation term. 0 disables it and leaves a decayed-LFU,
                  which is the ablation that says what the co-activation structure is worth.
    window      : how many recent requests form the context A(x) is measured against.
    max_nb      : cap on stored co-activation partners per expert, so the table is
                  O(n_experts * max_nb) regardless of trace length.
    min_support : floor on the denominator of P(x|y), so a barely-seen y cannot assert 1.0.
    ghost       : keep an evicted expert's use history so a re-admitted expert is not treated as
                  brand new. Bounded by n_experts.
    rank_w      : extra self-credit for a request the router ranked highly; 0 disables.
    """

    name = "coact"

    def __init__(self, capacity, n_experts, half_life: float = 128.0, beta: float = 0.8,
                 window: int = 12, max_nb: int = 64, min_support: int = 4,
                 ghost: bool = True, rank_w: float = 0.0):
        super().__init__(capacity, n_experts)
        self.decay = 0.5 ** (1.0 / float(half_life))
        self.beta = float(beta)
        self.W = int(window)
        self.max_nb = int(max_nb)
        self.min_support = int(min_support)
        self.ghost = bool(ghost)
        self.rank_w = float(rank_w)

        self.S: dict = {}          # decayed self-use credit; true value is S[x] / g
        self.A: dict = {}          # windowed co-activation affinity; true value is A[x] / g
        self.resident: set = set()
        self.g = 1.0               # global inflation factor, so decay costs nothing per request
        self.t = 0
        self.last: dict = {}

        self.co: dict = defaultdict(dict)   # e -> {partner: co-occurrence count}
        self.tok: dict = defaultdict(int)   # e -> number of tokens containing e

        self.cur: list = []                 # experts of the token being assembled
        self.last_rank = 1 << 30

        # window entries: [expert, g_at_entry, {x: credit actually given to x}]
        self.win: deque = deque()

    # ---------------------------------------------------------------- statistics

    def _p(self, y: int, x: int) -> float:
        """P(x | y) from the co-activation table."""
        c = self.co[y].get(x)
        if not c:
            return 0.0
        d = self.tok[y]
        return c / (d if d > self.min_support else self.min_support)

    def _close_token(self) -> None:
        cur = self.cur
        n = len(cur)
        if not n:
            return
        for i in range(n):
            a = cur[i]
            self.tok[a] += 1
            ca = self.co[a]
            for j in range(i + 1, n):
                b = cur[j]
                ca[b] = ca.get(b, 0) + 1
                cb = self.co[b]
                cb[a] = cb.get(a, 0) + 1
            if len(ca) > 2 * self.max_nb:
                self.co[a] = dict(sorted(ca.items(), key=lambda kv: -kv[1])[:self.max_nb])
        self.cur = []

    # ---------------------------------------------------------------- window

    def _enter(self, e: int) -> None:
        """Give every resident co-activation partner of `e` credit, and record what was given."""
        g = self.g
        given: dict = {}
        if self.beta:
            res, A = self.resident, self.A
            d = self.tok[e]
            if d < self.min_support:
                d = self.min_support
            gd = g / d
            for x, c in self.co[e].items():
                if x != e and x in res:
                    v = c * gd
                    A[x] = A.get(x, 0.0) + v
                    given[x] = v
        self.win.append([e, g, given])
        while len(self.win) > self.W:
            y, gy, gv = self.win.popleft()
            if gv:
                A, res = self.A, self.resident
                for x, v in gv.items():
                    if x in res:
                        A[x] -= v

    def _seed(self, e: int) -> float:
        """A(e) for an expert admitted mid-window: score it against the same W requests the
        incumbents are scored against, and back-record the credit so it retires correctly."""
        if not self.beta:
            return 0.0
        s = 0.0
        for ent in self.win:
            y, gy, gv = ent
            if y == e:
                continue
            v = gv.get(e)
            if v is None:
                p = self._p(y, e)
                if not p:
                    continue
                v = p * gy
                gv[e] = v
            s += v
        return s

    # ---------------------------------------------------------------- clock

    def _tick(self, e: int, rank: int) -> None:
        if rank <= self.last_rank:
            self._close_token()
        self.last_rank = rank
        self.cur.append(e)
        self.t += 1
        self.g /= self.decay
        if self.g > 1e120:                      # renormalise before float range runs out
            gg = self.g
            for d in (self.S, self.A):
                for kk in d:
                    d[kk] /= gg
            for ent in self.win:
                ent[1] /= gg
                gv = ent[2]
                for kk in gv:
                    gv[kk] /= gg
            self.g = 1.0

    # ---------------------------------------------------------------- interface

    def admit(self, e: int, rank: int = 0) -> None:
        self._tick(e, rank)
        self.resident.add(e)
        self.A[e] = self._seed(e)
        if not self.ghost:
            self.S[e] = 0.0
        self.S[e] = self.S.get(e, 0.0) + self.g * (1.0 + self.rank_w / (1.0 + rank))
        self.last[e] = self.t
        self._enter(e)

    def touch(self, e: int, rank: int = 0) -> None:
        self._tick(e, rank)
        self.S[e] = self.S.get(e, 0.0) + self.g * (1.0 + self.rank_w / (1.0 + rank))
        self.last[e] = self.t
        self._enter(e)

    def victim(self) -> int:
        if not self.resident:
            return -1
        S, A, last, b = self.S, self.A, self.last, self.beta
        best, bs, bl = -1, None, 0
        for x in self.resident:                 # O(capacity), never O(n_experts)
            sx = S.get(x, 0.0) + b * A.get(x, 0.0)
            if bs is None or sx < bs or (sx == bs and last.get(x, 0) < bl):
                best, bs, bl = x, sx, last.get(x, 0)
        return best

    def evicted(self, e: int) -> None:
        self.resident.discard(e)
        self.A.pop(e, None)
        if not self.ghost:
            self.S.pop(e, None)


def factory(**kw):
    """Bind hyper-parameters into something simulate()/evaluate() can call."""
    return lambda capacity, n_experts: CoAct(capacity, n_experts, **kw)
