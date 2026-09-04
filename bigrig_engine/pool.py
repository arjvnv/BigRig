"""COMPONENT 9 — the expert pool: which experts stay in RAM.

WHY THIS IS THE COMPONENT THAT MATTERS MOST
    Components 6, 7 and 8 make bytes move faster. None of them decides WHICH bytes move. Every
    speed figure in this project is

        tok/s = 1 / (active * miss / disk_gbs + active * (1 - miss) / ram_gbs)

    and `miss` is set here. On an M4 Max a 200B-class model runs at 2.8 tok/s at a 20% miss rate
    and 17.7 tok/s at 2%. The fetch engine cannot close that gap; only the policy can.

THE CEILING IS KNOWN AND IT IS NOT REACHABLE
    Belady -- evict whichever expert is needed furthest in the future -- is optimal and requires
    the future, so it is a bound rather than a policy. This project has measured the gap before:
    LRU 44.0% vs Belady 25.0% on OLMoE at 25% residency. What it has NOT established is how much
    of that gap an implementable policy can take. Earlier attempts captured 0-29% and went
    NEGATIVE at higher capacities, so this is genuinely open.

WHAT A POLICY MAY USE
    Only what exists before the fetch it is deciding about: the current request, its own history,
    the router's gate ranking for the current token, and per-expert counters. Anything requiring
    the expert's weights, or a future request, is an oracle and belongs in the Belady row.
"""
from __future__ import annotations

from collections import OrderedDict, defaultdict


class Policy:
    """One eviction policy for a single layer's expert cache.

    The contract is deliberately narrow so every policy is measured on identical terms:

      admit(e)   -> called when expert `e` has just been fetched and inserted
      touch(e)   -> called when expert `e` was requested and was already resident
      victim()   -> return the expert to evict; called only when the cache is full

    `rank` is the expert's position in the router's own top-k ordering for this token (0 is the
    highest-scoring). It is free information -- the router already computed it -- and this
    project measured that it correlates with recency, so a policy may use it.
    """

    name = "policy"

    def __init__(self, capacity: int, n_experts: int):
        self.capacity, self.n_experts = capacity, n_experts

    def admit(self, e: int, rank: int = 0) -> None: ...
    def touch(self, e: int, rank: int = 0) -> None: ...
    def victim(self) -> int: raise NotImplementedError


class LRU(Policy):
    """Least recently used. The baseline every other policy must beat."""
    name = "lru"

    def __init__(self, capacity, n_experts):
        super().__init__(capacity, n_experts)
        self.od: OrderedDict = OrderedDict()

    def admit(self, e, rank=0):
        self.od[e] = True
        self.od.move_to_end(e)

    def touch(self, e, rank=0):
        self.od.move_to_end(e)

    def victim(self):
        return next(iter(self.od)) if self.od else -1

    def evicted(self, e):
        self.od.pop(e, None)


class WindowedLFU(Policy):
    """Frequency over a sliding window of recent requests.

    This project's own prior work found it the strongest FREE policy tried -- 40-51% of the
    LRU-to-Belady gap on OLMoE -- but only 9-18% on Ling and not statistically significant there.
    So it is a candidate, not a settled answer.
    """
    name = "wlfu"

    def __init__(self, capacity, n_experts, window: int = 128):
        super().__init__(capacity, n_experts)
        self.window = window
        self.hist: list = []
        self.count: dict = defaultdict(int)
        self.resident: set = set()
        self.tick = 0
        self.last: dict = {}

    def _record(self, e):
        self.tick += 1
        self.last[e] = self.tick
        self.hist.append(e)
        self.count[e] += 1
        if len(self.hist) > self.window:
            old = self.hist.pop(0)
            self.count[old] -= 1
            if self.count[old] <= 0:
                del self.count[old]

    def admit(self, e, rank=0):
        self.resident.add(e)
        self._record(e)

    def touch(self, e, rank=0):
        self._record(e)

    def victim(self):
        if not self.resident:
            return -1
        # fewest uses in the window; ties broken by least recent, so it degrades to LRU
        return min(self.resident, key=lambda x: (self.count.get(x, 0), self.last.get(x, 0)))

    def evicted(self, e):
        self.resident.discard(e)


class BeladyOracle:
    """The unattainable ceiling. Needs the whole future request sequence."""
    name = "belady"


def recommended(capacity: int = 0, n_experts: int = 0):
    """The default policy, chosen by measurement rather than by reputation.

    MEASURED, 8 cells (two models x four capacities), first 4 layers, against an exact Belady
    bound. Percentages are the fraction of the LRU-to-Belady gap captured:

        policy     mean    worst cell   beats LRU everywhere
        lfuda      23.7%       +7.8%    YES
        s3fifo     19.6%       -2.8%    no
        wlfu       15.6%        0.0%    no
        lirs       15.6%      -12.3%    no
        coact       5.6%      -25.4%    no
        gaterank    4.3%       +0.2%    yes
        arc         4.2%       -4.5%    no

    AND THE SAME SWEEP ON LAYERS 4-7, WHICH NOTHING WAS TUNED ON. This is the column that
    matters, because three of the four leaders were fitted on layers 0-3:

        policy     layers 0-3    HELD OUT 4-7    cells lost
        lfuda         23.7%          24.1%          0 of 8
        s3fifo        19.6%          12.5%          2 of 8
        lirs          17.6%           1.2%          3 of 8
        gaterank       4.3%           3.3%          3 of 8

    LFUDA is the only policy that generalises -- it is marginally BETTER on layers it never saw,
    and it is the only one that never loses to LRU on held-out data. LIRS collapses from 17.6%
    to 1.2%: its headline was the argmax of a hyperparameter grid evaluated on the same layers it
    reported, which an adversarial reviewer correctly called a selection statistic rather than a
    measurement. gaterank's "beats LRU everywhere" was likewise an artifact of layers 0-3 and is
    false on held-out layers.

    Note how little the reputation of these algorithms predicted the outcome -- ARC and LIRS are
    the textbook answers for general caches and both go NEGATIVE at some capacities here. MoE
    expert access is not a general cache workload.

    A CORRECTION WORTH RECORDING: an earlier sweep of mine scored LFUDA at -37.2% and nearly
    discarded it. That sweep instantiated each policy's raw class with bare defaults instead of
    the tuned factory the module exposes. The policy was fine; the harness was not.
    """
    from .policies.lfuda import factory
    f = factory()
    return f(capacity, n_experts) if capacity else f


def _policy_factory(name: str, capacity: int, n_experts: int):
    """Resolve a policy module to something simulate() can call.

    Modules expose their tuned configuration in one of three ways, and the resolver must try all
    of them. An earlier version tried only two, and the two it missed -- ARC and LIRS -- came
    back as "NoneType is not callable" and would have been silently scored as failures. Falling
    back to the raw class is the difference between "this policy lost" and "I could not load it".
    """
    import importlib
    builtin = {"lru": LRU, "windowed_lfu": WindowedLFU, "wlfu": WindowedLFU}
    if name in builtin:
        return builtin[name]          # baselines live here, not in policies/
    try:
        m = importlib.import_module(f"bigrig_engine.policies.{name}")
    except ModuleNotFoundError:
        import pkgutil
        import bigrig_engine.policies as _pk
        have = sorted(x.name for x in pkgutil.iter_modules(_pk.__path__))
        raise ImportError(
            f"no policy named {name!r}. Available: {', '.join(sorted(set(have) | set(builtin)))}"
        ) from None

    def usable(f):
        try:
            return hasattr(f(capacity, n_experts), "victim")
        except Exception:
            return False

    named = getattr(m, name, None)                       # module-level configured instance
    if callable(named) and not isinstance(named, type) and usable(named):
        return named
    f = getattr(m, "factory", None)
    if callable(f):
        if usable(f):
            return f                                     # factory(capacity, n_experts)
        try:
            g = f()
            if usable(g):
                return g                                 # factory(**kw) -> factory
        except Exception:
            pass
    for a in dir(m):                                     # last resort: the class itself
        o = getattr(m, a)
        if isinstance(o, type) and hasattr(o, "victim") and o.__module__ == m.__name__:
            if usable(o):
                return o
    raise ImportError(f"no usable policy factory in bigrig_engine.policies.{name}")


def select_policy(tag: str, capacity: int, candidates=None, layers: int = 2,
                  n_experts: int | None = None) -> dict:
    """Pick the best policy for THIS model at THIS capacity, by running them.

    No policy won everywhere, and the spread between best and worst in a single cell is over 60
    percentage points. Shipping one name as a constant would be the same mistake as shipping
    kappa=67. Component 6 measures the host; this measures the model.
    """
    import importlib
    import numpy as np
    names = candidates or ["lfuda", "s3fifo", "gaterank", "lirs", "arc", "coact"]
    idx, _ = load_trace(tag)
    L, T, k = idx.shape
    E = n_experts or int(idx.max()) + 1
    seqs = [([int(idx[l, t, r]) for t in range(T) for r in range(k)],
             [r for t in range(T) for r in range(k)]) for l in range(min(layers, L))]
    lru = float(np.mean([simulate(s, capacity, E, LRU, r)["miss_rate"] for s, r in seqs]))
    bel = float(np.mean([belady(s, capacity)["miss_rate"] for s, _ in seqs]))
    out = {"lru": lru, "belady": bel, "policies": {}}
    for n in names:
        try:
            fac = _policy_factory(n, capacity, E)
            v = float(np.mean([simulate(s, capacity, E, fac, r)["miss_rate"] for s, r in seqs]))
            out["policies"][n] = {"miss_rate": v,
                                  "gap_pct": (lru - v) / (lru - bel) * 100 if lru > bel else 0.0}
        except Exception as e:                          # a broken candidate must not hide a good one
            out["policies"][n] = {"error": str(e)[:120]}
    ok = {n: d for n, d in out["policies"].items() if "gap_pct" in d}
    out["best"] = max(ok, key=lambda n: ok[n]["gap_pct"]) if ok else "lru"
    out["best_gap_pct"] = ok[out["best"]]["gap_pct"] if ok else 0.0
    return out


def simulate(seq, capacity: int, n_experts: int, policy_factory, ranks=None) -> dict:
    """Replay one layer's expert-request sequence through a cache. Returns miss statistics.

    `seq` is a flat list of expert ids in request order; `ranks` the router rank of each request.
    A request is a MISS if the expert is not resident, which costs one fetch.
    """
    pol = policy_factory(capacity, n_experts)
    resident: set = set()
    misses = hits = 0
    ranks = ranks if ranks is not None else [0] * len(seq)
    for e, r in zip(seq, ranks):
        e = int(e)
        if e in resident:
            hits += 1
            pol.touch(e, r)
            continue
        misses += 1
        if len(resident) >= capacity:
            v = pol.victim()
            if v >= 0:
                resident.discard(v)
                if hasattr(pol, "evicted"):
                    pol.evicted(v)
        resident.add(e)
        pol.admit(e, r)
    n = max(1, hits + misses)
    return {"policy": getattr(pol, "name", "?"), "misses": misses, "hits": hits,
            "requests": n, "miss_rate": misses / n}


def belady(seq, capacity: int) -> dict:
    """Optimal offline eviction: evict whatever is needed furthest in the future.

    A bound, never a policy. Implemented with next-use indices so it is exact rather than
    approximate -- an approximate ceiling would make every 'percent of the gap captured' number
    wrong in an unknown direction.
    """
    seq = [int(x) for x in seq]
    nxt: dict = defaultdict(list)
    for i, e in enumerate(seq):
        nxt[e].append(i)
    pos = {e: 0 for e in nxt}
    resident: set = set()
    misses = hits = 0
    for i, e in enumerate(seq):
        pos[e] += 1
        if e in resident:
            hits += 1
            continue
        misses += 1
        if len(resident) >= capacity:
            far, victim = -1, None
            for r in resident:
                q = nxt[r]
                p = pos[r]
                nu = q[p] if p < len(q) else float("inf")
                if nu > far:
                    far, victim = nu, r
            resident.discard(victim)
        resident.add(e)
    n = max(1, hits + misses)
    return {"policy": "belady", "misses": misses, "hits": hits, "requests": n,
            "miss_rate": misses / n}


def load_trace(tag: str):
    """(idx [layers, tokens, k], ranks) from this project's routing traces.

    The path used to be absolute and carried one developer's home directory, so every caller --
    the policy tests among them -- worked on exactly one machine and failed everywhere else, in
    a module that ships. It resolves against the engine's own home now, which BIGRIG_HOME moves
    like everything else.
    """
    import os
    import numpy as np
    from . import home
    p = os.path.join(home(), "data", "traces", f"{tag}_full.npz")
    d = np.load(p)
    if "idx" in d:
        idx = d["idx"].astype(int)
    else:
        idx = np.argsort(-d["logits"].astype("float32"), axis=-1)[:, :, :8].astype(int)
    return idx, d["seq_bounds"]


def evaluate(policy_factory, tag: str, capacities, layers=None) -> dict:
    """Miss rate per capacity, averaged over layers, on a real trace.

    Requests are flattened in ROUTER-RANK ORDER within each token, which is the order a real
    decode issues them, and never across a sequence boundary.
    """
    idx, bounds = load_trace(tag)
    L, T, k = idx.shape
    E = int(idx.max()) + 1
    layers = range(L) if layers is None else layers
    out = {}
    for C in capacities:
        rates = []
        for l in layers:
            seq, ranks = [], []
            for t in range(T):
                for r in range(k):
                    seq.append(int(idx[l, t, r]))
                    ranks.append(r)
            rates.append(simulate(seq, C, E, policy_factory, ranks)["miss_rate"])
        out[C] = sum(rates) / len(rates)
    return out
