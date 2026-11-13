"""Held prefill views: a layer's expert arrays kept across the chunks of layer-major prefill.

WHY THIS FILE EXISTS
    StreamingSwitchGLU.holding() lets every chunk of a prompt reuse the expert arrays an earlier
    chunk built for the same layer, instead of fetching, probing and wrapping them again. That is
    only allowed if nothing a chunk computes changes. This file checks that on real float
    weights, for both prefill paths (the per-expert views path and the exact sorted-gather path):
    outputs bit-identical with and without holding, each expert fetched once per block instead
    of once per chunk, nothing held after the block, and the decoder-layer plumbing
    (StreamHandle.prefill_layer, layered.prefill's layer_scope) entering one block per layer, in order.

Run: .venv/bin/python tests/test_held_views.py
"""
import os
import sys

import mlx.core as mx
import mlx.nn as nn
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from bigrig_engine import stream  # noqa: E402

FAIL = []


def check(n, c, d=""):
    (print(f"  PASS  {n}") if c else (FAIL.append(n), print(f"  FAIL  {n}  {d}")))


E, C, D, K = 12, 4, 16, 3
SPEC = {p: {"weight": {"shape": [D, D], "dtype": "float32", "nbytes": D * D * 4}} for p in stream.PROJECTIONS}
rng = np.random.default_rng(0)
BLOBS = {e: rng.standard_normal((len(stream.PROJECTIONS), D, D)).astype(np.float32).tobytes() for e in range(E)}


class _SG:
    def __init__(self):
        for p in stream.PROJECTIONS:
            setattr(self, p, {"weight": mx.zeros((E, D, D), dtype=mx.float32)})


class _Fetcher:
    """Serves each expert's bytes as a memoryview, as WeightStore does, and records every call."""

    def __init__(self, refuse=False):
        self.asked, self.events, self.refuse = [], [], refuse

    def fetch(self, keys):
        keys = list(keys)
        self.asked += keys
        self.events.append(("fetch", keys))
        return {k: memoryview(BLOBS[k[1]]) for k in keys}

    def prefetch(self, keys):
        if self.refuse:
            raise RuntimeError("pending cap")
        self.events.append(("prefetch", list(keys)))

    def drop(self, keys):
        self.events.append(("drop", list(keys)))


def module(layer=0, fetcher=None):
    pool = stream.ExpertPool(layer, _SG(), E, C, SPEC, policy="lfuda", views=True).build()
    return stream.StreamingSwitchGLU(pool, fetcher or _Fetcher(), lambda up, gate: nn.silu(gate) * up,
                                     quantized=False, qparams={})


CHUNKS = []
for n in (9, 9, 5):
    x = mx.array(rng.standard_normal((n, D)).astype(np.float32))
    flat = np.stack([rng.choice(E, K, replace=False) for _ in range(n)])
    CHUNKS.append((x, flat))
DISTINCT = len(np.unique(np.concatenate([f.reshape(-1) for _, f in CHUNKS])))
PER_CHUNK = sum(len(np.unique(f)) for _, f in CHUNKS)


def run(path, hold):
    m = module()
    fwd = getattr(m, path)
    if hold:
        with m.holding():
            ys = [fwd(x, f) for x, f in CHUNKS]
    else:
        ys = [fwd(x, f) for x, f in CHUNKS]
    mx.eval(ys)
    return m, ys


print("=" * 78)
print("1. HOLDING CHANGES NO OUTPUT, ON EITHER PREFILL PATH")
print("=" * 78)
for path in ("_forward_views", "_forward_views_exact"):
    m0, y0 = run(path, hold=False)
    m1, y1 = run(path, hold=True)
    check(f"{path}: every chunk bit-identical with and without holding",
          all(mx.array_equal(a, b).item() for a, b in zip(y0, y1)))
    check(f"{path}: the outputs are not trivially zero (the comparison can fail)",
          all(float(mx.abs(a).max()) > 0 for a in y0))
    check(f"{path}: without holding, each chunk fetches its own experts ({PER_CHUNK} fetches)",
          len(m0._fetcher.asked) == PER_CHUNK, len(m0._fetcher.asked))
    check(f"{path}: holding, each expert is fetched once for the block ({DISTINCT} fetches)",
          len(m1._fetcher.asked) == DISTINCT and len(set(m1._fetcher.asked)) == DISTINCT,
          len(m1._fetcher.asked))
    check(f"{path}: nothing is held after the block", m1._held is None)

# The reference itself: a chunk computed by hand from the experts' weights, so "identical with and
# without holding" cannot pass by both sides being wrong in the same way.
x, flat = CHUNKS[0]
w = {e: np.frombuffer(BLOBS[e], dtype=np.float32).reshape(len(stream.PROJECTIONS), D, D) for e in range(E)}
gi, ui, di = (stream.PROJECTIONS.index(p) for p in ("gate_proj", "up_proj", "down_proj"))
xn = np.array(x)
ref = np.stack([np.stack([
    ((xn[t] @ w[e][ui].T) * (lambda g: g / (1 + np.exp(-g)))(xn[t] @ w[e][gi].T)) @ w[e][di].T
    for e in flat[t]]) for t in range(len(xn))]).reshape(-1, D)
_, yh = run("_forward_views", hold=True)
rel = float(np.abs(np.array(yh[0]) - ref).max() / np.abs(ref).max())
check("the held path matches a by-hand computation of the same experts (max relative |d| < 1e-5)",
      rel < 1e-5, rel)

print("=" * 78)
print("2. A FAILED CHUNK STILL RELEASES THE LAYER")
print("=" * 78)
m = module()
try:
    with m.holding():
        m._forward_views(*CHUNKS[0])
        raise RuntimeError("boom")
except RuntimeError:
    pass
check("an exception inside the block still drops the held arrays", m._held is None)

print("=" * 78)
print("3. THE DECODER-LAYER PLUMBING")
print("=" * 78)
h = stream.StreamHandle.__new__(stream.StreamHandle)
h.mods = [module()]
with h.prefill_layer(0):
    held_inside = h.mods[0]._held is not None
check("StreamHandle.prefill_layer(layer) holds that layer's module", held_inside and h.mods[0]._held is None)
with h.prefill_layer(7):
    pass
check("a layer the engine does not stream gets a no-op block", h.mods[0]._held is None)
h.mods[0]._pool = None
with h.prefill_layer(0):
    pass
check("a closed module (no pool) gets a no-op block, not an error", True)

print("=" * 78)
print("4. READ-AHEAD: THE NEXT LAYER'S READS START AFTER THIS LAYER'S, AND LEFTOVERS ARE RELEASED")
print("=" * 78)
f = _Fetcher()
h = stream.StreamHandle.__new__(stream.StreamHandle)
h.manifest = {"top_k": K}
h.mods = [module(0, f), module(1, f)]
with h.prefill_layer(0, read_ahead=True):
    before_first = list(f.events)
    ys = [h.mods[0]._forward_views(x, fl) for x, fl in CHUNKS]
kinds = [ev[0] for ev in f.events]
check("nothing is read ahead before layer 0's first chunk fetches", before_first == [], before_first)
check("layer 1 is read ahead exactly once, right after layer 0's first fetch",
      kinds[:2] == ["fetch", "prefetch"] and kinds.count("prefetch") == 1, kinds)
check("...and the read-ahead names every expert of layer 1",
      f.events[1][1] == [(1, e) for e in range(E)], f.events[1][1][:3])
check("layer 0's own block releases nothing it did not read ahead", "drop" not in kinds, kinds)
with h.prefill_layer(1, read_ahead=True):
    h.mods[1]._forward_views(*CHUNKS[0])
check("layer 1's block, ending, releases the read-ahead keys no chunk claimed",
      f.events[-1] == ("drop", [(1, e) for e in range(E)]), f.events[-1][:1])
check("...and the last layer reads nothing further ahead",
      [ev[0] for ev in f.events].count("prefetch") == 1)
_, ref = run("_forward_views", hold=False)
check("read-ahead changes no output", all(mx.array_equal(a, b).item() for a, b in zip(ys, ref)))
f = _Fetcher()
h.mods = [module(0, f), module(1, f)]
with h.prefill_layer(0, read_ahead=False):
    h.mods[0]._forward_views(*CHUNKS[0])
check("read_ahead=False reads nothing ahead", all(ev[0] == "fetch" for ev in f.events))
f = _Fetcher(refuse=True)
m = module(1, f)
m.read_ahead()
check("a refused read-ahead (pending cap) is skipped, not raised, and holds nothing", m._ahead == [])
check("short prompts do not read ahead: 15 tokens x top-3 < 4 x 12 experts",
      not h.routes_to_most_experts(15) and h.routes_to_most_experts(16))

print("=" * 78)
print("5. REWARM: EACH LAYER'S MOST-ROUTED EXPERTS, COUNTED ONLY DURING LAYER-MAJOR PREFILL")
print("=" * 78)


class _RFetcher(_Fetcher):
    def refresh(self, keys):
        self.events.append(("refresh", list(keys)))
        return 3


f = _RFetcher()
m0, m1 = module(0, f), module(1, f)
m0._forward_views(*CHUNKS[0])
check("routing outside a held block is not counted", m0.prefill_routed is None)
with m0.holding():
    for x, fl in CHUNKS:
        m0._forward_views(x, fl)
with m1.holding():
    m1._forward_views_exact(*CHUNKS[1])
want0 = np.bincount(np.concatenate([fl.reshape(-1) for _, fl in CHUNKS]), minlength=E)
check("a held block counts every (token, expert) pair of every chunk", np.array_equal(m0.prefill_routed, want0))
check("...on the exact path too", np.array_equal(m1.prefill_routed, np.bincount(CHUNKS[1][1].reshape(-1), minlength=E)))
with m0.holding():
    pass
check("a new block starts the count again", m0.prefill_routed.sum() == 0)
with m0.holding():
    for x, fl in CHUNKS:
        m0._forward_views(x, fl)
h = stream.StreamHandle.__new__(stream.StreamHandle)
h.mods, h.fetcher = [m1, m0], f
cold = h.rewarm(3)
keys = f.events[-1][1]
top0 = [int(e) for e in np.argsort(-want0, kind="stable")[:3]]
check("rewarm reads layer 0's three most-routed experts first, then layer 1's", keys[:3] == [(0, e) for e in top0]
      and all(k[0] == 1 for k in keys[3:]), keys)
check("...never an expert the prompt did not route to", all(m1.prefill_routed[e] > 0 for _, e in keys[3:]))
check("...and reports the fetcher's cold count", cold == 3)

print("=" * 78)
print("6. DECODE WARMING: THE NEXT LAYER'S GUESSES, QUEUED AFTER THIS LAYER'S OWN FETCH")
print("=" * 78)


class _WFetcher(_Fetcher):
    def warm(self, keys, max_queued=64):
        self.events.append(("warm", list(keys)))
        return len(keys)


W = mx.array(rng.standard_normal((D, E)).astype(np.float32))
wf = _WFetcher()
a, b = module(0, wf), module(1, wf)
a._next, a._next_gate, a._pred_m = b._pool, (lambda xf: xf @ W), 3
b._pool.touch([0], [0]); b._pool.admit(0, memoryview(BLOBS[0]))      # expert 0 already in the next pool
x1 = mx.array(rng.standard_normal((1, D)).astype(np.float32))
a(x1, mx.array(np.array([[3, 4, 5]])))
kinds = [ev[0] for ev in wf.events]
guess = np.argpartition(-np.array(mx.softmax(x1 @ W, axis=-1, precise=True))[0], 2)[:3]
want = [(1, int(e)) for e in guess if e != 0]
check("a decode step warms once, after its own fetch", kinds == ["fetch", "warm"], kinds)
check("...naming the next router's top guesses that are not already in the next pool",
      sorted(wf.events[-1][1]) == sorted(want), (wf.events[-1][1], want))
wf2 = _WFetcher()
a2, b2 = module(0, wf2), module(1, wf2)
a2._next, a2._next_gate, a2._pred_m = b2._pool, (lambda xf: xf @ W), 3
a2(mx.array(rng.standard_normal((5, D)).astype(np.float32)),
   mx.array(np.stack([rng.choice(E, K, replace=False) for _ in range(5)])))
check("a prefill chunk warms nothing (long prompts read whole layers ahead instead)",
      "warm" not in [ev[0] for ev in wf2.events])
saved = stream.PREFETCH_WARM
stream.PREFETCH_WARM = False
wf3 = _WFetcher()
a3, b3 = module(0, wf3), module(1, wf3)
a3._next, a3._next_gate, a3._pred_m = b3._pool, (lambda xf: xf @ W), 3
a3(x1, mx.array(np.array([[3, 4, 5]])))
stream.PREFETCH_WARM = saved
check("BIGRIG_PREFETCH_WARM=0 turns it off", [ev[0] for ev in wf3.events] == ["fetch"])

print("\n" + "=" * 78)
print(f"{'ALL TESTS PASSED' if not FAIL else str(len(FAIL)) + ' FAILURES: ' + ', '.join(FAIL)}")
print("=" * 78)
sys.exit(1 if FAIL else 0)
