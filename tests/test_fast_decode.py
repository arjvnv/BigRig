"""The decode host path (stream.FAST_DECODE) must change no bit and no eviction.

WHY THIS FILE EXISTS
    FAST_DECODE removes host work from a decode token: experts already in the page cache are
    wrapped on the calling thread instead of through the fetch pool (ParallelFetcher.fetch_local),
    and one token's experts are computed straight off the token's row in the router's order
    (_forward_views_one) instead of through the chunk path's sort and gathers. Nothing it removes
    is supposed to carry information, so with it on and off:
      - every decode step's output is bit-identical, on gated (gate/up/down) and plain (fc1/fc2) layers;
      - the pool admits and evicts the same experts in the same order (same tables, same counters);
      - the bytes fetch_local returns are fetch()'s bytes, and it reads exactly the cold regions;
      - a real model's greedy reply is token-for-token the same.
    The outputs are also checked against arrays this file builds itself from the raw bytes (copies,
    not the engine's zero-copy views), so the two paths are not identical by sharing one mistake.

Run: .venv/bin/python tests/test_fast_decode.py
"""
import os
import sys
import tempfile

import mlx.core as mx
import mlx.nn as nn
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from bigrig_engine import fetch, stream  # noqa: E402

FAIL = []


def check(n, c, d=""):
    (print(f"  PASS  {n}") if c else (FAIL.append(n), print(f"  FAIL  {n}  {d}")))


E, K, D, GS, BITS = 64, 8, 512, 64, 4
Q = {"group_size": GS, "bits": BITS, "mode": "affine"}
PAGE = fetch._PAGE
rng = np.random.default_rng(7)


def _bytes(a: mx.array) -> bytes:
    if a.dtype == mx.bfloat16:
        a = a.view(mx.uint16)
    return np.array(a).tobytes()


def make_experts(projections, quantized=True, bias=False, order=None):
    """E experts' packed bytes, their spec, and each component's bytes.

    `quantized` False stores plain bfloat16 weights (the layer then computes `x @ w.T`); `bias`
    adds a per-projection bias vector (gpt-oss ships one); `order` lists the components in another
    order than weight/scales/biases/bias, as a spec is free to."""
    blobs, spec, parts = {}, {}, {}
    for e in range(E):
        chunks = []
        for p in projections:
            w = mx.array(rng.standard_normal((D, D)).astype(np.float32) * 0.05).astype(mx.bfloat16)
            if quantized:
                wq, s, b = mx.quantize(w, group_size=GS, bits=BITS)
                comps = {"weight": wq, "scales": s, "biases": b}
            else:
                comps = {"weight": w}
            if bias:
                comps["bias"] = mx.array(rng.standard_normal((D,)).astype(np.float32)).astype(mx.bfloat16)
            if order:
                comps = {c: comps[c] for c in order if c in comps}
            spec[p] = {c: {"shape": list(a.shape), "dtype": str(a.dtype).split(".")[-1], "nbytes": a.nbytes}
                       for c, a in comps.items()}
            for c, a in comps.items():
                parts[(e, p, c)] = _bytes(a)
                chunks.append(parts[(e, p, c)])
        blobs[e] = b"".join(chunks)
    return blobs, spec, parts


class _SG:
    def __init__(self, spec):
        for p in spec:
            setattr(self, p, {c: mx.zeros((E, *m["shape"]), dtype=getattr(mx, m["dtype"]))
                              for c, m in spec[p].items()})


def write_blob(blobs, path, layer=0):
    """The experts in one file, each region page-aligned and a whole number of pages (zero-copy)."""
    per = len(blobs[0])
    padded = -(-per // PAGE) * PAGE
    with open(path, "wb") as f:
        for e in range(E):
            f.write(blobs[e] + b"\0" * (padded - per))
    return {(layer, e): fetch.Region(e * padded, per) for e in range(E)}, padded


def act_gated(up, gate):
    return nn.silu(gate) * up


def act_plain(h):
    return nn.silu(h)


TMP = tempfile.mkdtemp(prefix="fast_decode_")
# Every shape of layer the engine serves: the two projection layouts, 4-bit and plain weights,
# with and without a bias, and a spec that lists its components in another order.
CASES = {}
for _name, _proj, _quant, _bias, _order in (
        ("gated", stream.PROJECTIONS, True, False, None),
        ("plain", stream.MLP_PROJECTIONS, True, False, None),
        ("gated + bias", stream.PROJECTIONS, True, True, None),
        ("plain + bias", stream.MLP_PROJECTIONS, True, True, None),
        ("gated, unquantized", stream.PROJECTIONS, False, False, None),
        ("gated, unquantized + bias", stream.PROJECTIONS, False, True, None),
        ("gated, components listed biases/bias/scales/weight", stream.PROJECTIONS, True, True,
         ("biases", "bias", "scales", "weight"))):
    _blobs, _spec, _parts = make_experts(_proj, _quant, _bias, _order)
    _path = os.path.join(TMP, _name.replace(" ", "_").replace("/", "_").replace(",", "").replace("+", "and") + ".experts")
    _layout, _ = write_blob(_blobs, _path)
    CASES[_name] = {"blobs": _blobs, "spec": _spec, "parts": _parts, "path": _path, "layout": _layout,
                    "gated": _proj == stream.PROJECTIONS, "quantized": _quant, "bias": _bias}
GATED = (CASES["gated"]["blobs"], CASES["gated"]["spec"], CASES["gated"]["parts"])
PLAIN = (CASES["plain"]["blobs"], CASES["plain"]["spec"], CASES["plain"]["parts"])
LAYOUT_G, LAYOUT_P = CASES["gated"]["layout"], CASES["plain"]["layout"]
assert CASES["gated"]["path"].endswith("gated.experts") and CASES["plain"]["path"].endswith("plain.experts")


def module(kind, capacity, fetcher=None, views=True):
    c = CASES[kind]
    f = fetcher or fetch.ParallelFetcher(fetch.WeightStore(c["path"], c["layout"]), threads=4)
    pool = stream.ExpertPool(0, _SG(c["spec"]), E, capacity, c["spec"], policy="lfuda", views=views).build()
    return stream.StreamingSwitchGLU(pool, f, act_gated if c["gated"] else act_plain,
                                     quantized=c["quantized"], qparams=Q if c["quantized"] else {}), f


# A decode run: 48 tokens, each routed to 8 distinct experts, with the overlap between
# consecutive tokens a real router shows (about 3 of 8 kept), so the pool hits, misses and evicts.
STEPS = []
prev = rng.choice(E, K, replace=False)
for _ in range(48):
    keep = rng.choice(prev, 3, replace=False)
    rest = rng.choice(np.setdiff1d(np.arange(E), keep), K - 3, replace=False)
    row = rng.permutation(np.concatenate([keep, rest]))
    x = mx.array(rng.standard_normal((1, 1, D)).astype(np.float32)).astype(mx.bfloat16)
    STEPS.append((x, row.reshape(1, 1, K)))
    prev = row


def run(kind, capacity, fast, steps=STEPS, views=True):
    stream.FAST_DECODE = fast
    try:
        m, f = module(kind, capacity, views=views)
        ys = []
        for x, idx in steps:
            y = m(x, mx.array(idx))
            mx.eval(y)
            ys.append(y)
        pool = m._pool
        state = {"g2s": pool.g2s.copy(), "s2g": pool.s2g.copy(), "hits": pool.hits, "misses": pool.misses,
                 "admits": pool.admits, "evicts": pool.evicts, "uses": pool.uses.copy(),
                 "resident": sorted(pool._views), "local": f.n_fetches, "chunked": m.chunked_calls,
                 "seen": set(f._seen)}
        f.close()
        return ys, state
    finally:
        stream.FAST_DECODE = True


def same_state(a, b):
    return all(np.array_equal(a[k], b[k]) for k in ("g2s", "s2g", "uses")) and \
        all(a[k] == b[k] for k in ("hits", "misses", "admits", "evicts", "resident"))


print("=" * 78)
print("1. FAST_DECODE ON == OFF: EVERY DECODE STEP BIT-IDENTICAL, SAME ADMISSIONS AND EVICTIONS")
print("=" * 78)
assert stream.FAST_DECODE, "default changed; this file tests the default"
for kind in CASES:
    # Views mode (the default pool) at three capacities; the slot pool (--slot-pool) at one, where
    # only the fetch changes and the stock gather computes the step.
    for cap, views in ((8, True), (13, True), (32, True), (13, False)):
        tag = f"{kind}, capacity {cap}" + ("" if views else ", slot pool")
        on, s_on = run(kind, cap, True, views=views)
        off, s_off = run(kind, cap, False, views=views)
        same = [mx.array_equal(a, b).item() for a, b in zip(on, off)]
        check(f"{tag}: {len(STEPS)} steps bit-identical on vs off", all(same),
              [i for i, s in enumerate(same) if not s][:5])
        check(f"{tag}: same pool tables, hits, misses, admits, evicts, use counts",
              same_state(s_on, s_off), {k: (s_on[k], s_off[k]) for k in ("hits", "misses", "admits", "evicts")})
        check(f"{tag}: the pool really cycled (misses and evictions happened)",
              s_on["misses"] > len(STEPS) and (cap >= 32 or s_on["evicts"] > 0), (s_on["misses"], s_on["evicts"]))

print("=" * 78)
print("2. AGAINST ARRAYS BUILT HERE FROM THE RAW BYTES (COPIES, NOT THE ENGINE'S VIEWS)")
print("=" * 78)


def arr(case, e, p, c):
    m = case["spec"][p][c]
    npdt = {"uint32": np.uint32, "bfloat16": np.uint16, "float16": np.float16, "float32": np.float32}[m["dtype"]]
    a = mx.array(np.frombuffer(case["parts"][(e, p, c)], dtype=npdt).copy().reshape(m["shape"]))
    return a.view(mx.bfloat16) if m["dtype"] == "bfloat16" else a


def by_hand(kind, x, row):
    """The layer's definition, written out: per expert, y = x @ W.T (+ bias) for each projection."""
    case = CASES[kind]

    def proj(e, p, inp):
        if case["quantized"]:
            y = mx.quantized_matmul(inp, arr(case, e, p, "weight"), arr(case, e, p, "scales"),
                                    arr(case, e, p, "biases"), transpose=True, group_size=GS, bits=BITS)
        else:
            y = inp @ arr(case, e, p, "weight").T
        return y + arr(case, e, p, "bias") if case["bias"] else y
    xf = x.reshape(1, D)
    outs = []
    for e in row:
        e = int(e)
        if case["gated"]:
            outs.append(proj(e, "down_proj", act_gated(proj(e, "up_proj", xf), proj(e, "gate_proj", xf))))
        else:
            outs.append(proj(e, "fc2", act_plain(proj(e, "fc1", xf))))
    return mx.concatenate(outs, axis=0).reshape(1, 1, K, D)


for kind in CASES:
    on, _ = run(kind, 8, True)
    ok = [mx.array_equal(on[i], by_hand(kind, STEPS[i][0], STEPS[i][1].reshape(-1))).item()
          for i in (0, 1, 17, 47)]
    check(f"{kind}: fast-path outputs equal per-expert products on independently built arrays", all(ok), ok)
    fin = all(bool(mx.isfinite(y.astype(mx.float32)).all().item()) for y in on)
    nz = float(mx.abs(on[0].astype(mx.float32)).max().item())
    check(f"{kind}: outputs are finite and not all zero", fin and nz > 0, (fin, nz))
# The oracle must be able to tell a bias from no bias, and W from W.T, or the checks above prove less
# than they say: the same step computed without the bias, and with the weight untransposed, differs.
x0, row0 = STEPS[0][0], STEPS[0][1].reshape(-1)
ref_b = by_hand("gated, unquantized + bias", x0, row0)
CASES["_nobias"] = dict(CASES["gated, unquantized + bias"], bias=False)
check("the by-hand answer changes when the bias is left out (the oracle sees the bias)",
      not mx.array_equal(ref_b, by_hand("_nobias", x0, row0)).item())
del CASES["_nobias"]
w0 = arr(CASES["gated, unquantized"], int(row0[0]), "up_proj", "weight")
check("x @ W.T differs from x @ W on these weights (the oracle sees the transpose)",
      not mx.array_equal(x0.reshape(1, D) @ w0.T, x0.reshape(1, D) @ w0).item())

print("=" * 78)
print("3. THE FAST PATH IS THE ONE RUNNING, AND ONLY WHERE IT APPLIES")
print("=" * 78)
calls = {"one": 0, "views": 0}
orig_one, orig_views = stream.StreamingSwitchGLU._forward_views_one, stream.StreamingSwitchGLU._forward_views


def spy_one(self, *a, **k):
    calls["one"] += 1
    return orig_one(self, *a, **k)


def spy_views(self, *a, **k):
    calls["views"] += 1
    return orig_views(self, *a, **k)


local_calls = {"n": 0, "none": 0}
orig_local = fetch.ParallelFetcher.fetch_local


def spy_local(self, keys):
    got = orig_local(self, keys)
    local_calls["n"] += 1
    local_calls["none"] += got is None
    return got


stream.StreamingSwitchGLU._forward_views_one, stream.StreamingSwitchGLU._forward_views = spy_one, spy_views
fetch.ParallelFetcher.fetch_local = spy_local
try:
    run("gated", 8, True)
    check("on: every one-row step takes _forward_views_one, none the chunk path",
          calls == {"one": len(STEPS), "views": 0}, calls)
    check("on: every step's missing experts come through fetch_local, and it never declines here",
          local_calls["n"] >= len(STEPS) - 1 and local_calls["none"] == 0, local_calls)
    calls.update(one=0, views=0)
    local_calls.update(n=0, none=0)
    run("gated", 8, False)
    check("off: every step takes the chunk path (the switch does something)",
          calls == {"one": 0, "views": len(STEPS)}, calls)
    check("off: fetch_local is never called (the fetch pool serves every expert, as before)",
          local_calls["n"] == 0, local_calls)
    # A row naming one expert twice (cache-aware rerouting can produce it) keeps the chunk path,
    # whose two-row product for that expert is what the engine computed before.
    dup = np.array([3, 5, 3, 9, 11, 13, 15, 17]).reshape(1, 1, K)
    xd = STEPS[0][0]
    calls.update(one=0, views=0)
    y_on, _ = run("gated", 8, True, steps=[(xd, dup)])
    c_on = dict(calls)
    y_off, _ = run("gated", 8, False, steps=[(xd, dup)])
    check("a row with a repeated expert stays on the chunk path and gives the same output",
          c_on == {"one": 0, "views": 1} and mx.array_equal(y_on[0], y_off[0]).item(), c_on)
    # A prompt chunk asks for most of the layer at once: served by the fetch pool as before.
    x40 = mx.array(rng.standard_normal((1, 40, D)).astype(np.float32)).astype(mx.bfloat16)
    i40 = np.stack([rng.choice(E, K, replace=False) for _ in range(40)]).reshape(1, 40, K)
    need = len(np.unique(i40))
    local_calls.update(n=0, none=0)
    y40_on, _ = run("gated", 8, True, steps=[(x40, i40)])
    l40 = dict(local_calls)
    y40_off, _ = run("gated", 8, False, steps=[(x40, i40)])
    check(f"a 40-row chunk needing {need} experts (> LOCAL_FETCH_MAX) never calls fetch_local, same output",
          need > stream.LOCAL_FETCH_MAX and l40["n"] == 0 and mx.array_equal(y40_on[0], y40_off[0]).item(), l40)
    # Layer-major prefill's last chunk can be one token; its routing must still be counted for
    # the re-warm that follows (holding() collects prefill_routed), on the one-row path too.
    counted = {}
    for fast in (True, False):
        stream.FAST_DECODE = fast
        m, f = module("gated", 8)
        with m.holding():
            mx.eval(m(STEPS[0][0], mx.array(STEPS[0][1])))
            counted[fast] = m.prefill_routed.copy()
        f.close()
    stream.FAST_DECODE = True
    want_counts = np.bincount(STEPS[0][1].reshape(-1), minlength=E)
    check("a one-token prefill chunk is counted in prefill_routed on the one-row path, as on the chunk path",
          np.array_equal(counted[True], want_counts) and np.array_equal(counted[False], want_counts),
          (int(counted[True].sum()), int(counted[False].sum())))
    # Two rows (a prompt chunk that fits the pool) is not decode: chunk path, unchanged.
    x2 = mx.array(rng.standard_normal((1, 2, D)).astype(np.float32)).astype(mx.bfloat16)
    i2 = np.stack([np.arange(K), np.arange(K)]).reshape(1, 2, K)
    calls.update(one=0, views=0)
    y2_on, _ = run("gated", 8, True, steps=[(x2, i2)])
    c2 = dict(calls)
    y2_off, _ = run("gated", 8, False, steps=[(x2, i2)])
    check("a two-row chunk stays on the chunk path and gives the same output",
          c2 == {"one": 0, "views": 1} and mx.array_equal(y2_on[0], y2_off[0]).item(), c2)
finally:
    stream.StreamingSwitchGLU._forward_views_one, stream.StreamingSwitchGLU._forward_views = orig_one, orig_views
    fetch.ParallelFetcher.fetch_local = orig_local

# ensure() records the experts it has under live views as resident, under THEIR layer's keys.
_, st_on = run("gated", 8, True)
used = {(0, int(e)) for _x, idx in STEPS for e in idx.reshape(-1)}
check("a decode run records exactly the (layer, expert) keys it used as resident", st_on["seen"] == used,
      (len(st_on["seen"]), len(used), sorted(st_on["seen"] - used)[:3]))
layout3, _ = write_blob(GATED[0], os.path.join(TMP, "layer3.experts"), layer=3)
f3s = fetch.ParallelFetcher(fetch.WeightStore(os.path.join(TMP, "layer3.experts"), layout3), threads=2)
pool3 = stream.ExpertPool(3, _SG(GATED[1]), E, 8, GATED[1], policy="lfuda", views=True).build()
m3 = stream.StreamingSwitchGLU(pool3, f3s, act_gated, quantized=True, qparams=Q)
y3 = [m3(x, mx.array(idx)) for x, idx in STEPS[:6]]
mx.eval(y3)
used3 = {(3, int(e)) for _x, idx in STEPS[:6] for e in idx.reshape(-1)}
ref3, _ = run("gated", 8, True, steps=STEPS[:6])
check("...for a layer that is not layer 0 too (keys carry the pool's own layer), with the same outputs",
      set(f3s._seen) == used3 and all(mx.array_equal(a, b).item() for a, b in zip(y3, ref3)),
      sorted(set(f3s._seen) - used3)[:3])
f3s.close()

print("=" * 78)
print("4. fetch_local: fetch()'S BYTES, NO POOL FOR WARM KEYS, EXACTLY THE COLD REGIONS READ")
print("=" * 78)
blobs = GATED[0]
store = fetch.WeightStore(os.path.join(TMP, "gated.experts"), LAYOUT_G)
f = fetch.ParallelFetcher(store, threads=4)
keys = [(0, e) for e in (5, 0, 63, 17, 5)]
loc = f.fetch_local(keys)
ref = f.fetch(keys)
check("same keys, same bytes as fetch(), and as the experts written (duplicates collapsed)",
      loc is not None and set(loc) == set(ref) == {(0, 5), (0, 0), (0, 63), (0, 17)}
      and all(bytes(loc[k]) == bytes(ref[k]) == blobs[k[1]] for k in loc))
check("the views are zero-copy slices of the store's map (memoryviews, page-aligned, right length)",
      all(isinstance(v, memoryview) and len(v) == len(blobs[0])
          and np.frombuffer(v, dtype=np.uint8).ctypes.data % PAGE == 0 for v in loc.values()))
check("an empty request is an empty answer", f.fetch_local([]) == {})

submitted, filled, probed = [], [], []
cold_set = set()
orig_submit, orig_fill, orig_res, orig_recent = f._pool.submit, store.fill, fetch.pages_resident, fetch.RECENT_S
map_base = store._map(store.path)[3]
f._pool.submit = lambda fn, *a, **k: (submitted.append(a), orig_submit(fn, *a, **k))[1]
store.fill = lambda fd, off, n: (filled.append((off, n)), orig_fill(fd, off, n))[1]
fetch.pages_resident = lambda addr, n: (probed.append(addr - map_base), (addr - map_base) not in cold_set)[1]
OFF = {e: LAYOUT_G[(0, e)].offset for e in range(E)}
try:
    fetch.RECENT_S = 0.0                                      # every key is asked about, every time
    n0, p0, r0 = f.n_fetches, store.warm_probes, store.warm_reads
    f.fetch_local([(0, e) for e in range(8)])
    check("all warm: nothing handed to the fetch pool, nothing read, 8 probes made and counted",
          submitted == [] and filled == [] and f.n_fetches == n0 + 1 and store.warm_probes == p0 + 8
          and store.warm_reads == r0 and sorted(probed) == [OFF[e] for e in range(8)], (submitted, filled))
    cold_set = {OFF[3]}
    got = f.fetch_local([(0, e) for e in range(8)])
    check("one cold key: read on the calling thread, exactly its region, bytes still right",
          submitted == [] and filled == [(OFF[3], LAYOUT_G[(0, 3)].length)]
          and bytes(got[(0, 3)]) == blobs[3], (submitted, filled))
    filled.clear()
    cold_set = {OFF[e] for e in (1, 4, 6)}
    r1 = store.warm_reads
    got = f.fetch_local([(0, e) for e in range(8)])
    want = sorted((OFF[e], LAYOUT_G[(0, e)].length) for e in (1, 4, 6))
    check("three cold keys: each read once on the fetch pool, exactly their regions, counted",
          len(submitted) == 3 and sorted(filled) == want and store.warm_reads == r1 + 3
          and all(bytes(got[(0, e)]) == blobs[e] for e in range(8)), (len(submitted), filled))
    filled.clear()
    store.fill = lambda fd, off, n: (_ for _ in ()).throw(OSError("read failed"))
    cold_set = {OFF[2]}
    got = f.fetch_local([(0, 2), (0, 9)])
    check("a failed read is not an error: the views are still the right bytes",
          got is not None and bytes(got[(0, 2)]) == blobs[2] and bytes(got[(0, 9)]) == blobs[9])
    store.fill = lambda fd, off, n: (filled.append((off, n)), orig_fill(fd, off, n))[1]

    # RECENT_S: a key known resident a moment ago is not asked about again; an older one is.
    fetch.RECENT_S = 3600.0
    cold_set = set()
    f._seen.clear()
    probed.clear(); filled.clear()
    s0 = f.recent_skips
    f.fetch_local([(0, e) for e in (20, 21, 22)])
    first = sorted(probed)
    probed.clear()
    f.fetch_local([(0, e) for e in (20, 21, 22, 23)])
    check("keys fetched a moment ago are not probed again; the new key is (1 probe, 3 skips counted)",
          first == [OFF[20], OFF[21], OFF[22]] and probed == [OFF[23]] and f.recent_skips == s0 + 3,
          (first, probed, f.recent_skips - s0))
    probed.clear()
    cold_set = {OFF[20]}                                      # it left the page cache, unseen
    got = f.fetch_local([(0, 20)])
    check("...which means a recent key that DID go cold is not read here (the view is still the right "
          "bytes; the GPU faults it in)", probed == [] and filled == [] and bytes(got[(0, 20)]) == blobs[20])
    f._seen[(0, 20)] -= 7200.0                                # last known resident two hours ago
    got = f.fetch_local([(0, 20)])
    check("a key last known resident longer ago than RECENT_S is probed, found cold and read",
          probed == [OFF[20]] and filled == [(OFF[20], LAYOUT_G[(0, 20)].length)]
          and bytes(got[(0, 20)]) == blobs[20], (probed, filled))
    probed.clear(); filled.clear(); cold_set = set()
    f.mark_resident([(0, 30), (0, 31)])
    f.fetch_local([(0, 30), (0, 31), (0, 32)])
    check("mark_resident (experts under a live view) spares their probe", probed == [OFF[32]], probed)
    probed.clear()
    cold_set = {OFF[40]}
    f.warm([(0, 40), (0, 41)])
    f._warm_pool.shutdown(wait=True); f._warm_pool = None     # let both warm jobs finish
    warm_probed = sorted(probed)
    probed.clear()
    f.fetch_local([(0, 40), (0, 41), (0, 42)])
    check("a key the warm pool just read or found resident is not probed again by fetch_local",
          warm_probed == [OFF[40], OFF[41]] and (OFF[40], LAYOUT_G[(0, 40)].length) in filled
          and probed == [OFF[42]], (warm_probed, filled, probed))
    fetch.RECENT_S = 0.0
    probed.clear()
    f.fetch_local([(0, 40), (0, 41)])
    check("RECENT_S = 0 asks about every key every time", sorted(probed) == [OFF[40], OFF[41]], probed)

    # What is recorded as resident: only what was probed resident or read. Not a key whose probe
    # was skipped (its time must not be renewed, or a busy key would never be asked about again),
    # and not a key whose read failed.
    fetch.RECENT_S = 3600.0
    f._seen.clear(); probed.clear(); filled.clear(); cold_set = set()
    f.fetch_local([(0, 55)])
    f._seen[(0, 55)] -= 1000.0                                # still recent, but an older time
    t_old = f._seen[(0, 55)]
    f.fetch_local([(0, 55)])
    check("a skipped key keeps its old time (a probe skipped is not a probe made)",
          probed == [OFF[55]] and f._seen[(0, 55)] == t_old, (probed, f._seen[(0, 55)] - t_old))
    store.fill = lambda fd, off, n: (_ for _ in ()).throw(OSError("read failed"))
    cold_set = {OFF[56], OFF[57], OFF[58]}
    f.fetch_local([(0, 56)])
    f.fetch_local([(0, 57), (0, 58), (0, 59)])
    check("keys whose read failed are not recorded as resident (one read here, two on the pool); "
          "the warm key beside them is",
          (0, 56) not in f._seen and (0, 57) not in f._seen and (0, 58) not in f._seen and (0, 59) in f._seen,
          sorted(k for k in f._seen if k[1] >= 56))
    store.fill = lambda fd, off, n: (filled.append((off, n)), orig_fill(fd, off, n))[1]
    orig_make = store.make_resident
    store.make_resident = lambda path, off, n: None           # the warm job could not check or read
    f.warm([(0, 60)])
    f._warm_pool.shutdown(wait=True); f._warm_pool = None
    store.make_resident = orig_make
    check("a warm job that could not make its region resident records nothing", (0, 60) not in f._seen)
    saved_warm_mode = fetch.WARM_MODE
    fetch.WARM_MODE = "on"                                    # BIGRIG_WARM_READ=1
    cold_set = set(); probed.clear()
    f.fetch_local([(0, 55), (0, 59)])
    fetch.WARM_MODE = saved_warm_mode
    check("BIGRIG_WARM_READ=1 asks about every key even when it was resident a moment ago",
          sorted(probed) == [OFF[55], OFF[59]], probed)
    real_submit = f._pool.submit
    f._pool.submit = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("cannot schedule new futures"))
    cold_set = {OFF[61], OFF[62]}; filled.clear(); f._seen.clear()
    got = f.fetch_local([(0, 61), (0, 62)])
    f._pool.submit = real_submit
    check("a fetch pool that is shutting down does not fail the request: cold keys are read here instead",
          got is not None and sorted(filled) == sorted((OFF[e], LAYOUT_G[(0, e)].length) for e in (61, 62))
          and bytes(got[(0, 61)]) == blobs[61], filled)
finally:
    f._pool.submit, store.fill, fetch.pages_resident, fetch.RECENT_S = orig_submit, orig_fill, orig_res, orig_recent

f.prefetch([(0, 50)])
check("with a prefetched read outstanding it declines (fetch() owns collecting it)",
      f.fetch_local([(0, 50), (0, 51)]) is None)
f.fetch([(0, 50)])
check("...and applies again once fetch() has collected it", f.fetch_local([(0, 51)]) is not None)
check("a key outside the layout declines rather than guessing", f.fetch_local([(0, 999)]) is None)
f.close()
check("a closed fetcher declines", f.fetch_local([(0, 1)]) is None)
f2 = fetch.ParallelFetcher(store, threads=2, mapped=False)
check("an unmapped fetcher declines", f2.fetch_local([(0, 1)]) is None)
f2.close()
seg_store = fetch.WeightStore("", {(0, 0): [fetch._Seg(os.path.join(TMP, "gated.experts"), 0, PAGE),
                                            fetch._Seg(os.path.join(TMP, "gated.experts"), PAGE, PAGE)]})
f3 = fetch.ParallelFetcher(seg_store, threads=2)
check("a store read from the model's own shards (several segments a key) declines",
      f3.fetch_local([(0, 0)]) is None)
f3.close()
saved_mode = fetch.WARM_MODE
fetch.WARM_MODE = "off"
try:
    f4 = fetch.ParallelFetcher(fetch.WeightStore(os.path.join(TMP, "gated.experts"), LAYOUT_G), threads=2)
    p0 = f4.store.warm_probes
    got = f4.fetch_local([(0, 1)])
    check("BIGRIG_WARM_READ=0 is honoured: no probe, no read, bytes right",
          f4.store.warm_probes == p0 and bytes(got[(0, 1)]) == blobs[1] and f4._seen == {})
    f4.close()
finally:
    fetch.WARM_MODE = saved_mode

print("=" * 78)
print("5. RESIDENCY PROBES: THE VECTORISED CHECK ANSWERS AS THE PAGE-BY-PAGE ONE DID")
print("=" * 78)
for raw, want in ((b"\x01" * 108, True), (b"\x01" * 107 + b"\x00", False), (b"\x00" + b"\x01" * 107, False),
                  (bytes([1, 3, 5, 7, 9, 0x81, 0xff]), True), (bytes([1, 3, 2]), False), (b"\x00" * 4, False)):
    check(f"mincore vector {raw[:4].hex()}... x{len(raw)} -> {want}",
          (not raw.translate(None, fetch._IN_CORE)) is want and all(b & 1 for b in raw) is want)


class _FakeLibc:
    """mincore over a made-up address space: which pages were asked about, and chosen answers."""
    asked, cold = [], set()

    @classmethod
    def mincore(cls, addr, length, vec):
        first, n = addr // PAGE, length // PAGE
        cls.asked.append((first, n))
        vec.raw = bytes(0 if (first + i) in cls.cold else 1 for i in range(n))
        return 0


real_libc = fetch._libc
fetch.pages_resident(0, 1)                                    # binds the real mincore first
real_libc = fetch._libc
fetch._libc = _FakeLibc
try:
    A = 1000 * PAGE
    check("108 warm pages: resident, asked about in one call covering all 108",
          fetch.pages_resident(A, 108 * PAGE) is True and _FakeLibc.asked == [(1000, 108)], _FakeLibc.asked)
    for cold in (1000, 1010, 1107):
        _FakeLibc.cold = {cold}
        check(f"one cold page anywhere (page {cold - 1000}) makes the region not resident",
              fetch.pages_resident(A, 108 * PAGE) is False)
    _FakeLibc.cold = {1108}
    check("a cold page just past the region does not", fetch.pages_resident(A, 108 * PAGE) is True)
    _FakeLibc.asked.clear()
    check("a region that starts mid-page is asked about from its page's start, through its last page",
          fetch.pages_resident(A + 100, PAGE) is True and _FakeLibc.asked == [(1000, 2)], _FakeLibc.asked)
finally:
    fetch._libc = real_libc
mm_store = fetch.WeightStore(os.path.join(TMP, "gated.experts"), LAYOUT_G)
base = mm_store._map(mm_store.path)[3]
os.pread(mm_store._map(mm_store.path)[2], PAGE * 4, 0)
check("on the real map, a region just read is reported resident", fetch.pages_resident(base, PAGE * 4) is True)
check("make_resident: False for a resident region, None for one past the end of the file",
      mm_store.make_resident(mm_store.path, 0, PAGE * 4) is False
      and mm_store.make_resident(mm_store.path, mm_store.size, PAGE) is None)
mm_store.close_maps()

print("=" * 78)
print("6. END TO END: A REAL MODEL'S GREEDY REPLY IS TOKEN-FOR-TOKEN THE SAME ON AND OFF (OLMoE)")
print("=" * 78)
OLMOE = os.path.join(ROOT, "models", "OLMoE-1B-7B-0125-4bit")
BLOB = os.path.join(ROOT, "data", "blobs", "OLMoE-1B-7B-0125-4bit.experts")
SKIPPED = []
if not (os.path.isdir(OLMOE) and os.path.exists(BLOB)):
    print("  SKIPPED - needs models/OLMoE-1B-7B-0125-4bit and its packed blob.")
    SKIPPED.append("section 6 (a real model's reply, on and off)")
else:
    import shutil
    from bigrig_engine import session as _session
    tmp_home = tempfile.mkdtemp(prefix="fast_decode_home_")
    os.makedirs(os.path.join(tmp_home, "data"))
    os.symlink(os.path.join(ROOT, "data", "blobs"), os.path.join(tmp_home, "data", "blobs"))
    saved_home = os.environ.get("BIGRIG_HOME")
    os.environ["BIGRIG_HOME"] = tmp_home
    try:
        s = _session.Session(OLMOE, budget_gb=4.9, verbose=False, persist=False, monitor=False, remember=False)
        st = s.stats()
        prompts = ["Explain how a heat pump moves warmth from cold outdoor air into a house.",
                   "Write a Python function that returns the n-th Fibonacci number iteratively."]

        def reply(p):
            ids = []
            for _chunk, info in s.stream_text([{"role": "user", "content": p}], max_tokens=48,
                                              temperature=0.0, seed=0):
                g = int(info.get("generation_tokens") or 0)
                if g > len(ids) and info.get("token") not in (None, -1):
                    ids.append(int(info["token"]))
            return ids

        seen = {"one": 0, "local": 0}
        o1, o2 = stream.StreamingSwitchGLU._forward_views_one, fetch.ParallelFetcher.fetch_local
        stream.StreamingSwitchGLU._forward_views_one = \
            lambda self, *a, **k: (seen.__setitem__("one", seen["one"] + 1), o1(self, *a, **k))[1]
        fetch.ParallelFetcher.fetch_local = \
            lambda self, keys: (seen.__setitem__("local", seen["local"] + 1), o2(self, keys))[1]
        try:
            runs = []
            for fast in (True, False, True):
                stream.FAST_DECODE = fast
                seen.update(one=0, local=0)
                runs.append(([reply(p) for p in prompts], dict(seen)))
        finally:
            stream.FAST_DECODE = True
            stream.StreamingSwitchGLU._forward_views_one, fetch.ParallelFetcher.fetch_local = o1, o2
        on1, off1, on2 = runs
        check(f"stream mode at capacity {st.get('capacity')}, replies of {[len(r) for r in on1[0]]} tokens",
              st["mode"] == "stream" and all(len(r) >= 16 for r in on1[0]), st["mode"])
        check("on == off, token for token, both prompts", on1[0] == off1[0],
              [next((i for i, (a, b) in enumerate(zip(x, y)) if a != b), None) for x, y in zip(on1[0], off1[0])])
        check("on again == on (the comparison is not order-dependent)", on1[0] == on2[0])
        check("on: the one-row path and the local fetch ran; off: neither did",
              on1[1]["one"] > 100 and on1[1]["local"] > 100 and off1[1] == {"one": 0, "local": 0},
              (on1[1], off1[1]))
        s.close()
        del s
    finally:
        stream.FAST_DECODE = True
        if saved_home is None:
            os.environ.pop("BIGRIG_HOME", None)
        else:
            os.environ["BIGRIG_HOME"] = saved_home
        shutil.rmtree(tmp_home, ignore_errors=True)

import shutil as _sh  # noqa: E402
_sh.rmtree(TMP, ignore_errors=True)
print("\n" + "=" * 78)
if SKIPPED:
    print(f"SKIPPED: {', '.join(SKIPPED)} (model not on disk)")
print(f"{'ALL TESTS PASSED' if not FAIL else str(len(FAIL)) + ' FAILURES: ' + ', '.join(FAIL)}"
      + (" (with skips above)" if SKIPPED and not FAIL else ""))
print("=" * 78)
sys.exit(1 if FAIL else 0)
