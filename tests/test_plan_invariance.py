"""Same answers at every memory budget: a MoE layer's output must not depend on the expert plan.

WHY THIS FILE EXISTS
    BigRig turns a memory budget into an expert plan: how many experts each streamed layer keeps
    resident (its capacity) and which layers are held whole. Before SAME_ANSWERS, a prompt chunk's
    arithmetic depended on that plan: a chunk routed to more experts than the capacity took a
    sorted gather cut at the capacity, and a whole layer took one sorted gather over all its
    experts, while a chunk that fit took one quantized_matmul per expert. Those kernels round
    differently, so the same prompt gave a different reply at a different budget (measured
    2026-11-12 on Qwen3.6-35B-A3B-4bit: 11 slots + 3 whole layers left the 8-slot reply at about
    token 15). With SAME_ANSWERS on, every multi-row call is one quantized_matmul per expert over
    the rows routed to it, whatever the plan.

    Checked here on real StreamingSwitchGLU modules over 4-bit quantized experts (bfloat16 scales,
    as the product model): streamed layers at three capacities and one whole layer, each prompt
    chunk size from one row (decode) to more rows than any capacity, outputs compared bit for bit.
    With SAME_ANSWERS off the plans must disagree somewhere, so the comparison can fail; and the
    outputs must match a by-hand dequantized computation, so they are not identical by all being
    wrong the same way.

Run: .venv/bin/python tests/test_plan_invariance.py
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


# 64 experts, top-8, hidden 512: small enough to run in a second, large enough that the old
# paths' kernels round differently (at 16 experts, top-4, hidden 128-512 every plan agreed even
# without SAME_ANSWERS, so the comparison could not fail).
E, K, D, GS, BITS = 64, 8, 512, 64, 4
Q = {"group_size": GS, "bits": BITS, "mode": "affine"}
rng = np.random.default_rng(0)


def _bytes(a: mx.array) -> bytes:
    """An array's raw bytes (bfloat16 has no numpy dtype, so go through its 16-bit pattern)."""
    if a.dtype == mx.bfloat16:
        a = a.view(mx.uint16)
    return np.array(a).tobytes()


WEIGHTS, BLOBS, SPEC = {}, {}, {}
for e in range(E):
    parts = []
    for p in stream.PROJECTIONS:
        w = mx.array(rng.standard_normal((D, D)).astype(np.float32) * 0.05).astype(mx.bfloat16)
        wq, s, b = mx.quantize(w, group_size=GS, bits=BITS)
        WEIGHTS[(e, p)] = mx.dequantize(wq, s, b, group_size=GS, bits=BITS)
        comps = {"weight": wq, "scales": s, "biases": b}
        SPEC[p] = {c: {"shape": list(a.shape), "dtype": str(a.dtype).split(".")[-1], "nbytes": a.nbytes}
                   for c, a in comps.items()}
        parts += [_bytes(a) for a in comps.values()]
    BLOBS[e] = b"".join(parts)


class _SG:
    """The model's own expert tensors, which ExpertPool.build() drops."""

    def __init__(self):
        for p in stream.PROJECTIONS:
            setattr(self, p, {c: mx.zeros((E, *m["shape"]), dtype=getattr(mx, m["dtype"]))
                              for c, m in SPEC[p].items()})


class _Fetcher:
    def fetch(self, keys):
        return {k: memoryview(BLOBS[k[1]]) for k in keys}

    def prefetch(self, keys):
        pass

    def drop(self, keys):
        pass


def act(up, gate):
    return nn.silu(gate) * up


def streamed(capacity):
    pool = stream.ExpertPool(0, _SG(), E, capacity, SPEC, policy="lfuda", views=True).build()
    return stream.StreamingSwitchGLU(pool, _Fetcher(), act, quantized=True, qparams=Q)


def whole():
    pool = stream.ExpertPool(0, _SG(), E, E, SPEC, policy="lfuda", views=False).build()
    for e in range(E):                       # as StreamHandle.warm fills a whole layer
        pool.touch([e], [0])
        pool.admit(e, memoryview(BLOBS[e]))
    m = stream.StreamingSwitchGLU(pool, _Fetcher(), act, quantized=True, qparams=Q)
    m.refresh()
    return m


# Prompt chunks: one row (decode); 4 rows, more experts than capacity 8 or 16 hold but fewer than
# 32 (chunked at the small capacities, not at 32); 31 rows, under VIEWS_MIN_TOKENS and more experts
# than any streamed capacity (the short-chunk path that used to cut its gather at the capacity);
# 40 rows (the long-chunk views path).
CHUNKS = []
for n in (1, 4, 31, 40):
    x = mx.array(rng.standard_normal((1, n, D)).astype(np.float32)).astype(mx.bfloat16)
    idx = np.stack([rng.choice(E, K, replace=False) for _ in range(n)]).reshape(1, n, K)
    CHUNKS.append((x, idx))


def outputs(plan_factory):
    """Every chunk through a fresh module of one plan, in order, as a fresh session would."""
    m = plan_factory()
    ys = [m(x, mx.array(idx)) for x, idx in CHUNKS]
    mx.eval(ys)
    return m, ys


PLANS = {"streamed, capacity 8 (= top-k)": lambda: streamed(8),
         "streamed, capacity 16": lambda: streamed(16),
         "streamed, capacity 32": lambda: streamed(32),
         "whole layer (capacity 64 = every expert)": whole}

print("=" * 78)
print("1. WITH SAME_ANSWERS: EVERY PLAN, EVERY CHUNK, BIT-IDENTICAL")
print("=" * 78)
assert stream.SAME_ANSWERS and stream.VIEWS_PREFILL, "defaults changed; this file tests the defaults"
res = {name: outputs(f) for name, f in PLANS.items()}
check("the whole layer runs on the sync-free path it would take in a real plan",
      res["whole layer (capacity 64 = every expert)"][0]._sync_free)
u31 = len(np.unique(CHUNKS[2][1]))
check("the 31-row chunk is short and routes to more experts than capacity 32 holds (the old cut path)",
      CHUNKS[2][0].shape[1] < stream.VIEWS_MIN_TOKENS and u31 > 32, u31)
ref_name = next(iter(PLANS))
for name in list(PLANS)[1:]:
    same = [mx.array_equal(a, b).item() for a, b in zip(res[ref_name][1], res[name][1])]
    check(f"{name} == {ref_name}, chunk sizes {[c[0].shape[1] for c in CHUNKS]}", all(same), same)

# By hand: dequantized weights, every (token, expert) pair, in float32.
x, idx = CHUNKS[2]
xn = np.array(x.astype(mx.float32))[0]
wn = {k: np.array(v.astype(mx.float32)) for k, v in WEIGHTS.items()}


def silu(g):
    return g / (1 + np.exp(-g))


ref = np.stack([np.stack([(silu(xn[t] @ wn[(e, "gate_proj")].T) * (xn[t] @ wn[(e, "up_proj")].T))
                          @ wn[(e, "down_proj")].T for e in idx[0, t]]) for t in range(xn.shape[0])])
got = np.array(res[ref_name][1][2].astype(mx.float32))[0]
rel = float(np.abs(got - ref).max() / np.abs(ref).max())
check("the shared answer matches a by-hand dequantized computation (max relative |d| < 2%, bfloat16)",
      rel < 0.02, rel)

print("=" * 78)
print("2. WITHOUT IT (BIGRIG_SAME_ANSWERS=0) THE PLANS DISAGREE: THE COMPARISON CAN FAIL")
print("=" * 78)
stream.SAME_ANSWERS = False
try:
    off = {name: outputs(f)[1] for name, f in PLANS.items()}
finally:
    stream.SAME_ANSWERS = True
differs = {name: [not mx.array_equal(a, b).item() for a, b in zip(off[ref_name], off[name])]
           for name in list(PLANS)[1:]}
check("some plan's prompt chunk differs from capacity 8's (the bug this fixes is reproduced)",
      any(any(d) for d in differs.values()), differs)
check("...but never on a one-row (decode) chunk, which was already plan-invariant",
      not any(d[0] for d in differs.values()), differs)

print("=" * 78)
print("3. OUTSIDE STREAM MODE A WHOLE LAYER KEEPS THE OLD GATHER (attach(same_answers=False))")
print("=" * 78)
# Compress mode holds every layer whole on requantised weights: nothing streamed to agree with, so
# the per-expert path would only cost time. Session passes same_answers only in stream mode.
m_off = whole()
m_off.same_answers = False
y_off = [m_off(x, mx.array(idx)) for x, idx in CHUNKS]
mx.eval(y_off)
whole_name = "whole layer (capacity 64 = every expert)"
check("a whole layer with same_answers off computes exactly the old gather",
      all(mx.array_equal(a, b).item() for a, b in zip(y_off, off[whole_name])))
check("...which is not the shared per-expert answer (the switch does something)",
      not all(mx.array_equal(a, b).item() for a, b in zip(y_off, res[ref_name][1])))
m_soff = streamed(8)
m_soff.same_answers = False
y_soff = [m_soff(x, mx.array(idx)) for x, idx in CHUNKS]
mx.eval(y_soff)
check("a streamed layer with same_answers off also keeps the old paths (the capacity-cut gather)",
      all(mx.array_equal(a, b).item() for a, b in zip(y_soff, off[ref_name])))

print("=" * 78)
print("4. ONE ROW PER EXPERT: THE GATHER EQUALS PER-EXPERT quantized_matmul AT REAL SHAPES")
print("=" * 78)
# Decode relies on this: a whole layer's sync-free gather and a streamed layer's per-expert matmul
# agree bit for bit at one row per expert. Checked here at Qwen3.6's projection shapes (2048 -> 512,
# 512 -> 2048) and OLMoE's (2048 -> 1024, 1024 -> 2048), so an MLX upgrade that breaks it fails here.
for d_in, d_out in ((2048, 512), (512, 2048), (2048, 1024), (1024, 2048)):
    ne, rng2 = 4, np.random.default_rng(d_in + d_out)
    w = mx.array(rng2.standard_normal((ne, d_out, d_in)).astype(np.float32) * 0.02).astype(mx.bfloat16)
    wq, sc, bi = mx.quantize(w, group_size=GS, bits=BITS)
    x1 = mx.array(rng2.standard_normal((ne, 1, 1, d_in)).astype(np.float32)).astype(mx.bfloat16)
    got = mx.gather_qmm(x1, wq, sc, bi, rhs_indices=mx.arange(ne).reshape(ne, 1), transpose=True,
                        group_size=GS, bits=BITS)
    ref1 = mx.stack([mx.quantized_matmul(x1[e, 0], wq[e], sc[e], bi[e], transpose=True,
                                         group_size=GS, bits=BITS) for e in range(ne)])
    mx.eval(got, ref1)
    check(f"one row per expert, {d_in} -> {d_out}: gather == per-expert, bit for bit",
          mx.array_equal(got.reshape(ref1.shape), ref1).item())

print("=" * 78)
print("5. THE SESSION PASSES same_answers ONLY IN STREAM MODE")
print("=" * 78)
OLMOE = os.path.join(ROOT, "models", "OLMoE-1B-7B-0125-4bit")
if not (os.path.isdir(OLMOE) and os.path.exists(os.path.join(ROOT, "data", "blobs", "OLMoE-1B-7B-0125-4bit.experts"))):
    print("  SKIPPED - needs models/OLMoE-1B-7B-0125-4bit and its packed blob (no weights are loaded).")
else:
    from bigrig_engine import precision, session as _session

    class _Stop(Exception):
        pass

    seen = []

    def _no_load(*a, **k):
        seen.append(k.get("same_answers"))
        raise _Stop()

    saved = stream.load_streaming, precision.ensure_compressed
    stream.load_streaming = _no_load
    precision.ensure_compressed = lambda blob, *a, **k: blob     # no requantised copy is written
    try:
        for budget, pref in ((4.9, None), (6.0, "compress")):
            try:
                _session.Session(OLMOE, budget_gb=budget, preference=pref, verbose=False, persist=False,
                                 monitor=False, remember=False)   # never record a compress consent
            except _Stop:
                pass
    finally:
        stream.load_streaming, precision.ensure_compressed = saved
    check("stream mode (OLMoE at 4.9 GB) attaches with same_answers on, compress mode (6.0 GB) with it off",
          seen == [True, False], seen)

handed = []
saved = stream.load_lenient, stream.attach
stream.load_lenient = lambda *a, **k: (None, None)
stream.attach = lambda *a, **k: (handed.append(k.get("same_answers")), (_ for _ in ()).throw(KeyError("stop")))[1]
try:
    for flag in (True, False):
        try:
            stream.load_streaming("model", "blob", same_answers=flag)
        except KeyError:
            pass
finally:
    stream.load_lenient, stream.attach = saved
check("load_streaming hands same_answers to attach", handed == [True, False], handed)

print("=" * 78)
print("6. END TO END: THE SETTING REACHES EVERY LAYER AND THE SESSION'S STATS (real OLMoE session)")
print("=" * 78)
# A real stream-mode Session (OLMoE at 4.9 GB, no expert read: warm leaves a streamed pool empty),
# under a throwaway engine home so it writes no record into any checkout. With SAME_ANSWERS off
# (as BIGRIG_SAME_ANSWERS=0 sets it) every layer and the stats must say so.
if not (os.path.isdir(OLMOE) and os.path.exists(os.path.join(ROOT, "data", "blobs", "OLMoE-1B-7B-0125-4bit.experts"))):
    print("  SKIPPED - needs models/OLMoE-1B-7B-0125-4bit and its packed blob.")
else:
    import shutil
    import tempfile
    from bigrig_engine import session as _session
    tmp_home = tempfile.mkdtemp(prefix="plan_invariance_home_")
    os.makedirs(os.path.join(tmp_home, "data"))
    os.symlink(os.path.join(ROOT, "data", "blobs"), os.path.join(tmp_home, "data", "blobs"))
    saved_home = os.environ.get("BIGRIG_HOME")
    os.environ["BIGRIG_HOME"] = tmp_home
    try:
        for flag in (True, False):
            stream.SAME_ANSWERS = flag
            s = _session.Session(OLMOE, budget_gb=4.9, verbose=False, persist=False, monitor=False,
                                 remember=False)
            st = s.stats()
            layers = {m.same_answers for m in s.handle.mods}
            check(f"SAME_ANSWERS={flag}: stream mode, every layer and Session.stats() report {flag}",
                  st["mode"] == "stream" and layers == {flag} and st["same_answers"] is flag,
                  (st["mode"], layers, st.get("same_answers")))
            if flag:
                s.handle.mods[-1].same_answers = False
                check("...and one layer without it makes the handle report False (all layers, not any)",
                      s.handle.stats()["same_answers"] is False)
            s.close()
            del s
        stream.SAME_ANSWERS = True
        _m, _t, h = stream.load_streaming(OLMOE, capacity=8, warm="lazy", verbose=False, same_answers=False)
        check("load_streaming(same_answers=False), as compress mode loads: every layer and the handle report False",
              {m.same_answers for m in h.mods} == {False} and h.stats()["same_answers"] is False)
        h.close()
        del _m, _t, h
    finally:
        stream.SAME_ANSWERS = True
        if saved_home is None:
            os.environ.pop("BIGRIG_HOME", None)
        else:
            os.environ["BIGRIG_HOME"] = saved_home
        shutil.rmtree(tmp_home, ignore_errors=True)

print("\n" + "=" * 78)
print(f"{'ALL TESTS PASSED' if not FAIL else str(len(FAIL)) + ' FAILURES: ' + ', '.join(FAIL)}")
print("=" * 78)
sys.exit(1 if FAIL else 0)
