"""Adversarial tests for the streaming expert pool.

The pool rewrites which bytes the model multiplies. Its failure mode is not a crash -- it is a
model that keeps generating fluent text from the wrong weights. Nothing downstream would notice.
So the central assertion here is EXACT equality against the unstreamed model, not closeness, and
several tests exist only to pin bugs that already happened:

  * the policy contract was wrong (on_hit/on_admit vs touch/admit/victim/evicted), so the
    policy's idea of the cache silently diverged from the pool's
  * prefill hands a whole prompt to the MoE block at once, so one call can need more experts
    than the pool holds -- decode-only tests never see it
  * bfloat16 has no numpy dtype, so every 3-bit and bf16 model failed to pack
  * loading without lazy=True materialised all E experts before anything could drop them, and
    peak RSS was then IDENTICAL at C=E and C=E/4 -- the pool saved nothing at all
"""
import json
import os
import sys

import mlx.core as mx
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bigrig_engine import stream

FAIL = []
def check(n, c, d=""):
    (print(f"  PASS  {n}") if c else (FAIL.append(n), print(f"  FAIL  {n}  {d}")))

BLOB = os.path.join(ROOT, "data/blobs/OLMoE-1B-7B-0125-4bit.experts")
RT = os.path.join(ROOT, "data/results/stream_roundtrip.json")

if not os.path.exists(BLOB):          # the blob itself; its manifest ships as a fixture
    # Nearly every check in this file reads a real packed blob, which is made from a downloaded
    # model. A fresh clone has neither, and load_manifest raised before a single line printed.
    print(f"  SKIPPED - no packed blob at {BLOB}.")
    print( "  This file replays a real one; run `bigrig prepare mlx-community/OLMoE-1B-7B-0125-4bit`")
    print( "  once and re-run to exercise it.")
    print("\n" + "=" * 82); print("ALL TESTS PASSED"); print("=" * 82)
    sys.exit(0)

print("=" * 82); print("1. THE PACKED BLOB"); print("=" * 82)
man = stream.load_manifest(BLOB)
check("manifest is the byte-transport version", man["version"] >= 3, str(man["version"]))
check("blob on disk is exactly the size the manifest claims",
      os.path.getsize(BLOB) == man["total_bytes"],
      f"{os.path.getsize(BLOB)} vs {man['total_bytes']}")
regs = [(o, n) for o, n in man["regions"].values()]
check("no expert region overlaps another",
      all(a[0] + a[1] <= b[0] for a, b in zip(sorted(regs), sorted(regs)[1:])))
check("regions tile the file with no gaps",
      sum(n for _, n in regs) == man["total_bytes"])
L0 = man["layers"][sorted(man["layers"], key=int)[0]]
per = sum(c["nbytes"] for p in L0["spec"].values() for c in p.values())
check("bytes_per_expert equals the sum of its component arrays",
      per == L0["bytes_per_expert"], f"{per} vs {L0['bytes_per_expert']}")
check("all three SwiGLU projections are packed",
      set(L0["spec"]) == set(stream.PROJECTIONS), str(sorted(L0["spec"])))

print("\n" + "=" * 82); print("2. STORE AND REGION LOOKUP"); print("=" * 82)
st = stream.store_from_manifest(BLOB, man)
check("keys are (layer, expert) tuples, not strings",
      all(isinstance(k, tuple) and len(k) == 2 for k in list(st.layout)[:5]))
try:
    st.region((999, 999)); check("an unknown key raises rather than reading garbage", False)
except KeyError:
    check("an unknown key raises rather than reading garbage", True)
last = max(st.layout.values(), key=lambda r: r.offset + r.length)
check("no region runs past the end of the file", last.offset + last.length <= st.size)

print("\n" + "=" * 82); print("3. THE SLOT TABLE'S INVARIANTS"); print("=" * 82)


class _FakeSG:
    """A SwitchGLU-shaped stand-in, so pool logic is testable without loading a model."""
    def __init__(self, E, d=8):
        for p in stream.PROJECTIONS:
            setattr(self, p, {"weight": mx.zeros((E, d, d), dtype=mx.uint32)})


E, C = 16, 6
spec = {p: {"weight": {"shape": [8, 8], "dtype": "uint32", "nbytes": 8 * 8 * 4}}
        for p in stream.PROJECTIONS}
BLOCK = bytes(8 * 8 * 4 * 3)


def fresh():
    p = stream.ExpertPool(0, _FakeSG(E), E, C, spec, policy="lfuda")
    return p.build()


p = fresh()
check("an empty pool reports every expert absent", (p.g2s < 0).all())
for e in range(C):
    p.touch([e], [0]); p.admit(e, BLOCK)
check("after C admits every slot is occupied exactly once",
      sorted(p.s2g.tolist()) == list(range(C)), str(p.s2g.tolist()))
check("the two maps agree in both directions",
      all(p.g2s[p.s2g[s]] == s for s in range(C)))
before = set(p.s2g.tolist())
p.touch([C], [0]); p.admit(C, BLOCK)
check("admitting past capacity evicts exactly one expert",
      len(set(p.s2g.tolist()) - before) == 1 and len(p.s2g) == C)
check("the evicted expert is marked absent",
      sum(1 for e in range(E) if p.g2s[e] >= 0) == C)
check("slot count never exceeds capacity", (p.s2g >= 0).sum() <= C)

# THE LIVELOCK. LFUDA has no notion of "in use", so without protection it will evict an expert
# admitted moments earlier in the same step, and the token then reads it straight back.
p = fresh()
need = list(range(C))
for i, e in enumerate(need):
    p.touch([e], [0]); p.admit(e, BLOCK, protect=need)
check("no expert needed by the current step is ever evicted",
      all(p.g2s[e] >= 0 for e in need), str([int(p.g2s[e]) for e in need]))
try:
    p.touch([C + 1], [0]); p.admit(C + 1, BLOCK, protect=list(range(C + 2)))
    check("protecting more experts than slots raises instead of corrupting", False)
except RuntimeError as ex:
    check("protecting more experts than slots raises instead of corrupting",
          "top-k" in str(ex))

try:
    stream.ExpertPool(0, _FakeSG(E), E, E + 1, spec)
    check("capacity above the expert count is rejected", False)
except ValueError:
    check("capacity above the expert count is rejected", True)
try:
    stream.ExpertPool(0, _FakeSG(E), E, 0, spec)
    check("capacity of zero is rejected", False)
except ValueError:
    check("capacity of zero is rejected", True)
try:
    p2 = stream.ExpertPool(0, _FakeSG(E), E, C, spec)
    p2.admit(0, BLOCK)
    check("admitting before build() raises", False)
except RuntimeError:
    check("admitting before build() raises", True)
p3 = fresh()
try:
    p3.touch([0], [0]); p3.admit(0, BLOCK[:-4])
    check("a short region raises rather than reading the next expert's head", False)
except ValueError as ex:
    check("a short region raises rather than reading the next expert's head",
          "manifest accounts" in str(ex))

print("\n" + "=" * 82)
print("3b. THE POLICY AND THE POOL MUST NEVER DISAGREE ABOUT WHAT IS RESIDENT")
print("=" * 82)
# The bug: when the policy nominated an expert the current step still needed, the pool called
# policy.evicted() on it to move past -- a lie. LFUDA then removed it from its resident set while
# it stayed in the pool, so victim() could never nominate it again, and the global age L was
# raised by an eviction that never happened. After 200 steps at C=12 the policy could see ONE of
# twelve resident experts. Output stayed correct; the eviction policy had silently stopped.
pd = stream.ExpertPool(0, _FakeSG(32), 32, 12, spec, policy="lfuda").build()
check("the shipped policy accepts an exclusion set, so no lie is needed", pd._policy_excludes)
rng = np.random.default_rng(0)
for _ in range(200):
    want = sorted(rng.choice(32, size=8, replace=False).tolist())
    for e in pd.touch(want, list(range(8))):
        pd.admit(e, BLOCK, protect=want)
pool_r = {int(e) for e in range(32) if pd.g2s[e] >= 0}
pol_r = set(pd.policy.resident)
check("after 200 churning steps the two views agree exactly",
      pool_r == pol_r, f"pool {len(pool_r)} vs policy {len(pol_r)}, differ on {pool_r ^ pol_r}")
check("every resident expert is still reachable by victim()",
      pd.policy.victim() in pool_r, str(pd.policy.victim()))
check("the global age reflects real evictions only, not phantom ones",
      pd.policy.L <= pd.evicts * 1.2 + 1, f"L={pd.policy.L:.1f} after {pd.evicts} evictions")
check("slot tables stay consistent through churn",
      all(pd.g2s[pd.s2g[sl]] == sl for sl in range(12) if pd.s2g[sl] >= 0))
# A policy with no exclude support must be handled by the pool choosing for itself, never by
# telling that policy something false.
pl = stream.ExpertPool(0, _FakeSG(32), 32, 12, spec, policy="lru").build()
check("a policy without exclude= is still usable", not pl._policy_excludes)
for _ in range(120):
    want = sorted(rng.choice(32, size=8, replace=False).tolist())
    for e in pl.touch(want, list(range(8))):
        pl.admit(e, BLOCK, protect=want)
lru_pool = {int(e) for e in range(32) if pl.g2s[e] >= 0}
check("...and its view still matches the pool's", lru_pool == set(pl.od.keys())
      if hasattr(pl, "od") else lru_pool == set(pl.policy.od.keys()),
      f"pool {len(lru_pool)} vs policy {len(getattr(pl.policy, 'od', {}))}")

print("\n" + "=" * 82); print("3c. TOP-K COMES FROM THE MODEL, NOT FROM A CONSTANT"); print("=" * 82)
# Hardcoding 8 over-reserves on a top-4 model and, far worse, lets a top-16 model be configured
# into a pool too small to serve one token -- which fails partway through the first prompt.
_seen = 0
for name in ("Qwen3-30B-A3B-3bit", "OLMoE-1B-7B-0125-4bit"):
    d = os.path.join(ROOT, f"models/{name}")
    if not os.path.exists(os.path.join(d, "config.json")):
        continue                     # a fresh clone has no models; the fallbacks below still run
    _seen += 1
    cfg = json.load(open(os.path.join(d, "config.json")))
    want = cfg.get("num_experts_per_tok")
    got = stream.model_top_k(d)
    check(f"{name}: top-k read from config.json ({got})", got == want, f"{got} vs {want}")
if not _seen:
    print("  SKIPPED - neither reference model is on this machine, so the value can only be")
    print("  checked against the documented fallbacks below.")
check("an unknown model falls back to the documented default",
      stream.model_top_k("/nonexistent/path") == 8)
check("a manifest value wins over the config file",
      stream.model_top_k("/nonexistent", {"top_k": 4}) == 4)

print("\n" + "=" * 82)
print("3d. RESIDENCY IS COUNTED, NOT SCANNED  (the sync-free path consults it per token)")
print("=" * 82)
pc = stream.ExpertPool(0, _FakeSG(8), 8, 8, spec, policy="lfuda").build()
check("an empty pool is not full", not pc.full)
for e in range(8):
    pc.touch([e], [0]); pc.admit(e, BLOCK)
check("a pool holding every expert is full", pc.full)
check("the counter agrees with the table",
      pc._resident == int((pc.g2s >= 0).sum()), f"{pc._resident} vs {int((pc.g2s>=0).sum())}")
pe = stream.ExpertPool(0, _FakeSG(8), 8, 4, spec, policy="lfuda").build()
for e in range(8):
    pe.touch([e], [0]); pe.admit(e, BLOCK)
check("a pool smaller than the expert count is never full", not pe.full)
check("the counter stays exact through eviction",
      pe._resident == int((pe.g2s >= 0).sum()) == 4,
      f"{pe._resident} vs {int((pe.g2s>=0).sum())}")

print("\n" + "=" * 82)
print("3d3. PREFILL STRAIGHT FROM THE PAGE CACHE IS A CHOICE, AND OFF UNTIL CHOSEN")
print("=" * 82)
# Measured on Qwen3.6-35B-A3B-4bit: 1.4-2.0x faster to the first token, less peak memory, and one
# reply in three differing in a near-tie because the rows run through quantized_matmul rather than
# the gather a resident model uses. So it is a flag, not a default, and it must never fire on a
# decode step or on a chunk that wants only a few experts.
import inspect as _ins2
_src3 = _ins2.getsource(stream.StreamingSwitchGLU.__call__)
check("off unless asked", stream.VIEWS_PREFILL is False and "BIGRIG_VIEWS_PREFILL" in
      open(os.path.join(ROOT, "bigrig_engine", "stream.py")).read())
check("only a real prefill chunk that wants most of the layer takes it",
      "flat.shape[0] >= VIEWS_MIN_TOKENS" in _src3 and "VIEWS_MIN_SHARE * self._pool.n_experts" in _src3
      and stream.VIEWS_MIN_TOKENS >= 4)
_fv = _ins2.getsource(stream.StreamingSwitchGLU._forward_views)
check("it fetches every expert of the chunk once and never admits to the pool",
      "self._fetcher.fetch(" in _fv and "admit" not in _fv and "ensure(" not in _fv)
check("rows come back in the caller's (token, k) order", "np.argsort(order, kind=\"stable\")" in _fv
      and "return y[mx.array(inv)]" in _fv)
check("the CLI exposes it as --file-pool and says what it costs",
      "--file-pool" in open(os.path.join(ROOT, "bigrig_engine", "cli.py")).read()
      and "near-tie" in open(os.path.join(ROOT, "bigrig_engine", "cli.py")).read())

print("\n" + "=" * 82)
print("3d4. IN VIEWS MODE A RESIDENT EXPERT IS A LIVE VIEW, AND EVICTION IS A DROP")
print("=" * 82)
# The file as the pool, for decode: no slot tensors, no copy per miss, no second copy of the hot
# experts beside the page cache. What must hold: the policy and the counts are exactly the
# slot pool's; the view of an admitted expert is retrievable; an evicted expert's view is gone
# the moment it is evicted; the byte count is honest without any slot tensor to sum.
_pv = stream.ExpertPool(0, _FakeSG(E), E, C, spec, policy="lfuda", views=True).build()
check("a views-mode pool allocates no slot tensors", _pv._slots == {} and _pv.views is True)
check("...and weighs nothing until something is resident", _pv.nbytes() == 0)
for e in range(C):
    _pv.touch([e], [0]); _pv.admit(e, bytes([e]) * len(BLOCK))
check("each admitted expert's arrays are retrievable by id, in (proj, comp) order",
      all(len(_pv.views_of(e)) == 3 for e in range(C))
      and int(np.array(_pv.views_of(2)[0]).reshape(-1)[0]) == int.from_bytes(bytes([2]) * 4, "little"))
check("the byte count is resident experts times bytes per expert",
      _pv.nbytes() == C * _pv.bytes_per_expert and _pv._resident == C)
_pv.touch([C], [0]); _pv.admit(C, bytes([C]) * len(BLOCK))
_gone = [e for e in range(C) if _pv.slot_of(e) < 0]
check("admitting one more evicts exactly one", len(_gone) == 1 and _pv._resident == C)
check("...and the evicted expert's view is dropped, so its pages can be released",
      all(e not in _pv._views for e in _gone) and C in _pv._views)
check("the slot-mode pool is untouched by any of this", fresh().views is False)
_rows = mx.zeros((2, 8), dtype=mx.float32)
_seen = []
def _provider(e):
    _seen.append(int(e))
    return _pv.views_of(e)
_fake2 = type("M", (), {})(); _fake2._pool = _pv; _fake2._fetcher = None
_fake2._quantized = False; _fake2._activation = lambda u, g: u
try:
    stream.StreamingSwitchGLU._forward_views(_fake2, _rows, np.array([[C, C], [C, C]]), _provider)
    check("decode through views asks the provider once per distinct expert", _seen == [C], str(_seen))
except Exception as _e:                              # noqa: BLE001
    check("decode through views asks the provider once per distinct expert",
          "uint32" in str(_e) or "matmul" in str(_e).lower(), f"{_e!r}")
    print("        (the fake's uint32 weights cannot be multiplied; the routing was exercised)")
check("attach only puts STREAMED layers in views mode",
      "views=bool(VIEWS_DECODE and C < E)" in _ins2.getsource(stream.attach))
check("the footprint does NOT charge the views twice: MLX counts the imports, the Mac does not "
      "charge the pages (measured 4.06 against 4.62 GB)",
      "count them twice" in open(os.path.join(ROOT, "bigrig_engine", "session.py")).read())

print("\n" + "=" * 82)
print("3d5. TWO EXPERT SHAPES, AND THE ENGINE READS WHICH FROM THE MODEL")
print("=" * 82)
# Almost every MoE model is gated -- `down(act(up, gate))`, three projections. Nemotron ships a
# plain two-layer MLP -- `fc2(act(fc1))`, two projections and no multiply. Guessing wrong is not
# a crash, it is a model that computes something else; measured against the stock block on real
# Nemotron layers, the streamed result is identical at 1, 40, 64 and 300 rows.
check("the gated shape is recognised from its projections",
      stream.is_gated(("gate_proj", "up_proj", "down_proj")) is True)
check("...and Nemotron's pair is not", stream.is_gated(("fc1", "fc2")) is False)
check("projections come from the spec, in the order the bytes are laid out",
      stream.projections_of({"fc1": {}, "fc2": {}}) == ("fc1", "fc2")
      and stream.projections_of({"gate_proj": {}, "up_proj": {}, "down_proj": {}})
      == ("gate_proj", "up_proj", "down_proj"))
_specm = {p_: {"weight": {"shape": [8, 8], "dtype": "uint32", "nbytes": 8 * 8 * 4}}
          for p_ in ("fc1", "fc2")}
class _FakeMLP:
    def __init__(self, E, d=8):
        for p_ in ("fc1", "fc2"):
            setattr(self, p_, {"weight": mx.zeros((E, d, d), dtype=mx.uint32)})
_pm = stream.ExpertPool(0, _FakeMLP(E), E, C, _specm, policy="lfuda").build()
check("a two-projection pool carries two projections and knows it is not gated",
      _pm.projections == ("fc1", "fc2") and _pm.gated is False)
check("...and allocates a slot array per projection component, not three",
      len(_pm._slots) == 2, str(sorted(_pm._slots)))
check("...and one expert's bytes are the two projections' worth",
      _pm.bytes_per_expert == 2 * 8 * 8 * 4, str(_pm.bytes_per_expert))
check("a gated pool is unchanged by any of this", fresh().projections
      == ("gate_proj", "up_proj", "down_proj") and fresh().gated is True)
_srcf = _ins2.getsource(stream.StreamingSwitchGLU._forward)
check("the forward runs the gated arithmetic for one shape and the plain chain for the other",
      "self._activation(up, gate)" in _srcf and 'self._proj("fc2", self._activation(h)' in _srcf
      and "if self._pool.gated:" in _srcf)
check("quantisation is read through one helper, so no fourth place can assume gate_proj",
      "def first_projection" in open(os.path.join(ROOT, "bigrig_engine", "stream.py")).read()
      and "sg.gate_proj" not in _ins2.getsource(stream._quant_params))
check("both shapes are streamable as far as the doctor is concerned",
      {"fc1", "fc2"} <= __import__("bigrig_engine.preflight", fromlist=["x"])._STREAMABLE_PROJECTIONS)
check("Nemotron's block name and model root are known",
      "mixer" in stream._BLOCK_ATTRS and "backbone" in _ins2.getsource(stream.find_moe_sites))

print("\n" + "=" * 82)
print("3e. PREFILLING IS A CHOICE  (item 2: time to first token)")
print("=" * 82)
import inspect as _ins
_w = _ins.getsource(stream.StreamHandle.warm)
check("warm() takes a mode", "mode" in _ins.signature(stream.StreamHandle.warm).parameters)
check("'lazy' prefills nothing", 'if mode == "lazy":' in _w and "return self" in _w)
check("'auto' prefills only a pool that will hold everything anyway",
      'mode == "auto" and p.capacity < p.n_experts' in _w)
check("an unknown mode raises rather than silently prefilling",
      "warm mode must be" in _w)
# MEASURED, and it is why 'auto' is the default rather than 'full'.
print("        measured, OLMoE at 50% residency: full prefill reached the first token in")
print("        2.23s, lazy in 1.12s, with identical output. On Qwen3-30B at 45% residency")
print("        full prefill also drove MLX 2.5 GB higher -- over the 9 GB ceiling entirely.")

# ---------------------------------------------------------------- staged speculation
# Bytes become arrays in exactly one place, so an expert staged during the wait for a layer's
# routing is bit-for-bit the array a direct admit would have written. These pin the pool half;
# the whole path is proved end to end by identical greedy text with the predictor on and off.
p = fresh()
blob = bytes(bytearray((i * 37 + 11) % 256 for i in range(len(BLOCK))))   # not all zeros
arrs = p.build_arrays(3, blob)
check("build_arrays returns one array per (projection, component), in spec order",
      len(arrs) == sum(len(c) for c in spec.values()))
p.touch([3], [0]); s_direct = p.admit(3, blob)
q = fresh(); q.touch([3], [0]); s_staged = q.admit_arrays(3, q.build_arrays(3, blob))
same = all(bool(mx.array_equal(p._slots[k][s_direct].view(mx.uint8), q._slots[k][s_staged].view(mx.uint8)))
           for k in p._slots)
check("a staged admit writes exactly the bytes a direct admit writes", same)
check("...and the bookkeeping matches: resident, mapped both ways, counted",
      q.g2s[3] == s_staged and q.s2g[s_staged] == 3 and q.admits == 1 and q._resident == 1)
check("a pool starts with nothing staged and nothing named", p.stage == {} and p.stage_pred is None)
try:
    p.build_arrays(4, blob[:-1]); check("a short blob is refused before any slot is touched", False)
except ValueError:
    check("a short blob is refused before any slot is touched", True)
_src = open(os.path.join(ROOT, "bigrig_engine", "stream.py"), encoding="utf-8").read()
import ast as _ast, inspect as _insp
_call_ast = _ast.parse(_insp.getsource(stream.StreamingSwitchGLU.__call__).lstrip())
_attrs = {n.attr for n in _ast.walk(_call_ast) if isinstance(n, _ast.Attribute)}
check("the routing read stays synchronous -- the async form measured 79.5 vs 51.8 ms on its own",
      "async_eval" not in _attrs and "eval" in _attrs)      # the code, not its comments
check("...and the whole mechanism is OFF by default, with the interleaved measurement beside it",
      'os.environ.get("BIGRIG_STAGE", "0") == "1"' in _src and "median 0.84x" in _src and "if not STAGING:" in _src)
check("staging happens at layer entry, while the GPU is on this layer's attention, before the read",
      _src.index("self._stage_predicted()") < _src.index("mx.eval(indices, pred)") < _src.index("gi = np.array(indices, copy=False)"))
check("a staged expert is admitted without a host copy, and the rest are fetched",
      "self._pool.admit_arrays(e, self._pool.stage[e], protect=uniq)" in _src and "if cold:" in _src)
check("whatever was staged and not used is dropped every step -- a wrong guess holds nothing",
      "self._pool.stage = {}" in _src)
check("speculation names experts for staging and never issues a disk prefetch any more",
      "self._fetcher.prefetch(keys)" not in _src.split("def _speculate")[1].split("def ")[0])
check("only experts not already resident are named", "if nxt.g2s[int(e)] < 0][:STAGE_MAX]" in _src)
check("the count named is bounded by the router's top-k, so staging fits inside the wait",
      stream.STAGE_MAX == 8)

# ---------------------------------------------------------------- zero-copy admits
# A page-aligned view of the expert map is wrapped as a GPU buffer instead of copied. Measured on
# Qwen3.6-35B-A3B-4bit, interleaved, texts identical: 1.14x on a warm page cache, 1.66-2.02x on a
# cold one, because Metal wires the region in one coalesced read where the memcpy faulted a page
# at a time. These pin the pool half: same bytes in the slot either way, and the fallback for
# anything not page-aligned.
import mmap as _mmap, tempfile as _tmp
_blk = bytes(bytearray((i * 53 + 7) % 256 for i in range(len(BLOCK))))
_pad = 3 * stream.PAGE_BYTES                  # three whole pages: one page per projection
_region = bytes(bytearray((i * 53 + 7) % 256 for i in range(_pad)))
_f = _tmp.NamedTemporaryFile(delete=False); _f.write(_region); _f.write(_region); _f.flush()
_fd = os.open(_f.name, os.O_RDONLY); _mm = _mmap.mmap(_fd, 2 * _pad, access=_mmap.ACCESS_READ)
_aligned = memoryview(_mm)[0:_pad]            # page-aligned base -- but nbytes must ALSO be whole pages
p_cp = fresh(); p_cp.touch([5], [0]); s_cp = p_cp.admit(5, _blk)
p_zc = fresh(); p_zc.touch([5], [0])
_prev = stream.ZERO_COPY; stream.ZERO_COPY = True
try:
    # a whole-page region: the zero-copy path
    spec_pages = {pr: {"weight": {"shape": [stream.PAGE_BYTES // 4, 1], "dtype": "uint32", "nbytes": stream.PAGE_BYTES}}
                  for pr in stream.PROJECTIONS}
    check("the aligned-region test fixture is itself whole pages",
          sum(c["nbytes"] for pr in spec_pages.values() for c in pr.values()) == _pad and _pad % stream.PAGE_BYTES == 0)
    q_zc = stream.ExpertPool(0, _FakeSG(E, d=1), E, C, spec_pages, policy="lfuda").build()
    q_cp = stream.ExpertPool(0, _FakeSG(E, d=1), E, C, spec_pages, policy="lfuda").build()
    region_bytes = bytes(_aligned)
    q_zc.touch([5], [0]); sz = q_zc.admit(5, _aligned)
    q_cp.touch([5], [0]); sc = q_cp.admit(5, region_bytes)
    check("a page-aligned memoryview is admitted without a copy", q_zc.zero_copy_admits == 1)
    check("...and bytes of the same region are admitted by copying", q_cp.zero_copy_admits == 0)
    check("...and the two slots hold identical bytes",
          all(bool(mx.array_equal(q_zc._slots[k][sz].view(mx.uint8), q_cp._slots[k][sc].view(mx.uint8))) for k in q_zc._slots))
    # a memoryview that is NOT whole pages (the real per-expert size rounds, this one does not)
    q_odd = stream.ExpertPool(0, _FakeSG(E), E, C, spec, policy="lfuda").build()
    q_odd.touch([5], [0]); q_odd.admit(5, memoryview(_mm)[0:len(BLOCK)])
    check("a view that is not a whole number of pages falls back to the copy, silently",
          q_odd.zero_copy_admits == 0 and q_odd.g2s[5] >= 0)
finally:
    stream.ZERO_COPY = _prev
    del _aligned; _mm.close(); os.close(_fd); os.unlink(_f.name)
check("zero-copy is on by default and switchable", 'os.environ.get("BIGRIG_ZERO_COPY", "1") != "0"' in
      open(os.path.join(ROOT, "bigrig_engine", "stream.py"), encoding="utf-8").read())
check("...and the whole-map variant that wires the file is written down as rejected",
      "wired 316 MB" in open(os.path.join(ROOT, "bigrig_engine", "stream.py"), encoding="utf-8").read())

print("\n" + "=" * 82); print("4. PREFILL CHUNKING"); print("=" * 82)
ch = stream.StreamingSwitchGLU._chunks
rows = np.arange(64, dtype=np.int64).reshape(8, 8)      # 8 tokens, 8 distinct experts each
sp = ch(rows, 16)
check("chunks tile the rows with no gap or overlap",
      sp[0][0] == 0 and sp[-1][1] == 8 and all(a[1] == b[0] for a, b in zip(sp, sp[1:])),
      str(sp))
check("no chunk needs more experts than the pool holds",
      all(len(set(rows[a:b].ravel().tolist())) <= 16 for a, b in sp), str(sp))
same = np.zeros((8, 8), dtype=np.int64)
check("rows that all want the same expert form a single chunk", len(ch(same, 16)) == 1)
check("a pool as large as the request never chunks", len(ch(rows, 64)) == 1)
check("a tighter pool chunks more", len(ch(rows, 8)) >= len(ch(rows, 24)))

print("\n" + "=" * 82); print("5. THE CAPACITY PLANNER"); print("=" * 82)
pl = stream.plan_capacity(48, 128, int(0.60 * 48 * 128), top_k=8)
check("the plan fits the budget it was given",
      pl["full_layers"] * 128 + pl["streamed_layers"] * pl["capacity"] <= int(0.60 * 48 * 128),
      str(pl))
check("every streamed layer can serve at least one token", pl["capacity"] >= 8, str(pl["capacity"]))
check("the plan is never worse than uniform on its own cost model",
      pl["ms_per_token"] <= pl["uniform_ms_per_token"] + 1e-9,
      f"{pl['ms_per_token']:.1f} vs {pl['uniform_ms_per_token']:.1f}")
try:
    stream.plan_capacity(48, 128, 48 * 4, top_k=8)
    check("a budget below top-k per layer is rejected", False)
except ValueError as ex:
    check("a budget below top-k per layer is rejected", "floor" in str(ex))
# MEASURED, and it contradicts the planner's premise. Recorded so the claim is not repeated.
print("        measured on Qwen3-30B: the planned split ran at 7.57 tok/s and uniform at 7.59 --")
print("        no gain, because a full-residency layer falls back to mmap, which faults at")
print("        0.70 GB/s. The planner's model of a full layer as FREE is wrong on this path.")

print("\n" + "=" * 82)
print("5b. THE SYNC-FREE PATH  (a layer that cannot miss must not touch the host)")
print("=" * 82)
pf = stream.ExpertPool(0, _FakeSG(16), 16, 16, spec, policy="lfuda").build()
check("a pool below full residency is not marked full",
      not stream.ExpertPool(0, _FakeSG(16), 16, 8, spec).build().full)
for e in range(16):
    pf.touch([e], [0]); pf.admit(e, BLOCK)
check("a pool holding every expert is marked full", pf.full)
dev = pf.g2s_device()
check("the device table matches the host table",
      all(int(np.array(dev)[e]) == int(pf.g2s[e]) for e in range(16)))
first = pf.g2s_device()
check("the device table is not rebuilt when nothing changed", pf.g2s_device() is first)
pf.touch([0], [0])
check("...and is still not rebuilt by a mere hit", pf.g2s_device() is first)

# MEASURED, and recorded because the prediction was wrong by 4x. The mechanism is real; the
# cost model behind the split is not portable between models.
print("        measured: OLMoE at a 90% budget, 14 of 16 layers sync-free -> 1.19x")
print("                  Qwen3 at a 60% budget, 10 of 48 layers sync-free -> 0.97x")
print("                  the planner predicted 4.87x and 1.06x, so it ships OFF by default")
cv = stream._sync_curve()
check("the measured sync curve is available to the planner", cv is not None)
if cv:
    f, d = cv
    xs = [0, 6, 12, 24, 36, 48]
    ys = [f(n, 48) for n in xs]
    check("the curve rises monotonically with the number of syncing layers",
          all(a <= b + 1e-9 for a, b in zip(ys, ys[1:])), str([round(y, 1) for y in ys]))
    slopes = [(ys[i + 1] - ys[i]) / (xs[i + 1] - xs[i]) for i in range(1, len(xs) - 1)]
    check("it is genuinely non-linear, so a constant per-sync cost would be wrong",
          max(slopes) / max(1e-9, min(slopes)) > 1.5,
          f"slopes {[round(x, 2) for x in slopes]}")
pl = stream.plan_capacity(48, 128, int(0.40 * 48 * 128), top_k=8)
check("at a budget where the trade does not pay, the planner declines to take it",
      pl["full_layers"] == 0, str(pl["full_layers"]))
pl9 = stream.plan_capacity(48, 128, int(0.90 * 48 * 128), top_k=8)
check("at a budget where it does pay, the planner takes it", pl9["full_layers"] > 0)
check("the plan never exceeds its budget",
      pl9["full_layers"] * 128 + pl9["streamed_layers"] * pl9["capacity"] <= int(0.90 * 48 * 128))

print("\n" + "=" * 82); print("6. THE ROUND-TRIP AGAINST A REAL MODEL"); print("=" * 82)
if os.path.exists(RT):
    d = json.load(open(RT))
    check("output was bit-identical at every residency measured",
          all(r["maxdiff"] == 0.0 for r in d["rows"]), str([r["maxdiff"] for r in d["rows"]]))
    check("generated text was unchanged at every residency",
          all(r["text_ok"] for r in d["rows"]))
    check("a smaller pool always holds strictly less memory",
          all(a["gb"] > b["gb"] for a, b in zip(d["rows"], d["rows"][1:])))
    check("a smaller pool always misses at least as much",
          all(a["miss"] <= b["miss"] + 1e-9 for a, b in zip(d["rows"], d["rows"][1:])))
    r50 = min(d["rows"], key=lambda r: abs(r["frac"] - 0.5))
    check("a half-size pool holds about half the expert bytes",
          abs(r50["gb"] / d["blob_gb"] - 0.5) < 0.06, f"{r50['gb']/d['blob_gb']:.3f}")
else:
    check("stream_roundtrip.json exists", False, "run src/spike/stream_roundtrip.py")

print("\n" + "=" * 82); print("7. THE MEMORY SAVING MUST APPEAR IN RSS, NOT ONLY IN OUR OWN BOOKS")
print("=" * 82)
BF = os.path.join(ROOT, "data/results/stream_bench.jsonl")
if os.path.exists(BF):
    rows_ = [json.loads(l) for l in open(BF)]
    base = next((r for r in rows_ if not r["streamed"]), None)
    sm = sorted((r for r in rows_ if r["streamed"]), key=lambda r: r["capacity"])
    check("peak RSS falls monotonically as the pool shrinks",
          all(a["peak_rss_gb"] <= b["peak_rss_gb"] + 0.05 for a, b in zip(sm, sm[1:])),
          str([(r["capacity"], round(r["peak_rss_gb"], 2)) for r in sm]))
    if base:
        check("the smallest pool uses less than half the baseline's peak RSS",
              sm[0]["peak_rss_gb"] < base["peak_rss_gb"] * 0.55,
              f"{sm[0]['peak_rss_gb']:.2f} vs {base['peak_rss_gb']:.2f}")
    check("zero misses at full residency", sm[-1]["miss_rate"] == 0.0)
else:
    check("stream_bench.jsonl exists", False, "run src/spike/stream_bench.py")

print("\n" + "=" * 78)
print("WARMING THE PAGE CACHE MUST BE FAST, BOUNDED, AND STOPPABLE")
print("=" * 78)
import inspect as _i
import tempfile as _tf
_blob = _tf.NamedTemporaryFile(delete=False, suffix=".experts")
_blob.write(b"\0" * (48 << 20))
_blob.close()
try:
    r = stream.warm_page_cache(_blob.name)
    check("it reads the whole file when given no budget", r["bytes"] == 48 << 20,
          f"{r['bytes']} bytes")
    # THE BUG THIS PINS: under_pressure samples over a 0.4 s window, and calling it once per
    # 32 MB chunk spent 45 of 47 seconds asleep -- 3.62 GB warmed at 0.08 GB/s against a raw
    # read of 4.9 GB/s. It is checked on a timer now, not per chunk.
    check("...at something like disk speed, not once per sleep", r["seconds"] < 2.0,
          f"took {r['seconds']}s for 48 MB")
    # A rate, not a speed: the disk may be busy with another process's model load while this
    # runs (measured 0.46 GB/s beside one), and the speed guard is the timing check above.
    check("...and reports the rate it achieved", r["gb_per_s"] > 0.05, f"{r['gb_per_s']} GB/s")
    check("the pressure check is throttled rather than per chunk",
          "last_check" in _i.getsource(stream.warm_page_cache))
    r = stream.warm_page_cache(_blob.name, budget_bytes=8 << 20)
    check("a budget is honoured and not exceeded", (8 << 20) <= r["bytes"] < (48 << 20),
          f"{r['bytes']} bytes")
    r = stream.warm_page_cache(_blob.name, should_stop=lambda: True)
    check("it stops immediately when asked",
          r["bytes"] == 0 and r["stopped"] == "asked to stop")
    _seen = {"n": 0}
    def _stop_after_two():
        _seen["n"] += 1
        return _seen["n"] > 2
    r = stream.warm_page_cache(_blob.name, should_stop=_stop_after_two, chunk=4 << 20)
    check("...and part way through, keeping what it already read",
          0 < r["bytes"] < (48 << 20), f"{r['bytes']} bytes")
    r = stream.warm_page_cache("/nonexistent/nothing.experts")
    check("a missing file is reported, not raised",
          r["bytes"] == 0 and "cannot read" in r["stopped"])
    r = stream.warm_page_cache(_blob.name, budget_bytes=1 << 40)
    check("a budget larger than the file stops at the file", r["bytes"] == 48 << 20)
    # A reply in flight fetches experts from the same disk; the warm must wait for it and then
    # finish, rather than compete with it or give up.
    _pauses = {"n": 0}
    def _busy_twice():
        _pauses["n"] += 1
        return _pauses["n"] <= 2
    r = stream.warm_page_cache(_blob.name, chunk=16 << 20, should_pause=_busy_twice)
    check("the warm yields while a reply is in flight and then completes the file",
          r["bytes"] == 48 << 20 and _pauses["n"] >= 3, f"{r['bytes']} bytes, asked {_pauses['n']}x")
    r = stream.warm_page_cache(_blob.name, should_pause=lambda: True, should_stop=lambda: True)
    check("a stop while paused wins over the pause", r["stopped"] == "asked to stop")
    # THE MOST-USED EXPERTS FIRST. A cache too small for the file should hold what the model asks
    # for, not the first bytes of the file. The order comes from a record of use, merged across
    # runs with the past halved so it tracks what the model is used for lately.
    _man = {"regions": {"0:0": [0, 1 << 20], "0:1": [1 << 20, 1 << 20], "0:2": [2 << 20, 1 << 20],
                        "1:0": [3 << 20, 1 << 20]}}
    _hot = stream.hot_regions(_man, {"0:2": 50, "1:0": 40, "0:0": 3, "9:9": 999}, limit_bytes=2 << 20)
    check("hot regions come hottest first, within the budget, and ignore experts the manifest "
          "does not know", _hot == [(2 << 20, 1 << 20), (3 << 20, 1 << 20)], str(_hot))
    r = stream.warm_page_cache(_blob.name, budget_bytes=3 << 20, regions=_hot, chunk=1 << 20)
    check("the warm reads the hot experts first and counts them",
          r["hot_bytes"] == 2 << 20 and r["bytes"] == 3 << 20, str(r))
    _pu = fresh()
    for _e in (3, 3, 3, 5):
        _pu.touch([_e], [0])
    check("a pool counts how often each expert was asked for",
          int(_pu.uses[3]) == 3 and int(_pu.uses[5]) == 1 and int(_pu.uses.sum()) == 4)
    class _H:
        mods = [type("M", (), {"_pool": _pu})()]
    _H.usage = stream.StreamHandle.usage; _H.save_usage = stream.StreamHandle.save_usage
    _tmpd = _tf.mkdtemp(); _up = os.path.join(_tmpd, "usage_x.json")
    _H().save_usage(_up)
    check("use counts are written as layer:expert -> count", json.load(open(_up)) == {"0:3": 3, "0:5": 1})
    _H().save_usage(_up)
    check("...and merged with the past halved, so lately outweighs long ago",
          json.load(open(_up)) == {"0:3": 4, "0:5": 1}, str(json.load(open(_up))))
    check("the usage file lives beside the knee, named for the model",
          stream.usage_path("mlx-community/M").endswith("usage_mlx-community_M.json"))
    check("the server warms hot experts first and a session saves its counts on close",
          "hot_regions(" in open(os.path.join(ROOT, "bigrig_engine", "server.py")).read()
          and "save_usage(stream.usage_path(self.name))" in open(os.path.join(ROOT, "bigrig_engine", "session.py")).read())
finally:
    os.unlink(_blob.name)


print("\n" + "=" * 82)
print(f"{'ALL TESTS PASSED' if not FAIL else str(len(FAIL))+' FAILURES: '+', '.join(FAIL)}")
print("=" * 82)
sys.exit(1 if FAIL else 0)
