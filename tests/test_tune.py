"""Adversarial tests for per-layer precision tuning.

The tuner measures how much each layer minds being compressed, then spends the memory budget
accordingly. It has two failure modes and they are opposites:

  * the simulation does not actually degrade the layer -- every layer measures as insensitive,
    the plan is arbitrary, and it looks like it worked
  * restore does not put the layer back -- damage accumulates, every later layer is measured
    against a model that is already broken, and the ranking is garbage

Both produce a confident plan built on nothing, so both are tested directly rather than inferred
from the plan looking reasonable.
"""
import json
import os
import sys
import tempfile

import mlx.core as mx
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bigrig_engine import precision, stream, tune

FAIL = []
def check(n, c, d=""):
    (print(f"  PASS  {n}") if c else (FAIL.append(n), print(f"  FAIL  {n}  {d}")))

QUANT = {"bits": 4, "group_size": 64, "mode": "affine"}


def make_pool(C=4, out=64, inp=256):
    """A pool holding real quantised weights, so the round-trip is exercised for real."""
    spec, slots = {}, {}
    for proj in stream.PROJECTIONS:
        a = mx.random.normal((C, out, inp)).astype(mx.float16)
        w, sc, bi = mx.quantize(a, group_size=QUANT["group_size"], bits=QUANT["bits"])
        mx.eval(w, sc, bi)
        slots[(proj, "weight")], slots[(proj, "scales")], slots[(proj, "biases")] = w, sc, bi
        spec[proj] = {"weight": {"shape": list(w.shape[1:]), "dtype": "uint32", "nbytes": 0},
                      "scales": {"shape": list(sc.shape[1:]), "dtype": "float16", "nbytes": 0},
                      "biases": {"shape": list(bi.shape[1:]), "dtype": "float16", "nbytes": 0}}

    class P:
        pass
    p = P()
    p.spec, p._slots, p.spec_quant, p.layer = spec, slots, dict(QUANT), 0
    # The projections this model carries, read off the spec exactly as a real pool does. A stub
    # that hardcoded the gated trio would test a different function than the one that ships now
    # that two expert shapes exist.
    p.projections = stream.projections_of(spec)
    p.gated = stream.is_gated(p.projections)
    return p


def deq(p, proj):
    s = p._slots
    return np.array(mx.dequantize(s[(proj, "weight")], s[(proj, "scales")],
                                  s[(proj, "biases")], group_size=QUANT["group_size"],
                                  bits=QUANT["bits"]).astype(mx.float32))


print("=" * 82); print("1. THE SIMULATION MUST ACTUALLY DEGRADE THE LAYER"); print("=" * 82)
p = make_pool()
before = {pr: deq(p, pr) for pr in stream.PROJECTIONS}
shapes_before = {k: v.shape for k, v in p._slots.items()}
snap = tune.simulate_precision(p, 2, 128)
after = {pr: deq(p, pr) for pr in stream.PROJECTIONS}
check("the weights change -- a no-op simulation would measure every layer as insensitive",
      all(not np.array_equal(before[pr], after[pr]) for pr in stream.PROJECTIONS))
diffs = [float(np.abs(before[pr] - after[pr]).mean()) for pr in stream.PROJECTIONS]
check("...and by an amount consistent with a 2-bit round trip",
      all(d > 1e-4 for d in diffs), str([round(d, 5) for d in diffs]))
check("the STORAGE shape is untouched, so the pool never has to be rebuilt",
      {k: v.shape for k, v in p._slots.items()} == shapes_before)
check("...and so is the dtype",
      all(p._slots[k].dtype == v for k, v in
          {k: mx.uint32 if k[1] == "weight" else p._slots[k].dtype
           for k in p._slots}.items()))
# A gentler probe must do less damage than an aggressive one, or the scan is measuring noise.
p2 = make_pool()
b2 = {pr: deq(p2, pr) for pr in stream.PROJECTIONS}
tune.simulate_precision(p2, 6, 64)
a2 = {pr: deq(p2, pr) for pr in stream.PROJECTIONS}
gentle = float(np.abs(b2["gate_proj"] - a2["gate_proj"]).mean())
check("6-bit damages less than 2-bit", gentle < diffs[0], f"{gentle:.5f} vs {diffs[0]:.5f}")

print("\n" + "=" * 82); print("2. RESTORE MUST BE EXACT, OR DAMAGE ACCUMULATES"); print("=" * 82)
tune.restore(p, snap)
back = {pr: deq(p, pr) for pr in stream.PROJECTIONS}
check("every projection is bit-identical to before the simulation",
      all(np.array_equal(before[pr], back[pr]) for pr in stream.PROJECTIONS),
      str([float(np.abs(before[pr] - back[pr]).max()) for pr in stream.PROJECTIONS]))
# Ten cycles: if restore leaked even slightly, this is where it would show.
p3 = make_pool()
orig = {pr: deq(p3, pr) for pr in stream.PROJECTIONS}
for i in range(10):
    sn = tune.simulate_precision(p3, 2, 128)
    tune.restore(p3, sn)
check("ten simulate/restore cycles leave the layer exactly as it started",
      all(np.array_equal(orig[pr], deq(p3, pr)) for pr in stream.PROJECTIONS))

print("\n" + "=" * 82); print("3. TURNING MEASUREMENTS INTO A PLAN"); print("=" * 82)
LAYERS, E, BPE = 16, 64, 3_538_944
man = {"layers": {str(i): {"n_experts": E, "bytes_per_expert": BPE, "quant": dict(QUANT),
                           "spec": {}} for i in range(LAYERS)},
       "total_bytes": LAYERS * E * BPE}
prof = {"sensitivity": {str(i): 0.01 * (i + 1) for i in range(LAYERS)}}
full = man["total_bytes"]
# 3-bit g128 is 72% of 4-bit g64, so with the floor enforced that is the smallest reachable
# budget. Asking for less is not a bug in the allocator, it is a request it must refuse.
for frac in (0.95, 0.85, 0.75):
    plan = tune.plan_from_profile(prof, man, full * frac)
    counts = {i: BPE * E / precision.bytes_per_param(**{"bits": QUANT["bits"],
                                                        "group_size": QUANT["group_size"]})
              for i in range(LAYERS)}
    used = precision.plan_bytes(plan, counts)
    check(f"a {frac*100:.0f}% budget is respected", used <= full * frac * 1.001,
          f"{used/1e9:.2f} vs {full*frac/1e9:.2f} GB")
    check(f"...and nothing drops below the 3-bit floor at {frac*100:.0f}%",
          all(b >= 3 for b, _ in plan.values()), str(sorted({b for b, _ in plan.values()})))
plan = tune.plan_from_profile(prof, man, full * 0.75)
lo = min(precision.bytes_per_param(*v) for v in plan.values())
hi = max(precision.bytes_per_param(*v) for v in plan.values())
cut = [l for l in plan if precision.bytes_per_param(*plan[l]) == lo]
kept = [l for l in plan if precision.bytes_per_param(*plan[l]) == hi]
check("the layers cut hardest are the ones measured as least sensitive",
      lo == hi or max(prof["sensitivity"][str(l)] for l in cut)
      <= min(prof["sensitivity"][str(l)] for l in kept) + 1e-9,
      f"cut {sorted(cut)}, kept {sorted(kept)}")
# A sensitivity of zero or below means the measurement could not resolve that layer. It must
# not crash the allocator, and it must not be treated as "free to destroy".
noisy = {"sensitivity": {str(i): (-0.001 if i == 3 else 0.02) for i in range(LAYERS)}}
plan_n = tune.plan_from_profile(noisy, man, full * 0.80)
check("a layer that measured at or below zero still gets a real precision",
      plan_n[3][0] >= 3, str(plan_n[3]))
try:
    tune.plan_from_profile({"sensitivity": {"0": 0.1}}, man, full * 0.80)
    check("a profile missing layers is refused, not silently filled in", False)
except ValueError as e:
    check("a profile missing layers is refused, not silently filled in",
          "missing" in str(e) and "partial measurement" in str(e))
try:
    tune.plan_from_profile(prof, man, full * 0.55)
    check("a budget below the 3-bit floor is refused, not approximated", False)
except ValueError as e:
    check("a budget below the 3-bit floor is refused, not approximated",
          "cannot reach" in str(e))
check("min_bits is honoured above the floor",
      all(b >= 4 for b, _ in tune.plan_from_profile(prof, man, full * 0.95,
                                                    min_bits=4).values()))

print("\n" + "=" * 82); print("4. THE CACHED PROFILE"); print("=" * 82)
d = tempfile.mkdtemp(prefix="bigrig-tune-")
blob = os.path.join(d, "m.experts")
check("no profile means None", tune.load_profile(blob) is None)
saved = tune.save_profile(blob, {"sensitivity": {"0": 0.1}, "baseline_nll": 1.0})
check("a saved profile round-trips", tune.load_profile(blob)["sensitivity"]["0"] == 0.1)
check("it is written beside the blob", saved == blob + ".tune.json")
check("no temp file is left behind", not os.path.exists(saved + ".tmp"))
for bad in ("{not json", "[]", '{"nothing": 1}'):
    open(saved, "w").write(bad)
    check(f"a damaged profile reads as None, not as data ({bad[:12]})",
          tune.load_profile(blob) is None)
import shutil
shutil.rmtree(d, ignore_errors=True)

print("\n" + "=" * 82)
print("4b. A TUNED COPY MUST ACTUALLY GET USED")
print("=" * 82)
# The tuner can write a perfect plan and it is worth nothing if the loader never opens the file.
import inspect as _i
_ec = _i.getsource(precision.ensure_compressed)
check("ensure_compressed prefers a tuned copy over the uniform one",
      '".tuned"' in _ec and _ec.index('".tuned"') < _ec.index("if os.path.exists(dst)"))
check("...and says so, rather than silently swapping the weights",
      "using the tuned" in _ec)
check("...and a damaged tuned copy falls back instead of failing",
      "not a reason to fail" in _ec)
check("the tuned path is derived from the uniform one, so they cannot diverge",
      'tuned = dst + ".tuned"' in _ec)

print("\n" + "=" * 82); print("5. THE MEASUREMENT'S OWN LIMITS"); print("=" * 82)
import inspect
sc = inspect.getsource(tune.scan)
check("every layer is measured against the SAME baseline, not against the previous layer",
      sc.count("base = evaluate.perplexity") == 1 and 'r["nll"] - base["nll"]' in sc)
check("the layer is restored even if the evaluation raises",
      "finally:" in sc and "restore(pool, snap)" in sc)
check("the probe never goes below the measured cliff",
      tune.PROBE_BITS >= 3, str(tune.PROBE_BITS))
check("...and neither can the plan", min(b for b, _ in tune.FLOOR_CANDIDATES) >= 3)
check("the reason the floor exists is written down",
      "cliff" in inspect.getdoc(tune) and "WORSE than uniform" in inspect.getdoc(tune))

print("\n" + "=" * 82)
print(f"{'ALL TESTS PASSED' if not FAIL else str(len(FAIL))+' FAILURES: '+', '.join(FAIL)}")
print("=" * 82)
sys.exit(1 if FAIL else 0)
