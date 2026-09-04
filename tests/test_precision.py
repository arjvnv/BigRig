"""Adversarial tests for precision-by-layer, the memory/quality dial, and the strategy chooser.

This code decides how much of a model a user actually gets. Its failure modes are quiet:
  * a size formula that is wrong makes every plan overrun or under-use memory
  * a requantised blob whose manifest still claims the OLD precision decodes to plausible
    garbage -- fluent text from wrong weights, the worst outcome this project can ship
  * a strategy chooser that reaches for the engine when the model already fits makes things
    slower and calls it a feature
"""
import json
import os
import subprocess
import sys

import mlx.core as mx
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bigrig_engine import autoconfig, evaluate, precision, stream

FAIL = []
def check(n, c, d=""):
    (print(f"  PASS  {n}") if c else (FAIL.append(n), print(f"  FAIL  {n}  {d}")))

BLOBS = os.path.join(ROOT, "data", "blobs")
SRC = os.path.join(BLOBS, "OLMoE-1B-7B-0125-4bit.experts")
SCRATCH = os.path.join(BLOBS, "_test")
os.makedirs(SCRATCH, exist_ok=True)

print("=" * 84); print("1. THE SIZE ARITHMETIC EVERY PLAN RESTS ON"); print("=" * 84)
bpp = precision.bytes_per_param
check("4-bit g64 is 4.5 bits per parameter", abs(bpp(4, 64) - 4.5 / 8) < 1e-12, str(bpp(4, 64)))
check("2-bit g128 is 2.25 bits per parameter", abs(bpp(2, 128) - 2.25 / 8) < 1e-12)
check("a larger group is always cheaper", bpp(4, 128) < bpp(4, 64) < bpp(4, 32))
check("more bits is always dearer", bpp(2, 64) < bpp(3, 64) < bpp(4, 64) < bpp(8, 64))
check("2-bit g128 is the floor among the shipped candidates",
      min(precision.CANDIDATES, key=lambda c: bpp(*c)) == (2, 128))
# The arithmetic must match what MLX actually produces, not what the formula hopes.
a = mx.random.normal((64, 4096)).astype(mx.float16)
for b, g in ((2, 128), (3, 64), (4, 64), (8, 64)):
    w, s, bi = mx.quantize(a, group_size=g, bits=b)
    mx.eval(w, s, bi)
    real = sum(int(np.prod(x.shape)) * x.dtype.size for x in (w, s, bi))
    pred = a.size * bpp(b, g)
    check(f"{b}-bit g{g}: predicted bytes match MLX's actual output",
          abs(real - pred) / pred < 0.02, f"{real} vs {pred:.0f}")

print("\n" + "=" * 84); print("2. REQUANTISING A BLOB ON DISK"); print("=" * 84)
if not os.path.exists(SRC):           # the blob itself; its manifest ships as a fixture
    # A packed blob is made from a downloaded model, so a fresh clone has neither. That is the
    # normal state for someone who has just cloned the repo, not a defect -- report it as a skip
    # so `./run-tests.sh` is green on first contact and still says what was not exercised.
    print(f"  SKIPPED - no packed blob at {SRC}. Section 2 requantises a real blob on disk;")
    print( "  run `bigrig prepare <model>` once and re-run to exercise it.")
else:
    man = stream.load_manifest(SRC)
    check("the source blob records its own quantisation", "quant" in
          man["layers"][sorted(man["layers"], key=int)[0]], "re-pack: blobs must be v4+")
    L, sq = len(man["layers"]), man["layers"][sorted(man["layers"], key=int)[0]]["quant"]

    dst = os.path.join(SCRATCH, "t2g128.experts")
    m2 = precision.requantize_blob(SRC, dst, {i: (2, 128) for i in range(L)}, progress=False)
    exp = man["total_bytes"] * bpp(2, 128) / bpp(sq["bits"], sq["group_size"])
    check("the compressed blob is the size the arithmetic predicts",
          abs(m2["total_bytes"] - exp) / exp < 0.01,
          f"{m2['total_bytes']/1e9:.3f} vs {exp/1e9:.3f} GB")
    check("the file on disk is exactly that size", os.path.getsize(dst) == m2["total_bytes"])
    check("the new precision is recorded in the manifest",
          m2["layers"]["0"]["quant"] == {"group_size": 128, "bits": 2, "mode": "affine"},
          str(m2["layers"]["0"]["quant"]))
    check("every layer is present", len(m2["layers"]) == L)
    check("every expert region is present", len(m2["regions"]) == len(man["regions"]))
    sizes = {n for _, n in m2["regions"].values()}
    check("all expert regions are the same size", len(sizes) == 1, str(sorted(sizes)[:3]))
    check("regions tile the file with no gap or overlap",
          sum(n for _, n in m2["regions"].values()) == m2["total_bytes"])
    check("top_k is carried across the requantisation", m2.get("top_k") == man.get("top_k"))

    # Requantising to the SAME precision must be a near-identity. If it is not, every
    # measured quality delta is contaminated by the round trip rather than by the precision.
    same = os.path.join(SCRATCH, "same.experts")
    ms = precision.requantize_blob(SRC, same, {i: (sq["bits"], sq["group_size"])
                                               for i in range(L)}, progress=False)
    check("a same-precision requantisation reproduces the original size",
          ms["total_bytes"] == man["total_bytes"],
          f"{ms['total_bytes']} vs {man['total_bytes']}")
    # NOT byte-identical: dequantise-then-requantise re-derives each group's scale from values
    # that have already been rounded once, so boundary cases land in the neighbouring bucket.
    # Measured: 14.25% of bytes differ. That matters because it means EVERY precision delta in
    # the study carries a round-trip component, so the study must measure a same-precision row
    # as its control and report deltas against that, not against the untouched checkpoint.
    a_ = np.frombuffer(open(SRC, "rb").read(1 << 20), dtype=np.uint8).astype(np.int16)
    b_ = np.frombuffer(open(same, "rb").read(1 << 20), dtype=np.uint8).astype(np.int16)
    frac = float((a_ != b_).mean())
    check("a same-precision round trip is close to the original, though not byte-identical",
          frac < 0.30, f"{frac*100:.2f}% of the first MB differs")
    print(f"        {frac*100:.2f}% of bytes change on a no-op requantisation -- which is why "
          f"the study\n        measures a same-precision control row and reports against it")
    for f in (dst, dst + ".manifest.json", same, same + ".manifest.json"):
        if os.path.exists(f):
            os.remove(f)

    # A blob whose manifest predates quantisation metadata must be refused, not guessed at.
    bad = os.path.join(SCRATCH, "v3.experts")
    open(bad, "wb").write(b"\0" * 16)
    old = dict(man); old["version"] = 3
    json.dump(old, open(bad + ".manifest.json", "w"))
    try:
        precision.requantize_blob(bad, bad + ".out", {0: (2, 128)}, progress=False)
        check("a blob without recorded precision is refused", False, "it guessed instead")
    except ValueError as e:
        check("a blob without recorded precision is refused", "quantisation parameters" in str(e))
    for f in (bad, bad + ".manifest.json"):
        os.remove(f)

print("\n" + "=" * 84); print("3. ALLOCATION"); print("=" * 84)
counts = {i: 1_000_000 for i in range(8)}
sens = {i: float(i + 1) for i in range(8)}          # layer 7 is 8x as sensitive as layer 0
full = sum(counts.values()) * bpp(8, 64)
for frac in (0.9, 0.6, 0.4, 0.30):
    budget = full * frac
    plan = precision.allocate(sens, counts, budget)
    used = precision.plan_bytes(plan, counts)
    check(f"a {frac*100:.0f}% budget is respected", used <= budget * 1.001,
          f"{used:.0f} > {budget:.0f}")
    check(f"...and every layer gets a real precision at {frac*100:.0f}%",
          all(b >= 2 and g in (32, 64, 128) for b, g in plan.values()))
plan = precision.allocate(sens, counts, full * 0.5)
lo = [l for l in plan if bpp(*plan[l]) == min(bpp(*v) for v in plan.values())]
hi = [l for l in plan if bpp(*plan[l]) == max(bpp(*v) for v in plan.values())]
check("the least sensitive layers are the ones cut hardest",
      min(sens[l] for l in hi) >= max(sens[l] for l in lo) or set(lo) == set(hi),
      f"cut {sorted(lo)}, kept {sorted(hi)}")
check("a budget larger than the model changes nothing",
      precision.plan_bytes(precision.allocate(sens, counts, full * 10), counts) <= full * 1.001)
try:
    precision.allocate(sens, counts, full * 0.01)
    check("an impossible budget raises rather than returning a fiction", False)
except ValueError as e:
    check("an impossible budget raises rather than returning a fiction", "cannot reach" in str(e))
# The floor is real and must be reported as a refusal, not silently approximated. 2-bit g128 is
# 26.5% of an 8-bit g64 model, so anything below that is unreachable by construction.
try:
    precision.allocate(sens, counts, full * 0.26)
    check("a budget just below the 2-bit floor is refused", False, "it returned a plan")
except ValueError:
    check("a budget just below the 2-bit floor is refused", True)
check("a budget just above the floor is reachable",
      precision.plan_bytes(precision.allocate(sens, counts, full * 0.27), counts)
      <= full * 0.27 * 1.001)

print("\n" + "=" * 84); print("4. THE STRATEGY CHOOSER"); print("=" * 84)
if os.path.exists(SRC + ".manifest.json"):
    man = stream.load_manifest(SRC)
    modes = {}
    for budget in (24.0, 12.0, 9.0, 8.0, 6.5, 5.0, 4.0):
        try:
            s = autoconfig.choose_strategy(man, budget_gb=budget, non_expert_gb=0.4)
            modes[budget] = s["mode"]
        except MemoryError:
            modes[budget] = "refused"
    order = ["native", "compress", "stream", "refused"]
    seq = [modes[b] for b in sorted(modes, reverse=True)]
    idx = [order.index(m) for m in seq]
    check("as memory shrinks the strategy only ever degrades, never improves",
          all(a <= b for a, b in zip(idx, idx[1:])), str(list(zip(sorted(modes, reverse=True), seq))))
    check("a roomy machine runs the model untouched", modes[24.0] == "native")
    check("a machine that cannot hold it even at the floor is told so, not given a bad plan",
          modes[4.0] in ("stream", "refused"))
    big = autoconfig.choose_strategy(man, budget_gb=24.0, non_expert_gb=0.4)
    check("the native verdict says why", "already fits" in big["reason"])
    check("...and it does not put the engine in the path", "capacity" not in big)
    comp = [autoconfig.choose_strategy(man, budget_gb=b, non_expert_gb=0.4)
            for b in (9.0, 8.0, 7.0) ]
    comp = [c for c in comp if c["mode"] == "compress"]
    if comp:
        c = comp[0]
        check("compression gives up the fewest bits that will fit",
              c["expert_gb"] <= c["room_gb"] and
              c["expert_gb"] * bpp(c["bits"] + 1, c["group_size"]) / bpp(*(c["bits"],
                  c["group_size"])) > c["room_gb"],
              f"{c['bits']}b g{c['group_size']} -> {c['expert_gb']:.2f} of {c['room_gb']:.2f} GB")
        check("...and it keeps every expert resident", "capacity" not in c)
    check("min_bits is honoured, so a user can refuse to trade accuracy",
          autoconfig.choose_strategy(man, budget_gb=8.0, non_expert_gb=0.4,
                                     min_bits=99)["mode"] == "stream")

print("\n" + "=" * 84); print("5. THE PERPLEXITY HARNESS ITSELF"); print("=" * 84)
for w, s_, why in ((1, 1, "a window of 1"), (16, 0, "a stride of 0"), (16, 99, "stride > window")):
    try:
        evaluate.perplexity(None, np.zeros(1000, dtype=np.int64), window=w, stride=s_)
        check(f"rejects {why}", False, "accepted")
    except ValueError:
        check(f"rejects {why}", True)
try:
    evaluate.perplexity(None, np.zeros(10, dtype=np.int64), window=1024, stride=512)
    check("rejects a corpus shorter than the window", False)
except ValueError as e:
    check("rejects a corpus shorter than the window", "corpus has" in str(e))
r = evaluate.compare({"nll": 1.0, "ppl": np.e}, {"nll": 1.5, "ppl": np.e ** 1.5})
check("compare reports the nll gap", abs(r["d_nll"] - 0.5) < 1e-9)
check("...and the perplexity ratio", abs(r["ppl_ratio"] - np.e ** 0.5) < 1e-9)
check("...and a percentage a person can read", r["ppl_pct"] > 0)
check("a better candidate reports a negative gap",
      evaluate.compare({"nll": 1.5, "ppl": 1.0}, {"nll": 1.0, "ppl": 1.0})["d_nll"] < 0)

print("\n" + "=" * 84)
print(f"{'ALL TESTS PASSED' if not FAIL else str(len(FAIL))+' FAILURES: '+', '.join(FAIL)}")
print("=" * 84)
sys.exit(1 if FAIL else 0)
