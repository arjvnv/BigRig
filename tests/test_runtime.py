"""Adversarial tests for the engine.

The engine is the only component that can fail by COMPOSITION -- every part correct, the whole
wrong. That already happened once: plan() correctly clamps the miss rate to zero for a model that
fits, the pool correctly reports a 49.78% miss at a 16-of-64 capacity, and together they printed
"213 tok/s, disk 0% of the time" one line after measuring half the requests missing. Neither
component's own tests could have caught it.
"""
import json
import os
import struct
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bigrig_engine.repack import repack
from bigrig_engine.runtime import Engine, EngineStats

FAIL = []
def check(n, c, d=""):
    (print(f"  PASS  {n}") if c else (FAIL.append(n), print(f"  FAIL  {n}  {d}")))


def tiny_checkpoint(d, E=8, L=3):
    os.makedirs(d, exist_ok=True)
    json.dump({"model_type": "olmoe", "num_experts": E, "num_experts_per_tok": 2,
               "num_hidden_layers": L}, open(os.path.join(d, "config.json"), "w"))
    rng = np.random.default_rng(0)
    hdr, blobs, off = {}, [], 0
    for l in range(L):
        for proj in ("gate_proj", "up_proj", "down_proj"):
            a = rng.integers(0, 2**31, (E, 8, 16), dtype=np.uint32)
            b = a.tobytes()
            hdr[f"model.layers.{l}.mlp.switch_mlp.{proj}.weight"] = {
                "dtype": "U32", "shape": list(a.shape), "data_offsets": [off, off + len(b)]}
            blobs.append(b); off += len(b)
    raw = json.dumps(hdr).encode()
    with open(os.path.join(d, "model.safetensors"), "wb") as f:
        f.write(struct.pack("<Q", len(raw))); f.write(raw)
        for b in blobs: f.write(b)
    return d


TMP = tempfile.mkdtemp(prefix="bigrig_rt_")
SRC = tiny_checkpoint(os.path.join(TMP, "m"))
PACKED = os.path.join(TMP, "packed.experts")
repack(SRC, PACKED, 8, verify_every=True)
PROFILE = {"ram_gbs": 90.0, "disk_gbs": 5.0, "kappa": 18.0, "fetch_threads": 4,
           "available_gb": 16.0, "under_pressure": False}

print("=" * 78); print("1. THE ENGINE COMPOSES AT ALL"); print("=" * 78)
with Engine(SRC, PACKED, capacity=4, profile=PROFILE) as eng:
    check("it describes the model", eng.spec.n_experts == 8 and eng.spec.top_k == 2,
          eng.spec.summary())
    check("it builds one pool per MoE layer", len(eng.pools) == 3, str(len(eng.pools)))
    check("capacity is honoured", eng.capacity == 4)
    check("nothing is resident before the first token",
          all(len(v) == 0 for v in eng.resident.values()))

print("\n" + "=" * 78); print("2. THE FIRST TOUCH OF EVERY EXPERT MUST MISS"); print("=" * 78)
with Engine(SRC, PACKED, capacity=8, profile=PROFILE) as eng:
    miss1 = eng.route(0, [0, 1])
    check("a cold pool misses on both experts", miss1 == [0, 1], str(miss1))
    miss2 = eng.route(0, [0, 1])
    check("the same experts hit the second time", miss2 == [], str(miss2))
    check("accounting is exact: 4 requests, 2 misses",
          eng.stats.expert_requests == 4 and eng.stats.misses == 2, eng.stats.report())
    check("miss rate is misses/requests", abs(eng.stats.miss_rate - 0.5) < 1e-9)
    check("bytes were actually fetched", eng.stats.bytes_fetched > 0)

print("\n" + "=" * 78); print("3. THE POOL MUST NEVER EXCEED ITS CAPACITY"); print("=" * 78)
rng = np.random.default_rng(0)
with Engine(SRC, PACKED, capacity=3, profile=PROFILE) as eng:
    over = False
    for _ in range(200):
        eng.route(0, rng.integers(0, 8, 2).tolist())
        if len(eng.resident[0]) > 3:
            over = True
    check("resident set never exceeds capacity", not over, str(len(eng.resident[0])))
    check("...and it does fill up", len(eng.resident[0]) == 3, str(len(eng.resident[0])))
    check("every miss is counted", eng.stats.misses <= eng.stats.expert_requests)

print("\n" + "=" * 78); print("4. A BIGGER POOL MUST MISS LESS"); print("=" * 78)
rates = {}
for cap in (2, 4, 8):
    with Engine(SRC, PACKED, capacity=cap, profile=PROFILE) as eng:
        r = np.random.default_rng(7)
        for _ in range(400):
            eng.route(0, r.integers(0, 8, 2).tolist())
        rates[cap] = eng.stats.miss_rate
check("miss rate falls monotonically with capacity",
      rates[2] >= rates[4] >= rates[8], str(rates))
check("a pool holding every expert misses only compulsory",
      rates[8] < 0.05, f"{rates[8]:.4f}")
print(f"        capacity 2 -> {rates[2]:.3f},  4 -> {rates[4]:.3f},  8 -> {rates[8]:.3f}")

print("\n" + "=" * 78); print("5. THE COMPOSITION BUG MUST STAY FIXED"); print("=" * 78)
# plan() clamps the miss rate to zero for a model that fits. The engine's projection must use
# its OWN configured capacity instead, or it reports "disk 0% of the time" right after
# measuring half the requests missing.
with Engine(SRC, PACKED, capacity=2, profile=PROFILE) as eng:
    r = np.random.default_rng(3)
    for _ in range(300):
        eng.route(0, r.integers(0, 8, 2).tolist())
    p = eng.projected_tok_s()
    check("the projection uses the MEASURED miss rate, not a clamped one",
          abs(p["miss_rate_used"] - eng.stats.miss_rate) < 1e-9,
          f"projection {p['miss_rate_used']} vs measured {eng.stats.miss_rate}")
    check("a high miss rate means disk dominates the time",
          p["disk_fraction_of_time"] > 0.5, str(p))
    check("residency reflects the CONFIGURED pool, not free memory",
          abs(p["residency"] - 2 / 8) < 1e-9, str(p))
    print(f"        miss {eng.stats.miss_rate*100:.1f}% -> {p['tok_s']} tok/s, "
          f"disk {p['disk_fraction_of_time']*100:.0f}% of the time")

with Engine(SRC, PACKED, capacity=8, profile=PROFILE) as eng:
    for _ in range(50):
        eng.route(0, [0, 1])
    p = eng.projected_tok_s()
    # MY EXPECTATION WAS WRONG HERE, NOT THE CODE. I asserted that a near-zero miss rate means a
    # near-zero disk share. With kappa = 18 it does not: a 2% miss rate puts 27% of the time on
    # disk, because each miss costs 18 times what a hit does. The correct assertion is that the
    # disk share follows the arithmetic, so it is checked against the formula rather than a guess.
    m = eng.stats.miss_rate
    dp = m / PROFILE["disk_gbs"]
    rp = (1 - m) / PROFILE["ram_gbs"]
    check("the disk share matches the arithmetic exactly",
          abs(p["disk_fraction_of_time"] - dp / (dp + rp)) < 0.002,
          f"reported {p['disk_fraction_of_time']} vs computed {dp/(dp+rp):.4f}")
    check("...and a low miss rate still means a LOWER disk share than a high one",
          p["disk_fraction_of_time"] < 0.5, str(p))
    print(f"        miss {m*100:.1f}% -> disk is {p['disk_fraction_of_time']*100:.0f}% of the "
          f"time. A 'small' miss rate is not a small cost.")

print("\n" + "=" * 78); print("6. BYTE VERIFICATION MUST BE ABLE TO FAIL"); print("=" * 78)
# TEETH: bind a live-weights source that deliberately returns wrong bytes.
with Engine(SRC, PACKED, capacity=4, profile=PROFILE, verify_bytes=16) as eng:
    eng.bind_live_weights(lambda l, e: b"\x00" * 64)
    eng.route(0, [0, 1, 2])
    check("wrong 'live' bytes are detected as mismatches", eng.stats.byte_mismatches > 0,
          f"{eng.stats.byte_mismatches} mismatches of {eng.stats.byte_checks} checks")
with Engine(SRC, PACKED, capacity=4, profile=PROFILE, verify_bytes=16) as eng:
    manifest = eng.manifest
    def truthful(l, e):
        blk = manifest["layout"].get(f"{l}/{e}")
        if blk is None: return None
        with open(PACKED, "rb") as f:
            f.seek(blk["offset"]); return f.read(blk["length"])
    eng.bind_live_weights(truthful)
    eng.route(0, [0, 1, 2])
    check("correct bytes produce zero mismatches",
          eng.stats.byte_checks > 0 and eng.stats.byte_mismatches == 0,
          f"{eng.stats.byte_mismatches}/{eng.stats.byte_checks}")

print("\n" + "=" * 78); print("7. HOSTILE INPUT AND LIFECYCLE"); print("=" * 78)
with Engine(SRC, PACKED, capacity=4, profile=PROFILE) as eng:
    check("routing an unknown layer is ignored rather than crashing", eng.route(99, [0]) == [])
    check("an empty expert list is harmless", eng.route(0, []) == [])
    before = eng.stats.expert_requests
    eng.route(0, [0], ranks=[0])
    check("explicit ranks are accepted", eng.stats.expert_requests == before + 1)
e = Engine(SRC, PACKED, capacity=4, profile=PROFILE)
e.close(); e.close()
check("closing twice is harmless", True)
try:
    Engine(os.path.join(TMP, "nope"), PACKED, profile=PROFILE)
    check("a missing model directory raises", False)
except (FileNotFoundError, NotADirectoryError):
    check("a missing model directory raises", True)

with Engine(SRC, PACKED, capacity=4, profile=PROFILE) as eng:
    rep = eng.report()
    for k in ("miss_rate", "gb_fetched", "capacity_per_layer", "residency_pct", "policy",
              "host", "projected"):
        check(f"report contains `{k}`", k in rep)
    check("the default policy is the measured winner", rep["policy"] == "lfuda", rep["policy"])

import shutil
shutil.rmtree(TMP, ignore_errors=True)
print("\n" + "=" * 78)
print(f"{'ALL TESTS PASSED' if not FAIL else str(len(FAIL)) + ' FAILURES: ' + ', '.join(FAIL)}")
print("=" * 78)
sys.exit(1 if FAIL else 0)
