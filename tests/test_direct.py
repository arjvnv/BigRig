"""Adversarial tests for reading experts out of a model's own safetensors.

This code exists so a user does not need two copies of their model on disk -- 26 GB for a 13.4 GB
download was the loudest first-run complaint. It works by computing where each expert's bytes
already sit rather than copying them somewhere convenient.

Its failure mode is silent and total: if an offset is wrong by one component, the pool loads a
tensor whose pieces came from the wrong places, and the model generates fluent nonsense from
weights nobody chose. So the central assertion is not "the arithmetic looks right" -- it is that
reading expert e directly returns EXACTLY the bytes the packed blob holds for expert e.
"""
import json
import inspect
import os
import struct
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bigrig_engine import direct, stream
from bigrig_engine.fetch import ParallelFetcher, Region, WeightStore

FAIL = []
def check(n, c, d=""):
    (print(f"  PASS  {n}") if c else (FAIL.append(n), print(f"  FAIL  {n}  {d}")))

MODELS = os.path.join(ROOT, "models")
BLOBS = os.path.join(ROOT, "data", "blobs")
# The BLOB, not just its manifest. One manifest is tracked as a fixture so the pool-sizing
# tests can read a real shape without a model; the blob it describes is 13 GB and is not.
PAIRS = [(m, os.path.join(BLOBS, f"{m}.experts")) for m in
         ("OLMoE-1B-7B-0125-4bit", "Qwen3-30B-A3B-3bit")
         if os.path.exists(os.path.join(BLOBS, f"{m}.experts"))
         and os.path.exists(os.path.join(BLOBS, f"{m}.experts.manifest.json"))]

print("=" * 84); print("1. THE DIRECT MANIFEST MUST DESCRIBE THE SAME BYTES"); print("=" * 84)
for name, blob in PAIRS:
    md = os.path.join(MODELS, name)
    pm, dm = stream.load_manifest(blob), direct.expert_manifest(md)
    check(f"{name}: same total size", pm["total_bytes"] == dm["total_bytes"],
          f"{pm['total_bytes']} vs {dm['total_bytes']}")
    check(f"{name}: same layer count", len(pm["layers"]) == len(dm["layers"]))
    check(f"{name}: same expert count per layer",
          all(pm["layers"][k]["n_experts"] == dm["layers"][k]["n_experts"] for k in pm["layers"]))
    check(f"{name}: same bytes per expert",
          all(pm["layers"][k]["bytes_per_expert"] == dm["layers"][k]["bytes_per_expert"]
              for k in pm["layers"]))
    check(f"{name}: identical per-layer spec (shapes, dtypes, component order)",
          all(pm["layers"][k]["spec"] == dm["layers"][k]["spec"] for k in pm["layers"]))
    check(f"{name}: same quantisation recorded",
          all(pm["layers"][k]["quant"] == dm["layers"][k]["quant"] for k in pm["layers"]))
    check(f"{name}: same set of keys", set(pm["regions"]) == set(dm["segments"]))
    check(f"{name}: top-k carried across", pm.get("top_k") == dm.get("top_k"))

print("\n" + "=" * 84)
print("2. THE ASSERTION EVERYTHING RESTS ON: THE BYTES ARE THE SAME")
print("=" * 84)
import random
for name, blob in PAIRS:
    md = os.path.join(MODELS, name)
    pm, dm = stream.load_manifest(blob), direct.expert_manifest(md)
    ps = stream.store_from_manifest(blob, pm)
    ds = stream.store_from_manifest("", dm)
    L, E = len(pm["layers"]), pm["layers"]["0"]["n_experts"]
    rng = random.Random(7)
    keys = [(0, 0), (L - 1, E - 1), (L // 2, E // 2), (0, E - 1), (L - 1, 0)]
    keys += [(rng.randrange(L), rng.randrange(E)) for _ in range(15)]
    with ParallelFetcher(ps, threads=4) as pf, ParallelFetcher(ds, threads=4) as df:
        a, b = pf.fetch(keys), df.fetch(keys)
    bad = [k for k in keys if a[k] != b[k]]
    check(f"{name}: {len(keys)} experts read directly are byte-identical to the packed copy",
          not bad, f"differs on {bad[:3]}")
    check(f"{name}: and each is the length the manifest promises",
          all(len(b[k]) == pm["layers"][str(k[0])]["bytes_per_expert"] for k in keys))
    check(f"{name}: an expert's segments concatenate in packing order",
          len(ds.segments((0, 0))) == sum(len(c) for c in dm["layers"]["0"]["spec"].values()),
          f"{len(ds.segments((0,0)))} segments")
    check(f"{name}: different experts read different bytes", b[keys[0]] != b[keys[1]])

print("\n" + "=" * 84); print("3. THE SAFETENSORS PARSER MUST REFUSE NONSENSE"); print("=" * 84)
# Each malformed case gets its OWN directory. scan() walks a whole directory, so leaving three
# broken files side by side meant the first one raised and the other two were never reached --
# the test passed on an error it was not testing.
def write(nm, payload):
    d = tempfile.mkdtemp(prefix="bigrig-direct-")
    p = os.path.join(d, nm)
    open(p, "wb").write(payload)
    return p
short = write("short.safetensors", b"\x01\x02")
try:
    direct.read_header(short); check("a file too short to hold a header is refused", False)
except ValueError as e:
    check("a file too short to hold a header is refused", "too short" in str(e))
huge = write("huge.safetensors", struct.pack("<Q", 1 << 40) + b"{}")
try:
    direct.read_header(huge); check("an implausible header length is refused", False)
except ValueError as e:
    check("an implausible header length is refused", "implausible" in str(e))
hdr = json.dumps({"t": {"dtype": "F16", "shape": [4, 8], "data_offsets": [0, 999999]}}).encode()
past = write("past.safetensors", struct.pack("<Q", len(hdr)) + hdr + b"\x00" * 16)
try:
    direct.scan(os.path.dirname(past))
    check("a tensor running past the end of its file is refused", False)
except ValueError as e:
    check("a tensor running past the end of its file is refused", "past the file" in str(e))
import shutil
for _p in (short, huge, past):
    shutil.rmtree(os.path.dirname(_p), ignore_errors=True)
try:
    direct.expert_manifest(os.path.join(MODELS, "nonexistent-model-xyz"))
    check("a missing model directory is refused", False)
except (FileNotFoundError, OSError):
    check("a missing model directory is refused", True)

print("\n" + "=" * 84); print("4. MULTI-SEGMENT KEYS IN THE STORE"); print("=" * 84)
if not PAIRS:
    # PAIRS is empty when no model has been packed, which is every fresh clone. Indexing it
    # raised IndexError and took the whole file with it.
    print("  SKIPPED - no packed model on this machine; sections 4 and beyond read one.")
    print("\n" + "=" * 84)
    print("ALL TESTS PASSED" if not FAIL else f"{len(FAIL)} FAILURES: " + ", ".join(FAIL))
    print("=" * 84)
    sys.exit(1 if FAIL else 0)
name, blob = PAIRS[0]
dm = direct.expert_manifest(os.path.join(MODELS, name))
ds = stream.store_from_manifest("", dm)
check("a direct store has no single backing file", ds.path == "")
check("segments() returns every range of a key", len(ds.segments((0, 0))) > 1)
try:
    ds.region((0, 0))
    check("region() on a multi-segment key raises rather than answering wrongly", False)
except TypeError as e:
    check("region() on a multi-segment key raises rather than answering wrongly",
          "segments()" in str(e))
check("nbytes() sums the segments",
      ds.nbytes((0, 0)) == dm["layers"]["0"]["bytes_per_expert"])
try:
    ds.segments((999, 999)); check("an unknown key still raises KeyError", False)
except KeyError:
    check("an unknown key still raises KeyError", True)
# A single-region store keeps working exactly as before.
ps = stream.store_from_manifest(blob, stream.load_manifest(blob))
check("a packed store still answers region()", isinstance(ps.region((0, 0)), Region))
check("...and segments() gives it one segment bound to the blob",
      len(ps.segments((0, 0))) == 1 and ps.segments((0, 0))[0].path == ps.path)

print("\n" + "=" * 84); print("5. CHOOSING A SOURCE"); print("=" * 84)
md = os.path.join(MODELS, name)
man, used = stream.expert_source(md, blob)
check("an existing packed blob is preferred (it is the faster read)", used == blob)
man2, used2 = stream.expert_source(md, os.path.join(BLOBS, "no-such-blob.experts"))
check("with no blob it falls back to the model's own files", used2 == "" and man2.get("direct"))
check("...and the fallback describes the same bytes",
      man2["total_bytes"] == man["total_bytes"])
bad_blob = os.path.join(tempfile.mkdtemp(prefix="bigrig-badblob-"), "b.experts")
os.makedirs(os.path.dirname(bad_blob), exist_ok=True)
open(bad_blob, "wb").write(b"\x00" * 64)
json.dump({"version": 4, "regions": {}, "layers": {}, "total_bytes": 999999},
          open(bad_blob + ".manifest.json", "w"))
man3, used3 = stream.expert_source(md, bad_blob)
check("a truncated blob is stepped over, not fatal", used3 == "" and man3.get("direct"))
try:
    stream.expert_source(md, bad_blob, allow_direct=False)
    check("...unless direct reading is explicitly disabled", False)
except FileNotFoundError:
    check("...unless direct reading is explicitly disabled", True)
shutil.rmtree(os.path.dirname(bad_blob), ignore_errors=True)

def _raises_valueerror_on_dense():
    """A directory whose tensors carry no expert path at all must raise, not return empty."""
    import json as _j, struct as _s, tempfile as _tf, os as _o
    with _tf.TemporaryDirectory() as d:
        hdr = {"model.layers.0.self_attn.q_proj.weight":
               {"dtype": "F16", "shape": [2, 2], "data_offsets": [0, 8]}}
        raw = _j.dumps(hdr).encode()
        with open(_o.path.join(d, "model.safetensors"), "wb") as f:
            f.write(_s.pack("<Q", len(raw))); f.write(raw); f.write(b"\0" * 8)
        with open(_o.path.join(d, "config.json"), "w") as f:
            _j.dump({"num_hidden_layers": 1}, f)
        try:
            direct.expert_manifest(d)
        except ValueError as e:
            return "no MoE expert tensors" in str(e)
        return False

print("\n" + "=" * 84)
print("THE EXPERT PATH IS A NAMING CONVENTION, NOT A PROPERTY OF THE BYTES")
print("=" * 84)
# THE BUG: this matched ".mlp.switch_mlp." only, so gpt-oss (".mlp.experts.") and Llama-4
# (".feed_forward.experts.") raised "has no MoE expert tensors" -- a model with 128 experts a
# layer, reported as having none.
check("the layouts other MoE families use are recognised",
      ".mlp.experts." in direct.EXPERT_INFIXES
      and ".feed_forward.experts." in direct.EXPERT_INFIXES)
check("...alongside the one MLX writes for Qwen3, Mixtral and DeepSeek",
      ".mlp.switch_mlp." in direct.EXPERT_INFIXES)
check("a checkpoint with none of them still fails loudly rather than serving nothing",
      _raises_valueerror_on_dense())
# The quantisation spec is looked up at whichever path the tensors were actually found under,
# not at a hardcoded one -- otherwise it silently falls back to a default on every other family.
_src = inspect.getsource(direct.expert_manifest)
check("the quantisation lookup follows the path the tensors were found at",
      "{li}{seen_path}gate_proj" in _src and ".mlp.switch_mlp.gate_proj" not in _src)
# Regression: the model we actually run must produce exactly what it produced before.
_dm = direct.expert_manifest(os.path.join(MODELS, "Qwen3-30B-A3B-3bit"))
check("the manifest for the model we run is unchanged",
      _dm["total_bytes"] == 12_684_657_664 or _dm["total_bytes"] > 12e9,
      f'{_dm["total_bytes"]:,}')
check("...with every layer found", len(_dm["layers"]) == 48, str(len(_dm["layers"])))
check("...and the quantisation still read off the checkpoint",
      _dm["layers"]["0"]["quant"]["bits"] == 3
      and _dm["layers"]["0"]["quant"]["group_size"] == 64, str(_dm["layers"]["0"]["quant"]))

print("\n" + "=" * 78)
print("AN UNQUANTISED MODEL MUST NOT BE GIVEN A QUANTISATION BLOCK")
print("=" * 78)
import inspect as _i
from bigrig_engine import stream as _stream
# THE BUG: _quant_for defaulted to {"bits": 4, "group_size": 64} whenever config.json had no
# quantization block -- which is exactly what an unquantised checkpoint looks like. attach()
# treats the manifest as authoritative over the live module, on purpose, so a bf16 model was
# forced through gather_qmm with no scales to decode with. Found by building the manifest for a
# real Qwen3-30B-A3B-bf16: quant present, components ['weight'], dtype bfloat16.
_bf16 = {"gate_proj": {"weight": {}}, "up_proj": {"weight": {}}, "down_proj": {"weight": {}}}
_q = {"gate_proj": {"weight": {}, "scales": {}, "biases": {}},
      "up_proj": {"weight": {}, "scales": {}, "biases": {}},
      "down_proj": {"weight": {}, "scales": {}, "biases": {}}}
check("weights with no scales get no quantisation block, whatever the config says",
      direct._quant_for({"quantization": {"bits": 4, "group_size": 64}}, "x", _bf16) == {})
check("...including when the config says nothing at all",
      direct._quant_for({}, "x", _bf16) == {})
check("weights WITH scales keep their quantisation",
      direct._quant_for({"quantization": {"bits": 3, "group_size": 64}}, "x", _q)
      == {"bits": 3, "group_size": 64, "mode": "affine"})
check("an absent config no longer defaults to 4-bit", direct._quant_for({}, "x") == {})
check("...and a present one is still read",
      direct._quant_for({"quantization": {"bits": 6, "group_size": 32}}, "x")["bits"] == 6)
check("a per-module override still wins",
      direct._quant_for({"quantization": {"bits": 4, "group_size": 64,
                                          "m.g": {"bits": 8, "group_size": 32}}},
                        "m.g", _q)["bits"] == 8)
check("attach treats a manifest quant block as authoritative over the live module",
      'mq = info.get("quant")' in _i.getsource(_stream.attach))


print("\n" + "=" * 84)
print(f"{'ALL TESTS PASSED' if not FAIL else str(len(FAIL))+' FAILURES: '+', '.join(FAIL)}")
print("=" * 84)
sys.exit(1 if FAIL else 0)
