"""Adversarial tests for the model adapter.

The adapter's job is to describe a checkpoint. Its failure mode is describing one WRONGLY and
confidently -- the engine then packs the wrong tensors, fetches the wrong bytes, and the quality
meter reports degradation it cannot attribute. So most of these tests are about refusing.
"""
import json
import os
import struct
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bigrig_engine.adapter import MoESpec, describe, infer_routing, plan_for

FAIL = []
def check(n, c, d=""):
    (print(f"  PASS  {n}") if c else (FAIL.append(n), print(f"  FAIL  {n}  {d}")))


def write_ckpt(d, cfg, tensors):
    os.makedirs(d, exist_ok=True)
    json.dump(cfg, open(os.path.join(d, "config.json"), "w"))
    hdr, blobs, off = {}, [], 0
    for name, arr in tensors.items():
        b = arr.tobytes()
        dt = {np.dtype("float16"): "F16", np.dtype("uint32"): "U32"}[arr.dtype]
        hdr[name] = {"dtype": dt, "shape": list(arr.shape), "data_offsets": [off, off + len(b)]}
        blobs.append(b); off += len(b)
    raw = json.dumps(hdr).encode()
    with open(os.path.join(d, "model.safetensors"), "wb") as f:
        f.write(struct.pack("<Q", len(raw))); f.write(raw)
        for b in blobs: f.write(b)
    return d


rng = np.random.default_rng(0)
TMP = tempfile.mkdtemp(prefix="bigrig_adapter_")

print("=" * 78); print("1. THE REAL CHECKPOINTS ARE DESCRIBED CORRECTLY"); print("=" * 78)
real = {}
for name, exp_experts, exp_k in (("OLMoE-1B-7B-0125-4bit", 64, 8), ("Ling-mini-2.0-3bit", 256, 8)):
    p = os.path.join(ROOT, f"models/{name}")
    if not os.path.exists(p):
        print(f"  SKIP  {name} not present"); continue
    s = describe(p)
    real[name] = s
    check(f"{name}: expert count", s.n_experts == exp_experts, f"{s.n_experts}")
    check(f"{name}: top-k", s.top_k == exp_k, f"{s.top_k}")
    check(f"{name}: every MoE layer has the same tensor count",
          s.expert_tensors_per_layer == 9, str(s.expert_tensors_per_layer))
    check(f"{name}: active bytes per token is positive and below the whole model",
          0 < s.active_bytes_per_token < s.total_expert_bytes)
if "Ling-mini-2.0-3bit" in real:
    s = real["Ling-mini-2.0-3bit"]
    check("Ling's shared expert is detected", s.shared_experts == 1, str(s.shared_experts))
    check("Ling's group-limited sigmoid routing is detected",
          "sigmoid" in s.routing and "group-limited" in s.routing, s.routing)
if "OLMoE-1B-7B-0125-4bit" in real:
    check("OLMoE is detected as softmax with no shared expert",
          real["OLMoE-1B-7B-0125-4bit"].routing.startswith("softmax")
          and real["OLMoE-1B-7B-0125-4bit"].shared_experts == 0,
          real["OLMoE-1B-7B-0125-4bit"].routing)

print("\n" + "=" * 78); print("2. IT MUST REFUSE WHAT IT CANNOT DESCRIBE"); print("=" * 78)
E, H, I = 8, 4, 6
experts = {f"model.layers.{l}.mlp.switch_mlp.{p}.weight":
           rng.integers(0, 2**31, (E, H, I), dtype=np.uint32)
           for l in range(2) for p in ("gate_proj", "up_proj", "down_proj")}

dense = write_ckpt(os.path.join(TMP, "dense"), {"model_type": "llama", "num_hidden_layers": 2},
                   {"model.layers.0.self_attn.q_proj.weight":
                    rng.integers(0, 2**31, (H, I), dtype=np.uint32)})
try:
    describe(dense); check("a DENSE model is refused, not mis-described", False)
except ValueError as e:
    check("a DENSE model is refused, not mis-described", "no expert count" in str(e))

nok = write_ckpt(os.path.join(TMP, "nok"), {"model_type": "x", "num_experts": E}, experts)
try:
    describe(nok); check("experts declared but no top-k is refused", False)
except ValueError as e:
    check("experts declared but no top-k is refused", "top-k" in str(e))

lying = write_ckpt(os.path.join(TMP, "lying"),
                   {"model_type": "x", "num_experts": 999, "num_experts_per_tok": 2}, experts)
try:
    describe(lying); check("a config that LIES about the expert count is refused", False)
except ValueError as e:
    check("a config that LIES about the expert count is refused",
          "NO tensor has that leading dimension" in str(e))

ragged = dict(experts)
ragged["model.layers.1.mlp.switch_mlp.extra.weight"] = rng.integers(0, 2**31, (E, H, I),
                                                                    dtype=np.uint32)
rg = write_ckpt(os.path.join(TMP, "ragged"),
                {"model_type": "x", "num_experts": E, "num_experts_per_tok": 2}, ragged)
try:
    describe(rg); check("layers with DIFFERING tensor counts are refused", False)
except ValueError as e:
    check("layers with DIFFERING tensor counts are refused", "disagree on tensor count" in str(e))

try:
    describe(os.path.join(TMP, "nothing_here"))
    check("a missing directory is refused", False)
except (FileNotFoundError, NotADirectoryError):
    check("a missing directory is refused", True)

noshards = os.path.join(TMP, "noshards"); os.makedirs(noshards, exist_ok=True)
json.dump({"model_type": "x", "num_experts": E, "num_experts_per_tok": 2},
          open(os.path.join(noshards, "config.json"), "w"))
try:
    describe(noshards); check("a config with no weight files is refused", False)
except FileNotFoundError:
    check("a config with no weight files is refused", True)

print("\n" + "=" * 78); print("3. THE ROUTER MUST NOT BE COUNTED AS AN EXPERT"); print("=" * 78)
with_router = dict(experts)
for l in range(2):
    with_router[f"model.layers.{l}.mlp.gate.weight"] = rng.integers(0, 2**31, (E, I),
                                                                    dtype=np.uint32)
wr = write_ckpt(os.path.join(TMP, "wr"),
                {"model_type": "x", "num_experts": E, "num_experts_per_tok": 2}, with_router)
s = describe(wr)
check("the 2-D router is excluded from the expert tensor count",
      s.expert_tensors_per_layer == 3, str(s.expert_tensors_per_layer))
check("...so expert_bytes counts only real expert weights",
      s.expert_bytes == 3 * H * I * 4, f"{s.expert_bytes} vs {3*H*I*4}")

print("\n" + "=" * 78); print("4. ALTERNATIVE CONFIG SPELLINGS"); print("=" * 78)
for key in ("num_local_experts", "n_routed_experts", "moe_num_experts"):
    d = write_ckpt(os.path.join(TMP, "k_" + key),
                   {"model_type": "x", key: E, "num_experts_per_tok": 2}, experts)
    check(f"expert count under `{key}` is understood", describe(d).n_experts == E)
for key in ("moe_top_k", "num_experts_per_token"):
    d = write_ckpt(os.path.join(TMP, "t_" + key),
                   {"model_type": "x", "num_experts": E, key: 3}, experts)
    check(f"top-k under `{key}` is understood", describe(d).top_k == 3)

check("routing is inferred as sigmoid when grouping is present",
      "sigmoid" in infer_routing({"n_group": 8, "topk_group": 4}))
check("routing defaults to softmax with no hints",
      infer_routing({}) == "softmax")
check("an explicit score_function wins over the heuristic",
      infer_routing({"score_function": "softmax", "n_group": 8}).startswith("softmax"))

print("\n" + "=" * 78); print("5. IT COMPOSES WITH THE HOST PROFILE"); print("=" * 78)
prof = json.load(open(os.path.join(ROOT, "data/results/host_profile.json")))
if real:
    s = list(real.values())[0]
    r = plan_for(s, prof, miss_rate=0.05)
    check("plan_for returns a token rate", r["tok_s"] > 0, str(r))
    check("a model that fits reports its miss rate as clamped to zero",
          (not r["fits"]) or r["miss_rate_clamped"] is True, str(r))
    big = MoESpec(**{**s.__dict__, "total_expert_bytes": 400 * 10**9})
    rb = plan_for(big, prof, miss_rate=0.05)
    check("a 400 GB model does not fit and keeps its miss rate", not rb["fits"]
          and rb["miss_rate_used"] == 0.05, str(rb))
    check("...and is slower than the one that fits", rb["tok_s"] < r["tok_s"])

import shutil
shutil.rmtree(TMP, ignore_errors=True)
print("\n" + "=" * 78)
print(f"{'ALL TESTS PASSED' if not FAIL else str(len(FAIL)) + ' FAILURES: ' + ', '.join(FAIL)}")
print("=" * 78)
sys.exit(1 if FAIL else 0)
