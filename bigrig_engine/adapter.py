"""COMPONENT 11 — describe an arbitrary MoE checkpoint well enough to run it.

WHAT THE ENGINE NEEDS TO KNOW, AND WHERE IT COMES FROM
    expert count, top-k, layer count, the routing rule, whether there is a shared expert, and
    WHICH tensors are experts. The first few come from config.json. The last one does not, and
    is the part that breaks when a new architecture appears.

THE RULE THIS USES, AND WHY IT IS STRUCTURAL
    Expert tensors are found by SHAPE: leading dimension equal to the expert count, and at least
    three dimensions. Not by matching names like "switch_mlp", which is a bet on one library's
    naming. The three-dimension part is load-bearing and was found by inspecting a real file:
    the ROUTER also leads with the expert count (it emits one logit per expert) but is 2-D, and
    treating it as expert data would force it to be reassembled from E fragments every token.

FAIL LOUDLY, NEVER GUESS
    An adapter that quietly mis-describes a model produces an engine that computes with the
    wrong weights and a quality meter that cannot say why. Every field is either read from the
    config, derived and then CHECKED against the tensors, or refused.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

from .repack import _nbytes, find_expert_tensors, read_header

# config keys differ between architectures; the engine should not care which one it is reading
EXPERT_KEYS = ("num_experts", "num_local_experts", "n_routed_experts", "moe_num_experts")
TOPK_KEYS = ("num_experts_per_tok", "moe_top_k", "num_experts_per_token", "top_k")
LAYER_KEYS = ("num_hidden_layers", "n_layers", "num_layers")
SHARED_KEYS = ("num_shared_experts", "n_shared_experts", "moe_num_shared_experts",
               "shared_expert_intermediate_size")


def _first(cfg: dict, keys, default=None):
    for k in keys:
        if k in cfg and cfg[k] is not None:
            return cfg[k]
    return default


@dataclass
class MoESpec:
    """Everything the engine needs about one checkpoint, all of it checked against the file."""
    path: str
    model_type: str
    n_experts: int
    top_k: int
    n_layers: int
    moe_layers: list
    shared_experts: int
    routing: str
    expert_bytes: int
    expert_tensors_per_layer: int
    total_expert_bytes: int
    dense_bytes: int
    shards: list
    quantization: dict = field(default_factory=dict)

    @property
    def active_bytes_per_token(self) -> int:
        """Expert bytes read per token: (top_k + shared) experts at every MoE layer."""
        return (self.top_k + self.shared_experts) * self.expert_bytes * len(self.moe_layers)

    def summary(self) -> str:
        return (f"{self.model_type}: {self.n_experts} experts, top-{self.top_k}"
                f"{f' + {self.shared_experts} shared' if self.shared_experts else ''}, "
                f"{len(self.moe_layers)} of {self.n_layers} layers are MoE, "
                f"{self.routing} routing, {self.expert_bytes/1e6:.2f} MB/expert, "
                f"{self.total_expert_bytes/1e9:.2f} GB of experts, "
                f"{self.active_bytes_per_token/1e9:.2f} GB active per token")


def infer_routing(cfg: dict) -> str:
    """softmax top-k, or sigmoid with a learned bias and group-limited selection.

    Reported rather than assumed: the toll's scale and the meter's calibration both depend on it,
    and this project measured the gamma-to-lambda ratio differing 4.53 vs 6.89 between the two.
    """
    fn = str(cfg.get("score_function") or cfg.get("scoring_func") or "").lower()
    if "sigmoid" in fn:
        base = "sigmoid"
    elif "softmax" in fn:
        base = "softmax"
    else:
        base = "sigmoid" if cfg.get("n_group") or cfg.get("topk_group") else "softmax"
    bits = [base]
    if cfg.get("n_group") or cfg.get("topk_group"):
        bits.append(f"group-limited({cfg.get('topk_group')}/{cfg.get('n_group')})")
    if cfg.get("norm_topk_prob"):
        bits.append("normalised")
    return "+".join(bits)


def describe(model_dir: str) -> MoESpec:
    """Read a checkpoint and return a checked description, or raise saying exactly what is wrong."""
    d = os.path.expanduser(model_dir)
    cfgp = os.path.join(d, "config.json")
    if not os.path.exists(cfgp):
        raise FileNotFoundError(f"no config.json in {d}")
    cfg = json.load(open(cfgp))

    n_experts = _first(cfg, EXPERT_KEYS)
    if not n_experts:
        raise ValueError(
            f"{d}: no expert count found under any of {EXPERT_KEYS}. This may not be an MoE, or "
            f"it uses a key this adapter does not know. Refusing to guess.")
    top_k = _first(cfg, TOPK_KEYS)
    if not top_k:
        raise ValueError(f"{d}: expert count is {n_experts} but no top-k under {TOPK_KEYS}.")
    n_layers = _first(cfg, LAYER_KEYS, 0)

    shared = _first(cfg, SHARED_KEYS, 0)
    if isinstance(shared, int) and shared > 8:
        shared = 1          # some configs give an intermediate SIZE, not a count
    shared = int(shared or 0)

    shards = sorted(f for f in os.listdir(d) if f.endswith(".safetensors"))
    if not shards:
        raise FileNotFoundError(f"{d}: no .safetensors files")

    moe_layers: set = set()
    per_layer_names: dict = {}
    expert_bytes = 0
    total_bytes = 0
    all_bytes = 0
    for sh in shards:
        hdr, _ = read_header(os.path.join(d, sh))
        for name, info in hdr.items():
            if name == "__metadata__":
                continue
            all_bytes += _nbytes(info["shape"], info["dtype"])
        per = find_expert_tensors(hdr, int(n_experts))
        for layer, tensors in per.items():
            moe_layers.add(layer)
            per_layer_names.setdefault(layer, set()).update(tensors)
            b = sum(_nbytes(i["shape"][1:], i["dtype"]) for i in tensors.values())
            expert_bytes = max(expert_bytes, b)
            total_bytes += b * int(n_experts)

    if not moe_layers:
        raise ValueError(
            f"{d}: config declares {n_experts} experts but NO tensor has that leading dimension "
            f"with 3+ dims. Either the expert count is wrong or this layout is unsupported. "
            f"Refusing to proceed rather than packing the wrong tensors.")

    counts = {len(v) for v in per_layer_names.values()}
    if len(counts) != 1:
        raise ValueError(
            f"{d}: MoE layers disagree on tensor count {sorted(counts)}. A ragged layout would "
            f"produce expert blocks of differing size; refusing.")

    return MoESpec(
        path=d, model_type=cfg.get("model_type", "unknown"), n_experts=int(n_experts),
        top_k=int(top_k), n_layers=int(n_layers) or (max(moe_layers) + 1),
        moe_layers=sorted(moe_layers), shared_experts=shared, routing=infer_routing(cfg),
        expert_bytes=expert_bytes, expert_tensors_per_layer=counts.pop(),
        total_expert_bytes=total_bytes, dense_bytes=max(0, all_bytes - total_bytes),
        shards=shards, quantization=cfg.get("quantization", {}) or {})


def plan_for(spec: MoESpec, profile: dict, miss_rate: float) -> dict:
    """What this checkpoint can do on this host, using both measured descriptions."""
    from .calibrate import plan
    return plan(profile, model_gb=spec.total_expert_bytes / 1e9,
                active_gb=spec.active_bytes_per_token / 1e9, miss_rate=miss_rate)
