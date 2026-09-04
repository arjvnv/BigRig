"""Read experts straight out of the model's own safetensors. No second copy on disk.

WHY THIS REPLACED PACKING AS THE DEFAULT
    `pack_experts` rewrites every expert into one contiguous blob, which makes admitting an
    expert a single read instead of nine. It also means the user now stores their model TWICE:
    26 GB on disk for a 13.4 GB download. That is the loudest complaint a first-time user has,
    and it arrives before they have seen anything work.

    They do not have to be copied. A safetensors tensor of shape (E, ...) stores expert e's slice
    contiguously at a computable offset, so the bytes are already addressable exactly where they
    sit. What is lost is contiguity ACROSS an expert's nine component tensors -- gate/up/down x
    weight/scales/biases -- so admitting one expert becomes nine reads rather than one.

    Packing is therefore still available and still faster; it is just no longer the price of
    entry. `bigrig prepare --pack` buys the speed back for anyone who has the disk to spare.

THE INVARIANT THAT MAKES THIS SAFE
    Reading expert e directly must produce EXACTLY the bytes the packed blob holds for expert e.
    Same order, same length, byte for byte. If that ever diverges, the pool loads a tensor whose
    components come from the wrong places and the model generates fluent nonsense. test_direct.py
    asserts it against a real packed blob rather than trusting the arithmetic.
"""
from __future__ import annotations

import glob
import json
import os
import struct

from .stream import COMPONENTS, MLP_PROJECTIONS, PROJECTIONS

# Where each MoE family writes its fused expert tensors. A checkpoint uses exactly one of these;
# which one is a naming convention, not a difference in the bytes.
# WHERE A MODEL KEEPS ITS STACKED EXPERTS, WHICH IS NOT A SETTLED CONVENTION.
#     A stacked expert tensor is (n_experts, out, in) and every family agrees on that shape while
#     disagreeing about the path. Three were known; `.ffn.switch_mlp.` was found by pointing
#     `bigrig doctor` at DeepSeek-V4-Flash -- 16,353 downloads in a month -- and being told it
#     "is not a mixture-of-experts model". It is: 256 experts, correctly stacked, at
#     `model.layers.N.ffn.switch_mlp.gate_proj.weight` with shape [256, 2048, 512]. One missing
#     string, and a model nobody could run.
#
#     NOT a general prefix match. `shared_experts` also lives under `.ffn.` and is NOT routed --
#     it runs for every token, so streaming it would fetch it every time and gain nothing. The
#     infixes are exact for that reason.
EXPERT_INFIXES = (".mlp.switch_mlp.", ".mlp.experts.", ".feed_forward.experts.",
                  ".ffn.switch_mlp.", ".ffn.experts.", ".mixer.switch_mlp.",
                  ".block_sparse_moe.switch_mlp.", ".block_sparse_moe.experts.")

# safetensors dtype names -> (mlx dtype name, bytes per element)
DTYPES = {"F64": ("float64", 8), "F32": ("float32", 4), "F16": ("float16", 2),
          "BF16": ("bfloat16", 2), "I64": ("int64", 8), "I32": ("int32", 4),
          "I16": ("int16", 2), "I8": ("int8", 1), "U8": ("uint8", 1),
          "U32": ("uint32", 4), "U16": ("uint16", 2), "BOOL": ("bool_", 1)}


class Segment:
    """One contiguous byte range in one file."""
    __slots__ = ("path", "offset", "length")

    def __init__(self, path: str, offset: int, length: int):
        if offset < 0 or length <= 0:
            raise ValueError(f"invalid segment {path} offset={offset} length={length}")
        self.path, self.offset, self.length = path, offset, length

    def __repr__(self):
        return f"Segment({os.path.basename(self.path)}, {self.offset}, {self.length})"

    def __eq__(self, o):
        return (isinstance(o, Segment) and (self.path, self.offset, self.length) ==
                (o.path, o.offset, o.length))


def read_header(path: str) -> tuple:
    """(tensor metadata, byte offset where tensor data begins)."""
    with open(path, "rb") as f:
        raw = f.read(8)
        if len(raw) != 8:
            raise ValueError(f"{path} is too short to be a safetensors file")
        n = struct.unpack("<Q", raw)[0]
        if not 0 < n < (1 << 30):
            raise ValueError(f"{path}: implausible safetensors header length {n}")
        hdr = json.loads(f.read(n))
    if not isinstance(hdr, dict):
        raise ValueError(f"{path}: safetensors header is not an object")
    return hdr, 8 + n


def scan(model_dir: str) -> dict:
    """{tensor name: (file, absolute offset, byte length, dtype, shape)} across every shard."""
    model_dir = os.path.expanduser(model_dir)
    files = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
    if not files:
        raise FileNotFoundError(f"no safetensors in {model_dir}")
    out = {}
    for p in files:
        hdr, base = read_header(p)
        size = os.path.getsize(p)
        for name, e in hdr.items():
            if name == "__metadata__":
                continue
            a, b = e["data_offsets"]
            if b < a or base + b > size:
                raise ValueError(
                    f"{p}: tensor {name} claims bytes {a}..{b}, which runs past the file "
                    f"({size} bytes). Refusing to read past the end.")
            out[name] = (p, base + a, b - a, e["dtype"], tuple(e["shape"]))
    return out


def _quant_for(cfg: dict, module_path: str, spec: dict | None = None) -> dict:
    """This layer's quantisation, or {} if the weights are not quantised.

    THE EMPTY CASE IS THE POINT, AND IT USED TO RETURN 4-BIT INSTEAD.
        This defaulted to {"bits": 4, "group_size": 64} whenever config.json had no
        quantization block -- which is exactly what an UNQUANTISED checkpoint looks like. The
        manifest then carried a quant block for bf16 weights, and `attach` treats the manifest
        as the authority precisely so a requantised blob cannot be decoded with the module's
        stale bits:

            mq = info.get("quant")
            if mq:  quant, qp = True, {...}

        So a bf16 model was forced down the quantised kernel, `gather_qmm`, with no scales and
        no biases to decode with. Found by running Qwen3-30B-A3B-bf16 through the manifest
        builder: 'quant' block present: True, components ['weight'], dtype bfloat16.

    The tensors decide, not the config. A quantised expert has `scales` beside its `weight`;
    an unquantised one does not, whatever config.json does or does not say.
    """
    if spec is not None:
        has_scales = any("scales" in comps for comps in spec.values())
        if not has_scales:
            return {}
    q = cfg.get("quantization") or cfg.get("quantization_config") or {}
    if not q:
        return {}
    v = q.get(module_path)
    if isinstance(v, dict):
        return {"bits": int(v["bits"]), "group_size": int(v["group_size"]),
                "mode": v.get("mode", "affine")}
    return {"bits": int(q.get("bits", 4)), "group_size": int(q.get("group_size", 64)),
            "mode": q.get("mode", "affine")}


def expert_manifest(model_dir: str) -> dict:
    """A manifest shaped like a packed blob's, but pointing INTO the original files.

    Same keys, same per-layer spec, same component order -- so everything downstream (the pool,
    the fetcher, the compressor) works without knowing where the bytes came from.
    """
    model_dir = os.path.expanduser(model_dir)
    tensors = scan(model_dir)
    with open(os.path.join(model_dir, "config.json")) as f:
        cfg = json.load(f)

    layers, seen_path, foreign = {}, None, set()
    for name in tensors:
        # Matching one spelling made every other MoE family look like a model with no experts.
        for infix in EXPERT_INFIXES:
            if infix in name:
                break
        else:
            continue
        head, tail = name.split(infix, 1)
        li = int(head.rsplit(".", 1)[1])
        proj, comp = tail.rsplit(".", 1)
        if proj not in PROJECTIONS and proj not in MLP_PROJECTIONS:
            foreign.add(proj)                  # experts found, but not a layout we compute
            continue
        if comp not in COMPONENTS:
            continue
        seen_path = infix
        layers.setdefault(li, {}).setdefault(proj, {})[comp] = name
    if not layers:
        if foreign:
            # Same sentence preflight gives for a hub id, so a local directory does not get a
            # worse answer than a repo name would have.
            raise ValueError(
                f"{model_dir} is a mixture-of-experts model, but its experts use a "
                f"{'/'.join(sorted(foreign))} layout that this engine cannot stream yet -- it "
                f"streams the gate/up/down (SwiGLU) and fc1/fc2 layouts. Support for this "
                f"layout is on the list.")
        raise ValueError(
            f"{model_dir} has no MoE expert tensors (looked for "
            f"{', '.join('*' + i + '*' for i in EXPERT_INFIXES)})")

    man = {"version": 4, "direct": True, "model_dir": model_dir,
           "segments": {}, "layers": {}, "total_bytes": 0}
    tk = cfg.get("num_experts_per_tok") or cfg.get("moe_topk")
    if isinstance(tk, int) and tk > 0:
        man["top_k"] = tk

    total = 0
    for li in sorted(layers):
        spec, n_experts, per_expert = {}, None, 0
        # ordered exactly as pack_experts writes: projection outer, component inner
        ordered = []
        # WHICHEVER SHAPE THIS CHECKPOINT CARRIES, in the order `pack_experts` writes it. Asking
        # for gate/up/down on a model that has fc1/fc2 raised "layer 0 is missing gate_proj; the
        # checkpoint is incomplete", which is an accusation against a perfectly good file.
        _shape = PROJECTIONS if "gate_proj" in layers[li] else MLP_PROJECTIONS
        for proj in _shape:
            comps = layers[li].get(proj)
            if not comps:
                raise ValueError(f"layer {li} is missing {proj}; the checkpoint is incomplete")
            spec[proj] = {}
            for comp in COMPONENTS:
                nm = comps.get(comp)
                if nm is None:
                    continue
                path, off, blen, dt, shape = tensors[nm]
                if len(shape) < 2:
                    raise ValueError(f"{nm} has shape {shape}; expert tensors lead with E")
                if n_experts is None:
                    n_experts = int(shape[0])
                elif int(shape[0]) != n_experts:
                    raise ValueError(
                        f"layer {li}: {nm} has {shape[0]} experts but a sibling has "
                        f"{n_experts}. Refusing to guess which is right.")
                mx_dt, isize = DTYPES[dt]
                stride = 1
                for x in shape[1:]:
                    stride *= int(x)
                stride *= isize
                if stride * n_experts != blen:
                    raise ValueError(
                        f"{nm}: {n_experts} x {stride} != {blen} bytes on disk. The header and "
                        f"the data disagree.")
                spec[proj][comp] = {"shape": [int(x) for x in shape[1:]], "dtype": mx_dt,
                                    "nbytes": stride}
                ordered.append((path, off, stride))
                per_expert += stride
        for e in range(n_experts):
            man["segments"][f"{li}:{e}"] = [[p, o + e * s, s] for p, o, s in ordered]
        man["layers"][str(li)] = {
            "n_experts": n_experts, "spec": spec, "bytes_per_expert": per_expert,
            "quant": _quant_for(cfg, f"model.layers.{li}{seen_path}gate_proj", spec)}
        total += per_expert * n_experts
    man["total_bytes"] = total
    return man


def store(model_dir: str):
    """A WeightStore reading the model's own files, with nothing copied."""
    from .stream import store_from_manifest
    return store_from_manifest("", expert_manifest(model_dir))
