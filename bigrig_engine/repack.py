"""COMPONENT 7 — repack a checkpoint so one expert is one contiguous read.

THE PROBLEM, MEASURED ON A REAL CHECKPOINT
    MLX fuses an MoE layer's experts into `switch_mlp` tensors whose FIRST axis is the expert
    index, and splits each projection into three pieces. OLMoE-1B-7B-0125-4bit, layer 0:

        switch_mlp.gate_proj.weight   [64, 1024, 256]  U32   67.11 MB
        switch_mlp.gate_proj.scales   [64, 1024,  32]  F16    4.19 MB
        switch_mlp.gate_proj.biases   [64, 1024,  32]  F16    4.19 MB
        ... the same for up_proj and down_proj

    Expert e's slice of each tensor is contiguous, but the NINE tensors sit at nine unrelated
    offsets. So fetching one expert means nine reads, six of which are ~65 KB. Small scattered
    reads are exactly what the fetch engine's parallelism cannot rescue -- the device spends its
    time on request overhead rather than bytes.

WHAT THIS DOES
    Streams the checkpoint and writes a new file in which all nine pieces of a given
    (layer, expert) sit in one contiguous block, plus a sidecar describing the layout. One
    expert then becomes ONE pread, which is what `ParallelFetcher` is built to issue.

CORRECTNESS IS THE WHOLE JOB
    A repacker that corrupts one byte produces a model that is subtly wrong in a way no quality
    meter can attribute and no test of the engine would catch. So every block is verified
    byte-for-byte against the source before the output is accepted, and `verify()` re-checks a
    finished file independently of the code that wrote it.

MEMORY
    Streams tensor by tensor and expert by expert. Peak resident is one expert block (a few MB),
    never the model -- a 3.89 GB checkpoint repacks inside a few hundred MB.
"""
from __future__ import annotations

import json
import os
import re
import struct
from dataclasses import asdict, dataclass

DTYPE_BYTES = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2, "I64": 8, "I32": 4,
               "U32": 4, "I16": 2, "U16": 2, "I8": 1, "U8": 1, "BOOL": 1}
PIECES = ("weight", "scales", "biases")
PROJ = ("gate_proj", "up_proj", "down_proj")


@dataclass(frozen=True)
class Piece:
    """One tensor slice belonging to an expert, inside its block."""
    name: str
    rel_offset: int
    length: int
    shape: list
    dtype: str


def read_header(path: str):
    """(header dict, byte offset where tensor data begins)."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    return hdr, 8 + n


def _nbytes(shape, dtype):
    n = DTYPE_BYTES.get(dtype)
    if n is None:
        raise ValueError(f"unknown dtype {dtype!r}; refusing to guess its size")
    t = n
    for d in shape:
        t *= d
    return t


def find_expert_tensors(hdr: dict, n_experts: int) -> dict:
    """{layer: {tensor_name: info}} for tensors whose first axis is the expert index.

    Selection is STRUCTURAL, not by name: a tensor qualifies only if its leading dimension is the
    expert count AND it has at least three dimensions. Matching on names like "switch_mlp" would
    break on the next architecture.

    THE THREE-DIMENSION RULE IS LOAD-BEARING, and a smoke test on the real checkpoint is what
    found it. The ROUTER (`mlp.gate.weight`, shape [64, 256]) also leads with the expert count,
    because it emits one logit per expert -- but it is read WHOLE on every token, not per expert.
    Packing its rows into per-expert blocks would force the router to be reassembled from 64
    fragments every token, which is the exact problem this component exists to remove.

    An expert's own weights are a 2-D matrix, so with the expert axis prepended they are 3-D.
    The router's are 1-D per expert, so it is 2-D. That distinction is a property of what the
    tensors MEAN, not of how they happen to be named.
    """
    out: dict = {}
    for name, info in hdr.items():
        if name == "__metadata__":
            continue
        shape = info.get("shape") or []
        if len(shape) < 3 or shape[0] != n_experts:
            continue
        m = re.search(r"layers\.(\d+)\.", name)
        if m is None:
            continue
        out.setdefault(int(m.group(1)), {})[name] = info
    return out


def repack(src_dir: str, out_path: str, n_experts: int, verify_every: bool = True,
           progress=None) -> dict:
    """Write an expert-contiguous file plus its layout sidecar. Returns the manifest."""
    src_dir = os.path.expanduser(src_dir)
    out_path = os.path.expanduser(out_path)
    shards = sorted(f for f in os.listdir(src_dir) if f.endswith(".safetensors"))
    if not shards:
        raise FileNotFoundError(f"no .safetensors in {src_dir}")

    layout, blocks_written, bytes_written = {}, 0, 0
    tensors_used = set()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    with open(out_path, "wb") as out:
        for shard in shards:
            sp = os.path.join(src_dir, shard)
            hdr, data0 = read_header(sp)
            per_layer = find_expert_tensors(hdr, n_experts)
            if not per_layer:
                continue
            with open(sp, "rb") as src:
                for layer in sorted(per_layer):
                    names = sorted(per_layer[layer])
                    tensors_used |= set(names)
                    for e in range(n_experts):
                        block_start = out.tell()
                        pieces, rel = [], 0
                        for name in names:
                            info = hdr[name]
                            shape = info["shape"]
                            dtype = info["dtype"]
                            per_expert = _nbytes(shape[1:], dtype)
                            base = data0 + info["data_offsets"][0]
                            src.seek(base + e * per_expert)
                            buf = src.read(per_expert)
                            if len(buf) != per_expert:
                                raise IOError(
                                    f"short read on {name} expert {e}: "
                                    f"{len(buf)} of {per_expert} bytes")
                            out.write(buf)
                            pieces.append(Piece(name, rel, per_expert, list(shape[1:]), dtype))
                            rel += per_expert
                        layout[f"{layer}/{e}"] = {
                            "offset": block_start, "length": rel,
                            "pieces": [asdict(p) for p in pieces]}
                        blocks_written += 1
                        bytes_written += rel
                        if progress and blocks_written % 200 == 0:
                            progress(blocks_written, bytes_written)

    manifest = {"source": src_dir, "shards": shards, "n_experts": n_experts,
                "blocks": blocks_written, "bytes": bytes_written,
                "tensors_packed": sorted(tensors_used), "layout": layout}
    with open(out_path + ".json", "w") as f:
        json.dump(manifest, f)

    if verify_every:
        bad = verify(src_dir, out_path, n_experts, sample=None)
        if bad:
            raise ValueError(f"repack verification failed on {len(bad)} blocks: {bad[:3]}")
    return manifest


def verify(src_dir: str, packed_path: str, n_experts: int, sample: int | None = 64,
           seed: int = 0) -> list:
    """Re-read blocks from the packed file and compare against the SOURCE, byte for byte.

    Deliberately re-derives offsets from the source headers rather than trusting the manifest,
    so a bug that wrote the wrong bytes AND recorded them consistently is still caught.
    """
    import random
    src_dir = os.path.expanduser(src_dir)
    packed_path = os.path.expanduser(packed_path)
    with open(packed_path + ".json") as f:
        man = json.load(f)
    keys = list(man["layout"])
    if sample is not None and sample < len(keys):
        keys = random.Random(seed).sample(keys, sample)

    headers = {}
    for shard in man["shards"]:
        sp = os.path.join(src_dir, shard)
        headers[sp] = read_header(sp)

    bad = []
    with open(packed_path, "rb") as pf:
        for key in keys:
            layer, e = key.split("/")
            e = int(e)
            blk = man["layout"][key]
            pf.seek(blk["offset"])
            got = pf.read(blk["length"])
            if len(got) != blk["length"]:
                bad.append((key, "short read from packed file"))
                continue
            for p in blk["pieces"]:
                found = False
                for sp, (hdr, data0) in headers.items():
                    if p["name"] not in hdr:
                        continue
                    info = hdr[p["name"]]
                    per = _nbytes(info["shape"][1:], info["dtype"])
                    if per != p["length"]:
                        bad.append((key, f"{p['name']} length {p['length']} != source {per}"))
                        found = True
                        break
                    with open(sp, "rb") as src:
                        src.seek(data0 + info["data_offsets"][0] + e * per)
                        ref = src.read(per)
                    if got[p["rel_offset"]:p["rel_offset"] + per] != ref:
                        bad.append((key, f"{p['name']} bytes differ"))
                    found = True
                    break
                if not found:
                    bad.append((key, f"{p['name']} not found in any source shard"))
    return bad


def load_layout(packed_path: str):
    """Layout for `WeightStore`: one Region per (layer, expert)."""
    from .fetch import Region
    with open(os.path.expanduser(packed_path) + ".json") as f:
        man = json.load(f)
    out = {}
    for key, blk in man["layout"].items():
        layer, e = key.split("/")
        out[(int(layer), int(e))] = Region(blk["offset"], blk["length"])
    return out, man


def unpack_block(block: bytes, key: str, manifest: dict) -> dict:
    """Split one fetched block back into its named tensor slices."""
    blk = manifest["layout"][key]
    if len(block) != blk["length"]:
        raise ValueError(f"block for {key} is {len(block)} bytes, expected {blk['length']}")
    return {p["name"]: block[p["rel_offset"]:p["rel_offset"] + p["length"]]
            for p in blk["pieces"]}
