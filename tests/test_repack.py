"""Adversarial tests for the repacker.

This component rewrites model weights. If it corrupts a single byte the model is subtly wrong in
a way no quality meter can attribute, no engine test would catch, and no user could diagnose. So
the tests are weighted heavily toward "does it ever return different bytes", and the verifier
itself is tested with deliberate corruption -- a checker that cannot fail proves nothing.
"""
import json
import os
import struct
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bigrig_engine.fetch import ParallelFetcher, WeightStore
from bigrig_engine.repack import (_nbytes, find_expert_tensors, load_layout, read_header,
                                    repack, unpack_block, verify)

FAIL = []
def check(n, c, d=""):
    (print(f"  PASS  {n}") if c else (FAIL.append(n), print(f"  FAIL  {n}  {d}")))


def make_checkpoint(dirpath, n_experts=4, layers=2, seed=0):
    """A miniature safetensors MoE checkpoint with the same STRUCTURE as a real one:
    3-D fused expert tensors, a 2-D router that must NOT be treated as expert data, and
    non-expert tensors that must be left alone."""
    os.makedirs(dirpath, exist_ok=True)
    rng = np.random.default_rng(seed)
    tensors, blobs = {}, []
    off = 0

    def add(name, arr):
        nonlocal off
        b = arr.tobytes()
        dt = {np.dtype("float16"): "F16", np.dtype("uint32"): "U32"}[arr.dtype]
        tensors[name] = {"dtype": dt, "shape": list(arr.shape),
                         "data_offsets": [off, off + len(b)]}
        blobs.append(b)
        off += len(b)

    for l in range(layers):
        for proj in ("gate_proj", "up_proj", "down_proj"):
            add(f"model.layers.{l}.mlp.switch_mlp.{proj}.weight",
                rng.integers(0, 2**31, (n_experts, 8, 16), dtype=np.uint32))
            add(f"model.layers.{l}.mlp.switch_mlp.{proj}.scales",
                rng.standard_normal((n_experts, 8, 4)).astype(np.float16))
            add(f"model.layers.{l}.mlp.switch_mlp.{proj}.biases",
                rng.standard_normal((n_experts, 8, 4)).astype(np.float16))
        # the ROUTER: leads with n_experts but is 2-D and must be excluded
        add(f"model.layers.{l}.mlp.gate.weight",
            rng.integers(0, 2**31, (n_experts, 16), dtype=np.uint32))
        # something entirely unrelated
        add(f"model.layers.{l}.self_attn.q_proj.weight",
            rng.integers(0, 2**31, (32, 16), dtype=np.uint32))
    add("model.embed_tokens.weight", rng.integers(0, 2**31, (64, 16), dtype=np.uint32))

    hdr = json.dumps(tensors).encode()
    with open(os.path.join(dirpath, "model.safetensors"), "wb") as f:
        f.write(struct.pack("<Q", len(hdr)))
        f.write(hdr)
        for b in blobs:
            f.write(b)
    return tensors


TMP = tempfile.mkdtemp(prefix="bigrig_repack_")
SRC = os.path.join(TMP, "src")
OUT = os.path.join(TMP, "packed.experts")
E, L = 4, 2
TENSORS = make_checkpoint(SRC, E, L)

print("=" * 78); print("1. THE ROUTER MUST NOT BE TREATED AS EXPERT DATA"); print("=" * 78)
hdr, _ = read_header(os.path.join(SRC, "model.safetensors"))
per = find_expert_tensors(hdr, E)
names = {n for d in per.values() for n in d}
check("expert tensors are found", len(names) == L * 9, f"{len(names)} found")
check("the 2-D router is EXCLUDED even though it leads with n_experts",
      not any(n.endswith("mlp.gate.weight") for n in names), str([n for n in names if 'gate.weight' in n]))
check("unrelated tensors are excluded",
      not any("q_proj" in n or "embed" in n for n in names))

print("\n" + "=" * 78); print("2. EVERY BYTE MUST SURVIVE THE ROUND TRIP"); print("=" * 78)
man = repack(SRC, OUT, E, verify_every=True)
check("repack completes and self-verifies", man["blocks"] == E * L, str(man["blocks"]))
check("the manifest records the packed byte total",
      man["bytes"] == os.path.getsize(OUT), f"{man['bytes']} vs {os.path.getsize(OUT)}")
bad = verify(SRC, OUT, E, sample=None)
check("an INDEPENDENT verify pass finds no differing bytes", not bad, str(bad[:2]))

# and check it by hand, without using the repacker's own helpers
src_path = os.path.join(SRC, "model.safetensors")
hdr, data0 = read_header(src_path)
mani = json.load(open(OUT + ".json"))
handbad = []
with open(OUT, "rb") as pf, open(src_path, "rb") as sf:
    for key, blk in mani["layout"].items():
        layer, e = key.split("/"); e = int(e)
        pf.seek(blk["offset"]); block = pf.read(blk["length"])
        for p in blk["pieces"]:
            info = hdr[p["name"]]
            per_e = _nbytes(info["shape"][1:], info["dtype"])
            sf.seek(data0 + info["data_offsets"][0] + e * per_e)
            if block[p["rel_offset"]:p["rel_offset"] + per_e] != sf.read(per_e):
                handbad.append((key, p["name"]))
check("a hand-written comparison agrees, byte for byte", not handbad, str(handbad[:2]))

print("\n" + "=" * 78); print("3. TEETH — the verifier must catch corruption"); print("=" * 78)
# A verifier that cannot fail is decoration. Flip one byte in the packed file.
with open(OUT, "r+b") as f:
    f.seek(mani["layout"]["0/1"]["offset"] + 7)
    orig = f.read(1)
    f.seek(mani["layout"]["0/1"]["offset"] + 7)
    f.write(bytes([orig[0] ^ 0xFF]))
bad = verify(SRC, OUT, E, sample=None)
check("flipping ONE byte is detected", any(k == "0/1" for k, _ in bad), str(bad[:2]))
check("...and only the corrupted block is reported", len(bad) == 1, f"{len(bad)} blocks flagged")
with open(OUT, "r+b") as f:                       # repair
    f.seek(mani["layout"]["0/1"]["offset"] + 7); f.write(orig)
check("repairing the byte clears the failure", not verify(SRC, OUT, E, sample=None))

# truncation must also be caught
with open(OUT, "r+b") as f:
    f.truncate(os.path.getsize(OUT) - 64)
check("truncating the packed file is detected", bool(verify(SRC, OUT, E, sample=None)))
man = repack(SRC, OUT, E, verify_every=True)      # rebuild

print("\n" + "=" * 78); print("4. THE LAYOUT MUST BE STRUCTURALLY SOUND"); print("=" * 78)
mani = json.load(open(OUT + ".json"))
blocks = sorted((b["offset"], b["length"], k) for k, b in mani["layout"].items())
overlap = [(blocks[i][2], blocks[i + 1][2]) for i in range(len(blocks) - 1)
           if blocks[i][0] + blocks[i][1] > blocks[i + 1][0]]
check("no two blocks overlap", not overlap, str(overlap[:2]))
gaps = [blocks[i + 1][0] - (blocks[i][0] + blocks[i][1]) for i in range(len(blocks) - 1)]
check("blocks are packed with no wasted space", all(g == 0 for g in gaps), f"gaps {set(gaps)}")
check("every (layer, expert) pair is present",
      set(mani["layout"]) == {f"{l}/{e}" for l in range(L) for e in range(E)})
check("the last block ends exactly at end of file",
      blocks[-1][0] + blocks[-1][1] == os.path.getsize(OUT))

print("\n" + "=" * 78); print("5. IT PLUGS INTO THE FETCH ENGINE"); print("=" * 78)
layout, mani = load_layout(OUT)
store = WeightStore(OUT, layout)
check("load_layout yields one region per expert", len(store) == E * L, str(len(store)))
with ParallelFetcher(store, threads=4) as f:
    got = f.fetch([(0, 0), (1, 3)])
check("the fetch engine can read packed blocks", len(got) == 2)
pieces = unpack_block(got[(0, 0)], "0/0", mani)
check("a fetched block splits back into named tensors", len(pieces) == 9, str(len(pieces)))
ref_name = sorted(pieces)[0]
info = hdr[ref_name]
per_e = _nbytes(info["shape"][1:], info["dtype"])
with open(src_path, "rb") as sf:
    sf.seek(data0 + info["data_offsets"][0] + 0)
    check("an unpacked slice equals the original tensor slice", pieces[ref_name] == sf.read(per_e))
try:
    unpack_block(got[(0, 0)][:-1], "0/0", mani)
    check("unpacking a wrong-sized block raises", False)
except ValueError:
    check("unpacking a wrong-sized block raises", True)

print("\n" + "=" * 78); print("6. HOSTILE INPUT"); print("=" * 78)
try:
    repack("/tmp/does_not_exist_at_all", OUT + ".x", E); check("a missing source raises", False)
except (FileNotFoundError, NotADirectoryError):
    check("a missing source raises", True)
empty = os.path.join(TMP, "empty"); os.makedirs(empty, exist_ok=True)
try:
    repack(empty, OUT + ".y", E); check("a directory with no safetensors raises", False)
except FileNotFoundError:
    check("a directory with no safetensors raises", True)
try:
    _nbytes([4, 4], "MADE_UP"); check("an unknown dtype raises rather than guessing", False)
except ValueError as e:
    check("an unknown dtype raises rather than guessing", "refusing to guess" in str(e))
m2 = repack(SRC, os.path.join(TMP, "wrongE.experts"), 999, verify_every=False)
check("a wrong expert count packs nothing rather than mangling data",
      m2["blocks"] == 0, str(m2["blocks"]))

print("\n" + "=" * 78); print("7. THE POINT: FEWER, LARGER READS"); print("=" * 78)
scattered_reads = len({(l, e, n) for l in range(L) for e in range(E) for n in per[l]})
packed_reads = len(mani["layout"])
check("one expert becomes one read", packed_reads * 9 == scattered_reads,
      f"{scattered_reads} -> {packed_reads}")
sizes = [b["length"] for b in mani["layout"].values()]
check("every block is the same size (uniform experts)", len(set(sizes)) == 1, str(set(sizes)))
print(f"        {scattered_reads} scattered reads -> {packed_reads} packed reads, "
      f"{sizes[0]/1024:.1f} KB each")

import shutil
shutil.rmtree(TMP, ignore_errors=True)
print("\n" + "=" * 78)
print(f"{'ALL TESTS PASSED' if not FAIL else str(len(FAIL)) + ' FAILURES: ' + ', '.join(FAIL)}")
print("=" * 78)
sys.exit(1 if FAIL else 0)
