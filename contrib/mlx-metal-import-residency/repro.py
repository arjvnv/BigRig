"""Importing host pages with mx.from_dlpack: Metal makes the WHOLE buffer resident.

Zero copy import is the right way to hand mapped file pages to the GPU on Apple Silicon. The
undocumented part is the unit of residency. Touching one small slice of a large imported buffer
wires the entire buffer, not the slice, so a design that imports a big region and indexes into
it wires the whole region on first use.

Each trial runs in a fresh subprocess, because releasing an import is deferred and would
otherwise land inside the next trial's measurement.

Run:  python repro.py [size_mb]
Needs: mlx, numpy, macOS. Writes and deletes one temp file. No download.
"""
import mmap
import os
import subprocess
import sys
import tempfile

import numpy as np

PAGE = mmap.PAGESIZE
SLICE = 2 * 1024 * 1024                      # what we actually read: about one expert


def wired_bytes():
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    for line in out.splitlines():
        if "wired down" in line:
            return int(line.split(":")[1].strip().rstrip(".")) * PAGE
    raise RuntimeError("vm_stat did not report wired pages")


def child(mode, path, size):
    """One measurement, then exit so the import is definitely released."""
    import mlx.core as mx
    with open(path, "rb") as fh:
        mp = mmap.mmap(fh.fileno(), size, access=mmap.ACCESS_READ)
    base = np.frombuffer(mp, dtype=np.uint8)
    if base.ctypes.data % PAGE:
        print("UNALIGNED")
        return 2
    mx.clear_cache()
    before = wired_bytes()
    if mode == "none":                       # baseline: what an MLX process costs by itself
        target = mx.zeros((SLICE // 4,), dtype=mx.float32)
    elif mode == "whole":
        target = mx.from_dlpack(base)[:SLICE]
    else:
        target = mx.from_dlpack(base[:SLICE])
    s = mx.sum(target.view(mx.float32))      # force a Metal kernel to read it
    mx.eval(s)
    print(f"{wired_bytes() - before}")
    return 0


if len(sys.argv) > 2 and sys.argv[1] == "--child":
    raise SystemExit(child(sys.argv[2], sys.argv[3], int(sys.argv[4])))

SIZE_MB = int(sys.argv[1]) if len(sys.argv) > 1 else 384
SIZE = (SIZE_MB * 1024 * 1024 // PAGE) * PAGE
TRIALS = 3

import mlx.core as mx                                                       # noqa: E402
print("=" * 96)
print(f"mlx {mx.__version__}   file {SIZE / 1e6:.0f} MB, page size {PAGE} B, "
      f"reading a {SLICE / 1e6:.0f} MB slice of it")
print("=" * 96)

with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as fh:
    path = fh.name
    chunk = np.zeros(1024 * 1024, dtype=np.uint8).tobytes()
    for _ in range(SIZE // len(chunk)):
        fh.write(chunk)
    fh.flush()
    os.fsync(fh.fileno())

try:
    print(f"\n  Same {SLICE / 1e6:.0f} MB read, imported two ways, each in a fresh process.\n")
    print(f"  {'import':40s} " + "".join(f"{'trial ' + str(i + 1):>12s}" for i in range(TRIALS))
          + f"{'median':>12s}")
    res = {}
    for mode, label in (("none", "baseline: no import at all"),
                        ("whole", "whole file as ONE buffer"),
                        ("piece", "just the slice as its own buffer")):
        vals = []
        for _ in range(TRIALS):
            out = subprocess.run([sys.executable, os.path.abspath(__file__), "--child", mode,
                                  path, str(SIZE)], capture_output=True, text=True)
            vals.append(int(out.stdout.strip()) if out.returncode == 0 else float("nan"))
        res[mode] = float(np.median(vals))
        print(f"  {label:40s} " + "".join(f"{v / 1e6:+11.1f}M" for v in vals)
              + f"{res[mode] / 1e6:+11.1f}M")

    mw = res["whole"] - res["none"]
    mp_ = res["piece"] - res["none"]
    print()
    print(f"  Marginal cost of the import, over a bare MLX process:")
    print(f"    whole file as one buffer     {mw / 1e6:+8.1f} MB "
          f"= {mw / SIZE * 100:5.1f}% of the {SIZE / 1e6:.0f} MB file, to read {SLICE / 1e6:.0f} MB")
    print(f"    slice as its own buffer      {mp_ / 1e6:+8.1f} MB "
          f"= {mp_ / SLICE * 100:5.1f}% of the {SLICE / 1e6:.0f} MB slice")
    ok = mw > 0.5 * SIZE and mw > 8 * max(mp_, 1e6)
    print()
    if ok:
        print("  REPRODUCED: residency is per buffer, not per page touched. Import each usable")
        print("              piece as its own buffer; never one buffer over the whole region.")
    else:
        print("  NOT REPRODUCED on this machine at this size.")
    print("=" * 96)
finally:
    os.unlink(path)
raise SystemExit(0 if ok else 1)
