"""COMPONENT 6 — measure THIS machine at startup instead of shipping constants.

WHY THIS COMPONENT EXISTS, AND IT IS NOT A NICE-TO-HAVE
    Three wrong verdicts in this project came from treating a measured constant as universal:

      kappa = 67          inherited from an H100 over PCIe5. This Mac reads 4.9 on an idle
                          machine with a fresh file, and ~25 under the memory pressure that
                          actually exists when streaming matters. Three different numbers, all
                          "correct", none transferable.
      0.70 GB/s streaming measured through mmap, single-threaded. Parallel explicit reads give
                          2-7x more. A verdict of "physics forbids it" rested on the slow one.
      8 threads           the optimum here. calibrate_threads finds 4 on one file and 12 on
                          another ON THE SAME MACHINE, so even this is not a machine constant.

    Every one of those would have been caught by measuring at startup rather than assuming.

WHAT IT MEASURES
    ram_gbs        achieved memory bandwidth, not the spec sheet. The M4 is rated 120 GB/s and
                   delivers ~87 -- llama.cpp gets the same, so this is the real ceiling.
    disk_gbs       parallel cold-read throughput at expert granularity, the number the fetch
                   engine is sized against.
    kappa          ram_gbs / disk_gbs. Every hurdle rate in the engine derives from it.
    threads        the thread count that actually maximises disk_gbs here.
    avail_gb       memory genuinely available, which is NOT `free` -- see below.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
import time

import numpy as np

PAGE = 16384


def _vm(key: str) -> int:
    v = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    m = re.search(rf"{re.escape(key)}:\s+(\d+)", v)
    return int(m.group(1)) if m else 0


def total_gb() -> float:
    """Physical RAM in this machine. The one memory number that does not move."""
    try:
        out = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True,
                             timeout=5)
        return int(out.stdout.strip()) / 1e9
    except (OSError, ValueError, subprocess.SubprocessError):
        return 0.0


def available_gb() -> float:
    """Memory an engine may actually plan to use.

    NOT `Pages free`. On macOS a low free count usually means the file cache is full, and the
    cache is returned instantly -- verified by allocating 3 GB in 0.16 s while free read 0.21 GB.
    Sizing a cache off `free` would refuse to run on a perfectly healthy machine.
    """
    return (_vm("Pages free") + _vm("Pages speculative")
            + _vm("Pages purgeable") + _vm("Pages inactive")) * PAGE / 1e9


def under_pressure(window_s: float = 0.4, grow_mb: float = 24.0) -> bool:
    """True if macOS is compressing anonymous memory RIGHT NOW.

    Not "has ever compressed". The compressor's occupancy is cumulative -- after a few heavy runs
    it sits at a gigabyte or more on a completely idle machine, and a static threshold then tells
    every user to close something while 13 GB is free. What actually indicates pressure is the
    compressor GROWING, or the swap file being written to, during a short window.
    """
    c0, so0 = _vm("Pages occupied by compressor"), _vm("Swapouts")
    time.sleep(window_s)
    c1, so1 = _vm("Pages occupied by compressor"), _vm("Swapouts")
    return ((c1 - c0) * PAGE / 1e6 > grow_mb) or (so1 > so0)


def phys_footprint_gb(pid: int | None = None) -> float:
    """What the Mac itself charges this process: its physical footprint, the number Activity
    Monitor calls Memory. It counts wired file-backed pages a GPU kernel has touched -- which
    `mx.get_active_memory()` never sees, because they are not MLX allocations -- and it is the
    number the kernel's memory pressure actually responds to. 0.0 if it cannot be read.

    rusage_info_v4 (sys/resource.h): ri_uuid[16], then uint64 ri_user_time, ri_system_time,
    ri_pkg_idle_wkups, ri_interrupt_wkups, ri_pageins, ri_wired_size, ri_resident_size,
    ri_phys_footprint, ... -- so the footprint is the eighth 64-bit field after the uuid."""
    import ctypes
    import ctypes.util
    try:
        lib = ctypes.CDLL(ctypes.util.find_library("proc") or "libproc.dylib")

        class RUsage(ctypes.Structure):
            _fields_ = [("ri_uuid", ctypes.c_uint8 * 16)] + [
                (f"f{i}", ctypes.c_uint64) for i in range(48)]
        lib.proc_pid_rusage.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
        lib.proc_pid_rusage.restype = ctypes.c_int
        buf = RUsage()
        if lib.proc_pid_rusage(int(pid or os.getpid()), 4, ctypes.byref(buf)) != 0:
            return 0.0
        return float(buf.f7) / 1e9
    except Exception:                            # noqa: BLE001 -- a reading, never a failure
        return 0.0


def measure_ram_gbs(bytes_n: int = 1 << 28, reps: int = 5) -> float:
    """Achieved memory bandwidth. Uses MLX if present, since that is what the engine will run on,
    and falls back to numpy so calibration never depends on a GPU framework being importable."""
    try:
        import mlx.core as mx
        n = bytes_n // 4
        x = mx.ones((n,), dtype=mx.float32)
        y = mx.zeros((n,), dtype=mx.float32)
        mx.eval(x, y)
        best = 0.0
        for _ in range(reps):
            t0 = time.perf_counter()
            z = x + y
            mx.eval(z)
            best = max(best, 3 * n * 4 / (time.perf_counter() - t0) / 1e9)
        return best
    except Exception:
        n = bytes_n
        a = np.ones(n, dtype=np.uint8)
        b = np.empty_like(a)
        np.copyto(b, a)
        best = 0.0
        for _ in range(reps):
            t0 = time.perf_counter()
            np.copyto(b, a)
            best = max(best, 2 * n / (time.perf_counter() - t0) / 1e9)
        return best


def measure_disk_gbs(path: str | None = None, chunk: int = 64 << 20,
                     candidates=(1, 2, 4, 8, 12), probe_gb: float = 0.75,
                     reps: int = 2) -> dict:
    """Parallel cold-read throughput, and the thread count that achieves it.

    Reads are cache-bypassed, unmapped, and land on distinct offsets, because a cached read is
    not a read and measuring one would produce exactly the kind of over-optimistic constant this
    component exists to prevent.

    THE MAP MUST BE OFF HERE, AND THIS IS NOT A DETAIL. Serving reads an expert as a view of the
    file, which is 5.4x cheaper and the right thing for serving. Measuring the DISK that way
    times the page cache instead: it read 1,267 GB/s against a real RAM bandwidth of 95.6, which
    made kappa -- the ratio this whole planner is built on -- collapse to zero. A planner told
    the disk is thirteen times faster than memory will choose configurations that do not exist.
    """
    from .fetch import ParallelFetcher, Region, WeightStore

    tmp = None
    if path is None or not os.path.exists(os.path.expanduser(path)):
        tmp = tempfile.NamedTemporaryFile(prefix="bigrig_cal_", suffix=".bin", delete=False)
        rng = np.random.default_rng(0)
        need = max(int(probe_gb * 1e9 * 3), chunk * (max(candidates) + 2))
        written = 0
        while written < need:
            b = rng.integers(0, 255, min(1 << 24, need - written), dtype=np.uint8).tobytes()
            tmp.write(b)
            written += len(b)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp.close()
        path = tmp.name
    path = os.path.expanduser(path)

    try:
        size = os.path.getsize(path)
        nreg = max(1, (size - chunk) // chunk)
        layout = {i: Region(i * chunk, chunk) for i in range(nreg)}
        store = WeightStore(path, layout)
        keys = list(layout)
        per = max(1, int(probe_gb * 1e9 // chunk))
        rng = np.random.default_rng(1)
        res: dict = {}
        for rep in range(reps):
            order = candidates if rep % 2 == 0 else tuple(reversed(candidates))
            for t in order:
                sel = [keys[int(i)] for i in rng.choice(len(keys), min(per, len(keys)),
                                                        replace=False)]
                with ParallelFetcher(store, threads=t, nocache=True, mapped=False) as f:
                    t0 = time.perf_counter()
                    got = f.fetch(sel)
                    dt = time.perf_counter() - t0
                res.setdefault(t, []).append(sum(len(v) for v in got.values()) / dt / 1e9)
        med = {t: float(np.median(v)) for t, v in res.items()}
        best = max(med, key=med.get)
        return {"disk_gbs": med[best], "threads": best, "by_threads": med,
                "speedup_vs_1": med[best] / med.get(1, med[best])}
    finally:
        if tmp is not None:
            try:
                os.remove(path)
            except OSError:
                pass


def calibrate(weight_path: str | None = None, save: str | None = None) -> dict:
    """Full host profile. Run once at engine startup; it takes a few seconds."""
    t0 = time.perf_counter()
    ram = measure_ram_gbs()
    disk = measure_disk_gbs(weight_path)
    prof = {
        "ram_gbs": round(ram, 2),
        "disk_gbs": round(disk["disk_gbs"], 2),
        "kappa": round(ram / disk["disk_gbs"], 1) if disk["disk_gbs"] > 0 else None,
        "fetch_threads": disk["threads"],
        "disk_by_threads": {k: round(v, 2) for k, v in disk["by_threads"].items()},
        "parallel_speedup": round(disk["speedup_vs_1"], 2),
        "available_gb": round(available_gb(), 2),
        "under_pressure": under_pressure(),
        "measured_on": weight_path or "scratch file",
        "seconds": round(time.perf_counter() - t0, 2),
    }
    if save:
        with open(os.path.expanduser(save), "w") as f:
            json.dump(prof, f, indent=1)
    return prof


def plan(profile: dict, model_gb: float, active_gb: float, miss_rate: float) -> dict:
    """What this host can do with a given model, from ITS OWN measured numbers.

    `miss_rate` is supplied rather than guessed: it depends on the eviction policy and the
    model's routing, which are components 9 and 5, not this one.
    """
    avail = max(0.5, profile["available_gb"] - 2.0)         # leave headroom for the runtime
    residency = min(1.0, avail / model_gb) if model_gb > 0 else 1.0
    # A model that fits cannot miss. Reporting "100% resident, 5% miss" is incoherent, and it
    # is the kind of incoherence that gets quoted as a result -- so the miss rate is clamped to
    # what the residency permits and the caller is told it was clamped.
    clamped = False
    if residency >= 1.0 and miss_rate > 0:
        miss_rate, clamped = 0.0, True
    disk_part = active_gb * miss_rate / profile["disk_gbs"]
    ram_part = active_gb * (1 - miss_rate) / profile["ram_gbs"]
    t = disk_part + ram_part
    return {"residency": round(residency, 3), "fits": model_gb <= avail,
            "seconds_per_token": round(t, 4), "tok_s": round(1 / t, 2),
            "miss_rate_used": miss_rate, "miss_rate_clamped": clamped,
            "disk_fraction_of_time": round(disk_part / t, 3)}
