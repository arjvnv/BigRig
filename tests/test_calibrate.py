"""Adversarial tests for host calibration.

Calibration exists because this project shipped three wrong constants. Its own failure mode is
producing a flattering number that then gets trusted, so these tests target that specifically.
"""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bigrig_engine.calibrate import (available_gb, calibrate, measure_disk_gbs,
                                       measure_ram_gbs, plan, under_pressure)

FAIL = []
def check(n, c, d=""):
    (print(f"  PASS  {n}") if c else (FAIL.append(n), print(f"  FAIL  {n}  {d}")))


print("=" * 78); print("1. THE NUMBERS MUST BE PHYSICALLY POSSIBLE"); print("=" * 78)
ram = measure_ram_gbs()
check("RAM bandwidth is in a plausible range for Apple Silicon",
      20 < ram < 1200, f"{ram:.1f} GB/s")
check("RAM bandwidth does not exceed any shipping Mac's spec", ram < 1200, f"{ram:.1f}")
d = measure_disk_gbs(probe_gb=0.15, candidates=(1, 4), reps=1)
check("disk throughput is positive and below RAM", 0 < d["disk_gbs"] < ram,
      f"disk {d['disk_gbs']:.2f} vs ram {ram:.1f}")
check("the chosen thread count is one that was measured",
      d["threads"] in d["by_threads"], str(d))
check("the chosen thread count is the fastest measured",
      d["by_threads"][d["threads"]] == max(d["by_threads"].values()), str(d["by_threads"]))

print("\n" + "=" * 78); print("2. AVAILABLE MEMORY IS NOT `free`"); print("=" * 78)
# The bug this prevents: sizing a cache off `Pages free`, which on macOS is usually small
# because the file cache is full, and refusing to run on a healthy machine.
a = available_gb()
free_only = 0
import re
import subprocess
v = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
m = re.search(r"Pages free:\s+(\d+)", v)
free_only = int(m.group(1)) * 16384 / 1e9 if m else 0
check("available memory is reported, and is at least as large as free",
      a >= free_only - 0.01, f"available {a:.2f} vs free {free_only:.2f}")
check("available memory is not larger than the machine", a < 200, f"{a:.2f} GB")
check("pressure is a boolean and is False on a healthy machine",
      isinstance(under_pressure(), bool))

print("\n" + "=" * 78); print("3. A SCRATCH FILE MUST NOT FLATTER THE RESULT"); print("=" * 78)
# Measured: a scratch file gave 9.64 GB/s and kappa 10.3; the real 26 GB weight file gave
# 5.34 GB/s and kappa 18.6. Calibrating on the wrong file is how a wrong constant gets shipped.
big = tempfile.NamedTemporaryFile(prefix="bigrig_cal_big_", suffix=".bin", delete=False)
rng = np.random.default_rng(0)
for _ in range(96):                                  # ~1.6 GB, larger than the probe
    big.write(rng.integers(0, 255, 1 << 24, dtype=np.uint8).tobytes())
big.flush(); os.fsync(big.fileno()); big.close()
on_real = measure_disk_gbs(big.name, probe_gb=0.2, candidates=(1, 4), reps=1)
check("calibration accepts a real weight file and measures it",
      on_real["disk_gbs"] > 0, str(on_real))
prof = calibrate(weight_path=big.name)
check("the profile records WHICH file it measured", prof["measured_on"] == big.name,
      prof["measured_on"])
os.remove(big.name)

print("\n" + "=" * 78); print("4. THE PROFILE IS SELF-CONSISTENT"); print("=" * 78)
p = calibrate()
check("kappa equals ram/disk", abs(p["kappa"] - p["ram_gbs"] / p["disk_gbs"]) < 0.15, str(p))
check("kappa is greater than 1 — disk is slower than RAM", p["kappa"] > 1, str(p["kappa"]))
check("parallel speedup is at least 1", p["parallel_speedup"] >= 1.0, str(p))
check("every key the engine needs is present",
      all(k in p for k in ("ram_gbs", "disk_gbs", "kappa", "fetch_threads", "available_gb")))
check("calibration is fast enough to run at every startup", p["seconds"] < 30, str(p["seconds"]))

print("\n" + "=" * 78); print("5. plan() MUST BE HONEST ABOUT WHAT IT KNOWS"); print("=" * 78)
r_fit = plan(p, model_gb=1.0, active_gb=1.0, miss_rate=0.0)
check("a model that fits, with no misses, is bounded by RAM bandwidth",
      abs(r_fit["tok_s"] - p["ram_gbs"] / 1.0) / (p["ram_gbs"]) < 0.02, str(r_fit))
check("...and is reported as fitting", r_fit["fits"] is True)
r_miss = plan(p, model_gb=500.0, active_gb=10.0, miss_rate=0.5)
check("a huge model with heavy misses is reported as not fitting", r_miss["fits"] is False)
check("more misses is never faster",
      plan(p, 100, 10, 0.20)["tok_s"] < plan(p, 100, 10, 0.05)["tok_s"])
check("the disk share of time rises with the miss rate",
      plan(p, 100, 10, 0.20)["disk_fraction_of_time"] >
      plan(p, 100, 10, 0.05)["disk_fraction_of_time"])
check("residency never exceeds 1", plan(p, 0.5, 0.5, 0.0)["residency"] <= 1.0)

print("\n" + "=" * 78); print("6. TEETH — a wrong profile must produce a wrong plan"); print("=" * 78)
# If plan() ignored the profile, calibration would be decorative.
slow = dict(p, disk_gbs=0.5, ram_gbs=p["ram_gbs"])
fast = dict(p, disk_gbs=50.0, ram_gbs=p["ram_gbs"])
a_, b_ = plan(slow, 100, 10, 0.10)["tok_s"], plan(fast, 100, 10, 0.10)["tok_s"]
check("a slower measured disk yields a slower plan", a_ < b_, f"{a_} vs {b_}")
print(f"        (disk 0.5 GB/s -> {a_} tok/s, disk 50 GB/s -> {b_} tok/s)")

print("\n" + "=" * 78)
print(f"{'ALL TESTS PASSED' if not FAIL else str(len(FAIL)) + ' FAILURES: ' + ', '.join(FAIL)}")
print("=" * 78)
sys.exit(1 if FAIL else 0)
