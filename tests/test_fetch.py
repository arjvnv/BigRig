"""Adversarial tests for the parallel fetch engine.

The engine returns BYTES. If it ever returns the wrong bytes, every layer above it computes with
silently corrupted weights and the model degrades in a way no quality meter can attribute. So
correctness is checked first, exhaustively, against independently-read ground truth, and speed is
only measured once the bytes are proven right.
"""
import os
import sys
import threading
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bigrig_engine.fetch import (DEFAULT_THREADS, ParallelFetcher, Region, WeightStore,
                                   calibrate_threads)

FAIL = []
def check(n, c, d=""):
    (print(f"  PASS  {n}") if c else (FAIL.append(n), print(f"  FAIL  {n}  {d}")))


TMP = "/tmp/bigrig_fetch_test.bin"
SZ = 64 << 20
rng = np.random.default_rng(0)
DATA = rng.integers(0, 255, SZ, dtype=np.uint8).tobytes()
with open(TMP, "wb") as f:
    f.write(DATA)

EXPERT = 1 << 20
LAYOUT = {("L%d" % (i // 8), "E%d" % (i % 8)): Region(i * EXPERT, EXPERT)
          for i in range(SZ // EXPERT)}
STORE = WeightStore(TMP, LAYOUT)

print("=" * 78); print("1. THE BYTES MUST BE EXACTLY RIGHT"); print("=" * 78)
with ParallelFetcher(STORE, threads=8) as f:
    keys = list(LAYOUT)[:24]
    got = f.fetch(keys)
    bad = [k for k in keys
           if got[k] != DATA[LAYOUT[k].offset:LAYOUT[k].offset + LAYOUT[k].length]]
check("every fetched region matches the file byte for byte", not bad, f"{len(bad)} wrong")
check("one entry returned per requested key", len(got) == len(keys))
check("each is the right length", all(len(v) == EXPERT for v in got.values()))

# the ordering trap: results are keyed, so a reordered completion must not mis-assign bytes
with ParallelFetcher(STORE, threads=8) as f:
    ks = list(LAYOUT)[:16][::-1]
    g = f.fetch(ks)
    mis = [k for k in ks if g[k] != DATA[LAYOUT[k].offset:LAYOUT[k].offset + EXPERT]]
check("out-of-order completion does not mis-assign bytes to keys", not mis, f"{len(mis)} wrong")

print("\n" + "=" * 78); print("2. THREAD COUNT MUST NOT CHANGE THE ANSWER"); print("=" * 78)
ref = None
same = True
for t in (1, 2, 4, 8, 12):
    with ParallelFetcher(STORE, threads=t, nocache=True, mapped=False) as f:
        g = f.fetch(list(LAYOUT)[:12])
    if ref is None:
        ref = g
    elif g != ref:
        same = False
check("1, 2, 4, 8 and 12 threads all return identical bytes", same)

print("\n" + "=" * 78); print("3. HOSTILE INPUT"); print("=" * 78)
with ParallelFetcher(STORE, threads=4) as f:
    check("an empty request returns an empty dict", f.fetch([]) == {})
    dup = list(LAYOUT)[:3] * 4
    g = f.fetch(dup)
    check("duplicate keys collapse to one entry each", len(g) == 3)
    try:
        f.fetch([("nope", "missing")]); check("an unknown key raises KeyError", False)
    except KeyError as e:
        check("an unknown key raises KeyError", "not in this store" in str(e))

try:
    WeightStore(TMP, {"past_end": Region(SZ - 10, 1000)})
    check("a region past the end of the file is refused at construction", False)
except ValueError as e:
    check("a region past the end of the file is refused at construction",
          "runs past the end" in str(e))
for bad_r in ((-1, 10), (0, 0), (0, -5)):
    try:
        Region(*bad_r); check(f"Region{bad_r} is refused", False)
    except ValueError:
        check(f"Region{bad_r} is refused", True)
try:
    WeightStore("/tmp/definitely_not_here.bin", {}); check("a missing file raises", False)
except FileNotFoundError:
    check("a missing file raises", True)
try:
    ParallelFetcher(STORE, threads=0); check("threads=0 is refused", False)
except ValueError:
    check("threads=0 is refused", True)

print("\n" + "=" * 78); print("4. PREFETCH — the whole point of the engine"); print("=" * 78)
with ParallelFetcher(STORE, threads=8) as f:
    ks = list(LAYOUT)[:16]
    f.prefetch(ks)
    time.sleep(0.35)                       # stand in for "layer L computes"
    t0 = time.perf_counter()
    g = f.fetch(ks)
    after = time.perf_counter() - t0
    ok = all(g[k] == DATA[LAYOUT[k].offset:LAYOUT[k].offset + EXPERT] for k in ks)
check("prefetched bytes are correct", ok)
with ParallelFetcher(STORE, threads=8) as f2:
    ks2 = list(LAYOUT)[16:32]
    t0 = time.perf_counter()
    f2.fetch(ks2)
    cold = time.perf_counter() - t0
check("a prefetched fetch returns faster than a cold one", after < cold,
      f"prefetched {after*1000:.1f} ms vs cold {cold*1000:.1f} ms")
print(f"        (prefetched {after*1000:.1f} ms, cold {cold*1000:.1f} ms)")

with ParallelFetcher(STORE, threads=4) as f:
    k = list(LAYOUT)[:4]
    f.prefetch(k); f.prefetch(k); f.prefetch(k)
    g = f.fetch(k)
check("prefetching the same keys repeatedly does not read them repeatedly",
      f.stats()["n_fetches"] == 1 and len(g) == 4)

print("\n" + "=" * 78); print("5. CONCURRENCY AND LIFECYCLE"); print("=" * 78)
errs = []
with ParallelFetcher(STORE, threads=8) as f:
    def worker(i):
        try:
            ks = list(LAYOUT)[i * 4:(i + 1) * 4]
            g = f.fetch(ks)
            for k in ks:
                assert g[k] == DATA[LAYOUT[k].offset:LAYOUT[k].offset + EXPERT]
        except Exception as e:               # noqa: BLE001
            errs.append(e)
    ts = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    [t.start() for t in ts]; [t.join() for t in ts]
check("eight threads fetching concurrently all get correct bytes", not errs, str(errs[:1]))

before = threading.active_count()
for _ in range(5):
    with ParallelFetcher(STORE, threads=8) as f:
        f.fetch(list(LAYOUT)[:4])
time.sleep(0.4)
check("closing the fetcher leaves no threads behind",
      threading.active_count() <= before, f"{before} -> {threading.active_count()}")

f = ParallelFetcher(STORE, threads=2); f.close()
try:
    f.fetch(list(LAYOUT)[:1]); check("using a closed fetcher raises", False)
except RuntimeError:
    check("using a closed fetcher raises", True)
f.close()
check("close() twice is harmless", True)

print("\n" + "=" * 78); print("6. A SHORT READ MUST RAISE, NOT RETURN JUNK"); print("=" * 78)
# TEETH: point a store at a file that is then truncated. Silently returning short or zeroed
# buffers is the failure mode that would corrupt weights invisibly.
T2 = "/tmp/bigrig_fetch_trunc.bin"
with open(T2, "wb") as fh:
    fh.write(DATA[:8 << 20])
s2 = WeightStore(T2, {"a": Region(0, 4 << 20)})
with open(T2, "wb") as fh:
    fh.write(DATA[:1 << 20])                 # truncate underneath the store
with ParallelFetcher(s2, threads=2) as f:
    try:
        f.fetch(["a"]); check("a truncated file raises instead of returning junk", False)
    except IOError as e:
        check("a truncated file raises instead of returning junk", "short read" in str(e))
os.remove(T2)

print("\n" + "=" * 78); print("7. SPEED — only now that the bytes are proven"); print("=" * 78)
res = {}
for t in (1, 2, 4, 8):
    best = 0.0
    for _ in range(3):
        with ParallelFetcher(STORE, threads=t, nocache=True, mapped=False) as f:
            ks = list(LAYOUT)
            t0 = time.perf_counter()
            g = f.fetch(ks)
            dt = time.perf_counter() - t0
        best = max(best, sum(len(v) for v in g.values()) / dt / 1e9)
    res[t] = best
    print(f"        {t:2d} threads: {best:6.2f} GB/s")
check("parallel reads are faster than a single reader", res[8] > res[1],
      f"1 thread {res[1]:.2f} GB/s, 8 threads {res[8]:.2f} GB/s")
# The SIZE of the gain is a property of the machine's current load, not of this code. Asserting
# a ratio made this test fail at 1.21x while a background job was hammering the same disk --
# a correct implementation reported as a regression. The invariant is that parallelism helps;
# the magnitude is reported, not asserted.
print(f"        parallel speedup {res[8]/res[1]:.2f}x "
      f"({'idle-machine range is 1.8-2x' if res[8]/res[1] >= 1.3 else 'machine is busy; this is depressed'})")

with ParallelFetcher(STORE, threads=4) as f:
    f.fetch(list(LAYOUT)[:8])
    st = f.stats()
check("stats report bytes actually read", st["bytes_read"] == 8 * EXPERT, str(st))

print("\n" + "=" * 78); print("8. CALIBRATION PICKS A THREAD COUNT FROM MEASUREMENT"); print("=" * 78)
cal = calibrate_threads(STORE, candidates=(1, 4, 8), probe_gb=0.05, reps=2)
check("calibration returns a candidate it actually measured",
      cal["best_threads"] in (1, 4, 8), str(cal))
check("calibration reports a positive throughput", cal["best_gbs"] > 0, str(cal))
print(f"        best {cal['best_threads']} threads at {cal['best_gbs']:.2f} GB/s "
      f"| by_threads {({k: round(v,2) for k,v in cal['by_threads'].items()})}")

print("\n" + "=" * 78)
print("9. THE OS PAGE CACHE IS AN ALLY, NOT A COMPETITOR")
print("=" * 78)
import inspect as _i
_fsrc = _i.getsource(ParallelFetcher)
_sig = _i.signature(ParallelFetcher.__init__)
# F_NOCACHE shipped ON, for the stated reason that caching the same bytes twice double-counts
# memory and pushes a large model into swap. Measured, that is backwards: the expert pool is
# ANONYMOUS memory macOS may swap, while the page cache is FILE-BACKED memory it merely drops.
# Two interleaved A/B pairs on Qwen3-30B at 43 of 128 read 7.62 / 7.51 tok/s with it on and
# 9.52 / 9.07 with it off, and the swapout counter did not move either way.
check("expert reads are allowed into the OS page cache by default",
      _sig.parameters["nocache"].default is False)
check("...and why it used to be otherwise is written down, with the measurement",
      "ANONYMOUS" in _fsrc and "FILE-BACKED" in _fsrc and "9.52" in _fsrc)
check("the flag still exists, because anything MEASURING read cost must bypass the cache",
      "nocache" in _sig.parameters and "a cached read is not a read" in _fsrc)
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_cli = open(os.path.join(_root, "bigrig_engine", "cli.py")).read()
# knee deliberately does NOT opt out: it chooses the capacity the model will be SERVED at, and
# measured cold it picks the wrong one -- 53 beats 43 with the cache bypassed, 43 beats 53 the
# way it actually runs.
check("`bigrig knee` measures the way the model is served, cache and all",
      "the way it will be served" in _cli.lower() or "SERVED, cache and all" in _cli)

print("\n" + "=" * 78)
print("10. READING A REGION AS A VIEW MUST GIVE THE SAME BYTES AS READING IT")
print("=" * 78)
_lay = {("m", i): Region(i * 4096, 4096) for i in range(6)}
_st2 = WeightStore(TMP, _lay)
try:
    with ParallelFetcher(_st2, threads=2, mapped=True) as _fm, \
         ParallelFetcher(_st2, threads=2, mapped=False) as _fr:
        _keys = [("m", 1), ("m", 3), ("m", 5)]
        _m = {k: bytes(v) for k, v in _fm.fetch(_keys).items()}
        _r = {k: bytes(v) for k, v in _fr.fetch(_keys).items()}
    # The whole justification for the mapped path is that it is the same bytes obtained more
    # cheaply -- 0.014 ms against 0.394 in the engine. If it were ever NOT the same bytes it
    # would be a different model, silently.
    check("the mapped path and the read path agree byte for byte", _m == _r)
    check("...for every key, at full length", all(len(v) == 4096 for v in _m.values()))
    check("...and it is actually the mapped path being exercised",
          ParallelFetcher(_st2, threads=1, mapped=True).mapped is True)
    # A map cannot be allowed to hand out a region that is not wholly inside the file: touching
    # a page past the end is SIGBUS, which takes the process down rather than raising.
    check("a region past the end of the map is refused, not truncated",
          _st2.region_view(TMP, os.path.getsize(TMP) - 10, 4096) is None)
    check("...and a negative offset likewise", _st2.region_view(TMP, -1, 16) is None)
    check("a region wholly inside is served",
          len(_st2.region_view(TMP, 0, 4096)) == 4096)
    # Anything MEASURING the disk must not be handed the page cache: the calibrator read
    # 1,267 GB/s through a map against a real RAM bandwidth of 95.6, collapsing kappa to zero.
    check("bypassing the cache also bypasses the map, so the disk can still be timed",
          ParallelFetcher(_st2, threads=1, nocache=True, mapped=True).mapped is False)
finally:
    _st2.close_maps()

os.remove(TMP)
print("\n" + "=" * 78)
print(f"{'ALL TESTS PASSED' if not FAIL else str(len(FAIL)) + ' FAILURES: ' + ', '.join(FAIL)}")
print("=" * 78)
sys.exit(1 if FAIL else 0)
print("\n" + "=" * 80)
print("WARM-THEN-WRAP: A COLD REGION IS READ IN BEFORE THE GPU SEES IT, AND THE BYTES DO NOT CHANGE")
print("=" * 80)
# Measured 2026-09-02: a zero-copy view of a cold region is faulted in by the GPU one page at a
# time, 3.2-3.9 GB/s against a disk that reads 5.34 GB/s in parallel. So a region `mincore` says
# is not resident is read once with pread first. What must hold: the bytes handed out are the
# file's bytes, resident pages are reported resident, and the switch is honoured.
import tempfile as _tf
import numpy as _np
from bigrig_engine import fetch as _fx
_tmp = _tf.mkdtemp()
_fp = os.path.join(_tmp, "w.bin")
_data = _np.random.default_rng(7).integers(0, 255, size=4 * _fx._PAGE, dtype=_np.uint8).tobytes()
open(_fp, "wb").write(_data)
_store = _fx.WeightStore(_fp, {(0, 0): _fx.Region(_fx._PAGE, 2 * _fx._PAGE)})
_v = _store.region_view(_fp, _fx._PAGE, 2 * _fx._PAGE)
check("a region view is exactly the file's bytes, warm read or not",
      _v is not None and bytes(_v) == _data[_fx._PAGE:3 * _fx._PAGE])
check("...and matches a plain pread of the same range",
      bytes(_v) == os.pread(os.open(_fp, os.O_RDONLY), 2 * _fx._PAGE, _fx._PAGE))
_mm, _size, _fd, _base = _store._maps[_fp]
check("a region just written and just read is reported resident",
      _fx.pages_resident(_base + _fx._PAGE, 2 * _fx._PAGE) is True)
check("asking about an address that is not mapped does not raise, it declines to add a read",
      _fx.pages_resident(0x10, 64) in (True, False))
check("the warm-read count is kept, so a run can say how often the cache was cold",
      isinstance(_store.warm_reads, int) and _store.warm_reads >= 0)
check("the mode is read from the environment and is adaptive unless asked: one admit in eight "
      "probed, every admit for 64 after a cold one",
      "BIGRIG_WARM_READ" in open(os.path.join(ROOT, "bigrig_engine", "fetch.py")).read()
      and _fx.WARM_MODE == "auto" and _fx.PROBE_EVERY == 8 and _fx.FULL_AFTER_COLD == 64)
# Warm, the probe must be rare: 800 views of resident regions probe about one in eight.
_st2 = _fx.WeightStore(_fp, {(0, 0): _fx.Region(0, _fx._PAGE)})
for _i in range(800):
    _st2.region_view(_fp, 0, _fx._PAGE)
check("a warm cache is probed about one admit in eight, so the check costs ~0.5%",
      90 <= _st2.warm_probes <= 110, str(_st2.warm_probes))
check("...and a resident region is never read again", _st2.warm_reads == 0)
_st2.close_maps()
check("closing the store closes the file it kept open for warm reads",
      (_store.close_maps(), True)[1] and _fp not in _store._maps)
del _v

