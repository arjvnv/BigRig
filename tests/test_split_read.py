"""Split cold reads (fetch.SPLIT_PARTS): one cold expert read as several page-aligned requests at once.

WHY THIS FILE EXISTS
    Decode waits on about one cold expert read per layer, and a single 1.77 MB pread runs at
    queue depth 1. WeightStore.fill splits it into parts issued together. The parts must cover the
    region exactly (no page skipped, none read by two parts), small regions must stay one request,
    the switch must turn it off, and the view handed out must be the file's bytes whichever way
    the page cache was filled.

Run: .venv/bin/python tests/test_split_read.py
"""
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from bigrig_engine import fetch  # noqa: E402

FAIL = []


def check(n, c, d=""):
    (print(f"  PASS  {n}") if c else (FAIL.append(n), print(f"  FAIL  {n}  {d}")))


PAGE, EXPERT = 16384, 1769472

print("=" * 78)
print("1. THE PARTS COVER THE REGION EXACTLY, CUT ON PAGES")
print("=" * 78)
for off, n, k in ((0, EXPERT, 4), (EXPERT * 37, EXPERT, 4), (EXPERT * 3, EXPERT, 3), (100, EXPERT, 4),
                  (5 * PAGE, 7 * PAGE + 11, 4)):
    r = fetch.split_ranges(off, n, k, PAGE)
    covers = r[0][0] == off and sum(m for _, m in r) == n and all(a + m == b for (a, m), (b, _) in zip(r, r[1:]))
    cuts = all(b % PAGE == 0 for b, _ in r[1:])
    count = len(r) <= k + (1 if off % PAGE else 0)
    check(f"offset {off}, {n} bytes, {k} parts: contiguous, exact, page cuts, {len(r)} parts",
          covers and cuts and count, r)
check("an aligned expert splits into equal parts", fetch.split_ranges(0, EXPERT, 4, PAGE) ==
      [(i * EXPERT // 4, EXPERT // 4) for i in range(4)])
check("one part, or a region of a page or less, is a single request",
      fetch.split_ranges(0, EXPERT, 1, PAGE) == [(0, EXPERT)] and fetch.split_ranges(0, PAGE, 4, PAGE) == [(0, PAGE)])

print("=" * 78)
print("2. A COLD REGION IS FILLED BY ITS PARTS, AND THE VIEW IS STILL THE FILE'S BYTES")
print("=" * 78)
data = os.urandom(4 * EXPERT)
with tempfile.NamedTemporaryFile(delete=False) as f:
    f.write(data)
path = f.name
calls = []
real_pread, real_resident, real_mode = os.pread, fetch.pages_resident, fetch.WARM_MODE


def spy(fd, n, o):
    calls.append((o, n))
    return real_pread(fd, n, o)


try:
    os.pread = spy
    fetch.pages_resident = lambda addr, length: False           # every probe finds the region cold
    fetch.WARM_MODE = "on"
    store = fetch.WeightStore(path, {("x", e): fetch.Region(e * EXPERT, EXPERT) for e in range(4)})
    v = store.region_view(path, 2 * EXPERT, EXPERT)
    check("the view is the file's bytes", bytes(v) == data[2 * EXPERT:3 * EXPERT])
    check(f"the cold region was read as {fetch.SPLIT_PARTS} parts covering it exactly",
          sorted(calls) == fetch.split_ranges(2 * EXPERT, EXPERT, fetch.SPLIT_PARTS), sorted(calls))
    check("...and counted once as a cold read", store.warm_reads == 1)
    calls.clear()
    store.region_view(path, 0, 300_000)
    check("a region under SPLIT_MIN_BYTES is one request", calls == [(0, 300_000)], calls)
    calls.clear()
    fetch.SPLIT_PARTS = 1
    store.region_view(path, EXPERT, EXPERT)
    check("BIGRIG_SPLIT_READ=1 reads the region with one request", calls == [(EXPERT, EXPERT)], calls)
    calls.clear()
    fetch.SPLIT_PARTS = 4
    check("refresh reports a cold region as cold", store.refresh(("x", 3)) == 1)
    check("...and reads all of it", sorted(calls) == fetch.split_ranges(3 * EXPERT, EXPERT, 4), sorted(calls))
    fetch.pages_resident = lambda addr, length: True
    calls.clear()
    check("refresh of a cached region reports nothing cold", store.refresh(("x", 1)) == 0)
    check("...but still reads it, so it counts as recently used", sum(n for _, n in calls) == EXPERT, calls)
    fetch.pages_resident = lambda addr, length: False
    calls.clear()
    check("warm_region reads a cold region and says so", store.warm_region(path, 0, EXPERT) is True
          and sorted(calls) == fetch.split_ranges(0, EXPERT, 4), sorted(calls))
    fetch.pages_resident = lambda addr, length: True
    calls.clear()
    check("warm_region leaves a cached region alone", store.warm_region(path, 0, EXPERT) is False and calls == [])
    pf = fetch.ParallelFetcher(store, threads=2)
    fetch.pages_resident = lambda addr, length: False
    calls.clear()
    n = pf.warm([("x", 0), ("x", 1)], max_queued=1)
    pf.close()
    check("warm() queues at most max_queued regions and drops the rest, never blocking",
          n == 1 and pf.warm_issued == 1, (n, pf.warm_issued))
    check("...and the queued one was read before close() returned", pf.warm_read == 1 and len(calls) == 4, calls)
    def broken(fd, n, o):
        raise OSError(5, "I/O error")
    os.pread = broken
    check("refresh survives a read error (it only decides what is cached)", store.refresh(("x", 2)) == 1)
    os.pread = spy
    store.close_maps()
    check("closing the store stops the part pool", store._split_pool is None)
finally:
    os.pread, fetch.pages_resident, fetch.WARM_MODE = real_pread, real_resident, real_mode
    os.unlink(path)

print("\n" + "=" * 78)
print(f"{'ALL TESTS PASSED' if not FAIL else str(len(FAIL)) + ' FAILURES: ' + ', '.join(FAIL)}")
print("=" * 78)
sys.exit(1 if FAIL else 0)
