"""COMPONENT 8 — the parallel fetch engine.

WHAT IT DOES
    Pulls expert weights off the SSD into RAM when the model needs them: decides what to fetch,
    issues many reads at once so the device runs at full speed instead of a fraction of it, and
    lets the caller overlap those fetches with computation.

WHY IT EXISTS, IN ONE MEASURED NUMBER
    Reading the same cold bytes off this machine's SSD:

        1 thread, mmap (what llama.cpp does)              0.70 GB/s
        1 thread, explicit read()                         3.40 GB/s
        8 threads, explicit read(), random offsets,
          while a large model occupies RAM                5.96 GB/s     <- what this delivers

    A single synchronous read leaves the device at queue depth 1, and NVMe hardware is built to
    be fed from many outstanding requests at once. Everything above the fetch engine -- eviction,
    the toll, prefetch -- only decides HOW OFTEN we come here. This decides how much it costs
    when we do.

    The 5.96 GB/s figure is the HARDEST case measured, not the best: random expert-sized offsets
    under memory pressure. Sequential reads on an idle machine reached 23 GB/s, and designing to
    that number would be designing for a situation the engine never sees.

WHAT IT DELIBERATELY DOES NOT DO
    No caching, no eviction, no prediction. It fetches exactly what it is asked for. Deciding
    what to ask for is component 9's job, and mixing the two makes both untestable.
"""
from __future__ import annotations

import fcntl
import ctypes
import mmap
import os

import numpy as np
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass

F_NOCACHE = 48                 # macOS: read around the unified buffer cache
DEFAULT_THREADS = 8            # measured optimum on M4; calibrate_threads() re-measures per host
DEFAULT_CHUNK = 4 << 20        # per-syscall read size


class _Seg:
    """A byte range in a named file. Interchangeable with a Region bound to the store's path."""
    __slots__ = ("path", "offset", "length")

    def __init__(self, path, offset, length):
        self.path, self.offset, self.length = path, offset, length


@dataclass(frozen=True)
class Region:
    """Where one expert's bytes live in a weight file."""
    offset: int
    length: int

    def __post_init__(self):
        if self.offset < 0 or self.length <= 0:
            raise ValueError(f"invalid region offset={self.offset} length={self.length}")


# READ THE PAGES IN BEFORE THE GPU TOUCHES THEM, WHEN THEY ARE NOT THERE YET.
#     A zero-copy view of a region whose pages are not in the page cache is faulted in by the GPU
#     one page at a time. Measured on a file written with F_NOCACHE (measured 2026-09-02):
#     expert-sized regions reach the GPU at 1.53 GB/s that way, at 3.35 GB/s if `pread` first
#     through the page cache, and at 9.29 GB/s once cached. So a region that `mincore` reports as
#     not resident is read once with `pread`, on the fetch pool's own threads, and only then
#     handed out as a view. Same bytes by construction.
#
#     ADAPTIVE, BECAUSE THE CHECK IS NOT FREE ON A WARM CACHE. Asking `mincore` on every admit
#     cost 20.5 -> 19.7 tok/s (about 230 calls a token) when everything was already resident. The
#     admit timer cannot see a cold fault -- the GPU work is lazy -- so the signal is the check
#     itself, sampled: one admit in eight is probed, and a probe that finds a cold region switches
#     to probing every admit for the next 64, because a cache that is cold here is cold next door.
#     Warm, that is one call in eight, about 0.5%. BIGRIG_WARM_READ=1 probes every admit, =0 never.
WARM_MODE = {"1": "on", "0": "off"}.get(os.environ.get("BIGRIG_WARM_READ", ""), "auto")
PROBE_EVERY = 8
FULL_AFTER_COLD = 64
_PAGE = os.sysconf("SC_PAGESIZE")
_libc = None


def pages_resident(addr: int, length: int) -> bool:
    """True if every page of [addr, addr+length) is in memory. True on any failure to ask, so a
    machine without mincore behaves exactly as before."""
    global _libc
    try:
        if _libc is None:
            lib = ctypes.CDLL(None)
            lib.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_char_p]
            lib.mincore.restype = ctypes.c_int
            _libc = lib
        start = addr - (addr % _PAGE)
        n = (addr + length - start + _PAGE - 1) // _PAGE
        vec = ctypes.create_string_buffer(n)
        if _libc.mincore(start, n * _PAGE, vec) != 0:
            return True
        return all(b & 1 for b in vec.raw)
    except Exception:                            # noqa: BLE001 -- never let a hint become a fault
        return True


class WeightStore:
    """A weight file plus a map from key to byte region.

    The layout is supplied rather than parsed here, so this works against any file format --
    a repacked expert-contiguous blob (component 7), a GGUF, or raw safetensors.
    """

    def __init__(self, path: str, layout: dict, verify: bool = True):
        # `path` may be empty when every key carries its own file -- that is how the engine reads
        # experts straight out of a model's own safetensors shards instead of a packed copy.
        self.path = os.path.expanduser(path) if path else ""
        if self.path:
            if not os.path.exists(self.path):
                raise FileNotFoundError(self.path)
            self.size = os.path.getsize(self.path)
        else:
            self.size = 0
        self.layout = dict(layout)
        # One memory map per file, created on first use and never remapped. See `region_view`.
        self._maps = {}
        self._map_lock = threading.Lock()
        self.warm_reads = 0                   # regions read in ahead of the GPU because they were cold
        self.warm_probes = 0                  # residency checks made
        self._view_n = 0                      # admits seen, for the sampled probe
        self._full_until = 0                  # probe every admit until this count
        if verify:
            sizes = {}
            for k in self.layout:
                for seg in self.segments(k):
                    if seg.path not in sizes:
                        if not os.path.exists(seg.path):
                            raise FileNotFoundError(seg.path)
                        sizes[seg.path] = os.path.getsize(seg.path)
                    if seg.offset + seg.length > sizes[seg.path]:
                        raise ValueError(
                            f"region for {k!r} runs past the end of "
                            f"{os.path.basename(seg.path)}: {seg.offset}+{seg.length} > "
                            f"{sizes[seg.path]}")

    def region(self, key) -> Region:
        try:
            r = self.layout[key]
        except KeyError:
            raise KeyError(f"{key!r} is not in this store's layout") from None
        if isinstance(r, Region):
            return r
        # A multi-segment key has no single (offset, length); callers that need one are asking
        # a question that does not apply, and a silently-wrong answer is worse than an error.
        raise TypeError(
            f"{key!r} spans {len(r)} segments across the model's own files; use segments() "
            f"instead of region()")

    def region_view(self, path: str, offset: int, length: int):
        """A zero-copy memoryview of one region, backed by a memory map of `path`.

        WHY THIS EXISTS: THE READ PATH COST 5.4x MORE THAN THE BYTES DO.
            Reading an expert used to be lseek, then a read loop into a bytearray, then
            bytes(buf), then np.frombuffer, then mx.array -- three separate copies of 2 MB to
            move 2 MB. Timed on the real blob, same experts, same cache state:

                the shipped path (lseek + read loop + bytearray + mx)   0.300 ms
                pread straight to mx.array                              0.158 ms
                mmap slice to mx.array                                  0.056 ms
                pread alone, building nothing                           0.104 ms

            A map is faster than even a bare read because a page-cache hit becomes a pointer
            rather than a syscall. At about 140 reads a token that is roughly 34 ms of a 115 ms
            token.

        WHY THIS CANNOT READ THE WRONG BYTES.
            Same file, same offset, same length -- the bytes are identical by construction, and
            `_read_region` asserts the length it got. What a map CAN do that a read cannot is
            take the process down with SIGBUS if the file is truncated underneath it. Two things
            stop that: WeightStore's constructor has already checked every region against the
            real file size, and the map is created once and never grown, so a file that changes
            later cannot silently extend what this hands out.
        """
        with self._map_lock:
            ent = self._maps.get(path)
            if ent is None:
                fd = os.open(path, os.O_RDONLY)
                size = os.fstat(fd).st_size
                if size == 0:
                    os.close(fd)
                    return None
                mm = mmap.mmap(fd, size, access=mmap.ACCESS_READ)
                # The map's base address, for `mincore`; taken once and the array let go, so
                # nothing here keeps a buffer export open against `mm.close()`.
                base = int(np.frombuffer(mm, dtype=np.uint8, count=1).ctypes.data)
                ent = (mm, size, fd, base)          # the fd stays open for the warm read
                self._maps[path] = ent
        mm, size, fd, base = ent
        view = self._slice(mm, size, offset, length)
        if view is not None and WARM_MODE != "off":
            # Counters are approximate under the fetch pool's threads, and that is fine: they
            # decide how often to look, never what bytes are handed out.
            self._view_n += 1
            n = self._view_n
            if WARM_MODE == "on" or n <= self._full_until or n % PROBE_EVERY == 0:
                self.warm_probes += 1
                if not pages_resident(base + offset, length):
                    self._full_until = n + FULL_AFTER_COLD
                    try:
                        os.pread(fd, length, offset)   # fills the page cache; the bytes are dropped
                        self.warm_reads += 1
                    except OSError:
                        pass                           # the view is still right; it will just fault
        return view

    @staticmethod
    def _slice(mm, size: int, offset: int, length: int):
        """None rather than a short view if the region is not wholly inside the mapped file.

        Returning a short view would hand the caller fewer bytes than it asked for and let the
        length check downstream decide -- which is right, but a caller that then falls back to a
        read is safer than one that has already touched a page past the end of the file.
        """
        if offset < 0 or length < 0 or offset + length > size:
            return None
        return memoryview(mm)[offset:offset + length]

    def close_maps(self) -> None:
        with self._map_lock:
            for mm, _size, fd, _base in self._maps.values():
                try:
                    mm.close()
                except (BufferError, ValueError):
                    pass          # a view is still outstanding; the map dies with the process
                try:
                    os.close(fd)
                except OSError:
                    pass
            self._maps.clear()

    def segments(self, key) -> list:
        """Every byte range this key is made of, in the order they concatenate."""
        try:
            r = self.layout[key]
        except KeyError:
            raise KeyError(f"{key!r} is not in this store's layout") from None
        if isinstance(r, Region):
            return [_Seg(self.path, r.offset, r.length)]
        return list(r)

    def nbytes(self, key) -> int:
        return sum(s.length for s in self.segments(key))

    def __len__(self):
        return len(self.layout)


class ParallelFetcher:
    """Fetches byte regions in parallel. Thread-safe; one instance per store is enough.

    `nocache=True` bypasses the OS buffer cache with F_NOCACHE.

    IT SHIPPED ON, AND THAT COST 1.23x. The reasoning was that an engine managing its own
    residency should not let macOS cache the same bytes, because that double-counts memory and
    pushes a large model into swap. Two of those three claims turned out to be wrong once the
    memory classes were actually measured:

        the expert pool is ANONYMOUS memory       macOS may swap it, and will
        the page cache is FILE-BACKED             macOS drops it, and never swaps it

    So the page cache is the better place for a byte to live, not the worse one. Measured on
    Qwen3-30B at 43 of 128, two interleaved A/B pairs:

        F_NOCACHE on     7.62, 7.51 tok/s      inactive 7.3-8.1 GB
        cache allowed    9.52, 9.07 tok/s      inactive 9.2-9.6 GB, swapouts unchanged

    The cached bytes land in `inactive`, which is exactly the memory macOS reclaims first, so
    this buys a multi-gigabyte second-tier expert cache that costs us nothing under pressure.

    The one claim that survives is the third: a cached read is not a read, so anything MEASURING
    read cost must set this True. `bigrig knee` and `bigrig calibrate` do.
    """

    def __init__(self, store: WeightStore, threads: int = DEFAULT_THREADS,
                 chunk: int = DEFAULT_CHUNK, nocache: bool = False,
                 mapped: bool = True):
        if threads < 1:
            raise ValueError(f"threads must be >= 1, got {threads}")
        self.store, self.threads, self.chunk, self.nocache = store, threads, chunk, nocache
        # Reading a region as a view of the file rather than copying it three times. On by
        # default; `mapped=False` restores the read path, which is what the A/B is run with.
        self.mapped = bool(mapped) and not nocache
        self._pool = ThreadPoolExecutor(max_workers=threads,
                                        thread_name_prefix="bigrig-fetch")
        self._inflight: dict = {}
        self._lock = threading.Lock()
        self._closed = False
        self.bytes_read = 0
        self.seconds = 0.0
        self.n_fetches = 0
        # A prefetch only pays if it has FINISHED by the time the read is actually wanted.
        # One that is still in flight was issued too late to have hidden anything.
        self.hit_done = 0          # wanted, already read
        self.hit_pending = 0       # wanted, issued but not finished
        self.cold = 0              # wanted, never speculated on

    # ------------------------------------------------------------------ core
    def _read_region(self, key):
        """Read every segment of `key` and concatenate them, in layout order.

        A packed blob gives one segment; reading straight from a model's own shards gives nine
        (three projections x weight/scales/biases), possibly in different files. The bytes that
        come out must be identical either way -- that equivalence is what lets the engine skip
        making a second copy of the model, and test_direct.py asserts it byte for byte.
        """
        segs = self.store.segments(key)
        # THE MAPPED PATH, TRIED FIRST AND NEVER TRUSTED BLINDLY.
        #     A view of the file is 5.4x cheaper than reading it (see WeightStore.region_view).
        #     If any segment cannot be served from a map -- no map, a region past the end, a
        #     platform without mmap -- the whole key falls back to the read path below rather
        #     than mixing the two, so a key is never assembled from two different sources.
        if self.mapped:
            views = []
            for seg in segs:
                v = self.store.region_view(seg.path, seg.offset, seg.length)
                if v is None or len(v) != seg.length:
                    views = None
                    break
                views.append(v)
            if views is not None:
                if len(views) == 1:
                    return views[0]
                return b"".join(bytes(v) for v in views)
        parts, opened = [], {}
        try:
            for seg in segs:
                fd = opened.get(seg.path)
                if fd is None:
                    fd = os.open(seg.path, os.O_RDONLY)
                    opened[seg.path] = fd
                    if self.nocache:
                        try:
                            fcntl.fcntl(fd, F_NOCACHE, 1)
                        except OSError:
                            pass              # not fatal; the read still works, just cached
                os.lseek(fd, seg.offset, os.SEEK_SET)
                buf = bytearray(seg.length)
                got = 0
                while got < seg.length:
                    b = os.read(fd, min(self.chunk, seg.length - got))
                    if not b:
                        raise IOError(
                            f"short read for {key!r}: got {got} of {seg.length} bytes at "
                            f"offset {seg.offset} in {os.path.basename(seg.path)}. The file "
                            f"may have been truncated or replaced.")
                    buf[got:got + len(b)] = b
                    got += len(b)
                parts.append(bytes(buf))
        finally:
            for fd in opened.values():
                os.close(fd)
        return parts[0] if len(parts) == 1 else b"".join(parts)

    def _submit(self, keys):
        """Return {key: Future}, reusing any fetch already in flight for the same key."""
        out, fresh = {}, []
        with self._lock:
            if self._closed:
                raise RuntimeError("fetcher is closed")
            for k in keys:
                f = self._inflight.get(k)
                if f is None:
                    f = Future()
                    self._inflight[k] = f
                    fresh.append((k, f))
                out[k] = f
        for k, fut in fresh:
            self._pool.submit(self._run, k, fut)
        return out

    def _run(self, key, fut):
        """Complete the future and LEAVE IT in `_inflight` for the caller to collect.

        The first version popped the key here, which made prefetch a no-op: the future was
        discarded the moment it completed, so the later fetch() created a fresh one and read the
        same bytes again. The test caught it -- a prefetched fetch took 9.1 ms against 8.9 ms
        cold, when it should have been near zero. Completed entries are now retained until
        fetch() or drop() takes them.
        """
        try:
            fut.set_result(self._read_region(key))
        except BaseException as e:            # noqa: BLE001 - the waiter re-raises it
            fut.set_exception(e)

    # ------------------------------------------------------------------ api
    def prefetch(self, keys, max_pending: int = 512) -> None:
        """Start fetching, return immediately. Call `fetch` later for the bytes.

        This is the whole point of the engine: issue layer L+1's reads while layer L computes.

        Unclaimed prefetches hold their buffers, so `max_pending` caps how many may be
        outstanding. Exceeding it raises rather than quietly consuming memory -- an engine whose
        prefetcher can grow without bound is the same failure as no eviction policy at all.
        """
        keys = list(keys)
        with self._lock:
            if len(self._inflight) + len(keys) > max_pending:
                raise RuntimeError(
                    f"{len(self._inflight)} prefetches already pending; {len(keys)} more would "
                    f"exceed max_pending={max_pending}. Call fetch() or drop() first.")
        self._submit(keys)

    def fetch(self, keys) -> dict:
        """Fetch every key, in parallel, and block until all are present.

        Duplicate keys are collapsed, and a key already prefetched is not read twice -- its
        pending or completed future is collected instead. Collecting removes it, so the buffers
        a prefetch is holding are released once the caller has them.
        """
        keys = list(keys)
        if not keys:
            return {}
        with self._lock:
            for k in keys:
                f = self._inflight.get(k)
                if f is None:
                    self.cold += 1
                elif f.done():
                    self.hit_done += 1
                else:
                    self.hit_pending += 1
        t0 = time.perf_counter()
        futures = self._submit(keys)
        out = {k: f.result() for k, f in futures.items()}      # re-raises worker exceptions
        dt = time.perf_counter() - t0
        with self._lock:
            for k in futures:
                self._inflight.pop(k, None)                    # consumed
            self.bytes_read += sum(len(v) for v in out.values())
            self.seconds += dt
            self.n_fetches += 1
        return out

    def drop(self, keys=None) -> int:
        """Discard prefetched-but-unclaimed buffers. Returns how many were dropped.

        A prefetch that is never fetched holds its bytes forever, so a caller that speculatively
        prefetches -- which is the point -- needs a way to abandon the guesses that were wrong.
        """
        with self._lock:
            ks = list(self._inflight) if keys is None else [k for k in keys if k in self._inflight]
            for k in ks:
                f = self._inflight.get(k)
                if f is not None and f.done():
                    self._inflight.pop(k, None)
            return len(ks)

    def pending(self) -> int:
        """Keys prefetched and not yet collected."""
        with self._lock:
            return len(self._inflight)

    def stats(self) -> dict:
        with self._lock:
            gbs = (self.bytes_read / self.seconds / 1e9) if self.seconds > 0 else 0.0
            return {"bytes_read": self.bytes_read, "gb_read": self.bytes_read / 1e9,
                    "seconds": self.seconds, "throughput_gbs": gbs,
                    "n_fetches": self.n_fetches, "threads": self.threads}

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._pool.shutdown(wait=True)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


# ---------------------------------------------------------------------- calibration
def calibrate_threads(store: WeightStore, candidates=(1, 2, 4, 8, 12),
                      probe_gb: float = 0.5, reps: int = 3) -> dict:
    """Measure this HOST's best thread count instead of shipping the M4's.

    The optimum is a property of the device and the OS, not of the model. On this machine 8 was
    best and 16 was worse than 8 -- past the knee, extra threads only add contention. A different
    SSD will have a different knee, and shipping 8 as a constant is the same class of mistake as
    shipping kappa=67.
    """
    keys = list(store.layout)
    if not keys:
        raise ValueError("store has an empty layout; nothing to calibrate against")
    per_key = store.region(keys[0]).length
    need = max(1, int(probe_gb * 1e9 // max(per_key, 1)))
    results = {}
    for rep in range(reps):
        order = candidates if rep % 2 == 0 else tuple(reversed(candidates))
        for t in order:
            sel = [keys[(i * 7 + rep * 3) % len(keys)] for i in range(min(need, len(keys)))]
            with ParallelFetcher(store, threads=t) as f:
                t0 = time.perf_counter()
                got = f.fetch(sel)
                dt = time.perf_counter() - t0
            gbs = sum(len(v) for v in got.values()) / dt / 1e9
            results.setdefault(t, []).append(gbs)
    med = {t: sorted(v)[len(v) // 2] for t, v in results.items()}
    best = max(med, key=med.get)
    return {"best_threads": best, "best_gbs": med[best], "by_threads": med,
            "speedup_vs_1": med[best] / med[min(med)] if min(med) in med else 1.0}
