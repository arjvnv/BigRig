# Draft comment on vllm-project/vllm-metal #713 (macOS benchmarking pitfalls)

---

All three of these match what we hit independently, and the clock parking one cost us a published
claim: we reported that MLX reaches only 16% of memory bandwidth, and it was our own measurement
error from timing ops individually, which charges each one a GPU start and stop. The real figure
on a live forward pass is 62%. So a vote of confidence on pitfall 3 from a second machine.

Four more traps that are not on the list, each one from a wrong number we published internally
before catching it.

**1. Peak memory is a high water mark, so the second configuration in a process inherits the
first one's peak.** Measuring two settings in one process reports the larger of the two for both.
This is the memory equivalent of your in process decode penalty and it silently flatters
whichever configuration you measure second. Separate processes, or reset the counter and prove
the reset works.

**2. Peak memory is not monotonic in cache size.** We assumed a smaller expert pool would peak
lower. A pool of 26 experts per layer peaked at 9.07 GB and a pool of 44 peaked at 7.80 GB,
because the smaller pool misses more and every miss is a transient buffer. Anything that tunes a
cache size against a memory ceiling has to measure the peak at each point rather than infer it.

**3. Per process RSS does not see MLX memory, and the direction is not the obvious one.** On a
live server holding a 4.64 GB expert pool, RSS read 1.03 GB, and allocating 512 MB on the GPU
moved RSS by 0.031 GB. A watchdog reading RSS concludes it has room it does not have. What
matched reality for us was `phys_footprint` from libproc, `rusage_info_v4` field f7, with
`max(rss, mx.get_active_memory())` as the cheap version.

The second half of this is worth stating because we recorded it backwards for two weeks: the pool
is not wired. Across a server start, `vm_stat` moved wired by +0.06 GB and anonymous by +5.00 GB,
and 1.55 GB of it landed in inactive, the pages macOS reclaims first. So under pressure the OS
can quietly turn resident weights into compressed or swapped ones, which converts every cache hit
into the disk read you were avoiding, and nothing errors. Judge health from compressor growth and
swapouts over a short window, never from a static threshold and never from RSS.

**4. Page cache state moves decode by 30% on its own, so A and B have to be interleaved in one
process.** Same code, same model, same prompt: 77.2 ms per token in one process and 53.0 ms in
the next. Every A/B we ran across processes was noise. Related and more embarrassing: our first
A/B of an expert staging feature read 1.28x, and its baseline had a different async primitive
enabled with nothing to stage, so it measured the harm of the baseline rather than the good of
the change. Interleave, and state what the baseline actually is.

**A small one that costs an hour every time.** Detach stdin on any background benchmark run.
A hidden prompt hung a job for an hour at 0% CPU while looking perfectly alive. Monitor for alive
and 0% CPU, not just alive.

**On probe design, agreeing with your pitfall 3.** Our own microbenchmark compares three matrix
shapes whose timings sit within 15% of each other. It passed three of three idle and failed once
inside a full suite run, 13.65 against 18.09 and 17.74 microseconds per unit, because anything
else on the Mac inflates whichever round it lands in. It takes the best of three rounds now
rather than the mean of one, which is how to time on a machine you do not own.

Happy to write any of this up for a "Benchmarking on macOS" page if that is the form you want,
or to leave it here as notes for whoever writes it. All of it is from BigRig
(github.com/arjvnv/BigRig), Apache 2.0, measured on an M4.
