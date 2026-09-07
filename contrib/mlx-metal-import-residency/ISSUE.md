# Draft issue for ml-explore/mlx

**Title:** Docs: an imported buffer is made resident whole, so `mx.from_dlpack` over a large mapped region wires all of it

---

`mx.from_dlpack` over mapped file pages is the natural way to hand host memory to the GPU on
Apple Silicon, and it works well. What is not written down anywhere is the unit of residency:
Metal makes a referenced buffer resident as a whole, so touching one small slice of a large
imported buffer wires the entire buffer.

This is easy to get wrong in the appealing direction. Importing one big region and indexing into
it is the obvious design, it is faster than importing pieces, and it looks correct right up to
the point where the region is large.

### Measured

Reproducer below. It writes a temp file, imports it two ways, reads the same 2 MB from each, and
reads `vm_stat` wired pages around the kernel. Each trial runs in a fresh subprocess, because
releasing an import is deferred and otherwise lands inside the next measurement. Numbers are
marginal over a bare MLX process, M4, macOS 26.3.1, mlx 0.32.2.

| file | import | wired, marginal | share of the buffer |
|---|---|---|---|
| 403 MB | whole file as one buffer | +401.0 MB | 99.6% |
| 403 MB | the 2 MB slice as its own buffer | +0.0 MB | 0.0% |
| 1074 MB | whole file as one buffer | +1090.9 MB | 101.6% |
| 1074 MB | the 2 MB slice as its own buffer | +0.0 MB | 0.0% |

Reading 2 MB costs the whole buffer, and it scales with the buffer, not with what is read.

### Why it is worth a line in the docs

The workload this came from streams Mixture of Experts weights off an SSD. Importing the whole
17 GB expert file once and slicing per expert is the design you reach for first. On real layer
sized regions, one 453 MB view alive per layer, the command buffer fails with Insufficient
Memory, at the default wired limit and at a raised one alike, and the failure gives no hint that
residency granularity is the cause.

The rule that works is: make the unit of import the unit you actually use. One buffer per expert,
1.77 MB at a time, released with the array, costs nothing measurable.

A sentence on `mx.from_dlpack`, or in the unified memory page, saying that residency is per
buffer and that importing a large region makes all of it resident on first use, would have saved
the whole detour. Happy to send that docs PR if you tell me where you would like it.

### Environment

Apple M4, macOS 26.3.1, mlx 0.32.2, Python 3.12. `repro.py` in this folder, standalone, exits
non zero if it does not reproduce.
