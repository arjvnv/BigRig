# Draft comment on vllm-project/vllm RFC #38256 (Incremental MoE Expert Offloading)

---

We have been running an expert cache with disk backed misses on Apple Silicon for a while, and
four of the things we measured bear directly on the choices in this RFC. Read them as evidence
from a different machine, not as a verdict on yours.

**The transfer caveat first, because it decides how much of this applies.** Every number below is
Apple Silicon unified memory, where the page cache and the GPU share one physical memory and a
missed expert can be handed over without a copy. On a discrete GPU the host to device path is
PCIe and does not contend with compute the way ours does, so the prefetch result in particular
may simply not carry. The eviction and hit rate results are more likely to.

**1. LFRU eviction.** We shipped a variant of this and removed it. Prompt warmed expert pinning
is LFU with no aging, and it measured 1.4x to 2.3x worse than plain LRU on our traces. Frequency
counts collected during prefill go stale fast once generation moves to a different topic, and
without aging the cache defends experts the prompt wanted rather than the ones the reply wants.
If LFRU keeps the F term, the aging half is the part that has to earn its place in a benchmark.

**2. A bigger cache did not buy a better hit rate.** Raising the memory ceiling on Qwen3.6-35B-A3B
and letting the tuner pick, same day, same page cache state:

| ceiling | experts resident per layer the tuner chose | miss rate | decode |
|---|---|---|---|
| 9.7 GB | 38 | 62.5% | 10.5 tok/s |
| 12.0 GB | 38 (tried 43 and 72) | 64.7% at 72 | 9.9 tok/s |
| 14.0 GB | 38 (tried 44 and 77) | | 9.0 to 10.4 tok/s |

Holding twice as many experts moved the miss rate by two points, because routing on this model is
spread thin across the experts. Worth checking whether GPT-OSS-20B's 97 to 100% hit rate is the
model or the method: expert count and routing entropy differ enormously between MoE families, and
a hit rate measured on one is not a property of the design.

**3. The knee is not the ceiling.** Filling residency to the memory ceiling is the wrong policy
even when it does raise the hit rate. On Qwen3-30B, going from 44 to 53 resident experts cost
0.89 GB and bought 0.26 tok/s, while the longest reply the budget still allowed collapsed from
9,820 tokens to 748, because the KV cache and the transient buffers of a miss have to come out of
the same budget. Relevant to the RFC's open question 2: if the GPU slot buffers are invisible
during profiling, this trade is invisible too.

Also worth knowing before sizing anything: peak memory is not monotonic in residency. A pool of
26 experts peaked at 9.07 GB while a pool of 44 peaked at 7.80 GB, because a smaller pool misses
more and every miss is a transient buffer.

**4. The async pipeline is the one we most expected to work, and it did not.** Staging predicted
experts onto the GPU during the layer's own attention, with a good predictor, measured 0.94x to
0.95x on like for like runs. Admit time fell from about 30 ms to 7 ms and 66 of 150 misses a
token arrived already staged, and it was still slower, because the staging copy contends for the
bandwidth the GPU is already using to stream that layer's weights. This is the finding most
likely to be an Apple Silicon artifact, and the one most worth checking early on your hardware,
because the whole of PR 2 rests on it.

The predictor itself is worth having regardless, and it is free. Applying layer L+1's own router
to layer L's hidden state, with nothing fitted and nothing downloaded, one 2048x256 matmul per
layer, measured on Qwen3.6-35B-A3B over 65 decode steps: recall@8 83.6%, recall@16 96.8%,
recall@32 99.1%, weakest in the early layers at 54 to 64%. A ridge map fitted on 546 steps
managed 47.3 / 59.2 / 68.6% on the same trace, so the untrained version was the better one.

**One methodology note.** Our first A/B of staging read 1.28x and was wrong. The baseline had
`mx.async_eval` on the routing read with nothing to stage, so it measured the harm of async_eval
rather than the good of staging. Same process interleaved A/B after that, because the same
configuration in two separate processes disagreed by up to 30% on page cache state alone.

Happy to answer questions on any of these, or to re run something on our side if a specific
comparison would help. Numbers are from BigRig (github.com/arjvnv/BigRig), Apache 2.0, on an
M4 with a 9.7 GB ceiling.
