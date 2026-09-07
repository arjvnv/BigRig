# Draft comment on ml-explore/mlx-lm #1438 (MoE expert streaming / SSD offload)

---

This works, and it has been running for a few months. Posting the measured costs rather than the
pitch, because the interesting part is which of them is actually the bottleneck, and it is not
the disk.

**What holds up.** Keeping the routed experts a model is currently using in RAM and reading the
rest off the SSD does decouple the memory a model needs from its size. What has to stay resident
is the attention weights plus roughly one layer's worth of experts. Six families run end to end
here, 3.9 to 20.4 GB on disk at memory ceilings under 10 GB, from 6.2 to 25.9 tok/s on an M4:
Qwen3, Qwen3.6, GLM-4.7-Flash, DeepSeek-Coder-V2-Lite, Nemotron-3-Nano and OLMoE.

**The dominant cost is not the SSD.** To know which expert to fetch you have to read the router's
output back from the GPU, and that read stops MLX pipelining and drains the queue. On
Qwen3.6-35B-A3B at a 9.7 GB ceiling, a 58 ms decode token is about 63% waiting on that read, one
per streamed layer, against 23 ms for the same model's kernels with no host reads at all. The
read itself is 0.2 to 0.35 ms. It is the serialisation that costs, and it scales with layer
count, so a 48 layer model pays it 48 times per token. Any design that plans around disk
bandwidth alone will predict speeds it does not reach.

**Four things that surprised us, in case they save time.**

Zero copy is the single biggest lever, and it is an Apple Silicon specific one. The page cache
and the GPU share one memory, so a missed expert's pages can be wrapped as a Metal buffer instead
of copied. That took a token from swinging between 53 and 95 ms depending on cache state to a
steady 47 ms. It needs each expert to be one page aligned region, and in raw safetensors shards
0 of 360 expert tensors start on a page boundary, so it costs a second, repacked copy on disk.

More memory for the pool did not buy speed. Raising the ceiling from 9.7 to 14 GB, the tuner
still chose 38 resident experts per layer, and 72 experts moved the miss rate from 62.5% to 64.7%
because routing on that model is spread thin. Memory not spent on the pool is page cache, and the
page cache is what serves the misses. A warm cache is the lever, not a bigger pool.

Filling to the ceiling is the wrong policy anyway. On Qwen3-30B, 44 to 53 resident experts bought
0.26 tok/s and collapsed the longest reply the budget allowed from 9,820 tokens to 748, because
the KV cache comes out of the same budget.

Prefetching did not pay, and we tried hard. Staging predicted experts during the layer's own
attention measured 0.94x to 0.95x even with a good predictor, because the staging copy contends
for the bandwidth the GPU is already using. The predictor is free and worth knowing about
separately: layer L+1's own router applied to layer L's hidden state gives recall@8 83.6% and
recall@32 99.1%, nothing fitted, one small matmul per layer.

**Exactness, since streaming invites the question.** Weights are untouched. Decode is bit
identical to the same model held resident, checked layer by layer across 4,392 live layer
computations. Prompt processing matches to the last bit or one bit off it, which is the same
variation mlx_lm itself shows between two `prefill_step_size` values.

**Where this does not answer the question in this issue.** Every number above is one M4 with 24
GB. We have never run a 120B class model, let alone 395 GB on 128 GB, so we cannot vouch for that
case. The static answer for gpt-oss-120b at 65.8 GB on disk is a prediction from measured machine
bandwidth, not a run. Ratios that large are exactly where a prediction is least trustworthy.

**On whether this belongs in mlx-lm.** Most of it probably does not. What is genuinely the
library's is small: a way to load a model without materialising the routed experts, and a hook
per MoE layer to fetch the experts an index vector asks for. The rest is a cache policy, a memory
controller, a repacking step and a tuner, which is a lot of surface for a library that is
deliberately simple. Two small hooks would let this and the other implementations live outside
without each one monkeypatching `switch_layers`.

Ours is at github.com/arjvnv/BigRig, Apache 2.0, `pip install bigrig`. Happy to answer anything,
and happy to be told the hooks idea is wrong.
