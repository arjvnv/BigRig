# Draft issue for ml-explore/mlx

**Title:** `gather_qmm`: `sorted_indices` changes the result, and the `False` path is about 4x less accurate

---

`sorted_indices` is documented as a performance hint: "May allow a faster implementation if the
passed indices are sorted." On identical, already sorted indices the two settings do not return
the same values, and `sorted_indices=False` lands about 4.4x further from an exact reference.

Reproducer attached below. It needs only mlx and numpy, no model, and runs in a few seconds.

### What is measured

Weights are quantised once. The reference is computed in float64 in numpy from `mx.dequantize`
of those same quantised weights, so quantisation error cancels and only the kernel's own
arithmetic is scored. Relative error is RMS of the difference over RMS of the reference.

E=32, 8 rows per expert, 4 bit affine, group size 64:

| dtype | N | K | rel err sorted=True | rel err sorted=False | ratio | max abs diff |
|---|---|---|---|---|---|---|
| float16 | 512 | 2048 | 2.072e-04 | 9.116e-04 | 4.40x | 0.19141 |
| float16 | 2048 | 512 | 2.076e-04 | 9.228e-04 | 4.45x | 0.12500 |
| float16 | 1024 | 1024 | 2.073e-04 | 9.276e-04 | 4.47x | 0.15625 |
| bfloat16 | 512 | 2048 | 1.655e-03 | 7.295e-03 | 4.41x | 2.00000 |
| bfloat16 | 2048 | 512 | 1.659e-03 | 7.531e-03 | 4.54x | 1.00000 |
| bfloat16 | 1024 | 1024 | 1.661e-03 | 7.514e-03 | 4.52x | 1.12500 |
| float32 | 512 | 2048 | 7.579e-07 | 2.419e-07 | 0.32x | 0.00032 |

float32 is unaffected. The ratio is stable across five seeds: 4.40, 4.50, 4.41, 4.48, 4.46.

### Which of the two is correct

A third, independent path, per expert `mx.quantized_matmul` over the same rows and the same
weights, float16 512x2048:

| path | rel err |
|---|---|
| `mx.quantized_matmul`, per expert | 2.704e-04 |
| `gather_qmm`, `sorted_indices=True` | 2.072e-04 |
| `gather_qmm`, `sorted_indices=False` | 9.116e-04 |

`quantized_matmul` sits with `sorted_indices=True`. The unsorted path is the outlier.

### When it appears

It needs at least 16 total rows and at least 4 rows per expert. Below either of those, both
settings return the same values.

| experts | rows per expert | total rows | ratio |
|---|---|---|---|
| 32 | 3 | 96 | 1.00x |
| 32 | 4 | 128 | 4.47x |
| 2 | 4 | 8 | 1.00x |
| 4 | 4 | 16 | 4.60x |
| 1 | 16 | 16 | 4.57x |

Bit widths, float16, 512x2048, 8 rows per expert: 2 bit 4.23x, 3 bit 5.94x, 4 bit 4.40x,
5 bit 6.00x, 6 bit 4.43x, 8 bit 1.37x.

### Why it matters in practice

`mlx_lm`'s `SwitchGLU` sets `do_sort = indices.size >= 64`, so the flag flips with the number of
tokens in the call. Independently, a real MoE layer only reaches 4 rows per expert once the
prompt is long enough. Measured on OLMoE-1B-7B-0125-4bit (E=64, top_k=8, float16 scales), the
layer's own relative error against float64:

| tokens | rows per expert | rel err |
|---|---|---|
| 1 to 24 | 0.12 to 3.00 | 1.50e-03 to 1.59e-03 |
| 32 and above | 4.00 and above | 5.36e-04 to 5.39e-04 |

A 3.0x step at 32 tokens. Every decode step sits on the less accurate side.

### What I could not show

On a 157 token prompt, running the whole model, the largest difference in the final logits
between prefill chunk widths was 0.06 and the greedy argmax did not move. Chunk width changes
the logits by a similar amount whether or not it crosses the threshold, so I cannot attribute
an output change to this alone. I am reporting an accuracy difference in the kernel, not a
correctness bug in generation.

### Environment

Apple M4, macOS 26.3.1, mlx 0.32.2, Python 3.12.

I checked the neighbouring reports first. #3856 and its fix #3922 are M5 NAX with M > 32768,
#3887 is M5 NAX with K % 64 != 0, and #3200 is a build environment test failure. This is an M4
and does not match any of them. If the intent is that `sorted_indices` may legitimately trade
accuracy for speed, then the one line change is to say so in the docstring, since today it reads
as a pure performance hint.
