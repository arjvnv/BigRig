"""What the gather_qmm accuracy step means for a real MoE layer in mlx_lm.

mlx_lm's SwitchGLU sets `do_sort = indices.size >= 64`, and separately the accurate kernel
needs about 4 rows per expert, which a real MoE layer only reaches on a longer prompt. So the
same layer, same weights, computes its experts more accurately for a long prompt than for a
short one or for a decode step.

Run:  python mlx_lm_effect.py [path-to-an-mlx-moe-model]
Skips cleanly if the model is not present. No download.
"""
import os
import sys

import numpy as np
import mlx.core as mx

DEFAULT = os.path.expanduser("~/bigrig/models/OLMoE-1B-7B-0125-4bit")
path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT

if not os.path.isdir(path):
    print(f"SKIPPED: no model at {path}")
    print("         pass the path to any local MLX MoE checkpoint to run this.")
    raise SystemExit(0)

from mlx_lm import load                                                    # noqa: E402

model, _ = load(path)
layer = None
for lyr in model.model.layers:
    mlp = getattr(lyr, "mlp", None) or getattr(lyr, "mixer", None)
    sm = getattr(mlp, "switch_mlp", None)
    if sm is not None and hasattr(sm, "gate_proj"):
        layer = sm
        break
if layer is None:
    print(f"SKIPPED: {os.path.basename(path)} has no gated switch_mlp layer to measure")
    raise SystemExit(0)

E = layer.gate_proj.weight.shape[0]
K = layer.gate_proj.scales.shape[2] * layer.gate_proj.group_size
TOPK = 8
DTYPE = layer.gate_proj.scales.dtype     # match the checkpoint, or dtype promotion hides the effect

DW = {}
for p in ("gate_proj", "up_proj", "down_proj"):
    lin = getattr(layer, p)
    DW[p] = np.array(mx.dequantize(lin.weight, lin.scales, lin.biases,
                                   group_size=lin.group_size,
                                   bits=lin.bits).astype(mx.float32)).astype(np.float64)


def ref64(xf, ii):
    out = np.zeros((ii.shape[0], ii.shape[1], xf.shape[-1]))
    for t in range(ii.shape[0]):
        xt = xf[t]
        for j, e in enumerate(ii[t]):
            g = DW["gate_proj"][e] @ xt
            u = DW["up_proj"][e] @ xt
            out[t, j] = DW["down_proj"][e] @ (u * (g / (1.0 + np.exp(-g))))
    return out


def rms(a):
    return float(np.sqrt(np.mean(a ** 2)))


mx.random.seed(11)
rng = np.random.default_rng(11)
T_MAX = 256
x_all = mx.random.normal((T_MAX, K)).astype(DTYPE)
ind_all = mx.array(rng.integers(0, E, size=(T_MAX, TOPK)).astype(np.uint32))
mx.eval(x_all, ind_all)
xf_all = np.array(x_all.astype(mx.float32)).astype(np.float64)
ii_all = np.array(ind_all)

print(f"{os.path.basename(path)}: E={E} K={K} top_k={TOPK} dtype={DTYPE}, mlx {mx.__version__}")
print(f"\n  {'tokens':>7s} {'idx.size':>9s} {'do_sort':>8s} {'rows/expert':>12s} "
      f"{'rel RMS err vs float64':>23s}")
prev, step = None, None
for T in (1, 2, 4, 8, 16, 24, 32, 48, 64, 128, 256):
    y = layer(x_all[:T], ind_all[:T])
    mx.eval(y)
    got = np.array(y.astype(mx.float32)).astype(np.float64)
    e = rms(got - ref64(xf_all[:T], ii_all[:T])) / rms(ref64(xf_all[:T], ii_all[:T]))
    mark = ""
    if prev is not None and prev / e > 2:
        mark, step = f"   <-- {prev / e:.1f}x more accurate from here", (T, prev / e)
    prev = e
    print(f"  {T:>7d} {T * TOPK:>9d} {str(T * TOPK >= 64):>8s} {T * TOPK / E:>12.2f} {e:23.3e}{mark}")

print()
if step:
    print(f"MEASURED: accuracy steps {step[1]:.1f}x at {step[0]} tokens "
          f"({step[0] * TOPK / E:.0f} rows per expert). Decode always sits on the worse side.")
else:
    print("NO STEP on this model at these prompt lengths.")
