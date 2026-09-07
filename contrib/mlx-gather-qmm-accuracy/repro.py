"""gather_qmm: sorted_indices changes the ANSWER, not just the speed.

`sorted_indices` is documented as a performance hint ("May allow a faster implementation if the
passed indices are sorted"). On identical, already sorted indices the two settings return
different values, and the sorted_indices=False result is about 4.4x further from an exact
float64 reference.

Run:  python repro.py
Needs: mlx, numpy. No model, no download. A few seconds.

The reference is computed in float64 from mx.dequantize of the SAME quantised weights the
kernel reads, so quantisation error cancels and only the kernel's own arithmetic is scored.
A third, independent path (per expert mx.quantized_matmul) is included to show which of the
two gather results is the correct one.
"""
import numpy as np
import mlx.core as mx

GS = 64


def rms(a):
    return float(np.sqrt(np.mean(a ** 2)))


def make(E, rows_per_expert, N, K, dtype, bits, seed):
    """R rows, sorted ascending by expert, `rows_per_expert` rows for each of E experts."""
    R = E * rows_per_expert
    mx.random.seed(seed)
    wq, sc, bi = mx.quantize(mx.random.normal((E, N, K)).astype(dtype), group_size=GS, bits=bits)
    x = mx.random.normal((R, 1, K)).astype(dtype)
    idx = mx.array(np.repeat(np.arange(E), rows_per_expert).astype(np.uint32))
    mx.eval(wq, sc, bi, x, idx)
    return (wq, sc, bi), x, idx


def gather(W, x, idx, bits, sorted_indices):
    wq, sc, bi = W
    y = mx.gather_qmm(x, wq, sc, bi, rhs_indices=idx, transpose=True,
                      group_size=GS, bits=bits, sorted_indices=sorted_indices)
    mx.eval(y)
    return np.array(y.astype(mx.float32)).reshape(x.shape[0], -1).astype(np.float64)


def reference(W, x, idx, bits):
    wq, sc, bi = W
    dw = np.array(mx.dequantize(wq, sc, bi, group_size=GS, bits=bits)
                  .astype(mx.float32)).astype(np.float64)
    xf = np.array(x.astype(mx.float32)).reshape(x.shape[0], -1).astype(np.float64)
    return np.einsum("rk,rnk->rn", xf, dw[np.array(idx)])


def per_expert_qmm(W, x, idx, bits, E, rows_per_expert, K):
    wq, sc, bi = W
    out = []
    for e in range(E):
        xe = x[e * rows_per_expert:(e + 1) * rows_per_expert].reshape(rows_per_expert, K)
        y = mx.quantized_matmul(xe, wq[e], sc[e], bi[e], transpose=True, group_size=GS, bits=bits)
        mx.eval(y)
        out.append(np.array(y.astype(mx.float32)).astype(np.float64))
    return np.concatenate(out, axis=0)


def relerr(got, ref):
    return rms(got - ref) / rms(ref)


FAIL = []
print("=" * 96)
print(f"mlx {mx.__version__}   gather_qmm: sorted_indices=True vs False on the same sorted input")
print("=" * 96)

print("\n1. THE EFFECT, ACROSS SHAPES AND DTYPES  (E=32, 8 rows per expert, 4 bit affine)")
print(f"   {'dtype':>9s} {'N':>6s} {'K':>6s} {'rel err sorted':>15s} {'rel err unsorted':>17s} "
      f"{'ratio':>7s} {'max|T-F|':>10s}")
for dtype, dn in ((mx.float16, "float16"), (mx.bfloat16, "bfloat16"), (mx.float32, "float32")):
    for N, K in ((512, 2048), (2048, 512), (1024, 1024)):
        W, x, idx = make(32, 8, N, K, dtype, 4, seed=0)
        ref = reference(W, x, idx, 4)
        t, f = gather(W, x, idx, 4, True), gather(W, x, idx, 4, False)
        et, ef = relerr(t, ref), relerr(f, ref)
        print(f"   {dn:>9s} {N:>6d} {K:>6d} {et:15.3e} {ef:17.3e} {ef / et:6.2f}x "
              f"{np.abs(t - f).max():10.5f}")
        if dtype is not mx.float32 and ef / et < 3:
            FAIL.append(f"{dn} {N}x{K} ratio {ef / et:.2f}")

print("\n2. WHICH ONE IS RIGHT?  A third path, per expert mx.quantized_matmul, float16 512x2048")
W, x, idx = make(32, 8, 512, 2048, mx.float16, 4, seed=0)
ref = reference(W, x, idx, 4)
q = per_expert_qmm(W, x, idx, 4, 32, 8, 2048)
t, f = gather(W, x, idx, 4, True), gather(W, x, idx, 4, False)
for name, arr in (("quantized_matmul (per expert)", q), ("gather sorted_indices=True", t),
                  ("gather sorted_indices=False", f)):
    print(f"   {name:32s} rel err {relerr(arr, ref):.3e}")
print(f"   quantized_matmul agrees with sorted=True to {np.abs(q - t).max():.5f} "
      f"and with sorted=False to {np.abs(q - f).max():.5f}")
if not relerr(q, ref) < relerr(f, ref) / 2:
    FAIL.append("quantized_matmul did not side with sorted=True")

print("\n3. WHEN DOES IT APPEAR?  float16, N=512, K=2048, 4 bit")
print(f"   {'experts':>8s} {'rows/expert':>12s} {'total rows':>11s} {'ratio unsorted/sorted':>22s}")
for E, rpe in ((32, 1), (32, 2), (32, 3), (32, 4), (32, 8),
               (1, 4), (2, 4), (4, 4), (1, 16), (2, 8)):
    W, x, idx = make(E, rpe, 512, 2048, mx.float16, 4, seed=0)
    ref = reference(W, x, idx, 4)
    r = relerr(gather(W, x, idx, 4, False), ref) / relerr(gather(W, x, idx, 4, True), ref)
    print(f"   {E:>8d} {rpe:>12d} {E * rpe:>11d} {r:21.2f}x")

print("\n4. BIT WIDTHS  (E=32, 8 rows per expert, float16, 512x2048)")
for bits in (2, 3, 4, 5, 6, 8):
    W, x, idx = make(32, 8, 512, 2048, mx.float16, bits, seed=0)
    ref = reference(W, x, idx, bits)
    et = relerr(gather(W, x, idx, bits, True), ref)
    ef = relerr(gather(W, x, idx, bits, False), ref)
    print(f"   {bits} bit: sorted {et:.3e}   unsorted {ef:.3e}   ratio {ef / et:.2f}x")

print("\n5. STABILITY  (same config, five seeds, float16 512x2048)")
ratios = []
for s in range(5):
    W, x, idx = make(32, 8, 512, 2048, mx.float16, 4, seed=s)
    ref = reference(W, x, idx, 4)
    ratios.append(relerr(gather(W, x, idx, 4, False), ref) / relerr(gather(W, x, idx, 4, True), ref))
print("   ratios: " + "  ".join(f"{r:.2f}x" for r in ratios))
if min(ratios) < 3:
    FAIL.append(f"unstable across seeds: {ratios}")

print("\n" + "=" * 96)
if FAIL:
    print("NOT REPRODUCED on this machine: " + "; ".join(FAIL))
else:
    print("REPRODUCED: sorted_indices=False is consistently about 4x further from the exact")
    print("            answer than sorted_indices=True on identical, already sorted input.")
print("=" * 96)
raise SystemExit(1 if FAIL else 0)
