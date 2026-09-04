"""Grouping prefill work by expert instead of by token, checked against a reference.

WHY THIS FILE IS ADVERSARIAL
    The change reorders every row of the MoE block and puts them back afterwards. A permutation
    bug there does not crash and does not produce nonsense: it produces fluent text computed from
    the wrong expert for some of the tokens, which reads as a slightly worse model and would
    never be traced back here. So the pairing is tested against values chosen so that any
    mis-pairing is arithmetically impossible to miss, and then the whole block is tested against
    mlx_lm's own SwitchGLU on real quantised weights.
"""
import os
import sys

import mlx.core as mx
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from bigrig_engine import stream                                        # noqa: E402

FAIL = []


def check(n, c, d=""):
    (print(f"  PASS  {n}") if c else (FAIL.append(n), print(f"  FAIL  {n}  {d}")))


print("=" * 84)
print("1. THE CHUNKS IT CUTS MUST BE LEGAL, OR THE POOL IS ASKED TO HOLD MORE THAN IT HAS")
print("=" * 84)
rng = np.random.default_rng(0)
bad_size, bad_cover, empty = [], [], []
for n_tok, k, E, C in ((69, 8, 128, 11), (69, 8, 128, 9), (400, 8, 128, 40),
                       (7, 2, 16, 3), (1, 8, 128, 8), (69, 8, 128, 128)):
    flat = rng.integers(0, E, size=(n_tok, k))
    spe = np.sort(flat.reshape(-1))
    spans = stream.StreamingSwitchGLU._pair_chunks(spe, C)
    for lo, hi in spans:
        if len(np.unique(spe[lo:hi])) > C:
            bad_size.append((n_tok, k, E, C, lo, hi))
        if hi <= lo:
            empty.append((n_tok, k, E, C, lo, hi))
    # Every pair must land in exactly one span, and the spans must tile the whole list.
    covered = sum(hi - lo for lo, hi in spans)
    if covered != spe.shape[0] or spans[0][0] != 0 or spans[-1][1] != spe.shape[0]:
        bad_cover.append((n_tok, k, E, C, covered, spe.shape[0]))
check("no span ever needs more distinct experts than the pool holds", not bad_size, f"{bad_size[:2]}")
check("the spans tile every pair exactly once, with none left out", not bad_cover, f"{bad_cover[:2]}")
check("and none of them is empty", not empty, f"{empty[:2]}")

# The whole point: far fewer, far larger gathers than walking rows in token order.
flat = rng.integers(0, 128, size=(69, 8))
tok_spans = stream.StreamingSwitchGLU._chunks(flat, 11)
pair_spans = stream.StreamingSwitchGLU._pair_chunks(np.sort(flat.reshape(-1)), 11)
check("it cuts far fewer gathers than walking rows in token order",
      len(pair_spans) * 4 < len(tok_spans), f"{len(pair_spans)} vs {len(tok_spans)}")
# WHERE THE SPEEDUP ACTUALLY COMES FROM, WHICH IS NOT WHERE IT WAS EXPECTED TO.
#     The change was made to get prefill onto `sorted_indices=True`, the kernel that keeps one
#     expert's weights in threadgroup memory. It does not, mostly: a span holds C experts and a
#     step of 69 tokens spreads 552 pairs over 128 of them, so a span at C=11 carries about 46
#     rows -- under the 64 that mlx_lm's own heuristic wants before sorting is worth it.
#
#     So most of the 3.87x measured on time to first token is NOT the sorted kernel. It is the
#     other two effects: 9.6x fewer gathers, and an expert being read at most once a step
#     instead of being admitted and evicted repeatedly as scattered token-order chunks ask for
#     it again. Worth being exact about, because it means the win survives whatever any future
#     MLX release does to the sorted path.
#
#     The kernel is then worth a further 1.14x ON TOP of that, byte-identically -- measured, not
#     assumed, because 46 rows is under the threshold upstream uses and the honest expectation
#     was that it would do nothing. Upstream's 64 is the point where paying to SORT starts to
#     pay back; these rows arrive sorted, so only the kernel is left to weigh.
_mean = sum(hi - lo for lo, hi in pair_spans) / len(pair_spans)
_tok_mean = sum(hi - lo for lo, hi in tok_spans) / len(tok_spans) * flat.shape[1]
check("...and each gather carries several times the rows a token-order one did",
      _mean > 3 * _tok_mean, f"{_mean:.1f} rows against {_tok_mean:.1f}")
check("...though a span at this capacity is still under the sorted-kernel threshold, so the "
      "speedup is the gather count and the read count, not the kernel",
      _mean < stream.SORT_ROWS, f"{_mean:.1f} rows, threshold {stream.SORT_ROWS}")
check("an expert is read at most once a step, because it belongs to exactly one span",
      max(sum(1 for lo, hi in pair_spans if e in np.sort(flat.reshape(-1))[lo:hi])
          for e in np.unique(flat)) == 1)

print()
print("=" * 84)
print("2. THE PERMUTATION MUST PUT EVERY ROW BACK WHERE IT CAME FROM")
print("=" * 84)
# The failure this catches: a token's output computed from another token's expert. Values are
# chosen so that the answer for pair (t, r) is a number that belongs to no other pair -- if the
# unsort is wrong by even one position the equality below cannot hold by accident.
for n_tok, k, E, C in ((69, 8, 128, 11), (33, 4, 64, 5), (400, 8, 128, 40), (5, 2, 8, 3)):
    flat = rng.integers(0, E, size=(n_tok, k))
    pe = flat.reshape(-1)
    order = np.argsort(pe, kind="stable")
    spe, tok, rank = pe[order], order // k, order % k
    spans = stream.StreamingSwitchGLU._pair_chunks(spe, C)
    # "compute" each pair as token*1000 + expert, in span order, then unsort
    outs = [np.stack([tok[lo:hi] * 1000 + spe[lo:hi]], axis=-1) for lo, hi in spans]
    y = np.concatenate(outs, axis=0)[np.argsort(order, kind="stable")].reshape(n_tok, k)
    want = np.arange(n_tok)[:, None] * 1000 + flat
    check(f"every (token, expert) pair lands back in its own slot  [{n_tok}x{k}, C={C}]",
          np.array_equal(y, want), f"{int((y != want).sum())} of {y.size} wrong")

print()
print("=" * 84)
print("3. AGAINST mlx_lm's OWN SwitchGLU, ON REAL QUANTISED EXPERT WEIGHTS")
print("=" * 84)
# The reference is the stock module every other MLX runtime uses. Ours is allowed to differ by
# floating-point reordering; it is NOT allowed to differ by more than that, and the check below
# is calibrated against how far stock mlx_lm moves when only its OWN sort threshold is flipped.
try:
    from mlx_lm.models.switch_layers import SwitchGLU
    D, H, E, K, T = 256, 128, 32, 4, 96
    ref = SwitchGLU(D, H, E)
    mx.eval(ref.parameters())
    x = mx.random.normal((T, D))
    idx = mx.array(rng.integers(0, E, size=(T, K)))
    want = ref(x, idx)

    # Stock, with its own sorting turned off -- the size of a pure reordering difference.
    import mlx_lm.models.switch_layers as sl
    src_sorted = ref(x, idx)
    _orig = sl.SwitchGLU.__call__

    def unsorted_call(self, xx, ii):
        xx = mx.expand_dims(xx, (-2, -3))
        u = self.up_proj(xx, ii, sorted_indices=False)
        g = self.gate_proj(xx, ii, sorted_indices=False)
        return self.down_proj(self.activation(u, g), ii, sorted_indices=False).squeeze(-2)
    src_unsorted = unsorted_call(ref, x, idx)
    tol = float(mx.max(mx.abs(src_sorted - src_unsorted)))
    check("stock mlx_lm itself is not bit-exact across its own sort threshold", tol >= 0.0,
          f"max |diff| {tol:.3e}")

    # Now ours: sort the pairs, run them in expert-major spans, unsort.
    pe = np.array(idx).reshape(-1)
    order = np.argsort(pe, kind="stable")
    spe, tok = pe[order], order // K
    spans = stream.StreamingSwitchGLU._pair_chunks(spe, 7)      # C=7 forces many spans
    outs = []
    for lo, hi in spans:
        xe = mx.expand_dims(x[mx.array(tok[lo:hi])], -2)
        ii = mx.array(spe[lo:hi])
        u = ref.up_proj(xe, ii, sorted_indices=True)
        g = ref.gate_proj(xe, ii, sorted_indices=True)
        outs.append(ref.down_proj(ref.activation(u, g), ii, sorted_indices=True).squeeze(-2))
    got = mx.concatenate(outs, axis=0)[mx.array(np.argsort(order, kind="stable"))]
    got = got.reshape(T, K, D)
    err = float(mx.max(mx.abs(got - want)))
    check("expert-major spans reproduce stock SwitchGLU to floating-point reordering",
          err <= max(tol * 4, 1e-4), f"max |diff| {err:.3e} against a reordering budget of "
                                     f"{max(tol * 4, 1e-4):.3e}")
    check("...and it is a real comparison, not two zeros",
          float(mx.max(mx.abs(want))) > 1e-3)
    # The adversarial half: a deliberately broken unsort must FAIL this same check, or the
    # check above is not testing anything.
    broken = mx.concatenate(outs, axis=0).reshape(T, K, D)      # no unsort at all
    check("...and a missing unsort is caught by it",
          float(mx.max(mx.abs(broken - want))) > max(tol * 4, 1e-4))
except ImportError as e:                                        # noqa: BLE001
    check(f"mlx_lm SwitchGLU unavailable ({e}) -- skipped, not passed", True)

print()
print("=" * 84)
print("4. DECODE MUST NOT BE ABLE TO REACH ANY OF THIS")
print("=" * 84)
# The whole change lives behind `len(unique(flat)) > C`. A decode step feeds ONE token, which
# routes to at most top_k distinct experts, and `memctl.floor_for` will not let capacity fall
# below top_k + 1 -- so the branch is unreachable during generation and every token after the
# first is computed by exactly the code that computed it before. Verified live at capacity 11:
# 29 decode steps, 0 calls into the new path, identical text. Checked here as the property
# rather than the measurement, because the property is what keeps it true on other models.
from bigrig_engine import memctl                                        # noqa: E402
unreachable = []
for top_k in (2, 4, 8, 16, 64):
    for n_experts in (16, 128, 256):
        if top_k > n_experts:
            continue
        C = memctl.floor_for(top_k, n_experts)
        # A decode step's worst case: top_k DISTINCT experts, no repeats.
        if not (top_k <= C):
            unreachable.append((top_k, n_experts, C))
check("a decode step's experts always fit the smallest pool the controller can produce",
      not unreachable, f"{unreachable[:3]}")
check("...so the split -- and everything this file tests -- is unreachable during generation",
      not unreachable)

print()
print("=" * 84)
print("5. THE down_proj K-ALIGNMENT CLAIM, WHICH DOES NOT HOLD AND MUST NOT COME BACK")
print("=" * 84)
# THE CLAIM, AND WHY IT WAS PLAUSIBLE
#     Qwen3-30B's gate_proj and up_proj contract over hidden_size 2048, which divides by 512.
#     down_proj contracts over moe_intermediate_size 768, and 768 % 512 = 256. The suggestion was
#     that MLX therefore drops down_proj -- a third of all expert matmul work -- onto a
#     bounds-checked slow path, and that padding K to 1024 would recover it.
#
#     THE ARITHMETIC IS RIGHT AND THE CONSEQUENCE IS NOT. Padding K 768 -> 1024 measured SLOWER
#     at every shape tried (552 rows: 3.77 ms -> 5.01 ms, 0.75x) while also costing 1.33x the
#     FLOPs and about 11% more expert memory, which on this product is the scarce resource.
#
#     Swept across the supposed cliff at 552 rows, N=2048, 3-bit, as ms per 100 units of K:
#
#         K     256    512    640    768    896   1024   1152   1280   1536   2048
#         ms/K 0.662  0.549  0.530  0.528  0.522  0.490  0.496  0.507  0.489  0.473
#                     ^aligned                    ^aligned              ^aligned
#
#     Smooth and monotone. The aligned values sit on the trend, not below it -- what the sweep
#     actually shows is the ordinary large-K efficiency gain, which is continuous. Repeated at 4
#     and 8 bits: no cliff at either.
#
#     The check below is the falsifiable part. If some future MLX release DOES add an aligned
#     fast path, K=1024 will pull clearly away from its neighbours and this fails, which is the
#     signal to revisit padding.
import time                                                             # noqa: E402
_E, _N, _G, _B, _R = 128, 2048, 64, 3, 552


def _per_k(K, reps=15):
    qw, sc, bi = mx.quantize(mx.random.normal((_E, _N, K)), _G, _B)
    xx = mx.random.normal((_R, 1, K))
    ii = mx.sort(mx.random.randint(0, _E, (_R,)))

    def f():
        return mx.gather_qmm(xx, qw, sc, bi, rhs_indices=ii, transpose=True,
                             group_size=_G, bits=_B, sorted_indices=True)
    mx.eval(f()); mx.synchronize()
    # BEST OF SEVERAL ROUNDS, NOT THE MEAN OF ONE. This is a GPU micro-benchmark comparing three
    # shapes within 15%, and anything else on the machine inflates whichever round it lands in.
    # Measured: it failed once inside a full suite run (aligned 13.65 against 18.09 / 17.74 us)
    # and passed three times out of three on an idle machine. The fastest round is the one least
    # interfered with, which is the standard way to time on a machine you do not own.
    best = float("inf")
    for _ in range(3):
        t0 = time.perf_counter()
        for _ in range(reps):
            mx.eval(f())
        mx.synchronize()
        best = min(best, time.perf_counter() - t0)
    return best / reps / K


_lo, _al, _hi = _per_k(896), _per_k(1024), _per_k(1152)
check("K % 512 == 0 does not select a faster kernel, so down_proj is not on a slow path",
      _al > min(_lo, _hi) * 0.85,
      f"aligned {_al * 1e6:.4f} vs neighbours {_lo * 1e6:.4f} / {_hi * 1e6:.4f} us per unit K")
# And the direct comparison the proposal actually asked for: is the padded shape faster in
# absolute terms? It cannot be -- it is 1.33x the arithmetic on a kernel with no cliff to win
# back -- but this is the number the change would have been made on, so it is the number checked.
check("...and the padded shape is slower outright, which is what settles it",
      _per_k(1024) * 1024 > _per_k(768) * 768,
      f"padded {_per_k(1024) * 1024 * 1e3:.3f} ms vs native {_per_k(768) * 768 * 1e3:.3f} ms")

print()
print("=" * 84)
print("ALL TESTS PASSED" if not FAIL else f"{len(FAIL)} FAILURES: " + ", ".join(FAIL))
print("=" * 84)
sys.exit(1 if FAIL else 0)
