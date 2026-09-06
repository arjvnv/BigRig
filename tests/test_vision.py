"""The vision tower and the multimodal positions, against their PyTorch reference.

What must hold. Preprocessing lands on the reference's grid and pixel values (torchvision's
bicubic and PIL's differ by at most one quantisation level on a handful of pixels). The MLX tower
reproduces transformers' Qwen3_5MoeVisionModel on the same pixels to float32 noise. The text-side
positions equal the reference's `get_rope_index` for one image and for three images across two
turns, and the interleaved rotary's cos/sin equal the reference's. Everything the reference
produced is kept as fixtures; the tower check needs the Qwen3.6 checkpoint on disk and skips
cleanly otherwise, the rest runs everywhere.
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from bigrig_engine import vision                                        # noqa: E402

FAIL = []


def check(n, c, d=""):
    (print(f"  PASS  {n}") if c else (FAIL.append(n), print(f"  FAIL  {n}  {d}")))


FIX = os.path.join(ROOT, "tests", "fixtures", "vision")
REF = json.load(open(os.path.join(FIX, "reference.json")))
MD = os.path.join(ROOT, "models", "Qwen3.6-35B-A3B-4bit")
IMG_ID = 248056

print("=" * 84); print("1. THE RESIZE RULE AND THE PATCH ORDER"); print("=" * 84)
check("a 640x400 image lands on a 24x40 patch grid (multiples of 32 pixels)",
      vision.smart_resize(400, 640, 32, 65536, 1_050_000) == (384, 640))
check("a tiny image is scaled UP to the minimum pixel count", vision.smart_resize(20, 20, 32, 65536, 1_050_000) == (256, 256))
h, w = vision.smart_resize(1080, 1920, 32, 65536, 1_050_000)
check("a 1080p screenshot is scaled DOWN under the cap, keeping its shape",
      h * w <= 1_050_000 and h % 32 == 0 and w % 32 == 0 and abs(w / h - 1920 / 1080) < 0.05, f"{h}x{w}")
try:
    vision.smart_resize(10, 5000, 32, 65536, 1_050_000); check("an absurd aspect ratio is refused", False)
except ValueError:
    check("an absurd aspect ratio is refused with a sentence", True)
rows, cols = vision.block_order_rows_cols(4, 6, 2)
check("patches are listed block by block: the first four are the top-left 2x2 block",
      list(zip(rows[:4], cols[:4])) == [(0, 0), (0, 1), (1, 0), (1, 1)] and (rows[4], cols[4]) == (0, 2))
check("...and every patch appears exactly once", sorted(zip(rows, cols)) == [(r, c) for r in range(4) for c in range(6)])

print("\n" + "=" * 84); print("2. POSITIONS EQUAL THE REFERENCE'S get_rope_index"); print("=" * 84)
for name in ("prompt1", "prompt2"):
    p = REF[name]
    pos, delta = vision.rope_positions(p["ids"], IMG_ID, [tuple(g) for g in p["grid"]], merge=2)
    check(f"{name}: every (t, h, w) position equals the reference ({len(p['ids'])} tokens, {len(p['grid'])} image(s))",
          pos.shape == (3, len(p["ids"])) and (pos == np.array(p["positions"], dtype=np.int32)).all())
    check(f"{name}: the delta later text adds to its offset equals the reference ({p['delta']})", delta == p["delta"])
p = REF["prompt1"]
check("an image's tokens take fewer position steps than tokens: 240 tokens, 12 steps",
      int(np.array(p["positions"]).max()) + 1 - len(p["ids"]) == p["delta"] == -220)
try:
    vision.rope_positions(p["ids"], IMG_ID, [(1, 24, 40), (1, 2, 2)], merge=2); check("extra images are refused", False)
except ValueError:
    check("more images than placeholder runs is refused", True)
try:
    vision.rope_positions(p["ids"][:-1] + [IMG_ID], IMG_ID, [(1, 24, 40)], merge=2); check("a torn run is refused", False)
except ValueError:
    check("a placeholder run that does not match its grid is refused", True)
text_only = list(range(1000, 1020))
pos_t, delta_t = vision.rope_positions(text_only, IMG_ID, [], merge=2)
check("text-only positions are the plain 0..L-1 on all three streams, delta 0",
      (pos_t == np.arange(20)[None, :]).all() and delta_t == 0)

print("\n" + "=" * 84); print("3. THE INTERLEAVED ROTARY EQUALS THE REFERENCE'S"); print("=" * 84)
import mlx.core as mx                                                   # noqa: E402
rope = vision.MRope(dims=64, base=10_000_000.0, section=[11, 11, 10])
check("frequency streams are laid out T,H,W interleaved with the tail on T (11 + 11 + 10)",
      list(rope.which[:6]) == [0, 1, 2, 0, 1, 2] and int((rope.which == 1).sum()) == 11
      and int((rope.which == 2).sum()) == 10 and int((rope.which == 0).sum()) == 11)
cos, sin = rope.cos_sin(np.array(REF["prompt1"]["positions"], dtype=np.int32))
rc, rs = np.load(os.path.join(FIX, "mrope_cos.npy")), np.load(os.path.join(FIX, "mrope_sin.npy"))
check("cos and sin equal the reference on the one-image prompt (257 x 64)",
      cos.shape == rc.shape and np.abs(np.array(cos) - rc).max() < 1e-5 and np.abs(np.array(sin) - rs).max() < 1e-5,
      f"{np.abs(np.array(cos) - rc).max():.2e} / {np.abs(np.array(sin) - rs).max():.2e}")
# With equal streams the rotation is the plain rotary mlx uses for text: check against mx.fast.rope.
L = 9
x = mx.random.normal((1, 2, L, 256)).astype(mx.float32)
plain = mx.fast.rope(x, 64, traditional=False, base=10_000_000.0, scale=1.0, offset=5)
c2, s2 = rope.cos_sin(np.tile(np.arange(5, 5 + L, dtype=np.int32)[None, :], (3, 1)))
ours = rope.apply(x, c2, s2)
check("with all three streams equal it IS the text engine's plain rotary (offset included)",
      float(mx.max(mx.abs(ours - plain))) < 1e-4, f"{float(mx.max(mx.abs(ours - plain))):.2e}")
check("...and only the first 64 of 256 head dims are rotated", bool(mx.array_equal(ours[..., 64:], x[..., 64:])))

print("\n" + "=" * 84); print("4. THE TOWER AGAINST ITS PYTORCH REFERENCE"); print("=" * 84)
if not os.path.isdir(MD):
    print("  SKIPPED - Qwen3.6-35B-A3B-4bit is not on disk")
else:
    pre = vision.Preprocessor(MD)
    pv, grid = pre(open(os.path.join(FIX, "screenshot.png"), "rb").read())
    check("the screenshot lands on the reference's grid", list(grid) == REF["screenshot_grid"], str(grid))
    check("...as 960 patches of 3x2x16x16", pv.shape == (960, 1536) and pv.dtype == np.float32)
    check("the token count is predictable from the image size alone", pre.tokens_for(400, 640) == 240)
    tower = vision.load_tower(MD)
    check("the tower loads its 0.89 GB in the checkpoint's bfloat16", 0.88 < tower.nbytes / 1e9 < 0.90
          and tower.patch_embed.weight.dtype == mx.bfloat16, f"{tower.nbytes / 1e9:.3f}")
    feats = np.array(tower(mx.array(pv), [grid]))
    ref = np.load(os.path.join(FIX, "screenshot_features_f16.npy")).astype(np.float32)
    cos_ = (feats * ref).sum(1) / (np.linalg.norm(feats, axis=1) * np.linalg.norm(ref, axis=1))
    check("240 image tokens of 2048, in float32", feats.shape == (240, 2048) and feats.dtype == np.float32)
    check("every token matches the reference (cosine >= 0.9999; float16 storage of the reference is the limit)",
          bool((cos_ >= 0.9999).all()), f"min cos {cos_.min():.6f}")
    check("...and no token is off by more than the float16 rounding of the reference",
          np.abs(feats - ref).max() < 0.02, f"max |d| {np.abs(feats - ref).max():.3e}")
    # The bf16-residual failure this guards against read 0.78 cosine on a noise image; a wrong
    # patch order or a wrong rotary reads far below 0.99 on a structured one.
    mx.reset_peak_memory()
    base = mx.get_active_memory()
    feats2 = tower(mx.array(pv), [grid]); mx.eval(feats2)
    check("encoding a 640x400 image needs under 0.5 GB above the resident tower",
          (mx.get_peak_memory() - base) / 1e9 < 0.5, f"{(mx.get_peak_memory() - base) / 1e9:.2f} GB")

print()
print("=" * 84)
print("ALL TESTS PASSED" if not FAIL else f"{len(FAIL)} FAILURES: " + ", ".join(FAIL))
print("=" * 84)
sys.exit(1 if FAIL else 0)
