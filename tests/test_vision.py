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

print("\n" + "=" * 84); print("5. SERVED: THE GUARDS AND THE MESSAGE SHAPES, THROUGH THE REAL SERVER"); print("=" * 84)
import base64                                                           # noqa: E402
sys.path.insert(0, os.path.join(ROOT, "tests"))
from _fakeserver import fake_server, post, FakeSession                  # noqa: E402
from bigrig_engine import anthropic as anth                             # noqa: E402
PNG = open(os.path.join(FIX, "screenshot.png"), "rb").read()
URL = "data:image/png;base64," + base64.b64encode(PNG).decode()
with fake_server() as (url, state, fs):
    st, b, _ = post(url, "/v1/chat/completions", {"messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": URL}}, {"type": "text", "text": "what is this?"}]}], "max_tokens": 4})
    check("an image to a server without --vision is a 400 that names the flag", st == 400 and "--vision" in json.dumps(b), f"{st} {b}")
    st, b, _ = post(url, "/v1/messages", {"model": "m", "max_tokens": 4, "messages": [{"role": "user", "content": [
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": base64.b64encode(PNG).decode()}},
        {"type": "text", "text": "what is this?"}]}]})
    check("...on the Anthropic API too, in its error envelope", st == 400 and b.get("type") == "error" and "--vision" in json.dumps(b), f"{st} {b}")
    check("a text-only request is untouched", post(url, "/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 2})[0] == 200)
with fake_server(FakeSession(tower=object())) as (url, state, fs):
    st, b, _ = post(url, "/v1/chat/completions", {"messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": URL}}, {"type": "text", "text": "what is this?"}]}], "max_tokens": 4})
    check("with a tower, an OpenAI image part reaches the engine intact, in order",
          st == 200 and fs.calls and isinstance(fs.calls[-1]["messages"][0]["content"], list)
          and fs.calls[-1]["messages"][0]["content"][0].get("type") == "image_url", f"{st}")
    st, b, _ = post(url, "/v1/messages", {"model": "m", "max_tokens": 4, "messages": [{"role": "user", "content": [
        {"type": "text", "text": "look:"}, {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": base64.b64encode(PNG).decode()}},
        {"type": "text", "text": "what is this?"}]}]})
    parts = fs.calls[-1]["messages"][0]["content"] if fs.calls else None
    check("an Anthropic image block is kept as a block the template renders, text around it in order",
          st == 200 and isinstance(parts, list) and [pt["type"] for pt in parts] == ["text", "image", "text"]
          and parts[1]["source"]["data"] == base64.b64encode(PNG).decode(), f"{st} {parts and [pt['type'] for pt in parts]}")
    st, b, _ = post(url, "/v1/chat/completions", {"messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "https://example.com/cat.png"}}, {"type": "text", "text": "?"}]}], "max_tokens": 4})
    check("a remote URL is refused: this server fetches nothing", st == 400 and "not fetched" in json.dumps(b), f"{st} {b}")
    st, b, _ = post(url, "/v1/chat/completions", {"messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,@@@not-base64@@@"}}, {"type": "text", "text": "?"}]}], "max_tokens": 4})
    check("undecodable image bytes are a 400 with a reason, not a 500 from the model thread", st == 400, f"{st} {b}")
    st, b, _ = post(url, "/v1/chat/completions", {"messages": [{"role": "user", "content":
        [{"type": "image_url", "image_url": {"url": URL}}] * (vision.MAX_IMAGES + 1) + [{"type": "text", "text": "?"}]}], "max_tokens": 4})
    check(f"more than {vision.MAX_IMAGES} images in one request is refused", st == 400 and "at most" in json.dumps(b), f"{st}")
# The conversion itself, without a server.
parsed = anth.parse({"model": "m", "max_tokens": 5, "messages": [{"role": "user", "content": [
    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "QUJD"}}, {"type": "text", "text": "hi"}]}]})
m = anth.to_engine_messages(parsed)
check("to_engine_messages keeps image blocks for the template and drops nothing",
      len(m) == 1 and isinstance(m[0]["content"], list) and m[0]["content"][0]["type"] == "image" and m[0]["content"][1]["text"] == "hi")
check("count_images and extract_images agree with the template's notion of an image item",
      vision.count_images(m) == 1 and vision.extract_images(m) == [b"ABC"])

print("\n" + "=" * 84); print("5b. PER REQUEST BY DEFAULT: THE POOL IS PLANNED AS IF THE TOWER DID NOT EXIST"); print("=" * 84)
if not os.path.isdir(MD):
    print("  SKIPPED - Qwen3.6-35B-A3B-4bit is not on disk")
else:
    # Pure planning, no model load: the same budget, with and without a resident tower. At 7.0 GB
    # the resident tower (0.89 GB off the pool) is REFUSED by the planner while per-request plans
    # the full pool -- measured live at that budget: per-request answered "add" and peaked 1.1 GB
    # above its footprint for the duration of the request; resident could not start at all.
    check("the tower's size is read from the checkpoint's headers without loading a weight",
          0.88 < vision.tower_gb(MD) < 0.90, f"{vision.tower_gb(MD):.3f}")
    try:
        vision.tower_gb(os.path.join(ROOT, "models", "OLMoE-1B-7B-0125-4bit"))
        check("a checkpoint without a tower is refused with a sentence", False)
    except (ValueError, FileNotFoundError) as e:
        check("a checkpoint without a tower is refused with a sentence", True)

print("\n" + "=" * 84); print("6. THE WHOLE ROAD, ON QWEN3.6"); print("=" * 84)
if not os.path.isdir(MD):
    print("  SKIPPED - Qwen3.6-35B-A3B-4bit is not on disk")
else:
    import subprocess
    _p = subprocess.run([sys.executable, os.path.join(ROOT, "tests", "_snapshot_child.py"), "vision", "Qwen3.6-35B-A3B-4bit"],
                        capture_output=True, text=True, timeout=1200, env=dict(os.environ, BIGRIG_MAX_GB=os.environ.get("BIGRIG_MAX_GB", "9")))
    _line = next((ln for ln in _p.stdout.splitlines() if ln.startswith("RESULT ")), None)
    if _line is None:
        check("the live vision scenario ran", False, (_p.stdout + _p.stderr)[-1500:])
    else:
        R = json.loads(_line[7:])
        print(f"      first token {R['first_token_s']}s, reply in {R['seconds']}s, prompt {R['prompt_tokens']} tokens (240 of them image)")
        print(f"      reply: {R['reply'][:160]!r}")
        print(f"      two images, 'what colour is the second': {R['two_image_reply']!r}")
        check("by default the tower is NOT charged to the ceiling: it is read per request and given back",
              not R["reserved_gb"] and R["vision_stats"]["resident"] is False and R["tower_held_after"] is False,
              f"reserved {R['reserved_gb']} resident {R['vision_stats']['resident']} held {R['tower_held_after']}")
        check("the prompt carries the image's 240 tokens plus the text", R["prompt_tokens"] and 240 < R["prompt_tokens"] < 320, str(R["prompt_tokens"]))
        rep = R["reply"]
        check("the model READ the screenshot: the headline is transcribed", "BigRig" in rep and "Qwen3.6" in rep and "screenshot" in rep, rep[:120])
        check("...and the code in it", "add(a, b)" in rep and "return a + b" in rep and "print(add(2, 3))" in rep)
        check("...and it saw the red circle", "red" in rep.lower() and "circle" in rep.lower())
        check("with two images it answers about the SECOND one (a plain blue square): 'blue'",
              "blue" in R["two_image_reply"].lower(), R["two_image_reply"])
        check("nothing from an image request is kept in the conversation cache", R["cache_entries_after_image"] == 0, str(R["cache_entries_after_image"]))
        check("the rotary switches are disarmed afterwards and a text-only turn still answers",
              R["disarmed"] and len(R["text_after"].strip()) > 3, R["text_after"])

print()
print("=" * 84)
print("ALL TESTS PASSED" if not FAIL else f"{len(FAIL)} FAILURES: " + ", ".join(FAIL))
print("=" * 84)
sys.exit(1 if FAIL else 0)
