"""Seeing: the vision tower Qwen3.5/3.6 ships, in MLX, and the positions its text side needs.

WHY. Qwen3.6-35B-A3B's checkpoint carries a 0.89 GB vision encoder (333 tensors, bf16) that the
text engine discards at load. Loading it turns "a model your Mac cannot hold" into one that reads
a screenshot, on the same hardware. The maintained way to do this is mlx-vlm, whose dependency
list (OpenCV, an audio stack, a web framework) is far more than a 300-line encoder; so, as with
embeddings, the tower is written here against the reference implementation and checked against
it numerically (tests/test_vision.py).

WHAT IS HERE. Three pieces, each a straight port of transformers' Qwen3_5Moe code:
  * preprocessing -- smart_resize to a multiple of 32 within the pixel budget, bicubic resize,
    normalise, patchify in spatial-merge-block order (2x2 blocks of 16px patches, each patch
    repeated on a temporal axis of 2), giving `pixel_values` (N, 1536) and `grid_thw`;
  * the tower -- patch embedding, a learned 48x48 position table resampled bilinearly to the
    image grid, 27 blocks of 2-D rotary attention within each image, and the merger that turns
    every 2x2 block into one 2048-wide token for the language model;
  * positions -- the text model's interleaved multimodal rotary embedding: each image token
    carries (t, h, w) positions, and every text token after an image is shifted by a delta the
    reference calls `rope_deltas`. mlx_lm's text model applies plain 1-D rotary, exact for text
    and wrong for images, so the engine swaps in `mrope` for the multimodal prefill and shifts the
    offset afterwards (session.py).

WHAT IS NOT. Video. Remote image URLs (the server makes no external requests; send base64).
"""
from __future__ import annotations

import base64
import io
import json
import math
import os

import numpy as np

import mlx.core as mx
import mlx.nn as nn

VISION_START, VISION_END, IMAGE_PAD = "<|vision_start|>", "<|vision_end|>", "<|image_pad|>"
MAX_IMAGE_BYTES = 30_000_000       # a decoded image far past any screenshot; a guard, not a limit
MAX_IMAGES = 8                     # per request


# ------------------------------------------------------------------------------ preprocessing
def smart_resize(height: int, width: int, factor: int, min_pixels: int, max_pixels: int) -> tuple:
    """The reference's rule: both sides a multiple of `factor`, total pixels inside the budget,
    aspect ratio kept as closely as possible."""
    if max(height, width) / min(height, width) > 200:
        raise ValueError("the image's aspect ratio is over 200:1; it cannot be tiled")
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


class Preprocessor:
    """Image bytes -> (pixel_values (N, C*T*P*P) float32, grid_thw (1, h, w)), the reference way."""

    def __init__(self, model_dir: str, max_pixels: int | None = None):
        with open(os.path.join(model_dir, "preprocessor_config.json")) as f:
            c = json.load(f)
        self.patch = int(c.get("patch_size", 16))
        self.merge = int(c.get("merge_size", 2))
        self.temporal = int(c.get("temporal_patch_size", 2))
        size = c.get("size") or {}
        self.min_pixels = int(size.get("shortest_edge") or c.get("min_pixels") or 56 * 56)
        self.max_pixels = int(size.get("longest_edge") or c.get("max_pixels") or 14 * 14 * 4 * 1280)
        # A CEILING OF OUR OWN ON TOP OF THE MODEL'S. The checkpoint allows 16.7 megapixels, which
        # is 65,536 image tokens -- more than most Macs' context ceiling and a 27-block attention
        # over 65k patches besides. 1.05 MP is 1024 tokens (a 1280x820 screenshot at full detail);
        # `--vision-pixels` raises it for a machine with the memory.
        self.max_pixels = min(self.max_pixels, int(max_pixels or 1_050_000))
        self.mean = np.array(c.get("image_mean", [0.5, 0.5, 0.5]), dtype=np.float32)
        self.std = np.array(c.get("image_std", [0.5, 0.5, 0.5]), dtype=np.float32)
        self.resample = int(c.get("resample", 3))          # 3 = bicubic

    def tokens_for(self, height: int, width: int) -> int:
        h, w = smart_resize(height, width, self.patch * self.merge, self.min_pixels, self.max_pixels)
        return (h // self.patch) * (w // self.patch) // (self.merge ** 2)

    def __call__(self, data: bytes) -> tuple:
        from PIL import Image
        if len(data) > MAX_IMAGE_BYTES:
            raise ValueError(f"the image is {len(data) / 1e6:.0f} MB; at most {MAX_IMAGE_BYTES / 1e6:.0f} MB")
        img = Image.open(io.BytesIO(data))
        img.load()
        img = img.convert("RGB")
        w0, h0 = img.size
        h, w = smart_resize(h0, w0, self.patch * self.merge, self.min_pixels, self.max_pixels)
        resample = {0: Image.NEAREST, 1: Image.LANCZOS, 2: Image.BILINEAR, 3: Image.BICUBIC}.get(self.resample, Image.BICUBIC)
        img = img.resize((w, h), resample=resample)
        x = np.asarray(img, dtype=np.float32) / 255.0                        # (h, w, 3)
        x = (x - self.mean) / self.std
        x = x.transpose(2, 0, 1)                                                # (3, h, w), the reference's CHW
        gh, gw = h // self.patch, w // self.patch
        m, p = self.merge, self.patch
        # The reference: reshape (C, gh/m, m, p, gw/m, m, p) -> permute to
        # (gh/m, gw/m, m, m, C, p, p) -> repeat along a temporal axis -> flatten to (gh*gw, C*T*p*p).
        x = x.reshape(3, gh // m, m, p, gw // m, m, p).transpose(1, 4, 2, 5, 0, 3, 6)
        x = np.repeat(x[:, :, :, :, :, None, :, :], self.temporal, axis=5)       # (.., C, T, p, p)
        pixel_values = np.ascontiguousarray(x.reshape(gh * gw, 3 * self.temporal * p * p))
        return pixel_values, (1, gh, gw)


def decode_image_ref(ref: str) -> bytes:
    """Bytes of an image given as a data URL or bare base64. A remote URL is refused: the server
    makes no requests of its own, and a client can fetch and inline it."""
    s = ref.strip()
    if s.startswith("data:"):
        head, _, payload = s.partition(",")
        if ";base64" not in head:
            raise ValueError("only base64 data URLs are supported (data:image/png;base64,...)")
        s = payload
    elif s.startswith(("http://", "https://", "file://")):
        raise ValueError("remote image URLs are not fetched by this server; send the image as a "
                         "base64 data URL (data:image/png;base64,...)")
    try:
        return base64.b64decode(s, validate=False)
    except (ValueError, TypeError) as e:
        raise ValueError(f"the image is not valid base64: {e}")


# ------------------------------------------------------------------------------ the tower
def _interp(index: np.ndarray, size: np.ndarray, side: int) -> tuple:
    """Bilinear, align_corners=True, border padding: (taps (N,2), weights (N,2)) -- the
    reference's `_interpolation_axis_taps_weights` for the mode the tower uses."""
    index = index.astype(np.float32)                        # float32, as the reference computes it
    src = index * np.float32(side - 1) / np.maximum(size - 1, 1).astype(np.float32)
    floor = np.floor(src)
    offsets = np.arange(2)
    raw = floor[:, None].astype(np.int64) + offsets
    taps = np.clip(raw, 0, side - 1)
    dist = np.abs(src[:, None] - floor[:, None] - offsets)
    weights = np.clip(1.0 - dist, 0.0, None)
    return taps, weights.astype(np.float32)


def block_order_rows_cols(gh: int, gw: int, m: int) -> tuple:
    """(row, col) of every patch in the order pixel_values lists them: 2x2 merge blocks, block by
    block, raster order within a block."""
    within = np.arange(gh * gw)
    blocks_w = gw // m
    in_col = within % m
    in_row = (within // m) % m
    block_col = (within // (m * m)) % blocks_w
    block_row = within // (m * m * blocks_w)
    return block_row * m + in_row, block_col * m + in_col


class _VBlock(nn.Module):
    def __init__(self, dim: int, heads: int, inter: int):
        super().__init__()
        self.heads = heads
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.fc1 = nn.Linear(dim, inter)
        self.fc2 = nn.Linear(inter, dim)

    def __call__(self, x, cos, sin, segments):
        N, D = x.shape
        h, d = self.heads, D // self.heads
        qkv = self.qkv(self.norm1(x)).reshape(N, 3, h, d)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]                              # (N, h, d)
        # Rotary in float32, as the reference does, then back to the working dtype.
        q32, k32 = q.astype(mx.float32), k.astype(mx.float32)
        c, s = cos[:, None, :], sin[:, None, :]
        q32 = q32 * c + _rotate_half(q32) * s
        k32 = k32 * c + _rotate_half(k32) * s
        q, k = q32.astype(x.dtype), k32.astype(x.dtype)
        outs = []
        for a, b in segments:                                                   # one image = one segment
            qs = q[a:b].transpose(1, 0, 2)[None]                                # (1, h, n, d)
            ks = k[a:b].transpose(1, 0, 2)[None]
            vs = v[a:b].transpose(1, 0, 2)[None]
            o = mx.fast.scaled_dot_product_attention(qs, ks, vs, scale=1.0 / math.sqrt(d))
            outs.append(o[0].transpose(1, 0, 2).reshape(b - a, D))
        x = x + self.proj(mx.concatenate(outs, axis=0) if len(outs) > 1 else outs[0])
        return x + self.fc2(nn.gelu_approx(self.fc1(self.norm2(x))))


def _rotate_half(x):
    half = x.shape[-1] // 2
    return mx.concatenate([-x[..., half:], x[..., :half]], axis=-1)


class VisionTower(nn.Module):
    """Qwen3.5/3.6's vision encoder: pixel patches in, one 2048-wide token per 2x2 block out."""

    def __init__(self, vcfg: dict):
        super().__init__()
        self.dim = int(vcfg["hidden_size"])
        self.heads = int(vcfg["num_heads"])
        self.patch = int(vcfg["patch_size"])
        self.temporal = int(vcfg.get("temporal_patch_size", 2))
        self.merge = int(vcfg.get("spatial_merge_size", 2))
        self.in_channels = int(vcfg.get("in_channels", 3))
        self.side = int(math.isqrt(int(vcfg["num_position_embeddings"])))
        self.out_dim = int(vcfg["out_hidden_size"])
        self.patch_embed = nn.Linear(self.in_channels * self.temporal * self.patch * self.patch, self.dim)
        self.pos_embed = nn.Embedding(self.side * self.side, self.dim)
        self.blocks = [_VBlock(self.dim, self.heads, int(vcfg["intermediate_size"])) for _ in range(int(vcfg["depth"]))]
        merged = self.dim * self.merge * self.merge
        self.merger_norm = nn.LayerNorm(self.dim, eps=1e-6)
        self.merger_fc1 = nn.Linear(merged, merged)
        self.merger_fc2 = nn.Linear(merged, self.out_dim)
        head_dim = self.dim // self.heads
        rot = head_dim // 2
        self.inv_freq = 1.0 / (10000.0 ** (np.arange(0, rot, 2, dtype=np.float64) / rot))     # (rot/2,)

    def _positions(self, grids: list) -> tuple:
        """For the concatenated images: pos-embed taps/weights, rotary (cos, sin) and segments."""
        taps_all, w_all, rot_all, segs, start = [], [], [], [], 0
        for (t, gh, gw) in grids:
            rows, cols = block_order_rows_cols(gh, gw, self.merge)
            ht, hw = _interp(rows, np.full_like(rows, gh), self.side)
            wt, ww = _interp(cols, np.full_like(cols, gw), self.side)
            idx = (ht[:, :, None] * self.side + wt[:, None, :]).reshape(-1, 4)
            wgt = (hw[:, :, None] * ww[:, None, :]).reshape(-1, 4)
            pos = np.stack([rows, cols], axis=-1).astype(np.float64)              # (n, 2)
            freqs = (pos[:, :, None] * self.inv_freq[None, None, :]).reshape(len(rows), -1)   # (n, 2*rot/2)
            for _ in range(t):
                taps_all.append(idx); w_all.append(wgt); rot_all.append(freqs)
                segs.append((start, start + gh * gw)); start += gh * gw
        taps = mx.array(np.concatenate(taps_all)); wgt = mx.array(np.concatenate(w_all).astype(np.float32))
        emb = np.concatenate(rot_all); emb = np.concatenate([emb, emb], axis=-1)   # (N, head_dim)
        return taps, wgt, mx.array(np.cos(emb).astype(np.float32)), mx.array(np.sin(emb).astype(np.float32)), segs

    def __call__(self, pixel_values, grids: list):
        """Image tokens (N/4, out_dim) in float32. The caller casts to the language model's dtype.

        ACTIVATIONS IN FLOAT32, WEIGHTS AS STORED. Measured on the noise test image: the residual
        stream reaches a scale of 12,000 by the last block, where bfloat16 resolves to about 64,
        and a bf16 residual stream ended 0.78 cosine from the float32 reference (0.98 after the
        merger; block 13 alone 0.86). The weights stay bf16 in memory -- 0.89 GB -- and each
        matmul promotes to float32 on the way; the activations are a few megabytes. It was also
        not slower: 699 ms against 847 ms for the bf16 run on the same image.
        """
        taps, wgt, cos, sin, segs = self._positions(grids)
        x = self.patch_embed(pixel_values.astype(mx.float32))
        pos = (self.pos_embed(taps).astype(mx.float32) * wgt[:, :, None]).sum(axis=1)   # (N, dim)
        x = x + pos
        for blk in self.blocks:
            x = blk(x, cos, sin, segs)
            # Evaluated block by block. Left lazy, the whole graph's float32 promotions of every
            # block's weights are alive at once: measured 1.2 GB above the resident tower for any
            # image size. Evaluated as it goes, only one block's are.
            mx.eval(x)
        x = self.merger_norm(x).reshape(-1, self.dim * self.merge * self.merge)
        return self.merger_fc2(nn.gelu(self.merger_fc1(x)))                     # (N/4, out_dim) float32

    @property
    def nbytes(self) -> int:
        from mlx.utils import tree_flatten
        return sum(v.nbytes for _, v in tree_flatten(self.parameters()))


def load_tower(model_dir: str, dtype=mx.bfloat16) -> VisionTower:
    """The tower from the checkpoint's own shards. Only `vision_tower.*` tensors are materialised."""
    with open(os.path.join(model_dir, "config.json")) as f:
        cfg = json.load(f)
    vcfg = cfg.get("vision_config")
    if not vcfg:
        raise ValueError("this checkpoint has no vision tower")
    tower = VisionTower(vcfg)
    idx = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))["weight_map"]
    files = sorted({v for k, v in idx.items() if k.startswith("vision_tower.")})
    if not files:
        raise ValueError("this checkpoint's weights hold no vision tower")
    params = {}
    for fn in files:
        raw = mx.load(os.path.join(model_dir, fn))
        for k, v in raw.items():
            if not k.startswith("vision_tower."):
                continue
            name = k[len("vision_tower."):]
            if name == "patch_embed.proj.weight":
                # MLX's channel-last conv layout (out, T, H, W, in) back to the reference's
                # (out, in, T, H, W), flattened to match the patch order pixel_values uses.
                v = v.transpose(0, 4, 1, 2, 3).reshape(v.shape[0], -1)
            name = (name.replace("merger.norm.", "merger_norm.").replace("merger.linear_fc1.", "merger_fc1.")
                        .replace("merger.linear_fc2.", "merger_fc2.").replace("patch_embed.proj.", "patch_embed.")
                        .replace(".mlp.linear_fc1.", ".fc1.").replace(".mlp.linear_fc2.", ".fc2.")
                        .replace(".attn.qkv.", ".qkv.").replace(".attn.proj.", ".proj."))
            params[name] = v.astype(dtype)
    tower.load_weights(list(params.items()), strict=True)
    mx.eval(tower.parameters())
    return tower


# ------------------------------------------------------------------------------ text-side positions
def rope_positions(ids: list, image_token_id: int, grids: list, merge: int) -> tuple:
    """(positions (3, L) int32, delta) for a token list holding image placeholders, the reference's
    `get_rope_index` for images. Text tokens count up; an image's tokens take (t, h, w) positions
    from the current count and advance it by max(h, w) // merge; the delta is what later text
    must add to its offset."""
    L = len(ids)
    out = np.zeros((3, L), dtype=np.int64)
    cur = 0
    i = 0
    gi = 0
    while i < L:
        if ids[i] != image_token_id:
            j = i
            while j < L and ids[j] != image_token_id:
                j += 1
            out[:, i:j] = np.arange(j - i) + cur
            cur += j - i
            i = j
        else:
            if gi >= len(grids):
                raise ValueError("more image placeholders than images")
            t, gh, gw = grids[gi]
            gi += 1
            lt, lh, lw = t, gh // merge, gw // merge
            n = lt * lh * lw
            if ids[i:i + n] != [image_token_id] * n:
                raise ValueError("an image's placeholder run does not match its grid")
            tt, hh, ww = np.meshgrid(np.arange(lt), np.arange(lh), np.arange(lw), indexing="ij")
            out[0, i:i + n] = tt.reshape(-1) + cur
            out[1, i:i + n] = hh.reshape(-1) + cur
            out[2, i:i + n] = ww.reshape(-1) + cur
            cur += max(lh, lw)
            i += n
    if gi != len(grids):
        raise ValueError("fewer image placeholders than images")
    delta = int(out.max()) + 1 - L
    return out.astype(np.int32), delta


class MRope:
    """Interleaved multimodal rotary embedding for the text model's full-attention layers.

    Frequencies are laid out [T H W T H W ...] up to each section's count (the reference's
    `apply_interleaved_mrope`), so the first `mrope_section[0]` frequency triples use the T
    position, the H section the H position, the W section the W position, and the tail (T only)
    the T position. With all three streams equal -- every text-only token -- this is exactly the
    plain rotary the text engine already applies.
    """

    def __init__(self, dims: int, base: float, section: list):
        self.dims = int(dims)
        half = self.dims // 2
        self.inv_freq = 1.0 / (float(base) ** (np.arange(0, self.dims, 2, dtype=np.float64) / self.dims))
        which = np.zeros(half, dtype=np.int64)                  # which position stream each frequency reads
        for stream, count in ((1, section[1]), (2, section[2])):
            length = count * 3
            which[stream:length:3] = stream
        self.which = which

    def cos_sin(self, positions: np.ndarray) -> tuple:
        """(cos, sin) of shape (L, dims) for positions (3, L)."""
        pos = positions.astype(np.float64)                      # (3, L)
        freqs = pos[:, :, None] * self.inv_freq[None, None, :]   # (3, L, half)
        picked = np.take_along_axis(freqs, np.broadcast_to(self.which[None, None, :], (1,) + freqs.shape[1:]), axis=0)[0]
        emb = np.concatenate([picked, picked], axis=-1)          # (L, dims)
        return mx.array(np.cos(emb).astype(np.float32)), mx.array(np.sin(emb).astype(np.float32))

    def apply(self, x, cos, sin):
        """Rotate the first `dims` of the head dimension of x (B, heads, L, D), the rest untouched."""
        d = self.dims
        rot, rest = x[..., :d], x[..., d:]
        r32 = rot.astype(mx.float32)
        c, s = cos[None, None, :, :], sin[None, None, :, :]
        out = (r32 * c + _rotate_half(r32) * s).astype(x.dtype)
        return mx.concatenate([out, rest], axis=-1) if rest.shape[-1] else out


# ------------------------------------------------------------------------------ the engine's side
class RopeSwitch:
    """Stands in for a full-attention layer's rotary. Plain rotary until a multimodal request is
    in flight; then the multimodal positions inside the prompt, and the plain rotary at
    `offset + delta` for everything generated after it -- the reference's `rope_deltas` rule.
    Text-only requests never arm it, so their arithmetic is exactly what it was."""

    def __init__(self, inner, mrope: MRope):
        self.inner, self.mrope = inner, mrope
        self.positions = None
        self.cos = self.sin = None
        self.delta = 0

    def arm(self, positions: np.ndarray, delta: int) -> None:
        self.positions = positions
        self.cos, self.sin = self.mrope.cos_sin(positions)
        self.delta = int(delta)

    def disarm(self) -> None:
        self.positions = None
        self.cos = self.sin = None
        self.delta = 0

    def __call__(self, x, offset: int = 0):
        if self.positions is None:
            return self.inner(x, offset=offset)
        L, Lp = int(x.shape[-2]), int(self.positions.shape[1])
        if offset >= Lp:                                   # generated tokens: one stream, shifted
            return self.inner(x, offset=offset + self.delta)
        if offset + L <= Lp:                               # inside the prompt
            return self.mrope.apply(x, self.cos[offset:offset + L], self.sin[offset:offset + L])
        # A pass straddling the prompt's end: prompt positions, then shifted plain ones.
        tail = np.tile(np.arange(Lp, offset + L, dtype=np.int32) + self.delta, (3, 1))
        pos = np.concatenate([self.positions[:, offset:Lp], tail], axis=1)
        c, s = self.mrope.cos_sin(pos)
        return self.mrope.apply(x, c, s)


def full_attention_layers(model) -> list:
    """The attention modules that carry a rotary, in layer order."""
    tm = getattr(model, "language_model", None) or model
    inner = getattr(tm, "model", None)
    out = []
    for layer in getattr(inner, "layers", []) or []:
        attn = getattr(layer, "self_attn", None)
        if attn is not None and hasattr(attn, "rope") and not getattr(layer, "is_linear", False):
            out.append(attn)
    return out


def _image_items(content) -> list:
    if not isinstance(content, list):
        return []
    return [it for it in content if isinstance(it, dict)
            and ("image_url" in it or "image" in it or it.get("type") == "image")]


def count_images(messages) -> int:
    return sum(len(_image_items(m.get("content"))) for m in (messages or []) if isinstance(m, dict))


def extract_images(messages) -> list:
    """The bytes of every image in the messages, in the order the template renders them.
    OpenAI: {"type": "image_url", "image_url": {"url": <data URL>}} (or a string url).
    Anthropic: {"type": "image", "source": {"type": "base64", "data": ...}}.
    Raises ValueError with a sentence for anything that cannot be decoded."""
    out = []
    for m in messages or []:
        if not isinstance(m, dict):
            continue
        for it in _image_items(m.get("content")):
            if "image_url" in it:
                u = it["image_url"]
                u = u.get("url") if isinstance(u, dict) else u
                if not isinstance(u, str):
                    raise ValueError("`image_url` must be a string or {\"url\": ...}")
                out.append(decode_image_ref(u))
            elif isinstance(it.get("source"), dict):
                src = it["source"]
                if src.get("type") != "base64" or not isinstance(src.get("data"), str):
                    raise ValueError("an image `source` must be {\"type\": \"base64\", \"data\": ...}; "
                                     "URLs are not fetched by this server")
                out.append(decode_image_ref(src["data"]))
            elif isinstance(it.get("image"), str):
                out.append(decode_image_ref(it["image"]))
            else:
                raise ValueError("an image item needs `image_url` (a data URL) or a base64 `source`")
    if len(out) > MAX_IMAGES:
        raise ValueError(f"{len(out)} images in one request; at most {MAX_IMAGES}")
    return out


def expand_placeholders(ids: list, grids: list, image_token_id: int, merge: int) -> list:
    """Each single `<|image_pad|>` the template wrote becomes the image's run of placeholders,
    one per 2x2 block -- the processor's expansion."""
    out, gi = [], 0
    for t in ids:
        if t == image_token_id:
            if gi >= len(grids):
                raise ValueError("more image placeholders in the prompt than images")
            t_, gh, gw = grids[gi]
            gi += 1
            out.extend([image_token_id] * (t_ * gh * gw // (merge * merge)))
        else:
            out.append(t)
    if gi != len(grids):
        raise ValueError("fewer image placeholders in the prompt than images")
    return out
