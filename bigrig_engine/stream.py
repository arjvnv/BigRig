"""REAL STREAMING: a bounded expert pool that replaces MLX's full-tensor gather.

WHAT THIS FIXES THAT SHADOW MODE DID NOT
    Every earlier engine component OBSERVED routing and fetched real bytes, but the model still
    called `mx.gather_qmm` against the complete (E, out, in) expert tensor. MLX loads that tensor
    from a memory-mapped safetensors file, so nothing was ever actually saved -- the OS page cache
    decided what stayed resident, and the model's peak footprint was still the whole file.

    This module makes the saving real. The model holds C expert slots, never E. C is chosen from
    the memory budget. Experts outside the pool live on disk and are read with pread on demand.

WHY A POOL AND NOT JUST mmap
    Three measurements from this project, each of which had to be made twice because the first
    one generalised a single configuration into a law:

      1. mmap fault path        0.70 GB/s    <- what you get by letting the OS do it
         explicit pread         3.47 GB/s    <- what this does
      2. one reader thread      2.23 GB/s
         twelve reader threads  4.76 GB/s    <- pread can be parallel; a page fault cannot
      3. OS page cache eviction is approximately LRU. On real routing traces LFUDA beats LRU by
         23.7%, and at kappa=18.6 a miss costs ~18x a hit, so that gap is not cosmetic.

    So: explicit residency, explicit reads, our eviction policy. Not the kernel's.

THE INVARIANT THIS MODULE LIVES OR DIES BY
    With C == E the streamed output must be BIT-IDENTICAL to the unstreamed model. Not close --
    identical. The same bytes go through the same kernel; only their address changed. Any drift
    means a slot is being written wrong, and a numerically-plausible-but-wrong answer is the
    single worst failure this product can ship. test_stream.py asserts exact equality, and the
    end-to-end check asserts identical generated TEXT.
"""
from __future__ import annotations

import inspect
import json
import os
import time

from . import home
import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .fetch import ParallelFetcher, Region, WeightStore
from .pool import _policy_factory

# The three projections of a SwiGLU expert, in the order they are packed on disk.
# TWO EXPERT SHAPES ARE IN THE WILD, AND THEY ARE DIFFERENT ARITHMETIC.
#     Almost everything -- Qwen, DeepSeek, GLM, gpt-oss, OLMoE -- uses a gated pair: the expert
#     is `down(act(gate(x)) * up(x))`, three projections. Nemotron uses a plain two-layer MLP:
#     `fc2(act(fc1(x)))`, two projections and no multiply. The names are the reliable signal for
#     which, because they are what the checkpoint carries. `PROJECTIONS` is the gated default,
#     used only where no model is in hand; everywhere a spec IS in hand, `projections_of` reads
#     the truth off it, so a model never has its own layout guessed at.
PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
MLP_PROJECTIONS = ("fc1", "fc2")


def first_projection(sg):
    """The expert bank's first projection module, whichever shape it is.

    Quantisation parameters are read off it, and they are identical across an expert's
    projections. Reading `sg.gate_proj` directly raised `'SwitchMLP' object has no attribute
    'gate_proj'` on Nemotron in three separate places, so there is one place now.
    """
    mod = getattr(sg, "gate_proj", None)
    return mod if mod is not None else getattr(sg, "fc1", None)


def projections_of(spec: dict) -> tuple:
    """The projections this model's experts actually carry, in the order its bytes are laid out."""
    return tuple(spec)


def is_gated(projections) -> bool:
    """True for the gate/up/down shape, False for the fc1/fc2 one."""
    return "gate_proj" in tuple(projections)
# Component arrays within one projection. `biases` is the quantisation zero-point and is absent
# for some quant modes; `bias` (singular) is the affine bias and is usually absent too.
COMPONENTS = ("weight", "scales", "biases", "bias")


# --------------------------------------------------------------------------- packing to disk
def _dtype_name(a) -> str:
    return str(a.dtype).rsplit(".", 1)[-1]


def _raw(a) -> np.ndarray:
    """One MLX array as flat host bytes, for ANY dtype.

    Transporting as uint8 rather than as the array's own numpy dtype is not fussiness. bfloat16
    has no numpy equivalent, so `np.array(bf16_array)` raises
        "Item size 2 for PEP 3118 buffer format string B does not match the dtype B item size 1"
    and every 3-bit and bf16 model -- which is most of the ones worth streaming -- would be
    unsupported. Bytes are bytes; the dtype is restored from the manifest on the way back in.
    """
    return np.array(mx.contiguous(a).view(mx.uint8), copy=False).reshape(-1)


def _expert_slice(arr, e: int) -> np.ndarray:
    """Expert e's slab of a (E, ...) parameter, as contiguous host bytes."""
    return _raw(arr[e])


def pack_experts(model, out_path: str, progress=True) -> dict:
    """Write every MoE expert to an expert-contiguous blob and return its manifest.

    Expert-contiguous means all of expert e's tensors for one layer sit in one byte range, so
    admitting an expert is ONE read. The alternative -- safetensors' natural tensor-major order --
    scatters an expert's three projections across the file and costs three seeks. Component 7
    measured that difference at 1.58x on the real blob.
    """
    out_path = os.path.expanduser(out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    layers = find_moe_layers(model)
    if not layers:
        raise ValueError("no MoE layers found -- this model has nothing to stream")

    manifest = {"regions": {}, "layers": {}, "version": 4}
    off = 0
    t0 = time.perf_counter()
    with open(out_path, "wb") as f:
        for li, sg in layers:
            E = _num_experts(sg)
            spec = {}
            # This model's own projections, not the gated default: Nemotron carries fc1/fc2.
            for proj in (PROJECTIONS if hasattr(sg, "gate_proj") else MLP_PROJECTIONS):
                mod = getattr(sg, proj)
                spec[proj] = {}
                for comp in COMPONENTS:
                    arr = mod.get(comp) if hasattr(mod, "get") else None
                    if arr is None:
                        continue
                    a0 = arr[0]
                    spec[proj][comp] = {"shape": [int(x) for x in a0.shape],
                                        "dtype": _dtype_name(a0),
                                        "nbytes": int(_raw(a0).nbytes)}
            per_expert = sum(c["nbytes"] for p in spec.values() for c in p.values())
            # The blob must carry its own quantisation parameters. Reading them off the live
            # module works only while the blob and the model agree -- which stops being true the
            # moment a blob is requantised to a different precision, and a gather run with the
            # wrong bits produces plausible garbage rather than an error.
            _first = first_projection(sg)
            g0 = getattr(_first, "group_size", None)
            b0 = getattr(_first, "bits", None)
            entry = {"n_experts": E, "spec": spec, "bytes_per_expert": per_expert}
            if g0 and b0:
                entry["quant"] = {"group_size": int(g0), "bits": int(b0),
                                  "mode": getattr(_first, "mode", "affine")}
            manifest["layers"][str(li)] = entry
            for e in range(E):
                start = off
                for proj in spec:
                    mod = getattr(sg, proj)
                    for comp in COMPONENTS:
                        if comp not in spec[proj]:
                            continue
                        b = _expert_slice(mod[comp], e).tobytes()
                        if len(b) != spec[proj][comp]["nbytes"]:
                            raise ValueError(
                                f"layer {li} expert {e} {proj}.{comp}: packed {len(b)} bytes "
                                f"but the manifest says {spec[proj][comp]['nbytes']}. A shape "
                                f"that varies per expert would silently corrupt every slot.")
                        f.write(b)
                        off += len(b)
                manifest["regions"][f"{li}:{e}"] = [start, off - start]
            if progress:
                print(f"    layer {li:>3}: {E} experts x {per_expert/1e6:.1f} MB "
                      f"= {E*per_expert/1e9:.2f} GB  [{time.perf_counter()-t0:.0f}s]", flush=True)
    manifest["total_bytes"] = off
    tk = None
    for layer in getattr(getattr(model, "model", model), "layers", []):
        blk = getattr(layer, "mlp", None)
        for attr in ("top_k", "num_experts_per_tok"):
            v = getattr(blk, attr, None) if blk is not None else None
            if isinstance(v, int) and v > 0:
                tk = v
                break
        if tk:
            break
    if tk:
        manifest["top_k"] = int(tk)
    with open(out_path + ".manifest.json", "w") as f:
        json.dump(manifest, f)
    return manifest


def expert_source(model_dir: str, blob_path: str = "", allow_direct: bool = True) -> tuple:
    """Where this model's experts will be read from. Returns (manifest, blob path or "").

    Prefers a packed blob when one already exists -- it is a single read per expert instead of
    nine, measured 1.2x faster on the fetch path. Falls back to reading the model's own
    safetensors, which is 18% slower there and costs nothing on disk.

    PACKING IS DONE ON A STREAMED MODEL'S FIRST RUN, and this function is what finds the result.
    It was once opt-in, because it doubles a user's disk (26 GB for a 13.4 GB download) before
    they have seen the thing work once. What changed the default is that the packed blob is the
    only layout the zero-copy admit path can use -- 0 of 360 tensors in the raw shards are
    page-aligned -- so without it the fast path is unreachable and the tune measures an engine
    the model will not run on. `_auto_pack` prints the size first, skips itself when the disk is
    tight, and `--no-pack` declines it.
    """
    model_dir = os.path.expanduser(model_dir)
    blob_path = os.path.expanduser(blob_path) or os.path.join(
        os.path.join(home(), "data", "blobs"), os.path.basename(model_dir) + ".experts")
    if os.path.exists(blob_path) and os.path.exists(blob_path + ".manifest.json"):
        try:
            m = load_manifest(blob_path)
            if os.path.getsize(blob_path) == m["total_bytes"] and m.get("version", 0) >= 4:
                return m, blob_path
        except (ValueError, KeyError, OSError):
            pass                      # a damaged blob is not a reason to fail; read the model
    if not allow_direct:
        raise FileNotFoundError(
            f"no usable packed blob at {blob_path} and direct reading was disabled")
    from .direct import expert_manifest
    return expert_manifest(model_dir), ""


def load_manifest(blob_path: str) -> dict:
    p = os.path.expanduser(blob_path) + ".manifest.json"
    if not os.path.exists(p):
        raise FileNotFoundError(
            f"{p} not found -- run pack_experts() before streaming. Streaming without a manifest "
            f"would have to guess the byte layout, and a wrong guess reads plausible garbage.")
    with open(p) as f:
        return json.load(f)


def store_from_manifest(blob_path: str, manifest: dict) -> WeightStore:
    """A store for either a packed blob or the model's own shards, from the same manifest shape."""
    if manifest.get("direct"):
        from .direct import Segment
        layout = {tuple(int(x) for x in k.split(":")):
                  [Segment(p, o, n) for p, o, n in segs]
                  for k, segs in manifest["segments"].items()}
        return WeightStore("", layout)
    layout = {tuple(int(x) for x in k.split(":")): Region(o, n)
              for k, (o, n) in manifest["regions"].items()}
    return WeightStore(os.path.expanduser(blob_path), layout)


# --------------------------------------------------------------------------- model inspection
def _num_experts(sg) -> int:
    return int(first_projection(sg)["weight"].shape[0])


def load_lenient(model_dir: str, lazy: bool = True):
    """Load a model, tolerating LEFTOVER weights in the checkpoint but never MISSING ones.

    WHY THIS IS NEEDED
        Community checkpoints commonly ship a quantised `lm_head.scales` / `.biases` alongside
        `tie_word_embeddings: true`, which means the model has no separate lm_head at all. The
        weights are harmless leftovers, but mlx_lm's strict load refuses the whole model over
        them. Measured on Qwen3-MOE-4x0.6B: refuses to load strictly, loads and generates
        correctly ("The capital of France is" -> " Paris") once the extras are tolerated.

    WHY IT IS NOT JUST strict=False
        MLX checks for EXTRA parameters before it checks for MISSING ones, so a checkpoint that
        has both reports only the extras. Blindly retrying with strict=False would then accept a
        model with genuinely absent weights, which loads happily and generates fluent nonsense --
        the single worst failure this project can ship. So after the lenient load, the model's
        own parameter set is compared against the checkpoint's and any true absence is raised.
    """
    from mlx_lm import load as _load
    model_dir = os.path.expanduser(model_dir)
    try:
        return _load(model_dir, lazy=lazy)
    except ValueError as e:
        if "not in model" not in str(e):
            raise                      # missing or mis-shaped: a real problem, do not paper over

    import glob as _glob
    from mlx.utils import tree_flatten
    from mlx_lm.utils import load_model, load_tokenizer
    from pathlib import Path
    path = Path(model_dir)
    model, cfg = load_model(path, lazy=lazy, strict=False)

    weights = {}
    for wf in _glob.glob(str(path / "model*.safetensors")):
        weights.update(mx.load(wf))
    if hasattr(model, "sanitize"):
        weights = model.sanitize(weights)
    expected = {k for k, _ in tree_flatten(model.parameters())}
    missing = sorted(expected - set(weights))
    if missing:
        raise ValueError(
            f"{path.name} is missing {len(missing)} weights the model needs, e.g. "
            f"{', '.join(missing[:3])}. Refusing to load: a model with absent weights still "
            f"generates fluent text, so this would fail silently rather than loudly.")
    extra = sorted(set(weights) - expected)
    print(f"  note: ignoring {len(extra)} leftover weight(s) not used by this architecture "
          f"({', '.join(extra[:3])}{'...' if len(extra) > 3 else ''})")
    return model, load_tokenizer(path, eos_token_ids=cfg.get("eos_token_id"))


def model_top_k(model_dir: str, manifest: dict | None = None, default: int = 8) -> int:
    """How many experts this model routes to per token.

    Read from the model's own config, never assumed. The pool must hold at least top-k experts
    per layer or a single token cannot be served, so a hardcoded 8 silently over-reserves on a
    top-4 model and -- far worse -- lets a top-16 model be configured into a pool that will
    raise partway through the first prompt.
    """
    if manifest and manifest.get("top_k"):
        return int(manifest["top_k"])
    cfg = os.path.join(os.path.expanduser(model_dir), "config.json")
    if os.path.exists(cfg):
        try:
            with open(cfg) as fh:
                c = json.load(fh)
        except (json.JSONDecodeError, OSError):
            return default
        from .preflight import top_k_from_config
        return top_k_from_config(c, default)
    return default


# Where different architectures hang their fused expert bank. Qwen3, Mixtral and DeepSeek use
# `layer.mlp.switch_mlp`; gpt-oss uses `layer.mlp.experts`; Llama-4 uses
# `layer.feed_forward.experts`.
_BLOCK_ATTRS = ("mlp", "feed_forward", "block_sparse_moe", "mixer")
_SWITCH_ATTRS = ("switch_mlp", "experts")


def find_moe_layers(model):
    """Every (layer_index, SwitchGLU) in the model, in depth order.

    THE ATTRIBUTE NAME IS NOT THE THING BEING LOOKED FOR, AND MATCHING ON IT SILENTLY FAILED.
        This used to test for `layer.mlp.switch_mlp` and nothing else. On gpt-oss the same object
        is `layer.mlp.experts`, and on Llama-4 it is `layer.feed_forward.experts` -- so the engine
        found ZERO MoE blocks, attached to nothing, and streamed nothing. It would not have
        raised: a model with no MoE blocks is a legitimate thing to be handed, and the failure
        would have looked like a model that simply did not benefit.

        What identifies the expert bank is that it carries the three fused projections, so that
        is what is tested. The names are only where to look.
    """
    return [(i, sg) for i, _blk, _attr, sg in find_moe_sites(model)]


# The most speculation that may be outstanding at once, across every layer. Deliberately small
# next to the model: guesses are an optimisation and must never compete for memory or bandwidth
# with the reads the model is actually waiting on.
SPECULATION_BUDGET_BYTES = 256 << 20


def find_moe_sites(model):
    """(layer_index, owning_block, attribute_name, SwitchGLU) -- what `attach` needs to replace it.

    The owner and the attribute come back with the module because the engine has to PUT something
    back in the same place it found it, and that place is not the same on every architecture.
    """
    out = []
    # THE LAYER LIST IS NOT ALWAYS UNDER `.model`. Nemotron hangs it off `.backbone`, and a bare
    # text model has it directly. Looking in one place found zero MoE blocks and streamed
    # nothing, which does not raise -- a model with no experts is a legitimate thing to be
    # handed -- so it would have looked like a model that simply did not benefit.
    root = model
    for attr in ("model", "backbone", "language_model"):
        inner = getattr(root, attr, None)
        if inner is not None and getattr(inner, "layers", None) is not None:
            root = inner
            break
        if inner is not None and getattr(getattr(inner, "model", None), "layers", None) is not None:
            root = inner.model
            break
    for i, layer in enumerate(getattr(root, "layers", [])):
        for battr in _BLOCK_ATTRS:
            blk = getattr(layer, battr, None)
            if blk is None:
                continue
            found = False
            for sattr in _SWITCH_ATTRS:
                sg = getattr(blk, sattr, None)
                if sg is None:
                    continue
                # Either shape identifies an expert bank: the gated trio, or Nemotron's pair.
                if (hasattr(sg, "gate_proj") and hasattr(sg, "down_proj")) or \
                        (hasattr(sg, "fc1") and hasattr(sg, "fc2")):
                    out.append((i, blk, sattr, sg))
                    found = True
                    break
            if found:
                break
    return out


# --------------------------------------------------------------------------- the resident pool
class ExpertPool:
    """C slots of expert weights for ONE layer, plus the table saying which expert is in which.

    Admission writes into a slot in place. Eviction is just marking a slot free -- the bytes are
    overwritten by the next admission, never zeroed, because zeroing C slots costs a memcpy that
    buys nothing.

    THE SLOT WRITE LOOKS LIKE THE BIGGEST COST IN THE ENGINE. IT IS NOT REMOVABLE, AND HERE IS
    THE MEASUREMENT, SO NOBODY SPENDS ANOTHER DAY ON IT.
        MLX arrays are functional, so `self._slots[k][slot] = expert` does not write 4.4 MB into
        a buffer -- it rebinds the whole 177 MB tensor. Timed in isolation that is 63.7% of a
        token, and the obvious fix is to hold each slot as its own array: the write becomes a
        list assignment costing nothing, and the step's gather is built by stacking only the
        experts that step uses. Microbenchmarked on this model's real shapes with the real fused
        kernel, that looked decisive:

            misses in the step    one (C,...) tensor    per-slot + stack
                1                      1.215 ms            0.693 ms    1.75x
                2                      1.172 ms            0.683 ms    1.71x
                4                      1.196 ms            0.665 ms    1.80x

        Built and run end to end it made the model SLOWER: 6.14 -> 4.91 tok/s, 0.80x, with all
        five test replies byte-identical and the miss rate unchanged, so the implementation was
        right and the idea was wrong.

        The microbenchmark was measuring an artefact. It forced `mx.eval` between the write and
        the gather to imitate the host round-trip; the engine does not impose that barrier per
        projection, so MLX fuses the slice-update into the gather that follows it and most of
        the write never costs what it costs standing alone. What replaced it -- stacking top-k
        experts three times per layer, 48 layers deep, every step -- is unavoidable new work.

        The 63.7% figure is real as a profile line and misleading as a budget. It is not a
        removable 63.7%.

        RE-MEASURED once expert reads were allowed into the OS page cache, on the reasoning that
        cheap reads make the write a larger share of what is left, which is exactly when
        replacing it should start to pay. It does not:

            one (C,...) tensor    9.74, 8.38 tok/s      median 9.06
            per-slot arrays       7.45, 6.47 tok/s      median 6.96      0.77x

        Byte-exact again -- five of five replies identical, 871 characters both ways -- so the
        implementation was right a second time and the idea is still wrong. 0.80x before the
        page cache, 0.77x after. Two measurements, two years of plausibility, no win. Do not
        build it a third time.
    """

    def __init__(self, layer: int, sg, n_experts: int, capacity: int, spec: dict,
                 policy: str = "lfuda", quant: dict | None = None, views: bool = False):
        if not 1 <= capacity <= n_experts:
            raise ValueError(
                f"capacity must be in [1, {n_experts}], got {capacity}. A pool larger than the "
                f"expert count wastes memory; a pool of zero cannot serve any token.")
        self.layer, self.sg, self.n_experts, self.capacity = layer, sg, n_experts, capacity
        self.spec = spec
        # Read off the spec, never assumed: gate/up/down for almost everything, fc1/fc2 for
        # Nemotron. The order is the order this model's bytes are laid out in.
        self.projections = projections_of(spec)
        self.gated = is_gated(self.projections)
        # The quantisation these slots are stored in. Needed by anything that has to decode the
        # weights in place -- the tuner round-trips them through a lower precision and back, and
        # guessing the source format there would silently corrupt every slot it touched.
        self.spec_quant = dict(quant or {"bits": 4, "group_size": 64, "mode": "affine"})
        # _policy_factory returns a FACTORY, not an instance. Calling on_hit/on_admit on the
        # factory raised AttributeError immediately here, which was lucky: a policy whose view of
        # the cache silently diverges from the pool's makes every reported miss rate fiction.
        self.policy = _policy_factory(policy, capacity, n_experts)(capacity, n_experts)
        # Does this policy accept an exclusion set? If not, the pool must never work around it by
        # lying -- it falls back to choosing a victim itself instead.
        try:
            self._policy_excludes = "exclude" in inspect.signature(self.policy.victim).parameters
        except (TypeError, ValueError):
            self._policy_excludes = False
        self._slot_tick = np.zeros(capacity, dtype=np.int64)
        self._tick = 0
        # expert -> slot, and slot -> expert. -1 means absent/free.
        self.g2s = np.full(n_experts, -1, dtype=np.int32)
        self.s2g = np.full(capacity, -1, dtype=np.int32)
        # The same table as an MLX array. A layer that can never miss reads its slots straight
        # off this on the GPU and never touches the host, which is what makes it sync-free.
        self._g2s_mx = None
        self._g2s_dirty = True
        self._free = list(range(capacity))
        self._resident = 0
        self.hits = 0
        self.misses = 0
        # HOW OFTEN EACH EXPERT WAS ASKED FOR, kept for the warm pass: a cache too small for the
        # whole file should hold the experts this model actually uses, not the first bytes of it.
        self.uses = np.zeros(n_experts, dtype=np.int64)
        self.admits = 0
        # Seconds spent copying missed experts into slots -- the host->GPU upload. Measured at
        # 22.5 ms a token against 1.7 ms of fetch on Qwen3.6-35B-A3B-4bit: THIS is what a miss
        # costs, and anything pricing a miss from fetch time alone is off by an order of magnitude.
        self.admit_seconds = 0.0
        # STAGED SPECULATION. The previous layer's predictor names experts this layer will
        # probably want; their bytes are copied to the GPU while the host is otherwise idle
        # waiting for this layer's routing, and ensure() then admits them without a host copy.
        # A wrong guess costs one idle-time memcpy and nothing else -- ensure() still admits
        # exactly what the router chose. See StreamingSwitchGLU._stage_predicted.
        self.stage_pred = None     # expert ids named for THIS layer, set by the previous one
        self.stage = {}            # expert id -> tuple of device arrays, ready to write
        self.stage_issued = 0
        self.stage_hits = 0
        self.zero_copy_admits = 0
        self.evicts = 0
        self._slots = {}          # (proj, comp) -> mx.array of shape (C, ...)
        self.views = bool(views)  # resident experts as live page-cache views, no slot tensors
        self._views = {}          # expert id -> tuple of device arrays, (proj, comp) order
        self._built = False

    # -- construction -------------------------------------------------------
    @property
    def bytes_per_expert(self) -> int:
        """One expert's packed size at this layer -- what a speculative read actually costs."""
        return sum(c["nbytes"] for pr in self.spec.values() for c in pr.values())

    def build(self):
        """Allocate the C-slot arrays and DROP the model's full expert tensors.

        The drop is the whole point. Until the module stops referencing the (E, ...) arrays, MLX
        keeps them alive and the pool is pure overhead -- which is exactly what shadow mode was.
        """
        for proj in self.projections:
            mod = getattr(self.sg, proj)
            # Drop the (E, ...) originals BEFORE allocating this projection's slots, not after.
            # Allocating first held both alive at once and pushed peak RSS to 7.06 GB on a 3.6 GB
            # model -- the pool looked like pure overhead, which is precisely the bug streaming
            # exists to fix. Order matters more than the drop itself.
            for comp in list(self.spec[proj]):
                try:
                    del mod[comp]
                except Exception:
                    setattr(mod, comp, None)
            mx.clear_cache()
            if self.views:
                continue          # no slot tensors: a resident expert is a live view
            for comp, meta in self.spec[proj].items():
                shape = (self.capacity, *meta["shape"])
                dt = getattr(mx, meta["dtype"])
                self._slots[(proj, comp)] = mx.zeros(shape, dtype=dt)
            mx.eval([self._slots[(proj, c)] for c in self.spec[proj]])
        mx.clear_cache()
        self._built = True
        return self

    def nbytes(self) -> int:
        if self.views:
            return self._resident * self.bytes_per_expert
        return sum(int(np.prod(v.shape)) * v.dtype.size for v in self._slots.values())

    def views_of(self, e: int) -> tuple:
        """The resident arrays of expert e in views mode. KeyError if it is not resident."""
        return self._views[int(e)]

    # -- residency ----------------------------------------------------------
    def slot_of(self, e: int) -> int:
        return int(self.g2s[e])

    def touch(self, experts, ranks=None) -> list:
        """Record a reference to each expert and return those NOT resident.

        `ranks` is the router's own top-k position for each expert (0 = highest scoring). It is
        free -- the router already computed it -- and the tuned policies read it, so dropping it
        would quietly evaluate every policy in its untuned configuration.
        """
        need, nrank = [], []
        ranks = list(ranks) if ranks is not None else [0] * len(experts)
        for e, r in zip(experts, ranks):
            e, r = int(e), int(r)
            self.uses[e] += 1
            if self.g2s[e] >= 0:
                self.hits += 1
                self.policy.touch(e, r)
            else:
                self.misses += 1
                need.append(e)
                nrank.append(r)
        self._pending_rank = dict(zip(need, nrank))
        return need

    def _victim(self, protect=()) -> int:
        """Free a slot. The policy is only ever told the truth about what left the cache.

        THE BUG THIS REPLACED
            The previous version asked the policy for a victim, and when that victim was one the
            current token still needed, it called `policy.evicted(victim)` and asked again. That
            call is a lie: nothing was evicted. For LFUDA it removed the expert from the policy's
            resident set while it stayed in the pool, so `victim()` could never nominate it
            again, and it raised the global age L as though a real eviction had occurred.
            Measured after 200 steps at C=12: the policy could see 1 resident expert out of 12,
            and L had been driven to 1576 by 996 evictions that never happened. Output stayed
            correct -- the pool's own tables were fine -- but the eviction policy had silently
            stopped working, and the miss rate is what the product's speed is made of.
        """
        if self._free:
            return self._free.pop()
        protect = {int(x) for x in protect}
        ev = -1
        v = (self.policy.victim(exclude=protect) if self._policy_excludes
             else self.policy.victim())
        if v is not None and v >= 0 and v not in protect and self.g2s[v] >= 0:
            ev = int(v)
        if ev < 0:
            # The policy could not name a legal victim. Choose one here, on least-recently
            # admitted, and tell the policy about THIS eviction only.
            best, best_t = -1, None
            for slot in range(self.capacity):
                g = int(self.s2g[slot])
                if g < 0 or g in protect:
                    continue
                if best_t is None or self._slot_tick[slot] < best_t:
                    best, best_t = g, self._slot_tick[slot]
            ev = best
        if ev < 0:
            raise RuntimeError(
                f"layer {self.layer}: every one of {self.capacity} slots holds an expert this "
                f"step still needs. Capacity must be at least top-k.")
        slot = int(self.g2s[ev])
        if hasattr(self.policy, "evicted"):
            self.policy.evicted(ev)
        self.g2s[ev] = -1
        self.s2g[slot] = -1
        self._views.pop(ev, None)       # views mode: dropping the view releases the pages
        self._resident -= 1
        self._g2s_dirty = True
        self.evicts += 1
        return slot

    def admit(self, e: int, blob: bytes, protect=()) -> int:
        """Place expert e's packed bytes into a slot and return the slot index."""
        if not self._built:
            raise RuntimeError("build() must run before admit()")
        want = sum(c["nbytes"] for pr in self.spec.values() for c in pr.values())
        if len(blob) != want:
            # Checked BEFORE any slot is touched. Letting numpy discover it mid-write leaves the
            # slot half-overwritten with one expert's head and another's tail -- still a valid
            # tensor, still fluent output, silently wrong.
            raise ValueError(
                f"layer {self.layer} expert {e}: the manifest accounts for {want} bytes but the "
                f"region supplied {len(blob)}. Refusing to write a partially-filled slot.")
        _t0 = time.perf_counter()
        return self.admit_arrays(e, self.build_arrays(e, blob), protect=protect, _t0=_t0)

    def build_arrays(self, e: int, blob) -> tuple:
        """One expert's packed bytes as its device arrays, in (projection, component) order.

        The ONLY place bytes become arrays. Staged speculation and the direct path both call
        it, so a staged expert is bit-for-bit the array a direct admit would have written --
        the 1-D uint8 -> view -> reshape sequence, never a view of a 2-D or sliced array, which
        MLX does not reproduce identically for bfloat16.
        """
        out, pos = [], 0
        # ZERO-COPY WHEN THE BYTES ALREADY SIT IN MEMORY THE GPU CAN READ.
        #     On Apple Silicon the page cache and the GPU share one memory. A memoryview from
        #     the expert map is those pages; mx.from_dlpack wraps them as a Metal buffer with no
        #     copy (measured: MLX active memory +0.00 MB for a 1.69 MB expert, bit-identical),
        #     and the slot write below becomes a GPU-side copy instead of a CPU memcpy of ~350
        #     MB a token at ~25 GB/s. Metal needs the base page-aligned and the length a whole
        #     number of pages; the packed blob lays every expert out that way, and anything that
        #     is not falls back to the copy. One import per expert, not one per component: nine
        #     Metal buffers an expert cost more than the memcpy they replaced (measured 0.66x),
        #     one costs less (1.11x on the copy alone). The whole map as ONE buffer was faster
        #     still (1.37x) and wired 316 MB of a 348 MB file the moment a slice was used --
        #     Metal keeps a referenced buffer resident whole -- so on a 17 GB blob it would wire
        #     17 GB. Not that. Per expert, 1.77 MB at a time, released with the array.
        flat = None
        if ZERO_COPY and isinstance(blob, memoryview):
            arr = np.frombuffer(blob, dtype=np.uint8)
            if arr.ctypes.data % PAGE_BYTES == 0 and arr.nbytes % PAGE_BYTES == 0:
                try:
                    flat = mx.from_dlpack(arr)
                    self.zero_copy_admits += 1
                except Exception:                 # noqa: BLE001 -- the copy path is always right
                    flat = None
        for proj in self.projections:
            for comp, meta in self.spec[proj].items():
                n = meta["nbytes"]
                dt = getattr(mx, meta["dtype"])
                if flat is not None:
                    out.append(flat[pos:pos + n].view(dt).reshape(meta["shape"]))
                else:
                    # Explicit count, never count=-1: frombuffer with -1 raises whenever the
                    # remaining tail is not a whole multiple of the itemsize, which turns a
                    # benign layout question into a crash mid-generation.
                    raw = np.frombuffer(blob, dtype=np.uint8, count=n, offset=pos)
                    out.append(mx.array(raw).view(dt).reshape(meta["shape"]))
                pos += n
        if pos != len(blob):
            raise ValueError(
                f"expert {e}: manifest accounts for {pos} bytes but the region is {len(blob)}. "
                f"The pool would be reading one expert's tail as another's head.")
        return tuple(out)

    def admit_arrays(self, e: int, arrays: tuple, protect=(), _t0=None) -> int:
        """Place already-built device arrays into a slot. Bookkeeping identical to admit()."""
        if _t0 is None:
            _t0 = time.perf_counter()
        slot = self._victim(protect=protect)
        if self.views:
            self._views[int(e)] = tuple(arrays)
        else:
            i = 0
            for proj in self.projections:
                for comp in self.spec[proj]:
                    self._slots[(proj, comp)][slot] = arrays[i]
                    i += 1
        if self.s2g[slot] < 0:
            self._resident += 1
        self.g2s[e] = slot
        self.s2g[slot] = e
        self._g2s_dirty = True
        self._tick += 1
        self._slot_tick[slot] = self._tick
        self.policy.admit(e, int(getattr(self, "_pending_rank", {}).get(e, 0)))
        self.admits += 1
        self.admit_seconds += time.perf_counter() - _t0
        return slot

    @property
    def full(self) -> bool:
        """True when every expert is resident, so no request can ever miss.

        Counted, not scanned. This is consulted per layer per token on the sync-free path, and
        an O(E) numpy scan there would be 128 elements x 48 layers x every token.
        """
        return self.capacity == self.n_experts and self._resident == self.n_experts

    def g2s_device(self):
        """The expert->slot table on the GPU, rebuilt only when residency actually changed."""
        if self._g2s_dirty or self._g2s_mx is None:
            self._g2s_mx = mx.array(np.maximum(self.g2s, 0).astype(np.uint32))
            mx.eval(self._g2s_mx)
            self._g2s_dirty = False
        return self._g2s_mx

    def arrays(self, proj: str):
        s = self._slots
        return (s[(proj, "weight")], s.get((proj, "scales")), s.get((proj, "biases")),
                s.get((proj, "bias")))

    def stats(self) -> dict:
        tot = self.hits + self.misses
        return {"layer": self.layer, "capacity": self.capacity, "n_experts": self.n_experts,
                "hits": self.hits, "misses": self.misses,
                "miss_rate": self.misses / tot if tot else 0.0,
                "admits": self.admits, "evicts": self.evicts,
                "resident_bytes": self.nbytes()}


# --------------------------------------------------------------------------- the streaming layer
class StreamingSwitchGLU(nn.Module):
    """Drop-in for mlx_lm's SwitchGLU that gathers from C slots instead of E experts.

    The math is untouched: same gather kernel, same activation, same order. The ONLY change is
    that `indices` are rewritten from global expert ids to pool slot ids after the needed experts
    have been made resident.
    """

    def __init__(self, pool: ExpertPool, fetcher: ParallelFetcher, activation,
                 quantized: bool, qparams: dict):
        super().__init__()
        self._pool = pool
        self._fetcher = fetcher
        self._activation = activation
        self._quantized = quantized
        self._q = qparams
        self.fetch_seconds = 0.0
        self.fetch_bytes = 0
        self.chunked_calls = 0
        self.views_calls = 0            # prefill chunks served straight from page-cache views
        self.sync_seconds = 0.0
        self.bypass = False        # timing control; see __call__
        self._sync_free = False    # set by refresh() once the pool is known to be full
        # Prefetch, driven by a predictor over this layer's hidden state. See predict.py for why
        # the hidden state works where the routing history does not, and why being wrong is safe.
        self._pred_w = None        # (hidden, n_experts) weights, on device (fitted predictor)
        self._next_gate = None     # the NEXT layer's own router, applied to this layer's input
        self._pred_m = 0           # how many experts to name
        self._pred_budget = 0      # bytes of speculation this layer may have outstanding
        # Cache-aware rerouting. OFF unless a tolerance is set, because it CHANGES THE ANSWER.
        self._gate = None          # this layer's router, so a substitute can be scored
        self._reroute = 0.0        # how much of an expert's gate probability may be given up
        self.reroutes = 0
        self.reroute_lost = 0.0
        self._next = None          # the pool this prediction is ABOUT (the next layer's)
        self._speculated = ()      # keys issued for this layer, so leftovers can be released
        self.pred_issued = 0
        self.pred_dropped = 0

    def _reroute_to_resident(self, flat, x):
        """Swap an absent expert for a resident one the router scored nearly as highly.

        THIS CHANGES THE ANSWER, AND THAT IS THE POINT.
            Everything else in this engine moves weights without touching them. This does not: a
            different expert runs, so the output differs. It is offered for the same reason
            compression is -- sometimes a small, measured, disclosed loss of quality is worth a
            large gain in speed -- and like compression it is off unless asked for.

        WHAT IT COSTS AND WHAT IT BUYS, MEASURED
            Perplexity against tolerance, two corpora, Qwen3-30B-A3B-3bit:

                            man pages      wikitext2      reads
                    5%        +0.72%         -0.88%        -5%
                   10%        +2.41%         -1.68%       -10%
                   25%        +3.19%         -0.55%       -27%

            The man-pages column is the one to trust: it degrades monotonically with tolerance,
            which is the shape this should have. wikitext2 appearing to IMPROVE, and
            non-monotonically, is corpus luck and not something to rely on. Call it up to ~3.2%.

            What that buys is small: 4.21 -> 4.49 tok/s on Qwen3-30B, 1.07x, because the disk is
            only 27% of a token there. And it buys NOTHING on gpt-oss-120b, which is the model
            that needs it: at 4 experts resident and top-4 routing, every resident expert is
            already chosen, so there is never a free substitute. Zero swaps, measured.

            Rerouting therefore helps least where speed is needed most. Off by default.

        WHAT EXACTLY IS SUBSTITUTED
            Only an expert that is NOT resident, and only for one whose gate probability is at
            least (1 - tolerance) of the original's. The gate weight applied outside this module
            is the one the router computed for the expert it chose, which is left alone; the
            token therefore gets a similar expert's transform at the weight it asked for. That is
            not identical to rerouting at the router itself, and the difference is measurable
            rather than assumed -- see the perplexity numbers in the docs.
        """
        p = mx.softmax(self._gate(x).astype(mx.float32), axis=-1)
        p = np.array(p)
        out = flat.copy()
        resident = np.nonzero(self._pool.g2s >= 0)[0]
        if resident.size == 0:
            return out
        for r in range(out.shape[0]):
            row = p[r] if p.ndim > 1 else p
            chosen = set(int(v) for v in out[r])
            free = [c for c in resident if c not in chosen]
            if not free:
                continue
            for j in range(out.shape[1]):
                e = int(out[r, j])
                if self._pool.g2s[e] >= 0:
                    continue
                best = max(free, key=lambda c: row[c])
                if row[best] >= row[e] * (1.0 - self._reroute):
                    self.reroutes += 1
                    self.reroute_lost += float(row[e] - row[best])
                    out[r, j] = best
                    chosen.discard(e); chosen.add(int(best))
                    free = [c for c in free if c != best]
                    if not free:
                        break
        return out

    def ensure(self, indices, ranks=None) -> np.ndarray:
        """Make every requested expert resident and return the slot-remapped indices.

        `ranks` overrides where the top-k rank is read from. The default reads it off the last
        axis, which is right when `indices` is still (tokens x top_k) and the column index IS the
        rank. The expert-sorted prefill path has already flattened that axis away, so it carries
        the ranks alongside -- without this they would all arrive as 0 and every tuned policy
        would silently be evaluating in its untuned configuration.
        """
        gi = np.asarray(indices) if isinstance(indices, np.ndarray) else \
            np.array(indices, copy=False).astype(np.int64)
        best = {}
        if ranks is not None:
            for e, r in zip(np.asarray(gi).reshape(-1), np.asarray(ranks).reshape(-1)):
                e, r = int(e), int(r)
                if e not in best or r < best[e]:
                    best[e] = r
            flat = gi.reshape(-1, 1)
        else:
            flat = gi.reshape(-1, gi.shape[-1])
            # The router's top-k ordering IS the last axis, so an expert's column index is its
            # rank. Where an expert appears at several ranks, its BEST (lowest) rank is honest.
            for row in flat:
                for r, e in enumerate(row):
                    e = int(e)
                    if e not in best or r < best[e]:
                        best[e] = r
        uniq = sorted(best)
        need = self._pool.touch(uniq, [best[e] for e in uniq])
        if need:
            if len(need) > self._pool.capacity:
                raise RuntimeError(
                    f"layer {self._pool.layer} needs {len(need)} experts for one step but the "
                    f"pool holds {self._pool.capacity}. Capacity must be at least top-k; below "
                    f"that a token evicts an expert it is still using.")
            staged = [int(e) for e in need if int(e) in self._pool.stage]
            cold = [int(e) for e in need if int(e) not in self._pool.stage]
            for e in staged:
                self._pool.admit_arrays(e, self._pool.stage[e], protect=uniq)
                self._pool.stage_hits += 1
            if cold:
                t0 = time.perf_counter()
                blobs = self._fetcher.fetch([(self._pool.layer, e) for e in cold])
                self.fetch_seconds += time.perf_counter() - t0
                for e in cold:
                    b = blobs[(self._pool.layer, e)]
                    self.fetch_bytes += len(b)
                    self._pool.admit(e, b, protect=uniq)
        self._pool.stage = {}                       # whatever was not used was a wrong guess
        return self._pool.g2s[gi]

    def _prefetch_span(self, rows) -> list:
        """Start the disk read for a later chunk while the current one is still on the GPU.

        THIS IS THE ONLY PLACE PREFETCHING IS POSSIBLE, AND THE REASON IS WORTH WRITING DOWN.
        Prefetching across a token boundary needs to know what the NEXT router will select, and
        it has not run yet. Guessing does not work: replaying a 2,271-token trace against a
        44-of-128 pool, the previous token predicted 0.00% of the misses and the previous layer
        5.91%. That is not a weak predictor, it is the definition of a miss -- an expert the
        recent past used is still resident, so it is exactly the experts with no recent history
        that have to be read, and nothing in the history names them.

        Prefill is different. _chunks splits one call into spans, and every span's routing came
        out of the same tensor, so it is already known. Nothing is predicted and nothing can be
        wrong. Two things then overlap that used to be serial: the next span's read runs against
        this span's GPU work, and both spans' reads are in flight together, which matters more --
        measured, a one-expert request gets 2.2 GB/s and an eight-expert request 3.8 GB/s, so the
        engine spends most of its time at the wrong end of its own bandwidth curve.

        Returns the keys it issued so the caller can release any the spans never claimed.
        """
        want = {int(e) for e in np.asarray(rows).reshape(-1)}
        keys = [(self._pool.layer, e) for e in want if self._pool.g2s[e] < 0]
        if keys:
            try:
                self._fetcher.prefetch(keys)
            except RuntimeError:
                # The pending cap is a memory guard, and a prefetch is an optimisation. Declining
                # to speculate is always correct; ensure() reads what it needs either way.
                return []
        return keys

    def _stage_predicted(self) -> None:
        """Copy the experts the previous layer named for this one onto the GPU, if any."""
        ids = self._pool.stage_pred
        self._pool.stage_pred = None
        if not ids:
            return
        keys = [(self._pool.layer, e) for e in ids]
        try:
            blobs = self._fetcher.fetch(keys)       # views of the page cache; the copy is below
        except Exception:                           # noqa: BLE001 -- speculation is optional
            return
        for (layer, e), b in blobs.items():
            self._pool.stage[e] = self._pool.build_arrays(e, b)
            self._pool.stage_issued += 1

    def _speculate(self, predicted) -> None:
        """Start reading what the NEXT layer is likely to want, while this one computes.

        THIS IS OFF BY DEFAULT AND SHOULD STAY OFF. IT NOW MAKES THE MODEL SLOWER.
            It was worth 1.06x while expert reads bypassed the OS page cache. Once they were
            allowed into it the sign flipped, measured over two interleaved A/B pairs on
            Qwen3-30B at 36 of 128:

                prefetch off    8.05, 7.86 tok/s     17% of reads already in hand
                prefetch on     5.82, 5.59 tok/s     35% of reads already in hand

            It still does what it claims -- it doubles the share of reads already in hand -- and
            that is now the problem rather than the point. A guess is right about 65% of the
            time, and every wrong one pulls a page into the cache by pushing a useful one out.
            When a miss meant a trip to the SSD that trade paid; now that a miss is usually a
            memory copy, buying 18 more points of readiness costs 28% of the throughput.

        Only experts that are not already in the next layer's pool are asked for -- predicting an
        expert that is already resident is correct and costs nothing to act on, so acting on it
        would be pure waste. Nothing here can change an answer: `ensure()` still admits exactly
        what the real router selected.
        """
        # NAMED FOR STAGING, NOT READ FROM DISK. The earlier version issued these as page-cache
        # prefetches and made the model slower (measured above): a wrong guess pulled a page in
        # by pushing a useful one out. Now a guess is only ever copied from memory to the GPU,
        # and only during the wait for this layer's routing, when the host has nothing else to
        # do. It touches no disk and evicts nothing. Only experts not already resident are named.
        if not STAGING:
            return
        nxt = self._next
        want = [int(e) for e in predicted if nxt.g2s[int(e)] < 0][:STAGE_MAX]
        # THE BYTE CAP STILL APPLIES TO A MEMCPY. Eight guesses is 14 MB a layer on Qwen3.6 and
        # 106 MB a layer on gpt-oss's 13 MB experts -- 4 ms of copying to hide inside a wait of
        # under one. A guess must never be able to outweigh the work it is speculating about.
        per = getattr(nxt, "bytes_per_expert", 0) or 0
        if per and self._pred_budget:
            want = want[:max(1, self._pred_budget // per)]
        nxt.stage_pred = want or None
        self.pred_issued += len(want)

    @staticmethod
    def _chunks(flat: np.ndarray, C: int):
        """Split token rows so no single gather needs more than C distinct experts.

        THIS IS NOT AN OPTIMISATION, IT IS A CORRECTNESS REQUIREMENT.
        Prefill hands the MoE block every prompt token at once. A 20-token prompt at top-8 can
        touch 64 distinct experts in ONE call, so a 48-slot pool is asked to hold 64 experts
        simultaneously and there is no legal eviction -- the first run died exactly there.
        Decode (one token, k experts) never trips it, so a test that only ever generates would
        have shipped this.
        """
        n = flat.shape[0]
        out, lo, cur = [], 0, set()
        for i in range(n):
            row = set(int(v) for v in flat[i])
            if i > lo and len(cur | row) > C:
                out.append((lo, i))
                lo, cur = i, row
            else:
                cur |= row
        out.append((lo, n))
        return out

    @staticmethod
    def _pair_chunks(sorted_experts: np.ndarray, C: int):
        """Cut a list of (token, expert) pairs, ALREADY SORTED BY EXPERT, into legal gathers.

        WHY THIS EXISTS AND WHAT IT REPLACED
            `_chunks` walks rows in the order the tokens arrived. One row is one token and
            carries top_k distinct experts, so on a pool of 11 slots a chunk is full after about
            one and a half tokens. Instrumented on a 395-token prompt at capacity 11, prefill
            issued 12,982 gathers averaging 12 (token, expert) pairs each -- and 12 is below the
            64 rows that make sorting worth doing, so every one of them also took the slow
            kernel.

            Sorting the pairs by expert first puts every pair that wants expert 40 next to every
            other pair that wants expert 40, so a chunk holds C WHOLE experts instead of C
            experts' worth of one token's scattered routing. On the same prompt that is 1,348
            gathers of 116 pairs -- 9.63x fewer, and every one now over the sorting threshold.

            It also fixes something the chunk count does not show. In token order an expert can
            be admitted, evicted and read again several times within a single prefill step,
            because the pairs that want it are scattered across every chunk. Sorted, an expert
            belongs to exactly one chunk, so it is read at most once per step.

            Legal by construction: the slice from the i-th distinct expert to the (i+C)-th
            contains exactly C distinct experts, which is what the pool can hold.
        """
        uniq, starts = np.unique(sorted_experts, return_index=True)
        n = int(sorted_experts.shape[0])
        return [(int(starts[i]), int(starts[i + C]) if i + C < len(uniq) else n)
                for i in range(0, len(uniq), C)]

    def _forward(self, x, slots):
        """The three expert projections, over whichever slots this span routed to.

        SORTING THE ROWS BY SLOT BEFORE THE GATHER, WHICH IS WHAT MAKES PREFILL FAST.
            `gather_qmm` has two kernels behind it. The general one walks the index list and
            re-reads an expert's weights every time a row asks for it. The other, selected only
            by `sorted_indices=True`, requires the indices to arrive in order and can then keep
            one expert's weights in threadgroup memory while every row routed to it is consumed.
            Whether that is worth anything depends entirely on how many rows share an expert.

            In DECODE it is worth nothing: one token routes to top_k experts, one row each, so
            there is nothing to reuse and the sort is pure overhead. In PREFILL a step of 69
            tokens sends 552 rows at the same 128 experts, so each expert serves four rows on
            average and the reuse is real. This is why the threshold below is on the number of
            rows and not on anything about the model.

            64 is the same threshold mlx_lm's own SwitchGLU uses, kept deliberately: below it
            our arithmetic is the arithmetic upstream runs, and above it our arithmetic is the
            arithmetic upstream runs. Diverging from upstream on the fast path would mean this
            engine's prompt processing rounds differently from every other MLX runtime, which is
            a difference nobody asked for and nobody could audit.

            The sort happens ONCE here rather than inside `_proj`, because all three projections
            index with the same slots -- sorting per projection would pay for it three times and
            unsort twice for nothing.
        """
        x = mx.expand_dims(x, (-2, -3))
        do_sort = slots.size >= SORT_ROWS
        idx, inv = slots, None
        if do_sort:
            x, idx, inv = _gather_sort(x, slots)
        # THE TWO SHAPES ARE DIFFERENT ARITHMETIC, and each mirrors its own upstream block
        # exactly. Gated (SwitchGLU): `down(act(up, gate))`, where the activation is SwiGLU and
        # takes both. Plain (SwitchMLP, which is what Nemotron ships): `fc2(act(fc1))` -- one
        # argument, no multiply, two projections instead of three.
        if self._pool.gated:
            up = self._proj("up_proj", x, idx, do_sort)
            gate = self._proj("gate_proj", x, idx, do_sort)
            out = self._proj("down_proj", self._activation(up, gate), idx, do_sort)
        else:
            h = self._proj("fc1", x, idx, do_sort)
            out = self._proj("fc2", self._activation(h), idx, do_sort)
        if do_sort:
            out = _scatter_unsort(out, inv, slots.shape)
        return out.squeeze(-2)

    def _forward_views(self, xf, flat, arrays_for=None):
        """Every (token, expert) pair of this chunk, computed per expert on a zero-copy view of
        that expert's bytes. Returns rows in the caller's (token, k) order. `arrays_for(e)`
        supplies an expert's arrays; by default they are fetched for this call (prefill), in
        views mode they are the pool's resident views (decode). See VIEWS_PREFILL, VIEWS_DECODE."""
        pool = self._pool
        k = flat.shape[1]
        pe = flat.reshape(-1)
        order = np.argsort(pe, kind="stable")
        spe = pe[order]
        tok = order // k
        experts, starts = np.unique(spe, return_index=True)
        ends = np.append(starts[1:], len(spe))
        if arrays_for is None:
            blobs = self._fetcher.fetch([(pool.layer, int(e)) for e in experts])
            arrays_for = lambda e: pool.build_arrays(int(e), blobs[(pool.layer, int(e))])  # noqa: E731
        comps = [(proj, comp) for proj in pool.projections for comp in pool.spec[proj]]
        outs = []
        for e, lo, hi in zip(experts, starts, ends):
            arrs = dict(zip(comps, arrays_for(int(e))))
            rows = xf[mx.array(tok[lo:hi])]
            def proj(name, inp):
                w = arrs[(name, "weight")]
                if self._quantized:
                    y = mx.quantized_matmul(inp, w, arrs.get((name, "scales")),
                                            arrs.get((name, "biases")), transpose=True,
                                            group_size=self._q["group_size"],
                                            bits=self._q["bits"], mode=self._q["mode"])
                else:
                    y = inp @ w.T
                b = arrs.get((name, "bias"))
                return y if b is None else y + b
            if pool.gated:
                up, gate = proj("up_proj", rows), proj("gate_proj", rows)
                outs.append(proj("down_proj", self._activation(up, gate)))
            else:
                outs.append(proj("fc2", self._activation(proj("fc1", rows))))
        y = mx.concatenate(outs, axis=0)
        inv = np.argsort(order, kind="stable")
        return y[mx.array(inv)]

    def _forward_pairs(self, x, slots, arrays=None):
        """The three projections over (token, expert) pairs already grouped by expert.

        Same arithmetic as `_forward`, minus the sort it would otherwise do and then undo: the
        caller sorted once for the whole step and will unsort once at the end, so paying per
        chunk would be paying for the same reordering a dozen times over.
        """
        xe = mx.expand_dims(x, -2)
        sk = PAIRS_SORTED_KERNEL
        if self._pool.gated:
            up = self._proj("up_proj", xe, slots, sk, arrays)
            gate = self._proj("gate_proj", xe, slots, sk, arrays)
            out = self._proj("down_proj", self._activation(up, gate), slots, sk, arrays)
        else:
            h = self._proj("fc1", xe, slots, sk, arrays)
            out = self._proj("fc2", self._activation(h), slots, sk, arrays)
        return out.squeeze(-2)

    def _forward_views_exact(self, xf, flat, batch: int | None = None):
        """A prefill chunk in views mode, through the SAME sorted gather kernel the pool path uses,
        over the SAME spans.

        WHY NOT THE PER-EXPERT MATMUL. Measured on random data at this model's shapes: with one
        row per expert `gather_qmm` and `quantized_matmul` agree bit for bit (0 of 50 trials
        differ), which is why decode through views is exact. With sixteen rows per expert they do
        not (max |d| 1.0 in bf16). The sorted gather needs one (E, ...) array, so each span's
        experts are stacked -- a temporary copy, the same bytes a pool admit would have copied --
        and run through the pool path's own `_forward_pairs`. The spans are cut by `_pair_chunks`
        at the pool's capacity, exactly as the pool path cuts them, so every gather carries the
        same rows in the same order as it would there: exact by construction, not by argument.
        Costs what the pool path's prefill costs; `--fast-prefill` is the fast, inexact one.
        """
        pool = self._pool
        C = int(batch or pool.capacity)
        k = flat.shape[1]
        pe = flat.reshape(-1)
        order = np.argsort(pe, kind="stable")
        spe = pe[order]
        tok = order // k
        blobs = self._fetcher.fetch([(pool.layer, int(e)) for e in np.unique(spe)])
        comps = [(proj, comp) for proj in pool.projections for comp in pool.spec[proj]]
        outs = []
        for lo, hi in self._pair_chunks(spe, C):
            group = np.unique(spe[lo:hi])
            per = [pool.build_arrays(int(e), blobs[(pool.layer, int(e))]) for e in group]
            stacked = {key: mx.stack([per[j][i] for j in range(len(group))])
                       for i, key in enumerate(comps)}
            arrays = {proj: (stacked[(proj, "weight")], stacked.get((proj, "scales")),
                             stacked.get((proj, "biases")), stacked.get((proj, "bias")))
                      for proj in pool.projections}
            local = np.searchsorted(group, spe[lo:hi]).astype(np.uint32)
            outs.append(self._forward_pairs(xf[mx.array(tok[lo:hi])], mx.array(local), arrays))
        y = mx.concatenate(outs, axis=0)
        inv = np.argsort(order, kind="stable")
        return y[mx.array(inv)]

    def __call__(self, x, indices) -> mx.array:
        if self._sync_free:
            # A LAYER THAT CANNOT MISS DOES NOT NEED THE HOST.
            # Every expert is resident, so the only thing left to do is translate expert ids to
            # slot ids -- one gather, on device. No np.array(indices), so MLX keeps pipelining
            # instead of draining the queue. Measured: each per-layer drain costs about 2 ms on
            # this model, against 0.44 ms of actual GPU work for the whole layer.
            return self._forward(x, mx.take(self._pool.g2s_device(), indices.astype(mx.uint32)))
        if self.bypass:
            # MEASUREMENT ONLY -- OUTPUT IS DELIBERATELY WRONG.
            # Everything stays on device: no host read, so MLX pipelines all 48 layers exactly
            # as the stock model does. Same kernels, same shapes, same bytes; only which expert
            # lands in which slot is scrambled.
            #
            # This replaced a control that did NOT do that. The earlier version only swapped out
            # ensure(), while __call__ still ran np.array(indices) two lines below -- so the
            # "no demand paging" baseline was still synchronising 48 times per token, and every
            # number derived from it understated what demand paging costs.
            return self._forward(x, (indices % self._pool.capacity).astype(mx.uint32))
        # Reading `indices` on the host forces MLX to evaluate everything queued up to this
        # layer. Timed separately because it is the single number that decides whether this
        # design is viable: if the stall dominates the fetch, the bottleneck is the ARCHITECTURE
        # of demand paging, not the disk, and no amount of faster I/O fixes it.
        # PREDICT THE NEXT LAYER'S EXPERTS, AND PAY NOTHING EXTRA TO DO IT.
        #     The host read below is already a synchronisation point -- it drains the queue
        #     whatever else is happening. Queueing the prediction BEFORE it and evaluating both
        #     together means the prediction rides along on a stall that was going to happen
        #     anyway, rather than adding a second one.
        # COPY THE NAMED EXPERTS IN NOW, WHILE THE GPU IS STILL BUSY WITH THIS LAYER'S ATTENTION.
        #     Everything queued before this module -- attention, norms, the router -- has been
        #     committed to the GPU in batches of eight ops and is running. The host would spend
        #     that time waiting at the read below. It spends it here instead, on the memcpy that
        #     would otherwise follow the read on the critical path.
        if STAGING:
            self._stage_predicted()
        pred = None
        if STAGING and self._next is not None and self._pred_m > 0 and (
                self._next_gate is not None or self._pred_w is not None):
            xf = x.reshape(-1, x.shape[-1])[:1]
            if self._next_gate is not None:
                # THE NEXT LAYER'S OWN ROUTER, ONE LAYER EARLY. Measured on Qwen3.6-35B-A3B-4bit
                # over 65 decode steps against the true routing at the next layer:
                #     recall@8  83.6%     recall@16  96.8%     recall@32  99.1%
                # against 47.3 / 59.2 / 68.6% for the ridge map fitted on 546 steps. The residual
                # stream moves little between adjacent layers, so the next router applied to this
                # layer's input mostly agrees with itself applied to its own -- the signal two
                # 2025-26 offloading papers report at 85-90%, confirmed here. Nothing to fit,
                # nothing to download, one 2048x256 matmul a layer that rides the existing drain.
                scores = mx.softmax(self._next_gate(xf), axis=-1, precise=True)[0]
            else:
                scores = (xf @ self._pred_w)[0]
            pred = mx.argpartition(-scores, self._pred_m - 1)[:self._pred_m]
        _t = time.perf_counter()
        # A BLOCKING EVALUATION, ON PURPOSE. Starting the drain in the non-blocking form and
        # staging inside the wait read well and measured badly: with nothing at all to stage,
        # the non-blocking form alone was 79.5 ms/token against 51.8 for this line, interleaved
        # in one process. The staging copy happens at the top of __call__ instead, while the GPU
        # is still on this layer's attention and router -- work committed before this module
        # was entered.
        if pred is not None:
            mx.eval(indices, pred)              # one drain, two answers
        gi = np.array(indices, copy=False).astype(np.int64)
        self.sync_seconds += time.perf_counter() - _t
        if pred is not None:
            self._speculate(np.array(pred))
        k = gi.shape[-1]
        C = self._pool.capacity
        if k > C:
            raise ValueError(
                f"layer {self._pool.layer}: top-k is {k} but the pool holds {C}. A single token "
                f"cannot be served, so no capacity below top-k is ever valid.")
        flat = gi.reshape(-1, k)
        if self._reroute > 0.0 and self._gate is not None:
            # Applied BEFORE _chunks, so a substitution can also relieve the split it would
            # otherwise have caused.
            flat = self._reroute_to_resident(flat, x.reshape(-1, x.shape[-1]))
            gi = flat.reshape(gi.shape)
        # WHETHER THIS CALL NEEDS SPLITTING AT ALL, ANSWERED WITHOUT BUILDING THE SPLIT.
        #     `_chunks` is a Python loop over every row that builds a set per row. Asking it and
        #     then discarding the answer -- which is what the expert-sorted path below does, since
        #     it cuts its own spans -- put that loop on the critical path of every prefill call
        #     for nothing. One row's worth of distinct experts fitting in the pool is the whole
        #     question, and `np.unique` answers it in C.
        if len(np.unique(flat)) <= C:
            # ISSUING THE RESIDENT EXPERTS' ROWS BEFORE COPYING THE MISSED ONES IN DOES NOT PAY.
            #     The rows of a gather_qmm are independent -- measured bit-for-bit, one gather
            #     over eight slots equals five-then-three equals eight singles -- so the hits
            #     could be issued the instant the routing is known and the misses after their
            #     bytes land, and the GPU would work while the CPU copied. Built and A/B'd on
            #     Qwen3.6-35B-A3B-4bit, greedy, 128 tokens, 547 characters identical both ways:
            #
            #         hits first off    57.6 ms/token   sync 38.7   admit 16.5
            #         hits first on     72.6 ms/token   sync 57.3   admit 11.8     0.79x
            #
            #     The copy did move off the critical path (admit -5 ms) and the sync grew by 19:
            #     two gathers, a concatenate and a take per layer are five more ops in a queue
            #     that the next layer's host read has to drain, and that cost more than the
            #     overlap bought. Same shape of lesson as the per-slot arrays above. Not built
            #     a second time.
            if self._pool.views:
                # VIEWS MODE: residency is the pool's business as before; the arithmetic runs on
                # the resident views, one expert at a time. See VIEWS_DECODE.
                self.ensure(gi)
                y = self._forward_views(x.reshape(-1, x.shape[-1]), flat, self._pool.views_of)
                return y.reshape(*gi.shape[:-1], k, y.shape[-1])
            # Unchunked path: byte-for-byte the shape handling of the stock SwitchGLU, so the
            # C == E identity test exercises exactly the code the fast path runs.
            return self._forward(x, mx.array(self.ensure(gi)))
        self.chunked_calls += 1
        xf = x.reshape(-1, x.shape[-1])
        if VIEWS_PREFILL and (flat.shape[0] >= VIEWS_MIN_TOKENS
                              and len(np.unique(flat)) >= VIEWS_MIN_SHARE * self._pool.n_experts):
            self.views_calls += 1
            y = self._forward_views(xf, flat)          # fast, and not the pool path's arithmetic
            return y.reshape(*gi.shape[:-1], k, y.shape[-1])
        if self._pool.views:
            self.views_calls += 1
            y = self._forward_views_exact(xf, flat)    # exact: the pool path's own kernel
            return y.reshape(*gi.shape[:-1], k, y.shape[-1])
        outs, issued = [], []
        if EXPERT_SORTED_PREFILL:
            # GROUP THE WORK BY EXPERT INSTEAD OF BY TOKEN. See _pair_chunks for the measurements.
            # A (token, expert) pair is an independent row -- SwitchGLU returns them unreduced and
            # the router's weights are applied by the caller -- so they may be computed in any
            # order provided they are put back in the caller's order, which `inv` does.
            pe = flat.reshape(-1)
            order = np.argsort(pe, kind="stable")
            spe = pe[order]
            tok = order // k                      # the token each pair belongs to
            rank = order % k                      # and its top-k rank, kept for the policy
            pspans = self._pair_chunks(spe, C)
            for j, (lo, hi) in enumerate(pspans):
                # How far ahead to start reading. See PREFILL_PREFETCH_DEPTH. Zero means none
                # at all -- this used to say max(1, depth), which quietly made "off" the same
                # thing as "one ahead" and turned the whole sweep into a comparison of 1 against
                # 1, 2 and 4. It read as a clean null result and was an error in the harness.
                if PREFILL_PREFETCH_DEPTH > 0:
                    if j == 0:
                        for d in range(1, min(PREFILL_PREFETCH_DEPTH, len(pspans))):
                            issued += self._prefetch_span(spe[pspans[d][0]:pspans[d][1]])
                    nd = j + PREFILL_PREFETCH_DEPTH
                    if nd < len(pspans):
                        nlo, nhi = pspans[nd]
                        issued += self._prefetch_span(spe[nlo:nhi])
                sl = self.ensure(spe[lo:hi], ranks=rank[lo:hi])
                outs.append(self._forward_pairs(xf[mx.array(tok[lo:hi])],
                                                mx.array(sl.reshape(-1))))
            if issued:
                self._fetcher.drop(issued)
            y = mx.concatenate(outs, axis=0)
            inv = np.argsort(order, kind="stable")
            y = y[mx.array(inv)]
            return y.reshape(*gi.shape[:-1], k, y.shape[-1])
        spans = self._chunks(flat, C)
        for j, (lo, hi) in enumerate(spans):
            if j + 1 < len(spans):
                nlo, nhi = spans[j + 1]
                issued += self._prefetch_span(flat[nlo:nhi])
            # ensure() is unchanged, and so is the order it admits and evicts in. A prefetched
            # key is collected by fetch() rather than read twice, so the bytes and the slot
            # assignments are identical to the serial path -- which is what keeps this bit-exact.
            outs.append(self._forward(xf[lo:hi], mx.array(self.ensure(flat[lo:hi]))))
        if issued:
            # Anything a later span turned out not to need is still holding its buffer.
            self._fetcher.drop(issued)
        y = mx.concatenate(outs, axis=0)
        return y.reshape(*gi.shape[:-1], k, y.shape[-1])

    def refresh(self):
        """Re-decide whether this layer can skip the host round-trip. Cheap; call after warming.

        THE ROUND-TRIP IS NOW 73% OF EVERYTHING, AND SKIPPING IT FINALLY PAYS.
            Profiled with reads memory-mapped, over 42 tokens: `numpy.array` -- which is the
            single `np.array(indices)` each streamed layer does to read its routing back to the
            host -- accounted for 4.488 s of a 6.1 s run across 2,016 calls. That is 2.23 ms a
            layer, 107 ms of a 127 ms token. Not the disk (1.7 ms), not the arithmetic.

            A layer holding every expert cannot miss, so it translates expert ids to slots on
            device and never pays it. It costs 128 slots, which starves every other layer -- and
            that used to lose, because a miss cost 0.665 ms. A miss now costs 0.014 ms, so the
            cost side of the trade has collapsed. Measured at IDENTICAL total memory:

                 0 layers full, rest at 36 slots   12.78 tok/s   3.57 GB
                 6 layers full, rest at 22 slots   13.78 tok/s   3.49 GB   1.08x
                10 layers full, rest at 11 slots   14.48 tok/s   3.51 GB   1.13x

            Monotone across three points at the same footprint. The spread on this machine is
            wide enough that 1.13x is near the edge of what can be resolved, so it wants more
            repeats before a planner is built on it -- but the direction is not in doubt, and
            `full_layers` has been a parameter of this loader since the beginning.
        """
        self._sync_free = self._pool.full
        return self._sync_free

    def _proj(self, name, x, slots, sorted_indices: bool = False, arrays=None):
        """One projection, on the GPU.

        RUNNING SOME EXPERTS ON THE CPU DOES NOT WORK HERE, AND IT IS NOT CLOSE.
            ktransformers and Fiddler split MoE work across CPU and GPU, and on Apple Silicon
            the memory is shared so there is no transfer to pay for -- the CPU simply sits idle
            for the whole of the GPU's 20 ms. Measured on this model's real shapes:

                all 8 experts on the GPU      0.50 ms
                all 8 experts on the CPU     27.63 ms      55x slower
                6 on GPU and 2 on CPU         6.92 ms      0.07x -- fourteen times worse

            MLX's CPU path for a 3-bit quantised gather is not vectorised, so even two experts
            on the CPU makes the whole step wait for it. The technique works on x86 with CUDA,
            where the host has AVX-512 and the weights are usually fp16. It does not translate.
        """
        w, sc, bi, b = arrays[name] if arrays is not None else self._pool.arrays(name)
        if self._quantized:
            y = mx.gather_qmm(x, w, sc, bi, rhs_indices=slots, transpose=True,
                              group_size=self._q["group_size"], bits=self._q["bits"],
                              mode=self._q["mode"], sorted_indices=sorted_indices)
        else:
            y = mx.gather_mm(x, w.swapaxes(-1, -2), rhs_indices=slots,
                             sorted_indices=sorted_indices)
        if b is not None:
            y = y + mx.expand_dims(b[slots], -2)
        return y


# --------------------------------------------------------------------------- wiring it up
# How many expert-rows a gather must carry before sorting them by expert pays for itself.
# mlx_lm's SwitchGLU uses the same number and this deliberately matches it -- see _forward.
SORT_ROWS = 64

# Group prefill work by expert rather than by token -- see _pair_chunks. Switchable so the two
# paths can be measured against each other on the same process.
EXPERT_SORTED_PREFILL = True
# THE FILE AS THE POOL, FOR PREFILL: NO SLOT, NO COPY, THE GPU READS THE PAGE CACHE.
#     A prefill chunk routes to nearly every expert of a layer, and the shipped path admits each
#     one into a pool slot -- a GPU copy of 1.77 MB -- before the gather can run: about 18 GB of
#     copies per 512-token chunk, which is what a chunk costs (measured 5.1 s, disk idle). This
#     path skips the slot. Each expert's bytes are wrapped as a zero-copy view of the page cache
#     and the rows routed to it go through `quantized_matmul` on that view directly; the views
#     live for this layer only, so Metal wires 453 MB at a time and releases it. Same weights,
#     same arithmetic per row. Whether it pays is a measurement, not an argument; it is OFF
#     unless BIGRIG_VIEWS_PREFILL=1 until it has.
VIEWS_PREFILL = os.environ.get("BIGRIG_VIEWS_PREFILL", "0") == "1"
# THE FILE AS THE POOL, FOR DECODE: RESIDENT EXPERTS ARE LIVE VIEWS, NOT COPIES.
#     In views mode a streamed layer's pool holds no slot tensors. Admitting an expert wraps its
#     page-cache bytes as a zero-copy view and keeps the view alive; Metal wires those pages while
#     the view lives and releases them when it is dropped, so eviction is a `del`. There is no
#     1.77 MB copy per miss and, more to the point, no second copy of the hot experts sitting in
#     anonymous memory beside the page cache that already holds them -- on this Mac that was
#     2.6 GB the cache could have used. Each routed expert's rows run through `quantized_matmul`
#     on its view (1.5 ms a token for 36 layers x 8 x 3, measured, against 1.0 for the gathers).
#     The policy, the accounting and the ceiling are the pool's own; only what a slot IS changes.
#     What MLX's own memory counter cannot see -- wired file pages -- `calibrate.phys_footprint_gb`
#     can, and the session's footprint adds the resident bytes in this mode. OFF unless
#     BIGRIG_VIEWS_DECODE=1 until its identity and speed are measured.
VIEWS_DECODE = os.environ.get("BIGRIG_VIEWS_DECODE", "0") == "1"
VIEWS_MIN_TOKENS = 32
VIEWS_MIN_SHARE = 0.5
# Most experts a layer will copy to the GPU on the previous layer's say-so. Eight is the
# router's top-k: enough to catch what recall@8 catches, small enough (8 x 1.77 MB on Qwen3.6)
# to fit inside the wait it hides in.
STAGE_MAX = 8
# Wrap page-cache pages as GPU buffers instead of copying them. See ExpertPool.build_arrays.
ZERO_COPY = os.environ.get("BIGRIG_ZERO_COPY", "1") != "0"
PAGE_BYTES = int(os.sysconf("SC_PAGESIZE")) if hasattr(os, "sysconf") else 16384
# OFF BY DEFAULT. BUILT, CORRECT, AND MEASURED NOT TO PAY ON THE MODEL IT WAS BUILT FOR.
#     Interleaved before/after in one process, greedy, 128 tokens, all texts identical:
#
#         prose   before 51.8 / 53.6 ms      after 63.6 / 57.1 ms      median 0.84x
#         code    before 77.2 / 53.0 ms      after 78.5 / 73.4 ms      median 0.98x
#
#     The copy does leave the critical path -- admit fell from ~30 to ~7 ms a token and 66 of
#     ~150 misses arrived staged -- but the staging memcpy runs while the GPU is streaming ~23 MB
#     of expert weights a layer over the same unified memory, and at 47% recall more than half
#     of what is staged is never used. The contention costs more than the overlap saves. A first
#     A/B said 1.28x; its baseline was the mx.async_eval form with nothing to stage, which is
#     itself 79.5 against 51.8 ms, so that number measured the harm of async_eval, not the good
#     of staging. Separate processes disagree by up to 30% on the same configuration, which is
#     the page cache; only interleaved runs in one process were believed here.
#
#     RE-MEASURED with a far sharper predictor (the next layer's own router, recall@8 83.6%) and
#     with zero-copy admits, so there was no memcpy left to contend for bandwidth. Same
#     interleaved discipline, texts identical:
#
#         prose   off 51.6 / 49.3 ms     on 60.2 / 51.9 ms
#         code    off 62.1 / 49.2 ms     on 52.8 / 52.4 ms
#
#     On like-for-like warm runs it is 0.94-0.95x; it wins only when the OFF run happened to be
#     cold, and zero-copy already removed most of the cold penalty. 77 of 150 misses arrived
#     staged and admit fell 12 -> 7 ms, but the wrapper creation, the extra matmul and the
#     Python around them cost more than that. Still off. The router predictor stays wired: it
#     is free, needs no fitted file, and is the right signal if staging is ever worth it.
#     BIGRIG_STAGE=1 turns it on.
STAGING = os.environ.get("BIGRIG_STAGE", "0") == "1"

# WHETHER THE EXPERT-MAJOR SPANS ALSO ASK FOR THE SORTED GATHER KERNEL.
#     They are always genuinely sorted, so this is only ever a question of speed. mlx_lm's own
#     heuristic says do not bother under 64 rows, and a span at capacity 11 carries about 46, so
#     the expectation was that it would not pay here. Measured on a 395-token prompt, three runs
#     each, best of:
#
#         sorted kernel on   6.38s to first token       off   7.25s       1.14x
#
#     and the reply is byte-identical either way, which it has to be for a change that only
#     schedules the same arithmetic differently. Kept on. The threshold upstream uses is tuned
#     for a resident model choosing whether to pay for a SORT; these rows arrive sorted already,
#     so the only thing left to weigh is the kernel, and the kernel wins earlier than the sort
#     does.
PAIRS_SORTED_KERNEL = True

# HOW MANY EXPERT-MAJOR SPANS AHEAD PREFILL STARTS READING. ZERO, AND THAT IS A MEASUREMENT.
#     Prefetching during prefill is the one variant with no guesswork in it: every span's routing
#     came out of the same tensor, so what a later span needs is already known and nothing can be
#     mispredicted. It was on, at a depth of one -- classic double buffering, the next span's disk
#     read overlapping this span's GPU work.
#
#     Once prefill was regrouped by expert it stopped paying, and then started costing. Best of
#     three runs on a 395-token prompt, time to first token:
#
#         depth        0        1        2        4
#         warm      4.89s    4.85s    4.87s    4.91s        page cache on
#         cold     10.87s   11.14s   11.47s   11.75s        page cache bypassed
#
#     Monotone in the cold column across four points, which is the column that should have shown
#     the benefit. The reason is that the regrouping already did the job: `ensure()` collects
#     every miss in a span and issues them as ONE fetch across the thread pool, and an expert now
#     belongs to exactly one span so it is read once. There is no serial read left to hide. What
#     a prefetch adds is competition -- the same threads and the same buffer budget, spent on
#     experts nothing is waiting for yet, while the span that is actually blocking waits longer.
#
#     Kept as a parameter rather than deleted, because the trade turns on read bandwidth and this
#     was measured on one machine's SSD. On a slower disk the overlap could pay again.
#
#     AN EARLIER VERSION OF THIS SWEEP WAS WRONG AND READ AS A CLEAN NULL. The depth was applied
#     as max(1, depth), so "off" was really "one ahead" and the table compared 1 against 1, 2 and
#     4. It was caught by counting the reads each setting actually issued, which is worth doing
#     to any result that comes out flat.
PREFILL_PREFETCH_DEPTH = 0


def _gather_sort(x, indices):
    """Reorder rows so every row routed to one expert is contiguous. Returns (x, idx, inverse).

    `order // M` maps a position in the flattened (tokens x top_k) index list back to the token
    it came from, so a token's row is replicated once per expert it routed to.
    """
    *_, M = indices.shape
    indices = indices.flatten()
    order = mx.argsort(indices)
    return x.flatten(0, -3)[order // M], indices[order], mx.argsort(order)


def _scatter_unsort(x, inv_order, shape=None):
    x = x[inv_order]
    return x if shape is None else mx.unflatten(x, 0, shape)


def _quant_params(sg) -> tuple:
    mod = first_projection(sg)
    if "scales" in mod:
        return True, {"group_size": int(getattr(mod, "group_size", 64)),
                      "bits": int(getattr(mod, "bits", 4)),
                      "mode": getattr(mod, "mode", "affine")}
    return False, {}


def attach(model, blob_path: str, capacity, policy: str = "lfuda", threads: int = 8,
           predictors: dict | None = None, reroute: float = 0.0,
           nocache: bool = False, warm: bool = True, verbose: bool = True, full_layers=(),
           manifest: dict | None = None):
    """Replace every SwitchGLU in `model` with a pool-backed one. Returns a handle.

    `capacity` is per layer: an int, or a float in (0, 1] read as a fraction of the expert count.
    """
    manifest = manifest if manifest is not None else load_manifest(blob_path)
    store = store_from_manifest(blob_path, manifest)
    fetcher = ParallelFetcher(store, threads=threads, nocache=nocache,
                              mapped=os.environ.get('BIGRIG_MMAP', '1')
                              not in ('0', 'false', 'no'))
    sites = find_moe_sites(model)
    layers = [(i, sg) for i, _b, _a, sg in sites]
    _owner = {i: b for i, b, _a, _sg in sites}
    _attr = {i: a for i, _b, a, _sg in sites}
    # `full` names layers left entirely alone: stock kernel, mmap-backed, no engine, no sync.
    # They cost E slots of memory and zero milliseconds of overhead.
    full = set(full_layers or ())
    pools, mods = [], []
    for li, sg in layers:
        info = manifest["layers"][str(li)]
        E = info["n_experts"]
        if li in full:
            C = E
        else:
            C = int(round(capacity * E)) if isinstance(capacity, float) and capacity <= 1.0 \
                else int(capacity)
        C = max(1, min(E, C))
        quant, qp = _quant_params(sg)
        mq = info.get("quant")
        if mq:
            # The blob is the authority. A requantised blob has different bits from the module
            # it was derived from, and trusting the module would silently decode it wrong.
            quant, qp = True, {"group_size": int(mq["group_size"]), "bits": int(mq["bits"]),
                               "mode": mq.get("mode", "affine")}
        pool = ExpertPool(li, sg, E, C, info["spec"], policy=policy,
                          quant=info.get("quant"), views=bool(VIEWS_DECODE and C < E)).build()
        m = StreamingSwitchGLU(pool, fetcher, sg.activation, quant, qp)
        # Put it back exactly where it was found. Hardcoding `.mlp.switch_mlp` here would have
        # raised on gpt-oss rather than failing silently, but only after find_moe_sites had
        # already located the block under a different name.
        setattr(_owner[li], _attr[li], m)
        pools.append(pool)
        mods.append(m)
    # WIRE THE PREDICTORS LAST, WHEN EVERY POOL EXISTS.
    #     Layer L's predictor names layer L+1's experts, so it needs a reference to the pool it
    #     is predicting about -- which does not exist until every layer has been built. A layer
    #     with no next layer gets none: there is nothing to prefetch for.
    # EVERY STREAMED LAYER KNOWS THE NEXT ONE'S ROUTER. Wired unconditionally: it costs one small
    # matmul a layer and only matters when staging is on, but it needs no fitted file, so a model
    # never has to be 'predicted' before it can benefit. `layers` and `mods` are in the same
    # order, so mods[j]'s next streamed layer is layers[j + 1].
    per_layer_budget = max(1, SPECULATION_BUDGET_BYTES // max(1, len(mods)))
    for j, mod in enumerate(mods):
        if j + 1 >= len(mods):
            continue
        nxt_block = _owner[layers[j + 1][0]]
        gate = getattr(nxt_block, "gate", None)
        if gate is None or not callable(gate):
            continue
        mod._next_gate = gate
        mod._next = pools[j + 1]
        mod._pred_m = STAGE_MAX
        mod._pred_budget = per_layer_budget
    if predictors is not None:
        Ws = predictors.get("W") or []
        m_pred = int(predictors.get("width") or 0)
        # Outstanding speculation is capped in BYTES and shared across layers, so a model with
        # large experts cannot quietly queue gigabytes of guesses. See _speculate.
        total_budget = int(predictors.get("budget_bytes") or SPECULATION_BUDGET_BYTES)
        per_layer_budget = max(1, total_budget // max(1, len(mods)))
        wired = 0
        for i, mod in enumerate(mods):
            if i + 1 >= len(mods) or i >= len(Ws) or m_pred <= 0:
                continue
            W = np.asarray(Ws[i])
            if W.size == 0 or not np.any(W):
                continue
            mod._pred_w = mx.array(W.astype(np.float16))
            mod._pred_m = m_pred
            mod._next = pools[i + 1]
            mod._pred_budget = per_layer_budget
            wired += 1
        if verbose and wired:
            print(f"  prefetch predictor on {wired} layers, naming {m_pred} experts a layer "
                  f"(recall@{m_pred} was {predictors.get('meta', {}).get('recall_at_8', 0):.0%} "
                  f"at 8 when it was fitted)")

    if reroute and reroute > 0.0:
        for i, mod in enumerate(mods):
            g = getattr(_owner.get(i), "gate", None)
            if g is not None:
                mod._gate = g
                mod._reroute = float(reroute)
        if verbose:
            print(f"  cache-aware rerouting ON at {reroute:.0%} tolerance -- a token whose "
                  f"expert is not\n  in memory may be sent to a resident one that scored nearly "
                  f"as well. THE OUTPUT CHANGES.")

    h = StreamHandle(model, pools, mods, fetcher, manifest, full_layers=sorted(full))
    h.warm(warm if warm is not True else "auto")
    for m in mods:
        m.refresh()
    if verbose:
        sf = sum(1 for m in mods if m._sync_free)
        streamed = [p for p in pools if p.capacity < p.n_experts]
        cap = streamed[0].capacity if streamed else (pools[0].capacity if pools else 0)
        ne = pools[0].n_experts if pools else 0
        print(f"  {len(streamed)} streamed layers at {cap}/{ne}, {sf} sync-free, "
              f"pool {h.resident_gb():.2f} GB of {manifest['total_bytes']/1e9:.2f} GB on disk")
    return h


class StreamHandle:
    """Everything the caller needs to warm, measure, and tear down a streamed model."""

    def __init__(self, model, pools, mods, fetcher, manifest, full_layers=()):
        self.model, self.pools, self.mods = model, pools, mods
        self.fetcher, self.manifest = fetcher, manifest
        self.full_layers = list(full_layers)

    def warm(self, mode="auto"):
        """Fill the pools before generation, or deliberately do not.

        WHY THIS IS NOT ALWAYS WORTH DOING
            Prefilling reads C experts per layer. In compress mode C == E, so those bytes have to
            be read regardless -- prefilling is just the model load, and it also switches on the
            sync-free path, which needs every expert present.

            In STREAM mode C < E, and prefilling is a guess about which experts the prompt will
            want. Measured on Qwen3-30B at 60% residency it reads 7.6 GB before the first token,
            while the first token itself needs only top-k per layer -- about 0.8 GB. The guess
            costs seconds of dead time to save misses that the cache would have filled anyway.

        `mode`: "auto" prefills only when C == E, "full" always, "lazy" never.
        """
        if mode not in ("auto", "full", "lazy", True, False):
            raise ValueError(f"warm mode must be auto/full/lazy, got {mode!r}")
        if mode is True:
            mode = "full"
        if mode is False:
            mode = "lazy"
        if mode == "lazy":
            return self
        CHUNK = 16     # experts of raw bytes held at once; a whole layer at E=256 is GBs
        for p in self.pools:
            if mode == "auto" and p.capacity < p.n_experts:
                p.hits = p.misses = 0
                continue
            for lo in range(0, p.capacity, CHUNK):
                hi = min(lo + CHUNK, p.capacity)
                blobs = self.fetcher.fetch([(p.layer, e) for e in range(lo, hi)])
                for e in range(lo, hi):
                    p.touch([e], [0])
                    p.admit(e, blobs.pop((p.layer, e)))
                del blobs
            p.hits = p.misses = 0          # warming is setup, not workload
            mx.clear_cache()
        return self

    def usage(self) -> dict:
        """{"layer:expert": uses} for every streamed expert asked for at least once."""
        out = {}
        for m in self.mods:
            p = m._pool
            for e in np.nonzero(p.uses)[0]:
                out[f"{p.layer}:{int(e)}"] = int(p.uses[e])
        return out

    def save_usage(self, path: str, decay: float = 0.5) -> str:
        """Merge this run's use counts into what earlier runs saw, and write it.

        Earlier counts are halved before the merge so the file tracks what this model has been
        used for lately rather than forever; a file that never forgot would warm for the first
        week's prompts for the rest of its life.
        """
        prev = {}
        try:
            with open(path) as fh:
                prev = json.load(fh) or {}
        except (OSError, ValueError):
            prev = {}
        merged = {k: int(v * decay) for k, v in prev.items() if int(v * decay) > 0}
        for k, v in self.usage().items():
            merged[k] = merged.get(k, 0) + v
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(merged, fh)
        os.replace(tmp, path)
        return path

    def resident_gb(self) -> float:
        """Every expert byte held in RAM, pooled or not.

        Full-residency layers are not in `pools` but their weights are absolutely still in
        memory. Omitting them would under-report the footprint by exactly the amount the
        planner just chose to spend, which is the one number a user sizing a machine relies on.
        """
        # Full-residency layers now live in `pools` at C == E, so summing the pools is the
        # whole footprint. Adding self.full_layers again would double-count them.
        return sum(p.nbytes() for p in self.pools) / 1e9

    def stats(self) -> dict:
        h = sum(p.hits for p in self.pools)
        m = sum(p.misses for p in self.pools)
        return {"layers": len(self.pools),
                "full_layers": sum(1 for p in self.pools if p.capacity == p.n_experts),
                "sync_free": sum(1 for m in self.mods if m._sync_free),
                "hits": h, "misses": m,
                "miss_rate": m / (h + m) if (h + m) else 0.0,
                # THE STREAMED CAPACITY, NOT POOL ZERO'S.
                #     Layers held whole have capacity == n_experts, and `full_layers` puts them
                #     first, so pools[0] reported 128 for a pool that was really 11 nearly
                #     everywhere. Session copies this into plan["capacity"], and on the next
                #     reload the planner re-planned from 128 and asked for 37 whole layers --
                #     several times the memory it had. That is what took the server down.
                "capacity": (min((p.capacity for p in self.pools
                                  if p.capacity < p.n_experts), default=
                                 (self.pools[0].capacity if self.pools else 0))),
                "full_capacity": self.pools[0].capacity if self.pools else 0,
                "n_experts": self.pools[0].n_experts if self.pools else 0,
                "resident_gb": self.resident_gb(),
                "disk_gb": self.manifest["total_bytes"] / 1e9,
                "fetch_seconds": sum(m_.fetch_seconds for m_ in self.mods),
                "admit_seconds": sum(m_._pool.admit_seconds for m_ in self.mods),
                "stage_issued": sum(m_._pool.stage_issued for m_ in self.mods),
                "stage_hits": sum(m_._pool.stage_hits for m_ in self.mods),
                "zero_copy_admits": sum(m_._pool.zero_copy_admits for m_ in self.mods),
                "views": any(m_._pool.views for m_ in self.mods),
                "fetch_gb": sum(m_.fetch_bytes for m_ in self.mods) / 1e9,
                "chunked_calls": sum(m_.chunked_calls for m_ in self.mods),
                "sync_seconds": sum(m_.sync_seconds for m_ in self.mods),
             "reroutes": sum(m_.reroutes for m_ in self.mods),
             "reroute_lost": round(sum(m_.reroute_lost for m_ in self.mods), 4),
             # WHETHER SPECULATION ACTUALLY LANDS. The fetcher has counted this from the start
             # and nothing ever read it, so prefetch shipped for weeks with no way to tell a
             # prediction that arrived in time from one that arrived too late or was simply
             # wrong. `pred_used` next to it was declared and never incremented, which is worse
             # than no counter: it reported zero uses whatever happened.
             "spec_ready": getattr(self.fetcher, "hit_done", 0),
             "spec_late": getattr(self.fetcher, "hit_pending", 0),
             "spec_none": getattr(self.fetcher, "cold", 0)}

    def close(self) -> None:
        """Release every slot array this handle owns, whatever still holds the handle.

        WHY REFCOUNTING WAS NOT ENOUGH
            Reloading a model to change its residency drops the old session and builds a new one.
            Dropping it works in isolation -- the pool frees and MLX active memory returns to
            zero -- and did not work inside the server: after `del old` the object was still
            alive with no referrer that `gc.get_referrers` could name, so both pools existed at
            once. A 40 -> 53 reload reported 10.56 GB for 5.25 GB of weights, and because the
            footprint feeds the reply ceiling, every reply after a reload was capped at the
            256-token floor.

            Chasing the holder is the wrong fix in any case. The arrays are the memory; freeing
            them explicitly makes the reload safe whether or not something is still pointing at
            the object that used to own them.
        """
        for pool in getattr(self, "pools", []) or []:
            try:
                pool._slots.clear()
                pool._built = False
            except AttributeError:
                pass
        for m in getattr(self, "mods", []) or []:
            m._pool = None
            m._fetcher = None
        self.pools = []
        self.mods = []
        try:
            self.fetcher.close()          # stop the reader threads too
        except Exception:                 # noqa: BLE001 -- teardown must not raise
            pass
        mx.clear_cache()

    def reset_stats(self):
        for p in self.pools:
            p.hits = p.misses = 0
        for m in self.mods:
            m.fetch_seconds = 0.0
            m._pool.admit_seconds = 0.0
            m.fetch_bytes = 0
        return self


def usage_path(model_name: str) -> str:
    """Where a model's expert use counts live, beside its knee."""
    safe = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in model_name)
    return os.path.join(home(), "data", "results", f"usage_{safe}.json")


def hot_regions(manifest: dict, usage: dict, limit_bytes: int) -> list:
    """(offset, length) of the most-used experts, hottest first, up to `limit_bytes` of them.
    Only experts the manifest knows; a usage file from a model that changed shape is ignored."""
    regions = manifest.get("regions") or {}
    ranked = sorted(((int(v), k) for k, v in usage.items() if k in regions), reverse=True)
    out, total = [], 0
    for _uses, key in ranked:
        off, n = regions[key]
        if total + int(n) > limit_bytes:
            break
        out.append((int(off), int(n)))
        total += int(n)
    return out


# THE HOT SET, READ BEFORE THE FIRST PROMPT RATHER THAN AFTER IT.
#     warm_page_cache runs in the background after the port opens, on the reasoning that reading
#     a whole 12.68 GB model takes three seconds and no request should wait for that. Right, and
#     incomplete: routing is skewed, not flat. On Qwen3.6's own usage record the most-used 10% of
#     experts carry 47% of all uses, and that 10% is 1.8 GB -- under half a second at disk speed.
#     Reading THAT set synchronously costs a fraction of a second at startup.
#
#     WHAT IT BUYS, MEASURED, AND WHAT IT DOES NOT. On a page cache churned by streaming another
#     16.5 GB model through it, Qwen3.6's first request:
#
#                          first token     decode
#         no warm          2.91 / 2.64 s   11.0 / 10.6 tok/s
#         2 GB in 0.37 s   2.67 / 2.27 s   10.6 / 10.2 tok/s
#         4 GB in 0.91 s        2.27 s          10.2 tok/s
#
#     About 10-12% off the first token, and nothing for decode -- and the reason is the pool
#     itself. LFUDA already keeps the hottest experts in slots, so the set warmed here is largely
#     the set already resident, and a decode miss is by definition an expert OUTSIDE it: the cold
#     tail, which a 2 GB or a 4 GB warm barely reaches. Prefill is different: a prompt touches
#     nearly every expert, so the hot pages it finds in memory are pages it would otherwise have
#     read from disk. This is a prefill win, priced honestly, and 2 GB is where it stops paying.
#     Bounded twice: by bytes, so it never becomes the three-second read this replaces, and by
#     wall clock, so a slow disk cannot turn "sub-second" into a stall at startup.
HOT_WARM_MAX_GB = 2.0
HOT_WARM_MAX_S = 1.5


def warm_hot_set(model_dir: str, model_name: str, max_gb: float | None = None,
                 max_seconds: float | None = None, spare_gb: float | None = None) -> dict:
    """Read this model's most-used experts into the page cache, and return what was read.

    Returns a dict with `bytes`, `seconds`, `experts` and, when nothing was read, `stopped`
    saying why in a sentence. Never raises: a warm that fails is a warm that did not happen, and
    the first prompt reads from disk exactly as it would have anyway.
    """
    import json as _json
    import time as _t
    out = {"bytes": 0, "seconds": 0.0, "experts": 0, "stopped": ""}
    try:
        man, blob = expert_source(model_dir)
    except Exception as e:                       # noqa: BLE001 -- warming is never worth an exception
        out["stopped"] = f"no expert source: {type(e).__name__}"
        return out
    if not blob:
        out["stopped"] = "no packed blob; the hot set is read from the model's own files as needed"
        return out
    try:
        with open(usage_path(model_name)) as fh:
            usage = _json.load(fh) or {}
    except (OSError, ValueError):
        out["stopped"] = "no usage record yet; the first run writes one"
        return out
    cap_gb = HOT_WARM_MAX_GB if max_gb is None else float(max_gb)
    if spare_gb is not None:
        # Never take what the machine does not have. The same rule the background warm uses.
        cap_gb = min(cap_gb, max(0.0, float(spare_gb) - 1.0))
    if cap_gb <= 0:
        out["stopped"] = "not enough spare memory to warm anything"
        return out
    hot = hot_regions(man, usage, int(cap_gb * 1e9))
    if not hot:
        out["stopped"] = "usage record names no experts this manifest knows"
        return out
    total = sum(int(n) for _, n in hot)
    deadline = _t.perf_counter() + (HOT_WARM_MAX_S if max_seconds is None else float(max_seconds))
    res = warm_page_cache(blob, budget_bytes=total,
                          should_stop=lambda: _t.perf_counter() > deadline, regions=hot)
    out.update({"bytes": int(res.get("bytes", 0)), "seconds": float(res.get("seconds", 0.0)),
                "experts": len(hot), "stopped": res.get("stopped", "")})
    if out["bytes"] < total and not out["stopped"]:
        out["stopped"] = "hit the time limit"
    return out


def warm_page_cache(path: str, budget_bytes: int = 0, should_stop=None,
                    chunk: int = 32 << 20, should_pause=None, regions=None) -> dict:
    """Pull the expert file into the OS page cache, sequentially, in the background.

    WHAT THIS IS FOR, AND WHAT IT IS NOT FOR.
        It is NOT for a first-request ramp, because there is not one. That was checked before
        this was written: ten consecutive requests on Qwen3-30B read 8.37 tok/s across the first
        three and 7.70 across the last three, with a 27% spread. The "6.80 -> 9.21 warming up"
        reported earlier was run-to-run noise, and OLMoE started cold from a blob untouched for
        three days showed no ramp at all -- 14.4 first, 15.2 sixth.

        It IS for a genuinely cold machine. Expert reads now go through the page cache, and on a
        machine that has just booted there is nothing in it: measured, that is the difference
        between 7.5 and 9.3 tok/s. The cache fills on its own within a few requests, so this only
        moves that forward.

    WHY IT RUNS IN THE BACKGROUND AND CANNOT BE ALLOWED TO HURT.
        Reading 12.68 GB takes about three seconds at 4.9 GB/s, which is not a cost worth adding
        to every startup for a benefit that applies once. Backgrounded it costs nothing and the
        cache is warm before most people finish typing. It stops the moment the machine is short
        of memory, because filling a cache is never worth causing the pressure this engine spends
        the rest of its time avoiding.
    """
    import time as _t
    from .calibrate import under_pressure
    out = {"bytes": 0, "seconds": 0.0, "stopped": ""}
    try:
        total = os.path.getsize(path)
    except OSError as e:
        out["stopped"] = f"cannot read {os.path.basename(path)}: {e}"
        return out
    cap = int(budget_bytes) if budget_bytes else total
    t0 = _t.perf_counter()
    last_check = t0
    fd = None
    out["hot_bytes"] = 0
    try:
        fd = os.open(path, os.O_RDONLY)
        # THE MOST-USED EXPERTS FIRST, when a previous run left a record of them. A cache that
        # cannot hold the whole file then holds what this model actually asks for; a cache that
        # can holds everything either way, and the sequential pass below fills in the rest.
        for off, n in (regions or []):
            if out["bytes"] >= cap or (should_stop is not None and should_stop()):
                break
            if should_pause is not None:
                while should_pause():
                    if should_stop is not None and should_stop():
                        break
                    _t.sleep(0.05)
            try:
                b = os.pread(fd, int(n), int(off))
            except OSError:
                break
            out["bytes"] += len(b)
            out["hot_bytes"] += len(b)
        while out["bytes"] < min(cap, total):
            if should_stop is not None and should_stop():
                out["stopped"] = "asked to stop"
                break
            # YIELD TO A REPLY. Warming reads the disk at full speed, and a reply that has to
            # fetch experts from the same disk at the same moment loses. While a request is in
            # flight the warm waits; it resumes the moment the server is idle again.
            if should_pause is not None:
                while should_pause():
                    if should_stop is not None and should_stop():
                        break
                    _t.sleep(0.05)
            # CHECKED PERIODICALLY, NOT PER CHUNK. `under_pressure` samples over a 0.4 s
            # window because that is what distinguishes a compressor that is growing from one
            # that merely has history. Calling it every 32 MB spent 45 of 47 seconds asleep and
            # warmed 3.62 GB at 0.08 GB/s against a raw read of 4.9. The machine's memory still
            # cannot turn unnoticed -- it is checked every couple of seconds, which is far
            # faster than a cache fills.
            now = _t.perf_counter()
            if now - last_check >= 2.0:
                last_check = now
                if under_pressure():
                    out["stopped"] = "the machine went short of memory"
                    break
            b = os.read(fd, chunk)
            if not b:
                break
            out["bytes"] += len(b)
    except OSError as e:
        out["stopped"] = f"read failed: {e}"
    finally:
        if fd is not None:
            os.close(fd)
    out["seconds"] = round(_t.perf_counter() - t0, 2)
    out["gb_per_s"] = round(out["bytes"] / 1e9 / out["seconds"], 2) if out["seconds"] else 0.0
    return out


# --------------------------------------------------------------------------- lazy entry point
def load_streaming(model_dir: str, blob_path: str = "", capacity=0.5, policy: str = "lfuda",
                   threads: int = 8, nocache: bool = False, verbose: bool = True,
                   full_layers=(), manifest: dict | None = None, warm=True,
                   predictors: dict | None = None, reroute: float = 0.0):
    """Load a model whose expert weights are NEVER materialised, then attach the pool.

    THE ONE LINE THAT MAKES STREAMING REAL IS `lazy=True`.
    mlx_lm's default load evaluates every parameter up front, so the full (E, ...) expert tensors
    exist in memory before anything can drop them. Measured consequence: peak RSS was IDENTICAL
    at C=64/64 and C=16/64 -- 4.07 GB both times -- so a pool holding a quarter of the experts
    saved exactly nothing, and MLX ran 2.4 GB over its own allocator limit, which cost 3x in
    throughput on top. `mx.load` hands back lazy, mmap-backed arrays; deleting one before it is
    ever evaluated means those bytes are never read.

    Nothing may touch the model between load and attach. A single forward pass, a dtype probe, an
    `mx.eval(model)` -- any of them faults the whole expert tensor in and the saving is gone.
    """
    model_dir = os.path.expanduser(model_dir)
    blob_path = os.path.expanduser(blob_path) or os.path.join(
        os.path.join(home(), "data", "blobs"), os.path.basename(model_dir) + ".experts")
    model, tok = load_lenient(model_dir, lazy=True)
    h = attach(model, blob_path, capacity=capacity, policy=policy, threads=threads,
               nocache=nocache, verbose=verbose, full_layers=full_layers, manifest=manifest,
               predictors=predictors, reroute=reroute,
               warm=warm)
    mx.eval(model.parameters())      # the non-expert weights only; experts live in the pool
    return model, tok, h


def ensure_packed(model_dir: str, blob_path: str = "", verbose: bool = True) -> str:
    """Pack a model's experts to disk if that has not already been done. Returns the blob path."""
    model_dir = os.path.expanduser(model_dir)
    blob_path = os.path.expanduser(blob_path) or os.path.join(
        os.path.join(home(), "data", "blobs"), os.path.basename(model_dir) + ".experts")
    BLOB_VERSION = 4
    if os.path.exists(blob_path) and os.path.exists(blob_path + ".manifest.json"):
        m = load_manifest(blob_path)
        if os.path.getsize(blob_path) != m["total_bytes"]:
            if verbose:
                print(f"  blob is {os.path.getsize(blob_path)} bytes but the manifest says "
                      f"{m['total_bytes']} -- repacking rather than reading a truncated file")
        elif m.get("version", 0) < BLOB_VERSION:
            # A blob packed by an older build lacks the recorded quantisation parameters, so it
            # cannot be compressed. Upgrading it silently is right: the alternative is a raw
            # ValueError from deep inside the requantiser, which tells a user nothing they can
            # act on and makes the product look broken over a stale cache file.
            if verbose:
                print(f"  blob was packed by an older version ({m.get('version', 0)} < "
                      f"{BLOB_VERSION}); repacking so it carries its precision")
        else:
            return blob_path
    model, _ = load_lenient(model_dir, lazy=True)
    pack_experts(model, blob_path, progress=verbose)
    del model
    mx.clear_cache()
    return blob_path


# --------------------------------------------------------------------------- capacity planning
def _measured_curve(tag: str = "qwen3"):
    """Warm miss-vs-residency, measured on real traces. Returns f(residency) -> miss rate."""
    p = os.path.join(home(), "data", "results", "warm_locality.json")
    with open(p) as fh:
        c = {float(k): v for k, v in json.load(fh)[tag]["curve"].items()}
    xs = sorted(c)

    def f(r):
        if r >= 1.0:
            return 0.0
        if r <= xs[0]:
            return c[xs[0]]
        for a, b in zip(xs, xs[1:]):
            if a <= r <= b:
                return c[a] + (r - a) / (b - a) * (c[b] - c[a])
        return c[xs[-1]]
    return f


def _sync_curve():
    """Measured cost of running N of this model's MoE layers through the host round-trip.

    NOT a constant per sync. An earlier planner assumed one, predicted 17% faster, and measured
    1.00x. The real curve is 0.0 / 11.4 / 28.5 / 64.2 / 79.8 / 98.5 ms at 0 / 6 / 12 / 24 / 36 /
    48 syncing layers -- marginal cost between 1.3 and 3.0 ms depending where you sit on it, so
    the shape has to be carried rather than averaged away.
    """
    fp = os.path.join(home(), "data", "results", "sync_curve.json")
    if not os.path.exists(fp):
        return None
    with open(fp) as fh:
        d = json.load(fh)
    pts = sorted((int(k), v - d["floor_ms"]) for k, v in d["ms_by_syncing_layers"].items())
    meas_layers = pts[-1][0]

    def f(n, n_layers):
        x = n * (meas_layers / max(1, n_layers))       # scale to the measured layer count
        if x <= pts[0][0]:
            return pts[0][1]
        if x >= pts[-1][0]:
            return pts[-1][1] * x / pts[-1][0]
        for (a, va), (b, vb) in zip(pts, pts[1:]):
            if a <= x <= b:
                return va + (x - a) / (b - a) * (vb - va)
        return pts[-1][1]
    return f, d


def plan_capacity(n_layers: int, n_experts: int, slot_budget: int, top_k: int = 8,
                  sync_ms: float | None = None, fetch_ms_per_expert: float | None = None,
                  curve_tag: str = "qwen3", measured=None) -> dict:
    """Split a slot budget between sync-free layers and streamed ones, on measured costs.

    A layer held at C == E can never miss, so it needs no host round-trip: its expert-to-slot
    lookup runs on the GPU and MLX keeps pipelining. A streamed layer pays that round-trip every
    token. Measured on this machine it is worth about 2 ms, while a cache miss is worth 1.46 ms
    -- so ONE SYNC COSTS ABOUT ONE AND A HALF MISSES.

    Whether the trade pays depends on the budget, and not in the obvious direction. At 50% of
    experts, buying a sync-free layer costs 64 slots taken from the rest, which drives their miss
    rate up faster than the saved sync pays back. At 90% it is strongly positive. So every split
    is evaluated against the measured curve rather than assumed.

    THE CURVE DOES NOT TRANSFER BETWEEN MODELS, AND THIS IS WHY THE PLANNER IS OPT-IN
        The shipped curve was measured on Qwen3-30B, whose per-layer graph is large and whose
        round-trip costs about 2 ms. OLMoE-1B's costs about 0.9 ms. Scaling by layer count does
        not correct for that, and the error is not small:

            OLMoE at a 90% budget    predicted 4.87x    measured 1.19x
            Qwen3 at a 60% budget    predicted 1.06x    measured 0.97x

        The MECHANISM is sound -- a sync-free layer really does skip the round-trip, and the
        mixed configuration is bit-identical -- but the arithmetic that chooses the split needs
        the curve measured for THIS model on THIS machine. Until that calibration exists, the
        planner stays off by default and uniform capacity is what ships.
    """
    miss = _measured_curve(curve_tag)
    # A curve measured on THIS model, from `bigrig calibrate`, beats anything else here. It is
    # the whole reason this planner has shipped off: the arithmetic was sound and it was being
    # fed another model's constants.
    if measured is not None:
        f_meas, mpm_meas, miss_meas = measured
        if miss_meas is not None:
            miss = miss_meas               # this model's own residency curve, not a shared one
        mpm = mpm_meas if fetch_ms_per_expert is None else fetch_ms_per_expert
        if mpm is None:
            mpm = 1.46
        best = None
        for k in range(0, n_layers + 1):
            rest = n_layers - k
            if k * n_experts > slot_budget:
                break
            if rest == 0:
                C, m = n_experts, 0.0
            else:
                C = min(n_experts, (slot_budget - k * n_experts) // rest)
                if C < top_k:
                    continue
                m = 0.0 if C >= n_experts else miss(C / n_experts)
            cost = f_meas(rest) + rest * top_k * m * mpm
            row = {"full_layers": k, "streamed_layers": rest, "capacity": int(C),
                   "residency": C / n_experts, "ms_per_token": cost, "miss_rate": m,
                   "source": "measured"}
            if best is None or cost < best["ms_per_token"]:
                best = row
        if best is None:
            raise ValueError(
                f"a budget of {slot_budget} slots cannot serve {n_layers} layers at top-{top_k}: "
                f"every layer needs at least {top_k} slots, so {n_layers * top_k} is the floor.")
        uni = min(n_experts, slot_budget // n_layers)
        um = 0.0 if uni >= n_experts else miss(uni / n_experts)
        best["uniform_capacity"] = int(uni)
        best["uniform_ms_per_token"] = (0.0 if uni >= n_experts
                                        else f_meas(n_layers) + n_layers * top_k * um * mpm)
        return best

    cv = _sync_curve()
    if cv is None or sync_ms is not None:
        smd = 2.05 if sync_ms is None else sync_ms
        mpm = 1.46 if fetch_ms_per_expert is None else fetch_ms_per_expert

        def sync_cost(n):
            return n * smd
    else:
        f, d = cv
        mpm = d["ms_per_miss"] if fetch_ms_per_expert is None else fetch_ms_per_expert

        def sync_cost(n):
            return f(n, n_layers)

    best = None
    for k in range(0, n_layers + 1):
        rest = n_layers - k
        if k * n_experts > slot_budget:
            break
        if rest == 0:
            C, m = n_experts, 0.0
        else:
            C = min(n_experts, (slot_budget - k * n_experts) // rest)
            if C < top_k:
                continue
            m = 0.0 if C >= n_experts else miss(C / n_experts)
        cost = sync_cost(rest) + rest * top_k * m * mpm
        row = {"full_layers": k, "streamed_layers": rest, "capacity": int(C),
               "residency": C / n_experts, "ms_per_token": cost, "miss_rate": m}
        if best is None or cost < best["ms_per_token"]:
            best = row
    if best is None:
        raise ValueError(
            f"a budget of {slot_budget} slots cannot serve {n_layers} layers at top-{top_k}: "
            f"every layer needs at least {top_k} slots, so {n_layers * top_k} is the floor.")
    uni = min(n_experts, slot_budget // n_layers)
    um = 0.0 if uni >= n_experts else miss(uni / n_experts)
    best["uniform_capacity"] = int(uni)
    best["uniform_ms_per_token"] = (0.0 if uni >= n_experts
                                    else sync_cost(n_layers) + n_layers * top_k * um * mpm)
    return best
