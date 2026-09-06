"""Keep the input-embedding table on disk and gather only the rows a step needs.

WHY. A model's input-embedding table is used one row at a time -- a token looks up its own row --
but the whole table is held resident because indexing it (`weight[ids]`) needs the full array in
memory. On Qwen3.6 that is 0.29 GB of wired memory to serve one 256-wide row per token. Streamed
models are the ones short of memory, and this is 2-4% of their floor, freed with no quality cost:
the arithmetic is a gather followed by the model's own dequantise, byte for byte the same rows.

WHAT IT DOES NOT TOUCH. Only the INPUT embedding, and only when it is a separate tensor from the
output head. A tied model uses one matrix for both, and the output head needs the whole matrix
every token (a matmul over the vocabulary), so there is nothing to stream. `attach` returns
False in that case and the model is left exactly as it was.

HOW. `QuantizedEmbedding.__call__` is `dequantize(weight[x], scales[x], biases[x], ...)`. The
weight, scales and biases are kept here as numpy memory-maps of the model's own safetensors --
page cache, which macOS reclaims under pressure and never counts against the reserve -- and each
call gathers the rows for `x` from those maps into small arrays, then runs the identical MLX
dequantise. The result is bit-identical to the resident module; the only difference is where the
untouched bytes live.
"""
from __future__ import annotations

import numpy as np

import mlx.core as mx
import mlx.nn as nn

# safetensors dtype -> numpy dtype, for the three arrays a quantised embedding holds.
_NP = {"U32": np.uint32, "I32": np.int32, "F16": np.float16, "BF16": np.uint16, "F32": np.float32}


def _memmap(loc: tuple) -> tuple:
    """A read-only numpy memmap of one tensor, plus a flag for bf16 (which numpy has no dtype for
    and must be viewed as uint16 and reinterpreted on gather)."""
    path, off, nbytes, dtype, shape = loc
    npdt = _NP.get(dtype)
    if npdt is None:
        raise TypeError(f"embedding streaming does not handle dtype {dtype}")
    count = nbytes // np.dtype(npdt).itemsize
    mm = np.memmap(path, dtype=npdt, mode="r", offset=off, shape=(count,))
    return mm.reshape(shape), dtype == "BF16"


class StreamingQuantizedEmbedding(nn.Module):
    """A drop-in for QuantizedEmbedding whose table lives in the page cache, not in wired memory.

    Holds numpy memory-maps of the packed weight, scales and biases. On call it gathers the rows
    for the given ids and runs mx.dequantize with the module's own bits/group_size/mode, so the
    output is exactly what the resident module would have produced.
    """

    def __init__(self, weight_loc, scales_loc, biases_loc, bits, group_size, mode):
        super().__init__()
        self._w, self._w_bf16 = _memmap(weight_loc)
        self._s, self._s_bf16 = _memmap(scales_loc)
        if biases_loc is not None:
            self._b, self._b_bf16 = _memmap(biases_loc)
        else:
            self._b, self._b_bf16 = None, False
        self.bits = int(bits)
        self.group_size = int(group_size)
        self.mode = mode

    @staticmethod
    def _gather(mm, is_bf16, ids: np.ndarray) -> mx.array:
        rows = np.ascontiguousarray(mm[ids])       # fancy-index reads only the touched rows
        if is_bf16:
            # numpy has no bfloat16: the bytes were mapped as uint16; hand them to MLX as the
            # raw bits and reinterpret, so no value is altered on the way through.
            return mx.array(rows).view(mx.bfloat16)
        return mx.array(rows)

    def __call__(self, x):
        ids = np.asarray(x, dtype=np.int64)
        flat = ids.reshape(-1)
        w = self._gather(self._w, self._w_bf16, flat)
        s = self._gather(self._s, self._s_bf16, flat)
        b = self._gather(self._b, self._b_bf16, flat) if self._b is not None else None
        out = mx.dequantize(w, scales=s, biases=b, group_size=self.group_size,
                            bits=self.bits, mode=self.mode)
        return out.reshape(*ids.shape, out.shape[-1])


def _locate(model_dir: str):
    """(weight, scales, biases-or-None) locations for a streamable embedding, or None.

    None means "leave the embedding resident": the model is tied (the head needs the whole
    matrix), the embedding is not quantised with weight+scales, or the file cannot be read. This
    is the SINGLE check both the plan-time size and the load-time attach go through, so the plan
    never reserves a saving that attach then cannot deliver.
    """
    from . import direct
    try:
        loc = direct.scan(model_dir)
    except Exception:                              # noqa: BLE001
        return None
    hit = next((n for n in loc if n.endswith("embed_tokens.weight")), None)
    if hit is None:
        return None
    base = hit[: -len("weight")]
    wl, sl, bl = loc.get(base + "weight"), loc.get(base + "scales"), loc.get(base + "biases")
    if wl is None or sl is None:
        return None                                # unquantised or shaped unexpectedly
    if wl[3] not in _NP or sl[3] not in _NP or (bl is not None and bl[3] not in _NP):
        return None                                # a dtype the gather does not handle
    return wl, sl, bl


def _tied(model_dir: str) -> bool:
    import json
    import os
    try:
        cfg = json.load(open(os.path.join(model_dir, "config.json")))
    except (OSError, ValueError):
        return False
    cfg = {**cfg.get("text_config", {}), **cfg}
    return bool(cfg.get("tie_word_embeddings", False))


def streamable_gb(model_dir: str) -> float:
    """GB the input embedding would free if streamed, or 0.0 when it cannot be.

    Used at plan time to size the pool as if the embedding were not resident -- but only when it
    is certain to be streamable, so the pool never grows on a saving that does not arrive. The
    weight, scales and biases all leave wired memory; their packed sizes are the saving.
    """
    if _tied(model_dir):
        return 0.0
    loc = _locate(model_dir)
    if loc is None:
        return 0.0
    return sum(t[2] for t in loc if t is not None) / 1e9


def _find_embedding(model):
    """(parent module, attribute name, module) for the model's QuantizedEmbedding, or None."""
    for name, mod in model.named_modules():
        if isinstance(mod, nn.QuantizedEmbedding) and name.endswith("embed_tokens"):
            parent = model
            *parents, leaf = name.split(".")
            for p in parents:
                parent = parent[int(p)] if p.isdigit() else getattr(parent, p)
            return parent, leaf, mod
    return None


def attach(model, model_dir: str) -> bool:
    """Replace the model's input embedding with a streaming one. Returns True if it did.

    Refuses, leaving the model untouched, when: the model ties its input and output weights (the
    head needs the whole matrix); the embedding is not quantised in a way this handles; or the
    tensor cannot be located on disk. A refusal is never an error -- the model runs resident, as
    before.
    """
    found = _find_embedding(model)
    if found is None or _tied(model_dir):
        return False
    parent, leaf, emb = found
    loc = _locate(model_dir)
    if loc is None:
        return False
    wl, sl, bl = loc
    biases = getattr(emb, "get", lambda _k: None)("biases")
    try:
        streamed = StreamingQuantizedEmbedding(
            wl, sl, bl if biases is not None else None,
            emb.bits, emb.group_size, emb.mode)
    except (TypeError, ValueError):
        return False
    setattr(parent, leaf, streamed)
    return True
