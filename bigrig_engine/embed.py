"""Sentence embeddings, for the `/v1/embeddings` endpoint.

WHY. Every retrieval and codebase-indexing tool -- and most agent memories -- calls
`/v1/embeddings`, and a server without it cannot be their backend at all. The chat model is the
wrong tool for it: a decoder's hidden states are not trained to be compared, and running a
35B-parameter model to embed a paragraph would be slow and worse than a 33M-parameter encoder
trained for exactly that.

WHAT. A BERT encoder in MLX, written here rather than pulled in: the maintained MLX embeddings
package depends on mlx-vlm, which depends on OpenCV, an audio stack and a web framework -- far
too much to hang on `pip install bigrig` for a 150-line model. The weights are the model's own
`model.safetensors` from the hub, in sentence-transformers form (BERT + pooling + normalise), read
directly. The default is BAAI/bge-small-en-v1.5: MIT, 133 MB, 384 dimensions, 512 tokens; any
BERT-shaped sentence-transformers model with a tokenizer.json loads the same way.

CORRECTNESS. Checked against sentence-transformers (PyTorch) on the same texts: see
tests/test_embed.py for the numbers. Float32 throughout, so the comparison is a comparison and
not a tolerance argument.

MEMORY. The encoder lives beside the expert pool and is charged to the ceiling before the pool
is planned (Session's `reserved_gb`), the same way a draft model is.
"""
from __future__ import annotations

import base64
import json
import math
import os

import numpy as np

import mlx.core as mx
import mlx.nn as nn

from . import home

DEFAULT_REPO = "BAAI/bge-small-en-v1.5"
# What a sentence-transformers BERT checkpoint needs to run here. Nothing else is fetched -- no
# ONNX or OpenVINO exports, which are most of some repos.
FILES = ["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json",
         "special_tokens_map.json", "modules.json", "1_Pooling/config.json",
         "sentence_bert_config.json", "config_sentence_transformers.json"]
MAX_ITEMS = 256          # inputs per request; a bigger batch is several requests
MAX_CHARS = 200_000      # per input, before tokenising; a guard against a pasted binary


def models_dir() -> str:
    return os.path.join(home(), "models", "embeddings")


def local_dir(repo: str) -> str:
    return os.path.join(models_dir(), repo.replace("/", "--"))


def is_local(repo: str) -> bool:
    d = local_dir(repo)
    return all(os.path.exists(os.path.join(d, f)) for f in ("config.json", "model.safetensors", "tokenizer.json"))


def fetch(repo: str = DEFAULT_REPO, quiet: bool = False) -> str:
    """Download what the encoder needs into BigRig's models directory. Returns the directory.
    Asked for by `--embeddings`, which is the consent: the flag names a download of this size."""
    from huggingface_hub import snapshot_download
    d = local_dir(repo)
    if is_local(repo):
        return d
    os.makedirs(d, exist_ok=True)
    if not quiet:
        print(f"  fetching the embedding model {repo} into {d}", flush=True)
    snapshot_download(repo_id=repo, local_dir=d, allow_patterns=FILES)
    return d


# ------------------------------------------------------------------------------ the model
class _Layer(nn.Module):
    def __init__(self, hidden: int, heads: int, inter: int, eps: float):
        super().__init__()
        self.heads = heads
        self.query = nn.Linear(hidden, hidden)
        self.key = nn.Linear(hidden, hidden)
        self.value = nn.Linear(hidden, hidden)
        self.attn_out = nn.Linear(hidden, hidden)
        self.attn_norm = nn.LayerNorm(hidden, eps=eps)
        self.inter = nn.Linear(hidden, inter)
        self.out = nn.Linear(inter, hidden)
        self.out_norm = nn.LayerNorm(hidden, eps=eps)

    def __call__(self, x, mask):
        B, L, D = x.shape
        h = self.heads
        q = self.query(x).reshape(B, L, h, D // h).transpose(0, 2, 1, 3)
        k = self.key(x).reshape(B, L, h, D // h).transpose(0, 2, 1, 3)
        v = self.value(x).reshape(B, L, h, D // h).transpose(0, 2, 1, 3)
        a = mx.fast.scaled_dot_product_attention(q, k, v, scale=1.0 / math.sqrt(D // h), mask=mask)
        a = a.transpose(0, 2, 1, 3).reshape(B, L, D)
        x = self.attn_norm(x + self.attn_out(a))
        f = self.out(nn.gelu(self.inter(x)))            # BERT's `gelu` is the exact erf form
        return self.out_norm(x + f)


class Bert(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        hidden, eps = int(cfg["hidden_size"]), float(cfg.get("layer_norm_eps", 1e-12))
        self.word = nn.Embedding(int(cfg["vocab_size"]), hidden)
        self.position = nn.Embedding(int(cfg["max_position_embeddings"]), hidden)
        self.token_type = nn.Embedding(int(cfg.get("type_vocab_size", 2)), hidden)
        self.norm = nn.LayerNorm(hidden, eps=eps)
        self.layers = [_Layer(hidden, int(cfg["num_attention_heads"]), int(cfg["intermediate_size"]), eps)
                       for _ in range(int(cfg["num_hidden_layers"]))]

    def __call__(self, ids, type_ids, attn_mask):
        B, L = ids.shape
        x = self.word(ids) + self.position(mx.arange(L)[None, :]) + self.token_type(type_ids)
        x = self.norm(x)
        # Padded keys are masked out for every query; an additive mask in the attention's dtype.
        neg = mx.array(-1e9, dtype=x.dtype)
        mask = mx.where(attn_mask[:, None, None, :].astype(mx.bool_), mx.array(0.0, dtype=x.dtype), neg)
        for layer in self.layers:
            x = layer(x, mask)
        return x


# HF BERT parameter names -> ours. `bert.` prefixes (BertForX checkpoints) are stripped first.
_MAP = {
    "embeddings.word_embeddings.weight": "word.weight",
    "embeddings.position_embeddings.weight": "position.weight",
    "embeddings.token_type_embeddings.weight": "token_type.weight",
    "embeddings.LayerNorm.weight": "norm.weight", "embeddings.LayerNorm.bias": "norm.bias",
}
_LAYER_MAP = {
    "attention.self.query": "query", "attention.self.key": "key", "attention.self.value": "value",
    "attention.output.dense": "attn_out", "attention.output.LayerNorm": "attn_norm",
    "intermediate.dense": "inter", "output.dense": "out", "output.LayerNorm": "out_norm",
}


def _rename(name: str):
    if name.startswith("bert."):
        name = name[5:]
    if name in _MAP:
        return _MAP[name]
    if name.startswith("encoder.layer."):
        rest = name[len("encoder.layer."):]
        idx, rest = rest.split(".", 1)
        for hf, ours in _LAYER_MAP.items():
            if rest.startswith(hf + "."):
                return f"layers.{idx}.{ours}.{rest[len(hf) + 1:]}"
    return None                                       # pooler, cls head: not used for embeddings


class Embedder:
    """A loaded sentence-transformers BERT: tokenizer, encoder, pooling and normalisation."""

    def __init__(self, model_dir: str):
        from tokenizers import Tokenizer
        self.dir = model_dir
        self.name = os.path.basename(model_dir.rstrip("/")).replace("--", "/")
        with open(os.path.join(model_dir, "config.json")) as f:
            self.cfg = json.load(f)
        if self.cfg.get("model_type") not in ("bert", None):
            raise ValueError(f"{self.name} is a {self.cfg.get('model_type')} model; only BERT-shaped "
                             f"sentence-transformers encoders are supported here")
        self.model = Bert(self.cfg)
        weights = mx.load(os.path.join(model_dir, "model.safetensors"))
        params = {}
        for k, v in weights.items():
            ours = _rename(k)
            if ours is not None:
                params[ours] = v.astype(mx.float32)
        self.model.load_weights(list(params.items()), strict=True)
        mx.eval(self.model.parameters())
        self.nbytes = sum(v.nbytes for v in params.values())
        self.dimensions = int(self.cfg["hidden_size"])
        # How long an input may be, in tokens: the smaller of the position table and what the
        # sentence-transformers config says the model was used with.
        self.max_tokens = int(self.cfg["max_position_embeddings"])
        try:
            with open(os.path.join(model_dir, "sentence_bert_config.json")) as f:
                self.max_tokens = min(self.max_tokens, int(json.load(f).get("max_seq_length") or self.max_tokens))
        except (OSError, ValueError, TypeError):
            pass
        self.pooling, self.normalize = self._pooling_config(model_dir)
        self.tok = Tokenizer.from_file(os.path.join(model_dir, "tokenizer.json"))
        self.tok.enable_truncation(self.max_tokens)
        self.tok.no_padding()

    @staticmethod
    def _pooling_config(model_dir: str) -> tuple:
        pooling, normalize = "mean", True
        try:
            with open(os.path.join(model_dir, "1_Pooling", "config.json")) as f:
                pc = json.load(f)
            if pc.get("pooling_mode_cls_token"):
                pooling = "cls"
            elif pc.get("pooling_mode_mean_tokens"):
                pooling = "mean"
        except (OSError, ValueError):
            pass
        try:
            with open(os.path.join(model_dir, "modules.json")) as f:
                normalize = any("Normalize" in (m.get("type") or "") for m in json.load(f))
        except (OSError, ValueError):
            pass
        return pooling, normalize

    @property
    def gb(self) -> float:
        return self.nbytes / 1e9

    def count_tokens(self, text: str) -> int:
        return len(self.tok.encode(text).ids)

    def embed(self, texts: list, batch_size: int = 32) -> tuple:
        """(embeddings [n, d] float32, tokens counted, inputs truncated).

        Each batch is padded to its own longest input; inputs longer than `max_tokens` are cut to
        it -- the model has no positions past there -- and counted in the third return so the
        caller can say so rather than pretend the whole text was read.
        """
        out = np.zeros((len(texts), self.dimensions), dtype=np.float32)
        n_tokens = truncated = 0
        for start in range(0, len(texts), batch_size):
            chunk = texts[start:start + batch_size]
            encs = self.tok.encode_batch(chunk)
            for e in encs:
                n_tokens += len(e.ids)               # what was actually read, after any cut
                truncated += 1 if e.overflowing else 0
            L = max(len(e.ids) for e in encs)
            ids = np.zeros((len(chunk), L), dtype=np.int32)
            types = np.zeros((len(chunk), L), dtype=np.int32)
            am = np.zeros((len(chunk), L), dtype=np.int32)
            for i, e in enumerate(encs):
                n = len(e.ids)
                ids[i, :n] = e.ids
                types[i, :n] = e.type_ids
                am[i, :n] = 1
            h = self.model(mx.array(ids), mx.array(types), mx.array(am))
            if self.pooling == "cls":
                pooled = h[:, 0, :]
            else:
                m = mx.array(am).astype(h.dtype)[:, :, None]
                pooled = (h * m).sum(axis=1) / mx.maximum(m.sum(axis=1), 1.0)
            if self.normalize:
                pooled = pooled / mx.maximum(mx.linalg.norm(pooled, axis=-1, keepdims=True), 1e-12)
            mx.eval(pooled)
            out[start:start + len(chunk)] = np.array(pooled, dtype=np.float32)
        return out, n_tokens, truncated


# ------------------------------------------------------------------------------ the wire shape
def parse_input(body: dict) -> list:
    """The `input` of an OpenAI embeddings request as a list of strings, or raise ValueError
    with a sentence. Token-id inputs are refused: they would be another model's ids."""
    inp = body.get("input")
    if isinstance(inp, str):
        items = [inp]
    elif isinstance(inp, list) and inp and all(isinstance(x, str) for x in inp):
        items = list(inp)
    elif isinstance(inp, list) and inp and all(isinstance(x, list) for x in inp):
        raise ValueError("`input` as token ids is not supported here; send text -- the ids would "
                         "belong to another model's tokenizer")
    else:
        raise ValueError("`input` must be a string or a non-empty list of strings")
    if len(items) > MAX_ITEMS:
        raise ValueError(f"`input` has {len(items)} items; at most {MAX_ITEMS} per request")
    if any(not x.strip() for x in items):
        raise ValueError("`input` contains an empty string; there is nothing to embed")
    if any(len(x) > MAX_CHARS for x in items):
        raise ValueError(f"an `input` item is longer than {MAX_CHARS} characters")
    fmt = body.get("encoding_format", "float")
    if fmt not in ("float", "base64"):
        raise ValueError("`encoding_format` must be \"float\" or \"base64\"")
    dims = body.get("dimensions")
    if dims is not None and (not isinstance(dims, int) or dims <= 0):
        raise ValueError("`dimensions` must be a positive integer")
    return items


def response(vectors: np.ndarray, model_name: str, n_tokens: int, truncated: int,
             encoding_format: str = "float", dimensions=None) -> dict:
    """The OpenAI embeddings response. `dimensions` cuts the vector (Matryoshka-style clients
    ask for it) and re-normalises, as the OpenAI API does."""
    vecs = vectors
    if dimensions is not None and dimensions < vecs.shape[1]:
        vecs = vecs[:, :dimensions]
        norms = np.maximum(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-12)
        vecs = vecs / norms
    data = []
    for i, v in enumerate(vecs):
        if encoding_format == "base64":
            emb = base64.b64encode(v.astype("<f4").tobytes()).decode("ascii")
        else:
            emb = [float(x) for x in v]
        data.append({"object": "embedding", "index": i, "embedding": emb})
    out = {"object": "list", "data": data, "model": model_name,
           "usage": {"prompt_tokens": int(n_tokens), "total_tokens": int(n_tokens)}}
    if truncated:
        out["bigrig"] = {"truncated_inputs": int(truncated),
                         "note": "inputs longer than the model's window were cut to it"}
    return out
