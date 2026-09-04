"""The model's own next-token head, used to guess one token ahead so the big model can check two
tokens in one pass.

WHAT IT IS
    Qwen3.5 and 3.6 checkpoints ship a small "multi-token prediction" head: one transformer
    layer that takes the backbone's last hidden state and the embedding of the token just
    chosen, and predicts the token after that. Trained with the model, so it agrees with the
    model far more often than a separate small draft model does. The MLX quantisations strip
    it; mlx-community publishes it separately in bf16 (1.69 GB) and 4-bit.

WHAT IT IS NOT
    It is not a change to the model's output. Every guess is checked by the target: a verify
    pass runs the target over [token, guess] and keeps the guess only if the target's own
    choice at that position is the guess. A rejected guess costs the pass and is replaced by
    what the target said. Nothing is ever emitted that the target did not choose.

THE ONE CAVEAT, SAID PLAINLY
    A two-token pass and a one-token pass are the same arithmetic through different kernels,
    and on a rejected guess this engine does not trust them to agree: it puts the state back
    and re-runs the one confirmed token on its own, so a miss is bit-identical to ordinary
    decoding. On an accepted guess the two tokens come from the two-token pass, and a two-row
    quantised matmul is a different kernel from the one-row GEMV: measured, about one position
    in a hundred is a near-tie that flips (0.8-2.5% of positions; 0 of 6 250-token replies
    byte-identical), the same phenomenon batching has. So MTP is a
    choice, not a default, and the product's bit-identical claim does not extend to it.

MEASURED FIRST, BEFORE ANY OF THE LOOP WAS WRITTEN
    On Qwen3.6-35B-A3B-4bit streamed at the 9.7 GB ceiling, greedy, 160 tokens a prompt, the
    bf16 head's guess matched the 4-bit target's next token 88.7% of the time over 1,280
    tokens (89.5% with thinking on, 88.0% off), at 6.2 ms a guess. That is the number the rest
    of this file exists for.

THE RECURRENT-STATE PROBLEM
    Qwen3.5/3.6 is a hybrid: three of every four layers are linear attention with a fixed
    recurrent state, not a KV cache. A KV cache can be trimmed after a rejected token; a
    recurrent state cannot -- once the wrong token has been folded in there is no undo. mlx_lm's
    own speculative decoding refuses such models for exactly this reason. Here the state of
    every recurrent layer is snapshotted before the two-token pass (the arrays are immutable,
    so a snapshot is two references a layer and costs nothing) and put back on a miss.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

# The head's own tensors. Everything else -- embeddings, the output projection -- it borrows
# from the target, because the checkpoint says so (`mtp_use_dedicated_embeddings: false`).
HEAD_TENSORS = 20


@dataclass
class Stats:
    rounds: int = 0          # verify passes issued
    drafted: int = 0         # guesses made (one per round)
    accepted: int = 0        # guesses the target agreed with
    recomputed: int = 0      # misses, each paid with one extra single-token pass

    @property
    def acceptance(self) -> float:
        return self.accepted / self.drafted if self.drafted else 0.0

    def as_dict(self) -> dict:
        return {"rounds": self.rounds, "drafted": self.drafted, "accepted": self.accepted,
                "recomputed": self.recomputed, "acceptance": round(self.acceptance, 4)}


class MTPHead(nn.Module):
    """fc(concat(norm(embed(next)), norm(hidden))) -> one decoder layer -> norm. Logits come
    from the target's lm_head, applied by the caller."""

    def __init__(self, text_args):
        super().__init__()
        from mlx_lm.models import qwen3_5 as _q
        a = text_args
        self.fc = nn.Linear(2 * a.hidden_size, a.hidden_size, bias=False)
        self.pre_fc_norm_hidden = nn.RMSNorm(a.hidden_size, eps=a.rms_norm_eps)
        self.pre_fc_norm_embedding = nn.RMSNorm(a.hidden_size, eps=a.rms_norm_eps)
        # A full-attention layer index, so the layer is built with attention rather than the
        # recurrent mixer. The head has a KV cache of its own and no recurrent state.
        self.layers = [_q.DecoderLayer(a, layer_idx=a.full_attention_interval - 1)]
        self.norm = nn.RMSNorm(a.hidden_size, eps=a.rms_norm_eps)

    def __call__(self, hidden: mx.array, tok_embed: mx.array, cache) -> mx.array:
        from mlx_lm.models.base import create_attention_mask
        fused = self.fc(mx.concatenate([self.pre_fc_norm_embedding(tok_embed),
                                        self.pre_fc_norm_hidden(hidden)], axis=-1))
        mask = create_attention_mask(fused, cache[0])
        return self.norm(self.layers[0](fused, mask=mask, cache=cache[0]))


def head_path(model_dir: str) -> str:
    """Where the head for a model lives, by the naming mlx-community uses: `<model>-MTP-bf16`
    beside the model, with the quantisation suffix of the target dropped."""
    base = os.path.basename(os.path.expanduser(model_dir).rstrip("/"))
    stem = base
    for sfx in ("-4bit-DWQ", "-4bit", "-5bit", "-6bit", "-8bit", "-bf16", "-mxfp4", "-nvfp4",
                "-mxfp8"):
        if stem.endswith(sfx):
            stem = stem[: -len(sfx)]
            break
    return os.path.join(os.path.dirname(os.path.expanduser(model_dir).rstrip("/")),
                        stem + "-MTP-bf16")


def head_gb(path: str) -> float:
    f = os.path.join(os.path.expanduser(path), "model.safetensors")
    return os.path.getsize(f) / 1e9 if os.path.exists(f) else 0.0


def load_head(path: str, text_args, quantize_bits: int | None = None) -> MTPHead:
    """Build the head and load its weights strictly: a missing tensor is an error, never a
    silently random layer. `quantize_bits` quantises the head's expert weights to fit smaller
    machines; the acceptance it costs is measured, not assumed."""
    f = os.path.join(os.path.expanduser(path), "model.safetensors")
    if not os.path.exists(f):
        raise FileNotFoundError(f"no MTP head at {path}")
    w = mx.load(f)
    if len(w) != HEAD_TENSORS:
        raise ValueError(f"{path} has {len(w)} tensors; an MTP head has {HEAD_TENSORS}. "
                         f"Refusing to guess which layer is which.")
    head = MTPHead(text_args)
    head.load_weights(list(w.items()), strict=True)
    if quantize_bits:
        nn.quantize(head, group_size=64, bits=int(quantize_bits),
                    class_predicate=lambda p, m: "switch_mlp" in p and hasattr(m, "to_quantized"))
    mx.eval(head.parameters())
    return head


# ------------------------------------------------------------------ the target, seen from here
def text_model(model):
    """The Qwen3.5-family text model inside whatever wrapper the checkpoint came with, or None
    when this model is not one the head can drive."""
    tm = getattr(model, "language_model", None) or model
    inner = getattr(tm, "model", None)
    if inner is None or not all(hasattr(inner, k) for k in ("layers", "norm", "embed_tokens",
                                                                "fa_idx", "ssm_idx")):
        return None
    if not all(hasattr(l, "is_linear") for l in inner.layers):
        return None
    return tm


def lm_head_of(tm):
    if getattr(tm.args, "tie_word_embeddings", False):
        return tm.model.embed_tokens.as_linear
    return tm.lm_head


def forward(tm, ids: mx.array, cache) -> tuple:
    """The text model's forward with the pre-norm hidden state kept. (logits, hidden), both
    (B, S, ...). Mirrors Qwen3_5TextModel.__call__ exactly; the only difference is that the
    hidden state is returned instead of dropped."""
    from mlx_lm.models.base import create_attention_mask, create_ssm_mask
    inner = tm.model
    h = inner.embed_tokens(ids)
    fa = create_attention_mask(h, cache[inner.fa_idx])
    sm = create_ssm_mask(h, cache[inner.ssm_idx])
    for layer, c in zip(inner.layers, cache):
        h = layer(h, mask=(sm if layer.is_linear else fa), cache=c)
    return lm_head_of(tm)(inner.norm(h)), h


# ------------------------------------------------------------------- snapshot and restore
def snapshot(cache) -> list:
    """Enough to put every cache entry back exactly as it is now. Immutable arrays are held
    by reference, so this costs nothing; a KV cache is a write position."""
    out = []
    for c in cache:
        if hasattr(c, "cache") and isinstance(getattr(c, "cache"), list):
            out.append(("arrays", list(c.cache)))
        elif hasattr(c, "offset") and hasattr(c, "trim") and c.is_trimmable():
            out.append(("offset", int(c.offset)))
        else:
            raise TypeError(f"cannot snapshot a {type(c).__name__}; MTP needs caches it can "
                            f"put back after a rejected guess")
    return out


def restore(cache, snap: list) -> None:
    for c, (kind, v) in zip(cache, snap):
        if kind == "arrays":
            c.cache = list(v)
        else:
            c.trim(max(0, int(c.offset) - int(v)))


def supports(model, cache=None) -> str:
    """"" if MTP can drive this model, else why not."""
    tm = text_model(model)
    if tm is None:
        return "not a Qwen3.5-family text model"
    if cache is not None:
        try:
            snapshot(cache)
        except TypeError as e:
            return str(e)
    return ""


# ------------------------------------------------------------------------------- the loop
def stream(model, head: MTPHead, tokenizer, prompt, max_tokens: int = 256, prompt_cache=None,
           sampler=None, prefill_step_size: int = 64, stats: Stats | None = None,
           prompt_progress_callback=None, on_logits=None):
    """Generate with the model's own head guessing one token ahead, yielding exactly what
    mlx_lm's stream_generate yields. See lookahead.stream for why the shape is not negotiable:
    the end-of-sequence token is never yielded and the final response repeats the last segment
    with a finish_reason, and everything downstream depends on both.

    `sampler` maps a (rows, vocab) log-probability array to (rows,) tokens; None is greedy.
    Sampling stays unbiased for the same reason it does in lookahead: every token kept was
    drawn from the target's own conditional distribution, and a guess is only ever "noticed",
    never trusted -- so the head's own probabilities are not needed and are not used.
    """
    import time as _t

    from mlx_lm.generate import GenerationResponse, generation_stream, wired_limit
    from mlx_lm.models.cache import KVCache, make_prompt_cache
    from mlx_lm.tokenizer_utils import TokenizerWrapper

    if not isinstance(tokenizer, TokenizerWrapper):
        tokenizer = TokenizerWrapper(tokenizer)
    tm = text_model(model)
    if tm is None:
        raise TypeError("MTP: not a model the head can drive")
    ids = list(prompt) if not isinstance(prompt, str) else tokenizer.encode(prompt)
    ids = [int(t) for t in ids]
    if prompt_cache is None:
        prompt_cache = make_prompt_cache(model)
    snapshot(prompt_cache)                       # refuse now, not after the first miss
    st = stats if stats is not None else Stats()
    detok = tokenizer.detokenizer
    detok.reset()
    with wired_limit(model, [generation_stream]), mx.stream(generation_stream):
        yield from _run(tm, head, tokenizer, ids, max_tokens, prompt_cache, sampler,
                        prefill_step_size, st, prompt_progress_callback, on_logits, detok,
                        GenerationResponse, KVCache, _t)


def _pick(sampler, rows: mx.array):
    """Tokens for each row of raw logits, plus the log-probabilities the meter wants."""
    lps = rows - mx.logsumexp(rows, axis=-1, keepdims=True)
    toks = mx.argmax(lps, axis=-1) if sampler is None else sampler(lps)
    return toks, lps


def _run(tm, head, tokenizer, ids, max_tokens, cache, sampler, step, st, progress, on_logits,
         detok, GenerationResponse, KVCache, _t):
    inner = tm.model
    embed = inner.embed_tokens
    hcache = [KVCache()]
    tic = _t.perf_counter()
    # PREFILL, IN THE ENGINE'S OWN WIDTH, KEEPING EVERY POSITION'S HIDDEN STATE.
    #     The head needs the backbone's hidden state at each prompt position, paired with the
    #     embedding of the token after it, to build its own context. Those states are what the
    #     backbone computes anyway; keeping them costs 4 KB a token.
    step = max(1, int(step))
    hidden = []
    body = ids[:-1]
    for i in range(0, len(body), step):
        chunk = mx.array(body[i:i + step])[None]
        _lg, h = forward(tm, chunk, cache)
        mx.eval(h, *[c.state for c in cache if hasattr(c, "state")])
        hidden.append(h)
        if progress is not None:
            try:
                progress(min(i + step, len(body)), len(ids))
            except Exception:                  # noqa: BLE001 -- progress is never worth a failure
                pass
    logits, h = forward(tm, mx.array([ids[-1]])[None], cache)
    hidden.append(h)
    H = mx.concatenate(hidden, axis=1) if len(hidden) > 1 else hidden[0]
    # The head reads the prompt too: position i with the token at i+1, for every i but the
    # last. The last position waits for the first generated token.
    if len(ids) > 1:
        for i in range(0, len(ids) - 1, step):
            j = min(i + step, len(ids) - 1)
            _ = head(H[:, i:j], embed(mx.array(ids[i + 1:j + 1])[None]), hcache)
            mx.eval(_)
    toks, lps = _pick(sampler, logits[0, -1:])
    mx.eval(toks)
    prompt_time = _t.perf_counter() - tic
    prompt_tps = len(ids) / prompt_time if prompt_time > 0 else 0.0
    tic = _t.perf_counter()

    token = int(toks[0])                       # the first generated token, confirmed
    last_hidden = H[:, -1:]                    # the state that produced it
    # What still has to be folded into the head's context before it can guess again: the
    # (hidden, next-token) pair of every confirmed position it has not yet seen.
    pending_head = [(last_hidden, token)]
    n = -1
    logprobs, from_draft = lps[0], False
    pending = [(token, lps[0], False, logits[0, -1])]
    while True:
        if not pending:
            # 1. GUESS. Fold the confirmed positions into the head and take its last output.
            hs = mx.concatenate([h_ for h_, _ in pending_head], axis=1)
            es = embed(mx.array([t_ for _, t_ in pending_head])[None])
            guess_lg = lm_head_of(tm)(head(hs, es, hcache))[0, -1]
            d = int(mx.argmax(guess_lg))
            pending_head = []
            st.rounds += 1
            st.drafted += 1
            # 2. VERIFY. One pass over [token, guess]; the target's row 0 is what really follows
            #    `token`, and row 1 is what follows the guess if the guess was right.
            snap = snapshot(cache)
            lg2, h2 = forward(tm, mx.array([token, d])[None], cache)
            t2, lp2 = _pick(sampler, lg2[0])
            mx.eval(t2)
            got0, got1 = int(t2[0]), int(t2[1])
            if got0 == d:
                st.accepted += 1
                # Both positions are the target's own choices: row 0 chose `d`, row 1 chose
                # what follows it. The head still owes an entry for `token`'s position (paired
                # with d) and for d's position (paired with got1).
                pending = [(d, lp2[0], True, lg2[0, 0]), (got1, lp2[1], False, lg2[0, 1])]
                pending_head = [(h2[:, 0:1], d), (h2[:, 1:2], got1)]
            else:
                # 3. MISS. Put every cache back and run the confirmed token on its own, so the
                #    state and the token are exactly what ordinary decoding would have produced.
                restore(cache, snap)
                st.recomputed += 1
                lg1, h1 = forward(tm, mx.array([token])[None], cache)
                t1, lp1 = _pick(sampler, lg1[0])
                mx.eval(t1)
                got = int(t1[0])
                pending = [(got, lp1[0], False, lg1[0, 0])]
                pending_head = [(h1[:, 0:1], got)]
        token, logprobs, from_draft, raw = pending.pop(0)
        if on_logits is not None:
            on_logits(raw)
        n += 1
        if token in tokenizer.eos_token_ids:
            break
        detok.add_token(token)
        if (n + 1) == max_tokens:
            break
        yield GenerationResponse(
            text=detok.last_segment, token=token, logprobs=logprobs, from_draft=from_draft,
            prompt_tokens=len(ids), prompt_tps=prompt_tps, generation_tokens=n + 1,
            generation_tps=(n + 1) / max(_t.perf_counter() - tic, 1e-9),
            peak_memory=mx.get_peak_memory() / 1e9, finish_reason=None)
        if n % 256 == 0:
            mx.clear_cache()
    detok.finalize()
    yield GenerationResponse(
        text=detok.last_segment, token=token, logprobs=logprobs, from_draft=from_draft,
        prompt_tokens=len(ids), prompt_tps=prompt_tps, generation_tokens=n + 1,
        generation_tps=(n + 1) / max(_t.perf_counter() - tic, 1e-9),
        peak_memory=mx.get_peak_memory() / 1e9,
        finish_reason="stop" if token in tokenizer.eos_token_ids else "length")
