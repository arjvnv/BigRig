"""Layer-major prefill (bigrig_engine/layered.py) must leave exactly the cache the chunked loop leaves.

WHY THIS FILE EXISTS
    Layer-major prefill reorders the work of reading a prompt so streamed experts come off the
    disk once per layer instead of once per chunk. Its whole claim is that only the order
    changes. This file holds it to that, bit for bit, on a small random Qwen3.5-MoE built in
    memory: the same architecture class as Qwen3.6-35B-A3B (Gated DeltaNet + full attention +
    routed experts), 4-bit quantized like the real model, no download. Every cache tensor and the
    next-token logits must be identical, then twelve greedy tokens decoded from each cache must
    match, over prompt lengths that hit the chunk-boundary edge cases.

Run: .venv/bin/python tests/test_layered.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import mlx.core as mx                                                    # noqa: E402
import mlx.nn as nn                                                      # noqa: E402
from mlx_lm.models import qwen3_5_moe                                    # noqa: E402

from bigrig_engine import layered                                        # noqa: E402

FAIL = []


def check(n, c, d=""):
    (print(f"  PASS  {n}") if c else (FAIL.append(n), print(f"  FAIL  {n}  {d}")))


def tiny_model():
    """8 layers (6 linear-attention, 2 full-attention), 8 experts top-2, 4-bit, bfloat16."""
    mx.random.seed(0)
    text = dict(model_type="qwen3_5_moe_text", hidden_size=64, intermediate_size=128,
                num_hidden_layers=8, num_attention_heads=4, num_key_value_heads=2, head_dim=64,
                vocab_size=512, linear_num_value_heads=4, linear_num_key_heads=2,
                linear_key_head_dim=16, linear_value_head_dim=16, linear_conv_kernel_dim=4,
                full_attention_interval=4, num_experts=8, num_experts_per_tok=2,
                moe_intermediate_size=32, shared_expert_intermediate_size=32)
    model = qwen3_5_moe.Model(qwen3_5_moe.ModelArgs.from_dict({"model_type": "qwen3_5_moe",
                                                               "text_config": text}))
    model.set_dtype(mx.bfloat16)
    nn.quantize(model, group_size=32, bits=4)
    mx.eval(model.parameters())
    return model


def chunked(model, tokens, step):
    """mlx_lm's generate_step prefill loop, verbatim in effect: chunks of `step`, all but the last token."""
    cache = model.make_cache()
    head = tokens[:-1]
    for i in range(0, len(head), step):
        model(mx.array(head[i:i + step])[None], cache=cache)
        mx.eval([c.state for c in cache])
    return cache


def layer_major(model, tokens, step):
    cache = model.make_cache()
    ok = layered.prefill(model, tokens[:-1], cache, step)
    return cache, ok


def same_state(a, b):
    sa, sb = [c.state for c in a], [c.state for c in b]
    def flat(st):
        return [x for part in st for x in (flat(part) if isinstance(part, (list, tuple)) else [part])]
    fa, fb = flat(sa), flat(sb)
    return len(fa) == len(fb) and all(
        (x is None and y is None) or (x is not None and y is not None and x.shape == y.shape
                                      and x.dtype == y.dtype and mx.array_equal(x, y).item())
        for x, y in zip(fa, fb))


def untouched(cache):
    """True if no layer of `cache` has absorbed a token (a fresh cache cannot report .state)."""
    return all(getattr(c, "keys", None) is None and all(x is None for x in getattr(c, "cache", []))
               for c in cache)


def greedy(model, cache, last, n=12):
    out, y = [], mx.array([[last]])
    for _ in range(n):
        logits = model(y, cache=cache)[0, -1]
        t = int(mx.argmax(logits).item())
        out.append(t)
        y = mx.array([[t]])
    return out


model = tiny_model()
mx.random.seed(1)

print("=" * 78)
print("1. SUPPORT IS DECLARED, NOT ASSUMED")
print("=" * 78)
check("Qwen3.5-MoE is supported", layered.text_model(model) is not None)


class NotQwen:
    pass


check("an unknown model is refused (caller falls back to mlx_lm)", layered.text_model(NotQwen()) is None)
check("refusal leaves the cache untouched", layered.prefill(NotQwen(), [1, 2, 3], [], 4) is False)

print("=" * 78)
print("2. BIT-IDENTICAL TO THE CHUNKED LOOP")
print("=" * 78)
for step in (16, 64, 128):
    # 2 tokens (one-token prefill), exactly one chunk, one chunk + 1, several chunks + a ragged tail
    for n in (2, step + 1, step + 2, 3 * step + 7):
        tokens = mx.random.randint(0, 512, (n,)).tolist()
        a = chunked(model, tokens, step)
        b, ok = layer_major(model, tokens, step)
        check(f"step {step:3d}, {n:3d} tokens: supported", ok)
        check(f"step {step:3d}, {n:3d} tokens: every cache tensor identical", same_state(a, b))
        la = model(mx.array([[tokens[-1]]]), cache=a)
        lb = model(mx.array([[tokens[-1]]]), cache=b)
        check(f"step {step:3d}, {n:3d} tokens: next-token logits identical", mx.array_equal(la, lb).item())
        check(f"step {step:3d}, {n:3d} tokens: 12 greedy tokens identical",
              greedy(model, a, int(mx.argmax(la[0, -1]).item())) == greedy(model, b, int(mx.argmax(lb[0, -1]).item())))

print("=" * 78)
print("3. THE TEST CAN FAIL: a different chunk size is NOT identical")
print("=" * 78)
tokens = mx.random.randint(0, 512, (3 * 64 + 7,)).tolist()
check("chunked at 64 vs layer-major at 16 differ (so 'identical' above means something)",
      not same_state(chunked(model, tokens, 64), layer_major(model, tokens, 16)[0]))

print("=" * 78)
print("4. ON TOP OF A CACHE THAT ALREADY HOLDS A PREFIX (a prompt-cache hit, then the rest)")
print("=" * 78)
for prefix_len, rest_len in ((16, 3 * 16 + 5), (21, 40), (64, 2)):
    tokens = mx.random.randint(0, 512, (prefix_len + rest_len,)).tolist()
    a, b = model.make_cache(), model.make_cache()
    for cache in (a, b):
        for i in range(0, prefix_len, 16):
            model(mx.array(tokens[i:min(i + 16, prefix_len)])[None], cache=cache)
        mx.eval([c.state for c in cache])
    rest = tokens[prefix_len:]
    for i in range(0, len(rest), 16):
        model(mx.array(rest[i:i + 16])[None], cache=a)
    mx.eval([c.state for c in a])
    check(f"prefix {prefix_len}, then {rest_len} tokens: supported", layered.prefill(model, rest, b, 16))
    check(f"prefix {prefix_len}, then {rest_len} tokens: every cache tensor identical", same_state(a, b))

print("=" * 78)
print("5. THE SESSION'S GUARDS (Session._read_layer_major, Session._prefill_layer_major)")
print("=" * 78)
from bigrig_engine import session as _session                            # noqa: E402

S = _session.Session.__new__(_session.Session)
S.model, S.kv_bits, S.prefill_step, S._pre_prefill, S.handle = model, None, 16, 0, None
S._gen_read = 0
tokens = mx.random.randint(0, 512, (2 * 16 + 5,)).tolist()
pc = model.make_cache()
check("a plain cache is read layer by layer", S._read_layer_major(pc, tokens[:-1], 16)
      and same_state(chunked(model, tokens, 16), pc))
from mlx_lm.generate import maybe_quantize_kv_cache                       # noqa: E402
from mlx_lm.models.cache import QuantizedKVCache                         # noqa: E402

nq = {}
for start in (0, 20, 40, 10_000):
    S.kv_bits, S.kv_quant_start = 4, start
    tokens = mx.random.randint(0, 512, (3 * 16 + 9,)).tolist()
    a = model.make_cache()
    for i in range(0, len(tokens) - 1, 16):            # the snapshot loop: chunk, quantize, eval
        model(mx.array(tokens[:-1][i:i + 16])[None], cache=a)
        maybe_quantize_kv_cache(a, quantized_kv_start=start, kv_group_size=_session.KV_GROUP_SIZE, kv_bits=4)
        mx.eval([c.state for c in a])
    b = model.make_cache()
    ok = S._read_layer_major(b, tokens[:-1], 16)
    kinds = lambda cache: [type(c).__name__ for c in cache]   # noqa: E731
    nq[start] = sum(isinstance(c, QuantizedKVCache) for c in a)
    check(f"KV quantized from token {start}: read layer by layer", ok)
    check(f"KV quantized from token {start}: the same layers end up quantized ({nq[start]})",
          kinds(a) == kinds(b), (kinds(a), kinds(b)))
    check(f"KV quantized from token {start}: every cache tensor identical", same_state(a, b))
    la, lb = model(mx.array([[tokens[-1]]]), cache=a), model(mx.array([[tokens[-1]]]), cache=b)
    check(f"KV quantized from token {start}: 12 greedy tokens identical",
          greedy(model, a, int(mx.argmax(la[0, -1]).item())) == greedy(model, b, int(mx.argmax(lb[0, -1]).item())))
check("the quantizing cases do quantize, and past the prompt nothing is (so 'identical' covers both)",
      all(nq[s_] > 0 for s_ in (0, 20, 40)) and nq[10_000] == 0, nq)
S.kv_bits = None
tokens = mx.random.randint(0, 512, (2 * 16 + 5,)).tolist()
check("no cache, or nothing to read, declines", S._read_layer_major(None, tokens, 16) is False
      and S._read_layer_major(model.make_cache(), [], 16) is False)
pc = model.make_cache()
left = S._prefill_layer_major(pc, list(tokens), {}, None)
check("generation's prefill reads all but the last token and leaves that one", left == tokens[-1:])
check("...building the chunked loop's cache", same_state(chunked(model, tokens, 16), pc))
check("...and counts what it read", S._pre_prefill == len(tokens) - 1)
check("...and reports it as prompt tokens read for generation (API usage)", S._gen_read == len(tokens) - 1)
pc = model.make_cache()
check("input embeddings (images) are left to mlx_lm",
      S._prefill_layer_major(pc, list(tokens), {"input_embeddings": None}, None) == tokens
      and untouched(pc))
saved, _session.LAYERED_PREFILL = _session.LAYERED_PREFILL, False
check("BIGRIG_LAYERED_PREFILL=0 turns it off", S._read_layer_major(model.make_cache(), tokens, 16) is False)
_session.LAYERED_PREFILL = saved

print("=" * 78)
print("6. THE ORDER: EVERY CHUNK THROUGH LAYER L BEFORE ANY REACHES LAYER L+1")
print("=" * 78)
tm = layered.text_model(model)
index = {id(layer): i for i, layer in enumerate(tm.layers)}
cls = type(tm.layers[0])
seen, orig = [], cls.__call__


def spy(self, x, *a, **k):
    seen.append((index[id(self)], x.shape[1]))
    return orig(self, x, *a, **k)


tokens = mx.random.randint(0, 512, (3 * 16 + 6,)).tolist()   # chunks of 16, 16, 16 and a last of 5
cls.__call__ = spy
try:
    layered.prefill(model, tokens[:-1], model.make_cache(), 16)
finally:
    cls.__call__ = orig
n = len(tm.layers)
want = [(li, m_) for li in range(n) for m_ in (16, 16, 16, 5)]
check("every chunk, in order, through each layer before the next layer", seen == want, seen[:6])

print("=" * 78)
print("6b. THE SESSION RE-WARMS THE PAGE CACHE AFTER A LONG PROMPT, AND ONLY THEN")
print("=" * 78)
import contextlib                                                        # noqa: E402


class FakeHandle:
    def __init__(self):
        self.rewarms = []

    def routes_to_most_experts(self, n):
        return n >= 30

    def prefill_layer(self, li, read_ahead=False):
        return contextlib.nullcontext()

    def rewarm(self, per_layer):
        self.rewarms.append(per_layer)
        return 0


class FakeHandle2(FakeHandle):
    def __init__(self):
        super().__init__()
        self.released = 0

    def release_prefill(self):
        self.released += 1


FakeHandle = FakeHandle2
S2 = _session.Session.__new__(_session.Session)
S2.model, S2.kv_bits, S2.prefill_step, S2._pre_prefill, S2.handle = model, None, 16, 0, FakeHandle()
S2._read_layer_major(model.make_cache(), mx.random.randint(0, 512, (40,)).tolist(), 16)
check("a long prompt is followed by one re-warm of REWARM_EXPERTS per layer",
      S2.handle.rewarms == [_session.REWARM_EXPERTS], S2.handle.rewarms)
S2._read_layer_major(model.make_cache(), mx.random.randint(0, 512, (12,)).tolist(), 16)
check("a short prompt is not", S2.handle.rewarms == [_session.REWARM_EXPERTS], S2.handle.rewarms)
S2._read_layer_major(model.make_cache(), mx.random.randint(0, 512, (40,)).tolist(), 64)
check("nor is one that fits in a single chunk (its order, and page cache, are the chunked loop's)",
      S2.handle.rewarms == [_session.REWARM_EXPERTS], S2.handle.rewarms)
check("unclaimed read-ahead is released after every layer-major read", S2.handle.released == 3,
      S2.handle.released)


def abort_after_two(done, total, _n=[0]):
    _n[0] += 1
    if _n[0] == 2:
        raise RuntimeError("client hung up")


S2.handle = FakeHandle()
try:
    S2._read_layer_major(model.make_cache(), mx.random.randint(0, 512, (40,)).tolist(), 16, abort_after_two)
    aborted = False
except RuntimeError:
    aborted = True
check("an abandoned prefill propagates the abort...", aborted)
check("...and still releases its read-ahead, with no re-warm", S2.handle.released == 1 and S2.handle.rewarms == [])
import inspect as _inspect                                               # noqa: E402
_st = _inspect.getsource(_session.Session.stream_text)
check("generation's prefill runs before the model class is patched (an abort cannot leave the patch)",
      _st.index("self._prefill_layer_major(pc, prompt_in") < _st.index("_mcls.__call__ = _capture"))


class BrokenHandle(FakeHandle):
    def rewarm(self, per_layer):
        raise RuntimeError("disk gone")


S2.handle = BrokenHandle()
pc = model.make_cache()
toks = mx.random.randint(0, 512, (40,)).tolist()
check("a failing re-warm does not fail the prompt: the cache is still read, and identical",
      S2._read_layer_major(pc, toks, 16) and same_state(chunked(model, toks + [0], 16), pc))

print("=" * 78)
print("6c. AGAINST MLX_LM ITSELF: generate_step on the whole prompt vs layer-major + generate_step")
print("=" * 78)
from mlx_lm.generate import generate_step                                # noqa: E402

for kv in (None, 4):
    for n in (37, 130):
        toks = mx.random.randint(0, 512, (n,)).tolist()
        kw = {"kv_bits": kv, "kv_group_size": _session.KV_GROUP_SIZE, "quantized_kv_start": 20} if kv else {}
        ref = [int(t) for t, _ in generate_step(mx.array(toks), model, max_tokens=12, prefill_step_size=16,
                                               prompt_cache=model.make_cache(), **kw)]
        S.kv_bits, S.kv_quant_start, S.handle = kv, 20, None
        pc = model.make_cache()
        left = S._prefill_layer_major(pc, list(toks), {}, 16)
        got = [int(t) for t, _ in generate_step(mx.array(left), model, max_tokens=12, prefill_step_size=16,
                                               prompt_cache=pc, **kw)]
        check(f"{n} tokens, KV {'4-bit from 20' if kv else 'full precision'}: 12 tokens identical to mlx_lm's own loop",
              got == ref, (got, ref))
S.kv_bits = None

print("=" * 78)
print("6d. DECLINED, NEVER RAISED, WHEN THE SCHEDULE IS NOT KNOWN TO BE EXACT")
print("=" * 78)
print(f"        mlx_lm's Qwen3_5TextModel.__call__ hashes to {layered._forward_source(layered.text_model(model).__class__)}")
saved = layered._KNOWN_FORWARD
layered._KNOWN_FORWARD = "a different forward pass"
check("a changed mlx_lm forward body is declined (mlx_lm's own loop reads the prompt)", layered.text_model(model) is None)
layered._KNOWN_FORWARD = saved
check("...and the installed one is accepted", layered.text_model(model) is not None)
tm = layered.text_model(model)
flags = [layer.is_linear for layer in tm.layers]
for layer in tm.layers:
    layer.is_linear = True
check("a stack with no full-attention layer is declined, not an exception", layered.text_model(model) is None)
for layer, f_ in zip(tm.layers, flags):
    layer.is_linear = f_

print("=" * 78)
print("7. layer_scope: ONE BLOCK PER LAYER, IN ORDER, AND NO EFFECT ON THE CACHE")
print("=" * 78)
events = []


class Scope:
    def __init__(self, li):
        self.li = li

    def __enter__(self):
        events.append(("in", self.li))

    def __exit__(self, *exc):
        events.append(("out", self.li))


tokens = mx.random.randint(0, 512, (2 * 16 + 5,)).tolist()
scoped = model.make_cache()
check("prefill with a layer_scope is supported",
      layered.prefill(model, tokens[:-1], scoped, 16, layer_scope=Scope))
check("each layer's block opens and closes before the next layer's opens",
      events == [(s, li) for li in range(8) for s in ("in", "out")], events[:6])
check("the scope changes nothing in the cache", same_state(chunked(model, tokens, 16), scoped))

print("\n" + "=" * 78)
print(f"{'ALL TESTS PASSED' if not FAIL else str(len(FAIL)) + ' FAILURES: ' + ', '.join(FAIL)}")
print("=" * 78)
sys.exit(1 if FAIL else 0)
