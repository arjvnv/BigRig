"""Adversarial tests for the Anthropic API, `rig launch`, and the web interface.

These three exist so someone can actually USE the thing: point Claude Code at a local model,
or open a browser and see whether the model is degrading. Their failure modes are all the same
shape -- a client connects, sends a request, and then hangs or silently gets nothing -- so the
tests check the shapes real clients send, not the shapes that were convenient to build.
"""
import inspect
import json
from collections import deque
import re

import numpy as _np
import os
import sys
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT_DIR = ROOT
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bigrig_engine import anthropic as anth
from bigrig_engine import autoconfig, launch, server, session

FAIL = []
def check(n, c, d=""):
    (print(f"  PASS  {n}") if c else (FAIL.append(n), print(f"  FAIL  {n}  {d}")))


def _raises(fn, exc):
    """True if fn() raises exc. Rejecting a bad request is behaviour worth asserting, and a bare
    try/except around each one buries the assertion in boilerplate."""
    try:
        fn()
    except exc:
        return True
    except Exception:
        return False
    return False

OK = {"model": "x", "max_tokens": 16, "messages": [{"role": "user", "content": "hi"}]}

print("=" * 84); print("1. THE ANTHROPIC REQUEST SHAPE"); print("=" * 84)
p = anth.parse(OK)
check("a minimal request parses", p["messages"] == [{"role": "user", "content": "hi"}])
check("...with Anthropic's temperature default", p["temperature"] == 0.7)

# `system` is a TOP-LEVEL field here, not a message. Getting this wrong drops the system prompt.
p2 = anth.parse({**OK, "system": "Be terse."})
check("system is read from the top level", p2["system"] == "Be terse.")
check("...and becomes a leading system message for the template",
      anth.to_engine_messages(p2)[0] == {"role": "system", "content": "Be terse."})
check("no system field means no injected message", len(anth.to_engine_messages(p)) == 1)
p3 = anth.parse({**OK, "system": [{"type": "text", "text": "A"}, {"type": "text", "text": "B"}]})
check("system may itself be a list of blocks", p3["system"] == "AB", p3["system"])

# content is a string OR a list of typed blocks.
b = anth.parse({**OK, "messages": [{"role": "user", "content": [
    {"type": "text", "text": "one "}, {"type": "text", "text": "two"}]}]})
check("content blocks are flattened in order", b["messages"][0]["content"] == "one two")
img = anth.parse({**OK, "messages": [{"role": "user", "content": [
    {"type": "image", "source": {}}, {"type": "text", "text": "describe"}]}]})
check("a block this engine cannot render is skipped, not rejected",
      img["messages"][0]["content"] == "describe")
tr = anth.parse({**OK, "messages": [{"role": "user", "content": [
    {"type": "tool_result", "content": [{"type": "text", "text": "42"}]}]}]})
check("tool_result content is flattened rather than dropped",
      tr["messages"][0]["content"] == "42")

for bad, why in (({k: v for k, v in OK.items() if k != "max_tokens"}, "no max_tokens"),
                 ({**OK, "max_tokens": 0}, "max_tokens of 0"),
                 ({**OK, "max_tokens": True}, "a boolean max_tokens"),
                 ({**OK, "max_tokens": "16"}, "a string max_tokens"),
                 ({**OK, "messages": []}, "no messages"),
                 ({**OK, "messages": "hi"}, "messages that are not a list"),
                 ({**OK, "messages": [{"role": "system", "content": "x"}]}, "role system"),
                 ({**OK, "messages": [{"role": "user"}]}, "a message with no content"),
                 ({**OK, "temperature": 1.5}, "temperature above Anthropic's range"),
                 ({**OK, "temperature": "warm"}, "a non-numeric temperature"),
                 ({**OK, "stop_sequences": "x"}, "stop_sequences that are not a list")):
    try:
        anth.parse(bad)
        check(f"rejects {why}", False, "accepted")
    except anth.BadRequest:
        check(f"rejects {why}", True)
try:
    anth.parse({**OK, "messages": [{"role": "system", "content": "x"}]})
except anth.BadRequest as e:
    check("...and the role error points at the `system` field", "system" in str(e).lower())
# The count endpoint has nothing to do with reply length, so it must not demand max_tokens.
c = anth.parse({k: v for k, v in OK.items() if k != "max_tokens"}, require_max_tokens=False)
check("count_tokens does not require max_tokens", c["messages"][0]["content"] == "hi")
check("OpenAI's wider temperature range is NOT used here",
      "1.0" in inspect.getsource(anth.parse) and "2.0" not in inspect.getsource(anth.parse))

print("\n" + "=" * 84); print("2. THE ANTHROPIC RESPONSE SHAPE"); print("=" * 84)
m = anth.message("msg_1", "mymodel", "hello", 5, 2, "stop", {"mode": "native"})
check("type is 'message'", m["type"] == "message")
check("role is assistant", m["role"] == "assistant")
check("content is a LIST of blocks, not a string",
      isinstance(m["content"], list) and m["content"][0] == {"type": "text", "text": "hello"})
check("usage names input_tokens / output_tokens",
      m["usage"] == {"input_tokens": 5, "output_tokens": 2})
check("there is no OpenAI `choices` array", "choices" not in m)
check("stop_reason maps 'stop' to end_turn", m["stop_reason"] == "end_turn")
check("...and 'length' to max_tokens", anth.stop_reason("length") == "max_tokens")
check("...and an unknown reason does not crash", anth.stop_reason("weird") == "end_turn")
check("our extras ride along under a namespaced key", m["bigrig"]["mode"] == "native")

print("\n" + "=" * 84); print("3. ANTHROPIC STREAMING FRAMES"); print("=" * 84)
f = anth.sse("content_block_delta", {"a": 1})
check("every frame carries an `event:` line -- their SDK dispatches on it",
      f.startswith(b"event: content_block_delta\ndata: "))
check("...and terminates with a blank line", f.endswith(b"\n\n"))
starts = anth.start_frames("msg_1", "m", 7)
ends = anth.end_frames(12, "stop")
names = [x.split(b"\n")[0].decode().removeprefix("event: ")
         for x in starts + [anth.delta_frame("hi")] + ends]
check("the frame sequence is exactly what the protocol specifies",
      names == ["message_start", "content_block_start", "content_block_delta",
                "content_block_stop", "message_delta", "message_stop"], str(names))
ms = json.loads(starts[0].split(b"data: ")[1])
check("message_start reports input tokens", ms["message"]["usage"]["input_tokens"] == 7)
check("...and starts with empty content", ms["message"]["content"] == [])
md = json.loads(ends[1].split(b"data: ")[1])
check("message_delta carries the output count", md["usage"]["output_tokens"] == 12)
check("...and the stop reason", md["delta"]["stop_reason"] == "end_turn")

# A MODEL THAT THINKS FIRST PRODUCES TWO BLOCKS, NOT ONE BLOCK WITH TWO KINDS OF DELTA.
#     Anthropic names the reasoning delta `thinking_delta` and carries it in a `thinking` field.
#     A client switching on the delta type drops reasoning sent as text; one rendering it as text
#     shows the scratchpad as the answer, which is what the split exists to prevent. Before this,
#     the streamed Anthropic path dropped the reasoning entirely -- and that is the endpoint
#     `bigrig launch` points coding agents at.
_tstart = anth.start_frames("msg_2", "m", 5, "thinking")
_cb = json.loads(_tstart[1].split(b"data: ")[1])["content_block"]
check("the first block can open as thinking", _cb == {"type": "thinking", "thinking": ""}, str(_cb))
check("...and still opens as text by default",
      json.loads(anth.start_frames("m", "m", 1)[1].split(b"data: ")[1])["content_block"]
      == {"type": "text", "text": ""})
_td = json.loads(anth.delta_frame("weighing it", 0, "thinking").split(b"data: ")[1])
check("a thinking delta is named and carried the way the protocol says",
      _td["delta"] == {"type": "thinking_delta", "thinking": "weighing it"}, str(_td["delta"]))
check("...and a text delta is unchanged",
      json.loads(anth.delta_frame("hi").split(b"data: ")[1])["delta"]
      == {"type": "text_delta", "text": "hi"})
check("a delta can be addressed to a block other than 0, so text can follow thinking",
      json.loads(anth.delta_frame("hi", 1).split(b"data: ")[1])["index"] == 1)
check("blocks open and close by index",
      json.loads(anth.block_start(3, "text").split(b"data: ")[1])["index"] == 3
      and json.loads(anth.block_stop(3).split(b"data: ")[1])["type"] == "content_block_stop")
# The whole sequence a thinking reply produces, as the handler emits it.
_seq = (anth.start_frames("msg_3", "m", 5, "thinking")
        + [anth.delta_frame("thinking...", 0, "thinking"), anth.block_stop(0),
           anth.block_start(1, "text"), anth.delta_frame("4", 1)]
        + anth.end_frames(9, "stop", index=1))
_names = [x.split(b"\n")[0].decode().removeprefix("event: ") for x in _seq]
check("the thinking sequence is exactly what the protocol specifies",
      _names == ["message_start", "content_block_start", "content_block_delta",
                 "content_block_stop", "content_block_start", "content_block_delta",
                 "content_block_stop", "message_delta", "message_stop"], str(_names))
_opened = [json.loads(x.split(b"data: ")[1])["index"] for x in _seq
           if b"content_block_start" in x.split(b"\n")[0]]
_closed = [json.loads(x.split(b"data: ")[1])["index"] for x in _seq
           if b"content_block_stop" in x.split(b"\n")[0]]
check("...every block it opens is closed exactly once, indices consecutive from 0",
      _opened == [0, 1] and sorted(_closed) == [0, 1], f"{_opened} / {_closed}")
_srv = open(os.path.join(ROOT, "bigrig_engine", "server.py"), encoding="utf-8").read()
check("the streamed handler counts from the open block rather than assuming 0",
      "anth.block_stop(idx)" in _srv and '"index": 0}))' not in _srv.split("Each call is its OWN")[1][:400])
check("...and the blocking reply carries the reasoning as a thinking block",
      'reasoning="".join(reasoning)' in _srv
      and '{"type": "thinking", "thinking": reasoning}' in
      open(os.path.join(ROOT, "bigrig_engine", "anthropic.py"), encoding="utf-8").read())

print("\n" + "=" * 84); print("4. TOKENS PER SECOND MUST NEVER BE ABSURD"); print("=" * 84)
# Measured: mlx_lm reported 31,128 tok/s on the first chunk, because elapsed time is ~0 there.
# Rendering that anywhere a person can see makes every other number untrustworthy.
check("a rate from the first token is withheld", session._sane_tps(31128.4, 1) is None)
check("...and from the second and third", session._sane_tps(9000.0, 2) is None)
check("a plausible rate survives once there is enough to measure",
      session._sane_tps(150.4, 20) == 150.4)
check("an impossible rate is withheld however many tokens",
      session._sane_tps(90000.0, 500) is None)
check("None in, None out", session._sane_tps(None, 500) is None)
check("a non-numeric rate does not crash", session._sane_tps("fast", 500) is None)
check("zero and negative rates are withheld",
      session._sane_tps(0.0, 500) is None and session._sane_tps(-5.0, 500) is None)
# Behaviour at the boundary, not the wording of the comment above it. An earlier version of
# this test grepped the source for a particular word and failed when the word moved.
check("the boundary is exactly at the documented ceiling",
      session._sane_tps(session.MAX_PLAUSIBLE_TOK_S, 500) == session.MAX_PLAUSIBLE_TOK_S
      and session._sane_tps(session.MAX_PLAUSIBLE_TOK_S + 1, 500) is None)
check("...and at the documented token minimum",
      session._sane_tps(150.0, session.MIN_TOKENS_FOR_RATE) == 150.0
      and session._sane_tps(150.0, session.MIN_TOKENS_FOR_RATE - 1) is None)
check("the web UI clamps it too, in case one ever gets through",
      "t.tok_s<=2000" in re.sub(r"\s+", "", open(
          os.path.join(ROOT, "bigrig_engine/webui.html")).read()))

print("\n" + "=" * 84)
print("4b. A STREAM MUST TELL THE CLIENT WHERE IT ENDS")
print("=" * 84)
# THE BUG: SSE responses were sent with `Connection: keep-alive` and no Content-Length and no
# chunked encoding, so a browser's fetch reader could never see the body end. The page sent one
# message, waited forever for a `done` that never came, and the input stayed disabled. curl
# never noticed because it reads until the socket closes.
_sv = inspect.getsource(server.make_handler)
# Slice to end_headers(), which is where the header block genuinely ends. A fixed character
# window failed here because the explanation above the line is longer than the window was.
_sse_blocks = [b.split("end_headers()")[0] for b in _sv.split("text/event-stream")[1:]]
check("every SSE response declares Connection: close",
      all('send_header("Connection", "close")' in b for b in _sse_blocks),
      f"{len(_sse_blocks)} SSE handlers")
check("...and actually closes the socket, not just says so",
      all("self.close_connection = True" in b for b in _sse_blocks))
# Three now: OpenAI, Anthropic, and the Responses API that Codex CLI requires. Each one is a
# separate handler, so each has to declare and honour Connection: close on its own.
check("all three streams -- OpenAI, Anthropic and Responses -- are covered",
      len(_sse_blocks) == 3,
      str(len(_sse_blocks)))
check("the reason is recorded where the next person will look",
      "input box stays disabled after the first message" in _sv)
# Blocking replies use Content-Length instead, which is the other valid way to end a body.
check("blocking replies declare a Content-Length",
      'send_header("Content-Length"' in inspect.getsource(server._json))

print("\n" + "=" * 84)
print("4c. A MODEL MUST NOT WRITE BOTH SIDES OF THE CONVERSATION")
print("=" * 84)
# Observed: OLMoE-1B-7B-0125-4bit ships NO chat template, so the prompt falls back to a plain
# "user:/assistant:" format and the base model simply continues the pattern -- inventing the
# user's next three questions and answering them, all presented as one reply.
check("turn boundaries are treated as stop sequences",
      hasattr(session, "TURN_STOPS") and len(session.TURN_STOPS) >= 4)
check("...covering both capitalisations a template fallback produces",
      any(x == "\nUser:" for x in session.TURN_STOPS)
      and any(x == "\nassistant:" for x in session.TURN_STOPS))
_st = inspect.getsource(session.Session.stream_text)
check("output that might be the start of a stop sequence is held back, not emitted then retracted",
      "hold" in _st and "could still turn out to be the start" in _st)
check("the stops are applied only when the model has no template of its own",
      "if not self.has_chat_template" in _st)
check("a model's ability to chat is detected once, at load",
      hasattr(session.Session, "_detect_chat_template"))
check("...and reported, because a fabricated dialogue is indistinguishable from a bad answer",
      '"chat_template": self.has_chat_template' in inspect.getsource(session.Session.stats))
_ui = open(os.path.join(ROOT, "bigrig_engine/webui.html")).read()
# Greps against source text kept failing on line wrapping and spacing rather than on behaviour --
# five times in this suite's history. Match against normalised copies instead: _uic with all
# whitespace stripped for code, _uip with runs collapsed to one space for prose.
_uic = re.sub(r"\s+", "", _ui)
_uip = " ".join(_ui.split())
check("the interface warns when a model was not built for conversation",
      "ships no chat template" in _uip)

print("\n" + "=" * 84)
print("4d. THE DASHBOARD MUST NOT INVENT NUMBERS")
print("=" * 84)
check("an absent value renders as an em dash, never as zero",
      'constNA="—"' in _uic and "v==null?NA" in _uic)
# Every set() call site must pass something that can be null rather than a bare server field,
# because set() renders null as an em dash and a bare `h.foo` for a field this model does not
# report renders "undefined". Checking the call sites is what the four hardcoded string greps
# that used to live here were reaching for, and it does not break when a guard is reworded.
_sets = re.findall(r'set\("#[a-z0-9-]+",\s*([^;]+?)\);', " ".join(_ui.split()))
_bare = [e for e in _sets
         if re.fullmatch(r"[hH]\.[a-z_]+", e.strip())]
check("no dashboard field is printed straight from the server without a null guard",
      not _bare, f"unguarded: {_bare[:3]}" if _bare else "")
check("...and there are call sites to check at all", len(_sets) >= 10)
check("an absent field renders as an em dash rather than the word undefined",
      "v==null?NA" in _uic and 'NA="—"' in _uic)
check("cache statistics are hidden entirely when nothing is being streamed",
      '#sec-cache' in _ui and 'h.mode==="stream"' in _uic)
check("the token rate is still clamped to something believable", "t.tok_s<=2000" in _uic)
check("time-to-first-token is measured in the page, not taken on trust",
      "performance.now()" in _ui and "ttft" in _ui)
check("every dashboard field has a server field behind it",
      all(k in _ui for k in ("requests_served", "queue_depth", "uptime_s", "flagged_tokens")))

print("\n" + "=" * 84)
print("4e. LIMITS COME FROM THE MODEL, NOT FROM A NUMBER SOMEONE PICKED")
print("=" * 84)
_ui = open(os.path.join(ROOT, "bigrig_engine/webui.html")).read()
_ss = inspect.getsource(session.Session._model_limits)
check("the context length is read from the checkpoint",
      "max_position_embeddings" in _ss and "max_seq_len" in _ss)
check("...and a checkpoint that does not say gets zero, not a guess",
      "return 0, 0" in _ss)
# RUN the estimator rather than read it. Every number below was checked against what mlx_lm's
# own cache objects allocate for a real published config -- see the family sweep further down.
_kv = session.kv_bytes
# Qwen3-30B: 48 layers x 4 kv heads x 128 dims x 2 (k+v) x 2 bytes = 98,304.
check("the KV cost per token is computed from the real attention shape",
      _kv({"num_hidden_layers": 48, "num_key_value_heads": 4, "head_dim": 128,
           "num_attention_heads": 32, "hidden_size": 4096}) == (98304, 0))
check("...and head_dim is derived when the config does not state it",
      _kv({"num_hidden_layers": 2, "num_attention_heads": 4,
           "hidden_size": 64}) == (2 * 4 * 16 * 2 * 2, 0))
# DeepSeek-V3 caches one 512-wide latent and one 64-wide rope key per layer, not per head.
check("multi-head latent attention is priced as a latent, not as per-head keys and values",
      _kv({"num_hidden_layers": 61, "kv_lora_rank": 512, "qk_rope_head_dim": 64,
           "num_key_value_heads": 128, "num_attention_heads": 128,
           "hidden_size": 7168}) == (61 * (512 + 64) * 2, 0))
check("...which is 24.9x cheaper than treating it as ordinary attention",
      round((61 * 128 * 56 * 2 * 2) / (61 * (512 + 64) * 2), 1) == 24.9)
# Qwen3-Next: only every 4th layer holds a KV cache; the rest are gated delta nets.
check("linear-attention layers are not charged for a cache they do not have",
      _kv({"num_hidden_layers": 48, "full_attention_interval": 4, "num_key_value_heads": 2,
           "head_dim": 256, "num_attention_heads": 16})[0] == 12 * 2 * 256 * 2 * 2)
# gpt-oss: sliding layers stop growing at the window, so they are a fixed cost.
check("sliding-window layers are charged once at their window, not per token",
      _kv({"num_hidden_layers": 4, "num_key_value_heads": 8, "head_dim": 64,
           "num_attention_heads": 64, "sliding_window": 128,
           "layer_types": ["full_attention", "sliding_attention"] * 2})
      == (2 * 8 * 64 * 2 * 2, 2 * 8 * 64 * 2 * 2 * 128))
# Phi-3.5-MoE and Qwen1.5-MoE both declare a sliding_window and neither overrides make_cache.
check("a sliding_window field alone does not make a layer windowed",
      _kv({"num_hidden_layers": 32, "num_key_value_heads": 8, "head_dim": 96,
           "num_attention_heads": 32, "sliding_window": 262144})
      == (32 * 8 * 96 * 2 * 2, 0))
check("a malformed config cannot crash the load",
      _kv({}) == (0, 0) and _kv({"num_hidden_layers": "x"}) == (0, 0)
      and _kv({"num_hidden_layers": 4}) == (0, 0))
check("...and neither can one whose attention shape is nonsense",
      _kv({"num_hidden_layers": 4, "num_attention_heads": 0, "hidden_size": 8}) == (0, 0))
check("the config estimate is a plan; the loaded model is asked directly",
      "make_cache" in inspect.getsource(session.Session._measure_kv))
check("...and a failure to ask keeps the estimate rather than losing the load",
      "return" in inspect.getsource(session.Session._measure_kv).split("except")[-1])
check("the reply-limit menu is built from what the server will accept, not hardcoded",
      "health.max_completion_tokens" in _uic)
check("...so every option in it is one the server would allow", "filter(v=>v<cap)" in _uic)
check("...and the ceiling itself is always offered", "opts.push(cap)" in _uic)
check("the cost of a limit is shown before it is chosen",
      "GB of KV cache if it runs to the limit" in _ui and "kv_bytes_per_token" in _ui)
check("...in both memory and time", "at ${tps} tok/s" in _ui or "tok/s`" in _ui)

print("\n" + "=" * 84)
print("4x. METAL COMMAND BUFFERS ARE SIZED FOR A LAYER, NOT FOR MLX'S DEFAULT")
print("=" * 84)
import bigrig_engine as _pkg2                                           # noqa: E402
_init = inspect.getsource(_pkg2)
# 1.56x, byte-identical, from one environment variable. Every streamed layer reads its router
# back to the host and that read waits for whatever is queued; MLX's default queues far more
# than a layer's worth, so each of the 48 reads a token drains work for layers not yet asked
# about. Measured 14 samples a value, twice each in interleaved order: default 12.96 tok/s,
# 8 ops 20.18, and the spread falls from 2.44 to 0.82.
check("the package sets a command-buffer size at import",
      "MLX_MAX_OPS_PER_BUFFER" in _init)
check("...to the value that measured fastest", 'setdefault("MLX_MAX_OPS_PER_BUFFER", "8")'
      in _init.replace("'", '"'))
check("...with setdefault, so a user's own value survives", "setdefault" in _init)
check("...and the curve that chose it is written down",
      "20.18" in _init and "12.96" in _init and "unimodal" in _init.lower())
check("it is actually in the environment once the package is imported",
      os.environ.get("MLX_MAX_OPS_PER_BUFFER") == "8",
      f"got {os.environ.get('MLX_MAX_OPS_PER_BUFFER')}")
# It schedules work; it cannot change what the work computes. If that ever stops being true the
# comment above is a lie and the win is not free.
check("...and it is a scheduling knob, so the note says output is unchanged",
      "byte-identical" in _init)

print("\n" + "=" * 84)
print("4y. THE STREAMING API MUST ACTUALLY STREAM")
print("=" * 84)
import ast as _ast                                                      # noqa: E402
_sess_src = open(os.path.join(ROOT_DIR, "bigrig_engine", "session.py")).read()
_fn = next(n for n in _ast.walk(_ast.parse(_sess_src))
           if isinstance(n, _ast.FunctionDef) and n.name == "stream_text")
_tries = [t for t in _ast.walk(_fn) if isinstance(t, _ast.Try)]
_in_finally = sum(1 for t in _tries for b in t.finalbody
                  for x in _ast.walk(b) if isinstance(x, _ast.Yield))
_in_loop = sum(1 for l in _ast.walk(_fn) if isinstance(l, _ast.For)
               for b in l.body for x in _ast.walk(b) if isinstance(x, _ast.Yield))
# THE DEFECT THIS PINS: every yield sat in the `finally` block, so the generator produced
# nothing until generation had finished and then delivered the whole reply in ONE chunk.
# Measured before the fix: a 32-token reply arrived as a single yield, and five test replies
# came back as [1, 1, 1, 1, 1] chunks. Every client of this server -- the web interface, the
# OpenAI endpoint, the Anthropic endpoint -- was therefore not streaming, which at single-digit
# tokens a second is forty seconds of blank screen followed by a wall of text.
check("no token is emitted from a finally block", _in_finally == 0,
      f"{_in_finally} yields in finally")
check("...they are emitted from the generation loop, as tokens arrive", _in_loop >= 4,
      f"{_in_loop} yields in the loop")
check("the finally block still restores what it patched",
      "_mcls.__call__ = _orig_call" in _sess_src)
check("...and says why producing output does not belong there",
      "Restoration belongs in `finally`" in _sess_src)

print("\n" + "=" * 84)
print("4z. AN INSTALLED PACKAGE MUST NOT STORE 60 GB OF WEIGHTS IN site-packages")
print("=" * 84)
import bigrig_engine as _pkg                                            # noqa: E402
_src = inspect.getsource(_pkg.home)
check("BIGRIG_HOME wins outright", "BIGRIG_HOME" in _src)
# Deriving from __file__ unconditionally put models inside site-packages, where reinstalling the
# package or deleting the virtualenv destroys them. Verified in a clean install before publishing.
check("a source checkout is recognised by its pyproject.toml, not assumed",
      "pyproject.toml" in _src)
check("...and anything else falls back to a real user directory",
      "~/.bigrig" in _src)
check("running in this checkout resolves to this checkout",
      _pkg.home() == ROOT_DIR, f"{_pkg.home()} vs {ROOT_DIR}")
_prev = os.environ.get("BIGRIG_HOME")
os.environ["BIGRIG_HOME"] = "/tmp/bigrig-home-check"
check("...and the environment variable overrides even that",
      _pkg.home() == "/tmp/bigrig-home-check")
if _prev is None:
    del os.environ["BIGRIG_HOME"]
else:
    os.environ["BIGRIG_HOME"] = _prev
check("every data path in the engine derives from home(), none spell the product name",
      not any("~/bigrig" in open(os.path.join(ROOT_DIR, "bigrig_engine", f)).read()
              for f in os.listdir(os.path.join(ROOT_DIR, "bigrig_engine"))
              if f.endswith(".py") and f != "__init__.py"))

print("\n" + "=" * 84)
print("4m. THE REPLY CEILING IS DERIVED, NOT DECLARED")
print("=" * 84)
# THE BUG: max_tokens was clamped to a hardcoded 32768 on both API paths. That was simultaneously
# too low -- Qwen3-30B has a 40,960-token context -- and dangerously high: at 96 KB per token a
# 32,768-token reply needs 3.2 GB of KV cache, which on top of 7.73 GB of resident weights blows
# a 9 GB ceiling. Neither number was connected to the machine it was running on.
class _Sess:
    """Enough of a Session to exercise the ceiling, with no model to load."""
    def __init__(self, ctx, kv, budget, foot, working=None, kv_bits=None):
        self.context_length, self.kv_bytes_per_token = ctx, kv
        self.budget_gb, self.footprint_gb = budget, foot
        # Compressing the conversation cache changes what a token of reply costs, so the ceiling
        # reads it. None is what the engine ships; the trade is measured in test_ceiling.py.
        self.kv_bits, self.kv_quant_start = kv_bits, session.KV_QUANT_START
        # Per model, not a constant: a step's transient cost scales with what it may have to
        # fetch, and gpt-oss's experts are six times the size of Qwen3's.
        self.working_memory_gb = session.WORKING_MEMORY_GB if working is None else working
    ceiling = session.Session._token_ceiling
_QWEN = dict(ctx=40960, kv=98304, budget=9.0, foot=5.03)
_lim, _why = _Sess(**_QWEN).ceiling()
check("memory binds before the context window on the machine this was measured on",
      _why == "memory", _why)
# And the whole point of compressing the cache. Whether the MODEL's window ends up binding
# depends on the footprint -- on the shipped 4.18 GB it does, and 18,514 becomes the full 40,960
# -- so what is asserted here is the thing that is true at any footprint: a token of reply costs
# less, so more of them fit. Roughly threefold, which matches the 3.56x measured on real caches
# less the fp16 prefix that stays uncompressed.
_qlim, _qwhy = _Sess(**_QWEN, kv_bits=4).ceiling()
check("...and compressing the cache buys about three times the reply",
      2.5 < _qlim / _lim < 4.0, f"{_lim} ({_why}) -> {_qlim} ({_qwhy})")
check("...capped by the model's own window, never beyond it",
      _Sess(ctx=4096, kv=98304, budget=9.0, foot=4.18, kv_bits=4).ceiling() == (4096, "context"))
check("...and the figure is exactly the arithmetic, not a rounded guess",
      _lim == int((9.0 - 5.03 - session.WORKING_MEMORY_GB) * 1e9 // 98304), str(_lim))
check("...which is far below the 32,768 that used to be allowed", _lim < 32768, str(_lim))
_lim2, _why2 = _Sess(ctx=40960, kv=98304, budget=24.0, foot=7.73).ceiling()
check("give it room and the model's own context window takes over",
      (_lim2, _why2) == (40960, "context"), f"{_lim2} {_why2}")
check("...which is more than the old hardcoded ceiling ever allowed", _lim2 > 32768)
check("a smaller pool raises the ceiling, because KV has more room",
      _Sess(ctx=40960, kv=98304, budget=9.0, foot=3.80).ceiling()[0] > _lim)
check("a checkpoint that does not state a context gets the stated assumption, not a guess",
      _Sess(ctx=0, kv=0, budget=9.0, foot=1.0).ceiling() == (session.ASSUMED_CONTEXT, "context"))
check("an unknown KV cost cannot silently produce a memory limit",
      _Sess(ctx=8192, kv=0, budget=9.0, foot=8.9).ceiling() == (8192, "context"))
check("memory so tight the arithmetic goes negative still returns a usable floor",
      _Sess(ctx=40960, kv=98304, budget=9.0, foot=9.5).ceiling()
      == (session.MIN_REPLY_TOKENS, "memory"))
# The measured step-128 prefill peaks ran 1.29 to 2.44 GB above the resident weights; the
# reserve has to clear the largest of those and stay well under the budget it is carved from.
check("the reserve clears the largest prefill peak that was measured",
      2.44 < session.WORKING_MEMORY_GB < 3.5, str(session.WORKING_MEMORY_GB))

# Both API paths must honour it -- they had separate copies of the old constant.
_S = server.make_handler(server._State(None))._sampling
check("the OpenAI path rejects a request above the ceiling",
      _raises(lambda: _S({"max_tokens": 5000}, 4780, "memory"), ValueError))
check("...and accepts one at exactly the ceiling",
      _S({"max_tokens": 4780}, 4780, "memory")["max_tokens"] == 4780)
try:
    _S({"max_tokens": 5000}, 4780, "memory")
except ValueError as e:
    _msg = str(e)
check("...and the refusal says what the ceiling is", "4780" in _msg, _msg)
check("...and why it is that, so the caller can do something about it",
      "memory budget leaves for KV cache" in _msg, _msg)
try:
    _S({"max_tokens": 99999}, 40960, "context")
except ValueError as e:
    _msg2 = str(e)
check("...and names the other bound when that is the one in force",
      "context window" in _msg2, _msg2)
check("the Anthropic path takes the same ceiling",
      _raises(lambda: anth.parse({"messages": [{"role": "user", "content": "x"}],
                                  "max_tokens": 5000}, max_tokens_limit=4780), anth.BadRequest))
check("...and accepts what fits",
      anth.parse({"messages": [{"role": "user", "content": "x"}], "max_tokens": 4780},
                 max_tokens_limit=4780)["max_tokens"] == 4780)
check("...and falls back to the fixed ceiling when no session is available",
      _raises(lambda: anth.parse({"messages": [{"role": "user", "content": "x"}],
                                  "max_tokens": anth.MAX_TOKENS_LIMIT + 1}), anth.BadRequest))
check("no hardcoded 32768 is left deciding anything on either path",
      "32768" not in inspect.getsource(server.make_handler))

print("\n" + "=" * 84)
print("4n. THE MEMORY CEILING IS ACTUALLY THE CEILING")
print("=" * 84)
# THE GAP: BIGRIG_MEM_GB was read only by src/memguard.py, which the engine never armed. So
# `BIGRIG_MEM_GB=9 rig serve` planned against whatever the machine had free -- 13.6 GB of a
# 24 GB Mac -- and stayed under 9 GB only because --residency happened to land there.
_init = inspect.getsource(session.Session.__init__)
check("the engine reads the budget from the environment",
      'os.environ.get("BIGRIG_MEM_GB")' in inspect.getsource(session.resolve_budget))
check("an explicit --memory still wins over the environment",
      session.resolve_budget(4.0) == 4.0)
check("a malformed value falls back rather than crashing the server",
      "except ValueError" in inspect.getsource(session.resolve_budget))
check("...and one place answers it, so no two commands can disagree",
      "resolve_budget" in _init)
check("the resolved budget is what capacity planning is given",
      _init.index("budget_gb = self.budget_gb")
      < _init.index("autoconfig.choose_capacity(man, budget_gb=budget_gb"))
check("...and the strategy choice too",
      _init.index("budget_gb = self.budget_gb")
      < _init.index("man, budget_gb=budget_gb"))
check("the server reports the budget and what it is using of it",
      all(k in inspect.getsource(session.Session.stats)
          for k in ('"budget_gb"', '"footprint_gb"', '"max_completion_tokens"',
                    '"token_limit_reason"')))
check("the page builds its menu from the ceiling, not from the context window",
      "health.max_completion_tokens" in _uic and "Math.min(ctx,32768)" not in _uic)
check("...and says which limit is in force", "Capped by memory, not by the model" in _uip)

print("\n" + "=" * 84)
print("4o. THE POOL LEAVES ROOM FOR THE REPLY")
print("=" * 84)
# THE BUG: autoconfig held back a flat 3.0 GB for "the OS, the runtime and the KV cache". It was
# tuned for this model, so it worked here and would have silently overrun on a model with a
# heavier KV cache. Same reserve, derived: the machine terms stay fixed, the KV term comes from
# the checkpoint.
# The reserve is a measured constant, not a formula. It was briefly
# `base + top_k * bytes_per_expert * n_layers`, which was arithmetically fine and mechanically
# wrong: a layer's fetched bytes are released once they are in slots, so the peak carries one
# layer's worth (53 MB on gpt-oss), not the sum over 36. That inflated the reserve by 1.9 GB and
# made the planner refuse a model that runs at 7.56 GB.
check("the reserve clears both models that were measured",
      session.WORKING_MEMORY_GB >= 2.96, str(session.WORKING_MEMORY_GB))
check("...without being so large it refuses models that fit",
      session.WORKING_MEMORY_GB <= 3.5, str(session.WORKING_MEMORY_GB))
check("it does not pretend to a per-model formula it cannot support with two points",
      session.Session._working_memory({"layers": {"0": {"bytes_per_expert": 13_240_000}}}, 4)
      == session.Session._working_memory({}, 8) == session.WORKING_MEMORY_GB)

# THE WIDTH OF THE PREFILL PASS IS A PROPERTY OF THE MODEL AND MUST NOT BE ONE OF THE POOL.
#     It used to be `MAX_PREFILL_SPANS * capacity // top_k`, and these checks used to assert
#     exactly that -- a roomy pool got a wide pass, a tight one got a narrow pass. The premise
#     was that the memory peak grew with the number of _chunks splits a tight pool forces.
#     Measured on Qwen3-30B-A3B-3bit at capacities 40 and 11, the chunk counts differ sevenfold
#     (103 against 94, and 15 against 14) and the step-18-to-128 peak delta is the same either
#     way, 1.11 GB against 1.25 GB. The peak tracks the STEP. Splitting costs time, not memory.
#
#     Which turned the old formula into a defect, because chunked prefill is not bit-exact
#     across step sizes -- a 395-token prompt comes back worded differently at 16, 32, 64, 256
#     and 2048, and identically when the same step is run twice. A step read off capacity is a
#     step the memory controller can change underneath a live conversation: a 9.0 -> 6.5 GB
#     shrink moved it 80 -> 30. Reclaiming memory is not allowed to change what the model says,
#     so the step is now derived from the model and is a constant for the life of the process.
class _P:
    def __init__(self, cap, k, d="models/Qwen3-30B-A3B-3bit"):
        self.plan, self.config_dir = {"capacity": cap, "top_k": k}, d
_step = session.Session._prefill_step
# The width is derived from a real config, so these checks need that model on disk. A fresh
# clone has no models at all -- every one of them would then read the floor and the section
# would report five failures for a machine that is working correctly.
if os.path.exists(os.path.join(ROOT, "models", "Qwen3-30B-A3B-3bit", "config.json")):
    _caps = {_step(_P(c, 8)) for c in (9, 11, 22, 40, 44, 70, 128, 1000)}
    check("capacity cannot move the prefill step, at any capacity the controller can reach",
          len(_caps) == 1, f"{sorted(_caps)}")
    check("...and the step it settles on is the one the activation budget buys",
          _caps == {69}, f"{sorted(_caps)}")
    # top_k is a model property, not a pool one, so it MAY move the step -- a model pushing
    # tokens through twice as many experts a token costs twice the activations for the same width.
    check("a model with a wider expert path gets a narrower pass",
          _step(_P(44, 16)) < _step(_P(44, 8)))
    check("it never exceeds the measured safe ceiling",
          _step(_P(44, 1)) == session.PREFILL_STEP, str(_step(_P(44, 1))))
else:
    print("  SKIPPED - models/Qwen3-30B-A3B-3bit absent; the prefill width is read from a real"
          "\n  config, so these four checks need that model prepared.")
check("...and never collapses to a width that would never finish",
      _step(_P(44, 4096)) == 16, str(_step(_P(44, 4096))))
check("a model whose config cannot be read takes the floor, not the ceiling",
      _step(_P(44, 8, d="/nonexistent")) == 16)
check("...and so does a plan with no top-k to size against",
      _step(_P(44, 0)) == 16)

# Non-expert weights are resident whatever the pool does, and were not being subtracted.
_cap = autoconfig.choose_capacity(
    {"layers": {str(i): {"n_experts": 128, "bytes_per_expert": 13_240_000} for i in range(36)},
     "total_bytes": 61_000_000_000}, budget_gb=9.0, top_k=4, reserve_gb=3.3, non_expert_gb=2.39)
check("the pool is sized after the always-resident weights are taken out",
      _cap["pool_gb"] <= 9.0 - 3.3 - 1.0 - 2.39 + 0.01, f'{_cap["pool_gb"]:.2f}')
check("...and gpt-oss still clears its top-k floor at a 9 GB budget",
      _cap["capacity"] >= 4, str(_cap["capacity"]))
check("the reserve reaches the capacity planner",
      "reserve_gb=self.serving_reserve_gb" in _init)
check("...and the strategy chooser, which had its own copy of the constant",
      inspect.getsource(session.Session.__init__).count("reserve_gb=self.serving_reserve_gb") == 2)
check("the budget itself is no longer reduced a second time before planning",
      "plan_budget" not in _init)
check("a checkpoint with no KV figure falls back to the tuned constant, not to zero",
      "self.serving_reserve_gb = autoconfig.RESERVE_GB" in _init)
check("an explicit residency that leaves no room says so out loud",
      "leaves no room for a reply" in _init)
check("...and says what to do about it", "lower --residency, or raise the budget" in _init)
check("...only when it is actually the binding problem",
      'self.token_limit_reason == "memory"' in _init and "MIN_REPLY_TOKENS" in _init)
check("the reserve is reported, so the arithmetic can be checked from outside",
      '"serving_reserve_gb"' in inspect.getsource(session.Session.stats))

print("\n" + "=" * 84)
print("4p. PREFILL IS THE WIDEST PASS, SO IT IS THE ONE THAT HAS TO BE BOUNDED")
print("=" * 84)
# THE BUG: mlx_lm pushes up to prefill_step_size prompt tokens through the model in ONE forward
# pass, default 2048. Nothing in the budget accounted for that pass, because the reserve had been
# measured during decode, where the pass is one token wide. Measured at 44/128 on a 4,109-token
# prompt: step 256 peaked at 9.29 GB against a 9.00 GB ceiling; step 128 peaked at 7.80 GB.
check("the engine does not accept mlx_lm's default", session.PREFILL_STEP < 2048)
check("...and picks a width that was measured to fit", session.PREFILL_STEP == 128)
# On the zero-copy path the peak fell as the pass widened (7.87 GB at 104 tokens, 5.69 at 512, on
# a 1,749-token prompt at the 9.7 GB ceiling) and time to first token fell 1.8-2.7x. A packed
# model takes the wider pass; the model's own files still take the measured-safe narrow one.
check("a packed model on the zero-copy path takes the wider, measured pass",
      session.PREFILL_STEP_PACKED == 512 and session.PREFILL_STEP_PACKED < 2048)
class _PP(_P):
    packed = True
if os.path.exists(os.path.join(ROOT, "models", "Qwen3-30B-A3B-3bit", "config.json")):
    check("...and only a packed model does, still bounded by the model's expert width",
          69 < _step(_PP(44, 8)) <= 512 and _step(_P(44, 8)) == 69,
          f"{_step(_PP(44, 8))} vs {_step(_P(44, 8))}")
check("...and Qwen3.6-35B-A3B (top-8, 512-wide experts) gets the full 512 measured above",
      _step(_PP(44, 8, "models/Qwen3.6-35B-A3B-4bit")) == 512
      if os.path.exists(os.path.join(ROOT, "models/Qwen3.6-35B-A3B-4bit/config.json")) else True)
check("...still bounded for a model with a much wider expert path", _step(_PP(44, 4096)) == 16)
check("the width reaches the generator", "prefill_step_size=(self.prefill_step"
      in inspect.getsource(session.Session.stream_text).replace("\n", "").replace(" ", "")
      or "prefill_step_size" in inspect.getsource(session.Session.stream_text))
check("...and a caller can still override it for a measurement",
      "prefill_step_size: int | None = None" in inspect.getsource(session.Session.stream_text))
check("the reserve is now sized from prefill, not from decode",
      session.WORKING_MEMORY_GB > 2.0, str(session.WORKING_MEMORY_GB))

print("\n" + "=" * 84)
print("4q. A SMALLER POOL IS NOT AUTOMATICALLY A SAFER ONE")
print("=" * 84)
# Every miss reads an expert into a host buffer on its way to a slot, so shrinking the pool
# trades resident bytes for transient ones -- and past a point the trade loses. Measured on one
# 4,109-token prompt: 44/128 peaked at 7.89 GB with 1.09 GB of RSS, 34/128 at 9.07 GB with
# 6.23 GB of RSS and the guard killed it. Reserving more made the server less safe.
_init = inspect.getsource(session.Session.__init__)
check("the reply's KV is not reserved in the plan as well as enforced in the ceiling",
      "TARGET_REPLY_TOKENS" not in _init.split("serving_reserve_gb = round")[1][:200])
check("...and the reason is recorded, because the direction is counter-intuitive",
      "A smaller pool is not automatically a safer one" in _init)
check("the OS slack is not counted twice either",
      session.OS_AND_RUNTIME_GB < 1.0 and "MIN_REPLY_TOKENS" in inspect.getsource(session))
check("the reserve is still the runtime plus the measured prefill peak",
      abs(session.OS_AND_RUNTIME_GB + session.WORKING_MEMORY_GB
          - (session.OS_AND_RUNTIME_GB + 3.0)) < 0.01)

print("\n" + "=" * 84)
print("4r. PREFETCHING, WHERE IT IS POSSIBLE AT ALL")
print("=" * 84)
# Prefetching across a token boundary needs the NEXT router's output, which has not been computed.
# Guessing does not work: replaying a 2,271-token trace against a 44-of-128 pool, the previous
# token predicted 0.00% of the misses and the previous layer 5.91% -- an expert the recent past
# used is still resident, so the ones that must be read are exactly the ones with no history.
# Prefill is the exception: _chunks splits one call into spans whose routing is already known.
from bigrig_engine import stream as _stream
class _FakePool:
    layer = 3
    def __init__(self, resident):
        self.g2s = _np.full(64, -1, dtype=_np.int64)
        for i, e in enumerate(resident):
            self.g2s[e] = i
class _FakeFetcher:
    def __init__(self, boom=False): self.asked, self.boom = [], boom
    def prefetch(self, keys):
        if self.boom:
            raise RuntimeError("pending cap")
        self.asked += list(keys)
_ps = _stream.StreamingSwitchGLU._prefetch_span
_obj = type("M", (), {"_pool": _FakePool([1, 2, 3]), "_fetcher": _FakeFetcher()})()
_keys = _ps(_obj, _np.array([[1, 2, 9], [3, 9, 40]]))
check("only the experts that are NOT resident are asked for",
      sorted(k[1] for k in _keys) == [9, 40], str(_keys))
check("...tagged with this layer, since pools are per layer",
      all(k[0] == 3 for k in _keys))
check("...and each is asked for once, however many rows wanted it",
      len(_keys) == len(set(_keys)))
_obj2 = type("M", (), {"_pool": _FakePool([1, 2, 3]), "_fetcher": _FakeFetcher()})()
check("a span whose experts are all resident asks for nothing",
      _ps(_obj2, _np.array([[1, 2], [3, 1]])) == [])
_obj3 = type("M", (), {"_pool": _FakePool([1]), "_fetcher": _FakeFetcher(boom=True)})()
check("hitting the pending cap declines to speculate rather than failing the request",
      _ps(_obj3, _np.array([[7, 8]])) == [])
_src = inspect.getsource(_stream.StreamingSwitchGLU.__call__)
# Only a SPLIT call can prefetch, because only a split call already knows what a later span
# will ask for. The unchunked path returns before reaching any of it -- checked as "the early
# return comes first", which stays true however the branches below it are arranged. This used to
# be pinned to the position of `spans = self._chunks`, and broke the moment that call moved.
check("only a split call prefetches, because only it knows the later spans",
      "_prefetch_span" in _src
      and _src.index("return self._forward(x, mx.array(self.ensure(gi)))")
      < _src.index("_prefetch_span"))
check("what a span never claimed is released rather than held",
      _src.count("self._fetcher.drop(issued)") == 2)
# Admission goes through ensure() on EVERY path -- the unchunked one, its views-mode twin, the
# expert-sorted split and the token-order split. A path that reached the pool any other way would
# skip the residency guarantee, which is the one invariant the whole design rests on. (The
# prefill-through-views path admits nothing: its views live for one layer and never enter the
# pool, which is why it has no ensure() and is not counted here.)
check("every path admits through ensure(), so none of them can skip the residency guarantee",
      _src.count("self.ensure(") == 4, str(_src.count("self.ensure(")))
check("...and nothing in the hot path touches the pool behind ensure()'s back",
      ".touch(" not in _src and "._victim(" not in _src)

print("\n" + "=" * 84)
print("4s. BATCHING: WHAT IT COSTS, AND WHY IT IS NOT THE DEFAULT")
print("=" * 84)
from bigrig_engine import batch as _batch
class _BS:
    """A session's worth of numbers, with no model behind them."""
    kv_bytes_per_token = 98304
    budget_gb, footprint_gb, working_memory_gb = 9.0, 5.03, 2.6
    capacity, top_k = 44, 8
_p = _batch.plan_batch(_BS(), 8, 200, 512)
check("a batch is capped by the pool as well as by memory", _p["size"] == 5, str(_p))
check("...and says which cap bound it", _p["reason"] == "pool", _p["reason"])
# 44 slots / top-8 = 5 sequences before a step wants more experts than a layer holds.
check("the pool cap is the point where a step stops fitting in one pass",
      _p["by_pool"] == _BS.capacity // _BS.top_k)
check("asking for what fits is granted unchanged",
      _batch.plan_batch(_BS(), 4, 200, 512)["reason"] == "requested")
class _Tight(_BS):
    footprint_gb = 8.3
_t = _batch.plan_batch(_Tight(), 8, 4000, 8000)
check("a long request against a full pool is held back by memory",
      _t["reason"] == "memory" and _t["size"] == 1, str(_t))
check("...and never returns a size below one, which would serve nobody",
      _batch.plan_batch(_Tight(), 8, 100000, 100000)["size"] == 1)
check("a session that cannot price KV is not blocked by the memory term",
      _batch.plan_batch(type("S", (_BS,), {"kv_bytes_per_token": 0})(), 4, 200, 512)["size"] == 4)

_rows, _pads, _w = _batch._pad_left([[1, 3, 5], [7], [2, 6, 8, 9]], 0)
check("prompts are padded on the LEFT, so every row ends at the same column",
      [list(r) for r in _rows.tolist()] == [[0, 1, 3, 5], [0, 0, 0, 7], [2, 6, 8, 9]],
      str(_rows.tolist()))
check("...and the padding is reported for the cache to mask",
      _pads == [1, 3, 0] and _w == 4, f"{_pads} {_w}")
check("the cache used is the one that knows about left padding",
      "BatchKVCache" in inspect.getsource(_batch))
check("per-row reply limits are honoured, since requests rarely want the same length",
      "len(limits) != B" in inspect.getsource(_batch.generate_batch))
check("prefill is chunked in the batched path too",
      "prefill_step" in inspect.getsource(_batch.generate_batch))

# Batching is off unless asked for, because it changes the tokens. Measured: the same prompt at
# batch 1 against batch 2, no padding involved, max |logit difference| 1.500 -- and at batch 1
# re-run against itself, exactly 0.000. Served through the real server, 1 of 4 replies matched.
check("batching is off by default", server._State(None).batch_size == 1)
_pump = inspect.getsource(server._State.pump)
check("...and the default path never groups", "if self.batch_size > 1" in _pump)
check("only requests already waiting join a pass, so nobody is delayed for a maybe",
      "get_nowait" in _pump and "Holding the first one back" in _pump)
check("the queue-depth accounting is kept correct when a group is drained",
      _pump.count("self.waiting -= 1") == 2)
_rb = inspect.getsource(server._State._run_batch)
check("a batch that fails does not take its whole group down with it",
      "self._run_one(j)" in _rb)
check("...and only jobs that have emitted nothing are retried, so no text is repeated",
      "j.out.qsize() == 0" in _rb)
check("the divergence is documented where it is implemented",
      "DOES NOT PRODUCE THE SAME TOKENS" in inspect.getsource(session.Session.stream_batch))
check("...and the quality meter reports not-measured rather than a wrong number",
      '"degraded": None' in inspect.getsource(session.Session.stream_batch))
check("the server says so out loud when batching is switched on",
      "Replies will differ from the same request served alone" in inspect.getsource(server.serve))
check("the flag's help says it too",
      "batching changes the reduction order" in open(
          os.path.join(ROOT, "bigrig_engine/cli.py")).read())

print("\n" + "=" * 84)
print("4t. STOPPING, WITHOUT LOSING THE CONVERSATION")
print("=" * 84)
# A closed socket is only noticed when the next write fails, and the OS buffers writes. At single
# digit tokens per second that is seconds of a model still running after the user pressed stop,
# so a request carries a name and can be stopped by it. Measured against a live server: 12 chunks
# received at the moment of the stop, 12 in total -- nothing arrived afterwards.
_st8 = server._State(None)
_j = _st8.submit({"messages": []}, rid="abc")
check("a job can be registered under the client's own id", _st8.by_id.get("abc") is _j)
check("...and stopped by that id", _st8.cancel("abc") is True and _j.cancelled.is_set())
check("an id nobody is using reports that nothing was stopped", _st8.cancel("nope") is False)
_st8.forget("abc")
check("the registry is emptied when the request ends, so it cannot grow", "abc" not in _st8.by_id)
check("forgetting an id that was never there does not raise", _st8.forget("") is None)
_j2 = _st8.submit({"messages": []})
check("a request with no id still runs, it just cannot be stopped by name", not _st8.by_id)

_S2 = server.make_handler(server._State(None))._sampling
check("the client's id is carried out of the request body",
      _S2({"bigrig_id": "xyz"})["_rid"] == "xyz")
check("...and is absent when a conforming OpenAI client does not send one",
      _S2({})["_rid"] == "")
check("...and a hostile one cannot make it unbounded",
      len(_S2({"bigrig_id": "z" * 5000})["_rid"]) == 64)
check("...and a nonsense type is dropped rather than crashing",
      _S2({"bigrig_id": {"a": 1}})["_rid"] == "")
# _rid must never reach the generator, which has no such parameter.
_ro = inspect.getsource(server._State._run_one)
check("internal fields are stripped before generation is called",
      'not k.startswith("_")' in _ro)
check("...on the batched path too",
      'not k.startswith("_")' in inspect.getsource(server._State._run_batch))
check("stopping by name is a route", '"/v1/cancel"' in inspect.getsource(server.make_handler))
check("...that refuses a request with no id",
      "`id` is required" in inspect.getsource(server.make_handler))

check("the page sends an id it can later stop by", "bigrig_id:reqId" in _uic)
check("...and stops by name BEFORE closing the socket, not instead of it",
      _uic.index('fetch("/v1/cancel"') < _uic.index("ctrl.abort()"))
check("the send button becomes the stop button while it is generating",
      'sendEl.textContent="Stop"' in _uic and 'sendEl.textContent="Send"' in _uic)
check("...and stays enabled, since a disabled button cannot be pressed to stop",
      'sendEl.classList.add("stop");sendEl.disabled=false' in _uic)
check("submitting while it is generating stops instead of queueing another",
      "if(busy){stopNow();return;}" in _uic)
check("a deliberate stop is not reported as a connection failure",
      "if(!stopped && out===base)" in _ui or "if(!stopped&&out===base)" in _uic)
check("...and is not coloured as a problem, because the user chose it",
      "(lastCut && !stopped)" in _ui or "(lastCut&&!stopped)" in _uic)
check("what was generated before the stop stays in the conversation",
      "the next question still has the context of this answer" in _uip)
check("...and can be picked up from, exactly like hitting the limit",
      'lastCut = finish==="length" || stopped' in _ui)
check("stopping before it said anything leaves no empty turn behind",
      "history.pop(); bot.remove();" in _ui)

print("\n" + "=" * 84)
print("4u. A MODEL THAT DOES NOT ANSWER IN PLAIN TEXT")
print("=" * 84)
# gpt-oss emits a channelled transcript, not an answer:
#   <|channel|>analysis<|message|>working it out<|end|><|start|>assistant<|channel|>final<|message|>the answer
# Passed through, the user is shown the control tokens and the model's private reasoning as
# though it were the reply. Measured before this was fixed, verbatim from a real run:
#   "<|channel|>analysis<|message|>User asks: ... <|end|><|start|>assistant<|channel|>final..."
_RAW = ('<|channel|>analysis<|message|>User asks about hash tables.<|end|>'
        '<|start|>assistant<|channel|>final<|message|>A hash table maps keys to slots.')
_V = session.Session.visible(_RAW)
check("the reasoning channel becomes the marker every other model here already uses",
      _V.startswith("<think>User asks about hash tables.</think>"), _V[:60])
check("...and the answer follows it as ordinary text",
      _V.endswith("A hash table maps keys to slots."), _V[-40:])
check("no control tokens survive", "<|" not in _V and "|>" not in _V, _V)
check("...and no channel NAME is left behind as if the model had said it",
      "analysis" not in _V.split("</think>")[-1] and "assistantfinal" not in _V)
check("a model that answers in plain text is untouched",
      session.Session.visible("Just a normal reply.") == "Just a normal reply.")
check("...as is a model that already uses think markers",
      session.Session.visible("<think>hm</think>Answer") == "<think>hm</think>Answer")
check("an unknown channel is dropped rather than shown",
      "<|" not in session.Session.visible("<|channel|>commentary<|message|>x"))

# Arriving a piece at a time must give the same text as arriving all at once. Rewriting only the
# newly-arrived tail split `<|channel|>analysis` from its `<|message|>`, so the pair rewrite
# never matched and the user was shown the bare word "analysis".
def _stream_out(chunk):
    raw, sent, out = "", 0, ""
    for i in range(0, len(_RAW), chunk):
        raw += _RAW[i:i + chunk]
        settled = raw[:len(raw) - session._harmony_hold(raw)]
        vis = session.Session.visible(settled)
        if len(vis) > sent:
            out += vis[sent:]; sent = len(vis)
    vis = session.Session.visible(raw)
    return out + vis[sent:]
_outs = {c: _stream_out(c) for c in (1, 2, 7, 31, 1000)}
check("the text is the same however the chunks fall", len(set(_outs.values())) == 1,
      str({k: v[:30] for k, v in _outs.items()}))
check("...and equals rewriting it all at once", _outs[1] == _V)
check("a construct still arriving is held back until it is whole",
      session._harmony_hold("text<|channel|>analysis<|mess") > 0)
check("...and a complete one is released immediately, not after a fixed window",
      session._harmony_hold("text<|channel|>analysis<|message|>more") == 0)
check("the rewrite is applied in the engine, so the API is clean too, not just the page",
      "self.visible(settled)" in inspect.getsource(session.Session.stream_text))
check("...and whatever was held back is released when generation stops",
      "full = self.visible(raw)" in inspect.getsource(session.Session.stream_text))
check("only a model that needs it pays for it",
      "_detect_harmony" in inspect.getsource(session.Session.__init__))
check("...detected from the model's own template, not a name",
      '"<|channel|>" in out' in inspect.getsource(session.Session._detect_harmony))

print("\n" + "=" * 84)
print("4v. A DRAFT MODEL: WHAT IT COSTS BEFORE IT IS ALLOWED TO HELP")
print("=" * 84)
# Speculative decoding proposes tokens with a small model and checks them in one pass of the big
# one. The argument for it here was stronger than usual -- the per-layer host round-trip is paid
# per STEP, so a step that settles n tokens amortises it n-fold. Measured on Qwen3-30B with a
# Qwen3-MOE-4x0.6B draft, the draft's tokens were accepted 52% of the time at n=2 and 63% at n=4,
# and the speed-up was 0.99x and 1.02x. The mechanism works; it does not pay. Widening a step
# multiplies the FETCH -- n tokens want up to n*top_k distinct experts at a layer -- and fetch is
# the dominant cost here, so only the stall is amortised while the reads are not. The draft also
# took 9 experts a layer of residency (40/128 -> 31/128), which the target then missed on.
_si = inspect.getsource(session.Session.__init__)
check("a draft's weights come out of the budget BEFORE the pool is planned",
      _si.index("self.draft_gb = _dir_gb") < _si.index("choose_capacity"))
check("...and cannot take the whole budget with it", "budget_gb * 0.5" in _si)
# Behaviour, not the comment above it: the budget the pool is planned against is the budget
# minus the draft. (Greping the prose failed on a line wrap -- the rule in CONTRIBUTING.)
_sic = re.sub(r"\s+", "", _si)
check("...and the budget planned against is reduced by exactly the draft's weights",
      "budget_gb-self.draft_gb" in _sic)
check("...measured from the weights on disk, not guessed",
      "self.draft_gb=_dir_gb(" in _sic
      and "*.safetensors" in inspect.getsource(session._dir_gb))
_ld = inspect.getsource(session.Session._load_draft)
check("a draft is checked by what its ids MEAN, not by a vocabulary size",
      "tok.encode(text)" in _ld and "self.tokenizer.encode(text)" in _ld)
check("...because comparing sizes compared two different quantities",
      "151,936 against 151,643" in _ld)
check("...over several kinds of text, not one", _ld.count('"') > 8 and "probes" in _ld)
check("a mismatch is refused with what it would have caused",
      "fluent and wrong" in _ld)
check("the engine's own lenient loader is used, not the strict one",
      "stream.load_lenient" in _ld)
check("...because the checkpoint used to measure this is refused by the strict one",
      "tie_word_embeddings" in _ld)
check("whether a token came from the draft is reported per token",
      '"from_draft"' in inspect.getsource(session.Session.stream_text))
check("...and the acceptance rate is reported, so a bad draft is distinguishable "
      "from a bad idea",
      '"draft_acceptance"' in inspect.getsource(session.Session.stats))
check("acceptance is a share of tokens actually produced",
      "self.draft_accepted / self.total_tokens" in inspect.getsource(session.Session.stats))
check("no draft means no draft fields pretending to be measurements",
      '"draft": self.draft_name or None' in inspect.getsource(session.Session.stats))
check("the flag exists on the commands that generate",
      '"--draft"' in open(os.path.join(ROOT, "bigrig_engine/cli.py")).read())
check("...and says the draft is paid for out of the same budget",
      "same memory budget" in open(
          os.path.join(ROOT, "bigrig_engine/cli.py")).read())

print("\n" + "=" * 84)
print("4w. A SLOW MODEL MUST NOT LOOK LIKE A BROKEN ONE")
print("=" * 84)
# REPORTED AS A HANG, AND IT WAS NOT ONE. "hi" to gpt-oss showed nothing for 66 seconds: an empty
# bubble, 0 tok/s, and a dash in every field. Two causes, both ours.
#   1. "hi" is 68 tokens once the chat template is applied, and reading a prompt takes ~40s when
#      97% of the model is on disk -- with no progress reported at all.
#   2. A fixed 96-character hold-back meant nothing displayed until 96 characters existed, which
#      at ~1 token/second is another ~26s AFTER generation had started.
# Measured after: first progress at 0.1s, first text at 34.4s.
_hh = session._harmony_hold
check("a marker cut by the chunk boundary is held", _hh("answer so far<|chan") == 6)
# The pair rewrite needs BOTH halves. Holding only the second lets the catch-all strip the first
# on its own and print the bare word "analysis" -- which corrupted the stream to
# "<nalysis<er asks about hash ta" while this was being built.
check("...and so is a channel opener whose message marker has not arrived",
      _hh("a<|channel|>analysis<|mess") == 25)
check("a completed pair is released immediately",
      _hh("a<|channel|>analysis<|message|>b") == 0)
# `<|end|>` becomes `</think>` only once the `<|start|>...<|message|>` after it lands.
check("an end marker waits for what turns it into a think close", _hh("done<|end|>") == 7)
check("the think marker the rewrite inserts is NOT held",
      _hh("<think>reasoning so far") == 0)
check("ordinary text is never held", _hh("A hash table maps keys.") == 0)
check("the fixed window is gone", not hasattr(session, "HARMONY_HOLD"))
_stx = " ".join(inspect.getsource(session.Session.stream_text).split())
check("...and the hold is computed on the RAW text, before the rewrite",
      "_harmony_hold(raw)" in _stx)
check("...and only for models that emit these markers at all",
      "if self._harmony:" in _stx and "full = raw" in _stx)

check("the engine can report how far into the prompt it is",
      "on_prefill" in inspect.getsource(session.Session.stream_text))
check("...through the callback mlx_lm provides, not a guess",
      '"prompt_progress_callback"' in inspect.getsource(session.Session.stream_text))
_ro = inspect.getsource(server._State._run_one)
check("the server forwards that progress", 'j.out.put_nowait(("prefill"' in _ro)
check("...without ever blocking the one thread allowed to touch MLX",
      "put_nowait" in _ro and "except queue.Full" in _ro)
check("...and stops reporting to a client that has gone", "j.cancelled.is_set()" in _ro)
_sv = inspect.getsource(server.make_handler)
check("progress is sent as its own frame, not as generated text",
      '"prefill_done": info["prefill_done"]' in _sv)
check("...and is never mistaken for a token", 'if c is None and "prefill_total" in info' in _sv)
check("the page shows what it is waiting for", "reading your message" in _uip)
check("...with the elapsed time, so a slow model reads as slow and not as stuck",
      "working · ${s}s" in _ui or "working \u00b7 ${s}s" in _ui)
check("...and it is cleared however the request ends, not only on success",
      "}finally{" in _ui and _ui.split("}finally{")[1][:200].count("doneWaiting()") == 1)

print("\n" + "=" * 84)
print("4f. CONTINUING A CUT-OFF REPLY")
print("=" * 84)
_pr = inspect.getsource(session.Session._prompt)
check("continuing leaves the assistant turn OPEN instead of starting a new one",
      "continue_final_message" in _pr)
check("...and it is one or the other, never both",
      '"continue_final_message" if continue_last else "add_generation_prompt"' in _pr)
check("the reason is recorded", "carries on the sentence it was cut off in" in _pr)
check("the flag reaches the engine from the request",
      '"continue_last"' in inspect.getsource(server.make_handler))
check("the button only appears when the reply actually hit the limit",
      'lastCut = finish==="length"' in _ui and "if(lastCut){" in _ui)
check("continuing appends to the same reply rather than starting a new one",
      "history[history.length-1].content=out" in _ui)
check("...and removes the button once it has been used", 'querySelector(".cont")?.remove()' in _ui)

# _prompt built with a stub, so the branch is exercised rather than read.
class _Stub:
    _warned_template = False
    _starts_in_reasoning = False
    # The real ones, not fakes: `_prompt` records whether the rendered prompt leaves the model
    # mid-thought, and a stub that no-oped that would test a different function than ships.
    THINK_OPEN = session.Session.THINK_OPEN
    THINK_CLOSE = session.Session.THINK_CLOSE
    _note_reasoning_start = session.Session._note_reasoning_start
    _split_reasoning = session.Session._split_reasoning
    def __init__(self, toggle): self.can_toggle_thinking = toggle
    class _Tok:
        @staticmethod
        def apply_chat_template(messages, tokenize=False, **kw): return json.dumps(kw, sort_keys=True)
    tokenizer = _Tok()
_M = [{"role": "user", "content": "hi"}]
# WHICH HALF OF THE REPLY IS THE ANSWER. Two prompt shapes are in the wild: the template opens
# `<think>` and the reply closes it (Qwen3.6, GLM-4.7), or the reply opens and closes it itself
# (Qwen3-30B). Getting this wrong means an API hands a coding agent the model's scratchpad.
_sp = _Stub(True)
_sp._note_reasoning_start("...<|assistant|>\n<think>\n")
check("a prompt that ends mid-thought is recognised", _sp._starts_in_reasoning is True)
check("...and everything before </think> is reasoning, the rest is the answer",
      _sp._split_reasoning("weighing it up</think>\n\nParis") == ("weighing it up", "Paris"))
check("...and a reply still thinking has no answer yet, rather than a partial one",
      _sp._split_reasoning("still weighing") == ("still weighing", ""))
_sp2 = _Stub(True)
_sp2._note_reasoning_start("...<|im_start|>assistant\n")
check("a prompt that does not open the block is recognised too",
      _sp2._starts_in_reasoning is False)
check("...and the reply's own <think> pair is split out",
      _sp2._split_reasoning("<think>hmm</think>\n\nParis") == ("hmm", "Paris"))
check("...and a model that never thinks is passed through untouched",
      _sp2._split_reasoning("Paris") == ("", "Paris"))
check("a stray marker mid-answer is not mistaken for reasoning",
      _sp2._split_reasoning("the tag </think> appears here") == ("", "the tag </think> appears here"))

check("turning reasoning off reaches the template as enable_thinking=False",
      json.loads(session.Session._prompt(_Stub(True), _M, think=False)).get("enable_thinking") is False)
check("...and is left out entirely for a model that has no such switch",
      "enable_thinking" not in json.loads(session.Session._prompt(_Stub(False), _M, think=False)))
check("leaving it on does not pass the flag either",
      "enable_thinking" not in json.loads(session.Session._prompt(_Stub(True), _M, think=True)))
check("a normal turn asks for a fresh assistant reply",
      json.loads(session.Session._prompt(_Stub(True), _M)).get("add_generation_prompt") is True)
check("continuing does the opposite, and never both at once",
      json.loads(session.Session._prompt(_Stub(True), _M, continue_last=True))
      == {"continue_final_message": True})

print("\n" + "=" * 84)
print("4k. ONE VIEW AT A TIME, AND NO STYLESHEET CAN OVERRULE IT")
print("=" * 84)
# THE BUG: `#v-stats{...display:block}` sat in the stylesheet alongside `.view{display:none}`.
# An id selector scores 1,0,0 and a class 0,1,0, so the analytics page rendered underneath the
# chat page no matter which tab was selected -- the chat area was squeezed to a fraction of the
# window and the analytics cards hung below it. Visibility now rides on the `hidden` attribute
# with !important, which wins against any specificity a later rule could reach for.
import re as _re
_css = _ui.split("<style>")[1].split("</style>")[0]
_cssc = _re.sub(r"\s+", "", _css)
check("hidden is enforced with !important, so specificity cannot beat it",
      "[hidden]{display:none!important}" in _cssc)
check("the analytics view starts hidden in the markup",
      _re.search(r'<div[^>]*id="v-stats"[^>]*\shidden', _ui) is not None)
check("no rule sets display on the analytics view outside that mechanism",
      not _re.search(r"#v-stats\{[^}]*display:(none|block|flex)[^}]*\}",
                     _cssc.replace("#v-stats{display:block;overflow-y:auto", "#v-stats{OK")))
check("switching views drives the attribute, not a class the CSS might ignore",
      '$("#v-"+k).hidden=k!==v' in _uic)
# Read the list out of the page rather than restating it here. Hardcoding the three views meant
# adding a fourth broke this check while the page was perfectly correct -- and a test that fails
# for being out of date teaches people to edit tests rather than read them.
# `_uic` has had its whitespace stripped, so match the raw page.
_views = re.findall(r'"([a-z]+)"', re.search(r"const VIEWS\s*=\s*\[([^\]]*)\]", _ui).group(1))
check("every view the nav offers has a nav entry and a section to show",
      len(_views) >= 3
      and all(f'id="v-{v}"' in _ui for v in _views)
      and all(f'id="n-{v}"' in _ui for v in _views), f"{_views}")
check("...and every section in the markup is reachable from the nav",
      sorted(set(re.findall(r'class="view" id="v-([a-z]+)"', _ui))) == sorted(_views),
      f"{sorted(set(re.findall(chr(39) + 'class=.view. id=.v-([a-z]+).' + chr(39), _ui)))} vs {sorted(_views)}")
check("the polling loop reads the same attribute",
      'if(!$("#v-stats").hidden)loadStats()' in _uic)
check("a view is chosen explicitly at start-up rather than left to the markup",
      'show("chat");' in _uip)

print("\n" + "=" * 84)
print("4l. THE ANALYTICS PAGE RUNS ALL THE WAY THROUGH")
print("=" * 84)
# THE BUG: `const H=health` was declared near the end of loadStats but read near the top. The
# temporal dead zone made every call throw ReferenceError right after time-to-first-token, so
# quality, reply length, the memory chart, the whole server card and the table were never filled
# in -- they showed the em dash that means "not measured", which was a lie.
_ls = _ui.split("async function loadStats()")[1].split("\n}")[0]
check("H is bound before anything reads it",
      _ls.index("H=health") < min(_ls.index(f"H.{f}") for f in
                                  ("uptime_s", "requests_served", "load_seconds", "flagged_tokens")))
check("...and only once, so no later declaration can shadow it into a dead zone",
      _ls.count("H=health") == 1)
check("every name loadStats reads is one it declared or a known global",
      not (set(_re.findall(r"\b([A-Z][A-Za-z0-9_]*)\s*\.", _ls))
           - {"H", "Math", "Object", "NA", "Array", "Number", "JSON", "String"}),
      str(set(_re.findall(r"\b([A-Z][A-Za-z0-9_]*)\s*\.", _ls))))

print("\n" + "=" * 84)
print("4h. THE REASONING CONTROL SAYS WHAT IT ACTUALLY DOES")
print("=" * 84)
# THE BUG: the box was labelled "show reasoning", which reads as a display toggle, but it decided
# whether the model reasons at all -- and reasoning is drawn from the same budget as the answer.
# Measured on Qwen3-30B, "Name three primary colours." at a 60-token limit: with it on, all 60
# tokens went to <think> and the answer never began; with it off, the reply finished in 41.
check("the control is not labelled as if it only affected display", "show reasoning" not in _uip)
check("...it is named for what it does", "think first" in _uip)
check("...and the token cost is stated where it is set",
      "same reply limit as the answer" in _uip)
check("it is off by default, so a reply is not spent before it starts",
      'id="c-think"' in _uic and 'id="c-think"checked' not in _uic)
check("the cut-off note reports a measured share, not an adjective",
      "much of it went on reasoning" not in _uip and "reasoning took ${pct}% of it" in _uip)
# The reasoning arrives in its own field now, so the share is a straight comparison of the two
# rather than a hunt for markers inside the answer -- and the answer no longer contains any.
check("...computed from the reasoning the server actually sent, against the answer",
      "100*reason.length/Math.max(reason.length+out.length,1)" in _uic
      and 'out.indexOf("<think>")' not in _uic)
check("the page shows the thinking folded away rather than dropping it",
      "withThink" in _uic and 'reason+=rc' in _uic and "delta?.reasoning_content" in _uic)

print("\n" + "=" * 84)
print("4j. THE COST ESTIMATE NEEDS A RATE, SO /health CARRIES ONE")
print("=" * 84)
# THE BUG: the chat page polls only /health, but the measured rate lived only in /stats, and
# /stats was fetched only while the analytics view was open. refresh() then replaced `health`
# wholesale every 4s, wiping the one field that had been merged in. Result: the cost line showed
# the KV size and never the time.
class _FakeState:
    def __init__(self, rows):
        import threading
        self.count_lock, self.history = threading.Lock(), list(rows)
check("an empty history aggregates to nothing, not to zeros",
      server._aggregate(_FakeState([])) == {})
_rows = [{"tok_s": 5.0, "ttft": 1.0, "tokens": 10, "seconds": 2.0, "flagged": 0, "finish": "stop"},
         {"tok_s": 9.0, "ttft": 0.5, "tokens": 20, "seconds": 2.0, "flagged": 1, "finish": "length"},
         {"tok_s": 7.0, "ttft": 2.0, "tokens": 30, "seconds": 4.0, "flagged": 0, "finish": "stop"}]
_a = server._aggregate(_FakeState(_rows))
check("the median is the middle value, not the mean", _a["median_tok_s"] == 7.0, str(_a))
check("...and it ignores replies that had no measurable rate",
      server._aggregate(_FakeState(_rows + [{"tokens": 1, "finish": "stop"}]))["median_tok_s"] == 7.0)
check("the median ttft is separate from the token rate", _a["median_ttft"] == 1.0, str(_a))
check("replies cut off at the limit are counted", _a["cut_off"] == 1)
check("totals add up", _a["tokens"] == 60 and _a["seconds"] == 8.0 and _a["flagged"] == 1, str(_a))
check("a single reply is its own median",
      server._aggregate(_FakeState(_rows[:1]))["median_tok_s"] == 5.0)
check("/health carries the rate so the chat page can estimate time",
      '"median_tok_s": agg.get("median_tok_s")' in inspect.getsource(server.make_handler))
check("...from the same helper /stats uses, so the two can never disagree",
      inspect.getsource(server.make_handler).count("_aggregate(state)") == 2)
check("the page no longer patches the rate in behind refresh's back",
      "health.median_tok_s = a.median_tok_s" not in _ui)
check("the estimate is drawn from that field", "health.median_tok_s" in _uic)

print("\n" + "=" * 84)
print("4i. THE ANALYTICS PAGE REPORTS THE SERVER ITSELF")
print("=" * 84)
for f in ("uptime_s", "requests_served", "queue_depth", "load_seconds",
          "flagged_tokens", "flagged_share"):
    check(f"{f} is shown only when the server supplies it",
          f"H.{f}!=null" in _uic or f"H.{f}||0" in _uic, f)
check("the server's own counters are served",
      all(k in inspect.getsource(server.make_handler)
          for k in ("uptime_s", "queue_depth", "requests_served")))
check("...and the model's counters come from the session",
      all(k in inspect.getsource(session.Session.stats)
          for k in ('"load_seconds"', '"flagged_tokens"', '"flagged_share"')))

print("\n" + "=" * 84)
print("4g. THE ANALYTICS VIEW PLOTS MEASUREMENTS, NOT A SNAPSHOT")
print("=" * 84)
_svr = inspect.getsource(server)
check("the server keeps a rolling record of every reply", "self.history: deque" in _svr)
check("...bounded, so a week-long server does not grow without limit",
      "maxlen=HISTORY" in _svr and server.HISTORY > 0)
# Was a grep for a sentence in a comment, which failed because the sentence wrapped across a
# line -- the third time in this suite. Test the bound itself.
_hist_deque = server._State.__init__.__code__.co_consts
check("...and the bound is a real number, not unlimited",
      isinstance(server.HISTORY, int) and 50 <= server.HISTORY <= 5000,
      str(server.HISTORY))
_d = deque(maxlen=server.HISTORY)
for i in range(server.HISTORY + 50):
    _d.append(i)
check("...which actually discards the oldest rather than growing",
      len(_d) == server.HISTORY and _d[0] == 50)
check("every reply is logged with what actually happened",
      all(k in _svr for k in ('"tokens"', '"seconds"', '"ttft"', '"tok_s"', '"flagged"',
                              '"finish"', '"miss_rate"')))
# Four: blocking and streaming, on the OpenAI pair and on the Responses pair. A reply that is
# not logged is a reply the analytics page cannot see, so every path that finishes one must.
check("every path that finishes a reply logs it",
      _svr.count("self._log(") == 4, str(_svr.count("self._log(")))
check("/stats is served", '"/stats"' in _svr)

# WORK NOBODY IS WAITING FOR MUST NOT BE DONE, AND THE PREFILL IS WHERE THE TIME IS.
#     `_run_one` checked `cancelled` between chunks, which is too late by exactly the length of
#     the prefill. Measured against a real agent: Codex's client-side timeout is shorter than a
#     12,704-token prompt takes on this model, so it gives up and RETRIES -- and each retry
#     queued another full prefill for a request it had already abandoned. Observed three deep,
#     with the GPU at 70% doing entirely wasted work.
#
#     Two checks now. One before any work starts, for a job cancelled while it sat in the queue.
#     One inside the prefill callback, which is the only code that runs during a prefill --
#     `stream_text` yields nothing until it finishes, so the between-chunks check is never
#     reached. Measured after: a streaming request abandoned mid-prefill no longer delays the
#     next one, 44.6s -> 1.4s.
check("a job cancelled before it starts is never started",
      "if j.cancelled.is_set():" in _svr
      and _svr.index("if j.cancelled.is_set():") < _svr.index("def on_prefill"))
check("...and one abandoned DURING the prefill aborts rather than finishing it",
      "raise _Abandoned()" in _svr)
check("...which is treated as a clean end, not an error",
      "except _Abandoned:" in _svr)
check("the reason is written down where the next person will look",
      "12,704" in _svr and "wasted work" in _svr)
# HONEST LIMIT. A blocking request writes nothing until generation is finished, so the handler
# cannot learn its client has gone and there is nothing to cancel. Only streamed requests are
# detectable this way -- and streaming is what agents use.
check("the limit is stated rather than implied -- this works for streamed requests only",
      "THE LIMIT, STATED" in _svr and "blocking" in _svr)
check("aggregates use a median, not a mean that one slow reply can drag",
      "median_tok_s" in _svr and "sorted(" in _svr)
check("an empty history produces an empty aggregate rather than dividing by zero",
      server._aggregate(type("S", (), {"count_lock": __import__("threading").Lock(),
                                       "history": []})()) == {})
check("the view has two pages and a way between them",
      'id="v-chat"' in _ui and 'id="v-stats"' in _ui and 'id="n-stats"' in _ui)
for c in ("ch-tps", "ch-ttft", "ch-q", "ch-len", "ch-mem"):
    check(f"chart {c} exists", f'id="{c}"' in _ui)
check("charts are inline SVG -- nothing is fetched to draw them",
      "<svg viewBox" in _ui and "chart.js" not in _ui.lower()
      and not re.search(r"\bd3\b|d3\.min|d3\.js", _ui.lower()))
check("...and the page fetches nothing from anywhere at all",
      not re.search(r'(src|href)\s*=\s*"(https?:)?//', _ui))
check("an empty series says so instead of drawing a blank chart", "no data yet" in _uip)
check("the memory picture separates what is in RAM from what is on disk",
      "experts in RAM" in _ui and "experts on disk" in _ui and "KV cache" in _ui)
check("the reply table marks the ones that hit the limit", '"hit limit"' in _ui)

print("\n" + "=" * 84); print("5. LAUNCHING A CODING AGENT"); print("=" * 84)
check("claude is wired to the Anthropic API",
      launch.AGENTS["claude"].env["ANTHROPIC_BASE_URL"] == "{base}")
check("...with an auth token set, because the client refuses to start without one",
      "ANTHROPIC_AUTH_TOKEN" in launch.AGENTS["claude"].env)
check("opencode is wired to the OpenAI API by environment alone",
      launch.AGENTS["opencode"].env["OPENAI_BASE_URL"].endswith("/v1"))
# CODEX CANNOT BE. Its wire protocol is chosen by `wire_api` in config.toml, and since 0.152.0
# the only accepted value is "responses" -- `wire_api = "chat"` is no longer supported, in the
# binary's own words. OPENAI_BASE_URL alone aimed it at this server using an API it will not
# speak, which is why `bigrig launch codex` was quietly dead. It gets a throwaway CODEX_HOME
# instead, so `~/.codex` is never opened.
check("codex is given a config rather than only variables",
      launch.AGENTS["codex"].needs_config is True)
check("...and that config selects the responses wire API",
      'wire_api = "responses"' in inspect.getsource(launch._codex_home))
check("...written to a throwaway home that is removed when the agent exits",
      "mkdtemp" in inspect.getsource(launch._codex_home)
      and "shutil.rmtree(tmp_home" in inspect.getsource(launch.run)
      if hasattr(launch, "run") else "mkdtemp" in inspect.getsource(launch._codex_home))
check("...and the user's own ~/.codex is never touched",
      "~/.codex" not in inspect.getsource(launch._codex_home).replace(
          "`~/.codex` is never opened", ""))
env = launch.AGENTS["claude"].environment("http://127.0.0.1:9999")
check("the base url is substituted, not left as a template",
      env["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:9999")
before = dict(os.environ)
launch.AGENTS["claude"].environment("http://x")
check("building the environment does not touch this process's own",
      dict(os.environ) == before)
src = inspect.getsource(launch)
# The property is BEHAVIOURAL: an install hint is text shown to a person, never something this
# module executes. Grepping for "curl" fails on the hint strings themselves, which is the
# opposite of what it should conclude.
check("an install hint is inert text, never a command that gets run",
      all(isinstance(a.install_hint, str) for a in launch.AGENTS.values()))
check("...and the module runs exactly one program: the agent itself",
      src.count("subprocess.call") == 1 and src.count("subprocess.Popen") == 1)
check("the only Popen is the bigrig server, passed in by the caller",
      "subprocess.Popen(serve_argv" in src)
# A bare `open(` grep matches `urlopen(` and `Popen(`, which is how this test failed while the
# code was correct. Match the call itself, and check the AST for write modes.
import ast as _ast, re as _re
_calls = [n for n in _ast.walk(_ast.parse(src))
          if isinstance(n, _ast.Call) and isinstance(n.func, _ast.Name) and n.func.id == "open"]
# THE RULE IS NOT "NEVER WRITE A FILE", IT IS "NEVER CHANGE STATE THE USER DID NOT ASK ABOUT".
#     This used to assert that `open()` was never called at all, which was a fair proxy while
#     every agent could be pointed at a server by environment alone. Codex cannot: its wire
#     protocol is chosen by `wire_api` in config.toml, and since 0.152.0 the only accepted value
#     is "responses". So one file IS written -- into a temporary CODEX_HOME that is deleted when
#     the agent exits. What must still be true is that nothing the USER owns is opened.
_home_src = inspect.getsource(launch._codex_home)
check("exactly one file is written, and only for the agent that needs one",
      len(_calls) == 1, f"{len(_calls)} open() calls")
check("...into a directory this command created for the purpose",
      "mkdtemp" in _home_src and _home_src.count("open(") == 1)
check("...and it is removed when the agent exits", "rmtree" in src)
check("no path under the user's home is opened",
      not _re.search(r'open\(\s*["\']?~', src) and "expanduser" not in src)
check("the agent is configured through its environment only",
      "env=env" in src and _re.search(r"child process only", src) is not None)
try:
    launch.run("nonesuch", "m", [])
    check("an unknown agent raises", False)
except ValueError as e:
    check("an unknown agent raises", "Known:" in str(e))
try:
    launch.AGENTS["codex"] = launch.Agent("codex", "definitely-not-installed-xyz",
                                          {"A": "{base}"}, "see the docs")
    launch.run("codex", "m", [])
    check("a missing agent explains rather than crashing", False)
except FileNotFoundError as e:
    check("a missing agent explains rather than crashing", "not installed" in str(e))
    check("...and gives the install command", "see the docs" in str(e))
port = launch.free_port()
check("free_port returns a usable port", isinstance(port, int) and 1024 < port < 65536)
check("nothing is serving on a fresh port", not launch.port_is_serving(port, timeout=0.4))

print("\n" + "=" * 84); print("6. THE WEB INTERFACE"); print("=" * 84)
ui = open(os.path.join(ROOT, "bigrig_engine/webui.html")).read()
import re
ext = re.findall(r'(?:src|href)\s*=\s*["\']https?://', ui)
check("no external scripts, styles or fonts -- it is one self-contained file", not ext, str(ext))
check("...so it works with no network at all", "cdn" not in ui.lower())
check("the quality meter is on screen whichever page you are on",
      'id="m-bar-q"' in ui and 'id="m-q"' in ui and "--warn" in ui
      and 'class="rail"' in ui and 'id="c-q"' in ui)
check("it explains what a flag MEANS, not just that one happened",
      "looping or losing coherence" in ui)
check("it says outright when weights were changed",
      "differ from the model you downloaded" in ui and "CHANGED to fit" in ui)
check("it reads the per-chunk quality signal", '"degraded" in t' in ui)
# Dark-first: :root carries the dark palette and a media query overrides it for light. Either
# order is fine; what must not happen is a theme that is only defined behind a media query.
check("it works in dark mode as well as light",
      "prefers-color-scheme" in ui and "data-theme=light" in ui and "data-theme=dark" in ui)
check("...and every colour token is defined on bare :root, not only inside a media query",
      set(re.findall(r"(--[a-z0-9]+):", ui.split("@media")[0]))
      >= set(re.findall(r"(--[a-z0-9]+):", ui)) - {"--x"})
check("the page is served from GET /", '"/"' in inspect.getsource(server.make_handler))
check("...and a missing file gives an error, not a traceback",
      "web interface is missing" in inspect.getsource(server.make_handler))
check("the server ships the page as package data",
      'bigrig_engine = ["webui.html"]' in open(
          os.path.join(ROOT, "pyproject.toml")).read())
sv = inspect.getsource(server.serve)
check("the browser link is listed before the curl examples",
      sv.index("in a browser") < sv.index("api   "))

print("\n" + "=" * 84)
print("7. A CLIENT THAT WENT AWAY, AND THE MEMORY DEFAULT")
print("=" * 84)

# A streaming handler learns its client left by writing to a dead socket. A blocking one writes
# nothing until it has finished, so before `client_gone` it generated a whole reply for a client
# that had hung up. THE FALSE POSITIVE IS THE DANGEROUS DIRECTION -- aborting a live request is
# worse than finishing a dead one -- so most of these assert a present client is NOT reported gone.
import socket as _sock
import time as _time

_a, _b = _sock.socketpair()
check("a live quiet client is not reported gone", server.client_gone(_a) is False)
_b.sendall(b"POST /v1/chat/completions HTTP/1.1\r\n")
_time.sleep(0.05)
check("a client that sent a pipelined request is not reported gone",
      server.client_gone(_a) is False)
check("...and the peek did not consume it", _a.recv(4, _sock.MSG_PEEK) == b"POST")

_c, _d = _sock.socketpair(); _d.close(); _time.sleep(0.05)
check("a client that closed is reported gone", server.client_gone(_c) is True)

# Data still queued outranks the EOF behind it: there is a request there to answer.
_e, _f = _sock.socketpair(); _f.sendall(b"x"); _f.close(); _time.sleep(0.05)
check("a closed client with unread data reads present until drained",
      server.client_gone(_e) is False)
_e.recv(1)
check("...and gone once drained", server.client_gone(_e) is True)

_g, _h = _sock.socketpair(); _g.close()
check("an unusable socket is reported gone", server.client_gone(_g) is True)
check("no connection at all is not reported gone", server.client_gone(None) is False)

_i, _j = _sock.socketpair(); _j.shutdown(_sock.SHUT_WR); _time.sleep(0.05)
check("a half-closed client is reported gone", server.client_gone(_i) is True)

_k, _l = _sock.socketpair()
_t0 = _time.time()
for _ in range(500):
    server.client_gone(_k)
check("polling is cheap enough to do every half second", _time.time() - _t0 < 0.25)
for _s in (_a, _b, _c, _e, _f, _h, _i, _j, _k, _l):
    try:
        _s.close()
    except OSError:
        pass

_runsrc = inspect.getsource(server.make_handler)
check("the blocking wait polls for a departed client", "_client_gone()" in _runsrc)
check("...and cancels the job so the engine stops too", "job.cancelled.set()" in _runsrc)
check("...and GET_TIMEOUT_S still means longest SILENCE, not longest request",
      "deadline = now + GET_TIMEOUT_S" in _runsrc)
# THE REGRESSION THIS EXISTS TO STOP. The first version checked only in the `queue.Empty` branch,
# which never runs while tokens are flowing -- a chunk every 77 ms at 13 tok/s means get() never
# times out. It passed every unit test here and still let an abandoned blocking request produce
# all 400 tokens. The check has to be on a clock, so it runs whether or not output is arriving.
check("the departed-client check runs on a clock, not only when the queue is idle",
      "next_check" in _runsrc and _runsrc.index("next_check = now + CLIENT_POLL_S")
      < _runsrc.index("if kind is None:"))
check("...and is reached even when a chunk arrived this pass",
      _runsrc.index("kind = None") < _runsrc.index("if self._client_gone():"))
check("the poll interval is short enough to matter and cheap enough to be free",
      0.05 <= server.CLIENT_POLL_S <= 2.0)

import bigrig_engine.cli as _cli
_p = _cli.build_parser()
check("releasing memory under pressure is ON by default",
      _p.parse_args(["serve", "m"]).release_memory is True)
check("...and can still be turned off explicitly",
      _p.parse_args(["serve", "m", "--no-release-memory"]).release_memory is False)
check("...and the last flag on the line wins",
      _p.parse_args(["serve", "m", "--no-release-memory", "--release-memory"]
                    ).release_memory is True)
# Measured: one squeeze left a live server at 30 of 38 experts a layer for the rest of the day.
# Growing back is on by default now, still bounded by home and the three-minute quiet rule.
check("...and TAKING memory back is on by default too",
      _p.parse_args(["serve", "m"]).reclaim_memory is True)
check("...and can be turned off explicitly",
      _p.parse_args(["serve", "m", "--no-reclaim-memory"]).reclaim_memory is False)
check("the serve() default agrees with the CLI default",
      inspect.signature(server.serve).parameters["release_memory"].default is True)

# ---------------------------------------------------------------- the first-run tune
# `bigrig knee` existed and nobody ran it, so every streamed model ran on the planner's estimate.
# The tune runs once, unasked, and these pin down every case in which it must NOT run -- each
# one is a user who has already answered the question, or a command that IS the measurement.
import types as _types
_cli_src = inspect.getsource(_cli._auto_tune)

class _FakeSession:
    def __init__(self, streamed=True, name="M", budget=9.0):
        self.streamed, self.name, self.budget_gb = streamed, name, budget
        self.closed = False
    def close(self): self.closed = True

def _args(**kw):
    a = _types.SimpleNamespace(cmd="run", no_tune=False, residency=None, memory=None)
    for k, v in kw.items(): setattr(a, k, v)
    return a

_s = _FakeSession(streamed=False)
check("a model that fits in memory is not tuned (no capacity to choose)",
      _cli._auto_tune(_args(), _s) is _s and not _s.closed)
_s = _FakeSession()
check("--no-tune is honoured", _cli._auto_tune(_args(no_tune=True), _s) is _s and not _s.closed)
_s = _FakeSession()
check("an explicit --residency is an answer, so no question is asked",
      _cli._auto_tune(_args(residency=0.3), _s) is _s and not _s.closed)
for _c in ("knee", "calibrate"):
    _s = _FakeSession()
    check(f"`bigrig {_c}` does not tune first -- it IS the measurement",
          _cli._auto_tune(_args(cmd=_c), _s) is _s and not _s.closed)
# a knee already on file at this budget means it is in use already; nothing to do
import bigrig_engine.knee as _kn
_orig_load = _kn.load
try:
    _kn.load = lambda name, budget=None: {"capacity": 40}
    _s = _FakeSession()
    check("a knee already measured at this budget is used, not re-measured",
          _cli._auto_tune(_args(), _s) is _s and not _s.closed)
finally:
    _kn.load = _orig_load

check("the tune closes the running session before probing -- one pool alive at a time",
      _cli_src.index("s.close()") < _cli_src.index("_knee.measure("))
check("...and rebuilds from init_kwargs, so the served session is the tuned one",
      "_S(**rebuild)" in _cli_src and "s.init_kwargs" in _cli_src)
check("a failed measurement falls back to the estimate rather than blocking the run",
      "except Exception" in _cli_src and "using the safe estimate" in _cli_src)
check("the user is told it is one-time and how to skip it",
      "one-time" in _cli_src and "--no-tune" in _cli_src and "--residency" in _cli_src)
check("`bigrig knee` and the tune share one make_session, so they cannot drift apart",
      "_knee_maker(a, path, budget)" in inspect.getsource(_cli.cmd_knee)
      and '_knee_maker(a, path, inp["budget"])' in inspect.getsource(_cli._auto_tune))
# The probes and the rebuilt session must plan from the budget the tune measured at. A `None`
# budget resolves against free memory at that instant, and the instant after a pool is dropped
# MLX has not yet returned it -- measured: tuned at 9.1 GB, rebuilt at 8.4, knee not found.
check("...and the rebuilt session is pinned to the budget the tune used",
      'rebuild["budget_gb"] = s.budget_gb' in inspect.getsource(_cli._auto_tune))
check("...and one set of inputs",
      "_knee_inputs(" in inspect.getsource(_cli.cmd_knee) and "_knee_inputs(" in _cli_src)
# ---------------------------------------------------------------- launch: what the user sees
# With the server's stdout swallowed, a first run that now tunes for a minute or two was two
# minutes of silence after "starting ...". The server's own lines are relayed until it answers.
_lsrc = inspect.getsource(launch.run)
check("launch relays the server's startup lines to the user",
      "threading.Thread" in _lsrc and "forwarding" in _lsrc)
check("...and stops relaying once the port answers, so the agent's screen is not interleaved",
      _lsrc.index("wait_until_ready(port, started)") < _lsrc.index("forwarding[0] = False"))
check("...while still draining the pipe, so a chatty server never stalls on a full buffer",
      "del tail[:-40]" in _lsrc)
_lcli = inspect.getsource(_cli.cmd_launch)
for _f in ("no_tune", "kv-bits", "no_full_layers", "trust_remote_code"):
    check(f"launch forwards {_f.replace('_','-')} to the server it starts", _f in _lcli)
check("--no-tune exists on run, serve and launch",
      all("--no-tune" in _p.parse_known_args([c, "m", "--no-tune"])[0].__dict__ or
          _p.parse_args([c, "m", "--no-tune"]).no_tune is True for c in ("run", "serve", "launch")))

print("\n" + "=" * 84)
print(f"{'ALL TESTS PASSED' if not FAIL else str(len(FAIL))+' FAILURES: '+', '.join(FAIL)}")
print("=" * 84)
sys.exit(1 if FAIL else 0)
