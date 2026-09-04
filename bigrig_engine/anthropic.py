"""The Anthropic Messages API, translated to and from what the engine speaks.

WHY THIS IS A SEPARATE PROTOCOL AND NOT A RENAME OF THE OPENAI ONE
    Claude Code -- and everything built on the Anthropic SDK -- cannot talk to an OpenAI-shaped
    endpoint. The differences are structural, not cosmetic:

      * `system` is a TOP-LEVEL field, not a message with role "system"
      * message content may be a bare string OR a list of typed blocks
      * `max_tokens` is REQUIRED; the OpenAI shape defaults it
      * the reply is a list of content blocks, not a `choices` array
      * streaming is seven distinct SSE EVENT TYPES with an `event:` line, where OpenAI's is one
        repeated chunk shape and a `[DONE]` sentinel

    Getting any one of those wrong produces a client that connects, sends a request, and then
    hangs or drops the response -- so the translation lives here, in one place, and is tested
    against the shapes the real clients send rather than against what is convenient to build.

WHAT THIS DELIBERATELY DOES NOT DO
    No authentication. This binds to localhost and serves a model on the user's own disk; an
    `x-api-key` check would be theatre. Any value in that header, including none, is accepted.
"""
from __future__ import annotations

import json
import time
import uuid

# Anthropic requires max_tokens. This is only the fallback ceiling, used when no session is on
# hand to say what the loaded model and the memory budget actually allow -- Session._token_ceiling
# is the real one. A fixed number here was both too low for a model with a 40,960-token context
# and far too high for the memory it would have taken to reach it.
MAX_TOKENS_LIMIT = 32768


class BadRequest(ValueError):
    """A request that cannot be served, with a message meant for whoever sent it."""


def _text_of(content) -> str:
    """Flatten Anthropic content -- a string, or a list of typed blocks -- into plain text.

    Non-text blocks (images, tool_use, tool_result) are skipped rather than rejected: a client
    that sends one should get an answer to the text it also sent, not an error about a block
    this engine has no way to render.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for b in content:
            if isinstance(b, str):
                out.append(b)
            elif isinstance(b, dict):
                if b.get("type") == "text" and isinstance(b.get("text"), str):
                    out.append(b["text"])
                elif b.get("type") == "tool_result":
                    out.append(_text_of(b.get("content")))
        return "".join(out)
    raise BadRequest(f"content must be a string or a list of blocks, got {type(content).__name__}")


def parse(body: dict, require_max_tokens: bool = True,
          max_tokens_limit: int | None = None) -> dict:
    """Validate an Anthropic request and return what the engine needs.

    Returns {messages, system, max_tokens, temperature, top_p, stream, stop_sequences}.
    """
    if not isinstance(body, dict):
        raise BadRequest("body must be a JSON object")
    msgs = body.get("messages")
    if not isinstance(msgs, list) or not msgs:
        raise BadRequest("`messages` must be a non-empty array")

    out_msgs = []
    for i, m in enumerate(msgs):
        if not isinstance(m, dict):
            raise BadRequest(f"messages[{i}] must be an object")
        role = m.get("role")
        if role not in ("user", "assistant"):
            raise BadRequest(
                f"messages[{i}].role must be 'user' or 'assistant', got {role!r}. "
                f"A system prompt goes in the top-level `system` field, not in messages.")
        if "content" not in m:
            raise BadRequest(f"messages[{i}] has no `content`")
        # A TURN CAN CARRY A CALL OR ITS RESULT, AND FLATTENING BOTH TO PROSE LOSES THE LOOP.
        #     Anthropic puts both in the content list: the assistant's `tool_use` blocks and the
        #     user's `tool_result` blocks. Chat templates do not read that shape -- they expect an
        #     assistant message with `tool_calls`, and a separate message with role "tool" holding
        #     each result, which is what this model's template branches on.
        #
        #     Flattened, the model saw its own call disappear and a result arrive from nowhere.
        #     It usually still answered, because the result text alone is often enough context --
        #     which is the worst kind of bug, one that works until the conversation is long enough
        #     or the call ambiguous enough that it does not.
        blocks = m["content"] if isinstance(m["content"], list) else []
        uses = [b for b in blocks if isinstance(b, dict) and b.get("type") == "tool_use"]
        results = [b for b in blocks if isinstance(b, dict) and b.get("type") == "tool_result"]
        text = _text_of(m["content"]) if not results else _text_of(
            [b for b in blocks if isinstance(b, dict) and b.get("type") == "text"])
        if role == "assistant" and uses:
            msg = {"role": "assistant", "content": text}
            msg["tool_calls"] = [{"type": "function", "function": {
                "name": u.get("name", ""), "arguments": u.get("input") or {}}} for u in uses]
            out_msgs.append(msg)
        elif results:
            # Each result is its own turn, in the order the calls were made. Any prose the user
            # sent alongside them stays a user turn so it is not silently attributed to a tool.
            if text.strip():
                out_msgs.append({"role": "user", "content": text})
            for r in results:
                out_msgs.append({"role": "tool", "content": _text_of(r.get("content"))})
        else:
            out_msgs.append({"role": role, "content": text})

    # max_tokens is REQUIRED by /v1/messages. Defaulting it would make us accept requests a real
    # Anthropic endpoint rejects, so a client tested against us would break against them.
    # /v1/messages/count_tokens does NOT require it -- counting the input has nothing to do with
    # how long the reply may be -- and demanding it there rejects perfectly valid requests.
    if "max_tokens" not in body:
        if require_max_tokens:
            raise BadRequest("`max_tokens` is required by the Messages API")
        mt = 1
    else:
        mt = body["max_tokens"]
    ceiling = int(max_tokens_limit) if max_tokens_limit else MAX_TOKENS_LIMIT
    if not isinstance(mt, int) or isinstance(mt, bool) or not 1 <= mt <= ceiling:
        raise BadRequest(f"`max_tokens` must be an integer in [1, {ceiling}], got {mt!r}")

    def num(key, default, lo, hi):
        v = body.get(key, default)
        if v is None:
            return default
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise BadRequest(f"`{key}` must be a number, got {v!r}")
        if not lo <= v <= hi:
            raise BadRequest(f"`{key}` must be between {lo} and {hi}, got {v}")
        return float(v)

    stops = body.get("stop_sequences") or []
    if not isinstance(stops, list) or any(not isinstance(x, str) for x in stops):
        raise BadRequest("`stop_sequences` must be an array of strings")

    return {"tools": tools_to_openai(body.get("tools")) or None,
            "messages": out_msgs, "system": _text_of(body.get("system")),
            "max_tokens": mt,
            # Anthropic's temperature range is [0, 1]; OpenAI's is [0, 2]. Honour theirs.
            "temperature": num("temperature", 0.7, 0.0, 1.0),
            "top_p": num("top_p", 0.95, 0.0, 1.0),
            "stream": bool(body.get("stream")),
            "stop_sequences": stops}


def to_engine_messages(parsed: dict) -> list:
    """Anthropic's top-level `system` becomes a leading system message for the chat template."""
    msgs = list(parsed["messages"])
    if parsed.get("system"):
        msgs.insert(0, {"role": "system", "content": parsed["system"]})
    return msgs


def new_id() -> str:
    return f"msg_{uuid.uuid4().hex[:24]}"


def tools_to_openai(tools) -> list:
    """Anthropic tool definitions in the shape chat templates expect.

    The two APIs describe the same thing differently: Anthropic sends
    `{name, description, input_schema}` flat, OpenAI wraps it as
    `{type: "function", function: {name, description, parameters}}`. Every chat template in
    mlx_lm renders the OpenAI shape, because that is what the models were trained on, so an
    Anthropic request is translated once here rather than in two endpoints.

    Anything already in the wrapped shape is passed through untouched -- some clients send it.
    """
    out = []
    for t in tools or []:
        if not isinstance(t, dict):
            raise BadRequest("each entry of `tools` must be an object")
        if t.get("type") == "function" and isinstance(t.get("function"), dict):
            out.append(t)
            continue
        name = t.get("name")
        if not isinstance(name, str) or not name:
            raise BadRequest("each tool needs a `name`")
        out.append({"type": "function", "function": {
            "name": name, "description": t.get("description") or "",
            "parameters": t.get("input_schema") or {"type": "object", "properties": {}}}})
    return out


def stop_reason(finish: str | None, tool_calls=None) -> str:
    if tool_calls:
        # The turn ended because the model wants a tool run, not because it ran out of room or
        # finished speaking. A client that loops on `end_turn` would stop here and never call it.
        return "tool_use"
    return {"length": "max_tokens", "stop": "end_turn", None: "end_turn"}.get(finish, "end_turn")


def message(rid: str, model: str, text: str, in_tok: int, out_tok: int,
            finish: str | None, extra: dict | None = None, tool_calls=None,
            reasoning: str = "") -> dict:
    # Content is a LIST of typed blocks here, not a string, so a reply that both says something
    # and calls a tool carries both -- and an empty text block is omitted rather than sent as "",
    # which some clients render as a blank assistant turn.
    blocks = []
    # A MODEL THAT THINKS FIRST GETS A THINKING BLOCK, which is this API's own shape for it and
    # goes before the text. Without it a reply that spent its whole budget reasoning came back as
    # a single empty text block -- and this is the endpoint `bigrig launch` points coding agents
    # at, so an empty assistant turn is what the agent would have received.
    if reasoning and reasoning.strip():
        blocks.append({"type": "thinking", "thinking": reasoning})
    if text and text.strip():
        blocks.append({"type": "text", "text": text})
    for c in tool_calls or []:
        blocks.append({"type": "tool_use", "id": f"toolu_{uuid.uuid4().hex[:24]}",
                       "name": c.get("name", ""), "input": c.get("arguments", {}) or {}})
    if not blocks:
        blocks = [{"type": "text", "text": text or ""}]
    m = {"id": rid, "type": "message", "role": "assistant",
         "content": blocks,
         "model": model, "stop_reason": stop_reason(finish, tool_calls), "stop_sequence": None,
         "usage": {"input_tokens": in_tok, "output_tokens": out_tok}}
    if extra:
        m["bigrig"] = extra          # unknown keys are ignored by conforming clients
    return m


# ------------------------------------------------------------------ streaming
def sse(event: str, data: dict) -> bytes:
    """One Anthropic SSE frame. The `event:` line is required -- their SDK dispatches on it."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n".encode()


def block_start(index: int, kind: str = "text") -> bytes:
    """Open one content block. `kind` is "text" or "thinking" -- a model that reasons before it
    answers produces both, and they are SEPARATE blocks in this protocol, not one block with two
    kinds of delta. A client reads `index` to know which block a delta belongs to."""
    body = {"type": "thinking", "thinking": ""} if kind == "thinking" else {"type": "text",
                                                                            "text": ""}
    return sse("content_block_start", {"type": "content_block_start", "index": index,
                                       "content_block": body})


def block_stop(index: int) -> bytes:
    return sse("content_block_stop", {"type": "content_block_stop", "index": index})


def start_frames(rid: str, model: str, in_tok: int, kind: str = "text") -> list:
    """message_start, then the first content block. Which kind that block is depends on what the
    model produces first: a thinking model's first token is reasoning, not an answer."""
    return [
        sse("message_start", {"type": "message_start", "message": {
            "id": rid, "type": "message", "role": "assistant", "content": [],
            "model": model, "stop_reason": None, "stop_sequence": None,
            "usage": {"input_tokens": in_tok, "output_tokens": 0}}}),
        block_start(0, kind),
    ]


def delta_frame(text: str, index: int = 0, kind: str = "text") -> bytes:
    """One delta on an open block. Anthropic names the reasoning delta `thinking_delta` and
    carries it in a `thinking` field, not `text` -- a client switching on the delta type would
    silently drop reasoning sent as text, and one rendering it as text would show the scratchpad
    as the answer, which is the thing this whole split exists to prevent.

    No `signature_delta` is sent. Anthropic's own models sign their reasoning; nothing here can
    produce a valid signature, and inventing one would be worse than omitting it.
    """
    if kind == "thinking":
        return sse("content_block_delta", {"type": "content_block_delta", "index": index,
                                           "delta": {"type": "thinking_delta",
                                                     "thinking": text}})
    return sse("content_block_delta", {"type": "content_block_delta", "index": index,
                                       "delta": {"type": "text_delta", "text": text}})


def tool_frames(call: dict, index: int) -> list:
    """One tool call as its own content block, started and stopped around its input.

    A `tool_use` block is a SEPARATE block from the text, not a delta on it, so the text block has
    to be closed before the first one opens -- which is why `end_frames` takes the index and the
    caller keeps count. Clients read `index` to know which block a delta belongs to, and two
    blocks sharing an index is how a reply ends up with a tool call attached to the wrong text.

    The input is sent whole in one `input_json_delta`. The schema permits it in fragments and
    clients accumulate them, but nothing here is streamed in fragments: a call is only recognised
    once its closing delimiter has arrived, so the entire argument object is known before the
    first frame can be written. Splitting it would imitate a liveness the parser does not have.
    """
    return [
        sse("content_block_start", {
            "type": "content_block_start", "index": index,
            "content_block": {"type": "tool_use", "id": f"toolu_{uuid.uuid4().hex[:24]}",
                              "name": call.get("name", ""), "input": {}}}),
        sse("content_block_delta", {
            "type": "content_block_delta", "index": index,
            "delta": {"type": "input_json_delta",
                      "partial_json": json.dumps(call.get("arguments", {}) or {})}}),
        sse("content_block_stop", {"type": "content_block_stop", "index": index}),
    ]


def end_frames(out_tok: int, finish: str | None, extra: dict | None = None,
               index: int | None = 0, tool_calls=None) -> list:
    """The frames that close a streamed message.

    `index` is the block still OPEN, or None when the caller has already closed everything it
    started. With no tool calls that is the text block, 0, exactly as before. With them, each
    `tool_frames` block closes itself, so passing an index here would emit a second
    `content_block_stop` for a block already stopped -- which is what this did, and it is the kind
    of protocol error a lenient client absorbs silently and a strict one rejects.
    """
    d = {"type": "message_delta",
         "delta": {"stop_reason": stop_reason(finish, tool_calls), "stop_sequence": None},
         "usage": {"output_tokens": out_tok}}
    if extra:
        d["bigrig"] = extra
    out = []
    if index is not None:
        out.append(sse("content_block_stop", {"type": "content_block_stop", "index": index}))
    out.append(sse("message_delta", d))
    out.append(sse("message_stop", {"type": "message_stop"}))
    return out


def count_tokens(tokenizer, parsed: dict) -> int:
    """Best-effort input token count for /v1/messages/count_tokens."""
    text = "\n".join(m["content"] for m in to_engine_messages(parsed))
    try:
        return len(tokenizer.encode(text))
    except Exception:
        return max(1, len(text) // 4)          # a coarse fallback beats failing the request
