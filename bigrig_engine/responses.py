"""OpenAI's Responses API, which is the only wire protocol Codex CLI still speaks.

WHY THIS EXISTS RATHER THAN A DELETED LAUNCHER
    `bigrig launch codex` pointed OPENAI_BASE_URL at this server and expected
    /v1/chat/completions. It has been dead for a while and nobody noticed, because a launcher
    that starts an agent which then fails to connect looks like the agent's problem.

    Confirmed rather than assumed, by reading the shipped binary of codex-cli 0.152.0:

        `wire_api = "chat"` is no longer supported.
        `wire_api = "responses"` in your provider config.

    and the binary names `/responses` twenty-six times and `/chat/completions` not once.

WHAT IS AND IS NOT IMPLEMENTED
    Enough of the API for an agent to hold a conversation and call tools: text in, text and
    function calls out, streamed or not. Stateful features are NOT implemented and are refused
    rather than ignored -- `previous_response_id` asks this server to remember a conversation it
    never stored, and silently starting a fresh one would give the client an answer with no
    context and no way to tell.
"""
from __future__ import annotations

import json
import time
import uuid


class BadRequest(ValueError):
    """A request that is wrong in a way the caller can fix."""


def new_id() -> str:
    return f"resp_{uuid.uuid4().hex[:24]}"


def _text_of(content) -> str:
    """Flatten one item's content, which may be a string or a list of typed parts."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for p in content:
            if isinstance(p, str):
                out.append(p)
            elif isinstance(p, dict):
                # input_text and output_text are the same thing pointing in two directions; a
                # conversation replayed back to us contains both.
                if p.get("type") in ("input_text", "output_text", "text", "summary_text"):
                    out.append(p.get("text") or "")
        return "".join(out)
    return ""


def parse(body: dict, max_tokens_limit: int | None = None) -> dict:
    """Validate a Responses request and return what the engine needs.

    `input` is either a bare string -- the whole user turn -- or a list of items, which is how a
    multi-turn conversation and its tool results come back. Both are converted to the ordinary
    message list every other endpoint here produces, so nothing downstream has to know which
    API a request arrived on.
    """
    if not isinstance(body, dict):
        raise BadRequest("the request body must be a JSON object")
    if body.get("previous_response_id"):
        raise BadRequest(
            "`previous_response_id` is not supported: this server keeps no conversation state, "
            "so it cannot continue one it never stored. Send the full `input` instead.")

    msgs = []
    inp = body.get("input")
    if isinstance(inp, str):
        msgs.append({"role": "user", "content": inp})
    elif isinstance(inp, list):
        for i, item in enumerate(inp):
            if not isinstance(item, dict):
                raise BadRequest(f"input[{i}] must be an object")
            t = item.get("type") or "message"
            if t == "message":
                role = item.get("role") or "user"
                if role not in ("user", "assistant", "system", "developer"):
                    raise BadRequest(f"input[{i}].role is not one this server accepts: {role!r}")
                # `developer` is the Responses API's name for a system turn.
                msgs.append({"role": "system" if role == "developer" else role,
                             "content": _text_of(item.get("content"))})
            elif t == "function_call":
                # The model's own previous call, replayed. Rendered as the assistant turn it was,
                # so the template sees a call followed by its result rather than a result alone.
                args = item.get("arguments")
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except ValueError:
                        args = {}
                msgs.append({"role": "assistant", "content": "",
                             "tool_calls": [{"type": "function", "function": {
                                 "name": item.get("name", ""), "arguments": args or {}}}]})
            elif t == "function_call_output":
                msgs.append({"role": "tool", "content": _text_of(item.get("output"))})
            elif t in ("reasoning", "item_reference"):
                continue                      # nothing here can render it; dropping is honest
            else:
                raise BadRequest(f"input[{i}].type {t!r} is not supported by this server")
    elif inp is not None:
        raise BadRequest("`input` must be a string or a list of items")
    if not msgs:
        raise BadRequest("`input` is required and must contain at least one message")

    if body.get("instructions"):
        msgs.insert(0, {"role": "system", "content": str(body["instructions"])})

    mt = body.get("max_output_tokens")
    ceiling = int(max_tokens_limit) if max_tokens_limit else 8192
    if mt is None:
        mt = min(1024, ceiling)
    if isinstance(mt, bool) or not isinstance(mt, int) or not 1 <= mt <= ceiling:
        raise BadRequest(f"`max_output_tokens` must be an integer in [1, {ceiling}], got {mt!r}")

    def num(key, default, lo, hi):
        v = body.get(key, default)
        if v is None:
            return default
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise BadRequest(f"`{key}` must be a number, got {v!r}")
        if not lo <= v <= hi:
            raise BadRequest(f"`{key}` must be between {lo} and {hi}, got {v}")
        return float(v)

    return {"messages": msgs, "max_tokens": mt,
            "temperature": num("temperature", 0.7, 0.0, 2.0),
            "top_p": num("top_p", 0.95, 0.0, 1.0),
            "stream": bool(body.get("stream", False)),
            "tools": tools_to_openai(body.get("tools"))}


def tools_to_openai(tools, _depth: int = 0) -> list | None:
    """Every tool Codex sends, rendered as the function tools a chat template understands.

    THE FIRST VERSION REJECTED WHAT IT DID NOT RECOGNISE, AND THAT STOPPED THE AGENT DEAD.
        Codex CLI 0.152.0 was pointed at this server and answered with

            ERROR: tool type 'namespace' is not available on a local model

        which was this function refusing the whole request over one entry. It sends four kinds:
        `function`, `custom`, `local_shell`, and `namespace` -- and a namespace is not a tool at
        all, it is a CONTAINER holding them ("dynamic tool namespace must contain at least one
        tool", from the binary). Refusing it threw away every real tool inside it.

        So nothing is refused for its type. A namespace is flattened, anything carrying a name is
        rendered as a function with whatever schema it came with, and anything nameless -- the
        platform tools like `web_search`, which OpenAI's own infrastructure executes -- is
        skipped. A model offered a tool it cannot use simply does not call it, which is a better
        outcome than an agent that cannot start; and every tool it CAN call is executed by the
        client, so the only question is whether the model can be told the tool exists.
    """
    if not tools:
        return None
    if _depth > 3:
        raise BadRequest("`tools` is nested more deeply than this server will follow")
    out = []
    for t in tools:
        if not isinstance(t, dict):
            raise BadRequest("each entry of `tools` must be an object")
        kind = t.get("type")
        if kind == "function" and isinstance(t.get("function"), dict):
            out.append(t)
            continue
        if kind == "namespace":
            inner = t.get("tools") or t.get("functions") or []
            prefix = t.get("name") or ""
            for sub in tools_to_openai(inner, _depth + 1) or []:
                # Kept distinct, because two namespaces may each hold a `search` and a model
                # given the same name twice cannot say which one it meant.
                if prefix and not sub["function"]["name"].startswith(prefix):
                    sub = {**sub, "function": {**sub["function"],
                                               "name": f"{prefix}.{sub['function']['name']}"}}
                out.append(sub)
            continue
        name = t.get("name")
        if not isinstance(name, str) or not name:
            # A PLATFORM TOOL, WHICH THIS SERVER CANNOT PROVIDE AND MUST NOT REFUSE OVER.
            #     `web_search` and its relatives arrive with no name and no schema because they
            #     are executed by OpenAI's infrastructure, not by the client and not here. There
            #     is nothing to render into a prompt and nothing for the model to call.
            #
            #     Skipped, not refused. Refusing was the second thing that stopped Codex dead
            #     -- after `namespace` -- and each refusal threw away every OTHER tool in the
            #     same request. A model simply not offered web search does not call it, which is
            #     the truthful outcome; a model whose entire toolset was rejected cannot work.
            continue
        params = t.get("parameters") or t.get("input_schema")
        if not isinstance(params, dict):
            # `local_shell` and `custom` carry no JSON schema. A permissive object is honest
            # about that: the model is told the tool exists and the client validates the call.
            params = {"type": "object", "properties": {}}
        out.append({"type": "function", "function": {
            "name": name, "description": t.get("description") or "", "parameters": params}})
    return out


def _output_items(text: str, tool_calls) -> list:
    items = []
    if text and text.strip():
        items.append({"type": "message", "id": f"msg_{uuid.uuid4().hex[:24]}",
                      "status": "completed", "role": "assistant",
                      "content": [{"type": "output_text", "text": text, "annotations": []}]})
    for c in tool_calls or []:
        items.append({"type": "function_call", "id": f"fc_{uuid.uuid4().hex[:24]}",
                      "call_id": f"call_{uuid.uuid4().hex[:24]}", "status": "completed",
                      "name": c.get("name", ""),
                      "arguments": json.dumps(c.get("arguments", {}) or {})})
    return items


def response(rid: str, model: str, text: str, in_tok: int, out_tok: int, finish: str | None,
             tool_calls=None, extra: dict | None = None) -> dict:
    """A finished response. `status` is "incomplete" when the reply was cut, not "completed"."""
    incomplete = finish == "length"
    d = {"id": rid, "object": "response", "created_at": int(time.time()),
         "status": "incomplete" if incomplete else "completed",
         "model": model, "output": _output_items(text, tool_calls),
         "parallel_tool_calls": True, "tool_choice": "auto", "tools": [],
         "usage": {"input_tokens": in_tok, "output_tokens": out_tok,
                   "total_tokens": in_tok + out_tok}}
    if incomplete:
        d["incomplete_details"] = {"reason": "max_output_tokens"}
    if extra:
        d["bigrig"] = extra
    return d


def sse(event: str, data: dict) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n".encode()


def stream_frames(rid: str, model: str):
    """The event sequence a Responses client expects, as a small state machine.

    Returns (created, text_delta, finish) callables. The output item and content part have to be
    ANNOUNCED before any delta referring to them, and closed after -- a client that receives a
    delta for an item it was never told about has nowhere to put the text.
    """
    seq = {"n": 0}

    def nxt():
        seq["n"] += 1
        return seq["n"]

    def base(status="in_progress", output=None):
        return {"id": rid, "object": "response", "created_at": int(time.time()),
                "status": status, "model": model, "output": output or []}

    def created():
        item = {"type": "message", "id": f"msg_{uuid.uuid4().hex[:24]}",
                "status": "in_progress", "role": "assistant", "content": []}
        return [sse("response.created", {"type": "response.created",
                                         "sequence_number": nxt(), "response": base()}),
                sse("response.in_progress", {"type": "response.in_progress",
                                             "sequence_number": nxt(), "response": base()}),
                sse("response.output_item.added",
                    {"type": "response.output_item.added", "sequence_number": nxt(),
                     "output_index": 0, "item": item}),
                sse("response.content_part.added",
                    {"type": "response.content_part.added", "sequence_number": nxt(),
                     "item_id": item["id"], "output_index": 0, "content_index": 0,
                     "part": {"type": "output_text", "text": "", "annotations": []}})], item

    def text_delta(item_id, chunk):
        return sse("response.output_text.delta",
                   {"type": "response.output_text.delta", "sequence_number": nxt(),
                    "item_id": item_id, "output_index": 0, "content_index": 0, "delta": chunk})

    def call_frames(call, index):
        item = {"type": "function_call", "id": f"fc_{uuid.uuid4().hex[:24]}",
                "call_id": f"call_{uuid.uuid4().hex[:24]}", "status": "in_progress",
                "name": call.get("name", ""), "arguments": ""}
        args = json.dumps(call.get("arguments", {}) or {})
        done = dict(item, status="completed", arguments=args)
        return [sse("response.output_item.added",
                    {"type": "response.output_item.added", "sequence_number": nxt(),
                     "output_index": index, "item": item}),
                sse("response.function_call_arguments.delta",
                    {"type": "response.function_call_arguments.delta",
                     "sequence_number": nxt(), "item_id": item["id"],
                     "output_index": index, "delta": args}),
                sse("response.function_call_arguments.done",
                    {"type": "response.function_call_arguments.done",
                     "sequence_number": nxt(), "item_id": item["id"],
                     "output_index": index, "arguments": args}),
                sse("response.output_item.done",
                    {"type": "response.output_item.done", "sequence_number": nxt(),
                     "output_index": index, "item": done})], done

    def finish(item, text, calls_done, in_tok, out_tok, cut):
        item = dict(item, status="completed",
                    content=[{"type": "output_text", "text": text, "annotations": []}])
        out = ([item] if text and text.strip() else []) + list(calls_done)
        r = base("incomplete" if cut else "completed", out)
        r["usage"] = {"input_tokens": in_tok, "output_tokens": out_tok,
                      "total_tokens": in_tok + out_tok}
        if cut:
            r["incomplete_details"] = {"reason": "max_output_tokens"}
        frames = []
        if text and text.strip():
            frames += [
                sse("response.output_text.done",
                    {"type": "response.output_text.done", "sequence_number": nxt(),
                     "item_id": item["id"], "output_index": 0, "content_index": 0, "text": text}),
                sse("response.content_part.done",
                    {"type": "response.content_part.done", "sequence_number": nxt(),
                     "item_id": item["id"], "output_index": 0, "content_index": 0,
                     "part": {"type": "output_text", "text": text, "annotations": []}}),
                sse("response.output_item.done",
                    {"type": "response.output_item.done", "sequence_number": nxt(),
                     "output_index": 0, "item": item})]
        frames.append(sse("response.completed" if not cut else "response.incomplete",
                          {"type": "response.completed" if not cut else "response.incomplete",
                           "sequence_number": nxt(), "response": r}))
        return frames

    return created, text_delta, call_frames, finish
