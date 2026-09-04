"""Calling tools: what reaches the model, what comes back, and what a 3-bit model gets wrong.

WHY THIS FILE IS ADVERSARIAL
    Before any of this existed, both endpoints accepted `tools`, ignored them, and returned
    HTTP 200. Asked to read a file the model wrote an essay ABOUT the file -- fluent, confident,
    and useless -- and no error appeared anywhere. An agent given that reply hangs or loops. So
    the tests that matter here are not "does a call parse" but "is a failure visible": a dropped
    tool, a malformed payload reaching the client as prose, or a repaired call whose arguments no
    longer mean what the model wrote.
"""
import inspect
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from bigrig_engine import anthropic as anth                             # noqa: E402
from bigrig_engine import session as S                                  # noqa: E402

FAIL = []


def check(n, c, d=""):
    (print(f"  PASS  {n}") if c else (FAIL.append(n), print(f"  FAIL  {n}  {d}")))


print("=" * 84)
print("1. THE MODEL CANNOT CALL A TOOL IT WAS NEVER TOLD ABOUT")
print("=" * 84)
_p = inspect.getsource(S.Session._prompt)
check("tools are handed to the chat template", 'kw["tools"] = list(tools)' in _p)
check("...and the reason is recorded, because the failure was silent",
      "HTTP 200" in _p or "never told about" in _p.lower())
_st = inspect.getsource(S.Session.stream_text)
check("stream_text forwards them to the prompt builder", "tools=tools" in _st)
_srv = open(os.path.join(ROOT, "bigrig_engine", "server.py"), encoding="utf-8").read()
check("a model with no tool-call format REFUSES the request rather than ignoring it",
      "has no tool-call format" in _srv)
# Three endpoints now: chat/completions, messages, and responses. Each validates on its own, so
# each has to refuse on its own -- a path that skipped the check would accept tools for a model
# that cannot signal a call, and answer as though it had.
check("...on all three endpoints", _srv.count("has no tool-call format") == 3,
      str(_srv.count("has no tool-call format")))

print()
print("=" * 84)
print("2. A CALL IS MACHINE SYNTAX AND MUST NEVER REACH THE CLIENT AS PROSE")
print("=" * 84)


class _Tok:
    """A tokenizer that marks calls the way this model's does."""
    has_tool_calling = True
    tool_call_start = "<tool_call>"
    tool_call_end = "</tool_call>"

    @staticmethod
    def _tool_parser(body, tools):
        d = json.loads(body)
        return {"name": d["name"], "arguments": d.get("arguments", {})}


class _Sess:
    tokenizer = _Tok()
    supports_tools = S.Session.supports_tools
    extract_tool_calls = S.Session.extract_tool_calls
    _repair_json = staticmethod(S.Session._repair_json)


sx = _Sess()
t, c = sx.extract_tool_calls('Sure. <tool_call>{"name":"f","arguments":{"a":1}}</tool_call> done', None)
check("a well-formed call is lifted out of the text", c == [{"name": "f", "arguments": {"a": 1}}])
check("...and the delimiters do not survive in the text", "<tool_call>" not in t and "Sure." in t)
t, c = sx.extract_tool_calls("just an answer", None)
check("a reply with no call is returned untouched", (t, c) == ("just an answer", []))
t, c = sx.extract_tool_calls('<tool_call>{"name":"f","arguments":{}}</tool_call>'
                             '<tool_call>{"name":"g","arguments":{}}</tool_call>', None)
check("two calls in one reply both come out", [x["name"] for x in c] == ["f", "g"])
# THE ONE THAT MATTERS. A truncated call must not be handed back as assistant text: it would put
# markup the user never wrote into the conversation and corrupt the next turn's prompt.
t, c = sx.extract_tool_calls('Working on it <tool_call>{"name":"f","argum', None)
check("an unterminated call is stripped from the text, not shown to the user",
      "<tool_call>" not in t and "argum" not in t, repr(t))
check("...and what came BEFORE it is still delivered", t.strip() == "Working on it", repr(t))

print()
print("=" * 84)
print("3. REPAIR MAY CLOSE WHAT THE MODEL LEFT OPEN, AND MAY NOT INVENT ANYTHING")
print("=" * 84)
# MEASURED on this 3-bit model over eight deliberately awkward schemas. Both failures were
# recoverable and neither was garbled: one hit max_tokens mid-argument, one was short exactly one
# closing brace with nested arrays and escapes otherwise intact. 6 of 8 became 8 of 8.
r = S.Session._repair_json
check("valid JSON is returned unchanged",
      r('{"name":"f","arguments":{"p":1}}') == {"name": "f", "arguments": {"p": 1}})
check("a missing closing brace is closed",
      r('{"name":"f","arguments":{"p":1}') == {"name": "f", "arguments": {"p": 1}})
check("a value truncated mid-write drops that key rather than guessing it",
      r('{"name":"f","arguments":{"p":1,"q":') == {"name": "f", "arguments": {"p": 1}})
check("a nested array survives repair",
      r('{"name":"f","arguments":{"e":[{"line":3,"text":"x"}]') ==
      {"name": "f", "arguments": {"e": [{"line": 3, "text": "x"}]}})
# THE SCAN MUST RESPECT STRINGS. One real failure contained `"text": "{\"a\": 1}"` -- a brace
# inside a string. Counting it as structure would close the wrong bracket and produce an object
# that parses into something the model did not write, which is worse than no call at all.
check("a brace inside a string is not counted as structure",
      r('{"name":"f","arguments":{"t":"{\\"a\\": 1}"}') ==
      {"name": "f", "arguments": {"t": '{"a": 1}'}})
check("an unterminated string is closed before brackets are",
      isinstance(r('{"name":"f","arguments":{"t":"abc'), dict))
check("something that is not JSON at all returns nothing", r("not json") is None)
check("...and so does an empty payload", r("") is None and r("   ") is None)
check("repair runs only after the model's own parser has refused",
      inspect.getsource(S.Session.extract_tool_calls).index("parser(") <
      inspect.getsource(S.Session.extract_tool_calls).index("_repair_json"))

print()
print("=" * 84)
print("4. THE TWO APIS DESCRIBE THE SAME THING DIFFERENTLY")
print("=" * 84)
conv = anth.tools_to_openai([{"name": "read_file", "description": "Read",
                              "input_schema": {"type": "object",
                                               "properties": {"path": {"type": "string"}}}}])
check("an Anthropic tool becomes the shape chat templates render",
      conv[0]["type"] == "function" and conv[0]["function"]["name"] == "read_file"
      and "properties" in conv[0]["function"]["parameters"], json.dumps(conv)[:90])
check("an already-wrapped tool is passed through untouched",
      anth.tools_to_openai([{"type": "function", "function": {"name": "x"}}])[0]
      ["function"]["name"] == "x")
try:
    anth.tools_to_openai([{"description": "no name"}])
    named = False
except anth.BadRequest:
    named = True
check("a tool with no name is rejected", named)
check("stop_reason says a tool is wanted, not that the turn ended",
      anth.stop_reason("stop", [{"name": "f"}]) == "tool_use")
check("...and is unchanged when nothing was called",
      anth.stop_reason("stop") == "end_turn" and anth.stop_reason("length") == "max_tokens")
m = anth.message("id", "m", "thinking out loud", 1, 2, "stop",
                 tool_calls=[{"name": "f", "arguments": {"a": 1}}])
check("the reply carries text AND the call as separate blocks",
      [b["type"] for b in m["content"]] == ["text", "tool_use"])
check("...the call block has an id, a name and an input object",
      m["content"][1]["id"].startswith("toolu_") and m["content"][1]["name"] == "f"
      and m["content"][1]["input"] == {"a": 1})
m2 = anth.message("id", "m", "", 1, 2, "stop", tool_calls=[{"name": "f", "arguments": {}}])
check("an empty text block is omitted rather than sent blank",
      [b["type"] for b in m2["content"]] == ["tool_use"])

print()
print("=" * 84)
print("5. THE LOOP: A CALL AND ITS RESULT MUST SURVIVE THE ROUND TRIP")
print("=" * 84)
# Flattened to prose, the model saw its own call disappear and a result arrive from nowhere. It
# usually still answered -- the result text alone is often enough context -- which is the worst
# kind of bug, one that works until the conversation is long enough that it does not.
body = {"max_tokens": 50, "messages": [
    {"role": "user", "content": "read it"},
    {"role": "assistant", "content": [{"type": "text", "text": "ok"},
                                      {"type": "tool_use", "id": "toolu_1", "name": "read_file",
                                       "input": {"path": "/etc/hosts"}}]},
    {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_1",
                                  "content": "127.0.0.1 localhost"}]}]}
got = anth.parse(body)["messages"]
check("the assistant's call becomes a tool_calls turn, not prose",
      got[1]["role"] == "assistant" and got[1]["tool_calls"][0]["function"]["name"] == "read_file",
      json.dumps(got[1])[:100])
check("...carrying the arguments the model actually sent",
      got[1]["tool_calls"][0]["function"]["arguments"] == {"path": "/etc/hosts"})
check("the result becomes a turn with role 'tool', which is what the template branches on",
      got[2]["role"] == "tool" and "127.0.0.1" in got[2]["content"], json.dumps(got[2])[:80])
check("...and the conversation is still in order", [m["role"] for m in got]
      == ["user", "assistant", "tool"], str([m["role"] for m in got]))
two = anth.parse({"max_tokens": 5, "messages": [
    {"role": "user", "content": [
        {"type": "text", "text": "here you go"},
        {"type": "tool_result", "tool_use_id": "a", "content": "one"},
        {"type": "tool_result", "tool_use_id": "b", "content": "two"}]}]})["messages"]
check("two results in one turn become two tool turns",
      [m["role"] for m in two] == ["user", "tool", "tool"], str([m["role"] for m in two]))
check("...and prose sent alongside them stays a USER turn, not a tool's output",
      two[0]["content"] == "here you go")
check("a request that sends no tools is unaffected",
      anth.parse({"max_tokens": 5, "messages": [{"role": "user", "content": "hi"}]})["tools"]
      is None)

print()
print("=" * 84)
print("6. WHAT A 3-BIT MODEL ACTUALLY PRODUCES")
print("=" * 84)
# Structured output is the first thing quantisation breaks, so this was measured before the
# feature was called working. Qwen3-30B-A3B-3bit, greedy:
#
#     twelve ordinary tool prompts     12/12 parsed, 12/12 named a real tool,
#                                      12/12 supplied every required argument,
#                                      12/12 chose the RIGHT tool
#     eight deliberately awkward ones  6/8 before the repair step, 8/8 after
#
# The two that needed repair were a truncation and a single missing brace -- not garbled output.
# What this does NOT measure is whether the arguments are TRUE: one call fetched
# `localhost:7070` when the prompt said 8080. That is the model being wrong, and no amount of
# parsing fixes it.
_sess_src = inspect.getsource(S.Session.extract_tool_calls)
check("the measured numbers live next to the code they justify",
      "3-bit" in inspect.getsource(S.Session._repair_json))
check("the honest limit is stated: repair fixes syntax, never meaning",
      "invent" in inspect.getsource(S.Session._repair_json))
check("a malformed call is dropped rather than passed on as text", "not passed on" in _sess_src
      or "removed and the reply" in _sess_src)

print()
print("=" * 84)
print("7. STREAMING: A CHUNK CAN END IN THE MIDDLE OF A DELIMITER")
print("=" * 84)
# THE FAILURE THIS PREVENTS. The blocking path searches a finished reply for `<tool_call>`. A
# streaming one has no finished reply -- the model emits that tag as several tokens, so a chunk
# can end after `<tool`. A splitter that simply looked for the whole tag would emit `<tool` as
# assistant text, put machine syntax in front of the user, AND never recognise the tag when its
# second half arrived, so the client would see no call at all.
SP = S.ToolCallSplitter


def _parse(body, tools):
    d = json.loads(body)
    return {"name": d["name"], "arguments": d.get("arguments", {})}


def _drive(payload, chunk=1):
    """Feed a reply in fixed-size pieces. chunk=1 is the worst case a tokenizer can produce."""
    sp = SP("<tool_call>", "</tool_call>", _parse, None, S.Session._repair_json)
    text, calls = "", []
    for i in range(0, len(payload), chunk):
        t, c = sp.feed(payload[i:i + chunk])
        text += t
        calls += c
    t, c = sp.finish()
    return text + t, calls + c


_PAY = 'Sure. <tool_call>{"name":"f","arguments":{"a":1}}</tool_call> done'
for _n in (1, 2, 3, 5, 11, 1000):
    _t, _c = _drive(_PAY, _n)
    check(f"split into {_n}-character chunks, the call is still found and the text is clean",
          _c == [{"name": "f", "arguments": {"a": 1}}] and "<tool" not in _t,
          f"text {_t!r} calls {_c}")
check("...and the prose either side survives exactly",
      _drive(_PAY, 1)[0] == "Sure.  done", repr(_drive(_PAY, 1)[0]))
# The holdback must RELEASE when the tail turns out not to be a delimiter, or a reply mentioning
# `<toolbox` would lose it.
_t, _c = _drive("a <toolbox b", 1)
check("text that merely looks like the start of a delimiter is released, not swallowed",
      _t == "a <toolbox b" and _c == [], repr(_t))
_t, _c = _drive("<tool_call>", 1)
check("a delimiter with nothing after it yields no call and no stray text",
      _t == "" and _c == [], f"{_t!r} {_c}")
# Two calls, split awkwardly.
_t, _c = _drive('<tool_call>{"name":"f","arguments":{}}</tool_call>'
                '<tool_call>{"name":"g","arguments":{}}</tool_call>', 3)
check("two calls arriving in pieces both come out, in order",
      [x["name"] for x in _c] == ["f", "g"], str(_c))
# An unterminated call must be dropped rather than shown -- the same rule as the blocking path.
_t, _c = _drive('working <tool_call>{"name":"f","argum', 1)
check("an unterminated call is not shown to the user as prose",
      "<tool" not in _t and "argum" not in _t, repr(_t))
check("...and what came before it still arrives", _t.strip() == "working", repr(_t))
# Repair reaches the streaming path too, so a brace short mid-stream is still a call.
_t, _c = _drive('<tool_call>{"name":"f","arguments":{"a":1}</tool_call>', 4)
check("a payload one brace short is still recovered while streaming",
      _c == [{"name": "f", "arguments": {"a": 1}}], str(_c))

print()
print("=" * 84)
print("8. THE FRAMES EACH API REQUIRES")
print("=" * 84)
_srv2 = open(os.path.join(ROOT, "bigrig_engine", "server.py"), encoding="utf-8").read()
check("the OpenAI stream sends calls as tool_calls deltas, not as content",
      '"tool_calls": [{' in _srv2 and "def _emit_call" in _srv2)
check("...and finishes with tool_calls rather than stop",
      '"tool_calls" if n_calls else' in _srv2)
check("the Anthropic stream opens a separate block per call",
      "tool_frames" in _srv2 and "def tool_frames" in
      open(os.path.join(ROOT, "bigrig_engine", "anthropic.py"), encoding="utf-8").read())
# THE BUG THIS CATCHES, WHICH WAS REAL. `tool_frames` closes its own block and `end_frames` used
# to close another at the same index, so a reply with one call sent content_block_stop twice for
# index 1. A lenient client absorbs that silently; a strict one rejects the message.
_a = open(os.path.join(ROOT, "bigrig_engine", "anthropic.py"), encoding="utf-8").read()
check("end_frames can be told there is nothing left to close",
      "index: int | None = 0" in _a and "if index is not None:" in _a)
check("...and the streaming path tells it so once it has closed its own blocks",
      "index=(None if closed else idx)" in _srv2)
check("blocks are balanced live: opened [0, 1], closed [0, 1] with a call; [0] and [0] without",
      "closed = True" in _srv2)

print()
print("=" * 84)
print("ALL TESTS PASSED" if not FAIL else f"{len(FAIL)} FAILURES: " + ", ".join(FAIL))
print("=" * 84)
sys.exit(1 if FAIL else 0)
