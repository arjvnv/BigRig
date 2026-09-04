"""OpenAI's Responses API, which is the only wire protocol Codex CLI still speaks.

WHY THIS FILE IS ADVERSARIAL
    `bigrig launch codex` had been dead for a while and nobody noticed, because a launcher that
    starts an agent which then fails to connect looks like the agent's problem. Confirmed rather
    than assumed, by reading the shipped binary of codex-cli 0.152.0:

        `wire_api = "chat"` is no longer supported.
        `wire_api = "responses"` in your provider config.

    and the binary names `/responses` twenty-six times and `/chat/completions` not once.

    Then the endpoint was written and pointed at the real client, which refused it twice more --
    each time over a tool this server had decided it would not accept. Both refusals threw away
    every OTHER tool in the same request. Those two are the tests that matter here.
"""
import inspect
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from bigrig_engine import responses as R                                # noqa: E402

FAIL = []


def check(n, c, d=""):
    (print(f"  PASS  {n}") if c else (FAIL.append(n), print(f"  FAIL  {n}  {d}")))


print("=" * 84)
print("1. INPUT ARRIVES IN TWO SHAPES AND BOTH BECOME ORDINARY MESSAGES")
print("=" * 84)
p = R.parse({"input": "hello", "max_output_tokens": 64})
check("a bare string is the whole user turn",
      p["messages"] == [{"role": "user", "content": "hello"}], str(p["messages"]))
p = R.parse({"input": [{"type": "message", "role": "user",
                        "content": [{"type": "input_text", "text": "hi"}]}],
             "max_output_tokens": 8})
check("a list of items is flattened to the same thing",
      p["messages"] == [{"role": "user", "content": "hi"}], str(p["messages"]))
check("`instructions` becomes the system turn, at the front",
      R.parse({"input": "x", "instructions": "be terse", "max_output_tokens": 8})
      ["messages"][0] == {"role": "system", "content": "be terse"})
check("`developer` is the Responses API's name for a system turn",
      R.parse({"input": [{"type": "message", "role": "developer", "content": "rules"}],
               "max_output_tokens": 8})["messages"][0]["role"] == "system")
# A conversation comes BACK with the model's own calls in it. Rendering a result without the call
# that produced it leaves the model reading an answer to a question it cannot see it asked.
p = R.parse({"max_output_tokens": 8, "input": [
    {"type": "message", "role": "user", "content": "read it"},
    {"type": "function_call", "name": "read_file", "arguments": '{"path": "/etc/hosts"}'},
    {"type": "function_call_output", "call_id": "c1", "output": "127.0.0.1 localhost"}]})
check("a replayed call becomes the assistant turn it was",
      p["messages"][1]["role"] == "assistant"
      and p["messages"][1]["tool_calls"][0]["function"]["name"] == "read_file",
      json.dumps(p["messages"][1])[:90])
check("...with its arguments decoded from the string the API sends them as",
      p["messages"][1]["tool_calls"][0]["function"]["arguments"] == {"path": "/etc/hosts"})
check("...and its result becomes a turn with role 'tool'",
      p["messages"][2] == {"role": "tool", "content": "127.0.0.1 localhost"},
      str(p["messages"][2]))
check("reasoning items are dropped rather than rendered as prose",
      len(R.parse({"max_output_tokens": 8, "input": [
          {"type": "reasoning", "summary": []},
          {"type": "message", "role": "user", "content": "hi"}]})["messages"]) == 1)

print()
print("=" * 84)
print("2. WHAT IS REFUSED, AND WHAT MUST NOT BE")
print("=" * 84)
# Refusing a request this server genuinely cannot honour is right. Refusing one it CAN honour,
# over one entry it did not recognise, is what stopped the real client twice.
for body, why in (({"max_output_tokens": 8}, "no input"),
                  ({"input": "x", "max_output_tokens": 0}, "max_output_tokens of zero"),
                  ({"input": "x", "max_output_tokens": 8, "temperature": 9}, "temperature 9"),
                  ({"input": 42, "max_output_tokens": 8}, "input is a number")):
    try:
        R.parse(body)
        ok = False
    except R.BadRequest:
        ok = True
    check(f"rejected: {why}", ok)
# STATE. `previous_response_id` asks this server to continue a conversation it never stored.
# Silently starting a fresh one would hand back an answer with no context and no way to tell.
try:
    R.parse({"input": "x", "max_output_tokens": 8, "previous_response_id": "resp_1"})
    stateful = False
except R.BadRequest as e:
    stateful = "keeps no conversation state" in str(e)
check("a request for server-side state is refused, and says why", stateful)

print()
print("=" * 84)
print("3. THE TOOLS CODEX ACTUALLY SENDS")
print("=" * 84)
# THE TWO REFUSALS, IN ORDER. Each was this server rejecting the whole request over one entry.
#
#   ERROR: tool type 'namespace' is not available on a local model
#   ERROR: a tool of type 'web_search' arrived with no `name`
#
# A namespace is not a tool, it is a CONTAINER -- "dynamic tool namespace must contain at least
# one tool", from the binary -- so refusing it threw away every real tool inside it. And
# `web_search` is executed by OpenAI's own infrastructure, so there is nothing to render and
# nothing to call; refusing over it threw away the rest of the toolset with it.
f = R.tools_to_openai
check("a namespace is flattened, not refused",
      [t["function"]["name"] for t in f([{"type": "namespace", "name": "fs", "tools": [
          {"type": "function", "name": "read"}, {"type": "function", "name": "write"}]}])]
      == ["fs.read", "fs.write"])
check("...with the namespace kept in the name, so two namespaces may both hold a `search`",
      f([{"type": "namespace", "name": "a", "tools": [{"type": "function", "name": "search"}]},
         {"type": "namespace", "name": "b", "tools": [{"type": "function", "name": "search"}]}])
      [0]["function"]["name"] != f([{"type": "namespace", "name": "a", "tools": [
          {"type": "function", "name": "search"}]},
          {"type": "namespace", "name": "b",
           "tools": [{"type": "function", "name": "search"}]}])[1]["function"]["name"])
check("a nameless platform tool is skipped, not refused",
      f([{"type": "web_search"}]) == [])
check("...and skipping it does not take the rest of the request with it",
      [t["function"]["name"] for t in f([{"type": "web_search"},
                                         {"type": "function", "name": "read"}])] == ["read"])
check("local_shell and custom are rendered as functions, so the model is told they exist",
      f([{"type": "local_shell", "name": "shell"}])[0]["function"]["name"] == "shell")
check("...with a permissive schema, since they carry none",
      f([{"type": "local_shell", "name": "shell"}])[0]["function"]["parameters"]["type"]
      == "object")
check("an already-wrapped function tool is passed through untouched",
      f([{"type": "function", "function": {"name": "g"}}])[0]["function"]["name"] == "g")
check("nesting is bounded, so a malicious payload cannot recurse forever",
      "_depth > 3" in inspect.getsource(R.tools_to_openai))
check("no tools at all is None rather than an empty list, which the template treats differently",
      f([]) is None and f(None) is None)

print()
print("=" * 84)
print("4. THE RESPONSE SHAPE")
print("=" * 84)
r = R.response("resp_1", "m", "hello", 3, 4, "stop")
check("a plain reply is one message item", [i["type"] for i in r["output"]] == ["message"])
check("...carrying output_text", r["output"][0]["content"][0]["type"] == "output_text")
check("...and status completed", r["status"] == "completed")
r = R.response("resp_1", "m", "", 3, 4, "stop", [{"name": "f", "arguments": {"a": 1}}])
check("a call is its own item, with a call_id the client answers against",
      r["output"][0]["type"] == "function_call" and r["output"][0]["call_id"].startswith("call_"))
check("...and its arguments are a STRING, which is what the API specifies",
      isinstance(r["output"][0]["arguments"], str)
      and json.loads(r["output"][0]["arguments"]) == {"a": 1})
check("an empty text block is omitted rather than sent blank",
      [i["type"] for i in r["output"]] == ["function_call"])
# A truncated reply is "incomplete", not "completed". A client that retries on truncation cannot
# tell the difference otherwise, and will treat a cut-off answer as the finished one.
r = R.response("resp_1", "m", "hel", 3, 4, "length")
check("a reply cut at the token limit is reported incomplete, not completed",
      r["status"] == "incomplete" and r["incomplete_details"]["reason"] == "max_output_tokens")
check("usage adds up", R.response("r", "m", "x", 3, 4, "stop")["usage"]
      == {"input_tokens": 3, "output_tokens": 4, "total_tokens": 7})

print()
print("=" * 84)
print("5. THE STREAM MUST ANNOUNCE AN ITEM BEFORE ANY DELTA REFERS TO IT")
print("=" * 84)
# A client places text by the item and index it was told about. A delta for an item that was
# never announced has nowhere to go, and a client that is strict about it drops the whole reply.
created, text_delta, call_frames, finish = R.stream_frames("resp_1", "m")
frames, item = created()
kinds = [json.loads(f.decode().split("data: ", 1)[1])["type"] for f in frames]
check("the opening frames announce the response, then the item, then the content part",
      kinds == ["response.created", "response.in_progress",
                "response.output_item.added", "response.content_part.added"], str(kinds))
d = json.loads(text_delta(item["id"], "hi").decode().split("data: ", 1)[1])
check("a text delta names the item it belongs to", d["item_id"] == item["id"])
fr, done = call_frames({"name": "f", "arguments": {"a": 1}}, 1)
ks = [json.loads(f.decode().split("data: ", 1)[1])["type"] for f in fr]
check("a call is announced, filled and closed, in that order",
      ks == ["response.output_item.added", "response.function_call_arguments.delta",
             "response.function_call_arguments.done", "response.output_item.done"], str(ks))
check("...at its own output_index, not the text's",
      json.loads(fr[0].decode().split("data: ", 1)[1])["output_index"] == 1)
end = [json.loads(f.decode().split("data: ", 1)[1]) for f in finish(item, "hi", [done], 2, 3, False)]
check("the closing frames close the part, then the item, then the response",
      [e["type"] for e in end] == ["response.output_text.done", "response.content_part.done",
                                   "response.output_item.done", "response.completed"],
      str([e["type"] for e in end]))
check("...and the final response carries both the text and the call",
      [i["type"] for i in end[-1]["response"]["output"]] == ["message", "function_call"])
cut = [json.loads(f.decode().split("data: ", 1)[1])
       for f in finish(item, "hi", [], 2, 3, True)]
check("a truncated stream ends with response.incomplete, not response.completed",
      cut[-1]["type"] == "response.incomplete")
# Sequence numbers must increase. A client that reorders or deduplicates on them breaks if they
# repeat, and nothing about that failure would be visible here.
seqs = [e.get("sequence_number") for e in end if "sequence_number" in e]
check("sequence numbers strictly increase", seqs == sorted(set(seqs)) and len(seqs) == len(set(seqs)),
      str(seqs))

print()
print("=" * 84)
print("6. THE ENDPOINT IS REACHABLE AND THE LAUNCHER POINTS AT IT")
print("=" * 84)
srv = open(os.path.join(ROOT, "bigrig_engine", "server.py"), encoding="utf-8").read()
check("/v1/responses is routed", '"/v1/responses"' in srv and "_responses(body)" in srv)
check("...and has both a blocking and a streaming handler",
      "_responses_blocking" in srv and "_responses_stream" in srv)
check("a model with no tool-call format refuses tools here too, as on the other endpoints",
      srv.count("has no tool-call format") == 3, str(srv.count("has no tool-call format")))
lau = open(os.path.join(ROOT, "bigrig_engine", "launch.py"), encoding="utf-8").read()
check("the codex launcher tells it to use the responses wire API",
      "responses" in lau, "the launcher still assumes chat/completions")

print()
print("=" * 84)
print("7. WHAT THE REAL CLIENT DID WITH IT")
print("=" * 84)
# Not asserted here -- a test cannot install Codex -- but recorded, because it is the only
# evidence that matters and it took four attempts to get. From codex-cli 0.152.0's own debug log,
# pointed at this server:
#
#   POST url=http://127.0.0.1:8080/v1/responses status=200 OK
#     content-type: text/event-stream
#   codex.sse_event  event.kind=response.created
#   codex.sse_event  event.kind=response.in_progress
#   codex.sse_event  event.kind=response.output_item.added
#     Output item item_type="message" item_id="msg_82040c0f..."
#   codex.sse_event  event.kind=response.content_part.added
#
# It consumed the events BY NAME and parsed the output item. The wire protocol is right.
#
# Three things cost time and are worth writing down, because each looked like a server fault:
#   - `codex exec` blocks reading stdin. Without `< /dev/null` it never sends a request at all.
#   - killing it leaves sqlite WAL locks in CODEX_HOME that hang the next run.
#   - an agent's prompt is large. 12,704 prompt tokens took 137s on this model, and killing it
#     before that looks exactly like a hang.
check("the endpoint the real client posted to is the one that is routed",
      '"/v1/responses"' in srv)
check("...and it answers with an event stream, which is what it asked for",
      "text/event-stream" in srv[srv.index("_responses_stream"):])
check("the first four events it consumed are the four this server sends first",
      [json.loads(f.decode().split("data: ", 1)[1])["type"]
       for f in R.stream_frames("r", "m")[0]()[0]]
      == ["response.created", "response.in_progress",
          "response.output_item.added", "response.content_part.added"])

print()
print("=" * 84)
print("ALL TESTS PASSED" if not FAIL else f"{len(FAIL)} FAILURES: " + ", ".join(FAIL))
print("=" * 84)
sys.exit(1 if FAIL else 0)
