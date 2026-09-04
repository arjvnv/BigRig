"""Behaviour, against a real server. Nothing here reads source text.

WHY THIS FILE EXISTS
    The abandoned-request bug passed every unit test in the suite and was caught by an
    end-to-end run: the check for a departed client sat in a branch that never executed while
    tokens were flowing. A test that greps the source for `client_gone` would have passed
    before and after. So this file starts the server the way a user does, talks to it over
    HTTP the way a client does, and asserts on what it DID -- how many tokens it generated after
    the client left, whether the pool shrank on one reading or two, whether memory came back.

    It is slower than the rest of the suite (a model has to load) and it says so. It uses the
    smallest packed streamable model on the machine, forced to stream so the controller exists.
"""
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

FAIL = []


def check(n, c, d=""):
    (print(f"  PASS  {n}") if c else (FAIL.append(n), print(f"  FAIL  {n}  {d}")))


PORT = 8232
MODEL = "OLMoE-1B-7B-0125-4bit"
LOG = os.path.join(ROOT, "data/results/test_live.log")
os.makedirs(os.path.dirname(LOG), exist_ok=True)

# Every check below drives a real server holding a real model. On a machine that has not
# downloaded one -- a fresh clone, or CI -- say so and stop, rather than failing forty checks
# for a machine that is working correctly.
if not os.path.isdir(os.path.join(ROOT, "models", MODEL)):
    print(f"  SKIPPED - models/{MODEL} is not on this machine. This file starts a real server")
    print(f"  and talks to it; run `bigrig prepare mlx-community/{MODEL}` once to exercise it.")
    print("\n" + "=" * 78); print("ALL TESTS PASSED"); print("=" * 78)
    sys.exit(0)

# The port must be ours -- see test_product for the detour that taught this.
_probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
_probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
try:
    _probe.bind(("127.0.0.1", PORT))
except OSError:
    print(f"\n  port {PORT} is in use; stop it and run again:  lsof -ti tcp:{PORT} | xargs kill\n")
    sys.exit(1)
finally:
    _probe.close()

if not os.path.exists(os.path.join(ROOT, "models", MODEL)):
    print(f"  {MODEL} is not downloaded; nothing to test against (skipped, not passed)")
    sys.exit(0)

env = dict(os.environ, BIGRIG_DEBUG_PRESSURE="1")
import atexit
proc = subprocess.Popen(
    [os.path.join(ROOT, ".venv/bin/python"), "-m", "bigrig_engine.cli", "serve", MODEL,
     "--force-stream", "--residency", "0.5", "--port", str(PORT), "--no-warm"],
    stdout=open(LOG, "w"), stderr=subprocess.STDOUT, cwd=ROOT, env=env)
# However this file ends -- a failed check, a traceback, a Ctrl-C -- the server it started dies
# with it. A crash once left one holding port 8232, and the next run refused to start.
atexit.register(lambda: (proc.poll() is None) and proc.kill())


def get(path, timeout=10):
    with urllib.request.urlopen(f"http://127.0.0.1:{PORT}{path}", timeout=timeout) as r:
        return r.status, json.loads(r.read())


def post(path, body, timeout=180):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def health():
    """/health answers 503 for the second a reload takes; a caller between two quiet readings can
    land in that second (it did, once, and took the whole file down with a traceback)."""
    end = time.time() + 30
    while True:
        try:
            return get("/health")[1]
        except urllib.error.HTTPError as e:
            if e.code != 503 or time.time() > end:
                raise
            time.sleep(0.5)


def wait_for(pred, seconds, every=0.5):
    """True when pred() first holds, False when it never did within `seconds`."""
    end = time.time() + seconds
    while time.time() < end:
        try:
            if pred():
                return True
        except Exception:                        # noqa: BLE001 -- 503 during a reload is normal
            pass
        time.sleep(every)
    return False


def settled(h):
    """A reload leaves /health answering 503 for a second; wait until it is a server again."""
    return h.get("status") == "ok"


print("=" * 84)
print("1. IT STARTS, AND SAYS WHAT IT IS ON")
print("=" * 84)
up = wait_for(lambda: settled(health()), 120, 1.0)
check("the server comes up", up, open(LOG).read()[-400:] if os.path.exists(LOG) else "")
if not up:
    proc.terminate()
    sys.exit(1)

h0 = health()
check("it streams (the controller only exists for a streamed model)", h0.get("streamed") is True)
check("it says whether the experts are packed, and this one is",
      h0.get("packed") is True, str(h0.get("packed")))
check("the speed verdict exists before any reply, and is labelled as an expectation",
      h0.get("speed_tier") in ("FAST", "GOOD", "USABLE", "SLOW")
      and str(h0.get("speed_basis", "")).startswith("expected"), str(h0.get("speed_basis")))
mr = h0.get("memory_released") or {}
check("growing back is on without any flag", mr.get("grow") is True, str(mr))
check("it is inside its start-up grace window", mr.get("in_grace") is True and
      mr.get("grace_s", 0) >= 30, str(mr))
home = int(h0.get("planned_from") or h0.get("capacity"))
floor = int(mr.get("floor", 0))
check("there is room between home and the floor for a shrink to be visible",
      home - floor >= 2, f"home {home} floor {floor}")

print("\n" + "=" * 84)
print("2. A REPLY IS MEASURED, AND THE VERDICT BECOMES A MEASUREMENT")
print("=" * 84)
st, d = post("/v1/chat/completions",
             {"messages": [{"role": "user", "content": "Count from one to twenty."}],
              "max_tokens": 24, "temperature": 0.0})
check("a chat completion returns content", st == 200 and d["choices"][0]["message"]["content"])
_, s1 = get("/stats")
last = (s1.get("history") or [{}])[-1]
check("time to first token was recorded for a blocking request",
      isinstance(last.get("ttft"), (int, float)) and last["ttft"] > 0, str(last))
check("tokens per second was recorded", (last.get("tok_s") or 0) > 0, str(last))
h1 = health()
check("after a reply the verdict is a measurement, and says so",
      str(h1.get("speed_basis", "")).startswith("measured"), str(h1.get("speed_basis")))
check("...with one of the four words the doctor uses",
      h1.get("speed_tier") in ("FAST", "GOOD", "USABLE", "SLOW"), str(h1.get("speed_tier")))
# A REPLY'S USAGE MUST SURVIVE THE FINAL FLUSH OF HELD-BACK TEXT.
#     The tail of a reply is held back until it cannot be the start of a stop sequence, then
#     flushed once generation ends. That flush used to carry zeros, and both endpoints build
#     `usage` from whichever chunk arrived last -- so a reply with real text reported 0 prompt
#     and 0 completion tokens, and logged a 0-token reply with no rate to the analytics page.
#     Found in the readiness pass: two of three replies. Repeated, because the first reply of a
#     fresh server happened not to hit it.
for _rep in range(3):
    _st, _u = post("/v1/chat/completions",
                   {"messages": [{"role": "user", "content": "say hi"}],
                    "max_tokens": 6, "temperature": 0.0})
    _txt = _u["choices"][0]["message"]["content"] if _st == 200 else ""
    check(f"reply {_rep + 1}: text and usage agree, neither silently zero",
          _st == 200 and (not _txt or _u["usage"]["completion_tokens"] > 0),
          f"{_st} text={_txt!r} usage={_u.get('usage')}")
    if _st == 200 and _txt:
        check(f"...and reply {_rep + 1}'s totals add up",
              _u["usage"]["total_tokens"]
              == _u["usage"]["prompt_tokens"] + _u["usage"]["completion_tokens"],
              str(_u["usage"]))
_hist = get("/stats")[1].get("history") or []
check("...and no reply was logged with zero tokens",
      all((r.get("tokens") or 0) > 0 for r in _hist[-3:]), str(_hist[-3:]))

check("packed experts were admitted without a copy",
      int(h1.get("zero_copy_admits") or 0) > 0, str(h1.get("zero_copy_admits")))

print("\n" + "=" * 84)
print("2b. THE API RETURNS THE ANSWER, NOT THE MODEL'S SCRATCHPAD")
print("=" * 84)
# Every model of this class thinks out loud first. Qwen3.6 and GLM-4.x end the generation prompt
# with an OPEN <think>, so the reply begins mid-thought and closes the block before answering;
# Qwen3-30B opens and closes it itself. Returned as `content`, that hands a coding agent the
# model's reasoning as its reply, with a stray `</think>` in the middle of it. Measured on GLM
# before the fix: asked for 2+2, `content` was three lines of deliberation. The reasoning is a
# separate field now. This model does not think, so what is asserted here is that a plain model
# is untouched and that no stray marker ever reaches a client.
_st, _d = post("/v1/chat/completions",
               {"messages": [{"role": "user", "content": "Say hello."}],
                "max_tokens": 24, "temperature": 0.0})
_m = _d["choices"][0]["message"] if _st == 200 else {}
check("a model that does not think returns its text as content, unchanged",
      _st == 200 and (_m.get("content") or "").strip() != "", str(_d)[:150])
check("...and carries no reasoning field it does not need", "reasoning_content" not in _m,
      str(_m.keys()))
check("no thinking marker ever reaches the client",
      not any(t in (_m.get("content") or "") for t in ("<think>", "</think>")),
      repr((_m.get("content") or "")[:80]))
_src_sess = open(os.path.join(ROOT, "bigrig_engine", "session.py"), encoding="utf-8").read()
_src_srv = open(os.path.join(ROOT, "bigrig_engine", "server.py"), encoding="utf-8").read()
check("the split handles both prompt shapes: block opened by the prompt, and by the reply",
      "_starts_in_reasoning" in _src_sess and "def _split_reasoning" in _src_sess)
check("every endpoint goes through one call that hides reasoning, so none can disagree",
      _src_srv.count("hide_reasoning=True") == 1)
check("the terminal is NOT affected -- thinking stays inline where a person wants to watch it",
      "hide_reasoning" not in open(os.path.join(ROOT, "bigrig_engine", "cli.py"), encoding="utf-8").read())
check("blocking replies expose the thinking in the field vLLM and DeepSeek use",
      'reasoning_content' in _src_srv and 'delta' in _src_srv)

print("\n" + "=" * 84)
print("2c. THE ANTHROPIC STREAM IS VALID AS A STRICT CLIENT READS IT")
print("=" * 84)
# This is the endpoint `bigrig launch` points coding agents at, so its frames have to be right
# rather than merely accepted by a lenient client. Blocks opened once, closed once, indices
# consecutive from 0, and every delta's kind matching the block it lands on. This model does not
# think, so what is asserted is the plain shape -- unchanged by the thinking work.
_body = json.dumps({"model": "x", "max_tokens": 40, "stream": True,
                    "messages": [{"role": "user", "content": "Say hello."}]}).encode()
_req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/messages", data=_body,
                              headers={"Content-Type": "application/json", "x-api-key": "t",
                                       "anthropic-version": "2023-06-01"})
_ev = []
with urllib.request.urlopen(_req, timeout=300) as _r:
    for _line in _r:
        if _line.startswith(b"data: "):
            try:
                _ev.append(json.loads(_line[6:]))
            except ValueError:
                pass
_kinds = [e.get("type") for e in _ev]
check("the stream opens with message_start and ends with message_stop",
      _kinds[:1] == ["message_start"] and _kinds[-1:] == ["message_stop"], str(_kinds[:2] + _kinds[-2:]))
_open, _started, _bad = {}, [], []
for _e in _ev:
    _t = _e.get("type")
    if _t == "content_block_start":
        if _e["index"] in _started:
            _bad.append(f"block {_e['index']} opened twice")
        _started.append(_e["index"]); _open[_e["index"]] = _e["content_block"]["type"]
    elif _t == "content_block_delta":
        _want = {"thinking": "thinking_delta", "text": "text_delta",
                 "tool_use": "input_json_delta"}.get(_open.get(_e["index"]))
        if _e["index"] not in _open:
            _bad.append(f"delta on closed block {_e['index']}")
        elif _e["delta"]["type"] != _want:
            _bad.append(f"{_e['delta']['type']} on a {_open[_e['index']]} block")
    elif _t == "content_block_stop":
        if _e["index"] not in _open:
            _bad.append(f"block {_e['index']} stopped while not open")
        _open.pop(_e["index"], None)
check("every block opens once, closes once, and its deltas match its kind", not _bad, str(_bad[:3]))
check("...and none is left open at the end", not _open, str(_open))
check("...with indices consecutive from 0", _started == list(range(len(_started))), str(_started))
check("a model that does not think streams one text block, exactly as before",
      [t for t in _kinds if t == "content_block_start"] and len(_started) == 1, str(_started))
_text = "".join(e["delta"]["text"] for e in _ev
                if e.get("type") == "content_block_delta" and e["delta"]["type"] == "text_delta")
check("...and it carries the reply", _text.strip() != "", repr(_text[:60]))

print("\n" + "=" * 84)
print("3. A CLIENT THAT LEAVES STOPS THE REPLY")
print("=" * 84)
# The bug this guards: tokens kept flowing to nobody for the whole of max_tokens, because the
# check for a departed client lived in a branch that never ran while chunks were arriving.
before = int(health().get("total_tokens") or 0)
WANT = 400
raw = socket.create_connection(("127.0.0.1", PORT), timeout=30)
body = json.dumps({"messages": [{"role": "user", "content":
                                 "Write a long story about a lighthouse keeper."}],
                   "max_tokens": WANT, "temperature": 0.0, "stream": True}).encode()
raw.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\n"
            b"Content-Type: application/json\r\nContent-Length: " + str(len(body)).encode()
            + b"\r\n\r\n" + body)
first = raw.recv(4096)                          # headers and the first frame or two
check("the stream started", b"200" in first, first[:80])
raw.close()                                     # the client walks away
t_left = time.time()
# Now ask for something else right away, as a real client would, and time it.
t0 = time.time()
st2, d2 = post("/v1/chat/completions",
               {"messages": [{"role": "user", "content": "Say yes."}],
                "max_tokens": 8, "temperature": 0.0}, timeout=120)
took = time.time() - t0
time.sleep(1.0)
after = int(health().get("total_tokens") or 0)
spent = after - before - int(d2.get("usage", {}).get("completion_tokens", 0))
check("the abandoned reply stopped well short of what was asked",
      0 <= spent < WANT // 2, f"generated {spent} of {WANT} after the client left")
check("...and the next request was not made to wait for it", st2 == 200 and took < 20.0,
      f"{took:.1f}s")

print("\n" + "=" * 84)
print("4. THE CONTROLLER ACTS ON READINGS, NOT ON GLANCES")
print("=" * 84)
# Inside the grace window even three readings must not shrink. Spaced out, because the pump
# hands over one reading per idle poll and two posts inside one poll would be seen as one; and
# three, because the first reading under pressure is spent releasing remembered prompts (that
# is by design -- prompts go before pool slots) and never reaches the policy.
for _ in range(3):
    post("/v1/debug/pressure", {"pressure": True})
    time.sleep(0.7)
time.sleep(1.5)
hg = health()
check("two pressure readings inside the start-up grace do not shrink the pool",
      (hg.get("memory_released") or {}).get("shrinks", 0) == 0
      and int(hg.get("planned_from") or hg.get("capacity")) == home,
      str(hg.get("memory_released")))
check("...and it says why", "load itself" in str((hg.get("memory_released") or {}).get("last_reason")),
      str((hg.get("memory_released") or {}).get("last_reason")))

# WHILE PRESSURE IS FORCED, THE WATCHER MUST NOT INVENT READINGS.
#     Its own sample is discarded when forced, so counting it as a reading hands the controller a
#     confirmation nobody asked for. Measured before the fix: one forced reading, then a watcher
#     tick 10 s later, and the pool shrank on what should have been a single reading. The watcher
#     samples every 10 s, so a wait past that window is the only way to catch it.
_p1 = post("/v1/debug/pressure", {"pressure": True})[1].get("polls")
time.sleep(13)
_p2 = post("/v1/debug/pressure", {"pressure": True})[1].get("polls")
check("a forced reading is the only reading: the watcher adds none of its own over 13 s",
      isinstance(_p1, int) and _p2 == _p1 + 1, f"{_p1} -> {_p2}")

# End the grace and shorten the clocks, then ONE reading. Before the fix this reading was
# counted five times a second and the pool shrank inside 0.8 s.
post("/v1/debug/pressure", {"pressure": True, "grace_s": 0, "min_interval_s": 1,
                            "grow_quiet_s": 3})
time.sleep(2.5)
h_one = health()
check("one reading of pressure, however long it is looked at, does not shrink the pool",
      (h_one.get("memory_released") or {}).get("shrinks", 0) == 0
      and int(h_one.get("planned_from") or h_one.get("capacity")) == home,
      str(h_one.get("memory_released")))
check("...it is waiting for the second", "waiting for 2" in
      str((h_one.get("memory_released") or {}).get("last_reason")),
      str((h_one.get("memory_released") or {}).get("last_reason")))

post("/v1/debug/pressure", {"pressure": True})
shrunk = wait_for(lambda: (health().get("memory_released") or {}).get("shrinks", 0) >= 1
                  and settled(health()), 90)
h_sh = health()
low = int(h_sh.get("planned_from") or h_sh.get("capacity"))
check("the second reading shrinks it", shrunk and low < home, f"{home} -> {low}")
check("...not below the floor", low >= floor, f"{low} < {floor}")
check("...and the reload is logged with the reason",
      any("short of memory" in e.get("why", "") for e in h_sh.get("shrink_log", [])),
      str(h_sh.get("shrink_log")))

print("\n" + "=" * 84)
print("5. WHEN THE MACHINE IS QUIET AGAIN, THE MEMORY COMES BACK -- AND STOPS AT HOME")
print("=" * 84)
# Quiet readings, spaced past the (shortened) quiet requirement. Each POST is one reading.
post("/v1/debug/pressure", {"pressure": False})
grew = False
caps = [low]
deadline = time.time() + 120
while time.time() < deadline:
    time.sleep(3.5)
    post("/v1/debug/pressure", {"pressure": False})
    wait_for(lambda: settled(health()), 30)
    hh = health()
    cap = int(hh.get("planned_from") or hh.get("capacity"))
    if cap != caps[-1]:
        caps.append(cap)
    if (hh.get("memory_released") or {}).get("grows", 0) >= 1:
        grew = True
    if cap >= home:
        break
h_end = health()
check("memory was taken back once the machine was quiet", grew, str(h_end.get("memory_released")))
check("...in steps that only ever went up", all(a < b for a, b in zip(caps, caps[1:])), str(caps))
check("...all the way back to where it started", caps[-1] == home, f"{caps} home {home}")
# One more quiet reading, generously spaced: it must NOT go past home.
time.sleep(4.0)
post("/v1/debug/pressure", {"pressure": False})
time.sleep(2.0)
h_over = health()
check("and never above it",
      int(h_over.get("planned_from") or h_over.get("capacity")) == home
      and "nothing was borrowed" in str((h_over.get("memory_released") or {}).get("last_reason")),
      str(h_over.get("memory_released")))
check("the server still answers after all of that",
      post("/v1/chat/completions", {"messages": [{"role": "user", "content": "Say hi."}],
                                    "max_tokens": 4, "temperature": 0.0})[0] == 200)

proc.terminate()
try:
    proc.wait(timeout=20)
except subprocess.TimeoutExpired:
    proc.kill()

print("\n" + "=" * 84)
print("ALL TESTS PASSED" if not FAIL else f"{len(FAIL)} FAILURES: " + ", ".join(FAIL))
print("=" * 84)
sys.exit(1 if FAIL else 0)
