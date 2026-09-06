"""A real BigRig HTTP server around a fake model, for the tests.

The routes, headers, request parsing, queueing, cancellation and streaming framing are the
server's real code -- `make_handler` and `_State.pump` exactly as `rig serve` runs them. Only
the model is a stand-in: `FakeSession` answers with a fixed reply, word by word, and records
what it was asked. That is what lets a test say "POST this, expect that" instead of reading the
server's source text for the string it hopes is there, and it runs in a fraction of a second on
a machine with no model downloaded.

    with fake_server() as (url, state, session):
        st, body = post(url, "/v1/chat/completions", {...})

A test that needs a real model still starts one (tests/test_product.py does); this is for the
surface between the client and the engine.
"""
import contextlib
import json
import threading
import time
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

from bigrig_engine import server


class FakeTokenizer:
    """Whitespace tokens. No tool-call format, so tools read as unsupported -- the honest default."""
    eos_token_id = 0
    chat_template = None

    def encode(self, text, **kw):
        return list(range(1, len(str(text).split()) + 1))

    def decode(self, ids, **kw):
        return " ".join("w" for _ in ids)

    def apply_chat_template(self, messages, tokenize=False, **kw):
        return "\n".join(f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages) + "\nassistant:"


class FakeSession:
    """Everything the server touches on a Session, with a scripted reply.

    `reply` is streamed one word per chunk, `delay` seconds apart, so a test can watch a reply in
    flight (cancel it, count chunks). `calls` records every stream_text call's arguments.
    """

    def __init__(self, reply="Hello from the fake model, four words more.", delay=0.0,
                 name="fake-model", max_completion_tokens=4096, supports_tools=False, tower=None):
        self.reply, self.delay, self.name = reply, delay, name
        self.tower = tower                    # not None: the server believes it can read images
        self.streamed, self.packed = False, False
        self.model_dir = "/nonexistent/fake-model"
        self.budget_gb, self.footprint_gb = 9.0, 1.0
        self.max_completion_tokens = max_completion_tokens
        self.token_limit_reason = "the model's context window"
        self.context_length = max_completion_tokens
        self.capacity, self.top_k = 0, 0
        self.plan = {"n_experts": 1, "capacity": 0, "n_layers": 1, "bytes_per_expert": 0}
        self.handle = None
        self.init_kwargs = {}
        self.strategy = {"mode": "resident"}
        self.tokenizer = FakeTokenizer()
        self._supports_tools = supports_tools
        self.calls = []
        self.closed = False
        self.flushes = 0
        self.total_tokens = 0
        self._used = 0

    # ---- what /health and /stats read
    def stats(self):
        return {"model": self.name, "streamed": self.streamed, "mode": self.strategy["mode"],
                "serving": "fake, held resident", "supports_tools": self._supports_tools,
                "weights_altered": False, "decision": "n/a", "tuned": False,
                "capacity_source": "an estimate", "load_seconds": 0.01,
                "total_tokens": self.total_tokens, "flagged_tokens": 0, "flagged_share": 0.0,
                "flag_runs": 0, "monitor": False, "chat_template": False,
                "can_toggle_thinking": False, "context_length": self.context_length,
                "max_completion_tokens": self.max_completion_tokens,
                "token_limit_reason": self.token_limit_reason,
                "context_used": self._used,
                "context_remaining": max(0, self.max_completion_tokens - self._used),
                "budget_gb": self.budget_gb, "packed": False, "file_pool": False,
                "prompt_cache_gb": 0.5, "prompt_cache_bytes": 0, "prompt_cache_hits": 0,
                "prompt_cache_misses": 0, "persist": False, "history_snapshots": 0,
                "resumed_conversations": 0, "footprint_gb": self.footprint_gb}

    def supports_tools(self):
        return self._supports_tools

    def tool_splitter(self, tools=None):
        return None

    def extract_tool_calls(self, text, tools=None):
        return text, []

    # ---- generation
    def stream_text(self, messages=None, prompt="", max_tokens=512, **kw):
        self.calls.append({"messages": messages, "prompt": prompt, "max_tokens": max_tokens, **kw})
        words = self.reply.split(" ")
        n_prompt = len(self.tokenizer.encode(prompt or json.dumps(messages or [])))
        limit = min(len(words), int(max_tokens))
        # Prefill progress, the way the engine reports it: through the callback, before any text.
        cb = kw.get("on_prefill")
        if cb is not None and n_prompt:
            for done in sorted({0, n_prompt // 2, n_prompt}):
                cb(done, n_prompt)
            if self.delay:
                time.sleep(self.delay)
        for i in range(limit):
            if self.delay:
                time.sleep(self.delay)
            chunk = words[i] if i == 0 else " " + words[i]
            last = i == limit - 1
            self.total_tokens += 1
            self._used = n_prompt + i + 1
            yield chunk, {"token": i + 1, "finish_reason": ("stop" if len(words) <= max_tokens else "length")
                          if last else None, "tok_s": 50.0, "from_draft": False, "reasoning_delta": "",
                          "prompt_tokens": n_prompt, "generation_tokens": i + 1, "degraded": False}

    def stream_batch(self, *a, **kw):
        raise NotImplementedError("the fake serves one request at a time")

    # ---- housekeeping the server calls
    def trim_prompt_cache(self, to_bytes=0):
        return 0

    def flush_sessions(self):
        self.flushes += 1
        return {"written": 0, "removed": 0, "bytes": 0, "on_disk": 0}

    def close(self):
        self.closed = True

    def _settle(self):
        pass

    def _token_ceiling(self):
        return self.max_completion_tokens, self.token_limit_reason


@contextlib.contextmanager
def fake_server(session=None, **state_attrs):
    """A live server on a free loopback port. Yields (base_url, state, session); stops on exit."""
    session = session or FakeSession()
    state = server._State(session)
    for k, v in state_attrs.items():
        setattr(state, k, v)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.make_handler(state))
    httpd.daemon_threads = True
    port = httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, name="fake-http", daemon=True).start()
    pump = threading.Thread(target=state.pump, kwargs={"poll": 0.02}, name="fake-pump", daemon=True)
    pump.start()
    try:
        yield f"http://127.0.0.1:{port}", state, session
    finally:
        state.stopping = True
        httpd.shutdown()
        httpd.server_close()
        pump.join(timeout=2)


def request(url, path, body=None, method=None, headers=None, timeout=10, raw=False):
    """(status, parsed-json-or-bytes, headers). Never raises on an HTTP error status."""
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url + path, data=data, method=method or ("POST" if data else "GET"))
    req.add_header("Content-Type", "application/json")
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            payload = r.read()
            hdrs = dict(r.headers)
            status = r.status
    except urllib.error.HTTPError as e:
        payload, hdrs, status = e.read(), dict(e.headers), e.code
    if raw:
        return status, payload, hdrs
    try:
        return status, json.loads(payload.decode() or "null"), hdrs
    except ValueError:
        return status, payload, hdrs


def get(url, path, **kw):
    return request(url, path, **kw)


def post(url, path, body, **kw):
    return request(url, path, body=body, **kw)


def sse_events(payload: bytes):
    """The JSON objects of an SSE body, in order; the literal [DONE] as the string 'DONE'."""
    out = []
    for line in payload.decode().splitlines():
        if not line.startswith("data:"):
            continue
        d = line[5:].strip()
        if d == "[DONE]":
            out.append("DONE")
        else:
            try:
                out.append(json.loads(d))
            except ValueError:
                out.append(d)
    return out
