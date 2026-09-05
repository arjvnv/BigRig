"""An OpenAI-compatible HTTP server, on the standard library alone.

WHY NO FRAMEWORK
    Adding FastAPI and uvicorn means a user who wants to try this installs a web stack before
    they can ask a model a question. `http.server` ships with Python and is entirely adequate for
    a single-machine inference endpoint, which is not serving thousands of connections.

WHAT IT SPEAKS
    POST /v1/chat/completions       OpenAI, streaming (SSE) and blocking
    POST /v1/completions            the legacy OpenAI shape, same engine
    POST /v1/messages               Anthropic Messages -- this is what Claude Code speaks
    POST /v1/messages/count_tokens  Anthropic token counting
    GET  /v1/models
    GET  /stats                     a rolling record of every reply served, for the
                                    analytics view -- measurements, not a live snapshot
    GET  /                          a self-contained web interface, for people who do not
                                    want a terminal -- and the only place the quality meter
                                    is actually VISIBLE rather than a number in a JSON field
    GET  /health                live residency, miss rate and quality -- the endpoint this
                                product actually differentiates on

GENERATION RUNS ON THE MAIN THREAD, AND IT HAS TO
    mlx_lm builds its generation stream with `mx.new_thread_local_stream(...)` at import time.
    Thread-local means exactly that: on any other thread the stream does not exist, and MLX
    aborts the whole process with

        libc++abi: terminating due to uncaught exception of type std::runtime_error:
        There is no Stream(gpu, 1) in current thread.

    Not an exception a handler can catch -- the server dies. So HTTP is served on background
    threads and every generation is handed to the main thread through a queue.

    That also gives the serialisation the engine needs anyway. One model, one expert pool: two
    requests decoding at once would evict each other's experts every step, and both would finish
    later than if they had simply queued. `/health` reports the depth so the wait is visible
    rather than mysterious.
"""
from __future__ import annotations

import json
import os
import queue
import select
import socket
import threading
import time
import uuid
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import anthropic as anth
from . import responses as resp

MAX_BODY = 4 << 20          # a prompt larger than this is a mistake, not a request
HISTORY = 400               # replies kept for the analytics view

# How many engine events the console keeps. Enough that a user who looks away for a coffee break
# still sees what happened while they were gone, small enough to be irrelevant to memory.
EVENTS = 200

# How many tokens a request guesses at once when it asks for lookahead and does not say how many.
# Swept on this model against text being reproduced almost verbatim, where drafting is worth
# something at all: k=2 1.12x, k=4 1.19x, k=8 1.47x, k=12 1.59x. It keeps climbing because the
# cost of a verifying pass grows more slowly than the tokens it can return -- but so does the
# waste when a guess is wrong, and 8 is where the two are in reasonable balance.
LOOKAHEAD_TOKENS = 8
PUT_TIMEOUT_S = 30.0        # longest the generator will wait on a stalled client before dropping
GET_TIMEOUT_S = 900.0       # longest a handler waits on the generator before giving up

# HOW OFTEN A WAITING HANDLER LOOKS TO SEE WHETHER ITS CLIENT IS STILL THERE.
#     Only a poll of an already-open socket -- no allocation, no syscall storm -- so this
#     is cheap enough to do often and slow enough not to matter. It bounds how long a
#     blocking request keeps generating after its client has gone, and nothing else.
CLIENT_POLL_S = 0.5


def client_gone(sock) -> bool:
    """Has the client hung up? Asked of the socket, never guessed.

    A FALSE POSITIVE IS THE ONLY DANGEROUS ANSWER -- it would abort a request whose client is
    sitting there waiting -- so the check is built so it cannot give one. `select` with a zero
    timeout says whether the socket has anything to report; only then does it peek. A peek that
    returns BYTES is a client that sent something: a pipelined request on a kept-alive connection
    is the real case, and that is a client very much still there, so it reads False. Only an
    EMPTY peek is EOF, which on a socket means the peer closed its end and cannot mean anything
    else.

    MSG_PEEK, so nothing is consumed. Reading the byte would corrupt the next request on a
    kept-alive connection, which is a worse bug than the one this fixes.

    Module level rather than a method, because the interesting cases are socket states and this
    way they can be tested against real sockets without standing up an HTTP server.
    """
    if sock is None:
        return False                     # nothing to ask; assume present, never abort
    try:
        if sock.fileno() < 0:
            return True
        readable, _, _ = select.select([sock], [], [], 0)
        if not readable:
            return False                 # quiet socket: connected, nothing pending
        return sock.recv(1, socket.MSG_PEEK) == b""
    except (BlockingIOError, InterruptedError):
        return False                     # would have blocked; no evidence either way
    except (OSError, ValueError):
        return True                      # closed or otherwise unusable


class _Abandoned(Exception):
    """Raised from the prefill callback when the client has gone. Not an error.

    The alternative is finishing a prompt nobody will read, which on this engine is minutes of
    the one thread allowed to touch MLX.
    """


class _Job:
    """One generation request, plus the flag that says the client stopped caring.

    THE BUG THIS FLAG EXISTS FOR
        Without it, a client that hangs up mid-stream killed the server permanently. The HTTP
        thread stopped draining `out`, the main thread kept generating, and after 256 queued
        chunks `out.put()` blocked forever -- so the ONE thread allowed to touch MLX was stuck
        and no request ever completed again. `/health` kept answering, which made it look alive.
        Reproduced: curl with `-m 1` against a 400-token request, and the next request hung.
    """
    __slots__ = ("kw", "out", "cancelled")

    def __init__(self, kw):
        self.kw = kw
        self.out = queue.Queue(maxsize=256)
        self.cancelled = threading.Event()


class _State:
    def __init__(self, session):
        self.session = session
        self.count_lock = threading.Lock()
        self.jobs: queue.Queue = queue.Queue()
        # A rolling record of what actually happened, so the analytics view plots measurements
        # rather than a single live snapshot. Bounded: a server left running for a week must not
        # accumulate an unbounded list in the one process that also holds the model.
        self.history: deque = deque(maxlen=HISTORY)
        self.waiting = 0
        # 1 means never batch, which is the default: batching changes the arithmetic and so the
        # tokens. See Session.stream_batch.
        self.batch_size = 1
        # In-flight jobs by the id their client gave them. A stop button cannot rely on the
        # socket erroring: the OS buffers writes, so a client that has gone away may not be
        # noticed for several more tokens, and at single-digit tokens per second that is
        # seconds of a model still running after the user asked it to stop.
        self.by_id: dict = {}
        # A settings change the user has asked for, applied by the pump between requests. It
        # cannot be applied where it is requested: rebuilding the model touches MLX, and only
        # the pump's thread is allowed to.
        self.pending_config = None
        self.config_result = None
        self.config_event = threading.Event()
        self.reloads = 0
        self.served = 0
        # Totals that outlive a reload. The session's own counters start at zero when the pool is
        # rebuilt, and the page showed "0 tokens generated" beside a history of forty replies.
        self.tokens_before = 0
        self.flagged_before = 0
        self.started = time.time()
        self.stopping = False
        # Who may talk to this server, filled in by `serve()`. Defaults are the safe ones so a
        # _State built directly by a test is not accidentally wide open.
        self.hot_warm: dict = {}
        self.bind_host = "127.0.0.1"
        self.loopback = True
        self.origins: frozenset = frozenset()
        # GIVING MEMORY BACK. The pool is ordinary anonymous memory -- measured, not assumed:
        # holding 4.64 GB moved `anonymous` by 5.00 GB and `wired` by 0.06 -- so macOS may
        # reclaim it under pressure, turning cache hits into the disk reads streaming exists to
        # avoid, with nothing anywhere reporting a problem. `memctl` decides when to hand some
        # back; None until the caller enables it, because a controller nobody asked for should
        # not act on their hardware.
        self.memctl = None
        self.pressure = False
        self.pressure_at = 0.0
        self.pressure_polls = 0
        self.pressure_fed = -1             # the last reading handed to the controller
        self.pressure_forced = False
        self.shrink_log: deque = deque(maxlen=32)
        # WHAT THE ENGINE DID, IN ORDER, SO A USER CAN SEE IT RATHER THAN INFER IT.
        #     This engine does things to itself while it runs -- it hands memory back under
        #     pressure, drops remembered prompts, rebuilds the pool when a setting changes, warms
        #     the page cache in the background. Every one of those changes how fast the next
        #     reply arrives, and until now every one of them was invisible: the numbers on the
        #     page simply moved, with nothing saying why.
        #
        #     Bounded, because a server left running for a week must not grow a list inside the
        #     one process that also holds the model. `seq` rather than a timestamp is what the
        #     page polls on -- clocks can go backwards, a counter cannot, and it makes "what is
        #     new since I last asked" exact instead of approximate.
        self.events: deque = deque(maxlen=EVENTS)
        self.event_seq = 0
        self.warm = None
        # Physical memory, read once. The speed verdict's ceiling hint needs it and `sysctl`
        # is not something to run on every /health poll.
        try:
            from .calibrate import total_gb as _tg
            self.total_gb = float(_tg())
        except Exception:                        # noqa: BLE001 -- a hint, never a failure
            self.total_gb = 0.0
        # The measured disk, if this Mac has been calibrated, for the pre-measurement verdict.
        self.disk_gbs = None
        try:
            import json as _json
            from .stream import home as _home
            _pp = os.path.join(_home(), "data", "results", "host_profile.json")
            if os.path.exists(_pp):
                self.disk_gbs = float(_json.load(open(_pp)).get("disk_gbs") or 0) or None
        except Exception:                        # noqa: BLE001
            self.disk_gbs = None

    def event(self, kind: str, text: str, **extra) -> None:
        """Record one thing that happened. Safe to call from any thread.

        Never raises. This is called from the pump, from the pressure watcher and from request
        handlers, and a logging call that can fail is a logging call that takes down whichever of
        those it was on.
        """
        try:
            with self.count_lock:
                self.event_seq += 1
                self.events.append({"seq": self.event_seq, "at": round(time.time() - self.started, 1),
                                    "kind": str(kind)[:24], "text": str(text)[:300], **extra})
        except Exception:                    # noqa: BLE001
            pass

    def record(self, rec: dict) -> None:
        with self.count_lock:
            self.history.append(rec)

    def submit(self, kw, rid: str = "") -> _Job:
        j = _Job(kw)
        with self.count_lock:
            self.waiting += 1
            if rid:
                self.by_id[rid] = j
        self.jobs.put(j)
        return j

    def cancel(self, rid: str) -> bool:
        """Stop a named job. True if one was running under that id."""
        with self.count_lock:
            j = self.by_id.get(rid)
        if j is None:
            return False
        j.cancelled.set()
        return True

    def forget(self, rid: str) -> None:
        if not rid:
            return
        with self.count_lock:
            self.by_id.pop(rid, None)

    def _apply_config(self) -> None:
        """Rebuild the session with new settings. Runs on the pump's thread, between requests.

        The old session is dropped BEFORE the new one is built. Holding both would need twice the
        pool at once, which on a machine chosen for not having enough memory is the one thing
        that must never happen.
        """
        want = self.pending_config
        self.pending_config = None
        try:
            from .session import Session
            old = self.session
            kw = dict(old.init_kwargs)
            kw.update({k: v for k, v in want.items() if v is not None})
            self.tokens_before += int(getattr(old, "total_tokens", 0) or 0)
            self.flagged_before += int(getattr(old, "flagged_tokens", 0) or 0)
            self.session = None
            # Explicit, not implicit. Dropping the reference is not enough -- see
            # StreamHandle.close for what happened when this relied on refcounting alone.
            import mlx.core as _mx
            # Explicit, not implicit. Dropping the reference is not enough -- see
            # StreamHandle.close for what happened when this relied on refcounting alone.
            old.close()
            del old
            _mx.clear_cache()
            self.session = Session(**kw)
            # THE FOOTPRINT MUST BE RE-MEASURED AFTER THE OLD POOL IS ACTUALLY GONE.
            #     `Session.__init__` reads `mx.get_active_memory()` the moment it finishes, and
            #     at that moment MLX is still holding the buffers the dropped session freed. On a
            #     40 -> 53 reload that read 10.56 GB for 5.25 GB of weights, and the number feeds
            #     the reply ceiling -- so a reload silently shortened every reply that followed.
            # AND AFTER THE NEW POOL HAS SETTLED, WHICH IS A SECOND, DIFFERENT REASON.
            #     `clear_cache` returns what the OLD session freed. It does not return what the
            #     NEW one abandoned filling its pool -- warming rebinds each slot tensor and the
            #     superseded ones stay counted until real multi-token work forces their reuse.
            #     That was 2.64 GB on this model, and re-measuring here without settling first
            #     would put the over-estimate straight back after every reload, which is exactly
            #     the 256-token ceiling this pair of lines exists to prevent.
            self.session._settle()
            _mx.clear_cache()
            self.session.footprint_gb = _mx.get_active_memory() / 1e9
            (self.session.max_completion_tokens,
             self.session.token_limit_reason) = self.session._token_ceiling()
            self.reloads += 1
            self.event("reload", f"rebuilt at {self.session.capacity} experts a layer, "
                                 f"{self.session.footprint_gb:.2f} GB resident",
                       capacity=self.session.capacity)
            self.config_result = {"ok": True, **self.session.stats()}
        except Exception as e:
            # Put the old settings back rather than leave the server with no model at all.
            try:
                from .session import Session
                self.session = Session(**self.session_kwargs_backup)
                self.config_result = {"ok": False, "error": f"{e}; previous settings restored"}
                self.event("error", f"settings change failed, previous settings restored: {e}")
            except Exception as e2:
                self.config_result = {"ok": False, "error": f"{e}; and could not restore: {e2}"}
        finally:
            self.config_event.set()

    def _maybe_shrink(self) -> None:
        """Hand memory back if the machine is short of it. Called only from the idle branch.

        Nothing here is allowed to fail loudly: this runs on the thread that owns the model, and
        an exception would take the server down over an optimisation nobody asked for.
        """
        if self.memctl is None or self.pending_config is not None:
            return
        with self.count_lock:
            idle = self.waiting == 0 and not self.by_id
        if not idle:
            return
        # ONE READING, ONE DECISION.
        #     This runs on every idle poll, five times a second; the pressure watcher samples
        #     once every ten seconds. Handing the controller the same reading on every poll
        #     turned its "two consecutive readings" into two consecutive glances at ONE reading,
        #     0.2 s apart -- measured, a server shrank 0.8 s after it started, before it had
        #     served a token. A reading is handed over exactly once, when it is new.
        if self.pressure_polls == self.pressure_fed:
            return
        self.pressure_fed = self.pressure_polls
        # THE CONTROLLER WORKS IN BUDGET, NOT IN REPORTED CAPACITY.
        #     With whole layers in play those are different numbers: a budget of 36 becomes 10
        #     whole layers plus 11 elsewhere, and 11 is what gets reported. A controller acting
        #     on 11 shrinks from 11, records 11 as home, and can never restore the plan --
        #     measured, it shrank once to the floor and then reported `grows 0` however long the
        #     machine stayed quiet. Budget in, budget out, so the plan is a fixed point.
        cap = int(getattr(self.session, "planned_from", None)
                  or getattr(self.session, "capacity", 0) or 0)
        if cap <= 0:
            return
        # REMEMBERED PROMPTS GO FIRST, BEFORE ANY POOL SLOT DOES.
        #     Shrinking the pool costs decode speed for as long as the shrink lasts and needs a
        #     full rebuild of the model to undo. Dropping a remembered prompt costs one re-read of
        #     one conversation, and nothing else the engine can do changes. So under pressure this
        #     is released first, and only if the machine is STILL short does a slot get taken.
        #
        #     It is also released without the rebuild path, so it can happen while the pool stays
        #     exactly as planned -- which is the difference between a pause and a reconfiguration.
        if self.pressure:
            freed = 0
            try:
                freed = self.session.trim_prompt_cache(0)
            except Exception:            # noqa: BLE001 -- never take the server down for this
                freed = 0
            if freed > 0:
                print(f"  [memory] released {freed / 1e9:.2f} GB of remembered prompts before "
                      f"touching the pool", flush=True)
                self.event("cache", f"released {freed / 1e9:.2f} GB of remembered prompts rather "
                                    f"than taking pool slots", bytes=int(freed))
                return
        d = self.memctl.decide(cap, self.pressure, time.time(), idle=True)
        if not d:
            return
        new, why = d
        # Through the same path a user's slider takes, so there is exactly one way the pool is
        # ever rebuilt and it is the one that has been tested.
        self.pending_config = {"capacity": int(new)}
        self.config_event.clear()
        self.shrink_log.append({"at": round(time.time() - self.started, 1),
                                "from": int(cap), "to": int(new), "why": why})
        print(f"  [memory] {why}", flush=True)
        self.event("memory", why, **{"from": int(cap), "to": int(new)})

    def watch_pressure(self, period: float = 10.0) -> None:
        """Sample memory pressure on a thread of its own, forever. Never touches the model.

        On its own thread because `under_pressure` sleeps for its sampling window, and the pump
        is the thread that answers requests -- it must never be the one waiting on a stopwatch.
        What it reads is the compressor GROWING or swap being written during that window, not
        absolute occupancy, which is cumulative and sits at a gigabyte on an idle machine.
        """
        from .calibrate import under_pressure
        while not self.stopping:
            try:
                p = under_pressure()
            except Exception:                    # noqa: BLE001 -- a reading must never kill it
                p = False
            if getattr(self, "pressure_forced", False):
                # FORCED: THE WATCHER STOPS PRODUCING READINGS ALTOGETHER.
                #     Its sample is discarded here, so counting it as a reading would hand the
                #     controller a confirmation nobody asked for. Measured: a test posted one
                #     forced reading and the pool shrank, because a watcher tick 10 s later
                #     supplied the second confirmation on the still-forced value. When forced,
                #     `/v1/debug/pressure` is the only source of readings, which is the whole
                #     point of being able to force it.
                self.pressure_at = time.time()
            else:
                if bool(p) != bool(self.pressure):
                    self.event("pressure", "the machine is short of memory" if p
                               else "memory pressure has cleared")
                self.pressure = bool(p)
                self.pressure_at = time.time()
                self.pressure_polls += 1
            for _ in range(int(max(1.0, period) * 2)):
                if self.stopping:
                    return
                time.sleep(0.5)

    def pump(self, poll: float = 0.2):
        """Run queued generations on THIS thread until stopped. Must be the main thread."""
        while not self.stopping:
            if self.pending_config is not None:
                self._apply_config()
                continue
            try:
                j = self.jobs.get(timeout=poll)
            except queue.Empty:
                # IDLE, AND ONLY HERE. This branch is reached when the queue has been empty for
                # a whole poll, which is the one moment no reply is in flight -- a rebuild any
                # other time would resize the pool underneath a running generation.
                self._maybe_shrink()
                continue
            with self.count_lock:
                self.waiting -= 1
            group = [j]
            if self.batch_size > 1:
                # Only requests ALREADY waiting join the pass. Holding the first one back for a
                # few milliseconds in the hope that a second arrives would trade a certain
                # latency cost for a speculative throughput gain, which is the wrong way round
                # for a server most often used by one person.
                while len(group) < self.batch_size:
                    try:
                        k = self.jobs.get_nowait()
                    except queue.Empty:
                        break
                    with self.count_lock:
                        self.waiting -= 1
                    group.append(k)
            if len(group) == 1:
                self._run_one(group[0])
            else:
                self._run_batch(group)

    def _run_one(self, j):
        try:
            # NOBODY IS WAITING FOR THIS ONE. CHECKED BEFORE ANY WORK, NOT AFTER THE FIRST TOKEN.
            #     The loop below checks `cancelled` between chunks, which is too late by exactly
            #     the length of the prefill -- and the prefill is where the time is. Measured with
            #     a real agent: Codex's client-side timeout is shorter than a 12,704-token prompt
            #     takes on this model, so it gives up and retries, and each retry queued ANOTHER
            #     full prefill for a request it had already abandoned. Observed three deep, with
            #     the GPU at 70% doing entirely wasted work.
            if j.cancelled.is_set():
                self._finish(j, ("done", None, None))
                return
            kw = {k: v for k, v in j.kw.items() if not k.startswith("_")}

            def on_prefill(done, total):
                # Never block the one thread allowed to touch MLX on a slow client: progress is
                # a courtesy, and a dropped progress frame costs nothing.
                if j.cancelled.is_set():
                    # AND ABORT MID-PREFILL, WHICH IS THE ONLY PLACE IT CAN BE ABORTED.
                    #     A client that hangs up while its prompt is being read leaves nothing
                    #     else to notice: `stream_text` yields nothing until prefill is finished,
                    #     so the check between chunks is never reached. This is the only code
                    #     that runs during a prefill.
                    #
                    #     THE LIMIT, STATED. This works for STREAMED requests only. A blocking
                    #     one writes nothing until generation has finished, so its handler never
                    #     touches the socket, never learns the client has gone, and never sets
                    #     `cancelled` -- there is nothing here to notice. Measured: an abandoned
                    #     streaming request stops costing anything (44.6s -> 1.4s for the request
                    #     behind it), an abandoned blocking one still runs to completion. Agents
                    #     stream, which is why this is worth having anyway.
                    raise _Abandoned()
                try:
                    j.out.put_nowait(("prefill", int(done), int(total)))
                except queue.Full:
                    pass

            # THE API RETURNS THE ANSWER, NOT THE SCRATCHPAD. Every modern model of this class
            # thinks out loud first; the terminal shows that inline, an API must not, or a coding
            # agent receives the model's reasoning as its reply. The reasoning is handed back
            # separately, per chunk, and each endpoint reports it in its own shape.
            for chunk, info in self.session.stream_text(on_prefill=on_prefill,
                                                        hide_reasoning=True, **kw):
                if j.cancelled.is_set():
                    break                                # client is gone; stop burning tokens
                try:
                    j.out.put(("chunk", chunk, info), timeout=PUT_TIMEOUT_S)
                except queue.Full:
                    # A consumer this far behind is not coming back. Bounded wait, then drop
                    # the request -- never an unbounded put on the only MLX thread.
                    j.cancelled.set()
                    break
            self._finish(j, ("done", None, None))
            with self.count_lock:
                self.served += 1
                if self.served % 100 == 0 and getattr(self.session, "handle", None) is not None:
                    try:                        # a crash should not lose a week of use counts
                        from . import stream as _stm
                        self.session.handle.save_usage(_stm.usage_path(self.session.name))
                    except Exception:           # noqa: BLE001 -- a record, never a failure
                        pass
        except _Abandoned:
            # Not an error. The client went away mid-prefill and this stopped rather than
            # finishing a prompt nobody will read. Reported as a clean end so nothing downstream
            # has to distinguish it -- there is no client left to tell either way.
            self._finish(j, ("done", None, None))
        except Exception as e:                           # never let one request kill the server
            self._finish(j, ("error", e, None))

    def _run_batch(self, group):
        """Serve a group in one pass. Falls back to one at a time if the batch cannot run.

        A failure here would otherwise take down several clients at once, so the fallback is not
        optional: whatever the batch could not do, the serial path still can.
        """
        def emit(i, chunk, info):
            j = group[i]
            if j.cancelled.is_set():
                return
            try:
                j.out.put(("chunk", chunk, info), timeout=PUT_TIMEOUT_S)
            except queue.Full:
                j.cancelled.set()
        try:
            self.session.stream_batch(
                [{k: v for k, v in j.kw.items() if not k.startswith("_")} for j in group], emit)
        except Exception:
            # Anything already emitted is text the client has seen, so re-running a partially
            # served job would repeat it. Only the untouched ones fall back.
            fresh = [j for j in group if j.out.qsize() == 0 and not j.cancelled.is_set()]
            for j in group:
                if j not in fresh:
                    self._finish(j, ("done", None, None))
            for j in fresh:
                self._run_one(j)
            with self.count_lock:
                self.served += len(group) - len(fresh)
            return
        for j in group:
            self._finish(j, ("done", None, None))
        with self.count_lock:
            self.served += len(group)

    @staticmethod
    def _finish(j, msg):
        try:
            j.out.put(msg, timeout=PUT_TIMEOUT_S)
        except queue.Full:
            pass


def allowed_origins(host: str, port: int, extra=()) -> frozenset:
    """Origins a browser is allowed to READ this server's replies from.

    The bundled page is served by this very server, so its own origin has to be on the list or
    the UI stops working; everything else is opt-in through `--cors-origin`. `0.0.0.0` is a bind
    address, never an origin a browser sends, so it is not added -- a LAN client's page origin is
    the interface address it dialled, which the user must name explicitly.
    """
    names = {host, "localhost", "127.0.0.1", "[::1]"}
    names.discard("0.0.0.0")
    names.discard("::")
    out = set()
    for n in names:
        out.add(f"http://{n}:{port}")
        out.add(f"https://{n}:{port}")
    return frozenset(out | {e.rstrip("/") for e in extra})


def _cors(handler):
    """The Origin to echo back, or None. Set by `_guard` on every request."""
    return getattr(handler, "cors_origin", None)


def _json(handler, code, payload):
    b = json.dumps(payload).encode()
    handler.send_response(code)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(b)))
    # WHY THIS IS NOT `*`. This server has no authentication, so a wildcard let any page the
    # user happened to have open POST to /v1/chat/completions and READ the reply -- their model,
    # their hardware, their conversation, on someone else's page. The echo is now an exact match
    # against a list that holds this server's own origin and whatever `--cors-origin` added.
    if _cors(handler):
        handler.send_header("Access-Control-Allow-Origin", _cors(handler))
        handler.send_header("Vary", "Origin")
    if getattr(handler, "close_connection", False):
        # Setting close_connection alone drops the socket but tells the client nothing. On a
        # keep-alive connection whose body we refused to read, the client needs the header --
        # otherwise it reuses the socket and its next request lands on unread body bytes.
        handler.send_header("Connection", "close")
    handler.end_headers()
    handler.wfile.write(b)


def _median(xs):
    xs = sorted(xs)
    return round(xs[len(xs) // 2], 2) if xs else None


def _aggregate(state) -> dict:
    """Summarise the rolling reply history. Empty history gives an empty dict, not zeros --
    a zero would read as "measured 0 tok/s" rather than "nothing measured yet"."""
    with state.count_lock:
        hist = list(state.history)
    if not hist:
        return {}
    return {"replies": len(hist),
            "tokens": sum(r.get("tokens", 0) for r in hist),
            "seconds": round(sum(r.get("seconds", 0) for r in hist), 1),
            "flagged": sum(r.get("flagged", 0) for r in hist),
            "median_tok_s": _median([r["tok_s"] for r in hist if r.get("tok_s")]),
            "median_ttft": _median([r["ttft"] for r in hist if r.get("ttft")]),
            "cut_off": sum(1 for r in hist if r.get("finish") == "length")}


def make_handler(state: _State):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "bigrig"
        # A CLIENT THAT ANNOUNCES A BODY AND THEN SENDS NOTHING USED TO PIN A THREAD FOREVER.
        #     ThreadingHTTPServer gives every connection a thread and `rfile.read(n)` on a socket
        #     with no timeout blocks until the peer speaks. Measured: 25 connections each sending
        #     `Content-Length: 4000000` and one byte held 25 threads indefinitely. It also fires
        #     by accident, whenever a client is killed mid-upload. A socket timeout only trips on
        #     a BLOCKING operation, so a long prefill with nothing to write is unaffected.
        timeout = 60

        def log_message(self, fmt, *a):        # keep the console for the model, not for access logs
            pass

        # ---------------------------------------------------------------- origin and host
        def _guard(self, writes: bool) -> bool:
            """Decide whether this request may proceed. True means stop; the reply is sent.

            THREE CHECKS, AND ALL THREE ARE NEEDED. Dropping the wildcard CORS header alone does
            not close the hole: a POST with `Content-Type: text/plain` is a *simple* request, so
            the browser sends it with no preflight and the write lands even though the reply is
            unreadable. So the Origin is checked on writes as well as echoed on reads, and a JSON
            content type is required -- which cannot be set cross-origin without a preflight.
            The Host check is for DNS rebinding, where a name the attacker controls resolves to
            127.0.0.1 and the Origin is then legitimately theirs.
            """
            def refuse(code, message):
                # EVERY REFUSAL HERE HAPPENS BEFORE THE BODY IS READ, so the socket still holds
                # however many bytes the client announced. On a keep-alive connection those get
                # parsed as the next request line. Closing is the only correct answer, and
                # `_json` turns this flag into the `Connection: close` header the client needs.
                self.close_connection = True
                _json(self, code, {"error": {"message": message,
                                             "type": "invalid_request_error"}})
                return True

            # A MISSING Host IS ALLOWED; A WRONG ONE IS NOT. Every browser sends one, so DNS
            # rebinding -- a name the attacker owns, resolved to 127.0.0.1, carrying their own
            # legitimate Origin -- always arrives with their host in it. Plenty of non-browser
            # clients omit the header, and refusing those buys nothing.
            host = (self.headers.get("Host") or "").strip().lower()
            if state.loopback and host:
                bare = host.rsplit(":", 1)[0] if host.count(":") == 1 else host
                if bare.strip("[]") not in ("localhost", "127.0.0.1", "::1", state.bind_host):
                    return refuse(421, f"this server answers only to localhost, not {host!r}")
            origin = (self.headers.get("Origin") or "").strip().rstrip("/")
            if origin:
                if origin not in state.origins:
                    return refuse(403, f"origin {origin} is not allowed to use this server; "
                                       f"start it with --cors-origin {origin} if that is what "
                                       f"you want")
                self.cors_origin = origin
            if writes:
                ct = (self.headers.get("Content-Type") or "").split(";", 1)[0].strip().lower()
                if ct != "application/json":
                    return refuse(415, "Content-Type must be application/json")
            return False

        # ---------------------------------------------------------------- routing
        def do_OPTIONS(self):
            if self._guard(writes=False):
                return
            self.send_response(204)
            if _cors(self):
                self.send_header("Access-Control-Allow-Origin", _cors(self))
                self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
                self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
                self.send_header("Vary", "Origin")
            self.send_header("Content-Length", "0")
            self.end_headers()

        def do_GET(self):
            if self._guard(writes=False):
                return
            path = self.path.split("?", 1)[0].rstrip("/") or "/"
            if path == "/":
                # The whole UI is one self-contained file: no CDN, no build step, no third-party
                # request from a page that is meant to prove the model runs entirely locally.
                try:
                    with open(os.path.join(os.path.dirname(__file__), "webui.html"), "rb") as f:
                        page = f.read()
                except OSError:
                    return _json(self, 500, {"error": {
                        "message": "the web interface is missing from this install",
                        "type": "server_error"}})
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(page)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(page)
                return
            if self.path.rstrip("/") in ("/health", "/v1/health"):
                # THE SESSION IS None WHILE IT IS BEING REBUILT, AND THAT IS A NORMAL SECOND.
                #     `_apply_config` drops the old session before building the new one, on
                #     purpose, so two pools never exist at once. Every request arriving in that
                #     window used to crash the handler thread on NoneType.stats(), which is how
                #     a routine reload turned into a dead server.
                if state.session is None:
                    return _json(self, 503, {"status": "reloading", "server": "bigrig",
                                             "detail": "the model is being rebuilt; retry in "
                                                       "a moment"})
                s = state.session.stats()
                # Lifetime totals, across reloads.
                if state.tokens_before:
                    s["total_tokens"] = int(s.get("total_tokens") or 0) + state.tokens_before
                    s["flagged_tokens"] = int(s.get("flagged_tokens") or 0) + state.flagged_before
                    s["flagged_share"] = (s["flagged_tokens"] / s["total_tokens"]
                                          if s["total_tokens"] else 0.0)
                # The measured rate belongs here as well as in /stats. The chat page reads only
                # /health on its refresh, and without a rate it cannot say how long a reply of a
                # given length will take -- so the estimate it showed was half blank.
                agg = _aggregate(state)
                spd = _speed_verdict(state, s, agg)
                # `server` is here so `bigrig launch --port N` can tell OUR server from whatever
                # else happens to answer 200 on that port. Without it, naming a port already held
                # by another process pointed the coding agent -- and the source it sends -- at a
                # stranger.
                return _json(self, 200, {"server": "bigrig",
                    "status": "ok", "uptime_s": round(time.time() - state.started, 1),
                    "speed_tier": spd["label"], "speed_why": spd["why"],
                    "speed_basis": spd["basis"], "speed_hint": spd["hint"],
                    "queue_depth": state.waiting, "requests_served": state.served,
                    "reloads": state.reloads,
                    "release_memory": state.memctl is not None,
                    "under_pressure": bool(state.pressure) if state.memctl else None,
                    "memory_released": (state.memctl.stats() if state.memctl else None),
                    "shrink_log": list(state.shrink_log),
                    "page_cache_warm": state.warm,
                    "median_tok_s": agg.get("median_tok_s"),
                    "median_ttft": agg.get("median_ttft"), **s})
            if path in ("/stats", "/v1/stats") and state.session is None:
                return _json(self, 503, {"status": "reloading"})
            if path in ("/stats", "/v1/stats"):
                with state.count_lock:
                    hist = list(state.history)
                return _json(self, 200, {"history": hist, "aggregate": _aggregate(state),
                                         "model": state.session.name})
            if path in ("/events", "/v1/events"):
                # `since` makes this a feed rather than a snapshot: the page sends the last seq
                # it saw and gets only what happened after it, so a console left open for an
                # hour is not re-rendered from scratch every two seconds.
                try:
                    since = int((self.path.split("since=", 1) + ["0"])[1].split("&", 1)[0])
                except (ValueError, IndexError):
                    since = 0
                with state.count_lock:
                    evs = [e for e in state.events if e["seq"] > since]
                    seq = state.event_seq
                return _json(self, 200, {"events": evs, "seq": seq})
            if self.path.rstrip("/") == "/v1/models":
                return _json(self, 200, {"object": "list", "data": [
                    {"id": state.session.name, "object": "model",
                     "created": int(state.started), "owned_by": "bigrig"}]})
            return _json(self, 404, {"error": {"message": f"no route {self.path}",
                                               "type": "invalid_request_error"}})

        def do_POST(self):
            if self._guard(writes=True):
                return
            # do_GET splits the query string off; this did not, so a client that appended one --
            # several OpenAI-compatible ones do -- got a confusing 404 on a route that exists.
            p = self.path.split("?", 1)[0].rstrip("/")
            if state.session is None and p != "/v1/debug/pressure":
                return _json(self, 503, {"error": {
                    "message": "the model is being rebuilt; retry in a moment",
                    "type": "server_error"}})
            if p not in ("/v1/chat/completions", "/v1/completions", "/v1/messages",
                         "/v1/messages/count_tokens", "/v1/cancel", "/v1/config",
                         "/v1/responses", "/v1/debug/pressure"):
                return _json(self, 404, {"error": {"message": f"no route {self.path}",
                                                   "type": "invalid_request_error"}})
            try:
                n = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                return self._bad("Content-Length is not a number")
            if n <= 0:
                return self._bad("a request body is required")
            if n > MAX_BODY:
                self.close_connection = True
                return self._bad(f"body of {n} bytes exceeds the {MAX_BODY} byte limit")
            try:
                body = json.loads(self.rfile.read(n) or b"{}")
            except json.JSONDecodeError as e:
                return self._bad(f"body is not valid JSON: {e}")
            if not isinstance(body, dict):
                return self._bad("body must be a JSON object")

            if p == "/v1/debug/pressure":
                # FORCING THE SIGNAL, SO THE CONTROLLER CAN BE TESTED WITHOUT SQUEEZING A REAL
                # MACHINE. Deliberately exhausting 26 GB of RAM to watch a shrink risks the
                # machine and proves nothing extra: what needs testing is that a decision
                # reaches the pool, never lands mid-reply, and stops at the floor. Whether
                # macOS's pressure reading is right is `calibrate.under_pressure`'s business.
                #
                # It exists only when BIGRIG_DEBUG_PRESSURE is set, so a normal install has no
                # route by which anything can lie to the controller.
                if not os.environ.get("BIGRIG_DEBUG_PRESSURE"):
                    return _json(self, 404, {"error": {
                        "message": "not enabled; set BIGRIG_DEBUG_PRESSURE=1 to expose it",
                        "type": "invalid_request_error"}})
                state.pressure = bool(body.get("pressure"))
                state.pressure_forced = True
                # One forced reading per call, so a test drives exactly the confirmations it
                # means to -- the controller is handed a reading once, not once per idle poll.
                state.pressure_polls += 1
                state.pressure_at = time.time()
                if state.memctl is not None:
                    for k in ("grow_quiet_s", "min_interval_s", "grace_s"):
                        if k in body:
                            setattr(state.memctl, k, max(0.0, float(body[k])))
                return _json(self, 200, {"ok": True, "pressure": state.pressure,
                                         "forced": True, "polls": state.pressure_polls})

            if p == "/v1/config":
                # Changing residency means rebuilding the pool, which means touching MLX, which
                # only the pump's thread may do. The request is handed over and waited on.
                if state.pending_config is not None:
                    return self._bad("a settings change is already being applied")
                want = {}
                cap = body.get("capacity")
                if cap is not None:
                    try:
                        cap = float(cap)
                    except (TypeError, ValueError):
                        return self._bad("`capacity` must be a number")
                    n_exp = state.session.plan.get("n_experts") or 1
                    if not 0 < cap <= n_exp:
                        return self._bad("`capacity` must be between 1 and the expert count")
                    cap = int(cap) if cap > 1 else cap
                    # PRICE IT BEFORE ACCEPTING IT.
                    #     A capacity given explicitly is honoured everywhere else in this engine,
                    #     which is right for a flag someone typed and wrong for a slider in a
                    #     browser. Asked for all 128 experts of a 12.68 GB model against a 9 GB
                    #     ceiling, this endpoint said 200 and took the footprint to 30.68 GB on a
                    #     26 GB machine -- the exact failure the ceiling exists to prevent.
                    # Numbers only, and no reference to the session kept past this block. This
                    # handler's thread then blocks on `config_event` for as long as the reload
                    # takes, and a local holding the OLD session would keep its pool alive while
                    # the pump builds the new one -- two pools at once is precisely what the
                    # reload sequence is written to avoid. Measured with the reference held: a
                    # 40 -> 53 reload reported 10.56 GB for 5.25 GB of weights, which then fed
                    # the reply ceiling and cut it from 13,852 tokens to the 256 floor.
                    _s = state.session
                    budget, non_exp = _s.budget_gb, _s.non_expert_gb
                    work = _s.working_memory_gb
                    per = (_s.plan.get("bytes_per_expert", 0) * _s.plan.get("n_layers", 0)) / 1e9
                    del _s
                    slots = int(round(cap * n_exp)) if cap <= 1.0 else int(cap)
                    need = slots * per + non_exp + work
                    if need > budget:
                        fits = max(1, int((budget - non_exp - work) // per)) if per else n_exp
                        return self._bad(
                            f"{slots} experts per layer needs about {need:.1f} GB and the budget "
                            f"is {budget:.1f} GB. The most that fits is {fits}.")
                    want["capacity"] = cap
                # CLEARING REMEMBERED PROMPTS IS NOT A RELOAD, AND MUST NOT BE TREATED AS ONE.
                #     Everything else on this endpoint rebuilds the pool, which takes seconds and
                #     goes through the pump. This just drops a dictionary. Answering it here, and
                #     returning immediately, is the difference between a button that responds and
                #     a button that appears to hang the page.
                if body.get("clear_prompt_cache"):
                    try:
                        freed = state.session.trim_prompt_cache(0)
                    except Exception as e:                     # noqa: BLE001
                        return _json(self, 500, {"error": {"message": f"could not clear: {e}",
                                                           "type": "engine_error"}})
                    state.event("cache", f"remembered prompts cleared by hand, "
                                         f"{freed / 1e9:.2f} GB released", bytes=int(freed))
                    return _json(self, 200, {"ok": True, "freed_bytes": int(freed),
                                             "prompt_cache_bytes": 0})
                for name in ("prefetch_width", "reroute", "draft_tokens", "prompt_cache_gb"):
                    if body.get(name) is not None:
                        want[name] = body[name]
                if want.get("prompt_cache_gb") is not None:
                    try:
                        v = float(want["prompt_cache_gb"])
                    except (TypeError, ValueError):
                        return self._bad("`prompt_cache_gb` must be a number")
                    if not 0 <= v <= 8:
                        return self._bad("`prompt_cache_gb` must be between 0 and 8")
                    want["prompt_cache_gb"] = v
                if not want:
                    return self._bad("nothing to change")
                state.config_event.clear()
                state.config_result = None
                state.pending_config = want
                if not state.config_event.wait(timeout=300.0):
                    return _json(self, 504, {"error": {"message": "the model did not finish "
                                                       "reloading in time",
                                                       "type": "engine_error"}})
                res = state.config_result or {"ok": False, "error": "no result"}
                return _json(self, 200 if res.get("ok") else 500, res)

            if p == "/v1/cancel":
                # Stopping by name rather than by hanging up. A closed socket is only noticed
                # when the next write fails, and the OS buffers writes -- at single-digit tokens
                # per second that is seconds of a model still running after the user pressed
                # stop. Whatever was generated up to the stop is kept: the client already has it,
                # and it stays in the conversation so the next turn still has the context.
                rid = body.get("id")
                if not isinstance(rid, (str, int)) or not str(rid):
                    return self._bad("`id` is required and must be the id sent with the request")
                found = state.cancel(str(rid)[:64])
                return _json(self, 200, {"cancelled": found, "id": str(rid)[:64]})

            if p == "/v1/responses":
                return self._responses(body)
            if p.startswith("/v1/messages"):
                return self._anthropic(body, count_only=p.endswith("count_tokens"))

            chat = p.endswith("chat/completions")
            msgs = body.get("messages") if chat else None
            prompt = "" if chat else (body.get("prompt") or "")
            if chat and not (isinstance(msgs, list) and msgs):
                return self._bad("`messages` must be a non-empty array")
            if not chat and not isinstance(prompt, str):
                return self._bad("`prompt` must be a string")
            if chat:
                for m in msgs:
                    if not isinstance(m, dict) or "content" not in m:
                        return self._bad("each message needs a `content` field")

            try:
                kw = self._sampling(body, state.session.max_completion_tokens,
                                    state.session.token_limit_reason)
            except ValueError as e:
                return self._bad(str(e))

            if body.get("stream"):
                return self._stream(msgs, prompt, kw, chat)
            return self._blocking(msgs, prompt, kw, chat)

        # ---------------------------------------------------------------- helpers
        def _bad(self, msg):
            _json(self, 400, {"error": {"message": msg, "type": "invalid_request_error"}})

        @staticmethod
        def _sampling(body, ceiling: int = anth.MAX_TOKENS_LIMIT, why: str = "") -> dict:
            def num(key, default, lo, hi):
                v = body.get(key, default)
                if v is None:
                    return default
                try:
                    v = float(v)
                except (TypeError, ValueError):
                    raise ValueError(f"`{key}` must be a number, got {v!r}")
                if not lo <= v <= hi:
                    raise ValueError(f"`{key}` must be between {lo} and {hi}, got {v}")
                return v
            mt = body.get("max_tokens", 512)
            try:
                mt = int(mt) if mt is not None else 512
            except (TypeError, ValueError):
                raise ValueError(f"`max_tokens` must be an integer, got {mt!r}")
            # The ceiling is the model's own context window or what the memory budget leaves for
            # KV cache, whichever is smaller -- not a number chosen in advance. Naming which one
            # bound is what turns it from an arbitrary restriction into something actionable.
            if not 1 <= mt <= ceiling:
                tail = {"memory": " (what the memory budget leaves for KV cache)",
                        "context": " (the model's context window)"}.get(why, "")
                raise ValueError(f"`max_tokens` must be between 1 and {ceiling}{tail}, got {mt}")
            # `think` is a bigrig extension, not part of the OpenAI schema. Conforming
            # clients never send it and are unaffected; ours sends it so a reasoning model can
            # be asked for the answer without the working.
            # `_rid` is the client's own name for this request, used only so it can be stopped
            # by name later. A conforming OpenAI client never sends one; without it the stop
            # button falls back to closing the socket, which works but is not immediate.
            rid = body.get("bigrig_id")
            # GUESSING AHEAD, ASKED FOR PER REQUEST BECAUSE ONLY THE REQUEST KNOWS IF IT HELPS.
            #     Whether drafting from earlier text pays is a property of the WORK, not of the
            #     machine, and the two directions are both large. Measured on this model:
            #     reproducing a passage almost verbatim 1.48x, ordinary prose 0.86x. A server
            #     default would be right for one caller and wrong for the next, so there is none.
            # TOOLS. Passed through to the chat template, which renders the signatures in the
            # form this model was trained on. Rejected rather than ignored when the model has no
            # tool-call delimiters: a client whose tools are silently dropped gets a fluent
            # answer to a question it did not ask and no way to discover why.
            tools = body.get("tools")
            if tools is not None:
                if not isinstance(tools, list) or not all(isinstance(t, dict) for t in tools):
                    raise ValueError("`tools` must be a list of tool definitions")
                # `state.session` is None for the second or two a reload takes. do_POST turns
                # that into a 503 before reaching here, but this function is also called to
                # validate a request in isolation -- and a validator that raises AttributeError
                # on a rebuild is a validator that takes the handler thread down with it.
                _sess = state.session
                if tools and _sess is not None and not _sess.supports_tools():
                    raise ValueError(
                        f"{_sess.name} has no tool-call format, so it cannot be asked to "
                        f"call one. Serve a model whose chat template defines tool calling.")
            la = bool(body.get("lookahead", False))
            lat = body.get("lookahead_tokens", LOOKAHEAD_TOKENS)
            try:
                lat = max(1, min(32, int(lat)))
            except (TypeError, ValueError):
                lat = LOOKAHEAD_TOKENS
            return {"max_tokens": mt, "temperature": num("temperature", 0.7, 0.0, 2.0),
                    "top_p": num("top_p", 0.95, 0.0, 1.0),
                    "think": bool(body.get("think", True)),
                    "continue_last": bool(body.get("continue_last", False)),
                    "lookahead": la, "lookahead_tokens": lat,
                    # None means "whatever the server was started with"; a request can only
                    # switch the head OFF, which is what an A/B on one warm server needs.
                    "mtp": (None if "mtp" not in body else bool(body.get("mtp"))),
                    "tools": tools or None,
                    "_rid": str(rid)[:64] if isinstance(rid, (str, int)) else ""}

        def _client_gone(self) -> bool:
            return client_gone(getattr(self, "connection", None))

        def _run(self, msgs, prompt, kw, rid: str = "", want_prefill: bool = False):
            """Hand the work to the main thread and re-yield what comes back.

            The `finally` is load-bearing: closing this generator -- which is what happens when
            the client disconnects and the caller stops iterating -- signals the generator thread
            to stop. Without it the server deadlocks on the next full queue.
            """
            job = state.submit({"messages": msgs, "prompt": prompt, **kw}, rid=rid)
            try:
                # THE CHECK IS ON A CLOCK, NOT ON AN IDLE QUEUE, AND THAT DISTINCTION IS THE
                # WHOLE FIX.
                #     The first version of this only looked for a departed client in the
                #     `queue.Empty` branch, which reads as though it covers the waiting case and
                #     does not cover any case at all: at 13 tok/s a chunk lands every 77 ms, the
                #     get() never times out, and the branch never runs. It passed every unit test
                #     and failed the only measurement that mattered -- an abandoned blocking
                #     request still produced all 400 tokens in 29.4 s. Checking on elapsed time
                #     instead runs whether or not tokens are flowing, which is exactly when a
                #     blocking request needs it.
                #
                # `deadline` is the no-output deadline, pushed back every time something arrives,
                # so GET_TIMEOUT_S keeps its old meaning: longest SILENCE, not longest request.
                deadline = time.monotonic() + GET_TIMEOUT_S
                next_check = time.monotonic() + CLIENT_POLL_S
                while True:
                    try:
                        kind, a, b = job.out.get(timeout=CLIENT_POLL_S)
                    except queue.Empty:
                        kind = None
                    now = time.monotonic()
                    if now >= next_check:
                        next_check = now + CLIENT_POLL_S
                        if self._client_gone():
                            # THE BLOCKING PATH'S ONLY WAY TO LEARN IT IS ALONE.
                            #     A streaming handler finds out by writing to a dead socket. A
                            #     blocking one writes nothing until it is finished, so without
                            #     this it generates a full reply for a client that hung up.
                            #     `cancelled` is what `_run_one` already checks between chunks
                            #     and mid-prefill, so setting it here reuses the whole existing
                            #     mechanism rather than adding a second one; the `finally` below
                            #     sets it too, and setting an Event twice is harmless.
                            job.cancelled.set()
                            return
                    if kind is None:
                        if now >= deadline:
                            raise TimeoutError(
                                f"no output for {GET_TIMEOUT_S:.0f}s; giving up on this request")
                        continue
                    deadline = now + GET_TIMEOUT_S
                    if kind == "chunk":
                        yield a, b
                    elif kind == "prefill":
                        # Dropped unless the caller asked for it. A progress report is not a
                        # token, and every other consumer here appends what it is given straight
                        # into the reply -- the blocking path crashed on the first one with
                        # `c` being None, which took the whole request down with it.
                        if want_prefill:
                            yield None, {"prefill_done": a, "prefill_total": b}
                    elif kind == "done":
                        return
                    else:
                        raise a
            finally:
                job.cancelled.set()
                state.forget(rid)

        # ---------------------------------------------------------------- anthropic
        def _anthropic(self, body, count_only=False):
            try:
                parsed = anth.parse(body, require_max_tokens=not count_only,
                                    max_tokens_limit=state.session.max_completion_tokens)
            except anth.BadRequest as e:
                # Anthropic's error envelope, not OpenAI's -- their SDK parses this shape.
                return _json(self, 400, {"type": "error",
                                         "error": {"type": "invalid_request_error",
                                                   "message": str(e)}})
            msgs = anth.to_engine_messages(parsed)
            if count_only:
                n = anth.count_tokens(state.session.tokenizer, parsed)
                return _json(self, 200, {"input_tokens": n})
            if parsed.get("tools") and not state.session.supports_tools():
                return self._bad(
                    f"{state.session.name} has no tool-call format, so it cannot be asked to "
                    f"call one. Serve a model whose chat template defines tool calling.")
            kw = {"max_tokens": parsed["max_tokens"], "temperature": parsed["temperature"],
                  "top_p": parsed["top_p"], "tools": parsed.get("tools")}
            if parsed["stream"]:
                return self._anthropic_stream(msgs, kw)
            return self._anthropic_blocking(msgs, kw)

        def _anthropic_blocking(self, msgs, kw):
            rid = anth.new_id()
            parts, last, degraded, reasoning = [], {}, 0, []
            gen = self._run(msgs, "", kw)
            try:
                for c, info in gen:
                    parts.append(c)
                    last = info
                    reasoning.append(info.get("reasoning_delta") or "")
                    degraded += int(bool(info.get("degraded")))
            except Exception as e:
                return _json(self, 500, {"type": "error",
                                         "error": {"type": "api_error", "message": str(e)}})
            finally:
                gen.close()
            n_out = last.get("generation_tokens", len(parts))
            st = state.session.stats()
            _txt = "".join(parts)
            _calls = []
            if kw.get("tools"):
                _txt, _calls = state.session.extract_tool_calls(_txt, kw.get("tools"))
            _json(self, 200, anth.message(
                rid, state.session.name, _txt, last.get("prompt_tokens", 0), n_out,
                last.get("finish_reason"),
                {"tok_s": round(last.get("tok_s") or 0.0, 2), "degraded_tokens": degraded,
                 "degraded_share": round(degraded / max(1, n_out), 4),
                 "mode": st.get("mode"), "weights_altered": st.get("weights_altered")},
                tool_calls=_calls, reasoning="".join(reasoning)))

        def _responses(self, body):
            """OpenAI's Responses API. The only wire protocol Codex CLI still speaks."""
            try:
                parsed = resp.parse(body, state.session.max_completion_tokens)
            except resp.BadRequest as e:
                return self._bad(str(e))
            if parsed["tools"] and not state.session.supports_tools():
                return self._bad(
                    f"{state.session.name} has no tool-call format, so it cannot be asked to "
                    f"call one. Serve a model whose chat template defines tool calling.")
            kw = {"max_tokens": parsed["max_tokens"], "temperature": parsed["temperature"],
                  "top_p": parsed["top_p"], "tools": parsed["tools"]}
            if parsed["stream"]:
                return self._responses_stream(parsed["messages"], kw)
            return self._responses_blocking(parsed["messages"], kw)

        def _responses_blocking(self, msgs, kw):
            rid = resp.new_id()
            parts, last, degraded, reasoning = [], {}, 0, []
            _t0, _ttft = time.time(), None
            gen = self._run(msgs, "", kw)
            try:
                for c, info in gen:
                    # TIME TO FIRST TOKEN MEANS WHEN THE MODEL STARTED PRODUCING, and for a
                    # model that thinks first that is the first reasoning token -- otherwise the
                    # number is blank for exactly the models that take longest to answer.
                    if (c or info.get("reasoning_delta")) and _ttft is None:
                        _ttft = time.time() - _t0
                    parts.append(c)
                    last = info
                    reasoning.append(info.get("reasoning_delta") or "")
                    degraded += int(bool(info.get("degraded")))
            except Exception as e:                        # noqa: BLE001
                return _json(self, 500, {"error": {"message": str(e), "type": "engine_error"}})
            finally:
                gen.close()
            txt = "".join(parts)
            calls = []
            if kw.get("tools"):
                txt, calls = state.session.extract_tool_calls(txt, kw["tools"])
            n_out = last.get("generation_tokens", len(parts))
            self._log(last, degraded, time.time() - _t0, _ttft, n_out)
            st = state.session.stats()
            return _json(self, 200, resp.response(
                rid, state.session.name, txt, last.get("prompt_tokens", 0), n_out,
                last.get("finish_reason"), calls,
                {"tok_s": round(last.get("tok_s") or 0.0, 2), "degraded_tokens": degraded,
                 "mode": st.get("mode"), "weights_altered": st.get("weights_altered")}))

        def _responses_stream(self, msgs, kw):
            rid = resp.new_id()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.close_connection = True
            if _cors(self):
                self.send_header("Access-Control-Allow-Origin", _cors(self))
                self.send_header("Vary", "Origin")
            self.end_headers()
            created, text_delta, call_frames, finish = resp.stream_frames(
                rid, state.session.name)
            splitter = (state.session.tool_splitter(kw.get("tools"))
                        if kw.get("tools") else None)
            gen = self._run(msgs, "", kw)
            _t0, _ttft = time.time(), None
            text, calls, degraded, last, n = "", [], 0, {}, 0
            try:
                frames, item = created()
                for f in frames:
                    self.wfile.write(f)
                self.wfile.flush()
                for c, info in gen:
                    # TIME TO FIRST TOKEN MEANS WHEN THE MODEL STARTED PRODUCING, and for a
                    # model that thinks first that is the first reasoning token -- otherwise the
                    # number is blank for exactly the models that take longest to answer.
                    if (c or info.get("reasoning_delta")) and _ttft is None:
                        _ttft = time.time() - _t0
                    last = info
                    n = info.get("generation_tokens", n + 1)
                    degraded += int(bool(info.get("degraded")))
                    if not c:
                        continue
                    if splitter is None:
                        text += c
                        self.wfile.write(text_delta(item["id"], c))
                    else:
                        t, found = splitter.feed(c)
                        if t:
                            text += t
                            self.wfile.write(text_delta(item["id"], t))
                        calls += found
                    self.wfile.flush()
                if splitter is not None:
                    t, found = splitter.finish()
                    if t:
                        text += t
                        self.wfile.write(text_delta(item["id"], t))
                    calls += found
                # Calls are announced AFTER the text item, at output_index 1 and up, because a
                # client places an item by the index it was told, not by arrival order.
                done = []
                for i, call in enumerate(calls):
                    fr, d = call_frames(call, i + 1)
                    for f in fr:
                        self.wfile.write(f)
                    done.append(d)
                for f in finish(item, text, done, last.get("prompt_tokens", 0), n,
                                last.get("finish_reason") == "length"):
                    self.wfile.write(f)
                self.wfile.flush()
                self._log(last, degraded, time.time() - _t0, _ttft, n)
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as e:                        # noqa: BLE001
                try:
                    self.wfile.write(resp.sse("error", {"type": "error", "message": str(e)}))
                except Exception:
                    pass
            finally:
                gen.close()

        def _anthropic_stream(self, msgs, kw):
            rid = anth.new_id()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            # Connection: close, NOT keep-alive. An SSE response has no Content-Length and this
            # server does not chunk, so on a kept-alive connection the browser can never tell
            # where the body ends: fetch's reader never reports `done`, the page waits forever,
            # and the input box stays disabled after the first message. Closing the connection
            # IS the end-of-body signal. curl never noticed because it just reads until EOF.
            self.send_header("Connection", "close")
            self.close_connection = True
            if _cors(self):
                self.send_header("Access-Control-Allow-Origin", _cors(self))
                self.send_header("Vary", "Origin")
            self.end_headers()
            # See ToolCallSplitter: a chunk can end mid-delimiter, so text is held back whenever
            # its tail could still turn into one.
            splitter = (state.session.tool_splitter(kw.get("tools"))
                        if kw.get("tools") else None)
            calls = []
            gen = self._run(msgs, "", kw)
            sent_start = False
            # THE REASONING AND THE ANSWER ARE SEPARATE BLOCKS IN THIS PROTOCOL.
            #     A model that thinks first produces reasoning before any answer text, so the
            #     block opened at index 0 is a thinking block whenever that is what arrives
            #     first; it is closed and a text block opened beside it the moment the answer
            #     starts. `idx` is the block currently open, and everything downstream -- the
            #     tool-call blocks and the final stop -- counts from it rather than assuming 0.
            idx, kind = 0, "text"
            try:
                degraded, last, n = 0, {}, 0
                for c, info in gen:
                    rd = info.get("reasoning_delta") or ""
                    if not sent_start:
                        # Held until the first token so input_tokens is real rather than guessed.
                        kind = "thinking" if (rd and not c) else "text"
                        for f in anth.start_frames(rid, state.session.name,
                                                   info.get("prompt_tokens", 0), kind):
                            self.wfile.write(f)
                        sent_start = True
                    last = info
                    n = info.get("generation_tokens", n + 1)
                    degraded += int(bool(info.get("degraded")))
                    if rd and kind == "thinking":
                        self.wfile.write(anth.delta_frame(rd, idx, "thinking"))
                        self.wfile.flush()
                    if c:
                        if kind == "thinking":
                            # The answer has started: close the thinking block and open the text
                            # one beside it. Two kinds of delta on one block is a protocol error.
                            self.wfile.write(anth.block_stop(idx))
                            idx += 1
                            kind = "text"
                            self.wfile.write(anth.block_start(idx, "text"))
                        if splitter is None:
                            self.wfile.write(anth.delta_frame(c, idx))
                        else:
                            txt, found = splitter.feed(c)
                            if txt:
                                self.wfile.write(anth.delta_frame(txt, idx))
                            calls += found
                        self.wfile.flush()
                if splitter is not None:
                    txt, found = splitter.finish()
                    if txt:
                        if kind == "thinking":
                            self.wfile.write(anth.block_stop(idx))
                            idx += 1
                            kind = "text"
                            self.wfile.write(anth.block_start(idx, "text"))
                        self.wfile.write(anth.delta_frame(txt, idx))
                    calls += found
                if not sent_start:
                    for f in anth.start_frames(rid, state.session.name, 0):
                        self.wfile.write(f)
                # Each call is its OWN content block, so the text block is closed first and the
                # index advances. Two blocks sharing an index is how a client ends up attaching a
                # call to the wrong text.
                closed = False
                if calls:
                    self.wfile.write(anth.block_stop(idx))
                    for call in calls:
                        idx += 1
                        for f in anth.tool_frames(call, idx):
                            self.wfile.write(f)
                    closed = True              # every block this reply opened is now stopped
                    self.wfile.flush()
                st = state.session.stats()
                for f in anth.end_frames(n, last.get("finish_reason"),
                                         {"degraded_tokens": degraded,
                                          "mode": st.get("mode"),
                                          "weights_altered": st.get("weights_altered")},
                                         index=(None if closed else idx),
                                         tool_calls=calls):
                    self.wfile.write(f)
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass                                   # the client hung up; not an error
            except Exception as e:
                try:
                    self.wfile.write(anth.sse("error", {"type": "error", "error": {
                        "type": "api_error", "message": str(e)}}))
                    self.wfile.flush()
                except Exception:
                    pass
            finally:
                gen.close()

        def _log(self, last, degraded, secs, ttft, n_out):
            st = state.session.stats()
            state.record({"t": round(time.time(), 2), "tokens": n_out,
                          "seconds": round(secs, 3), "ttft": round(ttft, 3) if ttft else None,
                          "tok_s": round(last.get("tok_s"), 2) if last.get("tok_s") else None,
                          "flagged": degraded, "finish": last.get("finish_reason"),
                          "prompt_tokens": last.get("prompt_tokens", 0),
                          "miss_rate": st.get("miss_rate"), "mode": st.get("mode")})
            # One line per reply, in the words a user would use. The analytics view already plots
            # these; the console says them, which is what makes a slow reply explicable at the
            # moment it happens rather than after opening a second page.
            reused = st.get("prompt_tokens_reused") or 0
            # WHAT THE GUESSING BOUGHT, WHEN IT WAS ASKED FOR. Without this a caller turns the
            # option on and has no way to learn whether it helped -- and it is a trade that goes
            # both ways, so "no way to learn" means "no way to use it correctly".
            _mt = getattr(state.session, "mtp_last", None)
            if _mt is not None and _mt.drafted:
                state.event("mtp",
                            f"{_mt.accepted} of {_mt.drafted} guesses right "
                            f"({_mt.acceptance:.0%}), {n_out} tokens in {_mt.rounds} passes"
                            + (f", {_mt.recomputed} redone" if _mt.recomputed else ""),
                            **_mt.as_dict())
            _la = getattr(state.session, "lookahead_stats", None)
            if _la is not None and _la.rounds:
                state.event("lookahead",
                            f"{_la.accepted} of {_la.drafted} guesses accepted "
                            f"({_la.acceptance:.0%}), {n_out} tokens in {_la.passes} passes"
                            + (f", backed off {_la.backed_off}x" if _la.backed_off else ""),
                            **_la.as_dict())
            state.event("reply", f"{n_out} tokens in {secs:.1f}s"
                                 + (f", first after {ttft:.1f}s" if ttft else "")
                                 + (f", {last['tok_s']:.1f} tok/s" if last.get("tok_s") else "")
                                 + f", {last.get('prompt_tokens', 0):,} prompt tokens"
                                 + (f" ({reused:,} reused)" if reused else ""),
                        tokens=n_out, seconds=round(secs, 2),
                        ttft=round(ttft, 2) if ttft else None)

        def _blocking(self, msgs, prompt, kw, chat):
            rid = f"chatcmpl-{uuid.uuid4().hex[:24]}"
            _client_rid = kw.pop("_rid", "")
            parts, last, degraded, reasoning = [], {}, 0, []
            _t0, _ttft = time.time(), None
            gen = self._run(msgs, prompt, kw, rid=_client_rid)
            try:
                for c, info in gen:
                    # TIME TO FIRST TOKEN MEANS WHEN THE MODEL STARTED PRODUCING, and for a
                    # model that thinks first that is the first reasoning token -- otherwise the
                    # number is blank for exactly the models that take longest to answer.
                    if (c or info.get("reasoning_delta")) and _ttft is None:
                        _ttft = time.time() - _t0
                    parts.append(c)
                    last = info
                    reasoning.append(info.get("reasoning_delta") or "")
                    degraded += int(bool(info.get("degraded")))
            except Exception as e:
                return _json(self, 500, {"error": {"message": str(e), "type": "engine_error"}})
            finally:
                gen.close()
            txt = "".join(parts)
            # A CALL IS NOT PROSE. The model marks one with its own delimiters; leaving them in
            # `content` would hand the client machine syntax as assistant text, and put it back
            # into the next turn's prompt.
            calls = []
            if kw.get("tools"):
                txt, calls = state.session.extract_tool_calls(txt, kw.get("tools"))
            n_out = last.get("generation_tokens", len(parts))
            self._log(last, degraded, time.time() - _t0, _ttft, n_out)
            n_in = last.get("prompt_tokens", 0)
            _msg = {"role": "assistant", "content": txt if txt.strip() else None}
            # The de-facto field for this, as served by vLLM, DeepSeek and OpenRouter. Additive:
            # a client that does not know it ignores it, and one that does can show the thinking.
            if any(reasoning):
                _msg["reasoning_content"] = "".join(reasoning)
            if calls:
                _msg["tool_calls"] = [
                    {"id": f"call_{uuid.uuid4().hex[:24]}", "type": "function",
                     "function": {"name": c.get("name", ""),
                                  "arguments": json.dumps(c.get("arguments", {}))}}
                    for c in calls]
            _fin = ("tool_calls" if calls else (last.get("finish_reason") or "stop"))
            body = {"id": rid, "object": "chat.completion" if chat else "text_completion",
                    "created": int(time.time()), "model": state.session.name,
                    "choices": [({"index": 0, "message": _msg, "finish_reason": _fin}
                                 if chat else
                                 {"index": 0, "text": txt,
                                  "finish_reason": last.get("finish_reason") or "stop"})],
                    "usage": {"prompt_tokens": n_in, "completion_tokens": n_out,
                              "total_tokens": n_in + n_out},
                    # Not part of the OpenAI schema, and deliberately so: a client that ignores
                    # unknown keys is unaffected, and one that reads them learns whether the
                    # answer it just got was generated while the meter was flagging degradation.
                    "bigrig": {"tok_s": round(last.get("tok_s") or 0.0, 2),
                                 "degraded_tokens": degraded,
                                 "degraded_share": round(degraded / max(1, n_out), 4),
                                 "residency": state.session.stats().get("residency"),
                                 "miss_rate": state.session.stats().get("miss_rate")}}
            _json(self, 200, body)

        def _stream(self, msgs, prompt, kw, chat):
            rid = f"chatcmpl-{uuid.uuid4().hex[:24]}"
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            # Connection: close, NOT keep-alive. An SSE response has no Content-Length and this
            # server does not chunk, so on a kept-alive connection the browser can never tell
            # where the body ends: fetch's reader never reports `done`, the page waits forever,
            # and the input box stays disabled after the first message. Closing the connection
            # IS the end-of-body signal. curl never noticed because it just reads until EOF.
            self.send_header("Connection", "close")
            self.close_connection = True
            if _cors(self):
                self.send_header("Access-Control-Allow-Origin", _cors(self))
                self.send_header("Vary", "Origin")
            self.end_headers()
            created = int(time.time())

            def sse(obj):
                self.wfile.write(f"data: {json.dumps(obj)}\n\n".encode())
                self.wfile.flush()

            def frame(delta, finish=None, extra=None):
                d = {"id": rid, "object": "chat.completion.chunk", "created": created,
                     "model": state.session.name,
                     "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
                if extra:
                    d["bigrig"] = extra
                return d
            # A CALL MUST NOT BE STREAMED OUT AS PROSE ONE CHARACTER AT A TIME.
            #     The blocking path can search the finished reply for the delimiters. This one
            #     cannot: a chunk may end in the middle of `<tool_call>`, and emitting `<tool` as
            #     assistant text puts machine syntax in front of the user and loses the call.
            #     See ToolCallSplitter.
            splitter = (state.session.tool_splitter(kw.get("tools"))
                        if kw.get("tools") else None)
            gen = self._run(msgs, prompt, kw, rid=kw.pop("_rid", ""), want_prefill=True)
            _t0, _ttft = time.time(), None
            n_calls = 0
            try:
                sse(frame({"role": "assistant", "content": ""}))
                degraded, last = 0, {}
                for c, info in gen:
                    if c is None and "prefill_total" in info:
                        # Not a token -- the prompt is still being read. Sent as its own frame so
                        # the page can say what it is waiting for instead of showing nothing.
                        sse(frame({}, None, {"prefill_done": info["prefill_done"],
                                             "prefill_total": info["prefill_total"]}))
                        continue
                    last = info
                    d = bool(info.get("degraded"))
                    degraded += int(d)
                    # THE THINKING, AS IT HAPPENS, IN ITS OWN FIELD. A client that knows this
                    # field can show the model working; one that does not sees only the answer,
                    # which is the point. Same shape vLLM and DeepSeek's own API use.
                    _rd = info.get("reasoning_delta")
                    # THE STREAMING ENDPOINT NEVER RECORDED THIS, and it is the one the web page
                    # uses -- so every reply typed into the browser logged no time to first token
                    # and the analytics median was built from API calls alone. Found while making
                    # thinking models report it at all.
                    if (c or _rd) and _ttft is None:
                        _ttft = time.time() - _t0
                    if _rd:
                        sse(frame({"reasoning_content": _rd}))
                    if c:
                        # The per-chunk flag is what makes a live quality meter possible. A
                        # single verdict at the end is a report; a signal per token is a gauge,
                        # and the gauge is the thing worth looking at while it generates.
                        ex = {"degraded": d}
                        if info.get("tok_s"):          # absent until it is a real measurement
                            ex["tok_s"] = round(info["tok_s"], 1)
                        if splitter is None:
                            sse(frame({"content": c}, extra=ex))
                        else:
                            txt, calls = splitter.feed(c)
                            if txt:
                                sse(frame({"content": txt}, extra=ex))
                            for call in calls:
                                n_calls += _emit_call(sse, frame, call, n_calls)
                if splitter is not None:
                    txt, calls = splitter.finish()
                    if txt:
                        sse(frame({"content": txt}))
                    for call in calls:
                        n_calls += _emit_call(sse, frame, call, n_calls)
                sse(frame({}, ("tool_calls" if n_calls else
                               (last.get("finish_reason") or "stop")),
                          {"tok_s": round(last.get("tok_s") or 0.0, 2),
                           "degraded_tokens": degraded,
                           "miss_rate": state.session.stats().get("miss_rate")}))
                self._log(last, degraded, time.time() - _t0, _ttft,
                          last.get("generation_tokens", 0))
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass                                   # the client hung up; not an error
            except Exception as e:
                try:
                    sse({"error": {"message": str(e), "type": "engine_error"}})
                except Exception:
                    pass
            finally:
                gen.close()                              # tells the generator thread to stop
    return Handler


def _emit_call(sse, frame, call, index: int) -> int:
    """One tool call as an OpenAI streaming delta. Returns 1, so callers can count them.

    Sent as a single delta rather than split across frames. The schema allows a name and its
    arguments to arrive in pieces, and clients handle that, but there is nothing to gain here:
    the call is only recognised once its closing delimiter has arrived, so by the time anything
    can be sent the whole of it is already known. Streaming it in fragments would be pretending
    to a liveness the parser does not have.
    """
    sse(frame({"tool_calls": [{
        "index": index, "id": f"call_{uuid.uuid4().hex[:24]}", "type": "function",
        "function": {"name": call.get("name", ""),
                     "arguments": json.dumps(call.get("arguments", {}))}}]}))
    return 1


def _warm_in_background(state, session, verbose: bool) -> None:
    """Pull this model's experts into the OS page cache after the port is open.

    After, not before: reading 12.68 GB takes about three seconds, and a server that will not
    answer for three seconds to save a fraction of one request has made the wrong trade. By the
    time most people have typed a first message the cache is warm, and if they are faster than
    that they lose nothing -- the reads they need happen either way.
    """
    from . import stream as _st
    try:
        _man, blob = _st.expert_source(session.model_dir)
    except Exception:                       # noqa: BLE001 -- warming is never worth an exception
        return
    if not blob:
        return
    # A BUDGET, BECAUSE WARMING CAUSED THE PRESSURE IT THEN STOPPED FOR.
    #     Reading all 12.68 GB of a model into the page cache on a machine whose whole ceiling
    #     is 9 GB evicts more than it gains: it read 11.71 GB, reported "the machine went short
    #     of memory", and the shrink controller -- watching the same signal -- then reloaded the
    #     model. Warming may only take what is genuinely spare.
    try:
        from .calibrate import available_gb
        spare = max(0.0, available_gb() - getattr(session, "working_memory_gb", 3.0))
    except Exception:                       # noqa: BLE001 -- never worth an exception
        spare = 0.0
    # WHAT IS SPARE, LESS A MARGIN -- NOT HALF OF IT.
    #     Half was the rule when warming had once caused the pressure it then stopped for. The
    #     stop-on-pressure check is what actually protects the machine; halving the budget on top
    #     of it left two thirds of Qwen3.6's experts cold on a 25.8 GB Mac and the server at half
    #     speed (measured 10.5 tok/s cold against 21 warm) until enough requests had faulted them
    #     in one page at a time. The margin is one gigabyte, and the warm yields to any reply.
    budget = int(max(0.0, spare - 1.0) * 1e9)
    if budget < (256 << 20):
        state.warm = {"bytes": 0, "seconds": 0.0,
                      "stopped": "not enough spare memory to be worth warming"}
        return

    def _busy() -> bool:
        with state.count_lock:
            return state.waiting > 0 or bool(state.by_id)

    hot = []
    try:
        with open(_st.usage_path(session.name)) as fh:
            hot = _st.hot_regions(_man, json.load(fh) or {}, budget)
    except (OSError, ValueError):
        hot = []
    res = _st.warm_page_cache(blob, budget_bytes=budget,
                              should_stop=lambda: state.stopping, should_pause=_busy,
                              regions=hot)
    state.warm = res
    if res.get("bytes"):
        state.event("warm", f"warmed {res['bytes'] / 1e9:.2f} GB of experts into the page cache "
                            f"at {res.get('gb_per_s', 0):.2f} GB/s"
                            + (f", stopped: {res['stopped']}" if res.get("stopped") else ""),
                    bytes=int(res["bytes"]))
    if verbose and res.get("bytes"):
        why = f" -- stopped: {res['stopped']}" if res.get("stopped") else ""
        print(f"  [cache] warmed {res['bytes'] / 1e9:.2f} GB of experts into the page cache "
              f"in {res['seconds']}s{why}", flush=True)


def _speed_verdict(state, s: dict, agg: dict) -> dict:
    """The doctor's four words, for the page: FAST, GOOD, USABLE or SLOW, and on what basis.

    The console showed a number and left the person to judge it, while `doctor` had been
    speaking in tiers all along -- so the same machine got a verdict before the download and
    none after. Measured when there is a measurement; expected from the plan when there is not;
    and the basis is reported, because a plan's guess and a median over a hundred replies must
    never look like the same kind of fact.
    """
    from .preflight import speed_tier, tier_for_tok_s
    tps = agg.get("median_tok_s")
    n = int(state.served or 0)
    if tps:
        label, why = tier_for_tok_s(float(tps))
        basis = f"measured: median over {n} repl{'y' if n == 1 else 'ies'} on this Mac"
    elif s.get("streamed") and s.get("n_layers") and s.get("gb_per_slot"):
        # The plan as the session reports it, in the shape the doctor's prediction reads.
        shape = {"n_layers": int(s["n_layers"]), "n_experts": int(s["n_experts"]),
                 "top_k": int(s.get("top_k") or 1),
                 "bytes_per_expert": int(float(s["gb_per_slot"]) * 1e9 / int(s["n_layers"]))}
        plan = {"capacity": int(s.get("planned_from") or s.get("capacity") or 0)}
        label, why = speed_tier(shape, plan, getattr(state, "disk_gbs", None), False)
        basis = "expected from the plan; nothing measured yet"
    else:
        label, why = speed_tier(None, None, None, True)
        basis = "the whole model is in memory"
    # No ceiling hint here: it needs the model's manifest to plan at a higher ceiling honestly,
    # and `bigrig doctor <model>` is where that is computed. The field stays so the page's
    # contract does not change.
    return {"label": label, "why": why, "basis": basis, "hint": ""}


def serve(session, host: str = "127.0.0.1", port: int = 8080, verbose: bool = True,
          batch: int = 1, release_memory: bool = True, reclaim_memory: bool = True,
          warm_cache: bool = True, cors_origins=()):
    import mlx_lm.generate          # noqa: F401  -- create the thread-local stream HERE
    state = _State(session)
    state.bind_host = host.strip("[]").lower()
    state.loopback = state.bind_host in ("127.0.0.1", "localhost", "::1")
    state.origins = allowed_origins(state.bind_host, port, cors_origins)
    state.event("start", f"{session.name} ready at {session.capacity} experts a layer, "
                         f"{session.footprint_gb:.2f} GB resident, "
                         f"{session.budget_gb:.2f} GB ceiling")
    if (release_memory or reclaim_memory) and getattr(session, "streamed", False):
        from . import memctl as _mc
        floor = _mc.floor_for(int(session.top_k or 1), int(session.plan["n_experts"]))
        # Home is where it starts, which is whatever `bigrig knee` measured, or the planner's
        # estimate if nothing has been measured. The controller may borrow from it and give it
        # back; it may never exceed it.
        # HOME IS THE BUDGET THE PLAN WAS MADE FROM, NOT WHAT THE PLAN PRODUCED.
        #     With whole layers in play the reported capacity is the STREAMED one -- 11 where
        #     the budget was 36. Growing back to 11 re-plans into zero whole layers and silently
        #     loses the 1.23x the server started with. `planned_from` is the budget.
        home = int(getattr(session, "planned_from", None)
                   or getattr(session, "capacity", 0) or session.plan["capacity"])
        # GROWING BACK IS ON BY DEFAULT NOW. Measured: one squeeze shrank a live server from 38
        # to 30 experts a layer and it stayed there for the rest of the day, 19 tok/s instead
        # of 21, because nothing ever took the memory back -- a restart was the only recovery.
        # The grow half still stops at `home` and still waits three quiet minutes per step.
        state.memctl = _mc.ShrinkPolicy(floor=floor, grow=bool(reclaim_memory), ceiling=home,
                                        started=state.started)
        threading.Thread(target=state.watch_pressure, name="bigrig-pressure",
                         daemon=True).start()
        if verbose:
            print(f"\n  RELEASING MEMORY UNDER PRESSURE. When this machine runs short, experts "
                  f"are handed\n  back between replies -- never during one -- down to a floor "
                  f"of {floor} a layer.")
            if reclaim_memory:
                print(f"  Once it has been quiet for three minutes it takes them back, in "
                      f"smaller steps,\n  and stops at {home} -- the capacity it started from. "
                      f"It never goes above that\n  (--no-reclaim-memory to only ever give "
                      f"memory back).")
            else:
                print("  It only ever gives memory back (--no-reclaim-memory); nothing is taken "
                      "back once\n  the machine is quiet again.")
    if verbose and getattr(session, "streamed", False) and not getattr(session, "packed", True):
        print(f"\n  EXPERTS ARE READ FROM THE MODEL'S OWN FILES, which the GPU cannot read in "
              f"place.\n  Every expert is copied on the way in. `bigrig prepare {session.name}` "
              f"makes the\n  page-aligned copy the zero-copy path needs.")
        state.event("path", "experts are copied in from the model's own files; "
                            f"`bigrig prepare {session.name}` makes the zero-copy copy")
    if batch and int(batch) > 1:
        from . import batch as _b
        plan = _b.plan_batch(session, int(batch), 512, 512)
        state.batch_size = plan["size"]
        if verbose:
            print(f"\n  BATCHING ON, up to {plan['size']} requests per pass"
                  f"{'' if plan['reason'] == 'requested' else f" (asked for {plan['requested']}, "
                     f"held to {plan['size']} by {plan['reason']})"}")
            print("  Replies will differ from the same request served alone. Batching changes "
                  "the shape of\n  the matmul and so the order of the reduction; in bfloat16 "
                  "that moves logits enough to\n  change a token where the top two are close. "
                  "Use one request per pass if you need\n  reproducibility.")
    # THE HOT SET GOES INTO THE PAGE CACHE BEFORE THE PORT OPENS. Everything else warms in the
    # background afterwards, as before. The split is the point: the most-used tenth of a model's
    # experts carry about half of all expert reads and take half a second to read, so the first
    # request gets them from memory instead of paying the cold-disk price on every one.
    if warm_cache and getattr(session, "streamed", False):
        from . import stream as _stm
        try:
            from .calibrate import available_gb as _avail
            _spare = _avail()
        except Exception:                   # noqa: BLE001 -- never worth an exception
            _spare = None
        hot = _stm.warm_hot_set(session.model_dir, session.name, spare_gb=_spare)
        state.hot_warm = hot
        if hot.get("bytes"):
            state.event("warm", f"read the {hot['experts']} most-used experts "
                                f"({hot['bytes'] / 1e9:.2f} GB) into the page cache in "
                                f"{hot['seconds']:.2f}s before opening the port")
    httpd = ThreadingHTTPServer((host, port), make_handler(state))
    httpd.daemon_threads = True
    threading.Thread(target=httpd.serve_forever, name="bigrig-http", daemon=True).start()
    if warm_cache and getattr(session, "streamed", False):
        threading.Thread(target=_warm_in_background, args=(state, session, verbose),
                         name="bigrig-warm", daemon=True).start()
    if verbose:
        # flush=True is load-bearing, not tidiness. Piped to a file, Python block-buffers stdout,
        # so the banner -- which carries the disclosure of whether the weights were altered --
        # sat unwritten and was lost entirely when the process was killed. A disclosure that only
        # appears if the process exits cleanly is not a disclosure.
        s = session.stats()
        out = [f"\n  bigrig serving {s['model']} on http://{host}:{port}",
               f"  {s['serving']}"]
        if s.get("streamed") and s.get("resident_gb"):
            out.append(f"  {s['resident_gb']:.1f} GB of expert weights held in memory")
        hw = getattr(state, "hot_warm", None) or {}
        if hw.get("bytes"):
            out.append(f"  {hw['bytes'] / 1e9:.1f} GB of the most-used experts read into the page "
                       f"cache before the port opened ({hw['seconds']:.2f}s)")
        out.append(f"  quality monitor: {'on' if s['monitor'] else 'off'}")
        out.append("")
        # The browser link goes FIRST among the things to try. Most people who need this tool
        # are not going to reach for curl, and burying the one option that needs no terminal
        # under two shell snippets is the opposite of the point.
        out.append(f"  open  http://{host}:{port}  in a browser to chat and watch quality")
        out.append(f"  api   {host}:{port}/v1/chat/completions   (OpenAI)")
        out.append(f"        {host}:{port}/v1/messages           (Anthropic -- Claude Code)")
        out.append(f"  agent bigrig launch {s['model']}        (wires Claude Code to this)")
        out.append("")
        print("\n".join(out), flush=True)
    try:
        state.pump()
    except KeyboardInterrupt:
        print("\n  stopping")
    finally:
        state.stopping = True
        httpd.shutdown()
        httpd.server_close()
        session.close()
    return 0
