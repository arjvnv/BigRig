"""Point a coding agent at a model running on this Mac, in one command.

THE STEP THIS REMOVES
    Without it, a user has to: start the server, find out which port it chose, work out which
    environment variable their agent reads, set it correctly, and remember to unset it later.
    Every one of those is a place to give up, and none of them is interesting.

    `bigrig launch <model>` starts the server, waits for it to answer, sets the variables in
    the agent's environment only, and runs it. When the agent exits, the server is stopped.

WHY IT DOES NOT INSTALL THE AGENT FOR YOU
    FreeToken's equivalent pipes a URL into a shell -- `curl -fsSL ... | bash` -- to install
    Codex, Claude Code or OpenCode. That is a lot of trust to ask for from a tool whose job is
    running a model, and it is trust the user never explicitly granted. If the agent is missing
    this prints the one command to install it and stops. One extra step, no arbitrary code.

WHY ENVIRONMENT VARIABLES AND NOT CONFIG FILES
    Editing `~/.codex/config.toml` or the Claude Code settings changes state that outlives the
    command and that the user did not ask to have changed. The variables here are set on the
    child process only: nothing on disk is touched, and closing the agent restores exactly the
    setup they had.
"""
from __future__ import annotations

import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request


class Agent:
    """How to recognise one coding agent and how to point it at a local server."""

    def __init__(self, name, binary, env, install_hint, note="", needs_config=False):
        self.name, self.binary, self.env = name, binary, env
        self.install_hint, self.note = install_hint, note
        # Some agents cannot be pointed at a server by environment alone. See `_codex_home`.
        self.needs_config = needs_config

    def environment(self, base: str) -> dict:
        return {k: v.format(base=base) for k, v in self.env.items()}


AGENTS = {
    # Claude Code speaks the Anthropic Messages API, which is why /v1/messages exists.
    # ANTHROPIC_AUTH_TOKEN must be set to something or the client refuses to start; the value is
    # never checked, because a server bound to localhost serving weights off the user's own disk
    # has nothing to authenticate.
    "claude": Agent(
        "claude", "claude",
        {"ANTHROPIC_BASE_URL": "{base}", "ANTHROPIC_AUTH_TOKEN": "bigrig-local"},
        "https://claude.ai/install.sh  (or: npm i -g @anthropic-ai/claude-code)",
        "uses the Anthropic Messages API"),
    "codex": Agent(
        "codex", "codex",
        {"OPENAI_API_KEY": "bigrig-local"},
        "npm i -g @openai/codex",
        "uses the Responses API", needs_config=True),
    "opencode": Agent(
        "opencode", "opencode",
        {"OPENAI_BASE_URL": "{base}/v1", "OPENAI_API_KEY": "bigrig-local"},
        "https://opencode.ai/install",
        "uses the OpenAI API"),
}


def free_port(preferred: int = 0) -> int:
    """A port nothing else is on. 0 asks the OS to choose."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", preferred))
        return s.getsockname()[1]


def port_is_serving(port: int, timeout: float = 1.0) -> bool:
    """Is OUR server on this port -- not merely something that answers 200.

    `--port N` plus the default reuse meant that any local process answering /health was
    treated as a bigrig server, and the coding agent was pointed at it. An agent's prompts are
    the user's source code, so handing them to an unknown process is the one mistake here worth
    a round trip to prevent. /health names itself; anything that does not is not us.
    """
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=timeout) as r:
            return json.loads(r.read() or b"{}").get("server") == "bigrig"
    except (urllib.error.URLError, OSError, ValueError):
        return False


def wait_until_ready(port: int, proc, deadline_s: float = 900.0, poll: float = 0.5,
                     on_wait=None) -> bool:
    """Block until the server answers, or it dies, or we give up.

    Watching the process as well as the port matters: a server that failed to start would
    otherwise leave this spinning for the whole deadline against a port that will never open.
    """
    t0 = time.time()
    told = False
    while time.time() - t0 < deadline_s:
        if proc is not None and proc.poll() is not None:
            return False
        if port_is_serving(port):
            return True
        if on_wait and not told and time.time() - t0 > 3:
            on_wait()
            told = True
        time.sleep(poll)
    return False


def run(agent_name: str, model: str, serve_argv: list, port: int = 0,
        agent_args=(), reuse: bool = True, writer=print) -> int:
    """Start a server if needed, point the agent at it, run the agent, then clean up."""
    agent = AGENTS.get(agent_name)
    if agent is None:
        raise ValueError(
            f"unknown agent {agent_name!r}. Known: {', '.join(sorted(AGENTS))}")
    exe = shutil.which(agent.binary)
    if not exe:
        raise FileNotFoundError(
            f"{agent.name} is not installed, or not on PATH.\n"
            f"    Install it:  {agent.install_hint}\n"
            f"    Then re-run this command. (bigrig will not install it for you -- running an "
            f"installer from the internet is not something a model runner should do on your "
            f"behalf.)")

    started = None
    if port and reuse and port_is_serving(port):
        writer(f"  reusing the server already on port {port}")
    else:
        port = port or free_port()
        writer(f"  starting {model} on port {port} ...")
        started = subprocess.Popen(serve_argv + ["--port", str(port)],
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        # SHOW THE SERVER'S OWN STARTUP LINES UNTIL IT ANSWERS. With stdout swallowed, the
        # first run of a streamed model -- which now spends a minute or two measuring its
        # fastest setting -- was two minutes of silence after "starting ...", and a person
        # reasonably concludes it hung. The server already says what it is doing; this lets
        # them read it. Forwarding stops the moment the port answers, so the agent's own
        # interface is not interleaved with request logs afterwards; the thread keeps
        # draining so the pipe never fills and stalls the server.
        tail, forwarding = [], [True]

        def _relay():
            for line in started.stdout or ():
                tail.append(line)
                del tail[:-40]
                if forwarding[0] and line.strip():
                    writer("  " + line.rstrip())
        relay = threading.Thread(target=_relay, daemon=True)
        relay.start()
        ok = wait_until_ready(port, started)
        forwarding[0] = False
        if not ok:
            _stop(started)
            relay.join(timeout=2.0)
            out = "".join(tail)[-1200:]
            raise RuntimeError(
                f"the server did not come up on port {port}.\n{out}"
                if out else f"the server did not come up on port {port}")

    base = f"http://127.0.0.1:{port}"
    env = dict(os.environ)
    env.update(agent.environment(base))
    tmp_home = None
    if getattr(agent, "needs_config", False):
        tmp_home = _codex_home(base)
        env["CODEX_HOME"] = tmp_home
    writer(f"  {agent.name} -> {base}  ({agent.note})")
    writer(f"  set for this process only: {', '.join(sorted(agent.environment(base)))}")
    writer("")
    try:
        return subprocess.call([exe, *agent_args], env=env)
    except KeyboardInterrupt:
        return 130
    finally:
        if tmp_home:
            shutil.rmtree(tmp_home, ignore_errors=True)
        if started is not None:
            writer("\n  stopping the server")
            _stop(started)


def _codex_home(base: str) -> str:
    """A throwaway CODEX_HOME holding one provider definition. Returns the directory.

    WHY A FILE AT ALL, GIVEN THIS MODULE'S RULE AGAINST THEM.
        Codex cannot be pointed at a server by environment alone. Its wire protocol is chosen by
        `wire_api` in config.toml, and since 0.152.0 the only accepted value is "responses" --
        `wire_api = "chat"` is no longer supported, in the binary's own words. OPENAI_BASE_URL
        alone therefore aims it at this server using an API it will not speak, which is why
        `bigrig launch codex` had been quietly dead.

        The rule that matters is not "never write a file", it is "never change state the user did
        not ask to have changed". CODEX_HOME moves the ENTIRE config directory, so this writes to
        a temporary one and deletes it when the agent exits. `~/.codex` is never opened, never
        read, and never written -- a user with their own Codex setup still has exactly it
        afterwards, and one without still has none.
    """
    import tempfile
    home = tempfile.mkdtemp(prefix="bigrig-codex-")
    with open(os.path.join(home, "config.toml"), "w", encoding="utf-8") as f:
        f.write(
            'model = "bigrig"\n'
            'model_provider = "bigrig"\n'
            '\n[model_providers.bigrig]\n'
            'name = "BigRig"\n'
            f'base_url = "{base}/v1"\n'
            'wire_api = "responses"\n'
            'requires_openai_auth = false\n')
    return home


def _stop(proc) -> None:
    if proc is None or proc.poll() is not None:
        return
    try:
        proc.send_signal(signal.SIGINT)
        proc.wait(timeout=10)
    except (subprocess.TimeoutExpired, ProcessLookupError, OSError):
        try:
            proc.kill()
            proc.wait(timeout=5)
        except Exception:
            pass
