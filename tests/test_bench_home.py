"""The engine home every harness gives its BigRig workers: a throwaway one, holding the checkout's blobs.

WHY THIS FILE EXISTS
    The harnesses run BigRig from `git archive` exports of bigrig_engine/, which have no
    pyproject.toml beside them, so bigrig_engine.home() fell back to ~/.bigrig. There is no packed
    blob there, so the engine silently read experts from the safetensors shards: a slower path with
    other kernels (found 2026-11-13, when two identical engines scored 2.08x apart). Pointing the
    home at the checkout fixed that and broke something else: each worker's Session restores and
    saves conversations under home/data/sessions and deletes ones it did not write, so evals
    overwrote the developer's saved conversations and each run reloaded the previous run's.

    So every harness except the frozen confirm_turn now runs its workers under
    climb_turn.scratch_home(): the checkout's blobs, results, corpora and traces linked in, an empty
    sessions folder, removed afterwards. It drops BigRig switches inherited from the shell, and
    before any run it asks each arm's own expert_source whether it will read the packed blob
    (climb_turn.engine_reads_blob), which also catches a missing manifest or an outdated blob.

    These tests run each harness's real main() with git, workers and the Mac stubbed, against a
    temporary checkout, and read the environment every worker would have been given. They also
    run the real scratch_home and engine_reads_blob on a fake arm.

Run: .venv/bin/python tests/test_bench_home.py
"""
import contextlib
import importlib.util
import io
import json
import os
import shutil
import signal
import sys
import tempfile
import textwrap
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Every temporary folder this file makes, the harnesses' scratch homes included, goes in a folder of
# its own, so the leak check at the end never sees, or removes, a real harness's scratch home.
tempfile.tempdir = tempfile.mkdtemp(prefix="test_bench_home_")
FAIL = []
MODEL_NAME = "Qwen3.6-35B-A3B-4bit"


def check(n, c, d=""):
    (print(f"  PASS  {n}") if c else (FAIL.append(n), print(f"  FAIL  {n}  {d}")))


def load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "bench", name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def quiet_mac():
    return {"ac_power": True, "pressure": 1, "swap_used_gb": 0.0, "used_gb": 5.0, "cpu_speed_limit": None}


def checkout(parent):
    """A stand-in checkout: data/blobs with the model's blob, data/results with one record."""
    c = os.path.join(parent, "checkout")
    for sub in ("blobs", "results", "sessions"):
        os.makedirs(os.path.join(c, "data", sub))
    open(os.path.join(c, "data", "blobs", MODEL_NAME + ".experts"), "w").close()
    open(os.path.join(c, "data", "results", "sync_curve.json"), "w").close()
    open(os.path.join(c, "data", "sessions", "my-saved-conversation"), "w").close()
    return c


def layout(home):
    """What an engine home looks like right now (taken while a run is using it)."""
    sessions = os.path.join(home or "", "data", "sessions")
    return {"home": home, "blobs": os.path.realpath(os.path.join(home or "", "data", "blobs")),
            "own_sessions": os.path.isdir(sessions) and not os.path.islink(sessions)}


def is_scratch_of(home_or_layout, c):
    """A scratch home for checkout `c`: its own sessions folder, the checkout's blobs linked in."""
    lay = home_or_layout if isinstance(home_or_layout, dict) else layout(home_or_layout)
    return (bool(lay["home"]) and lay["home"] != c and lay["own_sessions"]
            and lay["blobs"] == os.path.realpath(os.path.join(c, "data", "blobs")))


class FakeProc:
    """A worker that exits at once with an error: the harness records it and moves on."""
    pid, returncode = 0, 1

    def poll(self):
        return 1

    def wait(self, timeout=None):
        return 1


def fake_subprocess(envs: list, llama_version: str = ""):
    """A stand-in for a harness's subprocess module that records every worker launch's env.
    git archive / tar / rev-parse calls succeed without doing anything."""
    def run(cmd, **kw):
        if "env" in kw:
            envs.append(dict(kw["env"]))
        if cmd and cmd[0].endswith("llama-server"):
            out = llama_version
        else:
            out = b"" if kw.get("capture_output") and not kw.get("text") else "tree"
        return types.SimpleNamespace(returncode=1 if "env" in kw else 0, stdout=out)

    def popen(cmd, **kw):
        envs.append(dict(kw.get("env") or {}))
        return FakeProc()

    return types.SimpleNamespace(run=run, Popen=popen, PIPE=-1, STDOUT=-2, DEVNULL=-3,
                                 TimeoutExpired=TimeoutError)


def bigrig_keys(env):
    return {k: v for k, v in env.items() if k.startswith("BIGRIG_")}


def developer_shell():
    """Settings a developer's shell might carry: an engine home elsewhere and a BigRig switch."""
    os.environ.update(BIGRIG_HOME="/somewhere/else", BIGRIG_SPLIT_READ="1", LLAMA_ARG_THREADS="2")
    for k in ("LAB_RESULTS", "LAB_RUN_DIR"):
        os.environ.pop(k, None)


ct = load("climb_turn")

print("=" * 78)
print("0. scratch_home AND engine_reads_blob, FOR REAL")
print("=" * 78)
with tempfile.TemporaryDirectory() as tmp:
    c = checkout(tmp)
    h = ct.scratch_home(c)
    check("a scratch home links the checkout's blobs and results and has its own empty sessions",
          is_scratch_of(h, c) and os.listdir(os.path.join(h, "data", "sessions")) == []
          and os.path.exists(os.path.join(h, "data", "results", "sync_curve.json")))
    try:
        open(os.path.join(h, "data", "sessions", "written-by-a-run"), "w").close()
        writable = True
    except PermissionError:
        writable = False
    check("its sessions folder is read-only, so no run restores or saves a conversation", not writable)
    shutil.rmtree(h)
    check("removing it leaves the checkout's blobs, results and saved conversations alone",
          not os.path.exists(h) and sorted(os.listdir(os.path.join(c, "data", "sessions"))) == ["my-saved-conversation"]
          and os.path.exists(os.path.join(c, "data", "blobs", MODEL_NAME + ".experts"))
          and os.path.exists(os.path.join(c, "data", "results", "sync_curve.json")))
    # A fake arm whose expert_source, like the real one, looks for the blob under its home.
    arm = os.path.join(tmp, "arm")
    os.makedirs(os.path.join(arm, "bigrig_engine"))
    open(os.path.join(arm, "bigrig_engine", "__init__.py"), "w").close()
    with open(os.path.join(arm, "bigrig_engine", "stream.py"), "w") as f:
        f.write(textwrap.dedent('''
            import os
            def expert_source(model_dir):
                p = os.path.join(os.environ.get("BIGRIG_HOME", "~/.bigrig"), "data", "blobs",
                                 os.path.basename(model_dir) + ".experts")
                return {}, (p if os.path.exists(p) else "")
        '''))
    h = ct.scratch_home(c)
    model = os.path.join(tmp, "models", MODEL_NAME)
    check("engine_reads_blob: yes when the arm's own expert_source finds the blob under the env's home",
          ct.engine_reads_blob(arm, {**os.environ, "BIGRIG_HOME": h}, model))
    check("...no under a home without it (the shards path)",
          not ct.engine_reads_blob(arm, {**os.environ, "BIGRIG_HOME": os.path.join(tmp, "empty")}, model))
    with open(os.path.join(arm, "bigrig_engine", "stream.py"), "a") as f:
        f.write("\nraise RuntimeError('broken engine')\n")
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        broken = ct.engine_reads_blob(arm, {**os.environ, "BIGRIG_HOME": h}, model)
    check("...and no when the arm's engine cannot even be imported, saying why rather than 'no blob'",
          not broken and "broken engine" in err.getvalue(), err.getvalue()[-200:])
    shutil.rmtree(h)

print("=" * 78)
print("1. bench/climb_turn.py main() (the locked eval)")
print("=" * 78)
developer_shell()
seen, asked, exported = [], [], []
ct.wait_quiet = lambda *a, **k: quiet_mac()
ct.export_tree = lambda commit, dest: (exported.append(commit), dest)[1]
ct.sh = lambda cmd, **kw: "commit\n"
ct.run_eval = lambda base, cand: (seen.append({**os.environ, "_layout": layout(os.environ.get("BIGRIG_HOME"))}),
                                  {"speedup": 1.0})[1]
answer = [False]
ct.engine_reads_blob = lambda arm, env, model=None: (asked.append((arm, env.get("BIGRIG_HOME"))), answer[0])[1]
with tempfile.TemporaryDirectory() as tmp:
    c = checkout(tmp)
    ct.ROOT = c
    out = os.path.join(tmp, "res.json")
    sys.argv = ["climb_turn.py", "--out", out]
    try:
        ct.main()
        refused = False
    except SystemExit as e:
        refused = "would not read the packed blob" in str(e)
    homes = {hm for _, hm in asked}
    check("when an arm's engine would not read the blob, it stops before any run", refused and not seen,
          (refused, len(seen)))
    check("...and removes its scratch home", all(not os.path.exists(hm) for hm in homes), homes)
    asked.clear()
    answer[0] = True
    developer_shell()
    ct.main()
    env = seen[-1]
    check("both arms were asked, the base export and the checkout, under one home",
          len(asked) == 2 and os.path.basename(asked[0][0]) == "base" and asked[1][0] == c
          and asked[0][1] == asked[1][1], asked)
    check("the workers' engine home is a scratch home holding the checkout's blobs, not the checkout",
          is_scratch_of(env["_layout"], c), env["_layout"])
    check("...the same one the arms were asked about", env.get("BIGRIG_HOME") == asked[0][1])
    check("...and no BigRig switch from the developer's shell reaches them",
          set(bigrig_keys(env)) == {"BIGRIG_HOME"}, bigrig_keys(env))
    check("the scratch home is removed after the eval, the checkout's saved conversations untouched",
          not os.path.exists(env["BIGRIG_HOME"])
          and os.listdir(os.path.join(c, "data", "sessions")) == ["my-saved-conversation"])
developer_shell()

print("=" * 78)
print("2. bench/confirm_decode.py")
print("=" * 78)
cd = load("confirm_decode")
envs, exported, asked = [], [], []
cd.subprocess = fake_subprocess(envs)
cd.mac_state = quiet_mac
cd.export_arms = lambda dest: (exported.append(dest),
                               {a: {"commit": k, "engine_tree": "t", "dir": os.path.join(dest, a)}
                                for a, k in cd.ARMS.items()})[1]
answer = [True]
cd.ct.engine_reads_blob = lambda arm, env, model=None: (asked.append((arm, env.get("BIGRIG_HOME"))), answer[0])[1]
registered, handlers = [], []
cd.atexit = types.SimpleNamespace(register=lambda f, *a, **k: registered.append((f, a)))
cd.signal = types.SimpleNamespace(signal=lambda sig, fn: handlers.append((sig, fn)), SIGTERM=signal.SIGTERM)
with tempfile.TemporaryDirectory() as tmp:
    c = checkout(tmp)
    cd.ROOT = c
    out = os.path.join(tmp, "res.json")
    os.environ["LAB_RUN_DIR"] = tmp
    sys.argv = ["confirm_decode.py", out]
    try:
        cd.main()
    except SystemExit:
        pass                                          # every fake worker fails: the harness stops
    res = json.load(open(out))
    check("main ran workers until an arm failed twice", "failed twice" in res.get("stopped_reason", ""),
          res.get("stopped_reason"))
    check("every arm was asked whether its engine reads the blob, under the workers' home",
          sorted(os.path.basename(a) for a, _ in asked) == sorted(cd.ARMS)
          and {hm for _, hm in asked} == {envs[0].get("BIGRIG_HOME")}, asked)
    check("every worker got a scratch home holding the checkout's blobs, and no other BigRig switch",
          len(envs) >= 2 and all(is_scratch_of(e.get("BIGRIG_HOME"), c) and set(bigrig_keys(e)) == {"BIGRIG_HOME"}
                                 for e in envs), [bigrig_keys(e) for e in envs])
    check("its removal is registered for exit, and a SIGTERM exits through it",
          registered[:1] == [(shutil.rmtree, (envs[0]["BIGRIG_HOME"],))] and handlers[0][0] == signal.SIGTERM,
          (registered, handlers))
    try:
        handlers[0][1](signal.SIGTERM, None)
        exited = None
    except SystemExit as e:
        exited = e.code
    check("...(the handler exits with 143)", exited == 143, exited)
    envs.clear()
    asked.clear()
    answer[0] = False
    try:
        cd.main()
        refused = False
    except SystemExit:
        refused = "would not read the packed blob" in json.load(open(out)).get("stopped_reason", "")
    check("when an arm's engine would not read the blob, it stops before starting a worker",
          refused and not envs and len(asked) == 1, (refused, len(envs), len(asked)))
    os.environ.pop("LAB_RUN_DIR")
    for _, a in registered:
        shutil.rmtree(a[0], ignore_errors=True)

print("=" * 78)
print("3. bench/race_llamacpp.py (smoke mode: one BigRig run, one llama.cpp run)")
print("=" * 78)
developer_shell()
ra = load("race_llamacpp")
env = ra.worker_env("/a/scratch/home")
check("worker_env: the given home, offline, no BigRig or llama.cpp switch inherited",
      bigrig_keys(env) == {"BIGRIG_HOME": "/a/scratch/home"} and env.get("HF_HUB_OFFLINE") == "1"
      and not any(k.startswith("LLAMA_") for k in env), bigrig_keys(env))
envs, asked = [], []
ra.subprocess = fake_subprocess(envs, llama_version=ra.LLAMA_VERSION)
ra.mac_state = quiet_mac
answer = [True]
ra.ct.engine_reads_blob = lambda arm, env, model=None: (asked.append((arm, env.get("BIGRIG_HOME"))), answer[0])[1]
registered, handlers = [], []
ra.atexit = types.SimpleNamespace(register=lambda f, *a, **k: registered.append((f, a)))
ra.signal = types.SimpleNamespace(signal=lambda sig, fn: handlers.append((sig, fn)), SIGTERM=signal.SIGTERM,
                                  SIGKILL=signal.SIGKILL)
with tempfile.TemporaryDirectory() as tmp:
    c = checkout(tmp)
    ra.ROOT = c
    gguf = os.path.join(tmp, "model.gguf")
    with open(gguf, "wb") as f:
        f.write(b"\0" * 64)
    ra.GGUF, ra.GGUF_BYTES = gguf, 64
    os.makedirs(os.path.join(tmp, "one"))
    sys.argv = ["race_llamacpp.py", "--smoke", "--out", os.path.join(tmp, "one", "res.json")]
    ra.main()
    res = json.load(open(os.path.join(tmp, "one", "res.json")))
    check("smoke ran one BigRig and one llama.cpp worker", len(res.get("smoke", [])) == 2, res.get("smoke"))
    check("the BigRig arm was asked whether it reads the blob, under the workers' home",
          len(asked) == 1 and asked[0][1] == envs[0].get("BIGRIG_HOME"), asked)
    check("both workers got a scratch home holding the checkout's blobs, and no other BigRig switch",
          len(envs) == 2 and all(is_scratch_of(e.get("BIGRIG_HOME"), c) and set(bigrig_keys(e)) == {"BIGRIG_HOME"}
                                 for e in envs), [bigrig_keys(e) for e in envs])
    check("its removal is registered for exit, and a SIGTERM exits through it",
          registered[:1] == [(shutil.rmtree, (envs[0]["BIGRIG_HOME"],))] and handlers[0][0] == signal.SIGTERM,
          (registered, handlers))
    envs.clear()
    asked.clear()
    answer[0] = False
    os.makedirs(os.path.join(tmp, "two"))
    sys.argv = ["race_llamacpp.py", "--smoke", "--out", os.path.join(tmp, "two", "res.json")]
    try:
        ra.main()
        refused = False
    except SystemExit:
        refused = "would not read the packed blob" in json.load(open(os.path.join(tmp, "two", "res.json"))).get("stopped_reason", "")
    check("when the BigRig arm would not read the blob, it stops before starting a worker",
          refused and not envs and len(asked) == 1, (refused, len(envs)))
    for _, a in registered:
        shutil.rmtree(a[0], ignore_errors=True)

print("=" * 78)
print("4. bench/same_answers.py main(): --base, and the engine home")
print("=" * 78)
developer_shell()
sa = load("same_answers")
seen, asked, exported = [], [], []
sa.ct.wait_quiet = lambda *a, **k: quiet_mac()
sa.ct.export_tree = lambda commit, dest: (exported.append(commit), dest)[1]
sa.ct.sh = lambda cmd, **kw: ("\n" if cmd[-1].startswith("bad-ref")
                              else "sha-of-" + cmd[-1].removesuffix("^{commit}") + "\n")
sa.ct.engine_reads_blob = lambda arm, env, model=None: (asked.append((arm, env.get("BIGRIG_HOME"), model)), True)[1]
sa.run_job = lambda base, cand: (seen.append({**os.environ, "_layout": layout(os.environ.get("BIGRIG_HOME"))}),
                                 {"decision": "ships", "cells_compared": 0, "cells_differing": 0, "healthy": 1, "speed": {}})[1]
with tempfile.TemporaryDirectory() as tmp:
    c = checkout(tmp)
    sa.ROOT = c
    out = os.path.join(tmp, "res.json")
    sys.argv = ["same_answers.py", "--out", out, "--candidate", "cand-ref", "--base", "base-ref"]
    sa.main()
    res = json.load(open(out))
    env = seen[-1]
    check("--base names the base arm, which is exported and recorded with its tree",
          exported == ["sha-of-base-ref", "sha-of-cand-ref"] and res["base_commit"] == "sha-of-base-ref"
          and res["base_tree"] == "sha-of-sha-of-base-ref:bigrig_engine", (exported, res.get("base_commit")))
    check("both arms were asked about both models' blobs before any run",
          len(asked) == 4 and {m for _, _, m in asked} == {m["dir"] for m in sa.MODELS.values()}, asked)
    check("the workers' engine home is a scratch home holding the checkout's blobs, no other switch",
          is_scratch_of(env["_layout"], c) and set(bigrig_keys(env)) == {"BIGRIG_HOME"}, bigrig_keys(env))
    check("...removed after the job", not os.path.exists(env["BIGRIG_HOME"]))
    exported.clear()
    sys.argv = ["same_answers.py", "--out", out, "--candidate", "cand-ref"]
    sa.main()
    check("without --base it compares with the released engine, as before",
          exported[0] == "sha-of-" + sa.BASE_COMMIT, exported)
    exported.clear()
    sys.argv = ["same_answers.py", "--out", out, "--candidate", "bad-ref-typo"]
    try:
        sa.main()
        refused = False
    except SystemExit as e:
        refused = "not a commit: bad-ref-typo" in str(e)
    check("a ref that is not a commit is refused before anything is exported", refused and not exported, exported)

print("=" * 78)
print("5. NOTHING LEFT BEHIND")
print("=" * 78)
left = sorted(n for n in os.listdir(tempfile.tempdir) if n.startswith("bigrig_home_"))
check("every scratch home made in this file was removed (refusals and kills included)", not left, left)
shutil.rmtree(tempfile.tempdir, ignore_errors=True)

print("\n" + "=" * 78)
print(f"{'ALL TESTS PASSED' if not FAIL else str(len(FAIL)) + ' FAILURES: ' + ', '.join(FAIL)}")
print("=" * 78)
sys.exit(1 if FAIL else 0)
