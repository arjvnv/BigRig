"""climb_turn, confirm_decode and race_llamacpp: every BigRig worker runs with the checkout as its engine home.

WHY THIS FILE EXISTS
    Both harnesses run BigRig from a `git archive` export of bigrig_engine/, which has no
    pyproject.toml beside it, so bigrig_engine.home() falls back to ~/.bigrig. There is no packed
    blob there, so the engine silently read experts from the safetensors shards: a slower path with
    other kernels that users never run (found 2026-11-13, when two identical engines scored 2.08x
    apart). climb_turn pins BIGRIG_HOME in main(); the other two build their workers' environment
    in worker_env(), which drops inherited BigRig switches and pins the home. All three refuse to
    start without the blob. These tests run each harness's real main() with its workers and the
    Mac stubbed, and read the environment every worker would have been given.
    (confirm_turn and same_answers pin the home themselves; their own tests cover it.)

Run: .venv/bin/python tests/test_bench_home.py
"""
import importlib.util
import json
import os
import sys
import tempfile
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAIL = []


def check(n, c, d=""):
    (print(f"  PASS  {n}") if c else (FAIL.append(n), print(f"  FAIL  {n}  {d}")))


def load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "bench", name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def quiet_mac():
    return {"ac_power": True, "pressure": 1, "swap_used_gb": 0.0, "used_gb": 5.0, "cpu_speed_limit": None}


class FakeProc:
    """A worker that exits at once with an error: the harness records it and moves on."""
    pid, returncode = 0, 1

    def poll(self):
        return 1

    def wait(self, timeout=None):
        return 1


def fake_subprocess(envs: list, llama_version: str = ""):
    """A stand-in for the subprocess module that records the env of every worker launch."""
    def run(cmd, **kw):
        if "env" in kw:
            envs.append(kw["env"])
        out = llama_version if cmd and cmd[0].endswith("llama-server") else (b"" if kw.get("capture_output") and not kw.get("text") else "tree")
        return types.SimpleNamespace(returncode=1 if "env" in kw else 0, stdout=out)

    def popen(cmd, **kw):
        envs.append(kw.get("env"))
        return FakeProc()

    return types.SimpleNamespace(run=run, Popen=popen, PIPE=-1, STDOUT=-2, DEVNULL=-3,
                                 TimeoutExpired=TimeoutError)


# Settings a developer's shell might carry: an engine home elsewhere and a BigRig switch.
os.environ.update(BIGRIG_HOME="/somewhere/else", BIGRIG_SPLIT_READ="1", LLAMA_ARG_THREADS="2")
for k in ("LAB_RESULTS", "LAB_RUN_DIR"):
    os.environ.pop(k, None)


def bigrig_keys(env):
    return {k: v for k, v in env.items() if k.startswith("BIGRIG_")}


print("=" * 78)
print("0. bench/climb_turn.py (locked eval)")
print("=" * 78)
ct = load("climb_turn")
seen = []
ct.wait_quiet = lambda *a, **k: quiet_mac()
ct.export_tree = lambda commit, dest: dest
ct.sh = lambda cmd, **kw: "commit\n"
ct.run_eval = lambda base, cand: (seen.append(os.environ.get("BIGRIG_HOME")), {"speedup": 1.0})[1]
with tempfile.TemporaryDirectory() as tmp:
    checkout = os.path.join(tmp, "checkout")
    os.makedirs(os.path.join(checkout, "data", "blobs"))
    ct.ROOT = checkout
    out = os.path.join(tmp, "res.json")
    sys.argv = ["climb_turn.py", "--out", out]
    try:
        ct.main()
        refused = False
    except SystemExit as e:
        refused = "no packed blob" in str(e)
    check("without the packed blob it refuses before running a worker", refused and not seen, (refused, seen))
    open(os.path.join(checkout, "data", "blobs", os.path.basename(ct.MODEL) + ".experts"), "w").close()
    ct.main()
    check("both arms' workers see the checkout as their engine home", seen == [checkout], seen)
os.environ["BIGRIG_HOME"] = "/somewhere/else"        # main() set it; restore the developer's shell

print("=" * 78)
print("1. bench/confirm_decode.py")
print("=" * 78)
cd = load("confirm_decode")
check("worker_env: engine home is the checkout, no other BigRig switch inherited",
      bigrig_keys(cd.worker_env()) == {"BIGRIG_HOME": ROOT}, bigrig_keys(cd.worker_env()))
envs, exported = [], []
cd.subprocess = fake_subprocess(envs)
cd.mac_state = quiet_mac
cd.export_arms = lambda dest: (exported.append(dest),
                               {a: {"commit": c, "engine_tree": "t", "dir": os.path.join(dest, a)}
                                for a, c in cd.ARMS.items()})[1]
with tempfile.TemporaryDirectory() as tmp:
    out = os.path.join(tmp, "res.json")
    os.environ["LAB_RUN_DIR"] = tmp
    sys.argv = ["confirm_decode.py", out]
    cd.BLOB = os.path.abspath(__file__)              # stands in for the packed blob
    try:
        cd.main()
    except SystemExit:
        pass                                          # every fake worker fails: the harness stops
    res = json.load(open(out))
    check("main ran workers until an arm failed twice", "failed twice" in res.get("stopped_reason", ""), res.get("stopped_reason"))
    check("every worker main() launched got the checkout as its engine home and no other switch",
          len(envs) >= 2 and all(bigrig_keys(e) == {"BIGRIG_HOME": ROOT} for e in envs),
          [bigrig_keys(e) for e in envs])
    envs.clear()
    exported.clear()
    cd.BLOB = os.path.join(tmp, "missing.experts")
    try:
        cd.main()
        refused = False
    except SystemExit:
        refused = "no packed blob" in json.load(open(out)).get("stopped_reason", "")
    check("without the packed blob it stops before exporting an arm or starting a worker",
          refused and not exported and not envs, (refused, exported, len(envs)))
    os.environ.pop("LAB_RUN_DIR")

print("=" * 78)
print("2. bench/race_llamacpp.py (smoke mode: one BigRig run, one llama.cpp run)")
print("=" * 78)
ra = load("race_llamacpp")
env = ra.worker_env()
check("worker_env: engine home is the checkout, offline, no BigRig or llama.cpp switch inherited",
      bigrig_keys(env) == {"BIGRIG_HOME": ROOT} and env.get("HF_HUB_OFFLINE") == "1"
      and not any(k.startswith("LLAMA_") for k in env), bigrig_keys(env))
envs = []
ra.subprocess = fake_subprocess(envs, llama_version=ra.LLAMA_VERSION)
ra.mac_state = quiet_mac
with tempfile.TemporaryDirectory() as tmp:
    gguf = os.path.join(tmp, "model.gguf")
    with open(gguf, "wb") as f:
        f.write(b"\0" * 64)
    ra.GGUF, ra.GGUF_BYTES = gguf, 64
    ra.BLOB = os.path.abspath(__file__)
    out = os.path.join(tmp, "res.json")
    sys.argv = ["race_llamacpp.py", "--smoke", "--out", out]
    ra.main()
    res = json.load(open(out))
    bigrig_env = [e for e in envs if e is not None]
    check("smoke ran one BigRig and one llama.cpp worker", len(res.get("smoke", [])) == 2, res.get("smoke"))
    check("both workers got the checkout as BigRig's engine home and no other switch",
          len(bigrig_env) == 2 and all(bigrig_keys(e) == {"BIGRIG_HOME": ROOT} for e in bigrig_env),
          [bigrig_keys(e) for e in bigrig_env])
    envs.clear()
    ra.BLOB = os.path.join(tmp, "missing.experts")
    os.makedirs(os.path.join(tmp, "again"))
    sys.argv = ["race_llamacpp.py", "--smoke", "--out", os.path.join(tmp, "again", "res.json")]
    try:
        ra.main()
        refused = False
    except SystemExit:
        refused = "no packed blob" in json.load(open(os.path.join(tmp, "again", "res.json"))).get("stopped_reason", "")
    check("without the packed blob it stops before starting a worker", refused and not envs, (refused, len(envs)))

print("\n" + "=" * 78)
print(f"{'ALL TESTS PASSED' if not FAIL else str(len(FAIL)) + ' FAILURES: ' + ', '.join(FAIL)}")
print("=" * 78)
sys.exit(1 if FAIL else 0)
