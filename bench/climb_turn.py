"""Locked eval for climb 2: how long one realistic turn takes on a 16 GB Mac, against released BigRig.

WHAT ONE EVAL DOES
    Runs the released engine (BASE_COMMIT, exported from git) and the candidate (this checkout's
    bigrig_engine/) alternately, base / candidate / candidate / base, each in a fresh process with
    the same fixed plan, budget and balloon. Every run answers the same workload after one
    discarded warm-up reply:

        long    "Summarize the following document in two sentences." + README.md at 626c47a
                (first 6,000 characters, 1,771 tokens), 128 tokens out
        short   the three prompts of the locked bench/climb_decode.py, 128 tokens out each

    Greedy, seed 0, thinking off. turn_s for a run is the wall-clock time from each call to its
    last token, summed over the four prompts: time to first token and decoding both count.

THE METRIC (estimator_id below)
    speedup = exp(mean over the two pairs of ln(base turn_s / candidate turn_s)). Pair 1 is runs
    1 and 2, pair 2 is runs 3 and 4, so drift (heat, page cache, other apps) lands on both arms
    and the first-run position alternates. 1.00 means as fast as released BigRig; 2.00 means a
    turn takes half the time.

GATES (results keys the climb config gates on)
    exact               every candidate reply has the same text and token count as the base reply
                        in its pair. Base replies must also match across the two pairs.
    healthy             every reply has its keyword, at least 64 tokens and does not loop.
    plan_ok             every run held the pinned plan and loaded bigrig_engine only from its arm.
    balloon_kept        every balloon ended at >= BALLOON_KEEP of its start.
    decode_ratio        candidate over base decode tok/s (geometric mean over pairs): the climb
                        may not buy a faster first token with slower writing.
    peak_over_base_gb   largest (candidate - base) peak physical footprint within a pair.
    swap_growth_gb      swap at the end minus swap at the start of the eval.

WHY THE MEASUREMENT CODE LIVES HERE
    The climb may edit every file in bigrig_engine/. Memory, balloon and Mac-state readings are
    therefore implemented in this locked file and run in the parent process, which never imports
    the candidate's code. Reply timing runs inside the worker, next to the engine, as in
    climb_decode.py; the parent's own wall clock for each run is recorded as a cross-check.

WHY A FIXED PLAN AND A BALLOON
    BigRig picks its expert plan from the memory free at start, and its reply depends on the plan
    (2026-11-12: 11 slots + 3 whole layers left the 8-slot reply at about token 15). Pinning
    capacity 8 with no whole layers makes base and candidate comparable token for token whatever
    else is open. The 5 GB balloon holds back page cache so that streamed reads cost what they
    would on a 16 GB Mac (card warm-ahead-confirm-4 explains why 5 GB and not 8).
"""
import argparse
import ctypes
import ctypes.util
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = os.path.join(ROOT, "models", "Qwen3.6-35B-A3B-4bit")
ESTIMATOR_ID = "turn-speedup-2pairs-v1"
BASE_COMMIT = "6e87fcd4030a4f79493f8c4156fa5f3b098e9ae6"   # released engine (tree 860744d4 = main 626c47a)
README_COMMIT = "626c47a"
BUDGET_GB = 6.1
PIN = {"capacity": 8, "full_layers": 0}
BALLOON_GB = 5.0
BALLOON_KEEP = 0.9
MAX_TOKENS = 128
ORDER = ("base", "cand", "cand", "base")
RUN_TIMEOUT_S = 600
QUIET_USED_GB = 11.0          # memory in use before the eval may start (balloon + engine come on top)
QUIET_WAIT_S = 240          # under lab run's 300 s zero-CPU stall limit; the climb loop waits for quiet before each step
SWAP_KILL_GB = 1.5            # a run is killed if swap grows this much during it
WARMUP = "Name three rivers in Europe."
LONG_PREFIX = "Summarize the following document in two sentences.\n\n"
LONG_CHARS = 6000
SHORT_PROMPTS = (
    ("Explain how a heat pump moves warmth from cold outdoor air into a house.", "heat"),
    ("Write a Python function that returns the n-th Fibonacci number iteratively, with a docstring.",
     "def fib"),
    ("A train leaves at 9:40 and arrives at 13:15. How long is the journey? Show your working.", "9:40"),
)
LONG_KEYWORD = "BigRig"
MIN_TOKENS = {"long": 16, "short": 64}   # a two-sentence summary may end well before 128 tokens


# ---------------------------------------------------------------- measurement (parent process only)

def sh(cmd) -> str:
    return subprocess.run(cmd, capture_output=True, text=True).stdout


def phys_footprint_gb(pid: int) -> float:
    """Physical footprint of `pid` (Activity Monitor's Memory; counts Metal memory, unlike RSS).

    rusage_info_v4: ri_uuid[16], then uint64 fields; ri_phys_footprint is the eighth. 0.0 if the
    process is gone or the call fails.
    """
    class RUsage(ctypes.Structure):
        _fields_ = [("ri_uuid", ctypes.c_uint8 * 16)] + [(f"f{i}", ctypes.c_uint64) for i in range(48)]

    lib = ctypes.CDLL(ctypes.util.find_library("proc") or "libproc.dylib")
    lib.proc_pid_rusage.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
    lib.proc_pid_rusage.restype = ctypes.c_int
    buf = RUsage()
    return buf.f7 / 1e9 if lib.proc_pid_rusage(int(pid), 4, ctypes.byref(buf)) == 0 else 0.0


def used_mem_gb() -> float:
    """Anonymous - purgeable + wired + compressed memory, as `lab run` counts it (file cache excluded)."""
    out = sh(["vm_stat"])
    page = int(re.search(r"page size of (\d+)", out).group(1))
    v = {k.strip(): int(n) for k, n in re.findall(r"^([^:\n]+):\s+(\d+)\.?$", out, re.M)}
    return (v.get("Anonymous pages", 0) - v.get("Pages purgeable", 0) + v.get("Pages wired down", 0)
            + v.get("Pages occupied by compressor", 0)) * page / 1e9


def mac_state() -> dict:
    swap = re.search(r"used = ([\d.]+)M", sh(["sysctl", "vm.swapusage"]))
    therm = re.search(r"CPU_Speed_Limit\s*=\s*(\d+)", sh(["pmset", "-g", "therm"]))
    return {"ac_power": "AC Power" in (sh(["pmset", "-g", "batt"]).splitlines() or [""])[0],
            "swap_used_gb": round(float(swap.group(1)) / 1024, 3) if swap else 0.0,
            "pressure": int(sh(["sysctl", "-n", "kern.memorystatus_vm_pressure_level"]).strip() or 0),
            "used_gb": round(used_mem_gb(), 3),
            "cpu_speed_limit": int(therm.group(1)) if therm else None}


def danger(swap0_gb: float) -> str | None:
    """Why a run must be killed now (critical pressure, swap growth, battery), or None. Cheap: 3 calls."""
    swap = re.search(r"used = ([\d.]+)M", sh(["sysctl", "vm.swapusage"]))
    if swap and float(swap.group(1)) / 1024 - swap0_gb > SWAP_KILL_GB:
        return f"swap grew more than {SWAP_KILL_GB} GB"
    if int(sh(["sysctl", "-n", "kern.memorystatus_vm_pressure_level"]).strip() or 0) >= 4:
        return "memory pressure critical"
    if "AC Power" not in (sh(["pmset", "-g", "batt"]).splitlines() or [""])[0]:
        return "power left AC"
    return None


def wait_quiet(limit_s: float = QUIET_WAIT_S) -> dict:
    """Block until the Mac is on AC, at normal pressure and under QUIET_USED_GB for two checks 30 s apart."""
    t0, ok = time.time(), 0
    while True:
        st = mac_state()
        ok = ok + 1 if st["ac_power"] and st["pressure"] == 1 and st["used_gb"] <= QUIET_USED_GB else 0
        if ok >= 2:
            return st
        if time.time() - t0 > limit_s:
            raise SystemExit(f"the Mac did not get quiet within {limit_s:.0f} s: {st}")
        time.sleep(30)


def start_balloon(gb: float):
    """A child process holding `gb` of random Metal memory until its stdin closes; (process, footprint GB).

    Random data, because zeros compress and the OS would squeeze the balloon instead of the page
    cache. 268 MB pieces, because one MLX array past 2**31 elements overflows int32 sizes.
    """
    code = ("import math, sys, mlx.core as mx\n"
            f"n = math.ceil({gb} * 1e9 / (4 * 2**26))\n"
            "mx.random.seed(0)\n"
            "a = [mx.random.uniform(shape=(2**26,)) for _ in range(n)]\n"
            "mx.eval(*a)\n"
            "print('ready', flush=True)\n"
            "sys.stdin.read()\n")
    p = subprocess.Popen([sys.executable, "-c", code], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    if p.stdout.readline().strip() != "ready":
        raise RuntimeError("balloon did not start")
    time.sleep(1.0)
    return p, phys_footprint_gb(p.pid)


def long_prompt() -> str:
    readme = subprocess.run(["git", "-C", ROOT, "show", f"{README_COMMIT}:README.md"],
                            capture_output=True, text=True, check=True).stdout
    return LONG_PREFIX + readme[:LONG_CHARS]


def export_tree(commit: str, dest: str) -> str:
    """bigrig_engine/ at `commit`, exported with git archive into `dest`; returns dest."""
    os.makedirs(dest, exist_ok=True)
    tar = subprocess.run(["git", "-C", ROOT, "archive", commit, "bigrig_engine"], capture_output=True, check=True)
    subprocess.run(["tar", "-x", "-C", dest], input=tar.stdout, check=True)
    return dest


# ---------------------------------------------------------------- worker (fresh process per run)

def one_reply(session, prompt: str) -> dict:
    """One greedy reply through the public streaming call, timed by wall clock.

    A yield is not one token (text that could start a stop sequence is held back), so decode
    speed is the engine's token count advanced between the first and last yield over the time
    between them, as in climb_decode.py. turn_s runs from the call to the last yield.
    """
    chunks, t_first, t_last, g_first, g_last, prompt_tokens = [], None, None, None, 0, None
    t0 = time.perf_counter()
    for chunk, info in session.stream_text([{"role": "user", "content": prompt}], max_tokens=MAX_TOKENS,
                                           temperature=0.0, seed=0, think=False):
        now = time.perf_counter()
        g = int(info.get("generation_tokens") or 0)
        prompt_tokens = info.get("prompt_tokens") or prompt_tokens
        if chunk and t_first is None:
            t_first, g_first = now, g
        if t_first is not None:
            t_last, g_last = now, max(g_last, g)
        chunks.append(chunk)
    if t_first is None or g_last <= g_first:
        raise RuntimeError(f"no decode measured for prompt {prompt[:40]!r}")
    return {"text": "".join(chunks), "tokens": g_last, "prompt_tokens": prompt_tokens,
            "ttft_s": t_first - t0, "turn_s": t_last - t0,
            "decode_tokens": g_last - g_first, "decode_s": t_last - t_first}


def worker(arm: str, out: str) -> None:
    """Load bigrig_engine from `arm` (first on sys.path), answer the workload, write JSON to `out`."""
    sys.path.insert(0, arm)
    import bigrig_engine  # noqa: F401 -- imported first so every submodule resolves inside `arm`
    from bigrig_engine.session import Session

    t0 = time.perf_counter()
    s = Session(MODEL, budget_gb=BUDGET_GB, capacity=PIN["capacity"], full_layers=(), verbose=False)
    load_s = time.perf_counter() - t0
    st = s.stats()
    one_reply(s, WARMUP)
    prompts = [long_prompt()] + [p for p, _ in SHORT_PROMPTS]
    rows = [one_reply(s, p) for p in prompts]
    s.close()
    modules = {n: getattr(m, "__file__", None) for n, m in list(sys.modules.items())
               if n == "bigrig_engine" or n.startswith("bigrig_engine.")}
    with open(out, "w") as f:
        json.dump({"rows": rows, "load_s": round(load_s, 2), "modules": modules,
                   "plan": {"capacity": st.get("capacity"), "full_layers": st.get("full_layers")}}, f)


# ---------------------------------------------------------------- one run, watched from the parent

def run_arm(arm: str, balloon_gb: float, python: str = sys.executable) -> dict:
    """One worker run under a fresh balloon; the parent samples footprint and swap every 0.1 s."""
    balloon, b0 = start_balloon(balloon_gb) if balloon_gb > 0 else (None, 0.0)
    fd, out = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    swap0 = mac_state()["swap_used_gb"]
    t0 = time.perf_counter()
    p = subprocess.Popen([python, os.path.abspath(__file__), "--worker", arm, out], stdin=subprocess.DEVNULL)
    peak, killed, stop = [0.0], [None], threading.Event()

    def watch():
        tick = 0
        while not stop.is_set():
            peak[0] = max(peak[0], phys_footprint_gb(p.pid))
            why = danger(swap0) if tick % 20 == 0 else None          # Mac state every ~2 s
            if why is None and time.perf_counter() - t0 > RUN_TIMEOUT_S:
                why = f"timeout {RUN_TIMEOUT_S} s"
            if why:
                killed[0] = why
                p.kill()
                return
            tick += 1
            time.sleep(0.1)

    th = threading.Thread(target=watch, daemon=True)
    th.start()
    code = p.wait()
    wall_s = time.perf_counter() - t0
    stop.set()
    th.join()
    b1 = phys_footprint_gb(balloon.pid) if balloon is not None else 0.0
    if balloon is not None:
        balloon.stdin.close()
        balloon.wait(timeout=30)
    try:
        if code != 0 or killed[0]:
            raise RuntimeError(f"run of {arm} failed: exit {code}, {killed[0] or 'no kill'}")
        with open(out) as f:
            res = json.load(f)
    finally:
        os.unlink(out)
    res.update(arm=arm, wall_s=round(wall_s, 2), peak_gb=round(peak[0], 3),
               balloon_start_gb=round(b0, 2), balloon_end_gb=round(b1, 2))
    return res


# ---------------------------------------------------------------- analysis (pure; unit-tested)

def healthy(row: dict, keyword: str, min_tokens: int) -> bool:
    """Has its keyword, at least `min_tokens` tokens, and no 20-character run repeated 5 times."""
    flat = re.sub(r"\s+", " ", row["text"])
    loops = any(flat.count(flat[i:i + 20]) >= 5 for i in range(0, max(0, len(flat) - 20), 10))
    return keyword in row["text"] and row["tokens"] >= min_tokens and not loops


def plan_ok(run: dict) -> bool:
    arm = os.path.realpath(run["arm"])
    inside = all(f is None or os.path.realpath(f).startswith(arm + os.sep) for f in run["modules"].values())
    return run["plan"] == PIN and inside and bool(run["modules"])


def analyse(runs: list, swap_growth_gb: float) -> dict:
    """The results dict from four runs in ORDER. Pure: no I/O, no clock."""
    if [r["tag"] for r in runs] != list(ORDER):
        raise ValueError(f"runs must be in order {ORDER}")
    pairs = [(runs[0], runs[1]), (runs[3], runs[2])]          # (base, candidate)
    checks = [(LONG_KEYWORD, MIN_TOKENS["long"])] + [(k, MIN_TOKENS["short"]) for _, k in SHORT_PROMPTS]
    key = lambda run: [(r["text"], r["tokens"]) for r in run["rows"]]
    turn = lambda run: sum(r["turn_s"] for r in run["rows"])
    decode = lambda run: sum(r["decode_tokens"] for r in run["rows"]) / sum(r["decode_s"] for r in run["rows"])
    gmean = lambda xs: math.exp(sum(math.log(x) for x in xs) / len(xs))

    speedups = [turn(b) / turn(c) for b, c in pairs]
    return {
        "speedup": round(gmean(speedups), 4),
        "speedup_pairs": [round(x, 4) for x in speedups],
        "exact": int(all(key(b) == key(c) for b, c in pairs) and key(pairs[0][0]) == key(pairs[1][0])),
        "healthy": int(all(len(run["rows"]) == len(checks) and
                           all(healthy(row, kw, n) for row, (kw, n) in zip(run["rows"], checks)) for run in runs)),
        "plan_ok": int(all(plan_ok(r) for r in runs)),
        "balloon_kept": int(all(r["balloon_end_gb"] >= BALLOON_KEEP * r["balloon_start_gb"] for r in runs)),
        "decode_ratio": round(gmean([decode(c) / decode(b) for b, c in pairs]), 4),
        "ttft_long_ratio": round(gmean([c["rows"][0]["ttft_s"] / b["rows"][0]["ttft_s"] for b, c in pairs]), 4),
        "peak_over_base_gb": round(max(c["peak_gb"] - b["peak_gb"] for b, c in pairs), 3),
        "swap_growth_gb": round(swap_growth_gb, 3),
        "turn_s": {"base": [round(turn(b), 2) for b, _ in pairs], "cand": [round(turn(c), 2) for _, c in pairs]},
        "decode_tok_s": {"base": [round(decode(b), 3) for b, _ in pairs],
                         "cand": [round(decode(c), 3) for _, c in pairs]},
        "ttft_long_s": {"base": [round(b["rows"][0]["ttft_s"], 2) for b, _ in pairs],
                        "cand": [round(c["rows"][0]["ttft_s"], 2) for _, c in pairs]},
        "long_prompt_tokens": runs[0]["rows"][0]["prompt_tokens"],
        "estimator_id": ESTIMATOR_ID,
    }


# ---------------------------------------------------------------- the eval

def run_eval(base_arm: str, cand_arm: str, balloon_gb: float = BALLOON_GB, python: str = sys.executable) -> dict:
    start = mac_state()
    runs = []
    for tag in ORDER:
        r = run_arm(base_arm if tag == "base" else cand_arm, balloon_gb, python)
        r["tag"] = tag
        runs.append(r)
    end = mac_state()
    res = analyse(runs, end["swap_used_gb"] - start["swap_used_gb"])
    res.update(mac_start=start, mac_end=end, budget_gb=BUDGET_GB, balloon_gb=balloon_gb, pin=PIN,
               base_commit=BASE_COMMIT, max_tokens=MAX_TOKENS,
               runs=[{k: r[k] for k in ("tag", "arm", "wall_s", "peak_gb", "load_s", "plan",
                                        "balloon_start_gb", "balloon_end_gb", "rows")} for r in runs])
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=os.environ.get("LAB_RESULTS"))
    ap.add_argument("--worker", nargs=2, metavar=("ARM", "OUT"), help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.worker:
        worker(*a.worker)
        return
    if not a.out:
        sys.exit("set LAB_RESULTS or --out")
    wait_quiet()
    tmp = tempfile.mkdtemp(prefix="climb_turn_")
    try:
        res = run_eval(export_tree(BASE_COMMIT, os.path.join(tmp, "base")), ROOT)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    res["cand_commit"] = sh(["git", "-C", ROOT, "rev-parse", "HEAD"]).strip()
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: v for k, v in res.items() if k not in ("runs", "mac_start", "mac_end")}))


if __name__ == "__main__":
    main()
