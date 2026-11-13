"""Locked speed eval: how fast the flagship writes once a reply has started, on a 16 GB Mac's budget.

WHAT IT MEASURES
    Qwen3.6-35B-A3B-4bit, streamed, at a 6.1 GB ceiling (the smallest budget the planner accepts
    for this model on a 16 GB Mac). Three fixed prompts, greedy, thinking off, 128 tokens each,
    after one discarded warm-up reply. Writes JSON to $LAB_RESULTS (or --out):

        decode_tok_s   decode tokens / decode seconds, wall clock, summed over the prompts
        ttft_s         median wall-clock seconds from the call to the first token
        peak_mem_gb    peak phys_footprint of this process (counts MLX/Metal memory, unlike RSS)
        exact          1 if every reply's text AND token count equal the locked reference, else 0
        match_frac     share of reference characters matched before the first difference

WHY EVERY NUMBER IS WALL CLOCK
    The engine reports its own tok/s. That is the code under test grading itself, so it is
    recorded for comparison and never used as the metric.

WHY A BALLOON
    This Mac has 24 GB. Spare RAM becomes page cache, the page cache holds most of the 19 GB of
    weights, and every streamed read looks cheap: a speed-up found that way may not exist on the
    16 GB Mac this budget is for. A separate process holds BALLOON_GB of Metal memory for the
    whole run so the page cache is the size it would be there. This is a simulation; one run on
    a real 16 GB Mac is the check on it. The balloon's own footprint at start and end is
    recorded (the climb gates on the end value), and so is compressor growth -- a diagnostic
    only: measured 2026-11-11, the OS compressed 9.54 GB of OTHER apps' memory during one run
    while the balloon held 8.07 GB, so it tracks what else is open, not the engine.

WHY EXACT
    Weights are untouched and decode is bit-identical by design. A change that alters even one
    token is out of scope for a speed climb, however fast it is. This is deliberately strict: a
    kernel change that flips a near-tie fails too.

    The reference is this engine's own output at commit 626c47a, so `exact` guards against change,
    not against an error that was already there.
"""
import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

MODEL = os.path.join(ROOT, "models", "Qwen3.6-35B-A3B-4bit")
BUDGET_GB = 6.1
BALLOON_GB = 8.0
MAX_TOKENS = 128
REF = os.path.join(ROOT, "bench", "climb_decode_ref.json")
WARMUP = "Name three rivers in Europe."
PROMPTS = [
    "Explain how a heat pump moves warmth from cold outdoor air into a house.",
    "Write a Python function that returns the n-th Fibonacci number iteratively, with a docstring.",
    "A train leaves at 9:40 and arrives at 13:15. How long is the journey? Show your working.",
]


def vm_gb() -> dict:
    """vm_stat in GB: file-backed pages (the page cache) and pages held by the compressor."""
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    page = int(re.search(r"page size of (\d+)", out).group(1))
    g = lambda k: int(re.search(k + r":\s+(\d+)", out).group(1)) * page / 1e9
    return {"file": g("File-backed pages"), "compressor": g("Pages occupied by compressor")}


def start_balloon(gb: float):
    """A child process holding `gb` of touched memory until its stdin closes.

    Returns (process, its phys_footprint in GB). Measured 2026-11-11 from a full page cache: an
    8 GB balloon left wired pages unchanged and shrank file-backed pages by 8.1 GB, which is
    the effect wanted (from an already-small cache it displaces nothing, so that is not the
    number recorded). RANDOM data, because an array of ones compresses to almost nothing: under
    pressure the OS would squeeze the balloon instead of the page cache.
    In 268 MB pieces: one array past 2**31 elements overflows MLX's int32 sizes, and a single
    Metal buffer that large can exceed the device's maximum buffer length.
    """
    code = ("import math, sys, mlx.core as mx\n"
            f"n = math.ceil({gb} * 1e9 / (4 * 2**26))\n"
            "mx.random.seed(0)\n"
            "a = [mx.random.uniform(shape=(2**26,)) for _ in range(n)]\n"
            "mx.eval(*a)\n"
            "print('ready', flush=True)\n"
            "sys.stdin.read()\n")
    p = subprocess.Popen([sys.executable, "-c", code], stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE, text=True)
    if p.stdout.readline().strip() != "ready":
        raise RuntimeError("balloon did not start")
    time.sleep(1.0)
    from bigrig_engine.calibrate import phys_footprint_gb
    return p, phys_footprint_gb(p.pid)


def thermal_limit():
    """CPU_Speed_Limit from pmset, or None when the Mac has recorded no thermal event."""
    out = subprocess.run(["pmset", "-g", "therm"], capture_output=True, text=True).stdout
    m = re.search(r"CPU_Speed_Limit\s*=\s*(\d+)", out)
    return int(m.group(1)) if m else None


def one_reply(session, prompt: str):
    """One greedy reply through the public streaming call, timed by wall clock.

    Returns {text, tokens, ttft_s, decode_tokens, decode_s, engine_tok_s}. A yield is not one
    token: text that could start a stop sequence is held back and sent later, and the final
    flush repeats the last count. So decode speed is the engine's token count advanced between
    the first and the last yield, over the wall-clock time between them; `tokens` and `text`
    are what the exactness check compares.
    """
    chunks, t_first, t_last, g_first, g_last, engine_tps = [], None, None, None, 0, None
    t0 = time.perf_counter()
    for chunk, info in session.stream_text([{"role": "user", "content": prompt}],
                                           max_tokens=MAX_TOKENS, temperature=0.0, seed=0,
                                           think=False):
        now = time.perf_counter()
        g = int(info.get("generation_tokens") or 0)
        if chunk and t_first is None:
            t_first, g_first = now, g
        if t_first is not None:
            t_last, g_last = now, max(g_last, g)
        chunks.append(chunk)
        engine_tps = info.get("tok_s", engine_tps)
    if t_first is None or g_last <= g_first:
        raise RuntimeError(f"no decode measured for prompt {prompt[:40]!r}")
    return {"text": "".join(chunks), "tokens": g_last, "ttft_s": t_first - t0,
            "decode_tokens": g_last - g_first, "decode_s": t_last - t_first,
            "engine_tok_s": engine_tps}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.environ.get("LAB_RESULTS"))
    ap.add_argument("--write-ref", action="store_true",
                    help="record the reference token ids (once, before locking)")
    ap.add_argument("--no-balloon", action="store_true")
    a = ap.parse_args()
    if not a.out and not a.write_ref:
        sys.exit("set LAB_RESULTS or --out")

    from bigrig_engine.calibrate import phys_footprint_gb
    from bigrig_engine.session import Session

    vm0 = vm_gb()
    balloon, balloon_gb0 = (None, 0.0) if a.no_balloon else start_balloon(BALLOON_GB)
    peak, stop = [0.0], threading.Event()

    def sample():
        while not stop.is_set():
            peak[0] = max(peak[0], phys_footprint_gb())
            time.sleep(0.05)

    th = threading.Thread(target=sample, daemon=True)
    th.start()
    therm = [thermal_limit()]
    try:
        t0 = time.perf_counter()
        s = Session(MODEL, budget_gb=BUDGET_GB, verbose=False)
        load_s = time.perf_counter() - t0
        one_reply(s, WARMUP)
        rows = []
        for p in PROMPTS:
            rows.append(one_reply(s, p))
            therm.append(thermal_limit())
        capacity = getattr(s, "capacity", None)
        vm1 = vm_gb()
        balloon_gb1 = 0.0 if balloon is None else phys_footprint_gb(balloon.pid)
        s.close()
    finally:
        stop.set()
        th.join()
        if balloon is not None:
            balloon.stdin.close()
            balloon.wait(timeout=30)

    got = [{"text": r["text"], "tokens": r["tokens"]} for r in rows]
    if a.write_ref:
        os.makedirs(os.path.dirname(REF), exist_ok=True)
        with open(REF, "w") as f:
            json.dump({"model": os.path.basename(MODEL), "budget_gb": BUDGET_GB,
                       "max_tokens": MAX_TOKENS, "prompts": PROMPTS, "replies": got}, f, indent=1)
        print(f"reference written: {REF}")

    ref = json.load(open(REF))["replies"]
    matched = total = 0
    for g, w in zip(got, ref):
        n = 0
        while n < min(len(g["text"]), len(w["text"])) and g["text"][n] == w["text"][n]:
            n += 1
        matched += n
        total += len(w["text"])
    exact = int(len(got) == len(ref) and all(g == w for g, w in zip(got, ref)))

    decode_tokens = sum(r["decode_tokens"] for r in rows)
    decode_s = sum(r["decode_s"] for r in rows)
    limits = [t for t in therm if t is not None]
    res = {
        "decode_tok_s": round(decode_tokens / decode_s, 4),
        "ttft_s": round(statistics.median(r["ttft_s"] for r in rows), 4),
        "peak_mem_gb": round(peak[0], 3),
        "exact": exact,
        "match_frac": round(matched / total, 4) if total else 0.0,
        "decode_tokens": decode_tokens,
        "load_s": round(load_s, 2),
        "engine_tok_s": [r["engine_tok_s"] for r in rows],
        "capacity": capacity,
        "budget_gb": BUDGET_GB,
        "balloon_gb": 0.0 if a.no_balloon else BALLOON_GB,
        "balloon_start_gb": round(balloon_gb0, 2),
        "balloon_end_gb": round(balloon_gb1, 2),
        "compressor_growth_gb": round(vm1["compressor"] - vm0["compressor"], 2),
        "cpu_speed_limit_min": min(limits) if limits else None,
    }
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)
    print(json.dumps(res))


if __name__ == "__main__":
    main()
