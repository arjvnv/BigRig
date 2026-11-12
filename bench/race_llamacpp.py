"""Race for card `llamacpp-race-16gb-3`: BigRig against llama.cpp on a simulated 16 GB Mac.

SAME MACHINE FOR BOTH
    This 24 GB Mac with a fixed 5 GB random-data balloon held for every run (why -3: card -1 had a
    fixed 8 GB balloon, which with a browser open pushes swap past the guard; card -2 sized it to
    a memory total, and the reading right after a run made it swing 0.05-7 GB). With the browser
    open, 5 GB is about last night's quiet-Mac + 8 GB total. Each engine gets its own best settings
    for that machine, chosen by a fixed sweep (one run per setting), then the two winners race in
    12 interleaved rounds. BigRig is the engine on main (626c47a), as released.

SAME MODEL, EACH ENGINE'S OWN 4-BIT FORMAT
    BigRig: Qwen3.6-35B-A3B-4bit (MLX, 19 GB). llama.cpp: Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf
    (22.36 GB, sha256 707a55a8...). The bytes differ, so this compares the engines as people
    download and run them, not kernels on identical weights. Quality is not compared.

SAME PROMPTS, SAME TIMING
    The warm-up and 3 prompts of the locked bench/climb_decode.py (128 tokens, greedy, thinking
    off), plus one long prompt (the first 6,000 characters of README.md at 626c47a, ~1,500
    tokens, 32 tokens out) for time to first token. Both engines get the same rendered chat text;
    llama.cpp's prompt token count must equal the MLX tokenizer's. Speed is wall clock between
    the first and last streamed token; neither engine's own tok/s is used.

ESTIMATOR (estimator_id below)
    d_r = ln(BigRig decode tok/s in round r) - ln(llama.cpp decode tok/s in round r);
    ratio = exp(mean d), 95% CI = exp(mean d +- t(0.975, n-1) * sd(d) / sqrt(n)), n = 12.
    The same estimator on the long prompt's time to first token is secondary.

MAC SAFETY
    Polled every second during each run; the run's process group is killed on swap more than
    1.0 GB above the start, memory pressure critical, more than 11.0 GB of extra RAM, power off
    AC, or 900 s. In the sweep a killed setting is unhealthy and skipped; in the race it stops all.
"""
import argparse
import http.client
import importlib.util
import json
import math
import os
import re
import signal
import statistics
import subprocess
import sys
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CARD = "llamacpp-race-16gb-3"
ESTIMATOR_ID = "race-paired-logratio-t95-v1"
MODEL = os.path.join(ROOT, "models", "Qwen3.6-35B-A3B-4bit")
GGUF = os.path.join(ROOT, "models", "Qwen3.6-35B-A3B-GGUF", "Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf")
GGUF_BYTES = 22360456160
LLAMA = "/opt/homebrew/bin/llama-server"
LLAMA_VERSION = "version: 0.9.0 (build 12110, commit cd959bae9)"
PORT = 18931
BALLOON_GB = 5.0
MAX_TOKENS = 128
LONG_MAX_TOKENS = 32
README_COMMIT = "626c47a29abbabc94a627da546bcd7cc2b654dcf"
LONG_CHARS = 6000
BIGRIG_COMMITS = {"base": "6e87fcd4030a4f79493f8c4156fa5f3b098e9ae6",
                  "warm": "1c89e3df5aa4c40ad85fdc642357cbd8dc972091",
                  "warm_fast": "0a9bf849f5231711dc070221264ce3addbcb1884"}
# Sweep order is least memory first: a setting killed for memory skips the heavier ones after it.
BIGRIG_GRID = [("bigrig-6.1", 6.1), ("bigrig-7.5", 7.5), ("bigrig-9.0", 9.0)]
# --no-repack: by default llama.cpp copies CPU-side weights into a repacked layout in RAM. With the
# experts on the CPU that is ~20 GB of anonymous memory, which no 16 GB Mac has (trial run
# 2026-11-12: +8.6 GB and swap +1.25 GB in 52 s, without the balloon). Without it the experts
# stay memory-mapped from the file and stream through the page cache, as BigRig's do.
LLAMA_COMMON = ["-c", "4096", "-np", "1", "--no-webui", "--no-repack"]
LLAMA_GRID = [
    ("llama-cmoe-t4", ["-ngl", "99", "--cpu-moe", "-t", "4"]),
    ("llama-cmoe-t8", ["-ngl", "99", "--cpu-moe", "-t", "8"]),
    ("llama-ngl0-t8", ["-ngl", "0", "-t", "8"]),
    ("llama-ncmoe34-t4", ["-ngl", "99", "--n-cpu-moe", "34", "-t", "4"]),
    ("llama-ncmoe34-t8", ["-ngl", "99", "--n-cpu-moe", "34", "-t", "8"]),
    ("llama-ncmoe28-t4", ["-ngl", "99", "--n-cpu-moe", "28", "-t", "4"]),
    ("llama-ncmoe28-t8", ["-ngl", "99", "--n-cpu-moe", "28", "-t", "8"]),
]
HEAVIER = {"bigrig-7.5": ["bigrig-9.0"], "llama-ncmoe34-t4": ["llama-ncmoe28-t4", "llama-ncmoe28-t8"],
           "llama-ncmoe34-t8": ["llama-ncmoe28-t4", "llama-ncmoe28-t8"]}
RACE_ROUNDS = 12
SWAP_GROWTH_MAX_GB = 1.0
EXTRA_MEM_MAX_GB = 11.0
BALLOON_KEEP = 0.9
RUN_TIMEOUT_S = 900
KEYWORDS = ["heat", "def fib", "9:40"]   # one per climb prompt, inside any competent first 128 tokens


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def sh(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw).stdout


def long_prompt() -> str:
    text = sh(["git", "-C", ROOT, "show", f"{README_COMMIT}:README.md"], check=True)
    return "Summarize the following document in two sentences.\n\n" + text[:LONG_CHARS]


def used_mem_gb() -> float:
    """Anonymous + wired + compressed memory, as `lab run` counts it (file cache excluded)."""
    out = sh(["vm_stat"])
    page = int(re.search(r"page size of (\d+)", out).group(1))
    v = {k.strip(): int(n) for k, n in re.findall(r"^([^:\n]+):\s+(\d+)\.?$", out, re.M)}
    return (v.get("Anonymous pages", 0) - v.get("Pages purgeable", 0) + v.get("Pages wired down", 0)
            + v.get("Pages occupied by compressor", 0)) * page / 1e9


def mac_state() -> dict:
    swap = re.search(r"used = ([\d.]+)M", sh(["sysctl", "vm.swapusage"]))
    return {"ac_power": "AC Power" in (sh(["pmset", "-g", "batt"]).splitlines() or [""])[0],
            "swap_used_gb": round(float(swap.group(1)) / 1024, 3) if swap else 0.0,
            "pressure": int(sh(["sysctl", "-n", "kern.memorystatus_vm_pressure_level"]).strip() or 0),
            "used_gb": round(used_mem_gb(), 3)}


def timed(events, t0) -> dict:
    """events yields (text chunk, tokens generated so far). Same rule as climb_decode.one_reply:
    decode = tokens advanced between the first and last event, over the wall time between them."""
    chunks, t_first, t_last, g_first, g_last = [], None, None, None, 0
    for chunk, g in events:
        now = time.perf_counter()
        if chunk and t_first is None:
            t_first, g_first = now, g
        if t_first is not None:
            t_last, g_last = now, max(g_last, g)
        chunks.append(chunk)
    if t_first is None or g_last <= g_first:
        raise RuntimeError("no decode measured")
    return {"text": "".join(chunks), "tokens": g_last, "ttft_s": t_first - t0,
            "decode_tokens": g_last - g_first, "decode_s": t_last - t_first}


def llama_events(prompt: str, n: int, final: dict, port: int = PORT):
    """Stream /completion; count tokens from the ids each event carries (return_tokens)."""
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=RUN_TIMEOUT_S)
    conn.request("POST", "/completion", json.dumps({
        "prompt": prompt, "n_predict": n, "temperature": 0.0, "top_k": 1, "seed": 0,
        "cache_prompt": False, "stream": True, "return_tokens": True}),
        {"Content-Type": "application/json"})
    resp = conn.getresponse()
    if resp.status != 200:
        raise RuntimeError(f"llama-server HTTP {resp.status}: {resp.read()[:300]!r}")
    g = 0
    for raw in resp:
        line = raw.decode("utf-8", "replace").strip()
        if not line.startswith("data: "):
            continue
        ev = json.loads(line[6:])
        g += len(ev.get("tokens") or [])
        if ev.get("stop"):
            final.update({k: ev.get(k) for k in ("tokens_evaluated", "tokens_predicted", "stop_type")})
        yield ev.get("content", ""), g
    conn.close()


class Peak:
    """Samples a footprint function every 50 ms on a thread; .value is the maximum seen."""
    def __init__(self, fn):
        self.value, self._stop = 0.0, threading.Event()
        self._fn = fn
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def _run(self):
        while not self._stop.is_set():
            self.value = max(self.value, self._fn())
            time.sleep(0.05)

    def stop(self):
        self._stop.set()
        self._t.join()


def summarise(rows, long, extra) -> dict:
    dt, ds = sum(r["decode_tokens"] for r in rows), sum(r["decode_s"] for r in rows)
    return {"decode_tok_s": round(dt / ds, 4), "ttft_s": round(statistics.median(r["ttft_s"] for r in rows), 4),
            "ttft_long_s": round(long["ttft_s"], 4), "long_prompt_tokens": long.get("prompt_tokens"),
            "rows": rows, "long": long, **extra}


def worker_bigrig(arm_dir: str, budget: float, out: str, balloon_gb: float = BALLOON_GB):
    sys.path.insert(0, arm_dir)
    import bigrig_engine  # noqa: F401  -- first, so every bigrig_engine.* resolves inside the arm
    cdc = _load("climb_decode", os.path.join(ROOT, "bench", "climb_decode.py"))
    from bigrig_engine.calibrate import phys_footprint_gb
    from bigrig_engine.session import Session
    balloon, b0 = cdc.start_balloon(balloon_gb) if balloon_gb > 0 else (None, 0.0)
    peak = Peak(phys_footprint_gb)
    try:
        t0 = time.perf_counter()
        s = Session(MODEL, budget_gb=budget, verbose=False)
        load_s = time.perf_counter() - t0

        def reply(content, n):
            msgs = [{"role": "user", "content": content}]
            t = time.perf_counter()
            r = timed(((c, int(i.get("generation_tokens") or 0)) for c, i in s.stream_text(
                msgs, max_tokens=n, temperature=0.0, seed=0, think=False)), t)
            r["prompt"] = s.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                                          enable_thinking=False)
            return r

        reply(cdc.WARMUP, MAX_TOKENS)
        rows = [reply(p, MAX_TOKENS) for p in cdc.PROMPTS]
        long = reply(long_prompt(), LONG_MAX_TOKENS)
        st = s.stats()
        capacity, full_layers = st.get("capacity"), st.get("full_layers")
        b1 = phys_footprint_gb(balloon.pid) if balloon else 0.0
        s.close()
    finally:
        peak.stop()
        if balloon:
            balloon.stdin.close()
            balloon.wait(timeout=30)
    ref = json.load(open(cdc.REF))["replies"]
    exact = int(all({"text": r["text"], "tokens": r["tokens"]} == w for r, w in zip(rows, ref)))
    mods = {n: getattr(m, "__file__", None) for n, m in list(sys.modules.items())
            if n == "bigrig_engine" or n.startswith("bigrig_engine.")}
    res = summarise(rows, long, {"engine": "bigrig", "budget_gb": budget, "capacity": capacity, "full_layers": full_layers, "exact": exact,
                                 "load_s": round(load_s, 2), "peak_mem_gb": round(peak.value, 3),
                                 "balloon_start_gb": round(b0, 2), "balloon_end_gb": round(b1, 2),
                                 "modules_outside_arm": [n for n, f in mods.items() if not f or not
                                                         os.path.realpath(f).startswith(arm_dir + os.sep)]})
    json.dump(res, open(out, "w"), indent=1)


def worker_llama(flags: list, out: str, balloon_gb: float = BALLOON_GB):
    cdc = _load("climb_decode", os.path.join(ROOT, "bench", "climb_decode.py"))
    from bigrig_engine.calibrate import phys_footprint_gb
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)

    def render(content):
        return tok.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False)

    balloon, b0 = cdc.start_balloon(balloon_gb) if balloon_gb > 0 else (None, 0.0)
    srv, peak = None, None
    try:
        t0 = time.perf_counter()
        srv = subprocess.Popen([LLAMA, "-m", GGUF, *LLAMA_COMMON, "--host", "127.0.0.1", "--port", str(PORT), *flags],
                               stdin=subprocess.DEVNULL, stdout=open(out + ".server.log", "w"),
                               stderr=subprocess.STDOUT)
        while True:
            if srv.poll() is not None:
                raise RuntimeError(f"llama-server exited {srv.returncode} while loading")
            try:
                c = http.client.HTTPConnection("127.0.0.1", PORT, timeout=2)
                c.request("GET", "/health")
                if c.getresponse().status == 200:
                    break
            except OSError:
                pass
            time.sleep(0.5)
        load_s = time.perf_counter() - t0
        peak = Peak(lambda: phys_footprint_gb(srv.pid))

        def reply(content, n):
            p, final = render(content), {}
            t = time.perf_counter()
            r = timed(llama_events(p, n, final), t)
            r.update(prompt=p, prompt_tokens=final.get("tokens_evaluated"), stop_type=final.get("stop_type"),
                     mlx_prompt_tokens=len(tok(p, add_special_tokens=False)["input_ids"]))
            return r

        reply(cdc.WARMUP, MAX_TOKENS)
        rows = [reply(p, MAX_TOKENS) for p in cdc.PROMPTS]
        long = reply(long_prompt(), LONG_MAX_TOKENS)
        b1 = phys_footprint_gb(balloon.pid) if balloon else 0.0
    finally:
        if peak:
            peak.stop()
        if srv and srv.poll() is None:
            srv.terminate()
            try:
                srv.wait(timeout=30)
            except subprocess.TimeoutExpired:
                srv.kill()
        if balloon:
            balloon.stdin.close()
            balloon.wait(timeout=30)
    res = summarise(rows, long, {"engine": "llama", "flags": flags, "load_s": round(load_s, 2),
                                 "peak_mem_gb": round(peak.value, 3), "balloon_start_gb": round(b0, 2),
                                 "balloon_end_gb": round(b1, 2)})
    json.dump(res, open(out, "w"), indent=1)


def unhealthy(res) -> list:
    """Reasons a finished run cannot count. Empty list = healthy."""
    why = []
    for r, kw in zip(res["rows"], KEYWORDS):
        flat = re.sub(r"\s+", " ", r["text"])
        if kw not in flat.lower():
            why.append(f"reply lacks {kw!r}")
        if r["tokens"] < 64:
            why.append(f"reply only {r['tokens']} tokens")
        if any(flat.count(flat[i:i + 20]) >= 5 for i in range(0, max(0, len(flat) - 20), 10)):
            why.append("reply repeats itself")
    if res["balloon_start_gb"] <= 0 or res["balloon_end_gb"] < BALLOON_KEEP * res["balloon_start_gb"]:
        why.append(f"balloon {res['balloon_start_gb']} -> {res['balloon_end_gb']} GB")
    if res["engine"] == "bigrig":
        # Not gated on the locked reference: BigRig's reply depends on the expert plan it picks
        # from the memory free at start (trial 2026-11-12: 11 slots left the 8-slot reference at
        # about token 15), and the sweep varies the budget on purpose. `exact` is recorded.
        if res["modules_outside_arm"]:
            why.append("bigrig_engine loaded from outside its arm")
    else:
        for r in res["rows"] + [res["long"]]:
            if r["prompt_tokens"] != r["mlx_prompt_tokens"]:
                why.append(f"prompt tokens differ: llama {r['prompt_tokens']} vs MLX {r['mlx_prompt_tokens']}")
    return why


def paired(b_rows, l_rows, key, alpha=0.05) -> dict:
    """BigRig over llama.cpp, per round: ratio = exp(mean ln(b/l)) with a t interval."""
    t_quantile = _load("confirm_decode", os.path.join(ROOT, "bench", "confirm_decode.py")).t_quantile
    d = [math.log(b[key]) - math.log(l[key]) for b, l in zip(b_rows, l_rows)]
    m, s = statistics.mean(d), statistics.stdev(d)
    h = t_quantile(1 - alpha / 2, len(d) - 1) * s / math.sqrt(len(d))
    return {"ratio": round(math.exp(m), 4), "lo": round(math.exp(m - h), 4), "hi": round(math.exp(m + h), 4),
            "n": len(d), "sd_log": round(s, 5)}


def decide(dec: dict, ttft: dict) -> dict:
    if dec["ratio"] >= 1.10 and dec["lo"] > 1.0:
        d = "bigrig-faster"
    elif dec["hi"] < 1.0:
        d = "llama-faster"
    else:
        d = "no-claim"
    if ttft["hi"] < 1.0:
        t = "bigrig-first-token-faster"
    elif ttft["lo"] > 1.0:
        t = "llama-first-token-faster"
    else:
        t = "no-claim"
    return {"decision": d, "decision_ttft_long": t}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", choices=["bigrig", "llama"])
    ap.add_argument("--arm-dir")
    ap.add_argument("--budget", type=float)
    ap.add_argument("--flags")
    ap.add_argument("--out")
    ap.add_argument("--balloon-gb", type=float, default=BALLOON_GB)
    ap.add_argument("--smoke", action="store_true",
                    help="plumbing check only: one run per engine, no balloon, base BigRig; never a result")
    a = ap.parse_args()
    if a.worker == "bigrig":
        return worker_bigrig(a.arm_dir, a.budget, a.out, a.balloon_gb)
    if a.worker == "llama":
        return worker_llama(json.loads(a.flags), a.out, a.balloon_gb)

    res_path = os.environ.get("LAB_RESULTS") or a.out
    work = os.environ.get("LAB_RUN_DIR") or os.path.dirname(os.path.abspath(res_path))
    start = mac_state()
    res = {"estimator_id": ESTIMATOR_ID, "card": CARD, "complete": False, "mac_start": start, "sweep": {}, "race": []}

    def save():
        json.dump(res, open(res_path, "w"), indent=1)

    def stop(reason):
        res.update(stopped_reason=reason, mac_end=mac_state())
        save()
        print(f"STOPPED: {reason}", flush=True)
        sys.exit(3)

    if not start["ac_power"]:
        stop("not on AC power (battery throttles the GPU about 1.7x)")
    if LLAMA_VERSION not in subprocess.run([LLAMA, "--version"], stdout=subprocess.PIPE,
                                           stderr=subprocess.STDOUT, text=True).stdout:
        stop(f"llama-server is not {LLAMA_VERSION}")
    if os.path.getsize(GGUF) != GGUF_BYTES:
        stop("GGUF size changed")
    run, arm, commit = "fixed", "base", BIGRIG_COMMITS["base"]
    arm_dir = os.path.realpath(os.path.join(work, "bigrig_arm"))
    os.makedirs(arm_dir)
    tar = subprocess.run(["git", "-C", ROOT, "archive", commit, "bigrig_engine"], capture_output=True, check=True).stdout
    subprocess.run(["tar", "-x", "-C", arm_dir], input=tar, check=True)
    res["bigrig"] = {"arm": arm, "commit": commit, "from_run": run,
                     "engine_tree": sh(["git", "-C", ROOT, "rev-parse", f"{commit}:bigrig_engine"]).strip()}
    res["llama"] = {"version": LLAMA_VERSION, "gguf": os.path.basename(GGUF), "common_flags": LLAMA_COMMON}
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("BIGRIG_", "LLAMA_")) and k not in ("LAB_RESULTS", "LAB_RUN_DIR")}
    env["HF_HUB_OFFLINE"] = "1"
    configs = {n: ["--worker", "bigrig", "--arm-dir", arm_dir, "--budget", str(b)] for n, b in BIGRIG_GRID}
    configs.update({n: ["--worker", "llama", "--flags", json.dumps(f)] for n, f in LLAMA_GRID})
    res["balloon_gb"] = 0.0 if a.smoke else BALLOON_GB

    def one(name, tag):
        out = os.path.join(work, tag + ".json")
        before = mac_state()
        bgb = 0.0 if a.smoke else BALLOON_GB
        p = subprocess.Popen([sys.executable, os.path.abspath(__file__), *configs[name], "--balloon-gb", str(bgb),
                              "--out", out], cwd=ROOT,
                             env=env, stdin=subprocess.DEVNULL, stdout=open(out + ".log", "w"),
                             stderr=subprocess.STDOUT, start_new_session=True)
        t0, killed, peak_extra, swap_max = time.time(), None, 0.0, 0.0
        while p.poll() is None:
            time.sleep(1.0)
            st = mac_state()
            peak_extra = max(peak_extra, st["used_gb"] - start["used_gb"])
            swap_max = max(swap_max, st["swap_used_gb"] - start["swap_used_gb"])
            killed = ("power left AC" if not st["ac_power"] else "memory pressure critical" if st["pressure"] >= 4
                      else f"swap grew {swap_max:.2f} GB" if swap_max > SWAP_GROWTH_MAX_GB
                      else f"extra RAM {peak_extra:.2f} GB" if peak_extra > EXTRA_MEM_MAX_GB
                      else "timeout" if time.time() - t0 > RUN_TIMEOUT_S else None)
            if killed:
                os.killpg(p.pid, signal.SIGTERM)
                try:
                    p.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(p.pid, signal.SIGKILL)
                    p.wait()
                break
        r = {"config": name, "tag": tag, "killed": killed, "exit": p.returncode, "wall_s": round(time.time() - t0, 1),
             "balloon_gb": bgb, "used_before_gb": before["used_gb"],
             "peak_extra_gb": round(peak_extra, 3), "swap_growth_max_gb": round(swap_max, 3)}
        if killed == "power left AC":
            stop(f"power left AC during {tag}")
        if not killed and p.returncode == 0 and os.path.exists(out):
            r["result"] = json.load(open(out))
            r["unhealthy"] = unhealthy(r["result"])
        else:
            r["unhealthy"] = [killed or f"exit {p.returncode}"]
        line = (f"{tag}: {r['result']['decode_tok_s']} tok/s, ttft {r['result']['ttft_s']} s, long ttft "
                f"{r['result']['ttft_long_s']} s" if "result" in r else f"{tag}: no result")
        print(line + (f"  UNHEALTHY {r['unhealthy']}" if r["unhealthy"] else ""), flush=True)
        return r

    if a.smoke:
        res["smoke"] = [one("bigrig-6.1", "smoke-bigrig-6.1"), one("llama-cmoe-t4", "smoke-llama-cmoe-t4")]
        res["mac_end"] = mac_state()
        save()
        return
    skip = set()
    for name in [n for n, _ in BIGRIG_GRID] + [n for n, _ in LLAMA_GRID]:
        if name in skip:
            res["sweep"][name] = {"config": name, "skipped": "a lighter setting was killed for memory"}
            continue
        r = res["sweep"][name] = one(name, f"sweep-{name}")
        save()
        if r["killed"] and r["killed"] != "timeout":
            skip.update(HEAVIER.get(name, []))

    def best(prefix):
        ok = [r for n, r in res["sweep"].items() if n.startswith(prefix) and "result" in r and not r["unhealthy"]]
        return max(ok, key=lambda r: r["result"]["decode_tok_s"])["config"] if ok else None

    b_cfg, l_cfg = best("bigrig"), best("llama")
    res["chosen"] = {"bigrig": b_cfg, "llama": l_cfg}
    if b_cfg is None:
        stop("no healthy BigRig setting")
    if l_cfg is None:
        res.update(complete=True, decision="llama-did-not-run", mac_end=mac_state())
        save()
        print("llama.cpp had no healthy setting on the simulated 16 GB Mac", flush=True)
        return
    for rd in range(RACE_ROUNDS + 1):
        order = [b_cfg, l_cfg] if rd % 2 == 0 else [l_cfg, b_cfg]
        for name in order:
            r = one(name, f"race-r{rd:02d}-{name}")
            r["round"] = rd
            res["race"].append(r)
            save()
            if r["unhealthy"]:
                stop(f"race run {r['tag']} unhealthy: {r['unhealthy']}")
    scored = [r for r in res["race"] if r["round"] >= 1]
    b_rows = sorted((r for r in scored if r["config"] == b_cfg), key=lambda r: r["round"])
    l_rows = sorted((r for r in scored if r["config"] == l_cfg), key=lambda r: r["round"])
    b_res, l_res = [r["result"] for r in b_rows], [r["result"] for r in l_rows]
    dec = paired(b_res, l_res, "decode_tok_s")
    ttft = paired(b_res, l_res, "ttft_long_s")
    res.update(decode=dec, ttft_long=ttft, **decide(dec, ttft))
    for eng, rows in (("bigrig", b_res), ("llama", l_res)):
        res[eng].update({f"median_{k}": round(statistics.median(r[k] for r in rows), 4)
                         for k in ("decode_tok_s", "ttft_s", "ttft_long_s", "peak_mem_gb")})
    res["bigrig"]["median_peak_extra_gb"] = round(statistics.median(r["peak_extra_gb"] for r in b_rows), 3)
    res["llama"]["median_peak_extra_gb"] = round(statistics.median(r["peak_extra_gb"] for r in l_rows), 3)
    res.update(complete=True, mac_end=mac_state())
    res["swap_growth_gb"] = round(res["mac_end"]["swap_used_gb"] - start["swap_used_gb"], 3)
    save()
    print(json.dumps({k: res[k] for k in ("chosen", "decode", "ttft_long", "decision", "decision_ttft_long")}, indent=1))


if __name__ == "__main__":
    main()
