"""Confirm run for card `warm-ahead-confirm`: do the climb's two decode speed-ups hold up?

ARMS (bigrig_engine/ exported from git at each commit; nothing else differs)
    base       6e87fcd  the climb's starting code (bigrig_engine identical to main 626c47a)
    warm       1c89e3d  + read the next layer's predicted experts in the background (climb iter 5)
    warm_fast  0a9bf84  + decode host fast path (climb iter 8, on top of iter 5)

REGIMES (the locked bench/climb_decode.py, unchanged, run once per arm per round)
    16gb       6.1 GB budget, 8 GB balloon: the page cache a 16 GB Mac would have. 18 rounds.
    24gb_warm  same, --no-balloon: page cache holds most of the weights. No-regression guard. 12 rounds.

    Each regime starts with one discarded warm-up round (thermal state, page cache). Round r
    runs the arms in permutation (r - 1) % 6 of all six orders, so every order appears equally
    often and drift (heat, cache left by the previous run) lands on every arm alike.

ESTIMATOR (estimator_id below; the card's decision rule reads these numbers)
    For arms X over Y in a regime, d_r = ln(tok_s X in round r) - ln(tok_s Y in round r).
    ratio = exp(mean d); 97.5% CI = exp(mean d +- t(0.9875, n-1) * sd(d) / sqrt(n)).
    97.5% rather than 95% because the card asks two primary questions (Bonferroni).

STOPS (kill criteria): any reply not exact; Mac not on AC power; swap more than 1.5 GB above
the start; memory pressure critical; the balloon squeezed below 7.5 GB; a run that fails or
times out twice in one arm; an arm whose capacity differs; code loaded from the wrong arm.
"""
import itertools
import json
import math
import os
import re
import statistics
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLIMB = os.path.join(ROOT, "bench", "climb_decode.py")
CARD = "warm-ahead-confirm"
ESTIMATOR_ID = "paired-logratio-t9875-v1"
ARMS = {
    "base": "6e87fcd4030a4f79493f8c4156fa5f3b098e9ae6",
    "warm": "1c89e3df5aa4c40ad85fdc642357cbd8dc972091",
    "warm_fast": "0a9bf849f5231711dc070221264ce3addbcb1884",
}
REGIMES = [("16gb", [], 18), ("24gb_warm", ["--no-balloon"], 12)]
ORDERS = list(itertools.permutations(ARMS))
RUN_TIMEOUT_S = 420
SWAP_GROWTH_MAX_GB = 1.5
BALLOON_MIN_GB = 7.5
ALPHA = 0.025

# Loads bigrig_engine from the arm's export, then runs the locked eval file unchanged. The
# package is imported first, so climb_decode's own sys.path.insert(ROOT) cannot swap it back:
# every bigrig_engine.* submodule resolves through the already-imported package's __path__.
SHIM = r"""
import json, runpy, sys
arm, prov, climb = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, arm)
import bigrig_engine
sys.argv = [climb] + sys.argv[4:]
try:
    runpy.run_path(climb, run_name="__main__")
finally:
    with open(prov, "w") as f:
        json.dump({n: getattr(m, "__file__", None) for n, m in list(sys.modules.items())
                   if n == "bigrig_engine" or n.startswith("bigrig_engine.")}, f)
"""


def t_quantile(p: float, df: int) -> float:
    """Student-t quantile by Simpson integration of the density plus bisection (no scipy here)."""
    c = math.exp(math.lgamma((df + 1) / 2) - math.lgamma(df / 2)) / math.sqrt(df * math.pi)
    pdf = lambda x: c * (1 + x * x / df) ** (-(df + 1) / 2)

    def cdf(t, n=4000):
        h = t / n
        s = pdf(0) + pdf(t) + sum((4 if i % 2 else 2) * pdf(i * h) for i in range(1, n))
        return 0.5 + s * h / 3

    lo, hi = 0.0, 50.0
    for _ in range(80):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if cdf(mid) < p else (lo, mid)
    return (lo + hi) / 2


def sh(cmd):
    return subprocess.run(cmd, capture_output=True, text=True).stdout


def mac_state() -> dict:
    swap = re.search(r"used = ([\d.]+)M", sh(["sysctl", "vm.swapusage"]))
    return {"ac_power": "AC Power" in sh(["pmset", "-g", "batt"]).splitlines()[0],
            "swap_used_gb": round(float(swap.group(1)) / 1024, 3) if swap else None,
            "pressure": int(sh(["sysctl", "-n", "kern.memorystatus_vm_pressure_level"]).strip() or 0)}


def export_arms(dest: str) -> dict:
    out = {}
    for arm, commit in ARMS.items():
        d = os.path.join(dest, arm)
        os.makedirs(d)
        tar = subprocess.run(["git", "-C", ROOT, "archive", commit, "bigrig_engine"],
                             capture_output=True, check=True).stdout
        subprocess.run(["tar", "-x", "-C", d], input=tar, check=True)
        tree = sh(["git", "-C", ROOT, "rev-parse", f"{commit}:bigrig_engine"]).strip()
        out[arm] = {"commit": commit, "engine_tree": tree, "dir": os.path.realpath(d)}
    return out


def paired(rows, regime, num, den, key="decode_tok_s") -> dict:
    by = {(r["round"], r["arm"]): r[key] for r in rows if r["regime"] == regime and r["round"] >= 1}
    rounds = sorted({rd for rd, _ in by if (rd, num) in by and (rd, den) in by})
    d = [math.log(by[(rd, num)]) - math.log(by[(rd, den)]) for rd in rounds]
    if len(d) < 2:
        return {"n": len(d)}
    m, s = statistics.mean(d), statistics.stdev(d)
    h = t_quantile(1 - ALPHA / 2, len(d) - 1) * s / math.sqrt(len(d))
    return {"ratio": round(math.exp(m), 4), "lo": round(math.exp(m - h), 4),
            "hi": round(math.exp(m + h), 4), "n": len(d), "sd_log": round(s, 5)}


def peak_diff(rows, regime, num, den) -> float:
    by = {(r["round"], r["arm"]): r["peak_mem_gb"] for r in rows if r["regime"] == regime and r["round"] >= 1}
    return round(statistics.median(by[(rd, num)] - by[(rd, den)] for rd, a in by
                                   if a == num and (rd, den) in by), 3)


def analyse(rows) -> dict:
    out = {"comparisons": {}, "arms": {}}
    for regime, _, _ in REGIMES:
        for num, den in (("warm", "base"), ("warm_fast", "warm"), ("warm_fast", "base")):
            k = f"{num}_vs_{den}_{regime}"
            out["comparisons"][k] = {"decode": paired(rows, regime, num, den),
                                     "ttft": paired(rows, regime, num, den, "ttft_s"),
                                     "peak_diff_gb": peak_diff(rows, regime, num, den)}
        for arm in ARMS:
            rs = [r for r in rows if r["regime"] == regime and r["arm"] == arm and r["round"] >= 1]
            out["arms"][f"{arm}_{regime}"] = {
                "n": len(rs), "exact_all": all(r["exact"] == 1 for r in rs),
                **{f"median_{k}": round(statistics.median(r[k] for r in rs), 4)
                   for k in ("decode_tok_s", "ttft_s", "peak_mem_gb")}}

    c = out["comparisons"]

    def verdict(name, num, den, min_ratio, needs=None):
        why = []
        if needs and out["decisions"][needs] != "ship":
            why.append(f"{needs} did not ship, so this was only measured on top of it")
        main, guard = c[f"{num}_vs_{den}_16gb"], c[f"{num}_vs_{den}_24gb_warm"]
        if main["decode"]["ratio"] < min_ratio:
            why.append(f"16gb decode ratio {main['decode']['ratio']} < {min_ratio}")
        if main["decode"]["lo"] <= 1.0:
            why.append(f"16gb CI lower bound {main['decode']['lo']} <= 1.00")
        if guard["decode"]["lo"] < 0.97:
            why.append(f"24gb_warm CI lower bound {guard['decode']['lo']} < 0.97")
        for g, cmp in (("16gb", main), ("24gb_warm", guard)):
            if cmp["ttft"]["ratio"] > 1.05:
                why.append(f"{g} ttft ratio {cmp['ttft']['ratio']} > 1.05")
            if cmp["peak_diff_gb"] > 0.10:
                why.append(f"{g} peak +{cmp['peak_diff_gb']} GB > 0.10")
        if not all(out["arms"][f"{a}_{g}"]["exact_all"] for a in (num, den) for g, _, _ in REGIMES):
            why.append("a reply was not exact")
        out["decisions"][name] = "ship" if not why else "no-ship"
        out["decisions"][name + "_reasons"] = why

    out["decisions"] = {}
    verdict("warm", "warm", "base", 1.03)
    verdict("warm_fast", "warm_fast", "warm", 1.02, needs="warm")
    return out


def main():
    res_path = os.environ.get("LAB_RESULTS") or sys.argv[1]
    work = os.environ.get("LAB_RUN_DIR") or tempfile.mkdtemp()
    start = mac_state()
    res = {"estimator_id": ESTIMATOR_ID, "card": CARD, "complete": False, "mac_start": start,
           "regimes": {g: {"flags": f, "rounds": n} for g, f, n in REGIMES}, "rows": []}

    def save():
        with open(res_path, "w") as f:
            json.dump(res, f, indent=1)

    def stop(reason):
        res["stopped_reason"] = reason
        res["mac_end"] = mac_state()
        save()
        print(f"STOPPED: {reason}", flush=True)
        sys.exit(3)

    if not start["ac_power"]:
        stop("not on AC power (battery throttles the GPU about 1.7x)")
    res["arms"] = export_arms(os.path.join(work, "arms"))
    env = {k: v for k, v in os.environ.items()
           if not k.startswith("BIGRIG_") and k not in ("LAB_RESULTS", "LAB_RUN_DIR")}
    failures = {}
    capacity = {}
    for regime, flags, n_rounds in REGIMES:
        for rd in range(n_rounds + 1):
            order = ORDERS[(rd - 1) % len(ORDERS)] if rd else ORDERS[0]
            for pos, arm in enumerate(order):
                st = mac_state()
                if not st["ac_power"]:
                    stop("power left AC")
                if st["pressure"] >= 4:
                    stop("memory pressure critical")
                if st["swap_used_gb"] - start["swap_used_gb"] > SWAP_GROWTH_MAX_GB:
                    stop(f"swap grew {st['swap_used_gb'] - start['swap_used_gb']:.2f} GB")
                tag = f"{regime}-r{rd:02d}-{arm}"
                out, prov = (os.path.join(work, f"{tag}.{x}.json") for x in ("result", "prov"))
                t0 = time.time()
                try:
                    p = subprocess.run([sys.executable, "-c", SHIM, res["arms"][arm]["dir"], prov, CLIMB,
                                        "--out", out, *flags], cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                       timeout=RUN_TIMEOUT_S)
                    ok = p.returncode == 0 and os.path.exists(out)
                except subprocess.TimeoutExpired:
                    ok = False
                wall = round(time.time() - t0, 1)
                if not ok:
                    failures[arm] = failures.get(arm, 0) + 1
                    print(f"{tag}: FAILED ({wall} s)", flush=True)
                    if failures[arm] >= 2:
                        stop(f"arm {arm} failed twice")
                    continue
                files = json.load(open(prov)).values()
                if not all(f and os.path.realpath(f).startswith(res["arms"][arm]["dir"] + os.sep) for f in files):
                    stop(f"{tag} loaded bigrig_engine from outside its arm")
                r = json.load(open(out))
                row = {"regime": regime, "round": rd, "pos": pos, "arm": arm, "wall_s": wall,
                       "mac_before": st, **r}
                res["rows"].append(row)
                save()
                print(f"{tag}: {r['decode_tok_s']} tok/s, ttft {r['ttft_s']} s, exact {r['exact']}", flush=True)
                if r["exact"] != 1:
                    stop(f"{tag} reply not exact")
                if not flags and r["balloon_end_gb"] < BALLOON_MIN_GB:
                    stop(f"{tag} balloon squeezed to {r['balloon_end_gb']} GB")
                if capacity.setdefault(regime, r["capacity"]) != r["capacity"]:
                    stop(f"{tag} capacity {r['capacity']} differs from {capacity[regime]}")
    res.update(analyse(res["rows"]))
    res["complete"] = True
    res["mac_end"] = mac_state()
    res["swap_growth_gb"] = round(res["mac_end"]["swap_used_gb"] - start["swap_used_gb"], 3)
    save()
    print(json.dumps({"comparisons": res["comparisons"], "decisions": res["decisions"]}, indent=1))


if __name__ == "__main__":
    main()
