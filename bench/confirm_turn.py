"""Confirm run for card turn-speedup-confirm: released BigRig vs a candidate, one turn on a 16 GB Mac.

WHY THIS EXISTS
    bench/climb_turn.py (locked) measures two base/candidate pairs per call: enough to steer a
    hill-climb, not enough to back a claim. This runs ROUNDS of it, each a fresh base/cand/cand/base
    sequence of fresh processes, and reports the paired speed-up of a whole turn (prefill and
    decode of the climb's four prompts) with a 95% confidence interval, plus the same for decode
    alone, and the card's decision.

    Every worker's engine home is pinned to this checkout (BIGRIG_HOME). The arms are `git archive`
    exports with no pyproject.toml beside them, so bigrig_engine.home() would otherwise resolve to
    ~/.bigrig, which has no packed blob: the engine would read experts from the safetensors shards,
    a slower path with different kernels that users never run. Found 2026-11-13 when two identical
    engines scored 2.08x apart in the climb_turn baseline.

Run: lab run --mode confirm --card turn-speedup-confirm -- .venv/bin/python bench/confirm_turn.py [--candidate REF]
"""
import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "bench"))
import climb_turn as ct  # noqa: E402

ESTIMATOR = "turn-speedup-paired-v1"
ROUNDS = 6                     # 12 base/candidate pairs
CANDIDATE_REF = "feat/layer-major-prefill"
# The card's decision rule (.lab/cards/turn-speedup-confirm.md). Changing any of these is a new card.
SHIP_RATIO, SHIP_LO = 1.15, 1.10
DECODE_LO = 0.97
PEAK_MAX_GB, SWAP_MAX_GB = 0.25, 0.5


def t_quantile(p: float, df: int) -> float:
    """Student-t quantile by Simpson integration of the density plus bisection (no scipy here)."""
    c = math.gamma((df + 1) / 2) / (math.sqrt(df * math.pi) * math.gamma(df / 2))
    pdf = lambda x: c * (1 + x * x / df) ** (-(df + 1) / 2)  # noqa: E731

    def cdf(x, n=2000):
        h = x / n
        s = pdf(0) + pdf(x) + sum((4 if i % 2 else 2) * pdf(i * h) for i in range(1, n))
        return 0.5 + s * h / 3

    lo, hi = 0.0, 50.0
    for _ in range(100):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if cdf(mid) < p else (lo, mid)
    return (lo + hi) / 2


def paired(ratios: list) -> dict:
    """Geometric-mean ratio and its 95% CI from per-pair ratios (t interval on the log ratios)."""
    d = [math.log(r) for r in ratios]
    n, mean = len(d), sum(d) / len(d)
    sd = math.sqrt(sum((x - mean) ** 2 for x in d) / (n - 1)) if n > 1 else float("inf")
    half = t_quantile(0.975, n - 1) * sd / math.sqrt(n) if n > 1 else float("inf")
    return {"ratio": round(math.exp(mean), 4), "lo": round(math.exp(mean - half), 4),
            "hi": round(math.exp(mean + half), 4), "n": n, "sd_log": round(sd, 5)}


def pair_ratios(ev: dict) -> tuple:
    """(turn speed-ups, decode ratios) of one climb_turn evaluation's two (base, candidate) pairs."""
    runs = ev["runs"]
    turn = lambda run: sum(r["turn_s"] for r in run["rows"])                                 # noqa: E731
    dec = lambda run: sum(r["decode_tokens"] for r in run["rows"]) / sum(r["decode_s"] for r in run["rows"])  # noqa: E731
    pairs = [(runs[0], runs[1]), (runs[3], runs[2])]
    return [turn(b) / turn(c) for b, c in pairs], [dec(c) / dec(b) for b, c in pairs]


def decide(evals: list) -> dict:
    """The card's decision from every round's climb_turn evaluation. Pure."""
    turns, decs = [], []
    for ev in evals:
        t, d = pair_ratios(ev)
        turns += t
        decs += d
    res = {"turn": paired(turns), "decode": paired(decs), "rounds": len(evals),
           "exact_all": all(ev["exact"] == 1 for ev in evals),
           "healthy_all": all(ev["healthy"] == 1 for ev in evals),
           "plan_all": all(ev["plan_ok"] == 1 for ev in evals),
           "balloon_all": all(ev["balloon_kept"] == 1 for ev in evals),
           "peak_over_base_max_gb": max(ev["peak_over_base_gb"] for ev in evals),
           "swap_growth_max_gb": max(ev["swap_growth_gb"] for ev in evals)}
    why = []
    for key in ("exact_all", "healthy_all", "plan_all", "balloon_all"):
        if not res[key]:
            why.append(f"{key} false")
    if res["turn"]["ratio"] < SHIP_RATIO:
        why.append(f"turn speed-up {res['turn']['ratio']} < {SHIP_RATIO}")
    if res["turn"]["lo"] < SHIP_LO:
        why.append(f"turn CI lower bound {res['turn']['lo']} < {SHIP_LO}")
    if res["decode"]["lo"] < DECODE_LO:
        why.append(f"decode CI lower bound {res['decode']['lo']} < {DECODE_LO}")
    if res["peak_over_base_max_gb"] > PEAK_MAX_GB:
        why.append(f"peak over base {res['peak_over_base_max_gb']} GB > {PEAK_MAX_GB}")
    if res["swap_growth_max_gb"] > SWAP_MAX_GB:
        why.append(f"swap growth {res['swap_growth_max_gb']} GB > {SWAP_MAX_GB}")
    res.update(decision="ships" if not why else "no-ship", reasons=why)
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=os.environ.get("LAB_RESULTS"))
    ap.add_argument("--candidate", default=CANDIDATE_REF, help="commit or branch to test")
    ap.add_argument("--rounds", type=int, default=ROUNDS)
    a = ap.parse_args()
    if not a.out:
        sys.exit("set LAB_RESULTS or --out")
    blob = os.path.join(ROOT, "data", "blobs", os.path.basename(ct.MODEL) + ".experts")
    if not os.path.exists(blob):
        sys.exit(f"no packed blob at {blob}: run from the checkout that holds it (a worktree reads the shards)")
    os.environ["BIGRIG_HOME"] = ROOT          # inherited by every worker; see the module docstring
    cand = subprocess.run(["git", "-C", ROOT, "rev-parse", a.candidate], capture_output=True, text=True,
                          check=True).stdout.strip()
    tmp = tempfile.mkdtemp(prefix="confirm_turn_")
    evals = []
    try:
        base_arm = ct.export_tree(ct.BASE_COMMIT, os.path.join(tmp, "base"))
        cand_arm = ct.export_tree(cand, os.path.join(tmp, "cand"))
        for i in range(a.rounds):
            ct.wait_quiet()
            ev = ct.run_eval(base_arm, cand_arm)
            evals.append(ev)
            print(f"round {i + 1}: speedup {ev['speedup']} exact {ev['exact']} decode {ev['decode_ratio']}", flush=True)
            if ev["exact"] != 1 or ev["healthy"] != 1 or ev["plan_ok"] != 1:
                break                         # a kill criterion: the rest cannot rescue it
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    res = decide(evals)
    res.update(estimator_id=ESTIMATOR, base_commit=ct.BASE_COMMIT, candidate_commit=cand,
               candidate_tree=subprocess.run(["git", "-C", ROOT, "rev-parse", f"{cand}:bigrig_engine"],
                                             capture_output=True, text=True).stdout.strip(),
               home=ROOT, evals=evals)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: res[k] for k in ("decision", "reasons", "turn", "decode", "rounds")}))


if __name__ == "__main__":
    main()
