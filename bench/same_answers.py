"""Confirm run for card `same-answers-every-budget`: does BigRig give bit-identical replies at every memory budget?

WHAT IT RUNS (the card is the authority; this file implements it)
    Two models, each through every expert plan its budgets reach, each plan in a fresh process:
        qwen3.6  pinned (whole layers, other layers' capacity) (0,8) (0,11) (0,16) (1,12) (3,11)
                 (5,12) (6,11) at budget 9.0, plus the planner's own plan at 6.1 / 7.5 / 9.0 GB
        olmoe    pinned (0,8) (0,12) (4,10) (9,13) (14,24) and (0,64) = every expert resident, at
                 budget 5.0, plus the planner's plan at 2.0 / 3.0 / 5.0 GB
    Every cell answers P1-P5 greedily, 128 tokens: the three climb_decode prompts, the 1,771-token
    README summary, and a two-turn conversation. Every primary reply is read from an empty prompt
    cache; turn 2 is also read once with turn 1 still cached (P5_hit, reported, not decided on).

    Candidate (the branch head, exported from git): every condition twice, the second pass in
    reverse order. Base (released 6e87fcd): five cells once, to record today's behaviour and
    prove the comparison catches a real difference. Then the speed guard: 6 rounds of the locked
    climb_turn workload at 6.1 GB with the 5 GB balloon, base and candidate alternating.

ESTIMATOR (same-answers-v1)
    Primary: per model and prompt, the reply token ids of every candidate cell against the first
    cell, exact equality. Speed guard: d_r = ln(candidate) - ln(base) per
    round for decode tok/s (P1-P3) and for P4's time to first token; ratio = exp(mean d), 95% CI
    = exp(mean d +- t(0.975, 5) sd(d) / sqrt(6)). Decision: ships / opt-in / no-claim (card rule).
"""
import argparse
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location("climb_turn", os.path.join(ROOT, "bench", "climb_turn.py"))
ct = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ct)          # locked measurement helpers; this file never imports the engine

CARD = "same-answers-every-budget"
ESTIMATOR_ID = "same-answers-v1"
BASE_COMMIT = ct.BASE_COMMIT
CANDIDATE_REF = "feat/same-answers"
MAX_TOKENS = 128
RUN_TIMEOUT_S = 900
SPEED_ROUNDS = 6
FOLLOW_UP = "Now explain what happens when the outdoor coil ices up."
MODELS = {
    "qwen3.6": {"dir": os.path.join(ROOT, "models", "Qwen3.6-35B-A3B-4bit"), "checked": True,
                "pinned": [(0, 8), (0, 11), (0, 16), (1, 12), (3, 11), (5, 12), (6, 11)],
                "pinned_budget": 9.0, "budgets": [6.1, 7.5, 9.0]},
    "olmoe": {"dir": os.path.join(ROOT, "models", "OLMoE-1B-7B-0125-4bit"), "checked": False,
              "pinned": [(0, 8), (0, 12), (4, 10), (9, 13), (14, 24), (0, 64)],
              "pinned_budget": 5.0, "budgets": [2.0, 3.0, 5.0]},
}
BASE_CELLS = {"qwen3.6": ["pin 0,8", "pin 3,11", "budget 9.0"], "olmoe": ["pin 0,8", "pin 0,64"]}
PROMPT_NAMES = ["P1", "P2", "P3", "P4", "P5"]
KEYWORDS = {"P1": ("heat",), "P2": ("def fib",), "P3": ("9:40",), "P4": ("BigRig",), "P5": ("ice", "defrost")}


def conditions(model: str) -> list:
    m = MODELS[model]
    pinned = [{"name": f"pin {f},{c}", "full": f, "capacity": c, "budget": m["pinned_budget"]} for f, c in m["pinned"]]
    return pinned + [{"name": f"budget {b}", "budget": b} for b in m["budgets"]]


# ---------------------------------------------------------------- worker (fresh process per cell)

def reply(session, messages) -> dict:
    text = "".join(chunk for chunk, _info in session.stream_text(messages, max_tokens=MAX_TOKENS, temperature=0.0,
                                                                 seed=0, think=False))
    return {"ids": list(session._generated_ids), "text": text}


def worker(arm: str, model: str, cond: dict, out: str) -> None:
    sys.path.insert(0, arm)
    import bigrig_engine  # noqa: F401 -- first, so every submodule resolves inside `arm`
    from bigrig_engine.session import Session

    kw = {"budget_gb": cond["budget"], "verbose": False}
    if "capacity" in cond:
        kw.update(capacity=cond["capacity"], full_layers=tuple(range(cond["full"])))
    s = Session(MODELS[model]["dir"], **kw)
    st = s.stats()
    user = lambda t: {"role": "user", "content": t}                                   # noqa: E731

    def cold(messages):
        # Every primary reply is read from an empty prompt cache, so it depends on the plan and
        # nothing else: how much a cache holds is itself budget-dependent, and a hit re-chunks the
        # prompt (chunk invariance, reported separately as P5_hit).
        s.trim_prompt_cache(0)
        return reply(s, messages)

    rows = {name: cold([user(p)]) for name, (p, _k) in zip(PROMPT_NAMES[:3], ct.SHORT_PROMPTS)}
    rows["P4"] = cold([user(ct.long_prompt())])
    first = cold([user(ct.SHORT_PROMPTS[0][0])])
    convo = [user(ct.SHORT_PROMPTS[0][0]), {"role": "assistant", "content": first["text"]}, user(FOLLOW_UP)]
    hits0 = s.prompt_cache_hits
    rows["P5_hit"] = reply(s, convo)                  # turn 1 is still cached: a hit if it fit
    rows["P5_hit"]["cache_hit"] = s.prompt_cache_hits > hits0
    rows["P5"] = cold(convo)
    s.close()
    modules = {n: getattr(mod, "__file__", None) for n, mod in list(sys.modules.items())
               if n == "bigrig_engine" or n.startswith("bigrig_engine.")}
    try:
        from bigrig_engine import stream
        blob = stream.expert_source(MODELS[model]["dir"])[1]   # the session's own choice of source
    except ImportError:                                    # the tests' fake engines have no stream
        blob = None
    with open(out, "w") as f:
        json.dump({"rows": rows, "modules": modules, "blob": blob,
                   "plan": {"capacity": st.get("capacity"), "full_layers": st.get("full_layers")}}, f)


# ---------------------------------------------------------------- one cell, watched from the parent

def run_cell(arm: str, model: str, cond: dict, python: str = sys.executable) -> dict:
    fd, out = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    swap0 = ct.mac_state()["swap_used_gb"]
    p = subprocess.Popen([python, os.path.abspath(__file__), "--worker", arm, model, json.dumps(cond), out],
                         stdin=subprocess.DEVNULL)
    t0, killed, stop = time.perf_counter(), [None], threading.Event()

    def watch():
        while not stop.wait(2.0):
            why = ct.danger(swap0) or (f"timeout {RUN_TIMEOUT_S} s" if time.perf_counter() - t0 > RUN_TIMEOUT_S else None)
            if why:
                killed[0] = why
                p.kill()
                return

    th = threading.Thread(target=watch, daemon=True)
    th.start()
    code = p.wait()
    stop.set()
    th.join()
    try:
        if code != 0 or killed[0]:
            raise RuntimeError(f"{model} {cond['name']} on {arm}: exit {code}, {killed[0] or 'no kill'}")
        with open(out) as f:
            res = json.load(f)
    finally:
        os.unlink(out)
    res.update(arm=arm, model=model, cond=cond["name"], pin=cond.get("capacity") is not None,
               want_plan={"capacity": cond.get("capacity"), "full_layers": cond.get("full")},
               wall_s=round(time.perf_counter() - t0, 1))
    return res


# ---------------------------------------------------------------- analysis (pure; unit-tested)

def healthy(model: str, name: str, row: dict) -> bool:
    if not MODELS[model]["checked"]:
        return len(row["ids"]) >= 16
    flat = " ".join(row["text"].split())
    loops = any(flat.count(flat[i:i + 20]) >= 5 for i in range(0, max(0, len(flat) - 20), 10))
    floor = 16 if name == "P4" else 64
    return any(k in row["text"] for k in KEYWORDS[name.split("_")[0]]) and len(row["ids"]) >= floor and not loops


def cell_problems(cell: dict) -> list:
    """Reasons a cell invalidates the job (card kill criteria): wrong plan, foreign code, unhealthy reply."""
    why = []
    if cell["pin"] and cell["plan"] != cell["want_plan"]:
        why.append(f"plan {cell['plan']} != pinned {cell['want_plan']}")
    arm = os.path.realpath(cell["arm"])
    if not cell["modules"] or any(f and not os.path.realpath(f).startswith(arm + os.sep) for f in cell["modules"].values()):
        why.append("bigrig_engine loaded from outside its arm")
    if cell.get("blob") == "":
        why.append("experts read from the safetensors shards, not the packed blob users run")
    why += [f"unhealthy {n}" for n, row in cell["rows"].items() if not healthy(cell["model"], n, row)]
    return why


def first_diff(a: list, b: list):
    n = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)
    return n if n is not None else (None if len(a) == len(b) else min(len(a), len(b)))


def paired(xs: list) -> dict:
    """ratio and 95% CI of exp(mean d) for per-round log ratios d."""
    n, mean = len(xs), sum(xs) / len(xs)
    sd = math.sqrt(sum((x - mean) ** 2 for x in xs) / (n - 1))
    half = t_quantile(0.975, n - 1) * sd / math.sqrt(n)
    return {"ratio": round(math.exp(mean), 4), "lo": round(math.exp(mean - half), 4),
            "hi": round(math.exp(mean + half), 4), "n": n, "sd_log": round(sd, 5)}


def t_quantile(p: float, df: int) -> float:
    """Student-t quantile by Simpson integration of the density plus bisection (no scipy here)."""
    c = math.exp(math.lgamma((df + 1) / 2) - math.lgamma(df / 2)) / math.sqrt(df * math.pi)
    pdf = lambda x: c * (1 + x * x / df) ** (-(df + 1) / 2)                          # noqa: E731

    def cdf(t, n=4000):
        h = t / n
        return 0.5 + (pdf(0) + pdf(t) + sum((4 if i % 2 else 2) * pdf(i * h) for i in range(1, n))) * h / 3

    lo, hi = 0.0, 50.0
    for _ in range(80):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if cdf(mid) < p else (lo, mid)
    return (lo + hi) / 2


def analyse(cand: list, base: list, speed: list) -> dict:
    """cand/base: finished cells; speed: [(base run, candidate run)] from climb_turn.run_arm. Pure."""
    differing, total, secondary = [], 0, {"p5_cold_vs_hit": [], "cand_vs_base_match": {}}
    for model in MODELS:
        cells = [c for c in cand if c["model"] == model]
        if not cells:
            continue
        ref = cells[0]
        for c in cells:
            for name in PROMPT_NAMES:
                total += 1
                d = first_diff(ref["rows"][name]["ids"], c["rows"][name]["ids"])
                if d is not None:
                    differing.append({"model": model, "prompt": name, "cond": c["cond"], "pass": c.get("pass"),
                                      "first_diff": d})
            d = first_diff(c["rows"]["P5"]["ids"], c["rows"]["P5_hit"]["ids"])
            secondary["p5_cold_vs_hit"].append({"model": model, "cond": c["cond"], "pass": c.get("pass"),
                                                "cache_hit": c["rows"]["P5_hit"].get("cache_hit"), "first_diff": d})
        b0 = next((b for b in base if b["model"] == model and b["cond"] == "pin 0,8"), None)
        if b0 is not None:
            secondary["cand_vs_base_match"][model] = {
                n: round(sum(x == y for x, y in zip(ref["rows"][n]["ids"], b0["rows"][n]["ids"])) /
                         max(1, len(b0["rows"][n]["ids"])), 4) for n in PROMPT_NAMES}
    base_diffs = []
    for model in MODELS:
        bc = [b for b in base if b["model"] == model]
        for b in bc[1:]:
            for name in PROMPT_NAMES:
                d = first_diff(bc[0]["rows"][name]["ids"], b["rows"][name]["ids"])
                if d is not None:
                    base_diffs.append({"model": model, "prompt": name, "cond": b["cond"], "first_diff": d})
    healthy_all = all(not [w for w in cell_problems(c) if w.startswith("unhealthy")] for c in cand)
    dec = lambda run: sum(r["decode_tokens"] for r in run["rows"][1:4]) / sum(r["decode_s"] for r in run["rows"][1:4])
    speed_res = {}
    if speed:
        speed_res = {"decode": paired([math.log(dec(c) / dec(b)) for b, c in speed]),
                     "ttft_long": paired([math.log(c["rows"][0]["ttft_s"] / b["rows"][0]["ttft_s"]) for b, c in speed])}
    invariant = not differing and healthy_all and total > 0
    fast = bool(speed_res) and speed_res["decode"]["lo"] >= 0.95 and speed_res["ttft_long"]["hi"] <= 1.05
    return {"decision": "ships" if invariant and fast else ("opt-in" if invariant else "no-claim"),
            "cells_compared": total, "cells_differing": len(differing), "differing": differing,
            "healthy": int(healthy_all), "speed": speed_res, "base_cells_differing": base_diffs,
            "secondary": secondary, "estimator_id": ESTIMATOR_ID, "card": CARD}


# ---------------------------------------------------------------- the job

def run_job(base_arm: str, cand_arm: str, models=None, speed_rounds: int = SPEED_ROUNDS,
            balloon_gb: float = ct.BALLOON_GB, python: str = sys.executable) -> dict:
    models = models or list(MODELS)

    def checked(cell):
        why = cell_problems(cell)
        if why:
            raise SystemExit(f"STOP ({cell['model']} {cell['cond']}): " + "; ".join(why))
        return cell

    base = [checked(run_cell(base_arm, m, c, python)) for m in models for c in conditions(m)
            if c["name"] in BASE_CELLS[m]]
    cand = []
    for pass_no in (1, 2):
        for m in models:
            conds = conditions(m) if pass_no == 1 else conditions(m)[::-1]
            for c in conds:
                cell = checked(run_cell(cand_arm, m, c, python))
                cell["pass"] = pass_no
                cand.append(cell)
    speed = []
    for r in range(speed_rounds + 1):                                   # round 0 is the warm-up
        order = ("base", "cand") if r % 2 == 0 else ("cand", "base")
        got = {tag: ct.run_arm(base_arm if tag == "base" else cand_arm, balloon_gb, python) for tag in order}
        if r > 0:
            speed.append((got["base"], got["cand"]))
    return analyse(cand, base, speed)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=os.environ.get("LAB_RESULTS"))
    ap.add_argument("--candidate", default=CANDIDATE_REF, help="commit or branch to test")
    ap.add_argument("--worker", nargs=4, help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.worker:
        arm, model, cond, out = a.worker
        worker(arm, model, json.loads(cond), out)
        return
    if not a.out:
        sys.exit("set LAB_RESULTS or --out")
    # THE ENGINE'S HOME, PINNED FOR EVERY WORKER. Each arm is a `git archive` export with no
    # pyproject.toml beside it, so bigrig_engine.home() would resolve to ~/.bigrig: no packed blob
    # there, so the engine would read experts from the safetensors shards, a slower path with
    # different kernels that users never run (the CLI packs on first run). Found 2026-11-13 when
    # two identical engines scored 2.08x apart in the climb_turn baseline. Workers inherit this,
    # including climb_turn.run_arm's for the speed guard; cell_problems stops on a cell without
    # the blob.
    os.environ["BIGRIG_HOME"] = ROOT
    ct.wait_quiet()
    cand_commit = ct.sh(["git", "-C", ROOT, "rev-parse", a.candidate]).strip()
    tmp = tempfile.mkdtemp(prefix="same_answers_")
    try:
        res = run_job(ct.export_tree(BASE_COMMIT, os.path.join(tmp, "base")),
                      ct.export_tree(cand_commit, os.path.join(tmp, "cand")))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    res.update(candidate_commit=cand_commit,
               candidate_tree=ct.sh(["git", "-C", ROOT, "rev-parse", f"{cand_commit}:bigrig_engine"]).strip(),
               base_commit=BASE_COMMIT)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: res[k] for k in ("decision", "cells_compared", "cells_differing", "healthy", "speed")}))


if __name__ == "__main__":
    main()
