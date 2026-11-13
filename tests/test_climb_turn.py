"""bench/climb_turn.py, the climb-2 eval: its arithmetic, its gates, and one end-to-end run on fake engines.

WHY THIS FILE EXISTS
    The eval decides every keep or revert of an overnight climb and is locked before it starts,
    so a bug in it cannot be fixed mid-climb. Each gate is tested by feeding it the failure it
    exists to catch. The end-to-end test runs the real parent/worker plumbing (fresh processes,
    pinned plan, module provenance, timing) against two tiny fake `bigrig_engine` packages, so it
    needs no model, no MLX and no balloon.

Run: .venv/bin/python tests/test_climb_turn.py
"""
import importlib.util
import os
import sys
import tempfile
import textwrap

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location("climb_turn", os.path.join(ROOT, "bench", "climb_turn.py"))
ct = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ct)

FAIL = []


def check(n, c, d=""):
    (print(f"  PASS  {n}") if c else (FAIL.append(n), print(f"  FAIL  {n}  {d}")))


def near(a, b, tol=1e-3):
    return abs(a - b) <= tol


LONG_TEXT = "BigRig runs large mixture-of-experts models on small Macs by streaming experts from the SSD."
SHORT_TEXTS = ["A heat pump moves heat " + "from outside to inside, step by step, " * 3,
               "def fib(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a",
               "From 9:40 to 13:15 is 3 hours and 35 minutes; 9:40 to 10:00 is 20 minutes, then 3 h 15 min."]
TOKENS = [40, 128, 128, 128]


def run(tag, arm, scale=1.0, texts=None, plan=None, modules=None, peak=3.0, balloon=(5.0, 5.0)):
    """One fabricated run: 10 s per reply at scale 1, the given replies, plan and provenance."""
    texts = texts or [LONG_TEXT] + SHORT_TEXTS
    rows = [{"text": t, "tokens": n, "prompt_tokens": 30, "turn_s": 10.0 * scale, "ttft_s": 2.0 * scale,
             "decode_tokens": n - 1, "decode_s": 8.0 * scale} for t, n in zip(texts, TOKENS)]
    return {"tag": tag, "arm": arm, "rows": rows, "plan": plan or dict(ct.PIN),
            "modules": modules or {"bigrig_engine": arm + "/bigrig_engine/__init__.py"},
            "peak_gb": peak, "balloon_start_gb": balloon[0], "balloon_end_gb": balloon[1]}


def four(cand_scale=0.5, **cand):
    """Four runs in eval order; keyword arguments change the candidate's two runs only."""
    c = dict(scale=cand_scale, **cand)
    return [run("base", "/arms/base"), run("cand", "/arms/cand", **c),
            run("cand", "/arms/cand", **c), run("base", "/arms/base")]


print("=" * 78)
print("1. THE METRIC")
print("=" * 78)
res = ct.analyse(four(cand_scale=0.5), swap_growth_gb=0.0)
check("half the turn time is a 2.00x speedup", near(res["speedup"], 2.0), res["speedup"])
check("decode ratio is candidate over base", near(res["decode_ratio"], 2.0), res["decode_ratio"])
check("long-prompt first-token ratio is candidate over base", near(res["ttft_long_ratio"], 0.5))
check("all gates pass on clean runs",
      (res["exact"], res["healthy"], res["plan_ok"], res["balloon_kept"]) == (1, 1, 1, 1), res)
check("estimator_id is written", res["estimator_id"] == ct.ESTIMATOR_ID)
runs = four(cand_scale=1.0)
runs[1] = run("cand", "/arms/cand", scale=0.5)            # pair 1 twice as fast, pair 2 equal
check("speedup is the geometric mean of the pairs (sqrt 2)",
      near(ct.analyse(runs, 0.0)["speedup"], 2 ** 0.5), ct.analyse(runs, 0.0)["speedup_pairs"])

print("=" * 78)
print("2. EACH GATE CATCHES THE FAILURE IT EXISTS FOR")
print("=" * 78)
changed = [LONG_TEXT, SHORT_TEXTS[0], SHORT_TEXTS[1], SHORT_TEXTS[2].replace("35", "36")]
check("one changed candidate reply breaks exact", ct.analyse(four(texts=changed), 0.0)["exact"] == 0)
runs = four()
runs[3] = run("base", "/arms/base", texts=[LONG_TEXT + "!"] + SHORT_TEXTS)
runs[2] = run("cand", "/arms/cand", scale=0.5, texts=[LONG_TEXT + "!"] + SHORT_TEXTS)
check("a base that disagrees with itself breaks exact", ct.analyse(runs, 0.0)["exact"] == 0)
check("a different plan fails plan_ok",
      ct.analyse(four(plan={"capacity": 11, "full_layers": 3}), 0.0)["plan_ok"] == 0)
foreign = {"bigrig_engine": "/arms/base/bigrig_engine/__init__.py"}
check("candidate code loaded from the base tree fails plan_ok",
      ct.analyse(four(modules=foreign), 0.0)["plan_ok"] == 0)
check("a balloon squeezed to 88% fails balloon_kept",
      ct.analyse(four(balloon=(5.0, 4.4)), 0.0)["balloon_kept"] == 0)
looping = [LONG_TEXT, "heat " + "the coil gets cold. " * 10, SHORT_TEXTS[1], SHORT_TEXTS[2]]
check("a looping reply fails healthy", ct.analyse(four(texts=looping), 0.0)["healthy"] == 0)
check("a reply without its keyword fails healthy",
      ct.analyse(four(texts=[LONG_TEXT.replace("BigRig", "It")] + SHORT_TEXTS), 0.0)["healthy"] == 0)
runs = four()
runs[1]["rows"][1]["tokens"] = 50
check("a short-prompt reply under 64 tokens fails healthy", ct.analyse(runs, 0.0)["healthy"] == 0)
check("a 40-token summary passes healthy (it may end early)", ct.analyse(four(), 0.0)["healthy"] == 1)
runs = four()
runs[2]["peak_gb"] = 3.4
check("peak_over_base_gb is the worst pair", near(ct.analyse(runs, 0.0)["peak_over_base_gb"], 0.4))
runs = four()
runs[0], runs[1] = runs[1], runs[0]
try:
    ct.analyse(runs, 0.0)
    check("runs out of order are refused", False)
except ValueError:
    check("runs out of order are refused", True)

print("=" * 78)
print("3. END TO END ON FAKE ENGINES (real processes, no model)")
print("=" * 78)
# The fake reports the pinned plan only when its engine home is the one the caller set, so a
# worker that does not inherit BIGRIG_HOME fails plan_ok (run_arm's launch feeds climb_turn,
# confirm_turn and same_answers' speed guard alike).
FAKE_SESSION = textwrap.dedent('''
    import os, time
    TEXT = {"Summarize": %r, "heat": %r, "Fibonacci": %r, "9:40": %r, "rivers": "Danube, Rhine and Seine."}
    DELAY = %r
    HOME = %r

    class Session:
        def __init__(self, model_dir, budget_gb=None, capacity=None, full_layers=None, verbose=True):
            self.capacity, self.full = capacity, tuple(full_layers or ())

        def stats(self):
            home_ok = os.environ.get("BIGRIG_HOME") == HOME
            return {"capacity": self.capacity if home_ok else -1, "full_layers": len(self.full)}

        def stream_text(self, messages, max_tokens=128, temperature=0.0, seed=0, think=False):
            prompt = messages[-1]["content"]
            words = next(v for k, v in TEXT.items() if k in prompt).split(" ")
            time.sleep(DELAY * 10)                                       # "prefill"
            for i in range(40 if "Summarize" in prompt else max_tokens):
                time.sleep(DELAY)
                yield (words[i] + " " if i < len(words) else ""), {"generation_tokens": i + 1,
                                                                    "prompt_tokens": 30}

        def close(self):
            pass
''')


def fake_arm(parent, name, delay):
    pkg = os.path.join(parent, name, "bigrig_engine")
    os.makedirs(pkg)
    open(os.path.join(pkg, "__init__.py"), "w").close()
    with open(os.path.join(pkg, "session.py"), "w") as f:
        f.write(FAKE_SESSION % (LONG_TEXT, *SHORT_TEXTS, delay, CALLER_HOME))
    return os.path.join(parent, name)


CALLER_HOME = "/the/engine/home/the/caller/set"
saved_home = os.environ.get("BIGRIG_HOME")
os.environ["BIGRIG_HOME"] = CALLER_HOME
with tempfile.TemporaryDirectory() as tmp:
    base, cand = fake_arm(tmp, "base", 0.004), fake_arm(tmp, "cand", 0.002)
    res = ct.run_eval(base, cand, balloon_gb=0, python=sys.executable)
    check("gates pass end to end", (res["exact"], res["healthy"], res["plan_ok"]) == (1, 1, 1),
          {k: res[k] for k in ("exact", "healthy", "plan_ok")})
    check("...so every worker inherited the caller's engine home (the fake's plan depends on it)",
          res["plan_ok"] == 1)
    check("a 2x faster fake measures between 1.4x and 2.6x", 1.4 < res["speedup"] < 2.6, res["speedup"])
    check("runs happen in base/cand/cand/base order", [r["tag"] for r in res["runs"]] == list(ct.ORDER))
    real = {os.path.realpath(base), os.path.realpath(cand)}
    check("every run used its own arm", all(os.path.realpath(r["arm"]) in real for r in res["runs"]))
    # The same plumbing must notice a candidate that answers differently.
    with open(os.path.join(cand, "bigrig_engine", "session.py")) as f:
        src = f.read()
    with open(os.path.join(cand, "bigrig_engine", "session.py"), "w") as f:
        f.write(src.replace("3 hours and 35 minutes", "3 hours and 36 minutes"))
    check("end to end, a changed reply fails exact", ct.run_eval(base, cand, balloon_gb=0)["exact"] == 0)
    os.environ["BIGRIG_HOME"] = "/a/different/home"
    check("a worker under another engine home fails plan_ok", ct.run_eval(base, cand, balloon_gb=0)["plan_ok"] == 0)
if saved_home is None:
    os.environ.pop("BIGRIG_HOME", None)
else:
    os.environ["BIGRIG_HOME"] = saved_home

print("\n" + "=" * 78)
print(f"{'ALL TESTS PASSED' if not FAIL else str(len(FAIL)) + ' FAILURES: ' + ', '.join(FAIL)}")
print("=" * 78)
sys.exit(1 if FAIL else 0)
