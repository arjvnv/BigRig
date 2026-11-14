"""bench/same_answers.py, card same-answers-every-budget: comparison, decision rule and plumbing, on fake engines.

WHY THIS FILE EXISTS
    The card's decision is computed by this script, so the script is tested against the cases it
    must tell apart before any real run: an engine whose replies never change (ships / opt-in),
    one whose reply moves with its expert plan (no-claim, with the differing cell named), a wrong
    plan or foreign code (the job stops), and the speed guard's thresholds. The end-to-end part
    runs the real parent/worker plumbing against fake `bigrig_engine` packages; no model needed.

Run: .venv/bin/python tests/test_same_answers.py
"""
import importlib.util
import os
import sys
import tempfile
import textwrap

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location("same_answers", os.path.join(ROOT, "bench", "same_answers.py"))
sa = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sa)

FAIL = []


def check(n, c, d=""):
    (print(f"  PASS  {n}") if c else (FAIL.append(n), print(f"  FAIL  {n}  {d}")))


FAKE_SESSION = textwrap.dedent('''
    import time
    WORDS = {"heat": "heat", "Fibonacci": "def fib", "9:40": "9:40", "Summarize": "BigRig", "ices up": "ice",
             "rivers": "rivers"}
    PLAN_DEPENDENT = %r
    SAME_ANSWERS = %r

    class Session:
        # Reports its plan as the real engine does: a layer holding every expert counts as whole,
        # so capacity == experts reads back as every layer whole.
        def __init__(self, model_dir, budget_gb=None, capacity=None, full_layers=None, verbose=True):
            self.n_experts, self.n_layers = (64, 16) if "OLMoE" in model_dir else (256, 40)
            self.capacity = capacity if capacity is not None else int(budget_gb * 2)
            self.full = len(tuple(full_layers or ()))
            self.prompt_cache_hits, self._generated_ids, self._cached = 0, [], False

        def stats(self):
            whole = self.n_layers if self.capacity >= self.n_experts else self.full
            return {"capacity": self.capacity, "full_layers": whole, "same_answers": SAME_ANSWERS}

        def trim_prompt_cache(self, to_bytes=0):
            self._cached = False
            return 0

        def stream_text(self, messages, max_tokens=128, temperature=0.0, seed=0, think=False):
            prompt = messages[-1]["content"]
            key = next(w for k, w in WORDS.items() if k in prompt)
            if len(messages) > 1 and self._cached:
                self.prompt_cache_hits += 1
            self._generated_ids = []
            n = 40 if key == "BigRig" else max_tokens
            for i in range(n):
                tok = 1000 + i + (self.capacity if PLAN_DEPENDENT and i > 15 else 0)
                self._generated_ids.append(tok)
                text = (key + " ") if i == 0 else ("w%%d " %% tok)
                time.sleep(0.0005)
                yield text, {"generation_tokens": i + 1, "prompt_tokens": 30}
            self._cached = True

        def close(self):
            pass
''')


def fake_arm(parent, name, plan_dependent, same_answers=True):
    pkg = os.path.join(parent, name, "bigrig_engine")
    os.makedirs(pkg)
    open(os.path.join(pkg, "__init__.py"), "w").close()
    with open(os.path.join(pkg, "session.py"), "w") as f:
        f.write(FAKE_SESSION % (plan_dependent, same_answers))
    return os.path.join(parent, name)


print("=" * 78)
print("1. THE DECISION RULE (synthetic cells)")
print("=" * 78)


def cell(model, cond, ids_tail=(), arm="/a/cand", pass_no=1):
    text = lambda n: sa.KEYWORDS[n.split("_")[0]][0] + " " + " ".join(f"word{i}" for i in range(80))  # noqa: E731
    rows = {n: {"ids": list(range(100)) + list(ids_tail), "text": text(n)} for n in sa.PROMPT_NAMES + ["P5_hit"]}
    rows["P5_hit"]["cache_hit"] = True
    return {"model": model, "cond": cond, "pass": pass_no, "arm": arm, "pin": False, "plan": {}, "want_plan": {},
            "modules": {"bigrig_engine": arm + "/bigrig_engine/__init__.py"}, "rows": rows}


def speed_pair(base_dec, cand_dec, base_ttft, cand_ttft):
    run = lambda dec, ttft: {"rows": [{"ttft_s": ttft, "decode_tokens": 0, "decode_s": 1.0}] +
                             [{"ttft_s": 1.0, "decode_tokens": dec, "decode_s": 1.0}] * 3,
                             "plan": dict(sa.SPEED_PLAN)}
    return run(base_dec, base_ttft), run(cand_dec, cand_ttft)


same = [cell("qwen3.6", f"c{i}") for i in range(4)]
fast = [speed_pair(10, 10 * (1 + 0.002 * (-1) ** r), 80, 80) for r in range(6)]
slow = [speed_pair(10, 8.5 + 0.01 * r, 80, 80) for r in range(6)]
check("invariant + speed guard passes -> ships", sa.analyse(same, [], fast)["decision"] == "ships")
off_plan = [(dict(b, plan={"capacity": 8, "full_layers": 0}), c) for b, c in fast]
r_off = sa.analyse(same, [], off_plan)
check("...but not when the guard's runs did not hold SPEED_PLAN (they priced another plan)",
      r_off["decision"] == "opt-in" and r_off["speed_plan_ok"] == 0, r_off["decision"])
check("invariant + 15% slower decode -> opt-in", sa.analyse(same, [], slow)["decision"] == "opt-in")
moved = same[:3] + [cell("qwen3.6", "c3", ids_tail=[7])]
res = sa.analyse(moved, [], fast)
check("one reply that moves with the plan -> no-claim", res["decision"] == "no-claim", res["decision"])
check("...and the differing cell is named with its first differing position",
      res["differing"][0]["cond"] == "c3" and res["differing"][0]["first_diff"] == 100, res["differing"][:1])
check("first_diff finds the first differing token", sa.first_diff([1, 2, 3], [1, 9, 3]) == 1)
check("first_diff calls a length difference a difference", sa.first_diff([1, 2], [1, 2, 3]) == 2)
check("first_diff of equal lists is None", sa.first_diff([1, 2], [1, 2]) is None)
p = sa.paired([0.0, 0.0, 0.1, -0.1, 0.05, -0.05])
check("paired ratio of symmetric log ratios is 1.0", abs(p["ratio"] - 1.0) < 1e-9, p)
check("t(0.975, 5) is 2.5706", abs(sa.t_quantile(0.975, 5) - 2.5706) < 1e-3, sa.t_quantile(0.975, 5))

print("=" * 78)
print("2. THE JOB STOPS ON A WRONG PLAN, FOREIGN CODE OR AN UNHEALTHY REPLY")
print("=" * 78)
c = cell("qwen3.6", "pin 3,11")
c.update(pin=True, plan={"capacity": 8, "full_layers": 0}, want_plan={"capacity": 11, "full_layers": 3})
check("a plan that is not the pinned one is a stop", any("plan" in w for w in sa.cell_problems(c)))
c = cell("qwen3.6", "x")
c["modules"] = {"bigrig_engine": "/somewhere/else/bigrig_engine/__init__.py"}
check("code loaded from outside the arm is a stop", any("outside" in w for w in sa.cell_problems(c)))
c = cell("qwen3.6", "x")
c["rows"]["P2"]["text"] = "no keyword here " * 20
check("a Qwen reply without its keyword is a stop", "unhealthy P2" in sa.cell_problems(c))
c = cell("qwen3.6", "x")
c["blob"] = ""
check("a cell that read the safetensors shards instead of the packed blob is a stop",
      any("packed blob" in w for w in sa.cell_problems(c)), sa.cell_problems(c))
c["blob"] = "/x/data/blobs/m.experts"
check("...and one that read the packed blob is not", sa.cell_problems(c) == [], sa.cell_problems(c))
c = cell("olmoe", "x")
c["rows"]["P2"]["text"] = "no keyword here " * 20
check("OLMoE (base model) only needs 16 tokens", sa.cell_problems(c) == [], sa.cell_problems(c))
c["rows"]["P3"]["ids"] = list(range(10))
check("...and an OLMoE reply under 16 tokens is still a stop", "unhealthy P3" in sa.cell_problems(c))

print("=" * 78)
print("3. END TO END ON FAKE ENGINES (real processes, no model)")
print("=" * 78)
with tempfile.TemporaryDirectory() as tmp:
    base, steady, moving = (fake_arm(tmp, "base", True, same_answers=None), fake_arm(tmp, "steady", False),
                            fake_arm(tmp, "moving", True))
    res = sa.run_job(base, steady, models=["qwen3.6"], speed_rounds=2, balloon_gb=0)
    check("an engine whose replies never change: no differing cell", res["cells_differing"] == 0, res["differing"][:2])
    check("...over every condition twice (10 x 2 x 5 prompts)", res["cells_compared"] == 100, res["cells_compared"])
    check("...and healthy", res["healthy"] == 1)
    check("the released-style base differs across its own cells (the comparison can fail)",
          len(res["base_cells_differing"]) > 0)
    # Which side of the guard two rounds of fake engines land on depends on timing noise (an idle
    # Mac gave a 0.08% spread, a CI narrow enough to pass), so assert the rule, not an outcome.
    sp = res["speed"]
    guard = sp["decode"]["lo"] >= 0.95 and sp["ttft_long"]["hi"] <= 1.05
    check("...so the decision is the speed guard's: ships if it passes, opt-in if it does not",
          res["decision"] == ("ships" if guard else "opt-in"), (res["decision"], sp))
    res = sa.run_job(base, moving, models=["qwen3.6"], speed_rounds=2, balloon_gb=0)
    check("an engine whose reply moves with capacity: no-claim", res["decision"] == "no-claim", res["decision"])
    check("the speed guard's runs held SPEED_PLAN end to end", res["speed_plan_ok"] == 1, res.get("speed_plan_ok"))
    # OLMoE with every expert resident: the engine reads that back as 16 whole layers.
    every = next(c for c in sa.conditions("olmoe") if c.get("capacity") == 64)
    got = sa.run_cell(steady, "olmoe", every)
    check(f"the OLMoE every-expert pin {every['name']!r} reads back as itself (no wrong-plan stop)",
          not [w for w in sa.cell_problems(got, candidate=True) if "plan" in w], got["plan"])
    old = dict(every, name="pin 0,64", full=0)
    check("...where the earlier (0,64) pin would have stopped the job",
          any("plan" in w for w in sa.cell_problems(sa.run_cell(steady, "olmoe", old))))
    check("a candidate whose engine did not run with SAME_ANSWERS is a stop",
          any("same_answers" in w for w in sa.cell_problems(sa.run_cell(base, "olmoe", every), candidate=True)))

print("=" * 78)
print("4. EVERY BUDGET THE JOB USES STREAMS (real Session, loading stubbed)")
print("=" * 78)
# The planner refuses a budget too small for the model and compresses (with consent) above the
# streaming range; either would end the job. Checked against the engine in this checkout.
if not all(os.path.isdir(m["dir"]) for m in sa.MODELS.values()):
    print("  SKIPPED - needs both models under models/ (no weights are loaded).")
else:
    from bigrig_engine import stream as _stream
    from bigrig_engine.session import Session as _Session

    class _Loaded(Exception):
        pass

    def _no_load(*a, **k):
        raise _Loaded()

    saved = _stream.load_streaming
    _stream.load_streaming = _no_load
    try:
        for name, m in sa.MODELS.items():
            for b in m["budgets"] + [m["pinned_budget"]]:
                try:
                    _Session(m["dir"], budget_gb=b, verbose=False, persist=False, monitor=False)
                    outcome = "constructed"
                except _Loaded:
                    outcome = "streams"
                except Exception as e:                          # noqa: BLE001 -- the outcome is the check
                    outcome = type(e).__name__
                check(f"{name} at {b} GB streams", outcome == "streams", outcome)
    finally:
        _stream.load_streaming = saved

print("\n" + "=" * 78)
print(f"{'ALL TESTS PASSED' if not FAIL else str(len(FAIL)) + ' FAILURES: ' + ', '.join(FAIL)}")
print("=" * 78)
sys.exit(1 if FAIL else 0)
