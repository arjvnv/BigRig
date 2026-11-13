"""bench/confirm_turn.py, card turn-speedup-confirm: pair arithmetic, decision rule, home pinning.

WHY THIS FILE EXISTS
    The card's decision is computed by this script, so it is tested on the cases it must tell
    apart before any real run: a clear speed-up (ships), one inside the noise or under the
    threshold (no-ship), a decode regression, an inexact reply, too much memory. The plumbing is
    run with climb_turn's evaluation stubbed out, to show every worker would see the checkout as
    its engine home (the defect this harness exists to avoid).

Run: .venv/bin/python tests/test_confirm_turn.py
"""
import importlib.util
import json
import math
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location("confirm_turn", os.path.join(ROOT, "bench", "confirm_turn.py"))
cf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cf)

FAIL = []


def check(n, c, d=""):
    (print(f"  PASS  {n}") if c else (FAIL.append(n), print(f"  FAIL  {n}  {d}")))


def run(turn, dec):
    """A run whose four replies sum to `turn` seconds and decode at `dec` tokens/s."""
    return {"rows": [{"turn_s": turn / 4, "decode_tokens": 100, "decode_s": 100 / dec}] * 4}


def ev(base_turn, cand_turn, base_dec=10.0, cand_dec=10.0, wobble=0.0, **over):
    e = {"runs": [run(base_turn, base_dec), run(cand_turn * (1 + wobble), cand_dec),
                  run(cand_turn * (1 - wobble), cand_dec), run(base_turn, base_dec)],
         "exact": 1, "healthy": 1, "plan_ok": 1, "balloon_kept": 1, "peak_over_base_gb": 0.0,
         "swap_growth_gb": 0.0, "speedup": 1.0, "decode_ratio": 1.0}
    e.update(over)
    return e


print("=" * 78)
print("1. ARITHMETIC")
print("=" * 78)
check("t(0.975, 11) is 2.2010 (table value)", abs(cf.t_quantile(0.975, 11) - 2.2010) < 1e-3, cf.t_quantile(0.975, 11))
p = cf.paired([1.2, 1.3])
# n = 2: sample sd = |d1 - d2| / sqrt(2), half-width = t * sd / sqrt(2) = t * |d1 - d2| / 2.
lo = math.exp((math.log(1.2) + math.log(1.3)) / 2 - 12.706 * abs(math.log(1.3) - math.log(1.2)) / 2)
check("a 2-pair CI uses t(0.975, 1) = 12.706", abs(p["lo"] - lo) < 2e-3, (p, lo))
t, d = cf.pair_ratios(ev(60, 50, 10, 11))
check("pair ratios: base turn / candidate turn, candidate decode / base decode",
      all(abs(x - 1.2) < 1e-9 for x in t) and all(abs(x - 1.1) < 1e-9 for x in d), (t, d))

print("=" * 78)
print("2. THE DECISION RULE")
print("=" * 78)
clear = [ev(64, 50, 12, 13, wobble=0.01 * (-1) ** i) for i in range(6)]
r = cf.decide(clear)
check("a steady 1.28x turn speed-up with faster decode ships", r["decision"] == "ships", r["reasons"])
small = [ev(64, 58, wobble=0.01 * (-1) ** i) for i in range(6)]
check("a steady 1.10x does not (under 1.15)", cf.decide(small)["decision"] == "no-ship")
noisy = [ev(64, 54, wobble=0.25 * (-1) ** i) for i in range(6)]
rn = cf.decide(noisy)
check("a mean over 1.15 with a wide CI does not (lower bound under 1.10)",
      rn["decision"] == "no-ship" and any("CI lower" in w for w in rn["reasons"]), rn["reasons"])
slow = [ev(64, 50, 12, 11.4, wobble=0.01 * (-1) ** i) for i in range(6)]
check("a decode regression past 3% does not", cf.decide(slow)["decision"] == "no-ship")
inexact = clear[:5] + [ev(64, 50, exact=0)]
check("one inexact round does not", "exact_all false" in cf.decide(inexact)["reasons"])
fat = clear[:5] + [ev(64, 50, 12, 13, peak_over_base_gb=0.4)]
check("one round over the memory gate does not", any("peak" in w for w in cf.decide(fat)["reasons"]))

print("=" * 78)
print("3. PLUMBING (climb_turn's evaluation stubbed): EVERY WORKER SEES THE CHECKOUT AS HOME")
print("=" * 78)
seen = []
cf.ct.wait_quiet = lambda: None
cf.ct.export_tree = lambda commit, dest: dest
cf.ct.run_eval = lambda b, c: (seen.append(os.environ.get("BIGRIG_HOME")), ev(64, 50, 12, 13))[1]
saved_home = os.environ.pop("BIGRIG_HOME", None)
_exists = cf.os.path.exists
cf.os.path.exists = lambda p_: True if p_.endswith(".experts") else _exists(p_)   # a worktree has no blob
with tempfile.TemporaryDirectory() as tmp:
    out = os.path.join(tmp, "res.json")
    sys.argv = ["confirm_turn.py", "--out", out, "--candidate", "HEAD", "--rounds", "3"]
    cf.main()
    res = json.load(open(out))
cf.os.path.exists = _exists
if saved_home is not None:
    os.environ["BIGRIG_HOME"] = saved_home
sys.argv = ["confirm_turn.py", "--out", "/dev/null", "--candidate", "HEAD", "--rounds", "1"]
cf.ROOT, real_root = "/nonexistent-checkout", cf.ROOT
try:
    cf.main()
    refused = False
except SystemExit as e:
    refused = "no packed blob" in str(e)
cf.ROOT = real_root
check("a checkout without the packed blob is refused before anything runs", refused)
check("BIGRIG_HOME is the checkout for every round", seen == [cf.ROOT] * 3, seen)
check("...the result names its estimator, home and rounds",
      res["estimator_id"] == cf.ESTIMATOR and res["home"] == cf.ROOT and res["rounds"] == 3)

print("\n" + "=" * 78)
print(f"{'ALL TESTS PASSED' if not FAIL else str(len(FAIL)) + ' FAILURES: ' + ', '.join(FAIL)}")
print("=" * 78)
sys.exit(1 if FAIL else 0)
