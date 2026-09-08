"""When the planner says no, it must say the true reason and the number that would be yes.

WHY THIS FILE IS ADVERSARIAL
    The refusal used to end "it cannot run on this machine right now". On a 16 GB Mac -- the
    machine this engine is for -- that sentence was false: the default ceiling there is 5.6 GB,
    the flagship needs about 6.1, and at 7 the same planner says GOOD. A new user read it, believed
    it, and had no way to know a flag would have fixed it. Two things must therefore hold and stay
    held: the number printed is the planner's own (a flag it then refuses is worse than no flag),
    and doctor, serve and run print the SAME number for the same model on the same machine.
"""
import io
import os
import re
import subprocess
import sys
from contextlib import redirect_stdout

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from bigrig_engine import autoconfig as A, preflight as P                # noqa: E402
from bigrig_engine import session as S                                   # noqa: E402

FAIL = []


def check(n, c, d=""):
    (print(f"  PASS  {n}") if c else (FAIL.append(n), print(f"  FAIL  {n}  {d}")))


def shape(n_layers=40, n_experts=256, top_k=8, per=1_540_096, ne=1.1):
    return {"n_layers": n_layers, "n_experts": n_experts, "top_k": top_k, "bytes_per_expert": per,
            "expert_gb": per * n_experts * n_layers / 1e9, "non_expert_gb": ne,
            "manifest": {"layers": {str(i): {"n_experts": n_experts, "bytes_per_expert": per,
                                             "spec": {}} for i in range(n_layers)},
                         "total_bytes": per * n_experts * n_layers}}


# Pressure sampling sleeps 0.4 s on every accepted plan; none of these questions is about pressure.
A.under_pressure = lambda: False
_real_machine = A.machine_gb


def machine(total, wall):
    A.machine_gb = lambda: (total, wall)


def refusal(sh, budget, reserve=3.7, reserve_fn=None, **kw):
    try:
        A.choose_capacity(sh["manifest"], budget_gb=budget, top_k=sh["top_k"], reserve_gb=reserve,
                          non_expert_gb=sh["non_expert_gb"], headroom_gb=A.scaled_headroom(budget),
                          reserve_fn=reserve_fn, **kw)
    except A.CeilingRefusal as e:
        return e
    return None


def accepts(sh, budget, reserve=3.7):
    try:
        A.choose_capacity(sh["manifest"], budget_gb=budget, top_k=sh["top_k"], reserve_gb=reserve,
                          non_expert_gb=sh["non_expert_gb"], headroom_gb=A.scaled_headroom(budget))
        return True
    except MemoryError:
        return False


print("=" * 84)
print("1. THE NUMBER IS THE PLANNER'S OWN, AND IT IS TIGHT")
print("=" * 84)
machine(25.8, 19.1)
grid = [shape(), shape(48, 128, 8, 2_064_384, 0.67), shape(23, 128, 6, 5_600_000, 2.4),
        shape(16, 64, 8, 700_000, 0.3), shape(40, 256, 8, 1_540_096, 0.0)]
for sh in grid:
    for reserve in (3.0, 3.7, 3.85):
        need_bisect, _ = P.smallest_ceiling(sh, reserve)
        need_solved = A.smallest_budget_gb(sh, sh["top_k"], reserve, sh["non_expert_gb"])
        tag = f"L{sh['n_layers']} E{sh['n_experts']} k{sh['top_k']} ne{sh['non_expert_gb']} r{reserve}"
        check(f"solved == bisected planner  [{tag}]", abs(need_bisect - need_solved) < 1e-9,
              f"bisection {need_bisect} vs solved {need_solved}")
        check(f"...accepted at that budget, refused a tenth below  [{tag}]",
              accepts(sh, need_solved, reserve) and not accepts(sh, round(need_solved - 0.1, 1), reserve))
        e = refusal(sh, round(need_solved - 0.1, 1), reserve)
        check(f"...and the refusal one tenth below names exactly it  [{tag}]",
              e is not None and abs(e.needs_gb - need_solved) < 1e-9,
              f"refusal says {getattr(e, 'needs_gb', None)}")

# A reserve that moves with the budget (the prompt cache is 6% of it): the printed number must be
# accepted by the planner AT THAT NUMBER with the reserve recomputed there, not at the old budget.
sh = shape()
rf = lambda gb: round(3.3 + min(0.5, 0.06 * gb), 2)                     # noqa: E731
e = refusal(sh, 5.6, reserve=rf(5.6), reserve_fn=rf)
ok_at = A.choose_capacity(sh["manifest"], budget_gb=e.needs_gb, top_k=8, reserve_gb=rf(e.needs_gb),
                          non_expert_gb=1.1, headroom_gb=A.scaled_headroom(e.needs_gb))
below = round(e.needs_gb - 0.1, 1)
try:
    A.choose_capacity(sh["manifest"], budget_gb=below, top_k=8, reserve_gb=rf(below),
                      non_expert_gb=1.1, headroom_gb=A.scaled_headroom(below))
    refused_below = False
except MemoryError:
    refused_below = True
check("with a budget-dependent reserve the number is accepted where it is printed",
      ok_at["capacity"] >= 8)
check("...and refused one tenth below, so it is the smallest", refused_below)
fixed = refusal(sh, 5.6, reserve=rf(5.6)).needs_gb
check("...and it is not smaller than the fixed-reserve answer would wrongly be",
      e.needs_gb >= fixed, f"moving {e.needs_gb} vs fixed {fixed}")

print()
print("=" * 84)
print("2. THE WORDS: NEVER BLAME THE MACHINE FOR A NUMBER THE ENGINE PICKED")
print("=" * 84)
machine(17.2, 11.4)                                    # a 16 GB Mac, as Metal reports it
e = refusal(shape(), 5.6)
check("on a 16 GB Mac the flagship's refusal knows the Mac can", e.machine_can)
check("...and says 'a choice, not a wall'", "choice, not a wall" in str(e))
check("...and prints the flag with the number", f"BIGRIG_MAX_GB={e.needs_gb:g}" in str(e))
check("...and never the old false sentence", "cannot run" not in str(e))
e.context(model="Qwen3.6-35B-A3B-4bit", ceiling_gb=5.6, free_gb=8.2)
check("with context the command names the model",
      f"BIGRIG_MAX_GB={e.needs_gb:g} bigrig run Qwen3.6-35B-A3B-4bit" in str(e))
check("...and says the ceiling is the cause", "ceiling is 5.6 GB" in str(e))
check("...and says how much the Mac has", "17.2 GB" in str(e))
e2 = refusal(shape(), 5.6).context(model="m", ceiling_gb=5.6, free_gb=4.0)
check("when the ceiling is the cause AND memory is short, both are said",
      "BIGRIG_MAX_GB=" in str(e2) and "close something as well" in str(e2))
e3 = refusal(shape(), 5.0).context(model="m", ceiling_gb=9.0, free_gb=8.0, requested=True)
check("a --memory request under a ceiling that allows it is told to ask for more, not to raise the ceiling",
      "--memory" in str(e3) and "BIGRIG_MAX_GB" not in str(e3) and "ceiling allows it" in str(e3))
e4 = refusal(shape(), 5.0).context(model="m", ceiling_gb=9.0, free_gb=5.0)
check("a budget short because memory is busy is told to close something",
      "Close something" in str(e4) and "BIGRIG_MAX_GB" not in str(e4) and "5.0 GB is free" in str(e4))
e5 = refusal(shape(), 5.6)
check("without context the advice is still correct and names the flag",
      "choice, not a wall" in str(e5) and "BIGRIG_MAX_GB=" in str(e5) and "<model>" in str(e5))

# A model that needs more than Metal will wire: no flag helps, and the message must not offer one.
big = shape(60, 256, 8, 8_000_000, 6.0)                # top-8 alone is 3.8 GB; needs ~14 GB
eb = refusal(big, 5.6)
check("a model above the GPU's limit is told the truth: cannot run on this Mac", not eb.machine_can
      and "cannot run on this Mac" in str(eb) and "No flag" in str(eb))
check("...with the GPU limit and the installed memory both named",
      "11.4 GB" in str(eb) and "17.2 GB" in str(eb))
check("...and no flag is offered", "BIGRIG_MAX_GB" not in str(eb) and "--memory" not in str(eb))
machine(17.2, 0.0)                                     # Metal could not be asked: installed memory decides
eb2 = refusal(big, 5.6)
check("without a GPU figure, installed memory decides: 14.5 GB on a 17.2 GB Mac is a choice",
      eb2.machine_can is True and "17.2 GB" in str(eb2) and "BIGRIG_MAX_GB=" in str(eb2))
machine(12.0, 0.0)
eb3 = refusal(big, 5.6)
check("...and on a 12 GB Mac it is not, and the total is what is named",
      eb3.machine_can is False and "12.0 GB in total" in str(eb3) and "BIGRIG_MAX_GB" not in str(eb3))
machine(64.0, 48.0)
check("the same model on a 64 GB Mac is a choice again", refusal(big, 5.6).machine_can)

# The compress floor is a fact about compression only when compression was on the table.
machine(17.2, 11.4)
try:
    A.choose_strategy(big["manifest"], budget_gb=5.6, top_k=8, reserve_gb=3.7, non_expert_gb=6.0,
                      headroom_gb=A.scaled_headroom(5.6), min_bits=3)
except A.CeilingRefusal as es:
    check("choose_strategy's refusal carries the 3-bit floor when compression was allowed",
          es.floor is not None and es.floor[0] == 3 and "Compressing would not help" in str(es))
try:
    A.choose_strategy(big["manifest"], budget_gb=5.6, top_k=8, reserve_gb=3.7, non_expert_gb=6.0,
                      headroom_gb=A.scaled_headroom(5.6), min_bits=99)
except A.CeilingRefusal as es:
    check("...and none when it was declined (--exact)", es.floor is None
          and "Compressing" not in str(es))
try:
    A.choose_strategy(shape()["manifest"], budget_gb=5.6, top_k=8, reserve_gb=3.7, non_expert_gb=1.1,
                      headroom_gb=A.scaled_headroom(5.6), min_bits=3)
except A.CeilingRefusal as es:
    check("a runnable model's strategy refusal is a MemoryError still (callers catching that keep working)",
          isinstance(es, MemoryError) and es.machine_can)

src = open(os.path.join(ROOT, "bigrig_engine", "autoconfig.py")).read()
check("the false sentence is gone from the planner", "cannot run on this machine right now" not in src)
check("...and so is 'Close something, or use a smaller model' as a blanket answer",
      "Close something, or use a smaller model" not in src)

print()
print("=" * 84)
print("3. THE TWO PLANNER STEPS AGREE ON WHEN STREAMING IS POSSIBLE")
print("=" * 84)
# serve plans the pool with choose_capacity (embedding streamed) and picks the mode with
# choose_strategy. If the second charged the full embedding to the streaming room, it refused at
# budgets the first accepted -- and the flag the first printed, the second then refused.
sh = shape(ne=1.6)
for budget in (5.9, 6.0, 6.1, 6.2, 6.5):
    cap_ok = True
    try:
        A.choose_capacity(sh["manifest"], budget_gb=budget, top_k=8, reserve_gb=3.7,
                          non_expert_gb=1.1, headroom_gb=A.scaled_headroom(budget))
    except MemoryError:
        cap_ok = False
    st_ok = True
    try:
        A.choose_strategy(sh["manifest"], budget_gb=budget, top_k=8, reserve_gb=3.7,
                          non_expert_gb=1.6, stream_non_expert_gb=1.1,
                          headroom_gb=A.scaled_headroom(budget))
    except MemoryError:
        st_ok = False
    check(f"at {budget} GB capacity and strategy agree ({'run' if cap_ok else 'refuse'})", cap_ok == st_ok)
small = shape(4, 8, 2, 500_000, 0.2)                   # fits resident whole
a = A.choose_strategy(small["manifest"], budget_gb=9.0, top_k=2, reserve_gb=3.7, non_expert_gb=0.2)
b = A.choose_strategy(small["manifest"], budget_gb=9.0, top_k=2, reserve_gb=3.7, non_expert_gb=0.2,
                      stream_non_expert_gb=0.0)
check("the native decision is judged against the FULL resident figure, unchanged by the streamed one",
      a["mode"] == "native" and a == b)

print()
print("=" * 84)
print("4. THE CLAMP: ASKING WITH --memory IS NOT THE SAME AS RAISING THE CEILING, AND IT SAYS SO")
print("=" * 84)
_max = S.MAX_ALLOWED_GB
S.MAX_ALLOWED_GB = 5.6
try:
    buf = io.StringIO()
    with redirect_stdout(buf):
        got = S.resolve_budget(8.0)
    out = buf.getvalue()
    check("a --memory request above the ceiling is clamped to it", abs(got - 5.6) < 1e-9)
    check("...and the message names the knob turned and the knob needed",
          "--memory asked for 8.0 GB" in out and "BIGRIG_MAX_GB=8" in out)
    check("...and never claims 8 GB 'is free'", "is free" not in out)
    os.environ["BIGRIG_MEM_GB"] = "7.5"
    buf = io.StringIO()
    with redirect_stdout(buf):
        S.resolve_budget(None)
    check("the environment request is named as BIGRIG_MEM_GB",
          "BIGRIG_MEM_GB asked for 7.5 GB" in buf.getvalue() and "BIGRIG_MAX_GB=7.5" in buf.getvalue())
    del os.environ["BIGRIG_MEM_GB"]
    buf = io.StringIO()
    with redirect_stdout(buf):
        S.resolve_budget(None)
    check("no request: the old 'is free; using the ceiling' wording stands",
          "is free; using the 5.6 GB ceiling" in buf.getvalue())
    check("requested_budget_gb reads the flag, then the environment, else None",
          S.requested_budget_gb(4.0) == 4.0 and S.requested_budget_gb(None) is None)
    os.environ["BIGRIG_MEM_GB"] = "nonsense"
    check("...and a malformed environment value is None, not a crash", S.requested_budget_gb(None) is None)
    del os.environ["BIGRIG_MEM_GB"]
finally:
    S.MAX_ALLOWED_GB = _max

print()
print("=" * 84)
print("5. DOCTOR, SERVE AND RUN, ON THIS MACHINE, SAY ONE NUMBER -- AND THE FLAG THEY PRINT WORKS")
print("=" * 84)
A.machine_gb = _real_machine
PY = sys.executable
from bigrig_engine.cli import MODELS_DIR                                 # noqa: E402
CANDIDATES = ["Qwen3.6-35B-A3B-4bit", "NVIDIA-Nemotron-3-Nano-30B-A3B-4bit", "GLM-4.7-Flash-4bit"]
local = next((m for m in CANDIDATES if os.path.isdir(os.path.join(MODELS_DIR, m))), None)


def cli(*args, env_max="5.6", timeout=120, extra_env=None):
    env = {**os.environ, "BIGRIG_MAX_GB": env_max, **(extra_env or {})}
    env.pop("BIGRIG_MEM_GB", None)
    r = subprocess.run([PY, "-m", "bigrig_engine.cli", *args], capture_output=True, text=True,
                       env=env, timeout=timeout, stdin=subprocess.DEVNULL, cwd=ROOT)
    return r.returncode, r.stdout + r.stderr


if local is None:
    print("  SKIP  no streamed model on this machine; the CLI checks need one (not a failure)")
else:
    code, out = cli("doctor", local, "--no-recommend")
    m = re.search(r"BIGRIG_MAX_GB=([\d.]+) bigrig run " + re.escape(local), out)
    check(f"doctor at a 16 GB Mac's default ceiling refuses {local} with the flag",
          "NOT AT THIS CEILING" in out and m is not None, out[-600:])
    check("...and never says the Mac cannot", "cannot run" not in out)
    check("...and says what the flag buys, with a speed word",
          re.search(r"at [\d.]+ GB it would hold \d+ of \d+ experts", out) is not None
          and any(t in out for t in ("GOOD", "USABLE", "SLOW", "FAST")))
    if m:
        need = m.group(1)
        code2, out2 = cli("doctor", local, "--no-recommend", env_max=need)
        check(f"the printed flag works: doctor at {need} GB says it runs", "Streamed" in out2, out2[-400:])
        below = f"{float(need) - 0.1:.1f}"
        code3, out3 = cli("doctor", local, "--no-recommend", env_max=below)
        check(f"...and one tenth below ({below}) doctor still names {need}",
              f"BIGRIG_MAX_GB={need} bigrig run" in out3, out3[-400:])
        # run: the same object, the same number, before any weight is loaded (fast) and without a
        # traceback. --no-pack --no-tune so that a machine where 5.6 GB happens to suffice does not
        # start writing a blob; that case is reported as a skip, not a failure.
        code4, out4 = cli("run", local, "--no-pack", "--no-tune", timeout=90)
        if code4 != 0:
            check("`rig run` refuses with exit code 1, not a traceback",
                  code4 == 1 and "Traceback" not in out4, out4[-400:])
            check(f"...and names the SAME number doctor did ({need})",
                  f"BIGRIG_MAX_GB={need} bigrig run {local}" in out4, out4[-400:])
            check("...and says it is a choice, not a wall", "choice, not a wall" in out4)
        else:
            print("  SKIP  this machine runs the model at 5.6 GB; the run refusal cannot be exercised here")
        # prepare's closing verdict is the one serve will reach -- it used to plan against free
        # memory and say "Streamed, GOOD" right before serve refused at the ceiling.
        code7, out7 = cli("prepare", local, "--no-pack", timeout=120)
        check(f"`rig prepare` closes with the same number as doctor and run ({need})",
              f"BIGRIG_MAX_GB={need} bigrig run {local}" in out7 and "Streamed" not in out7, out7[-500:])
    code5, out5 = cli("doctor", "--memory", "8", local, "--no-recommend")
    check("doctor's MACHINE block says a clamped --memory request was clamped, and which knob raises it",
          "--memory asked for 8.0 GB" in out5 and "BIGRIG_MAX_GB=8 raises it" in out5, out5[:900])
    code6, out6 = cli("doctor", "--memory", "5", local, "--no-recommend", env_max="9")
    check("a --memory request under a ceiling that allows the model is a BUDGET verdict with --memory advice",
          "NOT AT THIS BUDGET" in out6 and "--memory" in out6.split("PREPARED MODELS")[-1]
          and "BIGRIG_MAX_GB=" not in out6.split("PREPARED MODELS")[-1], out6[-600:])

print()
print("=" * 84)
print("ALL TESTS PASSED" if not FAIL else f"{len(FAIL)} FAILURES: " + ", ".join(FAIL))
print("=" * 84)
sys.exit(1 if FAIL else 0)
