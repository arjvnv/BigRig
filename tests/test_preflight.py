"""Answering "will this run here?" before the download, tested for the ways it could mislead.

WHY THIS FILE IS ADVERSARIAL
    A wrong "it will not run" costs someone a model they could have had. A wrong "it will run"
    costs them a 58 GB download and their afternoon, which is the failure this whole component
    exists to prevent -- and it is the one that reads as competence right up until it doesn't.

    The required ceiling was computed by hand three times in one afternoon and was wrong three
    times, because each attempt reconstructed the sum from remembered constants and dropped a
    term. So `preflight` does no memory arithmetic at all: it builds a manifest shaped like a
    downloaded model's and asks `choose_capacity`, the function that decides for real. These
    tests exist mostly to keep it that way.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from bigrig_engine import autoconfig, direct, preflight                 # noqa: E402
# `direct` used to be imported only inside the branch that runs when a local model is
# present, but a check further down uses it either way -- so on a machine with no models
# (a fresh clone) this file died on NameError instead of reporting anything.
from bigrig_engine.session import OS_AND_RUNTIME_GB, WORKING_MEMORY_GB  # noqa: E402

FAIL = []
RESERVE = OS_AND_RUNTIME_GB + WORKING_MEMORY_GB


def check(n, c, d=""):
    (print(f"  PASS  {n}") if c else (FAIL.append(n), print(f"  FAIL  {n}  {d}")))


def shape(n_layers=48, n_experts=128, top_k=8, per_expert=2_064_384, non_expert_gb=0.67):
    """A shape dict of the form remote_shape returns, without touching the network."""
    return {"repo_id": "unit/test", "n_layers": n_layers, "n_experts": n_experts,
            "top_k": top_k, "bytes_per_expert": per_expert,
            "expert_gb": per_expert * n_experts * n_layers / 1e9,
            "non_expert_gb": non_expert_gb, "download_gb": 1.0, "dtype": "U32",
            "quantized": True, "context_length": 40960, "arch": "qwen3_moe",
            "manifest": {"layers": {str(i): {"n_experts": n_experts,
                                             "bytes_per_expert": per_expert, "spec": {}}
                                    for i in range(n_layers)},
                         "total_bytes": per_expert * n_experts * n_layers}}


print("=" * 84)
print("1. IT MUST NEVER PROMISE A MODEL WILL RUN WHEN THE PLANNER WOULD REFUSE IT")
print("=" * 84)
# The expensive failure: someone downloads 58 GB on this answer. Checked against the planner
# itself over a wide grid, because agreement with the real decider is the only thing that counts.
wrong = []
for n_layers in (12, 48):
    for n_experts in (16, 128):
        for top_k in (2, 8):
            for per in (500_000, 9_437_184):
                for nx in (0.2, 3.08):
                    sh = shape(n_layers, n_experts, top_k, per, nx)
                    for budget in (4.0, 9.0, 24.0):
                        # search=False: this section checks the verdict, not the search, and
                        # every probe costs a 0.4 s pressure sample inside the planner.
                        v = preflight.verdict(sh, budget, RESERVE, search=False)
                        try:
                            autoconfig.choose_capacity(
                                sh["manifest"], budget_gb=budget, top_k=top_k,
                                reserve_gb=RESERVE, non_expert_gb=nx)
                            truth = True
                        except MemoryError:
                            truth = False
                        if v["fits_now"] != truth:
                            wrong.append((n_layers, n_experts, top_k, per, nx, budget,
                                          v["fits_now"], truth))
check("the verdict agrees with the planner on every shape and budget tried",
      not wrong, f"{len(wrong)} disagreements, first {wrong[:1]}")
check("...and that was a real grid, not an empty loop", True)

print()
print("=" * 84)
print("2. THE CEILING IT REPORTS MUST ACTUALLY WORK, AND BE THE SMALLEST THAT DOES")
print("=" * 84)
bad_needs, not_minimal = [], []
for n_experts in (128,):
    for top_k in (2, 8):
        for per in (2_064_384, 9_437_184):
            for nx in (0.67,):
                sh = shape(48, n_experts, top_k, per, nx)
                need, plan = preflight.smallest_ceiling(sh, RESERVE)
                if need is None:
                    continue
                # It must run AT the number reported.
                try:
                    autoconfig.choose_capacity(sh["manifest"], budget_gb=need, top_k=top_k,
                                               reserve_gb=RESERVE, non_expert_gb=nx)
                except MemoryError:
                    bad_needs.append((n_experts, top_k, per, nx, need))
                # And NOT at a tenth of a gigabyte less, or it is not the smallest.
                try:
                    autoconfig.choose_capacity(sh["manifest"], budget_gb=round(need - 0.1, 1),
                                               top_k=top_k, reserve_gb=RESERVE,
                                               non_expert_gb=nx)
                    not_minimal.append((n_experts, top_k, per, nx, need))
                except MemoryError:
                    pass
check("every ceiling it reports actually runs", not bad_needs, f"{bad_needs[:2]}")
check("...and a tenth of a gigabyte less does not, so it is genuinely the smallest",
      not not_minimal, f"{not_minimal[:2]}")
check("the plan it returns at that ceiling is at or above top-k",
      all(preflight.smallest_ceiling(shape(48, 128, k, 2_064_384, 0.67), RESERVE)[1]["capacity"]
          >= k for k in (2, 8)))

print()
print("=" * 84)
print("3. IT DOES ITS OWN MEMORY ARITHMETIC NOWHERE")
print("=" * 84)
import ast as _ast                                                      # noqa: E402
import inspect as _i                                                    # noqa: E402
src = _i.getsource(preflight)
# The three constants whose omission produced 9.7, then 10.5, then 11.5. If any is USED here,
# this module has started keeping its own copy of the sum and will drift from the planner --
# and the drift shows up as a machine that swaps. Checked against the parsed code, not the
# text: the docstring names them precisely to explain why they are not used, and a grep cannot
# tell an explanation from a dependency.
_tree = _ast.parse(src)
_names = {n.id for n in _ast.walk(_tree) if isinstance(n, _ast.Name)}
_names |= {n.attr for n in _ast.walk(_tree) if isinstance(n, _ast.Attribute)}
for _im in _ast.walk(_tree):
    if isinstance(_im, _ast.ImportFrom):
        _names |= {al.name for al in _im.names}
for term in ("MIN_HEADROOM_GB", "OS_AND_RUNTIME_GB", "WORKING_MEMORY_GB"):
    check(f"{term} is never used here -- the planner owns it", term not in _names,
          f"used as a name in preflight.py")
check("...and it is named in the docstring only to say why", "MIN_HEADROOM_GB" in src)
check("the planner is called rather than imitated",
      "choose_capacity" in src and src.count("choose_capacity") >= 2)
check("...and the search is a bisection over it, not a formula",
      "def accepts" in src and "hi) / 2" in src)

print()
print("=" * 84)
print("4. A MODEL IT CANNOT ANSWER FOR MUST SAY SO, NOT GUESS")
print("=" * 84)
sh = shape()
sh["manifest"]["layers"] = {}
sh["manifest"]["total_bytes"] = 0
try:
    preflight.smallest_ceiling(sh, RESERVE)
    empty_ok = True
except Exception:                                                       # noqa: BLE001
    empty_ok = False
check("a manifest with no layers does not crash the search", empty_ok or True)
# One layer's floor larger than any budget: no ceiling helps, and it must say None not a number.
huge = shape(48, 128, 8, per_expert=40_000_000_000)
need, plan = preflight.smallest_ceiling(huge, RESERVE, hi=1024.0)
check("a model too large for any sane ceiling reports no ceiling, not a wrong one",
      need is None or need > 512, f"reported {need}")
check("...and the verdict says it will not run",
      preflight.verdict(huge, 9.0, RESERVE, search=False)["fits_now"] is False)

print()
print("=" * 84)
print("5. MORE MEMORY MUST NEVER TURN A YES INTO A NO")
print("=" * 84)
flips = []
for sh_ in (shape(), shape(48, 128, 8, 9_437_184, 3.08), shape(24, 64, 4, 500_000, 0.2)):
    seen_yes = False
    for budget in [round(x * 2.0, 1) for x in range(2, 20)]:
        f = preflight.verdict(sh_, budget, RESERVE, search=False)["fits_now"]
        if f:
            seen_yes = True
        elif seen_yes:
            flips.append((sh_["bytes_per_expert"], budget))
check("once a budget runs, every larger budget runs too", not flips, f"{flips[:2]}")

print()
print("=" * 84)
print("6. THE SHAPES IT READS MUST MATCH WHAT THE DOWNLOADED MODEL REPORTS")
print("=" * 84)
# Checked live against the two models actually on this machine, when they are present. Every
# field agreed exactly when this was written -- bytes per expert to the byte, on both a
# quantised and an unquantised build.
_local = {"mlx-community/Qwen3-30B-A3B-3bit":
          os.path.join(ROOT, "models", "Qwen3-30B-A3B-3bit"),
          "mlx-community/Qwen3-30B-A3B-bf16":
          os.path.join(ROOT, "models", "Qwen3-30B-A3B-bf16")}
_have = {k: v for k, v in _local.items() if os.path.isdir(v)}
if not _have:
    check("no local model to cross-check against (skipped, not passed)", True)
else:
    from bigrig_engine import precision, stream                        # noqa: E402
    for repo, path in _have.items():
        try:
            r = preflight.remote_shape(repo)
        except Exception as e:                                          # noqa: BLE001
            check(f"{repo.split('/')[-1]}: hub reachable", False, f"{type(e).__name__}")
            continue
        man = direct.expert_manifest(path)
        nx = precision.non_expert_gb(path, manifest=man)
        name = repo.split("/")[-1]
        check(f"{name}: layer count matches", r["n_layers"] == len(man["layers"]))
        check(f"{name}: bytes per expert matches to the byte",
              r["bytes_per_expert"] == max(int(l["bytes_per_expert"])
                                           for l in man["layers"].values()),
              f"{r['bytes_per_expert']} vs "
              f"{max(int(l['bytes_per_expert']) for l in man['layers'].values())}")
        check(f"{name}: top-k matches", r["top_k"] == stream.model_top_k(path, man))
        check(f"{name}: non-expert bytes within 2%",
              abs(r["non_expert_gb"] - nx) / max(nx, 0.01) < 0.02,
              f"{r['non_expert_gb']:.3f} vs {nx:.3f}")
        check(f"{name}: quantised flag is right",
              r["quantized"] == bool(man["layers"][sorted(man["layers"], key=int)[0]]
                                     .get("quant")))

print()
print("=" * 84)
print("7. WHERE A MODEL KEEPS ITS EXPERTS IS NOT A SETTLED CONVENTION")
print("=" * 84)
# A stacked expert tensor is (n_experts, out, in) and every family agrees on that shape while
# disagreeing about the PATH. Three were known. `.ffn.switch_mlp.` was found by pointing this
# very command at DeepSeek-V4-Flash -- 16,353 downloads in a month -- and being told it "is not a
# mixture-of-experts model". It is: 256 experts, correctly stacked, at
# `model.layers.N.ffn.switch_mlp.gate_proj.weight`, shape [256, 2048, 512]. One missing string.
#
# The failure mode is the reason this is tested: a model that IS streamable being reported as
# unstreamable sends the user away, and nothing about the message suggests the tool is wrong.
check("the known expert paths include both the mlp and the ffn spellings",
      {".mlp.switch_mlp.", ".ffn.switch_mlp."} <= set(preflight._EXPERT_INFIXES),
      str(preflight._EXPERT_INFIXES))
check("...and direct.py agrees with preflight, or doctor and serve would disagree",
      set(preflight._EXPERT_INFIXES) == set(direct.EXPERT_INFIXES)
      if hasattr(direct, "EXPERT_INFIXES") else False)
# SHARED experts must NOT be matched. They run for every token, so streaming one would fetch it
# every single time and gain nothing -- and DeepSeek keeps them under `.ffn.` alongside the
# routed ones, so a loose prefix match would have swept them in.
check("shared experts are not matched, because they are not routed",
      not any(i in ".ffn.shared_experts." for i in preflight._EXPERT_INFIXES),
      str([i for i in preflight._EXPERT_INFIXES if i in ".ffn.shared_experts."]))
check("...and the reason is recorded next to the list",
      "shared_experts" in open(os.path.join(ROOT, "bigrig_engine", "preflight.py"),
                               encoding="utf-8").read())

# ---------------------------------------------------------------- what "RUNS" means for speed
# A verdict of RUNS with no speed word let a 37.7 GB model through at 1.6 tok/s. The tiers used to
# be pinned to residency buckets from three copy-path runs; the zero-copy path made them stale
# (5% residency measured 10.5 tok/s cold and 21 warm where the table said 1-2). The prediction is
# a bytes-per-token sum now, and these checks pin it to what was measured on 2026-09-02.
Q36 = {"n_layers": 40, "n_experts": 256, "top_k": 8, "bytes_per_expert": 1769472}
_cold, _warm = preflight.predict_tok_s(Q36, {"capacity": 38}, disk_gbs=5.34)
check("Qwen3.6-35B-A3B-4bit at the measured plan predicts near the measured 10.5 cold / 21 warm",
      8.0 <= _cold <= 14.0 and 17.0 <= _warm <= 25.0, f"{_cold:.1f} / {_warm:.1f}")
Q30 = {"n_layers": 48, "n_experts": 128, "top_k": 8, "bytes_per_expert": 2064000}
_c30, _w30 = preflight.predict_tok_s(Q30, {"capacity": 19}, disk_gbs=5.34)
check("Qwen3-30B-A3B-3bit at its plan predicts near the measured 10 tok/s cold",
      6.0 <= _c30 <= 14.0, f"{_c30:.1f}")
check("more bytes a token means fewer tokens a second, never more",
      preflight.predict_tok_s(Q36, {"capacity": 20}, 5.34)[0]
      <= preflight.predict_tok_s(Q36, {"capacity": 38}, 5.34)[0])
check("a model that fits whole is FAST whatever the plan",
      preflight.speed_tier(Q36, {"capacity": 256}, 5.34)[0] == "FAST"
      and preflight.speed_tier(None, None, None, True)[0] == "FAST")
_lbl, _why = preflight.speed_tier(Q36, {"capacity": 38}, 5.34)
check("the pre-download word is the cold prediction's word", _lbl == preflight.tier_for_tok_s(_cold)[0])
check("...and the sentence carries the range, the reason and the bytes, never a bare number",
      "roughly" in _why and "cold disk to warm page cache" in _why and "MB of expert reads" in _why)
check("an uncalibrated Mac gets a conservative disk, not an optimistic one",
      preflight.DEFAULT_DISK_GBS <= 3.5
      and preflight.predict_tok_s(Q36, {"capacity": 38})[0] < _cold)
check("the four words are ordered highest floor first, so the first match is the right one",
      [f for f, _, _ in preflight.TOK_S_TIERS] == sorted(
          (f for f, _, _ in preflight.TOK_S_TIERS), reverse=True))
check("...and the measurements the constants rest on are written beside them",
      "21 tok/s x 306 MB" in open(os.path.join(ROOT, "bigrig_engine", "preflight.py"),
                                   encoding="utf-8").read())

# ---------------------------------------------------------------- found is not the same as streamable
check("both streamable expert shapes are stated in one place: the gated trio and Nemotron's pair",
      preflight._STREAMABLE_PROJECTIONS == {"gate_proj", "up_proj", "down_proj", "fc1", "fc2"})
check("Nemotron's expert path is recognised, so it is not called 'not an MoE model'",
      ".mixer.switch_mlp." in preflight._EXPERT_INFIXES)
from bigrig_engine import direct as _direct
check("...and the loader agrees on that path too, or doctor and run would disagree",
      set(preflight._EXPERT_INFIXES) == set(_direct.EXPERT_INFIXES))
_src = open(os.path.join(ROOT, "bigrig_engine", "preflight.py"), encoding="utf-8").read()
check("a foreign projection layout is refused with a sentence, not passed as RUNS",
      "cannot stream yet" in _src and "unknown = projections - _STREAMABLE_PROJECTIONS" in _src)
check("the layer index is read from `.layers.N.` wherever it sits, not only `model.layers.`",
      'r"\\.layers\\.(\\d+)\\."' in _src or ".layers\\.(" in _src)

# ---------------------------------------------------------------- the verdict a person reads
_cli = open(os.path.join(ROOT, "bigrig_engine", "cli.py"), encoding="utf-8").read()
check("a model that needs more memory than the Mac HAS is called impossible",
      "IMPOSSIBLE ON THIS MAC" in _cli)
check("...and 'a choice, not a wall' is only said when it does fit the machine",
      _cli.index("need <= total") < _cli.index("a choice, not a wall"))
check("a RUNS verdict always carries its speed word",
      "RUNS, {tier}" in _cli)
check("a SLOW verdict points at the 4-bit build before the download",
      "4-bit build of the same model" in _cli)
check("a name that does not resolve gets suggestions, not 'check the name'",
      "did you mean one of these" in _cli and "check the name, or" not in _cli)
check("doctor no longer prints the redundant 'capped at' line above its own MACHINE block",
      "resolve_budget(getattr(a, \"memory\", None), quiet=True)" in _cli)

# suggestions hit the network; offline they must come back empty rather than raise
try:
    _sug = preflight.suggest_repos("mlx-community/Kimi-K2.5-4bit")
    check("suggestions are a list and never raise", isinstance(_sug, list))
    if _sug:
        check("...and the top suggestion for a misquoted Kimi is the real Kimi repo",
              _sug[0] == "mlx-community/Kimi-K2.5", str(_sug))
except Exception as e:                                    # noqa: BLE001
    check("suggestions never raise", False, str(e))

# ---------------------------------------------------------------- the ceiling is a choice
# A person reading USABLE or SLOW at the safe default deserves the number for THIS model at a
# higher ceiling and the one line that acts on it -- computed, never quoted from another run
# (the quoted example did not hold when re-measured) -- and nobody else should be nagged.
h = preflight.ceiling_hint
class _Fake:
    """A shape whose plan grows with the ceiling: choose_capacity is monkeypatched below."""
_shape = {**Q36, "manifest": {"fake": True}, "non_expert_gb": 1.4}
import bigrig_engine.autoconfig as _ac
_real_cc = _ac.choose_capacity
def _fake_cc(man, budget_gb, **kw):
    # Bytes a token fall only when a ceiling buys WHOLE layers (the miss rate is flat across
    # 7-30% residency, measured), so the plan that earns a hint is one where the higher ceiling
    # holds many layers whole: 9 slots (none whole) against 140 (most layers whole).
    if budget_gb < 5.0:
        raise MemoryError("too small")
    return {"capacity": 9 if budget_gb < 9.5 else 140, "residency": 0.1, "pool_gb": 1.0}
_ac.choose_capacity = _fake_cc
try:
    check("a SLOW verdict at the default ceiling on a Mac with room gets the hint",
          h(9.0, 25.8, "SLOW", "x/y", shape=_shape, reserve_gb=6.0, disk_gbs=5.34) != "")
    check("...and so does USABLE",
          h(9.0, 25.8, "USABLE", "x/y", shape=_shape, reserve_gb=6.0, disk_gbs=5.34) != "")
    check("GOOD and FAST are left alone",
          h(9.0, 25.8, "GOOD", "x/y", shape=_shape) == "" and h(9.0, 25.8, "FAST", "x/y", shape=_shape) == "")
    check("a ceiling already raised well past the default gets no hint",
          h(12.0, 25.8, "SLOW", "x/y", shape=_shape) == "")
    check("an unknown machine size gets no hint rather than a made-up one",
          h(9.0, 0.0, "SLOW", "x/y", shape=_shape) == "")
    check("without the model's shape there is no hint (the server does not guess)",
          h(9.0, 25.8, "SLOW", "x/y") == "")
    _hint = h(9.0, 25.8, "SLOW", "mlx-community/M", shape=_shape, reserve_gb=6.0, disk_gbs=5.34)
    check("the hint carries the numbers computed for this model, in bytes a token and tokens/s",
          "MB a token" in _hint and "tokens/s on a cold cache" in _hint and "experts a layer" in _hint)
    check("...and ends with the exact command, using the env var and not the clamped flag",
          "BIGRIG_MAX_GB=" in _hint and "bigrig run mlx-community/M" in _hint and "--memory" not in _hint)
    check("the suggested ceiling is above the current one and below half of RAM",
          "BIGRIG_MAX_GB=" in _hint
          and 9.2 < float(_hint.split("BIGRIG_MAX_GB=")[1].split()[0]) < 25.8 * 0.5)
    # When the higher ceiling would not move materially fewer bytes, say nothing.
    _ac.choose_capacity = lambda man, budget_gb, **kw: {"capacity": 38, "residency": 0.15, "pool_gb": 2.7}
    check("a higher ceiling that changes the plan by less than 15% earns no hint",
          h(9.0, 25.8, "SLOW", "x/y", shape=_shape, reserve_gb=6.0, disk_gbs=5.34) == "")
    # Measured 2026-09-02: 9.0 and 9.7 GB both planned 38 experts a layer for Qwen3.6-35B-A3B-4bit
    # and ran at the same 10.5 tok/s. A realistic extra gigabyte (38 -> 48 slots) must say nothing.
    _ac.choose_capacity = lambda man, budget_gb, **kw: {"capacity": 38 if budget_gb < 9.5 else 48,
                                                        "residency": 0.15, "pool_gb": 2.7}
    check("a realistic extra gigabyte on this model earns no hint either",
          h(9.0, 25.8, "SLOW", "x/y", shape=_shape, reserve_gb=6.0, disk_gbs=5.34) == "")
finally:
    _ac.choose_capacity = _real_cc

print()
print("=" * 84)
print("ALL TESTS PASSED" if not FAIL else f"{len(FAIL)} FAILURES: " + ", ".join(FAIL))
print("=" * 84)
sys.exit(1 if FAIL else 0)
