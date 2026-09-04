"""Per-model calibration: what it measures, and what it refuses to conclude.

THE HISTORY THIS FILE EXISTS TO PIN
    `plan_capacity` has modelled the sync-versus-miss trade since the beginning and has shipped
    OFF the whole time, because it was fed Qwen3's constants for every model: OLMoE at a 90%
    budget was predicted 4.87x and measured 1.19x.

    Measuring the round-trip per model did not fix it. With a per-model sync curve the planner
    predicted 1.15x on OLMoE and delivered 0.88x -- SLOWER. The half still being guessed was how
    fast the miss rate rises as residency falls. Measuring that too still predicted 1.63x and
    delivered 0.81x, because a decode step on a small model is noisy enough that the curve comes
    out non-monotonic between runs, and once even 3.7 ms/token for all 16 layers syncing against
    7.6 for none -- a negative cost for a thing that cannot be free.

    So nothing here is trusted on prediction. The proposed split is RUN against uniform and only
    the measured ratio decides. Verified live on OLMoE: proposed 6 layers whole, measured 0.76x,
    declined.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bigrig_engine import stream, synccal          # noqa: E402

FAIL = []


def check(n, c, d=""):
    (print(f"  PASS  {n}") if c else (FAIL.append(n), print(f"  FAIL  {n}  {d}")))


print("=" * 84); print("1. A CURVE THAT DISAGREES WITH ITSELF IS NOT A CURVE"); print("=" * 84)
GOOD = {"ms_by_syncing_layers": {"0": 10.0, "8": 30.0, "16": 60.0}, "ms_per_sync_layer": 3.1,
        "miss_by_residency": {"0.25": 0.48, "0.5": 0.28, "0.75": 0.03}, "ms_per_miss": 1.0}
check("a monotonic curve with a positive cost is usable", synccal.usable(GOOD)[0])
# Seen live: all 16 layers syncing timed at 3.67 ms/token against 7.60 for none.
NEG = {"ms_by_syncing_layers": {"0": 7.6, "8": 51.7, "16": 3.7}, "ms_per_sync_layer": -0.25}
ok, why = synccal.usable(NEG)
check("a round-trip that measured free or negative is refused", not ok)
check("...and says it is noise, not a discovery", "noise" in why, why)
DIP = {"ms_by_syncing_layers": {"0": 10.0, "8": 30.0, "16": 20.0}, "ms_per_sync_layer": 0.6}
ok2, why2 = synccal.usable(DIP)
check("more syncing layers timing faster than fewer is refused", not ok2)
check("...naming the two points that contradict", "16" in why2 and "8" in why2, why2)
check("too few points is refused",
      not synccal.usable({"ms_by_syncing_layers": {"0": 1.0}, "ms_per_sync_layer": 1.0})[0])
check("a curve with no measurements at all is refused", not synccal.usable({})[0])

print("\n" + "=" * 84); print("2. BOTH HALVES, OR NEITHER"); print("=" * 84)
check("a curve with no miss half cannot drive the planner",
      synccal.as_cost_fn({k: v for k, v in GOOD.items() if k != "miss_by_residency"}) is None)
check("...and with both halves it can", synccal.as_cost_fn(GOOD) is not None)
_f, _mpm, _miss = synccal.as_cost_fn(GOOD)
check("the round-trip cost is measured from the floor up", _f(0) == 0.0)
check("...rises with the number of syncing layers", _f(16) > _f(8) > _f(0))
# 0 and 8 syncing layers measured 10.0 and 30.0 ms, so 4 sits halfway: 20.0 - 10.0 = 10.0.
check("...and interpolates between measured points", abs(_f(4) - 10.0) < 1e-9, str(_f(4)))
check("...held flat past the last real point rather than extrapolated",
      _f(999) == _f(16), f"{_f(999)} vs {_f(16)}")
check("the miss curve is this model's own", abs(_miss(0.5) - 0.28) < 1e-9)
check("...interpolates", 0.03 < _miss(0.6) < 0.28)
check("...is zero when everything is resident", _miss(1.0) == 0.0)
check("...and is held flat below the lowest residency measured", _miss(0.01) == _miss(0.25))
check("a miss curve with one point is not a curve",
      synccal.as_miss_fn({"miss_by_residency": {"0.5": 0.2}}) is None)

print("\n" + "=" * 84); print("3. THE PLANNER USES THIS MODEL'S NUMBERS"); print("=" * 84)
_p = stream.plan_capacity(16, 64, int(0.8 * 16 * 64), top_k=8, measured=synccal.as_cost_fn(GOOD))
check("a plan built from a measurement says so", _p.get("source") == "measured")
check("...and reports what it compared against", "uniform_ms_per_token" in _p)
check("...and never proposes a layer below top-k", _p["capacity"] >= 8)
_cheap = dict(GOOD, ms_by_syncing_layers={"0": 10.0, "8": 10.5, "16": 11.0},
              ms_per_sync_layer=0.06)
_pc = stream.plan_capacity(16, 64, int(0.8 * 16 * 64), top_k=8,
                           measured=synccal.as_cost_fn(_cheap))
check("when the round-trip is cheap, holding layers whole is not worth the experts",
      _pc["full_layers"] <= _p["full_layers"], f'{_pc["full_layers"]} vs {_p["full_layers"]}')

print("\n" + "=" * 84); print("4. NOTHING IS BELIEVED ON PREDICTION ALONE"); print("=" * 84)
import inspect  # noqa: E402
from bigrig_engine import cli, session  # noqa: E402
_init = inspect.getsource(session.Session.__init__)
check("a split is used only when a measured speed-up was recorded",
      'curve.get("verified_speedup")' in _init)
check("...and only when it was actually faster", "got > 1.02" in _init)
check("...never from the planner's prediction", "ms_per_token\"] < " not in _init)
check("calibration runs the split it proposes rather than trusting it",
      "_verify_split" in inspect.getsource(cli))
check("...timing both configurations", inspect.getsource(cli._verify_split).count("timed(") == 3)
check("an unusable measurement is not allowed to drive one",
      "not good enough to plan with" in inspect.getsource(cli.cmd_calibrate))

print("\n" + "=" * 84); print("5. A BAD FILE MUST NOT STOP A MODEL LOADING"); print("=" * 84)
import json  # noqa: E402
import tempfile  # noqa: E402
check("a model with no curve gets None", synccal.load_curve("no-such-model-xyz") is None)
_d = tempfile.mkdtemp()
_old = synccal.CURVE_DIR
try:
    synccal.CURVE_DIR = _d
    open(synccal.curve_path("broken"), "w").write("{not json")
    check("a corrupt curve reads as absent rather than raising",
          synccal.load_curve("broken") is None)
    json.dump({"model": "empty"}, open(synccal.curve_path("empty"), "w"))
    check("a curve with no measurements reads as absent", synccal.load_curve("empty") is None)
    check("a model name with a slash cannot escape the results directory",
          "/" not in os.path.basename(synccal.curve_path("a/../../b")))
    _saved = synccal.save(dict(GOOD, model="round-trip"))
    check("what is saved is what is loaded",
          synccal.load_curve("round-trip")["ms_per_sync_layer"] == GOOD["ms_per_sync_layer"])
finally:
    synccal.CURVE_DIR = _old

print("\n" + "=" * 84)
print(f"{'ALL TESTS PASSED' if not FAIL else str(len(FAIL)) + ' FAILURES: ' + ', '.join(FAIL)}")
print("=" * 84)
sys.exit(1 if FAIL else 0)
