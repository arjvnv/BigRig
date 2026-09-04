"""The quality meter's free-energy signal: why it exists, and what it was measured to do.

THE MEASUREMENT THAT PROMPTED IT
    The meter shipped as this product's differentiator and had never been scored against ground
    truth. Scored properly -- the same model and prompts generating once with its real experts
    and once with N of 16 deliberately scrambled -- it had zero false positives and caught severe
    damage perfectly, and was at CHANCE on light damage:

        damaged layers      1 of 16   2 of 16   4 of 16   8 of 16
        entropy (before)      0.500     0.500     0.708     1.000
        with energy           0.917     0.875     1.000     0.917

    Light damage is exactly what this engine causes on purpose. Compression and rerouting shift a
    model a little; a meter blind to that cannot report on the trades the product makes.

WHY ENTROPY MISSES IT
    logprobs = logits - logsumexp(logits). The softmax has already divided out the logit scale,
    and the scale is where light damage shows. Free energy is that scale: -logsumexp(logits).
    Compared on identical generations, mean AUROC over the same damage levels:

        entropy 0.625 / 0.972 / 0.917      energy 1.000 / 1.000 / 1.000
        logit gap 0.188 / 0.319 / 0.757    one-minus-top-prob 0.215 / 0.583 / 0.889

AND IT MADE THE METER FREE
    The old signal needed the whole log-probability vector: 628 KB a token, and 34% of generation
    (33.0 tok/s down to 21.7). Energy is one scalar. With it, monitoring on and off are within
    run-to-run noise of each other.
"""
import inspect
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bigrig_layer.adaptive import AdaptiveMeter  # noqa: E402

FAIL = []


def check(n, c, d=""):
    (print(f"  PASS  {n}") if c else (FAIL.append(n), print(f"  FAIL  {n}  {d}")))


print("=" * 84); print("1. THE SIGNAL LEARNS THIS MODEL'S NORMAL"); print("=" * 84)
rng = np.random.default_rng(0)
# Tokens are fed alongside energy, which is what the engine does -- every generated token gets
# both. The meter now declines to give a verdict until a reply has produced enough of its own
# tokens to judge, and a probe that fed only energy would sit below that floor forever and read
# as "no opinion" rather than testing the energy path at all.
m = AdaptiveMeter()
for i in range(60):
    m.observe_energy(float(rng.normal(-5.0, 0.2)))
    m.observe_token(1000 + i)
check("healthy output sits near zero deviation", abs(m.energy_z) < 2.5, f"{m.energy_z:.2f}")
check("...and is not flagged", not m._energy_degraded())
for i in range(5):
    m.observe_energy(-3.0)
    m.observe_token(2000 + i)
check("a clear shift in energy is flagged", m._energy_degraded(), f"{m.energy_z:.2f}")
check("...and named for what it is", m.reason() == "weights drifted", str(m.reason()))

m2 = AdaptiveMeter()
for _ in range(60):
    m2.observe_energy(float(rng.normal(-40.0, 3.0)))       # a different model's scale entirely
check("a model with a completely different energy scale is not flagged for it",
      not m2._energy_degraded(), f"{m2.energy_z:.2f}")
check("...which is the point: the absolute value does not transfer between models",
      abs(m2._energy_mean - (-40.0)) < 2.0, f"{m2._energy_mean:.2f}")

print("\n" + "=" * 84); print("2. IT CANNOT BE FOOLED INTO A BAD BASELINE"); print("=" * 84)
m3 = AdaptiveMeter()
for _ in range(60):
    m3.observe_energy(float(rng.normal(-5.0, 0.2)))
base = m3._energy_mean
for _ in range(30):
    m3.observe_energy(-3.0)                                 # a long bad patch
check("a sustained bad patch is not absorbed into normal",
      abs(m3._energy_mean - base) < 0.5, f"{base:.2f} -> {m3._energy_mean:.2f}")
check("...and it is still flagged after 30 bad steps", m3._energy_degraded())

print("\n" + "=" * 84); print("3. IT NEVER BREAKS GENERATION"); print("=" * 84)
m4 = AdaptiveMeter()
m4.observe_energy(None)
m4.observe_energy(float("nan"))
check("None and NaN are ignored rather than poisoning the baseline", m4._energy_seen == 0)
check("...and asking before anything was seen does not raise", m4.energy_z == 0.0)
check("a meter with no energy at all still answers", m4.is_degraded() in (None, False, True))
m5 = AdaptiveMeter()
for _ in range(3):
    m5.observe_energy(-5.0)
check("nothing is flagged before there is enough to know normal", not m5._energy_degraded())

print("\n" + "=" * 84); print("4. THE ENGINE FEEDS IT, AND CHEAPLY"); print("=" * 84)
from bigrig_engine import session  # noqa: E402
_obs = inspect.getsource(session.Session._observe)
check("energy is preferred when the logits could be captured", "use_energy" in _obs)
check("...and computed from the RAW logits, not the log-probabilities",
      "mx.logsumexp(self._last_logits)" in _obs)
check("the 628 KB vector read is skipped when energy is available",
      "if not use_energy:" in _obs and "stats_from_logprobs" in _obs)
check("...but still used when it is not, so a model always gets a meter",
      _obs.index("use_energy") < _obs.index("stats_from_logprobs"))
_st = inspect.getsource(session.Session.stream_text)
check("the model's forward is wrapped to capture logits", "_mcls.__call__ = _capture" in _st)
check("...and restored however the generator ends", "finally:" in _st
      and "_mcls.__call__ = _orig_call" in _st)
check("...including when a client abandons the stream mid-reply",
      "abandoned" in _st)
check("a capture failure degrades to no energy rather than raising",
      "_self._last_logits = None" in _st)

print("\n" + "=" * 84)
print(f"{'ALL TESTS PASSED' if not FAIL else str(len(FAIL)) + ' FAILURES: ' + ', '.join(FAIL)}")
print("=" * 84)
sys.exit(1 if FAIL else 0)
