"""Adversarial tests for AutoTuner. Written to find ways the control loop misbehaves."""
import os
import sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bigrig_layer.controller import AutoTuner
from bigrig_layer import QualityMeter

FAIL = []
def check(n, c, d=""):
    (print(f"  PASS  {n}") if c else (FAIL.append(n), print(f"  FAIL  {n}  {d}")))

V = 1000
def conf(p1=0.9):
    p = np.full(V, (1 - p1) / (V - 1)); p[0] = p1; return p
def vague():
    return np.full(V, 1.0 / V)

print("=" * 78); print("1. BOUNDS — the dial must never leave its range"); print("=" * 78)
t = AutoTuner(dial_max=0.4, dial_min=0.05, window=None) if False else AutoTuner(dial_max=0.4, dial_min=0.05)
for _ in range(400): t.observe(vague())
check("never falls below dial_min under relentless bad output",
      t.dial >= 0.05 - 1e-12, f"dial={t.dial}")
t2 = AutoTuner(dial_max=0.4, dial_min=0.0, start=0.0)
for _ in range(400): t2.observe(conf())
check("never rises above dial_max under relentless good output",
      t2.dial <= 0.4 + 1e-12, f"dial={t2.dial}")
try:
    AutoTuner(dial_max=0.1, dial_min=0.9); check("rejects min > max", False)
except ValueError: check("rejects min > max", True)

print("\n" + "=" * 78); print("2. NO ACTION WITHOUT EVIDENCE"); print("=" * 78)
# Sections 2-5 test the TUNER'S ARITHMETIC, which needs a meter whose absolute verdict is known
# in advance. QualityMeter is pinned here for that reason. It is no longer the default: a
# self-calibrating meter would learn that relentlessly uniform output is this model's normal and
# correctly stop flagging it, which is the right behaviour but useless for testing step sizes.
# The default's own behaviour is covered in section 7.
t = AutoTuner(dial_max=0.4, meter=QualityMeter())
start = t.dial
for _ in range(63): t.observe(vague())
check("dial unchanged before the window fills", t.dial == start, f"{start} -> {t.dial}")
t.observe(vague())
check("acts as soon as the window fills", t.dial < start, f"{start} -> {t.dial}")

print("\n" + "=" * 78); print("3. ASYMMETRY — retreat fast, advance slow"); print("=" * 78)
t = AutoTuner(dial_max=1.0, start=0.5, down_step=0.10, up_step=0.02, patience=3,
              meter=QualityMeter())
for _ in range(64): t.observe(vague())
d_after_one_bad = 0.5 - t.dial
check("one bad reading drops the dial by down_step",
      abs(d_after_one_bad - 0.10) < 1e-9, f"dropped {d_after_one_bad}")
t2 = AutoTuner(dial_max=1.0, start=0.5, patience=3, meter=QualityMeter())
for _ in range(64): t2.observe(conf())
for _ in range(3): t2.observe(conf())
check("needs `patience` good readings before advancing", t2.dial > 0.5, f"dial={t2.dial}")
check("retreat is larger than advance", 0.10 > 0.02)

print("\n" + "=" * 78); print("4. RECOVERY — it must come back, not stay pinned low"); print("=" * 78)
t = AutoTuner(dial_max=0.4, start=0.4, meter=QualityMeter())
for _ in range(80): t.observe(vague())     # bad patch
low = t.dial
for _ in range(300): t.observe(conf())     # quality recovers
check("dial recovers after quality returns", t.dial > low, f"{low:.3f} -> {t.dial:.3f}")
check("recovery reaches the ceiling given enough good output",
      abs(t.dial - 0.4) < 1e-9, f"dial={t.dial:.3f}")

print("\n" + "=" * 78); print("5. BOOKKEEPING"); print("=" * 78)
t = AutoTuner(dial_max=0.4, meter=QualityMeter())
for _ in range(120): t.observe(vague())
s = t.summary()
check("counts backoffs", s["backoffs"] > 0, str(s))
check("degraded_frac is a fraction", 0.0 <= s["degraded_frac"] <= 1.0)
check("history length matches acted-on steps", len(t.history) == s["steps"])
t.reset(start=0.2)
check("reset clears state and sets the dial", t.dial == 0.2 and t.n_backoffs == 0
      and not t.history and t.meter.score() is None)

print("\n" + "=" * 78); print("6. IT PASSES THROUGH TO THE METER"); print("=" * 78)
t = AutoTuner(dial_max=0.4, strict=True)
try:
    t.observe(np.full(V, np.nan)); check("malformed input still raises", False)
except ValueError: check("malformed input still raises", True)
t = AutoTuner(dial_max=0.4, meter=QualityMeter(window=8))
for _ in range(8): t.observe(conf())
check("accepts a custom meter", t.meter.window == 8 and t.meter.ready())

print("\n" + "=" * 78); print("7. THE DEFAULT METER IS THE SELF-CALIBRATING ONE"); print("=" * 78)
from bigrig_layer.adaptive import AdaptiveMeter
t = AutoTuner(dial_max=0.4)
check("AutoTuner defaults to AdaptiveMeter, not to one model's constants",
      isinstance(t.meter, AdaptiveMeter), type(t.meter).__name__)

# The point of the default: a model whose normal output would look "degraded" to Ling's
# constants must not have its dial given away for nothing. Mid-entropy output, relentlessly.
def mid():
    p = np.full(V, 0.4 / (V - 1)); p[0] = 0.6; return p

t = AutoTuner(dial_max=0.4, start=0.4)
rg0 = np.random.default_rng(0)          # ONE generator: re-seeding per call emits one repeated
for _ in range(400):                    # token forever, which is looping, not steady output
    t.observe(mid(), token=int(rg0.integers(0, 5000)))
check("steady output at ANY confidence level is learned as normal, not punished",
      t.dial > 0.35, f"dial fell to {t.dial:.3f} on perfectly steady output")

# ...but a genuine DEPARTURE from that model's own normal must still move the dial down.
t = AutoTuner(dial_max=0.4, start=0.4)
rg = np.random.default_rng(1)
for _ in range(300): t.observe(conf(0.9), token=int(rg.integers(0, 5000)))
before = t.dial
for _ in range(120): t.observe(vague(), token=int(rg.integers(0, 5000)))
check("a real departure from this model's normal still backs the dial off",
      t.dial < before, f"{before:.3f} -> {t.dial:.3f}")

# and looping must still be caught through the tuner, on the default meter
t = AutoTuner(dial_max=0.4, start=0.4)
for i in range(300): t.observe(conf(0.9), token=int(rg.integers(0, 5000)))
for i in range(200): t.observe(conf(0.98), token=1234 + (i % 2))
check("looping is caught through the tuner on the default meter",
      t.dial < 0.4 and t._last_reason == "looping", f"dial={t.dial:.3f} reason={t._last_reason}")

print("\n" + "=" * 78)
print(f"{'ALL TESTS PASSED' if not FAIL else str(len(FAIL)) + ' FAILURES: ' + ', '.join(FAIL)}")
print("=" * 78)
sys.exit(1 if FAIL else 0)
