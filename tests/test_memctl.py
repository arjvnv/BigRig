"""The shrink controller, tested for every way it could damage a machine or a reply.

WHY THIS FILE IS WRITTEN TO BREAK IT
    This is the one component that acts on the user's hardware without being asked. A capacity
    chooser that is wrong makes a model slow; a memory controller that is wrong rebuilds the pool
    underneath a running reply, or oscillates until the machine is unusable, or shrinks to a
    capacity that cannot serve a token at all.

    So the policy is pure -- no MLX, no clock, no I/O -- and every rule it has is exercised here
    against adversarial inputs: pressure that flickers, clocks that jump, floors that are already
    reached, callers that lie about being idle.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from bigrig_engine.memctl import ShrinkPolicy, floor_for                 # noqa: E402

FAIL = []


def check(n, c, d=""):
    (print(f"  PASS  {n}") if c else (FAIL.append(n), print(f"  FAIL  {n}  {d}")))


def _ceiling_refused():
    try:
        ShrinkPolicy(floor=20, ceiling=5)
        return False
    except ValueError:
        return True


def fresh(floor=9, **kw):
    return ShrinkPolicy(floor=floor, **kw)


print("=" * 84)
print("1. IT NEVER TOUCHES A REPLY IN FLIGHT")
print("=" * 84)
p = fresh()
# The single most damaging thing it could do: rebuild the pool with a generation running.
check("under pressure but busy, it does nothing",
      p.decide(40, True, 100.0, idle=False) is None)
check("...however many times it is asked",
      all(p.decide(40, True, 100.0 + i, idle=False) is None for i in range(20)))
check("...and being busy does not secretly build up a streak",
      p.streak == 0, f"streak {p.streak}")
check("it says why it declined", "in flight" in p.last_reason)
check("once idle, the same pressure is acted on normally",
      p.decide(40, True, 200.0) is None and p.decide(40, True, 201.0) is not None)

print()
print("=" * 84)
print("2. ONE READING IS NOISE; IT WAITS FOR A TREND")
print("=" * 84)
p = fresh()
check("the first pressure reading does nothing", p.decide(40, True, 10.0) is None)
check("...and says it is waiting", "waiting for 2" in p.last_reason, p.last_reason)
check("the second acts", p.decide(40, True, 11.0) is not None)
p = fresh()
p.decide(40, True, 10.0)
check("a quiet reading in between resets the streak",
      p.decide(40, False, 11.0) is None and p.decide(40, True, 12.0) is None)
_flick = fresh()          # its own policy: the checks above leave a streak behind
check("...so flickering pressure never triggers a change",
      all(_flick.decide(40, i % 2 == 0, 20.0 + i) is None for i in range(40)))
check("a policy needing three confirmations waits for three",
      [fresh(confirmations=3).decide(40, True, t) for t in (1, 2)].count(None) == 2)

print()
print("=" * 84)
print("3. IT CANNOT SHRINK BELOW WHAT CAN SERVE A TOKEN")
print("=" * 84)
p = fresh(floor=9)
cap, seen = 40, []
for i in range(200):                                  # relentless pressure, clock always moving
    d = p.decide(cap, True, 1000.0 + i * 60)
    if d:
        cap = d[0]
        seen.append(cap)
check("relentless pressure walks it down to the floor and stops",
      cap == 9, f"ended at {cap}")
check("...and never below it", min(seen) >= 9, f"min {min(seen)}")
check("...and never past it in a single step", all(c >= 9 for c in seen))
check("at the floor it declines and says so",
      p.decide(9, True, 99999.0) is None and "floor" in p.last_reason, p.last_reason)
check("a capacity already under the floor is left alone, not raised",
      fresh(floor=9).decide(5, True, 1.0) is None
      or fresh(floor=9).decide(5, True, 1.0)[0] >= 5)
check("every step is strictly downward", all(b < a for a, b in zip([40] + seen, seen)))
try:
    ShrinkPolicy(floor=0)
    _bad_floor_ok = True
except ValueError:
    _bad_floor_ok = False
check("a floor below one expert is refused outright, not silently corrected", not _bad_floor_ok)

print()
print("=" * 84)
print("4. IT NEVER GROWS, WHATEVER IT IS TOLD")
print("=" * 84)
p = fresh()
grew = []
for i in range(100):
    d = p.decide(30, i % 3 != 0, 500.0 + i * 100)
    if d and d[0] > 30:
        grew.append(d)
check("no sequence of readings ever returns a larger capacity", not grew, f"{grew[:2]}")
check("absence of pressure is not an instruction to grow",
      all(p.decide(20, False, 5000.0 + i) is None for i in range(50)))

print()
print("=" * 84)
print("5. IT DOES NOT THRASH")
print("=" * 84)
p = fresh(min_interval_s=45.0)
first = p.decide(40, True, 0.0) or p.decide(40, True, 1.0)
check("it acts once", first is not None)
# Acting resets the streak, so a trend has to rebuild before the rate limit is even consulted.
# Both gates must hold: neither alone is enough to stop a second change one second later.
check("a second action one second later is refused",
      all(p.decide(first[0], True, 2.0 + i) is None for i in range(3)))
check("...and once a trend rebuilds it is the rate limit that says no",
      "waiting 45s" in p.last_reason, p.last_reason)
acted = 0
cap = 40
p2 = fresh(min_interval_s=45.0)
for i in range(600):                                  # ten minutes of unbroken pressure, 1s poll
    d = p2.decide(cap, True, float(i))
    if d:
        cap = d[0]
        acted += 1
check("ten minutes of unbroken pressure produces a handful of changes, not hundreds",
      acted <= 14, f"{acted} changes")
check("...and each one was a real reduction", cap < 40)

print()
print("=" * 84)
print("6. A CLOCK THAT MISBEHAVES CANNOT UNLOCK IT")
print("=" * 84)
p = fresh(min_interval_s=45.0)
p.decide(40, True, 1000.0)
d = p.decide(40, True, 1001.0)
check("it acted once at a normal clock", d is not None)
# A clock that jumps backwards must not read as "the interval has elapsed".
back = [p.decide(d[0], True, 1001.0 - j) for j in range(1, 30)]
check("a clock jumping backwards does not unlock another change",
      all(x is None for x in back), f"{[x for x in back if x][:2]}")
_fwd = fresh(min_interval_s=45.0)
_fwd.decide(40, True, 1000.0)
_d2 = _fwd.decide(40, True, 1001.0)
check("a clock jumping far forwards does unlock one, which is correct",
      _fwd.decide(_d2[0], True, 1e9) is None
      and _fwd.decide(_d2[0], True, 1e9 + 1) is not None)

print()
print("=" * 84)
print("7. THE STEP IS SANE AT EVERY SIZE")
print("=" * 84)
for cap in (10, 11, 13, 40, 128, 999):
    q = fresh(floor=9)
    q.decide(cap, True, 0.0)
    d = q.decide(cap, True, 1.0)
    if cap <= 9:
        continue
    check(f"a pool of {cap} shrinks by at least one expert and not below the floor",
          d is not None and 9 <= d[0] < cap, f"{d}")
q = fresh(floor=9, step_fraction=0.0001)
q.decide(40, True, 0.0)
check("an absurdly small step still moves by one, never by zero",
      q.decide(40, True, 1.0)[0] == 39)
q = fresh(floor=9, step_fraction=50.0)
q.decide(40, True, 0.0)
check("an absurdly large step is clamped and still lands on the floor",
      q.decide(40, True, 1.0)[0] == 9)

print()
print("=" * 84)
print("8. WHAT IT REPORTS IS TRUE")
print("=" * 84)
p = fresh()
p.decide(40, True, 0.0)
d = p.decide(40, True, 1.0)
st = p.stats()
check("it counts the changes it made", st["shrinks"] == 1)
check("it remembers where it started, so a grow half could undo it",
      st["released_from"] == 40)
check("the reason names both capacities", f"40 -> {d[0]}" in d[1], d[1])
check("the reason is the same string it reports in stats", st["last_reason"] == d[1])
check("a policy that has done nothing reports no elapsed time",
      fresh().stats()["seconds_since_action"] is None)
check("the floor it reports is the floor it enforces", st["floor"] == 9)

print()
print("=" * 84)
print("9. THE FLOOR IS DERIVED FROM THE MODEL, NOT PICKED")
print("=" * 84)
check("the floor sits just above top-k", floor_for(8, 128) == 9)
check("...for every model shape", all(floor_for(k, 128) == k + 1 for k in (2, 4, 8, 16)))
check("it can never exceed the expert count", floor_for(64, 8) == 8)
check("...nor fall below one", floor_for(0, 128) >= 1)

print()
print("=" * 84)
print("10. TAKING MEMORY BACK IS SLOWER AND MORE CAUTIOUS THAN GIVING IT UP")
print("=" * 84)


def lent(cap=43, floor=9, **kw):
    """A policy that has already given memory back once, so there is something to take back."""
    q = ShrinkPolicy(floor=floor, grow=True, ceiling=cap, **kw)
    q.decide(cap, True, 0.0)
    d = q.decide(cap, True, 1.0)
    return q, d[0]


q, cap = lent()
check("it does not take memory back the moment pressure stops",
      q.decide(cap, False, 2.0) is None)
check("...nor after a short quiet spell", q.decide(cap, False, 100.0) is None)
check("...and it says how long it is waiting", "waiting 180s" in q.last_reason, q.last_reason)
d = q.decide(cap, False, 400.0)
check("after three unbroken quiet minutes it takes a little back", d is not None, q.last_reason)
check("...a SMALLER step than it gave up", d[0] - cap < 43 - cap, f"{cap} -> {d[0]}")
check("...and it says where home is", "home is 43" in d[1], d[1])

# The ceiling is the whole safety property of the grow half.
q2, cap2 = lent()
c, t = cap2, 1000.0
for _ in range(400):
    r = q2.decide(c, False, t)
    if r:
        c = r[0]
    t += 60
check("unbroken quiet walks it home and stops there", c == 43, f"ended at {c}")
check("...and never one expert past it", c <= 43)
check("at home it declines and says so",
      q2.decide(43, False, t + 1e6) is None and "nothing was borrowed" in q2.last_reason,
      q2.last_reason)

# A squeeze during recovery must reset the clock, or a flickering machine ratchets upward.
q3, cap3 = lent()
q3.decide(cap3, False, 100.0)
q3.decide(cap3, True, 150.0)                      # one squeeze
check("a single squeeze restarts the quiet clock",
      q3.decide(cap3, False, 200.0) is None and "waiting 180s" in q3.last_reason, q3.last_reason)

# Growing is opt-in. A policy that was never asked to grow must never do it.
q4 = ShrinkPolicy(floor=9, ceiling=43)
q4.decide(43, True, 0.0)
low = q4.decide(43, True, 1.0)[0]
check("a policy not asked to grow never grows, however long it is quiet",
      all(q4.decide(low, False, t) is None for t in range(100, 100000, 997)))
check("...and says only that there is no pressure", "no memory pressure" in q4.last_reason)

check("a ceiling below the floor is refused rather than silently swapped",
      _ceiling_refused())
check("what it reports counts both directions",
      set(q.stats()) >= {"shrinks", "grows", "floor", "ceiling", "grow"})
check("...and the grow count is real", q.stats()["grows"] >= 1)

print("\n" + "=" * 84)
print("11. THE FIRST MINUTE IS THE LOAD, NOT A SQUEEZE")
print("=" * 84)
# Measured: a server shrank 0.8 s after it started, before it had served a token, on pressure
# its own load had just caused. Readings inside the grace window must not count -- and must not
# count LATER either, as a half-built streak waiting for one more glance.
g = ShrinkPolicy(floor=9, ceiling=43, started=1000.0)
check("the default grace is a minute, not zero and not forever",
      30.0 <= g.grace_s <= 180.0, str(g.grace_s))
check("pressure inside the window is not a confirmation",
      g.decide(43, True, 1000.5) is None and g.decide(43, True, 1010.0) is None
      and g.decide(43, True, 1059.9) is None)
check("...and it says why in words about the load", "load itself" in g.last_reason,
      g.last_reason)
check("...and it left no streak behind to be completed later", g.streak == 0)
check("the first reading after the window starts a streak from zero, so one reading is "
      "still not enough", g.decide(43, True, 1060.0) is None and g.streak == 1)
check("...and the second one shrinks, exactly as it would have without any grace",
      g.decide(43, True, 1070.0) is not None and g.shrinks == 1)
check("it reports whether it is in the window", g.stats()["in_grace"] is False
      and "grace_s" in g.stats())
check("a policy given no start time has no grace, so nothing that relied on the old "
      "behaviour changes", ShrinkPolicy(floor=9).in_grace(0.0) is False)
# The window guards SHRINKING. Nothing has been borrowed at start-up, so there is nothing for it
# to guard on the grow side; and quiet inside the window must still count as quiet afterwards.
g2 = ShrinkPolicy(floor=9, ceiling=43, grow=True, started=0.0)
check("a quiet reading inside the window is not turned into pressure",
      g2.decide(43, False, 10.0) is None and "nothing was borrowed" in g2.last_reason,
      g2.last_reason)
g3 = ShrinkPolicy(floor=9, ceiling=43, started=0.0, grace_s=0.0)
g3.decide(43, True, 0.1)
check("a zero-second grace is exactly the old policy",
      g3.decide(43, True, 0.2) is not None)
# Pressure that OUTLIVES the window is real, and must be confirmed from scratch, not waved
# through because it was seen earlier.
g4 = ShrinkPolicy(floor=9, ceiling=43, started=0.0)
for t in (1.0, 20.0, 40.0, 59.0):
    g4.decide(43, True, t)
check("four readings inside the window do not pre-confirm the first one outside it",
      g4.decide(43, True, 61.0) is None and g4.streak == 1)
check("...the next one does", g4.decide(43, True, 71.0) is not None)

print()
print("=" * 84)
print("ALL TESTS PASSED" if not FAIL else f"{len(FAIL)} FAILURES: " + ", ".join(FAIL))
print("=" * 84)
sys.exit(1 if FAIL else 0)
