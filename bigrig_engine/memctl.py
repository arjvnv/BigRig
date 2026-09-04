"""Give memory back when the machine needs it, between replies and never during one.

WHY THIS EXISTS
    BigRig's expert pool is ordinary anonymous memory. Measured on a live server holding 4.64 GB:

        class        stopped   running    delta
        wired          2.94      3.00     +0.06     <- barely moves
        anonymous      7.48     12.48     +5.00     <- the pool lives here
          of which inactive     +1.55               <- reclaimed first

    So macOS may compress or swap it, and under pressure it will -- taking memory back from us
    rather than failing whatever else asked for it. Every expert it reclaims turns a cache hit
    into the disk read that streaming exists to avoid, and nothing anywhere reports a problem.
    The model just gets slower and the user blames the model.

    Worse, the same run measures 1.03 GB of process RSS against 4.64 GB of MLX memory. A
    controller reading RSS would see a gigabyte and conclude it had room it does not have. The
    only trustworthy signal is the one `calibrate.under_pressure` already reads: the compressor
    GROWING, or swap being written, during a short window. Not its absolute occupancy, which is
    cumulative and sits at a gigabyte on an idle machine after a few heavy runs.

WHAT IT WILL AND WILL NOT DO
    It gives memory back under pressure, and takes back what it lent once the machine is quiet
    again. The two halves are deliberately NOT symmetric, because their failure modes are not:
    a wrong shrink makes a model slower on a machine that had room, which is recoverable and
    visible in the interface, while a wrong grow is how a controller makes a machine unusable.
    So shrinking needs two consecutive readings and moves in fifths; growing needs three unbroken
    minutes of quiet and moves in tenths.

    It never goes above the capacity it started from. `bigrig knee` is the only thing allowed to
    decide the ceiling, because that number came from timing the model; a controller that grew
    past it would be overruling the one measurement in the system. Growing is off unless asked
    for, and shrinking works without it.

    It acts only between replies. Releasing memory mid-reply would mean rebuilding the pool with
    a generation in flight, and the pool cannot be resized without rebuilding it. A reply is
    bounded -- at single-digit tokens a second and a few thousand tokens, minutes at worst -- so
    the wait is bounded too.

    It requires the same reading twice and rate-limits itself, because a single sample is noise
    and a controller that reacts to noise oscillates: shrink, recover, shrink, recover, with a
    model reload each time.
"""
from __future__ import annotations

import time

# One reading is noise. Two consecutive readings, taken at least a poll apart, is a trend.
CONFIRMATIONS = 2

# The smallest gap between two actions. A reload costs about a second and a half and the machine
# needs time to settle before the next reading means anything; without this the controller would
# shrink to its floor during one burst of pressure.
MIN_INTERVAL_S = 45.0

# How much to give back at once, as a fraction of the current pool. Small enough that one bad
# reading is cheap, large enough that a real squeeze is relieved in two or three steps.
STEP_FRACTION = 0.20

# TAKING MEMORY BACK IS NOT THE MIRROR OF GIVING IT UP, AND IS DELIBERATELY SLOWER.
#     A wrong shrink costs speed on a machine that had room: recoverable, visible, and the user
#     can see the capacity in the interface. A wrong grow is how a controller takes a machine
#     down. So growing needs a longer quiet spell than shrinking needs a squeeze, moves in
#     smaller steps, and stops at the capacity it started from -- it restores what it borrowed
#     and never goes past it. `bigrig knee` is the only thing allowed to decide the ceiling.
GROW_QUIET_S = 180.0
GROW_STEP_FRACTION = 0.10

# THE FIRST MINUTE IS THE ENGINE ITSELF, NOT A SQUEEZE.
#     Loading a model and warming the page cache move gigabytes, and the compressor reading
#     cannot tell our own start-up from someone else's shortage. Measured: a server shrank
#     0.8 s after it started, before it had served a token, on pressure its own load had just
#     caused. A reading inside this window is not counted toward a shrink; after it, two
#     confirmations in fifths, as before. Growing is unaffected -- nothing has been borrowed
#     yet -- and so is the prompt-cache release, which costs nothing to undo.
GRACE_S = 60.0


class ShrinkPolicy:
    """Decides whether to give memory back. Pure: no MLX, no clock of its own, no I/O.

    Everything that could make this dangerous is a decision, and every decision is here where it
    can be tested exhaustively without allocating a byte.
    """

    def __init__(self, floor: int, step_fraction: float = STEP_FRACTION,
                 min_interval_s: float = MIN_INTERVAL_S, confirmations: int = CONFIRMATIONS,
                 grow: bool = False, ceiling: int | None = None,
                 grow_quiet_s: float = GROW_QUIET_S,
                 grow_step_fraction: float = GROW_STEP_FRACTION,
                 started: float | None = None, grace_s: float = GRACE_S):
        if floor < 1:
            raise ValueError(f"floor must be at least 1 expert per layer, got {floor}")
        if ceiling is not None and int(ceiling) < int(floor):
            raise ValueError(f"ceiling {ceiling} is below the floor {floor}")
        self.floor = int(floor)
        self.step_fraction = max(0.01, min(0.9, float(step_fraction)))
        self.min_interval_s = max(0.0, float(min_interval_s))
        self.confirmations = max(1, int(confirmations))
        self.grow = bool(grow)
        # Never above where it started. Growing past the measured knee would be the controller
        # overruling the only number here that came from timing the model.
        self.ceiling = int(ceiling) if ceiling is not None else None
        self.grow_quiet_s = max(0.0, float(grow_quiet_s))
        self.grow_step_fraction = max(0.01, min(0.9, float(grow_step_fraction)))
        # When the server started, on the caller's clock, or None for no grace at all.
        self.started = float(started) if started is not None else None
        self.grace_s = max(0.0, float(grace_s))
        self.streak = 0
        self.last_action = 0.0
        self.quiet_since = None            # when the machine last stopped being short of memory
        self.shrinks = 0
        self.grows = 0
        self.released_from = None          # the capacity it started at, so grow knows where home is
        self.last_reason = ""

    def decide(self, capacity: int, pressure: bool, now: float, idle: bool = True):
        """(new capacity, why) if memory should be given back now, else None.

        `idle` is not advisory. A caller that passes True while a reply is in flight will get a
        decision that rebuilds the pool underneath a running generation.
        """
        if not idle:
            self.last_reason = "a reply is in flight"
            return None
        if pressure and self.in_grace(now):
            # Not a confirmation, and not the end of a streak either: the window simply does
            # not count. A squeeze that outlives it is confirmed from scratch afterwards.
            self.streak = 0
            self.quiet_since = None
            self.last_reason = (f"started {now - self.started:.0f}s ago; pressure in the first "
                                f"{self.grace_s:.0f}s is the load itself, not a squeeze")
            return None
        if not pressure:
            # A single quiet reading ends the streak. Pressure that comes and goes is not the
            # sustained squeeze this is for.
            self.streak = 0
            if self.quiet_since is None:
                self.quiet_since = now
            return self._maybe_grow(capacity, now)
        self.quiet_since = None
        self.streak += 1
        if self.streak < self.confirmations:
            self.last_reason = (f"pressure seen {self.streak} time"
                                f"{'' if self.streak == 1 else 's'}, waiting for "
                                f"{self.confirmations}")
            return None
        if self.last_action and (now - self.last_action) < self.min_interval_s:
            self.last_reason = (f"gave memory back {now - self.last_action:.0f}s ago, "
                                f"waiting {self.min_interval_s:.0f}s between changes")
            return None
        if capacity <= self.floor:
            self.last_reason = (f"already at the floor of {self.floor} experts a layer; "
                                f"any smaller cannot serve a token")
            return None
        step = max(1, int(round(capacity * self.step_fraction)))
        new = max(self.floor, capacity - step)
        if new >= capacity:                       # cannot happen with step >= 1; belt and braces
            self.last_reason = "no smaller capacity available"
            return None
        if self.released_from is None:
            self.released_from = int(capacity)
        self.last_action = now
        self.shrinks += 1
        self.streak = 0                           # a fresh trend must be established after acting
        self.last_reason = (f"the machine is short of memory, so {capacity - new} experts a "
                            f"layer were given back ({capacity} -> {new})")
        return int(new), self.last_reason

    def _maybe_grow(self, capacity: int, now: float):
        """Take a little back, but only after a long quiet spell and never past where it started.

        Called only when the machine is NOT short of memory. Everything about it is more
        cautious than shrinking, because the failure modes are not symmetric: giving too much
        memory back makes a model slow, taking too much back makes a machine unusable.
        """
        if not self.grow:
            self.last_reason = "no memory pressure"
            return None
        home = self.released_from if self.released_from is not None else self.ceiling
        if home is None or capacity >= home:
            self.last_reason = "no memory pressure; nothing was borrowed to give back"
            return None
        quiet_for = now - (self.quiet_since if self.quiet_since is not None else now)
        if quiet_for < self.grow_quiet_s:
            self.last_reason = (f"quiet for {quiet_for:.0f}s; waiting "
                                f"{self.grow_quiet_s:.0f}s before taking memory back")
            return None
        if self.last_action and (now - self.last_action) < self.min_interval_s:
            self.last_reason = (f"changed capacity {now - self.last_action:.0f}s ago, "
                                f"waiting {self.min_interval_s:.0f}s between changes")
            return None
        step = max(1, int(round(capacity * self.grow_step_fraction)))
        new = min(int(home), capacity + step)
        if new <= capacity:
            self.last_reason = "already back to where it started"
            return None
        self.last_action = now
        self.quiet_since = now             # a fresh quiet spell must elapse before the next step
        self.grows += 1
        self.last_reason = (f"the machine has been quiet for {quiet_for:.0f}s, so "
                            f"{new - capacity} experts a layer were taken back "
                            f"({capacity} -> {new}, home is {home})")
        return int(new), self.last_reason

    def in_grace(self, now: float) -> bool:
        return self.started is not None and (now - self.started) < self.grace_s

    def stats(self) -> dict:
        return {"shrinks": self.shrinks, "grows": self.grows, "floor": self.floor,
                "grow": self.grow, "ceiling": self.ceiling,
                "grace_s": self.grace_s, "in_grace": self.in_grace(time.time()),
                "streak": self.streak, "released_from": self.released_from,
                "last_reason": self.last_reason,
                "seconds_since_action": (round(time.time() - self.last_action, 1)
                                         if self.last_action else None)}


def floor_for(top_k: int, n_experts: int) -> int:
    """The smallest pool that can still serve a step.

    A layer routing to top_k experts needs at least that many slots or `_chunks` splits every
    single step, which is correct and ruinous. One above top_k leaves room for the pool to hold
    anything at all between steps.
    """
    return max(1, min(int(n_experts), int(top_k) + 1))
