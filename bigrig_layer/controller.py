"""AutoTuner — the closed loop that uses the quality meter to set the speed/quality dial itself.

WHY THIS EXISTS
    Every local-inference system today makes the USER pick a speed/quality setting and live with
    it. Pick conservatively and you give up speed you could have had; pick aggressively and you
    may be running in a degraded state without knowing. The meter makes a third option possible:
    push for speed while quality holds, and back off the moment it does not.

WHAT IT CAN AND CANNOT KNOW -- read this before trusting it
    The meter reports THAT quality is poor, not WHY. A damage alarm -- attributing degradation to
    the dial specifically -- was built, tested and REFUTED (placebo 0.75 > claimed 0.55; see). So this controller cannot distinguish "the dial broke it" from
    "this prompt is intrinsically hard".

    The consequence is concrete and must be stated: ON A HARD PROMPT THE CONTROLLER WILL BACK OFF
    UNNECESSARILY. It loses speed, not quality. That is the tolerable direction to fail in, and it
    is the reason the loop is asymmetric -- fast to retreat, slow to advance.
"""
from __future__ import annotations
from .adaptive import AdaptiveMeter
from .meter import QualityMeter


class AutoTuner:
    """Adjusts a scalar quality/speed dial from the live quality signal.

    `dial` is whatever your engine exposes: a cache-aware routing strength, a quantisation
    aggressiveness, an expert budget. 0.0 = safest/slowest, `dial_max` = fastest/riskiest.

        tuner = AutoTuner(dial_max=0.4)
        for step in generation:
            tuner.observe(probs)
            engine.set_dial(tuner.dial)
    """

    def __init__(self, dial_max: float = 0.4, dial_min: float = 0.0,
                 start: float | None = None, *, up_step: float = 0.02,
                 down_step: float = 0.10, patience: int = 3,
                 meter=None, strict: bool = True):
        if dial_min > dial_max:
            raise ValueError("dial_min must not exceed dial_max")
        self.dial_max, self.dial_min = float(dial_max), float(dial_min)
        self.dial = float(dial_max if start is None else start)
        self.dial = min(max(self.dial, self.dial_min), self.dial_max)
        # ASYMMETRIC BY DESIGN: retreat 5x faster than we advance. A false alarm costs a little
        # speed; a missed degradation costs the user's output.
        self.up_step, self.down_step = float(up_step), float(down_step)
        self.patience = int(patience)
        # DEFAULTS TO THE SELF-CALIBRATING METER, because the tuner has no way to know which
        # model it is attached to. QualityMeter's constants were fitted on Ling-mini-2.0-3bit
        # and its repetition threshold fires on 21.6% of HEALTHY output of a base model -- the
        # tuner would read those false alarms as damage and give away speed it never needed to.
        # Pass meter=QualityMeter() explicitly when the constants have been fitted for YOUR
        # model; that is strictly better, but only once someone has actually done the fitting.
        self.meter = meter or AdaptiveMeter(strict=strict)
        self._healthy_run = 0
        self._last_reason: str | None = None
        self.history: list[tuple[int, float, float | None, bool]] = []
        self.n_backoffs = 0

    def observe(self, probs=None, *, logits=None, token: int | None = None) -> float:
        """Feed one generation step. Returns the dial the engine should now use.

        PASS `token` -- the id actually emitted. Without it the loop is blind to repetition,
        which is the most common visible failure. A looping model is maximally CONFIDENT, so
        every probability-based signal reads healthy while it emits "is is is is is".
        """
        self.meter.observe(probs=probs, logits=logits)
        if token is not None:
            self.meter.observe_token(token)
        deg = self.meter.is_degraded()
        if deg is None:                       # window not full yet -- do not act on no evidence
            return self.dial
        if deg:
            self.dial = max(self.dial_min, self.dial - self.down_step)
            self._healthy_run = 0
            self.n_backoffs += 1
        else:
            self._healthy_run += 1
            if self._healthy_run >= self.patience:
                self.dial = min(self.dial_max, self.dial + self.up_step)
                self._healthy_run = 0
        self.history.append((self.meter.n_observed, self.dial, self.meter.score(), bool(deg)))
        if deg:
            self._last_reason = self.meter.reason()
        return self.dial

    def reset(self, start: float | None = None) -> None:
        self.meter.reset()
        self._healthy_run = 0
        self.n_backoffs = 0
        self.history.clear()
        if start is not None:
            self.dial = min(max(float(start), self.dial_min), self.dial_max)

    def summary(self) -> dict:
        sc = [s for _, _, s, _ in self.history if s is not None]
        return {"final_dial": self.dial, "backoffs": self.n_backoffs,
                "last_reason": self._last_reason,
                "steps": len(self.history),
                "mean_score": (sum(sc) / len(sc)) if sc else None,
                "degraded_frac": (sum(1 for *_, d in self.history if d) / len(self.history))
                                 if self.history else None}
