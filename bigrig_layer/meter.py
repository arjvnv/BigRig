"""The quality meter itself. Pure numpy, no framework dependency."""
from __future__ import annotations
import math
from collections import deque

import numpy as np

# Fitted on Ling-mini-2.0-3bit, 24 prompts x 6 categories x 4 seeds x 6 toll settings,
# CALIBRATION prompts only, evaluated once on held-out prompts (rho=0.893, robust 0.844-0.914
# across four rotated splits).
CALIBRATION = {
    "model": "Ling-mini-2.0-3bit",
    "features": ["entropy", "top1", "margin"],
    "mean": [0.5918, 0.8352, 0.7518],
    "std": [0.5178, 0.0972, 0.1209],
    "weight": [-0.737, -2.953, 1.629],
    "intercept": -0.419,
    "rho_heldout": 0.893,
    # threshold at 10% false-positive rate on healthy calibration windows
    "threshold": -0.143,
    # SECOND, INDEPENDENT CHECK. The confidence signals above are BLIND TO LOOPING: a model
    # stuck repeating "is is is is" is maximally confident, so entropy is low, top-1 is high and
    # the margin is wide -- every confidence signal reads HEALTHY. Observed directly: broken
    # looping output scored -1.167 while healthy prose scored -0.983, i.e. the meter rated the
    # broken text BETTER.
    #
    # The research did not catch this because the meter was validated against judge-NLL, which
    # the earlier result had already shown is ALSO blind to looping. Validating a blind
    # metric against a blind metric produced rho=0.893 -- they agree because they share the gap.
    #
    # So repetition is checked SEPARATELY and combined with OR, not folded into the regression.
    # Healthy Ling-mini-2.0-3bit scores 0.035; visibly broken output scores 0.15-0.47.
    #
    # THIS CONSTANT IS MODEL-SPECIFIC, which was assumed otherwise until it was measured. On
    # OLMoE-1B-7B-0125, a BASE model with no chat template, healthy windows average 0.085 with a
    # p99 of 0.410, so this threshold fires on 21.6% of undamaged output. If you are not running
    # Ling-mini-2.0-3bit, either refit it with `calibrate(..., repetitions=...)` or use
    # AdaptiveMeter, which learns the rate and gets that false-alarm rate down to 3.4%.
    "repetition_threshold": 0.15,
    "repetition_n": 4,
    "window": 64,      # in observations; at the default stride=1 that is 64 tokens
}


class QualityMeter:
    """A live estimate of how poor the current output is.

    Higher score = worse. `is_degraded()` compares against a threshold calibrated at a 10%
    false-positive rate.

    The meter needs `window` tokens before it will report anything; until then `score()` returns
    None. In testing it flagged degraded output at the first available window (~64 tokens, about
    36 words) with 100% recall -- but 64 IS the window size, so that is an upper bound on
    latency, not a measurement of it.
    """

    def __init__(self, calibration: dict | None = None, window: int | None = None,
                 strict: bool = True, stride: int = 1):
        """strict=True (default) RAISES on a malformed distribution. strict=False skips it and
        counts it in `n_skipped`.

        The default is strict on purpose. A monitor that silently returns a wrong number is
        worse than no monitor -- that is the entire premise of this library, and it would be
        incoherent to then let NaN logits produce a confident-looking score. Testing found
        exactly that: a single NaN anywhere in the distribution turned the output into NaN with
        no warning."""
        c = dict(CALIBRATION)
        if calibration:
            c.update(calibration)
        self.c = c
        self.window = int(window or c["window"])
        self.strict = bool(strict)
        # SAMPLE EVERY `stride` TOKENS. DEFAULT 1 (every token) BECAUSE stride>1 IS UNVALIDATED.
        #
        # Measured overhead at stride=1 is +1.8 to +2.7 ms/token, i.e. 5-8% of a token on
        # Ling-mini-2.0-3bit (data/results/layer_overhead*.json). Subsampling would divide that.
        #
        # An earlier version defaulted to stride=4 on the strength of a test showing rho held at
        # 0.886. That test was WRONG for this purpose: it subsampled how often the SCORE was
        # computed, while every score still used all 64 tokens. Skipping tokens is a different
        # operation and its effect on accuracy has not been measured. stride>1 is therefore
        # offered but explicitly unvalidated -- do not rely on it without re-calibrating.
        self.stride = max(1, int(stride))
        self._seen = 0
        self._buf: deque[tuple[float, float, float]] = deque(maxlen=self.window)
        self._toks: deque[int] = deque(maxlen=self.window)
        self.n_observed = 0
        self.n_skipped = 0

    # ---------------------------------------------------------------- input
    def observe(self, probs=None, *, logits=None) -> None:
        """Feed one step. Give EITHER `probs` (already normalised) or `logits`.

        Accepts anything numpy can read -- a numpy array, a torch tensor on CPU, a list.
        Cost is three reductions over the vocabulary; it does not copy the distribution.
        """
        if probs is None and logits is None:
            raise ValueError("observe() needs probs= or logits=")
        self._seen += 1
        if (self._seen - 1) % self.stride:
            return                      # sampled out -- see `stride` in __init__
        raw = np.asarray(logits if probs is None else probs, dtype=np.float64).ravel()

        # ---- validation. A malformed distribution must never become a plausible-looking score.
        bad = None
        if raw.size < 2:
            bad = f"need at least 2 entries, got {raw.size}"
        elif not np.all(np.isfinite(raw)):
            n = int((~np.isfinite(raw)).sum())
            bad = f"{n} non-finite value(s) (NaN or inf) in the distribution"
        if bad is None and probs is not None:
            if np.any(raw < 0):
                bad = "probabilities contain negative values"
            elif raw.sum() <= 0:
                bad = "probabilities sum to zero"
        if bad is not None:
            self.n_skipped += 1
            if self.strict:
                raise ValueError(
                    f"QualityMeter.observe(): {bad}. This usually means the engine produced an "
                    f"invalid distribution. Pass strict=False to skip such steps instead.")
            return

        if probs is None:
            a = raw - raw.max()
            p = np.exp(a)
            p /= p.sum()
        else:
            p = raw
            t = p.sum()
            if not (0.98 < t < 1.02):
                p = p / t
        # partial sort is enough for the top two -- no full sort of the vocabulary
        k = np.argpartition(p, -2)[-2:]
        t1, t2 = np.sort(p[k])[::-1]
        ent = float(-(p[p > 0] * np.log(p[p > 0])).sum())
        self._buf.append((ent, float(t1), float(t1 - t2)))
        self.n_observed += 1

    def observe_token(self, token: int) -> None:
        """Feed the token that was actually emitted. REQUIRED for looping detection.

        Without this the meter can only see the confidence signals, which are blind to
        repetition -- and repetition is the most visually obvious way a model breaks.
        """
        self._toks.append(int(token))

    def repetition(self) -> float | None:
        """Fraction of n-grams in the window that already appeared. None until enough tokens."""
        n = self.c["repetition_n"]
        if len(self._toks) < 2 * n:
            return None
        t = list(self._toks)
        g = [tuple(t[i:i + n]) for i in range(len(t) - n + 1)]
        seen, rep = set(), 0
        for x in g:
            if x in seen: rep += 1
            else: seen.add(x)
        return rep / len(g)

    def observe_stats(self, entropy: float, top1: float, margin: float) -> None:
        """FAST PATH. Feed the three statistics directly, computed however you like.

        `observe(probs)` has to move the whole distribution from device to host. Measured on
        Ling-mini-2.0-3bit (157k vocab) that costs ~2.5 ms/token -- about 7% of a token, which
        is NOT free. The transfer is the cost, not the arithmetic: three scalars is 12 bytes,
        the distribution is ~628 KB.

        So an engine that cares about speed computes these three on-device and passes them here:

            ent    = -(p * log p).sum()
            top1   = p.max()
            margin = top1 - second-largest(p)

        That is the same arithmetic `observe()` does, moved to where the data already lives.
        """
        self._seen += 1
        if (self._seen - 1) % self.stride:
            return
        vals = (float(entropy), float(top1), float(margin))
        if not all(v == v and abs(v) != float("inf") for v in vals):
            self.n_skipped += 1
            if self.strict:
                raise ValueError("observe_stats(): non-finite value in (entropy, top1, margin)")
            return
        self._buf.append(vals)
        self.n_observed += 1

    # ---------------------------------------------------------------- output
    def should_observe(self) -> bool:
        """True if the NEXT observe() call will actually be used.

        For the fast path this lets you skip computing the statistics at all on sampled-out
        steps -- which is where the saving comes from:

            if meter.should_observe():
                ...compute ent/top1/margin on device...
                meter.observe_stats(ent, top1, margin)
            else:
                meter.observe_stats(0, 0, 0)   # or simply call nothing; see note below

        NOTE: observe_stats() counts the step itself, so if you skip the call entirely you must
        not call it at all for that step -- the meter's counter only advances when called.
        Prefer calling observe_stats() every step and letting it drop the sampled-out ones,
        unless the device-side computation is what you are trying to avoid.
        """
        return self._seen % self.stride == 0 if self.stride > 1 else True

    def ready(self) -> bool:
        return len(self._buf) >= self.window

    def score(self) -> float | None:
        """Current quality score, or None until `window` tokens have been observed.
        Higher is worse. Roughly log(expected NLL of the text under a clean model)."""
        if not self.ready():
            return None
        f = np.asarray(self._buf, dtype=np.float64).mean(axis=0)
        z = (f - np.asarray(self.c["mean"])) / np.asarray(self.c["std"])
        return float(z @ np.asarray(self.c["weight"]) + self.c["intercept"])

    def is_degraded(self) -> bool | None:
        """True if EITHER check fires. The two are combined with OR, never averaged, because
        each is blind to what the other catches -- averaging would let one hide the other."""
        s = self.score()
        r = self.repetition()
        if s is None and r is None:
            return None
        if r is not None and r >= self.c["repetition_threshold"]:
            return True
        return None if s is None else s > self.c["threshold"]

    def reason(self) -> str | None:
        """Which check fired: 'looping', 'incoherent', or None."""
        r = self.repetition()
        if r is not None and r >= self.c["repetition_threshold"]:
            return "looping"
        s = self.score()
        if s is not None and s > self.c["threshold"]:
            return "incoherent"
        return None

    def reset(self) -> None:
        self._buf.clear()
        self._toks.clear()
        self.n_observed = 0
        self.n_skipped = 0
        self._seen = 0

    # ---------------------------------------------------------------- calibration
    @staticmethod
    def calibrate(features, targets, repetitions=None,
                  rep_floor: float = 0.15, rep_ceiling: float = 0.60) -> dict:
        """Re-fit for a different model.

        `features` is [n, 3] of (entropy, top1, margin) averaged over each generation's windows;
        `targets` is [n] of a reference quality measure (we used the NLL of the generated text
        under the untouched model). Returns a calibration dict for the constructor.

        `repetitions` is [n] of the window-level repetition rate on HEALTHY generations from the
        same model. Pass it. Without it this returns no `repetition_threshold`, and the caller
        keeps the shipped one -- which was fitted on an instruct model and fires on 21.6% of
        healthy windows of a base model. A recalibration that silently carries that constant
        forward is worse than no recalibration, because it looks complete.

        The shipped calibration was fitted on ONE model. It is a starting point, not a universal
        constant, and this is how you replace it.
        """
        X = np.asarray(features, dtype=np.float64)
        y = np.log(np.asarray(targets, dtype=np.float64) + 1e-3)
        mu, sd = X.mean(0), X.std(0) + 1e-9
        A = np.hstack([(X - mu) / sd, np.ones((len(X), 1))])
        w, *_ = np.linalg.lstsq(A, y, rcond=None)
        pred = A @ w
        out = {"mean": mu.tolist(), "std": sd.tolist(), "weight": w[:-1].tolist(),
               "intercept": float(w[-1]),
               "threshold": float(np.quantile(pred, 0.90))}
        if repetitions is not None:
            r = np.asarray(repetitions, dtype=np.float64)
            r = r[np.isfinite(r)]
            if r.size:
                # the 99th percentile of HEALTHY repetition, i.e. a 1% false-alarm rate by
                # construction, bounded below so a near-constant model cannot collapse it and
                # above by the rate that is broken in any register.
                out["repetition_threshold"] = float(
                    min(rep_ceiling, max(rep_floor, np.quantile(r, 0.99))))
        return out
