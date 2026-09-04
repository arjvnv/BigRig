"""AdaptiveMeter — a quality meter that calibrates itself to whatever model it is watching.

THE PROBLEM IT SOLVES
    QualityMeter ships constants fitted on ONE model (Ling-mini-2.0-3bit). Different models have
    different personalities: a naturally hesitant model looks "degraded" to a meter calibrated on
    a confident one. Absolute thresholds cannot transfer.

WHAT IS LEARNED, AND WHY IT IS BOTH SIGNALS
    An earlier version of this file claimed repetition was "a property of the TEXT, not of the
    model", so its threshold was "absolute and transfers". THAT CLAIM WAS WRONG, and it was
    tested and falsified rather than reasoned about. Measured on healthy generations:

        Ling-mini-2.0-3bit (instruct)   window repetition  mean 0.006  p99 0.082
        OLMoE-1B-7B-0125   (base)       window repetition  mean 0.085  p99 0.410

    The RATE is a property of the string, but what rate counts as ABNORMAL is a property of the
    model and its register. A base model with no chat template is simply more repetitive, and
    the threshold fitted on the instruct model fired on 21.6% of the base model's healthy
    windows -- one false alarm in five, on undamaged output. Learning it instead brings that to
    3.4% and leaves the instruct model bit-for-bit unchanged (0.41% before, 0.41% after).

    confidence  is a property of the MODEL. Entropy of 2.0 might be normal for one model and
                alarming for another. Scored as a DEVIATION from this model's own normal.
    repetition  is scored against a LEARNED rate for this model too, with an absolute ceiling
                underneath it that no healthy window of either model came close to.

HOW THE BASELINE IS LEARNED
    A running mean and variance of the confidence features, updated ONLY from windows currently
    judged healthy. Most output is healthy, so the estimate converges on "normal for this model"
    without any labels. Updating only on healthy windows stops a long bad patch from redefining
    normal -- the failure mode that would make an anomaly detector go quiet exactly when it
    matters most.

WHAT IT STILL CANNOT DO
    Everything QualityMeter cannot do: it reports THAT quality is poor, not WHY, and it sees only
    surface properties of the token stream. A confidently-stated false fact looks perfectly
    healthy to it. See README.md.
"""
from __future__ import annotations
from collections import deque

import numpy as np

from .meter import CALIBRATION


class AdaptiveMeter:
    # Most tokens of a prompt whose n-grams are remembered. A long document would otherwise make
    # this set the largest object in the process, for a structure only ever asked one question.
    PROMPT_GRAM_LIMIT = 8192

    def __init__(self, window: int = 64, warmup: int = 3, z_threshold: float = 2.5,
                 repetition_threshold: float | None = None, repetition_n: int = 4,
                 ema: float = 0.02, strict: bool = True, min_window: int = 16,
                 rep_ceiling: float = 0.60, rep_floor: float = 0.15,
                 rep_history: int = 256, rep_quantile: float = 0.99,
                 rep_min_history: int = 16, rep_margin: float = 0.10,
                 z_quantile: float = 0.90, z_history: int = 256,
                 z_min_history: int = 16, z_ceiling: float = 8.0,
                 min_reply_tokens: int = 48):
        """
        warmup       windows to observe before flagging anything. Below this the meter has no
                     idea what normal looks like and reports None rather than guessing.
        z_threshold  how many standard deviations from this model's normal counts as degraded.
        ema          how fast the baseline adapts. Deliberately slow (0.02) so a genuine
                     degradation cannot be absorbed into "normal" within one generation.
        """
        self.window, self.warmup = int(window), int(warmup)
        # The fewest tokens THIS REPLY must have produced before the meter will offer a verdict.
        # Below it `is_degraded` returns None -- "not enough output to say". A short answer used
        # to be judged against a baseline built from other prompts and a repetition window that
        # still contained the previous reply; see is_degraded.
        self.min_reply_tokens = int(min_reply_tokens)
        # n-grams the prompt already contains, so requested repetition is not read as damage.
        self._prompt_grams: set = set()
        # Free-energy baseline, learned per model. See observe_energy for why entropy alone
        # misses the light damage this engine causes on purpose.
        self._energy_n = 0
        self._energy_mean = 0.0
        self._energy_var = 0.0
        self._energy_last_z = 0.0
        self._energy_seen = 0
        self.z_threshold, self.ema = float(z_threshold), float(ema)
        # An explicit repetition_threshold PINS the rate and switches learning off. That is the
        # right call only when the model's normal rate is already known and measured -- passing
        # a guess here reinstates exactly the failure the learning exists to avoid.
        self.rep_override = None if repetition_threshold is None else float(repetition_threshold)
        self.rep_n = int(repetition_n)
        # REPETITION IS NOT MODEL-AGNOSTIC, despite being a property of the string. Measured on
        # healthy generations (window-level, the granularity the live meter sees):
        #     Ling-mini-2.0-3bit (instruct)  mean 0.006  p99 0.082  -> 0.15 fires on  0.4%
        #     OLMoE-1B-7B-0125   (base)      mean 0.085  p99 0.410  -> 0.15 fires on 21.6%
        # A base model with no chat template is simply more repetitive, so a threshold fitted on
        # an instruct model false-alarms on one healthy window in five. The rate is therefore
        # LEARNED like everything else, with two guards:
        #   rep_ceiling  an absolute backstop. No healthy window of EITHER model reached 0.60,
        #                so text this repetitive is broken in any register and needs no baseline.
        #   rep_floor    a floor under the learned threshold, so a model whose repetition is
        #                near-constant cannot drive its own threshold down onto noise.
        self.rep_ceiling = float(rep_ceiling)
        self.rep_floor = float(rep_floor)
        # LEARNING LOCK. The baseline may only fold in windows it judges healthy -- otherwise a
        # sustained loop teaches the meter that looping is normal. But with no baseline the
        # threshold starts at `rep_floor`, so on a model whose NORMAL rate sits above that floor
        # every ordinary window is judged to be looping, excluded, and the baseline can never
        # learn the truth. Measured on OLMoE: the threshold stayed pinned at 0.15 forever. So
        # until there is enough history the ONLY exclusion is the absolute ceiling, which is
        # model-independent; the tighter learned gate engages afterwards. The worst case is
        # bounded either way, because the learned threshold can never exceed the ceiling.
        #
        # A QUANTILE, not mean + k*sd: repetition is heavily right-skewed (OLMoE healthy windows
        # have median 0.033 but p99 0.410), and a symmetric spread estimate on that shape sits
        # far below the real tail and false-alarms on it.
        self.rep_history = int(rep_history)
        self.rep_quantile = float(rep_quantile)
        # windows of healthy text before the learned threshold replaces the backstop. 16 windows
        # is ~1k tokens; its p99 is essentially the observed max, which is noisy but errs high,
        # i.e. toward silence rather than toward false alarms.
        self.rep_min_history = int(rep_min_history)
        # Headroom above the learned normal. Without it a model whose repetition rate is very
        # steady produces a quantile equal to that rate, and the meter flags its own ordinary
        # output as looping the moment it reproduces it exactly.
        self.rep_margin = float(rep_margin)
        # THE Z THRESHOLD IS LEARNED TOO, for the same reason the repetition rate is.
        # `z_threshold` assumes the internal standard deviation is a true one. It is not: the
        # baseline may only fold in windows it judges healthy, which truncates exactly the
        # high-deviation tail the spread is supposed to measure. Measured on Ling, the internal
        # sd came out 2.9x too small even after the folding bug above was fixed, so a nominal
        # 2.5 sigma fired at 0.9 real sigma and flagged 44% of HEALTHY windows.
        #
        # So the SCALE is left as it is and the BAR is learned: the quantile of z actually
        # observed on this model. That is self-correcting -- whatever bias the sd carries, the
        # quantile of the resulting z still lands at the intended false-alarm rate. The history
        # is gated on an absolute ceiling ONLY, never on the meter's own verdict, because gating
        # it on the verdict is the circularity that caused this in the first place.
        # THE DESIGN POINT. z_quantile IS the false-alarm rate: the bar sits at that quantile of
        # observed z, so (1 - z_quantile) of healthy windows clear it by construction. Measured
        # on real healthy Ling output, after the bar has converged:
        #
        #     z_quantile   design FP   measured FP   catches damage
        #        0.90         10%        10.6%           64%
        #        0.95          5%         8.1%           47%
        #        0.98          2%         4.2%           35%
        #        0.99          1%         1.8%           30%
        #
        # 0.90 is right for AutoTuner, where a false alarm costs a little speed and the loop is
        # asymmetric to absorb it. It is WRONG for anything a user reads: at 0.90 one healthy
        # window in ten raises a warning. Use 0.98-0.99 for a user-facing signal.
        #
        # A NOTE ON WHAT THIS IS NOT. An earlier session recorded "after a long coding session the
        # meter gets twitchy on prose, 12.7%" and filed it as register switching. It is not. The
        # control -- the same session with the code phase replaced by MORE PROSE -- gives 10.0%,
        # statistically the same. The rise is this convergence, and it happens regardless of
        # register. A baseline-per-register bank was built to fix it and reverted, because it
        # fixed nothing that was broken.
        self.z_quantile, self.z_ceiling = float(z_quantile), float(z_ceiling)
        self.z_history, self.z_min_history = int(z_history), int(z_min_history)
        self._zbuf: deque = deque(maxlen=self.z_history)
        self._rbuf: deque = deque(maxlen=self.rep_history)
        # Repetition is measured over the whole token buffer, so consecutive per-token readings
        # differ by one token in sixty-four and are almost the same number. Folding every one of
        # them would make a 512-entry history about eight INDEPENDENT windows wearing a disguise,
        # and its quantile would track the last window rather than the model. One sample per
        # window is taken instead, so the history means what its length says.
        self._since_fold = 0
        self.strict = bool(strict)
        # COLD START. A full window costs 64 tokens of blindness -- about two sentences, which is
        # long enough for a user to see broken output before any warning. From `min_window`
        # tokens the meter reports a PROVISIONAL reading instead, with the threshold widened by
        # sqrt(window/n): the mean of n samples has standard error proportional to 1/sqrt(n), so
        # a partial window is genuinely noisier and must clear a correspondingly higher bar.
        self.min_window = max(4, int(min_window))
        # A buffer of maxlen `window` can never hold `min_window` entries if min_window is the
        # larger of the two, so ready() would be False forever and the meter would go SILENTLY
        # blind -- no reading, no error, no way to notice. Refuse the configuration instead.
        if self.min_window > self.window:
            raise ValueError(
                f"min_window ({self.min_window}) exceeds window ({self.window}); the meter could "
                f"never become ready. Lower min_window or raise window.")
        self._buf: deque = deque(maxlen=self.window)
        self._toks: deque = deque(maxlen=self.window)
        self._mu = None          # running baseline mean of (entropy, top1, margin)
        self._var = None
        self.n_windows = 0       # completed windows folded into the baseline
        self.n_observed = 0
        self.n_tokens = 0
        self.n_skipped = 0

    # ------------------------------------------------------------------ input
    def observe(self, probs=None, *, logits=None) -> None:
        if probs is None and logits is None:
            raise ValueError("observe() needs probs= or logits=")
        raw = np.asarray(logits if probs is None else probs, dtype=np.float64).ravel()
        bad = None
        if raw.size < 2:
            bad = f"need at least 2 entries, got {raw.size}"
        elif not np.all(np.isfinite(raw)):
            bad = f"{int((~np.isfinite(raw)).sum())} non-finite value(s)"
        elif probs is not None and (np.any(raw < 0) or raw.sum() <= 0):
            bad = "probabilities are negative or sum to zero"
        if bad:
            self.n_skipped += 1
            if self.strict:
                raise ValueError(f"AdaptiveMeter.observe(): {bad}")
            return
        if probs is None:
            a = raw - raw.max(); p = np.exp(a); p /= p.sum()
        else:
            p = raw; t = p.sum()
            if not (0.98 < t < 1.02): p = p / t
        k = np.argpartition(p, -2)[-2:]
        t1, t2 = np.sort(p[k])[::-1]
        nz = p[p > 0]
        self._buf.append((float(-(nz * np.log(nz)).sum()), float(t1), float(t1 - t2)))
        self.n_observed += 1
        if len(self._buf) == self.window:
            self._maybe_update()          # baseline only ever learns from FULL windows

    def observe_energy(self, energy: float) -> None:
        """Feed the free energy of this step: -logsumexp over the raw logits.

        WHY THIS SIGNAL AND NOT ENTROPY ALONE
            Entropy is computed after the softmax, which normalises the logits and throws away
            their overall scale -- and the scale is where light damage shows. Measured on OLMoE
            with N of 16 layers deliberately scrambled, AUROC against healthy generations:

                                    2 of 16    4 of 16    8 of 16
                entropy               0.625      0.972      0.917
                energy                1.000      1.000      1.000
                logit gap             0.188      0.319      0.757
                one minus top prob    0.215      0.583      0.889

            Entropy is near chance on light damage. Energy separates every level of THAT damage
            perfectly.

        AND IT DOES NOT TRANSFER TO QUANTISATION, WHICH IS THE DAMAGE THIS PRODUCT ACTUALLY
        CAUSES. MEASURED 2026-09-01, AND THE RESULT IS BAD.
            The ladder above is SCRAMBLED LAYERS -- structural damage, catastrophic, and nothing
            the engine does on purpose. What ships is quantisation. Re-run on OLMoE-1B-7B-0125,
            everything resident so streaming plays no part, twelve prompts per condition, experts
            round-tripped through a lower precision so they carry exactly its error:

                comparison             by energy   by repetition   as the meter ships
                healthy vs 3-bit           0.458           0.625                0.500
                healthy vs 2-bit           0.312           0.708                0.542

            As shipped it is a coin flip. Worse, the energy signal is INVERTED: quantised output
            has a LOWER peak deviation than healthy output, so the feature this meter is built
            around points the wrong way on the one kind of damage it exists to report. That is
            not a threshold that needs moving -- flipping it would flag confident healthy output
            instead. Repetition is weakly informative (0.625, 0.708) and is doing what little
            work is being done.

            The mechanism is not mysterious. A scrambled layer produces incoherent logits, which
            is a large energy excursion. Quantisation produces SMOOTHER, more confident logits --
            it rounds the model toward its own strongest opinions. Those look healthier than
            healthy by this measure.

            So: this meter detects loops and incoherence, and reports honestly on catastrophic
            failure. It does NOT measure the cost of compression, and nothing built on it may
            claim that it does until a signal is found that separates the ladder above.

            Scored as a deviation from THIS model's own baseline, for the same reason confidence
            is: the absolute value depends on vocabulary size and logit scale and does not
            transfer between models.
        """
        if energy is None or energy != energy:          # None or NaN
            return
        self._energy_seen += 1
        if self._energy_n < 2 or not self._ready_energy():
            self._energy_update(energy)
            self._energy_last_z = 0.0
            return
        mean, var = self._energy_mean, max(self._energy_var, 1e-9)
        z = (energy - mean) / (var ** 0.5)
        self._energy_last_z = z
        if z < self.z_threshold:                        # healthy: fold into the baseline
            self._energy_update(energy)

    def _ready_energy(self) -> bool:
        return self._energy_n >= max(8, self.warmup)

    def _energy_update(self, x: float) -> None:
        self._energy_n += 1
        d = x - self._energy_mean
        self._energy_mean += d / self._energy_n
        self._energy_var += (d * (x - self._energy_mean) - self._energy_var) / self._energy_n

    @property
    def energy_z(self) -> float:
        """How far this model's free energy has drifted from its own normal, in sigmas."""
        return self._energy_last_z

    def observe_stats(self, entropy: float, top1: float, margin: float) -> None:
        """FAST PATH, identical in contract to `Meter.observe_stats`.

        `observe(probs)` must move the whole distribution from device to host; three scalars is
        12 bytes against ~628 KB. An engine that computes the statistics on-device passes them
        here instead. This exists so `AdaptiveMeter` is a drop-in for `Meter`: code written
        against one must not break when swapped to the other.
        """
        vals = (float(entropy), float(top1), float(margin))
        if not all(v == v and abs(v) != float("inf") for v in vals):
            self.n_skipped += 1
            if self.strict:
                raise ValueError(
                    "AdaptiveMeter.observe_stats(): non-finite value in (entropy, top1, margin)")
            return
        self._buf.append(vals)
        self.n_observed += 1
        if len(self._buf) == self.window:
            self._maybe_update()          # baseline only ever learns from FULL windows

    def observe_token(self, token: int) -> None:
        """One generated token. Counted separately from `n_observed`, and that is not pedantry.

        `n_observed` counts CONFIDENCE observations, and there are two paths into the meter: the
        cheap energy path, which is the default, and the log-probability path used when the raw
        logits could not be captured. The energy path never touches `n_observed`. A reply-length
        gate written against it therefore read zero for every token of every normal generation --
        which silenced the meter completely, including on output that was visibly a hard loop.
        Measured while it was wrong: a 2-bit model emitting "A person is a person." forever, with
        repetition at 0.869 against a threshold of 0.600, reported 0 of 132 tokens flagged.
        """
        self._toks.append(int(token))
        self.n_tokens += 1

    # ------------------------------------------------------------------ baseline
    def _features(self):
        return np.asarray(self._buf, dtype=np.float64).mean(axis=0) if self._buf else None

    def _maybe_update(self):
        """Fold ONE independent window into the baseline.

        Called on every observation, but acts only once per `window` of them. Consecutive
        readings differ by a single token in `window`, so folding all of them makes the running
        mean chase itself: the deviations it squares are the one-token jitter BETWEEN adjacent
        overlapping windows, not the real spread between independent ones. Measured consequence
        of getting this wrong -- the internal standard deviation came out 2.8x to 5.2x too small,
        so a nominal 2.5-sigma threshold fired at 0.62 real sigma and flagged 40.7% of HEALTHY
        windows as degraded.
        """
        f = self._features()
        if self._mu is None:
            self._mu, self._var = f.copy(), np.ones_like(f) * 1e-6
            self.n_windows = 1
            self._since_fold = 0
            return
        self._since_fold += 1
        if self._since_fold < self.window:
            return
        self._since_fold = 0
        # z history FIRST, and deliberately NOT gated on the healthy verdict -- see __init__.
        zc = self._raw_z()
        if zc is not None and np.isfinite(zc) and zc < self.z_ceiling:
            self._zbuf.append(float(zc))
        # ONLY fold healthy windows into the baseline, or a long bad patch becomes "normal"
        if self.n_windows < self.warmup or not self._confidence_degraded():
            a = self.ema if self.n_windows >= self.warmup else 1.0 / (self.n_windows + 1)
            d = f - self._mu
            self._mu = self._mu + a * d
            self._var = (1 - a) * (self._var + a * d * d)
            # the model's NORMAL repetition rate, learned from the same healthy windows. Gated
            # on _looping() as well, or a sustained loop would teach the meter that looping is
            # normal -- the same trap the confidence baseline is gated against.
            # Gated on the ABSOLUTE CEILING ONLY, never on the meter's own looping verdict.
            # Gating on the verdict is a permanent self-reinforcing lock: every window above the
            # current threshold is excluded, so the history can never learn that this register
            # sits higher. Measured -- a coding session stayed pinned at 0.150 for as long as it
            # ran, false-alarming on 19.8% of one prompt's windows every single time, because
            # code repeats structurally (`self.`, `return`, indentation).
            #
            # The trade this makes, stated plainly: a SUSTAINED soft loop in the 0.2-0.5 band can
            # raise the threshold toward the ceiling over the length of the history. Degenerate
            # looping is not affected -- it runs at 0.8-1.0, above the ceiling, and is excluded
            # from the history and flagged regardless of what has been learned.
            # Gated at a MULTIPLE of the current threshold, never at the threshold itself.
            # Gating at the threshold is a permanent self-reinforcing lock: every window above it
            # is excluded, so the history can never learn that this register sits higher.
            # Measured -- a coding session stayed pinned at 0.150 for as long as it ran,
            # false-alarming on 19.8% of one prompt's windows every time, because code repeats
            # structurally (`self.`, `return`, indentation).
            #
            # Gating at the CEILING alone is the opposite error: genuine loops enter the history
            # and push the bar up, which cost 5 points of detection on the model where damage is
            # real. The band between 1x and 2x is where "this register is simply more repetitive"
            # lives; above 2x is not a register, it is a loop.
            r = self.repetition()
            if r is not None and r < min(self.rep_ceiling, 2.0 * self.rep_threshold()):
                self._rbuf.append(float(r))
            self.n_windows += 1

    def _sd(self):
        """Standard deviation with a FLOOR.

        Without one, a very consistent model drives the running variance toward zero and any
        deviation produces an enormous z -- observed z=+10952 on constant input. The floor is
        relative to the mean (a 2% coefficient of variation) so it scales across models rather
        than assuming an absolute unit.
        """
        return np.maximum(np.sqrt(np.maximum(self._var, 0.0)),
                          np.maximum(0.02 * np.abs(self._mu), 1e-3))

    def _confidence_degraded(self) -> bool:
        if self._mu is None or self.n_windows < self.warmup:
            return False
        f = self._features()
        if f is None:
            return False
        # entropy UP, top1 DOWN, margin DOWN all mean less certain
        z = self._raw_z()
        return bool(z is not None and z > self._threshold())

    # ------------------------------------------------------------------ output
    def ready(self) -> bool:
        return len(self._buf) >= self.min_window and self.n_windows >= self.warmup

    def provisional(self) -> bool:
        """True when the reading comes from a partial window and is therefore less reliable."""
        return self.ready() and len(self._buf) < self.window

    def _threshold(self) -> float:
        """The learned bar, widened for a partial window: the mean of n samples has standard
        error proportional to 1/sqrt(n), so a provisional reading is genuinely noisier and must
        clear a correspondingly higher bar."""
        n = len(self._buf)
        bar = self.z_bar()
        if n >= self.window:
            return bar
        return bar * float(np.sqrt(self.window / max(n, 1)))

    def set_prompt(self, tokens) -> None:
        """Tell the meter what was ASKED, so it can tell requested repetition from the other kind.

        THE FALSE ALARM THIS REMOVES
            The repetition signal counts n-grams in the window that have appeared before. That is
            the right test for a model stuck in a loop and the wrong one for a model doing as it
            was told. Asked to reproduce a passage, quote a document, fill in a table or continue
            a list, a HEALTHY model repeats -- and the meter called it damage. Observed on this
            engine at 7% of all tokens flagged, every one of them from prompts that asked for
            repetition or from replies too short to judge.

            An n-gram already present in the prompt is not evidence of anything. It is the answer
            to the question. Only n-grams the model produced twice ON ITS OWN still count.

            Stored as hashes rather than tuples: a 40,000-token prompt has 40,000 n-grams, and
            keeping them as tuples costs several megabytes per request for a set that is only
            ever asked "is this in you".
        """
        self._prompt_grams = set()
        if not tokens:
            return
        t = [int(x) for x in tokens]
        n = self.rep_n
        if len(t) < n:
            return
        # Bounded: an enormous prompt should not turn the meter into the biggest object in the
        # process. The most recent tokens are kept, being the likeliest to be echoed.
        if len(t) > self.PROMPT_GRAM_LIMIT:
            t = t[-self.PROMPT_GRAM_LIMIT:]
        self._prompt_grams = {hash(tuple(t[i:i + n])) for i in range(len(t) - n + 1)}

    def repetition(self) -> float | None:
        """How much of this window the model repeated of its OWN accord. See set_prompt."""
        n = self.rep_n
        if len(self._toks) < 2 * n: return None
        t = list(self._toks)
        g = [tuple(t[i:i + n]) for i in range(len(t) - n + 1)]
        pg = self._prompt_grams
        seen, rep, counted = set(), 0, 0
        for x in g:
            if pg and hash(x) in pg:
                # Asked for. Excluded from the numerator AND the denominator: leaving it in the
                # denominator would let a long quoted passage dilute the rate toward zero and
                # hide a genuine loop running alongside it.
                continue
            counted += 1
            if x in seen: rep += 1
            else: seen.add(x)
        if counted < 2 * n:
            # Almost everything in this window came from the prompt. There is not enough of the
            # model's own output left to judge, and a rate over three or four n-grams is noise.
            return None
        return rep / counted

    def rep_threshold(self) -> float:
        """The repetition rate that counts as looping FOR THIS MODEL.

        Before enough history exists this is the absolute backstop only -- deliberately
        insensitive rather than wrong, since a fixed guess is exactly what false-alarmed on one
        healthy window in five. After that it is the `rep_quantile` of recent healthy windows,
        i.e. a false-alarm rate of (1 - rep_quantile) by construction, clamped into
        [rep_floor, rep_ceiling].
        """
        if self.rep_override is not None:
            return self.rep_override
        if len(self._rbuf) < self.rep_min_history:
            return self.rep_ceiling
        q = float(np.quantile(np.asarray(self._rbuf, dtype=np.float64), self.rep_quantile))
        q *= (1.0 + self.rep_margin)
        return float(min(self.rep_ceiling, max(self.rep_floor, q)))

    def _looping(self) -> bool:
        r = self.repetition()
        if r is None:
            return False
        return r >= self.rep_ceiling or r >= self.rep_threshold()

    def _raw_z(self):
        if self._mu is None or not self._buf: return None
        z = (self._features() - self._mu) / self._sd()
        return float(max(z[0], -z[1], -z[2]))

    def z_score(self) -> float | None:
        """How far outside this model's normal the current window sits."""
        if not self.ready(): return None
        return self._raw_z()

    def z_bar(self) -> float:
        """The z that counts as degraded FOR THIS MODEL. Until there is history this is the
        absolute ceiling -- insensitive rather than wrong, since a fixed guess is what flagged
        44% of healthy output. After that it is the `z_quantile` of z actually observed."""
        if len(self._zbuf) < self.z_min_history:
            return self.z_ceiling
        q = float(np.quantile(np.asarray(self._zbuf, dtype=np.float64), self.z_quantile))
        return float(min(self.z_ceiling, max(self.z_threshold, q)))

    def score(self) -> float | None:
        """Alias of `z_score`, present so code written against `QualityMeter` keeps working.

        ORIENTATION matches QualityMeter: higher is worse. The SCALE does not. QualityMeter
        returns roughly log(expected NLL) and is thresholded at a fitted constant; this returns
        standard deviations from THIS model's learned normal. Do not carry a threshold from one
        to the other -- call `is_degraded()`, which applies each meter's own bar.
        """
        return self.z_score()

    def should_observe(self) -> bool:
        """Always True. QualityMeter can subsample with `stride`; this meter does not, because
        its baseline is estimated from the observations themselves and thinning them would
        widen that estimate rather than only costing resolution."""
        return True

    def _energy_degraded(self) -> bool:
        return self._ready_energy() and self._energy_last_z >= self.z_threshold

    def is_degraded(self) -> bool | None:
        """None means "not enough output to say", which is not the same as "fine".

        THE ORDER USED TO BE WRONG AND IT MATTERED. `_looping()` and `_energy_degraded()` were
        both tested BEFORE the readiness gate, so a sixteen-token reply could be flagged on a
        baseline borrowed from other prompts and a repetition window that still held the tail of
        the PREVIOUS reply. Measured on this engine: four consecutive sixteen-token replies to
        the same question flagged 1, 6, 15 and 15 tokens -- the model was not looping, the meter
        was reading across the boundary between two answers.

        Judging now needs two things: enough tokens IN THIS REPLY to have an opinion, and enough
        history to know what normal looks like.
        """
        if self.n_tokens < self.min_reply_tokens:
            return None
        if self._looping():
            return True
        if self._energy_degraded():
            return True
        if not self.ready():
            return None
        return self._confidence_degraded()

    def reason(self) -> str | None:
        if self.n_tokens < self.min_reply_tokens: return None
        if self._looping(): return "looping"
        if self._energy_degraded(): return "weights drifted"
        if self.ready() and self._confidence_degraded(): return "incoherent"
        return None

    def reset(self, keep_baseline: bool = True) -> None:
        """keep_baseline=True (default) carries what it learned about the model into the next
        generation -- that is the point of self-calibration."""
        self._buf.clear(); self._toks.clear()
        self.n_observed = 0; self.n_skipped = 0; self.n_tokens = 0
        self._energy_last_z = 0.0
        if not keep_baseline:
            self._energy_n = 0; self._energy_mean = 0.0; self._energy_var = 0.0
            self._energy_seen = 0
            self._mu = self._var = None; self.n_windows = 0
            self._rbuf.clear(); self._zbuf.clear(); self._since_fold = 0
