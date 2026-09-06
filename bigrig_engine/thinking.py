"""Cap how long a reasoning model may think before it has to answer.

WHY. Qwen3.6, GLM and Nemotron reason before answering, and on a slow machine a model that
spends 800 tokens thinking before a one-line answer is the thing that makes it feel unusable --
the user waits a minute watching a scratchpad for a reply that needed a sentence. A thinking
budget is the standard control for this (Anthropic exposes `budget_tokens`); ours accepted the
field and ignored it.

HOW. A logits processor counts the reasoning tokens generated so far. While the model is under
budget it is left completely alone -- same tokens, same order. When it reaches the budget and is
still inside the reasoning block, the processor forces the closing tag, one token at a time, and
then steps out of the way: the model sees `</think>` in its own context and answers, exactly as
it would have if it had chosen to stop there. Nothing is truncated after the answer begins, and
a model that finishes thinking under budget never notices this exists.

WHAT COUNTS AS REASONING. Two shapes, the same two the reply splitter handles: the prompt opens
the block and the reply closes it (Qwen3.6, GLM), or the reply opens and closes it itself
(Qwen3-30B, Nemotron). In the first, every token is reasoning until the close tag; in the
second, counting starts at the opening tag. Either way the budget is spent only on reasoning --
a model that is already answering is never interrupted.
"""
from __future__ import annotations

import mlx.core as mx

THINK_OPEN, THINK_CLOSE = "<think>", "</think>"


class ThinkingBudget:
    """Force a reasoning model to stop thinking after `budget` reasoning tokens.

    Matches mlx_lm's logits-processor signature: (tokens_so_far, logits) -> logits. Holds the
    close-tag token ids for this tokenizer and, once the budget is spent and the block is still
    open, forces them out before returning control to the model.
    """

    def __init__(self, tokenizer, budget: int, starts_in_reasoning: bool):
        self.tok = tokenizer
        self.budget = int(budget)
        self.in_reasoning = bool(starts_in_reasoning)   # already thinking at token 0?
        self.seen = -1                                   # -1 until the first call anchors the prompt
        self.reasoning_tokens = 0
        self.text = ""
        self.closed = False                              # the block has ended (naturally or forced)
        self.forcing: list = []                          # close-tag ids still to emit
        self.forced = False                              # the budget, not the model, ended the block
        # The close tag, as this tokenizer's ids. A leading newline is included so the tag lands
        # on its own line the way the templates write it; if the tag is a single id (it is on
        # every model here) that is all that is forced.
        self.close_ids = self._encode(THINK_CLOSE)
        self.open_ids = self._encode(THINK_OPEN)
        self._cache: dict = {}

    def _encode(self, s: str) -> list:
        try:
            return [int(t) for t in self.tok.encode(s, add_special_tokens=False)]
        except Exception:                                # noqa: BLE001
            return []

    def _decode(self, tid: int) -> str:
        s = self._cache.get(tid)
        if s is None:
            try:
                s = self.tok.decode([tid])
            except Exception:                            # noqa: BLE001
                s = ""
            self._cache[tid] = s
        return s

    def _fold(self, tokens) -> None:
        """Advance over tokens generated since the last call, tracking the reasoning count and
        whether the block has opened or closed."""
        try:
            n = int(tokens.shape[0]) if hasattr(tokens, "shape") else len(tokens)
        except Exception:                                # noqa: BLE001
            n = 0
        if self.seen < 0:
            self.seen = n                                # the first call carries the prompt; skip it
            return
        while self.seen < n:
            tid = int(tokens[self.seen])
            self.seen += 1
            s = self._decode(tid)
            self.text += s
            if self.closed:
                continue
            if not self.in_reasoning:
                # Waiting for the block to open (reply-opens-its-own shape).
                if THINK_OPEN in self.text:
                    self.in_reasoning = True
                continue
            if THINK_CLOSE in self.text:
                self.closed = True                       # the model closed on its own
                continue
            self.reasoning_tokens += 1

    def __call__(self, tokens, logits):
        self._fold(tokens)
        row = logits[-1] if logits.ndim > 1 else logits
        vocab = int(row.shape[-1])
        # Still forcing the close tag out, one id per step.
        if self.forcing:
            tid = self.forcing.pop(0)
            if not self.forcing:
                self.closed = True                       # the last forced id lands; block is closed
            return self._only(tid, vocab, logits.ndim)
        # ONCE CLOSED, THE TAGS ARE ILLEGAL. A model whose thought was cut mid-sentence sometimes
        # emits `</think>` again as its first "answer" token -- measured once in four runs -- and
        # it would arrive in the text block as a literal tag. Masking the tag tokens after the
        # close costs two logits and makes the answer clean deterministically. The model's other
        # choices are untouched.
        if self.closed:
            tags = [t[0] for t in (self.close_ids, self.open_ids) if t]
            if not tags:
                return logits
            return self._without(tags, row, logits.ndim)
        # Under budget, or never a reasoning model: hands off entirely.
        if not self.in_reasoning or self.reasoning_tokens < self.budget or not self.close_ids:
            return logits
        # Budget spent and still thinking: begin forcing the close tag this step.
        self.forced = True
        self.forcing = list(self.close_ids)
        tid = self.forcing.pop(0)
        if not self.forcing:
            self.closed = True
        return self._only(tid, vocab, logits.ndim)

    @staticmethod
    def _without(tids: list, row, ndim: int):
        """The row with just these token ids made impossible; everything else as the model had it."""
        import numpy as np
        m = np.zeros(int(row.shape[-1]), dtype=np.float32)
        m[tids] = -np.inf
        out = row + mx.array(m)
        return out[None] if ndim > 1 else out

    @staticmethod
    def _only(tid: int, vocab: int, ndim: int):
        """Logits that permit exactly one token."""
        import numpy as np
        m = np.full(vocab, -np.inf, dtype=np.float32)
        m[tid] = 0.0
        out = mx.array(m)
        return out[None] if ndim > 1 else out


def resolve_budget(thinking_budget, reasoning_effort=None) -> int | None:
    """The reasoning-token cap to enforce, from the fields a request may carry.

    `thinking_budget` is the direct integer (also where Anthropic's budget_tokens is mapped).
    `reasoning_effort` is OpenAI's coarse dial, translated to a token cap so a client that speaks
    only that still gets a bound. None, or a non-positive number, means no cap -- the model
    thinks as long as it wants, which is the default and unchanged behaviour.
    """
    if thinking_budget is not None:
        try:
            b = int(thinking_budget)
        except (TypeError, ValueError):
            raise ValueError("`thinking_budget` must be an integer number of tokens")
        return b if b > 0 else None
    if reasoning_effort is not None:
        return {"low": 256, "medium": 1024, "high": 4096}.get(str(reasoning_effort).lower())
    return None
