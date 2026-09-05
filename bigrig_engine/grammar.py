"""Structured output: constrain the sampler so what comes back parses.

WHAT THIS IS FOR. A coding agent that asks for `response_format: {"type": "json_object"}` needs
the reply to be JSON it can hand to a parser -- not prose about JSON, not JSON wrapped in a
code fence, not JSON with a trailing sentence. Without this the model is asked politely and
usually complies; "usually" is a parse error the user sees once a day and stops trusting.

HOW IT WORKS, AND WHY IT COSTS NOTHING PER TOKEN THE MODEL DID NOT ALREADY PAY FOR.
    mlx_lm's generation step accepts `logits_processors`: callables that see the tokens so far
    and the raw logits, and return logits. `JSONProcessor` keeps a small character automaton
    that accepts every PREFIX of a valid JSON document. At each step it takes the model's top
    candidates, decodes each, feeds the characters through a copy of the automaton, and sets
    the logit of every candidate the automaton rejects to -inf. The model's own preference
    order among the legal tokens is untouched, so this changes WHICH token is chosen only when
    the model's first choice would have broken the document -- which is exactly the case the
    user asked to prevent. When the document is complete the only legal token is EOS.

    Checking the whole vocabulary every step would cost 150,000 decodes a token. The top 384
    candidates are checked instead; the legal continuation of a JSON document is essentially
    never outside a model's top few hundred choices, and if it ever is, the slow path checks
    everything rather than emitting nothing.

WHAT IT GUARANTEES, AND WHAT IT DOES NOT.
    The output is one complete JSON value -- an object by default -- and nothing else. With a
    `json_schema`, the top-level value is an object and every `required` property name appears
    as a key; property TYPES are not enforced at the token level here. The schema itself is
    placed in front of the model so it knows what is wanted, and the required-key check is the
    part a parser cannot recover from. Full type enforcement is a grammar compiler and is not
    what this is.

WHY THINKING IS TURNED OFF UNDER A RESPONSE FORMAT.
    A model whose template opens `<think>` in the prompt writes reasoning first and the answer
    after `</think>`. A constraint that starts at the first token cannot let it write either
    tag, and constraining only after `</think>` means trusting the model to close a tag it may
    spend the whole budget inside. OpenAI's json_object mode has the same semantics -- the reply
    IS the JSON -- so a response format asks for a direct answer, and the request is made with
    thinking off.
"""
from __future__ import annotations

import json

import mlx.core as mx
import numpy as np

TOP_CANDIDATES = 384
# WHEN THE MODEL CANNOT SATISFY A SCHEMA, THE REPLY STILL HAS TO PARSE. Holding the closing
# brace back until every required key appears is right while the model is making progress. A
# small model that has recited the schema and does not know how to add a top-level key it has
# never heard of will instead emit whitespace, token after token, until max_tokens -- and the
# client gets an UNCLOSED document, which is the one outcome worse than a missing key: a missing
# key can be checked for; broken JSON cannot be repaired. So after this many consecutive
# whitespace-only tokens at a point where the object could otherwise close, the required-key
# gate is released and the object is allowed to close. The schema was also placed in front of
# the model in words; this is the floor under what happens when it did not help.
STALL_TOKENS = 8
_WS = " \t\n\r"
_LITERALS = ("true", "false", "null")


class JSONPrefix:
    """Accepts any prefix of a valid JSON value, one character at a time.

    `feed(ch)` returns False and leaves the automaton unchanged when `ch` cannot continue any
    valid document from here. `done` is True exactly when the text so far is a complete value.
    A small pushdown machine: the stack holds the open containers, `state` says what the next
    character may be, and strings, numbers and literals are tracked character by character so
    that "tru" is a valid prefix and "trx" is not.
    """

    __slots__ = ("stack", "state", "lit", "num", "esc", "hex_left", "done", "count",
                 "first_container", "keys", "cur_key", "in_key")

    # states
    VALUE = 0          # a value may start here
    IN_STRING = 1
    IN_NUMBER = 2
    IN_LITERAL = 3
    AFTER_VALUE = 4    # , or closing bracket, or end at depth 0
    KEY_OR_CLOSE = 5   # inside {} right after { : a key string or }
    KEY = 6            # inside {} after , : a key string
    COLON = 7          # after a key
    COMPLETE = 8
    ARRAY_START = 9    # inside [] right after [ : a value or ]

    def __init__(self, require_object: bool = False):
        self.stack: list = []
        self.state = self.VALUE
        self.lit = ""
        self.num = ""
        self.esc = False
        self.hex_left = 0
        self.done = False
        self.count = 0
        self.first_container = "{" if require_object else ""
        self.keys: list = []            # top-level keys seen, for the required-key check
        self.cur_key = ""
        self.in_key = False

    def copy(self) -> "JSONPrefix":
        c = JSONPrefix.__new__(JSONPrefix)
        c.stack = list(self.stack)
        c.state, c.lit, c.num, c.esc = self.state, self.lit, self.num, self.esc
        c.hex_left, c.done, c.count = self.hex_left, self.done, self.count
        c.first_container = self.first_container
        c.keys, c.cur_key, c.in_key = list(self.keys), self.cur_key, self.in_key
        return c

    # -- helpers -----------------------------------------------------------------------------
    def _close_value(self) -> None:
        """A complete scalar or container just ended; decide what may follow."""
        if not self.stack:
            self.state = self.COMPLETE
            self.done = True
        else:
            self.state = self.AFTER_VALUE

    def _start_value(self, ch: str) -> bool:
        if ch in _WS:
            return True
        if ch == "{":
            if not self.stack and self.first_container and ch != self.first_container:
                return False
            self.stack.append("{")
            self.state = self.KEY_OR_CLOSE
            return True
        if ch == "[":
            if not self.stack and self.first_container and ch != self.first_container:
                return False
            self.stack.append("[")
            self.state = self.ARRAY_START
            return True
        if not self.stack and self.first_container:
            return False                     # a bare scalar when an object was required
        if ch == '"':
            self.state = self.IN_STRING
            self.esc = False
            self.in_key = False
            return True
        if ch == "-" or ch.isdigit():
            self.num = ch
            self.state = self.IN_NUMBER
            return True
        for lit in _LITERALS:
            if lit.startswith(ch):
                self.lit = ch
                self.state = self.IN_LITERAL
                return True
        return False

    def _number_ok(self, s: str) -> bool:
        """Is `s` a prefix of some valid JSON number?"""
        i, n = 0, len(s)
        if i < n and s[i] == "-":
            i += 1
        if i >= n:
            return True
        if s[i] == "0":
            i += 1
        elif s[i].isdigit():
            while i < n and s[i].isdigit():
                i += 1
        else:
            return False
        if i < n and s[i] == ".":
            i += 1
            while i < n and s[i].isdigit():
                i += 1
        if i < n and s[i] in "eE":
            i += 1
            if i < n and s[i] in "+-":
                i += 1
            while i < n and s[i].isdigit():
                i += 1
        return i == n

    def _number_complete(self, s: str) -> bool:
        """Is `s` a complete number (so a delimiter may follow)?"""
        if not s or s in ("-",) or s[-1] in ".eE+-":
            return False
        return self._number_ok(s)

    # -- the transition function ------------------------------------------------------------
    def feed(self, ch: str) -> bool:
        st = self.state
        if st == self.COMPLETE:
            return ch in _WS                 # trailing whitespace only
        if st == self.IN_STRING:
            if self.hex_left:
                if ch in "0123456789abcdefABCDEF":
                    self.hex_left -= 1
                    if self.in_key:
                        self.cur_key += ch
                    return True
                return False
            if self.esc:
                if ch in '"\\/bfnrt':
                    self.esc = False
                    if self.in_key:
                        self.cur_key += ch
                    return True
                if ch == "u":
                    self.esc = False
                    self.hex_left = 4
                    return True
                return False
            if ch == "\\":
                self.esc = True
                return True
            if ch == '"':
                if self.in_key:
                    if len(self.stack) == 1:
                        self.keys.append(self.cur_key)
                    self.in_key = False
                    self.state = self.COLON
                else:
                    self._close_value()
                return True
            if ord(ch) < 0x20:
                return False                 # control characters must be escaped
            if self.in_key:
                self.cur_key += ch
            return True
        if st == self.IN_NUMBER:
            if self._number_ok(self.num + ch):
                self.num += ch
                return True
            if not self._number_complete(self.num):
                return False
            # the number ended; `ch` must be a legal follower
            self.num = ""
            self._close_value()
            return self.feed(ch)
        if st == self.IN_LITERAL:
            cand = self.lit + ch
            for lit in _LITERALS:
                if lit.startswith(cand):
                    self.lit = cand
                    if cand == lit:
                        self.lit = ""
                        self._close_value()
                    return True
            return False
        if st == self.VALUE:
            return self._start_value(ch)
        if st == self.ARRAY_START:
            # `[]` is a complete, empty array; anything else must be a value.
            if ch in _WS:
                return True
            if ch == "]":
                self.stack.pop()
                self._close_value()
                return True
            self.state = self.VALUE
            if self._start_value(ch):
                return True
            self.state = self.ARRAY_START
            return False
        if st == self.KEY_OR_CLOSE or st == self.KEY:
            if ch in _WS:
                return True
            if ch == "}" and st == self.KEY_OR_CLOSE:
                self.stack.pop()
                self._close_value()
                return True
            if ch == '"':
                self.state = self.IN_STRING
                self.esc = False
                self.in_key = True
                self.cur_key = ""
                return True
            return False
        if st == self.COLON:
            if ch in _WS:
                return True
            if ch == ":":
                self.state = self.VALUE
                return True
            return False
        if st == self.AFTER_VALUE:
            if ch in _WS:
                return True
            top = self.stack[-1]
            if ch == ",":
                self.state = self.KEY if top == "{" else self.VALUE
                return True
            if (ch == "}" and top == "{") or (ch == "]" and top == "["):
                self.stack.pop()
                self._close_value()
                return True
            return False
        return False

    def feed_text(self, text: str) -> bool:
        """Feed a whole string; on rejection the automaton is left as it was before the call."""
        saved = self.copy()
        for ch in text:
            if not self.feed(ch):
                for k in self.__slots__:          # roll back every field
                    setattr(self, k, getattr(saved, k))
                return False
        return True


def required_keys(schema: dict | None) -> list:
    """The top-level `required` names of a JSON schema, or [] when it has none."""
    if not isinstance(schema, dict):
        return []
    inner = schema.get("schema") if isinstance(schema.get("schema"), dict) else schema
    req = inner.get("required") if isinstance(inner, dict) else None
    return [str(k) for k in req] if isinstance(req, list) else []


class JSONProcessor:
    """A logits processor that keeps generation inside valid JSON.

    Built once per request. Holds the automaton, the decoded text of every token it has
    accepted, and the tokenizer needed to decode candidates. `__call__` matches mlx_lm's
    logits-processor signature: (tokens_so_far, logits) -> logits.
    """

    def __init__(self, tokenizer, schema: dict | None = None, top: int = TOP_CANDIDATES):
        self.tok = tokenizer
        self.required = required_keys(schema)
        # Both json_object and json_schema mean "an object", as they do in OpenAI's API.
        self.auto = JSONPrefix(require_object=True)
        self.top = int(top)
        self.eos = set(int(t) for t in getattr(tokenizer, "eos_token_ids", []) or [])
        self.seen = -1                      # -1 until the first call anchors past the prompt
        self.text = ""
        self.forced_eos = False
        self.stalled = 0                    # consecutive whitespace-only tokens while gated
        self.relaxed = False                # the required-key gate was released (see STALL_TOKENS)
        self._cache: dict = {}

    def _decode(self, tid: int) -> str:
        s = self._cache.get(tid)
        if s is None:
            try:
                s = self.tok.decode([tid])
            except Exception:               # noqa: BLE001 -- an undecodable id is simply illegal
                s = "\x00"
            self._cache[tid] = s
        return s

    def _fold(self, tokens) -> None:
        """Advance the automaton over tokens generated since the last call."""
        try:
            n = int(tokens.shape[0]) if hasattr(tokens, "shape") else len(tokens)
        except Exception:                   # noqa: BLE001
            n = 0
        if self.seen < 0:
            # mlx_lm hands the processor the PROMPT as well as the generated tokens. The first
            # call arrives before anything has been generated, so whatever is here is prompt,
            # and prose fed to a JSON automaton would look like a broken document and force EOS
            # on the spot -- which is exactly what happened before this anchor existed.
            self.seen = n
            return
        while self.seen < n:
            tid = int(tokens[self.seen])
            self.seen += 1
            if tid in self.eos:
                continue
            s = self._decode(tid)
            if not self.auto.feed_text(s):
                # The model produced something the mask should have prevented (the slow path
                # found nothing legal, or a multi-byte token decoded oddly). Freeze here rather
                # than let the document drift further: only EOS remains legal.
                self.forced_eos = True
            self.text += s
            # Stall detection: whitespace-only output while the only thing keeping the object
            # open is a required key the model is not producing.
            if self.required and not self.relaxed and s.strip() == "" \
                    and self.auto.state == JSONPrefix.AFTER_VALUE and len(self.auto.stack) == 1:
                self.stalled += 1
                if self.stalled >= STALL_TOKENS:
                    self.relaxed = True
            else:
                self.stalled = 0

    def _legal(self, tid: int) -> bool:
        if tid in self.eos:
            return self.auto.done and self._required_satisfied()
        if self.forced_eos:
            return False
        s = self._decode(tid)
        if not s or "\x00" in s:
            return False
        probe = self.auto.copy()
        if not probe.feed_text(s):
            return False
        # THE CLOSING BRACE IS WHERE REQUIRED KEYS ARE ENFORCED, NOT EOS. Once the top-level
        # object has closed nothing can add a key to it, so a check at EOS time is a check made
        # after the only moment it could have mattered -- the model recited a schema, closed the
        # object, and the processor could then permit nothing but whitespace until it gave up.
        # Refusing the `}` itself keeps the object open, and `,` plus a key string stay legal.
        if probe.done and not self.relaxed and not all(k in probe.keys for k in self.required):
            return False
        return True

    def _required_satisfied(self) -> bool:
        return self.relaxed or all(k in self.auto.keys for k in self.required)

    def __call__(self, tokens, logits):
        self._fold(tokens)
        row = logits[-1] if logits.ndim > 1 else logits
        vocab = int(row.shape[-1])
        if (self.auto.done and self._required_satisfied()) or self.forced_eos:
            # Complete: EOS is the only legal continuation.
            legal = sorted(self.eos) if self.eos else []
        else:
            k = min(self.top, vocab)
            cand = [int(i) for i in mx.argpartition(-row, k - 1)[:k].tolist()]
            legal = [t for t in cand if self._legal(t)]
            if not legal:
                # Slow path: nothing in the top candidates continues the document. Check all.
                legal = [t for t in range(vocab) if self._legal(t)]
            if not legal:
                self.forced_eos = True
                legal = sorted(self.eos) if self.eos else []
        if not legal:
            return logits                   # no EOS id at all: nothing sensible to force
        m = np.full(vocab, -np.inf, dtype=np.float32)
        m[legal] = 0.0
        out = row + mx.array(m)
        return out[None] if logits.ndim > 1 else out


def parse_response_format(rf) -> tuple:
    """Validate an OpenAI-style `response_format`. Returns (kind, schema_or_None) or raises
    ValueError with a sentence a client can act on."""
    if rf is None:
        return None, None
    if not isinstance(rf, dict) or "type" not in rf:
        raise ValueError('`response_format` must be an object with a `type`')
    kind = rf.get("type")
    if kind == "text":
        return None, None
    if kind == "json_object":
        return "json_object", None
    if kind == "json_schema":
        js = rf.get("json_schema")
        if not isinstance(js, dict):
            raise ValueError('`response_format.json_schema` must be an object')
        schema = js.get("schema", js)
        if not isinstance(schema, dict):
            raise ValueError('`response_format.json_schema.schema` must be an object')
        try:
            json.dumps(schema)
        except (TypeError, ValueError):
            raise ValueError('`response_format.json_schema.schema` is not serialisable')
        return "json_schema", schema
    raise ValueError(f'`response_format.type` must be "text", "json_object" or "json_schema", '
                     f'not {kind!r}')


def schema_instruction(kind: str | None, schema: dict | None) -> str:
    """One paragraph placed in front of the model so it knows the shape that is wanted. The
    constraint enforces validity; this is what makes the content sensible."""
    if kind is None:
        return ""
    if kind == "json_object":
        return "Respond with a single JSON object and nothing else."
    req = required_keys(schema)
    body = json.dumps(schema, separators=(",", ":"))
    line = "Respond with a single JSON object and nothing else, matching this JSON schema: " + body
    if req:
        line += ". The keys " + ", ".join(f'"{k}"' for k in req) + " are required."
    return line
