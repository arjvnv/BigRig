"""Adversarial tests for consent.

This module exists to stop ONE thing: a user's model being altered without them knowing. Every
test below is a way that could happen, and most of them are ways it nearly did.

The bug this replaced: the engine chose compression on its own, announced it once, cached the
compressed copy, and from the second run onward served quantised weights in silence -- while the
summary line read "100% of experts kept in RAM", which was true and completely misleading.
"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bigrig_engine import consent

FAIL = []
def check(n, c, d=""):
    (print(f"  PASS  {n}") if c else (FAIL.append(n), print(f"  FAIL  {n}  {d}")))

TMP = tempfile.mkdtemp(prefix="bigrig-consent-")
BLOB = os.path.join(TMP, "m.experts")
BLOB2 = os.path.join(TMP, "other.experts")

COMPRESS = {"mode": "compress", "bits": 3, "group_size": 128, "expert_gb": 6.0,
            "original_gb": 12.7, "room_gb": 7.3, "stream_capacity": 61,
            "stream_residency": 0.48, "reason": "fits at 3 bits"}
NATIVE = {"mode": "native", "expert_gb": 3.6, "room_gb": 9.0, "reason": "already fits"}
STREAM = {"mode": "stream", "capacity": 61, "n_experts": 128, "residency": 0.48,
          "room_gb": 7.3, "reason": "does not fit at any allowed precision"}


def R(strategy=COMPRESS, blob=BLOB, **kw):
    kw.setdefault("source_bits", 4)
    kw.setdefault("model_name", "TestModel")
    return consent.resolve(strategy, blob, kw.pop("source_bits"), kw.pop("model_name"), **kw)


print("=" * 82); print("1. CONSENT IS ASKED FOR EXACTLY ONE THING"); print("=" * 82)
for st, name in ((NATIVE, "native"), (STREAM, "stream")):
    s, d = R(st)
    check(f"{name} needs no consent -- it does not alter the weights",
          d["source"] == "no-choice-needed" and s["mode"] == st["mode"])
try:
    R(COMPRESS)
    check("compress without a flag, memory or terminal is REFUSED", False, "it proceeded")
except consent.ConsentRequired as e:
    msg = str(e)
    check("compress without a flag, memory or terminal is REFUSED", True)
    check("...and the refusal names both flags", "--compress" in msg and "--exact" in msg)
    check("...and says plainly that the weights change", "WEIGHTS CHANGE" in msg)
    check("...and explains why it refuses rather than guessing", "worse than one that stops" in msg)

print("\n" + "=" * 82); print("2. AN EXPLICIT FLAG ALWAYS WINS"); print("=" * 82)
consent.forget_choice(BLOB)
s, d = R(preference="exact")
check("--exact never compresses", s["mode"] == "stream" and d["mode"] == "exact")
check("...and it inherits the fallback capacity computed with the same memory accounting",
      s["capacity"] == COMPRESS["stream_capacity"], str(s.get("capacity")))
check("...and it is not remembered as consent to compress",
      (consent.load_choice(BLOB) or {}).get("mode") != "compress")
consent.forget_choice(BLOB)
s, d = R(preference="compress")
check("--compress compresses", s["mode"] == "compress" and d["source"] == "flag")
check("...and is remembered", (consent.load_choice(BLOB) or {}).get("mode") == "compress")
consent.forget_choice(BLOB)
s, d = R(preference="compress", remember=False)
check("remember=False leaves no trace", consent.load_choice(BLOB) is None)

print("\n" + "=" * 82); print("3. A REMEMBERED CHOICE, AND WHEN IT STOPS APPLYING"); print("=" * 82)
consent.forget_choice(BLOB)
consent.save_choice(BLOB, "exact")
s, d = R()
check("a remembered 'exact' is honoured with no terminal",
      s["mode"] == "stream" and d["source"] == "remembered")
check("...and is NEVER silently upgraded to compression", s["mode"] != "compress")
consent.forget_choice(BLOB)
consent.save_choice(BLOB, "compress", 3, 128, 4)
s, d = R()
check("a remembered 'compress' at the SAME precision is honoured",
      s["mode"] == "compress" and d["source"] == "remembered")
# Free memory moves between runs. Consent to 3-bit is not consent to 2-bit.
consent.forget_choice(BLOB)
consent.save_choice(BLOB, "compress", 3, 128, 4)
try:
    R({**COMPRESS, "bits": 2, "group_size": 128})
    check("consent to one precision is NOT consent to a lower one", False, "it reused it")
except consent.ConsentRequired:
    check("consent to one precision is NOT consent to a lower one", True)

print("\n" + "=" * 82); print("4. THE STORED FILE ITSELF"); print("=" * 82)
consent.forget_choice(BLOB)
consent.save_choice(BLOB, "compress", 3, 64, 4)
consent.save_choice(BLOB2, "exact")
check("choices do not leak between models",
      consent.load_choice(BLOB)["mode"] == "compress" and
      consent.load_choice(BLOB2)["mode"] == "exact")
check("no temp file is left behind", not os.path.exists(consent.choice_path(BLOB) + ".tmp"))
for bad, why in (("{not json", "malformed JSON"), ("[]", "wrong shape"),
                 ('{"version": 999, "mode": "compress"}', "a future version"),
                 ('{"version": 1, "mode": "banana"}', "an unknown mode"),
                 ('{"version": 1, "mode": "compress"}', "compress with no precision")):
    open(consent.choice_path(BLOB), "w").write(bad)
    check(f"{why} is treated as NO choice, not as consent", consent.load_choice(BLOB) is None)
    try:
        R()
        check(f"...and {why} still refuses rather than proceeding", False, "it proceeded")
    except consent.ConsentRequired:
        check(f"...and {why} still refuses rather than proceeding", True)
consent.forget_choice(BLOB)
check("forgetting a choice that is not there is not an error",
      consent.forget_choice(BLOB) is False)
try:
    consent.save_choice(BLOB, "sideways")
    check("an invalid mode cannot be stored", False)
except ValueError:
    check("an invalid mode cannot be stored", True)

print("\n" + "=" * 82); print("5. THE QUESTION PUT TO A HUMAN"); print("=" * 82)
out = []
def w(s): out.append(s)
for answers, want, why in ((["1"], "compress", "1"), (["2"], "exact", "2"),
                           (["compress"], "compress", "the word"), (["exact"], "exact", "the word"),
                           ([""], "exact", "a bare Enter"),
                           (["x", "y", "1"], "compress", "two bad answers then a good one"),
                           (["x", "y", "z"], "exact", "three unusable answers")):
    it = iter(answers)
    got = consent.ask("M", COMPRESS, 4, reader=lambda _: next(it), writer=w)
    check(f"{why} -> {want}", got == want, got)
def boom(_):
    raise EOFError
check("EOF chooses the option that cannot harm them",
      consent.ask("M", COMPRESS, 4, reader=boom, writer=w) == "exact")
def ctrlc(_):
    raise KeyboardInterrupt
check("Ctrl-C chooses the option that cannot harm them",
      consent.ask("M", COMPRESS, 4, reader=ctrlc, writer=w) == "exact")
q = consent.render_question("MyModel", COMPRESS, 4)
check("the question names the model", "MyModel" in q)
check("...states both sizes exactly", "12.7 GB" in q and "6.0 GB" in q)
check("...says the weights change, in those words", "THE WEIGHTS CHANGE" in q)
check("...offers the exact alternative", "the weights untouched" in q)
check("...mentions the monitor stays on", "quality monitor" in q)
check("...labels the reference measurement as a reference, not a promise",
      "not a promise" in q and "OLMoE" in q)
check("...invents no speed or quality number for THIS model",
      "tok/s" not in q.replace("4.4x faster", ""))

print("\n" + "=" * 82); print("6. THE PROMPT PATH END TO END"); print("=" * 82)
consent.forget_choice(BLOB)
it = iter(["1"])
s, d = R(interactive=True, reader=lambda _: next(it), writer=w)
check("answering 1 compresses and remembers",
      s["mode"] == "compress" and d["source"] == "asked" and
      consent.load_choice(BLOB)["mode"] == "compress")
consent.forget_choice(BLOB)
it = iter(["2"])
s, d = R(interactive=True, reader=lambda _: next(it), writer=w)
check("answering 2 streams and remembers",
      s["mode"] == "stream" and consent.load_choice(BLOB)["mode"] == "exact")
check("...and the streamed fallback keeps its capacity", s.get("capacity") == 61)
consent.forget_choice(BLOB)
it = iter(["1"])
s, d = R(interactive=True, reader=lambda _: next(it), writer=w, remember=False)
check("a one-off answer is not remembered", consent.load_choice(BLOB) is None)

print("\n" + "=" * 82); print("7. THE TERMINAL TEST ITSELF"); print("=" * 82)
import inspect
src = inspect.getsource(consent.is_interactive)
check("a terminal means BOTH ends, not just stdout",
      "stdin.isatty" in src and "stderr.isatty" in src)
check("...and a missing stream is not mistaken for a terminal",
      "except" in src and "return False" in src)
# interactive=True is a REQUEST, not an assertion; the real check still has to pass.
try:
    R(interactive=True)     # no tty under a test runner
    check("interactive=True with no real terminal still refuses", False, "it proceeded")
except (consent.ConsentRequired, StopIteration):
    check("interactive=True with no real terminal still refuses", True)

import shutil
shutil.rmtree(TMP, ignore_errors=True)
print("\n" + "=" * 82)
print(f"{'ALL TESTS PASSED' if not FAIL else str(len(FAIL))+' FAILURES: '+', '.join(FAIL)}")
print("=" * 82)
sys.exit(1 if FAIL else 0)
