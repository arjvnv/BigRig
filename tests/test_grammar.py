"""Structured output: the JSON automaton and the logits processor built on it.

The guarantee under test is narrow and must be exact: whatever the model prefers, every token
that gets through keeps the text a valid prefix of one JSON object, and once the object is
complete nothing but EOS gets through. Everything here is pure -- no model, no server -- so it
runs on a fresh clone. The end-to-end behaviour against a live model is in test_product.
"""
import json
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import mlx.core as mx                                                    # noqa: E402
from bigrig_engine.grammar import (JSONPrefix, JSONProcessor,            # noqa: E402
                                   parse_response_format, required_keys, schema_instruction)

FAIL = []


def check(n, c, d=""):
    (print(f"  PASS  {n}") if c else (FAIL.append(n), print(f"  FAIL  {n}  {d}")))


def accepts(text, require_object=True):
    a = JSONPrefix(require_object=require_object)
    return all(a.feed(ch) for ch in text), a


print("=" * 84); print("1. EVERY PREFIX OF A VALID DOCUMENT IS ACCEPTED, AND ONLY THE WHOLE IS DONE"); print("=" * 84)
DOCS = ['{}', '{"a":1}',
        '{"a": -1.5e+3, "b": [true, false, null, "x\\"y\\n", {"c": []}]}',
        '{"k": "\\u00e9\\u4e2d"}', '{"nested": {"deep": {"deeper": [1, [2, [3]]]}}}',
        '{"s": "with spaces and\\ttabs", "n": 0, "m": 0.0, "e": 1E10, "neg": -0}',
        '{"empty_str": "", "empty_arr": [], "empty_obj": {}}', '{"a":[[],[[]],{}]}',
        '{"unicode": "héllo wörld 日本語 🎉"}', ' \n {"ws": 1} \n ']
for doc in DOCS:
    json.loads(doc)                                     # the fixture must itself be valid
    a = JSONPrefix(require_object=True)
    bad_at = next((i for i, ch in enumerate(doc) if not a.feed(ch)), None)
    check(f"every prefix accepted: {doc.strip()[:34]!r}", bad_at is None,
          f"rejected at {bad_at}: {doc[:bad_at + 1]!r}" if bad_at is not None else "")
    ok, b = accepts(doc.strip())
    check(f"...and the whole document is done: {doc.strip()[:34]!r}", ok and b.done)
    check(f"...and refuses content after it: {doc.strip()[:34]!r}", not b.feed("x"))
for s in ['{', '{"a"', '{"a":', '{"a":1', '{"a":[1,2', '{"a":"str', '{"a":tr', '{"a":1e', '{"a":[']:
    ok, a = accepts(s)
    check(f"a partial document is accepted but not done: {s!r}", ok and not a.done)

print("\n" + "=" * 84); print("2. EVERY CORRUPTION IS REJECTED"); print("=" * 84)
random.seed(7)
rejected = total = 0
for doc in DOCS:
    d = doc.strip()
    for _ in range(60):
        i = random.randrange(len(d))
        bad = d[:i] + random.choice("@#$%^&*()~`;<>?]}") + d[i + 1:]
        try:
            json.loads(bad)
            continue                                    # a corruption that stayed valid
        except ValueError:
            pass
        total += 1
        ok, a = accepts(bad)
        rejected += int(not (ok and a.done))
check("every corrupted document is rejected somewhere before completion",
      rejected == total, f"{rejected} of {total}")
for s in ['x', '[1]', '{"a" 1}', '{"a":}', '{,}', '{"a":1,}', '{"a":01}', '{"a":1.}', '{"a":.5}',
          '{"a":tru }', '{"a":"\\q"}', '{"a":"\n"}', '{"a":1}}', '{{}', '{"a":[,]}', '{"a":[1,]}']:
    ok, a = accepts(s)
    check(f"rejects {s!r}", not (ok and a.done))
check("a bare array is refused when an object is required", not accepts('[1]')[0])
check("...but accepted when it is not", accepts('[1]', require_object=False)[1].done)

print("\n" + "=" * 84); print("3. KEYS, COPIES AND ROLLBACK"); print("=" * 84)
ok, a = accepts('{"name": "x", "inner": {"nested": 1}, "n": 2}')
check("top-level keys are recorded and nested ones are not", a.keys == ["name", "inner", "n"], str(a.keys))
a = JSONPrefix(require_object=True)
for ch in '{"a":':
    a.feed(ch)
b = a.copy()
b.feed('1'); b.feed('}')
check("copy() is a real snapshot: feeding the copy never moves the original", not a.done and b.done)
a = JSONPrefix(require_object=True)
a.feed_text('{"a":')
before = a.copy()
check("feed_text rejects a bad string and leaves the automaton exactly as it was",
      not a.feed_text('1}}x') and a.state == before.state and a.stack == before.stack)
check("...and accepts a good one", a.feed_text('1}') and a.done)

print("\n" + "=" * 84); print("4. THE REQUEST SHAPE"); print("=" * 84)
check("None is no format", parse_response_format(None) == (None, None))
check("text is no format", parse_response_format({"type": "text"}) == (None, None))
check("json_object", parse_response_format({"type": "json_object"}) == ("json_object", None))
k, sc = parse_response_format({"type": "json_schema", "json_schema": {"name": "x", "schema": {"type": "object", "required": ["a"]}}})
check("json_schema unwraps the nested schema", k == "json_schema" and sc == {"type": "object", "required": ["a"]})
k, sc = parse_response_format({"type": "json_schema", "json_schema": {"type": "object", "required": ["b"]}})
check("...and accepts a schema given flat", sc == {"type": "object", "required": ["b"]})
for bad in ("json", {"kind": "x"}, {"type": "yaml"}, {"type": "json_schema"},
            {"type": "json_schema", "json_schema": "no"}):
    try:
        parse_response_format(bad)
        check(f"refuses {bad!r}", False)
    except ValueError as e:
        check(f"refuses {bad!r} with a sentence", len(str(e)) > 10)
check("required_keys reads a nested or flat schema",
      required_keys({"schema": {"required": ["a", "b"]}}) == ["a", "b"]
      and required_keys({"required": ["c"]}) == ["c"] and required_keys(None) == [])
check("the instruction names the required keys",
      '"a"' in schema_instruction("json_schema", {"required": ["a"]}) and "JSON" in schema_instruction("json_object", None))
check("...and is empty when there is no format", schema_instruction(None, None) == "")

print("\n" + "=" * 84); print("5. THE PROCESSOR, WITH A REAL TOKENIZER"); print("=" * 84)
_tok_dir = next((os.path.join(ROOT, "models", m) for m in
                 ("OLMoE-1B-7B-0125-4bit", "Qwen3.6-35B-A3B-4bit", "DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx")
                 if os.path.exists(os.path.join(ROOT, "models", m, "tokenizer.json"))), None)
if _tok_dir is None:
    print("  SKIPPED - no local tokenizer; the processor checks need a real vocabulary.")
else:
    from transformers import AutoTokenizer
    from mlx_lm.tokenizer_utils import TokenizerWrapper
    tok = TokenizerWrapper(AutoTokenizer.from_pretrained(_tok_dir))
    V = len(tok._tokenizer)
    enc = lambda s: tok._tokenizer.encode(s, add_special_tokens=False)   # noqa: E731
    PROMPT = mx.array(enc("Tell me about apples."), dtype=mx.int32)

    def run(proc, gen_ids, logits):
        return proc(mx.concatenate([PROMPT, mx.array(gen_ids, dtype=mx.int32)]) if gen_ids
                    else PROMPT, logits)

    p = JSONProcessor(tok)
    lg = mx.full((V,), -10.0).at[enc("Sure! Here")[0]].add(20.0)
    run(p, [], lg)                                     # first call anchors past the prompt
    top = int(mx.argmax(run(p, [], lg)))
    check("the prompt is skipped and a prose opener is masked to an object opener",
          tok.decode([top]).strip().startswith("{"), repr(tok.decode([top])))

    random.seed(11)
    closers = [enc(t)[0] for t in ("}", "]", '"', ",", ":")]
    for trial in range(3):
        p = JSONProcessor(tok); ids = []; text = ""; ended = None
        run(p, [], mx.zeros((V,)))
        for i in range(120):
            lg = mx.array([random.uniform(-3, 3) for _ in range(V)])
            if i > 25:
                lg = lg.at[mx.array(closers)].add(6.0)
            nxt = int(mx.argmax(run(p, ids, lg)))
            if nxt in p.eos:
                ended = i
                break
            ids.append(nxt); text += tok.decode([nxt])
            if not accepts(text)[0]:
                break
        check(f"trial {trial}: a random model was kept inside a valid JSON prefix for {len(ids)} tokens",
              accepts(text)[0], repr(text[-50:]))
        if ended is not None:
            try:
                json.loads(text); parses = True
            except ValueError:
                parses = False
            check(f"trial {trial}: it ended by EOS with a document that parses", parses, repr(text[:60]))
            out = run(p, ids, mx.zeros((V,)))
            check(f"trial {trial}: after completion only EOS is legal",
                  int((out > -1e30).sum()) == len(p.eos))

    doc = '{"name": "Ada", "age": 36, "tags": ["x", "y"], "ok": true}'
    p = JSONProcessor(tok); ids = []
    run(p, [], mx.zeros((V,)))
    changed = None
    for t in enc(doc):
        got = int(mx.argmax(run(p, ids, mx.full((V,), -10.0).at[t].add(20.0))))
        ids.append(got)
        if got != t:
            changed = (tok.decode([t]), tok.decode([got])); break
    check("a valid document the model wanted passes through token for token unchanged",
          changed is None and tok.decode(ids) == doc, str(changed))

    # REQUIRED KEYS ARE ENFORCED AT THE CLOSING BRACE. Once `}` closes the top-level object
    # nothing can add a key, so refusing EOS afterwards would be too late -- the model would
    # be left with nothing legal but whitespace. The brace itself must be refused.
    brace = enc("}")[0]
    p = JSONProcessor(tok, schema={"type": "object", "required": ["name", "age"]})
    run(p, [], mx.zeros((V,)))
    ids = enc('{"name": "x"')
    got = int(mx.argmax(run(p, ids, mx.full((V,), -10.0).at[brace].add(20.0))))
    check("with a required key missing, the closing brace is refused and the object stays open",
          got != brace and tok.decode([got]).strip() in (",",), repr(tok.decode([got])))
    p = JSONProcessor(tok, schema={"type": "object", "required": ["name", "age"]})
    run(p, [], mx.zeros((V,)))
    ids = enc('{"name": "x", "age": 3')
    got = int(mx.argmax(run(p, ids, mx.full((V,), -10.0).at[brace].add(20.0))))
    check("with every required key present, the closing brace is allowed", got == brace)
    out = run(p, ids + [brace], mx.zeros((V,)))
    check("...and after it only EOS is legal", int((out > -1e30).sum()) == len(p.eos))
    # THE STALL RULE. A model that cannot produce a required key emits whitespace forever and
    # the reply never closes. After STALL_TOKENS whitespace-only tokens the gate is released.
    from bigrig_engine.grammar import STALL_TOKENS
    p = JSONProcessor(tok, schema={"type": "object", "required": ["name", "age"]})
    run(p, [], mx.zeros((V,)))
    ids = enc('{"type": "x"')
    nl = enc("\n")[0]
    for i in range(STALL_TOKENS):
        got = int(mx.argmax(run(p, ids, mx.full((V,), -10.0).at[brace].add(20.0).at[nl].add(15.0))))
        check(f"stall {i + 1}: the brace is still refused while the model only emits whitespace",
              got != brace, repr(tok.decode([got]))) if i < STALL_TOKENS - 1 else None
        ids.append(got)
    got = int(mx.argmax(run(p, ids, mx.full((V,), -10.0).at[brace].add(20.0))))
    check(f"after {STALL_TOKENS} whitespace-only tokens the gate releases and the brace is allowed",
          got == brace and p.relaxed, f"relaxed={p.relaxed} got={tok.decode([got])!r}")
    try:
        json.loads(tok.decode(ids + [brace])); closes = True
    except ValueError:
        closes = False
    check("...so the reply is a parseable object rather than an unclosed one", closes)

print()
print("=" * 84)
print("ALL TESTS PASSED" if not FAIL else f"{len(FAIL)} FAILURES: " + ", ".join(FAIL))
print("=" * 84)
sys.exit(1 if FAIL else 0)
