"""The web interface, checked the way a browser would break it rather than by reading it.

WHY THIS FILE IS ADVERSARIAL
    Every failure this catches is silent in a browser. A selector with a typo returns null and
    the next property access throws inside an async refresh -- no red text, no console for a
    non-developer to open, just a page that stopped updating and still looks fine. A stray
    innerHTML on engine-supplied text is a script tag away from executing. A syntax error in one
    function takes down the whole script tag, including the parts that were correct.

    So: the script is parsed by a real JavaScript engine, every id the script reaches for is
    checked against the markup that must contain it, and the page is required to fetch nothing
    from anywhere.
"""
import json
import os
import re
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAGE = os.path.join(ROOT, "bigrig_engine", "webui.html")
FAIL = []


def check(n, c, d=""):
    (print(f"  PASS  {n}") if c else (FAIL.append(n), print(f"  FAIL  {n}  {d}")))


html = open(PAGE, encoding="utf-8").read()
script = html[html.index("<script>") + 8:html.rindex("</script>")]
markup = html[:html.index("<script>")]

print("=" * 84)
print("1. THE SCRIPT MUST PARSE, OR NONE OF IT RUNS")
print("=" * 84)
node = shutil.which("node")
if not node:
    check("no JavaScript engine available to parse with (skipped, not passed)", True)
else:
    # `--check` parses without executing, which is what is wanted: executing would need a DOM.
    tmp = os.path.join(ROOT, ".webui-syntax-check.js")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(script)
        r = subprocess.run([node, "--check", tmp], capture_output=True, text=True)
        check("the page's script parses as JavaScript", r.returncode == 0,
              (r.stderr or "").strip().split("\n")[-1][:200])
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)

print()
print("=" * 84)
print("2. EVERY ELEMENT THE SCRIPT REACHES FOR MUST EXIST IN THE MARKUP")
print("=" * 84)
# THE BUG THIS CATCHES. `$("#c-tps")` on an id that is not there returns null, and the very next
# `.textContent` throws. Inside `refresh()` -- an async function whose rejection nothing handles
# -- that stops the entire page updating, with no visible error. A renamed id in the markup and
# a missed rename in the script is a one-character change that produces exactly this.
ids_in_markup = set(re.findall(r'id="([A-Za-z0-9_-]+)"', markup))
used = set(re.findall(r'\$\("#([A-Za-z0-9_-]+)"\)', script))
used |= set(re.findall(r'querySelector\("#([A-Za-z0-9_-]+)"\)', script))
used |= set(re.findall(r'set\("#([A-Za-z0-9_-]+)"', script))
# Ids the script CREATES at runtime are legitimately absent from the static markup. They are
# listed explicitly rather than pattern-matched, so adding one is a deliberate act.
created = {"presets"}
missing = sorted(u for u in used if u not in ids_in_markup and u not in created)
check("no selector in the script points at an element that does not exist",
      not missing, f"missing: {missing}")
check("...and that was a real check, not an empty set", len(used) > 40, f"{len(used)} selectors")

# The reverse direction is a weaker signal (markup may hold ids only CSS uses), so it is
# reported rather than failed -- except for the panels added with the console, which exist to be
# driven and are dead weight if nothing drives them.
for must in ("feed-body", "feed-cnt", "feed-pause", "pc-reused", "pc-hits", "pc-bytes",
             "pc-clear", "pc-fill", "pc-msg"):
    check(f"the console drives #{must}", must in used and must in ids_in_markup,
          f"in markup: {must in ids_in_markup}, used by script: {must in used}")

print()
print("=" * 84)
print("3. ENGINE TEXT MUST NOT BE ABLE TO BECOME MARKUP")
print("=" * 84)
# Event text carries model names, file paths and exception strings straight out of the engine.
# The feed builds its rows with textContent and createElement for that reason; an innerHTML on
# that path would make a `<` in an error message into a tag.
feed_fn = script[script.index("async function pollFeed"):script.index("function paintPromptCache")]
check("the activity feed builds rows with textContent, never innerHTML",
      ".innerHTML" not in feed_fn.replace('body.innerHTML = ""', ""),
      "innerHTML appears on the event path")
check("...and it says why, so the next edit does not undo it",
      "textContent, never innerHTML" in feed_fn)
check("the feed is bounded on the page as well as on the server",
      "removeChild" in feed_fn or "slice" in feed_fn)

print()
print("=" * 84)
print("4. THE PAGE MUST FETCH NOTHING FROM ANYWHERE")
print("=" * 84)
# The product's claim is that the model runs entirely on this machine. A page that pulls a font
# or a chart library from a CDN contradicts it in the network tab, whatever the text says.
ext = re.findall(r'(?:src|href)="(https?:)?//[^"]+"', html)
check("no script, style, font or image is loaded from another host", not ext, f"{ext[:3]}")
check("...and no fetch() goes anywhere but this server",
      not re.findall(r'fetch\(\s*["\'`]https?://', script))
check("charts are drawn inline rather than by a library",
      "<svg" in script and "cdn" not in html.lower())

print()
print("=" * 84)
print("5. THE CONTROLS MUST MATCH WHAT THE SERVER ACTUALLY ACCEPTS")
print("=" * 84)
# A button that posts a field the server ignores is a control that appears to work and does
# nothing. Checked against the server's own source rather than against a list kept here.
srv = open(os.path.join(ROOT, "bigrig_engine", "server.py"), encoding="utf-8").read()
posted = set(re.findall(r'JSON\.stringify\(\{\s*([a-z_]+)\s*:', script))
for field in sorted(posted):
    check(f"the server reads `{field}`, which the page posts", f'"{field}"' in srv or
          f"'{field}'" in srv, "posted by the page, unknown to the server")
check("the page asks for events with a `since`, so the feed is incremental",
      "/v1/events?since=" in script and "since" in srv)
# stats() is built in session.py and served verbatim by /health, so that is where the contract
# actually lives -- checking server.py for these was checking the wrong file and passed only by
# accident of the words appearing in a comment.
_sess = open(os.path.join(ROOT, "bigrig_engine", "session.py"), encoding="utf-8").read()
_reads = set(re.findall(r'h\.([a-z_]+)', script)) | set(re.findall(r'health\.([a-z_]+)', script))
_pc = {k for k in _reads if k.startswith("prompt_")}
check("every prompt-cache field the panel reads is one the session reports",
      _pc and all(f'"{k}"' in _sess for k in _pc), f"{sorted(_pc)}")
check("...and the panel reads more than one of them, so that was a real check",
      len(_pc) >= 4, f"{sorted(_pc)}")
check("clearing remembered prompts does not go through the reload path",
      "clear_prompt_cache" in srv
      and srv.index("clear_prompt_cache") < srv.index("state.pending_config = want"))

print()
print("=" * 84)
print("6. THE PAGE MUST ACTUALLY RUN, NOT MERELY PARSE")
print("=" * 84)
# Sections 1 to 5 read the page. This one EXECUTES it, against a minimal DOM and against
# responses captured from a real server -- including a run where the machine genuinely went short
# of memory and the pool was rebuilt, so the fixtures contain the events that only appear when
# something goes wrong. Anything the page throws is a failure here; in a browser it would be
# invisible, because it throws inside an async refresh whose rejection nothing handles.
FIXDIR = os.path.join(ROOT, "tests", "webui", "fixtures")
RUNNER = os.path.join(ROOT, "tests", "webui", "run.js")
if not node:
    check("no JavaScript engine to execute with (skipped, not passed)", True)
elif not os.path.isdir(FIXDIR):
    check("no captured server responses to run against (skipped, not passed)", True)
else:
    r = subprocess.run([node, RUNNER, FIXDIR], capture_output=True, text=True)
    try:
        out = json.loads(r.stdout)
    except ValueError:
        out = None
        check("the harness produced a result", False, (r.stderr or r.stdout)[-300:])
    if out is not None:
        check("the page loads and refreshes without throwing",
              not out["problems"], f"{out['problems'][:3]}")
        check("...and it asked the server for the things it needs",
              any("/health" in u for u in out["fetched"])
              and any("/v1/events" in u for u in out["fetched"]), f"{out['fetched']}")

        w = out["writes"]
        # A page that runs but writes nothing is a page that silently shows dashes. Each of these
        # is a value a person reads to decide something.
        check("the model is named", bool(w.get("m-model")) and "—" not in w.get("m-model", "—"),
              repr(w.get("m-model")))
        check("the memory meter is filled in", "GB" in (w.get("m-mem") or ""), repr(w.get("m-mem")))
        check("the speed card is a number, not a dash",
              (w.get("c-tps") or "—") != "—", repr(w.get("c-tps")))

        # The two surfaces added with the console.
        check("remembered prompts reports tokens reused",
              (w.get("pc-reused") or "") not in ("", "—"), repr(w.get("pc-reused")))
        check("...and how much memory it is holding, against its limit",
              "GB" in (w.get("pc-bytes") or "") and "set aside" in (w.get("pc-bytes-s") or ""),
              f"{w.get('pc-bytes')!r} / {w.get('pc-bytes-s')!r}")
        check("...and how many replies started from something already read",
              "/" in (w.get("pc-hits") or ""), repr(w.get("pc-hits")))
        check("the activity feed rendered one row per event",
              out["feedRows"] == len(json.load(open(os.path.join(FIXDIR, "events.json")))["events"]),
              f"{out['feedRows']} rows")
        # THE BUG THE `since` PROTOCOL EXISTS TO PREVENT. Two polls happen in this run. If the
        # page re-rendered from scratch, or asked for everything each time, the row count would be
        # double the event count and the console would show every line twice.
        check("...and a second poll does not append the same events again",
              out["feedRows"] == len(json.load(open(os.path.join(FIXDIR, "events.json")))["events"]),
              f"{out['feedRows']} rows after two polls")
        check("...having asked only for what it had not already seen",
              any("since=0" in u for u in out["fetched"])
              and any(re.search(r"since=[1-9]", u) for u in out["fetched"]), f"{out['fetched']}")
        check("the feed counts what it showed", "event" in (w.get("feed-cnt") or ""),
              repr(w.get("feed-cnt")))

# A CONTROL THAT OFFERS A SETTING THE SERVER WOULD REFUSE IS A CONTROL THAT LIES.
#     The page decides how far its sliders may be dragged by doing the planner's arithmetic in
#     JavaScript. That arithmetic is a copy, and copies drift: when the engine began setting
#     memory aside for remembered prompts, the reserve gained a term and the page did not, so the
#     slider offered about five more experts a layer than the planner would accept.
#
#     Checked by term, against the server's own reserve, rather than by re-deriving the sum here
#     -- which would be a third copy and would drift from both.
#     The sum now lives in ONE method, `Session.slot_room`; server.price_config refuses above it
#     and stats() reports its result, which the page starts from instead of re-adding terms. It
#     replaced copies that had drifted: /config priced budget - non-expert - working memory and
#     accepted 65 slots on Qwen3.6-35B-A3B at 9 GB (measured peak 10.36 GB), and this page
#     offered 57 where the planner allows 43.
#
#     WIRING, read off the source.
_stats_src = _sess[_sess.index("    def stats(self)"):]
_stats_src = _stats_src[:_stats_src.index("\n    def ", 10)]
_unreported = [k for k in ("pool_budget_gb", "slot_fixed_gb", "slot_cache_in_reserve")
               if f'"{k}"' not in _stats_src]
check("the server reports the slot sum's result, its budget and its cache rule",
      not _unreported, f"never reported: {_unreported}")
check("...and the reported sum and budget ARE slot_room's, not a fourth copy",
      'self.slot_room()["fixed_gb"]' in _stats_src and 'self.slot_room()["budget_gb"]' in _stats_src)
check("...and the cache rule reported is the one decided at planning time",
      "reserve_holds_cache" in _stats_src[_stats_src.index('"slot_cache_in_reserve"'):][:160])
_fixed = script[script.index("function slotFixedGb"):script.index("function slotsThatFit")]
_fit = script[script.index("function slotsThatFit"):script.index("function kvTokensThatFit")]
check("the page's expert slider starts from the server's sum and its cache rule",
      "h.slot_fixed_gb" in _fixed and "h.slot_cache_in_reserve" in _fixed)
check("...against the POOL budget, which a draft model or a head has already been taken off",
      "h.pool_budget_gb" in _fit and "slotFixedGb()" in _fit)
_kv = script[script.index("function kvTokensThatFit"):]
_kv = _kv[:_kv.index("\n}")]
# The reply-length slider mirrors the server's REPLY limit (Session._token_ceiling: budget -
# resident - working memory, no headroom), not the planner's slot sum. Pointed at the slot sum it
# offered 2,519 tokens at 43 slots on Qwen3.6 at 9 GB, far below what the server accepts.
check("the reply-length slider prices the terms the server's reply limit uses",
      "prompt_cache_gb" in _kv and "working_memory_gb" in _kv and "slotFixedGb()" not in _kv)

# THE NUMBER, AGAINST THE PLANNER RATHER THAN AGAINST ITSELF. Terms are the ones a live
# Qwen3.6-35B-A3B-4bit session reported at a 9.00 GB budget on 2026-11-11 (40 layers, 256 experts,
# 0.0708 GB a slot); its planner said 43. autoconfig.choose_capacity is separate code.
import math as _math                                                        # noqa: E402
import types as _types                                                      # noqa: E402
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))
from bigrig_engine import autoconfig as _A, server as _server               # noqa: E402
from bigrig_engine.session import Session as _Sess                          # noqa: E402
_L, _B, _NE = 40, 1769472, 1.103603855
_PER = _B * _L / 1e9
_man = {"layers": {str(i): {"n_experts": 256, "bytes_per_expert": _B} for i in range(_L)},
        "total_bytes": 256 * _B * _L}


def _session(pool=9.0, kvb=20480, planned_from=38, explicit=None, flat=False, knee=None,
             **over):
    """A session's planning terms as Session.__init__ derives them for a POOL budget.

    flat=True is a model whose config gave no KV figure: the reserve is the flat RESERVE_GB and
    holds no prompt cache, while kv_bytes_per_token is still filled in after load by
    _measure_kv -- the state a loaded session is really in. `knee` is a saved knee, which
    __init__ applies over the planner's choice when no capacity was named.
    """
    pc = round(min(0.5, pool * 0.06), 2)
    reserve = _A.RESERVE_GB if flat else round(0.3 + 3.0 + pc, 2)
    t = dict(plan={"n_experts": 256, "bytes_per_expert": _B, "n_layers": _L, "capacity": 13,
                   "available_gb": pool},
             budget_gb=9.0, pool_budget_gb=round(pool, 2), serving_reserve_gb=reserve,
             prompt_cache_gb=pc, headroom_gb=_A.scaled_headroom(pool),
             streamed_non_expert_gb=_NE, kv_bytes_per_token=kvb, reserve_holds_cache=not flat,
             planned_from=planned_from, init_kwargs={"capacity": explicit}, knee=knee)
    t.update(over)
    ns = _types.SimpleNamespace(**t)
    ns.slot_room = _types.MethodType(_Sess.slot_room, ns)
    return ns


# The planner also reports `under_pressure`, a live system reading that takes about half a
# second and plays no part in the capacity; a constant keeps this file fast.
_A.under_pressure = lambda: False


def _planner(ns):
    return _A.choose_capacity(_man, budget_gb=ns.plan["available_gb"], top_k=8,
                              reserve_gb=ns.serving_reserve_gb, non_expert_gb=_NE,
                              headroom_gb=ns.headroom_gb)["capacity"]


_q = _session()
check("on the measured Qwen3.6 terms the server's sum equals the planner's answer (43)",
      _q.slot_room()["fits"] == _planner(_q) == 43, f"{_q.slot_room()['fits']} vs {_planner(_q)}")
_d = _session(pool=8.4, planned_from=30)                 # a 0.6 GB draft off a 9 GB ceiling
check("with a draft model it prices the POOL budget, and still equals the planner (35, not 43)",
      _d.slot_room()["fits"] == _planner(_d) == 35, f"{_d.slot_room()['fits']} vs {_planner(_d)}")
check("a reply's KV cache only ever lowers it",
      _q.slot_room(kv_tokens=100_000)["fits"] < _q.slot_room()["fits"])
check("a smaller previewed cache frees exactly its own slots (0.5 GB -> 50)",
      _q.slot_room(prompt_cache_gb=0.0)["fits"] == int((9.0 - 3.3 - 1.0 - _NE) / _PER) == 50)
_flat = _session(flat=True)               # flat RESERVE_GB, KV figure measured after load
check("...but not when the reserve is a flat constant that never held the cache, even once the "
      "KV figure has been measured after load",
      _flat.slot_room(prompt_cache_gb=0.0)["fits"] == _flat.slot_room()["fits"] == _planner(_flat)
      == 55, f"{_flat.slot_room(prompt_cache_gb=0.0)['fits']} vs planner {_planner(_flat)}")
# A budget where rounding to 2 places moves the answer: slot_room must use the exact one.
def _rounding_moves_it(b):
    """Does the PLANNER answer differently at b and at b rounded to 2 places? (The search uses
    the planner, not slot_room, so a slot_room that rounds still gets a telling budget.)"""
    exact = _session(pool=b)
    return _planner(exact) != _planner(_session(pool=b, plan={**exact.plan,
                                                              "available_gb": round(b, 2)}))


_rb = next(b / 1000 for b in range(8000, 8200) if _rounding_moves_it(b / 1000))
_r = _session(pool=_rb, planned_from=20)
check(f"it prices the planner's exact budget, not the rounded one ({_rb} GB)",
      _r.slot_room()["fits"] == _planner(_r), f"{_r.slot_room()['fits']} vs {_planner(_r)}")

# THE /config PRICE, CALLED, NOT READ. A gate that only finds the function name in the handler
# passes with the comparison broken (a reviewer changed it to `fits + 20` and this file stayed
# green); these call the same price_config the handler calls.
def _price(ns, body):
    err, want = _server.price_config(ns, body)
    return err, want


check("65 slots, which peaked at 10.36 GB against a 9 GB ceiling, are refused",
      "most that fits is 43" in (_price(_q, {"capacity": 65})[0] or ""))
check("...and so is 44, one past the planner", _price(_q, {"capacity": 44})[0] is not None)
check("...while 43 is accepted as asked", _price(_q, {"capacity": 43}) == (None, {"capacity": 43}))
check("a cache raised in the same body is priced: capacity 43 with an 8 GB cache is refused",
      _price(_q, {"capacity": 43, "prompt_cache_gb": 8})[0] is not None)
check("a cache lowered in the same body buys its slots: 50 with no cache is accepted, 51 is not",
      _price(_q, {"capacity": 50, "prompt_cache_gb": 0})[0] is None
      and _price(_q, {"capacity": 51, "prompt_cache_gb": 0})[0] is not None)
check("with a draft model the price is the pool's: 36 refused, 35 accepted",
      _price(_d, {"capacity": 36})[0] is not None and _price(_d, {"capacity": 35})[0] is None)
_x = _session(explicit=43)
check("a cache change alone is priced when the capacity was set explicitly at launch",
      _price(_x, {"prompt_cache_gb": 8})[0] is not None
      and _price(_x, {"prompt_cache_gb": 0.25})[0] is None)
check("...and passes when the planner chose it, because the reload re-plans around the cache",
      _price(_q, {"prompt_cache_gb": 8}) == (None, {"prompt_cache_gb": 8.0}))
_k = _session(knee={"capacity": 38})       # the flagship's saved 9 GB knee
check("...unless a saved knee will be applied over the re-plan: 38 with a 1 GB cache is refused",
      "most that fits is 36" in (_price(_k, {"prompt_cache_gb": 1.0})[0] or ""),
      str(_price(_k, {"prompt_cache_gb": 1.0})))
check("...as is 38 with a 3 GB cache, which re-plans to 8 and used to reload at 38",
      _price(_k, {"prompt_cache_gb": 3.0})[0] is not None)
check("...while the same knee with the cache it already has, or less, passes",
      _price(_k, {"prompt_cache_gb": 0.5})[0] is None and _price(_k, {"prompt_cache_gb": 0})[0] is None)
check("a flat reserve does not hand out a cache it never held: 56 with no cache is refused",
      _price(_flat, {"capacity": 56, "prompt_cache_gb": 0})[0] is not None
      and _price(_flat, {"capacity": 55, "prompt_cache_gb": 0})[0] is None)
_over = _session(planned_from=60, explicit=60)           # started over plan with --residency
check("a session over plan can step down even while still above the plan",
      _price(_over, {"capacity": 55})[0] is None)
check("...including by lowering only its cache",
      _price(_over, {"prompt_cache_gb": 0})[0] is None)
check("...but not by also raising its cache, and not upward",
      _price(_over, {"capacity": 55, "prompt_cache_gb": 1.0})[0] is not None
      and _price(_over, {"capacity": 61})[0] is not None)
check("a fraction is priced as the slots it becomes: 0.2 (51) refused, 0.16 (41) accepted",
      _price(_q, {"capacity": 0.2})[0] is not None
      and _price(_q, {"capacity": 0.16}) == (None, {"capacity": 0.16}))
_bad = [{"capacity": v} for v in ("abc", None and 0, 0, -1, 300, _math.nan, _math.inf)]
_bad += [{"prompt_cache_gb": v} for v in ("abc", -1, 9, _math.nan)]
_bad = [b for b in _bad if list(b.values())[0] is not None]
check("hostile values are refused with a message, never raised",
      all(isinstance(_price(_q, b)[0], str) for b in _bad),
      str([b for b in _bad if not isinstance(_price(_q, b)[0], str)]))

# THE PAGE'S ARITHMETIC, RUN, AGAINST slot_room. The page's own functions, cut from the page
# this file already parsed, evaluated under node on what stats() would report for each session
# (same fields, same rounding). Matching names is not enough: a flipped sign, a ceil for a floor
# or a swap that ignores the cache rule all left the name checks green.
def _health(ns):
    room = ns.slot_room()
    return {"pool_budget_gb": round(room["budget_gb"], 6), "slot_fixed_gb": round(room["fixed_gb"], 6),
            "slot_cache_in_reserve": bool(ns.reserve_holds_cache), "prompt_cache_gb": ns.prompt_cache_gb,
            "gb_per_slot": round(room["per_slot_gb"], 6), "n_experts": 256,
            "kv_bytes_per_token": ns.kv_bytes_per_token, "budget_gb": ns.budget_gb}


_cases = [("plain", _q, None, 0), ("plain, no cache previewed", _q, 0.0, 0),
          ("plain, 2 GB cache previewed", _q, 2.0, 0), ("plain, 100k-token reply", _q, None, 100000),
          ("draft", _d, None, 0), ("flat reserve, no cache previewed", _flat, 0.0, 0),
          (f"exact budget {_rb}", _r, None, 0)]
if node:
    _js = (script[script.index("function slotFixedGb"):script.index("function kvTokensThatFit")]
           + "\nconst out = []; for (const c of JSON.parse(process.argv[1])) { health = c.h; "
           "cfgWant = c.pc === null ? null : {pcache: c.pc}; out.push(slotsThatFit(c.kv)); }"
           "\nconsole.log(JSON.stringify(out));")
    _arg = json.dumps([{"h": _health(ns), "pc": pc, "kv": kv} for _, ns, pc, kv in _cases])
    _r_node = subprocess.run([node, "-e", "let health, cfgWant;\n" + _js, _arg],
                             capture_output=True, text=True)
    try:
        _page = json.loads(_r_node.stdout)
    except ValueError:
        _page = None
    check("the page's slider functions run under node", _page is not None, _r_node.stderr[-300:])
    for (label, ns, pc, kv), got in zip(_cases, _page or []):
        want = ns.slot_room(kv_tokens=kv, prompt_cache_gb=pc)["fits"]
        check(f"page slider = server's slot_room: {label} ({want})", got == want, f"page {got}")
else:
    check("node is available to run the page's slider arithmetic", False, "install node")

# AND THROUGH THE REAL HANDLER, so the wiring is proven by a refusal over HTTP, not by a string.
from _fakeserver import fake_server as _fake, post as _fpost, FakeSession as _FS  # noqa: E402
_fs = _FS()
for _k, _v in vars(_session()).items():
    if _k != "slot_room":
        setattr(_fs, _k, _v)
_fs.slot_room = _types.MethodType(_Sess.slot_room, _fs)
with _fake(_fs) as (_url, _state, _):
    _st, _body, _ = _fpost(_url, "/v1/config", {"capacity": 65})
check("/v1/config over HTTP refuses 65 with the planner's number",
      _st == 400 and "most that fits is 43" in json.dumps(_body), f"{_st} {_body}")

# A SLIDER MUST SHOW THE SETTING, NOT ITS CONSEQUENCE. With whole layers in play the residency
# a user asks for and the residency the streamed layers end up with are different numbers -- 12
# becomes one full layer plus 9 slots elsewhere, and `capacity` reports the 9. Seeding from it
# made Apply look like a refusal: the slider jumped back every time it was used.
check("the residency slider is seeded from the budget that was set, not from what it produced",
      "planned_from" in script, "the page reads only `capacity`, which is the derived number")
check("...and the server reports that budget for it to read", '"planned_from"' in _sess)

print()
print("=" * 84)
print("8. THE CODE PAGE MUST SAY WHEN AN AGENT WILL NOT WORK")
print("=" * 84)
# WHY THE VERDICT IS THE FIRST THING ON THE PAGE. An agent pointed at a model with no tool-call
# format does not fail loudly -- it ANSWERS THE QUESTION instead of doing the work. That reads as
# a bad model rather than a wrong setup, and there was previously nowhere at all to find out
# which it was. Everything else on the page was already true and none of it was findable: the
# endpoints were printed once at startup and scrolled away, `bigrig launch` was a line in --help.
check("the page leads with whether this model can call tools at all",
      "tool-verdict" in markup and "supports_tools" in script)
check("...read from the session rather than assumed",
      '"supports_tools": self.supports_tools()' in
      open(os.path.join(ROOT, "bigrig_engine", "session.py"), encoding="utf-8").read())
check("all three endpoints are shown, not just the two that were printed at startup",
      all(x in markup for x in ("ep-openai", "ep-anthropic", "ep-responses")))
check("the launch command is built per agent rather than hardcoded once",
      "AGENTS" in script and "bigrig launch --agent codex" in script)
check("the model name goes in as text, never as markup",
      "textContent, not innerHTML" in script)
check("a copy button that cannot reach the clipboard says nothing false",
      "no clipboard permission" in script)
check("the page does not poll while it is hidden",
      '$("#v-code").hidden) return' in script)

if node and os.path.isdir(FIXDIR):
    # THE CASE THAT MATTERS. Rendered against a model that CANNOT call tools -- which is the
    # situation the page exists to make visible. A page that only ever renders the happy path is
    # a page nobody has checked.
    import copy as _copy
    _h = json.load(open(os.path.join(FIXDIR, "health.json")))
    _tmp = os.path.join(ROOT, ".webui-notools")
    try:
        os.makedirs(_tmp, exist_ok=True)
        _bad = _copy.deepcopy(_h)
        _bad["supports_tools"] = False
        with open(os.path.join(_tmp, "health.json"), "w") as f:
            json.dump(_bad, f)
        for _n in ("events.json", "stats.json"):
            with open(os.path.join(_tmp, _n), "w") as f:
                f.write(open(os.path.join(FIXDIR, _n)).read())
        r = subprocess.run([node, RUNNER, _tmp], capture_output=True, text=True)
        out3 = json.loads(r.stdout)
        w3 = out3["writes"]
        check("a model that cannot call tools is reported as such, not left blank",
              "cannot call tools" in (w3.get("tool-verdict-b") or ""),
              repr(w3.get("tool-verdict-b")))
        check("...and the page says what to do about it",
              "Serve a model that does" in (w3.get("tool-verdict-s") or ""),
              repr(w3.get("tool-verdict-s")))
        check("...without throwing", not out3["problems"], f"{out3['problems'][:2]}")
        # And the happy path still reads correctly, so the two are genuinely distinguished.
        r2 = subprocess.run([node, RUNNER, FIXDIR], capture_output=True, text=True)
        w2 = json.loads(r2.stdout)["writes"]
        check("a model that CAN call tools reads differently",
              "can call tools" in (w2.get("tool-verdict-b") or "")
              and "cannot" not in (w2.get("tool-verdict-b") or ""),
              repr(w2.get("tool-verdict-b")))
        check("...and the endpoints are filled in with this server's own host",
              (w2.get("ep-responses") or "").endswith("/v1/responses"),
              repr(w2.get("ep-responses")))
    finally:
        import shutil as _sh
        _sh.rmtree(_tmp, ignore_errors=True)
else:
    check("no engine to render the Code page with (skipped, not passed)", True)

print()
print("=" * 84)
print("7. AND THE HARNESS ITSELF MUST BE ABLE TO FAIL")
print("=" * 84)
# A test that passes on a broken page is worse than no test. The page is deliberately damaged in
# the one way that matters -- an id the script reaches for is renamed in the markup, which is the
# exact real-world edit that produces a silently dead page -- and the harness must catch it.
if node and os.path.isdir(FIXDIR):
    broken = os.path.join(ROOT, ".webui-broken.html")
    try:
        with open(broken, "w", encoding="utf-8") as f:
            f.write(html.replace('id="m-model"', 'id="m-model-RENAMED"', 1))
        env = dict(os.environ, BIGRIG_WEBUI=broken)
        r = subprocess.run([node, RUNNER, FIXDIR], capture_output=True, text=True, env=env)
        try:
            out2 = json.loads(r.stdout)
            caught = bool(out2["problems"]) or not out2["writes"].get("m-model")
        except ValueError:
            caught = True
        check("renaming an element the script drives is caught, not passed over", caught,
              "the harness reported a healthy page after the markup was broken")
    finally:
        if os.path.exists(broken):
            os.remove(broken)
else:
    check("no engine to prove the harness can fail (skipped, not passed)", True)

# ---------------------------------------------------------------- how the model is used
# The page hardcoded temperature 0.7 and had no way to give the model standing instructions, so a
# person could not change how it behaved without leaving the page. Both already existed on the
# server; these pin down that the page now exposes them and never stores the instructions where
# the error-recovery path (history.pop) or continue_last could see them.
_page = open(os.path.join(ROOT, "bigrig_engine", "webui.html"), encoding="utf-8").read()
_js = _page[_page.index("<script>"):]
check("the chat page no longer posts a hardcoded temperature", "temperature:0.7," not in _js)
check("...it posts the creativity control's value", "temperature:currentTemp()" in _js)
check("standing instructions are prepended to what is SENT, not stored in the history",
      "messages:withInstructions(history)" in _js and 'history.push({role:"system"' not in _js)
check("...and only when switched on with something written",
      "on && txt" in _js and 'role:"system"' in _js)
check("creativity and instructions are remembered per browser inside try/catch",
      all(f'localStorage.setItem("{k}"' in _js for k in ("tg-temp", "tg-sys", "tg-sys-on"))
      and _js.count("try{") >= _js.count("localStorage.setItem"))
check("a new conversation is refused while a reply is in flight",
      "if(busy) return; history.length=0;" in _js)
check("the empty-state is hidden, not removed, so a new conversation can bring it back",
      '_b.hidden=true' in _js and '$("#blank").hidden=false' in _js
      and '$("#blank")?.remove()' not in _js)
check("the creativity control explains what the number does, in words",
      "0 always picks the likeliest next word" in _page)
check("the instructions control says the text stays in this browser", "Kept in this browser only" in _page)

print("\n" + "=" * 84); print("7b. THE ATTACH-IMAGE CONTROL APPEARS ONLY WHEN THE SERVER CAN SEE"); print("=" * 84)
if node:
    import copy as _cp
    _h = json.load(open(os.path.join(FIXDIR, "health.json")))
    _tmp = os.path.join(ROOT, ".webui-vision")
    try:
        os.makedirs(_tmp, exist_ok=True)
        for _n in ("events.json", "stats.json"):
            with open(os.path.join(_tmp, _n), "w") as f:
                f.write(open(os.path.join(FIXDIR, _n)).read())
        _on = _cp.deepcopy(_h); _on["vision"] = {"tower_gb": 0.893, "max_pixels": 1050000, "max_images": 8}
        with open(os.path.join(_tmp, "health.json"), "w") as f:
            json.dump(_on, f)
        _r = json.loads(subprocess.run([node, RUNNER, _tmp], capture_output=True, text=True).stdout)
        check("with vision on, the attach control is shown", _r["hidden"].get("l-attach") is False, str(_r["hidden"].get("l-attach")))
        check("...without throwing", not _r["problems"], str(_r["problems"][:2]))
        _off = _cp.deepcopy(_h); _off["vision"] = None
        with open(os.path.join(_tmp, "health.json"), "w") as f:
            json.dump(_off, f)
        _r2 = json.loads(subprocess.run([node, RUNNER, _tmp], capture_output=True, text=True).stdout)
        check("with vision off, it is hidden", _r2["hidden"].get("l-attach") is True, str(_r2["hidden"].get("l-attach")))
    finally:
        import shutil as _sh2
        _sh2.rmtree(_tmp, ignore_errors=True)
check("an attached image is sent as an OpenAI image_url part before the text, and stays in the history",
      'image_url:{url:attached.url}' in script and "history.push({role:\"user\",content})" in script)
check("the page reads the image locally and never uploads it anywhere but this server",
      "readAsDataURL" in script and "nothing is uploaded anywhere" in html)

print("\n" + "=" * 84); print("7c. THE THINKING LIMIT: HALF THE REPLY BY DEFAULT, SENT AS thinking_budget"); print("=" * 84)
check("the control exists with 'half the reply' as its first (default) option",
      'id="c-think-limit"' in html and html.index('value="half"') < html.index('value="none"'))
check("it is shown only for a model whose thinking can be toggled, and only when thinking is on",
      "health.can_toggle_thinking" in script and '$("#c-think").checked' in script and "paintThinkLimit" in script)
check("the budget is derived from the reply limit for the fractional choices, never below 64 tokens",
      "Math.floor(lim/2)" in script and "Math.floor(lim/4)" in script and "Math.max(64," in script)
check("...and sent to the server as thinking_budget", "thinking_budget:thinkBudget()" in script)
check("'none' sends no cap, and thinking off sends none", 'if(v==="none") return null;' in script and 'if(!$("#c-think").checked) return null;' in script)
check("when the cap acted, the reply's meta line says so with the count", "thinking capped at ${thinkCut} tokens" in script and "t.thinking_cut" in script)
check("the choice is remembered", '"tg-think-limit"' in script)

print("\n" + "=" * 84); print("7d. WHEN THE METER ENDS A REPLY, THE PAGE SAYS SO, WHY, AND WHAT TO TRY"); print("=" * 84)
check("the page reads the meter's verdict from the final frame", 't.stopped_for==="quality"' in script and "t.remedy" in script)
check("...and the reply's meta line says it was STOPPED by the meter, with the reason and the remedy",
      "STOPPED by the quality meter" in script and "qualityStop.reason" in script and "qualityStop.remedy" in script)
check("...coloured as a problem, since the user did not choose it", "qualityStop){ bot.classList.add(\"flagged\"); meta.className=\"meta warn\"" in script)

print("\n" + "=" * 84); print("8. THE CONTEXT METER READS THE SERVER, NEVER RE-DERIVES THE LIMIT"); print("=" * 84)
check("the page has a Context meter", 'id="m-ctx"' in html and 'id="m-bar-ctx"' in html)
check("...filled from context_used and context_remaining as the server reports them",
      "h.context_used" in script and "h.context_remaining" in script)
check("...against the ceiling the server computed, not one the page re-derives",
      "h.max_completion_tokens" in script)
check("...and it says which limit binds when nothing is used yet", "token_limit_reason" in script)
check("a nearly full context turns the bar the warning colour", '"warn"' in script)

print()
print("=" * 84)
print("ALL TESTS PASSED" if not FAIL else f"{len(FAIL)} FAILURES: " + ", ".join(FAIL))
print("=" * 84)
sys.exit(1 if FAIL else 0)
