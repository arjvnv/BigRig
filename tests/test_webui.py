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
_reserve_terms = re.search(r"self\.serving_reserve_gb\s*=\s*round\((.*?),\s*2\)",
                           _sess, re.S)
check("the server's reserve is written as a sum this test can read", bool(_reserve_terms))
if _reserve_terms:
    _terms = set(re.findall(r"self\.([a-z_]+_gb)", _reserve_terms.group(1)))
    _fit = script[script.index("function slotsThatFit"):script.index("function kvTokensThatFit")]
    _absent = sorted(t for t in _terms if t not in _fit)
    check("every per-session term in that reserve is subtracted by the page's own arithmetic",
          not _absent, f"the page never subtracts: {_absent}")
    check("...and there was more than one term to check", len(_terms) >= 2, f"{sorted(_terms)}")
_kv = script[script.index("function kvTokensThatFit"):]
_kv = _kv[:_kv.index("\n}")]
check("the reply-length slider uses the same terms as the expert slider",
      "prompt_cache_gb" in _kv)
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

print()
print("=" * 84)
print("ALL TESTS PASSED" if not FAIL else f"{len(FAIL)} FAILURES: " + ", ".join(FAIL))
print("=" * 84)
sys.exit(1 if FAIL else 0)
