/* Run the page's own script against a minimal DOM and real captured server responses.
 *
 * WHY THIS EXISTS. Parsing the script proves it is valid JavaScript; it does not prove it runs.
 * Every interesting failure in this page is a runtime one -- a null element, a field the server
 * stopped sending, a number formatted as undefined -- and all of them are invisible in a browser
 * because they throw inside an async function nobody awaits. So the script is executed here,
 * against the shapes the server actually returns, and anything it throws is a failure.
 *
 * The DOM is a shim, not a browser: it answers the calls this page makes and records what was
 * written, which is enough to assert that the page filled itself in. Where the shim is thinner
 * than a browser it errs toward being STRICTER -- querySelector on an unknown id returns null
 * exactly as a browser does, so the null-dereference bug this is here to catch still happens. */
const fs = require("fs");
const path = require("path");

const ROOT = path.resolve(__dirname, "..", "..");
const FIX = process.argv[2];
// BIGRIG_WEBUI lets the test point this at a deliberately damaged copy, which is how the test
// suite proves this harness is capable of failing at all.
const PAGE = process.env.BIGRIG_WEBUI || path.join(ROOT, "bigrig_engine", "webui.html");
const html = fs.readFileSync(PAGE, "utf8");
const script = html.slice(html.indexOf("<script>") + 8, html.lastIndexOf("</script>"));
const markup = html.slice(0, html.indexOf("<script>"));
const IDS = new Set([...markup.matchAll(/id="([A-Za-z0-9_-]+)"/g)].map(m => m[1]));
const CLASSES = [...markup.matchAll(/class="([^"]+)"/g)].flatMap(m => m[1].split(/\s+/));

const writes = {};                      // id -> last textContent written, for the assertions
const problems = [];

function makeStyle() {
  return new Proxy({}, { get: (t, k) => t[k] ?? "", set: (t, k, v) => (t[k] = v, true) });
}
function El(id, tag) {
  const kids = [];
  const el = {
    id: id || "", tagName: (tag || "div").toUpperCase(),
    _text: "", _html: "", className: "", value: "", disabled: false, hidden: false,
    style: makeStyle(), dataset: {}, children: kids,
    scrollTop: 0, scrollHeight: 100, clientHeight: 100, offsetHeight: 20,
    classList: {
      _s: new Set(),
      add(...c) { c.forEach(x => this._s.add(x)); },
      remove(...c) { c.forEach(x => this._s.delete(x)); },
      toggle(c, on) { on === undefined ? (this._s.has(c) ? this._s.delete(c) : this._s.add(c))
                                       : (on ? this._s.add(c) : this._s.delete(c)); },
      contains(c) { return this._s.has(c); },
    },
    addEventListener() {}, removeEventListener() {}, focus() {}, blur() {}, click() {},
    scrollIntoView() {}, remove() {}, setAttribute() {}, getAttribute() { return null; },
    closest() { return null; },
    querySelector(sel) { return document.querySelector(sel); },
    querySelectorAll() { return []; },
    append(...n) { n.forEach(x => typeof x === "object" && kids.push(x)); },
    appendChild(n) { kids.push(n); return n; },
    removeChild(n) { const i = kids.indexOf(n); if (i >= 0) kids.splice(i, 1); return n; },
    insertBefore(n) { kids.unshift(n); return n; },
  };
  Object.defineProperty(el, "textContent", {
    get() { return el._text; },
    set(v) { el._text = String(v); if (el.id) writes[el.id] = String(v); },
  });
  Object.defineProperty(el, "innerHTML", {
    get() { return el._html; },
    set(v) { el._html = String(v); if (el.id) writes[el.id] = String(v); if (v === "") kids.length = 0; },
  });
  Object.defineProperty(el, "firstElementChild", { get: () => kids[0] || null });
  Object.defineProperty(el, "lastElementChild", { get: () => kids[kids.length - 1] || null });
  Object.defineProperty(el, "childElementCount", { get: () => kids.length });
  return el;
}

const pool = new Map();
function get(id) {
  if (!IDS.has(id)) return null;                 // exactly what a browser does, on purpose
  if (!pool.has(id)) pool.set(id, El(id));
  return pool.get(id);
}
const document = {
  documentElement: El("", "html"), body: El("", "body"),
  createElement: t => El("", t),
  getElementById: get,
  addEventListener() {},
  querySelector(sel) {
    if (sel.startsWith("#")) return get(sel.slice(1));
    const c = sel.replace(/^\./, "").split(/[ .:\[]/)[0];
    return CLASSES.includes(c) ? El("", "div") : null;
  },
  querySelectorAll(sel) {
    const c = sel.replace(/^\./, "").split(/[ .:\[]/)[0];
    return CLASSES.includes(c) ? [El("", "div"), El("", "div")] : [];
  },
};

const FIXTURES = {
  "/health": JSON.parse(fs.readFileSync(path.join(FIX, "health.json"), "utf8")),
  "/stats": JSON.parse(fs.readFileSync(path.join(FIX, "stats.json"), "utf8")),
  "/v1/events": JSON.parse(fs.readFileSync(path.join(FIX, "events.json"), "utf8")),
};
const fetched = [];
async function fetchStub(url, opts) {
  fetched.push(String(url));
  const p = String(url).split("?")[0];
  let body = FIXTURES[p] ?? (p === "/v1/config" ? { ok: true, freed_bytes: 25165824 } : {});
  if (FIXTURES[p] === undefined && p !== "/v1/config") problems.push("unstubbed fetch: " + p);
  // Honour `since` exactly as the server does. A stub that ignored it would hand the same events
  // back on every poll, the page would append them twice, and the duplication -- which is the
  // precise bug the incremental protocol exists to prevent -- would look like correct behaviour.
  if (p === "/v1/events") {
    const m = /since=(\d+)/.exec(String(url));
    const since = m ? Number(m[1]) : 0;
    body = { events: body.events.filter(e => e.seq > since), seq: body.seq };
  }
  return { ok: true, status: 200, json: async () => body, text: async () => JSON.stringify(body),
           body: null, headers: { get: () => null } };
}

const timers = [];
const sandbox = {
  document, fetch: fetchStub, console,
  window: { matchMedia: () => ({ matches: false, addEventListener() {} }),
            addEventListener() {}, location: { host: "127.0.0.1:8099", origin: "http://x" } },
  localStorage: { getItem: () => null, setItem() {}, removeItem() {} },
  location: { host: "127.0.0.1:8099", hostname: "127.0.0.1", port: "8099",
              origin: "http://127.0.0.1:8099", protocol: "http:", href: "http://127.0.0.1:8099/" },
  navigator: { clipboard: { writeText: async () => {} }, userAgent: "node" },
  setInterval: (f, ms) => { timers.push([f, ms]); return timers.length; },
  clearInterval: () => {}, setTimeout: (f) => { return 0; }, clearTimeout: () => {},
  requestAnimationFrame: (f) => 0, AbortController: class { constructor(){ this.signal = {}; } abort(){} },
  TextDecoder: global.TextDecoder, JSON, Math, Date, Number, String, Object, Array, Promise, Error,
  URLSearchParams: global.URLSearchParams, encodeURIComponent, decodeURIComponent, isNaN, parseInt, parseFloat,
};
sandbox.globalThis = sandbox;
sandbox.self = sandbox;

const vm = require("vm");
vm.createContext(sandbox);
process.on("unhandledRejection", e => problems.push("unhandled rejection: " + (e && e.message || e)));

(async () => {
  try {
    vm.runInContext(script, sandbox, { filename: "webui.js" });
  } catch (e) {
    problems.push("threw while loading: " + (e && e.message || e));
  }
  // The page calls refresh() and pollFeed() itself at load; give the promises a chance to settle,
  // then drive every interval once more, which is what a browser does two seconds later.
  await new Promise(r => setImmediate(r));
  await new Promise(r => setImmediate(r));
  for (const [f] of timers) { try { await f(); } catch (e) { problems.push("interval threw: " + (e && e.message || e)); } }
  await new Promise(r => setImmediate(r));

  console.log(JSON.stringify({
    problems, writes, fetched,
    feedRows: (pool.get("feed-body") || { childElementCount: 0 }).childElementCount,
    timers: timers.length,
  }, null, 1));
})();
