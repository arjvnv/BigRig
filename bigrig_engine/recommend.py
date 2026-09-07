"""`bigrig doctor` with no model named: what fits this Mac, ranked, by what you want to do.

WHY. `doctor <model>` assumes the user knows what to name. Someone who has just installed and owns
no models has the harder question -- which of the hundreds of MoE checkpoints on the hub is worth
their disk and their afternoon -- and the engine already has every piece of the answer: the same
`remote_shape` and `verdict` the named doctor uses, and the same speed tier. This runs them over
a short curated list and lays the answers side by side.

THE LIST IS CURATED, NOT CRAWLED. Every entry is a Mixture-of-Experts checkpoint from the
mlx-community organisation in a family this engine streams, with a plain note of what it is for.
It is a starting point for someone with no models, not a leaderboard: nothing here ranks models
by quality, because this engine has not measured that, and a list that pretended to would be the
kind of claim this project does not make. What IS ranked is what the engine knows -- whether it
runs on this Mac, at what speed tier, from how much disk.

NOTHING IS DOWNLOADED. Shapes are read from the hub's metadata, the way `doctor <model>` does.
"""
from __future__ import annotations

# (repo, tags, what it is for). Tags are used by `--for`; the note is shown as written.
CURATED = [
    ("mlx-community/Qwen3.6-35B-A3B-4bit", ("chat", "coding", "reasoning", "vision"),
     "general assistant that reasons first and reads images; the model measured most on this engine"),
    ("mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit", ("coding",),
     "tuned for coding agents; the natural pick behind Claude Code or Codex"),
    ("mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit", ("chat", "coding"),
     "general assistant that answers directly, without a thinking block"),
    ("mlx-community/GLM-4.7-Flash-4bit", ("coding", "chat", "reasoning"),
     "agentic coding and chat, reasons first; measured here at 6-7 tok/s at a 9 GB ceiling"),
    ("mlx-community/NVIDIA-Nemotron-3-Nano-30B-A3B-4bit", ("reasoning", "chat"),
     "reasoning-first assistant with a hybrid (recurrent) design"),
    ("mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx", ("coding",),
     "the small coding model: 8.8 GB on disk, fast, no thinking block"),
    ("mlx-community/gpt-oss-20b-MXFP4-Q4", ("reasoning", "chat"),
     "OpenAI's open-weight reasoning model, the smaller of the two"),
    ("mlx-community/Qwen3-Next-80B-A3B-Instruct-4bit", ("chat", "coding"),
     "a large model that still activates only 3B per token: big on disk, quick to decode"),
    ("mlx-community/gpt-oss-120b-MXFP4-Q4", ("reasoning", "chat"),
     "OpenAI's larger open-weight reasoning model; 60+ GB of disk"),
    ("mlx-community/Qwen3.5-122B-A10B-4bit", ("chat", "coding", "reasoning", "vision"),
     "the large Qwen3.5: 10B active per token, so slower to decode than the A3B models"),
    ("mlx-community/Qwen3-235B-A22B-Instruct-2507-4bit", ("chat", "coding"),
     "the largest here: over 120 GB of disk and 22B active per token"),
    ("mlx-community/OLMoE-1B-7B-0125-4bit", ("chat",),
     "small and fully open (AI2); the model this engine's own tests run on. Modest answers"),
]

# The doctor's four words (preflight.TOK_S_TIERS), in the order a user would rank them.
TIER_ORDER = {"FAST": 0, "GOOD": 1, "USABLE": 2, "SLOW": 3}


def rank(budget_gb: float, reserve_gb: float, want: str | None = None, disk_gbs=None,
         progress=None, repos=None) -> list:
    """One row per curated model: the doctor's verdict at this budget, its speed tier, disk.

    Rows are ordered runs-fast, runs, slow, does-not-fit; within a class by download size
    descending -- a rough proxy for how much model you get, said as such. `want` keeps only rows
    tagged with it. `progress(repo, i, n)` is called before each hub read so a caller can show
    that something is happening: twelve reads take a while.
    """
    from .preflight import remote_shape, verdict, speed_tier
    entries = [e for e in (repos or CURATED) if not want or want in e[1]]
    rows = []
    for i, (repo, tags, note) in enumerate(entries):
        if progress:
            progress(repo, i, len(entries))
        row = {"repo": repo, "tags": tags, "note": note, "error": None}
        try:
            sh = remote_shape(repo)
        except Exception as e:                       # noqa: BLE001 -- network, a renamed repo
            row["error"] = str(e).splitlines()[0][:100] if str(e) else type(e).__name__
            rows.append(row)
            continue
        v = verdict(sh, budget_gb, reserve_gb, search=False)
        row.update({"download_gb": sh["download_gb"], "n_experts": sh["n_experts"], "top_k": sh["top_k"],
                    "n_layers": sh["n_layers"], "arch": sh["arch"], "fits": bool(v["fits_now"]),
                    "why_not": v.get("why_not") or ""})
        if v["fits_now"]:
            p = v["plan"]
            tier, why = speed_tier(sh, p, disk_gbs, bool(p.get("fits_entirely")))
            row.update({"tier": tier, "why": why, "capacity": p["capacity"], "residency": p["residency"],
                        "pool_gb": p["pool_gb"], "fits_entirely": bool(p.get("fits_entirely"))})
        rows.append(row)

    def key(r):
        # Classes: 0-3 the four tiers of a model that runs, 4 does not fit, 9 unreadable. Distinct
        # numbers on purpose -- a first version keyed "does not fit" at 3 and a SLOW model that
        # runs sorted after a huge one that does not.
        if r["error"]:
            return (9, 0.0)
        if not r["fits"]:
            return (4, -r["download_gb"])
        return (TIER_ORDER.get(r.get("tier"), 3), -r["download_gb"])
    rows.sort(key=key)
    return rows


def render(rows: list, budget_gb: float, want: str | None = None) -> str:
    """The table, and one line of advice per question a new user asks."""
    from .cli import _human
    out = []
    head = f"  WHAT FITS THIS MAC at a {_human(budget_gb)} budget" + (f", for {want}" if want else "")
    out.append(head)
    out.append("  (verdicts from the hub's metadata; nothing was downloaded. Ranked by whether it runs and how fast,\n"
               "   then by size. Not a quality ranking -- this engine has not measured that.)\n")
    out.append(f"  {'model':<52} {'disk':>8}  {'verdict':<28} for")
    for r in rows:
        name = r["repo"].split("/", 1)[1]
        if r["error"]:
            out.append(f"  {name:<52} {'?':>8}  {'could not read: ' + r['error'][:40]:<28}")
            continue
        if r["fits"]:
            if r.get("fits_entirely"):
                verdict = "RUNS, everything resident"
            else:
                verdict = f"RUNS {r['tier']}, {r['capacity']}/{r['n_experts']} experts held"
        else:
            verdict = "does not fit at this budget"
        out.append(f"  {name:<52} {_human(r['download_gb']):>8}  {verdict:<28} {', '.join(r['tags'])}")
        out.append(f"  {'':<52} {'':>8}  {r['note']}")
    fits = [r for r in rows if not r["error"] and r["fits"]]
    streamed = [r for r in fits if not r.get("fits_entirely")]
    resident = [r for r in fits if r.get("fits_entirely")]
    coding = [r for r in fits if "coding" in r["tags"]]
    out.append("")
    if not fits:
        out.append("  Nothing on this list runs at this budget. Free some memory, or raise the ceiling with BIGRIG_MAX_GB.")
    else:
        # Rows are already ranked: first of each kind is the pick. Said as what it is -- the best
        # tier at this budget among the biggest models -- not as "the best model".
        if streamed:
            b = streamed[0]
            out.append(f"  Best tier, streamed:  {b['repo'].split('/')[1]}  ({b['tier']}, {_human(b['download_gb'])} on disk)")
        if resident:
            out.append(f"  Runs entirely in RAM: {resident[0]['repo'].split('/')[1]}")
        if coding:
            out.append(f"  For coding:           {coding[0]['repo'].split('/')[1]}")
        small = min(fits, key=lambda r: r["download_gb"])
        out.append(f"  Smallest download:    {small['repo'].split('/')[1]} ({_human(small['download_gb'])})")
        out.append(f"\n  Next:  bigrig prepare <model>     downloads it and makes the packed copy the fast path needs")
        out.append(f"         bigrig doctor <model>      the full account for one model, with the smallest ceiling it needs")
    return "\n".join(out)
