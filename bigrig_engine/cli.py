"""bigrig -- run large MoE models on a Mac that does not have room for them.

START HERE
    bigrig doctor <model>         will it run here, and how fast? Nothing is downloaded.
    bigrig run <model>            chat with it in the terminal
    bigrig serve <model>          a local server: browser page, OpenAI and Anthropic APIs
    bigrig launch <model>         run Claude Code or Codex against it, in one command

    <model> is a Hugging Face repo id (downloaded on first use, after doctor says yes) or a
    folder. The first run of a streamed model measures the fastest setting for this Mac,
    once, and remembers it. Skip that with --no-tune.

MORE
    bigrig list                   models already on this machine
    bigrig prepare <model>        download now and make the packed copy the fast path needs
    bigrig compress <model>       shrink it so every expert fits in memory (faster, lossy)
    bigrig knee <model>           re-measure the fastest setting (run does this once for you)
    bigrig calibrate <model>      measure the host round-trip and whole-layer split
    bigrig diff <model>           the same prompt through shipped and compressed weights
"""
from __future__ import annotations


import argparse
import glob
import json
import os
import re
import shutil
import sys
import time

from . import calibrate, home

MODELS_DIR = os.path.expanduser(os.environ.get("BIGRIG_MODELS") or os.path.join(home(), "models"))
BLOBS_DIR = os.path.expanduser(os.environ.get("BIGRIG_BLOBS") or os.path.join(home(), "data", "blobs"))


def _human(gb: float) -> str:
    return f"{gb:.1f} GB"


def measured_disk_gbs() -> float | None:
    """This Mac's measured read rate, or None if it has never been calibrated.

    Shared, so a speed word printed for a local model and one printed for the same model on the
    hub cannot rest on different numbers.
    """
    prof_p = os.path.join(home(), "data", "results", "host_profile.json")
    if not os.path.exists(prof_p):
        return None
    try:
        return float(json.load(open(prof_p)).get("disk_gbs") or 0) or None
    except (OSError, ValueError, TypeError):
        return None


def resolve_model(name: str, allow_download: bool = True,
                  trust_remote_code: bool = False) -> str:
    """A local directory, a name under the models dir, or a Hugging Face repo id."""
    if os.path.isdir(os.path.expanduser(name)):
        return os.path.expanduser(name)
    local = os.path.join(MODELS_DIR, os.path.basename(name))
    if os.path.isdir(local):
        return local
    if "/" not in name:
        raise FileNotFoundError(
            f"no model named {name!r} in {MODELS_DIR}. Give a path, or a Hugging Face repo id "
            f"like 'mlx-community/Qwen3-30B-A3B-4bit'.")
    if not allow_download:
        raise FileNotFoundError(f"{name} is not downloaded and downloading is disabled")
    from huggingface_hub import snapshot_download
    # *.jinja is NOT optional. Newer checkpoints (Qwen3 among them) ship their chat template
    # only as chat_template.jinja, and without it apply_chat_template falls back to a generic
    # "user:/assistant:" format. The model still answers, so nothing looks broken -- the output
    # is just quietly worse, on the very first thing a new user runs.
    patterns = ["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja"]
    if trust_remote_code:
        patterns.append("*.py")
    print(f"  downloading {name} ...")
    path = snapshot_download(repo_id=name, local_dir=local, allow_patterns=patterns)
    # Executable model code is left out unless asked for. Say so rather than failing later with
    # an import error the user cannot connect to a download filter.
    cfg = os.path.join(path, "config.json")
    if not trust_remote_code and os.path.exists(cfg):
        try:
            with open(cfg) as fh:
                c = json.load(fh)
        except (json.JSONDecodeError, OSError):
            c = {}
        if c.get("model_file") or c.get("auto_map"):
            print(f"  NOTE: {name} ships custom Python that runs on your machine. It was not "
                  f"downloaded.\n        Re-run with --trust-remote-code if you want it.")
    return path


def _fmt_plan(plan: dict) -> str:
    from .autoconfig import describe
    return describe(plan)


# ------------------------------------------------------------------------------ commands
def _doctor_remote(repo_id: str, budget_gb: float) -> int:
    """Answer for a model that has not been downloaded. Nothing is fetched but metadata."""
    from .preflight import remote_shape, verdict
    from .session import serving_reserve_gb
    print(f"\n  {repo_id}")
    print("    reading the model's shape from the hub (no weights are downloaded) ...")
    try:
        sh = remote_shape(repo_id)
    except ValueError as e:
        print(f"\n    {e}\n")
        return 1
    except Exception as e:                      # noqa: BLE001 -- network, auth, a bad id
        from .preflight import suggest_repos
        msg = str(e)
        if "safetensors" in msg or "404" in msg or "not found" in msg.lower():
            print(f"\n    there is no model called {repo_id} on the hub.")
        else:
            print(f"\n    could not read {repo_id}: {msg.splitlines()[0][:120]}")
        near = suggest_repos(repo_id)
        if near:
            print(f"    did you mean one of these?")
            for n in near:
                print(f"      bigrig doctor {n}")
        else:
            print(f"    check the spelling, or `huggingface-cli login` if it is a gated repo.")
        print()
        return 1

    prec = "quantised" if sh["quantized"] else "full precision"
    print(f"\n    architecture     {sh['arch']}, {prec} ({sh['dtype']})")
    print(f"    experts          {sh['n_experts']} per layer x {sh['n_layers']} layers, "
          f"top-{sh['top_k']}")
    print(f"    download         {_human(sh['download_gb'])}")
    per = sh["bytes_per_expert"]
    per_s = f"{per / 1e6:.1f} MB" if per < 1e9 else _human(per / 1e9)
    print(f"    of which experts {_human(sh['expert_gb'])}  ({per_s} each, "
          f"{_human(sh['bytes_per_expert'] * sh['n_layers'] * sh['top_k'] / 1e9)} for the "
          f"minimum {sh['top_k']} a layer)")
    print(f"    always resident  {_human(sh['non_expert_gb'])}")

    v = verdict(sh, budget_gb, serving_reserve_gb())
    from .session import MAX_ALLOWED_GB
    print(f"\n    your budget      {_human(budget_gb)}"
          + (f"  (ceiling is {_human(MAX_ALLOWED_GB)}; less is free right now)"
             if budget_gb < MAX_ALLOWED_GB - 0.05 else ""))
    from .preflight import speed_tier
    total = calibrate.total_gb()
    disk_gbs = measured_disk_gbs()
    if v["fits_now"]:
        p = v["plan"]
        tier, why = speed_tier(sh, p, disk_gbs, bool(p.get("fits_entirely")))
        print(f"    VERDICT          RUNS, {tier} -- {p['capacity']} of {sh['n_experts']} experts "
              f"in memory ({p['residency']:.0%}), {_human(p['pool_gb'])} of pool")
        print(f"                     {why}")
        if disk_gbs is None and not p.get("fits_entirely"):
            print(f"                     (disk assumed at 3 GB/s; `bigrig doctor --calibrate` "
                  f"measures it)")
        if tier == "SLOW":
            # The one case where the right advice is often a different download, said plainly.
            print(f"                     a 4-bit build of the same model would roughly halve the "
                  f"bytes each token reads; check for one before downloading this")
        from .preflight import ceiling_hint
        if budget_gb < MAX_ALLOWED_GB - 0.05:
            # The budget is what is FREE right now, not the ceiling. Telling this person to raise
            # the ceiling would be wrong twice: it is not what is limiting them, and the same
            # command with more memory free would already plan larger.
            print(f"                     memory is busy right now ({_human(budget_gb)} free of a "
                  f"{_human(MAX_ALLOWED_GB)} ceiling); with more free, the plan holds more experts")
        else:
            hint = ceiling_hint(budget_gb, total, tier, repo_id, shape=sh,
                                reserve_gb=serving_reserve_gb(),
                                disk_gbs=disk_gbs)
            for line in hint.splitlines():
                print(f"                     {line}")
        print(f"\n    to run it:  bigrig run {repo_id}")
    else:
        need = v["needs_gb"]
        if need and need <= total:
            # IT FITS THE MACHINE, JUST NOT THE SAFE DEFAULT. A decision, and said so.
            pn = v["plan_at_needs"]
            tier, why = speed_tier(sh, pn, disk_gbs)
            print(f"    VERDICT          NOT AT THIS CEILING -- your budget is {_human(budget_gb)}, "
                  f"it needs {_human(need)}")
            print(f"                     at {_human(need)} it would hold {pn['capacity']} of "
                  f"{sh['n_experts']} experts ({pn['residency']:.0%}) and be {tier}: {why}")
            print(f"\n    this Mac has {total:.1f} GB, so that is a choice, not a wall. To make it:")
            print(f"      BIGRIG_MAX_GB={need} bigrig run {repo_id}")
            print(f"    Leave less for everything else while it runs.")
        elif need:
            # IT DOES NOT FIT THE MACHINE. This used to print "a decision rather than an
            # impossibility" for a 657 GB model needing 41 GB on a 25.8 GB Mac. It is not.
            print(f"    VERDICT          IMPOSSIBLE ON THIS MAC")
            print(f"                     even at its lowest workable setting it needs "
                  f"{_human(need)} of memory, and this Mac has {total:.1f} GB in total.")
            print(f"                     No flag changes that. It needs a machine with more memory.")
        else:
            print(f"    VERDICT          IMPOSSIBLE ON THIS MAC")
            print(f"                     one layer's worth of experts is larger than any budget "
                  f"this machine could offer.")
        print(f"\n    NOTHING HAS BEEN DOWNLOADED. This would have cost "
              f"{_human(sh['download_gb'])} of disk.")
    print()
    return 0 if v["fits_now"] else 2


def cmd_doctor(a) -> int:
    from .calibrate import available_gb, under_pressure
    from .session import resolve_budget, serving_reserve_gb
    print("\n  MACHINE")
    avail = available_gb()
    budget = resolve_budget(getattr(a, "memory", None), quiet=True)
    print(f"    memory available now      {_human(avail)}")
    if abs(budget - avail) > 0.05:
        if getattr(a, "memory", None):
            src = "--memory"
        elif os.environ.get("BIGRIG_MEM_GB"):
            src = "BIGRIG_MEM_GB"
        else:
            from .session import SAFE_SHARE_OF_RAM
            src = (f"the safe default: {SAFE_SHARE_OF_RAM:.0%} of installed memory, so the rest "
                   f"of your Mac keeps working. Raise it with BIGRIG_MAX_GB")
        print(f"    budget for this run       {_human(budget)}  ({src})")
    print(f"    memory under pressure     {'YES -- close something first' if under_pressure() else 'no'}")
    prof_p = os.path.join(home(), "data", "results", "host_profile.json")
    if os.path.exists(prof_p):
        pr = json.load(open(prof_p))
        print(f"    RAM bandwidth (measured)  {pr['ram_gbs']:.0f} GB/s")
        print(f"    disk bandwidth (measured) {pr['disk_gbs']:.2f} GB/s")
        print(f"    disk vs memory            {pr['kappa']:.0f}x slower  -- what an expert "
              f"read from disk costs against one already in memory")
    else:
        print("    run `bigrig doctor --calibrate` to measure this machine's bandwidths")
    if a.calibrate:
        from .calibrate import calibrate
        print("\n  measuring ...")
        pr = calibrate(save=prof_p)
        print(f"    RAM {pr['ram_gbs']:.0f} GB/s, disk {pr['disk_gbs']:.2f} GB/s, "
              f"kappa {pr['kappa']:.1f}")

    # A MODEL THAT IS NOT HERE YET CAN STILL BE ANSWERED FOR, AND THAT IS THE WHOLE POINT.
    #     This command could always say whether a model would run. It just said it after the
    #     weights were on disk -- 58 GB downloaded, then "it needs an 11.5 GB ceiling and yours
    #     is 9.0". Every term in that sentence is readable from the hub in about two seconds.
    want = getattr(a, "model", None)
    if want and "/" in want and not os.path.isdir(os.path.expanduser(want)) \
            and not any(r["name"] == want or r["name"].startswith(want) for r in _prepared()):
        return _doctor_remote(want, budget)

    print("\n  PREPARED MODELS")
    rows = _prepared()
    want = getattr(a, "model", None)
    if want:
        rows = [r for r in rows if r["name"] == want or r["name"].startswith(want)]
        if not rows:
            print(f"    no model here matching {want!r} -- `bigrig list` shows what there is")
            print()
            return 1
    if not rows:
        print("    none yet -- `bigrig prepare <model>` to add one")
    for r in rows:
        try:
            from .autoconfig import choose_strategy, describe_strategy
            from .stream import model_top_k
            from .precision import non_expert_gb
            md = os.path.join(MODELS_DIR, r["name"])
            # The manifest is passed explicitly. Without it non_expert_gb falls back to reading
            # one from the blob path, which is "" for a model that was never packed -- and the
            # resulting `.manifest.json not found -- run pack_experts()` went to stderr, naming a
            # Python function no user has ever called, in the middle of an unrelated report.
            # The same budget and the same reserve `serve` will use, so the two cannot report
            # different plans for the same machine -- which was NOT true until both sides were
            # made to call `serving_reserve_gb`; this path was short by the prompt cache.
            ne = non_expert_gb(md, manifest=r["manifest"]) if os.path.isdir(md) else 0.0
            st = choose_strategy(r["manifest"], budget_gb=budget,
                                 top_k=model_top_k(md, r["manifest"]),
                                 reserve_gb=serving_reserve_gb(),
                                 non_expert_gb=ne)
            verdict = describe_strategy(st, measured_disk_gbs())
        except MemoryError as e:
            verdict = f"WILL NOT RUN: {e}"
        except (OSError, ValueError, KeyError) as e:
            verdict = f"could not be read: {e}"
        print(f"    {r['name'][:40]:<42} {_human(r['gb']):>9}   {verdict}")
        if r["variants"]:
            print(f"    {'':<42} {'':>9}   already compressed: {', '.join(r['variants'])}")
    print()
    return 0


def _prepared() -> list:
    """Every model on this machine, with whatever extras exist beside it.

    Enumerates MODELS, not blobs. Listing blobs made a model with no packed copy look like it was
    not installed, which stopped being true the moment packing became optional -- experts are read
    from the model's own safetensors unless a blob happens to be there.
    """
    out = []
    variants, packed = {}, {}
    if os.path.isdir(BLOBS_DIR):
        for f in sorted(os.listdir(BLOBS_DIR)):
            m = re.match(r"^(.*)\.experts\.q(\d+)g(\d+)\.manifest\.json$", f)
            if m:
                variants.setdefault(m.group(1), []).append(f"{m.group(2)}-bit g{m.group(3)}")
                continue
            m = re.match(r"^(.*)\.experts\.manifest\.json$", f)
            if m:
                packed[m.group(1)] = os.path.join(BLOBS_DIR, m.group(1) + ".experts")
    if not os.path.isdir(MODELS_DIR):
        return out
    for name in sorted(os.listdir(MODELS_DIR)):
        d = os.path.join(MODELS_DIR, name)
        if not os.path.isdir(d) or not glob.glob(os.path.join(d, "*.safetensors")):
            continue
        try:
            from .direct import expert_manifest
            man = expert_manifest(d)
        except (ValueError, FileNotFoundError, KeyError, json.JSONDecodeError):
            continue                          # not an MoE, or unreadable; nothing to say about it
        blob = packed.get(name, "")
        if blob and not os.path.exists(blob):
            blob = ""
        out.append({"name": name, "dir": d, "blob": blob, "manifest": man,
                    "gb": man["total_bytes"] / 1e9, "complete": True,
                    "packed": bool(blob), "variants": sorted(variants.get(name, []))})
    return out


def cmd_list(a) -> int:
    rows = _prepared()
    if not rows:
        print("  no models prepared yet")
        return 0
    print(f"\n  {'model':<44} {'experts':>10} {'extras':>26}")
    for r in rows:
        extras = []
        if r["packed"]:
            extras.append("packed (zero-copy path)")
        else:
            extras.append("not packed: experts are copied in; `bigrig prepare` fixes that")
        if r["variants"]:
            extras.append("compressed: " + ", ".join(r["variants"]))
        print(f"  {r['name'][:44]:<44} {_human(r['gb']):>10} "
              f"{(extras[0] if extras else 'ready'):>26}")
        for e in extras[1:]:
            print(f"  {'':<44} {'':>10} {e:>26}")
    print()
    return 0


def _pack_room(total_bytes: int) -> tuple:
    """(enough room, free bytes) for a packed copy, with a gigabyte kept back for everything else."""
    where = BLOBS_DIR if os.path.isdir(BLOBS_DIR) else os.path.dirname(BLOBS_DIR)
    free = shutil.disk_usage(where).free
    return total_bytes + (1 << 30) <= free, free


def cmd_prepare(a) -> int:
    """Make a model ready to run: download it if needed, then make the packed copy.

    PACKING IS THE DEFAULT AGAIN, FOR A DIFFERENT REASON THAN THE FIRST TIME.
        It was made optional when it bought ~1.2x on the read path for double the disk. It now
        buys the zero-copy admit: the GPU reads an expert straight out of the page cache instead
        of the CPU copying it in, and that needs every expert page-aligned -- which the model's
        own shards never are (0 of 360 expert tensors in Qwen3.6-35B-A3B-4bit start on a page).
        Measured on that model, same plan, same Mac: {PACK_FROM} -> {PACK_TO} tok/s with a warm
        page cache, and only {PACK_COLD_RATIO}x when the disk is the bottleneck -- both paths wait
        on the same reads then. `--no-pack` keeps the disk and the copy path.
    """
    from . import stream
    from .preflight import PACK_FROM, PACK_TO, PACK_COLD_RATIO
    path = resolve_model(a.model, trust_remote_code=getattr(a, "trust_remote_code", False))
    man, blob = stream.expert_source(path)
    print(f"  {os.path.basename(path)}: {len(man['layers'])} MoE layers, "
          f"{man['layers'][sorted(man['layers'], key=int)[0]]['n_experts']} experts, "
          f"{_human(man['total_bytes'] / 1e9)} of expert weights")
    if blob:
        print("  packed -- the GPU reads experts in place (the zero-copy path)")
    elif getattr(a, "no_pack", False):
        print("  not packed, as asked. Experts will be copied in from the model's own files; "
              f"packing later\n  is `bigrig prepare {os.path.basename(path)}` "
              f"(measured {PACK_FROM} -> {PACK_TO} tok/s on Qwen3.6-35B-A3B-4bit with a warm "
              f"page cache, {PACK_COLD_RATIO}x when the disk is the bottleneck)")
    else:
        ok, free = _pack_room(man["total_bytes"])
        if not ok:
            print(f"  NOT PACKED: the copy needs {_human(man['total_bytes']/1e9)} of free disk "
                  f"and {_human(free/1e9)} is available.\n  It will run, with every expert "
                  f"copied in from the model's own files -- the slower path.")
            return 1
        print(f"  making a contiguous, page-aligned copy of the experts "
              f"(+{_human(man['total_bytes']/1e9)} on disk).\n  This is the copy the GPU can "
              f"read without the CPU copying it: measured {PACK_FROM} -> {PACK_TO} tok/s "
              f"on\n  Qwen3.6-35B-A3B-4bit at the same plan with a warm page cache "
              f"({PACK_COLD_RATIO}x when the disk\n  is the bottleneck). --no-pack skips it.")
        t0 = time.perf_counter()
        stream.ensure_packed(path, os.path.join(BLOBS_DIR,
                                                os.path.basename(path) + ".experts"),
                             verbose=a.verbose)
        print(f"  packed in {time.perf_counter()-t0:.0f}s")
    from .autoconfig import choose_strategy, describe_strategy
    from .precision import non_expert_gb
    try:
        st = choose_strategy(man, top_k=stream.model_top_k(path, man),
                             non_expert_gb=non_expert_gb(path, manifest=man))
        print(f"  {describe_strategy(st)}")
    except MemoryError as e:
        print(f"  WARNING: {e}")
    return 0


def cmd_compress(a) -> int:
    """Requantise a model's experts so all of them fit in RAM at once."""
    from . import precision, stream
    from .autoconfig import choose_strategy, describe_strategy
    path = resolve_model(a.model, trust_remote_code=getattr(a, "trust_remote_code", False))
    man, blob = stream.expert_source(path)
    tk = stream.model_top_k(path, man)
    ne = precision.non_expert_gb(path, manifest=man)

    if a.bits:
        bits, group = a.bits, a.group
    else:
        st = choose_strategy(man, budget_gb=a.memory, top_k=tk, non_expert_gb=ne)
        if st["mode"] != "compress":
            print(f"  {describe_strategy(st)}")
            if st["mode"] == "native":
                print("  Nothing to do -- compressing a model that already fits only makes it "
                      "worse.")
                return 0
            print("  Compressing will not be enough on its own; it would still stream. "
                  "Trading accuracy for nothing is not worth doing.")
            return 1
        bits, group = st["bits"], st["group_size"]

    if not a.tune:
        dst = precision.ensure_compressed(blob, bits, group, verbose=True, manifest=man,
                                          name=os.path.basename(path))
        m = stream.load_manifest(dst)
        print(f"  ready: {_human(m['total_bytes'] / 1e9)} at {bits}-bit g{group} "
              f"(from {_human(man['total_bytes'] / 1e9)})")
        print(f"  `rig run {a.model}` will now use it and keep every expert in RAM.")
        return 0
    return _compress_tuned(a, path, man, blob, bits, group, tk, ne)


def _compress_tuned(a, path, man, blob, bits, group, top_k, non_expert) -> int:
    """Measure which layers can afford to be compressed, then spend the budget accordingly."""
    from . import evaluate, precision, stream, tune
    key = blob or os.path.join(BLOBS_DIR, os.path.basename(path) + ".experts")
    prof = None if a.retune else tune.load_profile(key)

    if prof is None:
        # The scan rewrites one layer's resident experts at a time, so nothing may be evicted
        # underneath it. That means every expert has to be in memory at the SOURCE precision --
        # if it is not, an evicted expert is re-read from disk mid-measurement at its original
        # precision and the layer under test silently un-degrades.
        # `--memory` is the budget the ENGINE may use at run time; it is what makes the model
        # need compressing in the first place. The measurement is a one-off that can use the
        # whole machine, and applying the run-time budget to it made --tune refuse exactly the
        # models it exists for: tight enough to need compressing means too tight to measure.
        need = man["total_bytes"] / 1e9 + non_expert
        from .autoconfig import available_gb, RESERVE_GB, MIN_HEADROOM_GB
        have = (a.tune_memory if a.tune_memory is not None
                else available_gb()) - RESERVE_GB - MIN_HEADROOM_GB
        if need > have:
            print(f"  --tune needs every expert in memory at the model's own precision while "
                  f"it measures: {_human(need)} against {_human(max(have, 0))} free right now.")
            print(f"  This is the MEASUREMENT's requirement, not the plan's -- the plan will "
                  f"still target {_human(0)} " .replace(_human(0), "your --memory budget."))
            print(f"  Free some memory and retry, or compress without --tune. A plan measured "
                  f"on a partly-resident model would be measuring the cache, not the layers.")
            return 1
        print(f"  measuring per-layer sensitivity ({len(man['layers'])} layers, one evaluation "
              f"pass each) ...")
        print(f"  this happens once and is cached; it is why --tune is not the default.")
        t0 = time.perf_counter()
        E = man["layers"][sorted(man["layers"], key=int)[0]]["n_experts"]
        model, tok, h = stream.load_streaming(path, blob, capacity=E, manifest=man,
                                              warm="full", verbose=False)
        try:
            ids = evaluate.tokenize_once(tok, {"wiki": a.corpus}, 120_000)["wiki"]
            prof = tune.scan(model, h, ids, evaluate, window=a.window, windows=a.windows,
                             log=lambda m: print(m, flush=True))
        finally:
            h.close()
            del model, h, tok
            import gc
            import mlx.core as mx
            gc.collect()
            mx.clear_cache()
        prof["model"] = os.path.basename(path)
        prof["windows"] = a.windows
        prof["seconds"] = round(time.perf_counter() - t0, 1)
        print(f"  measured in {prof['seconds']:.0f}s -> {tune.save_profile(key, prof)}")
    else:
        print(f"  using the cached sensitivity profile ({len(prof['sensitivity'])} layers; "
              f"--retune to measure again)")

    sv = sorted((float(v), int(k)) for k, v in prof["sensitivity"].items())
    print(f"  least sensitive: " + ", ".join(f"L{l} {v:+.4f}" for v, l in sv[:3]))
    print(f"  most  sensitive: " + ", ".join(f"L{l} {v:+.4f}" for v, l in sv[-3:]))
    # A layer that measures at or below zero did not get worse when it was degraded, which is
    # not possible -- it means the effect is smaller than this measurement can see. Reporting a
    # ratio against it produced "spread 145449161.5x" once; the number is meaningless and the
    # honest thing is to say the measurement is thin rather than dress it up.
    noise = [l for v, l in sv if v <= 0]
    mid = sv[len(sv) // 2][0]
    if mid > 0:
        print(f"  spread {sv[-1][0] / mid:.1f}x against the median layer")
    if noise:
        print(f"  NOTE: {len(noise)} layer(s) measured at or below zero "
              f"({', '.join('L' + str(l) for l in noise[:5])}"
              f"{'...' if len(noise) > 5 else ''}) -- the effect there is smaller than "
              f"{prof.get('windows', '?')} window(s) of text can resolve.")
        print(f"        The plan still fits, but --windows 8 or more would separate those "
              f"layers instead of ranking them by noise.")

    budget = man["total_bytes"] * (precision.bytes_per_param(bits, group)
                                   / precision.bytes_per_param(
                                       man["layers"][sorted(man["layers"], key=int)[0]]
                                       ["quant"]["bits"],
                                       man["layers"][sorted(man["layers"], key=int)[0]]
                                       ["quant"]["group_size"]))
    # A tuned plan is floored at 3 bits, so it cannot reach a budget that only 2-bit would fit.
    # Crashing there with an allocator traceback tells the user nothing they can act on.
    try:
        plan = tune.plan_from_profile(prof, man, budget, min_bits=max(3, a.min_bits or 3))
    except ValueError as e:
        print(f"  {e}")
        print(f"  --tune will not allocate below 3 bits, and this budget needs {bits}-bit to "
              f"fit. The measurement is cached, so nothing was wasted:")
        print(f"    rig compress {a.model} --memory <more>   to fit with a tuned plan, or")
        print(f"    rig compress {a.model}                   to take the uniform {bits}-bit copy")
        return 1
    print(f"  plan at {_human(budget / 1e9)}: {tune.describe_plan(plan, man)}")
    dst = precision.compressed_path(
        blob or os.path.join(BLOBS_DIR, os.path.basename(path) + ".experts"), bits, group)
    dst = dst + ".tuned"
    if os.path.exists(dst) and os.path.exists(dst + ".manifest.json"):
        print("  a tuned copy already exists; delete it to rebuild")
    else:
        t0 = time.perf_counter()
        m = precision.requantize_blob(blob, dst, plan, progress=False, manifest=man)
        print(f"  wrote {_human(m['total_bytes'] / 1e9)} in {time.perf_counter() - t0:.0f}s")
    print(f"  ready. Measured on OLMoE-1B-7B this was worth 3.9% better perplexity than "
          f"uniform at the same size.")
    return 0


def _session(a):
    from .session import Session
    path = resolve_model(a.model, trust_remote_code=getattr(a, "trust_remote_code", False))
    cap = None
    if getattr(a, "residency", None):
        cap = a.residency if a.residency <= 1.0 else int(a.residency)
    if getattr(a, "exact", False) and getattr(a, "compress", False):
        raise ValueError("--exact and --compress contradict each other; pick one")
    pref = ("exact" if getattr(a, "exact", False)
            else "compress" if getattr(a, "compress", False) else None)
    draft = getattr(a, "draft", None)
    if draft:
        draft = resolve_model(draft)
    if getattr(a, "file_pool", False):
        from . import stream as _stm
        _stm.VIEWS_PREFILL = True
        _stm.VIEWS_DECODE = True
    mtp = getattr(a, "mtp", None)
    if mtp:
        from . import mtp as _mtp
        mtp = _mtp.head_path(path) if mtp == "auto" else os.path.expanduser(mtp)
        if not os.path.exists(os.path.join(mtp, "model.safetensors")):
            raise FileNotFoundError(
                f"no MTP head at {mtp}. mlx-community publishes one for Qwen3.5/3.6 models as "
                f"<model>-MTP-bf16; download it beside the model, or pass --mtp <path>.")
    if getattr(a, "forget_choice", False):
        from . import stream as _st, consent as _cs
        b = os.path.join(BLOBS_DIR, os.path.basename(path) + ".experts")
        print("  forgot the remembered choice" if _cs.forget_choice(b)
              else "  no remembered choice to forget")
    # The first session is built quietly about its PLAN: a first run may pack and tune and build
    # again, and only the session actually served should announce what it holds. The rebuilds
    # announce; if nothing was rebuilt, the plan is printed here.
    s0 = Session(path, capacity=cap, threads=a.threads,
                monitor=not getattr(a, "no_monitor", False),
                budget_gb=getattr(a, "memory", None), kv_bits=getattr(a, "kv_bits", None),
                kv_quant_start=getattr(a, "kv_quant_start", None), verbose=True,
                force_stream=getattr(a, "force_stream", False),
                min_bits=getattr(a, "min_bits", None),
                preference=pref, interactive=True,
                draft=draft, draft_tokens=getattr(a, "draft_tokens", 3),
                mtp=mtp, mtp_bits=(getattr(a, "mtp_bits", 4) or None),
                prefetch_width=getattr(a, "prefetch", 0),
                reroute=getattr(a, "reroute", 0.0),
                no_full_layers=getattr(a, "no_full_layers", False), announce=False)
    s = _auto_tune(a, _auto_pack(a, s0))
    if not s.init_kwargs.get("announce", True) and s.plan_lines:
        print(s.plan_summary(), flush=True)      # a session built quietly is the one being served
    return s


def _auto_pack(a, s):
    """Make the packed copy on the first run of a streamed model, before anything is measured.

    The knee is measured on whatever path the model runs on. Tuned on the copy path and packed
    afterwards, the remembered capacity is a number about a different engine; so the copy comes
    first, and the tune that follows measures the path the model will actually take. Skipped
    when the model is not streamed (a resident model gains nothing), when the disk has no room
    (said so; the copy path is used), or on --no-pack. See `cmd_prepare` for the measurement.
    """
    if not getattr(s, "streamed", False) or getattr(s, "packed", False):
        return s
    if getattr(a, "no_pack", False):
        return s
    from . import stream as _st
    from .preflight import PACK_FROM, PACK_TO, PACK_COLD_RATIO
    from .session import Session as _S
    path = s.init_kwargs["model_dir"]
    try:
        man, _ = _st.expert_source(path)
    except Exception:                            # noqa: BLE001 -- packing must never block a run
        return s
    ok, free = _pack_room(int(man["total_bytes"]))
    if not ok:
        print(f"\n  NOT PACKING {s.name}: the fast path needs a {_human(man['total_bytes']/1e9)} "
              f"copy of its experts and {_human(free/1e9)} of disk is free.\n  Running with every "
              f"expert copied in from the model's own files, the slower path.\n", flush=True)
        return s
    print(f"\n  first run of {s.name}: making a contiguous, page-aligned copy of its experts "
          f"(+{_human(man['total_bytes']/1e9)} on disk).\n  This is the copy the GPU can read "
          f"without the CPU copying it -- measured {PACK_FROM} -> {PACK_TO} tok/s on\n  "
          f"Qwen3.6-35B-A3B-4bit at the same plan with a warm page cache ({PACK_COLD_RATIO}x when "
          f"the disk\n  is the bottleneck). One-time; --no-pack skips it.\n", flush=True)
    rebuild = dict(s.init_kwargs)
    rebuild["budget_gb"] = s.budget_gb        # the ceiling this run resolved, not "what is free"
    rebuild["announce"] = False               # the tune may still rebuild; the last one speaks
    s.close()
    del s
    try:
        _st.ensure_packed(path, os.path.join(BLOBS_DIR, os.path.basename(path) + ".experts"),
                          verbose=True)
    except Exception as e:                       # noqa: BLE001 -- a failed pack is a slower run
        print(f"  could not pack ({str(e)[:80]}); running on the copy path.\n", flush=True)
    return _S(**rebuild)


def cmd_run(a) -> int:
    s = _session(a)
    st = s.stats()
    print(f"\n  {st['model']} ready in {st['load_seconds']}s", flush=True)
    print(f"  {st['serving']}", flush=True)      # the disclosure must never sit in a buffer
    if st.get("streamed") and st.get("resident_gb"):
        print(f"  {st['resident_gb']:.1f} GB of expert weights held in memory", flush=True)
    print(f"  quality monitor {'on' if st['monitor'] else 'off'}. Ctrl-C to stop, "
          f"'/stats' for numbers, '/quit' to exit.\n")
    history = []
    try:
        while True:
            try:
                q = input("  you > ").strip()
            except EOFError:
                break
            if not q:
                continue
            if q in ("/quit", "/exit"):
                break
            if q == "/stats":
                for k, v in s.stats().items():
                    print(f"    {k:<16} {v}")
                continue
            history.append({"role": "user", "content": q})
            print("  bot > ", end="", flush=True)
            parts, flagged, t0 = [], 0, time.perf_counter()
            run, longest = 0, 0
            for c, info in s.stream_text(history, max_tokens=a.max_tokens,
                                         temperature=a.temperature):
                print(c, end="", flush=True)
                parts.append(c)
                d = bool(info.get("degraded"))
                flagged += int(d)
                run = run + 1 if d else 0
                longest = max(longest, run)
            dt = time.perf_counter() - t0
            n = len(parts)
            history.append({"role": "assistant", "content": "".join(parts)})
            note = ""
            # A lone flag is noise (about one healthy token in a hundred trips the meter); a run
            # of them, or a real share, is what damage looks like. Say which.
            from .session import FLAG_RUN, FLAG_NOISE_SHARE
            if flagged and (longest >= FLAG_RUN or flagged / max(n, 1) >= FLAG_NOISE_SHARE):
                note = f", QUALITY WARNING on {flagged}/{n} tokens"
            elif flagged:
                note = f", {flagged} unusual token{'s' if flagged > 1 else ''} (within noise)"
            print(f"\n       [{n} tokens, {n/dt if dt else 0:.1f} tok/s"
                  f"{', miss ' + format(s.stats().get('miss_rate', 0)*100, '.1f') + '%' if st.get('streamed') else ''}"
                  f"{note}]\n")
    except KeyboardInterrupt:
        print()
    finally:
        s.close()
    return 0


def cmd_launch(a) -> int:
    """Start a server and run a coding agent against it, in one command."""
    from . import launch
    path = resolve_model(a.model, trust_remote_code=getattr(a, "trust_remote_code", False))
    # Re-invoke ourselves for the server so it inherits this interpreter and this venv, rather
    # than whatever `bigrig` happens to resolve to on the agent's PATH.
    argv = [sys.executable, "-m", "bigrig_engine.cli", "serve", a.model]
    for flag, val in (("--memory", a.memory), ("--residency", a.residency),
                      ("--threads", a.threads), ("--min-bits", a.min_bits),
                      ("--kv-bits", getattr(a, "kv_bits", None)),
                      ("--kv-quant-start", getattr(a, "kv_quant_start", None))):
        if val is not None:
            argv += [flag, str(val)]
    # Every switch `common()` offers has to reach the server, or a flag typed on `launch` is
    # silently ignored. --no-tune was: the first run tuned anyway.
    for flag in ("exact", "compress", "no_monitor", "force_stream", "no_tune", "no_full_layers",
                 "trust_remote_code"):
        if getattr(a, flag, False):
            argv.append("--" + flag.replace("_", "-"))
    try:
        return launch.run(a.agent, os.path.basename(path), argv, port=a.port,
                          agent_args=a.agent_args)
    except (FileNotFoundError, RuntimeError, ValueError) as e:
        print(f"\n  {e}\n", file=sys.stderr)
        return 1


def _verify_split(a, plan, n_layers, n_experts, uniform_cap, tokens):
    """Run uniform and the proposed split, and return measured uniform_ms / split_ms."""
    import time as _t
    from .session import Session as _S

    def timed(cap, full):
        s = _S(resolve_model(a.model), capacity=cap, preference="stream", interactive=False,
               verbose=False, budget_gb=getattr(a, "memory", None), kv_bits=getattr(a, "kv_bits", None), kv_quant_start=getattr(a, "kv_quant_start", None), full_layers=full)
        msgs = [{"role": "user", "content": "Count to twenty."}]
        for _p, _i in s.stream_text(msgs, max_tokens=16, think=False):
            pass
        t0, first, produced = None, 0, 0
        for _p, info in s.stream_text(msgs, max_tokens=tokens, think=False):
            produced = info.get("generation_tokens") or produced
            if t0 is None and produced:
                t0, first = _t.perf_counter(), produced
        steps = max(1, produced - first)
        ms = ((_t.perf_counter() - t0) * 1000.0 / steps) if t0 else float("inf")
        s.close() if hasattr(s, "close") else None
        del s
        return ms

    u = timed(uniform_cap, ())
    v = timed(plan["capacity"], tuple(range(plan["full_layers"])))
    print(f"    uniform {uniform_cap}/{n_experts}: {u:.2f} ms/token   "
          f"{plan['full_layers']} whole + {plan['capacity']}/{n_experts}: {v:.2f} ms/token")
    return u / v if v else 0.0


TRAIN_PROMPTS = [
    "Explain what a hash table is and how collisions are handled.",
    "Write a short paragraph about how a heat pump moves heat.",
    "List the steps to make a cup of tea, then explain why the order matters.",
    "What is the difference between a stack and a queue? Give an example of each.",
    "Describe how a suspension bridge carries load down to its towers.",
    "Write a Python function that merges two sorted lists, and explain it.",
]


def cmd_predict(a) -> int:
    """Fit the predictor that lets this model's experts be read a layer early."""
    from . import predict as _p
    s = _session(a)
    if not getattr(s, "streamed", False):
        print(f"\n  {s.name} runs entirely in memory here, so nothing is ever read from disk "
              f"and\n  there is nothing to prefetch.\n")
        return 0
    print(f"\n  fitting a predictor for {s.name} -- {len(s.handle.mods)} layers, "
          f"{a.tokens} tokens per prompt\n")
    got = _p.train(s, TRAIN_PROMPTS[:a.prompts], tokens=a.tokens)
    m = got["meta"]
    print(f"\n  fitted on {m['steps']} decode steps, held out {m['holdout_steps']}")
    print(f"  recall on the held-out steps -- how much of the NEXT layer's routing it names:")
    print(f"    naming  8 experts: {m['recall_at_8']:.1%}")
    print(f"    naming 16 experts: {m['recall_at_16']:.1%}")
    print(f"    naming 32 experts: {m['recall_at_32']:.1%}")
    if m["recall_at_8"] < 0.25:
        print("\n  that is too low to be worth the bandwidth it would spend. Not saving it.")
        return 1
    path = _p.save(s.name, got["W"], m)
    print(f"\n  saved to {path}")
    print("  `bigrig serve` and `bigrig run` will use it. A wrong prediction only wastes a\n"
          "  read -- the model's own router still decides what runs, so answers are unchanged.\n")
    return 0


def cmd_diff(a) -> int:
    """Run one prompt through this model as shipped and compressed, and show where they part.

    WHY THIS COMMAND EXISTS, AND WHY IT IS NOT THE QUALITY METER
        The engine ships an anomaly meter, and it is measured to be a coin flip on the damage
        compression causes: AUROC 0.500 at 3 bits and 0.542 at 2, against healthy output. Six
        different statistics of the token stream were tried -- free energy, entropy, repetition,
        top-1 probability, peak logit, logsumexp -- and at 3 bits, which is what this product
        ships, every one of them was chance. At 2 bits some of them separate, which only says
        that visibly broken output is visibly broken.

        That is not a tuning failure, it is a definition. "Damage" means "different from the
        model you would otherwise have run", and no statistic of a single token stream can see a
        difference from something it has never seen. The meter has no copy of the original.

        This command supplies the missing half. It runs the same prompt twice against the same
        weights, once as they ship and once carrying the error of a lower precision, and reports
        the first token where the two disagree. Nobody else is positioned to do this cleanly:
        being both the compressor and the runtime is what makes the comparison exact rather than
        approximate.
    """
    import mlx.core as mx
    from . import tune as _tune
    from .session import Session as _S
    path = resolve_model(a.model)
    prompt = a.prompt or "Explain in three short sentences why the sky is blue."

    # STREAMED EVEN IF IT WOULD FIT. The comparison is done by round-tripping resident experts
    # through a lower precision, which needs the pools that streaming builds -- and a model small
    # enough to load whole has none. Forcing it changes nothing about the arithmetic: streaming is
    # bit-exact, verified on this model at 0.000e+00 on both corpora, so the "as shipped" side of
    # the comparison is the same text either way.
    s = _S(path, preference="stream", force_stream=True, interactive=False, verbose=False,
           budget_gb=getattr(a, "memory", None), kv_bits=getattr(a, "kv_bits", None), kv_quant_start=getattr(a, "kv_quant_start", None), prompt_cache_gb=0, monitor=False)
    if not getattr(s, "handle", None):
        print(f"\n  {s.name} could not be streamed here, so there are no expert pools to "
              f"compare against.\n")
        s.close()
        return 0

    def run():
        return "".join(c for c, _ in s.stream_text(
            [{"role": "user", "content": prompt}], max_tokens=a.max_tokens, think=False,
            temperature=0.0) if c)

    print(f"\n  {s.name}  --  {a.bits}-bit against the checkpoint as shipped")
    print(f"  prompt: {prompt[:70]!r}\n")
    ref = run()
    snaps = [(pl, _tune.simulate_precision(pl, int(a.bits), int(a.group))) for pl in s.handle.pools]
    mx.clear_cache()
    got = run()
    for pl, sn in snaps:
        _tune.restore(pl, sn)
    mx.clear_cache()
    s.close()

    if ref == got:
        print(f"  IDENTICAL. {len(ref)} characters, not one of them different.\n")
        return 0
    i = next((k for k, (x, y) in enumerate(zip(ref, got)) if x != y), min(len(ref), len(got)))
    # Token-level would be tidier and character-level is what a person can check by eye against
    # the two texts printed below it.
    print(f"  They agree for {i} characters, then part.\n")
    lo = max(0, i - 60)
    print(f"    shared    ...{ref[lo:i]}")
    print(f"    shipped   -> {ref[i:i + 70]!r}")
    print(f"    {a.bits}-bit     -> {got[i:i + 70]!r}\n")
    print(f"  as shipped : {ref.strip()[:300]}\n")
    print(f"  at {a.bits} bits  : {got.strip()[:300]}\n")
    return 0


def _knee_inputs(probe) -> dict:
    """What `knee.measure` needs, read off a live session. Shared by `bigrig knee` and the
    first-run tune so the two cannot drift apart."""
    # The pool's budget, not the ceiling: with a head or a draft beside the pool they differ,
    # and a knee keyed on the ceiling is found and skipped while the plan cannot use it.
    budget = getattr(probe, "pool_budget_gb", probe.budget_gb)
    n_exp = int(probe.plan["n_experts"])
    n_layers = int(probe.plan["n_layers"])
    gb_per_slot = probe.plan["bytes_per_expert"] * n_layers / 1e9
    fits = max(1, int((budget - probe.non_expert_gb - probe.working_memory_gb) // gb_per_slot))
    return {"budget": budget, "n_exp": n_exp, "top_k": int(probe.top_k or 1),
            "n_layers": n_layers, "gb_per_slot": gb_per_slot, "fits": fits}


def _knee_maker(a, path, budget_gb=None):
    """A `make_session(capacity)` for `knee.measure`, built the way it will be SERVED, cache and all.

    It is tempting to bypass the page cache here so the disk is priced honestly, and that is
    right for a disk benchmark and wrong for this: the capacity being chosen is the one the
    model will RUN at, and it will run with the cache. Measured cold, 53 experts looks faster
    than 43; measured as served, 43 wins 11.09 tok/s against 9.24.
    """
    from .session import Session as _S

    def make(c):
        return _S(path, capacity=int(c), preference="stream", interactive=False,
                  verbose=False,
                  budget_gb=(budget_gb if budget_gb is not None else getattr(a, "memory", None)),
                  kv_bits=getattr(a, "kv_bits", None),
                  kv_quant_start=getattr(a, "kv_quant_start", None))
    return make


def _auto_tune(a, s):
    """Measure this model's fastest capacity on first run, once, and remember it.

    WHY THIS RUNS WITHOUT BEING ASKED. `bigrig knee` existed and nobody ran it, so every
    streamed model ran on the planner's estimate -- safe, and on the one model measured, 1.23x
    slower than the setting the knee finds. A user should not have to know a command exists to
    get the speed their machine can give. The measurement is a minute or two, happens once per
    model and budget, and is skipped for anyone who has already answered the question with
    --residency or asked not to with --no-tune.

    Returns the session to use: the one passed in if nothing was measured, otherwise a fresh
    one built after the knee was saved, so the running session is the tuned one.
    """
    from . import knee as _knee
    if not getattr(s, "streamed", False):
        return s                                  # resident model: no capacity to choose
    if getattr(a, "no_tune", False) or getattr(a, "residency", None) is not None:
        return s                                  # the user has already decided
    if getattr(a, "cmd", "") in ("knee", "calibrate"):
        return s                                  # those commands ARE the measurement
    if _knee.load(s.name, getattr(s, "pool_budget_gb", s.budget_gb)) is not None:
        return s                                  # measured before, at this budget; in use already
    inp = _knee_inputs(s)
    if inp["fits"] <= inp["top_k"] + 1:
        return s                                  # nothing to choose between
    path = s.init_kwargs["model_dir"]
    rebuild = dict(s.init_kwargs)
    # THE REBUILT SESSION MUST PLAN FROM THE BUDGET THE TUNE MEASURED AT.
    #     A `None` budget resolves against free memory at that instant, and the instant after a
    #     pool is dropped MLX has not yet returned it. Measured: the tune ran at a 9.1 GB pool
    #     budget and saved its knee; the session built next resolved 8.4 GB, planned from that,
    #     and found no knee for it -- so the first run served an untuned plan after tuning.
    rebuild["budget_gb"] = s.budget_gb
    rebuild["announce"] = True                # this is the session that will be served
    print(f"\n  first run of {s.name} at {inp['budget']:.1f} GB: measuring how many experts to "
          f"keep in memory for the best speed on this Mac.", flush=True)
    # THE OLD LINE PROMISED A DURATION IT COULD NOT KEEP -- one to two minutes, against the two
    # to five these models actually take -- and a wait that overruns its promise feels longer
    # than one that was never quantified. The measurement reports its own progress and its own
    # estimate as it goes, so the opening line does not have to guess.
    print(f"  one-time, a few minutes; it says where it is as it goes. Skip it with --no-tune, "
          f"or pick a number yourself with --residency.\n", flush=True)
    # ONE POOL ALIVE AT A TIME. The probes below each build a session; the one we were handed
    # has to go first, or a machine picked for not having enough memory holds two pools.
    s.close()
    del s
    from .session import Session as _S
    try:
        res = _knee.measure(_knee_maker(a, path, inp["budget"]), os.path.basename(path), inp["budget"],
                            inp["n_exp"], inp["top_k"], inp["n_layers"], inp["gb_per_slot"],
                            inp["fits"], probes=3, verbose=True)
        _knee.save(res)
        # The probe's tok/s is a like-for-like ranking under identical short prompts, NOT the
        # speed a user will see on a real reply (measured 18.3 in the probe against 5.7 on the
        # reply that followed). So the pick is reported and the probe number is not.
        tried = ", ".join(str(c) for c in sorted(res.get("measured", {}), key=int))
        print(f"\n  measured: keeping {res['capacity']} of {inp['n_exp']} experts in memory is "
              f"fastest here (tried {tried})", flush=True)
        print(f"  remembered; every run from now on uses it.\n", flush=True)
    except Exception as e:                       # noqa: BLE001 -- a tune must never block a run
        print(f"\n  could not finish the measurement ({str(e)[:80]}); using the safe estimate.\n",
              flush=True)
    return _S(**rebuild)


def cmd_knee(a) -> int:
    """Find the capacity where holding more experts stops paying for itself, and remember it."""
    from . import knee as _knee
    from .session import Session as _S
    path = resolve_model(a.model)

    # One session at a time, ALWAYS. Two pools alive at once is the single thing a machine picked
    # for not having enough memory cannot survive, so the probe loop closes each before the next.
    probe = _S(path, preference="stream", interactive=False, verbose=False,
               budget_gb=getattr(a, "memory", None), kv_bits=getattr(a, "kv_bits", None), kv_quant_start=getattr(a, "kv_quant_start", None))
    # The knee is a measurement of the path the model runs on, so the packed copy comes first
    # here too -- a knee measured on the copy path and served on the zero-copy one is a number
    # about a different engine.
    probe = _auto_pack(a, probe)
    if not getattr(probe, "streamed", False):
        print(f"\n  {probe.name} fits in memory here, so every expert is already resident and\n"
              f"  there is no capacity to choose.\n")
        probe.close()
        return 0
    inp = _knee_inputs(probe)
    budget, n_exp, gb_per_slot, fits = inp["budget"], inp["n_exp"], inp["gb_per_slot"], inp["fits"]
    probe.close()

    print(f"\n  {os.path.basename(path)} on this machine: {n_exp} experts per layer, "
          f"{gb_per_slot:.3f} GB each,\n  {budget:.1f} GB budget, at most {fits} fit\n")

    res = _knee.measure(_knee_maker(a, path, budget), os.path.basename(path), budget, n_exp,
                        inp["top_k"], inp["n_layers"], gb_per_slot, fits,
                        probes=a.probes, tolerance=a.tolerance)
    where = _knee.save(res)
    print(f"\n  knee: {res['capacity']} of {n_exp} experts per layer "
          f"({res['resident_gb']} GB)")
    print(f"  {res['why']}")
    print(f"\n  saved to {where}")
    print(f"  `bigrig serve {a.model}` will use it. Delete that file to measure again.\n")
    return 0


def cmd_calibrate(a) -> int:
    """Measure what the host round-trip costs on THIS model, and remember it."""
    from . import synccal
    s = _session(a)
    if not getattr(s, "streamed", False):
        print(f"\n  {s.name} runs entirely in memory here, so it never pays a host round-trip "
              f"and\n  there is nothing to calibrate.\n")
        return 0
    print(f"\n  measuring {s.name} -- {len(s.handle.mods)} streamed layers, "
          f"{a.tokens} tokens per point")
    print("  the text produced during this is deliberately wrong and is not shown: layers are\n"
          "  forced past the round-trip to time it, not to answer with it\n")
    curve = synccal.measure(s, tokens=a.tokens, points=a.points)

    # THE OTHER HALF. A per-model sync curve alone still predicted 1.15x on OLMoE and delivered
    # 0.88x, because how fast the miss rate rises as residency falls was still coming from
    # somebody else's traces. Both halves are measured or neither is worth having.
    n_exp = s.handle.stats()["n_experts"]
    here = s.handle.stats()["capacity"] / n_exp
    miss = {}
    r0, m0 = synccal.observe_miss(s, tokens=a.tokens)
    miss[round(r0, 4)] = round(m0, 5)
    print(f"\n  miss rate at {r0:.0%} residency: {m0:.1%}")
    from .session import Session as _S
    del s
    for frac in (0.25, 0.5, 0.75, 0.9):
        if abs(frac - here) < 0.03 or round(frac, 4) in miss:
            continue
        try:
            s2 = _S(resolve_model(a.model), capacity=frac, preference="stream",
                    interactive=False, verbose=False, budget_gb=getattr(a, "memory", None), kv_bits=getattr(a, "kv_bits", None), kv_quant_start=getattr(a, "kv_quant_start", None))
            r, m = synccal.observe_miss(s2, tokens=a.tokens)
            miss[round(r, 4)] = round(m, 5)
            print(f"  miss rate at {r:.0%} residency: {m:.1%}", flush=True)
            del s2
        except (MemoryError, ValueError) as e:
            print(f"  {frac:.0%} residency could not be measured here: {e}")
    curve["miss_by_residency"] = {str(k): v for k, v in sorted(miss.items())}
    path = synccal.save(curve)
    print(f"\n  round-trip cost   {curve['ms_per_sync_layer']:.2f} ms per syncing layer")
    if curve.get("ms_per_miss"):
        print(f"  cache miss cost   {curve['ms_per_miss']:.2f} ms")
        ratio = curve["ms_per_sync_layer"] / curve["ms_per_miss"]
        print(f"  one round-trip is worth {ratio:.1f} misses on this model")
    # PREDICTING THIS WAS NOT GOOD ENOUGH, SO IT IS MEASURED INSTEAD.
    #     With both curves measured on this model the planner still predicted 1.63x and delivered
    #     0.81x -- it made the model slower. The sync curve is noisy enough to come out
    #     non-monotonic between runs (11 syncing layers timing above 16), and a planner fed a
    #     noisy curve will confidently pick a configuration nobody has run. So the split it
    #     proposes is RUN, against uniform, and only the measured ratio is stored. A split that
    #     did not actually help is recorded as not helping.
    from . import stream as _stream, synccal as _sc
    ok, why = _sc.usable(curve)
    curve["usable"], curve["unusable_reason"] = ok, why
    curve["verified_speedup"] = None
    if not ok:
        print(f"\n  this measurement is not good enough to plan with: {why}.")
        print("  Uniform capacity will keep shipping, which is what happens today. The miss\n"
              "  curve above is still recorded and still true.")
    fn = _sc.as_cost_fn(curve) if ok else None
    if fn:
        try:
            man2, _b2 = _stream.expert_source(resolve_model(a.model))
            nl2 = len(man2["layers"])
            ne2 = man2["layers"]["0"]["n_experts"]
            slots = int(here * ne2) * nl2
            pl = _stream.plan_capacity(nl2, ne2, slots, top_k=curve.get("top_k") or 8, measured=fn)
            if pl["full_layers"]:
                print(f"\n  the plan wants {pl['full_layers']} layers held whole. Checking it "
                      f"against uniform, on this machine:")
                ratio = _verify_split(a, pl, nl2, ne2, int(here * ne2), a.tokens)
                curve["verified_speedup"] = round(ratio, 4)
                curve["verified_plan"] = {"full_layers": pl["full_layers"],
                                          "capacity": pl["capacity"]}
                verdict = ("faster -- it will be used" if ratio > 1.02
                           else "NOT faster -- uniform capacity will keep shipping")
                print(f"    measured {ratio:.2f}x  ({verdict})")
            else:
                print("\n  the plan wants no layers held whole, so uniform capacity is already "
                      "the answer here.")
        except (MemoryError, ValueError, KeyError, ZeroDivisionError) as e:
            print(f"\n  the split could not be checked here ({e}); uniform capacity will ship.")

    if len(miss) < 2:
        print("\n  only one residency could be measured, so the planner will keep using uniform\n"
              "  capacity -- half a curve is what made this feature wrong the first time.")
    print(f"\n  saved to {path}")
    print("  `bigrig serve` and `bigrig run` will use it from now on.\n")
    return 0


def cmd_serve(a) -> int:
    from .server import serve
    s = _session(a)          # any consent question is asked HERE, before the port opens
    # A NON-LOOPBACK BIND PUTS AN UNAUTHENTICATED MODEL ON THE NETWORK. Say so, once, loudly.
    # There is no API key in this server; anyone who can reach the port can use the model and
    # read /stats. On a home or cafe network that is everyone on the Wi-Fi.
    if a.host.strip("[]").lower() not in ("127.0.0.1", "localhost", "::1"):
        print(f"\n  WARNING: --host {a.host} puts this server on the network, and it has no\n"
              f"  authentication. Anyone who can reach {a.host}:{a.port} can use the model, read\n"
              f"  /stats, and change the pool size. Use 127.0.0.1 unless you meant this.\n",
              flush=True)
    return serve(s, host=a.host, port=a.port, batch=getattr(a, "batch", 1),
                 release_memory=getattr(a, "release_memory", True),
                 reclaim_memory=getattr(a, "reclaim_memory", True),
                 warm_cache=not getattr(a, "no_warm", False),
                 cors_origins=tuple(getattr(a, "cors_origin", ()) or ()))


# ------------------------------------------------------------------------------ entry point
def build_parser():
    p = argparse.ArgumentParser(prog="bigrig", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd")

    d = sub.add_parser("doctor", help="what this machine can run")
    d.add_argument("model", nargs="?", help="report on one model instead of all of them")
    d.add_argument("--calibrate", action="store_true", help="measure bandwidths (takes ~30s)")
    d.add_argument("--memory", type=float, default=None,
                   help="plan against this many GB instead of what is free now")
    d.set_defaults(fn=cmd_doctor)

    l = sub.add_parser("list", help="models already prepared")
    l.set_defaults(fn=cmd_list)

    pr = sub.add_parser("prepare", help="download a model and make the packed copy the fast "
                                        "path needs")
    pr.add_argument("model")
    pr.add_argument("-q", "--quiet", dest="verbose", action="store_false", default=True)
    pr.add_argument("--no-pack", action="store_true",
                    help="skip the packed copy. It doubles the model's disk footprint, and "
                         "without it every expert is copied in from the model's own files "
                         "instead of read in place by the GPU (measured 5.4 -> 21 tok/s with a "
                         "warm page cache, 1.07x when the disk is the bottleneck)")
    pr.add_argument("--pack", action="store_true", help=argparse.SUPPRESS)   # the old default
    pr.add_argument("--trust-remote-code", action="store_true",
                    help="also download the model's custom Python (it will be executed)")
    pr.set_defaults(fn=cmd_prepare)

    def common(x):
        x.add_argument("model")
        x.add_argument("--no-pack", action="store_true",
                       help="do not make the packed copy of the experts on first run; keep the "
                            "disk and take the slower copy path")
        x.add_argument("--residency", type=float, default=None,
                       help="fraction of experts to keep resident (default: from free memory)")
        x.add_argument("--memory", type=float, default=None,
                       help="GB the engine may use (default: what is actually free)")
        x.add_argument("--threads", type=int, default=8)
        x.add_argument("--kv-bits", type=int, default=None, dest="kv_bits",
                       choices=(2, 3, 4, 5, 6, 8),
                       help="compress the conversation cache once it passes --kv-quant-start "
                            "tokens. 3.56x less memory at 4 bits, which on this model is the "
                            "difference between reaching 45%% of its context window and all of "
                            "it. It is a compression: replies past the threshold will differ.")
        x.add_argument("--kv-quant-start", type=int, default=None, dest="kv_quant_start",
                       help="tokens that stay uncompressed before --kv-bits engages "
                            "(default 4096), so short conversations are untouched")
        x.add_argument("--no-full-layers", action="store_true",
                       help="do not hold any layer's complete expert set. Holding a few whole "
                            "lets them skip the pause to ask the CPU which experts to load -- "
                            "the largest single cost in the engine, 1.23x measured. This turns "
                            "that off.")
        x.add_argument("--no-tune", action="store_true",
                       help="skip the one-time speed measurement a streamed model gets on its "
                            "first run here. Without a measurement the engine uses a safe "
                            "estimate for how many experts to keep in memory; the measurement "
                            "(a minute or two, once per model and budget) finds the fastest "
                            "setting on this machine and remembers it.")
        x.add_argument("--draft", default=None,
                       help="a small model to propose tokens, checked by the big one in a single "
                            "pass (speculative decoding). Must share the target's vocabulary. "
                            "Its weights come out of the same memory budget.")
        x.add_argument("--draft-tokens", type=int, default=3,
                       help="tokens the draft proposes per step (default 3)")
        x.add_argument("--reroute", type=float, default=0.0, metavar="TOL",
                       help="send a token to a resident expert when the one the router chose is "
                            "not in memory and the substitute scored within TOL of it (e.g. 0.1 "
                            "for 10%%). THIS CHANGES THE OUTPUT -- it is a speed-for-quality "
                            "trade like --compress, not a free win. Off by default.")
        x.add_argument("--mtp", nargs="?", const="auto", default=None, metavar="PATH",
                       help="guess one token ahead with the model's own multi-token-prediction "
                            "head, and have the model check every guess. PATH is the head's "
                            "directory; given alone, <model>-MTP-bf16 beside the model. "
                            "Measured on Qwen3.6-35B-A3B-4bit: 88%% of guesses right.")
        x.add_argument("--mtp-bits", type=int, default=4, choices=(0, 4, 8),
                       help="quantise the head's experts to this many bits (default 4: a third "
                            "of the memory, same acceptance measured; 0 keeps bf16)")
        x.add_argument("--file-pool", action="store_true",
                       help="run the experts straight from the file's cached pages instead of "
                            "copying each into a pool slot. Measured on Qwen3.6-35B-A3B-4bit: "
                            "1.4-2.0x faster to the first token, about 1.1x faster decode, and a "
                            "0.5 GB smaller footprint because the hot experts no longer exist "
                            "twice. The arithmetic runs through a different kernel than a "
                            "resident model's, so a reply can differ in a near-tie (1 of 3 "
                            "measured). Off by default for that reason.")
        x.add_argument("--prefetch", type=int, default=0, metavar="N",
                       help="experts to name a layer ahead from the hidden state. OFF by "
                            "default and measured not to pay on Qwen3.6 (0.84x prose, 0.98x code, "
                            "interleaved): with BIGRIG_STAGE=1 the named experts are copied to the "
                            "GPU during the layer's own attention, and the copy contends with the "
                            "weights the GPU is streaming. Kept for models with a sharper predictor. "
                            "A wrong guess never changes an answer: the router still decides what runs. Try 8 when the disk "
                            "has headroom. Needs `bigrig predict` first. A wrong guess wastes "
                            "a read and never changes an answer.")
        x.add_argument("--no-monitor", action="store_true", help="turn off quality monitoring")
        x.add_argument("--force-stream", action="store_true",
                       help="stream even if the model fits (for measurement)")
        x.add_argument("--trust-remote-code", action="store_true",
                       help="also download the model's custom Python (it will be executed)")
        x.add_argument("--min-bits", type=int, default=None,
                       help="never compress below this precision (default 3; 2-bit measured "
                            "at +83 to +145%% perplexity, so it is opt-in)")
        x.add_argument("--exact", action="store_true",
                       help="keep the weights untouched (decode bit-identical to the original); stream "
                            "instead of compressing")
        x.add_argument("--compress", action="store_true",
                       help="agree to shrink the model to fit (the weights change)")
        x.add_argument("--forget-choice", action="store_true",
                       help="discard the remembered compress/exact decision and ask again")

    r = sub.add_parser("run", help="chat in the terminal")
    common(r)
    r.add_argument("--max-tokens", type=int, default=512)
    r.add_argument("--temperature", type=float, default=0.7)
    r.set_defaults(fn=cmd_run)

    cp = sub.add_parser("compress", help="shrink a model so every expert fits in RAM")
    cp.add_argument("model")
    cp.add_argument("--bits", type=int, default=0, help="force a precision instead of choosing")
    cp.add_argument("--group", type=int, default=128)
    cp.add_argument("--memory", type=float, default=None, help="GB the engine may use")
    cp.add_argument("--trust-remote-code", action="store_true")
    cp.add_argument("--tune", action="store_true",
                    help="measure which layers can afford to be compressed and spend the budget "
                         "accordingly (slower, once per model, cached)")
    cp.add_argument("--retune", action="store_true",
                    help="discard the cached sensitivity profile and measure again")
    cp.add_argument("--min-bits", type=int, default=3,
                    help="never allocate below this precision (default 3)")
    cp.add_argument("--corpus", default="wikitext2_valid.txt",
                    help="corpus in data/corpora used to measure sensitivity")
    cp.add_argument("--tune-memory", type=float, default=None,
                    help="GB the one-off sensitivity measurement may use (default: what is "
                         "actually free). Separate from --memory, which sizes the result.")
    cp.add_argument("--window", type=int, default=512)
    cp.add_argument("--windows", type=int, default=3)
    cp.set_defaults(fn=cmd_compress)

    ln = sub.add_parser("launch", help="run a coding agent against a local model")
    # The two options that are about LAUNCHING come first in --help; the thirty engine flags
    # that follow are the same on every command and a person choosing an agent should not have
    # to read past them to find --agent.
    ln.add_argument("--agent", default="claude", choices=("claude", "codex", "opencode"),
                    help="which coding agent to launch (default: claude)")
    ln.add_argument("--port", type=int, default=0,
                    help="port for the server (default: pick a free one)")
    common(ln)
    ln.add_argument("agent_args", nargs="*",
                    help="anything after the model is passed to the agent")
    ln.set_defaults(fn=cmd_launch)

    pr = sub.add_parser("predict",
                        help="fit the predictor that lets experts be read a layer early")
    common(pr)
    pr.add_argument("--tokens", type=int, default=90, help="tokens generated per prompt")
    pr.add_argument("--prompts", type=int, default=6, help="how many prompts to fit on")
    pr.set_defaults(fn=cmd_predict)

    cal = sub.add_parser("calibrate",
                         help="measure this model's host round-trip cost, so the planner can "
                              "stop using another model's")
    common(cal)
    cal.add_argument("--tokens", type=int, default=24, help="tokens timed per point")
    cal.add_argument("--points", type=int, default=5, help="points on the curve")
    cal.set_defaults(fn=cmd_calibrate)

    kn = sub.add_parser("knee",
                        help="find the capacity where more experts stop paying for themselves")
    common(kn)
    kn.add_argument("--probes", type=int, default=3,
                    help="cheap miss-rate probes used to fit the curve (default 3)")
    kn.add_argument("--tolerance", type=float, default=0.10,
                    help="speed you are willing to give up to give memory back (default 0.10)")
    kn.set_defaults(fn=cmd_knee)

    df = sub.add_parser("diff",
                        help="show where compression changes this model's answer")
    common(df)
    df.add_argument("--prompt", default="", help="the prompt to compare on")
    df.add_argument("--bits", type=int, default=3,
                    help="precision to compare against the shipped checkpoint (default 3)")
    df.add_argument("--group", type=int, default=64, help="quantisation group size (default 64)")
    df.add_argument("--max-tokens", type=int, default=160, dest="max_tokens")
    df.set_defaults(fn=cmd_diff)

    sv = sub.add_parser("serve", help="OpenAI- and Anthropic-compatible server")
    common(sv)
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8080)
    sv.add_argument("--cors-origin", action="append", metavar="ORIGIN",
                    help="let a web page on ORIGIN (e.g. https://example.com) call this server "
                         "from a browser. Repeatable. The bundled page needs none of this -- it "
                         "is served from here. Without it a browser may not read this server's "
                         "replies cross-origin, which is what stops a page you happen to have "
                         "open from quietly using your model.")
    sv.add_argument("--no-warm", action="store_true",
                    help="do not read this model's experts into the OS page cache after "
                         "starting. Warming runs in the background and only helps a machine "
                         "that has just booted.")
    # ON BY DEFAULT, AND THE DEFAULT IS THE WHOLE POINT.
    #     Off, the failure is not a slow server, it is a dead one: Metal kills the process with
    #     'Insufficient Memory' when something else on the machine wants memory the pool is
    #     holding, which is exactly what happened here running the test suite beside a server.
    #     A user cannot opt into a flag protecting them from a crash they have not had yet.
    #     What it costs when it does fire is one pool rebuild, about a second and a half, and a
    #     slower model until the squeeze passes -- visible, recoverable, and strictly better
    #     than being killed. Only the shrinking half is on: `--reclaim-memory` still has to be
    #     asked for, because a wrong grow is how a controller takes a machine down.
    sv.add_argument("--release-memory", dest="release_memory", action="store_true", default=True,
                    help="(on by default) hand experts back between replies when the machine "
                         "runs short of memory. Only ever gives memory back, never takes more, "
                         "and never acts during a reply.")
    sv.add_argument("--no-release-memory", dest="release_memory", action="store_false",
                    help="never hand experts back. The pool keeps every expert it started with, "
                         "and a machine that runs short while this server holds memory can kill "
                         "it with Metal 'Insufficient Memory' instead of shrinking to fit.")
    sv.add_argument("--reclaim-memory", dest="reclaim_memory", action="store_true",
                    default=True,
                    help="(on by default) take the borrowed experts back once the machine has "
                         "been quiet for three minutes, in small steps, never above the "
                         "capacity it started from. Measured without it: one squeeze left a "
                         "server at 30 of 38 experts a layer for the rest of the day.")
    sv.add_argument("--no-reclaim-memory", dest="reclaim_memory", action="store_false",
                    help="only ever give memory back; never take it back once the machine is "
                         "quiet. A restart is then the only way to recover speed after a squeeze.")
    sv.add_argument("--batch", type=int, default=1,
                    help="serve up to N waiting requests in one pass (default 1). Raises "
                         "throughput for concurrent users, but replies will not match those "
                         "from serving one at a time -- batching changes the reduction order.")
    sv.set_defaults(fn=cmd_serve)
    return p


def main(argv=None) -> int:
    p = build_parser()
    a = p.parse_args(argv)
    if not getattr(a, "fn", None):
        p.print_help()
        return 1
    os.makedirs(BLOBS_DIR, exist_ok=True)
    from .consent import ConsentRequired
    try:
        return a.fn(a)
    except ConsentRequired as e:
        # A decision the user has to make is not a crash. Showing a traceback for it makes a
        # deliberate refusal look like a bug and buries the two flags that resolve it.
        print(f"\n  {e}\n", file=sys.stderr)
        return 2
    except ModuleNotFoundError as e:
        # A missing engine dependency is a setup problem with a one-line fix, not a bug. Before
        # this, `pip install bigrig` followed by `bigrig doctor` -- the first command in this
        # tool's own help text -- ended in a traceback whose last line was
        # `ModuleNotFoundError: No module named 'mlx'`, which tells a user nothing they can act
        # on. MLX is now installed by default on the machines it runs on; this is what is left.
        if (e.name or "").split(".")[0] not in ("mlx", "mlx_lm", "huggingface_hub"):
            raise
        import platform
        apple = sys.platform == "darwin" and platform.machine() == "arm64"
        print(f"\n  bigrig needs {e.name}, which is not installed.\n", file=sys.stderr)
        if apple:
            print("      pip install --upgrade 'bigrig[engine]'\n", file=sys.stderr)
        else:
            print(f"  The engine runs on Apple Silicon only, and this is "
                  f"{sys.platform}/{platform.machine()}. Streaming experts off disk needs MLX's\n"
                  f"  unified-memory model, which has no equivalent here.\n\n"
                  f"  The quality meter (`bigrig_layer`) does run anywhere and needs nothing "
                  f"beyond numpy.\n", file=sys.stderr)
        return 1
    except (FileNotFoundError, MemoryError, ValueError) as e:
        print(f"\n  {e}\n", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print()
        return 130


if __name__ == "__main__":
    sys.exit(main())
