#!/usr/bin/env python3
"""
Same-cycle drift + direction measurement, and the feedback it writes back.

    python -m ops.drift_track --uid 250 --leaders Attila115/co3,hunter-04/co228
    python -m ops.drift_track --uid 250 --watch          # loop, once per cycle

Why this exists
---------------
An earlier comparison diffed our shard against a global checkpoint from a LATER
cycle and produced cosine -0.987 against the leaders -- an arithmetic artifact,
not a finding. The validator merges top miners into each new global, so
diffing a checkpoint against a future global yields roughly
`our_update - leaders_update`, which points backwards by construction.

The only valid comparison pairs every shard with the global checkpoint it was
actually trained from. This script enforces that pairing:

  1. read our miner's own commit log for the (timestamp, global_rev) it
     downloaded at Distribute, and the (timestamp, our_rev) it uploaded at
     MinerCommit2;
  2. our upload is attributed to the LAST global downloaded BEFORE it;
  3. for each leader repo, pick the HF commit that falls in the same window --
     after that global was published, and closest to our own upload time;
  4. diff everything against that one global revision.

What it reports
---------------
  ||u||            magnitude of (miner - global): how far each party trained
  cos(u_i, u_j)    direction agreement between update vectors

Direction is the decisive number. If cos(ours, leaders) is strongly positive we
are travelling the same way and magnitude is the lever. If it is near zero we
are optimising a different objective -- a data problem, not a schedule problem.
Shrinking the learning rate only helps in the first case.

All arithmetic accumulates in float64 per tensor. fp32 dot products over 1.67B
elements lose enough precision to return cosines outside [-1, 1], which is how
the first attempt produced nonsense.
"""

from __future__ import annotations

import argparse
import math
import os
import re
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from huggingface_hub import HfApi, hf_hub_download
from safetensors.torch import load_file

LOG_DIR = Path(os.environ.get("LOG_DIR", "/root/SN102/logs"))
TUNER_DB = os.environ.get("TUNER_DB", "/root/SN102/data/tuner.sqlite")
GROUP = int(os.environ.get("EXPERT_GROUP_ID", "3"))
ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _parse_log(uid: int):
    """Return (downloads, uploads) as lists of (datetime, value)."""
    path = LOG_DIR / f"uid{uid}-commit.log"
    downloads, uploads = [], []
    if not path.is_file():
        return downloads, uploads
    for raw in path.open(errors="ignore"):
        line = ANSI.sub("", raw)
        ts = line[:19]
        try:
            when = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        if "Downloaded checkpoint (verified)" in line:
            m = re.search(r"hf_repo_id=(\S+)\s+hf_revision=(\S+)", line)
            if m:
                downloads.append((when, m.group(1), m.group(2).rstrip(".,'\"")))
        elif "MinerCommit2> committing" in line:
            m = re.search(r"hf_repo_id=(\S+)\s+hf_revision=(\S+)", line)
            if m:
                uploads.append((when, m.group(1), m.group(2).rstrip(".,'\"")))
    return downloads, uploads


def _pair_latest(uid: int):
    """Our most recent upload, paired with the global it was trained from."""
    downloads, uploads = _parse_log(uid)
    if not uploads:
        raise SystemExit(f"uid {uid}: no MinerCommit2 uploads found in log")
    up_when, up_repo, up_rev = uploads[-1]
    prior = [d for d in downloads if d[0] < up_when]
    if not prior:
        raise SystemExit(
            f"uid {uid}: upload {up_rev} at {up_when} has no preceding global download; "
            f"cannot form a valid same-cycle comparison"
        )
    g_when, g_repo, g_rev = prior[-1]
    return (g_when, g_repo, g_rev), (up_when, up_repo, up_rev)


def global_publish_window(global_repo: str, global_rev: str, token):
    """[published, next_published) for a global revision.

    Any miner shard committed inside this window was necessarily trained from
    THIS global -- it is the only one that existed. That is the only sound way
    to pair shards across operators; guessing from our own upload time is what
    produced an earlier bogus comparison.
    """
    commits = HfApi(token=token).list_repo_commits(repo_id=global_repo, revision="main")
    commits = sorted(commits, key=lambda c: c.created_at)
    start = end = None
    for i, c in enumerate(commits):
        if c.commit_id.startswith(global_rev):
            start = c.created_at.astimezone(timezone.utc)
            if i + 1 < len(commits):
                end = commits[i + 1].created_at.astimezone(timezone.utc)
            break
    if start is None:
        raise SystemExit(f"global revision {global_rev} not found in {global_repo}")
    if end is None:
        end = datetime.now(timezone.utc)
    return start, end


def _leader_rev_in_window(repo: str, start, end, token):
    """Leader commit inside the SAME global-publish window."""
    try:
        commits = HfApi(token=token).list_repo_commits(repo_id=repo, revision="main")
    except Exception as exc:  # noqa: BLE001
        print(f"  {repo}: cannot list commits ({str(exc)[:70]})")
        return None
    inside = [(c.created_at.astimezone(timezone.utc), c.commit_id[:7])
              for c in commits if start <= c.created_at.astimezone(timezone.utc) < end]
    if not inside:
        print(f"  {repo}: no commit within {start:%H:%M}..{end:%H:%M}")
        return None
    return max(inside)  # last submission in the window


def _load(repo: str, rev: str, token):
    return load_file(hf_hub_download(
        repo_id=repo, filename=f"model_expgroup_{GROUP}.safetensors", revision=rev, token=token))


def _gram(global_sd, others: dict[str, dict]):
    """float64 Gram matrix over update vectors (miner - global), per tensor."""
    names = list(others)
    keys = sorted(set(global_sd).intersection(*[set(v) for v in others.values()]))
    n = len(names)
    dots = [[0.0] * n for _ in range(n)]
    gnorm = 0.0
    for k in keys:
        g = global_sd[k].to(torch.float64)
        gnorm += float(torch.sum(g * g))
        us = [others[nm][k].to(torch.float64) - g for nm in names]
        for i in range(n):
            for j in range(i, n):
                v = float(torch.sum(us[i] * us[j]))
                dots[i][j] += v
                if i != j:
                    dots[j][i] += v
    return names, dots, math.sqrt(gnorm), len(keys)


def measure(uid: int, leaders: list[str], token) -> float | None:
    (g_when, g_repo, g_rev), (up_when, up_repo, up_rev) = _pair_latest(uid)
    print(f"uid {uid}")
    print(f"  global trained from : {g_repo}@{g_rev}   (downloaded {g_when:%Y-%m-%d %H:%M:%S}Z)")
    print(f"  our submission      : {up_repo}@{up_rev}  (uploaded   {up_when:%Y-%m-%d %H:%M:%S}Z)")

    w_start, w_end = global_publish_window(g_repo, g_rev, token)
    print(f"  global valid window : {w_start:%H:%M:%S}Z .. {w_end:%H:%M:%S}Z")
    if not (w_start <= up_when < w_end):
        print(f"  WARNING: our upload at {up_when:%H:%M:%S}Z is OUTSIDE that window -- "
              f"comparison would be invalid")
    sds = {"ours": _load(up_repo, up_rev, token)}
    for repo in leaders:
        hit = _leader_rev_in_window(repo, w_start, w_end, token)
        if hit:
            print(f"  leader {repo}@{hit[1]} (committed {hit[0]:%H:%M:%S}Z)")
            sds[repo] = _load(repo, hit[1], token)

    gsd = _load(g_repo, g_rev, token)
    names, dots, gnorm, nkeys = _gram(gsd, sds)

    print(f"\n  compared {nkeys} tensors  (||global|| = {gnorm:.4f})")
    print("  update magnitudes ||miner - global||:")
    norms = [math.sqrt(max(0.0, dots[i][i])) for i in range(len(names))]
    for i, nm in enumerate(names):
        print(f"    {nm:<28} {norms[i]:10.4f}   rel-L2 {norms[i]/gnorm:.6f}")
    print("  direction agreement (cosine between update vectors):")
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            c = dots[i][j] / (norms[i] * norms[j]) if norms[i] > 0 and norms[j] > 0 else float("nan")
            print(f"    {names[i]:<20} vs {names[j]:<20} {c:+.6f}")

    our_rel = norms[names.index("ours")] / gnorm if gnorm > 0 else None
    return our_rel


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--uid", type=int, required=True)
    ap.add_argument("--leaders", default="")
    ap.add_argument("--cycle", type=int, default=None,
                    help="tuner cycle_index to record drift against")
    ap.add_argument("--watch", action="store_true")
    ap.add_argument("--interval", type=int, default=1800)
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN_CORPUS") or os.environ.get("HF_TOKEN")
    leaders = [x for x in args.leaders.split(",") if x]

    while True:
        try:
            rel = measure(args.uid, leaders, token)
            if rel is not None and args.cycle is not None:
                db = sqlite3.connect(TUNER_DB)
                db.execute("UPDATE trials SET drift=? WHERE uid=? AND cycle_index=?",
                           (rel, args.uid, args.cycle))
                db.commit()
                print(f"\n  recorded drift={rel:.6f} for uid {args.uid} cycle {args.cycle}")
        except SystemExit as exc:
            print(exc)
        except Exception as exc:  # noqa: BLE001
            print(f"measurement failed: {type(exc).__name__}: {exc}")
        if not args.watch:
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
