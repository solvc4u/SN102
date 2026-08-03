#!/usr/bin/env python3
"""Rival intelligence: watch what the leaders do, compare it to us, act on it.

Everything here reads PUBLIC data -- the v3 dashboard and the miner checkpoints
that validators must be able to fetch anonymously. Nothing is copied into our
model; the point is to explain *why* a miner at 1.62 is beating us at 2.55, in
terms we can act on.

Three commands:

    python -m ops.rivals snapshot        # append leaderboard state to a JSONL
    python -m ops.rivals history         # trajectories + cohort transitions
    python -m ops.rivals compare --top 3 # tensor-level diff vs the leaders

`snapshot` is cheap and meant to run on a timer -- the interesting signal is not
any single leaderboard but how a miner's loss and cohort MOVE across rounds, and
the dashboard keeps no history we can query.

`compare` is the expensive one. It downloads each rival's
`model_expgroup_N.safetensors` plus the global checkpoint they trained from, and
reports:

  * ||their_delta|| vs ||our_delta||  -- how hard they push per round
  * cos(their_delta, our_delta)       -- same direction, or different objective
  * per-layer breakdown               -- which experts they actually move

Norms and dot products accumulate in float64. An earlier fp32 attempt over 1.67B
elements returned cosine values outside [-1, 1] -- the error term swamped the
signal.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

# The endpoint the dashboard UI itself calls. The one used before
# (dashboard-api-v2 /api/v3/leaderboard) serves a leaderboard that lags by one
# to two cycles: on 2026-08-03 it reported uid 178 at 2.3715 from cycle 16719
# while the UI and this endpoint both showed 3.5572 for the live round. Two
# configuration changes were made on the strength of that stale number.
DASHBOARD = os.environ.get(
    "CONNITO_DASHBOARD_API",
    "https://dashboard-dev.connito.ai/api/gw/api/v2/leaderboard")
LOG_DIR = Path(os.environ.get("LOG_DIR", "/root/SN102/logs"))
SNAPSHOT_FILE = LOG_DIR / "rivals.jsonl"
OUR_UIDS = {250, 178, 121}


# --------------------------------------------------------------------------
# dashboard
# --------------------------------------------------------------------------
def fetch_leaderboard(attempts: int = 10) -> dict:
    """Fetch the v3 leaderboard, retrying past the API's frequent truncation.

    The endpoint regularly fails with IncompleteRead/TimeoutError under load;
    `Accept-Encoding: identity` avoids a chunked-gzip path that truncates more
    often than plain.
    """
    last = None
    for i in range(attempts):
        try:
            req = urllib.request.Request(
                DASHBOARD, headers={"User-Agent": "sn102-ops", "Accept-Encoding": "identity"})
            return json.loads(urllib.request.urlopen(req, timeout=120).read())["data"]
        except Exception as exc:  # noqa: BLE001
            last = f"{type(exc).__name__}"
            time.sleep(5)
    raise SystemExit(f"dashboard unreachable after {attempts} attempts ({last})")


def _val_loss(m: dict) -> float | None:
    """Loss from the FRESHEST validator sample, not the most flattering one.

    This used to return min() across slots. That is wrong twice over: it picks
    whichever validator likes us best, and it happily returns a stale sample
    when a newer worse one exists. On 2026-08-03 it reported uid 178 at 2.3715
    from a cycle-16719 sample while the dashboard showed 3.5572 for the current
    cycle -- and two changes were made on the strength of that number.

    Prefer the highest sample_cycle; break ties on lowest sample_age_blocks.
    """
    cands = [v for v in (m.get("validator_metrics") or []) if v.get("val_loss") is not None]
    if not cands:
        return None
    # The gw/v2 endpoint omits sample_cycle/sample_age_blocks entirely (they come
    # back None), so ordering by them alone silently degrades to "first slot".
    # Rank by freshness where it is reported, then by eval_status_label == ok.
    def key(v):
        return (v.get("sample_cycle") if v.get("sample_cycle") is not None else -1,
                1 if v.get("eval_status_label") == "ok" else 0,
                -(v.get("sample_age_blocks") if v.get("sample_age_blocks") is not None else 1 << 30))
    return max(cands, key=key).get("val_loss")


def _val_loss_detail(m: dict) -> list[str]:
    """Per-slot view, so a disagreement between validators is visible."""
    out = []
    for v in (m.get("validator_metrics") or []):
        out.append(f"slot{v.get('validator_slot')}="
                   f"{('%.4f' % v['val_loss']) if v.get('val_loss') is not None else v.get('eval_status_label') or '-'}"
                   f"@c{v.get('sample_cycle')}"
                   f"{'' if v.get('sample_is_fresh') else '(stale)'}")
    return out


def rows_from(data: dict) -> list[dict]:
    lb = data["leaderboard"]
    lb = lb if isinstance(lb, list) else list(lb.values())
    out = []
    for m in lb:
        out.append({
            "uid": m["uid"],
            "val_loss": _val_loss(m),
            "cohort": m.get("cohort_group"),
            "incentive": m.get("incentive") or 0.0,
            "repo": m.get("hf_repo_id"),
            "rev": m.get("hf_revision"),
            "committed": m.get("committed_this_cycle"),
            "evaluated": m.get("evaluated_this_round"),
        })
    return out


def cmd_snapshot(_args) -> int:
    data = fetch_leaderboard()
    ph, rnd = data.get("phase", {}), data.get("round", {})
    rec = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cycle": ph.get("cycle_index"),
        "phase": ph.get("name"),
        "round_id": rnd.get("id"),
        "baseline": rnd.get("baseline_loss"),
        "miners": rows_from(data),
    }
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with SNAPSHOT_FILE.open("a") as fh:
        fh.write(json.dumps(rec) + "\n")
    scored = [r for r in rec["miners"] if r["val_loss"] is not None]
    ours = [r for r in rec["miners"] if r["uid"] in OUR_UIDS]
    print(f"[{rec['ts']}] cycle={rec['cycle']} phase={rec['phase']} "
          f"baseline={rec['baseline']} scored={len(scored)}")
    lb = data["leaderboard"]
    lb = lb if isinstance(lb, list) else list(lb.values())
    detail = {m["uid"]: _val_loss_detail(m) for m in lb}
    for r in sorted(ours, key=lambda x: x["uid"]):
        print(f"   uid {r['uid']}: val_loss={r['val_loss']} cohort={r['cohort']} "
              f"inc={r['incentive']:.5f}")
        if detail.get(r["uid"]):
            print(f"            {'  '.join(detail[r['uid']])}")
    return 0


# --------------------------------------------------------------------------
# history
# --------------------------------------------------------------------------
def cmd_history(args) -> int:
    if not SNAPSHOT_FILE.exists():
        print(f"no snapshots yet -- run `python -m ops.rivals snapshot` on a timer")
        return 1
    recs = [json.loads(l) for l in SNAPSHOT_FILE.read_text().splitlines() if l.strip()]
    if not recs:
        print("snapshot file is empty")
        return 1
    print(f"{len(recs)} snapshots, {recs[0]['ts']} -> {recs[-1]['ts']}\n")

    # per-uid series
    series: dict[int, list[dict]] = defaultdict(list)
    for r in recs:
        for m in r["miners"]:
            if m["val_loss"] is not None or m["incentive"] > 0:
                series[m["uid"]].append({**m, "ts": r["ts"], "cycle": r["cycle"]})

    # who earns, ranked by latest incentive
    latest = {m["uid"]: m for m in recs[-1]["miners"]}
    earners = sorted([m for m in latest.values() if m["incentive"] > 0],
                     key=lambda m: -m["incentive"])
    print("=== EARNERS (latest snapshot) ===")
    for m in earners:
        s = series.get(m["uid"], [])
        losses = [x["val_loss"] for x in s if x["val_loss"] is not None][-6:]
        trend = " ".join(f"{v:.4f}" for v in losses) or "-"
        print(f"  uid {m['uid']:<5} inc={m['incentive']:.5f} cohort={str(m['cohort']):<6} "
              f"{str(m['repo'] or '-'):<26} {trend}")

    print("\n=== US ===")
    for uid in sorted(OUR_UIDS):
        s = series.get(uid, [])
        if not s:
            print(f"  uid {uid}: no scored observations yet")
            continue
        losses = [x["val_loss"] for x in s if x["val_loss"] is not None][-6:]
        cohorts = [x["cohort"] for x in s][-6:]
        print(f"  uid {uid}: loss {' '.join(f'{v:.4f}' for v in losses) or '-'}")
        print(f"          cohort {' -> '.join(str(c) for c in cohorts)}")

    # cohort churn: who moved up, who moved down
    if len(recs) >= 2:
        first = {m["uid"]: m["cohort"] for m in recs[0]["miners"]}
        last = {m["uid"]: m["cohort"] for m in recs[-1]["miners"]}
        rank = {"A": 4, "B": 3, "C": 2, "tail": 1, "none": 0, None: 0}
        moves = [(u, first.get(u), last.get(u)) for u in last
                 if rank.get(last.get(u), 0) != rank.get(first.get(u), 0)]
        ups = [m for m in moves if rank.get(m[2], 0) > rank.get(m[1], 0)]
        if ups:
            print(f"\n=== PROMOTED since first snapshot ({len(ups)}) ===")
            for u, a, b in sorted(ups, key=lambda m: -rank.get(m[2], 0))[:12]:
                mark = "  <-- US" if u in OUR_UIDS else ""
                print(f"  uid {u:<5} {str(a):<6} -> {str(b):<6}{mark}")
    return 0


# --------------------------------------------------------------------------
# weight comparison
# --------------------------------------------------------------------------
def _download(repo: str, filename: str, revision: str | None, token: str) -> Path:
    """Fetch a checkpoint shard, tolerating the two ways the dashboard misleads.

    The `hf_revision` the leaderboard reports is often unreachable -- either the
    miner force-pushed over it (RevisionNotFoundError) or that revision predates
    the group-4 file (EntryNotFoundError). Both are the rival's repo history, not
    our bug, and both are recoverable: fall back to the branch head, and then to
    the group-3 filename for miners who have not migrated. Two of three rivals
    were unusable before this.
    """
    from huggingface_hub import hf_hub_download
    cache = os.environ.get("HF_HOME")
    attempts = [(filename, revision), (filename, None)]
    if filename.endswith("_4.safetensors"):
        attempts.append((filename.replace("_4.", "_3."), None))
    last: Exception | None = None
    for fn, rev in attempts:
        try:
            return Path(hf_hub_download(repo_id=repo, filename=fn, revision=rev,
                                        token=token, cache_dir=cache))
        except Exception as exc:  # noqa: BLE001
            last = exc
    raise last  # type: ignore[misc]


def _delta_stats(rival_path: Path, base_path: Path, ours_path: Path | None):
    """Per-tensor delta magnitudes, and cosine between their update and ours.

    Streams tensor-by-tensor via safetensors' lazy reader: three 3.3 GB files
    materialised at once would not fit comfortably alongside two running miners.
    """
    import torch
    from safetensors import safe_open

    stats = {"their_norm2": 0.0, "our_norm2": 0.0, "dot": 0.0,
             "n_tensors": 0, "n_changed": 0, "per_layer": defaultdict(float)}

    with safe_open(rival_path, framework="pt") as fr, safe_open(base_path, framework="pt") as fb:
        keys = [k for k in fr.keys() if k in set(fb.keys())]
        ours = safe_open(ours_path, framework="pt") if ours_path else None
        our_keys = set(ours.keys()) if ours else set()
        for k in keys:
            b = fb.get_tensor(k).to(torch.float64)
            t = fr.get_tensor(k).to(torch.float64) - b
            tn = float(t.pow(2).sum())
            stats["their_norm2"] += tn
            stats["n_tensors"] += 1
            if tn > 0:
                stats["n_changed"] += 1
                # group by transformer layer index when present
                parts = k.split(".")
                layer = next((parts[i + 1] for i, p in enumerate(parts) if p == "layers"), "other")
                stats["per_layer"][layer] += tn
            if ours is not None and k in our_keys:
                o = ours.get_tensor(k).to(torch.float64) - b
                stats["our_norm2"] += float(o.pow(2).sum())
                stats["dot"] += float((t * o).sum())
            del b, t
        if ours is not None:
            ours.__exit__(None, None, None)
    return stats


def cmd_compare(args) -> int:
    token = os.environ.get("HF_TOKEN_CORPUS") or os.environ.get("HF_TOKEN")
    if not token:
        print("need HF_TOKEN_CORPUS or HF_TOKEN in env")
        return 1

    data = fetch_leaderboard()
    rows = rows_from(data)
    scored = sorted([r for r in rows if r["val_loss"] is not None and r["uid"] not in OUR_UIDS],
                    key=lambda r: r["val_loss"])
    rivals = [r for r in scored if r["repo"]][: args.top]
    if not rivals:
        print("no scored rivals with a published repo")
        return 1

    # the global checkpoint everyone trains from this round
    base_repo, base_file = "g-connito/co", f"model_expgroup_{args.group}.safetensors"
    print(f"downloading global checkpoint {base_repo}/{base_file} ...", flush=True)
    base = _download(base_repo, base_file, None, token)

    ours_path = None
    if args.ours_repo:
        print(f"downloading ours {args.ours_repo} ...", flush=True)
        try:
            ours_path = _download(args.ours_repo, base_file, None, token)
        except Exception as exc:  # noqa: BLE001
            print(f"  could not fetch ours: {type(exc).__name__}")

    our_ref = None
    print(f"\n{'uid':<6}{'val_loss':<10}{'cohort':<8}{'||delta||':<12}{'cos(us)':<10}{'repo'}")
    for r in rivals:
        try:
            p = _download(r["repo"], base_file, r["rev"], token)
        except Exception as exc:  # noqa: BLE001
            print(f"{r['uid']:<6}{r['val_loss']:<10.4f}{str(r['cohort']):<8}"
                  f"{'-':<12}{'-':<10}{r['repo']}  ({type(exc).__name__})")
            continue
        s = _delta_stats(p, base, ours_path)
        their = s["their_norm2"] ** 0.5
        if our_ref is None and s["our_norm2"] > 0:
            our_ref = s["our_norm2"] ** 0.5
        cos = (s["dot"] / ((s["their_norm2"] ** 0.5) * (s["our_norm2"] ** 0.5))
               if s["their_norm2"] > 0 and s["our_norm2"] > 0 else None)
        print(f"{r['uid']:<6}{r['val_loss']:<10.4f}{str(r['cohort']):<8}"
              f"{their:<12.4f}{(f'{cos:+.4f}' if cos is not None else '-'):<10}{r['repo']}")
        if args.layers:
            top = sorted(s["per_layer"].items(), key=lambda kv: -kv[1])[:6]
            tot = sum(s["per_layer"].values()) or 1.0
            frag = "  ".join(f"L{k}:{100*v/tot:.0f}%" for k, v in top)
            print(f"      tensors changed {s['n_changed']}/{s['n_tensors']}   {frag}")

    if our_ref is not None:
        print(f"\nour ||delta|| = {our_ref:.4f}")
        print("  cos > 0  same direction as us (differ in magnitude / data)")
        print("  cos ~ 0  different objective -- they are optimising something we are not")
    return 0


def _our_config(uid: int) -> dict:
    """What this uid is actually configured to do, read from the files that drive it."""
    ops = Path("/root/SN102/ops")
    lr_file = ops / f"lr_override_{uid}.json"
    if not lr_file.exists():
        lr_file = ops / "lr_override.json"
    lr = {}
    try:
        lr = json.loads(lr_file.read_text())
    except Exception:  # noqa: BLE001
        pass
    ds_file = ops / f"dataset_{uid}.txt"
    ds = ds_file.read_text().strip() if ds_file.exists() else "(group config)"
    if ds in ("default", "stream", "streaming", "none"):
        ds = "HF streaming"
    return {"peak_lr": lr.get("peak_lr"), "floor": lr.get("min_frac"),
            "dataset": ds, "lr_file": lr_file.name}


def cmd_versus(args) -> int:
    """Side-by-side: our miners against the earners and the val_loss leaders.

    The point is to make the two questions separable at a glance -- are we
    losing on LOSS, or on COHORT? They are different problems with different
    fixes, and the leaderboard routinely has the best-loss miner earning zero.
    """
    data = fetch_leaderboard()
    base = (data.get("round") or {}).get("baseline_loss")
    ph = data.get("phase") or {}
    meta = data.get("meta") or {}
    rows = rows_from(data)
    lb = data["leaderboard"]
    lb = lb if isinstance(lb, list) else list(lb.values())
    detail = {m["uid"]: _val_loss_detail(m) for m in lb}

    print(f"cycle={ph.get('cycle_index')} phase={ph.get('name')} baseline={base}")
    if meta:
        print(f"validators: {meta.get('polled_validator_count')}/{meta.get('validator_count')} polled, "
              f"contributing={meta.get('contributing_validators')}"
              f"{'  STALE: ' + str(meta.get('stale_reason')) if meta.get('stale') else ''}")

    scored = sorted([r for r in rows if r["val_loss"] is not None], key=lambda r: r["val_loss"])
    earners = sorted([r for r in rows if r["incentive"] > 0], key=lambda r: -r["incentive"])

    print(f"\n=== OURS ===")
    for uid in sorted(OUR_UIDS):
        r = next((x for x in rows if x["uid"] == uid), None)
        if r is None:
            continue
        cfg = _our_config(uid)
        rank = next((i for i, x in enumerate(scored, 1) if x["uid"] == uid), None)
        vl = f"{r['val_loss']:.4f}" if r["val_loss"] is not None else "-"
        dl = f"{base - r['val_loss']:+.4f}" if (base and r["val_loss"] is not None) else "-"
        print(f"  uid {uid}: val_loss={vl:<9} delta={dl:<9} rank={rank}/{len(scored)} "
              f"cohort={r['cohort']} inc={r['incentive']:.5f}")
        print(f"           lr={cfg['peak_lr']} floor={cfg['floor']} data={cfg['dataset']}")
        if detail.get(uid):
            print(f"           {'  '.join(detail[uid])}")

    print(f"\n=== EARNERS ({len(earners)}) -- what it takes to be paid ===")
    for r in earners[:6]:
        vl = f"{r['val_loss']:.4f}" if r["val_loss"] is not None else "-"
        print(f"  uid {r['uid']:<5} inc={r['incentive']:.5f} cohort={str(r['cohort']):<6} "
              f"val_loss={vl:<9} {r['repo'] or '-'}")

    print(f"\n=== BEST LOSS (top {args.top}) -- note how many earn nothing ===")
    for i, r in enumerate(scored[: args.top], 1):
        print(f"  {i:>3}. uid {r['uid']:<5} {r['val_loss']:.4f}  cohort={str(r['cohort']):<6} "
              f"inc={r['incentive']:.4f}  {r['repo'] or '-'}")

    if earners:
        e_losses = [r["val_loss"] for r in earners if r["val_loss"] is not None]
        ours = [r["val_loss"] for r in rows if r["uid"] in OUR_UIDS and r["val_loss"] is not None]
        if e_losses and ours:
            print(f"\ngap to the worst-scoring EARNER: {min(ours) - max(e_losses):+.4f}")
            print("  (positive = we must improve loss; negative = loss is fine, cohort is the blocker)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("snapshot", help="append current leaderboard to the JSONL")
    sub.add_parser("history", help="trajectories, cohort moves, who earns")

    v = sub.add_parser("versus", help="our miners vs earners and loss leaders")
    v.add_argument("--top", type=int, default=8)

    c = sub.add_parser("compare", help="tensor-level diff against top miners")
    c.add_argument("--top", type=int, default=3)
    c.add_argument("--group", type=int, default=4)
    c.add_argument("--ours-repo", default="solvc4u/co2")
    c.add_argument("--layers", action="store_true", help="per-layer breakdown")

    args = ap.parse_args()
    return {"snapshot": cmd_snapshot, "history": cmd_history,
            "versus": cmd_versus, "compare": cmd_compare}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
