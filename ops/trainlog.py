#!/usr/bin/env python3
"""Per-cycle training record, built to answer one question: why is our val_loss
2.8-3.1 when the miners being paid sit near 1.56?

    python -m ops.trainlog record      # one row per uid per cycle -> JSONL
    python -m ops.trainlog report      # what correlates with a better score
    python -m ops.trainlog gap         # local eval vs validator, per uid

Why a separate log at all. The pieces needed to diagnose this live in four
places that are never joined: the trainer log (LR, steps, local eval), the
commit log (what was uploaded and when), the chain (what we advertised), and the
dashboard (what a validator actually scored). Reading them one at a time is how
several wrong conclusions got made today -- a number from one source compared
against a number from another taken at a different moment.

The central anomaly this exists to explain:

    uid 250   local eval 1.64      validator 3.11      baseline 1.69

Locally the model looks competitive with the leaders. On the validator's seeded
slice it is nearly twice as bad. Both cannot be describing the same model on the
same distribution, and the difference between them is the whole problem -- every
lever tuned so far (LR, corpus, rank, repo) was aimed at the local number, which
does not predict the one that scores.

`gap` measures that directly by evaluating the SAME checkpoint two ways: on the
slice we train on, and on a validator-style seeded slice drawn from the full
dataset. A large gap means overfitting to our stripe; a small gap means the
problem is elsewhere and the local number was simply never comparable.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

LOG_DIR = Path(os.environ.get("LOG_DIR", "/root/SN102/logs"))
OPS = Path("/root/SN102/ops")
TRAINLOG = LOG_DIR / "trainlog.jsonl"
UIDS = (250, 178, 121)
ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _tail(path: Path, nbytes: int = 3_000_000) -> str:
    if not path.exists():
        return ""
    with path.open("rb") as fh:
        fh.seek(max(0, path.stat().st_size - nbytes))
        return ANSI.sub("", fh.read().decode("utf-8", "replace"))


def _last(pattern: str, text: str, group: int = 1):
    m = None
    for m in re.finditer(pattern, text):
        pass
    return m.group(group) if m else None


def _local_eval(uid: int) -> tuple[int | None, float | None]:
    """Newest (step, val_loss) our own recorder wrote."""
    for d in Path("/root/SN102/data/checkpoints/miner").glob(f"*/*/uid{uid}/*/local_eval.json"):
        try:
            data = json.loads(d.read_text())
            if data:
                k = max(int(x) for x in data)
                return k, float(data[str(k)])
        except Exception:  # noqa: BLE001
            continue
    return None, None


def collect_uid(uid: int, board: dict | None) -> dict:
    train = _tail(LOG_DIR / f"uid{uid}-train.log")
    commit = _tail(LOG_DIR / f"uid{uid}-commit.log")
    step, local = _local_eval(uid)

    row = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "uid": uid,
        # --- configuration actually in force, read back from the process log
        "peak_lr": _last(r"peak_lr=([0-9.e+-]+)", train),
        "data_rank": _last(r"rank overridden from CONNITO_DATA_RANK.*?rank=(\d+)", train),
        "dataset_class": _last(r"dataset_class overridden from CONNITO_DATASET_CLASS dataset_class=(\S+)", train),
        "expert_group": _last(r'"expert_group_name":\s*"([^"]+)"', train),
        "inner_opt_step": _last(r"inner_opt_step=(\d+)", train),
        # --- our own eval
        "local_eval_step": step,
        "local_eval": local,
        # --- what we actually shipped
        "upload_repo": _last(r"Uploaded checkpoint to HF\s+repo_id=(\S+)", commit),
        "upload_rev": _last(r"Uploaded checkpoint to HF.*?revision=(\S+)", commit),
        "chain_committed": bool(re.search(r"Committed status to chain", commit)),
        "distribute_failed": bool(re.search(r"All download attempts failed", commit[-200_000:])),
        "global_ver_used": _last(r"'v': (\d+)", commit),
    }

    if board:
        for m in board.get("_rows", []):
            if m["uid"] == uid:
                row["validator_val_loss"] = m["val_loss"]
                row["cohort"] = m["cohort"]
                row["incentive"] = m["incentive"]
                break
        row["baseline"] = board.get("_baseline")
        row["cycle"] = board.get("_cycle")
        if row.get("validator_val_loss") is not None and row.get("baseline") is not None:
            row["delta"] = row["baseline"] - row["validator_val_loss"]
        if row.get("validator_val_loss") is not None and local is not None:
            # The number this log exists to explain.
            row["local_vs_validator"] = row["validator_val_loss"] - local
    return row


def _board() -> dict | None:
    try:
        sys.path.insert(0, "/root/SN102")
        from ops.rivals import fetch_leaderboard, rows_from
        data = fetch_leaderboard()
        rows = rows_from(data)
        return {
            "_rows": rows,
            "_baseline": (data.get("round") or {}).get("baseline_loss"),
            "_cycle": (data.get("phase") or {}).get("cycle_index"),
            "_scored": sorted([r for r in rows if r["val_loss"] is not None],
                              key=lambda r: r["val_loss"]),
        }
    except Exception as exc:  # noqa: BLE001
        print(f"  dashboard unavailable ({type(exc).__name__}); recording local fields only")
        return None


def cmd_record(_a) -> int:
    board = _board()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with TRAINLOG.open("a") as fh:
        for uid in UIDS:
            row = collect_uid(uid, board)
            fh.write(json.dumps(row) + "\n")
            vl = row.get("validator_val_loss")
            print(f"  uid {uid}: lr={row['peak_lr']} rank={row['data_rank']} "
                  f"step={row['inner_opt_step']} local={row['local_eval']} "
                  f"validator={vl if vl is not None else '-'} "
                  f"gap={row.get('local_vs_validator')}")
    if board and board["_scored"]:
        top = board["_scored"][:3]
        print(f"  field top-3: " + "  ".join(f"uid{r['uid']}={r['val_loss']:.4f}" for r in top))
    return 0


def cmd_report(_a) -> int:
    if not TRAINLOG.exists():
        print("no trainlog yet -- run `python -m ops.trainlog record` on a timer")
        return 1
    rows = [json.loads(l) for l in TRAINLOG.read_text().splitlines() if l.strip()]
    scored = [r for r in rows if r.get("validator_val_loss") is not None]
    print(f"{len(rows)} records, {len(scored)} with a validator score\n")
    if not scored:
        print("  nothing scored yet -- no correlation can be computed")
        return 0

    by_uid = defaultdict(list)
    for r in scored:
        by_uid[r["uid"]].append(r)
    print("=== per uid ===")
    for uid, rs in sorted(by_uid.items()):
        best = min(rs, key=lambda r: r["validator_val_loss"])
        print(f"  uid {uid}: {len(rs)} scored, best validator val_loss "
              f"{best['validator_val_loss']:.4f} at lr={best['peak_lr']} rank={best['data_rank']}")

    print("\n=== does a setting track the score? ===")
    for field in ("peak_lr", "data_rank", "dataset_class"):
        groups = defaultdict(list)
        for r in scored:
            if r.get(field) is not None:
                groups[str(r[field])].append(r["validator_val_loss"])
        if len(groups) < 2:
            print(f"  {field:<14} only one value observed -- nothing to compare")
            continue
        print(f"  {field}:")
        for val, losses in sorted(groups.items(), key=lambda kv: sum(kv[1]) / len(kv[1])):
            print(f"     {val:<22} mean {sum(losses)/len(losses):.4f}  n={len(losses)}")

    gaps = [r["local_vs_validator"] for r in scored if r.get("local_vs_validator") is not None]
    if gaps:
        print(f"\n=== local eval vs validator ===")
        print(f"  mean gap {sum(gaps)/len(gaps):+.4f} over {len(gaps)} scored records")
        print("  positive = the validator scores us WORSE than our own eval does.")
        print("  a large stable gap means we are tuning against a number that does")
        print("  not predict the one that pays -- run `trainlog gap` to localise it.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("record", help="append one row per uid for this cycle")
    sub.add_parser("report", help="what correlates with a better validator score")
    a = ap.parse_args()
    return {"record": cmd_record, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
