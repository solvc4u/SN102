#!/usr/bin/env python3
"""Change repo, training data or LR on a RUNNING miner. No restart.

    python -m ops.setcfg show
    python -m ops.setcfg repo   250 solvc4u/sn102-250   # where we publish
    python -m ops.setcfg data   250 stream              # HF streaming
    python -m ops.setcfg data   250 nvidia/Nemotron-CC-Math-v1:4plus
    python -m ops.setcfg lr     250 auto                # hand LR to the tuner
    python -m ops.setcfg lr     250 3e-5                # pin it

Everything here writes a small control file that the miner re-reads on its own
schedule -- LR at each cycle boundary, repo and squash on each submission. A
restart is what costs a round (the scheduler always waits for the next
Distribute), so a config change must never require one.

`data` is the one that needs care. Which dataset the miner streams comes from
the expert-group config, which is shared by every uid and rebuilt from disk on
every load, so a per-uid value in the miner yaml is silently discarded. Setting
a source therefore writes BOTH:

  * ops/dataset_<uid>.txt        - selects streaming vs a local class (env)
  * ops/datasource_<uid>.json    - the source spec, applied by ops.shared_dataset

When the subnet moves to a different dataset repo, `data <uid> <repo>:<subset>`
is the whole change.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

OPS = Path("/root/SN102/ops")
UIDS = (250, 178, 121)


def _write(path: Path, payload) -> None:
    path.write_text(payload if isinstance(payload, str) else json.dumps(payload, indent=1) + "\n")
    print(f"  wrote {path.name}: {payload if isinstance(payload, str) else json.dumps(payload)}")


def cmd_repo(uid: int, value: str) -> int:
    """Publish to `value`, resetting its history before each upload."""
    _write(OPS / f"hf_repo_{uid}.json", {"repo": value, "squash_before_upload": True})
    print(f"  uid {uid} will publish to {value}, history squashed before each push")
    print("  (takes effect on the next submission -- no restart)")
    return 0


def cmd_data(uid: int, value: str) -> int:
    if value in ("stream", "streaming", "default", "hf"):
        _write(OPS / f"dataset_{uid}.txt", "default")
        (OPS / f"datasource_{uid}.json").unlink(missing_ok=True)
        print(f"  uid {uid} streams the expert-group's configured sources from HF")
    elif value in ("local",):
        _write(OPS / f"dataset_{uid}.txt", "ops.shared_dataset:LocalSharedDataset")
        print(f"  uid {uid} reads the local corpus under data/corpus/")
    else:
        # "<repo>[:<subset>]" -- override which HF dataset is streamed
        repo, _, subset = value.partition(":")
        _write(OPS / f"dataset_{uid}.txt", "default")
        _write(OPS / f"datasource_{uid}.json",
               {"path": repo, "name": subset or None, "text_column": "text"})
        print(f"  uid {uid} streams {repo}" + (f" / {subset}" if subset else ""))
    print("  (dataset source is read when the dataloader is rebuilt -- next epoch or next cycle)")
    return 0


def cmd_lr(uid: int, value: str) -> int:
    path = OPS / f"lr_override_{uid}.json"
    if value in ("auto", "automatic", "tuner"):
        path.unlink(missing_ok=True)
        print(f"  uid {uid}: LR handed to the tuner (removed {path.name})")
    else:
        _write(path, {"peak_lr": float(value), "warmup": 0.03, "min_frac": 0.02})
        print(f"  uid {uid}: LR pinned at {value}")
    print("  (applies at the next cycle boundary -- no restart)")
    return 0


def cmd_startlr(uid: int, value: str) -> int:
    """Where this uid's LR search begins before it has a scored round.

    Read at process start (it is an env var), so unlike `lr` this one needs a
    restart to change -- and it only matters until the first scored round, after
    which the tuner takes over regardless.
    """
    path = OPS / f"start_lr_{uid}.txt"
    if value in ("clear", "default", "none"):
        path.unlink(missing_ok=True)
        print(f"  uid {uid}: start LR cleared (falls back to CONNITO_START_LR)")
    else:
        float(value)  # reject typos before they reach a miner
        _write(path, value)
        print(f"  uid {uid}: LR search will start at {value}")
    print("  (read at process start -- applies on next start/restart)")
    return 0


def cmd_show(*_a) -> int:
    for uid in UIDS:
        lr_file = OPS / f"lr_override_{uid}.json"
        shared = OPS / "lr_override.json"
        if lr_file.exists():
            lr = json.loads(lr_file.read_text()).get("peak_lr")
        elif shared.exists():
            lr = f"{json.loads(shared.read_text()).get('peak_lr')} (shared)"
        else:
            lr = "auto (tuner)"
        ds_file = OPS / f"dataset_{uid}.txt"
        ds = ds_file.read_text().strip() if ds_file.exists() else "(group config)"
        ds = "HF streaming" if ds in ("default", "stream", "streaming") else ds
        src_file = OPS / f"datasource_{uid}.json"
        if src_file.exists():
            s = json.loads(src_file.read_text())
            ds += f" [{s.get('path')}" + (f"/{s['name']}" if s.get("name") else "") + "]"
        repo_file = OPS / f"hf_repo_{uid}.json"
        if repo_file.exists():
            r = json.loads(repo_file.read_text())
            repo = f"{r.get('repo') or '(config)'}" + (" +squash" if r.get("squash_before_upload") else "")
        else:
            repo = "(config)"
        sl_file = OPS / f"start_lr_{uid}.txt"
        sl = sl_file.read_text().strip() if sl_file.exists() else "-"
        print(f"  uid {uid}:  lr={lr:<18} start={sl:<8} data={ds:<24} repo={repo}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("show", help="current settings for every uid")
    for name, helptext in (("repo", "publish target"), ("data", "training source"),
                           ("lr", "learning rate"), ("startlr", "initial LR for the tuner")):
        p = sub.add_parser(name, help=helptext)
        p.add_argument("uid", type=int)
        p.add_argument("value")
    a = ap.parse_args()
    if a.cmd == "show":
        return cmd_show()
    if a.uid not in UIDS:
        print(f"unknown uid {a.uid}; expected one of {UIDS}", file=sys.stderr)
        return 1
    return {"repo": cmd_repo, "data": cmd_data, "lr": cmd_lr,
            "startlr": cmd_startlr}[a.cmd](a.uid, a.value)


if __name__ == "__main__":
    raise SystemExit(main())
