#!/usr/bin/env python3
"""
One-time shared corpus download for SN102.

Why this exists
---------------
The stock miner streams its training data (`load_dataset(..., streaming=True)`
in connito/shared/dataloader.py). Streaming responses are never written to the
local datasets cache, so N miners on one box = N independent request streams to
huggingface.co for the whole 300-block Train phase. That is the main source of
429s when running several UIDs from one machine.

This script materialises the exact source mixture the validator evaluates
against into local parquet, once. ops/shared_dataset.py then reads it off disk,
so Train makes zero HF dataset requests.

Sources are taken from the expert group's own config.yaml so this stays in sync
if the group definition changes. For exp_math that is:
    allenai/c4                    (en,    weight 0.5)
    nvidia/Nemotron-CC-Math-v1    (4plus, weight 0.5)

Usage
-----
    python ops/fetch_corpus.py --expert-group exp_legal --rows 2000000
    python ops/fetch_corpus.py --expert-group exp_legal --rows 2000000 --verify

Output layout
-------------
    $CORPUS_DIR/<expert_group>/<sanitised_source>/part-00000.parquet
    $CORPUS_DIR/<expert_group>/manifest.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import yaml


def _load_group_sources(connito_root: Path, group: str) -> tuple[list[dict], int, dict]:
    """Read dataset_sources + sequence_length straight from the group's config.yaml."""
    cfg_path = connito_root / "expert_groups" / group / "config.yaml"
    if not cfg_path.is_file():
        sys.exit(f"no such expert group config: {cfg_path}")
    cfg = yaml.safe_load(cfg_path.read_text())
    data = cfg.get("data") or {}
    sources = data.get("dataset_sources") or []
    if not sources:
        sys.exit(f"{cfg_path} declares no data.dataset_sources")
    return sources, int(data.get("sequence_length", 1024)), data


def _sanitise(path: str, name: str | None) -> str:
    return (path + ("__" + name if name else "")).replace("/", "__")


def _min_chars(data_cfg: dict) -> int:
    """
    Mirror the validator's data-quality gate exactly.

    `connito/shared/dataloader.py:365` reads `eval_min_text_chars` off the
    group's data config, defaulting to `DataCfg.eval_min_text_chars = 200`
    (config.py:319), and applies `_min_text_chars_filter` only on the seeded
    eval path.

    Do NOT be cleverer than this. An earlier version of this script filtered at
    `sequence_length * 4` (4096 chars), which sounds reasonable -- "drop rows
    too short to fill a window" -- but c4 documents average well under that, so
    it would have silently trained us on a long-document-only slice while the
    validator evaluates the full-length distribution. Proof-of-Loss scores us on
    the validator's mixture; any divergence here is a systematic handicap.
    """
    return int(data_cfg.get("eval_min_text_chars", 200) or 200)


def fetch_source(
    src: dict,
    out_dir: Path,
    rows: int,
    min_chars: int,
    token: str | None,
    rows_per_shard: int,
) -> dict:
    from datasets import load_dataset

    path = src["path"]
    name = src.get("name")
    text_column = src.get("text_column", "text")
    trust_remote_code = bool(src.get("trust_remote_code", False))

    out_dir.mkdir(parents=True, exist_ok=True)

    kwargs: dict = {"streaming": True, "split": "train"}
    if name:
        kwargs["name"] = name
    if trust_remote_code:
        # Only exp_legal's Multi_Legal_Pile needs this; it ships a builder
        # script. The flag is per-source in the group config precisely so it
        # does not leak to c4.
        kwargs["trust_remote_code"] = True
    if token:
        kwargs["token"] = token

    print(f"[fetch] {path}{'/' + name if name else ''} -> {out_dir}  (target {rows:,} rows)")
    ds = load_dataset(path, **kwargs)

    import pyarrow as pa
    import pyarrow.parquet as pq

    kept = 0
    skipped = 0
    shard_idx = 0
    buf: list[str] = []
    t0 = time.time()

    def flush() -> None:
        nonlocal buf, shard_idx
        if not buf:
            return
        table = pa.table({"text": pa.array(buf, type=pa.string())})
        pq.write_table(table, out_dir / f"part-{shard_idx:05d}.parquet", compression="zstd")
        shard_idx += 1
        buf = []

    for example in ds:
        text = example.get(text_column)
        if not isinstance(text, str) or len(text) < min_chars:
            skipped += 1
            continue
        buf.append(text)
        kept += 1
        if len(buf) >= rows_per_shard:
            flush()
            rate = kept / max(1e-6, time.time() - t0)
            print(f"  {kept:,}/{rows:,} rows  ({rate:,.0f}/s, {skipped:,} short rows dropped)")
        if kept >= rows:
            break
    flush()

    meta = {
        "path": path,
        "name": name,
        "text_column": text_column,
        "weight": float(src.get("weight", 1.0)),
        "rows": kept,
        "shards": shard_idx,
        "min_chars": min_chars,
        "seconds": round(time.time() - t0, 1),
    }
    print(f"[done] {path}: {kept:,} rows in {shard_idx} shards ({meta['seconds']}s)")
    return meta


def verify(group_dir: Path) -> int:
    """Re-open every shard and confirm it is readable and non-empty."""
    import pyarrow.parquet as pq

    manifest_path = group_dir / "manifest.json"
    if not manifest_path.is_file():
        print(f"FAIL: no manifest at {manifest_path}")
        return 1
    manifest = json.loads(manifest_path.read_text())
    bad = 0
    total = 0
    for entry in manifest["sources"]:
        d = group_dir / _sanitise(entry["path"], entry.get("name"))
        shards = sorted(d.glob("part-*.parquet"))
        if len(shards) != entry["shards"]:
            print(f"FAIL: {d} has {len(shards)} shards, manifest says {entry['shards']}")
            bad += 1
        n = 0
        for s in shards:
            try:
                n += pq.read_metadata(s).num_rows
            except Exception as exc:  # noqa: BLE001 - report and continue
                print(f"FAIL: unreadable shard {s}: {exc}")
                bad += 1
        total += n
        print(f"  ok  {entry['path']}: {n:,} rows across {len(shards)} shards")
    print(f"{'OK' if not bad else 'PROBLEMS'}: {total:,} rows total, {bad} problem(s)")
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--expert-group", default=os.environ.get("EXPERT_GROUP", "exp_legal"))
    ap.add_argument("--rows", type=int, default=2_000_000,
                    help="rows to keep PER SOURCE (default 2M)")
    ap.add_argument("--rows-per-shard", type=int, default=50_000)
    ap.add_argument("--connito-root", default=os.environ.get("CONNITO_ROOT", "/root/SN102/Connito"))
    ap.add_argument("--corpus-dir", default=os.environ.get("CORPUS_DIR", "/root/SN102/data/corpus"))
    ap.add_argument("--verify", action="store_true", help="verify an existing corpus and exit")
    args = ap.parse_args()

    connito_root = Path(args.connito_root)
    group_dir = Path(args.corpus_dir) / args.expert_group

    if args.verify:
        return verify(group_dir)

    sources, seq_len, data_cfg = _load_group_sources(connito_root, args.expert_group)
    token = os.environ.get("HF_TOKEN_CORPUS") or os.environ.get("HF_TOKEN")

    min_chars = _min_chars(data_cfg)
    print(f"min_text_chars gate: {min_chars} (matches validator eval filter)")

    group_dir.mkdir(parents=True, exist_ok=True)
    metas = []
    for src in sources:
        out_dir = group_dir / _sanitise(src["path"], src.get("name"))
        metas.append(fetch_source(src, out_dir, args.rows, min_chars, token, args.rows_per_shard))

    manifest = {
        "expert_group": args.expert_group,
        "sequence_length": seq_len,
        "rows_requested_per_source": args.rows,
        "sources": metas,
    }
    (group_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nmanifest -> {group_dir / 'manifest.json'}")
    print("next: point task.exp.data.dataset_class at ops.shared_dataset:LocalSharedDataset")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
