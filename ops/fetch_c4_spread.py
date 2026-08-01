#!/usr/bin/env python3
"""
Re-fetch the c4 half of the corpus, spread across the shard space.

    python -m ops.fetch_c4_spread --rows 800000 --shards 48

Why
---
`fetch_corpus.py` streamed c4 and stopped after N rows, so our c4 sample comes
from the first shards only. The validator picks ONE shard uniformly from the
~1024 `en/c4-train.NNNNN-of-01024.json.gz` files
(eval_shard_pick._KNOWN_SOURCES[("allenai/c4","en")], row_count_source=
"constant", safe_floor_rows=340_000), so any given round may draw shard 700
while we trained only on shards 0-2.

This is the same bug class that made the legal corpus 93% Czech/Bulgarian. It
matters less here -- c4 is homogeneous English web text and the shards are
publisher-balanced at ~356k rows each -- but "less" is not "not at all", and
the fix is cheap.

Strategy: sample evenly from a spread of shard indices rather than the head of
the stream, taking an even stride within each shard.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import time
from pathlib import Path

REPO = "allenai/c4"
TOTAL_SHARDS = 1024
SHARD_ROWS = 356_317  # from the policy's verified spot-checks


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=800_000)
    ap.add_argument("--shards", type=int, default=48,
                    help="how many of the 1024 shards to sample from")
    ap.add_argument("--rows-per-file", type=int, default=50_000)
    ap.add_argument("--corpus-dir", default=os.environ.get("CORPUS_DIR", "/root/SN102/data/corpus"))
    ap.add_argument("--expert-group", default=os.environ.get("EXPERT_GROUP", "exp_legal"))
    ap.add_argument("--min-chars", type=int, default=200)
    args = ap.parse_args()

    import pyarrow as pa
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    # Even spread across the shard index space, not the first N shards.
    step = max(1, TOTAL_SHARDS // args.shards)
    picks = [i for i in range(0, TOTAL_SHARDS, step)][: args.shards]
    per_shard = max(1, args.rows // len(picks))
    stride = max(1, SHARD_ROWS // per_shard)
    print(f"sampling {len(picks)} shards spread over {TOTAL_SHARDS} "
          f"(indices {picks[0]}, {picks[1]}, ... {picks[-1]})")
    print(f"{per_shard:,} rows/shard, stride {stride} within each -> ~{args.rows:,} total\n")

    token = os.environ.get("HF_TOKEN_CORPUS") or os.environ.get("HF_TOKEN")
    out_dir = Path(args.corpus_dir) / args.expert_group / "allenai__c4__en"
    staging = out_dir.with_name(out_dir.name + ".new")
    staging.mkdir(parents=True, exist_ok=True)

    kept_total = 0
    file_idx = 0
    buf: list[str] = []
    t0 = time.time()

    def flush():
        nonlocal buf, file_idx
        if not buf:
            return
        pq.write_table(pa.table({"text": pa.array(buf, type=pa.string())}),
                       staging / f"part-{file_idx:05d}.parquet", compression="zstd")
        file_idx += 1
        buf = []

    for n, shard in enumerate(picks):
        fname = f"en/c4-train.{shard:05d}-of-{TOTAL_SHARDS:05d}.json.gz"
        try:
            local = hf_hub_download(repo_id=REPO, repo_type="dataset", filename=fname,
                                    revision="main", token=token)
        except Exception as exc:  # noqa: BLE001
            print(f"  SKIP {fname}: {str(exc)[:80]}")
            continue
        kept = 0
        try:
            with gzip.open(local, "rt", encoding="utf-8", errors="ignore") as fh:
                for i, line in enumerate(fh):
                    if i % stride:
                        continue
                    try:
                        text = json.loads(line).get("text") or ""
                    except Exception:  # noqa: BLE001
                        continue
                    if len(text) < args.min_chars:
                        continue
                    buf.append(text)
                    kept += 1
                    kept_total += 1
                    if len(buf) >= args.rows_per_file:
                        flush()
                    if kept >= per_shard:
                        break
        except Exception as exc:  # noqa: BLE001
            print(f"  ERROR {fname}: {str(exc)[:80]}")
        if n % 8 == 0 or kept < per_shard:
            print(f"  shard {shard:>5}: {kept:>7,}  [{kept_total:,} total, {time.time()-t0:.0f}s]",
                  flush=True)
    flush()

    if kept_total == 0:
        sys.exit("no rows collected -- refusing to replace the existing corpus")

    backup = out_dir.with_name(out_dir.name + ".old")
    if out_dir.exists():
        if backup.exists():
            import shutil
            shutil.rmtree(backup)
        out_dir.rename(backup)
    staging.rename(out_dir)
    print(f"\nswapped in {kept_total:,} rows across {file_idx} parquet files")

    man_path = Path(args.corpus_dir) / args.expert_group / "manifest.json"
    man = json.loads(man_path.read_text())
    for s in man["sources"]:
        if s["path"] == REPO:
            s.update(rows=kept_total, shards=file_idx,
                     sampling=f"spread-{len(picks)}-of-{TOTAL_SHARDS}-shards")
    man_path.write_text(json.dumps(man, indent=2))
    print(f"manifest updated: {man_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
