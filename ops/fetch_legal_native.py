#!/usr/bin/env python3
"""
Re-fetch the legal corpus from the SAME pool the validator evaluates on.

    python -m ops.fetch_legal_native --rows 2500000

Why this replaces the Multi_Legal_Pile half of ops/fetch_corpus.py
------------------------------------------------------------------
`fetch_corpus.py` streamed `load_dataset("joelniklaus/Multi_Legal_Pile",
"all_all", streaming=True)` and stopped after N rows. That builder emits data
grouped by language in alphabetical order, so a head-of-stream prefix is not a
sample of the mixture -- it is whatever sorts first. Measured on what we
actually downloaded:

    cs (Czech)   ~69%      en   0.7%
    bg (Bulgarian) 24%     de   0.4%
    pt            4.9%     fr   0.2%

The validator draws its eval slice from the repo's NATIVE `data/**.jsonl.xz`
files at a pinned revision (connito/shared/eval_shard_pick.py:_KNOWN_SOURCES),
where Portuguese/Brazilian caselaw is 6.7M of the rows -- 67% of the corpus by
bytes -- and Bulgarian is ~29k rows, well under 1%.

So we were optimising Bulgarian and Czech while being scored on mostly
Portuguese. That is consistent with every measurement: update magnitude matched
the leaders (rel-L2 0.0036 vs 0.0056) while direction was uncorrelated
(cosine -0.03) and val_loss sat ABOVE baseline -- i.e. our training made the
model worse on the distribution being scored.

This script instead samples the native files in proportion to the verified row
counts, so the training distribution matches the eval pool.

The row counts and the file allowlist are read directly from
`eval_shard_pick._KNOWN_SOURCES` rather than duplicated, so this cannot drift
away from what the validator actually uses.
"""

from __future__ import annotations

import argparse
import io
import json
import lzma
import os
import random
import sys
import time
from pathlib import Path

SOURCE = ("joelniklaus/Multi_Legal_Pile", "all_all")


def _policy():
    sys.path.insert(0, os.environ.get("CONNITO_ROOT", "/root/SN102/Connito"))
    from connito.shared.eval_shard_pick import _KNOWN_SOURCES

    pol = _KNOWN_SOURCES.get(SOURCE)
    if pol is None:
        sys.exit("Multi_Legal_Pile policy not found in eval_shard_pick._KNOWN_SOURCES")
    return pol


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=2_500_000,
                    help="total rows to keep, split across shards by verified row count")
    ap.add_argument("--rows-per-shard-file", type=int, default=50_000)
    ap.add_argument("--corpus-dir", default=os.environ.get("CORPUS_DIR", "/root/SN102/data/corpus"))
    ap.add_argument("--expert-group", default=os.environ.get("EXPERT_GROUP", "exp_legal"))
    ap.add_argument("--min-chars", type=int, default=200,
                    help="must match DataCfg.eval_min_text_chars")
    args = ap.parse_args()

    import pyarrow as pa
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    pol = _policy()
    table = dict(pol.verified_shard_rows)
    total_rows = sum(table.values())
    print(f"pinned revision : {pol.revision}")
    print(f"eval allowlist  : {len(table)} shards, {total_rows:,} verified rows")

    # UNIFORM per shard, NOT proportional to row count.
    #
    # eval_shard_pick.pick_shard_for_source picks the eval shard as
    #     shard_idx = h256_int(...) % len(shards)
    # i.e. one shard drawn UNIFORMLY from the 29-shard allowlist each round.
    # A 10,556-row Belgian file is therefore evaluated exactly as often as a
    # 3.5M-row Brazilian one. Weighting training by row count over-represents
    # Portuguese 3.4x (91% of rows but only 24% of shards) and starves French
    # and German, which are 5 and 4 shards -> 31% of eval rounds combined.
    #
    # Small shards cap out at their available rows; the remainder is
    # redistributed across shards that still have headroom so the total is met.
    per = args.rows // len(table)
    quotas = {f: min(per, n) for f, n in table.items()}
    short = args.rows - sum(quotas.values())
    if short > 0:
        room = [f for f, n in table.items() if n > quotas[f]]
        while short > 0 and room:
            add = max(1, short // len(room))
            for f in list(room):
                take = min(add, table[f] - quotas[f], short)
                quotas[f] += take
                short -= take
                if quotas[f] >= table[f]:
                    room.remove(f)
                if short <= 0:
                    break
    print(f"target          : {sum(quotas.values()):,} rows, UNIFORM per shard "
          f"({per:,}/shard before capping)\n")
    for f, q in sorted(quotas.items(), key=lambda kv: -kv[1])[:6]:
        print(f"   {q:>9,}  {f}")
    print(f"   ... {len(quotas)-6} more shards\n")

    token = os.environ.get("HF_TOKEN_CORPUS") or os.environ.get("HF_TOKEN")
    out_dir = Path(args.corpus_dir) / args.expert_group / "joelniklaus__Multi_Legal_Pile__all_all"
    staging = out_dir.with_name(out_dir.name + ".new")
    staging.mkdir(parents=True, exist_ok=True)

    kept_total = 0
    shard_idx = 0
    buf: list[str] = []
    t0 = time.time()

    def flush():
        nonlocal buf, shard_idx
        if not buf:
            return
        pq.write_table(pa.table({"text": pa.array(buf, type=pa.string())}),
                       staging / f"part-{shard_idx:05d}.parquet", compression="zstd")
        shard_idx += 1
        buf = []

    for fname, quota in sorted(quotas.items()):
        try:
            local = hf_hub_download(repo_id=SOURCE[0], repo_type="dataset", filename=fname,
                                    revision=pol.revision, token=token)
        except Exception as exc:  # noqa: BLE001
            print(f"  SKIP {fname}: {str(exc)[:90]}")
            continue
        kept = 0
        # Reservoir-free: take an evenly spaced stride through the shard rather
        # than its head, so we sample the file rather than its first rows --
        # the exact mistake this script exists to correct.
        stride = max(1, table[fname] // quota)
        try:
            with lzma.open(local, "rt", encoding="utf-8", errors="ignore") as fh:
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
                    if len(buf) >= args.rows_per_shard_file:
                        flush()
                    if kept >= quota:
                        break
        except Exception as exc:  # noqa: BLE001
            print(f"  ERROR reading {fname}: {str(exc)[:90]}")
        print(f"  {kept:>8,} / {quota:<8,} {fname}   [{kept_total:,} total, {time.time()-t0:.0f}s]",
              flush=True)
    flush()

    if kept_total == 0:
        sys.exit("no rows collected -- refusing to replace the existing corpus")

    # Swap in only after a successful build, so a failure never leaves the
    # miners without a corpus.
    backup = out_dir.with_name(out_dir.name + ".old")
    if out_dir.exists():
        if backup.exists():
            import shutil
            shutil.rmtree(backup)
        out_dir.rename(backup)
    staging.rename(out_dir)
    print(f"\nswapped in {kept_total:,} rows across {shard_idx} parquet files")
    print(f"previous corpus kept at {backup}")

    # Refresh the manifest entry so ops/shared_dataset.py picks it up.
    man_path = Path(args.corpus_dir) / args.expert_group / "manifest.json"
    man = json.loads(man_path.read_text())
    for s in man["sources"]:
        if s["path"] == SOURCE[0]:
            s.update(rows=kept_total, shards=shard_idx, min_chars=args.min_chars,
                     sampling="uniform-per-shard", revision=pol.revision)
    man_path.write_text(json.dumps(man, indent=2))
    print(f"manifest updated: {man_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
