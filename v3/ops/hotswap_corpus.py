#!/usr/bin/env python3
"""
Swap in a larger corpus WITHOUT restarting the miners.

    python -m ops.hotswap_corpus \
        --src  /root/SN102/data/corpus-next/exp_legal/joelniklaus__Multi_Legal_Pile__all_all \
        --dest /root/SN102/data/corpus/exp_legal/joelniklaus__Multi_Legal_Pile__all_all

Why this works
--------------
ops/shared_dataset.py:_ParquetRowStream._source_rows captures the shard PATH
list once at dataloader construction, then loops forever:

    shards = list(source["shards"])      # fixed at startup
    while True:
        for shard in shards:
            pf = pq.ParquetFile(shard)   # re-opened by path every pass

Adding new files is therefore invisible to a running miner -- they are not in
the captured list. But REPLACING the contents at the existing paths is picked
up on the next pass, because each shard is re-opened by name.

So this repacks the source rows into exactly as many files as the destination
already has, using the same names, and swaps each in with os.replace(). That is
atomic on one filesystem: a miner mid-read keeps its open inode until it closes
the file, and its next open gets the new one. No torn reads, no missing paths,
no restart.

Constraint: the row count per file rises (here ~160k vs ~50k) because we must
not change the FILE COUNT. Parquet handles that fine; row groups keep memory
flat on the read side.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="directory holding the new parquet files")
    ap.add_argument("--dest", required=True, help="live directory the miners are reading")
    ap.add_argument("--row-group", type=int, default=20_000)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    import pyarrow as pa
    import pyarrow.parquet as pq

    src, dest = Path(args.src), Path(args.dest)
    src_files = sorted(src.glob("part-*.parquet"))
    dest_files = sorted(dest.glob("part-*.parquet"))
    if not src_files:
        sys.exit(f"no parquet files in {src}")
    if not dest_files:
        sys.exit(f"no parquet files in {dest} -- nothing to swap into")

    src_rows = sum(pq.read_metadata(f).num_rows for f in src_files)
    dest_rows = sum(pq.read_metadata(f).num_rows for f in dest_files)
    n_out = len(dest_files)
    per_file = -(-src_rows // n_out)  # ceil

    print(f"source : {len(src_files):>4} files, {src_rows:>10,} rows")
    print(f"live   : {len(dest_files):>4} files, {dest_rows:>10,} rows")
    print(f"repack : {n_out} files x ~{per_file:,} rows  (file COUNT must not change)")
    if args.dry_run:
        print("dry run -- nothing written")
        return 0

    tmp_dir = dest.parent / (dest.name + ".swapin")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    out_idx = 0
    writer = None
    written_in_file = 0
    written_total = 0
    produced: list[Path] = []

    def open_writer(i: int):
        path = tmp_dir / dest_files[i].name
        return pq.ParquetWriter(path, pa.schema([("text", pa.string())]), compression="zstd"), path

    writer, cur_path = open_writer(out_idx)
    for f in src_files:
        pf = pq.ParquetFile(f)
        for rg in range(pf.num_row_groups):
            tbl = pf.read_row_group(rg, columns=["text"])
            col = tbl.column("text")
            start = 0
            while start < len(col):
                room = per_file - written_in_file
                take = min(room, len(col) - start)
                writer.write_table(pa.table({"text": col.slice(start, take)}))
                written_in_file += take
                written_total += take
                start += take
                if written_in_file >= per_file and out_idx < n_out - 1:
                    writer.close()
                    produced.append(cur_path)
                    out_idx += 1
                    writer, cur_path = open_writer(out_idx)
                    written_in_file = 0
        print(f"  packed {written_total:>10,} rows  (file {out_idx+1}/{n_out}, {time.time()-t0:.0f}s)",
              flush=True)
    writer.close()
    produced.append(cur_path)

    if len(produced) != n_out:
        # A short write would leave paths the miners still expect. Refuse rather
        # than swap in a partial set -- a missing path is a crash mid-Train.
        for p in produced:
            p.unlink(missing_ok=True)
        sys.exit(f"produced {len(produced)} files but need exactly {n_out}; aborted, live corpus untouched")

    print(f"\nrepacked {written_total:,} rows into {len(produced)} files ({time.time()-t0:.0f}s)")
    print("swapping atomically...")
    for p in produced:
        target = dest / p.name
        os.replace(p, target)          # atomic within one filesystem
    tmp_dir.rmdir()
    final = sum(pq.read_metadata(f).num_rows for f in sorted(dest.glob("part-*.parquet")))
    print(f"live corpus now {final:,} rows across {len(dest_files)} files "
          f"(was {dest_rows:,}) -- miners pick this up on their next shard pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
