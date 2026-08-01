#!/usr/bin/env python3
"""
Per-shard weakness: which of the 29 eval shards do we lose rounds on?

    CUDA_VISIBLE_DEVICES=2 python -m ops.shard_probe --path ops/configs/uid250.yaml
    CUDA_VISIBLE_DEVICES=2 python -m ops.shard_probe --path ops/configs/uid250.yaml \
        --checkpoint /data/checkpoints/.../globalver_X_inneropt_Y

Why
---
`eval_shard_pick.pick_shard_for_source` draws ONE shard uniformly from the
29-shard allowlist each round:

    shard_idx = h256_int("eval_shard_pick", repo_id, str(name), int_seed) % len(shards)

so a round is Brazilian caselaw, or Bulgarian legislation, or Swiss French --
never a mixture. Measured on the live board, the FIELD MEDIAN swings 1.4510 ->
2.2066 (0.76) between rounds purely from which shard came up. Our own absolute
val_loss is therefore mostly not ours to control.

What we can control is the spread of our loss ACROSS shards. If we are strong
on Portuguese and weak on Swiss French, we lose every round Swiss French is
drawn. This measures loss per shard on the real eval files (native
`data/**.jsonl.xz` at the pinned revision), so we can see which shards drag us
down and up-weight them in the corpus.

Runs on the spare GPU against a saved checkpoint -- no miner is disturbed and
no cycle is spent.
"""

from __future__ import annotations

import argparse
import json
import lzma
import os
import statistics
import sys
import time
from pathlib import Path

import torch


def _policy():
    from connito.shared.eval_shard_pick import _KNOWN_SOURCES

    return _KNOWN_SOURCES[("joelniklaus/Multi_Legal_Pile", "all_all")]


def shard_rows(local_path: str, tokenizer, seq_len: int, n_rows: int, min_chars: int, stride: int):
    """Yield tokenised batches from one eval shard, sampled by stride."""
    kept = 0
    with lzma.open(local_path, "rt", encoding="utf-8", errors="ignore") as fh:
        for i, line in enumerate(fh):
            if i % stride:
                continue
            try:
                text = json.loads(line).get("text") or ""
            except Exception:  # noqa: BLE001
                continue
            if len(text) < min_chars:
                continue
            toks = tokenizer(text, truncation=True, max_length=seq_len,
                             padding="max_length", add_special_tokens=True)
            yield toks["input_ids"], toks["attention_mask"]
            kept += 1
            if kept >= n_rows:
                return


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True, help="miner config yaml")
    ap.add_argument("--checkpoint", default=None,
                    help="checkpoint dir to load; default = newest local")
    ap.add_argument("--rows-per-shard", type=int, default=192)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--out", default="/root/SN102/logs/shard_probe.json")
    args = ap.parse_args()

    from connito.shared.config import MinerConfig
    from connito.shared.expert_manager import ExpertManager
    from connito.shared.model import load_model, freeze_parameters
    from connito.shared.modeling.mycelia import get_base_tokenizer
    from connito.shared.chain import setup_chain_worker
    from connito.shared.checkpoints import select_best_checkpoint
    from connito.shared.checkpoint_helper import load_checkpoint
    from huggingface_hub import hf_hub_download

    cfg = MinerConfig.from_path(args.path, auto_update_config=True)
    device = torch.device("cuda:0")
    tok = get_base_tokenizer(cfg)
    seq_len = int(cfg.task.exp.data.sequence_length)

    wallet, subtensor, _ = setup_chain_worker(cfg, serve=False)
    em = ExpertManager(cfg)
    model, _ = load_model(0, cfg, em, subtensor, wallet, partial=True)
    model = model.to(device)
    model = freeze_parameters(model=model, expert_manager=em,
                             expert_group_id=cfg.task.exp.group_id, upcast_trainable=False)

    ckpt_dir = args.checkpoint
    if ckpt_dir is None:
        best = select_best_checkpoint(Path(cfg.ckpt.checkpoint_path))
        ckpt_dir = str(best.path) if best and best.path else None
    if ckpt_dir:
        print(f"loading checkpoint {ckpt_dir}", flush=True)
        try:
            load_checkpoint(config=cfg, checkpoint_path=Path(ckpt_dir), rank=0, device=device)
        except Exception as exc:  # noqa: BLE001
            print(f"  could not load checkpoint ({exc}); probing the global as-is")
    else:
        print("no local checkpoint; probing the downloaded global as-is")

    model.eval()
    pol = _policy()
    table = dict(pol.verified_shard_rows)
    token = os.environ.get("HF_TOKEN_CORPUS") or os.environ.get("HF_TOKEN")
    dtype = torch.bfloat16 if cfg.model.precision == "bf16-mixed" else torch.float16

    results = {}
    t0 = time.time()
    for n, (fname, nrows) in enumerate(sorted(table.items()), 1):
        try:
            local = hf_hub_download(repo_id="joelniklaus/Multi_Legal_Pile", repo_type="dataset",
                                    filename=fname, revision=pol.revision, token=token)
        except Exception as exc:  # noqa: BLE001
            print(f"  SKIP {fname}: {str(exc)[:70]}")
            continue
        stride = max(1, nrows // args.rows_per_shard)
        ids, masks, losses = [], [], []
        with torch.no_grad():
            for tid, tmask in shard_rows(local, tok, seq_len, args.rows_per_shard,
                                         int(getattr(cfg.task.exp.data, "eval_min_text_chars", 200)), stride):
                ids.append(tid); masks.append(tmask)
                if len(ids) == args.batch:
                    input_ids = torch.tensor(ids, device=device)
                    attn = torch.tensor(masks, device=device)
                    # Mask padding out of the labels, exactly as the validator's
                    # DataCollatorForLanguageModeling(mlm=False) does
                    # (connito/shared/dataloader.py:562). Scoring pad positions
                    # penalises SHORT documents: german legislation has a median
                    # of 530 chars vs czech caselaw at 16,296, and an unmasked
                    # probe reported 17.11 vs 0.36 -- a length artifact, not a
                    # model weakness. 17.11 is worse than random (ln 100k ~ 11.5),
                    # which was the tell that the measurement was wrong.
                    labels = input_ids.clone()
                    labels[attn == 0] = -100
                    b = {"input_ids": input_ids, "attention_mask": attn, "labels": labels}
                    with torch.amp.autocast("cuda", enabled=True, dtype=dtype):
                        out = model(**b)
                    if torch.isfinite(out.loss):
                        losses.append(float(out.loss))
                    ids, masks = [], []
        if losses:
            results[fname] = {"mean": statistics.mean(losses), "n_batches": len(losses)}
            print(f"  [{n:>2}/{len(table)}] {statistics.mean(losses):7.4f}  {fname}"
                  f"   ({time.time()-t0:.0f}s)", flush=True)

    if not results:
        sys.exit("no shard produced a finite loss")

    ordered = sorted(results.items(), key=lambda kv: -kv[1]["mean"])
    vals = [v["mean"] for v in results.values()]
    print(f"\n{'=' * 66}\nWORST SHARDS (these are the rounds we lose):")
    for f, v in ordered[:8]:
        print(f"   {v['mean']:7.4f}  {f}")
    print("\nBEST SHARDS:")
    for f, v in ordered[-5:]:
        print(f"   {v['mean']:7.4f}  {f}")
    print(f"\nmean {statistics.mean(vals):.4f}   median {statistics.median(vals):.4f}")
    print(f"spread {max(vals) - min(vals):.4f}   stdev {statistics.pstdev(vals):.4f}")
    print("A large spread means our score is decided by the shard draw, not by the model.")

    Path(args.out).write_text(json.dumps(
        {"checkpoint": ckpt_dir, "per_shard": results,
         "spread": max(vals) - min(vals), "mean": statistics.mean(vals)}, indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
