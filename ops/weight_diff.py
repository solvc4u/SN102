#!/usr/bin/env python3
"""
Why are we 0.23 val_loss behind a field clustered inside 0.024?

    python -m ops.weight_diff --global-repo g-connito/co --global-rev ecb1c3b \
        --miner Attila115/co3:a9c006e --miner solvc4u/co2:6406459 --group 3

Downloads the validator's global checkpoint shard plus one or more miners'
submitted shards and compares them tensor-by-tensor on CPU. No GPU needed, so
this does not touch the cards.

What it answers
---------------
68 of 71 freshly-scored miners land between 1.7735 and 1.7979 -- a 0.024 spread
across ~68 independent operators. Independent training runs do not converge that
tightly. Either they are all submitting something very close to the global
checkpoint, or they share a recipe we don't.

  * If a top miner's shard is ~identical to the global checkpoint, the winning
    play is "barely change it", and our training is actively moving away from
    the eval distribution.
  * If a top miner's shard is far from global but they still score 1.77, they
    have found a direction we haven't, and the gap is a training-quality
    problem rather than a drift problem.
  * If OUR shard is far from global while theirs is close, that localises the
    fault to our training loop, not the data or the schedule.

Comparing published public weights for diagnosis is fair game -- validators
require these repos to be public in order to score them. This does not copy
anything into our submissions; it measures distance so we know which problem we
actually have.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def fetch(repo: str, rev: str, group: int, token: str | None) -> Path:
    from huggingface_hub import hf_hub_download

    fname = f"model_expgroup_{group}.safetensors"
    print(f"  fetching {repo}@{rev}:{fname}", flush=True)
    return Path(hf_hub_download(repo_id=repo, filename=fname, revision=rev, token=token))


def load(path: Path) -> dict:
    from safetensors.torch import load_file

    return load_file(str(path))


def compare(name: str, ref: dict, other: dict) -> None:
    import torch

    keys_ref, keys_other = set(ref), set(other)
    shared = sorted(keys_ref & keys_other)
    print(f"\n=== {name} vs GLOBAL ===")
    print(f"  tensors: ref={len(keys_ref)} other={len(keys_other)} shared={len(shared)}")
    if keys_ref - keys_other:
        print(f"  only in global: {len(keys_ref - keys_other)}")
    if keys_other - keys_ref:
        print(f"  only in miner : {len(keys_other - keys_ref)}")
    if not shared:
        print("  NO SHARED TENSORS -- different expert group or format")
        return

    identical = 0
    tot_sq_diff = 0.0
    tot_sq_ref = 0.0
    max_rel = 0.0
    max_rel_key = ""
    cos_num = 0.0
    cos_a = 0.0
    cos_b = 0.0

    for k in shared:
        a = ref[k].to(torch.float32).flatten()
        b = other[k].to(torch.float32).flatten()
        if a.shape != b.shape:
            print(f"  shape mismatch {k}: {tuple(a.shape)} vs {tuple(b.shape)}")
            continue
        if torch.equal(ref[k], other[k]):
            identical += 1
        d = (a - b)
        sq = float(torch.dot(d, d))
        rn = float(torch.dot(a, a))
        tot_sq_diff += sq
        tot_sq_ref += rn
        cos_num += float(torch.dot(a, b))
        cos_a += rn
        cos_b += float(torch.dot(b, b))
        rel = (sq / rn) ** 0.5 if rn > 0 else 0.0
        if rel > max_rel:
            max_rel, max_rel_key = rel, k

    rel_l2 = (tot_sq_diff / tot_sq_ref) ** 0.5 if tot_sq_ref > 0 else float("nan")
    cos = cos_num / ((cos_a ** 0.5) * (cos_b ** 0.5)) if cos_a > 0 and cos_b > 0 else float("nan")
    print(f"  bit-identical tensors : {identical}/{len(shared)}")
    print(f"  relative L2 distance  : {rel_l2:.6e}   (0 = identical to global)")
    print(f"  cosine similarity     : {cos:.9f}")
    print(f"  largest per-tensor rel: {max_rel:.6e}  ({max_rel_key})")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--global-repo", required=True)
    ap.add_argument("--global-rev", required=True)
    ap.add_argument("--miner", action="append", default=[],
                    help="repo:revision, repeatable")
    ap.add_argument("--group", type=int, default=3)
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN_CORPUS") or os.environ.get("HF_TOKEN")

    print("downloading shards...")
    try:
        gpath = fetch(args.global_repo, args.global_rev, args.group, token)
        gref = load(gpath)
    except Exception as exc:  # noqa: BLE001
        print(f"FAILED to fetch global checkpoint: {exc}")
        return 1
    print(f"  global: {len(gref)} tensors, "
          f"{sum(v.numel() for v in gref.values())/1e6:.1f}M params, "
          f"{gpath.stat().st_size/2**20:.0f}MiB")

    for spec in args.miner:
        repo, _, rev = spec.partition(":")
        try:
            p = fetch(repo, rev, args.group, token)
            m = load(p)
        except Exception as exc:  # noqa: BLE001
            print(f"\n=== {spec} ===\n  FAILED: {exc}")
            continue
        compare(spec, gref, m)

    print("\ninterpretation:")
    print("  rel L2 ~0        -> submitting the global checkpoint essentially untouched")
    print("  rel L2 small     -> light fine-tune, staying near the merged model")
    print("  rel L2 large     -> heavy training / drift away from the merged model")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
