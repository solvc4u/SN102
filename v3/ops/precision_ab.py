#!/usr/bin/env python3
"""
Does bf16 (no fp32 master weights) cost us val_loss?

    CUDA_VISIBLE_DEVICES=2 python -m ops.precision_ab --path ops/configs/uid250.yaml \
        --steps 120 --eval-batches 40

Runs the SAME training recipe twice from the SAME starting checkpoint, differing
only in `freeze_parameters(upcast_trainable=...)`, and evaluates both on an
identical held-out probe set. Prints the val_loss delta between them.

Why this matters
----------------
`upcast_trainable=True` is the stock profile, but it needs ~37GiB (3.38B
trainable params in fp32, plus fp32 grads) and OOMs on a 31.4GiB RTX 5090. We
run bf16 instead. That was a forced choice, and bf16 has ~8 mantissa bits, so
small optimizer updates can vanish -- on a subnet where the field is clustered
inside 0.001 of val_loss, that could be the whole gap.

Right now UID 250 scores 2.1332 while ~25 miners sit at ~1.9105. This measures
how much of that 0.22 gap, if any, is precision rather than data or schedule.

Deliberately offline on the spare GPU: it costs no mining cycle and no
submission window, and both arms see identical data in identical order.

`--paged` is on by default for the fp32 arm since that is the only way it has a
chance of fitting; if it OOMs anyway the script reports that as the result
rather than dying, because "fp32 does not fit" is itself the answer.
"""

from __future__ import annotations

import argparse
import copy
import statistics
import sys

import torch


def build(cfg, upcast: bool, device):
    from connito.shared.expert_manager import ExpertManager
    from connito.shared.model import load_model, freeze_parameters
    from connito.shared.chain import setup_chain_worker

    wallet, subtensor, _ = setup_chain_worker(cfg, serve=False)
    em = ExpertManager(cfg)
    model, _ = load_model(0, cfg, em, subtensor, wallet, partial=True)
    model = model.to(device)
    model = freeze_parameters(
        model=model, expert_manager=em,
        expert_group_id=cfg.task.exp.group_id, upcast_trainable=upcast,
    )
    return model


def run_arm(cfg, upcast: bool, paged: bool, steps: int, batches, eval_batches, lr: float, device):
    import bitsandbytes as bnb

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    label = "fp32-master" if upcast else "bf16"
    print(f"\n=== arm: {label} (paged={paged}) ===", flush=True)

    try:
        model = build(cfg, upcast, device)
    except torch.OutOfMemoryError:
        print(f"{label}: OOM during model construction -- does not fit on this card")
        return None

    trainable = [p for p in model.parameters() if p.requires_grad]
    OptCls = bnb.optim.PagedAdamW8bit if paged else bnb.optim.AdamW
    kw = {} if paged else {"optim_bits": 8}
    opt = OptCls(trainable, lr=lr, weight_decay=0.1, betas=(0.9, 0.95), **kw)

    dtype = torch.bfloat16 if cfg.model.precision == "bf16-mixed" else torch.float16
    n_train = len(batches) - eval_batches

    # --- pre-training probe: the untrained floor for this arm ---------------
    model.eval()
    pre = []
    with torch.no_grad():
        for b in batches[n_train:]:
            with torch.amp.autocast("cuda", enabled=True, dtype=dtype):
                pre.append(float(model(**{k: v.to(device) for k, v in b.items()}).loss))
    pre_loss = statistics.mean(pre)
    print(f"{label}: probe loss BEFORE training = {pre_loss:.6f}", flush=True)

    # --- train --------------------------------------------------------------
    model.train()
    try:
        for i in range(steps):
            b = batches[i % n_train]
            with torch.amp.autocast("cuda", enabled=True, dtype=dtype):
                loss = model(**{k: v.to(device) for k, v in b.items()}).loss
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            if (i + 1) % 40 == 0:
                print(f"  {label}: step {i+1}/{steps} loss={float(loss):.4f}", flush=True)
    except torch.OutOfMemoryError:
        print(f"{label}: OOM during training -- does not fit on this card")
        return None

    # --- post-training probe, identical batches -----------------------------
    model.eval()
    post = []
    with torch.no_grad():
        for b in batches[n_train:]:
            with torch.amp.autocast("cuda", enabled=True, dtype=dtype):
                post.append(float(model(**{k: v.to(device) for k, v in b.items()}).loss))
    post_loss = statistics.mean(post)
    peak = torch.cuda.max_memory_allocated() / 2**20
    print(f"{label}: probe loss AFTER training  = {post_loss:.6f} "
          f"(improvement {pre_loss - post_loss:+.6f}, peak {peak:.0f}MiB)", flush=True)

    del model, opt
    torch.cuda.empty_cache()
    return {"label": label, "pre": pre_loss, "post": post_loss,
            "improvement": pre_loss - post_loss, "peak_mib": peak}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True)
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--eval-batches", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-4)
    args = ap.parse_args()

    from connito.shared.config import MinerConfig
    from connito.shared.dataloader import get_dataloader
    from connito.shared.modeling.mycelia import get_base_tokenizer

    cfg = MinerConfig.from_path(args.path, auto_update_config=True)
    device = torch.device("cuda:0")

    # Materialise ONE batch list up front and reuse it for both arms, so the
    # only difference between them is precision -- not data order.
    tok = get_base_tokenizer(cfg)
    dl = get_dataloader(cfg, rank=0, world_size=cfg.task.exp.data.world_size, tokenizer=tok)
    need = args.steps + args.eval_batches
    batches = []
    for b in dl:
        batches.append({k: v.clone() for k, v in b.items()})
        if len(batches) >= need:
            break
    print(f"materialised {len(batches)} batches "
          f"({args.steps} train / {args.eval_batches} probe), lr={args.lr:.1e}", flush=True)

    results = []
    for upcast, paged in ((False, True), (True, True)):
        r = run_arm(cfg, upcast, paged, args.steps, batches, args.eval_batches, args.lr, device)
        if r:
            results.append(r)

    print("\n================ RESULT ================")
    for r in results:
        print(f"  {r['label']:<12} pre={r['pre']:.6f} post={r['post']:.6f} "
              f"improvement={r['improvement']:+.6f} peak={r['peak_mib']:.0f}MiB")
    if len(results) == 2:
        bf16, fp32 = results[0], results[1]
        gap = bf16["post"] - fp32["post"]
        print(f"\n  fp32-master is {gap:+.6f} better than bf16 on the probe set.")
        print("  Our live gap to the leading pack is ~0.22 val_loss.")
        if abs(gap) < 0.01:
            print("  -> precision is NOT the problem. Look at data distribution / schedule.")
        else:
            print("  -> precision matters; solving memory properly is worth the effort.")
    else:
        print("\n  fp32 arm did not fit -- bf16 is forced on this hardware regardless,")
        print("  so precision cannot be traded away here. Focus on data/schedule.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
