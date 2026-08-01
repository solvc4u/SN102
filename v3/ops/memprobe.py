#!/usr/bin/env python3
"""
Where does the VRAM actually go?

    CUDA_VISIBLE_DEVICES=3 python -m ops.memprobe --path ops/configs/uid250.yaml

Loads the miner model exactly as `setup_training` does and reports allocated /
reserved VRAM at each stage, then runs one forward, one backward and one
optimizer step so the true training peak is measured rather than estimated.

Written because two OOMs on a 31.4GiB card disagreed with a first-principles
estimate of 11-16GiB. The estimate was wrong somewhere; this finds out where
instead of guessing again.
"""

from __future__ import annotations

import argparse
import os
import sys

import torch


def mb(x: int) -> float:
    return x / 2**20


def report(tag: str) -> None:
    torch.cuda.synchronize()
    a = torch.cuda.memory_allocated()
    r = torch.cuda.memory_reserved()
    p = torch.cuda.max_memory_allocated()
    print(f"{tag:<44} allocated={mb(a):9.0f}MiB reserved={mb(r):9.0f}MiB peak={mb(p):9.0f}MiB", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True)
    ap.add_argument("--no-upcast", action="store_true",
                    help="keep trainable params in bf16 instead of upcasting to fp32")
    ap.add_argument("--paged", action="store_true", help="use PagedAdamW8bit")
    ap.add_argument("--bs", type=int, default=None, help="override per-device batch size")
    ap.add_argument("--bench", type=int, default=0, help="time N train steps after warmup")
    args = ap.parse_args()

    from connito.shared.config import MinerConfig
    from connito.shared.expert_manager import ExpertManager
    from connito.shared.model import load_model, freeze_parameters
    from connito.shared.modeling.mycelia import get_base_tokenizer
    from connito.shared.chain import setup_chain_worker

    cfg = MinerConfig.from_path(args.path, auto_update_config=True)
    print(f"precision={cfg.model.precision} seq_len={cfg.task.exp.data.sequence_length} "
          f"per_device_bs={cfg.task.exp.data.per_device_train_batch_size} "
          f"adamw_bits={cfg.opt.adamw_optim_bits} group={cfg.task.exp.group_id}", flush=True)

    device = torch.device("cuda:0")
    torch.cuda.reset_peak_memory_stats()
    report("baseline")

    wallet, subtensor, _ = setup_chain_worker(cfg, serve=False)
    em = ExpertManager(cfg)
    model, _ckpt = load_model(0, cfg, em, subtensor, wallet, partial=True)
    model = model.to(device)
    report("after load_model + .to(cuda)")

    model = freeze_parameters(
        model=model, expert_manager=em,
        expert_group_id=cfg.task.exp.group_id, upcast_trainable=not args.no_upcast,
    )
    report(f"after freeze_parameters(upcast={not args.no_upcast})")

    trainable = [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in trainable)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"trainable params: {n_train/1e6:.1f}M / total {n_total/1e6:.1f}M "
          f"({len(trainable)} tensors)", flush=True)
    by_dtype = {}
    for p in model.parameters():
        by_dtype[str(p.dtype)] = by_dtype.get(str(p.dtype), 0) + p.numel()
    print("params by dtype:", {k: f"{v/1e6:.1f}M" for k, v in by_dtype.items()}, flush=True)

    print(f"gradient_checkpointing flag: "
          f"{getattr(getattr(model, 'model', model), 'gradient_checkpointing', 'n/a')}", flush=True)

    import bitsandbytes as bnb
    OptCls = bnb.optim.PagedAdamW8bit if args.paged else bnb.optim.AdamW
    kw = {} if args.paged else {"optim_bits": 8}
    opt = OptCls(trainable, lr=1e-5, weight_decay=0.1, betas=(0.9, 0.95), **kw)
    print("optimizer:", OptCls.__name__, flush=True)
    report("after optimizer construction (state is lazy)")

    seq = int(cfg.task.exp.data.sequence_length)
    bs = args.bs or int(cfg.task.exp.data.per_device_train_batch_size)
    ids = torch.randint(0, 1000, (bs, seq), device=device)
    batch = {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": ids}

    dtype = torch.bfloat16 if cfg.model.precision == "bf16-mixed" else torch.float16
    model.train()
    with torch.amp.autocast("cuda", enabled=True, dtype=dtype):
        out = model(**batch)
        loss = out.loss
    report("after forward")

    loss.backward()
    report("after backward")

    opt.step()
    report("after optimizer.step() (8-bit state materialised)")
    opt.zero_grad(set_to_none=True)
    report("after zero_grad")

    total = torch.cuda.get_device_properties(0).total_memory
    print(f"\nTRAINING PEAK (bs={bs}): {mb(torch.cuda.max_memory_allocated()):.0f}MiB "
          f"of {mb(total):.0f}MiB card ({100*torch.cuda.max_memory_allocated()/total:.1f}%)", flush=True)

    if args.bench:
        import time
        for _ in range(2):  # warmup
            with torch.amp.autocast("cuda", enabled=True, dtype=dtype):
                loss = model(**batch).loss
            loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(args.bench):
            with torch.amp.autocast("cuda", enabled=True, dtype=dtype):
                loss = model(**batch).loss
            loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        per_step = dt / args.bench
        print(f"BENCH bs={bs}: {per_step*1000:.0f} ms/step, "
              f"{bs/per_step:.2f} samples/s, {bs*seq/per_step:,.0f} tok/s, "
              f"peak={mb(torch.cuda.max_memory_allocated()):.0f}MiB", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
