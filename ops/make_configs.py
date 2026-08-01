#!/usr/bin/env python3
"""
Generate the three per-UID miner configs.

Run AFTER setup_env.sh (imports MinerConfig, which imports torch):

    set -a; . /root/SN102/ops/.env; set +a
    python ops/make_configs.py

Writes ops/configs/uid121.yaml, uid178.yaml, uid250.yaml.

Building these through `MinerConfig` rather than hand-writing YAML means
pydantic validates every field and the locked-field machinery
(`check_and_prompt_locked`) sees exactly what it expects. Hand-rolled YAML
silently drifts from the schema and gets reset on next startup.

Per-UID settings that must differ, and why
------------------------------------------
chain.port        Miners serve an axon: `connito/miner/train.py:308` calls
                  setup_chain_worker(config) whose `serve` defaults to True
                  (the validator is the one that passes serve=False). Three
                  miners on one box all defaulting to port 8000 collide.
                  Nothing dials the axon -- submission is HF + chain commit --
                  but the extrinsic must carry this host, not a stale one.
chain.ip          Published address. Set to this box's public IP.
run.run_name      Part of the local checkpoint path. Sharing a run_name across
                  UIDs would let one miner commit another's checkpoint via
                  select_best_checkpoint.
hf.checkpoint_repo  Separate repo (and separate token) per UID. See
                  ops/README-429.md.
data.rank         Selects a disjoint stripe of the shared corpus so our own
                  UIDs never produce bit-identical val_loss -- which zeroes
                  BOTH miners under the duplicate-submission heuristic in
                  finalize_round_scores.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, os.environ.get("CONNITO_ROOT", "/root/SN102/Connito"))

from connito.shared.config import MinerConfig  # noqa: E402

OPS = Path(os.environ.get("OPS_ROOT", "/root/SN102/ops"))
CONFIG_DIR = OPS / "configs"

# uid -> (hotkey env var, repo env var, axon port, data stripe)
MINERS = {
    121: ("HOTKEY_UID121", "HF_REPO_UID121", 8000, 0),
    178: ("HOTKEY_UID178", "HF_REPO_UID178", 8001, 1),
    250: ("HOTKEY_UID250", "HF_REPO_UID250", 8002, 2),
}

# expert_group_name is LOCKED to exp_legal in TaskCfg._LOCKED_FIELDS after the
# subnet-wide switch off exp_math; auto_update_config resets any other value on
# load. We do not set it here -- the default is the only valid choice.
EXPERT_GROUP = "exp_legal"


def build(uid: int, hotkey: str, repo: str, port: int, stripe: int) -> MinerConfig:
    cfg = MinerConfig()

    cfg.chain.coldkey_name = os.environ.get("COLDKEY_NAME", "Apollo")
    cfg.chain.hotkey_name = hotkey
    cfg.chain.ip = os.environ.get("CONNITO_PUBLIC_IP", "0.0.0.0")
    cfg.chain.port = port
    # netuid / network are locked; leave them alone.
    #
    # lite_network is NOT locked. Default "finney" started returning HTTP 429 on
    # the websocket handshake and every miner process hung indefinitely inside
    # setup_chain_worker -- 3.4h of silent downtime with the tmux window still
    # alive. chain.py reuses the archive connection when lite_network ==
    # network, so this both avoids the rate-limited endpoint and halves our
    # chain connections (2 miners x 2 processes were opening 4).
    cfg.chain.lite_network = cfg.chain.network

    cfg.run.run_name = f"uid{uid}"

    # cfg.task.expert_group_name is locked; leave at its exp_legal default.

    # Read from the shared local corpus instead of streaming from HF.
    # dataset_class is set in expert_groups/exp_legal/config.yaml, not here:
    # _update_by_task replaces task.exp wholesale from that file on load.
    # world_size here is the DATA stripe count, not the GPU count. Three
    # concurrent miners -> three disjoint stripes.
    cfg.task.exp.data.world_size = len(MINERS)
    cfg.task.exp.data.rank = stripe

    cfg.hf.checkpoint_repo = repo

    # 32GB RTX 5090, not the 47GB A6000 the fp32 path is sized for. 8-bit
    # AdamW state is ~4x smaller and is the documented mitigation in
    # connito/shared/config.py:OptimizerCfg.
    cfg.opt.adamw_optim_bits = 8

    # setup_env.sh confirmed bf16 is supported on these cards. The stock
    # default is fp16-mixed, and train_worker carries a lot of machinery for
    # non-finite losses (skip-batch counters, a FloatingPointError after 50
    # consecutive bad batches) because fp16 overflows on this model. bf16 has
    # the same exponent range as fp32 and does not need a GradScaler, so
    # batches that fp16 miners silently drop still contribute gradient here.
    # Fewer skipped batches per Train phase is free val_loss.
    cfg.model.precision = "bf16-mixed"

    # autolr.py overrides the schedule at runtime; these are the fallback
    # values if you ever run the stock trainer directly.
    cfg.opt.lr = 1e-4
    cfg.sched.warmup_steps = 0

    # Local evaluation costs ~25s and fired every metric_interval(20) steps.
    # Now that we do ~320 steps/hour that is ~16 evals/hour = ~7 min of the
    # Train phase spent on a diagnostic the validator does not read.
    cfg.log.metric_interval = 100

    # Give best-checkpoint selection something to choose from. The stock
    # topk=2 leaves almost no candidates; aligning checkpoint_interval with
    # metric_interval means every saved checkpoint has a recorded eval score.
    # ~700 steps/hour / 100 = ~7 checkpoints per Train phase, 4 retained.
    cfg.ckpt.checkpoint_interval = 100
    cfg.ckpt.checkpoint_topk = 4

    # gradient_accumulation_steps only auto-derives from batch_size/per_device
    # when it is explicitly 0 (config.py:956); its default of 4 otherwise
    # sticks. With per_device now 4 that would make the effective batch 16 and
    # CUT optimizer steps 4x -- the opposite of what we need. Our submitted
    # update is ~14.5x smaller than the leaders' because we complete too few
    # steps in the fixed 60-minute Train phase, so total displacement is the
    # thing to grow.
    #
    # Setting accumulation to 1 keeps the effective batch at 4 exactly as
    # before, but reaches it in ONE forward of size 4 instead of four forwards
    # of size 1. Same optimizer semantics, more steps per hour.
    cfg.local_par.gradient_accumulation_steps = 1

    # Single GPU per miner. world_size>1 exists in the code path but is
    # explicitly not part of the supported miner profile.
    cfg.local_par.world_size = 1

    return cfg


def main() -> int:
    missing = [v for _, (h, r, _, _) in MINERS.items() for v in (h, r) if not os.environ.get(v)]
    if missing:
        print("missing env vars (fill ops/.env):", ", ".join(sorted(set(missing))))
        return 1

    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    for uid, (hk_var, repo_var, port, stripe) in MINERS.items():
        repo = os.environ[repo_var]
        if len(repo) > 32:
            print(f"FAIL uid{uid}: repo id {repo!r} is {len(repo)} chars; "
                  f"chain commit caps it at 32 (CHAIN_COMMIT_MAX_HF_REPO_ID_CHARS)")
            return 1
        cfg = build(uid, os.environ[hk_var], repo, port, stripe)
        out = CONFIG_DIR / f"uid{uid}.yaml"
        # MinerConfig.write() targets its own derived path; dump explicitly here
        # so the configs live together under ops/configs/.
        import yaml
        data = cfg.model_dump(mode="json")
        out.write_text(yaml.safe_dump(data, sort_keys=False))
        print(f"wrote {out}  (hotkey={cfg.chain.hotkey_name} port={port} stripe={stripe} repo={repo})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
