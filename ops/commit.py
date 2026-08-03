#!/usr/bin/env python3
"""
Submission process for a SN102 miner, with upload retry.

`connito.miner.train` ONLY trains. Download (Distribute) and submission
(MinerCommit1/2) live in a separate process, `connito.miner.model_io`, whose
`run_system` starts the scheduler plus a download worker and a commit worker.
Run only the trainer and the miner trains forever and never submits anything.

So each UID needs two processes:

    python -m ops.autolr  --uid N --path <config>   # trains
    python -m ops.commit         --path <config>   # downloads + submits

They hand off through the local checkpoint directory: the trainer writes
checkpoints, `select_best_checkpoint` picks the newest, the commit worker
uploads it.

--------------------------------------------------------------------------------
Why the retry wrapper
--------------------------------------------------------------------------------
`connito/miner/model_io.py:_upload_checkpoint_to_hf_safe` has **no retry**. It
wraps `upload_checkpoint_to_hf` in a single try/except, and on any exception
logs and returns `(None, None)`. The caller then still commits to chain, just
without `hf_repo_id`/`hf_revision` — so validators cannot locate the checkpoint
and you are counted missing for the round.

One transient 429 or a dropped TCP connection therefore burns a whole cycle,
and Weight Group 1 requires scores in 3 of the last 5 cycles. There is no
reason for a retryable error to cost that much.

`_classify_upload_error` does not even have a bucket for 429 — it falls through
to "rpc" or "unknown".

This wrapper retries on rate limits and transient transport errors with
jittered exponential backoff, bounded by a wall-clock budget so we never
overrun the commit window and make things worse. Non-retryable failures (401
bad token, 403 no write access, 404 missing repo) fail fast — retrying those
just wastes the window.
"""

from __future__ import annotations

import json
import os
import random
import sys
import time

# Budget in seconds for all upload attempts combined. MinerCommit1 and
# MinerCommit2 are 10 blocks (~2 min) each; staying well inside one phase
# means a retry cannot push the chain commit past its window.
UPLOAD_RETRY_BUDGET_S = float(os.environ.get("CONNITO_UPLOAD_RETRY_BUDGET_S", "100"))
UPLOAD_MAX_ATTEMPTS = int(os.environ.get("CONNITO_UPLOAD_MAX_ATTEMPTS", "4"))


def _status_code(exc: BaseException) -> int | None:
    resp = getattr(exc, "response", None)
    code = getattr(resp, "status_code", None)
    if isinstance(code, int):
        return code
    code = getattr(exc, "status_code", None)
    return code if isinstance(code, int) else None


def _is_retryable(exc: BaseException) -> bool:
    code = _status_code(exc)
    if code is not None:
        # 429 rate limit; 5xx server-side. Everything else in the 4xx range is
        # a config problem no amount of retrying fixes.
        return code == 429 or 500 <= code < 600
    msg = str(exc).lower()
    if any(k in msg for k in ("401", "403", "404", "unauthorized", "forbidden", "not found")):
        return False
    return any(
        k in msg
        for k in (
            "429", "rate limit", "too many requests", "throttl",
            "timeout", "timed out", "connection", "reset by peer",
            "incompleteread", "broken pipe", "502", "503", "504",
        )
    )


def install_upload_retry() -> None:
    import connito.miner.model_io as mio

    original = mio.upload_checkpoint_to_hf

    def retrying_upload(*args, **kwargs):
        deadline = time.monotonic() + UPLOAD_RETRY_BUDGET_S
        last: BaseException | None = None
        for attempt in range(1, UPLOAD_MAX_ATTEMPTS + 1):
            try:
                return original(*args, **kwargs)
            except BaseException as exc:  # noqa: BLE001 - classify then re-raise
                last = exc
                code = _status_code(exc)
                if not _is_retryable(exc):
                    print(
                        f"[commit] upload failed permanently "
                        f"(status={code}, {type(exc).__name__}): {exc}",
                        file=sys.stderr,
                    )
                    raise
                remaining = deadline - time.monotonic()
                if attempt >= UPLOAD_MAX_ATTEMPTS or remaining <= 0:
                    print(
                        f"[commit] upload retries exhausted after {attempt} attempt(s); "
                        f"MISSING FOR THIS ROUND (status={code}): {exc}",
                        file=sys.stderr,
                    )
                    raise
                # Jitter so co-located miners that hit the same limit at the
                # same instant do not retry in lockstep and collide again.
                backoff = min(remaining, (2 ** (attempt - 1)) * 5 * (0.5 + random.random()))
                print(
                    f"[commit] upload attempt {attempt} failed (status={code}); "
                    f"retrying in {backoff:.1f}s ({remaining:.0f}s budget left)",
                    file=sys.stderr,
                )
                time.sleep(backoff)
        assert last is not None
        raise last

    mio.upload_checkpoint_to_hf = retrying_upload
    print(
        f"[commit] upload retry installed "
        f"(max {UPLOAD_MAX_ATTEMPTS} attempts, {UPLOAD_RETRY_BUDGET_S:.0f}s budget)"
    )


EVAL_LOG_NAME = "local_eval.json"


def install_repo_override() -> None:
    """Let the HF upload repo be changed WITHOUT restarting the miner.

    `config.hf.checkpoint_repo` is read once at process start, so changing where
    we publish used to cost a restart -- and a restart near MinerCommit costs the
    round. `resolve_hf_repo_ids` is called inside `_upload_checkpoint_to_hf_safe`
    on every submission, so wrapping it there gives a value that can change
    between cycles with the miner still running. Same principle as
    ops/lr_override.json and the corpus hot-swap.

    Control file, re-read on EVERY submission:  ops/hf_repo_<uid>.json

        {"repo": "solvc4u/co-250-a"}          pin to one repo
        {"mode": "rotate",
         "prefix": "solvc4u/co-250-"}          fresh repo per submission

    Rotation exists because of what the leaderboard shows: every operator
    currently earning (Attila115, Infinite3214) publishes ONE submission per
    repo and then moves on, while both operators that append to a long-lived
    repo earn nothing -- athena2634 holds the four best val_loss on the subnet
    (1.4418) and is paid zero, and our own co2 carries 103 submissions. A
    validator fetching a revision out of ~340 GB of LFS history inside a bounded
    phase window is doing very different work than fetching a repo with one
    file. Unproven as a mechanism, cheap to test.

    Missing or malformed file -> stock behaviour, never an exception: this runs
    on the submission path and a crash here forfeits the round.
    """
    from connito.miner import model_io

    orig = model_io.resolve_hf_repo_ids
    uid = os.environ.get("CONNITO_UID", "")
    path = os.environ.get(
        "CONNITO_HF_REPO_OVERRIDE", f"/root/SN102/ops/hf_repo_{uid}.json")

    def patched(hf_cfg, *a, **kw):
        try:
            with open(path) as fh:
                ov = json.load(fh)
        except FileNotFoundError:
            return orig(hf_cfg, *a, **kw)
        except Exception as exc:  # noqa: BLE001
            print(f"[commit] ignoring bad repo override ({exc})", flush=True)
            return orig(hf_cfg, *a, **kw)

        repo = ov.get("repo")
        if not repo and ov.get("mode") == "rotate":
            prefix = ov.get("prefix") or ""
            if prefix:
                # Suffix must be unique per submission and identical in the
                # chain commit, so derive it from the clock rather than a
                # counter that a restart would reset.
                repo = f"{prefix}{time.strftime('%m%d%H%M', time.gmtime())}"
        if not repo:
            return orig(hf_cfg, *a, **kw)

        # Route through the stock resolver so the chain-payload length check
        # and advertised-repo derivation still apply to the new value.
        try:
            hf_cfg = hf_cfg.model_copy(update={"checkpoint_repo": repo})
        except Exception:  # noqa: BLE001 - not a pydantic model
            try:
                hf_cfg.checkpoint_repo = repo
            except Exception as exc:  # noqa: BLE001
                print(f"[commit] cannot apply repo override ({exc})", flush=True)
                return orig(hf_cfg, *a, **kw)
        up, chain = orig(hf_cfg, *a, **kw)
        print(f"[commit] repo override -> upload={up} chain={chain}", flush=True)
        return up, chain

    model_io.resolve_hf_repo_ids = patched
    print(f"[commit] repo override installed (watching {path})", flush=True)


def install_best_checkpoint_selection() -> None:
    """Commit the LOWEST-LOSS checkpoint, not the newest one.

    `connito/shared/checkpoints.py:select_best_checkpoint` is a misnomer: it
    orders by (active, global_ver, inner_opt) and returns the most recent
    checkpoint. Training loss is noisy -- per-batch sigma ~0.9 at batch 4 -- so
    the final step is not reliably the best model of the phase.

    ops/autolr.py records every local eval as {inner_opt_step: val_loss} in
    `local_eval.json` beside the checkpoints. This picks the on-disk checkpoint
    whose inner_opt has the lowest recorded loss, falling back to stock
    behaviour whenever the record is missing, empty, or matches nothing -- so a
    failure here degrades to "commit the newest", never to "commit nothing".
    """
    import connito.miner.model_io as mio

    original = mio.select_best_checkpoint

    def choose(primary_dir, *args, **kwargs):
        stock = original(primary_dir, *args, **kwargs)
        try:
            import json as _json
            from pathlib import Path as _Path

            evals = _json.loads((_Path(primary_dir) / EVAL_LOG_NAME).read_text())
            if not evals:
                return stock
            from connito.shared.checkpoints import build_local_checkpoints

            cands = build_local_checkpoints(_Path(primary_dir), role="miner").checkpoints
            scored = []
            for c in cands:
                io_step = getattr(c, "inner_opt", None)
                if io_step is None:
                    continue
                loss = evals.get(str(int(io_step)))
                if loss is not None:
                    scored.append((float(loss), c))
            if not scored:
                print("[commit] no eval score matches an on-disk checkpoint; using newest")
                return stock
            mode = os.environ.get("CONNITO_CKPT_SELECT", "best").lower()
            if mode == "stable" and len(scored) >= 3:
                # STABILITY selection, not greedy argmin.
                #
                # Measured across the live board: what separates earning miners
                # from non-earning ones is NOT mean loss (earners average 1.8186
                # vs 1.8054 for non-earners) and NOT best-ever score (non-earners
                # are better: d_best -0.1066 vs -0.0135). It is the standard
                # deviation of their distance from the field median per round:
                #
                #   uid 158 (top earner)  d_sd 0.0016
                #   uid 249               d_sd 0.0016
                #   uid  98               d_sd 0.0096
                #   uid 193 (marginal)    d_sd 0.2824
                #   ours                  d_sd 0.5098 / 0.5176
                #
                # Weight Group 1 pays a rolling 8-round score_avg with a 3-of-5
                # recency gate, so being 0.01 off the median EVERY round beats
                # being 0.5 better half the time and 0.5 worse the rest.
                #
                # argmin over a noisy eval curve systematically picks the luckiest
                # single measurement, which does not reproduce on the validator's
                # slice. Smoothing over neighbouring checkpoints selects a genuine
                # low region instead of a noise trough.
                by_step = {int(getattr(c, "inner_opt", -1)): (l, c) for l, c in scored}
                steps = sorted(by_step)
                smoothed = []
                for i, st in enumerate(steps):
                    window = [by_step[steps[j]][0]
                              for j in range(max(0, i - 1), min(len(steps), i + 2))]
                    smoothed.append((sum(window) / len(window), by_step[st][0], by_step[st][1], st))
                smoothed.sort(key=lambda t: t[0])
                sm, raw, best, st = smoothed[0]
                print(f"[commit] STABLE selection: inner_opt={st} raw={raw:.6f} "
                      f"smoothed={sm:.6f} over {len(steps)} candidates")
                return best
            scored.sort(key=lambda t: t[0])
            best_loss, best = scored[0]
            stock_io = getattr(stock, "inner_opt", None)
            stock_loss = evals.get(str(int(stock_io))) if stock_io is not None else None
            print(
                f"[commit] best-of-{len(scored)} checkpoint: inner_opt={best.inner_opt} "
                f"val_loss={best_loss:.6f}"
                + (f" (newest was inner_opt={stock_io} val_loss={float(stock_loss):.6f})"
                   if stock_loss is not None else f" (newest inner_opt={stock_io}, unscored)")
            )
            return best
        except FileNotFoundError:
            return stock
        except Exception as exc:  # noqa: BLE001 - never block a commit
            print(f"[commit] best-checkpoint selection failed ({exc}); using newest")
            return stock

    mio.select_best_checkpoint = choose
    print("[commit] best-checkpoint selection installed")


def main() -> int:
    import bittensor

    import connito.miner.model_io as mio
    from connito.shared.config import MinerConfig, parse_args
    from connito.shared.chain import setup_chain_worker
    from connito.shared.expert_manager import ExpertManager

    install_upload_retry()
    install_best_checkpoint_selection()
    install_repo_override()

    args = parse_args()
    config = (
        MinerConfig.from_path(args.path, auto_update_config=args.auto_update_config)
        if args.path
        else MinerConfig()
    )
    config.write()

    # serve=False: the trainer process already publishes this hotkey's axon at
    # startup. Both entrypoints default to serve=True, so running them together
    # fires two identical serve_axon extrinsics seconds apart and the second is
    # rejected with `Custom type(1012): Transaction is temporarily banned`.
    # Harmless but noisy, and it wastes the subnet's serving_rate_limit window.
    wallet, subtensor, _lite = setup_chain_worker(config, serve=False)
    expert_manager = ExpertManager(config)
    mio.run_system(config, wallet, expert_manager, subtensor=subtensor)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
