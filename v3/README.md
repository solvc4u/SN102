# v3 — `exp_legal` (group_id 3)

Frozen record of the legal-text challenge, which ran until **2026-08-01 05:30 UTC**,
when SN102 v0.4.0 moved the subnet to `exp_nemotron_c4` (group 4). See [`../v4/`](../v4/)
for the successor.

Group 3 is dead network-wide: validators only score commits whose `expert_group`
matches the active task, so nothing here can be submitted again. It is kept because
the training logic, the measurements behind it, and the failure modes we mapped all
carry forward.

---

## 1. Task

| | |
|---|---|
| expert group | `exp_legal`, `group_id: 3` |
| sources | `joelniklaus/Multi_Legal_Pile` (`all_all`) 0.5 · `allenai/c4` (`en`) 0.5 |
| sequence length | 1024 |
| `world_size` / `rank` | 10 / 1 (stock — see §6, this was a mistake) |
| helper group | 2 (`exp_c4_p02`), frozen, natural-routing fallback |
| final baseline loss | ~1.7652 |

Cycle: Distribute(20) → Train(300) → MinerCommit1(10) → MinerCommit2(10) →
Submission(80) → Validate(10) → Merge(50) → ValidatorCommit1/2 — about 105 min.

## 2. Hardware and environment

4 × RTX 5090 (sm_120, 31.4 GiB), torch 2.10+cu128, bitsandbytes 0.50.0.
GPU map `250→0`, `178→1`, `121→2` (held back as a candidate); **GPU 3 forbidden** —
it belongs to another subnet's miners, as do 2 and 3 in later periods.

Each miner is **two processes**: `ops.autolr` (trainer, serves the axon) and
`ops.commit` (submission worker). Both are needed; a live trainer with a dead commit
worker scores nothing and looks healthy in `tmux`.

## 3. Training logic

**Learning rate — `ops/autolr.py`.** Stock builds
`get_cosine_schedule_with_warmup(warmup=0, total_steps=88_000)`, which over a ~700-step
Train phase is effectively a constant LR: it never anneals. Replaced with a
**phase-aware one-cycle** schedule that completes warmup→peak→floor inside each phase,
so every submission is made from an annealed point rather than mid-plateau.

Settings (via `ops/lr_override.json`, re-read **every cycle** — changes apply without a
restart, which matters because a restart costs the in-flight round):

```json
{"peak_lr": 3e-5, "warmup": 0.03, "min_frac": 0.02}
```

`3e-5` is deliberately off the stock `1e-4` (see §6).

**Precision.** `install_memory_profile()` forces `upcast_trainable=False`, keeping
trainable params in bf16. Measured: **37 GB → 16 GB peak**. Without it the model OOMs at
31.08 GiB on a 5090. The trainable set is **3.38 B params** (16 experts/layer), not the
225 M an early estimate assumed — measure, don't estimate.

**Throughput.** Stock calls `get_model_hash` twice per step and `sum_model_gradients`
per micro-batch: **53.6 of every 60 minutes** went to diagnostics. `CONNITO_DIAG_EVERY=25`
throttles both. **79 → 692 steps/hour.**

**Eval.** `install_eval_recorder()` writes `{inner_opt_step: val_loss}` to
`local_eval.json` — the only per-step signal we have, since the dashboard lags.

## 4. Corpus

`ops/fetch_legal_native.py` — **uniform per shard**, from pinned revision `911e1d21…`:

```python
per = args.rows // len(table)
quotas = {f: min(per, n) for f, n in table.items()}
```

Two earlier attempts were wrong, both found only by measuring the result:

- head-of-stream → **93 % Czech/Bulgarian**
- proportional-to-shard-size → **81 % Portuguese**

`ops/hotswap_corpus.py` swaps a corpus in **without restarting miners**: `_source_rows`
re-opens each shard by path every pass, so repacking into the *same file count* and
`os.replace()`-ing each is atomic and invisible to a running dataloader. The file count
must not change — a missing path is a crash mid-Train.

## 5. Submission and monitoring

`ops/commit.py` — `install_upload_retry()`: stock `_upload_checkpoint_to_hf_safe` has
**no retry**, so a single HF 429 silently costs the round.
`install_best_checkpoint_selection()` patches `select_best_checkpoint`, which is a
misnomer — it returns the *newest*, not the best.

`ops/monitor.py` — liveness by **log freshness**, not tmux topology: during the
3.4-hour outage every window was alive while all four processes hung on a finney
websocket 429 (fixed with `lite_network=archive`). Also a **commit-cadence** check;
a ~105 min cycle means a gap over 160 min is a missed round.

## 6. What we learned (the part that matters for v4)

**Scoring.** `delta = max(0, baseline_loss - val_loss)`, `score = delta ** 1.2`, then
**rank-mapped to the top 3 only** (2.25 / 1.5 / 1.0). Everyone else gets 0.0 — being
4th is worth exactly as much as being 100th. `baseline_loss` is the validator scoring
the *unmodified* global checkpoint on its own seeded slice, so miners cannot move it.

**Weight group 1** (98 % of weight) needs `record_count >= 3` *and* scores in ≥3 distinct
rounds within `5 × cycle_length`. **Group 2** (2 %) needs only 1 record. Cohorts A/B/C are
rebuilt every cycle; only a hotkey change resets score history.

**The tie rule — the most important thing here.** Two miners submitting an *exactly equal*
`val_loss` are **both zeroed** as suspected duplicates. Stock ships `int_seed = 42` and
`rank: 1 / world_size: 10`, so every default miner trains identical data in identical
order and the whole stock field mutually annihilates. **Divergence is a precondition for
scoring at all, not an optimisation.**

We got the LR divergence right (3e-5 vs stock 1e-4) but **left `rank` at the stock 1 for
both miners for the entire challenge** — so 250 and 178 were also colliding with each
other. Fixed in v4 via `CONNITO_DATA_RANK`.

**Upstream bugs we carry** (`connito-patches/`, re-apply after every `git pull`):
`.safetensors` download fallback in `shared/model.py` — stock requests
`model_expgroup_{N}.pt`, which **404s**, and the miner then silently trains from the base
architecture instead of the global checkpoint.

## 7. Results

| | uid 250 | uid 178 |
|---|---|---|
| final local val_loss | 1.5500 | **1.4262** |
| best local val_loss | 1.0385 | 1.2873 |
| steps trained | 51,200 | 36,800 |

Baseline was ~1.7652, so both were genuinely below it. **Neither ever reached the top 3**;
cohort oscillated `tail`/`C`, `score_avg` stayed 0.0, `chain_weight` 0.0. Ranks hovered
~90th–146th of ~104–148 scored miners.

Full per-step curves: `results/uid{250,178}-local_eval.json` (200 evals each).

**Honest read:** every lever we pulled — corpus composition, corpus size, LR, checkpoint
selection, throughput, precision — was either disproved by measurement or produced no
detectable field-relative effect. The one hypothesis we never tested is the one in §6:
that we were tie-zeroed against the field the whole time. v4 tests it.

## 8. Layout

```
ops/               tooling snapshot as it ran (minus .env — never committed)
connito-patches/   diffs against Connito upstream d51c47e
configs/           exp_legal group config
results/           local eval curves, corpus manifest, shard probe
```

`ops/launch.sh` and `ops/monitor.py` here also contain two changes made on 2026-08-01
that belong to v4 (per-uid `data_rank`, and the two-consecutive-poll restart gate);
they are documented in `../v4/README.md` rather than reverted out of this snapshot.
