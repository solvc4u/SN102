# v4 — `exp_nemotron_c4` (group_id 4) — ACTIVE

The current challenge. Announced and rolled out **2026-08-01**, replacing
[`../v3/`](../v3/) (`exp_legal`, group 3).

The live tooling is `/root/SN102/ops/` — this folder holds the v4-specific config,
the patches against upstream, and the reasoning. Upstream base: **`v0.4.0` / `4db4d8d`**.

---

## 1. What changed

| | v3 (`exp_legal`) | v4 (`exp_nemotron_c4`) |
|---|---|---|
| group_id | 3 | **4** |
| sources | Multi_Legal_Pile `all_all` + C4 `en` | **Nemotron-CC-Math-v1 `4plus`** + C4 `en` |
| baseline loss | ~1.7652 | **~3.6239** |
| `dataset_class` | `exp_legal.dataset:StreamingTorchDataset` | **unset** → `DefaultStreamingTorchDataset` |
| revision pins | corpus pinned locally | **both sources pinned in-config** |

**Everyone restarted from base DeepSeek-V2-Lite**, validators included. Prior checkpoints
fail validation because group 4 trains a different expert set. Score history is not wiped
but washes out, since scoring is a rolling average over recent scored rounds.

`expert_group_name` is a **locked field** — `auto_update_config` resets any manual value
on load. Do not set it. Confirmed working: the config reset `exp_legal` → `exp_nemotron_c4`
on first load after the pull.

`dataset_class` is deliberately unset upstream so `tokenize_windowed` applies — long
documents contribute a content-hash-derived window instead of always their most-templated
prefix. **Do not point this at `exp_math/dataset.py`**, which prefix-truncates and would
silently drop that anti-memorization fix. Our `ops/shared_dataset.py:LocalSharedDataset`
has the same caveat and is currently unused for that reason.

## 2. Gated dataset

`nvidia/Nemotron-CC-Math-v1` is gated. The HF account behind `HF_TOKEN_*` (**`solvc4u`**)
has accepted the licence. Without it the miner dies at dataloader build with
`GatedRepoError 403` — not at startup, so it looks like a training crash.

**`dataset_info()` succeeding does not mean you have access.** Metadata is public for
gated repos; only a file read proves it:

```python
hf_hub_download("nvidia/Nemotron-CC-Math-v1", repo_type="dataset",
                filename="4plus/part_000000.parquet", token=...)
```

## 3. The one real change in strategy: per-miner data rank

v3's unfalsified hypothesis was that we were **tie-zeroed** the whole time (see
`../v3/README.md` §6): the validator zeroes *both* miners on an exactly equal `val_loss`,
and stock ships `rank: 1` + `int_seed = 42`, so the entire default field trains identical
data in identical order. In v3 both our miners also sat on `rank: 1` — colliding with the
field *and* each other.

`rank` is **not** in `_LOCKED_FIELDS` (only `expert_group_name`, `helper_group_id`,
`routing_mode` are), and the group-4 config even ships `rank: 1 # TODO: based on your uid`.
But a per-uid override in the miner yaml is **discarded** — `_update_by_task` re-derives
`task.exp` from the group config on every load. Hence the patch in
`connito-patches/connito_shared_dataloader.py.patch`, which reads `CONNITO_DATA_RANK`:

```
uid 250 → rank 3
uid 178 → rank 7
uid 121 → rank 5   (held back)
world_size 10
```

Deliberately **not 1**, where the stock field sits. **Train only** — eval keeps the config
rank so local `val_loss` stays comparable to the validator baseline.

Verified: first batch losses now diverge from step 1
(`1.058 / 1.978 / 3.102` vs `2.715 / 3.114 / 2.978`). Under v3 these were identical.

The subnet-wide reset is the best conditions we have had to test this: the field's
accumulated score history and cohort-A membership evaporated at the same time.

## 4. Carried forward from v3

Unchanged and still load-bearing — see `../v3/README.md` §3 for the measurements:

- **`.safetensors` download fallback** (`shared/model.py`). v0.4.0 still requests
  `model_expgroup_4.pt` and 404s. The v4 startup log confirms `suffix=.safetensors`;
  without this patch neither miner could load the new global checkpoint at all.
- **`CONNITO_DIAG_EVERY=25`** diagnostic throttle — still absent upstream.
- **`upcast_trainable=False`** (bf16), 37 GB → 16 GB peak.
- **One-cycle LR** at `peak_lr 3e-5`, off the stock `1e-4`, via `ops/lr_override.json`
  (re-read every cycle, no restart needed).
- **Upload retry** in `ops/commit.py`.

## 5. Monitoring change

`ops/monitor.py` now requires the **same CRIT on two consecutive polls** before
auto-restarting. A single-poll CRIT restarted a *healthy* uid 250 at 02:32 on 2026-08-01
and desynced its commit worker into a 209-minute gap — the restart caused the exact
failure it was meant to prevent. A dead tmux window still restarts immediately; that
signal has no false-positive mode.

## 6. Open items

- [x] **Disk — caused a real outage, now guarded.** At 97 % full both miners crashed
      mid-save on 2026-08-01 06:11 / 06:16:
      `RuntimeError: [enforce fail at inline_container.cc:668] unexpected pos
      685328448 vs 685328336` — a truncated `torch.save` of `inner_optimizer.pt`.
      MinerCommit1 was due 06:39:47; both processes were restarting through it, so
      **cycle 16693 was missed entirely** ("no chain commit registered").
      Reclaimed ~46 GB from *re-fetchable* sources only — superseded
      `validator_checkpoint` downloads and older `exp_nemotron_c4` working
      checkpoints. **`exp_legal` was not touched** and is kept indefinitely.
      `checkpoint_topk` lowered 4 → 2, and `monitor.prune_checkpoints()` now enforces
      the same bound from outside the process, so it applies without restarting
      training. The guard is restricted to `PRUNABLE_GROUPS = ("exp_nemotron_c4",)`
      and to `ACTIVE_UIDS`, and orders by `inneropt_N` rather than mtime — a
      crash-truncated checkpoint can carry a newer mtime than the last good one.
- [ ] **Local corpus.** `ops/fetch_corpus.py --expert-group exp_nemotron_c4` was paused at
      8.7 GB to stop disk bleed. Miners currently stream from HF (the recommended path).
      Resume only after disk is resolved, and only with windowed tokenisation preserved.
- [ ] **First scored rounds.** Confirm the rank split produces *distinct* `val_loss` at the
      validator, not just locally. That is the whole hypothesis.
- [ ] **Burn.** 100 % as of 2026-08-01 — uid 0 holds incentive 1.0 and takes ~251 of ~295
      emission. **No miner earns anything at any rank right now.** Re-check before
      treating rank as income.
- [ ] uid 121 stays held back until 250/178 produce scores worth replicating.

## 7. Layout

```
configs/           group-4 config + the three uid configs
connito-patches/   diffs against upstream v0.4.0 / 4db4d8d
```

Re-apply patches after any `git pull` in `Connito/`:

```bash
cd /root/SN102/Connito
for p in ../v4/connito-patches/*.patch; do git apply "$p"; done
```
