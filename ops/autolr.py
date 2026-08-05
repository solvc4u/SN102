#!/usr/bin/env python3
"""
Adaptive learning-rate controller for the SN102 miner.

Run it exactly where you would run the stock trainer:

    python -m ops.autolr --path /root/SN102/ops/configs/uid250.yaml

It imports the stock `connito.miner.train`, replaces one function, and hands
control straight back. No fork of the miner, so upstream `git pull` keeps working.

--------------------------------------------------------------------------------
Why this is the lever worth pulling
--------------------------------------------------------------------------------
`connito/miner/train.py:setup_training` builds the schedule as:

    scheduler = get_cosine_schedule_with_warmup(
        inner_optimizer,
        num_warmup_steps=config.sched.warmup_steps,   # default 0
        num_training_steps=config.sched.total_steps,  # 88_000, LOCKED
    )

A Train phase is 300 blocks ~= 60 minutes. Nobody gets anywhere near 88,000
optimizer steps in an hour, so every miner running stock traverses a sliver of
that cosine and trains at an effectively **constant 1e-4**. That is the reason
~100 miners on the live leaderboard land inside a 0.00045 band of each other:
they are all running the identical fixed-LR recipe and the spread between them
is mostly noise.

`opt.lr` and `sched.warmup_steps` are not in `_LOCKED_FIELDS`, and the schedule
shape is ours to choose. This module replaces the sliver-of-a-cosine with a
true one-cycle schedule sized to the *actual* Train phase, so LR anneals to its
floor exactly when the checkpoint is captured for MinerCommit1. The final
anneal is where the last few thousandths of loss come from, and a few
thousandths is the entire margin between rank 1 and rank 20.

--------------------------------------------------------------------------------
Two feedback loops
--------------------------------------------------------------------------------
1. **Within a cycle** — phase-aware annealing. The scheduler polls the owner
   phase service for position within Train and drives the cosine off real
   progress, so it self-corrects if throughput changes (thermal throttling,
   a co-located miner starting up, a slow Distribute).

2. **Across cycles** — the tuner reads back the `val_loss` the validators
   actually recorded for this UID from the public dashboard API and uses it to
   pick the next cycle's peak LR. It optimises the scored quantity directly
   rather than a local proxy, which matters because our probe set and the
   validator's seeded eval slice are not the same rows.

State lives in $TUNER_DB (sqlite). Safe to stop and restart; history persists.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sqlite3
import sys
import threading
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

PHASE_API = "https://cycle-api.connito.ai/get_phase"
# The endpoint the dashboard UI itself calls. v1 served val_loss from a
# Prometheus gauge that is never cleared at round rollover, so most miners show
# a stale number with no way to tell which; dashboard-api-v2 /api/v3 lags the UI
# by one to two cycles. This one agrees with the UI exactly, and the tuner is
# only as good as the feedback it optimises against.
DASHBOARD_API = os.environ.get(
    "CONNITO_DASHBOARD_API",
    "https://dashboard-dev.connito.ai/api/gw/api/v2/leaderboard")

# Log-spaced peak-LR grid. Centred an order of magnitude around the stock 1e-4
# so the tuner can discover that stock is optimal if it in fact is.
LR_GRID = [2e-5, 3e-5, 5e-5, 7e-5, 1e-4, 1.4e-4, 2e-4, 3e-4, 4e-4, 6e-4]
# Where the search starts before any result exists for this uid. Operator
# choice, not a measurement -- the tuner walks away from it as soon as scored
# rounds arrive.
START_LR = float(os.environ.get("CONNITO_START_LR", "3e-4"))
WARMUP_GRID = [0.0, 0.03, 0.10]
MIN_LR_FRAC_GRID = [0.02, 0.10]

EXPLORE_CYCLES = 12      # pure exploration before exploitation kicks in
EPSILON = 0.2            # exploration rate afterwards

# --- drift control -----------------------------------------------------------
# Measured 2026-07-29 by ops/weight_diff.py against the live global checkpoint
# (g-connito/co@ecb1c3b), comparing published miner shards tensor-by-tensor:
#
#   Attila115/co3   rank 1  val_loss 1.773455  rel-L2 0.070855  cos 0.997537
#   hunter-04/co228 rank 4  val_loss 1.774909  rel-L2 0.070864  cos 0.997535
#   solvc4u/co2     OURS    val_loss 2.008251  rel-L2 0.118204  cos 0.993711
#   solvc4u/co_lo   OURS    val_loss 2.133191  rel-L2 0.121713  cos 0.993309
#
# Two unrelated top operators agree on rel-L2 to FOUR decimal places. That is a
# converged recipe, not coincidence. All four move in the same direction
# (cosine > 0.993) -- we simply travel ~1.7x too far, and val_loss rises
# monotonically with distance across every sample we have.
#
# So drift is a far better control signal than val_loss:
#   * measurable from our own artifacts, no validator round-trip
#   * available every cycle instead of every 3-4
#   * free of baseline noise (baseline swung 1.61 -> 5.32 -> 2.29 in a day)
TARGET_DRIFT = float(os.environ.get("CONNITO_TARGET_DRIFT", "0.0119"))
DRIFT_LR_MIN = 1e-6
DRIFT_LR_MAX = 1e-3
DRIFT_GAIN = 1.0         # 1.0 = full proportional correction
DRIFT_MAX_STEP = 2.0     # never move LR by more than this factor in one cycle


# ==============================================================================
# Phase tracking
# ==============================================================================

@dataclass
class PhasePosition:
    name: str
    fraction: float      # 0.0 at phase start, 1.0 at phase end
    cycle_index: int


class PhaseTracker:
    """
    Background poller for the owner phase service.

    Deliberately fail-soft: if the API is unreachable the scheduler must not
    stall or crash the miner, so we fall back to a wall-clock estimate of a
    300-block Train phase. A degraded schedule is survivable; a dead miner
    misses the round and drops out of Weight Group 1.
    """

    TRAIN_SECONDS = 300 * 12

    # 120s, not 20s. cycle-api.connito.ai returns 429 under load and the stock
    # miner already polls it in wait_till(); this tracker was adding a second
    # poller per miner on top. Each 429 costs an 8s backoff INSIDE the training
    # loop (~24s stalls observed). Phases run 20-300 blocks (4-60 min), so 120s
    # resolution loses nothing -- the cosine is driven by phase fraction, which
    # moves slowly.
    def __init__(self, poll_sec: float = 120.0) -> None:
        self._lock = threading.Lock()
        self._pos = PhasePosition("Unknown", 0.0, 0)
        self._last_ok = 0.0
        self._train_started_monotonic: float | None = None
        self._poll_sec = poll_sec
        t = threading.Thread(target=self._loop, daemon=True, name="phase-tracker")
        t.start()

    def _poll_once(self) -> None:
        req = urllib.request.Request(PHASE_API, headers={"User-Agent": "connito-miner"})
        with urllib.request.urlopen(req, timeout=8) as r:
            data = json.loads(r.read())
        name = str(data.get("phase_name", "Unknown"))
        into = float(data.get("blocks_into_phase", 0) or 0)
        left = float(data.get("blocks_remaining_in_phase", 0) or 0)
        span = into + left
        frac = (into / span) if span > 0 else 0.0
        with self._lock:
            if name == "Train" and self._pos.name != "Train":
                self._train_started_monotonic = time.monotonic()
            self._pos = PhasePosition(name, min(1.0, max(0.0, frac)), int(data.get("cycle_index", 0) or 0))
            self._last_ok = time.monotonic()

    def _loop(self) -> None:
        while True:
            try:
                self._poll_once()
            except Exception:  # noqa: BLE001 - fail-soft by design
                pass
            time.sleep(self._poll_sec)

    def position(self) -> PhasePosition:
        with self._lock:
            pos = self._pos
            stale = (time.monotonic() - self._last_ok) > 120
            started = self._train_started_monotonic
        if not stale:
            return pos
        # API stale: estimate from wall clock since Train was last seen.
        if started is not None:
            frac = min(1.0, (time.monotonic() - started) / self.TRAIN_SECONDS)
            return PhasePosition("Train", frac, pos.cycle_index)
        return pos


# ==============================================================================
# Cross-cycle tuner
# ==============================================================================

class Tuner:
    def __init__(self, db_path: str, uid: int) -> None:
        self.uid = uid
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self.db = sqlite3.connect(db_path, check_same_thread=False)
        self.db.execute(
            """
            CREATE TABLE IF NOT EXISTS trials (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                uid           INTEGER NOT NULL,
                cycle_index   INTEGER NOT NULL,
                started_at    REAL    NOT NULL,
                peak_lr       REAL    NOT NULL,
                warmup_frac   REAL    NOT NULL,
                min_lr_frac   REAL    NOT NULL,
                val_loss      REAL,
                baseline_loss REAL,
                drift         REAL,
                UNIQUE(uid, cycle_index)
            )
            """
        )
        # Older DBs predate the drift column; add it in place rather than
        # forcing operators to drop accumulated history.
        cols = {r[1] for r in self.db.execute("PRAGMA table_info(trials)")}
        if "drift" not in cols:
            self.db.execute("ALTER TABLE trials ADD COLUMN drift REAL")
        self.db.commit()

    # -- drift feedback -----------------------------------------------------
    def record_drift(self, cycle_index: int, drift: float) -> None:
        self.db.execute(
            "UPDATE trials SET drift=? WHERE uid=? AND cycle_index=?",
            (drift, self.uid, cycle_index),
        )
        self.db.commit()

    def last_drift_pair(self) -> tuple[float, float] | None:
        """Most recent (peak_lr, drift) this uid actually produced."""
        row = self.db.execute(
            "SELECT peak_lr, drift FROM trials WHERE uid=? AND drift IS NOT NULL "
            "ORDER BY cycle_index DESC LIMIT 1",
            (self.uid,),
        ).fetchone()
        return (float(row[0]), float(row[1])) if row and row[1] else None

    def choose_by_drift(self) -> tuple[float, float, float] | None:
        """Proportional control on measured drift.

        drift scales close to linearly with total update magnitude, so
        next_lr = lr * (target / measured) is a sound first-order correction.
        Clamped per-cycle so one bad measurement cannot slam the LR.
        """
        pair = self.last_drift_pair()
        if pair is None:
            return None
        lr, drift = pair
        if drift <= 0:
            return None
        ratio = (TARGET_DRIFT / drift) ** DRIFT_GAIN
        ratio = max(1.0 / DRIFT_MAX_STEP, min(DRIFT_MAX_STEP, ratio))
        new_lr = max(DRIFT_LR_MIN, min(DRIFT_LR_MAX, lr * ratio))
        print(
            f"[autolr] drift control: last lr={lr:.2e} produced drift={drift:.6f}, "
            f"target={TARGET_DRIFT:.6f} -> lr x{ratio:.3f} = {new_lr:.2e}"
        )
        return (new_lr, 0.03, 0.02)

    # -- parameter selection ------------------------------------------------
    def choose(self, cycle_index: int) -> tuple[float, float, float]:
        import random

        rows = self.db.execute(
            "SELECT peak_lr, warmup_frac, min_lr_frac, val_loss FROM trials "
            "WHERE uid=? AND val_loss IS NOT NULL",
            (self.uid,),
        ).fetchall()

        rng = random.Random(cycle_index * 7919 + self.uid)

        # No scored round for this uid yet -> start where the operator asked
        # rather than at whatever grid cell the sweep index lands on. Once even
        # one result exists the sweep takes over and this never fires again.
        if not rows:
            return (START_LR, 0.03, 0.02)

        if len(rows) < EXPLORE_CYCLES:
            # Deterministic sweep so the three UIDs cover the grid in parallel
            # rather than all racing to the same cell.
            i = len(rows) + self.uid
            return (
                LR_GRID[i % len(LR_GRID)],
                WARMUP_GRID[(i // len(LR_GRID)) % len(WARMUP_GRID)],
                MIN_LR_FRAC_GRID[i % len(MIN_LR_FRAC_GRID)],
            )

        if rng.random() < EPSILON:
            return (rng.choice(LR_GRID), rng.choice(WARMUP_GRID), rng.choice(MIN_LR_FRAC_GRID))

        best = min(rows, key=lambda r: r[3])
        # Exploit with a small jitter around the incumbent so we keep refining
        # instead of re-running an identical trial forever.
        idx = min(range(len(LR_GRID)), key=lambda k: abs(math.log(LR_GRID[k]) - math.log(best[0])))
        idx = max(0, min(len(LR_GRID) - 1, idx + rng.choice([-1, 0, 0, 1])))
        return (LR_GRID[idx], best[1], best[2])

    def record_start(self, cycle_index: int, peak_lr: float, warmup: float, min_frac: float) -> None:
        self.db.execute(
            "INSERT OR IGNORE INTO trials (uid, cycle_index, started_at, peak_lr, warmup_frac, min_lr_frac) "
            "VALUES (?,?,?,?,?,?)",
            (self.uid, cycle_index, time.time(), peak_lr, warmup, min_frac),
        )
        self.db.commit()

    def record_result(self, cycle_index: int, val_loss: float, baseline: float | None) -> None:
        self.db.execute(
            "UPDATE trials SET val_loss=?, baseline_loss=? WHERE uid=? AND cycle_index=?",
            (val_loss, baseline, self.uid, cycle_index),
        )
        self.db.commit()

    # -- feedback from the live subnet --------------------------------------
    def poll_observed(self, cycle_index: int) -> None:
        """Read back what validators actually scored for this UID."""
        try:
            req = urllib.request.Request(DASHBOARD_API, headers={"User-Agent": "connito-miner"})
            with urllib.request.urlopen(req, timeout=25) as r:
                payload = json.loads(r.read())
        except Exception as exc:  # noqa: BLE001
            print(f"[autolr] dashboard poll failed ({exc}); tuner will retry next cycle")
            return
        data = payload.get("data", {})
        baseline = (data.get("round") or {}).get("baseline_loss")
        for m in data.get("leaderboard", []):
            if int(m.get("uid", -1)) != self.uid:
                continue
            # gw/v2 reports val_loss per validator slot, not at top level.
            # Take the freshest ok sample rather than the first or the min --
            # min() picks whichever validator likes us best, which flattered
            # our numbers badly enough to drive two wrong config changes.
            vl = m.get("val_loss")
            if vl is None:
                cands = [v for v in (m.get("validator_metrics") or [])
                         if v.get("val_loss") is not None]
                if cands:
                    vl = max(cands, key=lambda v: (
                        v.get("sample_cycle") if v.get("sample_cycle") is not None else -1,
                        1 if v.get("eval_status_label") == "ok" else 0,
                    )).get("val_loss")
            if vl is None:
                print(f"[autolr] uid {self.uid}: no val_loss recorded this round")
                return
            self.record_result(cycle_index, float(vl), baseline)
            delta = (baseline - float(vl)) if baseline else None
            print(
                f"[autolr] uid {self.uid} cycle {cycle_index}: "
                f"val_loss={vl:.6f} baseline={baseline} delta={delta}"
            )
            return


# ==============================================================================
# The scheduler
# ==============================================================================

class PhaseAwareOneCycle:
    """
    Duck-typed LRScheduler. `setup_training` only ever calls `.step()` on this
    object and hands it to `save_checkpoint`, so implementing the LRScheduler
    surface (`step`, `get_last_lr`, `state_dict`, `load_state_dict`) is enough.

    LR is a pure function of position within the Train phase, not of step
    count. That is the point: step count depends on throughput we cannot
    predict, phase position is authoritative and chain-driven.
    """

    def __init__(self, optimizer, tracker: PhaseTracker, tuner: Tuner) -> None:
        self.optimizer = optimizer
        self.tracker = tracker
        self.tuner = tuner
        self.base_lrs = [g["lr"] for g in optimizer.param_groups]

        self._cycle = -1
        self._peak = 1e-4
        self._warmup = 0.03
        self._min_frac = 0.05
        self._last_lr = list(self.base_lrs)
        self._steps = 0
        self._retune(self.tracker.position())

    def _retune(self, pos: PhasePosition) -> None:
        """New cycle: bank the previous result, pick the next parameters."""
        if pos.cycle_index == self._cycle:
            return
        if self._cycle >= 0:
            self.tuner.poll_observed(self._cycle)
        self._cycle = pos.cycle_index

        # CONNITO_FORCE_LR pins the peak LR and bypasses the bandit. Its reason
        # for existing is the zero-LR control experiment: set it to ~1e-8 and
        # the miner submits the global checkpoint essentially untouched, which
        # measures what "do nothing" scores.
        #
        # That number gates every other decision. 25 miners sit inside 0.001 of
        # each other at val_loss ~1.9105 while we score 2.1332 -- a cluster that
        # tight means they are submitting near-identical weights, i.e. barely
        # training. If the untrained floor is ~1.910 then our training is moving
        # the model AWAY from the eval distribution and no LR schedule, batch
        # size or throughput work fixes that. If the floor is ~2.13 the problem
        # is structural and upstream of training entirely.
        #
        # The result is still recorded as a normal trial: a tiny LR that scores
        # well is a genuine finding the bandit should exploit, not an outlier
        # to discard.
        # File-based override, re-read every cycle boundary.
        #
        # CONNITO_FORCE_LR lives in the process environment, which is fixed at
        # exec -- so every LR change previously required a restart, and every
        # restart forfeits that cycle's commit (scheduler_service always waits
        # for the next Distribute). This file is re-read here on each new cycle,
        # so LR can be retuned live:
        #     echo '{"peak_lr": 5e-5}' > ops/lr_override.json
        # Same principle that let the corpus hot-swap work without a restart.
        # Optional keys: peak_lr, warmup, min_frac. Malformed file is ignored
        # rather than crashing the miner mid-run.
        self._apply_grad_accum_override(pos)

        forced = os.environ.get("CONNITO_FORCE_LR")
        override_path = os.environ.get(
            "CONNITO_LR_OVERRIDE", "/root/SN102/ops/lr_override.json")
        try:
            with open(override_path) as fh:
                ov = json.load(fh)
            if ov.get("peak_lr"):
                self._peak = float(ov["peak_lr"])
                self._warmup = float(ov.get("warmup", 0.03))
                self._min_frac = float(ov.get("min_frac", 0.02))
                self.tuner.record_start(pos.cycle_index, self._peak, self._warmup, self._min_frac)
                print(f"[autolr] cycle {pos.cycle_index}: OVERRIDE peak_lr={self._peak:.2e} "
                      f"warmup={self._warmup:.0%} floor={self._min_frac:.0%} (from {override_path})")
                return
        except FileNotFoundError:
            pass
        except Exception as exc:  # noqa: BLE001 - never break training on a bad file
            print(f"[autolr] ignoring bad LR override ({exc})")

        if forced:
            self._peak = float(forced)
            self._warmup = float(os.environ.get("CONNITO_FORCE_WARMUP", "0.0"))
            self._min_frac = float(os.environ.get("CONNITO_FORCE_MIN_FRAC", "1.0"))
            self.tuner.record_start(pos.cycle_index, self._peak, self._warmup, self._min_frac)
            print(
                f"[autolr] cycle {pos.cycle_index}: FORCED peak_lr={self._peak:.2e} "
                f"warmup={self._warmup:.0%} floor={self._min_frac:.0%} (control experiment)"
            )
            return

        # Drift control first: it optimises the quantity that actually predicts
        # rank. Fall back to the bandit only until the first drift measurement
        # exists.
        chosen = self.tuner.choose_by_drift()
        if chosen is None:
            chosen = self.tuner.choose(pos.cycle_index)
        self._peak, self._warmup, self._min_frac = chosen
        self.tuner.record_start(pos.cycle_index, self._peak, self._warmup, self._min_frac)
        print(
            f"[autolr] cycle {pos.cycle_index}: peak_lr={self._peak:.2e} "
            f"warmup={self._warmup:.0%} floor={self._min_frac:.0%}"
        )

    def _apply_grad_accum_override(self, pos: PhasePosition) -> None:
        """Retune gradient accumulation live, without restarting the miner.

        Why this is the knob that matters. The shipped config runs
        `per_device_train_batch_size: 1` with `gradient_accumulation_steps: 1`,
        so every optimizer step is estimated from ONE 1024-token document.
        A gradient from a single document is almost pure noise, and two such
        estimates from different documents are near-orthogonal by construction
        -- which is exactly what the weight comparison shows against the miners
        at the top of the board:

            uid 254 (1.2513)  ||delta|| 248.4  cos(us) +0.018
            uid 139 (1.2519)  ||delta|| 248.1  cos(us) +0.018
            per-layer cos +0.02..+0.03 across ALL 26 layers,
            none aligned (>0.1), none opposed (<-0.1)

        Uniform low cosine on identical architecture, task and data sources is
        the signature of sampling noise, not of a different objective -- a
        different objective would show some layers agreeing and others opposing.
        Learning rate cannot fix it: LR scales how far we step along a noisy
        direction, which is why sweeping it 23x (3e-6 -> 3e-4) moved ||delta||
        a great deal and val_loss almost not at all.

        Accumulation costs no extra VRAM -- the micro-batch stays 1, only the
        number of backwards per optimizer step changes -- so it is safe on cards
        that already OOM'd at 31.08 GiB with upcast_trainable=True.

        `config.local_par.gradient_accumulation_steps` is read on EVERY step
        (train.py:368-369), so assigning it here takes effect immediately rather
        than at the next restart. Control file, re-read each cycle:

            ops/gradaccum_<uid>.json   {"gradient_accumulation_steps": 16}

        Missing or malformed file leaves the configured value untouched.
        """
        path = os.environ.get(
            "CONNITO_GRAD_ACCUM_OVERRIDE",
            f"/root/SN102/ops/gradaccum_{os.environ.get('CONNITO_UID','')}.json")
        try:
            with open(path) as fh:
                want = int(json.load(fh)["gradient_accumulation_steps"])
        except FileNotFoundError:
            return
        except Exception as exc:  # noqa: BLE001 - never break training on a bad file
            print(f"[autolr] ignoring bad grad-accum override ({exc})")
            return
        if want < 1:
            return
        lp = getattr(_CAPTURED_CONFIG, "local_par", None) if _CAPTURED_CONFIG is not None else None
        if lp is None:
            return
        cur = getattr(lp, "gradient_accumulation_steps", None)
        if cur == want:
            return
        try:
            lp.gradient_accumulation_steps = want
        except Exception as exc:  # noqa: BLE001 - pydantic may freeze the field
            print(f"[autolr] cannot apply grad-accum override ({exc})")
            return
        print(f"[autolr] cycle {pos.cycle_index}: gradient_accumulation_steps "
              f"{cur} -> {want} (from {path}); effective batch = "
              f"{want} x per_device_train_batch_size")

    def _lr_for(self, pos: PhasePosition) -> float:
        if pos.name != "Train":
            # Outside Train nothing we produce is committed; hold at the floor
            # so a long idle stretch cannot drift the weights.
            return self._peak * self._min_frac
        p = pos.fraction
        if self._warmup > 0 and p < self._warmup:
            return self._peak * (p / self._warmup)
        # Cosine from peak down to floor across the remainder of Train.
        span = max(1e-6, 1.0 - self._warmup)
        q = min(1.0, (p - self._warmup) / span)
        cos = 0.5 * (1.0 + math.cos(math.pi * q))
        return self._peak * (self._min_frac + (1.0 - self._min_frac) * cos)

    def step(self, *_args, **_kwargs) -> None:
        self._steps += 1
        pos = self.tracker.position()
        self._retune(pos)
        lr = self._lr_for(pos)
        for g in self.optimizer.param_groups:
            g["lr"] = lr
        self._last_lr = [lr] * len(self.optimizer.param_groups)

    def get_last_lr(self) -> list[float]:
        return list(self._last_lr)

    def state_dict(self) -> dict:
        return {
            "steps": self._steps,
            "cycle": self._cycle,
            "peak": self._peak,
            "warmup": self._warmup,
            "min_frac": self._min_frac,
        }

    def load_state_dict(self, state: dict) -> None:
        self._steps = int(state.get("steps", 0))
        self._cycle = int(state.get("cycle", -1))
        self._peak = float(state.get("peak", 1e-4))
        self._warmup = float(state.get("warmup", 0.03))
        self._min_frac = float(state.get("min_frac", 0.05))


# ==============================================================================
# Entry point
# ==============================================================================

def install_memory_profile() -> None:
    """Keep trainable params in bf16 instead of upcasting to fp32.

    Measured on this box (ops/memprobe.py, logs/memprobe.log):

        upcast=True   load 8953MiB -> freeze 15405MiB -> backward 28532MiB
                      -> OOM allocating 8-bit optimizer state.  ~37GiB needed.
        upcast=False  load 8953MiB -> freeze  8953MiB -> backward 15629MiB
                      -> peak 15997MiB of 32111MiB (49.8%).

    The miner trains 3.38B params (16 local experts x 26 layers), not the
    ~225M a one-expert-per-layer reading of the assignment file suggests.
    `freeze_parameters(upcast_trainable=True)` therefore pays fp32 twice --
    once for the weights (+6.5GiB) and again for their gradients (+6.4GiB) --
    which is what puts the config over a 31.4GiB card. The stock profile is
    sized for a 47GB A6000.

    TRADEOFF, stated plainly: bf16 has ~8 mantissa bits, so dropping the fp32
    master copy loses update precision, and on this subnet precision IS the
    product -- the field is clustered inside 0.0005 of val_loss. AdamW8bit's
    blockwise dynamic quantisation is built for exactly this regime, but if
    A/B against the stock profile ever shows bf16 costing more val_loss than
    the headroom is worth, set CONNITO_UPCAST_TRAINABLE=1 and instead reclaim
    memory elsewhere (paged optimizer state, smaller batch).
    """
    if os.environ.get("CONNITO_UPCAST_TRAINABLE", "0") == "1":
        print("[mem] CONNITO_UPCAST_TRAINABLE=1 -> stock fp32 upcast (needs >32GiB)")
        return

    import connito.miner.train as train_mod

    original = train_mod.freeze_parameters

    def no_upcast(*args, **kwargs):
        kwargs["upcast_trainable"] = False
        return original(*args, **kwargs)

    train_mod.freeze_parameters = no_upcast
    print("[mem] trainable params kept in bf16 (upcast_trainable=False)")


# The live MinerConfig, captured when setup_training first receives it.
# autolr's main() never sees it -- it is built inside run_distributed_training --
# but `_apply_grad_accum_override` needs it to retune accumulation without a
# restart, and train.py reads local_par.gradient_accumulation_steps on every
# step so a live mutation takes effect immediately.
_CAPTURED_CONFIG = None


EVAL_LOG_NAME = "local_eval.json"


def install_eval_recorder() -> None:
    """Record every local eval as {inner_opt_step: val_loss} next to the checkpoints.

    `connito/miner/train.py` runs `evaluate_model(...)` every
    `config.log.metric_interval` steps and only logs the result. Meanwhile the
    commit path calls `select_best_checkpoint`, which despite the name orders by
    (active, global_ver, inner_opt) and returns the NEWEST checkpoint -- not the
    best one.

    Training loss here is noisy (per-batch sigma ~0.9 at batch 4), so the last
    step is not reliably the best model. Recording the eval curve lets
    ops/commit.py submit the lowest-loss checkpoint instead of the last one.
    Most miners submit their last; this is a cheap mechanism edge.

    The two processes are separate, so they hand off through this file.
    """
    import connito.miner.train as train_mod

    original = train_mod.evaluate_model

    def recording_eval(*args, **kwargs):
        out = original(*args, **kwargs)
        try:
            step = kwargs.get("step")
            vl = (out or {}).get("val_loss")
            if step is not None and vl is not None and vl == vl and vl != float("inf"):
                cfg = _EVAL_CFG.get("config")
                if cfg is not None:
                    path = Path(cfg.ckpt.checkpoint_path) / EVAL_LOG_NAME
                    try:
                        data = json.loads(path.read_text())
                    except Exception:  # noqa: BLE001
                        data = {}
                    data[str(int(step))] = float(vl)
                    # keep the file small; only recent steps matter
                    if len(data) > 200:
                        for k in sorted(data, key=lambda k: int(k))[:-200]:
                            data.pop(k, None)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(json.dumps(data))
                    print(f"[eval] step {step}: val_loss={vl:.6f} -> {path.name}")
        except Exception as exc:  # noqa: BLE001
            print(f"[eval] could not record: {exc}")
        return out

    train_mod.evaluate_model = recording_eval

    # capture the config once train_worker builds it
    orig_worker = train_mod.train_worker

    def worker(rank, world_size, config):
        _EVAL_CFG["config"] = config
        return orig_worker(rank, world_size, config)

    train_mod.train_worker = worker
    print("[eval] local-eval recorder installed")


_EVAL_CFG: dict = {}


def install_config_capture(train_mod) -> None:
    """Stash the MinerConfig the moment setup_training receives it."""
    orig = train_mod.setup_training

    def capturing(config, *a, **kw):
        global _CAPTURED_CONFIG
        _CAPTURED_CONFIG = config
        return orig(config, *a, **kw)

    train_mod.setup_training = capturing
    print("[autolr] config capture installed (enables live grad-accum retune)")


def main() -> int:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--uid", type=int, default=int(os.environ.get("CONNITO_UID", "0")))
    ap.add_argument("--tuner-db", default=os.environ.get("TUNER_DB", "/root/SN102/data/tuner.sqlite"))
    ap.add_argument("--no-autolr", action="store_true", help="run the stock schedule (A/B control)")
    known, passthrough = ap.parse_known_args()

    # Hand the remaining argv (--path, --debug, ...) to the stock parser.
    sys.argv = [sys.argv[0]] + passthrough

    import connito.miner.train as train_mod

    install_memory_profile()
    install_eval_recorder()
    install_config_capture(train_mod)

    if known.no_autolr:
        print("[autolr] disabled; using stock cosine schedule")
        return train_mod.run_distributed_training() or 0

    tracker = PhaseTracker()
    tuner = Tuner(known.tuner_db, known.uid)

    def _factory(optimizer, num_warmup_steps=None, num_training_steps=None, **_kw):
        print(
            f"[autolr] replacing cosine(warmup={num_warmup_steps}, total={num_training_steps}) "
            f"with phase-aware one-cycle"
        )
        return PhaseAwareOneCycle(optimizer, tracker, tuner)

    # The name is imported into the train module's namespace at import time
    # (`from transformers import get_cosine_schedule_with_warmup`), so patching
    # the module attribute is what takes effect in setup_training.
    train_mod.get_cosine_schedule_with_warmup = _factory

    return train_mod.run_distributed_training() or 0


if __name__ == "__main__":
    raise SystemExit(main())
