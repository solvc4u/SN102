#!/usr/bin/env bash
# =============================================================================
# Launch the three SN102 miners, one per GPU, in tmux.
#
#   ./ops/launch.sh start     # start all three
#   ./ops/launch.sh start 250 # start one
#   ./ops/launch.sh stop
#   ./ops/launch.sh status
#   ./ops/launch.sh logs 250
#
# GPU allocation (operator decision):
#   GPU 0 -> uid 250        active
#   GPU 1 -> uid 178        active
#   GPU 2 -> uid 121        HELD BACK: candidate, enable once 250/178 score.
#                           Until then GPU 2 is the offline bench/tuning card.
#   GPU 3 -> subnet 96      NOT OURS. Never launch an SN102 miner on it.
#
# `start` with no argument starts only the ACTIVE uids. Naming a uid explicitly
# starts it regardless, so `start 121` still works when you want to promote it.
# =============================================================================
set -euo pipefail

OPS_ROOT="${OPS_ROOT:-/root/SN102/ops}"
ENV_FILE="$OPS_ROOT/.env"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "missing $ENV_FILE — copy env.template and fill it in:" >&2
  echo "  cp $OPS_ROOT/env.template $ENV_FILE && chmod 600 $ENV_FILE" >&2
  exit 1
fi
set -a; . "$ENV_FILE"; set +a

CONNITO_ROOT="${CONNITO_ROOT:-/root/SN102/Connito}"
VENV="${VENV:-/root/SN102/venv}"
LOG_DIR="${LOG_DIR:-/root/SN102/logs}"
mkdir -p "$LOG_DIR"

# uid:gpu:hotkey_var:token_var:proxy_var:data_seed:data_rank
#
# data_rank shards the TRAINING stream via split_dataset_by_node over
# world_size 10 (see the CONNITO_DATA_RANK patch in shared/dataloader.py).
# Deliberately NOT 1: `expert_groups/*/config.yaml` ships `rank: 1`, so every
# miner running the stock config streams that same slice. Ranks 3/7/5 are
# spread across the partition so our miners collide neither with the field nor
# with each other -- identical updates get mutually zeroed by the validator's
# duplicate heuristic, so distinct data is a precondition for scoring at all.
MINERS=(
  "250:0:HOTKEY_UID250:HF_TOKEN_UID250:PROXY_UID250:3303:2"
  "178:2:HOTKEY_UID178:HF_TOKEN_UID178:PROXY_UID178:2202:7"
  "121:1:HOTKEY_UID121:HF_TOKEN_UID121:PROXY_UID121:1101:8"
)

# uids started by a bare `start`. 121 is a candidate and stays out until 250
# and 178 have produced scores worth replicating.
ACTIVE_UIDS="${ACTIVE_UIDS:-250 178}"

# Card reserved for subnet 96. Guard against ever binding a miner to it.
FORBIDDEN_GPU="${FORBIDDEN_GPU:-3}"

# Uploads land inside a 10-block (~2 min) MinerCommit2 window. Offsetting the
# three phase loops keeps three multi-GB pushes from stacking on the same
# instant. See ops/README-429.md.
STAGGER_SECONDS="${STAGGER_SECONDS:-40}"

# Restarting at the wrong moment costs a submission, and a missed cycle counts
# against the 3-of-5 recency gate for Weight Group 1. Unsafe windows:
#   MinerCommit1 / MinerCommit2 -- the miner is signing/uploading right now
#   late Train                  -- the phase's training is thrown away
# Safe: Submission..ValidatorCommit2 (we already committed this cycle),
# Distribute, and early Train.  Override with FORCE_RESTART=1.
phase_guard() {
  [[ "${FORCE_RESTART:-0}" == "1" ]] && { echo "FORCE_RESTART=1 -- skipping phase guard" >&2; return 0; }
  local json name left
  json=$("$VENV/bin/python" - <<'PYG' 2>/dev/null
import urllib.request, json
try:
    d = json.load(urllib.request.urlopen(urllib.request.Request(
        "https://dashboard-dev.connito.ai/api/gw/api/v2/leaderboard",
        headers={"User-Agent": "sn102-launch"}), timeout=20))["data"]["phase"]
    print(f'{d["name"]} {d["blocks_remaining"]}')
except Exception:
    print("UNKNOWN 0")
PYG
)
  name=${json%% *}; left=${json##* }
  case "$name" in
    MinerCommit1|MinerCommit2)
      echo "REFUSING: phase=$name -- a restart here loses this cycle's submission." >&2
      echo "  wait ~$(( (left+10)*12/60 )) min, or FORCE_RESTART=1 to override." >&2
      return 1 ;;
    Train)
      if [[ "$left" -lt 200 ]]; then
        echo "REFUSING: phase=Train with only $left blocks left -- restarting now" >&2
        echo "  discards this cycle's training. Wait ~$(( left*12/60 )) min, or FORCE_RESTART=1." >&2
        return 1
      fi ;;
    UNKNOWN)
      echo "warning: could not read phase; proceeding" >&2 ;;
  esac
  echo "phase=$name ($left blocks left) -- safe to restart"
  return 0
}

preflight() {
  [[ -d "$VENV" ]] || { echo "no venv at $VENV — run ops/setup_env.sh first" >&2; exit 1; }
  # A local corpus is only required when a uid is actually pointed at one via
  # ops/dataset_<uid>.txt. The default is HF streaming, which needs no manifest,
  # and demanding one here blocked startup after the retired legal corpus was
  # cleared.
  for f in "$OPS_ROOT"/dataset_*.txt; do
    [[ -e "$f" ]] || continue
    if grep -q "LocalSharedDataset" "$f" 2>/dev/null; then
      local grp="${EXPERT_GROUP:-exp_nemotron_c4}"
      local manifest="$CORPUS_DIR/$grp/manifest.json"
      [[ -f "$manifest" ]] || { echo "no local corpus at $manifest (needed by $(basename "$f"))" >&2; exit 1; }
    fi
  done
  command -v tmux >/dev/null || { echo "tmux not installed: apt-get install -y tmux" >&2; exit 1; }
}

start_one() {
  local uid="$1" gpu="$2" hotkey_var="$3" token_var="$4" proxy_var="$5" seed="$6" drank="$7"
  local session="sn102-$uid"
  local config="$OPS_ROOT/configs/uid$uid.yaml"

  [[ -f "$config" ]] || { echo "missing $config — run: python ops/make_configs.py" >&2; return 1; }

  local token="${!token_var:-}"
  if [[ -z "$token" ]]; then
    echo "!! $token_var is empty — uid $uid cannot upload, and a failed upload" >&2
    echo "   means no HF coords on the chain commit, which counts as a missed" >&2
    echo "   round and drops you out of Weight Group 1. Refusing to start." >&2
    return 1
  fi
  local proxy="${!proxy_var:-}"

  # Check BOTH windows, not just the session. A session whose train window
  # died but whose commit window survived still answers has-session, which
  # silently skips the restart and leaves the miner training nothing.
  if tmux has-session -t "$session" 2>/dev/null; then
    local have_train have_commit
    have_train=$(tmux list-panes -t "$session:train" -F x 2>/dev/null | wc -l)
    have_commit=$(tmux list-panes -t "$session:commit" -F x 2>/dev/null | wc -l)
    if [[ "$have_train" -gt 0 && "$have_commit" -gt 0 ]]; then
      echo "uid $uid already running (tmux: $session)"
      return 0
    fi
    echo "uid $uid session is partial (train=$have_train commit=$have_commit) -- recycling"
    tmux kill-session -t "$session" 2>/dev/null || true
    sleep 2
  fi

  # Per-miner environment. CUDA_VISIBLE_DEVICES is how we pin the GPU:
  # train_worker hardcodes device = cuda:{rank} and rank is always 0 for a
  # single-GPU miner, so the GPU choice has to happen at the driver level.
  local envs=(
    "CUDA_VISIBLE_DEVICES=$gpu"
    "HF_TOKEN=$token"
    "HF_HOME=$HF_HOME"
    "HF_HUB_ENABLE_HF_TRANSFER=${HF_HUB_ENABLE_HF_TRANSFER:-1}"
    "CORPUS_DIR=$CORPUS_DIR"
    "CONNITO_DATA_SEED=$seed"
    "CONNITO_DATA_RANK=$drank"
    "CONNITO_DATA_WORLD=${#MINERS[@]}"
    "CONNITO_UID=$uid"
    ${FORCE_LR:+"CONNITO_FORCE_LR=$FORCE_LR"}
    ${FORCE_WARMUP:+"CONNITO_FORCE_WARMUP=$FORCE_WARMUP"}
    ${FORCE_MIN_FRAC:+"CONNITO_FORCE_MIN_FRAC=$FORCE_MIN_FRAC"}
    ${CKPT_SELECT:+"CONNITO_CKPT_SELECT=$CKPT_SELECT"}
    "TUNER_DB=$TUNER_DB"
    # Where THIS uid's LR search starts before it has a scored round. Lets each
    # miner explore from a different point instead of all three converging from
    # the same seed, which would waste the parallelism. Written by
    # `ops.setcfg startlr <uid> <value>`; falls back to CONNITO_START_LR.
    $(f="$OPS_ROOT/start_lr_$uid.txt"; [[ -f "$f" ]] && echo "CONNITO_START_LR=$(tr -d '\n' < "$f")")
    # Per-uid LR override if ops/lr_override_<uid>.json exists, else the shared
    # file. autolr re-reads whichever path this points at on every cycle
    # boundary, so the VALUE stays hot-swappable without a restart -- only
    # choosing a different FILE needs one, since the env is fixed at exec.
    #
    # Divergent LR per miner is not exotic here: the bandit itself ran 4e-4 on
    # 178 and 2e-5 on 250 at cycle 16640, before a shared override flattened
    # both to one value.
    "CONNITO_LR_OVERRIDE=$(
        if [[ -f "$OPS_ROOT/lr_override_$uid.json" ]]; then
            echo "$OPS_ROOT/lr_override_$uid.json"
        else
            echo "$OPS_ROOT/lr_override.json"
        fi)"
    # Per-uid dataset source, so local-vs-streaming can actually be A/B'd:
    # `dataset_class` is in the shared group config, so without this both
    # miners always match. Write "default" into ops/dataset_<uid>.txt to force
    # HF streaming for that uid, or a "pkg.mod:Class" path to force a class.
    ${DATASET_CLASS_OVERRIDE:+"CONNITO_DATASET_CLASS=$DATASET_CLASS_OVERRIDE"}
    $(f="$OPS_ROOT/dataset_$uid.txt"; [[ -f "$f" ]] && echo "CONNITO_DATASET_CLASS=$(tr -d '\n' < "$f")")
    "PYTHONPATH=$CONNITO_ROOT:/root/SN102"
    # 31.4GiB cards vs the 47GB A6000 this config is sized for. The first OOM
    # showed 2.16GiB "reserved but unallocated" -- pure fragmentation.
    # expandable_segments lets the allocator grow segments instead of stranding
    # them, which is exactly the recommended remedy in the OOM message.
    "PYTORCH_ALLOC_CONF=expandable_segments:True"
  )
  # Only the first miner keeps the Prometheus endpoint: the port is hardcoded
  # to 8100+rank and rank is 0 for all three, so the other two would collide.
  # The failure is caught and logged upstream, not fatal, but silencing it
  # keeps the logs readable.
  if [[ "$uid" != "250" ]]; then
    envs+=("ENABLE_TELEMETRY=false")
  fi
  if [[ -n "$proxy" ]]; then
    envs+=("HTTP_PROXY=$proxy" "HTTPS_PROXY=$proxy" "http_proxy=$proxy" "https_proxy=$proxy")
    echo "uid $uid: routing egress via proxy"
  fi

  # A miner is TWO processes. connito.miner.train only trains; Distribute
  # (download) and MinerCommit1/2 (submit) live in connito.miner.model_io,
  # wrapped here by ops/commit.py for upload retry. Running only the trainer
  # means training forever and never submitting.
  local train_log="$LOG_DIR/uid$uid-train.log"
  local commit_log="$LOG_DIR/uid$uid-commit.log"

  local train_cmd="cd $CONNITO_ROOT && ${envs[*]} $VENV/bin/python -m ops.autolr --uid $uid --path $config 2>&1 | tee -a $train_log"
  local commit_cmd="cd $CONNITO_ROOT && ${envs[*]} $VENV/bin/python -m ops.commit --path $config 2>&1 | tee -a $commit_log"

  # Stash both commands so cmd_respawn can recreate a window that tmux has
  # already destroyed. tmux drops a window when its command exits, so a trainer
  # that dies takes its window with it -- and `respawn-window` cannot target a
  # window that no longer exists. Writing them here keeps the recreated command
  # byte-identical to the one that was launched, rather than rebuilt from a
  # second copy of this env block that could drift.
  mkdir -p "$OPS_ROOT/.cmds"
  printf '%s' "$train_cmd"  > "$OPS_ROOT/.cmds/uid$uid-train.cmd"
  printf '%s' "$commit_cmd" > "$OPS_ROOT/.cmds/uid$uid-commit.cmd"

  tmux new-session  -d -s "$session" -n train "$train_cmd"
  tmux new-window  -t "$session" -n commit "$commit_cmd"
  echo "started uid $uid on GPU $gpu (tmux: $session [train|commit])"
  echo "  train:  $train_log"
  echo "  commit: $commit_log"
}

cmd_start() {
  preflight
  phase_guard || exit 1
  local want="${1:-all}"
  local first=1
  for entry in "${MINERS[@]}"; do
    IFS=: read -r uid gpu hk tok px seed drank <<<"$entry"
    if [[ "$want" == "all" ]]; then
      [[ " $ACTIVE_UIDS " == *" $uid "* ]] || continue
    else
      [[ "$want" == "$uid" ]] || continue
    fi
    if [[ "$gpu" == "$FORBIDDEN_GPU" ]]; then
      echo "refusing uid $uid: GPU $gpu is reserved for subnet 96" >&2
      continue
    fi
    if [[ $first -eq 0 ]]; then
      echo "staggering ${STAGGER_SECONDS}s..."
      sleep "$STAGGER_SECONDS"
    fi
    start_one "$uid" "$gpu" "$hk" "$tok" "$px" "$seed" "$drank" || true
    first=0
  done
}

cmd_restart() {
  preflight
  phase_guard || exit 1
  cmd_stop "${1:-all}"
  sleep 5
  # guard already passed; do not re-check and risk leaving miners down
  FORCE_RESTART=1 cmd_start "${1:-all}"
}

cmd_stop() {
  if [[ "${SKIP_GUARD:-0}" != "1" ]]; then
    phase_guard || { echo "  (use SKIP_GUARD=1 to stop anyway)" >&2; exit 1; }
  fi
  local want="${1:-all}"
  for entry in "${MINERS[@]}"; do
    IFS=: read -r uid _ <<<"$entry"
    [[ "$want" == "all" || "$want" == "$uid" ]] || continue
    if tmux has-session -t "sn102-$uid" 2>/dev/null; then
      tmux kill-session -t "sn102-$uid"
      echo "stopped uid $uid"
    fi
  done
}

cmd_status() {
  printf '%-6s %-6s %-10s %s\n' UID GPU STATE LOG
  for entry in "${MINERS[@]}"; do
    IFS=: read -r uid gpu _ <<<"$entry"
    local state="stopped"
    tmux has-session -t "sn102-$uid" 2>/dev/null && state="running"
    printf "%-6s %-6s %-10s %s\n" "$uid" "$gpu" "$state" "$LOG_DIR/uid$uid-{train,commit}.log"
  done
  echo
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
    --format=csv,noheader 2>/dev/null || true
}

# Respawn ONE window, leaving the other alone.
#
# A miner is two independent processes and they fail independently. On
# 2026-08-01 08:27 uid 250's trainer was wedged while its commit worker was
# healthy and four seconds from submitting an already-uploaded checkpoint --
# a whole-miner restart killed the submission to fix the trainer.
#
# `tmux respawn-window` reuses the window's original command, so the full env
# (CUDA_VISIBLE_DEVICES, CONNITO_DATA_RANK, tokens) is preserved without
# rebuilding it here and risking drift from start_one.
cmd_respawn() {
  local uid="${1:?usage: launch.sh respawn <uid> [train|commit]}" win="${2:-train}"
  case "$win" in train|commit) ;; *) echo "window must be train|commit" >&2; return 1 ;; esac
  local session="sn102-$uid"
  tmux has-session -t "$session" 2>/dev/null || { echo "no session $session -- use start" >&2; return 1; }

  # If the window still exists, respawn in place -- tmux reuses its original
  # command. If the process already exited, tmux destroyed the window with it,
  # and respawn-window has nothing to target; recreate it from the command
  # stashed at launch. Without this, a DEAD trainer (the case auto-repair most
  # needs to handle) is unrepairable: uid 178's trainer exited at 12:28 on
  # 2026-08-01 and sat down 4.6 hours while the monitor reported a respawn each
  # poll and tmux rejected it for a missing window.
  if tmux list-panes -t "$session:$win" >/dev/null 2>&1; then
    tmux respawn-window -k -t "$session:$win" || return 1
    echo "respawned $session:$win (other window untouched)"
  else
    local cmd_file="$OPS_ROOT/.cmds/uid$uid-$win.cmd"
    [[ -f "$cmd_file" ]] || { echo "window $win gone and no stashed command at $cmd_file -- use restart" >&2; return 1; }
    tmux new-window -d -t "$session" -n "$win" "$(cat "$cmd_file")" || return 1
    echo "recreated $session:$win from stashed command (other window untouched)"
  fi
}

case "${1:-}" in
  start)   cmd_start "${2:-all}" ;;
  restart) cmd_restart "${2:-all}" ;;
  respawn) cmd_respawn "${2:-}" "${3:-train}" ;;
  stop)    cmd_stop "${2:-all}" ;;
  status) cmd_status ;;
  logs)   tail -f "$LOG_DIR/uid${2:?usage: launch.sh logs <uid> [train|commit]}-${3:-train}.log" ;;
  *) echo "usage: $0 {start [uid]|restart [uid]|respawn <uid> [train|commit]|stop|status|logs <uid>}" >&2; exit 1 ;;
esac
