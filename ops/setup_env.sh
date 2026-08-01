#!/usr/bin/env bash
# =============================================================================
# Build the SN102 mining environment for 4x RTX 5090 (Blackwell, sm_120).
#
#   ./ops/setup_env.sh
#
# Creates /root/SN102/venv. Does NOT touch /root/.venv, which already holds an
# unrelated bittensor 11.0.1 that conflicts with this repo's pinned 10.5.0.
# =============================================================================
set -euo pipefail

VENV="${VENV:-/root/SN102/venv}"
CONNITO_ROOT="${CONNITO_ROOT:-/root/SN102/Connito}"

# RTX 5090 is sm_120. cu128 is the first CUDA build line with Blackwell kernels;
# an older wheel imports fine and then fails at the first kernel launch with
# "no kernel image is available for execution on the device".
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"

# Large CUDA wheels (torch is ~917MB) routinely truncate mid-download and pip
# surfaces that as IncompleteRead/ProtocolError. Retry rather than making the
# operator restart a 900MB download by hand.
pip_retry() {
  local attempt
  for attempt in 1 2 3 4 5; do
    if "$VENV/bin/pip" install --retries 10 --timeout 120 "$@"; then
      return 0
    fi
    echo "--- pip attempt $attempt failed; retrying in $((attempt * 10))s ---" >&2
    sleep $((attempt * 10))
  done
  echo "pip install failed after 5 attempts: $*" >&2
  return 1
}

echo "=== [1/6] venv ==="
python3 -m venv "$VENV"
pip_retry --upgrade pip wheel setuptools

echo "=== [2/6] torch (cu128 / Blackwell) ==="
pip_retry --index-url "$TORCH_INDEX" torch==2.10.0

echo "=== [3/6] verifying sm_120 kernels actually run ==="
# Import success is not proof. Launch a real kernel.
"$VENV/bin/python" - <<'PY'
import sys, torch
print("torch", torch.__version__, "cuda", torch.version.cuda)
if not torch.cuda.is_available():
    sys.exit("FAIL: no CUDA device visible")
n = torch.cuda.device_count()
print("devices:", n)
for i in range(n):
    cap = torch.cuda.get_device_capability(i)
    print(f"  cuda:{i} {torch.cuda.get_device_name(i)} sm_{cap[0]}{cap[1]} "
          f"{torch.cuda.get_device_properties(i).total_memory/2**30:.1f}GiB")
x = torch.randn(512, 512, device="cuda:0")
y = (x @ x).sum().item()
assert y == y, "matmul produced NaN"
print("matmul on cuda:0 OK")
print("bf16 supported:", torch.cuda.is_bf16_supported())
PY

echo "=== [4/6] repo requirements ==="
# torch is already installed from the CUDA index; don't let the pin drag in a
# CPU wheel from PyPI over the top of it.
grep -v '^torch==' "$CONNITO_ROOT/requirements.txt" > /tmp/reqs-no-torch.txt
pip_retry -r /tmp/reqs-no-torch.txt

echo "=== [5/6] bitsandbytes (8-bit AdamW state) ==="
# The fp32 optimizer path is sized for a 47GB A6000; we have 32GB cards. 8-bit
# AdamW state is the documented mitigation (connito/shared/config.py:OptimizerCfg)
# and configs set opt.adamw_optim_bits=8. bitsandbytes must have sm_120 kernels
# or the miner dies at optimizer construction.
pip_retry "bitsandbytes>=0.45"
"$VENV/bin/python" - <<'PY'
import sys, torch
try:
    import bitsandbytes as bnb
except Exception as exc:
    sys.exit(f"FAIL: bitsandbytes import failed: {exc}")
print("bitsandbytes", bnb.__version__)
p = torch.nn.Parameter(torch.randn(4096, 256, device="cuda:0"))
opt = bnb.optim.AdamW([p], lr=1e-4, weight_decay=0.1, betas=(0.9, 0.95), optim_bits=8)
p.grad = torch.randn_like(p)
opt.step()          # forces 8-bit state init -> this is the sm_120 canary
torch.cuda.synchronize()
assert torch.isfinite(p).all(), "8-bit AdamW step produced non-finite params"
print("8-bit AdamW step on sm_120 OK")
PY

echo "=== [6/6] pyarrow (shared corpus) + import smoke test ==="
pip_retry "pyarrow>=15" hf_transfer
cd "$CONNITO_ROOT"
PYTHONPATH="$CONNITO_ROOT:/root/SN102" "$VENV/bin/python" - <<'PY'
from connito.shared.config import MinerConfig
from ops.shared_dataset import LocalSharedDataset
print("MinerConfig + LocalSharedDataset import OK")
print("netuid:", MinerConfig().chain.netuid)
PY

cat <<'EOF'

=== done ===
Next:
  1. cp ops/env.template ops/.env && chmod 600 ops/.env    # fill in HF tokens/repos
  2. set -a; . ops/.env; set +a
  3. python ops/fetch_corpus.py --expert-group exp_math --rows 2000000
  4. python ops/make_configs.py
  5. ./ops/launch.sh start 250        # ONE miner first, watch it for a full cycle
  6. ./ops/launch.sh start            # all three once 250 is proven
EOF
