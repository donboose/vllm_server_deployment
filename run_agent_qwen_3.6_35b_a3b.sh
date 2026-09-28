#!/usr/bin/env bash
# Qwen3.6-35B-A3B (AWQ, MTP, vision) on one RTX A6000, set up for an agent harness
# (DeepSeek Harness / any OpenAI-compatible client) + live speed stats in the log.
#
# Usage:
#   ./run_vllm.sh                      # defaults below
#   MTP_K=3 THINKING=1 ./run_vllm.sh   # override per run
#   tail -f logs/vllm_q36_*.log | grep --line-buffered -E "throughput|acceptance"   # live speed
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
source "$PROJECT_ROOT/.venv/bin/activate"

# --- Tunables (override via env) ---------------------------------------------
GPU="${GPU:-1}"                    # GPU 0 = idle card; GPU 1 has the display + another user's job
MTP_K="${MTP_K:-3}"                # tool calls/JSON had ~100% acceptance at both positions -> try 3
THINKING="${THINKING:-0}"          # default when the client doesn't say: 0 = fast, 1 = reasoning on
MAX_LEN="${MAX_LEN:-131072}"       # agent harnesses resend long histories + big tool lists
MAX_SEQS="${MAX_SEQS:-4}"          # parallel requests (harness sub-agents etc.)
GPU_UTIL="${GPU_UTIL:-0.93}"
PORT="${PORT:-8012}"
KV_DTYPE="${KV_DTYPE:-auto}"       # auto = bf16; try int8_per_token_head for long-context decode speed
BATCH_TOKENS="${BATCH_TOKENS:-4096}"  # prefill chunk size: lower = shorter decode stalls, slower TTFT

# --- GPU selection ------------------------------------------------------------
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=$GPU

# --- Keep every cache inside the project (shared server) ----------------------
export HF_HOME="$PROJECT_ROOT/.cache/huggingface"
export VLLM_CACHE_ROOT="$PROJECT_ROOT/.cache/vllm"
export XDG_CACHE_HOME="$PROJECT_ROOT/.cache"
export TRITON_CACHE_DIR="$PROJECT_ROOT/.cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$PROJECT_ROOT/.cache/torchinductor"
export TMPDIR="$PROJECT_ROOT/.tmp"
export HF_HUB_OFFLINE=1
export FLASHINFER_WORKSPACE_BASE="$PROJECT_ROOT"
export VLLM_USE_FLASHINFER_SAMPLER=0
mkdir -p "$HF_HOME" "$VLLM_CACHE_ROOT" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$TMPDIR" "$PROJECT_ROOT/logs"

# --- Sanity check: fail fast if CUDA is broken (e.g. Xid / reboot required) ---
python - <<'EOF' || { echo "CUDA init failed - check nvidia-smi / dmesg before starting vLLM"; exit 1; }
import torch, sys
if not torch.cuda.is_available():
    sys.exit(1)
print(f"CUDA OK: {torch.cuda.get_device_name(0)}")
EOF

# --- API key (generated once, stored in the project, owner-only) --------------
KEY_FILE="$PROJECT_ROOT/.vllm_api_key"
[ -f "$KEY_FILE" ] || { openssl rand -hex 32 > "$KEY_FILE"; chmod 600 "$KEY_FILE"; }
export VLLM_API_KEY="$(cat "$KEY_FILE")"

# --- Optional args --------------------------------------------------------------
SPEC_ARGS=()
[ "$MTP_K" -gt 0 ] && SPEC_ARGS=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$MTP_K}")

THINK_JSON=$([ "$THINKING" = "1" ] && echo true || echo false)

LOG="$PROJECT_ROOT/logs/vllm_q36_$(date +%Y%m%d_%H%M%S).log"
echo "GPU=$GPU MTP_K=$MTP_K THINKING=$THINKING MAX_LEN=$MAX_LEN KV=$KV_DTYPE BATCH=$BATCH_TOKENS -> $LOG"

vllm serve "$PROJECT_ROOT/models/qwen3.6-35b-a3b-awq" \
  --served-model-name qwen3.6-35b-a3b \
  --host 0.0.0.0 --port "$PORT" \
  --api-key "$VLLM_API_KEY" \
  --max-model-len "$MAX_LEN" \
  --gpu-memory-utilization "$GPU_UTIL" \
  --max-num-seqs "$MAX_SEQS" \
  --max-num-batched-tokens "$BATCH_TOKENS" \
  --kv-cache-dtype "$KV_DTYPE" \
  --enable-prefix-caching \
  --mamba-cache-mode align \
  --limit-mm-per-prompt '{"image": 4, "video": 1}' \
  --mm-processor-cache-gb 0 \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --enable-mfu-metrics \
  --default-chat-template-kwargs "{\"enable_thinking\": $THINK_JSON}" \
  "${SPEC_ARGS[@]}" \
  2>&1 | tee "$LOG"
