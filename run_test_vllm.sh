#!/usr/bin/env bash
# Qwen3.6-35B-A3B (AWQ, MTP, vision) on one RTX A6000, set up for an agent harness
# (DeepSeek Harness / any OpenAI-compatible client) + live speed stats in the log.
#
# Usage:
#   ./run_vllm.sh                                  # defaults below
#   MTP_K=4 ASYNC=on ./run_vllm.sh                 # override per run
#   ATTN_BACKEND=FLASHINFER ./run_vllm.sh          # attention backend experiment
#   PROFILE=1 ./run_vllm.sh                        # run under nsys (capture via profile_one.py)
#   tail -f logs/vllm_q36_*.log | grep --line-buffered -E "throughput|acceptance"   # live speed
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
source "$PROJECT_ROOT/.venv/bin/activate"

# --- Tunables (override via env) ---------------------------------------------
GPU="${GPU:-1}"
MTP_K="${MTP_K:-3}"                # 0 = speculative decoding off
THINKING="${THINKING:-0}"          # default when the client doesn't say: 0 = fast, 1 = reasoning on
MAX_LEN="${MAX_LEN:-131072}"       # agent harnesses resend long histories + big tool lists
MAX_SEQS="${MAX_SEQS:-4}"          # parallel requests (harness sub-agents etc.)
GPU_UTIL="${GPU_UTIL:-0.93}"
PORT="${PORT:-8012}"
KV_DTYPE="${KV_DTYPE:-auto}"       # auto = bf16
BATCH_TOKENS="${BATCH_TOKENS:-4096}"  # prefill chunk size: lower = shorter decode stalls, slower TTFT
# Experiment knobs (empty/default = vLLM's own choice)
ASYNC="${ASYNC:-default}"          # on | off | default
ATTN_BACKEND="${ATTN_BACKEND:-}"   # FLASHINFER | FLASH_ATTN | TRITON_ATTN
CG_MODE="${CG_MODE:-}"             # FULL_AND_PIECEWISE | FULL_DECODE_ONLY | PIECEWISE
MFU="${MFU:-1}"                    # 1 = --enable-mfu-metrics
PROFILE="${PROFILE:-0}"            # 1 = run under nsys, capture on /start_profile../stop_profile
EXTRA_ARGS="${EXTRA_ARGS:-}"       # anything else, passed through word-split
RUN_TAG="${RUN_TAG:-}"             # goes into the log file name

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

OPT_ARGS=()
case "$ASYNC" in
  on)  OPT_ARGS+=(--async-scheduling) ;;
  off) OPT_ARGS+=(--no-async-scheduling) ;;
esac
if [ -n "$ATTN_BACKEND" ]; then
  # Newer vLLM has --attention-backend; older versions only read the env var.
  HELP_TEXT="$(vllm serve --help=all 2>/dev/null || vllm serve --help 2>/dev/null || true)"
  if grep -q -- '--attention-backend' <<<"$HELP_TEXT"; then
    OPT_ARGS+=(--attention-backend "$ATTN_BACKEND")
  else
    export VLLM_ATTENTION_BACKEND="$ATTN_BACKEND"
  fi
fi
[ -n "$CG_MODE" ] && OPT_ARGS+=(--compilation-config "{\"cudagraph_mode\":\"$CG_MODE\"}")
[ "$MFU" = "1" ] && OPT_ARGS+=(--enable-mfu-metrics)
# shellcheck disable=SC2206
[ -n "$EXTRA_ARGS" ] && OPT_ARGS+=($EXTRA_ARGS)

PREFIX=()
if [ "$PROFILE" = "1" ]; then
  export VLLM_TORCH_CUDA_PROFILE=1   # enables /start_profile + /stop_profile -> cudaProfilerStart/Stop
  PREFIX=(nsys profile -o "$PROJECT_ROOT/logs/nsys_q36" --trace=cuda,nvtx,osrt
          --cuda-graph-trace=node --trace-fork-before-exec=true
          --capture-range=cudaProfilerApi --capture-range-end=repeat --force-overwrite=true)
fi

LOG="$PROJECT_ROOT/logs/vllm_q36_${RUN_TAG:+${RUN_TAG}_}$(date +%Y%m%d_%H%M%S).log"
echo "GPU=$GPU MTP_K=$MTP_K THINKING=$THINKING MAX_LEN=$MAX_LEN MAX_SEQS=$MAX_SEQS KV=$KV_DTYPE BATCH=$BATCH_TOKENS" \
     "ASYNC=$ASYNC ATTN=${ATTN_BACKEND:-default} CG=${CG_MODE:-default} MFU=$MFU PROFILE=$PROFILE" \
     "EXTRA='$EXTRA_ARGS' -> $LOG"

"${PREFIX[@]}" vllm serve "$PROJECT_ROOT/models/qwen3.6-35b-a3b-awq" \
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
  --limit-mm-per-prompt '{"image": 4, "video": 0}' \
  --mm-processor-cache-gb 0 \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --default-chat-template-kwargs "{\"enable_thinking\": $THINK_JSON}" \
  "${OPT_ARGS[@]}" \
  "${SPEC_ARGS[@]}" \
  2>&1 | tee "$LOG"
