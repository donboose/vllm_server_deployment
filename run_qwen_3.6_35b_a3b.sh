#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
source "$PROJECT_ROOT/.venv/bin/activate"

GPU=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=$GPU

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

KEY_FILE="$PROJECT_ROOT/.vllm_api_key"
[ -f "$KEY_FILE" ] || { openssl rand -hex 32 > "$KEY_FILE"; chmod 600 "$KEY_FILE"; }
export VLLM_API_KEY="$(cat "$KEY_FILE")"

MTP_K="${MTP_K:-2}"   # sweep 0-4: MTP_K=3 ./serve_qwen36.sh

SPEC_ARGS=()
[ "$MTP_K" -gt 0 ] && SPEC_ARGS=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$MTP_K}")

vllm serve "$PROJECT_ROOT/models/qwen3.6-35b-a3b-awq" \
  --served-model-name qwen3.6-35b-a3b \
  --host 0.0.0.0 --port 8012 \
  --api-key "$VLLM_API_KEY" \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.85 \
  --max-num-seqs 4 \
  --enable-prefix-caching \
  --mamba-cache-mode align \
  --mamba-ssm-cache-dtype bfloat16 \
  --limit-mm-per-prompt '{"image": 2, "video": 0}' \
  --mm-processor-cache-gb 0 \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  "${SPEC_ARGS[@]}" \
  2>&1 | tee "$PROJECT_ROOT/logs/vllm_q36_$(date +%Y%m%d_%H%M%S).log"
