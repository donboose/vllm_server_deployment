#!/usr/bin/env bash

set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
source "$PROJECT_ROOT/.venv/bin/activate"
 
# --- GPU selection: physical GPU1 only (indices match nvidia-smi) ---
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1
 
# --- Keep every cache inside the project (shared server) ---
export HF_HOME="$PROJECT_ROOT/.cache/huggingface"
export VLLM_CACHE_ROOT="$PROJECT_ROOT/.cache/vllm"
export XDG_CACHE_HOME="$PROJECT_ROOT/.cache"
export TRITON_CACHE_DIR="$PROJECT_ROOT/.cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$PROJECT_ROOT/.cache/torchinductor"
export TMPDIR="$PROJECT_ROOT/.tmp"
export HF_HUB_OFFLINE=1
export FLASHINFER_WORKSPACE_BASE="$PROJECT_ROOT"   # FlashInfer JIT cache -> $PROJECT_ROOT/.cache/flashinfer
 
# --- Use vLLM's built-in sampler instead of FlashInfer's JIT-compiled one ---
# (the FlashInfer sampler needs a full CUDA toolkit with headers to compile)
export VLLM_USE_FLASHINFER_SAMPLER=0
mkdir -p "$HF_HOME" "$VLLM_CACHE_ROOT" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$TMPDIR" "$PROJECT_ROOT/logs"
 
# --- API key (generated once, stored in the project, owner-only) ---
KEY_FILE="$PROJECT_ROOT/.vllm_api_key"
[ -f "$KEY_FILE" ] || { openssl rand -hex 32 > "$KEY_FILE"; chmod 600 "$KEY_FILE"; }
export VLLM_API_KEY="$(cat "$KEY_FILE")"
 
# Context: full native 262144. Startup log showed ~330k tokens of KV cache.
# Lower to 196608 or 131072 if you need several long requests at once.
vllm serve "$PROJECT_ROOT/qwen-3.8-27b_test" \
  --served-model-name qwen3.8-27b-test \
  --host 0.0.0.0 \
  --port 8011 \
  --api-key "$VLLM_API_KEY" \
  --dtype bfloat16 \
  --max-model-len 262144 \
  --gpu-memory-utilization 0.95 \
  --max-num-seqs 4 \
  --max-num-batched-tokens 8192 \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --chat-template "$PROJECT_ROOT/qwen38_chat_template_patched.jinja" \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --speculative-config '{"method": "mtp", "num_speculative_tokens": 3}' \
  --limit-mm-per-prompt '{"image": 2, "video": 1}' \
  --enable-mfu-metrics \
  2>&1 | tee "$PROJECT_ROOT/logs/vllm_$(date +%Y%m%d_%H%M%S).log"

