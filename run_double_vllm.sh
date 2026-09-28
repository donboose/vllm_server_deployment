#!/usr/bin/env bash
# Dual-GPU (2x RTX A6000) launch: tensor parallel across both cards.
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
source "$PROJECT_ROOT/.venv/bin/activate"

# --- GPU selection: both physical GPUs (indices match nvidia-smi) ---
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1

# --- Keep every cache inside the project (shared server) ---
export HF_HOME="$PROJECT_ROOT/.cache/huggingface"
export VLLM_CACHE_ROOT="$PROJECT_ROOT/.cache/vllm"
export XDG_CACHE_HOME="$PROJECT_ROOT/.cache"
export TRITON_CACHE_DIR="$PROJECT_ROOT/.cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$PROJECT_ROOT/.cache/torchinductor"
export TMPDIR="$PROJECT_ROOT/.tmp"
export HF_HUB_OFFLINE=1
export FLASHINFER_WORKSPACE_BASE="$PROJECT_ROOT"

# --- Use vLLM's built-in sampler instead of FlashInfer's JIT-compiled one ---
export VLLM_USE_FLASHINFER_SAMPLER=0

# --- Multi-GPU / NCCL ---
# Use shared memory + NVLink/PCIe P2P between the two workers.
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export NCCL_CUMEM_ENABLE=0
# If startup hangs at NCCL init or custom all-reduce (common on PCIe boards
# without working P2P), uncomment the next line:
# export NCCL_P2P_DISABLE=1

mkdir -p "$HF_HOME" "$VLLM_CACHE_ROOT" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$TMPDIR" "$PROJECT_ROOT/logs"

# --- Preflight: show interconnect and current usage of both GPUs ---
echo "== GPU topology (NV# = NVLink, PHB/PIX/SYS = PCIe) =="
nvidia-smi topo -m || true
echo "== Current GPU memory use =="
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv || true

# --- API key (generated once, stored in the project, owner-only) ---
KEY_FILE="$PROJECT_ROOT/.vllm_api_key"
[ -f "$KEY_FILE" ] || { openssl rand -hex 32 > "$KEY_FILE"; chmod 600 "$KEY_FILE"; }
export VLLM_API_KEY="$(cat "$KEY_FILE")"

# Context: full native 262144 per request. With TP=2 the weights are split
# across both cards, so the KV-cache pool should land well above 300k tokens
# (check the "GPU KV cache size" line in the log).
vllm serve "$PROJECT_ROOT/qwen-3.8-27b_test" \
  --served-model-name qwen3.8-27b-test \
  --host 0.0.0.0 \
  --port 8011 \
  --api-key "$VLLM_API_KEY" \
  --dtype bfloat16 \
  --tensor-parallel-size 2 \
  --max-model-len 262144 \
  --gpu-memory-utilization 0.93 \
  --max-num-seqs 8 \
  --max-num-batched-tokens 16384 \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --chat-template "$PROJECT_ROOT/qwen38_chat_template_patched.jinja" \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --speculative-config '{"method": "mtp", "num_speculative_tokens": 3}' \
  --limit-mm-per-prompt '{"image": 0, "video": 0}' \
  --enable-mfu-metrics \
  2>&1 | tee "$PROJECT_ROOT/logs/vllm_tp2_$(date +%Y%m%d_%H%M%S).log"

# Optional extras to try once the above runs cleanly:
#   --kv-cache-dtype fp8        roughly doubles KV capacity; small quality cost,
#                               and on Ampere it can be slightly slower per token.
#   --async-scheduling          overlaps CPU scheduling with GPU work (newer vLLM;
#                               may not combine with speculative decoding on your version).
