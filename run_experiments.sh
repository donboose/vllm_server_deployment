#!/usr/bin/env bash
# Runs experiments E0-E6 unattended ON THE SERVER: for each config start vLLM via run_vllm.sh,
# wait until /health answers, run bench_sweep.py against localhost, stop vLLM, next config.
# Results: results/<timestamp>/  (one .json/.txt per bench run, .server.log per config, summary.txt)
#
#   tmux new -s exp                    # takes hours; keep it alive if your SSH drops
#   ./run_experiments.sh               # everything
#   ONLY='^e3_' ./run_experiments.sh   # just the MTP_K sweep (regex on tags)
#   ONLY='^e0_|^e2_' ./run_experiments.sh
#
# Needs run_vllm.sh, bench_sweep.py, compare_sweeps.py in the project root.
set -uo pipefail
PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_ROOT"
source .venv/bin/activate

PORT="${PORT:-8012}"
KEY="$(cat .vllm_api_key)"
HARNESS="${HARNESS:---temperature 0.7 --top-p 0.8 --top-k 20}"   # <-- set to your harness's sampling
ONLY="${ONLY:-}"
OUT="$PROJECT_ROOT/results/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"

# Bench profiles (bench_sweep.py arguments)
FULL=""
SHORT="--contexts 1000,4000 --skip longgen,interference,multiturn"
CTX="--contexts 1000,16000,32000,64000,100000 --skip content,concurrency,longgen,interference,multiturn"
CONTENT="--skip context,concurrency,longgen,interference,multiturn"

# tag | server env (KEY=VAL, space separated, no spaces in values) | bench runs "sub=args" joined by ';'
# Every experiment changes ONE thing vs e0_base.
EXPERIMENTS=(
  "e0_base||greedy=$FULL;harness=$HARNESS --contexts 1000,32000 --skip longgen,interference"

  "e1_async_on|ASYNC=on|greedy=$SHORT"
  "e1_async_off|ASYNC=off|greedy=$SHORT"

  "e2_flashinfer|ATTN_BACKEND=FLASHINFER|greedy=$CTX"
  "e2_flash_attn|ATTN_BACKEND=FLASH_ATTN|greedy=$CTX"
  "e2_triton_attn|ATTN_BACKEND=TRITON_ATTN|greedy=$CTX"

  "e3_k0|MTP_K=0|greedy=$CONTENT;harness=$HARNESS $CONTENT"
  "e3_k1|MTP_K=1|greedy=$CONTENT;harness=$HARNESS $CONTENT"
  "e3_k2|MTP_K=2|greedy=$CONTENT;harness=$HARNESS $CONTENT"
  "e3_k3|MTP_K=3|greedy=$CONTENT;harness=$HARNESS $CONTENT"
  "e3_k4|MTP_K=4|greedy=$CONTENT;harness=$HARNESS $CONTENT"
  "e3_k5|MTP_K=5|greedy=$CONTENT;harness=$HARNESS $CONTENT"

  "e4_seqs1|MAX_SEQS=1|greedy=$SHORT --concurrency 1"

  "e5_full_and_piecewise|CG_MODE=FULL_AND_PIECEWISE|greedy=$SHORT"
  "e5_full_decode_only|CG_MODE=FULL_DECODE_ONLY|greedy=$SHORT"

  "e6_no_mfu|MFU=0|greedy=$SHORT"

  "e9_base_repeat||greedy=$SHORT"      # same as e0: drift check across the whole session
)

health() { curl -sf -o /dev/null -H "Authorization: Bearer $KEY" "http://127.0.0.1:$PORT/health"; }

wait_healthy() {   # up to 20 min: first start of a new config may recompile + capture graphs
  local pid=$1
  for _ in $(seq 1 240); do
    kill -0 "$pid" 2>/dev/null || return 1      # server died during startup
    health && return 0
    sleep 5
  done
  return 1
}

stop_server() {    # whole process group: run_vllm.sh, tee, vllm API server, EngineCore
  local pid=$1
  kill -TERM -- "-$pid" 2>/dev/null
  for _ in $(seq 1 90); do kill -0 -- "-$pid" 2>/dev/null || break; sleep 1; done
  kill -KILL -- "-$pid" 2>/dev/null
  wait "$pid" 2>/dev/null
  for _ in $(seq 1 30); do health || break; sleep 1; done
  sleep 10   # let the driver release GPU memory before the next start
}

if health; then
  echo "Something is already serving on port $PORT - stop it first."; exit 1
fi

echo "Results -> $OUT"
for exp in "${EXPERIMENTS[@]}"; do
  IFS='|' read -r tag envs benches <<<"$exp"
  [[ -n "$ONLY" && ! "$tag" =~ $ONLY ]] && continue
  echo; echo "=== $tag   [${envs:-defaults}]   $(date +%H:%M:%S)"

  # shellcheck disable=SC2086
  setsid env $envs RUN_TAG="$tag" ./run_test_vllm.sh >"$OUT/$tag.server.log" 2>&1 &
  pid=$!
  if ! wait_healthy "$pid"; then
    echo "!!! $tag did not come up - see $OUT/$tag.server.log (skipping)"
    stop_server "$pid"; continue
  fi

  IFS=';' read -ra runs <<<"$benches"
  for run in "${runs[@]}"; do
    sub="${run%%=*}"; args="${run#*=}"
    echo "--- ${tag}_$sub: $args"
    # shellcheck disable=SC2086
    python bench_sweep.py --server 127.0.0.1 --port "$PORT" --api-key "$KEY" \
     --tag "${tag}_$sub" --out "$OUT/${tag}_$sub.json" $args 2>&1 | tee "$OUT/${tag}_$sub.txt"
  done
  stop_server "$pid"
done

echo; echo "=== summary"
python compare_sweeps.py "$OUT"/*.json | tee "$OUT/summary.txt"
echo "CUDA graph capture per config (E5):"
grep -H -iE "captur(e|ing).*(cuda ?graph|cudagraph)|cudagraph_mode" "$OUT"/*.server.log | head -n 60 
