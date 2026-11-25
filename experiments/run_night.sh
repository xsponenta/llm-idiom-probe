#!/usr/bin/env bash
# Whole night on a 2-GPU machine (2 x 48 GB):
#   1) two queues in parallel, one per GPU, for every model up to 14B;
#   2) qwen3-32b (65 GB in bf16) spread over both GPUs;
#   3) analyze.py once -> data/results/gpu/summary/ (incl. summary_metrics.csv) and summary.txt.
#
#   nohup bash experiments/run_night.sh > night.log 2>&1 &
# Restart-safe: finished models / questions / shards are skipped.
set -u
cd "$(dirname "$0")/.."
PY=${PY:-python}
QUEUE0=${QUEUE0:-"qwen3-0.6b qwen3-14b mamaylm-12b llama3.1-8b qwen3-4b mamaylm-4b"}
QUEUE1=${QUEUE1:-"qwen3-1.7b gemma3-12b lapa-12b mistral-nemo qwen3-8b gemma3-4b"}
BIG=${BIG:-"qwen3-32b"}

CACHE_ROOT=${CACHE_ROOT:-/workspace/hf_queues}     # one private HF cache per queue, emptied after each model
export PRIVATE_CACHE=1

echo "=== $(date '+%F %T') start"
SKIP_ANALYZE=1 GPU=0 HF_HUB_CACHE=$CACHE_ROOT/q0 MODELS="$QUEUE0" bash experiments/run_all.sh >> run_gpu0.log 2>&1 &
p0=$!
SKIP_ANALYZE=1 GPU=1 HF_HUB_CACHE=$CACHE_ROOT/q1 MODELS="$QUEUE1" bash experiments/run_all.sh >> run_gpu1.log 2>&1 &
p1=$!
wait $p0 $p1
echo "=== $(date '+%F %T') both queues done"

SKIP_ANALYZE=1 GPU=0,1 HF_HUB_CACHE=$CACHE_ROOT/big MODELS="$BIG" E1_BATCH=8 ITEMS_PER_BATCH=2 \
  bash experiments/run_all.sh >> run_big.log 2>&1
echo "=== $(date '+%F %T') $BIG done"

$PY experiments/analyze.py > data/results/gpu/summary.txt 2>&1 || echo "!!! analyze failed"
grep -h "!!!" run_gpu0.log run_gpu1.log run_big.log || echo "no failures"
echo "=== $(date '+%F %T') ALL DONE -> data/results/gpu/summary/ and data/results/gpu/summary.txt"
