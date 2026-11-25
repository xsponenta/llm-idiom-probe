#!/usr/bin/env bash
# Second pass after two fixes found in the first run:
#   - E1 scoring v2 ("Відповідь:" prefix, plain and bold answers summed) for every model;
#   - Mistral Nemo: the hub tokenizer had a wrong regex -> all its results (E1-E4) are recomputed.
# Old files are kept next to the new ones (e1_v1.jsonl, lm_probe_v1/). Waits for a running run_night.sh first.
#
#   nohup bash experiments/rerun_v2.sh > rerun.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
PY=${PY:-python}

while pgrep -f "^bash experiments/run_night.sh" > /dev/null; do sleep 60; done
echo "=== $(date '+%F %T') first run finished, preparing v2"

for d in data/results/gpu/*/; do
  f=$d/e1.jsonl
  if [ -f "$f" ] && ! head -1 "$f" | grep -q '"method": "v2-prefix-bold"'; then
    mv "$f" "$d/e1_v1.jsonl" && echo "kept old E1 as $d/e1_v1.jsonl"
  fi
done
if [ -d data/results/gpu/mistral-nemo/lm_probe ]; then
  rm -rf data/results/gpu/mistral-nemo/lm_probe_v1
  mv data/results/gpu/mistral-nemo/lm_probe data/results/gpu/mistral-nemo/lm_probe_v1
  echo "kept old Nemo E2-E4 as lm_probe_v1/"
fi

QUEUE0=${QUEUE0:-"qwen3-14b lapa-12b llama3.1-8b qwen3-4b qwen3-0.6b mamaylm-4b"} \
QUEUE1=${QUEUE1:-"gemma3-12b mamaylm-12b mistral-nemo qwen3-8b gemma3-4b qwen3-1.7b"} \
BIG=${BIG:-"qwen3-32b"} bash experiments/run_night.sh       # qwen3-32b: E1 is already v2 -> skipped quickly
echo "=== $(date '+%F %T') v2 rerun finished"
