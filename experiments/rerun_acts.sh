#!/usr/bin/env bash
# Third pass, after rerun_v2.sh: E2-E4 again for every model with all activations saved (SAVE_ACTS=1) and
# uploaded to the private HF dataset (UPLOAD_ACTS=1, token in /root/.hf_token). The E2-E3 numbers come out the
# same; the Gemma-3 family (gemma3, MamayLM, Lapa) also gets its E4 features in float32 (float16 overflowed).
# E1 is already complete -> skipped. Ends with analyze.py (inside run_night.sh).
#
#   setsid nohup bash experiments/rerun_acts.sh > rerun_acts.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/.."
export PATH=/venv/main/bin:$PATH
PY=${PY:-python}
while pgrep -f "^bash experiments/rerun_v2.sh" > /dev/null; do sleep 60; done
echo "=== $(date '+%F %T') v2 rerun finished"
# without a working write token the activations would pile up on disk -> wait for it
until [ -s /root/.hf_token ] && $PY - <<'PYEOF'
from pathlib import Path
from huggingface_hub import HfApi
print("HF token ok:", HfApi(token=Path("/root/.hf_token").read_text().strip()).whoami()["name"])
PYEOF
do echo "$(date '+%T') waiting for a valid /root/.hf_token"; sleep 120; done
echo "=== $(date '+%F %T') saving activations"
export SAVE_ACTS=1 UPLOAD_ACTS=1
QUEUE0="gemma3-12b lapa-12b qwen3-14b qwen3-4b qwen3-0.6b gemma3-4b" \
QUEUE1="mamaylm-12b mistral-nemo llama3.1-8b qwen3-8b qwen3-1.7b mamaylm-4b" \
BIG="qwen3-32b" bash experiments/run_night.sh
echo "=== $(date '+%F %T') activations pass finished"
