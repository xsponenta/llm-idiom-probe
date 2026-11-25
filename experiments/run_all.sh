#!/usr/bin/env bash
# Overnight run: E1 + E2-E4 for every model, then the summary. Safe to restart: finished work is skipped.
#
#   bash experiments/run_all.sh                          # default model list below
#   MODELS="qwen3-1.7b qwen3-4b" bash experiments/run_all.sh
#   nohup bash experiments/run_all.sh > run.log 2>&1 &   # detach (survives ssh disconnect)
#
# Two GPUs = two queues in parallel, e.g. big models with the HF cache in RAM, small ones on disk:
#   GPU=0 HF_HUB_CACHE=/dev/shm/hf MODELS="qwen3-14b gemma3-12b ..." nohup bash experiments/run_all.sh > run_gpu0.log 2>&1 &
#   GPU=1 MODELS="qwen3-0.6b qwen3-1.7b ..."                          nohup bash experiments/run_all.sh > run_gpu1.log 2>&1 &
#
# If experiments/hf_manifest.json exists, every model is downloaded at its pinned revision and each file is
# checked against the manifest's checksums before the model runs (offline). A model that fails is skipped.
# CLEANUP=1 (default) deletes each model from the HF cache afterwards, so the disk / RAM does not fill up.
set -u
cd "$(dirname "$0")/.."
if [ -n "${GPU:-}" ]; then export CUDA_VISIBLE_DEVICES=$GPU; fi
if [ -n "${HF_HUB_CACHE:-}" ]; then export HF_HUB_CACHE; mkdir -p "$HF_HUB_CACHE"; fi

MODELS=${MODELS:-"qwen3-0.6b qwen3-1.7b qwen3-4b qwen3-8b qwen3-14b gemma3-4b mamaylm-4b gemma3-12b mamaylm-12b lapa-12b llama3.1-8b mistral-nemo"}
E1_BATCH=${E1_BATCH:-16}
ITEMS_PER_BATCH=${ITEMS_PER_BATCH:-4}
CLEANUP=${CLEANUP:-1}
PY=${PY:-python}
mkdir -p data/results/gpu/logs

cleanup_model() {
  [ "$CLEANUP" = "1" ] || return 0
  # A private cache dir per queue (set by run_night.sh) is emptied completely: newer huggingface_hub keeps the
  # weights in a shared hub/blobs store, so deleting only models--<repo> would leave them on disk.
  if [ "${PRIVATE_CACHE:-0}" = "1" ] && [ -n "${HF_HUB_CACHE:-}" ]; then
    rm -rf "${HF_HUB_CACHE:?}"/* && echo "emptied $HF_HUB_CACHE"
    return 0
  fi
  $PY - "$1" <<'EOF'
import shutil, sys
from pathlib import Path
sys.path.insert(0, "experiments")
from common import resolve
from huggingface_hub.constants import HF_HUB_CACHE
d = Path(HF_HUB_CACHE) / ("models--" + resolve(sys.argv[1])[1].replace("/", "--"))
shutil.rmtree(d, ignore_errors=True)
print(f"removed {d}")
EOF
}

for m in $MODELS; do
  log=data/results/gpu/logs/$m.log
  echo "=== $(date '+%F %T') $m" | tee -a "$log"
  if [ -f experiments/hf_manifest.json ]; then
    # SSL_CERT_FILE lets the download pass a host-side HTTPS proxy; the checksums catch any tampering
    if ! SSL_CERT_FILE=${SSL_CERT_FILE:-/etc/ssl/certs/ca-certificates.crt} \
         $PY experiments/hf_manifest.py fetch "$m" >> "$log" 2>&1; then
      echo "!!! $m skipped: download or checksum verification failed (see $log)" | tee -a "$log"
      cleanup_model "$m"
      continue
    fi
    tail -1 "$log"
    export HF_HUB_OFFLINE=1          # from here on nothing unverified can be fetched
  fi
  # one model failing (OOM, load error) must not stop the night
  $PY experiments/behavioral.py --models "$m" --batch "$E1_BATCH" >> "$log" 2>&1 \
    || echo "!!! E1 failed for $m (see $log)" | tee -a "$log"
  $PY experiments/lm_probe.py --models "$m" --items-per-batch "$ITEMS_PER_BATCH" >> "$log" 2>&1 \
    || echo "!!! E2-E4 failed for $m (see $log)" | tee -a "$log"
  unset HF_HUB_OFFLINE
  # SAVE_ACTS=1 makes lm_probe.py keep all activations; UPLOAD_ACTS=1 moves them to the private HF dataset
  # (verified by sha256, local copy deleted only then) so the disk does not fill up
  if [ "${SAVE_ACTS:-0}" = "1" ] && [ "${UPLOAD_ACTS:-0}" = "1" ]; then
    SSL_CERT_FILE=${SSL_CERT_FILE:-/etc/ssl/certs/ca-certificates.crt} \
      $PY experiments/upload_acts.py "$m" >> "$log" 2>&1 && tail -1 "$log" \
      || echo "!!! upload of $m activations failed, kept on disk (see $log)" | tee -a "$log"
  fi
  cleanup_model "$m"
  echo "=== $(date '+%F %T') $m done" | tee -a "$log"
done

if [ "${SKIP_ANALYZE:-0}" != "1" ]; then
  $PY experiments/analyze.py > data/results/gpu/summary.txt 2>&1 || echo "!!! analyze failed"
fi
echo "ALL DONE $(date '+%F %T')  ($MODELS)"
