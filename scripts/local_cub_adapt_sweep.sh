#!/usr/bin/env bash
# Local GPU sweep: CUB mean1 + α=0.35 + rp=2000, vary epochs / ffn_num / lr.
# Goal: push AA above current ~86.01 (ep80 / ffn64 / lr0.01).
#
# Usage:
#   bash scripts/local_cub_adapt_sweep.sh
#   PHASE=1 bash scripts/local_cub_adapt_sweep.sh   # screen grid only
#   PHASE=2 bash scripts/local_cub_adapt_sweep.sh   # promote winners (edit WINNERS below)

set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

REPO="$(pwd)"
CONFIG="${REPO}/configs/pride_image_cub200_buffree.json"
DATA_PATH="${REPO}/_datasets/images"
LOGROOT="${REPO}/_experiments/00_SIMPL_RUNs/pride_image_local_adapt_sweep"
SEED="${SEED:-2008}"
PHASE="${PHASE:-1}"
RUNNER=(aprun python)

mkdir -p "${LOGROOT}"
SUMMARY="${LOGROOT}/summary.tsv"
if [[ ! -f "${SUMMARY}" ]]; then
  printf "tag\tepochs\tffn\tlr\tfirst\tlast\taa\tforget\tlogfile\n" > "${SUMMARY}"
fi

log() { echo "[$(date '+%F %T')] $*"; }

# Extract First / Last / AA / Forget from a flat log.
summarize_one() {
  local tag="$1" ep="$2" ffn="$3" lr="$4" logfile="$5"
  aprun python - "${logfile}" "${tag}" "${ep}" "${ffn}" "${lr}" "${SUMMARY}" <<'PY'
import re, sys
from pathlib import Path
log, tag, ep, ffn, lr, summary = sys.argv[1:7]
text = Path(log).read_text(errors="replace")
# Prefer explicit curves if present
def last_float(pat):
    ms = list(re.finditer(pat, text))
    return float(ms[-1].group(1)) if ms else None

curve = None
m = re.search(r"CNN top1 curve:\s*\[([^\]]+)\]", text)
if m:
    curve = [float(x) for x in m.group(1).split(",")]
aa = last_float(r"Average Accuracy \(CNN\):\s*([0-9.]+)")
forget = last_float(r"[Ff]orget(?:ting)?[^0-9]*([0-9.]+)")
if curve:
    first, last = curve[0], curve[-1]
    if aa is None:
        aa = sum(curve) / len(curve)
else:
    first = last_float(r"Task 0.*?Top-1 accuracy\s*=\s*([0-9.]+)")
    last = last_float(r"Top-1 accuracy\s*=\s*([0-9.]+)")
row = f"{tag}\t{ep}\t{ffn}\t{lr}\t{first}\t{last}\t{aa}\t{forget}\t{log}\n"
Path(summary).open("a").write(row)
print(row.strip())
PY
}

run_one() {
  local ep="$1" ffn="$2" lr="$3"
  local tag="cub_mean1_a035_rp2000_ep${ep}_ffn${ffn}_lr${lr}"
  local logfile="${LOGROOT}/${tag}_seed=${SEED}.log"
  if [[ -f "${logfile}" ]] && grep -q "Average Accuracy (CNN)" "${logfile}" 2>/dev/null; then
    log "SKIP done ${tag}"
    summarize_one "${tag}" "${ep}" "${ffn}" "${lr}" "${logfile}" || true
    return 0
  fi
  log "START ${tag}"
  "${RUNNER[@]}" train_fscil_image.py \
    --config "${CONFIG}" \
    --data-path "${DATA_PATH}" \
    --logdir "${LOGROOT}" \
    --method-name "${tag}" \
    --log-layout flat \
    --seeds "${SEED}" \
    --set prefix="${tag}" \
    --set fscil_mean_best2_prototypes=false \
    --set fscil_shot_prototypes=false \
    --set insert_strategy=mean \
    --set max_prototypes_per_class=1 \
    --set fscil_increment_max_prototypes=1 \
    --set ridge_cache_alpha=0.35 \
    --set fscil_increment_ridge_alpha=0.35 \
    --set fscil_task1_ridge_alpha=0.35 \
    --set ridge_rp_dim=2000 \
    --set cache_budget=600 \
    --set evolution_policy=static \
    --set classifier_mode=ridge_cache \
    --set fscil_adapter_base_only=true \
    --set task0_adapter_warmup=true \
    --set increment_adapter_warmup=false \
    --set fscil_classifier_warmup_after_protos=false \
    --set fscil_task1_adapter_warmup=false \
    --set conflict_replay=true \
    --set conflict_replay_on_test=false \
    --set conflict_replay_use_test_cm=false \
    --set exemplar_buffer_shots=0 \
    --set save_checkpoint=false \
    --set ffn_num="${ffn}" \
    --set tuned_epoch="${ep}" \
    --set task0_init_lr="${lr}" \
    --set init_lr="${lr}" \
    2>&1 | tee "${logfile}"
  summarize_one "${tag}" "${ep}" "${ffn}" "${lr}" "${logfile}" || true
  log "DONE ${tag}"
}

# Phase 1: one-factor-at-a-time around current best, plus a few combos.
PHASE1=(
  "80 64 0.01"     # baseline (~86.01 AA)
  "120 64 0.01"    # more base epochs
  "160 64 0.01"
  "80 128 0.01"    # wider adapter
  "80 256 0.01"
  "80 64 0.025"    # higher LR
  "80 64 0.05"
  "120 128 0.01"   # epochs × width
  "120 64 0.025"   # epochs × LR
  "80 128 0.025"   # width × LR
)

# Phase 2: edit after phase-1 results (top combos).
PHASE2=(
  "160 128 0.01"
  "160 128 0.025"
  "120 256 0.01"
  "200 128 0.01"
)

if [[ "${PHASE}" == "2" ]]; then
  JOBS=("${PHASE2[@]}")
else
  JOBS=("${PHASE1[@]}")
fi

log "CUB adapt sweep phase=${PHASE} n=${#JOBS[@]} logroot=${LOGROOT}"
for spec in "${JOBS[@]}"; do
  # shellcheck disable=SC2086
  run_one ${spec}
done
log "All jobs finished. Summary: ${SUMMARY}"
column -t -s $'\t' "${SUMMARY}" || cat "${SUMMARY}"
