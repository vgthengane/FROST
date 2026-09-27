#!/usr/bin/env bash
# Local GPU sweep: MN40 FSCIL PRidE — vary base epochs / adapter width / LR.
# Locked recipe otherwise: mean_close_st, α=0.5, buffree, static, max pool, seed 2008.
#
# Usage:
#   PHASE=1 bash scripts/local_mn40_adapt_sweep.sh
#   (skip-done: re-run safely; incomplete logs are not skipped)

set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

REPO="$(pwd)"
CONFIG="${REPO}/configs/selfevo_fscil_mn40_5t_flex_compare.json"
LOGROOT="${REPO}/_experiments/00_SIMPL_RUNs/pride_mn40_local_adapt_sweep"
SEED="${SEED:-2008}"
PHASE="${PHASE:-1}"
RUNNER=(aprun python)
N_SESS=5

mkdir -p "${LOGROOT}"
SUMMARY="${LOGROOT}/summary.tsv"
if [[ ! -f "${SUMMARY}" ]]; then
  printf "tag\tepochs\tffn\tlr\tfirst\tlast\taa\tcurve\tlogfile\n" > "${SUMMARY}"
fi

log() { echo "[$(date '+%F %T')] $*"; }

summarize_one() {
  local tag="$1" ep="$2" ffn="$3" lr="$4" logfile="$5"
  aprun python - "${logfile}" "${tag}" "${ep}" "${ffn}" "${lr}" "${SUMMARY}" <<'PY'
import re, sys
from pathlib import Path
log, tag, ep, ffn, lr, summary = sys.argv[1:7]
text = Path(log).read_text(errors="replace")
curve = None
m = re.search(r"CNN top1 curve:\s*\[([^\]]+)\]", text)
# Prefer the last (final session) curve
ms = list(re.finditer(r"CNN top1 curve:\s*\[([^\]]+)\]", text))
if ms:
    curve = [float(x) for x in ms[-1].group(1).split(",")]
aa = None
for mm in re.finditer(r"Average Accuracy \(CNN\):\s*([0-9.]+)", text):
    aa = float(mm.group(1))
if curve:
    first, last = curve[0], curve[-1]
    if aa is None:
        aa = sum(curve) / len(curve)
    curve_s = ",".join(f"{x:.2f}" for x in curve)
else:
    first = last = None
    curve_s = ""
row = f"{tag}\t{ep}\t{ffn}\t{lr}\t{first}\t{last}\t{aa}\t{curve_s}\t{log}\n"
Path(summary).open("a").write(row)
print(row.strip())
PY
}

log_complete() {
  local logfile="$1"
  [[ -f "${logfile}" ]] || return 1
  local last n
  last="$(grep "CNN top1 curve:" "${logfile}" 2>/dev/null | tail -n 1 || true)"
  [[ -n "${last}" ]] || return 1
  n="$(echo "${last}" | tr -cd ',' | wc -c)"
  n=$((n + 1))
  [[ "${n}" -ge "${N_SESS}" ]]
}

run_one() {
  local ep="$1" ffn="$2" lr="$3"
  local tag="mn40_mcs_a05_ep${ep}_ffn${ffn}_lr${lr}"
  local logfile="${LOGROOT}/${tag}_seed=${SEED}.log"

  if log_complete "${logfile}"; then
    log "SKIP done ${tag}"
    summarize_one "${tag}" "${ep}" "${ffn}" "${lr}" "${logfile}" || true
    return 0
  fi

  # Incomplete prior attempt — move aside so tee starts fresh
  if [[ -f "${logfile}" ]]; then
    mv "${logfile}" "${logfile}.partial_$(date +%H%M%S)"
  fi

  log "START ${tag}"
  "${RUNNER[@]}" train_fscil.py \
    --config "${CONFIG}" \
    --logdir "${LOGROOT}" \
    --method-name "${tag}" \
    --log-layout flat \
    --seeds "${SEED}" \
    --set model_name=selfevo_bufferfree \
    --set fscil_adapter_base_only=true \
    --set task0_adapter_warmup=true \
    --set increment_adapter_warmup=false \
    --set fscil_classifier_warmup_after_protos=false \
    --set fscil_task1_adapter_warmup=false \
    --set conflict_replay=true \
    --set conflict_replay_on_test=false \
    --set conflict_replay_use_test_cm=false \
    --set prototype_pooling=max \
    --set pooling_temperature=0.07 \
    --set fscil_shot_prototypes=false \
    --set fscil_cluster_new_classes=false \
    --set fscil_mean_best2_prototypes=true \
    --set fscil_mean_close_far_prototypes=false \
    --set fscil_mean_nearmean_prototypes=false \
    --set fscil_mean_far2_prototypes=false \
    --set insert_strategy=mean \
    --set max_prototypes_per_class=3 \
    --set fscil_increment_max_prototypes=3 \
    --set evolution_policy=static \
    --set classifier_mode=ridge_cache \
    --set ridge_cache_alpha=0.5 \
    --set fscil_increment_ridge_alpha=0.5 \
    --set fscil_task1_ridge_alpha=0.5 \
    --set fscil_ridge_alpha_ramp=false \
    --set ridge_rp_dim=1000 \
    --set exemplar_buffer_shots=0 \
    --set exemplar_buffer_ridge_refresh_shots=0 \
    --set fscil_old_shot_replay=1 \
    --set fscil_base_ridge_refresh_per_class=1 \
    --set task0_save_warmup_ckpts=true \
    --set task0_use_best_warmup=true \
    --set save_checkpoint=false \
    --set ffn_num="${ffn}" \
    --set tuned_epoch="${ep}" \
    --set task0_init_lr="${lr}" \
    --set init_lr="${lr}" \
    2>&1 | tee "${logfile}"
  summarize_one "${tag}" "${ep}" "${ffn}" "${lr}" "${logfile}" || true
  log "DONE ${tag}"
}

# Phase 1: one-factor around locked ep10/ffn64/lr0.001, plus promising combos.
# ~3 min/epoch on A100; local 3090 similar → ep10 ~35min, ep40 ~2h, ep80 ~4h.
PHASE1=(
  "10 64 0.001"      # locked baseline
  "20 64 0.001"      # more base epochs
  "40 64 0.001"
  "10 128 0.001"     # wider adapter
  "10 256 0.001"
  "10 64 0.003"      # higher LR
  "10 64 0.005"
  "20 128 0.001"     # epochs × width
  "20 64 0.003"      # epochs × LR
  "40 128 0.001"
)

PHASE2=(
  "80 64 0.001"
  "40 128 0.003"
  "40 256 0.001"
  "80 128 0.001"
)

if [[ "${PHASE}" == "2" ]]; then
  JOBS=("${PHASE2[@]}")
else
  JOBS=("${PHASE1[@]}")
fi

log "MN40 adapt sweep phase=${PHASE} n=${#JOBS[@]} logroot=${LOGROOT}"
for spec in "${JOBS[@]}"; do
  # shellcheck disable=SC2086
  run_one ${spec}
done
log "All jobs finished. Summary: ${SUMMARY}"
column -t -s $'\t' "${SUMMARY}" || cat "${SUMMARY}"
