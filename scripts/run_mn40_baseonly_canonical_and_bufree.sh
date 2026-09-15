#!/bin/bash
# MN40 FSCIL: base-only adapter, canonical vs exemplar-buffer-free (seed 2008).
#
#   bash scripts/run_mn40_baseonly_canonical_and_bufree.sh
#   bash scripts/run_mn40_baseonly_canonical_and_bufree.sh --skip-canonical
#   bash scripts/run_mn40_baseonly_canonical_and_bufree.sh --only bufferfree

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

RUNNER=(aprun python)
LOGROOT="_experiments/00_SIMPL_ABL_ADAPTER"
CONFIG="configs/selfevo_fscil_mn40_5t_flex_compare.json"
SEED=2008
BASE_FLAGS=(
  "fscil_adapter_base_only=true"
  "task0_adapter_warmup=true"
  "increment_adapter_warmup=false"
  "fscil_classifier_warmup_after_protos=false"
)

RUN_CANONICAL=1
RUN_BUFFERFREE=1
SKIP_CANONICAL=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-canonical) SKIP_CANONICAL=1; shift ;;
    --only)
      case "${2:?}" in
        canonical) RUN_BUFFERFREE=0 ;;
        bufferfree) RUN_CANONICAL=0 ;;
        *) echo "Use --only canonical|bufferfree" >&2; exit 1 ;;
      esac
      shift 2
      ;;
    *) echo "Usage: $0 [--skip-canonical] [--only canonical|bufferfree]" >&2; exit 1 ;;
  esac
done

mkdir -p "${LOGROOT}"

run_one() {
  local method="$1"
  shift
  local -a extra=("$@")
  local -a args=(--config "${CONFIG}" --logdir "${LOGROOT}" --method-name "${method}"
    --log-layout flat --seeds "${SEED}")
  for flag in "${BASE_FLAGS[@]}" "${extra[@]}"; do
    args+=(--set "${flag}")
  done
  echo "[$(date -Is)] === START ${method} ==="
  "${RUNNER[@]}" train_fscil.py "${args[@]}" 2>&1 | tee "${LOGROOT}/${method}.log"
  echo "[$(date -Is)] === DONE ${method} ==="
  grep -E "CNN top1 curve|Average Accuracy \(CNN\)|FSCIL harmonic curve" \
    "${LOGROOT}/${method}.log" | tail -5 || true
}

if [[ "${RUN_CANONICAL}" -eq 1 && "${SKIP_CANONICAL}" -eq 0 ]]; then
  run_one "abl_mn40_baseonly_canonical" "fscil_preset=canonical"
fi

if [[ "${RUN_BUFFERFREE}" -eq 1 ]]; then
  run_one "abl_mn40_baseonly_exbuffree" "model_name=selfevo_bufferfree"
fi

echo "[$(date -Is)] MN40 base-only canonical + exemplar-buffer-free finished"
