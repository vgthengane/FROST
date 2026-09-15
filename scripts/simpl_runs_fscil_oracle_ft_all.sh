#!/bin/bash
#SBATCH --job-name=fscil_oracle_ft
#SBATCH --account=EUHPC_D36_199
#SBATCH --partition=boost_usr_prod
#SBATCH --array=0-11
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=_experiments/slurm_logs/%A_%a_fscil_oracle_ft.out

# =============================================================================
# Uni3D frozen + linear SGD head — Joint (oracle) and Fine-Tune on all 6 protocols
# =============================================================================
# Joint (B): session 0 = full base; t>0 = full base + 5-shot novels seen so far
# Fine-Tune (C): session 0 = full base; t>0 = current 5-shot novels + 1-shot old replay
# Shared: no adapter, linear CE head only, base_epoch=10, increment_epoch=5, seed=2008
#
# Array map (0-5 Joint, 6-11 Fine-Tune):
#   0 / 6   mn40       configs/fscil_{joint,finetune}_mn40_5t.json
#   1 / 7   sn55       configs/fscil_{joint,finetune}_sn55_7t.json
#   2 / 8   co3d       configs/fscil_{joint,finetune}_co3d_6t.json
#   3 / 9   mn_scanobj configs/fscil_{joint,finetune}_mn40_scanobj_4t.json
#   4 / 10  sn_scanobj configs/fscil_{joint,finetune}_sn55_scanobj_4t.json
#   5 / 11  sn_co3d    configs/fscil_{joint,finetune}_sn55_co3d_11t.json
#
# Logs:
#   _experiments/00_FSCIL_ORACLE_FT/oracle_<proto>_joint_ep10_5.log
#   _experiments/00_FSCIL_ORACLE_FT/oracle_<proto>_finetune_ep10_5.log
#
# On Leonardo:
#   bash sync_eurohpc.sh --code
#   sbatch scripts/simpl_runs_fscil_oracle_ft_all.sh
# One job:
#   bash scripts/simpl_runs_fscil_oracle_ft_all.sh --local --only-protocol mn40 --only-mode joint
# Joint only / FT only:
#   bash scripts/simpl_runs_fscil_oracle_ft_all.sh --local --only-mode joint
#   bash scripts/simpl_runs_fscil_oracle_ft_all.sh --local --only-mode finetune
# Regenerated configs (if needed):
#   aprun python scripts/generate_fscil_oracle_ft_configs.py
# =============================================================================

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

RUNNER=(aprun python)
LOGROOT="_experiments/00_FSCIL_ORACLE_FT"
SEED=2008
SKIP_DONE=0
ONLY_IDX=""
ONLY_PROTOCOL=""
ONLY_MODE=""
SMOKE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --local) RUNNER=(aprun python); shift ;;
    --skip-done) SKIP_DONE=1; shift ;;
    --smoke) SMOKE=1; shift ;;
    --array)
      ONLY_IDX="${2:?}"
      shift 2
      ;;
    --only-protocol)
      ONLY_PROTOCOL="${2:?}"
      shift 2
      ;;
    --only-mode)
      ONLY_MODE="${2:?}"
      shift 2
      ;;
    *)
      echo "Unknown arg: $1" >&2
      exit 1
      ;;
  esac
done

mkdir -p "${LOGROOT}" _experiments/slurm_logs

# idx uses array id; mode|run_id|config|prefix|n_sess|label
JOBS=(
  "joint|mn40|configs/fscil_joint_mn40_5t.json|mn40|5|MN40"
  "joint|sn55|configs/fscil_joint_sn55_7t.json|sn55|7|SN55"
  "joint|co3d|configs/fscil_joint_co3d_6t.json|co3d|6|CO3D"
  "joint|mn_scanobj|configs/fscil_joint_mn40_scanobj_4t.json|mn_scanobj|4|MN→ScanObj"
  "joint|sn_scanobj|configs/fscil_joint_sn55_scanobj_4t.json|sn_scanobj|4|SN→ScanObj"
  "joint|sn_co3d|configs/fscil_joint_sn55_co3d_11t.json|sn_co3d|11|SN→CO3D"
  "finetune|mn40|configs/fscil_finetune_mn40_5t.json|mn40|5|MN40"
  "finetune|sn55|configs/fscil_finetune_sn55_7t.json|sn55|7|SN55"
  "finetune|co3d|configs/fscil_finetune_co3d_6t.json|co3d|6|CO3D"
  "finetune|mn_scanobj|configs/fscil_finetune_mn40_scanobj_4t.json|mn_scanobj|4|MN→ScanObj"
  "finetune|sn_scanobj|configs/fscil_finetune_sn55_scanobj_4t.json|sn_scanobj|4|SN→ScanObj"
  "finetune|sn_co3d|configs/fscil_finetune_sn55_co3d_11t.json|sn_co3d|11|SN→CO3D"
)

log() { echo "[$(date -Is)] $*"; }

log_is_complete() {
  local logfile="$1"
  local n_sess="$2"
  [[ -f "${logfile}" ]] || return 1
  local last
  last="$(grep "CNN top1 curve:" "${logfile}" 2>/dev/null | tail -n 1 || true)"
  [[ -n "${last}" ]] || return 1
  local n
  n="$(echo "${last}" | tr -cd ',' | wc -c)"
  n=$((n + 1))
  [[ "${n}" -ge "${n_sess}" ]]
}

run_idx() {
  local idx="$1"
  IFS='|' read -r mode run_id config prefix n_sess label <<< "${JOBS[$idx]}"

  if [[ -n "${ONLY_PROTOCOL}" && "${run_id}" != "${ONLY_PROTOCOL}" ]]; then
    return 0
  fi
  if [[ -n "${ONLY_MODE}" && "${mode}" != "${ONLY_MODE}" ]]; then
    return 0
  fi

  local method="oracle_${prefix}_${mode}_ep10_5"
  local logfile="${LOGROOT}/${method}.log"

  if [[ "${SKIP_DONE}" -eq 1 ]] && log_is_complete "${logfile}" "${n_sess}"; then
    log "SKIP ${method}"
    return 0
  fi

  local -a extra_args=()
  if [[ "${SMOKE}" -eq 1 ]]; then
    extra_args+=(
      --max-task 1
      --set base_epoch=1
      --set increment_epoch=1
    )
    method="smoke_${method}"
    logfile="${LOGROOT}/${method}.log"
  fi

  log "=== [${idx}] ${method} | ${label} | ${mode} | frozen Uni3D + linear CE | ep10/5 ==="
  "${RUNNER[@]}" train_fscil.py \
    --config "${config}" \
    --logdir "${LOGROOT}" \
    --method-name "${method}" \
    --log-layout flat \
    --seeds "${SEED}" \
    "${extra_args[@]}" \
    2>&1 | tee "${logfile}"
  log "=== ${method} done ==="
}

if [[ -n "${ONLY_IDX}" ]]; then
  run_idx "${ONLY_IDX}"
elif [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  run_idx "${SLURM_ARRAY_TASK_ID}"
else
  for idx in $(seq 0 11); do
    run_idx "${idx}"
  done
fi

log "oracle/FT all-protocol campaign finished → ${LOGROOT}"
