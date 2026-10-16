#!/usr/bin/env bash
# Shared helpers for FROST paper reproduce launchers.
# Sourced by scripts/reproduce_{indomain,crossdomain,image}.sh — do not sbatch this file.
# (Named frost_common.sh so it is not caught by the repo .gitignore `lib/` rule.)
#
# Paper method name: FROST (model_name=frost / frost_bufferfree).
# Locked recipe (see _notes/frost_final.md for numbers):
#   ridge_rp_dim=2000, mean-only insert, α=0.5 flat, static, base-only adapter,
#   conflict_replay_use_test_cm=false

# shellcheck disable=SC2034

# Python launcher.
# Prefer FROST_PYTHON if set; else ``aprun python`` when aprun exists (EuroHPC);
# else plain ``python`` (publishable default).
frost_init_runner() {
  if [[ -n "${FROST_PYTHON:-}" ]]; then
    # shellcheck disable=SC2206
    RUNNER=(${FROST_PYTHON})
  elif command -v aprun >/dev/null 2>&1; then
    RUNNER=(aprun python)
  else
    RUNNER=(python)
  fi
}

frost_log() { echo "[$(date -Is)] $*"; }

frost_log_is_complete() {
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

# Full paper seed lists (also in configs/base/{pclouds,images}.json).
# Default launchers use the first seed only; pass --fullseeds for all five.
# Same order for PC and image (paper seed set).
FROST_PC_SEEDS=(1947 1991 2001 2008 2019)
FROST_IMAGE_SEEDS=(1947 1991 2001 2008 2019)

frost_select_seeds() {
  # Usage: frost_select_seeds DEST_ARRAY_NAME FULL_LIST_NAME fullseeds_flag
  # Default: first seed only. --fullseeds → entire paper list.
  local dest_name="$1"
  local full_name="$2"
  local fullseeds="${3:-0}"
  local -n _dest="${dest_name}"
  local -n _full="${full_name}"
  if [[ "${fullseeds}" -eq 1 ]]; then
    _dest=("${_full[@]}")
  else
    _dest=("${_full[0]}")
  fi
}

FROST_PC_LOGROOT_EF="_experiments/00_FINAL_SUB_MEAN_A05_RP2000_EP10_5SEEDS"
FROST_PC_LOGROOT_EB="_experiments/00_FINAL_SUB_MEAN_A05_RP2000_BUF_EP10_5SEEDS"
FROST_IMAGE_LOGROOT_EF="_experiments/00_PRIDE_IMAGE_MEANONLY_A05_RP2000_5SEEDS"
FROST_IMAGE_LOGROOT_EB="_experiments/00_PRIDE_IMAGE_MEANONLY_A05_RP2000_BUF_5SEEDS"

# Paper PC recipe overrides (ProtoMem/EF knobs live in fscil_preset).
frost_pc_recipe_flags() {
  cat <<'EOF'
fscil_adapter_base_only=true
task0_adapter_warmup=true
increment_adapter_warmup=false
tuned_epoch=10
classifier_mode=ridge_cache
ridge_cache_alpha=0.5
fscil_increment_ridge_alpha=0.5
fscil_task1_ridge_alpha=0.5
ridge_rp_dim=2000
EOF
}

frost_pc_buffer_flags() {
  local mode="$1" # buffree | buf
  if [[ "${mode}" == "buf" ]]; then
    printf '%s\n' "model_name=frost" "fscil_preset=canonical"
  else
    printf '%s\n' "model_name=frost_bufferfree"
  fi
}

frost_append_set_flags() {
  # Usage: frost_append_set_flags dest_array_name flag1 flag2 ...
  local -n _dest="$1"
  shift
  local flag
  for flag in "$@"; do
    _dest+=(--set "${flag}")
  done
}

frost_pc_build_extra_args() {
  # Prints nothing; fills named array via nameref.
  # frost_pc_build_extra_args EXTRA_ARGS_VAR mode seed [smoke]
  local -n _extra="$1"
  local mode="$2"
  local seed="$3"
  local smoke="${4:-0}"
  _extra=()
  local flag
  while IFS= read -r flag; do
    [[ -n "${flag}" ]] || continue
    _extra+=(--set "${flag}")
  done < <(frost_pc_recipe_flags; frost_pc_buffer_flags "${mode}")
  _extra+=(
    --set "fscil_old_shot_replay_seed=${seed}"
    --set "fscil_base_ridge_refresh_seed=${seed}"
  )
  if [[ "${smoke}" -eq 1 ]]; then
    _extra+=(
      --max-task 1
      --set tuned_epoch=1
      --set task0_save_warmup_ckpts=false
      --set save_checkpoint=false
    )
  else
    # Paper path: do not retain multi-GB warmup trees.
    _extra+=(
      --set task0_save_warmup_ckpts=false
      --set save_checkpoint=false
    )
  fi
}

frost_pc_method_name() {
  local prefix="$1"
  local mode="$2" # buffree | buf
  local smoke="${3:-0}"
  local tag
  if [[ "${mode}" == "buf" ]]; then
    tag="buf"
  else
    tag="buffree"
  fi
  local name="final_sub_${prefix}_mean_st_${tag}_a05_rp2000"
  if [[ "${smoke}" -eq 1 ]]; then
    name="smoke_${name}"
  fi
  printf '%s' "${name}"
}

frost_pc_logroot_for_mode() {
  local mode="$1"
  if [[ "${mode}" == "buf" ]]; then
    printf '%s' "${FROST_PC_LOGROOT_EB}"
  else
    printf '%s' "${FROST_PC_LOGROOT_EF}"
  fi
}

frost_parse_modes() {
  # Sets MODES array from WANT_EF WANT_EB (1/0). Default both if neither flag set.
  local want_ef="$1"
  local want_eb="$2"
  MODES=()
  if [[ "${want_ef}" -eq 0 && "${want_eb}" -eq 0 ]]; then
    want_ef=1
    want_eb=1
  fi
  [[ "${want_ef}" -eq 1 ]] && MODES+=("buffree")
  [[ "${want_eb}" -eq 1 ]] && MODES+=("buf")
}

frost_run_pc_pair() {
  # frost_run_pc_pair jobs_array_name proto_idx seed mode runner_name skip_done smoke only_protocol only_seed
  local -n _jobs="$1"
  local proto_idx="$2"
  local seed="$3"
  local mode="$4"
  local -n _runner="$5"
  local skip_done="$6"
  local smoke="$7"
  local only_protocol="$8"
  local only_seed="$9"

  IFS='|' read -r run_id config prefix n_sess label <<< "${_jobs[$proto_idx]}"

  if [[ -n "${only_protocol}" && "${run_id}" != "${only_protocol}" ]]; then
    return 0
  fi
  if [[ -n "${only_seed}" && "${seed}" != "${only_seed}" ]]; then
    return 0
  fi

  local logroot method logfile
  logroot="$(frost_pc_logroot_for_mode "${mode}")"
  mkdir -p "${logroot}"
  method="$(frost_pc_method_name "${prefix}" "${mode}" "${smoke}")"
  logfile="${logroot}/${method}_seed=${seed}.log"

  if [[ "${skip_done}" -eq 1 ]] && frost_log_is_complete "${logfile}" "${n_sess}"; then
    frost_log "SKIP ${method} seed=${seed}"
    return 0
  fi

  local -a extra_args=()
  frost_pc_build_extra_args extra_args "${mode}" "${seed}" "${smoke}"

  frost_log "=== [${proto_idx}] ${method} | ${label} | seed=${seed} | ${mode} | ep10 | α=0.5 | rp=2000 ==="
  "${_runner[@]}" train_fscil.py \
    --config "${config}" \
    --logdir "${logroot}" \
    --method-name "${method}" \
    --log-layout flat \
    --seeds "${seed}" \
    "${extra_args[@]}" \
    2>&1 | tee "${logfile}"
  frost_log "=== ${method} seed=${seed} done ==="
}

frost_dispatch_pc() {
  # frost_dispatch_pc JOBS_NAME SEEDS_NAME MODES_NAME RUNNER_NAME \
  #   only_idx skip_done smoke only_protocol only_seed
  local jobs_name="$1"
  local seeds_name="$2"
  local modes_name="$3"
  local runner_name="$4"
  local only_idx="$5"
  local skip_done="$6"
  local smoke="$7"
  local only_protocol="$8"
  local only_seed="$9"

  local -n _jobs="${jobs_name}"
  local -n _seeds="${seeds_name}"
  local -n _modes="${modes_name}"

  local n_proto=${#_jobs[@]}
  local n_seed=${#_seeds[@]}
  local n_mode=${#_modes[@]}
  local block=$((n_proto * n_seed))
  local n_jobs_total=$((block * n_mode))

  run_flat() {
    local flat="$1"
    if [[ "${flat}" -lt 0 || "${flat}" -ge "${n_jobs_total}" ]]; then
      echo "Array index ${flat} out of range [0, $((n_jobs_total - 1))] (proto=${n_proto} seed=${n_seed} mode=${n_mode})" >&2
      exit 1
    fi
    local mode_idx=$((flat / block))
    local rem=$((flat % block))
    local seed_idx=$((rem / n_proto))
    local proto_idx=$((rem % n_proto))
    frost_run_pc_pair "${jobs_name}" "${proto_idx}" "${_seeds[$seed_idx]}" "${_modes[$mode_idx]}" \
      "${runner_name}" "${skip_done}" "${smoke}" "${only_protocol}" "${only_seed}"
  }

  if [[ -n "${only_idx}" ]]; then
    run_flat "${only_idx}"
  elif [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    run_flat "${SLURM_ARRAY_TASK_ID}"
  else
    local mode seed proto_idx
    for mode in "${_modes[@]}"; do
      for seed in "${_seeds[@]}"; do
        for proto_idx in $(seq 0 $((n_proto - 1))); do
          frost_run_pc_pair "${jobs_name}" "${proto_idx}" "${seed}" "${mode}" \
            "${runner_name}" "${skip_done}" "${smoke}" "${only_protocol}" "${only_seed}"
        done
      done
    done
  fi
}
