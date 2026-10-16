#!/usr/bin/env bash
# Modular cleanup for _experiments/.
#
# Never touches _datasets/. Never deletes backbone weights under pretrained_weights/.
#
# Usage:
#   bash scripts/cleanup_experiments.sh --dry-run
#   bash scripts/cleanup_experiments.sh --apply                 # prune + strip-ckpts
#   bash scripts/cleanup_experiments.sh --prune-only --apply
#   bash scripts/cleanup_experiments.sh --strip-checkpoints --apply
#   bash scripts/cleanup_experiments.sh --logs-only --apply     # campaign dirs: keep *.log (+ *.md/*.csv summaries)
#
# Layout after cleanup (conceptual):
#   _experiments/
#     00_FINAL_SUB_MEAN_A05_RP2000_EP10_5SEEDS/*.log     # 3D EF paper
#     00_FINAL_SUB_MEAN_A05_RP2000_BUF_EP10_5SEEDS/*.log # 3D EB paper
#     00_PRIDE_IMAGE_MEANONLY_A05_RP2000_5SEEDS/*.log    # image EF paper
#     _paper_aux/                                        # optional ablation log twins
#     pretrained_weights/                                # Uni3D + ViT (needed to rerun)
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXP="${REPO}/_experiments"
APPLY=0
DO_PRUNE=0
DO_STRIP=0
DO_LOGS_ONLY=0
# Default when no mode flag: prune + strip
EXPLICIT_MODE=0

KEEP=(
  "00_FINAL_SUB_MEAN_A05_RP2000_EP10_5SEEDS"
  "00_FINAL_SUB_MEAN_A05_RP2000_BUF_EP10_5SEEDS"
  "00_PRIDE_IMAGE_MEANONLY_A05_RP2000_5SEEDS"
  "pretrained_weights"
  "_paper_aux"
)

PAPER_CAMPAIGNS=(
  "00_FINAL_SUB_MEAN_A05_RP2000_EP10_5SEEDS"
  "00_FINAL_SUB_MEAN_A05_RP2000_BUF_EP10_5SEEDS"
  "00_PRIDE_IMAGE_MEANONLY_A05_RP2000_5SEEDS"
)

AUX_RELOCATE=(
  "00_FINAL_SUB_MEAN_A05_RP1000_EP10_5SEEDS"
  "00_FINAL_SUB_MEAN_A05_RP1000_BUF_EP10_5SEEDS"
  "00_PRIDE_LOCKED_ABLATION_MN40_5SEEDS"
)

usage() {
  sed -n '1,22p' "$0"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run|-n) APPLY=0; shift ;;
    --apply) APPLY=1; shift ;;
    --prune-only) DO_PRUNE=1; EXPLICIT_MODE=1; shift ;;
    --strip-checkpoints) DO_STRIP=1; EXPLICIT_MODE=1; shift ;;
    --logs-only) DO_LOGS_ONLY=1; EXPLICIT_MODE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *)
      echo "Unknown arg: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ "${EXPLICIT_MODE}" -eq 0 ]]; then
  DO_PRUNE=1
  DO_STRIP=1
  DO_LOGS_ONLY=1
fi

is_keep() {
  local name="$1" k
  for k in "${KEEP[@]}"; do
    [[ "${name}" == "${k}" ]] && return 0
  done
  return 1
}

human_bytes() {
  awk -v b="${1:-0}" 'BEGIN {
    u="B"; n=b+0;
    if (n>1024) { n/=1024; u="K" }
    if (n>1024) { n/=1024; u="M" }
    if (n>1024) { n/=1024; u="G" }
    printf "%.1f%s", n, u
  }'
}

act() {
  # act LABEL path
  local label="$1" path="$2"
  if [[ "${APPLY}" -eq 1 ]]; then
    echo "${label} ${path}"
  else
    echo "WOULD ${label} ${path}"
  fi
}

# --- module: relocate small aux log campaigns ---------------------------------
module_relocate_aux() {
  mkdir -p "${EXP}/_paper_aux"
  local aux
  for aux in "${AUX_RELOCATE[@]}"; do
    if [[ -d "${EXP}/${aux}" && ! -e "${EXP}/_paper_aux/${aux}" ]]; then
      act "RELOCATE→_paper_aux" "${aux}"
      if [[ "${APPLY}" -eq 1 ]]; then
        mv "${EXP}/${aux}" "${EXP}/_paper_aux/${aux}"
      fi
    fi
  done
}

# --- module: prune top-level non-whitelist ------------------------------------
module_prune_toplevel() {
  [[ "${DO_PRUNE}" -eq 1 ]] || return 0
  echo "--- prune top-level (keep: ${KEEP[*]}) ---"
  local path base sz
  local freed=0
  while IFS= read -r -d '' path; do
    base="$(basename "${path}")"
    is_keep "${base}" && continue
    sz="$(du -sb "${path}" 2>/dev/null | awk '{print $1}')"
    freed=$((freed + sz))
    act "DELETE" "${base} ($(human_bytes "${sz}"))"
    if [[ "${APPLY}" -eq 1 ]]; then
      rm -rf "${path}"
    fi
  done < <(find "${EXP}" -mindepth 1 -maxdepth 1 -print0)
  echo "Prune reclaimable ≈ $(human_bytes "${freed}")"
}

# --- module: strip run checkpoints / weight dumps (not pretrained_weights) ---
module_strip_checkpoints() {
  [[ "${DO_STRIP}" -eq 1 ]] || return 0
  echo "--- strip checkpoints / *.pt *.pth *.ckpt (except pretrained_weights/) ---"
  local path
  # Checkpoint directories anywhere except under pretrained_weights
  while IFS= read -r -d '' path; do
    [[ "${path}" == *"/pretrained_weights/"* ]] && continue
    act "RMDIR" "${path#"${EXP}/"}"
    if [[ "${APPLY}" -eq 1 ]]; then
      rm -rf "${path}"
    fi
  done < <(find "${EXP}" -type d \( -name 'checkpoints' -o -name 'task0_warmup*' -o -name 'warmup_ckpts' \) -print0 2>/dev/null)

  while IFS= read -r -d '' path; do
    [[ "${path}" == *"/pretrained_weights/"* ]] && continue
    act "RM" "${path#"${EXP}/"}"
    if [[ "${APPLY}" -eq 1 ]]; then
      rm -f "${path}"
    fi
  done < <(find "${EXP}" -type f \( -name '*.pt' -o -name '*.pth' -o -name '*.ckpt' -o -name '*.safetensors' \) -print0 2>/dev/null)
}

# --- module: inside campaigns keep only logs (+ light summaries) -------------
module_logs_only_campaigns() {
  [[ "${DO_LOGS_ONLY}" -eq 1 ]] || return 0
  echo "--- campaign dirs: keep *.log + summary *.md/*.csv/*.tsv/*.tex; drop other trees ---"
  local camp root path rel
  local -a roots=()
  for camp in "${PAPER_CAMPAIGNS[@]}"; do
    roots+=("${EXP}/${camp}")
  done
  # Also clean aux campaigns if present
  if [[ -d "${EXP}/_paper_aux" ]]; then
    while IFS= read -r -d '' path; do
      roots+=("${path}")
    done < <(find "${EXP}/_paper_aux" -mindepth 1 -maxdepth 1 -type d -print0 2>/dev/null)
  fi

  for root in "${roots[@]}"; do
    [[ -d "${root}" ]] || continue
    echo "  clean ${root#"${EXP}/"}"
    # Remove nested method/run trees that are not flat logs
    while IFS= read -r -d '' path; do
      rel="${path#"${root}/"}"
      # Keep flat files we care about
      case "${rel}" in
        *.log|*.md|*.csv|*.tsv|*.tex) continue ;;
      esac
      # Drop nested run trees (legacy selfevo/ dirs, empty ckpt trees) and stray files
      act "RM" "${path#"${EXP}/"}"
      if [[ "${APPLY}" -eq 1 ]]; then
        rm -rf "${path}"
      fi
    done < <(find "${root}" -mindepth 1 -maxdepth 1 -print0)
  done
}

# --- module: prune empty directories ------------------------------------------
module_prune_empty() {
  echo "--- prune empty directories ---"
  if [[ "${APPLY}" -eq 1 ]]; then
    find "${EXP}" -type d -empty -not -path "${EXP}/pretrained_weights*" -delete 2>/dev/null || true
  else
    find "${EXP}" -type d -empty -not -path "${EXP}/pretrained_weights*" 2>/dev/null | while read -r path; do
      echo "WOULD RMDIR empty ${path#"${EXP}/"}"
    done
  fi
}

# --- main --------------------------------------------------------------------
if [[ ! -d "${EXP}" ]]; then
  echo "No ${EXP}; nothing to do."
  exit 0
fi

echo "=== cleanup_experiments (apply=${APPLY} prune=${DO_PRUNE} strip=${DO_STRIP} logs_only=${DO_LOGS_ONLY}) ==="
module_relocate_aux
module_prune_toplevel
module_strip_checkpoints
module_logs_only_campaigns
module_prune_empty

echo
if [[ "${APPLY}" -eq 0 ]]; then
  echo "Dry-run only. Re-run with --apply to delete."
else
  echo "Done. Layout:"
  du -sh "${EXP}" "${EXP}"/* 2>/dev/null || true
  echo
  echo "Remaining .pt outside pretrained_weights:"
  find "${EXP}" -name '*.pt' ! -path '*/pretrained_weights/*' 2>/dev/null || echo "(none)"
  echo "Remaining checkpoints dirs:"
  find "${EXP}" -type d -name 'checkpoints' 2>/dev/null || echo "(none)"
fi
