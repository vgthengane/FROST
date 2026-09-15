#!/bin/bash
# Adapter campaign launcher (in-domain + cross-domain).
#
# On Leonardo login node (after sync_eurohpc.sh --code):
#
#   bash scripts/run_adapter_ablation_campaign.sh submit-all
#   bash scripts/run_adapter_ablation_campaign.sh submit-indomain
#   bash scripts/run_adapter_ablation_campaign.sh submit-crossdomain
#   bash scripts/run_adapter_ablation_campaign.sh summarize-all
#
# Smoke (debug QoS, one session each):
#   sbatch --qos=boost_qos_dbg --time=01:00:00 --array=0 \
#     scripts/simpl_runs_adapter_ablation_indomain.sh --smoke
#
# Campaign matrix (both datasets: MN40 + SN55):
#   In-domain  (6 jobs):  MN40 + SN55 × {full_2ep, full_10ep, base_only}
#   Cross-dom  (8 jobs):  MN40→ScanObj + SN55→ScanObj × {full_2ep, full_10ep, base_t1_2ep, base_only}
#   submit-all → one array 0-13 via simpl_runs_adapter_ablation_all.sh
#
# Logs:
#   _experiments/00_ADAPTER_CAMPAIGN_ID/
#   _experiments/00_ADAPTER_CAMPAIGN_XD/

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

RUNNER=(aprun python)
MODE="${1:-help}"

summarize_all() {
  echo "=== Summarize adapter campaigns ==="
  "${RUNNER[@]}" scripts/summarize_fscil_adapter_campaign.py --scope all
}

case "${MODE}" in
  submit-indomain)
    echo "Submitting in-domain (MN40 + SN55, array 0-5)..."
    sbatch scripts/simpl_runs_adapter_ablation_indomain.sh
    ;;
  submit-crossdomain)
    echo "Submitting cross-domain (MN40 + SN55 → ScanObj, array 0-7)..."
    sbatch scripts/simpl_runs_adapter_ablation_crossdomain.sh
    ;;
  submit-all)
    echo "Submitting unified campaign (MN40 + SN55, in + cross, array 0-13)..."
    sbatch scripts/simpl_runs_adapter_ablation_all.sh
    echo "Monitor: squeue -u \$USER"
    ;;
  submit-all-split)
    sbatch scripts/simpl_runs_adapter_ablation_indomain.sh
    sbatch scripts/simpl_runs_adapter_ablation_crossdomain.sh
    echo "Monitor: squeue -u \$USER"
    ;;
  summarize-all|summarize)
    summarize_all
    ;;
  *)
    cat <<'EOF'
Usage: bash scripts/run_adapter_ablation_campaign.sh {submit-all|submit-all-split|submit-indomain|submit-crossdomain|summarize-all}

Files:
  scripts/simpl_runs_adapter_ablation_all.sh        — MN40 + SN55, in + cross (array 0-13)
  scripts/simpl_runs_adapter_ablation_indomain.sh   — MN40 + SN55 in-domain (array 0-5)
  scripts/simpl_runs_adapter_ablation_crossdomain.sh — MN40 + SN55 → ScanObj (array 0-7)
  scripts/summarize_fscil_adapter_campaign.py        — tables after jobs finish
EOF
    exit 1
    ;;
esac
