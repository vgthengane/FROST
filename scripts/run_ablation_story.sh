#!/bin/bash
# Run ablation story: A (reframe existing MN40 logs) + submit B (harder regimes on Leonardo).
#
# On Leonardo login node (after sync):
#   bash scripts/run_ablation_story.sh summarize-only   # A only
#   bash scripts/run_ablation_story.sh submit-regimes   # B sbatch 12 jobs
#   bash scripts/run_ablation_story.sh submit-baseline  # missing full MN40 array 0
#   bash scripts/run_ablation_story.sh summarize-all    # A + B tables after jobs finish
#
# Smoke (debug QoS, one 1-shot full variant, ~10 min):
#   sbatch --qos=boost_qos_dbg --time=00:30:00 --array=0 \
#     scripts/simpl_runs_ablations_regimes.sh --smoke --regime mn40_1shot

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

RUNNER=(aprun python)
MODE="${1:-summarize-all}"

summarize_a() {
  echo "=== A: MN40 ablations (Base/Novel/Harm reframe) ==="
  "${RUNNER[@]}" scripts/summarize_fscil_ablations_mn40.py \
    --out _experiments/00_SIMPL_ABLATIONS_MN40/ablation_tables.md
}

summarize_b() {
  echo "=== B: Harder-regime module ablations ==="
  "${RUNNER[@]}" scripts/summarize_fscil_regime_ablations.py \
    --out _experiments/00_SIMPL_ABLATIONS_REGIMES/regime_ablation_tables.md
}

case "${MODE}" in
  summarize-only)
    summarize_a
    ;;
  summarize-all)
    summarize_a
    summarize_b
    ;;
  submit-regimes)
    echo "Submitting B: 12 harder-regime jobs (array 0-11)..."
    sbatch scripts/simpl_runs_ablations_regimes.sh
    echo "Monitor: squeue -u \$USER"
    ;;
  submit-baseline)
    echo "Submitting missing MN40 full baseline (array 0)..."
    sbatch --array=0 scripts/simpl_runs_ablations_mn40.sh --skip-done
    ;;
  *)
    echo "Usage: $0 {summarize-only|summarize-all|submit-regimes|submit-baseline}" >&2
    exit 1
    ;;
esac
