#!/bin/bash
# Wait for MN40 base-only canonical to finish, then run exemplar-buffer-free.
#
#   bash scripts/run_mn40_baseonly_bufree_after_canonical.sh
#   screen -S mn40_bf -dm bash scripts/run_mn40_baseonly_bufree_after_canonical.sh
#   screen -r mn40_bf

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

LOGROOT="_experiments/00_SIMPL_ABL_ADAPTER"
CANON_LOG="${LOGROOT}/abl_mn40_baseonly_canonical.log"
CANON_FLAT="${LOGROOT}/abl_mn40_baseonly_canonical_seed=2008.log"
BF_LOG="${LOGROOT}/abl_mn40_baseonly_exbuffree.log"
N_SESS=5
CANON_PID="${1:-}"

log() { echo "[$(date -Is)] $*"; }

log_is_complete() {
  local logfile="$1"
  [[ -f "${logfile}" ]] || return 1
  local last
  last="$(grep "CNN top1 curve:" "${logfile}" 2>/dev/null | tail -n 1 || true)"
  [[ -n "${last}" ]] || return 1
  local n
  n="$(echo "${last}" | tr -cd ',' | wc -c)"
  n=$((n + 1))
  [[ "${n}" -ge "${N_SESS}" ]]
}

wait_for_canonical() {
  log "Waiting for canonical (need ${N_SESS} sessions in log)..."
  while true; do
    if log_is_complete "${CANON_FLAT}" || log_is_complete "${CANON_LOG}"; then
      log "Canonical complete."
      return 0
    fi
    if [[ -n "${CANON_PID}" ]] && ! kill -0 "${CANON_PID}" 2>/dev/null; then
      if log_is_complete "${CANON_FLAT}" || log_is_complete "${CANON_LOG}"; then
        log "Canonical process exited with complete log."
        return 0
      fi
      log "ERROR: canonical PID ${CANON_PID} exited but log incomplete."
      tail -20 "${CANON_LOG}" 2>/dev/null || true
      exit 1
    fi
    sleep 60
  done
}

if log_is_complete "${BF_LOG}" || log_is_complete "${LOGROOT}/abl_mn40_baseonly_exbuffree_seed=2008.log"; then
  log "SKIP buffer-free — already complete."
  exit 0
fi

wait_for_canonical

log "Starting exemplar-buffer-free (base-only adapter, MN40)..."
bash scripts/run_mn40_baseonly_canonical_and_bufree.sh --only bufferfree

log "Done."
