#!/usr/bin/env bash
# Remote Leonardo helpers from Surrey (or any host with a live Cineca SSH cert).
#
# Prerequisites:
#   1) Short-lived Cineca OIDC cert in an ssh-agent (via sync.sh / leonardo.sh).
#      Certs expire; renew with browser SSO when needed.
#   2) Network path to login.leonardo.cineca.it
#
# Usage:
#   bash scripts/leonardo_remote.sh check
#   bash scripts/leonardo_remote.sh squeue
#   bash scripts/leonardo_remote.sh sync-file datasets/lgsp_fscil.py
#   bash scripts/leonardo_remote.sh sbatch scripts/reproduce_image.sh --array=2
#   bash scripts/leonardo_remote.sh sh 'cd $REMOTE_DIR && bash scripts/check_frost_image_data.sh'
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CINECA_USER="${CINECA_USER:-vthengan}"
LEONARDO_HOST="${LEONARDO_HOST:-login.leonardo.cineca.it}"
REMOTE_DIR="${REMOTE_DIR:-/leonardo_scratch/fast/EUHPC_D36_199/projects/2027-ICLR-FROST}"
SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o BatchMode=yes -o ConnectTimeout=15"

die() { echo "ERROR: $*" >&2; exit 1; }

find_cineca_agent() {
  # Prefer an already-exported sock if it works.
  if [[ -n "${SSH_AUTH_SOCK:-}" ]]; then
    if ssh-add -l 2>/dev/null | grep -qiE 'cineca|surrey.ac.uk|ECDSA-CERT|ED25519-CERT'; then
      echo "${SSH_AUTH_SOCK}"
      return 0
    fi
  fi
  local sock
  while IFS= read -r sock; do
    [[ -S "${sock}" ]] || continue
    if SSH_AUTH_SOCK="${sock}" ssh-add -l 2>/dev/null | grep -qiE 'cineca|surrey.ac.uk|ECDSA-CERT|ED25519-CERT'; then
      echo "${sock}"
      return 0
    fi
  done < <(find /tmp -maxdepth 3 -type s -name 'agent*' 2>/dev/null)
  return 1
}

ensure_agent() {
  local sock
  sock="$(find_cineca_agent)" || die \
    "No Cineca SSH certificate found in any ssh-agent.
Renew once (interactive), then retry:
  bash sync.sh --code --force
  # or: bash leonardo.sh --renew
On Surrey this needs a Mac tunnel:
  ssh -N -L 10000:127.0.0.1:10000 surrey_ws"
  export SSH_AUTH_SOCK="${sock}"
}

ssh_leo() {
  ensure_agent
  # shellcheck disable=SC2086
  ssh ${SSH_OPTS} "${CINECA_USER}@${LEONARDO_HOST}" "$@"
}

cmd="${1:-}"
shift || true

case "${cmd}" in
  check)
    ensure_agent
    echo "SSH_AUTH_SOCK=${SSH_AUTH_SOCK}"
    ssh-add -l | head -3
    ssh_leo 'hostname; echo CONNECTED; ls '"${REMOTE_DIR}"'/scripts/reproduce_image.sh'
    ;;
  squeue)
    ssh_leo "squeue -u ${CINECA_USER}"
    ;;
  sync-file)
    [[ $# -ge 1 ]] || die "usage: sync-file <repo-relative-path>"
    rel="$1"
    [[ -f "${REPO}/${rel}" ]] || die "missing local file: ${REPO}/${rel}"
    ensure_agent
    # shellcheck disable=SC2086
    rsync -avz -e "ssh ${SSH_OPTS}" \
      "${REPO}/${rel}" \
      "${CINECA_USER}@${LEONARDO_HOST}:${REMOTE_DIR}/${rel}"
    ;;
  sbatch)
    [[ $# -ge 1 ]] || die "usage: sbatch <script> [sbatch-args...]"
    script="$1"; shift
    ssh_leo "cd ${REMOTE_DIR} && sbatch $* ${script} && squeue -u ${CINECA_USER} | head -20"
    ;;
  sh|ssh)
    [[ $# -ge 1 ]] || die "usage: sh '<remote command>'"
    ssh_leo "cd ${REMOTE_DIR} && $*"
    ;;
  ""|-h|--help|help)
    sed -n '2,16p' "$0" | sed 's/^# \{0,1\}//'
    ;;
  *)
    die "unknown command: ${cmd} (try: check|squeue|sync-file|sbatch|sh)"
    ;;
esac
