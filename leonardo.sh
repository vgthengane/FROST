#!/bin/bash
# ==============================================================================
# leonardo — One-command connection to Leonardo EuroHPC
# ==============================================================================
#
# USAGE:
#   leonardo                  SSH into Leonardo (auto-detects machine)
#   leonardo --gpu            SSH + launch 1-GPU interactive session
#   leonardo --gpu=4          SSH + launch 4-GPU interactive session
#   leonardo --check          Check certificate status only
#   leonardo --renew          Force renew certificate then connect
#   leonardo --help           Show this help
#
# SETUP (run once):
#   Mac:           cp leonardo.sh /usr/local/bin/leonardo && chmod +x /usr/local/bin/leonardo
#   Surrey server: cp leonardo.sh ~/.local/bin/leonardo   && chmod +x ~/.local/bin/leonardo
# ==============================================================================

set -euo pipefail

# --- Config -------------------------------------------------------------------

CINECA_USER="vthengan"
SURREY_USER="vt00262"
SURREY_EMAIL="v.thengane@surrey.ac.uk"
SURREY_WS="surrey_ws"
SURREY_SERVER="ulws102"
LEONARDO_HOST="login05-ext.leonardo.cineca.it"
CA_URL="https://sshproxy.hpc.cineca.it"
PROVISIONER="cineca-hpc"
OAUTH_PORT=10000

# GPU session defaults (Leonardo Booster)
#   partition boost_usr_prod + qos boost_qos_dbg: wall ≤00:30:00, priority 80,
#   max 2 jobs running/pending per account — interactive only, not full campaigns.
#   Full training: omit --qos (normal) or --qos=normal, wall ≤24:00:00, priority 40.
GPU_ACCOUNT="EUHPC_D36_199"
GPU_PARTITION="boost_usr_prod"
GPU_QOS="boost_qos_dbg"
GPU_TIME="00:30:00"
GPU_CPUS=24
GPU_MEM="80G"

# --- Terminal styling ---------------------------------------------------------

if [ -t 1 ]; then
  RED="\033[0;31m"; GREEN="\033[0;32m"; YELLOW="\033[0;33m"
  CYAN="\033[0;36m"; DIM="\033[2m"; BOLD="\033[1m"; NC="\033[0m"
else
  RED=""; GREEN=""; YELLOW=""; CYAN=""; DIM=""; BOLD=""; NC=""
fi

log_info()    { echo -e "${CYAN}  →${NC} $*"; }
log_success() { echo -e "${GREEN}  ✓${NC} $*"; }
log_warn()    { echo -e "${YELLOW}  ⚠${NC} $*"; }
log_error()   { echo -e "${RED}  ✗${NC} $*" >&2; }

divider() { echo -e "${DIM}──────────────────────────────────────────${NC}"; }

print_banner() {
  echo ""
  divider
  echo -e "${BOLD}${CYAN}  Leonardo EuroHPC${NC}  ${DIM}connecting as ${CINECA_USER}${NC}"
  divider
  echo ""
}

# --- Detect which machine we are on ------------------------------------------

detect_machine() {
  local hostname
  hostname=$(hostname)
  case "$hostname" in
    *PKQM*|*MacBook*|*.local|*mbp*)
      echo "mac"
      ;;
    *ulws*|*surrey*|*uos*)
      echo "server"
      ;;
    *)
      # Fallback: check if we can reach surrey gateway directly
      if ssh -o BatchMode=yes -o ConnectTimeout=3 "$SURREY_WS" exit 2>/dev/null; then
        echo "mac"
      else
        echo "server"
      fi
      ;;
  esac
}

# --- Certificate management --------------------------------------------------

cert_is_valid() {
  # Returns 0 if a valid cineca certificate is loaded in ssh-agent
  ssh-add -L 2>/dev/null | grep -q "cineca\|step\|OIDC" && return 0

  # Also check if any cert is loaded and not expired
  if ssh-add -L 2>/dev/null | grep -q "cert"; then
    return 0
  fi

  # Try a quick SSH probe to Leonardo
  if ssh -o BatchMode=yes \
         -o ConnectTimeout=5 \
         -o StrictHostKeyChecking=no \
         "${CINECA_USER}@${LEONARDO_HOST}" exit 2>/dev/null; then
    return 0
  fi

  return 1
}

ensure_ssh_agent() {
  if ! ssh-add -l &>/dev/null; then
    log_info "Starting ssh-agent..."
    eval "$(ssh-agent -s)" > /dev/null
    log_success "ssh-agent started"
  fi
}

free_oauth_port() {
  local pid
  pid=$(lsof -ti ":${OAUTH_PORT}" 2>/dev/null || true)
  if [ -n "$pid" ]; then
    log_warn "Port ${OAUTH_PORT} in use — clearing..."
    echo "$pid" | xargs kill -9 2>/dev/null || true
    sleep 1
  fi
}

renew_cert_mac() {
  log_info "Getting Leonardo certificate via browser..."
  free_oauth_port
  ensure_ssh_agent

  step ssh login \
    --ca-url "$CA_URL" \
    --provisioner "$PROVISIONER" \
    "$SURREY_EMAIL"

  log_success "Certificate obtained"
}

renew_cert_server() {
  log_info "Setting up port forward from Mac to server..."
  echo ""
  echo -e "  ${YELLOW}Run this on your Mac in a separate terminal:${NC}"
  echo -e "  ${BOLD}ssh -N -L ${OAUTH_PORT}:127.0.0.1:${OAUTH_PORT} ${SURREY_WS}${NC}"
  echo ""
  read -r -p "  Press Enter once the tunnel is running on your Mac... "

  free_oauth_port
  ensure_ssh_agent

  step ssh login \
    --ca-url "$CA_URL" \
    --provisioner "$PROVISIONER" \
    "$SURREY_EMAIL"

  log_success "Certificate obtained"
}

# --- Known hosts cleanup -----------------------------------------------------

fix_known_hosts() {
  ssh-keygen -f "${HOME}/.ssh/known_hosts" -R "$LEONARDO_HOST" 2>/dev/null || true
  # Also clean the generic login endpoint
  ssh-keygen -f "${HOME}/.ssh/known_hosts" -R "login.leonardo.cineca.it" 2>/dev/null || true
}

# --- SSH options -------------------------------------------------------------

ssh_opts() {
  echo "-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"
}

# --- Connect -----------------------------------------------------------------

connect_direct() {
  local gpu_count="${1:-0}"

  log_info "Connecting to Leonardo..."
  echo ""

  if [ "$gpu_count" -eq 0 ]; then
    # Plain shell
    ssh $(ssh_opts) "${CINECA_USER}@${LEONARDO_HOST}"
  else
    # GPU interactive session
    local cpus=$GPU_CPUS
    local mem=$GPU_MEM
    [ "$gpu_count" -gt 1 ] && cpus=32 && mem="200G"

    local srun_cmd="srun \
      --account=${GPU_ACCOUNT} \
      --partition=${GPU_PARTITION} \
      --qos=${GPU_QOS} \
      --time=${GPU_TIME} \
      --nodes=1 \
      --ntasks=1 \
      --cpus-per-task=${cpus} \
      --mem=${mem} \
      --gres=gpu:${gpu_count} \
      --pty bash"

    log_info "Launching ${gpu_count}-GPU interactive session..."
    ssh $(ssh_opts) -t "${CINECA_USER}@${LEONARDO_HOST}" "$srun_cmd"
  fi
}

# --- Parse args --------------------------------------------------------------

MODE="connect"
GPU_COUNT=0
FORCE_RENEW=false

for arg in "$@"; do
  case "$arg" in
    --gpu)        GPU_COUNT=1 ;;
    --gpu=*)      GPU_COUNT="${arg#*=}" ;;
    --check)      MODE="check" ;;
    --renew)      FORCE_RENEW=true ;;
    --help|-h)
      sed -n '3,14p' "$0" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *)
      log_error "Unknown option: $arg"
      exit 1
      ;;
  esac
done

# --- Main --------------------------------------------------------------------

main() {
  print_banner

  local machine
  machine=$(detect_machine)
  log_info "Detected: ${BOLD}${machine}${NC}"

  # Check-only mode
  if [ "$MODE" = "check" ]; then
    ensure_ssh_agent
    if cert_is_valid; then
      log_success "Certificate is valid — ready to connect"
      echo ""
      ssh-add -L 2>/dev/null | head -1 | awk '{print "  Key:", $NF}'
    else
      log_warn "No valid certificate found"
      echo -e "  Run: ${CYAN}leonardo --renew${NC}"
    fi
    echo ""
    exit 0
  fi

  # Ensure agent is running
  ensure_ssh_agent

  # Renew certificate if needed or forced
  if [ "$FORCE_RENEW" = true ] || ! cert_is_valid; then
    if [ "$FORCE_RENEW" = false ]; then
      log_warn "No valid certificate — renewing..."
    fi
    echo ""
    if [ "$machine" = "mac" ]; then
      renew_cert_mac
    else
      renew_cert_server
    fi
    echo ""
  else
    log_success "Certificate valid — skipping login"
  fi

  # Connect
  connect_direct "$GPU_COUNT"
}

main "$@"