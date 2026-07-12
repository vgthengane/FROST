#!/bin/bash
# Build CO3D 1024 FPS + normal train/test caches (ModelNet / ScanObjectNN style).
#
# Reads official CO3D pointcloud.ply sequences + microshape FSCIL index files.
# Writes:
#   _datasets/CO3D/co3d_train_1024pts_fps_normal.dat
#   _datasets/CO3D/co3d_test_1024pts_fps_normal.dat
#   _datasets/CO3D/co3d_class_names.txt
#
# Usage:
#   bash scripts/run_build_co3d_1024_normals.sh
#   bash scripts/run_build_co3d_1024_normals.sh --background
#   bash scripts/run_build_co3d_1024_normals.sh --workers 12 --background
#   bash scripts/run_build_co3d_1024_normals.sh --force --background
#
# Environment overrides:
#   OFFICIAL_ROOT   raw CO3D tree (category/seq/pointcloud.ply)
#   OUT_ROOT        output directory (default: <repo>/_datasets/CO3D)
#   INDEX_ROOT      microshape FSCIL index files
#   WORKERS         parallel worker processes (default: nproc - 2)
#   SCRATCH_BIND    host path to bind into aprun (default: /scratch/vt00262)
#   APPTAINER_IMAGE container image name under /scratch/vt00262/apptainer/images
#   LOG             log file path
#   SCREEN_NAME     screen session name (default: co3d_1024_build)

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DEFAULT_OFFICIAL="/scratch/vt00262/datasets/_helpers/co3d/_datasets/co3d"
DEFAULT_OUT="${REPO}/_datasets/CO3D"
DEFAULT_WORKERS="$(nproc 2>/dev/null || echo 8)"
if [[ "$DEFAULT_WORKERS" -gt 2 ]]; then
  DEFAULT_WORKERS=$((DEFAULT_WORKERS - 2))
else
  DEFAULT_WORKERS=1
fi

OFFICIAL_ROOT="${OFFICIAL_ROOT:-$DEFAULT_OFFICIAL}"
OUT_ROOT="${OUT_ROOT:-$DEFAULT_OUT}"
INDEX_ROOT="${INDEX_ROOT:-${REPO}/third_party/microshape/data/index_files/co3d}"
WORKERS="${WORKERS:-$DEFAULT_WORKERS}"
SCRATCH_BIND="${SCRATCH_BIND:-/scratch/vt00262}"
APPTAINER_IMAGE="${APPTAINER_IMAGE:-${CLIMB3D_PP_SIF:-climb3d_pp.sif}}"
LOG="${LOG:-${REPO}/_experiments/logs/co3d_1024_normals_build.log}"
SCREEN_NAME="${SCREEN_NAME:-co3d_1024_build}"
PID_FILE="${OUT_ROOT}/.co3d_1024_build.pid"

RUN_MODE="foreground"
FORCE="${FORCE:-false}"
EXPORT_PT="${EXPORT_PT:-false}"

usage() {
  sed -n '2,24p' "$0" | sed 's/^# \{0,1\}//'
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --background|-b)
      RUN_MODE="background"
      shift
      ;;
    --screen|-s)
      RUN_MODE="screen"
      shift
      ;;
    --force)
      FORCE=true
      shift
      ;;
    --export-pt)
      EXPORT_PT=true
      shift
      ;;
    --workers)
      WORKERS="${2:?--workers requires a value}"
      shift 2
      ;;
    --workers=*)
      WORKERS="${1#*=}"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

python_args() {
  local -a args=(--workers "$WORKERS")
  [[ "$FORCE" == true ]] && args+=(--force)
  [[ "$EXPORT_PT" == true ]] && args+=(--export-pt)
  printf '%s\n' "${args[@]}"
}

aprun_python() {
  local -a cmd=(aprun)
  if [[ -d "$SCRATCH_BIND" ]]; then
    cmd+=(--bind "${SCRATCH_BIND}:${SCRATCH_BIND}")
  fi
  if [[ -n "$APPTAINER_IMAGE" ]]; then
    cmd+=(--image "$APPTAINER_IMAGE")
  fi
  cmd+=(python "$@")
  "${cmd[@]}"
}

run_build() {
  mkdir -p "$(dirname "$LOG")" "$OUT_ROOT"

  if [[ ! -d "$OFFICIAL_ROOT" ]]; then
    echo "Error: CO3D official root not found: $OFFICIAL_ROOT" >&2
    exit 1
  fi
  if [[ ! -d "$INDEX_ROOT" ]]; then
    echo "Error: CO3D index root not found: $INDEX_ROOT" >&2
    exit 1
  fi

  mapfile -t PY_ARGS < <(python_args)

  {
    echo "[$(date -Is)] CO3D 1024+normals build start (pid=$$)"
    echo "  repo:          $REPO"
    echo "  official-root: $OFFICIAL_ROOT"
    echo "  output-root:   $OUT_ROOT"
    echo "  index-root:    $INDEX_ROOT"
    echo "  workers:       $WORKERS"
    echo "  scratch-bind:  $SCRATCH_BIND"
    echo "  image:         $APPTAINER_IMAGE"
    echo "  python-args:   ${PY_ARGS[*]}"
  } | tee -a "$LOG"

  aprun_python "${REPO}/scripts/build_co3d_official_1024_normals.py" \
    --official-root "$OFFICIAL_ROOT" \
    --output-root "$OUT_ROOT" \
    --index-root "$INDEX_ROOT" \
    "${PY_ARGS[@]}" 2>&1 | tee -a "$LOG"

  echo "[$(date -Is)] CO3D 1024+normals build finished" | tee -a "$LOG"
  rm -f "$PID_FILE"
}

dispatch_background() {
  mkdir -p "$(dirname "$LOG")" "$OUT_ROOT"
  if [[ -f "$PID_FILE" ]]; then
    old_pid="$(cat "$PID_FILE" 2>/dev/null || true)"
    if [[ -n "$old_pid" ]] && kill -0 "$old_pid" 2>/dev/null; then
      echo "Build already running (pid=${old_pid})."
      echo "  tail -f ${LOG}"
      exit 0
    fi
  fi

  nohup env \
    OFFICIAL_ROOT="$OFFICIAL_ROOT" \
    OUT_ROOT="$OUT_ROOT" \
    INDEX_ROOT="$INDEX_ROOT" \
    WORKERS="$WORKERS" \
    SCRATCH_BIND="$SCRATCH_BIND" \
    APPTAINER_IMAGE="$APPTAINER_IMAGE" \
    LOG="$LOG" \
    FORCE="$FORCE" \
    EXPORT_PT="$EXPORT_PT" \
    bash "$0" \
    > "${LOG}.nohup" 2>&1 &

  echo $! > "$PID_FILE"
  echo "Started background build (pid=$(cat "$PID_FILE"), workers=${WORKERS})."
  echo "  output: ${OUT_ROOT}"
  echo "  tail -f ${LOG}"
  echo "  kill \$(cat ${PID_FILE})   # stop"
}

dispatch_screen() {
  if command -v screen >/dev/null 2>&1 && screen -list 2>/dev/null | grep -q "[.]${SCREEN_NAME}[[:space:]]"; then
    echo "Screen session '${SCREEN_NAME}' already running."
    echo "  screen -r ${SCREEN_NAME}"
    echo "  tail -f ${LOG}"
    exit 0
  fi

  local -a flags=()
  [[ "$FORCE" == true ]] && flags+=(--force)
  [[ "$EXPORT_PT" == true ]] && flags+=(--export-pt)
  flags+=(--workers "$WORKERS")

  screen -dmS "${SCREEN_NAME}" bash -lc \
    "cd \"${REPO}\" && OFFICIAL_ROOT=\"${OFFICIAL_ROOT}\" OUT_ROOT=\"${OUT_ROOT}\" INDEX_ROOT=\"${INDEX_ROOT}\" WORKERS=\"${WORKERS}\" SCRATCH_BIND=\"${SCRATCH_BIND}\" APPTAINER_IMAGE=\"${APPTAINER_IMAGE}\" LOG=\"${LOG}\" FORCE=\"${FORCE}\" EXPORT_PT=\"${EXPORT_PT}\" bash \"${REPO}/scripts/run_build_co3d_1024_normals.sh\" ${flags[*]}"

  echo "Started screen session: ${SCREEN_NAME} (workers=${WORKERS})"
  echo "  output: ${OUT_ROOT}"
  echo "  tail -f ${LOG}"
  echo "  screen -r ${SCREEN_NAME}"
}

case "$RUN_MODE" in
  background) dispatch_background ;;
  screen) dispatch_screen ;;
  *) run_build ;;
esac
