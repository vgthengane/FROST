#!/bin/bash
# Deprecated wrapper — use scripts/run_build_co3d_1024_normals.sh instead.
#
# Usage:
#   bash scripts/run_build_co3d_1024_normals.sh --background

set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo "Note: run_co3d_official_build.sh is deprecated." >&2
echo "Use: bash scripts/run_build_co3d_1024_normals.sh [--background|--screen]" >&2
exec bash "${REPO}/scripts/run_build_co3d_1024_normals.sh" "$@"
