#!/usr/bin/env bash
# Download official CUB-200-2011 for LGSP image FSCIL.
#
# Prefer a path under /vol/research/... so default ``aprun`` can see it
# (``/scratch`` is NOT bound unless you pass ``--bind /scratch``).
#
# Usage:
#   bash scripts/download_cub200.sh
#   bash scripts/download_cub200.sh /vol/research/.../SelfEvo-CL3D/_datasets/images
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${1:-${REPO}/_datasets/images}"
mkdir -p "${DEST}"
DEST="$(cd "${DEST}" && pwd)"
TGZ="${DEST}/CUB_200_2011.tgz"
URL="https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz"

if [[ -d "${DEST}/CUB_200_2011/images" ]]; then
  echo "CUB already present: ${DEST}/CUB_200_2011"
  exit 0
fi

echo "Downloading CUB-200-2011 -> ${TGZ}"
curl -L --fail --retry 3 -o "${TGZ}" "${URL}"
echo "Extracting with /usr/bin/tar ..."
/usr/bin/tar -xzf "${TGZ}" -C "${DEST}"
rm -f "${TGZ}"
echo "Done: ${DEST}/CUB_200_2011"
ls "${DEST}/CUB_200_2011" | head
