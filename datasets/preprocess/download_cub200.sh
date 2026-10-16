#!/usr/bin/env bash
# Download official CUB-200-2011 for LGSP image FSCIL.
#
# Prefer a path under /vol/research/... so default ``aprun`` can see it
# (``/scratch`` is NOT bound unless you pass ``--bind /scratch``).
#
# Extracts as CUB_200_2011 then renames to CUB200 (canonical FROST layout).
#
# Usage:
#   bash datasets/preprocess/download_cub200.sh
#   bash datasets/preprocess/download_cub200.sh /vol/research/.../2027-ICLR-FROST/_datasets/images
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEST="${1:-${REPO}/_datasets/images}"
mkdir -p "${DEST}"
DEST="$(cd "${DEST}" && pwd)"
TGZ="${DEST}/CUB_200_2011.tgz"
URL="https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz"

if [[ -d "${DEST}/CUB200/images" ]]; then
  echo "CUB already present: ${DEST}/CUB200"
  exit 0
fi

# Finish a partial extract that still has the upstream name.
if [[ -d "${DEST}/CUB_200_2011/images" && ! -e "${DEST}/CUB200" ]]; then
  mv "${DEST}/CUB_200_2011" "${DEST}/CUB200"
  echo "Renamed ${DEST}/CUB_200_2011 -> ${DEST}/CUB200"
  exit 0
fi

echo "Downloading CUB-200-2011 -> ${TGZ}"
curl -L --fail --retry 3 -o "${TGZ}" "${URL}"
echo "Extracting with /usr/bin/tar ..."
/usr/bin/tar -xzf "${TGZ}" -C "${DEST}"
rm -f "${TGZ}"
mv "${DEST}/CUB_200_2011" "${DEST}/CUB200"
echo "Done: ${DEST}/CUB200"
ls "${DEST}/CUB200" | head
