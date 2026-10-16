#!/usr/bin/env bash
# Download official FGVC-Aircraft 2013b for image FSCIL.
#
# Extracts as fgvc-aircraft-2013b then renames to FGVC-Aircraft (canonical layout).
#
# Usage:
#   bash datasets/preprocess/download_fgvc_aircraft.sh
#   bash datasets/preprocess/download_fgvc_aircraft.sh /vol/research/.../2027-ICLR-FROST/_datasets/images
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEST="${1:-${REPO}/_datasets/images}"
mkdir -p "${DEST}"
DEST="$(cd "${DEST}" && pwd)"
TGZ="${DEST}/fgvc-aircraft-2013b.tar.gz"
URL="https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz"
CANON="${DEST}/FGVC-Aircraft"

if [[ -d "${CANON}/data/images" ]]; then
  echo "FGVC-Aircraft already present: ${CANON}"
  exit 0
fi

# Migrate older local names.
for old in "${DEST}/Aircraft" "${DEST}/fgvc-aircraft-2013b"; do
  if [[ -d "${old}/data/images" && ! -e "${CANON}" ]]; then
    mv "${old}" "${CANON}"
    echo "Renamed ${old} -> ${CANON}"
    exit 0
  fi
done

echo "Downloading FGVC-Aircraft (~2.6GB) -> ${TGZ}"
curl -L --fail --retry 3 -o "${TGZ}" "${URL}"
echo "Extracting with /usr/bin/tar ..."
/usr/bin/tar -xzf "${TGZ}" -C "${DEST}"
rm -f "${TGZ}"
mv "${DEST}/fgvc-aircraft-2013b" "${CANON}"
echo "Done: ${CANON}"
ls "${CANON}/data" | head
