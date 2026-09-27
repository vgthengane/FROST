#!/usr/bin/env bash
# Download official FGVC-Aircraft 2013b for LGSP image FSCIL.
#
# Usage:
#   bash scripts/download_fgvc_aircraft.sh
#   bash scripts/download_fgvc_aircraft.sh /vol/research/.../SelfEvo-CL3D/_datasets/images
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${1:-${REPO}/_datasets/images}"
mkdir -p "${DEST}"
DEST="$(cd "${DEST}" && pwd)"
TGZ="${DEST}/fgvc-aircraft-2013b.tar.gz"
URL="https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz"

if [[ -d "${DEST}/fgvc-aircraft-2013b/data/images" ]]; then
  echo "Aircraft already present: ${DEST}/fgvc-aircraft-2013b"
  exit 0
fi

echo "Downloading FGVC-Aircraft (~2.6GB) -> ${TGZ}"
curl -L --fail --retry 3 -o "${TGZ}" "${URL}"
echo "Extracting with /usr/bin/tar ..."
/usr/bin/tar -xzf "${TGZ}" -C "${DEST}"
rm -f "${TGZ}"
echo "Done: ${DEST}/fgvc-aircraft-2013b"
ls "${DEST}/fgvc-aircraft-2013b/data" | head
