#!/usr/bin/env bash
# Download iNF200 (CLIP-M3 / LGSP fungi FSCIL) from Hugging Face and stage it.
#
# Source: https://huggingface.co/datasets/tldoan/iNF200
#   iNF200_train.tar  (~1.76 GB)  → train_mini/
#   iNF200_val.tar    (~352 MB)   → val/
#
# Expected result:
#   <data_root>/iNF200/train_mini/<class_folder>/*.jpg
#   <data_root>/iNF200/val/<class_folder>/*.jpg
#
# Usage (on a networked login node):
#   bash datasets/preprocess/download_inf200.sh
#   bash datasets/preprocess/download_inf200.sh /leonardo_scratch/.../SelfEvo-CL3D/_datasets/images
#
# Needs: aprun + huggingface_hub (climb3d_pp.sif). Optional: HF_TOKEN for faster downloads.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEST="${1:-${REPO}/_datasets/images}"
mkdir -p "${DEST}"
DEST="$(cd "${DEST}" && pwd)"
TARGET="${DEST}/iNF200"

if [[ -d "${TARGET}/train_mini" && -d "${TARGET}/val" ]]; then
  n_train="$(find "${TARGET}/train_mini" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l | tr -d ' ')"
  n_val="$(find "${TARGET}/val" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l | tr -d ' ')"
  if [[ "${n_train}" -ge 100 && "${n_val}" -ge 100 ]]; then
    echo "iNF200 already present: ${TARGET} (train_mini=${n_train} classes, val=${n_val} classes)"
    exit 0
  fi
fi

mkdir -p "${TARGET}"
echo "Staging iNF200 under ${TARGET}"
echo "Downloading from Hugging Face (tldoan/iNF200) ..."

# Download + extract inside aprun so HF + tar share the same Python env.
# Extract into TARGET; tars already contain top-level train_mini/ and val/.
aprun python - <<PY
import os
import tarfile
from pathlib import Path
from huggingface_hub import hf_hub_download

target = Path(${TARGET@Q})
target.mkdir(parents=True, exist_ok=True)

def stage(filename: str, expect_subdir: str) -> None:
    print(f"Fetching {filename} ...", flush=True)
    path = hf_hub_download(
        repo_id="tldoan/iNF200",
        filename=filename,
        repo_type="dataset",
    )
    print(f"Extracting {path} -> {target}", flush=True)
    with tarfile.open(path) as tar:
        # Python 3.12+ supports filter=; older aprun images may not.
        try:
            tar.extractall(path=target, filter="data")
        except TypeError:
            tar.extractall(path=target)
    sub = target / expect_subdir
    if not sub.is_dir():
        raise FileNotFoundError(
            f"After extracting {filename}, expected {sub} "
            f"(got: {[p.name for p in target.iterdir()]})"
        )
    n = sum(1 for p in sub.iterdir() if p.is_dir())
    print(f"OK {expect_subdir}: {n} class folders", flush=True)

stage("iNF200_train.tar", "train_mini")
stage("iNF200_val.tar", "val")
print("Done:", target)
PY

echo "Done: ${TARGET}"
ls "${TARGET}"
echo "Class counts:"
echo -n "  train_mini "; find "${TARGET}/train_mini" -mindepth 1 -maxdepth 1 -type d | wc -l
echo -n "  val        "; find "${TARGET}/val" -mindepth 1 -maxdepth 1 -type d | wc -l
echo
echo "Next: bash scripts/check_frost_image_data.sh --protocol inf200"
