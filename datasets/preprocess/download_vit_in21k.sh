#!/usr/bin/env bash
# Stage ViT-B/16 ImageNet-21k weights for offline Leonardo compute nodes.
#
# Prefer copying from a local Hugging Face hub cache; otherwise download once
# on a networked machine (login node / workstation), then sync to Leonardo:
#   bash sync.sh --folder=_experiments/pretrained_weights/ViT
#
# Usage:
#   bash datasets/preprocess/download_vit_in21k.sh
#   bash datasets/preprocess/download_vit_in21k.sh /path/to/_experiments/pretrained_weights/ViT
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEST="${1:-${REPO}/_experiments/pretrained_weights/ViT}"
mkdir -p "${DEST}"
DEST="$(cd "${DEST}" && pwd)"

IN21K_NAME="vit_base_patch16_224.augreg_in21k.safetensors"
IN1K_NAME="vit_base_patch16_224.augreg2_in21k_ft_in1k.safetensors"
IN21K_DEST="${DEST}/${IN21K_NAME}"
IN1K_DEST="${DEST}/${IN1K_NAME}"

HF_HUB="${HOME}/.cache/huggingface/hub"

copy_from_hf_cache() {
  local repo_id="$1"
  local dest="$2"
  local snap
  snap="$(ls -1d "${HF_HUB}/${repo_id}/snapshots/"* 2>/dev/null | tail -n 1 || true)"
  if [[ -n "${snap}" && -f "${snap}/model.safetensors" ]]; then
    echo "Copying ${repo_id} -> ${dest}"
    cp -f "${snap}/model.safetensors" "${dest}"
    return 0
  fi
  return 1
}

download_hf() {
  local repo="$1"
  local dest="$2"
  echo "Downloading ${repo} via huggingface_hub -> ${dest}"
  aprun python - <<PY
from huggingface_hub import hf_hub_download
import shutil
path = hf_hub_download(repo_id="${repo}", filename="model.safetensors")
shutil.copy(path, "${dest}")
print("Wrote", "${dest}")
PY
}

if [[ -f "${IN21K_DEST}" ]]; then
  echo "Already present: ${IN21K_DEST}"
else
  copy_from_hf_cache "models--timm--vit_base_patch16_224.augreg_in21k" "${IN21K_DEST}" \
    || download_hf "timm/vit_base_patch16_224.augreg_in21k" "${IN21K_DEST}"
fi

if [[ -f "${IN1K_DEST}" ]]; then
  echo "Already present: ${IN1K_DEST}"
else
  copy_from_hf_cache "models--timm--vit_base_patch16_224.augreg2_in21k_ft_in1k" "${IN1K_DEST}" \
    || download_hf "timm/vit_base_patch16_224.augreg2_in21k_ft_in1k" "${IN1K_DEST}"
fi

ls -lh "${DEST}"
echo "Done. On Leonardo sync with:"
echo "  bash sync.sh --folder=_experiments/pretrained_weights/ViT"
