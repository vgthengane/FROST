"""Offline-friendly ViT weight loading for Leonardo compute nodes.

Compute partitions typically have no outbound network, so ``timm.create_model(...,
pretrained=True)`` fails when it tries to reach Hugging Face. Prefer a vendored
checkpoint under ``_experiments/pretrained_weights/ViT/`` (same pattern as Uni3D).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import timm
import torch

# Repo root: lamba_pilot/backbone/this_file.py → parents[2]
_REPO_ROOT = Path(__file__).resolve().parents[2]
_VIT_WEIGHT_DIR = _REPO_ROOT / "_experiments" / "pretrained_weights" / "ViT"

# Canonical filenames (timm tags for ViT-B/16).
_CKPT_NAMES = {
    "vit_base_patch16_224_in21k": "vit_base_patch16_224.augreg_in21k.safetensors",
    "vit_base_patch16_224": "vit_base_patch16_224.augreg2_in21k_ft_in1k.safetensors",
}

# Hugging Face hub cache layout (login nodes / interactive).
_HF_REPO_IDS = {
    "vit_base_patch16_224_in21k": "models--timm--vit_base_patch16_224.augreg_in21k",
    "vit_base_patch16_224": "models--timm--vit_base_patch16_224.augreg2_in21k_ft_in1k",
}


def resolve_vit_checkpoint(timm_name: str) -> Optional[Path]:
    """Return a local checkpoint path, or None if none is available."""
    env = os.environ.get("PRIDE_VIT_CKPT") or os.environ.get("SELFECO_VIT_CKPT")
    if env:
        path = Path(env).expanduser()
        if path.is_file():
            return path.resolve()

    fname = _CKPT_NAMES.get(timm_name)
    if fname is not None:
        candidate = _VIT_WEIGHT_DIR / fname
        if candidate.is_file():
            return candidate.resolve()

    hf_id = _HF_REPO_IDS.get(timm_name)
    if hf_id is not None:
        hub = Path.home() / ".cache" / "huggingface" / "hub" / hf_id / "snapshots"
        if hub.is_dir():
            matches = sorted(hub.glob("*/model.safetensors"))
            if matches:
                return matches[-1].resolve()

    return None


def _load_state_dict(path: Path) -> dict:
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        return load_file(str(path))
    obj = torch.load(str(path), map_location="cpu")
    if isinstance(obj, dict):
        for key in ("state_dict", "model", "model_state"):
            if key in obj and isinstance(obj[key], dict):
                return obj[key]
    if not isinstance(obj, dict):
        raise ValueError(f"Unsupported checkpoint format at {path}")
    return obj


def create_timm_vit(timm_name: str, num_classes: int = 0):
    """Create a timm ViT, loading weights from disk when possible (no HF needed)."""
    ckpt = resolve_vit_checkpoint(timm_name)
    if ckpt is not None:
        logging.info("Loading ViT weights offline from %s", ckpt)
        model = timm.create_model(timm_name, pretrained=False, num_classes=num_classes)
        state = _load_state_dict(ckpt)
        incompatible = model.load_state_dict(state, strict=False)
        logging.info(
            "ViT load: missing=%d unexpected=%d",
            len(incompatible.missing_keys),
            len(incompatible.unexpected_keys),
        )
        return model

    logging.warning(
        "No local ViT checkpoint for %s under %s (or PRIDE_VIT_CKPT). "
        "Falling back to HuggingFace download — this fails on offline compute nodes. "
        "Run: bash datasets/preprocess/download_vit_in21k.sh && sync _experiments/pretrained_weights/ViT",
        timm_name,
        _VIT_WEIGHT_DIR,
    )
    return timm.create_model(timm_name, pretrained=True, num_classes=num_classes)
