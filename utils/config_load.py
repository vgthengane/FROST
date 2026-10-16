"""JSON experiment config loader with optional ``_base`` inheritance.

Child configs may set::

    {"_base": "base/pclouds.json", ...only overrides...}

``_base`` is resolved relative to the child file's directory (usually
``configs/``). Nested bases are merged depth-first; child keys win.

Paper component blocks (``adapter``, ``corrmem``, ``protomem``, ``fusion``)
are flattened to the internal FROST arg names after merge.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# Paper Sec. 3 names → internal training keys.
_FUSION_MODE = {
    "corr_proto": "ridge_cache",  # CorrMem + ProtoMem (default)
    "corrmem": "ridge",
    "protomem": "cache",
    # legacy aliases still accepted inside fusion.mode
    "ridge_cache": "ridge_cache",
    "ridge": "ridge",
    "cache": "cache",
}

_ADAPTER_MAP = {
    "ffn_num": "ffn_num",
    "base_only": "fscil_adapter_base_only",
    "task0_warmup": "task0_adapter_warmup",
    "increment_warmup": "increment_adapter_warmup",
    "epochs": "tuned_epoch",
    "lr": "task0_init_lr",
    "init_lr": "init_lr",
    "weight_decay": "weight_decay",
    "min_lr": "min_lr",
    "optimizer": "optimizer",
    "task0_optimizer": "task0_optimizer",
    "task0_weight_decay": "task0_weight_decay",
}

_CORRMEM_MAP = {
    "lift_v": "ridge_rp_dim",
}

# Paper ProtoMem: only ``insert`` is listed in bases; the rest comes from
# ``fscil_preset`` (see ``utils/fscil_ablation_config._PAPER_PROTOMEM``).
_PROTOMEM_MAP = {
    "insert": "insert_strategy",
    "budget": "cache_budget",  # optional override; else sized from num_category
}


def load_json(path: str | Path) -> dict[str, Any]:
    """Load a JSON config, recursively merging ``_base`` parents."""
    path = Path(path).expanduser().resolve()
    merged = _load_with_bases(path, stack=())
    return _finalize_config(merged)


def _load_with_bases(path: Path, stack: tuple[Path, ...]) -> dict[str, Any]:
    if path in stack:
        cycle = " -> ".join(str(p) for p in (*stack, path))
        raise ValueError(f"Config _base cycle: {cycle}")

    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise TypeError(f"Config must be a JSON object: {path}")

    base_ref = raw.pop("_base", None)
    merged: dict[str, Any] = {}
    if base_ref:
        refs = base_ref if isinstance(base_ref, list) else [base_ref]
        for ref in refs:
            base_path = Path(ref)
            if not base_path.is_absolute():
                base_path = (path.parent / base_path).resolve()
            parent = _load_with_bases(base_path, (*stack, path))
            merged.update(parent)

    merged.update(raw)
    return merged


def _finalize_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Flatten paper blocks and drop underscore meta keys (except kept internals)."""
    out = dict(cfg)
    _flatten_paper_blocks(out)
    for key in list(out):
        if key.startswith("_"):
            out.pop(key)
    return out


def _flatten_paper_blocks(cfg: dict[str, Any]) -> None:
    """Expand adapter / corrmem / protomem / fusion into flat training args.

    Flat keys already set by a child overlay (e.g. ``tuned_epoch``) win over
    values from paper blocks (``setdefault``).
    """
    adapter = cfg.pop("adapter", None)
    if isinstance(adapter, dict):
        for src, dst in _ADAPTER_MAP.items():
            if src in adapter:
                cfg.setdefault(dst, adapter[src])

    corrmem = cfg.pop("corrmem", None)
    if isinstance(corrmem, dict):
        for src, dst in _CORRMEM_MAP.items():
            if src in corrmem:
                cfg.setdefault(dst, corrmem[src])

    protomem = cfg.pop("protomem", None)
    if isinstance(protomem, dict):
        for src, dst in _PROTOMEM_MAP.items():
            if src in protomem:
                cfg.setdefault(dst, protomem[src])

    fusion = cfg.pop("fusion", None)
    if isinstance(fusion, dict):
        if "mode" in fusion:
            mode = str(fusion["mode"])
            cfg.setdefault("classifier_mode", _FUSION_MODE.get(mode, mode))
        if "alpha" in fusion:
            alpha = fusion["alpha"]
            cfg.setdefault("ridge_cache_alpha", alpha)
            cfg.setdefault("fscil_increment_ridge_alpha", alpha)
            cfg.setdefault("fscil_task1_ridge_alpha", alpha)
