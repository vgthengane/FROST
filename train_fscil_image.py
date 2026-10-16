"""
Image FSCIL entry point (LGSP dataset protocol + LAMDA-PILOT ViT PTM).

Uses:
  - ``lamba_pilot/utils/inc_net.py`` — timm ViT backbones (NOT Uni3D)
  - ``lamba_pilot/``                — trainer, data manager
  - ``models/selfevo.py`` / FROST  — image FSCIL method
  - ``datasets/lgsp_fscil.py``      — CUB / FGVC-Aircraft / iNF200 + ``datasets/session_images``

Run from the repository root:

    aprun python train_fscil_image.py --config configs/frost_img_cub200.json
    sbatch scripts/reproduce_image.sh
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
LAMBA_ROOT = REPO_ROOT / "lamba_pilot"


def _setup_paths() -> None:
    """Prefer lamba_pilot utils (ViT) over repo-root utils (Uni3D)."""
    for p in (str(REPO_ROOT), str(LAMBA_ROOT)):
        if p in sys.path:
            sys.path.remove(p)
    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(LAMBA_ROOT))
    os.chdir(LAMBA_ROOT)
    if sys.path and sys.path[0] in ("", "."):
        sys.path.pop(0)
    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(LAMBA_ROOT))


def _resolve_paths(args: dict) -> None:
    for key in (
        "data_path",
        "log_dir",
        "sessions_root",
        "session_index_root",
        "lgsp_index_root",  # legacy alias
        "lgsp_data_root",
    ):
        value = args.get(key)
        if not value:
            continue
        path = Path(value)
        if not path.is_absolute():
            path = (REPO_ROOT / path).resolve()
        args[key] = str(path)
        if key == "log_dir":
            path.mkdir(parents=True, exist_ok=True)


def _parse_override(raw: str):
    if raw.lower() == "true":
        return True
    if raw.lower() == "false":
        return False
    if raw.startswith("[") or raw.startswith("{"):
        return json.loads(raw)
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        return raw


def _apply_overrides(args: dict, overrides: list[str]) -> None:
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must be key=value, got: {item!r}")
        key, value = item.split("=", 1)
        args[key.strip()] = _parse_override(value.strip())


def _parse_seed_list(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def load_json(path: str | Path) -> dict:
    # Always load from repo-root utils/ (lamba_pilot/utils is on path for ViT).
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_frost_config_load", REPO_ROOT / "utils" / "config_load.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.load_json(path)


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Image FSCIL with FROST on LGSP dataset splits."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="JSON config (e.g. configs/frost_img_cub200.json)",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Parent folder containing CUB200 / FGVC-Aircraft / iNF200",
    )
    parser.add_argument("--device", type=str, nargs="+", default=None)
    parser.add_argument("--seed", type=int, nargs="+", default=None)
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Comma-separated seeds (overrides config), e.g. 2008",
    )
    parser.add_argument(
        "--fullseeds",
        action="store_true",
        help="Run every seed listed in the config (default: first seed only).",
    )
    parser.add_argument(
        "--logdir",
        type=str,
        default=None,
        help="Log directory (mapped to args['log_dir']).",
    )
    parser.add_argument("--method-name", type=str, default=None)
    parser.add_argument(
        "--log-layout",
        type=str,
        choices=["nested", "flat"],
        default=None,
    )
    parser.add_argument("--max-task", type=int, default=None)
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        help="Override config key=value (repeatable).",
    )
    return parser


def main() -> None:
    _setup_paths()
    from trainer import train  # noqa: WPS433

    cli = setup_parser().parse_args()
    config_path = Path(cli.config)
    if not config_path.is_absolute():
        config_path = (REPO_ROOT / config_path).resolve()
    args = load_json(config_path)
    args["config"] = str(config_path)
    args.setdefault("lgsp_fscil", True)
    args.setdefault("save_checkpoint", False)

    if cli.set:
        _apply_overrides(args, cli.set)
    if cli.data_path:
        args["data_path"] = cli.data_path
    if cli.device is not None:
        args["device"] = [int(d) if str(d).lstrip("-").isdigit() else d for d in cli.device]
    configured = args.get("seed")
    if isinstance(configured, int):
        configured = [configured]
    elif not isinstance(configured, list):
        configured = []
    else:
        configured = [int(s) for s in configured]

    if cli.seeds is not None:
        args["seed"] = _parse_seed_list(cli.seeds)
    elif cli.seed is not None:
        args["seed"] = list(cli.seed)
    elif cli.fullseeds:
        if not configured:
            raise ValueError("--fullseeds requires a non-empty seed list in the config")
        args["seed"] = configured
    elif len(configured) > 1:
        args["seed"] = [configured[0]]
        print(
            f"Using first seed only ({configured[0]}); "
            f"pass --fullseeds for {configured}"
        )
    else:
        args["seed"] = configured

    if not args.get("seed"):
        raise ValueError("No seeds left to run")
    print(f"Seeds to run: {args['seed']}")

    if cli.logdir is not None:
        args["log_dir"] = cli.logdir
    if cli.method_name is not None:
        args["method_name"] = cli.method_name
    if cli.log_layout is not None:
        args["log_layout"] = cli.log_layout
    if cli.max_task is not None:
        args["max_task"] = cli.max_task

    if "log_dir" not in args:
        args["log_dir"] = str(
            (REPO_ROOT / "_experiments" / "image_fscil" / args.get("dataset", "cub200")).resolve()
        )

    _resolve_paths(args)
    train(args)


if __name__ == "__main__":
    main()
