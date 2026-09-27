"""
Image FSCIL entry point (LGSP dataset protocol + LAMDA-PILOT ViT PTM).

Uses:
  - ``lamba_pilot/utils/inc_net.py`` — timm ViT backbones (NOT Uni3D)
  - ``lamba_pilot/``                — trainer, data manager
  - ``models/selfevo.py``           — PRidE / SelfEvo method
  - ``datasets/lgsp_fscil.py``      — CUB / Aircraft / iNF200 + LGSP index_list

Run from the repository root:

    aprun python train_fscil_image.py --config configs/pride_image_cub200_buffree.json
    sbatch scripts/simpl_runs_pride_image_alpha05_ep10.sh
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
    for key in ("data_path", "log_dir", "lgsp_index_root", "lgsp_data_root"):
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
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Image FSCIL with SelfEvo/PRidE on LGSP dataset splits."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="JSON config (e.g. configs/pride_image_cub200_buffree.json)",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Parent folder containing CUB_200_2011 / fgvc-aircraft-2013b / iNF200",
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
    if cli.seeds is not None:
        args["seed"] = _parse_seed_list(cli.seeds)
    elif cli.seed is not None:
        args["seed"] = cli.seed
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
