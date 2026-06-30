"""
3D class-incremental learning entry point.

Uses:
  - repo-root ``utils/inc_net.py``  — Uni3D backbones, MOSNet, SimpleVitNet, …
  - ``lamba_pilot/``                — trainer, data manager
  - ``models/``                     — 3D continual-learning learners
  - ``backbone/``                   — Uni3D backbone implementations
  - ``datasets/``                   — ModelNet40 point-cloud data loaders

Run from the repository root:

    aprun python train_cil.py --config configs/mos_modelnet40.json
    bash scripts/train_cil.sh simplecil mos
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
LAMBA_ROOT = REPO_ROOT / "lamba_pilot"
ROOT_INC_NET = REPO_ROOT / "utils" / "inc_net.py"

SMOKE_ENV_KEYS = {
    "prefix": "CIL_SMOKE_PREFIX",
    "max_classes": "CIL_SMOKE_MAX_CLASSES",
    "max_train_per_class": "CIL_SMOKE_MAX_TRAIN_PER_CLASS",
    "max_test_per_class": "CIL_SMOKE_MAX_TEST_PER_CLASS",
    "tuned_epoch": "CIL_SMOKE_TUNED_EPOCH",
    "crct_epochs": "CIL_SMOKE_CRCT_EPOCHS",
    "batch_size": "CIL_SMOKE_BATCH_SIZE",
    "num_workers": "CIL_SMOKE_NUM_WORKERS",
}

SMOKE_DEFAULTS = {
    "prefix": "smoke_mn40",
    "max_classes": 8,
    "max_train_per_class": 10,
    "max_test_per_class": 5,
    "tuned_epoch": 2,
    "crct_epochs": 2,
    "batch_size": 4,
    "num_workers": 2,
    "epochs": 2,
    "ca_epochs": 1,
    "init_epochs": 2,
    "later_epochs": 2,
}


def _preload_root_inc_net() -> None:
    """Register repo-root utils/inc_net.py before lamba_pilot/utils shadows it."""
    spec = importlib.util.spec_from_file_location("utils.inc_net", ROOT_INC_NET)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    sys.modules["utils.inc_net"] = module


def _setup_paths() -> None:
    sys.path.insert(0, str(LAMBA_ROOT))
    sys.path.insert(0, str(REPO_ROOT))
    os.chdir(LAMBA_ROOT)
    if sys.path and sys.path[0] in ("", "."):
        sys.path.pop(0)
    sys.path.insert(0, str(REPO_ROOT))
    _preload_root_inc_net()


def _resolve_paths(args: dict) -> None:
    for key in ("data_path", "pretrained_pc", "log_dir"):
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
    """Parse ``key=value`` override values from the CLI."""
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


def _apply_smoke_overrides(args: dict) -> None:
    """Apply smoke overrides from CIL_SMOKE_* env vars or built-in defaults."""
    applied = []
    for arg_key, env_key in SMOKE_ENV_KEYS.items():
        value = os.environ.get(env_key)
        if value is None:
            continue
        args[arg_key] = _parse_override(value)
        applied.append(arg_key)
    if not applied:
        args.update(SMOKE_DEFAULTS)
        applied = list(SMOKE_DEFAULTS.keys())
    args["save_checkpoint"] = False
    print("Smoke overrides: {}".format({k: args[k] for k in applied if k in args}))


def load_json(path: str | Path) -> dict:
    with open(path) as f:
        return json.load(f)


def parse_args():
    parser = argparse.ArgumentParser(description="3D continual learning on ModelNet40")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Experiment JSON under configs/",
    )
    parser.add_argument(
        "--logdir",
        type=str,
        default=None,
        help="Experiment log directory (default: _experiments/<dataset>/methods).",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Short dry-run with built-in smoke-test overrides.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the latest per-task checkpoint under the run log directory.",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a config field (repeatable). Example: --set tuned_epoch=2",
    )
    return parser.parse_args()


def main():
    cli = parse_args()
    _setup_paths()

    config_path = Path(cli.config)
    if not config_path.is_absolute():
        config_path = (REPO_ROOT / config_path).resolve()
    if not config_path.exists():
        config_path = (REPO_ROOT / "configs" / Path(cli.config).name).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {cli.config}")

    args = load_json(config_path)
    if cli.smoke:
        _apply_smoke_overrides(args)
    if cli.set:
        _apply_overrides(args, cli.set)
    if cli.resume:
        args["resume"] = True
        args["save_checkpoint"] = True
    if cli.logdir:
        args["log_dir"] = cli.logdir
    _resolve_paths(args)
    if "log_dir" not in args:
        dataset = args.get("dataset", "modelnet40").lower()
        args["log_dir"] = str(
            (REPO_ROOT / "_experiments" / dataset / "methods").resolve()
        )

    from trainer import train

    train(args)


if __name__ == "__main__":
    main()
