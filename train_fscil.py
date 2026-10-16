"""
3D FSCIL entry point (FROST + baselines).

Uses:
  - repo-root ``utils/inc_net.py``  — Uni3D backbones, SimpleVitNet, …
  - ``lamba_pilot/``                — trainer, data manager
  - ``models/``                     — FROST and FSCIL learners
  - ``backbone/``                   — Uni3D backbone implementations
  - ``datasets/``                   — point-cloud FSCIL loaders

Run from the repository root:

    aprun python train_fscil.py --config configs/frost_pc_in_mn40.json
    sbatch scripts/reproduce_indomain.sh
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

# Config loader lives under repo-root utils/; import after path setup in main.

# Standard ModelNet40 baseline seeds (1993 is the LAMDA-PILOT default).
MODELNET40_SEEDS = [1993, 1994, 1995, 1996, 1997]

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

TOY_DEFAULTS = {
    "prefix": "toy_mn40",
    "max_classes": 20,
    "max_train_per_class": 25,
    "max_test_per_class": 10,
    "tuned_epoch": 3,
    "crct_epochs": 3,
    "batch_size": 8,
    "num_workers": 4,
    "epochs": 3,
    "ca_epochs": 2,
    "init_epochs": 3,
    "later_epochs": 3,
    "save_checkpoint": False,
}


def _ensure_repo_utils_package() -> None:
    """Make ``utils`` the repo-root package (not ``lamba_pilot/utils``)."""
    import types

    utils_dir = REPO_ROOT / "utils"
    init_py = utils_dir / "__init__.py"
    pkg = types.ModuleType("utils")
    pkg.__file__ = str(init_py)
    pkg.__path__ = [str(utils_dir)]  # type: ignore[attr-defined]
    sys.modules["utils"] = pkg


def _preload_root_inc_net() -> None:
    """Register repo-root utils/inc_net.py before lamba_pilot/utils shadows it."""
    _ensure_repo_utils_package()
    spec = importlib.util.spec_from_file_location(
        "utils.inc_net", ROOT_INC_NET, submodule_search_locations=[]
    )
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


def _apply_toy_overrides(args: dict) -> None:
    """Medium-scale overnight toy run: 5 tasks, 20 classes, single seed."""
    args.update(TOY_DEFAULTS)
    if isinstance(args.get("seed"), list) and len(args["seed"]) > 1:
        args["seed"] = [args["seed"][0]]
    args["save_checkpoint"] = False
    print("Toy overrides: {}".format(TOY_DEFAULTS))


def _parse_seed_list(raw: str) -> list[int]:
    """Parse comma-separated seeds, e.g. ``1994,1995,1996,1997``."""
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def _normalize_seed_list(raw) -> list[int]:
    if raw is None:
        return []
    if isinstance(raw, int):
        return [raw]
    if isinstance(raw, list):
        return [int(s) for s in raw]
    raise TypeError(f"seed must be int or list[int], got {type(raw)!r}")


def _apply_seed_selection(
    args: dict,
    seeds: str | None,
    skip_seeds: str | None,
    *,
    fullseeds: bool = False,
) -> None:
    """Select seeds to run.

    Config may list the full paper set. By default only the **first** seed runs
    unless ``--fullseeds`` or ``--seeds`` is passed.
    """
    configured = _normalize_seed_list(args.get("seed"))
    if seeds:
        args["seed"] = _parse_seed_list(seeds)
    elif fullseeds:
        if not configured:
            raise ValueError("--fullseeds requires a non-empty seed list in the config")
        args["seed"] = configured
    elif len(configured) > 1:
        args["seed"] = [configured[0]]
        print(
            "Using first seed only ({}); pass --fullseeds for {}".format(
                configured[0], configured
            )
        )
    else:
        args["seed"] = configured

    if skip_seeds:
        skip = set(_parse_seed_list(skip_seeds))
        args["seed"] = [seed for seed in args["seed"] if seed not in skip]

    if not args.get("seed"):
        raise ValueError("No seeds left to run after --seeds / --skip-seeds filtering")

    print("Seeds to run: {}".format(args["seed"]))


def load_json(path: str | Path) -> dict:
    _ensure_repo_utils_package()
    from utils.config_load import load_json as _load

    return _load(path)


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
        help="Per-method log directory (e.g. _experiments/full_runs_5seeds/simplecil).",
    )
    parser.add_argument(
        "--method-name",
        type=str,
        default=None,
        help="Short method label for flat logs: {method}_seed={seed}.log",
    )
    parser.add_argument(
        "--log-layout",
        type=str,
        choices=("flat", "nested"),
        default=None,
        help="Log path layout. Default: flat under full_runs_5seeds / toy_runs_5seeds.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Short dry-run with built-in smoke-test overrides.",
    )
    parser.add_argument(
        "--toy",
        action="store_true",
        help="Medium overnight toy run (20 classes, 5 tasks, seed 1993 only).",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        metavar="S1,S2,...",
        help="Override config seeds (comma-separated). Example: --seeds 1994,1995,1996,1997",
    )
    parser.add_argument(
        "--fullseeds",
        action="store_true",
        help="Run every seed listed in the config (default: first seed only).",
    )
    parser.add_argument(
        "--skip-seeds",
        type=str,
        default=None,
        metavar="S1,S2,...",
        help="Skip seeds already completed. Example: --skip-seeds 1993",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the latest per-task checkpoint under the run log directory.",
    )
    parser.add_argument(
        "--save-checkpoint",
        action="store_true",
        help="Save a checkpoint after each task (default: off).",
    )
    parser.add_argument(
        "--warm-start-checkpoint",
        type=str,
        default=None,
        help="Load task-0 (or mid-run) checkpoint and continue from the next task.",
    )
    parser.add_argument(
        "--start-task",
        type=int,
        default=None,
        help="First task index to run (default: 0, or cur_task+1 after warm-start).",
    )
    parser.add_argument(
        "--max-task",
        type=int,
        default=None,
        help="Stop after this task index (inclusive). Useful for task-0-only warmstart.",
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
        # If the config was moved to an archive, try the archived location.
        config_name = Path(cli.config).name
        config_path = (REPO_ROOT / "configs" / config_name).resolve()
    if not config_path.exists():
        archive_path = (REPO_ROOT / "_archive" / "configs" / "cil" / Path(cli.config).name).resolve()
        if archive_path.exists():
            config_path = archive_path
        else:
            raise FileNotFoundError(f"Config not found: {cli.config}")

    args = load_json(config_path)
    if cli.smoke:
        _apply_smoke_overrides(args)
    if cli.toy:
        _apply_toy_overrides(args)
    if cli.set:
        _apply_overrides(args, cli.set)
    if args.get("fscil_shot_protocol") and args.pop("max_train_per_class", None) is not None:
        print(
            "fscil_shot_protocol: dropped global max_train_per_class before data load "
            "(task 0 = full data, tasks 1+ = increment_train_per_class)"
        )
    if "save_checkpoint" not in args:
        args["save_checkpoint"] = False
    _apply_seed_selection(
        args, cli.seeds, cli.skip_seeds, fullseeds=bool(cli.fullseeds)
    )
    if cli.save_checkpoint:
        args["save_checkpoint"] = True
    if cli.resume:
        args["resume"] = True
        # FSCIL: task-0 warmup lives under task0_warmup_seed=*; do not save tasks 1+.
        if not (args.get("fscil_shot_protocol") or args.get("fscil3d_protocol")):
            args["save_checkpoint"] = True
    if cli.warm_start_checkpoint:
        path = Path(cli.warm_start_checkpoint)
        if not path.is_absolute():
            path = (REPO_ROOT / path).resolve()
        args["warm_start_checkpoint"] = str(path)
    if cli.start_task is not None:
        args["start_task"] = cli.start_task
    if cli.max_task is not None:
        args["max_task"] = cli.max_task
    if cli.logdir:
        args["log_dir"] = cli.logdir
    _resolve_paths(args)
    if cli.method_name:
        args["method_name"] = cli.method_name
    elif cli.logdir:
        args["method_name"] = Path(cli.logdir).name
    if cli.log_layout:
        args["log_layout"] = cli.log_layout
    elif "log_dir" in args and any(
        token in str(args["log_dir"])
        for token in (
            "full_runs_5seeds",
            "full_runs_5seeds_sn53",
            "data_efficiency_5seeds",
            "toy_runs_5seeds",
        )
    ):
        args["log_layout"] = "flat"
    if "log_dir" not in args:
        dataset = args.get("dataset", "modelnet40").lower()
        args["log_dir"] = str(
            (REPO_ROOT / "_experiments" / dataset / "methods").resolve()
        )
        args.setdefault("log_layout", "nested")

    from trainer import train

    train(args)


if __name__ == "__main__":
    main()
