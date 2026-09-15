#!/usr/bin/env python3
"""Smoke-test harder-regime module ablation launcher (2 sessions, 1 epoch).

Runs one representative variant (mn40_1shot / full) to verify config + overrides.
Invoke via aprun once from repo root:

    aprun python scripts/smoke_test_fscil_regime_ablations.py
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LOGROOT = REPO / "_experiments" / "00_SIMPL_ABLATIONS_REGIMES"
CONFIG = REPO / "configs" / "selfevo_fscil_mn40_5t_flex_compare.json"
METHOD = "seel_mn40_1shot_full"
CURVE_RE = re.compile(r"CNN top1 curve:\s*\[([^\]]+)\]")

SMOKE_FLAGS = [
    "--max-task",
    "1",
    "--set",
    "increment_train_per_class=1",
    "--set",
    "fscil_increment_max_prototypes=1",
    "--set",
    "tuned_epoch=1",
    "--set",
    "increment_tuned_epoch=1",
    "--set",
    "fscil_task1_warmup_epochs=1",
    "--set",
    "fscil_last_task_warmup_epochs=1",
    "--set",
    "task0_save_warmup_ckpts=false",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant",
        choices=("full", "ridge", "cache", "static"),
        default="full",
    )
    args = parser.parse_args()

    extra: list[str] = []
    if args.variant == "ridge":
        extra += ["--set", "classifier_mode=ridge", "--set", "evolution_policy=static"]
    elif args.variant == "cache":
        extra += ["--set", "classifier_mode=cache"]
    elif args.variant == "static":
        extra += ["--set", "evolution_policy=static"]

    method = f"seel_mn40_1shot_{args.variant}_smoke"
    log_path = LOGROOT / f"{method}.log"
    LOGROOT.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(REPO / "train_fscil.py"),
        "--config",
        str(CONFIG),
        "--logdir",
        str(LOGROOT),
        "--method-name",
        method,
        "--log-layout",
        "flat",
        *extra,
        *SMOKE_FLAGS,
    ]
    print("Smoke:", " ".join(cmd))
    with log_path.open("w") as log_f:
        proc = subprocess.run(cmd, cwd=REPO, stdout=log_f, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        raise SystemExit(f"Smoke failed exit={proc.returncode}, see {log_path}")

    text = log_path.read_text(errors="replace")
    curves = CURVE_RE.findall(text)
    if not curves:
        raise SystemExit(f"No CNN top1 curve in {log_path}")
    nums = [float(x.strip()) for x in curves[-1].split(",") if x.strip()]
    if len(nums) < 2:
        raise SystemExit(f"Expected ≥2 sessions, got {nums}")
    print(f"OK {method}: {len(nums)} sessions, curve={nums}")


if __name__ == "__main__":
    main()
