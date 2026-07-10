#!/usr/bin/env python3
"""Smoke-test FSCIL MN40 ablation launcher (2 sessions, minimal warmup).

Runs representative variants (full, ridge-only, per-shot protos, buffer-free) with
``--max-task 1`` and 1 warmup epoch, then checks logs for a 2-point accuracy curve.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LOGROOT = REPO / "_experiments" / "00_SIMPL_ABLATIONS_MN40"
CONFIG = REPO / "configs" / "selfevo_fscil_mn40_5t_flex_compare.json"
CURVE_RE = re.compile(r"CNN top1 curve:\s*\[([^\]]+)\]")

SMOKE_VARIANTS = [
    ("seel_mn40_abl_full", "full baseline", []),
    (
        "seel_mn40_abl_ridge",
        "ridge only",
        ["--set", "classifier_mode=ridge", "--set", "evolution_policy=static"],
    ),
    (
        "seel_mn40_abl_pshot",
        "per-shot protos P≤5",
        [
            "--set",
            "fscil_shot_prototypes=true",
            "--set",
            "fscil_increment_max_prototypes=5",
        ],
    ),
    (
        "seel_mn40_abl_bufferfree",
        "buffer-free selfevo (no old-class disk re-read; cache conflict replay)",
        [
            "--set",
            "model_name=selfevo_bufferfree",
            "--set",
            "evolution_policy=static",
        ],
    ),
]

SMOKE_FLAGS = [
    "--max-task",
    "1",
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


def _train_runner() -> list[str]:
    """Train subprocess uses the current interpreter (invoke smoke via aprun once)."""
    return [sys.executable]


def run_variant(method: str, label: str, extra: list[str]) -> Path:
    log_path = LOGROOT / f"{method}_smoke_seed=2008.log"
    cmd = [
        *_train_runner(),
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
    print(f"\n--- smoke: {method} ({label}) ---")
    print(" ".join(cmd))
    LOGROOT.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log_f:
        proc = subprocess.run(
            cmd,
            cwd=REPO,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if proc.returncode != 0:
        raise RuntimeError(f"{method} failed with exit code {proc.returncode}")
    return log_path


def check_log(path: Path) -> tuple[int, list[float]]:
    text = path.read_text(errors="replace")
    curves = CURVE_RE.findall(text)
    if not curves:
        raise RuntimeError(f"No CNN top1 curve in {path}")
    nums = [float(x.strip()) for x in curves[-1].split(",") if x.strip()]
    if len(nums) < 2:
        raise RuntimeError(f"Expected ≥2 sessions in curve, got {nums} ({path})")
    return len(nums), nums


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test FSCIL MN40 ablations")
    parser.add_argument(
        "--variant",
        choices=("all", "full", "ridge", "pshot", "bufferfree"),
        default="all",
        help="Run one variant or all three (default: all)",
    )
    args = parser.parse_args()

    selected = SMOKE_VARIANTS
    if args.variant != "all":
        key = {"full": 0, "ridge": 1, "pshot": 2, "bufferfree": 3}[args.variant]
        selected = [SMOKE_VARIANTS[key]]

    print("=" * 72)
    print("FSCIL MN40 ablation smoke test")
    print("=" * 72)

    for method, label, extra in selected:
        log_path = run_variant(method, label, extra)
        n_sess, curve = check_log(log_path)
        print(f"OK {method}: {n_sess} sessions, curve={curve}")

    print("\nAll smoke checks passed.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"SMOKE FAILED: {exc}", file=sys.stderr)
        sys.exit(1)
