#!/usr/bin/env python3
"""Summarize FSCIL text-modality runs (gamma sweep + text_v1)."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_LOGROOT = REPO / "_experiments" / "fscil_style_mn40_5t"

METHODS = [
    ("selfevo_fscil_v3_fewshot_train", "baseline ridge_cache"),
    ("selfevo_fscil_text_ablation", "text γ=0.10 (ablation)"),
    ("selfevo_fscil_text_g003", "text γ=0.03"),
    ("selfevo_fscil_text_g005", "text γ=0.05"),
    ("selfevo_fscil_text_g008", "text γ=0.08"),
    ("selfevo_fscil_text_v1", "text_v1 (tuned)"),
]


def parse_log(path: Path) -> dict | None:
    text = path.read_text(errors="replace")
    m_curve = re.findall(r"CNN top1 curve: (\[[^\]]+\])", text)
    m_avg = re.findall(r"Average Accuracy \(CNN\): ([0-9.]+)", text)
    if not m_curve:
        return None
    curve = m_curve[-1]
    nums = [float(x.strip()) for x in curve.strip("[]").split(",")]
    return {
        "curve": curve,
        "last": nums[-1] if nums else None,
        "avg": float(m_avg[-1]) if m_avg else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-root", type=Path, default=DEFAULT_LOGROOT)
    parser.add_argument("--seed", type=int, default=1993)
    args = parser.parse_args()

    rows = []
    for method, label in METHODS:
        log = args.log_root / f"{method}_seed={args.seed}.log"
        if not log.is_file():
            rows.append((label, "—", "—", "—", str(log.name)))
            continue
        stats = parse_log(log)
        if stats is None:
            rows.append((label, "incomplete", "—", "—", log.name))
            continue
        rows.append(
            (label, stats["curve"], f"{stats['last']:.2f}", f"{stats['avg']:.2f}", log.name)
        )

    print(f"FSCIL text tuning (seed {args.seed})")
    print(f"{'label':<28} {'last%':>7} {'avg%':>7}  curve")
    print("-" * 72)
    for label, curve, last, avg, _ in rows:
        if last == "—":
            print(f"{label:<28} {'—':>7} {'—':>7}  (missing)")
        else:
            print(f"{label:<28} {last:>7} {avg:>7}  {curve}")


if __name__ == "__main__":
    main()
