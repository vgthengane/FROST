#!/usr/bin/env python3
"""Summarize the controlled MN40 PRidE component ablation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from summarize_fscil_results import discover_logs, parse_fscil_log

REPO = Path(__file__).resolve().parents[1]
EXPECTED_SESSIONS = 5
ROWS = [
    ("pride_mn40_comp_linear", "Linear", "Frozen Uni3D; CE head every session"),
    (
        "pride_mn40_comp_adapter_linear",
        "Adapter + Linear",
        "Base-only adapter; CE head every session",
    ),
    (
        "pride_mn40_comp_ridge",
        "Ridge only",
        "Base-only adapter; analytical ridge",
    ),
    (
        "pride_mn40_comp_cache",
        "Cache only",
        "Base-only adapter; static mean+two-close cache",
    ),
    (
        "pride_mn40_comp_ridge_cache",
        "Ridge + Cache",
        "Locked final PRidE; flat α=0.5",
    ),
]


def _run_for_method(root: Path, method: str) -> dict | None:
    """Return the complete seed-2008 run for one ablation row."""
    candidates = discover_logs(root).get(method, [])
    for seed, path in candidates:
        if seed != 2008:
            continue
        parsed = parse_fscil_log(path, EXPECTED_SESSIONS)
        if parsed and parsed["complete"]:
            return parsed
    return None


def _fmt(value: float | None) -> str:
    return "—" if value is None else f"{value:.2f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--log-root",
        type=Path,
        default=REPO / "_experiments" / "00_PRIDE_COMPONENT_ABLATION_MN40",
    )
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    lines = [
        "# PRidE component ablation — FSCIL ModelNet40",
        "",
        "Protocol: 20 base + 4×5-way 5-shot sessions; seed 2008; base adapter EP10.",
        "",
        "| Variant | Train/update path | Curve | AA ↑ | Last ↑ | Δ ↓ | Novel ↑ | Harm ↑ |",
        "|---------|-------------------|-------|------|--------|-----|---------|--------|",
    ]
    for method, label, description in ROWS:
        run = _run_for_method(args.log_root, method)
        if run is None:
            lines.append(f"| {label} | {description} | — | — | — | — | — | — |")
            continue
        curve = " ".join(f"{value:.1f}" for value in run["curve"])
        lines.append(
            f"| {label} | {description} | {curve} | {_fmt(run['aa'])} | "
            f"{_fmt(run['last'])} | {_fmt(run['delta'])} | "
            f"{_fmt(run['novel_last'])} | {_fmt(run['harmonic_last'])} |"
        )

    text = "\n".join(lines) + "\n"
    print(text)
    out_path = args.out or (args.log_root / "component_ablation.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
