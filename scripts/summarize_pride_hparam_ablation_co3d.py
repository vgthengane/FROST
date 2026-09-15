#!/usr/bin/env python3
"""Summarize one-factor CO3D PRidE hyperparameter ablations."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from summarize_fscil_results import parse_fscil_log

REPO = Path(__file__).resolve().parents[1]
EXPECTED_SESSIONS = 6
def _new_log(root: Path, variant: str) -> Path:
    """Return the expected flat log path for a new CO3D sweep variant."""
    return root / f"pride_co3d_hp_{variant}_seed=2008.log"


def _run(path: Path) -> dict | None:
    """Parse one complete six-session result."""
    if not path.is_file():
        return None
    parsed = parse_fscil_log(path, EXPECTED_SESSIONS)
    return parsed if parsed and parsed["complete"] else None


def _fmt(value: float | None) -> str:
    return "—" if value is None else f"{value:.2f}"


def _table(title: str, parameter: str, rows: list[tuple[str, Path]]) -> list[str]:
    lines = [
        f"## {title}",
        "",
        f"| {parameter} | Curve | AA ↑ | Last ↑ | Δ ↓ | Novel ↑ | Harm ↑ |",
        "|---:|-------|------|--------|-----|---------|--------|",
    ]
    for label, path in rows:
        run = _run(path)
        if run is None:
            lines.append(f"| {label} | — | — | — | — | — | — |")
            continue
        curve = " ".join(f"{value:.1f}" for value in run["curve"])
        lines.append(
            f"| {label} | {curve} | {_fmt(run['aa'])} | {_fmt(run['last'])} | "
            f"{_fmt(run['delta'])} | {_fmt(run['novel_last'])} | "
            f"{_fmt(run['harmonic_last'])} |"
        )
    lines.append("")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--log-root",
        type=Path,
        default=REPO / "_experiments" / "00_PRIDE_HPARAM_ABLATION_CO3D",
    )
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    alpha_rows = [
        ("0.1", _new_log(args.log_root, "alpha01")),
        ("0.3", _new_log(args.log_root, "alpha03")),
        ("0.5 (default)", _new_log(args.log_root, "alpha05")),
        ("0.7", _new_log(args.log_root, "alpha07")),
        ("0.9", _new_log(args.log_root, "alpha09")),
    ]
    default_log = _new_log(args.log_root, "alpha05")
    component_rows = [
        ("Linear", _new_log(args.log_root, "comp_linear")),
        ("Adapter + Linear", _new_log(args.log_root, "comp_adapter_linear")),
        ("Ridge only", _new_log(args.log_root, "comp_ridge")),
        ("Cache only", _new_log(args.log_root, "comp_cache")),
        ("Ridge + Cache (default)", default_log),
    ]
    prototype_rows = [
        ("Mean", _new_log(args.log_root, "proto_mean")),
        ("Mean + two close (default)", default_log),
        ("Mean + close + far", _new_log(args.log_root, "proto_close_far")),
        ("Mean + two far", _new_log(args.log_root, "proto_mean_far2")),
        ("All five shots", _new_log(args.log_root, "proto_all")),
    ]
    adapter_rows = [
        ("32", _new_log(args.log_root, "adapter32")),
        ("64 (default)", default_log),
        ("128", _new_log(args.log_root, "adapter128")),
        ("256", _new_log(args.log_root, "adapter256")),
    ]
    ridge_rows = [
        ("250", _new_log(args.log_root, "ridge250")),
        ("500", _new_log(args.log_root, "ridge500")),
        ("1000 (default)", default_log),
        ("2000", _new_log(args.log_root, "ridge2000")),
        ("4000", _new_log(args.log_root, "ridge4000")),
    ]

    lines = [
        "# PRidE hyperparameter ablations — FSCIL CO3D",
        "",
        "Protocol: 25 base + 5 incremental 5-shot sessions; seed 2008; "
        "buffer-free; base-only adapter EP10. Each section changes one setting "
        "from locked PRidE.",
        "",
    ]
    lines.extend(_table("Components", "Component", component_rows))
    lines.extend(_table("Fusion weight", "α", alpha_rows))
    lines.extend(
        _table("Prototype insertion", "Novel-class prototypes", prototype_rows)
    )
    lines.extend(_table("Adapter bottleneck width", "Width", adapter_rows))
    lines.extend(_table("Ridge random-projection dimension", "Dimension", ridge_rows))

    text = "\n".join(lines).rstrip() + "\n"
    print(text)
    out_path = args.out or (args.log_root / "hparam_ablation.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
