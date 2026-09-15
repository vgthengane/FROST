#!/usr/bin/env python3
"""Summarize FSCIL MN40 ablation logs into module + hyperparameter tables."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from summarize_fscil_results import discover_logs, parse_fscil_log

REPO = Path(__file__).resolve().parents[1]
EXPECTED_SESSIONS = 5

MODULE_ROWS = [
    ("seel_mn40_abl_full", "Full SIMPL", "ridge_cache + rules + replay + gated"),
    ("seel_mn40_abl_ridge", "Ridge only", "classifier_mode=ridge"),
    ("seel_mn40_abl_cache", "Cache only", "classifier_mode=cache"),
    ("seel_mn40_abl_static", "w/o evolution", "evolution_policy=static"),
    ("seel_mn40_abl_noreplay", "w/o conflict replay", "conflict_replay=false"),
    ("seel_mn40_abl_nogated", "w/o gated growth", "enable_gated_growth=false"),
]

HYPER_ROWS = [
    ("seel_mn40_abl_full", "Full (α=0.85, B=200, P=3/8)", "baseline"),
    ("seel_mn40_abl_a05", "α=0.5", "ridge_cache_alpha"),
    ("seel_mn40_abl_a07", "α=0.7", "ridge_cache_alpha"),
    ("seel_mn40_abl_a10", "α=1.0", "ridge-only fusion"),
    ("seel_mn40_abl_B100", "B=100", "cache_budget"),
    ("seel_mn40_abl_B300", "B=300", "cache_budget"),
    ("seel_mn40_abl_p33", "P=3 fixed", "max_prototypes + incr cap"),
    ("seel_mn40_abl_p35", "P=3 base, P=5 incr", "fscil_increment_max_prototypes"),
    ("seel_mn40_abl_pshot", "Per-shot, P≤5", "fscil_shot_prototypes"),
]


def _best_run(root: Path, method: str) -> dict | None:
    for _seed, path in sorted(discover_logs(root).get(method, []), key=lambda x: x[0]):
        parsed = parse_fscil_log(path, EXPECTED_SESSIONS)
        if parsed and parsed["complete"]:
            return parsed
    for _seed, path in sorted(discover_logs(root).get(method, []), key=lambda x: x[0]):
        parsed = parse_fscil_log(path, EXPECTED_SESSIONS)
        if parsed:
            return parsed
    return None


def _fmt_metric(value: float | None, suffix: str = "") -> str:
    if value is None:
        return "—"
    return f"{value:.2f}{suffix}"


def _delta_vs_baseline(row: dict | None, baseline: dict | None) -> str:
    if row is None or baseline is None or row.get("last") is None or baseline.get("last") is None:
        return "—"
    return f"{row['last'] - baseline['last']:+.2f}"


def build_table(title: str, rows: list[tuple[str, str, str]], root: Path, baseline: dict | None) -> str:
    lines = [f"## {title}\n"]
    lines.append("| Setting | Config | AA ↑ | Last ↑ | Δ ↓ | ΔLast vs full |")
    lines.append("|---------|--------|------|--------|-----|---------------|")
    for method, label, note in rows:
        run = _best_run(root, method)
        if run is None:
            lines.append(f"| {label} | {note} | — | — | — | — |")
            continue
        lines.append(
            f"| {label} | {note} | "
            f"{_fmt_metric(run.get('aa'))} | {_fmt_metric(run.get('last'))} | "
            f"{_fmt_metric(run.get('delta'))} | {_delta_vs_baseline(run, baseline)} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize FSCIL MN40 ablation tables")
    parser.add_argument(
        "--log-root",
        type=Path,
        default=REPO / "_experiments" / "00_SIMPL_ABLATIONS_MN40",
    )
    parser.add_argument("--out", type=Path, default="")
    args = parser.parse_args()

    root = args.log_root
    baseline = _best_run(root, "seel_mn40_abl_full")

    text_parts = [
        f"# FSCIL MN40 ablations — `{root.name}`\n",
        "Protocol: FSCIL-3D ModelNet40 (5 sessions, 5-shot), seed 2008.\n",
        build_table("Table A — Module ablations", MODULE_ROWS, root, baseline),
        build_table("Table B — Hyperparameter ablations", HYPER_ROWS, root, baseline),
    ]
    if baseline:
        text_parts.append(
            f"*Baseline (full): AA={baseline['aa']:.2f}, Last={baseline['last']:.2f}, "
            f"Δ={baseline['delta']:.2f}*\n"
        )

    text = "\n".join(text_parts)
    print(text)
    out_path = args.out or (root / "ablation_tables.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
