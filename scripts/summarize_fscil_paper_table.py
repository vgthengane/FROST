#!/usr/bin/env python3
"""Build a side-by-side FSCIL comparison table: SEEL vs 3D-FLEG (paper reference).

Reads flat logs from the FSCIL-3D class-order campaigns and writes a single markdown
table under ``_experiments/fscil_style/results_table.md``.

Usage::

    aprun python scripts/summarize_fscil_paper_table.py
    aprun python scripts/summarize_fscil_paper_table.py --out _experiments/fscil_style/results_table.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from summarize_fscil_results import REFERENCE, discover_logs, parse_fscil_log, _fmt_curve

REPO = Path(__file__).resolve().parents[1]

CAMPAIGNS: tuple[tuple[str, str, str, int], ...] = (
    (
        "ModelNet40",
        "_experiments/fscil_style_mn40_5t_fscil3d",
        "selfevo_fscil_fscil3d_fleg_hparams",
        5,
    ),
    (
        "ShapeNet55",
        "_experiments/fscil_style_sn55_7t_fscil3d",
        "selfevo_fscil_fscil3d_fleg_hparams",
        7,
    ),
)

CAMPAIGNS_FLEX_COMPARE: tuple[tuple[str, str, str, int], ...] = (
    (
        "ModelNet40",
        "_experiments/flex_compare",
        "seel_mn40_flex_compare",
        5,
    ),
    (
        "ShapeNet55",
        "_experiments/flex_compare",
        "seel_sn55_flex_compare",
        7,
    ),
)

CAMPAIGN_CHOICES = {
    "fscil_style": CAMPAIGNS,
    "flex_compare": CAMPAIGNS_FLEX_COMPARE,
}

REF_KEYS = {
    5: "modelnet40_5t",
    7: "shapenet55_7t",
}

FOOTNOTE = (
    "Protocol notes: FSCIL-3D **class order** (fixed session splits); "
    "5-shot increments with 1-shot old-class replay; frozen Uni3D-b backbone. "
    "We use FPS point clouds + random 5-shot subsampling, not FSCIL-3D `.pt` index samples — "
    "numbers are comparable in split schedule but not sample-identical to 3D-FLEG."
)


def _best_complete_run(root: Path, method: str, expected_sessions: int) -> dict | None:
    logs = discover_logs(root).get(method, [])
    complete = []
    for seed, path in logs:
        parsed = parse_fscil_log(path, expected_sessions)
        if parsed and parsed["complete"]:
            complete.append((seed, parsed))
    if not complete:
        return None
    complete.sort(key=lambda x: x[0])
    return complete[-1][1]


def build_table(campaigns: tuple[tuple[str, str, str, int], ...], title_suffix: str) -> str:
    lines: list[str] = []
    lines.append(f"# FSCIL vs 3D-FLEG — SEEL ({title_suffix})\n")
    lines.append(f"*{FOOTNOTE}*\n")
    lines.append("| Dataset | Method | AA ↑ | Last ↑ | Δ ↓ | Base@Last | Accuracy curve (per session) |")
    lines.append("|---------|--------|------|--------|-----|-----------|------------------------------|")

    for dataset_name, log_rel, method, n_sess in campaigns:
        root = REPO / log_rel
        ref = REFERENCE[REF_KEYS[n_sess]]
        ours = _best_complete_run(root, method, n_sess)

        ref_delta = ref.get("delta", ref["curve"][0] - ref["last"])
        lines.append(
            f"| {dataset_name} | {ref['name']} (paper) | "
            f"{ref['aa']:.1f} | {ref['last']:.1f} | {ref_delta:.1f} | — | "
            f"{_fmt_curve(ref['curve'])} |"
        )

        if ours is None:
            lines.append(
                f"| {dataset_name} | SEEL (ours) | — | — | — | — | *run pending* |"
            )
            continue

        base = ours.get("base_last")
        base_str = f"{base:.1f}" if base is not None else "—"
        lines.append(
            f"| {dataset_name} | SEEL (ours, seed 2008) | "
            f"{ours['aa']:.2f} | {ours['last']:.2f} | {ours['delta']:.2f} | {base_str} | "
            f"{_fmt_curve(ours['curve'])} |"
        )

        d_aa = ours["aa"] - ref["aa"]
        d_last = ours["last"] - ref["last"]
        lines.append(
            f"| {dataset_name} | Δ (ours − paper) | "
            f"{d_aa:+.2f} | {d_last:+.2f} | {ours['delta'] - ref_delta:+.2f} | — | — |"
        )

    lines.append("")
    lines.append("## Session detail\n")
    for dataset_name, log_rel, method, n_sess in campaigns:
        root = REPO / log_rel
        ref = REFERENCE[REF_KEYS[n_sess]]
        ours = _best_complete_run(root, method, n_sess)
        lines.append(f"### {dataset_name}\n")
        lines.append("| Session | 3D-FLEG | SEEL | Δ |")
        lines.append("|---------|---------|----------|---|")
        for i in range(n_sess):
            ref_v = ref["curve"][i] if i < len(ref["curve"]) else float("nan")
            if ours and i < len(ours["curve"]):
                our_v = ours["curve"][i]
                lines.append(f"| {i} | {ref_v:.1f} | {our_v:.2f} | {our_v - ref_v:+.2f} |")
            else:
                lines.append(f"| {i} | {ref_v:.1f} | — | — |")
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="FSCIL paper comparison table")
    parser.add_argument(
        "--campaign",
        type=str,
        choices=sorted(CAMPAIGN_CHOICES),
        default="fscil_style",
        help="Log campaign to summarize (default: fscil_style)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="",
    )
    args = parser.parse_args()
    campaigns = CAMPAIGN_CHOICES[args.campaign]
    if args.campaign == "flex_compare":
        default_out = REPO / "_experiments/flex_compare/results_table.md"
        title_suffix = "flex_compare / FSCIL-3D class order"
    else:
        default_out = REPO / "_experiments/fscil_style/results_table.md"
        title_suffix = "FSCIL-3D class order"
    out_path = Path(args.out) if args.out else default_out
    text = build_table(campaigns, title_suffix)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text)
    print(text)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
