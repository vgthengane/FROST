#!/usr/bin/env python3
"""Summarize in-domain and cross-domain adapter campaign logs."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from summarize_fscil_results import discover_logs, parse_fscil_log

REPO = Path(__file__).resolve().parents[1]
CURVE_RE = re.compile(r"CNN top1 curve:\s*\[([^\]]+)\]")
NOVEL_RE = re.compile(r"FSCIL novel accuracy curve:\s*\[([^\]]+)\]")
HARM_RE = re.compile(r"FSCIL harmonic curve:\s*\[([^\]]+)\]")
BASE_RE = re.compile(r"FSCIL base retention curve:\s*\[([^\]]+)\]")
AVG_RE = re.compile(r"Average Accuracy \(CNN\):\s*([\d.]+)")

INDOMAIN = [
    ("mn40", "ModelNet40 FSCIL", 5, "abl_id_mn40"),
    ("sn55", "ShapeNet55 FSCIL", 7, "abl_id_sn55"),
]

CROSSDOMAIN = [
    ("mn_scanobj", "MN40 → ScanObjectNN", 4, "abl_xd_mn"),
    ("sn_scanobj", "SN55 → ScanObjectNN", 4, "abl_xd_sn"),
]

INDOMAIN_VARIANTS = [
    ("full_2ep", "Full adapter 10/2"),
    ("full_10ep", "Full adapter 10/10"),
    ("base_only", "Base-only (task-0)"),
]

CROSSDOMAIN_VARIANTS = [
    ("full_2ep", "Full adapter 10/2"),
    ("full_10ep", "Full adapter 10/10"),
    ("base_t1_2ep", "Base + task-1 adapter 2 ep"),
    ("base_only", "Base-only (task-0)"),
]


def _floats(raw: str) -> list[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def parse_rich(path: Path, expected: int | None) -> dict | None:
    text = path.read_text(errors="replace")
    curves = CURVE_RE.findall(text)
    if not curves:
        return None
    top1 = _floats(curves[-1])
    avgs = AVG_RE.findall(text)
    novel = _floats(NOVEL_RE.findall(text)[-1]) if NOVEL_RE.findall(text) else []
    harm = _floats(HARM_RE.findall(text)[-1]) if HARM_RE.findall(text) else []
    base = _floats(BASE_RE.findall(text)[-1]) if BASE_RE.findall(text) else []
    complete = expected is None or len(top1) >= expected
    return {
        "aa": float(avgs[-1]) if avgs else (sum(top1) / len(top1) if top1 else None),
        "last": top1[-1] if top1 else None,
        "delta": (top1[0] - top1[-1]) if len(top1) >= 2 else None,
        "base_last": base[-1] if base else None,
        "novel_last": novel[-1] if novel else None,
        "harmonic_last": harm[-1] if harm else None,
        "complete": complete,
        "sessions": len(top1),
        "path": str(path),
    }


def find_run(root: Path, method: str, expected: int) -> dict | None:
    methods = discover_logs(root)
    for _seed, path in sorted(methods.get(method, []), key=lambda x: x[0]):
        parsed = parse_rich(path, expected)
        if parsed:
            return parsed
    tee = root / f"{method}.log"
    if tee.is_file():
        return parse_rich(tee, expected)
    for path in sorted(root.glob(f"{method}*.log")):
        parsed = parse_rich(path, expected)
        if parsed:
            return parsed
    return None


def fmt(v: float | None) -> str:
    return "—" if v is None else f"{v:.2f}"


def build_section(
    title: str,
    protocols: list[tuple[str, str, int, str]],
    variants: list[tuple[str, str]],
    root: Path,
) -> str:
    lines = [f"## {title}\n", f"Log root: `{root}`\n"]
    for _pid, ptitle, n_sess, prefix in protocols:
        lines.append(f"### {ptitle}\n")
        lines.append(
            "| Variant | AA ↑ | Last ↑ | Base@Last | Novel@Last | Harm@Last | ΔLast vs full_2ep |"
        )
        lines.append(
            "|---------|------|--------|-----------|------------|-----------|-------------------|"
        )
        ref = find_run(root, f"{prefix}_full_2ep", n_sess)
        for vid, label in variants:
            run = find_run(root, f"{prefix}_{vid}", n_sess)
            if run is None:
                lines.append(f"| {label} | — | — | — | — | — | — |")
                continue
            dlast = "—"
            if ref and run.get("last") is not None and ref.get("last") is not None:
                dlast = f"{run['last'] - ref['last']:+.2f}"
            flag = "" if run["complete"] else " *"
            lines.append(
                f"| {label}{flag} | {fmt(run.get('aa'))} | {fmt(run.get('last'))} | "
                f"{fmt(run.get('base_last'))} | {fmt(run.get('novel_last'))} | "
                f"{fmt(run.get('harmonic_last'))} | {dlast} |"
            )
        lines.append("")
    lines.append("*Incomplete runs are marked with `*`.*\n")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize adapter campaign tables")
    parser.add_argument(
        "--scope",
        choices=("indomain", "crossdomain", "all"),
        default="all",
    )
    parser.add_argument(
        "--id-root",
        type=Path,
        default=REPO / "_experiments" / "00_ADAPTER_CAMPAIGN_ID",
    )
    parser.add_argument(
        "--xd-root",
        type=Path,
        default=REPO / "_experiments" / "00_ADAPTER_CAMPAIGN_XD",
    )
    parser.add_argument("--out", type=Path, default="")
    args = parser.parse_args()

    parts = ["# Adapter campaign — seed 2008\n"]
    if args.scope in ("indomain", "all"):
        parts.append(
            build_section(
                "In-domain (MN40, SN55)",
                INDOMAIN,
                INDOMAIN_VARIANTS,
                args.id_root,
            )
        )
    if args.scope in ("crossdomain", "all"):
        parts.append(
            build_section(
                "Cross-domain (→ ScanObjectNN)",
                CROSSDOMAIN,
                CROSSDOMAIN_VARIANTS,
                args.xd_root,
            )
        )

    text = "\n".join(parts)
    print(text)

    if args.scope == "indomain":
        default_out = args.id_root / "adapter_campaign_tables.md"
    elif args.scope == "crossdomain":
        default_out = args.xd_root / "adapter_campaign_tables.md"
    else:
        default_out = REPO / "_experiments" / "adapter_campaign_tables.md"

    out_path = args.out or default_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
