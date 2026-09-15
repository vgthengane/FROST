#!/usr/bin/env python3
"""Summarize 4-variant × 4-protocol regime ablations (evolution story table).

Reports total Last / AA plus novel & harmonic (the metrics that matter for the
stability–plasticity claim).

Logs expected under::

    _experiments/00_SIMPL_REGIME_ABLATIONS/
      abl_mn40_* ...
      abl_sn55_* ...
      abl_mn_scanobj_* ...
      abl_sn_scanobj_* ...

or toy_*-prefixed names when --toy was used.
"""

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

PROTOCOLS = [
    ("mn40", "ModelNet40 FSCIL", 5),
    ("sn55", "ShapeNet55 FSCIL", 7),
    ("mn_scanobj", "MN40 → ScanObjectNN", 4),
    ("sn_scanobj", "SN55 → ScanObjectNN", 4),
]

VARIANTS = [
    ("ridge", "Ridge only"),
    ("cache", "Cache only"),
    ("static", "Ridge+cache (static)"),
    ("rules", "Ridge+cache+evolution"),
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
        "curve": top1,
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


def method_name(prefix: str, variant: str, toy: bool) -> str:
    base = f"abl_{prefix}_{variant}"
    return f"toy_{base}" if toy else base


def find_run(root: Path, method: str, expected: int) -> dict | None:
    # Prefer flat *_seed=*.log; fall back to tee *.log
    methods = discover_logs(root)
    for seed, path in sorted(methods.get(method, []), key=lambda x: x[0]):
        parsed = parse_rich(path, expected)
        if parsed:
            return parsed
    tee = root / f"{method}.log"
    if tee.is_file():
        return parse_rich(tee, expected)
    # Incomplete / alternate seed naming
    for path in sorted(root.glob(f"{method}*.log")):
        parsed = parse_rich(path, expected)
        if parsed:
            return parsed
    return None


def fmt(v: float | None) -> str:
    return "—" if v is None else f"{v:.2f}"


def build(root: Path, toy: bool) -> str:
    lines: list[str] = []
    mode = "TOY (2 sessions)" if toy else "FULL"
    lines.append(f"# Regime ablations — {mode}\n")
    lines.append(f"Log root: `{root}`\n")
    lines.append(
        "Story matrix: **ridge** vs **cache** vs **ridge+cache (static)** vs "
        "**ridge+cache+evolution (rules)**.\n"
    )
    lines.append(
        "Report **novel** & **harmonic** (plasticity / balance), not just total Last.\n"
    )

    for prefix, title, n_sess in PROTOCOLS:
        lines.append(f"## {title}\n")
        lines.append(
            "| Variant | AA ↑ | Last ↑ | Base@Last | Novel@Last | Harm@Last | ΔLast vs rules |"
        )
        lines.append(
            "|---------|------|--------|-----------|------------|-----------|----------------|"
        )
        rules = find_run(root, method_name(prefix, "rules", toy), n_sess if not toy else 2)
        for vid, label in VARIANTS:
            run = find_run(root, method_name(prefix, vid, toy), n_sess if not toy else 2)
            if run is None:
                lines.append(f"| {label} | — | — | — | — | — | — |")
                continue
            dlast = "—"
            if rules and run.get("last") is not None and rules.get("last") is not None:
                dlast = f"{run['last'] - rules['last']:+.2f}"
            flag = "" if run["complete"] else " *"
            lines.append(
                f"| {label}{flag} | {fmt(run.get('aa'))} | {fmt(run.get('last'))} | "
                f"{fmt(run.get('base_last'))} | {fmt(run.get('novel_last'))} | "
                f"{fmt(run.get('harmonic_last'))} | {dlast} |"
            )
        lines.append("")
        lines.append(
            "*Incomplete runs (fewer sessions than expected) are marked with `*`.*"
        )
        lines.append("")

    lines.append("## How to read this\n")
    lines.append(
        "- **Cache only** should lead Novel but lose Base.\n"
        "- **Ridge only** should lead Base but lag Novel.\n"
        "- **Ridge+cache (static)** = fusion without prototype reconfiguration.\n"
        "- **Ridge+cache+evolution** should improve Novel/Harmonic over static "
        "while keeping Base high — that is the proposed method story.\n"
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize regime ablations")
    parser.add_argument(
        "--log-root",
        type=Path,
        default=REPO / "_experiments" / "00_SIMPL_REGIME_ABLATIONS",
    )
    parser.add_argument("--toy", action="store_true", help="Read toy_* method names")
    parser.add_argument("--out", type=Path, default="")
    args = parser.parse_args()

    text = build(args.log_root, args.toy)
    print(text)
    out = args.out or (args.log_root / ("regime_ablation_tables_toy.md" if args.toy else "regime_ablation_tables.md"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
