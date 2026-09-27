#!/usr/bin/env python3
"""Compare MN40 cache scoring: mean-only max vs logsumexp vs diag_gauss.

Matched recipe (final_sub rp=1000): insert=mean, α=0.5, EP10, static, base-only.
Default: seed **2008** check (no multi-seed until diag_gauss looks promising).

Usage::

    aprun python scripts/summarize_pride_distribution_mn40.py
    aprun python scripts/summarize_pride_distribution_mn40.py --buffer buffree
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from summarize_fscil_results import parse_fscil_log

REPO = Path(__file__).resolve().parents[1]
N_SESS = 5
DEFAULT_SEEDS = (2008,)

# (label, log_root, method template with {buf})
POOLINGS: list[tuple[str, Path, str]] = [
    (
        "max (mean-only baseline)",
        REPO / "_experiments" / "00_FINAL_SUB_MEAN_A05_RP1000_EP10_5SEEDS",
        "final_sub_mn40_mean_st_{buf}_a05_rp1000",
    ),
    (
        "logsumexp",
        REPO / "_experiments" / "00_FINAL_SUB_MEAN_LOGSUMEXP_A05_RP1000_MN40",
        "final_sub_mn40_mean_logsumexp_{buf}_a05_rp1000",
    ),
    (
        "diag_gauss",
        REPO / "_experiments" / "00_FINAL_SUB_MEAN_DIAG_GAUSS_A05_RP1000_MN40",
        "final_sub_mn40_mean_diag_gauss_{buf}_a05_rp1000",
    ),
]

# Baseline log dirs: buffree uses EP10_5SEEDS, buf uses BUF_EP10_5SEEDS.
BASELINE_ROOT = {
    "buffree": REPO
    / "_experiments"
    / "00_FINAL_SUB_MEAN_A05_RP1000_EP10_5SEEDS",
    "buf": REPO
    / "_experiments"
    / "00_FINAL_SUB_MEAN_A05_RP1000_BUF_EP10_5SEEDS",
}


def _mean_std(vals: list[float]) -> tuple[float, float]:
    if not vals:
        return float("nan"), float("nan")
    if len(vals) == 1:
        return vals[0], 0.0
    return statistics.mean(vals), statistics.stdev(vals)


def _fmt(v: float | None) -> str:
    if v is None or v != v:
        return "—"
    return f"{v:.2f}"


def _fmt_ms(mu: float, sd: float) -> str:
    if mu != mu:
        return "—"
    if sd == 0.0:
        return f"{mu:.2f}"
    return f"{mu:.2f}±{sd:.2f}"


def _fmt_curve(curve: list[float] | None) -> str:
    if not curve:
        return "—"
    return " ".join(f"{x:.1f}" for x in curve)


def _resolve_root(label: str, root_template: Path, buf: str) -> Path:
    if "max" in label:
        return BASELINE_ROOT[buf]
    return root_template  # already absolute, no {BUF} for pool variants


def _method(template: str, buf: str) -> str:
    return template.format(buf=buf)


def _load_runs(
    root: Path, method: str, seeds: tuple[int, ...]
) -> list[tuple[int, dict]]:
    out: list[tuple[int, dict]] = []
    for seed in seeds:
        path = root / f"{method}_seed={seed}.log"
        if not path.is_file():
            continue
        parsed = parse_fscil_log(path, N_SESS)
        if parsed and parsed["complete"]:
            out.append((seed, parsed))
    return out


def _delta_vs_base(
    runs: list[tuple[int, dict]],
    base: dict[int, dict],
    key: str,
) -> list[float]:
    deltas: list[float] = []
    for seed, m in runs:
        b = base.get(seed)
        if b is None or m.get(key) is None or b.get(key) is None:
            continue
        deltas.append(float(m[key]) - float(b[key]))
    return deltas


def summarize(buffer: str, seeds: tuple[int, ...]) -> str:
    lines: list[str] = [
        f"# PRidE MN40 distribution vs prototypes ({buffer})",
        "",
        "Matched: insert=mean, α=0.5, rp=1000, EP10, static, base-only.",
        f"Seeds: {', '.join(str(s) for s in seeds)}",
        "",
        "| Pooling | n | AA | Last | ΔLast vs max | Novel@Last | "
        "ΔNovel vs max | Harm@Last | Mean curve |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]

    detail: list[str] = ["", "## Per-seed", ""]

    # Load baseline first for Δ columns.
    base_label, _, base_tmpl = POOLINGS[0]
    base_root = BASELINE_ROOT[buffer]
    base_method = _method(base_tmpl, buffer)
    base_runs = _load_runs(base_root, base_method, seeds)
    base_by_seed = {s: m for s, m in base_runs}

    for label, root_tmpl, method_tmpl in POOLINGS:
        root = _resolve_root(label, root_tmpl, buffer)
        method = _method(method_tmpl, buffer)
        runs = _load_runs(root, method, seeds)

        detail.append(f"### {label} (`{method}`)")
        detail.append(f"Root: `{root}`")
        detail.append("")
        if not runs:
            detail.append("_No complete runs._")
            detail.append("")
            lines.append(f"| {label} | 0 | — | — | — | — | — | — | — |")
            continue

        for seed, m in runs:
            detail.append(
                f"- seed={seed}: AA={_fmt(m['aa'])} Last={_fmt(m['last'])} "
                f"Novel={_fmt(m['novel_last'])} Harm={_fmt(m['harmonic_last'])} "
                f"curve=[{_fmt_curve(m['curve'])}]"
            )
        detail.append("")

        def col(key: str) -> str:
            vals = [m[key] for _, m in runs if m.get(key) is not None]
            return _fmt_ms(*_mean_std(vals))

        d_last = _delta_vs_base(runs, base_by_seed, "last")
        d_novel = _delta_vs_base(runs, base_by_seed, "novel_last")
        curves = [m["curve"] for _, m in runs]
        mean_curve = []
        if curves:
            n = min(len(c) for c in curves)
            mean_curve = [statistics.mean(c[i] for c in curves) for i in range(n)]

        # Baseline row: Δ vs self is 0.
        if "max" in label:
            d_last_s = "0.00"
            d_novel_s = "0.00" if any(
                m.get("novel_last") is not None for _, m in runs
            ) else "—"
        else:
            d_last_s = _fmt_ms(*_mean_std(d_last)) if d_last else "—"
            d_novel_s = _fmt_ms(*_mean_std(d_novel)) if d_novel else "—"

        lines.append(
            f"| {label} | {len(runs)} | {col('aa')} | {col('last')} | "
            f"{d_last_s} | {col('novel_last')} | {d_novel_s} | "
            f"{col('harmonic_last')} | `{_fmt_curve(mean_curve)}` |"
        )

    missing: list[str] = []
    for label, root_tmpl, method_tmpl in POOLINGS:
        root = _resolve_root(label, root_tmpl, buffer)
        method = _method(method_tmpl, buffer)
        have = {s for s, _ in _load_runs(root, method, seeds)}
        for seed in seeds:
            if seed not in have:
                missing.append(f"{root.name}/{method}_seed={seed}.log")
    if missing:
        lines.extend(["", "## Missing", ""])
        lines.extend(f"- `{name}`" for name in missing)

    lines.extend(detail)
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- With **1 key/class** (mean insert), `logsumexp` ≡ `max` "
            "(sanity control; expect Δ≈0).",
            "- `diag_gauss` uses shrunk diagonal shot variance — the real "
            "distributional alternative at 5-shot.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--buffer",
        choices=("buffree", "buf", "both"),
        default="both",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override seed (default: 2008).",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Comma-separated seed list (default: 2008 only).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Write markdown here (default: under diag_gauss log root).",
    )
    args = parser.parse_args()

    if args.seed is not None:
        seeds = (args.seed,)
    elif args.seeds:
        seeds = tuple(int(x) for x in args.seeds.split(","))
    else:
        seeds = DEFAULT_SEEDS

    buffers = ("buffree", "buf") if args.buffer == "both" else (args.buffer,)
    chunks = [summarize(b, seeds) for b in buffers]
    text = "\n---\n\n".join(chunks) + "\n"
    print(text)

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
        print(f"Wrote {args.out}", file=sys.stderr)
    else:
        out = (
            REPO
            / "_experiments"
            / "00_FINAL_SUB_MEAN_DIAG_GAUSS_A05_RP1000_MN40"
            / "results_distribution_vs_max.md"
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text)
        print(f"Wrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
