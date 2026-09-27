#!/usr/bin/env python3
"""Summarize PRidE multi-seed FSCIL campaigns (α=0.5 EP10, 6 protocols).

Expects flat logs from::

    scripts/simpl_runs_pride_alpha05_ep10_multiseed.sh
    scripts/simpl_runs_pride_alpha05_buf_ep10_multiseed.sh
    scripts/simpl_runs_pride_final_sub_ep10_multiseed.sh
    scripts/simpl_runs_pride_final_sub_buf_ep10_multiseed.sh
    scripts/simpl_runs_pride_final_sub_rp1000_ep10_multiseed.sh
    scripts/simpl_runs_pride_final_sub_rp1000_buf_ep10_multiseed.sh

Log patterns::

    final10_<proto>_mean_close_st_<buffree|buf>_a05_seed=<seed>.log
    final_sub_<proto>_mean_st_<buffree|buf>_a05_rp2000_seed=<seed>.log
    final_sub_<proto>_mean_st_<buffree|buf>_a05_rp1000_seed=<seed>.log

Usage::

    aprun python scripts/summarize_pride_multiseed.py --variant buffree
    aprun python scripts/summarize_pride_multiseed.py --variant buf
    aprun python scripts/summarize_pride_multiseed.py --variant both
    aprun python scripts/summarize_pride_multiseed.py --variant final_sub_buffree
    aprun python scripts/summarize_pride_multiseed.py --variant final_sub_buf
    aprun python scripts/summarize_pride_multiseed.py --variant final_sub_both
    aprun python scripts/summarize_pride_multiseed.py --variant final_sub_rp1000_both
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from summarize_fscil_results import discover_logs, parse_fscil_log

REPO = Path(__file__).resolve().parents[1]

PROTOCOLS: list[tuple[str, str, int]] = [
    ("mn40", "MN40", 5),
    ("sn55", "SN55", 7),
    ("co3d", "CO3D", 6),
    ("mn_scanobj", "MN→ScanObj", 4),
    ("sn_scanobj", "SN→ScanObj", 4),
    ("sn_co3d", "SN→CO3D", 11),
]

# (log_root, method_suffix, label, method_prefix, expected_seeds)
VARIANTS = {
    "buffree": (
        REPO / "_experiments" / "00_ABLATION_ALPHA05_EP10_5SEEDS",
        "mean_close_st_buffree_a05",
        "buffer-free",
        "final10",
        (1947, 1991, 2001, 2008, 2019),
    ),
    "buf": (
        REPO / "_experiments" / "00_ABLATION_ALPHA05_BUF_EP10_5SEEDS",
        "mean_close_st_buf_a05",
        "buffer",
        "final10",
        (1947, 1991, 2001, 2008, 2019),
    ),
    "final_sub_buffree": (
        REPO / "_experiments" / "00_FINAL_SUB_MEAN_A05_RP2000_EP10_5SEEDS",
        "mean_st_buffree_a05_rp2000",
        "final_sub mean-only buffer-free α=0.5 rp=2000",
        "final_sub",
        (2008, 1947, 1991, 2001, 2019),
    ),
    "final_sub_buf": (
        REPO / "_experiments" / "00_FINAL_SUB_MEAN_A05_RP2000_BUF_EP10_5SEEDS",
        "mean_st_buf_a05_rp2000",
        "final_sub mean-only buffer α=0.5 rp=2000",
        "final_sub",
        (2008, 1947, 1991, 2001, 2019),
    ),
    "final_sub_rp1000_buffree": (
        REPO / "_experiments" / "00_FINAL_SUB_MEAN_A05_RP1000_EP10_5SEEDS",
        "mean_st_buffree_a05_rp1000",
        "final_sub mean-only buffer-free α=0.5 rp=1000",
        "final_sub",
        (1947, 1991, 2001, 2008, 2019),
    ),
    "final_sub_rp1000_buf": (
        REPO / "_experiments" / "00_FINAL_SUB_MEAN_A05_RP1000_BUF_EP10_5SEEDS",
        "mean_st_buf_a05_rp1000",
        "final_sub mean-only buffer α=0.5 rp=1000",
        "final_sub",
        (1947, 1991, 2001, 2008, 2019),
    ),
}

DEFAULT_SEEDS = (1947, 1991, 2001, 2008, 2019)


def _mean_std(vals: list[float]) -> tuple[float, float]:
    if not vals:
        return float("nan"), float("nan")
    if len(vals) == 1:
        return vals[0], 0.0
    return statistics.mean(vals), statistics.stdev(vals)


def _fmt_ms(mu: float, sd: float) -> str:
    if mu != mu:  # NaN
        return "—"
    return f"{mu:.2f}±{sd:.2f}"


def _fmt_curve_mean(curves: list[list[float]]) -> str:
    if not curves:
        return "—"
    n = min(len(c) for c in curves)
    means = [statistics.mean(c[i] for c in curves) for i in range(n)]
    return ",".join(f"{v:.1f}" for v in means)


def _method_name(prefix: str, proto: str, suffix: str) -> str:
    return f"{prefix}_{proto}_{suffix}"


def summarize_variant(variant: str, log_root: Path | None, out: Path | None) -> str:
    default_root, suffix, label, prefix, expected_seeds = VARIANTS[variant]
    root = log_root or default_root
    discovered = discover_logs(root)

    lines: list[str] = [
        f"# PRidE multi-seed — α=0.5 EP10 ({label})",
        "",
        f"Log root: `{root}`",
        f"Expected seeds: {', '.join(str(s) for s in expected_seeds)}",
        "",
        "| Protocol | n | AA | Last | Δ | Base@Last | Novel@Last | Harm@Last | Mean curve |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]

    detail: list[str] = ["", "## Per-seed detail", ""]

    for proto, title, n_sess in PROTOCOLS:
        method = _method_name(prefix, proto, suffix)
        runs = discovered.get(method, [])
        complete: list[tuple[int, dict]] = []
        for seed, path in sorted(runs, key=lambda x: x[0]):
            parsed = parse_fscil_log(path, n_sess)
            if parsed and parsed["complete"]:
                complete.append((seed, parsed))

        detail.append(f"### {title} (`{method}`)")
        detail.append("")
        if not complete:
            detail.append("_No complete runs._")
            detail.append("")
            lines.append(
                f"| {title} | 0 | — | — | — | — | — | — | — |"
            )
            continue

        for seed, m in complete:
            detail.append(
                f"- seed={seed}: AA={m['aa']:.2f} Last={m['last']:.2f} "
                f"Δ={m['delta']:.2f} curve={m['curve']}"
            )
        detail.append("")

        def col(key: str) -> str:
            vals = [m[key] for _, m in complete if m.get(key) is not None]
            return _fmt_ms(*_mean_std(vals))

        curves = [m["curve"] for _, m in complete]
        lines.append(
            f"| {title} | {len(complete)} | {col('aa')} | {col('last')} | "
            f"{col('delta')} | {col('base_last')} | {col('novel_last')} | "
            f"{col('harmonic_last')} | `{_fmt_curve_mean(curves)}` |"
        )

    missing = []
    for proto, _, _ in PROTOCOLS:
        method = _method_name(prefix, proto, suffix)
        have = {s for s, _ in discovered.get(method, [])}
        for seed in expected_seeds:
            if seed not in have:
                missing.append(f"{method}_seed={seed}.log")
    if missing:
        lines.extend(["", "## Missing logs", ""])
        lines.extend(f"- `{name}`" for name in missing)

    lines.extend(detail)
    text = "\n".join(lines) + "\n"

    out_path = out or (root / "results_multiseed.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text)
    print(text)
    print(f"Wrote {out_path}", file=sys.stderr)
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant",
        choices=(
            "buffree",
            "buf",
            "both",
            "final_sub_buffree",
            "final_sub_buf",
            "final_sub_both",
            "final_sub_rp1000_buffree",
            "final_sub_rp1000_buf",
            "final_sub_rp1000_both",
        ),
        default="both",
    )
    parser.add_argument("--log-root", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    if args.variant == "both":
        variants = ("buffree", "buf")
    elif args.variant == "final_sub_both":
        variants = ("final_sub_buffree", "final_sub_buf")
    elif args.variant == "final_sub_rp1000_both":
        variants = ("final_sub_rp1000_buffree", "final_sub_rp1000_buf")
    else:
        variants = (args.variant,)

    combo = ("both", "final_sub_both", "final_sub_rp1000_both")
    if args.variant in combo and args.log_root is not None:
        parser.error("--log-root requires a single --variant (not a *both combo)")
    if args.variant in combo and args.out is not None:
        parser.error("--out requires a single --variant (not a *both combo)")

    for variant in variants:
        summarize_variant(variant, args.log_root, args.out)


if __name__ == "__main__":
    main()
