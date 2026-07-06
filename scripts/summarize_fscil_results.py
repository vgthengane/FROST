#!/usr/bin/env python3
"""Summarize FSCIL logs in 3D-FLEG / paper-style metrics.

Protocol (matches our FSCIL configs and 3D-FLEG reporting):
  - **Total accuracy** per session: top-1 on all classes seen so far (``CNN top1 curve``).
  - **AA**: mean of per-session total accuracies (``Average Accuracy (CNN)`` at end).
  - **Last**: total accuracy after the final session (last value in curve).
  - **Δ**: session-0 total acc minus Last (performance drop; 3D-FLEG reports this).
  - **Base @ Last**: base-class retention on session-0 classes at final session.
  - **Harmonic @ Last**: harmonic mean of old/new acc at final session (when logged).

Log layout (canonical)::

    _experiments/flex_compare/
      <method>_seed=<seed>.log

Usage::

    aprun python scripts/summarize_fscil_results.py --log-root _experiments/flex_compare
    aprun python scripts/summarize_fscil_results.py --log-root _experiments/flex_compare --out _experiments/flex_compare/results.md
"""

from __future__ import annotations

import argparse
import ast
import re
import statistics
from pathlib import Path

AVG_RE = re.compile(r"Average Accuracy \(CNN\):\s*([\d.]+)")
CURVE_RE = re.compile(r"CNN top1 curve:\s*\[([^\]]+)\]")
BASE_CURVE_RE = re.compile(r"FSCIL base retention curve:\s*\[([^\]]+)\]")
HARM_CURVE_RE = re.compile(r"FSCIL harmonic curve:\s*\[([^\]]+)\]")
CNN_GROUPED_RE = re.compile(r"CNN:\s*(\{[^}]+\})")
FLAT_LOG_RE = re.compile(r"^(?P<method>.+)_seed=(?P<seed>\d+)\.log$")

# 3D-FLEG (ICML 2025) reference numbers for comparison — Table 1 style FSCIL.
REFERENCE = {
    "modelnet40_5t": {
        "name": "3D-FLEG",
        "sessions": 5,
        "curve": [98.4, 90.0, 94.7, 87.5, 87.3],  # approximate from paper plot
        "aa": 93.2,
        "last": 87.3,
        "delta": 11.1,
    },
    "shapenet55_7t": {
        "name": "3D-FLEG",
        "sessions": 7,
        "curve": [94.3, 91.4, 89.7, 89.2, 88.4, 86.2, 83.0],
        "aa": 89.0,  # mean of curve
        "last": 83.0,
        "delta": 12.0,
    },
}


def infer_expected_sessions(method: str, root_name: str) -> int | None:
    """Infer session count from method or log-root name (MN40=5, SN55=7)."""
    key = f"{method}_{root_name}".lower()
    if "sn55" in key or "shapenet55" in key:
        return 7
    if "mn40" in key or "modelnet40" in key:
        return 5
    if "shapenet" in key and "scanobjectnn" not in key:
        return 7
    if "modelnet" in key and "scanobjectnn" not in key:
        return 5
    return None


def reference_for_sessions(expected: int | None) -> dict | None:
    if expected == 7:
        return REFERENCE.get("shapenet55_7t")
    if expected == 5:
        return REFERENCE.get("modelnet40_5t")
    return None


def _parse_float_list(raw: str) -> list[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def _parse_grouped_dict(text: str) -> dict | None:
    matches = CNN_GROUPED_RE.findall(text)
    if not matches:
        return None
    try:
        return ast.literal_eval(matches[-1])
    except (SyntaxError, ValueError):
        return None


def parse_fscil_log(path: Path, expected_sessions: int | None) -> dict | None:
    text = path.read_text(errors="replace")
    curves = CURVE_RE.findall(text)
    if not curves:
        return None
    top1 = _parse_float_list(curves[-1])
    avgs = [float(v) for v in AVG_RE.findall(text)]
    base_curves = BASE_CURVE_RE.findall(text)
    harm_curves = HARM_CURVE_RE.findall(text)
    base = _parse_float_list(base_curves[-1]) if base_curves else []
    harm = _parse_float_list(harm_curves[-1]) if harm_curves else []
    grouped = _parse_grouped_dict(text)

    complete = expected_sessions is None or len(top1) >= expected_sessions
    return {
        "sessions": len(top1),
        "curve": top1,
        "aa": avgs[-1] if avgs else None,
        "last": top1[-1] if top1 else None,
        "delta": (top1[0] - top1[-1]) if len(top1) >= 2 else None,
        "base_last": base[-1] if base else None,
        "harmonic_last": harm[-1] if harm else None,
        "old_last": grouped.get("old") if grouped else None,
        "new_last": grouped.get("new") if grouped else None,
        "complete": complete,
        "path": str(path),
    }


def discover_logs(root: Path) -> dict[str, list[tuple[int, Path]]]:
    found: dict[str, list[tuple[int, Path]]] = {}
    if not root.is_dir():
        return found
    for log in sorted(root.glob("*.log")):
        match = FLAT_LOG_RE.match(log.name)
        if not match:
            continue
        method = match.group("method")
        seed = int(match.group("seed"))
        found.setdefault(method, []).append((seed, log))
    return found


def _fmt_curve(curve: list[float]) -> str:
    return " → ".join(f"{v:.1f}" for v in curve)


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    if len(values) == 1:
        return values[0], 0.0
    return statistics.mean(values), statistics.stdev(values)


def print_method_table(method: str, runs: list[tuple[int, dict]], ref: dict | None) -> None:
    print(f"\n### {method}\n")
    print(
        f"{'Seed':>6} {'Sess':>4} {'AA':>7} {'Last':>7} "
        f"{'Δ':>6} {'Base':>7} {'Harm':>7} {'Old':>7} {'New':>7}"
    )
    print("-" * 72)
    complete_runs = []
    for seed, m in sorted(runs, key=lambda x: x[0]):
        if not m["complete"]:
            flag = " (incomplete)"
        else:
            flag = ""
            complete_runs.append(m)
        print(
            f"{seed:>6} {m['sessions']:>4} "
            f"{m['aa']:>7.2f} {m['last']:>7.2f} "
            f"{m['delta']:>6.2f} "
            f"{(m['base_last'] or 0):>7.2f} "
            f"{(m['harmonic_last'] or 0):>7.2f} "
            f"{(m['old_last'] or 0):>7.2f} "
            f"{(m['new_last'] or 0):>7.2f}{flag}"
        )
        print(f"        curve: {_fmt_curve(m['curve'])}")

    if len(complete_runs) >= 2:
        for key, label in [
            ("aa", "AA"),
            ("last", "Last"),
            ("delta", "Δ"),
            ("base_last", "Base@Last"),
            ("harmonic_last", "Harm@Last"),
        ]:
            vals = [r[key] for r in complete_runs if r.get(key) is not None]
            if vals:
                mu, sd = _mean_std(vals)
                print(f"  mean±std {label}: {mu:.2f} ± {sd:.2f}  (n={len(vals)})")
    elif len(complete_runs) == 1:
        m = complete_runs[0]
        print(f"  single complete run: AA={m['aa']:.2f} Last={m['last']:.2f} Δ={m['delta']:.2f}")

    if ref:
        print(f"\n  Reference {ref['name']}: AA≈{ref['aa']:.1f} Last={ref['last']:.1f}", end="")
        if "delta" in ref:
            print(f" Δ={ref['delta']:.1f}", end="")
        print(f"\n  Ref curve: {_fmt_curve(ref['curve'])}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize FSCIL results (3D-FLEG style)")
    parser.add_argument("--log-root", type=str, required=True)
    parser.add_argument(
        "--expected-sessions",
        type=int,
        default=None,
        help="Required session count for 'complete' (default: infer from log-root name)",
    )
    parser.add_argument(
        "--methods",
        type=str,
        default="",
        help="Comma-separated method name filter (substring match on log prefix)",
    )
    parser.add_argument("--out", type=str, default="", help="Write markdown summary to this path")
    args = parser.parse_args()

    root = Path(args.log_root)
    root_default_expected = args.expected_sessions
    if root_default_expected is None:
        root_default_expected = infer_expected_sessions("", root.name)

    methods = discover_logs(root)
    if not methods:
        print(f"No *_seed=*.log files under {root}")
        return

    filters = [x.strip() for x in args.methods.split(",") if x.strip()]
    if filters:
        methods = {k: v for k, v in methods.items() if any(f in k for f in filters)}

    lines: list[str] = []
    lines.append(f"# FSCIL results — `{root}`\n")
    lines.append("Protocol: FSCIL-3D class order, 5-shot increments, metrics per 3D-FLEG.\n")

    print(f"\nFSCIL summary — {root}")
    print("Metrics: Total acc curve | AA | Last | Δ | Base@Last | Harmonic@Last")

    for method in sorted(methods):
        expected = args.expected_sessions
        if expected is None:
            expected = infer_expected_sessions(method, root.name) or root_default_expected
        ref = reference_for_sessions(expected)
        runs = []
        for seed, path in methods[method]:
            parsed = parse_fscil_log(path, expected)
            if parsed is None:
                continue
            runs.append((seed, parsed))
        if not runs:
            continue
        print(f"\n--- {method} (expected sessions: {expected}) ---")
        print_method_table(method, runs, ref)

        lines.append(f"## {method}\n")
        lines.append("| Seed | AA | Last | Δ | Base@Last | Curve |")
        lines.append("|------|-----|------|---|-----------|-------|")
        for seed, m in sorted(runs, key=lambda x: x[0]):
            if not m["complete"]:
                continue
            lines.append(
                f"| {seed} | {m['aa']:.2f} | {m['last']:.2f} | {m['delta']:.2f} | "
                f"{m['base_last']:.2f} | {_fmt_curve(m['curve'])} |"
            )
        complete = [m for _, m in runs if m["complete"]]
        if len(complete) >= 2:
            aa_mu, aa_sd = _mean_std([m["aa"] for m in complete])
            last_mu, last_sd = _mean_std([m["last"] for m in complete])
            lines.append(f"\n**Mean ± std (n={len(complete)}):** AA {aa_mu:.2f}±{aa_sd:.2f}, Last {last_mu:.2f}±{last_sd:.2f}\n")
        if ref:
            lines.append(f"*Reference ({ref['name']}): AA ≈ {ref['aa']:.1f}, Last = {ref['last']:.1f}*\n")
        lines.append("")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("\n".join(lines))
        print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
