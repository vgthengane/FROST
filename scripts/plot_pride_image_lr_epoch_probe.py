#!/usr/bin/env python3
"""Plot AdaptFormer warmup convergence for the local LR × epoch probe.

Reads logs under ``_experiments/00_PRIDE_IMAGE_LR_EPOCH_PROBE/`` written by
``scripts/run_pride_image_lr_epoch_probe.sh`` (lines ``warmup_curve task=0 ...``).

Usage::

    aprun python scripts/plot_pride_image_lr_epoch_probe.py
    aprun python scripts/plot_pride_image_lr_epoch_probe.py --fast-only
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO / "_experiments" / "00_PRIDE_IMAGE_LR_EPOCH_PROBE"

CURVE_RE = re.compile(
    r"warmup_curve task=0 epoch=(?P<epoch>\d+)/(?P<total>\d+) "
    r"loss=(?P<loss>[0-9.]+) train_acc=(?P<train>[0-9.]+) "
    r"test_acc=(?P<test>[0-9.]+) lr=(?P<lr>[0-9.]+)"
)
METHOD_RE = re.compile(
    r"^(?:fast_)?probe_cub_lr(?P<lr_tag>[0-9p]+)_ep(?P<epochs>\d+)\.log$"
)


def _lr_from_tag(tag: str) -> float:
    return float(tag.replace("p", "."))


def parse_log(path: Path) -> dict | None:
    text = path.read_text(errors="replace")
    points = []
    for m in CURVE_RE.finditer(text):
        points.append(
            {
                "epoch": int(m.group("epoch")),
                "loss": float(m.group("loss")),
                "train": float(m.group("train")),
                "test": float(m.group("test")),
                "lr": float(m.group("lr")),
            }
        )
    if not points:
        return None
    # Keep last occurrence per epoch (in case of restarts).
    by_ep = {p["epoch"]: p for p in points}
    ordered = [by_ep[e] for e in sorted(by_ep)]
    m = METHOD_RE.match(path.name)
    return {
        "path": path,
        "fast": path.name.startswith("fast_"),
        "lr": ordered[0]["lr"] if ordered else (_lr_from_tag(m.group("lr_tag")) if m else float("nan")),
        "points": ordered,
        "final_test": ordered[-1]["test"],
        "final_train": ordered[-1]["train"],
        "best_test": max(p["test"] for p in ordered),
        "best_epoch": max(ordered, key=lambda p: p["test"])["epoch"],
    }


def discover(root: Path, fast_only: bool, full_only: bool) -> list[dict]:
    runs = []
    for path in sorted(root.glob("*probe_cub_lr*.log")):
        parsed = parse_log(path)
        if not parsed:
            continue
        if fast_only and not parsed["fast"]:
            continue
        if full_only and parsed["fast"]:
            continue
        runs.append(parsed)
    runs.sort(key=lambda r: (r["fast"], r["lr"]))
    return runs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--fast-only", action="store_true")
    parser.add_argument("--full-only", action="store_true")
    args = parser.parse_args()

    runs = discover(args.log_root, args.fast_only, args.full_only)
    if not runs:
        raise SystemExit(
            f"No warmup_curve logs in {args.log_root}. "
            "Run scripts/run_pride_image_lr_epoch_probe.sh first."
        )

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(f"matplotlib required for plotting: {exc}") from exc

    out = args.out or (args.log_root / "lr_epoch_convergence.png")
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)

    for run in runs:
        eps = [p["epoch"] for p in run["points"]]
        label = f"lr={run['lr']:g}" + (" (fast)" if run["fast"] else "")
        axes[0].plot(eps, [p["test"] for p in run["points"]], marker="o", ms=3, label=label)
        axes[1].plot(eps, [p["train"] for p in run["points"]], marker="o", ms=3, label=label)
        axes[2].plot(eps, [p["loss"] for p in run["points"]], marker="o", ms=3, label=label)

    axes[0].set_title("Base test acc (session 0)")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("test acc %")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)

    axes[1].set_title("Base train acc")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("train acc %")
    axes[1].grid(True, alpha=0.3)

    axes[2].set_title("Warmup CE loss")
    axes[2].set_xlabel("epoch")
    axes[2].set_ylabel("loss")
    axes[2].grid(True, alpha=0.3)

    fig.suptitle("CUB AdaptFormer warmup: LR vs convergence (task-0 only)", fontsize=12)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    print(f"Wrote {out}")

    md = args.log_root / "lr_epoch_summary.md"
    lines = [
        "# CUB LR × epoch AdaptFormer probe",
        "",
        f"Log root: `{args.log_root}`",
        "",
        "| Run | LR | epochs logged | best test@ep | final test | final train |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for run in runs:
        tag = "fast" if run["fast"] else "full"
        lines.append(
            f"| {tag} | {run['lr']:g} | {len(run['points'])} | "
            f"{run['best_test']:.2f}@{run['best_epoch']} | "
            f"{run['final_test']:.2f} | {run['final_train']:.2f} |"
        )
    lines.append("")
    lines.append(f"![convergence]({out.name})")
    lines.append("")
    md.write_text("\n".join(lines) + "\n")
    print(f"Wrote {md}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
