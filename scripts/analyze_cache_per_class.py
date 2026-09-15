#!/usr/bin/env python3
"""Parse per-class KeyCache prototype counts from training logs.

Reads lines of the form::

    KeyCache per-class keys task=4 n_classes=40 min=1 max=8 mean=4.62 total=185 histogram=1x20,5x10,8x10
    KeyCache per-class keys task=4 cls_keys=0:1,1:1,2:5,3:8,...

Legacy logs (total size only) are supported with a warning.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

SUMMARY_RE = re.compile(
    r"KeyCache per-class keys task=(\d+) n_classes=(\d+) min=(\d+) max=(\d+) "
    r"mean=([\d.]+) total=(\d+)(?: histogram=([\d,x]+))?"
)
CLS_KEYS_RE = re.compile(r"KeyCache per-class keys task=(\d+) cls_keys=([0-9:,]+)")
LEGACY_SUMMARY_RE = re.compile(
    r"KeyCache per-class counts task=(\d+) min=(\d+) max=(\d+) "
    r"mean=([\d.]+) n_classes=(\d+)"
)
EVOLVE_RE = re.compile(
    r"KeyCache evolve task=(\d+) policy=\S+ size=(\d+)"
)

REPO = Path(__file__).resolve().parents[1]


def _parse_cls_keys(raw: str) -> dict[int, int]:
    out: dict[int, int] = {}
    for part in raw.split(","):
        if not part.strip() or ":" not in part:
            continue
        cls_s, n_s = part.split(":", 1)
        out[int(cls_s)] = int(n_s)
    return out


def _parse_histogram(raw: str | None) -> dict[int, int]:
    """Parse ``1x20,5x10`` -> {1: 20, 5: 10} (#keys -> #classes)."""
    if not raw:
        return {}
    out: dict[int, int] = {}
    for part in raw.split(","):
        if "x" not in part:
            continue
        k_s, n_s = part.split("x", 1)
        out[int(k_s)] = int(n_s)
    return out


def parse_log(path: Path) -> dict[int, dict]:
    text = path.read_text(errors="replace")
    rows: dict[int, dict] = {}

    for m in SUMMARY_RE.finditer(text):
        task = int(m.group(1))
        rows[task] = {
            "task": task,
            "n_classes": int(m.group(2)),
            "min": int(m.group(3)),
            "max": int(m.group(4)),
            "mean": float(m.group(5)),
            "total": int(m.group(6)),
            "histogram": _parse_histogram(m.group(7)),
            "cls_keys": {},
            "source": "logged",
        }

    for m in CLS_KEYS_RE.finditer(text):
        task = int(m.group(1))
        if task not in rows:
            rows[task] = {"task": task, "source": "logged"}
        rows[task]["cls_keys"] = _parse_cls_keys(m.group(2))

    if rows:
        return rows

    for m in LEGACY_SUMMARY_RE.finditer(text):
        task = int(m.group(1))
        rows[task] = {
            "task": task,
            "min": int(m.group(2)),
            "max": int(m.group(3)),
            "mean": float(m.group(4)),
            "n_classes": int(m.group(5)),
            "source": "legacy_summary",
        }

    if rows:
        return rows

    for task_s, size_s in EVOLVE_RE.findall(text):
        task = int(task_s)
        rows[task] = {
            "task": task,
            "total": int(size_s),
            "source": "legacy_total_only",
        }
    return rows


def summarize_run(path: Path, show_classes: bool = False) -> None:
    rows = parse_log(path)
    if not rows:
        print(f"{path.name}: no KeyCache evolve lines")
        return

    last_task = max(rows)
    last = rows[last_task]
    print(f"\n## {path.name}")

    if last.get("source") == "logged" or last.get("source") == "legacy_summary":
        print(
            f"Last task {last_task}: min={last.get('min')} max={last.get('max')} "
            f"mean={last.get('mean', 0):.2f} n_classes={last.get('n_classes')} "
            f"total={last.get('total', '—')}"
        )
        if last.get("histogram"):
            print(f"Histogram (keys_per_class x num_classes): {last['histogram']}")
        if show_classes and last.get("cls_keys"):
            items = ", ".join(
                f"{c}:{n}" for c, n in sorted(last["cls_keys"].items())
            )
            print(f"Per-class keys: {items}")
    else:
        print(
            f"Last task {last_task}: total_cache_size={last['total']} "
            "(per-class detail not logged — re-run with updated code)"
        )

    print("\n| task | min | max | mean | n_classes | total | histogram |")
    print("|-----:|----:|----:|-----:|----------:|------:|-----------|")
    for task in sorted(rows):
        r = rows[task]
        hist = r.get("histogram") or {}
        hist_s = ",".join(f"{k}x{v}" for k, v in sorted(hist.items())) if hist else "—"
        if r.get("min") is not None:
            print(
                f"| {task} | {r.get('min', '—')} | {r.get('max', '—')} | "
                f"{r.get('mean', 0):.2f} | {r.get('n_classes', '—')} | "
                f"{r.get('total', '—')} | {hist_s} |"
            )
        else:
            print(
                f"| {task} | — | — | — | — | {r.get('total', '—')} | — |"
            )

    if show_classes:
        print("\n### Per-class keys (last task)")
        cls = last.get("cls_keys") or {}
        if not cls:
            print("(not available)")
        else:
            for c in sorted(cls):
                print(f"  class {c}: {cls[c]} key(s)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="*", type=Path, help="Log files to parse")
    parser.add_argument(
        "--ep5-per-shot",
        action="store_true",
        help="Summarize MN40 per-shot EP5 buffer runs",
    )
    parser.add_argument(
        "--all-ep5",
        action="store_true",
        help="All 00_BUF_CAMPAIGN_EP5 logs",
    )
    parser.add_argument(
        "--show-classes",
        action="store_true",
        help="Print every class id and its key count for the last task",
    )
    args = parser.parse_args()

    paths: list[Path] = list(args.logs)
    if args.ep5_per_shot:
        root = REPO / "_experiments" / "00_BUF_CAMPAIGN_EP5"
        paths.extend(sorted(root.glob("abl_mn40_ep5_*.log")))
    if args.all_ep5:
        root = REPO / "_experiments" / "00_BUF_CAMPAIGN_EP5"
        paths.extend(sorted(root.glob("abl_*_ep5_*.log")))

    if not paths:
        parser.error("Provide log paths or --ep5-per-shot / --all-ep5")

    for path in paths:
        summarize_run(path, show_classes=args.show_classes)


if __name__ == "__main__":
    main()
