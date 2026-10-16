#!/usr/bin/env python3
"""Print FSCIL-3D / microshape session schedule (class IDs per session)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def show_protocol(label: str, protocol_name: str, data_path: str | None = None) -> None:
    from datasets.fscil3d_index import load_protocol
    from datasets.modelnet40 import iModelNet40

    p = load_protocol(protocol_name, data_path=data_path)
    names = None
    if protocol_name == "modelnet" and data_path:
        names = iModelNet40.shape_names(data_path)

    print(f"\n{'=' * 72}")
    print(f"{label} — {p.n_sessions} sessions, {p.total_classes} classes")
    print(f"Increments: {p.increment_sizes}")
    print(f"Few-shot: {p.fewshot} novel | {p.old_shot_replay} old replay")
    print(f"{'=' * 72}")

    cumulative = 0
    full_order: list[int] = []
    for sess, task_ids in enumerate(p.task_class_ids):
        cumulative += len(task_ids)
        full_order.extend(task_ids)
        print(f"\nSession {sess}: +{len(task_ids)} classes (cumulative {cumulative})")
        print(f"  ids: {task_ids}")
        if names is not None:
            print(f"  names: {[names[i] for i in task_ids]}")

    print(f"\nFull class_order ({len(full_order)}): {full_order}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path",
        default=str(REPO / "_datasets" / "pclouds" / "ModelNet40"),
    )
    args = parser.parse_args()
    show_protocol("ModelNet40 (FSCIL-3D / microshape)", "modelnet", data_path=args.data_path)
    show_protocol("ShapeNet55 (FSCIL-3D / microshape)", "shapenet")


if __name__ == "__main__":
    main()
