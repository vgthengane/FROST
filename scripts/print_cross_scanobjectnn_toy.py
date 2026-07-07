#!/usr/bin/env python3
"""Print cross-dataset FSCIL toy protocol: ModelNet/ShapeNet base + ScanObjectNN novel."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from datasets.scanobjectnn_taxonomy import (
    MODELNET_SCANOBJECTNN_BASE,
    MODELNET_SCANOBJECTNN_NOVEL,
    SHAPENET_SCANOBJECTNN_BASE,
    SHAPENET_SCANOBJECTNN_NOVEL,
)


def _novel_by_sessions(novel: dict[int, str], session_increments: list[int], base_count: int) -> list[list[str]]:
    """Group novel class names by incremental session (tasks 2..N)."""
    ordered = [novel[cid] for cid in sorted(novel)]
    groups: list[list[str]] = []
    offset = 0
    for inc in session_increments[1:]:
        groups.append(ordered[offset : offset + inc])
        offset += inc
    return groups


def _print_protocol(
    title: str,
    *,
    base_dataset: str,
    base_names: tuple[str, ...],
    novel_dataset: str,
    session_increments: list[int],
    novel_table: dict[int, str],
    show_counts: bool = False,
    config_path: Path | None = None,
) -> None:
    print(f"\n{'=' * 78}")
    print(title)
    print(f"Sessions: {len(session_increments)} | Classes: {sum(session_increments)}")
    print(f"Increments: {session_increments}")
    print(f"{'=' * 78}")

    # Task 1 — base
    print(f"\nTask 1 | Dataset: {base_dataset}")
    print(f"  Classes ({len(base_names)}): {', '.join(base_names)}")

    novel_groups = _novel_by_sessions(novel_table, session_increments, len(base_names))
    for task_idx, names in enumerate(novel_groups, start=2):
        print(f"\nTask {task_idx} | Dataset: {novel_dataset}")
        print(f"  Classes ({len(names)}): {', '.join(names)}")

    if not show_counts or config_path is None:
        return

    import json

    sys.path.insert(0, str(REPO / "lamba_pilot"))
    from utils.data_manager import DataManager
    from datasets.cross_scanobjectnn import iCrossScanObjectNN
    cfg = json.loads(config_path.read_text())
    cfg.update(
        {
            "max_train_per_class": 3,
            "max_test_per_class": 2,
            "seed": 2008,
            "process_data": True,
        }
    )
    dm = DataManager(
        dataset_name=cfg["dataset"],
        shuffle=False,
        seed=2008,
        init_cls=cfg["init_cls"],
        increment=cfg["increment"],
        args=cfg,
    )

    print(f"\nToy sample counts (max 3 train / 2 test per class):")
    offset = 0
    for task in range(dm.nb_tasks):
        n_cls = dm.get_task_size(task)
        cls_ids = list(range(offset, offset + n_cls))
        train_ds = dm.get_dataset(cls_ids, source="train", mode="train")
        test_ds = dm.get_dataset(cls_ids, source="test", mode="test")
        dataset = base_dataset if task == 0 else novel_dataset
        print(
            f"  Task {task + 1} [{dataset}]: "
            f"train={len(train_ds)} test={len(test_ds)} classes={n_cls}"
        )
        offset += n_cls


def main() -> None:
    parser = argparse.ArgumentParser(description="Print cross ScanObjectNN FSCIL toy protocol")
    parser.add_argument(
        "--show-counts",
        action="store_true",
        help="Load toy-subsampled data and print per-task sample counts",
    )
    args = parser.parse_args()

    # (b) ShapeNet -> ScanObjectNN
    _print_protocol(
        "(b) ShapeNet -> ScanObjectNN",
        base_dataset="ShapeNet",
        base_names=SHAPENET_SCANOBJECTNN_BASE,
        novel_dataset="ScanObjectNN",
        session_increments=[44, 5, 5, 5],
        novel_table=SHAPENET_SCANOBJECTNN_NOVEL,
        show_counts=args.show_counts,
        config_path=REPO / "configs" / "toy_sn55_scanobjectnn_4t.json",
    )

    # (c) ModelNet -> ScanObjectNN
    _print_protocol(
        "(c) ModelNet -> ScanObjectNN",
        base_dataset="ModelNet",
        base_names=MODELNET_SCANOBJECTNN_BASE,
        novel_dataset="ScanObjectNN",
        session_increments=[26, 4, 4, 3],
        novel_table=MODELNET_SCANOBJECTNN_NOVEL,
        show_counts=args.show_counts,
        config_path=REPO / "configs" / "toy_mn40_scanobjectnn_4t.json",
    )


if __name__ == "__main__":
    main()
