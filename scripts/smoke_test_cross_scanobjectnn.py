#!/usr/bin/env python3
"""Smoke test for ModelNet/ShapeNet -> ScanObjectNN cross-dataset pipelines."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "lamba_pilot"))

from datasets.cross_scanobjectnn import iCrossScanObjectNN
from datasets.fscil3d_index import load_protocol
from datasets.scanobjectnn import iScanObjectNN
from datasets.scanobjectnn_taxonomy import (
    MODELNET_SCANOBJECTNN_NOVEL,
    SHAPENET_SCANOBJECTNN_NOVEL,
    scanobjectnn_name_to_label,
)
from utils.data_manager import DataManager


SMOKE_ARGS = {
    "max_train_per_class": 3,
    "max_test_per_class": 2,
    "seed": 2008,
    "num_point": 1024,
    "use_normals": True,
    "process_data": True,
    "use_uniform_sample": True,
}

CONFIGS = [
    (
        "modelnet_scanobjectnn",
        REPO / "configs" / "selfevo_fscil_mn40_scanobjectnn_4t.json",
        [26, 4, 4, 3],
        37,
        26,
    ),
    (
        "shapenet_scanobjectnn",
        REPO / "configs" / "selfevo_fscil_sn55_scanobjectnn_4t.json",
        [44, 5, 5, 5],
        59,
        44,
    ),
]


def _check_protocol(name: str, expected_increments: list[int], expected_total: int) -> None:
    proto = load_protocol(name)
    assert proto.increment_sizes == expected_increments, (
        f"{name}: increments {proto.increment_sizes} != {expected_increments}"
    )
    assert proto.total_classes == expected_total, (
        f"{name}: total {proto.total_classes} != {expected_total}"
    )
    print(f"[protocol] {name}: increments={proto.increment_sizes}, total={expected_total}")


def _check_scanobjectnn_loader() -> None:
    args = {
        **SMOKE_ARGS,
        "scanobjectnn_data_path": "_datasets/scanobjectnn/main_split_nobg",
    }
    ds = iScanObjectNN(args)
    ds.download_data()
    assert ds.train_data.ndim == 3 and ds.train_data.shape[1:] == (1024, 6)
    assert ds.test_data.ndim == 3
    assert len(np.unique(ds.train_targets)) <= 15
    print(
        f"[scanobjectnn] train={ds.train_data.shape}, test={ds.test_data.shape}, "
        f"classes={len(np.unique(ds.train_targets))}"
    )


def _check_cross_loader(protocol: str, cfg: dict, expected_total: int, base_count: int) -> None:
    args = {**cfg, **SMOKE_ARGS}
    ds = iCrossScanObjectNN(args)
    ds.download_data()

    assert len(ds.class_order) == expected_total
    assert ds.train_data.shape[-1] == 6
    assert ds.test_data.shape[-1] == 6

    train_ids = set(np.unique(ds.train_targets))
    test_ids = set(np.unique(ds.test_targets))
    expected_ids = set(range(expected_total))
    missing_train = expected_ids - train_ids
    missing_test = expected_ids - test_ids
    if missing_train:
        raise AssertionError(f"{protocol}: missing train classes {sorted(missing_train)}")
    if missing_test:
        raise AssertionError(f"{protocol}: missing test classes {sorted(missing_test)}")

    novel_ids = set(range(base_count, expected_total))
    base_ids = set(range(base_count))
    assert base_ids.issubset(train_ids)
    assert novel_ids.issubset(train_ids)

    if protocol == "modelnet_scanobjectnn":
        inv = {v: k for k, v in iCrossScanObjectNN._scanobjectnn_to_cross(protocol).items()}
        for cross_id in sorted(novel_ids):
            native = inv[cross_id]
            expected = MODELNET_SCANOBJECTNN_NOVEL[cross_id]
            actual = scanobjectnn_name_to_label()
            actual = [k for k, v in actual.items() if v == native][0]
            assert actual == expected, f"cross {cross_id}: {actual} != {expected}"
    if protocol == "shapenet_scanobjectnn":
        inv = {v: k for k, v in iCrossScanObjectNN._scanobjectnn_to_cross(protocol).items()}
        for cross_id in sorted(novel_ids):
            native = inv[cross_id]
            expected = SHAPENET_SCANOBJECTNN_NOVEL[cross_id]
            actual = [k for k, v in scanobjectnn_name_to_label().items() if v == native][0]
            assert actual == expected

    print(
        f"[cross:{protocol}] train={ds.train_data.shape}, test={ds.test_data.shape}, "
        f"base={len(base_ids)} novel={len(novel_ids)}"
    )


def _check_task0_base_only(cfg: dict, base_count: int) -> None:
    args = {**cfg, **SMOKE_ARGS}
    dm = DataManager(
        dataset_name=cfg["dataset"],
        shuffle=False,
        seed=2008,
        init_cls=cfg["init_cls"],
        increment=cfg["increment"],
        args=args,
    )
    task0 = dm.get_dataset(np.arange(base_count), source="train", mode="train")
    labels = set(np.unique(task0.labels).tolist())
    assert labels == set(range(base_count)), f"task0 labels {sorted(labels)} != base 0..{base_count-1}"
    print(f"[task0:{cfg['dataset']}] train_n={len(task0)} classes={base_count} (base only)")


def _check_data_manager(cfg: dict, expected_increments: list[int], expected_total: int) -> None:
    args = {**cfg, **SMOKE_ARGS}
    dm = DataManager(
        dataset_name=cfg["dataset"],
        shuffle=False,
        seed=2008,
        init_cls=cfg["init_cls"],
        increment=cfg["increment"],
        args=args,
    )
    task_sizes = [dm.get_task_size(i) for i in range(dm.nb_tasks)]
    assert task_sizes == expected_increments, f"task_sizes {task_sizes} != {expected_increments}"
    assert dm.nb_classes == expected_total
    assert int(dm._train_targets.max()) == expected_total - 1
    assert int(dm._test_targets.max()) == expected_total - 1
    print(
        f"[datamanager:{cfg['dataset']}] nb_tasks={dm.nb_tasks}, "
        f"task_sizes={task_sizes}, train={dm._train_data.shape}"
    )


def main() -> None:
    print("=" * 72)
    print("Cross ScanObjectNN smoke test")
    print("=" * 72)

    _check_scanobjectnn_loader()

    for protocol, cfg_path, increments, total, base_count in CONFIGS:
        cfg = json.loads(cfg_path.read_text())
        _check_protocol(protocol, increments, total)
        _check_cross_loader(protocol, cfg, total, base_count)
        _check_data_manager(cfg, increments, total)
        _check_task0_base_only(cfg, base_count)

    print("\nSMOKE TEST PASSED")


if __name__ == "__main__":
    main()
