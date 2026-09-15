#!/usr/bin/env python3
"""Smoke test for CO3D in-dataset and ShapeNet->CO3D cross-dataset FSCIL runs."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "lamba_pilot"))

from datasets.co3d import iCO3D
from datasets.co3d_taxonomy import SHAPENET_CO3D_NOVEL
from datasets.cross_co3d import iCrossCO3D
from datasets.fscil3d_index import load_protocol
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
        "co3d",
        REPO / "configs" / "selfevo_fscil_co3d_6t.json",
        [25, 5, 5, 5, 5, 5],
        50,
        25,
    ),
    (
        "shapenet_co3d",
        REPO / "configs" / "selfevo_fscil_sn55_co3d_11t.json",
        [39, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
        89,
        39,
    ),
]


def _check_protocol(name: str, expected_increments: list[int], expected_total: int) -> None:
    proto = load_protocol(name)
    assert proto.increment_sizes == expected_increments, (
        f"{name}: increments {proto.increment_sizes} != {expected_increments}"
    )
    assert proto.total_classes == expected_total
    print(f"[protocol] {name}: increments={proto.increment_sizes}, total={expected_total}")


def _check_co3d_loader() -> None:
    args = {**SMOKE_ARGS, "data_path": "_datasets/CO3D", "num_category": 50}
    ds = iCO3D(args)
    ds.download_data()
    assert ds.train_data.ndim == 3 and ds.train_data.shape[1:] == (1024, 6)
    assert ds.test_data.ndim == 3
    assert len(ds.class_order) == 50
    print(
        f"[co3d] train={ds.train_data.shape}, test={ds.test_data.shape}, "
        f"classes={len(np.unique(ds.train_targets))}"
    )


def _check_cross_loader(expected_total: int, base_count: int) -> None:
    cfg_path = REPO / "configs" / "selfevo_fscil_sn55_co3d_11t.json"
    cfg = json.loads(cfg_path.read_text())
    args = {**cfg, **SMOKE_ARGS}
    ds = iCrossCO3D(args)
    ds.download_data()

    assert len(ds.class_order) == expected_total
    assert ds.train_data.shape[-1] == 6
    assert ds.test_data.shape[-1] == 6

    train_ids = set(np.unique(ds.train_targets))
    test_ids = set(np.unique(ds.test_targets))
    expected_ids = set(range(expected_total))
    assert expected_ids.issubset(train_ids), f"missing train {expected_ids - train_ids}"
    assert expected_ids.issubset(test_ids), f"missing test {expected_ids - test_ids}"

    novel_ids = set(range(base_count, expected_total))
    base_ids = set(range(base_count))
    assert base_ids.issubset(train_ids)
    assert novel_ids.issubset(train_ids)

    native_to_cross = ds._co3d_to_cross()
    for cross_id, cls_name in SHAPENET_CO3D_NOVEL.items():
        natives = [n for n, c in native_to_cross.items() if c == cross_id]
        assert len(natives) == 1

    print(
        f"[cross:shapenet_co3d] train={ds.train_data.shape}, test={ds.test_data.shape}, "
        f"base={len(base_ids)} novel={len(novel_ids)}"
    )


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
    assert task_sizes == expected_increments
    assert dm.nb_classes == expected_total
    assert int(dm._train_targets.max()) == expected_total - 1
    assert int(dm._test_targets.max()) == expected_total - 1
    print(
        f"[datamanager:{cfg['dataset']}] nb_tasks={dm.nb_tasks}, "
        f"task_sizes={task_sizes}, train={dm._train_data.shape}"
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
    assert labels == set(range(base_count))
    print(f"[task0:{cfg['dataset']}] train_n={len(task0)} classes={base_count} (base only)")


def main() -> None:
    print("=" * 72)
    print("CO3D FSCIL smoke test")
    print("=" * 72)

    _check_co3d_loader()

    for protocol, cfg_path, increments, total, base_count in CONFIGS:
        cfg = json.loads(cfg_path.read_text())
        _check_protocol(protocol, increments, total)
        if protocol == "shapenet_co3d":
            _check_cross_loader(total, base_count)
        _check_data_manager(cfg, increments, total)
        _check_task0_base_only(cfg, base_count)

    print("\nSMOKE TEST PASSED")


if __name__ == "__main__":
    main()
