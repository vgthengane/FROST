"""ScanObjectNN loader for the PILOT DataManager (point clouds)."""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]


class iScanObjectNN(object):
    use_path = False
    is_pointcloud = True
    train_trsf = []
    test_trsf = []
    common_trsf = []
    class_order = list(range(15))

    def __init__(self, args):
        self.args = args or {}
        self.train_data = None
        self.train_targets = None
        self.test_data = None
        self.test_targets = None

    def download_data(self):
        data_path = self.args.get(
            "scanobjectnn_data_path",
            self.args.get(
                "data_path",
                str(REPO_ROOT / "_datasets" / "pclouds" / "ScanObjNN" / "main_split_nobg"),
            ),
        )
        root = Path(data_path) if Path(data_path).is_absolute() else REPO_ROOT / data_path
        root = root.resolve()

        num_point = int(self.args.get("num_point", 1024))
        use_normals = bool(self.args.get("use_normals", True))
        suffix = "_normal" if use_normals else ""
        train_file = root / f"scanobjectnn_train_{num_point}pts_fps{suffix}.dat"
        test_file = root / f"scanobjectnn_test_{num_point}pts_fps{suffix}.dat"

        if not train_file.is_file() or not test_file.is_file():
            raise FileNotFoundError(
                "ScanObjectNN cache files not found. Expected:\n"
                f"  {train_file}\n"
                f"  {test_file}\n"
                "Build them with: aprun python datasets/preprocess/build_scanobjectnn_1024_normals.py"
            )

        self.train_data, self.train_targets = self._load_cache(train_file)
        self.test_data, self.test_targets = self._load_cache(test_file)
        self.class_order = list(range(len(np.unique(self.train_targets))))

        protocol = self.args.get("fscil3d_protocol")
        if protocol:
            from datasets.fscil3d_index import fscil3d_class_order, protocol_summary

            self.class_order = fscil3d_class_order(protocol)
            logging.info(protocol_summary(protocol))
            logging.info("FSCIL-3D class_order: %s", self.class_order)

        if any(k in self.args for k in ("max_classes", "max_train_per_class", "max_test_per_class")):
            self._apply_subsample_limits()

    def _apply_subsample_limits(self):
        """Optional cap on classes / samples per class (smoke tests, debugging)."""
        seed_val = self.args.get("seed", 1993)
        if isinstance(seed_val, list):
            seed_val = seed_val[0]
        rng = np.random.default_rng(int(seed_val))

        max_classes = int(self.args.get("max_classes", len(self.class_order)))
        max_train = int(self.args.get("max_train_per_class", 10**9))
        max_test = int(self.args.get("max_test_per_class", 10**9))

        active_classes = list(self.class_order)[:max_classes]
        self.train_data, self.train_targets = self._subsample_split(
            self.train_data, self.train_targets, active_classes, max_train, rng
        )
        self.test_data, self.test_targets = self._subsample_split(
            self.test_data, self.test_targets, active_classes, max_test, rng
        )
        self.class_order = active_classes

    @staticmethod
    def _subsample_split(data, targets, class_ids, max_per_class, rng):
        keep_idx = []
        for cls in class_ids:
            cls_idx = np.where(targets == cls)[0]
            if len(cls_idx) == 0:
                continue
            n = min(max_per_class, len(cls_idx))
            chosen = rng.choice(cls_idx, size=n, replace=False)
            keep_idx.append(chosen)
        if not keep_idx:
            return data, targets
        keep_idx = np.concatenate(keep_idx)
        keep_idx.sort()
        return data[keep_idx], targets[keep_idx]

    @staticmethod
    def _load_cache(path: Path) -> tuple[np.ndarray, np.ndarray]:
        with path.open("rb") as f:
            payload = pickle.load(f)
        if not isinstance(payload, list) or len(payload) != 2:
            raise ValueError(f"Unexpected ScanObjectNN cache structure in {path}")
        data, labels = payload
        data = np.asarray(data, dtype=np.float32)
        labels = np.asarray(labels, dtype=np.int64)
        return data, labels
