"""ShapeNet-53 loader for the PILOT DataManager (3D point clouds)."""

import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from tqdm import tqdm

from datasets.shapenet55_dataloader import ShapeNet55DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]

# Smallest categories in ShapeNet-55; standard 53-class CIL benchmark (RL-CL3D / I3DOL).
DEFAULT_EXCLUDED_SYNSETS = ("02843684", "02954340")  # birdhouse, cap


class iShapeNet53(object):
    use_path = False
    is_pointcloud = True
    train_trsf = []
    test_trsf = []
    common_trsf = []
    class_order = None

    def __init__(self, args):
        self.args = args or {}
        self.train_data = None
        self.train_targets = None
        self.test_data = None
        self.test_targets = None

    def download_data(self):
        data_path = self.args.get("data_path", str(REPO_ROOT / "_datasets" / "pclouds" / "ShapeNet55"))
        data_path = str(Path(data_path) if Path(data_path).is_absolute() else REPO_ROOT / data_path)
        excluded = self.args.get("excluded_synsets", list(DEFAULT_EXCLUDED_SYNSETS))
        num_category = int(self.args.get("num_category", 53))

        loader_args = SimpleNamespace(
            num_point=int(self.args.get("num_point", 1024)),
            use_normals=bool(self.args.get("use_normals", True)),
            use_uniform_sample=bool(self.args.get("use_uniform_sample", True)),
        )
        cache_version = self.args.get("cache_version")
        if cache_version is not None and cache_version != "":
            cache_version = str(cache_version)
        else:
            cache_version = None

        train_loader = ShapeNet55DataLoader(
            root=data_path,
            args=loader_args,
            split="train",
            process_data=bool(self.args.get("process_data", True)),
            excluded_synsets=excluded,
            cache_version=cache_version,
        )
        test_loader = ShapeNet55DataLoader(
            root=data_path,
            args=loader_args,
            split="test",
            process_data=bool(self.args.get("process_data", True)),
            excluded_synsets=excluded,
            cache_version=cache_version,
        )
        self.train_data, self.train_targets = self._load_split(train_loader, "train")
        self.test_data, self.test_targets = self._load_split(test_loader, "test")

        n_classes = len(train_loader.classes)
        self.class_order = list(range(n_classes))
        if n_classes != num_category:
            raise ValueError(
                f"Expected {num_category} ShapeNet classes after exclusions, got {n_classes}. "
                f"Check excluded_synsets={excluded!r} and data under {data_path}."
            )

        protocol = self.args.get("fscil3d_protocol")
        if protocol:
            from datasets.fscil3d_index import fscil3d_class_order, protocol_summary

            self.class_order = fscil3d_class_order(protocol)
            logging.info(protocol_summary(protocol))
            logging.info("FSCIL-3D class_order: %s", self.class_order)

        if any(
            k in self.args
            for k in (
                "max_classes",
                "max_train_per_class",
                "max_test_per_class",
                "train_fraction",
                "test_fraction",
            )
        ):
            self._apply_subsample_limits()

    def _apply_subsample_limits(self):
        """Optional cap on classes / samples per class (set via --set in the launcher)."""
        seed_val = self.args.get("seed", 1993)
        if isinstance(seed_val, list):
            seed_val = seed_val[0]
        rng = np.random.default_rng(int(seed_val))

        max_classes = int(self.args.get("max_classes", self.args.get("num_category", 53)))
        max_train = int(self.args.get("max_train_per_class", 10**9))
        max_test = int(self.args.get("max_test_per_class", 10**9))
        train_fraction = self.args.get("train_fraction")
        test_fraction = self.args.get("test_fraction")
        if train_fraction is not None:
            train_fraction = float(train_fraction)
        if test_fraction is not None:
            test_fraction = float(test_fraction)

        self.train_data, self.train_targets = self._subsample_split(
            self.train_data,
            self.train_targets,
            max_classes,
            max_train,
            rng,
            fraction=train_fraction,
        )
        self.test_data, self.test_targets = self._subsample_split(
            self.test_data,
            self.test_targets,
            max_classes,
            max_test,
            rng,
            fraction=test_fraction,
        )

    @staticmethod
    def _class_sample_limit(n_available, max_per_class, fraction):
        if fraction is not None:
            return max(1, int(round(n_available * fraction)))
        return min(max_per_class, n_available)

    @staticmethod
    def _subsample_split(data, targets, max_classes, max_per_class, rng, fraction=None):
        keep_idx = []
        for cls in range(max_classes):
            cls_idx = np.where(targets == cls)[0]
            if len(cls_idx) == 0:
                continue
            n = iShapeNet53._class_sample_limit(len(cls_idx), max_per_class, fraction)
            chosen = rng.choice(cls_idx, size=n, replace=False)
            keep_idx.append(chosen)
        if not keep_idx:
            return data, targets
        keep_idx = np.concatenate(keep_idx)
        keep_idx.sort()
        return data[keep_idx], targets[keep_idx]

    @staticmethod
    def _load_split(loader, split_name):
        if isinstance(loader.list_of_points, np.ndarray):
            return loader.list_of_points, loader.list_of_labels.astype(np.int64)

        data_list, target_list = [], []
        for i in tqdm(range(len(loader)), desc=f"Load ShapeNet53 {split_name}", leave=False):
            points, label = loader[i]
            if hasattr(points, "numpy"):
                points = points.numpy()
            data_list.append(points)
            target_list.append(int(label.item() if hasattr(label, "item") else label))
        return np.stack(data_list, axis=0), np.array(target_list, dtype=np.int64)
