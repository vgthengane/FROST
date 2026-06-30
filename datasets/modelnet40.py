"""ModelNet40 loader for the PILOT DataManager (3D point clouds)."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
from tqdm import tqdm

from datasets.modelnet_dataloader import ModelNetDataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]


class iModelNet40(object):
    use_path = False
    is_pointcloud = True
    train_trsf = []
    test_trsf = []
    common_trsf = []
    class_order = list(range(40))

    def __init__(self, args):
        self.args = args or {}
        self.train_data = None
        self.train_targets = None
        self.test_data = None
        self.test_targets = None

    def download_data(self):
        data_path = self.args.get(
            "data_path", str(REPO_ROOT / "_datasets" / "modelnet40_normal_resampled")
        )
        data_path = str(Path(data_path) if Path(data_path).is_absolute() else REPO_ROOT / data_path)

        loader_args = SimpleNamespace(
            num_point=int(self.args.get("num_point", 1024)),
            use_normals=bool(self.args.get("use_normals", True)),
            use_uniform_sample=bool(self.args.get("use_uniform_sample", True)),
            num_category=int(self.args.get("num_category", 40)),
        )

        train_loader = ModelNetDataLoader(
            root=data_path,
            args=loader_args,
            split="train",
            process_data=bool(self.args.get("process_data", True)),
        )
        test_loader = ModelNetDataLoader(
            root=data_path,
            args=loader_args,
            split="test",
            process_data=bool(self.args.get("process_data", True)),
        )
        self.train_data, self.train_targets = self._load_split(train_loader, "train")
        self.test_data, self.test_targets = self._load_split(test_loader, "test")

        if any(k in self.args for k in ("max_classes", "max_train_per_class", "max_test_per_class")):
            self._apply_subsample_limits()

    def _apply_subsample_limits(self):
        """Optional cap on classes / samples per class (set via --set in the launcher)."""
        seed_val = self.args.get("seed", 1993)
        if isinstance(seed_val, list):
            seed_val = seed_val[0]
        rng = np.random.default_rng(int(seed_val))

        max_classes = int(self.args.get("max_classes", 40))
        max_train = int(self.args.get("max_train_per_class", 10**9))
        max_test = int(self.args.get("max_test_per_class", 10**9))

        self.train_data, self.train_targets = self._subsample_split(
            self.train_data, self.train_targets, max_classes, max_train, rng
        )
        self.test_data, self.test_targets = self._subsample_split(
            self.test_data, self.test_targets, max_classes, max_test, rng
        )

    @staticmethod
    def _subsample_split(data, targets, max_classes, max_per_class, rng):
        keep_idx = []
        for cls in range(max_classes):
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
    def _load_split(loader, split_name):
        data_list, target_list = [], []
        for i in tqdm(range(len(loader)), desc=f"Load ModelNet40 {split_name}", leave=False):
            points, label = loader[i]
            if hasattr(points, "numpy"):
                points = points.numpy()
            data_list.append(points)
            target_list.append(int(label.item() if hasattr(label, "item") else label))
        return np.stack(data_list, axis=0), np.array(target_list, dtype=np.int64)
