"""CO3D FSCIL cache loader (``co3d_{split}_1024pts_fps_normal.dat``)."""

from __future__ import annotations

import os
import pickle

from pathlib import Path

import numpy as np
from torch.utils.data import Dataset

from datasets.co3d_official import DEFAULT_INDEX_ROOT, parse_index_split


def _scalar_label(label) -> int:
    if isinstance(label, np.ndarray):
        return int(label.reshape(-1)[0])
    return int(label)


class CO3DDataLoader(Dataset):
    """Load prebuilt CO3D FPS + normal caches indexed by FSCIL-3D sample lists."""

    def __init__(
        self,
        root: str,
        args,
        split: str = "train",
        process_data: bool = True,
        index_root: str | None = None,
    ):
        self.root = root
        self.npoints = args.num_point
        self.use_normals = args.use_normals
        self.split = split
        self.process_data = process_data

        idx_root = Path(index_root) if index_root else DEFAULT_INDEX_ROOT
        self.samples = parse_index_split(idx_root, split)
        self.rel_paths = sorted(self.samples.keys())

        norm_tag = "_normal" if self.use_normals else ""
        self.save_path = os.path.join(
            root, f"co3d_{split}_{self.npoints}pts_fps{norm_tag}.dat"
        )

        self.list_of_points = None
        self.list_of_labels = None

        if self.process_data:
            if not os.path.isfile(self.save_path):
                raise FileNotFoundError(
                    f"Missing CO3D cache: {self.save_path}\n"
                    "Run: aprun python datasets/preprocess/build_co3d_official_1024_normals.py"
                )
            with open(self.save_path, "rb") as f:
                points, labels = pickle.load(f)
            self.list_of_points = points
            self.list_of_labels = labels
            if len(self.list_of_points) != len(self.rel_paths):
                raise ValueError(
                    f"Cache size mismatch for {self.save_path}: "
                    f"{len(self.list_of_points)} vs {len(self.rel_paths)} indexed samples"
                )

    def __len__(self) -> int:
        return len(self.rel_paths)

    def __getitem__(self, index: int):
        if not self.process_data:
            raise NotImplementedError("CO3DDataLoader requires process_data=True (cached .dat)")
        point_set = self.list_of_points[index]
        label = _scalar_label(self.list_of_labels[index])
        return point_set, label
