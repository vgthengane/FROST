"""ShapeNet-55 point cloud loader (.npy) for ShapeNet-53 CIL."""

import os
import pickle
import warnings

import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm

from datasets.modelnet_dataloader import farthest_point_sample, pc_normalize
from datasets.pointcloud_normals import with_normals

warnings.filterwarnings("ignore")

# Version tag for FPS-derived normal caches (xyz FPS .dat → Open3D normals).
# Raw-.npy builds use cache_version=None (no suffix).
FPS_NORMAL_CACHE_VERSION = "v0"


def _feat_dim(use_normals):
    return 6 if use_normals else 3


def _scalar_label(label):
    if isinstance(label, np.ndarray):
        return int(label.reshape(-1)[0])
    return int(label)


class ShapeNet55DataLoader(Dataset):
    def __init__(
        self,
        root,
        args,
        split="train",
        process_data=False,
        excluded_synsets=None,
        cache_version=None,
        force_build_from_raw=False,
    ):
        self.root = root
        self.npoints = args.num_point
        self.process_data = process_data
        self.uniform = args.use_uniform_sample
        self.use_normals = args.use_normals
        self.cache_version = cache_version
        self.force_build_from_raw = force_build_from_raw
        self.split = split
        excluded = set(excluded_synsets or [])
        split_file = os.path.join(root, "splits", f"{split}.txt")
        pc_dir = os.path.join(root, "shapenet_pc")

        entries = []
        synsets = set()
        with open(split_file, "r") as f:
            for line in f:
                name = line.strip()
                if not name:
                    continue
                synset = name.split("-")[0]
                if synset in excluded:
                    continue
                synsets.add(synset)
                entries.append((synset, os.path.join(pc_dir, name)))

        self.classes = {syn: idx for idx, syn in enumerate(sorted(synsets))}
        self.datapath = entries
        self.tag = "shapenet53" if excluded else "shapenet55"
        print("The size of %s data is %d" % (split, len(self.datapath)))

        self.save_path = os.path.join(root, self._cache_filename(self.tag, split))

        self.list_of_points = None
        self.list_of_labels = None

        if self.process_data:
            if self.force_build_from_raw:
                print("Building cache from raw .npy: %s ..." % self.save_path)
                self._build_cache()
            elif os.path.exists(self.save_path):
                print("Load processed data from %s..." % self.save_path)
                self._load_cache()
            elif self.use_normals:
                fps_cache = self._resolve_fps_cache_path()
                if fps_cache is not None:
                    print("Building normal cache from FPS cache %s ..." % fps_cache)
                    self._build_normal_cache_from_fps(fps_cache)
                elif self.cache_version:
                    raise FileNotFoundError(
                        f"Missing FPS-derived normal cache: {self.save_path}\n"
                        f"No xyz FPS cache found under {self.root}.\n"
                        "Run: aprun python scripts/build_shapenet_normals_from_fps_cache.py"
                    )
                else:
                    print("Processing data %s (only running in the first time)..." % self.save_path)
                    self._build_cache()
            else:
                print("Processing data %s (only running in the first time)..." % self.save_path)
                self._build_cache()

    def _cache_stem(self, tag: str, split: str) -> str:
        norm_tag = "_normal" if self.use_normals else ""
        if self.uniform:
            return f"{tag}_{split}_{self.npoints}pts_fps{norm_tag}"
        return f"{tag}_{split}_{self.npoints}pts{norm_tag}"

    def _cache_filename(self, tag: str, split: str) -> str:
        stem = self._cache_stem(tag, split)
        if self.cache_version:
            return f"{stem}_{self.cache_version}.dat"
        return f"{stem}.dat"

    def _legacy_cache_filename(self, tag: str, split: str) -> str:
        """Pre-v0 caches (no version suffix)."""
        return f"{self._cache_stem(tag, split)}.dat"

    def _fps_cache_path(self) -> str:
        return self.save_path.replace("_normal", "")

    def _xyz_cache_stem(self, split: str) -> str:
        if self.uniform:
            return f"{self.tag}_{split}_{self.npoints}pts_fps"
        return f"{self.tag}_{split}_{self.npoints}pts"

    def _resolve_fps_cache_path(self) -> str | None:
        """FPS xyz cache used to derive normals (v0 first, then legacy)."""
        stem = self._xyz_cache_stem(self.split)
        candidates = []
        if self.cache_version:
            candidates.append(os.path.join(self.root, f"{stem}_{self.cache_version}.dat"))
        candidates.append(os.path.join(self.root, f"{stem}.dat"))
        for path in candidates:
            if os.path.exists(path):
                return path
        return None

    def _build_normal_cache_from_fps(self, fps_cache_path: str) -> None:
        """Estimate normals on cached FPS xyz clouds (no .npy re-read)."""
        with open(fps_cache_path, "rb") as f:
            points, labels = pickle.load(f)
        if len(points) != len(self.datapath):
            raise ValueError(
                f"FPS cache size mismatch for {fps_cache_path}: "
                f"{len(points)} vs {len(self.datapath)} samples"
            )
        n = len(points)
        out = np.zeros((n, self.npoints, 6), dtype=np.float32)
        for i in tqdm(range(n), total=n, desc="Normals from FPS cache"):
            out[i] = with_normals(np.asarray(points[i], dtype=np.float32))
        tmp_path = f"{self.save_path}.tmp"
        with open(tmp_path, "wb") as f:
            pickle.dump([out, labels], f, protocol=4)
        os.replace(tmp_path, self.save_path)
        self.list_of_points = out
        self.list_of_labels = labels

    def _load_cache(self):
        with open(self.save_path, "rb") as f:
            points, labels = pickle.load(f)
        self.list_of_points = points
        self.list_of_labels = labels
        if len(self.list_of_points) != len(self.datapath):
            raise ValueError(
                f"Cache size mismatch for {self.save_path}: "
                f"{len(self.list_of_points)} vs {len(self.datapath)} samples"
            )

    def _prepare_points(self, point_set: np.ndarray) -> np.ndarray:
        if point_set.ndim != 2 or point_set.shape[1] < 3:
            raise ValueError(f"Expected Nx3+ array, got {point_set.shape}")

        if self.uniform:
            point_set = farthest_point_sample(point_set, self.npoints)
        else:
            point_set = point_set[: self.npoints]

        point_set = point_set.astype(np.float32)
        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])
        if self.use_normals:
            return with_normals(point_set)
        return point_set[:, 0:3]

    def _build_cache(self):
        n = len(self.datapath)
        c = _feat_dim(self.use_normals)
        points = np.zeros((n, self.npoints, c), dtype=np.float32)
        labels = np.zeros(n, dtype=np.int32)

        for index in tqdm(range(n), total=n, desc="Normals from raw .npy"):
            synset, path = self.datapath[index]
            point_set = np.load(path).astype(np.float32)
            point_set = self._prepare_points(point_set)
            points[index] = point_set
            labels[index] = self.classes[synset]

        tmp_path = f"{self.save_path}.tmp"
        with open(tmp_path, "wb") as f:
            pickle.dump([points, labels], f, protocol=4)
        os.replace(tmp_path, self.save_path)
        self.list_of_points = points
        self.list_of_labels = labels

    def __len__(self):
        return len(self.datapath)

    def _get_item(self, index):
        if self.process_data:
            point_set = self.list_of_points[index]
            label = _scalar_label(self.list_of_labels[index])
        else:
            synset, path = self.datapath[index]
            label = self.classes[synset]
            point_set = np.load(path).astype(np.float32)
            point_set = self._prepare_points(point_set)
        return point_set, label

    def __getitem__(self, index):
        return self._get_item(index)
