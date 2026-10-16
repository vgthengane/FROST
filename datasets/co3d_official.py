"""Official CO3D (Facebook) → FSCIL-3D microshape layout conversion helpers.

Maps index paths like ``apple/train/110_13052_23173.pt`` to official sequence
folders ``<official_root>/apple/110_13052_23173/pointcloud.ply``, then FPS
subsamples to 1024 points and estimates normals (ModelNet-style [N, 6]).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import open3d as o3d

from datasets.modelnet_dataloader import farthest_point_sample, pc_normalize
from datasets.pointcloud_normals import with_normals

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OFFICIAL_ROOT = REPO_ROOT / "data" / "co3d"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "_datasets" / "pclouds" / "CO3D"
DEFAULT_INDEX_ROOT = REPO_ROOT / "datasets" / "session_pclouds" / "co3d"


def normalize_index_rel_path(rel_path: str) -> str:
    """Strip microshape version prefix (e.g. ``v10-co3d/apple/...`` → ``apple/...``)."""
    if rel_path.startswith("v10-co3d/"):
        return rel_path[len("v10-co3d/") :]
    return rel_path


def parse_index_split(index_root: Path, split: str) -> dict[str, int]:
    """Map relative ``category/{train,test}/seq.pt`` paths to FSCIL class ids."""
    split_dir = index_root / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Missing CO3D index split: {split_dir}")

    samples: dict[str, int] = {}
    for session_file in sorted(split_dir.glob("session_*.txt")):
        for line in session_file.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            rel_path = normalize_index_rel_path(parts[0])
            label = int(parts[-1])
            if rel_path in samples and samples[rel_path] != label:
                raise ValueError(
                    f"Inconsistent label for {rel_path}: "
                    f"{samples[rel_path]} vs {label} in {session_file.name}"
                )
            samples[rel_path] = label
    if not samples:
        raise ValueError(f"No samples parsed under {split_dir}")
    return samples


def index_rel_to_ply(official_root: Path, rel_path: str) -> Path:
    """``apple/test/110_13051_23361.pt`` → ``.../apple/110_13051_23361/pointcloud.ply``."""
    parts = rel_path.split("/")
    if len(parts) != 3:
        raise ValueError(f"Expected category/split/file.pt, got {rel_path!r}")
    category, _split, filename = parts
    seq_id = filename.removesuffix(".pt")
    return official_root / category / seq_id / "pointcloud.ply"


def load_ply_xyz(path: Path) -> np.ndarray:
    """Load xyz from an official CO3D ``pointcloud.ply``."""
    pcd = o3d.io.read_point_cloud(str(path))
    xyz = np.asarray(pcd.points, dtype=np.float32)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or xyz.shape[0] < 3:
        raise ValueError(f"Invalid point cloud in {path}: shape={getattr(xyz, 'shape', None)}")
    return xyz


def presubsample_for_fps(xyz: np.ndarray, max_input: int, seed: int | None = None) -> np.ndarray:
    """Random cap before FPS on million-point CO3D meshes (keeps FPS tractable)."""
    n = xyz.shape[0]
    if n <= max_input:
        return xyz
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=max_input, replace=False)
    return xyz[idx]


def prepare_points(
    point_set: np.ndarray,
    num_point: int,
    use_normals: bool,
    *,
    max_fps_input: int = 50_000,
    seed: int | None = None,
) -> np.ndarray:
    """Normalize, FPS to ``num_point``, optionally append estimated normals."""
    xyz = point_set[:, :3].astype(np.float32, copy=True)
    xyz = presubsample_for_fps(xyz, max_fps_input, seed=seed)
    if xyz.shape[0] >= num_point:
        xyz = farthest_point_sample(xyz, num_point)
    else:
        reps = int(np.ceil(num_point / max(xyz.shape[0], 1)))
        xyz = np.tile(xyz, (reps, 1))[:num_point]
    xyz[:, 0:3] = pc_normalize(xyz[:, 0:3])
    if use_normals:
        return with_normals(xyz)
    return xyz[:, 0:3]


def write_class_names(data_root: Path, samples: dict[str, int]) -> Path:
    """Write ``co3d_class_names.txt`` aligned with FSCIL class ids."""
    if not samples:
        raise ValueError("No samples for class name table")
    max_id = max(samples.values())
    names = [""] * (max_id + 1)
    for rel_path, label in samples.items():
        cat = rel_path.split("/")[0]
        if names[label] and names[label] != cat:
            raise ValueError(f"Label {label} maps to {names[label]!r} and {cat!r}")
        names[label] = cat
    missing = [i for i, n in enumerate(names) if not n]
    if missing:
        raise ValueError(f"Missing class names for ids: {missing[:10]}")
    out = data_root / "co3d_class_names.txt"
    out.write_text("\n".join(names) + "\n")
    return out
