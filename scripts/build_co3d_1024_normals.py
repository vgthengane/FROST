#!/usr/bin/env python3
"""Build CO3D FPS + normal caches for FSCIL-3D (microshape index layout).

Reads ``.pt`` point clouds under ``<data_root>/<category>/{train,test}/`` listed in
``third_party/microshape/data/index_files/co3d/`` and writes ModelNet-style pickles:

  co3d_train_1024pts_fps_normal.dat
  co3d_test_1024pts_fps_normal.dat
  co3d_class_names.txt

Assumes raw ``.pt`` tensors are ``[N, 3]`` (FSCIL-3D preprocessed FPS clouds may
already be 1024 points; we re-FPS if ``N != num_point`` for consistency).

Usage::

    aprun python scripts/build_co3d_1024_normals.py
    aprun python scripts/build_co3d_1024_normals.py --data-root _datasets/CO3D --force
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from datasets.modelnet_dataloader import farthest_point_sample, pc_normalize
from datasets.pointcloud_normals import with_normals

DEFAULT_DATA_ROOT = REPO / "_datasets" / "CO3D"
DEFAULT_INDEX_ROOT = REPO / "third_party" / "microshape" / "data" / "index_files" / "co3d"


def parse_index_split(index_root: Path, split: str) -> dict[str, int]:
    """Map relative ``category/train|test/file.pt`` paths to FSCIL class ids."""
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
            rel_path = parts[0]
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


def load_pt_points(path: Path) -> np.ndarray:
    """Load xyz from a microshape/FSCIL ``.pt`` point cloud."""
    try:
        obj = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        obj = torch.load(path, map_location="cpu")
    if isinstance(obj, torch.Tensor):
        arr = obj.detach().cpu().numpy()
    elif isinstance(obj, np.ndarray):
        arr = obj
    else:
        raise TypeError(f"Unsupported point type in {path}: {type(obj)}")
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] < 3:
        raise ValueError(f"Expected Nx3+ in {path}, got {arr.shape}")
    return arr[:, :3]


def prepare_points(point_set: np.ndarray, num_point: int, use_normals: bool) -> np.ndarray:
    if point_set.shape[0] >= num_point:
        point_set = farthest_point_sample(point_set, num_point)
    else:
        reps = int(np.ceil(num_point / max(point_set.shape[0], 1)))
        point_set = np.tile(point_set, (reps, 1))[:num_point]
    point_set = point_set.astype(np.float32, copy=False)
    point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])
    if use_normals:
        return with_normals(point_set)
    return point_set[:, 0:3]


def write_class_names(data_root: Path, samples: dict[str, int]) -> None:
    if not samples:
        return
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
    print(f"Wrote {out} ({len(names)} classes)")


def build_split_cache(
    *,
    data_root: Path,
    index_root: Path,
    split: str,
    num_point: int,
    use_normals: bool,
    force: bool,
) -> Path:
    suffix = "_normal" if use_normals else ""
    cache_path = data_root / f"co3d_{split}_{num_point}pts_fps{suffix}.dat"
    if cache_path.is_file() and not force:
        print(f"Skip {split}: cache exists ({cache_path})")
        return cache_path

    samples = parse_index_split(index_root, split)
    rel_paths = sorted(samples.keys())
    feat_dim = 6 if use_normals else 3
    points = np.zeros((len(rel_paths), num_point, feat_dim), dtype=np.float32)
    labels = np.zeros(len(rel_paths), dtype=np.int32)

    missing: list[str] = []
    for i, rel_path in enumerate(tqdm(rel_paths, desc=f"CO3D {split}")):
        src = data_root / rel_path
        if not src.is_file():
            missing.append(rel_path)
            continue
        xyz = load_pt_points(src)
        points[i] = prepare_points(xyz, num_point, use_normals)
        labels[i] = samples[rel_path]

    if missing:
        preview = "\n  ".join(missing[:8])
        raise FileNotFoundError(
            f"{len(missing)} indexed CO3D files missing under {data_root}.\n"
            f"First missing:\n  {preview}\n"
            "Run: bash scripts/download_co3d_fscil3d.sh"
        )

    tmp = cache_path.with_suffix(".dat.tmp")
    with tmp.open("wb") as f:
        pickle.dump([points, labels], f, protocol=4)
    tmp.replace(cache_path)
    print(f"Wrote {cache_path} — {len(rel_paths)} samples, shape {points.shape}")
    return cache_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build CO3D 1024 FPS + normal caches")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="CO3D root with <category>/{train,test}/*.pt",
    )
    parser.add_argument(
        "--index-root",
        type=Path,
        default=DEFAULT_INDEX_ROOT,
        help="FSCIL-3D CO3D index files (microshape)",
    )
    parser.add_argument("--num-point", type=int, default=1024)
    parser.add_argument("--no-normals", action="store_true")
    parser.add_argument("--force", action="store_true", help="Rebuild even if cache exists")
    args = parser.parse_args()

    data_root = args.data_root if args.data_root.is_absolute() else REPO / args.data_root
    index_root = args.index_root if args.index_root.is_absolute() else REPO / args.index_root
    data_root.mkdir(parents=True, exist_ok=True)

    if not index_root.is_dir():
        raise FileNotFoundError(f"CO3D index root not found: {index_root}")

    train_samples = parse_index_split(index_root, "train")
    test_samples = parse_index_split(index_root, "test")
    write_class_names(data_root, {**train_samples, **test_samples})

    use_normals = not args.no_normals
    build_split_cache(
        data_root=data_root,
        index_root=index_root,
        split="train",
        num_point=args.num_point,
        use_normals=use_normals,
        force=args.force,
    )
    build_split_cache(
        data_root=data_root,
        index_root=index_root,
        split="test",
        num_point=args.num_point,
        use_normals=use_normals,
        force=args.force,
    )


if __name__ == "__main__":
    main()
