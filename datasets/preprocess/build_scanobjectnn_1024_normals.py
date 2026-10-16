#!/usr/bin/env python3
"""Build ScanObjectNN 1024 FPS + normals caches from official h5 files.

Reads:
  <data-root>/training_objectdataset.h5
  <data-root>/test_objectdataset.h5

Writes:
  <output-root>/scanobjectnn_train_1024pts_fps_normal.dat
  <output-root>/scanobjectnn_test_1024pts_fps_normal.dat
  <output-root>/scanobjectnn_class_names.txt

Usage:
  aprun python datasets/preprocess/build_scanobjectnn_1024_normals.py
  aprun python datasets/preprocess/build_scanobjectnn_1024_normals.py \
      --data-root _datasets/pclouds/ScanObjNN/main_split_nobg --force
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from datasets.modelnet_dataloader import farthest_point_sample, pc_normalize
from datasets.pointcloud_normals import with_normals

DEFAULT_DATA_ROOT = REPO / "_datasets" / "pclouds" / "ScanObjNN" / "main_split_nobg"
DEFAULT_OUTPUT_ROOT = DEFAULT_DATA_ROOT

SCANOBJECTNN_CLASS_NAMES = [
    "bag",
    "bin",
    "box",
    "cabinet",
    "chair",
    "desk",
    "display",
    "door",
    "shelf",
    "table",
    "bed",
    "pillow",
    "sink",
    "sofa",
    "toilet",
]


def _load_h5(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as f:
        if "data" not in f or "label" not in f:
            raise KeyError(f"{path} must contain datasets: data, label")
        points = np.asarray(f["data"][:], dtype=np.float32)
        labels = np.asarray(f["label"][:], dtype=np.int64)
    if points.ndim != 3 or points.shape[2] < 3:
        raise ValueError(f"Expected [N, P, 3+] data in {path}, got {points.shape}")
    if labels.ndim != 1 or labels.shape[0] != points.shape[0]:
        raise ValueError(
            f"Label shape mismatch in {path}: labels {labels.shape}, points {points.shape}"
        )
    return points[:, :, :3], labels


def _prepare_points(point_set: np.ndarray, num_point: int, use_normals: bool) -> np.ndarray:
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


def _build_split_cache(
    *,
    data_root: Path,
    output_root: Path,
    split: str,
    num_point: int,
    use_normals: bool,
    force: bool,
) -> Path:
    src_name = "training_objectdataset.h5" if split == "train" else "test_objectdataset.h5"
    src_path = data_root / src_name
    if not src_path.is_file():
        raise FileNotFoundError(f"Missing source h5: {src_path}")

    suffix = "_normal" if use_normals else ""
    cache_path = output_root / f"scanobjectnn_{split}_{num_point}pts_fps{suffix}.dat"
    if cache_path.is_file() and not force:
        print(f"Skip {split}: cache exists ({cache_path})")
        return cache_path

    raw_points, raw_labels = _load_h5(src_path)
    feat_dim = 6 if use_normals else 3
    points = np.zeros((raw_points.shape[0], num_point, feat_dim), dtype=np.float32)
    labels = raw_labels.astype(np.int32, copy=False)

    for i in tqdm(range(raw_points.shape[0]), desc=f"ScanObjectNN {split}"):
        points[i] = _prepare_points(raw_points[i], num_point, use_normals)

    tmp = cache_path.with_suffix(".dat.tmp")
    with tmp.open("wb") as f:
        pickle.dump([points, labels], f, protocol=4)
    tmp.replace(cache_path)
    print(f"Wrote {cache_path} — samples={len(labels)}, shape={points.shape}")
    return cache_path


def _write_class_names(output_root: Path) -> Path:
    class_file = output_root / "scanobjectnn_class_names.txt"
    class_file.write_text("\n".join(SCANOBJECTNN_CLASS_NAMES) + "\n")
    print(f"Wrote {class_file} ({len(SCANOBJECTNN_CLASS_NAMES)} classes)")
    return class_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Build ScanObjectNN 1024 FPS + normals caches")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Directory containing training_objectdataset.h5/test_objectdataset.h5",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory where .dat caches will be written",
    )
    parser.add_argument("--num-point", type=int, default=1024)
    parser.add_argument("--no-normals", action="store_true")
    parser.add_argument("--force", action="store_true", help="Rebuild cache files even if present")
    args = parser.parse_args()

    data_root = args.data_root if args.data_root.is_absolute() else REPO / args.data_root
    output_root = args.output_root if args.output_root.is_absolute() else REPO / args.output_root
    if not data_root.is_dir():
        raise FileNotFoundError(f"ScanObjectNN root not found: {data_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    use_normals = not args.no_normals
    _write_class_names(output_root)
    _build_split_cache(
        data_root=data_root,
        output_root=output_root,
        split="train",
        num_point=args.num_point,
        use_normals=use_normals,
        force=args.force,
    )
    _build_split_cache(
        data_root=data_root,
        output_root=output_root,
        split="test",
        num_point=args.num_point,
        use_normals=use_normals,
        force=args.force,
    )


if __name__ == "__main__":
    main()
