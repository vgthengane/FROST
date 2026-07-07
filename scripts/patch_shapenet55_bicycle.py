#!/usr/bin/env python3
"""Patch local ShapeNet55 with bicycle (synset 02834778) for shapenet_scanobjectnn.

Why this exists
---------------
The Point-BERT / RL-CL3D ShapeNet55 release uses ShapeNetCore **v2** (55 synsets).
It keeps ``bed`` (02818832) and drops ``bicycle`` (02834778). Microshape's
ShapeNet -> ScanObjectNN protocol (ShapeNetCore **v1**) requires bicycle at cross id 4.

This script:
  1. Imports bicycle point clouds (microshape ``.pt``, existing ``.npy``, or ShapeNet v1 OBJ)
  2. Removes ``bed`` entries from ``splits/{train,test}.txt`` (not used by the protocol)
  3. Adds bicycle ``.npy`` paths to the splits
  4. Deletes stale FPS / normal ``.dat`` caches so they rebuild on next load

Typical usage (microshape preprocessed bicycle class folder)::

  # After unzipping microshape dataset.zip under third_party/microshape/data/dataset/
  aprun python scripts/patch_shapenet55_bicycle.py \\
      --microshape-root third_party/microshape/data/dataset/v9-shapenet-scanobjectnn/4

Or from ShapeNetCore v1 OBJ meshes::

  aprun python scripts/patch_shapenet55_bicycle.py \\
      --shapenet-v1-synset /path/to/ShapeNetCore.v1/02834778
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

BICYCLE_SYNSET = "02834778"
BED_SYNSET = "02818832"
DEFAULT_SHAPENET_ROOT = REPO / "_datasets" / "ShapeNet55"
DEFAULT_MICROSHAPE_INDEX = (
    REPO / "third_party" / "microshape" / "data" / "index_files" / "shapenet_scanobjectnn"
)
MODEL_STEM_RE = re.compile(r"model_(\d+)\.pt$", re.IGNORECASE)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shapenet-root",
        type=Path,
        default=DEFAULT_SHAPENET_ROOT,
        help="ShapeNet55 root (contains shapenet_pc/ and splits/)",
    )
    parser.add_argument(
        "--microshape-root",
        type=Path,
        default=None,
        help="Folder with microshape bicycle class, e.g. .../v9-shapenet-scanobjectnn/4",
    )
    parser.add_argument(
        "--shapenet-v1-synset",
        type=Path,
        default=None,
        help="ShapeNetCore v1 synset directory, e.g. .../02834778/<model_id>/models/model_normalized.obj",
    )
    parser.add_argument(
        "--npy-source-dir",
        type=Path,
        default=None,
        help="Directory already containing 02834778-*.npy files",
    )
    parser.add_argument(
        "--microshape-index",
        type=Path,
        default=DEFAULT_MICROSHAPE_INDEX,
        help="microshape shapenet_scanobjectnn index root (used to pick train/test split)",
    )
    parser.add_argument(
        "--target-points",
        type=int,
        default=8192,
        help="Points saved per bicycle .npy (matches Point-BERT raw clouds)",
    )
    parser.add_argument(
        "--rebuild-cache",
        action="store_true",
        help="Rebuild shapenet55 FPS(+normal) caches after patching",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned changes without writing files",
    )
    return parser.parse_args()


def _npy_name(model_id: str) -> str:
    digest = hashlib.md5(f"bicycle-{model_id}".encode()).hexdigest()
    return f"{BICYCLE_SYNSET}-{digest}.npy"


def _upsample_points(xyz: np.ndarray, target: int) -> np.ndarray:
    xyz = np.asarray(xyz, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] < 3:
        raise ValueError(f"Expected [N,3+] array, got {xyz.shape}")
    xyz = xyz[:, :3]
    n = xyz.shape[0]
    if n == 0:
        raise ValueError("Empty point cloud")
    if n >= target:
        return xyz[:target]
    reps = int(np.ceil(target / n))
    tiled = np.tile(xyz, (reps, 1))[:target]
    return tiled


def _load_microshape_pt(pt_path: Path, target_points: int) -> np.ndarray:
    import torch

    obj = torch.load(pt_path, map_location="cpu", weights_only=False)
    if isinstance(obj, torch.Tensor):
        pts = obj.detach().cpu().numpy()
    elif isinstance(obj, np.ndarray):
        pts = obj
    elif isinstance(obj, (list, tuple)) and obj:
        pts = np.asarray(obj[0])
    else:
        raise TypeError(f"Unsupported microshape tensor type in {pt_path}: {type(obj)}")
    return _upsample_points(pts, target_points)


def _load_obj_mesh(obj_path: Path, target_points: int) -> np.ndarray:
    import trimesh

    mesh = trimesh.load(obj_path, force="mesh", process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    pts, _ = trimesh.sample.sample_surface(mesh, target_points)
    return np.asarray(pts, dtype=np.float64)


def _discover_v1_models(synset_dir: Path) -> list[tuple[str, Path]]:
    out: list[tuple[str, Path]] = []
    if not synset_dir.is_dir():
        return out
    for model_dir in sorted(synset_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        candidates = [
            model_dir / "models" / "model_normalized.obj",
            model_dir / "model.obj",
        ]
        for obj in candidates:
            if obj.is_file():
                out.append((model_dir.name, obj))
                break
    return out


def _discover_microshape_models(micro_root: Path) -> list[tuple[str, str, Path]]:
    """Return (split, model_id, pt_path)."""
    out: list[tuple[str, str, Path]] = []
    for split in ("train", "test"):
        split_dir = micro_root / split
        if not split_dir.is_dir():
            continue
        for pt in sorted(split_dir.glob("model_*.pt")):
            m = MODEL_STEM_RE.search(pt.name)
            if m is None:
                continue
            out.append((split, m.group(1), pt))
    return out


def _discover_npy_models(npy_dir: Path) -> list[tuple[str, str, Path]]:
    out: list[tuple[str, str, Path]] = []
    for npy in sorted(npy_dir.glob(f"{BICYCLE_SYNSET}-*.npy")):
        model_id = npy.stem.split("-", 1)[1]
        out.append(("train", model_id, npy))
    return out


def _read_microshape_split(index_root: Path) -> dict[str, set[str]]:
    """Map split -> model ids listed for bicycle (cross class 4) in session_1."""
    split_ids: dict[str, set[str]] = {"train": set(), "test": set()}
    for split in ("train", "test"):
        index_file = index_root / split / "session_1.txt"
        if not index_file.is_file():
            continue
        for line in index_file.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2 or int(parts[-1]) != 4:
                continue
            rel = parts[0]
            m = MODEL_STEM_RE.search(Path(rel).name)
            if m is None:
                continue
            split_ids[split].add(m.group(1))
    return split_ids


def _collect_bicycle_samples(
    args: argparse.Namespace,
) -> dict[str, list[tuple[str, Path]]]:
    """Return split -> list of (model_id, source_path)."""
    by_split: dict[str, list[tuple[str, Path]]] = {"train": [], "test": []}
    index_split = _read_microshape_split(args.microshape_index)

    if args.microshape_root is not None:
        micro_root = args.microshape_root
        if not micro_root.is_dir():
            raise FileNotFoundError(f"microshape bicycle folder not found: {micro_root}")
        discovered = _discover_microshape_models(micro_root)
        if not discovered:
            raise FileNotFoundError(f"No model_*.pt under {micro_root}/{{train,test}}")
        for split, model_id, pt_path in discovered:
            if index_split[split] and model_id not in index_split[split]:
                continue
            by_split[split].append((model_id, pt_path))
        if not by_split["train"] and not by_split["test"]:
            for split, model_id, pt_path in discovered:
                by_split[split].append((model_id, pt_path))

    elif args.shapenet_v1_synset is not None:
        models = _discover_v1_models(args.shapenet_v1_synset)
        if not models:
            raise FileNotFoundError(f"No OBJ models under {args.shapenet_v1_synset}")
        train_ids = index_split["train"] or None
        test_ids = index_split["test"] or None
        for idx, (model_id, obj_path) in enumerate(models):
            if train_ids and model_id in train_ids:
                by_split["train"].append((model_id, obj_path))
            elif test_ids and model_id in test_ids:
                by_split["test"].append((model_id, obj_path))
            elif train_ids is None and test_ids is None:
                by_split["train" if idx % 5 else "test"].append((model_id, obj_path))

    elif args.npy_source_dir is not None:
        for split, model_id, npy_path in _discover_npy_models(args.npy_source_dir):
            by_split[split].append((model_id, npy_path))
    else:
        raise ValueError(
            "Provide one of --microshape-root, --shapenet-v1-synset, or --npy-source-dir"
        )

    if not by_split["train"]:
        raise RuntimeError("No bicycle train samples collected")
    if not by_split["test"]:
        raise RuntimeError("No bicycle test samples collected")
    return by_split


def _write_npy_from_source(src: Path, dst: Path, target_points: int) -> None:
    if src.suffix.lower() == ".npy":
        pts = np.load(src)
        pts = _upsample_points(pts, target_points)
    elif src.suffix.lower() == ".pt":
        pts = _load_microshape_pt(src, target_points)
    elif src.suffix.lower() == ".obj":
        pts = _load_obj_mesh(src, target_points)
    else:
        raise ValueError(f"Unsupported source type: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.save(dst, pts.astype(np.float64, copy=False))


def _patch_split_file(split_path: Path, bicycle_names: list[str], dry_run: bool) -> tuple[int, int]:
    lines = split_path.read_text().splitlines()
    kept = [ln for ln in lines if ln.strip() and not ln.strip().startswith(f"{BED_SYNSET}-")]
    removed_bed = sum(1 for ln in lines if ln.strip().startswith(f"{BED_SYNSET}-"))
    existing = {ln.strip() for ln in kept}
    added = [name for name in bicycle_names if name not in existing]
    new_lines = kept + added
    if not dry_run:
        split_path.write_text("\n".join(new_lines) + "\n")
    return removed_bed, len(added)


def _delete_caches(root: Path, dry_run: bool) -> list[Path]:
    removed: list[Path] = []
    for path in sorted(root.glob("shapenet55_*_1024pts*.dat")):
        removed.append(path)
        if not dry_run:
            path.unlink()
    return removed


def _rebuild_caches(root: Path) -> None:
    from datasets.shapenet55_dataloader import ShapeNet55DataLoader

    loader_args = SimpleNamespace(
        num_point=1024,
        use_normals=True,
        use_uniform_sample=True,
    )
    for split in ("train", "test"):
        print(f"Rebuilding cache for split={split} ...")
        ShapeNet55DataLoader(
            root=str(root),
            args=loader_args,
            split=split,
            process_data=True,
            excluded_synsets=[],
            force_build_from_raw=True,
        )


def main() -> None:
    args = _parse_args()
    root = args.shapenet_root if args.shapenet_root.is_absolute() else REPO / args.shapenet_root
    pc_dir = root / "shapenet_pc"
    splits_dir = root / "splits"
    if not pc_dir.is_dir() or not splits_dir.is_dir():
        raise FileNotFoundError(f"Invalid ShapeNet55 root: {root}")

    samples = _collect_bicycle_samples(args)
    print(
        f"Collected bicycle samples: train={len(samples['train'])}, test={len(samples['test'])}"
    )

    split_to_names: dict[str, list[str]] = {"train": [], "test": []}
    for split in ("train", "test"):
        for model_id, src in samples[split]:
            name = _npy_name(model_id)
            dst = pc_dir / name
            print(f"  [{split}] {src.name} -> {name}")
            if not args.dry_run:
                _write_npy_from_source(src, dst, args.target_points)
            split_to_names[split].append(name)

    for split in ("train", "test"):
        split_path = splits_dir / f"{split}.txt"
        removed, added = _patch_split_file(split_path, split_to_names[split], args.dry_run)
        print(f"{split}.txt: removed bed={removed}, added bicycle={added}")

    caches = _delete_caches(root, args.dry_run)
    if caches:
        print(f"Removed {len(caches)} stale cache file(s)")
    elif not args.dry_run:
        print("No stale shapenet55 cache files found")

    if args.rebuild_cache and not args.dry_run:
        _rebuild_caches(root)

    print("Done. Re-run scripts/smoke_test_cross_scanobjectnn.py to verify shapenet_scanobjectnn.")


if __name__ == "__main__":
    main()
