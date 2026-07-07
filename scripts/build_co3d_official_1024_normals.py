#!/usr/bin/env python3
"""Build CO3D 1024 FPS + normal caches from official Facebook CO3D ``pointcloud.ply``.

Pipeline:
  1. Map FSCIL-3D index entries → ``data/co3d/<cat>/<seq>/pointcloud.ply``
  2. Optional: export microshape-layout ``.pt`` xyz caches under ``_datasets/CO3D/``
  3. Write ``co3d_{train,test}_1024pts_fps_normal.dat`` + ``co3d_class_names.txt``

The single-sequence CO3D subset only covers a fraction of indexed sequences.
Use ``--allow-partial`` to build caches for available ply files only (smoke / dev).
Full FSCIL training needs the complete CO3D download (not ``--single_sequence_subset``).

Usage::

    aprun --image climb3d_pp.sif python scripts/build_co3d_official_1024_normals.py
    aprun --image climb3d_pp.sif python scripts/build_co3d_official_1024_normals.py --allow-partial
    bash scripts/run_co3d_official_build.sh --screen
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from datasets.co3d_official import (  # noqa: E402
    DEFAULT_INDEX_ROOT,
    DEFAULT_OFFICIAL_ROOT,
    DEFAULT_OUTPUT_ROOT,
    index_rel_to_ply,
    load_ply_xyz,
    parse_index_split,
    prepare_points,
    write_class_names,
)


def export_pt(xyz_fps: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(torch.from_numpy(xyz_fps[:, :3].astype(np.float32)), path)


def build_split(
    *,
    official_root: Path,
    output_root: Path,
    index_root: Path,
    split: str,
    num_point: int,
    use_normals: bool,
    export_pt_files: bool,
    allow_partial: bool,
    max_fps_input: int,
    force: bool,
    seed: int,
) -> dict:
    suffix = "_normal" if use_normals else ""
    cache_path = output_root / f"co3d_{split}_{num_point}pts_fps{suffix}.dat"
    manifest_path = output_root / f"co3d_{split}_build_manifest.json"

    if cache_path.is_file() and not force:
        print(f"Skip {split}: cache exists ({cache_path})")
        return json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}

    samples = parse_index_split(index_root, split)
    rel_paths = sorted(samples.keys())
    feat_dim = 6 if use_normals else 3
    points = np.zeros((len(rel_paths), num_point, feat_dim), dtype=np.float32)
    labels = np.zeros(len(rel_paths), dtype=np.int32)

    missing: list[str] = []
    no_ply: list[str] = []
    built = 0

    for i, rel_path in enumerate(tqdm(rel_paths, desc=f"CO3D official {split}")):
        labels[i] = samples[rel_path]
        ply_path = index_rel_to_ply(official_root, rel_path)
        if not ply_path.is_file():
            no_ply.append(rel_path)
            continue
        try:
            xyz = load_ply_xyz(ply_path)
            pts = prepare_points(
                xyz,
                num_point,
                use_normals,
                max_fps_input=max_fps_input,
                seed=seed + i,
            )
        except Exception as exc:
            missing.append(f"{rel_path}: {exc}")
            continue

        points[i] = pts
        built += 1
        if export_pt_files:
            export_pt(pts, output_root / rel_path)

    manifest = {
        "split": split,
        "indexed": len(rel_paths),
        "built": built,
        "no_pointcloud_ply": len(no_ply),
        "errors": len(missing),
    }

    if not allow_partial and (no_ply or missing):
        preview = "\n  ".join((no_ply[:5] + [f"ERR: {m}" for m in missing[:3]]))
        raise FileNotFoundError(
            f"CO3D {split}: {len(no_ply)} sequences without pointcloud.ply, "
            f"{len(missing)} load errors.\n"
            f"First issues:\n  {preview}\n"
            "Download full CO3D (not single-sequence subset) or pass --allow-partial."
        )

    if built == 0:
        if allow_partial:
            manifest["skipped_cache"] = True
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
            print(
                f"Skip {split} cache: 0/{len(rel_paths)} samples had pointcloud.ply "
                f"(manifest: {manifest_path})"
            )
            return manifest
        raise RuntimeError(
            f"CO3D {split}: no samples built. Check official_root={official_root}"
        )

    tmp = cache_path.with_suffix(".dat.tmp")
    with tmp.open("wb") as f:
        pickle.dump([points, labels], f, protocol=4)
    tmp.replace(cache_path)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        f"Wrote {cache_path} — indexed={len(rel_paths)}, built={built}, "
        f"no_ply={len(no_ply)}, errors={len(missing)}"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build CO3D 1024 FPS + normals from official pointcloud.ply"
    )
    parser.add_argument(
        "--official-root",
        type=Path,
        default=DEFAULT_OFFICIAL_ROOT,
        help="Official CO3D root (category/seq/pointcloud.ply)",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Output root for .dat caches (and optional .pt exports)",
    )
    parser.add_argument(
        "--index-root",
        type=Path,
        default=DEFAULT_INDEX_ROOT,
        help="FSCIL-3D CO3D index files",
    )
    parser.add_argument("--num-point", type=int, default=1024)
    parser.add_argument("--no-normals", action="store_true")
    parser.add_argument("--export-pt", action="store_true", help="Also write microshape .pt files")
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Allow missing pointcloud.ply (zeros in cache for missing rows)",
    )
    parser.add_argument(
        "--max-fps-input",
        type=int,
        default=50_000,
        help="Random presubsample cap before FPS on large ply clouds",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--seed", type=int, default=2008)
    args = parser.parse_args()

    official_root = args.official_root if args.official_root.is_absolute() else REPO / args.official_root
    output_root = args.output_root if args.output_root.is_absolute() else REPO / args.output_root
    index_root = args.index_root if args.index_root.is_absolute() else REPO / args.index_root

    if not official_root.is_dir():
        raise FileNotFoundError(f"Official CO3D root not found: {official_root}")
    if not index_root.is_dir():
        raise FileNotFoundError(f"CO3D index root not found: {index_root}")

    output_root.mkdir(parents=True, exist_ok=True)
    use_normals = not args.no_normals

    train_samples = parse_index_split(index_root, "train")
    test_samples = parse_index_split(index_root, "test")
    names_path = write_class_names(output_root, {**train_samples, **test_samples})
    print(f"Wrote {names_path}")

    summary = {}
    for split in ("train", "test"):
        summary[split] = build_split(
            official_root=official_root,
            output_root=output_root,
            index_root=index_root,
            split=split,
            num_point=args.num_point,
            use_normals=use_normals,
            export_pt_files=args.export_pt,
            allow_partial=args.allow_partial,
            max_fps_input=args.max_fps_input,
            force=args.force,
            seed=args.seed,
        )

    summary_path = output_root / "co3d_build_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Done — summary: {summary_path}")


if __name__ == "__main__":
    main()
