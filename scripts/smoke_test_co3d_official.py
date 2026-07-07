#!/usr/bin/env python3
"""Smoke test for official CO3D single-sequence subset under data/co3d/."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import open3d as o3d

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from datasets.modelnet_dataloader import farthest_point_sample, pc_normalize
from datasets.pointcloud_normals import with_normals

CO3D = REPO / "data" / "co3d"
EXPECTED_CATEGORIES = 48  # single-sequence subset: 48 cats × 2 zips = 96 archives


def main() -> None:
    if not CO3D.is_dir():
        raise SystemExit(f"Missing CO3D root: {CO3D}")

    cats = sorted(p.name for p in CO3D.iterdir() if p.is_dir() and not p.name.startswith("_"))
    print(f"[1] categories: {len(cats)} (expected {EXPECTED_CATEGORIES})")
    if len(cats) < EXPECTED_CATEGORIES - 3:
        raise SystemExit(f"FAIL: too few categories ({len(cats)})")

    ply_files = list(CO3D.glob("*/**/pointcloud.ply"))
    print(f"[2] pointcloud.ply files: {len(ply_files)}")
    if not ply_files:
        raise SystemExit("FAIL: no pointcloud.ply found")

    sample = ply_files[0]
    print(f"[3] sample ply: {sample.relative_to(CO3D)}")

    pcd = o3d.io.read_point_cloud(str(sample))
    xyz = np.asarray(pcd.points, dtype=np.float32)
    print(f"[4] loaded points: shape={xyz.shape}")

    xyz_n = pc_normalize(xyz.copy())
    if xyz_n.shape[0] >= 1024:
        xyz_fps = farthest_point_sample(xyz_n, 1024)
    else:
        xyz_fps = xyz_n
        print(f"    warn: only {xyz_n.shape[0]} points, skipping FPS")

    pts6 = with_normals(xyz_fps)
    nrm = np.linalg.norm(pts6[:, 3:], axis=1)
    print(f"[5] fps+normals: shape={pts6.shape}")
    print(f"    normal norms: min={nrm.min():.4f} max={nrm.max():.4f}")

    for cat in cats[:8]:
        cat_plys = list((CO3D / cat).glob("*/pointcloud.ply"))
        seq_dirs = [p for p in (CO3D / cat).iterdir() if p.is_dir() and p.name[0].isdigit()]
        if cat_plys:
            p = o3d.io.read_point_cloud(str(cat_plys[0]))
            print(
                f"[6] {cat}: {len(seq_dirs)} sequences, "
                f"{len(cat_plys)} with pointcloud.ply, first has {len(p.points)} points"
            )
        else:
            print(f"[6] {cat}: {len(seq_dirs)} sequences, no pointcloud.ply (images-only seq)")

    cats_with_ply = sum(1 for c in cats if list((CO3D / c).glob("*/pointcloud.ply")))
    print(f"[7] categories with pointcloud.ply: {cats_with_ply}/{len(cats)}")
    if cats_with_ply < 10:
        raise SystemExit("FAIL: too few categories with pointcloud.ply")

    print("\nSMOKE TEST PASSED")


if __name__ == "__main__":
    main()
