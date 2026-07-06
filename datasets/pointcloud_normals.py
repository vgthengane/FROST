"""Estimate point-cloud normals for xyz-only datasets (e.g. ShapeNet)."""

from __future__ import annotations

import numpy as np


def estimate_normals(xyz: np.ndarray, k: int = 30) -> np.ndarray:
    """Estimate unit normals for ``xyz`` [N, 3] using Open3D (k-NN PCA).

    Assumes ``xyz`` is already centered/scaled the same way as ModelNet loading
    (``pc_normalize`` on coordinates only).
    """
    import open3d as o3d

    n = xyz.shape[0]
    if n < 3:
        return np.zeros((n, 3), dtype=np.float32)

    k_eff = max(3, min(int(k), n))
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k_eff),
        fast_normal_computation=True,
    )
    try:
        pcd.orient_normals_consistent_tangent_plane(k_eff)
    except Exception:
        pass

    normals = np.asarray(pcd.normals, dtype=np.float32)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    return normals / norms


def with_normals(point_set: np.ndarray, k: int = 30) -> np.ndarray:
    """Return [N, 6] array: xyz (+ existing cols ignored) + estimated normals."""
    xyz = point_set[:, :3].astype(np.float32)
    if point_set.shape[1] >= 6:
        return point_set[:, :6].astype(np.float32)
    normals = estimate_normals(xyz, k=k)
    return np.concatenate([xyz, normals], axis=1).astype(np.float32)
