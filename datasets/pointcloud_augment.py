"""Train-time point-cloud augmentations (microshape-aligned, normals-aware).

Applied only during adapter SGD; ridge/cache prototypes and eval use clean clouds.
"""

from __future__ import annotations

import torch


def random_scale_xyz(
    points: torch.Tensor,
    scale_low: float = 0.8,
    scale_high: float = 1.25,
) -> torch.Tensor:
    """Per-cloud uniform scale on xyz. ``points``: [B, N, C], C >= 3."""
    if points.dim() != 3 or points.size(-1) < 3:
        raise ValueError(f"expected [B, N, C>=3], got {tuple(points.shape)}")
    scales = torch.empty(points.size(0), 1, 1, device=points.device, dtype=points.dtype)
    scales.uniform_(scale_low, scale_high)
    out = points.clone()
    out[:, :, :3] = out[:, :, :3] * scales
    return out


def random_shift_xyz(
    points: torch.Tensor,
    shift_range: float = 0.1,
) -> torch.Tensor:
    """Per-cloud xyz translation. ``points``: [B, N, C], C >= 3."""
    if points.dim() != 3 or points.size(-1) < 3:
        raise ValueError(f"expected [B, N, C>=3], got {tuple(points.shape)}")
    shifts = torch.empty(points.size(0), 1, 3, device=points.device, dtype=points.dtype)
    shifts.uniform_(-shift_range, shift_range)
    out = points.clone()
    out[:, :, :3] = out[:, :, :3] + shifts
    return out


def random_rotate_z(
    points: torch.Tensor,
) -> torch.Tensor:
    """Random rotation about the z axis on xyz; normals rotated if C >= 6."""
    if points.dim() != 3 or points.size(-1) < 3:
        raise ValueError(f"expected [B, N, C>=3], got {tuple(points.shape)}")
    b = points.size(0)
    theta = torch.rand(b, device=points.device, dtype=points.dtype) * (2.0 * torch.pi)
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)
    rot = torch.zeros(b, 3, 3, device=points.device, dtype=points.dtype)
    rot[:, 0, 0] = cos_t
    rot[:, 0, 1] = -sin_t
    rot[:, 1, 0] = sin_t
    rot[:, 1, 1] = cos_t
    rot[:, 2, 2] = 1.0

    out = points.clone()
    xyz = out[:, :, :3]
    out[:, :, :3] = torch.bmm(xyz, rot)
    if out.size(-1) >= 6:
        nrm = out[:, :, 3:6]
        out[:, :, 3:6] = torch.bmm(nrm, rot)
    return out


def augment_task0_point_cloud(
    points: torch.Tensor,
    *,
    scale_low: float = 0.8,
    scale_high: float = 1.25,
    shift_range: float = 0.1,
    rotate_z: bool = True,
) -> torch.Tensor:
    """Microshape-style geom aug: optional z-rot, scale, shift (xyz only for latter two)."""
    out = points
    if rotate_z:
        out = random_rotate_z(out)
    out = random_scale_xyz(out, scale_low=scale_low, scale_high=scale_high)
    out = random_shift_xyz(out, shift_range=shift_range)
    return out
