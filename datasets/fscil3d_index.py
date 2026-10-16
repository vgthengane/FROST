"""FSCIL-3D (ECCV 2022) session schedule — same class IDs and sessions as microshape.

Defines which **dataset class IDs** appear in each incremental session (e.g. MN40:
20+4×5 over 5 sessions). Implementation reads the canonical FSCIL-3D schedule;
alignment with microshape is by **class id + session**, not by sample lists.

3D-FLEG uses related but not identical splits (e.g. 44 vs 39 ShapeNet base for SN→CO3D).
Pick one protocol and cite it; do not mix numbers across papers.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

REPO_ROOT = Path(__file__).resolve().parents[1]
# Prefer in-repo vendored indexes (no third_party clone required for training).
VENDORED_INDEX_ROOT = REPO_ROOT / "datasets" / "session_pclouds"
FSCIL3D_ROOT = REPO_ROOT / "external" / "FSCIL-3D"
MICROSHAPE_INDEX_ROOT = REPO_ROOT / "third_party" / "microshape" / "data" / "index_files"

ProtocolName = Literal[
    "modelnet",
    "shapenet",
    "co3d",
    "modelnet_scanobjectnn",
    "shapenet_scanobjectnn",
    "shapenet_co3d",
]

PROTOCOL_NAMES: tuple[str, ...] = (
    "modelnet",
    "shapenet",
    "co3d",
    "modelnet_scanobjectnn",
    "shapenet_scanobjectnn",
    "shapenet_co3d",
)


@dataclass(frozen=True)
class FSCIL3DProtocol:
    """Fixed session schedule: class IDs per session (FSCIL-3D / microshape)."""

    name: str
    n_sessions: int
    task_class_ids: list[list[int]]
    cumulative_class_counts: list[int]
    increment_sizes: list[int]
    fewshot: int = 5
    old_shot_replay: int = 1

    @property
    def total_classes(self) -> int:
        return self.cumulative_class_counts[-1] if self.cumulative_class_counts else 0

    @property
    def base_class_count(self) -> int:
        return self.increment_sizes[0] if self.increment_sizes else 0


def get_index_root() -> Path:
    """FSCIL-3D session/class index files (class IDs + relative sample paths)."""
    for candidate in (
        VENDORED_INDEX_ROOT,
        FSCIL3D_ROOT / "data" / "index_files",
        MICROSHAPE_INDEX_ROOT,
    ):
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        "FSCIL-3D index files not found. Expected datasets/session_pclouds/ "
        "(vendored) or external/FSCIL-3D / third_party/microshape."
    )


def get_fscil3d_root() -> Path:
    """Best-effort root that *contains* ``data/index_files`` (legacy callers)."""
    if VENDORED_INDEX_ROOT.is_dir():
        return VENDORED_INDEX_ROOT.parent  # datasets/
    if FSCIL3D_ROOT.is_dir():
        return FSCIL3D_ROOT
    micro = REPO_ROOT / "third_party" / "microshape"
    if micro.is_dir() and MICROSHAPE_INDEX_ROOT.is_dir():
        return micro
    raise FileNotFoundError(
        "FSCIL-3D indexes not found. Use datasets/session_pclouds/ (vendored) "
        "or clone https://github.com/townim-faisal/FSCIL-3D.git into external/FSCIL-3D."
    )


def _class_names_in_session_file(path: Path) -> list[str]:
    """Return unique class names in first-appearance order from an index file."""
    names: list[str] = []
    seen: set[str] = set()
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        cls_name = line.split("/")[0]
        if cls_name not in seen:
            seen.add(cls_name)
            names.append(cls_name)
    return names


def _class_ids_in_session_file(path: Path) -> list[int]:
    """Return unique numeric class ids in first-appearance order."""
    ids: list[int] = []
    seen: set[int] = set()
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        label = int(parts[0]) if parts[0].isdigit() else int(parts[-1])
        if label not in seen:
            seen.add(label)
            ids.append(label)
    return ids


# FSCIL-3D ShapeNet session order (README table). Global ids match ShapeNet55
# synset-sorted indices when ``excluded_synsets=[]``.
_SHAPENET_FSCIL3D_TASK_CLASS_IDS: tuple[list[int], ...] = (
    [49, 18, 0, 17, 47, 44, 30, 53, 6, 32, 13, 22, 12, 4, 26, 24, 19, 41, 28, 10, 31, 9, 50, 29, 52],
    [36, 1, 25, 40, 38],
    [5, 48, 37, 11, 54],
    [42, 27, 35, 46, 51],
    [14, 3, 15, 39, 33],
    [21, 45, 2, 23, 8],
    [34, 43, 20, 7, 16],
)


def _modelnet_name_to_id(data_path: str) -> dict[str, int]:
    from datasets.modelnet40 import iModelNet40

    names = iModelNet40.shape_names(data_path)
    return {name: idx for idx, name in enumerate(names)}


def fscil3d_class_order(protocol: str, data_path: str | None = None) -> list[int]:
    """Canonical dataset class indices in FSCIL-3D session order (for DataManager)."""
    p = load_protocol(protocol, data_path=data_path)
    order: list[int] = []
    for task_ids in p.task_class_ids:
        order.extend(task_ids)
    if len(order) != p.total_classes:
        raise ValueError(
            f"FSCIL-3D {protocol}: class_order length {len(order)} != {p.total_classes}"
        )
    return order


def _protocol_from_index_files(
    name: str, name_to_id: dict[str, int] | None = None
) -> FSCIL3DProtocol:
    train_dir = get_index_root() / name / "train"
    if not train_dir.is_dir():
        raise FileNotFoundError(f"Missing index files: {train_dir}")

    session_files = sorted(
        train_dir.glob("session_*.txt"),
        key=lambda p: int(p.stem.split("_")[1]),
    )
    task_class_ids: list[list[int]] = []
    seen: set[int] = set()
    increment_sizes: list[int] = []

    for sf in session_files:
        if name_to_id is not None:
            cls_names = _class_names_in_session_file(sf)
            cls_ids = []
            for cls_name in cls_names:
                if cls_name not in name_to_id:
                    raise KeyError(
                        f"FSCIL-3D {name}: unknown class {cls_name!r} in {sf.name}"
                    )
                cls_ids.append(name_to_id[cls_name])
        else:
            cls_ids = _class_ids_in_session_file(sf)
        new_ids = [c for c in cls_ids if c not in seen]
        if not new_ids and cls_ids:
            new_ids = cls_ids
        task_class_ids.append(new_ids)
        increment_sizes.append(len(new_ids))
        seen.update(cls_ids)

    cumulative = []
    total = 0
    for inc in increment_sizes:
        total += inc
        cumulative.append(total)

    return FSCIL3DProtocol(
        name=name,
        n_sessions=len(session_files),
        task_class_ids=task_class_ids,
        cumulative_class_counts=cumulative,
        increment_sizes=increment_sizes,
    )


def _load_shapenet_co3d_config() -> dict:
    candidates = (
        VENDORED_INDEX_ROOT / "shapenet_co3d_info.py",
        FSCIL3D_ROOT / "configs" / "shapenet_co3d_info.py",
        REPO_ROOT / "third_party" / "microshape" / "configs" / "shapenet_co3d_info.py",
    )
    cfg_path = next((p for p in candidates if p.is_file()), None)
    if cfg_path is None:
        raise FileNotFoundError(
            "shapenet_co3d_info.py not found. Expected one of:\n"
            + "\n".join(f"  - {p}" for p in candidates)
        )
    ns: dict = {}
    exec(cfg_path.read_text(), ns)  # noqa: S102 — trusted upstream config
    return ns


def load_protocol(
    name: ProtocolName, data_path: str | None = None
) -> FSCIL3DProtocol:
    """Load FSCIL-3D protocol metadata by name.

    For ``modelnet``, pass ``data_path`` so index class names map to canonical ids.
    """
    get_index_root()  # fail fast if indexes missing
    if name not in PROTOCOL_NAMES:
        raise ValueError(f"Unknown protocol {name!r}; choose from {PROTOCOL_NAMES}")

    name_to_id = None
    if name == "modelnet":
        if not data_path:
            raise ValueError("load_protocol('modelnet') requires data_path")
        name_to_id = _modelnet_name_to_id(data_path)

    if name == "shapenet":
        task_class_ids = [list(t) for t in _SHAPENET_FSCIL3D_TASK_CLASS_IDS]
        increment_sizes = [len(t) for t in task_class_ids]
        cumulative = []
        total = 0
        for inc in increment_sizes:
            total += inc
            cumulative.append(total)
        return FSCIL3DProtocol(
            name=name,
            n_sessions=len(task_class_ids),
            task_class_ids=task_class_ids,
            cumulative_class_counts=cumulative,
            increment_sizes=increment_sizes,
        )

    proto = _protocol_from_index_files(name, name_to_id=name_to_id)

    if name == "shapenet_co3d":
        cfg = _load_shapenet_co3d_config()
        len_cls = list(cfg["len_cls"])
        task_ids = [list(x) for x in cfg["task_ids_total"]]
        if len(len_cls) != len(task_ids):
            raise ValueError("shapenet_co3d_info.py len_cls / task_ids_total mismatch")
        return FSCIL3DProtocol(
            name=name,
            n_sessions=len(task_ids),
            task_class_ids=task_ids,
            cumulative_class_counts=len_cls,
            increment_sizes=cfg["model_heads"],
            fewshot=5,
            old_shot_replay=1,
        )

    return proto


def protocol_summary(name: ProtocolName, data_path: str | None = None) -> str:
    """One-line description for logging."""
    p = load_protocol(name, data_path=data_path)
    return (
        f"FSCIL-3D {name}: {p.n_sessions} sessions, "
        f"{p.total_classes} classes, increments={p.increment_sizes}"
    )
