"""LGSP-style image FSCIL datasets for PRidE / SelfEvo.

Protocol source: https://github.com/Jywsuperman/LGSP (ICCV 2025)

We reuse their ``session_*.txt`` index lists only. Method / PTM backbone stay
in this repo (``frost/frost.py`` + ``lamba_pilot`` ViT).

Shot protocol is baked into the index files:
  - session_1: full base-class training images
  - session_t>1: 5-shot novel classes for that session
Test sets use the dataset's official test split for all classes seen so far
(handled by DataManager class filtering).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Iterable

import numpy as np
from torchvision import transforms

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INDEX_ROOT = REPO_ROOT / "datasets" / "session_images"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _vit_train_trsf():
    return [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]


def _vit_test_trsf():
    return [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]


def _read_lines(path: Path) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]


def _list2dict(lines: Iterable[str]) -> dict[int, str]:
    out: dict[int, str] = {}
    for line in lines:
        parts = line.split(" ")
        idx = int(parts[0])
        val = parts[1]
        if idx in out:
            raise ValueError(f"Duplicate id {idx} in metadata")
        out[idx] = val
    return out


def _resolve_repo_path(value: str | Path) -> Path:
    """Resolve absolute or repo-relative paths (cwd-independent)."""
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def resolve_index_root(args: dict | None) -> Path:
    """Session index root (``datasets/session_images`` by default; relative paths OK)."""
    if args:
        custom = (
            args.get("sessions_root")
            or args.get("session_index_root")
            or args.get("lgsp_index_root")  # legacy alias
            or args.get("index_list_root")
        )
        if custom:
            return _resolve_repo_path(custom)
    return DEFAULT_INDEX_ROOT


def resolve_data_root(args: dict | None, default_name: str) -> Path:
    """Image root containing CUB200 / FGVC-Aircraft / iNF200."""
    if args:
        for key in ("data_path", "dataroot", "lgsp_data_root"):
            val = args.get(key)
            if val:
                return _resolve_repo_path(val)
    # Prefer FROST layout: _datasets/images
    preferred = REPO_ROOT / "_datasets" / "images"
    if preferred.exists():
        return preferred.resolve()
    candidates = [
        REPO_ROOT / "data",
        REPO_ROOT.parent / "data",
        Path("/scratch/vt00262/datasets"),
        Path("/vol/research/Vishal_Thengane/datasets"),
    ]
    for c in candidates:
        if (c / default_name).exists():
            return c.resolve()
    return preferred.resolve()


def resolve_image_subdir(data_root: Path, *names: str) -> Path:
    """Pick first existing subdirectory (canonical name first, then legacy)."""
    for name in names:
        path = data_root / name
        if path.exists():
            return path
    return data_root / names[0]



def _maybe_subsample(args: dict, train_data, train_targets, test_data, test_targets):
    """Optional smoke caps: max_classes / max_train_per_class / max_test_per_class."""
    if not any(
        k in args for k in ("max_classes", "max_train_per_class", "max_test_per_class")
    ):
        return train_data, train_targets, test_data, test_targets

    seed_val = args.get("seed", 1993)
    if isinstance(seed_val, list):
        seed_val = seed_val[0]
    rng = np.random.default_rng(int(seed_val))
    max_classes = int(args.get("max_classes", args.get("num_category", 200)))
    max_train = int(args.get("max_train_per_class", 10**9))
    max_test = int(args.get("max_test_per_class", 10**9))

    def _cap(data, targets, max_per_class):
        keep = []
        for cls in range(max_classes):
            idx = np.where(targets == cls)[0]
            if len(idx) == 0:
                continue
            n = min(max_per_class, len(idx))
            keep.append(rng.choice(idx, size=n, replace=False))
        if not keep:
            return data, targets
        keep = np.sort(np.concatenate(keep))
        return data[keep], targets[keep]

    train_data, train_targets = _cap(train_data, train_targets, max_train)
    test_data, test_targets = _cap(test_data, test_targets, max_test)
    logging.info(
        "LGSP smoke subsample: train=%d test=%d max_classes=%d "
        "max_train/class=%d max_test/class=%d",
        len(train_data),
        len(test_data),
        max_classes,
        max_train,
        max_test,
    )
    return train_data, train_targets, test_data, test_targets


class iCUB200FSCIL:
    """CUB-200-2011 under LGSP FSCIL splits (100 base + 10×10 novel, 5-shot)."""

    use_path = True
    is_pointcloud = False

    def __init__(self, args=None):
        self.args = args or {}
        self.train_trsf = _vit_train_trsf()
        self.test_trsf = _vit_test_trsf()
        self.common_trsf = []
        self.class_order = np.arange(200).tolist()
        self.train_data = None
        self.train_targets = None
        self.test_data = None
        self.test_targets = None

    def download_data(self):
        data_root = resolve_data_root(self.args, "CUB200")
        cub_root = resolve_image_subdir(data_root, "CUB200", "CUB_200_2011")
        if not cub_root.exists():
            raise FileNotFoundError(
                f"CUB not found under {data_root}. "
                "Expected _datasets/images/CUB200/. "
                "Download with datasets/preprocess/download_cub200.sh."
            )

        index_root = resolve_index_root(self.args) / "cub200"
        if not index_root.exists():
            raise FileNotFoundError(f"Missing LGSP index root: {index_root}")

        id2image = _list2dict(_read_lines(cub_root / "images.txt"))
        id2train = _list2dict(_read_lines(cub_root / "train_test_split.txt"))
        id2class = _list2dict(_read_lines(cub_root / "image_class_labels.txt"))

        # path -> label (0-based)
        train_path2label: dict[str, int] = {}
        test_paths: list[str] = []
        test_labels: list[int] = []
        for k in sorted(id2train.keys()):
            rel = id2image[k]
            path = str((cub_root / "images" / rel).resolve())
            label = int(id2class[k]) - 1
            if id2train[k] == "1":
                train_path2label[path] = label
            else:
                test_paths.append(path)
                test_labels.append(label)

        # Train = union of LGSP session index files (base full + novel 5-shot).
        train_paths: list[str] = []
        train_labels: list[int] = []
        seen: set[str] = set()
        session_files = sorted(
            index_root.glob("session_*.txt"),
            key=lambda p: int(p.stem.split("_")[1]),
        )
        if not session_files:
            raise FileNotFoundError(f"No session_*.txt under {index_root}")

        for sess in session_files:
            for rel in _read_lines(sess):
                # Index lines: CUB200/images/... (legacy CUB_200_2011/ also accepted)
                rel_norm = rel.replace("CUB_200_2011/", "CUB200/", 1)
                full = str((data_root / rel_norm).resolve())
                if full in seen:
                    continue
                if full not in train_path2label:
                    logging.warning("LGSP index path missing from CUB train split: %s", rel)
                    continue
                seen.add(full)
                train_paths.append(full)
                train_labels.append(train_path2label[full])

        self.train_data = np.array(train_paths)
        self.train_targets = np.array(train_labels)
        self.test_data = np.array(test_paths)
        self.test_targets = np.array(test_labels)
        (
            self.train_data,
            self.train_targets,
            self.test_data,
            self.test_targets,
        ) = _maybe_subsample(
            self.args,
            self.train_data,
            self.train_targets,
            self.test_data,
            self.test_targets,
        )
        logging.info(
            "iCUB200FSCIL: train=%d test=%d classes_train=%d data_root=%s",
            len(self.train_data),
            len(self.test_data),
            len(np.unique(self.train_targets)),
            data_root,
        )


class iFGVCAircraftFSCIL:
    """FGVC-Aircraft under LGSP FSCIL (50 base + 10×5 novel, 5-shot)."""

    use_path = True
    is_pointcloud = False

    def __init__(self, args=None):
        self.args = args or {}
        self.train_trsf = _vit_train_trsf()
        self.test_trsf = _vit_test_trsf()
        self.common_trsf = []
        self.class_order = np.arange(100).tolist()
        self.train_data = None
        self.train_targets = None
        self.test_data = None
        self.test_targets = None

    def download_data(self):
        data_root = resolve_data_root(self.args, "FGVC-Aircraft")
        ac_base = resolve_image_subdir(
            data_root, "FGVC-Aircraft", "Aircraft", "fgvc-aircraft-2013b"
        )
        ac_root = ac_base / "data"
        if not ac_root.exists():
            raise FileNotFoundError(
                f"FGVC-Aircraft not found under {data_root}. "
                "Expected _datasets/images/FGVC-Aircraft/data/. "
                "Download with datasets/preprocess/download_fgvc_aircraft.sh."
            )

        variants = _read_lines(ac_root / "variants.txt")
        label_dict = {v: i for i, v in enumerate(variants)}

        def load_split(split_name: str) -> tuple[list[str], list[int]]:
            paths, labels = [], []
            for line in _read_lines(ac_root / split_name):
                img_id, variant = line.split(" ", 1)
                path = str((ac_root / "images" / f"{img_id}.jpg").resolve())
                paths.append(path)
                labels.append(label_dict[variant])
            return paths, labels

        train_paths_all, train_labels_all = load_split("images_variant_train.txt")
        id_to_label = {
            Path(p).stem: y for p, y in zip(train_paths_all, train_labels_all)
        }
        path_by_id = {Path(p).stem: p for p in train_paths_all}

        index_root = resolve_index_root(self.args) / "FGVC-Aircraft"
        if not index_root.is_dir():
            # Legacy folder name from LGSP release
            index_root = resolve_index_root(self.args) / "FGVCAircraft"
        train_paths: list[str] = []
        train_labels: list[int] = []
        seen: set[str] = set()
        for sess in sorted(
            index_root.glob("session_*.txt"),
            key=lambda p: int(p.stem.split("_")[1]),
        ):
            for rel in _read_lines(sess):
                img_id = Path(rel).stem
                if img_id not in path_by_id:
                    logging.warning("FGVC-Aircraft index id missing: %s", img_id)
                    continue
                full = path_by_id[img_id]
                if full in seen:
                    continue
                seen.add(full)
                train_paths.append(full)
                train_labels.append(id_to_label[img_id])

        test_paths, test_labels = load_split("images_variant_test.txt")
        self.train_data = np.array(train_paths)
        self.train_targets = np.array(train_labels)
        self.test_data = np.array(test_paths)
        self.test_targets = np.array(test_labels)
        logging.info(
            "iFGVCAircraftFSCIL: train=%d test=%d data_root=%s",
            len(self.train_data),
            len(self.test_data),
            data_root,
        )


class iNF200FSCIL:
    """iNF200 under LGSP FSCIL (100 base + 10×10 novel, 5-shot)."""

    use_path = True
    is_pointcloud = False

    def __init__(self, args=None):
        self.args = args or {}
        self.train_trsf = _vit_train_trsf()
        self.test_trsf = _vit_test_trsf()
        self.common_trsf = []
        self.class_order = np.arange(200).tolist()
        self.train_data = None
        self.train_targets = None
        self.test_data = None
        self.test_targets = None

    def download_data(self):
        data_root = resolve_data_root(self.args, "iNF200")
        inf_root = data_root / "iNF200"
        train_dir = inf_root / "train_mini"
        val_dir = inf_root / "val"
        if not train_dir.exists() or not val_dir.exists():
            raise FileNotFoundError(
                f"iNF200 not found under {data_root} "
                "(expected iNF200/train_mini and iNF200/val)."
            )

        # Label map must follow LGSP session indexes (exactly 200 classes).
        # train_mini/val may contain extra folders from the HF dump; indexing
        # all of them shifts labels past 0..199 and breaks DataManager mapping.
        index_root = resolve_index_root(self.args) / "iNF200"
        session_folders: set[str] = set()
        session_files = sorted(
            index_root.glob("session_*.txt"),
            key=lambda p: int(p.stem.split("_")[1]),
        )
        for sess in session_files:
            for rel in _read_lines(sess):
                parts = Path(rel).parts
                if parts[0] == "iNF200" and len(parts) >= 3:
                    session_folders.add(parts[2])
                elif parts[0] == "train_mini" and len(parts) >= 2:
                    session_folders.add(parts[1])
                elif len(parts) >= 2:
                    session_folders.add(parts[1])

        class_folders = sorted(session_folders)
        if len(class_folders) != 200:
            raise RuntimeError(
                f"iNF200 session indexes cover {len(class_folders)} classes "
                f"(expected 200) under {index_root}"
            )
        label_dict = {folder: i for i, folder in enumerate(class_folders)}
        disk_train = {
            d for d in os.listdir(train_dir) if (train_dir / d).is_dir()
        }
        missing = [f for f in class_folders if f not in disk_train]
        if missing:
            raise FileNotFoundError(
                f"iNF200 train_mini missing {len(missing)} session classes "
                f"(e.g. {missing[:3]})"
            )
        extra = len(disk_train) - len(class_folders)
        if extra > 0:
            logging.info(
                "iNF200: ignoring %d extra train_mini folders not in LGSP sessions",
                extra,
            )

        train_paths: list[str] = []
        train_labels: list[int] = []
        seen: set[str] = set()
        for sess in session_files:
            for rel in _read_lines(sess):
                # Expected: train_mini/<class>/<img> or iNF200/train_mini/...
                parts = Path(rel).parts
                if parts[0] == "iNF200":
                    full = str((data_root / rel).resolve())
                    class_folder = parts[2] if len(parts) >= 3 else parts[1]
                elif parts[0] == "train_mini":
                    full = str((inf_root / rel).resolve())
                    class_folder = parts[1]
                else:
                    full = str((inf_root / rel).resolve())
                    class_folder = parts[1] if len(parts) > 1 else parts[0]
                if class_folder not in label_dict:
                    logging.warning("iNF200 unknown class folder: %s", class_folder)
                    continue
                if not os.path.exists(full) or full in seen:
                    continue
                seen.add(full)
                train_paths.append(full)
                train_labels.append(label_dict[class_folder])

        test_paths: list[str] = []
        test_labels: list[int] = []
        for folder in class_folders:
            cdir = val_dir / folder
            if not cdir.is_dir():
                logging.warning("iNF200 val missing class folder: %s", folder)
                continue
            for img in sorted(os.listdir(cdir)):
                test_paths.append(str((cdir / img).resolve()))
                test_labels.append(label_dict[folder])

        self.train_data = np.array(train_paths)
        self.train_targets = np.array(train_labels)
        self.test_data = np.array(test_paths)
        self.test_targets = np.array(test_labels)
        n_train_cls = len(np.unique(self.train_targets)) if len(self.train_targets) else 0
        n_test_cls = len(np.unique(self.test_targets)) if len(self.test_targets) else 0
        logging.info(
            "iNF200FSCIL: train=%d test=%d classes_train=%d classes_test=%d data_root=%s",
            len(self.train_data),
            len(self.test_data),
            n_train_cls,
            n_test_cls,
            data_root,
        )
