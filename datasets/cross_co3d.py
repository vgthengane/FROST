"""Cross-dataset pipeline: ShapeNet55 base + CO3D novel (microshape shapenet_co3d).

Protocol ``shapenet_co3d``: 39 ShapeNet base classes (session 1) + 50 CO3D novel
classes in 10 incremental sessions of 5 (sessions 2-11), 89 classes total.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from datasets.co3d import iCO3D
from datasets.co3d_taxonomy import (
    CO3D_SHAPENET_NAME_ALIASES,
    SHAPENET_CO3D_BASE,
    SHAPENET_CO3D_NOVEL,
    load_co3d_class_names,
)
from datasets.scanobjectnn_taxonomy import SHAPENET55_SYNSET_TO_NAME
from datasets.fscil3d_index import load_protocol
from datasets.shapenet53 import iShapeNet53

REPO_ROOT = Path(__file__).resolve().parents[1]


class iCrossCO3D(object):
    use_path = False
    is_pointcloud = True
    train_trsf = []
    test_trsf = []
    common_trsf = []
    class_order = None

    def __init__(self, args):
        self.args = args or {}
        self.train_data = None
        self.train_targets = None
        self.test_data = None
        self.test_targets = None

    def download_data(self):
        protocol_name = str(self.args.get("fscil3d_protocol", "")).strip()
        if protocol_name != "shapenet_co3d":
            raise ValueError(
                "Cross CO3D loader requires fscil3d_protocol='shapenet_co3d'."
            )

        proto = load_protocol(protocol_name)
        self.class_order = [c for task in proto.task_class_ids for c in task]

        src_train_x, src_train_y, src_test_x, src_test_y = self._load_source_shapenet()
        src_map = self._shapenet_source_to_cross()

        tgt_train_x, tgt_train_y, tgt_test_x, tgt_test_y = self._load_target_co3d()
        tgt_map = self._co3d_to_cross()

        src_train_x, src_train_y = self._remap_split(src_train_x, src_train_y, src_map)
        src_test_x, src_test_y = self._remap_split(src_test_x, src_test_y, src_map)
        tgt_train_x, tgt_train_y = self._remap_split(tgt_train_x, tgt_train_y, tgt_map)
        tgt_test_x, tgt_test_y = self._remap_split(tgt_test_x, tgt_test_y, tgt_map)

        self.train_data = np.concatenate([src_train_x, tgt_train_x], axis=0)
        self.train_targets = np.concatenate([src_train_y, tgt_train_y], axis=0)
        self.test_data = np.concatenate([src_test_x, tgt_test_x], axis=0)
        self.test_targets = np.concatenate([src_test_y, tgt_test_y], axis=0)

        base_max = max(proto.task_class_ids[0])
        train_ids = set(np.unique(self.train_targets).tolist())
        novel_in_train = sorted(c for c in train_ids if c > base_max)
        base_in_train = sorted(c for c in train_ids if c <= base_max)
        logging.info(
            "Cross shapenet_co3d: base classes in pool=%d (ids 0-%d), "
            "novel CO3D classes in pool=%d",
            len(base_in_train),
            base_max,
            len(novel_in_train),
        )

    def _load_source_shapenet(self):
        src_args = dict(self.args)
        src_args["fscil3d_protocol"] = None
        src_args["data_path"] = self.args.get(
            "source_data_path",
            self.args.get("shapenet_data_path", self.args.get("data_path")),
        )
        src_args.setdefault("excluded_synsets", [])
        src_args["num_category"] = 55
        source = iShapeNet53(src_args)
        source.download_data()
        return source.train_data, source.train_targets, source.test_data, source.test_targets

    def _load_target_co3d(self):
        tgt_args = dict(self.args)
        tgt_args["fscil3d_protocol"] = None
        tgt_args["data_path"] = self.args.get(
            "co3d_data_path",
            self.args.get("data_path", "_datasets/pclouds/CO3D"),
        )
        tgt_args["num_category"] = 50
        target = iCO3D(tgt_args)
        target.download_data()
        return target.train_data, target.train_targets, target.test_data, target.test_targets

    def _shapenet_source_to_cross(self) -> dict[int, int]:
        data_path = self.args.get(
            "source_data_path",
            self.args.get("shapenet_data_path", self.args.get("data_path")),
        )
        root = Path(data_path) if Path(data_path).is_absolute() else REPO_ROOT / data_path
        split_file = root / "splits" / "train.txt"
        if not split_file.is_file():
            raise FileNotFoundError(f"Missing ShapeNet split file: {split_file}")

        excluded = set(self.args.get("excluded_synsets", []))
        synsets = sorted(
            {
                line.strip().split("-")[0]
                for line in split_file.read_text().splitlines()
                if line.strip() and line.strip().split("-")[0] not in excluded
            }
        )
        name_to_local = {
            SHAPENET55_SYNSET_TO_NAME[syn]: idx
            for idx, syn in enumerate(synsets)
            if syn in SHAPENET55_SYNSET_TO_NAME
        }

        mapping: dict[int, int] = {}
        missing: list[str] = []
        for cross_id, proto_name in enumerate(SHAPENET_CO3D_BASE):
            local_name = CO3D_SHAPENET_NAME_ALIASES.get(proto_name, proto_name)
            if local_name not in name_to_local:
                missing.append(proto_name)
                continue
            mapping[name_to_local[local_name]] = cross_id

        if missing:
            logging.warning(
                "ShapeNet55 missing %d/%d base classes for shapenet_co3d: %s",
                len(missing),
                len(SHAPENET_CO3D_BASE),
                missing,
            )
        if not mapping:
            raise RuntimeError("No ShapeNet base classes mapped for shapenet_co3d")
        return mapping

    def _co3d_to_cross(self) -> dict[int, int]:
        co3d_path = self.args.get(
            "co3d_data_path",
            self.args.get("data_path", "_datasets/pclouds/CO3D"),
        )
        names = load_co3d_class_names(co3d_path)
        name_to_native = {name: idx for idx, name in enumerate(names)}
        mapping: dict[int, int] = {}
        for cross_id, cls_name in SHAPENET_CO3D_NOVEL.items():
            if cls_name not in name_to_native:
                raise KeyError(f"CO3D class {cls_name!r} missing from cache names")
            mapping[name_to_native[cls_name]] = cross_id
        return mapping

    @staticmethod
    def _remap_split(x, y, mapping: dict[int, int]):
        keep = np.array([int(label) in mapping for label in y], dtype=bool)
        y_new = np.array([mapping[int(label)] for label in y[keep]], dtype=np.int64)
        return x[keep], y_new
