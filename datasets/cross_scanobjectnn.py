"""Cross-dataset pipelines with ScanObjectNN novel classes.

Supported protocols (microshape / FSCIL-3D README):
- modelnet_scanobjectnn: ModelNet40 base (session 1) + ScanObjectNN novel (sessions 2-4)
- shapenet_scanobjectnn: ShapeNet55 base (session 1) + ScanObjectNN novel (sessions 2-4)

ShapeNetCore v2 uses bed at cross base id 4 (microshape v1 used bicycle).

Training uses FSCIL task schedule (init_cls / increment): task 0 trains on base classes
only; later tasks add few-shot ScanObjectNN novel classes. Both domains are loaded into
one pool, but class ids follow the cross-dataset protocol order.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np

from datasets.fscil3d_index import get_index_root, load_protocol
from datasets.modelnet40 import iModelNet40
from datasets.scanobjectnn import iScanObjectNN
from datasets.scanobjectnn_taxonomy import (
    MODELNET_SCANOBJECTNN_NOVEL,
    SHAPENET55_SYNSET_TO_NAME,
    SHAPENET_NAME_ALIASES,
    SHAPENET_SCANOBJECTNN_BASE,
    SHAPENET_SCANOBJECTNN_NOVEL,
    scanobjectnn_name_to_label,
)
from datasets.shapenet53 import iShapeNet53

REPO_ROOT = Path(__file__).resolve().parents[1]


class iCrossScanObjectNN(object):
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
        if protocol_name not in ("modelnet_scanobjectnn", "shapenet_scanobjectnn"):
            raise ValueError(
                "Cross ScanObjectNN loader requires fscil3d_protocol in "
                "{'modelnet_scanobjectnn', 'shapenet_scanobjectnn'}."
            )

        proto = load_protocol(protocol_name)
        self.class_order = [c for task in proto.task_class_ids for c in task]

        src_train_x, src_train_y, src_test_x, src_test_y = self._load_source(protocol_name)
        src_map = (
            self._modelnet_source_to_cross()
            if protocol_name == "modelnet_scanobjectnn"
            else self._shapenet_source_to_cross()
        )

        tgt_train_x, tgt_train_y, tgt_test_x, tgt_test_y = self._load_target_scanobjectnn()
        tgt_map = self._scanobjectnn_to_cross(protocol_name)

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
            "Cross %s: base classes in pool=%d (ids 0-%d), "
            "novel ScanObjectNN classes in pool=%d (ids %s)",
            protocol_name,
            len(base_in_train),
            base_max,
            len(novel_in_train),
            novel_in_train[:5] if novel_in_train else [],
        )

    def _load_source(self, protocol_name: str):
        if protocol_name == "modelnet_scanobjectnn":
            src_args = dict(self.args)
            src_args["fscil3d_protocol"] = None
            src_args["data_path"] = self.args.get(
                "source_data_path",
                self.args.get("modelnet_data_path", self.args.get("data_path")),
            )
            src_args["num_category"] = 40
            source = iModelNet40(src_args)
        else:
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

    def _load_target_scanobjectnn(self):
        tgt_args = dict(self.args)
        tgt_args["fscil3d_protocol"] = None
        target = iScanObjectNN(tgt_args)
        target.download_data()
        return target.train_data, target.train_targets, target.test_data, target.test_targets

    def _modelnet_source_to_cross(self) -> dict[int, int]:
        """Map canonical ModelNet40 ids -> cross ids for session-1 base classes."""
        index_file = get_index_root() / "modelnet_scanobjectnn" / "train" / "session_1.txt"
        if not index_file.is_file():
            raise FileNotFoundError(index_file)
        source_data_path = self.args.get(
            "source_data_path",
            self.args.get("modelnet_data_path", self.args.get("data_path")),
        )
        name_to_src = {name: i for i, name in enumerate(iModelNet40.shape_names(str(source_data_path)))}
        mapping: dict[int, int] = {}
        for line in index_file.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            cross_id = int(parts[-1])
            if cross_id > 25:
                continue
            fname = Path(parts[0]).name
            m = re.match(r"(.+)_\d+\.pt$", fname)
            if m is None:
                continue
            cls_name = m.group(1)
            if cls_name not in name_to_src:
                raise KeyError(f"ModelNet class {cls_name!r} from index not found in shape_names")
            mapping[name_to_src[cls_name]] = cross_id
        return mapping

    def _shapenet_source_to_cross(self) -> dict[int, int]:
        """Map ShapeNet55 synset-sorted ids -> cross ids for session-1 base classes."""
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
            SHAPENET55_SYNSET_TO_NAME[syn]: idx for idx, syn in enumerate(synsets) if syn in SHAPENET55_SYNSET_TO_NAME
        }

        mapping: dict[int, int] = {}
        missing: list[str] = []
        for cross_id, proto_name in enumerate(SHAPENET_SCANOBJECTNN_BASE):
            local_name = SHAPENET_NAME_ALIASES.get(proto_name, proto_name)
            if local_name not in name_to_local:
                missing.append(proto_name)
                continue
            mapping[name_to_local[local_name]] = cross_id

        if missing:
            logging.warning(
                "ShapeNet55 missing %d/%d base classes for shapenet_scanobjectnn: %s",
                len(missing),
                len(SHAPENET_SCANOBJECTNN_BASE),
                missing,
            )
        if not mapping:
            raise RuntimeError("No ShapeNet base classes mapped for shapenet_scanobjectnn")
        return mapping

    @staticmethod
    def _scanobjectnn_to_cross(protocol_name: str) -> dict[int, int]:
        """Map native ScanObjectNN labels -> cross novel ids (by class name)."""
        name_to_scan = scanobjectnn_name_to_label()
        novel_table = (
            MODELNET_SCANOBJECTNN_NOVEL
            if protocol_name == "modelnet_scanobjectnn"
            else SHAPENET_SCANOBJECTNN_NOVEL
        )
        mapping: dict[int, int] = {}
        for cross_id, cls_name in novel_table.items():
            if cls_name not in name_to_scan:
                raise KeyError(f"Unknown ScanObjectNN class name {cls_name!r}")
            mapping[name_to_scan[cls_name]] = cross_id
        return mapping

    @staticmethod
    def _remap_split(x, y, mapping: dict[int, int]):
        keep = np.array([int(label) in mapping for label in y], dtype=bool)
        y_new = np.array([mapping[int(label)] for label in y[keep]], dtype=np.int64)
        return x[keep], y_new
