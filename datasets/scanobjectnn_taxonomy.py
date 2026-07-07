"""ScanObjectNN + cross-dataset class tables (microshape / FSCIL-3D README)."""

from __future__ import annotations

# Native ScanObjectNN label order (main_split_nobg h5 -> build script).
SCANOBJECTNN_CLASS_NAMES: tuple[str, ...] = (
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
)

# ModelNet40 -> ScanObjectNN: 26 base ModelNet classes (session 1).
MODELNET_SCANOBJECTNN_BASE: tuple[str, ...] = (
    "airplane",
    "bathtub",
    "bottle",
    "bowl",
    "car",
    "cone",
    "cup",
    "curtain",
    "flower pot",
    "glass box",
    "guitar",
    "keyboard",
    "lamp",
    "laptop",
    "mantel",
    "night stand",
    "person",
    "piano",
    "plant",
    "radio",
    "range hood",
    "stairs",
    "tent",
    "tv stand",
    "vase",
    "xbox",
)

# ModelNet40 -> ScanObjectNN: 11 novel ScanObjectNN classes (sessions 2-4).
MODELNET_SCANOBJECTNN_NOVEL: dict[int, str] = {
    26: "cabinet",
    27: "chair",
    28: "desk",
    29: "display",
    30: "door",
    31: "shelf",
    32: "table",
    33: "bed",
    34: "sink",
    35: "sofa",
    36: "toilet",
}

# ShapeNet55 -> ScanObjectNN: 15 novel ScanObjectNN classes (sessions 2-4).
SHAPENET_SCANOBJECTNN_NOVEL: dict[int, str] = {
    44: "bag",
    45: "bin",
    46: "box",
    47: "cabinet",
    48: "chair",
    49: "desk",
    50: "display",
    51: "door",
    52: "shelf",
    53: "table",
    54: "bed",
    55: "pillow",
    56: "sink",
    57: "sofa",
    58: "toilet",
}

# ShapeNet55 -> ScanObjectNN base session (44 classes, cross ids 0..43).
# Microshape/FSCIL-3D v1 uses bicycle at id 4; ShapeNetCore v2 keeps bed (02818832) instead.
SHAPENET_SCANOBJECTNN_BASE: tuple[str, ...] = (
    "airplane",
    "basket",
    "bathtub",
    "bench",
    "bed",
    "birdhouse",
    "bottle",
    "bowl",
    "bus",
    "camera",
    "can",
    "cap",
    "car",
    "clock",
    "keyboard",
    "dishwasher",
    "earphone",
    "faucet",
    "file_cabinet",
    "guitar",
    "helmet",
    "jar",
    "knife",
    "lamp",
    "laptop",
    "loudspeaker",
    "microphone",
    "microwaves",
    "motorbike",
    "mug",
    "piano",
    "pistol",
    "flowerpot",
    "printer",
    "remote",
    "rifle",
    "rocket",
    "skateboard",
    "stove",
    "telephone",
    "tower",
    "train",
    "watercraft",
    "washer",
)

# ShapeNet-55 synset id -> category name (sorted-synset loader order).
SHAPENET55_SYNSET_TO_NAME: dict[str, str] = {
    "02691156": "airplane",
    "02747177": "trash_bin",
    "02773838": "bag",
    "02801938": "basket",
    "02808440": "bathtub",
    "02818832": "bed",
    "02828884": "bench",
    "02834778": "bicycle",
    "02843684": "birdhouse",
    "02871439": "bookshelf",
    "02876657": "bottle",
    "02880940": "bowl",
    "02924116": "bus",
    "02933112": "cabinet",
    "02942699": "camera",
    "02946921": "can",
    "02954340": "cap",
    "02958343": "car",
    "02992529": "cellphone",
    "03001627": "chair",
    "03046257": "clock",
    "03085013": "keyboard",
    "03207941": "dishwasher",
    "03211117": "display",
    "03261776": "earphone",
    "03325088": "faucet",
    "03337140": "file_cabinet",
    "03467517": "guitar",
    "03513137": "helmet",
    "03593526": "jar",
    "03624134": "knife",
    "03636649": "lamp",
    "03642806": "laptop",
    "03691459": "loudspeaker",
    "03710193": "mailbox",
    "03759954": "microphone",
    "03761084": "microwave",
    "03790512": "motorbike",
    "03797390": "mug",
    "03928116": "piano",
    "03938244": "pillow",
    "03948459": "pistol",
    "03991062": "flowerpot",
    "04004475": "printer",
    "04074963": "remote",
    "04090263": "rifle",
    "04099429": "rocket",
    "04225987": "skateboard",
    "04256520": "sofa",
    "04330267": "stove",
    "04379243": "table",
    "04401088": "telephone",
    "04460130": "tower",
    "04468005": "train",
    "04530566": "watercraft",
    "04554684": "washer",
}

# Microshape README name aliases -> local ShapeNet55 category name.
SHAPENET_NAME_ALIASES: dict[str, str] = {
    "microwaves": "microwave",
}


def scanobjectnn_name_to_label() -> dict[str, int]:
    return {name: idx for idx, name in enumerate(SCANOBJECTNN_CLASS_NAMES)}


def cross_scanobjectnn_class_label(protocol: str, class_idx: int) -> tuple[str, str]:
    """Return ``(dataset, class_name)`` for a mapped cross-protocol class index."""
    if protocol == "modelnet_scanobjectnn":
        if class_idx < len(MODELNET_SCANOBJECTNN_BASE):
            return "ModelNet", MODELNET_SCANOBJECTNN_BASE[class_idx]
        if class_idx in MODELNET_SCANOBJECTNN_NOVEL:
            return "ScanObjectNN", MODELNET_SCANOBJECTNN_NOVEL[class_idx]
    elif protocol == "shapenet_scanobjectnn":
        if class_idx < len(SHAPENET_SCANOBJECTNN_BASE):
            name = SHAPENET_SCANOBJECTNN_BASE[class_idx].replace("_", " ")
            return "ShapeNet", name
        if class_idx in SHAPENET_SCANOBJECTNN_NOVEL:
            return "ScanObjectNN", SHAPENET_SCANOBJECTNN_NOVEL[class_idx]
    raise KeyError(f"Unknown cross class {class_idx} for protocol {protocol!r}")


def format_cross_scanobjectnn_task_classes(protocol: str, class_indices: list[int]) -> str:
    """Multi-line summary of class ids/names for one FSCIL task."""
    lines = []
    for idx in class_indices:
        dataset, name = cross_scanobjectnn_class_label(protocol, idx)
        lines.append(f"  [{idx:2d}] {name} ({dataset})")
    return "\n".join(lines)
