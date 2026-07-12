"""CO3D + ShapeNet->CO3D cross-dataset class tables (microshape README)."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Native CO3D label order (matches ``co3d_class_names.txt`` / FSCIL index ids 0..49).
CO3D_CLASS_NAMES: tuple[str, ...] = (
    "backpack",
    "handbag",
    "carrot",
    "teddybear",
    "bowl",
    "chair",
    "vase",
    "book",
    "keyboard",
    "plant",
    "ball",
    "hairdryer",
    "umbrella",
    "laptop",
    "orange",
    "toytruck",
    "suitcase",
    "wineglass",
    "mouse",
    "cellphone",
    "broccoli",
    "remote",
    "apple",
    "cake",
    "toilet",
    "bicycle",
    "hydrant",
    "toaster",
    "bottle",
    "motorcycle",
    "sandwich",
    "bench",
    "toyplane",
    "couch",
    "car",
    "banana",
    "donut",
    "stopsign",
    "cup",
    "kite",
    "toybus",
    "pizza",
    "frisbee",
    "baseballglove",
    "skateboard",
    "baseballbat",
    "hotdog",
    "microwave",
    "tv",
    "parkingmeter",
)

# ShapeNet -> CO3D: 39 base ShapeNet classes (session 1, cross ids 0..38).
SHAPENET_CO3D_BASE: tuple[str, ...] = (
    "airplane",
    "trash bin",
    "basket",
    "bathtub",
    "bed",
    "birdhouse",
    "bookshelf",
    "bus",
    "cabinet",
    "camera",
    "can",
    "cap",
    "clock",
    "dishwasher",
    "display",
    "faucet",
    "file cabinet",
    "guitar",
    "helmet",
    "jar",
    "knife",
    "lamp",
    "loudspeaker",
    "mailbox",
    "microphone",
    "mug",
    "piano",
    "pillow",
    "pistol",
    "flowerpot",
    "printer",
    "rifle",
    "rocket",
    "stove",
    "table",
    "tower",
    "train",
    "vessel",
    "washer",
)

# ShapeNet -> CO3D: 50 novel CO3D classes (sessions 2-11, cross ids 39..88).
SHAPENET_CO3D_NOVEL: dict[int, str] = {
    39: "apple",
    40: "backpack",
    41: "ball",
    42: "banana",
    43: "baseballbat",
    44: "baseballglove",
    45: "bench",
    46: "bicycle",
    47: "book",
    48: "bottle",
    49: "bowl",
    50: "broccoli",
    51: "cake",
    52: "car",
    53: "carrot",
    54: "cellphone",
    55: "chair",
    56: "couch",
    57: "cup",
    58: "donut",
    59: "frisbee",
    60: "hairdryer",
    61: "handbag",
    62: "hotdog",
    63: "hydrant",
    64: "keyboard",
    65: "kite",
    66: "laptop",
    67: "microwave",
    68: "motorcycle",
    69: "mouse",
    70: "orange",
    71: "parkingmeter",
    72: "pizza",
    73: "plant",
    74: "remote",
    75: "sandwich",
    76: "skateboard",
    77: "stopsign",
    78: "suitcase",
    79: "teddybear",
    80: "toaster",
    81: "toilet",
    82: "toybus",
    83: "toyplane",
    84: "toytruck",
    85: "tv",
    86: "umbrella",
    87: "vase",
    88: "wineglass",
}

# Microshape README / ShapeNet55 naming aliases.
CO3D_SHAPENET_NAME_ALIASES: dict[str, str] = {
    "trash bin": "trash_bin",
    "file cabinet": "file_cabinet",
    "vessel": "watercraft",
    "cabinet": "cabinet",
}


def co3d_name_to_label() -> dict[str, int]:
    return {name: idx for idx, name in enumerate(CO3D_CLASS_NAMES)}


def load_co3d_class_names(data_path: str | Path | None = None) -> list[str]:
    """Load ``co3d_class_names.txt`` if present, else fall back to taxonomy."""
    if data_path is not None:
        path = Path(data_path) if Path(data_path).is_absolute() else REPO_ROOT / data_path
        names_file = path / "co3d_class_names.txt"
        if names_file.is_file():
            return names_file.read_text().splitlines()
    return list(CO3D_CLASS_NAMES)


def cross_co3d_class_label(class_idx: int) -> tuple[str, str]:
    """Return ``(dataset, class_name)`` for a mapped shapenet_co3d class index."""
    if class_idx < len(SHAPENET_CO3D_BASE):
        name = SHAPENET_CO3D_BASE[class_idx]
        return "ShapeNet", name
    if class_idx in SHAPENET_CO3D_NOVEL:
        return "CO3D", SHAPENET_CO3D_NOVEL[class_idx]
    raise KeyError(f"Unknown shapenet_co3d class index {class_idx}")


def format_cross_co3d_task_classes(class_indices: list[int]) -> str:
    lines = []
    for idx in class_indices:
        dataset, name = cross_co3d_class_label(idx)
        lines.append(f"  [{idx:2d}] {name} ({dataset})")
    return "\n".join(lines)
