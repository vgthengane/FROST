"""Generate lean Joint/FT linprobe configs for all 6 FSCIL protocols."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "configs"

COMMON = {
    "memory_size": 0,
    "memory_per_class": 0,
    "fixed_memory": False,
    "shuffle": False,
    "backbone_type": "uni3d_base_patch14_448",
    "pc_model": "eva02_base_patch14_448",
    "pretrained_pc": "_experiments/pretrained_weights/Uni3D/uni3d-b.pt",
    "encoder_dim": 512,
    "embed_dim": 1024,
    "trans_dim": 768,
    "num_group": 512,
    "group_size": 64,
    "device": ["0"],
    "seed": [2008],
    "num_point": 1024,
    "use_normals": True,
    "process_data": True,
    "use_uniform_sample": True,
    "batch_size": 8,
    "num_workers": 4,
    "fscil_shot_protocol": True,
    "increment_train_per_class": 5,
    "base_epoch": 10,
    "increment_epoch": 5,
    "init_lr": 0.01,
    "weight_decay": 0.0005,
    "min_lr": 0.0,
    "optimizer": "sgd",
}

PROTOCOLS = {
    "mn40": {
        "tag": "mn40_5t",
        "n_sess": 5,
        "fields": {
            "dataset": "modelnet40",
            "data_path": "_datasets/modelnet40_normal_resampled",
            "fscil3d_protocol": "modelnet",
            "init_cls": 20,
            "increment": 5,
            "num_category": 40,
        },
    },
    "sn55": {
        "tag": "sn55_7t",
        "n_sess": 7,
        "fields": {
            "dataset": "shapenet55",
            "data_path": "_datasets/ShapeNet55",
            "fscil3d_protocol": "shapenet",
            "init_cls": 25,
            "increment": 5,
            "num_category": 55,
            "excluded_synsets": [],
        },
    },
    "co3d": {
        "tag": "co3d_6t",
        "n_sess": 6,
        "fields": {
            "dataset": "co3d",
            "data_path": "_datasets/CO3D",
            "fscil3d_protocol": "co3d",
            "init_cls": 25,
            "increment": 5,
            "num_category": 50,
        },
    },
    "mn_scanobj": {
        "tag": "mn40_scanobj_4t",
        "n_sess": 4,
        "fields": {
            "dataset": "modelnet_scanobjectnn",
            "fscil3d_protocol": "modelnet_scanobjectnn",
            "source_data_path": "_datasets/modelnet40_normal_resampled",
            "scanobjectnn_data_path": "_datasets/scanobjectnn/main_split_nobg",
            "init_cls": 26,
            "increment": 4,
            "num_category": 37,
        },
    },
    "sn_scanobj": {
        "tag": "sn55_scanobj_4t",
        "n_sess": 4,
        "fields": {
            "dataset": "shapenet_scanobjectnn",
            "fscil3d_protocol": "shapenet_scanobjectnn",
            "source_data_path": "_datasets/ShapeNet55",
            "scanobjectnn_data_path": "_datasets/scanobjectnn/main_split_nobg",
            "excluded_synsets": [],
            "init_cls": 44,
            "increment": 5,
            "num_category": 59,
        },
    },
    "sn_co3d": {
        "tag": "sn55_co3d_11t",
        "n_sess": 11,
        "fields": {
            "dataset": "shapenet_co3d",
            "fscil3d_protocol": "shapenet_co3d",
            "source_data_path": "_datasets/ShapeNet55",
            "co3d_data_path": "_datasets/CO3D",
            "excluded_synsets": [],
            "init_cls": 39,
            "increment": 5,
            "num_category": 89,
        },
    },
}


def main() -> None:
    written = []
    for pid, meta in PROTOCOLS.items():
        for mode in ("joint", "finetune"):
            cfg = dict(COMMON)
            cfg.update(meta["fields"])
            cfg["prefix"] = f"{pid}_fscil_{mode}_linprobe"
            cfg["model_name"] = f"fscil_{mode}"
            cfg["linprobe_mode"] = mode
            if mode == "joint":
                cfg["fscil_old_shot_replay"] = 0
            else:
                cfg["fscil_old_shot_replay"] = 1
                cfg["fscil_old_shot_replay_seed"] = 2008
            path = CFG / f"fscil_{mode}_{meta['tag']}.json"
            path.write_text(json.dumps(cfg, indent=4) + "\n")
            written.append((pid, mode, path.name, meta["n_sess"]))
    for row in written:
        print("\t".join(map(str, row)))
    print(f"wrote {len(written)} configs")


if __name__ == "__main__":
    main()
