"""FSCIL-style data caps: full base session, few-shot increments, optional old-class replay."""

from __future__ import annotations

import logging

import numpy as np

# Per-class index picks are cached so old-class replay (and increment shots) stay
# identical across later tasks within a run / seed.
_CLASS_INDEX_CACHE: dict[tuple, np.ndarray] = {}


def clear_fscil_shot_cache() -> None:
    """Reset cached shot indices (e.g. between multi-seed jobs in one process)."""
    _CLASS_INDEX_CACHE.clear()


def _class_sample_seed(base_seed: int, class_id: int) -> int:
    return int(base_seed) + int(class_id) * 7919


def _pick_class_indices(
    pool_size: int,
    n_pick: int,
    *,
    cache_key: tuple,
) -> np.ndarray:
    """Return sorted indices into a class pool; reuse cached pick when present."""
    if cache_key in _CLASS_INDEX_CACHE:
        return _CLASS_INDEX_CACHE[cache_key]
    n = min(int(n_pick), int(pool_size))
    if n <= 0:
        idx = np.array([], dtype=np.int64)
    elif n >= pool_size:
        idx = np.arange(pool_size, dtype=np.int64)
    else:
        _, class_id, role_seed, _ = cache_key
        rng = np.random.default_rng(_class_sample_seed(role_seed, class_id))
        idx = np.sort(rng.choice(pool_size, size=n, replace=False))
    _CLASS_INDEX_CACHE[cache_key] = idx
    return idx


def subsample_dataset_by_class(
    dataset,
    max_per_class: int,
    seed: int,
    *,
    cache_role: str = "incr",
):
    """Cap samples per class in a DummyDataset."""
    if max_per_class is None or max_per_class <= 0:
        return dataset
    from utils.data_manager import DummyDataset

    images = dataset.images
    labels = dataset.labels
    keep_idx = []
    role_seed = int(seed)
    for cls in np.unique(labels):
        cls = int(cls)
        cls_idx = np.where(labels == cls)[0]
        cache_key = (cache_role, cls, role_seed, int(max_per_class))
        rel = _pick_class_indices(len(cls_idx), max_per_class, cache_key=cache_key)
        keep_idx.append(cls_idx[rel])
    keep_idx = np.sort(np.concatenate(keep_idx))
    return DummyDataset(
        images[keep_idx],
        labels[keep_idx],
        dataset.trsf,
        dataset.use_path,
        dataset.is_pointcloud,
    )


def concat_datasets(datasets):
    from utils.data_manager import DummyDataset

    datasets = [ds for ds in datasets if ds is not None and len(ds) > 0]
    if len(datasets) == 1:
        return datasets[0]
    images = np.concatenate([ds.images for ds in datasets], axis=0)
    labels = np.concatenate([ds.labels for ds in datasets], axis=0)
    ref = datasets[0]
    return DummyDataset(
        images, labels, ref.trsf, ref.use_path, ref.is_pointcloud
    )


def append_old_class_shots(
    dataset,
    data_manager,
    known_classes: int,
    shots_per_class: int,
    seed: int,
):
    """Append train samples per old class (FSCIL old-class replay).

    Uses a run-wide fixed seed and per-class cache so the same old-class
    exemplars are replayed on every incremental task.
    """
    if known_classes <= 0 or shots_per_class <= 0:
        return dataset
    from utils.data_manager import DummyDataset

    role_seed = int(seed)
    old_parts = []
    for cls in range(known_classes):
        old_ds = data_manager.get_dataset([cls], source="train", mode="test")
        if len(old_ds) == 0:
            continue
        cache_key = ("old_replay", cls, role_seed, int(shots_per_class))
        idx = _pick_class_indices(
            len(old_ds.images), shots_per_class, cache_key=cache_key
        )
        old_parts.append(
            DummyDataset(
                old_ds.images[idx],
                old_ds.labels[idx],
                old_ds.trsf,
                old_ds.use_path,
                old_ds.is_pointcloud,
            )
        )
    if not old_parts:
        return dataset
    return concat_datasets([dataset] + old_parts)


def prepare_fscil_task_datasets(
    data_manager,
    known_classes: int,
    total_classes: int,
    cur_task: int,
    args: dict,
):
    """Build train + proto datasets for one FSCIL task."""
    train_dataset = data_manager.get_dataset(
        np.arange(known_classes, total_classes),
        source="train",
        mode="train",
    )
    proto_dataset = data_manager.get_dataset(
        np.arange(known_classes, total_classes),
        source="train",
        mode="test",
    )

    if cur_task == 0:
        logging.info(
            "Base session: task=0 classes=0-%d train_n=%d (full data)",
            total_classes - 1,
            len(train_dataset),
        )
        return train_dataset, proto_dataset

    incr_cap = args.get("increment_train_per_class")
    if incr_cap is None:
        return train_dataset, proto_dataset

    seed_val = args.get("seed", 1993)
    if isinstance(seed_val, list):
        seed_val = seed_val[0]
    # Novel-class few-shot: seed depends on the introducing task (fixed per class).
    subsample_seed = int(seed_val)
    train_dataset = subsample_dataset_by_class(
        train_dataset, int(incr_cap), subsample_seed, cache_role="incr"
    )
    proto_dataset = subsample_dataset_by_class(
        proto_dataset, int(incr_cap), subsample_seed, cache_role="incr"
    )

    old_shots = int(
        args.get("exemplar_buffer_shots", args.get("fscil_old_shot_replay", 0))
    )
    if old_shots > 0:
        # Exemplar buffer: same raw train indices on every task (not re-drawn per task).
        replay_seed = int(
            args.get(
                "exemplar_buffer_seed",
                args.get("fscil_old_shot_replay_seed", int(seed_val)),
            )
        )
        train_dataset = append_old_class_shots(
            train_dataset, data_manager, known_classes, old_shots, replay_seed
        )
        proto_dataset = append_old_class_shots(
            proto_dataset, data_manager, known_classes, old_shots, replay_seed
        )
        logging.info(
            "FSCIL old-shot replay: task=%d old_classes=%d shots/class=%d "
            "total_n=%d fixed_seed=%d",
            cur_task,
            known_classes,
            old_shots,
            len(proto_dataset),
            replay_seed,
        )

    logging.info(
        "Few-shot increment: task=%d classes=%d-%d cap=%d new_n=%d",
        cur_task,
        known_classes,
        total_classes - 1,
        int(incr_cap),
        int(incr_cap) * (total_classes - known_classes),
    )
    return train_dataset, proto_dataset


def prepare_joint_fscil_train_dataset(
    data_manager,
    total_classes: int,
    cur_task: int,
    args: dict,
):
    """FLEG-style Joint under the FSCIL budget.

    Session 0: full base-class training data.
    Session t>0: full base data + the same cached ``increment_train_per_class``
    shots for every novel class seen so far (no old-class 1-shot replay buffer).
    """
    base_n = int(args["init_cls"])
    seed_val = args.get("seed", 1993)
    if isinstance(seed_val, list):
        seed_val = seed_val[0]
    seed_val = int(seed_val)
    incr_cap = int(args.get("increment_train_per_class", 5))

    if cur_task == 0 or total_classes <= base_n:
        train_dataset = data_manager.get_dataset(
            np.arange(0, total_classes),
            source="train",
            mode="train",
        )
        logging.info(
            "Joint base session: task=%d classes=0-%d train_n=%d (full data)",
            cur_task,
            total_classes - 1,
            len(train_dataset),
        )
        return train_dataset

    base_dataset = data_manager.get_dataset(
        np.arange(0, base_n),
        source="train",
        mode="train",
    )
    novel_dataset = data_manager.get_dataset(
        np.arange(base_n, total_classes),
        source="train",
        mode="train",
    )
    novel_dataset = subsample_dataset_by_class(
        novel_dataset, incr_cap, seed_val, cache_role="incr"
    )
    train_dataset = concat_datasets([base_dataset, novel_dataset])
    logging.info(
        "Joint session: task=%d base=0-%d(full n=%d) novel=%d-%d(%d-shot n=%d) "
        "total_n=%d",
        cur_task,
        base_n - 1,
        len(base_dataset),
        base_n,
        total_classes - 1,
        incr_cap,
        len(novel_dataset),
        len(train_dataset),
    )
    return train_dataset
