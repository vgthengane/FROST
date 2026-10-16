"""Modular FSCIL ablation config with explicit buffer vs embedding-only paths.

Paper components → code
-----------------------
| Paper        | Module / class              | Main knobs                          |
|--------------|-----------------------------|-------------------------------------|
| Adapter      | backbone + frost warmup     | ``fscil_adapter_base_only``, epochs |
| CorrMem      | ``frost/corrmem.py``        | ``ridge_rp_dim`` (V), λ             |
| ProtoMem     | ``frost/protomem.py``       | ``insert_strategy=mean``, static    |
| Fusion       | ``frost/frost.py`` ``_classify`` | ``classifier_mode=ridge_cache``, α |

Locked paper presets: ``exemplar_buffer_free`` (EF) / ``canonical`` (EB) via
``_PAPER_PROTOMEM`` (mean, 1 key/class, static, train-side conflict replay).

Terminology
-----------
**Exemplar buffer** — re-read fixed raw old-class *train* point clouds from disk
each incremental task (``exemplar_buffer_shots``, ``exemplar_buffer_ridge_refresh_shots``).

**Old-class proxies** — embedding vectors for old classes (ProtoMem prototypes when
``old_class_proxy_source=cache``). No raw sample storage.

**Conflict replay** — selective CorrMem re-weighting from embeddings (ProtoMem for
old classes, in-memory few-shot for novel). Never loads extra raw old-class samples
when proxies come from cache.

Legacy JSON keys (``fscil_old_shot_replay``, ``old_data_source``, …) remain supported
via :func:`prepare_fscil_args`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# ---------------------------------------------------------------------------
# Module configs (one dataclass per ablation axis)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExemplarBufferConfig:
    """Raw old-class train sample re-read from disk (the exemplar buffer)."""

    shots_per_old_class: int = 0
    shot_seed: int = 2008
    ridge_refresh_shots_per_old_class: int = 0
    ridge_refresh_weight: float = 1.0

    @property
    def enabled(self) -> bool:
        return self.shots_per_old_class > 0 or self.ridge_refresh_shots_per_old_class > 0


@dataclass(frozen=True)
class OldClassProxyConfig:
    """Embedding-only old-class signals (adaptation batch + conflict replay old path)."""

    source: str = "cache"  # cache | none

    @property
    def use_cache(self) -> bool:
        return self.source.lower() == "cache"

    @property
    def disabled(self) -> bool:
        return self.source.lower() in ("none", "off", "disabled")


@dataclass(frozen=True)
class ConflictReplayConfig:
    """Ridge stat re-weighting for low-recall classes (embedding sources only)."""

    enabled: bool = False
    use_test_confusion_matrix: bool = False
    max_old_classes: int = 5
    max_novel_classes: int = 5
    old_recall_threshold: float = 0.85
    old_weight: float = 2.0
    novel_weight: float = 3.0
    novel_threshold: float = 0.70


@dataclass(frozen=True)
class AdapterConfig:
    """Uni3D adapter SGD warmup schedule."""

    task0_warmup: bool = False
    increment_warmup: bool = False
    classifier_warmup_after_protos: bool = False
    base_only: bool = False
    task1_warmup: bool = False
    restore_task0_anchor_before_increment: bool = True
    increment_epochs: int = 2
    task1_epochs: int = 2
    last_task_epochs: int = 2


@dataclass(frozen=True)
class ClassifierConfig:
    """Ridge / cache inference and fusion."""

    mode: str = "ridge_cache"
    ridge_cache_alpha: float = 0.85

    @property
    def uses_ridge(self) -> bool:
        return self.mode in ("ridge", "ridge_cache", "ridge_ncm")

    @property
    def uses_cache_inference(self) -> bool:
        return self.mode in ("cache", "ridge_cache")


@dataclass(frozen=True)
class EvolutionConfig:
    """ProtoMem evolution policy."""

    policy: str = "rules"  # static | rules | bandit | aggressive | rl


@dataclass(frozen=True)
class FscilModules:
    """Resolved ablation modules for one FROST FSCIL run."""

    exemplar_buffer: ExemplarBufferConfig
    old_class_proxies: OldClassProxyConfig
    conflict_replay: ConflictReplayConfig
    adapter: AdapterConfig
    classifier: ClassifierConfig
    evolution: EvolutionConfig
    preset: str | None = None


# ---------------------------------------------------------------------------
# Legacy key aliases (old JSON name -> canonical resolver)
# ---------------------------------------------------------------------------

_LEGACY_INT = {
    "exemplar_buffer_shots": "fscil_old_shot_replay",
    "exemplar_buffer_ridge_refresh_shots": "fscil_base_ridge_refresh_per_class",
    "conflict_replay_max_old": "replay_max_classes",
    "conflict_replay_max_novel": "novel_replay_max_classes",
}

_LEGACY_BOOL = {
    "conflict_replay_use_test_cm": "conflict_replay_on_test",
}

_LEGACY_STR = {
    "old_class_proxy_source": "old_data_source",
}


def _int_arg(args: dict[str, Any], canonical: str, default: int) -> int:
    if canonical in args and args[canonical] is not None:
        return int(args[canonical])
    legacy = _LEGACY_INT.get(canonical)
    if legacy and legacy in args and args[legacy] is not None:
        return int(args[legacy])
    return default


def _bool_arg(args: dict[str, Any], canonical: str, default: bool) -> bool:
    if canonical in args and args[canonical] is not None:
        return bool(args[canonical])
    legacy = _LEGACY_BOOL.get(canonical)
    if legacy and legacy in args and args[legacy] is not None:
        return bool(args[legacy])
    return default


def _str_arg(args: dict[str, Any], canonical: str, default: str) -> str:
    if canonical in args and args[canonical] is not None:
        return str(args[canonical])
    legacy = _LEGACY_STR.get(canonical)
    if legacy and legacy in args and args[legacy] is not None:
        return str(args[legacy])
    return default


def _run_seed(args: dict[str, Any]) -> int:
    seed_val = args.get("seed", 1993)
    if isinstance(seed_val, list):
        seed_val = seed_val[0]
    return int(seed_val)


def resolve_fscil_modules(args: dict[str, Any]) -> FscilModules:
    """Build module config from args (canonical + legacy keys)."""
    seed = _run_seed(args)
    shot_seed = int(
        args.get(
            "exemplar_buffer_seed",
            args.get("fscil_old_shot_replay_seed", seed),
        )
    )
    recall_thr = float(
        args.get(
            "conflict_replay_old_recall_threshold",
            args.get("replay_recall_threshold", args.get("recall_refresh_threshold", 0.85)),
        )
    )
    return FscilModules(
        exemplar_buffer=ExemplarBufferConfig(
            shots_per_old_class=_int_arg(args, "exemplar_buffer_shots", 0),
            shot_seed=shot_seed,
            ridge_refresh_shots_per_old_class=_int_arg(
                args, "exemplar_buffer_ridge_refresh_shots", 0
            ),
            ridge_refresh_weight=float(
                args.get(
                    "exemplar_buffer_ridge_refresh_weight",
                    args.get("fscil_base_ridge_refresh_weight", 1.0),
                )
            ),
        ),
        old_class_proxies=OldClassProxyConfig(
            source=_str_arg(args, "old_class_proxy_source", "cache").lower(),
        ),
        conflict_replay=ConflictReplayConfig(
            enabled=bool(args.get("conflict_replay", False)),
            use_test_confusion_matrix=_bool_arg(args, "conflict_replay_use_test_cm", False),
            max_old_classes=_int_arg(args, "conflict_replay_max_old", 5),
            max_novel_classes=_int_arg(args, "conflict_replay_max_novel", 5),
            old_recall_threshold=recall_thr,
            old_weight=float(args.get("replay_weight", 2.0)),
            novel_weight=float(args.get("novel_replay_weight", 3.0)),
            novel_threshold=float(args.get("novel_replay_threshold", 0.70)),
        ),
        adapter=AdapterConfig(
            task0_warmup=bool(args.get("task0_adapter_warmup", False)),
            increment_warmup=bool(args.get("increment_adapter_warmup", False)),
            classifier_warmup_after_protos=bool(
                args.get("fscil_classifier_warmup_after_protos", False)
            ),
            base_only=bool(args.get("fscil_adapter_base_only", False)),
            task1_warmup=bool(args.get("fscil_task1_adapter_warmup", False)),
            restore_task0_anchor_before_increment=bool(
                args.get("fscil_restore_task0_adapter_before_increment_warmup", True)
            ),
            increment_epochs=int(args.get("increment_tuned_epoch", 2)),
            task1_epochs=int(args.get("fscil_task1_warmup_epochs", 2)),
            last_task_epochs=int(args.get("fscil_last_task_warmup_epochs", 2)),
        ),
        classifier=ClassifierConfig(
            mode=str(args.get("classifier_mode", "ridge_cache")),
            ridge_cache_alpha=float(args.get("ridge_cache_alpha", 0.85)),
        ),
        evolution=EvolutionConfig(policy=str(args.get("evolution_policy", "rules"))),
        preset=args.get("fscil_preset"),
    )


def sync_legacy_args(args: dict[str, Any], modules: FscilModules) -> None:
    """Write canonical and legacy keys so configs, logs, and scripts stay compatible."""
    buf = modules.exemplar_buffer
    args["exemplar_buffer_shots"] = buf.shots_per_old_class
    args["fscil_old_shot_replay"] = buf.shots_per_old_class
    args["exemplar_buffer_seed"] = buf.shot_seed
    args["fscil_old_shot_replay_seed"] = buf.shot_seed
    args["exemplar_buffer_ridge_refresh_shots"] = buf.ridge_refresh_shots_per_old_class
    args["fscil_base_ridge_refresh_per_class"] = buf.ridge_refresh_shots_per_old_class
    args["exemplar_buffer_ridge_refresh_weight"] = buf.ridge_refresh_weight
    args["fscil_base_ridge_refresh_weight"] = buf.ridge_refresh_weight

    prox = modules.old_class_proxies
    args["old_class_proxy_source"] = prox.source
    args["old_data_source"] = prox.source

    cr = modules.conflict_replay
    args["conflict_replay"] = cr.enabled
    args["conflict_replay_use_test_cm"] = cr.use_test_confusion_matrix
    args["conflict_replay_on_test"] = cr.use_test_confusion_matrix
    args["conflict_replay_max_old"] = cr.max_old_classes
    args["replay_max_classes"] = cr.max_old_classes
    args["conflict_replay_max_novel"] = cr.max_novel_classes
    args["novel_replay_max_classes"] = cr.max_novel_classes
    args["replay_recall_threshold"] = cr.old_recall_threshold
    args["replay_weight"] = cr.old_weight
    args["novel_replay_weight"] = cr.novel_weight
    args["novel_replay_threshold"] = cr.novel_threshold

    args["classifier_mode"] = modules.classifier.mode
    args["ridge_cache_alpha"] = modules.classifier.ridge_cache_alpha
    args["evolution_policy"] = modules.evolution.policy


# ---------------------------------------------------------------------------
# Presets (composable ablation entry points)
# ---------------------------------------------------------------------------

# Paper ProtoMem + train-side conflict replay (shared by EF / EB presets).
_PAPER_PROTOMEM: dict[str, Any] = {
    "insert_strategy": "mean",
    "max_prototypes_per_class": 1,
    "fscil_increment_max_prototypes": 1,
    "fscil_cluster_new_classes": False,
    "prototype_pooling": "max",
    "pooling_temperature": 0.07,
    "evolution_policy": "static",
    "old_class_proxy_source": "cache",
    "old_data_source": "cache",
    "conflict_replay": True,
    "conflict_replay_use_test_cm": False,
    "conflict_replay_on_test": False,
    # LAMDA rehearsal buffer unused by FROST (ProtoMem / exemplar_buffer_* instead).
    "memory_size": 0,
    "memory_per_class": 0,
    "fixed_memory": False,
    "task0_save_warmup_ckpts": False,
    "save_checkpoint": False,
}

_PRESET_PATCHES: dict[str, dict[str, Any]] = {
    # Exemplar-based (EB) paper twin: 1 old-shot/class on disk.
    "canonical": {
        **_PAPER_PROTOMEM,
        "exemplar_buffer_shots": 1,
        "exemplar_buffer_ridge_refresh_shots": 1,
        "classifier_mode": "ridge_cache",
    },
    # Exemplar-free (EF) paper default.
    "exemplar_buffer_free": {
        **_PAPER_PROTOMEM,
        "exemplar_buffer_shots": 0,
        "exemplar_buffer_ridge_refresh_shots": 0,
    },
    "ridge_only": {
        "classifier_mode": "ridge",
        "evolution_policy": "static",
    },
    "cache_only": {
        "classifier_mode": "cache",
        "evolution_policy": "rules",
    },
    "static_fusion": {
        "classifier_mode": "ridge_cache",
        "evolution_policy": "static",
    },
    "no_conflict_replay": {
        "conflict_replay": False,
        "conflict_replay_use_test_cm": False,
    },
}


def apply_fscil_preset(args: dict[str, Any], preset: str) -> dict[str, Any]:
    """Overlay a named preset onto args (does not resolve modules).

    Explicit user/CLI values for ``conflict_replay_use_test_cm`` and
    ``conflict_replay_on_test`` always win: presets must not re-enable
    test-set confusion-matrix replay after the user turned it off.
    """
    patch = _PRESET_PATCHES.get(preset)
    if patch is None:
        raise ValueError(f"Unknown fscil_preset={preset!r}; choose from {sorted(_PRESET_PATCHES)}")
    out = dict(args)
    protected = ("conflict_replay_use_test_cm", "conflict_replay_on_test")
    saved = {k: out[k] for k in protected if k in out}
    out.update(patch)
    out.update(saved)
    out["fscil_preset"] = preset
    return out


def apply_fscil_protocol_defaults(args: dict[str, Any]) -> dict[str, Any]:
    """Defaults for ``fscil_shot_protocol`` runs (before preset / JSON overrides)."""
    out = dict(args)
    if not bool(out.get("fscil_shot_protocol", False)):
        return out
    out.setdefault("increment_train_per_class", 5)
    out.setdefault("task0_adapter_warmup", True)
    out.setdefault("conflict_replay", True)
    # Never default to test-label replay (paper / release protocol).
    out.setdefault("conflict_replay_use_test_cm", False)
    out.setdefault("conflict_replay_on_test", False)
    out.setdefault("exemplar_buffer_shots", 1)
    out.setdefault("fscil_old_shot_replay", 1)
    # Legacy shot-protocol default; paper presets override to False (mean ProtoMem).
    out.setdefault("fscil_cluster_new_classes", True)
    out.setdefault("fscil_increment_ridge_alpha", 0.55)
    out.setdefault("fscil_shot_prototypes", False)
    out.setdefault("fscil_classifier_warmup_after_protos", False)
    out.setdefault("old_class_proxy_source", "cache")
    out.setdefault("old_data_source", "cache")
    out.setdefault("conflict_replay_max_old", 12)
    out.setdefault("replay_max_classes", 12)
    out.setdefault("conflict_replay_max_novel", 5)
    out.setdefault("novel_replay_max_classes", 5)
    out.setdefault("increment_adapter_warmup", True)
    out.setdefault("increment_tuned_epoch", 2)
    out.setdefault("fscil_task1_warmup_epochs", 2)
    out.setdefault("fscil_last_task_warmup_epochs", 2)
    out.setdefault("fscil_restore_task0_adapter_before_increment_warmup", True)
    cross_protocol = str(out.get("fscil3d_protocol", "")) in (
        "modelnet_scanobjectnn",
        "shapenet_scanobjectnn",
    )
    if cross_protocol:
        out.setdefault("fscil_task1_adapter_warmup", True)
    _apply_base_only_adapter_defaults(out)
    return out


def _apply_base_only_adapter_defaults(args: dict[str, Any]) -> None:
    """Adapter SGD on session 0 only (optional task-1 warmup for domain shift)."""
    if not bool(args.get("fscil_adapter_base_only", False)):
        return
    args["task0_adapter_warmup"] = True
    if bool(args.get("fscil_task1_adapter_warmup", False)):
        args["increment_adapter_warmup"] = True
    else:
        args["increment_adapter_warmup"] = False
        args["fscil_classifier_warmup_after_protos"] = False


def _ensure_runtime_defaults(args: dict[str, Any]) -> dict[str, Any]:
    """Fill LAMDA / ProtoMem knobs that paper configs no longer list explicitly."""
    out = dict(args)
    out.setdefault("memory_size", 0)
    out.setdefault("memory_per_class", 0)
    out.setdefault("fixed_memory", False)
    out.setdefault("shuffle", False)
    out.setdefault("task0_save_warmup_ckpts", False)
    out.setdefault("save_checkpoint", False)
    # ProtoMem budget: at least one slot per class (+ small headroom).
    n_cls = int(out.get("num_category", out.get("nb_classes", 40)))
    n_proto = int(out.get("max_prototypes_per_class", 1))
    out.setdefault("cache_budget", max(120, n_cls * max(n_proto, 1) + 50))
    return out


def prepare_fscil_args(args: dict[str, Any]) -> tuple[dict[str, Any], FscilModules]:
    """Normalize args: protocol defaults → preset → module resolve → legacy sync."""
    out = apply_fscil_protocol_defaults(args)
    preset = out.get("fscil_preset")
    if preset:
        out = apply_fscil_preset(out, str(preset))
    out = _ensure_runtime_defaults(out)
    modules = resolve_fscil_modules(out)
    sync_legacy_args(out, modules)
    return out, modules
