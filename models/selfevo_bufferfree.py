"""Buffer-free SelfEvo variant (no old-class raw sample re-read from disk).

This file is dynamically loaded by ``lamba_pilot/utils/factory.py`` via
``importlib.util.spec_from_file_location``. To avoid fragile package imports
(``models.*``), we load ``selfevo.py`` by absolute file path.

Buffer-free disables every path that re-loads old-class **raw train point clouds**
from disk. Old-class signals still come from the bounded KeyCache (prototype
embeddings) and from recursively accumulated ridge statistics ``G``, ``Q``.

Conflict replay remains enabled by default: it re-weights ridge stats for
low-recall classes using cache prototype embeddings, not disk samples.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path


_MODELS_DIR = Path(__file__).resolve().parent
_SELF_EVO_PATH = _MODELS_DIR / "selfevo.py"

_spec = importlib.util.spec_from_file_location("pilot_selfevo_base", _SELF_EVO_PATH)
assert _spec is not None and _spec.loader is not None
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

SelfEvoLearner = _mod.Learner


class Learner(SelfEvoLearner):
    """SelfEvo with old-class disk access disabled; cache-based replay retained."""

    def __init__(self, args):
        args = dict(args)
        # Prototype embeddings from KeyCache (includes base classes after session 0).
        args["old_data_source"] = "cache"
        # No raw old-class samples appended to train/proto loaders.
        args["fscil_old_shot_replay"] = 0
        # No separate disk re-read for base-class ridge refresh.
        args["fscil_base_ridge_refresh_per_class"] = 0
        # Cache-driven selective ridge re-weighting (uses old_class_embeddings from cache).
        args.setdefault("conflict_replay", True)
        args.setdefault("conflict_replay_on_test", True)
        # replay_max_classes / novel_replay_max_classes: inherit from config.
        super().__init__(args)
