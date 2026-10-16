"""FROST buffer-free: no exemplar buffer (no old-class raw disk re-read).

Canonical entry for ``model_name=frost_bufferfree``.
See ``utils/fscil_ablation_config.py`` for the modular preset definition.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path


_FROST_DIR = Path(__file__).resolve().parent
_FROST_PATH = _FROST_DIR / "frost.py"

_spec = importlib.util.spec_from_file_location("pilot_frost_base", _FROST_PATH)
assert _spec is not None and _spec.loader is not None
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

FrostLearner = _mod.Learner


class Learner(FrostLearner):
    """FROST with ``fscil_preset=exemplar_buffer_free`` applied by default."""

    def __init__(self, args):
        args = dict(args)
        args.setdefault("fscil_preset", "exemplar_buffer_free")
        super().__init__(args)
