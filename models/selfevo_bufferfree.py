"""Buffer-free SelfEvo: no exemplar buffer (no old-class raw disk re-read).

See ``models/fscil_ablation_config.py`` for the modular preset definition.
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
    """SelfEvo with ``fscil_preset=exemplar_buffer_free`` applied by default."""

    def __init__(self, args):
        args = dict(args)
        args.setdefault("fscil_preset", "exemplar_buffer_free")
        super().__init__(args)
