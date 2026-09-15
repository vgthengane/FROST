"""FSCIL naive Fine-Tune baseline: frozen Uni3D + linear SGD head."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_MODELS_DIR = Path(__file__).resolve().parent
_PATH = _MODELS_DIR / "fscil_linprobe.py"
_spec = importlib.util.spec_from_file_location("pilot_fscil_linprobe_ft", _PATH)
assert _spec is not None and _spec.loader is not None
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
_Base = _mod.Learner


class Learner(_Base):
    def __init__(self, args):
        args = dict(args)
        args["linprobe_mode"] = "finetune"
        args["model_name"] = args.get("model_name", "fscil_finetune")
        super().__init__(args)
