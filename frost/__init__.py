"""FROST paper method package (Adapter + CorrMem + ProtoMem + Fusion).

Baselines stay under ``models/``. Learners are loaded by name via
``lamba_pilot.utils.factory`` (``model_name=frost`` / ``frost_bufferfree``).
"""

__all__ = ["CorrMem", "ProtoMem"]
