# FROST (paper method)

Separated from `models/` baselines.

| File | Role |
|------|------|
| `frost.py` | Learner: Adapter + CorrMem + ProtoMem + Fusion |
| `frost_bufferfree.py` | EF entry (`model_name=frost_bufferfree`) |
| `corrmem.py` | CorrMem (`CorrMem`) |
| `protomem.py` | ProtoMem (`ProtoMem`) |
| `confusion.py` | Train-side confusion matrix helper |

Configs: `configs/frost_*.json` + `configs/base/`. Launch: `scripts/reproduce_*.sh`.
