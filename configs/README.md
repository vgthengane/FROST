# Config Guide (CIL branch)

This directory contains the **active** experiment configs for the **Class-Incremental Learning (CIL)** track (ModelNet40).

FSCIL (few-shot) configs are kept on the `main` branch and/or archived under `_archive/` for provenance.

## Active CIL configs

| Config file | Backbone / Method | Dataset | Protocol |
|---|---|---|---|
| `mos_modelnet40.json` | MOS | ModelNet40 | CIL |
| `ranpac_modelnet40.json` | RanPAC | ModelNet40 | CIL |
| `simplecil_modelnet40.json` | SimpleCIL | ModelNet40 | CIL |
| `slca_modelnet40.json` | SLC/A | ModelNet40 | CIL |
| `tuna_modelnet40.json` | TUNA | ModelNet40 | CIL |
| `aper_adapter_modelnet40.json` | Adapter (APER) | ModelNet40 | CIL |
| `coda_prompt_modelnet40.json` | CoDA-Prompt | ModelNet40 | CIL |
| `dualprompt_modelnet40.json` | DualPrompt | ModelNet40 | CIL |
| `ease_modelnet40.json` | EASE | ModelNet40 | CIL |
| `l2p_modelnet40.json` | L2P | ModelNet40 | CIL |

## How to run

Typical entry point:

```bash
sbatch scripts/core.sh
```

Or run directly with:

```bash
aprun python train_cil.py --config configs/mos_modelnet40.json
```

