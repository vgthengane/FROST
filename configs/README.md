# Config Guide

This directory now contains the **active** experiment configs only. Superseded
FSCIL tuning configs were moved to `_archive/configs/`.

## Canonical active configs (in this directory)

This directory currently contains **only** FSCIL configs for the primary
experiment track. Everything else (standard CIL configs) was moved to:

```text
_archive/configs/cil/
```

### FSCIL (primary track)

These follow the **FSCIL-3D task definition** via `fscil3d_protocol` and are the
main configs for the method paper experiments.

| Config | Dataset | Protocol | Purpose |
|--------|---------|----------|---------|
| `selfevo_fscil_mn40_5t_flex_compare.json` | ModelNet40 | `modelnet` | Canonical FSCIL run for our method |
| `selfevo_fscil_sn55_7t_flex_compare.json` | ShapeNet55 | `shapenet` | Canonical FSCIL run for our method |
| `ranpac_fscil_mn40_5t_fscil3d.json` | ModelNet40 | `modelnet` | FSCIL baseline to add into `flex_compare` |

Use with:

```bash
sbatch scripts/flex_compare.sh
```

Reference:
- `_notes/fscil_canonical.md`
- `external/FSCIL-3D/`

## Archived configs

Superseded FSCIL tuning configs and all non-FSCIL configs live in `_archive/`
(e.g. `_archive/configs/cil/` and `_archive/configs/`).

Those files are retained for provenance only and should not be used for new
runs unless you are reproducing a specific archived experiment tree.
