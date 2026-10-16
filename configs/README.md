# Config Guide

Configs use optional `_base` inheritance (see `utils/config_load.py`).
Child files only list **what changes**.

Paper blocks (`adapter`, `corrmem`, `protomem`, `fusion`) match Sec.~3 of the
paper and are flattened to internal training keys at load time.

ProtoMem details (static mean, 1 key/class, max pool, train-side conflict
replay) and unused LAMDA `memory_*` live in `fscil_preset`, not the JSON bases.

## Bases (shared recipe)

| File | Role |
|------|------|
| `base/pclouds.json` | Paper 3D recipe: Adapter + CorrMem (\(V{=}2000\)) + ProtoMem (mean) + Fusion (\(\alpha{=}0.5\)), EF default |
| `base/images.json` | Same dual-memory recipe for images (EF only) |

Both list the full paper seed set. **Default runs use the first seed only**;
pass `--fullseeds` (train entrypoints or reproduce scripts) for all five.

## Point clouds — in-domain

| Config | Protocol |
|--------|----------|
| `frost_pc_in_mn40.json` | ModelNet40 (5 sessions) |
| `frost_pc_in_sn55.json` | ShapeNet55 (7 sessions) |
| `frost_pc_in_co3d.json` | CO3D (6 sessions) |

## Point clouds — cross-domain

| Config | Protocol |
|--------|----------|
| `frost_pc_cross_mn40_scanobj.json` | MN → ScanObjectNN (4 sessions) |
| `frost_pc_cross_sn55_scanobj.json` | SN → ScanObjectNN (4 sessions) |
| `frost_pc_cross_sn55_co3d.json` | SN → CO3D (11 sessions) |

**Buffer (EB):** no separate JSON. Reproduce scripts flip via `--buffer` /
`--buffer-free` (`model_name=frost` + `fscil_preset=canonical`).

## Images (paper = buffer-free only)

| Config | Dataset |
|--------|---------|
| `frost_img_cub200.json` | CUB-200 |
| `frost_img_inf200.json` | iNF200 |
| `frost_img_aircraft.json` | FGVC-Aircraft |

## Example

```bash
# 3D — first seed only (default)
aprun python train_fscil.py --config configs/frost_pc_in_mn40.json

# 3D — all paper seeds
aprun python train_fscil.py --config configs/frost_pc_in_mn40.json --fullseeds

# 3D EB via script flag
sbatch scripts/reproduce_indomain.sh
sbatch --array=0-29 scripts/reproduce_indomain.sh --fullseeds

# Image
aprun python train_fscil_image.py --config configs/frost_img_cub200.json
sbatch --array=0-14 scripts/reproduce_image.sh --fullseeds
```

## Baseline

| Config | Notes |
|--------|-------|
| `ranpac_fscil_mn40_5t_fscil3d.json` | RanPAC FSCIL MN40 baseline (standalone) |
