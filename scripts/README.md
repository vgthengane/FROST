# Scripts

## Paper entrypoints

| Script | Role |
|--------|------|
| `reproduce_indomain.sh` | 3D in-domain FSCIL (MN40, SN55, CO3D; EF+EB) |
| `reproduce_crossdomain.sh` | 3D cross-domain FSCIL (EF+EB) |
| `reproduce_image.sh` | Image FSCIL (CUB, iNF200, Aircraft; EF) |
| `frost_common.sh` | Shared helpers (sourced by reproduce_*; do not sbatch) |
| `summarize_frost_multiseed.py` | Aggregate 3D multi-seed logs |
| `summarize_fscil_results.py` | Shared log parser |
| `summarize_frost_image_multiseed.sh` | Aggregate image multi-seed logs |
| `check_frost_image_data.sh` | Preflight image data + ViT weights |
| `cleanup_experiments.sh` | Prune experiment trees (optional) |

Default: **first paper seed only**. Pass `--fullseeds` for all five.

```bash
sbatch scripts/reproduce_indomain.sh
sbatch --array=0-29 scripts/reproduce_indomain.sh --fullseeds
sbatch scripts/reproduce_image.sh --fullseeds
```

## Data prep — `datasets/preprocess/`

```bash
bash datasets/preprocess/download_cub200.sh
bash datasets/preprocess/download_vit_in21k.sh
python datasets/preprocess/build_co3d_official_1024_normals.py
```

Scripts call plain `python` by default. On EuroHPC: `export FROST_PYTHON="aprun python"`.

See [`DATASETS.md`](../DATASETS.md) and [`datasets/preprocess/README.md`](../datasets/preprocess/README.md).

## Not for release tables

| Path | Role |
|------|------|
| `scripts/_archive/` | Old sweeps / smoke tests |
| `leonardo_remote.sh` | Optional cluster SSH helper (lab-only) |
