# Scripts

## Paper entrypoints (this directory)

| Script | Role |
|--------|------|
| `reproduce_indomain.sh` | 3D in-domain FSCIL (EF+EB) |
| `reproduce_crossdomain.sh` | 3D cross-domain FSCIL (EF+EB) |
| `reproduce_image.sh` | Image FSCIL (EF only) |
| `frost_common.sh` | Shared helpers (sourced by reproduce_*) |
| `summarize_pride_multiseed.py` | Aggregate 3D multi-seed logs |
| `summarize_fscil_results.py` | Shared log parser |
| `summarize_frost_image_multiseed.sh` | Aggregate image multi-seed logs |
| `check_frost_image_data.sh` | Preflight image data + ViT weights |
| `cleanup_experiments.sh` | Prune experiment trees |
| `leonardo_remote.sh` | Leonardo sync helpers |

Default: first paper seed only. Pass `--fullseeds` for all five.

## Data prep — `datasets/preprocess/`

Downloads, FPS+normals builders, protocol printers.

```bash
bash datasets/preprocess/download_cub200.sh
bash datasets/preprocess/download_vit_in21k.sh
aprun python datasets/preprocess/build_co3d_official_1024_normals.py
```

## Archive — `scripts/_archive/`

Local sweeps, ablation launchers, one-off smoke tests. Not on the paper reproduce path.
