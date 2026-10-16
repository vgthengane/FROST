# Dataset preprocess / download

Helpers that live next to `datasets/` loaders. Training entrypoints stay under `scripts/`.

| Script | Role |
|--------|------|
| `download_cub200.sh` | CUB-200-2011 → `_datasets/images/CUB200` |
| `download_fgvc_aircraft.sh` | FGVC-Aircraft → `_datasets/images/FGVC-Aircraft` |
| `download_inf200.sh` | iNF200 staging notes / layout |
| `download_vit_in21k.sh` | Offline ViT-B/16 ImageNet-21k weights |
| `download_co3d_index_minimal.sh` | Official CO3D ply (needs local `third_party/co3d`) |
| `build_co3d_official_1024_normals.py` | CO3D ply → 1024-pt FPS+normal caches |
| `build_scanobjectnn_1024_normals.py` | ScanObjectNN normals caches |
| `patch_shapenet55_bicycle.py` | ShapeNet55 bicycle class patch |
| `print_fscil_protocol.py` | Print FSCIL-3D session schedules |

See `DATASETS.md` for licences and layout.
