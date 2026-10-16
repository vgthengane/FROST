# FROST datasets

Raw data lives under **local** `_datasets/` (gitignored). This file is part of the **code release** so others can reproduce downloads and licences without shipping the data.

Do **not** commit or redistribute the contents of `_datasets/`.

## Layout (canonical)

```text
_datasets/
  pclouds/
    ModelNet40/          # was modelnet40_normal_resampled
    ShapeNet55/
    ScanObjNN/           # was scanobjectnn (main_split / main_split_nobg)
    CO3D/                # Official CO3D → our 1024-pt FPS+normal caches
  images/
    CUB200/              # downloaded as CUB_200_2011, renamed
    FGVC-Aircraft/       # downloaded as fgvc-aircraft-2013b, renamed
    iNF200/              # train_mini/ + val/
  # session indexes (in-repo, not under _datasets):
  #   datasets/session_pclouds/   # FSCIL-3D PC protocols
  #   datasets/session_images/    # CUB / Aircraft / iNF200
```

Config paths in this repo point at the canonical layout, e.g.:

- `_datasets/pclouds/ModelNet40`
- `_datasets/pclouds/ShapeNet55`
- `_datasets/pclouds/ScanObjNN/main_split_nobg`
- `_datasets/pclouds/CO3D`
- `_datasets/images` (parent of CUB200 / FGVC-Aircraft / iNF200)
- `datasets/session_pclouds` / `datasets/session_images`

---

## Can we redistribute the datasets?

**Short answer: no — do not bundle or publicly rehost these datasets with the paper code.**

| Advice | Why |
|--------|-----|
| **Do not** put `_datasets/` on GitHub / Zenodo / anonymous Dropbox with the paper | Most sources forbid redistribution or require each user to accept terms first |
| **Do** ship download / build scripts + **this** `DATASETS.md` | Others reproduce by obtaining data from official sources |
| **Do** ask users to **read and accept each dataset’s licence / Terms of Use** before download | Especially ShapeNet, ScanObjectNN, CO3D, FGVC-Aircraft |
| **Optional** | Share only **pointers** (URLs, script names, expected folder layout), never a zip of ModelNet/ShapeNet/ScanObjNN/CO3D/CUB/FGVC-Aircraft/iNF200 |

We are **not** lawyers. Licence summaries below are **best-effort** for researchers; always verify the official page before use or redistribution.

---

## Point clouds (`pclouds/`)

### ModelNet40 → `pclouds/ModelNet40`

| | |
|--|--|
| **What we use** | ModelNet40 with normals, resampled (1024 pts) — standard CL / FSCIL-3D style tree |
| **Official** | [Princeton ModelNet](https://modelnet.cs.princeton.edu/) |
| **How we obtained it** | Standard ModelNet40 download used in Uni3D / FSCIL-3D style pipelines; stored historically as `modelnet40_normal_resampled` |
| **What we did** | Stored under `pclouds/ModelNet40` |
| **Licence / ToU** | Check Princeton ModelNet download page. Treat as **research use**; **do not redistribute** the zip/files yourself |
| **Cite** | Wu et al., *3D ShapeNets*, CVPR 2015 |

### ShapeNet55 → `pclouds/ShapeNet55`

| | |
|--|--|
| **What we use** | ShapeNet-derived point clouds for FSCIL SN55 protocol (as in our configs) |
| **Official** | [ShapeNet](https://shapenet.org/) (registration + Terms of Use) |
| **How we obtained it** | ShapeNet access under Princeton/Stanford ToU; processed PC layout used by this codebase |
| **What we did** | Stored under `pclouds/ShapeNet55` |
| **Licence / ToU** | **Non-commercial research/education only.** Redistribution only to colleagues who **first accept the same ShapeNet terms**. See [ShapeNet Terms of Use](https://shapenet.org/) / HF ShapeNetCore card |
| **Cite** | Chang et al., *ShapeNet*, arXiv 2015 |

**Must accept ShapeNet ToU before download.** Do not host a public mirror.

### ScanObjectNN → `pclouds/ScanObjNN`

| | |
|--|--|
| **What we use** | `main_split` / `main_split_nobg` for cross-domain FSCIL (MN/SN → Scan) |
| **Official** | [ScanObjectNN project](https://hkust-vgd.github.io/scanobjectnn/) — download after **Terms of Use** form |
| **How we obtained it** | Official processed release after ToU; layout `scanobjectnn/main_split_nobg` |
| **What we did** | Moved under `pclouds/ScanObjNN`; optional normals build via `datasets/preprocess/build_scanobjectnn_1024_normals.py` |
| **Licence / ToU** | Dataset requires accepting ScanObjectNN Terms of Use. GitHub code MIT ≠ licence for the **data** |
| **Cite** | Uy et al., *ScanObjectNN*, ICCV 2019 |

**Must accept ToU before download.** Do not redistribute the HDF5/PC dumps.

### CO3D → `pclouds/CO3D`

| | |
|--|--|
| **What we use** | Official Meta CO3D `pointcloud.ply` → our `co3d_*_1024pts_fps_normal.dat` caches |
| **Upstream** | [Meta CO3D](https://ai.facebook.com/datasets/co3d-dataset/) |
| **Session indexes** | Vendored in `datasets/session_pclouds/co3d/` (FSCIL-3D class/session schedule) |
| **How we build** | Download official ply (optional helper: `download_co3d_index_minimal.sh`), then `build_co3d_official_1024_normals.py` |
| **Licence / ToU** | Typically **CC BY-NC 4.0** — **do not rehost** |
| **Cite** | Reizenstein et al., *Common Objects in 3D*, CVPR 2021; FSCIL-3D protocol as appropriate |

```bash
# After official CO3D ply is on disk (see download_co3d_index_minimal.sh):
aprun python datasets/preprocess/build_co3d_official_1024_normals.py
```

---

## Images (`images/`)

Parent path for image FSCIL: `_datasets/images` (configs keep `data_path: "_datasets/images"`).

### CUB-200-2011 → `images/CUB200`

| | |
|--|--|
| **Official** | [Caltech CUB-200-2011](https://www.vision.caltech.edu/datasets/cub_200_2011/) / [Caltech DATA](https://data.caltech.edu/records/65de6-vp158) |
| **How we downloaded** | `datasets/preprocess/download_cub200.sh` → extracts then renames to `CUB200` |
| **What we did** | Folder name `CUB200` (session indexes use `CUB200/images/...`) |
| **Licence / ToU** | Academic / research use via Caltech; **do not redistribute** the tarball as part of FROST |
| **Cite** | Wah et al., *CUB-200-2011*, Caltech Tech Report 2011 |

```bash
bash datasets/preprocess/download_cub200.sh _datasets/images
```

### FGVC-Aircraft → `images/FGVC-Aircraft`

| | |
|--|--|
| **Official** | [Oxford VGG FGVC-Aircraft](https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/) |
| **How we downloaded** | `datasets/preprocess/download_fgvc_aircraft.sh` → extracts then renames to `FGVC-Aircraft` |
| **What we did** | Folder name `FGVC-Aircraft` |
| **Licence / ToU** | **Non-commercial research**; image copyrights remain with original photographers — contact authors for other uses |
| **Cite** | Maji et al., *Fine-Grained Visual Classification of Aircraft*, 2013 |

```bash
bash datasets/preprocess/download_fgvc_aircraft.sh _datasets/images
```

### iNF200 → `images/iNF200`

| | |
|--|--|
| **Official staging** | Hugging Face [`tldoan/iNF200`](https://huggingface.co/datasets/tldoan/iNF200) (CLIP-M3 / LGSP fungi FSCIL) |
| **How we downloaded** | `datasets/preprocess/download_inf200.sh` → `iNF200/train_mini` + `iNF200/val` |
| **What we did** | Kept name `iNF200` under `images/` |
| **Licence / ToU** | Follow the HF dataset card and any upstream fungi / iNaturalist-derived terms; **do not rehost** without checking the card |
| **Cite** | As required by CLIP-M3 / LGSP / iNF200 authors |

```bash
bash datasets/preprocess/download_inf200.sh _datasets/images
```

Session indexes (not the images) live in-repo under `datasets/session_images/` (`cub200`, `FGVC-Aircraft`, `iNF200`). Configs set `"sessions_root": "datasets/session_images"` (repo-relative).

Preflight:

```bash
bash scripts/check_frost_image_data.sh --data-path _datasets/images
```

---

## Recommended reproduce checklist (for others)

1. Clone FROST code only (no `_datasets/` in git).
2. **Read and accept** each dataset’s ToU / licence (especially ShapeNet, ScanObjectNN, CO3D, Aircraft).
3. Download with the scripts above (or official pages) into the layout in this file.
4. Place Uni3D / ViT weights under `_experiments/pretrained_weights/` (`datasets/preprocess/download_vit_in21k.sh`).
5. Run `scripts/reproduce_indomain.sh` / `reproduce_crossdomain.sh` / `reproduce_image.sh`.

See also `_notes/REPRODUCE.md`.

---

## Paper / release wording (suggested)

> Datasets are obtained from their official sources under each provider’s licence or Terms of Use. We do **not** redistribute ModelNet40, ShapeNet, ScanObjectNN, CO3D, CUB-200-2011, FGVC-Aircraft, or iNF200 with this release. Scripts to download and arrange data into the layout expected by FROST are provided in `scripts/` and documented in `DATASETS.md`. Users must accept the relevant licences before downloading.

---

## Contact / maintenance

If a download URL moves, update the corresponding `datasets/preprocess/download_*.sh` and this file.  
When in doubt about redistribution: **link, don’t zip.**
