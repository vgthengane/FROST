# Experimental setup (paper recipe)

Code-aligned reference for the **locked FROST** FSCIL recipe.  
Configs: `configs/base/{pclouds,images}.json` + `configs/frost_*.json` ([`configs/README.md`](configs/README.md)).  
Method: `frost/` · presets: `utils/fscil_ablation_config.py` · flatten: `utils/config_load.py`.

For launchers and seeds see [`README.md`](README.md) and [`scripts/README.md`](scripts/README.md).

---

## 1. Software and hardware

| Item | Value |
|------|-------|
| Framework | PyTorch |
| Entry | `train_fscil.py` (3D) / `train_fscil_image.py` (2D) |
| Python | Plain `python` (install guide TBD). Optional: `export FROST_PYTHON="aprun python"` on EuroHPC |
| GPU / mem (SLURM defaults) | 1 GPU, 32–64 GB host, 8–24 CPUs |
| Paper seeds | `1947, 1991, 2001, 2008, 2019` (default run = first only; `--fullseeds` for all) |

---

## 2. Backbones

### Point clouds

| Item | Value | Config |
|------|-------|--------|
| Method | `frost` / `frost_bufferfree` | `model_name` |
| Encoder | Uni3D-Base (frozen) | `pretrained_pc` |
| Backbone | `uni3d_base_patch14_448` | `backbone_type` |
| Embed dim \(d\) | 1024 | `embed_dim` |
| Points \(M\) | 1024 + normals | `num_point`, `use_normals` |
| Batch size | 8 | `batch_size` |

### Images

| Item | Value | Config |
|------|-------|--------|
| Encoder | ViT-B/16 ImageNet-21k + adapter | `backbone_type` |
| Data root | `_datasets/images` | `data_path` |
| Sessions | `datasets/session_images` | `sessions_root` |
| Batch size | 48 | `batch_size` |

---

## 3. Paper modules

### Adapter

| Item | Paper value |
|------|-------------|
| Type | AdaptFormer bottleneck, \(d'{=}64\) (`ffn_num`) |
| Schedule | **Base session only**, 10 epochs |
| Novel sessions | adapter frozen (`increment_warmup=false`) |
| 3D LR / WD | \(10^{-3}\) / \(10^{-4}\) (SGD, cosine) |
| Image LR | \(10^{-2}\) (SGD) |

### CorrMem (`frost/corrmem.py`)

| Item | Paper value |
|------|-------------|
| Lift width \(V\) | **2000** (`corrmem.lift_v` → `ridge_rp_dim`) |
| Map | \(\mathrm{ReLU}(\mathbf{z} W_{\mathrm{rand}})\), \(W\) fixed from run seed |
| Solver | Analytical ridge on sufficient stats \(\mathbf{G},\mathbf{Q}\) |
| \(\lambda\) | Grid search on task 0 |

### ProtoMem (`frost/protomem.py`)

| Item | Paper value |
|------|-------------|
| Insert | **mean only** (1 key/class) |
| Evolution | **static** (no merge/split in main row) |
| Pooling | max (preset) |

### Fusion

| Item | Paper value |
|------|-------------|
| Mode | `corr_proto` (`classifier_mode=ridge_cache` internally) |
| \(\alpha\) | **0.5** flat (CorrMem weight) |

Score: \(\alpha\,\ell^{\mathrm{CorrMem}}+(1-\alpha)\,\ell^{\mathrm{ProtoMem}}\).

---

## 4. EF vs EB

| Variant | `model_name` | `fscil_preset` | Disk exemplars |
|---------|--------------|----------------|----------------|
| **EF** (main) | `frost_bufferfree` | `exemplar_buffer_free` | 0 |
| **EB** (twin) | `frost` | `canonical` | 1 shot/old class |

Reproduce scripts flip EF/EB with `--buffer-free` / `--buffer` (no separate JSON for EB).

Conflict replay uses **train-side** confusion (`conflict_replay_use_test_cm=false`).

---

## 5. FSCIL protocols

Session class IDs: `datasets/session_pclouds/` via `datasets/fscil3d_index.py`.

| Protocol | Sessions | Base + novel |
|----------|----------|--------------|
| ModelNet40 | 5 | 20 + 4×5 |
| ShapeNet55 | 7 | 25 + 5×6 |
| CO3D | 6 | 25 + 5×5 |
| MN → ScanObjectNN | 4 | 26 + … |
| SN → ScanObjectNN | 4 | 44 + … |
| SN → CO3D | 11 | 39 + 5×10 |
| CUB / iNF / Aircraft | LGSP sessions | see `session_images/` |

Novel sessions: **5 shots/class** (`increment_train_per_class: 5`).

---

## 6. Per-session pipeline (sketch)

1. Adapter warmup (task 0 only, paper recipe)  
2. Freeze backbone; extract embeddings  
3. Insert ProtoMem means  
4. Update CorrMem ridge stats  
5. Optional conflict replay (train CM)  
6. Classify with fusion  

---

## 7. Config map

| Protocol | Config |
|----------|--------|
| MN40 / SN55 / CO3D | `configs/frost_pc_in_{mn40,sn55,co3d}.json` |
| Cross-domain | `configs/frost_pc_cross_*.json` |
| CUB / iNF / Aircraft | `configs/frost_img_*.json` |

Paper blocks in JSON (`adapter`, `corrmem`, `protomem`, `fusion`) are flattened by `utils/config_load.py`.

---

## 8. Metrics

| Symbol | Meaning |
|--------|---------|
| \(\mathcal{A}_b\) | Session-0 total accuracy |
| \(\mathcal{A}_T\) | Last-session total accuracy |
| \(\bar{\mathcal{A}}\) | Mean of per-session total accuracies (`Average Accuracy (CNN)`) |
| \(\Delta\) | \(100(\mathcal{A}_b-\mathcal{A}_T)/\mathcal{A}_b\) |

**Do not** report \(\bar{\mathcal{A}}=(\mathcal{A}_b+\mathcal{A}_T)/2\).

---

## 9. Writing notes

- \(\lambda\) = ridge penalty; \(\alpha\) = fusion weight — do not conflate.  
- Main tables: **mean-only**, \(V{=}2000\), \(\alpha{=}0.5\). Other inserts / \(V{=}1000\) are ablations only.  
- Cite the reproduce launcher or config used for each table row.
