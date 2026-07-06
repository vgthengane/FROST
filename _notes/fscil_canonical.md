# FSCIL canonical experiment track

**Primary method focus:** SelfEvo `ridge_cache` on **FSCIL-3D class order** (few-shot incremental).

**Reference upstream:** `external/FSCIL-3D/` (ECCV 2022). Index files:
`external/FSCIL-3D/data/index_files/<protocol>/{train,test}/session_*.txt`

**Code integration:**
- `datasets/fscil3d_index.py` — protocol metadata, class order per session
- `datasets/fscil_protocol.py` — shot caps, old-class replay, task datasets
- `datasets/modelnet40.py`, `datasets/shapenet53.py` — apply `fscil3d_protocol` when set

---

## Task definition (FSCIL-3D within-dataset)

| Protocol | Sessions | Base | Increments | Total classes | Config |
|----------|----------|------|------------|---------------|--------|
| `modelnet` | 5 | 20 | 4×5 | 40 | `selfevo_fscil_mn40_5t_flex_compare.json` |
| `shapenet` | 7 | 25 | 6×5 | 55 | `selfevo_fscil_sn55_7t_flex_compare.json` |
| `co3d` | 6 | 25 | 5×5 | 50 | **not implemented** (metadata only) |

### Per-session training rules (our code)

| Session | Novel classes | Training data |
|---------|---------------|---------------|
| **0 (base)** | `init_cls` classes | **Full** train split per base class |
| **1…T−1** | `increment` classes | **5 shots** per novel class (`increment_train_per_class: 5`) |
| **All t>0** | Old classes | **1 shot** replay per old class (`fscil_old_shot_replay: 1`, fixed seed) |

### Evaluation

- After session \(t\): top-1 on **all classes seen so far** (cumulative test split)
- Metrics: **AA**, **Last**, **Δ** (session-0 total acc − final total acc), **Base@Last**, harmonic (trainer logs)
- Class order: **fixed** (`shuffle: false`, `fscil3d_protocol` set)

### Important caveat vs 3D-FLEG paper

We match **FSCIL-3D session class order** and shot counts, but sample **FPS point clouds + random 5-shot subsampling** from our caches — not FSCIL-3D’s prebuilt `.pt` index tensors. Compare split schedule, not sample identity.

---

## Canonical launcher (flex_compare)

```bash
sbatch scripts/flex_compare.sh              # MN40 + SN55
sbatch scripts/flex_compare.sh --mn40-only
bash scripts/flex_compare.sh --local --sn55-only
```

| Item | Path |
|------|------|
| Logs | `_experiments/flex_compare/<method>_seed=<seed>.log` |
| Methods | `seel_mn40_flex_compare`, `seel_sn55_flex_compare` |
| Summary | `_experiments/flex_compare/results_table.md` |
| Per-dataset MD | `_experiments/flex_compare/results.md` |

**Current results (seed 2008):** MN40 Last 90.96 (+3.66 vs 3D-FLEG); SN55 Last 88.67 (+5.67 vs 3D-FLEG).

### Main method config knobs (flex_compare)

- `classifier_mode: ridge_cache`, `ridge_cache_alpha: 0.85` (lower on few-shot sessions via `fscil_increment_ridge_alpha`)
- `fscil_shot_protocol: true`, adapter warmup task-0 + increments
- `old_data_source: cache` — adaptation batch old part = cache keys, not full old train
- `cache_budget: 200` (MN40/SN55 FSCIL), `max_prototypes_per_class: 3`
- `enable_gated_growth: true`, `conflict_replay: true`, `conflict_replay_on_test: true`

---

## Parallel track: standard CIL (not FSCIL)

Do **not** mix with FSCIL tables without footnotes.

| Track | Tasks | Shots | Scripts |
|-------|-------|-------|---------|
| ModelNet40 CIL | 10×4 classes | Full train / task | `core.sh`, `ours.sh` |
| ShapeNet53 CIL | 10×~5 classes | Full train / task | `core_shapenet53.sh`, `ours_shapenet53.sh` |

---

## Cross-dataset protocols (roadmap — not implemented)

Defined in `fscil3d_index.py`; need dataset loaders + configs.

| Protocol | Sessions | Description |
|----------|----------|-------------|
| `modelnet_scanobjectnn` | 4 | Train MN → test ScanObjectNN |
| `shapenet_scanobjectnn` | 4 | Train ShapeNet → test ScanObjectNN |
| `shapenet_co3d` | 11 | ShapeNet base + CO3D increments (39+11 classes) |
| `co3d` | 6 | Within CO3D only |

**Next engineering steps:**
1. ScanObjectNN or CO3D dataloader aligned to FSCIL-3D index paths
2. `configs/selfevo_fscil_*_<protocol>_flex_compare.json` mirroring MN40/SN55 layout
3. Extend `flex_compare.sh` or add `scripts/fscil_cross.sh`
4. Multi-seed FSCIL campaign (`1947,1991,2001,2008,2019`) + RanPAC FSCIL-3D baseline (`configs/ranpac_fscil_mn40_5t_fscil3d.json`)

---

## Archived (do not use)

Superseded tuning configs and `fscil_style_*` logs → `_archive/`. See `_archive/README.md`.
