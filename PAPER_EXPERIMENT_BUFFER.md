# PAPER_EXPERIMENT_BUFFER.md — With-buffer FSCIL results

**Last updated:** 2026-07-12  
**Companion to:** `PAPER_EXPERIMENT.md` (buffer-free / main paper recipe)  
**Method:** **PRidE**  
**Seed:** 2008  
**Adapter:** base-only, `tuned_epoch=5` (EP5 tables); EP10 noted separately  
**Classifier:** `ridge_cache`, α=0.85  
**Buffer:** canonical exemplar buffer (`fscil_preset=canonical` / `model_name=selfevo`) — 1 old-shot per seen class on disk  

This file is the **with-buffer numbers + log map**. Prefer `00_FINAL_EP5_BASEONLY` over legacy `00_BUF_CAMPAIGN_EP5*`.

---

## 0. Caveats (read before pasting tables)

| Issue | Detail |
|-------|--------|
| **No full `hybrid3_st` buf** | Close–far static × buffer was never fully run. In-domain close–far **proxy** = `hybrid4_evo` buf (mean+close+far + evolve, P≤4). Only incomplete smoke: `smoke_final_mn40_hybrid3_st_buf.log`. |
| **Pooling not uniform** | MN40/SN55 FINAL buf logs use **`logsumexp`** (config default). CO3D + cross-dataset use **`max`**. Max+buf for MN40/SN55 is still pending `00_FINAL_EP5_MAX_POOL/`. |
| **Legacy campaigns** | `00_BUF_CAMPAIGN_EP5*` used **test-aware** conflict replay — do **not** mix into final paper tables without a footnote. |
| **Main paper default** | Still **buffer-free** (`PAPER_EXPERIMENT.md`). This file is the buffer ablation / supplementary twin. |

**Domain-aware recipe under buffer (this file):**

- In-domain → **`hybrid4_evo` buf** (close–far proxy)  
- Cross-dataset → **`mean_st` buf**

---

## 1. Log directories

| Directory | What is there | Pooling | Use for paper? |
|-----------|---------------|---------|----------------|
| `_experiments/00_FINAL_EP5_BASEONLY/` | mean / pshot* / hybrid4 × **buf** × 6 protocols | MN40/SN55 = **logsumexp**; CO3D+cross = **max** | **Yes (primary)** |
| `_experiments/00_FINAL_EP10_BASEONLY/` | Same matrix, 10-ep base warmup | same | Optional appendix |
| `_experiments/00_BUF_CAMPAIGN_EP5/` | Dedicated buf vs buffree, **pshot + rules** | same as configs | Legacy only |
| `_experiments/00_BUF_CAMPAIGN_EP5_MEAN/` | Dedicated buf vs buffree, **mean** | same | Legacy only |
| `_experiments/00_FINAL_EP5_HYBRID_CLOSEFAR/` | Insert ladder | — | **buffree only** |
| `_experiments/00_FINAL_EP5_HYBRID_POOLING_SWAP/` | Pooling swap | — | **buffree only** |
| `_experiments/00_FINAL_EP5_MAX_POOL/` | MN40/SN55 forced max | **max** | Pending (includes buf slots) |

**Scripts**

| Script | Role |
|--------|------|
| `scripts/simpl_runs_final_ep5_baseonly.sh` | Final EP5 matrix (buf + buffree) |
| `scripts/simpl_runs_final_ep10_baseonly.sh` | EP10 matrix |
| `scripts/simpl_runs_baseonly_ep5_buffer_campaign.sh` | Legacy `00_BUF_CAMPAIGN_EP5` |
| `scripts/simpl_runs_baseonly_ep5_buffer_mean_campaign.sh` | Legacy `00_BUF_CAMPAIGN_EP5_MEAN` |
| `scripts/simpl_runs_max_pool_mn40_sn55_fill.sh` | Fill MN40/SN55 under max (buf + buffree) |

```bash
aprun python scripts/summarize_fscil_final_campaign.py --campaign ep5
# Filter Buffer rows from the printed tables
```

---

## 2. Main method rows — domain-aware **with buffer** (EP5)

Same insert policy as buffree main: close–far in-domain (here `hybrid4_evo`), mean cross-dataset.

| Protocol | Insert (buf) | Pool | AA | Last | Novel@Last | Harm@Last | Session curve (CNN top1) | Log |
|----------|--------------|------|---:|-----:|-----------:|----------:|--------------------------|-----|
| **MN40** | hybrid4_evo | logsumexp | 93.31 | **91.37** | 78.24 | 85.72 | `95.5 93.6 93.5 92.7 91.4` | `00_FINAL_EP5_BASEONLY/final_mn40_hybrid4_evo_buf.log` |
| **SN55** | hybrid4_evo | logsumexp | 89.79 | **88.72** | 77.63 | 83.81 | `91.8 90.9 89.7 89.6 89.2 88.5 88.7` | `.../final_sn55_hybrid4_evo_buf.log` |
| **CO3D** | hybrid4_evo | max | 69.27 | **66.11** | 31.45 | 44.62 | `76.8 70.7 68.5 67.7 65.8 66.1` | `.../final_co3d_hybrid4_evo_buf.log` |
| **MN→ScanObj** | mean_st | max | 89.55 | **87.16** | 69.89 | 79.68 | `92.2 90.9 87.9 87.2` | `.../final_mn_scanobj_mean_st_buf.log` |
| **SN→ScanObj** | mean_st | max | 91.98 | **90.74** | 65.58 | 76.99 | `93.3 92.5 91.4 90.7` | `.../final_sn_scanobj_mean_st_buf.log` |
| **SN→CO3D** | mean_st | max | 87.47 | **82.67** | 46.82 | 62.06 | `91.7 91.3 90.6 89.6 88.8 87.5 86.4 85.4 84.6 83.6 82.7` | `.../final_sn_co3d_mean_st_buf.log` |

**Caption idea:** “Same domain-aware insert policy as the main buffer-free recipe, with a 1-shot old-class exemplar buffer. MN40/SN55 still use logsumexp pooling in these logs.”

---

## 3. Buffer vs buffer-free (same regime, EP5 FINAL)

Δ = Last(buffree) − Last(buf). Negative ⇒ buffer slightly better on Last.

### 3.1 Domain-aware pair

| Protocol | Buf Last / Novel / Harm | Buffree Last / Novel / Harm | ΔLast |
|----------|------------------------:|----------------------------:|------:|
| MN40 (hybrid4) | 91.37 / 78.24 / 85.72 | 91.13 / 78.04 / 85.50 | −0.24 |
| SN55 (hybrid4) | 88.72 / 77.63 / 83.81 | 88.71 / 76.70 / 83.34 | −0.01 |
| CO3D (hybrid4) | 66.11 / 31.45 / 44.62 | 65.65 / 28.01 / 41.11 | −0.46 |
| MN→ScanObj (mean) | 87.16 / 69.89 / 79.68 | 87.16 / 69.47 / 79.45 | 0.00 |
| SN→ScanObj (mean) | 90.74 / 65.58 / 76.99 | 90.62 / 64.54 / 76.26 | −0.12 |
| SN→CO3D (mean) | 82.67 / 46.82 / 62.06 | 82.16 / 45.55 / 60.86 | −0.51 |

**Paper claim:** exemplar buffer is **optional**; gains are small (≤0.5 Last on most protocols). Buffer-free remains the main recipe.

### 3.2 Full buf matrix (Last / Novel / Harm)

Source: `00_FINAL_EP5_BASEONLY/final_<proto>_<regime>_buf.log`

#### MN40 (logsumexp)

| Regime | Last / Novel / Harm |
|--------|--------------------:|
| mean_st | 91.33 / 78.04 / 85.60 |
| pshot_st | 91.29 / 78.24 / 85.68 |
| pshot_evo | 91.41 / 78.82 / 86.03 |
| pshot_evo5 | 91.29 / 78.24 / 85.68 |
| hybrid4_evo | **91.37 / 78.24 / 85.72** |

#### SN55 (logsumexp)

| Regime | Last / Novel / Harm |
|--------|--------------------:|
| mean_st | 88.83 / 77.09 / 83.60 |
| pshot_st | 88.78 / 77.80 / 83.93 |
| pshot_evo | 88.76 / 78.29 / 84.16 |
| pshot_evo5 | 88.78 / 77.80 / 83.93 |
| hybrid4_evo | **88.72 / 77.63 / 83.81** |

#### CO3D (max)

| Regime | Last / Novel / Harm |
|--------|--------------------:|
| mean_st | 65.65 / 25.80 / 38.76 |
| pshot_st | 66.11 / 31.20 / 44.38 |
| pshot_evo | 66.40 / 32.68 / 45.84 |
| pshot_evo5 | 66.11 / 31.20 / 44.38 |
| hybrid4_evo | **66.11 / 31.45 / 44.62** |

#### MN→ScanObj (max)

| Regime | Last / Novel / Harm |
|--------|--------------------:|
| mean_st | **87.16 / 69.89 / 79.68** |
| pshot_st | 86.76 / 69.05 / 79.03 |
| pshot_evo | 86.61 / 68.42 / 78.61 |
| pshot_evo5 | 86.76 / 69.05 / 79.03 |
| hybrid4_evo | 86.91 / 69.68 / 79.44 |

#### SN→ScanObj (max)

| Regime | Last / Novel / Harm |
|--------|--------------------:|
| mean_st | **90.74 / 65.58 / 76.99** |
| pshot_st | 90.79 / 66.78 / 77.79 |
| pshot_evo | 90.76 / 66.27 / 77.45 |
| pshot_evo5 | 90.79 / 66.78 / 77.79 |
| hybrid4_evo | 90.85 / 67.30 / 78.15 |

#### SN→CO3D (max)

| Regime | Last / Novel / Harm |
|--------|--------------------:|
| mean_st | **82.67 / 46.82 / 62.06** |
| pshot_st | 82.37 / 45.79 / 61.12 |
| pshot_evo | 82.44 / 46.36 / 61.61 |
| pshot_evo5 | 82.37 / 45.79 / 61.12 |
| hybrid4_evo | 82.04 / 44.40 / 59.86 |

---

## 4. EP10 domain-aware buf (optional)

Source: `00_FINAL_EP10_BASEONLY/final10_*_buf.log`

| Protocol | Insert | Last / Novel / Harm | Session curve |
|----------|--------|--------------------:|---------------|
| MN40 | hybrid4_evo | 91.41 / 78.43 / 85.84 | `95.8 94.0 93.8 92.9 91.4` |
| SN55 | hybrid4_evo | 88.94 / 78.07 / 84.14 | `91.9 91.0 89.9 89.8 89.4 88.6 88.9` |
| CO3D | hybrid4_evo | 69.23 / 38.57 / 51.76 | `78.3 73.8 71.6 70.6 69.0 69.2` |
| MN→ScanObj | mean_st | 87.42 / 70.32 / 80.03 | `92.5 91.5 88.6 87.4` |
| SN→ScanObj | mean_st | 90.90 / 65.23 / 76.82 | `93.6 92.8 91.5 90.9` |
| SN→CO3D | mean_st | 82.11 / 43.65 / 59.23 | `92.1 91.5 91.0 89.2 88.6 87.4 86.5 85.4 84.6 83.2 82.1` |

---

## 5. Legacy buffer campaigns (do not mix)

Test-aware conflict replay. Kept for audit only.

### `00_BUF_CAMPAIGN_EP5` — pshot + rules + buf

| Protocol | Last / Novel / Harm | Curve |
|----------|--------------------:|-------|
| MN40 | 91.41 / 78.82 / 86.03 | `95.5 93.6 93.6 92.9 91.4` |
| SN55 | 88.76 / 78.29 / 84.16 | `91.8 90.9 89.7 89.7 89.3 88.5 88.8` |
| CO3D | 67.96 / 35.87 / 49.10 | `76.8 71.7 70.0 68.6 67.7 68.0` |
| MN→ScanObj | 86.45 / 67.79 / 78.20 | `92.2 91.0 87.3 86.5` |
| SN→ScanObj | 90.50 / 65.92 / 77.12 | `93.3 92.1 91.0 90.5` |
| SN→CO3D | 82.01 / 45.09 / 60.44 | `91.7 … 82.0` |

Logs: `abl_<proto>_ep5_buf.log`

### `00_BUF_CAMPAIGN_EP5_MEAN` — mean + rules + buf

| Protocol | Last / Novel / Harm |
|----------|--------------------:|
| MN40 | 91.29 / 77.06 / 85.09 |
| SN55 | 88.62 / 76.70 / 83.29 |
| CO3D | 66.86 / 31.94 / 45.25 |
| MN→ScanObj | 86.71 / 68.84 / 78.89 |
| SN→ScanObj | 90.48 / 65.75 / 77.01 |
| SN→CO3D | 81.83 / 44.17 / 59.61 |

Logs: `abl_<proto>_ep5_mean_buf.log`

---

## 6. Gaps checklist (buffer) + how to fill

| Item | Status | Submit |
|------|--------|--------|
| EP5 FINAL buf matrix (mean / pshot* / hybrid4 × 6) | **Done** | — |
| EP10 FINAL buf matrix | **Done** | — |
| Domain-aware buf main rows (§2) | **Done** (hybrid4 proxy) | — |
| Insert ladder **buf** (hybrid3 / nearmean / fps5 / hybrid4) × 6 | **Missing** | `sbatch scripts/simpl_runs_final_hybrid_closefar_buf.sh` (24 jobs) |
| `hybrid3_st` **buf + max** × 6 (paper twin of buffree max) | **Missing** | `sbatch scripts/simpl_runs_final_hybrid3_buf_max.sh` (6 jobs) |
| MN40/SN55 under **max** (buf + buffree fill) | **Missing** | `sbatch scripts/simpl_runs_max_pool_mn40_sn55_fill.sh` (18 jobs) |

### Recommended Leonardo submit order

```bash
# 1) Insert-ladder buffer twin of HYBRID_CLOSEFAR (config pooling; fair vs existing buffree)
sbatch scripts/simpl_runs_final_hybrid_closefar_buf.sh

# 2) Paper-aligned hybrid3 buf + max on all 6 protocols
sbatch scripts/simpl_runs_final_hybrid3_buf_max.sh

# 3) MN40/SN55 max fill (includes hybrid3/mean/hybrid4/pshot buf+buffree)
sbatch scripts/simpl_runs_max_pool_mn40_sn55_fill.sh
```

Hybrid3-only subset of (1):

```bash
sbatch --array=0,4,8,12,16,20 scripts/simpl_runs_final_hybrid_closefar_buf.sh
# indices: protocol*4 + 0  → hybrid3_st for each protocol
```

After logs land, refresh this file from:

```bash
aprun python scripts/summarize_fscil_final_campaign.py --campaign ep5_hybrid
# and inspect 00_FINAL_EP5_MAX_POOL/
```

---

## 7. One-paragraph settings (buffer ablation)

> Buffer ablation numbers use the same frozen Uni3D-B + base-only adapter (5 epochs) + ridge–prototype fusion (\(\alpha=0.85\)) + conflict replay (no test CM in FINAL campaigns) as the main recipe, but keep a **1-shot old-class exemplar buffer** (`fscil_preset=canonical`). In-domain numbers use close–far-style inserts via `hybrid4_evo` (no complete `hybrid3_st` buf run); cross-dataset uses mean inserts. MN40/SN55 buffer logs still use logsumexp pooling unless noted; CO3D and cross-dataset use max. Seed 2008.

---

## 8. Relation to buffer-free file

| Need | File |
|------|------|
| Main paper tables (buffree, max where possible) | `PAPER_EXPERIMENT.md` |
| Buffer ablation twin (this file) | `PAPER_EXPERIMENT_BUFFER.md` |
| Narrative / method prose | `PAPER.md` |
| Campaign inventory | `EXPERIMENTS.md` |

*End of PAPER_EXPERIMENT_BUFFER.md*
