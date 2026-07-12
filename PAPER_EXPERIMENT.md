# PAPER_EXPERIMENT.md — Final FSCIL results (max pooling)

**Last updated:** 2026-07-12  
**Method:** **PRidE** (Prototype-augmented Ridge for Exemplar-free FSCIL)  
**Seed:** 2008  
**Pooling decision:** **`prototype_pooling=max`** (hard max over per-class prototype similarities)  
**Adapter:** base-only, `tuned_epoch=5`  
**Classifier:** `ridge_cache`, α=0.85  

This file is the **paper numbers + log map** (buffer-free / main recipe). Prefer these tables over older logsumexp MN40/SN55 hybrid logs.

**Accepted buffer-free final method (locked card + logs):** `FINAL_METHOD_BUFFERFREE.md`  
**With-buffer twin:** see `PAPER_EXPERIMENT_BUFFER.md` / `scripts/simpl_runs_final_hybrid3_buf_max.sh`.

---

## 1. Pooling policy (locked)

| Setting | Value |
|---------|--------|
| Official for paper | **`max`** |
| Rejected as default | `logsumexp` (τ=0.07) — stronger on some protocols, but we standardize on max for clarity |

**Swap study** (`00_FINAL_EP5_HYBRID_POOLING_SWAP`, hybrid3 buffree only):

| Protocol | logsumexp → max ΔNovel | Prefer for raw score |
|----------|------------------------:|----------------------|
| MN40 | −1.18 | logsumexp |
| SN55 | −0.54 | logsumexp |
| CO3D | +1.97 (max→lse) | logsumexp |
| MN→ScanObj | −0.21 (max→lse) | max (tie) |
| SN→ScanObj | +0.86 (max→lse) | logsumexp |
| SN→CO3D | +0.17 (max→lse) | ~tie |

We still **report max** as the official method for a single clean inference rule. Optional appendix: logsumexp ablation.

---

## 2. Log directories (future lookup)

| Directory | What is there | Pooling |
|-----------|---------------|---------|
| `_experiments/00_FINAL_EP5_HYBRID_CLOSEFAR/` | Insert ablation buffree: nearmean / **hybrid3** / hybrid4 / fps5 | MN40/SN55 = **logsumexp**; CO3D+cross = **max** |
| `_experiments/00_FINAL_EP5_HYBRID_POOLING_SWAP/` | hybrid3 buffree pooling swap | MN40/SN55 = **max**; others = logsumexp (ablation) |
| `_experiments/00_FINAL_EP5_BASEONLY/` | Full matrix: mean / pshot* / hybrid4 × buf/buffree × 6 protocols | MN40/SN55 = **logsumexp**; CO3D+cross = **max** |
| `_experiments/00_FINAL_EP5_MAX_POOL/` | *(fill)* MN40/SN55 re-runs forced to **max** | **max** |
| `_experiments/00_SIMPL_REGIME_ABLATIONS/` | Ridge / cache / fusion / evolve story | logsumexp on MN40/SN55 configs |

**Scripts**

| Script | Role |
|--------|------|
| `scripts/simpl_runs_final_hybrid_closefar.sh` | Original hybrid insert campaign |
| `scripts/simpl_runs_hybrid_pooling_swap.sh` | Pooling swap (done) |
| `scripts/simpl_runs_final_ep5_baseonly.sh` | EP5 final matrix |
| `scripts/simpl_runs_max_pool_mn40_sn55_fill.sh` | **Submit to finish MN40/SN55 under max** |

```bash
sbatch scripts/simpl_runs_max_pool_mn40_sn55_fill.sh
# array 0-17: mean/nearmean/hybrid3/hybrid4/pshot × buf/buffree × MN40+SN55
```

---

## 3. Canonical max results — close–far ladder (buffree, static)

**Insert:** mean (1) → nearmean (2) → **close–far / hybrid3** (3). All **static**, **buffer-free**, **max** where available.

### 3.1 Confirmed **max** (use these)

| Protocol | mean_st | nearmean | **close–far (hybrid3)** | Source (hybrid3) |
|----------|--------:|---------:|------------------------:|------------------|
| **MN40** | *[need max fill]* | *[need max fill]* | **91.05 / 77.25 / 85.07** | `00_FINAL_EP5_HYBRID_POOLING_SWAP/final_mn40_hybrid3_st_buffree_max.log` |
| **SN55** | *[need max fill]* | *[need max fill]* | **88.87 / 77.20 / 83.67** | `.../final_sn55_hybrid3_st_buffree_max.log` |
| **CO3D** | 64.84 / 24.08 / 36.73 | 65.47 / 27.03 / 40.05 | **66.05 / 28.99 / 42.19** | `HYBRID_CLOSEFAR` + `BASEONLY` mean |
| **MN→ScanObj** | **87.16 / 69.47 / 79.45** | 86.91 / 68.63 / 78.87 | 87.01 / 69.05 / 79.15 | `BASEONLY` + `HYBRID_CLOSEFAR` |
| **SN→ScanObj** | 90.62 / 64.54 / 76.26 | 90.58 / 64.03 / 75.90 | 90.56 / 64.03 / 75.90 | same |
| **SN→CO3D** | **82.16 / 45.55 / 60.86** | 81.81 / 43.76 / 59.25 | 81.83 / 43.88 / 59.36 | same |

Cells are **Last / Novel@Last / Harm@Last**.

**Readout (max, buffree):**

- **In-domain CO3D:** mean → nearmean → close–far improves Novel/Harm.  
- **Cross-dataset:** **mean** often best (MN→ScanObj, SN→CO3D).  
- **MN40/SN55 close–far (max):** 91.05 / 88.87 Last — slightly below old logsumexp hybrid3 (91.21 / 88.90).

### 3.2 Log paths — buffree insert ladder (max)

| Protocol | mean_st | nearmean | hybrid3 close–far |
|----------|---------|----------|-------------------|
| MN40 | `00_FINAL_EP5_MAX_POOL/final_mn40_mean_st_buffree_max.log` *(pending)* | `.../final_mn40_hybrid_nearmean_st_buffree_max.log` *(pending)* | `00_FINAL_EP5_HYBRID_POOLING_SWAP/final_mn40_hybrid3_st_buffree_max.log` **done** |
| SN55 | `.../final_sn55_mean_st_buffree_max.log` *(pending)* | `.../nearmean...` *(pending)* | `.../final_sn55_hybrid3_st_buffree_max.log` **done** |
| CO3D | `00_FINAL_EP5_BASEONLY/final_co3d_mean_st_buffree.log` | `00_FINAL_EP5_HYBRID_CLOSEFAR/final_co3d_hybrid_nearmean_st_buffree.log` | `.../final_co3d_hybrid3_st_buffree.log` |
| MN→ScanObj | `BASEONLY/final_mn_scanobj_mean_st_buffree.log` | `HYBRID_CLOSEFAR/final_mn_scanobj_hybrid_nearmean_st_buffree.log` | `.../final_mn_scanobj_hybrid3_st_buffree.log` |
| SN→ScanObj | `BASEONLY/final_sn_scanobj_mean_st_buffree.log` | `HYBRID_CLOSEFAR/...nearmean...` | `HYBRID_CLOSEFAR/...hybrid3...` |
| SN→CO3D | `BASEONLY/final_sn_co3d_mean_st_buffree.log` | `HYBRID_CLOSEFAR/...nearmean...` | `HYBRID_CLOSEFAR/...hybrid3...` |

---

## 4. Close–far static vs evolve (buffree, max where possible)

| Protocol | hybrid3_st (static) | hybrid4_evo | ΔLast (evo−st) | Notes |
|----------|--------------------:|------------:|---------------:|-------|
| MN40 | 91.05 (max swap) | *[need max]*; old lse 91.21 | — | static≈evo under lse historically |
| SN55 | 88.87 (max swap) | *[need max]*; old lse 88.85 | — | |
| CO3D | 66.05 | 66.05 (`HYBRID_CLOSEFAR`) | 0.00 | identical |
| MN→ScanObj | 87.01 | 87.01 | 0.00 | |
| SN→ScanObj | 90.56 | 90.56 | 0.00 | |
| SN→CO3D | 81.83 | 81.83 | 0.00 | |

**Paper claim:** with close–far insert, **evolution is unnecessary** (static = main).

Prefer `HYBRID_CLOSEFAR` hybrid4 buffree for CO3D/cross (max). MN40/SN55 hybrid4 max → fill script.

---

## 5. Buffer vs buffer-free (max protocols)

CO3D + cross-dataset from `00_FINAL_EP5_BASEONLY` (max default). Cells = **Last**.

### mean_st

| Protocol | buf | buffree | Δ (bf−buf) |
|----------|----:|--------:|-----------:|
| CO3D | 65.65 | 64.84 | −0.81 |
| MN→ScanObj | 87.16 | 87.16 | 0.00 |
| SN→ScanObj | 90.74 | 90.62 | −0.12 |
| SN→CO3D | 82.67 | 82.16 | −0.51 |

### hybrid4_evo

| Protocol | buf | buffree | Δ |
|----------|----:|--------:|--:|
| CO3D | 66.11 | 65.65 (`BASEONLY`) / 66.05 (`HYBRID`) | ~0 |
| MN→ScanObj | 86.91 | 86.86 / 87.01 | ~0 |
| SN→ScanObj | 90.85 | 90.73 / 90.56 | small |
| SN→CO3D | 82.04 | 81.70 / 81.83 | small |

### pshot_st (static, 5 keys)

| Protocol | buf | buffree |
|----------|----:|--------:|
| CO3D | 66.11 | 66.22 |
| MN→ScanObj | 86.76 | 86.61 |
| SN→ScanObj | 90.79 | 90.68 |
| SN→CO3D | 82.37 | 81.77 |

**Paper claim:** exemplar disk buffer is **optional**; buffer-free is competitive under max + conflict replay.

MN40/SN55 buf vs buffree under **max** → pending `00_FINAL_EP5_MAX_POOL/`.

---

## 6. Main method rows to report (max, proposed)

Until MN40/SN55 fill completes, **main close–far buffree max** row:

| Protocol | Setting | Last | Novel | Harm | Log |
|----------|---------|-----:|------:|-----:|-----|
| MN40 | hybrid3_st buffree max | 91.05 | 77.25 | 85.07 | `HYBRID_POOLING_SWAP/final_mn40_hybrid3_st_buffree_max.log` |
| SN55 | hybrid3_st buffree max | 88.87 | 77.20 | 83.67 | `.../final_sn55_hybrid3_st_buffree_max.log` |
| CO3D | hybrid3_st buffree max | 66.05 | 28.99 | 42.19 | `HYBRID_CLOSEFAR/final_co3d_hybrid3_st_buffree.log` |
| MN→ScanObj | **mean_st** buffree max | 87.16 | 69.47 | 79.45 | `BASEONLY/final_mn_scanobj_mean_st_buffree.log` |
| SN→ScanObj | mean_st or hybrid3 | 90.62 / 90.56 | 64.54 / 64.03 | 76.26 / 75.90 | `BASEONLY` / `HYBRID_CLOSEFAR` |
| SN→CO3D | **mean_st** buffree max | 82.16 | 45.55 | 60.86 | `BASEONLY/final_sn_co3d_mean_st_buffree.log` |

**Domain-aware recipe (paper):**

- In-domain → **close–far (hybrid3), static, buffree, max**  
- Cross-dataset → **mean, static, buffree, max**

---

## 7. Do **not** mix into max-final tables

These are **logsumexp** (keep as ablation only):

| Location | Examples |
|----------|----------|
| `HYBRID_CLOSEFAR/final_mn40_*.log`, `final_sn55_*.log` | hybrid3/nearmean/fps5 under lse |
| `BASEONLY/final_mn40_*.log`, `final_sn55_*.log` | mean/pshot/hybrid4 under lse |
| `HYBRID_POOLING_SWAP/*_logsumexp.log` | CO3D/cross forced to lse |

---

## 8. Gaps checklist (max)

| Item | Status |
|------|--------|
| hybrid3 buffree max — all 6 protocols | **Done** (MN40/SN55 via swap; others via HYBRID_CLOSEFAR) |
| mean / nearmean / hybrid4 / pshot — CO3D+cross max | **Done** (`BASEONLY` / `HYBRID_CLOSEFAR`) |
| mean / nearmean / hybrid3 buf / hybrid4 / pshot — **MN40+SN55 max** | **Pending** → `sbatch scripts/simpl_runs_max_pool_mn40_sn55_fill.sh` |
| hybrid3 **buf** max — CO3D+cross | Optional (not required for buffree main) |

After fill lands, re-run:

```bash
aprun python scripts/summarize_fscil_final_campaign.py \
  --log-root _experiments/00_FINAL_EP5_MAX_POOL
```

and refresh §3–§6 of this file.

---

## 9. Regime fusion story (unchanged reference)

Still cite `_experiments/00_SIMPL_REGIME_ABLATIONS/` for ridge vs cache vs fusion (separate from insert/pooling).  
Those MN40/SN55 logs use logsumexp; fusion conclusions are about classifier mode, not prototype pooling.

---

## 10. One-paragraph method settings (copy for Experiments §)

> Unless noted, all PRidE FSCIL numbers use frozen Uni3D-B, base-only adapter warmup (5 epochs), ridge–prototype fusion (\(\alpha=0.85\)), conflict replay without test CM, buffer-free evaluation by default, **static** prototype memory, and **max** pooling over per-class prototype cosine similarities. In-domain novel classes use close–far inserts (3 keys); cross-dataset novel classes use a single mean key. Seed 2008.

---

*End of PAPER_EXPERIMENT.md*
