# PRidE — locked final FSCIL results (do not re-negotiate in chat)

**Paper / table name:** **PRidE**  
**Code:** `selfevo` / `ridge_cache` / `hybrid3_st`  
**Seed:** 2008  
**Locked:** 2026-07-12  

> If chat memory conflicts with this file, **this file wins**.

| Variant | Status | Doc | Launcher |
|---------|--------|-----|----------|
| **PRidE (buffer-free)** | **ACCEPTED — use in tables** | this file §2 | `sbatch scripts/simpl_runs_final_hybrid3_buffree_max.sh` |
| **PRidE (buffer)** | **TO RUN** | §3 | `sbatch scripts/simpl_runs_final_hybrid3_buf_max.sh` |

Also mirrored: `FINAL_METHOD_BUFFERFREE.md`, `FINAL_METHOD_BUFFER.md`.

---

## 1. Locked recipe (both variants share this)

| Knob | Value |
|------|--------|
| Insert | **mean–close–far** (`hybrid3_st`) on **ALL 6** protocols (in-domain **and** cross) |
| Keys | 3 / class, **`evolution_policy=static`** |
| Pooling | **`prototype_pooling=max`** |
| Classifier | **`ridge_cache`**, **α=0.85** |
| Adapter | **base-only**, `tuned_epoch=5` |
| Conflict replay | **on**, `conflict_replay_on_test=false`, `conflict_replay_use_test_cm=false` |
| Backbone | frozen Uni3D-B |
| Seed | **2008** |

**Only difference between variants:**

| | Buffer-free (accepted) | Buffer (to run) |
|--|------------------------|-----------------|
| `model_name` | `selfevo_bufferfree` | `selfevo` |
| `fscil_preset` | (buffer-free) | `canonical` |

### DO NOT confuse with

| Wrong | Why |
|-------|-----|
| **mean on cross** / “domain-aware insert” as main | Ablation story only — **not** locked PRidE main |
| logsumexp MN40/SN55 (`HYBRID_CLOSEFAR` mn40/sn55, `BASEONLY` mn40/sn55) | Not max |
| `hybrid4_evo`, `mean_st`, `pshot_*` | Ablations |
| `00_BUF_CAMPAIGN_EP5*` | Legacy test-aware CM |
| `simpl_runs_final_method_buf.sh` | Superseded (had mean on cross) |

---

## 2. PRidE buffer-free — ACCEPTED session curves

### In-domain

| Protocol | Curve | Δ | Log |
|----------|-------|--:|-----|
| MN40 | `95.5,93.7,93.5,92.9,91.0` | 4.4 | `00_FINAL_EP5_HYBRID_POOLING_SWAP/final_mn40_hybrid3_st_buffree_max.log` |
| SN55 | `91.8,90.9,89.7,89.5,89.1,88.5,88.9` | 3.0 | `.../final_sn55_hybrid3_st_buffree_max.log` |
| CO3D | `76.8,72.1,68.3,68.0,66.2,66.0` | 10.7 | `00_FINAL_EP5_HYBRID_CLOSEFAR/final_co3d_hybrid3_st_buffree.log` |

Paste (in-domain + Δs):
```
95.5,93.7,93.5,92.9,91.0,4.4,91.8,90.9,89.7,89.5,89.1,88.5,88.9,3.0,76.8,72.1,68.3,68.0,66.2,66.0,10.7
```

### Cross-domain (close–far — not mean)

| Protocol | Curve | Δ | Log |
|----------|-------|--:|-----|
| SN→CO3D | `91.7,90.9,90.1,88.9,88.1,87.0,85.9,85.0,84.4,83.2,81.8` | 9.9 | `00_FINAL_EP5_HYBRID_CLOSEFAR/final_sn_co3d_hybrid3_st_buffree.log` |
| MN→ScanObj | `92.2,91.3,88.2,87.0` | 5.2 | `.../final_mn_scanobj_hybrid3_st_buffree.log` |
| SN→ScanObj | `93.3,92.4,91.3,90.6` | 2.7 | `.../final_sn_scanobj_hybrid3_st_buffree.log` |

Paste (cross + Δs):
```
91.7,90.9,90.1,88.9,88.1,87.0,85.9,85.0,84.4,83.2,81.8,9.9,92.2,91.3,88.2,87.0,5.2,93.3,92.4,91.3,90.6,2.7
```

Settings dump twin: `_experiments/slurm_logs/49278234_1_hybrid_pool.out` (SN55 hybrid3 buffree max).

---

## 3. PRidE buffer — TO RUN (identical except buffer)

```bash
sbatch scripts/simpl_runs_final_hybrid3_buf_max.sh
```

6 jobs → `_experiments/00_FINAL_EP5_MAX_POOL/final_<proto>_hybrid3_st_buf_max.log`

| Array | Protocol |
|------:|----------|
| 0–2 | MN40, SN55, CO3D |
| 3–5 | MN→ScanObj, SN→ScanObj, SN→CO3D |

After done: fill §3 curves here and in `FINAL_METHOD_BUFFER.md`.

---

## 4. One-liner (paper)

> **PRidE:** frozen Uni3D-B, base-only adapter (5 epochs), ridge–cache fusion (α=0.85), static mean–close–far prototypes (3 keys) on all protocols, max pooling, conflict replay without test CM. Main tables use buffer-free; buffer is the matched twin ablation.

---

*End of `_notes/pride_final.md`*
