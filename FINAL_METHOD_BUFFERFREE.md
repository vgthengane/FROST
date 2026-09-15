# FINAL_METHOD_BUFFERFREE.md — **PRidE** buffer-free

Canonical copy: **`_notes/pride_final.md`** (agent context points there).  
This file is a short mirror so the repo root also has the lock.

**Name in tables:** **PRidE** (not “Ours” / CacheEvo / SelfEvo in prose).

| | |
|--|--|
| Insert | **mean + two close** (`mean_close_st` / `fscil_mean_best2_prototypes`) on **all 6** |
| Pooling | **max** |
| Buffer | **free** |
| Adapter | base-only **ep10** |
| Fusion | ridge_cache **α=0.5** (flat, no ramp), static |
| Seed | 2008 |
| Status | **PENDING** re-run (replaced close–far / `hybrid3_st`) |

### Curves
Fill after `00_ABLATION_ALPHA05_EP10/final10_*_mean_close_st_buffree_a05.log` complete.  
Do **not** reuse close–far paste rows as the main table.

**Re-run (all 6, max, α=0.5, ep10):** `sbatch scripts/simpl_runs_pride_alpha05_ep10.sh`  
Logs: `_experiments/00_ABLATION_ALPHA05_EP10/final10_*_mean_close_st_buffree_a05.log`

**Buffer twin:** `FINAL_METHOD_BUFFER.md` → `sbatch scripts/simpl_runs_pride_alpha05_buf_ep10.sh`
