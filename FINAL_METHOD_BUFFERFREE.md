# FINAL_METHOD_BUFFERFREE.md — **PRidE** buffer-free (ACCEPTED)

Canonical copy: **`_notes/pride_final.md`** (agent context points there).  
This file is a short mirror so the repo root also has the lock.

**Name in tables:** **PRidE** (not “Ours” / CacheEvo / SelfEvo in prose).

| | |
|--|--|
| Insert | close–far (`hybrid3_st`) on **all 6** |
| Pooling | **max** |
| Buffer | **free** |
| Adapter | base-only **ep10** |
| Fusion | ridge_cache **α=0.5** (flat, no ramp), static |
| Seed | 2008 |

### In-domain
`95.3,93.7,93.6,93.0,91.5,3.9,92.0,91.0,90.0,89.7,89.3,88.6,89.0,3.0,80.4,75.1,72.5,71.5,70.5,70.2,10.2`

### Cross (close–far)
`92.2,91.2,90.5,88.8,88.0,87.1,86.4,85.1,84.4,83.3,82.1,10.1,92.7,91.2,88.6,87.4,5.3,93.6,92.5,91.5,91.0,2.6`

**Re-run (all 6, max, α=0.5, ep10):** `sbatch scripts/simpl_runs_pride_alpha05_ep10.sh`  
Logs: `_experiments/00_ABLATION_ALPHA05_EP10/final10_*_hybrid3_st_buffree_a05.log`

**Buffer twin:** `FINAL_METHOD_BUFFER.md` → `sbatch scripts/simpl_runs_pride_alpha05_buf_ep10.sh`
