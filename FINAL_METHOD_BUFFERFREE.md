# FINAL_METHOD_BUFFERFREE.md — **PRidE** buffer-free (ACCEPTED)

Canonical copy: **`_notes/pride_final.md`** (agent context points there).  
This file is a short mirror so the repo root also has the lock.

**Name in tables:** **PRidE** (not “Ours” / CacheEvo / SelfEvo in prose).

| | |
|--|--|
| Insert | close–far (`hybrid3_st`) on **all 6** |
| Pooling | **max** |
| Buffer | **free** |
| Adapter | base-only ep5 |
| Fusion | ridge_cache α=0.85, static |
| Seed | 2008 |

### In-domain
`95.5,93.7,93.5,92.9,91.0,4.4,91.8,90.9,89.7,89.5,89.1,88.5,88.9,3.0,76.8,72.1,68.3,68.0,66.2,66.0,10.7`

### Cross (close–far)
`91.7,90.9,90.1,88.9,88.1,87.0,85.9,85.0,84.4,83.2,81.8,9.9,92.2,91.3,88.2,87.0,5.2,93.3,92.4,91.3,90.6,2.7`

**Re-run (all 6, max):** `sbatch scripts/simpl_runs_final_hybrid3_buffree_max.sh`

**Buffer twin:** `FINAL_METHOD_BUFFER.md` → `sbatch scripts/simpl_runs_final_hybrid3_buf_max.sh`
