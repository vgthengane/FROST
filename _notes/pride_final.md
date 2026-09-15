# PRidE — locked final FSCIL results (do not re-negotiate in chat)

**Paper / table name:** **PRidE**  
**Code:** `selfevo` / `ridge_cache` / `mean_close_st` (`fscil_mean_best2_prototypes`)  
**Seed:** 2008  
**Updated:** 2026-07-23  

> If chat memory conflicts with this file, **this file wins**.

| Variant | Status | Logs |
|---------|--------|------|
| **PRidE EP10 α=0.5 buffer-free** | **PENDING re-run** (mean + two close) | `00_ABLATION_ALPHA05_EP10/final10_*_mean_close_st_buffree_a05.log` |
| **PRidE EP10 α=0.5 buffer** | **PENDING re-run** (twin) | `00_ABLATION_ALPHA05_BUF_EP10/final10_*_mean_close_st_buf_a05.log` |

Recipe (both): **mean + two close** on **all 6**, max, ridge_cache **α=0.5** (flat; no ramp), static, base-only, **`tuned_epoch=10`**, conflict train-CM only, seed 2008.  
Only knob that changes: **buffer vs buffer-free**.

Launchers:
```bash
sbatch scripts/simpl_runs_pride_alpha05_ep10.sh       # buffer-free (main)
sbatch scripts/simpl_runs_pride_alpha05_buf_ep10.sh   # buffer twin
```

Also mirrored: `FINAL_METHOD_BUFFERFREE.md`, `FINAL_METHOD_BUFFER.md`.

### DO NOT confuse with
| Wrong | Why |
|-------|-----|
| close–far / `hybrid3_st` α=0.5 EP10 | Previous lock (2026-07-19); keep logs for insert ablation only |
| α=0.85 EP5 / EP10 (`00_FINAL_EP*` / `…_hybrid3_buffree_max*`) | Older candidate; **not** current main |
| mean on cross / domain-aware insert | Ablation only |
| `hybrid4_evo`, `mean_st`, `pshot_*`, α sweep other than 0.5 | Ablations |

---

## EP10 α=0.5 buffer-free (main table)

**Status:** awaiting `mean_close_st` campaign completion.  
**Do not paste close–far curves below as the main PRidE row.**

Previous close–far reference (ablation archive only):  
**In-domain:** `95.3,93.7,93.6,93.0,91.5,3.9,92.0,91.0,90.0,89.7,89.3,88.6,89.0,3.0,80.4,75.1,72.5,71.5,70.5,70.2,10.2`  
**Cross:** `92.2,91.2,90.5,88.8,88.0,87.1,86.4,85.1,84.4,83.3,82.1,10.1,92.7,91.2,88.6,87.4,5.3,93.6,92.5,91.5,91.0,2.6`

| Protocol | Curve | Δ | Last/Novel/Harm |
|----------|-------|--:|----------------:|
| MN40 | *TBD* | | |
| SN55 | *TBD* | | |
| CO3D | *TBD* | | |
| SN→CO3D | *TBD* | | |
| MN→Scan | *TBD* | | |
| SN→Scan | *TBD* | | |

---

## EP10 α=0.5 buffer

**Status:** awaiting twin re-run with `mean_close_st`.

---

## Quick readout

- Main paper row: **EP10 α=0.5 buffer-free**, insert = **mean + two nearest shots**.
- Insert rationale: local support around the class mean (cleaner than close–far; MN40≈flat, CO3D slightly prefers two-close over close–far in hparam ablation).
- After logs land: fill curves above and refresh `FINAL_METHOD_*.md`.

*End of `_notes/pride_final.md`*
