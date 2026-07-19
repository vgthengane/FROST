# PRidE — locked final FSCIL results (do not re-negotiate in chat)

**Paper / table name:** **PRidE**  
**Code:** `selfevo` / `ridge_cache` / `hybrid3_st`  
**Seed:** 2008  
**Updated:** 2026-07-19  

> If chat memory conflicts with this file, **this file wins**.

| Variant | Status | Logs |
|---------|--------|------|
| **PRidE EP10 α=0.5 buffer-free** | **ACCEPTED main** | `00_ABLATION_ALPHA05_EP10/final10_*_hybrid3_st_buffree_a05.log` |
| **PRidE EP10 α=0.5 buffer** | **DONE twin** | `00_ABLATION_ALPHA05_BUF_EP10/final10_*_hybrid3_st_buf_a05.log` |

Recipe (both): close–far on **all 6**, max, ridge_cache **α=0.5** (flat; no ramp), static, base-only, **`tuned_epoch=10`**, conflict train-CM only, seed 2008.  
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
| α=0.85 EP5 / EP10 (`00_FINAL_EP*` / `…_hybrid3_buffree_max*`) | Previous candidate; **not** current main |
| mean on cross / domain-aware insert | Ablation only |
| `hybrid4_evo`, `mean_st`, `pshot_*`, α sweep other than 0.5 | Ablations |

---

## EP10 α=0.5 buffer-free (main table)

**In-domain:** `95.3,93.7,93.6,93.0,91.5,3.9,92.0,91.0,90.0,89.7,89.3,88.6,89.0,3.0,80.4,75.1,72.5,71.5,70.5,70.2,10.2`  
**Cross:** `92.2,91.2,90.5,88.8,88.0,87.1,86.4,85.1,84.4,83.3,82.1,10.1,92.7,91.2,88.6,87.4,5.3,93.6,92.5,91.5,91.0,2.6`

| Protocol | Curve | Δ | Last/Novel/Harm |
|----------|-------|--:|----------------:|
| MN40 | 95.3 93.7 93.6 93.0 91.5 | 3.9 | 91.45 / 79.02 / 86.15 |
| SN55 | 92.0 91.0 90.0 89.7 89.3 88.6 89.0 | 3.0 | 89.03 / 77.30 / 83.80 |
| CO3D | 80.4 75.1 72.5 71.5 70.5 70.2 | 10.2 | 70.21 / 37.35 / 50.99 |
| SN→CO3D | 92.2 … 82.1 | 10.1 | 82.10 / 43.42 / 59.03 |
| MN→Scan | 92.7 91.2 88.6 87.4 | 5.3 | 87.37 / 70.74 / 80.23 |
| SN→Scan | 93.6 92.5 91.5 91.0 | 2.6 | 90.99 / 67.64 / 78.42 |

---

## EP10 α=0.5 buffer

**In-domain:** `95.3,93.7,93.8,93.2,91.4,3.9,92.1,91.0,89.9,89.7,89.4,88.6,89.0,3.0,80.4,74.4,72.5,71.0,69.5,69.4,11.0`  
**Cross:** `92.2,91.1,90.5,88.9,88.5,87.5,86.4,85.3,84.6,83.2,81.9,10.3,92.7,91.2,88.4,87.2,5.5,93.6,92.6,91.6,91.1,2.5`

| Protocol | Curve | Δ | Last/Novel/Harm |
|----------|-------|--:|----------------:|
| MN40 | 95.3 93.7 93.8 93.2 91.4 | 3.9 | 91.41 / 78.63 / 85.94 |
| SN55 | 92.1 91.0 89.9 89.7 89.4 88.6 89.0 | 3.0 | 89.04 / 77.69 / 84.00 |
| CO3D | 80.4 74.4 72.5 71.0 69.5 69.4 | 11.0 | 69.40 / 39.07 / 52.22 |
| SN→CO3D | 92.2 … 81.9 | 10.3 | 81.86 / 43.24 / 58.81 |
| MN→Scan | 92.7 91.2 88.4 87.2 | 5.5 | 87.21 / 71.79 / 80.69 |
| SN→Scan | 93.6 92.6 91.6 91.1 | 2.5 | 91.08 / 66.61 / 77.79 |

---

## Quick readout

- Main paper row: **EP10 α=0.5 buffer-free**.
- Buffer twin ≈ buffer-free on most protocols; optional for the table.
- Prior α=0.85 EP5/EP10 campaigns kept as historical / ablation context only.

*End of `_notes/pride_final.md`*
