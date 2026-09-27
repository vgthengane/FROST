# PRidE — locked final FSCIL results (do not re-negotiate in chat)

**Paper / table name:** **PRidE**  
**Code:** `selfevo` / `ridge_cache`  
**Seed set:** `1947, 1991, 2001, 2008, 2019`  
**Updated:** 2026-09-25  

> If chat memory conflicts with this file, **this file wins**.

## PAPER METRICS (LOCKED)

| Symbol | Meaning | Source in logs |
|--------|---------|----------------|
| \(\mathcal{A}_b\) | Base / session-0 total accuracy | first value of `CNN top1 curve` |
| \(\mathcal{A}_T\) | Final-session total accuracy | last value of `CNN top1 curve` |
| \(\bar{\mathcal{A}}\) / **AA** | **Average incremental accuracy** = mean of per-session total acc from base through last | `Average Accuracy (CNN)` = mean(`CNN top1 curve`) |
| \(\Delta\) | Relative drop \(100(\mathcal{A}_b-\mathcal{A}_T)/\mathcal{A}_b\) | from displayed Ab, AT |

**Never** set \(\bar{\mathcal{A}} = (\mathcal{A}_b+\mathcal{A}_T)/2\). That is wrong for this paper.

---

## PAPER RECIPE (LOCKED)

**Both point-cloud and image main tables use `ridge_rp_dim=2000` + mean-only.**

| Domain | Insert | RP | Fusion / other | Logs / tables |
|--------|--------|-----|----------------|---------------|
| **Point cloud (3D)** | **mean only** (`mean_st`, 1 key/class) | **2000** | α=0.5 flat, max, static, base-only ep10 | `00_FINAL_SUB_MEAN_A05_RP2000_*` → `full_single.tex`, `full_xdata.tex` |
| **Image (2D)** | **mean only** (`mean1`) | **2000** | α=0.5 flat, static, base-only | `00_PRIDE_IMAGE_MEANONLY_A05_RP2000_5SEEDS` → `image_fscil.tex` |

| PC variant | Logs |
|------------|------|
| **buffer-free (EF)** | `00_FINAL_SUB_MEAN_A05_RP2000_EP10*` |
| **buffer (EB)** | `00_FINAL_SUB_MEAN_A05_RP2000_BUF_EP10*` |

`aprun python scripts/summarize_pride_multiseed.py --variant final_sub_both`

**rp=1000** = ablation twin only (weaker CO3D).

### Do not mix image folders
| Folder | Insert | Use for paper? |
|--------|--------|----------------|
| `00_PRIDE_IMAGE_MEANONLY_A05_RP2000_5SEEDS` | **mean only** | **Yes** (matches `image_fscil.tex`) |
| `00_PRIDE_IMAGE_LOCKED_A05_RP2000_5SEEDS` | mean_best2 | No — insert ablation only |

### DO NOT confuse with
| Wrong | Why |
|-------|-----|
| \(\bar{\mathcal{A}}=(\mathcal{A}_b+\mathcal{A}_T)/2\) | Not our AA; use mean of all session accs |
| mean+2close / `mean_close_st` | Insert ablation only |
| mean_best2 (image) as mean-only | Different insert |
| close–far / `hybrid3_st` | Older lock |
| **rp=1000 as main** | Ablation only |
| α=0.85 EP5/EP10 | Older candidate |

---

## In-domain main table (PC mean_st, rp=2000, 5-seed)

Metrics: \(\mathcal{A}_b\), \(\mathcal{A}_T\), \(\bar{\mathcal{A}}\)=**avg incremental acc**, \(\Delta\) relative (see above).

| Variant | ModelNet | ShapeNet | CO3D |
|---------|----------|----------|------|
| **Ours (EF)** | 95.8, 91.3, 93.4, 4.7 | 92.2, 89.2, 90.1, 3.3 | 81.6, **72.0**, 75.3, **11.8** |
| **Ours (EB)** | 95.8, **91.5**, 93.5, **4.5** | 92.2, **89.3**, 90.2, **3.1** | 81.6, 71.9, 75.3, 11.9 |

## Cross-domain main table (PC mean_st, rp=2000, 5-seed)

Column order matches `full_xdata.tex`: SN→CO3D | MN→Scan | SN→Scan.

| Variant | SN→CO3D | MN→Scan | SN→Scan |
|---------|---------|---------|---------|
| **Ours (EF)** | 92.3, 82.5, 87.6, 10.6 | 92.8, **87.9**, 90.2, **5.3** | 93.4, 90.7, 91.9, 2.9 |
| **Ours (EB)** | 92.3, **82.9**, 87.8, **10.2** | 92.8, 87.9, 90.2, 5.3 | 93.4, **90.8**, 92.0, **2.8** |

vs FLEG Last (in-domain): MN40 87.3→**91.5**, SN55 83.0→**89.3**, CO3D 57.3→**72.0**.

---

## Quick readout

- Paper PC: **mean-only + rp=2000 + α=0.5 EP10**, EF + EB.
- Paper image: **mean-only + rp=2000** (`00_PRIDE_IMAGE_MEANONLY_A05_RP2000_5SEEDS`).
- \(\bar{\mathcal{A}}\) = average incremental accuracy (all sessions), never mid-point of Ab and AT.
- Why rp=2000: clearly better than rp=1000 on CO3D (~+2.4 Last); flat/slightly up elsewhere.

*End of `_notes/pride_final.md`*
