# Experiments

**Last updated:** 2026-07-11  
**Primary seed:** `2008` (single-seed unless noted)  
**Backbone:** frozen Uni3D-b  
**Paper method:** `selfevo`, `classifier_mode=ridge_cache`  
**Final campaign:** `00_FINAL_EP5_BASEONLY` (48 runs — see below)

Single source of truth for all FSCIL ablation campaigns under `_experiments/`.
Protocol details: `_notes/fscil_canonical.md`. Modular config presets:
`models/fscil_ablation_config.py`.

---

## Contents

1. [How to read metrics](#how-to-read-metrics)
2. [Final campaign (paper table)](#final-campaign-paper-table)
3. [Terminology: buffer vs buffer-free](#terminology-buffer-vs-buffer-free)
4. [Campaign index](#campaign-index)
5. [Paper story — regime matrix](#paper-story--regime-matrix)
6. [MN40 module & hyperparameter ablations](#mn40-module--hyperparameter-ablations)
7. [Adapter ablations](#adapter-ablations)
8. [Cross-dataset & CO3D runs](#cross-dataset--co3d-runs)
9. [Buffer & prototype ablations (EP5, base-only)](#buffer--prototype-ablations-ep5-base-only)
10. [Other ablations](#other-ablations)
11. [Per-session curves](#per-session-curves)
12. [Cross-cutting conclusions](#cross-cutting-conclusions)
13. [Open gaps & next steps](#open-gaps--next-steps)
14. [Log paths & summarize commands](#log-paths--summarize-commands)

---

## How to read metrics

| Metric | Meaning |
|--------|---------|
| **AA** | Average Accuracy — mean of per-session total top-1 |
| **Last** | Final-session total top-1 (paper “Last Acc”) |
| **Base@Last** | Old-class retention at final session |
| **Novel@Last** | Few-shot novel-class accuracy at final session |
| **Harm@Last** | Harmonic mean of base and novel at final session |
| **Δ** | Forgetting = session-0 total − Last |

Focus on **Novel@Last** and **Harm@Last** for plasticity; total Last can be
dominated by base-class count.

---

## Final campaign (paper table)

**Directory:** `_experiments/00_FINAL_EP5_BASEONLY/`  
**Script:** `scripts/simpl_runs_final_ep5_baseonly.sh`  
**Summarize:** `aprun python scripts/summarize_fscil_final_campaign.py`  
**Status:** Not yet run (0/60)

### Matrix (6 × 2 × 5 = 60 runs)

| Axis | Values |
|------|--------|
| **Protocols** | MN40, SN55, CO3D in-domain; MN40→ScanObj, SN55→ScanObj, SN55→CO3D |
| **Buffer** | `buf` (canonical, 1 old-shot disk re-read) · `buffree` (`selfevo_bufferfree`) |
| **Prototype regimes** | `mean_st` · `pshot_evo` · `pshot_evo5` · `pshot_st` · `hybrid4_evo` |

### Fixed settings (all 60 runs)

| Setting | Value | Notes |
|---------|-------|-------|
| Adapter | Base-only, task-0 only | `fscil_adapter_base_only=true`, `tuned_epoch=5` |
| Increment adapter | Off | `increment_adapter_warmup=false`, `fscil_task1_adapter_warmup=false` |
| Classifier | Ridge + cache fusion | `classifier_mode=ridge_cache`, `ridge_cache_alpha=0.85` |
| Conflict replay | On, **train-side CM only** | `conflict_replay=true`, **`conflict_replay_on_test=false`** |
| Old-class proxies | Cache keys | `old_data_source=cache` (buffer-free uses cache, not disk) |

### Five prototype regimes

| Suffix | KeyCache insert | Evolution | `max_prototypes` | Role |
|--------|-----------------|-----------|------------------|------|
| `mean_st` | `insert_strategy=mean` | `static` | 1 | Minimal memory; frozen cache |
| `pshot_evo` | per-shot (5) | `rules` | **3 base / 8 incr** | Main method (default caps) |
| `pshot_evo5` | per-shot (5) | `rules` | **5 base / 5 incr** | Fixed 5 keys/class cap |
| `pshot_st` | per-shot | `static` | 5 | Ablate evolution with full shots |
| `hybrid4_evo` | `mean_best2` (mean + 2 shots closest to mean) | `rules` | 3 base / **4** incr | 3 fixed + 1 slot for evolution |

**Hybrid insert (`mean_best2`):** class mean plus the two few-shot embeddings with highest
cosine similarity to that mean (most representative shots). Task 0 still uses a single
mean per base class; hybrid applies from task 1 onward.

### Method naming

```text
final_<protocol>_<regime>_<buf|buffree>_seed=2008.log

Examples:
  final_mn40_pshot_evo_buf_seed=2008.log
  final_co3d_mean_st_buffree_seed=2008.log
  final_sn_co3d_hybrid4_evo_buf_seed=2008.log
```

### Array layout

```text
idx = protocol_idx * 10 + buffer_idx * 5 + proto_idx
  protocol_idx ∈ [0..5], buffer_idx ∈ {0=buf, 1=buffree}, proto_idx ∈ [0..4]
```

### Per-class cache counts (prototypes per class)

New runs log after each task::

```text
KeyCache per-class keys task=4 n_classes=40 min=1 max=8 mean=4.62 total=185 histogram=1x20,5x4,8x16
KeyCache per-class keys task=4 cls_keys=0:1,1:1,2:1,...,39:8
```

- **`cls_keys`**: for **each class**, how many **keys** (prototypes) it has — `class_id:key_count`
- **`histogram`**: how many **classes** have each key count — `keys_per_class x num_classes`
  (e.g. `1x20` = 20 classes with exactly 1 key)

Parse with::

```bash
aprun python scripts/analyze_cache_per_class.py --show-classes \
  _experiments/00_FINAL_EP5_BASEONLY/final_mn40_pshot_evo_buf_seed=2008.log
```

**Legacy EP5 proxy** (`00_BUF_CAMPAIGN_EP5`, per-shot + rules, test-aware replay —
not final numbers). Total cache size only; per-class min/max inferred from caps +
insert/evolve behaviour:

#### Per-shot + evolve (`pshot_evo` proxy) — MN40 in-domain, last task

| Setting | Buffer | Total keys | Classes | Inferred min | Inferred max | Inferred last-task spread |
|---------|--------|----------:|--------:|-------------:|-------------:|---------------------------|
| P3/8 incr | buf | **185** | 40 | **1** (base) | **8** (incr cap) | base ≈1, novel ≈5–8 after merge/split |
| P3/8 incr | buffree | **120** | 40 | **1** | **5** | base 1, novel **exactly 5** (no evo growth) |

How to read this:

- **Task 0 (all regimes):** 1 key per base class (mean insert on full base data).
- **Per-shot incr insert:** up to 5 raw shot keys per new class.
- **Evolution (`rules`):** can add keys up to incr cap (8 for `pshot_evo`, 5 for
  `pshot_evo5`), then merge/evict; base classes usually stay at 1.
- **Buffer run with merges:** novel classes often grow toward cap 8 (MN40 buf:
  185−20 base = 165 over 20 novel → avg **8.25** → many classes at **8**).
- **Buffer-free:** little/no split growth; novel stays at **5** (120−20=100 → 5×20).

#### Hybrid (`hybrid4_evo`) — not run yet; expected at last task (MN40)

| | Expected min | Expected max | Expected typical |
|---|-------------:|-------------:|------------------|
| Base classes (20) | 1 | 1–2 | 1 (split on cache proxy rarely sticks) |
| Novel classes (20) | 3 | **4** | 3 insert + up to 1 evolution slot |
| **Global** | **1** | **4** | ~3.5 mean on novel, 1 on base |

`pshot_evo5` expected: min **1**, max **5** everywhere (both base cap and incr cap = 5).

### Submit

```bash
sbatch scripts/simpl_runs_final_ep5_baseonly.sh
# or local:
bash scripts/simpl_runs_final_ep5_baseonly.sh --local --skip-done
```

### Relation to earlier campaigns

| Final regime | Supersedes / differs from |
|--------------|---------------------------|
| `pshot_evo` | `00_BUF_CAMPAIGN_EP5` — same proto/evolve, but **no test CM** for replay |
| `mean_st` | `00_BUF_CAMPAIGN_EP5_MEAN` — adds `evolution_policy=static` + no test CM |
| `pshot_evo5` | **New** — 5/5 per-class cap |
| `pshot_st`, `hybrid4_evo` | **New** | used `conflict_replay_on_test=true`; **do not mix** with final
numbers without relabelling.

### Log verification (final runs)

| Signal | Expected |
|--------|----------|
| `Conflict replay task=…` | present |
| `Test-aware conflict replay task=…` | **absent** |
| `FSCIL old-shot replay: task=…` | present (buf) / absent (buffree) |

---

## Terminology: buffer vs buffer-free

Three mechanisms are often confused — they are **independent** (since Jul 10
refactor, commit `227c5d5`):

| Mechanism | What it does | Canonical | Correct buffer-free |
|-----------|--------------|-----------|---------------------|
| **Exemplar buffer** | Re-read raw old-class train point clouds from disk | 1 shot/class | **0** (off) |
| **Old-class proxies** | KeyCache prototype embeddings | on | on |
| **Conflict replay** | Ridge re-weight from cache embeddings (no disk) | on | **on** |

- **Canonical:** `fscil_preset=canonical` or `model_name=selfevo` with
  `fscil_old_shot_replay=1`.
- **Buffer-free:** `model_name=selfevo_bufferfree` → preset `exemplar_buffer_free`
  in `models/fscil_ablation_config.py`.

### Which buffer-free runs are valid?

| Campaign / log | Date | Valid? | Notes |
|----------------|------|--------|-------|
| `00_BUF_CAMPAIGN_EP5` / `_MEAN` (`abl_*_buffree`) | Jul 11 | **Yes** | No old-shot replay; conflict replay runs |
| `00_SIMPL_ABL_ADAPTER/abl_mn40_baseonly_exbuffree` | Jul 10 | **Yes** | Post-refactor |
| `00_SIMPL_ABL_ADAPTER/abl_adapter_mn40_base_only_bf` | Jul 9 | **No** | Also disabled conflict replay |
| `00_SCANOBJ_XD_RUNs/seel_*_bufferfree` | Jul 8 | **No** | Legacy; no conflict replay at runtime |
| `00_SIMPL_ABLATIONS_MN40/seel_mn40_abl_bufferfree_smoke` | Jul 8 | **No** | Legacy |

**How to verify in logs** (trainer header can be misleading on cross-dataset JSON):

| Runtime signal | Correct buffer-free | Legacy (wrong) |
|----------------|--------------------|--------------------|
| `FSCIL old-shot replay: task=…` | absent | absent |
| `Test-aware conflict replay task=…` | **present** | absent |

**MN40 base-only impact:**

| Run | Conflict replay | Last | Novel@Last |
|-----|-----------------|-----:|-----------:|
| `abl_adapter_mn40_base_only_bf` (wrong) | off | 90.68 | 72.94 |
| `abl_mn40_baseonly_exbuffree` (correct) | on | 91.29 | 78.82 |
| `abl_mn40_ep5_buf` (canonical buffer) | on | 91.41 | 78.82 |

Use **only Jul 10–11 EP5 campaigns** for buffer vs buffer-free paper claims.

---

## Campaign index

| # | Directory | Script | Status | What it tests |
|---|-----------|--------|--------|---------------|
| 1 | `00_SIMPL_RUNs` | `scripts/simpl_runs.sh` | Done (2/2) | Early canonical MN40 + SN55 |
| 2 | `00_SIMPL_REGIME_ABLATIONS` | `scripts/simpl_runs_ablations_regimes.sh` | Done (16/16) | Ridge / cache / static / rules × 4 protocols |
| 3 | `00_SIMPL_ABLATIONS_MN40` | `scripts/simpl_runs_ablations_mn40.sh` | Mostly (13/14) | Module + hyperparameter on MN40 |
| 4 | `00_SIMPL_ABL_ADAPTER` | `scripts/simpl_runs_ablation_adapter.sh` | Done (10/10) | Adapter on/off, base-only |
| 5 | `00_ADAPTER_CAMPAIGN_ID` | `scripts/simpl_runs_adapter_ablation_indomain.sh` | Done (6/6) | Adapter schedule in-domain |
| 6 | `00_ADAPTER_CAMPAIGN_XD` | `scripts/simpl_runs_adapter_ablation_crossdomain.sh` | Done (8/8) | Adapter schedule cross-domain |
| 7 | `00_SCANOBJ_XD_RUNs` | cross-dataset launcher | Done (4/4) | ScanObjectNN transfer (legacy bf invalid) |
| 8 | `00_CO3D_RUNs` | `scripts/simpl_runs_co3d.sh` | Done (2/2) | CO3D in-domain + SN55→CO3D |
| 9 | `00_BUF_CAMPAIGN_EP5` | `scripts/simpl_runs_baseonly_ep5_buffer_campaign.sh` | Done (12/12) | Per-shot KeyCache, buffer vs buffer-free |
| 10 | `00_BUF_CAMPAIGN_EP5_MEAN` | `scripts/simpl_runs_baseonly_ep5_buffer_mean_campaign.sh` | Done (12/12) | Mean-proto KeyCache, buffer vs buffer-free |
| 11 | `00_SCANOBJ_ABL_ADAPTER` | `scripts/simpl_runs_ablation_adapter_scanobjectnn.sh` | **Failed** (0/10) | Increment warmup on cross-dataset |
| 12 | `00_SIMPL_RUNs_ep10` | `scripts/simpl_runs_ep10.sh` | Done (2/2) | 10-epoch task-0 warmup |
| 13 | `00_SIMPL_RUNs_ep10_ep5` | `scripts/simpl_runs_ep10_ep5.sh` | Done (2/2) | 10 ep base + 5 ep increment |
| 14 | `00_SIMPL_RUNs_aug_sgd` | `scripts/simpl_runs_aug_sgd.sh` | Done (1/1) | Point aug + SGD warmup |
| 15 | `00_SIMPL_TOY_ABLATIONS_MN40` | `scripts/simpl_runs_ablations_mn40_toy.sh` | Smoke (1/1) | Fast 2-session regime check |
| **16** | **`00_FINAL_EP5_BASEONLY`** | **`scripts/simpl_runs_final_ep5_baseonly.sh`** | **Pending (0/60)** | **Paper table: 5 proto × buffer × 6 protocols** |

Summarize: `scripts/summarize_fscil_final_campaign.py` (final),
`scripts/summarize_fscil_regime_ablations.py`,
`scripts/summarize_fscil_ablations_mn40.py`,
`scripts/summarize_fscil_adapter_campaign.py`.

---

## Paper story — regime matrix

**Where:** `_experiments/00_SIMPL_REGIME_ABLATIONS/`  
**Scripts:** `scripts/simpl_runs_ablations_regimes.sh` (full),
`scripts/simpl_runs_ablations_regimes_toy.sh` (fast)  
**Matrix:** `ridge` vs `cache` vs `ridge+cache (static)` vs `ridge+cache+evolution (rules)`

### ModelNet40 in-domain (5 sessions)

| Variant | AA | Last | Base@Last | Novel@Last | Harm@Last |
|---------|---:|-----:|----------:|-----------:|----------:|
| Ridge only | 93.32 | 91.13 | 94.48 | 78.24 | 85.60 |
| Cache only | 89.91 | 86.43 | 86.62 | 85.69 | 86.15 |
| Ridge+cache (static) | 93.46 | 91.09 | 94.08 | 79.61 | 86.24 |
| **Ridge+cache+evolution** | **93.63** | **91.41** | **94.23** | **80.59** | **86.88** |

Evolution: **+0.98 Novel@Last**, **+0.64 Harm@Last** vs static. Core paper table.

### ShapeNet55 in-domain (7 sessions)

| Variant | AA | Last | Base@Last | Novel@Last | Harm@Last |
|---------|---:|-----:|----------:|-----------:|----------:|
| Ridge only | 89.63 | 88.30 | 89.87 | 80.85 | 85.12 |
| Cache only | 86.42 | 83.66 | 83.74 | 83.25 | 83.49 |
| Ridge+cache (static) | 89.84 | 88.54 | 89.91 | 82.05 | 85.80 |
| Ridge+cache+evolution | 89.88 | 88.53 | 89.82 | 82.43 | 85.97 |

### MN40 → ScanObjectNN (4 sessions)

| Variant | AA | Last | Base@Last | Novel@Last | Harm@Last |
|---------|---:|-----:|----------:|-----------:|----------:|
| Ridge only | 90.29 | 86.71 | 94.32 | 62.74 | 75.36 |
| Cache only | 87.62 | 84.07 | 88.84 | 69.05 | 77.70 |
| Ridge+cache (static) | 91.42 | 88.43 | 93.92 | 71.16 | 80.97 |
| Ridge+cache+evolution | 91.43 | 88.48 | 93.98 | 71.16 | 80.99 |

Fusion is the main gain; evolution negligible on cross-dataset ScanObj.

### SN55 → ScanObjectNN (4 sessions)

| Variant | AA | Last | Base@Last | Novel@Last | Harm@Last |
|---------|---:|-----:|----------:|-----------:|----------:|
| Ridge only | 90.67 | 87.81 | 93.18 | 33.05 | 48.79 |
| Cache only | 85.16 | 80.10 | 82.96 | 50.95 | 63.13 |
| Ridge+cache (static) | 91.20 | 88.35 | 91.39 | 57.31 | 70.44 |
| Ridge+cache+evolution | 91.14 | 88.24 | 91.27 | 57.31 | 70.41 |

### Toy sanity (2 sessions, MN40 only)

| Variant | AA | Last | Base@Last | Novel@Last | Harm@Last |
|---------|---:|-----:|----------:|-----------:|----------:|
| Ridge only | 94.53 | 93.55 | 95.40 | 73.33 | 82.92 |
| Cache only | 91.74 | 90.32 | 90.91 | 83.89 | 87.26 |
| Ridge+cache (static) | 94.59 | 93.78 | 95.30 | 77.22 | 85.31 |
| Ridge+cache+evolution | 94.59 | 93.78 | 95.30 | 77.22 | 85.31 |

### Evolution story (ICLR narrative)

1. **Stability anchor:** ridge → high Base@Last, lower Novel@Last.
2. **Plasticity module:** cache → high Novel@Last, base collapse.
3. **Fusion:** ridge_cache recovers base while keeping novel gains.
4. **Evolution:** static → rules improves Novel/Harmonic on MN40 in-domain.

---

## MN40 module & hyperparameter ablations

**Where:** `_experiments/00_SIMPL_ABLATIONS_MN40/`  
**Baseline proxy:** `seel_mn40_abl_B300` or `seel_mn40_abl_nogated` (Last 91.41).  
**Do not use:** `seel_mn40_abl_full.log` (truncated).

### Module ablations

| Variant | Config | AA | Last | Novel@Last | Harm@Last |
|---------|--------|---:|-----:|-----------:|----------:|
| Ridge only | `classifier_mode=ridge` | 93.32 | 91.13 | 78.24 | 85.60 |
| Cache only | `classifier_mode=cache` | 89.91 | 86.43 | 85.69 | 86.15 |
| w/o evolution | `evolution_policy=static` | 93.46 | 91.09 | 79.61 | 86.24 |
| w/o conflict replay | `conflict_replay=false` | 93.19 | 91.21 | 76.67 | 84.85 |
| w/o gated growth | `enable_gated_growth=false` | 93.63 | 91.41 | 80.59 | 86.88 |
| Full (B=300) | baseline | 93.63 | 91.41 | 80.59 | 86.88 |

**Conflict replay ablation detail:**

| Setting | AA | Last | Base@Last | Novel@Last | Harm@Last | Δ |
|---------|---:|-----:|----------:|-----------:|----------:|--:|
| Replay ON (full) | 93.63 | 91.41 | 94.23 | 80.59 | 86.88 | 4.40 |
| Replay OFF | 93.19 | 91.21 | 94.99 | 76.67 | 84.85 | 4.65 |
| Change (OFF − ON) | −0.44 | −0.20 | +0.76 | **−3.92** | −2.03 | +0.25 |

Replay helps plasticity (−3.9 Novel@Last without it), not just total Last.

### Hyperparameter ablations (vs full)

| Variant | Last | ΔLast |
|---------|-----:|------:|
| α=0.5 | 91.37 | −0.04 |
| α=0.7 | 91.37 | −0.04 |
| α=1.0 | 91.37 | −0.04 |
| B=100 | 91.05 | −0.36 |
| P=3 fixed | 91.25 | −0.16 |
| P=3 base / P=5 incr | 91.21 | −0.20 |
| Per-shot prototypes | 91.21 | −0.20 |

Defaults (α=0.85, B=200–300) are near-optimal.

---

## Adapter ablations

### Adapter on/off — `00_SIMPL_ABL_ADAPTER`

| Setting | MN40 Last | SN55 Last | Notes |
|---------|----------:|----------:|-------|
| With adapter (full) | 91.41 | 88.53 | |
| Without adapter | 90.96 | 86.20 | SN55 −2.3 Last |
| Base-only (task-0) | 91.49 | 88.98 | |
| Base-only + buffer-free (`base_only_bf`) | 90.68 | 88.90 | **legacy wrong bf** |

Use `abl_mn40_baseonly_exbuffree` (91.29) for correct buffer-free comparison.

### In-domain schedule — `00_ADAPTER_CAMPAIGN_ID`

| Variant | MN40 Last | SN55 Last |
|---------|----------:|----------:|
| Full adapter 10/2 ep | **91.57** | **89.04** |
| Full adapter 10/10 ep | 91.53 | 88.92 |
| Base-only (task-0, 10 ep) | 91.49 | 88.98 |

Base-only within **0.1%** of full adapter in-domain.

### Cross-domain schedule — `00_ADAPTER_CAMPAIGN_XD`

| Variant | MN40→ScanObj | SN55→ScanObj |
|---------|-------------:|-------------:|
| Full adapter 10/2 ep | 88.03 | 90.53 |
| Full adapter 10/10 ep | 88.03 | 90.62 |
| Base-only (task-0) | **88.23** | 90.28 |
| Base + task-1 warmup 2 ep | 88.18 | **90.77** |

SN55→ScanObj benefits from task-1 warmup (+0.24 Last vs full 2ep).

---

## Cross-dataset & CO3D runs

### ScanObjectNN transfer — `00_SCANOBJ_XD_RUNs` (full adapter)

| Run | AA | Last | Novel@Last | Harm@Last | Notes |
|-----|---:|-----:|-----------:|----------:|-------|
| MN40→ScanObj (buffer) | 91.43 | 88.48 | 71.16 | 80.99 | |
| MN40→ScanObj (buffer-free) | 91.72 | 89.09 | 70.95 | 81.18 | **legacy bf — invalid** |
| SN55→ScanObj (buffer) | 91.14 | 88.24 | 57.31 | 70.41 | |
| SN55→ScanObj (buffer-free) | 91.07 | 88.33 | 52.84 | 67.08 | **legacy bf — invalid** |

### CO3D — `00_CO3D_RUNs` (full adapter)

| Run | Sessions | AA | Last | Novel@Last | Harm@Last |
|-----|----------|---:|-----:|-----------:|----------:|
| CO3D in-domain | 6 | 72.26 | 65.99 | 25.80 | 38.82 |
| SN55 → CO3D | 11 | 86.03 | 78.09 | 26.10 | 40.63 |

CO3D is the hardest benchmark — main open problem.

### Early canonical — `00_SIMPL_RUNs` (superseded)

| Protocol | AA | Last |
|----------|---:|-----:|
| MN40 | 90.51 | 86.83 |
| SN55 | 89.28 | 87.80 |

Predates flex-compare configs; use regime matrix numbers (~91.4 / ~88.5 Last).

---

## Buffer & prototype ablations (EP5, base-only)

**Campaigns:** `00_BUF_CAMPAIGN_EP5` (per-shot KeyCache),
`00_BUF_CAMPAIGN_EP5_MEAN` (mean-proto KeyCache)  
**Scripts:** `scripts/simpl_runs_baseonly_ep5_buffer_campaign.sh`,
`scripts/simpl_runs_baseonly_ep5_buffer_mean_campaign.sh`

**Shared config (all EP5 buffer runs):**

- `fscil_adapter_base_only=true`
- `task0_adapter_warmup=true`, **`tuned_epoch=5`** (5-epoch task-0 warmup)
- `increment_adapter_warmup=false`
- Cross-domain: task-1 adapter warmup 2 ep enabled
- Per-shot campaign: default few-shot KeyCache prototypes
- Mean campaign: `fscil_shot_prototypes=false`, `insert_strategy=mean`

### Buffer vs buffer-free (per-shot KeyCache)

| Protocol | Buffer | Buffer-free | Δ (bf − buf) |
|----------|-------:|------------:|-------------:|
| MN40 in-domain | 91.41 | 91.37 | −0.04 |
| SN55 in-domain | 88.76 | 88.88 | +0.12 |
| CO3D in-domain | 67.96 | 67.32 | −0.64 |
| MN40 → ScanObjectNN | 86.45 | 86.45 | 0.00 |
| SN55 → ScanObjectNN | 90.50 | 90.77 | +0.27 |
| SN55 → CO3D | 82.01 | 81.74 | −0.27 |

Buffer design is **neutral** (~0.3% on most protocols) under correct buffer-free.

### Per-shot vs mean-proto — buffer variant

| Protocol | Per-shot | Mean-proto | Δ (mean − shot) |
|----------|--------:|-----------:|----------------:|
| MN40 in-domain | 91.41 | 91.29 | −0.12 |
| SN55 in-domain | 88.76 | 88.62 | −0.14 |
| CO3D in-domain | 67.96 | 66.86 | −1.10 |
| MN40 → ScanObjectNN | 86.45 | 86.71 | +0.26 |
| SN55 → ScanObjectNN | 90.50 | 90.48 | −0.02 |
| SN55 → CO3D | 82.01 | 81.83 | −0.18 |

Mean-proto slightly worse in-domain; neutral on cross-dataset.

### Per-shot vs mean-proto — buffer-free variant

| Protocol | Per-shot | Mean-proto | Δ (mean − shot) |
|----------|--------:|-----------:|----------------:|
| MN40 in-domain | 91.37 | 91.00 | **−0.37** |
| SN55 in-domain | 88.88 | 88.76 | −0.12 |
| CO3D in-domain | 67.32 | 65.42 | **−1.90** |
| MN40 → ScanObjectNN | 86.45 | 86.71 | +0.26 |
| SN55 → ScanObjectNN | 90.77 | 90.73 | −0.04 |
| SN55 → CO3D | 81.74 | 81.81 | +0.07 |

Mean-proto gap is **larger under buffer-free** (especially CO3D, −1.9). Per-shot
diversity matters more without exemplar disk re-read.

### Buffer vs buffer-free — mean-proto only

| Protocol | Mean buffer | Mean buffer-free | Δ |
|----------|------------:|-----------------:|--:|
| MN40 | 91.29 | 91.00 | −0.29 |
| SN55 | 88.62 | 88.76 | +0.14 |
| CO3D | 66.86 | 65.42 | −1.44 |
| MN40→ScanObj | 86.71 | 86.71 | 0.00 |
| SN55→ScanObj | 90.48 | 90.73 | +0.25 |
| SN55→CO3D | 81.83 | 81.81 | −0.02 |

**Not yet run:** mean-proto or per-shot buffer ablations under **full** adapter
(only base-only ep5 above). Regime matrix has no mean-proto variant.

---

## Other ablations

### Warmup epochs — `00_SIMPL_RUNs_ep10` / `_ep10_ep5`

| Config | MN40 Last | SN55 Last |
|--------|----------:|----------:|
| Default (regime rules) | 91.41 | 88.53 |
| Task-0 warmup 10 ep | 91.41 | 88.53 |
| Task-0 10 ep + increment 5 ep | 91.41 | **89.06** |

Neutral on MN40; SN55 +0.5 Last with ep10+ep5.

### Augmentation — `00_SIMPL_RUNs_aug_sgd`

| Run | AA | Last |
|-----|---:|-----:|
| MN40 + aug + SGD | 93.75 | 91.65 |

Slightly above default; single run, needs controlled rerun.

### Cross-dataset increment warmup — `00_SCANOBJ_ABL_ADAPTER` ❌

All 10 runs failed at task 1:

```
RuntimeError: size mismatch for fc.weight: [26, 1024] vs [30, 1024]
```

Fix `_restore_task0_adapter_anchor_if_needed()` to handle growing `fc`, then re-submit.

---

## Per-session curves

From `_experiments/00_SIMPL_REGIME_ABLATIONS/abl_*_*.log`.

### ModelNet40 FSCIL (5 sessions)

| Variant | Total | Base | Novel | Harmonic |
|---------|-------|------|-------|----------|
| Ridge only | 95.81 → … → 91.13 | 95.81 → … → 94.48 | 74.44 → … → 78.24 | 83.69 → … → 85.60 |
| Cache only | 93.62 → … → 86.43 | 93.62 → … → 86.62 | 84.44 → … → 85.69 | 87.70 → … → 86.15 |
| Ridge+cache (static) | 95.81 → … → 91.09 | 95.81 → … → 94.08 | 78.33 → … → 79.61 | 85.97 → … → 86.24 |
| Ridge+cache+evolution | 95.81 → … → 91.41 | 95.81 → … → 94.23 | 78.33 → … → 80.59 | 85.97 → … → 86.88 |

Full curves: see git history of this file or raw logs.

### ShapeNet55 FSCIL (7 sessions)

| Variant | Total (last) | Base (last) | Novel (last) | Harm (last) |
|---------|-------------|-------------|--------------|-------------|
| Ridge only | 88.30 | 89.87 | 80.85 | 85.12 |
| Cache only | 83.66 | 83.74 | 83.25 | 83.49 |
| Ridge+cache (static) | 88.54 | 89.91 | 82.05 | 85.80 |
| Ridge+cache+evolution | 88.53 | 89.82 | 82.43 | 85.97 |

### MN40 → ScanObjectNN (4 sessions)

| Variant | Total (last) | Base (last) | Novel (last) | Harm (last) |
|---------|-------------|-------------|--------------|-------------|
| Ridge only | 86.71 | 94.32 | 62.74 | 75.36 |
| Cache only | 84.07 | 88.84 | 69.05 | 77.70 |
| Ridge+cache (static) | 88.43 | 93.92 | 71.16 | 80.97 |
| Ridge+cache+evolution | 88.48 | 93.98 | 71.16 | 80.99 |

### SN55 → ScanObjectNN (4 sessions)

| Variant | Total (last) | Base (last) | Novel (last) | Harm (last) |
|---------|-------------|-------------|--------------|-------------|
| Ridge only | 87.81 | 93.18 | 33.05 | 48.79 |
| Cache only | 80.10 | 82.96 | 50.95 | 63.13 |
| Ridge+cache (static) | 88.35 | 91.39 | 57.31 | 70.44 |
| Ridge+cache+evolution | 88.24 | 91.27 | 57.31 | 70.41 |

---

## Cross-cutting conclusions

### Solid for the paper

| Claim | Source |
|-------|--------|
| Ridge + cache fusion + evolution story | Regime matrix |
| Replay helps plasticity | MN40 `noreplay` ablation |
| Base-only adapter sufficient in-domain | Adapter campaigns ID |
| Cross-dataset ScanObj fusion gain | Regime matrix + SCANOBJ_XD |

### Neutral / weak

| Claim | Source |
|-------|--------|
| Buffer vs buffer-free (~0.1% gap) | EP5 campaigns (correct bf only) |
| Mean-proto ≈ per-shot cross-domain | EP5_MEAN |
| Mean-proto worse in-domain | EP5_MEAN |
| Evolution on cross-dataset | Regime matrix |
| α, gated growth insensitive | MN40 hyper ablations |

### Suggested paper tables

| Table | Campaign |
|-------|----------|
| Main method vs ridge/cache/static/rules | `00_SIMPL_REGIME_ABLATIONS` |
| Module ablation (replay, evolution) | `00_SIMPL_ABLATIONS_MN40` |
| Adapter schedule | `00_ADAPTER_CAMPAIGN_*` |
| Cross-dataset + CO3D | Regime + `00_CO3D_RUNs` |
| **Buffer / prototype (paper supplementary)** | **`00_FINAL_EP5_BASEONLY`** |
| Legacy buffer / prototype | `00_BUF_CAMPAIGN_EP5*` (test-aware replay) |

---

## Open gaps & next steps

| Gap | Priority |
|-----|----------|
| **Run `00_FINAL_EP5_BASEONLY` (60 jobs)** | **High** |
| Multi-seed (3–5) on final campaign | **High** |
| Fix `00_SCANOBJ_ABL_ADAPTER` fc resize crash | Medium |
| Evolution operator analysis (merges/splits vs session) | Medium |
| `task0_use_best_warmup` ablation (test-free adapter pick) | Low |

### Evolution playground (supported in code)

Policies: `rules` (default), `static`, `aggressive`, `bandit`, `rl`.

```bash
--set evolution_policy=aggressive
--set evolution_policy=bandit --set bandit_ucb_c=1.0
--set evolution_policy=rl --set rl_lr=1e-3
```

Operator sensitivity: `merge_threshold`, `split_confusion_threshold`,
`recall_refresh_threshold`.

---

## Log paths & summarize commands

```
_experiments/00_SIMPL_RUNs/
_experiments/00_SIMPL_REGIME_ABLATIONS/
_experiments/00_SIMPL_ABLATIONS_MN40/
_experiments/00_SIMPL_ABL_ADAPTER/
_experiments/00_ADAPTER_CAMPAIGN_ID/
_experiments/00_ADAPTER_CAMPAIGN_XD/
_experiments/00_SCANOBJ_XD_RUNs/
_experiments/00_CO3D_RUNs/
_experiments/00_BUF_CAMPAIGN_EP5/
_experiments/00_BUF_CAMPAIGN_EP5_MEAN/
_experiments/00_SCANOBJ_ABL_ADAPTER/          # failed
_experiments/00_SIMPL_RUNs_ep10/
_experiments/00_SIMPL_RUNs_ep10_ep5/
_experiments/00_SIMPL_RUNs_aug_sgd/
_experiments/00_SIMPL_TOY_ABLATIONS_MN40/
_experiments/00_FINAL_EP5_BASEONLY/
```

```bash
aprun python scripts/summarize_fscil_final_campaign.py \
  --log-root _experiments/00_FINAL_EP5_BASEONLY

aprun python scripts/analyze_cache_per_class.py \
  _experiments/00_FINAL_EP5_BASEONLY/final_*_seed=2008.log

aprun python scripts/summarize_fscil_regime_ablations.py \
  --log-root _experiments/00_SIMPL_REGIME_ABLATIONS

aprun python scripts/summarize_fscil_adapter_campaign.py \
  --id-root _experiments/00_ADAPTER_CAMPAIGN_ID \
  --xd-root _experiments/00_ADAPTER_CAMPAIGN_XD \
  --out /tmp/adapter_summary.md

aprun python scripts/summarize_fscil_ablations_mn40.py \
  --log-root _experiments/00_SIMPL_ABLATIONS_MN40 \
  --out /tmp/mn40_ablations.md
```
