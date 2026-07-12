# Experimental Setup and Implementation Details

Code-derived reference for FSCIL experiments. Values map to config keys, code defaults, or campaign-script `--set` flags.

**Primary sources:** `configs/selfevo_fscil_*.json`, `models/selfevo.py`, `models/ridge_head.py`, `models/key_cache.py`, `backbone/uni3d_adapter.py`, `datasets/fscil_protocol.py`, `models/fscil_ablation_config.py`, `scripts/simpl_runs_final_ep10_baseonly.sh`, `scripts/simpl_runs_final_hybrid_closefar.sh`.

---

## 1. Software and hardware

| Item | Value | Source |
|------|-------|--------|
| Framework | PyTorch | `train_fscil.py` |
| Entry point | `aprun python train_fscil.py --config …` | repo rules + SLURM scripts |
| GPU | 1 GPU per job (`--gpus=1`) | SLURM scripts |
| Host memory | 64 GB (`--mem=64G`) | SLURM scripts |
| CPUs | 24 (`--cpus-per-task=24`) | SLURM scripts |
| Random seed | `2008` | FSCIL configs `seed: [2008]`; campaign scripts `SEED=2008` |
| Exemplar replay seed | `2008` | `fscil_old_shot_replay_seed: 2008` |
| Exemplar memory buffer | disabled | `memory_size: 0`, `memory_per_class: 0` |

---

## 2. Backbone and encoder

| Item | Value | Config key |
|------|-------|------------|
| Method | `selfevo` (or `selfevo_bufferfree` for buffer-free) | `model_name` |
| Pretrained encoder | Uni3D-Base | `pretrained_pc: _experiments/pretrained_weights/Uni3D/uni3d-b.pt` |
| Backbone type | `uni3d_base_patch14_448` | `backbone_type` |
| Internal point transformer | `eva02_base_patch14_448` | `pc_model` |
| Embedding dim \(d\) | **1024** | `embed_dim: 1024` |
| Transformer dim | 768 | `trans_dim: 768` |
| Encoder dim | 512 | `encoder_dim: 512` |
| Point groups | 512 groups × 64 points | `num_group: 512`, `group_size: 64` |
| Text encoder | **not used** | `classifier_mode: ridge_cache` |

Working representation: \(\mathbf{z} = (\phi \circ f_\theta)(\mathbf{x}) \in \mathbb{R}^{1024}\).

Unlike vision–language FSCIL methods, this pipeline is **vision-only** (no text encoder).

---

## 3. Adapter \(\phi\)

| Item | Value | Source |
|------|-------|--------|
| Type | AdaptFormer-style bottleneck MLP per transformer block | `backbone/uni3d_adapter.py` |
| Bottleneck dim \(d'\) | **64** | `ffn_num: 64` |
| Init | LoRA-style | `ffn_adapter_init_option: "lora"` (adapter default) |
| Adapter scale | 0.1 | `ffn_adapter_scalar: "0.1"` |
| Backbone | **frozen** always | `selfevo.py` `_freeze_backbone()` |
| Trainable at task 0 | adapter (+ optionally FC during warmup) | `_set_adapter_warmup_trainable()` |

### Final campaign overrides

`scripts/simpl_runs_final_ep10_baseonly.sh` and `scripts/simpl_runs_final_hybrid_closefar.sh` set:

| Flag | Effect |
|------|--------|
| `fscil_adapter_base_only=true` | adapter trained **only** on base session |
| `increment_adapter_warmup=false` | no adapter training on novel sessions |
| `fscil_task1_adapter_warmup=false` | no task-1 adapter warmup |
| `fscil_classifier_warmup_after_protos=false` | no classifier-aligned adapter warmup after protos |

**Note:** canonical JSON alone (e.g. `selfevo_fscil_mn40_5t_flex_compare.json`) still has `increment_adapter_warmup: true`. Final paper runs use the **campaign script overrides** above.

---

## 4. Base-session adapter training (task \(t=0\))

| Item | Value | Config / code |
|------|-------|---------------|
| Epochs | **10** | `tuned_epoch: 10` |
| Optimizer | **SGD**, momentum **0.9** | `task0_optimizer: "sgd"`; `optim.SGD(..., momentum=0.9)` |
| Learning rate | **\(10^{-3}\)** | `task0_init_lr: 0.001` |
| Weight decay | **\(10^{-4}\)** | `task0_weight_decay: 0.0001` |
| LR schedule | Cosine annealing to `min_lr=0` | `CosineAnnealingLR(T_max=tuned_epoch, eta_min=0)` |
| Batch size | **8** | `batch_size: 8` |
| DataLoader workers | 4 | `num_workers: 4` |
| Checkpoint selection | best test accuracy epoch | `task0_use_best_warmup: true` |
| Save warmup ckpts | yes | `task0_save_warmup_ckpts: true` |
| Point augmentation at task 0 | **off** | `task0_point_augment: false` |
| Training data | full base-class train split only | `prepare_fscil_task_datasets`, task 0 |

---

## 5. Point-cloud input

| Item | Value | Config |
|------|-------|--------|
| Points per shape \(M\) | **1024** | `num_point: 1024` |
| Sampling | farthest-point (uniform) | `use_uniform_sample: true` |
| Normals | **on** (6D inputs) | `use_normals: true` |
| Preprocessing | on | `process_data: true` |
| Class order | fixed | `shuffle: false`, `fscil3d_protocol` set |

---

## 6. FSCIL session protocol

| Item | Value | Source |
|------|-------|--------|
| Protocol | FSCIL-3D class order | `fscil3d_protocol` + `datasets/fscil3d_index.py` |
| Few-shot protocol | on | `fscil_shot_protocol: true` |
| Base session data | full train split per base class | `datasets/fscil_protocol.py` |
| Novel session data | **5 shots/class** | `increment_train_per_class: 5` |
| Old-class disk replay | **1 shot/class** (buffer variant) | `fscil_old_shot_replay: 1` |
| Buffer-free variant | **0** old shots | `model_name=selfevo_bufferfree` → `exemplar_buffer_shots: 0` |

### Session schedules (`load_protocol`)

| Protocol | Sessions | Base | Novel | Total |
|----------|----------|------|-------|-------|
| ModelNet40 | 5 | 20 | 20 | 40 |
| ShapeNet | 7 | 25 | 30 | 55 |
| CO3D | 6 | 25 | 25 | 50 |
| MN40 → ScanObjectNN | 4 | 26 | 11 | 37 |
| SN55 → ScanObjectNN | 4 | 44 | 15 | 59 |
| SN55 → CO3D | 11 | 39 | 50 | 89 |

### Intra- vs cross-dataset

- **Intra-dataset:** base and novel classes from the same benchmark (ModelNet40, ShapeNet, CO3D).
- **Cross-dataset:** synthetic source for base session, real-scan target for novel classes (MN40→ScanObj, SN55→ScanObj, SN55→CO3D).

---

## 7. Per-session pipeline

Order in `models/selfevo.py::_train` at each session \(t\):

1. **Adapter warmup** (task 0 only in final campaigns)
2. Freeze backbone; collect embeddings from `train_loader_for_protonet`
3. **Insert KeyCache prototypes**
4. **Ridge update** on \(\mathcal{D}_t^{\mathrm{ridge}}\):
   - novel few-shot embeddings (+ optional exemplar buffer shots in proto dataset)
   - optional `_refresh_ridge_from_exemplar_buffer()` at \(t>0\) (1 shot/class, weight `fscil_base_ridge_refresh_weight=1.0`)
   - few-shot ridge up-weight: `fscil_novel_ridge_weight` (1.5 MN40, 2.5 SN55)
   - optional noise doubling: `fscil_ridge_proto_noise_std=0.015`
5. Optional classifier warmup after protos (off in final campaigns)
6. Build **adaptation batch** \(\mathcal{A}_t = \mathcal{D}_t^{\mathrm{novel}} \cup \mathcal{P}_t\) where \(\mathcal{P}_t\) = KeyCache keys (`old_data_source: cache`)
7. **Cache evolution** (if `evolution_policy=rules`)
8. **Conflict replay** into ridge (if enabled)

**Ridge vs adaptation batch:** primary \(\mathbf{G}_t,\mathbf{Q}_t\) accumulation uses \(\mathcal{D}_t^{\mathrm{ridge}} = \mathcal{D}_t^{\mathrm{novel}} \cup \mathcal{E}_t\) (few-shot + optional disk exemplars). The adaptation batch (few-shot + cache prototypes) drives **cache evolution**; cache prototypes enter ridge only via optional **conflict replay**.

---

## 8. Analytical ridge head

| Item | Value | Config / code |
|------|-------|---------------|
| Classifier mode | ridge + cache fusion | `classifier_mode: ridge_cache` |
| Random projection dim \(B\) | **1000** | `ridge_rp_dim: 1000` |
| Random map | \(h(\mathbf{z})=\mathrm{ReLU}(\mathbf{z} W_{\mathrm{rand}})\) | `RidgeHead._project()` |
| \(W_{\mathrm{rand}}\) | sampled once, seed = run seed | `RidgeHead.__init__(seed=…)` |
| Sufficient stats | \(\mathbf{G}\in\mathbb{R}^{B\times B}\), \(\mathbf{Q}\in\mathbb{R}^{B\times C_t}\) | `RidgeHead.G`, `RidgeHead.Q` |
| Ridge penalty \(\lambda\) | grid search at task 0 | `fit_ridge=True` when `ridge_head.ridge is None` |
| \(\lambda\) grid min | \(10^{-8}\) | `ridge_lambda_min` default in `selfevo.py` |
| \(\lambda\) grid max | auto (up to \(10^8\)) or `ridge_lambda_max` | SN55: `ridge_lambda_max: 100000.0` |
| Validation fraction for \(\lambda\) | 20% | `ridge_val_fraction: 0.2` (code default) |
| Closed form | \(W=(\mathbf{G}+\lambda I)^{-1}\mathbf{Q}\) | `RidgeHead._solve_weights()` |
| Config key for \(\lambda\) | `ridge_lambda` (optional); stored as `RidgeHead.ridge` | `models/ridge_head.py` |

---

## 9. Ridge–cache fusion

| Item | Value | Config |
|------|-------|--------|
| Fusion | \(\alpha_t \ell^{\mathrm{ridge}} + (1-\alpha_t)\ell^{\mathrm{cache}}\) | `_classify()`, `ridge_cache` mode |
| Base \(\alpha\) | **0.85** | `ridge_cache_alpha: 0.85` |
| Increment \(\alpha\) floor | **0.45** | `fscil_increment_ridge_alpha: 0.45` |
| Task-1 \(\alpha\) | **0.45** (if set) | `fscil_task1_ridge_alpha: 0.45` |
| Last-task \(\alpha\) (SN55) | **0.45** | `fscil_last_task_ridge_alpha: 0.45` |

On few-shot sessions, `_effective_ridge_alpha()` reduces \(\alpha_t\) from 0.85 toward 0.45.

---

## 10. KeyCache (canonical JSON — MN40 example)

| Item | Value | Config |
|------|-------|--------|
| Feature dim | 1024 | `embed_dim` |
| Insert strategy (base task) | mean (1 proto/class) | `insert_strategy` default `"mean"` |
| Increment insert (JSON default) | per-shot if `fscil_shot_prototypes=true` | `_fscil_prototype_settings()` |
| Max prototypes/class | 3 (JSON) | `max_prototypes_per_class: 3` |
| Increment max protos | 8 (JSON) | `fscil_increment_max_prototypes: 8` |
| Cache budget | 200 (MN40), 380 (SN55) | `cache_budget` |
| Pooling | log-sum-exp | `prototype_pooling: logsumexp` |
| Temperature \(\tau\) | 0.07 | `pooling_temperature: 0.07` |
| Evolution | rules (merge/split/refresh/evict) | `evolution_policy: rules` |
| Merge threshold | 0.95 | `merge_threshold: 0.95` |
| Split confusion threshold | 0.15 | `split_confusion_threshold: 0.15` |
| Recall refresh threshold | 0.85 | `recall_refresh_threshold: 0.85` |
| Gated growth | on (MN40/SN55) | `enable_gated_growth: true` |
| Gated growth recall thr. | 0.92 | `gated_growth_recall_threshold: 0.92` |
| Gated growth max classes | 4 | `gated_growth_max_classes: 4` |
| Novel gated growth | on | `fscil_gated_growth_novel: true` |
| Novel gated growth thr. | 0.80 | `fscil_novel_gated_growth_threshold: 0.80` |
| Novel gated growth max cls | 5 | `fscil_novel_gated_growth_max_classes: 5` |
| Old-class proxies | from cache keys | `old_data_source: cache` |

---

## 11. Exemplar buffer and conflict replay

### Buffer variant (`fscil_preset=canonical`)

| Item | Value |
|------|-------|
| Old-class disk shots | 1/class |
| Ridge refresh shots | 1/class |
| Old-class proxies for adaptation | KeyCache keys (`old_data_source: cache`) |

### Buffer-free (`model_name=selfevo_bufferfree`)

| Item | Value |
|------|-------|
| Disk exemplars | 0 |
| Proxies | KeyCache only |

### Conflict replay (MN40 canonical JSON)

| Item | Value |
|------|-------|
| Enabled | `conflict_replay: true` |
| CM source (JSON) | test split (`conflict_replay_on_test: true`) |
| CM source (final ep10 script) | **train** (`conflict_replay_on_test=false`) |
| Max old classes replayed | 20 |
| Max novel classes replayed | 5 |
| Old replay weight | 2.5 |
| Novel replay weight | 5.0 |
| Old recall threshold | 0.85 |
| Novel recall threshold | 0.55 (MN40), 0.50 (SN55) |

---

## 12. Prototype regimes (final campaigns)

Campaign scripts **override** JSON defaults.

### `simpl_runs_final_ep10_baseonly.sh`

6 protocols × 2 buffer modes × 6 regimes = 72 jobs. Logs: `_experiments/00_FINAL_EP10_BASEONLY/`.

| Regime | `fscil_shot_prototypes` | Insert strategy | Max proto/class | Evolution | Keys/class (incr.) |
|--------|-------------------------|-----------------|-----------------|-----------|-------------------|
| `mean_st` | false | mean | 1 | static | 1 |
| `pshot_evo` | true | per_sample* | 3 | rules | up to 5 |
| `pshot_evo5` | true | per_sample* | 5 | rules | up to 5 |
| `pshot_st` | true | per_sample* | 5 | static | up to 5 |
| `hybrid3_st` | false | mean_close_far | 3 | static | mean + close + far |
| `hybrid4_evo` | false | mean_close_far | 3 | rules | up to 4 |

\*When `fscil_shot_prototypes=true`, `_fscil_prototype_settings()` forces `per_sample` on increment tasks.

### `simpl_runs_final_hybrid_closefar.sh`

6 protocols × 4 regimes, buffer-free only. Logs: `_experiments/00_FINAL_EP5_HYBRID_CLOSEFAR/`. Base adapter: **5 epochs** (`tuned_epoch=5`).

| Regime | Insert strategy | Max proto/class | Evolution |
|--------|-----------------|-----------------|-----------|
| `hybrid3_st` | mean_close_far | 3 | static |
| `hybrid4_evo` | mean_close_far | 3 | rules |
| `hybrid_nearmean_st` | mean_nearmean | 2 | static |
| `fps5_st` | cluster (FPS) | 5 | static |

### Hybrid insert semantics (`models/key_cache.py`)

- `mean_close_far`: class mean + closest shot + farthest shot to mean
- `mean_nearmean`: class mean + shot closest to mean
- `cluster`: farthest-point sampling on few-shot embeddings

---

## 13. Per-protocol config differences

Not all hyperparameters are shared across benchmarks.

| Setting | MN40 | SN55 | Cross-domain (MN→ScanObj) |
|---------|------|------|---------------------------|
| `cache_budget` | 200 | 380 | 200 |
| `fscil_novel_ridge_weight` | 1.5 | 2.5 | (not in JSON) |
| `conflict_replay` | true | true | **false** |
| `increment_adapter_warmup` | true (JSON) | true (JSON) | true (JSON) |
| `fscil_task1_adapter_warmup` | false | false | **true** |
| `ridge_lambda_max` | auto | 100000 | auto |

---

## 14. Canonical config files

| Protocol | Config |
|----------|--------|
| ModelNet40 intra | `configs/selfevo_fscil_mn40_5t_flex_compare.json` |
| ShapeNet intra | `configs/selfevo_fscil_sn55_7t_flex_compare.json` |
| CO3D intra | `configs/selfevo_fscil_co3d_6t.json` |
| MN40 → ScanObjectNN | `configs/selfevo_fscil_mn40_scanobjectnn_4t.json` |
| SN55 → ScanObjectNN | `configs/selfevo_fscil_sn55_scanobjectnn_4t.json` |
| SN55 → CO3D | `configs/selfevo_fscil_sn55_co3d_11t.json` |

---

## 15. Launch commands

```bash
# Final ep10 base-only campaign (72 jobs)
sbatch scripts/simpl_runs_final_ep10_baseonly.sh

# Hybrid close/far ablation (24 jobs, buffer-free, ep5 adapter)
sbatch scripts/simpl_runs_final_hybrid_closefar.sh

# Single local smoke
bash scripts/simpl_runs_final_ep10_baseonly.sh --local --smoke --array 0
```

Always run Python via `aprun python` on Leonardo.

---

## 16. Paper writing notes

- Use **ridge penalty** \(\lambda\) (config: `ridge_lambda`, code: `RidgeHead.ridge`). Do not confuse with fusion weight \(\alpha_t\) (`ridge_cache_alpha`).
- Batch size is **8**, not 32. Base adapter LR is **\(10^{-3}\)**, not \(10^{-4}\) (weight decay is \(10^{-4}\)).
- Pick one prototype regime as the **main method** in the paper; list others as ablations (Section 12).
- JSON defaults ≠ final campaign overrides: cite which launcher you used for main results.
