# FROST

**FRO**zen **ST**atistics — exemplar-free few-shot class-incremental learning (FSCIL) with a frozen pretrained encoder and two complementary analytical memories.

Paper components (Sec. 3):

| Paper name | Code | Role |
|------------|------|------|
| **Adapter** | AdaptFormer on frozen Uni3D / ViT | Base-session feature adaptation |
| **CorrMem** | `frost/corrmem.py` | Analytical ridge head after random lift \(V{=}2000\) |
| **ProtoMem** | `frost/protomem.py` | Mean prototype memory (static, 1 key/class) |
| **Fusion** | `frost/frost.py` `_classify` | \(\alpha\cdot\mathrm{CorrMem}+(1-\alpha)\cdot\mathrm{ProtoMem}\), \(\alpha{=}0.5\) |

Variants: **EF** (`frost_bufferfree`, exemplar-free) and **EB** (`frost` + `fscil_preset=canonical`).

## Repository layout

```text
frost/                  # Paper method (separated from baselines)
configs/
  base/{pclouds,images}.json   # Shared paper recipe
  frost_pc_*.json / frost_img_*.json
datasets/
  session_pclouds/      # FSCIL-3D session indexes (vendored)
  session_images/       # CUB / Aircraft / iNF200 session lists
  preprocess/           # Download + FPS/normals builders
models/                 # Baselines only (RanPAC, SimpleCIL, prompts, …)
scripts/
  reproduce_*.sh        # Paper table entrypoints
utils/                  # Config load + FSCIL presets
train_fscil.py          # 3D entry
train_fscil_image.py    # Image entry
```

Raw data and weights are **not** shipped. See [`DATASETS.md`](DATASETS.md).

## Quick start

1. **Environment** — create a Python env with PyTorch and project dependencies.  
   *(Full install instructions will be added with the public release; until then use your own CUDA/PyTorch stack.)*
2. **Data** — obtain datasets under `_datasets/` per [`DATASETS.md`](DATASETS.md) (accept each ToU first).
3. **Weights** — Uni3D-B and ViT-B/16-IN21k under `_experiments/pretrained_weights/` (ViT: `datasets/preprocess/download_vit_in21k.sh`).
4. **Run** — paper recipes via configs or SLURM launchers (commands use plain `python`):

```bash
# Single protocol, first paper seed (default)
python train_fscil.py --config configs/frost_pc_in_mn40.json
python train_fscil_image.py --config configs/frost_img_cub200.json

# All five paper seeds
python train_fscil.py --config configs/frost_pc_in_mn40.json --fullseeds

# Full table campaigns (SLURM)
sbatch scripts/reproduce_indomain.sh          # MN40 + SN55 + CO3D (EF+EB)
sbatch scripts/reproduce_crossdomain.sh       # MN/SN → ScanObj + SN→CO3D
sbatch scripts/reproduce_image.sh             # CUB / iNF / Aircraft (EF)
```

Pass `--fullseeds` on the reproduce scripts for the five-seed paper set  
`1947, 1991, 2001, 2008, 2019`. Default is the first seed only.

EF-only / EB-only:

```bash
sbatch --array=0-14 scripts/reproduce_indomain.sh --buffer-free
sbatch --array=0-14 scripts/reproduce_indomain.sh --buffer
```

Optional: if your cluster wraps Python (e.g. EuroHPC `aprun`), set  
`export FROST_PYTHON="aprun python"` before running the reproduce scripts.

## Paper recipe (locked)

| Knob | Value |
|------|-------|
| ProtoMem insert | mean only (1 key/class), `static` |
| CorrMem lift \(V\) | **2000** (`corrmem.lift_v`) |
| Fusion | `corr_proto`, \(\alpha{=}0.5\) flat |
| Adapter | base session only, 10 epochs |
| Conflict CM | train-side (`conflict_replay_use_test_cm=false`) |

Details: [`SETUP.md`](SETUP.md) · configs: [`configs/README.md`](configs/README.md) · scripts: [`scripts/README.md`](scripts/README.md).

## Summarize logs

```bash
python scripts/summarize_pride_multiseed.py --variant final_sub_both
bash scripts/summarize_frost_image_multiseed.sh \
  --log-root _experiments/00_PRIDE_IMAGE_MEANONLY_A05_RP2000_5SEEDS \
  --insert mean1
```

Paper metrics: \(\mathcal{A}_b\) = session 0, \(\mathcal{A}_T\) = last session,  
\(\bar{\mathcal{A}}\) = **mean of per-session total accuracy** (log line `Average Accuracy (CNN)`),  
**not** \((\mathcal{A}_b+\mathcal{A}_T)/2\).

## Licence note on data

Do **not** redistribute ModelNet40, ShapeNet, ScanObjectNN, CO3D, CUB-200-2011, FGVC-Aircraft, or iNF200 with this code. Users must download from official sources after accepting each licence / ToU. Scripts and layout: [`DATASETS.md`](DATASETS.md).

## Citation

If you use this code, please cite the FROST paper (ICLR 2027 submission) and the upstream datasets / Uni3D / LAMDA-PILOT toolkit as appropriate.
