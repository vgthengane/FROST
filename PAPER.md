# PAPER.md — ICLR writeup guide (PRidE / closefar+mean)

**Last updated:** 2026-07-12  
**Method name:** **PRidE**  
**Seed for FSCIL numbers unless noted:** 2008  
**Backbone:** frozen Uni3D-b  
**Status:** Working notes for paper drafting (not the camera-ready LaTeX).  
**LaTeX draft:** `_paper/` (abstract still mentions evolution — update to match this doc).

### Method naming (locked)

| | |
|--|--|
| **Name** | **PRidE** |
| **Reading** | **P**rototype-augmented **Rid**ge for **E**xemplar-free … |
| **Paper title** | Prototype-Augmented Ridge for Exemplar-Free Few-Shot Incremental Point Cloud Learning |
| **Title with acronym cue (LaTeX)** | see below |
| **One-liner** | Frozen Uni3D + analytical ridge (stability) fused with a bounded prototype memory (plasticity); domain-aware inserts (close–far in-domain, mean cross-dataset). |
| **LaTeX macro** | `\newcommand{\ourmethod}{PRidE}` |
| **Code (legacy)** | `selfevo` / `ridge_cache` — keep in repo; paper says PRidE |
| **Do not use in paper** | CAMERA, SelfEvo, CacheEvo, ProRidge (superseded), “self-evolving” as the method name |

**Title (recommended):**

```latex
\title{PRidE: \textbf{P}rototype-Augmented \textbf{Rid}ge for
  \textbf{E}xemplar-Free \\ Few-Shot Incremental Point Cloud Learning}
```

Or without repeating the name in the title line (acronym only in abstract/body):

```latex
\title{\textbf{P}rototype-Augmented \textbf{Rid}ge for
  \textbf{E}xemplar-Free \\ Few-Shot Incremental Point Cloud Learning}
% Then define: We call this approach PRidE.
```

**First mention in abstract/intro:**

> We propose **PRidE** (**P**rototype-augmented **Rid**ge for **E**xemplar-free few-shot incremental point cloud learning), …

**Terminology in prose:** call the store a **prototype memory**; module nickname **KeyCache** OK in figures/code.

---

## 0. Quick answers

### Can closefar + mean be the main method?

**Yes.** Recommended main recipe:

| Setting | Prototype insert | Keys / novel class | Evolution |
|---------|------------------|--------------------|-----------|
| **In-domain** (same dataset all sessions) | **Close–far** (`mean_close_far`) | 3 = mean + closest shot + farthest shot | **Static** |
| **Cross-dataset** (base on A, increments on B) | **Mean only** | 1 | **Static** |

Shared everywhere:

- Frozen Uni3D
- Analytical **ridge** head
- **Ridge + cache fusion** (`classifier_mode=ridge_cache`, α≈0.85)
- Exemplar **buffer-free** is OK (conflict replay / cache proxies still on)
- Base-only adapter warmup (task-0) is enough

Show in **ablation** (not only main table): closefar helps **in-domain**; mean is better or equal on **cross-dataset**. Do **not** claim closefar alone explains transfer.

### In-domain or in-dataset?

Use **in-domain** vs **cross-dataset**.

| Term | Use? | Meaning |
|------|------|---------|
| **In-domain** | **Yes (preferred)** | All FSCIL sessions from one dataset (MN40 / SN55 / CO3D). Standard CL wording with “cross-domain.” |
| **Cross-dataset** | **Yes (preferred)** | Base classes from dataset A, novel sessions from dataset B (e.g. MN40→ScanObjectNN). More precise than “cross-domain” for your splits. |
| Cross-domain | OK as synonym | Fine in prose once; prefer “cross-dataset” in tables/captions. |
| In-dataset | **Avoid** | Uncommon; sounds like “within one split,” not “same distribution.” |

**Suggested paper phrasing:**

> We evaluate *in-domain* FSCIL (ModelNet40, ShapeNet55, CO3D) and *cross-dataset* FSCIL (ModelNet40/ShapeNet55 → ScanObjectNN or CO3D).

---

## Introduction (paper draft — write this section next)

Use this as the source for `_paper/_secs/01_introduction.tex`.  
The old LaTeX intro still sells **confusion-driven cache evolution** — replace it with the draft below.

### How the intro is structured (paragraph map)

| ¶ | Job | ~length |
|---|-----|---------|
| 1 | Motivation: 3D + FSCIL matter | 4–6 sentences |
| 2 | Gap: heavy fine-tuning / complex 3D FSCIL fails under few-shot | 4–6 sentences |
| 3 | Insight: frozen PTM + non-destructive updates; ridge alone is not enough | 4–6 sentences |
| 4 | Method preview: ridge + cache fusion; **close–far vs mean** | 5–8 sentences |
| 5 | Protocols + empirical teaser (keep numbers honest / TBD vs 3D-FLEG) | 3–5 sentences |
| 6 | Contributions (bullets) | 4 bullets |
| 7 | Roadmap | 2–3 sentences |

### Writing tips

- Lead with **stability–plasticity** and **domain-aware prototypes**, not evolution.
- Mention **Novel / Harmonic** early so reviewers know you care about plasticity, not only total Last.
- One sentence on **exemplar-free**: key cache ≠ raw point-cloud replay.
- End contributions with evaluation scope (in-domain + cross-dataset).
- Placeholders marked `[TBD]` must be filled from locked Table 1 before submission.

---

### Full Introduction draft (prose)

Three-dimensional point cloud understanding underpins robotics, autonomous driving, and AR/VR, yet the set of object categories is rarely fixed at deployment time.
New classes arrive continuously, while labelled 3D scans remain expensive to collect.
**Few-shot class-incremental learning (FSCIL)** formalises this challenge: after a data-rich base session, the model must absorb novel categories from only a handful of examples per class (typically five) while retaining accuracy on all previously seen classes.
In 3D the problem is especially hard: point clouds are unordered and sparse, and a single aggressive update to a high-capacity encoder can erase geometric structure learned from abundant base data.

Recent 3D FSCIL systems pursue accuracy with increasingly elaborate pipelines—geometric auxiliary branches, repeated adapter fine-tuning, and vision–language fusion.
These designs add capacity, but they also make **destructive gradient updates** the default way to incorporate new knowledge.
With only a few labelled shapes per novel class, fine-tuning a backbone or large auxiliary head tends to **overfit the few-shot batch** and **catastrophically forget** base-class structure.
Raw exemplar replay can soften forgetting, yet storing point clouds scales poorly and reintroduces optimisation instability at every session.
The practical bottleneck is therefore not only *what* to remember, but *how* to update **without corrupting frozen structural priors**.

We take a different view: with a strong pretrained 3D encoder, robust FSCIL primarily needs **stable, non-destructive decision boundaries**, not continual end-to-end fine-tuning.
Large-scale pretraining (e.g.\ Uni3D) already yields transferable embeddings; the incremental learner should **accumulate** discriminative structure analytically and maintain a **small episodic memory** of class prototypes.
Closed-form classifiers—in particular random-projection ridge regression in the spirit of RanPAC—update sufficient statistics without rewriting the backbone and without storing raw exemplars for inference.
However, **ridge alone is not enough** under few-shot increments: it favours base-class stability but under-represents the local geometry of novel classes.
Conversely, a pure prototype cache is plastic but can collapse base recall.
This is the classical **stability–plasticity** dilemma in an exemplar-free 3D setting.

We propose **PRidE** (**P**rototype-augmented **Rid**ge for **E**xemplar-free few-shot incremental point cloud learning), a framework that freezes Uni3D and couples two bounded modules:
**(i)** an analytical **ridge** head that maintains global linear structure in closed form, and
**(ii)** a capacity-limited **prototype key cache** that stores a few normalised embeddings per class.
At inference we **fuse** ridge and cache logits so that ridge anchors old classes while the cache supports novel ones.
The key design question—largely overlooked when caches are treated as a single mean vector or as “store every shot”—is **how to construct prototypes under a tight per-class budget**, and whether the same construction is optimal when novel sessions come from a **different dataset**.

Our answer is a **domain-aware insert policy**.
For **in-domain** FSCIL (all sessions from one dataset), we store three keys per novel class: the class **mean**, the shot **closest** to the mean, and the shot **farthest** from the mean (**close–far**).
The mean provides a stable centre; the close shot captures a typical mode; the far shot adds coverage of intra-class variation—recovering most of the benefit of storing all five shots while using only three keys.
For **cross-dataset** FSCIL (e.g.\ ModelNet base classes, ScanObjectNN or CO3D increments), individual shots mix class signal with **domain style**; multi-key memories can overfit that style.
There we prefer a single **mean** prototype as a shrinkage estimator.
Importantly, once fusion and this insert policy are fixed, **online cache evolution is unnecessary**: a static cache matches or exceeds evolving variants in our ablations.
The backbone never receives destructive incremental SGD; learning happens in ridge statistics and a handful of keys.

We evaluate both **in-domain** protocols (ModelNet40, ShapeNet55, CO3D) and **cross-dataset** transfers (ModelNet40/ShapeNet55 → ScanObjectNN; ShapeNet55 → CO3D) under the FSCIL-3D class order with frozen Uni3D-Base.
Beyond total final accuracy, we emphasise **novel-class accuracy** and the **harmonic mean** of base and novel performance at the last session—the metrics that expose plasticity failures hidden by base-dominated averages.
Relative to published 3D-FLEG numbers on matched in-domain splits, our recipe achieves competitive or improved final accuracy with a markedly simpler optimisation profile `[TBD: lock +X.X / +Y.Y Last and Δ vs 3D-FLEG in Table 1]`.
Ablations isolate three messages: ridge–cache **fusion** is required for stability–plasticity balance; **close–far** beats mean-only in-domain (especially on CO3D novel accuracy); **mean** is preferable on several cross-dataset transfers; and evolution adds no consistent gain over a static close–far cache.

\paragraph{Contributions.}
Our main contributions are:
\begin{itemize}
    \item A \textbf{replay-free, frozen-encoder} approach to 3D FSCIL that replaces destructive backbone fine-tuning with closed-form ridge classification and a bounded prototype cache, fused at inference.
    \item A \textbf{domain-aware prototype construction}: close–far keys (mean + nearest + farthest shot) for in-domain few-shot sessions under a three-key budget, and mean-only keys under cross-dataset shift where extra shots can encode domain noise.
    \item An empirical \textbf{stability–plasticity analysis} showing that ridge alone under-serves novel classes, cache alone forgets base classes, and fusion recovers both—across in-domain and cross-dataset FSCIL-3D protocols.
    \item A finding that \textbf{online cache evolution is unnecessary} given the above insert policy; we adopt a static cache and validate buffer-free and base-only adapter variants for a simple, reproducible recipe.
\end{itemize}

The remainder of the paper is organised as follows.
Section~\ref{sec:related} reviews continual learning, analytical classifiers, and 3D FSCIL.
Section~\ref{sec:method} formalises the problem and details the ridge head, key cache, fusion rule, and domain-aware inserts.
Section~\ref{sec:experiments} presents in-domain and cross-dataset experiments and ablations.
Section~\ref{sec:conclusion} concludes with limitations and future work.

---

### Shorter “elevator” intro (if space is tight)

Use this if the venue forces a compressed opening:

> 3D FSCIL requires learning new object categories from few point clouds without forgetting old ones. Fine-tuning pretrained encoders on five-shot increments overfits and erases base geometry. **PRidE** freezes Uni3D and fuses an analytical ridge classifier (stability) with a small prototype memory (plasticity). Prototype *construction* depends on the regime: **close–far** keys for in-domain sessions, and a single **mean** under cross-dataset shift. Evolution is unnecessary; the exemplar-free recipe is strong on ModelNet40, ShapeNet55, CO3D, and ScanObjectNN/CO3D transfers.

---

### Contribution bullets — plain English (for slide / rebuttal)

1. **Don’t fine-tune the 3D backbone on 5 shots** — use ridge + small cache instead.  
2. **Fuse them** — ridge keeps old classes; cache helps new ones.  
3. **Build the cache differently by setting** — close–far in-domain; mean when the dataset changes.  
4. **Keep it static** — no evolution needed; buffer-free is fine.

---

### Sentences you can reuse (claim → evidence)

| Claim sentence | Point to |
|----------------|----------|
| “Ridge favours stability; cache favours plasticity.” | Regime table (MN40 / ScanObj) |
| “Fusion is required for balance.” | Ridge vs cache vs static fusion |
| “Close–far recovers diversity under a 3-key budget.” | mean → close–far → 5-shot (in-domain Novel/Harm) |
| “Under dataset shift, mean is safer than multi-key shots.” | MN→ScanObj, SN→CO3D mean vs close–far |
| “Evolution is unnecessary for close–far.” | hybrid3_st vs hybrid4_evo (identical on 5/6 protocols) |

---

### What to delete from the old intro (`_paper/_secs/01_introduction.tex`)

| Old text | Action |
|----------|--------|
| “self-evolving prototype cache” / split–merge–evict as main | Remove from lead; optional ablation only |
| “confusion-driven cache evolution” contribution bullet | Replace with domain-aware insert |
| Session-adaptive α as a headline contribution | Optional detail; α=0.85 fixed is fine for main |
| CIL “within 0.3 of RanPAC” as a lead claim | Demote or drop (5-seed CIL RanPAC still ahead) |
| Warning cites / TBD percentages | Fill before camera-ready |

---

### Checklist before pasting into LaTeX

- [ ] Set `\ourmethod` / method name to **PRidE** everywhere in `_paper/`
- [ ] Add real citations (3D-FLEG, Uni3D, RanPAC, FSCIL-3D protocol, surveys)
- [ ] Lock vs 3D-FLEG Last / Δ numbers; remove `[TBD]`
- [ ] Keep **in-domain** / **cross-dataset** wording consistent with tables
- [ ] Ensure Method section defines close–far mathematically (match pipeline § below)
- [ ] Do **not** reintroduce evolution as contribution #2

---

## Methodological pipeline (end-to-end)

This is the **Method section blueprint**: what happens at session 0, what happens at each few-shot increment, and how inference works.  
Code path: `train_cil.py` → `lamba_pilot/trainer.py` → `models/selfevo.py` (+ `ridge_head.py`, `key_cache.py`).

### Big picture (one figure worth drawing)

```text
                    ┌─────────────────────────────────────────┐
  Point cloud X ──►│  Frozen Uni3D encoder f_θ  (optional     │
                    │  tiny task-0 adapter warmup only)       │
                    └─────────────────┬───────────────────────┘
                                      │ embedding z ∈ R^d
                    ┌─────────────────┴───────────────────────┐
                    │                                         │
                    ▼                                         ▼
           ┌────────────────┐                      ┌──────────────────┐
           │  RidgeHead     │                      │  KeyCache M       │
           │  stats (G, Q)  │                      │  few keys/class  │
           │  closed-form W │                      │  mean or close–far│
           └────────┬───────┘                      └────────┬─────────┘
                    │ ℓ_ridge                               │ ℓ_cache
                    └─────────────────┬─────────────────────┘
                                      ▼
                         ℓ = α ℓ_ridge + (1-α) ℓ_cache
                                      ▼
                                 predicted class
```

**Stability** comes from ridge (global linear structure).  
**Plasticity** comes from the cache (local prototype matching).  
**Domain-aware insert** decides *what* goes into the cache.

---

### Notation (for the paper)

| Symbol | Meaning |
|--------|---------|
| Session / task \(t = 0,\ldots,T-1\) | Base then few-shot increments |
| \(C_t\) | New class IDs at session \(t\) |
| \(n_{\text{shot}}=5\) | Novel examples per class (FSCIL) |
| \(z = f_\theta(X)\) | Frozen Uni3D embedding |
| \(\mathcal{M}\) | Key cache: list of (key, class label) |
| \((G, Q)\) | Ridge sufficient statistics |
| \(\alpha\) | Fusion weight on ridge (default ≈ 0.85) |

---

### Phase A — Session 0 (base classes, many samples)

1. **Data.** Full training set for base classes (e.g. 20 on MN40).
2. **Optional adapter warmup.** Short SGD only on a small adapter (task-0 / base-only). Backbone PTM weights stay frozen for the rest of FSCIL. *(Ablation: helps a bit; not the main novelty.)*
3. **Extract embeddings** for all base train shapes with `f_θ` in eval mode.
4. **Init KeyCache (mean).** For each base class, insert **one mean** prototype  
   \(\mu_c = \mathrm{mean}\{z : y=c\}\).  
   (Base is data-rich; one centre is enough.)
5. **Init RidgeHead.** Accumulate \((G, Q)\) from base embeddings/labels; fit ridge λ (grid / closed form). Produces linear classifier in random-projection space (RanPAC-style).
6. **Evaluate** on all classes seen so far (base only at \(t=0\)).
7. **Evolution.** For the paper method: **static** → skip. (Code may still call `evolve` with `evolution_policy=static` = no-op.)

After session 0 you have: frozen encoder, ridge fit on base, one key per base class.

---

### Phase B — Session \(t \ge 1\) (few-shot novel classes)

#### B1. Build the session data

- **Novel set:** 5 labelled point clouds per new class (FSCIL-3D protocol).
- **Old-class handling (paper default = buffer-free):**
  - Do **not** re-read raw old point clouds from disk.
  - Use **existing cache keys** as old-class embedding proxies when needed (conflict replay / adaptation).
- Optional legacy: 1-shot disk replay (`fscil_old_shot_replay=1`) — ablation only.

#### B2. Embed novel shots

Pass the 5-shot clouds through frozen Uni3D → embeddings \(\{z_i\}_{i=1}^{5}\) per new class.

#### B3. Insert into KeyCache (**domain-aware — main design**)

For each novel class \(c \in C_t\):

**If in-domain (close–far):**

\[
\mu = \tfrac{1}{5}\sum_i z_i,\quad
z_{\mathrm{close}} = \arg\max_i \cos(z_i,\mu),\quad
z_{\mathrm{far}} = \arg\min_{i\neq\mathrm{close}} \cos(z_i,\mu)
\]

Store keys \(\{\mu,\, z_{\mathrm{close}},\, z_{\mathrm{far}}\}\) (cap = 3).

**If cross-dataset (mean):**

Store only \(\{\mu\}\) (cap = 1).

Base-class keys already in \(\mathcal{M}\) are left unchanged (unless a refresh path is enabled).

#### B4. Update RidgeHead

- Add novel embeddings into ridge sufficient statistics \((G, Q)\) (often with higher weight on few-shot samples, e.g. `fscil_novel_ridge_weight`).
- Optionally refresh ridge using old-class **cache** proxies / conflict replay — still **no** growing raw buffer required for the paper claim.
- Refit / solve for ridge weights in closed form.

#### B5. Optional conflict replay (keep as engineering detail)

- Build a small adaptation batch (novel embeddings + old cache keys).
- Identify confused / low-recall classes; re-inject their embeddings into ridge with higher weight.
- Paper: mention briefly; ablate on/off (helps Novel).

#### B6. Evolution (paper method = OFF)

- `evolution_policy=static`: do not split/merge/grow.
- Ablation only: `rules` — show ≈ no gain for close–far.

#### B7. Evaluate session \(t\)

Test on **all classes so far** \(\cup_{k=0}^{t} C_k\). Log total / base / novel / harmonic.

---

### Phase C — Inference (every test forward)

For a test cloud \(X\):

1. \(z = f_\theta(X)\) (frozen).
2. \(\ell_{\mathrm{ridge}} = \mathrm{RidgeHead}(z)\) over all known classes.
3. \(\ell_{\mathrm{cache}} = \mathrm{KeyCache}(z)\) — similarity to stored keys, aggregated per class (e.g. max/mean over that class’s keys).
4. Fuse:
   \[
   \ell = \alpha\,\ell_{\mathrm{ridge}} + (1-\alpha)\,\ell_{\mathrm{cache}}
   \]
   with \(\alpha \approx 0.85\) (slightly lower α on increments is optional).
5. \(\hat y = \arg\max \ell\).

**Ablation modes (for paper tables):**

| `classifier_mode` | Behaviour |
|-------------------|-----------|
| `ridge` | Only \(\ell_{\mathrm{ridge}}\) — stable, less novel |
| `cache` | Only \(\ell_{\mathrm{cache}}\) — plastic, forgets base |
| `ridge_cache` | **Main** fusion |

---

### Per-session flowchart (copy into Method figure caption)

```text
for session t = 0 … T-1:
    get train data for new classes C_t
        (t=0: full; t>0: 5-shot; buffer-free: no raw old clouds)
    optional: task-0 adapter warmup          # only t=0 in main recipe
    z ← Uni3D(X)                             # frozen encoder
    insert keys into M:
        t=0        → mean per base class
        t>0 in-dom → close–far (3 keys)
        t>0 cross  → mean (1 key)
    update ridge stats (G,Q) from z; solve W
    optional: conflict replay from cache proxies
    evolution: OFF (static)
    evaluate on all classes seen so far
```

---

### What is stored in memory (efficiency talking point)

| Component | Size (order of magnitude) |
|-----------|---------------------------|
| Uni3D weights | Frozen; not updated after optional task-0 adapter |
| Ridge \((G,Q)\) + \(W\) | Fixed by `ridge_rp_dim` (e.g. 1000), grows with #classes only in output dim |
| KeyCache | \(\approx\) (#base)×1 + (#novel)×{3 in-domain \| 1 cross} normalised vectors |
| Raw point-cloud buffer | **0** in buffer-free main recipe |

Contrast with methods that store 10–20 raw clouds per class.

---

### Why each stage exists (one line each — for writing)

| Stage | Why |
|-------|-----|
| Freeze Uni3D | Avoid few-shot overfitting of the representation |
| Task-0 adapter (optional) | Mild alignment on abundant base data only |
| Ridge | Stable global classifier; closed-form; no exemplar SGD |
| Cache | Local few-shot support the linear head lacks |
| Fusion α | Trade stability vs plasticity without another train loop |
| Close–far | Budgeted diversity when shots match test domain |
| Mean (cross) | Shrink domain noise in shifted shots |
| Static | Evolution unnecessary once insert is right |
| Buffer-free | Keys/proxies replace raw replay |

---

### Mapping: paper name → code

| Paper | Code / config |
|-------|----------------|
| Frozen encoder | Uni3D via `utils/inc_net.py`; `requires_grad=False` after warmup |
| Ridge head | `models/ridge_head.py` |
| Key cache | `models/key_cache.py` |
| Close–far | `fscil_mean_close_far_prototypes=true` → `_mean_close_far_prototypes` |
| Mean insert | `insert_strategy=mean` / `mean_st` regime |
| Fusion | `classifier_mode=ridge_cache`, `ridge_cache_alpha` |
| Static | `evolution_policy=static` |
| Buffer-free | `model_name=selfevo_bufferfree` / preset `exemplar_buffer_free` |
| Session loop | `SelfEvo.incremental_train` → `_train` |

---

### Method section outline (suggested subsections)

1. **Problem setup** — FSCIL sessions, shots, in-domain vs cross-dataset  
2. **Frozen encoder** — Uni3D; optional base-only adapter  
3. **Analytical ridge classifier** — \((G,Q)\), update rule, classify  
4. **Prototype key cache** — classify by similarity; budget  
5. **Domain-aware prototype construction** — close–far vs mean (equations)  
6. **Logit fusion** — \(\alpha\)  
7. **Training algorithm** — Algorithm 1 pseudocode (Phase A/B/C above)  
8. **Complexity / memory** — table of stored tensors  

Skip a long “evolution operators” subsection in the main text; one paragraph + appendix ablation is enough.

---

## 1. Strongest ICLR story (commit to this)

### One-sentence pitch

> Exemplar-free 3D FSCIL with a frozen PTM: fuse closed-form ridge (stability) with a small prototype cache (plasticity), and build that cache with **close–far keys in-domain** and a **single mean under dataset shift**.

### Why this is stronger than “SelfEvo / evolution”

| Old pitch | Problem | New pitch |
|-----------|---------|-----------|
| Cache self-evolves | Static ≈ evolve for closefar; weak novelty | **Static** cache; evolution = negative ablation |
| One insert everywhere | Mean wins some transfers; closefar mid on others | **Domain-aware insert** (insight + method) |
| Beat everything | CIL RanPAC still ahead | Own the **FSCIL-3D + cross-dataset + analysis** lane |

### Contribution bullets (copy into intro)

1. **Stability–plasticity fusion for 3D FSCIL.** Analytical ridge preserves base classes; a bounded prototype cache restores novel accuracy; fusion combines both without backbone SGD on increments.
2. **Domain-aware compact prototypes.** Under a per-class key budget, **close–far** (mean + nearest + farthest shot) improves in-domain Novel/Harm vs mean-only; under **cross-dataset** shift, multi-key shots can encode domain noise and a **single mean** is often safer.
3. **Exemplar-free, evolution-free recipe.** No raw exemplar replay required; online cache evolution is unnecessary once insert + fusion are fixed.
4. **Broad protocol.** In-domain MN40 / SN55 / CO3D and cross-dataset transfers to ScanObjectNN / CO3D under frozen Uni3D.

### Naming

| In code | In paper |
|---------|----------|
| `selfevo`, `ridge_cache` | **PRidE** (Prototype-augmented Ridge) |
| `hybrid3_st`, `mean_close_far` | **Close–far** prototypes (static) |
| `mean_st` | **Mean** prototypes (static) |
| `evolution_policy=rules` | Ablation only: “online evolution” |
| Do **not** lead with SelfEvo / RL / bandit | Demote to appendix or drop |

Update `_paper/_secs/00_abstract.tex`: it still says the memory “evolves dynamically” — rewrite to static close–far / mean.

---

## 2. Method (what to write in §Method)

### Pipeline (one session)

1. Freeze Uni3D; extract embeddings for new-class few-shots (and base at session 0).
2. Update **RidgeHead** sufficient statistics (RanPAC-style analytical ridge).
3. Insert class keys into **KeyCache** (see insert policies below).
4. Optional: conflict replay from **cache embeddings** (not disk exemplars).
5. Classify with fusion:
   \[
   \ell = \alpha\,\ell_{\mathrm{ridge}} + (1-\alpha)\,\ell_{\mathrm{cache}},\quad \alpha \approx 0.85.
   \]

### Close–far insert (in-domain main)

For each novel class with shot embeddings \(\{e_i\}_{i=1}^{n}\) (typically \(n=5\)):

1. \(\mu = \mathrm{mean}_i e_i\)
2. \(e_{\mathrm{close}} = \arg\max_i \cos(e_i, \mu)\)
3. \(e_{\mathrm{far}} = \arg\min_{i \neq \mathrm{close}} \cos(e_i, \mu)\)
4. Store keys \(\{\mu,\, e_{\mathrm{close}},\, e_{\mathrm{far}}\}\) (cap = 3)

**Intuition (write this):** mean = class center (stability); close = typical mode; far = coverage of variation — diversity under a **3-key budget** without storing all shots.

Code: `models/key_cache.py` → `_mean_close_far_prototypes`; flag `fscil_mean_close_far_prototypes=true`.

### Mean insert (cross-dataset main)

Store only \(\mu\). Shrinkage cancels shot-level domain residuals.

Code: `insert_strategy=mean` / regime `mean_st`.

### What is *not* main

| Component | Role in paper |
|-----------|----------------|
| Evolution (`rules`) | Ablation: ≈ static for close–far |
| All-5-shot (`pshot_st`) | Upper bound / memory comparison |
| FPS-5 (`fps5_st`) | Redundant when \(k=n=5\) (same 5 points, reordered) |
| Disk exemplar buffer | Ablation: ≈ buffer-free |
| Full per-task adapter | Ablation: base-only ≈ full in-domain |

---

## 3. Explanation: why close–far in-domain, mean cross-dataset

**Write this in Discussion / Method intuition (plain language):**

**In-domain.** Few-shot shapes come from the same dataset as the test set. Shots are noisy samples of the *true* class manifold. One mean underfits pose/part variation; close + far add a typical sample and a diverse sample → higher Novel and Harmonic accuracy with only 3 keys.

**Cross-dataset.** Few-shots come from a different sensor/dataset (e.g. ScanObjectNN after ModelNet). Each shot mixes **class signal + domain style**. Saving many shots (or a far shot) can memorize domain-specific directions that do **not** match target-domain test features. Averaging shrinks that noise → mean often matches or beats multi-key caches on Novel/Harm.

**Do not say:** “Close–far is designed for transfer.”  
**Do say:** “Fusion + frozen PTM carry transfer; under shift, **simpler** prototypes are preferable.”

---

## 4. Metrics (how to report)

| Symbol | Meaning | Paper use |
|--------|---------|-----------|
| **AA** | Mean of per-session total top-1 | Secondary |
| **Last** | Final-session total top-1 | Main “accuracy” |
| **Base@Last** | Accuracy on session-0 classes at end | Stability |
| **Novel@Last** | Accuracy on all novel classes at end | Plasticity — **emphasize** |
| **Harm@Last** | Harmonic mean of base & novel at end | Balance — **emphasize** |
| **Δ** | Session-0 total − Last | Forgetting (optional) |

Total Last can be dominated by many base classes; **Novel + Harm** sell the plasticity story.

---

## 5. Numbers for tables (seed 2008)

### 5.1 Stability–plasticity (regime matrix)

**Source:** `_experiments/00_SIMPL_REGIME_ABLATIONS/`  
**Note:** Older full-adapter regime runs (not EP5 base-only final). Still the cleanest fusion story figure.

#### ModelNet40 in-domain (5 sessions)

| Variant | AA | Last | Base@Last | Novel@Last | Harm@Last |
|---------|---:|-----:|----------:|-----------:|----------:|
| Ridge only | 93.32 | 91.13 | 94.48 | 78.24 | 85.60 |
| Cache only | 89.91 | 86.43 | 86.62 | 85.69 | 86.15 |
| Ridge+cache (static) | 93.46 | 91.09 | 94.08 | 79.61 | 86.24 |
| Ridge+cache+evolution | 93.63 | 91.41 | 94.23 | 80.59 | 86.88 |

#### ShapeNet55 in-domain (7 sessions)

| Variant | AA | Last | Base@Last | Novel@Last | Harm@Last |
|---------|---:|-----:|----------:|-----------:|----------:|
| Ridge only | 89.63 | 88.30 | 89.87 | 80.85 | 85.12 |
| Cache only | 86.42 | 83.66 | 83.74 | 83.25 | 83.49 |
| Ridge+cache (static) | 89.84 | 88.54 | 89.91 | 82.05 | 85.80 |
| Ridge+cache+evolution | 89.88 | 88.53 | 89.82 | 82.43 | 85.97 |

#### MN40 → ScanObjectNN

| Variant | AA | Last | Base@Last | Novel@Last | Harm@Last |
|---------|---:|-----:|----------:|-----------:|----------:|
| Ridge only | 90.29 | 86.71 | 94.32 | 62.74 | 75.36 |
| Cache only | 87.62 | 84.07 | 88.84 | 69.05 | 77.70 |
| Ridge+cache (static) | 91.42 | 88.43 | 93.92 | 71.16 | 80.97 |
| Ridge+cache+evolution | 91.43 | 88.48 | 93.98 | 71.16 | 80.99 |

#### SN55 → ScanObjectNN

| Variant | AA | Last | Base@Last | Novel@Last | Harm@Last |
|---------|---:|-----:|----------:|-----------:|----------:|
| Ridge only | 90.67 | 87.81 | 93.18 | 33.05 | 48.79 |
| Cache only | 85.16 | 80.10 | 82.96 | 50.95 | 63.13 |
| Ridge+cache (static) | 91.20 | 88.35 | 91.39 | 57.31 | 70.44 |
| Ridge+cache+evolution | 91.14 | 88.24 | 91.27 | 57.31 | 70.41 |

**Talking points:**

- Ridge = high Base, lower Novel; Cache = high Novel, Base collapse.
- Fusion recovers both; biggest Novel jump on **cross-dataset** (esp. SN→ScanObj: 33→57 Novel).
- Evolution adds little beyond static fusion on ScanObj; small Novel gain on MN40.

---

### 5.2 Main insert ablation — mean vs close–far vs 5-shot

**Setting:** buffer-free, EP5 base-only adapter.  
**Sources:**  
- mean / pshot: `_experiments/00_FINAL_EP5_BASEONLY/`  
- close–far static: `_experiments/00_FINAL_EP5_HYBRID_CLOSEFAR/` (`hybrid3_st`)

| Protocol | Type | mean Last / Novel / Harm | **close–far** Last / Novel / Harm | 5-shot Last / Novel / Harm |
|----------|------|--------------------------:|----------------------------------:|---------------------------:|
| MN40 | in-domain | 91.00 / 76.47 / 84.65 | **91.21 / 78.43 / 85.73** | 91.37 / 79.41 / 86.29 |
| SN55 | in-domain | 88.74 / 76.00 / 83.00 | **88.90 / 77.74 / 83.95** | 88.85 / 77.85 / 83.99 |
| CO3D | in-domain | 64.84 / 24.08 / 36.73 | **66.05 / 28.99 / 42.19** | 66.22 / 30.71 / 43.93 |
| MN→ScanObj | cross-dataset | **87.16 / 69.47 / 79.45** | 87.01 / 69.05 / 79.15 | 86.61 / 67.37 / 78.03 |
| SN→ScanObj | cross-dataset | 90.62 / 64.54 / 76.26 | 90.56 / 64.03 / 75.90 | **90.68 / 65.23 / 76.74** |
| SN→CO3D | cross-dataset | **82.16 / 45.55 / 60.86** | 81.83 / 43.88 / 59.36 | 81.77 / 43.82 / 59.29 |

**How to narrate (honest):**

| Protocol | Prefer in main recipe | Why |
|----------|----------------------|-----|
| MN40, SN55, CO3D | **Close–far** | Large Novel/Harm lift vs mean; close to 5-shot with **3 vs 5** keys |
| MN→ScanObj, SN→CO3D | **Mean** | Best Novel/Harm |
| SN→ScanObj | Mixed | 5-shot slightly best; mean ≈ close–far — say “competitive; mean sufficient” |

**In-domain Δ (close–far − mean), Novel / Harm:**

| Protocol | ΔNovel | ΔHarm |
|----------|-------:|------:|
| MN40 | +1.96 | +1.08 |
| SN55 | +1.74 | +0.95 |
| CO3D | **+4.91** | **+5.46** |

**Cross-dataset Δ (mean − close–far), Novel / Harm:**

| Protocol | ΔNovel | ΔHarm |
|----------|-------:|------:|
| MN→ScanObj | +0.42 | +0.30 |
| SN→CO3D | +1.67 | +1.50 |
| SN→ScanObj | +0.51 | +0.36 |

---

### 5.3 Evolution unnecessary for close–far

**Source:** `_experiments/00_FINAL_EP5_HYBRID_CLOSEFAR/` (buffree)

| Protocol | static Last / Novel / Harm | evolve Last / Novel / Harm |
|----------|---------------------------:|---------------------------:|
| MN40 | 91.21 / 78.43 / 85.73 | 91.21 / 78.43 / 85.73 |
| SN55 | 88.90 / 77.74 / 83.95 | 88.85 / 77.36 / 83.74 |
| CO3D | 66.05 / 28.99 / 42.19 | 66.05 / 28.99 / 42.19 |
| MN→ScanObj | 87.01 / 69.05 / 79.15 | 87.01 / 69.05 / 79.15 |
| SN→ScanObj | 90.56 / 64.03 / 75.90 | 90.56 / 64.03 / 75.90 |
| SN→CO3D | 81.83 / 43.88 / 59.36 | 81.83 / 43.88 / 59.36 |

**Claim:** With close–far insert, online evolution does not improve Last/Novel/Harm (SN55 slightly worse). Prefer **static**.

---

### 5.4 Suggested “main results” row (domain-aware recipe)

Report one line per protocol using the policy:

| Protocol | Insert used | Last | Novel@Last | Harm@Last |
|----------|-------------|-----:|-----------:|----------:|
| MN40 | close–far | 91.21 | 78.43 | 85.73 |
| SN55 | close–far | 88.90 | 77.74 | 83.95 |
| CO3D | close–far | 66.05 | 28.99 | 42.19 |
| MN→ScanObj | mean | 87.16 | 69.47 | 79.45 |
| SN→ScanObj | mean* | 90.62 | 64.54 | 76.26 |
| SN→CO3D | mean | 82.16 | 45.55 | 60.86 |

\*Or report 5-shot as oracle upper bound in ablation; main recipe can still use mean for consistency of the “cross-dataset → mean” rule.

**Caption idea:** “We use close–far prototypes for in-domain sessions and mean prototypes for cross-dataset increments (buffer-free, static cache, ridge–cache fusion).”

---

## 6. Paper structure (section checklist)

### Abstract (rewrite)

- Problem: 3D FSCIL; heavy fine-tuning overfits few-shots / forgets base.
- Method: frozen Uni3D + ridge + compact cache; **close–far in-domain, mean cross-dataset**; **no** evolution required.
- Evidence: fusion ablations; insert ablations; buffer-free; multi-protocol.
- Avoid promising “xx%” until Table 1 vs 3D-FLEG is locked.

### Intro

1. Hook: novel categories arrive with few 3D examples.
2. Gap: PTM fine-tuning fragile; prior 3D FSCIL heavy; exemplar replay undesirable.
3. Insight: need **stable** linear boundaries + **small** episodic memory; memory **construction** depends on domain match.
4. Contributions (4 bullets above).
5. Teaser figure: ridge vs cache vs fusion; or keys-vs-Novel curve.

### Related work

- 2D FSCIL / analytic classifiers (e.g. RanPAC-style ridge)
- 3D FSCIL (3D-FLEG, etc.) — fair protocol footnotes
- Prototype / cache methods; exemplar-free CL
- Cross-dataset continual learning

### Method

- Setup notation (sessions, shots, base/novel)
- Ridge head
- KeyCache + fusion
- Close–far and mean inserts + **when to use which**
- Buffer-free / conflict replay one paragraph
- Complexity / memory (#keys)

### Experiments

1. Setup & metrics & protocols (in-domain vs cross-dataset table)
2. Main results (domain-aware recipe + baselines)
3. Ablation A: ridge / cache / fusion
4. Ablation B: mean → nearmean → close–far → 5-shot (keys vs Novel/Harm)
5. Ablation C: in-domain vs cross-dataset insert preference (the insight table)
6. Ablation D: static vs evolve
7. Ablation E: buffer-free; adapter base-only
8. Limitations: CO3D still hard; single-seed primary; CIL RanPAC not claimed SOTA

### Conclusion

Restate fusion + domain-aware compact memory; evolution unnecessary.

---

## 7. Figures that sell acceptance

| Fig | Content |
|-----|---------|
| **Fig 1 (teaser)** | Stability–plasticity: Base vs Novel for ridge / cache / fusion (MN40 + one cross-dataset) |
| **Fig 2** | Method diagram: Uni3D → ridge + cache; close–far vs mean switch |
| **Fig 3** | Keys/class (1→2→3→5) vs Novel@Last, facets: MN40, CO3D, MN→ScanObj |
| **Fig 4** | Optional: qualitative — far key hurts under ScanObj shift (embedding sketch) |

Fig 3 is the “aha” figure for close–far vs mean.

---

## 8. Reviewer FAQ (pre-write answers)

**Q: Isn’t this just RanPAC?**  
A: Ridge alone under-serves Novel (regime table). The cache + fusion are required for plasticity; insert policy is an additional FSCIL-specific design under a key budget.

**Q: Why close–far rather than all shots?**  
A: 3 keys recover most of the in-domain Novel/Harm gain of 5-shot storage (MN40/SN55/CO3D). Memory–accuracy tradeoff, not raw max accuracy.

**Q: Why mean on cross-dataset?**  
A: Shots carry domain residuals; multi-key caches can overfit domain style. Mean is shrinkage. Supported by MN→ScanObj and SN→CO3D Novel/Harm.

**Q: Where is evolution / SelfEvo?**  
A: We evaluate online evolution; with close–far it matches or underperforms static. We adopt the simpler static method.

**Q: CO3D numbers are low.**  
A: Hardest residual setting; mean→close–far still recovers ~+5 Novel / +5 Harm — diagnosis (need diversity in-domain) holds; absolute SOTA left as future work.

**Q: Single seed?**  
A: Primary seed 2008; multi-seed is a clear revision / camera-ready item. Do not overclaim variance-free SOTA.

**Q: CIL ModelNet vs RanPAC?**  
A: Separate protocol; 5-seed CIL RanPAC still ahead on Last. Paper focus is **FSCIL-3D**, not CIL SOTA.

---

## 9. What not to claim

- Evolution is the key contribution  
- Close–far is best on every protocol  
- FPS-5 is a distinct algorithmic win when \(k=n=5\)  
- Buffer is essential (it is not, under correct buffer-free)  
- Beat RanPAC on CIL 5-seed mean Last  
- Mix Uni3D rows with PointNet++ paper rows without footnotes  

---

## 10. Log / script map (reproducibility)

| Claim | Logs | Summarize |
|-------|------|-----------|
| Fusion story | `00_SIMPL_REGIME_ABLATIONS` | `scripts/summarize_fscil_regime_ablations.py` |
| Close–far / nearmean / fps | `00_FINAL_EP5_HYBRID_CLOSEFAR` | `scripts/summarize_fscil_final_campaign.py --campaign ep5_hybrid` |
| Mean / pshot / hybrid4 | `00_FINAL_EP5_BASEONLY` | `scripts/summarize_fscil_final_campaign.py --campaign ep5` |
| EP10 sensitivity | `00_FINAL_EP10_BASEONLY` | `--campaign ep10` |
| Buffer campaigns | `00_BUF_CAMPAIGN_EP5*` | (legacy test-aware replay — footnote if used) |

Configs: `configs/selfevo_fscil_*.json`  
Insert code: `models/key_cache.py`  
Learner: `models/selfevo.py`  
Campaign launchers: `scripts/simpl_runs_final_*.sh`

---

## 11. Open items before submission

| Item | Priority | Notes |
|------|----------|-------|
| Lock Table 1 vs **3D-FLEG** under identical splits | **High** | Fill abstract % gap |
| Multi-seed (3–5) on main recipe | **High** | Reviewer variance |
| Align `_paper/*.tex` with this story (drop evolution lead) | **High** | Abstract outdated |
| Optional: automatic domain-shift detector (vs oracle in-domain/cross rule) | Medium | Strengthens “adaptive” claim |
| `print_forget=true` forgetting table | Medium | |
| CO3D absolute gains (EP10 helps ~+3 Last) | Medium | Report as sensitivity |
| Fix failed `00_SCANOBJ_ABL_ADAPTER` if claiming increment warmup | Low | |

---

## 12. Abstract draft (drop-in, edit numbers later)

> Few-shot class-incremental learning (FSCIL) on 3D point clouds is challenging: updating a deep backbone on a handful of novel shapes overfits easily and erodes base-class structure. We argue that robust 3D FSCIL with a strong pretrained encoder does not require continual backbone SGD. We propose **PRidE**, an exemplar-free framework that freezes Uni3D, fits a closed-form ridge classifier for stable decision boundaries, and maintains a small prototype memory for plasticity. Predictions fuse ridge and memory scores. Critically, we show that *how* prototypes are built should depend on the evaluation regime: for **in-domain** sessions we store three keys per class—the class mean, the nearest shot, and the farthest shot (**close–far**)—recovering most of the benefit of storing all five shots under a tight memory budget; for **cross-dataset** increments, a single **mean** prototype is often preferable because extra shot keys can encode domain-specific noise. Online memory evolution is unnecessary once this insert policy and fusion are fixed. Experiments on ModelNet40, ShapeNet55, CO3D, and transfers to ScanObjectNN/CO3D support the stability–plasticity analysis and the domain-aware prototype design.

---

## 13. Terminology cheat-sheet for writing

| Write | Don’t write |
|-------|-------------|
| in-domain FSCIL | in-dataset FSCIL |
| cross-dataset FSCIL / transfer | casually “cross-domain” only in tables |
| close–far prototypes | hybrid3_st (code name → footnote only) |
| mean prototypes | mean_st |
| static cache | “no evolution” as a positive simplicity claim |
| Novel@Last, Harm@Last | only total Last |
| buffer-free / exemplar-free | “no memory” (you still have a key cache) |

---

*End of PAPER.md — sync numbers from summarize scripts if campaigns are re-run.*
