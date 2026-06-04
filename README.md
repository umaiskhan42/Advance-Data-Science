# Zero-Shot Microfauna Anomaly Detection with Frozen DINOv3 on MD-AS-2025

**Umais Khan · 2025311037 · Dept. of Environmental Science & Engineering**
Supervisor: Prof. Won Hee Lee

Implementation of the project proposal: use a **frozen DINOv3 ViT** as a feature
extractor and a **PatchCore** memory bank of *healthy* microfauna to flag
*stressed* microfauna as anomalies — with **no task supervision and no fine
tuning**. Stressed taxa (Arcellinida, Nematoda) indicate the unfavourable end of
the Sludge Biotic Index and are never seen during "training".

> **This is an extension of our lab's (EMSEL) existing work, not new research.**
> EMSEL's anomaly-detection lineage — "model the normal, flag the deviation" —
> runs from KPCA/ICA process monitoring (2004) → deep variational-residual
> autoencoders for WWTP fault detection (2022) → **memory-augmented autoencoders
> with learned normal prototypes (2025)** → unseen-fault detection (2026). This
> project takes that same recipe and changes only two things: the **modality**
> (sensor time-series → activated-sludge microscopy images) and the **feature
> source** (trained autoencoder → a *frozen* self-supervised vision foundation
> model). See "Relationship to prior lab work" below and
> [`Literature/CLOSEST_LAB_WORK.md`](Literature/CLOSEST_LAB_WORK.md).

---

## Idea in one paragraph

Sludge degradation in activated-sludge wastewater treatment shows up first as a
shift in the microfauna community (healthy ciliates/rotifers → stressed
nematodes and testate amoebae), but manual microscopy is slow and labelled
degradation events are rare. We therefore treat degradation as an **anomaly
detection** problem: build a memory bank of patch embeddings from *healthy*
taxa using a frozen DINOv3 backbone, then score each test image by how far its
patches sit from that healthy manifold. High distance ⇒ likely stressed.

## Relationship to prior lab work (this project = an extension)

This project is positioned as the **next step in EMSEL's anomaly-detection
research**, not a separate new line. The mapping is one-to-one:

| EMSEL prior work | What it did | This project (the extension) |
|---|---|---|
| KPCA / ICA process monitoring (2004) | model the normal operating region in a feature space, flag departures | normal region = healthy patch-feature manifold |
| Deep variational residual autoencoder, *Chemosphere* 2022 | learn the normal manifold of **WWTP** sensors, detect/reconstruct faults | same WWTP target, but the "sludge health" signal read from **images** |
| **Memory-augmented autoencoder with learned *normal prototypes*, KJChE 2025** | **memorise normal prototypes; score deviation from them** | **PatchCore = memory bank of healthy patch prototypes; score NN deviation** |
| Transfer-learning *unseen*-fault detection, 2026 | detect categories never seen in training | stressed taxa never placed in the memory bank (zero-shot) |

Only two things change versus the lab's 2025 memory-prototype paper:
**(1) modality** — sensor time-series → activated-sludge microscopy images;
**(2) feature source** — a trained autoencoder → a *frozen* self-supervised
foundation model (DINOv3), so the normal model needs no training or labels.
The contribution is therefore *porting the lab's proven recipe to vision*, in the
same activated-sludge / WWTP-monitoring problem the lab already studies.

## Method (PatchCore + DINOv3)

1. **Backbone** — DINOv3 ViT-S/16 (`timm: vit_small_patch16_dinov3.lvd1689m`),
   frozen, no gradient updates.
2. **Feature extraction** — patch-token grid (14×14×384 at 224 px) + CLS token;
   3×3 local neighbourhood aggregation; L2 normalisation.
3. **Memory bank** — healthy-class patch embeddings, reservoir-pooled then
   reduced with a **greedy k-center coreset** (random-projected for speed).
4. **Anomaly scoring** — each test patch → nearest-neighbour (cosine) distance to
   the bank; image score = max patch distance.
5. **Decision threshold** — unsupervised (95th percentile of healthy scores) and
   Youden-J operating points.
6. **Per-taxon validation** — healthy-vs-Arcellinida and healthy-vs-Nematoda
   AUROC reported separately (per-failure-mode).

**Baselines for H1:** identical PatchCore pipeline with (a) a *random-init* ViT
(no pretraining) and (b) an *ImageNet ResNet-50* (the original PatchCore CNN
backbone), plus a DINOv3 CLS-kNN image-level reference.

---

# Getting started

Everything below assumes a terminal opened at the **repository root** (the folder
that contains this `README.md`).

## 1. Clone the repository

```bash
git clone https://github.com/umaiskhan42/Advance-Data-Science.git
cd Advance-Data-Science
```

## 2. Create an environment and install dependencies

Python **3.10–3.13** is supported. A GPU is *optional* — the anomaly-detection
pipeline is CPU-only by default.

```bash
# create + activate a virtual environment
python -m venv .venv

# Windows (PowerShell)
.venv\Scripts\Activate.ps1
# macOS / Linux
source .venv/bin/activate

# install everything
pip install --upgrade pip
pip install -r requirements.txt
```

> The DINOv3 and ResNet-50 backbones download automatically from the **ungated
> timm / Hugging Face mirror** on first run — no token or login required. No
> model weights are stored in this repo (`weights/` is created locally and is
> git-ignored).

## 3. Download the dataset (MD-AS-2025)

The raw images (~4 GB) are **not** in this repo. Download the cropped
single-object images for each taxon from Zenodo and extract them into
`data/raw/` so each taxon has its own folder:

```
data/raw/
├── Hypotrichida/     # healthy   — Zenodo 10.5281/zenodo.15099197
├── Digononta/        # healthy   — Zenodo 10.5281/zenodo.17140129
├── Monogononta/      # healthy   — Zenodo 10.5281/zenodo.17140144
├── Arcellinida/      # stressed  — Zenodo 10.5281/zenodo.17140068
├── Nematoda/         # stressed  — Zenodo 10.5281/zenodo.17142917
└── annotated/        # boxes     — Zenodo 10.5281/zenodo.15210103   (object-detection only)
    └── train_val_test/
        ├── images/
        └── labels/   # PASCAL-VOC XML
```

Folder **names must match exactly** (they are read by `scripts/config.py`).
Dataset paper: *Scientific Data* (2025), `10.1038/s41597-025-06228-6`. See
[`Literature/DATASET_USAGE.md`](Literature/DATASET_USAGE.md) for details. The
`annotated/` set is only needed for the optional object-detection scripts; the
core anomaly pipeline only needs the five taxon folders.

| Role | Taxa | Label |
|------|------|-------|
| Healthy (normal) → memory bank + healthy test | Hypotrichida, Digononta, Monogononta | 0 |
| Stressed (anomaly) → test only | Arcellinida, Nematoda | 1 |

## 4. Run the pipeline

### Option A — everything at once

```bash
python scripts/run_all.py
```

This runs stages 1→3 (preprocess → PatchCore → visualise) and writes all CSVs
and figures under `results/`.

### Option B — stage by stage (anomaly detection, the core experiment)

```bash
python scripts/01_preprocess.py   # build manifest + dataset summary
python scripts/02_patchcore.py    # memory bank, scoring, metrics (~10 min CPU)
python scripts/03_visualize.py    # t-SNE/UMAP, attention, heatmaps, failure cases
```

### Option C — optional object-detection baseline (frozen-DINOv3 detector vs YOLOv8)

Requires the `annotated/` set from step 3.

```bash
python scripts/od_01_prepare.py         # VOC XML -> YOLO labels + data.yaml
python scripts/od_02_train_eval.py      # train/eval YOLOv8 reproduction (GPU recommended)
python scripts/od_03_dinov3_detector.py # frozen-DINOv3 detection head + per-class AP
```

### Option D — downstream WWTP-impact simulation & figures

```bash
python scripts/wwtp_impact_sim.py      # anomaly detector as a WWTP component (with vs without)
python scripts/paper_style_figures.py  # paper-style comparison figures
python scripts/research_figures.py     # research-report figure panels
```

> `scripts/make_proposal_docx.py` regenerates the Word proposal and needs
> `python-docx`; it is not part of the experiment pipeline.

### Knobs

Hyper-parameters (image size, coreset size, per-taxon sampling caps, seed,
backbones, …) all live in [`scripts/config.py`](scripts/config.py).
`scripts/od_02_train_eval.py` also reads optional env overrides:

```bash
# macOS / Linux
OD_MODEL=yolov8n.pt OD_IMGSZ=640 OD_EPOCHS=40 python scripts/od_02_train_eval.py
# Windows PowerShell
$env:OD_EPOCHS=40; python scripts/od_02_train_eval.py
```

## 5. Where the outputs land

Running the scripts creates these locally. Large/binary outputs are git-ignored;
the CSV tables and run logs that back the reported numbers are kept in the repo.

```
results/csv/              manifest, per-image scores, metrics, ROC, per-taxon, OD AP, WWTP sim   (tracked)
results/*.log             stage run logs                                                          (tracked)
results/figures/          AUROC bars, ROC, score dists, per-taxon, OD vs paper, WWTP   (generated)
results/tsne/             t-SNE / UMAP plots + coordinates                              (generated)
results/attention/        DINOv3 attention overlays                                     (generated)
results/heatmaps/         PatchCore anomaly heatmaps                                     (generated)
results/failure_cases/    worst FP / FN montages + CSV                                  (generated)
data/processed/           memory banks, CLS embeddings, heatmap cache, OD features      (generated)
```

## Repository layout

```
scripts/        config.py, common.py, 01–03 (anomaly), run_all.py,
                od_01..03 (detection), wwtp_impact_sim.py, *_figures.py
data/raw/       (you download) extracted MD-AS-2025 taxa folders + annotated set
data/processed/ (generated)    memory banks, CLS embeddings, caches, OD features
results/        CSVs + logs (tracked); figures/heatmaps/etc (generated)
Literature/     REFERENCES.md, CLOSEST_LAB_WORK.md, DATASET_USAGE.md (PDFs not tracked)
```

## Reports

| File | Contents |
|---|---|
| [RESULTS.md](RESULTS.md) | Anomaly detection — H1–H4, DINOv3 vs baselines, per-taxon |
| [RESULTS_DETECTION.md](RESULTS_DETECTION.md) | Object detection — frozen-DINOv3 detector vs the dataset paper |
| [RESULTS_WWTP_IMPACT.md](RESULTS_WWTP_IMPACT.md) | Anomaly detector as a WWTP component — with vs without |
| [POSITIONING.md](POSITIONING.md) | Copy-paste framing: this project as an extension of EMSEL's work |
| [Literature/CLOSEST_LAB_WORK.md](Literature/CLOSEST_LAB_WORK.md) | Closest prior work from the lab |
| [Literature/DATASET_USAGE.md](Literature/DATASET_USAGE.md) | Papers that have used MD-AS-2025 |

## Hypotheses (from the proposal)

- **H1** DINOv3 beats random-init and CNN baselines on AUROC (both stressed taxa).
- **H2** Attention maps highlight biologically meaningful regions.
- **H3** t-SNE shows clean healthy/stressed separation in DINOv3 feature space.
- **H4** SSL vision models are a viable low-cost front-end for WWTP monitoring.

See [RESULTS.md](RESULTS.md) for the measured outcomes.

## Troubleshooting

- **`FileNotFoundError` / empty manifest** — `data/raw/` is missing or the taxon
  folders are misnamed. Folder names must match step 3 exactly.
- **Backbone download is slow on first run** — DINOv3/ResNet weights are fetched
  once from the timm HF mirror and cached under `~/.cache/huggingface`.
- **`ModuleNotFoundError: ultralytics` / `umap`** — only the optional detection
  (Option C) and UMAP plots need these; `pip install -r requirements.txt`
  installs them. t-SNE runs fine without `umap-learn`.
- **Out of memory / too slow** — lower `CORESET_SIZE`, `POOL_MAX`, or the
  `MAX_*_PER_TAXON` sampling caps in [`scripts/config.py`](scripts/config.py).
