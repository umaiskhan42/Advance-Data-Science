# Literature — Zero-Shot Microfauna Anomaly Detection with Frozen DINOv3 on MD-AS-2025

This folder collects the papers behind the project plus the publications from
**Prof. ChangKyoo Yoo's lab (EMSEL, Kyung Hee University,
[emsel.khu.ac.kr](https://emsel.khu.ac.kr); Google Scholar
[KTKMPW0AAAAJ](https://scholar.google.com/citations?user=KTKMPW0AAAAJ&hl=en))**
that line up with it methodologically and by application domain.

---

## A. Foundational / necessary papers  (PDFs downloaded into this folder)

| File | Paper | Why it matters here |
|------|-------|---------------------|
| `DINOv3_2025_Simeoni_arXiv2508.10104.pdf` | Siméoni et al., **DINOv3**, arXiv:2508.10104 (2025) | The frozen self-supervised ViT backbone used as the feature extractor. |
| `DINOv2_2023_Oquab_arXiv2304.07193.pdf` | Oquab et al., **DINOv2: Learning Robust Visual Features without Supervision**, arXiv:2304.07193 (2023) | Predecessor SSL ViT; establishes the dense-feature paradigm. |
| `DINO_2021_Caron_EmergingProperties_arXiv2104.14294.pdf` | Caron et al., **Emerging Properties in Self-Supervised ViTs (DINO)**, arXiv:2104.14294 (2021) | Shows ViT self-attention localises objects — the basis for the attention-map analysis (Analysis #2). |
| `PatchCore_2022_Roth_TotalRecall_arXiv2106.08265.pdf` | Roth et al., **Towards Total Recall in Industrial Anomaly Detection (PatchCore)**, CVPR 2022, arXiv:2106.08265 | The exact anomaly-detection recipe implemented: patch memory bank + greedy coreset + nearest-neighbour scoring. |
| `ViT_2021_Dosovitskiy_arXiv2010.11929.pdf` | Dosovitskiy et al., **An Image is Worth 16×16 Words (ViT)**, ICLR 2021, arXiv:2010.11929 | The Vision Transformer architecture underlying DINOv3. |
| `MD-AS-2025_Dataset_2025_NatureScientificData_s41597-025-06228-6.pdf` | **Image Dataset of Microfauna in Activated Sludge (MD-AS-2025)**, *Scientific Data* (2025), doi:10.1038/s41597-025-06228-6 | The dataset itself: 14,257 cropped images across 5 taxa. |

---

## B. Matching papers from your lab (Prof. ChangKyoo Yoo / EMSEL)

Grouped by how directly they connect to this project. Most appear in
Elsevier/Springer journals (paywalled — links/DOIs given for retrieval through
the university library); open-access ones are noted.

### B1 — Memory-bank / prototype-based anomaly detection  *(direct cousins of PatchCore)*
- **Ghorbani, V., Tariq, S., Yoo, C. (2025). "Validation of subway indoor air
  quality (IAQ) data using memory-augmented autoencoders with learned normal
  prototypes." *Korean Journal of Chemical Engineering.***
  → **Closest conceptual match.** Stores *learned normal prototypes* and flags
  deviations from them — the same core idea as PatchCore's healthy-class memory
  bank, applied to sensor data instead of image patches.
- **Ali, U., Tariq, S., Kim, K., Chang-Silva, R., Yoo, C. (2026).
  "Transfer learning-informed sensor validation for detecting and diagnosing
  *unseen* air quality faults in underground building environment."
  *Tunnelling and Underground Space Technology.***
  → Detecting **unseen faults** ≈ the zero-shot anomaly-detection framing of
  this project; also leans on transfer learning from a pretrained model.
- **Ali, U., Tariq, S., Kim, K., Chang-Silva, R., Yoo, C. (2026).
  "Interpretable distance-adaptive GCN-autoencoder for soft sensor validation
  and remote reconstruction in urban air quality monitoring networks."
  *ISA Transactions.***
  → Distance/reconstruction-based anomaly scoring with an interpretability
  angle (cf. our attention/heatmap analyses).
- **Jeong, C.H., Tariq, S., Woo, T.Y., Kim, S.Y., Nam, K.J., Yoo, C. (2026).
  "Towards health-resilient subway ventilation: an integrated framework for
  fault prognostics, health-aware control, and IAQ resilience evaluation."
  *ISA Transactions.***
  → Fault prognostics framework; the monitoring→control loop our microfauna
  front-end is meant to feed (cf. hypothesis H4).

### B2 — Classical multivariate process monitoring  *(the anomaly-detection lineage)*
- **Lee, J.M., Yoo, C., et al. (2004). "Nonlinear process monitoring using
  kernel principal component analysis." *Chemical Engineering Science.*** (1,444
  citations) → Kernel novelty detection: model the *normal* operating region and
  flag departures — the statistical ancestor of feature-space anomaly detection.
- **Lee, J.M., Yoo, C., Lee, I.B. (2004). "Statistical process monitoring with
  independent component analysis." *Journal of Process Control.*** → ICA-based
  fault detection; same "learn normal, detect deviation" philosophy.

### B3 — Wastewater treatment & water-quality monitoring  *(application domain)*
- **Kim, S.Y., Woo, T.Y., Jeong, C.H., …, Yoo, C. (2026). "Offline
  reinforcement-learning-driven feedforward control in a sequencing batch
  reactor for TMAH-rich semiconductor wastewater." *Journal of Water Process
  Engineering.*** → Activated-sludge-style biological reactor control — the very
  process whose microfauna health this project assesses.
- **Kim, M., Park, S., …, Yoo, C. (2025). "Optical analysis based on UV
  absorption spectrum for monitoring total organic carbon and nitrate nitrogen
  in river water." *Water* (MDPI, open access).** → Low-cost optical water-quality
  monitoring; sibling to a vision-based sludge-health sensor.
- EMSEL also has a long line of **WWTP modelling / influent-disturbance and
  benchmark-simulation (BSM)** studies that frame microfauna as indicators of
  treatment performance.

### B4 — Deep learning for environmental data
- **Ghorbani, V. & A., Yoo, C. (2025). "Evaluating deep learning data imputation
  for subway indoor air quality." *Building and Environment.***
- **Nam, K.J., Hwangbo, S., Yoo, C. (2020). "A deep learning-based forecasting
  model for renewable-energy scenarios." *Renewable Energy.***
  → The lab's broader use of deep learning on environmental time-series.

---

## C. Where this project sits relative to the lab's work

EMSEL's signature is **monitoring and fault/anomaly detection for environmental
and industrial systems** — KPCA/ICA process monitoring, autoencoder soft-sensor
validation, *memory-augmented* normal-prototype models, and unseen-fault
detection — applied heavily to **wastewater treatment and air-quality networks**.

This project extends that exact lineage in two ways:
1. **From sensor signals to images.** It carries the lab's "model the normal,
   flag the deviation" principle into the *visual* domain using a frozen vision
   foundation model (DINOv3) and a PatchCore memory bank of healthy microfauna.
2. **From hand-engineered features to self-supervised features.** Where the
   classic monitoring work relies on KPCA/ICA or trained autoencoders, here the
   "normal model" is built on label-free, pretrained DINOv3 patch embeddings —
   a low-cost, label-efficient front-end for WWTP microfauna monitoring (H4).

> Note on access: B1–B4 are paywalled except the MDPI *Water* paper. Titles,
> authors, venues, and years above are sufficient to pull the PDFs through the
> Kyung Hee library / publisher sites. Tell me if you want me to fetch any
> specific open-access versions or BibTeX entries.
