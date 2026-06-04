# Positioning: an extension of EMSEL lab work (copy-paste ready)

Use these in the proposal/report so the project reads as a **continuation of the
lab's anomaly-detection research**, not standalone new research.

---

### One-sentence framing (for the abstract)
> We extend EMSEL's normal-prototype anomaly-detection approach — previously
> applied to wastewater-treatment and indoor-air-quality **sensor data** — to the
> **visual domain**, using a frozen DINOv3 vision foundation model and a
> PatchCore memory bank of *healthy* microfauna to flag stress-indicating taxa in
> activated-sludge microscopy.

### Contribution statement (for the introduction)
> This work does not propose a new anomaly-detection paradigm. Instead, it ports
> the lab's established "model the normal, flag the deviation" recipe to a new
> data modality. EMSEL has built this lineage on sensor signals — KPCA/ICA
> process monitoring (Lee & Yoo, 2004), deep variational-residual autoencoders
> for WWTP sensor fault detection (Ba-Alawi et al., *Chemosphere* 2022), and
> memory-augmented autoencoders with **learned normal prototypes** (Ghorbani,
> Tariq & Yoo, *KJChE* 2025). Our PatchCore memory bank of healthy DINOv3 patch
> embeddings is the direct **image-domain analogue** of that 2025
> normal-prototype memory. The two changes are (i) modality — sensor time-series
> → microscopy images — and (ii) feature source — a trained autoencoder → a
> *frozen*, label-free self-supervised model.

### Related-work paragraph
> The closest prior work is the lab's own memory-augmented autoencoder with
> learned normal prototypes for IAQ data validation (Ghorbani, Tariq & Yoo,
> 2025), which memorises normal prototypes and scores deviations from them — the
> same mechanism as PatchCore. In the wastewater domain specifically, the lab's
> deep variational-residual autoencoder for sensor self-validation (Ba-Alawi
> et al., 2022) established "learn the normal manifold, detect/reconstruct
> anomalies" for WWTP operation, and its transfer-learning detection of *unseen*
> faults (Ali et al., 2026) mirrors our zero-shot setting, where stressed taxa
> are never added to the memory bank. This project unifies these threads on
> activated-sludge **microscopy imagery**, addressing the lab's goal of a
> low-cost front-end for WWTP / aeration-energy monitoring.

### Why this matters for the lab
> The lab already treats microfauna/microbial state as an indicator of sludge
> health and aeration demand, but its tooling reads that state from sensors and
> sequencing. A frozen vision model that detects stressed microfauna directly
> from a microscope frame — with no labels and no training — gives the lab a new,
> cheaper input channel feeding the same monitoring-and-control loop it already
> builds (e.g., reinforcement-learning aeration control in sequencing batch
> reactors, Kim et al., 2026).

---
Sources/details: [`Literature/CLOSEST_LAB_WORK.md`](Literature/CLOSEST_LAB_WORK.md),
[`Literature/REFERENCES.md`](Literature/REFERENCES.md).
