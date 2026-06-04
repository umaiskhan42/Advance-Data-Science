# Closest prior work from your lab (Prof. ChangKyoo Yoo / EMSEL, Kyung Hee Univ.)

**Question:** which of the lab's own papers is closest to this project —
*memory-bank anomaly detection of stressed microfauna with a frozen DINOv3
backbone, plus object detection on activated-sludge microscopy*?

First, a clarification worth recording: the **MD-AS-2025 dataset is *not* from
your lab** — it was produced by Z. Pang, A. Wang, Y. Tao et al. (a Chinese
environmental-microbiology group; *Scientific Data* 2025). So "closest work"
means the nearest match among EMSEL's own publications.

---

## 🥇 The single closest work — by mechanism

> **Ghorbani, V., Tariq, S., & Yoo, C. (2025). "Validation of Subway Indoor Air
> Quality (IAQ) Data Using Memory-Augmented Autoencoders with *Learned Normal
> Prototypes*." *Korean Journal of Chemical Engineering*, 42, 2231–2252.**
> doi:10.1007/s11814-025-00451-y

This is the closest because it uses **the same core idea as PatchCore**, the
method in this project:

| | Lab paper (Ghorbani & Yoo, 2025) | This project |
|---|---|---|
| Core idea | memorise **normal prototypes**, flag deviations | memory bank of **healthy patch embeddings**, flag deviations |
| "Memory" | learned normal-prototype memory in an autoencoder | greedy-coreset memory bank of DINOv3 patches |
| Anomaly score | reconstruction error vs. nearest normal prototype | nearest-neighbour distance to nearest normal patch |
| Supervision | trained only on normal data (one-class) | built only on healthy taxa (zero-shot for stressed) |
| Data | IAQ sensor time-series | activated-sludge microscopy images |
| Feature source | autoencoder learned on the data | **frozen** self-supervised DINOv3 (no training) |

It is described as "the first validation method that utilises normal prototypes
for reconciling corrupted measurements" — conceptually the time-series sibling of
a PatchCore memory bank. The two differences are exactly this project's
contribution: **images instead of sensors**, and a **frozen vision foundation
model instead of a trained autoencoder**.

---

## 🥈 Closest work — by application domain (wastewater)

> **Ba-Alawi, A. H., Loy-Benitez, J., Kim, S., & Yoo, C. (2022). "Missing data
> imputation and sensor self-validation towards a sustainable operation of
> wastewater treatment plants via deep variational residual autoencoders."
> *Chemosphere*, 288, 132647.** (Yoo = corresponding author)

Closest by **domain + principle**: it performs **fault/anomaly detection for
wastewater-treatment-plant sensors** (100% fault-detection rate) by learning the
normal operating manifold with a variational/residual autoencoder and flagging
+ reconstructing deviations. Same "model the normal, detect the anomaly" logic
this project applies — in the exact WWTP setting the project ultimately targets
(hypothesis H4: a front-end for WWTP monitoring).

---

## 🥉 Closest work — by the *zero-shot* framing

> **Ali, U., Tariq, S., Kim, K., Chang-Silva, R., & Yoo, C. (2026).
> "Transfer-learning-informed sensor validation for detecting and diagnosing
> *unseen* faults in underground building environments." *Tunnelling and
> Underground Space Technology.***

Closest by the **"detect categories never seen in training"** framing — the same
zero-shot spirit as flagging stressed taxa that were never put in the memory
bank — and it also relies on transfer learning from pretrained representations.

---

## Lineage in one line

EMSEL's anomaly-detection signature — **KPCA/ICA process monitoring (2004) →
deep variational-residual-autoencoder WWTP fault detection (2022) →
memory-augmented *normal-prototype* autoencoders (2025) → unseen-fault detection
(2026)** — is precisely "learn the normal, flag the deviation." No EMSEL paper
has yet done this on **microscopy images** or with a **frozen vision foundation
model**: that gap is exactly what this project fills, carrying the lab's
prototype/memory-based anomaly-detection lineage from sensor signals into
computer vision for activated-sludge microfauna.

> Verified author/venue details via PubMed, Springer, and the EMSEL site.
> Full PDFs are paywalled (Springer/Elsevier); retrieve through the KHU library.
