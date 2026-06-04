# Papers that have used the MD-AS-2025 dataset

**Short answer:** as of May 2026, the MD-AS-2025 dataset is very new (published
**10 Dec 2025** in *Scientific Data*), and I could find **no third-party paper
that uses this exact dataset yet**. The only work using it is:

1. **The dataset descriptor itself** —
   Pang, Z., Wang, A., Chen, Y., Xu, T., Shao, Z., Ge, C., He, C., & Tao, Y.
   (2025). *Image Dataset of Microfauna in Activated Sludge.* **Scientific Data**
   12:1972. doi:10.1038/s41597-025-06228-6.
   It establishes the YOLOv5/YOLOv8 detection benchmark (3600/400 split;
   mAP@0.5 per taxon: Arcellinida 0.958, Digononta 0.916, Monogononta 0.834,
   Hypotrichida 0.991, Nematoda 0.778).

2. **This project** (Umais Khan, 2025311037) — uses MD-AS-2025 for (a) zero-shot
   anomaly detection with frozen DINOv3 + PatchCore, and (b) object detection
   (frozen-DINOv3 head + YOLOv8 reproduction).

I checked Nature, Semantic Scholar, and Google-Scholar-style queries; no other
citations using the data were found (expected for a ~5-month-old data paper). A
ResearchGate "figure" hit (publication 398550106) is the dataset paper's own
directory-structure figure, not an independent user.

---

## Related works on activated-sludge microscopy + deep learning (DIFFERENT datasets)

These are the nearest neighbours in the literature — same task family
(microorganism/floc detection or sludge-health from microscopy) but built on
**other** datasets, not MD-AS-2025. Useful for the related-work section:

- Liang et al. (2025). *Automatic visual detection of activated sludge
  microorganisms based on microscopic phase-contrast image optimisation and deep
  learning.* **Journal of Microscopy.** doi:10.1111/jmi.13385.
- *Microorganism Detection in Activated Sludge Microscopic Images Using Improved
  YOLO.* **Applied Sciences** 13(22):12406 (2023). (YOLOv8n-SimAM variant.)
- *Evaluation of Activated Sludge Settling Characteristics from Microscopy Images
  with Deep CNNs and Transfer Learning.* arXiv:2402.09367 / ScienceDirect (2024).
- *Microscopic Studies of Activated Sludge Supported by Automatic Image Analysis
  Based on Deep Learning Neural Networks.* **J. Ecol. Eng.** (JEENG).
- *Deep learning-based morphology classification of activated sludge flocs in
  WWTPs* (2020).

> Bottom line: this project is, in practice, among the **first independent uses**
> of MD-AS-2025 — and (to my knowledge) the first to apply a self-supervised
> vision foundation model (DINOv3) and a memory-bank anomaly-detection framing to
> it. Worth stating explicitly in the report.
