"""
Object-detection stage 2 - train a YOLO detector on MD-AS-2025 and compare with
the dataset paper's reported benchmark.

The paper (Scientific Data 2025) trained YOLOv5/YOLOv8 on a 3600/400 split and
reported mAP@0.5 per taxon.  We reproduce the benchmark with Ultralytics YOLOv8.
Because this machine is CPU-only, model size / image size / epochs are kept
modest and configurable via environment variables; the comparison is reported
honestly with that caveat.

Env knobs (defaults in brackets):
    OD_MODEL  [yolov8n.pt]   OD_IMGSZ [640]   OD_EPOCHS [40]
    OD_BATCH  [16]           OD_PATIENCE [15]

Outputs:
    results/csv/od_metrics_overall.csv
    results/csv/od_per_class_ap.csv          (ours vs paper)
    results/figures/od_comparison_vs_paper.png
    results/figures/od_predictions.png
    runs/ (full Ultralytics training artifacts: curves, confusion matrix, ...)
"""
from __future__ import annotations
import os
import sys
import glob
import shutil

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C

OD_DIR = os.path.join(C.DATA_PROCESSED, "od")
YAML = os.path.join(OD_DIR, "md_as_2025.yaml")
RUNS = os.path.join(C.ROOT, "runs")

# Paper's reported mAP@0.5 per taxon (Technical Validation table).
PAPER_MAP50 = {
    "Arcellinida": 0.958, "Digononta": 0.916, "Monogononta": 0.834,
    "Nematoda": 0.778, "Hypotrichida": 0.991,
}
PAPER_OVERALL = float(np.mean(list(PAPER_MAP50.values())))   # ~0.895


def main():
    from ultralytics import YOLO

    model_name = os.environ.get("OD_MODEL", "yolov8n.pt")
    imgsz = int(os.environ.get("OD_IMGSZ", 640))
    epochs = int(os.environ.get("OD_EPOCHS", 40))
    batch = int(os.environ.get("OD_BATCH", 16))
    patience = int(os.environ.get("OD_PATIENCE", 15))
    print(f"model={model_name} imgsz={imgsz} epochs={epochs} batch={batch}")

    model = YOLO(model_name)
    model.train(
        data=YAML, epochs=epochs, imgsz=imgsz, batch=batch, device="cpu",
        workers=4, patience=patience, seed=C.SEED, deterministic=True,
        project=RUNS, name="md_as_2025_yolo", exist_ok=True, verbose=True,
        plots=True,
    )

    # ---- validation metrics ------------------------------------------------ #
    metrics = model.val(data=YAML, imgsz=imgsz, device="cpu", split="val",
                        project=RUNS, name="md_as_2025_yolo_val", exist_ok=True)
    names = model.names                                  # {idx: name}
    ap50 = metrics.box.ap50                              # per-class AP@0.5
    map50 = float(metrics.box.map50)
    map5095 = float(metrics.box.map)
    mp = float(metrics.box.mp)
    mr = float(metrics.box.mr)

    pd.DataFrame([dict(model=model_name, imgsz=imgsz, epochs=epochs,
                       mAP50=map50, mAP50_95=map5095,
                       mean_precision=mp, mean_recall=mr,
                       paper_overall_mAP50=PAPER_OVERALL)]).to_csv(
        os.path.join(C.RES_CSV, "od_metrics_overall.csv"), index=False)

    # ---- per-class comparison --------------------------------------------- #
    rows = []
    for ci in range(len(names)):
        nm = names[ci]
        ours = float(ap50[ci]) if ci < len(ap50) else float("nan")
        paper = PAPER_MAP50.get(nm, float("nan"))
        rows.append(dict(taxon=nm, ours_AP50=ours, paper_mAP50=paper,
                         diff=ours - paper))
    perclass = pd.DataFrame(rows)
    perclass.loc["overall"] = ["OVERALL (mAP50)", map50, PAPER_OVERALL,
                               map50 - PAPER_OVERALL]
    perclass.to_csv(os.path.join(C.RES_CSV, "od_per_class_ap.csv"), index=False)
    print(perclass.to_string(index=False))

    # ---- comparison figure ------------------------------------------------- #
    pc = perclass.iloc[:-1]
    x = np.arange(len(pc)); w = 0.38
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w / 2, pc.ours_AP50, w, label=f"Ours ({model_name}, {epochs}ep)",
           color="#264653")
    ax.bar(x + w / 2, pc.paper_mAP50, w, label="MD-AS-2025 paper",
           color="#e9c46a")
    ax.set_xticks(x); ax.set_xticklabels(pc.taxon, rotation=15)
    ax.set_ylim(0, 1.05); ax.set_ylabel("AP@0.5")
    ax.axhline(map50, ls="--", c="#264653", lw=1,
               label=f"our overall mAP50={map50:.3f}")
    ax.axhline(PAPER_OVERALL, ls="--", c="#e9c46a", lw=1,
               label=f"paper overall={PAPER_OVERALL:.3f}")
    ax.set_title("Object detection vs MD-AS-2025 paper (AP@0.5 per taxon)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(C.RES_FIG, "od_comparison_vs_paper.png"), dpi=140)
    plt.close(fig)

    # ---- sample predictions ------------------------------------------------ #
    val_imgs = [l.strip() for l in open(os.path.join(OD_DIR, "val.txt"))
                if l.strip()][:8]
    preds = model.predict(val_imgs, imgsz=imgsz, device="cpu", conf=0.25,
                          verbose=False)
    n = len(preds); cols = 4; rows_ = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows_, cols, figsize=(cols * 4, rows_ * 3))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for ax, pr in zip(axes, preds):
        ax.imshow(pr.plot()[:, :, ::-1])   # BGR->RGB
        ax.axis("off")
    fig.suptitle("YOLO predictions on MD-AS-2025 validation frames")
    fig.tight_layout()
    fig.savefig(os.path.join(C.RES_FIG, "od_predictions.png"), dpi=120)
    plt.close(fig)

    # copy a couple of Ultralytics curve plots into results/figures
    run_dir = os.path.join(RUNS, "md_as_2025_yolo")
    for fn in ["results.png", "confusion_matrix_normalized.png",
               "PR_curve.png", "BoxP_curve.png"]:
        src = os.path.join(run_dir, fn)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(C.RES_FIG, f"od_{fn}"))

    print(f"\nOURS mAP@0.5={map50:.3f}  |  PAPER mAP@0.5={PAPER_OVERALL:.3f}")
    print("wrote od_metrics_overall.csv, od_per_class_ap.csv, comparison figure")


if __name__ == "__main__":
    main()
