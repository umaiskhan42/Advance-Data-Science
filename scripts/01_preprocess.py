"""
Stage 1 - Preprocessing.

Scans the extracted MD-AS-2025 cropped folders, verifies every image opens,
samples a tractable subset per taxon, and assigns the anomaly-detection split:

    HEALTHY taxa  -> label 0 (normal);  TRAIN_FRACTION go to the memory bank,
                     the rest become healthy test images.
    STRESSED taxa -> label 1 (anomaly); all become test images.

Outputs
    results/csv/manifest.csv          one row per selected image
    results/csv/dataset_summary.csv   per-taxon counts (full vs sampled vs split)
    results/figures/dataset_distribution.png
"""
from __future__ import annotations
import os
import sys

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
import common


def list_images(taxon: str) -> list[str]:
    base = os.path.join(C.DATA_RAW, taxon)
    out = []
    for r, _d, fs in os.walk(base):
        for fn in fs:
            if fn.lower().endswith(C.IMG_EXTS):
                out.append(os.path.join(r, fn))
    return sorted(out)


def verify(path: str) -> bool:
    try:
        with Image.open(path) as im:
            im.verify()
        return True
    except Exception:
        return False


def main() -> None:
    C.ensure_dirs()
    common.set_seed()
    rng = np.random.default_rng(C.SEED)

    rows = []
    summary = []
    for taxon in C.ALL_TAXA:
        is_healthy = taxon in C.HEALTHY_TAXA
        cap = C.MAX_HEALTHY_PER_TAXON if is_healthy else C.MAX_STRESSED_PER_TAXON

        all_imgs = [p for p in list_images(taxon) if verify(p)]
        n_full = len(all_imgs)

        sel = list(all_imgs)
        rng.shuffle(sel)
        sel = sel[:cap]
        n_sel = len(sel)

        if is_healthy:
            n_train = int(round(n_sel * C.TRAIN_FRACTION))
            train_set = set(sel[:n_train])
            label = C.LABEL_NORMAL
        else:
            train_set = set()                     # stressed taxa are test-only
            label = C.LABEL_ANOMALY

        n_tr = n_te = 0
        for p in sel:
            split = "train" if p in train_set else "test"
            n_tr += split == "train"
            n_te += split == "test"
            rows.append(dict(
                path=os.path.relpath(p, C.ROOT).replace("\\", "/"),
                taxon=taxon,
                group="healthy" if is_healthy else "stressed",
                label=label,
                split=split,
            ))

        summary.append(dict(taxon=taxon,
                            group="healthy" if is_healthy else "stressed",
                            n_available=n_full, n_selected=n_sel,
                            n_train=n_tr, n_test=n_te))
        print(f"{taxon:14s} avail={n_full:5d} selected={n_sel:4d} "
              f"train={n_tr:4d} test={n_te:4d}")

    manifest = pd.DataFrame(rows)
    manifest.to_csv(os.path.join(C.RES_CSV, "manifest.csv"), index=False)

    summ = pd.DataFrame(summary)
    summ.loc["TOTAL"] = summ.sum(numeric_only=True)
    summ.loc["TOTAL", ["taxon", "group"]] = ["TOTAL", ""]
    summ.to_csv(os.path.join(C.RES_CSV, "dataset_summary.csv"), index=False)

    # ---- distribution figure ---------------------------------------------- #
    plot_df = pd.DataFrame(summary)
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    colors = ["#2a9d8f" if g == "healthy" else "#e76f51"
              for g in plot_df["group"]]
    ax[0].bar(plot_df["taxon"], plot_df["n_available"], color=colors)
    ax[0].set_title("MD-AS-2025 cropped images available per taxon")
    ax[0].set_ylabel("images")
    ax[0].tick_params(axis="x", rotation=30)

    width = 0.4
    x = np.arange(len(plot_df))
    ax[1].bar(x - width / 2, plot_df["n_train"], width, label="train (bank)",
              color="#264653")
    ax[1].bar(x + width / 2, plot_df["n_test"], width, label="test",
              color="#e9c46a")
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(plot_df["taxon"], rotation=30)
    ax[1].set_title("Sampled split used in experiments")
    ax[1].set_ylabel("images")
    ax[1].legend()
    fig.suptitle("Healthy (green/train) = normal  |  Stressed (orange) = anomaly",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(C.RES_FIG, "dataset_distribution.png"), dpi=130)
    plt.close(fig)

    print(f"\nmanifest rows: {len(manifest)}")
    print("wrote manifest.csv, dataset_summary.csv, dataset_distribution.png")


if __name__ == "__main__":
    main()
