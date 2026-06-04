"""
Stage 3 - Analysis & visualisation (the four interrogations in the proposal).

  01  t-SNE / UMAP   of DINOv3 CLS embeddings (healthy vs stressed, by taxon)
  02  Attention maps  DINOv3 last-block CLS self-attention overlays
  03  Anomaly heatmaps  patch-level PatchCore scores overlaid on the image
  04  Failure cases   worst false positives / false negatives + a CSV

Also renders the quantitative comparison figures (AUROC bars, ROC curves,
score distributions, per-taxon AUROC) from the stage-2 CSVs.
"""
from __future__ import annotations
import os
import sys

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
import common

TAXON_COLORS = {
    "Hypotrichida": "#2a9d8f", "Digononta": "#43aa8b", "Monogononta": "#90be6d",
    "Arcellinida": "#e76f51", "Nematoda": "#d62828",
}


# --------------------------------------------------------------------------- #
# 01  t-SNE / UMAP of CLS embeddings
# --------------------------------------------------------------------------- #
def embedding_plots():
    npz = np.load(os.path.join(C.DATA_PROCESSED, "cls_embeddings_dinov3.npz"),
                  allow_pickle=True)
    X = common.l2norm(npz["test_cls"].astype(np.float32))
    taxa = npz["test_taxon"].astype(str)
    labels = npz["test_label"]

    print("  running t-SNE ...")
    ts = TSNE(n_components=2, perplexity=30, init="pca",
              random_state=C.SEED, max_iter=1000)
    emb = ts.fit_transform(X)
    _scatter(emb, taxa, labels, "t-SNE", "tsne")
    pd.DataFrame(dict(x=emb[:, 0], y=emb[:, 1], taxon=taxa,
                      label=labels)).to_csv(
        os.path.join(C.RES_TSNE, "tsne_coords.csv"), index=False)

    try:
        import umap
        print("  running UMAP ...")
        um = umap.UMAP(n_components=2, random_state=C.SEED, n_neighbors=30,
                       min_dist=0.1)
        emb2 = um.fit_transform(X)
        _scatter(emb2, taxa, labels, "UMAP", "umap")
        pd.DataFrame(dict(x=emb2[:, 0], y=emb2[:, 1], taxon=taxa,
                          label=labels)).to_csv(
            os.path.join(C.RES_TSNE, "umap_coords.csv"), index=False)
    except Exception as e:
        print(f"  UMAP skipped ({e})")


def _scatter(emb, taxa, labels, title, stem):
    # by taxon
    fig, ax = plt.subplots(figsize=(8, 7))
    for t in C.ALL_TAXA:
        m = taxa == t
        ax.scatter(emb[m, 0], emb[m, 1], s=10, alpha=0.6,
                   c=TAXON_COLORS[t], label=t)
    ax.set_title(f"DINOv3 CLS embeddings - {title} (by taxon)")
    ax.legend(markerscale=2, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(os.path.join(C.RES_TSNE, f"{stem}_taxa.png"), dpi=140)
    plt.close(fig)

    # healthy vs stressed
    fig, ax = plt.subplots(figsize=(8, 7))
    for lab, col, name in [(0, "#2a9d8f", "healthy (normal)"),
                           (1, "#e76f51", "stressed (anomaly)")]:
        m = labels == lab
        ax.scatter(emb[m, 0], emb[m, 1], s=10, alpha=0.5, c=col, label=name)
    ax.set_title(f"DINOv3 CLS embeddings - {title} (healthy vs stressed)")
    ax.legend(markerscale=2)
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(os.path.join(C.RES_TSNE, f"{stem}_health.png"), dpi=140)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 02  Attention maps  (DINOv3 last-block CLS attention)
# --------------------------------------------------------------------------- #
def attention_maps(test_df):
    bb = common.Backbone("dinov3")
    g = C.IMG_SIZE // 16
    for taxon in C.ALL_TAXA:
        paths = test_df[test_df.taxon == taxon]["path"].tolist()[:C.N_VIS_PER_TAXON]
        for i, rel in enumerate(paths):
            x = common.load_image(os.path.join(C.ROOT, rel)).unsqueeze(0)
            with common.capture_attention() as grab:
                with torch.no_grad():
                    bb.model.forward_features(x)
            attn = grab.last[0, :, 0, bb.n_prefix:].mean(0)        # (196,)
            amap = attn.reshape(g, g).cpu().numpy()
            amap = (amap - amap.min()) / (np.ptp(amap) + 1e-8)
            _overlay(rel, amap, taxon, i, C.RES_ATTN, "attn",
                     "DINOv3 attention")
    print(f"  attention overlays -> {C.RES_ATTN}")


# --------------------------------------------------------------------------- #
# 03  Anomaly heatmaps  (PatchCore patch scores)
# --------------------------------------------------------------------------- #
def anomaly_heatmaps(test_df):
    cache_path = os.path.join(C.DATA_PROCESSED, "heatmap_cache_dinov3.npz")
    if not os.path.exists(cache_path):
        print("  heatmap cache missing - skipped")
        return
    cache = np.load(cache_path)
    keymap = {k.replace("__", "/"): k for k in cache.files}
    for taxon in C.ALL_TAXA:
        paths = test_df[test_df.taxon == taxon]["path"].tolist()[:C.N_VIS_PER_TAXON]
        for i, rel in enumerate(paths):
            if rel not in keymap:
                continue
            dmap = cache[keymap[rel]]
            dmap = (dmap - dmap.min()) / (np.ptp(dmap) + 1e-8)
            _overlay(rel, dmap, taxon, i, C.RES_HEAT, "heatmap",
                     "PatchCore anomaly")
    print(f"  anomaly heatmaps -> {C.RES_HEAT}")


def _overlay(rel, gridmap, taxon, i, outdir, stem, title):
    raw = common.load_image_raw(os.path.join(C.ROOT, rel))
    up = np.array(_resize(gridmap, C.IMG_SIZE))
    fig, ax = plt.subplots(1, 3, figsize=(11, 4))
    ax[0].imshow(raw); ax[0].set_title(f"{taxon}"); ax[0].axis("off")
    ax[1].imshow(up, cmap="jet"); ax[1].set_title(title); ax[1].axis("off")
    ax[2].imshow(raw)
    ax[2].imshow(up, cmap="jet", alpha=0.5)
    ax[2].set_title("overlay"); ax[2].axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"{stem}_{taxon}_{i}.png"), dpi=110)
    plt.close(fig)


def _resize(arr2d, size):
    from PIL import Image
    im = Image.fromarray((arr2d * 255).astype(np.uint8)).resize(
        (size, size), Image.BICUBIC)
    return np.asarray(im, dtype=np.float32) / 255.0


# --------------------------------------------------------------------------- #
# 04  Failure cases
# --------------------------------------------------------------------------- #
def failure_cases():
    scores = pd.read_csv(os.path.join(
        C.RES_CSV, f"per_image_scores_{C.PRIMARY_BACKBONE}.csv"))
    thr = pd.read_csv(os.path.join(C.RES_CSV, "threshold_metrics.csv"))
    t = float(thr[(thr.backbone == C.PRIMARY_BACKBONE) &
                  (thr.operating_point == "youden_J")]["threshold"].iloc[0])

    fp = scores[(scores.label == 0) & (scores.anomaly_score >= t)].sort_values(
        "anomaly_score", ascending=False)
    fn = scores[(scores.label == 1) & (scores.anomaly_score < t)].sort_values(
        "anomaly_score", ascending=True)

    out = pd.concat([
        fp.head(C.N_FAILURE_CASES).assign(error_type="false_positive"),
        fn.head(C.N_FAILURE_CASES).assign(error_type="false_negative")])
    out.to_csv(os.path.join(C.RES_CSV, "failure_cases.csv"), index=False)

    _montage(fp.head(C.N_FAILURE_CASES), "False positives (healthy scored anomalous)",
             os.path.join(C.RES_FAIL, "false_positives.png"))
    _montage(fn.head(C.N_FAILURE_CASES), "False negatives (stressed scored normal)",
             os.path.join(C.RES_FAIL, "false_negatives.png"))
    print(f"  failure cases -> {C.RES_FAIL} (FP={len(fp)}, FN={len(fn)}, "
          f"threshold={t:.3f})")


def _montage(df, title, out):
    n = len(df)
    if n == 0:
        return
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for ax, (_, r) in zip(axes, df.iterrows()):
        ax.imshow(common.load_image_raw(os.path.join(C.ROOT, r["path"])))
        ax.set_title(f"{r['taxon']}\n{r['anomaly_score']:.3f}", fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Quantitative comparison figures
# --------------------------------------------------------------------------- #
def comparison_figures():
    summ = pd.read_csv(os.path.join(C.RES_CSV, "metrics_summary.csv"))
    patch = summ[summ.method == "patchcore_patch"]

    # AUROC bar chart
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(patch.backbone, patch.auroc,
                  color=["#264653", "#bbbbbb", "#e9c46a"][:len(patch)])
    ax.axhline(0.5, ls="--", c="r", lw=1, label="chance")
    ax.set_ylim(0, 1); ax.set_ylabel("Image-level AUROC")
    ax.set_title("H1 - PatchCore AUROC by backbone (healthy vs stressed)")
    for b, v in zip(bars, patch.auroc):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}",
                ha="center", fontsize=10)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(C.RES_FIG, "auroc_comparison.png"), dpi=140)
    plt.close(fig)

    # ROC curves
    fig, ax = plt.subplots(figsize=(7, 6))
    for name in C.BACKBONES:
        f = os.path.join(C.RES_CSV, f"roc_curve_{name}.csv")
        if os.path.exists(f):
            r = pd.read_csv(f)
            au = float(patch[patch.backbone == name]["auroc"].iloc[0])
            ax.plot(r.fpr, r.tpr, label=f"{name} (AUROC={au:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves - PatchCore by backbone")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(C.RES_FIG, "roc_curves.png"), dpi=140)
    plt.close(fig)

    # per-taxon AUROC
    pt = pd.read_csv(os.path.join(C.RES_CSV, "per_taxon_auroc.csv"))
    fig, ax = plt.subplots(figsize=(8, 5))
    comps = pt.comparison.unique()
    x = np.arange(len(comps))
    w = 0.25
    for k, name in enumerate(C.BACKBONES):
        sub = pt[pt.backbone == name].set_index("comparison").reindex(comps)
        ax.bar(x + (k - 1) * w, sub.auroc, w, label=name)
    ax.set_xticks(x); ax.set_xticklabels(comps, rotation=10)
    ax.axhline(0.5, ls="--", c="r", lw=1)
    ax.set_ylim(0, 1); ax.set_ylabel("AUROC")
    ax.set_title("Per-taxon (per-failure-mode) AUROC")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(C.RES_FIG, "per_taxon_auroc.png"), dpi=140)
    plt.close(fig)

    # score distributions (primary backbone)
    s = pd.read_csv(os.path.join(
        C.RES_CSV, f"per_image_scores_{C.PRIMARY_BACKBONE}.csv"))
    fig, ax = plt.subplots(figsize=(8, 5))
    for taxon in C.ALL_TAXA:
        sub = s[s.taxon == taxon]
        ax.hist(sub.anomaly_score, bins=40, alpha=0.5, density=True,
                color=TAXON_COLORS[taxon], label=taxon)
    ax.set_xlabel("anomaly score"); ax.set_ylabel("density")
    ax.set_title("DINOv3 PatchCore anomaly-score distribution by taxon")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(C.RES_FIG, "score_distributions.png"), dpi=140)
    plt.close(fig)
    print(f"  comparison figures -> {C.RES_FIG}")


def main():
    C.ensure_dirs()
    common.set_seed()
    test_df = pd.read_csv(os.path.join(C.RES_CSV, "manifest.csv"))
    test_df = test_df[test_df.split == "test"].reset_index(drop=True)

    print("[01] embedding plots (t-SNE / UMAP)")
    embedding_plots()
    print("[02] attention maps")
    attention_maps(test_df)
    print("[03] anomaly heatmaps")
    anomaly_heatmaps(test_df)
    print("[04] failure cases")
    failure_cases()
    print("[05] comparison figures")
    comparison_figures()
    print("done.")


if __name__ == "__main__":
    main()
