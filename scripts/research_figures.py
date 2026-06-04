"""
Publication-grade re-rendering of every key figure in the project, following the
reference style (Times New Roman bold, hatched grouped bars, dotted y-grid,
subplot tags, two-tone teal/peach palette, 300 dpi PNG + 400 dpi LZW-TIFF).

Existing figures in results/figures/ are overwritten so the .docx / .pdf
proposal automatically picks up the upgraded versions on re-export.
"""
from __future__ import annotations
import os
import sys
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C

FIG = Path(C.RES_FIG)
TIF = FIG / "tiff"; TIF.mkdir(exist_ok=True)

# --------------------------------------------------------------------------- #
# Unified journal style (rcParams)
# --------------------------------------------------------------------------- #
plt.rcParams.update({
    "font.family": "Times New Roman", "font.size": 14, "font.weight": "bold",
    "axes.labelweight": "bold", "axes.titleweight": "bold",
    "axes.labelsize": 15, "axes.titlesize": 15,
    "xtick.labelsize": 13, "ytick.labelsize": 12, "legend.fontsize": 13,
    "axes.linewidth": 1.2, "axes.edgecolor": "black",
    "xtick.direction": "out", "ytick.direction": "out",
    "lines.linewidth": 2.0, "patch.linewidth": 0.8,
})

C_DINOV3 = "#1F4E4A"        # deep teal  (primary / "with-AD" / proposed)
C_PEACH = "#E8A079"         # warm peach (baseline / "without-AD")
C_GRAY = "#9CA3AF"          # neutral gray (third comparator)
C_HEALTHY = "#2A9D8F"
C_STRESSED = "#E76F51"
H1, H2 = "////", "...."     # hatch patterns
TAX_COLORS = {"Hypotrichida": "#2A9D8F", "Digononta": "#43AA8B",
              "Monogononta": "#588157", "Arcellinida": "#E76F51",
              "Nematoda":   "#D62828"}
HEALTHY = ["Hypotrichida", "Digononta", "Monogononta"]
STRESSED = ["Arcellinida", "Nematoda"]


def save(fig, name):
    fig.savefig(FIG / f"{name}.png", dpi=300, bbox_inches="tight", pad_inches=0.1)
    fig.savefig(TIF / f"{name}.tiff", dpi=400, format="tiff",
                bbox_inches="tight", pad_inches=0.1,
                pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)


def panel_tag(ax, tag):
    ax.set_title(tag, loc="left", fontweight="bold")


def style_axes(ax):
    ax.grid(True, axis="y", linestyle=":", alpha=0.45)
    ax.set_axisbelow(True)
    ax.margins(y=0.18)


# --------------------------------------------------------------------------- #
# 1. Dataset distribution                          (overwrites Fig 1 of proposal)
# --------------------------------------------------------------------------- #
def fig_dataset():
    df = pd.read_csv(os.path.join(C.RES_CSV, "dataset_summary.csv"))
    df = df[df.taxon != "TOTAL"].copy()
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    ax = axes[0]
    colors = [C_HEALTHY if g == "healthy" else C_STRESSED for g in df.group]
    ax.bar(df.taxon, df.n_available, color=colors, edgecolor="black",
           linewidth=0.8)
    for i, v in enumerate(df.n_available):
        ax.text(i, v + max(df.n_available) * 0.02, f"{int(v):,}", ha="center",
                fontsize=11, fontweight="bold")
    ax.set_ylabel("# cropped images"); ax.tick_params(axis="x", rotation=20)
    panel_tag(ax, "(a) Cropped images per taxon (full MD-AS-2025)")
    style_axes(ax)
    ax.legend(handles=[mpatches.Patch(color=C_HEALTHY, label="Healthy taxa"),
                       mpatches.Patch(color=C_STRESSED, label="Stressed taxa")],
              loc="upper right", frameon=True, framealpha=0.95)

    ax = axes[1]
    x = np.arange(len(df)); w = 0.38
    ax.bar(x - w / 2, df.n_train, w, color=C_DINOV3, hatch=H1,
           edgecolor="black", linewidth=0.7, label="Train (memory bank)")
    ax.bar(x + w / 2, df.n_test, w, color=C_PEACH, hatch=H2,
           edgecolor="black", linewidth=0.7, label="Test")
    ax.set_xticks(x); ax.set_xticklabels(df.taxon, rotation=20)
    ax.set_ylabel("# images")
    panel_tag(ax, "(b) Sampled split used in the experiments")
    style_axes(ax); ax.legend(loc="upper right", frameon=True, framealpha=0.95)

    fig.tight_layout(); save(fig, "dataset_distribution")


# --------------------------------------------------------------------------- #
# 2. Headline AUROC by backbone  (H1)
# --------------------------------------------------------------------------- #
def fig_auroc():
    s = pd.read_csv(os.path.join(C.RES_CSV, "metrics_summary.csv"))
    s = s[s.method == "patchcore_patch"].set_index("backbone").reindex(
        ["dinov3", "resnet50", "randvit"])
    labels = ["DINOv3\n(frozen)", "ResNet-50\n(ImageNet)", "ViT\n(random init)"]
    colors = [C_DINOV3, C_PEACH, C_GRAY]
    hatches = [H1, H2, "xxxx"]

    fig, ax = plt.subplots(figsize=(8, 6))
    x = np.arange(3)
    bars = ax.bar(x, s.auroc, color=colors, hatch=hatches, edgecolor="black",
                  linewidth=0.8)
    for b, v in zip(bars, s.auroc):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.3f}",
                ha="center", fontsize=12, fontweight="bold")
    ax.axhline(0.5, ls="--", c="r", lw=1.2, label="chance (AUROC = 0.5)")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Image-level AUROC"); ax.set_ylim(0, 1.0)
    panel_tag(ax, "H1 — PatchCore AUROC by backbone (healthy vs stressed)")
    style_axes(ax); ax.legend(loc="upper right", frameon=True, framealpha=0.95)
    fig.tight_layout(); save(fig, "auroc_comparison")


# --------------------------------------------------------------------------- #
# 3. Per-taxon AUROC (grouped bars: backbone x stressed taxon)
# --------------------------------------------------------------------------- #
def fig_per_taxon():
    pt = pd.read_csv(os.path.join(C.RES_CSV, "per_taxon_auroc.csv"))
    comps = ["healthy_vs_Arcellinida", "healthy_vs_Nematoda"]
    labels = ["Healthy vs Arcellinida", "Healthy vs Nematoda"]
    x = np.arange(len(comps)); w = 0.27

    fig, ax = plt.subplots(figsize=(10, 6))
    for k, (name, color, hatch, disp) in enumerate(
            [("dinov3",   C_DINOV3, H1,    "DINOv3 (frozen)"),
             ("resnet50", C_PEACH,  H2,    "ResNet-50 (ImageNet)"),
             ("randvit",  C_GRAY,   "xxxx", "ViT (random init)")]):
        v = pt[pt.backbone == name].set_index("comparison").reindex(comps).auroc
        bars = ax.bar(x + (k - 1) * w, v, w, color=color, hatch=hatch,
                      edgecolor="black", linewidth=0.7, label=disp)
        for b, val in zip(bars, v):
            ax.text(b.get_x() + b.get_width() / 2, val + 0.012, f"{val:.3f}",
                    ha="center", fontsize=10, fontweight="bold")
    ax.axhline(0.5, ls="--", c="r", lw=1.2)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("AUROC"); ax.set_ylim(0, 1.0)
    panel_tag(ax, "Per-failure-mode AUROC")
    style_axes(ax)
    ax.legend(loc="upper right", ncol=1, frameon=True, framealpha=0.95)
    fig.tight_layout(); save(fig, "per_taxon_auroc")


# --------------------------------------------------------------------------- #
# 4. ROC curves
# --------------------------------------------------------------------------- #
def fig_roc():
    s = pd.read_csv(os.path.join(C.RES_CSV, "metrics_summary.csv"))
    s = s[s.method == "patchcore_patch"].set_index("backbone")
    fig, ax = plt.subplots(figsize=(8, 7))
    for name, color, hatch_ls, disp in [
            ("dinov3",   C_DINOV3, "-",  "DINOv3 (frozen)"),
            ("resnet50", C_PEACH,  "--", "ResNet-50 (ImageNet)"),
            ("randvit",  C_GRAY,   ":",  "ViT (random init)")]:
        r = pd.read_csv(os.path.join(C.RES_CSV, f"roc_curve_{name}.csv"))
        au = s.loc[name].auroc
        ax.plot(r.fpr, r.tpr, color=color, lw=2.4, ls=hatch_ls,
                label=f"{disp}   AUROC = {au:.3f}")
    ax.plot([0, 1], [0, 1], color="black", lw=1, ls="--", alpha=0.5)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    panel_tag(ax, "ROC curves — PatchCore by backbone")
    ax.grid(True, linestyle=":", alpha=0.45); ax.set_axisbelow(True)
    ax.legend(loc="lower right", frameon=True, framealpha=0.95)
    fig.tight_layout(); save(fig, "roc_curves")


# --------------------------------------------------------------------------- #
# 5. Score distributions (by taxon and by health)
# --------------------------------------------------------------------------- #
def fig_score_dist():
    s = pd.read_csv(os.path.join(C.RES_CSV,
                                 "per_image_scores_dinov3.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    ax = axes[0]
    bins = np.linspace(s.anomaly_score.min(), s.anomaly_score.max(), 36)
    for taxon in HEALTHY + STRESSED:
        sub = s[s.taxon == taxon].anomaly_score
        ax.hist(sub, bins=bins, density=True, histtype="stepfilled",
                color=TAX_COLORS[taxon], alpha=0.35, edgecolor=TAX_COLORS[taxon],
                lw=1.6, label=taxon)
    ax.set_xlabel("Anomaly score"); ax.set_ylabel("Density")
    panel_tag(ax, "(a) Score distribution by taxon")
    ax.grid(True, linestyle=":", alpha=0.45); ax.set_axisbelow(True)
    ax.legend(ncol=2, frameon=True, framealpha=0.95)

    ax = axes[1]
    for lab, color, hatch, name in [(0, C_HEALTHY, H1, "Healthy"),
                                    (1, C_STRESSED, H2, "Stressed")]:
        ax.hist(s[s.label == lab].anomaly_score, bins=bins, density=True,
                color=color, alpha=0.55, edgecolor="black", linewidth=0.8,
                hatch=hatch, label=name)
    ax.set_xlabel("Anomaly score"); ax.set_ylabel("Density")
    panel_tag(ax, "(b) Healthy vs stressed (aggregated)")
    ax.grid(True, linestyle=":", alpha=0.45); ax.set_axisbelow(True)
    ax.legend(frameon=True, framealpha=0.95)

    fig.tight_layout(); save(fig, "score_distributions")


# --------------------------------------------------------------------------- #
# 6. WWTP with vs without anomaly detection (2x2)
# --------------------------------------------------------------------------- #
def fig_wwtp():
    ts = pd.read_csv(os.path.join(C.RES_CSV, "wwtp_simulation_timeseries.csv"))
    none = ts[ts.policy == "none"].reset_index(drop=True)
    ad = ts[ts.policy == "ad"].reset_index(drop=True)
    summ = pd.read_csv(os.path.join(C.RES_CSV, "wwtp_impact_summary.csv"))
    d_none = int(summ.iloc[0].action_day)
    d_ad = int(summ.iloc[1].action_day)
    d = none.day; D0, DUR, LIM = 20, 18, 30.0

    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    # (a) sludge health
    ax = axes[0, 0]
    ax.axvspan(D0, D0 + DUR, color="grey", alpha=0.12, label="Disturbance")
    ax.plot(d, none.health, color=C_PEACH, lw=2.3, label="Without AD (reactive)")
    ax.plot(d, ad.health,   color=C_DINOV3, lw=2.3, label="With AD (proactive)")
    ax.axvline(d_none, color=C_PEACH, ls="--", lw=1.2)
    ax.axvline(d_ad,   color=C_DINOV3, ls="--", lw=1.2)
    ax.set_xlabel("Day"); ax.set_ylabel("Sludge health  (1 = healthy)")
    ax.set_ylim(0, 1.0); panel_tag(ax, "(a) Sludge health (microfauna state)")
    ax.grid(True, linestyle=":", alpha=0.45); ax.set_axisbelow(True)
    ax.legend(loc="lower right", frameon=True, framealpha=0.95)

    # (b) detector signal
    ax = axes[0, 1]
    ax.plot(d, ad.flagged_frac, color=C_DINOV3, lw=2.2,
            label="Detector flagged-stressed fraction")
    ax.plot(d, 1 - ad.health, color="black", lw=1.4, ls=":",
            label="True stressed fraction")
    ax.axhline(0.45, color="red", ls="--", lw=1.2, label="Alarm level")
    ax.axvline(d_ad, color=C_DINOV3, ls="--", lw=1.2,
               label=f"AD acts (day {d_ad})")
    ax.set_xlabel("Day"); ax.set_ylabel("Fraction"); ax.set_ylim(0, 1)
    panel_tag(ax, "(b) Detector signal (TPR=0.68, FPR=0.24)")
    ax.grid(True, linestyle=":", alpha=0.45); ax.set_axisbelow(True)
    ax.legend(loc="upper right", frameon=True, framealpha=0.95, fontsize=11)

    # (c) effluent TSS
    ax = axes[1, 0]
    ax.plot(d, none.eff_tss, color=C_PEACH, lw=2.3, label="Without AD")
    ax.plot(d, ad.eff_tss,   color=C_DINOV3, lw=2.3, label="With AD")
    ax.axhline(LIM, color="black", lw=1.3, ls="--", label="Discharge limit")
    ax.fill_between(d, LIM, none.eff_tss, where=none.eff_tss > LIM,
                    color=C_PEACH, alpha=0.30)
    ax.fill_between(d, LIM, ad.eff_tss, where=ad.eff_tss > LIM,
                    color=C_DINOV3, alpha=0.30)
    ax.set_xlabel("Day"); ax.set_ylabel("Effluent TSS (mg/L)")
    panel_tag(ax, "(c) Effluent quality")
    ax.grid(True, linestyle=":", alpha=0.45); ax.set_axisbelow(True)
    ax.legend(loc="upper right", frameon=True, framealpha=0.95)

    # (d) cumulative energy
    ax = axes[1, 1]
    ax.plot(d, np.cumsum(none.energy_kwh), color=C_PEACH, lw=2.3,
            label="Without AD")
    ax.plot(d, np.cumsum(ad.energy_kwh), color=C_DINOV3, lw=2.3,
            label="With AD")
    saved = np.cumsum(none.energy_kwh).iloc[-1] - np.cumsum(ad.energy_kwh).iloc[-1]
    ax.fill_between(d, np.cumsum(ad.energy_kwh), np.cumsum(none.energy_kwh),
                    color="#FFD166", alpha=0.35, label=f"Saved ≈ {saved:.0f} kWh")
    ax.set_xlabel("Day"); ax.set_ylabel("Cumulative aeration energy (kWh)")
    panel_tag(ax, "(d) Energy cost")
    ax.grid(True, linestyle=":", alpha=0.45); ax.set_axisbelow(True)
    ax.legend(loc="lower right", frameon=True, framealpha=0.95)

    fig.suptitle("Microfauna anomaly detection as a WWTP component  —  with vs "
                 "without", fontsize=16, fontweight="bold", y=1.005)
    fig.tight_layout(); save(fig, "wwtp_with_vs_without_ad")


# --------------------------------------------------------------------------- #
# 7. Object detection vs the paper (grouped bars per taxon)
# --------------------------------------------------------------------------- #
def fig_od_vs_paper():
    df = pd.read_csv(os.path.join(C.RES_CSV, "od_dinov3_per_class_ap.csv"))
    df = df[df.taxon != "OVERALL"].copy()
    x = np.arange(len(df)); w = 0.38
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(x - w / 2, df.dinov3_AP50, w, color=C_DINOV3, hatch=H1,
           edgecolor="black", linewidth=0.7,
           label="Frozen DINOv3 + head (ours)")
    ax.bar(x + w / 2, df.paper_mAP50, w, color=C_PEACH, hatch=H2,
           edgecolor="black", linewidth=0.7,
           label="MD-AS-2025 paper (YOLOv5/v8)")
    for i, (a, b) in enumerate(zip(df.dinov3_AP50, df.paper_mAP50)):
        ax.text(i - w / 2, a + 0.012, f"{a:.2f}", ha="center", fontsize=10,
                fontweight="bold")
        ax.text(i + w / 2, b + 0.012, f"{b:.2f}", ha="center", fontsize=10,
                fontweight="bold")
    o = pd.read_csv(os.path.join(C.RES_CSV, "od_dinov3_per_class_ap.csv"))
    o_ours = float(o[o.taxon == "OVERALL"].dinov3_AP50.iloc[0])
    o_paper = float(o[o.taxon == "OVERALL"].paper_mAP50.iloc[0])
    ax.axhline(o_ours, color=C_DINOV3, lw=1.4, ls="--",
               label=f"our overall mAP@0.5 = {o_ours:.3f}")
    ax.axhline(o_paper, color=C_PEACH, lw=1.4, ls="--",
               label=f"paper overall mAP@0.5 = {o_paper:.3f}")
    ax.set_xticks(x); ax.set_xticklabels(df.taxon, rotation=15)
    ax.set_ylabel("AP@0.5"); ax.set_ylim(0, 1.05)
    panel_tag(ax, "Object detection — per-taxon AP@0.5 vs the dataset paper")
    style_axes(ax)
    ax.legend(loc="upper right", frameon=True, framealpha=0.95, fontsize=11)
    fig.tight_layout(); save(fig, "od_dinov3_vs_paper")


# --------------------------------------------------------------------------- #
# 8. Detector performance: confusion matrix / F1-conf / PR  (paper Fig.5 style)
# --------------------------------------------------------------------------- #
def fig_detector_performance():
    cm = pd.read_csv(os.path.join(C.RES_CSV, "paper_fig5_confusion_matrix.csv"),
                     index_col=0)
    labels = list(cm.index)
    M = cm.values
    Mn = M / np.clip(M.sum(1, keepdims=True), 1, None)

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(Mn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_yticklabels(labels)
    for r in range(len(labels)):
        for c in range(len(labels)):
            if M[r, c]:
                ax.text(c, r, f"{Mn[r,c]:.2f}", ha="center", va="center",
                        fontsize=10, fontweight="bold",
                        color="white" if Mn[r, c] > 0.5 else "black")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    panel_tag(ax, "Frozen-DINOv3 detector — normalised confusion matrix")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout(); save(fig, "paper_fig5_detection_performance")


# --------------------------------------------------------------------------- #
# 9. Pixel-dimension scatter per taxon (paper Fig.6 style)
# --------------------------------------------------------------------------- #
def fig_pixel_dims():
    fig, axes = plt.subplots(1, 5, figsize=(22, 4.5))
    for ax, taxon in zip(axes, HEALTHY + STRESSED):
        files = []
        for ext in C.IMG_EXTS:
            files += glob.glob(os.path.join(C.DATA_RAW, taxon, f"*{ext}"))
        ws, hs = [], []
        for fp in files:
            try:
                with Image.open(fp) as im:
                    w, h = im.size
                ws.append(w); hs.append(h)
            except Exception:
                continue
        ws = np.array(ws); hs = np.array(hs)
        ax.scatter(ws, hs, s=8, alpha=0.4, c=TAX_COLORS[taxon],
                   edgecolors="none")
        med = (int(np.median(ws)), int(np.median(hs)))
        ax.scatter([med[0]], [med[1]], s=120, marker="x", c="black", lw=2.2,
                   label=f"median {med[0]}x{med[1]}")
        ax.set_xlabel("Width (px)"); ax.set_ylabel("Length / height (px)")
        panel_tag(ax, f"{taxon}  (n={len(ws):,})")
        ax.grid(True, linestyle=":", alpha=0.45); ax.set_axisbelow(True)
        ax.legend(loc="upper right", fontsize=10, frameon=True, framealpha=0.95)
    fig.suptitle("Pixel dimensions of cropped microfauna (paper-style, cf. Fig. 6)",
                 fontsize=16, fontweight="bold", y=1.04)
    fig.tight_layout(); save(fig, "paper_fig6_pixel_dimensions")


# --------------------------------------------------------------------------- #
# 10. t-SNE / UMAP panel (research style)
# --------------------------------------------------------------------------- #
def fig_embedding_panels():
    for stem in ("tsne", "umap"):
        p = os.path.join(C.RES_TSNE, f"{stem}_coords.csv")
        if not os.path.exists(p):
            continue
        df = pd.read_csv(p)
        fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))

        ax = axes[0]
        for t in HEALTHY + STRESSED:
            m = df.taxon == t
            ax.scatter(df.x[m], df.y[m], s=12, alpha=0.55, c=TAX_COLORS[t],
                       edgecolors="none", label=t)
        ax.set_xticks([]); ax.set_yticks([])
        panel_tag(ax, f"(a) {stem.upper()} — by taxon")
        ax.legend(markerscale=2.2, loc="best", frameon=True, framealpha=0.95,
                  fontsize=11)

        ax = axes[1]
        for lab, col, name in [(0, C_HEALTHY, "Healthy (normal)"),
                               (1, C_STRESSED, "Stressed (anomaly)")]:
            m = df.label == lab
            ax.scatter(df.x[m], df.y[m], s=12, alpha=0.55, c=col,
                       edgecolors="none", label=name)
        ax.set_xticks([]); ax.set_yticks([])
        panel_tag(ax, f"(b) {stem.upper()} — by health state")
        ax.legend(markerscale=2.2, loc="best", frameon=True, framealpha=0.95)
        fig.suptitle(f"DINOv3 CLS embeddings — {stem.upper()} projection",
                     fontsize=16, fontweight="bold", y=1.02)
        fig.tight_layout()
        out = FIG / f"embedding_{stem}.png"
        fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.1)
        fig.savefig(TIF / f"embedding_{stem}.tiff", dpi=400, format="tiff",
                    bbox_inches="tight", pad_inches=0.1,
                    pil_kwargs={"compression": "tiff_lzw"})
        plt.close(fig)


# --------------------------------------------------------------------------- #
def main():
    for fn in (fig_dataset, fig_auroc, fig_per_taxon, fig_roc, fig_score_dist,
               fig_wwtp, fig_od_vs_paper, fig_detector_performance,
               fig_pixel_dims, fig_embedding_panels):
        try:
            print(f"-> {fn.__name__}")
            fn()
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"   FAILED: {e}")
    print(f"\nfigures regenerated in: {FIG}")
    print(f"TIFF (LZW) copies in:   {TIF}")


if __name__ == "__main__":
    main()
