"""
Reproduce the MD-AS-2025 paper's figure styles with OUR data / results.

  Fig. 3  -> paper_fig3_directory_structure.png   (dataset tree)
  Fig. 4  -> paper_fig4_annotated_examples.png    (annotated frames + class stats)
  Fig. 5  -> paper_fig5_detection_performance.png (confusion matrix, F1-conf, PR)
  Fig. 6  -> paper_fig6_pixel_dimensions.png      (length x width per taxon)

Fig. 5 retrains the lightweight detection head on the cached frozen-DINOv3
features (fast) so we can collect predictions for the confusion matrix / curves.
All figures are written to results/figures/.
"""
from __future__ import annotations
import os
import sys
import glob

os.environ.setdefault("DET_IMG", "448")          # must match the cached features

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
import common
import od_03_dinov3_detector as od    # reuse DetHead, helpers, constants

NAMES = od.NAMES
NC = od.NC
GRID = od.GRID
TAX_COLORS = ["#e76f51", "#43aa8b", "#90be6d", "#d62828", "#2a9d8f"]


# --------------------------------------------------------------------------- #
# Fig. 6 - pixel dimension scatter per taxon (from the cropped dataset)
# --------------------------------------------------------------------------- #
def fig6_pixel_dimensions():
    fig, axes = plt.subplots(1, 5, figsize=(20, 4.2))
    rows = []
    for ax, taxon, col in zip(axes, C.ALL_TAXA, TAX_COLORS):
        files = []
        for ext in C.IMG_EXTS:
            files += glob.glob(os.path.join(C.DATA_RAW, taxon, f"*{ext}"))
        ws, hs = [], []
        for fp in files:
            try:
                with Image.open(fp) as im:
                    w, h = im.size            # lazy: header only
                ws.append(w); hs.append(h)
            except Exception:
                continue
        ws = np.array(ws); hs = np.array(hs)
        ax.scatter(ws, hs, s=6, alpha=0.35, c=col, edgecolors="none")
        ax.set_title(f"{taxon}\n(n={len(ws)})", fontsize=11)
        ax.set_xlabel("width (px)"); ax.set_ylabel("length / height (px)")
        rows.append(dict(taxon=taxon, n=len(ws),
                         mean_w=round(float(ws.mean()), 1),
                         mean_h=round(float(hs.mean()), 1),
                         med_w=int(np.median(ws)), med_h=int(np.median(hs))))
    fig.suptitle("Pixel dimension distributions of cropped microfauna "
                 "(paper-style, cf. Fig. 6)", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(C.RES_FIG, "paper_fig6_pixel_dimensions.png"),
                dpi=140)
    plt.close(fig)
    pd.DataFrame(rows).to_csv(
        os.path.join(C.RES_CSV, "paper_fig6_pixel_dimension_stats.csv"),
        index=False)
    print("  Fig.6 pixel dimensions ->", "paper_fig6_pixel_dimensions.png")


# --------------------------------------------------------------------------- #
# Fig. 4 - annotated example frames with boxes + class statistics
# --------------------------------------------------------------------------- #
ABBR = {0: "Arcellinida", 1: "Digononta", 2: "Monogononta", 3: "Nematoda",
        4: "Hypotrichida"}
COUNTS = {"Arcellinida": 2808, "Digononta": 1352, "Monogononta": 669,
          "Nematoda": 560, "Hypotrichida": 432}


def fig4_annotated_examples():
    items = od.read_split("train") + od.read_split("val")
    # find one frame that contains each class
    chosen = {}
    for path, boxes in items:
        if len(boxes) == 0:
            continue
        cls_set = set(int(c) for c in boxes[:, 0])
        for c in cls_set:
            if c not in chosen and len(boxes) <= 4:   # readable frames
                chosen[c] = (path, boxes)
        if len(chosen) == NC:
            break

    fig = plt.figure(figsize=(18, 7))
    for i in range(NC):
        ax = fig.add_subplot(2, 3, i + 1)
        if i not in chosen:
            ax.axis("off"); continue
        path, boxes = chosen[i]
        img = Image.open(path).convert("RGB")
        W, H = img.size
        ax.imshow(img); ax.axis("off")
        ax.set_title(NAMES[i], fontsize=11)
        for c, cx, cy, w, h in boxes:
            x1 = (cx - w / 2) * W; y1 = (cy - h / 2) * H
            ax.add_patch(mpatches.Rectangle((x1, y1), w * W, h * H, fill=False,
                         edgecolor=TAX_COLORS[int(c)], lw=2))
            ax.text(x1, y1 - 6, NAMES[int(c)][:4], color="white", fontsize=8,
                    bbox=dict(facecolor=TAX_COLORS[int(c)], pad=0,
                              edgecolor="none"))
    # class-distribution bar (6th panel)
    ax = fig.add_subplot(2, 3, 6)
    tx = list(COUNTS.keys()); cnt = list(COUNTS.values())
    ax.bar(tx, cnt, color=["#e76f51", "#43aa8b", "#90be6d", "#d62828",
                           "#2a9d8f"])
    for j, v in enumerate(cnt):
        ax.text(j, v + 30, str(v), ha="center", fontsize=9)
    ax.set_title("Annotated objects per taxon"); ax.tick_params(axis="x",
                                                                rotation=25)
    ax.set_ylabel("# labelled objects")
    fig.suptitle("Annotated full-frame examples and class statistics "
                 "(paper-style, cf. Fig. 4)", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(C.RES_FIG, "paper_fig4_annotated_examples.png"),
                dpi=140)
    plt.close(fig)
    print("  Fig.4 annotated examples ->", "paper_fig4_annotated_examples.png")


# --------------------------------------------------------------------------- #
# Fig. 5 - detection performance (confusion matrix / F1-conf / PR)
# --------------------------------------------------------------------------- #
def _train_head(ftr, train_items):
    tgt = [od.build_targets(b) for _, b in train_items]
    obj_t = torch.from_numpy(np.stack([t[0] for t in tgt]))
    cls_t = torch.from_numpy(np.stack([t[1] for t in tgt]))
    box_t = torch.from_numpy(np.stack([t[2] for t in tgt]))
    head = od.DetHead()
    opt = torch.optim.Adam(head.parameters(), lr=2e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 80)
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(64.0))
    n = len(train_items); idx = np.arange(n); B = 16
    for ep in range(80):
        np.random.shuffle(idx); head.train()
        for s in range(0, n, B):
            bi = idx[s:s + B]
            x = torch.from_numpy(np.ascontiguousarray(ftr[bi])).float().permute(0, 3, 1, 2)
            o, cl, bo = head(x); o = o.squeeze(1)
            ot = obj_t[bi]; ct = cls_t[bi]; bt = box_t[bi]; mask = ot > 0.5
            lo = bce(o, ot)
            if mask.any():
                lc = F.cross_entropy(cl.permute(0, 2, 3, 1)[mask], ct[mask])
                lb = F.smooth_l1_loss(od.decode_boxes(bo)[mask], bt[mask])
            else:
                lc = torch.zeros(()); lb = torch.zeros(())
            opt.zero_grad(); (lo + lc + 5 * lb).backward(); opt.step()
        sched.step()
    torch.save(head.state_dict(), os.path.join(od.OD_DIR, "dethead.pt"))
    return head


def _val_predictions(head, fva, conf=0.01):
    """Return per-image lists of (cls, score, xyxy) after per-class NMS."""
    preds = []
    head.eval()
    with torch.no_grad():
        for i in range(fva.shape[0]):
            x = torch.from_numpy(np.ascontiguousarray(fva[i:i+1])).float().permute(0, 3, 1, 2)
            o, cl, bo = head(x)
            obj = torch.sigmoid(o.squeeze(1))[0]
            prob = torch.softmax(cl, 1)[0]
            boxes = od.decode_boxes(bo)[0].numpy()
            img_preds = []
            for c in range(NC):
                sc = (obj * prob[c]).numpy().reshape(-1)
                keep = sc > conf
                if not keep.any():
                    continue
                bx = od.cxcywh_to_xyxy(boxes.reshape(-1, 4)[keep]); scc = sc[keep]
                for j in od.nms(bx, scc, 0.5):
                    img_preds.append((c, float(scc[j]), bx[j]))
            preds.append(img_preds)
    return preds


def fig5_detection_performance(val_items, preds):
    gts = []
    for _, boxes in val_items:
        gts.append([(int(c), od.cxcywh_to_xyxy(np.array([[cx, cy, w, h]]))[0])
                    for c, cx, cy, w, h in boxes])

    # ---- confusion matrix at conf=0.25 (rows=true, +background) ---------- #
    CONF = 0.25
    cm = np.zeros((NC + 1, NC + 1), dtype=int)        # last index = background
    for ip, gt in zip(preds, gts):
        dets = [p for p in ip if p[1] >= CONF]
        used = [False] * len(dets)
        for tc, gbox in gt:
            best, bj = 0.5, -1
            for j, (pc, ps, pb) in enumerate(dets):
                if used[j]:
                    continue
                iou = od.iou_xyxy(gbox[None], pb[None])[0, 0]
                if iou >= best:
                    best, bj = iou, j
            if bj >= 0:
                cm[tc, dets[bj][0]] += 1; used[bj] = True
            else:
                cm[tc, NC] += 1                        # missed -> background
        for j, u in enumerate(used):
            if not u:
                cm[NC, dets[j][0]] += 1                # false positive

    cmn = cm / np.clip(cm.sum(1, keepdims=True), 1, None)
    labels = NAMES + ["background"]

    # ---- F1 vs confidence ----------------------------------------------- #
    threshs = np.linspace(0.02, 0.95, 40)
    f1s, precs, recs = [], [], []
    total_gt = sum(len(g) for g in gts)
    for t in threshs:
        tp = fp = 0
        for ip, gt in zip(preds, gts):
            dets = [p for p in ip if p[1] >= t]
            used = [False] * len(gt)
            for pc, ps, pb in sorted(dets, key=lambda z: -z[1]):
                m, mj = 0.5, -1
                for k, (tc, gbox) in enumerate(gt):
                    if used[k] or tc != pc:
                        continue
                    iou = od.iou_xyxy(gbox[None], pb[None])[0, 0]
                    if iou >= m:
                        m, mj = iou, k
                if mj >= 0:
                    tp += 1; used[mj] = True
                else:
                    fp += 1
        prec = tp / (tp + fp) if tp + fp else 0
        rec = tp / total_gt if total_gt else 0
        precs.append(prec); recs.append(rec)
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0)

    # ---- plot ------------------------------------------------------------ #
    fig, ax = plt.subplots(1, 3, figsize=(19, 5.5))
    im = ax[0].imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax[0].set_xticks(range(NC + 1)); ax[0].set_xticklabels(labels, rotation=45,
                                                           ha="right", fontsize=8)
    ax[0].set_yticks(range(NC + 1)); ax[0].set_yticklabels(labels, fontsize=8)
    for r in range(NC + 1):
        for c in range(NC + 1):
            if cm[r, c]:
                ax[0].text(c, r, f"{cmn[r,c]:.2f}", ha="center", va="center",
                           fontsize=7,
                           color="white" if cmn[r, c] > 0.5 else "black")
    ax[0].set_xlabel("predicted"); ax[0].set_ylabel("true")
    ax[0].set_title("(a) Confusion matrix (normalized, conf=0.25)")
    fig.colorbar(im, ax=ax[0], fraction=0.046)

    ax[1].plot(threshs, f1s, color="#264653", lw=2)
    bi = int(np.argmax(f1s))
    ax[1].axvline(threshs[bi], ls="--", c="r", lw=1)
    ax[1].set_title(f"(b) F1 vs confidence  (max F1={f1s[bi]:.2f} @ "
                    f"{threshs[bi]:.2f})")
    ax[1].set_xlabel("confidence"); ax[1].set_ylabel("F1"); ax[1].set_ylim(0, 1)

    order = np.argsort(recs)
    ax[2].plot(np.array(recs)[order], np.array(precs)[order], color="#e76f51",
               lw=2)
    ax[2].set_title("(c) Precision-Recall (all taxa)")
    ax[2].set_xlabel("recall"); ax[2].set_ylabel("precision")
    ax[2].set_xlim(0, 1); ax[2].set_ylim(0, 1)

    fig.suptitle("Frozen-DINOv3 detector performance (paper-style, cf. Fig. 5)",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(C.RES_FIG, "paper_fig5_detection_performance.png"),
                dpi=140)
    plt.close(fig)
    pd.DataFrame(cm, index=labels, columns=labels).to_csv(
        os.path.join(C.RES_CSV, "paper_fig5_confusion_matrix.csv"))
    print("  Fig.5 detection performance ->",
          "paper_fig5_detection_performance.png")


# --------------------------------------------------------------------------- #
# Fig. 3 - directory structure schematic
# --------------------------------------------------------------------------- #
def fig3_directory_structure():
    tree = (
        "MD-AS-2025\n"
        "|\n"
        "+-- Annotated Dataset (Dataset A)  -- 4,000 frames\n"
        "|     +-- train_val_test/images/   (.JPG)\n"
        "|     +-- train_val_test/labels/   (.xml  PASCAL-VOC)\n"
        "|\n"
        "+-- Cropped Dataset (Dataset B)    -- 14,257 single-object images\n"
        "      +-- Arcellinida/   (7,007)\n"
        "      +-- Digononta/     (4,780)\n"
        "      +-- Monogononta/   (1,207)\n"
        "      +-- Nematoda/      (   707)\n"
        "      +-- Hypotrichida/  (   556)\n"
    )
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.axis("off")
    ax.text(0.02, 0.98, tree, va="top", ha="left", family="monospace",
            fontsize=12)
    ax.set_title("MD-AS-2025 directory structure (paper-style, cf. Fig. 3)",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(C.RES_FIG, "paper_fig3_directory_structure.png"),
                dpi=140)
    plt.close(fig)
    print("  Fig.3 directory structure ->", "paper_fig3_directory_structure.png")


def main():
    C.ensure_dirs()
    common.set_seed()
    torch.set_num_threads(max(1, os.cpu_count() or 4))

    print("[Fig.3] directory structure")
    fig3_directory_structure()
    print("[Fig.4] annotated examples + class stats")
    fig4_annotated_examples()
    print("[Fig.6] pixel-dimension scatter (reads cropped image sizes)")
    fig6_pixel_dimensions()

    print("[Fig.5] detection performance (retrain head on cached features)")
    ftr = np.load(os.path.join(od.OD_DIR, "feat_train.npy"), mmap_mode="r")
    fva = np.load(os.path.join(od.OD_DIR, "feat_val.npy"), mmap_mode="r")
    train_items = od.read_split("train"); val_items = od.read_split("val")
    head = _train_head(ftr, train_items)
    preds = _val_predictions(head, fva)
    fig5_detection_performance(val_items, preds)
    print("done. paper-style figures in results/figures/")


if __name__ == "__main__":
    main()
