"""
Object detection with a FROZEN DINOv3 backbone.

Keeps the project on a single self-supervised backbone: DINOv3 ViT-S/16 is frozen
and used purely as a dense feature extractor; only a lightweight detection head is
trained.  Because the backbone never updates, its 28x28x384 patch features are
cached to disk once and the head then trains in seconds per epoch on CPU.

Head:    a per-cell anchor-free YOLO-style head on the DINOv3 patch grid
         (objectness + 5-way class + box).  GT boxes are assigned to the cell
         containing their centre.
Eval:    per-class AP@0.5 and mAP@0.5 (self-contained implementation), compared
         with the MD-AS-2025 paper's reported YOLOv5/v8 numbers.

Outputs:
    results/csv/od_dinov3_per_class_ap.csv
    results/csv/od_methods_comparison.csv          (DINOv3 vs YOLOv8 vs paper)
    results/figures/od_dinov3_vs_paper.png
    results/figures/od_dinov3_predictions.png
    data/processed/od/feat_{train,val}.npy         (cached frozen features)
"""
from __future__ import annotations
import os
import sys
import time

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

OD_DIR = os.path.join(C.DATA_PROCESSED, "od")
NAMES = ["Arcellinida", "Digononta", "Monogononta", "Nematoda", "Hypotrichida"]
NC = len(NAMES)

DET_IMG = int(os.environ.get("DET_IMG", 448))     # detection input resolution
GRID = DET_IMG // 16                               # DINOv3 patch grid (28)
EPOCHS = int(os.environ.get("DET_EPOCHS", 80))
LR = float(os.environ.get("DET_LR", 2e-3))
BATCH = int(os.environ.get("DET_BATCH", 16))
CONF_EVAL = 0.01                                   # low conf, AP integrates
NMS_IOU = 0.5

PAPER_MAP50 = {"Arcellinida": 0.958, "Digononta": 0.916, "Monogononta": 0.834,
               "Nematoda": 0.778, "Hypotrichida": 0.991}
PAPER_OVERALL = float(np.mean(list(PAPER_MAP50.values())))


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def read_split(name):
    paths = [l.strip() for l in open(os.path.join(OD_DIR, f"{name}.txt"))
             if l.strip()]
    items = []
    for p in paths:
        lbl = p.replace("/images/", "/labels/")
        lbl = os.path.splitext(lbl)[0] + ".txt"
        boxes = []
        if os.path.exists(lbl):
            for line in open(lbl):
                a = line.split()
                if len(a) == 5:
                    boxes.append([int(a[0])] + [float(v) for v in a[1:]])
        items.append((p, np.array(boxes, dtype=np.float32).reshape(-1, 5)))
    return items


_MEAN = torch.tensor(C.NORM_MEAN).view(3, 1, 1)
_STD = torch.tensor(C.NORM_STD).view(3, 1, 1)


def load_det_image(path):
    img = Image.open(path).convert("RGB").resize((DET_IMG, DET_IMG), Image.BICUBIC)
    arr = torch.from_numpy(np.asarray(img, np.float32) / 255.0).permute(2, 0, 1)
    return (arr - _MEAN) / _STD


def cache_features(items, name, bb):
    out = os.path.join(OD_DIR, f"feat_{name}.npy")
    if os.path.exists(out):
        print(f"  using cached {out}")
        return np.load(out, mmap_mode="r")
    n = len(items)
    feats = np.lib.format.open_memmap(out, mode="w+", dtype=np.float16,
                                      shape=(n, GRID, GRID, 384))
    t0 = time.time()
    for i in range(0, n, BATCH):
        chunk = items[i:i + BATCH]
        batch = torch.stack([load_det_image(p) for p, _ in chunk])
        with torch.no_grad():
            grid, _ = bb.extract(batch)            # (b, GRID, GRID, 384)
        feats[i:i + len(chunk)] = grid.cpu().numpy().astype(np.float16)
        if (i // BATCH) % 20 == 0:
            print(f"  [{name}] {i+len(chunk)}/{n} ({time.time()-t0:.0f}s)")
    feats.flush()
    return np.load(out, mmap_mode="r")


# --------------------------------------------------------------------------- #
# Targets
# --------------------------------------------------------------------------- #
def build_targets(boxes):
    """boxes: (k,5) [cls,cx,cy,w,h] normalised -> grid targets."""
    obj = np.zeros((GRID, GRID), np.float32)
    cls = np.zeros((GRID, GRID), np.int64)
    box = np.zeros((GRID, GRID, 4), np.float32)
    area = np.full((GRID, GRID), -1.0, np.float32)
    for c, cx, cy, w, h in boxes:
        col = min(int(cx * GRID), GRID - 1)
        row = min(int(cy * GRID), GRID - 1)
        a = w * h
        if a > area[row, col]:                     # keep the larger object
            area[row, col] = a
            obj[row, col] = 1.0
            cls[row, col] = int(c)
            box[row, col] = [cx, cy, w, h]
    return obj, cls, box


# --------------------------------------------------------------------------- #
# Head
# --------------------------------------------------------------------------- #
class DetHead(nn.Module):
    def __init__(self, dim=384, nc=NC):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(dim, 256, 1), nn.GroupNorm(32, 256), nn.SiLU(),
            nn.Conv2d(256, 256, 1), nn.GroupNorm(32, 256), nn.SiLU())
        self.obj = nn.Conv2d(256, 1, 1)
        self.cls = nn.Conv2d(256, nc, 1)
        self.box = nn.Conv2d(256, 4, 1)

    def forward(self, x):                           # x: (B,384,G,G)
        f = self.stem(x)
        return self.obj(f), self.cls(f), self.box(f)


def decode_boxes(box_logit):
    """box_logit (B,4,G,G) -> normalised cxcywh (B,G,G,4)."""
    b = box_logit.permute(0, 2, 3, 1)               # (B,G,G,4)
    tx, ty, tw, th = b.unbind(-1)
    cols = torch.arange(GRID).view(1, 1, GRID).to(b)
    rows = torch.arange(GRID).view(1, GRID, 1).to(b)
    cx = (cols + torch.sigmoid(tx)) / GRID
    cy = (rows + torch.sigmoid(ty)) / GRID
    w = torch.sigmoid(tw)
    h = torch.sigmoid(th)
    return torch.stack([cx, cy, w, h], -1)


# --------------------------------------------------------------------------- #
# Geometry / metrics
# --------------------------------------------------------------------------- #
def cxcywh_to_xyxy(b):
    cx, cy, w, h = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], -1)


def iou_xyxy(a, b):
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    iw = np.clip(ix2 - ix1, 0, None); ih = np.clip(iy2 - iy1, 0, None)
    inter = iw * ih
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + ab[None, :] - inter + 1e-9)


def nms(boxes, scores, iou_thr):
    order = scores.argsort()[::-1]
    keep = []
    while len(order):
        i = order[0]; keep.append(i)
        if len(order) == 1:
            break
        ious = iou_xyxy(boxes[i:i + 1], boxes[order[1:]])[0]
        order = order[1:][ious < iou_thr]
    return keep


def ap_per_class(preds, gts, cls_id, iou_thr=0.5):
    """preds: list of (img_id, score, box_xyxy); gts: dict img_id->list box_xyxy."""
    P = sorted(preds, key=lambda x: -x[1])
    npos = sum(len(v) for v in gts.values())
    if npos == 0:
        return float("nan")
    matched = {k: np.zeros(len(v), bool) for k, v in gts.items()}
    tp = np.zeros(len(P)); fp = np.zeros(len(P))
    for i, (img, sc, bx) in enumerate(P):
        g = gts.get(img, [])
        if len(g) == 0:
            fp[i] = 1; continue
        ious = iou_xyxy(bx[None], np.array(g))[0]
        j = int(ious.argmax())
        if ious[j] >= iou_thr and not matched[img][j]:
            tp[i] = 1; matched[img][j] = True
        else:
            fp[i] = 1
    tpc = np.cumsum(tp); fpc = np.cumsum(fp)
    rec = tpc / (npos + 1e-9)
    prec = tpc / (tpc + fpc + 1e-9)
    # all-point AP
    mrec = np.concatenate([[0], rec, [1]])
    mpre = np.concatenate([[0], prec, [0]])
    for k in range(len(mpre) - 1, 0, -1):
        mpre[k - 1] = max(mpre[k - 1], mpre[k])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


# --------------------------------------------------------------------------- #
# Train / evaluate
# --------------------------------------------------------------------------- #
def main():
    C.ensure_dirs()
    common.set_seed()
    torch.set_num_threads(max(1, os.cpu_count() or 4))
    print(f"DINOv3 detector  img={DET_IMG} grid={GRID} epochs={EPOCHS}")

    bb = common.Backbone("dinov3")
    train_items = read_split("train")
    val_items = read_split("val")
    print(f"train {len(train_items)} | val {len(val_items)}")

    ftr = cache_features(train_items, "train", bb)
    fva = cache_features(val_items, "val", bb)

    # precompute train targets
    tgt = [build_targets(b) for _, b in train_items]
    obj_t = torch.from_numpy(np.stack([t[0] for t in tgt]))         # (N,G,G)
    cls_t = torch.from_numpy(np.stack([t[1] for t in tgt]))
    box_t = torch.from_numpy(np.stack([t[2] for t in tgt]))
    pos_per_img = obj_t.flatten(1).sum(1).clamp(min=1)

    head = DetHead()
    opt = torch.optim.Adam(head.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(64.0), reduction="none")

    n = len(train_items)
    idx_all = np.arange(n)
    print("training head ...")
    for ep in range(EPOCHS):
        np.random.shuffle(idx_all)
        head.train(); tot = 0.0
        for s in range(0, n, BATCH):
            bi = idx_all[s:s + BATCH]
            x = torch.from_numpy(np.ascontiguousarray(ftr[bi])).float()
            x = x.permute(0, 3, 1, 2)                # (b,384,G,G)
            o, cl, bo = head(x)
            o = o.squeeze(1)                          # (b,G,G)
            ot = obj_t[bi]; ct = cls_t[bi]; bt = box_t[bi]
            mask = ot > 0.5
            # objectness
            lo = bce(o, ot).mean()
            # classification on positives
            if mask.any():
                cl_p = cl.permute(0, 2, 3, 1)[mask]   # (P,nc)
                lc = F.cross_entropy(cl_p, ct[mask])
                pred_box = decode_boxes(bo)[mask]      # (P,4)
                lb = F.smooth_l1_loss(pred_box, bt[mask])
            else:
                lc = torch.zeros(()); lb = torch.zeros(())
            loss = lo + lc + 5.0 * lb
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss)
        sched.step()
        if ep % 10 == 0 or ep == EPOCHS - 1:
            print(f"  epoch {ep:3d}  loss={tot/ (n//BATCH+1):.4f}")

    # ---- evaluate on val -------------------------------------------------- #
    head.eval()
    preds_by_cls = {c: [] for c in range(NC)}
    gts_by_cls = {c: {} for c in range(NC)}
    for c in range(NC):
        for i in range(len(val_items)):
            gts_by_cls[c][i] = []
    for i, (_, boxes) in enumerate(val_items):
        for c, cx, cy, w, h in boxes:
            xyxy = cxcywh_to_xyxy(np.array([[cx, cy, w, h]]))[0]
            gts_by_cls[int(c)][i].append(xyxy)

    with torch.no_grad():
        for s in range(0, len(val_items), BATCH):
            bi = list(range(s, min(s + BATCH, len(val_items))))
            x = torch.from_numpy(np.ascontiguousarray(fva[bi])).float().permute(0, 3, 1, 2)
            o, cl, bo = head(x)
            obj = torch.sigmoid(o.squeeze(1))         # (b,G,G)
            prob = torch.softmax(cl, 1)               # (b,nc,G,G)
            boxes = decode_boxes(bo).numpy()          # (b,G,G,4)
            score = (obj.unsqueeze(1) * prob).numpy()  # (b,nc,G,G)
            for k, img_id in enumerate(bi):
                for c in range(NC):
                    sc = score[k, c].reshape(-1)
                    keep = sc > CONF_EVAL
                    if not keep.any():
                        continue
                    bx = cxcywh_to_xyxy(boxes[k].reshape(-1, 4)[keep])
                    scc = sc[keep]
                    kk = nms(bx, scc, NMS_IOU)
                    for j in kk:
                        preds_by_cls[c].append((img_id, float(scc[j]), bx[j]))

    rows = []
    aps = []
    for c in range(NC):
        ap = ap_per_class(preds_by_cls[c], gts_by_cls[c], c)
        aps.append(ap)
        rows.append(dict(taxon=NAMES[c], dinov3_AP50=ap,
                         paper_mAP50=PAPER_MAP50[NAMES[c]],
                         diff=ap - PAPER_MAP50[NAMES[c]]))
    mAP = float(np.nanmean(aps))
    perclass = pd.DataFrame(rows)
    perclass.loc["overall"] = ["OVERALL", mAP, PAPER_OVERALL, mAP - PAPER_OVERALL]
    perclass.to_csv(os.path.join(C.RES_CSV, "od_dinov3_per_class_ap.csv"),
                    index=False)
    print(perclass.to_string(index=False))
    print(f"\nDINOv3 detector mAP@0.5 = {mAP:.3f}  (paper {PAPER_OVERALL:.3f})")

    _comparison_figure(perclass)
    _methods_table(mAP)
    _sample_predictions(head, val_items, fva)


def _comparison_figure(perclass):
    pc = perclass.iloc[:-1]
    x = np.arange(len(pc)); w = 0.38
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w / 2, pc.dinov3_AP50, w, label="Frozen DINOv3 + head",
           color="#264653")
    ax.bar(x + w / 2, pc.paper_mAP50, w, label="MD-AS-2025 paper (YOLOv5/8)",
           color="#e9c46a")
    ax.set_xticks(x); ax.set_xticklabels(pc.taxon, rotation=15)
    ax.set_ylim(0, 1.05); ax.set_ylabel("AP@0.5")
    ax.set_title("Object detection: frozen DINOv3 vs MD-AS-2025 paper")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(C.RES_FIG, "od_dinov3_vs_paper.png"), dpi=140)
    plt.close(fig)


def _methods_table(dinov3_map):
    rows = [dict(method="MD-AS-2025 paper (YOLOv5/v8)", mAP50=PAPER_OVERALL,
                 note="reported in dataset paper"),
            dict(method="Frozen DINOv3 + light head (ours)", mAP50=dinov3_map,
                 note=f"CPU, {EPOCHS}ep head-only, img{DET_IMG}")]
    yolo = os.path.join(C.RES_CSV, "od_metrics_overall.csv")
    if os.path.exists(yolo):
        y = pd.read_csv(yolo).iloc[0]
        rows.append(dict(method=f"YOLOv8 reproduction ({int(y.epochs)}ep)",
                         mAP50=float(y.mAP50), note="CPU full fine-tune"))
    pd.DataFrame(rows).to_csv(
        os.path.join(C.RES_CSV, "od_methods_comparison.csv"), index=False)


def _sample_predictions(head, val_items, fva, conf=0.3):
    head.eval()
    sel = list(range(min(8, len(val_items))))
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.ravel()
    colors = plt.cm.tab10(np.linspace(0, 1, NC))
    with torch.no_grad():
        for ax, i in zip(axes, sel):
            img = Image.open(val_items[i][0]).convert("RGB").resize(
                (DET_IMG, DET_IMG))
            ax.imshow(img); ax.axis("off")
            x = torch.from_numpy(np.ascontiguousarray(fva[i:i+1])).float().permute(0, 3, 1, 2)
            o, cl, bo = head(x)
            obj = torch.sigmoid(o.squeeze(1))[0]
            prob = torch.softmax(cl, 1)[0]
            boxes = decode_boxes(bo)[0].numpy()
            for c in range(NC):
                sc = (obj * prob[c]).numpy().reshape(-1)
                keep = sc > conf
                if not keep.any():
                    continue
                bx = cxcywh_to_xyxy(boxes.reshape(-1, 4)[keep]) * DET_IMG
                scc = sc[keep]
                for j in nms(bx, scc, NMS_IOU):
                    x1, y1, x2, y2 = bx[j]
                    ax.add_patch(mpatches.Rectangle(
                        (x1, y1), x2 - x1, y2 - y1, fill=False,
                        edgecolor=colors[c], lw=2))
                    ax.text(x1, y1 - 2, f"{NAMES[c][:4]} {scc[j]:.2f}",
                            color="white", fontsize=7,
                            bbox=dict(facecolor=colors[c], pad=0, edgecolor="none"))
    fig.suptitle("Frozen-DINOv3 detector predictions (val)")
    fig.tight_layout()
    fig.savefig(os.path.join(C.RES_FIG, "od_dinov3_predictions.png"), dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
