"""
Stage 2 - PatchCore anomaly detection with a frozen backbone.

For each backbone (DINOv3 ViT-S/16, random-init ViT, ImageNet ResNet-50):

  TRAIN  : extract patch embeddings from HEALTHY training images, build a
           greedy-coreset memory bank (PatchCore).
  TEST   : score every test image by the max patch-to-bank nearest-neighbour
           distance; healthy = normal (0), stressed = anomaly (1).
  EVAL   : image-level AUROC / AP overall and per stressed taxon, plus
           threshold-based metrics (F1, accuracy, specificity, recall).

Artifacts (results/csv, data/processed):
  per_image_scores_<backbone>.csv      every test image, score, label, taxon
  roc_curve_<backbone>.csv             ROC points
  metrics_summary.csv                  one row per backbone (appended)
  per_taxon_auroc.csv                  healthy-vs-each-stressed-taxon AUROC
  threshold_metrics.csv                operating-point metrics per backbone
  memorybank_<backbone>.npy            the coreset memory bank
  cls_embeddings_dinov3.npz            CLS descriptors (for t-SNE / UMAP)
  heatmap_cache_dinov3.npz             patch score maps for example images
"""
from __future__ import annotations
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
import common


# --------------------------------------------------------------------------- #
# Feature extraction over a list of manifest rows
# --------------------------------------------------------------------------- #
def iter_batches(df: pd.DataFrame):
    paths = df["path"].tolist()
    for chunk in common.batched(paths, C.BATCH_SIZE):
        batch = torch.stack([common.load_image(os.path.join(C.ROOT, p))
                             for p in chunk])
        yield chunk, batch


def build_memory_bank(bb: common.Backbone, train_df: pd.DataFrame, rng):
    """Single pass over training images -> reservoir-sampled patch pool ->
    greedy coreset -> (memory_bank [M,D], train_cls [N,D])."""
    pool = None
    pool_fill = 0
    seen = 0
    train_cls = []
    t0 = time.time()
    for i, (chunk, batch) in enumerate(iter_batches(train_df)):
        grid, cls = bb.extract(batch)
        grid = common.neighbourhood_aggregate(grid)
        b, h, w, d = grid.shape
        feats = grid.reshape(b * h * w, d).cpu().numpy().astype(np.float32)
        feats = common.l2norm(feats)
        train_cls.append(cls.cpu().numpy().astype(np.float32))

        if pool is None:
            pool = np.empty((C.POOL_MAX, d), dtype=np.float32)
        # reservoir sampling into a fixed-size pool
        for v in feats:
            if pool_fill < C.POOL_MAX:
                pool[pool_fill] = v
                pool_fill += 1
            else:
                j = int(rng.integers(seen + 1))
                if j < C.POOL_MAX:
                    pool[j] = v
            seen += 1
        if (i + 1) % 20 == 0:
            print(f"    train batch {i+1}: pooled={pool_fill} "
                  f"({time.time()-t0:.0f}s)")
    pool = pool[:pool_fill]
    train_cls = np.concatenate(train_cls, 0)

    # greedy coreset in a random-projected space (PatchCore speed-up)
    if pool.shape[1] > C.CORESET_PROJ_DIM:
        proj = rng.standard_normal((pool.shape[1], C.CORESET_PROJ_DIM)
                                   ).astype(np.float32)
        proj /= np.linalg.norm(proj, axis=0, keepdims=True)
        pool_proj = pool @ proj
    else:
        pool_proj = pool
    idx = common.greedy_coreset(pool_proj, C.CORESET_SIZE, seed=C.SEED)
    bank = pool[idx]
    print(f"    memory bank: {bank.shape[0]} x {bank.shape[1]} "
          f"(from pool {pool.shape[0]})")
    return bank, train_cls


def score_images(bb: common.Backbone, test_df: pd.DataFrame, bank: np.ndarray,
                 keep_maps_for: set | None = None):
    """Score every test image.  Returns (scores [N], test_cls [N,D],
    score_maps dict{path: (h,w) array} for requested paths)."""
    bank_t = torch.from_numpy(bank)            # (M, D), L2-normalised
    scores = []
    test_cls = []
    maps = {}
    t0 = time.time()
    for i, (chunk, batch) in enumerate(iter_batches(test_df)):
        grid, cls = bb.extract(batch)
        grid = common.neighbourhood_aggregate(grid)
        b, h, w, d = grid.shape
        test_cls.append(cls.cpu().numpy().astype(np.float32))
        feats = grid.reshape(b, h * w, d)
        feats = feats / (feats.norm(dim=-1, keepdim=True) + 1e-8)
        # cosine similarity -> nearest-neighbour euclidean distance
        sims = torch.matmul(feats, bank_t.t())          # (b, hw, M)
        max_sim, _ = sims.max(dim=-1)                    # (b, hw)
        patch_dist = torch.sqrt(torch.clamp(2.0 - 2.0 * max_sim, min=0.0))
        img_score = patch_dist.max(dim=1).values         # (b,)
        scores.append(img_score.cpu().numpy())
        if keep_maps_for:
            dm = patch_dist.reshape(b, h, w).cpu().numpy()
            for k, p in enumerate(chunk):
                if p in keep_maps_for:
                    maps[p] = dm[k]
        if (i + 1) % 30 == 0:
            print(f"    test batch {i+1}/{(len(test_df)+C.BATCH_SIZE-1)//C.BATCH_SIZE}"
                  f" ({time.time()-t0:.0f}s)")
    scores = np.concatenate(scores)
    test_cls = np.concatenate(test_cls, 0)
    return scores, test_cls, maps


def cls_knn_scores(train_cls: np.ndarray, test_cls: np.ndarray, k: int = 5):
    """Image-level baseline: mean distance to k nearest healthy CLS vectors."""
    tr = common.l2norm(train_cls)
    te = common.l2norm(test_cls)
    sims = te @ tr.T                                     # (Nte, Ntr)
    topk = np.sort(sims, axis=1)[:, -k:]                 # k largest sims
    dist = np.sqrt(np.clip(2.0 - 2.0 * topk, 0, None)).mean(1)
    return dist


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def threshold_metrics(y, s, thr):
    pred = (s >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    acc = (tp + tn) / len(y)
    return dict(threshold=float(thr), TP=tp, TN=tn, FP=fp, FN=fn,
                precision=prec, recall=rec, specificity=spec, f1=f1,
                accuracy=acc)


def evaluate_backbone(name, test_df, scores):
    y = test_df["label"].to_numpy()
    auroc = roc_auc_score(y, scores)
    ap = average_precision_score(y, scores)

    # operating points
    healthy = scores[y == 0]
    thr_p95 = float(np.percentile(healthy, 95))   # unsupervised: 95th pct healthy
    fpr, tpr, thr = roc_curve(y, scores)
    j = tpr - fpr
    thr_youden = float(thr[int(np.argmax(j))])

    m_p95 = threshold_metrics(y, scores, thr_p95)
    m_p95["operating_point"] = "healthy_p95"
    m_y = threshold_metrics(y, scores, thr_youden)
    m_y["operating_point"] = "youden_J"
    for m in (m_p95, m_y):
        m["backbone"] = name

    roc_df = pd.DataFrame(dict(fpr=fpr, tpr=tpr, threshold=thr))
    return auroc, ap, [m_p95, m_y], roc_df


def per_taxon_auroc(name, test_df, scores):
    y = test_df["label"].to_numpy()
    healthy_mask = y == 0
    h_scores = scores[healthy_mask]
    rows = []
    for taxon in C.STRESSED_TAXA:
        m = test_df["taxon"].to_numpy() == taxon
        s = np.concatenate([h_scores, scores[m]])
        lab = np.concatenate([np.zeros(len(h_scores)), np.ones(int(m.sum()))])
        rows.append(dict(backbone=name, comparison=f"healthy_vs_{taxon}",
                         n_healthy=int(healthy_mask.sum()),
                         n_stressed=int(m.sum()),
                         auroc=float(roc_auc_score(lab, s)),
                         ap=float(average_precision_score(lab, s))))
    return rows


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    C.ensure_dirs()
    common.set_seed()
    torch.set_num_threads(max(1, os.cpu_count() or 4))

    manifest = pd.read_csv(os.path.join(C.RES_CSV, "manifest.csv"))
    train_df = manifest[manifest.split == "train"].reset_index(drop=True)
    test_df = manifest[manifest.split == "test"].reset_index(drop=True)
    print(f"train(healthy)={len(train_df)}  test={len(test_df)} "
          f"(healthy={int((test_df.label==0).sum())}, "
          f"stressed={int((test_df.label==1).sum())})")

    # example images to cache patch-score maps for (DINOv3 heatmaps later)
    keep_maps = set()
    for taxon in C.ALL_TAXA:
        sub = test_df[test_df.taxon == taxon]["path"].tolist()[:C.N_VIS_PER_TAXON]
        keep_maps.update(sub)

    summary_rows, taxon_rows, thr_rows = [], [], []

    for name in C.BACKBONES:
        print(f"\n=== backbone: {name} ===")
        rng = np.random.default_rng(C.SEED)
        bb = common.Backbone(name)

        t0 = time.time()
        bank, train_cls = build_memory_bank(bb, train_df, rng)
        np.save(os.path.join(C.DATA_PROCESSED, f"memorybank_{name}.npy"), bank)

        keep = keep_maps if name == C.PRIMARY_BACKBONE else None
        scores, test_cls, maps = score_images(bb, test_df, bank, keep)
        print(f"    feature+score time: {time.time()-t0:.0f}s")

        # ---- per-image scores csv ----------------------------------------- #
        pi = test_df.copy()
        pi["anomaly_score"] = scores
        pi.to_csv(os.path.join(C.RES_CSV, f"per_image_scores_{name}.csv"),
                  index=False)

        # ---- metrics ------------------------------------------------------ #
        auroc, ap, thr_metrics, roc_df = evaluate_backbone(name, test_df, scores)
        roc_df.to_csv(os.path.join(C.RES_CSV, f"roc_curve_{name}.csv"),
                      index=False)
        thr_rows.extend(thr_metrics)
        taxon_rows.extend(per_taxon_auroc(name, test_df, scores))
        f1 = max(m["f1"] for m in thr_metrics)
        summary_rows.append(dict(backbone=name, method="patchcore_patch",
                                 auroc=auroc, ap=ap, best_f1=f1,
                                 bank_size=int(bank.shape[0]),
                                 feat_dim=int(bank.shape[1])))
        print(f"    AUROC={auroc:.4f}  AP={ap:.4f}  bestF1={f1:.4f}")

        # ---- DINOv3 extras: CLS embeddings + heatmap cache + cls-knn ------- #
        if name == C.PRIMARY_BACKBONE:
            np.savez_compressed(
                os.path.join(C.DATA_PROCESSED, "cls_embeddings_dinov3.npz"),
                train_cls=train_cls, test_cls=test_cls,
                test_paths=np.array(test_df["path"]),
                test_taxon=np.array(test_df["taxon"]),
                test_label=test_df["label"].to_numpy())
            if maps:
                np.savez_compressed(
                    os.path.join(C.DATA_PROCESSED, "heatmap_cache_dinov3.npz"),
                    **{p.replace("/", "__"): m for p, m in maps.items()})

            cls_scores = cls_knn_scores(train_cls, test_cls)
            a2 = roc_auc_score(test_df["label"], cls_scores)
            p2 = average_precision_score(test_df["label"], cls_scores)
            summary_rows.append(dict(backbone="dinov3", method="cls_knn",
                                     auroc=a2, ap=p2, best_f1=np.nan,
                                     bank_size=len(train_cls),
                                     feat_dim=train_cls.shape[1]))
            print(f"    [dinov3 CLS-kNN baseline] AUROC={a2:.4f}")

        del bb

    pd.DataFrame(summary_rows).to_csv(
        os.path.join(C.RES_CSV, "metrics_summary.csv"), index=False)
    pd.DataFrame(taxon_rows).to_csv(
        os.path.join(C.RES_CSV, "per_taxon_auroc.csv"), index=False)
    pd.DataFrame(thr_rows).to_csv(
        os.path.join(C.RES_CSV, "threshold_metrics.csv"), index=False)

    print("\n==== SUMMARY ====")
    print(pd.DataFrame(summary_rows).to_string(index=False))
    print("\nwrote metrics_summary.csv, per_taxon_auroc.csv, "
          "threshold_metrics.csv, per-image scores, ROC curves, memory banks")


if __name__ == "__main__":
    main()
