"""
Central configuration for the project:
  "Zero-Shot Microfauna Anomaly Detection with Frozen DINOv3 on MD-AS-2025"

All paths are resolved relative to the project root (the parent of this
`scripts/` directory) so the pipeline runs the same regardless of the current
working directory.
"""
from __future__ import annotations
import os

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPTS_DIR)

DATA_RAW = os.path.join(ROOT, "data", "raw")
DATA_PROCESSED = os.path.join(ROOT, "data", "processed")
WEIGHTS_DIR = os.path.join(ROOT, "weights")

RESULTS = os.path.join(ROOT, "results")
RES_CSV = os.path.join(RESULTS, "csv")
RES_FIG = os.path.join(RESULTS, "figures")
RES_TSNE = os.path.join(RESULTS, "tsne")
RES_ATTN = os.path.join(RESULTS, "attention")
RES_HEAT = os.path.join(RESULTS, "heatmaps")
RES_FAIL = os.path.join(RESULTS, "failure_cases")

ALL_DIRS = [DATA_PROCESSED, WEIGHTS_DIR, RESULTS, RES_CSV, RES_FIG,
            RES_TSNE, RES_ATTN, RES_HEAT, RES_FAIL]


def ensure_dirs() -> None:
    for d in ALL_DIRS:
        os.makedirs(d, exist_ok=True)


# --------------------------------------------------------------------------- #
# Dataset definition  (MD-AS-2025, cropped single-object images)
# --------------------------------------------------------------------------- #
# Healthy taxa -> "normal" class, dominant in well-performing sludge.
# Stressed taxa -> "anomaly" class, indicate the unfavourable end of the SBI.
HEALTHY_TAXA = ["Hypotrichida", "Digononta", "Monogononta"]
STRESSED_TAXA = ["Arcellinida", "Nematoda"]
ALL_TAXA = HEALTHY_TAXA + STRESSED_TAXA

# label convention: 0 = normal/healthy, 1 = anomaly/stressed
LABEL_NORMAL = 0
LABEL_ANOMALY = 1

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

# --------------------------------------------------------------------------- #
# Experiment hyper-parameters
# --------------------------------------------------------------------------- #
SEED = 42

# Image preprocessing (DINOv3 / ImageNet normalisation).
IMG_SIZE = 224
NORM_MEAN = (0.485, 0.456, 0.406)
NORM_STD = (0.229, 0.224, 0.225)

# How many images to sample per taxon (keeps a CPU run tractable while still
# giving a statistically meaningful evaluation).  Hypotrichida (556) and
# Nematoda (707) are used almost in full; the larger taxa are capped.
MAX_HEALTHY_PER_TAXON = 800
MAX_STRESSED_PER_TAXON = 800
TRAIN_FRACTION = 0.70          # fraction of healthy images used to build the bank

# PatchCore memory bank.
NEIGHBOURHOOD = 3              # k x k local feature aggregation (0/1 disables)
POOL_MAX = 40000              # max patch vectors held before coreset subsampling
CORESET_SIZE = 5000           # final memory-bank size after greedy coreset
CORESET_PROJ_DIM = 128       # random-projection dim used to speed up the coreset
NN_K = 1                      # nearest-neighbours used for the patch anomaly score
DIST_METRIC = "euclidean"    # features are L2-normalised, so euclidean ~ cosine

# Backbones to compare (the proposal's H1: DINOv3 vs random-init vs CNN).
BACKBONES = ["dinov3", "randvit", "resnet50"]
PRIMARY_BACKBONE = "dinov3"

TIMM_DINOV3 = "vit_small_patch16_dinov3.lvd1689m"
TIMM_RESNET = "resnet50.a1_in1k"

# Inference batch size for feature extraction.
BATCH_SIZE = 16

# Number of example images to render for each qualitative visualisation.
N_VIS_PER_TAXON = 6
N_FAILURE_CASES = 12
