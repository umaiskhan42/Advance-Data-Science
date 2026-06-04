"""
Object-detection stage 1 - prepare the annotated MD-AS-2025 frames for YOLO.

Converts the PASCAL-VOC XML boxes into YOLO-format .txt labels (written next to
the existing labels so Ultralytics finds them by the images->labels path rule),
and writes a train / val split matching the dataset paper (3600 / 400) plus a
data.yaml.

Class mapping (XML abbreviation -> taxon), verified against the paper's counts:
    Ar -> Arcellinida (2808)   Do -> Digononta (1352)   Mo -> Monogononta (669)
    Ne -> Nematoda  (560)      Eu -> Hypotrichida (432)
"""
from __future__ import annotations
import os
import sys
import glob
import xml.etree.ElementTree as ET

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C

ANN_ROOT = os.path.join(C.DATA_RAW, "annotated", "train_val_test")
IMG_DIR = os.path.join(ANN_ROOT, "images")
LBL_DIR = os.path.join(ANN_ROOT, "labels")
OD_DIR = os.path.join(C.DATA_PROCESSED, "od")

# fixed class order (index = YOLO class id)
NAMES = ["Arcellinida", "Digononta", "Monogononta", "Nematoda", "Hypotrichida"]
ABBR2IDX = {"Ar": 0, "Do": 1, "Mo": 2, "Ne": 3, "Eu": 4}

N_VAL = 400        # paper: 3600 train / 400 val (/ 400 test)
SEED = C.SEED


def voc_to_yolo(xml_path: str) -> list[str]:
    root = ET.parse(xml_path).getroot()
    w = float(root.find("size/width").text)
    h = float(root.find("size/height").text)
    lines = []
    for obj in root.findall("object"):
        cls = obj.find("name").text.strip()
        if cls not in ABBR2IDX:
            continue
        idx = ABBR2IDX[cls]
        b = obj.find("bndbox")
        xmin = float(b.find("xmin").text); ymin = float(b.find("ymin").text)
        xmax = float(b.find("xmax").text); ymax = float(b.find("ymax").text)
        cx = (xmin + xmax) / 2 / w
        cy = (ymin + ymax) / 2 / h
        bw = (xmax - xmin) / w
        bh = (ymax - ymin) / h
        cx, cy = min(max(cx, 0), 1), min(max(cy, 0), 1)
        bw, bh = min(max(bw, 0), 1), min(max(bh, 0), 1)
        lines.append(f"{idx} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return lines


def main():
    os.makedirs(OD_DIR, exist_ok=True)
    xmls = sorted(glob.glob(os.path.join(LBL_DIR, "*.xml")))
    print(f"found {len(xmls)} XML annotations")

    # map each xml to its image (try common extensions)
    items = []
    n_obj = 0
    for x in xmls:
        stem = os.path.splitext(os.path.basename(x))[0]
        img = None
        for ext in (".JPG", ".jpg", ".jpeg", ".png", ".PNG"):
            cand = os.path.join(IMG_DIR, stem + ext)
            if os.path.exists(cand):
                img = cand
                break
        if img is None:
            continue
        lines = voc_to_yolo(x)
        if not lines:
            continue
        # write YOLO label next to the XML (same stem, .txt)
        with open(os.path.join(LBL_DIR, stem + ".txt"), "w") as f:
            f.write("\n".join(lines) + "\n")
        n_obj += len(lines)
        items.append(img)

    print(f"matched {len(items)} image/label pairs, {n_obj} objects")

    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(items))
    val_idx = set(idx[:N_VAL].tolist())
    train = [items[i] for i in range(len(items)) if i not in val_idx]
    val = [items[i] for i in range(len(items)) if i in val_idx]
    print(f"split -> train {len(train)} | val {len(val)}")

    with open(os.path.join(OD_DIR, "train.txt"), "w") as f:
        f.write("\n".join(p.replace("\\", "/") for p in train) + "\n")
    with open(os.path.join(OD_DIR, "val.txt"), "w") as f:
        f.write("\n".join(p.replace("\\", "/") for p in val) + "\n")

    yaml_path = os.path.join(OD_DIR, "md_as_2025.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"path: {OD_DIR.replace(os.sep, '/')}\n")
        f.write("train: train.txt\n")
        f.write("val: val.txt\n")
        f.write(f"nc: {len(NAMES)}\n")
        f.write("names:\n")
        for i, n in enumerate(NAMES):
            f.write(f"  {i}: {n}\n")
    print("wrote", yaml_path)


if __name__ == "__main__":
    main()
