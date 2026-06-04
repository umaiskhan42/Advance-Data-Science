"""
Run the whole pipeline end to end:
    1. preprocess  ->  manifest + dataset summary
    2. patchcore   ->  memory bank, scoring, metrics (DINOv3 + baselines)
    3. visualize   ->  t-SNE/UMAP, attention, heatmaps, failure cases, figures

Usage:  python run_all.py
"""
import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

STAGES = ["01_preprocess.py", "02_patchcore.py", "03_visualize.py"]

if __name__ == "__main__":
    for s in STAGES:
        print("\n" + "#" * 70)
        print(f"# RUNNING {s}")
        print("#" * 70)
        runpy.run_path(os.path.join(HERE, s), run_name="__main__")
    print("\nAll stages complete. See results/ for CSVs and figures.")
