"""Stratified Train/Validation/Test Split for Grape Leaf Disease Dataset.

Creates a unified 70/15/15 stratified split based on the original filenames (fixed seed 42)
and saves it to split.json. This split is used consistently across all preprocessing
folders to ensure fair comparisons and eliminate data leakage.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
from sklearn.model_selection import train_test_split

TARGET_CLASSES = [
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
]


def create_stratified_split(
    original_root: Path,
    seed: int = 42,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> Dict:
    """Generate a stratified split of relative paths."""
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "Ratios must sum to 1.0"

    all_rel_paths = []
    all_labels = []

    for cls_idx, cls_name in enumerate(TARGET_CLASSES):
        cls_dir = original_root / cls_name
        if not cls_dir.exists():
            raise FileNotFoundError(f"Missing class folder: {cls_dir}")

        images = sorted([
            f.name for f in cls_dir.iterdir()
            if f.is_file() and f.suffix.lower() in [".jpg", ".jpeg", ".png"]
        ])

        for fname in images:
            all_rel_paths.append(f"{cls_name}/{fname}")
            all_labels.append(cls_idx)

    print(f"[*] Total dataset size for splitting: {len(all_rel_paths)} images")

    # Step 1: Split train vs temp (val + test)
    temp_ratio = val_ratio + test_ratio  # 0.30
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        all_rel_paths,
        all_labels,
        test_size=temp_ratio,
        stratify=all_labels,
        random_state=seed,
    )

    # Step 2: Split temp into val and test (equal parts 0.15 / 0.30 = 0.50)
    val_share = val_ratio / temp_ratio
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths,
        temp_labels,
        test_size=(1.0 - val_share),
        stratify=temp_labels,
        random_state=seed,
    )

    split_dict = {
        "metadata": {
            "seed": seed,
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "test_ratio": test_ratio,
            "total_samples": len(all_rel_paths),
            "classes": TARGET_CLASSES,
        },
        "train": sorted(train_paths),
        "val": sorted(val_paths),
        "test": sorted(test_paths),
    }

    return split_dict


def print_split_summary(split_dict: Dict):
    """Print class counts across train, val, and test splits."""
    classes = split_dict["metadata"]["classes"]
    print("\n" + "=" * 70)
    print(f"{'Class Name':<42} | {'Train':>6} | {'Val':>6} | {'Test':>6} | {'Total':>6}")
    print("-" * 70)

    total_tr, total_va, total_te = 0, 0, 0
    for cls_name in classes:
        tr_cnt = sum(1 for p in split_dict["train"] if p.startswith(cls_name + "/"))
        va_cnt = sum(1 for p in split_dict["val"] if p.startswith(cls_name + "/"))
        te_cnt = sum(1 for p in split_dict["test"] if p.startswith(cls_name + "/"))
        tot = tr_cnt + va_cnt + te_cnt

        total_tr += tr_cnt
        total_va += va_cnt
        total_te += te_cnt

        print(f"{cls_name:<42} | {tr_cnt:>6} | {va_cnt:>6} | {te_cnt:>6} | {tot:>6}")

    print("-" * 70)
    grand_total = total_tr + total_va + total_te
    print(
        f"{'TOTAL':<42} | {total_tr:>6} ({total_tr/grand_total*100:.1f}%) | "
        f"{total_va:>6} ({total_va/grand_total*100:.1f}%) | "
        f"{total_te:>6} ({total_te/grand_total*100:.1f}%) | {grand_total:>6}"
    )
    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Create unified stratified train/val/test split.")
    parser.add_argument("--data-dir", type=str, default="data/grape_original", help="Path to grape_original data.")
    parser.add_argument("--output", type=str, default="split.json", help="Path to output split.json file.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    original_root = project_root / args.data_dir
    output_path = project_root / args.output

    if not original_root.exists():
        print(f"[-] Error: Original data root not found at: {original_root.resolve()}")
        print("Please run download_data.py first!")
        return

    print(f"[*] Creating stratified 70/15/15 split with seed={args.seed}...")
    split_dict = create_stratified_split(original_root, seed=args.seed)
    print_split_summary(split_dict)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(split_dict, f, indent=2)

    print(f"[+] Saved split manifest to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
