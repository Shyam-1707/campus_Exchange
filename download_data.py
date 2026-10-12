"""Download and extract Grape classes from the PlantVillage dataset.

This script:
1. Downloads the PlantVillage dataset via kagglehub (or instructions for kaggle.json).
2. Filters exclusively for the 4 Grape disease classes from the color image subset.
3. Copies them into data/grape_original/<class_name>/ with original filenames.
4. Outputs the image count per class.
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

# The 4 target Grape classes
TARGET_CLASSES = [
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
]

DEFAULT_DATASET = "abdallahalidev/plantvillage-dataset"


def check_kaggle_credentials() -> bool:
    """Check if Kaggle credentials exist either in ~/.kaggle/kaggle.json or environment."""
    home_kaggle = Path.home() / ".kaggle" / "kaggle.json"
    env_user = os.environ.get("KAGGLE_USERNAME")
    env_key = os.environ.get("KAGGLE_KEY")
    if home_kaggle.exists() or (env_user and env_key):
        return True
    return False


def print_credential_guide():
    """Print instructions for placing kaggle.json."""
    kaggle_path = Path.home() / ".kaggle" / "kaggle.json"
    print("\n" + "=" * 65)
    print("NOTE: Kaggle credentials (kaggle.json) not detected.")
    print(f"If download prompts for authentication, place your kaggle.json at:")
    print(f"  --> {kaggle_path.resolve()}")
    print("Or set environment variables:")
    print("  KAGGLE_USERNAME=<username>")
    print("  KAGGLE_KEY=<api_key>")
    print("=" * 65 + "\n")


def download_dataset(dataset_name: str) -> Path:
    """Download the dataset using kagglehub or return path if already cached."""
    print(f"[*] Downloading / checking dataset: {dataset_name} via kagglehub...")
    if not check_kaggle_credentials():
        print_credential_guide()

    try:
        import kagglehub
        dataset_path = kagglehub.dataset_download(dataset_name)
        print(f"[+] Dataset available at: {dataset_path}")
        return Path(dataset_path)
    except Exception as e:
        print(f"[-] kagglehub download failed: {e}")
        print("Attempting kaggle CLI fallback if installed...")
        try:
            import subprocess
            subprocess.run(["kaggle", "datasets", "download", "-d", dataset_name, "--unzip"], check=True)
            return Path(".")
        except Exception as err:
            raise RuntimeError(
                f"Failed to download dataset. Please verify kaggle.json at {Path.home() / '.kaggle' / 'kaggle.json'}. "
                f"Error: {err}"
            ) from e


def find_color_folder(root_path: Path) -> Path:
    """Locate the 'color' directory within the downloaded PlantVillage dataset."""
    candidates = list(root_path.rglob("color"))
    for c in candidates:
        if c.is_dir() and any(c.glob("Grape*")):
            return c
    
    # Check if grape classes are directly under root_path or subfolders
    for p in root_path.rglob("*"):
        if p.is_dir() and "color" in p.name.lower():
            if any(p.glob("Grape*")):
                return p
                
    # Direct search for any Grape___ directory
    grape_folders = list(root_path.rglob("Grape___*"))
    if grape_folders:
        return grape_folders[0].parent

    raise FileNotFoundError(f"Could not find 'color' folder with Grape classes in {root_path}")


def extract_grape_classes(source_color_dir: Path, output_dir: Path, force: bool = False):
    """Copy the 4 Grape classes into output_dir/grape_original/<class_name>/."""
    grape_original_dir = output_dir / "grape_original"
    grape_original_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[*] Extracting Grape classes from: {source_color_dir}")
    print(f"[*] Destination: {grape_original_dir.resolve()}\n")

    summary_counts = {}

    for cls_name in TARGET_CLASSES:
        src_cls_dir = source_color_dir / cls_name
        dest_cls_dir = grape_original_dir / cls_name
        dest_cls_dir.mkdir(parents=True, exist_ok=True)

        if not src_cls_dir.exists():
            # Try case-insensitive matching if exact name fails
            matched = [d for d in source_color_dir.iterdir() if d.is_dir() and d.name.lower() == cls_name.lower()]
            if matched:
                src_cls_dir = matched[0]
            else:
                print(f"[!] Warning: Class directory '{cls_name}' not found in {source_color_dir}")
                summary_counts[cls_name] = 0
                continue

        src_images = [f for f in src_cls_dir.iterdir() if f.is_file() and f.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]]
        copied = 0
        skipped = 0

        for img in src_images:
            target_img = dest_cls_dir / img.name
            if target_img.exists() and not force:
                skipped += 1
            else:
                shutil.copy2(img, target_img)
                copied += 1

        total_in_dest = len(list(dest_cls_dir.glob("*.*")))
        summary_counts[cls_name] = total_in_dest
        print(f"  • {cls_name:<45} : {total_in_dest:>5} images (Copied: {copied}, Skipped: {skipped})")

    total_images = sum(summary_counts.values())
    print("\n" + "=" * 65)
    print(f"  Total Grape Images Extracted: {total_images}")
    print("=" * 65 + "\n")
    return summary_counts


def main():
    parser = argparse.ArgumentParser(description="Download and extract PlantVillage Grape classes.")
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET, help="Kaggle dataset identifier.")
    parser.add_argument("--data-dir", type=str, default="data", help="Target base directory for datasets.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing extracted files.")
    parser.add_argument("--dataset-path", type=str, default=None, help="Optional manual path to downloaded dataset.")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    data_dir = project_root / args.data_dir

    if args.dataset_path:
        root_path = Path(args.dataset_path)
    else:
        root_path = download_dataset(args.dataset)

    color_dir = find_color_folder(root_path)
    print(f"[+] Found PlantVillage color directory: {color_dir}")

    extract_grape_classes(color_dir, data_dir, force=args.force)
    print("[+] Download and extraction complete!")


if __name__ == "__main__":
    main()
