"""Main Entry Point for Grape Leaf Disease Classification & Preprocessing Benchmark.

Run interactively:
    python main.py

Run specific tasks:
    python main.py --evaluate
    python main.py --predict "path/to/image.jpg"
    python main.py --demo
    python main.py --train grape_augmented
    python main.py --train-all
    python main.py --preprocess
    python main.py --split
    python main.py --download
"""

import argparse
import random
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

BANNER = r"""
================================================================================
   🍇 GRAPE LEAF DISEASE CLASSIFICATION & PREPROCESSING BENCHMARK 🍇
               Deep Learning (MobileNetV2) & Computer Vision
================================================================================
"""


def find_sample_image() -> Path | None:
    """Find a sample leaf image from original grape data."""
    orig_dir = PROJECT_ROOT / "data" / "grape_original"
    if not orig_dir.exists():
        return None
    images = list(orig_dir.rglob("*.JPG")) + list(orig_dir.rglob("*.jpg")) + list(orig_dir.rglob("*.png"))
    if images:
        return random.choice(images)
    return None


def run_predict(image_path: str | None = None, model_path: str | None = None):
    """Run prediction on an image."""
    predict_script = PROJECT_ROOT / "predict.py"
    if not image_path:
        sample = find_sample_image()
        if not sample:
            print("[-] No images found in data/grape_original/ to test.")
            return
        image_path = str(sample)
        print(f"[*] Selected sample image: {sample.relative_to(PROJECT_ROOT)}")

    cmd = [sys.executable, str(predict_script), "--image", str(image_path)]
    if model_path:
        cmd.extend(["--model", str(model_path)])
    subprocess.run(cmd)


def run_evaluate():
    """Run benchmarking and evaluation aggregation."""
    eval_script = PROJECT_ROOT / "evaluate.py"
    subprocess.run([sys.executable, str(eval_script)])


def run_preprocess():
    """Run 12 preprocessing techniques."""
    script = PROJECT_ROOT / "preprocess.py"
    subprocess.run([sys.executable, str(script)])


def run_split():
    """Run stratified dataset splitting."""
    script = PROJECT_ROOT / "split_data.py"
    subprocess.run([sys.executable, str(script)])


def run_train(dataset: str = "grape_augmented", train_all: bool = False, force: bool = False):
    """Run model training."""
    script = PROJECT_ROOT / "train.py"
    cmd = [sys.executable, str(script)]
    if train_all:
        cmd.append("--all")
    else:
        cmd.extend(["--dataset", dataset])
    if force:
        cmd.append("--force")
    subprocess.run(cmd)


def run_download(force: bool = False):
    """Run dataset download script."""
    script = PROJECT_ROOT / "download_data.py"
    cmd = [sys.executable, str(script)]
    if force:
        cmd.append("--force")
    subprocess.run(cmd)


def interactive_menu():
    """Display interactive CLI menu for project operations."""
    print(BANNER)
    while True:
        print("\nChoose an option:")
        print("  [1] 🔍 Run Leaf Disease Diagnosis (Demo / Sample Image)")
        print("  [2] 🖼️  Run Diagnosis on Custom Image Path")
        print("  [3] 📊 View Comparative Benchmark Evaluation Results")
        print("  [4] ⚡ Run 12 Preprocessing Techniques (OpenCV + NumPy)")
        print("  [5] 📂 Generate Stratified Train/Val/Test Split (70/15/15)")
        print("  [6] 🧠 Train MobileNetV2 Model (Transfer Learning)")
        print("  [7] 📥 Download PlantVillage Grape Dataset")
        print("  [0] ❌ Exit")

        try:
            choice = input("\nEnter choice [0-7] (default 1): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not choice or choice == "1":
            print("\n[*] Running disease prediction demo...")
            run_predict()
        elif choice == "2":
            img_path = input("Enter path to image: ").strip().strip('"').strip("'")
            if img_path:
                run_predict(img_path)
            else:
                print("[-] No image path provided.")
        elif choice == "3":
            print("\n[*] Aggregating evaluation benchmarks...")
            run_evaluate()
        elif choice == "4":
            print("\n[*] Starting preprocessing pipeline...")
            run_preprocess()
        elif choice == "5":
            print("\n[*] Generating dataset split...")
            run_split()
        elif choice == "6":
            tech = input("Enter technique name (e.g. grape_augmented, or 'all'): ").strip()
            if tech.lower() == "all":
                run_train(train_all=True)
            elif tech:
                run_train(dataset=tech)
            else:
                run_train(dataset="grape_augmented")
        elif choice == "7":
            run_download()
        elif choice == "0":
            print("\nGoodbye!")
            break
        else:
            print("[-] Invalid choice. Please select 0 to 7.")


def main():
    parser = argparse.ArgumentParser(description="Grape Leaf Disease Project Entry Point.")
    parser.add_argument("--predict", type=str, default=None, help="Path to leaf image for disease diagnosis.")
    parser.add_argument("--model", type=str, default=None, help="Model weights checkpoint to use.")
    parser.add_argument("--demo", action="store_true", help="Run disease prediction demo on a random leaf image.")
    parser.add_argument("--evaluate", action="store_true", help="Display benchmark table & update comparison chart.")
    parser.add_argument("--preprocess", action="store_true", help="Run 12 CV preprocessing algorithms.")
    parser.add_argument("--split", action="store_true", help="Generate stratified train/val/test split.")
    parser.add_argument("--train", type=str, default=None, help="Train MobileNetV2 on specific dataset.")
    parser.add_argument("--train-all", action="store_true", help="Train sequentially on all 12 datasets.")
    parser.add_argument("--download", action="store_true", help="Download PlantVillage grape dataset.")
    parser.add_argument("--force", action="store_true", help="Force overwrite or retraining.")
    parser.add_argument("--menu", action="store_true", help="Launch interactive menu.")
    args = parser.parse_args()

    # If any specific CLI action was requested, execute it
    if args.predict:
        run_predict(args.predict, args.model)
    elif args.demo:
        run_predict(model_path=args.model)
    elif args.evaluate:
        run_evaluate()
    elif args.preprocess:
        run_preprocess()
    elif args.split:
        run_split()
    elif args.train:
        run_train(dataset=args.train, force=args.force)
    elif args.train_all:
        run_train(train_all=True, force=args.force)
    elif args.download:
        run_download(force=args.force)
    else:
        # Default when run simply as `python main.py`
        interactive_menu()


if __name__ == "__main__":
    main()
