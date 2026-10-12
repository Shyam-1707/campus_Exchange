"""Image Preprocessing Pipeline for Grape Leaf Disease Dataset.

Applies 12 distinct computer vision preprocessing techniques to Grape leaf images:
 1. grape_resized      - 224x224 INTER_AREA resizing
 2. grape_grayscale    - Grayscale converted to 3-channel
 3. grape_filtered     - Gaussian blur (5x5) + Median filter
 4. grape_equalized    - CLAHE on L channel in LAB color space
 5. grape_sharpened    - Unsharp masking sharpening
 6. grape_edges        - Canny edge detection (3-channel)
 7. grape_negative     - Photographic negative (255 - image)
 8. grape_blended      - Alpha blending (0.5 original + 0.5 sharpened)
 9. grape_morphology   - Otsu thresholding + opening + closing (5x5 elliptical kernel)
 10. grape_segmented   - Leaf segmentation & background removal (HSV + Otsu + morphology)
 11. grape_normalized  - Min-Max normalization scaled to 0-255
 12. grape_augmented   - Flips, rotations, and brightness shifts
"""

import argparse
import csv
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


# Configure logging
logging.basicConfig(
    filename="preprocessing_errors.log",
    filemode="a",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.ERROR,
)

TARGET_SIZE = (224, 224)

TECHNIQUES = [
    "grape_resized",
    "grape_grayscale",
    "grape_filtered",
    "grape_equalized",
    "grape_sharpened",
    "grape_edges",
    "grape_negative",
    "grape_blended",
    "grape_morphology",
    "grape_segmented",
    "grape_normalized",
    "grape_augmented",
]


def resize_image(img: np.ndarray) -> np.ndarray:
    """Resize image to 224x224 using cv2.INTER_AREA for downsampling fidelity."""
    if img.shape[:2] == TARGET_SIZE:
        return img
    return cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_AREA)


def apply_grayscale(img: np.ndarray) -> np.ndarray:
    """Convert to grayscale and replicate across 3 channels."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def apply_filtered(img: np.ndarray) -> np.ndarray:
    """Noise removal: Gaussian blur (5x5) followed by median blur (size 5)."""
    gauss = cv2.GaussianBlur(img, (5, 5), sigmaX=0)
    return cv2.medianBlur(gauss, 5)


def apply_equalized(img: np.ndarray) -> np.ndarray:
    """Apply CLAHE on the Lightness (L) channel in LAB color space."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l_channel)
    merged = cv2.merge((cl, a_channel, b_channel))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def apply_sharpened(img: np.ndarray) -> np.ndarray:
    """Sharpening using unsharp masking."""
    gaussian = cv2.GaussianBlur(img, (0, 0), sigmaX=2.0)
    # unsharp formula: original + 1.5 * (original - gaussian) = 2.5 * original - 1.5 * gaussian
    sharpened = cv2.addWeighted(img, 1.5, gaussian, -0.5, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def apply_edges(img: np.ndarray) -> np.ndarray:
    """Canny edge detection returned as 3-channel image."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blurred, 50, 150)
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)


def apply_negative(img: np.ndarray) -> np.ndarray:
    """Invert pixel values (255 - image)."""
    return cv2.bitwise_not(img)


def apply_blended(img: np.ndarray) -> np.ndarray:
    """Blend 50% original image with 50% sharpened image."""
    sharp = apply_sharpened(img)
    return cv2.addWeighted(img, 0.5, sharp, 0.5, 0)


def apply_morphology(img: np.ndarray) -> np.ndarray:
    """Thresholding (Otsu) followed by opening + closing using a 5x5 elliptical kernel."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    return cv2.cvtColor(closed, cv2.COLOR_GRAY2BGR)


def apply_segmented(img: np.ndarray) -> np.ndarray:
    """Leaf segmentation and background removal using HSV color range, Otsu, and morphology."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Broad leaf/green/yellowish mask
    lower_leaf = np.array([15, 20, 20], dtype=np.uint8)
    upper_leaf = np.array([105, 255, 255], dtype=np.uint8)
    mask_hsv = cv2.inRange(hsv, lower_leaf, upper_leaf)

    # Otsu thresholding on saturation
    s_channel = hsv[:, :, 1]
    _, mask_otsu = cv2.threshold(s_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Combined mask
    combined_mask = cv2.bitwise_or(mask_hsv, mask_otsu)

    # Morphological cleaning
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)

    # Retain significant leaf contours
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    final_mask = np.zeros_like(cleaned)
    if contours:
        # Fill contours that are at least 5% of the total image area
        min_area = 0.03 * (img.shape[0] * img.shape[1])
        valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) >= min_area]
        if valid_contours:
            cv2.drawContours(final_mask, valid_contours, -1, 255, thickness=cv2.FILLED)
        else:
            final_mask = cleaned
    else:
        final_mask = cleaned

    return cv2.bitwise_and(img, img, mask=final_mask)


def apply_normalized(img: np.ndarray) -> np.ndarray:
    """Min-Max normalization scaled to 0-255."""
    norm = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    return norm.astype(np.uint8)


def generate_augmented_variants(img: np.ndarray, stem: str) -> List[Tuple[str, np.ndarray]]:
    """Generate augmented copies: horizontal flip, rotation, and brightness adjustment."""
    variants = []
    # 1. Base resized image
    variants.append((f"{stem}.jpg", img))

    # 2. Horizontal flip
    h_flip = cv2.flip(img, 1)
    variants.append((f"{stem}_aug_flip.jpg", h_flip))

    # 3. 90-degree clockwise rotation
    rot_90 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    variants.append((f"{stem}_aug_rot90.jpg", rot_90))

    # 4. Brightness scaling (+20%)
    bright = cv2.convertScaleAbs(img, alpha=1.2, beta=15)
    variants.append((f"{stem}_aug_bright.jpg", bright))

    return variants


def process_technique(
    technique: str,
    original_files: List[Path],
    original_root: Path,
    output_base: Path,
    force: bool = False,
) -> int:
    """Process a single technique across all images with tqdm progress bar."""
    dest_folder = output_base / technique
    dest_folder.mkdir(parents=True, exist_ok=True)
    count = 0

    pbar = tqdm(original_files, desc=f"Processing {technique:<18}", unit="img")
    for img_path in pbar:
        try:
            rel_path = img_path.relative_to(original_root)
            dest_img_path = dest_folder / rel_path
            dest_img_path.parent.mkdir(parents=True, exist_ok=True)

            if technique == "grape_augmented":
                stem = img_path.stem
                expected_first = dest_img_path.parent / f"{stem}.jpg"
                if expected_first.exists() and not force:
                    count += 4
                    continue

                bgr = cv2.imread(str(img_path))
                if bgr is None:
                    raise ValueError(f"Could not read image: {img_path}")
                img_224 = resize_image(bgr)

                aug_items = generate_augmented_variants(img_224, stem)
                for fname, aug_img in aug_items:
                    out_f = dest_img_path.parent / fname
                    if not out_f.exists() or force:
                        cv2.imwrite(str(out_f), aug_img)
                    count += 1
                continue

            # Check if file exists and skip if not forced
            if dest_img_path.exists() and not force:
                count += 1
                continue

            bgr = cv2.imread(str(img_path))
            if bgr is None:
                raise ValueError(f"Could not read image: {img_path}")

            img_224 = resize_image(bgr)

            if technique == "grape_resized":
                processed = img_224
            elif technique == "grape_grayscale":
                processed = apply_grayscale(img_224)
            elif technique == "grape_filtered":
                processed = apply_filtered(img_224)
            elif technique == "grape_equalized":
                processed = apply_equalized(img_224)
            elif technique == "grape_sharpened":
                processed = apply_sharpened(img_224)
            elif technique == "grape_edges":
                processed = apply_edges(img_224)
            elif technique == "grape_negative":
                processed = apply_negative(img_224)
            elif technique == "grape_blended":
                processed = apply_blended(img_224)
            elif technique == "grape_morphology":
                processed = apply_morphology(img_224)
            elif technique == "grape_segmented":
                processed = apply_segmented(img_224)
            elif technique == "grape_normalized":
                processed = apply_normalized(img_224)
            else:
                raise ValueError(f"Unknown technique: {technique}")

            cv2.imwrite(str(dest_img_path), processed)
            count += 1

        except Exception as e:
            logging.error(f"Failed technique={technique} on {img_path}: {e}")

    return count


def create_preview_grid(sample_path: Path, output_file: Path):
    """Generate preview/preprocessing_samples.png displaying one leaf through all techniques."""
    bgr = cv2.imread(str(sample_path))
    if bgr is None:
        print(f"[-] Could not load sample image for preview: {sample_path}")
        return

    img_224 = resize_image(bgr)

    previews = {
        "1. Original (224x224)": img_224,
        "2. Grayscale (3-ch)": apply_grayscale(img_224),
        "3. Filtered (Gauss+Median)": apply_filtered(img_224),
        "4. Equalized (CLAHE L-ch)": apply_equalized(img_224),
        "5. Sharpened (Unsharp)": apply_sharpened(img_224),
        "6. Edges (Canny)": apply_edges(img_224),
        "7. Negative (255 - img)": apply_negative(img_224),
        "8. Blended (Orig + Sharp)": apply_blended(img_224),
        "9. Morphology (Otsu+Op+Cl)": apply_morphology(img_224),
        "10. Segmented (Leaf mask)": apply_segmented(img_224),
        "11. Normalized (0-255)": apply_normalized(img_224),
        "12. Augmented (Flip Horiz)": cv2.flip(img_224, 1),
    }

    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.flatten()

    for idx, (title, img) in enumerate(previews.items()):
        # Convert BGR to RGB for matplotlib
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        axes[idx].imshow(rgb)
        axes[idx].set_title(title, fontsize=12, fontweight="bold", pad=8)
        axes[idx].axis("off")

    plt.suptitle("Grape Leaf Preprocessing Pipeline - Visual Comparison Grid", fontsize=16, fontweight="bold", y=0.98)
    plt.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output_file), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved sample preview grid to: {output_file.resolve()}")


def main():
    parser = argparse.ArgumentParser(description="Apply CV preprocessing techniques to Grape dataset.")
    parser.add_argument("--data-dir", type=str, default="data", help="Base data directory.")
    parser.add_argument("--force", action="store_true", help="Force overwrite existing preprocessed images.")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    data_dir = project_root / args.data_dir
    original_root = data_dir / "grape_original"

    if not original_root.exists():
        print(f"[-] Error: Original data directory not found at {original_root.resolve()}")
        print("Please run download_data.py first!")
        return

    original_files = sorted(
        [f for f in original_root.rglob("*.*") if f.is_file() and f.suffix.lower() in [".jpg", ".jpeg", ".png"]]
    )

    if not original_files:
        print(f"[-] Error: No images found in {original_root.resolve()}")
        return

    print(f"\n[*] Found {len(original_files)} original grape leaf images across all classes.")
    print(f"[*] Commencing preprocessing of {len(TECHNIQUES)} techniques...\n")

    summary_rows = []

    for tech in TECHNIQUES:
        count = process_technique(tech, original_files, original_root, data_dir, force=args.force)
        dest_folder_name = tech
        summary_rows.append({
            "Technique": tech,
            "Folder": f"data/{dest_folder_name}",
            "Image Count": count
        })

    # Write summary CSV
    summary_csv = project_root / "preprocessing_summary.csv"
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Technique", "Folder", "Image Count"])
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"\n[+] Preprocessing summary written to: {summary_csv.resolve()}")

    # Generate preview grid
    preview_png = project_root / "preview" / "preprocessing_samples.png"
    sample_img = original_files[0]
    create_preview_grid(sample_img, preview_png)

    # Print summary table
    print("\n" + "=" * 65)
    print(f"{'Technique':<22} | {'Folder':<22} | {'Image Count':>12}")
    print("-" * 65)
    for row in summary_rows:
        print(f"{row['Technique']:<22} | {row['Folder']:<22} | {row['Image Count']:>12}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
