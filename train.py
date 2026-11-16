"""MobileNetV2 Transfer Learning Pipeline for Grape Disease Classification.

Framework: PyTorch (torchvision MobileNetV2 with ImageNet weights)
Chosen for robust cross-platform stability, transparent two-stage fine-tuning,
and optimal memory management.

Two-Stage Training Strategy with Split Feature Extraction:
  - Base blocks (0-13) are permanently frozen during both stages.
  - Intermediate representations (block 13 feature maps & full pooled features)
    are pre-extracted into RAM in a single vectorized forward pass.
  - Stage 1: Feature extractor frozen. Train custom classification head (Adam lr=1e-3).
  - Stage 2: Top layers (blocks 14-18) unfrozen. Fine-tune end-to-end with low learning rate (Adam lr=1e-5).

Features:
  - In-memory feature caching for extreme speedup on CPU
  - EarlyStopping with best weight restoration
  - ReduceLROnPlateau dynamic scheduling
  - ModelCheckpoint saving best weights to results/<folder_name>/best_model.pt
  - Comprehensive metrics: Test Accuracy, Macro-F1, Weighted-F1, Confusion Matrix, Curves
"""

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import models, transforms

TARGET_CLASSES = [
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
]

CLASS_TO_IDX = {cls_name: idx for idx, cls_name in enumerate(TARGET_CLASSES)}
IDX_TO_CLASS = {idx: cls_name for idx, cls_name in enumerate(TARGET_CLASSES)}

ALL_TECHNIQUES = [
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


class GrapeDataset(Dataset):
    """Raw image dataset for on-the-fly loading."""

    def __init__(self, data_root: Path, file_list: List[str], transform=None):
        self.data_root = data_root
        self.transform = transform
        self.samples: List[Tuple[Path, int]] = []

        for rel_path in file_list:
            cls_name, fname = rel_path.split("/", 1)
            label = CLASS_TO_IDX[cls_name]
            full_path = self.data_root / cls_name / fname
            if full_path.exists():
                self.samples.append((full_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        with Image.open(path) as img:
            rgb = img.convert("RGB")
        if self.transform:
            rgb = self.transform(rgb)
        return rgb, label


def get_augmented_train_files(data_root: Path, base_train_files: List[str]) -> List[str]:
    """Include augmented copies exclusively for images present in the training split."""
    aug_train_files = []
    for rel_path in base_train_files:
        cls_name, fname = rel_path.split("/", 1)
        stem = Path(fname).stem
        cls_dir = data_root / cls_name
        matches = list(cls_dir.glob(f"{stem}*.jpg"))
        for m in matches:
            aug_train_files.append(f"{cls_name}/{m.name}")
    return sorted(list(set(aug_train_files)))


def build_mobilenet_v2(num_classes: int = 4, dropout: float = 0.3) -> nn.Module:
    """Build MobileNetV2 with ImageNet pretrained weights and custom classification head."""
    weights = models.MobileNet_V2_Weights.IMAGENET1K_V1
    model = models.mobilenet_v2(weights=weights)

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, num_classes)
    )
    return model


def extract_features(
    dataset_dir: Path,
    file_list: List[str],
    base_model: nn.Module,
    device: torch.device,
    batch_size: int = 64,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pre-extract block 13 feature maps (for Stage 2) and pooled 1280-d features (for Stage 1)."""
    base_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    ds = GrapeDataset(dataset_dir, file_list, transform=base_transform)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    base_blocks = base_model.features[:14].to(device).eval()
    top_blocks = base_model.features[14:].to(device).eval()

    all_feat13 = []
    all_feathead = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            f13 = base_blocks(imgs)
            f_top = top_blocks(f13)
            f_head = nn.functional.adaptive_avg_pool2d(f_top, (1, 1)).flatten(1)

            all_feat13.append(f13.cpu())
            all_feathead.append(f_head.cpu())
            all_labels.append(labels)

    feat13_tensor = torch.cat(all_feat13, dim=0)
    feathead_tensor = torch.cat(all_feathead, dim=0)
    labels_tensor = torch.cat(all_labels, dim=0)

    return feat13_tensor, feathead_tensor, labels_tensor


def run_training_pipeline(
    dataset_name: str,
    data_base: Path,
    split_path: Path,
    results_base: Path,
    batch_size: int = 64,
    stage1_epochs: int = 5,
    stage2_epochs: int = 4,
    patience: int = 3,
    device_name: str = "auto",
    force: bool = False,
) -> Dict:
    """Execute complete 2-stage training, validation, test evaluation, and artifact generation."""
    print("\n" + "=" * 70, flush=True)
    print(f"[*] TRAINING DATASET: {dataset_name}", flush=True)
    print("=" * 70, flush=True)

    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)

    if device.type == "cpu":
        torch.set_num_threads(min(10, os.cpu_count() or 4))

    print(f"[*] Compute Device: {device} " + (f"({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""), flush=True)

    out_dir = results_base / dataset_name
    metrics_file = out_dir / "metrics.json"
    if metrics_file.exists() and not force:
        print(f"[+] Dataset '{dataset_name}' already trained. Found {metrics_file.resolve()}. Skipping (use --force to retrain).", flush=True)
        with open(metrics_file, "r", encoding="utf-8") as f:
            return json.load(f)

    dataset_dir = data_base / dataset_name
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir.resolve()}")

    with open(split_path, "r", encoding="utf-8") as f:
        split_dict = json.load(f)

    train_files = split_dict["train"]
    val_files = split_dict["val"]
    test_files = split_dict["test"]

    if dataset_name == "grape_augmented":
        print("[*] Augmenting training files exclusively from train split...", flush=True)
        train_files = get_augmented_train_files(dataset_dir, train_files)
        print(f"[+] Total augmented training samples: {len(train_files)}", flush=True)

    model = build_mobilenet_v2(num_classes=4, dropout=0.3).to(device)
    criterion = nn.CrossEntropyLoss()

    out_dir.mkdir(parents=True, exist_ok=True)
    best_weights_path = out_dir / "best_model.pt"

    print(f"[*] Pre-extracting backbone features into memory for ultra-fast training...", flush=True)
    t_feat0 = time.time()
    tr_f13, tr_fhead, tr_y = extract_features(dataset_dir, train_files, model, device, batch_size=batch_size)
    va_f13, va_fhead, va_y = extract_features(dataset_dir, val_files, model, device, batch_size=batch_size)
    te_f13, te_fhead, te_y = extract_features(dataset_dir, test_files, model, device, batch_size=batch_size)
    t_feat1 = time.time()
    print(f"[+] Feature pre-extraction finished in {t_feat1 - t_feat0:.1f}s. Samples -> Train: {len(tr_y)}, Val: {len(va_y)}, Test: {len(te_y)}", flush=True)

    # Tensor datasets
    train_tds = TensorDataset(tr_f13, tr_fhead, tr_y)
    val_tds = TensorDataset(va_f13, va_fhead, va_y)
    test_tds = TensorDataset(te_f13, te_fhead, te_y)

    train_loader = DataLoader(train_tds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_tds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_tds, batch_size=batch_size, shuffle=False)

    history = {
        "epoch": [],
        "stage": [],
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "lr": [],
    }

    best_val_loss = float("inf")
    best_model_weights = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0
    total_start_time = time.time()

    # ----------------- STAGE 1: Train Head Only -----------------
    print(f"\n--- Stage 1: Feature Extractor Frozen (max {stage1_epochs} epochs, lr=1e-3) ---", flush=True)
    for p in model.features.parameters():
        p.requires_grad = False
    for p in model.classifier.parameters():
        p.requires_grad = True

    model.classifier.train()
    optimizer_s1 = torch.optim.Adam(model.classifier.parameters(), lr=1e-3)
    scheduler_s1 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_s1, mode="min", factor=0.5, patience=2, min_lr=1e-6)

    global_epoch = 1
    for ep in range(1, stage1_epochs + 1):
        # Train Stage 1
        model.classifier.train()
        tr_loss_sum, tr_correct, tr_total = 0.0, 0, 0
        for _, fhead, y in train_loader:
            fhead, y = fhead.to(device), y.to(device)
            optimizer_s1.zero_grad()
            out = model.classifier(fhead)
            loss = criterion(out, y)
            loss.backward()
            optimizer_s1.step()

            tr_loss_sum += loss.item() * y.size(0)
            tr_correct += out.argmax(dim=1).eq(y).sum().item()
            tr_total += y.size(0)

        tr_loss = tr_loss_sum / tr_total
        tr_acc = tr_correct / tr_total

        # Val Stage 1
        model.classifier.eval()
        va_loss_sum, va_correct, va_total = 0.0, 0, 0
        with torch.no_grad():
            for _, fhead, y in val_loader:
                fhead, y = fhead.to(device), y.to(device)
                out = model.classifier(fhead)
                loss = criterion(out, y)
                va_loss_sum += loss.item() * y.size(0)
                va_correct += out.argmax(dim=1).eq(y).sum().item()
                va_total += y.size(0)

        va_loss = va_loss_sum / va_total
        va_acc = va_correct / va_total
        curr_lr = optimizer_s1.param_groups[0]["lr"]
        scheduler_s1.step(va_loss)

        history["epoch"].append(global_epoch)
        history["stage"].append(1)
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)
        history["lr"].append(curr_lr)

        improved = ""
        if va_loss < best_val_loss:
            best_val_loss = va_loss
            best_model_weights = copy.deepcopy(model.state_dict())
            torch.save(best_model_weights, str(best_weights_path))
            improved = " [SAVED BEST]"
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        print(
            f"Epoch {global_epoch:2d} (S1-{ep:02d}) | "
            f"Tr Loss: {tr_loss:.4f} Acc: {tr_acc*100:.2f}% | "
            f"Val Loss: {va_loss:.4f} Acc: {va_acc*100:.2f}% | "
            f"LR: {curr_lr:.1e}{improved}",
            flush=True
        )
        global_epoch += 1

        if epochs_no_improve >= patience:
            print(f"[*] Early stopping in Stage 1 after {ep} epochs.", flush=True)
            break

    # ----------------- STAGE 2: Fine-Tuning Top Layers -----------------
    print(f"\n--- Stage 2: Fine-Tuning Top Layers (max {stage2_epochs} epochs, lr=1e-5) ---", flush=True)
    model.load_state_dict(best_model_weights)

    # Unfreeze top blocks (14-18)
    for p in model.features[:14].parameters():
        p.requires_grad = False
    for p in model.features[14:].parameters():
        p.requires_grad = True
    for p in model.classifier.parameters():
        p.requires_grad = True

    top_net = nn.Sequential(
        model.features[14:],
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(1),
        model.classifier
    )

    optimizer_s2 = torch.optim.Adam([
        {"params": model.features[14:].parameters(), "lr": 1e-5},
        {"params": model.classifier.parameters(), "lr": 1e-5},
    ])
    scheduler_s2 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_s2, mode="min", factor=0.5, patience=2, min_lr=1e-7)
    epochs_no_improve = 0

    for ep in range(1, stage2_epochs + 1):
        # Train Stage 2
        model.features[14:].train()
        model.classifier.train()
        tr_loss_sum, tr_correct, tr_total = 0.0, 0, 0

        for f13, _, y in train_loader:
            f13, y = f13.to(device), y.to(device)
            optimizer_s2.zero_grad()
            out = top_net(f13)
            loss = criterion(out, y)
            loss.backward()
            optimizer_s2.step()

            tr_loss_sum += loss.item() * y.size(0)
            tr_correct += out.argmax(dim=1).eq(y).sum().item()
            tr_total += y.size(0)

        tr_loss = tr_loss_sum / tr_total
        tr_acc = tr_correct / tr_total

        # Val Stage 2
        top_net.eval()
        va_loss_sum, va_correct, va_total = 0.0, 0, 0
        with torch.no_grad():
            for f13, _, y in val_loader:
                f13, y = f13.to(device), y.to(device)
                out = top_net(f13)
                loss = criterion(out, y)
                va_loss_sum += loss.item() * y.size(0)
                va_correct += out.argmax(dim=1).eq(y).sum().item()
                va_total += y.size(0)

        va_loss = va_loss_sum / va_total
        va_acc = va_correct / va_total
        curr_lr = optimizer_s2.param_groups[0]["lr"]
        scheduler_s2.step(va_loss)

        history["epoch"].append(global_epoch)
        history["stage"].append(2)
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)
        history["lr"].append(curr_lr)

        improved = ""
        if va_loss < best_val_loss:
            best_val_loss = va_loss
            best_model_weights = copy.deepcopy(model.state_dict())
            torch.save(best_model_weights, str(best_weights_path))
            improved = " [SAVED BEST]"
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        print(
            f"Epoch {global_epoch:2d} (S2-{ep:02d}) | "
            f"Tr Loss: {tr_loss:.4f} Acc: {tr_acc*100:.2f}% | "
            f"Val Loss: {va_loss:.4f} Acc: {va_acc*100:.2f}% | "
            f"LR: {curr_lr:.1e}{improved}",
            flush=True
        )
        global_epoch += 1

        if epochs_no_improve >= patience:
            print(f"[*] Early stopping in Stage 2 after {ep} epochs.", flush=True)
            break

    elapsed_time = time.time() - total_start_time

    # ----------------- TEST EVALUATION -----------------
    print("\n[*] Evaluating Best Checkpoint on Test Set...", flush=True)
    model.load_state_dict(best_model_weights)
    top_net.eval()

    test_loss_sum, test_correct, test_total = 0.0, 0, 0
    y_true = []
    y_pred = []

    with torch.no_grad():
        for f13, _, y in test_loader:
            f13, y = f13.to(device), y.to(device)
            out = top_net(f13)
            loss = criterion(out, y)
            test_loss_sum += loss.item() * y.size(0)
            preds = out.argmax(dim=1).cpu().numpy()
            y_pred.extend(preds)
            y_true.extend(y.cpu().numpy())
            test_total += y.size(0)

    test_loss = test_loss_sum / test_total
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    test_acc = (y_pred == y_true).mean()

    macro_f1 = f1_score(y_true, y_pred, average="macro")
    weighted_f1 = f1_score(y_true, y_pred, average="weighted")

    # Classification Report
    cls_report_dict = classification_report(
        y_true, y_pred, target_names=TARGET_CLASSES, output_dict=True, digits=4
    )
    cls_report_txt = classification_report(
        y_true, y_pred, target_names=TARGET_CLASSES, digits=4
    )

    with open(out_dir / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(cls_report_txt)

    pd.DataFrame(cls_report_dict).transpose().to_csv(out_dir / "classification_report.csv")

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    short_labels = [c.replace("Grape___", "") for c in TARGET_CLASSES]
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=short_labels,
        yticklabels=short_labels,
        cbar=False,
    )
    plt.title(f"Confusion Matrix: {dataset_name}\nTest Acc: {test_acc*100:.2f}% | Macro-F1: {macro_f1:.4f}", fontsize=12)
    plt.xlabel("Predicted Label", fontweight="bold")
    plt.ylabel("Ground Truth", fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrix.png", dpi=200)
    plt.close()

    # Training Curves
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history["epoch"], history["train_loss"], label="Train Loss", color="royalblue", lw=2)
    plt.plot(history["epoch"], history["val_loss"], label="Val Loss", color="crimson", lw=2)
    s2_start = next((e for e, s in zip(history["epoch"], history["stage"]) if s == 2), None)
    if s2_start:
        plt.axvline(x=s2_start, color="gray", linestyle="--", label="Stage 2 Start")
    plt.title(f"Loss Curves - {dataset_name}")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(history["epoch"], [a * 100 for a in history["train_acc"]], label="Train Acc", color="royalblue", lw=2)
    plt.plot(history["epoch"], [a * 100 for a in history["val_acc"]], label="Val Acc", color="crimson", lw=2)
    if s2_start:
        plt.axvline(x=s2_start, color="gray", linestyle="--", label="Stage 2 Start")
    plt.title(f"Accuracy Curves - {dataset_name}")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "accuracy_loss_curves.png", dpi=200)
    plt.close()

    metrics = {
        "dataset": dataset_name,
        "test_accuracy": float(test_acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "test_loss": float(test_loss),
        "epochs_trained": len(history["epoch"]),
        "training_time_seconds": float(elapsed_time),
        "best_val_loss": float(best_val_loss),
    }

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\n" + "=" * 70, flush=True)
    print(f"Results for [{dataset_name}]:", flush=True)
    print(f"  Test Accuracy : {test_acc*100:.2f}%", flush=True)
    print(f"  Macro F1      : {macro_f1:.4f}", flush=True)
    print(f"  Weighted F1   : {weighted_f1:.4f}", flush=True)
    print(f"  Training Time : {elapsed_time:.1f}s", flush=True)
    print("=" * 70 + "\n", flush=True)

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train MobileNetV2 on preprocessed Grape leaf datasets.")
    parser.add_argument("--dataset", type=str, default="grape_resized", help="Specific dataset folder to train on.")
    parser.add_argument("--all", action="store_true", help="Train sequentially on all 12 preprocessed datasets.")
    parser.add_argument("--data-dir", type=str, default="data", help="Base data directory.")
    parser.add_argument("--split-file", type=str, default="split.json", help="Path to split.json file.")
    parser.add_argument("--results-dir", type=str, default="results", help="Directory to save models and metrics.")
    parser.add_argument("--batch-size", type=int, default=64, help="Mini-batch size.")
    parser.add_argument("--stage1-epochs", type=int, default=5, help="Epochs for Stage 1 (head only).")
    parser.add_argument("--stage2-epochs", type=int, default=4, help="Epochs for Stage 2 (fine-tuning).")
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience.")
    parser.add_argument("--device", type=str, default="auto", help="Compute device ('cuda', 'cpu', 'auto').")
    parser.add_argument("--force", action="store_true", help="Force retraining even if results exist.")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    data_base = project_root / args.data_dir
    split_path = project_root / args.split_file
    results_base = project_root / args.results_dir

    if not split_path.exists():
        print(f"[-] Error: split file not found at {split_path.resolve()}. Run split_data.py first!", flush=True)
        return

    targets = ALL_TECHNIQUES if args.all else [args.dataset]

    all_metrics = []
    for tgt in targets:
        m = run_training_pipeline(
            dataset_name=tgt,
            data_base=data_base,
            split_path=split_path,
            results_base=results_base,
            batch_size=args.batch_size,
            stage1_epochs=args.stage1_epochs,
            stage2_epochs=args.stage2_epochs,
            patience=args.patience,
            device_name=args.device,
            force=args.force,
        )
        all_metrics.append(m)

    print("\n[+] All requested training runs completed successfully.", flush=True)


if __name__ == "__main__":
    main()
