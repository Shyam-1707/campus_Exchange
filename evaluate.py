"""Comparative Evaluation and Benchmarking Script.

Aggregates metrics from all trained preprocessing pipelines in results/*/,
compiles results/comparison.csv, generates an attractive comparison bar chart
(results/comparison_chart.png) highlighting test accuracy and Macro-F1,
and determines the top-performing preprocessing technique.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def gather_results(results_dir: Path) -> pd.DataFrame:
    """Collect metrics from all subfolders in results/ containing metrics.json."""
    records = []

    for sub in sorted(results_dir.iterdir()):
        if sub.is_dir():
            metric_file = sub / "metrics.json"
            if metric_file.exists():
                try:
                    with open(metric_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    records.append({
                        "Technique": data.get("dataset", sub.name),
                        "Test Accuracy (%)": round(data.get("test_accuracy", 0.0) * 100, 2),
                        "Macro F1": round(data.get("macro_f1", 0.0), 4),
                        "Weighted F1": round(data.get("weighted_f1", 0.0), 4),
                        "Test Loss": round(data.get("test_loss", 0.0), 4),
                        "Training Time (s)": round(data.get("training_time_seconds", 0.0), 1),
                        "Best Val Loss": round(data.get("best_val_loss", 0.0), 4),
                    })
                except Exception as e:
                    print(f"[-] Error reading {metric_file}: {e}")

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    # Sort by Test Accuracy descending, then Macro F1 descending
    df = df.sort_values(by=["Test Accuracy (%)", "Macro F1"], ascending=[False, False]).reset_index(drop=True)
    return df


def plot_comparison_chart(df: pd.DataFrame, output_path: Path):
    """Generate high-resolution comparative bar chart for Accuracy and Macro-F1."""
    if df.empty:
        return

    fig, ax1 = plt.subplots(figsize=(14, 7))

    x = np.arange(len(df))
    width = 0.38

    # Format labels for clean display
    labels = [t.replace("grape_", "") for t in df["Technique"]]

    bars1 = ax1.bar(x - width / 2, df["Test Accuracy (%)"], width, label="Test Accuracy (%)", color="#2b5c8f", edgecolor="black", alpha=0.9)
    ax1.set_ylabel("Test Accuracy (%)", fontsize=12, fontweight="bold", color="#2b5c8f")
    ax1.set_ylim(0, 105)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=30, ha="right", fontsize=11, fontweight="bold")
    ax1.grid(axis="y", linestyle="--", alpha=0.4)

    # Secondary axis for Macro F1
    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + width / 2, df["Macro F1"], width, label="Macro F1", color="#e67e22", edgecolor="black", alpha=0.9)
    ax2.set_ylabel("Macro-F1 Score", fontsize=12, fontweight="bold", color="#e67e22")
    ax2.set_ylim(0, 1.05)

    # Value labels on bars
    for b in bars1:
        height = b.get_height()
        ax1.annotate(f"{height:.1f}%",
                     xy=(b.get_x() + b.get_width() / 2, height),
                     xytext=(0, 3), textcoords="offset points",
                     ha="center", va="bottom", fontsize=8, fontweight="bold")

    for b in bars2:
        height = b.get_height()
        ax2.annotate(f"{height:.3f}",
                     xy=(b.get_x() + b.get_width() / 2, height),
                     xytext=(0, 3), textcoords="offset points",
                     ha="center", va="bottom", fontsize=8, fontweight="bold")

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", framealpha=0.9)

    best_tech = df.iloc[0]["Technique"]
    best_acc = df.iloc[0]["Test Accuracy (%)"]
    best_f1 = df.iloc[0]["Macro F1"]

    plt.title(
        f"Grape Leaf Preprocessing Comparison (MobileNetV2 Transfer Learning)\n"
        f"Top Technique: {best_tech} (Acc: {best_acc:.2f}%, Macro-F1: {best_f1:.4f})",
        fontsize=14, fontweight="bold", pad=12
    )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved comparison bar chart to: {output_path.resolve()}")


def main():
    parser = argparse.ArgumentParser(description="Generate benchmark comparison across preprocessing methods.")
    parser.add_argument("--results-dir", type=str, default="results", help="Directory where model results are stored.")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    results_dir = project_root / args.results_dir

    if not results_dir.exists():
        print(f"[-] Error: Results directory {results_dir.resolve()} does not exist.")
        return

    df = gather_results(results_dir)
    if df.empty:
        print(f"[-] No completed training runs found in {results_dir.resolve()}")
        return

    # Save comparison CSV
    csv_out = results_dir / "comparison.csv"
    df.to_csv(csv_out, index=False)
    print(f"\n[+] Saved comparison table to: {csv_out.resolve()}")

    # Save comparison Chart
    chart_out = results_dir / "comparison_chart.png"
    plot_comparison_chart(df, chart_out)

    # Print summary table
    print("\n" + "=" * 88)
    print("                     PREPROCESSING BENCHMARK COMPARISON TABLE")
    print("=" * 88)
    print(df.to_string(index=False))
    print("=" * 88)

    best = df.iloc[0]
    print(f"\n[*] BEST PERFORMING TECHNIQUE: {best['Technique']}")
    print(f"    - Test Accuracy : {best['Test Accuracy (%)']}%")
    print(f"    - Macro F1      : {best['Macro F1']}")
    print(f"    - Test Loss     : {best['Test Loss']}")
    print(f"    - Training Time : {best['Training Time (s)']}s\n")


if __name__ == "__main__":
    main()
