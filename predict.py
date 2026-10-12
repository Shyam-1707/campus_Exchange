"""Inference script for Grape Disease Classification.

Usage:
  python predict.py --image path/to/leaf.jpg
  python predict.py --image path/to/leaf.jpg --model results/grape_augmented/best_model.pt
"""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

TARGET_CLASSES = [
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
]

FRIENDLY_NAMES = {
    "Grape___Black_rot": "Black Rot (Guignardia bidwellii)",
    "Grape___Esca_(Black_Measles)": "Esca / Black Measles",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": "Leaf Blight (Isariopsis Leaf Spot)",
    "Grape___healthy": "Healthy Leaf",
}


def load_model(weights_path: Path, device: torch.device) -> nn.Module:
    """Load MobileNetV2 with custom classification head and trained weights."""
    model = models.mobilenet_v2(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, len(TARGET_CLASSES))
    )

    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def predict_image(
    image_path: Path,
    model: nn.Module,
    device: torch.device
) -> Tuple[str, float, Dict[str, float]]:
    """Predict disease class and probabilities for a single leaf image."""
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    with Image.open(image_path) as img:
        rgb_img = img.convert("RGB")

    tensor = preprocess(rgb_img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    top_idx = int(probs.argmax())
    predicted_class = TARGET_CLASSES[top_idx]
    confidence = float(probs[top_idx])

    all_probs = {cls_name: float(probs[i]) for i, cls_name in enumerate(TARGET_CLASSES)}
    return predicted_class, confidence, all_probs


def main():
    parser = argparse.ArgumentParser(description="Classify a grape leaf disease from an image.")
    parser.add_argument("--image", type=str, required=True, help="Path to input grape leaf image.")
    parser.add_argument(
        "--model",
        type=str,
        default="results/grape_augmented/best_model.pt",
        help="Path to trained PyTorch model checkpoint.",
    )
    parser.add_argument("--device", type=str, default="auto", help="Compute device ('cuda', 'cpu', 'auto').")
    args = parser.parse_args()

    img_path = Path(args.image)
    if not img_path.exists():
        print(f"[-] Error: Image file '{img_path}' does not exist.")
        return

    weights_path = Path(args.model)
    if not weights_path.exists():
        print(f"[-] Error: Model checkpoint '{weights_path}' does not exist.")
        return

    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) or args.device == "cuda" else "cpu")
    print(f"[*] Loading model from: {weights_path}")
    print(f"[*] Compute device: {device}")
    model = load_model(weights_path, device)

    predicted_class, confidence, all_probs = predict_image(img_path, model, device)

    print("\n" + "=" * 65)
    print(f"  IMAGE: {img_path.name}")
    print("=" * 65)
    print(f"  PREDICTED CLASS : {predicted_class}")
    print(f"  DIAGNOSIS       : {FRIENDLY_NAMES.get(predicted_class, predicted_class)}")
    print(f"  CONFIDENCE      : {confidence * 100:.2f}%")
    print("-" * 65)
    print("  Class Probabilities:")
    for cls_name, prob in sorted(all_probs.items(), key=lambda x: x[1], reverse=True):
        bar = "#" * int(prob * 30)
        print(f"    - {cls_name:<42} : {prob*100:6.2f}% | {bar}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
