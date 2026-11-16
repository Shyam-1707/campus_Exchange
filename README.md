
<div align="center">
  <img src="./Content/assets/Campxchange-home-banner.png" alt="CampusXchange Banner" width="100%" />
</div>

<h1 align="center">CampusXchange (CampX)</h1>

<p align="center">
  <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=500&size=20&duration=2500&pause=800&color=3F6B3E&center=true&vCenter=true&width=460&lines=A+campus+thrift+board%2C+not+a+storefront.;Verified+students.+Real+items.+Real+trust.;List+it.+Tag+it.+Trade+it." alt="typing banner" />
</p>

<p align="center">
  <img src="https://img.shields.io/badge/stack-MERN-3F6B3E?style=flat-square" />
  <img src="https://img.shields.io/badge/status-in%20development-E0A72E?style=flat-square" />
  <img src="https://img.shields.io/badge/scope-3--month%20build-17181A?style=flat-square" />
</p>

> The typing line above is a live animated SVG (regenerates on every view) — the closest thing a Markdown README gets to real motion without embedding a GIF. If it doesn't render on your platform, it's `readme-typing-svg`, a well-known GitHub README trick.

---

## What this is

CampX is a **campus-only marketplace** where verified students buy and sell used items — books, electronics, furniture — with each other. The trust comes from identity verification (college email + PRN + ID card review), not from being another unverified listing site. Think secondhand bookstore / notice board, digitized.

## Docs — read in this order

| Doc | What's in it |
|---|---|
| [`prd.md`](./prd.md) | Requirements, user stories, scope, risks |
| [`architecture.md`](./architecture.md) | System design, data model, request flows |
| [`design.md`](./design.md) | Visual identity, tokens, micro-interaction spec |
| [`phases.md`](./phases.md) | 12-week build plan with exit criteria |
| [`rules.md`](./rules.md) | Always-apply rules for AI coding editors working in this repo |

## Stack

**Frontend** — React + Redux Toolkit + React Router + Tailwind CSS
**Backend** — Node.js + Express (REST)
**Database** — MongoDB (Atlas)
**Verification service** — Python + FastAPI (OCR-based PRN/ID matching)
**Images** — Cloudinary
**Payments** — Razorpay
**Email** — Nodemailer / SendGrid

## Core flow

```
sign up → email OTP → ID card + PRN check → verified
                                              │
                                              ▼
                              list an item (photo, category, price)
                                              │
                                              ▼
                            buyer finds it → cart → Razorpay checkout
                                              │
                                    webhook confirms payment
                                              │
                                              ▼
                          receipt + pickup details emailed to buyer
```

## Getting started

```bash
# clone and enter the repo
git clone <repo-url>
cd campusxchange

# install dependencies
npm --prefix client install
npm --prefix server install

# configure server/.env with a working PostgreSQL DATABASE_URL, then create
# the schema and import the student roster from StudentDB.csv
npm run migrate
npm run import:students

# run the API and frontend in separate terminals
npm run dev:server
npm run dev:client

# optional verification service
cd verification-service
pip install -r requirements.txt
uvicorn main:app --reload
```

Copy `.env.example` → `.env` in `client/`, `server/`, and `verification-service/`.
Configure `DATABASE_URL`, JWT secrets, and any required `CLOUDINARY_*`,
`RAZORPAY_*`, and `SMTP_*` values in `server/.env`. The server imports students
from the root `StudentDB.csv`; PRN is the student roster primary key, and CSV
passwords are bcrypt-hashed during import. Keep that CSV private and do not
commit it. Set `FIREBASE_PROJECT_ID` in `server/.env` to the same project ID as
`VITE_FIREBASE_PROJECT_ID` in `client/.env`. Google sign-in verifies the Firebase
ID token on the backend and only grants access to a verified Google email in
the imported student roster. Enable Google under Firebase Authentication
providers and add `localhost` plus the deployed client hostname to its
authorized domains.

## Vercel deployment

The root `vercel.json` configures three services: `client` serves `/` and the SPA fallback, `server` serves `/api/*` and `/socket.io/*`, and `verification-service` remains internal. Frontend asset paths are routed to their actual files; other frontend paths use `index.html` so React Router can handle direct navigation. The `server` service receives the verification service URL through the `VERIFICATION_SERVICE_URL` binding; do not set that variable manually in Vercel.

Configure the client build variables and backend secrets in Vercel for the appropriate services. The Socket.IO route is included because the client uses it, but the current Socket.IO server is initialized only by local HTTP startup. Live chat therefore needs a WebSocket-capable deployment/integration; the chat UI has a REST fallback.

## Folder structure

```
/client                 → React app
/server                 → Express API
/verification-service   → FastAPI OCR service
/content/gif/account-login → post-auth celebration GIFs (see prompt-signin-signup.md)
```

## Design snapshot

Bone `#F2EFE6` · Ink `#17181A` · Moss `#3F6B3E` · Marigold `#E0A72E` · Clay `#C9C2B2` · Rust `#B5482A`
General Sans (headings) · Inter (body) · IBM Plex Mono (data only — prices, PRNs, order IDs)

Full spec, including the hover/click/icon motion rules, lives in [`design.md`](./design.md).

---

<div align="center">
<sub>built by students, for students — one price tag at a time 🏷️</sub>
</div>
=======
# Grape Leaf Disease Classification: Preprocessing Benchmark & Transfer Learning

A complete, production-ready computer vision and deep learning project that extracts grape leaf pathology images from the **PlantVillage** dataset, benchmarks **12 distinct image preprocessing and computer vision techniques**, and trains a transfer learning classifier (**MobileNetV2**) with two-stage fine-tuning.

---

## 🍇 Target Grape Classes
The project filters and trains on the 4 Grape classes:
1. `Grape___Black_rot` (*Guignardia bidwellii*)
2. `Grape___Esca_(Black_Measles)` (*Phaeomoniella chlamydospora*, etc.)
3. `Grape___Leaf_blight_(Isariopsis_Leaf_Spot)` (*Pseudocercospora vitis*)
4. `Grape___healthy`

---

## 🛠 Project Structure

```text
sem project 1ni/
 ├─ download_data.py             # Downloads PlantVillage and extracts Grape classes
 ├─ preprocess.py                # Applies 12 CV preprocessing techniques (OpenCV + NumPy)
 ├─ split_data.py                # Stratified 70/15/15 train/val/test split (split.json)
 ├─ train.py                     # Two-stage MobileNetV2 transfer learning pipeline
 ├─ evaluate.py                  # Benchmarks all techniques & generates comparison charts
 ├─ requirements.txt             # Environment dependencies
 ├─ README.md                    # Documentation & run instructions
 ├─ split.json                   # Unified stratified train/val/test manifest
 ├─ preprocessing_summary.csv    # Summary of preprocessed image counts
 ├─ preview/
 │   └─ preprocessing_samples.png# Visual 12-panel comparison grid of sample leaf
 ├─ data/
 │   ├─ grape_original/          # Extracted original color images
 │   ├─ grape_resized/           # 224x224 INTER_AREA resizing
 │   ├─ grape_grayscale/         # Grayscale (3-channel)
 │   ├─ grape_filtered/          # Gaussian (5x5) + Median filter
 │   ├─ grape_equalized/         # LAB CLAHE on L channel
 │   ├─ grape_sharpened/         # Unsharp masking
 │   ├─ grape_edges/             # Canny edge detection
 │   ├─ grape_negative/          # Photographic negative (255 - image)
 │   ├─ grape_blended/           # Alpha blend (0.5 original + 0.5 sharpened)
 │   ├─ grape_morphology/        # Otsu thresholding + opening + closing
 │   ├─ grape_segmented/         # HSV + Otsu mask background removal
 │   ├─ grape_normalized/        # Min-Max normalized (0-255)
 │   └─ grape_augmented/         # Data augmented (flips, rotation, brightness)
 └─ results/
     ├─ <technique_name>/
     │   ├─ best_model.pt        # Saved PyTorch model checkpoint
     │   ├─ accuracy_loss_curves.png
     │   ├─ confusion_matrix.png
     │   ├─ classification_report.txt
     │   ├─ classification_report.csv
     │   └─ metrics.json
     ├─ comparison.csv           # Benchmark comparison table across all techniques
     └─ comparison_chart.png     # Dual-axis bar chart (Accuracy vs Macro-F1)
```

---

## 🔬 Preprocessing Techniques Overview

Each technique is saved to its own directory under `data/`, preserving the original class subfolder structure and filenames:

| Technique | Folder Name | Description |
| :--- | :--- | :--- |
| **Resized** | `data/grape_resized/` | Resized to 224x224 using `cv2.INTER_AREA` (baseline) |
| **Grayscale** | `data/grape_grayscale/` | Grayscale conversion replicated across 3 RGB channels |
| **Filtered** | `data/grape_filtered/` | Noise suppression: 5x5 Gaussian blur + 5x5 median blur |
| **Equalized** | `data/grape_equalized/` | Contrast enhancement: CLAHE on Lightness (L) channel in LAB color space |
| **Sharpened** | `data/grape_sharpened/` | Edge enhancement via unsharp masking |
| **Edges** | `data/grape_edges/` | Canny edge map (3-channel) |
| **Negative** | `data/grape_negative/` | Inverted pixel intensities (`255 - image`) |
| **Blended** | `data/grape_blended/` | 50/50 linear blend (`cv2.addWeighted`) of original and sharpened |
| **Morphology** | `data/grape_morphology/` | Otsu threshold followed by opening & closing using 5x5 elliptical kernel |
| **Segmented** | `data/grape_segmented/` | Leaf mask via HSV green-range + Otsu saturation threshold with `bitwise_and` |
| **Normalized** | `data/grape_normalized/` | Min-Max scaling mapped to full dynamic range [0, 255] |
| **Augmented** | `data/grape_augmented/` | Horizontal flip, 90° rotation, and brightness shifts applied to train split |

---

## ⚙️ Installation & Setup

1. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Kaggle API Credentials (Optional):**
   If you have a Kaggle account:
   - Place your `kaggle.json` at `C:\Users\<YourUsername>\.kaggle\kaggle.json` (or set `KAGGLE_USERNAME` and `KAGGLE_KEY` environment variables).
   - If missing, `kagglehub` will still attempt download of public datasets.

---

## 🚀 Execution Guide

### 🌟 Quick Start (All-in-One CLI)
You can run the project interactively or with quick flags using [main.py](file:///c:/Users/shyam/Downloads/sem%20project%201ni/main.py):
```bash
# Launch interactive terminal menu
python main.py

# Or quick commands:
python main.py --demo                      # Run random test prediction
python main.py --evaluate                  # View benchmark table & chart
python main.py --predict "path/to/img.jpg" # Predict disease on an image
```

---

### Step 1: Download & Extract Grape Classes
```bash
python download_data.py
```
*Downloads the PlantVillage color dataset, extracts the 4 Grape classes, and copies them to `data/grape_original/<class_name>/`.*

### Step 2: Apply Preprocessing Techniques
```bash
python preprocess.py
```
*Processes every Grape leaf image through the 12 techniques, writes `preprocessing_summary.csv`, and generates visual comparison grid `preview/preprocessing_samples.png`.*

### Step 3: Stratified Train / Validation / Test Split
```bash
python split_data.py
```
*Creates a deterministic 70% / 15% / 15% stratified split based on original filenames (seed 42) and saves to `split.json`.*

### Step 4: Model Training (MobileNetV2 Transfer Learning)
- **Train on a specific technique:**
  ```bash
  python train.py --dataset grape_segmented
  ```
- **Train on all 12 techniques sequentially:**
  ```bash
  python train.py --all
  ```
  *Two-stage training:*
  - **Stage 1 (Head Training):** Base model frozen, trains custom head for 10 epochs (Adam lr=1e-3).
  - **Stage 2 (Fine-Tuning):** Top feature blocks unfrozen for 10 epochs (Adam lr=1e-5).
  - Includes `EarlyStopping`, `ReduceLROnPlateau`, and `ModelCheckpoint`.

### Step 5: Comparative Evaluation & Benchmarking
```bash
python evaluate.py
```
*Scans all trained runs, produces `results/comparison.csv`, renders `results/comparison_chart.png`, and identifies the highest performing technique.*

### Step 6: Single-Image Inference / Prediction
```bash
# Using the best overall model (grape_augmented)
python predict.py --image "data/grape_original/Grape___healthy/<image_name>.jpg"

# Or specifying any trained technique checkpoint
python predict.py --image "data/grape_original/Grape___Black_rot/<image_name>.jpg" --model "results/grape_segmented/best_model.pt"
```

---

## 📊 Final Benchmark Results

Aggregated test set evaluation results across all 12 preprocessing techniques (MobileNetV2, 610 held-out test images):

| Rank | Technique | Test Accuracy (%) | Macro-F1 | Weighted-F1 | Test Loss | Best Val Loss |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| 🥇 | **grape_augmented** | **99.67%** | **0.9972** | **0.9967** | **0.0101** | **0.0113** |
| 🥈 | **grape_resized** (baseline) | 99.51% | 0.9961 | 0.9951 | 0.0166 | 0.0210 |
| 🥉 | **grape_segmented** | 99.51% | 0.9961 | 0.9951 | 0.0210 | 0.0232 |
| 4 | **grape_equalized** (CLAHE) | 99.18% | 0.9933 | 0.9918 | 0.0250 | 0.0240 |
| 5 | **grape_normalized** | 99.02% | 0.9920 | 0.9902 | 0.0297 | 0.0287 |
| 6 | **grape_blended** | 98.69% | 0.9894 | 0.9869 | 0.0322 | 0.0258 |
| 7 | **grape_sharpened** | 98.69% | 0.9892 | 0.9869 | 0.0375 | 0.0299 |
| 8 | **grape_negative** | 98.03% | 0.9843 | 0.9803 | 0.0472 | 0.0362 |
| 9 | **grape_filtered** | 97.87% | 0.9830 | 0.9787 | 0.0563 | 0.0728 |
| 10 | **grape_grayscale** | 94.75% | 0.9566 | 0.9475 | 0.1423 | 0.1255 |
| 11 | **grape_edges** (Canny) | 86.39% | 0.8734 | 0.8623 | 0.3168 | 0.3049 |
| 12 | **grape_morphology** | 69.84% | 0.7019 | 0.6750 | 0.6961 | 0.6947 |

### 🔍 Key Computer Vision & Deep Learning Insights
1. **Data Augmentation dominates**: Expanding the training set with geometric and photometric variations achieved the highest accuracy (**99.67%**) and lowest loss (**0.0101**).
2. **Background Segmentation is highly effective**: Removing non-leaf background noise matched baseline resized accuracy (**99.51%**) while ensuring features are strictly leaf-focused.
3. **Color and Texture Importance**: Discarding color information (`grape_grayscale`: 94.75%) or texture (`grape_edges`: 86.39%, `grape_morphology`: 69.84%) significantly hinders disease differentiation because pathogens like Black Rot, Esca, and Leaf Blight rely heavily on chromatic lesion patterns.
>>>>>>> 0661c2a231ac7d3ae0ccfdb6adcdae8ca9d03a21
