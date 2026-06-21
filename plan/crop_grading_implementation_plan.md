# Pakistani Crop Quality Grading System — Complete Implementation Plan

> **Project:** AI-powered crop quality grader for Pakistani agriculture  
> **Target:** Grade 6 core crops (A/B/C/D) with 85–95% accuracy; 40–60% on unknown crops  
> **Architecture:** Multi-task CNN (EfficientNet backbone) + AgriCLIP fallback  
> **Timeline:** ~16 weeks (MVP)

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Architecture Design](#2-architecture-design)
3. [Phase 1 — Dataset Collection & Preparation (Weeks 1–4)](#3-phase-1--dataset-collection--preparation-weeks-14)
4. [Phase 2 — Model Development (Weeks 5–10)](#4-phase-2--model-development-weeks-510)
5. [Phase 3 — Evaluation & Tuning (Weeks 11–12)](#5-phase-3--evaluation--tuning-weeks-1112)
6. [Phase 4 — Deployment (Weeks 13–16)](#6-phase-4--deployment-weeks-1316)
7. [Project Structure](#7-project-structure)
8. [Tech Stack](#8-tech-stack)
9. [Grading Standards Reference](#9-grading-standards-reference)
10. [Risk Register](#10-risk-register)
11. [Milestones Checklist](#11-milestones-checklist)

---

## 1. System Overview

### What It Does

A farmer photographs a crop using a mobile phone. The system:

1. Detects what crop is in the image (crop identification head)
2. Grades its quality as **A, B, C, or D** (quality grading head)
3. Returns a confidence score — if low, routes to a VLM fallback for unknown crops
4. Displays the grade with a simple explanation the farmer can act on

### The Core Insight

One shared backbone extracts visual features from any crop image. Two lightweight heads run on top of those features simultaneously — one for "what is it?" and one for "how good is it?" This means adding a new crop later only requires collecting new labeled images and retraining the heads — not rebuilding the system.

### Target Crops (Phase 1)

| # | Crop | Pakistani Relevance | Dataset Availability |
|---|------|---------------------|----------------------|
| 1 | Wheat (گندم) | Largest food crop, ~9.5M hectares | Good (GrainSet, Kaggle) |
| 2 | Rice (چاول) | Major export crop | Excellent (75k images) |
| 3 | Mango (آم) | Top fruit export | Excellent (Pakistani dataset on Mendeley) |
| 4 | Maize (مکئی) | Growing importance, animal feed | Good (GrainSet) |
| 5 | Cotton (کپاس) | Industrial crop, major revenue | Sparse — custom collection needed |
| 6 | Sugarcane (گنا) | Second-largest crop area | Sparse — custom collection needed |

---

## 2. Architecture Design

### 2.1 High-Level Pipeline

```
Farmer Image
     │
     ▼
┌─────────────────────────────────┐
│   Preprocessing                 │
│   - Resize to 224×224           │
│   - Normalize (ImageNet stats)  │
│   - Augmentation (train only)   │
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│   Shared Backbone               │
│   EfficientNet-B3               │
│   (pretrained on AgriCLIP/      │
│    ImageNet weights)            │
│   Output: 1536-dim feature vec  │
└──────┬──────────────────┬───────┘
       │                  │
       ▼                  ▼
┌─────────────┐   ┌─────────────────┐
│ Crop ID     │   │ Quality Grader  │
│ Head        │   │ Head            │
│ FC(1536,512)│   │ FC(1536,512)    │
│ ReLU        │   │ ReLU            │
│ Dropout 0.3 │   │ Dropout 0.3     │
│ FC(512, 6)  │   │ FC(512, 4)      │
│ Softmax     │   │ Softmax         │
│             │   │                 │
│ 6 crops     │   │ A / B / C / D   │
└──────┬──────┘   └────────┬────────┘
       │                   │
       └────────┬──────────┘
                │
                ▼
       ┌────────────────┐
       │ Confidence     │
       │ Threshold      │
       │ check          │
       └───────┬────────┘
               │
       ┌───────┴────────┐
       │                │
  conf ≥ 0.70      conf < 0.70
       │                │
       ▼                ▼
  Final Output    AgriCLIP VLM
  (high trust)    Fallback
                  (low confidence
                   flag shown)
```

### 2.2 Model Details

**Backbone: EfficientNet-B3**
- Pretrained weights: Start from AgriCLIP fine-tuned checkpoint, else ImageNet
- Output feature dimension: 1536
- Frozen for first 3 epochs, then unfreeze top 30% of layers

**Crop ID Head**
- Linear(1536 → 512) → BatchNorm → ReLU → Dropout(0.3) → Linear(512 → 6)
- Loss: CrossEntropyLoss
- Output: crop class probabilities for 6 crops

**Quality Grader Head**
- Linear(1536 → 512) → BatchNorm → ReLU → Dropout(0.3) → Linear(512 → 4)
- Loss: CrossEntropyLoss with class weights (handle grade imbalance)
- Output: A, B, C, D probabilities

**Total trainable parameters (approx):** ~14M (backbone) + ~1.6M (heads) = ~15.6M

### 2.3 Loss Function

```python
total_loss = (λ1 * crop_id_loss) + (λ2 * quality_loss)
# λ1 = 0.4, λ2 = 0.6 (quality grading is primary task)
```

### 2.4 AgriCLIP Fallback

When backbone confidence < 0.70:
- Pass image + text prompt to AgriCLIP
- Prompt template: `"A photo of [crop_name] with [grade] quality: well-formed, no defects, good color"`
- Compute cosine similarity between image embedding and each grade text embedding
- Return highest similarity grade with `low_confidence=True` flag

---

## 3. Phase 1 — Dataset Collection & Preparation (Weeks 1–4)

### 3.1 Dataset Sources

#### Rice (target: 8,000+ images)
- **Primary:** Rice Image Dataset on Kaggle — 75,000 grain images across 5 varieties
  - URL: `https://www.kaggle.com/datasets/muratkokludataset/rice-image-dataset`
  - Filter to quality-relevant images, label A/B/C/D by grain fullness, color, broken %
- **Secondary:** Rice Quality Dataset on Kaggle (broken, full, mixed categories)

#### Wheat (target: 6,000+ images)
- **Primary:** GrainSet dataset — 350,000+ single-kernel images, wheat included
  - Paper: GrainSet (2023), request access via corresponding author
- **Secondary:** Wheat grain Kaggle datasets (germinated, diseased, damaged, perfect)
  - Search: `kaggle datasets list -s "wheat grain quality"`

#### Mango (target: 6,000+ images)
- **Primary:** Pakistani Mango Dataset — Mendeley Data
  - URL: `https://data.mendeley.com/datasets/pakistani-mango`
  - 8 Pakistani varieties, graded for export quality
- **Secondary:** Mango leaf + fruit datasets from Roboflow Universe

#### Maize (target: 5,000+ images)
- **Primary:** GrainSet (maize subset, 40,000 images in OOD-GrainSet)
- **Secondary:** Corn/maize quality datasets on Kaggle

#### Cotton (target: 3,000+ images — custom collection required)
- No sufficient public dataset exists for cotton boll quality grading
- See Section 3.3 for custom data collection plan

#### Sugarcane (target: 2,000+ images — custom collection required)
- No sufficient public dataset for sugarcane stalk quality grading
- See Section 3.3 for custom data collection plan

### 3.2 Grading Label Map

Each crop has different quality indicators. Map them to a unified A/B/C/D scale:

**Rice**
| Grade | Criteria |
|-------|----------|
| A | >95% whole grains, no discoloration, uniform size, moisture 12–14% |
| B | 85–95% whole grains, slight discoloration allowed, minor size variation |
| C | 70–85% whole grains, visible broken grains, some discoloration |
| D | <70% whole grains, heavy discoloration, mixed varieties, visible damage |

**Wheat**
| Grade | Criteria |
|-------|----------|
| A | Full kernels, golden color, no visible disease/mold, hard texture |
| B | Mostly full kernels, minor shriveling, slight discoloration |
| C | Visible shriveled/broken kernels, partial discoloration, possible germination |
| D | Heavy damage, mold visible, significant germination, insect damage |

**Mango**
| Grade | Criteria |
|-------|----------|
| A | Uniform size (>300g), no blemishes, full color development, export quality |
| B | Slight size variation, minor surface marks, good color |
| C | Surface defects allowed, size variation, not for export |
| D | Significant bruising, disease spots, overripe, unsuitable for market |

**Cotton**
| Grade | Criteria |
|-------|----------|
| A | Pure white, fully opened bolls, no foreign matter, staple length >28mm |
| B | Mostly white, slight creamy tint, minor foreign matter |
| C | Yellowish/gray tint, visible foreign matter, shorter staple |
| D | Dark discoloration, high foreign matter, immature/weathered bolls |

**Maize**
| Grade | Criteria |
|-------|----------|
| A | Uniform kernel color, no mold, moisture <14%, full cob, no insect damage |
| B | Minor color variation, slight surface damage, moisture 14–16% |
| C | Significant color variation, partial mold, visible insect damage |
| D | Heavy mold, rot, high moisture, severe insect damage |

**Sugarcane**
| Grade | Criteria |
|-------|----------|
| A | High juice content (Brix >18%), clean cuts, no disease, uniform maturity |
| B | Good juice content (Brix 16–18%), minor surface damage |
| C | Moderate juice content (Brix 14–16%), some disease spots, mixed maturity |
| D | Low juice content (<14% Brix), heavy disease, delayed harvest rot |

### 3.3 Custom Data Collection Plan (Cotton & Sugarcane)

**Target:** 3,000 cotton + 2,000 sugarcane images with expert labels

**Collection protocol:**
1. Partner with 2–3 agricultural universities in Punjab (UAF Faisalabad, PMAS Arid Agriculture)
2. Partner with PSQCA (Pakistan Standards and Quality Control Authority) for labeling
3. Visit mandi (wholesale market) in Faisalabad (cotton) and Mardan/Charsadda (sugarcane)

**Photography guidelines for farmers:**
- Shoot in natural daylight, avoid harsh shadows
- Hold phone 30–40cm from subject
- Photograph 200–300g sample spread flat on white paper/cloth
- Take 3 angles: top-down, 45-degree, close-up
- Note: variety name, harvest date, location (district)

**Labeling workflow:**
- Upload raw images to Label Studio (self-hosted, free)
- 2 independent labelers assign A/B/C/D per grading standard
- Third expert resolves conflicts (inter-annotator agreement target: κ > 0.75)
- Export as JSON for training pipeline

**Estimated cost:** Transport + labeler fees ≈ PKR 15,000–25,000

### 3.4 Data Augmentation Strategy

Apply during training, not stored on disk (use `torchvision.transforms`):

```python
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
    transforms.RandomRotation(degrees=30),
    transforms.RandomGrayscale(p=0.05),  # simulate poor lighting
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

val_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
```

**Extra augmentations for sparse classes (cotton/sugarcane):**
- MixUp (alpha=0.2) — blend two images from same crop
- CutMix — cut region from one image, paste into another of same class
- Use `albumentations` library for advanced transforms

### 3.5 Dataset Split

| Split | Ratio | Purpose |
|-------|-------|---------|
| Train | 70% | Model training |
| Validation | 15% | Hyperparameter tuning |
| Test | 15% | Final evaluation only (never peek during dev) |

Split is **stratified by crop AND grade** — each split must have balanced representation.

```python
from sklearn.model_selection import StratifiedShuffleSplit
# Combine crop_label and grade_label into single stratification key
strat_key = [f"{crop}_{grade}" for crop, grade in zip(crops, grades)]
```

### 3.6 Final Dataset Summary (Target)

| Crop | Train | Val | Test | Total |
|------|-------|-----|------|-------|
| Rice | 5,600 | 1,200 | 1,200 | 8,000 |
| Wheat | 4,200 | 900 | 900 | 6,000 |
| Mango | 4,200 | 900 | 900 | 6,000 |
| Maize | 3,500 | 750 | 750 | 5,000 |
| Cotton | 2,100 | 450 | 450 | 3,000 |
| Sugarcane | 1,400 | 300 | 300 | 2,000 |
| **Total** | **21,000** | **4,500** | **4,500** | **30,000** |

---

## 4. Phase 2 — Model Development (Weeks 5–10)

### 4.1 Environment Setup

```bash
# Create conda environment
conda create -n cropgrade python=3.10
conda activate cropgrade

# Core ML
pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu118
pip install timm==0.9.12          # EfficientNet pretrained weights
pip install transformers==4.38.0  # For AgriCLIP/CLIP fallback
pip install open_clip_torch       # AgriCLIP is based on OpenCLIP

# Data & training
pip install albumentations==1.3.1
pip install scikit-learn==1.3.0
pip install pandas numpy matplotlib seaborn
pip install tensorboard wandb      # experiment tracking

# Labeling tool
pip install label-studio           # for custom cotton/sugarcane labeling

# Deployment
pip install onnx onnxruntime
pip install fastapi uvicorn        # API server
pip install pillow
```

### 4.2 Project File Structure

```
crop_grading/
├── data/
│   ├── raw/                    # Downloaded datasets, never modified
│   │   ├── rice/
│   │   ├── wheat/
│   │   ├── mango/
│   │   ├── maize/
│   │   ├── cotton/             # Custom collected
│   │   └── sugarcane/          # Custom collected
│   ├── processed/              # After cleaning and labeling
│   │   ├── train/
│   │   │   ├── rice/
│   │   │   │   ├── A/
│   │   │   │   ├── B/
│   │   │   │   ├── C/
│   │   │   │   └── D/
│   │   │   └── ... (other crops)
│   │   ├── val/
│   │   └── test/
│   └── metadata/
│       ├── train_labels.csv
│       ├── val_labels.csv
│       └── test_labels.csv
├── models/
│   ├── backbone/               # Pretrained checkpoints
│   ├── checkpoints/            # Training checkpoints
│   └── exports/                # ONNX exports for deployment
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py          # PyTorch Dataset class
│   │   ├── transforms.py       # Augmentation pipelines
│   │   └── dataloader.py       # DataLoader setup
│   ├── models/
│   │   ├── __init__.py
│   │   ├── backbone.py         # EfficientNet feature extractor
│   │   ├── heads.py            # Crop ID + Quality grader heads
│   │   ├── multitask_model.py  # Combined model
│   │   └── clip_fallback.py    # AgriCLIP fallback
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py          # Training loop
│   │   ├── losses.py           # Multi-task loss
│   │   └── metrics.py          # Accuracy, F1, confusion matrix
│   ├── inference/
│   │   ├── __init__.py
│   │   ├── predictor.py        # Single image inference
│   │   └── batch_predictor.py  # Batch inference
│   └── utils/
│       ├── __init__.py
│       ├── config.py           # Hyperparameters config
│       └── visualization.py    # Grad-CAM, confusion matrix plots
├── api/
│   ├── main.py                 # FastAPI app
│   ├── routes/
│   │   └── grade.py            # /grade endpoint
│   └── schemas.py              # Request/response Pydantic models
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_baseline_training.ipynb
│   ├── 03_evaluation.ipynb
│   └── 04_error_analysis.ipynb
├── scripts/
│   ├── download_datasets.py
│   ├── prepare_data.py
│   ├── train.py                # Main training entry point
│   ├── evaluate.py
│   └── export_onnx.py
├── configs/
│   └── default.yaml            # All hyperparameters
├── requirements.txt
└── README.md
```

### 4.3 Core Model Code

#### `src/models/multitask_model.py`

```python
import torch
import torch.nn as nn
import timm

class CropGradingModel(nn.Module):
    """
    Multi-task model for simultaneous crop identification and quality grading.
    One shared EfficientNet-B3 backbone, two classification heads.
    """

    CROP_CLASSES = ['wheat', 'rice', 'mango', 'maize', 'cotton', 'sugarcane']
    GRADE_CLASSES = ['A', 'B', 'C', 'D']

    def __init__(
        self,
        backbone_name: str = 'efficientnet_b3',
        num_crops: int = 6,
        num_grades: int = 4,
        dropout_rate: float = 0.3,
        pretrained: bool = True,
    ):
        super().__init__()

        # Shared backbone — EfficientNet-B3
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,          # Remove classifier head
            global_pool='avg',      # Global average pooling
        )
        feature_dim = self.backbone.num_features  # 1536 for B3

        # Crop identification head
        self.crop_head = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_crops),
        )

        # Quality grading head
        self.grade_head = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_grades),
        )

    def forward(self, x):
        features = self.backbone(x)
        crop_logits = self.crop_head(features)
        grade_logits = self.grade_head(features)
        return crop_logits, grade_logits

    def freeze_backbone(self):
        """Freeze backbone for warm-up phase."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone_top(self, fraction: float = 0.3):
        """Unfreeze top `fraction` of backbone layers."""
        layers = list(self.backbone.children())
        unfreeze_from = int(len(layers) * (1 - fraction))
        for layer in layers[unfreeze_from:]:
            for param in layer.parameters():
                param.requires_grad = True
```

#### `src/training/losses.py`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiTaskLoss(nn.Module):
    """
    Weighted combination of crop identification loss and quality grading loss.
    Includes label smoothing to prevent overconfident predictions.
    """

    def __init__(
        self,
        crop_weight: float = 0.4,
        grade_weight: float = 0.6,
        label_smoothing: float = 0.1,
        grade_class_weights=None,  # torch.Tensor of shape [4]
    ):
        super().__init__()
        self.crop_weight = crop_weight
        self.grade_weight = grade_weight

        self.crop_loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.grade_loss_fn = nn.CrossEntropyLoss(
            weight=grade_class_weights,
            label_smoothing=label_smoothing,
        )

    def forward(self, crop_logits, grade_logits, crop_labels, grade_labels):
        crop_loss = self.crop_loss_fn(crop_logits, crop_labels)
        grade_loss = self.grade_loss_fn(grade_logits, grade_labels)
        total = (self.crop_weight * crop_loss) + (self.grade_weight * grade_loss)
        return total, crop_loss, grade_loss
```

#### `src/training/trainer.py`

```python
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm

class Trainer:
    def __init__(self, model, train_loader, val_loader, loss_fn, config, device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.config = config
        self.device = device
        self.writer = SummaryWriter(log_dir=config['log_dir'])
        self.best_val_acc = 0.0

        self.optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config['lr'],
            weight_decay=config['weight_decay'],
        )

        # Warmup for 5 epochs, then cosine anneal
        warmup = LinearLR(self.optimizer, start_factor=0.1, end_factor=1.0, total_iters=5)
        cosine = CosineAnnealingLR(self.optimizer, T_max=config['epochs'] - 5)
        self.scheduler = SequentialLR(self.optimizer, [warmup, cosine], milestones=[5])

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = crop_correct = grade_correct = total = 0

        # Phase: Unfreeze backbone top 30% after epoch 3
        if epoch == 3:
            self.model.unfreeze_backbone_top(fraction=0.3)
            print("Unfroze top 30% of backbone.")

        for images, crop_labels, grade_labels in tqdm(self.train_loader, desc=f"Epoch {epoch}"):
            images = images.to(self.device)
            crop_labels = crop_labels.to(self.device)
            grade_labels = grade_labels.to(self.device)

            self.optimizer.zero_grad()
            crop_logits, grade_logits = self.model(images)
            loss, _, _ = self.loss_fn(crop_logits, grade_logits, crop_labels, grade_labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            crop_correct += (crop_logits.argmax(1) == crop_labels).sum().item()
            grade_correct += (grade_logits.argmax(1) == grade_labels).sum().item()
            total += images.size(0)

        self.scheduler.step()
        avg_loss = total_loss / len(self.train_loader)
        crop_acc = crop_correct / total
        grade_acc = grade_correct / total

        self.writer.add_scalar('Train/Loss', avg_loss, epoch)
        self.writer.add_scalar('Train/CropAcc', crop_acc, epoch)
        self.writer.add_scalar('Train/GradeAcc', grade_acc, epoch)

        return avg_loss, crop_acc, grade_acc

    def validate(self, epoch):
        self.model.eval()
        grade_correct = crop_correct = total = 0

        with torch.no_grad():
            for images, crop_labels, grade_labels in self.val_loader:
                images = images.to(self.device)
                crop_labels = crop_labels.to(self.device)
                grade_labels = grade_labels.to(self.device)

                crop_logits, grade_logits = self.model(images)
                crop_correct += (crop_logits.argmax(1) == crop_labels).sum().item()
                grade_correct += (grade_logits.argmax(1) == grade_labels).sum().item()
                total += images.size(0)

        crop_acc = crop_correct / total
        grade_acc = grade_correct / total

        self.writer.add_scalar('Val/CropAcc', crop_acc, epoch)
        self.writer.add_scalar('Val/GradeAcc', grade_acc, epoch)

        # Save best model
        if grade_acc > self.best_val_acc:
            self.best_val_acc = grade_acc
            torch.save(self.model.state_dict(), f"{self.config['checkpoint_dir']}/best_model.pth")
            print(f"  ✓ New best model saved: grade_acc={grade_acc:.4f}")

        return crop_acc, grade_acc
```

### 4.4 Hyperparameter Config

#### `configs/default.yaml`

```yaml
model:
  backbone: efficientnet_b3
  pretrained: true
  dropout_rate: 0.3
  num_crops: 6
  num_grades: 4

training:
  epochs: 50
  batch_size: 32
  lr: 0.001
  weight_decay: 0.01
  label_smoothing: 0.1
  crop_loss_weight: 0.4
  grade_loss_weight: 0.6
  confidence_threshold: 0.70   # Below this: route to AgriCLIP fallback

data:
  image_size: 224
  num_workers: 4
  train_split: 0.70
  val_split: 0.15
  test_split: 0.15

paths:
  data_dir: data/processed
  checkpoint_dir: models/checkpoints
  export_dir: models/exports
  log_dir: runs/

inference:
  confidence_threshold: 0.70
  batch_size: 1                # For mobile/edge deployment
```

### 4.5 Inference with Fallback

#### `src/inference/predictor.py`

```python
import torch
import torch.nn.functional as F
from PIL import Image
import open_clip
from src.models.multitask_model import CropGradingModel
from src.data.transforms import val_transforms

GRADE_DESCRIPTIONS = {
    'A': 'Grade A quality: excellent condition, premium market value',
    'B': 'Grade B quality: good condition, standard market value',
    'C': 'Grade C quality: below average, local market only',
    'D': 'Grade D quality: poor condition, significant defects',
}

CROP_NAMES = CropGradingModel.CROP_CLASSES
GRADE_NAMES = CropGradingModel.GRADE_CLASSES


class CropGradePredictor:
    def __init__(self, model_path: str, config: dict, device: str = 'cpu'):
        self.device = torch.device(device)
        self.config = config
        self.threshold = config['inference']['confidence_threshold']

        # Load main model
        self.model = CropGradingModel(
            backbone_name=config['model']['backbone'],
            pretrained=False,
        )
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device).eval()

        # Load AgriCLIP fallback
        self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
            'ViT-B-32',
            pretrained='openai'  # Replace with AgriCLIP weights path if available
        )
        self.clip_model.to(self.device).eval()
        self.tokenizer = open_clip.get_tokenizer('ViT-B-32')

    def predict(self, image_path: str) -> dict:
        image = Image.open(image_path).convert('RGB')
        tensor = val_transforms(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            crop_logits, grade_logits = self.model(tensor)

        crop_probs = F.softmax(crop_logits, dim=1)[0]
        grade_probs = F.softmax(grade_logits, dim=1)[0]

        crop_conf, crop_idx = crop_probs.max(0)
        grade_conf, grade_idx = grade_probs.max(0)

        crop_name = CROP_NAMES[crop_idx.item()]
        grade = GRADE_NAMES[grade_idx.item()]
        overall_conf = (crop_conf.item() * grade_conf.item()) ** 0.5  # geometric mean

        if overall_conf >= self.threshold:
            return {
                'crop': crop_name,
                'grade': grade,
                'confidence': round(overall_conf, 3),
                'crop_probabilities': {c: round(p.item(), 3) for c, p in zip(CROP_NAMES, crop_probs)},
                'grade_probabilities': {g: round(p.item(), 3) for g, p in zip(GRADE_NAMES, grade_probs)},
                'low_confidence': False,
                'method': 'multitask_cnn',
            }
        else:
            # Fallback to AgriCLIP
            return self._clip_fallback(image, crop_name, overall_conf)

    def _clip_fallback(self, image: Image.Image, crop_name: str, conf: float) -> dict:
        clip_image = self.clip_preprocess(image).unsqueeze(0).to(self.device)
        texts = [f"{crop_name} crop, {desc}" for desc in GRADE_DESCRIPTIONS.values()]
        tokens = self.tokenizer(texts).to(self.device)

        with torch.no_grad():
            image_features = self.clip_model.encode_image(clip_image)
            text_features = self.clip_model.encode_text(tokens)
            image_features = F.normalize(image_features, dim=-1)
            text_features = F.normalize(text_features, dim=-1)
            similarities = (image_features @ text_features.T)[0]
            probs = F.softmax(similarities * 100, dim=0)

        grade_idx = probs.argmax().item()
        return {
            'crop': crop_name,
            'grade': GRADE_NAMES[grade_idx],
            'confidence': round(probs[grade_idx].item(), 3),
            'grade_probabilities': {g: round(p.item(), 3) for g, p in zip(GRADE_NAMES, probs)},
            'low_confidence': True,
            'method': 'clip_fallback',
            'warning': 'Low confidence — result may be less accurate for this crop type',
        }
```

### 4.6 Training Script

#### `scripts/train.py`

```python
import yaml
import torch
from torch.utils.data import DataLoader
from src.data.dataset import CropGradeDataset
from src.data.transforms import train_transforms, val_transforms
from src.models.multitask_model import CropGradingModel
from src.training.trainer import Trainer
from src.training.losses import MultiTaskLoss
import numpy as np

def compute_class_weights(dataset):
    """Compute inverse-frequency weights for grade classes."""
    grade_counts = np.bincount([item['grade_label'] for item in dataset.samples])
    weights = 1.0 / grade_counts
    weights = weights / weights.sum() * len(weights)
    return torch.FloatTensor(weights)

def main():
    with open('configs/default.yaml') as f:
        config = yaml.safe_load(f)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training on: {device}")

    # Datasets
    train_dataset = CropGradeDataset(
        data_dir=f"{config['data']['data_dir']}/train",
        labels_csv=f"{config['data']['data_dir']}/../metadata/train_labels.csv",
        transform=train_transforms,
    )
    val_dataset = CropGradeDataset(
        data_dir=f"{config['data']['data_dir']}/val",
        labels_csv=f"{config['data']['data_dir']}/../metadata/val_labels.csv",
        transform=val_transforms,
    )

    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'],
                              shuffle=True, num_workers=config['data']['num_workers'],
                              pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'],
                            shuffle=False, num_workers=config['data']['num_workers'])

    # Model
    model = CropGradingModel(
        backbone_name=config['model']['backbone'],
        num_crops=config['model']['num_crops'],
        num_grades=config['model']['num_grades'],
        dropout_rate=config['model']['dropout_rate'],
        pretrained=config['model']['pretrained'],
    )
    model.freeze_backbone()  # Start with frozen backbone

    # Loss
    grade_weights = compute_class_weights(train_dataset).to(device)
    loss_fn = MultiTaskLoss(
        crop_weight=config['training']['crop_loss_weight'],
        grade_weight=config['training']['grade_loss_weight'],
        grade_class_weights=grade_weights,
    )

    # Train
    trainer = Trainer(model, train_loader, val_loader, loss_fn, config['training'], device)

    print(f"Training for {config['training']['epochs']} epochs...")
    for epoch in range(1, config['training']['epochs'] + 1):
        train_loss, train_crop_acc, train_grade_acc = trainer.train_epoch(epoch)
        val_crop_acc, val_grade_acc = trainer.validate(epoch)

        print(
            f"Epoch {epoch:3d} | Loss: {train_loss:.4f} | "
            f"Crop: {train_crop_acc:.3f}/{val_crop_acc:.3f} | "
            f"Grade: {train_grade_acc:.3f}/{val_grade_acc:.3f}"
        )

if __name__ == '__main__':
    main()
```

---

## 5. Phase 3 — Evaluation & Tuning (Weeks 11–12)

### 5.1 Evaluation Metrics

For each crop independently, and also overall:

| Metric | What it measures | Target |
|--------|-----------------|--------|
| Top-1 Accuracy | Correct A/B/C/D grade | ≥85% for 6 core crops |
| Macro F1-Score | Per-class F1 averaged equally | ≥0.82 |
| Adjacent Accuracy | Grade within ±1 step (e.g. A predicted as B) | ≥95% |
| Confusion Matrix | Where model confuses grades | Visualize |
| Calibration Error | Is confidence score trustworthy? | ECE < 0.05 |

**Adjacent accuracy is critical for MVP** — a farmer should never get A when the answer is D, or vice versa. Being off by one grade is tolerable; being off by two is a market trust issue.

### 5.2 Evaluation Script

```python
# scripts/evaluate.py
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model(predictor, test_loader, device):
    all_crop_preds, all_crop_labels = [], []
    all_grade_preds, all_grade_labels = [], []

    model.eval()
    with torch.no_grad():
        for images, crop_labels, grade_labels in test_loader:
            images = images.to(device)
            crop_logits, grade_logits = model(images)
            all_crop_preds.extend(crop_logits.argmax(1).cpu().numpy())
            all_crop_labels.extend(crop_labels.numpy())
            all_grade_preds.extend(grade_logits.argmax(1).cpu().numpy())
            all_grade_labels.extend(grade_labels.numpy())

    print("=== Crop Identification ===")
    print(classification_report(all_crop_labels, all_crop_preds,
                                 target_names=CropGradingModel.CROP_CLASSES))

    print("=== Quality Grading ===")
    print(classification_report(all_grade_labels, all_grade_preds,
                                 target_names=CropGradingModel.GRADE_CLASSES))

    # Adjacent accuracy
    adjacent = sum(abs(p - l) <= 1
                   for p, l in zip(all_grade_preds, all_grade_labels))
    print(f"Adjacent accuracy: {adjacent / len(all_grade_labels):.4f}")

    # Confusion matrix
    cm = confusion_matrix(all_grade_labels, all_grade_preds)
    sns.heatmap(cm, annot=True, fmt='d',
                xticklabels=['A', 'B', 'C', 'D'],
                yticklabels=['A', 'B', 'C', 'D'])
    plt.title('Grade Confusion Matrix')
    plt.savefig('outputs/confusion_matrix.png', dpi=150, bbox_inches='tight')
```

### 5.3 Error Analysis Protocol

After evaluation, run these analyses:

1. **Per-crop breakdown** — which crops have worst accuracy? (Cotton/sugarcane expected)
2. **Grade confusion patterns** — does the model confuse B↔C more than A↔D? (Acceptable)
3. **Confidence calibration** — when model says 90% confidence, is it right 90% of the time?
4. **Hard negatives** — manually review 50 worst predictions, look for:
   - Labeling errors in dataset (fix labels)
   - Systematic visual confusions (adjust augmentation)
   - Lighting/angle artifacts (collect more diverse images)
5. **Grad-CAM visualization** — verify model is looking at crop, not background

```python
# Grad-CAM: visualize what the model looks at
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

cam = GradCAM(model=model, target_layers=[model.backbone.blocks[-1]])
grayscale_cam = cam(input_tensor=image_tensor)
visualization = show_cam_on_image(original_image, grayscale_cam[0])
```

### 5.4 If Accuracy Is Below Target

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| Cotton/sugarcane < 70% | Too few training samples | Collect 500 more images each |
| Grade B/C confusion | Similar visual appearance | Add sharper augmentations, review labels |
| High train acc, low val acc | Overfitting | Increase dropout to 0.4, add weight decay |
| Low confidence on valid images | Overconfident threshold | Lower threshold from 0.70 to 0.60 |
| Model ignores crop features | Backbone not fine-tuned enough | Unfreeze more layers, train for 10 more epochs |

---

## 6. Phase 4 — Deployment (Weeks 13–16)

### 6.1 Model Export

```python
# scripts/export_onnx.py
import torch
import torch.onnx
from src.models.multitask_model import CropGradingModel

model = CropGradingModel(pretrained=False)
model.load_state_dict(torch.load('models/checkpoints/best_model.pth'))
model.eval()

dummy_input = torch.randn(1, 3, 224, 224)

torch.onnx.export(
    model,
    dummy_input,
    'models/exports/crop_grader.onnx',
    input_names=['image'],
    output_names=['crop_logits', 'grade_logits'],
    dynamic_axes={'image': {0: 'batch_size'}},
    opset_version=17,
)

print("ONNX export done.")
# Verify: python -c "import onnx; onnx.checker.check_model('models/exports/crop_grader.onnx')"
```

For mobile (TFLite):
```python
# Convert ONNX → TFLite via tf2onnx
# pip install tf2onnx tensorflow
import subprocess
subprocess.run([
    'python', '-m', 'tf2onnx.convert',
    '--onnx', 'models/exports/crop_grader.onnx',
    '--output', 'models/exports/crop_grader.tflite',
    '--tflite',
])
```

### 6.2 REST API

#### `api/main.py`

```python
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import tempfile, os
from src.inference.predictor import CropGradePredictor
import yaml

app = FastAPI(title="Crop Quality Grading API", version="1.0.0")

with open('configs/default.yaml') as f:
    config = yaml.safe_load(f)

predictor = CropGradePredictor(
    model_path='models/checkpoints/best_model.pth',
    config=config,
    device='cpu',
)

@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": True}

@app.post("/grade")
async def grade_crop(file: UploadFile = File(...)):
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image.")

    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        result = predictor.predict(tmp_path)
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.unlink(tmp_path)
```

Run with: `uvicorn api.main:app --host 0.0.0.0 --port 8000`

### 6.3 API Response Format

```json
{
  "crop": "mango",
  "grade": "B",
  "confidence": 0.847,
  "crop_probabilities": {
    "wheat": 0.012,
    "rice": 0.008,
    "mango": 0.931,
    "maize": 0.024,
    "cotton": 0.015,
    "sugarcane": 0.010
  },
  "grade_probabilities": {
    "A": 0.093,
    "B": 0.847,
    "C": 0.052,
    "D": 0.008
  },
  "low_confidence": false,
  "method": "multitask_cnn"
}
```

If confidence is low (AgriCLIP fallback was used):
```json
{
  "crop": "unknown_crop",
  "grade": "C",
  "confidence": 0.541,
  "low_confidence": true,
  "method": "clip_fallback",
  "warning": "Low confidence — result may be less accurate for this crop type"
}
```

### 6.4 Mobile Integration (Flutter)

```dart
// Quick reference for Flutter integration
Future<GradeResult> gradeImage(File imageFile) async {
  final uri = Uri.parse('http://YOUR_SERVER_IP:8000/grade');
  final request = http.MultipartRequest('POST', uri);
  request.files.add(await http.MultipartFile.fromPath('file', imageFile.path));

  final response = await request.send();
  final body = await response.stream.bytesToString();
  final json = jsonDecode(body);

  return GradeResult(
    crop: json['crop'],
    grade: json['grade'],
    confidence: json['confidence'],
    isLowConfidence: json['low_confidence'],
    warning: json['warning'],
  );
}
```

### 6.5 Farmer-Facing UI Guidelines

The output screen should show:

- Large **grade letter** (A/B/C/D) in color (green=A, yellow=B, orange=C, red=D)
- Crop name (in English + Urdu)
- Confidence bar
- One-line market advice ("Suitable for export" / "Local market only" / "Sell soon")
- If `low_confidence=true`: show a warning icon + "Result may vary for this crop"

---

## 7. Project Structure

See Section 4.2 above for the complete directory layout.

Key principles:
- `data/raw/` is read-only after download — never modify source files
- `data/processed/` is the working directory — all cleaning applied here
- All hyperparameters live in `configs/default.yaml` — no magic numbers in code
- `models/checkpoints/best_model.pth` is always the production model
- Every experiment logs to `runs/` for TensorBoard comparison

---

## 8. Tech Stack

| Component | Technology | Reason |
|-----------|-----------|--------|
| Model training | PyTorch 2.2 + timm | Best ecosystem for CV research |
| Backbone pretrained weights | AgriCLIP / ImageNet EfficientNet-B3 | Domain-adapted features |
| CLIP fallback | OpenCLIP (ViT-B-32) | Open source, AgriCLIP compatible |
| Data augmentation | albumentations | Faster than torchvision for complex augments |
| Experiment tracking | TensorBoard / W&B | Free, works offline too |
| Labeling tool | Label Studio | Self-hosted, supports image classification |
| Model export | ONNX + TFLite | Cross-platform inference |
| API | FastAPI | Async, auto-docs, fast |
| Mobile | Flutter | Cross-platform, existing project context |
| Training compute | Google Colab Pro / Kaggle (free GPU) | No cost for MVP |

---

## 9. Grading Standards Reference

Pakistani crop grades broadly align with these official standards:

| Body | Standard | Applicable Crops |
|------|----------|-----------------|
| PSQCA | PS:1583 | Wheat, Rice |
| TDAP | Export Grade Standards | Mango, Cotton |
| RECP | Rice Export Corp Pakistan | Rice export grades |
| Provincial Agriculture Dept | Local market grades | Sugarcane, Maize |

When in doubt, use PSQCA standards as the baseline and adjust with local expert input.

---

## 10. Risk Register

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|-----------|
| Cotton/sugarcane dataset too small | High | High | Start collection in Week 1, use aggressive augmentation |
| Model confuses B and C grades | Medium | Medium | Add more B/C boundary images, use adjacent accuracy metric |
| Farmers photograph in poor lighting | High | Medium | Include dark/blurry images in augmentation pipeline |
| AgriCLIP weights unavailable | Low | Medium | Fall back to standard CLIP; still functional |
| Labeling inconsistency between annotators | Medium | High | Use 2-annotator protocol with expert tie-breaker from Week 2 |
| GPU quota exceeded on Colab | Medium | Low | Move to Kaggle (30h/week free GPU), or use local machine |
| Model overfits cotton/sugarcane | Medium | Medium | Early stopping, cross-validation for small datasets |

---

## 11. Milestones Checklist

### Phase 1 — Data (Weeks 1–4)
- [ ] Download rice, wheat, mango, maize datasets
- [ ] Set up Label Studio for cotton/sugarcane labeling
- [ ] Complete cotton data collection (500+ images)
- [ ] Complete sugarcane data collection (400+ images)
- [ ] Finalize grading labels with agricultural expert
- [ ] Build train/val/test splits with stratification
- [ ] Verify dataset class balance, generate distribution plots

### Phase 2 — Model (Weeks 5–10)
- [ ] Implement `CropGradingModel` with EfficientNet-B3 backbone
- [ ] Implement `MultiTaskLoss` with crop + grade loss
- [ ] Implement `Trainer` with warmup + cosine schedule
- [ ] Run baseline training run (3–5 epochs) to verify pipeline
- [ ] Full training run (50 epochs) on all 6 crops
- [ ] Implement `CropGradePredictor` with AgriCLIP fallback
- [ ] Test end-to-end inference with 10 sample images

### Phase 3 — Evaluation (Weeks 11–12)
- [ ] Run full test set evaluation, generate classification report
- [ ] Generate per-crop accuracy breakdown
- [ ] Generate confusion matrix for grades
- [ ] Compute adjacent accuracy metric
- [ ] Run Grad-CAM on 20 images, verify model attention
- [ ] Fix top 3 error patterns found in analysis
- [ ] Re-evaluate — confirm ≥85% accuracy on 6 core crops

### Phase 4 — Deployment (Weeks 13–16)
- [ ] Export model to ONNX
- [ ] Build FastAPI endpoint, test locally
- [ ] Deploy to server (or cloud: Railway / Render free tier)
- [ ] Integrate API into Flutter mobile app
- [ ] Field test with 3–5 farmers, collect feedback
- [ ] Fix UX issues from field test
- [ ] Document API with examples
- [ ] Write project README

---

*Last updated: Implementation v1.0*  
*Designed for: Pakistani MVP — Wheat, Rice, Mango, Maize, Cotton, Sugarcane*
