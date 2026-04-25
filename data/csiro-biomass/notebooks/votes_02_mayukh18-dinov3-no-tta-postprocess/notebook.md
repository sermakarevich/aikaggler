# DinoV3 no TTA | PostProcess

- **Author:** Mayukh Bhattacharyya
- **Votes:** 541
- **Ref:** mayukh18/dinov3-no-tta-postprocess
- **URL:** https://www.kaggle.com/code/mayukh18/dinov3-no-tta-postprocess
- **Last run:** 2026-01-18 05:24:44.830000

---

# CSIRO Biomass Prediction - Inference Notebook

This notebook performs inference using trained models, loading one model at a time to conserve memory.

```python
!pip uninstall -y timm
!pip install -q --no-deps /kaggle/input/wheels-csiro/timm-1.0.22-py3-none-any.whl
```

```python
import os
import random
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import timm
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder, StandardScaler
import gc

SEED = 42
def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

seed_everything()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Current Device:", device)
```

```python
class CFG:
    TARGETS = ["Dry_Green_g", "Dry_Dead_g", "Dry_Clover_g", "GDM_g", "Dry_Total_g"]
    MAX_SPECIES_LEN = 5
    DATA_DIR = Path("/kaggle/input/csiro-biomass")
    MODEL_DIR = Path("/kaggle/input/biomass-model-dinov3")
    IMG_SIZE = 512
    N_FOLDS = 4
    BACKBONE = "vit_huge_plus_patch16_dinov3.lvd1689m"
    BATCH_SIZE = 2  # Can increase if memory allows
    
    # TTA configuration
    USE_TTA = False
```

```python
# Load data
train_df = pd.read_csv(CFG.DATA_DIR / "train.csv")
test_df = pd.read_csv(CFG.DATA_DIR / "test.csv")

print(f"Train rows: {len(train_df)}")
print(f"Test rows: {len(test_df)}")
print("\nTest data preview:")
test_df.head()
```

## Prepare Test Data with Metadata

```python
test_wide = test_df[["image_path"]].drop_duplicates().reset_index(drop=True)

print(f"Test data prepared: {len(test_wide)} samples")
test_wide.head()
```

## Define Model Architecture

```python
class LocalMambaBlock(nn.Module):
    """Lightweight Mamba-like block"""

    def __init__(self, dim: int, kernel_size: int = 5, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=kernel_size,
                                 padding=kernel_size // 2, groups=dim)
        self.gate = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.norm(x)
        g = torch.sigmoid(self.gate(x))
        x = x * g
        x = x.transpose(1, 2)
        x = self.dwconv(x)
        x = x.transpose(1, 2)
        x = self.proj(x)
        x = self.drop(x)
        return shortcut + x


class BiomassModel(nn.Module):
    """DINOv3 + Mamba Fusion + Multi-Head Regression"""

    def __init__(self, model_name: str, pretrained: bool = True,
                 backbone_path: Optional[Path] = None):
        super().__init__()
        self.model_name = model_name
        self.backbone_path = backbone_path

        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0, global_pool=''
        )
        nf = self.backbone.num_features

        self.fusion = nn.Sequential(
            LocalMambaBlock(nf, kernel_size=5, dropout=0.1),
            LocalMambaBlock(nf, kernel_size=5, dropout=0.1)
        )
        self.pool = nn.AdaptiveAvgPool1d(1)

        self.head_green = nn.Sequential(
            nn.Linear(nf, nf // 2), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(nf // 2, 1), nn.Softplus()
        )
        self.head_dead = nn.Sequential(
            nn.Linear(nf, nf // 2), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(nf // 2, 1), nn.Softplus()
        )
        self.head_clover = nn.Sequential(
            nn.Linear(nf, nf // 2), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(nf // 2, 1), nn.Softplus()
        )

    def forward(self, x):
        # x is a tuple (left, right)
        if isinstance(x, tuple):
            left, right = x
        else:
            raise ValueError("Input must be a tuple of (left, right)")

        x_l = self.backbone(left)
        x_r = self.backbone(right)
        x_cat = torch.cat([x_l, x_r], dim=1)
        x_fused = self.fusion(x_cat)
        x_pool = self.pool(x_fused.transpose(1, 2)).flatten(1)

        green = self.head_green(x_pool)
        dead = self.head_dead(x_pool)
        clover = self.head_clover(x_pool)
        gdm = green + clover
        total = gdm + dead

        # Return as a single tensor (batch, 5)
        return torch.cat([green, dead, clover, gdm, total], dim=1)
```

## Define Test Dataset and DataLoader

```python
class PastureImageTestDataset(Dataset):
    """Dataset for test images with metadata"""
    def __init__(self, df, image_root, transform=None):
        self.df = df.reset_index(drop=True)
        self.image_root = image_root
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load image
        img_path = self.image_root / row["image_path"]
        img = Image.open(img_path).convert("RGB")
        w, h = img.size  # expect (2000, 1000)

        # Split in the middle (deterministic for test)
        left = img.crop((0, 0, h, h))        # (0, 0, 1000, 1000)
        right = img.crop((w - h, 0, w, h))   # (1000, 0, 2000, 1000)

        if self.transform:
            img1 = self.transform(left)
            img2 = self.transform(right)

        # Return images and row info
        return (img1, img2), row.to_dict()


# Define transforms
test_tfms = T.Compose([
    T.Resize((CFG.IMG_SIZE, CFG.IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def get_tta_transforms():
    """Returns a list of transform pipelines for TTA during inference."""
    base_transforms = [
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]

    # View 1: Original
    original_view = T.Compose([
        T.Resize((CFG.IMG_SIZE, CFG.IMG_SIZE)),
        T.ToTensor(),
        *base_transforms
    ])

    # View 2: Horizontal Flip
    hflip_view = T.Compose([
        T.Resize((CFG.IMG_SIZE, CFG.IMG_SIZE)),
        T.RandomHorizontalFlip(p=1.0),
        T.ToTensor(),
        *base_transforms
    ])

    # View 3: Vertical Flip
    vflip_view = T.Compose([
        T.Resize((CFG.IMG_SIZE, CFG.IMG_SIZE)),
        T.RandomVerticalFlip(p=1.0),
        T.ToTensor(),
        *base_transforms
    ])

    # View 4: Rotate90
    vflip_view = T.Compose([
        T.Resize((CFG.IMG_SIZE, CFG.IMG_SIZE)),
        T.Lambda(lambda img: T.functional.rotate(img, angle=90)),
        T.ToTensor(),
        *base_transforms
    ])

    return [original_view, hflip_view, vflip_view]


def collate_fn_test(batch):
    """Custom collate function for test data"""
    imgs1, imgs2 = [], []
    row_infos = []

    for (img1, img2), row_info in batch:
        imgs1.append(img1)
        imgs2.append(img2)
        row_infos.append(row_info)

    imgs1 = torch.stack(imgs1)
    imgs2 = torch.stack(imgs2)

    return (imgs1, imgs2), row_infos


print("Test dataset and transforms defined")
```

## Inference Function

```python
def predict_with_model(model, loader, use_tta=False):
    """
    Make predictions with a single model.
    
    Args:
        model: Trained model
        test_loader: DataLoader for test data
        use_tta: Whether to use test-time augmentation
    
    Returns:
        predictions: numpy array of shape (n_samples, 5) with all 5 targets
    """
    model.eval()
    # Standard inference without TTA
    
    preds_all = []
    with torch.no_grad():
        for (imgs1, imgs2), _ in tqdm(test_loader, desc="Inference"):
            imgs1, imgs2 = imgs1.to(device), imgs2.to(device)
            with torch.amp.autocast('cuda'):
                pred = model((imgs1, imgs2))

            preds_all.append(pred.cpu().numpy())
    
    return np.vstack(preds_all)



print("Inference function defined")
```

## Run Inference with Model Ensemble

Load one model at a time, make predictions, and ensemble the results.

```python
# Create test dataset and dataloader
test_dataset = PastureImageTestDataset(
    test_wide,
    CFG.DATA_DIR,
    test_tfms
)

test_loader = DataLoader(
    test_dataset,
    batch_size=CFG.BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
    collate_fn=collate_fn_test
)

print(f"Test loader created with {len(test_dataset)} samples")

# Storage for ensemble predictions
all_fold_predictions = []

# Loop through each fold
for fold in range(0, CFG.N_FOLDS):
    print(f"\n{'='*60}")
    print(f"Processing Fold {fold}")
    print(f"{'='*60}")

    model_file = (
        "biomass_model_easy_fold1_0.6228.pth" if fold == 1 else
        "biomass_model_easy_fold3.pth" if fold == 3 else
        "x"
    )
    model_path = CFG.MODEL_DIR / model_file
    if not os.path.exists(model_path):
        continue
    print(f"Loading model from {model_path}...")
    
    model = BiomassModel(
        model_name=CFG.BACKBONE,
        pretrained=False
    )
    
    # Load weights
    state_dict = torch.load(model_path, map_location='cpu')
    
    # Handle DataParallel state dict
    if list(state_dict.keys())[0].startswith('module.'):
        # Remove 'module.' prefix
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict)
    model = model.to(device)
    
    # Make predictions
    fold_predictions = predict_with_model(
        model, test_loader, use_tta=CFG.USE_TTA
    )
    weight = (
        0.75 if fold == 1 else
        1.25 if fold == 3 else
        1.0
    )
    all_fold_predictions.append(weight * fold_predictions)
    
    print(f"\nFold {fold} predictions shape: {fold_predictions.shape}")
    print(f"Predictions stats: min={fold_predictions.min():.4f}, max={fold_predictions.max():.4f}, mean={fold_predictions.mean():.4f}")
    
    # Clean up to free memory
    del model
    del state_dict
    torch.cuda.empty_cache()
    gc.collect()
    print(f"\nMemory cleaned after fold {fold}")

print(f"\n{'='*60}")
print(f"All folds processed: {len(all_fold_predictions)} models")
print(f"{'='*60}")
```

## Ensemble Predictions

```python
# Average predictions across all folds
if len(all_fold_predictions) == 0:
    raise ValueError("No model predictions were generated. Check if model files exist.")

ensemble_predictions = np.mean(all_fold_predictions, axis=0)

print(f"Ensemble predictions shape: {ensemble_predictions.shape}")
print(f"Ensemble stats: min={ensemble_predictions.min():.4f}, max={ensemble_predictions.max():.4f}, mean={ensemble_predictions.mean():.4f}")

# Show predictions for first sample
print("\nFirst sample predictions:")
for i, target in enumerate(CFG.TARGETS):
    print(f"  {target}: {ensemble_predictions[0, i]:.4f}")
```

## Create Submission File

```python
# Create submission dataframe
# Each test image should have 5 rows (one for each target)

submission_rows = []

for idx, row in test_wide.iterrows():
    image_id = row['image_path'].split('/')[-1].replace('.jpg', '')
    
    for target_idx, target_name in enumerate(CFG.TARGETS):
        sample_id = f"{image_id}__{target_name}"
        prediction = ensemble_predictions[idx, target_idx]
        if target_name == "Dry_Clover_g":
            prediction = prediction * 0.8
        elif target_name == "Dry_Dead_g":
            if prediction > 20:
                prediction *= 1.1
            elif prediction < 10:
                prediction *= 0.9
        
        submission_rows.append({
            'sample_id': sample_id,
            'target': prediction
        })

submission_df = pd.DataFrame(submission_rows)
submission_path = "submission.csv"
submission_df.to_csv(submission_path, index=False)
print(f"\nSubmission saved to {submission_path}")
```

```python
# Display statistics
print("\nSubmission statistics by target:")
for target in CFG.TARGETS:
    target_rows = submission_df[submission_df['sample_id'].str.contains(target)]
    print(f"  {target}:")
    print(f"    Count: {len(target_rows)}")
    print(f"    Min: {target_rows['target'].min():.4f}")
    print(f"    Max: {target_rows['target'].max():.4f}")
    print(f"    Mean: {target_rows['target'].mean():.4f}")
    print(f"    Median: {target_rows['target'].median():.4f}")

submission_df.head(10)
```