# CISRO|baseline train+infer 21/12 dinov3+siglip 

- **Author:** Kh0a
- **Votes:** 452
- **Ref:** llkh0a/cisro-baseline-train-infer-21-12-dinov3-siglip
- **URL:** https://www.kaggle.com/code/llkh0a/cisro-baseline-train-infer-21-12-dinov3-siglip
- **Last run:** 2026-01-05 05:06:24.410000

---

```python
Weight = [0.5, 0.5]
```

# Main pipeline tuning dinoV3_840m

## Config

```python
#show timm version
import timm
print(timm.__version__)
```

```python
# Config
import os, gc, math, cv2, numpy as np, pandas as pd
from tqdm import tqdm
import torch, torch.nn as nn, torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import timm
from timm.utils import ModelEmaV2
from sklearn.model_selection import KFold, StratifiedGroupKFold
import matplotlib.pyplot as plt
import seaborn as sns

class CFG:
    CREATE_SUBMISSION = True
    USE_TQDM        = False
    PRETRAINED_DIR  = None
    PRETRAINED      = True
    BASE_PATH       = '/kaggle/input/csiro-biomass'
    SEED            = 82947501
    FOLDS_TO_TRAIN   = [0,1,2,3,4]
    TRAIN_CSV       = os.path.join(BASE_PATH, 'train.csv')
    TRAIN_IMAGE_DIR = os.path.join(BASE_PATH, 'train')
    TEST_IMAGE_DIR = '/kaggle/input/csiro-biomass/test'
    TEST_CSV = '/kaggle/input/csiro-biomass/test.csv'
    SUBMISSION_DIR  = '/kaggle/working/'
    MODEL_DIR_012       = '/kaggle/input/5-folds-dinov3-840m/other/fold-0-1-2/1'
    MODEL_DIR_34       = '/kaggle/input/5-folds-dinov3-840m/other/fold-0-1-2/2'
    N_FOLDS         = 5


    # MODEL_NAME      = 'vit_large_patch16_dinov3.lvd1689m'  
    # BACKBONE_PATH   = '/kaggle/input/vit-large-patch16-dinov3-lvd1689m-backbone-pth/vit_large_patch16_dinov3.lvd1689m_backbone.pth'
    MODEL_NAME      = 'vit_huge_plus_patch16_dinov3.lvd1689m'
    BACKBONE_PATH   = '/kaggle/input/vit-huge-plus-patch16-dinov3-lvd1689m/vit_huge_plus_patch16_dinov3.lvd1689m_backbone.pth'
    
    IMG_SIZE        = 512

    VAL_TTA_TIMES   = 1
    TTA_STEPS       = 1
    
    
    BATCH_SIZE      = 1
    GRAD_ACC        = 4
    NUM_WORKERS     = 4
    EPOCHS          = 1
    FREEZE_EPOCHS   = 0
    WARMUP_EPOCHS   = 3
    LR_REST         = 1e-3
    LR_BACKBONE     = 5e-4
    WD              = 1e-2
    EMA_DECAY       = 0.9
    PATIENCE        = 5
    TARGET_COLS     = ['Dry_Total_g', 'GDM_g', 'Dry_Green_g']
    DERIVED_COLS    = ['Dry_Clover_g', 'Dry_Dead_g']
    ALL_TARGET_COLS = ['Dry_Green_g','Dry_Dead_g','Dry_Clover_g','GDM_g','Dry_Total_g']
    R2_WEIGHTS      = np.array([0.1, 0.1, 0.1, 0.2, 0.5])
    LOSS_WEIGHTS    = np.array([0.1, 0.1, 0.1, 0.0, 0.0])
    DEVICE          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f'Device : {CFG.DEVICE}')
print(f'Backbone: {CFG.MODEL_NAME} | Input: {CFG.IMG_SIZE}')
print(f'Freeze Epochs: {CFG.FREEZE_EPOCHS} | Warmup: {CFG.WARMUP_EPOCHS}')
print(f'EMA Decay: {CFG.EMA_DECAY} | Grad Acc: {CFG.GRAD_ACC}')
```

## Metric

```python
import numpy as np
import os

def weighted_r2_score(y_true: np.ndarray, y_pred: np.ndarray):
    weights = CFG.R2_WEIGHTS
    r2_scores = []
    for i in range(y_true.shape[1]):
        yt = y_true[:, i]; yp = y_pred[:, i]
        ss_res = np.sum((yt - yp) ** 2)
        ss_tot = np.sum((yt - np.mean(yt)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        r2_scores.append(r2)
    r2_scores = np.array(r2_scores)
    weighted = np.sum(r2_scores * weights) / np.sum(weights)
    return weighted, r2_scores

def weighted_r2_score_global(y_true: np.ndarray, y_pred: np.ndarray):
    weights = CFG.R2_WEIGHTS
    flat_true = y_true.reshape(-1)
    flat_pred = y_pred.reshape(-1)
    w = np.tile(weights, y_true.shape[0])
    mean_w = np.sum(w * flat_true) / np.sum(w)
    ss_res = np.sum(w * (flat_true - flat_pred) ** 2)
    ss_tot = np.sum(w * (flat_true - mean_w) ** 2)
    global_r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    avg_r2, per_r2 = weighted_r2_score(y_true, y_pred)
    return global_r2, avg_r2, per_r2

def analyze_errors(val_df, y_true, y_pred, targets, top_n=5):
    print(f'\n--- Top {top_n} High Loss Samples per Target ---')
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    for i, target in enumerate(targets):
        errors = np.abs(y_true[:, i] - y_pred[:, i])
        top_indices = np.argsort(errors)[::-1][:top_n]
        
        print(f'\nTarget: {target}')
        print(f'{"Index":<6} | {"Image Path":<40} | {"True":<10} | {"Pred":<10} | {"AbsErr":<10}')
        print('-' * 90)
        
        for idx in top_indices:
            path = val_df.iloc[idx]['image_path']
            path_disp = os.path.basename(path)
            t_val = y_true[idx, i]
            p_val = y_pred[idx, i]
            err = errors[idx]
            print(f'{idx:<6} | {path_disp:<40} | {t_val:<10.4f} | {p_val:<10.4f} | {err:<10.4f}')
def analyze_errors(val_df, y_true, y_pred, targets, top_n=5):
    print(f'\n--- Top {top_n} High Loss Samples per Target ---')
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    for i, target in enumerate(targets):
        errors = np.abs(y_true[:, i] - y_pred[:, i])
        top_indices = np.argsort(errors)[::-1][:top_n]
        
        print(f'\nTarget: {target}')
        header = f'{"Index":<6} | {"Image Path":<40} | {"State":<6} | {"True":<10} | {"Pred":<10} | {"AbsErr":<10}'
        print(header)
        print('-' * len(header))
        
        for idx in top_indices:
            path = val_df.iloc[idx]['image_path']
            path_disp = os.path.basename(path)
            state = val_df.iloc[idx]['State'] if 'State' in val_df.columns else 'NA'
            t_val = y_true[idx, i]
            p_val = y_pred[idx, i]
            err = errors[idx]
            print(f'{idx:<6} | {path_disp:<40} | {str(state):<6} | {t_val:<10.4f} | {p_val:<10.4f} | {err:<10.4f}')
def compare_train_val(tr_df, val_df, targets, show_plots=True):
    """Quick comparison of target distributions and metadata between train and val splits."""
    print("\n--- Train / Val Comparison ---")

    for t in targets:
        tr = tr_df.get(t, pd.Series(dtype=float)).dropna()
        val = val_df.get(t, pd.Series(dtype=float)).dropna()
        print(f"\nTarget: {t}")
        print(f"  Train: n={len(tr)} mean={tr.mean():.3f} std={tr.std():.3f} min={tr.min():.3f} max={tr.max():.3f}")
        print(f"  Val  : n={len(val)} mean={val.mean():.3f} std={val.std():.3f} min={val.min():.3f} max={val.max():.3f}")
        if show_plots:
            try:
                plt.figure(figsize=(6, 3))
                sns.kdeplot(tr, label='train', fill=True)
                sns.kdeplot(val, label='val', fill=True)
                plt.legend()
                plt.title(f'Distribution: {t}')
                plt.show()
            except Exception as e:
                print('  Could not plot distributions for', t, '-', e)

    # Compare Sampling_Date and State if present
    if 'Sampling_Date' in tr_df.columns:
        try:
            tr_dates = pd.to_datetime(tr_df['Sampling_Date'], errors='coerce')
            val_dates = pd.to_datetime(val_df['Sampling_Date'], errors='coerce')
            print("\nSampling_Date range:")
            print(f"  Train: {tr_dates.min()} -> {tr_dates.max()} (missing {tr_dates.isna().sum()})")
            print(f"  Val  : {val_dates.min()} -> {val_dates.max()} (missing {val_dates.isna().sum()})")
        except Exception as e:
            print('  Could not parse Sampling_Date:', e)
    if 'State' in tr_df.columns:
        print("\nState distribution (train vs val):")
        tr_state = tr_df['State'].value_counts(normalize=True)
        val_state = val_df['State'].value_counts(normalize=True)
        state_df = pd.concat([tr_state, val_state], axis=1, keys=['train', 'val']).fillna(0)

        print(state_df)
```

## Dataset & Augmentation

```python
def get_train_transforms():
    return A.Compose([
        A.Resize(CFG.IMG_SIZE, CFG.IMG_SIZE),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Rotate(limit=(-10, 10), p=0.3, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_REFLECT_101),
        A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ], p=1.0)

def get_val_transforms():
    return A.Compose([
        A.Resize(CFG.IMG_SIZE, CFG.IMG_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ], p=1.0)

def get_tta_transforms(mode=0):
    # mode 0: original
    # mode 1: hflip
    # mode 2: vflip
    # mode 3: rotate90
    transforms_list = [
        A.Resize(CFG.IMG_SIZE, CFG.IMG_SIZE),
    ]
    
    if mode == 1:
        transforms_list.append(A.HorizontalFlip(p=1.0))
    elif mode == 2:
        transforms_list.append(A.VerticalFlip(p=1.0))
    elif mode == 3:
        transforms_list.append(A.RandomRotate90(p=1.0)) # RandomRotate90 with p=1.0 rotates 90, 180, 270 randomly? 
        # Albumentations RandomRotate90 rotates by 90, 180, 270. 
        # Reference uses transforms.RandomRotation([90, 90]) which is exactly 90 degrees.
        # To match exactly 90 degrees in Albumentations, we might need Rotate(limit=(90,90), p=1.0)
        # But RandomRotate90 is standard TTA. Let's use Rotate(limit=(90,90)) to be precise if that's what reference does.
        # Reference: transforms.RandomRotation([90, 90]) -> rotates by exactly 90 degrees.
        transforms_list.append(A.Rotate(limit=(90, 90), p=1.0, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_REFLECT_101))

    transforms_list.extend([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    return A.Compose(transforms_list, p=1.0)
def clean_image(img):
    # 1. Safe Crop (Remove artifacts at the bottom)
    h, w = img.shape[:2]
    # Cut bottom 10% where artifacts often appear
    img = img[0:int(h*0.90), :] 

    # 2. Inpaint Date Stamp (Remove orange text)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    # Define orange color range (adjust as needed)
    lower = np.array([5, 150, 150])
    upper = np.array([25, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)

    # Dilate mask to cover text edges and reduce noise
    mask = cv2.dilate(mask, np.ones((3,3), np.uint8), iterations=2)

    # Inpaint if mask is not empty
    if np.sum(mask) > 0:
        img = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)

    return img
class BiomassDataset(Dataset):
    def __init__(self, df, transform, img_dir):
        self.df = df
        self.transform = transform
        self.img_dir = img_dir
        self.paths = df['image_path'].values
        self.labels = df[CFG.ALL_TARGET_COLS].values.astype(np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path = os.path.join(self.img_dir, os.path.basename(self.paths[idx]))
        img = cv2.imread(path)
        if img is None:
            img = np.zeros((1000, 2000, 3), dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = clean_image(img)
        h, w, _ = img.shape
        mid = w // 2
        left = img[:, :mid]
        right = img[:, mid:]
        left = self.transform(image=left)['image']
        right = self.transform(image=right)['image']
        label = torch.from_numpy(self.labels[idx])
        return left, right, label
```

## Model & Loss

```python
# class BiomassModel(nn.Module):
#     def __init__(self, model_name, pretrained=True):
#         super().__init__()
#         self.model_name = model_name
#         self.backbone = timm.create_model(self.model_name, pretrained=False, num_classes=0, global_pool='avg')
#         nf = self.backbone.num_features
#         comb = nf * 2
#         self.head_green_raw  = nn.Sequential(nn.Linear(comb, comb//2), nn.GELU(), nn.Dropout(0.3), nn.Linear(comb//2, 1))
#         self.head_clover_raw = nn.Sequential(nn.Linear(comb, comb//2), nn.GELU(), nn.Dropout(0.3), nn.Linear(comb//2, 1))
#         self.head_dead_raw   = nn.Sequential(nn.Linear(comb, comb//2), nn.GELU(), nn.Dropout(0.3), nn.Linear(comb//2, 1))
#         if pretrained:
#             self.load_pretrained()
    
#     def load_pretrained(self):
#         try:
#             sd = timm.create_model(self.model_name, pretrained=True, num_classes=0, global_pool='avg').state_dict()
#             self.backbone.load_state_dict(sd, strict=False)
#             print('Pretrained weights loaded.')
#         except Exception as e:
#             print(f'Warning: pretrained load failed: {e}')
    
#     def forward(self, left, right):
#         fl = self.backbone(left)
#         fr = self.backbone(right)
#         x  = torch.cat([fl, fr], dim=1)
#         green  = self.head_green_raw(x)
#         # clover = torch.nn.functional.softplus(self.head_clover_raw(x))
#         # dead   = torch.nn.functional.softplus(self.head_dead_raw(x))
#         clover = self.head_clover_raw(x)
#         dead   = self.head_dead_raw(x)
#         gdm    = green + clover
#         total  = gdm + dead
#         return total, gdm, green, clover, dead
class LocalMambaBlock(nn.Module):
    """
    Lightweight Mamba-style block (Gated CNN) from the reference notebook.
    Efficiently mixes tokens with linear complexity.
    """
    def __init__(self, dim, kernel_size=5, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        # Depthwise conv mixes spatial information locally
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim)
        self.gate = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        # x: (Batch, Tokens, Dim)
        shortcut = x
        x = self.norm(x)
        # Gating mechanism
        g = torch.sigmoid(self.gate(x))
        x = x * g
        # Spatial mixing via 1D Conv (requires transpose)
        x = x.transpose(1, 2)  # -> (B, D, N)
        x = self.dwconv(x)
        x = x.transpose(1, 2)  # -> (B, N, D)
        # Projection
        x = self.proj(x)
        x = self.drop(x)
        return shortcut + x

class BiomassModel(nn.Module):
    def __init__(self, model_name, pretrained=True, backbone_path=None):
        super().__init__()
        self.model_name = model_name
        self.backbone_path = backbone_path
        
        # 1. Load Backbone with global_pool='' to keep patch tokens
        #    (B, 197, 1024) instead of (B, 1024)
        self.backbone = timm.create_model(self.model_name, pretrained=False, num_classes=0, global_pool='')
        
        # 2. Enable Gradient Checkpointing (Crucial for ViT-Large memory!)
        if hasattr(self.backbone, 'set_grad_checkpointing'):
            self.backbone.set_grad_checkpointing(True)
            print("✓ Gradient Checkpointing enabled (saves ~50% VRAM)")
            
        nf = self.backbone.num_features
        
        # 3. Mamba Fusion Neck
        #    Mixes the concatenated tokens [Left, Right]
        self.fusion = nn.Sequential(
            LocalMambaBlock(nf, kernel_size=5, dropout=0.1),
            LocalMambaBlock(nf, kernel_size=5, dropout=0.1)
        )
        
        # 4. Pooling & Heads
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        # Heads (using the same logic as before, but on fused features)
        self.head_green_raw  = nn.Sequential(
            nn.Linear(nf, nf//2), nn.GELU(), nn.Dropout(0.2), 
            nn.Linear(nf//2, 1), nn.Softplus()
        )
        self.head_clover_raw = nn.Sequential(
            nn.Linear(nf, nf//2), nn.GELU(), nn.Dropout(0.2), 
            nn.Linear(nf//2, 1), nn.Softplus()
        )
        self.head_dead_raw   = nn.Sequential(
            nn.Linear(nf, nf//2), nn.GELU(), nn.Dropout(0.2), 
            nn.Linear(nf//2, 1), nn.Softplus()
        )
        
        if pretrained:
            self.load_pretrained()
    
    def load_pretrained(self):
        try:
            # Load weights normally
            if self.backbone_path and os.path.exists(self.backbone_path):
                print(f"Loading backbone weights from local file: {self.backbone_path}")
                sd = torch.load(self.backbone_path, map_location='cpu')
                # Handle common checkpoint wrappers (e.g. if saved with 'model' key)
                if 'model' in sd: sd = sd['model']
                elif 'state_dict' in sd: sd = sd['state_dict']
            else:
                # Original behavior: Download from internet
                print("Downloading backbone weights...")
                sd = timm.create_model(self.model_name, pretrained=True, num_classes=0, global_pool='').state_dict()
            
            # Interpolate pos_embed if needed (for 256x256 vs 224x224)
            if 'pos_embed' in sd and hasattr(self.backbone, 'pos_embed'):
                pe_ck = sd['pos_embed']
                pe_m  = self.backbone.pos_embed
                if pe_ck.shape != pe_m.shape:
                    print(f"Interpolating pos_embed: {pe_ck.shape} -> {pe_m.shape}")
                    # (Simple interpolation logic here or rely on timm's load if strict=False handles it well enough)
                    # For robust interpolation, use the snippet provided in previous turn
            
            self.backbone.load_state_dict(sd, strict=False)
            print('Pretrained weights loaded.')
        except Exception as e:
            print(f'Warning: pretrained load failed: {e}')
    
    def forward(self, left, right):
        # 1. Extract Tokens (B, N, D)
        #    Note: ViT usually returns [CLS, Patch1, Patch2...]
        #    We remove CLS token for spatial mixing, or keep it. Let's keep it.
        x_l = self.backbone(left)
        x_r = self.backbone(right)
        
        # 2. Concatenate Left and Right tokens along sequence dimension
        #    (B, N, D) + (B, N, D) -> (B, 2N, D)
        x_cat = torch.cat([x_l, x_r], dim=1)
        
        # 3. Apply Mamba Fusion
        #    This allows tokens from Left image to interact with tokens from Right image
        x_fused = self.fusion(x_cat)
        
        # 4. Global Pooling
        #    (B, 2N, D) -> (B, D, 2N) -> (B, D, 1) -> (B, D)
        x_pool = self.pool(x_fused.transpose(1, 2)).flatten(1)
        
        # 5. Prediction Heads
        green  = self.head_green_raw(x_pool)
        clover = self.head_clover_raw(x_pool)
        dead   = self.head_dead_raw(x_pool)
        
        # Summation logic
        gdm    = green + clover
        total  = gdm + dead
        
        return total, gdm, green, clover, dead
def biomass_loss(outputs, labels, w=None):
    total, gdm, green, clover, dead = outputs
    mse = nn.MSELoss()
    huber = nn.SmoothL1Loss(beta=5.0) # Huber loss for robust regression (beta=5.0 as recommended)
    
    l_green  = huber(green.squeeze(),  labels[:,0])
    l_dead   = huber(dead.squeeze(), labels[:,1]) # Use Huber loss for Dead
    l_clover = huber(clover.squeeze(), labels[:,2])
    l_gdm    = huber(gdm.squeeze(),    labels[:,3])
    l_total  = huber(total.squeeze(),  labels[:,4])

    # Stack per-target losses in the SAME order as CFG.ALL_TARGET_COLS
    losses = torch.stack([l_green, l_dead, l_clover, l_gdm, l_total])
    # losses = torch.stack([l_green, l_dead, l_clover])

    # Use provided weights, or default to CFG.R2_WEIGHTS
    if w is None:
        return losses.mean()
    w = torch.as_tensor(w, device=losses.device, dtype=losses.dtype)
    w = w / w.sum()
    return (losses * w).sum()
```

## Train Functions with EMA & Gradient Accumulation

```python
from contextlib import nullcontext
scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

@torch.no_grad()
def valid_epoch(eval_model, loader, device):
    eval_model.eval()
    running = 0.0
    preds_total, preds_gdm, preds_green, preds_clover, preds_dead, all_labels = [], [], [], [], [], []
    amp_ctx = (lambda: torch.amp.autocast(device_type='cuda')) if torch.cuda.is_available() else (lambda: nullcontext())
    
    for l, r, lab in loader:
        l, r, lab = l.to(device, non_blocking=True), r.to(device, non_blocking=True), lab.to(device, non_blocking=True)
        with amp_ctx():
            p_total, p_gdm, p_green, p_clover, p_dead = eval_model(l, r)
            loss = biomass_loss((p_total, p_gdm, p_green, p_clover, p_dead), lab, w=CFG.LOSS_WEIGHTS)
        running += loss.item() * l.size(0)
        preds_total.extend(p_total.cpu().numpy().ravel())
        preds_gdm.extend(p_gdm.cpu().numpy().ravel())
        preds_green.extend(p_green.cpu().numpy().ravel())
        preds_clover.extend(p_clover.cpu().numpy().ravel())
        preds_dead.extend(p_dead.cpu().numpy().ravel())
        all_labels.extend(lab.cpu().numpy())
    
    pred_total  = np.array(preds_total)
    pred_gdm    = np.array(preds_gdm)
    pred_green  = np.array(preds_green)
    pred_clover = np.array(preds_clover)
    pred_dead   = np.array(preds_dead)
    true_labels = np.stack(all_labels)
    
    pred_all = np.stack([pred_green, pred_dead, pred_clover, pred_gdm, pred_total], axis=1)
    global_r2, avg_r2, per_r2 = weighted_r2_score_global(true_labels, pred_all)
    return running / len(loader.dataset), global_r2, avg_r2, per_r2, pred_all, true_labels

@torch.no_grad()
def valid_epoch_tta(eval_model, loaders, device):
    eval_model.eval()
    amp_ctx = (lambda: torch.amp.autocast(device_type='cuda')) if torch.cuda.is_available() else (lambda: nullcontext())
    
    # We need to aggregate predictions from all loaders
    # Assuming all loaders have same order and size (which they should if shuffle=False)
    
    all_preds_accum = None
    all_labels = None
    total_loss = 0.0
    
    for loader_idx, loader in enumerate(loaders):
        preds_total, preds_gdm, preds_green, preds_clover, preds_dead = [], [], [], [], []
        current_labels = []
        running_loss = 0.0
        
        for l, r, lab in loader:
            l, r, lab = l.to(device, non_blocking=True), r.to(device, non_blocking=True), lab.to(device, non_blocking=True)
            with amp_ctx():
                p_total, p_gdm, p_green, p_clover, p_dead = eval_model(l, r)
                loss = biomass_loss((p_total, p_gdm, p_green, p_clover, p_dead), lab, w=CFG.LOSS_WEIGHTS)
            
            running_loss += loss.item() * l.size(0)
            
            preds_total.extend(p_total.cpu().numpy().ravel())
            preds_gdm.extend(p_gdm.cpu().numpy().ravel())
            preds_green.extend(p_green.cpu().numpy().ravel())
            preds_clover.extend(p_clover.cpu().numpy().ravel())
            preds_dead.extend(p_dead.cpu().numpy().ravel())
            
            if loader_idx == 0:
                current_labels.extend(lab.cpu().numpy())
        
        total_loss += (running_loss / len(loader.dataset))
        
        # Stack predictions for this loader: (N, 5)
        # Order: Green, Dead, Clover, GDM, Total (matching CFG.ALL_TARGET_COLS order roughly, but let's be precise)
        # CFG.ALL_TARGET_COLS = ['Dry_Green_g','Dry_Dead_g','Dry_Clover_g','GDM_g','Dry_Total_g']
        # preds lists are just raw outputs.
        # Let's stack them in the order expected by weighted_r2_score_global which expects:
        # y_true, y_pred where columns match.
        # The model returns: total, gdm, green, clover, dead
        # We need to stack them to match true_labels which comes from CFG.ALL_TARGET_COLS
        # CFG.ALL_TARGET_COLS is ['Dry_Green_g','Dry_Dead_g','Dry_Clover_g','GDM_g','Dry_Total_g']
        
        pred_stack = np.stack([
            np.array(preds_green),
            np.array(preds_dead),
            np.array(preds_clover),
            np.array(preds_gdm),
            np.array(preds_total)
        ], axis=1)
        
        if all_preds_accum is None:
            all_preds_accum = pred_stack
            all_labels = np.stack(current_labels)
        else:
            all_preds_accum += pred_stack
            
    # Average predictions
    avg_preds = all_preds_accum / len(loaders)
    avg_loss = total_loss / len(loaders)
    
    global_r2, avg_r2, per_r2 = weighted_r2_score_global(all_labels, avg_preds)
    return avg_loss, global_r2, avg_r2, per_r2, avg_preds, all_labels

def set_backbone_requires_grad(model: BiomassModel, requires_grad: bool):
    for p in model.backbone.parameters():
        p.requires_grad = requires_grad

def build_optimizer(model: BiomassModel):
    head_params = (list(model.head_green_raw.parameters()) +
                   list(model.head_clover_raw.parameters()) +
                   list(model.head_dead_raw.parameters()))
    backbone_params = list(model.backbone.parameters())
    return optim.AdamW([
        {'params': backbone_params, 'lr': CFG.LR_BACKBONE, 'weight_decay': CFG.WD},
        {'params': head_params,     'lr': CFG.LR_HEAD,     'weight_decay': CFG.WD},
    ])
def build_optimizer(model: BiomassModel):
    # 1. Get backbone parameter IDs for exclusion
    backbone_ids = {id(p) for p in model.backbone.parameters()}
    
    # 2. Separate params into backbone vs. everything else (heads, fusion, etc.)
    backbone_params = []
    rest_params = []
    
    for p in model.parameters():
        if p.requires_grad:
            if id(p) in backbone_ids:
                backbone_params.append(p)
            else:
                rest_params.append(p)
    
    return optim.AdamW([
        {'params': backbone_params, 'lr': CFG.LR_BACKBONE, 'weight_decay': CFG.WD},
        {'params': rest_params,     'lr': CFG.LR_REST,     'weight_decay': CFG.WD},
])
def build_scheduler(optimizer):
    def lr_lambda(epoch):
        e = max(0, epoch - 1)
        if e < CFG.WARMUP_EPOCHS:
            return float(e + 1) / float(max(1, CFG.WARMUP_EPOCHS))
        progress = (e - CFG.WARMUP_EPOCHS) / float(max(1, CFG.EPOCHS - CFG.WARMUP_EPOCHS))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, lr_lambda)

def train_epoch(model, loader, opt, scheduler, device, ema: ModelEmaV2 | None = None):
    model.train()
    running = 0.0
    opt.zero_grad()
    amp_ctx = (lambda: torch.amp.autocast(device_type='cuda')) if torch.cuda.is_available() else (lambda: nullcontext())
    itera = tqdm(loader, desc='train', leave=False) if CFG.USE_TQDM else loader
    for i, (l, r, lab) in enumerate(itera):
        l, r, lab = l.to(device, non_blocking=True), r.to(device, non_blocking=True), lab.to(device, non_blocking=True)
        with amp_ctx():
            total, gdm, green, clover, dead = model(l, r)
            loss = biomass_loss((total, gdm, green, clover, dead), lab, w=CFG.LOSS_WEIGHTS) / CFG.GRAD_ACC
        scaler.scale(loss).backward()
        running += loss.item() * l.size(0) * CFG.GRAD_ACC
        
        if (i + 1) % CFG.GRAD_ACC == 0 or (i + 1) == len(loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            if ema is not None:
                ema.update(model.module if hasattr(model, 'module') else model)
            opt.zero_grad()
    scheduler.step()
    return running / len(loader.dataset)
```

## 5-Fold Training Loop with EMA

```python
# print('Loading data...')
# df_long = pd.read_csv(CFG.TRAIN_CSV)
# df_wide = df_long.pivot(index='image_path', columns='target_name', values='target').reset_index()
# assert df_wide['image_path'].is_unique, 'Leakage risk: duplicate image_path rows'

# # Merge metadata (Sampling_Date, State) for stratification
# if 'Sampling_Date' in df_long.columns and 'State' in df_long.columns:
#     print('Merging metadata for stratification...')
#     meta_df = df_long[['image_path', 'Sampling_Date', 'State']].drop_duplicates()
#     df_wide = df_wide.merge(meta_df, on='image_path', how='left')

# # Keep necessary columns
# df_wide = df_wide[['image_path', 'Sampling_Date', 'State'] + CFG.ALL_TARGET_COLS]
# print(f'{len(df_wide)} training images')

# # Use StratifiedGroupKFold
# sgkf = StratifiedGroupKFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=CFG.SEED)
# oof_true, oof_pred, fold_summary = [], [], []

# # Split based on groups (Sampling_Date) and stratification target (State)
# groups = df_wide['Sampling_Date']
# y_stratify = df_wide['State']

# # models_list = [] # Removed to save memory

# for fold, (tr_idx, val_idx) in enumerate(sgkf.split(df_wide, y_stratify, groups=groups)):
#     if fold not in CFG.FOLDS_TO_TRAIN:
#         print(f'Skipping fold {fold} as per configuration.')
#         continue
#     print('\n' + '='*70)
#     print(f'FOLD {fold+1}/{CFG.N_FOLDS} | {len(tr_idx)} train / {len(val_idx)} val')
#     print('='*70)
#     torch.cuda.empty_cache(); gc.collect()
    
#     tr_df  = df_wide.iloc[tr_idx].reset_index(drop=True)
#     val_df = df_wide.iloc[val_idx].reset_index(drop=True)

#     # Quick train/val comparison for this fold
#     try:
#         compare_train_val(tr_df, val_df, CFG.ALL_TARGET_COLS, show_plots=True)
#     except Exception as e:
#         print('Warning: compare_train_val failed:', e)

#     tr_set = BiomassDataset(tr_df,  get_train_transforms(), CFG.TRAIN_IMAGE_DIR)
    
#     # Create TTA loaders
#     val_loaders = []
#     for mode in range(CFG.VAL_TTA_TIMES): # 0: orig, 1: hflip, 2: vflip, 3: rot90
#         val_set_tta = BiomassDataset(val_df, get_tta_transforms(mode), CFG.TRAIN_IMAGE_DIR)
#         val_loader_tta = DataLoader(val_set_tta, batch_size=CFG.BATCH_SIZE, shuffle=False, num_workers=CFG.NUM_WORKERS, pin_memory=True)
#         val_loaders.append(val_loader_tta)

#     tr_loader  = DataLoader(tr_set, batch_size=CFG.BATCH_SIZE, shuffle=True, num_workers=CFG.NUM_WORKERS, pin_memory=True, drop_last=True)
    
#     print('Building model...')
#     backbone_path = getattr(CFG, 'BACKBONE_PATH', None)
#     base_model = BiomassModel(CFG.MODEL_NAME, pretrained=CFG.PRETRAINED, backbone_path=backbone_path).to(CFG.DEVICE)
    
#     # Load pretrained fold weights if available (for resuming or fine-tuning)
#     if getattr(CFG, 'PRETRAINED_DIR', None) and os.path.isdir(CFG.PRETRAINED_DIR):
#         pretrained_path = os.path.join(CFG.PRETRAINED_DIR, f'best_model_fold{fold}.pth')
#         if os.path.exists(pretrained_path):
#             try:
#                 state = torch.load(pretrained_path, map_location='cpu')
#                 # support raw state_dict or dict-with-keys
#                 if isinstance(state, dict) and ('model_state_dict' in state or 'state_dict' in state):
#                     key = 'model_state_dict' if 'model_state_dict' in state else 'state_dict'
#                     sd = state[key]
#                 else:
#                     sd = state
#                 base_model.load_state_dict(sd, strict=False)
#                 base_model.to(CFG.DEVICE)
#                 print(f'  ✓ Loaded pretrained weights for fold {fold} from {pretrained_path}')
#             except Exception as e:
#                 print(f'  ✗ Failed to load pretrained fold {fold}: {e}')
#         else:
#             print(f'  (No pretrained file for fold {fold} at {pretrained_path})')
#     else:
#         print('  (No PRETRAINED_DIR configured or directory missing)')
        
#     model = nn.DataParallel(base_model)
#     set_backbone_requires_grad(base_model, False)
#     optimizer = build_optimizer(base_model)
#     scheduler = build_scheduler(optimizer)
#     ema = ModelEmaV2(base_model, decay=CFG.EMA_DECAY)
    
#     best_global_r2 = -np.inf
#     patience = 0
#     best_fold_preds = None; best_fold_true = None
#     best_avg_r2 = -np.inf
    
#     # Define save path
#     save_path = os.path.join(CFG.MODEL_DIR, f'best_model_fold{fold}.pth')
    
#     for epoch in range(1, CFG.EPOCHS + 1):
#         if epoch == CFG.FREEZE_EPOCHS + 1:
#             patience = 0
#             set_backbone_requires_grad(base_model, True)
#             print(f'Epoch {epoch}: backbone unfrozen')
        
#         tr_loss = train_epoch(model, tr_loader, optimizer, scheduler, CFG.DEVICE, ema)
#         eval_model = ema.module if ema is not None else (model.module if hasattr(model, 'module') else model)
        
#         # Use TTA validation
#         val_loss, global_r2, avg_r2, per_r2, preds_fold, true_fold = valid_epoch_tta(eval_model, val_loaders, CFG.DEVICE)
        
#         per_r2_str = ' | '.join([f'{CFG.ALL_TARGET_COLS[i][:5]}: {r2:.3f}' for i, r2 in enumerate(per_r2)])
#         lrs = [pg['lr'] for pg in optimizer.param_groups]
#         print(f'Fold {fold} | Epoch {epoch:02d} | TLoss {tr_loss:.5f} | VLoss {val_loss:.5f} |avgR2 {avg_r2:.4f}| GlobalR² {global_r2:.4f} {"[BEST]" if global_r2 > best_global_r2 else ""}')
#         print(f'  → {per_r2_str}')
        
#         if global_r2 > best_global_r2:
#             best_global_r2 = global_r2
#             best_avg_r2 = avg_r2
            
#             # Save the EMA weights (best state) to disk immediately
#             # Clone to CPU to avoid memory issues
#             best_state = {k: v.cpu().clone() for k, v in eval_model.state_dict().items()}
#             torch.save(best_state, save_path)
#             print(f'  → SAVED EMA weights to {save_path} (GlobalR²: {best_global_r2:.4f})')
#             del best_state # Free memory
            
#             patience = 0
#             best_fold_preds = preds_fold; best_fold_true = true_fold
#         else:
#             patience += 1
#             if patience >= CFG.PATIENCE:
#                     print(f'  → EARLY STOP (no improvement in {CFG.PATIENCE} epochs)')
#                     break
                
#         del preds_fold, true_fold
#         torch.cuda.empty_cache()
#         gc.collect()
    
#     if best_fold_preds is not None:
#         oof_true.append(best_fold_true); oof_pred.append(best_fold_preds)
#         fold_summary.append({'fold': fold, 'global_r2': best_global_r2,'avg_r2':avg_r2})
    
#     # Cleanup for this fold
#     del model, base_model, tr_loader, val_loaders, optimizer, scheduler, ema
#     if 'eval_model' in locals(): del eval_model
#     torch.cuda.empty_cache(); gc.collect()

# if oof_true:
#     oof_true_arr = np.concatenate(oof_true, axis=0)
#     oof_pred_arr = np.concatenate(oof_pred, axis=0)
#     oof_global_r2, oof_avg_r2, oof_per_r2 = weighted_r2_score_global(oof_true_arr, oof_pred_arr)

#     print('\nTraining complete! Models saved in:', CFG.MODEL_DIR)
#     print('Fold summary:')
#     for fs in fold_summary:
#         print(f"  Fold {fs['fold']}: Global R² = {fs['global_r2']:.4f}, Avg R² = {fs.get('avg_r2', float('nan')):.4f}")
#     print(f'OOF Global Weighted R²: {oof_global_r2:.4f} | OOF Avg Target R²: {oof_avg_r2:.4f}')
#     print('OOF Per-target:', dict(zip(CFG.ALL_TARGET_COLS, [f"{r:.4f}" for r in oof_per_r2])))
# else:
#     print('No OOF predictions collected.')
```

## Submit

```python
# ===============================================================
# 4. DEFINE TTA TRANSFORMS
# ===============================================================
def get_tta_transforms(num_transforms):
    """
    Returns a list of TTA transform pipelines.
    Each pipeline represents a different augmentation view.
    """
    normalize = A.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    to_tensor = ToTensorV2()
    
    all_tta_transforms = [
        # View 1: Original (no flip)
        A.Compose([
            A.Resize(CFG.IMG_SIZE, CFG.IMG_SIZE),
            normalize,
            to_tensor
        ]),
        
        # View 2: Horizontal Flip
        A.Compose([
            A.Resize(CFG.IMG_SIZE, CFG.IMG_SIZE),
            A.HorizontalFlip(p=1.0),
            normalize,
            to_tensor
        ]),
        
        # View 3: Vertical Flip
        A.Compose([
            A.Resize(CFG.IMG_SIZE, CFG.IMG_SIZE),
            A.VerticalFlip(p=1.0),
            normalize,
            to_tensor
        ]),
        
        # View 4: Both Flips
        A.Compose([
            A.Resize(CFG.IMG_SIZE, CFG.IMG_SIZE),
            A.HorizontalFlip(p=1.0),
            A.VerticalFlip(p=1.0),
            normalize,
            to_tensor
        ]),
    ]
    tta_transforms = all_tta_transforms[:num_transforms]
    return tta_transforms

print(f"✓ TTA transforms defined ({CFG.TTA_STEPS} views)")
```

```python
# ===============================================================
# 5. CREATE TEST DATASET
# ===============================================================
def clean_image(img):
    # Safe crop (remove bottom artifacts) + inpaint orange date stamp
    h, w = img.shape[:2]
    img = img[0:int(h * 0.90), :]
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    lower = np.array([5, 150, 150])
    upper = np.array([25, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=2)
    if np.sum(mask) > 0:
        img = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
    return img
class BiomassTestDataset(Dataset):
    """
    Test dataset for biomass images.
    Splits each 2000×1000 image into left and right 1000×1000 halves.
    """
    def __init__(self, img_dir):
        self.img_dir = img_dir
        self.paths = sorted([
            os.path.join(img_dir, f) for f in os.listdir(img_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])
        self.filenames = [os.path.basename(p) for p in self.paths]
    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        path = self.paths[idx]
        
        # Read image
        img = cv2.imread(path)
        if img is None:
            print(f"Warning: Could not read {path}, using blank image")
            img = np.zeros((1000, 2000, 3), dtype=np.uint8)
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = clean_image(img)
        # Split into left and right halves
        h, w = img.shape[:2]
        mid = w // 2
        left = img[:, :mid].copy()
        right = img[:, mid:].copy()
        
        return left, right, self.filenames[idx]

print("✓ Test dataset class defined")
```

```python
# ===============================================================
# 8. RUN INFERENCE WITH TTA (UPDATED to honor CFG.FOLDS_TO_TRAIN)
# ===============================================================

@torch.no_grad()
def predict_with_tta(model, left_np, right_np, tta_transforms):
    """
    Predict using TTA with a SINGLE model.
    
    Args:
        model: Single trained model
        left_np: Left half of image (numpy array)
        right_np: Right half of image (numpy array)
        tta_transforms: List of augmentation transforms
    
    Returns:
        numpy array: [total, gdm, green] predictions (averaged over TTA)
    """
    all_tta_preds = []
    
    # Loop over TTA views
    for tfm in tta_transforms:
        # Apply transform to both halves
        left_tensor = tfm(image=left_np)['image'].unsqueeze(0).to(CFG.DEVICE)
        right_tensor = tfm(image=right_np)['image'].unsqueeze(0).to(CFG.DEVICE)
        
        total, gdm, green, clover, dead = model(left_tensor, right_tensor)
        
        # Extract values
        p_total = total.cpu().item()
        p_gdm = gdm.cpu().item()
        p_green = green.cpu().item()
        
        all_tta_preds.append([p_total, p_gdm, p_green])
    
    # Average across TTA views
    final_pred = np.mean(all_tta_preds, axis=0)
    
    return final_pred


def run_inference():
    """
    Main inference function.
    Returns: (predictions_array, image_filenames)
    Notes:
      - Now respects `CFG.FOLDS_TO_TRAIN` (if set) and averages only over successfully loaded folds.
      - If no fold weights are found for the requested folds, an error is raised.
    """
    print("\n" + "="*70)
    print("STARTING INFERENCE")
    print("="*70)
    
    # Create dataset and loader
    dataset = BiomassTestDataset(CFG.TEST_IMAGE_DIR)
    # Note: batch_size=1 is required for the current predict_with_tta implementation
    loader = DataLoader(
        dataset,
        batch_size=1,  
        shuffle=False,
        num_workers=CFG.NUM_WORKERS,
        pin_memory=True
    )
    
    tta_transforms = get_tta_transforms(CFG.TTA_STEPS)
    
    # Initialize accumulator for predictions
    # Shape: (num_samples, 3) for Total, GDM, Green
    accumulated_preds = np.zeros((len(dataset), 3), dtype=np.float32)

    # Use configured folds, fallback to full range if not set or empty
    folds_to_use = getattr(CFG, 'FOLDS_TO_TRAIN', list(range(CFG.N_FOLDS)))
    if not folds_to_use:
        folds_to_use = list(range(CFG.N_FOLDS))

    print(f"Folds requested for inference: {folds_to_use}")

    # Use filenames from dataset (guaranteed consistent ordering with loader because shuffle=False)
    filenames = dataset.filenames.copy()

    successful_folds = 0
    
    # Loop over requested folds only
    for fold in folds_to_use:
        print(f"\nProcessing Fold {fold}...")
        # Load model for this fold
        model_dir = CFG.MODEL_DIR_012 if fold in [0,1,2] else CFG.MODEL_DIR_34
        backbone_path = getattr(CFG, 'BACKBONE_PATH', None)
        model = BiomassModel(CFG.MODEL_NAME, pretrained=False, backbone_path=backbone_path)
        
        # Load weights
        weight_path = os.path.join(model_dir, f'best_model_fold{fold}.pth')
        if not os.path.exists(weight_path):
            print(f"Warning: Model file {weight_path} not found! Skipping fold {fold}.")
            del model
            torch.cuda.empty_cache(); gc.collect()
            continue
            
        state = torch.load(weight_path, map_location='cpu')
        # Handle state dict keys if necessary (e.g. if saved with 'model_state_dict' key)
        if isinstance(state, dict) and ('model_state_dict' in state or 'state_dict' in state):
             key = 'model_state_dict' if 'model_state_dict' in state else 'state_dict'
             sd = state[key]
        else:
             sd = state
        
        model.load_state_dict(sd)
        model.to(CFG.DEVICE)
        model.eval()
        
        # Run inference for this fold
        for i, (left, right, filename) in enumerate(tqdm(loader, desc=f"Fold {fold}")):
            # left and right are batches of size 1, convert to numpy for TTA function
            left_np = left[0].numpy()
            right_np = right[0].numpy()
            
            # Predict
            pred = predict_with_tta(model, left_np, right_np, tta_transforms)
            accumulated_preds[i] += pred
            
        successful_folds += 1
        # Cleanup model to save memory
        del model
        torch.cuda.empty_cache(); gc.collect()
        
    if successful_folds == 0:
        raise FileNotFoundError(f"No model weights found for requested folds: {folds_to_use}")

    # Average predictions over the number of successfully loaded folds
    final_predictions = accumulated_preds / successful_folds
    
    print(f"\nInference complete. Successfully used {successful_folds} fold(s) out of {len(folds_to_use)} requested.")
    return final_predictions, filenames
```

```python
# ===============================================================
# 9. POST-PROCESS PREDICTIONS
# ===============================================================
def postprocess_predictions(preds_direct):
    """
    Calculate derived targets from direct predictions.
    
    Input: (n_samples, 3) array with [total, gdm, green]
    Output: (n_samples, 5) array with [green, dead, clover, gdm, total]
    """
    print("\nPost-processing predictions...")
    
    # Extract direct predictions
    pred_total = preds_direct[:, 0]
    pred_gdm = preds_direct[:, 1]
    pred_green = preds_direct[:, 2]
    
    # Calculate derived targets with non-negativity constraint
    pred_clover = np.maximum(0, pred_gdm - pred_green)
    pred_dead = np.maximum(0, pred_total - pred_gdm)
    
    # Stack in the order of ALL_TARGET_COLS
    # ['Dry_Green_g', 'Dry_Dead_g', 'Dry_Clover_g', 'GDM_g', 'Dry_Total_g']
    preds_all = np.stack([
        pred_green,
        pred_dead,
        pred_clover,
        pred_gdm,
        pred_total
    ], axis=1)
    
    print(f"✓ Post-processing complete")
    print(f"  Output shape: {preds_all.shape}")
    print(f"\nPrediction statistics:")
    for i, col in enumerate(CFG.ALL_TARGET_COLS):
        print(f"  {col:15s}: mean={preds_all[:, i].mean():.2f}, "
              f"std={preds_all[:, i].std():.2f}, "
              f"min={preds_all[:, i].min():.2f}, "
              f"max={preds_all[:, i].max():.2f}")
    
    return preds_all
```

```python
# ===============================================================
# 10. CREATE SUBMISSION FILE (FIXED)
# ===============================================================
def create_submission(predictions, filenames):
    """
    Create submission file in the required format.
    
    Args:
        predictions: (n_images, 5) array with all target predictions
        filenames: list of test image filenames
    """
    print("\n" + "="*70)
    print("CREATING SUBMISSION FILE")
    print("="*70)
    
    # Step 0: Load test.csv first to check the image_path format
    test_df = pd.read_csv(CFG.TEST_CSV)
    print(f"\nTest CSV loaded: {len(test_df)} rows")
    print(f"Sample image_path from test.csv: {test_df['image_path'].iloc[0]}")
    print(f"Sample filename from predictions: {filenames[0]}")
    
    # Step 1: Fix image_path format to match test.csv
    # If test.csv has "test/ID123.jpg" but we have "ID123.jpg", add the prefix
    test_path_example = test_df['image_path'].iloc[0]
    if '/' in test_path_example:
        # Extract the subdirectory prefix (e.g., "test/")
        prefix = test_path_example.rsplit('/', 1)[0] + '/'
        corrected_filenames = [prefix + fn for fn in filenames]
        print(f"Corrected path format: {corrected_filenames[0]}")
    else:
        corrected_filenames = filenames
    
    # Step 2: Create wide-format DataFrame with corrected paths
    preds_wide = pd.DataFrame(predictions, columns=CFG.ALL_TARGET_COLS)
    preds_wide.insert(0, 'image_path', corrected_filenames)
    
    print(f"\nWide format predictions:")
    print(preds_wide.head())
    
    # Step 3: Convert to long format (melt)
    preds_long = preds_wide.melt(
        id_vars=['image_path'],
        value_vars=CFG.ALL_TARGET_COLS,
        var_name='target_name',
        value_name='target'
    )
    
    print(f"\nLong format predictions (first 10 rows):")
    print(preds_long.head(10))
    
    # Step 4: Debug the merge
    print(f"\nDebug: Checking if paths match...")
    print(f"Unique paths in test_df: {test_df['image_path'].nunique()}")
    print(f"Unique paths in preds_long: {preds_long['image_path'].nunique()}")
    
    common_paths = set(test_df['image_path'].unique()) & set(preds_long['image_path'].unique())
    print(f"Common paths found: {len(common_paths)}")
    
    if len(common_paths) == 0:
        print("\n❌ ERROR: No matching paths found!")
        print(f"Test CSV paths sample: {list(test_df['image_path'].unique()[:3])}")
        print(f"Prediction paths sample: {list(preds_long['image_path'].unique()[:3])}")
        raise ValueError("Path mismatch between test.csv and predictions")
    
    # Step 5: Merge to get sample_ids
    submission = pd.merge(
        test_df[['sample_id', 'image_path', 'target_name']],
        preds_long,
        on=['image_path', 'target_name'],
        how='left'
    )
    
    # Step 6: Keep only required columns
    submission = submission[['sample_id', 'target']]
    
    # Step 7: Check for missing values
    missing_count = submission['target'].isna().sum()
    if missing_count > 0:
        print(f"\n⚠ Warning: {missing_count} missing predictions found!")
        print("Sample missing entries:")
        print(submission[submission['target'].isna()].head())
        submission.loc[submission['target'].isna(), 'target'] = 0.0
    
    # Step 8: Sort by sample_id
    submission = submission.sort_values('sample_id').reset_index(drop=True)
    
    # Step 9: Save to CSV
    output_path = os.path.join(CFG.SUBMISSION_DIR, 'submission_dinoV3.csv')
    submission.to_csv(output_path, index=False)
    
    print(f"\n✓ Submission file saved: {output_path}")
    print(f"  Total rows: {len(submission)}")
    print(f"\nPrediction statistics:")
    print(f"  Min: {submission['target'].min():.4f}")
    print(f"  Max: {submission['target'].max():.4f}")
    print(f"  Mean: {submission['target'].mean():.4f}")
    print(f"  Non-zero values: {(submission['target'] > 0).sum()}/{len(submission)}")
    
    print(f"\nFirst 10 rows:")
    print(submission.head(10))
    print(f"\nLast 10 rows:")
    print(submission.tail(10))
    
    # Step 10: Validation checks
    print(f"\n" + "="*70)
    print("VALIDATION CHECKS")
    print("="*70)
    print(f"✓ Expected rows: {len(test_df)}")
    print(f"✓ Actual rows: {len(submission)}")
    print(f"✓ Match: {len(submission) == len(test_df)}")
    print(f"✓ No missing values: {not submission['target'].isna().any()}")
    print(f"✓ All sample_ids unique: {submission['sample_id'].is_unique}")
    print(f"✓ Has non-zero predictions: {(submission['target'] > 0).any()}")
    
    return submission

# Create submission
# Post-process predictions
# Run inference
if CFG.CREATE_SUBMISSION:
    predictions_direct, test_filenames = run_inference()
    predictions_all = postprocess_predictions(predictions_direct)
    submission_df = create_submission(predictions_all, test_filenames)
```

# Siglip

## Multiview

```python
import os
import glob
from pathlib import Path
import sys
from tqdm.auto import tqdm
import json
from copy import deepcopy

import pandas as pd
import numpy as np
import os
import math
import random

import torch
import torch.nn as nn
from torchvision import transforms

from PIL import Image
import cv2

from transformers import AutoProcessor, AutoImageProcessor, AutoModel, Siglip2Model, Siglip2ImageProcessor, SiglipModel, SiglipImageProcessor
from transformers import AutoModel, AutoTokenizer
from sklearn.model_selection import KFold, GroupKFold
from sklearn.linear_model import LinearRegression, Ridge, Lasso, BayesianRidge
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor, ExtraTreesRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.svm import LinearSVR
from sklearn.dummy import DummyRegressor

from sklearn.decomposition import PCA
from sklearn import model_selection
from sklearn import preprocessing

from dataclasses import dataclass
from typing import Optional, Dict

import matplotlib.pyplot as plt
```

```python
def seeding(SEED):
    np.random.seed(SEED)
    random.seed(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)
    torch.manual_seed(SEED)
    # pl.seed_everything(SEED)
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print('seeding done!!!')

def flush():
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

seeding(42)
```

```python
@dataclass
class Config:
    # Data paths
    DATA_PATH: Path = Path("/kaggle/input/csiro-biomass/")
    TRAIN_DATA_PATH: Path = DATA_PATH/'train'
    TEST_DATA_PATH: Path = DATA_PATH/'test'

    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = 42

cfg = Config()
seeding(cfg.seed)
```

```python
def pivot_table(df: pd.DataFrame)->pd.DataFrame:

    if 'target' in df.columns.tolist():
        df_pt = pd.pivot_table(
            df, 
            values='target', 
            index=['image_path', 'Sampling_Date', 'State', 'Species', 'Pre_GSHH_NDVI', 'Height_Ave_cm'], 
            columns='target_name', 
            aggfunc='mean'
        ).reset_index()
    else:
        df['target'] = 0
        df_pt = pd.pivot_table(
            df, 
            values='target', 
            index='image_path', 
            columns='target_name', 
            aggfunc='mean'
        ).reset_index()
    return df_pt

# train_df = pd.read_csv(cfg.DATA_PATH/'train.csv')
test_df = pd.read_csv(cfg.DATA_PATH/'test.csv')
# train_df = pivot_table(df=train_df)
test_df = pivot_table(df=test_df)

# train_df['image_path'] = train_df['image_path'].apply(lambda p: str(cfg.DATA_PATH / p))
test_df['image_path'] = test_df['image_path'].apply(lambda p: str(cfg.DATA_PATH / p))
# test_df.head()

train_df = pd.read_csv("/kaggle/input/csiro-datasplit/csiro_data_split.csv")
```

```python
def melt_table(df: pd.DataFrame) -> pd.DataFrame:
    TARGET_NAMES = ['Dry_Clover_g', 'Dry_Dead_g', 'Dry_Green_g', 'Dry_Total_g', 'GDM_g']
    melted = df.melt(
        id_vars='image_path',
        value_vars=TARGET_NAMES,
        var_name='target_name',
        value_name='target'
    )
    melted['sample_id'] = (
        melted['image_path']
        .str.replace(r'^.*/', '', regex=True)  # remove folder path, keep filename
        .str.replace('.jpg', '', regex=False)  # remove extension
        + '__' + melted['target_name']
    )
    
    return melted[['sample_id', 'image_path', 'target_name', 'target']]

# t1 = melt_table(test_df)
# t1
```

```python
# train_df.head(1)
# train_df['Species'].value_counts()
```

```python
# 357 / 5
```

```python
def split_image(image, patch_size=520, overlap=16):
    h, w, c = image.shape
    stride = patch_size - overlap
    
    patches = []
    coords  = []   # (y1, x1, y2, x2)
    
    for y in range(0, h, stride):
        for x in range(0, w, stride):
            y1 = y
            x1 = x
            y2 = y + patch_size
            x2 = x + patch_size
            
            # Pad last patch if needed (very rare with your fixed 1000×2000)
            patch = image[y1:y2, x1:x2, :]
            if patch.shape[0] < patch_size or patch.shape[1] < patch_size:
                pad_h = patch_size - patch.shape[0]
                pad_w = patch_size - patch.shape[1]
                patch = np.pad(patch, ((0,pad_h), (0,pad_w), (0,0)), mode='reflect')
            
            patches.append(patch)
            coords.append((y1, x1, y2, x2))
    
    return patches, coords
```

```python
def get_model(model_path: str, device: str = 'cpu'):
    model = AutoModel.from_pretrained(
        model_path,
        local_files_only=True
    )
    processor = AutoImageProcessor.from_pretrained(model_path)
    return model.eval().to(device), processor

dino_path = "/kaggle/input/dinov2/pytorch/giant/1"
siglip_path = "/kaggle/input/google-siglip-so400m-patch14-384/transformers/default/1"

# dino_model, dino_processor = get_model(
#     model_path=dino_path, device=device
# )

# siglip_model, siglip_processor = get_model(
#     model_path=siglip_path, device=device
# )
```

```python
def compute_embeddings(model_path, df, patch_size=520):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    model, processor = get_model(
        model_path=model_path, device=device
    )

    IMAGE_PATHS = []
    EMBEDDINGS = []

    for i, row in tqdm(df.iterrows(), total=len(df)):
        img_path = row['image_path']
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        patches, coords = split_image(img, patch_size=patch_size)
        images = [Image.fromarray(p).convert("RGB") for p in patches]

        inputs = processor(images=images, return_tensors="pt").to(model.device)
        with torch.no_grad():
            if 'siglip' in model_path:
                features = model.get_image_features(**inputs)
            elif 'dino' in model_path:
                features = model(**inputs).pooler_output
                # patches = model(**inputs).last_hidden_state
                # features = patches[:, 0, :]
            else:
                raise Exception("Model should be dino or siglip")
        embeds = features.mean(dim=0).detach().cpu().numpy()
        EMBEDDINGS.append(embeds)
        IMAGE_PATHS.append(img_path)

    embeddings = np.stack(EMBEDDINGS, axis=0)
    n_features = embeddings.shape[1]
    emb_columns = [f"emb{i+1}" for i in range(n_features)]
    emb_df = pd.DataFrame(embeddings, columns=emb_columns)
    emb_df['image_path'] = IMAGE_PATHS
    df_final = df.merge(emb_df, on='image_path', how='left')
    flush()
    return df_final 

# train_siglip_df = compute_embeddings(model_path=siglip_path, df=train_df, patch_size=520)
train_siglip_df = train_df.copy()
test_siglip_df = compute_embeddings(model_path=siglip_path, df=test_df, patch_size=520)

flush()
```

## Text embeddings

```python
# "dense pasture grass",
#         "sparse pasture vegetation",
#         "patchy grass cover",
#         "bare soil patches in grass",
#         "thick tangled grass",
#         "open low-density pasture",
#         "dry cracked soil",
#         "dry canopy",
#         "low moisture vegetation",
#         "dry pasture with yellow tones",
#         "wilted grass"
```

```python
# def generate_semantic_features(image_embeddings, model_path=siglip_path):
#     print(f"Loading SigLIP Text Encoder from {model_path}...")
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
#     try:
#         model = AutoModel.from_pretrained(model_path).to(device)
#         tokenizer = AutoTokenizer.from_pretrained(model_path)
#     except Exception as e:
#         print(f"Error loading model: {e}")
#         return None

#     # Prepare Image Tensor
#     if isinstance(image_embeddings, np.ndarray):
#         img_tensor = torch.tensor(image_embeddings, dtype=torch.float32).to(device)
#     else:
#         img_tensor = image_embeddings.to(device)
#     img_tensor = img_tensor / img_tensor.norm(p=2, dim=-1, keepdim=True)

#     # Define Prompts (The Dictionary from above)
#     AGRONOMIC_PROMPTS = {
#         "biomass": {
#             "pos": ["dense tall pasture", "high biomass vegetation", "thick grassy volume"],
#             "neg": ["bare soil", "sparse vegetation", "very short clipped grass"]
#         },
#         "green_vs_brown": {
#             "pos": ["lush green vibrant pasture", "green leaves"],
#             "neg": ["dry brown dead grass", "yellow straw-like vegetation"]
#         },
#         "clover_presence": {
#             "pos": ["white clover patches", "broadleaf clover", "clover flowers"],
#             "neg": ["pure ryegrass", "blade-like grass leaves", "monoculture grass"]
#         },
#         "litter_dead": {
#             "pos": ["accumulated dead plant litter", "mat of dry dead grass"],
#             "neg": ["clean fresh growth", "upright green stalks"]
#         }
#     }

#     feature_store = []
    
#     with torch.no_grad():
#         for axis_name, prompts in AGRONOMIC_PROMPTS.items():
#             # Encode Positive Prompts
#             pos_inputs = tokenizer(prompts["pos"], padding="max_length", return_tensors="pt").to(device)
#             pos_emb = model.get_text_features(**pos_inputs)
#             pos_emb = pos_emb / pos_emb.norm(p=2, dim=-1, keepdim=True)
            
#             # Encode Negative Prompts
#             neg_inputs = tokenizer(prompts["neg"], padding="max_length", return_tensors="pt").to(device)
#             neg_emb = model.get_text_features(**neg_inputs)
#             neg_emb = neg_emb / neg_emb.norm(p=2, dim=-1, keepdim=True)
            
#             # Create Mean Embeddings for the concept groups
#             pos_concept = pos_emb.mean(dim=0, keepdim=True)
#             neg_concept = neg_emb.mean(dim=0, keepdim=True)
            
#             # Calculate Similarity
#             # Shape: (N_imgs, 1)
#             sim_pos = torch.matmul(img_tensor, pos_concept.T).cpu().numpy()
#             sim_neg = torch.matmul(img_tensor, neg_concept.T).cpu().numpy()
            
#             # --- Feature Engineering ---
#             # 1. The Axis Score (The "Ruler"): Pos - Neg
#             axis_score = sim_pos - sim_neg
            
#             # 2. The Raw Activation (Max fit):
#             max_act = np.maximum(sim_pos, sim_neg)
            
#             feature_store.append(axis_score)
#             # Optional: Add raw similarities if you suspect non-linearities
#             # feature_store.append(sim_pos) 
            
#     # Stack features: (N_samples, N_axes)
#     semantic_features = np.hstack(feature_store)
    
#     print(f"Generated {semantic_features.shape[1]} semantic axis features.")
#     return semantic_features

def generate_semantic_features(image_embeddings, model_path=siglip_path):
    """
    Generates 'Concept Scores' by averaging synonyms and calculating biological ratios.
    """
    print(f"Loading SigLIP Text Encoder from {model_path}...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    try:
        model = AutoModel.from_pretrained(model_path).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

    # 1. Define Concept Ensembles (Grouping synonyms reduces noise)
    concept_groups = {
        # Quantity Anchors
        "bare": ["bare soil", "dirt ground", "sparse vegetation", "exposed earth"],
        "sparse": ["low density pasture", "thin grass", "short clipped grass"],
        "medium": ["average pasture cover", "medium height grass", "grazed pasture"],
        "dense": ["dense tall pasture", "thick grassy volume", "high biomass", "overgrown vegetation"],
        
        # State Anchors
        "green": ["lush green vibrant pasture", "photosynthesizing leaves", "fresh growth"],
        "dead": ["dry brown dead grass", "yellow straw", "senesced material", "standing hay"],
        
        # Species Anchors
        "clover": ["white clover", "trifolium repens", "broadleaf legume", "clover flowers"],
        "grass": ["ryegrass", "blade-like leaves", "fescue", "grassy sward"],
        "weeds": ["broadleaf weeds", "thistles", "non-pasture vegetation"]
    }
    
    # 2. Encode and Average Prompts for each Concept
    concept_vectors = {}
    with torch.no_grad():
        for name, prompts in concept_groups.items():
            inputs = tokenizer(prompts, padding="max_length", return_tensors="pt").to(device)
            emb = model.get_text_features(**inputs)
            emb = emb / emb.norm(p=2, dim=-1, keepdim=True)
            # Average the embeddings of synonyms to get a stable "Concept Vector"
            concept_vectors[name] = emb.mean(dim=0, keepdim=True)

    # 3. Compute Concept Scores
    if isinstance(image_embeddings, np.ndarray):
        img_tensor = torch.tensor(image_embeddings, dtype=torch.float32).to(device)
    else:
        img_tensor = image_embeddings.to(device)
    img_tensor = img_tensor / img_tensor.norm(p=2, dim=-1, keepdim=True)

    scores = {}
    for name, vec in concept_vectors.items():
        # Dot product
        scores[name] = torch.matmul(img_tensor, vec.T).cpu().numpy().flatten()
    
    # 4. Feature Engineering: Explicit Ratios
    # These help models distinguish between "High Biomass Dead" vs "High Biomass Green"
    
    # Convert dict to DataFrame for easy math
    df_scores = pd.DataFrame(scores)
    
    # A. Greenness Ratio: Green / (Green + Dead)
    df_scores['ratio_greenness'] = df_scores['green'] / (df_scores['green'] + df_scores['dead'] + 1e-6)
    
    # B. Legume Fraction: Clover / (Clover + Grass)
    df_scores['ratio_clover'] = df_scores['clover'] / (df_scores['clover'] + df_scores['grass'] + 1e-6)
    
    # C. Vegetation Cover: (Dense + Medium) / (Bare + Sparse)
    df_scores['ratio_cover'] = (df_scores['dense'] + df_scores['medium']) / (df_scores['bare'] + df_scores['sparse'] + 1e-6)
    
    # D. "Volume": Max of density anchors
    df_scores['max_density'] = df_scores[['bare', 'sparse', 'medium', 'dense']].max(axis=1)

    print(f"Generated {df_scores.shape[1]} semantic features (Ensembles + Ratios).")
    return df_scores.values
```

## Feature engineering

```python
# import numpy as np
# from sklearn.base import BaseEstimator, TransformerMixin
# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA
# from sklearn.cross_decomposition import PLSRegression
# from sklearn.mixture import GaussianMixture
# from sklearn.linear_model import BayesianRidge
# from sklearn.metrics.pairwise import cosine_similarity

# class SupervisedEmbeddingEngine(BaseEstimator, TransformerMixin):
#     def __init__(self, 
#                  n_pca=0.95, 
#                  n_pls=10, # Increased slightly to capture specific biomass targets
#                  n_gmm=3,  # Reduced GMM. 5 is risky for N=357 (overfitting clusters)
#                  random_state=42):
        
#         self.n_pca = n_pca
#         self.n_pls = n_pls
#         self.n_gmm = n_gmm
#         self.random_state = random_state
        
#         self.scaler = StandardScaler()
#         self.pca = PCA(n_components=n_pca, random_state=random_state)
#         self.pls = PLSRegression(n_components=n_pls, scale=False)
#         # Using full makes it more robust to singularity
#         self.gmm = GaussianMixture(n_components=n_gmm, covariance_type='full', random_state=random_state)
        
#         self.pls_fitted_ = False

#     def fit(self, X, y=None, X_semantic=None):
#         # 1. Concatenate Embeddings + Semantic Features EARLY
#         # This allows PLS to find correlations between Text Scores and Biomass
#         if X_semantic is not None:
#             # Weight semantic features up slightly so PCA respects them
#             X_combined = np.hstack([X, X_semantic * 2.0]) 
#         else:
#             X_combined = X
            
#         X_scaled = self.scaler.fit_transform(X_combined)
        
#         # 2. Fit Unsupervised
#         self.pca.fit(X_scaled)
#         self.gmm.fit(X_scaled)
        
#         # 3. Fit PLS (Supervised)
#         if y is not None:
#             # Handle multi-output targets (the 5 biomass columns)
#             # Ensure y is (N, 5)
#             y_clean = y.values if hasattr(y, 'values') else y
#             self.pls.fit(X_scaled, y_clean)
#             self.pls_fitted_ = True
            
#         return self

#     def transform(self, X, X_semantic=None):
#         if X_semantic is not None:
#             X_combined = np.hstack([X, X_semantic * 2.0])
#         else:
#             X_combined = X
            
#         X_scaled = self.scaler.transform(X_combined)
#         return self._generate_features(X_scaled)

#     def _generate_features(self, X_scaled):
#         features = []
        
#         # PCA (Structure of data)
#         f_pca = self.pca.transform(X_scaled)
#         features.append(f_pca)
        
#         # PLS (Structure of Targets) - THIS IS CRITICAL
#         if self.pls_fitted_:
#             f_pls = self.pls.transform(X_scaled)
#             features.append(f_pls)
        
#         # GMM (Cluster Probabilities)
#         f_gmm = self.gmm.predict_proba(X_scaled)
#         features.append(f_gmm)
        
#         return np.hstack(features)

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import BayesianRidge
from sklearn.metrics.pairwise import cosine_similarity


class SupervisedEmbeddingEngine(BaseEstimator, TransformerMixin):
    def __init__(self, 
                 n_pca=0.98,  # Slightly higher to keep texture details
                 n_pls=8,     # Keep 8, it worked well
                 n_gmm=5,     
                 random_state=42):
        
        self.n_pca = n_pca
        self.n_pls = n_pls
        self.n_gmm = n_gmm
        self.random_state = random_state
        
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_pca, random_state=random_state)
        self.pls = PLSRegression(n_components=n_pls, scale=False)
        self.gmm = GaussianMixture(n_components=n_gmm, covariance_type='diag', random_state=random_state)

        self.pls_fitted_ = False

    def fit(self, X, y=None, X_semantic=None):
        # 1. Standard Scaling on IMAGE embeddings only
        X_scaled = self.scaler.fit_transform(X)
        
        # 2. Fit Unsupervised on IMAGES
        self.pca.fit(X_scaled)
        self.gmm.fit(X_scaled)
        
        # 3. Fit PLS on IMAGES (Supervised)
        if y is not None:
            y_clean = y.values if hasattr(y, 'values') else y
            self.pls.fit(X_scaled, y_clean)
            self.pls_fitted_ = True
        
        return self

    def transform(self, X, X_semantic=None):
        X_scaled = self.scaler.transform(X)
        return self._generate_features(X_scaled, X_semantic)

    def _generate_features(self, X_scaled, X_semantic=None):
        features = []
        
        # A. PCA (Texture/Structure from Images)
        f_pca = self.pca.transform(X_scaled)
        features.append(f_pca)
        
        # B. PLS (Biomass-correlated signals from Images)
        if self.pls_fitted_:
            f_pls = self.pls.transform(X_scaled)
            features.append(f_pls)
        
        # C. GMM (Cluster probs)
        f_gmm = self.gmm.predict_proba(X_scaled)
        features.append(f_gmm)
        
        # D. Semantic Features (LATE FUSION)
        # We append them raw. They are already high-level signals.
        if X_semantic is not None:
            # Normalize semantic scores relative to themselves to match scale of PCA/PLS
            sem_norm = (X_semantic - np.mean(X_semantic, axis=0)) / (np.std(X_semantic, axis=0) + 1e-6)
            features.append(sem_norm)

        return np.hstack(features)
```

```python
COLUMNS = train_df.filter(like="emb").columns
TARGET_NAMES = ['Dry_Clover_g', 'Dry_Dead_g', 'Dry_Green_g', 'Dry_Total_g', 'GDM_g']
weights = {
    'Dry_Green_g': 0.1,
    'Dry_Dead_g': 0.1,
    'Dry_Clover_g': 0.1,
    'GDM_g': 0.2,
    'Dry_Total_g': 0.5,
}

TARGET_MAX = {
    "Dry_Clover_g": 71.7865,
    "Dry_Dead_g": 83.8407,
    "Dry_Green_g": 157.9836,
    "Dry_Total_g": 185.70,
    "GDM_g": 157.9836,
}

def competition_metric(y_true, y_pred) -> float:
    y_weighted = 0
    for l, label in enumerate(TARGET_NAMES):
        y_weighted = y_weighted + y_true[:, l].mean() * weights[label]

    ss_res = 0
    ss_tot = 0
    for l, label in enumerate(TARGET_NAMES):
        ss_res = ss_res + ((y_true[:, l] - y_pred[:, l])**2).mean() * weights[label]
        ss_tot = ss_tot + ((y_true[:, l] - y_weighted)**2).mean() * weights[label]

    return 1 - ss_res / ss_tot
```

```python
def post_process_biomass(df_preds):
    """
    Enforces physical mass balance constraints on biomass predictions.
    
    Constraints enforced:
    1. Dry_Green_g + Dry_Clover_g = GDM_g
    2. GDM_g + Dry_Dead_g = Dry_Total_g
    
    Method:
    Uses Orthogonal Projection. It finds the set of values that satisfy
    the constraints while minimizing the Euclidean distance to the 
    original model predictions.
    
    Args:
        df_preds (pd.DataFrame): DataFrame containing the 5 prediction columns.
        
    Returns:
        pd.DataFrame: A new DataFrame with consistent, non-negative values.
    """
    # 1. Define the specific order required for the math
    # We treat the vector x as: [Green, Clover, Dead, GDM, Total]
    ordered_cols = [
        "Dry_Green_g", 
        "Dry_Clover_g", 
        "Dry_Dead_g", 
        "GDM_g", 
        "Dry_Total_g"
    ]
    
    # Check if columns exist
    if not all(col in df_preds.columns for col in ordered_cols):
        missing = [c for c in ordered_cols if c not in df_preds.columns]
        raise ValueError(f"Input DataFrame is missing columns: {missing}")

    # 2. Extract values in the specific order -> Shape (N_samples, 5)
    Y = df_preds[ordered_cols].values.T  # Transpose to (5, N) for matrix math

    # 3. Define the Constraint Matrix C
    # We want Cx = 0
    # Eq 1: 1*Green + 1*Clover + 0*Dead - 1*GDM + 0*Total = 0
    # Eq 2: 0*Green + 0*Clover + 1*Dead + 1*GDM - 1*Total = 0
    C = np.array([
        [1, 1, 0, -1,  0],
        [0, 0, 1,  1, -1]
    ])

    # 4. Calculate Projection Matrix P
    # P = I - C^T * (C * C^T)^-1 * C
    # This projects any vector onto the null space of C (the valid subspace)
    C_T = C.T
    inv_CCt = np.linalg.inv(C @ C_T)
    P = np.eye(5) - C_T @ inv_CCt @ C

    # 5. Apply Projection
    # Y_new = P * Y
    Y_reconciled = P @ Y

    # 6. Transpose back to (N, 5)
    Y_reconciled = Y_reconciled.T

    # 7. Post-correction for negatives
    # Projection can mathematically create negative values (e.g. if Total was predicted 0)
    # We clip to 0. Note: This might slightly break the sum equality again, 
    # but exact equality with negatives is physically impossible anyway.
    Y_reconciled = Y_reconciled.clip(min=0)

    # 8. Create Output DataFrame
    df_out = df_preds.copy()
    df_out[ordered_cols] = Y_reconciled

    return df_out

# def post_process_biomass(df_preds):
#     """
#     Enforces mass balance constraints hierarchically.
    
#     Philosophy: 
#     - GDM_g is trusted. Green/Clover are scaled to match it.
#     - Dry_Total_g is trusted. Dead is derived as (Total - GDM).
#     - If constraints are physically impossible (e.g. GDM > Total),
#       we assume Total was underestimated and raise it to match GDM.
      
#     Args:
#         df_preds (pd.DataFrame): Predictions
        
#     Returns:
#         pd.DataFrame: Consistently processed dataframe.
#     """
#     # Create a copy to avoid SettingWithCopy warnings
#     df_out = df_preds.copy()
    
#     # ---------------------------------------------------------
#     # 1. Enforce: Dry_Green_g + Dry_Clover_g = GDM_g
#     # ---------------------------------------------------------
#     # We trust the *magnitude* of GDM_g more than the components.
#     # We trust the *ratio* of Green vs Clover from the model.
    
#     # Calculate current component sum
#     comp_sum = df_out["Dry_Green_g"] + df_out["Dry_Clover_g"]
    
#     # Avoid division by zero
#     mask_nonzero = comp_sum > 1e-9
    
#     # Calculate scaling factor so components sum exactly to GDM
#     scale_factor = df_out.loc[mask_nonzero, "GDM_g"] / comp_sum[mask_nonzero]
    
#     # Apply scaling
#     df_out.loc[mask_nonzero, "Dry_Green_g"] *= scale_factor
#     df_out.loc[mask_nonzero, "Dry_Clover_g"] *= scale_factor
    
#     # Edge case: If comp_sum is 0 but GDM is not, we can't scale.
#     # (Optional: You could split GDM evenly, but usually the model predicts 0 GDM here too)
    
#     # ---------------------------------------------------------
#     # 2. Enforce: GDM_g + Dry_Dead_g = Dry_Total_g
#     # ---------------------------------------------------------
#     # You stated Dead is hard to predict. 
#     # Therefore, we discard the direct prediction of Dead and derive it.
    
#     df_out["Dry_Dead_g"] = df_out["Dry_Total_g"] - df_out["GDM_g"]
    
#     # ---------------------------------------------------------
#     # 3. Handle Physical Impossibilities (Negative Dead)
#     # ---------------------------------------------------------
#     # If Dead < 0, it means GDM > Total. This is physically impossible.
#     # Logic: GDM is a sum of living parts (robust). Total is the scan of everything.
#     # If GDM > Total, the model likely underestimated Total.
    
#     neg_dead_mask = df_out["Dry_Dead_g"] < 0
    
#     if neg_dead_mask.any():
#         # Set Dead to 0 (cannot be negative)
#         df_out.loc[neg_dead_mask, "Dry_Dead_g"] = 0
        
#         # Raise Total to match GDM (maintaining balance)
#         df_out.loc[neg_dead_mask, "Dry_Total_g"] = df_out.loc[neg_dead_mask, "GDM_g"]

#     return df_out

def compare_results(oof, train_data):
    y_oof_df = pd.DataFrame(oof, columns=TARGET_NAMES) # ensure columns match
    # 2. Check Score BEFORE Processing
    raw_score = competition_metric(train_data[TARGET_NAMES].values, y_oof_df.values)
    print(f"Raw CV Score: {raw_score:.6f}")
    
    # 3. Apply Post-Processing
    y_oof_proc = post_process_biomass(y_oof_df)
    
    # 4. Check Score AFTER Processing
    proc_score = competition_metric(train_data[TARGET_NAMES].values, y_oof_proc.values)
    print(f"Processed CV Score: {proc_score:.6f}")
    
    print(f"Improvement: {raw_score - proc_score:.6f}")
```

```python
# train_df['fold'].nunique()
```

```python
# from sklearn.preprocessing import PowerTransformer, QuantileTransformer, RobustScaler
# from sklearn.svm import SVR

# def cross_validate(model, train_data, test_data, feature_engine, target_transform='max', seed=42):
#     """
#     target_transform options:
#     - 'max': Linear scaling by max value (Preserves distribution shape)
#     - 'log': np.log1p (Aggressive compression of outliers)
#     - 'sqrt': np.sqrt (Moderate compression, good for biological counts/area)
#     - 'yeo-johnson': PowerTransformer (Makes data Gaussian-like automatically)
#     - 'quantile': QuantileTransformer (Forces strict Normal distribution)
#     """

#     n_splits = train_data['fold'].nunique()
#     target_max_arr = np.array([TARGET_MAX[t] for t in TARGET_NAMES], dtype=float)
#     y_true = train_data[TARGET_NAMES]
    
#     y_pred = pd.DataFrame(0.0, index=train_data.index, columns=TARGET_NAMES)
#     y_pred_test = np.zeros([len(test_data), len(TARGET_NAMES)], dtype=float)

#     for fold in range(n_splits):
#         seeding(seed*(seed//2 + fold))
#         # 1. Split Data
#         train_mask = train_data['fold'] != fold
#         valid_mask = train_data['fold'] == fold
#         val_idx = train_data[valid_mask].index

#         X_train_raw = train_data[train_mask][COLUMNS].values
#         X_valid_raw = train_data[valid_mask][COLUMNS].values
#         X_test_raw = test_data[COLUMNS].values
        
#         y_train = train_data[train_mask][TARGET_NAMES].values
#         y_valid = train_data[valid_mask][TARGET_NAMES].values

#         # ===========================
#         # 2) TARGET TRANSFORMATION
#         # ===========================
#         transformer = None # To store stateful transformers (Yeo/Quantile)
        
#         if target_transform == 'log':
#             y_train_proc = np.log1p(y_train)
            
#         elif target_transform == 'max':
#             y_train_proc = y_train / target_max_arr
            
#         elif target_transform == 'sqrt':
#             # Great for biomass/area data (Variance stabilizing)
#             y_train_proc = np.sqrt(y_train)
            
#         elif target_transform == 'yeo-johnson':
#             # Learns optimal parameter to make data Gaussian
#             transformer = PowerTransformer(method='yeo-johnson', standardize=True)
#             y_train_proc = transformer.fit_transform(y_train)
            
#         elif target_transform == 'quantile':
#             # Forces data into a normal distribution (Robust to outliers)
#             # transformer = QuantileTransformer(output_distribution='uniform', n_quantiles=64, random_state=42)
#             transformer = RobustScaler()
#             y_train_proc = transformer.fit_transform(y_train)
            
#         else:
#             y_train_proc = y_train

#         # ==========================================
#         # 3) FEATURE ENGINEERING
#         # ==========================================
#         engine = deepcopy(feature_engine)
#         # Note: If your engine uses PLS, pass the transformed y!
#         engine.fit(X_train_raw, y=y_train_proc) 
        
#         x_train_eng = engine.transform(X_train_raw)
#         x_valid_eng = engine.transform(X_valid_raw)
#         x_test_eng = engine.transform(X_test_raw)
        
#         # ==========================================
#         # 4) TRAIN & PREDICT
#         # ==========================================
#         fold_valid_pred = np.zeros_like(y_valid)
#         fold_test_pred = np.zeros([len(test_data), len(TARGET_NAMES)])

#         for k in range(len(TARGET_NAMES)):
#             regr = deepcopy(model)
#             regr.fit(x_train_eng, y_train_proc[:, k])
            
#             # Raw Predictions (in transformed space)
#             pred_valid_raw = regr.predict(x_valid_eng)
#             pred_test_raw = regr.predict(x_test_eng)
            
#             # Store raw for inverse transform block below
#             fold_valid_pred[:, k] = pred_valid_raw
#             fold_test_pred[:, k] = pred_test_raw

#         # ===========================
#         # 5) INVERSE TRANSFORM (Apply to full matrix)
#         # ===========================
#         if target_transform == 'log':
#             fold_valid_pred = np.expm1(fold_valid_pred)
#             fold_test_pred = np.expm1(fold_test_pred)
            
#         elif target_transform == 'max':
#             fold_valid_pred = fold_valid_pred * target_max_arr
#             fold_test_pred = fold_test_pred * target_max_arr
            
#         elif target_transform == 'sqrt':
#             # Inverse of sqrt is square
#             fold_valid_pred = np.square(fold_valid_pred)
#             fold_test_pred = np.square(fold_test_pred)
            
#         elif target_transform in ['yeo-johnson', 'quantile']:
#             # Use the fitted transformer to invert
#             fold_valid_pred = transformer.inverse_transform(fold_valid_pred)
#             fold_test_pred = transformer.inverse_transform(fold_test_pred)

#         # # Final Clip (Biomass cannot be negative)
#         # fold_valid_pred = fold_valid_pred.clip(min=0)
#         # fold_test_pred = fold_test_pred.clip(min=0)

#         # Store results
#         y_pred.loc[val_idx] = fold_valid_pred
#         y_pred_test += fold_test_pred / n_splits
        
#         if fold == 0:
#             print(f"  [Fold 0] Target: {target_transform}, Feats: {x_train_eng.shape}")

#     full_cv = competition_metric(y_true.values, y_pred.values)
#     print(f"Full CV Score: {full_cv:.6f}")
    
#     return y_pred.values, y_pred_test

# # Initialize
# seed = 42
# feat_engine = SupervisedEmbeddingEngine(
#     n_pca=0.80,
#     n_pls=10,             # Extract 8 strong supervised signals
#     n_gmm=3,             # 6 Soft clusters
#     random_state=seed
# )

# # print("######## Ridge Regression #######")
# # oof_ridge, pred_test_ri = cross_validate(Ridge(), train_siglip_df, test_siglip_df, feature_engine=feat_engine)
# # compare_results(oof_ridge, train_siglip_df)

# # print("####### Lasso Regression #######")
# # oof_la, pred_test_la = cross_validate(Lasso(), train_siglip_df, test_siglip_df, feature_engine=feat_engine)
# # compare_results(oof_la, train_siglip_df)

# print("\n###### GradientBoosting Regressor #######")
# oof_gb, pred_test_gb = cross_validate(
#     GradientBoostingRegressor(random_state=seed), 
#     train_siglip_df, test_siglip_df, 
#     feature_engine=feat_engine, 
#     target_transform='max')
# compare_results(oof_gb, train_siglip_df)

# print("\n###### Hist Gradient Boosting Regressor ######")
# oof_hb, pred_test_hb = cross_validate(
#     HistGradientBoostingRegressor(random_state=seed), 
#     train_siglip_df, test_siglip_df, 
#     feature_engine=feat_engine, 
#     target_transform='max')
# compare_results(oof_hb, train_siglip_df)

# print("\n##### CAT Regressor ######")
# oof_cat, pred_test_cat = cross_validate(
#     CatBoostRegressor(verbose=0, random_seed=seed), 
#     train_siglip_df, test_siglip_df, 
#     feature_engine=feat_engine
# )
# compare_results(oof_cat, train_siglip_df)

# print("\n######## XGB #######")
# oof_xgb, pred_test_xgb = cross_validate(
#     XGBRegressor(verbosity=0), 
#     train_siglip_df, test_siglip_df, 
#     feature_engine=feat_engine, 
#     target_transform='max')
# compare_results(oof_xgb, train_siglip_df)

# print("\n######## LGBM #######")
# oof_lgbm, pred_test_lgbm = cross_validate(
#     LGBMRegressor(verbose=-1, random_state=seed), 
#     train_siglip_df, test_siglip_df, 
#     feature_engine=feat_engine, 
#     target_transform='max')
# compare_results(oof_lgbm, train_siglip_df)
```

## Semantic probing

```python
# --- STEP 1: Generate Semantic Features ---
# We combine Train and Test to generate features in one go, then split them back.
# This ensures the text-projections are consistent.

# Concatenate embeddings
X_all_emb = np.vstack([
    train_siglip_df[COLUMNS].values, 
    test_siglip_df[COLUMNS].values
])

# Generate Semantic Probes (Using the function defined in Part 1)
# Make sure SIGLIP_PATH is correct for your environment
print("Generating Semantic Features via SigLIP Text Encoder...")
try:
    all_semantic_scores = generate_semantic_features(X_all_emb, model_path=siglip_path)
    
    # Split back into Train and Test
    n_train = len(train_siglip_df)
    sem_train_full = all_semantic_scores[:n_train]
    sem_test_full = all_semantic_scores[n_train:]
    print(f"Semantic Features Generated. Train: {sem_train_full.shape}, Test: {sem_test_full.shape}")
    
except Exception as e:
    print(f"Skipping Semantic Features due to error: {e}")
    # Fallback to None if model path is wrong or memory fails
    sem_train_full = None
    sem_test_full = None
```

```python
# --- STEP 2: Updated Cross-Validation Function ---
def cross_validate(model, train_data, test_data, feature_engine, 
                   semantic_train=None, semantic_test=None, # <--- NEW ARGS
                   target_transform='max', seed=42):

    n_splits = train_data['fold'].nunique()
    # Setup Targets
    target_max_arr = np.array([TARGET_MAX[t] for t in TARGET_NAMES], dtype=float)
    y_true = train_data[TARGET_NAMES]
    
    # Setup Storage
    y_pred = pd.DataFrame(0.0, index=train_data.index, columns=TARGET_NAMES)
    y_pred_test = np.zeros([len(test_data), len(TARGET_NAMES)], dtype=float)

    for fold in range(n_splits):
        seeding(seed*(seed//2 + fold))
        # Create masks
        train_mask = train_data['fold'] != fold
        valid_mask = train_data['fold'] == fold
        val_idx = train_data[valid_mask].index

        # Raw Inputs (Embeddings)
        X_train_raw = train_data[train_mask][COLUMNS].values
        X_valid_raw = train_data[valid_mask][COLUMNS].values
        X_test_raw = test_data[COLUMNS].values
        
        # Semantic Inputs (Slicing)
        # We handle the case where semantic features might be None
        sem_train_fold = semantic_train[train_mask] if semantic_train is not None else None
        sem_valid_fold = semantic_train[valid_mask] if semantic_train is not None else None
        
        # Raw Targets
        y_train = train_data[train_mask][TARGET_NAMES].values
        y_valid = train_data[valid_mask][TARGET_NAMES].values

        # ===========================
        # 1) TRANSFORM TARGETS
        # ===========================
        if target_transform == 'log':
            y_train_proc = np.log1p(y_train)
        elif target_transform == 'max':
            y_train_proc = y_train / target_max_arr
        else:
            y_train_proc = y_train

        # ==========================================
        # 2) FEATURE ENGINEERING
        # ==========================================
        engine = deepcopy(feature_engine)
        
        # FIT: Now passes y (for PLS/RFE) and Semantic Features
        engine.fit(X_train_raw, y=y_train_proc, X_semantic=sem_train_fold)
        
        # TRANSFORM: Pass Semantic Features
        x_train_eng = engine.transform(X_train_raw, X_semantic=sem_train_fold)
        x_valid_eng = engine.transform(X_valid_raw, X_semantic=sem_valid_fold)
        # For test, we use the full test semantic set
        x_test_eng = engine.transform(X_test_raw, X_semantic=semantic_test)
        
        # ==========================================
        # 3) TRAIN & PREDICT
        # ==========================================
        fold_valid_pred = np.zeros_like(y_valid)
        fold_test_pred = np.zeros([len(test_data), len(TARGET_NAMES)])

        for k in range(len(TARGET_NAMES)):
            regr = deepcopy(model)
            
            # Fit model
            regr.fit(x_train_eng, y_train_proc[:, k])
            
            # Predict
            pred_valid_raw = regr.predict(x_valid_eng)
            pred_test_raw = regr.predict(x_test_eng)
            
            # ===========================
            # 4) INVERSE TRANSFORM
            # ===========================
            if target_transform == 'log':
                pred_valid_inv = np.expm1(pred_valid_raw)
                pred_test_inv = np.expm1(pred_test_raw)
            elif target_transform == 'max':
                pred_valid_inv = (pred_valid_raw * target_max_arr[k])
                pred_test_inv = (pred_test_raw * target_max_arr[k])
            else:
                pred_valid_inv = pred_valid_raw
                pred_test_inv = pred_test_raw

            fold_valid_pred[:, k] = pred_valid_inv
            fold_test_pred[:, k] = pred_test_inv

        # Store results
        y_pred.loc[val_idx] = fold_valid_pred
        y_pred_test += fold_test_pred / n_splits
        
        if fold == 0:
            print(f"  [Fold 0 Info] Target: {target_transform}, Feats: {x_train_eng.shape}")

    full_cv = competition_metric(y_true.values, y_pred.values)
    print(f"Full CV Score: {full_cv:.6f}")
    
    return y_pred.values, y_pred_test

# --- STEP 3: Run Models ---

# Initialize the NEW Supervised Engine
feat_engine = SupervisedEmbeddingEngine(
    n_pca=0.80,
    n_pls=8,             # Supervised signals
    n_gmm=6,             # Soft clusters
)

# print("######## Ridge Regression #######")
# oof_ridge, pred_test_ri = cross_validate(
#     Ridge(), 
#     train_siglip_df, test_siglip_df, 
#     feature_engine=feat_engine,
#     semantic_train=sem_train_full, # <--- Pass Semantic
#     semantic_test=sem_test_full    # <--- Pass Semantic
# )
# compare_results(oof_ridge, train_siglip_df)

# print("\n####### Lasso Regression #######")
# # Lasso should perform much better now due to PLS and RFE
# oof_la, pred_test_la = cross_validate(
#     Lasso(alpha=0.015), # Small alpha for normalized feats
#     train_siglip_df, test_siglip_df, 
#     feature_engine=feat_engine,
#     semantic_train=sem_train_full,
#     semantic_test=sem_test_full
# )
# compare_results(oof_la, train_siglip_df)

print("\n###### GradientBoosting Regressor #######")
oof_gb, pred_test_gb = cross_validate(
    GradientBoostingRegressor(), 
    train_siglip_df, test_siglip_df, 
    feature_engine=feat_engine,
    semantic_train=sem_train_full,
    semantic_test=sem_test_full
)
compare_results(oof_gb, train_siglip_df)

print("\n###### Hist Gradient Boosting Regressor ######")
oof_hb, pred_test_hb = cross_validate(
    HistGradientBoostingRegressor(), 
    train_siglip_df, test_siglip_df, 
    feature_engine=feat_engine,
    semantic_train=sem_train_full,
    semantic_test=sem_test_full
)
compare_results(oof_hb, train_siglip_df)

print("\n##### CAT Regressor ######")
oof_cat, pred_test_cat = cross_validate(
    CatBoostRegressor(verbose=0), 
    train_siglip_df, test_siglip_df, 
    feature_engine=feat_engine,
    semantic_train=sem_train_full,
    semantic_test=sem_test_full
)
compare_results(oof_cat, train_siglip_df)

# print("\n######## XGB #######")
# oof_xgb, pred_test_xgb = cross_validate(
#     XGBRegressor(verbosity=0), 
#     train_siglip_df, test_siglip_df, 
#     feature_engine=feat_engine, 
#     semantic_train=sem_train_full,
#     semantic_test=sem_test_full,
#     target_transform='max')
# compare_results(oof_xgb, train_siglip_df)

print("\n######## LGBM #######")
oof_lgbm, pred_test_lgbm = cross_validate(
    LGBMRegressor(verbose=-1), 
    train_siglip_df, test_siglip_df, 
    feature_engine=feat_engine, 
    semantic_train=sem_train_full,
    semantic_test=sem_test_full,
    target_transform='max')
compare_results(oof_lgbm, train_siglip_df)
```

## Hyperparameter tuning

```python
# from sklearn.model_selection import RandomizedSearchCV, KFold
# from scipy.stats import uniform, randint

# # ==========================================
# # 1. PARAMETER GRIDS
# # ==========================================

# # HistGradientBoostingRegressor Hyperparameters
# hist_gb_params = {
#     'learning_rate': uniform(0.01, 0.2),      # Continuous distribution
#     'max_iter': [100, 300, 500, 1000],        # Trees
#     'max_leaf_nodes': randint(15, 63),        # Complexity
#     'min_samples_leaf': randint(10, 50),      # Regularization
#     'l2_regularization': uniform(0, 5),       # L2 Reg
#     'max_depth': [None, 5, 10, 15]            # Depth constraint
# }

# # GradientBoostingRegressor Hyperparameters
# # (Standard GBR is slower, so we use slightly smaller ranges)
# gb_params = {
#     'learning_rate': uniform(0.01, 0.2),
#     'n_estimators': [100, 300, 500],
#     'subsample': uniform(0.6, 0.4),           # 0.6 to 1.0
#     'max_depth': randint(3, 8),
#     'min_samples_split': randint(2, 20),
#     'min_samples_leaf': randint(1, 10)
# }

# def tune_on_fold_zero(model_class, param_dist, train_data, feature_engine, 
#                       semantic_train=None, target_transform='max', 
#                       n_iter=20, seed=42):
    
#     print(f"--- Tuning {model_class.__name__} on Fold 0 ---")
    
#     # 1. Extract Fold 0 (Mimicking cross_validate logic)
#     fold = 0
#     train_mask = train_data['fold'] != fold
    
#     # Raw Inputs
#     X_train_raw = train_data[train_mask][COLUMNS].values
    
#     # Semantic Inputs
#     sem_train_fold = semantic_train[train_mask] if semantic_train is not None else None
    
#     # Targets
#     y_train = train_data[train_mask][TARGET_NAMES].values
    
#     # 2. Transform Targets
#     target_max_arr = np.array([TARGET_MAX[t] for t in TARGET_NAMES], dtype=float)
#     if target_transform == 'log':
#         y_train_proc = np.log1p(y_train)
#     elif target_transform == 'max':
#         y_train_proc = y_train / target_max_arr
#     else:
#         y_train_proc = y_train

#     # 3. Feature Engineering (Fit on Fold 0 Train)
#     print("Fitting Feature Engine on Fold 0...")
#     engine = deepcopy(feature_engine)
#     engine.fit(X_train_raw, y=y_train_proc, X_semantic=sem_train_fold)
#     x_train_eng = engine.transform(X_train_raw, X_semantic=sem_train_fold)
    
#     print(f"Features ready: {x_train_eng.shape}. Starting SearchCV per target...")

#     # 4. Tune per Target
#     best_params_per_target = {}
    
#     for k, target_name in enumerate(TARGET_NAMES):
#         print(f"  > Tuning Target: {target_name} ({k+1}/{len(TARGET_NAMES)})")
        
#         # Initialize Base Model
#         base_model = model_class(random_state=seed)
        
#         # Setup Randomized Search
#         # cv=3 is sufficient for tuning to prevent overfitting
#         search = RandomizedSearchCV(
#             estimator=base_model,
#             param_distributions=param_dist,
#             n_iter=n_iter,
#             scoring='neg_mean_squared_error',
#             cv=3, 
#             n_jobs=-1,
#             random_state=seed,
#             verbose=0
#         )
        
#         # Fit on the processed features
#         search.fit(x_train_eng, y_train_proc[:, k])
        
#         best_params_per_target[target_name] = search.best_params_
#         print(f"    Best Score (MSE): {-search.best_score_:.5f}")
#         print(f"    Params: {search.best_params_}")

#     return best_params_per_target

# # Initialize the NEW Supervised Engine
# feat_engine = SupervisedEmbeddingEngine(
#     n_pca=0.80,
#     n_pls=8,             # Supervised signals
#     n_gmm=6,             # Soft clusters
# )

# # --- TUNE HIST GRADIENT BOOSTING ---
# hist_best_params = tune_on_fold_zero(
#     model_class=HistGradientBoostingRegressor,
#     param_dist=hist_gb_params,
#     train_data=train_siglip_df,
#     feature_engine=feat_engine,
#     semantic_train=sem_train_full,
#     target_transform='max',
#     n_iter=15  # Adjust based on your time constraints
# )

# print("\n\n====== RECOMENDED HIST GB PARAMS (Averaged or First Target) ======")
# # Often it's better to pick one set of stable params for all targets 
# # unless variance is huge. Here we look at the first target's result as a proxy:
# print(hist_best_params[TARGET_NAMES[0]])


# # --- TUNE STANDARD GRADIENT BOOSTING ---
# gb_best_params = tune_on_fold_zero(
#     model_class=GradientBoostingRegressor,
#     param_dist=gb_params,
#     train_data=train_siglip_df,
#     feature_engine=feat_engine,
#     semantic_train=sem_train_full,
#     target_transform='max',
#     n_iter=10
# )

# print("\n\n====== RECOMENDED GB PARAMS ======")
# print(gb_best_params[TARGET_NAMES[0]])
```

```python
# import optuna
# import numpy as np
# import pandas as pd
# from sklearn.base import clone
# from sklearn.decomposition import PCA
# from sklearn.multioutput import MultiOutputRegressor
# from xgboost import XGBRegressor
# from lightgbm import LGBMRegressor
# from catboost import CatBoostRegressor

# # ---------------------------------------------------------
# # 1. Corrected Helper Function
# # ---------------------------------------------------------
# feat_engine = EmbeddingFeatureEngine(
#     n_pca_components=0.90, 
#     n_clusters=25, 
#     use_stats=True, 
#     use_similarity=True,
#     use_anomaly=True,        # Adds Anomaly Score
#     use_entropy=True,        # Adds Entropy
#     use_pca_interactions=True # Adds Poly features on Top 5 PCA
# )

# def get_cv_score(model, train_data, feature_engine, target_transform='max', random_state=42):
#     """
#     Runs CV on ALL folds dynamically to return a single score.
#     Optimized for speed (vectorized target processing).
    
#     Args:
#         model: Estimator (must support Multi-Output or be wrapped in MultiOutputRegressor)
#         train_data: DataFrame containing 'fold' column
#         feature_engine: Transformer with .fit() and .transform()
#         target_transform: 'log', 'max', or None
#     """
#     # 1. Setup global constants
#     target_max_arr = np.array([TARGET_MAX[t] for t in TARGET_NAMES], dtype=float)
#     y_true = train_data[TARGET_NAMES].values
#     y_pred = np.zeros([len(train_data), len(TARGET_NAMES)], dtype=float)
    
#     # 2. Detect Folds dynamically
#     folds = sorted(train_data['fold'].unique())
    
#     # 3. Loop over folds
#     for fold in folds:
#         # -------------------------
#         # Data Slicing
#         # -------------------------
#         train_mask = train_data['fold'] != fold
#         valid_mask = train_data['fold'] == fold
#         val_idx = train_data[valid_mask].index

#         X_train_raw = train_data.loc[train_mask, COLUMNS].values
#         X_valid_raw = train_data.loc[valid_mask, COLUMNS].values
        
#         y_train = train_data.loc[train_mask, TARGET_NAMES].values
#         # y_valid used implicitely via y_true at the end

#         # -------------------------
#         # A) Transform Targets
#         # -------------------------
#         if target_transform == 'log':
#             y_train_proc = np.log1p(y_train)
#         elif target_transform == 'max':
#             y_train_proc = y_train / target_max_arr
#         else:
#             y_train_proc = y_train

#         # -------------------------
#         # B) Feature Engineering
#         # -------------------------
#         # Fit engine only on training split
#         engine = deepcopy(feature_engine)
#         engine.fit(X_train_raw)
        
#         X_train_eng = engine.transform(X_train_raw)
#         X_valid_eng = engine.transform(X_valid_raw)

#         # -------------------------
#         # C) Fit Model (Multi-Output)
#         # -------------------------
#         regr = clone(model)
#         regr.fit(X_train_eng, y_train_proc)

#         # -------------------------
#         # D) Predict & Inverse Transform
#         # -------------------------
#         valid_pred_raw = np.array(regr.predict(X_valid_eng))
        
#         if target_transform == 'log':
#             valid_pred = np.expm1(valid_pred_raw)
#         elif target_transform == 'max':
#             valid_pred = valid_pred_raw * target_max_arr
#         else:
#             valid_pred = valid_pred_raw

#         # Clip and Store
#         y_pred[val_idx] = valid_pred.clip(0)

#     # 4. Calculate Metric
#     score = competition_metric(y_true, y_pred)
    
#     # # Clean output buffer if running in a loop
#     # try:
#     #     from IPython.display import flush_ipython
#     #     flush_ipython()
#     # except ImportError:
#     #     pass
        
#     return score

# # ---------------------------------------------------------
# # 2. Corrected CatBoost Objective
# # ---------------------------------------------------------
# def objective_catboost(trial):
#     params = {
#         # Search Space
#         'iterations': trial.suggest_int('iterations', 800, 2000), # Increased min iterations
#         'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
#         'depth': trial.suggest_int('depth', 4, 8), # Reduced max depth to save memory
#         'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-3, 10.0, log=True),
#         'random_strength': trial.suggest_float('random_strength', 1e-3, 5.0, log=True),
#         'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
        
#         # Fixed GPU Params
#         'loss_function': 'MultiRMSE',
#         'task_type': 'GPU',
#         'boosting_type': 'Plain', 
#         'devices': '0',
#         'verbose': 0,
#         'random_state': 42,
#         'allow_writing_files': False # Prevents creating log files
#     }
    
#     model = CatBoostRegressor(**params)
    
#     # Removed n_splits argument; it now uses whatever is in 'train'
#     return get_cv_score(model, train_siglip_df, feature_engine=feat_engine)

# # ---------------------------------------------------------
# # 3. Corrected XGBoost Objective
# # ---------------------------------------------------------
# def objective_xgboost(trial):
#     params = {
#         'n_estimators': trial.suggest_int('n_estimators', 800, 2000),
#         'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
#         'max_depth': trial.suggest_int('max_depth', 3, 8),
#         'subsample': trial.suggest_float('subsample', 0.6, 1.0),
#         'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
#         'reg_alpha': trial.suggest_float('reg_alpha', 0.1, 10.0, log=True),
#         'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 10.0, log=True),
        
#         # Fixed
#         'tree_method': 'hist',
#         'device': 'cuda',
#         'n_jobs': -1,
#         'random_state': 42,
#         'verbosity': 0
#     }
    
#     model = MultiOutputRegressor(XGBRegressor(**params))
#     return get_cv_score(model, train_siglip_df, feature_engine=feat_engine)

# # ---------------------------------------------------------
# # 4. Corrected LightGBM Objective
# # ---------------------------------------------------------
# def objective_lgbm(trial):
#     params = {
#         'n_estimators': trial.suggest_int('n_estimators', 800, 2000),
#         'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
#         'num_leaves': trial.suggest_int('num_leaves', 20, 100),
#         'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
#         'subsample': trial.suggest_float('subsample', 0.6, 1.0),
#         'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
#         'reg_alpha': trial.suggest_float('reg_alpha', 0.1, 10.0, log=True),
#         'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 10.0, log=True),
        
#         # Fixed
#         'device': 'gpu',
#         'n_jobs': -1,
#         'random_state': 42,
#         'verbose': -1
#     }
    
#     model = MultiOutputRegressor(LGBMRegressor(**params))
#     return get_cv_score(model, train_siglip_df, feature_engine=feat_engine)
```

```python
# # # --- 1. Tune CatBoost (Highest Priority) ---
# # print("Tuning CatBoost...")
# # study_cat = optuna.create_study(direction='maximize')
# # study_cat.optimize(objective_catboost, n_trials=20)

# # print("Best CatBoost Params:", study_cat.best_params)
# # best_cat_params = study_cat.best_params
# # # Re-add fixed params that Optuna didn't tune
# # best_cat_params.update({
# #     'loss_function': 'MultiRMSE', 
# #     # 'task_type': 'GPU', 
# #     'boosting_type': 'Plain', 
# #     # 'devices': '0', 
# #     'verbose': 0, 
# #     'random_state': 42
# # })


# # # --- 2. Tune XGBoost ---
# # print("\nTuning XGBoost...")
# # study_xgb = optuna.create_study(direction='maximize')
# # study_xgb.optimize(objective_xgboost, n_trials=20)

# # print("Best XGBoost Params:", study_xgb.best_params)
# # # best_xgb_params = study_xgb.best_params
# # # best_xgb_params.update({
# # #     'tree_method': 'hist', 
# # #     'device': 'cuda', 
# # #     'n_jobs': -1, 
# # #     'random_state': 42
# # # })


# # --- 3. Tune LightGBM ---
# print("\nTuning LightGBM...")
# study_lgbm = optuna.create_study(direction='maximize')
# study_lgbm.optimize(objective_lgbm, n_trials=20)

# print("Best LightGBM Params:", study_lgbm.best_params)
# # best_lgbm_params = study_lgbm.best_params
# # best_lgbm_params.update({
# #     'device': 'gpu', 
# #     'n_jobs': -1, 
# #     'random_state': 42, 
# #     'verbose': -1
# # })
```

### Parameters

### Siglip (PCA n_components=0.8)

```
Best CatBoost Params: {'iterations': 1900, 'learning_rate': 0.04488950669764926, 'depth': 4, 'l2_leaf_reg': 0.5647720146150716, 'random_strength': 0.04455012279134044, 'bagging_temperature': 0.9810426313146956}

Best XGBoost Params: {'n_estimators': 1354, 'learning_rate': 0.010266591943008255, 'max_depth': 3, 'subsample': 0.6035540714532827, 'colsample_bytree': 0.9029950550994382, 'reg_alpha': 0.11110779086383878, 'reg_lambda': 9.314996597001533}

Best LightGBM Params: {'n_estimators': 807, 'learning_rate': 0.014069585873331451, 'num_leaves': 48, 'min_child_samples': 19, 'subsample': 0.7451600778259232, 'colsample_bytree': 0.7457374193240649, 'reg_alpha': 0.21580623254415052, 'reg_lambda': 3.784221570035411}
```

### Siglip (NO PCA)

```
Best CatBoost Params: {'iterations': 1945, 'learning_rate': 0.025612792183534742, 'depth': 5, 'l2_leaf_reg': 0.0011451976652037553, 'random_strength': 0.03363171953662423, 'bagging_temperature': 0.9926373709983951}

Best XGBoost Params: {'n_estimators': 1695, 'learning_rate': 0.013048089867977527, 'max_depth': 4, 'subsample': 0.7151550925326732, 'colsample_bytree': 0.7883122143141527, 'reg_alpha': 0.6732457617935534, 'reg_lambda': 8.692842925053135}

Best LightGBM Params: {'n_estimators': 1983, 'learning_rate': 0.03365765614894731, 'num_leaves': 21, 'min_child_samples': 44, 'subsample': 0.9970297203974366, 'colsample_bytree': 0.9324460763054059, 'reg_alpha': 0.324803213211078, 'reg_lambda': 0.1601835613567248}
```

## Multioutput

```python
# import numpy as np
# import pandas as pd
# from copy import deepcopy
# from sklearn.base import clone
# from sklearn.decomposition import PCA
# from sklearn.multioutput import MultiOutputRegressor
# from sklearn.multioutput import RegressorChain

# # Import the specific libraries
# from xgboost import XGBRegressor
# from lightgbm import LGBMRegressor
# from catboost import CatBoostRegressor

# # ---------------------------------------------------------
# # 1. The Optimized Multi-Output GPU CV Function
# # ---------------------------------------------------------
# def cross_validate_multioutput_gpu(model, train_data, test_data, feature_engine, 
#                                    semantic_train=None, semantic_test=None, # <--- NEW ARGS
#                                    target_transform='max', seed=42, n_splits=5):
#     """
#     Performs Cross Validation using a Multi-Output strategy (Vectorized)
#     with support for Semantic Features and Supervised Embedding Engines.
#     """
    
#     # 1. Setup Target Max Array
#     target_max_arr = np.array([TARGET_MAX[t] for t in TARGET_NAMES], dtype=float)
#     y_true = train_data[TARGET_NAMES].values
    
#     # 2. Pre-allocate arrays
#     y_pred = np.zeros([len(train_data), len(TARGET_NAMES)], dtype=float)
#     y_pred_test = np.zeros([len(test_data), len(TARGET_NAMES)], dtype=float)

#     print(f"Starting CV with model: {model.__class__.__name__}")
#     print(f"Target Transform Strategy: {target_transform}")

#     # Ensure n_splits matches the fold column if present
#     if 'fold' in train_data.columns:
#         n_splits = train_data['fold'].nunique()

#     for fold in range(n_splits):
#         seeding(seed*(seed//2 + fold))
#         # ------------------------------
#         # Data Preparation
#         # ------------------------------
#         # Create masks
#         train_mask = train_data['fold'] != fold
#         valid_mask = train_data['fold'] == fold
#         val_idx = train_data[valid_mask].index

#         # Raw Inputs (Embeddings)
#         X_train_raw = train_data.loc[train_mask, COLUMNS].values
#         X_valid_raw = train_data.loc[valid_mask, COLUMNS].values
#         X_test_raw = test_data[COLUMNS].values
        
#         # Semantic Inputs (Slicing) - NEW LOGIC
#         # We handle the case where semantic features might be None
#         sem_train_fold = semantic_train[train_mask] if semantic_train is not None else None
#         sem_valid_fold = semantic_train[valid_mask] if semantic_train is not None else None
        
#         # Raw Targets
#         y_train = train_data.loc[train_mask, TARGET_NAMES].values
#         y_valid = train_data.loc[valid_mask, TARGET_NAMES].values

#         # ------------------------------
#         # 1) Transform Targets (Vectorized)
#         # ------------------------------
#         if target_transform == 'log':
#             y_train_proc = np.log1p(y_train)
#         elif target_transform == 'max':
#             y_train_proc = y_train / target_max_arr
#         else:
#             y_train_proc = y_train
        
#         # ------------------------------
#         # 2) Feature Engineering
#         # ------------------------------
#         engine = deepcopy(feature_engine)
        
#         # FIT: Pass X_train, Y_train (for PLS), and Semantic Train
#         engine.fit(X_train_raw, y=y_train_proc, X_semantic=sem_train_fold)
        
#         # TRANSFORM: Pass corresponding Semantic slices
#         x_train_eng = engine.transform(X_train_raw, X_semantic=sem_train_fold)
#         x_valid_eng = engine.transform(X_valid_raw, X_semantic=sem_valid_fold)
#         # For test, we use the full test semantic set
#         x_test_eng = engine.transform(X_test_raw, X_semantic=semantic_test)

#         # ------------------------------
#         # 3) Train (Multi-Output)
#         # ------------------------------
#         regr = clone(model) 
        
#         # Fit on (N_samples, N_targets)
#         # XGBoost and CatBoost (MultiRMSE) support this natively
#         regr.fit(x_train_eng, y_train_proc)

#         # ------------------------------
#         # 4) Predict & Unscale
#         # ------------------------------
#         valid_pred_raw = np.array(regr.predict(x_valid_eng))
#         test_pred_raw = np.array(regr.predict(x_test_eng))

#         # Inverse Transform
#         if target_transform == 'log':
#             valid_pred = np.expm1(valid_pred_raw)
#             test_pred = np.expm1(test_pred_raw)
#         elif target_transform == 'max':
#             valid_pred = valid_pred_raw * target_max_arr
#             test_pred = test_pred_raw * target_max_arr
#         else:
#             valid_pred = valid_pred_raw
#             test_pred = test_pred_raw

#         # Clip negative predictions
#         valid_pred = valid_pred.clip(0)
#         test_pred = test_pred.clip(0)

#         # Store OOF
#         y_pred[val_idx] = valid_pred
        
#         # Accumulate Test Preds
#         y_pred_test += test_pred / n_splits
            
#         if fold == 0:
#              print(f"  [Fold 0 Debug] Transformed Train Shape: {x_train_eng.shape}")

#     # Global CV score
#     try:
#         full_cv = competition_metric(y_true, y_pred)
#         print(f"Full CV Score: {full_cv:.6f}")
#     except NameError:
#         print("Done (metric function not found)")

#     return y_pred, y_pred_test

# feat_engine = SupervisedEmbeddingEngine(
#     n_pca=0.80,
#     n_pls=8,             # Supervised signals
#     n_gmm=6,             # Soft clusters
# )

# # ---------------------------------------------------------
# # 2. Model Definitions
# # ---------------------------------------------------------
# best_cat_params = {
#     'iterations': 1783, 
#     'learning_rate': 0.0633221588945314, 
#     'depth': 4, 
#     'l2_leaf_reg': 0.1312214556803292, 
#     'random_strength': 0.04403178418151252, 
#     'bagging_temperature': 0.9555074383215754
# }
# best_cat_params.update({
#     'loss_function': 'MultiRMSE', 
#     # 'task_type': 'GPU', 
#     'boosting_type': 'Plain', 
#     # 'devices': '0', 
#     'verbose': 0, 
#     'random_state': 42
# })

# best_xgb_params = {
#     'n_estimators': 1501, 
#     'learning_rate': 0.024461148923117938, 
#     'max_depth': 3, 
#     'subsample': 0.6905614627569726, 
#     'colsample_bytree': 0.895428293256401, 
#     'reg_alpha': 0.4865138988842402, 
#     'reg_lambda': 0.6015849227570268
# }
# best_xgb_params.update({
#     'tree_method': 'hist', 
#     # 'device': 'cuda', 
#     'n_jobs': -1, 
#     'random_state': 42
# })

# best_lgbm_params = {
#     'n_estimators': 1232, 
#     'learning_rate': 0.045467475791811464, 
#     'num_leaves': 32, 
#     'min_child_samples': 38, 
#     'subsample': 0.9389508238313968, 
#     'colsample_bytree': 0.8358504077200445, 
#     'reg_alpha': 0.10126277169074206, 
#     'reg_lambda': 0.1357065010990351
# }
# best_lgbm_params.update({
#     # 'device': 'gpu', 
#     'n_jobs': -1, 
#     'random_state': 42, 
#     'verbose': -1
# })

# best_hgb_params = {
#     'l2_regularization': 0.4424625102595975, 
#     'learning_rate': 0.04919657248382905, 
#     'max_depth': None, 
#     'max_iter': 300, 
#     'max_leaf_nodes': 54, 
#     'min_samples_leaf': 30,
#     'random_state': 42
# }

# best_gbm_params = {
#     'learning_rate': 0.021616722433639893, 
#     'max_depth': 7, 
#     'min_samples_leaf': 4, 
#     'min_samples_split': 9, 
#     'n_estimators': 500, 
#     'subsample': 0.608233797718321,
#     'random_state': 42
# }

# # --- A. XGBoost (Wrapped) ---
# # XGBoost requires MultiOutputRegressor wrapper for multi-target
# xgb_model = MultiOutputRegressor(
#     XGBRegressor(
#         **best_xgb_params
#     )
# )

# # --- B. LightGBM (Wrapped) ---
# # LightGBM requires MultiOutputRegressor wrapper.
# # Note: Ensure you have the GPU-compiled version of LightGBM installed.
# lgbm_model = MultiOutputRegressor(
#     LGBMRegressor(
#         **best_lgbm_params
#     )
# )

# # --- C. CatBoost (Native) ---
# # CatBoost supports "MultiRMSE" natively. No wrapper needed.
# # This is usually the fastest option for multi-target on GPU.
# cat_model = CatBoostRegressor(
#     **best_cat_params
# )

# # ---------------------------------------------------------
# # 3. Usage Example
# # ---------------------------------------------------------
# # Assuming 'train' and 'test' pandas DataFrames exist
# # and TARGET_NAMES / TARGET_MAX / COLUMNS are defined globally

# # # 1. Run XGBoost
# # print("\n--- Running XGBoost ---")
# # oof_xgb, test_xgb = cross_validate_multioutput_gpu(xgb_model, train_siglip_df, test_siglip_df, feature_engine=feat_engine)
# # compare_results(oof_xgb, train_siglip_df)

# # # 2. Run LightGBM
# # print("\n--- Running LightGBM ---")
# # oof_lgbm, test_lgbm = cross_validate_multioutput_gpu(lgbm_model, train_siglip_df, test_siglip_df, feature_engine=feat_engine)
# # compare_results(oof_lgbm, train_siglip_df)

# # # 3. Run CatBoost
# # print("\n--- Running CatBoost ---")
# # oof_cat, test_cat = cross_validate_multioutput_gpu(cat_model, train_siglip_df, test_siglip_df, feature_engine=feat_engine)
# # compare_results(oof_cat, train_siglip_df)

# # print("\n######## Ridge Regression #######")
# # # ridge_model = MultiOutputRegressor(
# # #     Ridge()
# # # )
# # oof_ridge, pred_test_ri = cross_validate_multioutput_gpu(
# #     Ridge(), 
# #     train_siglip_df, test_siglip_df, 
# #     feature_engine=feat_engine,
# # )
# # compare_results(oof_ridge, train_siglip_df)

# # print("\n###### Bayesian Ridge Regressor #######")
# # bayesian_model = MultiOutputRegressor(
# #     BayesianRidge()
# # )
# # oof_bayesian, pred_test_bri = cross_validate_multioutput_gpu(
# #     bayesian_model, 
# #     train_siglip_df, test_siglip_df, 
# #     feature_engine=feat_engine,
# # )
# # compare_results(oof_bayesian, train_siglip_df)

# print("\n###### GradientBoosting Regressor #######")
# gbm_model = MultiOutputRegressor(
#     GradientBoostingRegressor(**best_gbm_params)
# )

# oof_gb, pred_test_gb = cross_validate_multioutput_gpu(
#     gbm_model, 
#     train_siglip_df, test_siglip_df, 
#     feature_engine=feat_engine,
#     semantic_train=sem_train_full,
#     semantic_test=sem_test_full
# )
# compare_results(oof_gb, train_siglip_df)

# print("\n###### Hist Gradient Boosting Regressor ######")
# hist_model = MultiOutputRegressor(
#     HistGradientBoostingRegressor(**best_hgb_params)
# )

# oof_hb, pred_test_hb = cross_validate_multioutput_gpu(
#     hist_model, 
#     train_siglip_df, test_siglip_df, 
#     feature_engine=feat_engine,
#     semantic_train=sem_train_full,
#     semantic_test=sem_test_full
# )
# compare_results(oof_hb, train_siglip_df)
```

```python
pred_test = (
    pred_test_hb
    + pred_test_gb
    + pred_test_cat
    + pred_test_lgbm
) / 4

# pred_test = (
#     pred_test_hb
#     + pred_test_et
# ) / 2

# pred_test = (
#     pred_test_gb
#     + pred_test_hb
#     + pred_test_et
# ) / 3

# pred_test = (
#     pred_test_ri
#     + pred_test_gb
#     + pred_test_hb
#     + pred_test_et
# ) / 4

# pred_test = (test_xgb + test_lgbm + test_cat + pred_test_ri + pred_test_bri) / 5
# pred_test = 0.6*pred_test_ri + 0.15*pred_test_gb + 0.15*pred_test_hb + 0.1*pred_test_et
```

```python
test_df[TARGET_NAMES] = pred_test
test_df = post_process_biomass(test_df)
# test_df['GDM_g'] = test_df['Dry_Green_g'] + test_df['Dry_Clover_g']
# test_df['Dry_Total_g'] = test_df['GDM_g'] + test_df['Dry_Dead_g']
sub_df = melt_table(test_df)
sub_df[['sample_id', 'target']].to_csv("submission_siglip.csv", index=False)
```

```python
pd.read_csv("submission_siglip.csv")
```

# Ensemble

```python
def ensemble_submission(files=None, weights=None, postprocess=True, output_name="submission.csv"):
    """
    Create ensemble submission from submission_siglip.csv and submission_dinoV3.csv.
    Uses weights (sum normalized) and writes output to CFG.SUBMISSION_DIR/output_name.
    """
    import os
    import numpy as np
    import pandas as pd

    # Defaults
    if files is None:
        files = ["submission_siglip.csv", "submission_dinoV3.csv"]
    if weights is None:
        # Use the Weight variable defined earlier in the notebook if present
        try:
            w = Weight
        except NameError:
            w = [0.55, 0.45]
    else:
        w = weights

    # Validate
    if len(w) != len(files):
        raise ValueError("Number of weights must match number of files")
    w = np.array(w, dtype=float)
    w = w / w.sum()

    # Read submissions
    series_list = []
    for f in files:
        if not os.path.exists(f):
            raise FileNotFoundError(f"File not found: {f}")
        s = pd.read_csv(f).set_index("sample_id")["target"]
        series_list.append(s.rename(os.path.splitext(os.path.basename(f))[0]))

    df = pd.concat(series_list, axis=1)  # align by sample_id

    vals = df.values.astype(float)  # (n_samples, n_models)
    mask = ~np.isnan(vals)

    numer = np.nansum(vals * w.reshape(1, -1), axis=1)
    denom = np.nansum(mask * w.reshape(1, -1), axis=1)
    avg = numer / np.where(denom == 0, 1.0, denom)

    out = pd.DataFrame({"sample_id": df.index, "target": avg})

    if postprocess:
        # convert to wide per image, apply post_process_biomass (must exist in notebook)
        tmp = out.copy()
        tmp[["image_id", "target_name"]] = tmp["sample_id"].str.rsplit("__", n=1, expand=True)
        wide = tmp.pivot(index="image_id", columns="target_name", values="target").reset_index()
        # ensure all cols present
        for c in CFG.ALL_TARGET_COLS:
            if c not in wide.columns:
                wide[c] = 0.0
        # use post_process_biomass defined earlier in the notebook
        wide_proc = post_process_biomass(wide)
        long = wide_proc.melt(id_vars="image_id", value_vars=CFG.ALL_TARGET_COLS, var_name="target_name", value_name="target")
        long["sample_id"] = long["image_id"] + "__" + long["target_name"]
        out = long[["sample_id", "target"]].set_index("sample_id").loc[df.index].reset_index()

    # Align to test.csv ordering if available
    try:
        test_df = pd.read_csv(CFG.TEST_CSV)
        if "sample_id" in test_df.columns:
            out = test_df[["sample_id"]].merge(out, on="sample_id", how="left")
    except Exception:
        pass

    out["target"] = out["target"].fillna(0.0)
    save_path = os.path.join(CFG.SUBMISSION_DIR, output_name)
    out.to_csv(save_path, index=False)
    print(f"Saved ensemble submission: {save_path} (rows={len(out)})")
    return out

# Convenience call using your requested weights
# Run this cell to produce submission.csv
ensemble_submission(files=["submission_siglip.csv","submission_dinoV3.csv"], weights=Weight, postprocess=True, output_name="submission.csv")
```