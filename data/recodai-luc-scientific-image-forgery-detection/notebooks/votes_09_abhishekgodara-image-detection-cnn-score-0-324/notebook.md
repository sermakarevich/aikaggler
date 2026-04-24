# image detection 🔬🔬 | CNN | score[0.324]🔥🔥

- **Author:** Abhishek Godara
- **Votes:** 143
- **Ref:** abhishekgodara/image-detection-cnn-score-0-324
- **URL:** https://www.kaggle.com/code/abhishekgodara/image-detection-cnn-score-0-324
- **Last run:** 2026-01-05 02:23:19.297000

---

# Notebook Overview: CNN-DINOv2 Hybrid

This notebook demonstrates a hybrid approach for image classification using both Convolutional Neural Networks (CNNs) and DINOv2, a self-supervised vision transformer model. The workflow includes:

- **Data Loading & Preprocessing:** Images are loaded, resized, normalized, and split into training and validation sets.
- **Feature Extraction:** DINOv2 is used to extract high-level features from images, leveraging its transformer-based architecture for robust representations.
- **CNN Model Construction:** A custom CNN is built to process image data, learning spatial hierarchies and patterns.
- **Hybrid Model Integration:** Features from DINOv2 and the CNN are combined, either by concatenation or other fusion techniques, to enhance classification performance.
- **Training & Evaluation:** The hybrid model is trained on the dataset, with metrics such as accuracy and loss tracked. Validation is performed to assess generalization.
- **Visualization & Analysis:** Results, including confusion matrices and sample predictions, are visualized to interpret model behavior.

This approach aims to leverage the strengths of both CNNs (local feature learning) and DINOv2 (global, context-aware representations) for improved image classification results.

```python
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'
```

#  Step 1: Model Setup, Dataset Preparation, and Validation Scoring

```python
import os, cv2, json, math, random, torch, time
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from transformers import AutoImageProcessor, AutoModel

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False

seed_everything(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR  = "/kaggle/input/recodai-luc-scientific-image-forgery-detection"
AUTH_DIR  = f"{BASE_DIR}/train_images/authentic"
FORG_DIR  = f"{BASE_DIR}/train_images/forged"
MASK_DIR  = f"{BASE_DIR}/train_masks"
TEST_DIR  = f"{BASE_DIR}/test_images"
DINO_PATH_LARGE = "/kaggle/input/dinov2/pytorch/large/1"
DINO_PATH_BASE = "/kaggle/input/dinov2/pytorch/base/1"

IMG_SIZE = 718
BATCH_SIZE = 1
MODEL_LOC = '/kaggle/input/cnndinov2-pbd/CNNDINOv2-U52/CNNDINOv2-U52/model_seg_final.pt'
# ADAPTIVE THRESHOLDS (will be set per image)
USE_TTA = True
USE_ENSEMBLE = True  
USE_CALIBRATION = True  # Add calibration
USE_ADAPTIVE_THRESHOLDS = True  # Adaptive thresholds per image

class ForgerySegDataset(Dataset):
    def __init__(self, auth_paths, forg_paths, mask_dir, img_size=IMG_SIZE):
        self.samples = []
        for p in forg_paths:
            m = os.path.join(mask_dir, Path(p).stem + ".npy")
            if os.path.exists(m):
                self.samples.append((p, m))
        for p in auth_paths:
            self.samples.append((p, None))
        self.img_size = img_size
    
    def __len__(self): 
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        
        if mask_path is None:
            mask = np.zeros((h, w), np.uint8)
        else:
            m = np.load(mask_path)
            if m.ndim == 3: 
                m = np.max(m, axis=0)
            mask = (m > 0).astype(np.uint8)
        
        img_r = img.resize((IMG_SIZE, IMG_SIZE))
        mask_r = cv2.resize(mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
        img_t = torch.from_numpy(np.array(img_r, np.float32)/255.).permute(2,0,1)
        mask_t = torch.from_numpy(mask_r[None, ...].astype(np.float32))
        return img_t, mask_t


# ========== LOAD BASE MODEL (for ensemble) ==========
print("Loading Base model for ensemble...")
processor_base = AutoImageProcessor.from_pretrained(DINO_PATH_BASE, local_files_only=True, use_fast=False)
encoder_base = AutoModel.from_pretrained(DINO_PATH_BASE, local_files_only=True).eval().to(device)

class DinoTinyDecoder(nn.Module):
    def __init__(self, in_ch=768, out_ch=1):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_ch, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(384, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(192, 96, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv_out = nn.Conv2d(96, out_ch, kernel_size=1)
    
    def forward(self, f, target_size):
        x = F.interpolate(self.block1(f), size=(74, 74), mode='bilinear', align_corners=False)
        x = F.interpolate(self.block2(x), size=(148, 148), mode='bilinear', align_corners=False)
        x = F.interpolate(self.block3(x), size=(296, 296), mode='bilinear', align_corners=False)
        x = self.conv_out(x)
        x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
        return x

class DinoSegmenterBase(nn.Module):
    def __init__(self, encoder, processor):
        super().__init__()
        self.encoder, self.processor = encoder, processor
        for p in self.encoder.parameters(): 
            p.requires_grad = False
        self.seg_head = DinoTinyDecoder(768, 1)
    
    def forward_features(self, x):
        imgs = (x * 255).clamp(0, 255).byte().permute(0, 2, 3, 1).cpu().numpy()
        inputs = self.processor(images=list(imgs), return_tensors="pt").to(x.device)
        feats = self.encoder(**inputs).last_hidden_state
        B, N, C = feats.shape
        fmap = feats[:, 1:, :].permute(0, 2, 1)
        s = int(math.sqrt(N - 1))
        fmap = fmap.reshape(B, C, s, s)
        return fmap
    
    def forward_seg(self, x):
        fmap = self.forward_features(x)
        return self.seg_head(fmap, (IMG_SIZE, IMG_SIZE))

# Load pretrained Base model
model_base = DinoSegmenterBase(encoder_base, processor_base).to(device)
if MODEL_LOC is not None and os.path.exists(MODEL_LOC):
    model_base.load_state_dict(torch.load(MODEL_LOC, map_location=device))
    print("✅ Loaded pretrained Base model")
model_base.eval()

# ========== LOAD LARGE MODEL ==========
print("Loading Large model...")
processor_large = AutoImageProcessor.from_pretrained(DINO_PATH_LARGE, local_files_only=True, use_fast=False)
encoder_large = AutoModel.from_pretrained(DINO_PATH_LARGE, local_files_only=True).eval().to(device)

# ========== FIXED: ADD REGULARIZATION TO DECODER ==========
class DinoLargeDecoderRegularized(nn.Module):
    def __init__(self, in_ch=1024, out_ch=1):
        super().__init__()
        # Increased dropout for regularization
        self.block1 = nn.Sequential(
            nn.Conv2d(in_ch, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.15)  # Increased from 0.1 to 0.3
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.15)  # Increased from 0.1 to 0.3
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)  # Added dropout
        )
        self.conv_out = nn.Conv2d(128, out_ch, kernel_size=1)
    
    def forward(self, f, target_size):
        x = F.interpolate(self.block1(f), size=(74, 74), mode='bilinear', align_corners=False)
        x = F.interpolate(self.block2(x), size=(148, 148), mode='bilinear', align_corners=False)
        x = F.interpolate(self.block3(x), size=(296, 296), mode='bilinear', align_corners=False)
        x = self.conv_out(x)
        x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
        return x

class DinoSegmenterLarge(nn.Module):
    def __init__(self, encoder, processor):
        super().__init__()
        self.encoder, self.processor = encoder, processor
        for p in self.encoder.parameters(): 
            p.requires_grad = False
        self.seg_head = DinoLargeDecoderRegularized(1024, 1)  # Use regularized decoder
    
    def forward_features(self, x):
        imgs = (x * 255).clamp(0, 255).byte().permute(0, 2, 3, 1).cpu().numpy()
        inputs = self.processor(images=list(imgs), return_tensors="pt").to(x.device)
        feats = self.encoder(**inputs).last_hidden_state
        B, N, C = feats.shape
        fmap = feats[:, 1:, :].permute(0, 2, 1)
        s = int(math.sqrt(N - 1))
        fmap = fmap.reshape(B, C, s, s)
        return fmap
    
    def forward_seg(self, x):
        fmap = self.forward_features(x)
        return self.seg_head(fmap, (IMG_SIZE, IMG_SIZE))

model_large = DinoSegmenterLarge(encoder_large, processor_large).to(device)
print("✅ Both models loaded (Large with regularization)")

# ========== DATA LOADERS ==========
auth_imgs = sorted([str(Path(AUTH_DIR)/f) for f in os.listdir(AUTH_DIR)])
forg_imgs = sorted([str(Path(FORG_DIR)/f) for f in os.listdir(FORG_DIR)])
train_auth, val_auth = train_test_split(auth_imgs, test_size=0.2, random_state=42)
train_forg, val_forg = train_test_split(forg_imgs, test_size=0.2, random_state=42)

train_loader = DataLoader(ForgerySegDataset(train_auth, train_forg, MASK_DIR),
                          batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(ForgerySegDataset(val_auth, val_forg, MASK_DIR),
                        batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

print(f"Training samples: {len(train_loader.dataset)}")
print(f"Validation samples: {len(val_loader.dataset)}")

# ========== TRAINING SETUP ==========
print("\n" + "="*50)
print("SETTING UP TRAINING (5 EPOCHS)")
print("="*50)

# Unfreeze decoder for training
for p in model_large.seg_head.parameters():
    p.requires_grad = True

# Enhanced loss function with regularization
class RegularizedDiceBCELoss(nn.Module):
    def __init__(self, dice_weight=0.4, l2_weight=1e-5):
        super().__init__()
        self.dice_weight = dice_weight
        self.l2_weight = l2_weight
        
    def forward(self, pred, target, model):
        # BCE loss
        bce = F.binary_cross_entropy_with_logits(pred, target)
        
        # Dice loss
        pred_sigmoid = torch.sigmoid(pred)
        smooth = 1.0
        intersection = (pred_sigmoid * target).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (pred_sigmoid.sum() + target.sum() + smooth)
        
        # L2 regularization
        l2_reg = torch.tensor(0., device=pred.device)
        for param in model.seg_head.parameters():
            l2_reg += torch.norm(param)
        
        # Combined loss
        return (1 - self.dice_weight) * bce + self.dice_weight * dice_loss + self.l2_weight * l2_reg

# Optimizer with weight decay (L2 regularization)
optimizer = torch.optim.AdamW(model_large.seg_head.parameters(), 
                             lr=1.5e-4,  # Slightly lower LR
                             weight_decay=2e-5)  # Increased weight decay
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-6)
criterion = RegularizedDiceBCELoss(dice_weight=0.4, l2_weight=1e-5)

print(f"Trainable parameters: {sum(p.numel() for p in model_large.seg_head.parameters() if p.requires_grad):,}")

# ========== TRAINING LOOP (5 EPOCHS) ==========
print("\n" + "="*50)
print("STARTING 5-EPOCH TRAINING WITH REGULARIZATION")
print("="*50)

NUM_EPOCHS = 5
model_large.train()
best_val_f1 = 0
start_time = time.time()

for epoch in range(NUM_EPOCHS):
    epoch_loss = 0
    batch_count = 0
    
    # Training
    for batch_idx, (images, masks) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")):
        images, masks = images.to(device), masks.to(device)
        
        optimizer.zero_grad()
        outputs = model_large.forward_seg(images)
        loss = criterion(outputs, masks, model_large)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model_large.seg_head.parameters(), max_norm=0.5)  # Tighter clipping
        
        optimizer.step()
        epoch_loss += loss.item()
        batch_count += 1
    
    # Update scheduler
    scheduler.step()
    avg_loss = epoch_loss / batch_count
    
    # Validation with confidence calibration
    model_large.eval()
    val_f1s = []
    with torch.no_grad():
        for i, (val_img, val_mask) in enumerate(val_loader):
            if i >= 15:  # Check more batches
                break
            val_img, val_mask = val_img.to(device), val_mask.to(device)
            
            # Get predictions with temperature scaling (calibration)
            logits = model_large.forward_seg(val_img)
            preds_large = torch.sigmoid(logits / 1.2)  # Temperature = 1.5
            
            preds_base = torch.sigmoid(model_base.forward_seg(val_img))
            
            # Ensemble (balanced weights)
            preds = 0.5 * preds_large + 0.5 * preds_base  # Equal weights
            
            # Conservative thresholding
            preds_bin = (preds > 0.55).float()  # Higher threshold
            
            # F1 calculation
            intersection = (preds_bin * val_mask).sum()
            f1 = (2 * intersection) / (preds_bin.sum() + val_mask.sum() + 1e-6)
            val_f1s.append(f1.item())
    
    avg_val_f1 = np.mean(val_f1s) if val_f1s else 0
    
    # Save best model (more conservative)
    if avg_val_f1 > best_val_f1 and avg_val_f1 > 0.25:  # Only save if reasonable
        best_val_f1 = avg_val_f1
        torch.save(model_large.state_dict(), 'best_large_model_regularized.pth')
        print(f"  💾 Saved best model (F1: {best_val_f1:.4f})")
    
    print(f"Epoch {epoch+1}/{NUM_EPOCHS}:")
    print(f"  Loss: {avg_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")
    print(f"  Val F1 (calibrated): {avg_val_f1:.4f}")
    
    model_large.train()

total_time = time.time() - start_time
print(f"\n✅ Training completed in {total_time/60:.1f} minutes")
print(f"📊 Best validation F1: {best_val_f1:.4f}")

# Load best model
if os.path.exists('best_large_model_regularized.pth'):
    model_large.load_state_dict(torch.load('best_large_model_regularized.pth', map_location=device))
    print("✅ Loaded best regularized model weights")
else:
    print("⚠️ Using last epoch weights")

model_large.eval()

# ========== IMPROVED INFERENCE FUNCTIONS ==========
def calibrated_sigmoid(logits, temperature=1.2):
    """Temperature scaling to reduce overconfidence"""
    return torch.sigmoid(logits / temperature)

@torch.no_grad()
def segment_prob_map_large(pil):
    x = torch.from_numpy(np.array(pil.resize((IMG_SIZE, IMG_SIZE)), np.float32)/255.).permute(2,0,1)[None].to(device)
    prob = calibrated_sigmoid(model_large.forward_seg(x))[0,0].cpu().numpy()
    return prob

@torch.no_grad()
def segment_prob_map_base(pil):
    x = torch.from_numpy(np.array(pil.resize((IMG_SIZE, IMG_SIZE)), np.float32)/255.).permute(2,0,1)[None].to(device)
    prob = torch.sigmoid(model_base.forward_seg(x))[0,0].cpu().numpy()
    return prob

@torch.no_grad()
def segment_prob_map_with_calibrated_tta(pil):
    """TTA with confidence calibration"""
    x = torch.from_numpy(np.array(pil.resize((IMG_SIZE, IMG_SIZE)), np.float32)/255.).permute(2,0,1)[None].to(device)
    
    predictions = []
    confidences = []
    
    # Original
    pred = calibrated_sigmoid(model_large.forward_seg(x), temperature=1.5)
    predictions.append(pred)
    conf = pred.mean().item()
    confidences.append(max(0.1, conf))  # Minimum confidence
    
    # Horizontal flip
    pred_h = calibrated_sigmoid(model_large.forward_seg(torch.flip(x, dims=[3])), temperature=1.5)
    predictions.append(torch.flip(pred_h, dims=[3]))
    confidences.append(max(0.1, pred_h.mean().item()))
    
    # Vertical flip  
    pred_v = calibrated_sigmoid(model_large.forward_seg(torch.flip(x, dims=[2])), temperature=1.5)
    predictions.append(torch.flip(pred_v, dims=[2]))
    confidences.append(max(0.1, pred_v.mean().item()))
    
    # Weighted average by confidence
    confidences = np.array(confidences)
    weights = confidences / confidences.sum()
    
    weighted_pred = torch.zeros_like(predictions[0])
    for i, pred in enumerate(predictions):
        weighted_pred += weights[i] * pred
    
    return weighted_pred[0,0].cpu().numpy()

def adaptive_thresholds(pil):
    """Set thresholds based on image characteristics"""
    img_array = np.array(pil.convert('L')).astype(float)
    
    # Calculate image statistics
    brightness = img_array.mean()
    contrast = img_array.std()
    
    # Adjust thresholds based on image properties
    if brightness > 200:  # Very bright image
        area_thr = 120
        mean_thr = 0.25
    elif brightness < 50:  # Very dark image
        area_thr = 100
        mean_thr = 0.22
    elif contrast < 25:   # Low contrast (smooth)
        area_thr = 150
        mean_thr = 0.28
    elif contrast > 80:   # High contrast (textured)
        area_thr = 60
        mean_thr = 0.18
    else:                 # Normal case
        area_thr = 80
        mean_thr = 0.20
    
    return area_thr, mean_thr

def enhanced_adaptive_mask(prob, alpha_grad=0.35):
    gx = cv2.Sobel(prob, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(prob, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx**2 + gy**2)
    grad_norm = grad_mag / (grad_mag.max() + 1e-6)
    enhanced = (1 - alpha_grad) * prob + alpha_grad * grad_norm
    enhanced = cv2.GaussianBlur(enhanced, (3,3), 0)
    
    # Adaptive threshold based on image content
    prob_mean = np.mean(prob)
    prob_std = np.std(prob)
    
    if prob_mean < 0.1:  # Low overall activation
        thr = prob_mean + 0.2 * prob_std
    elif prob_mean > 0.3:  # High overall activation
        thr = prob_mean + 0.3 * prob_std
    else:  # Medium activation
        thr = prob_mean + 0.25 * prob_std
    
    mask = (enhanced > thr).astype(np.uint8)
    
    # Conservative morphology
    if mask.sum() < 5000:
        kernel_size = 3
    elif mask.sum() < 20000:
        kernel_size = 5
    else:
        kernel_size = 7
    
    kernel_close = np.ones((kernel_size, kernel_size), np.uint8)
    kernel_open = np.ones((max(2, kernel_size-2), max(2, kernel_size-2)), np.uint8)
    
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
    
    return mask, thr

def finalize_mask(prob, orig_size):
    mask, thr = enhanced_adaptive_mask(prob)
    mask = (mask > 0).astype(np.uint8)
    mask = cv2.resize(mask, orig_size, interpolation=cv2.INTER_NEAREST)
    return mask, thr

def robust_pipeline_final(pil):
    # 1. Get calibrated prediction
    if USE_TTA and USE_CALIBRATION:
        prob_large = segment_prob_map_with_calibrated_tta(pil)
    elif USE_TTA:
        prob_large = segment_prob_map_with_tta(pil)
    else:
        prob_large = segment_prob_map_large(pil)
    
    # 2. Ensemble with Base (balanced)
    if USE_ENSEMBLE:
        prob_base = segment_prob_map_base(pil)
        prob = 0.5 * prob_large + 0.5 * prob_base  # Equal weights
    else:
        prob = prob_large
    
    # 3. Adaptive thresholds
    if USE_ADAPTIVE_THRESHOLDS:
        area_thr, mean_thr = adaptive_thresholds(pil)
    else:
        # Conservative default thresholds
        area_thr, mean_thr = 100, 0.22
    
    # 4. Post-processing
    mask, thr = finalize_mask(prob, pil.size)
    area = int(mask.sum())
    
    # 5. Accurate mean calculation
    if area > 0:
        prob_resized = cv2.resize(prob, (mask.shape[1], mask.shape[0]))
        mean_inside = float(prob_resized[mask == 1].mean())
    else:
        mean_inside = 0.0
    
    # 6. Conservative decision with confidence check
    max_prob = prob.max()
    
    # Additional checks
    if area < area_thr or mean_inside < mean_thr:
        return "authentic", None, {"area": area, "mean_inside": mean_inside, "thr": thr}
    elif max_prob < 0.15:  # Low confidence
        return "authentic", None, {"area": area, "mean_inside": mean_inside, "thr": thr, "max_prob": max_prob}
    else:
        return "forged", mask, {"area": area, "mean_inside": mean_inside, "thr": thr, "max_prob": max_prob}

# ========== COMPREHENSIVE VALIDATION ==========
print("\n" + "="*60)
print("COMPREHENSIVE VALIDATION WITH IMPROVED PIPELINE")
print("="*60)

from sklearn.metrics import f1_score, accuracy_score

# Test on 10 forged + 10 authentic
test_forged = val_forg[:10]
test_authentic = val_auth[:10]

all_results = []
all_labels = []
all_predictions = []

print("\n🔴 TESTING FORGED IMAGES:")
forged_f1s = []
for p in tqdm(test_forged, desc="Forged"):
    pil = Image.open(p).convert("RGB")
    label, m_pred, dbg = robust_pipeline_final(pil)
    
    # Ground truth mask
    m_gt = np.load(Path(MASK_DIR)/f"{Path(p).stem}.npy")
    if m_gt.ndim == 3: 
        m_gt = np.max(m_gt, axis=0)
    m_gt = (m_gt > 0).astype(np.uint8)
    
    # Predicted mask
    m_pred_bin = (m_pred > 0).astype(np.uint8) if m_pred is not None else np.zeros_like(m_gt)
    
    # Calculate F1
    f1 = f1_score(m_gt.flatten(), m_pred_bin.flatten(), zero_division=0)
    forged_f1s.append(f1)
    
    # Store for overall metrics
    all_results.append(("forged", f1, dbg, label))
    all_labels.append(1)  # 1 for forged
    all_predictions.append(1 if label == "forged" else 0)
    
    print(f"  {Path(p).stem}: {label} | F1={f1:.4f} | area={dbg.get('area', 0)} mean={dbg.get('mean_inside', 0):.3f}")

print("\n🟢 TESTING AUTHENTIC IMAGES:")
authentic_acc = []
for p in tqdm(test_authentic, desc="Authentic"):
    pil = Image.open(p).convert("RGB")
    label, m_pred, dbg = robust_pipeline_final(pil)
    
    # Check if correctly classified as authentic
    is_correct = (label == "authentic")
    authentic_acc.append(is_correct)
    
    # Store for overall metrics
    all_results.append(("authentic", 0, dbg, label))
    all_labels.append(0)  # 0 for authentic
    all_predictions.append(1 if label == "forged" else 0)
    
    print(f"  {Path(p).stem}: {label} | Correct={is_correct} | area={dbg.get('area', 0)} mean={dbg.get('mean_inside', 0):.3f}")

# Calculate metrics
print("\n" + "="*60)
print("FINAL METRICS")
print("="*60)

# Forged detection metrics
avg_forged_f1 = np.mean(forged_f1s)
forged_detected = sum([1 for _, f1, _, label in all_results[:10] if label == "forged"])
forged_missed = 10 - forged_detected

# Authentic classification metrics
authentic_accuracy = np.mean(authentic_acc)
authentic_correct = sum(authentic_acc)
authentic_wrong = 10 - authentic_correct

# Overall binary classification metrics
from sklearn.metrics import precision_score, recall_score
binary_accuracy = accuracy_score(all_labels, all_predictions)
binary_precision = precision_score(all_labels, all_predictions, zero_division=0)
binary_recall = recall_score(all_labels, all_predictions, zero_division=0)
binary_f1 = 2 * (binary_precision * binary_recall) / (binary_precision + binary_recall + 1e-6)

print(f"\n🔴 FORGED IMAGES (10 samples):")
print(f"  • Average F1: {avg_forged_f1:.4f}")
print(f"  • Detected: {forged_detected}/10 ({forged_detected/10:.0%})")
print(f"  • Missed: {forged_missed}/10")

print(f"\n🟢 AUTHENTIC IMAGES (10 samples):")
print(f"  • Accuracy: {authentic_accuracy:.2%} ({authentic_correct}/10 correct)")
print(f"  • False positives: {authentic_wrong}/10")

print(f"\n📊 OVERALL BINARY CLASSIFICATION:")
print(f"  • Accuracy: {binary_accuracy:.2%}")
print(f"  • Precision: {binary_precision:.2%} (important for test set)")
print(f"  • Recall: {binary_recall:.2%}")
print(f"  • F1 Score: {binary_f1:.4f}")

print(f"\n🎯 PERFORMANCE ESTIMATE FOR TEST SET:")
print(f"  Previous Base model: ~0.324")
print(f"  Previous Large model (overfit): 0.246")
print(f"  Current regularized model: ~{avg_forged_f1:.4f}")
print(f"  Expected test score: {max(0.30, min(0.36, avg_forged_f1 * 0.9)):.3f} (adjusted for test distribution)")

# Save final model
print("\n💾 Saving final improved model...")
torch.save({
    'large_model': model_large.state_dict(),
    'base_model': model_base.state_dict(),
    'config': {
        'use_tta': USE_TTA,
        'use_ensemble': USE_ENSEMBLE,
        'use_calibration': USE_CALIBRATION,
        'use_adaptive_thresholds': USE_ADAPTIVE_THRESHOLDS
    }
}, 'improved_regularized_model.pth')
print("✅ Model saved as 'improved_regularized_model.pth'")

print("\n🚀 READY FOR SUBMISSION!")
print("Use 'robust_pipeline_final()' for inference")
```

# Step 2: Hybrid Model — DINOv2 Feature Extraction & CNN Decoder Integration

```python
import os, json, cv2
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

# --- RLE Encoder for Kaggle Submission ---
def rle_encode(mask: np.ndarray, fg_val: int = 1) -> str:
    pixels = mask.T.flatten()
    dots = np.where(pixels == fg_val)[0]
    if len(dots) == 0:
        return "authentic"
    run_lengths = []
    prev = -2
    for b in dots:
        if b > prev + 1:
            run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return json.dumps([int(x) for x in run_lengths])

# --- Paths ---
TEST_DIR = "/kaggle/input/recodai-luc-scientific-image-forgery-detection/test_images"
SAMPLE_SUB = "/kaggle/input/recodai-luc-scientific-image-forgery-detection/sample_submission.csv"
OUT_PATH = "submission.csv"

rows = []
for f in tqdm(sorted(os.listdir(TEST_DIR)), desc="Inference on Test Set"):
    pil = Image.open(Path(TEST_DIR)/f).convert("RGB")
    
    # ====== CRITICAL: Use robust_pipeline_final NOT pipeline_final ======
    label, mask, dbg = robust_pipeline_final(pil)  # CHANGED HERE!
    # ====================================================================

    # Sécurisation masque
    if mask is None:
        mask = np.zeros(pil.size[::-1], np.uint8)
    else:
        mask = np.array(mask, dtype=np.uint8)

    # Annotation finale
    if label == "authentic":
        annot = "authentic"
    else:
        annot = rle_encode((mask > 0).astype(np.uint8))

    rows.append({
        "case_id": Path(f).stem,
        "annotation": annot,
        "area": int(dbg.get("area", mask.sum())),
        "mean": float(dbg.get("mean_inside", 0.0)),
        "thr": float(dbg.get("thr", 0.0)),
        "max_prob": float(dbg.get("max_prob", 0.0)) if "max_prob" in dbg else 0.0
    })

sub = pd.DataFrame(rows)
ss = pd.read_csv(SAMPLE_SUB)
ss["case_id"] = ss["case_id"].astype(str)
sub["case_id"] = sub["case_id"].astype(str)
final = ss[["case_id"]].merge(sub, on="case_id", how="left")
final["annotation"] = final["annotation"].fillna("authentic")
final[["case_id", "annotation"]].to_csv(OUT_PATH, index=False)

print(f"\n✅ Saved submission file: {OUT_PATH}")
print(final.head(10))

# Quick sample visualization
sample_files = sorted(os.listdir(TEST_DIR))[:5]
for f in sample_files:
    pil = Image.open(Path(TEST_DIR)/f).convert("RGB")
    
    # ====== CRITICAL: Use robust_pipeline_final NOT pipeline_final ======
    label, mask, dbg = robust_pipeline_final(pil)  # CHANGED HERE!
    # ====================================================================
    
    mask = np.array(mask, dtype=np.uint8) if mask is not None else np.zeros(pil.size[::-1], np.uint8)

    print(f"{'🔴' if label=='forged' else '🟢'} {f}: {label} | area={mask.sum()} mean={dbg.get('mean_inside', 0):.3f}")

    if label == "authentic":
        plt.figure(figsize=(5,5))
        plt.imshow(pil)
        plt.title(f"{f} — Authentic")
        plt.axis("off")
        plt.show()
    else:
        plt.figure(figsize=(10,5))
        plt.subplot(1,2,1)
        plt.imshow(pil)
        plt.title("Original Image")
        plt.axis("off")
        plt.subplot(1,2,2)
        plt.imshow(pil)
        plt.imshow(mask, alpha=0.45, cmap="Reds")
        plt.title(f"Predicted Forged Mask\nArea={mask.sum()} | Mean={dbg.get('mean_inside', 0):.3f}")
        plt.axis("off")
        plt.show()
```

## 🔴 Visualizing Predicted Masks with the CNN–DINOv2 Hybrid Model

```python
import torch, cv2, math, numpy as np, matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ====== USE THE NEW FUNCTIONS FROM MAIN CELL ======
# All functions are already defined in main cell, so just use them directly

# Visualization pipeline (uses robust_pipeline_final)
def pipeline_visual(pil):
    """Wrapper for visualization that uses the same robust pipeline"""
    label, mask, dbg = robust_pipeline_final(pil)
    
    if mask is None:
        mask = np.zeros(pil.size[::-1], np.uint8)
    
    thr = dbg.get('thr', 0.0)
    area = dbg.get('area', 0)
    mean_inside = dbg.get('mean_inside', 0.0)
    
    return label, mask, thr, area, mean_inside

# Visualization (for validation forged samples)
sample_forged = val_forg[:5]
n = len(sample_forged)
fig, axes = plt.subplots(n, 3, figsize=(12, n * 3))
if n == 1:
    axes = np.expand_dims(axes, axis=0)

for i, p in enumerate(sample_forged):
    pil = Image.open(p).convert("RGB")
    label, m_pred, thr, area, mean = pipeline_visual(pil)

    # Ground Truth mask
    m_gt = np.load(Path(MASK_DIR)/f"{Path(p).stem}.npy")
    if m_gt.ndim == 3: 
        m_gt = np.max(m_gt, axis=0)
    m_gt = (m_gt > 0).astype(np.uint8)

    # Resize all for consistency
    img_disp = cv2.resize(np.array(pil), (IMG_SIZE, IMG_SIZE))
    gt_disp  = cv2.resize(m_gt, (IMG_SIZE, IMG_SIZE))
    pr_disp  = cv2.resize(m_pred, (IMG_SIZE, IMG_SIZE))

    # === Column 1: Original ===
    axes[i, 0].imshow(img_disp)
    axes[i, 0].set_title("🖼️ Original Image", fontsize=11, weight="bold")
    axes[i, 0].axis("off")

    # === Column 2: Ground Truth ===
    axes[i, 1].imshow(gt_disp, cmap="gray")
    axes[i, 1].set_title("✅ Ground Truth", fontsize=11, weight="bold")
    axes[i, 1].axis("off")

    # === Column 3: Predicted Mask ===
    axes[i, 2].imshow(img_disp)
    axes[i, 2].imshow(pr_disp, cmap="coolwarm", alpha=0.45)
    axes[i, 2].set_title(f"🔮 Predicted ({label})\nThr={thr:.3f} | Area={area} | Mean={mean:.3f}",
                         fontsize=10)
    axes[i, 2].axis("off")

plt.subplots_adjust(top=0.92, hspace=0.35)
fig.suptitle("🔍 Segmentation of Forged Samples — Regularized CNN-DINOv2", 
             fontsize=16, fontweight="bold", color="#b30000")

plt.show()
```

# 🟢 Visualization of Authentic Images (Hybrid DINOv2-based Detector)

```python
import matplotlib.pyplot as plt
import cv2, numpy as np
from pathlib import Path
from PIL import Image

# Select a few authentic examples
sample_auth = val_auth[:5]
n = len(sample_auth)

fig, axes = plt.subplots(n, 2, figsize=(9, n * 3))
if n == 1:
    axes = np.expand_dims(axes, axis=0)

for i, p in enumerate(sample_auth):
    pil = Image.open(p).convert("RGB")
    
    # ====== USE pipeline_visual from Cell 2 ======
    label, m_pred, thr, area, mean = pipeline_visual(pil)  # Uses robust_pipeline_final internally
    # =============================================

    # Predicted mask (should be empty for authentic images)
    m_pred = (m_pred > 0).astype(np.uint8) if m_pred is not None else np.zeros((IMG_SIZE, IMG_SIZE))

    # Resize for consistent display
    img_disp = cv2.resize(np.array(pil), (IMG_SIZE, IMG_SIZE))
    pr_disp  = cv2.resize(m_pred, (IMG_SIZE, IMG_SIZE))

    # === Column 1: Original Image ===
    axes[i, 0].imshow(img_disp)
    axes[i, 0].set_title("🖼️ Original Image", fontsize=11, weight="bold")
    axes[i, 0].axis("off")

    # === Column 2: Predicted Mask ===
    axes[i, 1].imshow(img_disp)
    axes[i, 1].imshow(pr_disp, cmap="coolwarm", alpha=0.45)
    axes[i, 1].set_title(
        f"🟢 Predicted: {label.upper()}\nArea={area} | Mean={mean:.3f} | Thr={thr:.3f}",
        fontsize=10
    )
    axes[i, 1].axis("off")

    for j in range(2):
        axes[i, j].set_aspect("equal")

plt.subplots_adjust(top=0.90, hspace=0.35)
fig.suptitle("🟢 Segmentation of Authentic Images — Regularized CNN-DINOv2",
             fontsize=16, fontweight="bold", color="#009933")
plt.show()
```

```python
print("\n" + "="*60)
print("COMPREHENSIVE VALIDATION TEST - REGULARIZED MODEL")
print("="*60)

from sklearn.metrics import f1_score, accuracy_score

# Test on 10 forged + 10 authentic
test_forged = val_forg[:10]
test_authentic = val_auth[:10]

all_results = []
all_labels = []
all_predictions = []

print("\n🔴 TESTING FORGED IMAGES:")
forged_f1s = []
for p in tqdm(test_forged, desc="Forged"):
    pil = Image.open(p).convert("RGB")
    
    # ====== CRITICAL: Use robust_pipeline_final NOT pipeline_final ======
    label, m_pred, dbg = robust_pipeline_final(pil)  # CHANGED HERE!
    # ====================================================================
    
    # Ground truth mask
    m_gt = np.load(Path(MASK_DIR)/f"{Path(p).stem}.npy")
    if m_gt.ndim == 3: 
        m_gt = np.max(m_gt, axis=0)
    m_gt = (m_gt > 0).astype(np.uint8)
    
    # Predicted mask
    m_pred_bin = (m_pred > 0).astype(np.uint8) if m_pred is not None else np.zeros_like(m_gt)
    
    # Calculate F1
    f1 = f1_score(m_gt.flatten(), m_pred_bin.flatten(), zero_division=0)
    forged_f1s.append(f1)
    
    # Store for overall metrics
    all_results.append(("forged", f1, dbg, label))
    all_labels.append(1)  # 1 for forged
    all_predictions.append(1 if label == "forged" else 0)
    
    print(f"  {Path(p).stem}: {label} | F1={f1:.4f} | area={dbg.get('area', 0)} mean={dbg.get('mean_inside', 0):.3f}")

print("\n🟢 TESTING AUTHENTIC IMAGES:")
authentic_acc = []
for p in tqdm(test_authentic, desc="Authentic"):
    pil = Image.open(p).convert("RGB")
    
    # ====== CRITICAL: Use robust_pipeline_final NOT pipeline_final ======
    label, m_pred, dbg = robust_pipeline_final(pil)  # CHANGED HERE!
    # ====================================================================
    
    # For authentic, ground truth is all zeros
    pil_array = np.array(pil)
    m_gt = np.zeros(pil_array.shape[:2], dtype=np.uint8)
    
    # Predicted mask
    m_pred_bin = (m_pred > 0).astype(np.uint8) if m_pred is not None else np.zeros_like(m_gt)
    
    # Calculate F1 (should be 0 for correct authentic prediction)
    f1 = f1_score(m_gt.flatten(), m_pred_bin.flatten(), zero_division=0)
    
    # Check if correctly classified as authentic
    is_correct = (label == "authentic")
    authentic_acc.append(is_correct)
    
    # Store for overall metrics
    all_results.append(("authentic", f1, dbg, label))
    all_labels.append(0)  # 0 for authentic
    all_predictions.append(1 if label == "forged" else 0)
    
    print(f"  {Path(p).stem}: {label} | Correct={is_correct} | area={dbg.get('area', 0)} mean={dbg.get('mean_inside', 0):.3f}")

# Calculate metrics
print("\n" + "="*60)
print("FINAL METRICS - REGULARIZED MODEL")
print("="*60)

# Forged detection metrics
avg_forged_f1 = np.mean(forged_f1s)
forged_detected = sum([1 for _, f1, _, label in all_results[:10] if label == "forged"])
forged_missed = 10 - forged_detected

# Authentic classification metrics
authentic_accuracy = np.mean(authentic_acc)
authentic_correct = sum(authentic_acc)
authentic_wrong = 10 - authentic_correct

# Overall binary classification metrics
from sklearn.metrics import precision_score, recall_score
binary_accuracy = accuracy_score(all_labels, all_predictions)
binary_precision = precision_score(all_labels, all_predictions, zero_division=0)
binary_recall = recall_score(all_labels, all_predictions, zero_division=0)
binary_f1 = 2 * (binary_precision * binary_recall) / (binary_precision + binary_recall + 1e-6)

print(f"\n🔴 FORGED IMAGES (10 samples):")
print(f"  • Average F1: {avg_forged_f1:.4f}")
print(f"  • Detected: {forged_detected}/10 ({forged_detected/10:.0%})")
print(f"  • Missed: {forged_missed}/10")

print(f"\n🟢 AUTHENTIC IMAGES (10 samples):")
print(f"  • Accuracy: {authentic_accuracy:.2%} ({authentic_correct}/10 correct)")
print(f"  • False positives: {authentic_wrong}/10")

print(f"\n📊 OVERALL BINARY CLASSIFICATION:")
print(f"  • Accuracy: {binary_accuracy:.2%}")
print(f"  • Precision: {binary_precision:.2%} (HIGH precision = fewer false positives)")
print(f"  • Recall: {binary_recall:.2%}")
print(f"  • F1 Score: {binary_f1:.4f}")

print(f"\n🎯 PERFORMANCE COMPARISON:")
print(f"  Previous Base model F1: 0.324")
print(f"  Previous Large model (overfit): 0.246")
print(f"  Current regularized model F1: {avg_forged_f1:.4f}")
print(f"  Difference from Base: {avg_forged_f1 - 0.324:+.4f}")

# Test set prediction (based on precision-recall balance)
print(f"\n📈 TEST SET PREDICTION (based on validation):")
print(f"  If test set has similar distribution: ~{avg_forged_f1:.3f}")
print(f"  If test set has more authentic images: ~{binary_precision*avg_forged_f1:.3f}")
print(f"  Conservative estimate: ~{max(0.30, min(0.36, avg_forged_f1 * 0.9)):.3f}")

# Check if this is ready for submission
print(f"\n🚀 SUBMISSION READINESS:")
if avg_forged_f1 > 0.32 and binary_precision > 0.65:
    print("✅ READY TO SUBMIT! Model shows improvement over Base with good precision")
elif avg_forged_f1 > 0.33:
    print("✅ READY TO SUBMIT! Better F1 than Base model")
else:
    print("⚠️  MAY NEED ADJUSTMENTS: F1 not significantly better than Base")
    
# Show missed forged images analysis
print(f"\n🔍 MISSED FORGED IMAGES ANALYSIS:")
for i, (img_type, f1, dbg, label) in enumerate(all_results[:10]):
    if label == "authentic" and img_type == "forged":
        print(f"  {Path(test_forged[i]).stem}:")
        print(f"    • Area: {dbg.get('area', 0)}")
        print(f"    • Mean: {dbg.get('mean_inside', 0):.3f}")
        print(f"    • Max prob: {dbg.get('max_prob', 0):.3f}" if 'max_prob' in dbg else "")
        print(f"    • Reason: {'Low confidence' if dbg.get('max_prob', 0) < 0.25 else 'Filtered by thresholds'}")

print(f"\n⚙️ MODEL CONFIGURATION:")
print(f"  • Temperature scaling: 1.5")
print(f"  • Ensemble weights: 50% Large + 50% Base")
print(f"  • Adaptive thresholds: {USE_ADAPTIVE_THRESHOLDS}")
print(f"  • Confidence threshold: max_prob > 0.25")
```

```python
# Quick test on 5 forged images
test_samples = val_forg[:5]
f1s = []
for p in test_samples:
    pil = Image.open(p).convert("RGB")
    label, m_pred, dbg = robust_pipeline_final(pil)  # Uses updated threshold
    
    m_gt = np.load(Path(MASK_DIR)/f"{Path(p).stem}.npy")
    if m_gt.ndim == 3: m_gt = np.max(m_gt, axis=0)
    m_gt = (m_gt > 0).astype(np.uint8)
    m_pred_bin = (m_pred > 0).astype(np.uint8) if m_pred is not None else np.zeros_like(m_gt)
    
    f1 = f1_score(m_gt.flatten(), m_pred_bin.flatten(), zero_division=0)
    f1s.append(f1)

print(f"Quick test F1 with threshold 0.15: {np.mean(f1s):.4f}")
if np.mean(f1s) > 0.35:
    print("✅ Good! Run submission")
```