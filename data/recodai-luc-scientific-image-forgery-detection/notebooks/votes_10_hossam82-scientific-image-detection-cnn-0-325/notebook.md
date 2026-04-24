# 🔥 Scientific image detection  | CNN | 0.325

- **Author:** Hossam Hamouda
- **Votes:** 126
- **Ref:** hossam82/scientific-image-detection-cnn-0-325
- **URL:** https://www.kaggle.com/code/hossam82/scientific-image-detection-cnn-0-325
- **Last run:** 2026-01-06 10:25:10.410000

---

# Scientific Image Forgery Detection with ELA
## CNN-DINOv2 Hybrid + Error Level Analysis

**ELA (Error Level Analysis)** reveals hidden image manipulations by analyzing JPEG compression artifacts.
Like "adding powder to paper" to reveal fingerprints, ELA highlights regions with different compression histories.

```python
import os, cv2, json, math, random, torch, io
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
DINO_PATH = "/kaggle/input/dinov2/pytorch/base/1"

IMG_SIZE = 518
BATCH_SIZE = 2
MODEL_LOC = '/kaggle/input/cnndinov2-pbd/CNNDINOv2-U52/CNNDINOv2-U52/model_seg_final.pt'

# INFERENCE UTILS
AREA_THR = 200
MEAN_THR = 0.22
USE_TTA = False
USE_ELA = True  # NEW: Enable ELA
ELA_WEIGHT = 0.3  # Weight for ELA contribution
ELA_QUALITY = 90  # JPEG quality for ELA

print(f'Device: {device}')
print(f'ELA Enabled: {USE_ELA}')
```

## Error Level Analysis (ELA) Functions

ELA works by:
1. Re-compressing image at known JPEG quality
2. Computing difference between original and re-compressed
3. Forged regions show different error levels

```python
def compute_ela(pil_image, quality=90):
    """Compute Error Level Analysis for an image.
    
    Like adding forensic powder to reveal hidden fingerprints,
    ELA reveals hidden manipulations by analyzing compression artifacts.
    
    Args:
        pil_image: PIL Image in RGB format
        quality: JPEG quality for re-compression (default 90)
    
    Returns:
        PIL Image showing ELA result
    """
    # Re-compress at specified quality
    buffer = io.BytesIO()
    pil_image.save(buffer, 'JPEG', quality=quality)
    buffer.seek(0)
    recompressed = Image.open(buffer).convert('RGB')
    
    # Compute difference
    original = np.array(pil_image, dtype=np.float32)
    compressed = np.array(recompressed, dtype=np.float32)
    ela = np.abs(original - compressed)
    
    # Scale for visibility (amplify differences)
    scale = 255.0 / (ela.max() + 1e-6)
    ela = np.clip(ela * scale, 0, 255).astype(np.uint8)
    
    return Image.fromarray(ela)


def compute_ela_heatmap(pil_image, quality=90):
    """Compute ELA and return as grayscale heatmap."""
    ela_img = compute_ela(pil_image, quality)
    ela_arr = np.array(ela_img, dtype=np.float32)
    # Convert to grayscale by taking max channel
    heatmap = ela_arr.max(axis=2)
    # Normalize
    heatmap = heatmap / (heatmap.max() + 1e-6)
    return heatmap


def visualize_ela(pil_image, quality=90):
    """Visualize original image and its ELA side by side."""
    ela = compute_ela(pil_image, quality)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(pil_image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(ela)
    axes[1].set_title(f'ELA (Quality={quality})')
    axes[1].axis('off')
    
    # Heatmap version
    heatmap = compute_ela_heatmap(pil_image, quality)
    axes[2].imshow(heatmap, cmap='hot')
    axes[2].set_title('ELA Heatmap')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()
```

```python
def detect_hidden_jpeg(pil_image):
    """Detect if a PNG/lossless image was likely originally a JPEG.
    
    Checks for 8x8 block artifacts using a simple horizontal/vertical gradient check.
    """
    arr = np.array(pil_image.convert('L'), dtype=np.float32)
    h, w = arr.shape
    if h < 16 or w < 16: return False, 0.0
    
    # Compute gradients
    grad_x = np.abs(arr[:, 1:] - arr[:, :-1])
    grad_y = np.abs(arr[1:, :] - arr[:-1, :])
    
    # Check for 8x8 periodic peaks in gradient sums (block boundaries)
    def check_periodicity(grads, axis=0):
        sums = grads.sum(axis=axis)
        if len(sums) < 16: return 0.0
        # Look at the 8-cycle variance vs local mean
        peaks = [sums[i::8].mean() for i in range(8)]
        return np.max(peaks) / (np.mean(peaks) + 1e-6)
    
    score_x = check_periodicity(grad_x, axis=0)
    score_y = check_periodicity(grad_y, axis=1)
    
    final_score = (score_x + score_y) / 2.0
    is_hjpeg = final_score > 1.2 # Empirical threshold
    
    return is_hjpeg, final_score

def visualize_ela_suitability(pil_image):
    is_hjt, score = detect_hidden_jpeg(pil_image)
    ela = compute_ela(pil_image, quality=ELA_QUALITY)
    
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(pil_image)
    plt.title(f"Original (H-JPEG: {is_hjt}, Score: {score:.2f})")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(ela)
    plt.title("ELA Transformation")
    plt.axis('off')
    plt.show()
```

```python
# MODEL (DINOv2 + Decoder)
processor = AutoImageProcessor.from_pretrained(DINO_PATH, local_files_only=True, use_fast=False)
encoder = AutoModel.from_pretrained(DINO_PATH, local_files_only=True).eval().to(device)

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
    
class DinoSegmenter(nn.Module):
    def __init__(self, encoder, processor):
        super().__init__()
        self.encoder, self.processor = encoder, processor
        for p in self.encoder.parameters(): p.requires_grad = False
        self.seg_head = DinoTinyDecoder(768,1)
        
    def forward_features(self, x):
        imgs = (x*255).clamp(0,255).byte().permute(0,2,3,1).cpu().numpy()
        inputs = self.processor(images=list(imgs), return_tensors="pt").to(x.device)
        feats = self.encoder(**inputs).last_hidden_state
        B, N, C = feats.shape
        fmap = feats[:,1:,:].permute(0,2,1)
        s = int(math.sqrt(N-1))
        fmap = fmap.reshape(B, C, s, s)
        return fmap
        
    def forward_seg(self, x):
        fmap = self.forward_features(x)
        return self.seg_head(fmap, (IMG_SIZE, IMG_SIZE))

model_seg = DinoSegmenter(encoder, processor).to(device)

if MODEL_LOC is not None and os.path.exists(MODEL_LOC):
    model_seg.load_state_dict(torch.load(MODEL_LOC, map_location=device))
    print(f"✅ Loaded pretrained model from: {MODEL_LOC}")
    model_seg.eval()
```

```python
@torch.no_grad()
def segment_prob_map(pil):
    x = torch.from_numpy(np.array(pil.resize((IMG_SIZE, IMG_SIZE)), np.float32)/255.).permute(2,0,1)[None].to(device)
    prob = torch.sigmoid(model_seg.forward_seg(x))[0,0].cpu().numpy()
    return prob

@torch.no_grad()
def segment_prob_map_with_tta(pil):
    x = torch.from_numpy(np.array(pil.resize((IMG_SIZE, IMG_SIZE)), np.float32)/255.).permute(2,0,1)[None].to(device)
    predictions = []
    pred_orig = torch.sigmoid(model_seg.forward_seg(x))
    predictions.append(pred_orig)
    pred_h = torch.sigmoid(model_seg.forward_seg(torch.flip(x, dims=[3])))
    predictions.append(torch.flip(pred_h, dims=[3]))
    pred_v = torch.sigmoid(model_seg.forward_seg(torch.flip(x, dims=[2])))
    predictions.append(torch.flip(pred_v, dims=[2]))
    prob = torch.stack(predictions).mean(0)[0, 0].cpu().numpy()
    return prob

def enhanced_adaptive_mask(prob, alpha_grad=0.45):
    gx = cv2.Sobel(prob, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(prob, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx**2 + gy**2)
    grad_norm = grad_mag / (grad_mag.max() + 1e-6)
    enhanced = (1 - alpha_grad) * prob + alpha_grad * grad_norm
    enhanced = cv2.GaussianBlur(enhanced, (3,3), 0)
    thr = np.mean(enhanced) + 0.3 * np.std(enhanced)
    mask = (enhanced > thr).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
    return mask, thr

def finalize_mask(prob, orig_size):
    mask, thr = enhanced_adaptive_mask(prob)
    mask = cv2.resize(mask, orig_size, interpolation=cv2.INTER_NEAREST)
    return mask, thr
```

## Enhanced Pipeline with ELA Integration

The new pipeline combines:
1. **RGB prediction** - Standard model prediction
2. **ELA prediction** - Model run on ELA-transformed image
3. **ELA heatmap boost** - Direct ELA signal to highlight compression anomalies

```python
def pipeline_final_with_ela(pil, use_ela=True, ela_weight=0.3):
    """Enhanced pipeline with ELA integration.
    
    Like adding forensic powder to paper, ELA reveals hidden manipulations
    by highlighting regions with different compression histories.
    """
    # Get RGB prediction
    if USE_TTA:
        prob_rgb = segment_prob_map_with_tta(pil)
    else:
        prob_rgb = segment_prob_map(pil)
    
    prob_combined = prob_rgb.copy()
    ela_info = {}
    
    if use_ela:
        # Compute ELA image
        ela_img = compute_ela(pil, quality=ELA_QUALITY)
        
        # Get model prediction on ELA image
        if USE_TTA:
            prob_ela = segment_prob_map_with_tta(ela_img)
        else:
            prob_ela = segment_prob_map(ela_img)
        
        # Get direct ELA heatmap (resized to model output size)
        ela_heatmap = compute_ela_heatmap(pil, quality=ELA_QUALITY)
        ela_heatmap = cv2.resize(ela_heatmap, (IMG_SIZE, IMG_SIZE))
        
        # Combine: RGB prob + weighted ELA prediction + small ELA heatmap boost
        prob_combined = (
            (1 - ela_weight) * prob_rgb + 
            ela_weight * 0.7 * prob_ela + 
            ela_weight * 0.3 * ela_heatmap  # Direct ELA signal
        )
        
        ela_info = {
            'ela_max': float(ela_heatmap.max()),
            'ela_mean': float(ela_heatmap.mean()),
        }
    
    mask, thr = finalize_mask(prob_combined, pil.size)
    area = int(mask.sum())
    mean_inside = float(prob_combined[cv2.resize(mask,(IMG_SIZE,IMG_SIZE),interpolation=cv2.INTER_NEAREST)==1].mean()) if area>0 else 0.0
    
    if area < AREA_THR or mean_inside < MEAN_THR:
        return "authentic", None, {"area": area, "mean_inside": mean_inside, "thr": thr, **ela_info}
    return "forged", mask, {"area": area, "mean_inside": mean_inside, "thr": thr, **ela_info}


def pipeline_final(pil):
    """Wrapper to use ELA-enhanced pipeline if enabled."""
    return pipeline_final_with_ela(pil, use_ela=USE_ELA, ela_weight=ELA_WEIGHT)
```

## Test ELA on Sample Images

```python
# Test ELA visualization on forged images
if os.path.exists(FORG_DIR):
    forg_imgs = sorted([str(Path(FORG_DIR)/f) for f in os.listdir(FORG_DIR)])[:3]
    
    for img_path in forg_imgs:
        print(f"\n📸 {Path(img_path).name}")
        pil = Image.open(img_path).convert("RGB")
        visualize_ela_suitability(pil)
```

## A/B Comparison: ELA vs No-ELA

Let's compare the performance on a few samples to see if ELA provides a boost.

```python
if os.path.exists(FORG_DIR):
    test_p = sorted([str(Path(FORG_DIR)/f) for f in os.listdir(FORG_DIR)])[5]
    pil = Image.open(test_p).convert("RGB")
    
    # 1. No ELA
    l1, m1, d1 = pipeline_final_with_ela(pil, use_ela=False)
    
    # 2. With ELA
    l2, m2, d2 = pipeline_final_with_ela(pil, use_ela=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(pil)
    axes[0].set_title("Original")
    
    if m1 is not None: axes[1].imshow(m1)
    axes[1].set_title(f"No ELA (Area: {d1['area']})")
    
    if m2 is not None: axes[2].imshow(m2)
    axes[2].set_title(f"With ELA (Area: {d2['area']})")
    
    plt.show()
```

## Inference on Test Set

```python
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

SAMPLE_SUB = f"{BASE_DIR}/sample_submission.csv"
OUT_PATH = "submission.csv"

rows = []
for f in tqdm(sorted(os.listdir(TEST_DIR)), desc="Inference on Test Set with ELA"):
    pil = Image.open(Path(TEST_DIR)/f).convert("RGB")
    label, mask, dbg = pipeline_final(pil)
    
    if mask is None:
        mask = np.zeros(pil.size[::-1], np.uint8)
    else:
        mask = np.array(mask, dtype=np.uint8)
    
    if label == "authentic":
        annot = "authentic"
    else:
        annot = rle_encode((mask > 0).astype(np.uint8))
    
    rows.append({
        "case_id": Path(f).stem,
        "annotation": annot,
        "area": int(dbg.get("area", mask.sum())),
        "mean": float(dbg.get("mean_inside", 0.0)),
        "thr": float(dbg.get("thr", 0.0))
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
```