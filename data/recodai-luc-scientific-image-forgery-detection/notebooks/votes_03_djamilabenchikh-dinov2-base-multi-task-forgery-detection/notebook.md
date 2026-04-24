# DINOv2-Base Multi-Task Forgery Detection

- **Author:** Djamila Benchikh
- **Votes:** 288
- **Ref:** djamilabenchikh/dinov2-base-multi-task-forgery-detection
- **URL:** https://www.kaggle.com/code/djamilabenchikh/dinov2-base-multi-task-forgery-detection
- **Last run:** 2025-11-04 17:29:58.090000

---

## 🧠 DINOv2 Multi-Task Pipeline

This notebook implements a **multi-task architecture** based on **DINOv2** —  
a frozen vision backbone that powers two lightweight heads:  
**Segmentation** (detecting manipulated regions) and  
**Classification** (authentic vs forged).

---

### ⚙️ Architecture Overview
- **Encoder** → DINOv2-Base (frozen feature extractor)  
- **Segmentation Head** → compact Conv–ReLU–Conv decoder  
- **Classification Head** → linear layer over the CLS token  
- **Optimization** → `AdamW` + `BCEWithLogitsLoss` / `CrossEntropyLoss`  
- **Metrics** → IoU, Dice, Pixel Accuracy  

---

### 🔄 Pipeline Steps
1. **Dataset Loading** → authentic / forged images + mask files (`.npy`)  
2. **Feature Extraction** → DINOv2 encoder  
3. **Segmentation** → binary mask prediction of forged regions  
4. **Classification** → authenticity decision (A / F)  
5. **Evaluation** → IoU, Dice & Accuracy on validation set  

---

> 🧩 **Note:** The model runs in **Offline Kaggle** mode —  
> DINOv2 is loaded from `/kaggle/input/dinov2/pytorch/base/1`.

```python
import os, gc, json, math, random
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import cv2

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Repro & device

def seed_everything(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
seed_everything(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths (adapte DINO_PATH)
BASE_DIR  = "/kaggle/input/recodai-luc-scientific-image-forgery-detection"
AUTH_DIR  = f"{BASE_DIR}/train_images/authentic"
FORG_DIR  = f"{BASE_DIR}/train_images/forged"
MASK_DIR  = f"{BASE_DIR}/train_masks"
TEST_DIR  = f"{BASE_DIR}/test_images"
SAMPLE_SUB = f"{BASE_DIR}/sample_submission.csv"

# DINOv2 local folder (contient config.json, preprocessor_config.json, pytorch_model.bin)
DINO_PATH = "/kaggle/input/dinov2/pytorch/base/1"

# Hyperparams
IMG_SIZE = 256
BATCH_SEG = 4
BATCH_CLS = 32
EPOCHS_SEG = 1
EPOCHS_CLS = 1
LR_SEG = 3e-4
LR_CLS = 1e-3
WEIGHT_DECAY = 1e-4


# Utils: metrics + RLE
def binarize(x, thr=0.5):
    return (x > thr).astype(np.uint8)

def iou_score(pred, gt, eps=1e-7):
    p = binarize(pred); g = binarize(gt)
    inter = (p & g).sum()
    union = (p | g).sum()
    return float(inter) / (float(union) + eps)

def dice_score(pred, gt, eps=1e-7):
    p = binarize(pred); g = binarize(gt)
    inter = (p & g).sum()
    return float(2*inter) / (p.sum() + g.sum() + eps)

def pixel_acc(pred, gt, thr=0.5):
    p = binarize(pred, thr); g = binarize(gt, thr)
    return float((p == g).sum()) / float(np.prod(g.shape))

def rle_encode_numpy(mask):
    """Kaggle-compatible RLE: mask: (H,W) uint8 {0,1}, column-major."""
    pixels = mask.T.flatten()
    dots = np.where(pixels == 1)[0]
    if len(dots) == 0:
        return "[]"  # JSON empty list (competition variant allows JSON-encoded runs)
    run_lengths, prev = [], -2
    for b in dots:
        if b > prev + 1:
            run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return json.dumps(run_lengths)

# -------------------------
# Dataset: segmentation
# - For forged: charge .npy ; si ndim==3 -> max proj
# - For authentic: masque vide
# -------------------------
class ForgerySegDataset(Dataset):
    def __init__(self, auth_paths, forg_paths, mask_dir, img_size=256):
        self.samples = []
        self.mask_dir = mask_dir
        self.img_size = img_size

        for p in auth_paths:
            self.samples.append((p, None))  # None => masque vide

        for p in forg_paths:
            stem = Path(p).stem
            m = os.path.join(mask_dir, stem + ".npy")
            self.samples.append((p, m if os.path.exists(m) else None))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        # mask
        if (mask_path is None):
            mask = np.zeros((h, w), dtype=np.uint8)
        else:
            m = np.load(mask_path)
            if m.ndim == 3:
                m = np.max(m, axis=0)
            mask = m.astype(np.uint8)
            if mask.shape != (h, w):
                # si jamais (rare), on redimensionne
                mask = np.array(Image.fromarray(mask).resize((w, h), Image.NEAREST), dtype=np.uint8)

        # resize to common size
        img_r = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        mask_r = np.array(Image.fromarray(mask).resize((IMG_SIZE, IMG_SIZE), Image.NEAREST), dtype=np.uint8)

        # to tensor
        img_t = torch.from_numpy(np.array(img_r, dtype=np.float32)/255.).permute(2,0,1)  # [3,H,W]
        mask_t = torch.from_numpy(mask_r[None, ...].astype(np.float32))                  # [1,H,W] in {0.,1.}
        return img_t, mask_t


# Liste des fichiers

auth_imgs = sorted([str(Path(AUTH_DIR)/f) for f in os.listdir(AUTH_DIR) if f.lower().endswith((".png",".jpg",".jpeg",".tif"))])
forg_imgs = sorted([str(Path(FORG_DIR)/f) for f in os.listdir(FORG_DIR) if f.lower().endswith((".png",".jpg",".jpeg",".tif"))])

# Split identique pour seg/cls
train_auth, val_auth = train_test_split(auth_imgs, test_size=0.2, random_state=42)
train_forg, val_forg = train_test_split(forg_imgs, test_size=0.2, random_state=42)


# DINOv2 local (offline)

from transformers import AutoImageProcessor, AutoModel
try:
    processor = AutoImageProcessor.from_pretrained(DINO_PATH, local_files_only=True)
    dino_encoder = AutoModel.from_pretrained(DINO_PATH, local_files_only=True)
except Exception as e:
    # Fallback minimal (sans internet) si jamais AutoModel échoue — on stoppe proprement
    raise RuntimeError(f"Impossible de charger DINOv2 localement depuis {DINO_PATH}: {e}")

dino_encoder.eval().to(device)
# Dimensions patch: on déduit la grille (H_p, W_p) depuis image_size & patch_size si dispo
cfg = getattr(dino_encoder, "config", None)
patch = getattr(cfg, "patch_size", 14) if cfg else 14  # typical ViT patch size
inp = getattr(processor, "size", {"shortest_edge": 224})
if isinstance(inp, dict):
    # pour ViT/DINO processors, on force 224x224
    proc_w = inp.get("width", 224); proc_h = inp.get("height", 224)
    if "shortest_edge" in inp: 
        proc_w = proc_h = int(inp["shortest_edge"])
else:
    proc_w = proc_h = 224
grid_h = proc_h // patch
grid_w = proc_w // patch
# si CLS token présent: tokens == grid_h*grid_w + 1


# Décodeur segmentation léger

class DinoTinyDecoder(nn.Module):
    def __init__(self, in_ch, out_ch=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_ch, 1)
        )
    def forward(self, f, out_size):
        x = nn.functional.interpolate(f, size=out_size, mode="bilinear", align_corners=False)
        return self.conv(x)


# Modèle complet: DINO frozen + heads
#  - Seg head (conv decoder)
#  - Cls head (linear sur pooled feat)
class DinoMultiTask(nn.Module):
    def __init__(self, encoder, processor, seg_ch=768, freeze=True):
        super().__init__()
        self.encoder = encoder
        self.processor = processor
        if freeze:
            for p in self.encoder.parameters():
                p.requires_grad = False
        self.seg_head = DinoTinyDecoder(in_ch=seg_ch, out_ch=1)
        self.cls_head = nn.Linear(seg_ch, 2)

    def forward_features(self, images):  # images: [B,3,H,W] en [0,1]
        B,_,H,W = images.shape
        mean = torch.tensor(self.processor.image_mean, device=images.device).view(1,3,1,1)
        std  = torch.tensor(self.processor.image_std,  device=images.device).view(1,3,1,1)
        x = (images - mean) / std

        # Prépare entrée processor (224x224 typiquement)
        imgs = (x*255).clamp(0,255).byte().permute(0,2,3,1).cpu().numpy()
        inputs = self.processor(images=list(imgs), return_tensors="pt").to(images.device)

        with torch.no_grad():
            feats = self.encoder(**inputs).last_hidden_state  # [B, N, C] (N = grid_h*grid_w [+1])
        B, N, C = feats.shape
        # Retire CLS si présent
        expected_tokens = grid_h*grid_w
        if N == expected_tokens + 1:
            feats_spatial = feats[:, 1:, :]  # drop CLS
            cls_token = feats[:, 0, :]       # [B,C]
        elif N == expected_tokens:
            feats_spatial = feats
            cls_token = feats.mean(dim=1)
        else:
            # Sécurisation: déduire s via sqrt en tenant compte éventuel CLS
            n_wo_cls = N-1 if int(math.isclose(math.sqrt(N-1), round(math.sqrt(N-1)))) else N
            if n_wo_cls == N-1:
                feats_spatial = feats[:,1:,:]; cls_token = feats[:,0,:]
            else:
                feats_spatial = feats; cls_token = feats.mean(dim=1)
            s = int(round(math.sqrt(feats_spatial.shape[1])))
            # override grid dims pour cohérence locale
            # (pas nécessaire pour la suite car on reshape dynamiquement)
        s = int(round(math.sqrt(feats_spatial.shape[1])))
        fmap = feats_spatial.permute(0,2,1).reshape(B, C, s, s)  # [B,C,s,s]
        return fmap, cls_token

    def forward_seg(self, images):
        fmap, _ = self.forward_features(images)
        B,_,H,W = images.shape
        logits = self.seg_head(fmap, out_size=(H,W))
        return logits

    def forward_cls(self, images):
        _, cls_token = self.forward_features(images)
        return self.cls_head(cls_token)

model = DinoMultiTask(dino_encoder, processor, seg_ch=768, freeze=True).to(device)


# Datasets & Loaders
train_seg_ds = ForgerySegDataset(train_auth, train_forg, MASK_DIR, img_size=IMG_SIZE)
val_seg_ds   = ForgerySegDataset(val_auth,   val_forg,   MASK_DIR, img_size=IMG_SIZE)

train_seg_loader = DataLoader(train_seg_ds, batch_size=BATCH_SEG, shuffle=True, num_workers=2, pin_memory=True)
val_seg_loader   = DataLoader(val_seg_ds,   batch_size=BATCH_SEG, shuffle=False, num_workers=2, pin_memory=True)

# Pour classification, on réutilise les mêmes images: label 0 (auth) / 1 (forg)
class ForgeryClsDataset(Dataset):
    def __init__(self, auth_paths, forg_paths, img_size=256):
        self.items = [(p,0) for p in auth_paths] + [(p,1) for p in forg_paths]
        self.img_size = img_size
    def __len__(self): return len(self.items)
    def __getitem__(self, idx):
        p, y = self.items[idx]
        img = Image.open(p).convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        x = torch.from_numpy(np.array(img, dtype=np.float32)/255.).permute(2,0,1)
        return x, torch.tensor(y, dtype=torch.long)

train_cls_ds = ForgeryClsDataset(train_auth, train_forg, IMG_SIZE)
val_cls_ds   = ForgeryClsDataset(val_auth,   val_forg, IMG_SIZE)
train_cls_loader = DataLoader(train_cls_ds, batch_size=BATCH_CLS, shuffle=True,  num_workers=2, pin_memory=True)
val_cls_loader   = DataLoader(val_cls_ds,   batch_size=BATCH_CLS, shuffle=False, num_workers=2, pin_memory=True)


# Train segmentation head

crit_seg = nn.BCEWithLogitsLoss()
opt_seg  = optim.AdamW(model.seg_head.parameters(), lr=LR_SEG, weight_decay=WEIGHT_DECAY)

for epoch in range(1, EPOCHS_SEG+1):
    model.train()
    tr_loss = 0.0
    for imgs, masks in tqdm(train_seg_loader, desc=f"[Seg] Epoch {epoch}/{EPOCHS_SEG}"):
        imgs, masks = imgs.to(device), masks.to(device)
        logits = model.forward_seg(imgs)
        loss = crit_seg(logits, masks)
        opt_seg.zero_grad(); loss.backward(); opt_seg.step()
        tr_loss += loss.item() * imgs.size(0)
    tr_loss /= len(train_seg_loader.dataset)

    # Val
    model.eval()
    val_loss, miou, mdice, macc = 0.0, 0.0, 0.0, 0.0
    with torch.no_grad():
        for imgs, masks in val_seg_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            logits = model.forward_seg(imgs)
            loss = crit_seg(logits, masks)
            val_loss += loss.item() * imgs.size(0)
            probs = torch.sigmoid(logits).cpu().numpy()
            gts   = masks.cpu().numpy()
            for p,g in zip(probs, gts):
                p = p[0]; g = g[0]
                miou  += iou_score(p, g)
                mdice += dice_score(p, g)
                macc  += pixel_acc(p, g)
    n_val = len(val_seg_loader.dataset)
    print(f"[Seg] Epoch {epoch} | train_loss={tr_loss:.4f} | val_loss={val_loss/n_val:.4f} "
          f"| IoU={miou/n_val:.3f} | Dice={mdice/n_val:.3f} | Acc={macc/n_val:.3f}")

torch.cuda.empty_cache(); gc.collect()


# Train classification head

crit_cls = nn.CrossEntropyLoss()
opt_cls  = optim.AdamW(model.cls_head.parameters(), lr=LR_CLS, weight_decay=WEIGHT_DECAY)

for epoch in range(1, EPOCHS_CLS+1):
    model.train()
    tr_loss = 0.0
    for imgs, labels in tqdm(train_cls_loader, desc=f"[Cls] Epoch {epoch}/{EPOCHS_CLS}"):
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model.forward_cls(imgs)
        loss = crit_cls(logits, labels)
        opt_cls.zero_grad(); loss.backward(); opt_cls.step()
        tr_loss += loss.item() * imgs.size(0)
    tr_loss /= len(train_cls_loader.dataset)

    # Val
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in val_cls_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model.forward_cls(imgs)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
    print(f"[Cls] Epoch {epoch} | train_loss={tr_loss:.4f} | val_acc={100.0*correct/total:.2f}%")

torch.cuda.empty_cache(); gc.collect()
```

🚀 Inference & Submission Phase

```python
# Inference + Submission (segmentation -> RLE, sinon 'authentic')

def predict_mask_prob(img_pil):
    img = img_pil.convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    x = torch.from_numpy(np.array(img, dtype=np.float32)/255.).permute(2,0,1)[None].to(device)
    with torch.no_grad():
        logits = model.forward_seg(x)
        prob = torch.sigmoid(logits)[0,0].cpu().numpy()
    return prob

THR = 0.5
rows = []
if os.path.exists(TEST_DIR):
    test_files = sorted(os.listdir(TEST_DIR))
    print("Test images:", len(test_files))
    for fname in tqdm(test_files, desc="Inference"):
        case_id = Path(fname).stem
        path = str(Path(TEST_DIR)/fname)
        pil = Image.open(path).convert("RGB")
        ow, oh = pil.size

        prob = predict_mask_prob(pil)  # IMG_SIZE x IMG_SIZE
        # resize to original size
        mask = cv2.resize(prob, (ow, oh), interpolation=cv2.INTER_NEAREST)
        binm = (mask > THR).astype(np.uint8)

        if binm.sum() == 0:
            annot = "authentic"
        else:
            annot = rle_encode_numpy(binm)  # JSON list of [start,length,...]

        rows.append({"case_id": case_id, "annotation": annot})

sub = pd.DataFrame(rows, columns=["case_id","annotation"])

# Aligne avec sample_submission
if os.path.exists(SAMPLE_SUB):
    ss = pd.read_csv(SAMPLE_SUB)
    ss["case_id"] = ss["case_id"].astype(str)
    if not sub.empty:
        sub["case_id"] = sub["case_id"].astype(str)
        sub = ss[["case_id"]].merge(sub, on="case_id", how="left")
        sub["annotation"] = sub["annotation"].fillna("authentic")
    else:
        sub = ss[["case_id"]].copy()
        sub["annotation"] = "authentic"

# Sauvegarde
OUT_PATH = "submission.csv"
sub.to_csv(OUT_PATH, index=False)
print(" Wrote submission:", OUT_PATH)
print(sub.head())
```

🖼️ Forgery Mask Visualization

```python
import matplotlib.pyplot as plt


# Function: predict probability mask

def predict_mask_prob(img_pil):
    """Predict a pixel-wise probability mask for a given image."""
    img = img_pil.convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    x = torch.from_numpy(np.array(img, dtype=np.float32) / 255.).permute(2, 0, 1)[None].to(device)
    with torch.no_grad():
        logits = model.forward_seg(x)
        prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
    return prob

# Visualization on random test images

N_SHOW = 5  # number of test images to visualize
THR = 0.5   # threshold for binary mask

if os.path.exists(TEST_DIR):
    test_files = sorted(os.listdir(TEST_DIR))
    samples = random.sample(test_files, min(N_SHOW, len(test_files)))

    plt.figure(figsize=(14, N_SHOW * 4))
    for i, fname in enumerate(samples, 1):
        path = str(Path(TEST_DIR) / fname)
        pil = Image.open(path).convert("RGB")
        ow, oh = pil.size

        # Predict probability mask
        prob = predict_mask_prob(pil)
        mask_resized = cv2.resize(prob, (ow, oh), interpolation=cv2.INTER_NEAREST)
        binary_mask = (mask_resized > THR).astype(np.uint8)

        # Visualization: original / prob mask / overlay
        plt.subplot(N_SHOW, 3, 3*(i-1)+1)
        plt.imshow(pil)
        plt.title(f"Original - {fname}")
        plt.axis("off")

        plt.subplot(N_SHOW, 3, 3*(i-1)+2)
        plt.imshow(mask_resized, cmap="viridis")
        plt.title("Probability Mask")
        plt.axis("off")

        plt.subplot(N_SHOW, 3, 3*(i-1)+3)
        plt.imshow(pil)
        plt.imshow(binary_mask, cmap="Reds", alpha=0.4)
        plt.title("Overlay Mask")
        plt.axis("off")

    plt.tight_layout()
    plt.show()
```

🧩 Authentic Image Mask Visualization

```python
import matplotlib.pyplot as plt

# Function: predict mask on tensor batch

def predict_val_batch(model, loader, n_show=5, thr=0.5):
    """Display a few validation images with GT mask and predicted mask."""
    model.eval()
    shown = 0

    plt.figure(figsize=(12, n_show * 4))

    with torch.no_grad():
        for imgs, masks in loader:
            imgs, masks = imgs.to(device), masks.to(device)
            logits = model.forward_seg(imgs)
            probs = torch.sigmoid(logits).cpu().numpy()
            imgs_np = imgs.cpu().permute(0, 2, 3, 1).numpy()
            gts = masks.cpu().numpy()

            for j in range(len(imgs)):
                if shown >= n_show:
                    break

                # Retrieve elements
                img = np.clip(imgs_np[j], 0, 1)
                gt = gts[j, 0]
                prob = probs[j, 0]
                pred_bin = (prob > thr).astype(np.uint8)

                # Show three panels
                plt.subplot(n_show, 3, 3 * shown + 1)
                plt.imshow(img)
                plt.title("Original")
                plt.axis("off")

                plt.subplot(n_show, 3, 3 * shown + 2)
                plt.imshow(gt, cmap="gray")
                plt.title("Ground Truth")
                plt.axis("off")

                plt.subplot(n_show, 3, 3 * shown + 3)
                plt.imshow(img)
                plt.imshow(pred_bin, cmap="Reds", alpha=0.4)
                plt.title("Predicted Mask")
                plt.axis("off")

                shown += 1

            if shown >= n_show:
                break

    plt.tight_layout()
    plt.show()

# Run visualization
predict_val_batch(model, val_seg_loader, n_show=5, thr=0.5)
```

🔥 Forged Image Mask Visualization

```python
import matplotlib.pyplot as plt

def show_forged_examples(model, loader, n_show=5, thr=0.5):
    """Show only forged examples from validation set (mask not empty)."""
    model.eval()
    shown = 0

    plt.figure(figsize=(12, n_show * 4))

    with torch.no_grad():
        for imgs, masks in loader:
            imgs, masks = imgs.to(device), masks.to(device)
            logits = model.forward_seg(imgs)
            probs = torch.sigmoid(logits).cpu().numpy()
            imgs_np = imgs.cpu().permute(0, 2, 3, 1).numpy()
            gts = masks.cpu().numpy()

            for j in range(len(imgs)):
                gt = gts[j, 0]
                if gt.sum() == 0:  # skip authentic images
                    continue

                img = np.clip(imgs_np[j], 0, 1)
                prob = probs[j, 0]
                pred_bin = (prob > thr).astype(np.uint8)

                # 3 panels: original / GT / predicted
                plt.subplot(n_show, 3, 3 * shown + 1)
                plt.imshow(img)
                plt.title("Original (Forged)")
                plt.axis("off")

                plt.subplot(n_show, 3, 3 * shown + 2)
                plt.imshow(gt, cmap="gray")
                plt.title("Ground Truth Mask")
                plt.axis("off")

                plt.subplot(n_show, 3, 3 * shown + 3)
                plt.imshow(img)
                plt.imshow(pred_bin, cmap="Reds", alpha=0.4)
                plt.title("Predicted Mask")
                plt.axis("off")

                shown += 1
                if shown >= n_show:
                    break

            if shown >= n_show:
                break

    plt.tight_layout()
    plt.show()

#  Run visualization
show_forged_examples(model, val_seg_loader, n_show=5, thr=0.5)
```