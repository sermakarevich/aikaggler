# Scientific Image Forgery Detection | DINOv2

- **Author:** Mahdi Ravaghi
- **Votes:** 462
- **Ref:** ravaghi/scientific-image-forgery-detection-dinov2
- **URL:** https://www.kaggle.com/code/ravaghi/scientific-image-forgery-detection-dinov2
- **Last run:** 2026-01-14 16:38:45.157000

---

# Imports and configs

```python
!pip uninstall -qy tensorflow
```

```python
from transformers import AutoImageProcessor, AutoModel
from torch.utils.data import Dataset
from tqdm.notebook import tqdm
from pathlib import Path
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import warnings
import torch
import json
import math
import cv2
import os

warnings.filterwarnings("ignore")
```

```python
class CFG:
    test_images_path = "/kaggle/input/recodai-luc-scientific-image-forgery-detection/test_images"
    sample_sub_path = "/kaggle/input/recodai-luc-scientific-image-forgery-detection/sample_submission.csv"

    dino_path = "/kaggle/input/dinov2/pytorch/base/1"
    dino_weights_path = "/kaggle/input/m/ravaghi/dinov2/pytorch/base/1/model.pt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_size = 512
    
    use_tta = True
```

# Model

```python
class DinoTinyDecoder(nn.Module):
    def __init__(self, in_ch=768, out_ch=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 256, 3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, out_ch, 1)
        )

    def forward(self, f, size):
        return self.net(F.interpolate(f, size=size, mode="bilinear", align_corners=False))


class DinoSegmenter(nn.Module):
    def __init__(self, encoder, processor):
        super().__init__()
        self.encoder, self.processor = encoder, processor
        
        for p in self.encoder.parameters():
            p.requires_grad = False
        
        self.seg_head = DinoTinyDecoder(768, 1)

    def forward_features(self, x):
        imgs = (x*255).clamp(0, 255).byte().permute(0, 2, 3, 1).cpu().numpy()
        inputs = self.processor(images=list(imgs), return_tensors="pt").to(x.device)
        
        with torch.no_grad():
            feats = self.encoder(**inputs).last_hidden_state
        
        B, N, C = feats.shape
        fmap = feats[:, 1:, :].permute(0, 2, 1)
        s = int(math.sqrt(N-1))
        fmap = fmap.reshape(B, C, s, s)
        
        return fmap

    def forward_seg(self, x):
        fmap = self.forward_features(x)
        return self.seg_head(fmap, (CFG.img_size, CFG.img_size))
```

```python
processor = AutoImageProcessor.from_pretrained(CFG.dino_path, local_files_only=True)
encoder = AutoModel.from_pretrained(CFG.dino_path, local_files_only=True).eval().to(CFG.device)

model = DinoSegmenter(encoder, processor).to(CFG.device)
model.load_state_dict(torch.load(CFG.dino_weights_path))
model.eval()
```

# Inference

```python
def rle_encode(mask):
    pixels = mask.T.flatten()
    dots = np.where(pixels == 1)[0]
    
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
```

```python
@torch.no_grad()
def predict_with_tta(model, image):
    predictions = []

    pred = torch.sigmoid(model.forward_seg(image))
    predictions.append(pred)

    pred = torch.sigmoid(model.forward_seg(torch.flip(image, dims=[3])))
    predictions.append(torch.flip(pred, dims=[3]))

    pred = torch.sigmoid(model.forward_seg(torch.flip(image, dims=[2])))
    predictions.append(torch.flip(pred, dims=[2]))

    return torch.stack(predictions).mean(0)[0, 0].cpu().numpy()


@torch.no_grad()
def predict(model, image):
    return torch.sigmoid(model.forward_seg(image))[0,0].cpu().numpy()


def postprocess(preds, original_size, alpha_grad=0.35):
    gx = cv2.Sobel(preds, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(preds, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx**2 + gy**2)
    grad_norm = grad_mag / (grad_mag.max() + 1e-6)
    enhanced = (1 - alpha_grad) * preds + alpha_grad * grad_norm
    enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)
    thr = np.mean(enhanced) + 0.3 * np.std(enhanced)
    mask = (enhanced > thr).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    
    mask = cv2.resize(mask, original_size, interpolation=cv2.INTER_NEAREST)
    
    return mask


def infer_image(image):
    image_array = np.array(image.resize((CFG.img_size, CFG.img_size)), np.float32) / 255
    image_array = torch.from_numpy(image_array).permute(2, 0, 1)[None].to(CFG.device)
    
    if CFG.use_tta:
        preds = predict_with_tta(model, image_array)
    else:
        preds = predict(model, image_array)
    
    mask = postprocess(preds, image.size)
    
    area = int(mask.sum())
    if area > 0:
        mean_inside = float(preds[cv2.resize(mask, (CFG.img_size, CFG.img_size), interpolation=cv2.INTER_NEAREST) == 1].mean())
    else:
        mean_inside = 0.0

    if area < 400 or mean_inside < 0.3:
        return "authentic", None    
    
    return "forged", mask
```

```python
predictions = []

for image_path in tqdm(sorted(os.listdir(CFG.test_images_path)), desc="Running Inference"):
    image = Image.open(Path(CFG.test_images_path)/image_path).convert("RGB")
    label, mask = infer_image(image)

    if mask is None:
        mask = np.zeros(image.size[::-1], np.uint8)
    else:
        mask = np.array(mask, dtype=np.uint8)

    if label == "authentic":
        annotation = "authentic"
    else:
        annotation = rle_encode((mask > 0).astype(np.uint8))

    predictions.append({
        "case_id": Path(image_path).stem,
        "annotation": annotation
    })
```

# Submission

```python
predictions = pd.DataFrame(predictions)
predictions["case_id"] = predictions["case_id"].astype(str)

submission = pd.read_csv(CFG.sample_sub_path)
submission["case_id"] = submission["case_id"].astype(str)

submission = submission[["case_id"]].merge(predictions, on="case_id", how="left")
submission["annotation"] = submission["annotation"].fillna("authentic")
submission[["case_id", "annotation"]].to_csv("submission.csv", index=False)
submission.head()
```