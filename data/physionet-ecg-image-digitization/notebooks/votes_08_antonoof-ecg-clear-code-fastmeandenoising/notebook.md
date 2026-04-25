# ECG clear code FastMeanDenoising

- **Author:** Antonoof
- **Votes:** 216
- **Ref:** antonoof/ecg-clear-code-fastmeandenoising
- **URL:** https://www.kaggle.com/code/antonoof/ecg-clear-code-fastmeandenoising
- **Last run:** 2025-12-24 11:53:20.690000

---

<div style="
    background: linear-gradient(135deg, #0c0c2e 0%, #1a1a4a 50%, #2d1a4a 100%);
    border: 2px solid #6366f1;
    border-radius: 15px;
    padding: 20px;
    margin: 15px 0;
    box-shadow: 0 0 25px rgba(99, 102, 241, 0.3),
                inset 0 0 15px rgba(255, 255, 255, 0.1);
    color: #e2e8f0;
    font-family: 'Segoe UI', system-ui, sans-serif;
    position: relative;
    overflow: hidden;
">

<div style="
    position: absolute;
    bottom: -30px;
    left: -30px;
    width: 80px;
    height: 80px;
    background: radial-gradient(circle, rgba(168, 85, 247, 0.3) 0%, transparent 70%);
    border-radius: 50%;
"></div>

<h1 style="
    color: #818cf8;
    margin-top: 0;
    text-align: center;
    font-weight: 600;
    text-shadow: 0 0 10px rgba(129, 140, 248, 0.5);
    position: relative;
    z-index: 1;
">
    Import libraries
</h1>

```python
import os
import cv2
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import albumentations as A
import torch.optim as optim

from scipy import signal as scipy_signal
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
```

<div style="
    background: linear-gradient(135deg, #0c0c2e 0%, #1a1a4a 50%, #2d1a4a 100%);
    border: 2px solid #6366f1;
    border-radius: 15px;
    padding: 20px;
    margin: 15px 0;
    box-shadow: 0 0 25px rgba(99, 102, 241, 0.3),
                inset 0 0 15px rgba(255, 255, 255, 0.1);
    color: #e2e8f0;
    font-family: 'Segoe UI', system-ui, sans-serif;
    position: relative;
    overflow: hidden;
">

<div style="
    position: absolute;
    bottom: -30px;
    left: -30px;
    width: 80px;
    height: 80px;
    background: radial-gradient(circle, rgba(168, 85, 247, 0.3) 0%, transparent 70%);
    border-radius: 50%;
"></div>

<h1 style="
    color: #818cf8;
    margin-top: 0;
    text-align: center;
    font-weight: 600;
    text-shadow: 0 0 10px rgba(129, 140, 248, 0.5);
    position: relative;
    z-index: 1;
">
    Config(u can change parameters)
</h1>

```python
class Config:
    batch_size = 10
    epochs = 1
    lr = 1e-4
    num_workers = 2
    img_size = (1240, 1024)
    target_length = 5000

config = Config()
```

<div style="
    background: linear-gradient(135deg, #0c0c2e 0%, #1a1a4a 50%, #2d1a4a 100%);
    border: 2px solid #6366f1;
    border-radius: 15px;
    padding: 20px;
    margin: 15px 0;
    box-shadow: 0 0 25px rgba(99, 102, 241, 0.3),
                inset 0 0 15px rgba(255, 255, 255, 0.1);
    color: #e2e8f0;
    font-family: 'Segoe UI', system-ui, sans-serif;
    position: relative;
    overflow: hidden;
">

<div style="
    position: absolute;
    bottom: -30px;
    left: -30px;
    width: 80px;
    height: 80px;
    background: radial-gradient(circle, rgba(168, 85, 247, 0.3) 0%, transparent 70%);
    border-radius: 50%;
"></div>

<h1 style="
    color: #818cf8;
    margin-top: 0;
    text-align: center;
    font-weight: 600;
    text-shadow: 0 0 10px rgba(129, 140, 248, 0.5);
    position: relative;
    z-index: 1;
">
    ECG image processor: preprocessing + extraction of 12-lead signals
</h1>

```python
class ECGImageProcessor:
    
    def __init__(self):
        self.lead_positions = {
            'I': (0.1, 0.15), 'II': (0.1, 0.3), 'III': (0.1, 0.45),
            'aVR': (0.1, 0.6), 'aVL': (0.1, 0.75), 'aVF': (0.1, 0.9),
            'V1': (0.55, 0.15), 'V2': (0.55, 0.3), 'V3': (0.55, 0.45),
            'V4': (0.55, 0.6), 'V5': (0.55, 0.75), 'V6': (0.55, 0.9)
        }
    
    def preprocess_image(self, image):
        """Basic ECG image preprocessing."""
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # Remove grid lines
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
        opened = cv2.morphologyEx(enhanced, cv2.MORPH_OPEN, kernel)
        
        # Binarize
        _, binary = cv2.threshold(opened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return binary
    
    def extract_lead_signal(self, image, lead_name, fs=500):
        """Extracting a specific lead signal."""
        try:
            # Get lead position
            h, w = image.shape[:2]
            x_ratio, y_ratio = self.lead_positions[lead_name]
            lead_x = int(w * x_ratio)
            lead_y = int(h * y_ratio)
            
            # Define ROI around lead
            roi_width = int(w * 0.4)
            roi_height = int(h * 0.08)
            roi_x = max(0, lead_x - roi_width//2)
            roi_y = max(0, lead_y - roi_height//2)
            
            roi = image[roi_y:roi_y+roi_height, roi_x:roi_x+roi_width]
            
            if roi.size == 0:
                return np.zeros(config.target_length)
            
            # Find signal line (dark pixels)
            signal_y = []
            for col in range(roi.shape[1]):
                column = roi[:, col]
                dark_pixels = np.where(column < 128)[0]
                if len(dark_pixels) > 0:
                    signal_y.append(np.mean(dark_pixels))
                else:
                    signal_y.append(roi.shape[0] / 2)
            
            if not signal_y:
                return np.zeros(config.target_length)
            
            # Convert to signal
            ecg_signal = np.array(signal_y)
            
            # Invert and normalize
            ecg_signal = roi_height - ecg_signal  # Invert y-axis
            ecg_signal = (ecg_signal - ecg_signal.mean()) / (ecg_signal.std() + 1e-8)
            
            # Resample to target length
            if len(ecg_signal) > 0:
                ecg_signal = scipy_signal.resample(ecg_signal, config.target_length)
            
            return ecg_signal.astype(np.float32)
            
        except Exception as e:
            print(f"Error extracting {lead_name}: {e}")
            return np.zeros(config.target_length)
```

<div style="
    background: linear-gradient(135deg, #0c0c2e 0%, #1a1a4a 50%, #2d1a4a 100%);
    border: 2px solid #6366f1;
    border-radius: 15px;
    padding: 20px;
    margin: 15px 0;
    box-shadow: 0 0 25px rgba(99, 102, 241, 0.3),
                inset 0 0 15px rgba(255, 255, 255, 0.1);
    color: #e2e8f0;
    font-family: 'Segoe UI', system-ui, sans-serif;
    position: relative;
    overflow: hidden;
">

<div style="
    position: absolute;
    bottom: -30px;
    left: -30px;
    width: 80px;
    height: 80px;
    background: radial-gradient(circle, rgba(168, 85, 247, 0.3) 0%, transparent 70%);
    border-radius: 50%;
"></div>

<h1 style="
    color: #818cf8;
    margin-top: 0;
    text-align: center;
    font-weight: 600;
    text-shadow: 0 0 10px rgba(129, 140, 248, 0.5);
    position: relative;
    z-index: 1;
">
    Electrocardiography Dataset
</h1>

```python
class ECGDataset(Dataset):
    
    def __init__(self, df, image_dir, transform=None, is_train=True):
        self.df = df
        self.image_dir = image_dir
        self.transform = transform
        self.is_train = is_train
        self.processor = ECGImageProcessor()
        self.leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        base_id = row['id']
        
        # Load image
        if self.is_train:
            img_path = os.path.join(self.image_dir, str(base_id), f"{base_id}-0001.png")
        else:
            img_path = os.path.join(self.image_dir, f"{base_id}.png")
            
        image = cv2.imread(img_path)
        if image is None:
            # Try alternative images
            if self.is_train:
                for seg in ['0003', '0004', '0005']:
                    img_path = os.path.join(self.image_dir, str(base_id), f"{base_id}-{seg}.png")
                    image = cv2.imread(img_path)
                    if image is not None:
                        break
            
        if image is None:
            # Create dummy image as last resort
            image = np.ones((1240, 1024, 3), dtype=np.uint8) * 255 # img_size
            print(f"Could not load image for {base_id}")
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.is_train:
            # Load ground truth from CSV
            csv_path = os.path.join(self.image_dir, str(base_id), f"{base_id}.csv")
            try:
                signals_df = pd.read_csv(csv_path)
                # Use lead II as target for training
                target_signal = signals_df['II'].values.astype(np.float32)
                
                # Resize to target length
                if len(target_signal) > config.target_length:
                    target_signal = target_signal[:config.target_length]
                else:
                    target_signal = np.pad(target_signal, (0, config.target_length - len(target_signal)), 
                                         mode='constant')
                
                # Normalize
                if target_signal.std() > 0:
                    target_signal = (target_signal - target_signal.mean()) / target_signal.std()
                    
            except Exception as e:
                print(f"Error loading CSV for {base_id}: {e}")
                # Create synthetic ECG as fallback
                t = np.linspace(0, 10, config.target_length)
                target_signal = (np.sin(2 * np.pi * 1 * t) + 
                               0.5 * np.sin(2 * np.pi * 2 * t) +
                               0.2 * np.sin(2 * np.pi * 0.5 * t))
                target_signal = target_signal.astype(np.float32)
            
            # Process image to extract features
            processed_img = self.processor.preprocess_image(image_rgb)
            
            # Extract lead II signal from image (this will be our input feature)
            extracted_signal = self.processor.extract_lead_signal(processed_img, 'II')
            
            # Prepare image for CNN
            if self.transform:
                image_tensor = self.transform(image=image_rgb)['image']
            else:
                # Default transform
                image_tensor = torch.from_numpy(
                    cv2.resize(image_rgb, config.img_size).transpose(2, 0, 1)
                ).float() / 255.0
            
            return image_tensor, torch.FloatTensor(extracted_signal), torch.FloatTensor(target_signal), base_id
            
        else:
            # For test - just return image
            if self.transform:
                image_tensor = self.transform(image=image_rgb)['image']
            else:
                image_tensor = torch.from_numpy(
                    cv2.resize(image_rgb, config.img_size).transpose(2, 0, 1)
                ).float() / 255.0
            
            return image_tensor, base_id
```

<div style="
    background: linear-gradient(135deg, #0c0c2e 0%, #1a1a4a 50%, #2d1a4a 100%);
    border: 2px solid #6366f1;
    border-radius: 15px;
    padding: 20px;
    margin: 15px 0;
    box-shadow: 0 0 25px rgba(99, 102, 241, 0.3),
                inset 0 0 15px rgba(255, 255, 255, 0.1);
    color: #e2e8f0;
    font-family: 'Segoe UI', system-ui, sans-serif;
    position: relative;
    overflow: hidden;
">

<div style="
    position: absolute;
    bottom: -30px;
    left: -30px;
    width: 80px;
    height: 80px;
    background: radial-gradient(circle, rgba(168, 85, 247, 0.3) 0%, transparent 70%);
    border-radius: 50%;
"></div>

<h1 style="
    color: #818cf8;
    margin-top: 0;
    text-align: center;
    font-weight: 600;
    text-shadow: 0 0 10px rgba(129, 140, 248, 0.5);
    position: relative;
    z-index: 1;
">
    Model architecture: Image to Signal Regression
</h1>

```python
class ECGNet(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        # CNN for image features
        self.cnn = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Block 2  
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Block 3
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        # Regression_head
        self.regressor = nn.Sequential(
            nn.Linear(256 * 4 * 4, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(), 
            nn.Dropout(0.3),
            nn.Linear(512, config.target_length)
        )
        
    def forward(self, x):
        # CNN features
        features = self.cnn(x)
        features = features.view(features.size(0), -1)
        
        # Regression
        output = self.regressor(features)
        return output
```

<div style="
    background: linear-gradient(135deg, #0c0c2e 0%, #1a1a4a 50%, #2d1a4a 100%);
    border: 2px solid #6366f1;
    border-radius: 15px;
    padding: 20px;
    margin: 15px 0;
    box-shadow: 0 0 25px rgba(99, 102, 241, 0.3),
                inset 0 0 15px rgba(255, 255, 255, 0.1);
    color: #e2e8f0;
    font-family: 'Segoe UI', system-ui, sans-serif;
    position: relative;
    overflow: hidden;
">

<div style="
    position: absolute;
    bottom: -30px;
    left: -30px;
    width: 80px;
    height: 80px;
    background: radial-gradient(circle, rgba(168, 85, 247, 0.3) 0%, transparent 70%);
    border-radius: 50%;
"></div>

<h1 style="
    color: #818cf8;
    margin-top: 0;
    text-align: center;
    font-weight: 600;
    text-shadow: 0 0 10px rgba(129, 140, 248, 0.5);
    position: relative;
    z-index: 1;
">
    ECG Loss Function
</h1>

```python
class ECGLoss(nn.Module):
    
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        snr = self.snr_loss(pred, target)
        smooth = self.smoothness_loss(pred)
        return snr + 0.1 * smooth  # smoothness as a regularizer

    def snr_loss(self, pred, target):
        noise = target - pred
        
        signal_power = torch.sum(target ** 2, dim=1)
        noise_power = torch.sum(noise ** 2, dim=1)
        
        snr = signal_power / (noise_power + self.eps)
        
        return -torch.mean(torch.log(snr + self.eps))

    def smoothness_loss(self, signal):
        if signal.size(1) < 3:
            return torch.tensor(0.0, device=signal.device)
        
        diff1 = signal[:, 1:] - signal[:, :-1]
        diff2 = diff1[:, 1:] - diff1[:, :-1]
        
        return torch.mean(diff2 ** 2)
```

<div style="
    background: linear-gradient(135deg, #0c0c2e 0%, #1a1a4a 50%, #2d1a4a 100%);
    border: 2px solid #6366f1;
    border-radius: 15px;
    padding: 20px;
    margin: 15px 0;
    box-shadow: 0 0 25px rgba(99, 102, 241, 0.3),
                inset 0 0 15px rgba(255, 255, 255, 0.1);
    color: #e2e8f0;
    font-family: 'Segoe UI', system-ui, sans-serif;
    position: relative;
    overflow: hidden;
">

<div style="
    position: absolute;
    bottom: -30px;
    left: -30px;
    width: 80px;
    height: 80px;
    background: radial-gradient(circle, rgba(168, 85, 247, 0.3) 0%, transparent 70%);
    border-radius: 50%;
"></div>

<h1 style="
    color: #818cf8;
    margin-top: 0;
    text-align: center;
    font-weight: 600;
    text-shadow: 0 0 10px rgba(129, 140, 248, 0.5);
    position: relative;
    z-index: 1;
">
    Model training
</h1>

```python
train_df = pd.read_csv('/kaggle/input/physionet-ecg-image-digitization/train.csv')
test_df = pd.read_csv('/kaggle/input/physionet-ecg-image-digitization/test.csv')

train_df = train_df.head(500)  # higher -> better results
    
# Transforms
train_transform = A.Compose([
    A.Resize(*config.img_size),
    A.HorizontalFlip(p=0.3),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=5, p=0.3),
    A.GridDistortion(p=0.1),  # imitation of paper distortion
    A.GaussNoise(p=0.2),
    A.RandomBrightnessContrast(p=0.3),
    A.MotionBlur(p=0.1),      # simulate blurring when photographing
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])
    
# Dataset
train_dataset = ECGDataset(
    train_df, 
    '/kaggle/input/physionet-ecg-image-digitization/train',
    transform=train_transform,
    is_train=True
)
    
train_loader = DataLoader(
    train_dataset, 
    batch_size=config.batch_size, 
    shuffle=True, 
    num_workers=config.num_workers
)
    
# Model
model = ECGNet()

# Two GPU's
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPU's")
    model = nn.DataParallel(model)

model = model.to(device)
    
# Loss and optimizer
criterion = ECGLoss()
optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
```

```python
best_loss = float('inf')
for epoch in range(config.epochs):
    model.train()
    running_loss = 0.0
        
    for batch_idx, (images, extracted, targets, base_ids) in enumerate(train_loader):
        images = images.to(device)
        targets = targets.to(device)
            
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
            
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
            
        running_loss += loss.item()
        
    epoch_loss = running_loss / len(train_loader)
    scheduler.step(epoch_loss)
        
    print(f'Epoch {epoch+1}/{config.epochs}, Loss: {epoch_loss:.4f}')
        
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        torch.save(model.state_dict(), 'best_ecg_model.pth')
```

<div style="
    background: linear-gradient(135deg, #0c0c2e 0%, #1a1a4a 50%, #2d1a4a 100%);
    border: 2px solid #6366f1;
    border-radius: 15px;
    padding: 20px;
    margin: 15px 0;
    box-shadow: 0 0 25px rgba(99, 102, 241, 0.3),
                inset 0 0 15px rgba(255, 255, 255, 0.1);
    color: #e2e8f0;
    font-family: 'Segoe UI', system-ui, sans-serif;
    position: relative;
    overflow: hidden;
">

<div style="
    position: absolute;
    bottom: -30px;
    left: -30px;
    width: 80px;
    height: 80px;
    background: radial-gradient(circle, rgba(168, 85, 247, 0.3) 0%, transparent 70%);
    border-radius: 50%;
"></div>

<h1 style="
    color: #818cf8;
    margin-top: 0;
    text-align: center;
    font-weight: 600;
    text-shadow: 0 0 10px rgba(129, 140, 248, 0.5);
    position: relative;
    z-index: 1;
">
    Predict, create submission and check results
</h1>

```python
from scipy.signal import butter, filtfilt

def smooth_ecg(x, fs=500, lowcut=0.5, highcut=40):
    nyq = 0.5 * fs
    b, a = butter(2, [lowcut / nyq, highcut / nyq], btype='band')
    return filtfilt(b, a, x)
```

```python
model.load_state_dict(torch.load('best_ecg_model.pth', map_location=device)) # Load best weights
test_df = pd.read_csv('/kaggle/input/physionet-ecg-image-digitization/test.csv')
    
# Test transform
test_transform = A.Compose([
    A.Resize(*config.img_size),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])
    
test_dataset = ECGDataset(
    test_df, 
    '/kaggle/input/physionet-ecg-image-digitization/test',
    transform=test_transform,
    is_train=False
)
    
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
submission_data = []
processor = ECGImageProcessor()
    
model.eval()
    
for batch_idx, (images, base_ids) in enumerate(test_loader):
    if batch_idx >= len(test_df):  # Safety check
        break
            
    test_row = test_df.iloc[batch_idx]
    base_id = test_row['id']
    lead = test_row['lead']
    num_rows = test_row['number_of_rows']
        
    images = images.to(device)
        
    with torch.no_grad():
        prediction = model(images).cpu().numpy().flatten()
        
    # Adjust length to required number of rows
    if len(prediction) > num_rows:
        prediction = prediction[:num_rows]
    elif len(prediction) < num_rows:
        prediction = np.pad(prediction, (0, num_rows - len(prediction)), mode='edge')
    
    prediction = smooth_ecg(prediction, fs=test_row['fs']) # New feature
        
    for row_id in range(num_rows):
        composite_id = f"{base_id}_{row_id}_{lead}"
        submission_data.append({
            'id': composite_id,
            'value': float(prediction[row_id])
        })

submission = pd.DataFrame(submission_data)
submission.to_csv('submission1.csv', index=False)
submission.head(30)
```

## second model

```python
!pip uninstall -y tensorflow
!uv pip install --no-deps --system --no-index --find-links='/kaggle/input/hengck23-submit-physionet/hengck23-submit-physionet/setup' 'connected-components-3d'
```

```python
import kagglehub
deterministic = kagglehub.package_import('wasupandceacar/deterministic').deterministic
deterministic.init_all(35, disable_list=['cuda_block'])

import sys
sys.path.append('/kaggle/input/hengck23-submit-physionet/hengck23-submit-physionet')

import os
import gc
import cv2
import torch
import traceback
import numpy as np
import pandas as pd

from tqdm import tqdm
from pathlib import Path
from tqdm.auto import tqdm
from shutil import copyfile
from scipy.signal import resample

from stage0_model import Net as Stage0Net
from stage0_common import *

if_submit = os.getenv('KAGGLE_IS_COMPETITION_RERUN')

if if_submit:
    test_meta = Path("/kaggle/input/physionet-ecg-image-digitization/test.csv")
    test_dir = Path("/kaggle/input/physionet-ecg-image-digitization/test")
else:
    test_meta = Path("/kaggle/input/physio-test-fake-dataset/test_fake/test.csv")
    test_dir = Path("/kaggle/input/physio-test-fake-dataset/test_fake")

valid_df = pd.read_csv(test_meta)
valid_df['id'] = valid_df['id'].astype(str) 
valid_id = valid_df['id'].unique().tolist()


global_dict = {
    "stage0_dir": "/kaggle/working/stage0",
    "stage1_dir": "/kaggle/working/stage1",
    "stage2_dir": "/kaggle/working/stage2",
}

def change_color(image_rgb):
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, h=10)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
    contrast_enhanced = clahe.apply(denoised)
    return cv2.cvtColor(contrast_enhanced, cv2.COLOR_GRAY2RGB)

stage0_dir = Path(global_dict["stage0_dir"])
stage0_dir.mkdir(exist_ok=True)

stage0_net = Stage0Net(pretrained=False)
stage0_net = load_net(stage0_net, '/kaggle/input/hengck23-submit-physionet/hengck23-submit-physionet/weight/stage0-last.checkpoint.pth')
stage0_net.to("cuda:0")
stage0_net.eval()

for n, sample_id in enumerate(tqdm(valid_id)):
    path = test_dir / f'{sample_id}.png'
    output_path = stage0_dir / f'{sample_id}.png'
    
    image_original = cv2.imread(str(path), cv2.IMREAD_COLOR)
    image_original = cv2.cvtColor(image_original, cv2.COLOR_BGR2RGB)
    image_for_model = change_color(image_original)
    
    batch = image_to_batch(image_for_model)

    try:
        with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.float32):
            output = stage0_net(batch)
        
        rotated, keypoint = output_to_predict(image_original, batch, output)
        normalised, _, _ = normalise_by_homography(rotated, keypoint)
        
        cv2.imwrite(str(output_path), cv2.cvtColor(normalised, cv2.COLOR_RGB2BGR))
    except Exception as e:
        traceback.print_exc()
        copyfile(path, output_path)


from stage1_model import Net as Stage1Net
from stage1_common import *

stage0_dir = Path(global_dict["stage0_dir"])
stage1_dir = Path(global_dict["stage1_dir"])
stage1_dir.mkdir(exist_ok=True)

stage1_net = Stage1Net(pretrained=False)
stage1_net = load_net(stage1_net, '/kaggle/input/hengck23-submit-physionet/hengck23-submit-physionet/weight/stage1-last.checkpoint.pth')
stage1_net.to("cuda:0")

for n, sample_id in enumerate(tqdm(valid_id)):
    path = stage0_dir / f'{sample_id}.png'
    output_path = stage1_dir / f'{sample_id}.png'
    image = cv2.imread(path, cv2.IMREAD_COLOR_RGB)
    batch = {'image': torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).unsqueeze(0)}

    try:
        with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.float32):
            output = stage1_net(batch)
        gridpoint_xy, _ = output_to_predict(image, batch, output)
        rectified = rectify_image(image, gridpoint_xy)
        cv2.imwrite(output_path, cv2.cvtColor(rectified, cv2.COLOR_RGB2BGR))
    except:
        traceback.print_exc()
        copyfile(path, output_path)

import torchvision.transforms as T
from stage2_model import *
from stage2_common import *
from scipy.signal import savgol_filter, medfilt


class Net3(nn.Module):
    
    def __init__(self, pretrained=True):
        super(Net3, self).__init__()
        encoder_dim = [64, 128, 256, 512]
        decoder_dim = [128, 64, 32, 16]

        self.encoder = timm.create_model(
            model_name='resnet34.a3_in1k', pretrained=pretrained, in_chans=3, num_classes=0, global_pool=''
        )

        self.decoder = MyCoordUnetDecoder(
            in_channel=encoder_dim[-1],
            skip_channel=encoder_dim[:-1][::-1] + [0],
            out_channel=decoder_dim,
            scale=[2, 2, 2, 2]
        )
        self.pixel = nn.Conv2d(decoder_dim[-1], 4, 1)

    def forward(self, image):
        encode = encode_with_resnet(self.encoder, image)
        last, _ = self.decoder(feature=encode[-1], skip=encode[:-1][::-1] + [None])
        pixel = self.pixel(last)
        return pixel

stage1_dir = Path(global_dict["stage1_dir"])
stage2_dir = Path(global_dict["stage2_dir"])
stage2_dir.mkdir(exist_ok=True)

stage2_net = Net3(pretrained=False).to("cuda:0")
model_path = "/kaggle/input/physio-seg-public/pytorch/net3_009_4200/1/iter_0004200.pt"
stage2_net.load_state_dict(torch.load(model_path))
stage2_net.eval()

x0, x1 = 0, 2176
y0, y1 = 0, 1696
zero_mv = [703.5, 987.5, 1271.5, 1531.5]
mv_to_pixel = 78.5
t0, t1 = 235, 4161

resize = T.Resize((1696, 4352), interpolation=T.InterpolationMode.BILINEAR)

for n, sample_id in enumerate(tqdm(valid_id)):
    path = stage1_dir / f'{sample_id}.png'
    output_path = stage2_dir / f'{sample_id}.npy'
    image = cv2.imread(path, cv2.IMREAD_COLOR_RGB)
    
    length = valid_df[(valid_df['id']==sample_id) & (valid_df['lead']=='II')].iloc[0].number_of_rows
    
    image = image[y0:y1, x0:x1] / 255
    batch = resize(torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).unsqueeze(0)).float().to("cuda:0")
    
    try:
        with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.float32):
            output = stage2_net(batch)
        
        pixel = torch.sigmoid(output).float().data.cpu().numpy()[0]
        series_in_pixel = pixel_to_series(pixel[..., t0:t1], zero_mv, length)
        series = (np.array(zero_mv).reshape(4, 1) - series_in_pixel) / mv_to_pixel
        
       
        for i in range(series.shape[0]):
            series[i] = savgol_filter(series[i], window_length=7, polyorder=2)

        np.save(output_path, series)
    except:
        traceback.print_exc()
        series = np.zeros((4, length)) 
        np.save(output_path, series)

def series_dict(series):
    d = {}
    for l in range(3):
        lead_names = [
            ['I',   'aVR', 'V1', 'V4'],
            ['II',  'aVL', 'V2', 'V5'],
            ['III', 'aVF', 'V3', 'V6'],
        ][l]
        split = np.array_split(series[l], 4)
        for (k, s) in zip(lead_names, split):
            d[k] = s
    
    d['II'] = series[3]
    
    return d


stage2_dir = Path(global_dict["stage2_dir"])

res = []
gb = valid_df.groupby('id')

for i, (sample_id, df) in enumerate(tqdm(gb)):
    series = np.load(stage2_dir / f'{sample_id}.npy')
    d_series = series_dict(series)

    for _, d in df.iterrows():
        s = d_series.get(d.lead, np.zeros(d.number_of_rows))
        
        if len(s) != d.number_of_rows:
            x_old = np.linspace(0, 1, len(s))
            x_new = np.linspace(0, 1, d.number_of_rows)
            s = np.interp(x_new, x_old, s)
        
        row_id = [f'{sample_id}_{x}_{d.lead}' for x in range(d.number_of_rows)]
        res.append(pd.DataFrame({'id': row_id, 'value': s}))

    if i % 100 == 0:
        gc.collect()

submission = pd.concat(res, axis=0, ignore_index=True)
submission.to_csv('submission2.csv', index=False)
submission.head(30)
```

```python
sub1 = pd.read_csv('submission1.csv')
sub2 = pd.read_csv('submission2.csv')

eps = 1e-69

weight_sub1 = eps
weight_sub2 = 1 - eps

sub2['value'] = weight_sub1 * sub1['value'] + weight_sub2 * sub2['value']
sub2.to_csv('submission.csv', index=False)
sub2.head(30)
```