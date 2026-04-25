# Train 224x224 DICOM->PNGs EfficientNetV2S

- **Author:** Ichigo_E
- **Votes:** 127
- **Ref:** ichigoe/train-224x224-dicom-pngs-efficientnetv2s
- **URL:** https://www.kaggle.com/code/ichigoe/train-224x224-dicom-pngs-efficientnetv2s
- **Last run:** 2025-08-07 07:43:09.413000

---

## References
- https://www.kaggle.com/code/dennisfong/dicom-pngs-for-rsna-intracranial-aneurysm/notebook

## Notebooks
- My Train Notebook: here
- My Inference Notebook: https://www.kaggle.com/code/ichigoe/inference-224x224-dicom-pngs-efficientnetb0

## Training Overview
### What to Train:
- 5-Frame EfficientNetV2-S (Optimized CNN Architecture)
- 14-class Multi-label Classifier (13 anatomical locations + Aneurysm Present)

### What to Train With:
- 5 frames simultaneous input (224×224 PNG/DICOM)
- Weighted BCE Loss with Focal Loss (Aneurysm Present weight=3.0)

### Key Features:
- **True Patient Separation**: DICOM StudyInstanceUID-based cross-validation for patient-level separation
- **Strategic 5-frame Sampling**: Central region focus (middle 60%) where aneurysms are most common
- **3-Channel Input Processing**: Middle slice + Maximum Intensity Projection + Standard deviation projection
- **CLAHE Contrast Adaptation**: Modality-specific enhancement for CTA/MRA/MRI variations
- **Strong Augmentation**: 15° rotation, elastic transforms, noise simulation for scanner robustness
- **Robust Percentile Normalization**: Outlier-resistant preprocessing using 1st-99th percentile clipping
- **Medical Metadata Integration**: Patient age and sex features for enhanced classification
- **Modality-specific Windowing**: Optimized intensity windows (CTA: 50/350, MRA: 600/1200, MRI: 40/80)
- **Mixed Precision Training**: GPU-optimized training with gradient accumulation (batch size 8, accumulation 4)
- **LRU Caching**: Performance optimization for frequently accessed DICOM data

### Improvements:
- **Patient Leakage Prevention**: DICOM metadata extraction ensures no patient overlap between train/validation

### Evaluation Metric:
- **Weighted Multi-label AUC ROC**
- **Final Score = (Aneurysm AUC + Individual AUC Average) / 2**

### Expected Performance:
- **Improved CV/LB Alignment**: True patient separation should reduce overfitting
- **Better Generalization**: Strategic sampling and robust preprocessing for real-world variation
- **Reduced CV/LB Gap**: From ~0.44 gap to healthy 0.10-0.15 range through proper validation

```python
# Environment setup and library imports
import os
import glob
import random
import warnings
import numpy as np
import pandas as pd
import cv2
import functools
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Optional
import gc

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.metrics import roc_auc_score
import pydicom

warnings.filterwarnings('ignore')

def set_seed(seed=42):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

set_seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"CUDA version: {torch.version.cuda}")
    torch.cuda.empty_cache()
else:
    raise RuntimeError("CUDA is not available! This code requires GPU.")
```

```python
class Config:
    # Data paths
    DATA_DIR = "/kaggle/input/rsna-2025-intracranial-aneurysm-png-224x224"
    CVT_PNG_DIR = os.path.join(DATA_DIR, "cvt_png")
    SERIES_MAPPING_PATH = os.path.join(DATA_DIR, "series_index_mapping.csv")
    LOCALIZERS_PATH = os.path.join(DATA_DIR, "train_localizers_with_relative.csv")
    TRAIN_CSV_PATH = "/kaggle/input/rsna-intracranial-aneurysm-detection/train.csv"
    
    # Model parameters for 8-frame processing
    NUM_FRAMES = 8
    IMAGE_SIZE = 224
    NUM_CLASSES = 14
    BATCH_SIZE = 6  # Reduced for 8-frame processing
    NUM_EPOCHS = 50
    LEARNING_RATE = 5e-5
    
    # Model configuration
    MODEL_NAME_BACKBONE = "tf_efficientnetv2_s.in1k"
    USE_METADATA = True
    USE_WINDOWING = True
    USE_3CHANNEL_INPUT = True
    USE_IMPROVED_LOSS = True
    USE_CLAHE = True
    USE_STRONG_AUGMENTATION = True
    
    # GPU optimization settings
    NUM_WORKERS = 2
    PIN_MEMORY = True
    PREFETCH_FACTOR = 2
    PERSISTENT_WORKERS = True
    
    # Training parameters with robust cross-validation
    NUM_FOLDS = 5
    FOLD = 0
    ACCUMULATION_STEPS = 5  # Adjusted for smaller batch size
    EARLY_STOPPING_PATIENCE = 3
    USE_GROUP_CV = True
    
    # Data loading optimization
    CACHE_SIZE = 100
    
    # Output
    OUTPUT_DIR = "/kaggle/working"
    MODEL_NAME = "eightframe_efficientnetv2s"

config = Config()

print("=== Configuration Summary ===")
print(f"Model Backbone: {config.MODEL_NAME_BACKBONE}")
print(f"Number of Frames: {config.NUM_FRAMES}")
print(f"Batch Size: {config.BATCH_SIZE}")
print(f"Accumulation Steps: {config.ACCUMULATION_STEPS}")
print(f"Effective Batch Size: {config.BATCH_SIZE * config.ACCUMULATION_STEPS}")
print(f"CLAHE Enabled: {config.USE_CLAHE}")
print(f"Strong Augmentation: {config.USE_STRONG_AUGMENTATION}")
print(f"Group Cross-Validation: {config.USE_GROUP_CV}")
```

```python
# Load data
print("Loading data...")
train_df = pd.read_csv(config.TRAIN_CSV_PATH)
series_mapping_df = pd.read_csv(config.SERIES_MAPPING_PATH)
localizers_df = pd.read_csv(config.LOCALIZERS_PATH)

print(f"Train data shape: {train_df.shape}")
print(f"Series mapping shape: {series_mapping_df.shape}")
print(f"Localizers shape: {localizers_df.shape}")

# Define target columns
TARGET_COLS = [
    'Left Infraclinoid Internal Carotid Artery',
    'Right Infraclinoid Internal Carotid Artery', 
    'Left Supraclinoid Internal Carotid Artery',
    'Right Supraclinoid Internal Carotid Artery',
    'Left Middle Cerebral Artery',
    'Right Middle Cerebral Artery',
    'Anterior Communicating Artery',
    'Left Anterior Cerebral Artery',
    'Right Anterior Cerebral Artery', 
    'Left Posterior Communicating Artery',
    'Right Posterior Communicating Artery',
    'Basilar Tip',
    'Other Posterior Circulation',
    'Aneurysm Present'
]

print(f"Target columns: {len(TARGET_COLS)}")
```

```python
def get_windowing_params(modality: str) -> Tuple[float, float]:
    """Get optimal windowing parameters for different modalities"""
    windows = {
        'CT': (40, 80),
        'CTA': (50, 350), 
        'MRA': (600, 1200),
        'MRI': (40, 80),
        'MR': (40, 80)
    }
    return windows.get(modality, (40, 80))

def apply_dicom_windowing(img: np.ndarray, window_center: float, window_width: float) -> np.ndarray:
    """Apply DICOM windowing to normalize image intensities"""
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    img = np.clip(img, img_min, img_max)
    img = (img - img_min) / (img_max - img_min + 1e-7)
    return (img * 255).astype(np.uint8)

def apply_clahe_normalization(img: np.ndarray, modality: str) -> np.ndarray:
    """Apply CLAHE with modality-specific optimization"""
    if not config.USE_CLAHE:
        return img
        
    if modality in ['CTA', 'MRA']:
        # Vascular imaging: stronger contrast improvement
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        img_clahe = clahe.apply(img.astype(np.uint8))
        img_clahe = cv2.convertScaleAbs(img_clahe, alpha=1.1, beta=5)
    elif modality in ['MRI', 'MR']:
        # MRI: gentler improvement with gamma correction
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_clahe = clahe.apply(img.astype(np.uint8))
        img_clahe = np.power(img_clahe / 255.0, 0.9) * 255
        img_clahe = img_clahe.astype(np.uint8)
    else:
        # CT: standard CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        img_clahe = clahe.apply(img.astype(np.uint8))
    
    return img_clahe

def robust_normalization(volume: np.ndarray) -> np.ndarray:
    """Apply robust normalization using percentiles"""
    p1, p99 = np.percentile(volume.flatten(), [1, 99])
    volume_norm = np.clip(volume, p1, p99)
    
    if p99 > p1:
        volume_norm = (volume_norm - p1) / (p99 - p1 + 1e-7)
    else:
        volume_norm = np.zeros_like(volume_norm)
        
    return (volume_norm * 255).astype(np.uint8)
```

```python
def create_3channel_input_8frame(volume: np.ndarray) -> np.ndarray:
    """Create 3-channel input from 8-frame volume optimized for aneurysm detection"""
    if len(volume) == 0:
        return np.zeros((config.IMAGE_SIZE, config.IMAGE_SIZE, 3), dtype=np.uint8)
    
    # Middle slice (most important for anatomical reference)
    middle_slice = volume[len(volume) // 2]
    
    # Maximum Intensity Projection (MIP) - optimized for vascular structures
    mip = np.max(volume, axis=0)
    
    # Standard deviation projection for texture analysis
    std_proj = np.std(volume, axis=0).astype(np.float32)
    
    # Normalize standard deviation projection with robust method
    if std_proj.max() > std_proj.min():
        p1, p99 = np.percentile(std_proj, [5, 95])
        std_proj = np.clip(std_proj, p1, p99)
        std_proj = ((std_proj - p1) / (p99 - p1 + 1e-7) * 255).astype(np.uint8)
    else:
        std_proj = np.zeros_like(std_proj, dtype=np.uint8)
    
    return np.stack([middle_slice, mip, std_proj], axis=-1)

def smart_8_frame_sampling(volume_paths: List[str], series_uid: str = None) -> List[str]:
    """Intelligent 8-frame sampling strategy using every other frame"""
    n = len(volume_paths)
    
    if n <= 8:
        # If we have 8 or fewer frames, use all available
        result = volume_paths[:]
        # Pad with repetitions if needed
        while len(result) < 8:
            result.extend(volume_paths[:8-len(result)])
        return result[:8]
    
    # Skip every other frame starting from a strategic position
    # Start from 10% into the volume to avoid empty slices at the beginning
    start_idx = max(0, int(n * 0.1))
    
    # Calculate step size to get 8 frames with good coverage
    available_frames = n - start_idx
    step = max(1, available_frames // 8)
    
    indices = []
    current_idx = start_idx
    while len(indices) < 8 and current_idx < n:
        indices.append(current_idx)
        current_idx += step
    
    # If we need more frames, fill from the remaining
    while len(indices) < 8:
        remaining = [i for i in range(n) if i not in indices]
        if remaining:
            indices.append(remaining[len(indices) % len(remaining)])
        else:
            indices.append(indices[-1])  # Duplicate last frame
    
    return [volume_paths[i] for i in indices[:8]]
```

```python
def extract_dicom_patient_info(series_uid: str) -> Tuple[str, str]:
    """Extract StudyInstanceUID and PatientID from DICOM metadata"""
    try:
        dicom_dir = f"/kaggle/input/rsna-intracranial-aneurysm-detection/series/{series_uid}"
        if os.path.exists(dicom_dir):
            dcm_files = [f for f in os.listdir(dicom_dir) if f.endswith('.dcm')]
            if dcm_files:
                ds = pydicom.dcmread(
                    os.path.join(dicom_dir, dcm_files[0]), 
                    stop_before_pixels=True, 
                    force=True
                )
                study_uid = getattr(ds, 'StudyInstanceUID', None)
                patient_id = getattr(ds, 'PatientID', None)
                return study_uid or f"fallback_{series_uid[:32]}", patient_id
    except Exception:
        pass
    
    # Fallback: use longer prefix from series UID
    return f"fallback_{series_uid[:32]}", f"fallback_{series_uid[:32]}"

@functools.lru_cache(maxsize=5000)
def get_patient_group_cached(series_uid: str) -> str:
    """Get patient group with caching for performance"""
    study_uid, patient_id = extract_dicom_patient_info(series_uid)
    # Use StudyInstanceUID as primary identifier
    return study_uid if study_uid and not study_uid.startswith('fallback_') else patient_id

def create_frame_paths_8frame():
    """Create mapping from series to frame paths optimized for 8-frame processing"""
    frame_paths = {}
    
    print("Creating 8-frame optimized paths from series_index_mapping.csv...")
    
    for series_uid in tqdm(train_df['SeriesInstanceUID'].unique(), desc="Processing series"):
        # Get series data from mapping
        series_data = series_mapping_df[series_mapping_df['SeriesInstanceUID'] == series_uid]
        
        if len(series_data) == 0:
            frame_paths[series_uid] = []
            continue
            
        # Get row from train_df to check which diseases are present
        train_row = train_df[train_df['SeriesInstanceUID'] == series_uid].iloc[0]
        
        # Find any disease location that has this series
        found_paths = []
        
        # Check each target column (except Aneurysm Present)
        for target_col in TARGET_COLS[:-1]:
            if train_row[target_col] == 1:
                location_clean = target_col.replace('/', '_')
                series_dir = os.path.join(config.CVT_PNG_DIR, location_clean, series_uid)
                
                if os.path.exists(series_dir):
                    png_files = sorted(glob.glob(os.path.join(series_dir, "*.png")))
                    if png_files:
                        found_paths = png_files
                        break
        
        # If no paths found in disease folders, use DICOM structure
        if not found_paths:
            dicom_dir = f"/kaggle/input/rsna-intracranial-aneurysm-detection/series/{series_uid}"
            if os.path.exists(dicom_dir):
                num_frames = len(series_data)
                found_paths = [f"dummy_path_{i:04d}.png" for i in range(num_frames)]
        
        # Apply smart 8-frame sampling
        if found_paths:
            found_paths = smart_8_frame_sampling(found_paths, series_uid)
        
        frame_paths[series_uid] = found_paths
    
    return frame_paths

# Create optimized paths
frame_paths_dict = create_frame_paths_8frame()
print(f"Created 8-frame optimized paths for {len(frame_paths_dict)} series")

# Filter data
valid_series = [uid for uid, paths in frame_paths_dict.items() if len(paths) > 0]
train_df_filtered = train_df[train_df['SeriesInstanceUID'].isin(valid_series)].copy()
print(f"Filtered train data shape: {train_df_filtered.shape}")

# Check distribution
aneurysm_dist_filtered = train_df_filtered['Aneurysm Present'].value_counts()
print(f"Aneurysm Present distribution: {aneurysm_dist_filtered.to_dict()}")
```

```python
def create_robust_cv_split(train_df, n_splits=5):
    """Create robust cross-validation split with true patient separation from DICOM"""
    
    print("Creating patient-separated cross-validation split...")
    print("Extracting true patient IDs from DICOM metadata...")
    print("This will take a few minutes but ensures proper patient separation.")
    
    # Extract true patient groups from DICOM metadata
    patient_groups = []
    for series_uid in tqdm(train_df['SeriesInstanceUID'], desc="Reading DICOM patient info"):
        patient_group = get_patient_group_cached(series_uid)
        patient_groups.append(patient_group)
    
    # Add patient groups to dataframe
    train_df = train_df.copy()
    train_df['patient_id'] = patient_groups
    
    n_groups = train_df['patient_id'].nunique()
    print(f"True patient groups found: {n_groups}")
    
    # Check if we have enough patient groups
    if n_groups < n_splits:
        print(f"Not enough patient groups ({n_groups}) for {n_splits}-fold CV.")
        print("Falling back to StratifiedKFold...")
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        return list(skf.split(train_df, train_df['Aneurysm Present']))
    
    # Create stratification key combining modality and aneurysm presence
    train_df['stratify_key'] = (
        train_df['Modality'].astype(str) + '_' + 
        train_df['Aneurysm Present'].astype(str)
    )
    
    print(f"Stratification keys: {train_df['stratify_key'].unique()}")
    
    # Use GroupKFold to ensure patient-level separation
    group_kfold = GroupKFold(n_splits=n_splits)
    
    splits = []
    for fold_idx, (train_idx, val_idx) in enumerate(group_kfold.split(
        train_df, 
        groups=train_df['patient_id']
    )):
        # Validate patient separation
        train_fold = train_df.iloc[train_idx]
        val_fold = train_df.iloc[val_idx]
        
        # Check for patient overlap (should be 0)
        train_patients = set(train_fold['patient_id'])
        val_patients = set(val_fold['patient_id'])
        overlap = train_patients.intersection(val_patients)
        
        train_dist = train_fold['Aneurysm Present'].value_counts(normalize=True)
        val_dist = val_fold['Aneurysm Present'].value_counts(normalize=True)
        
        print(f"Fold {fold_idx}:")
        print(f"  Train: {len(train_fold)} samples ({len(train_patients)} patients)")
        print(f"  Val: {len(val_fold)} samples ({len(val_patients)} patients)")
        print(f"  Patient overlap: {len(overlap)} (should be 0!)")
        print(f"  Aneurysm Present - Train: {train_dist.get(1, 0):.3f}, Val: {val_dist.get(1, 0):.3f}")
        
        if len(overlap) > 0:
            print(f"  WARNING: Found {len(overlap)} overlapping patients!")
        
        splits.append((train_idx, val_idx))
    
    return splits

# Create robust train/validation split
cv_splits = create_robust_cv_split(train_df_filtered, config.NUM_FOLDS)
train_indices, val_indices = cv_splits[config.FOLD]

train_fold_df = train_df_filtered.iloc[train_indices]
val_fold_df = train_df_filtered.iloc[val_indices]

print(f"\nRobust CV Fold {config.FOLD} Summary:")
print(f"Train fold size: {len(train_fold_df)}")
print(f"Validation fold size: {len(val_fold_df)}")

# Check distributions
print(f"Train Aneurysm Present: {train_fold_df['Aneurysm Present'].value_counts().to_dict()}")
print(f"Val Aneurysm Present: {val_fold_df['Aneurysm Present'].value_counts().to_dict()}")

# Check modality distribution
print(f"Train Modality distribution: {train_fold_df['Modality'].value_counts().to_dict()}")
print(f"Val Modality distribution: {val_fold_df['Modality'].value_counts().to_dict()}")
```

```python
# Data transforms with strong augmentation
if config.USE_STRONG_AUGMENTATION:
    print("Using strong augmentation for better generalization...")
    train_transform = A.Compose([
        # Geometric transformations (safe for medical images)
        A.Rotate(limit=15, p=0.7),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=10, p=0.6),
        
        # Advanced geometric transformations for robustness
        A.ElasticTransform(alpha=50, sigma=5, p=0.3),
        A.GridDistortion(num_steps=3, distort_limit=0.1, p=0.3),
        
        # Image quality variations (simulate different scanners/protocols)
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.6),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8,8), p=0.5),
        A.RandomGamma(gamma_limit=(80, 120), p=0.4),
        
        # Noise simulation (scanner differences)
        A.GaussNoise(var_limit=(10, 80), p=0.4),
        A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.3),
        A.Blur(blur_limit=3, p=0.2),
        
        # Medical image specific augmentations
        A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.3),
        
        # Normalization
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
else:
    print("Using standard augmentation...")
    train_transform = A.Compose([
        A.Rotate(limit=10, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
        A.GaussNoise(var_limit=(10, 50), p=0.2),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

val_transform = A.Compose([
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])
```

```python
class EightFrameDataset(Dataset):
    """Dataset optimized for 8-frame processing with CLAHE"""
    def __init__(self, df, frame_paths_dict, series_mapping_df, num_frames=8, 
                 transform=None, is_training=True):
        self.df = df.reset_index(drop=True)
        self.frame_paths_dict = frame_paths_dict
        self.series_mapping_df = series_mapping_df
        self.num_frames = num_frames
        self.transform = transform
        self.is_training = is_training
        
        # Simple LRU cache for recently accessed data
        self._cache = {}
        self._cache_keys = []
        self._max_cache_size = config.CACHE_SIZE
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        # Check cache first
        if idx in self._cache:
            return self._cache[idx]
        
        row = self.df.iloc[idx]
        series_uid = row['SeriesInstanceUID']
        
        # Get labels
        labels = torch.tensor(row[TARGET_COLS].values.astype(np.float32))
        
        # Extract metadata
        metadata = self._extract_metadata(row)
        
        # Load 8-frame 3-channel image
        image = self._load_8frame_3channel_image(series_uid, row)
        
        result = (image, labels, metadata)
        
        # Update cache
        self._update_cache(idx, result)
        
        return result
    
    def _update_cache(self, idx, data):
        """Update LRU cache"""
        if len(self._cache) >= self._max_cache_size:
            # Remove oldest entry
            oldest_idx = self._cache_keys.pop(0)
            del self._cache[oldest_idx]
        
        self._cache[idx] = data
        self._cache_keys.append(idx)
    
    def _extract_metadata(self, row) -> torch.Tensor:
        """Extract and normalize metadata"""
        if not config.USE_METADATA:
            return torch.tensor([0.0, 0.0], dtype=torch.float32)
        
        # Age processing
        age = row.get('PatientAge', 50)
        if pd.isna(age):
            age = 50
        elif isinstance(age, str):
            age = int(''.join(filter(str.isdigit, age[:3])) or '50')
        age = min(float(age), 100.0) / 100.0
        
        # Sex processing
        sex = row.get('PatientSex', 'M')
        sex = 1.0 if sex == 'M' else 0.0
        
        return torch.tensor([age, sex], dtype=torch.float32)
    
    def _load_8frame_3channel_image(self, series_uid: str, row) -> torch.Tensor:
        """Load 8-frame 3-channel image with processing"""
        paths = self.frame_paths_dict.get(series_uid, [])
        
        try:
            if len(paths) == 0 or paths[0].startswith('dummy_path'):
                volume = self._load_volume_from_dicom_8frame(series_uid, row)
            else:
                volume = self._load_volume_from_png_8frame(paths)
            
            # Apply robust normalization
            volume = robust_normalization(volume)
            
            # Create 3-channel input optimized for 8 frames
            image = create_3channel_input_8frame(volume)
            
            # Apply transforms
            if self.transform:
                transformed = self.transform(image=image)
                image = transformed['image']
            
            return image
            
        except Exception as e:
            print(f"Error loading {series_uid}: {e}")
            # Return dummy image
            dummy_image = np.zeros((config.IMAGE_SIZE, config.IMAGE_SIZE, 3), dtype=np.uint8)
            if self.transform:
                transformed = self.transform(image=dummy_image)
                return transformed['image']
            return torch.zeros(3, config.IMAGE_SIZE, config.IMAGE_SIZE)
    
    def _load_volume_from_png_8frame(self, paths: List[str]) -> np.ndarray:
        """Load PNG volume optimized for 8 frames"""
        volume = []
        
        # Ensure we have exactly 8 paths
        if len(paths) != 8:
            paths = smart_8_frame_sampling(paths)
        
        for path in paths:
            try:
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img = cv2.resize(img, (config.IMAGE_SIZE, config.IMAGE_SIZE), 
                                   interpolation=cv2.INTER_AREA)
                    volume.append(img)
            except:
                volume.append(np.zeros((config.IMAGE_SIZE, config.IMAGE_SIZE), dtype=np.uint8))
        
        return np.array(volume) if volume else np.zeros((8, config.IMAGE_SIZE, config.IMAGE_SIZE), dtype=np.uint8)
    
    def _load_volume_from_dicom_8frame(self, series_uid: str, row) -> np.ndarray:
        """Load DICOM volume optimized for 8 frames with CLAHE"""
        series_data = self.series_mapping_df[
            self.series_mapping_df['SeriesInstanceUID'] == series_uid
        ].sort_values('relative_index')
        
        if len(series_data) == 0:
            return np.zeros((8, config.IMAGE_SIZE, config.IMAGE_SIZE), dtype=np.uint8)
        
        volume = []
        modality = row.get('Modality', 'CT')
        
        # Sample exactly 8 slices using every-other-frame strategy
        if len(series_data) <= 8:
            sampled_data = series_data
        else:
            # Apply smart 8-frame sampling logic to indices
            all_indices = list(range(len(series_data)))
            sampled_indices = smart_8_frame_sampling([str(i) for i in all_indices])
            sampled_indices = [int(i) for i in sampled_indices]
            sampled_data = series_data.iloc[sampled_indices]
        
        for _, dicom_row in sampled_data.iterrows():
            try:
                ds = pydicom.dcmread(dicom_row['dicom_filename'])
                img = ds.pixel_array.astype(np.float32)
                
                # Handle multi-frame/color images
                if img.ndim == 3:
                    if img.shape[-1] == 3:
                        img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
                    else:
                        img = img[:, :, 0]
                
                # Apply rescale if available
                if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
                    img = img * ds.RescaleSlope + ds.RescaleIntercept
                
                # Apply windowing
                if config.USE_WINDOWING:
                    window_center, window_width = get_windowing_params(modality)
                    img = apply_dicom_windowing(img, window_center, window_width)
                else:
                    img_min, img_max = img.min(), img.max()
                    if img_max > img_min:
                        img = ((img - img_min) / (img_max - img_min) * 255).astype(np.uint8)
                    else:
                        img = np.zeros_like(img, dtype=np.uint8)
                
                # Apply CLAHE improvement
                img = apply_clahe_normalization(img, modality)
                
                # High quality resize
                img = cv2.resize(img, (config.IMAGE_SIZE, config.IMAGE_SIZE), 
                               interpolation=cv2.INTER_AREA)
                volume.append(img)
                
            except Exception as e:
                volume.append(np.zeros((config.IMAGE_SIZE, config.IMAGE_SIZE), dtype=np.uint8))
                continue
        
        # Ensure we have exactly 8 frames
        while len(volume) < 8:
            if volume:
                volume.append(volume[-1])  # Duplicate last frame
            else:
                volume.append(np.zeros((config.IMAGE_SIZE, config.IMAGE_SIZE), dtype=np.uint8))
        
        return np.array(volume[:8])  # Take exactly 8 frames
```

```python
# Create 8-frame datasets
print("Creating 8-frame datasets with CLAHE...")
train_dataset = EightFrameDataset(
    train_fold_df, 
    frame_paths_dict, 
    series_mapping_df,
    num_frames=config.NUM_FRAMES,
    transform=train_transform,
    is_training=True
)

val_dataset = EightFrameDataset(
    val_fold_df,
    frame_paths_dict,
    series_mapping_df,
    num_frames=config.NUM_FRAMES, 
    transform=val_transform,
    is_training=False
)

# Create optimized data loaders
print("Creating optimized data loaders for 8-frame processing...")
train_loader = DataLoader(
    train_dataset,
    batch_size=config.BATCH_SIZE,
    shuffle=True,
    num_workers=config.NUM_WORKERS,
    pin_memory=config.PIN_MEMORY,
    drop_last=True,
    prefetch_factor=config.PREFETCH_FACTOR,
    persistent_workers=config.PERSISTENT_WORKERS
)

val_loader = DataLoader(
    val_dataset,
    batch_size=config.BATCH_SIZE,
    shuffle=False,
    num_workers=config.NUM_WORKERS,
    pin_memory=config.PIN_MEMORY,
    prefetch_factor=config.PREFETCH_FACTOR,
    persistent_workers=config.PERSISTENT_WORKERS
)

print(f"Train batches: {len(train_loader)}")
print(f"Validation batches: {len(val_loader)}")

# Test 8-frame data loading speed
print("Testing 8-frame data loading speed...")
import time

start_time = time.time()
for i, batch in enumerate(train_loader):
    if i >= 5:  # Test first 5 batches
        break
    images, labels, metadata = batch
    print(f"Batch {i+1}: Images shape: {images.shape}, Device: {images.device}")

elapsed = time.time() - start_time
print(f"Loaded 5 batches in {elapsed:.2f} seconds ({elapsed/5:.2f}s per batch)")
```

```python
class ImprovedMultiFrameModel(nn.Module):
    """Model with EfficientNetV2-S and metadata integration for 8-frame processing"""
    def __init__(self, num_frames=8, num_classes=14, pretrained=True):
        super(ImprovedMultiFrameModel, self).__init__()
        self.num_frames = num_frames
        self.num_classes = num_classes
        self.use_3channel = config.USE_3CHANNEL_INPUT
        self.use_metadata = config.USE_METADATA
        
        # Backbone: EfficientNetV2-S
        print(f"Loading backbone: {config.MODEL_NAME_BACKBONE}")
        self.backbone = timm.create_model(
            config.MODEL_NAME_BACKBONE,
            pretrained=pretrained,
            num_classes=0,
            global_pool='avg'
        )
        
        self.feature_dim = self.backbone.num_features
        print(f"Backbone {config.MODEL_NAME_BACKBONE}: {self.feature_dim} features")
        
        # Metadata processing
        if self.use_metadata:
            self.meta_fc = nn.Sequential(
                nn.Linear(2, 16),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(16, 32),
                nn.ReLU()
            )
            classifier_input_dim = self.feature_dim + 32
        else:
            classifier_input_dim = self.feature_dim
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x, meta=None):
        # 3-channel input processing (optimized for 8-frame data)
        features = self.backbone(x)  # (batch_size, feature_dim)
        
        # Metadata integration
        if self.use_metadata and meta is not None:
            meta_features = self.meta_fc(meta)
            features = torch.cat([features, meta_features], dim=1)
        
        # Classification
        output = self.classifier(features)
        return output

# Initialize 8-frame model
print("Initializing 8-frame model...")
model = ImprovedMultiFrameModel(
    num_frames=config.NUM_FRAMES,
    num_classes=config.NUM_CLASSES,
    pretrained=True
)

model = model.to(device)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
print(f"Model device: {next(model.parameters()).device}")
```

```python
class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance"""
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * bce_loss
        return focal_loss.mean()

class WeightedMultiLabelLoss(nn.Module):
    """Weighted multi-label loss"""
    def __init__(self, aneurysm_weight=3.0):
        super(WeightedMultiLabelLoss, self).__init__()
        self.weights = torch.ones(config.NUM_CLASSES, device=device)
        self.weights[-1] = aneurysm_weight
        
    def forward(self, outputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(outputs, targets, reduction='none')
        weighted_loss = bce_loss * self.weights
        return weighted_loss.mean()

class ImprovedLoss(nn.Module):
    """Advanced combined loss function"""
    def __init__(self, aneurysm_weight=3.0, focal_weight=0.3):
        super(ImprovedLoss, self).__init__()
        self.aneurysm_weight = aneurysm_weight
        self.focal_weight = focal_weight
        
        self.weights = torch.ones(config.NUM_CLASSES, device=device)
        self.weights[-1] = aneurysm_weight
        
        self.focal_loss = FocalLoss(alpha=1, gamma=2)
        
    def forward(self, outputs, targets):
        # Weighted BCE
        bce_loss = F.binary_cross_entropy_with_logits(outputs, targets, reduction='none')
        weighted_bce = (bce_loss * self.weights).mean()
        
        # Focal Loss
        focal_loss = self.focal_loss(outputs, targets)
        
        # Combination
        return (1 - self.focal_weight) * weighted_bce + self.focal_weight * focal_loss

def get_loss_function():
    """Get loss function based on configuration"""
    if config.USE_IMPROVED_LOSS:
        return ImprovedLoss(aneurysm_weight=3.0, focal_weight=0.3)
    else:
        return WeightedMultiLabelLoss(aneurysm_weight=3.0)

def calculate_competition_metric(y_true, y_pred):
    """Calculate competition metric: weighted multilabel AUC ROC"""
    individual_aucs = []
    
    # Calculate AUC for first 13 classes
    for i in range(13):
        try:
            if len(np.unique(y_true[:, i])) > 1:
                auc = roc_auc_score(y_true[:, i], y_pred[:, i])
            else:
                auc = 0.5
            individual_aucs.append(auc)
        except:
            individual_aucs.append(0.5)
    
    # Calculate AUC for Aneurysm Present
    try:
        if len(np.unique(y_true[:, 13])) > 1:
            aneurysm_present_auc = roc_auc_score(y_true[:, 13], y_pred[:, 13])
        else:
            aneurysm_present_auc = 0.5
    except:
        aneurysm_present_auc = 0.5
    
    # Final score
    avg_individual = np.mean(individual_aucs)
    final_score = (aneurysm_present_auc + avg_individual) / 2
    
    return final_score, aneurysm_present_auc, avg_individual, individual_aucs

# Training setup
criterion = get_loss_function()
optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=config.NUM_EPOCHS, eta_min=1e-6)

# Mixed precision training
scaler = torch.cuda.amp.GradScaler()

print("Training setup complete")
print(f"Using loss function: {type(criterion).__name__}")
```

```python
def train_epoch_optimized(model, train_loader, criterion, optimizer, scaler, device, accumulation_steps):
    """Optimized training function for 8-frame processing"""
    model.train()
    running_loss = 0.0
    
    optimizer.zero_grad()
    
    for batch_idx, (images, targets, metadata) in enumerate(tqdm(train_loader, desc="Training 8-Frame")):
        # Move data to GPU efficiently
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        metadata = metadata.to(device, non_blocking=True)
        
        with torch.cuda.amp.autocast():
            outputs = model(images, metadata)
            loss = criterion(outputs, targets)
            loss = loss / accumulation_steps
        
        scaler.scale(loss).backward()
        
        if (batch_idx + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        running_loss += loss.item() * accumulation_steps
    
    # Handle remaining gradients
    if len(train_loader) % accumulation_steps != 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
    
    return running_loss / len(train_loader)

def validate_epoch_optimized(model, val_loader, criterion, device):
    """Optimized validation function for 8-frame processing"""
    model.eval()
    running_loss = 0.0
    all_outputs = []
    all_targets = []
    
    with torch.no_grad():
        for images, targets, metadata in tqdm(val_loader, desc="Validating 8-Frame"):
            # Move data to GPU efficiently
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            metadata = metadata.to(device, non_blocking=True)
                
            with torch.cuda.amp.autocast():
                logits = model(images, metadata)
                loss = criterion(logits, targets)
            
            outputs = torch.sigmoid(logits)
            
            running_loss += loss.item()
            all_outputs.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
    
    all_outputs = np.concatenate(all_outputs)
    all_targets = np.concatenate(all_targets)
    
    final_score, aneurysm_auc, avg_individual, individual_aucs = calculate_competition_metric(
        all_targets, all_outputs
    )
    
    return running_loss / len(val_loader), final_score, aneurysm_auc, avg_individual

def check_gpu_utilization():
    """Check current GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        max_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB, Total: {max_memory:.2f}GB")
        utilization = (allocated/max_memory)*100
        print(f"GPU Utilization: {utilization:.1f}%")
        return utilization
    return 0

print("Initial GPU status for 8-frame processing:")
check_gpu_utilization()
```

```python
# Training loop with 8-frame optimization and true patient separation
best_score = 0.0
best_epoch = 0
patience_counter = 0
train_losses = []
val_losses = []
val_scores = []

print("Starting 8-frame training with patient-separated CV...")
print(f"Batch size: {config.BATCH_SIZE}, Workers: {config.NUM_WORKERS}")
print(f"Frames per sample: {config.NUM_FRAMES}")
print(f"CLAHE enabled: {config.USE_CLAHE}")
print(f"Strong augmentation: {config.USE_STRONG_AUGMENTATION}")
print(f"True patient separation: {config.USE_GROUP_CV}")

for epoch in range(config.NUM_EPOCHS):
    print(f"\nEpoch {epoch+1}/{config.NUM_EPOCHS}")
    print("-" * 50)
    
    # Training with 8-frame processing
    train_loss = train_epoch_optimized(
        model, train_loader, criterion, optimizer, scaler, device, config.ACCUMULATION_STEPS
    )
    
    # Validation with 8-frame processing
    val_loss, val_score, aneurysm_auc, avg_individual = validate_epoch_optimized(
        model, val_loader, criterion, device
    )
    
    # Learning rate scheduling
    scheduler.step()
    
    # Log metrics
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    val_scores.append(val_score)
    
    print(f"Train Loss: {train_loss:.6f}")
    print(f"Val Loss: {val_loss:.6f}")
    print(f"Val Score: {val_score:.6f}")
    print(f"Aneurysm AUC: {aneurysm_auc:.6f}")
    print(f"Avg Individual AUC: {avg_individual:.6f}")
    print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.8f}")
    
    # GPU utilization
    gpu_util = check_gpu_utilization()
    
    # Early stopping and model saving
    if val_score > best_score:
        best_score = val_score
        best_epoch = epoch + 1
        patience_counter = 0
        
        # Save model
        model_path = os.path.join(config.OUTPUT_DIR, f"{config.MODEL_NAME}_best.pth")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_score': best_score,
            'val_loss': val_loss,
            'aneurysm_auc': aneurysm_auc,
            'avg_individual_auc': avg_individual,
            'config': config,
            'model_config': {
                'backbone': config.MODEL_NAME_BACKBONE,
                'num_frames': config.NUM_FRAMES,
                'use_3channel': config.USE_3CHANNEL_INPUT,
                'use_metadata': config.USE_METADATA,
                'use_windowing': config.USE_WINDOWING,
                'use_improved_loss': config.USE_IMPROVED_LOSS,
                'use_clahe': config.USE_CLAHE,
                'use_strong_augmentation': config.USE_STRONG_AUGMENTATION,
                'use_group_cv': config.USE_GROUP_CV
            }
        }, model_path)
        
        print(f"New best model saved! Score: {best_score:.6f}")
    else:
        patience_counter += 1
        print(f"No improvement. Patience: {patience_counter}/{config.EARLY_STOPPING_PATIENCE}")
        
        if patience_counter >= config.EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break
    
    # Memory cleanup
    torch.cuda.empty_cache()

print("\n" + "="*70)
print("8-FRAME TRAINING WITH PATIENT SEPARATION COMPLETED")
print("="*70)
print(f"Best Score: {best_score:.6f} at Epoch {best_epoch}")
```

```python
# Training results visualization and summary
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Loss curves
axes[0].plot(range(1, len(train_losses)+1), train_losses, 'b-', label='Train Loss', linewidth=2)
axes[0].plot(range(1, len(val_losses)+1), val_losses, 'r-', label='Val Loss', linewidth=2)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('8-Frame Training: Loss Curves')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Validation score
axes[1].plot(range(1, len(val_scores)+1), val_scores, 'g-', label='Val Score', linewidth=2)
axes[1].axhline(y=best_score, color='r', linestyle='--', alpha=0.7, 
                label=f'Best: {best_score:.6f}')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Competition Score')
axes[1].set_title('8-Frame Training: Competition Score')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Learning rate schedule
lr_values = []
temp_optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-4)
temp_scheduler = CosineAnnealingLR(temp_optimizer, T_max=config.NUM_EPOCHS, eta_min=1e-6)
for _ in range(config.NUM_EPOCHS):
    lr_values.append(temp_optimizer.param_groups[0]['lr'])
    temp_scheduler.step()

axes[2].plot(range(1, len(lr_values)+1), lr_values, 'purple', linewidth=2, label='Learning Rate')
axes[2].set_xlabel('Epoch')
axes[2].set_ylabel('Learning Rate')
axes[2].set_title('Learning Rate Schedule')
axes[2].legend()
axes[2].grid(True, alpha=0.3)
axes[2].set_yscale('log')

plt.tight_layout()
plt.show()

# Final model summary
model_path = os.path.join(config.OUTPUT_DIR, f"{config.MODEL_NAME}_best.pth")
if os.path.exists(model_path):
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    print("\n" + "="*60)
    print("8-FRAME MODEL WITH PATIENT SEPARATION SUMMARY")
    print("="*60)
    print(f"Best Epoch: {checkpoint['epoch']}")
    print(f"Best Score: {checkpoint['best_score']:.6f}")
    print(f"Aneurysm AUC: {checkpoint['aneurysm_auc']:.6f}")
    print(f"Avg Individual AUC: {checkpoint['avg_individual_auc']:.6f}")
    print(f"Model Size: {os.path.getsize(model_path) / (1024*1024):.1f} MB")
    print(f"- CLAHE contrast adaptation: {config.USE_CLAHE}")
    print(f"- Strong augmentation: {config.USE_STRONG_AUGMENTATION}")


print("="*60)
print("TRAINING COMPLETE WITH PATIENT-SEPARATED CV!")
print("This should significantly improve LB performance.")
print("="*60)

# Final cleanup
torch.cuda.empty_cache()
gc.collect()
```