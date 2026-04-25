# 13th Place Solution

- **Author:** Maruichi01
- **Date:** 2026-01-23T00:59:29.250Z
- **Topic ID:** 669546
- **URL:** https://www.kaggle.com/competitions/physionet-ecg-image-digitization/discussion/669546
---

Thanks to the organizers for this interesting competition! Here's a summary of our approach.

---

## Overview

Our solution is based on an **End-to-End (E2E) deep learning pipeline** that directly predicts ECG time-series signals from ECG images. The key insight was to combine image segmentation with signal refinement in a single differentiable pipeline.

---

## Pipeline Architecture

![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F2848721%2F1314856109fa4672e8b6ded6a3a4a12a%2Fpipeline.jpg?generation=1769129908598386&alt=media)

```
ECG Image (Stage 1 preprocessed)
    │  [H=1280, W=5600, C=3]
    ↓
Stage 2: UNet Segmentation (4-channel soft mask)
    │  [H=1280, W=5600, C=4]
    │  ← Aux Loss 1: Segmentation BCE Loss
    ↓
Differentiable Centroid Extraction (mask → 1D signal)
    │  [C=4, L=10250]
    │  ← Aux Loss 2: Signal Centroid L1 Loss
    ↓
Stage 3: 1D ResUNet Refinement
    │  [C=4, L=10250]
    │  ← Main Loss: L1 + SNR Loss
    ↓
Resampling Ensemble (decimate_fir + polyphase + linear)
    │  [C=4, L=fs×10] (e.g., 5000 for fs=500Hz)
    ↓
Final ECG Signal (12 leads)
```


### Stage 0/Stage 1: Preprocessing (from hengck23's notebook)
- We used **hengck23's excellent public notebook** for preprocessing
- Includes image rotation correction, cropping, and resizing to 5600×1280

### Stage 2: UNet Segmentation
- **Encoder**: EfficientNet-B3/B4 or ResNet34 (timm pretrained)
- **Decoder**: UNet with coordinate channels (CoordConv-style)
- **Output**: 4-channel soft segmentation mask (one per ECG row)
- **Loss**: BCE with pos_weight=10 for class imbalance

### Differentiable Centroid Conversion
This was a key component that enabled end-to-end training:

```python
# For each column, compute weighted centroid of the probability distribution
y_coords = torch.arange(H).view(1, 1, H, 1)
prob_sum = seg_prob.sum(dim=2) + eps
centroid_y = (seg_prob * y_coords).sum(dim=2) / prob_sum

# Convert pixel position to mV
signal_mv = (base_y_position - centroid_y) / y_scale
```

This allows gradients to flow from the signal loss back through the segmentation network.

### Stage 3: 1D ResUNet
- **Architecture**: Residual 1D UNet
- **Input/Output**: 4 channels (corresponding to 4 ECG rows)
- **Depth**: 4-5 levels
- **Purpose**: Refine the centroid-extracted signal, correct artifacts

---

## Training Strategy

### Loss Function
Combined segmentation and signal losses with scheduled weighting:

```python
# Early training: focus on segmentation
# Later training: shift focus to signal quality
seg_weight = seg_start + (seg_end - seg_start) * progress
signal_weight = sig_start + (sig_end - sig_start) * progress

loss = seg_weight * seg_bce_loss + signal_weight * (l1_loss + snr_loss)
```

### Key Training Details
- **Optimizer**: AdamW (lr=0.005, weight_decay=1e-4)
- **Scheduler**: Cosine annealing (60 epochs schedule, 32 epochs training)
- **Batch size**: 4 with gradient accumulation (effective batch 16)
- **Precision**: FP32 (FP16 caused NaN issues)
- **EMA**: Enabled (decay=0.995)
- **All-data training**: No validation split for final models

### Data Augmentation
Augmentation was crucial for generalization:

| Augmentation | Probability | Impact |
|--------------|-------------|--------|
| **Horizontal Flip** | 0.5 | **+0.6 dB** (biggest improvement!) |
| Grayscale | 0.2 | Helps with color variation |
| Brightness/Contrast | 0.3 | Robustness to lighting |
| JPEG Compression | 0.2 | Handles low-quality scans |
| Cutout | 0.2 | Handles occlusions/damage |
| Gaussian Noise | 0.2 | Handles sensor noise |

The horizontal flip augmentation was surprisingly effective (+0.6 dB), likely because it doubled the effective training data and helped the model learn orientation-invariant features.

---

## Ensemble Strategy

### Model Diversity
We trained multiple models with different configurations:

| Model | Encoder | Stage3 | hflip | LB Score(w/o Resampling Ensemble) |
|-------|---------|--------|-------|----------|
| m009 | ResNet34 | resunet1d (d=4) | 0.0 | 20.02 |
| m010 | EfficientNet-B3 | resunet1d (d=4) | 0.0 | 20.21 |
| m013 | ResNet34 | resunet1d (d=4) | 0.5 | 20.64 |
| m014 | EfficientNet-B3 | resunet1d (d=4) | 0.5 | 20.84 |
| m018 | EfficientNet-B3 | resunet1d (d=5) | 0.5 | 20.87 |
| m020 | EfficientNet-B4 | resunet1d (d=4) | 0.5 | 20.61 |

### Final Ensemble
```python
models = ['m009', 'm010', 'm013', 'm014', 'm018', 'm020']
weights = [0.05, 0.1, 0.2, 0.25, 0.25, 0.15]
```

Weighted average based on single-model performance, with slight diversity bonus for different architectures.

### Resampling Ensemble
A subtle but effective technique: **ensemble multiple resampling methods** when converting from model output length to target sampling frequency.

```python
ensemble_methods = ['decimate_fir', 'polyphase', 'linear']
```

Different resampling algorithms introduce different artifacts (especially at signal edges). Averaging them cancels out method-specific artifacts.

---

## What Worked

1. **End-to-end training** - Joint optimization of segmentation and signal extraction was better than separate stages
2. **Horizontal flip augmentation** - Surprisingly gave +0.6 dB improvement
3. **EfficientNet encoders** - Consistently +0.2 dB over ResNet
4. **Deeper Stage3** - depth=5 helped capture longer-range dependencies
5. **Resampling ensemble** - Reduced edge artifacts
6. **Model diversity in ensemble** - Mixing architectures and augmentation settings

---

## What Didn't Work

1. **BiLSTM for Stage3** - Worse than ResUNet (-0.9 dB)
2. **Higher resolution (7200×1280)** - No improvement, slower training
3. **Larger output_length (15000)** - Marginal improvement not worth complexity
4. **ConvNeXt encoder** - Training instability issues


---

## Acknowledgments

**Huge thanks to [@hengck23](https://www.kaggle.com/hengck23)** for the excellent public notebooks! Our Stage 1 preprocessing is entirely based on hengck23's work, which provided the foundation for our pipeline. The clean, standardized images from Stage 1 were essential for training our E2E model effectively.

Thanks also to the Kaggle community for the insightful discussions throughout the competition