# 4th Place Solution: ViT-Huge DINOv3 & Multi-Modal Feature Fusion with AUX predic

- **Author:** Jatin Mehra_666
- **Date:** 2026-01-29T05:44:24.957Z
- **Topic ID:** 670677
- **URL:** https://www.kaggle.com/competitions/csiro-biomass/discussion/670677

**GitHub links found:**
- https://github.com/Jatin-Mehra119/Kaggle-CSIRO-4th-Position-Solution-

---

# ViT-Huge DINOv3 & Multi-Modal Feature Fusion with AUX prediction

First, a huge thank you to the organizers for hosting this challenge and to my fellow competitors. Sharath and I are thrilled to achieve the **4th position (Gold Medal)**. Our solution relies on a heavy Vision Transformer backbone initialized with DINOv3 weights, a multi-modal fusion strategy combining images with tabular data, and a critical data cleaning pipeline.

## 1. Data Preprocessing: The "Cardboard" Cleanup

Before touching the model architecture, we realized a significant portion of the image data contained irrelevant noise—specifically, the cardboard backing used in the data collection process.

-   **Manual Cropping:** My teammate and I manually reviewed the dataset and cropped out the cardboard edges from the pasture images. This ensured the model focused purely on the biomass (grass/clover) rather than learning artifacts from the background.
    
-   **Impact:** This step was crucial. It provided a clear boost of **+0.01** on both leaderboards:
    
    -   **Public LB:** 0.74 →**0.75**
        
    -   **Private LB:** 0.64 →**0.65**
        

## 2. Main Model Architecture

My main approach was a **Multi-Modal Regression Network**. I treated the problem as a fusion of visual data (pasture images) and physical measurements (Height/NDVI).

### The Architecture Diagram

![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F18908513%2Fa13e6b6e61001440d967ec88f50218a0%2Fpart%20-%201.png?generation=1769666215659267&alt=media)

### Key Components

-   **Backbone:** `vit_huge_plus_patch16_dinov3.lvd1689m`  weights. The self-supervised nature of DINOv3 provided superior feature extraction for grass textures compared to standard supervised weights with first half layers frozen.
    
-   **Fusion Strategy:** I did not rely solely on images. `Pre_GSHH_NDVI` and `Height_Ave_cm` were encoded via a 2-layer MLP and concatenated with the ViT global average pooling features before the final regression heads.
    
-   **Loss Function:** `WeightedSmoothL1Loss`. I used custom weights `[0.1, 0.1, 0.1, 0.2, 0.5]` to prioritize the "Total" and "GDM" biomass targets.
    

## 3. The "Secret Sauce": Auxiliary Task Training

A major boost came from a secondary training stage where I repurposed the trained backbone to predict the _tabular features_ from the images.

### Auxiliary Pipeline Diagram

![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F18908513%2F4e9dd4df598970a80cc532db250f5c56%2Fpart%20-%202.png?generation=1769666274910752&alt=media)

-   **Logic:** By forcing the model to predict `NDVI` and `Height` solely from the RGB image, the backbone learned robust features correlated with plant health and density that the primary regression heads might have missed.
    
-   **Initialization:** This model was initialized with the **best weights from Fold 0** of the main training loop, essentially acting as a domain-specific fine-tuner.
    
-   **Impact:** This auxiliary training provided the final push for the Gold Medal, adding another **+0.01** to both scores:
    
    -   **Final Public LB:** 0.76
        
    -   **Final Private LB:** 0.66
        

## 4. What Didn't Work (The Graveyard)

Despite the success, many ideas were left on the cutting room floor.

-   **Loss Functions:** Pure MSE, Quantile Loss, Log Loss, and direct R2 optimization all failed to beat SmoothL1.
    
-   **Target Transforms:** Log transformation of targets and Scaled Sigmoid outputs reduced performance.
    
-   **Image Size:** Input sizes larger than 800x800 offered diminishing returns and OOM errors.
    
-   **Complex Input:** Attempting to process two images simultaneously (complex dual-input) did not yield better results.
    

### The "Promising but Failed" Architecture: Hybrid Texture Pooling

I experimented with a custom pooling layer designed to reconstruct spatial grids from ViT tokens and pool across the height dimension (simulating a scan). While the validation loss looked very promising, it did not generalize as well to the private leaderboard.

Python

```
class HybridTexturePooling(nn.Module):
    def __init__(self, embed_dim=1280, patch_size=16, num_extra_tokens=5):
        super().__init__()
        self.ps = patch_size
        self.num_extra_tokens = num_extra_tokens
        # Projection layer: maps ViT embedding to pixel-space (ps**2) * 2
        self.projection = nn.Linear(embed_dim, (patch_size ** 2) * 2)

    def forward(self, x, h, w):
        # x: [BS, N_tokens, Dim]
        # 1. Slice off CLS + Register tokens
        patch_tokens = x[:, self.num_extra_tokens:, :] 
        
        # 2. Project to pixel-like space
        x = self.projection(patch_tokens) 
        
        bs = x.shape[0]
        h_patches, w_patches = h // self.ps, w // self.ps
        
        # 3. Reconstruct Spatial Grid
        # Reshape to (BS, H_p, W_p, PS, PS, 2)
        x = x.view(bs, h_patches, w_patches, self.ps, self.ps, 2)
        # Permute to (BS, H_p, PS, W_p, PS, 2)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        # Flatten to (BS, H, W * 2)
        x = x.view(bs, h, w * 2)
        
        # 4. Mean across the Height dimension
        x = x.mean(dim=1) # Result: [BS, 1600] for 800px input
        return x
```

### 5. Training Configuration

We utilized a two-stage training pipeline. Below are the specific hyperparameters used for each stage.

#### **Stage 1: Main Biomass Regression**

-   **Model:** `vit_huge_plus_patch16_dinov3.lvd1689m` (Weights frozen for first 50% of layers)
    
-   **Image Size:** 800× 800
    
-   **Batch Size:** 10
    
-   **Optimizer:** AdamW
    
-   **Scheduler:** CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
    
-   **Loss Function:** `WeightedSmoothL1Loss`
    
    -   Weights: `[0.1, 0.1, 0.1, 0.2, 0.5]` (Heavily penalizing Total Biomass errors)
        
-   **Validation Strategy:** 5-Fold Cross-Validation (Seed 42)
    

#### **Stage 2: Auxiliary Feature Pre-training**

-   **Objective:** Predict Tabular Features (`NDVI`, `Height`) from Images
    
-   **Initialization:** Loaded best weights from **Stage 1 (Fold 0)**
    
-   **Batch Size:** 8
    
-   **Optimizer:** AdamW
    
-   **Scheduler:** ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4)
    
-   **Loss Function:** `MSELoss`
    
-   **Validation Strategy:** 5-Fold Cross-Validation (Seed 44)

Note: All regression targets and auxiliary features (tabular inputs) were normalized using StandardScaler prior to training to ensure stable convergence.


Thanks again to the community! I hope this write-up and the diagrams help explain the structure of the solution.