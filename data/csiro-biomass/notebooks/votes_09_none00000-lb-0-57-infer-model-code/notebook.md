# [LB 0.57]Infer + Model code 

- **Author:** none00000
- **Votes:** 245
- **Ref:** none00000/lb-0-57-infer-model-code
- **URL:** https://www.kaggle.com/code/none00000/lb-0-57-infer-model-code
- **Last run:** 2025-10-30 15:11:56.170000

---

---

# Methodology: CSIRO Pasture Biomass Prediction

## 1. Core Strategy: Predicting Key Components

The primary goal is to predict five biomass targets. Based on exploratory data analysis (EDA), we identified linear dependencies:
* `Dry_Total_g` $\approx$ `Dry_Green_g` + `Dry_Dead_g` + `Dry_Clover_g`
* `GDM_g` $\approx$ `Dry_Green_g` + `Dry_Clover_g`

To avoid redundancy, the model is trained to predict only the **three most visually distinct and/or highest-weighted targets**:
* `Dry_Total_g` (50% of the score)
* `GDM_g` (20% of the score)
* `Dry_Green_g` (10% of the score)

The remaining two targets (`Dry_Dead_g` and `Dry_Clover_g`) are then **calculated during validation and inference** using subtraction (e.g., `pred_Clover = max(0, pred_GDM - pred_Green)`).

---

## 2. Data Handling & K-Fold Strategy

* **Image Input:** All source images are high-resolution (`2000x1000` pixels).
* **Two-Stream Processing:** To preserve fine-grained details (like clover leaves) that would be lost by resizing the entire image, the `Dataset` class crops each image into two `1000x1000` patches (a "left" and "right" half).
* **High-Resolution Input:** Each `1000x1000` patch is then resized to **`768x768`**, maintaining a high level of detail.
* **K-Fold Strategy:** We use a **5-Fold Cross-Validation** strategy due to the small dataset (357 images).
* **Robust Splitting (GroupKFold):** To prevent data leakage (where similar images from the same day are in both train and validation), we use `GroupKFold` grouped by `Sampling_Date`. This ensures the model is validated on dates it has never seen.

---

## 3. Model Architecture: Two-Stream, Multi-Head

The model uses a "Two-Stream, Multi-Head" architecture.
* **Shared Backbone:** A single `timm` backbone (e.g., `convnext_tiny`) with pre-trained ImageNet weights is used.
* **Two-Stream Input:**
    * `img_left` $\rightarrow$ `backbone` $\rightarrow$ `features_left`
    * `img_right` $\rightarrow$ (same) `backbone` $\rightarrow$ `features_right`
* **Fusion:** The two feature vectors are concatenated: `combined_features = torch.cat([features_left, features_right])`.
* **Multi-Head Output:** This combined vector is fed into **three separate, specialized MLP heads** (one for each target: `head_total`, `head_gdm`, `head_green`) to allow for task specialization.

---

## 4. Data Augmentation

To compensate for the small dataset, augmentations are applied **independently** to the `img_left` and `img_right` patches.
* `HorizontalFlip (p=0.5)`
* `VerticalFlip (p=0.5)`
* `RandomRotate90 (p=0.5)` (Only 90-degree rotations)
* `ColorJitter`

This independent application creates a much larger variety of training combinations.

---

## 5. Loss Function: Weighted SmoothL1Loss

The model is optimized using a custom weighted loss function that aligns with the competition's scoring metric.
* **Base Loss:** `nn.SmoothL1Loss` (Huber Loss) is used instead of `MSELoss` to make training more stable and less sensitive to outliers.
* **Weighted Sum:** The final loss is a weighted sum of the individual losses, using the competition's scoring weights:
    $$Loss = (0.5 \cdot Loss_{Total}) + (0.2 \cdot Loss_{GDM}) + (0.1 \cdot Loss_{Green})$$

---

## 6. Training Strategy: Two-Stage Fine-Tuning

A two-stage "Freeze/Unfreeze" strategy is used to stabilize training on the small dataset.
* **Stage 1 (Freeze):**
    * **Epochs:** 1-5
    * **Action:** The entire `backbone` is frozen. Only the three MLP heads are trained.
    * **LR:** `1e-4`
* **Stage 2 (Unfreeze/Fine-Tuning):**
    * **Epochs:** 6-20
    * **Action:** The `backbone` is "unfrozen," and the entire model is trained.
    * **LR:** A very low learning rate (`1e-5`) is used to slowly adapt the backbone features.
* **Model Saving:** A `ModelCheckpoint` saves the model based on the **highest `Score (R^2)`** on the validation set, *not* the lowest loss. This is critical for capturing the model's peak performance (like the `R^2=0.64` spike at Epoch 11) and ignoring the unstable, overfitted epochs.

```python
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import timm
import cv2
from tqdm import tqdm
import gc

# ===============================================================
# 1. ⚙️ CONFIGURATION (PHẢI GIỐNG HỆT FILE TRAINING)
# ===============================================================
class CFG:
    # --- Đường dẫn (Paths) ---
    # (Hãy điều chỉnh các đường dẫn này cho đúng với môi trường của bạn)
    BASE_PATH = '/kaggle/input/csiro-biomass'
    TEST_CSV = os.path.join(BASE_PATH, 'test.csv')
    TEST_IMAGE_DIR = os.path.join(BASE_PATH, 'test')
    
    # Thư mục chứa 5 file .pth
    MODEL_DIR = '/kaggle/input/csiro/' # Giả sử 5 file .pth nằm cùng thư mục
    SUBMISSION_FILE = 'submission.csv'
    
    # --- Cài đặt Mô hình (PHẢI TRÙNG KHỚP) ---
    MODEL_NAME = 'convnext_tiny' # PHẢI GIỐNG HỆT LÚC TRAIN
    IMG_SIZE = 768               # PHẢI GIỐNG HỆT LÚC TRAIN
    
    # --- Cài đặt Inference ---
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    BATCH_SIZE = 1 # Có thể tăng batch size khi inference
    NUM_WORKERS = 1
    N_FOLDS = 5
    
    # --- Mục tiêu & Loss (PHẢI TRÙNG KHỚP) ---
    # 3 mục tiêu model đã dự đoán
    TARGET_COLS = ['Dry_Total_g', 'GDM_g', 'Dry_Green_g']
    
    # 5 mục tiêu để nộp bài
    ALL_TARGET_COLS = ['Dry_Green_g', 'Dry_Dead_g', 'Dry_Clover_g', 'GDM_g', 'Dry_Total_g']

print(f"Sử dụng thiết bị: {CFG.DEVICE}")
print(f"Backbone mô hình: {CFG.MODEL_NAME}")
print(f"Kích thước ảnh inference: {CFG.IMG_SIZE}x{CFG.IMG_SIZE}")


# ===============================================================
# 2. 🏞️ AUGMENTATIONS (CHỈ DÙNG VALIDATION)
# ===============================================================
from albumentations import (
    Compose, 
    Resize, 
    Normalize,
    HorizontalFlip, 
    VerticalFlip
)

def get_tta_transforms():
    """
    Trả về một LIST các pipeline transform cho TTA.
    Mỗi pipeline là một "view" khác nhau của ảnh.
    """
    
    # Đây là các bước chuẩn hóa cơ bản
    base_transforms = [
        Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ]
    
    # -----------------
    # View 1: Ảnh gốc (Chỉ Resize + Normalize)
    # -----------------
    original_view = Compose([
        Resize(CFG.IMG_SIZE, CFG.IMG_SIZE),
        *base_transforms
    ])
    
    # -----------------
    # View 2: Lật ngang (HFlip)
    # -----------------
    hflip_view = Compose([
        HorizontalFlip(p=1.0), # Luôn luôn lật
        Resize(CFG.IMG_SIZE, CFG.IMG_SIZE),
        *base_transforms
    ])
    
    # -----------------
    # View 3: Lật dọc (VFlip)
    # -----------------
    vflip_view = Compose([
        VerticalFlip(p=1.0), # Luôn luôn lật
        Resize(CFG.IMG_SIZE, CFG.IMG_SIZE),
        *base_transforms
    ])
    
    return [original_view, hflip_view, vflip_view]

print("Đã định nghĩa hàm get_tta_transforms().")


class TestBiomassDataset(Dataset):
    """
    Dataset tùy chỉnh cho ảnh test (Chiến lược "Hai luồng").
    Sửa đổi để chấp nhận một pipeline transform cụ thể cho TTA.
    """
    def __init__(self, df, transform_pipeline, image_dir):
        self.df = df
        # (SỬA ĐỔI) Chấp nhận một pipeline đã được khởi tạo
        self.transforms = transform_pipeline 
        self.image_dir = image_dir
        self.image_paths = df['image_path'].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # 1. Lấy thông tin
        img_path_suffix = self.image_paths[idx]
        
        # 2. Đọc ảnh gốc (2000x1000)
        filename = os.path.basename(img_path_suffix)
        full_path = os.path.join(self.image_dir, filename)
        
        image = cv2.imread(full_path)
        if image is None:
            print(f"Warning: Không thể đọc ảnh: {full_path}. Trả về ảnh đen.")
            image = np.zeros((1000, 2000, 3), dtype=np.uint8)
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 3. Cắt (Crop) thành 2 ảnh (Trái và Phải)
        height, width, _ = image.shape
        mid_point = width // 2
        img_left = image[:, :mid_point]
        img_right = image[:, mid_point:]
        
        # 4. Áp dụng TTA Transform (CÙNG MỘT TRANSFORM cho cả 2)
        # (Ví dụ: Cả 2 ảnh đều bị lật ngang)
        img_left_tensor = self.transforms(image=img_left)['image']
        img_right_tensor = self.transforms(image=img_right)['image']
        
        # 5. Trả về
        return img_left_tensor, img_right_tensor

# ===============================================================
# 4. 🧠 MODEL ARCHITECTURE (SAO CHÉP TỪ FILE TRAIN)
# ===============================================================
class BiomassModel(nn.Module):
    """
    Kiến trúc mô hình (Hai luồng, Ba đầu ra)
    PHẢI GIỐNG HỆT file training.
    """
    def __init__(self, model_name, pretrained, n_targets=3):
        super(BiomassModel, self).__init__()
        
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained, # Sẽ là False khi inference
            num_classes=0,
            global_pool='avg'
        )
        
        self.n_features = self.backbone.num_features
        self.n_combined_features = self.n_features * 2
        
        # --- Đầu cho Dry_Total_g ---
        self.head_total = nn.Sequential(
            nn.Linear(self.n_combined_features, self.n_combined_features // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(self.n_combined_features // 2, 1)
        )
        
        # --- Đầu cho GDM_g ---
        self.head_gdm = nn.Sequential(
            nn.Linear(self.n_combined_features, self.n_combined_features // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(self.n_combined_features // 2, 1)
        )
        
        # --- Đầu cho Dry_Green_g ---
        self.head_green = nn.Sequential(
            nn.Linear(self.n_combined_features, self.n_combined_features // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(self.n_combined_features // 2, 1)
        )

    def forward(self, img_left, img_right):
        features_left = self.backbone(img_left)
        features_right = self.backbone(img_right)
        combined = torch.cat([features_left, features_right], dim=1)
        
        out_total = self.head_total(combined)
        out_gdm = self.head_gdm(combined)
        out_green = self.head_green(combined)
        
        return out_total, out_gdm, out_green


def predict_one_view(models_list, test_loader, device):
    """
    Hàm con: Chạy dự đoán ensemble 5-fold cho MỘT view TTA.
    """
    view_preds_3 = {'total': [], 'gdm': [], 'green': []}
    
    with torch.no_grad():
        for (img_left, img_right) in tqdm(test_loader, desc="  Predicting View", leave=False):
            img_left = img_left.to(device)
            img_right = img_right.to(device)
            
            batch_preds_3_folds = {'total': [], 'gdm': [], 'green': []}
            
            # 1. Vòng lặp Ensemble 5-Fold
            for model in models_list:
                pred_total, pred_gdm, pred_green = model(img_left, img_right)
                batch_preds_3_folds['total'].append(pred_total.cpu())
                batch_preds_3_folds['gdm'].append(pred_gdm.cpu())
                batch_preds_3_folds['green'].append(pred_green.cpu())
            
            # 2. Lấy trung bình 5 Fold
            avg_pred_total = torch.mean(torch.stack(batch_preds_3_folds['total']), dim=0)
            avg_pred_gdm = torch.mean(torch.stack(batch_preds_3_folds['gdm']), dim=0)
            avg_pred_green = torch.mean(torch.stack(batch_preds_3_folds['green']), dim=0)
            
            view_preds_3['total'].append(avg_pred_total.numpy())
            view_preds_3['gdm'].append(avg_pred_gdm.numpy())
            view_preds_3['green'].append(avg_pred_green.numpy())

    # 3. Ghép kết quả các batch của view này
    preds_np = {
        'total': np.concatenate(view_preds_3['total']).flatten(),
        'gdm':   np.concatenate(view_preds_3['gdm']).flatten(),
        'green': np.concatenate(view_preds_3['green']).flatten()
    }
    return preds_np


def run_inference_with_tta():
    """
    Hàm inference chính, thực hiện TTA x Ensemble.
    """
    print(f"\n{'='*50}")
    print(f"🚀 BẮT ĐẦU INFERENCE (với TTA) 🚀")
    print(f"{'='*50}")

    # --- 1. Tải Dữ liệu Test ---
    print(f"Đang tải {CFG.TEST_CSV}...")
    try:
        test_df_long = pd.read_csv(CFG.TEST_CSV)
        test_df_unique = test_df_long.drop_duplicates(subset=['image_path']).reset_index(drop=True)
        print(f"Tìm thấy {len(test_df_unique)} ảnh test duy nhất.")
    except FileNotFoundError:
        print(f"LỖI: Không tìm thấy {CFG.TEST_CSV}")
        return None, None, None

    # --- 2. Tải 5 Mô hình (Ensemble) ---
    print("\nĐang tải 5 mô hình đã huấn luyện...")
    models_list = []
    # (Code tải 5 mô hình... giống hệt bước 16 của file trước)
    for fold in range(CFG.N_FOLDS):
        model_path = os.path.join(CFG.MODEL_DIR, f'best_model_fold{fold}.pth')
        if not os.path.exists(model_path):
            print(f"LỖI: Không tìm thấy file mô hình: {model_path}")
            return None, None, None
        model = BiomassModel(CFG.MODEL_NAME, pretrained=False)
        try:
            model.load_state_dict(torch.load(model_path, map_location=CFG.DEVICE))
        except RuntimeError:
            state_dict = torch.load(model_path, map_location=CFG.DEVICE)
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k.replace('module.', '')
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)
        model.eval()
        model.to(CFG.DEVICE)
        models_list.append(model)
    print(f"✓ Đã tải thành công {len(models_list)} mô hình.")

    # --- 3. Vòng lặp TTA (Vòng lặp ngoài) ---
    tta_transforms = get_tta_transforms()
    print(f"\nBắt đầu dự đoán với {len(tta_transforms)} TTA views...")
    
    all_tta_view_preds = [] # List để lưu kết quả của mỗi view TTA

    for i, tta_transform in enumerate(tta_transforms):
        print(f"--- Đang chạy TTA View {i+1}/{len(tta_transforms)} ---")
        
        # Tạo Dataset/Loader MỚI cho view TTA này
        test_dataset = TestBiomassDataset(
            df=test_df_unique,
            transform_pipeline=tta_transform, # Truyền pipeline TTA
            image_dir=CFG.TEST_IMAGE_DIR
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=CFG.BATCH_SIZE,
            shuffle=False,
            num_workers=CFG.NUM_WORKERS,
            pin_memory=True
        )
        
        # Chạy ensemble 5-fold cho view này
        view_preds_np = predict_one_view(models_list, test_loader, CFG.DEVICE)
        all_tta_view_preds.append(view_preds_np)
        print(f"✓ Hoàn thành TTA View {i+1}")

    # --- 4. Ensemble (Lấy trung bình) kết quả TTA ---
    print("\nĐang ensemble kết quả của các TTA views...")
    final_ensembled_preds = {
        'total': np.mean([d['total'] for d in all_tta_view_preds], axis=0),
        'gdm':   np.mean([d['gdm'] for d in all_tta_view_preds], axis=0),
        'green': np.mean([d['green'] for d in all_tta_view_preds], axis=0)
    }
    
    print("✓ Dự đoán hoàn tất.")
    
    del models_list, test_loader, test_dataset
    gc.collect()
    torch.cuda.empty_cache()
    
    return final_ensembled_preds, test_df_long, test_df_unique
# ===============================================================
# 6. ✍️ HÀM TẠO FILE SUBMISSION
# ===============================================================
def create_submission(preds_np, test_df_long, test_df_unique):
    """
    Hàm này nhận 3 dự đoán đã ensemble,
    tính toán 2 dự đoán còn lại,
    và định dạng file nộp bài.
    """
    if preds_np is None:
        print("Bỏ qua tạo submission do lỗi ở trên.")
        return

    print("\nĐang hậu xử lý và tạo file submission...")

    # 1. Lấy 3 dự đoán đã ensemble
    pred_total_final = preds_np['total']
    pred_gdm_final = preds_np['gdm']
    pred_green_final = preds_np['green']

    # 2. Tính 2 mục tiêu còn lại (Hậu xử lý)
    # Dùng np.maximum(0, ...) để đảm bảo không có giá trị âm
    pred_clover_final = np.maximum(0, pred_gdm_final - pred_green_final)
    pred_dead_final = np.maximum(0, pred_total_final - pred_gdm_final)

    # 3. Tạo một DataFrame "wide" chứa 5 dự đoán
    # (Đảm bảo thứ tự 5 cột giống CFG.ALL_TARGET_COLS)
    preds_wide_df = pd.DataFrame({
        'image_path': test_df_unique['image_path'],
        'Dry_Green_g': pred_green_final,
        'Dry_Dead_g': pred_dead_final,
        'Dry_Clover_g': pred_clover_final,
        'GDM_g': pred_gdm_final,
        'Dry_Total_g': pred_total_final
    })

    # 4. "Un-pivot" DataFrame (Chuyển sang dạng "long")
    # Biến nó từ 5 cột về dạng "long" (giống sample_submission)
    preds_long_df = preds_wide_df.melt(
        id_vars=['image_path'],
        value_vars=CFG.ALL_TARGET_COLS, # 5 cột mục tiêu
        var_name='target_name',        # Cột tên mục tiêu
        value_name='target'            # Cột giá trị dự đoán
    )

    # 5. Merge với file test.csv gốc (test_df_long)
    # Đây là bước quan trọng để lấy đúng 'sample_id'
    # (ví dụ: 'ID1001187975__Dry_Clover_g')
    submission_df = pd.merge(
        test_df_long[['sample_id', 'image_path', 'target_name']],
        preds_long_df,
        on=['image_path', 'target_name'],
        how='left'
    )

    # 6. Dọn dẹp và Lưu
    # Chỉ giữ lại 2 cột được yêu cầu
    submission_df = submission_df[['sample_id', 'target']]
    
    # Lưu file
    submission_df.to_csv(CFG.SUBMISSION_FILE, index=False)

    print(f"\n🎉 HOÀN TẤT! Đã lưu file submission tại: {CFG.SUBMISSION_FILE}")
    print("--- 5 hàng đầu của file submission ---")
    print(submission_df.head())
    print("\n--- 5 hàng cuối của file submission ---")
    print(submission_df.tail())
    
# ===============================================================
# 8. 🏁 CHẠY CHƯƠNG TRÌNH (Đã sửa)
# ===============================================================
if __name__ == "__main__":
    # 1. Chạy dự đoán (đã bao gồm TTA)
    all_preds_np, df_long, df_unique = run_inference_with_tta()
    
    # 2. Tạo file submission (Hàm create_submission giữ nguyên)
    create_submission(all_preds_np, df_long, df_unique)
```