# siglep+dinov3

- **Author:** Giovanny Rodríguez
- **Votes:** 702
- **Ref:** giovannyrodrguez/siglep-dinov3
- **URL:** https://www.kaggle.com/code/giovannyrodrguez/siglep-dinov3
- **Last run:** 2026-01-19 23:05:27.487000

---

# Thanks,

## Credits: Eng Adam Al mohammedi, Baidalin, Adilzhan, CigarCat, Mattia Angeli, Khoa, samu2505 

## siglep LB: 65 now: -

## dinov3 Lb: 70 now: +

```python
import os
import gc
import random
import warnings
import numpy as np
import pandas as pd
import cv2
import torch
from PIL import Image
from tqdm.auto import tqdm
from pathlib import Path
from copy import deepcopy
from dataclasses import dataclass

# Sklearn & Models
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from transformers import AutoModel, AutoImageProcessor, AutoTokenizer

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# =========================================================================================
# 1. CONFIGURATION & SEEDING
# =========================================================================================
@dataclass
class Config:
    DATA_PATH: Path = Path("/kaggle/input/csiro-biomass/")
    SPLIT_PATH: Path = Path("/kaggle/input/csiro-datasplit/csiro_data_split.csv")
    SIGLIP_PATH: str = "/kaggle/input/google-siglip-so400m-patch14-384/transformers/default/1"
    
    SEED: int = 42
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    PATCH_SIZE: int = 520
    OVERLAP: int = 16
    
    # Target definitions
    TARGET_NAMES = ['Dry_Clover_g', 'Dry_Dead_g', 'Dry_Green_g', 'Dry_Total_g', 'GDM_g']
    TARGET_MAX = {
        "Dry_Clover_g": 71.7865,
        "Dry_Dead_g": 83.8407,
        "Dry_Green_g": 157.9836,
        "Dry_Total_g": 185.70,
        "GDM_g": 157.9836,
    }

cfg = Config()

def seeding(SEED):
    np.random.seed(SEED)
    random.seed(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

seeding(cfg.SEED)

# =========================================================================================
# 2. DATA LOADING & PRE-PROCESSING
# =========================================================================================
def pivot_table(df: pd.DataFrame) -> pd.DataFrame:
    if 'target' in df.columns.tolist():
        # Train data
        df_pt = pd.pivot_table(
            df, 
            values='target', 
            index=['image_path', 'Sampling_Date', 'State', 'Species', 'Pre_GSHH_NDVI', 'Height_Ave_cm'], 
            columns='target_name', 
            aggfunc='mean'
        ).reset_index()
    else:
        # Test data
        df['target'] = 0
        df_pt = pd.pivot_table(
            df, 
            values='target', 
            index='image_path', 
            columns='target_name', 
            aggfunc='mean'
        ).reset_index()
    return df_pt

def melt_table(df: pd.DataFrame) -> pd.DataFrame:
    melted = df.melt(
        id_vars='image_path',
        value_vars=cfg.TARGET_NAMES,
        var_name='target_name',
        value_name='target'
    )
    # Create sample_id matching submission format
    melted['sample_id'] = (
        melted['image_path']
        .str.replace(r'^.*/', '', regex=True)
        .str.replace('.jpg', '', regex=False)
        + '__' + melted['target_name']
    )
    return melted[['sample_id', 'target']]

def post_process_biomass(df_preds):
    """
    Derive GDM_g and Dry_Total_g from primary predictions.
    IMPORTANT: Keeps Dry_Clover_g fixed at 0.0 and does NOT modify Dry_Green_g or Dry_Dead_g.
    """
    ordered_cols = ["Dry_Green_g", "Dry_Clover_g", "Dry_Dead_g", "GDM_g", "Dry_Total_g"]
    
    # Ensure cols exist
    for c in ordered_cols:
        if c not in df_preds.columns:
            df_preds[c] = 0.0
    
    df_out = df_preds.copy()
    
    # Keep Dry_Clover_g fixed at 0.0 (should already be 0.0 from model prediction)
    df_out['Dry_Clover_g'] = 0.0
    
    # Keep Dry_Green_g and Dry_Dead_g as predicted (NO MODIFICATION)
    # Only derive GDM_g and Dry_Total_g
    df_out['GDM_g'] = df_out['Dry_Green_g'] + df_out['Dry_Clover_g']
    df_out['Dry_Total_g'] = df_out['GDM_g'] + df_out['Dry_Dead_g']
    
    # Clip only derived targets to non-negative
    df_out['GDM_g'] = df_out['GDM_g'].clip(lower=0.0)
    df_out['Dry_Total_g'] = df_out['Dry_Total_g'].clip(lower=0.0)
    
    return df_out

print("Loading Data...")
# Load Train (Metadata Split)
train_df = pd.read_csv(cfg.SPLIT_PATH) 

# --- FIX: Remove pre-existing embedding columns to prevent duplication ---
cols_to_keep = [c for c in train_df.columns if not c.startswith('emb')]
train_df = train_df[cols_to_keep]
# -----------------------------------------------------------------------

# Ensure train paths match local environment
if not str(train_df['image_path'].iloc[0]).startswith('/'):
     train_df['image_path'] = train_df['image_path'].apply(lambda p: str(cfg.DATA_PATH / 'train' / os.path.basename(p)))

# Load Test
test_df_raw = pd.read_csv(cfg.DATA_PATH / 'test.csv')
test_df = pivot_table(test_df_raw)
test_df['image_path'] = test_df['image_path'].apply(lambda p: str(cfg.DATA_PATH / p))

# =========================================================================================
# 3. FEATURE EXTRACTION: SIGLIP IMAGE EMBEDDINGS
# =========================================================================================
def split_image(image, patch_size=520, overlap=16):
    h, w, c = image.shape
    stride = patch_size - overlap
    patches = []
    for y in range(0, h, stride):
        for x in range(0, w, stride):
            y2 = min(y + patch_size, h)
            x2 = min(x + patch_size, w)
            y1 = max(0, y2 - patch_size) # Ensure fixed size
            x1 = max(0, x2 - patch_size)
            patch = image[y1:y2, x1:x2, :]
            patches.append(patch)
    return patches

def compute_embeddings(model_path, df):
    print(f"Computing Embeddings for {len(df)} images...")
    model = AutoModel.from_pretrained(model_path, local_files_only=True).eval().to(cfg.DEVICE)
    processor = AutoImageProcessor.from_pretrained(model_path)
    
    EMBEDDINGS = []
    
    for _, row in tqdm(df.iterrows(), total=len(df)):
        try:
            img = cv2.imread(row['image_path'])
            if img is None: raise ValueError("Image not found")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            patches = split_image(img, patch_size=cfg.PATCH_SIZE, overlap=cfg.OVERLAP)
            images = [Image.fromarray(p) for p in patches]
            
            # Batch process patches
            inputs = processor(images=images, return_tensors="pt").to(cfg.DEVICE)
            with torch.no_grad():
                features = model.get_image_features(**inputs)
            
            # Average pooling of patches
            avg_embed = features.mean(dim=0).cpu().numpy()
            EMBEDDINGS.append(avg_embed)
        except Exception as e:
            print(f"Error processing {row['image_path']}: {e}")
            # Fallback zero embedding
            EMBEDDINGS.append(np.zeros(1152))
        
    torch.cuda.empty_cache()
    return np.stack(EMBEDDINGS)

# Compute Features
train_embeddings = compute_embeddings(cfg.SIGLIP_PATH, train_df)
test_embeddings = compute_embeddings(cfg.SIGLIP_PATH, test_df)

# Create Feature DataFrames
emb_cols = [f"emb{i}" for i in range(train_embeddings.shape[1])]
train_feat_df = pd.concat([train_df, pd.DataFrame(train_embeddings, columns=emb_cols)], axis=1)
test_feat_df = pd.concat([test_df, pd.DataFrame(test_embeddings, columns=emb_cols)], axis=1)

# Double check column counts
print(f"Train Features Shape: {train_feat_df.shape}")
print(f"Test Features Shape: {test_feat_df.shape}")

# =========================================================================================
# 4. FEATURE EXTRACTION: SEMANTIC FEATURES (TEXT PROBING)
# =========================================================================================
def generate_semantic_features(image_embeddings_np, model_path):
    print("Generating Semantic Features...")
    model = AutoModel.from_pretrained(model_path).to(cfg.DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Anchors
    concept_groups = {
        "bare": ["bare soil", "dirt ground", "sparse vegetation", "exposed earth"],
        "sparse": ["low density pasture", "thin grass", "short clipped grass"],
        "medium": ["average pasture cover", "medium height grass", "grazed pasture"],
        "dense": ["dense tall pasture", "thick grassy volume", "high biomass", "overgrown vegetation"],
        "green": ["lush green vibrant pasture", "photosynthesizing leaves", "fresh growth"],
        "dead": ["dry brown dead grass", "yellow straw", "senesced material", "standing hay"],
        "clover": ["white clover", "trifolium repens", "broadleaf legume", "clover flowers"],
        "grass": ["ryegrass", "blade-like leaves", "fescue", "grassy sward"]
    }
    
    # Encode Concepts
    concept_vectors = {}
    with torch.no_grad():
        for name, prompts in concept_groups.items():
            inputs = tokenizer(prompts, padding="max_length", return_tensors="pt").to(cfg.DEVICE)
            emb = model.get_text_features(**inputs)
            emb = emb / emb.norm(p=2, dim=-1, keepdim=True)
            concept_vectors[name] = emb.mean(dim=0, keepdim=True)
            
    # Compute Scores
    img_tensor = torch.tensor(image_embeddings_np, dtype=torch.float32).to(cfg.DEVICE)
    img_tensor = img_tensor / img_tensor.norm(p=2, dim=-1, keepdim=True)
    
    scores = {}
    for name, vec in concept_vectors.items():
        scores[name] = torch.matmul(img_tensor, vec.T).cpu().numpy().flatten()
        
    df_scores = pd.DataFrame(scores)
    # Ratios
    df_scores['ratio_greenness'] = df_scores['green'] / (df_scores['green'] + df_scores['dead'] + 1e-6)
    df_scores['ratio_clover'] = df_scores['clover'] / (df_scores['clover'] + df_scores['grass'] + 1e-6)
    df_scores['ratio_cover'] = (df_scores['dense'] + df_scores['medium']) / (df_scores['bare'] + df_scores['sparse'] + 1e-6)
    
    torch.cuda.empty_cache()
    return df_scores.values

# Combine for semantic generation to ensure consistency
all_emb = np.vstack([train_embeddings, test_embeddings])
all_semantic = generate_semantic_features(all_emb, cfg.SIGLIP_PATH)

sem_train = all_semantic[:len(train_df)]
sem_test = all_semantic[len(train_df):]

# =========================================================================================
# 5. SUPERVISED EMBEDDING ENGINE
# =========================================================================================
class SupervisedEmbeddingEngine:
    def __init__(self, n_pca=0.80, n_pls=8, n_gmm=6, random_state=42):
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_pca, random_state=random_state)
        self.pls = PLSRegression(n_components=n_pls, scale=False)
        self.gmm = GaussianMixture(n_components=n_gmm, covariance_type='diag', random_state=random_state)
        self.pls_fitted_ = False

    def fit(self, X, y=None, X_semantic=None):
        # Scale
        X_scaled = self.scaler.fit_transform(X)
        
        # Unsupervised
        self.pca.fit(X_scaled)
        self.gmm.fit(X_scaled)
        
        # Supervised
        if y is not None:
            self.pls.fit(X_scaled, y)
            self.pls_fitted_ = True
        return self

    def transform(self, X, X_semantic=None):
        X_scaled = self.scaler.transform(X)
        
        features = [self.pca.transform(X_scaled)]
        
        if self.pls_fitted_:
            features.append(self.pls.transform(X_scaled))
            
        features.append(self.gmm.predict_proba(X_scaled))
        
        if X_semantic is not None:
            # Normalize semantic
            sem_norm = (X_semantic - np.mean(X_semantic, axis=0)) / (np.std(X_semantic, axis=0) + 1e-6)
            features.append(sem_norm)
            
        return np.hstack(features)

# =========================================================================================
# 6. TRAINING & INFERENCE (5-FOLD CV)
# =========================================================================================
def cross_validate_predict(model_cls, model_params, train_data, test_data, sem_tr, sem_te, feature_engine):
    target_max_arr = np.array([cfg.TARGET_MAX[t] for t in cfg.TARGET_NAMES], dtype=float)
    y_pred_test_accum = np.zeros([len(test_data), len(cfg.TARGET_NAMES)], dtype=float)
    
    # Ensure n_splits is integer
    n_splits = int(train_data['fold'].nunique())
    
    # Pre-extract raw columns to avoid indexing overhead
    # Force float32 to save memory and ensure compatibility
    X_train_full = train_data[emb_cols].values.astype(np.float32)
    X_test_raw = test_data[emb_cols].values.astype(np.float32)
    y_train_full = train_data[cfg.TARGET_NAMES].values.astype(np.float32)
    
    for fold in range(n_splits):
        print(f"Processing Fold {fold}...")
        # Split
        train_mask = train_data['fold'] != fold
        
        X_tr = X_train_full[train_mask]
        y_tr = y_train_full[train_mask] / target_max_arr # Max Scaling
        
        sem_tr_fold = sem_tr[train_mask]
        
        # Feature Engineering (Fit on fold train)
        engine = deepcopy(feature_engine)
        engine.fit(X_tr, y=y_tr, X_semantic=sem_tr_fold)
        
        x_tr_eng = engine.transform(X_tr, X_semantic=sem_tr_fold)
        x_te_eng = engine.transform(X_test_raw, X_semantic=sem_te)
        
        # Train & Predict per target
        fold_test_pred = np.zeros([len(test_data), len(cfg.TARGET_NAMES)])
        
        for k in range(len(cfg.TARGET_NAMES)):
            target_name = cfg.TARGET_NAMES[k]
            
            # Dry_Clover_g: Always predict 0.0 (no training needed)
            if target_name == 'Dry_Clover_g':
                fold_test_pred[:, k] = 0.0
            else:
                # Train and predict for other targets
                model = model_cls(**model_params)
                model.fit(x_tr_eng, y_tr[:, k])
                pred_raw = model.predict(x_te_eng)
                fold_test_pred[:, k] = pred_raw * target_max_arr[k] # Inverse Scale
            
        y_pred_test_accum += fold_test_pred
        
    return y_pred_test_accum / n_splits

# Model Parameters (Optimized)
params_cat = {
    'iterations': 1900, 'learning_rate': 0.045, 'depth': 4, 'l2_leaf_reg': 0.56, 
    'random_strength': 0.045, 'bagging_temperature': 0.98, 'verbose': 0, 'random_state': 42,
    'allow_writing_files': False
}
params_xgb = { # Using GradientBoostingRegressor as proxy
    'n_estimators': 1354, 'learning_rate': 0.010, 'max_depth': 3, 'subsample': 0.60, 
    'random_state': 42
}
params_lgbm = {
    'n_estimators': 807, 'learning_rate': 0.014, 'num_leaves': 48, 'min_child_samples': 19, 
    'subsample': 0.745, 'colsample_bytree': 0.745, 'reg_alpha': 0.21, 'reg_lambda': 3.78,
    'verbose': -1, 'random_state': 42
}
params_hist = {
    'max_iter': 300, 'learning_rate': 0.05, 'max_depth': None, 'l2_regularization': 0.44,
    'random_state': 42
}

feat_engine = SupervisedEmbeddingEngine(n_pca=0.80, n_pls=8, n_gmm=6)

print("Training & Inferring Models...")

# 1. HistGradientBoosting
print("Model: HistGradientBoosting")
pred_hist = cross_validate_predict(
    HistGradientBoostingRegressor, params_hist, 
    train_feat_df, test_feat_df, sem_train, sem_test, feat_engine
)

# 2. GradientBoosting
print("Model: GradientBoosting")
pred_gb = cross_validate_predict(
    GradientBoostingRegressor, params_xgb, 
    train_feat_df, test_feat_df, sem_train, sem_test, feat_engine
)

# 3. CatBoost
print("Model: CatBoost")
pred_cat = cross_validate_predict(
    CatBoostRegressor, params_cat, 
    train_feat_df, test_feat_df, sem_train, sem_test, feat_engine
)

# 4. LightGBM
print("Model: LightGBM")
pred_lgbm = cross_validate_predict(
    LGBMRegressor, params_lgbm, 
    train_feat_df, test_feat_df, sem_train, sem_test, feat_engine
)

# =========================================================================================
# 7. ENSEMBLING & SUBMISSION
# =========================================================================================
print("Ensembling and Post-processing...")
# Simple Average Ensemble
final_pred = (pred_hist + pred_gb + pred_cat + pred_lgbm) / 4.0

# Assign to dataframe
test_feat_df[cfg.TARGET_NAMES] = final_pred

# Post-process (Mass Balance)
test_processed = post_process_biomass(test_feat_df)

# Create Submission File
sub_df = melt_table(test_processed)
output_path = "submission_siglip.csv"
sub_df.to_csv(output_path, index=False)

print(f"✓ Siglip submission generated: {output_path}")
print(sub_df.head())
```

```python
%%writefile csiro_infer.py
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, GroupKFold, StratifiedGroupKFold
from tqdm.auto import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T

import os
from pathlib import Path
import timm
import warnings 

warnings.filterwarnings('ignore')
tqdm.pandas()

class RegressionDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform
    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        image = item.image
        targets = [item['Dry_Green_g'], item['Dry_Clover_g'], item['Dry_Dead_g']]
        width, height = image.size
        mid_point = width // 2
        left_image = image.crop((0, 0, mid_point, height))
        right_image = image.crop((mid_point, 0, width, height))

        if self.transform is not None:
            left_image = self.transform(left_image)
            right_image = self.transform(right_image)

        return left_image, right_image, targets

def get_test_dataloaders(data, image_size, batch_size):
    res = []
    for trans in [None, T.RandomHorizontalFlip(p=1.0), T.RandomVerticalFlip(p=1.0)]:
        transform = T.Compose([
            T.Resize(image_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        if trans:
            transform = T.Compose([
                T.Resize(image_size),
                trans,
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        dataset = RegressionDataset(data, transform=transform)
        res.append(DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4))
    return res


class FiLM(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 2), 
            nn.ReLU(inplace=True), 
            nn.Linear(feat_dim // 2, feat_dim * 2)
        )

    def forward(self, context):
        gamma_beta = self.mlp(context)
        return torch.chunk(gamma_beta, 2, dim=1)

class CSIROModelRegressor(nn.Module):
    def __init__(self, model_name, pretrained=True, num_classes=3, dropout=0.0, freeze_backbone=False):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0, global_pool='avg')

        self.film = FiLM(self.backbone.num_features)

        self.dropout = nn.Dropout(dropout)

        def make_head():
            return nn.Sequential(
                nn.Linear(self.backbone.num_features * 2, 8),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(8, 1),
            )

        self.head_green = make_head()
        self.head_clover = make_head()
        self.head_dead = make_head()

        self.softplus = nn.Softplus(beta=1.0)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False


    def forward(self, left_img, right_img):
        left_feat = self.backbone(left_img)
        right_feat = self.backbone(right_img)

        context = (left_feat + right_feat) / 2
        gamma, beta = self.film(context)

        left_feat_modulated = left_feat * (1 + gamma) + beta
        right_feat_modulated = right_feat * (1 + gamma) + beta

        combined = torch.cat([left_feat_modulated, right_feat_modulated], dim=1)

        green = self.softplus(self.head_green(combined))   
        clover = self.softplus(self.head_clover(combined))
        dead = self.softplus(self.head_dead(combined)) 

        logits = torch.cat([green, clover, dead], dim=1)

        return logits

def predict(model, dataloader, device):
    model.to(device)
    model.eval()

    all_outputs = []
    with torch.no_grad():
        for left_images, right_images, targets in dataloader:
            left_images = left_images.to(device)
            right_images = right_images.to(device)

            outputs = model(left_images, right_images)
            all_outputs.append(outputs.detach().cpu())

    outputs = torch.cat(all_outputs).numpy()
    return outputs

def predict_loaders(model, dataloaders, device):
    all_outputs = []
    for dataloader in dataloaders:
        outputs = predict(model, dataloader, device)
        all_outputs.append(outputs)
    avg_outputs = np.mean(all_outputs, axis=0)
    return avg_outputs

def predict_folds(dataloaders,models_dir):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    all_preds = []
    for model_file in Path(models_dir).glob('*.pth'):
        model = CSIROModelRegressor(CFG.MODEL_NAME, pretrained=False, num_classes=3)
        model.load_state_dict(torch.load(model_file))
        preds = predict_loaders(model, dataloaders, device)
        all_preds.append(preds)

    avg_preds = np.mean(all_preds, axis=0)
    return avg_preds


class CFG:
    DATA_PATH="/kaggle/input/csiro-biomass/"
    TEST_DATA_PATH="/kaggle/input/csiro-biomass/test.csv"
    MODEL_NAME="vit_large_patch16_dinov3_qkvb"
    MODELS_DIR ='/kaggle/input/modelv3/pytorch/default/1/models_retrained'
    IMG_SIZE=(512,512)


test_df = pd.read_csv(CFG.TEST_DATA_PATH)

test_df['target'] = 0.0
test_df[['sample_id_prefix', 'sample_id_suffix']] = test_df.sample_id.str.split('__', expand=True)

test_data_df = test_df.groupby(['sample_id_prefix', 'image_path']).apply(lambda df: df.set_index('target_name').target)
test_data_df.reset_index(inplace=True)
test_data_df.columns.name = None

test_data_df['image'] = test_data_df.image_path.progress_apply(lambda path: Image.open(CFG.DATA_PATH + path).convert('RGB'))

test_loaders = get_test_dataloaders(test_data_df, CFG.IMG_SIZE, 32)
preds = predict_folds(test_loaders,models_dir=CFG.MODELS_DIR)

test_data_df[['Dry_Green_g', 'Dry_Clover_g', 'Dry_Dead_g']] = preds
test_data_df['Dry_Green_g'] = test_data_df['Dry_Green_g'].apply(lambda x: 0.0 if x < 0.09 else x)
test_data_df['Dry_Clover_g'] = test_data_df['Dry_Clover_g'].apply(lambda x: 0.0 if x < 0.09 else x)
test_data_df['Dry_Dead_g'] = test_data_df['Dry_Dead_g'].apply(lambda x: 0.0 if x < 0.09 else x)
test_data_df['Dry_Total_g'] = test_data_df.GDM_g + test_data_df.Dry_Dead_g

TARGETS = ['Dry_Green_g', 'Dry_Dead_g', 'Dry_Clover_g', 'GDM_g', 'Dry_Total_g']

submission_rows = []
for idx, row in test_data_df.iterrows():
    image_id = row['image_path'].split('/')[-1].replace('.jpg', '')
    
    for target_name in TARGETS:
        sample_id = f"{image_id}__{target_name}"
        prediction = row[target_name]
        
        # Post-processing
        if target_name == "Dry_Clover_g":
            prediction = prediction * 0.8
        elif target_name == "Dry_Dead_g":
            if prediction > 20:
                prediction *= 1.1
            elif prediction < 10:
                prediction *= 0.9
        
        submission_rows.append({
            'sample_id': sample_id,
            'target': prediction
        })

sub_df = pd.DataFrame(submission_rows)
sub_df.to_csv('submission_dinov2026.csv', index=False)

print(sub_df)
```

```python
!python csiro_infer.py
```

```python
import pandas as pd
import numpy as np
import os

# =============================================================================
# CONFIGURATION
# =============================================================================
# Weights for the ensemble (Must sum to 1.0)
# Adjust based on which model had better local CV or Public LB score.
# Example: If Siglip was better, give it 0.6.
W_SIGLIP = 0.35
W_DINO   = 0.65

FILES = {
    'siglip': 'submission_siglip.csv',
    'dino':   'submission_dinov2026.csv'
}

OUTPUT_FILE = 'submission72.csv'

# Target definitions required for Mass Balance
ALL_TARGETS = ['Dry_Green_g', 'Dry_Clover_g', 'Dry_Dead_g', 'GDM_g', 'Dry_Total_g']

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def enforce_mass_balance(df_wide, fixed_clover=None):
    """
    Applies Orthogonal Projection to enforce biological constraints:
    1. Dry_Green_g + Dry_Clover_g = GDM_g
    2. GDM_g + Dry_Dead_g = Dry_Total_g
    
    If fixed_clover is True, Dry_Clover_g is kept fixed and only other targets are adjusted.
    
    This finds the closest set of values to the predictions that satisfy 
    the constraints (minimizing Euclidean distance modification).
    """
    # 1. Ensure columns are in the specific order for the matrix math
    # Vector x = [Green, Clover, Dead, GDM, Total]
    ordered_cols = ['Dry_Green_g', 'Dry_Clover_g', 'Dry_Dead_g', 'GDM_g', 'Dry_Total_g']
    
    # Extract values: Shape (5, N_samples)
    Y = df_wide[ordered_cols].values.T
    
    if fixed_clover:
        # Keep Dry_Clover_g fixed, adjust only other targets
        # We have: Green + Clover_fixed = GDM, so GDM = Green + Clover_fixed
        # And: GDM + Dead = Total, so Total = GDM + Dead = Green + Clover_fixed + Dead
        # Extract fixed Clover values
        clover_fixed = Y[1, :].copy()  # Dry_Clover_g is index 1
        # Adjust: GDM = Green + Clover_fixed
        Y[3, :] = Y[0, :] + clover_fixed  # GDM_g
        # Adjust: Total = GDM + Dead = Green + Clover_fixed + Dead
        Y[4, :] = Y[3, :] + Y[2, :]  # Dry_Total_g
        # Keep Green and Dead as is (or apply minimal adjustment if needed)
        # Clover stays fixed (already set)
        Y_reconciled = Y
    else:
        # Original method: adjust all targets
        # 2. Define Constraint Matrix C where Cx = 0
        # Eq 1: 1*Gr + 1*Cl + 0*De - 1*GDM + 0*Tot = 0
        # Eq 2: 0*Gr + 0*Cl + 1*De + 1*GDM - 1*Tot = 0
        C = np.array([
            [1, 1, 0, -1,  0],
            [0, 0, 1,  1, -1]
        ])
        
        # 3. Calculate Projection Matrix P = I - C^T * (C * C^T)^-1 * C
        C_T = C.T
        try:
            inv_CCt = np.linalg.inv(C @ C_T)
            P = np.eye(5) - C_T @ inv_CCt @ C
        except np.linalg.LinAlgError:
            # Fallback if singular (unlikely with this specific matrix)
            print("Warning: Singular matrix in projection. Skipping constraint enforcement.")
            return df_wide

        # 4. Apply Projection
        Y_reconciled = P @ Y
    
    # 5. Transpose back to (N_samples, 5) and clip negatives
    Y_reconciled = Y_reconciled.T
    Y_reconciled = np.maximum(0, Y_reconciled) 
    
    # 6. Update DataFrame
    df_out = df_wide.copy()
    df_out[ordered_cols] = Y_reconciled
    
    return df_out

def robust_ensemble(file_paths, weights):
    print(f"--- Starting Ensemble ---")
    print(f"Weights: {weights}")
    print("NOTE: Using DINO-only for Dry_Clover_g (better detection)")
    
    dfs = []
    for name, path in file_paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing file: {path}")
        
        # Read and sort by sample_id to ensure alignment
        df = pd.read_csv(path).sort_values('sample_id').reset_index(drop=True)
        dfs.append(df)
        print(f"Loaded {name}: {len(df)} rows")

    # 1. Check alignment
    base_ids = dfs[0]['sample_id']
    if not all(df['sample_id'].equals(base_ids) for df in dfs[1:]):
        raise ValueError("Sample IDs do not match between submission files!")

    # 2. Split by target_name to handle Dry_Clover_g separately
    # Split sample_id into image_id and target_name
    dfs_split = []
    for df in dfs:
        df_split = df.copy()
        df_split[['image_id', 'target_name']] = df_split['sample_id'].str.rsplit('__', n=1, expand=True)
        dfs_split.append(df_split)
    
    # Get DINO and SigLIP dataframes (identify by order in file_paths)
    dino_idx = list(file_paths.keys()).index('dino')
    siglip_idx = list(file_paths.keys()).index('siglip')
    dino_df = dfs_split[dino_idx]
    siglip_df = dfs_split[siglip_idx]
    
    # 3. Separate targets: Dry_Clover_g uses DINO only, others use ensemble
    ensemble_results = []
    
    for target in ALL_TARGETS:
        # Filter for this target
        dino_target = dino_df[dino_df['target_name'] == target].copy()
        siglip_target = siglip_df[siglip_df['target_name'] == target].copy()
        
        if target == 'Dry_Clover_g':
            # Use DINO only for Clover
            print(f"Using DINO-only for {target}")
            ensemble_results.append(dino_target[['sample_id', 'target']])
        else:
            # Ensemble for other targets
            # Merge on sample_id to align
            merged = pd.merge(
                dino_target[['sample_id', 'target']].rename(columns={'target': 'dino_pred'}),
                siglip_target[['sample_id', 'target']].rename(columns={'target': 'siglip_pred'}),
                on='sample_id'
            )
            
            # Weighted average
            w_vec = np.array([weights['dino'], weights['siglip']])
            w_vec = w_vec / w_vec.sum()
            
            merged['target'] = merged['dino_pred'] * w_vec[0] + merged['siglip_pred'] * w_vec[1]
            ensemble_results.append(merged[['sample_id', 'target']])
    
    # 4. Combine all targets
    ensemble_df = pd.concat(ensemble_results, ignore_index=True)
    print("Weighted average complete (DINO-only for Clover).")

    # 5. Prepare for Mass Balance (Convert Long -> Wide)
    ensemble_df[['image_id', 'target_name']] = ensemble_df['sample_id'].str.rsplit('__', n=1, expand=True)
    
    # Pivot
    wide_df = ensemble_df.pivot(index='image_id', columns='target_name', values='target').reset_index()

    # 6. Apply Robust Constraints (with Dry_Clover_g fixed)
    print("Applying Mass Balance Constraints (Dry_Clover_g fixed to DINO values)...")
    wide_balanced = enforce_mass_balance(wide_df, fixed_clover=True)

    # 7. Convert back (Wide -> Long)
    long_balanced = wide_balanced.melt(
        id_vars='image_id', 
        value_vars=ALL_TARGETS,
        var_name='target_name',
        value_name='target'
    )

    # Reconstruct sample_id
    long_balanced['sample_id'] = long_balanced['image_id'] + '__' + long_balanced['target_name']

    # 8. Final Formatting
    final_submission = long_balanced[['sample_id', 'target']].sort_values('sample_id').reset_index(drop=True)

    return final_submission

# =============================================================================
# EXECUTION
# =============================================================================
if __name__ == "__main__":
    
    # Define weights
    ensemble_weights = {
        'siglip': W_SIGLIP,
        'dino': W_DINO
    }
    
    try:
        # Run Ensemble
        submission = robust_ensemble(FILES, ensemble_weights)
        
        # Save
        submission.to_csv(OUTPUT_FILE, index=False)
        print(f"\nSuccess! Saved to {OUTPUT_FILE}")
        print(submission.head())
        
        # Sanity Check Stats
        print("\nStats:")
        print(submission['target'].describe())
        
    except Exception as e:
        print(f"\nError during ensembling: {e}")
```

```python
# now import timm fresh
import timm, sys
print("python:", sys.version)
print("timm:", timm.__version__)
print("timm file:", timm.__file__)
print("dinov3 matches:", timm.list_models("*dinov3*")[:50])
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
import time
from timm.utils import ModelEmaV2
from sklearn.model_selection import KFold, StratifiedGroupKFold
import matplotlib.pyplot as plt
import seaborn as sns

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")

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
    SUBMISSION_DIR  = './'
    MODEL_DIR = '/kaggle/input/baseline-dinov3/pytorch/default/1/baseline-dinov3'
    N_FOLDS         = 5

    MODEL_NAME      = 'vit_huge_plus_patch16_dinov3.lvd1689m'
    BACKBONE_PATH   = '/kaggle/input/vit-huge-plus-patch16-dinov3-lvd1689m/vit_huge_plus_patch16_dinov3.lvd1689m_backbone.pth'
    DINO_GRAD_CHECKPOINTING = False

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
    CLIP_GRAD_NORM  = 1.0
    PATIENCE        = 5
    TARGET_COLS     = ['Dry_Total_g', 'GDM_g', 'Dry_Green_g']
    DERIVED_COLS    = ['Dry_Clover_g', 'Dry_Dead_g']
    ALL_TARGET_COLS = ['Dry_Green_g','Dry_Dead_g','Dry_Clover_g','GDM_g','Dry_Total_g']
    R2_WEIGHTS      = np.array([0.1, 0.1, 0.1, 0.2, 0.5])
    LOSS_WEIGHTS    = np.array([0.1, 0.1, 0.1, 0.0, 0.0])
    DEVICE          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

os.makedirs(CFG.MODEL_DIR, exist_ok=True)
print(f'Device : {CFG.DEVICE}')
print(f'Backbone: {CFG.MODEL_NAME} | Input: {CFG.IMG_SIZE}')
print(f'Freeze Epochs: {CFG.FREEZE_EPOCHS} | Warmup: {CFG.WARMUP_EPOCHS}')
print(f'EMA Decay: {CFG.EMA_DECAY} | Grad Acc: {CFG.GRAD_ACC}')
```

```python
# Metrics
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

```python
# Transforms
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

```python
# Layers
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
```

```python
# Model
class BiomassModel(nn.Module):
    def __init__(self, model_name, pretrained=True, backbone_path=None):
        super().__init__()
        self.model_name = model_name
        self.backbone_path = backbone_path
        
        # 1. Load Backbone with global_pool='' to keep patch tokens
        #    (B, 197, 1024) instead of (B, 1024)
        self.backbone = timm.create_model(self.model_name, pretrained=False, num_classes=0, global_pool='')
        
        # 2. Enable Gradient Checkpointing (Crucial for ViT-Large memory!)
        if hasattr(self.backbone, 'set_grad_checkpointing') and CFG.DINO_GRAD_CHECKPOINTING:
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
```

```python
# Utility Functions
def set_backbone_requires_grad(model: BiomassModel, requires_grad: bool):
    for p in model.backbone.parameters():
        p.requires_grad = requires_grad


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
```

```python
# Training and Validation Loops 

from contextlib import nullcontext

USE_BF16 = True
AMP_DTYPE = torch.bfloat16 if USE_BF16 else torch.float16

scaler = torch.amp.GradScaler(
    'cuda',
    enabled=(torch.cuda.is_available() and AMP_DTYPE == torch.float16)
)

def autocast_ctx():
    if not torch.cuda.is_available():
        return nullcontext()
    return torch.amp.autocast(device_type='cuda', dtype=AMP_DTYPE)

def _as_col(x: torch.Tensor, bs: int) -> torch.Tensor:
    """Force predictions to (B,1) regardless of model head returning (B,) or (B,1)."""
    if x.ndim == 1:
        return x.view(bs, 1)
    if x.ndim == 2:
        return x
    return x.view(bs, 1)

def _ensure_2d_lab(lab: torch.Tensor, bs: int) -> torch.Tensor:
    """Force labels to (B,T)."""
    if lab.ndim == 1:
        return lab.view(bs, -1)
    return lab

def _pred_pack(p_total, p_gdm, p_green, p_clover, p_dead, bs: int) -> torch.Tensor:
    """
    Pack predictions in the exact order expected by your metric:
    CFG.ALL_TARGET_COLS = ['Dry_Green_g','Dry_Dead_g','Dry_Clover_g','GDM_g','Dry_Total_g']
    """
    pg = _as_col(p_green,  bs)
    pd = _as_col(p_dead,   bs)
    pc = _as_col(p_clover, bs)
    pgdm = _as_col(p_gdm,  bs)
    pt = _as_col(p_total,  bs)
    return torch.cat([pg, pd, pc, pgdm, pt], dim=1)  # (B,5)

@torch.inference_mode()
def valid_epoch(eval_model, loader, device):
    eval_model.eval()

    n = len(loader.dataset)
    n_targets = len(CFG.ALL_TARGET_COLS)

    preds_cpu  = torch.empty((n, n_targets), dtype=torch.float32)
    labels_cpu = torch.empty((n, n_targets), dtype=torch.float32)

    total_loss = 0.0
    offset = 0

    for l, r, lab in loader:
        bs = l.size(0)
        l   = l.to(device, non_blocking=True)
        r   = r.to(device, non_blocking=True)
        lab = lab.to(device, non_blocking=True)

        with autocast_ctx():
            p_total, p_gdm, p_green, p_clover, p_dead = eval_model(l, r)
            loss = biomass_loss((p_total, p_gdm, p_green, p_clover, p_dead),
                                lab, w=CFG.LOSS_WEIGHTS)

        total_loss += loss.detach().float().item() * bs

        batch_pred = _pred_pack(p_total, p_gdm, p_green, p_clover, p_dead, bs).float().cpu()
        batch_lab  = _ensure_2d_lab(lab, bs).float().cpu()

        # Safety: ensure dims match (avoids silent 4/5 bugs)
        if batch_pred.shape[1] != n_targets or batch_lab.shape[1] != n_targets:
            raise RuntimeError(
                f"Target dim mismatch: pred={batch_pred.shape}, lab={batch_lab.shape}, "
                f"CFG.ALL_TARGET_COLS={n_targets}"
            )

        preds_cpu[offset:offset+bs]  = batch_pred
        labels_cpu[offset:offset+bs] = batch_lab
        offset += bs

    pred_all    = preds_cpu.numpy()
    true_labels = labels_cpu.numpy()
    global_r2, avg_r2, per_r2 = weighted_r2_score_global(true_labels, pred_all)

    return total_loss / n, global_r2, avg_r2, per_r2, pred_all, true_labels


@torch.inference_mode()
def valid_epoch_tta(eval_model, loaders, device):
    """
    Fixed: always accumulate (N,5). Works even if a head outputs (B,1).
    Assumes each loader iterates in identical order (shuffle=False).
    """
    eval_model.eval()
    assert len(loaders) > 0

    n = len(loaders[0].dataset)
    n_targets = len(CFG.ALL_TARGET_COLS)

    labels_cpu = torch.empty((n, n_targets), dtype=torch.float32)
    preds_sum  = torch.zeros((n, n_targets), dtype=torch.float32)

    total_loss = 0.0

    for tta_i, loader in enumerate(loaders):
        offset = 0
        tta_loss = 0.0

        for l, r, lab in loader:
            bs = l.size(0)
            l   = l.to(device, non_blocking=True)
            r   = r.to(device, non_blocking=True)
            lab = lab.to(device, non_blocking=True)

            with autocast_ctx():
                p_total, p_gdm, p_green, p_clover, p_dead = eval_model(l, r)
                loss = biomass_loss((p_total, p_gdm, p_green, p_clover, p_dead),
                                    lab, w=CFG.LOSS_WEIGHTS)

            tta_loss += loss.detach().float().item() * bs

            batch_pred = _pred_pack(p_total, p_gdm, p_green, p_clover, p_dead, bs).float().cpu()
            preds_sum[offset:offset+bs] += batch_pred

            if tta_i == 0:
                batch_lab = _ensure_2d_lab(lab, bs).float().cpu()
                if batch_lab.shape[1] != n_targets:
                    raise RuntimeError(f"Label dim mismatch: {batch_lab.shape} vs {n_targets}")
                labels_cpu[offset:offset+bs] = batch_lab

            offset += bs

        total_loss += tta_loss / n

    avg_preds   = (preds_sum / len(loaders)).numpy()
    true_labels = labels_cpu.numpy()
    global_r2, avg_r2, per_r2 = weighted_r2_score_global(true_labels, avg_preds)

    return total_loss / len(loaders), global_r2, avg_r2, per_r2, avg_preds, true_labels


def train_epoch(model, loader, opt, scheduler, device, ema: ModelEmaV2 | None = None):
    model.train()
    running = 0.0

    opt.zero_grad(set_to_none=True)
    itera = tqdm(loader, desc='train', leave=False) if CFG.USE_TQDM else loader

    for i, (l, r, lab) in enumerate(itera):
        bs = l.size(0)
        l   = l.to(device, non_blocking=True)
        r   = r.to(device, non_blocking=True)
        lab = lab.to(device, non_blocking=True)

        with autocast_ctx():
            p_total, p_gdm, p_green, p_clover, p_dead = model(l, r)
            loss = biomass_loss((p_total, p_gdm, p_green, p_clover, p_dead),
                                lab, w=CFG.LOSS_WEIGHTS)
            loss = loss / CFG.GRAD_ACC


        if not torch.isfinite(loss):
            print("Non-finite loss detected; skipping batch")
            opt.zero_grad(set_to_none=True)
            continue

        if scaler.is_enabled():
            scaler.scale(loss).backward()
        else:
            loss.backward()

        running += loss.detach().float().item() * bs * CFG.GRAD_ACC

        do_step = ((i + 1) % CFG.GRAD_ACC == 0) or ((i + 1) == len(loader))
        if do_step:
            if scaler.is_enabled():
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

            if ema is not None:
                ema.update(model.module if hasattr(model, "module") else model)

            opt.zero_grad(set_to_none=True)

    scheduler.step()
    return running / len(loader.dataset)
```

```python
# # === MAIN TRAINING LOOP === #

# # Helper for accurate GPU timings
# def _sync():
#     if torch.cuda.is_available():
#         torch.cuda.synchronize()

# print('Loading data...')
# df_long = pd.read_csv(CFG.TRAIN_CSV)

# df_wide = (
#     df_long.pivot(index='image_path', columns='target_name', values='target')
#     .reset_index()
# )
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

# # One place for loader kwargs (fast)
# DL_KW = dict(
#     num_workers=CFG.NUM_WORKERS,
#     pin_memory=True,
#     persistent_workers=(CFG.NUM_WORKERS > 0),
#     prefetch_factor=4 if CFG.NUM_WORKERS > 0 else None,
# )

# for fold, (tr_idx, val_idx) in enumerate(sgkf.split(df_wide, y_stratify, groups=groups)):
#     if fold not in CFG.FOLDS_TO_TRAIN:
#         print(f'Skipping fold {fold} as per configuration.')
#         continue

#     print('\n' + '='*70)
#     print(f'FOLD {fold+1}/{CFG.N_FOLDS} | {len(tr_idx)} train / {len(val_idx)} val')
#     print('='*70)

#     # NOTE: avoid empty_cache/gc inside epoch loop; only between folds if you really want
#     _sync()
#     torch.cuda.empty_cache()
#     gc.collect()

#     tr_df  = df_wide.iloc[tr_idx].reset_index(drop=True)
#     val_df = df_wide.iloc[val_idx].reset_index(drop=True)

#     tr_set = BiomassDataset(tr_df, get_train_transforms(), CFG.TRAIN_IMAGE_DIR)

#     # Create TTA loaders (keep TTAs as requested)
#     val_loaders = []
#     for mode in range(CFG.VAL_TTA_TIMES):  # 0: orig, 1: hflip, 2: vflip, 3: rot90
#         val_set_tta = BiomassDataset(val_df, get_tta_transforms(mode), CFG.TRAIN_IMAGE_DIR)
#         val_loader_tta = DataLoader(
#             val_set_tta,
#             batch_size=CFG.BATCH_SIZE,
#             shuffle=False,
#             drop_last=False,
#             **{k: v for k, v in DL_KW.items() if v is not None},
#         )
#         val_loaders.append(val_loader_tta)

#     tr_loader = DataLoader(
#         tr_set,
#         batch_size=CFG.BATCH_SIZE,
#         shuffle=True,
#         drop_last=True,
#         **{k: v for k, v in DL_KW.items() if v is not None},
#     )

#     print('Building model...')
#     backbone_path = getattr(CFG, 'BACKBONE_PATH', None)
#     model = BiomassModel(CFG.MODEL_NAME, pretrained=CFG.PRETRAINED, backbone_path=backbone_path).to(CFG.DEVICE)

#     # Load pretrained fold weights if available (for resuming or fine-tuning)
#     if getattr(CFG, 'PRETRAINED_DIR', None) and os.path.isdir(CFG.PRETRAINED_DIR):
#         pretrained_path = os.path.join(CFG.PRETRAINED_DIR, f'best_model_fold{fold}.pth')
#         if os.path.exists(pretrained_path):
#             try:
#                 state = torch.load(pretrained_path, map_location='cpu')
#                 if isinstance(state, dict) and ('model_state_dict' in state or 'state_dict' in state):
#                     key = 'model_state_dict' if 'model_state_dict' in state else 'state_dict'
#                     sd = state[key]
#                 else:
#                     sd = state
#                 model.load_state_dict(sd, strict=False)
#                 model.to(CFG.DEVICE)
#                 print(f'  ✓ Loaded pretrained weights for fold {fold} from {pretrained_path}')
#             except Exception as e:
#                 print(f'  ✗ Failed to load pretrained fold {fold}: {e}')
#         else:
#             print(f'  (No pretrained file for fold {fold} at {pretrained_path})')
#     else:
#         print('  (No PRETRAINED_DIR configured or directory missing)')

#     # Single GPU: DO NOT wrap in DataParallel
#     # model = nn.DataParallel(model)  # <-- removed

#     # Freeze/unfreeze backbone
#     set_backbone_requires_grad(model, False)

#     optimizer = build_optimizer(model)
#     scheduler = build_scheduler(optimizer)

#     # EMA on the real model
#     ema = ModelEmaV2(model, decay=CFG.EMA_DECAY)

#     best_global_r2 = -np.inf
#     best_avg_r2 = -np.inf
#     patience = 0
#     best_fold_preds = None
#     best_fold_true = None

#     save_path = os.path.join(CFG.MODEL_DIR, f'best_model_fold{fold}.pth')

#     for epoch in range(1, CFG.EPOCHS + 1):
#         if epoch == CFG.FREEZE_EPOCHS + 1:
#             patience = 0
#             set_backbone_requires_grad(model, True)
#             print(f'Epoch {epoch}: backbone unfrozen')

#         # ---- Train timing ----
#         _sync()
#         t0 = time.perf_counter()
#         tr_loss = train_epoch(model, tr_loader, optimizer, scheduler, CFG.DEVICE, ema)
#         _sync()
#         t1 = time.perf_counter()

#         # choose eval model (EMA weights)
#         eval_model = ema.module if ema is not None else model

#         # ---- Val timing (TTA) ----
#         _sync()
#         t2 = time.perf_counter()
#         val_loss, global_r2, avg_r2, per_r2, preds_fold, true_fold = valid_epoch_tta(eval_model, val_loaders, CFG.DEVICE)
#         _sync()
#         t3 = time.perf_counter()

#         time_tr = t1 - t0
#         time_val = t3 - t2
#         time_ep = t3 - t0

#         per_r2_str = ' | '.join([f'{CFG.ALL_TARGET_COLS[i][:5]}: {r2:.3f}' for i, r2 in enumerate(per_r2)])
#         lrs = [pg['lr'] for pg in optimizer.param_groups]
#         lr_str = ' '.join([f'lr{i}={lr:.3e}' for i, lr in enumerate(lrs)])

#         print(
#             f'Fold {fold} | Epoch {epoch:02d} | '
#             f'TLoss {tr_loss:.5f} | VLoss {val_loss:.5f} | '
#             f'avgR2 {avg_r2:.4f} | GlobalR² {global_r2:.4f} '
#             f'{"[BEST]" if global_r2 > best_global_r2 else ""} | '
#             f'{lr_str} | time_tr={time_tr:.1f}s time_val={time_val:.1f}s time_ep={time_ep:.1f}s'
#         )
#         print(f'  → {per_r2_str}')

#         if global_r2 > best_global_r2:
#             best_global_r2 = global_r2
#             best_avg_r2 = avg_r2

#             # Save EMA weights directly (no CPU clone) for speed
#             # NOTE: safest is to save state_dict tensors as-is; this is typically fine on DGX.
#             torch.save(eval_model.state_dict(), save_path)
#             print(f'  → SAVED EMA weights to {save_path} (GlobalR²: {best_global_r2:.4f})')

#             patience = 0
#             best_fold_preds = preds_fold
#             best_fold_true = true_fold
#         else:
#             patience += 1
#             if patience >= CFG.PATIENCE:
#                 print(f'  → EARLY STOP (no improvement in {CFG.PATIENCE} epochs)')
#                 break

#         # keep memory tidy but avoid heavy cache/gc churn
#         del preds_fold, true_fold

#     if best_fold_preds is not None:
#         oof_true.append(best_fold_true)
#         oof_pred.append(best_fold_preds)
#         fold_summary.append({'fold': fold, 'global_r2': best_global_r2, 'avg_r2': best_avg_r2})

#     # Cleanup for this fold
#     del model, tr_loader, val_loaders, optimizer, scheduler, ema, eval_model
#     _sync()
#     torch.cuda.empty_cache()
#     gc.collect()

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

```python
# Inference   

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
        model_dir = CFG.MODEL_DIR
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
    output_path = os.path.join(CFG.SUBMISSION_DIR, 'submission70.csv')
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

```python
import pandas as pd
import numpy as np
import os

# =============================================================================
# CONFIGURATION
# =============================================================================
# Weights for the ensemble (Must sum to 1.0)
# Adjust based on which model had better local CV or Public LB score.
# Example: If Siglip was better, give it 0.6.
W_SIGLIP = 0.35
W_DINO   = 0.65

FILES = {
    'siglip': 'submission70.csv',
    'dino':   'submission72.csv'
}

OUTPUT_FILE = 'submission.csv'

# Target definitions required for Mass Balance
ALL_TARGETS = ['Dry_Green_g', 'Dry_Clover_g', 'Dry_Dead_g', 'GDM_g', 'Dry_Total_g']

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def enforce_mass_balance(df_wide, fixed_clover=None):
    """
    Applies Orthogonal Projection to enforce biological constraints:
    1. Dry_Green_g + Dry_Clover_g = GDM_g
    2. GDM_g + Dry_Dead_g = Dry_Total_g
    
    If fixed_clover is True, Dry_Clover_g is kept fixed and only other targets are adjusted.
    
    This finds the closest set of values to the predictions that satisfy 
    the constraints (minimizing Euclidean distance modification).
    """
    # 1. Ensure columns are in the specific order for the matrix math
    # Vector x = [Green, Clover, Dead, GDM, Total]
    ordered_cols = ['Dry_Green_g', 'Dry_Clover_g', 'Dry_Dead_g', 'GDM_g', 'Dry_Total_g']
    
    # Extract values: Shape (5, N_samples)
    Y = df_wide[ordered_cols].values.T
    
    if fixed_clover:
        # Keep Dry_Clover_g fixed, adjust only other targets
        # We have: Green + Clover_fixed = GDM, so GDM = Green + Clover_fixed
        # And: GDM + Dead = Total, so Total = GDM + Dead = Green + Clover_fixed + Dead
        # Extract fixed Clover values
        clover_fixed = Y[1, :].copy()  # Dry_Clover_g is index 1
        # Adjust: GDM = Green + Clover_fixed
        Y[3, :] = Y[0, :] + clover_fixed  # GDM_g
        # Adjust: Total = GDM + Dead = Green + Clover_fixed + Dead
        Y[4, :] = Y[3, :] + Y[2, :]  # Dry_Total_g
        # Keep Green and Dead as is (or apply minimal adjustment if needed)
        # Clover stays fixed (already set)
        Y_reconciled = Y
    else:
        # Original method: adjust all targets
        # 2. Define Constraint Matrix C where Cx = 0
        # Eq 1: 1*Gr + 1*Cl + 0*De - 1*GDM + 0*Tot = 0
        # Eq 2: 0*Gr + 0*Cl + 1*De + 1*GDM - 1*Tot = 0
        C = np.array([
            [1, 1, 0, -1,  0],
            [0, 0, 1,  1, -1]
        ])
        
        # 3. Calculate Projection Matrix P = I - C^T * (C * C^T)^-1 * C
        C_T = C.T
        try:
            inv_CCt = np.linalg.inv(C @ C_T)
            P = np.eye(5) - C_T @ inv_CCt @ C
        except np.linalg.LinAlgError:
            # Fallback if singular (unlikely with this specific matrix)
            print("Warning: Singular matrix in projection. Skipping constraint enforcement.")
            return df_wide

        # 4. Apply Projection
        Y_reconciled = P @ Y
    
    # 5. Transpose back to (N_samples, 5) and clip negatives
    Y_reconciled = Y_reconciled.T
    Y_reconciled = np.maximum(0, Y_reconciled) 
    
    # 6. Update DataFrame
    df_out = df_wide.copy()
    df_out[ordered_cols] = Y_reconciled
    
    return df_out

def robust_ensemble(file_paths, weights):
    print(f"--- Starting Ensemble ---")
    print(f"Weights: {weights}")
    print("NOTE: Using DINO-only for Dry_Clover_g (better detection)")
    
    dfs = []
    for name, path in file_paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing file: {path}")
        
        # Read and sort by sample_id to ensure alignment
        df = pd.read_csv(path).sort_values('sample_id').reset_index(drop=True)
        dfs.append(df)
        print(f"Loaded {name}: {len(df)} rows")

    # 1. Check alignment
    base_ids = dfs[0]['sample_id']
    if not all(df['sample_id'].equals(base_ids) for df in dfs[1:]):
        raise ValueError("Sample IDs do not match between submission files!")

    # 2. Split by target_name to handle Dry_Clover_g separately
    # Split sample_id into image_id and target_name
    dfs_split = []
    for df in dfs:
        df_split = df.copy()
        df_split[['image_id', 'target_name']] = df_split['sample_id'].str.rsplit('__', n=1, expand=True)
        dfs_split.append(df_split)
    
    # Get DINO and SigLIP dataframes (identify by order in file_paths)
    dino_idx = list(file_paths.keys()).index('dino')
    siglip_idx = list(file_paths.keys()).index('siglip')
    dino_df = dfs_split[dino_idx]
    siglip_df = dfs_split[siglip_idx]
    
    # 3. Separate targets: Dry_Clover_g uses DINO only, others use ensemble
    ensemble_results = []
    
    for target in ALL_TARGETS:
        # Filter for this target
        dino_target = dino_df[dino_df['target_name'] == target].copy()
        siglip_target = siglip_df[siglip_df['target_name'] == target].copy()
        
        if target == 'Dry_Clover_g':
            # Use DINO only for Clover
            print(f"Using DINO-only for {target}")
            ensemble_results.append(dino_target[['sample_id', 'target']])
        else:
            # Ensemble for other targets
            # Merge on sample_id to align
            merged = pd.merge(
                dino_target[['sample_id', 'target']].rename(columns={'target': 'dino_pred'}),
                siglip_target[['sample_id', 'target']].rename(columns={'target': 'siglip_pred'}),
                on='sample_id'
            )
            
            # Weighted average
            w_vec = np.array([weights['dino'], weights['siglip']])
            w_vec = w_vec / w_vec.sum()
            
            merged['target'] = merged['dino_pred'] * w_vec[0] + merged['siglip_pred'] * w_vec[1]
            ensemble_results.append(merged[['sample_id', 'target']])
    
    # 4. Combine all targets
    ensemble_df = pd.concat(ensemble_results, ignore_index=True)
    print("Weighted average complete (DINO-only for Clover).")

    # 5. Prepare for Mass Balance (Convert Long -> Wide)
    ensemble_df[['image_id', 'target_name']] = ensemble_df['sample_id'].str.rsplit('__', n=1, expand=True)
    
    # Pivot
    wide_df = ensemble_df.pivot(index='image_id', columns='target_name', values='target').reset_index()

    # 6. Apply Robust Constraints (with Dry_Clover_g fixed)
    print("Applying Mass Balance Constraints (Dry_Clover_g fixed to DINO values)...")
    wide_balanced = enforce_mass_balance(wide_df, fixed_clover=True)

    # 7. Convert back (Wide -> Long)
    long_balanced = wide_balanced.melt(
        id_vars='image_id', 
        value_vars=ALL_TARGETS,
        var_name='target_name',
        value_name='target'
    )

    # Reconstruct sample_id
    long_balanced['sample_id'] = long_balanced['image_id'] + '__' + long_balanced['target_name']

    # 8. Final Formatting
    final_submission = long_balanced[['sample_id', 'target']].sort_values('sample_id').reset_index(drop=True)

    return final_submission

# =============================================================================
# EXECUTION
# =============================================================================
if __name__ == "__main__":
    
    # Define weights
    ensemble_weights = {
        'siglip': W_SIGLIP,
        'dino': W_DINO
    }
    
    try:
        # Run Ensemble
        submission = robust_ensemble(FILES, ensemble_weights)
        
        # Save
        submission.to_csv(OUTPUT_FILE, index=False)
        print(f"\nSuccess! Saved to {OUTPUT_FILE}")
        print(submission.head())
        
        # Sanity Check Stats
        print("\nStats:")
        print(submission['target'].describe())
        
    except Exception as e:
        print(f"\nError during ensembling: {e}")
```