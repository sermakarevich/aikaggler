# CMI3 Pyroch Baseline+model_add, Aug, folds

- **Author:** MYSO
- **Votes:** 418
- **Ref:** myso1987/cmi3-pyroch-baseline-model-add-aug-folds
- **URL:** https://www.kaggle.com/code/myso1987/cmi3-pyroch-baseline-model-add-aug-folds
- **Last run:** 2025-07-01 12:49:32.243000

---

This notebook is based on [Salman Ahmed](https://www.kaggle.com/salmanahmedtamu)'s "[PYTORCH TRAINING / INFER PIPELINE 0.72](https://www.kaggle.com/code/salmanahmedtamu/pytorch-training-infer-pipeline-0-72)", with modifications to the model, additional data augmentation, and the addition of 5-fold cross-validation.

```python
import os, json, joblib, numpy as np, pandas as pd
import random, math
from pathlib import Path
import warnings 
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import StratifiedKFold
from timm.scheduler import CosineLRScheduler
from scipy.signal import firwin
from cmi_2025_metric_copy_for_import import CompetitionMetric

import polars as pl
# Configuration
TRAIN = False                     # ← set to True when you want to train
RAW_DIR = Path("../input/cmi-detect-behavior-with-sensor-data")
PRETRAINED_DIR = Path("/kaggle/input/cmi3-models-p") # used when TRAIN=False
EXPORT_DIR = Path("./")                                    # artefacts will be saved here
BATCH_SIZE = 64
PAD_PERCENTILE = 100
maxlen = PAD_PERCENTILE
LR_INIT = 1e-3
WD = 3e-3
# MIXUP_ALPHA = 0.4
PATIENCE = 40
FOLDS = 5
random_state = 42
epochs_warmup = 20
warmup_lr_init = 1.822126131809773e-05
lr_min = 3.810323058740104e-09

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"▶ imports ready · pytorch {torch.__version__} · device: {device}")

# ================================
# Model Components
# ================================
mean = torch.tensor([
    0,  0, 0, 0, 0,
    0,  9.0319e-03,  1.0849e+00, -2.6186e-03,  3.7651e-03,
    -5.3660e-03, -2.8177e-03,  1.3318e-03, -1.5876e-04,  6.3495e-01,
     6.2877e-01,  6.0607e-01,  6.2142e-01,  6.3808e-01,  6.5420e-01,
     7.4102e-03, -3.4159e-03, -7.5237e-03, -2.6034e-02,  2.9704e-02,
    -3.1546e-02, -2.0610e-03, -4.6986e-03, -4.7216e-03, -2.6281e-02,
     1.5799e-02,  1.0016e-02
], dtype=torch.float32).view(1, -1, 1).to(device)         

std = torch.tensor([
    1, 1, 1, 1, 1, 1, 0.2067, 0.8583, 0.3162,
    0.2668, 0.2917, 0.2341, 0.3023, 0.3281, 1.0264, 0.8838, 0.8686, 1.0973,
    1.0267, 0.9018, 0.4658, 0.2009, 0.2057, 1.2240, 0.9535, 0.6655, 0.2941,
    0.3421, 0.8156, 0.6565, 1.1034, 1.5577
], dtype=torch.float32).view(1, -1, 1).to(device) + 1e-8  

class ImuFeatureExtractor(nn.Module):
    def __init__(self, fs=100., add_quaternion=False):
        super().__init__()
        self.fs = fs
        self.add_quaternion = add_quaternion

        k = 15
        self.lpf = nn.Conv1d(6, 6, kernel_size=k, padding=k//2,
                             groups=6, bias=False)
        nn.init.kaiming_uniform_(self.lpf.weight, a=math.sqrt(5))

        self.lpf_acc  = nn.Conv1d(3, 3, k, padding=k//2, groups=3, bias=False)
        self.lpf_gyro = nn.Conv1d(3, 3, k, padding=k//2, groups=3, bias=False)

    def forward(self, imu):
        # imu: 
        B, C, T = imu.shape
        acc  = imu[:, 0:3, :]                 # acc_x, acc_y, acc_z
        gyro = imu[:, 3:6, :]                 # gyro_x, gyro_y, gyro_z
        extra = imu[:, 6:, :]                 

        # 1) magnitude
        acc_mag  = torch.norm(acc,  dim=1, keepdim=True)          # (B,1,T)
        gyro_mag = torch.norm(gyro, dim=1, keepdim=True)

        # 2) jerk 
        jerk = F.pad(acc[:, :, 1:] - acc[:, :, :-1], (1,0))       # (B,3,T)
        gyro_delta = F.pad(gyro[:, :, 1:] - gyro[:, :, :-1], (1,0))

        # 3) energy
        acc_pow  = acc ** 2
        gyro_pow = gyro ** 2

        # 4) LPF / HPF 
        acc_lpf  = self.lpf_acc(acc)
        acc_hpf  = acc - acc_lpf
        gyro_lpf = self.lpf_gyro(gyro)
        gyro_hpf = gyro - gyro_lpf

        features = [
            acc, gyro,
            acc_mag, gyro_mag,
            jerk, gyro_delta,
            acc_pow, gyro_pow,
            acc_lpf, acc_hpf,
            gyro_lpf, gyro_hpf,
        ]
        return torch.cat(features, dim=1)  # (B, C_out, T)


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1)
        return x * y.expand_as(x)

class ResidualSECNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pool_size=2, dropout=0.3, weight_decay=1e-4):
        super().__init__()
        
        # First conv block
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        
        # Second conv block
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size//2, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # SE block
        self.se = SEBlock(out_channels)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm1d(out_channels)
            )
        
        self.pool = nn.MaxPool1d(pool_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        shortcut = self.shortcut(x)
        
        # First conv
        out = F.relu(self.bn1(self.conv1(x)))
        # Second conv
        out = self.bn2(self.conv2(out))
        
        # SE block
        out = self.se(out)
        
        # Add shortcut
        out += shortcut
        out = F.relu(out)
        
        # Pool and dropout
        out = self.pool(out)
        out = self.dropout(out)
        
        return out

class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        # x shape: (batch, seq_len, hidden_dim)
        scores = torch.tanh(self.attention(x))  # (batch, seq_len, 1)
        weights = F.softmax(scores.squeeze(-1), dim=1)  # (batch, seq_len)
        context = torch.sum(x * weights.unsqueeze(-1), dim=1)  # (batch, hidden_dim)
        return context

class TwoBranchModel(nn.Module):
    def __init__(self, pad_len, imu_dim_raw, tof_dim, n_classes, dropouts=[0.3, 0.3, 0.3, 0.3, 0.4, 0.5, 0.3], feature_engineering=True, **kwargs):
        super().__init__()
        self.feature_engineering = feature_engineering
        if feature_engineering:
            self.imu_fe = ImuFeatureExtractor(**kwargs)
            imu_dim = 32            
        else:
            self.imu_fe = nn.Identity()
            imu_dim = imu_dim_raw   
            
        self.imu_dim = imu_dim
        self.tof_dim = tof_dim

        self.fir_nchan = 7

        weight_decay = 3e-3

        numtaps = 33  
        fir_coef = firwin(numtaps, cutoff=1.0, fs=10.0, pass_zero=False)
        fir_kernel = torch.tensor(fir_coef, dtype=torch.float32).view(1, 1, -1)
        fir_kernel = fir_kernel.repeat(7, 1, 1)  # (imu_dim, 1, numtaps)
        self.register_buffer("fir_kernel", fir_kernel)
        
        # IMU deep branch
        self.imu_block1 = ResidualSECNNBlock(imu_dim, 64, 3, dropout=dropouts[0], weight_decay=weight_decay)
        self.imu_block2 = ResidualSECNNBlock(64, 128, 5, dropout=dropouts[1], weight_decay=weight_decay)
        
        # TOF/Thermal lighter branch
        self.tof_conv1 = nn.Conv1d(tof_dim, 64, 3, padding=1, bias=False)
        self.tof_bn1 = nn.BatchNorm1d(64)
        self.tof_pool1 = nn.MaxPool1d(2)
        self.tof_drop1 = nn.Dropout(dropouts[2])
        
        self.tof_conv2 = nn.Conv1d(64, 128, 3, padding=1, bias=False)
        self.tof_bn2 = nn.BatchNorm1d(128)
        self.tof_pool2 = nn.MaxPool1d(2)
        self.tof_drop2 = nn.Dropout(dropouts[3])
        
        # BiLSTM
        self.bilstm = nn.LSTM(256, 128, bidirectional=True, batch_first=True)
        self.lstm_dropout = nn.Dropout(dropouts[4])
        
        # Attention
        self.attention = AttentionLayer(256)  # 128*2 for bidirectional
        
        # Dense layers
        self.dense1 = nn.Linear(256, 256, bias=False)
        self.bn_dense1 = nn.BatchNorm1d(256)
        self.drop1 = nn.Dropout(dropouts[5])
        
        self.dense2 = nn.Linear(256, 128, bias=False)
        self.bn_dense2 = nn.BatchNorm1d(128)
        self.drop2 = nn.Dropout(dropouts[6])
        
        self.classifier = nn.Linear(128, n_classes)
        
    def forward(self, x):
        # Split input
        
        imu = x[:, :, :self.fir_nchan].transpose(1, 2)  # (batch, imu_dim, seq_len)
        tof = x[:, :, self.fir_nchan:].transpose(1, 2)  # (batch, tof_dim, seq_len)

        imu = self.imu_fe(imu)   # (B, imu_dim, T)
        filtered = F.conv1d(
            imu[:, :self.fir_nchan, :],        # (B,7,T)
            self.fir_kernel,
            padding=self.fir_kernel.shape[-1] // 2,
            groups=self.fir_nchan,
        )
        
        imu = torch.cat([filtered, imu[:, self.fir_nchan:, :]], dim=1)  
        imu = (imu - mean) / std 
        # IMU branch
        x1 = self.imu_block1(imu)
        x1 = self.imu_block2(x1)
        
        # TOF branch
        x2 = F.relu(self.tof_bn1(self.tof_conv1(tof)))
        x2 = self.tof_drop1(self.tof_pool1(x2))
        x2 = F.relu(self.tof_bn2(self.tof_conv2(x2)))
        x2 = self.tof_drop2(self.tof_pool2(x2))
        
        # Concatenate branches
        merged = torch.cat([x1, x2], dim=1).transpose(1, 2)  # (batch, seq_len, 256)
        
        # BiLSTM
        lstm_out, _ = self.bilstm(merged)
        lstm_out = self.lstm_dropout(lstm_out)
        
        # Attention
        attended = self.attention(lstm_out)
        
        # Dense layers
        x = F.relu(self.bn_dense1(self.dense1(attended)))
        x = self.drop1(x)
        x = F.relu(self.bn_dense2(self.dense2(x)))
        x = self.drop2(x)
        
        # Classification
        logits = (self.classifier(x))
        return logits


# ================================
# Data Handling
# ================================

def pad_sequences_torch(sequences, maxlen, padding='post', truncating='post', value=0.0):
    """PyTorch equivalent of Keras pad_sequences"""
    result = []
    for seq in sequences:
        if len(seq) >= maxlen:
            if truncating == 'post':
                seq = seq[:maxlen]
            else:  # 'pre'
                seq = seq[-maxlen:]
        else:
            pad_len = maxlen - len(seq)
            if padding == 'post':
                seq = np.concatenate([seq, np.full((pad_len, seq.shape[1]), value)])
            else:  # 'pre'
                seq = np.concatenate([np.full((pad_len, seq.shape[1]), value), seq])
        result.append(seq)
    return np.array(result, dtype=np.float32)

def preprocess_sequence(df_seq: pd.DataFrame, feature_cols: list, scaler: StandardScaler):
    """Normalizes and cleans the time series sequence"""
    mat = df_seq[feature_cols].ffill().bfill().fillna(0).values
    return scaler.transform(mat).astype('float32')

class CMI3Dataset(Dataset):
    def __init__(self,
                 X_list,
                 y_list,
                 maxlen,
                 mode="train",
                 imu_dim=7,
                 augment=None):
        self.X_list = X_list
        self.mode = mode
        self.y_list = y_list
        self.maxlen = maxlen
        self.imu_dim = imu_dim     
        self.augment = augment   

    def pad_sequences_torch(self, seq, maxlen, padding='post', truncating='post', value=0.0):

        if seq.shape[0] >= maxlen:
            if truncating == 'post':
                seq = seq[:maxlen]
            else:  # 'pre'
                seq = seq[-maxlen:]
        else:
            pad_len = maxlen - seq.shape[0]
            if padding == 'post':
                seq = np.concatenate([seq, np.full((pad_len, seq.shape[1]), value)])
            else:  # 'pre'
                seq = np.concatenate([np.full((pad_len, seq.shape[1]), value), seq])
        return seq  
        
    def __getitem__(self, index):
        X = self.X_list[index]
        y = self.y_list[index]

        # ---------- (A)  Augmentation ----------
        if self.mode == "train" and self.augment is not None:
            X = self.augment(X, self.imu_dim)     

        X = self.pad_sequences_torch(X, self.maxlen, 'pre', 'pre')
        return X, y
    
    def __len__(self):
        return len(self.X_list)


class EarlyStopping:
    """Early stopping utility"""
    def __init__(self, patience=7, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False
    
    def save_checkpoint(self, model):
        self.best_weights = model.state_dict().copy()

class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self, model):
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data = self.backup[name]
        self.backup = {}

def set_seed(seed: int = 42):
    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True)

class Augment:
    def __init__(self,
                 p_jitter=0.8, sigma=0.02, scale_range=[0.9,1.1],
                 p_dropout=0.3,
                 p_moda=0.5,          
                 drift_std=0.005,     
                 drift_max=0.25):      
        self.p_jitter  = p_jitter
        self.sigma     = sigma
        self.scale_min, self.scale_max = scale_range
        self.p_dropout = p_dropout
        self.p_moda    = p_moda
        self.drift_std = drift_std
        self.drift_max = drift_max

    # ---------- Jitter & Scaling ----------
    def jitter_scale(self, x: np.ndarray) -> np.ndarray:
        noise  = np.random.randn(*x.shape) * self.sigma
        scale  = np.random.uniform(self.scale_min,
                                   self.scale_max,
                                   size=(1, x.shape[1]))
        return (x + noise) * scale

    # ---------- Sensor Drop-out ----------
    def sensor_dropout(self,
                       x: np.ndarray,
                       imu_dim: int) -> np.ndarray:

        if random.random() < self.p_dropout:
            x[:, imu_dim:] = 0.0
        return x

    def motion_drift(self, x: np.ndarray, imu_dim: int) -> np.ndarray:

        T = x.shape[0]

        drift = np.cumsum(
            np.random.normal(scale=self.drift_std, size=(T, 1)),
            axis=0
        )
        drift = np.clip(drift, -self.drift_max, self.drift_max)   

        x[:, :6] += drift

        if imu_dim > 6:
            x[:, 6:imu_dim] += drift     
        return x
    
    # ---------- master call ----------
    def __call__(self,
                 x: np.ndarray,
                 imu_dim: int) -> np.ndarray:
        if random.random() < self.p_jitter:
            x = self.jitter_scale(x)

        if random.random() < self.p_moda:
            x = self.motion_drift(x, imu_dim)

        x = self.sensor_dropout(x, imu_dim)
        return x
```

```python
# ================================
# Training Pipeline
# ================================
if TRAIN:
    print("▶ TRAIN MODE – loading dataset …")
    df = pd.read_csv(RAW_DIR / "train.csv")

    # Label encoding
    le = LabelEncoder()
    df['gesture_int'] = le.fit_transform(df['gesture'])
    np.save(EXPORT_DIR / "gesture_classes.npy", le.classes_)

    # Feature list
    meta_cols = {'gesture', 'gesture_int', 'sequence_type', 'behavior', 'orientation',
                 'row_id', 'subject', 'phase', 'sequence_id', 'sequence_counter'}
    feature_cols = [c for c in df.columns if c not in meta_cols]

    imu_cols = [c for c in feature_cols if not (c.startswith('thm_') or c.startswith('tof_'))]
    tof_cols = [c for c in feature_cols if c.startswith('thm_') or c.startswith('tof_')]
    print(f"  IMU {len(imu_cols)} | TOF/THM {len(tof_cols)} | total {len(feature_cols)} features")

    # Global scaler
    scaler = StandardScaler().fit(df[feature_cols].ffill().bfill().fillna(0).values)
    joblib.dump(scaler, EXPORT_DIR / "scaler.pkl")

    # Build sequences
    seq_gp = df.groupby('sequence_id')
    X_list, y_list, id_list = [], [], []
    for seq_id, seq in seq_gp:
        mat = preprocess_sequence(seq, feature_cols, scaler)
        X_list.append(mat)
        y_list.append(seq['gesture_int'].iloc[0])
        id_list.append(seq_id)
        # lens.append(len(mat))
    
    pad_len = PAD_PERCENTILE#int(np.percentile(lens, PAD_PERCENTILE))
    print(pad_len)
    np.save(EXPORT_DIR / "sequence_maxlen.npy", pad_len)
    np.save(EXPORT_DIR / "feature_cols.npy", np.array(feature_cols))
    id_list = np.array(id_list)
    X_list_all = pad_sequences_torch(X_list, maxlen=pad_len, padding='pre', truncating='pre')
    y_list_all = np.eye(len(le.classes_))[y_list].astype(np.float32)  # One-hot encoding

    augmenter = Augment(
        p_jitter=0.9844818619033621, sigma=0.03291295776089293, scale_range=(0.7542342630597011,1.1625052821731077),
        p_dropout=0.41782786013520684,
        p_moda=0.3910622476959722, drift_std=0.0040285239353308015, drift_max=0.3929358950258158    
    )
```

```python
EPOCHS = 125
if TRAIN:
    # Split
    skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=random_state)
    models = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(id_list, np.argmax(y_list_all, axis=1))):

        train_list= X_list_all[train_idx]
        train_y_list= y_list_all[train_idx]
        val_list = X_list_all[val_idx]
        val_y_list= y_list_all[val_idx]

        
        # Data loaders
        train_dataset = CMI3Dataset(train_list, train_y_list, maxlen, mode="train", imu_dim=len(imu_cols),
                                augment=augmenter)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4,drop_last=True)
    
        val_dataset = CMI3Dataset(val_list, val_y_list, maxlen, mode="val")
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4,drop_last=True)

    
        # Model
        model = TwoBranchModel(maxlen, len(imu_cols), len(tof_cols), 
                      len(le.classes_)).to(device)
        ema = EMA(model, decay=0.999)
        # Optimizer and scheduler
        optimizer = Adam(model.parameters(), lr=LR_INIT, weight_decay=WD)
        
        steps_per_epoch = len(train_loader)
        nbatch = len(train_loader)
        warmup = epochs_warmup * nbatch
        nsteps = EPOCHS * nbatch
        scheduler = CosineLRScheduler(optimizer,
                          warmup_t=warmup, warmup_lr_init=warmup_lr_init, warmup_prefix=True,
                          t_initial=(nsteps - warmup), lr_min=lr_min) 
    
        early_stopping = EarlyStopping(patience=PATIENCE, restore_best_weights=True)
    
        train_loss = 0.0
        train_acc = 0.0
        val_loss = 0.0
        val_acc = 0.0
        val_best_acc = 0.0
        i_scheduler = 0
        
        # Training loop
        print("▶ Starting training...")
        for epoch in range(EPOCHS):
            model.train()
            train_preds = []
            train_targets = []
            for X, y in (train_loader):  
                X, y = X.float().to(device), y.to(device)
                optimizer.zero_grad()
                logits = model(X)
    
                loss = -torch.sum(F.log_softmax(logits, dim=1) * y, dim=1).mean()
                loss.backward()
                optimizer.step()
                ema.update(model)
                train_preds.extend(logits.argmax(dim=1).cpu().numpy())
                train_targets.extend(y.argmax(dim=1).cpu().numpy())
                scheduler.step(i_scheduler)
                i_scheduler +=1
    
                train_loss += loss.item()
                
            model.eval()
            with torch.inference_mode():
                val_preds = []
                val_targets = []
                for X, y in (val_loader):  
                    half = BATCH_SIZE // 2         

                    x_front = X[:half]               
                    x_back  = X[half:].clone()      
                    
                    x_back[:, :, 7:] = 0.0    
                    X = torch.cat([x_front, x_back], dim=0)  # (B, C, T)
                    X, y = X.float().to(device), y.to(device)
                    
                    logits = model(X)
                    val_preds.extend(logits.argmax(dim=1).cpu().numpy())
                    val_targets.extend(y.argmax(dim=1).cpu().numpy())
                    
                    loss = F.cross_entropy(logits, y)
                    val_loss += loss.item()
    
            train_acc = CompetitionMetric().calculate_hierarchical_f1(
                pd.DataFrame({'gesture': le.classes_[train_targets]}),
                pd.DataFrame({'gesture': le.classes_[train_preds]}))
            val_acc = CompetitionMetric().calculate_hierarchical_f1(
                pd.DataFrame({'gesture': le.classes_[val_targets]}),
                pd.DataFrame({'gesture': le.classes_[val_preds]}))
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
        models.append(model)
        # Save model
        torch.save({
            'model_state_dict': model.state_dict(),
            'imu_dim': len(imu_cols),
            'tof_dim': len(tof_cols),
            'n_classes': len(le.classes_),
            'pad_len': pad_len
        }, EXPORT_DIR / f"gesture_two_branch_fold{fold}.pth")
        print(f"fold: {fold} val_all_acc: {val_acc:.4f}")
        print("✔ Training done – artefacts saved in", EXPORT_DIR)

else:
    print("▶ INFERENCE MODE – loading artefacts from", PRETRAINED_DIR)
    feature_cols = np.load(PRETRAINED_DIR / "feature_cols.npy", allow_pickle=True).tolist()
    pad_len = int(np.load(PRETRAINED_DIR / "sequence_maxlen.npy"))
    scaler = joblib.load(PRETRAINED_DIR / "scaler.pkl")
    gesture_classes = np.load(PRETRAINED_DIR / "gesture_classes.npy", allow_pickle=True)

    imu_cols = [c for c in feature_cols if not (c.startswith('thm_') or c.startswith('tof_'))]
    tof_cols = [c for c in feature_cols if c.startswith('thm_') or c.startswith('tof_')]

    
    # Load model
    MODELS = [f'gesture_two_branch_fold{i}.pth' for i in range(5)]
    
    models = []
    for path in MODELS:
        checkpoint = torch.load(PRETRAINED_DIR / path, map_location=device)
        
        model = TwoBranchModel(
            checkpoint['pad_len'], 
            checkpoint['imu_dim'], 
            checkpoint['tof_dim'], 
            checkpoint['n_classes']
            ).to(device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        models.append(model)

    print("  model, scaler, pads loaded – ready for evaluation")

# Make sure gesture_classes exists in both modes
if TRAIN:
    gesture_classes = le.classes_

def predict(sequence: pl.DataFrame, demographics: pl.DataFrame) -> str:
    """Prediction function for Kaggle competition"""
    global gesture_classes
    if gesture_classes is None:
        gesture_classes = np.load(PRETRAINED_DIR / "gesture_classes.npy", allow_pickle=True)

    df_seq = sequence.to_pandas()
    mat = preprocess_sequence(df_seq, feature_cols, scaler)
    pad = pad_sequences_torch([mat], maxlen=pad_len, padding='pre', truncating='pre')
    
    with torch.no_grad():
        x = torch.FloatTensor(pad).to(device)
        outputs = None
        for model in models:
            model.eval()
            p = torch.softmax(model(x), dim=1)
            if outputs is None: outputs = p
            else: outputs += p
        outputs /= len(models)
        
        idx = int(outputs.argmax(dim=1)[0].cpu().numpy())
    
    return str(gesture_classes[idx])

# Kaggle competition interface
import kaggle_evaluation.cmi_inference_server
inference_server = kaggle_evaluation.cmi_inference_server.CMIInferenceServer(predict)

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    inference_server.run_local_gateway(
        data_paths=(
            '/kaggle/input/cmi-detect-behavior-with-sensor-data/test.csv',
            '/kaggle/input/cmi-detect-behavior-with-sensor-data/test_demographics.csv',
        )
    )
```