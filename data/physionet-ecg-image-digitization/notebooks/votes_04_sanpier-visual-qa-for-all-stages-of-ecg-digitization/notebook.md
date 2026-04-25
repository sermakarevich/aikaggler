# ❤️📈 Visual QA for all Stages of ECG Digitization

- **Author:** Saner Turhaner
- **Votes:** 284
- **Ref:** sanpier/visual-qa-for-all-stages-of-ecg-digitization
- **URL:** https://www.kaggle.com/code/sanpier/visual-qa-for-all-stages-of-ecg-digitization
- **Last run:** 2026-01-13 06:50:34.010000

---

** # ❤️📈 Visual QA for all Stages of ECG Digitization

Huge credits to **Hengck23** and other kagglers whose script + pretrained weights made this script possible. This notebook is built directly on top of that work and uses it as the core reconstruction backbone.

---

### What’s *new / valuable* in this notebook

This notebook focuses on making the public pipeline **transparent, inspectable, and easier to debug & understand**.

### ✅ 1) Clean, safer code structure
- Wrapped everything into a **Pipeline class**, so each stage can be run independently and outputs can be reused without re-running the whole notebook.

### ✅ 2) Stage-by-stage Visual QA
Added utilities to visualize what happens inside the pipeline for any **train ID + source image (0001..0012)**:
- original scan  
- normalized / rotated outputs  
- rectified grid-aligned outputs  
- segmentation predictions  
- reconstructed signals (lead-wise)

This makes it much easier to catch subtle failures like misalignment, bad rectification, or segmentation drift.

### ✅ 3) Source comparison + sanity scoring
Train samples come with multiple image sources (`0001..0012`).  
This notebook includes a “source QA” mode that runs the pipeline for **all available sources**, visualizes final reconstructions, and computes quick quality indicators like:
- average correlation vs GT
- variance / flat-signal detection
- SNR score using the official competition metric
This helps detect “bad source images” and compare stability across sources.

### ✅ 4) Controlled benchmarking of preprocessing improvements
This notebook also includes a reproducible benchmarking framework to evaluate whether preprocessing and source-aware heuristics actually improve ECG reconstruction quality.
It enables:
- fixed, seed-controlled sampling across image sources
- direct comparison between baseline vs preprocessed pipelines
- quantitative evaluation using the official SNR metric and simple signal sanity checks
This helps distinguish preprocessing steps that truly improve signal fidelity from those that only improve visual appearance, and avoids silent regressions through controlled, side-by-side evaluation.

---

**Overall:** this notebook is meant to turn the public pipeline solution into something you can hopefully **understand + debug + validate** visually and also **improve**, rather than treating it as a black-box submission script. Give a thumbs-up if you like it!

# Imports

```python
!pip uninstall -y tensorflow
!uv pip install --no-deps --system --no-index --find-links='/kaggle/input/hengck23-submit-physionet/hengck23-submit-physionet/setup' 'connected-components-3d'
```

```python
# =========================
# Imports & Paths
# =========================
import sys
sys.path.append('/kaggle/input/hengck23-submit-physionet/hengck23-submit-physionet')

import cv2
import gc
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random
import scipy.signal
import scipy.optimize
import time
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import traceback
from pathlib import Path
from shutil import copyfile
from tqdm.auto import tqdm
from typing import Tuple
```

```python
# =========================
# Import pipeline modules
# =========================
import stage0_common as s0c
import stage1_common as s1c
import stage2_common as s2c
import torchvision.transforms as T
from scipy.signal import savgol_filter
from stage0_common import *
from stage0_model import Net as Stage0Net
from stage1_common import *
from stage1_model import Net as Stage1Net
from stage2_model import *
from stage2_common import *
```

```python
# =========================
# Helper functions
# =========================
LEADS_ORDER = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

def change_color(image_rgb):
    """ Enhances contrast and removes noise on ECG scan for model robustness.
    """
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)

    v_denoised = cv2.fastNlMeansDenoising(v, h=5.46)

    std = np.std(v_denoised)
    clip_limit = max(1.0, min(3.5, 2.0 + std / 25))
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    v_enhanced = clahe.apply(v_denoised)

    hsv_enhanced = cv2.merge([h, s, v_enhanced])
    return cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2RGB)

def dw(series_dict, alpha=0.33):
    """ Einthoven correction on SHORT leads only: II_short ≈ I + III
    """
    if all(k in series_dict for k in ['I', 'II_short', 'III']):

        L1 = series_dict['I']
        L2 = series_dict['II_short']
        L3 = series_dict['III']

        # ✅ all same length now
        error = L2 - (L1 + L3)

        series_dict['I']        = L1 + (alpha * error)
        series_dict['III']      = L3 + (alpha * error)
        series_dict['II_short'] = L2 - (alpha * error)

    return series_dict

def series_dict(series_4row):
    """ Converts model output shape (4, length) into dict of 12 leads.
        Each of first 3 rows is split into 4 equal chunks.
    """
    series_4row = np.asarray(series_4row)

    # ---- enforce correct shape ----
    if series_4row.ndim == 3:
        series_4row = series_4row[0]  # remove batch dim
    if series_4row.shape[0] != 4 and series_4row.shape[1] == 4:
        series_4row = series_4row.T  # auto-fix transpose

    if series_4row.shape[0] != 4:
        raise ValueError(f"[series_dict] Expected (4, L), got {series_4row.shape}")

    L = series_4row.shape[1]
    if L < 1000:
        print(f"[WARN] suspicious length: {L}")

    d = {}

    names = [
        ['I',   'aVR', 'V1', 'V4'],
        ['II',  'aVL', 'V2', 'V5'],
        ['III', 'aVF', 'V3', 'V6'],
    ]

    for row_idx in range(3):
        splits = np.array_split(series_4row[row_idx], 4)
        for lead, arr in zip(names[row_idx], splits):
            d[lead] = np.asarray(arr, dtype=np.float32)

    d['II'] = np.asarray(series_4row[3], dtype=np.float32)
    return d

def dict_to_df_pred(d_series, df_signal, fs, sig_len):
    pred = {lead: np.full(sig_len, np.nan, dtype=np.float32) for lead in LEADS_ORDER}

    expected_short = int(round(fs * 2.5))

    # --- Lead II is full length ---
    y_ii = np.asarray(d_series.get("II", np.zeros(sig_len)), dtype=np.float32)
    if len(y_ii) != sig_len:
        y_ii = np.interp(np.linspace(0, 1, sig_len), np.linspace(0, 1, len(y_ii)), y_ii)
    pred["II"] = y_ii

    # --- all short leads: PLACE INTO GT SPAN ---
    for lead in LEADS_ORDER:
        if lead == "II":
            continue

        y = d_series.get(lead, None)
        if y is None:
            continue
        y = np.asarray(y, dtype=np.float32)

        # enforce 2.5s length
        if len(y) != expected_short:
            y = np.interp(np.linspace(0, 1, expected_short), np.linspace(0, 1, len(y)), y)

        # locate GT span (where label exists)
        mask = df_signal[lead].notna().values
        idx = np.flatnonzero(mask)
        if idx.size == 0:
            continue

        start, end = idx[0], idx[-1] + 1
        n_seg = end - start

        y_seg = y[:n_seg]
        if len(y_seg) < n_seg:
            y_seg = np.pad(y_seg, (0, n_seg - len(y_seg)), mode="edge")

        pred[lead][start:end] = y_seg

    return pd.DataFrame(pred, index=df_signal.index, columns=LEADS_ORDER)
```

# Helper Functions

# Model Pipeline

```python
# =========================
# Stage2 Net definition (Net3)
# =========================
class Net3(nn.Module):
    def __init__(self, pretrained=True):
        super(Net3, self).__init__()
        encoder_dim = [64, 128, 256, 512]
        decoder_dim = [128, 64, 32, 16]

        self.encoder = timm.create_model(
            model_name='resnet34.a3_in1k',
            pretrained=pretrained,
            in_chans=3,
            num_classes=0,
            global_pool=''
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
        
# =========================
# Pipeline Wrapper
# =========================
class PhysioPipeline:
    def __init__(self, device="cuda:0"):
        self.device = device

        self.stage0_net = None
        self.stage1_net = None
        self.stage2_net = None

        # stage2 constants
        self.x0, self.x1 = 0, 2176
        self.y0, self.y1 = 0, 1696
        self.zero_mv = [703.5, 987.5, 1271.5, 1531.5]
        self.mv_to_pixel = 78.8
        self.t0, self.t1 = 235, 4161

        self.resize = T.Resize((1696, 4352), interpolation=T.InterpolationMode.BILINEAR)

    def load_models(self, stage0_w, stage1_w, stage2_w):
        # ---- stage0 ----
        self.stage0_net = Stage0Net(pretrained=False)
        self.stage0_net = s0c.load_net(self.stage0_net, stage0_w)
        self.stage0_net.to(self.device)
        self.stage0_net.eval()

        # ---- stage1 ----
        self.stage1_net = Stage1Net(pretrained=False)
        self.stage1_net = s1c.load_net(self.stage1_net, stage1_w)
        self.stage1_net.to(self.device)
        self.stage1_net.eval()

        # ---- stage2 ----
        self.stage2_net = Net3(pretrained=False).to(self.device)
        self.stage2_net.load_state_dict(torch.load(stage2_w))
        self.stage2_net.eval()

    def run_stage0(self, img_bgr):
        """ Stage0: normalize rotation + homography"""
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_for_model = change_color(img_rgb)
        batch = s0c.image_to_batch(img_for_model)

        with torch.no_grad(), torch.amp.autocast(self.device.split(":")[0], dtype=torch.float32):
            output = self.stage0_net(batch)

        # ✅ must use stage0_common.output_to_predict
        rotated, keypoint = s0c.output_to_predict(img_rgb, batch, output)
        normalised, _, _ = s0c.normalise_by_homography(rotated, keypoint)

        return normalised

    def run_stage1(self, stage0_img_rgb):
        """ Stage1: rectify grid (perspective alignment)"""
        image = stage0_img_rgb
        batch = {'image': torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).unsqueeze(0)}

        with torch.no_grad(), torch.amp.autocast(self.device.split(":")[0], dtype=torch.float32):
            output = self.stage1_net(batch)

        # ✅ must use stage1_common.output_to_predict
        gridpoint_xy, _ = s1c.output_to_predict(image, batch, output)
        rectified = s1c.rectify_image(image, gridpoint_xy)

        return rectified

    def run_stage2(self, stage1_img_rgb, length):
        """ Stage2: segmentation → pixel_to_series"""
        img = stage1_img_rgb
        img = img[self.y0:self.y1, self.x0:self.x1] / 255.0

        batch = self.resize(torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))).unsqueeze(0)).float().to(self.device)

        with torch.no_grad(), torch.amp.autocast(self.device.split(":")[0], dtype=torch.float32):
            output = self.stage2_net(batch)

        pixel = torch.sigmoid(output).float().data.cpu().numpy()[0]  # (4,H,W)

        series_in_pixel = s2c.pixel_to_series(pixel[..., self.t0:self.t1], self.zero_mv, length)
        series = (np.array(self.zero_mv).reshape(4, 1) - series_in_pixel) / self.mv_to_pixel

        # smooth
        for i in range(4):
            series[i] = savgol_filter(series[i], window_length=7, polyorder=2)

        return pixel, series
    
    def run_full(self, img_bgr, df_signal, fs, sig_len):
        """ Full pipeline:
                stage0 -> stage1 -> stage2 -> series_dict -> df_pred
        
            Returns:
                df_pred: (sig_len, 12) prediction dataframe
                d_series: dict of 12 leads
                pixel: stage2 segmentation map (for debug)
        """
    
        # ---- stage0 ----
        stage0_img_rgb = self.run_stage0(img_bgr)
    
        # ---- stage1 ----
        stage1_img_rgb = self.run_stage1(stage0_img_rgb)
    
        # ---- stage2 ----
        length = sig_len   # Lead II length is sig_len = fs*10
        out = self.run_stage2(stage1_img_rgb, length)
    
        # ✅ support both old and new stage2 behavior
        if isinstance(out, tuple) and len(out) == 2:
            pixel, series_4row = out
        else:
            pixel = None
            series_4row = out
    
        # ---- convert 4-row -> 12-lead dict ----
        d_series = series_dict(series_4row)   # your function
    
        # ---- convert dict -> df_pred (sig_len x 12) ----
        df_pred = dict_to_df_pred(d_series, df_signal, fs, sig_len)
        return df_pred, d_series, pixel
```

```python
# =========================
# Predict one ECG image end-to-end
# =========================
def predict_one(sample_id, image_path, df_meta, pipeline: PhysioPipeline):
    img_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(image_path)

    # length = number_of_rows for lead II
    length = df_meta[(df_meta['id'] == sample_id) & (df_meta['lead'] == 'II')].iloc[0].number_of_rows

    s0 = pipeline.run_stage0(img_bgr)
    s1 = pipeline.run_stage1(s0)
    pixel, series = pipeline.run_stage2(s1, length)

    return s0, s1, pixel, series

# =========================
# Build submission
# =========================
def build_submission(df_test, test_dir, pipeline: PhysioPipeline):
    res = []
    gb = df_test.groupby("id")

    for i, (sample_id, df) in enumerate(tqdm(gb)):
        sample_id = str(sample_id)
        path = Path(test_dir) / f"{sample_id}.png"

        try:
            _, _, _, series = predict_one(sample_id, path, df_test, pipeline)
        except Exception as e:
            traceback.print_exc()
            length = df[df.lead == "II"].iloc[0].number_of_rows
            series = np.zeros((4, length))

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
    return submission
```

# Debug & Visualize & Evaluate

```python
# =========================
# Visualization Helpers
# =========================
def plot_all_leads_gt_vs_pred(df_signal, df_pred, fs, title="", max_points=None, show_pred_full=False):
    """ Fancy 4x3 grid plot for all leads.
        GT vs Pred on same subplot.
        Optional: show full pred signal (including NaNs) in the background.
    """
    ncols, nrows = 3, 4
    fig, axes = plt.subplots(nrows, ncols, figsize=(22, 14), sharex=False)
    axes = axes.flatten()

    t = np.arange(len(df_signal)) / fs

    for i, lead in enumerate(LEADS_ORDER):
        ax = axes[i]

        y_true = df_signal[lead].values.astype(np.float32)
        y_pred = df_pred[lead].values.astype(np.float32)

        # --- GT valid region ---
        mask_gt = np.isfinite(y_true)

        tt = t[mask_gt]
        yt = y_true[mask_gt]
        yp = y_pred[mask_gt]   # may contain NaNs if placement failed

        # downsample if requested
        if max_points is not None and len(tt) > max_points:
            step = max(1, len(tt) // max_points)
            tt = tt[::step]
            yt = yt[::step]
            yp = yp[::step]

        # --- plot GT ---
        ax.plot(tt, yt, lw=1.0, label="GT", alpha=0.9)

        # --- plot Pred (GT region only) ---
        ax.plot(tt, yp, lw=1.0, label="Pred", alpha=0.85)

        # --- optional full pred ---
        if show_pred_full:
            mask_pred = np.isfinite(y_pred)
            ax.plot(t[mask_pred], y_pred[mask_pred], lw=0.8, alpha=0.25, label="Pred full")

        # --- diagnostics: how much pred is NaN inside GT region? ---
        nan_in_gt = np.mean(~np.isfinite(y_pred[mask_gt])) if np.any(mask_gt) else 0.0
        if nan_in_gt > 0:
            ax.text(
                0.02, 0.95,
                f"⚠ NaN in GT: {nan_in_gt:.1%}",
                transform=ax.transAxes,
                fontsize=9,
                verticalalignment="top",
                color="red"
            )

        ax.set_title(lead, fontsize=11)
        ax.grid(alpha=0.25)

        if i == 0:
            ax.legend(fontsize=10)

    fig.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
```

```python
# ============================================================
# Core metric functions (same logic as competition notebook)
# ============================================================

# ---- Competition constants ----
MAX_TIME_SHIFT = 0.2
PERFECT_SCORE = 384
LEADS_ORDER = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

# ---- Competition-style exception ----
class ParticipantVisibleError(Exception):
    pass

def compute_power(label: np.ndarray, prediction: np.ndarray) -> Tuple[float, float]:
    if label.ndim != 1 or prediction.ndim != 1:
        raise ParticipantVisibleError("Inputs must be 1-dimensional arrays.")

    finite_mask = np.isfinite(prediction)
    if not np.any(finite_mask):
        raise ParticipantVisibleError("The 'prediction' array contains no finite values (all NaN or inf).")

    prediction = prediction.copy()
    prediction[~np.isfinite(prediction)] = 0

    noise = label - prediction
    p_signal = np.sum(label**2)
    p_noise = np.sum(noise**2)
    return p_signal, p_noise

def compute_snr(signal: float, noise: float) -> float:
    if noise == 0:
        return PERFECT_SCORE
    elif signal == 0:
        return 0
    else:
        return min(signal / noise, PERFECT_SCORE)

def align_signals(label: np.ndarray, pred: np.ndarray, max_shift: int) -> np.ndarray:
    """ Align pred to label via cross-correlation + vertical shift correction.
        Matches official evaluation.
    """
    if np.any(~np.isfinite(label)):
        raise ParticipantVisibleError("values in label should all be finite")
    if np.sum(np.isfinite(pred)) == 0:
        raise ParticipantVisibleError("prediction cannot all be infinite")

    label_arr = np.asarray(label, dtype=np.float64)
    pred_arr = np.asarray(pred, dtype=np.float64)

    label_mean = np.mean(label_arr)
    pred_mean = np.mean(pred_arr)

    label_centered = label_arr - label_mean
    pred_centered = pred_arr - pred_mean

    correlation = scipy.signal.correlate(label_centered, pred_centered, mode="full")

    n_label = len(label_arr)
    n_pred = len(pred_arr)
    lags = scipy.signal.correlation_lags(n_label, n_pred, mode="full")

    valid = (lags >= -max_shift) & (lags <= max_shift)
    max_corr = np.nanmax(correlation[valid])

    all_max_idx = np.flatnonzero(correlation == max_corr)
    best_idx = min(all_max_idx, key=lambda i: abs(lags[i]))

    time_shift = lags[best_idx]

    start_padding_len = max(time_shift, 0)
    pred_slice_start = max(-time_shift, 0)
    pred_slice_end = min(n_label - time_shift, n_pred)
    end_padding_len = max(n_label - n_pred - time_shift, 0)

    aligned_pred = np.concatenate((
        np.full(start_padding_len, np.nan),
        pred_arr[pred_slice_start:pred_slice_end],
        np.full(end_padding_len, np.nan)
    ))

    # vertical alignment
    def objective(v_shift):
        return np.nansum((label_arr - (aligned_pred - v_shift)) ** 2)

    if np.any(np.isfinite(label_arr) & np.isfinite(aligned_pred)):
        res = scipy.optimize.minimize_scalar(objective, method="Brent")
        aligned_pred -= res.x

    return aligned_pred

def _calculate_image_score(group: pd.DataFrame) -> float:
    unique_fs = group["fs"].unique()
    if len(unique_fs) != 1:
        raise ParticipantVisibleError("Sampling frequency should be consistent across each ECG.")
    fs = int(unique_fs[0])

    ii = group[group["lead"] == "II"]
    if ii.empty:
        raise ParticipantVisibleError("Missing lead II in this image group.")

    expected_len = fs * 10
    if len(ii) != expected_len:
        raise ParticipantVisibleError(f"Lead II length {len(ii)} != expected {expected_len} (fs={fs})")

    sum_signal = 0.0
    sum_noise = 0.0

    for lead in LEADS_ORDER:
        sub = group[group["lead"] == lead]
        if sub.empty:
            raise ParticipantVisibleError(f"Missing lead {lead} in this image group.")

        label = sub["value_true"].to_numpy()
        pred  = sub["value_pred"].to_numpy()

        aligned_pred = align_signals(label, pred, int(fs * MAX_TIME_SHIFT))
        p_signal, p_noise = compute_power(label, aligned_pred)

        sum_signal += float(p_signal)
        sum_noise  += float(p_noise)

    return float(compute_snr(sum_signal, sum_noise))

def snr_score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str = "id") -> float:
    """ Returns the public metric score.
        Exactly follows the official evaluation.
    """
    for df in [solution, submission]:
        if row_id_column_name not in df.columns:
            raise ParticipantVisibleError(f"'{row_id_column_name}' column not found.")
        if df["value"].isna().any():
            raise ParticipantVisibleError("NaN exists in solution/submission")
        if not np.isfinite(df["value"]).all():
            raise ParticipantVisibleError("Infinity exists in solution/submission")

    submission = submission[[row_id_column_name, "value"]]
    merged = pd.merge(solution, submission, on=row_id_column_name, suffixes=("_true", "_pred"))

    merged["image_id"] = merged[row_id_column_name].str.split("_").str[0]
    merged["row_id"] = merged[row_id_column_name].str.split("_").str[1].astype("int64")
    merged["lead"] = merged[row_id_column_name].str.split("_").str[2]

    merged.sort_values(by=["image_id", "row_id", "lead"], inplace=True)

    image_scores = merged.groupby("image_id").apply(_calculate_image_score, include_groups=False)

    return max(float(10 * np.log10(image_scores.mean())), -PERFECT_SCORE)

# ============================================================
# Build solution/submission DF from train pred/gt
# ============================================================
def make_eval_dfs_from_container(base_id: str, fs: int, df_signal: pd.DataFrame, df_pred: pd.DataFrame):
    """ Converts df_signal + df_pred into the exact (solution, submission) format expected by snr_score.
    """
    sol_rows = []
    sub_rows = []

    def extract_short_segment(arr_true, arr_pred):
        mask = np.isfinite(arr_true)
        idx = np.flatnonzero(mask)
        if idx.size == 0:
            return None, None

        start, end = idx[0], idx[-1] + 1
        seg_true = arr_true[start:end]
        seg_pred = arr_pred[start:end]

        n_expected = int(np.floor(fs * 2.5))
        if len(seg_true) > n_expected:
            seg_true = seg_true[:n_expected]
            seg_pred = seg_pred[:n_expected]
        elif len(seg_true) < n_expected:
            pad = n_expected - len(seg_true)
            seg_true = np.concatenate([seg_true, np.full(pad, seg_true[-1])])
            seg_pred = np.concatenate([seg_pred, np.full(pad, seg_pred[-1])])

        return seg_true, seg_pred

    for lead in LEADS_ORDER:
        y_true = df_signal[lead].values.astype(np.float64)
        y_pred = df_pred[lead].values.astype(np.float64)

        if lead == "II":
            seg_true, seg_pred = y_true, y_pred
        else:
            seg_true, seg_pred = extract_short_segment(y_true, y_pred)
            if seg_true is None:
                continue

        for row_idx, (vt, vp) in enumerate(zip(seg_true, seg_pred)):
            row_id = f"{base_id}_{row_idx}_{lead}"
            sol_rows.append({"id": row_id, "fs": fs, "value": float(vt)})
            sub_rows.append({"id": row_id, "value": float(vp)})

    solution = pd.DataFrame(sol_rows)
    submission = pd.DataFrame(sub_rows)
    return solution, submission
```

```python
# =========================
# Visual Debug: ALL stages for ONE train id across ALL sources
# =========================
TRAIN_DIR = Path("/kaggle/input/physionet-ecg-image-digitization/train")

def get_available_sources_for_train_id(train_id):
    """Return all available source suffixes like '0001', '0003', ... for this train id."""
    train_id = str(train_id)
    folder = TRAIN_DIR / train_id
    if not folder.exists():
        return []

    sources = []
    for p in folder.glob(f"{train_id}-*.png"):
        suffix = p.stem.split("-")[-1]
        sources.append(suffix)

    return sorted(list(set(sources)))

def compute_debug_stats(df_signal, df_pred):
    """Simple stats to catch degenerate outputs."""
    stats = {}

    y_pred = df_pred.values
    mask = np.isfinite(df_signal.values)
    stats["nan_rate"] = float(np.isnan(df_pred.values[mask]).mean())

    # lead-wise std (avoid NaNs)
    lead_stds = []
    lead_absmeans = []
    lead_corrs = []

    for lead in LEADS_ORDER:
        yt = df_signal[lead].values
        yp = df_pred[lead].values

        mask = np.isfinite(yt) & np.isfinite(yp)
        if mask.sum() < 10:
            lead_corrs.append(np.nan)
            continue

        lead_stds.append(np.std(yp[mask]))
        lead_absmeans.append(np.mean(np.abs(yp[mask])))

        # correlation
        c = np.corrcoef(yt[mask], yp[mask])[0, 1]
        lead_corrs.append(c)

    stats["pred_std_mean"] = float(np.nanmean(lead_stds)) if len(lead_stds) else np.nan
    stats["pred_abs_mean"] = float(np.nanmean(lead_absmeans)) if len(lead_absmeans) else np.nan
    stats["lead_corr_mean"] = float(np.nanmean(lead_corrs)) if len(lead_corrs) else np.nan

    return stats

def normalize_heatmap(hm, p_low=1, p_high=99):
    """Normalize heatmap using percentiles for visibility."""
    lo, hi = np.percentile(hm, [p_low, p_high])
    hm = np.clip(hm, lo, hi)
    return (hm - lo) / (hi - lo + 1e-9)
    
def debug_visualize_all_stages_for_train_id(
    df_train,
    pipeline,
    train_id,
    max_sources=None,              # optionally limit number of sources
    random_sources=False,          # optionally randomize selection
    show_pixel_mode="heatmap",     # "heatmap" or "overlay"
    pixel_channel=0,               # which channel to plot (0..3)
    plot_signal_per_source=False,  # if True: plot 12-lead GT vs Pred per source
    max_points_signal=4000,        # downsample signal plot
):
    """ Runs Stage0 -> Stage1 -> Stage2 for ONE train_id across ALL sources (0001..0012)
        and visualizes everything in a stage-gallery.
    
        Returns:
          df_debug: stats per source
          outputs: dict {source: {"df_pred":..., "pixel":..., "d_series":...}}
    """
    train_id = str(train_id)

    # --- metadata ---
    row_meta = df_train[df_train["id"] == int(train_id)].iloc[0]
    fs = int(row_meta.fs)
    sig_len = int(row_meta.sig_len)

    # --- load GT CSV ---
    df_signal = pd.read_csv(TRAIN_DIR / train_id / f"{train_id}.csv")

    # --- all sources ---
    sources = get_available_sources_for_train_id(train_id)
    if len(sources) == 0:
        raise ValueError(f"No sources found for train_id={train_id}")

    # optionally reduce sources
    if max_sources is not None and len(sources) > max_sources:
        if random_sources:
            sources = list(np.random.choice(sources, size=max_sources, replace=False))
        else:
            sources = sources[:max_sources]

    outputs = {}
    debug_rows = []

    # =========================================================
    # ✅ Run all sources first (so we can plot cleanly later)
    # =========================================================
    stage_imgs = []  # store images for gallery

    for src in sources:
        img_path = TRAIN_DIR / train_id / f"{train_id}-{src}.png"
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print(f"[WARN] Missing image: {img_path}")
            continue

        # --- Stage0 ---
        s0_rgb = pipeline.run_stage0(img_bgr)

        # --- Stage1 ---
        s1_rgb = pipeline.run_stage1(s0_rgb)

        # --- Stage2 ---
        pixel, series_4row = pipeline.run_stage2(s1_rgb, length=sig_len)

        # --- convert series -> 12 lead dict ---
        d_series = series_dict(series_4row)
        d_series = dw(d_series)

        # --- build df_pred ---
        df_pred = dict_to_df_pred(d_series, df_signal, fs, sig_len)
        solution, submission = make_eval_dfs_from_container(train_id, fs, df_signal, df_pred)
        snr = snr_score(solution, submission, row_id_column_name="id")
        
        # --- compute quick stats ---
        stats = compute_debug_stats(df_signal, df_pred)

        debug_rows.append({
            "id": train_id,
            "source": src,
            "fs": fs,
            "sig_len": sig_len,
            "snr_score": snr,
            **stats,
        })

        # store everything
        outputs[src] = {
            "df_pred": df_pred,
            "pixel": pixel,
            "d_series": d_series,
            "s0_rgb": s0_rgb,
            "s1_rgb": s1_rgb,
            "img_rgb": cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB),
        }

        # prep stage image for gallery
        px = np.asarray(pixel)
        if px.ndim == 4 and px.shape[0] == 1:
            px = px[0]  # remove batch dim

        # handle pixel map
        px_vis = None
        if px.ndim == 3 and px.shape[0] >= pixel_channel + 1:
            px_ch = px[pixel_channel]
            if show_pixel_mode == "heatmap":
                px_vis = px_ch
            else:
                # overlay mode: blend on rectified image
                img_overlay = (s1_rgb.copy() / 255.0).astype(np.float32)
                heat = (px_ch - px_ch.min()) / (px_ch.max() - px_ch.min() + 1e-9)
                heat = np.stack([heat, heat, heat], axis=-1)
                px_vis = np.clip(0.65 * img_overlay + 0.35 * heat, 0, 1)

        stage_imgs.append((src, outputs[src]["img_rgb"], s0_rgb, s1_rgb, px_vis))

    # =========================================================
    # ✅ Plot stage gallery
    # =========================================================
    n = len(stage_imgs)
    if n == 0:
        raise ValueError("No valid sources could be processed!")

    fig, axes = plt.subplots(nrows=n, ncols=4, figsize=(20, 4*n))
    if n == 1:
        axes = np.expand_dims(axes, 0)

    for r, (src, img_rgb, s0_rgb, s1_rgb, px_vis) in enumerate(stage_imgs):
        axes[r, 0].imshow(img_rgb)
        axes[r, 0].set_title(f"{src} | Original")
        axes[r, 0].axis("off")

        axes[r, 1].imshow(s0_rgb)
        axes[r, 1].set_title("Stage0")
        axes[r, 1].axis("off")

        axes[r, 2].imshow(s1_rgb)
        axes[r, 2].set_title("Stage1")
        axes[r, 2].axis("off")

        if px_vis is None:
            axes[r, 3].text(0.1, 0.5, "Pixel output missing", fontsize=12)
            axes[r, 3].axis("off")
        else:
            if show_pixel_mode == "heatmap":
                hm = px_vis.astype(np.float32)
                hm = normalize_heatmap(hm, 1, 99)
            
                im = axes[r, 3].imshow(
                    hm,
                    cmap="magma",
                    vmin=0,
                    vmax=1
                )
                axes[r, 3].set_title(f"Stage2 | Heatmap ch={pixel_channel} (p1–p99 norm)")
            
                # add a small colorbar
                cbar = fig.colorbar(im, ax=axes[r, 3], fraction=0.046, pad=0.04)
                cbar.ax.tick_params(labelsize=8)
            else:
                axes[r, 3].imshow(px_vis)
                axes[r, 3].set_title(f"Stage2 | Overlay ch={pixel_channel}")
            axes[r, 3].axis("off")

    plt.suptitle(f"Train ID {train_id} | Stage Gallery across sources", fontsize=18)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()

    # =========================================================
    # ✅ Optional: plot signals per source
    # =========================================================
    if plot_signal_per_source:
        for src in outputs:
            df_pred = outputs[src]["df_pred"]
            stats = next((r for r in debug_rows if r["source"] == src), None)
            title = (
                f"Train ID {train_id} | Source {src} | "
                f"SNR={snr:.2f} | corr={stats['lead_corr_mean']:.3f} | std={stats['pred_std_mean']:.3f}"
            )
            plot_all_leads_gt_vs_pred(
                df_signal, df_pred, fs,
                title=title,
                max_points=max_points_signal
            )

    df_debug = pd.DataFrame(debug_rows).sort_values("source").reset_index(drop=True)
    return df_debug, outputs
```

```python
# load training dataset & set pipeline
df_train = pd.read_csv("/kaggle/input/physionet-ecg-image-digitization/train.csv")

pipeline = PhysioPipeline(device="cuda:0")
pipeline.load_models(
    stage0_w="/kaggle/input/hengck23-submit-physionet/hengck23-submit-physionet/weight/stage0-last.checkpoint.pth",
    stage1_w="/kaggle/input/hengck23-submit-physionet/hengck23-submit-physionet/weight/stage1-last.checkpoint.pth",
    stage2_w="/kaggle/input/physio-seg-public/pytorch/net3_009_4200/1/iter_0004200.pt",
)
```

```python
# run stage debugging & performance check
train_id = 1006427285
df_debug, outputs = debug_visualize_all_stages_for_train_id(
    df_train, pipeline, train_id,
    plot_signal_per_source=True,
)
df_debug
```

# How to Read `df_debug` 🧐 

The `df_debug` dataframe is meant to be a **quick sanity + quality report** for your debug runs.
Ideally it contains **one row per `(train_id, source)`**, where *source* is the image suffix like `"0001"`, `"0003"`, … `"0012"`.
This makes it easy to compare how the same ECG behaves across different image renderings/sources.

---

## 📊 **Quality Stats**

| Column | Meaning | Expectation |
|-------|---------|-------------|
| `nan_rate` | Fraction of NaNs in predicted series | should always be `0.0` |
| `pred_std_mean` | Mean std dev across all predicted leads | should be **> 0**, otherwise model output is too flat |
| `pred_abs_mean` | Mean absolute amplitude across leads | should be **not too small**, otherwise signal is collapsed |
| `lead_corr_mean` | Mean Pearson correlation (Pred vs GT) across leads *(train only)* | higher = better |
| `snr_score` | SNR score against GT *(train only, optional)* | higher = better |

---

## ✅ Good Values

- `nan_rate == 0.0`
- `pred_std_mean` roughly `0.05 – 0.5` *(very rough range)*
- `lead_corr_mean > 0.6` → decent  
- `lead_corr_mean > 0.8` → very good  
- `snr_score` → depends on scoring scale, but **higher is always better**


---

## 🚨 Bad Values

- `nan_rate > 0` → prediction has NaNs, submission would fail
- `pred_std_mean < 0.01` → model basically produced flat lines
- `lead_corr_mean < 0.2` → pipeline output is very misaligned / broken
- `snr_score extremely negative` → something fundamental failed (wrong scale, wrong mapping, wrong ordering, etc.)

# Benchmarking on Sample

```python
def make_benchmark_manifest(sources, per_source=3):
    """ Build balanced manifest by scanning TRAIN_DIR on disk:
          - for each source suffix, find all matching PNGs
          - sample `per_source`
          - extract train_id from filename (no id formatting assumptions)
        Returns DataFrame: [id, source]
    """
    rng = np.random.default_rng(42)
    rows = []

    for src in sources:
        src = str(src)

        # find all images with this source suffix across all train folders
        # pattern: train/<id>/<id>-<src>.png
        paths = sorted(TRAIN_DIR.glob(f"*/ *-{src}.png".replace(" ", "")))
        if len(paths) < per_source:
            raise ValueError(f"Not enough files for source={src}: need {per_source}, found {len(paths)}")

        picked = rng.choice(paths, size=per_source, replace=False)

        for p in picked:
            # filename: "<id>-<src>.png"
            stem = p.stem
            train_id = stem.split("-")[0]  # keep exactly as in filesystem
            rows.append({"id": train_id, "source": src})

    manifest = pd.DataFrame(rows).sort_values(["source", "id"]).reset_index(drop=True)
    return manifest

def _get_row_meta(df_train, train_id):
    """
    Robust lookup for df_train row by id:
    handles id being str or int in df_train.
    """
    # try int match
    try:
        tid_int = int(str(train_id))
    except:
        tid_int = None

    if tid_int is not None and "id" in df_train.columns:
        m = df_train[df_train["id"].astype(str) == str(tid_int)]
        if len(m):
            return m.iloc[0]

    # fallback: raw string match
    m = df_train[df_train["id"].astype(str) == str(train_id)]
    if len(m):
        return m.iloc[0]

    raise KeyError(f"train_id={train_id} not found in df_train['id']")

def stage1_quality(s1_rgb):
    g = cv2.cvtColor(s1_rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY)

    # edges
    e = cv2.Canny(g, 50, 150)

    # edge density
    density = e.mean() / 255.0

    # anisotropy: prefer structured (grid-like) edges over random texture
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    ax = float(np.mean(np.abs(gx)))
    ay = float(np.mean(np.abs(gy)))
    anisotropy = max(ax, ay) / (min(ax, ay) + 1e-6)  # >=1

    # combine (weights are mild)
    return float(density * 0.7 + np.tanh(anisotropy - 1.0) * 0.3)
    
def debug_compute_one_source(
    df_train,
    pipeline,
    train_id,
    source,
    use_source_preprocess=True,    # ✅ new flag
    keep_artifacts=False,          # optional memory saver
):
    t0 = time.time()
    train_id = str(train_id)
    source = str(source)

    # metadata
    row_meta = _get_row_meta(df_train, train_id)
    fs = int(row_meta.fs)
    sig_len = int(row_meta.sig_len)

    # GT
    df_signal = pd.read_csv(TRAIN_DIR / train_id / f"{train_id}.csv")

    # image
    img_path = TRAIN_DIR / train_id / f"{train_id}-{source}.png"
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        raise ValueError(f"Missing/unreadable image: {img_path}")

    # ✅ source-aware preprocessing (train benchmarking)
    img_raw = img_bgr.copy()
    if use_source_preprocess:
        img_pp = preprocess_by_source(img_bgr, source)

    # Stage0/1/2
    # ---- Stage0/Stage1: choose better input ----
    if use_source_preprocess:
        s0_raw = pipeline.run_stage0(img_raw)
        s1_raw = pipeline.run_stage1(s0_raw)
        q_raw = stage1_quality(s1_raw)

        s0_pp = pipeline.run_stage0(img_pp)
        s1_pp = pipeline.run_stage1(s0_pp)
        q_pp = stage1_quality(s1_pp)

        # pick the better one (with a small margin to avoid flip-flopping)
        if q_pp > q_raw * 1.02:
            s0_rgb, s1_rgb = s0_pp, s1_pp
            chosen = "preprocessed"
        else:
            s0_rgb, s1_rgb = s0_raw, s1_raw
            chosen = "raw"
    else:
        s0_rgb = pipeline.run_stage0(img_raw)
        s1_rgb = pipeline.run_stage1(s0_rgb)
        chosen = "raw"
    # ---- continue Stage2 using selected s1_rgb ----
    pixel, series_4row = pipeline.run_stage2(s1_rgb, length=sig_len)

    d_series = series_dict(series_4row)
    d_series = dw(d_series)

    df_pred = dict_to_df_pred(d_series, df_signal, fs, sig_len)

    solution, submission = make_eval_dfs_from_container(train_id, fs, df_signal, df_pred)
    snr = snr_score(solution, submission, row_id_column_name="id")

    stats = compute_debug_stats(df_signal, df_pred)

    row = {
        "id": train_id,
        "source": source,
        "fs": fs,
        "sig_len": sig_len,
        "ok": True,
        "snr_score": float(snr),
        "chosen_input": chosen,
        **stats,
        "runtime_sec": float(time.time() - t0),
        "fail_reason": None,
        "use_source_preprocess": bool(use_source_preprocess),
    }

    if keep_artifacts:
        artifacts = {
            "df_pred": df_pred,
            "pixel": pixel,
            "s0_rgb": s0_rgb,
            "s1_rgb": s1_rgb,
            "img_rgb": cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB),
        }
    else:
        artifacts = {}

    return row, artifacts

def run_benchmark(df_train, pipeline, bench_manifest, verbose=True, **debug_kwargs):
    rows = []
    it = tqdm(bench_manifest.itertuples(index=False), total=len(bench_manifest), desc="benchmark") if verbose else bench_manifest.itertuples(index=False)

    for r in it:
        train_id = str(r.id)
        source = str(r.source)
        t0 = time.time()

        try:
            row, _ = debug_compute_one_source(
                df_train, pipeline,
                train_id=train_id,
                source=source,
                **debug_kwargs
            )
            # overwrite runtime to include any overhead
            row["runtime_sec"] = float(time.time() - t0)
        except Exception as e:
            row = {
                "id": train_id,
                "source": source,
                "ok": False,
                "snr_score": np.nan,
                "runtime_sec": float(time.time() - t0),
                "fail_reason": f"{type(e).__name__}: {str(e)[:250]}",
            }
        rows.append(row)

    df_results = pd.DataFrame(rows)
    ok = df_results["ok"] == True
    summary = {
        "n": int(len(df_results)),
        "ok_rate": float(ok.mean()) if len(df_results) else np.nan,
        "mean_snr": float(df_results.loc[ok, "snr_score"].mean()) if ok.any() else np.nan,
        "median_snr": float(df_results.loc[ok, "snr_score"].median()) if ok.any() else np.nan,
        "p10_snr": float(df_results.loc[ok, "snr_score"].quantile(0.10)) if ok.any() else np.nan,
        "mean_runtime_sec": float(df_results.loc[ok, "runtime_sec"].mean()) if ok.any() else np.nan,
    }

    return df_results, summary

def log_benchmark_run(name, summary):
    row = {"name": name, **summary}
    benchmark_history.append(row)
    return pd.DataFrame(benchmark_history).sort_values("name")

def compare_runs(df_old, df_new):
    a = df_old[["id","source","snr_score"]].rename(columns={"snr_score":"snr_old"})
    b = df_new[["id","source","snr_score"]].rename(columns={"snr_score":"snr_new"})
    m = a.merge(b, on=["id","source"], how="inner")
    m["delta"] = m["snr_new"] - m["snr_old"]
    return m.sort_values("delta")
```

```python
# set benchmarking sample data
#sources_9 = ["0001", "0003", "0004", "0005", "0006", "0009", "0010", "0011", "0012"]
#df_bench = make_benchmark_manifest(sources_9, per_source=3)
df_bench = pd.DataFrame()
df_bench["id"] = pd.Series(['1404687891', '3646789790', '4066709567', '1395556771', '3817987033', '474136527', '3166474998', '3926202474', '88853795', '1536586741', '3110772881', '4099267973', '1736488789', '2450104795', '3042142152', '2589114748', '3591987991', '4229829773', '1420515841', '1919286802', '2800037729', '1278116713', '4246336200', '474136527', '1665441750', '3828947205', '4019229265'])
df_bench["source"] = pd.Series(['0001', '0001', '0001', '0003', '0003', '0003', '0004', '0004', '0004', '0005', '0005', '0005', '0006', '0006', '0006', '0009', '0009', '0009', '0010', '0010', '0010', '0011', '0011', '0011', '0012', '0012', '0012'])

# run benchmarking
df_results_base, summary_base = run_benchmark(df_train, pipeline, df_bench, use_source_preprocess=False)
#display(df_results_base.sort_values("snr_score"))

# log performance
benchmark_history = []
df_hist = log_benchmark_run("baseline", summary_base)
display(df_hist)
```

```python
def clahe_luminance_bgr(img_bgr, clip=2.0, tile=8):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(int(tile), int(tile)))
    l2 = clahe.apply(l)
    lab2 = cv2.merge([l2, a, b])
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)

def grayworld_white_balance(img_bgr):
    img = img_bgr.astype(np.float32)
    b, g, r = cv2.split(img)
    mb, mg, mr = b.mean(), g.mean(), r.mean()
    m = (mb + mg + mr) / 3.0
    b *= (m / (mb + 1e-6))
    g *= (m / (mg + 1e-6))
    r *= (m / (mr + 1e-6))
    out = cv2.merge([b, g, r])
    return np.clip(out, 0, 255).astype(np.uint8)

def denoise_median(img_bgr, k=3):
    k = int(k) if int(k) % 2 == 1 else int(k) + 1
    return cv2.medianBlur(img_bgr, k)

def denoise_bilateral(img_bgr, d=7, sigmaColor=50, sigmaSpace=50):
    return cv2.bilateralFilter(img_bgr, d=int(d), sigmaColor=float(sigmaColor), sigmaSpace=float(sigmaSpace))

def illumination_strength(img_bgr, sigma=35):
    """How non-uniform is the illumination? higher => more shadows/gradients."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    blur = cv2.GaussianBlur(gray, (0, 0), sigma)
    return float(np.std(blur))

def bg_correct_lab_l(img_bgr, k=81):
    """
    Background correction on LAB L channel using morphological opening.
    Keeps A/B (color) intact -> safer for red grid / trace cues.
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    k = int(k)
    if k % 2 == 0:
        k += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

    bg = cv2.morphologyEx(l, cv2.MORPH_OPEN, kernel)
    l_corr = cv2.subtract(l, bg)
    l_corr = cv2.normalize(l_corr, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    lab2 = cv2.merge([l_corr, a, b])
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)

def preprocess_by_source(img_bgr, source):
    s = str(source)

    if s == "0001":
        return img_bgr

    # 0003: color scan
    if s == "0003":
        x = grayworld_white_balance(img_bgr)
        x = clahe_luminance_bgr(x, clip=1.2, tile=8)  # very mild
        return x

    # 0004: bw scan
    if s == "0004":
        # keep super safe (your earlier catastrophic case suggests don't touch much)
        return img_bgr

    # 0006: laptop screen photos (your earlier version worked)
    if s == "0006":
        x = denoise_bilateral(img_bgr, d=5, sigmaColor=25, sigmaSpace=25)
        x = clahe_luminance_bgr(x, clip=1.2, tile=8)
        return x

    # 0005: mobile photo of printed page
    if s == "0005":
        x = img_bgr
        illum = illumination_strength(x, sigma=35)
        if illum > 0.14:                     # stricter gate than before
            x = bg_correct_lab_l(x, k=81)    # LAB-L correction (safer than grayscale/division)
        # mild contrast only if needed
        if cv2.cvtColor(x, cv2.COLOR_BGR2GRAY).std() < 30:
            x = clahe_luminance_bgr(x, clip=1.1, tile=8)
        return x

    # 0009: stained/soaked
    if s == "0009":
        x = img_bgr
        illum = illumination_strength(x, sigma=35)
        if illum > 0.14:                     # strict gate
            x = bg_correct_lab_l(x, k=101)   # larger kernel for bigger stains/shadows
        # IMPORTANT: skip CLAHE here (often amplifies stains/grid)
        x = denoise_median(x, k=3)
        return x

    # 0010: damage (keep mild)
    if s == "0010":
        x = img_bgr
        illum = illumination_strength(x, sigma=35)
        if illum > 0.14:
            x = bg_correct_lab_l(x, k=81)
        if cv2.cvtColor(x, cv2.COLOR_BGR2GRAY).std() < 30:
            x = clahe_luminance_bgr(x, clip=1.15, tile=8)
        return x

    # 0011: mold color scan -> treat like scan (NO bg correction)
    if s == "0011":
        x = grayworld_white_balance(img_bgr)
        x = clahe_luminance_bgr(x, clip=1.2, tile=8)
        return x

    # 0012: mold bw scan -> safest is: do nothing (your deltas were often 0 anyway)
    if s == "0012":
        return img_bgr

    return img_bgr
```

# Try Improvements

```python
# run benchmarking
df_results_source_aware, summary_source_aware = run_benchmark(df_train, pipeline, df_bench, use_source_preprocess=True)
#display(df_results_source_aware.sort_values("snr_score"))

# log performance
df_hist = log_benchmark_run("source_aware", summary_source_aware)
display(df_hist)
```

```python
# compare runs
df_delta = compare_runs(df_results_base, df_results_source_aware)
display(df_delta)
print("Mean delta:", df_delta["delta"].mean())
```

# Do Submission

```python
# load test dataset
WORK_DIR = "/kaggle/input/physionet-ecg-image-digitization"
df_test = pd.read_csv(f"{WORK_DIR}/test.csv")
df_test['id'] = df_test['id'].astype(str) 

# load submission dataset
sample_submission = pd.read_parquet(f"{WORK_DIR}/sample_submission.parquet")
sample_submission[['id_root', 'row', 'lead']] = sample_submission['id'].str.split('_', expand=True)
```

```python
# ---- set these to match your classifier training config ----
CLS_MODEL_NAME = "efficientnet_b2"
CLS_NUM_CLASSES = 12
CLS_RESOLUTION = 256

# path to trained weights (put your .pth in a Kaggle Dataset or /kaggle/working)
CLS_CKPT_PATH = "/kaggle/input/physionet-image-multi-class-train/efficientnet_b2_full_train.pth"

def build_classifier(device="cuda"):
    model = timm.create_model(CLS_MODEL_NAME, pretrained=False, num_classes=CLS_NUM_CLASSES)
    state = torch.load(CLS_CKPT_PATH, map_location="cpu")
    # common patterns:
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    # remove possible "module." prefixes
    new_state = {}
    for k, v in state.items():
        new_state[k.replace("module.", "")] = v
    model.load_state_dict(new_state, strict=False)
    model.to(device).eval()
    return model

def cls_preprocess_bgr(img_bgr):
    # match typical training pipeline: resize -> RGB -> float -> normalize
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (CLS_RESOLUTION, CLS_RESOLUTION), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    # ImageNet norm (most timm pipelines use this unless you changed it)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std
    x = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)  # (1,3,H,W)
    return x

@torch.no_grad()
def predict_source_suffix(model, img_bgr, device="cuda"):
    x = cls_preprocess_bgr(img_bgr).to(device)
    logits = model(x)
    probs = F.softmax(logits, dim=1)[0]
    cls = int(torch.argmax(probs).item())          # 0..11
    conf = float(probs[cls].item())
    suffix = f"{cls+1:04d}"                        # 0001..0012
    return suffix, conf

def select_stage1_with_source(
    pipeline,
    img_raw_bgr,
    pred_source_suffix,
    use_stage1_selector=True,
    selector_margin=1.02
):
    # candidate preprocessing exactly like benchmark, but using predicted source
    img_pp = preprocess_by_source(img_raw_bgr.copy(), pred_source_suffix)

    # RAW
    s0_raw = pipeline.run_stage0(img_raw_bgr)
    s1_raw = pipeline.run_stage1(s0_raw)
    q_raw  = stage1_quality(s1_raw)

    if not use_stage1_selector:
        return s1_raw, "raw", q_raw, None

    # PP
    s0_pp  = pipeline.run_stage0(img_pp)
    s1_pp  = pipeline.run_stage1(s0_pp)
    q_pp   = stage1_quality(s1_pp)

    if q_pp > q_raw * selector_margin:
        return s1_pp, "preprocessed", q_raw, q_pp
    return s1_raw, "raw", q_raw, q_pp
```

```python
def d_series_to_submission_df(sample_id, fs, sig_len, d_series):
    """
    Build submission dataframe in competition format:
    columns = [id, value, id_root, row, lead]
    """
    rows = []

    id_root = int(sample_id)

    for lead in LEADS_ORDER:
        y = np.asarray(d_series[lead], dtype=np.float32)

        # safety: fix length
        if len(y) > sig_len:
            y = y[:sig_len]
        elif len(y) < sig_len:
            yy = np.zeros(sig_len, dtype=np.float32)
            yy[:len(y)] = y
            y = yy

        for i, v in enumerate(y):
            rows.append({
                "id": f"{id_root}_{i}_{lead}",
                "value": float(v),
                "id_root": id_root,
                "row": i,
                "lead": lead,
            })

    return pd.DataFrame(rows)
    
@torch.no_grad()
def predict_one_test(
    sample_id,
    image_path,
    df_meta,          # df_test
    pipeline,
    cls_model,
    device="cuda",
    selector_margin=1.02
):
    img_raw = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img_raw is None:
        raise FileNotFoundError(image_path)

    # infer length from metadata (same as your notebook)
    length = int(df_meta[(df_meta["id"] == sample_id) & (df_meta["lead"] == "II")].iloc[0].number_of_rows)

    # classify source (used ONLY to pick preprocessing recipe)
    pred_src, conf = predict_source_suffix(cls_model, img_raw, device=device)

    # ALWAYS run selector between raw and preprocess_by_source(pred_src)
    s1, chosen, q_raw, q_pp = select_stage1_with_source(
        pipeline, img_raw, pred_src,
        use_stage1_selector=True,
        selector_margin=selector_margin
    )

    pixel, series_4row = pipeline.run_stage2(s1, length=length)

    return series_4row, {
        "pred_source": pred_src,     # e.g. "0009"
        "conf": float(conf),         # classifier confidence (log only)
        "chosen": chosen,            # raw / preprocessed
        "q_raw": float(q_raw) if q_raw is not None else None,
        "q_pp": float(q_pp) if q_pp is not None else None,
        "length": int(length),
    }

def make_submission_from_pred(base_id: str, fs: int, sig_len: int, d_series: dict):
    """
    Build TEST submission DF exactly matching the submission part of make_eval_dfs_from_container:
      - rows: id, value
      - Lead II: full length = sig_len
      - Other leads: short segment length = floor(fs*2.5)
    Segment rule for test (no GT): take first n_expected samples (pad with last value).
    """
    sub_rows = []
    base_id = str(base_id)
    fs = int(fs)
    sig_len = int(sig_len)

    n_short = int(np.floor(fs * 2.5))

    def take_segment(y, n):
        y = np.asarray(y, dtype=np.float64)
        if len(y) == 0:
            return np.zeros(n, dtype=np.float64)
        if len(y) >= n:
            return y[:n]
        # pad with last value
        pad = n - len(y)
        return np.concatenate([y, np.full(pad, y[-1], dtype=np.float64)])

    for lead in LEADS_ORDER:
        y_pred = np.asarray(d_series[lead], dtype=np.float64)

        if lead == "II":
            seg = take_segment(y_pred, sig_len)
        else:
            seg = take_segment(y_pred, n_short)

        for row_idx, vp in enumerate(seg):
            row_id = f"{base_id}_{row_idx}_{lead}"
            sub_rows.append({"id": row_id, "value": float(vp)})

    return pd.DataFrame(sub_rows)


# do submission
device = "cuda" if torch.cuda.is_available() else "cpu"
cls_model = build_classifier(device=device)
print("Classifier ready:", type(cls_model))

res = []
logs = []
for sample_id, df_meta in tqdm(df_test.groupby("id"), total=df_test["id"].nunique()):
    img_path = f"{WORK_DIR}/test/{sample_id}.png"
    fs = int(df_meta.iloc[0].fs)
    sig_len = int(df_meta[df_meta["lead"] == "II"].iloc[0].number_of_rows)

    series_4row, dbg = predict_one_test(
        sample_id=sample_id,
        image_path=img_path,
        df_meta=df_test,
        pipeline=pipeline,
        cls_model=cls_model,
        selector_margin=1.02,
    )

    # ---- LOG CLASSIFIER OUTPUT ----
    logs.append({
        "id": int(sample_id),
        "pred_source": dbg["pred_source"],
        "conf": dbg["conf"],
        "chosen": dbg["chosen"],
        "q_raw": dbg["q_raw"],
        "q_pp": dbg["q_pp"],
    })

    # ---- submission build (unchanged) ----
    d_series = dw(series_dict(series_4row))
    sub_df = make_submission_from_pred(sample_id, fs, sig_len, d_series)
    res.append(sub_df)

# logs data
df_logs = pd.DataFrame(logs)
display(df_logs)

# submission data
df_submission = pd.concat(res, ignore_index=True)
df_submission.to_csv("submission.csv", index=False)
print("submission shape:", df_submission.shape)
df_submission.head()
```

```python
# do submission
print(all(df_submission.id == sample_submission.id))
df_submission.to_csv('submission.csv', index=False)
```