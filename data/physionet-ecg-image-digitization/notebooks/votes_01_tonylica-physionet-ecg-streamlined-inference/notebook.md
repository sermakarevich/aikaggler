# PhysioNet - ECG - streamlined inference

- **Author:** Tony Li
- **Votes:** 422
- **Ref:** tonylica/physionet-ecg-streamlined-inference
- **URL:** https://www.kaggle.com/code/tonylica/physionet-ecg-streamlined-inference
- **Last run:** 2026-01-14 15:56:04.027000

---

hengck23’s excellent solution (demo submission) - 16.1 baseline:
https://www.kaggle.com/code/hengck23/demo-submission

wasupandceacar’s Net3 pipeline - 17.75 baseline:
https://www.kaggle.com/code/wasupandceacar/physio-v2-3-public

Saner Turhaner ’s Visual QA  - 18.17 baseline:
https://www.kaggle.com/code/sanpier/visual-qa-for-all-stages-of-ecg-digitization

```python
!pip uninstall -y tensorflow
!uv pip install --no-deps --system --no-index --find-links='/kaggle/input/hengck23-submit-physionet/hengck23-submit-physionet/setup' connected-components-3d

import os, sys, gc, cv2, numpy as np, pandas as pd, torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import timm
from scipy.signal import savgol_filter

sys.path.append('/kaggle/input/hengck23-submit-physionet/hengck23-submit-physionet')
import stage0_common as s0c
import stage1_common as s1c
import stage2_common as s2c
from stage0_model import Net as Stage0Net
from stage1_model import Net as Stage1Net
from stage2_model import MyCoordUnetDecoder, encode_with_resnet

LEADS_ORDER = ["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]

def change_color(image_rgb):
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    v_denoised = cv2.fastNlMeansDenoising(v, h=5.46)
    std = np.std(v_denoised)
    clip_limit = max(1.0, min(3.5, 2.0 + std / 25))
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    v_enhanced = clahe.apply(v_denoised)
    hsv_enhanced = cv2.merge([h, s, v_enhanced])
    return cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2RGB)

def series_dict(series_4row):
    series_4row = np.asarray(series_4row)
    if series_4row.ndim == 3: series_4row = series_4row[0]
    if series_4row.shape[0] != 4 and series_4row.shape[1] == 4: series_4row = series_4row.T

    d = {}
    names = [
        ['I','aVR','V1','V4'],
        ['II_short','aVL','V2','V5'],   
        ['III','aVF','V3','V6'],
    ]
    for r in range(3):
        for lead, arr in zip(names[r], np.array_split(series_4row[r], 4)):
            d[lead] = np.asarray(arr, dtype=np.float32)

    d['II'] = np.asarray(series_4row[3], dtype=np.float32)  # full 10s II
    return d

# ✅ FIX: Einthoven correction on SHORT leads only
def dw(d, alpha=0.33):
    if all(k in d for k in ['I','II_short','III']):
        L1, L2s, L3 = d['I'], d['II_short'], d['III']
        e = L2s - (L1 + L3)
        d['I']        = L1 + alpha*e
        d['III']      = L3 + alpha*e
        d['II_short'] = L2s - alpha*e
    return d

def clahe_luminance_bgr(img_bgr, clip=2.0, tile=8):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(int(tile), int(tile)))
    l2 = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l2, a, b]), cv2.COLOR_LAB2BGR)

def grayworld_white_balance(img_bgr):
    img = img_bgr.astype(np.float32)
    b, g, r = cv2.split(img)
    mb, mg, mr = b.mean(), g.mean(), r.mean()
    m = (mb + mg + mr) / 3.0
    b *= (m / (mb + 1e-6)); g *= (m / (mg + 1e-6)); r *= (m / (mr + 1e-6))
    return np.clip(cv2.merge([b, g, r]), 0, 255).astype(np.uint8)

def denoise_median(img_bgr, k=3):
    k = int(k); k = k if k % 2 == 1 else k + 1
    return cv2.medianBlur(img_bgr, k)

def denoise_bilateral(img_bgr, d=7, sigmaColor=50, sigmaSpace=50):
    return cv2.bilateralFilter(img_bgr, d=int(d), sigmaColor=float(sigmaColor), sigmaSpace=float(sigmaSpace))

def illumination_strength(img_bgr, sigma=35):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    blur = cv2.GaussianBlur(gray, (0, 0), sigma)
    return float(np.std(blur))

def bg_correct_lab_l(img_bgr, k=81):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    k = int(k); k = k if k % 2 == 1 else k + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    bg = cv2.morphologyEx(l, cv2.MORPH_OPEN, kernel)
    l_corr = cv2.subtract(l, bg)
    l_corr = cv2.normalize(l_corr, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.cvtColor(cv2.merge([l_corr, a, b]), cv2.COLOR_LAB2BGR)

def preprocess_by_source(img_bgr, source):
    s = str(source)
    if s == "0001": return img_bgr
    if s == "0003": return clahe_luminance_bgr(grayworld_white_balance(img_bgr), clip=1.2, tile=8)
    if s == "0004": return img_bgr
    if s == "0006":
        x = denoise_bilateral(img_bgr, d=5, sigmaColor=25, sigmaSpace=25)
        return clahe_luminance_bgr(x, clip=1.2, tile=8)
    if s == "0005":
        x = img_bgr
        if illumination_strength(x, sigma=35) > 0.14: x = bg_correct_lab_l(x, k=81)
        if cv2.cvtColor(x, cv2.COLOR_BGR2GRAY).std() < 30: x = clahe_luminance_bgr(x, clip=1.1, tile=8)
        return x
    if s == "0009":
        x = img_bgr
        if illumination_strength(x, sigma=35) > 0.14: x = bg_correct_lab_l(x, k=101)
        return denoise_median(x, k=3)
    if s == "0010":
        x = img_bgr
        if illumination_strength(x, sigma=35) > 0.14: x = bg_correct_lab_l(x, k=81)
        if cv2.cvtColor(x, cv2.COLOR_BGR2GRAY).std() < 30: x = clahe_luminance_bgr(x, clip=1.15, tile=8)
        return x
    if s == "0011": return clahe_luminance_bgr(grayworld_white_balance(img_bgr), clip=1.2, tile=8)
    if s == "0012": return img_bgr
    return img_bgr

def stage1_quality(s1_rgb):
    g = cv2.cvtColor(s1_rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    e = cv2.Canny(g, 50, 150)
    density = e.mean() / 255.0
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    ax = float(np.mean(np.abs(gx))); ay = float(np.mean(np.abs(gy)))
    anis = max(ax, ay) / (min(ax, ay) + 1e-6)
    return float(density * 0.7 + np.tanh(anis - 1.0) * 0.3)

class Net3(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
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
        return self.pixel(last)

class PhysioPipeline:
    def __init__(self, device="cuda:0"):
        self.device = device
        self.stage0_net = self.stage1_net = self.stage2_net = None
        self.x0, self.x1 = 0, 2176
        self.y0, self.y1 = 0, 1696
        self.zero_mv = [703.5, 987.5, 1271.5, 1531.5]
        self.mv_to_pixel = 78.8
        self.t0, self.t1 = 235, 4161
        self.resize = T.Resize((1696, 4352), interpolation=T.InterpolationMode.BILINEAR)

    def load_models(self, stage0_w, stage1_w, stage2_w):
        self.stage0_net = s0c.load_net(Stage0Net(pretrained=False), stage0_w).to(self.device).eval()
        self.stage1_net = s1c.load_net(Stage1Net(pretrained=False), stage1_w).to(self.device).eval()
        self.stage2_net = Net3(pretrained=False).to(self.device).eval()
        st = torch.load(stage2_w, map_location="cpu")
        if isinstance(st, dict) and "state_dict" in st: st = st["state_dict"]
        self.stage2_net.load_state_dict(st, strict=True)

    def run_stage0(self, img_bgr):
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_for_model = change_color(img_rgb)
        batch = s0c.image_to_batch(img_for_model)
        with torch.no_grad(), torch.amp.autocast(self.device.split(":")[0], dtype=torch.float32):
            output = self.stage0_net(batch)
        rotated, keypoint = s0c.output_to_predict(img_rgb, batch, output)
        normalised, _, _ = s0c.normalise_by_homography(rotated, keypoint)
        return normalised

    def run_stage1(self, stage0_img_rgb):
        image = stage0_img_rgb
        batch = {'image': torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).unsqueeze(0)}
        with torch.no_grad(), torch.amp.autocast(self.device.split(":")[0], dtype=torch.float32):
            output = self.stage1_net(batch)
        gridpoint_xy, _ = s1c.output_to_predict(image, batch, output)
        return s1c.rectify_image(image, gridpoint_xy)

    def run_stage2(self, stage1_img_rgb, length):
        img = stage1_img_rgb[self.y0:self.y1, self.x0:self.x1] / 255.0
        batch = self.resize(torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))).unsqueeze(0)).float().to(self.device)
        with torch.no_grad(), torch.amp.autocast(self.device.split(":")[0], dtype=torch.float32):
            output = self.stage2_net(batch)
        pixel = torch.sigmoid(output).float().cpu().numpy()[0]
        series_in_pixel = s2c.pixel_to_series(pixel[..., self.t0:self.t1], self.zero_mv, length)
        series = (np.array(self.zero_mv).reshape(4, 1) - series_in_pixel) / self.mv_to_pixel
        for i in range(4):
            series[i] = savgol_filter(series[i], window_length=7, polyorder=2)
        return series

CLS_MODEL_NAME="efficientnet_b2"
CLS_NUM_CLASSES=12
CLS_RESOLUTION=256
CLS_CKPT_PATH="/kaggle/input/physionet-image-multi-class-train/efficientnet_b2_full_train.pth"

def build_classifier(device="cuda"):
    m = timm.create_model(CLS_MODEL_NAME, pretrained=False, num_classes=CLS_NUM_CLASSES)
    st = torch.load(CLS_CKPT_PATH, map_location="cpu")
    if isinstance(st, dict) and "state_dict" in st: st = st["state_dict"]
    st2 = {k.replace("module.",""): v for k, v in st.items()}
    m.load_state_dict(st2, strict=False)
    return m.to(device).eval()

def cls_preprocess_bgr(img_bgr):
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (CLS_RESOLUTION, CLS_RESOLUTION), interpolation=cv2.INTER_AREA).astype(np.float32)/255.0
    mean = np.array([0.485,0.456,0.406], np.float32); std = np.array([0.229,0.224,0.225], np.float32)
    img = (img - mean) / std
    return torch.from_numpy(img).permute(2,0,1).unsqueeze(0)

@torch.no_grad()
def predict_source_suffix(model, img_bgr, device="cuda"):
    x = cls_preprocess_bgr(img_bgr).to(device)
    p = F.softmax(model(x), dim=1)[0]
    cls = int(torch.argmax(p).item())
    return f"{cls+1:04d}"

def select_stage1_with_source(pipeline, img_raw_bgr, pred_source_suffix, selector_margin=1.02):
    img_pp = preprocess_by_source(img_raw_bgr.copy(), pred_source_suffix)
    s1_raw = pipeline.run_stage1(pipeline.run_stage0(img_raw_bgr))
    q_raw  = stage1_quality(s1_raw)
    s1_pp  = pipeline.run_stage1(pipeline.run_stage0(img_pp))
    q_pp   = stage1_quality(s1_pp)
    return s1_pp if q_pp > q_raw * selector_margin else s1_raw

def make_submission_from_pred(base_id: str, fs: int, sig_len: int, d_series: dict):
    base_id = str(base_id); fs = int(fs); sig_len = int(sig_len)
    n_short = int(np.floor(fs * 2.5))

    def take_segment(y, n):
        y = np.asarray(y, dtype=np.float64)
        if len(y) >= n: return y[:n]
        if len(y) == 0: return np.zeros(n, np.float64)
        return np.concatenate([y, np.full(n - len(y), y[-1], np.float64)])

    rows = []
    for lead in LEADS_ORDER:
        y = np.asarray(d_series[lead], dtype=np.float64)
        seg = take_segment(y, sig_len if lead=="II" else n_short)
        rows.append(pd.DataFrame({"id":[f"{base_id}_{i}_{lead}" for i in range(len(seg))],
                                  "value":seg.astype(np.float32)}))
    return pd.concat(rows, ignore_index=True)

WORK_DIR="/kaggle/input/physionet-ecg-image-digitization"
df_test = pd.read_csv(f"{WORK_DIR}/test.csv")
df_test["id"] = df_test["id"].astype(str)
sample_submission = pd.read_parquet(f"{WORK_DIR}/sample_submission.parquet")[["id"]]

device = "cuda" if torch.cuda.is_available() else "cpu"
pipeline = PhysioPipeline(device="cuda:0" if device=="cuda" else "cpu")
pipeline.load_models(
    stage0_w="/kaggle/input/hengck23-submit-physionet/hengck23-submit-physionet/weight/stage0-last.checkpoint.pth",
    stage1_w="/kaggle/input/hengck23-submit-physionet/hengck23-submit-physionet/weight/stage1-last.checkpoint.pth",
    stage2_w="/kaggle/input/physio-seg-public/pytorch/net3_009_4200/1/iter_0004200.pt",
)
cls_model = build_classifier(device=device)

res = []
for sample_id, g in df_test.groupby("id", sort=True):
    img_path = f"{WORK_DIR}/test/{sample_id}.png"
    img_raw = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img_raw is None:
        raise FileNotFoundError(img_path)

    fs = int(g.fs.iloc[0])
    sig_len = int(g.loc[g.lead=="II","number_of_rows"].iloc[0])

    pred_src = predict_source_suffix(cls_model, img_raw, device=device)
    s1 = select_stage1_with_source(pipeline, img_raw, pred_src, selector_margin=1.02)
    series_4row = pipeline.run_stage2(s1, length=sig_len)

    d = dw(series_dict(series_4row))         # ✅ now safe (uses II_short)
    res.append(make_submission_from_pred(sample_id, fs, sig_len, d))
    gc.collect()

df_submission = pd.concat(res, ignore_index=True)
df_submission = df_submission.set_index("id").reindex(sample_submission["id"]).reset_index()

assert df_submission["value"].notna().all()
assert (df_submission["id"].values == sample_submission["id"].values).all()

df_submission.to_csv("submission.csv", index=False)
print("OK  submission.csv", df_submission.shape)
```