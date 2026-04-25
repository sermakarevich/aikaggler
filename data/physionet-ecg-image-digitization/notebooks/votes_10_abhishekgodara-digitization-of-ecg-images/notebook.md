# Digitization of ECG images🔥🔥

- **Author:** Abhishek Godara
- **Votes:** 178
- **Ref:** abhishekgodara/digitization-of-ecg-images
- **URL:** https://www.kaggle.com/code/abhishekgodara/digitization-of-ecg-images
- **Last run:** 2026-01-06 21:25:32.427000

---

```python
!pip uninstall -y tensorflow
!uv pip install --no-deps --system --no-index --find-links='/kaggle/input/hengck23-submit-physionet/hengck23-submit-physionet/setup' 'connected-components-3d'
```

```python
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

test_dir = Path("/kaggle/input/physionet-ecg-image-digitization/test")

test = pd.read_csv("/kaggle/input/physionet-ecg-image-digitization/test.csv")
test['id'] = test['id'].astype(str) 
test_id = test['id'].unique().tolist()


global_dict = {
    "stage0_dir": "/kaggle/working/stage0",
    "stage1_dir": "/kaggle/working/stage1",
    "stage2_dir": "/kaggle/working/stage2",
}

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

def dw(series_dict, alpha=0.33):
    if all(k in series_dict for k in ['I', 'II', 'III']):
        L1 = series_dict['I']
        L2 = series_dict['II']
        L3 = series_dict['III']

        error = L2 - (L1 + L3)

        series_dict['I'] = L1 + (alpha * error)
        series_dict['III'] = L3 + (alpha * error)
        series_dict['II'] = L2 - (alpha * error)
    
    return series_dict

def series_to_dict_local(series_4row):
    d = {}
    names = [['I', 'aVR', 'V1', 'V4'], 
             ['II', 'aVL', 'V2', 'V5'], 
             ['III', 'aVF', 'V3', 'V6']]
    
    for i in range(3):
        splits = np.array_split(series_4row[i], 4)
        for name, data in zip(names[i], splits):
            d[name] = data
    
    d['II_Long'] = series_4row[3]
    return d

def dict_to_series_local(d, original_shape):
    new_series = np.zeros(original_shape)
    new_series[0] = np.concatenate([d['I'], d['aVR'], d['V1'], d['V4']])
    new_series[1] = np.concatenate([d['II'], d['aVL'], d['V2'], d['V5']])
    new_series[2] = np.concatenate([d['III'], d['aVF'], d['V3'], d['V6']])
    new_series[3] = d['II_Long']
    
    return new_series

stage0_dir = Path(global_dict["stage0_dir"])
stage0_dir.mkdir(exist_ok=True)

stage0_net = Stage0Net(pretrained=False)
stage0_net = load_net(stage0_net, '/kaggle/input/hengck23-submit-physionet/hengck23-submit-physionet/weight/stage0-last.checkpoint.pth')
stage0_net.to("cuda:0")
stage0_net.eval()

for n, sample_id in enumerate(tqdm(test_id)):
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

for n, sample_id in enumerate(tqdm(test_id)):
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
mv_to_pixel = 78.8
t0, t1 = 235, 4161

resize = T.Resize((1696, 4352), interpolation=T.InterpolationMode.BILINEAR)

for n, sample_id in enumerate(tqdm(test_id)):
    path = stage1_dir / f'{sample_id}.png'
    output_path = stage2_dir / f'{sample_id}.npy'
    image = cv2.imread(path, cv2.IMREAD_COLOR_RGB)
    
    length = test[(test['id']==sample_id) & (test['lead']=='II')].iloc[0].number_of_rows
    
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
        
        s_dict = series_to_dict_local(series)
        s_dict = dw(s_dict)
        series = dict_to_series_local(s_dict, series.shape)

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
gb = test.groupby('id')

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
submission.to_csv('submission.csv', index=False)
submission.head(30)
```