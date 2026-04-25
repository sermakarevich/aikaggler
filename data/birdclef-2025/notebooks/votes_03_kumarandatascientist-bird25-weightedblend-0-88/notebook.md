# Bird25 | WeightedBlend  | 0.88

- **Author:** Kumaran K
- **Votes:** 580
- **Ref:** kumarandatascientist/bird25-weightedblend-0-88
- **URL:** https://www.kaggle.com/code/kumarandatascientist/bird25-weightedblend-0-88
- **Last run:** 2025-05-31 16:57:06.447000

---

credit goes to this notebook and author just changed config here

https://www.kaggle.com/code/johnyim1570/bird25-weightedblend-nfnet-seresnext-0-878

# 🐦 BirdCLEF 2025: Weighted Blend Inference (LB 0.878)

## 🧠 Summary

This notebook achieves a public leaderboard score of **0.878** using a **Weighted Blend** approach.

## 🧩 Models Used

The final prediction is based on the ensemble of the following public models:

- 📘 **[Bird2025 | Single SED Model Inference [LB 0.857]](https://www.kaggle.com/code/i2nfinit3y/bird2025-single-sed-model-inference-lb-0-857)**  
  by [I2nfinit3y](https://www.kaggle.com/i2nfinit3y)  
  A strong single SED model serving as the base for one of the ensemble components.

- 🧪 **[Post-Processing with Power Adjustment for Low-Rank](https://www.kaggle.com/code/myso1987/post-processing-with-power-adjustment-for-low-rank)**  
  by [MYSO](https://www.kaggle.com/myso1987)  
  This notebook introduced a clever post-processing technique to boost low-confidence predictions via power transformation.

- 🔗 **[Bird25 | WeightedBlend | nfnet + convnextv2 | LB.860](https://www.kaggle.com/code/hideyukizushi/bird25-weightedblend-nfnet-convnextv2-lb-860)**  
  by [yukiZ](https://www.kaggle.com/hideyukizushi)  
  Provided the core blending logic used to combine model outputs.

## ⚖️ Weighted Blend Strategy

We use the weighted average of the two model outputs as follows:

- **nfnet**: 25%
- **seresnext**: 75%

While this blend achieves a high LB score, it may be overfitting to the public test set — a pattern often seen in BirdCLEF competitions from 2022 to 2024. Please consider this when evaluating the results.

---

<h1 style="color: #6cb4e4;  text-align: center;  padding: 0.25em;  border-top: solid 2.5px #6cb4e4;  border-bottom: solid 2.5px #6cb4e4;  background: -webkit-repeating-linear-gradient(-45deg, #f0f8ff, #f0f8ff 3px,#e9f4ff 3px, #e9f4ff 7px);  background: repeating-linear-gradient(-45deg, #f0f8ff, #f0f8ff 3px,#e9f4ff 3px, #e9f4ff 7px);height:45px;">
<b>
《《《Submission1(nfnet)》》》
</b></h1>

```python
import os
import gc
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import timm
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as AT
from contextlib import contextmanager
from typing import Union
import concurrent.futures
```

```python
def apply_power_to_low_ranked_cols(
    p: np.ndarray,
    top_k: int = 30,
    exponent: Union[int, float] = 2,
    inplace: bool = True
) -> np.ndarray:
    if not inplace:
        p = p.copy()

    # Identify columns whose max value ranks below `top_k`
    tail_cols = np.argsort(-p.max(axis=0))[top_k:]

    # Apply the power transformation to those columns
    p[:, tail_cols] = p[:, tail_cols] ** exponent
    return p
```

```python
test_audio_dir = '../input/birdclef-2025/test_soundscapes/'
file_list = [f for f in sorted(os.listdir(test_audio_dir))]
file_list = [file.split('.')[0] for file in file_list if file.endswith('.ogg')]

debug = False
if len(file_list) == 0:
    debug = True
    debug_st_num = 5
    debug_num = 8
    test_audio_dir = '../input/birdclef-2025/train_soundscapes/'
    file_list = [f for f in sorted(os.listdir(test_audio_dir))]
    file_list = [file.split('.')[0] for file in file_list if file.endswith('.ogg')]
    file_list = file_list[debug_st_num:debug_st_num+debug_num]

print('Debug mode:', debug)
print('Number of test soundscapes:', len(file_list))
```

```python
wav_sec = 5
sample_rate = 32000
min_segment = sample_rate*wav_sec

class_labels = sorted(os.listdir('../input/birdclef-2025/train_audio/'))

n_fft=1024
win_length=1024
hop_length=512
f_min=40
f_max=15000
n_mels=128

mel_spectrogram = AT.MelSpectrogram(
    sample_rate=sample_rate,
    n_fft=n_fft,
    win_length=win_length,
    hop_length=hop_length,
    center=True,
    f_min=f_min,
    f_max=f_max,
    pad_mode="reflect",
    power=2.0,
    norm='slaney',
    n_mels=n_mels,
    mel_scale="htk",
    # normalized=True
)

def normalize_std(spec, eps=1e-6):
    mean = torch.mean(spec)
    std = torch.std(spec)
    return torch.where(std == 0, spec-mean, (spec - mean) / (std+eps))

def audio_to_mel(filepath=None):
    waveform, sample_rate = torchaudio.load(filepath,backend="soundfile")
    len_wav = waveform.shape[1]
    waveform = waveform[0,:].reshape(1, len_wav) # stereo->mono mono->mono
    PREDS = []
    for i in range(12):
        waveform2 = waveform[:,i*sample_rate*5:i*sample_rate*5+sample_rate*5]
        melspec = mel_spectrogram(waveform2)
        melspec = torch.log(melspec+1e-6)
        melspec = normalize_std(melspec)
        melspec = torch.unsqueeze(melspec, dim=0)
        
        PREDS.append(melspec)
    return torch.vstack(PREDS)
```

```python
def init_layer(layer):
    nn.init.xavier_uniform_(layer.weight)
    if hasattr(layer, "bias"):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)


def init_bn(bn):
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.0)


def init_weights(model):
    classname = model.__class__.__name__
    if classname.find("Conv2d") != -1:
        nn.init.xavier_uniform_(model.weight, gain=np.sqrt(2))
        model.bias.data.fill_(0)
    elif classname.find("BatchNorm") != -1:
        model.weight.data.normal_(1.0, 0.02)
        model.bias.data.fill_(0)
    elif classname.find("GRU") != -1:
        for weight in model.parameters():
            if len(weight.size()) > 1:
                nn.init.orghogonal_(weight.data)
    elif classname.find("Linear") != -1:
        model.weight.data.normal_(0, 0.01)
        model.bias.data.zero_()


def interpolate(x, ratio):
    (batch_size, time_steps, classes_num) = x.shape
    upsampled = x[:, :, None, :].repeat(1, 1, ratio, 1)
    upsampled = upsampled.reshape(batch_size, time_steps * ratio, classes_num)
    return upsampled


def pad_framewise_output(framewise_output, frames_num):
    output = F.interpolate(
        framewise_output.unsqueeze(1),
        size=(frames_num, framewise_output.size(2)),
        align_corners=True,
        mode="bilinear").squeeze(1)

    return output


class AttBlockV2(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 activation="linear"):
        super().__init__()

        self.activation = activation
        self.att = nn.Conv1d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True)
        self.cla = nn.Conv1d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True)

        self.init_weights()

    def init_weights(self):
        init_layer(self.att)
        init_layer(self.cla)

    def forward(self, x):
        norm_att = torch.softmax(torch.tanh(self.att(x)), dim=-1)
        cla = self.nonlinear_transform(self.cla(x))
        x = torch.sum(norm_att * cla, dim=2)
        return x, norm_att, cla

    def nonlinear_transform(self, x):
        if self.activation == 'linear':
            return x
        elif self.activation == 'sigmoid':
            return torch.sigmoid(x)


class TimmSED(nn.Module):
    def __init__(self, base_model_name: str, pretrained=False, num_classes=24, in_channels=1, n_mels=24):
        super().__init__()

        self.bn0 = nn.BatchNorm2d(n_mels)

        base_model = timm.create_model(
            base_model_name, pretrained=pretrained, in_chans=in_channels)
        layers = list(base_model.children())[:-2]
        self.encoder = nn.Sequential(*layers)

        in_features = base_model.num_features

        self.fc1 = nn.Linear(in_features, in_features, bias=True)
        self.att_block2 = AttBlockV2(
            in_features, num_classes, activation="sigmoid")

        self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)
        init_layer(self.fc1)
        

    def forward(self, input_data):
        x = input_data.transpose(2,3)
        x = torch.cat((x,x,x),1)

        x = x.transpose(2, 3)

        x = self.encoder(x)
        
        x = torch.mean(x, dim=2)

        x1 = F.max_pool1d(x, kernel_size=3, stride=1, padding=1)
        x2 = F.avg_pool1d(x, kernel_size=3, stride=1, padding=1)
        x = x1 + x2

        x = x.transpose(1, 2)
        x = F.relu_(self.fc1(x))
        x = x.transpose(1, 2)

        (clipwise_output, norm_att, segmentwise_output) = self.att_block2(x)
        logit = torch.sum(norm_att * self.att_block2.cla(x), dim=2)

        output_dict = {
            'logit': logit,
        }

        return output_dict
```

```python
base_model_name='eca_nfnet_l0'
pretrained=False
in_channels=3

MODELS = [f'/kaggle/input/birdclef-2025-sed-models-p/sed{i}.pth' for i in range(3)]

MODELS
```

```python
models = []
for path in MODELS:
    model = TimmSED(base_model_name=base_model_name,
               pretrained=pretrained,
               num_classes=len(class_labels),
               in_channels=in_channels,
               n_mels=n_mels);
    model.load_state_dict(torch.load(path, weights_only=True, map_location=torch.device('cpu')))
    model.eval();
    models.append(model)
```

```python
def prediction(afile):    
    global pred
    path = test_audio_dir + afile + '.ogg'
    with torch.inference_mode():
        sig = audio_to_mel(path)
        outputs = None
        for model in models:
            model.eval()
            p = model(sig)
            p = torch.sigmoid(p['logit']).detach().cpu().numpy() 
            p = apply_power_to_low_ranked_cols(p, top_k=30,exponent=2)
            if outputs is None: outputs = p
            else: outputs += p
            
        outputs /= len(models)
        chunks = [[] for i in range(12)]
        for i in range(len(chunks)):        
            chunk_end_time = (i + 1) * 5
            row_id = afile + '_' + str(chunk_end_time)
            pred['row_id'].append(row_id)
            bird_no = 0
            for bird in class_labels:         
                pred[bird].append(outputs[i,bird_no])
                bird_no += 1
        gc.collect()
```

```python
pred = {'row_id': []}
for species_code in class_labels:
    pred[species_code] = []
    
start = time.time()
with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
    _ = list(executor.map(prediction, file_list))
end_t = time.time()

if debug == True:
    print(700*(end_t - start)/60/debug_num)
```

```python
results = pd.DataFrame(pred, columns = ['row_id'] + class_labels)
```

```python
results.to_csv("submission1.csv", index=False)    

sub = pd.read_csv('submission1.csv')
cols = sub.columns[1:]
groups = sub['row_id'].str.rsplit('_', n=1).str[0]
groups = groups.values
for group in np.unique(groups):
    sub_group = sub[group == groups]
    predictions = sub_group[cols].values
    new_predictions = predictions.copy()
    for i in range(1, predictions.shape[0]-1):
        new_predictions[i] = (predictions[i-1] * 0.2) + (predictions[i] * 0.6) + (predictions[i+1] * 0.2)
    new_predictions[0] = (predictions[0] * 0.8) + (predictions[1] * 0.2)
    new_predictions[-1] = (predictions[-1] * 0.8) + (predictions[-2] * 0.2)
    sub_group[cols] = new_predictions
    sub[group == groups] = sub_group
sub.to_csv("submission1.csv", index=False)
```

<h1 style="color: #6cb4e4;  text-align: center;  padding: 0.25em;  border-top: solid 2.5px #6cb4e4;  border-bottom: solid 2.5px #6cb4e4;  background: -webkit-repeating-linear-gradient(-45deg, #f0f8ff, #f0f8ff 3px,#e9f4ff 3px, #e9f4ff 7px);  background: repeating-linear-gradient(-45deg, #f0f8ff, #f0f8ff 3px,#e9f4ff 3px, #e9f4ff 7px);height:45px;">
<b>
《《《Submission2(seresnext)》》》
</b></h1>

```python
import os
import gc
import warnings
import logging
import time
import math
import cv2
from pathlib import Path
import joblib

import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from soundfile import SoundFile 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
import timm
from tqdm.auto import tqdm
from glob import glob
import torchaudio
import random
import itertools
from typing import Union

import concurrent.futures

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR)
```

```python
class CFG:
    
    seed = 42
    print_freq = 100
    num_workers = 4

    stage = 'train_bce'

    train_datadir = '/kaggle/input/birdclef-2025/train_audio'
    train_csv = '/kaggle/input/birdclef-2025/train.csv'
    test_soundscapes = '/kaggle/input/birdclef-2025/test_soundscapes'
    submission_csv = '/kaggle/input/birdclef-2025/sample_submission.csv'
    taxonomy_csv = '/kaggle/input/birdclef-2025/taxonomy.csv'
    model_files = ['/kaggle/input/bird2025-sed-ckpt/sedmodel.pth'
                  ]
 
    model_name = 'seresnext26t_32x4d'  
    pretrained = False
    in_channels = 1

    
    SR = 32000
    target_duration = 5
    train_duration = 10
    
    
    device = 'cpu'

cfg = CFG()
```

```python
print(f"Using device: {cfg.device}")
print(f"Loading taxonomy data...")
taxonomy_df = pd.read_csv(cfg.taxonomy_csv)
species_ids = taxonomy_df['primary_label'].tolist()
num_classes = len(species_ids)
print(f"Number of classes: {num_classes}")
```

```python
def set_seed(seed=42):
    """
    Set seed for reproducibility
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(cfg.seed)
```

```python
class AttBlockV2(nn.Module):
    def __init__(self, in_features: int, out_features: int, activation="linear"):
        super().__init__()

        self.activation = activation
        self.att = nn.Conv1d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )
        self.cla = nn.Conv1d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

        self.init_weights()

    def init_weights(self):
        init_layer(self.att)
        init_layer(self.cla)

    def forward(self, x):
        # x: (n_samples, n_in, n_time)
        norm_att = torch.softmax(torch.tanh(self.att(x)), dim=-1)
        cla = self.nonlinear_transform(self.cla(x))
        x = torch.sum(norm_att * cla, dim=2)
        return x, norm_att, cla

    def nonlinear_transform(self, x):
        if self.activation == "linear":
            return x
        elif self.activation == "sigmoid":
            return torch.sigmoid(x)


def init_layer(layer):
    nn.init.xavier_uniform_(layer.weight)

    if hasattr(layer, "bias"):
        if layer.bias is not None:
            layer.bias.data.fill_(0.0)

def init_bn(bn):
    bn.bias.data.fill_(0.0)
    bn.weight.data.fill_(1.0)
```

```python
class BirdCLEFModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
        taxonomy_df = pd.read_csv('/kaggle/input/birdclef-2025/taxonomy.csv')
        self.num_classes = len(taxonomy_df)

        self.bn0 = nn.BatchNorm2d(cfg['n_mels'])
        
        self.backbone = timm.create_model(
            cfg['model_name'],
            pretrained=False,
            in_chans=cfg['in_channels'],
            drop_rate=0.2,
            drop_path_rate=0.2,
        )

        layers = list(self.backbone.children())[:-2]
        self.encoder = nn.Sequential(*layers)
        
        if "efficientnet" in self.cfg['model_name']:
            backbone_out = self.backbone.classifier.in_features
        elif "eca" in self.cfg['model_name']:
            backbone_out = self.backbone.head.fc.in_features
        elif "res" in self.cfg['model_name']:
            backbone_out = self.backbone.fc.in_features
        else:
            backbone_out = self.backbone.num_features
            
        
        self.fc1 = nn.Linear(backbone_out, backbone_out, bias=True)
        self.att_block = AttBlockV2(backbone_out, self.num_classes, activation="sigmoid")

        self.melspec_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.cfg['SR'],
            hop_length=self.cfg['hop_length'],
            n_mels=self.cfg['n_mels'],
            f_min=self.cfg['f_min'],
            f_max=self.cfg['f_max'],
            n_fft=self.cfg['n_fft'],
            pad_mode="constant",
            norm="slaney",
            onesided=True,
            mel_scale="htk",
        )
        if self.cfg['device'] == "cuda":
            self.melspec_transform = self.melspec_transform.cuda()
        else:
            self.melspec_transform = self.melspec_transform.cpu()

        self.db_transform = torchaudio.transforms.AmplitudeToDB(
            stype="power", top_db=80
        )


    def extract_feature(self,x):
        x = x.permute((0, 1, 3, 2))
        frames_num = x.shape[2]
        
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        
        # if self.training:
        #    x = self.spec_augmenter(x)
        
        x = x.transpose(2, 3)
        # (batch_size, channels, freq, frames)
        x = self.encoder(x)
        
        # (batch_size, channels, frames)
        x = torch.mean(x, dim=2)
        
        # channel smoothing
        x1 = F.max_pool1d(x, kernel_size=3, stride=1, padding=1)
        x2 = F.avg_pool1d(x, kernel_size=3, stride=1, padding=1)
        x = x1 + x2
        
        x = F.dropout(x, p=0.5, training=self.training)
        x = x.transpose(1, 2)
        x = F.relu_(self.fc1(x))
        x = x.transpose(1, 2)
        x = F.dropout(x, p=0.5, training=self.training)
        return x, frames_num
        
    @torch.cuda.amp.autocast(enabled=False)
    def transform_to_spec(self, audio):

        audio = audio.float()
        
        spec = self.melspec_transform(audio)
        spec = self.db_transform(spec)

        if self.cfg['normal'] == 80:
            spec = (spec + 80) / 80
        elif self.cfg['normal'] == 255:
            spec = spec / 255
        else:
            raise NotImplementedError
                
        if self.cfg['in_channels'] == 3:
            spec = image_delta(spec)
        
        return spec

    def forward(self, x):

        with torch.no_grad():
            x = self.transform_to_spec(x)

        x, frames_num = self.extract_feature(x)
        
        (clipwise_output, norm_att, segmentwise_output) = self.att_block(x)
        logit = torch.sum(norm_att * self.att_block.cla(x), dim=2)
        segmentwise_logit = self.att_block.cla(x).transpose(1, 2)
        segmentwise_output = segmentwise_output.transpose(1, 2)

        return torch.logit(clipwise_output)

    def infer(self, x, tta_delta=2):
        with torch.no_grad():
            x = self.transform_to_spec(x)
        x,_ = self.extract_feature(x)
        time_att = torch.tanh(self.att_block.att(x))
        feat_time = x.size(-1)
        start = (
            feat_time / 2 - feat_time * (self.cfg['infer_duration'] / self.cfg['duration_train']) / 2
        )
        end = start + feat_time * (self.cfg['infer_duration'] / self.cfg['duration_train'])
        start = int(start)
        end = int(end)
        pred = self.attention_infer(start,end,x,time_att)

        start_minus = max(0, start-tta_delta)
        end_minus=end-tta_delta
        pred_minus = self.attention_infer(start_minus,end_minus,x,time_att)

        start_plus = start+tta_delta
        end_plus=min(feat_time, end+tta_delta)
        pred_plus = self.attention_infer(start_plus,end_plus,x,time_att)

        pred = 0.5*pred + 0.25*pred_minus + 0.25*pred_plus
        return pred
        
    def attention_infer(self,start,end,x,time_att):
        feat = x[:, :, start:end]
        # att = torch.softmax(time_att[:, :, start:end], dim=-1)
        #             print(feat_time, start, end)
        #             print(att_a.sum(), att.sum(), time_att.shape)
        framewise_pred = torch.sigmoid(self.att_block.cla(feat))
        framewise_pred_max = framewise_pred.max(dim=2)[0]
        # clipwise_output = torch.sum(framewise_pred * att, dim=-1)
        #logits = torch.sum(
        #    self.att_block.cla(feat) * att,
        #    dim=-1,
        #)

        # return clipwise_output
        return framewise_pred_max
```

```python
def load_sample(path, cfg):
    audio, orig_sr = sf.read(path, dtype="float32")
    seconds = []
    audio_length = cfg.SR * cfg.target_duration
    step = audio_length
    for i in range(audio_length, len(audio) + step, step):
        start = max(0, i - audio_length)
        end = start + audio_length
        if end > len(audio):
            pass
        else:
            seconds.append(int(end/cfg.SR))

    audio = np.concatenate([audio,audio,audio])
    audios = []
    for i,second in enumerate(seconds):
        end_seconds = int(second)
        start_seconds = int(end_seconds - cfg.target_duration)
    
        end_index = int(cfg.SR * (end_seconds + (cfg.train_duration - cfg.target_duration) / 2) ) + len(audio) // 3
        start_index = int(cfg.SR * (start_seconds - (cfg.train_duration - cfg.target_duration) / 2) ) + len(audio) // 3
        end_pad = int(cfg.SR * (cfg.train_duration - cfg.target_duration) / 2) 
        start_pad = int(cfg.SR * (cfg.train_duration - cfg.target_duration) / 2) 
        y = audio[start_index:end_index].astype(np.float32)
        if i==0:
            y[:start_pad] = 0
        elif i==(len(seconds)-1):
            y[-end_pad:] = 0
        audios.append(y)

    return audios

def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s
```

```python
def find_model_files(cfg):
    """
    Find all .pth model files in the specified model directory
    """
    model_files = []
    
    model_dir = Path(cfg.model_path)
    
    for path in model_dir.glob('**/*.pth'):
        model_files.append(str(path))
    
    return model_files

def load_models(cfg, num_classes):
    """
    Load all found model files and prepare them for ensemble
    """
    models = []
    
    # model_files = find_model_files(cfg)
    model_files = cfg.model_files
    
    if not model_files:
        print(f"Warning: No model files found under {cfg.model_path}!")
        return models
    
    print(f"Found a total of {len(model_files)} model files.")
    
    for i, model_path in enumerate(model_files):
        try:
            print(f"Loading model: {model_path}")
            checkpoint = torch.load(model_path, map_location=torch.device(cfg.device), weights_only=False)
            cfg_temp = checkpoint['cfg']
            cfg_temp['device'] = cfg.device
            
            model = BirdCLEFModel(cfg_temp)
            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.to(cfg.device)
            model.eval()
            model.zero_grad()
            model.half().float()
            
            models.append(model)
        except Exception as e:
            print(f"Error loading model {model_path}: {e}")
    
    return models

def predict_on_spectrogram(audio_path, models, cfg, species_ids):
    """Process a single audio file and predict species presence for each 5-second segment"""
    audio_path = str(audio_path)
    predictions = []
    row_ids = []
    soundscape_id = Path(audio_path).stem

    print(f"Processing {soundscape_id}")
    audio_data = load_sample(audio_path, cfg)
    for segment_idx, audio_input in enumerate(audio_data):
        
        end_time_sec = (segment_idx + 1) * cfg.target_duration
        row_id = f"{soundscape_id}_{end_time_sec}"
        row_ids.append(row_id)
        
        mel_spec = torch.tensor(audio_input, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        mel_spec = mel_spec.to(cfg.device)
        
        if len(models) == 1:
            with torch.no_grad():
                outputs = models[0].infer(mel_spec)
                final_preds = outputs.squeeze()
                # final_preds = torch.sigmoid(outputs).cpu().numpy().squeeze()

        else:
            segment_preds = []
            for model in models:
                with torch.no_grad():
                    outputs = model.infer(mel_spec)
                    probs = outputs.squeeze()
                    # probs = torch.sigmoid(outputs).cpu().numpy().squeeze()
                    segment_preds.append(probs)

            
            final_preds = np.mean(segment_preds, axis=0)
                
        predictions.append(final_preds)

    predictions = np.stack(predictions,axis=0)
    
    return row_ids, predictions
```

```python
def run_inference(cfg, models, species_ids):
    """Run inference on all test soundscapes"""
    test_files = list(Path(cfg.test_soundscapes).glob('*.ogg'))
    if len(test_files) == 0:
        test_files = sorted(glob(str(Path('/kaggle/input/birdclef-2025/train_soundscapes') / '*.ogg')))[:10]
    
    print(f"Found {len(test_files)} test soundscapes")

    all_row_ids = []
    all_predictions = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        results = list(
        executor.map(
            predict_on_spectrogram,
            test_files,
            itertools.repeat(models),
            itertools.repeat(cfg),
            itertools.repeat(species_ids)
        )
    )

    for rids, preds in results:
        all_row_ids.extend(rids)
        all_predictions.extend(preds)
    
    return all_row_ids, all_predictions

def create_submission(row_ids, predictions, species_ids, cfg):
    """Create submission dataframe"""
    print("Creating submission dataframe...")

    submission_dict = {'row_id': row_ids}
    
    for i, species in enumerate(species_ids):
        submission_dict[species] = [pred[i] for pred in predictions]

    submission_df = pd.DataFrame(submission_dict)

    submission_df.set_index('row_id', inplace=True)

    sample_sub = pd.read_csv(cfg.submission_csv, index_col='row_id')

    missing_cols = set(sample_sub.columns) - set(submission_df.columns)
    if missing_cols:
        print(f"Warning: Missing {len(missing_cols)} species columns in submission")
        for col in missing_cols:
            submission_df[col] = 0.0

    submission_df = submission_df[sample_sub.columns]

    submission_df = submission_df.reset_index()
    
    return submission_df


def smooth_submission(submission_path):
        """
        Post-process the submission CSV by smoothing predictions to enforce temporal consistency.
        
        For each soundscape (grouped by the file name part of 'row_id'), each row's predictions
        are averaged with those of its neighbors using defined weights.
        
        :param submission_path: Path to the submission CSV file.
        """
        print("Smoothing submission predictions...")
        sub = pd.read_csv(submission_path)
        cols = sub.columns[1:]
        # Extract group names by splitting row_id on the last underscore
        groups = sub['row_id'].str.rsplit('_', n=1).str[0].values
        unique_groups = np.unique(groups)
        
        for group in unique_groups:
            # Get indices for the current group
            idx = np.where(groups == group)[0]
            sub_group = sub.iloc[idx].copy()
            predictions = sub_group[cols].values
            new_predictions = predictions.copy()
            
            if predictions.shape[0] > 1:
                # Smooth the predictions using neighboring segments
                new_predictions[0] = (predictions[0] * 0.8) + (predictions[1] * 0.2)
                new_predictions[-1] = (predictions[-1] * 0.8) + (predictions[-2] * 0.2)
                for i in range(1, predictions.shape[0]-1):
                    new_predictions[i] = (predictions[i-1] * 0.2) + (predictions[i] * 0.6) + (predictions[i+1] * 0.2)
            # Replace the smoothed values in the submission dataframe
            sub.iloc[idx, 1:] = new_predictions
        
        sub.to_csv(submission_path, index=False)
        print(f"Smoothed submission saved to {submission_path}")
```

```python
def main():
    start_time = time.time()
    print("Starting BirdCLEF-2025 inference...")

    models = load_models(cfg, num_classes)
    
    if not models:
        print("No models found! Please check model paths.")
        return
    
    print(f"Model usage: {'Single model' if len(models) == 1 else f'Ensemble of {len(models)} models'}")

    row_ids, predictions = run_inference(cfg, models, species_ids)

    submission_df = create_submission(row_ids, predictions, species_ids, cfg)

    submission_path = 'submission.csv'
    submission_df.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")

    smooth_submission(submission_path)
    
    end_time = time.time()
    print(f"Inference completed in {(end_time - start_time)/60:.2f} minutes")
```

```python
if __name__ == "__main__":
    main()
```

<h1 style="color: #6cb4e4;  text-align: center;  padding: 0.25em;  border-top: solid 2.5px #6cb4e4;  border-bottom: solid 2.5px #6cb4e4;  background: -webkit-repeating-linear-gradient(-45deg, #f0f8ff, #f0f8ff 3px,#e9f4ff 3px, #e9f4ff 7px);  background: repeating-linear-gradient(-45deg, #f0f8ff, #f0f8ff 3px,#e9f4ff 3px, #e9f4ff 7px);height:45px;">
<b>
《《《Finaly Blending》》》
</b></h1>

```python
# ------------------------------------------- #
# [IMPORTANT]
# * Blending Weight
# ------------------------------------------- #
sub_w=[0.753, 0.247]
```

```python
list_TARGETs = sorted(os.listdir('/kaggle/input/birdclef-2025/train_audio/'))
list_targets_0 = [f'{TARGET} 0' for TARGET in list_TARGETs]
list_targets_1 = [f'{TARGET} 1' for TARGET in list_TARGETs]

df0 = pd.read_csv("/kaggle/working/submission.csv")
df1 = pd.read_csv("/kaggle/working/submission1.csv")

df0 = df0.rename(columns={TARGET : f'{TARGET} 0' for TARGET in list_TARGETs})
df1 = df1.rename(columns={TARGET : f'{TARGET} 1' for TARGET in list_TARGETs})

dfs = pd.merge(df0,df1,on=['row_id'])

for i in range(len(list_TARGETs)):
    dfs[list_TARGETs[i]] = dfs[list_targets_0[i]]*sub_w[0] + sub_w[1]*dfs[list_targets_1[i]]
             
for col0,col1 in zip(list_targets_0, list_targets_1):
    del dfs[col0]
    del dfs[col1]
    
    
dfs.to_csv("submission.csv", index=False)
```