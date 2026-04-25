# Bird25|nfnet+convnextv2|WeightedBlend|LB.860

- **Author:** yukiZ
- **Votes:** 433
- **Ref:** hideyukizushi/bird25-nfnet-convnextv2-weightedblend-lb-860
- **URL:** https://www.kaggle.com/code/hideyukizushi/bird25-nfnet-convnextv2-weightedblend-lb-860
- **Last run:** 2025-05-19 08:05:46.133000

---

<h1 style="color: #6cb4e4;  text-align: center;  padding: 0.25em;  border-top: solid 2.5px #6cb4e4;  border-bottom: solid 2.5px #6cb4e4;  background: -webkit-repeating-linear-gradient(-45deg, #f0f8ff, #f0f8ff 3px,#e9f4ff 3px, #e9f4ff 7px);  background: repeating-linear-gradient(-45deg, #f0f8ff, #f0f8ff 3px,#e9f4ff 3px, #e9f4ff 7px);height:45px;">
<b>
Only Submission(LoadLocalTrainModel)
</b></h1>

### **ℹ️INFO**
* This notebook is an weighted blend(nfnet * 0.6 + convnextv2 * 0.4).
    * **GreatWork LB.850(nfnet)** https://www.kaggle.com/code/myso1987/post-processing-with-power-adjustment-for-low-rank
    * **MyLocalTrainModel(convnextv2)** https://www.kaggle.com/datasets/hideyukizushi/bird25-d-330v2-ppv15-convnextv2-nano/data

### **ℹ️WeightedBlend**
* In CV tasks, it is common to adopt diverse backbones to ensure robustness, and I am sharing this in this competition because it is connected to the LB boost.
* P.S. Adjusting the blending weights should make it fit LB better and improve the score. In that case, it is redundant, so I recommend submitting in a private notebook rather than a public notebook.

### **ℹ️Appendix**
* My old LB.829 notebook from this competition
* https://www.kaggle.com/code/hideyukizushi/bird25-onlyinf-v2-s-focallossbce-cv-962-lb-829

<h1 style="color: #6cb4e4;  text-align: center;  padding: 0.25em;  border-top: solid 2.5px #6cb4e4;  border-bottom: solid 2.5px #6cb4e4;  background: -webkit-repeating-linear-gradient(-45deg, #f0f8ff, #f0f8ff 3px,#e9f4ff 3px, #e9f4ff 7px);  background: repeating-linear-gradient(-45deg, #f0f8ff, #f0f8ff 3px,#e9f4ff 3px, #e9f4ff 7px);height:45px;">
<b>
《《《Submission1(convnextv2)》》》
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

import numpy as np
import pandas as pd
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR)
```

```python
class CFG:
 
    test_soundscapes = '/kaggle/input/birdclef-2025/test_soundscapes'
    submission_csv = '/kaggle/input/birdclef-2025/sample_submission.csv'
    taxonomy_csv = '/kaggle/input/birdclef-2025/taxonomy.csv'
    
    # ------------------------------------------- #
    # [IMPORTANT]
    # * Melspectrogram & Audio Params
    # ------------------------------------------- #
    FS = 32000  
    WINDOW_SIZE = 5
    N_FFT = 2048
    HOP_LENGTH = 512
    N_MELS = 512
    FMIN = 20
    FMAX = 16000
    TARGET_SHAPE = (256, 256)

    # ------------------------------------------- #
    # * Model def
    # ------------------------------------------- #
    model_path = '/kaggle/input/bird25-d-330v2-ppv15-convnextv2-nano'
    model_name = 'convnextv2_nano.fcmae_ft_in22k_in1k'
    use_specific_folds = True
    folds = [0,1]
    in_channels = 1
    device = 'cpu'  
    
    # Inference parameters
    batch_size = 16
    use_tta = False  
    tta_count = 3
    threshold = 0.5

    # util
    debug = False
    debug_count = 3

cfg = CFG()

print(f"Using device: {cfg.device}")
print(f"Loading taxonomy data...")
taxonomy_df = pd.read_csv(cfg.taxonomy_csv)
species_ids = taxonomy_df['primary_label'].tolist()
num_classes = len(species_ids)
print(f"Number of classes: {num_classes}")
```

```python
class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1)*p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)

    def __repr__(self):
        return self.__class__.__name__ + \
                '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + \
                ', ' + 'eps=' + str(self.eps) + ')'
class BirdCLEFModel(nn.Module):
    def __init__(self, cfg, num_classes):
        super().__init__()
        self.cfg = cfg
        
        self.backbone = timm.create_model(
            cfg.model_name,
            pretrained=False,  
            in_chans=cfg.in_channels,
            drop_rate=0.0,    
            drop_path_rate=0.0
        )
        
        if 'efficientnet' in cfg.model_name:
            backbone_out = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()
        elif 'resnet' in cfg.model_name:
            backbone_out = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        else:
            backbone_out = self.backbone.get_classifier().in_features
            self.backbone.reset_classifier(0, '')
        
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.feat_dim = backbone_out
        self.classifier = nn.Linear(backbone_out, num_classes)
        
    def forward(self, x):
        features = self.backbone(x)
        
        if isinstance(features, dict):
            features = features['features']
            
        if len(features.shape) == 4:
            features = self.pooling(features)
            features = features.view(features.size(0), -1)
        
        logits = self.classifier(features)
        return logits
```

```python
def audio2melspec(audio_data, cfg):
    """Convert audio data to mel spectrogram"""
    if np.isnan(audio_data).any():
        mean_signal = np.nanmean(audio_data)
        audio_data = np.nan_to_num(audio_data, nan=mean_signal)

    mel_spec = librosa.feature.melspectrogram(
        y=audio_data,
        sr=cfg.FS,
        n_fft=cfg.N_FFT,
        hop_length=cfg.HOP_LENGTH,
        n_mels=cfg.N_MELS,
        fmin=cfg.FMIN,
        fmax=cfg.FMAX,
        power=2.0,
        pad_mode="reflect",
        norm='slaney',
        htk=True,
        center=True,
    )

    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)
    
    return mel_spec_norm

def process_audio_segment(audio_data, cfg):
    """Process audio segment to get mel spectrogram"""
    if len(audio_data) < cfg.FS * cfg.WINDOW_SIZE:
        audio_data = np.pad(audio_data, 
                          (0, cfg.FS * cfg.WINDOW_SIZE - len(audio_data)), 
                          mode='constant')
    
    mel_spec = audio2melspec(audio_data, cfg)
    
    # Resize if needed
    if mel_spec.shape != cfg.TARGET_SHAPE:
        mel_spec = cv2.resize(mel_spec, cfg.TARGET_SHAPE, interpolation=cv2.INTER_LINEAR)
        
    return mel_spec.astype(np.float32)
    
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
    
    model_files = find_model_files(cfg)
    
    if not model_files:
        print(f"Warning: No model files found under {cfg.model_path}!")
        return models
    
    print(f"Found a total of {len(model_files)} model files.")
    
    if cfg.use_specific_folds:
        filtered_files = []
        for fold in cfg.folds:
            fold_files = [f for f in model_files if f"fold{fold}" in f]
            filtered_files.extend(fold_files)
        model_files = filtered_files
        print(f"Using {len(model_files)} model files for the specified folds ({cfg.folds}).")
    
    for model_path in model_files:
        try:
            print(f"Loading model: {model_path}")
            checkpoint = torch.load(model_path, map_location=torch.device(cfg.device))
            
            model = BirdCLEFModel(cfg, num_classes)
            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.to(cfg.device)
            model.eval()
            
            models.append(model)
        except Exception as e:
            print(f"Error loading model {model_path}: {e}")
    
    return models

def predict_on_spectrogram(audio_path, models, cfg, species_ids):
    """Process a single audio file and predict species presence for each 5-second segment"""
    predictions = []
    row_ids = []
    soundscape_id = Path(audio_path).stem
    
    try:
        print(f"Processing {soundscape_id}")
        audio_data, _ = librosa.load(audio_path, sr=cfg.FS)
        
        total_segments = int(len(audio_data) / (cfg.FS * cfg.WINDOW_SIZE))
        
        for segment_idx in range(total_segments):
            start_sample = segment_idx * cfg.FS * cfg.WINDOW_SIZE
            end_sample = start_sample + cfg.FS * cfg.WINDOW_SIZE
            segment_audio = audio_data[start_sample:end_sample]
            
            end_time_sec = (segment_idx + 1) * cfg.WINDOW_SIZE
            row_id = f"{soundscape_id}_{end_time_sec}"
            row_ids.append(row_id)

            if cfg.use_tta:
                all_preds = []
                
                for tta_idx in range(cfg.tta_count):
                    mel_spec = process_audio_segment(segment_audio, cfg)
                    mel_spec = apply_tta(mel_spec, tta_idx)

                    mel_spec = torch.tensor(mel_spec, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                    mel_spec = mel_spec.to(cfg.device)

                    if len(models) == 1:
                        with torch.no_grad():
                            outputs = models[0](mel_spec)
                            probs = torch.sigmoid(outputs).cpu().numpy().squeeze()
                            all_preds.append(probs)
                    else:
                        segment_preds = []
                        for model in models:
                            with torch.no_grad():
                                outputs = model(mel_spec)
                                probs = torch.sigmoid(outputs).cpu().numpy().squeeze()
                                segment_preds.append(probs)
                        
                        avg_preds = np.mean(segment_preds, axis=0)
                        all_preds.append(avg_preds)

                final_preds = np.mean(all_preds, axis=0)
            else:
                mel_spec = process_audio_segment(segment_audio, cfg)
                
                mel_spec = torch.tensor(mel_spec, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                mel_spec = mel_spec.to(cfg.device)
                
                if len(models) == 1:
                    with torch.no_grad():
                        outputs = models[0](mel_spec)
                        final_preds = torch.sigmoid(outputs).cpu().numpy().squeeze()
                else:
                    segment_preds = []
                    for model in models:
                        with torch.no_grad():
                            outputs = model(mel_spec)
                            probs = torch.sigmoid(outputs).cpu().numpy().squeeze()
                            segment_preds.append(probs)

                    final_preds = np.mean(segment_preds, axis=0)
                    
            predictions.append(final_preds)
            
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
    
    return row_ids, predictions
```

```python
def apply_tta(spec, tta_idx):
    """Apply test-time augmentation"""
    if tta_idx == 0:
        # Original spectrogram
        return spec
    elif tta_idx == 1:
        # Time shift (horizontal flip)
        return np.flip(spec, axis=1)
    elif tta_idx == 2:
        # Frequency shift (vertical flip)
        return np.flip(spec, axis=0)
    else:
        return spec

def run_inference(cfg, models, species_ids):
    """Run inference on all test soundscapes"""
    test_files = list(Path(cfg.test_soundscapes).glob('*.ogg'))
    
    if cfg.debug:
        print(f"Debug mode enabled, using only {cfg.debug_count} files")
        test_files = test_files[:cfg.debug_count]
    
    print(f"Found {len(test_files)} test soundscapes")

    all_row_ids = []
    all_predictions = []

    for audio_path in tqdm(test_files):
        row_ids, predictions = predict_on_spectrogram(str(audio_path), models, cfg, species_ids)
        all_row_ids.extend(row_ids)
        all_predictions.extend(predictions)
    
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
```

```python
def main():
    start_time = time.time()
    print("Starting BirdCLEF-2025 inference...")
    print(f"TTA enabled: {cfg.use_tta} (variations: {cfg.tta_count if cfg.use_tta else 0})")

    models = load_models(cfg, num_classes)
    
    if not models:
        print("No models found! Please check model paths.")
        return
    
    print(f"Model usage: {'Single model' if len(models) == 1 else f'Ensemble of {len(models)} models'}")

    row_ids, predictions = run_inference(cfg, models, species_ids)

    submission_df = create_submission(row_ids, predictions, species_ids, cfg)

    submission_path = 'submission1.csv'
    submission_df.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")
    
    end_time = time.time()
    print(f"Inference completed in {(end_time - start_time)/60:.2f} minutes")

if __name__ == "__main__":
    main()
```

```python
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
    new_predictions[0] = (predictions[0] * 0.9) + (predictions[1] * 0.1)
    new_predictions[-1] = (predictions[-1] * 0.9) + (predictions[-2] * 0.1)
    sub_group[cols] = new_predictions
    sub[group == groups] = sub_group
sub.to_csv("submission1.csv", index=False)
```

<h1 style="color: #6cb4e4;  text-align: center;  padding: 0.25em;  border-top: solid 2.5px #6cb4e4;  border-bottom: solid 2.5px #6cb4e4;  background: -webkit-repeating-linear-gradient(-45deg, #f0f8ff, #f0f8ff 3px,#e9f4ff 3px, #e9f4ff 7px);  background: repeating-linear-gradient(-45deg, #f0f8ff, #f0f8ff 3px,#e9f4ff 3px, #e9f4ff 7px);height:45px;">
<b>
《《《Submission2(nfnet)》》》
</b></h1>

```python
import numpy as np
from typing import Union

def apply_power_to_low_ranked_cols(
    p: np.ndarray,
    top_k: int = 30,
    exponent: Union[int, float] = 2,
    inplace: bool = True
) -> np.ndarray:
    """
    Rank columns by their column‑wise maximum and raise every column whose
    rank falls below `top_k` to a given power.

    Parameters
    ----------
    p : np.ndarray
        A 2‑D array of shape **(n_chunks, n_classes)**.

        - **n_chunks** is the number of fixed‑length time chunks obtained
          after slicing the input audio (or other sequential data).  
          *Example:* In the BirdCLEF `test_soundscapes` set, each file is
          60 s long. If you extract non‑overlapping 5 s windows,  
          `n_chunks = 60 s / 5 s = 12`.
        - **n_classes** is the number of classes being predicted.
        - Each element `p[i, j]` is the score or probability of class *j*
          in chunk *i*.

    top_k : int, default=30
        The highest‑ranked columns (by their maximum value) that remain
        unchanged.

    exponent : int or float, default=2
        The power applied to the selected low‑ranked columns  
        (e.g. `2` squares, `0.5` takes the square root, `3` cubes).

    inplace : bool, default=True
        If `True`, modify `p` in place.  
        If `False`, operate on a copy and leave the original array intact.

    Returns
    -------
    np.ndarray
        The transformed array. It is the same object as `p` when
        `inplace=True`; otherwise, it is a new array.

    """
    if not inplace:
        p = p.copy()

    # Identify columns whose max value ranks below `top_k`
    tail_cols = np.argsort(-p.max(axis=0))[top_k:]

    # Apply the power transformation to those columns
    p[:, tail_cols] = p[:, tail_cols] ** exponent
    return p
```

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
import concurrent.futures
```

```python
test_audio_dir = '../input/birdclef-2025/test_soundscapes/'
file_list = [f for f in sorted(os.listdir(test_audio_dir))]
file_list = [file.split('.')[0] for file in file_list if file.endswith('.ogg')]

debug = False
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
f_min=50
f_max=16000
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
display(results.head())
```

```python
results.to_csv("submission.csv", index=False)    

sub = pd.read_csv('submission.csv')
cols = sub.columns[1:]
groups = sub['row_id'].str.rsplit('_', n=1).str[0]
groups = groups.values
for group in np.unique(groups):
    sub_group = sub[group == groups]
    predictions = sub_group[cols].values
    new_predictions = predictions.copy()
    for i in range(1, predictions.shape[0]-1):
        new_predictions[i] = (predictions[i-1] * 0.2) + (predictions[i] * 0.6) + (predictions[i+1] * 0.2)
    new_predictions[0] = (predictions[0] * 0.9) + (predictions[1] * 0.1)
    new_predictions[-1] = (predictions[-1] * 0.9) + (predictions[-2] * 0.1)
    sub_group[cols] = new_predictions
    sub[group == groups] = sub_group
sub.to_csv("submission.csv", index=False)


if debug:
    display(results)
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
sub_w=[0.6,0.4]
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