# Post-Processing with Power Adjustment for Low-Rank

- **Author:** MYSO
- **Votes:** 354
- **Ref:** myso1987/post-processing-with-power-adjustment-for-low-rank
- **URL:** https://www.kaggle.com/code/myso1987/post-processing-with-power-adjustment-for-low-rank
- **Last run:** 2025-05-15 20:43:26.687000

---

# Post-Processing with Power Adjustment for Low-Ranked Classes
This notebook demonstrates a simple post-processing method applied during inference. 
While this improves the LB score, please note that it may be overfitting to the leaderboard.

# Post-Processing Function

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

# Import

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

# Load data

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

# Model

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

# Predict

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
    new_predictions[0] = (predictions[0] * 0.8) + (predictions[1] * 0.2)
    new_predictions[-1] = (predictions[-1] * 0.8) + (predictions[-2] * 0.2)
    sub_group[cols] = new_predictions
    sub[group == groups] = sub_group
sub.to_csv("submission.csv", index=False)


if debug:
    display(results)
```

# Sound Event Visualisations
* Related Notebooks  
[BirdCLEF2025+:Sound Event Visualisations](https://www.kaggle.com/code/myso1987/birdclef2025-sound-event-visualisations)

```python
if debug == True:
    import numpy as np
    import matplotlib.pyplot as plt
    
    sample_rate = 32000
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
    
    def audio_to_mel_debug(filepath=None):
        waveform, sample_rate = torchaudio.load(filepath,backend="soundfile")
        len_wav = waveform.shape[1]
        waveform = waveform / torch.max(torch.abs(waveform))
        melspec = mel_spectrogram(waveform)
        melspec = 10*torch.log10(melspec)
        return melspec
    
    def plot_results(results, file_name):
        path = test_audio_dir + file_name + ".ogg"
        specgram = audio_to_mel_debug(path)
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))
        axes[0].set_title(file_name)
        im = axes[0].imshow((specgram[0]), origin="lower", aspect="auto")
        axes[0].set_ylabel("mel bin")
        axes[0].set_xlabel("frame")
        fig.colorbar(im, ax=axes[0])
        heatmap = axes[1].pcolor(results[results["row_id"].str.contains(file_name)].iloc[:12,1:].values.T, edgecolors='k', linewidths=0.1, vmin=0, vmax=1, cmap='Blues')
        fig.colorbar(heatmap, ax=axes[1])
        axes[1].set_xticks(np.arange(0, 12, 1))
        axes[1].set_xticklabels(np.arange(0,60,5))
        axes[1].set_ylabel("sec")
        axes[1].set_xlabel("species")
        fig.tight_layout()
        fig.show()
        
    for file_name in file_list:
        plot_results(results, file_name)
```