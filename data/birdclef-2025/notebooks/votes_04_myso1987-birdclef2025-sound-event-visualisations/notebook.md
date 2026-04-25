# BirdCLEF2025+:Sound Event Visualisations

- **Author:** MYSO
- **Votes:** 531
- **Ref:** myso1987/birdclef2025-sound-event-visualisations
- **URL:** https://www.kaggle.com/code/myso1987/birdclef2025-sound-event-visualisations
- **Last run:** 2025-05-25 13:00:14.160000

---

# BirdCLEF2025+:Sound Event Visualisations

I will introduce sound event visualizations for train_soundscapes data. 

This notebook uses [STEFAN KAHL](https://www.kaggle.com/stefankahl)'s [BirdCLEF+ 2025 Sample Submission](https://www.kaggle.com/code/stefankahl/birdclef-2025-sample-submission) as a starter. 

## Related Notebooks
* Dataset Creation:[BirdCLEF2025-1 Crop audio 5s](https://www.kaggle.com/code/myso1987/birdclef2025-1-crop-audio-5s)
* Training Notebook:[BirdCLEF2025-2 Train-baseline 5s](https://www.kaggle.com/code/myso1987/birdclef2025-2-train-baseline-5s)
*  Inference Notebook:[BirdCLEF2025-3 Submit-baseline 5s](https://www.kaggle.com/code/myso1987/birdclef2025-3-submit-baseline-5s)

## Reference
[[Birdclef2022] Soundscape Visualisations](https://www.kaggle.com/code/shinmurashinmura/birdclef2022-soundscape-visualisations/notebook)

# Import

```python
import os
import sys
import time
import torch
import torchaudio
import torchaudio.transforms as AT
sys.path.append("/kaggle/input/birdclef2025-utils")
from birdclef2025_utils import get_results
```

# Predict(train_soundscapes data)

```python
test_audio_dir = '../input/birdclef-2025/test_soundscapes/'
file_list = [f for f in sorted(os.listdir(test_audio_dir))]
file_list = [file.split('.')[0] for file in file_list if file.endswith('.ogg')]
class_labels = sorted(os.listdir('../input/birdclef-2025/train_audio/'))

debug = False
debug_st_num=0
debug_num=0
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
start = time.time()
results = get_results.get_results(test_audio_dir,file_list,debug,debug_num)
results.to_csv("submission.csv", index=False)    
end_t = time.time()

if debug == True:
    print('Estimated submission runtime(minutes):', 700*(end_t - start)/60/debug_num)
```

# Utilities for Sound Event Visualisations

```python
if debug == True:
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    
    sample_rate = 32000
    n_fft=1024
    win_length=1024
    hop_length=512
    f_min=20
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
        axes[1].set_ylabel("species")
        axes[1].set_xlabel("sec")
        fig.tight_layout()
        fig.show()

    def load_plot_results(csv_path):
        results_l = pd.read_csv(csv_path)
        results_l = results_l.set_index('row_id').loc[results['row_id']]
        results_l = results_l.reset_index()
        display(results_l.head())
    
        for file_name in file_list:
            plot_results(results_l, file_name)
```

# Sound Event Visualisations(LB latest)

```python
if debug == True:
    display(results.head())
    for file_name in file_list:
        plot_results(results, file_name)
```

# Sound Event Visualisations(LB 0.903)

```python
if debug == True:
    load_plot_results("/kaggle/input/birdclef-2025submission-csvtrain-soundscapes/submission_903.csv")
```

# Sound Event Visualisations(LB 0.883)

```python
if debug == True:
    load_plot_results("/kaggle/input/birdclef-2025submission-csvtrain-soundscapes/submission_883.csv")
```

# Sound Event Visualisations(LB 0.854)

```python
if debug == True:
    load_plot_results("/kaggle/input/birdclef-2025submission-csvtrain-soundscapes/submission_854.csv")
```

# Sound Event Visualisations(LB 0.815)

```python
if debug == True:
    load_plot_results("/kaggle/input/birdclef-2025submission-csvtrain-soundscapes/submission_815.csv")
```