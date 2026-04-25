# MedGemma - HAI-DEF

- **Author:** Marília Prata
- **Votes:** 106
- **Ref:** mpwolke/medgemma-hai-def
- **URL:** https://www.kaggle.com/code/mpwolke/medgemma-hai-def
- **Last run:** 2026-01-13 23:46:30.143000

---

Published on January 13, 2026. By Marília Prata, mpwolke.

```python
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import feature_extraction, linear_model, model_selection, preprocessing
import plotly.graph_objs as go
import plotly.offline as py
import plotly.express as px

#Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
```

### This is a Hackathon with no provided dataset.

![](https://media.licdn.com/dms/image/v2/D4D12AQGhSiCszCM1pQ/article-cover_image-shrink_720_1280/B4DZjquOqnGsAI-/0/1756284653366?e=2147483647&v=beta&t=w8YC9jZ0Zj9-U91zo3TC91ZLgCK6Xjl_3J3xD2V1QwU)

## Competition Citation: The MedGemma Impact Challenge

@misc{med-gemma-impact-challenge,

    author = {Fereshteh Mahvar and Yun Liu and Daniel Golden and Fayaz Jamil and Sunny Jansen and Can Kirmizi and Rory Pilgrim and David F. Steiner and Andrew Sellergren and Richa Tiwari and Sunny Virmani and Liron Yatziv and Rebecca Hemenway and Yossi Matias and Ronit Levavi Morad and Avinatan Hassidim and Shravya Shetty and María Cruz},
    
    title = {The MedGemma Impact Challenge},
    year = {2026},
    howpublished = {\url{https://kaggle.com/competitions/med-gemma-impact-challenge}},
    note = {Kaggle}
}

### Overview

"Google has released open-weight models specifically designed to help developers more efficiently create novel healthcare and life sciences applications. MedGemma and the rest of HAI-DEF collection.
Whether you’re building apps to streamline workflows, support patient communication, or facilitate diagnostics, your solution should demonstrate how these tools can enhance healthcare."

https://www.kaggle.com/competitions/med-gemma-impact-challenge

## Health AI Developer Foundations

Authors: Atilla P. Kiraly, Sebastien Baur, Kenneth Philbrick, Fereshteh Mahvar, Liron Yatziv, Tiffany Chen, Bram Sterling, Nick George, Fayaz Jamil, Jing Tang, Kai Bailey, Faruk Ahmed, Akshay Goel, Abbi Ward, Lin Yang, Andrew Sellergren, Yossi Matias, Avinatan Hassidim, Shravya Shetty, Daniel Golden, Shekoofeh Azizi, David F. Steiner, Yun Liu, Tim Thelin, Rory Pilgrim, Can Kirmizibayrak

"Robust medical Machine Learning (ML) models have the potential to revolutionize healthcare by accelerating clinical research, improving workflows and outcomes, and producing novel insights or capabilities. Developing such ML models from scratch is cost prohibitive and requires substantial compute, data, and time (e.g., expert labeling). To address these challenges, the authors introduced **Health AI Developer Foundations (HAI-DEF)**, a suite of pre-trained, domain-specific foundation models, tools, and recipes to accelerate building ML for health applications."

"The models cover various modalities and domains, including radiology (X-rays and computed tomography), histopathology, dermatological imaging, and audio. These models provide domain specific embeddings that facilitate AI development with less labeled data, shorter training times, and reduced computational costs compared to traditional approaches."

### MODELS

**CXR Foundation**

"CXR Foundation is a set of 3 models, all using an EfficientNet-L2 image encoder backbone. The three models learned representations of CXRs by leveraging both the image data and the clinically relevant information available in corresponding radiology reports."

**Path Foundation**

"Path Foundation is a Vision Transformer (ViT) encoder for histopathology image patches trained with self-supervised learning."

**Derm Foundation**

"Derm Foundation is a BiT ResNet-101x3 image encoder trained using a two-stage approach on over 16K natural and dermatology images."

**HeAR**

"HeAR is a ViT audio encoder trained using a Masked Autoencoder (MAE) approach. The model learns to reconstruct masked spectrogram patches, capturing rich acoustic representations of health-related sounds like coughs and breathing patterns."

**CT Foundation**

"CT Foundation provides embeddings suitable for downstream classification tasks. The underlying
model is VideoCoCa, a video-text model designed for efficient transfer learning from 2D Contrastive Captioners (CoCa)."

### Limitations

"The models were developed with a focus on **classification tasks**, and **prognosis tasks will need to be further evaluated**. **Image segmentation** and generation tasks are also currently **not supported**. Further,specific requirements such as smaller models (e.g. for on-device applications on a mobile device) or lower latency will need other techniques such as distillation to the target model size of interest."

https://arxiv.org/pdf/2411.15128

https://developers.google.com/health-ai-developer-foundations

https://github.com/Google-Health/google-health/blob/master/health_acoustic_representations/README.md

https://github.com/Google-Health/imaging-research/tree/master/ct-foundation

### I tried to run some demo, although I failed.

```python
!pip install lifelines
```

```python
!pip install loss
```

```python
!pip install network
```

```python
import lifelines
import numpy as np
import tensorflow as tf

import loss
import network

NUM_EXAMPLES = 64
SEQUENCE_LENGTH = 2
PATCH_SIZE = 128
NUM_EPOCHS = 32
```

```python
!pip install cluster_utils
```

```python
!pip install data utils
```

```python
import math
import sklearn

import cluster_utils
#import data_utils  #No module named 'data_utils'
```

## hear_demo.ipynb (attempt)

"Health Acoustics Representations (HeAR) is a machine learning (ML) model that produces embeddings based on health acoustic data. The embeddings can be used to efficiently build AI models for health acoustic-related tasks (for example, identifying disease status from cough sounds, or measuring lung function using exhalation sounds made during spirometry), requiring less data and less compute than having to fully train a model without the embeddings or the pretrained model."

https://developers.google.com/health-ai-developer-foundations/hear

```python
import concurrent.futures
import random

import google.auth
import google.auth.transport.requests
```

```python
#https://github.com/Google-Health/google-health/blob/master/health_acoustic_representations/hear_demo.ipynb

#We don't have this json file

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/path/to/your/credentials/json/file'
```

```python
!pip install api_utils
```

```python
#https://github.com/Google-Health/google-health/blob/master/health_acoustic_representations/hear_demo.ipynb

# Environment variable `GOOGLE_APPLICATION_CREDENTIALS` must be set for these
# imports to work.
import api_utils
```

## Online predictions - With raw audio

```python
#https://github.com/Google-Health/google-health/blob/master/health_acoustic_representations/hear_demo.ipynb

raw_audio = np.array([[random.random() for _ in range(32000)] for _ in range(4)])
embeddings = api_utils.make_prediction(
  endpoint_path=api_utils.RAW_AUDIO_ENDPOINT_PATH,
  instances=raw_audio,
)
```

## If you have a lot of queries to run

Example with the raw-audio endpoint (202) using ThreadPoolExecutor.

```python
#https://github.com/Google-Health/google-health/blob/master/health_acoustic_representations/hear_demo.ipynb

# 1000 batches of 4 clips. This is the format expected for the raw audio endpoint
instances = np.random.uniform(size=(1000, 4, 32000))  # update with your data

responses = {}

with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
  futures_to_batch_idx = {
    executor.submit(
        api_utils.make_prediction_with_exponential_backoff,
        api_utils.RAW_AUDIO_ENDPOINT_PATH,
        instance
    ): batch_idx
    for batch_idx, instance in enumerate(instances)
  }

  for future in concurrent.futures.as_completed(futures_to_batch_idx):
    batch_idx = futures_to_batch_idx[future]
    try:
      responses[batch_idx] = future.result()
    except Exception as e:
      print("An error occurred:", e)
```

## Patient communication

For the record, patient communication is one of the most difficult process in healthcare. Mostly, nowadays, when professionals spend many time looking to their computers instead of dedicating time to their patients. That's one of the majors patients complaints.

Therefore, it would be helpful if professionals improve their communication skills. This would provide better diagnosis, prognosis and correctness to choose treatments.

Communication improves with practice and experience. These communication involves all the healthcare team. Additionally, it's an effort that professionals should be engaged, no matter what Machine Learning model they adopt in their decisions.

![](https://www.worksure.org/wp-content/uploads/2024/07/Blog-153.png)

## After 1h:56m failing installing packages, my only option is to Back Off

module 'api_utils' has no attribute 'make_prediction_with_exponential_**backoff'**

#Acknowledgements:

https://github.com/Google-Health/google-health/blob/master/health_acoustic_representations/hear_demo.ipynb