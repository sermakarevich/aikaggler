# [EDA] 📸 IMC - 📊 &📍Locations

- **Author:** Mehdi Sharifi
- **Votes:** 117
- **Ref:** sharifi76/eda-imc-locations
- **URL:** https://www.kaggle.com/code/sharifi76/eda-imc-locations
- **Last run:** 2025-04-01 22:33:34.387000

---

# <center style="font-family: consolas; font-size: 32px; font-weight: bold;"> 📸 Image Matching Challenge - 📊 Exploratory Data Analysis</center>
<p><center style="color:#949494; font-family: consolas; font-size: 20px;">Reconstruct 3D scenes from 2D images over six different domains</center></p>

***

The objectives of this competition are to:
- Construct precise 3D maps using sets of images in diverse scenarios and environments by developing **a model to generate accurate spatial representations, regardless of the source domain** 
    - The process of reconstructing a 3D model of an environment from a collection of images is called **Structure from Motion (SfM)**. 
    - These images are often captured by trained operators or with additional sensor data. 
    - This ensures homogeneous, high-quality data. 
- Explore various image sources in more realistic and applicable scenarios: images can be taken from drones 🤖, amidst dense forests 🌲, during nighttime 🌙
    - It’s much more difficult to build 3D models from assorted images, the real-world examples that the organizers put together for this competition.
 

For this, the organizers have designated 6 categories of images with distinct challenges:
- 🏛️ **Phototourism and historical preservation**: different viewpoints, sensor types, time of day/year, and occlusions. Ancient historical sites add a unique set of challenges

- ☀️ **Night vs day and temporal changes**: combination of day and night photographs, including poor lighting, or photographs taken months or years apart, in different weather

- ✈️ **Aerial and mixed aerial-ground**: images from drones, featuring arbitrary in-plane rotations, matched against similar images and also images taken from the ground

- ♻️ **Repeated structures**: symmetrical objects require details to disambiguate perspective

- 🌲 **Natural environments**: highly non-regular structures such as trees and foliage

- 🪞 **Transparencies and reflections**: objects like glassware are lacking in texture and create reflections and specularities which pose a different set of problems


Inspired by [this notebook](https://www.kaggle.com/code/asarvazyan/eda-imc-interact-w-3d-plots-locations) from previous year's edition of the competition.

<a id="0"></a>
# Install & Import dependencies

```python
!pip install -q mediapy
```

```python
%cd /kaggle/working/
!rm -rf /kaggle/working/Hierarchical-Localization
!git clone --quiet --recursive https://github.com/cvg/Hierarchical-Localization/
%cd /kaggle/working/Hierarchical-Localization
!pip install -e .

from hloc import extract_features, match_features, reconstruction, visualization, pairs_from_exhaustive
from hloc.visualization import plot_images, read_image
from hloc.utils import viz_3d

%cd /kaggle/working/
```

```python
from pathlib import Path

import cv2
import mediapy
import pandas as pd
import plotly.express as px
import pycolmap
import os
import random
from PIL import Image
import matplotlib.pyplot as plt
```

<a id="1"></a>
# Dataset Overview

- `[train/test]/*/images`: A batch of images all taken near the same location. Some of training datasets may also contain a folder named images_full with additional images. The published test folder comprises a subset of the church scene from train and is provided solely for example purposes. The training data usually has a sequential capture ordering and significant image-to-image content overlap while the test set has limited image-to-image overlap and the image ordering is randomized.

- `train/*/LICENSE.txt`: The license for this dataset.

- `train/train_labels.csv`: A list of images in these datasets, with ground truths.

### 1️⃣ Lets inspect `train/train_labels.csv`

- `dataset`: The unique identifier for the dataset.
- `scene`: The unique identifier for the scene.
- `image_path`: The image filename, including the path.
- `rotation_matrix`: The first target column. A 3x3 matrix, flattened into a vector in row-major convention, with values separated by `;`.
- `translation_vector`: The second target column. A 3-D dimensional vector, with values separated by ;.

```python
train_labels = pd.read_csv("/kaggle/input/image-matching-challenge-2025/train_labels.csv")
train_labels
```

### 2️⃣ What is the relationship between datasets and scenes?

```python
train_labels.groupby("dataset")["scene"].nunique()
```

### 3️⃣ What is the distribution of the datasets?

```python
dataset_counts = train_labels["dataset"].value_counts()

plt.figure(figsize=(12, 7))
dataset_counts.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Distribution of Images Across Datasets', fontsize=16, fontweight='bold')
plt.xlabel('Dataset', fontsize=14)
plt.ylabel('Number of Images', fontsize=14)
plt.xticks(rotation=60, ha='right', fontsize=12)  
plt.yticks(fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
```

<a id="2"></a>
# Exploring each dataset

```python
def plot_random_images(scene_name, base_path="/kaggle/input/image-matching-challenge-2025/train"):
    scene_path = os.path.join(base_path, scene_name)
    image_filenames = [f for f in os.listdir(scene_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
    random_images = random.sample(image_filenames, min(5, len(image_filenames)))
    fig, axes = plt.subplots(1, len(random_images), figsize=(15, 5))
    if len(random_images) == 1:
        axes = [axes]

    for ax, img_filename in zip(axes, random_images):
        img_path = os.path.join(scene_path, img_filename)
        img = Image.open(img_path)
        ax.imshow(img)
        ax.set_title(img_filename, fontsize=10)
        ax.axis('off')
    plt.tight_layout()
    plt.show()
```

<a id="stairs"></a>
## 1️⃣ stairs

```python
plot_random_images("stairs")
```

<a id="pt_stpeters_stpauls"></a>
## 2️⃣ pt_stpeters_stpauls

```python
plot_random_images("pt_stpeters_stpauls")
```

<a id="pt_sacrecoeur_trevi_tajmahal"></a>
## 3️⃣ pt_sacrecoeur_trevi_tajmahal

```python
plot_random_images("pt_sacrecoeur_trevi_tajmahal")
```

<a id="pt_piazzasanmarco_grandplace"></a>
## 4️⃣ pt_piazzasanmarco_grandplace

```python
plot_random_images("pt_piazzasanmarco_grandplace")
```

<a id="pt_brandenburg_british_buckingham"></a>
## 5️⃣ pt_brandenburg_british_buckingham

```python
plot_random_images("pt_brandenburg_british_buckingham")
```

<a id="imc2024_lizard_pond"></a>
## 6️⃣ imc2024_lizard_pond

```python
plot_random_images("imc2024_lizard_pond")
```

<a id="imc2024_dioscuri_baalshamin"></a>
## 7️⃣ imc2024_dioscuri_baalshamin

```python
plot_random_images("imc2024_dioscuri_baalshamin")
```

<a id="imc2023_theather_imc2024_church"></a>
## 8️⃣ imc2023_theather_imc2024_church

```python
plot_random_images("imc2023_theather_imc2024_church")
```

<a id="imc2023_heritage"></a>
## 9️⃣ imc2023_heritage

```python
plot_random_images("imc2023_heritage")
```

<a id="imc2023_haiper"></a>
## 1️⃣0️⃣ imc2023_haiper

```python
plot_random_images("imc2023_haiper")
```

<a id="fbk_vineyard"></a>
## 1️⃣1️⃣ fbk_vineyard

```python
plot_random_images("fbk_vineyard")
```

<a id="amy_gardens"></a>
## 1️⃣2️⃣ amy_gardens

```python
plot_random_images("amy_gardens")
```

<a id="ETs"></a>
## 1️⃣3️⃣ ETs

```python
plot_random_images("ETs")
```

# Work in progress!

❤️ Thank you for taking the time to read through my notebook. I hope you found it interesting and informative. If you have any feedback or suggestions for improvement, please don't hesitate to let me know in the comments.