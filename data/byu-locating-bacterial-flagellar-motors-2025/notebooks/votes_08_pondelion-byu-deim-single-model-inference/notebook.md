# [BYU] DEIM single model inference

- **Author:** pln
- **Votes:** 342
- **Ref:** pondelion/byu-deim-single-model-inference
- **URL:** https://www.kaggle.com/code/pondelion/byu-deim-single-model-inference
- **Last run:** 2025-04-25 02:17:48.367000

---

# DEIM Single Model Inference Notebook

- Training Data  
only official image data with num_motors>0 used (no external data, no negative sampling).  
75% training, 25% validation  
- Image Size  
(384, 384, 3) (both training and inference)  
- Model weight and DEIM code (including training config) are not public.

```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.798
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.958
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.927
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.798
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.785
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.854
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.883
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.883
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets=100 ] = 1.000
 Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.971
```

```python
!pip install -q /kaggle/input/byu-private-dataset/faster_coco_eval-1.6.5-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
!pip install -q /kaggle/input/byu-private-dataset/calflops-0.3.2-py3-none-any.whl
```

```python
from glob import glob
import sys
import os
from typing import Literal

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
import numpy as np
from PIL import Image, ImageDraw
import pandas as pd
import cv2 
from fastprogress import progress_bar as pb
from tqdm import tqdm
from scipy.spatial import distance
from scipy.optimize import linear_sum_assignment
import networkx as nx
tqdm.pandas()

sys.path.append('/kaggle/input/byu-private-dataset/BYU_DEIM_code_exp101_2/DEIM')
from engine.core import YAMLConfig
```

## 0. Config

```python
device = 'cuda' if torch.cuda.is_available() else 'cpu'
DEIM_CONFIG_FILEPATH = '/kaggle/input/byu-private-dataset/BYU_DEIM_code_exp101_2/DEIM/configs/deim_exp101_2.yml'
DEIM_MODEL_FILEPATH = '/kaggle/input/byu-private-dataset/BYU_DEIM_exp101_2_best_stg1.pth'
IMAGE_SIZE = (384, 384)
SCORE_TH_PRE = SCORE_TH_AGG = 0.825
GROUP_DIST_TH = 20.0
MIN_DET_PER_GROUP = 1
# AGG_METHOD = 'score_highest'
AGG_METHOD = 'score_weighted_mean'
```

## 1. Data Preparation

```python
BASE_IMAGE_DIR = "/kaggle/input/byu-locating-bacterial-flagellar-motors-2025/"
TEST_IMAGE_DIR = os.path.join(BASE_IMAGE_DIR, "test")
test_tomo_dir_list = glob(f'{TEST_IMAGE_DIR}/*')
test_tomo_id_list = [d.split('/')[-1] for d in test_tomo_dir_list]
```

```python
def load_images(tomo_id, train_or_test='test', resize_size=IMAGE_SIZE, loader='torchvision'):
    assert loader in ['pil', 'torchvision']
    image_dir = f'{BASE_IMAGE_DIR}/{train_or_test}/{tomo_id}'
    image_files = sorted(glob(f'{image_dir}/*.*'))
    df_image_files = pd.DataFrame({'filepath': image_files})
    df_image_files['no'] = df_image_files['filepath'].map(lambda x: int(x.split('_')[-1].split('.')[0]))
    df_image_files = df_image_files.sort_values(by='no', ascending=True)
    # None : pil/torchvision resize results in slightly different values.
    if loader == 'pil':
        images = [Image.open(f).convert('L') for f in df_image_files['filepath']]
        org_image_size = images[0].size  # (w, h)
        if resize_size is not None:
            images = [image.resize(resize_size) for image in images]
        images = np.stack([np.asarray(image) for image in images])  # (n_frames, h, w)
    elif loader == 'torchvision':
        trainsforms = T.Resize(resize_size) if resize_size is not None else T.Compose([])
        images = [torchvision.io.read_image(f) for f in df_image_files['filepath']]
        org_image_size = (images[0].shape[2], images[0].shape[1])  # (w, h)
        images = [trainsforms(image) for image in images]
        images = torch.concatenate(images, dim=0)  # (n_frames, h, w)
        images = images.numpy()
    return images, df_image_files, org_image_size
```

```python
# %%time
# images, df_image_files, org_image_size = load_images(tomo_id='tomo_003acc', loader='torchvision', resize_size=IMAGE_SIZE)
```

```python
# %%time
# images2, df_image_files2, org_image_size2 = load_images(tomo_id='tomo_003acc', loader='pil', resize_size=IMAGE_SIZE)
```

## 2. Prepare DEIM Model

```python
class Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.model = cfg.model.deploy()
        self.postprocessor = cfg.postprocessor.deploy()

    def forward(self, images, orig_target_sizes):
        outputs = self.model(images)
        outputs = self.postprocessor(outputs, orig_target_sizes)
        return outputs


def prepare_deim_model(cfg_filepath: str, weight_filepath: str, device=device):
    cfg = YAMLConfig(cfg_filepath, resume=weight_filepath)
    checkpoint = torch.load(weight_filepath, map_location=device)
    if 'ema' in checkpoint:
        state = checkpoint['ema']['module']
    else:
        state = checkpoint['model']
    # Load train mode state and convert to deploy mode
    cfg.model.load_state_dict(state)
    model = Model(cfg).to(device)
    return model.eval()
```

```python
det_model = prepare_deim_model(
    cfg_filepath=DEIM_CONFIG_FILEPATH,
    weight_filepath=DEIM_MODEL_FILEPATH,
)
```

## 3. Run Detection

```python
def rolling_mean_image(image: torch.Tensor, dim: int, window: int) -> torch.Tensor:
    assert window % 2 == 1, "Window size must be odd"
    n_dim = image.ndim

    if dim != (n_dim - 1):
        image = image.transpose(n_dim - 1, dim)  # move target dim to last

    n_padding = (window - 1) // 2
    pad_image_head = image[..., [0]].repeat([1] * (n_dim - 1) + [n_padding]).to(image)
    pad_image_tail = image[..., [-1]].repeat([1] * (n_dim - 1) + [n_padding]).to(image)
    image_padded = torch.cat([pad_image_head, image, pad_image_tail], dim=-1)

    image_rolling_mean = image_padded.unfold(dimension=-1, size=window, step=1).mean(dim=-1)

    if dim != (n_dim - 1):
        image_rolling_mean = image_rolling_mean.transpose(n_dim - 1, dim)  # revert to original shape

    assert image.shape == image_rolling_mean.shape
    return image_rolling_mean
```

```python
@torch.inference_mode()
def inference_np_batch(model, np_image: np.ndarray, resize_size=IMAGE_SIZE):
    if np_image.dtype == np.uint8:
        np_image = np_image.astype(float) / 255
    tensor_image = torch.tensor(np_image).permute(0, 3, 1, 2).float()  # (bs, h, w, ch) => (bs, ch, h, w)

    transforms = T.Compose([
        T.Resize(IMAGE_SIZE),
    ])
    im_data = transforms(tensor_image).to(device)
    bs, ch, h, w = im_data.shape
    orig_size = torch.tensor([[w, h]]).to(device)

    output = model(im_data, orig_size)
    labels, boxes, scores = output
    return labels, boxes, scores


def filter_detection(labels, boxes, scores, thrh=0.4):
    n_query1, =  labels.shape
    n_query2, bbox_dim =  boxes.shape
    n_query3, =  scores.shape
    assert n_query1 == n_query2 == n_query3, (n_query1, n_query2, n_query3)
    assert bbox_dim == 4
    lab = labels[scores > thrh]
    box = boxes[scores > thrh]
    scrs = scores[scores > thrh]
    return lab, box, scrs


@torch.inference_mode()
def inference_tomo(model, tomo_id: str, batch_size: int = 4, th: float = 0.4) -> pd.DataFrame:
    # 1. Load images for target tomo_id
    images, df_image_files, org_image_size = load_images(tomo_id=tomo_id, loader='pil', resize_size=IMAGE_SIZE)
    images = images.transpose(1, 2, 0)  # (n_frames, h, w) => (h, w, n_frames)
    z_max = images.shape[-1] - 1
    w_org, h_org = org_image_size

    experimental = False
    if experimental:
        # calculate rolling mean along z-axis
        images = rolling_mean_image(torch.tensor(images).float(), dim=2, window=21).numpy() / 255

    # 2. Run detection on sliced 3ch images along z axis.
    image_sliced_list = []
    df_detection_list = []
    z_center_list = list(range(1, z_max-1))
    for z in pb(z_center_list):
        image_sliced = images[:, :, z-1:z+1+1]  # (h, w, 3)
        image_sliced_list.append(image_sliced)
        if (len(image_sliced_list) >= batch_size) or (z == z_center_list[-1]):
            image_sliced_batch = np.stack(image_sliced_list)  # (bs, h, w, 3)
            labels, boxes, scores = inference_np_batch(model, image_sliced_batch)
            for i in range(labels.shape[0]):
                lab, box, scrs = filter_detection(labels[i], boxes[i], scores[i], th)
                if len(lab) > 0:
                    df_det = pd.DataFrame(data=box.cpu().numpy(), columns=['x1', 'y1', 'x2', 'y2'])
                    df_det['z'] = z
                    df_det['x_384'] = 0.5 * (df_det['x1'] + df_det['x2'])
                    df_det['y_384'] = 0.5 * (df_det['y1'] + df_det['y2'])
                    df_det['x_normed'] = df_det['x_384'] / 384
                    df_det['y_normed'] = df_det['y_384'] / 384
                    df_det['x'] = w_org * df_det['x_normed']
                    df_det['y'] = h_org * df_det['y_normed']
                    df_det['label'] = lab.cpu().tolist()
                    df_det['score'] = scrs.cpu().tolist()
                    df_det['tomo_id'] = tomo_id
                    df_detection_list.append(df_det)
            image_sliced_list = []
    return pd.concat(df_detection_list) if len(df_detection_list) > 0 else pd.DataFrame([])
```

```python
# Run detection for each tomo_id
df_det_list = []

for tomo_id in pb(test_tomo_id_list):
    df_det_list.append(inference_tomo(det_model, tomo_id, th=SCORE_TH_PRE))

df_det_all = pd.concat(df_det_list)
df_det_all = df_det_all.reset_index(drop=True)
```

```python
df_det_all
```

## 4. Aggregate Detections

```python
def aggregate_detection(
    df_det: pd.DataFrame,
    score_th: float,
    group_dist_th: float = 5.0,
    min_det_per_group: int = None,
    agg_method: Literal['score_weighted_mean', 'score_highest'] = 'score_weighted_mean',
) -> pd.DataFrame:
    assert agg_method in ['score_weighted_mean', 'score_highest']
    # pre-filter by score threshold
    df_det = df_det[df_det['score'] >= score_th]
    df_agg_det_tomo_list = []
    for tomo_id, df_det_tomo in df_det.groupby('tomo_id'):
        # calculate euclidean distance matrix (in voxel space) between each detections in this tomo_id
        dist_mat = distance.cdist(df_det_tomo[['x', 'y', 'z']], df_det_tomo[['x', 'y', 'z']], metric='euclidean')
        # calculate adjacency matrix based on distance matrix and threshold distance
        adj_mat = (dist_mat <= group_dist_th).astype(int)
        np.fill_diagonal(adj_mat, 0)
        # group detections into connected graphs based on adjacency matrix
        G = nx.from_numpy_array(adj_mat)
        connected_components = list(nx.connected_components(G))
        agg_det_dict_list = []
        # Aggregate detections in each connected groups
        for group_idx_set in connected_components:
            df_det_grp = df_det_tomo.iloc[list(group_idx_set)]  # detections belonging to this group
            if agg_method == 'score_weighted_mean':
                z = (df_det_grp['z'] * df_det_grp['score']).sum() / df_det_grp['score'].sum()  # score weighted mean
                y = (df_det_grp['y'] * df_det_grp['score']).sum() / df_det_grp['score'].sum()  # score weighted mean
                x = (df_det_grp['x'] * df_det_grp['score']).sum() / df_det_grp['score'].sum()  # score weighted mean
                score_mean = df_det_grp['score'].mean()
            elif agg_method == 'score_highest':
                z = df_det_grp.sort_values(by='score', ascending=False).iloc[0].z
                y = df_det_grp.sort_values(by='score', ascending=False).iloc[0].y
                x = df_det_grp.sort_values(by='score', ascending=False).iloc[0].x
                score_mean = df_det_grp.sort_values(by='score', ascending=False).iloc[0].score
            else:
                raise ValueError(agg_method)
            agg_det = {
                'tomo_id': tomo_id,
                'x': x,
                'y': y,
                'z': z,
                'score_mean': score_mean,
                'group_det_count': len(group_idx_set),  # detection count in this group
            }
            agg_det_dict_list.append(agg_det)
        df_agg_det_tomo = pd.DataFrame(agg_det_dict_list)
        if min_det_per_group is not None:
            # delete the detections belonging to the groups that has detection count less than min_det_per_group
            df_agg_det_tomo = df_agg_det_tomo[df_agg_det_tomo['group_det_count'] >= min_det_per_group]
        # select highest (group_det_count, score_mean) group's aggregated detection as final detection for this tomo_id
        if agg_method == 'score_weighted_mean':
            order_by = ['group_det_count', 'score_mean']
        elif agg_method == 'score_highest':
            order_by = ['score_mean', 'group_det_count']
        else:
            raise ValueError(agg_method)
        df_agg_det_tomo = df_agg_det_tomo.sort_values(by=order_by, ascending=False).iloc[:1]
        df_agg_det_tomo_list.append(df_agg_det_tomo)
    return pd.concat(df_agg_det_tomo_list)
```

```python
df_det_agg = aggregate_detection(
    df_det_all,
    score_th=SCORE_TH_AGG,
    group_dist_th=GROUP_DIST_TH,
    min_det_per_group=MIN_DET_PER_GROUP,
    agg_method=AGG_METHOD,
)
assert not df_det_agg['tomo_id'].duplicated().any()
```

```python
df_det_agg
```

## 5. Submit

```python
# no motor detected tomo_id list 
no_motor_tomo_id_list = list(set(test_tomo_id_list) - set(df_det_agg['tomo_id']))
len(no_motor_tomo_id_list)
```

```python
# no motor detected predictions
df_det_no_motor = pd.DataFrame({
    'tomo_id': no_motor_tomo_id_list,
    'Motor axis 0': [-1] * len(no_motor_tomo_id_list),
    'Motor axis 1': [-1] * len(no_motor_tomo_id_list),
    'Motor axis 2': [-1] * len(no_motor_tomo_id_list),
})
# motor detected predictions
df_det_agg = df_det_agg.rename(
    columns={'z': 'Motor axis 0', 'y': 'Motor axis 1', 'x': 'Motor axis 2'}
)[['tomo_id', 'Motor axis 0', 'Motor axis 1', 'Motor axis 2']]
```

```python
display(df_det_no_motor)
display(df_det_agg)
```

```python
df_submission = pd.concat([df_det_agg, df_det_no_motor])
assert set(df_submission['tomo_id']) == set(test_tomo_id_list)
```

```python
df_submission.to_csv('submission.csv', index=False)
```