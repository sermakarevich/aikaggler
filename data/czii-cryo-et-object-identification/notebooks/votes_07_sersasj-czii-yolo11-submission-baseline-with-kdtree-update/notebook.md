# CZII YOLO11 Submission Baseline with KDTree Update

- **Author:** Sergio Alvarez
- **Votes:** 268
- **Ref:** sersasj/czii-yolo11-submission-baseline-with-kdtree-update
- **URL:** https://www.kaggle.com/code/sersasj/czii-yolo11-submission-baseline-with-kdtree-update
- **Last run:** 2025-01-11 21:14:43.110000

---

# CZII YOLO11 Submission Baseline with KDTree

When @ITK8191 shared his [great notebook](https://www.kaggle.com/code/itsuki9180/czii-yolo11-submission-baseline) I got inspired by it. My initial idea was to train a 2dUNET and use his approach to aggregate the results. However, I thought this wouldn't be feasible, as the YOLO notebook at that time required 10 hours to run. Because of this I tried to modify it by using KDTree. I managed to run a 2dUNET within the time limit but didn't got great results. I decided to share my results with the yolo model.

Later on @min fuka shared his [multi processing notebook](https://www.kaggle.com/code/minfuka/czii-yolo11-submission-baseline-speed-up-ver) and I added his ideas.


Time with KDTree was reduced to ~6500 seconds and with multiprocessing was reduced to ~4500 seconds.


If you think this notebook is good, please upvote the [@ITK8191 original notebook](https://www.kaggle.com/code/itsuki9180/czii-yolo11-submission-baseline), [@min fuka multi processing notebook](https://www.kaggle.com/code/minfuka/czii-yolo11-submission-baseline-speed-up-ver) (and this note)!

## Update
This is the furthest I got using YOLO. I've trained with the additional data, weights for the model can be find here [CZII YOLO L trained with synthetic data
](https://www.kaggle.com/datasets/sersasj/czii-yolo-l-trained-with-synthetic-data).  
The notebook to create with additional data is here [YOLO dataset with syntethic data
](https://www.kaggle.com/code/sersasj/czii-making-datasets-for-yolo-synthetic-data).  
Model was trained with TS_5_4, TS_69_2 TS_6_4 TS_6_6 as validation.

Parameters (z_distance, zy_distance, first_conf, conf_coef) were found with OPTUNA optimization + various submissions for evaluating. 
Perfomance on validation was 0.73 

```
Score for TS_5_4: 0.658783812957661,{'apo-ferritin': {'total_tp': 42, 'total_fp': 20, 'total_fn': 2, 'fbeta': 0.9321148825065274}, 'beta-galactosidase': {'total_tp': 5, 'total_fp': 27, 'total_fn': 7, 'fbeta': 0.3794642857142858}, 'ribosome': {'total_tp': 20, 'total_fp': 29, 'total_fn': 10, 'fbeta': 0.6427221172022684}, 'thyroglobulin': {'total_tp': 23, 'total_fp': 104, 'total_fn': 7, 'fbeta': 0.6441515650741352}, 'virus-like-particle': {'total_tp': 11, 'total_fp': 2, 'total_fn': 0, 'fbeta': 0.9894179894179894}} 

Score for TS_69_2: 0.8191956150699464,{'apo-ferritin': {'total_tp': 35, 'total_fp': 25, 'total_fn': 0, 'fbeta': 0.9596774193548387}, 'beta-galactosidase': {'total_tp': 13, 'total_fp': 46, 'total_fn': 3, 'fbeta': 0.7015873015873016}, 'ribosome': {'total_tp': 35, 'total_fp': 15, 'total_fn': 2, 'fbeta': 0.926791277258567}, 'thyroglobulin': {'total_tp': 28, 'total_fp': 84, 'total_fn': 6, 'fbeta': 0.7256097560975611}, 'virus-like-particle': {'total_tp': 9, 'total_fp': 1, 'total_fn': 0, 'fbeta': 0.9935064935064936}}

Score for TS_6_4: 0.685180923434018,{'apo-ferritin': {'total_tp': 45, 'total_fp': 34, 'total_fn': 12, 'fbeta': 0.7719475277497477}, 'beta-galactosidase': {'total_tp': 7, 'total_fp': 29, 'total_fn': 5, 'fbeta': 0.5219298245614036}, 'ribosome': {'total_tp': 54, 'total_fp': 59, 'total_fn': 12, 'fbeta': 0.7852865697177076}, 'thyroglobulin': {'total_tp': 24, 'total_fp': 77, 'total_fn': 6, 'fbeta': 0.70223752151463}, 'virus-like-particle': {'total_tp': 8, 'total_fp': 4, 'total_fn': 2, 'fbeta': 0.7906976744186046}}

Score for TS_6_6: 0.7575532250952666,{'apo-ferritin': {'total_tp': 37, 'total_fp': 39, 'total_fn': 2, 'fbeta': 0.8985714285714286}, 'beta-galactosidase': {'total_tp': 8, 'total_fp': 43, 'total_fn': 3, 'fbeta': 0.5991189427312775}, 'ribosome': {'total_tp': 17, 'total_fp': 11, 'total_fn': 6, 'fbeta': 0.7297979797979798}, 'thyroglobulin': {'total_tp': 31, 'total_fp': 120, 'total_fn': 4, 'fbeta': 0.7412095639943742}, 'virus-like-particle': {'total_tp': 19, 'total_fp': 2, 'total_fn': 0, 'fbeta': 0.9938461538461538}}
```


# If you have others ideas or suggestions that might improve this approach feel free to share with the community!

```python
!tar xfvz /kaggle/input/ultralytics-for-offline-install/archive.tar.gz
!pip install --no-index --find-links=./packages ultralytics
!rm -rf ./packages

!cp -r '/kaggle/input/hengck-czii-cryo-et-01/wheel_file' '/kaggle/working/'
!pip install /kaggle/working/wheel_file/asciitree-0.3.3/asciitree-0.3.3
!pip install --no-index --find-links=/kaggle/working/wheel_file zarr
```

```python
import os
import glob
import time
import sys
import warnings
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import torch
from tqdm import tqdm
from ultralytics import YOLO
import zarr
from scipy.spatial import cKDTree
from collections import defaultdict
```

```python
model_path = '/kaggle/input/czii-yolo-l-trained-with-synthetic-data/best_synthetic.pt'
model = YOLO(model_path)
```

```python
runs_path = '/kaggle/input/czii-cryo-et-object-identification/test/static/ExperimentRuns/*'
runs = sorted(glob.glob(runs_path))
runs = [os.path.basename(run) for run in runs]
sp = len(runs)//2
runs1 = runs[:sp]
runs1[:5]

#add by @minfuka
runs2 = runs[sp:]
runs2[:5]

#add by @minfuka
assert torch.cuda.device_count() == 2
```

```python
particle_names = [
    'apo-ferritin',
    'beta-amylase',
    'beta-galactosidase',
    'ribosome',
    'thyroglobulin',
    'virus-like-particle'
]

particle_to_index = {
    'apo-ferritin': 0,
    'beta-amylase': 1,
    'beta-galactosidase': 2,
    'ribosome': 3,
    'thyroglobulin': 4,
    'virus-like-particle': 5
}

index_to_particle = {index: name for name, index in particle_to_index.items()}

particle_radius = {
    'apo-ferritin': 60,
    'beta-amylase': 65,
    'beta-galactosidase': 90,
    'ribosome': 150,
    'thyroglobulin': 130,
    'virus-like-particle': 135,
}
```

```python
# add by @sesasj
class UnionFind:
    def __init__(self, size):
        self.parent = np.arange(size)
        self.rank = np.zeros(size, dtype=int)

    def find(self, u):
        if self.parent[u] != u:
            self.parent[u] = self.find(self.parent[u])  
        return self.parent[u]

    def union(self, u, v):
        u_root = self.find(u)
        v_root = self.find(v)
        if u_root == v_root:
            return
            
        if self.rank[u_root] < self.rank[v_root]:
            self.parent[u_root] = v_root
        else:
            self.parent[v_root] = u_root
            if self.rank[u_root] == self.rank[v_root]:
                self.rank[u_root] += 1

class PredictionAggregator:
    def __init__(self, first_conf=0.2, conf_coef=0.75):
        self.first_conf = first_conf
        self.conf_coef = conf_coef
        self.particle_confs = np.array([0.5, 0.0, 0.2, 0.5, 0.2, 0.5])
        
    def convert_to_8bit(self, volume):
        lower, upper = np.percentile(volume, (0.5, 99.5))
        clipped = np.clip(volume, lower, upper)
        scaled = ((clipped - lower) / (upper - lower + 1e-12) * 255).astype(np.uint8)
        return scaled

    def make_predictions(self, run_id, model, device_no):
        volume_path = f'/kaggle/input/czii-cryo-et-object-identification/test/static/ExperimentRuns/{run_id}/VoxelSpacing10.000/denoised.zarr'
        volume = zarr.open(volume_path, mode='r')[0]
        volume_8bit = self.convert_to_8bit(volume)
        num_slices = volume_8bit.shape[0]

        detections = {
            'particle_type': [],
            'confidence': [],
            'x': [],
            'y': [],
            'z': []
        }

        for slice_idx in range(num_slices):
            
            img = volume_8bit[slice_idx]
            input_image = cv2.resize(np.stack([img]*3, axis=-1), (640, 640))

            results = model.predict(
                input_image,
                save=False,
                imgsz=640,
                conf=self.first_conf,
                device=device_no,
                batch=1,
                verbose=False,
            )

            for result in results:
                boxes = result.boxes
                if boxes is None:
                    continue
                cls = boxes.cls.cpu().numpy().astype(int)
                conf = boxes.conf.cpu().numpy()
                xyxy = boxes.xyxy.cpu().numpy()

                xc = ((xyxy[:, 0] + xyxy[:, 2]) / 2.0) * 10 * (63/64) # 63/64 because of the resize
                yc = ((xyxy[:, 1] + xyxy[:, 3]) / 2.0) * 10 * (63/64)
                zc = np.full(xc.shape, slice_idx * 10 + 5)

                particle_types = [index_to_particle[c] for c in cls]

                detections['particle_type'].extend(particle_types)
                detections['confidence'].extend(conf)
                detections['x'].extend(xc)
                detections['y'].extend(yc)
                detections['z'].extend(zc)

        if not detections['particle_type']:
            return pd.DataFrame()  

        particle_types = np.array(detections['particle_type'])
        confidences = np.array(detections['confidence'])
        xs = np.array(detections['x'])
        ys = np.array(detections['y'])
        zs = np.array(detections['z'])

        aggregated_data = []

        for idx, particle in enumerate(particle_names):
            if particle == 'beta-amylase':
                continue 

            mask = (particle_types == particle)
            if not np.any(mask):
                continue  
                
            particle_confidences = confidences[mask]
            particle_xs = xs[mask]
            particle_ys = ys[mask]
            particle_zs = zs[mask]
            # -------------modified by @sersasj ------------------------
            coords = np.vstack((particle_xs, particle_ys, particle_zs)).T

           
            z_distance = 30 # How many slices can you "jump" to aggregate predictions 10 = 1, 20 = 2...
            xy_distance = 20 # xy_tol_p2 in original code by ITK8191
            
            max_distance = math.sqrt(z_distance**2 + xy_distance**2)
            tree = cKDTree(coords)            
            pairs = tree.query_pairs(r=max_distance, p=2)

            
            uf = UnionFind(len(coords))
            
            coords_xy = coords[:, :2]
            coords_z = coords[:, 2]
            for u, v in pairs:
                z_diff = abs(coords_z[u] - coords_z[v])
                if z_diff > z_distance:
                    continue  

                xy_diff = np.linalg.norm(coords_xy[u] - coords_xy[v])
                if xy_diff > xy_distance:
                    continue  

                uf.union(u, v)

            roots = np.array([uf.find(i) for i in range(len(coords))])
            unique_roots, inverse_indices, counts = np.unique(roots, return_inverse=True, return_counts=True)
            conf_sums = np.bincount(inverse_indices, weights=particle_confidences)
            
            aggregated_confidences = conf_sums / (counts ** self.conf_coef)
            cluster_per_particle = [4,1,2,9,4,8] # Update
            valid_clusters = (counts >= cluster_per_particle[idx]) & (aggregated_confidences > self.particle_confs[idx])

            if not np.any(valid_clusters):
                continue  

            cluster_ids = unique_roots[valid_clusters]

            centers_x = np.bincount(inverse_indices, weights=particle_xs) / counts
            centers_y = np.bincount(inverse_indices, weights=particle_ys) / counts
            centers_z = np.bincount(inverse_indices, weights=particle_zs) / counts

            centers_x = centers_x[valid_clusters]
            centers_y = centers_y[valid_clusters]
            centers_z = centers_z[valid_clusters]

            aggregated_df = pd.DataFrame({
                'experiment': [run_id] * len(centers_x),
                'particle_type': [particle] * len(centers_x),
                'x': centers_x,
                'y': centers_y,
                'z': centers_z
            })

            aggregated_data.append(aggregated_df)

        if aggregated_data:
            return pd.concat(aggregated_data, axis=0)
        else:
            return pd.DataFrame()
```

```python
# instance main class
aggregator = PredictionAggregator(first_conf=0.19,  conf_coef=0.34) #Update
aggregated_results = []


#add by @minfuka
from concurrent.futures import ProcessPoolExecutor #add by @minfuka

#add by @minfuka
def inference(runs, model, device_no):
    subs = []
    for r in tqdm(runs, total=len(runs)):
        df = aggregator.make_predictions(r, model, device_no)
        subs.append(df)
    
    return subs


start_time = time.time()

with ProcessPoolExecutor(max_workers=2) as executor:
    results = list(executor.map(inference, (runs1, runs2), (model, model), ("0", "1")))


end_time = time.time()

estimated_total_time = (end_time - start_time) / len(runs) * 500  
print(f'estimated total prediction time for 500 runs: {estimated_total_time:.4f} seconds')
```

```python
#change by @minfuka
submission0 = pd.concat(results[0])
submission1 = pd.concat(results[1])
submission = pd.concat([submission0, submission1]).reset_index(drop=True)
```

```python
submission.insert(0, 'id', range(len(submission)))
submission.to_csv("submission.csv", index=False)
submission.head()
```