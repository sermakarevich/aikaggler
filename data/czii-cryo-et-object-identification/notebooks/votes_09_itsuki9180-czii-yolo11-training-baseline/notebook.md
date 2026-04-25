# CZII YOLO11 Training Baseline

- **Author:** ITK8191
- **Votes:** 243
- **Ref:** itsuki9180/czii-yolo11-training-baseline
- **URL:** https://www.kaggle.com/code/itsuki9180/czii-yolo11-training-baseline
- **Last run:** 2024-12-07 09:13:06.107000

---

# CZII YOLO11 Training Baseline
 We created a training set adapted to YOLO from [the dataset baseline](https://www.kaggle.com/code/itsuki9180/czii-making-datasets-for-yolo).

In this notebook, we actually use it to train YOLO so that it can infer the xy coordinates of particles through 2D object detection.

# Install and Import modules

```python
!tar xfvz /kaggle/input/ultralytics-for-offline-install/archive.tar.gz
!pip install --no-index --find-links=./packages ultralytics
!rm -rf ./packages
```

```python
from tqdm import tqdm
import glob, os
from ultralytics import YOLO
```

# Prepare to train and instance YOLOmodel

```python
# Load a pretrained model
model = YOLO("/kaggle/input/yolo11/pytorch/default/1/yolo11l.pt")  # load a pretrained model (recommended for training)
```

# Let's train YOLO!

```python
# Train the model
_ = model.train(
    data="/kaggle/input/czii-yolo-datasets/czii_conf.yaml",
    epochs=100,
    warmup_epochs=10,
    optimizer='AdamW',
    cos_lr=True,
    lr0=3e-4,
    lrf=0.03,
    imgsz=640,
    device="0,1",
    weight_decay=0.005,
    batch=32,
    scale=0,
    flipud=0.5,
    fliplr=0.5,
    degrees=45,
    shear=5,
    mixup=0.2,
    copy_paste=0.25,
    seed=8620, # (｡•◡•｡)
)
```

```python
model = YOLO("/kaggle/working/runs/detect/train/weights/best.pt")
metrics = model.val(data="/kaggle/input/czii-yolo-datasets/czii_conf.yaml", imgsz=640, batch=16, conf=0.25, iou=0.6, device="0", save_json=True)  # no arguments needed, dataset and settings remembered
print(metrics.box.map)  # map50-95
print(metrics.box.map50)  # map50
print(metrics.box.map75)  # map75
print(metrics.box.maps)
```

# Prediction example

```python
results = model("/kaggle/input/czii-yolo-datasets/datasets/czii_det2d/images/val/TS_5_4_920.png")
results[0].show()
```

# Continue to [Submission Baseline...](https://www.kaggle.com/code/itsuki9180/czii-yolo11-submission-baseline)