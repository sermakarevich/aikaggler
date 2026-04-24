# [Inference] Vesuvius Surface 3D Detection 

- **Author:** Innat
- **Votes:** 568
- **Ref:** ipythonx/inference-vesuvius-surface-3d-detection
- **URL:** https://www.kaggle.com/code/ipythonx/inference-vesuvius-surface-3d-detection
- **Last run:** 2026-02-25 20:25:51.130000

---

# Training Notebooks

- [Vesuvius Surface 3D Detection in Keras-JAX](https://www.kaggle.com/code/ipythonx/vesuvius-surface-3d-detection-in-jax)
- [Vesuvius Surface 3D Detection in PyTorch](https://www.kaggle.com/code/ipythonx/vesuvius-surface-3d-detection-in-pytorch)
- [Vesuvius Surface 3D Detection in PyTorch Lightning](https://www.kaggle.com/code/ipythonx/train-vesuvius-surface-3d-detection-in-lightning)
- [[WIP] Vesuvius Surface 2.5D Detection](https://www.kaggle.com/code/ipythonx/wip-vesuvius-surface-2-5d-detection)

**Note**
1. The inference code below is adapted from the **Keras-JAX** version. The PyTorch and Lightning implementations follow the same workflow. Training was performed on a single Tesla T4 (16 GB VRAM) with extended epochs.
2. Both the training and inference pipelines are implemented using [`medicai`](https://github.com/innat/medic-ai), a **Keras 3** based multi-backend medical ML library designed for 2D and 3D classification and segmentation tasks. However, please note, `medicai` project is still new and actively evolving.

# Inference

```python
from IPython.display import clear_output

var="/kaggle/input/vsdetection-packages-offline-installer-only/whls"
!pip install \
  "$var"/keras_nightly-*.whl \
  "$var"/tifffile-*.whl \
  "$var"/imagecodecs-*.whl \
  "$var"/medicai-*.whl \
  --no-index \
  --find-links "$var"

clear_output()
```

```python
import os
os.environ["KERAS_BACKEND"] = "jax"

import keras
from keras import ops
from medicai.transforms import (
    Compose,
    ScaleIntensityRange,
    NormalizeIntensity
)
from medicai.models import SegFormer, TransUNet
from medicai.utils.inference import SlidingWindowInference

import numpy as np
import pandas as pd
import zipfile
import tifffile
import scipy.ndimage as ndi
from skimage.morphology import remove_small_objects
from matplotlib import pyplot as plt

keras.config.backend(), keras.version()
```

**Dataset**

```python
root_dir = "/kaggle/input/vesuvius-challenge-surface-detection"
test_dir = f"{root_dir}/test_images"
output_dir = "/kaggle/working/submission_masks"
zip_path = "/kaggle/working/submission.zip"
os.makedirs(output_dir, exist_ok=True)
```

```python
test_df = pd.read_csv(f"{root_dir}/test.csv")
test_df.head()
```

**Transformation**

```python
def val_transformation(image):
    data = {"image": image}
    pipeline = Compose([
        NormalizeIntensity(
            keys=["image"], 
            nonzero=True,
            channel_wise=False
        ),
    ])
    result = pipeline(data)
    return result["image"]
```

**Model**

```python
tta=1
num_classes=3
input_shape=(160, 160, 160)
```

```python
def get_model_a():
    # 0.551 (tta+pp)
    model = TransUNet(
        input_shape=(160, 160, 160, 1),
        encoder_name='seresnext50',
        classifier_activation=None, # For tta, do softmax later.
        num_classes=3,
    )
    model.load_weights(
        "/kaggle/input/models/ipythonx/vsd-model/keras/transunet/4/transunet.seresnext50.160px.duel.weights.h5"
    )
    return model

def get_model_b():
    # 0.522 (tta+pp)
    model = SegFormer(
        input_shape=input_shape + (1,),
        encoder_name='mit_b4',
        classifier_activation=None, # For tta, do softmax later.
        num_classes=num_classes,
    )
    model.load_weights(
        "/kaggle/input/models/ipythonx/vsd-model/keras/segformer/1/segformer.mit.b4.weights.h5"
    )
    return model
```

```python
models = [get_model_a(), get_model_b()]
[model.count_params() / 1e6 for model in models]
```

```python
for model in models:
    print(model.instance_describe())
```

**Sliding Window Inference**

```python
swis = [
    SlidingWindowInference(
        model=m,
        num_classes=num_classes,
        roi_size=input_shape,
        sw_batch_size=1,
        overlap=0.5,
        mode="gaussian",
    )
    for m in models
]
```

```python
def load_volume(path):
    vol = tifffile.imread(path)
    vol = vol.astype(np.float32)
    vol = vol[None, ..., None]
    return vol
```

**Test Time Augmentation (TTA)**

```python
def predict_with_tta(inputs, swi):
    logits = []

    # Original
    logits.append(swi(inputs))

    # Flips (spatial only)
    for axis in [1, 2, 3]:
        img_f = np.flip(inputs, axis=axis)
        p = swi(img_f)
        p = np.flip(p, axis=axis)
        logits.append(p)

    # Axial rotations (H, W)
    for k in [1, 2, 3]:
        img_r = np.rot90(inputs, k=k, axes=(2, 3))
        p = swi(img_r)
        p = np.rot90(p, k=-k, axes=(2, 3))
        logits.append(p)

    mean_logits = np.mean(logits, axis=0)
    return mean_logits
```

**Post Processing**

```python
# https://www.kaggle.com/code/choudharymanas/inference-baseline-transunet-lb-0-537
def build_anisotropic_struct(z_radius: int, xy_radius: int):
    z, r = z_radius, xy_radius
    if z == 0 and r == 0:
        return None
    if z == 0 and r > 0:
        size = 2 * r + 1
        struct = np.zeros((1, size, size), dtype=bool)
        cy, cx = r, r
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if dy * dy + dx * dx <= r * r:
                    struct[0, cy + dy, cx + dx] = True
        return struct
    if z > 0 and r == 0:
        struct = np.zeros((2 * z + 1, 1, 1), dtype=bool)
        struct[:, 0, 0] = True
        return struct
    depth = 2 * z + 1
    size = 2 * r + 1
    struct = np.zeros((depth, size, size), dtype=bool)
    cz, cy, cx = z, r, r
    for dz in range(-z, z + 1):
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if dy * dy + dx * dx <= r * r:
                    struct[cz + dz, cy + dy, cx + dx] = True
    return struct

def topo_postprocess(
    probs,
    T_low=0.90,
    T_high=0.90,
    z_radius=1,
    xy_radius=0,
    dust_min_size=100,
):
    # Step 1: 3D Hysteresis
    strong = probs >= T_high
    weak   = probs >= T_low

    if not strong.any():
        return np.zeros_like(probs, dtype=np.uint8)

    struct_hyst = ndi.generate_binary_structure(3, 3)
    mask = ndi.binary_propagation(
        strong, mask=weak, structure=struct_hyst
    )

    if not mask.any():
        return np.zeros_like(probs, dtype=np.uint8)

    # Step 2: 3D Anisotropic Closing
    if z_radius > 0 or xy_radius > 0:
        struct_close = build_anisotropic_struct(z_radius, xy_radius)
        if struct_close is not None:
            mask = ndi.binary_closing(mask, structure=struct_close)

    # Step 3: Dust Removal
    if dust_min_size > 0:
        mask = remove_small_objects(
            mask.astype(bool), min_size=dust_min_size
        )

    return mask.astype(np.uint8)
```

**Weighted Ensembling**

```python
weights = np.array([0.8, 0.2], dtype=np.float32)
def predict_weighted_ensemble(volume):
    logits = []
    for swi in swis:
        logits.append(
            predict_with_tta(volume, swi) if tta else swi(volume) 
        )
    logits = np.stack(logits, axis=0)
    weighted_logits = logits * weights[:, None, None, None, None, None]
    mean_logits = weighted_logits.sum(axis=0) / weights.sum()
    probs = ops.softmax(mean_logits, axis=-1)
    return probs.argmax(axis=-1).astype(np.uint8).squeeze()
```

**Prediction and Zip Submission**

```python
def inference_pipelines(
    volume,
    T_low=0.50,
    T_high=0.90,
    z_radius=1,
    xy_radius=0,
    dust_min_size=100,
):
    probs = predict_weighted_ensemble(volume)
    final = topo_postprocess(
        probs,
        T_low=T_low,
        T_high=T_high,
        z_radius=z_radius,
        xy_radius=xy_radius,
        dust_min_size=dust_min_size,
    )
    return final
```

```python
with zipfile.ZipFile(
    zip_path, "w", compression=zipfile.ZIP_DEFLATED
) as z:
    for image_id in test_df["id"]:
        tif_path = f"{test_dir}/{image_id}.tif"
        
        volume = load_volume(tif_path)
        volume = val_transformation(volume)
        output = inference_pipelines(
            volume, T_low=0.35, T_high=0.85, z_radius=3, xy_radius=2
        ) 
        out_path = f"{output_dir}/{image_id}.tif"
        tifffile.imwrite(out_path, output.astype(np.uint8))

        z.write(out_path, arcname=f"{image_id}.tif")
        os.remove(out_path)

print("Submission ZIP:", zip_path)
```

**Sample View**

```python
def plot_sample(x, y, sample_idx=0, max_slices=16):
    img = np.squeeze(x[sample_idx])  # make (D, H, W)
    mask = np.squeeze(y[sample_idx])  # make (D, H, W)
    D = img.shape[0]

    # Decide which slices to plot
    step = max(1, D // max_slices)
    slices = range(0, D, step)

    n_slices = len(slices)
    fig, axes = plt.subplots(2, n_slices, figsize=(3*n_slices, 6))

    for i, s in enumerate(slices):
        axes[0, i].imshow(img[s], cmap='gray')
        axes[0, i].set_title(f"Slice {s}")
        axes[0, i].axis('off')

        axes[1, i].imshow(mask[s], cmap='gray')
        axes[1, i].set_title(f"Mask {s}")
        axes[1, i].axis('off')

    plt.suptitle(f"Sample {sample_idx}")
    plt.tight_layout()
    plt.show()
```

```python
plot_sample(
    volume.numpy(), output[None], sample_idx=0, max_slices=5
)
```