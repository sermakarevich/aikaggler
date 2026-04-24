# [Inference] Baseline TransUNet   [LB 0.537]

- **Author:** Manas Choudhary
- **Votes:** 358
- **Ref:** choudharymanas/inference-baseline-transunet-lb-0-537
- **URL:** https://www.kaggle.com/code/choudharymanas/inference-baseline-transunet-lb-0-537
- **Last run:** 2025-12-31 18:58:40.963000

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
var="/kaggle/input/vsdetection-packages-offline-installer-only/whls"
!pip install \
    "$var"/keras_nightly-3.12.0.dev2025100703-py3-none-any.whl \
    "$var"/tifffile-2025.12.12-py3-none-any.whl \
    "$var"/imagecodecs-2025.11.11-cp311-abi3-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl \
    "$var"/medicai-0.0.3-py3-none-any.whl \
    --no-index \
    --find-links "$var"
```

```python
# --- Protobuf compatibility patch (for old code using MessageFactory.GetPrototype) ---

try:
    from google.protobuf import message_factory as _message_factory

    # Only patch if the method is missing (protobuf >= 5)
    if not hasattr(_message_factory.MessageFactory, "GetPrototype"):
        from google.protobuf.message_factory import GetMessageClass

        def _GetPrototype(self, descriptor):
            # Old API used MessageFactory().GetPrototype(descriptor)
            # New API is GetMessageClass(descriptor). We just bridge them.
            return GetMessageClass(descriptor)

        _message_factory.MessageFactory.GetPrototype = _GetPrototype
        print("Patched protobuf: added MessageFactory.GetPrototype")
    else:
        print("protobuf already has MessageFactory.GetPrototype; no patch needed.")
except Exception as e:
    print("Could not patch protobuf MessageFactory:", e)

import os
os.environ["KERAS_BACKEND"] = "jax"

import keras
from medicai.transforms import (
    Compose,
    ScaleIntensityRange,
)
from medicai.models import SegFormer, TransUNet
from medicai.utils.inference import SlidingWindowInference

import numpy as np
import pandas as pd
import zipfile
import tifffile
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
        ScaleIntensityRange(
            keys=["image"],
            a_min = 0,
            a_max = 255,
            b_min = 0,
            b_max = 1,
            clip = True,
        ),
    ])
    result = pipeline(data)
    return result["image"]
```

**Model**

```python
num_classes=3

def get_model():
    ## LB: 0.486
    # model = SegFormer(
    #     input_shape=(128, 128, 128, 1),
    #     encoder_name='mit_b2',
    #     classifier_activation='softmax',
    #     num_classes=num_classes,
    # )
    # model.load_weights(
    #     "/kaggle/input/vsd-model/keras/segformer.mit.b2/2/segformer.mit.b2.weights.h5"
    # )

    ## LB: 0.5 
    model = TransUNet(
        input_shape=(160, 160, 160, 1),
        encoder_name='seresnext50',
        classifier_activation='softmax',
        num_classes=num_classes,
    )
    model.load_weights(
        "/kaggle/input/train-vesuvius-surface-3d-detection-on-tpu/model.weights.h5"
    )
    return model
```

```python
model = get_model()
model.count_params() / 1e6
# predictor = tf.function(model, jit_compile=True)
```

```python
model.instance_describe()
```

**Sliding Window Inference**

```python
pred = SlidingWindowInference(
    model,
    roi_size=(160,160,160),
    num_classes = 3,
    mode="gaussian",
    overlap=0.50,
    sw_batch_size = 1
)
```

```python
import numpy as np
import tifffile
import scipy.ndimage as ndi
from skimage.morphology import remove_small_objects


def load_volume(path):
    vol = tifffile.imread(path)          # (D, H, W)
    vol = vol.astype(np.float32)
    vol = vol[None, ..., None]           # (1, D, H, W, 1)
    return vol


# ==========================================
# ROTATION TTA HELPERS (CLOCKWISE)
# ==========================================
def rot90_volume(vol, k):
    """
    Rotate volume k times 90° clockwise in HW plane.
    vol:
      (1, D, H, W, 1) OR (D, H, W)
    """
    if vol.ndim == 5:
        return np.rot90(vol, k=-k, axes=(2, 3))
    else:
        return np.rot90(vol, k=-k, axes=(1, 2))


def unrot90_volume(vol, k):
    return rot90_volume(vol, (4 - k) % 4)


def predict_probs_tta_rot(sample):
    """
    4x rotation TTA: 0°, 90°, 180°, 270°
    sample: (1, D, H, W, 1)
    returns: averaged probs (D, H, W)
    """
    probs_accum = []

    for k in range(4):
        s_rot = rot90_volume(sample, k)

        out = pred(s_rot)              # (1, D, H, W, 2)
        out = np.asarray(out)
        probs = out[0, ..., 1]         # (D, H, W)

        probs = unrot90_volume(probs, k)
        probs_accum.append(probs)

    return np.mean(probs_accum, axis=0)


# ==========================================
# HELPER: Anisotropic Structure Builder
# ==========================================
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


# ==========================================
# MAIN POST-PROCESSING LOGIC
# ==========================================
def topo_postprocess(
    probs,          # (D, H, W)
    T_low=0.90,
    T_high=0.90,
    z_radius=1,
    xy_radius=0,
    dust_min_size=100,
):
    # --- Step 1: 3D Hysteresis ---
    strong = probs >= T_high
    weak   = probs >= T_low

    if not strong.any():
        return np.zeros_like(probs, dtype=np.uint8)

    struct_hyst = ndi.generate_binary_structure(3, 3)
    mask = ndi.binary_propagation(strong, mask=weak, structure=struct_hyst)

    if not mask.any():
        return np.zeros_like(probs, dtype=np.uint8)

    # --- Step 2: 3D Anisotropic Closing ---
    if z_radius > 0 or xy_radius > 0:
        struct_close = build_anisotropic_struct(z_radius, xy_radius)
        if struct_close is not None:
            mask = ndi.binary_closing(mask, structure=struct_close)

    # --- Step 3: Dust Removal ---
    if dust_min_size > 0:
        mask = remove_small_objects(mask.astype(bool), min_size=dust_min_size)

    return mask.astype(np.uint8)


# ==========================================
# PREDICT (WITH ROTATION TTA)
# ==========================================
def predict(
    sample,
    iid=None,
    T_low=0.45,
    T_high=0.85,
    z_radius=1,
    xy_radius=0,
    dust_min_size=100,
):
    """
    sample: (1, D, H, W, 1)
    """

    # --------- ROTATION TTA PROBS ---------
    probs_fg = predict_probs_tta_rot(sample)   # (D, H, W)

    if iid is not None:
        np.save(iid, probs_fg)

    # --------- POSTPROCESS (UNCHANGED) ----
    final = topo_postprocess(
        probs_fg,
        T_low=T_low,
        T_high=T_high,
        z_radius=z_radius,
        xy_radius=xy_radius,
        dust_min_size=dust_min_size,
    )

    return final  # (D, H, W) uint8 {0,1}
```

**Prediction and Zip Submission**

```python
testing = False
```

```python
if testing:
    
    test_dir = "/kaggle/input/vesuvius-challenge-surface-detection/train_images"
    test_df = pd.read_csv(f"{root_dir}/train.csv")
    test_ids = {956073442, 961304774,969293709,975031774,985841575,992852942}
    test_df = (
        test_df
        .loc[test_df["id"].isin(test_ids)]
        .reset_index(drop=True)
    )
```

```python
with zipfile.ZipFile(
    zip_path, "w", compression=zipfile.ZIP_DEFLATED
) as z:
    for image_id in test_df["id"]:
        tif_path = f"{test_dir}/{image_id}.tif"
            
        volume = load_volume(tif_path)
        volume = val_transformation(volume)
        if testing :
            output = predict(volume,f"{image_id}") 
        else :
            output = predict(volume)
        
        out_path = f"{output_dir}/{image_id}.tif"
        tifffile.imwrite(out_path, output.astype(np.uint8))

        z.write(out_path, arcname=f"{image_id}.tif")
        os.remove(out_path)

print("Submission ZIP:", zip_path)
```