# [Train] Vesuvius Surface 3D Detection on TPU

- **Author:** Innat
- **Votes:** 266
- **Ref:** ipythonx/train-vesuvius-surface-3d-detection-on-tpu
- **URL:** https://www.kaggle.com/code/ipythonx/train-vesuvius-surface-3d-detection-on-tpu
- **Last run:** 2026-01-28 11:19:08.243000

---

<div align="center">
    <a href="https://github.com/innat/medic-ai">
        <img src="https://i.imgur.com/nWOYfUO.png" width="350">
    </a>
</div>

## About

- It is starter designed for **Vesuvius Challenge - Surface Detection** in kaggle.
- We will utilize [Medic-AI](https://github.com/innat/medic-ai), a high-performance library built on Keras 3 tailored for 2D and 3D medical image analysis. Thanks to its multi-backend architecture, it operates seamlessly across `tensorflow`, `torch`, and `jax`. This flexibility allows developers, including dedicated `torch` users to integrate state-of-the-art 2D or 3D segmentation models directly into their existing training workflows. One key architectural detail to note: Medic-AI adopts the (`depth, y, x, channel`) convention for input shapes. [This guide](https://www.kaggle.com/code/ipythonx/medicai-x-isic-2017-starter-x-binary-segmentation) demonstrates how to plug-and-play these capabilities into your own pure `torch` projects.
- The official documentaiton of [`Medic-AI`](https://github.com/innat/medic-ai) is bit out-dated at the moment, and so to get up-to-date documentation, please refer to the GitHub readme page created for each model, i.e. [segormer](https://github.com/innat/medic-ai/blob/main/medicai/models/segformer/README.md), [trans-unet](https://github.com/innat/medic-ai/blob/main/medicai/models/transunet/README.md), [unetr++](https://github.com/innat/medic-ai/blob/main/medicai/models/unetr_plus_plus/README.md), [swin-unetr](https://github.com/innat/medic-ai/blob/main/medicai/models/swin/README.md#swin-unetr), [upernet](https://github.com/innat/medic-ai/blob/main/medicai/models/upernet/README.md), [convnext](https://github.com/innat/medic-ai/blob/main/medicai/models/convnext/README.md) etc.
- Since [medicia](https://github.com/innat/medic-ai) is a new project, any feedback, suggestions, or contributions to help make it more reliable, robust, and improved are highly appreciated.

## Code

- **Training**: This notebook is the main training code, updated time to time. It uses `jax` backend. To use other backend (i.e., `torch`), see [this](https://www.kaggle.com/code/ipythonx/inference-vesuvius-surface-3d-detection).
-  **Inference**: The inference code can be found here: [[Inference] Vesuvius Surface 3D Detection](https://www.kaggle.com/code/ipythonx/inference-vesuvius-surface-3d-detection)

```python
from IPython.display import clear_output

# This is required for TPU training at the moment in kaggel env with Jax backend.
!pip install tensorflow -qU

var="/kaggle/input/vsdetection-packages-offline-installer-only/whls"
!pip install \
    "$var"/keras_nightly-3.12.0.dev2025100703-py3-none-any.whl \
    --no-index \
    --find-links "$var"

clear_output()
```

```python
# To get up-to-date feature, installing from scource is safe.
!pip install git+https://github.com/innat/medic-ai.git -q

# Installing is optional, we'll be using `tfrecord` format instead of `tif`.
# !pip install imagecodecs tifffile -q
```

```python
import os, warnings

os.environ["KERAS_BACKEND"] = "jax"
warnings.filterwarnings('ignore')
```

```python
import glob
import numpy as np
import pandas as pd

from PIL import Image
import matplotlib.pyplot as plt

# mainly for training API
import keras
from keras import ops
from keras.optimizers import SGD, AdamW, Muon
from keras.optimizers.schedules import CosineDecay, PolynomialDecay

# only for tf.data API
import tensorflow as tf

# mainly for 3D or 2D models, transformation, loss, metrics etc
import medicai
from medicai.transforms import (
    Compose,
    NormalizeIntensity,
    ScaleIntensityRange,
    Resize,
    RandShiftIntensity,
    RandRotate90,
    RandRotate,
    RandFlip,
    RandCutOut,
    RandSpatialCrop
)
from medicai.layers import ResizingND
from medicai.models import (
    UNet, SegFormer, TransUNet, SwinUNETR, UPerNet, ConvNeXtV2Tiny, UNETRPlusPlus
)
from medicai.losses import (
    SparseDiceCELoss, SparseTverskyLoss, SparseCenterlineDiceLoss
)
from medicai.metrics import SparseDiceMetric
from medicai.callbacks import SlidingWindowInferenceCallback
from medicai.utils import SlidingWindowInference
from medicai.utils import soft_skeletonize
```

```python
# due to distributed training only
keras.config.disable_flash_attention()

# reproducibility
keras.utils.set_random_seed(101)

# distributed config
devices = keras.distribution.list_devices()
data_parallel = keras.distribution.DataParallel(devices=devices)
keras.distribution.set_distribution(data_parallel)
total_device = len(devices)

print(f'detected devices: {devices}')
print(f'total device: {total_device}')
```

```python
keras.version(), keras.config.backend(), medicai.version()
```

## Data Loader

```python
input_shape=(128, 128, 128)
batch_size=1 * total_device
num_classes=3

# Each tfrecord contains 6 samples, total 786 samples.
num_samples = 780
epochs = 200
```

**TFRecord Decoder**

```python
def parse_tfrecord_fn(example):
    feature_description = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "label": tf.io.FixedLenFeature([], tf.string),
        "image_shape": tf.io.FixedLenFeature([3], tf.int64),
        "label_shape": tf.io.FixedLenFeature([3], tf.int64),
    }
    parsed_example = tf.io.parse_single_example(example, feature_description)
    image = tf.io.decode_raw(parsed_example["image"], tf.uint8)
    label = tf.io.decode_raw(parsed_example["label"], tf.uint8)
    image_shape = tf.cast(parsed_example["image_shape"], tf.int64)
    label_shape = tf.cast(parsed_example["label_shape"], tf.int64)
    image = tf.reshape(image, image_shape)
    label = tf.reshape(label, label_shape)
    return image, label
```

**Preprocessing and Augmentation**

```python
def prepare_inputs(image, label):
    # Add channel dimension
    image = image[..., None] # (D, H, W, 1)
    label = label[..., None] # (D, H, W, 1)

    # Convert to float32
    image = tf.cast(image, tf.float32)
    label = tf.cast(label, tf.float32)
    return image, label
```

```python
def train_transformation(image, label):
    data = {"image": image, "label": label}
    pipeline = Compose([
        ## Geometric transformation
        RandSpatialCrop(
            keys=["image", "label"],
            roi_size=input_shape,
            random_center=True,
            random_size=False,
            invalid_label=2,         
            min_valid_ratio=0.5,     
            max_attempts=10
        ),
        RandFlip(keys=["image", "label"], spatial_axis=[0], prob=0.5),
        RandFlip(keys=["image", "label"], spatial_axis=[1], prob=0.5),
        RandFlip(keys=["image", "label"], spatial_axis=[2], prob=0.5),
        RandRotate90(
            keys=["image", "label"], 
            prob=0.4, 
            max_k=3, 
            spatial_axes=(0, 1)
        ),
        RandRotate(
            keys=["image", "label"], 
            factor=0.2, 
            prob=0.7, 
            fill_mode="crop",
        ),

        ## Intensiry transformation
        NormalizeIntensity(
            keys=["image"], 
            nonzero=True,
            channel_wise=False
        ),
        RandShiftIntensity(
            keys=["image"], offsets=0.10, prob=0.5
        ),
        ## Spatial transformation 
        RandCutOut(
            keys=["image", "label"],
            invalid_label=2, 
            mask_size=[
                input_shape[1]//4,
                input_shape[2]//4
            ],
            fill_mode="constant", # "constant", "gaussian"
            cutout_mode='volume', # "slice", "volume"
            prob=0.8,
            num_cuts=5,
        ),
    ])
    result = pipeline(data)
    return result["image"], result["label"]


def val_transformation(image, label):
    data = {"image": image, "label": label}
    pipeline = Compose([
        NormalizeIntensity(
            keys=["image"], 
            nonzero=True,
            channel_wise=False
        ),
    ])
    result = pipeline(data)
    return result["image"], result["label"]
```

```python
def tfrecord_loader(tfrecord_pattern, batch_size=1, shuffle=True):
    dataset = tf.data.TFRecordDataset(
        tf.io.gfile.glob(tfrecord_pattern)
    )
    dataset = dataset.shuffle(buffer_size=100) if shuffle else dataset 
    dataset = dataset.map(
        parse_tfrecord_fn, num_parallel_calls=tf.data.AUTOTUNE
    )
    dataset = dataset.map(
        prepare_inputs,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    if shuffle:
        dataset = dataset.map(
            train_transformation,
            num_parallel_calls=tf.data.AUTOTUNE
        )
    else:
        dataset = dataset.map(
            val_transformation,
            num_parallel_calls=tf.data.AUTOTUNE
        )
    dataset = dataset.batch(batch_size, drop_remainder=shuffle)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset
```

```python
all_tfrec = sorted(
    glob.glob("/kaggle/input/vesuvius-tfrecords/*.tfrec"),
    key=lambda x: int(x.split("_")[-1].replace(".tfrec", ""))
)

val_idx = -1
val_patterns = [all_tfrec[val_idx]]
train_patterns = [
    f for i, f in enumerate(all_tfrec) if i != len(all_tfrec) + val_idx
]

train_ds = tfrecord_loader(
    train_patterns, batch_size=batch_size, shuffle=True
)
val_ds = tfrecord_loader(
    val_patterns, batch_size=1, shuffle=False
)
```

```python
x, y = next(iter(val_ds))
x.shape, y.shape
```

**Viz**

```python
def plot_sample(x, y, sample_idx=0, max_slices=16):
    img = np.squeeze(x[sample_idx])  # (D, H, W)
    mask = np.squeeze(y[sample_idx])  # (D, H, W)
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
def plot_planes(image, mask, alpha=0.4):
    # Central slices
    d, h, w = image.shape
    axial_img    = image[d // 2]
    coronal_img  = image[:, h // 2, :]
    sagittal_img = image[:, :, w // 2]

    axial_msk    = mask[d // 2]
    coronal_msk  = mask[:, h // 2, :]
    sagittal_msk = mask[:, :, w // 2]

    slices_img = [axial_img, coronal_img, sagittal_img]
    slices_msk = [axial_msk, coronal_msk, sagittal_msk]
    
    titles = ["Axial (XY plane)", "Coronal (XZ plane)", "Sagittal (YZ plane)"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for i, ax in enumerate(axes):
        ax.imshow(slices_img[i], cmap="gray")

        # overlay jet only where mask > 0
        m = slices_msk[i]
        if m.max() > 0:
            ax.imshow(m, cmap="jet", alpha=alpha)

        ax.set_title(titles[i])
        ax.axis("off")

    plt.tight_layout()
    plt.show()
```

```python
plot_sample(
    x, y, sample_idx=0, max_slices=4
)
```

```python
plot_planes(
    np.squeeze(x[0]), # picking one sample
    np.squeeze(y[0])  # picking one sample
)
```

```python
soft_skel = soft_skeletonize(
    ops.cast(y == 1, 'float32'),
    iters=10
)
soft_skel.shape
```

```python
plot_sample(
    y, soft_skel, sample_idx=0, max_slices=4
)
```

## Model

```python
## check available models (classification + segmentation)
# medicai.models.list_models()
```

```python
## Pre-build encoder
model = SegFormer(
    input_shape=input_shape + (1,),
    encoder_name='mit_b0',
    classifier_activation='softmax',
    num_classes=num_classes,
)

# model = UPerNet(
#     input_shape=input_shape + (1,),
#     encoder_name="convnext_base",
#     classifier_activation='softmax',
#     num_classes=num_classes,
# )

# model = UNETRPlusPlus(
#     input_shape=input_shape + (1,),
#     encoder_name='unetr_plusplus_encoder',
#     classifier_activation='softmax',
#     num_classes=num_classes,
# )

# model = TransUNet(
#     encoder_name='seresnext50',
#     input_shape=input_shape + (1,),
#     num_classes=num_classes,
#     classifier_activation='softmax'
# )
model.count_params() / 1e6
```

```python
# ## Custom encoder
# # backbone with 4 skip connection
# backbone = ConvNeXtV2Tiny(
#     input_shape=input_shape + (1,),
#     include_top=False
# )
# # print(backbone.pyramid_outputs)

# # segmentator
# segmentor = TransUNet(
#     encoder=backbone,
#     encoder_depth=4,
#     num_classes=num_classes,
#     classifier_activation='softmax'
# )
# inputs = keras.Input(shape=input_shape + (1,))
# x = segmentor(inputs)

# # final supsampling, 2x tmes.
# outputs = ResizingND(
#     target_shape=input_shape,
#     interpolation='trilinear',
#     align_corners=False
# )(x)
# model = keras.Model(inputs=inputs, outputs=outputs)
# model.count_params() / 1e6
```

```python
# ALERT: This attributes only available in medicai (not in core keras)
try:
    print(model.instance_describe())
except AttributeError:
    pass
```

# LR Schedules and Optimizer

```python
steps_per_epoch = num_samples // batch_size
total_steps = steps_per_epoch * epochs
warmup_steps = int(total_steps * 0.05)
decay_steps = max(1, total_steps - warmup_steps)
lr_schedule = CosineDecay(
    initial_learning_rate=1e-6,
    decay_steps=decay_steps,
    warmup_target=min(3e-4, 1e-4 * (batch_size / 2)),
    warmup_steps=warmup_steps,
    alpha=0.1,
)
```

```python
# define optomizer, loss, metrics
optim = keras.optimizers.AdamW(
    learning_rate=lr_schedule,
    weight_decay=1e-5,
)

dice_ce_loss_fn = SparseDiceCELoss(
    from_logits=False, 
    num_classes=num_classes,
    ignore_class_ids=2,
)
cldice_loss_fn = SparseCenterlineDiceLoss(
    from_logits=False, 
    num_classes=num_classes,
    target_class_ids=1,
    ignore_class_ids=2,
    iters=50
)
combined_loss_fn = lambda y_true, y_pred: (
    dice_ce_loss_fn(y_true, y_pred) + cldice_loss_fn(y_true, y_pred)
)


metrics = [
    SparseDiceMetric(
        from_logits=False, 
        num_classes=num_classes, 
        ignore_class_ids=2,
        name='dice'
    ),
]

model.compile(
    optimizer=optim,
    loss=combined_loss_fn,
    metrics=metrics,
)
```

```python
swi_callback_metric = SparseDiceMetric(
    from_logits=False,
    ignore_class_ids=2,
    num_classes=num_classes,
    name='val_dice',
)

swi_callback = SlidingWindowInferenceCallback(
    model,
    dataset=val_ds,
    metrics=swi_callback_metric,
    num_classes=num_classes,
    interval=5,
    overlap=0.5,
    mode='gaussian',
    roi_size=input_shape,
    sw_batch_size=1 * total_device,
    save_path="model.weights.h5"
)
```

```python
# ALERT: Starting may take time.
model.fit(
    train_ds,
    epochs=epochs,
    callbacks=[
        swi_callback
    ]
)
```

## Eval

```python
model.load_weights(
    "model.weights.h5"
)
swi = SlidingWindowInference(
    model,
    num_classes=num_classes,
    roi_size=input_shape,
    mode='gaussian',
    sw_batch_size=1 * total_device,
    overlap=0.5,
)
```

```python
dice = SparseDiceMetric(
    from_logits=False,
    num_classes=num_classes,
    ignore_class_ids=2,
    name='dice',
)
```

```python
for sample in val_ds:
    x, y = sample
    output = swi(x)
    y = ops.convert_to_tensor(y)
    output = ops.convert_to_tensor(output)
    dice.update_state(y, output)

dice_score = float(ops.convert_to_numpy(dice.result()))
print(f"Dice Score: {dice_score:.4f}")
dice.reset_state()
```

```python
x, y = next(iter(val_ds))
x.shape, y.shape
```

```python
y_pred = swi(x)
y_pred.shape
```

```python
segment = y_pred.argmax(-1).astype(np.uint8)
segment.shape, np.unique(segment)
```

```python
plot_sample(
    x, segment, sample_idx=0, max_slices=4
)
```

**Next Stop**

- [Affinity Feature Strengthening](https://arxiv.org/pdf/2211.06578)
- [Meta-Tubular-Net: A Robust Topology-Aware Re-Weighting Network](file:///C:/Users/ASUS/Pictures/Screenshots/ssrn-4132287.pdf)
- [Landmark-Assisted Anatomy-Sensitive](file:///C:/Users/ASUS/Downloads/diagnostics-13-02260-v2.pdf)
- [LEAD: Self-Supervised Landmark Estimation](https://arxiv.org/pdf/2204.02958)
- [TopoSeg: Topology-Aware](https://openaccess.thecvf.com/content/ICCV2023/papers/He_TopoSeg_Topology-Aware_Nuclear_Instance_Segmentation_ICCV_2023_paper.pdf)
- [Centerline Dice Loss](https://github.com/jocpae/clDice)
- [Skeleton-Recall Loss](https://github.com/MIC-DKFZ/Skeleton-Recall)
- [Virtually Unrolling the Herculaneum Papyri](https://arxiv.org/pdf/2512.04927v1)

## Deep Supervision Recipes

**Deep supervision** in segmentation models improves training by adding auxiliary loss functions to hidden layers, mitigating gradient vanishing, and speeding up convergence. It forces intermediate layers to learn more discriminative features, enhancing overall performance.

**Preface**

Say, we have an segmentation model with `5` level pyramid encoder blocks. So, typically we would have `5` upsampling stages. Some are deep, some are high level. For example, model like `TransUNet`, we can inspect:

```python
model = TransUNet(
    encoder_name='seresnext50',
    input_shape=target_shape + (1,),
    classifier_activation='softmax',
    num_classes=num_classes,
)

decoder_layers = [
    ("DS5", "decoder_proj_0"),
    ("DS4", "decoder_conv_4"),
    ("DS3", "decoder_conv_3"),
    ("DS2", "decoder_conv_2"),
    ("DS1", "decoder_conv_1"),
]

for name, layer_name in decoder_layers:
    feat = model.get_layer(layer_name).output
    print(name, feat.shape)
    
DS5 (None, 3, 3, 3, 256)
DS4 (None, 6, 6, 6, 128)
DS3 (None, 12, 12, 12, 64)
DS2 (None, 24, 24, 24, 32)
DS1 (None, 48, 48, 48, 16)
```

Ideally, we would pick higher level features, i.e. `DS3`, `DS2`, `DS1`. However for sake for demonstration, let's pick all. Now, in Keras, there are many ways we can model deep supervision. Here is one traditional way.


**Step 1**

```python
def prepare_multi_outputs(x, y):
    y_dict = {
        "final": y,
        "DS1": y,
        "DS2": y,
        "DS3": y,
        "DS4": y,
        "DS5": y,
    }
    return x, y_dict

def load_tfrecord_dataset(tfrecord_pattern, ...):
    ....
    if shuffle:
        dataset = dataset.map(
            train_transformation,
            num_parallel_calls=tf.data.AUTOTUNE
        )
        dataset = dataset.map( # < --- HERE
            prepare_multi_outputs, 
            num_parallel_calls=tf.data.AUTOTUNE
        )
    else:
        dataset = dataset.map(
            val_transformation,
            num_parallel_calls=tf.data.AUTOTUNE
        )
    dataset = dataset.batch(
        batch_size, drop_remainder=shuffle
    ).prefetch(tf.data.AUTOTUNE)
    return dataset
```

**Step 2**

Build the multi-output model.

```python
aux_outputs = {}
for name, layer_name in decoder_layers:
    feat = model.get_layer(layer_name).output

    # project features → class logits
    logits = keras.layers.Conv3D(
        filters=num_classes,
        kernel_size=1,
        padding="same",
        name=f"aux_logits_{name}",
    )(feat)

    # resize logits (trilinear)
    resized = ResizingND(
        target_shape=input_size,
        interpolation="trilinear",
        name=f"resize_{name}",
    )(logits)
    
    # softmax after resizing
    aux = keras.layers.Activation(
        "softmax",
        name=name,
        dtype="float32",
    )(resized)

    aux_outputs[name] = aux

deep_supervised_model = keras.Model(
    inputs=model.input,
    outputs={
        "final": model.output,
        **aux_outputs,
    },
)
```

Now, we compile this model.

```python
losses = {
    "final": SparseDiceCELoss | AnyLossWeWant,
    "DS1": SparseDiceCELoss | AnyLossWeWant,
    "DS2": SparseDiceCELoss | AnyLossWeWant,
    "DS3": SparseDiceCELoss | AnyLossWeWant,
    "DS4": SparseDiceCELoss | AnyLossWeWant,
    "DS5": SparseDiceCELoss | AnyLossWeWant,
}

loss_weights = {
    "final":  0.5079,
    "DS1":    0.2539,
    "DS2":    0.1269,
    "DS3":    0.0635,
    "DS4":    0.0317,
    "DS5":    0.0158,
}

deep_supervised_model.compile(
    optimizer=optim,
    loss=losses,
    loss_weights=loss_weights,
)
```

We use loss weights according to [nnUNet-tf](https://github.com/NVIDIA/DeepLearningExamples/blob/729963dd47e7c8bd462ad10bfac7a7b0b604e6dd/TensorFlow2/Segmentation/nnUNet/models/nn_unet.py#L82-L93).

**Step 3**

For validation or inference, just use raw model. These are Keras functional API. So, weights are already shared. 

```python
swi_callback = SlidingWindowInferenceCallback(
    model,
    ...
)
```

**Step 4**

Fit the model.

```python
deep_supervised_model.fit(
    train_ds,
    epochs=epochs,
    callbacks=[
        swi_callback,
    ]
)
```

## Resume Training from Interruption

Say, we set 500 epoch and saving checkpoint every 5 epochs. Now, the training got interrupted after epoch 350 before finising 351. In that case, we can resume training as follows:

```python
model = Model()
model.load_weights(...) # last saved checkpoint

# optim, lr sched etc, same as before, i.e.,
steps_per_epoch = num_samples // batch_size
lr_schedule = CosineDecay(
    initial_learning_rate=1e-6,
    decay_steps=decay_steps,
    warmup_target=min(3e-4, 1e-4 * (batch_size / 2)),
    warmup_steps=warmup_steps,
    alpha=0.1,
)
optim = keras.optimizers.AdamW(
    learning_rate=lr_schedule,
    weight_decay=1e-5,
)

# Important
already_trained_steps = 350 * steps_per_epoch
optim.iterations.assign(already_trained_steps)

model.fit(
    ...
    initial_epoch=350,
)
```

FYI, if we could save the **compiled model** with `.keras` format, it would be much easy.