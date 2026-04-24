# [Inference] Vesuvius Surface 3D Detection_Epoch_X5

- **Author:** Tao Li
- **Votes:** 248
- **Ref:** tonyai007/inference-vesuvius-surface-3d-detection-epoch-x5
- **URL:** https://www.kaggle.com/code/tonyai007/inference-vesuvius-surface-3d-detection-epoch-x5
- **Last run:** 2026-02-27 03:58:56.397000

---

### Inference From Innat
https://www.kaggle.com/code/ipythonx/inference-vesuvius-surface-3d-detection?scriptVersionId=294333704

### Training Notebooks From Innat
- [Vesuvius Surface 3D Detection in Keras-JAX](https://www.kaggle.com/code/ipythonx/vesuvius-surface-3d-detection-in-jax)
- [Vesuvius Surface 3D Detection in PyTorch](https://www.kaggle.com/code/ipythonx/vesuvius-surface-3d-detection-in-pytorch)
- [Vesuvius Surface 3D Detection in PyTorch Lightning](https://www.kaggle.com/code/ipythonx/train-vesuvius-surface-3d-detection-in-lightning)
- [[WIP] Vesuvius Surface 2.5D Detection](https://www.kaggle.com/code/ipythonx/wip-vesuvius-surface-2-5d-detection)

```python
from IPython.display import clear_output
import os
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ["KERAS_BACKEND"] = "jax"
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "4")

var="/kaggle/input/vsdetection-packages-offline-installer-only/whls"
!pip install "$var"/keras_nightly-*.whl "$var"/tifffile-*.whl "$var"/imagecodecs-*.whl "$var"/medicai-*.whl --no-index --find-links "$var"
clear_output()

import keras
from medicai.transforms import Compose, NormalizeIntensity
from medicai.models import TransUNet
from medicai.utils.inference import SlidingWindowInference
import numpy as np
import pandas as pd
import zipfile
import tifffile
import scipy.ndimage as ndi
from skimage.morphology import remove_small_objects

print(f"Keras backend: {keras.config.backend()}, Keras version: {keras.version()}")

# 数据路径配置
root_dir = "/kaggle/input/vesuvius-challenge-surface-detection"
test_dir = f"{root_dir}/test_images"
output_dir = "/kaggle/working/submission_masks"
zip_path = "/kaggle/working/submission.zip"
os.makedirs(output_dir, exist_ok=True)
test_df = pd.read_csv(f"{root_dir}/test.csv")
print("Test dataframe preview:")
test_df.head()

# 数据预处理函数
def val_transformation(image):
    data = {"image": image}
    pipeline = Compose([NormalizeIntensity(keys=["image"], nonzero=True, channel_wise=False)])
    result = pipeline(data)
    return result["image"]

# 模型配置参数
num_classes=3
input_shape=(160, 160, 160)
model_weights_paths = [
    # "/kaggle/input/notebooks/tonyai007/train-vesuvius-seresnext50-comboloss/final_model.weights.h5",      # 50轮
    "/kaggle/input/notebooks/tonyai007/train-vesuvius-seresnext50-comboloss-2/final_model.weights.h5",    # 250轮
    # "/kaggle/input/notebooks/tonyai007/train-vesuvius-seresnext50-comboloss-3/final_model.weights.h5",    # 450轮
    "/kaggle/input/notebooks/tonyai007/train-vesuvius-seresnext50-comboloss-4/final_model.weights.h5",    # 600轮
    "/kaggle/input/notebooks/tonyai007/train-vesuvius-seresnext50-comboloss-5/final_model.weights.h5"     # 830轮
]

# 加载单个模型的函数
def build_single_model(weights_path):
    model = TransUNet(input_shape=(160, 160, 160, 1), encoder_name='seresnext50', classifier_activation=None, num_classes=3)
    model.load_weights(weights_path)
    return model

# 加载所有模型并创建对应的SWI实例
def load_all_models_and_swi():
    models = []
    swi_instances = []
    for weights_path in model_weights_paths:
        model = build_single_model(weights_path)
        models.append(model)
        swi = SlidingWindowInference(model, num_classes=3, roi_size=input_shape, sw_batch_size=1, mode='gaussian', overlap=0.48)
        swi_instances.append(swi)
        print(f"Successfully loaded model: {weights_path}")
    return models, swi_instances

# 加载所有模型和SWI实例
models, swi_instances = load_all_models_and_swi()
print(f"Model parameter count (M): {models[0].count_params() / 1e6:.2f}")

# 加载3D体数据
def load_volume(path):
    vol = tifffile.imread(path)
    vol = vol.astype(np.float32)
    vol = vol[None, ..., None]
    return vol

# 带TTA和多模型融合的预测函数
def predict_with_tta_and_model_fusion(inputs, swi_instances):
    all_model_logits = []
    for swi in swi_instances:
        model_logits = [swi(inputs)]
        # 空间维度翻转
        for axis in [1, 2, 3]:
            img_f = np.flip(inputs, axis=axis)
            p = swi(img_f)
            p = np.flip(p, axis=axis)
            model_logits.append(p)
        # 轴向旋转
        for k in [1, 2, 3]:
            img_r = np.rot90(inputs, k=k, axes=(2, 3))
            p = swi(img_r)
            p = np.rot90(p, k=-k, axes=(2, 3))
            model_logits.append(p)
        model_mean_logits = np.mean(model_logits, axis=0)
        all_model_logits.append(model_mean_logits)
    final_logits = np.mean(all_model_logits, axis=0)
    return final_logits.argmax(-1).astype(np.uint8).squeeze()

# 构建各向异性结构元素
def build_anisotropic_struct(z_radius: int, xy_radius: int):
    z, r = z_radius, xy_radius
    if z == 0 and r == 0: return None
    if z == 0 and r > 0:
        size = 2 * r + 1
        struct = np.zeros((1, size, size), dtype=bool)
        cy, cx = r, r
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if dy * dy + dx * dx <= r * r: struct[0, cy + dy, cx + dx] = True
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
                if dy * dy + dx * dx <= r * r: struct[cz + dz, cy + dy, cx + dx] = True
    return struct

# 拓扑后处理函数
def topo_postprocess(probs, T_low=0.5, T_high=0.9, z_radius=3, xy_radius=2, dust_min_size=250):
    strong = probs >= T_high
    weak   = probs >= T_low
    if not strong.any(): return np.zeros_like(probs, dtype=np.uint8)
    struct_hyst = ndi.generate_binary_structure(3, 3)
    mask = ndi.binary_propagation(strong, mask=weak, structure=struct_hyst)
    if not mask.any(): return np.zeros_like(probs, dtype=np.uint8)
    if z_radius > 0 or xy_radius > 0:
        struct_close = build_anisotropic_struct(z_radius, xy_radius)
        if struct_close is not None: mask = ndi.binary_closing(mask, structure=struct_close)
    if dust_min_size > 0: mask = remove_small_objects(mask.astype(bool), min_size=dust_min_size)
    return mask.astype(np.uint8)

# 推理流程
def inference_pipelines(volume, swi_instances):
    probs = predict_with_tta_and_model_fusion(volume, swi_instances)
    final = topo_postprocess(probs)
    return final

# 生成提交文件
with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
    for idx, image_id in enumerate(test_df["id"]):
        print(f"Processing {idx+1}/{len(test_df)}: {image_id}")
        tif_path = f"{test_dir}/{image_id}.tif"
        volume = load_volume(tif_path)
        volume = val_transformation(volume)
        output = inference_pipelines(volume, swi_instances)
        out_path = f"{output_dir}/{image_id}.tif"
        tifffile.imwrite(out_path, output.astype(np.uint8))
        z.write(out_path, arcname=f"{image_id}.tif")
        os.remove(out_path)

print(f"Submission ZIP generated: {zip_path}")
print(f"ZIP file size: {os.path.getsize(zip_path) / 1024 / 1024:.2f} MB")
```

```python
import matplotlib.pyplot as plt
def plot_sample(x, y, sample_idx=0, max_slices=16):
    img = np.squeeze(x[sample_idx])
    mask = np.squeeze(y[sample_idx])
    D = img.shape[0]
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

plot_sample(volume.numpy(), output[None], sample_idx=0, max_slices=5)
```