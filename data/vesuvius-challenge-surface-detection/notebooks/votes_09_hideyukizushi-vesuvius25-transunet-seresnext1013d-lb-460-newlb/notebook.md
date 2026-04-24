# Vesuvius25|TransUNet SEResNeXt1013D|LB.460(NewLB)

- **Author:** yukiZ
- **Votes:** 213
- **Ref:** hideyukizushi/vesuvius25-transunet-seresnext1013d-lb-460-newlb
- **URL:** https://www.kaggle.com/code/hideyukizushi/vesuvius25-transunet-seresnext1013d-lb-460-newlb
- **Last run:** 2025-12-31 08:44:27.710000

---

<h1 style="color: #6cb4e4;  text-align: center;  padding: 0.25em;  border-top: solid 2.5px #6cb4e4;  border-bottom: solid 2.5px #6cb4e4;  background: -webkit-repeating-linear-gradient(-45deg, #f0f8ff, #f0f8ff 3px,#e9f4ff 3px, #e9f4ff 7px);  background: repeating-linear-gradient(-45deg, #f0f8ff, #f0f8ff 3px,#e9f4ff 3px, #e9f4ff 7px);height:45px;">
<b>
OnlyInfKernel
</b></h1>

### **ℹ️INFO**
* First, I want to thank you for sharing such a strong baseline.
    * [@INNAT, LB.507, Vesuvius Surface 3D Detection](https://www.kaggle.com/code/ipythonx/inference-vesuvius-surface-3d-detection?scriptVersionId=285843716)
 
### **ℹ️[LB.454 2025/12/26]MyUpdate**(※Please note that this model was trained on data before the Dataset update on Dec 23, 2025)
* Results of improving the models and train pipeline available in medic-ai.
* Since good CV/LB was obtained, we will share the model weights together.
    * **Model: TransUNet**
    * **Encoder: SEResNeXt1013D**
    * **Local Validation(DiceScore): .7685**
    * **Public LB: .454**
        * ↑My TrainModel Weight↑:https://www.kaggle.com/datasets/hideyukizushi/colab-a-162v5-gpu-transunet-seresnext101-x160
        * ※Training was performed on Google Colaboratory using GPU A100 (80GB).

### **ℹ️[LB.460 2025/12/31]MyUpdate**(※Please note that this model was trained on data before the Dataset update on Dec 23, 2025)
* The Public LB dataset has been updated, reference I am publishing the model that produced my score.
    * **Model: TransUNet**
    * **Encoder: SEResNeXt1013D**
    * **Local Validation(DiceScore): .7679**
    * **Public LB: .460**
        * ↑My TrainModel Weight↑:https://www.kaggle.com/datasets/hideyukizushi/colab-a-162v4-gpu-transunet-seresnext101-x160

```python
var="/kaggle/input/vesuvius25-packages-offline-installer-v20251226/whls"
!pip install \
    "$var"/keras_nightly-3.12.0.dev2025100703-py3-none-any.whl \
    "$var"/tifffile-2025.10.16-py3-none-any.whl \
    "$var"/imagecodecs-2025.11.11-cp311-abi3-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl \
    "$var"/medicai-0.0.3-py3-none-any.whl \
    --no-index \
    --find-links "$var"
```

<h1 style="color: #6cb4e4;  text-align: center;  padding: 0.25em;  border-top: solid 2.5px #6cb4e4;  border-bottom: solid 2.5px #6cb4e4;  background: -webkit-repeating-linear-gradient(-45deg, #f0f8ff, #f0f8ff 3px,#e9f4ff 3px, #e9f4ff 7px);  background: repeating-linear-gradient(-45deg, #f0f8ff, #f0f8ff 3px,#e9f4ff 3px, #e9f4ff 7px);height:45px;">
<b>
Model Detail
</b></h1>

* List of models available on MedicAI

```python
import warnings
warnings.simplefilter('ignore')

import medicai
medicai.models.list_models()
```

* Encoder SEResNeXt1013D Detail

```python
from medicai.models import SEResNeXt101

tmp_backbone = SEResNeXt101(
    input_shape=(160, 160, 160) + (1,),
    include_top=False
)
tmp_backbone.summary()
```

<h1 style="color: #6cb4e4;  text-align: center;  padding: 0.25em;  border-top: solid 2.5px #6cb4e4;  border-bottom: solid 2.5px #6cb4e4;  background: -webkit-repeating-linear-gradient(-45deg, #f0f8ff, #f0f8ff 3px,#e9f4ff 3px, #e9f4ff 7px);  background: repeating-linear-gradient(-45deg, #f0f8ff, #f0f8ff 3px,#e9f4ff 3px, #e9f4ff 7px);height:45px;">
<b>
Main
</b></h1>

# 》》》**Libs**

```python
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

# 》》》**Dataset**

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

# 》》》**Transformation**

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

# 》》》**Load Model**

```python
def get_model():
    model = TransUNet(
        input_shape=(160, 160, 160, 1),
        encoder_name='seresnext101',
        classifier_activation='softmax',
        num_classes=3,
    )
    model.load_weights(
        f"/kaggle/input/colab-a-162v4-gpu-transunet-seresnext101-x160/model.weights.h5"
    )
    return model
```

```python
model = get_model()
model.count_params() / 1e6
```

# 》》》**Util**

```python
swi = SlidingWindowInference(
    model,
    num_classes=3,
    roi_size=(160, 160, 160),
    sw_batch_size=1,
    mode='gaussian',
    overlap=0.52,
)
```

```python
def load_volume(path):
    vol = tifffile.imread(path)
    vol = vol.astype(np.float32)
    vol = vol[None, ..., None]
    return vol

def predict(sample):
    mask = swi(sample)
    output = mask.argmax(-1).astype(np.uint8).squeeze()
    return output
```

# 》》》**Predict & Submission**

```python
with zipfile.ZipFile(
    zip_path, "w", compression=zipfile.ZIP_DEFLATED
) as z:
    for image_id in test_df["id"]:
        tif_path = f"{test_dir}/{image_id}.tif"
        
        volume = load_volume(tif_path)
        volume = val_transformation(volume)
        output = predict(volume) 
        
        out_path = f"{output_dir}/{image_id}.tif"
        tifffile.imwrite(out_path, output.astype(np.uint8))

        z.write(out_path, arcname=f"{image_id}.tif")
        os.remove(out_path)

print("Submission ZIP:", zip_path)
```