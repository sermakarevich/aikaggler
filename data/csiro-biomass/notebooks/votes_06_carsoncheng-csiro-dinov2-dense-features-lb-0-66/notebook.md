# 🌿🌱 CSIRO | DINOv2 Dense Features | LB 0.66

- **Author:** Carson Cheng
- **Votes:** 340
- **Ref:** carsoncheng/csiro-dinov2-dense-features-lb-0-66
- **URL:** https://www.kaggle.com/code/carsoncheng/csiro-dinov2-dense-features-lb-0-66
- **Last run:** 2025-11-29 13:48:04.553000

---

# 🌿🌱 CSIRO Biomass Regression using Dense Features of DINOv2

- Use the dense patch-based features extracted by the DINOv2 model
- For each patch feature vector, use a common MLP (with weight sharing) to make predictions for each patch
- Average the MLP predictions for each patch to obtain final predictions
- Uses sharpness-aware minimization (https://github.com/davda54/sam) for training (training code not shown here)
- Computes loss only on image-level labels (TODO: incorporate regularizations to make the problem more well-conditioned, or use other methods to compute loss on patch-level labels)

# Imports

```python
#!pip install transformers==4.57.1
import pandas as pd
import numpy as np
import torch
import torchvision
import os
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset, Dataset
import torchvision.transforms as transforms
from PIL import Image
!cp -r "/kaggle/input/rsna-models/facebookresearch_dinov2_main (1)/root/.cache/torch/hub/facebookresearch_dinov2_main" /kaggle/working/dinov2
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
```

# Load DINOv2 backbone

```python
%cd /kaggle/working/dinov2
import torch.nn as nn
from transformers import Dinov2Model
model = Dinov2Model.from_pretrained('/kaggle/input/dinov2/pytorch/giant/1')
model.head = nn.Identity()
model = model.cuda()
```

# MLP Architecture

```python
import torch.nn as nn
class BiomassMLP(nn.Module):
    # and also weight initialization
    def __init__(self, input_size, hidden_size=512, dropout_rate=0.3):
        super(BiomassMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size // 4),
            #nn.BatchNorm1d(hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size // 4, 1)
        )
        #self.network = nn.Linear(input_size, 1) # use a simple MLP with a single activation function, or a polynomial function
        #self.network = SimplePolynomialLayer(input_size)
    def forward(self, x):
        #return self.network(x)
        return torch.mean(torch.relu(self.network(x)), dim=(1, 2))
```

# Inference

- TTA (test-time augmentation) is incorporated into the inference
- The augmentations include horizontal / vertical flips, rotations, and Gaussian blurs

```python
mapping = {"Dry_Clover_g": 0, "Dry_Dead_g": 1, "Dry_Green_g": 2, "Dry_Total_g": 3, "GDM_g": 4}
```

```python
test_embeds = {}
counter = 0
import torchvision.transforms as transforms
import pandas as pd
from PIL import Image
transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((224, 224)), transforms.Normalize(mean, std)])
test_df = pd.read_csv("/kaggle/input/csiro-biomass/test.csv")
augmentation_transforms = [
    transforms.Compose([transforms.ToTensor(), transforms.Resize((224, 224)), transforms.Normalize(mean, std), torchvision.transforms.GaussianBlur(5)]),  # Original
    transforms.Compose([transforms.RandomHorizontalFlip(p=1.0), transforms.ToTensor(), transforms.Resize((224, 224)), transforms.Normalize(mean, std), torchvision.transforms.GaussianBlur(5)]),  # Horizontal flip
    transforms.Compose([transforms.RandomVerticalFlip(p=1.0), transforms.ToTensor(), transforms.Resize((224, 224)), transforms.Normalize(mean, std), torchvision.transforms.GaussianBlur(5)]),  # Vertical flip
    transforms.Compose([transforms.RandomHorizontalFlip(p=1.0), transforms.RandomVerticalFlip(p=1.0), transforms.ToTensor(), transforms.Resize((224, 224)), transforms.Normalize(mean, std), torchvision.transforms.GaussianBlur(5)]),  # Both flips
    transforms.Compose([transforms.RandomRotation(degrees=90), transforms.ToTensor(), transforms.Resize((224, 224)), transforms.Normalize(mean, std), torchvision.transforms.GaussianBlur(5)]),  # 90 degree rotation
    transforms.Compose([transforms.RandomRotation(degrees=270), transforms.ToTensor(), transforms.Resize((224, 224)), transforms.Normalize(mean, std), torchvision.transforms.GaussianBlur(5)]),  # 270 degree rotation
]
root = "/kaggle/input/csiro-biomass/"
sample_ids = []
from tqdm import tqdm
for i in tqdm(range(len(test_df))):
    entry = test_df.iloc[i]
    file_path = root + entry['image_path']
    sample_id = entry['sample_id']
    #y = torch.tensor([[entry['target']]])
    if sample_id.split("_")[0] not in sample_ids:
        image_embeddings = []
        for aug in augmentation_transforms:
            img = Image.open(file_path)
            x = aug(img).unsqueeze(0)
            with torch.no_grad():
                x = x.cuda()
                image_embeddings.append(torch.cat([model(x).last_hidden_state[:,1:,:].cpu()[0]]).unsqueeze(0))
                counter += 1
        #print(image_embeddings[0].shape, sample_id)
        test_embeds[sample_id.split("_")[0]] = torch.stack(image_embeddings, dim=0)
        #print(torch.stack(image_embeddings, dim=0).shape)
        sample_ids.append(sample_id.split("_")[0])        
    if counter % 100 == 0:
        print(f"{counter} batches processed.")
```

```python
regressors = [[None for i in range(5)] for j in range(5)]
for i in range(5):
    for j in range(5):
        regressors[i][j] = torch.load(f"/kaggle/input/csiro-mlps-run1/target{i}/fold{j}.pt", weights_only=False)
import joblib
(selected_features, scalers) = joblib.load("/kaggle/input/csiro-mlps-run1/sfs_scalers_run1.joblib")
```

```python
predictions = []
sample_ids = []
test_df = pd.read_csv("/kaggle/input/csiro-biomass/test.csv")
for i in range(len(test_df)):
    entry = test_df.iloc[i]
    X = test_embeds[entry['sample_id'].split("__")[0]]
    sample_ids.append(entry['sample_id'])
    models = regressors[mapping[entry['sample_id'].split("__")[1]]]
    sfs = selected_features[mapping[entry['sample_id'].split("__")[1]]]
    scaler_list = scalers[mapping[entry['sample_id'].split("__")[1]]]
    prediction = 0.0
    for i in range(len(models)):
        item = models[i]
        scaler = scaler_list[i]
        sf = sfs[i]
        item.eval()
        #print(data)
        #print(item)
        #print(item(data))
        #print(data.shape)
        single_pred = torch.mean(torch.relu(item(X.squeeze(1).cuda())))
        #single_pred = item(torch.tensor(scaler.transform(X[:,sf])))
        if single_pred < 0.0:
            single_pred = 0.0
        prediction += single_pred.cpu()
    prediction = prediction / 5
    predictions.append(float(prediction))
```

```python
%cd /kaggle/working
submission = pd.DataFrame({
    'sample_id': sample_ids,
    'target': predictions
})

submission.to_csv('submission.csv', index=False)
submission
```