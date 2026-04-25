# csiro simple

- **Author:** Zhuang Jia
- **Votes:** 418
- **Ref:** jiazhuang/csiro-simple
- **URL:** https://www.kaggle.com/code/jiazhuang/csiro-simple
- **Last run:** 2025-12-17 05:24:59.093000

---

## Config

```python
import os
```

```python
# 环境设置

TRAIN = True  # submission时只用跑推理，设为 False

DEBUG = os.environ.get('KAGGLE_KERNEL_RUN_TYPE', '') == 'Interactive'  # 交互式环境下会少跑一些epoch，用于快速跑通流程，方便调试
LOCAL = os.environ.get('KAGGLE_KERNEL_RUN_TYPE', '') == ''  # 用于在本地开发，觉得代码ok后通过 `kaggle k push` 命令提交到 kaggle 平台

if LOCAL:
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
```

```python
# 训练设置

## CV
CV_STRATEGY = 'groupby_Sampling_Date'  # groupby_Sampling_Date
NFOLD = 5
KFOLD_SEED = 42

## Model
MODEL_NAME = 'efficientnet_b2'

## Training Hyper Params
LR = 1e-2
```

## Load Data

### 数据解读

#### 干物质（Dry Matter）
牧草含有大量水分，其含水量会因天气、时间、生长阶段等因素而剧烈变化。为了得到一个稳定、可比较的衡量标准来评估牧草的真实营养价值和数量，农学家和农民会先将牧草样本烘干，去除所有水分，然后称量剩余部分的重量。这个重量就是“干物质”或“干燥生物量”。比赛中的所有指标都是基于这个“干”重来计算的。

#### 1. Dry green vegetation (excluding clover) / 干燥的绿色植被（不包括三叶草）

*   **这是什么？**
    这主要是指牧场中正在生长的、绿色的禾本科牧草（如黑麦草、羊茅等）的干重。它代表了牧场中主要的、有活力的非豆科植物部分。
*   **为什么重要？**
    这是牲畜（如牛、羊）**最主要的能量来源**。这些绿色的草富含碳水化合物，为动物提供了日常活动和生长所需的基本能量。它的数量直接决定了牧场能养活多少动物。将其与三叶草分开测量，是因为它们的营养成分（特别是蛋白质含量）有显著差异。

#### 2. Dry dead material / 干燥的枯死物质

*   **这是什么？**
    这是指牧场中已经枯黄、死亡的植物部分的干重。它可能是前一个季节留下的老草，或是因干旱、过度成熟而死亡的植物。
*   **为什么重要？**
    这部分是**低质量的饲料**。它的营养价值非常低（蛋白质和能量含量都很低），消化率差，而且口感不好，牲畜通常会尽量避免采食。
    *   **指示作用**：如果这部分占比较高，说明牧场管理可能存在问题（例如，放牧不及时导致牧草长老、枯死），或者牧场健康状况不佳。它会“稀释”优质饲料的比例，降低牲畜的采食效率。
    *   **生态影响**：过多的枯死物质会覆盖在地面，阻碍阳光照射，抑制新草的生长。

#### 3. Dry clover biomass / 干燥的三叶草生物量

*   **这是什么？**
    这是指牧场中所有三叶草（Clover）或其他豆科植物（如苜蓿）的干重。
*   **为什么重要？**
    三叶草是牧场中的“**超级食物**”。
    *   **高蛋白质**：与禾本科牧草相比，三叶草的蛋白质含量要高得多。蛋白质是动物增重、产奶和维持健康的关键。因此，三叶草的含量直接关系到饲料的“质量”。
    *   **天然肥料**：三叶草具有“固氮”能力，能将空气中的氮气转化为土壤中的氮肥，为周围的禾本科牧草提供天然养分，从而提高整个牧场的生产力并减少对化肥的依赖。

#### 4. Green dry matter (GDM) / 绿色干物质

*   **这是什么？**
    这是 **(干燥的绿色植被) + (干燥的三叶草生物量)** 的总和。简单来说，它代表了牧场中所有**有生命的、绿色的植物**的总干重。
*   **为什么重要？**
    这是评估牧场**当前可用优质饲料总量**的核心指标。当农民决定一个牧区（paddock）可以放养多少头牛、能放养多少天时，他们最关心的就是GDM。它直接反映了牧场的“承载能力”（Carrying Capacity）。这个数值越高，意味着可供牲畜采食的优质饲料越多。

#### 5. Total dry biomass / 总干燥生物量

*   **这是什么？**
    这是牧场中所有地上部分生物量的总和，即 **(绿色植被) + (三叶草) + (枯死物质)** 的总干重。
*   **为什么重要？**
    这个指标反映了牧场上**所有植物物质的总量**。通过比较“总干燥生物量”和“绿色干物质（GDM）”，农民可以快速了解牧场的健康状况。
    *   **健康指标**：如果“总干燥生物量”很高，但“绿色干物质”占比很低，说明牧场里堆积了大量无用的枯死物质，需要进行管理（如通过短期重度放牧清理，或用机械割除）。
    *   **长期规划**：这个数据有助于了解牧场的季节性生长周期和整体生产力。

```python
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

from sklearn.model_selection import KFold, GroupKFold, StratifiedGroupKFold
from tqdm.auto import tqdm
tqdm.pandas()
```

```python
DATA_ROOT = '../input/' if LOCAL else '/kaggle/input/csiro-biomass/'
```

```python
train_df = pd.read_csv(f'{DATA_ROOT}/train.csv')
train_df.head()
```

```python
train_df[['sample_id_prefix', 'sample_id_suffix']] = train_df.sample_id.str.split('__', expand=True)
```

```python
(train_df.sample_id_suffix == train_df.target_name).all()
```

```python
cols = ['sample_id_prefix', 'image_path', 'Sampling_Date', 'State', 'Species', 'Pre_GSHH_NDVI', 'Height_Ave_cm']
agg_train_df = train_df.groupby(cols).apply(lambda df: df.set_index('target_name').target)
agg_train_df.reset_index(inplace=True)
agg_train_df.columns.name = None

agg_train_df['image'] = agg_train_df.image_path.progress_apply(
    lambda path: Image.open(DATA_ROOT + path).convert('RGB')
)

agg_train_df.head()
```

```python
agg_train_df['image_size'] = agg_train_df.image.apply(lambda x: x.size)
agg_train_df['image_size'].value_counts()
```

```python
np.isclose(
    agg_train_df[['Dry_Green_g', 'Dry_Clover_g']].sum(axis=1),
    agg_train_df['GDM_g'],
    atol=1e-04
).mean()
```

```python
np.isclose(
    agg_train_df[['GDM_g', 'Dry_Dead_g']].sum(axis=1),
    agg_train_df['Dry_Total_g'],
    atol=1e-04
).mean()
```

```python
plt.figure(figsize=(16, 4))
plt.subplot(1, 3, 1)
agg_train_df.Dry_Green_g.plot(kind='hist')
_ = plt.title('Dry_Green_g')

plt.subplot(1, 3, 2)
agg_train_df.Dry_Clover_g.plot(kind='hist')
_ = plt.title('Dry_Clover_g')

plt.subplot(1, 3, 3)
agg_train_df.Dry_Dead_g.plot(kind='hist')
_ = plt.title('Dry_Dead_g')
```

## CV Strategy

It appears there is a significant gap between the local CV score and the public LB score. We need to figure out the distributional differences between the test set and training set in order to select an appropriate CV strategy. Here are some discussions regarding the distribution of the test set.

1. [Is there data drift between the training data and the test data?](https://www.kaggle.com/competitions/csiro-biomass/discussion/613724)

    From **Competition Host**:
> The data split between the public and private sets is not completely random.
> 
> We used data from various seasons and states throughout the year as the training set. For validation and testing, we included some data from overlapping time periods and locations, **while also incorporating data from non-overlapping time periods to evaluate the model's generalization ability**.

2. [State and Species](https://www.kaggle.com/competitions/csiro-biomass/discussion/615003)

    From **Competition Host**:
> Hi there, images in the testing set contain the same State and Species

3. [Are There Any New Species in the Test Dataset?](https://www.kaggle.com/competitions/csiro-biomass/discussion/614083)
    From **Competition Host**:
> Great question. The direct answer is that all the dominant species in the test set are the same as those in the training set.
> 
> Please note that the "Species" column refers to the visually annotated dominant species in the images and does not include every species present (which would be practically impossible). Additionally, the "Species" column is only provided for the training set and is not available for the testing set (in case you didn't know it).

It appears the distribution differences between the test and training sets are **primarily reflected in the date**. Therefore, we could attempt using `Sampling_Date` as groups for K-fold cross-validation.

```python
agg_train_df['Sampling_Date_Month'] = agg_train_df.Sampling_Date.apply(lambda x: x.split('/')[1].strip())
```

```python
agg_train_df = agg_train_df.sort_index().sample(frac=1.0, random_state=31).copy()  # shuffle
```

```python
agg_train_df['idx'] = agg_train_df.index
```

```python
half_num = agg_train_df.shape[0] // 6
half_num
```

```python
head_df = agg_train_df.iloc[:half_num].reset_index(drop=True)
tail_df = agg_train_df.iloc[half_num:].reset_index(drop=True)
head_df.shape[0], tail_df.shape[0], len(set(head_df.idx) | set(tail_df.idx))
```

```python
agg_train_df['fold'] = None
```

```python
kfold = KFold(n_splits=NFOLD, shuffle=True, random_state=KFOLD_SEED)
for i, (trn_idx, val_idx) in enumerate(kfold.split(head_df.index, y=head_df.State, groups=head_df.Sampling_Date)):
    ori_val_idx = head_df.loc[val_idx, 'idx']
    agg_train_df.loc[ori_val_idx, 'fold'] = i
```

```python
kfold = StratifiedGroupKFold(n_splits=NFOLD, shuffle=True, random_state=KFOLD_SEED)
for i, (trn_idx, val_idx) in enumerate(kfold.split(tail_df.index, y=tail_df.State, groups=tail_df.Sampling_Date)):
    ori_val_idx = tail_df.loc[val_idx, 'idx']
    agg_train_df.loc[ori_val_idx, 'fold'] = i
```

```python
agg_train_df.sort_index(inplace=True)
```

```python
agg_train_df.fold.value_counts(dropna=False).sort_index()
```

```python
for i in range(NFOLD):
    trn_df = agg_train_df[agg_train_df.fold != i]
    val_df = agg_train_df[agg_train_df.fold == i]
    
    flag = val_df.Sampling_Date.isin(trn_df.Sampling_Date)
    print(f'trn({trn_df.shape[0]}) -> val({val_df.shape[0]}): {flag.mean()}')
```

## DataLoader

```python
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
```

```python
class RegressionDataset(Dataset):
    def __init__(self, data, transform=None, vertical_split=True):
        self.data = data
        self.transform = transform
        self.vertical_split = vertical_split

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        image = item.image
        targets = [item['Dry_Green_g'], item['Dry_Clover_g'], item['Dry_Dead_g']]
        
        if self.vertical_split:
            # 垂直均分成左右两张图片
            width, height = image.size
            mid_point = width // 2
            left_image = image.crop((0, 0, mid_point, height))
            right_image = image.crop((mid_point, 0, width, height))
            
            if self.transform:
                left_image = self.transform(left_image)
                right_image = self.transform(right_image)
            
            return left_image, right_image, targets
        
        else:
            if self.transform:
                image = self.transform(image)

            return image, targets


def create_dataloader(data, target_image_size=(256, 256), batch_size=32, shuffle=True, aug=True, tta_transform=None):    
    if aug:
        transform = transforms.Compose([
            transforms.Resize(target_image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomApply([transforms.RandomRotation([90, 90])], p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    else:
        if tta_transform:
            transform = transforms.Compose([
                transforms.Resize(target_image_size),
                tta_transform,
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(target_image_size),

                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    dataset = RegressionDataset(data, transform=transform)
    print('dataset size:', len(dataset))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4)
    return dataloader
```

```python
def get_tta_dataloaders(data, target_image_size, batch_size):
    res = []
    for transform in [None, transforms.RandomHorizontalFlip(p=1.0), transforms.RandomVerticalFlip(p=1.0), transforms.RandomRotation([90, 90])]:
        res.append(
            create_dataloader(data, target_image_size, batch_size, shuffle=False, aug=False, tta_transform=transform)
        )
    return res
```

## Model

```python
import timm
import torch
import torch.nn as nn
```

```python
model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=3)
```

```python
model.pretrained_cfg
```

```python
TARGET_IMAGE_SIZE = model.pretrained_cfg['input_size'][1:]
```

```python
class FiLM(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        hidden = max(32, feat_dim // 2)
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, feat_dim * 2)
        )

    def forward(self, context):
        gamma_beta = self.mlp(context)
        return torch.chunk(gamma_beta, 2, dim=1)
```

```python
class MultiTargetRegressor(nn.Module):
    def __init__(self, model_name, pretrained=True, num_classes=3, dropout=0.0, freeze_backbone=False):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0, global_pool='avg')
        
        self.film = FiLM(self.backbone.num_features)
        
        self.dropout = nn.Dropout(dropout)
        
        def make_head():
            return nn.Sequential(
                nn.Linear(self.backbone.num_features * 2, 8),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(8, 1),
            )

        self.head_green = make_head()
        self.head_clover = make_head()
        self.head_dead = make_head()
        
        self.softplus = nn.Softplus(beta=1.0)
    
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
    
    
    def forward(self, left_img, right_img):
        left_feat = self.backbone(left_img)
        right_feat = self.backbone(right_img)
        
        context = (left_feat + right_feat) / 2
        gamma, beta = self.film(context)
        
        left_feat_modulated = left_feat * (1 + gamma) + beta
        right_feat_modulated = right_feat * (1 + gamma) + beta
        
        combined = torch.cat([left_feat_modulated, right_feat_modulated], dim=1)
        
        green = self.softplus(self.head_green(combined))    # Bx1
        clover = self.softplus(self.head_clover(combined))  # Bx1
        dead = self.softplus(self.head_dead(combined))   # Bx1
    
        logits = torch.cat([green, clover, dead], dim=1)  # Bx3 

        return logits
```

## Train

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
```

```python
# ======== Weighted R² ========
def weighted_r2_score(y_true: np.ndarray, y_pred: np.ndarray):
    """
    y_true, y_pred: shape (N, 5): Green/Clover/Dead/GDM/Total
    """
    weights = np.array([0.1, 0.1, 0.1, 0.2, 0.5])
    r2_scores = []
    for i in range(5):
        y_t = y_true[:, i]
        y_p = y_pred[:, i]
        ss_res = np.sum((y_t - y_p) ** 2)
        ss_tot = np.sum((y_t - np.mean(y_t)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        r2_scores.append(r2)
    r2_scores = np.array(r2_scores)
    weighted_r2 = np.sum(r2_scores * weights) / np.sum(weights)
    return weighted_r2, r2_scores
```

```python
def calc_metric(outputs, targets):
    '''
        outputs/targets: shape (N, 3): Green/Clover/Dead
    '''
    y_true = np.column_stack((
        targets,
        targets[:, :2].sum(axis=1),
        targets.sum(axis=1),
    ))
    
    y_pred = np.column_stack((
        outputs,
        outputs[:, :2].sum(axis=1),
        outputs.sum(axis=1),
    ))
    
    weighted_r2, r2_scores = weighted_r2_score(y_true, y_pred)
    return weighted_r2, r2_scores
```

```python
def train_epoch(model, dataloader, criterion, optimizer, scheduler, device):
    model.train()
    total_loss = 0

    for left_images, right_images, targets in dataloader:
        left_images = left_images.to(device)
        right_images = right_images.to(device)
        targets = torch.stack(targets).T.float().to(device)

        optimizer.zero_grad()
        outputs = model(left_images, right_images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_outputs = []
    all_targets = []

    with torch.no_grad():
        for left_images, right_images, targets in dataloader:
            left_images = left_images.to(device)
            right_images = right_images.to(device)
            targets = torch.stack(targets).T.float().to(device)

            outputs = model(left_images, right_images)
            loss = criterion(outputs, targets)
            total_loss += loss.item()

            all_outputs.append(outputs.detach().cpu())
            all_targets.append(targets.detach().cpu())

    outputs = torch.cat(all_outputs).numpy()
    targets = torch.cat(all_targets).numpy()
    
    return total_loss / len(dataloader), outputs, targets
    
    # weighted_r2, r2_scores = calc_metric(outputs, targets)
    # return total_loss / len(dataloader), weighted_r2, r2_scores


def tta_validate(model, dataloaders, criterion, device):
    if not isinstance(dataloaders, list):
        dataloaders = [dataloaders]
    
    all_loss = []
    all_outputs = []
    all_targets = []
    for dataloader in dataloaders:
        loss, outputs, targets = validate(model, dataloader, criterion, device)
        all_loss.append(loss)
        all_outputs.append(outputs)
        all_targets.append(targets)
    
    avg_loss = np.mean(all_loss)
    avg_outputs = np.mean(all_outputs, axis=0)
    avg_targets = np.mean(all_targets, axis=0)
    
    weighted_r2, r2_scores = calc_metric(avg_outputs, avg_targets)
    
    return avg_loss, weighted_r2, r2_scores
```

```python
def train_fold(data, fold, batch_size=32, continue_training=False):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # Hyperparameters
    # batch_size = 8
    lr = LR
    patience = 10
    num_epochs = 5 if DEBUG else 100
    warmup_ratio = 0.05

    # data
    train_loader = create_dataloader(data[data.fold != fold], TARGET_IMAGE_SIZE, batch_size, shuffle=True, aug=True)
    val_loader = create_dataloader(data[data.fold == fold], TARGET_IMAGE_SIZE, batch_size, shuffle=False, aug=False)
    # val_loaders = get_tta_dataloaders(data[data.fold == fold], TARGET_IMAGE_SIZE, batch_size)

    # model, loss, optimizer
    # model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=3)
    if continue_training:
        model = MultiTargetRegressor(MODEL_NAME, pretrained=False, num_classes=3, freeze_backbone=False)
        model_file = f'{OUTPUT_DIR}/best_mode_fold{fold}.pth'
        model.load_state_dict(torch.load(model_file))
        
    else:
        model = MultiTargetRegressor(MODEL_NAME, pretrained=True, num_classes=3, freeze_backbone=True)
        
    model.to(device)

    criterion = nn.SmoothL1Loss()  # nn.MSELoss()
    optimizer = AdamW(model.parameters(), lr=lr / 500 if continue_training else lr)

    num_training_steps = num_epochs * len(train_loader)
    warmup_steps = int(warmup_ratio * num_training_steps)
    
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=patience//2)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_training_steps)

    # Training loop
    history = []
    best_score = -float('inf')
    # best_loss = float('inf')
    epochs_without_improvement = 0
    
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, scheduler, device)
        # val_loss, weighted_r2, r2_scores = validate(model, val_loader, criterion, device)
        val_loss, weighted_r2, r2_scores = tta_validate(model, val_loader, criterion, device)

        if epoch % 10 == 0:
            print(f"Epoch [{epoch}/{num_epochs}]: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, weighted_r2: {weighted_r2:.4f}, lr: {optimizer.param_groups[0]['lr']}")
        
        history.append({
            'train_loss': train_loss,
            'val_loss': val_loss,
            'lr': optimizer.param_groups[0]['lr'],
            'weighted_r2': weighted_r2,
            'r2_scores': r2_scores,
        })
        
        # 早停
        if weighted_r2 > best_score:
        # if val_loss < best_loss:
            best_loss = val_loss
            best_score = weighted_r2
            epochs_without_improvement = 0
            # 保存最佳模型
            torch.save(model.state_dict(), f'{OUTPUT_DIR}/best_mode_fold{fold}.pth')
        
        else:
            epochs_without_improvement += 1
        
        if epochs_without_improvement >= patience:
            print(f"早停: epoch={epoch}, {patience} 个 epoch 无改善")
            break

    print(f"\nTraining completed. Best weighted_r2: {best_score:.4f}")
    
    return history, best_score
```

```python
OUTPUT_DIR = 'trained_models/'
if not os.path.exists(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)
```

### Stage 1

```python
if TRAIN:
    all_best_score = []

    for i in range(2 if DEBUG else NFOLD):
        print(f'### fold={i}')
        history, best_score = train_fold(agg_train_df, fold=i, batch_size=32, continue_training=False)
        all_best_score.append(best_score)
        history = pd.DataFrame(history)

        history.to_json(
            f'{OUTPUT_DIR}/history_fold{i}.jsonl',
            orient='records',
            lines=True,
            force_ascii=False,
        )

        # plot
        plt.figure(figsize=(16, 4))

        plt.subplot(1, 3, 1)
        plt.title('LR')
        plt.plot(history.lr)

        plt.subplot(1, 3, 2)
        plt.title('Loss')
        plt.plot(history.train_loss, label='train')
        plt.plot(history.val_loss, label='val')
        plt.legend()

        plt.subplot(1, 3, 3)
        plt.title('weighted_r2')
        plt.plot(history.weighted_r2)
        plt.show()
    
    print('Avg CV:', np.mean(all_best_score))
```

### Stage2

```python
if TRAIN:
    all_best_score = []

    for i in range(2 if DEBUG else NFOLD):
        print(f'### fold={i}')
        history, best_score = train_fold(agg_train_df, fold=i, batch_size=8, continue_training=True)
        all_best_score.append(best_score)
        history = pd.DataFrame(history)

        history.to_json(
            f'{OUTPUT_DIR}/history_fold{i}.jsonl',
            orient='records',
            lines=True,
            force_ascii=False,
        )

        # plot
        plt.figure(figsize=(16, 4))

        plt.subplot(1, 3, 1)
        plt.title('LR')
        plt.plot(history.lr)

        plt.subplot(1, 3, 2)
        plt.title('Loss')
        plt.plot(history.train_loss, label='train')
        plt.plot(history.val_loss, label='val')
        plt.legend()

        plt.subplot(1, 3, 3)
        plt.title('weighted_r2')
        plt.plot(history.weighted_r2)
        plt.show()
    
    print('Avg CV:', np.mean(all_best_score))
```

## Inference

```python
import os
from pathlib import Path
```

```python
def get_lastest_saved_models():
    model_root = '/kaggle/input/csiro-simple-output/pytorch/default/'
    
    latest = 1
    for version in os.listdir(model_root):
        try:
            version = int(version)
        except:
            continue

        if version > latest:
            latest = version

    return f'{model_root}/{latest}/trained_models/'
```

```python
SAVED_MODELS = './trained_models/' if TRAIN else get_lastest_saved_models()
```

```python
def predict(model, dataloader, device):
    model.to(device)
    model.eval()

    all_outputs = []
    with torch.no_grad():
        for left_images, right_images, targets in dataloader:
            left_images = left_images.to(device)
            right_images = right_images.to(device)
    
            outputs = model(left_images, right_images)
            all_outputs.append(outputs.detach().cpu())
    
    outputs = torch.cat(all_outputs).numpy()
    return outputs
```

```python
def tta_predict(model, dataloaders, device):
    all_outputs = []
    for dataloader in dataloaders:
        outputs = predict(model, dataloader, device)
        all_outputs.append(outputs)
    avg_outputs = np.mean(all_outputs, axis=0)
    return avg_outputs
```

```python
def kfold_predict(dataloaders):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    all_preds = []
    for model_file in Path(SAVED_MODELS).glob('*.pth'):
        # model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=3)
        fold_idx = int(model_file.name.split('.')[0].split('fold')[1])
        if fold_idx >= NFOLD: continue
        print(model_file.name)
        model = MultiTargetRegressor(MODEL_NAME, pretrained=False, num_classes=3)
        model.load_state_dict(torch.load(model_file))

        preds = tta_predict(model, dataloaders, device)
        all_preds.append(preds)

    avg_preds = np.mean(all_preds, axis=0)
    return avg_preds
```

```python
test_df = pd.read_csv(DATA_ROOT + 'test.csv')

test_df['target'] = 0.0
test_df[['sample_id_prefix', 'sample_id_suffix']] = test_df.sample_id.str.split('__', expand=True)
```

```python
cols = ['sample_id_prefix', 'image_path']
agg_test_df = test_df.groupby(cols).apply(lambda df: df.set_index('target_name').target)
agg_test_df.reset_index(inplace=True)
agg_test_df.columns.name = None

agg_test_df['image'] = agg_test_df.image_path.progress_apply(
    lambda path: Image.open(DATA_ROOT + path).convert('RGB')
)

agg_test_df.head()
```

```python
test_loader = get_tta_dataloaders(agg_test_df, TARGET_IMAGE_SIZE, 64)
```

```python
preds = kfold_predict(test_loader)
```

```python
agg_test_df[['Dry_Green_g', 'Dry_Clover_g', 'Dry_Dead_g']] = preds
agg_test_df['GDM_g'] = agg_test_df.Dry_Green_g + agg_test_df.Dry_Clover_g
agg_test_df['Dry_Total_g'] = agg_test_df.GDM_g + agg_test_df.Dry_Dead_g
```

```python
agg_test_df.head()
```

```python
cols = ['Dry_Clover_g', 'Dry_Dead_g', 'Dry_Green_g', 'Dry_Total_g', 'GDM_g']
sub_df = agg_test_df.set_index('sample_id_prefix')[cols].stack()
sub_df = sub_df.reset_index()
sub_df.columns = ['sample_id_prefix', 'target_name', 'target']

sub_df['sample_id'] = sub_df.sample_id_prefix + '__' + sub_df.target_name
```

```python
cols = ['sample_id', 'target']
sub_df[cols].to_csv('submission.csv', index=False)
```

```python
!head submission.csv
```