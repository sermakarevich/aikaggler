# Summary of solutions (1~43)

- **Author:** suguuuuu
- **Date:** 2025-02-19T14:36:12.140Z
- **Topic ID:** 563862
- **URL:** https://www.kaggle.com/competitions/czii-cryo-et-object-identification/discussion/563862
---

I have summarized the solutions from several perspectives.
There may be mistakes.

My summary report in japanese.
https://speakerdeck.com/sugupoko/cziikonpezhen-rifan-ri-guan-xi-kagglerhui-jiao-liu-hui-in-osaka-2025-number-1

markdown table

| Rank | Link | Architecture | Task | Loss | Resolution | Pretraining | Extra Data | EMA/SWA | Ensemble / TTA | Key Point ? |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1st | Seg：kaggle.com; Det：Kaggle | 3D MONAIs FlexibleUnet; 3D Yolo | Seg & det | Seg: weighted CrossEntropy (pos256: neg 1); Det: PP-Yolo loss function with modifications | inference is lager | no | no | no | merge of different types with scaling of conf | diversity of strategies |
| 2nd | Kaggle | 3D; Various | seg | 1. Dice Loss, Tversky Loss and Cross-Entropy Loss; 2. Tversky Loss and Cross-Entropy Loss | inference is lager | no | no | no | 10 models; 7TTA | diversity of model; Change Norm layer |
| 3rd | kaggle.com | 3D Unet; Resnet101 | seg | Cross Entropy loss | Train: 64,128,128; Inference: 64,256,256 | no | yes | EMA; 0.995 | 4 model (7fold中の）; flip x,y,z; rot90 for x,y | Data diversity with strong models?; model soup |
| 4th | Kaggle | 2.5D Enc - 3D dec | heatmap | Weighted MSE | Train: 32×128×128; Inference: 32×128×128 | no | no | EMA/SWA | 7models(4+3); TTA: yes | diversity of model |
| 5th | kaggle.com | 3D | seg | Label smoothing cross-entropy | Train: (128,128,128); Inference | no | no | no | 4seed; TTA: flip 3, rot 3 | DeepFinder-like network; Make the model smaller and TTA |
| 6th | カグル | 2.5+3D; Various | seg | FocalTversky++ | Train: 64×128×128; Inference: 64×128×128 | Yes | Created by myself using Polnet. | no | 10 models |  |
| 7th | kaggle.com | 3DUnet; resnet50d, resnet50d, efficientnetv2-m | heatmap | weighted BCE (heavy pos weight) | Train: not mentioned; Inference: 192,128,128 | Yes | yes | ModelEMA | 3model, model soup; Average at logits; 4x Flip TTA | Smoothing before peak detection; model soup |
| 8th | Kaggle | 3D Unet | seg | DiceCELoss | Train: (128,128,128); Inference: (160,384,384) | Yes | Yes; Only 6 tomography | EMA used (decay 0.99) (Used or not used) | 4model, model soup; Average at logits | model soup; Number of channels and data usage vary |
| 9th | kaggle.com | 3D Unet; ConvNeXt-like | seg | BCE | Train: (32,256,256); Inference: (32,320,320) | no | no | no | TTA Rot90, 180, 270 | Small radius; Convnext |
| 10th | Kaggle | 3D Unet; monai | seg | Tversky loss and multiclass crossentropy | Train: not mentioned; Inference: not mentioned | Yes | no; (include implementation bug) | no | Average at logits | Original Post Processiung |
| 11th | Kaggle | 3D Unet; Resnet arch | seg | weighted combination of cross-entropy and Tversky loss (alpha=0.5, beta=8) | Train: 128x256x256?; Inference: 128x256x256 | no | no | no | 32models; Seed ensemble; Average at logits; NoTTA | PP 2D classification; InstanceNorm to batchNorm |
| 12th | Kaggle | 3types Unet | seg | dice, focal | Various patch size | no | yes | (not mentioned) | 10 models; different tta combinations | radius * 0.5; drop path |
| 13th | Kaggle | 3D unet with aux heads; Pixel shuffle dec; 2x r3d50, 2x r3d34, 1x r3d18 backbones | seg | CE?; (not mentioned) | Train/Inference: ?; 32,128,128; Using center 16,64,64 | No | no; (not mentioned) | EMA | ??; Cutpast; 2stage training | adding Stochastic Depth and DropBlock. |
| 22nd | Kaggle | 2d-3d Unet | seg | BCE + 2×TverskyLoss | Train/inference: 48×352×352 or 48×320×320 | yes | no | (not mentioned) | WBF-based post-processing | Post Processing using 2d classification |
| 23rd | Kaggle | 3d Unet | seg | BCE | 64,64,64 | No | no | (not mentioned) | 5model? |  |
| 26th | Kaggle | 3d Unet | seg | Weighted BCE and Tversky | 184x184x184 | No | no | (not mentioned) | Average at logits; Rot and flip 5TTA | Val TS_99_9 |
| Only Me |  | 3D Unet | heatmap | BCE | 128x128x128 | Yes | no | no | 10 models |  |
| Private about 30th |  | Resnext101 & effnetb4 & b5 |  |  |  |  |  |  | Flip TTA |  |
| 32nd | Kaggle | 3D Unet; 3 arch | seg | Weighted Tversky Loss; Distance Loss | 48,256,256 | No | no; (not mentioned) | EMA | 8model; 7 TTA |  |
| 33rd | Kaggle | 3D unet + 2d yolo | seg | CE | 128x128x128 | No | no; (not mentioned) | (not mentioned) | DBSCAN? |  |
| 43rd | Kaggle | WMCSFB; ZhaoWenzhao/WMCSFB | seg | Tverski loss that supports class weights (0.25,1.0,1.0,1.0,4.0,4.0) | 64x64x64 | No | no; (not mentioned) | (not mentioned) | DBSCAN | 1/3 or the particle radius |


image 

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F2930242%2Fd10b3ff877cc870db49e6ad63df552e5%2F2025-02-19%20233331.jpg?generation=1739975625923694&alt=media)

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F2930242%2Fd30dcae156d273d86f3bd5ad841ee566%2F2025-02-19%20233321.jpg?generation=1739975642377438&alt=media)

