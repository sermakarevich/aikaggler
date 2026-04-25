# byu-locating-bacterial-flagellar-motors-2025: cross-solution summary

This cryo-ET tomography competition focused on localizing bacterial flagellar motors in 3D volumes, where winning approaches heavily relied on GPU-accelerated augmentations, voxel resampling to fixed spacings, and robust post-processing like quantile thresholding or graph-based clustering. Top solutions diverged between dense segmentation/classification pipelines (3D U-Nets, ResNets) and keypoint/detection frameworks (YOLO variants, GraphSAGE), with all authors ultimately abandoning strict local cross-validation in favor of direct leaderboard-driven iteration and model soup or fusion ensembling.

## Competition flows
- Load/resize tomograms -> Gaussian heatmap labels -> 3D U-Net training -> depth-sliding window inference -> 8-seed ensemble -> quantile thresholding
- Scale/crop volumes -> 3D-ResNet18 classifier training -> sliding window inference -> quantile thresholding -> JIT/TensorRT acceleration
- Resample to 16.0 Å -> Gaussian heatmaps -> 3D/2D UNet training -> NMS decoding -> WBF + quantile thresholding
- YOLO11 feature extraction -> keypoint generation + RandomWalkPE -> dual k-NN/Delaunay graph -> GraphSAGE training -> maximum ensemble
- YOLO dataset creation -> yolov8l/yolo11l training -> multi-process FP16 inference -> DFS coordinate aggregation -> slice-skipping
- Trust-score filtering -> ConvNeXtV2-based YOLO training -> model soup averaging -> HDBSCAN clustering

## Data processing
- Resized to (128, 704, 704) via scipy.ndimage.zoom()
- Discarded tomograms without motors
- Manually added missing motors using Napari
- Generated 8x downsampled Gaussian heatmap labels
- Applied heavy augmentations (Mixup, Rescale/Zoom, rotations, flips, axis swaps, coarse dropout, color inversion, cutmix) primarily on GPU
- Incorporated external data from the CryoET Data Portal
- 3D images scaled to fixed voxel size of 15.6 and saved as int8
- External data from @brendanartley incorporated
- RandomCrop (96x160x160) via MONAI
- Flip on each axis within torch dataloader
- GPU-accelerated scale and rotation augmentations
- Custom MixUp ensuring max 1 motor per mixed patch
- Positive sample oversampling to 12.5% fraction
- Resampling all tomograms to fixed 16.0 Å voxel spacing
- Sliding window patching (3D: 224x448x448; 2D: 896x896)
- Gaussian heatmap generation with sigma=200Å, dynamically rescaled during augmentations
- Heavy 3D augmentations via MONAI (RandZoom, RandAffined, intensity shifts, RandSimulateLowResolutiond, RandCoarseDropoutKeepKeypoints)
- Heavy 2D augmentations via Albumentations (noise, downscaling, geometric transforms, safe cropping)
- External dataset integration: pseudo-labeling with low threshold (0.05), manual CVAT review of FPs/FNs, coordinate correction, re-spacing to max(16.0, ori_spacing)
- Positive/negative patch ratio control (20/1 for 3D, 0.75 for 2D)
- YOLO11 outputs filtered with low confidence threshold (conf=0.05)
- Sparse keypoints densified by uniformly sampling additional points within a specified radius around each prediction
- External data from Kaggle dataset
- YOLO dataset creation with TRUST=8 (train) / TRUST=2 (external) / TRUST=4 (val)
- Slices resized to 960x960
- Vertical, horizontal, and horizontal-vertical flipping augmentations on competition data only
- BOX_SIZE set to int(1000 / voxel_spacing) for competition data and 35 for external data
- 4-fold split
- Filtered training data to include only images with motors at trust=4, supplemented with 80% of an external dataset (trust=0)
- Applied standard YOLO augmentations (mosaic=0.5, mixup=0.4, flipud=0.5, scale=0.25, degrees=45) and a secondary heavy augmentation pipeline using Albumentations (Blur, MedianBlur, ToGray, CLAHE, RandomBrightnessContrast, RandomGamma, ImageCompression, ShiftScaleRotate, GaussNoise, GaussianBlur, UnsharpMask)
- Input image size was set to 960x960

## Features engineering
- C3K2 feature maps extracted from YOLO11 spatial positions
- AddRandomWalkPE (walk_length=8) for implicit positional encoding
- Feature similarity metrics and spatial radius filtering for dynamic label generation

## Models
- 3D U-Net
- ResNet200
- ResNet101
- 3D-ResNet18
- 3D UNet
- 3D ResNeXt50
- 3D DenseNet121
- 3D X3D-M
- 2D MaxViT
- 2D CoaT
- EfficientNet
- ConvNeXt
- I3D
- ResNet50
- ConvNeXt Tiny
- CoaT Lite Medium
- MaxViT Tiny
- YOLO11
- GraphSAGE
- YOLOv8-L
- YOLO11-L
- YOLO10-X
- ConvNeXtV2-Base
- EfficientNetV2-L
- ResNeXt-50
- CaFormer-B36
- ConvNeXt-Large

## Frameworks used
- MONAI
- PyTorch
- TensorRT
- Albumentations
- timm
- segmentation_models_pytorch_3d
- OpenCV
- PyTorch Geometric
- Ultralytics

## Loss functions
- SmoothBCE loss
- deep supervision loss
- max pooled loss
- CrossEntropy Loss
- DenseCrossEntropy1D
- BCE
- Binary Cross Entropy (positive_weight=8)
- DFL Loss

## CV strategies
- 4-fold cross-validation, switching to public LB validation beyond ~0.93 local CV score
- 4-fold grouped split: training data split by Voxel Size, external data split by dataset ID
- 4-StratifiedKFold by motor count, later shifted to training on all data with LB validation
- Fold-wise training
- 4-fold datasets and hold-out (8:2) dataset
- Split 80% of trust=4 images and 80% of external trust=0 images for training, with the remainder used as validation; evaluated using mAP50, mAP50-95, precision, and recall curves

## Ensembling
- The final submission uses an 8-seed ensemble where sigmoid is applied to each model output and the logits are summed, followed by ranking tomograms by their maximum predicted pixel value and applying quantile thresholding to determine motor presence
- The final submission combines the 3D-ResNet18 classifier with object detection models from MONAI, accelerated via JIT or TensorRT exports, and applies quantile-based thresholding for stable predictions across models
- Ensemble of 5 models with 2xTTA, decoded via NMS, and combined using Weighted Box Fusion (WBF) with L2 distance clustering and average confidence mode
- Predictions from multiple models/folds are combined using a maximum ensemble function, though the author notes HDBSCAN clustering performed better in certain scenarios
- The ensemble aggregates per-slice detection coordinates across all 8 models (yolov8l and yolo11l 4-folds) for joint DFS processing, accelerated by a slice-skipping strategy that randomly selects one slice out of every four per fold to cut inference time
- Combined three models via model soup weight averaging across multiple epochs, followed by HDBSCAN-based prediction clustering with size-dependent confidence adjustments and a 0.4 YOLO confidence threshold

## Insights
- Using quantile thresholding is critical for obtaining reliable feedback from the leaderboard
- Downsampling the Gaussian heatmap labels by 8x aligns well with the metric's high tolerance for distance error
- Increasing the encoder's capacity directly improves model performance
- Offloading augmentations to the GPU bypasses slow disk I/O bottlenecks during training
- The competition metric's forgiveness regarding localization precision justified simplifying a complex UNet decoder into a pure encoder with a classification head
- A custom MixUp augmentation that restricts mixed patches to at most one motor was highly effective for preventing overfitting during extended training
- Quantile-based thresholding provided significantly more stable predictions across different models compared to fixed thresholds
- Resampling to ~16.0 Å voxel spacing optimally balances detail preservation and computational load
- Fusing low-level features via a simple FPN neck significantly improves information propagation to coarse output maps
- Hierarchical Conv-Transformer backbones outperform pure CNNs for 2D tasks, emphasizing the importance of global context
- Heavy augmentations enable longer training epochs and improve robustness to domain shifts
- Pseudo-labeling combined with manual review of external data drastically improves model stability and recall
- Using a low confidence threshold on YOLO11 to generate abundant candidate keypoints significantly boosts downstream graph model performance
- Adding RandomWalk positional encodings to node features is critical for 3D spatial tasks, as it implicitly models point positions
- Combining k-NN and Delaunay triangulation graphs balances local density with long-range connections, yielding better generalization than either graph alone
- Avoiding external datasets prevents annotator bias mismatch, which is crucial when label semantics depend on specific human experts
- Increasing image size to 960 and batch size to 32 significantly improved LB scores
- Incorporating external cryo-ET data and targeted augmentations boosted performance
- Using the best DFL loss epoch model provided more stable scores than the standard best.pt checkpoint
- DFS-based coordinate aggregation effectively resolves 3D motor locations from 2D slice detections
- Leveraging the Ultralytics YOLO framework with custom Timm backbones outperformed traditional U-Nets and simple classification/segmentation heads
- Removing the P5 feature scale and incorporating P2 features via adaptive pooling significantly improved detection performance
- Model soup averaging proved necessary to mitigate unstable validation metrics and poor epoch selection correlation with the leaderboard
- P2 features were critically important for this task, as evidenced by the superior performance of the second model configuration

## Critical findings
- Fixed thresholds proved highly unstable for this competition's metric, necessitating a quantile-based approach
- The optimal quantile threshold remained nearly identical between the public and private leaderboards (0.565 vs 0.560), validating the thresholding strategy
- Surprisingly, a simple 3D-ResNet18 encoder outperformed more complex architectures and the initial 3D UNet approach when trained from scratch
- Cross-validation correlation with the leaderboard was poor, making the LB score the primary feedback mechanism for iteration rather than validation metrics
- The optimal target voxel spacing (16.0 Å) differs from the average test spacing (~13.1 Å), yet yields the highest LB score
- CV saturated and lost correlation with LB when scores exceeded ~0.96, making local validation unreliable for final hyperparameter tuning
- Mixup and Cutmix augmentations provided no clear improvement and were discarded
- Training on all data without local validation in the final week proved more effective than strict CV monitoring
- Pure keypoints generation or larger models failed to improve scores, whereas the specific combination of C3K2 features and RandomWalkPE caused an instant score boost
- YOLO11L produced more false positives but achieved a higher private score (0.843) than YOLO11X, which had a better public score but lower private score, highlighting the risk of overfitting to public LB
- Labeling points based on feature similarity within a spatial radius (rather than exact point matches) better captures the semantic tolerance provided by the competition metric
- yolov8l achieved a better LB score than yolo11l despite yolo11l being newer
- Slice-skipping strategy drastically cut inference time from 6+ hours to ~3 hours with minimal score impact
- TTA and WBF failed to improve results and were explicitly abandoned
- Larger architectures like ConvNeXt-Large actually performed worse than ConvNeXt-Base, contradicting the expectation that bigger models always help
- The team's true best private score (0.856) came from a single model configuration with a different epoch soup, revealing that their chosen ensemble strategy was suboptimal
- Validation metrics (mAP50, mAP50-95, precision, recall) showed almost no correlation with public leaderboard performance, making checkpoint selection highly unreliable

## What did not work
- Other architectures and alternatives performed much worse than the simple 3D-ResNet18
- Other losses: MSE, L1, Weighted BCE, Focal, Tversky, and combining multiple losses
- Mixup and Cutmix augmentations
- Injecting LSTM layers after 2D features (CV went up but LB unchanged)
- Predicting voxel spacing on test data (MAE=5.1, not useful)
- All segmentation models
- Pure keypoints generation
- Larger models
- TTA
- WBF
- Testing alternative backbones (EfficientNetV2-L, ResNeXt-50, CaFormer-B36) yielded ~0.8 performance but failed to surpass ConvNeXt-Base
- Using ConvNeXt-Large degraded performance compared to the Base variant
- Replacing the detection head with ConvNeXt blocks improved mAP50-95 but did not translate to leaderboard gains
- Relying on standard validation metrics and single-epoch checkpoint selection led to poor correlation with public leaderboard scores

## Notable individual insights
- Rank 1 (1st Place - 3D U-Net + Quantile Thresholding): Quantile thresholding on max predicted pixel values is critical for stable leaderboard feedback, as fixed thresholds proved highly unstable.
- Rank 2 (4th place: Simple ResNet18 classification): Simplifying a complex UNet decoder into a pure encoder was justified by the metric's forgiveness regarding localization precision.
- Rank 3 (3rd place solution: 3D/2D UNet with Gaussian Heatmap and WBF): Resampling to ~16.0 Å voxel spacing optimally balances detail preservation and computational load, outperforming the average test spacing.
- Rank 4 (20th place solution -- Keypoint-Based Dual-Graph Predictor): Combining k-NN and Delaunay triangulation graphs balances local density with long-range connections, yielding better generalization than either alone.
- Rank 5 (369th place solution YOLO part with PB0.840 notebook): Slice-skipping strategy drastically reduced inference time from 6+ hours to ~3 hours with minimal score impact.
- Rank 6 (17th Place Solution - Ultralytics + Timm): Removing the P5 feature scale and incorporating P2 features via adaptive pooling significantly improved detection performance in YOLO architectures.

## Solutions indexed
- #1 [[solutions/rank_01/solution|1st Place - 3D U-Net + Quantile Thresholding]]
- #3 [[solutions/rank_03/solution|3rd place solution: 3D/2D UNet with Gaussian Heatmap and  WBF]]
- #4 [[solutions/rank_04/solution|4th place: Simple ResNet18 classification]]
- #17 [[solutions/rank_17/solution|17th Place Solution - Ultralytics + Timm]]
- #20 [[solutions/rank_20/solution|20th place solution -- Keypoint-Based Dual-Graph Predictor]]
- #369 [[solutions/rank_369/solution|369th place solution YOLO part with PB0.840 notebook]]

## GitHub links
- [brendanartley/BYU-competition](https://github.com/brendanartley/BYU-competition) _(solution)_ — from [[solutions/rank_01/solution|1st Place - 3D U-Net + Quantile Thresholding]]
- [kenshohara/3D-ResNets-PyTorch](https://github.com/kenshohara/3D-ResNets-PyTorch) _(library)_ — from [[solutions/rank_01/solution|1st Place - 3D U-Net + Quantile Thresholding]]
- [ChristofHenkel/kaggle-cryoet-1st-place-segmentation](https://github.com/ChristofHenkel/kaggle-cryoet-1st-place-segmentation) _(reference)_ — from [[solutions/rank_04/solution|4th place: Simple ResNet18 classification]]
- [MIC-DKFZ/nnUNet](https://github.com/MIC-DKFZ/nnUNet) _(reference)_ — from [[solutions/rank_03/solution|3rd place solution: 3D/2D UNet with Gaussian Heatmap and  WBF]]
- [ZFTurbo/segmentation_models_pytorch_3d](https://github.com/ZFTurbo/segmentation_models_pytorch_3d) _(library)_ — from [[solutions/rank_03/solution|3rd place solution: 3D/2D UNet with Gaussian Heatmap and  WBF]]
- [ZFTurbo/timm_3d](https://github.com/ZFTurbo/timm_3d) _(library)_ — from [[solutions/rank_03/solution|3rd place solution: 3D/2D UNet with Gaussian Heatmap and  WBF]]
- [facebookresearch/SlowFast](https://github.com/facebookresearch/SlowFast) _(library)_ — from [[solutions/rank_03/solution|3rd place solution: 3D/2D UNet with Gaussian Heatmap and  WBF]]
- [albumentations-team/albumentations](https://github.com/albumentations-team/albumentations) _(library)_ — from [[solutions/rank_03/solution|3rd place solution: 3D/2D UNet with Gaussian Heatmap and  WBF]]
- [cvat-ai/cvat](https://github.com/cvat-ai/cvat) _(reference)_ — from [[solutions/rank_03/solution|3rd place solution: 3D/2D UNet with Gaussian Heatmap and  WBF]]
- [dangnh0611/kaggle_byu](https://github.com/dangnh0611/kaggle_byu) _(solution)_ — from [[solutions/rank_03/solution|3rd place solution: 3D/2D UNet with Gaussian Heatmap and  WBF]]
- [tom99763/21th-place-solution-BYU](https://github.com/tom99763/21th-place-solution-BYU) _(solution)_ — from [[solutions/rank_20/solution|20th place solution -- Keypoint-Based Dual-Graph Predictor]]
- [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics) _(library)_ — from [[solutions/rank_17/solution|17th Place Solution - Ultralytics + Timm]]
- [yang-0201/MHAF-YOLO](https://github.com/yang-0201/MHAF-YOLO) _(library)_ — from [[solutions/rank_17/solution|17th Place Solution - Ultralytics + Timm]]

## Papers cited
- [3D PAN (Path Aggregation Network)](https://arxiv.org/abs/1803.01534)
- [FPN](https://arxiv.org/pdf/1612.03144)
- [Unified FPN architecture](http://presentations.cocodataset.org/COCO17-Stuff-FAIR.pdf)
- [Model Soup](https://arxiv.org/abs/2203.05482)
