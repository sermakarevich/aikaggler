# csiro-biomass: cross-solution summary

This biomass regression competition challenged participants to predict pasture biomass from limited, artifact-prone imagery using self-supervised vision models. Winning approaches predominantly leveraged DINOv3 backbones with innovative architectural adaptations like multi-stream processing, visual prompt tuning, and multi-modal tabular fusion. Success hinged on aggressive data cleaning, synthetic/pseudo-label augmentation, carefully engineered cross-validation splits, and robust post-processing rather than complex loss functions or external data.

## Competition flows
- Left/right view splitting with shared backbone & attention fusion
- Timestamp artifact removal & multi-stream ViT with MoE heads
- Stratified CV with pseudo-labeled synthetic data & DPT density estimation
- Two-stage multi-modal pipeline fusing visual/tabular features with auxiliary tabular prediction
- Two-stage pipeline with Stage 1 regression & Stage 2 Test-Time Training (TTT) using pseudo-labels
- Frozen backbone with visual prompt tuning & multi-crop tiling for dense prediction

## Data processing
- Splitting images into left/right views
- HSV timestamp removal via cv2.inpaint
- Manual cropping of cardboard edges
- Synthetic data generation (Qwen Image Edit) & pseudo-labeling
- Multi-crop training & random cropping (448x448, 1024x1024, 1280x1280, 2048x1024, 800x800)
- HorizontalFlip, VerticalFlip, RandomRotate90, RandomGamma, RandomBrightnessContrast, HueSaturationValue, CLAHE/Equalize, Sharpen/Emboss, RandAugment, RandomErasing, Mixup, CutMix
- Random scaling with black padding
- ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
- StandardScaler normalization for targets/features
- State-based range clipping & WA-state constraint enforcement
- Empirical scaling adjustments (e.g., Clover * 0.8, thresholding <0.2)
- TTA (horizontal/vertical flips, mean aggregation)

## Features engineering
- Auxiliary regression targets (NDVI, average pasture height)
- Auxiliary classification targets (species, state)
- Meta feature embeddings for state/species
- Pre_GSHH_NDVI
- Height_Ave_cm

## Models
- DINOv3 (ViT-Large, ViT-Huge, DINO-Large, DINO-Base, DINO-Huge, vit_large_patch16_dinov3_qkvb, vit_huge_plus_patch16_dinov3.lvd1689m)
- DINOv2
- ConvNeXt-Large
- XCiT
- SigLIP
- EVA02
- SAM 2
- Depth Anything v3
- SegDINO DPT head
- PE spatial (Perception Encoder)
- BiomassModelBasic
- BiomassModelWithStateSpecies
- BiomassModelWithMetaEmbedding
- MetaMoE
- _AuxHead
- Multi-Modal Regression Network
- 2-layer MLP
- Hybrid Texture Pooling
- DETR-style query tokens
- Visual Prompt Tuning

## Frameworks used
- pytorch/torch
- albumentations
- opencv/cv2
- numpy
- pandas
- timm
- schedule-free
- ThreadPoolExecutor

## Loss functions
- SmoothL1Loss
- CrossEntropyLoss
- EpsilonInsensitiveLoss
- GaussianNLLLoss
- PhysicsConsistencyLoss
- WeightedSmoothL1Loss
- MSELoss
- r^2 loss
- orthogonality loss

## CV strategies
- 3-fold CV jointly by state and sampling date
- Full-dataset training (no CV)
- Stratified 5-fold CV by state with seed selection for label distribution matching
- 5-fold CV with specific seeds per stage
- StratifiedGroupKFold with subdivided virtual dates and brute-force seed search (10M seeds)
- State-stratified GroupKFold over sampling dates

## Ensembling
- SWA ensemble of pseudo-labeled models
- Weighted fusion of multiple models (e.g., 0.4/0.6 splits)
- Weighted ensemble of six models with TTA averaging
- Ensemble of separate target models trained at different convergence points
- 5-fold CV prediction aggregation
- Weighted blend of multi-stage outputs (e.g., 0.2 Stage 1 + 0.8 Stage 2)
- Checkpoint ensembling with EMA/SWA testing
- Heuristic scaling/mixing of species outputs

## Notable individual insights
- rank 1 (1st Place Solution): Interval-based classification heads significantly improve regression performance by constraining predictions to learned biomass ranges.
- rank 2 ((2nd) Weakly supervised semantic segmentation + Synthetic data): Synthetic data initially dropped public scores due to ground truth mismatch, which was resolved by pseudo-labeling the synthetic images with a model trained on original data.
- rank 4 (4th Place Solution: ViT-Huge DINOv3 & Multi-Modal Feature Fusion with AUX predic): Manually removing background artifacts like cardboard backing from images provides a significant and consistent score boost.
- rank 5 (Rank 5th Solution): Public LB scores poorly correlate with Private LB due to distribution shifts and small dataset size, making submission selection highly uncertain.
- rank 7 (7th Place Solution - single/dual+TTT): Subdividing the original three dates into seven virtual dates and brute-forcing 10 million seeds was essential for achieving a balanced CV split.
- rank 37 (37st Place Solution: Self Prompt Tuning with DINOv3): Allowing training R² to rise to 0.95 while validation R² plateaued at 0.72 still yielded incremental leaderboard gains, proving the model wasn't unlearning generalizable features.

## Solutions indexed
- #1 [[solutions/rank_01/solution|1st Place Solution]]
- #2 [[solutions/rank_02/solution|(2nd) Weakly supervised semantic segmentation + Synthetic data]]
- #4 [[solutions/rank_04/solution|4th Place Solution: ViT-Huge DINOv3 & Multi-Modal Feature Fusion with AUX predic]]
- #5 [[solutions/rank_05/solution|Rank 5th Solution]]
- #7 [[solutions/rank_07/solution|7th Place Solution - single/dual+TTT]]
- #37 [[solutions/rank_37/solution|🏅 37st Place Solution: Self Prompt Tuning with DINOv3]]

## GitHub links
- [yanqiangmiffy/2025-Kaggle-CSIRO-5th-solution](https://github.com/yanqiangmiffy/2025-Kaggle-CSIRO-5th-solution) _(solution)_ — from [[solutions/rank_05/solution|Rank 5th Solution]]
- [QwenLM/Qwen-Image](https://github.com/QwenLM/Qwen-Image) _(solution)_ — from [[solutions/rank_02/solution|(2nd) Weakly supervised semantic segmentation + Synthetic data]]
- [facebookresearch/schedule_free](https://github.com/facebookresearch/schedule_free) _(library)_ — from [[solutions/rank_02/solution|(2nd) Weakly supervised semantic segmentation + Synthetic data]]
- [script-Yang/SegDINO](https://github.com/script-Yang/SegDINO) _(reference)_ — from [[solutions/rank_02/solution|(2nd) Weakly supervised semantic segmentation + Synthetic data]]
- [facebookresearch/sam2](https://github.com/facebookresearch/sam2) _(reference)_ — from [[solutions/rank_02/solution|(2nd) Weakly supervised semantic segmentation + Synthetic data]]
- [khawar-islam/diffuseMix](https://github.com/khawar-islam/diffuseMix) _(reference)_ — from [[solutions/rank_02/solution|(2nd) Weakly supervised semantic segmentation + Synthetic data]]
- [ByteDance-Seed/Depth-Anything-3](https://github.com/ByteDance-Seed/Depth-Anything-3) _(reference)_ — from [[solutions/rank_02/solution|(2nd) Weakly supervised semantic segmentation + Synthetic data]]
- [quanvuhust/csiro](https://github.com/quanvuhust/csiro) _(solution)_ — from [[solutions/rank_02/solution|(2nd) Weakly supervised semantic segmentation + Synthetic data]]
- [Jatin-Mehra119/Kaggle-CSIRO-4th-Position-Solution-](https://github.com/Jatin-Mehra119/Kaggle-CSIRO-4th-Position-Solution-) _(solution)_ — from [[solutions/rank_04/solution|4th Place Solution: ViT-Huge DINOv3 & Multi-Modal Feature Fusion with AUX predic]]

## Papers cited
- Uniformity in heterogeneity: Diving deep into count interval partition for crowd counting
