# human-protein-atlas-image-classification: cross-solution summary

This competition focused on multi-label image classification of protein locations using CNN ensembles. Winning approaches typically combined diverse architectures (ResNet, DenseNet, Inception, SE-ResNeXt) with external HPA data, heavy deduplication of test/train sets, and careful threshold optimization to handle extreme class imbalance.

## Competition profile
- **Modality / task:** image / classification
- **Domain:** medical_imaging
- **Metric:** F-Score (Macro)
- **Labels:** multi_label
- **Test split:** unknown
- **Format:** standard
- **Dataset scale:** unknown
- **Data challenges:** class_imbalance

## Key challenges
- extreme class imbalance
- test-set duplicates
- noisy labeling in external data
- visual similarity between certain classes (e.g., lysosomes and endosomes)
- sensitivity of Macro F1 to thresholds

## Models
- BN-Inception
- DenseNet-121
- GAPNet
- InceptionV3
- InceptionV4
- ResNet-18
- ResNet-34
- ResNet-50
- SE-Blocks
- SE-ResNeXt-50
- Xception

## CV strategies
- Split validation set according to community discussion
- Split by antibody-id for metric learning
- 1/10 holdout set with 5-fold cross validation
- Adversarial Validation (8K sample holdout set created by similarity to test)
- 5-fold cross-validation
- Removing duplicates using imagehash to improve validation set

## Preprocessing
- Removed duplicate samples using hash methods (phash, ahash, imagehash)
- Calculated mean and std using train+test for normalization
- Image-specific mean and stddev normalization
- Official + external HPA data
- Resizing to 256x256, 512x512, 768x768, and 1024x1024
- RGB channel selection
- External HPA data in Gray format
- RGB images

## Augmentations
- Rotate 90
- Flip
- Random crop 512x512 from 768x768
- Random crop 1024x1024 from 1536x1536
- AutoAugment
- Brightness augmentations
- D4 transforms
- Wrap transforms
- Contrast (applied independently on each color channel)
- Rotate
- Scale
- Shear
- Shift
- Add
- Multiply
- Crop
- Affine
- Filplr
- Filpub
- 12 TTA
- 32 TTA
- Multi-scale resolution images

## Loss functions
- ArcFaceLoss
- BCE loss
- FocalLoss
- F1 loss
- LSEP loss
- Lovasz
- Segmentation loss
- Weighted BCE

## Ensemble patterns
- Weighted averaging ensemble
- Stacking ensemble using LightGBM for most classes and linear models/voting for rare classes
- Ensemble of GAPNet and dual-loss ResNet34
- Weighted average of models across different image resolutions
- Combination of class-specific expert models and a one-vote ensemble of top predictions

## Post-processing
- Maintained label ratios based on public test set or average of train and public test set
- Replaced test labels with nearest neighbor labels from metric learning
- Thresholds optimized by matching predicted positive proportions to validation set proportions
- Fitting thresholds for single models using adversarial holdout set
- Averaging output probabilities of duplicate images in the test set
- Class-specific thresholds
- Fixed threshold of 0.205
- Adjusting non-expert model thresholds using T_opt = T_opt + (1 - T_opt) * 0.3

## What worked
- Focal Loss as a validation metric
- Metric learning for label replacement in test set
- ArcFace loss for metric learning
- Using antibody-id as a face-id for metric learning
- SE-Blocks before Average Pooling layers
- Supervised attention via segmentation loss as a regularizer
- Model diversity
- Learning rate finder and cyclic learning rate
- Differential learning rate with gradual reducing
- LSEP loss
- Simple one-layer network head
- Brightness augmentations
- D4 transforms
- Wrap transforms
- Average of 32 TTAs
- Averaging the output probabilities of duplicates in the test set
- Removing duplicates using imagehash
- Learnable gamma correction layer per channel

## What did not work
- Macro F1 soft loss
- Oversampling
- Complex multi-label classification architectures from papers
- Global average pooling for 1024x1024 input
- Training on RGBY
- Training with sample pairing
- Mixup
- Complex network head
- Large architectures (Nasnet, Senet-154)
- Training one-vs-all models
- Training on subsets of similar classes
- Classifier over bottleneck features
- Multi Head Attention Module (Transformer) on crop predictions
- GapNet style/Feature Pyramid with ResNet18
- Training with 2048x2048 images

## Critical findings
- F1 is sensitive to the threshold and distribution of train/val sets
- The test set is slightly different from V18
- Distributions of image mean and stddev are very different between official and external data
- Macro F1 score is very sensitive to thresholds
- Relying on rare class score for early stopping is risky
- Yellow channel was not very helpful
- Labeling was likely noisy (contradictions between HPAv18 and competition data)
- Some classes (e.g., lysosomes and endosomes) are visually extremely similar
- There are tons of duplicates in the test set
- Models that overfit showed a pattern where Precision extremely dominated Recall in F1-curves

## Notable individual insights
- rank 1 (A CNN classifier and a Metric Learning model,1st Place Solution ): Use of ArcFace loss and nearest neighbor search to replace test labels with high-confidence matches from training/V18 data.
- rank 3 (3rd place solution with code.): Custom output layer for 1024x1024 images replacing global average pooling with weighted average logits.
- rank 4 (part of 4th place solution: GAPNet & dual loss ResNet): Use of a segmentation loss on the green channel as a regularizer for classification.
- rank 8 (8 place solution writeup): Use of adversarial validation to create a holdout set that mimics the test set distribution.
- rank 12 (12th place solution): Implementation of a learnable gamma correction layer per channel to handle dark exposure in images.

## Solutions indexed
- #1 [[solutions/rank_01/solution|A CNN classifier and a Metric Learning model,1st Place Solution ]]
- #3 [[solutions/rank_03/solution|3rd place solution with code.]]
- #4 [[solutions/rank_04/solution|title": "part of 4th place solution: GAPNet & dual loss ResNet]]
- #7 [[solutions/rank_07/solution|7th place solution]]
- #8 [[solutions/rank_08/solution|8 place solution writeup]]
- #11 [[solutions/rank_11/solution|11th Place Solution]]
- #12 [[solutions/rank_12/solution|12th place solution]]

## GitHub links
- [pudae/kaggle-hpa](https://github.com/pudae/kaggle-hpa) _(solution)_ — from [[solutions/rank_03/solution|3rd place solution with code.]]
- [Cadene/pretrained-models.pytorch](https://github.com/Cadene/pretrained-models.pytorch) _(library)_ — from [[solutions/rank_08/solution|8 place solution writeup]]
- [bfelbo/DeepMoji](https://github.com/bfelbo/DeepMoji) _(reference)_ — from [[solutions/rank_07/solution|7th place solution]]
- [Gary-Deeplearning/Human_Protein](https://github.com/Gary-Deeplearning/Human_Protein) _(solution)_ — from [[solutions/rank_11/solution|11th Place Solution]]

## Papers cited
- [ArcFace: Additive Angular Margin Loss for Deep Face Recognition](https://arxiv.org/pdf/1801.07698v1.pdf)
- [AutoAugment: Learning Augmentation Policies](https://arxiv.org/pdf/1805.09501.pdf)
- [LSEP loss](https://arxiv.org/pdf/1704.03135.pdf)
