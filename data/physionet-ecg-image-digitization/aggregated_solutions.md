# physionet-ecg-image-digitization: cross-solution summary

This competition focused on high-precision ECG image digitization, requiring robust pipelines to extract accurate 12-lead time-series signals from varied, often degraded, printed ECG strips. Winning approaches consistently combined advanced image rectification and grid alignment with high-resolution vision backbones, leveraging diverse architectures and specialized loss functions to maximize signal-to-noise ratio (SNR). Success heavily relied on strategic data augmentation, sub-pixel mask generation, FFT-based resampling, and physics-informed post-processing like Einthoven’s Law corrections.

## Competition flows
- Raw ECG strip images rectified via homography/remapping, passed through vision backbones, refined via Fourier-domain resampling & Einthoven’s Law blending
- Raw ECG images processed through rotation correction, lead detection/segmentation, digitized via ViT/MaxViT models, combined via mean ensemble with OOD filter
- Raw ECG images processed through baseline stage 0/1, fed into sparse mask segmentation models, reconstructed via weighted averaging & scipy resampling
- Raw ECG images rectified via two-step high-res mapping, fed into ConvNeXt V2/HRNet segmentation, refined via parabolic interpolation & Lead II fusion
- Raw ECG signals resampled via FFT, passed through backbone & Soft-Argmax regression head, output via TTA
- Raw ECG images preprocessed, passed through differentiable UNet for soft masks, converted via weighted centroid extraction, refined by 1D ResUNet, combined via model/resampling ensemble
- Raw ECG images processed through grid detection, coordinate indexing, rectification, vertical splitting, followed by regression & resampling
- Raw ECG images orientation-corrected, processed through heatmap keypoint detection, iterative registration, piecewise homography cropping, dual-encoder UNet prediction, fused & refined
- Raw ECG images normalized/rectified via Stage 0/1, fed into HRNet-W48, processed with TTA & Einthoven's Law correction, converted via softmax weighted-centroid extraction
- Raw ECG images oriented/warped, grid points refined via CNN cascade, signals extracted from patches via coordinate-injected ensemble, corrected via lead ensembling & Einthoven error correction

## Data processing
- Image rectification & warping (homography, piecewise homography, grid unwarping, coordinate indexing)
- Rotation correction & orientation alignment
- Grid detection & keypoint refinement (sliding window, cascade CNNs, soft labels)
- Lead detection & segmentation (two-pass cropping, min dimensions 16x64px, probability maps)
- Image preprocessing (cropping headers/margins, grayscale conversion, longest resize + padding, resizing/padding to fixed resolutions like 1536x1536 or 5600x1280, vertical splitting into segments)
- High-resolution training & scaling (1696x4352, 1440x1152, 2400x1920, 2x length scaling, Lanczos4 interpolation)
- Signal resampling (Fourier-domain/FFT-based via scipy.signal.resample, polyphase, linear/cubic spline, custom resample_torch)
- Synthetic data & mask generation (blank ECG templates, wrinkled data, sparse masks with fractional y-coordinates, one-pixel-wide masks, soft labels, truncated normal noise on keypoints)
- Augmentations (Moiré patterns, flares, overlays, color/distortion/blur/noise/shadow/JPEG artifacts, GridDropout, CoarseDropout, ElasticTransform, horizontal/vertical flips, affine/perspective shifts, memory-mapped caching for heavy augmentations)
- TTA & post-processing corrections (gamma adjustments, cropping/resizing, horizontal/vertical flips, Einthoven's Law physics-based correction, softmax weighted-centroid extraction, weighted averaging windows)
- Memory & optimization techniques (gradient checkpointing, memory-mapped caching)

## Features engineering
- Coordinate-based features concatenated with grayscale images for spatial context
- Partitioning backbone features by height into four segments corresponding to ECG data groups
- Expanding segments to four times input width
- Max pooling over height dimension in Strategy 3

## Models
- ConvNeXt V2
- EfficientNet (B3/B4/B6/B7/V2 L/M/S)
- U-Net (incl. 1D ResUNet, 2D UNet)
- HRNet (W48)
- ResNet (18/34)
- ViT (Dinov3)
- MaxViT
- HGNet-V2
- MambaOut
- CaFormer
- InceptionNeXt
- CoaT
- VGG19
- EfficientViT
- MINIMA-RoMa
- LoFTR
- FCMAE
- CoordConv
- PixelShuffle
- Deformable Convolution
- BiLSTM
- Multi-head Regression
- Conv2d/Conv3d Fusion Modules
- MyCoordUnetDecoder
- UnetDecoder
- Stage 0/1/2 Networks
- Heatmap CNNs

## Frameworks used
- PyTorch/torch
- SciPy/scipy.signal
- timm
- Albumentations
- OpenCV (cv2)
- SMP (segmentation-models-pytorch)
- NumPy

## Loss functions
- SmoothL1Loss
- SNR Loss (incl. SNR-based loss, optimized SNR/Mean terms)
- MAE/L1 Loss
- ArcFace Loss
- BCE/BCEWithLogitsLoss (with positive class weighting, e.g., pos_weight=10/20)
- Channel-Masked JSD / Column-wise JSD Loss
- MSELoss (on DSNT predictions)
- GLS (Geometric Loss Strategy)
- Combined Scheduled Loss

## CV strategies
- First ten samples as validation (with segmented image selection)
- k-fold cross-validation (with lightweight/expensive model split validation)
- 5-fold cross-validation with OOF predictions for pseudo-labeling
- 100/0 train/validation split (full data usage)
- no validation split (relying on scheduled loss weighting & gradient accumulation)

## Ensembling
- Weighted/mean/median ensembles of diverse backbones (EfficientNet, ConvNeXt, HRNet, U-Net variants)
- Signal-level vs. probability-level averaging
- Resampling ensembles (averaging outputs from decimate_fir, polyphase, linear methods)
- Lead II fusion & Einthoven's Law corrections
- OOD detection filters for masking predictions
- TTA with horizontal/vertical flips
- Refinement models for out-of-range predictions

## Notable individual insights
- rank 1 (1st place solution): Resampling in the Fourier domain using scipy.signal.resample yields higher SNR than linear interpolation for ECG signals.
- rank 2 (2nd place solution): Sparse masks with fractional y-coordinate distribution enable sub-pixel precision and higher reconstruction SNR than dense masks.
- rank 3 (3rd place solution): Signal-level ensembling (averaging post-processed signals) outperformed probability-level ensembling.
- rank 5 (5th place solution: Multi-stages Heatmap-based Modeling): Prioritizing input image resolution over model complexity is critical for preserving fine-grained details and reducing sub-pixel error.
- rank 6 (6th place solution): Direct regression bypasses cumulative error from segmentation and post-processing stages.
- rank 13 (13th Place Solution): Horizontal flip augmentation unexpectedly delivered a +0.6 dB improvement, likely due to doubling effective training data and promoting orientation invariance.
- rank 14 (14th Place Solution): High resolution (1696×4352) provided the largest single performance boost (~3 SNR).

## Solutions indexed
- #1 [[solutions/rank_01/solution|1st place solution]]
- #2 [[solutions/rank_02/solution|2nd place solution]]
- #3 [[solutions/rank_03/solution|3rd place solution]]
- #4 [[solutions/rank_04/solution|4th Place Solution]]
- #5 [[solutions/rank_05/solution|5th place solution: Multi-stages Heatmap-based Modeling]]
- #6 [[solutions/rank_06/solution|6th place solution]]
- #7 [[solutions/rank_07/solution|7th place solution]]
- #13 [[solutions/rank_13/solution|13th Place Solution]]
- #14 [[solutions/rank_14/solution|14th Place Solution]]
- #21 [[solutions/rank_21/solution|21st Place Solution]]
- ? [[solutions/rank_xx_613391/solution|reference solution: High Precision ECG Digitization Using Artificial Intelligence - medRxiv, 2024-sep]]

## GitHub links
- [TheoViel/kaggle_rsna_abdominal_trauma](https://github.com/TheoViel/kaggle_rsna_abdominal_trauma) _(reference)_ — from [[solutions/rank_07/solution|7th place solution]]
- [brendanartley/PhysioNet-Competition](https://github.com/brendanartley/PhysioNet-Competition) _(solution)_ — from [[solutions/rank_07/solution|7th place solution]]
- [someya-takashi/physionet-image-digitization](https://github.com/someya-takashi/physionet-image-digitization) _(solution)_ — from [[solutions/rank_02/solution|2nd place solution]]
- [tanghaozhe/physionet-ecg-image-digitization-3rd-place](https://github.com/tanghaozhe/physionet-ecg-image-digitization-3rd-place) _(solution)_ — from [[solutions/rank_03/solution|3rd place solution]]
- [GWwangshuo/Kaggle-2025-PhysioNet](https://github.com/GWwangshuo/Kaggle-2025-PhysioNet) _(solution)_ — from [[solutions/rank_06/solution|6th place solution]]
- [alphanumericslab/ecg-image-kit](https://github.com/alphanumericslab/ecg-image-kit) _(library)_ — from [[solutions/rank_04/solution|4th Place Solution]]
- [uchiyama33/physionet_4th_place](https://github.com/uchiyama33/physionet_4th_place) _(solution)_ — from [[solutions/rank_04/solution|4th Place Solution]]
- [dangnh0611/kaggle_ecg_digitization](https://github.com/dangnh0611/kaggle_ecg_digitization) _(solution)_ — from [[solutions/rank_05/solution|5th place solution: Multi-stages Heatmap-based Modeling]]
- [alphanumericslab/ecg-image-kit](https://github.com/alphanumericslab/ecg-image-kit) _(library)_ — from [[solutions/rank_05/solution|5th place solution: Multi-stages Heatmap-based Modeling]]
- [LSXI7/MINIMA](https://github.com/LSXI7/MINIMA) _(library)_ — from [[solutions/rank_05/solution|5th place solution: Multi-stages Heatmap-based Modeling]]
- [starrynites/PhysioNet_Digitization_of_ECG_Images](https://github.com/starrynites/PhysioNet_Digitization_of_ECG_Images) _(solution)_ — from [[solutions/rank_14/solution|14th Place Solution]]

## Papers cited
- [ECG-Image-Kit: a synthetic image generation toolbox to facilitate deep learning-based electrocardiogram digitization](https://doi.org/10.1088/1361-6579/ad4954)
- [ECG-Image-Database: A dataset of ECG images with real-world imaging and scanning artifacts; a foundation for computerized ECG image digitization and analysis](https://doi.org/10.48550/arXiv.2409.16612)
- [PTB-XL, a large publicly available electrocardiography dataset (version 1.0.3)](https://doi.org/10.13026/kfzx-aw45)
- [PTB-XL: A Large Publicly Available ECG Dataset](https://doi.org/10.1038/s41597-020-0495-6)
- [PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals](https://doi.org/10.1161/01.CIR.101.23.e215)
- Convnext v2: Co-designing and scaling convnets with masked autoencoders
- [ECG-image-kit: a synthetic image generation toolbox to facilitate deep learning-based electrocardiogram digitization](https://doi.org/10.1088/1361-6579/ad4954)
- [ECG-Image-Kit: A Toolkit for Synthesis, Analysis, and Digitization of Electrocardiogram Images](https://github.com/alphanumericslab/ecg-image-kit)
- [CODE-15%: a large scale annotated dataset of 12-lead ECGs (1.0.0)](https://doi.org/10.5281/zenodo.4916206)
- [Geometric Loss Strategy](https://arxiv.org/pdf/1904.08492)
- [DSNT](https://arxiv.org/pdf/1801.07372)
- [RoMA](https://arxiv.org/abs/2305.15404)
- [EDSR](https://arxiv.org/abs/1707.02921)
- High Precision ECG Digitization Using Artificial Intelligence
- [PhysioNet - Digitization of ECG Images](https://kaggle.com/competitions/physionet-ecg-image-digitization)
