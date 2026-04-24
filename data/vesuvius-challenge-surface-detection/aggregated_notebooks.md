# vesuvius-challenge-surface-detection: top public notebooks

The top-voted notebooks for the Vesuvius Challenge Surface Detection task predominantly focus on inference pipelines and baseline training workflows for 3D volumetric segmentation. They heavily leverage pre-trained architectures like TransUNet and SegFormer, emphasizing memory-efficient sliding window inference, extensive test-time augmentation, and domain-specific 3D morphological post-processing to refine ink surface predictions. While most entries are inference-focused, a few provide complete training baselines utilizing advanced loss functions (e.g., topology-aware CLDice, skeleton recall) and framework-specific optimizations like nnUNet auto-configuration or GPU-accelerated MONAI augmentations.

## Common purposes
- inference
- baseline
- training

## Competition flows
- Loads test TIFF volumes, normalizes intensities, runs sliding window inference with TTA on two pre-trained 3D models, applies 3D morphological post-processing, and exports the final masks as a submission ZIP.
- Loads test volumes, runs a pre-trained TransUNet with sliding window inference and TTA to extract logits, converts them to binary probabilities, applies seeded hysteresis thresholding with public anchors and 3D morphological operations, and zips the output masks for submission.
- Loads test TIFF volumes and metadata, normalizes intensities, runs sliding window inference with rotation TTA on a pre-trained TransUNet, applies 3D morphological post-processing, and packages the binary masks into a submission ZIP file.
- Loads 3D image/label volumes from disk, applies GPU-accelerated augmentations and dynamic resizing, trains a 3D SegResNet with PyTorch Lightning using combined DiceCE/Tversky loss, and generates binary segmentation masks for test fragments via inference and post-processing.
- Loads 3D test volumes, runs multi-stream sliding window inference with TTA to generate logits, converts them to probabilities, and applies a seeded hysteresis thresholding with morphological topology constraints to produce final binary ink masks for submission.
- Loads 3D volumetric data from TFRecords, applies geometric and intensity augmentations, trains a SegFormer segmentation model with a combined Dice/CE and CLDice loss, and evaluates predictions using sliding window inference to compute Dice scores.
- Loads 3D TIFF CT volumes and labels, converts them to nnUNet format via symlinks, runs preprocessing and training with nnUNetv2, performs inference on test volumes, and packages the binary segmentation masks into a submission ZIP.
- Loads 3D test volumes from TIFF files, normalizes them, runs sliding-window inference with TTA across five pre-trained TransUNet models, applies topological post-processing to the averaged logits, and packages the resulting binary masks into a ZIP submission file.
- Loads test CSV and TIFF volumes, normalizes pixel intensities, runs 3D TransUNet with sliding window inference, and packages predicted masks into a submission ZIP.
- Loads TFRecord data, applies 3D augmentations and generates a skeleton target, trains a TransUNet with a custom multi-task loss on TPUs, and saves periodic checkpoints with sliding window validation.

## Data reading
- Reads test.csv for image IDs using pandas.
- Loads 3D TIFF files with tifffile.imread, converts to np.float32, and expands dimensions to (1, D, H, W, 1).
- Reads test IDs from a CSV file.
- Loads 3D TIFF volumes using tifffile.imread, converts them to float32, and reshapes to (1, D, H, W, 1).
- Reads test.csv for image IDs and paths.
- Loads .tif files with tifffile.imread, converts to float32, and reshapes to (1, D, H, W, 1) for model input.
- Custom `SurfaceDataset3D` scans directories for `.npy`, `.npz`, or `.tif` files, loads them with `numpy`/`tifffile`/`imagecodecs`, and returns raw `(D, H, W)` arrays; labels are loaded from a parallel directory if available.
- Reads test IDs from `test.csv` and loads 3D TIFF volumes using `tifffile.imread`, reshaping them to `(1, D, H, W, 1)` float32 tensors.
- Parses TFRecord files using tf.io.parse_single_example with fixed-length string features for image, label, and their shapes.
- Decodes raw byte strings to uint8 tensors, reshapes to (D, H, W), casts to float32, and appends a channel dimension to form (D, H, W, 1).
- Reads 3D TIFF files directly using tifffile and a custom SimpleTiffIO reader
- Creates JSON sidecars for voxel spacing information
- Uses symlinks to avoid copying large competition files
- Reads test metadata from test.csv
- Loads 3D volumetric TIFF files using tifffile.imread
- Converts data to float32 and adds channel dimension [1, D, H, W, 1]
- Reads test image IDs from `test.csv` using pandas.
- Loads corresponding 3D volumetric TIFF files using `tifffile.imread`, converting them to float32 and adding batch/channel dimensions.
- Parses TFRecord files using tf.io.parse_single_example
- Decodes raw byte strings to tensors and reshapes them using stored shape metadata

## Data processing
- Applies NormalizeIntensity (nonzero=True, channel_wise=False) via medicai.transforms.Compose.
- Applies NormalizeIntensity (nonzero, channel-wise=False) via medicai.transforms
- Applies 3D sliding window inference with Gaussian overlap.
- Aggregates TTA predictions (flips along axes 1,2,3 and rotations in axes 2,3) via mean pooling.
- Converts multiclass logits to binary logits using logsumexp subtraction.
- Applies seeded hysteresis thresholding with public anchors.
- Performs 3D binary closing with anisotropic structuring elements.
- Removes small disconnected components (dust) using skimage.morphology.remove_small_objects.
- Intensity scaling to [0, 1] using ScaleIntensityRange.
- Sliding window inference with 160³ ROI, 50% overlap, and Gaussian blending.
- 4x rotation TTA with inverse rotation alignment and averaging.
- 3D hysteresis thresholding, anisotropic closing, and small object removal.
- Dynamic resizing to `(160, 160, 160)` via MONAI `Resized`.
- GPU-accelerated augmentations: 3D `RandFlipd`, `RandRotated`, `RandShiftIntensityd`, `RandGaussianNoised`.
- Validation only resizes.
- Custom collate returns lists to defer stacking until GPU transfer.
- Mixed precision (`16-mixed`) used during training.
- Applies channel-wise intensity normalization via `medicai.transforms.NormalizeIntensity`.
- Training augmentations via medicai.transforms.Compose: RandSpatialCrop, RandFlip (3 axes), RandRotate90, RandRotate, NormalizeIntensity, RandShiftIntensity, and RandCutOut.
- Validation uses only NormalizeIntensity.
- Inference employs SlidingWindowInference with Gaussian weighting, 0.5 overlap, and ROI size matching input shape.
- nnUNet auto-preprocessing (resampling, normalization)
- Isotropic voxel spacing (1.0, 1.0, 1.0) via JSON sidecars
- Ignore label (2) masking to exclude unlabeled regions from loss computation
- Applies `ScaleIntensityRange` normalization to map raw pixel values from [0, 255] to [0, 1].
- Uses sliding window inference with Gaussian weighting, 52% overlap, and a 160x160x160 ROI size to handle large volumes without OOM errors.
- Adds channel dimensions and casts to float32
- RandSpatialCrop to (160, 160, 160)
- RandFlip across 3 spatial axes (prob=0.5)
- RandRotate90 (prob=0.4)
- RandShiftIntensity (offsets=0.15, prob=0.5)
- ScaleIntensityRange normalization (0-255 to 0-1)
- Custom random cuboid occlusion augmentation that masks random 3D blocks to 0
- Validation path applies only intensity scaling

## Models
- TransUNet (SEResNeXt50 encoder)
- SegFormer (mit_b4 encoder)
- SegFormer (mit_b0 encoder)
- SegResNet
- nnUNet (nnUNetPlannerResEncM / ResNet encoder variant)
- UPerNet
- UNETRPlusPlus
- Custom ConvNeXtV2Tiny + TransUNet
- SEResNeXt1013D

## Frameworks used
- keras
- medicai
- numpy
- scipy
- scikit-image
- jax
- tifffile
- pandas
- pytorch
- pytorch_lightning
- monai
- imagecodecs
- tensorflow
- matplotlib
- PIL
- nnunetv2
- torch
- nibabel
- tqdm

## Loss functions
- DiceCELoss
- TverskyLoss (alpha=0.7, beta=0.3)
- SparseDiceCELoss
- SparseCenterlineDiceLoss
- nnUNet partial loss (excludes label 2/ignore regions from gradient computation)
- SkeletonRecallPlusDiceLoss (combines SparseDiceCELoss, skeleton recall loss, and false positive volume loss)

## CV strategies
- Holdout 80/20 train/val split (random seed 42)
- Holdout validation using the last TFRecord file as the validation set and all preceding files as training data.
- fold='all' (trains on all training data without cross-validation)
- Holdout validation using shard 130 for validation and shards 0-129 for training.

## Ensembling
- A weighted ensemble averages the logits from both models using weights [0.8, 0.2], applies softmax, and takes the argmax to generate final predictions.
- Aggregates logits across three sliding window overlap configurations and multiple TTA transformations via mean pooling before applying the final thresholding pipeline.
- Averages model predictions across four 90-degree clockwise rotations, applying inverse rotations to align outputs before averaging.
- Averages multiclass logits across TTA variants (flips and rotations) and combines outputs from two different sliding window overlap configurations to stabilize predictions.
- Averages prediction logits across five pre-trained TransUNet models and combines them with extensive test-time augmentation (flips and rotations) before applying topological post-processing.

## Insights
- Sliding window inference with Gaussian overlap enables processing large 3D volumes within limited VRAM.
- Test-time augmentation via spatial flips and axial rotations improves prediction stability without additional training.
- 3D hysteresis thresholding combined with anisotropic morphological closing effectively connects fragmented surface predictions and removes noise.
- Public predictions can be safely used as weak anchors for hysteresis expansion without introducing data leakage.
- Converting multiclass logits to binary probabilities via logsumexp subtraction preserves calibration better than argmax-based approaches.
- 3D anisotropic morphological closing is critical for maintaining continuity in ink trace masks.
- Sliding window overlap thresholds require precise, leaderboard-specific tuning to hit target scores.
- GPU-accelerated augmentations via MONAI drastically reduce training time by avoiding CPU-GPU data transfers.
- Pre-computed `.npy`/`.npz` volumes bypass slow TIFF decoding and improve data throughput.
- Custom collate functions combined with `on_after_batch_transfer` enable efficient handling of variable-sized 3D batches.
- Mixed precision and gradient clipping are essential for stable 3D volume training.
- Deep supervision with auxiliary decoder losses mitigates gradient vanishing and accelerates convergence in 3D segmentation.
- Combining standard Dice/CE with topology-aware CLDice loss improves surface and boundary detection accuracy.
- nnUNet automatically configures critical hyperparameters like patch size, batch size, and learning rate schedule, making manual tuning unnecessary.
- Using the nnUNetPlannerResEncM planner with a ResNet encoder often yields 1-2% better accuracy than the default U-Net planner.
- Deriving a tubed skeleton from ground truth masks effectively guides the model to learn thin, tubular ink structures.
- Combining recall and false positive penalties in the loss function improves precision without sacrificing recall.
- Cosine decay with a non-zero alpha prevents the learning rate from collapsing to zero during training.

## Critical findings
- Sliding window overlaps of 0.42 and 0.43 are explicitly required to match the public and private 0.55 scores, respectively.
- Kaggle container protobuf versions often lack MessageFactory.GetPrototype, requiring a manual compatibility patch.
- Model outputs contain significant dust-like false positives that must be filtered with a minimum object size threshold.

## Notable individual insights
- votes 458 (Vesuvius 0.552): Public predictions can be safely used as weak anchors for hysteresis expansion without introducing data leakage.
- votes 321 (Surface📜: train+inference 3D segm & GPU augment.): GPU-accelerated augmentations via MONAI drastically reduce training time by avoiding CPU-GPU data transfers.
- votes 266 ([Train] Vesuvius Surface 3D Detection on TPU): Deep supervision with auxiliary decoder losses mitigates gradient vanishing and accelerates convergence in 3D segmentation.
- votes 263 (Surface📜: nnUNet training + inference [with 2xT4]): nnUNet automatically configures critical hyperparameters like patch size, batch size, and learning rate schedule, making manual tuning unnecessary.
- votes 202 ([Train] TransUNet Baseline [LB 0.537]): Deriving a tubed skeleton from ground truth masks effectively guides the model to learn thin, tubular ink structures.
- votes 458 (Vesuvius 0.552): Sliding window overlaps of 0.42 and 0.43 are explicitly required to match the public and private 0.55 scores, respectively.
- votes 266 ([Train] Vesuvius Surface 3D Detection on TPU): Combining standard Dice/CE with topology-aware CLDice loss improves surface and boundary detection accuracy.

## Notebooks indexed
- #568 votes [[notebooks/votes_01_ipythonx-inference-vesuvius-surface-3d-detection/notebook|[Inference] Vesuvius Surface 3D Detection ]] ([kaggle](https://www.kaggle.com/code/ipythonx/inference-vesuvius-surface-3d-detection))
- #458 votes [[notebooks/votes_02_tonylica-vesuvius-0-552/notebook|Vesuvius 0.552]] ([kaggle](https://www.kaggle.com/code/tonylica/vesuvius-0-552))
- #358 votes [[notebooks/votes_03_choudharymanas-inference-baseline-transunet-lb-0-537/notebook|[Inference] Baseline TransUNet   [LB 0.537]]] ([kaggle](https://www.kaggle.com/code/choudharymanas/inference-baseline-transunet-lb-0-537))
- #321 votes [[notebooks/votes_04_jirkaborovec-surface-train-inference-3d-segm-gpu-augment/notebook|Surface📜: train+inference 3D segm & GPU augment.]] ([kaggle](https://www.kaggle.com/code/jirkaborovec/surface-train-inference-3d-segm-gpu-augment))
- #281 votes [[notebooks/votes_05_pankajiitr-inference-vesuvius-challenge/notebook|[Inference] Vesuvius Challenge ]] ([kaggle](https://www.kaggle.com/code/pankajiitr/inference-vesuvius-challenge))
- #266 votes [[notebooks/votes_06_ipythonx-train-vesuvius-surface-3d-detection-on-tpu/notebook|[Train] Vesuvius Surface 3D Detection on TPU]] ([kaggle](https://www.kaggle.com/code/ipythonx/train-vesuvius-surface-3d-detection-on-tpu))
- #263 votes [[notebooks/votes_07_jirkaborovec-surface-nnunet-training-inference-with-2xt4/notebook|Surface📜: nnUNet training + inference [with 2xT4]]] ([kaggle](https://www.kaggle.com/code/jirkaborovec/surface-nnunet-training-inference-with-2xt4))
- #248 votes [[notebooks/votes_08_tonyai007-inference-vesuvius-surface-3d-detection-epoch-x5/notebook|[Inference] Vesuvius Surface 3D Detection_Epoch_X5]] ([kaggle](https://www.kaggle.com/code/tonyai007/inference-vesuvius-surface-3d-detection-epoch-x5))
- #213 votes [[notebooks/votes_09_hideyukizushi-vesuvius25-transunet-seresnext1013d-lb-460-newlb/notebook|Vesuvius25|TransUNet SEResNeXt1013D|LB.460(NewLB)]] ([kaggle](https://www.kaggle.com/code/hideyukizushi/vesuvius25-transunet-seresnext1013d-lb-460-newlb))
- #202 votes [[notebooks/votes_10_choudharymanas-train-transunet-baseline-lb-0-537/notebook|[Train] TransUNet Baseline [LB 0.537]]] ([kaggle](https://www.kaggle.com/code/choudharymanas/train-transunet-baseline-lb-0-537))
