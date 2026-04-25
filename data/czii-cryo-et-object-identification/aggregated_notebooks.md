# czii-cryo-et-object-identification: top public notebooks

The top-voted notebooks for this cryo-electron tomography competition focus primarily on establishing robust baselines and inference pipelines for 3D particle detection and segmentation. The community heavily leverages hybrid approaches combining 2D slice-based detectors (YOLO) with 3D volumetric models (U-Net variants), emphasizing memory-efficient patching, sliding-window inference, and precise spatial post-processing to convert dense masks into coordinate-based submissions. Training strategies center on handling severe class imbalance via Tversky loss and class-weighted masking, while classical computer vision techniques like Hessian filtering and watershed segmentation are also explored as viable, lightweight alternatives.

## Common purposes
- inference
- baseline
- training

## Competition flows
- Loads pre-computed 3D volumetric numpy arrays, patches them into fixed blocks, trains a 3D U-Net with Tversky loss, and converts inference masks into connected component centroids
- Processes test cryo-ET volumes via 2D slice inference and 3D patch-based inference with TTA, aggregates detections using spatial clustering, and generates a combined submission CSV
- Configures a CoPick project to load 3D CryoET tomograms and segmentation masks, generates training targets, and trains a 3D Residual U-Net with class-weighted loss
- Processes 3D cryo-ET volumes via a sliding-window inference pipeline, extracts 3D particle coordinates using connected components, and generates a CSV submission alongside a custom local evaluation script
- Runs slice-by-slice 2D detection, aggregates cross-slice detections via a geometric graph and DFS to estimate 3D sphere centers, and exports the results as a submission CSV
- Processes 3D cryo-ET volumes via a sliding window scanner with TTA, applies class-specific thresholds and connected components for centroid extraction, and outputs a formatted submission CSV
- Runs 2D YOLO inference on resized slices, aggregates detections across slices using KDTree and Union-Find, and outputs a CSV submission with 3D particle coordinates and types
- Converts sparse particle picks into dense 3D multi-class segmentation masks, trains a 3D U-Net with Tversky loss using MONAI, and tracks metrics and model weights via MLflow
- Trains a pretrained YOLO11-L model for 100 epochs with data augmentations and AdamW optimizer, validates it using mAP metrics, and demonstrates inference on a sample image
- Applies a resolution-adaptive Hessian-based blob detection and watershed segmentation pipeline to multi-resolution 3D tomograms, and exports detected particle centroids as a CSV submission file

## Data reading
- Loads pre-computed 3D volumetric data directly from local numpy files organized into dictionaries with 'image' and 'label' keys
- Reads test volumes from zarr files using zarr.open and copick library
- Loads 3D tomographic volumes and segmentation masks via the copick library from a JSON configuration file
- Uses custom helper functions to load 3D volume tensors and ground truth coordinates from Kaggle input directories
- Reads test volumes from Zarr files, extracts sequential 2D slices, and converts them to normalized 8-bit images
- Loads 3D volumetric data and scale factors using a custom read_one_data function from a local dataset module
- Loads 3D volumetric data from zarr files located in the test directory, extracting the first channel as a numpy array
- Copies sparse particle picks using copick_utils.segmentation_from_picks, stored as dense multi-label NumPy arrays
- Reads dataset paths and annotations from a YAML configuration file that structures the data for YOLO object detection format
- Reads tomogram data from zarr stores using the copick library and a custom JSON configuration, selecting resolution levels based on a radius threshold

## Data processing
- CacheDataset with channel orientation & intensity normalization
- Random augmentations (label-based cropping, 90° rotation, flipping)
- Non-overlapping & overlapping patch extraction (96³, 72³, 64x64x64)
- Percentile clipping & 8-bit conversion
- 2D slice resizing (640x640) & RGB stacking
- Sliding-window inference with weighted probability averaging
- Class-specific probability thresholding
- Connected components analysis & blob size filtering
- Coordinate scaling & centroid extraction
- Multi-class mask generation from sparse picks
- Real-time training augmentations (mixup, copy-paste, shear, rotation)
- Hessian-based filtering & watershed segmentation
- Gaussian smoothing & distance transform

## Models
- UNet
- 3D UNet (MONAI)
- YOLO11
- YOLO11-L
- Residual U-Net (res_unet)
- ResNet34D (3D UNet with 2D encoder)

## Frameworks used
- PyTorch
- MONAI
- PyTorch Lightning
- CoPick
- connected-components-3d (cc3d)
- Pandas
- NumPy
- Ultralytics
- SciPy
- OpenCV
- scikit-learn
- DeepFindET
- Matplotlib
- h5py
- Zarr
- tqdm
- concurrent.futures
- MLflow

## Loss functions
- TverskyLoss

## CV strategies
- Fixed holdout split using 6 training experiments and 1 validation experiment (TS_6_4), processed via deterministic patching for consistent validation metrics
- Validation on TS_5_4, TS_69_2, TS_6_4, TS_6_6 with OPTUNA-optimized confidence and distance thresholds
- Fixed holdout split: first 5 runs for training, next 2 runs for validation, with metrics computed every 2 epochs

## Ensembling
- Blends YOLO and 3D UNet predictions, then applies particle-specific DBSCAN clustering to merge nearby detections and average their coordinates
- Aggregates 2D slice detections into 3D sphere centers using a confidence-weighted depth-first search on a geometric graph, applying particle-specific radius constraints and confidence thresholds
- Averages TTA probability maps and applies class-specific thresholds followed by connected components analysis to extract centroids
- Single YOLO model inference followed by KDTree-based spatial clustering across 2D slices using a Union-Find algorithm, confidence weighting, and per-class minimum cluster size filtering

## Insights
- Pre-computing raw volumetric data to numpy arrays significantly reduces notebook execution overhead.
- Avoiding stochastic augmentations during validation prevents metric noise and ensures reliable model monitoring.
- Patching large 3D volumes into fixed-size blocks is necessary for GPU memory constraints and enables consistent validation coverage.
- Converting dense segmentation masks to sparse coordinates requires connected component analysis and centroid filtering to match competition submission formats.
- Blending 2D slice-based detection with 3D volumetric segmentation can improve robustness in cryo-ET particle identification.
- Spatial deduplication using connected components and distance-based clustering is essential to avoid double-counting particles.
- Test-time augmentation (flipping and rotating) effectively boosts 3D UNet inference performance without retraining.
- CoPick provides a standardized way to manage 3D CryoET projects and segmentation targets without manual file path management.
- DeepFindET abstracts the 3D CNN training loop, allowing focus on data configuration and hyperparameters.
- Explicit class weighting is necessary to handle the extreme imbalance between rare protein targets and background/membrane classes.
- Sliding-window inference with overlapping slices and probability averaging effectively handles large 3D volumes while maintaining spatial consistency.
- Connected-component analysis on thresholded probability maps is a robust and efficient way to extract discrete 3D object locations.
- Custom evaluation metrics using the Hungarian algorithm for matching and class-weighted F-beta scores can accurately simulate competition leaderboards locally.
- 2D object detectors can be effectively repurposed for 3D volumetric localization by processing sequential slices and aggregating results spatially.
- Graph-based DFS aggregation with geometric tolerance thresholds significantly refines noisy detection outputs into accurate 3D coordinates.
- Particle-specific confidence thresholds and radius constraints are necessary to handle varying detection difficulties across different biological structures.
- Sliding window inference with overlapping slices and weighted averaging effectively handles large 3D volumes that exceed GPU memory limits.
- Simple rotation-based TTA improves prediction robustness without significant computational overhead.
- Class-specific thresholds and connected components analysis are essential for accurate centroid extraction in noisy cryo-ET data.
- Converting volumetric zarr data to 8-bit 2D slices enables efficient 2D YOLO inference without excessive memory overhead.
- KDTree spatial queries combined with Union-Find provide a fast and robust way to cluster overlapping 2D detections into coherent 3D particle instances.
- Multiprocessing across multiple GPUs and data splits significantly reduces total inference time for large-scale competition submissions.
- Sparse particle picks can be effectively converted into dense multi-class segmentation masks for training 3D volumetric models.
- Using class-balanced random cropping helps mitigate severe class imbalance in cryoET data.
- Integrating MLflow with a custom training loop enables straightforward hyperparameter logging, metric tracking, and model versioning.
- Strong augmentations like mixup and copy-paste are effective for improving detection performance on particle imagery.
- Using a pretrained YOLO11 model with AdamW and cosine learning rate scheduling provides a stable and efficient training baseline.
- The Ultralytics YAML configuration format allows quick adaptation of custom datasets without modifying core training code.
- Classical Hessian-based filtering combined with watershed segmentation can effectively detect 3D blobs in cryo-ET data without requiring neural networks.
- Adapting processing scales and resolution levels to match known particle radii significantly improves detection accuracy across heterogeneous datasets.
- GPU-accelerated PyTorch convolutions can efficiently replace traditional scikit-image operations for large 3D tomograms while maintaining mathematical equivalence.

## What did not work
- The author initially attempted to train a 2D U-Net but abandoned it due to time constraints and inferior performance compared to the YOLO baseline.

## Notable individual insights
- 621 (Baseline UNet train + submit): Avoiding stochastic augmentations during validation prevents metric noise and ensures reliable model monitoring.
- 451 (CZII|YOLO11+Unet3D-Monai|LB.707): Blending 2D slice-based detection with 3D volumetric segmentation can improve robustness in cryo-ET particle identification.
- 340 (3d-unet using 2d image encoder): Custom evaluation metrics using the Hungarian algorithm for matching and class-weighted F-beta scores can accurately simulate competition leaderboards locally.
- 334 (CZII YOLO11 Submission Baseline): Graph-based DFS aggregation with geometric tolerance thresholds significantly refines noisy detection outputs into accurate 3D coordinates.
- 268 (CZII YOLO11 Submission Baseline with KDTree Update): KDTree spatial queries combined with Union-Find provide a fast and robust way to cluster overlapping 2D detections into coherent 3D particle instances.
- 236 (BlobDetector): Classical Hessian-based filtering combined with watershed segmentation can effectively detect 3D blobs in cryo-ET data without requiring neural networks.
- 250 (3D U-Net : Training Only): Sparse particle picks can be effectively converted into dense multi-class segmentation masks for training 3D volumetric models.

## Notebooks indexed
- #621 votes [[notebooks/votes_01_fnands-baseline-unet-train-submit/notebook|Baseline UNet train + submit]] ([kaggle](https://www.kaggle.com/code/fnands/baseline-unet-train-submit))
- #451 votes [[notebooks/votes_02_hideyukizushi-czii-yolo11-unet3d-monai-lb-707/notebook|CZII|YOLO11+Unet3D-Monai|LB.707]] ([kaggle](https://www.kaggle.com/code/hideyukizushi/czii-yolo11-unet3d-monai-lb-707))
- #396 votes [[notebooks/votes_03_kharrington-deepfindet-train/notebook|DeepFindET_Train]] ([kaggle](https://www.kaggle.com/code/kharrington/deepfindet-train))
- #340 votes [[notebooks/votes_04_hengck23-3d-unet-using-2d-image-encoder/notebook|3d-unet using 2d image encoder]] ([kaggle](https://www.kaggle.com/code/hengck23/3d-unet-using-2d-image-encoder))
- #334 votes [[notebooks/votes_05_itsuki9180-czii-yolo11-submission-baseline/notebook|CZII YOLO11 Submission Baseline]] ([kaggle](https://www.kaggle.com/code/itsuki9180/czii-yolo11-submission-baseline))
- #295 votes [[notebooks/votes_06_hengck23-1-hr-fast-2d-3d-unet-resnet34d-scanner-tta/notebook|1 hr fast 2d/3d-unet resnet34d scanner tta ]] ([kaggle](https://www.kaggle.com/code/hengck23/1-hr-fast-2d-3d-unet-resnet34d-scanner-tta))
- #268 votes [[notebooks/votes_07_sersasj-czii-yolo11-submission-baseline-with-kdtree-update/notebook|CZII YOLO11 Submission Baseline with KDTree Update]] ([kaggle](https://www.kaggle.com/code/sersasj/czii-yolo11-submission-baseline-with-kdtree-update))
- #250 votes [[notebooks/votes_08_ahsuna123-3d-u-net-training-only/notebook|3D U-Net : Training Only]] ([kaggle](https://www.kaggle.com/code/ahsuna123/3d-u-net-training-only))
- #243 votes [[notebooks/votes_09_itsuki9180-czii-yolo11-training-baseline/notebook|CZII YOLO11 Training Baseline]] ([kaggle](https://www.kaggle.com/code/itsuki9180/czii-yolo11-training-baseline))
- #236 votes [[notebooks/votes_10_kharrington-blobdetector/notebook|BlobDetector]] ([kaggle](https://www.kaggle.com/code/kharrington/blobdetector))
