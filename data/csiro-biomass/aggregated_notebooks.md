# csiro-biomass: top public notebooks

The top-voted notebooks primarily focus on building robust regression pipelines for biomass estimation, heavily leveraging foundation vision models like DINOv2/v3 and SigLIP alongside custom fusion architectures such as Mamba blocks and FiLM layers. The community emphasizes semantic probing for feature engineering, strict post-processing constraints to enforce biological mass-balance and non-negativity, and extensive test-time augmentation combined with multi-fold averaging to stabilize predictions. A strong focus is placed on managing temporal distribution shifts through group-based cross-validation and optimizing large model training via gradient checkpointing and mixed precision.

## Common purposes
- ensemble
- baseline
- inference

## Competition flows
- Multi-pipeline ensemble combining SIGLIP embeddings and DINOv3 ViT with mass-balance constraints
- Memory-efficient inference pipeline with deterministic left/right cropping and target-specific scaling
- Multi-pipeline ensemble extracting semantic embeddings, training regressors/CNNs, and applying constraint reconciliation
- Baseline training/inference with artifact cleaning, dual-image splitting, Mamba fusion, and TTA averaging
- Temporal group-based baseline training with vertical image splitting and EfficientNet-B2
- Dense patch feature extraction via frozen DINOv2, shared MLP processing, and TTA averaging
- Production-ready inference aggregating three pre-trained vision models with extensive TTA and weighted averaging
- Complete training/inference pipeline for ViT-Huge-Plus-DINOv3 with Mamba fusion, EMA, and post-processing constraints
- Two-stream CNN inference predicting three targets directly and deriving two via constrained subtraction
- Post-processing ensemble merging five pre-generated CSV predictions via weighted averaging

## Data reading
- pd.read_csv for metadata splits and test CSVs
- cv2.imread and PIL.Image.open for raw image loading
- pandas pivot/melt operations to reshape long-format targets to wide and back
- Test images are loaded via PIL.Image.open and converted to RGB
- Reads CSV files for paths and labels, loads images via cv2.imread, converts BGR to RGB, handles missing images with zero-padding, and pivots/melts target columns for multi-output regression.
- Images are split into left/right halves (1000x1000 each) for dual-input processing
- Reads train.csv and test.csv via pandas, splits sample_id into prefixes/suffixes, groups rows by image path, and loads images using PIL's Image.open converted to RGB.
- Reads test.csv to retrieve image_path and sample_id, then opens each image using PIL.Image.open.
- Reads test.csv to extract image paths, loads images with cv2.imread, converts BGR to RGB, splits each image into left and right halves, and wraps them in a PyTorch Dataset that yields normalized tensor pairs.
- Reads train.csv and test.csv with pandas, loads images via cv2.imread, converts BGR to RGB, and parses target columns into numpy arrays.
- Reads test images using cv2.imread from paths listed in test.csv, converts BGR color space to RGB, and handles missing images by returning zero-filled arrays.
- Reads five submission CSV files (submission1.csv to submission5.csv) using pandas, each containing a sample_id column and a target column.

## Data processing
- Image patching for SIGLIP input
- Semantic probing via text prompts and cosine similarity scoring
- PCA, PLS, and GMM feature projections with standard/max scaling
- Orthogonal projection for mass-balance constraint enforcement
- Image cleaning (cropping bottom 10%, inpainting orange date stamps)
- Left/right image splitting and Albumentations augmentations (flips, rotation, color jitter)
- TTA via horizontal/vertical flips and 90° rotation
- Deterministic left/right image splitting
- Image resizing (224x224 to 768x768) and ImageNet normalization
- TTA with geometric flips, rotations, and Gaussian blur
- Target scaling (max-scaling, log1p) and constraint enforcement via projection matrices
- Gradient accumulation, EMA weight averaging, and gradient checkpointing for VRAM management
- HSV-based color masking and cv2.inpaint for artifact removal
- Model-specific input resizing and tensor conversion
- Independent augmentation on left/right patches
- Post-processing clamping (np.maximum(0, x)) and NaN handling
- DataFrame merging and weighted averaging for ensemble submissions

## Features engineering
- SIGLIP image embeddings
- Semantic probing ratios (green/dead/clover/bare/dense vegetation)
- PCA/PLS/GMM projections on scaled embeddings
- Target max-scaling and derived mass-balance targets (GDM_g, Dry_Total_g)
- Semantic cosine similarity scores from SigLIP text prompts
- Derived ratios (greenness, clover, cover)
- PCA variance retention, PLS regression components, and Gaussian Mixture probability features
- SigLIP embeddings extracted via sliding patches and mean-pooled
- Semantic features computed via prompt similarity (positive/negative agronomic descriptions)
- Derived targets computed post-hoc with non-negativity constraints

## Models
- SIGLIP-so400m-patch14-384
- CatBoostRegressor
- LightGBM
- HistGradientBoostingRegressor
- GradientBoostingRegressor
- vit_huge_plus_patch16_dinov3.lvd1689m
- LocalMambaBlock fusion neck
- BiomassModel (custom DINOv3 + LocalMambaBlock fusion + multi-head regression)
- CrossPVT_T2T_MambaDINO
- TiledFiLMDINO
- EfficientNet-B2
- FiLM
- MultiTargetRegressor
- DINOv2-Giant (frozen backbone)
- custom BiomassMLP
- EVA02Model
- convnext_tiny
- custom two-stream multi-head MLP architecture
- Multi-head regression architecture with Softplus activations
- SigLIP models (embedding extraction)

## Frameworks used
- PyTorch
- TorchVision
- Transformers
- scikit-learn
- LightGBM
- CatBoost
- XGBoost
- timm
- Albumentations
- pandas
- numpy
- PIL
- OpenCV
- matplotlib
- seaborn
- joblib

## Loss functions
- biomass_loss (custom weighted sum of Huber and MSE per target)
- nn.MSELoss
- nn.SmoothL1Loss(beta=5.0)
- Weighted SmoothL1Loss (Huber, beta=5.0) with R2-based weighting
- Sharpness-aware minimization (SAM)
- image-level loss
- Weighted nn.SmoothL1Loss (target-specific weights)

## CV strategies
- StratifiedGroupKFold(n_splits=5, groups=Sampling_Date, stratify=State)
- KFold(n_splits=5) using a pre-split fold column
- StratifiedGroupKFold with Sampling_Date grouping and State stratification
- GroupKFold and StratifiedGroupKFold split by Sampling_Date (temporal grouping)
- GroupKFold(n_splits=5) grouped by Sampling_Date

## Ensembling
- Weighted average of SigLIP tree and DINOv3 ViT submissions with orthogonal mass-balance projection
- Averaging two weighted models with target-specific scaling adjustments
- Averaging four tabular regressors and three model submissions with linear algebra reconciliation and negative clipping
- Averaging across TTA views and 5 folds with non-negativity constraints on derived targets
- Averaging across 5 K-Fold models with TTA during validation/inference
- Averaging across TTA augmentations and 5 fold models with ReLU clamping
- Weighted averaging of three diverse models (V4, MVP, EVA02) with internal checkpoint ensembling and non-negative clamping
- Averaging across 5 folds and 4 TTA views with post-processing derivation of constrained targets
- Averaging across 5 folds and 3 TTA views with constrained subtraction for remaining targets
- Weighted averaging of five pre-generated CSV predictions

## Insights
- Semantic probing can effectively extract domain-specific features from frozen foundation model embeddings without fine-tuning.
- Enforcing mass-balance constraints via orthogonal projection is critical for biologically plausible and competition-ready predictions.
- Gradient checkpointing and mixed precision are necessary to train large ViTs on limited GPU memory without OOM errors.
- Combining embedding-based tree models with end-to-end vision transformers improves robustness across heterogeneous data distributions.
- Loading models sequentially and clearing CUDA cache prevents OOM errors during multi-fold inference.
- Deterministic image splitting into left/right crops preserves spatial context without requiring complex data augmentation.
- Target-specific post-processing scaling can effectively calibrate regression outputs for submission without retraining.
- Semantic text embeddings from vision-language models can effectively generate domain-specific features for regression tasks.
- Tiling and processing left/right image halves separately captures broader spatial context than single-pass inference.
- Enforcing physical summation constraints via projection matrices significantly boosts competition metrics.
- Combining diverse backbones (SigLIP, DINOv2, DINOv3) with different fusion mechanisms yields complementary predictions.
- Splitting wide images into dual inputs improves feature extraction and model focus.
- Removing date stamp artifacts via HSV masking and inpainting prevents model confusion.
- Mamba blocks efficiently mix spatial tokens with linear complexity compared to standard attention.
- EMA stabilizes validation metrics and yields better generalization during training.
- Derived targets can be reliably computed post-inference with simple arithmetic constraints.
- Temporal distribution shift is the primary driver of train-test discrepancy, making date-based grouping essential for reliable validation.
- Splitting images vertically and using a FiLM layer to modulate shared backbone features effectively captures complementary spatial information.
- Softplus activation guarantees non-negative biomass predictions, aligning with physical constraints.
- TTA and multi-fold averaging significantly reduce prediction variance for regression targets.
- Dense patch features from foundation models can be efficiently aggregated via a shared MLP for regression tasks.
- TTA with geometric transforms and Gaussian blur significantly stabilizes inference.
- Clipping negative outputs is a necessary post-processing step for biomass regression.
- Extensive TTA across flips and rotations stabilizes predictions for diverse architectural backbones.
- Weighted ensembling of structurally different models (DINOv2, FiLM-DINO, EVA02) leverages complementary feature representations.
- Strict post-processing (clamping negatives, handling NaNs) is essential for valid multi-output regression submissions.
- Splitting large composite images into left/right halves and fusing them via a lightweight Mamba-style block effectively captures spatial context without excessive VRAM usage.
- Cleaning image artifacts (date stamps) via HSV masking and inpainting prevents the model from learning spurious patterns.
- Deriving constrained targets (e.g., clover = max(0, gdm - green)) post-inference ensures physically plausible biomass predictions.
- Using EMA weights and cosine annealing with warmup stabilizes training for foundation ViTs.
- Splitting high-resolution images into patches preserves fine-grained features like clover leaves that would be lost during direct resizing.
- Independent augmentation on left/right patches effectively multiplies training data diversity.
- Freezing the backbone initially stabilizes training on small datasets, and monitoring validation R^2 rather than loss prevents overfitting from capturing peak performance.
- Simple weighted averaging of diverse model predictions can effectively boost competition scores without additional training.
- External inference scripts can be orchestrated directly in a notebook to manage complex or time-consuming prediction pipelines.
- Proper column renaming and merging by a common identifier is crucial for aligning heterogeneous model outputs.

## Critical findings
- Dry_Clover_g is notoriously difficult to detect and is often fixed or handled separately to avoid prediction instability.
- Orange date stamps on images cause visual artifacts that require inpainting to prevent model confusion.
- Mass-balance constraints (Green+Clover=GDM, GDM+Dead=Total) must be applied post-prediction to maintain biological consistency.
- Gradient checkpointing reduces VRAM usage by approximately 50% for ViT-Large, enabling feasible training on standard Kaggle GPUs.
- The competition host confirmed that while States and Species are consistent across train/test, the test set includes non-overlapping time periods, causing significant temporal distribution shift.
- The Sampling_Date column is the most critical feature for aligning validation with the test distribution.
- Strong linear dependencies exist between the five biomass targets, allowing the model to predict only three directly and derive the others mathematically to avoid redundancy.
- Grouping folds by Sampling_Date is critical to prevent data leakage, as images from the same day are highly similar.

## What did not work
- 

## Notable individual insights
- votes 702 (siglep+dinov3): Enforcing mass-balance constraints via orthogonal projection is critical for biologically plausible and competition-ready predictions.
- votes 418 (csiro simple): Temporal distribution shift is the primary driver of train-test discrepancy, making date-based grouping essential for reliable validation.
- votes 245 ([LB 0.57]Infer + Model code): Strong linear dependencies exist between the five biomass targets, allowing the model to predict only three directly and derive the others mathematically to avoid redundancy.
- votes 340 (CSIRO | DINOv2 Dense Features): Dense patch features from foundation models can be efficiently aggregated via a shared MLP for regression tasks.
- votes 452 (CISRO|baseline train+infer 21/12 dinov3+siglip): Mamba blocks efficiently mix spatial tokens with linear complexity compared to standard attention.
- votes 232 (ensemble 5 models CSIRO | 0.7 score): Simple weighted averaging of diverse model predictions can effectively boost competition scores without additional training.

## Notebooks indexed
- #702 votes [[notebooks/votes_01_giovannyrodrguez-siglep-dinov3/notebook|siglep+dinov3]] ([kaggle](https://www.kaggle.com/code/giovannyrodrguez/siglep-dinov3))
- #541 votes [[notebooks/votes_02_mayukh18-dinov3-no-tta-postprocess/notebook|DinoV3 no TTA | PostProcess]] ([kaggle](https://www.kaggle.com/code/mayukh18/dinov3-no-tta-postprocess))
- #520 votes [[notebooks/votes_03_antonoof-0-69-ensemble-3-models-embeddings/notebook|0.69 ensemble 3 models + embeddings]] ([kaggle](https://www.kaggle.com/code/antonoof/0-69-ensemble-3-models-embeddings))
- #452 votes [[notebooks/votes_04_llkh0a-cisro-baseline-train-infer-21-12-dinov3-siglip/notebook|CISRO|baseline train+infer 21/12 dinov3+siglip ]] ([kaggle](https://www.kaggle.com/code/llkh0a/cisro-baseline-train-infer-21-12-dinov3-siglip))
- #418 votes [[notebooks/votes_05_jiazhuang-csiro-simple/notebook|csiro simple]] ([kaggle](https://www.kaggle.com/code/jiazhuang/csiro-simple))
- #340 votes [[notebooks/votes_06_carsoncheng-csiro-dinov2-dense-features-lb-0-66/notebook|🌿🌱 CSIRO | DINOv2 Dense Features | LB 0.66]] ([kaggle](https://www.kaggle.com/code/carsoncheng/csiro-dinov2-dense-features-lb-0-66))
- #296 votes [[notebooks/votes_07_antonoof-top90-3-models-ensemble-5h-inference/notebook|Top90 | 3 models ensemble 5h inference]] ([kaggle](https://www.kaggle.com/code/antonoof/top90-3-models-ensemble-5h-inference))
- #269 votes [[notebooks/votes_08_mattiaangeli-dinov3-baseline-lb-0-70/notebook|dinov3-baseline | LB 0.70]] ([kaggle](https://www.kaggle.com/code/mattiaangeli/dinov3-baseline-lb-0-70))
- #245 votes [[notebooks/votes_09_none00000-lb-0-57-infer-model-code/notebook|[LB 0.57]Infer + Model code ]] ([kaggle](https://www.kaggle.com/code/none00000/lb-0-57-infer-model-code))
- #232 votes [[notebooks/votes_10_antonoof-ensemble-5-models-csiro-0-7-score/notebook|ensemble 5 models CSIRO | 0.7 score]] ([kaggle](https://www.kaggle.com/code/antonoof/ensemble-5-models-csiro-0-7-score))
