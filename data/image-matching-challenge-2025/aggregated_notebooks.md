# image-matching-challenge-2025: top public notebooks

The community's top-voted notebooks for the Image Matching Challenge 2025 primarily focus on rapid baseline generation and exploratory data analysis, with a strong emphasis on chaining modern vision foundation models (like DINOv2, ALIKED, and LightGlue) with classical structure-from-motion solvers (COLMAP). Several entries also demonstrate traditional computer vision pipelines using SIFT/ORB with geometric verification, while others explore advanced multi-model orchestration, distributed inference, and direct end-to-end reconstruction using architectures like VGGT.

## Common purposes
- inference
- eda
- baseline
- tutorial

## Competition flows
- Reads image paths from a CSV, uses DINOv2 to shortlist pairs, extracts ALIKED keypoints/descriptors, matches them with LightGlue, imports results into COLMAP for pose estimation, and writes a formatted submission CSV with camera poses or outlier flags.
- Reads image paths from a CSV, uses DINOv2 to shortlist pairs via cosine similarity, extracts ALIKED keypoints and descriptors, matches them with LightGlue, imports results into COLMAP for incremental SfM reconstruction, and exports camera poses to a submission CSV.
- The notebook loads the competition dataset and labels, analyzes scene distributions and similarity thresholds, visualizes sample images, and demonstrates a baseline feature-matching pipeline using OpenCV's ORB detector and BFMatcher.
- Loads training/test image collections and metadata, extracts SIFT features, builds a pairwise geometric consistency graph, clusters images into scenes using connected components, performs incremental SfM via PnP to estimate camera poses per cluster, and formats the results into a submission.csv with cluster labels and pose matrices.
- Reads image paths from a CSV template, uses DINOv2 to shortlist candidate pairs, detects and matches features with ALIKED and LightGlue, runs COLMAP's RANSAC and incremental mapping to recover poses, and writes the final rotation/translation matrices to a submission CSV.
- Loads image paths from a CSV, shortlists pairs using DINOv2 global descriptors, extracts ALIKED keypoints and descriptors, matches them with LightGlue, runs PyColmap for RANSAC and incremental mapping, and outputs a CSV with predicted camera poses and cluster assignments.
- Loads the training labels CSV and image directories, inspects the distribution of scenes and target formats, and visualizes random samples from each dataset to understand the input-output structure for 3D reconstruction.
- Reads image paths from a CSV, uses DINOv2 to shortlist pairs, extracts ALIKED features, matches them with LightGlue, runs COLMAP incremental mapping to recover poses, and writes a formatted submission CSV.
- Loads a sequence of input images, passes them through a pretrained VGGT model to extract camera poses, depth maps, and 3D point clouds, then visualizes the reconstructed geometry and camera trajectories using Rerun.io.
- Sets up a custom image matching library and symlinks pre-trained vision models, loads a YAML pipeline configuration defining feature extractors, matchers, and a shortlisting ensemble, then executes the pipeline via distributed torchrun to generate a submission CSV.

## Data reading
- Loads image paths and metadata from a CSV (train_labels.csv or sample_submission.csv) using pandas.
- Reads images via Kornia (K.io.load_image) and PIL.
- Loads image paths and metadata from a CSV using pandas, iterates through rows to construct Prediction objects, and loads images using Kornia's load_image (RGB32 format) or LightGlue's load_image utility.
- pandas.read_csv used to load train_labels.csv and train_thresholds.csv
- cv2.imread with cv2.IMREAD_GRAYSCALE loads training images from /kaggle/input/image-matching-challenge-2025/train/
- Reads CSV metadata files (train_labels.csv, train_thresholds.csv, sample_submission.csv) using pandas.
- Loads images from directory structures using cv2.imread, handling .png, .jpg, and .jpeg extensions, and accounts for potential outliers subdirectories.
- Loads image paths and metadata from sample_submission.csv or train_labels.csv using pandas.
- Reads images via Kornia (K.io.load_image) and LightGlue's load_image utility.
- Stores intermediate keypoints and descriptors in HDF5 files using h5py.
- Parses image paths and metadata from `train_labels.csv` or `sample_submission.csv`.
- Loads images using `kornia.io.load_image` and `PIL.Image`.
- pd.read_csv() loads train_labels.csv containing dataset, scene, image_path, rotation_matrix, and translation_vector columns.
- Target columns rotation_matrix and translation_vector are stored as flattened vectors with values separated by semicolons.
- Images are loaded via PIL.Image.open() from scene-specific directories under /kaggle/input/image-matching-challenge-2025/train/.
- Uses glob to collect image paths from the competition directory
- Passes paths to vggt.utils.load_fn.load_and_preprocess_images for tensor conversion
- Data path is set via the DEFAULT_DATASET_DIR environment variable; actual loading and formatting are abstracted by the ns64_imc2025lib library.

## Data processing
- Resizes images to 1024px for ALIKED feature extraction.
- Normalizes DINOv2 descriptors via L2 normalization.
- Filters candidate pairs using a cosine distance threshold (sim_th = 0.3) and minimum pair count (min_pairs = 20).
- Saves intermediate keypoints, descriptors, and matches in HDF5 files.
- Converts missing poses to semicolon-separated 'nan' strings for CSV formatting.
- Converts semicolon-separated threshold strings into lists of floats using str.split and map
- Flattens threshold lists with np.concatenate and plots histograms
- Loads images in grayscale for feature detection
- Applies Lowe's ratio test (threshold 0.8) to filter ambiguous feature matches.
- Uses RANSAC with a 1.5-pixel reprojection threshold for Fundamental Matrix estimation.
- Filters graph clusters by a minimum size of 3 images to discard noise/outliers.
- Approximates camera intrinsics using a default matrix (focal length = 1.2 * max(width, height), principal point at image center) since real intrinsics are unavailable.
- DINOv2 processor normalizes and rescales images for global descriptor extraction.
- ALIKED auto-resizes inputs to 1024px during feature extraction.
- Shortlisting filters candidate pairs using a Euclidean distance threshold (0.3) on normalized DINO descriptors.
- Images are loaded as RGB32 tensors and moved to GPU for inference.
- Resizes inputs to 1024 for ALIKED feature extraction.
- Normalizes DINOv2 global descriptors using L2 normalization.
- Converts images to float32 tensors on GPU.
- Formats rotation and translation arrays as semicolon-separated strings for the submission CSV.
- load_and_preprocess_images for input formatting
- Automatic mixed precision (bfloat16 or float16) during inference based on GPU capability
- Percentile-based confidence thresholding (90th percentile) to mask low-confidence depth and point values
- Pipeline configuration specifies resizing functions (lg_resize: 1280, ml_resize: 1600), NMS radius (4), keypoint thresholds (0.0005), border removal (4), and match thresholds (1.001) for local feature extraction and matching.

## Models
- DINOv2
- ALIKED
- LightGlue
- COLMAP/PyColmap
- ORB
- VGGT
- Grounding DINO
- SAM
- SigLIP2
- SuperPoint
- MAST3R
- ISC

## Frameworks used
- PyTorch
- Kornia
- Hugging Face Transformers
- LightGlue
- PyColmap
- pandas
- NumPy
- h5py
- OpenCV
- Pillow
- Matplotlib/Seaborn
- NetworkX
- scikit-learn
- tqdm
- Plotly
- MediaPy
- Python Standard Library
- glob
- jaxtyping
- Rerun
- VGGT
- PyYAML
- Open3D
- ns64_imc2025lib

## CV strategies
- 

## Ensembling
- 
- The shortlist generator uses an ensemble of four retrieval methods (MAST3R-ASMK, MAST3R-SPoC, DINOv2, and ISC) with fallback thresholds and pair-swapping removal to improve matching robustness.

## Insights
- votes 825 (Baseline: DINOv2+ALIKED+LightGLUE): DINOv2 global descriptors effectively shortlist image pairs, reducing the combinatorial explosion of exhaustive matching.
- votes 412 (Baseline: DINO+ALIKED+LightGLUE): Lowering COLMAP's min_model_size to 3 ensures successful reconstruction even for scenes with very few registered images.
- votes 182 (Image Matching Challenge 2025): Geometric verification via RANSAC on the Fundamental Matrix is essential for robustly filtering false matches in unstructured image collections.
- votes 117 ([EDA] 📸 IMC - 📊 &📍Locations): Training data features sequential capture ordering with high image-to-image overlap, whereas the test set uses limited overlap and randomized ordering.
- votes 84 (Sample Code for VGGT Inference): VGGT can directly output camera poses, depth maps, and 3D point maps from image sequences without traditional SfM pipelines.
- votes 71 (imc2025-1st-place-solution): A unified YAML configuration can effectively orchestrate a complex, multi-model image matching and reconstruction pipeline.

## Critical findings
- votes 412 (Baseline: DINO+ALIKED+LightGLUE): COLMAP defaults to requiring at least 10 registered images to generate a reconstruction, so the mapper options must be explicitly adjusted to min_model_size=3 to avoid empty outputs.
- votes 412 (Baseline: DINO+ALIKED+LightGLUE): ALIKED has known precision issues with float16, requiring explicit casting to float32 for stable feature extraction.
- votes 182 (Image Matching Challenge 2025): Real camera intrinsics are typically unknown in this challenge, forcing a simplified default matrix approximation that may limit pose accuracy.
- votes 182 (Image Matching Challenge 2025): Bundle adjustment is acknowledged as necessary for high accuracy but is explicitly skipped due to computational complexity and kernel runtime limits.
- votes 114 (Baseline: DINO+ALIKED+LightGLUE edit Params): The author notes that the DINO similarity threshold (sim_th = 0.3) should be kept strict to avoid false positive matches.
- votes 114 (Baseline: DINO+ALIKED+LightGLUE edit Params): COLMAP's default min_model_size of 10 can cause reconstruction failures on scenes with fewer images, so lowering it to 8 is recommended.

## What did not work
- votes 182 (Image Matching Challenge 2025): Full bundle adjustment was skipped due to computational intensity and kernel runtime constraints, with the author opting for robust PnP registration instead.
- votes 182 (Image Matching Challenge 2025): Spectral clustering requires heuristic cluster estimation and can be unstable, so Connected Components is preferred as a reliable fallback.

## Notable individual insights
- votes 825 (Baseline: DINOv2+ALIKED+LightGLUE): DINOv2 global descriptors effectively shortlist image pairs, reducing the combinatorial explosion of exhaustive matching.
- votes 412 (Baseline: DINO+ALIKED+LightGLUE): Lowering COLMAP's min_model_size to 3 ensures successful reconstruction even for scenes with very few registered images.
- votes 182 (Image Matching Challenge 2025): Geometric verification via RANSAC on the Fundamental Matrix is essential for robustly filtering false matches in unstructured image collections.
- votes 117 ([EDA] 📸 IMC - 📊 &📍Locations): Training data features sequential capture ordering with high image-to-image overlap, whereas the test set uses limited overlap and randomized ordering.
- votes 84 (Sample Code for VGGT Inference): VGGT can directly output camera poses, depth maps, and 3D point maps from image sequences without traditional SfM pipelines.
- votes 71 (imc2025-1st-place-solution): A unified YAML configuration can effectively orchestrate a complex, multi-model image matching and reconstruction pipeline.

## Notebooks indexed
- #825 votes [[notebooks/votes_01_octaviograu-baseline-dinov2-aliked-lightglue/notebook|Baseline: DINOv2+ALIKED+LightGLUE]] ([kaggle](https://www.kaggle.com/code/octaviograu/baseline-dinov2-aliked-lightglue))
- #412 votes [[notebooks/votes_02_itahiro-baseline-dino-aliked-lightglue/notebook|Baseline: DINO+ALIKED+LightGLUE]] ([kaggle](https://www.kaggle.com/code/itahiro/baseline-dino-aliked-lightglue))
- #207 votes [[notebooks/votes_03_pragyatripathiii23-beginner-friendly-1st-submission-eda-orb/notebook|[Beginner Friendly] 1st Submission - EDA&ORB]] ([kaggle](https://www.kaggle.com/code/pragyatripathiii23/beginner-friendly-1st-submission-eda-orb))
- #182 votes [[notebooks/votes_04_olaflundstrom-image-matching-challenge-2025/notebook|Image Matching Challenge 2025]] ([kaggle](https://www.kaggle.com/code/olaflundstrom/image-matching-challenge-2025))
- #165 votes [[notebooks/votes_05_eduardtrulls-imc25-submission/notebook|IMC25-submission]] ([kaggle](https://www.kaggle.com/code/eduardtrulls/imc25-submission))
- #121 votes [[notebooks/votes_06_liuhaixu-fork-of-lb-28-12-first-dummy-submission-25aea8/notebook|Fork of [LB: 28.12] FIRST DUMMY SUBMISSION  25aea8]] ([kaggle](https://www.kaggle.com/code/liuhaixu/fork-of-lb-28-12-first-dummy-submission-25aea8))
- #117 votes [[notebooks/votes_07_sharifi76-eda-imc-locations/notebook|[EDA] 📸 IMC - 📊 &📍Locations]] ([kaggle](https://www.kaggle.com/code/sharifi76/eda-imc-locations))
- #114 votes [[notebooks/votes_08_suthcong-baseline-dino-aliked-lightglue-edit-params/notebook|Baseline: DINO+ALIKED+LightGLUE edit Params]] ([kaggle](https://www.kaggle.com/code/suthcong/baseline-dino-aliked-lightglue-edit-params))
- #84 votes [[notebooks/votes_09_columbia2131-sample-code-for-vggt-inference/notebook|Sample Code for VGGT Inference]] ([kaggle](https://www.kaggle.com/code/columbia2131/sample-code-for-vggt-inference))
- #71 votes [[notebooks/votes_10_ns6464-imc2025-1st-place-solution/notebook|imc2025-1st-place-solution]] ([kaggle](https://www.kaggle.com/code/ns6464/imc2025-1st-place-solution))
