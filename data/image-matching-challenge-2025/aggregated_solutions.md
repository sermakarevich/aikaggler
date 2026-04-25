# image-matching-challenge-2025: cross-solution summary

The Image Matching Challenge 2025 focused on robust 3D reconstruction and scene clustering from unstructured image collections, where winning approaches prioritized efficient candidate pair generation and precise geometric filtering over end-to-end model training. Top solutions consistently leveraged hybrid pipelines combining foundation models (like DINOv2 and MASt3R) for retrieval with modern local feature matchers (ALIKED and LightGlue) for dense correspondence. Success hinged on strategic multi-scale matching, dynamic pair pruning, and careful COLMAP parameter tuning to maximize reconstruction coverage while mitigating cluster bias and computational bottlenecks.

## Competition flows
- Raw images processed through multi-feature retrieval shortlist, matched via hybrid MASt3R/keypoint pipeline, and reconstructed with COLMAP.
- Orientation-corrected images paired via retrieval, filtered by linear transformer classifier, matched with RDD/LightGlue at multiple scales, and aggregated with TTA.
- Global-to-local feature matching with pruning/clustering, followed by multiple COLMAP incremental mapping runs per cluster and native API merging.
- CLIP-screened pairs matched with gimlightglue/alike_lightglue, filtered via DBSCAN and loop consistency, truncated to top 1500 pairs.
- Dynamic top-k pair selection via KeyNet-AdaLAM, ALIKED/LightGlue matching with DBSCAN cropping, and MAGSAC++ outlier removal via pycolmap.
- Orientation-aligned images matched at multiple scales with ALIKED/LightGlue, filtered via logarithmic formula/graph clustering, enhanced with 4-split/crops, and reconstructed with COLMAP.
- Pre-rotated images processed through ALIKED/LightGlue with multi-scale TTA and optimized thresholds for final matches.
- Multi-resolution feature extraction (ALIKED/LightGlue/DISK), optional global pre-clustering, RANSAC geometry computation, and COLMAP incremental mapping with multi-model generation.
- DINOv2 pre-matching, tiled-image feature extraction with ALIKED/LightGlue, RANSAC filtering, and pycolmap reconstruction with ascending-order map registration.
- Pairwise matching/rotation correction, similarity graph construction from inlier counts for connected component clustering, and track refinement.
- 1280px resized images processed through SuperPoint/ALIKE/SIFT ensembles with LightGlue/NN on full/cropped images, RANSAC filtered, and reconstructed with COLMAP.
- DINOv2-filtered pairs processed through multi-resolution ALIKED/LightGlue matching with DBSCAN cropping and RANSAC removal, fed into COLMAP for incremental reconstruction.

## Data processing
- Orientation correction/rotation (confidence thresholding, natural orientation, pre-rotation, augmentation)
- Multi-scale resizing (840, 1024, 1280, 1536, 2048, 2560px)
- Dynamic top-k pair selection based on visual similarity
- Pair pruning via linear transformer classifier / DINOv2 cosine similarity thresholds
- NetVLAD pair compensation
- GPU-accelerated Euclidean distance matrix computation for lightweight filtering
- Match thresholding (min distance, valid matches)
- Graph-based edge pruning / degree capping
- Clustering via weakly connected components / inlier counts
- Map merging based on shared images and reprojection error
- DBSCAN-based image cropping / dense correspondence region identification
- MAGSAC++ outlier removal
- Splitting images into quadrants / half-resolution originals
- Cropping high-match-density regions
- Caching keypoints and descriptors
- Multi-GPU acceleration with mixed precision
- Restoring keypoint coordinates to original space
- Track refinement
- Dataset partitioning by balancing sum of squares of image counts

## Models
- MASt3R
- ALIKED
- DINOv2
- LightGlue
- SuperPoint
- VGGT
- GLOMAP
- RDD
- NetVLAD
- COLMAP
- CLIP
- CNN
- Logistic Regression
- YOLO
- KeyNet-AdaLAM
- DISK
- SigLIP-v2
- MegaLoc
- SIFT
- VGGSfM
- RANSAC
- DBSCAN
- Linear Transformer
- Orientation Correction Model

## Frameworks used
- pytorch
- COLMAP
- pycolmap
- NetworkX
- COLMAP native APIs

## CV strategies
- Holdout validation set composed of IMC and MegaDepth samples
- CV used all training datasets; no specific fold or holdout strategy mentioned

## Ensembling
- TTA by running RDD+LightGlue matching on two image sizes (1024, 1280) and merging results
- Integrating gimlightglue and alike_lightglue outputs, followed by DBSCAN filtering, loop consistency pruning, and truncation to top 1500 matches
- Merging results from four matching scales (1024, 1280, 1536, 2048) and filtering before COLMAP; comparing two separate notebook pipelines
- TTA across four image scales with 8192 features combined with threshold tuning to stabilize leaderboard scores
- Ensembling matches from multiple scales (1280, 1536, 2048) on original/cropped images and combining ALIKED/LightGlue with DISK detectors
- Concatenating matches from SuperPoint, ALIKE, and SIFT on full/cropped images, filtered via RANSAC with ≥150 matches before reconstruction

## Notable individual insights
- rank 1 (1st Place Solution): Using MASt3R as a semi-dense, detector-free matcher significantly improves precision and robustness in messy scenes compared to traditional detector-based methods.
- rank 4 (4th Place Solution): Framing image clustering as a binary pair classification problem effectively filters false matches, and strategic pair compensation can balance runtime and accuracy when pruning removes true matches.
- rank 11 (11th place solution): Pruning image pairs to cap node degree at 6 prevents incremental mapping from being biased toward highly connected clusters, improving reconstruction of smaller structures.
- rank 8 (8th Place Solution): CLIP outperforms DINOv2 for scene screening due to more accurate segmentation, though it struggles with ambiguous scenes like staircases.
- rank 10 (10th Place Solution): Dynamic top-k selection is crucial for handling scenes with varying visual similarity, as global descriptors fail to distinguish visually similar scenes whereas keypoint matching successfully identifies adjacent views.
- rank 3 (3rd Place Solution): Tiling an image into four quadrants plus a half-resolution original creates 25 matching combinations per pair, greatly improving robustness to scale and perspective differences.
- rank 12 (12th Place Solution - Pushing Limits of COLMAP): Graph-based clustering outperforms density-based methods like HDBSCAN because bridging images in the same scene disrupt high-density regions.

## Solutions indexed
- #1 [[solutions/rank_01/solution|1st Place Solution]]
- #2 [[solutions/rank_02/solution|2nd Place Solution]]
- #3 [[solutions/rank_03/solution|3rd Place Solution]]
- #4 [[solutions/rank_04/solution|4th Place Solution]]
- #6 [[solutions/rank_06/solution|6th Place Solution]]
- #7 [[solutions/rank_07/solution|7th Place Solution]]
- #8 [[solutions/rank_08/solution|8th Place Solution]]
- #9 [[solutions/rank_09/solution|9th Place Solution: DINOv2-Optimized Filtering]]
- #10 [[solutions/rank_10/solution|10th Place Solution]]
- #11 [[solutions/rank_11/solution|11th place solution]]
- #12 [[solutions/rank_12/solution|12th Place Solution - Pushing Limits of COLMAP]]
- #15 [[solutions/rank_15/solution|15th Place Solution]]

## GitHub links
- [cvg/mpsfm](https://github.com/cvg/mpsfm) _(reference)_ — from [[solutions/rank_01/solution|1st Place Solution]]
- [naver/mast3r](https://github.com/naver/mast3r) _(library)_ — from [[solutions/rank_01/solution|1st Place Solution]]
- [ns-rokuyon/kaggle-image-matching-challenge-2025](https://github.com/ns-rokuyon/kaggle-image-matching-challenge-2025) _(solution)_ — from [[solutions/rank_01/solution|1st Place Solution]]
- [ternaus/check_orientation](https://github.com/ternaus/check_orientation) _(library)_ — from [[solutions/rank_04/solution|4th Place Solution]]
- [xtcpete/rdd](https://github.com/xtcpete/rdd) _(solution)_ — from [[solutions/rank_04/solution|4th Place Solution]]
- [yangyefd/IMC2025](https://github.com/yangyefd/IMC2025) _(solution)_ — from [[solutions/rank_08/solution|8th Place Solution]]
- [xuelunshen/gim](https://github.com/xuelunshen/gim) _(reference)_ — from [[solutions/rank_08/solution|8th Place Solution]]
- [facebookresearch/vggt](https://github.com/facebookresearch/vggt) _(reference)_ — from [[solutions/rank_10/solution|10th Place Solution]]
- [ternaus/check_orientation](https://github.com/ternaus/check_orientation) _(library)_ — from [[solutions/rank_06/solution|6th Place Solution]]
- [MujtabaJunaid/imc-2025-naive-attempt](https://github.com/MujtabaJunaid/imc-2025-naive-attempt) _(solution)_ — from [[solutions/rank_03/solution|3rd Place Solution]]
- [nianticlabs/mickey](https://github.com/nianticlabs/mickey) _(reference)_ — from [[solutions/rank_09/solution|9th Place Solution: DINOv2-Optimized Filtering]]
- [zju3dv/LoFTR](https://github.com/zju3dv/LoFTR) _(reference)_ — from [[solutions/rank_09/solution|9th Place Solution: DINOv2-Optimized Filtering]]

## Papers cited
- [MASt3R-SfM](https://arxiv.org/abs/2409.19152)
- [ISC](https://arxiv.org/abs/2112.04323)
- [MP-SfM](https://arxiv.org/abs/2504.20040)
- [Speedy MASt3R](https://arxiv.org/abs/2503.10017)
- [RDD: Robust Feature Detector and Descriptor using Deformable Transformer](https://xtcpete.github.io/rdd/)
- [VGGT: Visual Geometry Grounded Transformer](https://github.com/facebookresearch/vggt)
