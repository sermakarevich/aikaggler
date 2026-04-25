# cmi-detect-behavior-with-sensor-data: cross-solution summary

This competition focused on classifying gestures and behaviors from wearable sensor data (IMU, thermal, and time-of-flight) across multiple subjects. Winning approaches consistently relied on modality-specific deep learning architectures (CNNs, RNNs, Transformers) combined with rigorous handling of device orientation quirks, handedness normalization, and missing sensor data. Final success was driven by large-scale ensembling, constraint-aware post-processing, and dynamic model selection strategies that mitigated leaderboard variance and prevented data leakage.

## Competition flows
- Raw sensor sequences preprocessed for handedness/device orientation, passed through custom 1D-CNN/U-Net for modality-specific features and gesture segments, trained with seed averaging, and output ensemble averaged.
- Raw sensor data split into four missing-sensor variants, processed with coordinate corrections and phase-aligned mixup, fed into modality-specific CNN with phase-aware attention, and post-processed via no-repeat joint probability assignment.
- Raw sensor data cleaned/normalized/augmented, fed into large ensembles of CNN/GRU/LSTM/3D CNN with group-by-subject K-fold validation, logits blended, and post-processed for class/orientation constraints.
- Raw sequences grouped by subject, processed through CNN/RNN/Transformer with subject-level Transformer layer, dynamically selected/weighted based on missing values/historical data, and evaluated via repeated API submissions.
- Raw sensor data processed with minimal feature engineering/augmentations, fed into heavily bagged CNN/BERT models on full/IMU-only subsets, post-processed via constraint-aware max log-likelihood optimization on subject histories.
- Raw IMU/THM/TOF sequences preprocessed with modality-specific normalization/handedness flips, passed through separate multi-branch CNNs with group convs/SCSE attention, fused via GRUs/RoPE-enhanced MHA, trained with multi-task losses, and combined via constrained voting ensemble.
- Raw sensor sequences aligned/augmented, processed through two distinct multi-modal pipelines (cross-modal attention/auxiliary heads vs. gated single-model), and equally blended for submission.

## Data processing
- Extracted last 75 frames of sequences
- Aligned left-handed data by flipping signs/canceling true north
- Corrected upside-down devices via 180° z-axis rotation and channel swapping
- Replaced THM <20°C with nulls/filled with 0
- Replaced TOF -1 with 255 or 500
- Applied standardization/robust scaling
- Filled NaN/-1 in THM/TOF with zero
- Applied subject-specific coordinate corrections
- Horizontally flipped TOF grids/adjusted IMU channels for left-handed subjects
- Applied phase-aligned linear mixup
- Used online pseudo-labeling with test-time fine-tuning
- Removed subjects with incorrect wear/low gesture ratios/missing TOF
- Padded/cropped sequences to ~120 length
- Normalized handedness by flipping IMU axes/quaternions
- Grouped sequences by subject
- Sampled same-subject sequences at varying ratios
- Applied dynamic weighting based on historical record count
- Switched to IMU-only models when missing values >50%
- Dropped first/last records with specific phases
- Applied random noise
- Mixup of Transition/Gesture parts
- Scaled THM/TOF
- Removed gravity components from acceleration
- Approximated missing rot vectors from acc
- Removed moving average from acc vectors
- Flipped values for handedness == 0
- Normalized IMU/THM/TOF with modality-specific filters and handedness flips
- Applied 90° rotation of ToF sensors
- Handled NaNs via learnable embeddings/broadcasted parameters
- Applied random masking of THM/ToF features

## Features engineering
- derived_acc (linear/global acc)
- rotvec_diff (angular velocity proxy)
- TOF spatial convolution on 8x8 grid with stride-2 max pooling
- U-Net estimated gesture segments for temporal pooling
- IMU features (acc, quaternion 6D, angular/linear velocity)
- TOF features (tof1~5 grids)
- Composite label prediction
- 35 IMU features (jerks, correlations, gravity-removed acc, squared variants)
- Temporal differences of TOF/thermal data
- Rotational differences/magnitude calculations
- Independent processing of feature groups in early layers
- Scaled THM/TOF
- Acceleration without gravity
- Angles from rot vectors
- Rough estimates for missing rot vectors
- Removal of moving average from acc
- Flipping for handedness
- IMU features in device/world frames (gravity projection, abs_diff)
- Zero-padded feature matrices for group convolutions
- TOF nan indicator
- Raw acc/jerk
- 6D rotation matrix differences
- Gravity estimation/linear acc
- Subject-aligned tangential/lateral/vertical acc/jerk
- Tilt/difference calculations
- Temporal differences
- Rotation angle/velocity
- Distances
- Angular velocity norm
- 10-window rolling mean

## Models
- Custom 1D-CNN with MLP branches
- U-Net
- Grouped convolution
- Residual SE-CNN Block with Attention
- 2D CNN for TOF grids
- CNN GRU
- Pure CNN
- Dense GRU
- Self-excitation models
- 3D CNN
- LSTM
- Attention-based models
- RNN
- 1D CNN
- Transformer
- Bert-based architecture
- 1DResBlock
- 3DResBlock
- SCSE attention
- Multi-Head Attention (MHA)
- AttentionPooling
- MLP
- Depthwise separable convolutions
- BiLSTM with attention pooling
- GatingUnit_v2

## Loss functions
- Cross-entropy loss
- Auxiliary orientation prediction loss
- Weighted combination of predictions
- Focal loss
- Class weights
- Dice loss
- MSE
- MAE

## CV strategies
- 10-fold StratifiedGroupKFold
- 10-fold CV
- Group-by-subject K-fold (5-fold)
- Subject-based split with 5-fold CV

## Ensembling
- Averaged outputs of 100 neural networks via seed averaging
- Ensemble of 3 variants combined with post-processing maximizing conditional joint probability under no-repeat constraint
- Large ensembles of IMU-only, all-features, and TOF-only models blended as weighted sum of logits with post-processing
- Dynamic weighted fusion of subject-based and sequence-based models adjusted by missing value/historical count with final selection from repeated API submissions
- Blend of heavily bagged CNN and BERT-based models (full/IMU-only) with constraint-aware max log-likelihood optimization
- Class rank averaging across K-folds and voting across models restricted to high-CV performers
- Equal blend of two distinct pipelines (multi-modal vs. gated single-model)

## Notable individual insights
- Rank 6 (6th Place Solution): Aligning left-handed subject data distributions and correcting upside-down device placements were critical for accuracy.
- Rank 2 (2nd Place Solution): Training separate models for different missing sensor configurations yields better accuracy than handling missingness within a single model.
- Rank 1 (1st place solution): Using no scalers provided the largest improvement at the early training stage compared to using them.
- Rank 5 (5th place solution): Accumulating historical sequence information per subject significantly boosts performance, but degrades when historical data is insufficient.
- Rank 4 (4th place solution): Enforcing the "at most one sequence per target class per subject" constraint during inference significantly impacts final scores.
- Rank 12 (12th place solution): Voting across models performed better than class rank averaging, likely because it mitigates overconfident model predictions.
- Rank 17 (17th Place Solution): Manually flipping upside-down subjects for CV caused a significant CV/LB discrepancy, proving that orientation detection must be part of the training pipeline to avoid leakage.

## Solutions indexed
- #1 [[solutions/rank_01/solution|1st place solution]]
- #2 [[solutions/rank_02/solution|2nd Place Solution]]
- #4 [[solutions/rank_04/solution|4th place solution]]
- #5 [[solutions/rank_05/solution|5th place solution]]
- #6 [[solutions/rank_06/solution|6th Place Solution]]
- #12 [[solutions/rank_12/solution|12th place solution]]
- #17 [[solutions/rank_17/solution|17th Place Solution]]

## GitHub links
- [AshwinAshok3/cmi-detect-behavior-with-sensor-data](https://github.com/AshwinAshok3/cmi-detect-behavior-with-sensor-data) _(reference)_ — from [[solutions/rank_02/solution|2nd Place Solution]]
- [Yamato-Arai/kaggle-cmi-detect-behavior-2nd-place-solution](https://github.com/Yamato-Arai/kaggle-cmi-detect-behavior-2nd-place-solution) _(solution)_ — from [[solutions/rank_02/solution|2nd Place Solution]]
- [statist-bhfz/kaggle_cmi_1st_place_solution](https://github.com/statist-bhfz/kaggle_cmi_1st_place_solution) _(solution)_ — from [[solutions/rank_01/solution|1st place solution]]
