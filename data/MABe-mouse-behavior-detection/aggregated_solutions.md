# MABe-mouse-behavior-detection: cross-solution summary

This competition focused on recognizing social actions in mice across heterogeneous laboratory environments, requiring robust handling of inconsistent tracking data, varying frame rates, and lab-specific arena geometries. Winning approaches predominantly combined extensive cross-lab feature standardization with hybrid architectures (CNNs, Transformers, LSTMs, and gradient boosting), leveraging multi-scale temporal modeling, pairwise keypoint interactions, and rigorous threshold calibration. Success hinged on maximizing cross-lab transfer through invariant feature engineering, stratified validation schemes, and sophisticated ensembling strategies that balanced model diversity with metric-aligned loss optimization.

## Competition flows
- Raw tracking data transformed into lab-invariant features, fed into CNN-Transformer with multi-scale windows/augmentation, post-processed with smoothing/thresholds, and stacked with XGBoost
- Raw multi-lab data filtered to common body parts, enriched with ~60 features/embeddings, fed into blended LSTM/XGBoost with small batches/cosine decay, and threshold-optimized
- Raw tracking data converted to 30 FPS, fed into CNN/Transformer & SqueezeFormer with lab-stratified CV, sliding-window inference with flip TTA, and FPS scaling adjustment
- Raw coordinates normalized/resampled to 30 FPS, converted to 24D spatial-temporal features, fed into ST-GCN + Transformer with hybrid loss, and probability-averaged across varying architectures
- Raw frames/tracks assembled into geometric/kinematic space, split into video folds, trained as per-action models with negative sampling/temporal features, calibrated via Platt scaling, and smoothed/thresholded
- Standardized keypoints engineered into spatial/relational/temporal features, fed into per-lab XGBoost/3D CNN classifiers, and combined via CV selection/majority voting

## Data reading
- Combined input data for all labs, selecting 5 body parts present in most of the data
- Frames and tracks assembled per video_id and (agent_id, target_id) pair from whitelist, with behavior annotations loaded from train/test CSV files

## Data processing
- Transform raw coordinates into lab-invariant features using agent-centric coordinate systems and min-max scaling
- Rename inconsistent body part labels across labs to unified side_left/side_right
- Apply train-time augmentations (random FPS changes, rotation, x/y scaling, horizontal flips, body part relabeling, dropout)
- Post-process with temporal probability smoothing, invalid action filtering, and lab-specific thresholds
- Apply categorical embedding to 'Laboratory code' and mask invalid actions in softmax output
- Use small batches (16/32) with CosineDecayRestarts schedule
- Convert/resample tracking data to 30 FPS using linear/nearest-neighbor interpolation
- Standardize distance, velocity, and acceleration using precomputed statistics
- Fill missing values via linear interpolation or mean of existing features
- Apply Gaussian noise, time stretching, frame shift, left/right flipping, and Mixup
- Normalize (x, y) coordinates per lab using mean/std and map disparate keypoint names to a unified schema
- Apply affine transforms, mouse ID shuffling, CutMix, and keypoint dropout
- Apply horizontal flip averaging for TTA
- Normalize descriptors by agent body length
- Partition unlabeled frames into real and maybe negatives for sampling
- Smooth frame-level predictions via rolling windows and merge into consecutive action intervals
- Standardize sensors to specific keypoints (nose, ears, body_center, tail_base) and derive/interpolate missing ones
- Replace head with nose, remove headpiece/spine, and use linear temporal interpolation with binary flags
- Interpolate missing x/y coordinates while preserving availability masks

## Features engineering
- Agent-centric coordinate transformations and min-max scaled positions
- Inter-body-part distances (vector, magnitude, raw, and aggregated)
- Velocity vectors/magnitudes, acceleration, and jerk features
- pixels_per_cm feature
- Approximately 60 hand-crafted features (distances, angles, speeds, metadata)
- Categorical embedding for 'Laboratory code'
- Meta features: sex, frame-rate scale, lab ID (embedding), candidate action label
- Per-frame features: body-part distances, velocity, acceleration (solo vs pair configurations)
- 24-dimensional keypoint features: normalized x, y, velocity, acceleration (Δt=2)
- Relative position, relative velocity, Euclidean distance, and approach velocity to other mice
- Original FPS passed as an additional feature
- Lab, FPS, and action embeddings concatenated to pairwise features
- Invariant agent-centric geometry and kinematics
- Single-mouse kinematic descriptors
- Rolling mean summaries over three lagged windows
- Pair-wise distances between sensor keypoints
- Velocity for ear and tail
- Elongation vs. compactness features
- Body angle features
- Body center displacement, acceleration, and jerk features
- Angular velocity and turn-rate
- Grooming and allogroom specific features
- Mouse and pair metadata (sex, color, device, is_same_sex, is_same_color)
- Arena boundary proximity features
- Cross-mice relative speed, acceleration, jerk, and orientation
- Sliding window feature copying and window statistics aggregations (median, min, max, std)

## Models
- CNN Transformer
- XGBoost
- LSTM with dense layers
- CNN + RNN/Transformer (Multi-scale Conv, Conv Fusion, Bi-GRU, Transformer, Linear)
- SqueezeFormer
- ST-GCN
- Transformer
- LightGBM
- Catboost
- PyBoost
- TABM
- 3D CNN (dilated convolutions, residual connections, GroupNorms, InstanceNorm3d, ELU, pooling, gated max classification)

## Frameworks used
- XGBoost
- LightGBM
- Catboost
- PyBoost
- TABM

## Loss functions
- Binary Cross Entropy
- Cross-Entropy
- MacroSoftF1
- α × CE + (1-α) × MacroSoftF1
- BCEWithLogitsLoss (masked for NaN targets)
- Weighted by sqrt(binary_label_mean)

## CV strategies
- Leave-out validation scheme excluding labs MABe22, CalMS21, and CRIM13
- Out-of-fold predictions used for threshold optimization via grid search
- 5-fold stratified cross-validation stratified by test-set lab IDs (CalMS21/CRIM13 always in training)
- 4-fold StratifiedGroupKFold grouped by video_id and stratified by lab_id (CalMS21/CRIM13 always in training)
- Video-level 5-fold cross-validation with complete OOF predictions
- 4-fold CV grouped per video and stratified by action label with OOF threshold optimization

## Ensembling
- Equal-weighted average of multiple CNN-Transformers across feature families and time scales, blended with XGBoost on OOF outputs
- Blending NN and XGB probabilities with lab/action-specific OOF thresholds and conflict resolution via probability-to-threshold ratio
- Averaging five model variants and fold-specific thresholds with flip TTA and per-lab-action threshold selection
- Averaging probabilities across models with varying keypoints, sequence lengths, and architectures, plus horizontal flip TTA
- Weighted combination using OOF-fitted ensemble weights, temporal smoothing, per-action threshold fitting, and argmax selection
- Selecting per-lab models with highest CV scores, with frame-wise majority voting for specific labs

## Insights
- Transfer learning across labs is maximized by making all input features invariant to lab-specific arena sizes and tracked body parts.
- Correctly defining positive, negative, and masked targets is critical for training the multi-action output layer.
- Training the same architecture across multiple time scales (2s to 16s) with corresponding dilated convolutions captures diverse temporal dynamics.
- Data augmentation not only prevents overfitting but also harmonizes visual distributions across different laboratory environments.
- Combining data across all labs and focusing on five consistently present body parts allows the model to learn robust patterns across different experimental setups.
- Extensive feature engineering was more impactful than model complexity.
- Stratifying cross-validation folds by test-set lab IDs ensures validation metrics closely reflect the test distribution.
- Optimizing thresholds per lab-action pair at each epoch during training significantly improves early stopping and final calibration.
- Separating models for solo and pair actions prevents feature interference and improves task-specific performance.
- Jointly modeling all 16 (agent, target) pairs with cross-pair attention significantly improves performance over non-pairwise approaches.
- Directly optimizing a differentiable Macro Soft F1 approximation with a gradual schedule yields better metric alignment than standard Cross-Entropy alone.
- Per-lab coordinate normalization and unified keypoint mapping are critical for handling heterogeneous lab data.
- Formulating the problem as per-action binary tasks per lab significantly simplified modeling and improved tractability.
- Video-level cross-validation was critical for achieving strong CV-LB correlation and preventing data leakage.
- Raw boosting margins require Platt calibration with prior correction to be comparable across different actions and labs.
- Distinguishing between real negatives and maybe negatives during negative sampling helped control the true negative to unlabeled data ratio.
- Standardizing sensor keypoints across labs is essential for handling missing data.
- XGBoost handles high-dimensional features from sliding window copies robustly.
- Weighting loss by sqrt(binary_label_mean) effectively balances rare classes.
- Gated max classification improves performance when mice are visually mixed.

## Critical findings
- Excluding three specific labs from the validation set created a near-perfect correlation with the public leaderboard, though this correlation weakened for the private leaderboard.
- Renaming anatomically distinct but functionally similar body parts (hip/lateral) to a unified label significantly improved cross-lab transfer learning.
- Using inter-body-part distances and agent-centric coordinates naturally resolved arena size/shape discrepancies without explicit normalization.
- Some lab videos contain mislabeled actions that organizers did not correct.
- Certain labs exhibit actions with fixed lengths (e.g., multiples of 15 frames for BoisterousParrot, exactly 9 frames for CautiousGiraffe with alternating mouse attacks).
- Applying a 30/25 frame rate scaling adjustment to AdaptableSnail's 25 FPS tracking data improved the private score by +0.009, but only when mouse ID swaps were absent.
- Discarding 32 frames at both boundaries of the sliding window prevents edge artifacts and stabilizes prediction averaging.
- Models trained with pairwise feature extraction and cross-pair attention achieve a notable gap between LB (~0.525) and CV (~0.510) compared to non-pairwise models (~0.500 both), indicating strong generalization gains from explicit pair interactions.
- Gradually shifting the loss weight from Cross-Entropy to MacroSoftF1 over 32 epochs is crucial for aligning training with the competition metric.
- Long continuous action events can dominate the loss in boosting models, necessitating inverse length weighting for positive frames.
- Unlabeled frames must be carefully partitioned into real negatives and maybe negatives based on whitelist presence to avoid introducing label noise.
- Temporal interpolation significantly helps some labs but fails in others due to tracking ID-switches, necessitating binary interpolation flags.
- Training binary classifiers per action and combining them via residual thresholds outperforms direct multi-class approaches.

## What did not work
- Directly exploiting the noted data properties (mislabeled actions, fixed action lengths) did not yield significant benefits in this submission.
- Pseudo-labeling on test data
- Semi-supervised learning with MABe22 unlabeled data
- Merging rare classes per lab in training (e.g., AdaptableSnail: avoid → escape)

## Notable individual insights
- rank 7 (7th Place Gold - CNN Transformer with Invariant Features): Renaming anatomically distinct but functionally similar body parts (hip/lateral) to a unified label significantly improved cross-lab transfer learning.
- rank 2 (2nd place solution): Applying a 30/25 frame rate scaling adjustment to AdaptableSnail's 25 FPS tracking data improved the private score by +0.009, but only when mouse ID swaps were absent.
- rank 10 (10th Place Solution (ST-GCN + Transformer)): Gradually shifting the loss weight from Cross-Entropy to MacroSoftF1 over 32 epochs is crucial for aligning training with the competition metric.
- rank 12 (13th Public/12th Private Solution): Long continuous action events can dominate the loss in boosting models, necessitating inverse length weighting for positive frames.
- rank 4 (4th Place Solution - XGB + NN Ensemble): Temporal interpolation significantly helps some labs but fails in others due to tracking ID-switches, necessitating binary interpolation flags.
- rank 10 (10th Place Solution (ST-GCN + Transformer)): Jointly modeling all 16 (agent, target) pairs with cross-pair attention significantly improves performance over non-pairwise approaches.

## Solutions indexed
- #2 [[solutions/rank_02/solution|2nd place solution]]
- #3 [[solutions/rank_03/solution|3rd place solution]]
- #4 [[solutions/rank_04/solution|4th Place Solution - XGB + NN Ensemble]]
- #7 [[solutions/rank_07/solution|7th Place Gold - CNN Transformer with Invariant Features]]
- #10 [[solutions/rank_10/solution| 10th Place Solution (ST-GCN + Transformer)]]
- #12 [[solutions/rank_12/solution|13th Public/12th Private Solution]]
- ? [[solutions/rank_xx_609063/solution|place holder: 1d object detection solution]]

## GitHub links
- [neuroethology/MARS](https://github.com/neuroethology/MARS) _(reference)_ — from [[solutions/rank_xx_609063/solution|place holder: 1d object detection solution]]
- [T0chka/mabe_12th_place_solution](https://github.com/T0chka/mabe_12th_place_solution) _(solution)_ — from [[solutions/rank_12/solution|13th Public/12th Private Solution]]
