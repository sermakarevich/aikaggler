# nfl-big-data-bowl-2026-prediction: cross-solution summary

This competition focused on predicting NFL player trajectories using raw tracking and play-by-play data, with top solutions emphasizing spatio-temporal sequence modeling and robust uncertainty handling. Winning approaches consistently combined custom transformer or RNN-based architectures with extensive geometric/kinematic feature engineering, physics-aware augmentations, and sophisticated ensemble strategies that prioritized diversity and complementary error patterns over simple averaging.

## Competition flows
- Raw play data cleaned and directionally standardized, processed through a custom Transformer-Conv1d architecture trained with GaussianNLLLoss, and aggregated via a simple average of 100+ models.
- Raw NFL play data augmented via coordinate flips, processed into relative/decomposed features with the ball landing position as a distinct node, fed into a GRU-Transformer stack with Entmax pooling, and ensembled via TTA.
- Raw tracking data filtered and aligned, flattened into fixed player-slot tensors with geometric/kinematic features, processed through a dual-path Graph-Transformer with multi-task losses, and aggregated via TTA and model averaging.
- Raw tracking data standardized for field symmetry, processed through a custom data loader with chiral augmentation, fed into a heterogeneous ensemble of transformer/GNN/RNN models, and aggregated via polar vector averaging with TTA and noise injection.
- Pipeline predicts relative player movement deltas using RoPE-encoded temporal embeddings, trained in two stages with pseudo-labeling, and optimized via Muon with HuberLoss.
- Raw trajectory data normalized to a passer-centric coordinate system and aligned, fed into an ST-Transformer encoder with domain-biased attention, decoded by a TXP-CNN, and averaged across player permutations during inference.
- Raw play-by-play data processed into kinematic/spatial features, augmented via virtual pass-cut shifting, fed into an ST-transformer trained with time-decayed TemporalHuber, validated via a week-based split, and submitted as a weighted ensemble of 48 models.
- Raw tracking data and static features fed into a PyTorch model with an RNN encoder and custom temporal Huber loss to predict positional deltas, cumsummed for final coordinates, and ensembled across multiple models.
- Raw trajectory data processed through a Spatio-Temporal GRU with Gated Cross-Attention and a Coarse-to-Fine decoder, trained with a custom Cauchy loss and velocity regularization, then optimized via a Genetic Algorithm ensemble.

## Data processing
- Outlier removal
- 180-degree rotation
- Uniform rotation augmentation
- Vertical flip augmentation
- Shifting prediction start frames earlier
- Coordinate flips (horizontal, vertical, both)
- TTA via coordinate flips
- Flattening plays along time axis
- Zero-padding missing players
- Freezing player slot ordering
- Filtering external tracking data by event chain
- Deriving ball landing coordinates and player roles
- Horizontal flip augmentation
- Random player dropout
- Random player reordering
- Random input cropping
- Disabling augmentations during validation
- Standardizing play direction (left-to-right)
- Dynamic Random Y-Flip augmentation
- Filtering context to top-k entities
- Gaussian noise injection during inference and linear extrapolation for missing frames
- Passer-centric coordinate normalization
- Field alignment ('Unify Left')
- Transition sequences training
- Non-target prediction training
- TTA via player permutation averaging
- Virtual pre/post pass-cut augmentation
- Merging augmented data with weighted loss
- Feature set pruning
- Embedding player role and position
- Batch Normalization and Dropout application
- Minimal feature engineering on raw inputs

## Features engineering
- Dynamic kinematic features (x, y, s, a, dir, o, sin/cos transformations)
- Static features (one-hot player_role, frame counts, final coordinates)
- Target displacement (Δx, Δy)
- Ball landing position as distinct node
- Velocity decomposition (s*sin(dir), s*cos(dir))
- Deltas in speed/acceleration
- Relative differences between players
- Geometric/kinematic features (angle_to_ball, dir_rad)
- Distance metrics to passer, receiver, and ball landing
- Spatial relationships (projection_on_passing_line, triangle_area_ratio)
- Time elapsed
- Physical stats (height, weight)
- Ball metrics (ball_dx/dy, ball_closing_speed)
- Spatial relationships (dist_from_right, distance_to_goal, alignment)
- Temporal markers (post_pass, frame_to_pass)
- 8-dim embeddings for player_role and player_position

## Models
- Depthwise Conv1d encoders
- Conv1d decoders
- Transformer-based architectures (including Graph-Transformers and STTRE)
- GRU/RNN variants (including Bi-GRU and Spatio-Temporal GRU)
- GNNs (including ESG)
- CNN-based decoders (TXP-CNN)
- Social-LSTM
- BERT-like models
- CVAE
- Cluster-Based Mixture of Experts
- Temporal U-Net
- Linear bridge/head layers
- Spatial-temporal encoders
- Gated Cross-Attention mechanisms
- Coarse-to-Fine decoders

## Loss functions
- GaussianNLLLoss (including auxiliary/velocity variants)
- TemporalHuber (including adapted/time-decay variants)
- HuberLoss
- Position loss (MSE)
- Auxiliary Velocity Loss
- Inter-frame displacement loss
- End-point prediction loss
- Full spatio-temporal correspondence loss
- Laplacian NLL
- RMSE
- Cauchy loss
- Weak Velocity Regularization (Random Kernels)
- softplus(var) + 1e-3 constraint

## CV strategies
- Group 5-fold CV using game_id repeated 3 times
- 7 GroupKFold on game_id with different seeds
- Week-based split across 18 weeks into 6 folds with multiple seeds
- Out-of-fold validation using the last 5 weeks of data (avoiding k-fold for faster iteration)
- Collecting OOF predictions from multiple base models trained with varied seeds and hyperparameters

## Ensembling
- Simple average of 100+ models differing in feature configurations and CV splits
- Ensemble of TemporalHuber/GaussianNLLLoss models combined with TTA via coordinate flips
- Simple averaging of four 7-fold CV models combined with TTA (weighted average of original, flip, and crop predictions)
- Physics-aware polar vector averaging of heterogeneous models (STTRE, GNN, Bi-GRU) with dynamic time-decay weighting
- Combination of multiple diverse models (GNN, BERT-like, CVAE) without detailed weighting
- Weighted mean of 48 models across different output horizons
- Sequential addition of models to the ensemble
- Genetic Algorithm optimization of ensemble weights for complementary error patterns followed by weighted averaging

## Notable individual insights
- rank 1 (1st Place Solution): GaussianNLLLoss improves accuracy by automatically downweighting samples with large variance.
- rank 4 (Private 4th / Public 5th Place Solution): Treating the ball landing position as a separate node allows the Transformer's attention mechanism to dynamically weight its relevance per defender.
- rank 3 (3rd Place Solution): Randomly shuffling players and masking them as augmentation significantly boosted performance by forcing reliance on geometric features rather than slot ID bias.
- rank 33 ([33rd solution] Transformer+Augmentation Tricks): Brute-force coordinate averaging in ensembles causes kinetic energy loss, producing physically impossible trajectories that fail to match any individual model's speed.
- rank 19 (19th Place Solution): The TXP-CNN decoder, which convolves over features while treating time as channels, performed significantly better than a standard Transformer decoder despite being counterintuitive.
- rank 24 (Private 24th | Public 23rd solution - Data augmentation): CV and LB scores frequently misalign (CV drops while LB rises), indicating that validation metrics can be misleading for this specific data distribution and time-split structure.

## Solutions indexed
- #1 [[solutions/rank_01/solution|1st Place Solution]]
- #2 [[solutions/rank_02/solution|2nd place solution]]
- #3 [[solutions/rank_03/solution|3rd Place Solution]]
- #4 [[solutions/rank_04/solution|Private 4th / Public 5th Place Solution]]
- #5 [[solutions/rank_05/solution|5th Place Solution]]
- #19 [[solutions/rank_19/solution|19th Place Solution]]
- #24 [[solutions/rank_24/solution|Private 24th | Public 23rd solution - Data augmentation]]
- #33 [[solutions/rank_33/solution|[33rd solution] Transformer+Augmentation Tricks ]]
- ? [[solutions/rank_xx_651652/solution|GRU Solution]]

## Papers cited
- [STTRE: A Spatio-Temporal Transformer with Relative Embeddings for multivariate time series forecasting](https://doi.org/10.1016/j.neunet.2023.09.039)
- [A Spatio-temporal Transformer for 3D Human Motion Prediction](https://arxiv.org/abs/2004.08692)
