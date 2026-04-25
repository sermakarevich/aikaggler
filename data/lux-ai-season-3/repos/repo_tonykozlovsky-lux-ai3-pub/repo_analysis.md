# tonykozlovsky/lux-ai3-pub

- **URL:** https://github.com/tonykozlovsky/lux-ai3-pub
- **Source solution:** 1st place approach by Flat Neurons
- **Role:** solution
- **Rank:** #1

---

This repository implements a distributed reinforcement learning pipeline for the Lux AI Season 3 competition, combining hybrid CNN, ResNet, ConvLSTM, and Transformer architectures to process spatial-temporal game states. Training leverages the IMPALA and torchbeast frameworks with V-trace off-policy correction, dynamic reward scaling, and adaptive entropy regularization to stabilize learning in sparse, non-stationary environments. The system integrates a teacher-student distillation strategy alongside mixed self-play and frozen baseline policies, while carefully masking log-probabilities and entropies to prevent gradient updates for inactive units. Hand-crafted features are derived from observation deltas and Kaggle replay analytics to infer hidden game mechanics, and auxiliary prediction tasks enhance spatial reasoning. The pipeline concludes with strategic mixed-model submissions and logit noise injection to accurately evaluate performance while mitigating imitation learning risks on the public leaderboard.

## Competition flow
Raw environment observations and Kaggle replay data are parsed, normalized, and enriched with hand-crafted temporal/spatial features, processed by a hybrid CNN-ResNet-LSTM-Transformer agent, trained via distributed IMPALA/torchbeast with V-trace corrections and mixed policy mixing, and exported as masked action logits for strategic mixed-model competition submissions.

## Models
- IMPALA
- ResNet
- ConvLSTM
- Transformer
- MLP
- Enemy Future Prediction Head
- Value Function Head
- Movement Head
- Sap-Target Head
- SpatialTransformer
- ConvLSTMCell
- Sliser
- Predictor
- DictSeparateActorResnBinary
- DictSeparateActorResn
- ResidualBlockNoMask
- CrossAttentionTransformer
- TransformerWithPosEmbed
- DictInputLayer
- Custom Self-Attention Transformer
- Custom Cross-Attention Transformer
- 2D Sinusoidal Positional Embedding
- SELayerNoMask
- ParallelDilationResidualBlock
- monobeast
- ActionPredictionModel (custom CNN with initial conv block, residual blocks with Squeeze-and-Excitation, coordinate-based feature gathering, and linear decoder)
- conv_model
- LSTM
- Transformer_v2
- per_unit_resnet

## Frameworks used
- pytorch
- gym
- numpy
- einops
- lux-ai-s3
- torch.distributed
- torch.nn.parallel.DistributedDataParallel
- torch.optim.lr_scheduler.LambdaLR
- torch.compile
- jax
- hydra
- omegaconf
- wandb
- scikit-learn
- pandas
- requests

## Loss functions
- Weighted binary cross-entropy
- KL divergence
- Teacher baseline loss
- TD-lambda
- Entropy regularization
- policy gradient loss
- smooth L1 loss
- MSE loss
- custom_policy_gradient_loss
- custom_baseline_loss
- custom_teacher_kl_loss
- custom_entropy_loss
- BCEWithLogitsLoss
- UPGO losses

## Data reading
- Observations are loaded as dictionary-based state dicts from LuxAIS3GymEnv (numpy_output=True), containing map features, units, sensor masks, and steps, which are then normalized, discretized, and converted to standardized tensors.
- Queries Kaggle API endpoints for episode lists, team info, and replay JSONs via POST/GET requests.
- Loads and persists local CSVs using pandas.
- Receives pre-loaded GPU tensors via batch dict during training.

## Data processing
- Continuous features normalized and discretized via binning into one-hot vectors
- x/y coordinates flipped during inference for data augmentation
- Coordinates normalized to [-1, 1] for grid sampling
- Nearest-neighbor patches extracted via F.grid_sample
- One-hot and continuous unit features concatenated and merged with spatial patches via linear merger
- Parallel environment outputs stacked into batched numpy arrays
- Numpy arrays converted to PyTorch tensors with dtype validation and GPU pinning
- Rewards aggregated and scaled
- Standard Gym tuple outputs converted to dictionaries
- Convolution-like kernels applied for vision mapping
- Frequency-voted estimates computed for energy reduction and SAP dropoff
- Unit positions tracked across timesteps
- Map shifts detected by comparing tile types between steps
- Adjusts episode discounting by resetting to 1.0 when done masks transition from 1 to 0, preventing discount leakage across episode boundaries.
- Masks entropy and KL calculations to exclude steps with no actions taken.
- Synthetic generation creates two 24x24 binary grids for player and enemy units, computes relative coordinate masks (15x15 per unit) indicating enemy presence within an 8-cell radius, and formats them as PyTorch tensors.
- Parses replay JSONs to extract game parameters.
- Computes duration, overtime, and step counts from agent logs.
- Handles request retries, deduplication, and missing data.
- Flags strong opponents based on stderr logs.

## Features engineering
- Temporal gaps tracking steps since last observation or enemy sighting
- Aggregated internal state parameters like estimated vision reduction
- Per-unit attributes including energy and tile ID combined with spatial patches
- One-hot encoding of discrete features (103 classes)
- Continuous feature vectors (5 dims) concatenated with one-hot encodings
- Derived metrics including max/sum energy per tile, estimated move/sap capacity, energy deltas, relic probabilities, spawn distances, visibility/unseen ticks, and scaled game parameters
- Hand-crafted features inferred via discrete candidate voting including nebula_tile_energy_reduction, unit_sap_dropoff_factor, min/max_nebula_vision_reduction, unit_energy_void_factor, my/enemy_unexpected_energy_change, shift_param, and steps_to_shift
- Extracts game configuration parameters (energy node drift, nebula tile stats, unit sap/move costs, sensor range).
- Computes runtime metrics (total duration, overtime, steps).
- Derives an is_strong binary feature from agent log stderr.

## Ensembling
A two-model submission strategy is used where a weaker model plays 85% of matches and a stronger model plays 15%, with additional logit noise injection for larger models to mask capabilities and evaluate true win rates.

## Training setup
- Training utilizes the IMPALA algorithm with dynamic reward scaling, adaptive entropy scheduling, and a teacher-student framework to stabilize learning in sparse, non-stationary environments.
- Adam optimizer with custom epsilon and linear learning rate decay schedule.
- Adaptive entropy regularization and linearly decaying learning rate over a 300M-step horizon.
- Mixed precision training with gradient clipping.
- Distributed data parallelism (DDP) with periodic weight synchronization between learner and actor processes.
- Aggressive PyTorch Inductor and JAX compilation caching to minimize training overhead.

## Insights
- Large-scale self-play combined with ResNet, ConvLSTM, and Transformer architectures effectively learns complex multi-agent policies.
- Adaptive reward scaling and dynamic entropy scheduling stabilize training in sparse, non-stationary environments.
- Teacher-student frameworks and frozen opponent pools prevent catastrophic forgetting and overfitting to self-play dynamics.
- Strategic mixed-model submissions allow accurate leaderboard evaluation while protecting against imitation learning.
- Combining local patch extraction with global spatial features allows the model to capture both fine-grained unit details and broader map context.
- Using fixed 2D sinusoidal positional embeddings instead of learnable ones ensures spatial consistency and generalization across different grid resolutions.
- Applying available action masks directly to logits before sampling enables efficient handling of multi-discrete action spaces with dynamic constraints.
- Integrating a ConvLSTM with explicit termination mask handling preserves temporal state across episodes while correctly resetting hidden/cell states upon episode end.
- The architecture uses pre-normalization (LayerNorm before attention/FFN) to stabilize training dynamics.
- Parallel dilated convolutions efficiently capture multi-scale receptive fields without increasing model parameters.
- Squeeze-and-Excitation blocks adaptively recalibrate channel-wise feature responses to emphasize informative channels.
- Comparing current and previous observations allows accurate inference of hidden game mechanics like SAP dropoff and energy void factors without explicit environment exposure.
- Frequency voting over discrete candidate values provides a robust way to estimate continuous or hidden parameters in partially observable environments.
- Tracking units across timesteps via position/energy proximity handles ID switching and unit spawning/death gracefully.
- Reward shaping is implemented as a weighted sum of discrete, interpretable components rather than a single dense signal.
- Multipliers are used to isolate specific behaviors during different training phases without rewriting core logic.
- Multi-unit RL requires explicit masking of log-probabilities and entropies to prevent gradient updates for units that did not act.
- V-trace stabilizes off-policy learning by clipping importance weights using configurable thresholds for value targets and policy gradients separately.
- Coordinate-based feature extraction allows the network to focus on specific unit locations without relying on fixed grid pooling.
- The training pipeline relies on a curriculum-style policy mixing strategy to stabilize learning across self-play, teacher, and frozen baseline actors.
- Replay JSONs contain static game configuration parameters that can be directly extracted as features.

## Critical findings
- Imitation learning opponents can copy strong strategies, making it necessary to obscure model capabilities during public testing.
- Pure self-play eventually plateaus, requiring teacher KL alignment and diverse opponent pools to maintain progress.
- The unit matching logic includes a fallback for a rare edge case where position/energy proximity checks fail, indicating potential instability in state tracking under specific conditions.

## What did not work
- Behavior cloning from external replays was implemented but ultimately abandoned because the RL pipeline already achieved significant improvements without it.
- The explore_reward component is explicitly marked as broken in the code comments.
- The my_stuck_reward is noted as difficult to implement correctly without pathfinding.
- Several other granular rewards (unit stacking, energy waste, sensor mask, near units) are disabled via zero multipliers, indicating they were abandoned during tuning.
- behavior_cloning_actors is explicitly disabled with a comment noting it is NOT WORKING YET.

## Notable files
- `README.md` — documentation
- `final_versions/07_03_tune_against_mask/run_monobeast.py` — entrypoint
- `final_versions/07_03_tune_against_mask/conf/base.yaml` — config
- `final_versions/07_03_tune_against_mask/lux_ai/nns/models.py` — model definition
- `final_versions/07_03_tune_against_mask/lux_ai/torchbeast/core/vtrace.py` — RL algorithm
- `final_versions/07_03_tune_against_mask/lux_ai/torchbeast/core/losses_func_selfplay.py` — loss computation
- `final_versions/07_03_tune_against_mask/analytics/src.py` — data pipeline

## Files analyzed
- `README.md` _(documentation)_ — This README documents the winning solution for the Lux AI Season 3 competition, detailing a multi-agent reinforcement learning pipeline built around the IMPALA algorithm. It explains the custom network architecture combining ResNet, ConvLSTM, and Transformer layers, alongside specialized heads for enemy prediction, value estimation, and dual-action selection. The document highlights key training innovations like dynamic reward scaling, adaptive entropy scheduling, and a teacher-student framework to stabilize learning. It also outlines a strategic public testing approach using mixed-model submissions to evaluate performance while mitigating imitation learning risks. Readers should take away how advanced exploration, reward shaping, and careful leaderboard testing converge to solve complex, partially observable multi-agent environments.
- `final_versions/07_03_tune_against_mask/agent.py` _(inference)_ — This file implements the inference agent for a Kaggle competition, handling model loading, observation preprocessing, and action generation. It manages a PyTorch-based actor-critic model and maintains LSTM hidden states across game steps to ensure temporal consistency. The core technique is a rotation-based data augmentation strategy that averages policy logits across multiple spatial transformations to improve decision robustness. It also includes dynamic logic for switching between weak and strong models based on game rounds and remaining time, along with a fallback to single-player inference when overtime is low. Readers should note how spatial symmetries are leveraged for inference-time augmentation and how hidden states are carefully sliced to maintain consistency across augmented views.
- `final_versions/07_03_tune_against_mask/lux_ai/nns/models.py` _(model_definition)_ — This file defines a suite of PyTorch neural network modules tailored for a multi-agent reinforcement learning environment. It implements a hybrid architecture that combines a custom 2D sinusoidal positional embedding, a spatial transformer block for grid-based attention, and a ConvLSTM for temporal sequence modeling. The core actor network processes per-unit masks and spatial features, extracts local patches via a coordinate-aware `Sliser` module, and outputs discrete action logits for worker and sapper units. The design emphasizes handling multi-discrete action spaces with available action masking and integrating local/global features through linear merging and residual blocks.
- `final_versions/07_03_tune_against_mask/lux_ai/nns/transformer.py` _(model_definition)_ — This file defines a custom PyTorch implementation of a Transformer architecture designed for grid-based spatial data. It provides modular building blocks for self-attention, cross-attention, pre-normalization, and feed-forward networks, along with a custom 2D sinusoidal positional embedding for a 24x24 grid. The architecture reshapes 2D inputs into sequences for transformer processing and restores the spatial dimensions afterward. It serves as the core neural network backbone for the competition solution, offering flexible attention mechanisms without relying on high-level framework abstractions.
- `final_versions/07_03_tune_against_mask/lux_ai/nns/conv_blocks.py` _(model_definition)_ — This file defines custom PyTorch neural network building blocks for a convolutional architecture. It implements a Squeeze-and-Excitation attention mechanism, a standard residual block, and a parallel dilated residual block that processes inputs through two concurrent branches with different dilation rates. The blocks dynamically compute padding to preserve spatial dimensions across varying kernel sizes and strides. These modules are designed to be composed into larger CNNs for extracting spatial features from grid-based game states.
- `final_versions/07_03_tune_against_mask/lux_ai/lux_gym/obs_spaces.py` _(preprocessing)_ — This file defines the observation space structure and transformation logic for a LUX AI Season 3 Gym environment. It establishes a comprehensive dictionary of continuous, binary, and discrete channels covering unit states, map features, relic locations, game parameters, and temporal metrics. The `_Obs3` wrapper acts as a Gym wrapper that converts raw environment dictionaries into standardized, normalized, and discretized tensors ready for neural network input. It serves as the critical bridge between the raw simulation state and the agent's perception, ensuring consistent formatting and scaling across all game elements.
- `final_versions/07_03_tune_against_mask/lux_ai/lux_gym/wrappers.py` _(utility)_ — This file provides a suite of Gym environment wrappers tailored for a reinforcement learning pipeline in the LUX AI competition. It handles parallel environment stepping, converts environment outputs to PyTorch tensors, standardizes data formats, and manages reward computation and logging. The code emphasizes profiling, device placement, and handling Kaggle-specific observation formats. Readers should understand how raw environment interactions are transformed into batched, tensor-ready, and reward-scaled inputs for downstream training.
- `final_versions/07_03_tune_against_mask/lux_ai/lux_gym/sb3.py` _(utility)_ — This file implements a Gym environment wrapper (`SB3Wrapper`) for the Lux AI Season 3 competition, designed to feed rich, state-aware observations into a reinforcement learning agent. It wraps the official `LuxAIS3GymEnv` and injects hand-crafted features that infer hidden game mechanics, such as energy reduction on nebula tiles, SAP dropoff factors, vision reduction, and map shifts. By comparing current and previous observations, it uses frequency voting over discrete candidate values to estimate these parameters. The wrapper also handles unit tracking, unexpected energy change mapping, and shift detection, effectively bridging the gap between raw environment outputs and RL-ready state representations.
- `final_versions/07_03_tune_against_mask/lux_ai/lux_gym/reward_spaces.py` _(utility)_ — This file implements a modular reward calculation system for the LUX-AI Season 3 environment. It decomposes the training signal into numerous granular components, including point differentials, relic discovery, energy efficiency, enemy kills, and movement penalties. The architecture uses abstract base classes and a central computation function to allow easy switching between reward configurations via multiplier tuning. It serves as the core reward engineering module for the agent's training pipeline.
- `final_versions/07_03_tune_against_mask/lux_ai/torchbeast/core/losses.py` _(utility)_ — This file implements a collection of loss computation functions for a multi-agent reinforcement learning agent in the LUX AI competition. It handles policy gradient updates, entropy regularization, teacher-student KL divergence, and value/baseline loss calculation. The functions carefully manage multi-unit action masks to ensure gradients are only computed for units that actually executed actions. It provides both standard and alternative implementations to support different training configurations and sampling strategies.
- `final_versions/07_03_tune_against_mask/lux_ai/torchbeast/core/losses_func_selfplay.py` _(training)_ — This file computes the composite training loss for a self-play reinforcement learning agent in the LUX-AI competition. It aggregates policy gradient losses using V-trace and UP-GO advantage estimators, value baseline losses via TD-Lambda, entropy regularization, and teacher KL divergence. It also incorporates auxiliary prediction tasks (position, proximity, sensor mask) using weighted BCE losses. The function handles episode discounting, reset masking, and warmup phases before returning individual and combined loss components for backpropagation.
- `final_versions/07_03_tune_against_mask/lux_ai/torchbeast/core/vtrace.py` _(utility)_ — This file implements the V-trace algorithm, an off-policy correction method for actor-critic reinforcement learning, adapted for PyTorch. It computes importance-weighted value targets and policy gradient advantages by processing behavior and target policy log-probabilities, rewards, discounts, and value estimates over a trajectory. The implementation includes both standard tensor operations and a GPT-specific variant that explicitly handles 3D shapes for sequence, batch, and parallel dimensions. It serves as a mathematical utility to stabilize off-policy updates in distributed RL setups like IMPALA. Readers should understand how clipped importance weights prevent variance explosion during off-policy training.
- `final_versions/07_03_tune_against_mask/lux_ai/torchbeast/core/batch_and_learn.py` _(training)_ — This file implements the core training loop for a learner process in a distributed reinforcement learning setup. It initializes the distributed environment, loads a learner model (and optionally a frozen teacher model), and configures an optimizer with a custom learning rate scheduler. The main loop dynamically selects batch types based on probability borders, retrieves pre-processed batches from shared queues, and passes them to a `learn` function along with multiple loss handlers. It also periodically syncs updated weights back to the actor model to maintain policy consistency across processes.
- `final_versions/07_03_tune_against_mask/conf/base.yaml` _(config)_ — This file acts as the central configuration hub for a reinforcement learning experiment targeting the Lux AI Challenge Season 3. It consolidates environment specifications, reward structures, training hyperparameters, and optimizer settings using Hydra for hierarchical config management. The configuration explicitly sets up Weights & Biases for experiment tracking, defines a linear learning rate decay schedule, and specifies loss weighting parameters for TD-lambda and UPGO algorithms. It also includes flags for benchmarking against baseline models and competitor agents, alongside environment-specific toggles like action masking and early stopping. Readers should note how RL training pipelines can be systematically parameterized for reproducibility and rapid iteration.
- `final_versions/07_03_tune_against_mask/run_monobeast.py` _(entrypoint)_ — This file serves as the primary entry point for training a reinforcement learning agent in the LUX AI Season 3 competition using the `torchbeast` library. It orchestrates configuration merging via Hydra, applies environment-specific hyperparameter overrides, and compiles a custom C++ extension to interface with the game simulation. The script sets up distributed training parameters, enables aggressive compilation caching for PyTorch and JAX, and launches either a benchmarking or full training routine. Readers should note the heavy reliance on config-driven workflows and low-level environment compilation for performance.
- `final_versions/07_03_tune_against_mask/experiments/benchmark.py` _(training)_ — This file implements a synthetic benchmark training loop for a custom coordinate-aware convolutional network designed to predict relative enemy unit positions on a 24x24 grid. It generates its own training data by randomly placing 16 units on both sides and computing ground-truth relative masks. The model extracts features at specific coordinates, processes them through residual blocks with squeeze-and-excitation, and decodes them into 15x15 prediction masks. It demonstrates a complete PyTorch training pipeline with mixed precision, gradient clipping, and metric tracking, serving as a rapid prototyping environment for architectural tuning before applying it to actual competition data.
- `final_versions/07_03_tune_against_mask/submission_model/config.yaml` _(config)_ — This YAML file serves as the central configuration for a large-scale reinforcement learning training run in the LUX AI competition. It defines hyperparameters for a hybrid neural architecture combining convolutional, ResNet, LSTM, and Transformer components, alongside RL-specific settings like a 300M-step horizon, Adam optimizer, and adaptive entropy regularization. The configuration also manages a mixed-policy evaluation loop that weights self-play, teacher, and frozen baseline actors during training. Additionally, it specifies experiment tracking via Weights & Biases and handles checkpoint paths for model loading and saving.
- `final_versions/07_03_tune_against_mask/analytics/src.py` _(data_loading)_ — This file acts as a data collection and feature extraction pipeline for the LUX-AI Season 3 competition. It queries Kaggle's API to retrieve episode metadata, game replays, and agent logs, then systematically parses the JSON responses to extract game configuration parameters and runtime performance metrics. The processed data is deduplicated, cleaned, and persisted as local CSVs for downstream analysis. Readers should note how replay JSONs and agent logs are leveraged to derive tabular features like unit costs, nebula effects, and opponent strength indicators.
