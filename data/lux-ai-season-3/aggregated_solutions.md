# lux-ai-season-3: cross-solution summary

The Lux AI Season 3 competition centered on developing autonomous agents for a partially observable, multi-agent strategy game involving resource harvesting, relic discovery, and tactical combat. Winning approaches predominantly leveraged deep reinforcement learning and imitation learning pipelines, utilizing advanced neural architectures like UNets, Transformers, and recurrent models (xLSTM, ConvLSTM) to process complex spatial and temporal game states. Success hinged on sophisticated data augmentation, dynamic reward shaping, rigorous simulation engineering, and strategic ensembling or dual-model deployment to navigate fog-of-war constraints and opponent imitation.

## Competition flows
- Raw 24×24 map observations (~1000+ tile features) are encoded and compressed into a 24×24×128 feature map, processed through residual blocks, a ConvLSTM, and a Transformer, then passed to multi-head outputs (enemy prediction, value, movement, sap-target) trained via large-scale IMPALA self-play with dynamic reward/entropy scaling, and finally deployed via a dual-model Kaggle submission strategy to mask the strongest policy during public testing.
- The pipeline extracts observable states and infers hidden constants from top-team replays, standardizes map orientations via mirroring, filters out uninformative timesteps and excessive "Center" actions, trains two separate UNet models with weighted cross-entropy loss, and combines the outputs for final submission.
- Raw game observations are parsed by a custom Rust rules engine, processed into spatial/global features with a 10-step history and inferred hidden states, fed into a residual CNN for PPO training, and deployed at test-time with stochastic sampling and geometric TTA.
- Raw game observations are parsed into 24x24 map and scalar global features, which are fed into two dynamically switched UNet-based Imitation Learning models (Action and Sap Target) trained on filtered replay data, with inference applying TTA, probability thresholding, and overlap-aware action selection to generate unit commands.
- The pipeline uses a rule-based agent for the first match to reveal hidden relic locations, then switches to a multi-agent PPO policy with shared actors and a global critic that takes 47x47 egocentric observations, applies strict action masking, and optimizes for match win/loss rewards over 15 days of GPU training.
- Raw match data from top submissions is used to train a two-stage fine-tuned UNet policy network, which estimates hidden states and outputs action probabilities that are resolved into unit assignments via minimum cost flow, followed by TTA and tournament-based evaluation.
- Raw game state feature maps are processed through a convolutional ResNet to extract spatial features, concatenated with unit energy, and fed into a VTrace-trained RL agent that outputs masked actions for movement and SAP targeting, with hidden game states inferred via Bayesian/linear methods and energy field matching.
- Raw game state observations are processed into unit, spatial, and scalar feature tensors, fed into a JIT-compiled PPO policy with a centralized critic, trained via PFSP and BPTT with a dense-to-sparse reward schedule, and evaluated through self-play and frozen opponent sampling to generate final submissions.
- The pipeline consists of a rules-based game bot that observes the simulation state, applies five heuristic modifications (relic prioritization, energy harvesting/charging, late-game avalanche attacks, board change prediction, and blind SAP deployment), and outputs actions for submission.
- Raw game observations (global features, map tiles, unit states) are encoded via ConvNeXt and MLPs, passed through a Transformer and xLSTM recurrent core, and decoded by an actor network to predict action distributions, which are executed in a self-play PPO loop implemented in JAX.
- The pipeline curates winning episodes from other agents, constructs 50 map and global features while maintaining a fog-of-war-aware global state, trains two UNet models (action and sap) with agent-ID conditioning and symmetry normalization, and submits an ensemble of the separate-head and DiT-conditioned architectures.
- The pipeline involves initializing the LUX AI S3 simulation environment, applying a modified rule-based heuristic agent that handles exploration, resource harvesting, and relative-coordinate SAP attacks, and executing it against opponent bots across multiple games to generate a final submission.
- Raw simulation states are processed through a rule-based heuristic engine that computes movement, harvesting, and defensive actions, which are then submitted directly to the game API for scoring.

## Data processing
- Features encoded in normalized continuous form and discretized via binning into one-hot vectors.
- Internal state updater aggregated discovered environment parameters over time to calculate dynamic features.
- Training augmentation: flipping observations and actions to force the agent to always start at (0,0).
- Inference augmentation: flipping x/y coordinates during Kaggle inference to smooth map orientation bias (disabled if overtime > 30s to reduce latency).
- Mirrored maps to standardize spawn positions at (0,0), dropped 95% of "Center" action instances to reduce passivity and speed up learning, filtered out timesteps where game outcomes were already decided, and simulated fog-of-war visibility and hidden game constants by reconstructing observable states from replay agent observations.
- 47x47 egocentric observation space centered on each agent.
- Strict action masking for invalid moves (e.g., sapping without energy).
- Reward structure revised to weigh match win/loss outcomes over raw point differences.
- Map features leverage symmetry to infer unobserved coordinates.
- Hidden parameters like tile_type drift and unit_sap_dropoff_factor are estimated via exploration logic.
- Training data filtered to winning episodes against high-LEA opponents.
- Augmentations include 0/90/180/270 degree rotations and vertical flips.
- Inference applies TTA with the same augmentations and averages outputs.
- Normalized features to [-1, 1] with standardized energy scales.
- Applied mirroring augmentation along x/y axes during training.
- Used n_stack=4 for temporal context.
- Applied TTA mirroring at inference.
- Mirrored data for the left-top corner to augment training.
- Computed energy fields across all possible positions and matched them with known energy nodes.
- Inferred hidden relic cell positions using Bayesian posterior updates and linear equation solving over past observations.
- Tracked nebula drift speed and energy reduction to avoid misleading the agent with obsolete map information.
- Added temporal features including cell types for 3 future steps and team/enemy energy and location layers from the last 2 steps.
- Spatial features persist across steps with dynamic nebula/asteroid shifting; actions are masked by energy, map bounds, and visibility; team points, wins, and match steps are binary encoded.
- Flipping all features (positions, map) to keep the player in the top-left corner.
- Adding x/y coordinate positional encodings to map features.
- Masking invalid actions (out-of-map or out-of-reach sapping).
- Iterative updating of relic fragment location probabilities based on unit history.
- Mirroring game state along the main diagonal to standardize POV to player 0.
- Normalizing energy, costs, steps, and team points to [0,1].
- Masking cells without units for the action model and cells outside sap range for the sap model during training/inference.
- Filtering dataset to only winning matches and removing prematurely decided games.

## Features engineering
- ~1000+ features per tile: ~100 continuous features and ~900 one-hot encoded discrete features (asteroid presence, team unit presence, steps since last enemy sighting, relic node region membership).
- Temporal features tracking steps since last observation or enemy sighting.
- State tracking features for discovered environment parameters (e.g., vision reduction values).
- 3 additional 'enemy future prediction' features from the previous timestep.
- Per-unit features (energy, unit ID in tile) appended to per-unit patches.
- Feature maps (28x24x24) encoding unit positions/energy, fleet vision, nebulae, asteroids, node energy, relics, reward points, and vision duration; global features (17) including move/SAP costs, team points, match step/number, and inferred hidden constants (nebula drift speed, energy reduction, vision reduction, SAP dropoff factor, energy void factor).
- Global features (score, rules, step).
- Spatial features (ships, relics, energy field).
- Temporal features (10-step history of ships, scores).
- Inferred features (reflected map symmetry, deduced point tile locations, hidden rule/asteroid/nebula parameter deduction, precomputed energy field caching).
- Action masking (disallow off-map, asteroid collisions, insufficient energy, blind sap restrictions).
- 15 map features (self/opp unit positions, energy, enable flags, tile_type, visible_mask, map_energy, relic_nodes, point_prob_map, pre_self/opp_unit_pos).
- 12 global features (self/opp reward, match steps/round, team points/wins, unit costs/range/dropoff_factor).
- Symmetry-based inference for unobserved tiles.
- Exploration-based estimation for point_prob_map and unit_sap_dropoff_factor.
- 14-channel map features (tile types, energy, nebula reduction, sensor mask, relics, points, entropy, unit counts/energy, visit count, sap area).
- 16-channel scalar features (match steps, points, wins, costs, ranges, nebula/energy drift stats).
- Estimated hidden states (next tile type, energy map, point probabilities, opponent unit counts, vision/energy reduction metrics) using Bayesian inference and exhaustive search.
- Game parameters: step, match step, unit_move_cost, unit_sensor_range, nebula_vision_reduction, nebula_energy_reduction, team points, and team wins.
- Map features: cell type, cell energy, nebula, visible status, relic neighbor status, and team point layer.
- Temporal layers: cell type in 3 future steps, team energy/location, and enemy energy/location from the last 2 steps.
- Inferred parameters: energy void field factor and sap dropoff factor estimator.
- Unit features (ally/enemy flag, visibility, last 5 turns energy, turns since last seen, current/last known position).
- Spatial grid features (energy field, nebula & asteroid fields, relic nodes & points, sensor mask, visited tiles).
- Scalar features (binary encoded team points, wins, match steps, and deducible hidden game parameters).
- Relic fragment location probabilities updated iteratively from unit movement history.
- Positional encodings (x, y coordinates) added to map features.
- 18 map-level features: unit positions/energy, sap range, visibility, tile types, drift predictions, distance from center, relic/reward nodes, reward map.
- 32 global features: normalized steps, costs, sensor ranges, relic/reward discovery flags, reward history over X turns, team points, drift speed types, agent ID one-hot.

## Models
- Residual blocks
- ConvLSTM
- Transformer
- MLP
- Multi-head policy/value network (IMPALA-based)
- UNet (including World-wise Unit-UNet, Unit-wise SAP-UNet, Action/Sap Target variants)
- Residual CNN (with squeeze-excitation/ResBlocks)
- Vision Transformer
- MAPPO
- RNN
- PPO (with BPTT)
- PFSP
- Centralized Critic
- Conv2d
- Pointer Network
- xLSTM
- ConvNeXt
- Diffusion Transformer
- U-Net

## Frameworks used
- pytorch
- JAX
- PureJaxRL
- jax

## Loss functions
- weighted binary cross-entropy
- KL divergence
- teacher baseline loss
- entropy regularization
- TD losses
- weighted cross-entropy loss
- PPO with clipping
- Illegal action masking loss
- Entropy loss
- Teacher-KL loss
- GAE-Lambda for value estimation
- Sparse win/loss reward (+1/-1)
- Softmax value normalization
- BCEWithLogitsLoss
- cross-entropy loss
- DiceLoss
- Custom Focal Tversky Loss
- VTrace
- UPGO pg loss
- baseline loss
- PPO loss
- cross-entropy loss (knowledge distillation)
- Weighted Dice Loss

## CV strategies
- Large-scale self-play against latest self, frozen opponent pool, and a frozen teacher model; real-time win rate tracking across thousands of matches.
- Custom tournament evaluation using a modified luxai_runner to track win rates and p-values over 100 local episodes per model change.

## Ensembling
- Used a dual-model submission strategy where a weaker model was deployed 85% of the time and a stronger model 15% of the time to evaluate performance while masking the best policy from imitation learning opponents.
- Two separate UNets are used complementarily rather than ensembled; during inference, predicted actions for colliding units are assigned to the top half by energy to encourage spreading.
- Test-time ensembling is achieved by applying three geometric augmentations (diagonal reflections and 180-degree rotation), averaging the resulting policy outputs, and then sampling actions from the averaged distribution.
- The solution combines outputs from two separate IL models (Action and Sap Target) and applies TTA with 4 rotations and vertical flips, averaging 8 outputs, followed by cumulative probability thresholding (0.7 for actions, 0.6 for sap targets) and a coordinate-overlap probability adjustment mechanism to prevent action collisions.
- The final submissions consist of an ensemble of two network architectures: a UNet with separate prediction heads per agent and a UNet with a single prediction head conditioned via Diffusion Transformer blocks.

## Insights
- Large-scale self-play combined with advanced architectures (ResNet, ConvLSTM, Transformers) yields strong multi-agent policies.
- Adaptive reward scaling and dynamic entropy scheduling are critical for stabilizing training in complex, sparse-reward environments.
- Teacher-student frameworks and opponent pools effectively prevent catastrophic forgetting and overfitting to self-play dynamics.
- Careful public testing strategies are essential to evaluate true performance without exposing advanced tactics to potential imitation learning opponents.
- Accurately simulating fog-of-war visibility and hidden game constants from replay observations is critical for effective imitation learning.
- Mirroring maps to standardize spawn positions introduces action distribution skew that must be addressed with loss weighting.
- Sorting units by energy during inference effectively mitigates the model's inability to handle multiple units at the same position.
- Fine-tuning on a second team's replays yields better generalization than training exclusively on one team's data.
- Test-driven development drastically reduced debugging time and improved code correctness.
- Stochastic policies outperform deterministic ones in fog-of-war strategy games due to better blind sapping and evasion.
- Inferred features like symmetry and energy caching are critical for overcoming partial observability.
- GPU compute is the primary bottleneck, making simulation speed optimization highly valuable.
- Allowing the value function to see both teams' perspectives stabilizes training without affecting test-time performance.
- Leveraging map symmetry significantly improves feature extraction by allowing inference of unobserved coordinates.
- Dynamically switching models based on estimated hidden parameters like unit_sap_dropoff_factor enables adaptive gameplay performance.
- Fine-tuning on a small subset of high-performing episodes from the final leaderboard yields better results than training on the full dataset.
- Strict action masking must be implemented from the start rather than expecting agents to learn invalid move avoidance.
- Reward structures should prioritize match win/loss outcomes over raw point differences to encourage strategic sacrifices.
- Comprehensive performance metrics and monitoring should be implemented before training begins to track behavioral evolution.
- Vision Transformers show promise but may be computationally prohibitive compared to CNNs with residual connections under tight budgets.
- Estimating hidden environmental states and parameters via Bayesian inference and exhaustive search significantly improves policy accuracy.
- Using minimum cost flow for action assignment effectively handles multi-unit coordination and prevents collisions.
- TTA via mirroring provides a reliable performance boost at inference time.
- Self-play against a zero-sum baseline model was the key catalyst for the RL agent to start learning effectively.
- The Bayesian approach to relic detection offers noise robustness at the cost of slower convergence compared to linear equations.
- Energy node locations are only known until the next drift, which is determined by a randomly selected nearby position.
- Switching to sparse rewards late in training yielded significant performance gains despite being applied to only the final 10% of steps.
- Using a centralized critic with global observability improved value estimation and doubled sample efficiency, though it reduced training throughput.
- Prioritizing higher model capacity over maximum training throughput likely yields better final results in complex RL environments.
- Fully JIT-compilable pipelines maximize speed but severely reduce code readability, making extensive unit tests essential for debugging.
- Visualizing model inputs and outputs was essential for debugging a custom JAX implementation.
- xLSTM outperformed standard LSTMs for partially observable time-series tasks while requiring fewer parameters.
- The introduction of late-spawning relics made recurrent modeling highly advantageous for the competition.
- Small map sizes enabled training on consumer GPUs with limited VRAM.
- Conditioning the network on agent IDs at the bottleneck and prediction head stabilizes training across agents with different action distributions.
- Mirroring the game state along the main diagonal leverages map symmetries and ensures consistent data usage.
- Maintaining a global game state is essential for handling fog of war and hidden reward nodes in iterative decision-making.
- Prioritizing relic search when point differentials shift is critical to avoid falling behind.
- Late-game synchronization of energy-charged ships can overwhelm opponents and decisively shift the match.
- Predicting board state changes allows for proactive route optimization rather than reactive movement.
- Blind SAP deployment based on estimated opponent rewards effectively disrupts progress without requiring direct line-of-sight.
- Directing all units toward the discoverer of a new resource consolidates efforts and improves collection efficiency.
- Moving idle units toward unexplored nodes during early games significantly increases the probability of discovering new rewards and relics.
- SAP attacks must use relative coordinates (enemy position minus current position) rather than raw coordinates to function correctly.
- Restricting SAP usage to harvesting units with remaining energy prevents units from depleting their power before reaching their destination.
- Reinforcement learning approaches and practical implementation tips are critical next steps for this competition.
- Global optimization of sapling distribution across ships is necessary to avoid suboptimal greedy strategies.
- Understanding how top teams estimated the DROP_OFF_FACTOR provides a key strategic advantage.

## Critical findings
- Pure self-play plateaued quickly, necessitating the introduction of a frozen teacher agent and an opponent pool to maintain robustness and prevent overfitting to a single playstyle.
- Dynamic entropy scheduling that decays to zero and then resets to smaller values caused short-term performance dips but ultimately produced a significantly stronger final policy.
- Using a dual-model submission with a fixed probability mix allowed accurate evaluation of a stronger model on the public leaderboard without triggering imitation learning from top competitors.
- A simple IL approach trained for only a couple of hours outperformed a heavily optimized rule-based agent that took weeks to develop.
- Dropping 95% of "Center" action instances dramatically accelerated the learning process by reducing passive behavior.
- Mirroring maps to standardize spawn positions inherently skews the action distribution, increasing Right/Down frequency while decreasing Left/Up.
- Restrictive action masking for blind sapping was counterproductive, as opponents successfully used blind saps on seemingly irrelevant tiles.
- Vision transformers learned faster during imitation but were significantly harder to stabilize for large-scale training compared to CNNs.
- The model's performance plateaued around 200M steps but continued to show gradual improvements up to 300M steps.
- Removing past step information (pre_self_unit_pos, pre_opp_unit_pos) from the Sap Target Model input unexpectedly increased the win rate during local experiments.
- Attempting to use long-term temporal models like LSTM or Conv3D severely degraded performance, whereas simple frame-stacking of unit positions improved it.
- Converting the IL policy to Reinforcement Learning caused complete policy collapse or no improvement, highlighting the difficulty of RL training with limited computational resources.
- Using a 47x47 egocentric observation space with 75% NULL values surprisingly improved leaderboard standing from 150+ to 11-15 after minimal training.
- Greedy action selection severely hurt performance; action sampling was necessary to encourage agents to explore sapping and coordination.
- Feeding ground-truth fragment locations from match 2 onward doubled the mean points collected per match compared to pure RNN approaches.
- Mixing data from different policies did not significantly improve performance, suggesting policy alignment is critical for imitation learning.
- SAP targets are highly sparse, necessitating a custom Focal Tversky Loss with masking to prevent it from inhibiting main policy learning.
- Dropping the UPGO pg loss was counterintuitive but necessary because its magnitude was an order of magnitude larger than other losses, stalling training.
- A 26-layer ResNet significantly slowed training without yielding any performance gains.
- Pure rule-based heuristics plateaued at ~1500 score, proving insufficient for top-tier rankings.
- Sparse rewards applied to only the last 10% of training steps still drove significant performance gains.
- Larger model capacities increased VRAM usage and reduced parallel environments without delivering mid-stage performance improvements.
- Switching to sparse rewards during training caused significantly slower model improvement compared to shaped rewards.
- The model continued to improve after the final submission checkpoint, indicating that longer training would have yielded higher scores.
- Removing matches where agents stopped playing once a decided state was reached prevented performance degradation caused by significantly different action distributions in those endgame scenarios.
- Different agents inherently exhibit different action distributions, which introduces instabilities during imitation learning training.
- The game state can be decisively shifted in the final 10 steps through a coordinated energy avalanche.
- Estimating opponent reward fill via point gain enables effective blind SAP deployment, a non-obvious tactical advantage.
- Imitation learning from top teams failed because managing the simulation state properly was difficult, causing units to move in only one direction.
- Using raw coordinates for SAP attack vectors breaks the mechanic; relative coordinates are required.
- U-Net energy interpolation performed well in isolation but provided no score boost when integrated into live gameplay.
- Global sap optimization across all ships simultaneously failed to improve performance over simpler heuristics.
- The referenced paper (arXiv:2301.01609) looked promising theoretically but struggled to learn in the actual simulation environment.

## What did not work
- Behavior cloning (BC) from external replays was implemented but ultimately not incorporated into the final training because significant improvements were already observed without it.
- A rule-based agent approach hit performance limits after weeks of development without any progress, prompting a switch to Imitation Learning.
- Attempting to factorize the value function on a per-unit level failed to produce successful results.
- Stabilizing training for a large vision transformer model proved too difficult within the competition timeframe.
- Pure meta-RL with RNNs failed to learn exact fragment positions and combat behavior due to hidden information.
- Using total point difference as the reward structure caused agents to fail to learn strategic sacrifices, as they were rewarded for leading even when ultimately losing.
- Basic action masking was insufficient; agents failed to learn to avoid invalid moves like sapping without energy unless explicitly masked.
- Vision Transformers were too slow to train within the competition's time and compute constraints.
- Mixing data from different policies (multiple submissions) did not significantly improve performance.
- Initial RL attempts with move-action-only agents and various reward-shaping ideas (<700 score).
- Rule-based agent plateauing at 1500+.
- 26-layer ResNet slowing training with no improvement.
- UPGO pg loss causing training stagnation due to large magnitude.
- Using larger models (>1.8M parameters) consumed excessive VRAM and reduced parallel environment counts without delivering mid-stage performance gains.
- Imitation learning from top teams, which struggled with state management and caused units to move unidirectionally.
- Global sap optimization along all ships together
- Prediction of obstacles movement in pathfinding
- Enemy ramming logic
- Energy interpolation with U-Net
- Implementation of the referenced paper (arXiv:2301.01609)

## Notable individual insights
- rank 1 (1st place approach by Flat Neurons): Dynamic entropy scheduling that decays to zero and then resets to smaller values caused short-term performance dips but ultimately produced a significantly stronger final policy.
- rank 3 (Imitation Learning: 3rd Place Solution): A simple IL approach trained for only a couple of hours outperformed a heavily optimized rule-based agent that took weeks to develop.
- rank 14 (Multi-agent RL Silver solution by 3Comets, 14th place): Using a 47x47 egocentric observation space with 75% NULL values surprisingly improved leaderboard standing from 150+ to 11-15 after minimal training.
- rank 10 (10th Place Solution – Boey – End-to-End JAX RL): Switching to sparse rewards late in training yielded significant performance gains despite being applied to only the final 10% of steps.
- rank 5 (Kiwis xLSTM agent (5th place solution)): xLSTM outperformed standard LSTMs for partially observable time-series tasks while requiring fewer parameters.
- rank 92 (Bronze medal border solution (finally in 92th place)): Imitation learning from top teams failed because managing the simulation state properly was difficult, causing units to move in only one direction.
- rank 65 (65th~ place solution just on Rules and will to experiment): U-Net energy interpolation performed well in isolation but provided no score boost when integrated into live gameplay.

## Solutions indexed
- #1 [[solutions/rank_01/solution|1st place approach by Flat Neurons]]
- #3 [[solutions/rank_03/solution|Imitation Learning: 3rd Place Solution]]
- #4 [[solutions/rank_04/solution|4th Place Solution - Imitation Learning Approach]]
- #5 [[solutions/rank_05/solution|Kiwis xLSTM agent (5th place solution)]]
- #8 [[solutions/rank_08/solution|8th Place Solution - Imitation Learning]]
- #9 [[solutions/rank_09/solution|9th Place Solution]]
- #10 [[solutions/rank_10/solution|10th Place Solution – Boey – End-to-End JAX RL]]
- #14 [[solutions/rank_14/solution|Multi-agent RL Silver solution by 3Comets, 14th place]]
- #29 [[solutions/rank_29/solution|A solution that I hope will earn a silver medal (currently in 29th place)]]
- #65 [[solutions/rank_65/solution|65th~ place solution just on Rules and will to experiment]]
- #92 [[solutions/rank_92/solution|Bronze medal border solution (finally in 92th place)]]
- ? [[solutions/rank_xx_568621/solution|Frog Parade's Solution]]
- ? [[solutions/rank_xx_568721/solution|EcoBangBang's Approach - Yet Another RL Solution]]

## GitHub links
- [tonykozlovsky/lux-ai3-pub](https://github.com/tonykozlovsky/lux-ai3-pub) _(solution)_ — from [[solutions/rank_01/solution|1st place approach by Flat Neurons]]
- [w9PcJLyb/lux3-bot](https://github.com/w9PcJLyb/lux3-bot) _(solution)_ — from [[solutions/rank_03/solution|Imitation Learning: 3rd Place Solution]]
- [IsaiahPressman/kaggle-lux-2024](https://github.com/IsaiahPressman/kaggle-lux-2024) _(solution)_ — from [[solutions/rank_xx_568621/solution|Frog Parade's Solution]]
- [Lux-AI-Challenge/Lux-Design-S3](https://github.com/Lux-AI-Challenge/Lux-Design-S3) _(reference)_ — from [[solutions/rank_xx_568621/solution|Frog Parade's Solution]]
- [KASSII/Kaggle_LuxAI-s3](https://github.com/KASSII/Kaggle_LuxAI-s3) _(solution)_ — from [[solutions/rank_04/solution|4th Place Solution - Imitation Learning Approach]]
- [L16H7/lux-3-comets](https://github.com/L16H7/lux-3-comets) _(solution)_ — from [[solutions/rank_14/solution|Multi-agent RL Silver solution by 3Comets, 14th place]]
- [dunnolab/xland-minigrid](https://github.com/dunnolab/xland-minigrid) _(reference)_ — from [[solutions/rank_14/solution|Multi-agent RL Silver solution by 3Comets, 14th place]]
- [kuto5046/kaggle-luxai-s3](https://github.com/kuto5046/kaggle-luxai-s3) _(solution)_ — from [[solutions/rank_09/solution|9th Place Solution]]
- [flynnwang/eco_bang_bang](https://github.com/flynnwang/eco_bang_bang) _(solution)_ — from [[solutions/rank_xx_568721/solution|EcoBangBang's Approach - Yet Another RL Solution]]
- [IsaiahPressman/Kaggle_Lux_AI_2021](https://github.com/IsaiahPressman/Kaggle_Lux_AI_2021) _(reference)_ — from [[solutions/rank_xx_568721/solution|EcoBangBang's Approach - Yet Another RL Solution]]
- [luchris429/purejaxrl](https://github.com/luchris429/purejaxrl) _(library)_ — from [[solutions/rank_10/solution|10th Place Solution – Boey – End-to-End JAX RL]]
- [Elias-Buerger/kaggle-lux-s3](https://github.com/Elias-Buerger/kaggle-lux-s3) _(solution)_ — from [[solutions/rank_05/solution|Kiwis xLSTM agent (5th place solution)]]
- [luchris429/purejaxrl](https://github.com/luchris429/purejaxrl) _(library)_ — from [[solutions/rank_05/solution|Kiwis xLSTM agent (5th place solution)]]
- [gregorlied/lux-s3](https://github.com/gregorlied/lux-s3) — from [[solutions/rank_08/solution|8th Place Solution - Imitation Learning]]
- [ArtemVeshkin/luxai-s3](https://github.com/ArtemVeshkin/luxai-s3) _(solution)_ — from [[solutions/rank_65/solution|65th~ place solution just on Rules and will to experiment]]

## Papers cited
- [Rl $^ 2$: Fast reinforcement learning via slow reinforcement learning.](https://arxiv.org/abs/1611.02779)
- [XLand-minigrid: Scalable meta-reinforcement learning environments in JAX.](https://github.com/dunnolab/xland-minigrid)
- [A closer look at invalid action masking in policy gradient algorithms.](https://arxiv.org/abs/2006.14171)
- [The surprising effectiveness of ppo in cooperative multi-agent games.](https://arxiv.org/abs/2103.01955)
- [AlphaStar](https://www.nature.com/articles/s41586-019-1724-z)
- [xlstm: Extended long short-term memory](https://arxiv.org/abs/2405.04517)
- [JointPPO: Diving deeper into the effectiveness of PPO in multi-agent reinforcement learning](https://arxiv.org/abs/2404.11831)
- [A convnet for the 2020s](https://openaccess.thecvf.com/content/CVPR2022/html/Liu_A_ConNet_for_the_2020s_CVPR_2022_paper.html)
- [Diffusion Transformer](https://arxiv.org/abs/2212.09748)
