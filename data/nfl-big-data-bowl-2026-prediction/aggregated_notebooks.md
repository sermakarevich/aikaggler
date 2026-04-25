# nfl-big-data-bowl-2026-prediction: top public notebooks

The community's top-voted notebooks primarily focus on building robust trajectory prediction pipelines that combine deterministic geometric baselines with residual learning approaches. They heavily emphasize extensive hand-crafted feature engineering (GNN-lite embeddings, route clustering, temporal lags) and custom PyTorch architectures like Transformers and GRUs, alongside production-ready inference boilerplates and multi-fold ensembling strategies.

## Common purposes
- training
- baseline
- inference
- utility

## Competition flows
- Kaggle inference server initialization with placeholder predict function and environment-based routing
- NFL tracking data loading with play direction unification, 150+ feature computation, PyTorch Transformer training, and 5-10 fold averaging
- Weekly tracking data loading with physics/sequence/formation/graph feature engineering, CatBoost residual training, and global/role-specific blending
- NFL play CSV loading with 167 spatio-temporal/geometric features, STTransformer training across 10 GroupKFold splits, and fold averaging
- Test data loading with extensive kinematic/spatial/geometric engineering, 5-fold PyTorch transformer ensemble, and coordinate inversion
- Weekly CSV loading with 167 geometric/spatial/temporal features, 5-fold GRU-Attention training, and field-clipped submission
- Player tracking CSV loading with geometric/route/neighbor features, 10-fold STTransformer training, and GRU ensemble
- Pre-trained model/scaler loading, sequence processing, TTA ensemble prediction, coordinate transformation, and API submission
- Trajectory CSV loading with direction unification, kinematic/interaction feature engineering, sliding-window sequence building, GRU-Attention training with custom loss, and multi-seed/fold/TTA averaging
- Weekly CSV loading with geometric baselines/spatial-temporal features, ST-Transformer training, and fold averaging

## Data reading
- pd.read_csv/pd.concat for weekly train input/output CSVs
- Multiprocessing for parallel loading
- Polars/pandas conversion for test batches
- kaggle_evaluation API for platform data
- int64 casting for IDs

## Data processing
- Play direction unification/mirroring
- Sequence padding to fixed window sizes
- NaN/missing value imputation (column/group means, forward fill, zeros)
- Coordinate clipping to field boundaries
- StandardScaler normalization (per-fold)
- KMeans route clustering
- Height/BMI conversion
- Velocity/acceleration vector derivation
- Ball-relative geometry & formation centroids
- Lagged features & rolling statistics (mean/std)
- Weighted k-NN/GNN-lite neighbor embeddings
- Infinity replacement & forward fill
- Time-decay weighting application
- Polars to Pandas conversion

## Features engineering
- Geometric endpoint baselines & projections
- GNN-lite weighted neighbor embeddings
- Route trajectory clustering
- Lag/rolling mean/std features
- Velocity/acceleration components
- Momentum/kinetic energy
- Pressure metrics
- Time-to-intercept
- Role-specific geometric coupling features
- Height/BMI & heading vectors
- Ball-frame projections
- Physics features
- Team centroid/width/length
- Relative coordinates & bearing
- Graph-based neighbor embeddings
- Kinematic derivatives
- Distance-to-ball metrics
- Multi-window rolling statistics
- Field position flags
- Sinusoidal time features
- Receiver-defender interaction features
- QB-relative geometry features
- Time-progress ratios
- Curvature/landing geometry

## Models
- SpatioTemporalTransformer
- ImprovedSeqModel
- CatBoostRegressor
- STTransformer
- JointSeqModel
- ResidualMLPHead
- GRU
- SeqModel
- ResidualMLP

## Frameworks used
- polars
- pandas
- kaggle_evaluation.nfl_inference_server
- pytorch/torch
- scikit-learn
- numpy
- catboost
- cupy

## Loss functions
- RMSE
- TemporalHuber (Huber loss with exponential time decay weighting)
- TemporalHuber (Huber loss with exponential time decay + 2nd-order velocity smoothness regularization)

## CV strategies
- GroupKFold (n_splits=5 to 10), grouped by game_id, game_id_play_id_nfl_id, or f'{game_id}_{play_id}'

## Ensembling
- Fold averaging (5-10 folds)
- Multi-seed averaging
- Blending global/role-specific predictions
- TTA (Gaussian noise/context flipping)
- Coordinate inversion/clipping

## Insights
- The inference API requires a specific predict function signature that accepts Polars DataFrames and returns predictions within strict time limits.
- Participants must use kaggle_evaluation.nfl_inference_server to wrap their logic and can run the gateway locally for testing.
- Polars is recommended over Pandas for inference performance.
- Geometric rules form a strong deterministic baseline that the model only needs to learn corrections for.
- Attention pooling over temporal sequences effectively captures player trajectory dynamics.
- GNN-lite weighted neighbor embeddings successfully encode spatial interactions without full graph neural networks.
- Route clustering and lag features provide crucial context for player intent and recent motion.
- Residual learning with a kinematic baseline stabilizes position predictions and simplifies the regression task for tree models.
- Role-specific heads capture distinct movement patterns for key player types without degrading global performance.
- Lightweight graph-based neighbor embeddings effectively encode spatial context without requiring heavy GNN frameworks.
- Multi-GPU CatBoost can be enabled with minimal code changes by passing detected device IDs directly to the devices parameter.
- Training the model to predict corrections to a deterministic geometric baseline is more effective than predicting raw coordinates.
- Role-specific geometric rules (e.g., receivers targeting ball landing, defenders mirroring receivers) provide strong priors that reduce the learning burden.
- Weighted neighbor embeddings and route clustering capture spatial and behavioral patterns without full GNN computation.
- Hand-crafted geometric and kinematic features significantly enrich the input representation for position prediction.
- GNN-style weighted neighbor embeddings effectively capture spatial relationships between players.
- Predicting positional residuals rather than absolute coordinates may contribute to overfitting.
- Simpler models combined with robust data preprocessing might outperform complex architectures in this task.
- Decomposing movement into a deterministic geometric baseline plus neural corrections reduces the learning burden and improves generalization.
- GNN-lite weighted neighbor embeddings effectively capture spatial coupling between players without full graph computation.
- Time-progress and ball-landing alignment features are critical for forecasting future trajectories in football.
- Combining a deterministic geometric baseline with learned neural corrections improves trajectory prediction accuracy.
- Route clustering and spatial neighbor embeddings effectively capture player roles and interactions.
- Time-decayed loss weighting prioritizes earlier prediction steps in trajectory forecasting.
- Pre-loading models once and caching them avoids repeated initialization overhead during inference.
- Test-time augmentation (TTA) across multiple model checkpoints improves prediction stability.
- Coordinate inversion based on play_direction is necessary to align predictions with the original field orientation.
- Unifying play direction to a single orientation simplifies sequence modeling and avoids directional bias.
- Using a custom temporal Huber loss with smoothness regularization improves trajectory continuity.
- Multi-seed and multi-fold averaging combined with TTA provides robustness against overfitting and data quirks.
- Geometric and physics-based rules capture the majority of player movement, allowing the model to focus on learning complex corrections.
- Route clustering and GNN-lite embeddings effectively encode spatial context without heavy graph computation.
- Time-decayed loss prioritizes accuracy at later prediction horizons.

## Critical findings
- The author explicitly notes the model is overfitting and hypothesizes it stems from predicting position residuals instead of absolute positions, and that some engineered features may be redundant.
- Ball landing features must be disabled if unavailable in the test set to prevent errors.
- Aggressive interaction features are commented out for a conservative version to avoid unstable behavior.

## Notable individual insights
- votes 362 (NFL 2026 Demo Submission): The inference API requires a specific predict function signature that accepts Polars DataFrames and returns predictions within strict time limits.
- votes 346 (Ensemble mymodell): Geometric rules form a strong deterministic baseline that the model only needs to learn corrections for.
- votes 316 (NFL Big Data - Baseline): Residual learning with a kinematic baseline stabilizes position predictions and simplifies the regression task for tree models.
- votes 303 (NFL Big Data Bowl 2026 - Geometry GNN): Training the model to predict corrections to a deterministic geometric baseline is more effective than predicting raw coordinates.
- votes 269 (1105sucess_infer): The author explicitly notes the model is overfitting and hypothesizes it stems from predicting position residuals instead of absolute positions.
- votes 245 (🏈NFL Big Data Bowl - Geometric GNN [LB .586]): Decomposing movement into a deterministic geometric baseline plus neural corrections reduces the learning burden and improves generalization.
- votes 194 (NFL2026 Prediction|LB0.584(Before adding eval API)): Unifying play direction to a single orientation simplifies sequence modeling and avoids directional bias.

## Notebooks indexed
- #362 votes [[notebooks/votes_01_sohier-nfl-2026-demo-submission/notebook|NFL 2026 Demo Submission]] ([kaggle](https://www.kaggle.com/code/sohier/nfl-2026-demo-submission))
- #346 votes [[notebooks/votes_02_goose666-ensemble-mymodell/notebook|Ensemble mymodell]] ([kaggle](https://www.kaggle.com/code/goose666/ensemble-mymodell))
- #316 votes [[notebooks/votes_03_hiwe0305-nfl-big-data-baseline/notebook|NFL Big Data - Baseline]] ([kaggle](https://www.kaggle.com/code/hiwe0305/nfl-big-data-baseline))
- #303 votes [[notebooks/votes_04_pankajiitr-nfl-big-data-bowl-2026-geometry-gnn/notebook|NFL Big Data Bowl 2026 - Geometry GNN]] ([kaggle](https://www.kaggle.com/code/pankajiitr/nfl-big-data-bowl-2026-geometry-gnn))
- #269 votes [[notebooks/votes_05_goose666-1105sucess-infer/notebook|1105sucess_infer]] ([kaggle](https://www.kaggle.com/code/goose666/1105sucess-infer))
- #245 votes [[notebooks/votes_06_ryanadamsai-nfl-big-data-bowl-geometric-gnn-lb-586/notebook|🏈NFL Big Data Bowl - Geometric GNN [LB .586]]] ([kaggle](https://www.kaggle.com/code/ryanadamsai/nfl-big-data-bowl-geometric-gnn-lb-586))
- #239 votes [[notebooks/votes_07_pankajiitr-ensemble-nfl-big-data-bowl-2026-v1/notebook|Ensemble NFL Big Data Bowl 2026 V1]] ([kaggle](https://www.kaggle.com/code/pankajiitr/ensemble-nfl-big-data-bowl-2026-v1))
- #199 votes [[notebooks/votes_08_mango789-nfl2026-submission/notebook|NFL2026 | Submission]] ([kaggle](https://www.kaggle.com/code/mango789/nfl2026-submission))
- #194 votes [[notebooks/votes_09_chengzhijiang-nfl2026-prediction-lb0-584-before-adding-eval-api/notebook|NFL2026 Prediction|LB0.584(Before adding eval API)]] ([kaggle](https://www.kaggle.com/code/chengzhijiang/nfl2026-prediction-lb0-584-before-adding-eval-api))
- #184 votes [[notebooks/votes_10_pankajiitr-nfl-2026-3-models/notebook|NFL 2026 3 Models]] ([kaggle](https://www.kaggle.com/code/pankajiitr/nfl-2026-3-models))
