# nfl-big-data-bowl-2026-analytics: top public notebooks

The community's top notebooks for the NFL Big Data Bowl 2026 heavily emphasize spatial-temporal exploratory data analysis and physics-based feature engineering to transform raw player tracking coordinates into actionable play-level metrics. Alongside comprehensive EDA and automated baseline tabular pipelines, several contributors explore advanced architectures like custom transformers for coverage prediction, while others introduce novel domain-specific indices such as Air-Time Defense Closure and Air Yards Efficiency Index. Visualization utilities and supplemental AI-generated charting data integration round out the collection, highlighting a strong focus on interpretability, domain knowledge, and frame-by-frame narrative over complex ensembling.

## Common purposes
- eda
- tutorial
- feature_engineering
- utility
- baseline

## Competition flows
- Loads tracking and supplementary CSV data across 18 weeks, computes spatial-temporal player metrics and field density visualizations, and exports summary statistics and metric distributions for competition writeups.
- Loads weekly tracking CSVs and supplementary metadata, merges them by game/play/player, computes spatial-temporal metrics like separation and speed distributions, and visualizes the data through interactive charts and animations to understand pass play dynamics.
- Loads raw NFL tracking CSVs and supplementary labels, preprocesses coordinates into kinematic sequences, trains a custom 4-layer transformer via PyTorch Lightning, and generates frame-level and play-level Man/Zone coverage predictions with evaluation metrics and animations.
- Loads NFL tracking CSV data for a selected play, computes a grid-based territorial control metric using player positions and velocities, and renders the result as an animated heatmap visualization.
- Loads raw NFL tracking CSV data, filters it for a specific game and play, plots player coordinates across frames on a custom-drawn field, and compiles the static plots into an animated GIF.
- Loads weekly NFL Big Data Bowl 2026 tracking CSVs, merges input and output data on game/play/player/frame identifiers, applies a multi-stage feature engineering pipeline combining physics, spatial, and contextual metrics, and outputs engineered features as Parquet files alongside comprehensive EDA visualizations and baseline feature importance scores.
- Loads supplemental frame-level and player-level coverage parquet files, analyzes coverage scheme evolution from snap to end of play, joins the data with official tracking data, and generates interactive Plotly visualizations and animations.
- Ingests multiple input/output CSVs via glob patterns, auto-detects the target column and task type, performs EDA and preprocessing, trains baseline models, and evaluates them with task-specific metrics and error analysis.
- Generates synthetic NFL tracking data, calculates a custom Air-Time Defense Closure metric, visualizes spatial-temporal relationships, engineers predictive features, and trains baseline regression models to evaluate defensive coverage effectiveness.
- Loads weekly NFL tracking and supplementary CSV files, computes frame-by-frame movement and distance metrics, aggregates them into a weighted Air Yards Efficiency Index per player-play, and analyzes the index across positions, roles, and pass depths.

## Data reading
- Iteratively loads input_2023_w{week:02d}.csv and output_2023_w{week:02d}.csv files using pd.read_csv() within a try/except loop over weeks 1-18.
- Concatenates tracking frames across weeks with pd.concat(ignore_index=True).
- Merges play-level metadata from supplementary_data.csv on ['game_id', 'play_id'] using pd.merge().
- Reads weekly `input_2023_w{week:02}.csv` and `output_2023_w{week:02}.csv` files via `pandas.read_csv`, concatenates them across weeks, and merges with `supplementary_data.csv` on `game_id` and `play_id`.
- Raw CSVs from data/train/ and data/supplementary_data.csv are loaded via a custom prep_data.main() function.
- Preprocessed splits are later read from parquet files using pandas.read_parquet and a custom load_datasets utility.
- Reads input and output tracking CSV files via pd.read_csv, filters rows by game_id and play_id, and extracts frame_id, x, y, nfl_id, and player_side columns.
- Reads a single CSV file (`train/input_2023_w01.csv`) using pandas `read_csv`.
- Reads parquet files using Polars (pl.read_parquet) for player-play and frame-level coverage data.
- Reads CSV tracking data using Polars (pl.read_csv).
- Uses glob.glob to match input_2023_*.csv and output_2023_*.csv patterns, reads them with pd.read_csv(..., low_memory=False), and concatenates them.
- Optionally loads supplementary_data.csv if present.
- pd.read_csv for input_2023_w{week}.csv, output_2023_w{week}.csv (weeks 1-18), and supplementary_data.csv
- Concatenated across weeks using pd.concat(ignore_index=True)

## Data processing
- Applies week-by-week error handling and safe sampling to prevent memory bottlenecks during visualization.
- Standard-scales numerical features (s, a, o, dir, x, y) using sklearn.preprocessing.StandardScaler for clustering.
- Computes Euclidean distances and frame-to-frame velocity changes using scipy.spatial.distance.euclidean and pandas diff operations.
- Filters and aggregates data by player_role, player_position, and game quarter for stratified analysis.
- Sorts merged frames by game/play/frame, filters by player position/role, and applies physics-based validation checks for speed and acceleration bounds.
- Converts polar to Cartesian coordinates and standardizes play directions.
- Extracts kinematic features (x, y, s, a, vx, vy, ox, oy) for 22 players.
- Splits data 70/15/15, pre-computes feature transforms, and packages into pickle/parquet.
- Filters predictions to the last frame per play for final evaluation.
- Calculates frame-to-frame velocity assuming a 0.1s timestep, pivots long-format player tracking data to wide format indexed by frame_id, and discretizes the field into a 30x13 grid for spatial computation.
- Filters data to a single play and extracts unique frame IDs for iteration.
- Dynamically maps categorical `player_role` values to colors using a fallback dictionary logic.
- Handles missing role data and NaN values with conditional checks; no numerical scaling, augmentation, or feature transformation is applied.
- Filters rows to player_to_predict == True.
- Converts height strings to inches and one-hot encodes categorical positions and roles.
- Applies sin/cos trigonometric transformations to orientation and direction columns to handle circular continuity.
- Bins yardline numbers into field thirds and computes Euclidean distances to ball landing spots, sidelines, and endzones.
- Groups tracking data by play and player to calculate maximum kinematic metrics per sequence.
- Imputes missing values with medians for baseline modeling.
- Filters data by game_id and play_id to isolate specific plays.
- Casts nfl_id to string for consistent joins across datasets.
- Broadcasts frame-level predictions to player-level data via left joins.
- Converts Polars DataFrames to Pandas for statistical aggregation and Plotly visualization.
- Merges input and output dataframes on detected join keys, drops non-feature columns, and splits data with train_test_split (stratified for classification).
- Imputes missing values using SimpleImputer (median for numeric, most frequent for categorical).
- One-hot encodes categorical features with OneHotEncoder(handle_unknown='ignore', sparse=True).
- Scales numeric features with StandardScaler(with_mean=False).
- Feature scaling via `StandardScaler` before modeling
- Categorical binning of ATDC scores into performance tiers
- Merges input and output tracking data on game_id and play_id
- Sorts by game_id, play_id, nfl_id, frame_id
- Uses shift for frame-to-frame comparisons
- Filters players with fewer than 15 plays
- Bins pass length into depth categories using pd.cut

## Features engineering
- Receiver Separation Score: Weighted combination of minimum and average defender distance at the catch point.
- Defensive Response Time: Mean absolute velocity change in the first 3 frames after ball release.
- Route Efficiency Index: Direct Euclidean distance to ball landing divided by total traveled distance.
- Aggregated speed and acceleration statistics grouped by player role and position.
- Computes receiver-defender separation metrics (nearest distance, defenders within 3 yards, average distance), derives throw time from frame counts, estimates pass distance from ball landing coordinates, and calculates displacement/speed/acceleration statistics by role.
- Hand-crafted time-to-target metrics for offense and defense using fixed elite speed and reaction time parameters, combined via a sigmoid function to generate a Pass Coverage Control probability grid.
- position one-hot encodings
- role one-hot encodings
- height in inches
- weight
- is_offense flag
- velocity_x/y components
- speed_squared
- is_moving flag
- momentum_x/y/magnitude
- kinetic_energy
- yardline
- field_third (binned)
- moving_left flag
- ball_land_x/y
- dist_to_ball_land
- frames_to_predict
- is_targeted_receiver
- is_passer
- is_wide_receiver
- dist_to_sideline
- dist_to_endzone
- eyes_on_qb
- eyes_on_receiver
- max_speed_this_play
- max_acceleration_this_play
- max_kinetic_energy_this_play
- max_momentum_this_play
- Speed differential between defender and receiver
- Distance ratio (final distance / initial distance)
- Pursuit efficiency (closure distance normalized by air time and defender speed)
- Binned ATDC performance categories (Poor, Average, Good, Excellent)
- Euclidean distance to ball landing location
- Frame-to-frame movement distance
- Closing velocity (scaled by 10 for yards/sec)
- Path efficiency (optimal straight-line distance / actual path distance)
- Closing efficiency (avg closing velocity / avg speed)
- Acceleration timing (middle-third acceleration / overall acceleration)
- Weighted AYEI composite score (0.5 path + 0.3 closing + 0.2 acceleration timing)
- Player-level aggregations (mean, std, count, percentile ranking)

## Models
- KMeans
- DBSCAN
- PCA
- t-SNE
- RandomForestRegressor
- RandomForestClassifier
- LogisticRegression
- LinearRegression
- DummyClassifier
- DummyRegressor
- GradientBoostingRegressor
- Custom transformer (LitModel_PA) with 64-dim embeddings, 4 layers, and 0.1 dropout

## Frameworks used
- pandas
- numpy
- matplotlib
- seaborn
- scipy
- scikit-learn
- plotly
- torch
- lightning
- polars
- plotly.express
- plotly.graph_objects
- PIL

## Loss functions
- BCEWithLogitsLoss

## CV strategies
- Fixed train/val/test split (70/15/15)
- 5-fold cross-validation using `cross_val_score` with `scoring='neg_mean_squared_error'`

## Ensembling
- None. The notebook trains and compares individual models without combining them.

## Insights
- High receiver separation and route efficiency strongly correlate with pass completion and expected points added, while early defensive reaction speed significantly impacts coverage success.
- Receiver separation and spatial positioning during the in-air phase are primary drivers of pass play outcomes.
- Frame-by-frame tracking data can be effectively structured and visualized using spatial-temporal plots to reveal strategic movement patterns.
- Player roles and positions fundamentally dictate speed distributions, acceleration profiles, and separation dynamics.
- Merging pre-throw and in-air tracking data with supplementary metadata enables comprehensive play-level context for feature derivation.
- Transformers effectively model spatio-temporal player interactions and handle variable player ordering via self-attention.
- Filtering predictions to the last frame per play aligns model outputs with play-level ground truth for accurate evaluation.
- Frame-by-frame probability visualization reveals how coverage predictions evolve dynamically during a play.
- Adapting soccer pitch control to NFL passing plays provides a quantifiable metric for defensive swarming and territorial advantage.
- Fixed elite speed and reaction time parameters offer a simplified but effective heuristic for estimating player arrival times.
- Pivoting long-format tracking data to wide format enables efficient frame-by-frame spatial calculations.
- Matplotlib animation capabilities can effectively broadcast tracking analytics in a replay-style format.
- Frame-by-frame coordinate plotting effectively translates raw tracking data into an intuitive spatial narrative of a single play.
- Dynamic role-to-color mapping ensures the visualization adapts to varying categorical labels without hardcoding every possible role.
- Highlighting specific entities (QB, targeted receiver, ball landing spot) with distinct markers and text labels makes complex positional data immediately actionable for analysis.
- Trigonometric encoding of orientation and direction prevents artificial discontinuities at 0°/360° boundaries.
- Distance to the ball landing spot strongly correlates with receiver and defender route adjustments.
- Player awareness metrics (orientation toward QB or targeted receiver) provide actionable spatial context beyond raw coordinates.
- Momentum and kinetic energy features capture physical constraints that raw speed and acceleration miss.
- Field position (yardline/third) fundamentally shifts route concepts and target location distributions.
- Grouping tracking data by play to extract maximum kinematic metrics reveals player capability ceilings that frame-level data obscures.
- AI-generated frame-level coverage data can effectively reveal how defensive schemes disguise and evolve from pre-snap to post-snap.
- Joining supplemental charting data with official tracking data enables rich, frame-by-frame visualizations of player alignments and responsibilities.
- Coverage distributions shift significantly during plays, with a notable trend toward 2-high shell coverages by the end of the play.
- Auto-detecting target columns and task types reduces boilerplate when adapting pipelines to new datasets.
- Using StandardScaler(with_mean=False) prevents numerical instability when combined with sparse one-hot encoded categorical features.
- Subgroup performance checks and top-error analysis quickly reveal where baseline models fail and guide targeted improvements.
- ATDC effectively quantifies defensive coverage by combining initial separation, final separation, and ball flight time.
- Defender-receiver speed differential and initial distance are the strongest predictors of defensive closure success.
- Tree-based models capture non-linear relationships in tracking data better than linear regression.
- Cross-validation confirms that baseline regression models generalize reasonably well to unseen synthetic plays.
- AYEI effectively captures multi-dimensional movement efficiency by weighting path optimization, speed reduction toward the ball, and acceleration timing.
- Elite receivers demonstrate substantially higher AYEI scores than average players, validating the metric's discriminative power.
- AYEI correlates with pass completion outcomes, suggesting it can predict receiver success.
- Defensive efficiency varies significantly by coverage type and pass depth, highlighting context-dependent performance.

## Critical findings
- Some strategic zone visualizations rely on simulated probability distributions rather than raw data, and frame-by-frame distance calculations require aggressive sampling to avoid memory bottlenecks.
- Tracking data strictly adheres to human physics limits, with speed and acceleration violations being negligible.
- Inside linebackers exhibit late-air acceleration bursts that close separation gaps, impacting receiver targeting windows.

## Notable individual insights
- votes 97 (EDA + MODEL + VIDEO + 3D VIDEO): High receiver separation and route efficiency strongly correlate with pass completion and expected points added, while early defensive reaction speed significantly impacts coverage success.
- votes 30 (Modeling with Transformers 2026): Transformers effectively model spatio-temporal player interactions and handle variable player ordering via self-attention.
- votes 28 (NFL 2026: The Pass Coverage Control (PCC) Model): Adapting soccer pitch control to NFL passing plays provides a quantifiable metric for defensive swarming and territorial advantage.
- votes 21 (🛠️ Physics - Football - Features - EDA - Updated): Trigonometric encoding of orientation and direction prevents artificial discontinuities at 0°/360° boundaries.
- votes 20 (SumerSports Supplement Investigation): AI-generated frame-level coverage data can effectively reveal how defensive schemes disguise and evolve from pre-snap to post-snap.
- votes 16 (NFL-BigDataBowl-2026-Analytics): ATDC effectively quantifies defensive coverage by combining initial separation, final separation, and ball flight time.
- votes 16 (NFL Big Data Bowl 2026/Air Yards Efficiency Index): AYEI effectively captures multi-dimensional movement efficiency by weighting path optimization, speed reduction toward the ball, and acceleration timing.

## Notebooks indexed
- #97 votes [[notebooks/votes_01_taylorsamarel-eda-model-video-3d-video/notebook|EDA + MODEL + VIDEO + 3D VIDEO]] ([kaggle](https://www.kaggle.com/code/taylorsamarel/eda-model-video-3d-video))
- #59 votes [[notebooks/votes_02_ajaysamp-nfl-big-data-bowl-2026-eda-upgraded/notebook|🏈 NFL Big Data Bowl 2026: EDA Upgraded ]] ([kaggle](https://www.kaggle.com/code/ajaysamp/nfl-big-data-bowl-2026-eda-upgraded))
- #30 votes [[notebooks/votes_03_vishakhsandwar-modeling-with-transformers-2026/notebook|Modeling with Transformers 2026]] ([kaggle](https://www.kaggle.com/code/vishakhsandwar/modeling-with-transformers-2026))
- #28 votes [[notebooks/votes_04_harshgupta4444-nfl-2026-the-pass-coverage-control-pcc-model/notebook|NFL 2026: The Pass Coverage Control (PCC) Model]] ([kaggle](https://www.kaggle.com/code/harshgupta4444/nfl-2026-the-pass-coverage-control-pcc-model))
- #25 votes [[notebooks/votes_05_stpeteishii-nfl-2026-track-animator/notebook|NFL 2026 Track Animator]] ([kaggle](https://www.kaggle.com/code/stpeteishii/nfl-2026-track-animator))
- #21 votes [[notebooks/votes_06_ajaysamp-physics-football-features-eda-updated/notebook|🛠️  Physics - Football - Features - EDA - Updated]] ([kaggle](https://www.kaggle.com/code/ajaysamp/physics-football-features-eda-updated))
- #20 votes [[notebooks/votes_07_vishakhsandwar-sumersports-supplement-investigation/notebook|SumerSports Supplement Investigation]] ([kaggle](https://www.kaggle.com/code/vishakhsandwar/sumersports-supplement-investigation))
- #18 votes [[notebooks/votes_08_anthonytherrien-nfl-2026-eda-base/notebook|NFL 2026 | EDA - Base]] ([kaggle](https://www.kaggle.com/code/anthonytherrien/nfl-2026-eda-base))
- #16 votes [[notebooks/votes_09_manish5323-nfl-bigdatabowl-2026-analytics/notebook|NFL-BigDataBowl-2026-Analytics]] ([kaggle](https://www.kaggle.com/code/manish5323/nfl-bigdatabowl-2026-analytics))
- #16 votes [[notebooks/votes_10_olaflundstrom-nfl-big-data-bowl-2026-air-yards-efficiency-index/notebook|NFL Big Data Bowl 2026/Air Yards Efficiency Index]] ([kaggle](https://www.kaggle.com/code/olaflundstrom/nfl-big-data-bowl-2026-air-yards-efficiency-index))
