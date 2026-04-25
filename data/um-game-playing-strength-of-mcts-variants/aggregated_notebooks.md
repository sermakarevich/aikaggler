# um-game-playing-strength-of-mcts-variants: top public notebooks

The community's top-voted notebooks primarily establish robust baseline pipelines for predicting MCTS agent utility, heavily emphasizing group-based cross-validation to prevent leakage across unseen game rulesets. They showcase extensive feature engineering workflows that parse agent configuration strings, extract readability metrics from rule texts, and apply TF-IDF vectorization, alongside iterative model training using gradient boosting (CatBoost, LightGBM, XGBoost) and neural networks (DeepTables). Final submissions consistently rely on post-processing techniques like prediction clipping, scaling, and multi-model blending to optimize RMSE on the leaderboard.

## Common purposes
- baseline
- eda
- ensemble
- tutorial
- utility
- training

## Competition flows
- Reads MCTS agent and game rule CSVs, engineers tabular and text features, trains two CatBoost models with StratifiedGroupKFold grouped by game ruleset, applies scaling and clipping post-processing, and generates a submission via Kaggle's inference server.
- Loads CSV data, applies extensive text and domain feature engineering, trains CatBoost and LightGBM models with StratifiedGroupKFold by game type, generates OOF predictions as meta-features, blends two model pipelines, and outputs a submission via Kaggle's inference server.
- Loads training data with Polars, trains a RandomForestRegressor with ordinal encoding, and serves predictions via a custom Kaggle gateway server that streams test chunks to a predict function.
- Initializes a custom MCTS inference server, defines a dummy predict function to handle batched test inputs, and serves predictions locally or on the hidden test set.
- Loads game-playing data, trains initial LightGBM and CatBoost models to generate out-of-fold predictions, adds these predictions as new features, retrains the models, and generates final utility predictions for the test set via a Kaggle inference server.
- Loads training and test CSVs, extracts player strategy components, filters features via permutation importance, trains LightGBM/XGBoost/CatBoost baselines with GroupKFold, blends them via Ridge regression weights, and saves models for inference.
- Loads tabular game data, preprocesses and scales features, trains a DeepTables neural network and a CatBoost regressor using GroupKFold cross-validation, and combines their predictions via weighted averaging to generate a competition submission.
- Loads CSV data with Polars, parses MCTS agent strings and game rules into numerical and TF-IDF features, trains LightGBM and CatBoost regressors using GroupKFold by game ruleset, and averages their predictions for submission.
- Reads CSV data via Polars, applies extensive text parsing, readability metrics, and ratio-based feature engineering, trains two CatBoost models with StratifiedGroupKFold grouped by game ruleset, and averages their clipped predictions for submission.
- Reads training data with Polars, preprocesses and casts datatypes, trains a LightGBM regressor via GroupKFold cross-validation, and wraps the inference logic in a Kaggle MCTS evaluation server for automated submission generation.

## Data reading
- polars.read_csv()
- pd.read_csv()
- Direct pass to predict function via evaluation gateway
- Hardcoded Kaggle input paths for local testing

## Data processing
- String cleaning (lowercasing, punctuation removal)
- Outlier clipping (PlayoutsPerSecond, MovesPerSecond)
- Memory dtype casting (int8/16/32/64, float16/32/64, Int16, Float32)
- TF-IDF vectorization (ngram 2-3, max_features 500-600)
- Dropping constant/low-frequency/null columns
- Parsing agent strings and LudRules
- Ordinal encoding for string/object columns
- Converting Polars DataFrames to Pandas
- Converting string columns to Pandas Categorical dtype
- Clipping predictions to (-1, 1) or [-0.985, 0.985]
- Splits agent identifiers into categorical components
- MinMaxScaler normalization (fit on training data)
- GPU acceleration flags
- One-hot encoding for known categories
- Dropping game-specific/board rule columns

## Features engineering
- Parsing MCTS agent strings into selection, exploration, playout, and score bounds
- Binary flags for predefined agent combinations/positions
- Domain-derived ratios and interaction features (e.g., Playouts/Moves, ComplexityBalanceInteraction)
- Readability indices (ARI, CLRI, McAlpine_EFLAW) on rule texts
- TF-IDF vectorization on EnglishRules and LudRules
- Out-of-fold (OOF) predictions as meta-features
- Permutation importance-based feature selection
- Cross-features (sum, diff, product, division) on selected features
- One-hot encoding for known categorical/board properties
- DeepTables auto-generated GBM features
- Dropping game names from LudRules to prevent leakage

## Models
- CatBoost
- LightGBM
- XGBoost
- RandomForestRegressor
- ExtraTrees
- LinearRegression
- Ridge
- KNeighborsRegressor
- ExplainableBoostingRegressor
- DummyRegressor
- DeepTables

## Frameworks used
- polars
- pandas
- numpy
- scikit-learn
- catboost
- dill
- matplotlib
- kaggle_evaluation
- kaggle_evaluation.mcts_inference_server
- lightgbm
- plotly
- xgboost
- interpret
- tensorflow
- deeptables

## Loss functions
- RMSE
- MSE

## CV strategies
- StratifiedGroupKFold(n_splits=5 or 10, random_state=2024, shuffle=True) grouped by GameRulesetName, with target rounded to integers for stratification.
- StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=2024) grouped by 'GameRulesetName'
- GroupKFold(n_splits=5) grouped by GameRulesetName
- GroupKFold(n_splits=5) split by GameRulesetName; GroupShuffleSplit(n_splits=5) for feature importance calculation
- GroupKFold(n_splits=6) grouped by GameRulesetName
- GroupKFold(n_splits=10) by GameRulesetName

## Ensembling
- Averages predictions across two CatBoost model variants and multiple folds, then applies a 1.1x scaling factor and clips results to [-0.985, 0.985] to minimize overall RMSE.
- Blends predictions from two independent GBM pipelines (Model 1 and Model 2) with equal weights (0.5 each); Model 2 internally blends LightGBM and CatBoost with weights 0.8 and 0.3.
- Combines predictions from initial and final models using fixed weights (LightGBM: 1.0, CatBoost: 0.0), effectively relying on the final LightGBM model trained with OOF features.
- Blends out-of-fold predictions from LightGBM, XGBoost, CatBoost, Linear Regression, Kernel Approximation, and KNN using Ridge regression weights learned on training data; post-processes predictions by clipping to (-1, 1).
- Weighted average of DeepTables and CatBoost predictions with final weights of 0.40 for NN and 0.60 for CatBoost.
- Averages predictions from LightGBM and CatBoost models.
- Averages predictions from 10 models (2 CatBoost variants × 5 folds), applies a linear scaling factor (×1.1) and clips outputs to [-0.985, 0.985] to minimize RMSE.
- Averages predictions across the 5 GroupKFold folds during inference.

## Insights
- votes 681 (MCTS Starter): Splitting CV by game ruleset is critical for generalizing to unknown games.
- votes 593 (MCTS | OOF PredFE+PubNB Blending|LB[.425]): OOF predictions can be safely reused as meta-features for a second training round without data leakage.
- votes 372 (MCTS EDA which makes sense ⭐⭐⭐⭐⭐): Only ~30-40% of the hundreds of game features are actually useful for prediction.
- votes 356 (MCTS: DeepTables NN): Proper train-only scaling is critical for neural network stability and prevents validation divergence.
- votes 303 (MCTS Starter | [TRAIN+INF] HPupd [LB.431]): Dropping game names from LudRules avoids CV/LB mismatch across different game types.
- votes 201 (MCTS Starter): Clipping predictions to [-0.985, 0.985] improves overall RMSE by balancing errors across the target distribution.
- votes 197 (MCTS | LightGBM Baseline): Using Polars with batch reading significantly reduces memory overhead during data ingestion.

## Critical findings
- Target variable is clipped to [-0.985, 0.985] and scaled by 1.1 during training/inference to constrain output ranges.
- Many high-cardinality or low-frequency categorical columns are dropped to prevent overfitting and reduce memory.
- Game ruleset names are used as group labels for CV, indicating that game types have distinct feature distributions.
- AdvantageP1 = 1.0 does not guarantee player 1 wins in the actual game, requiring cautious interpretation.
- DrawFrequency = 1.0 only results in a draw about one-third of the time, indicating a weak direct correlation.
- The training set has more extreme target values than the test set, but the distribution difference is not statistically significant.
- Scaling numerical features by fitting on combined train and test data temporarily fixes divergence but violates standard ML practices and should be avoided.
- GPU training introduces non-deterministic behavior that causes slight score variance across runs.
- DeepTables has known model persistence issues on Kaggle that require manual file copying to /tmp/workdir to resolve.

## What did not work
- Removing XGBoost, dropping the area feature, and removing agent1/agent2 TF-IDF features improved CV/LB scores.
- Version 7's outlier clipping and threshold adjustments initially worsened CV/LB, indicating over-regularization or information loss.

## Notable individual insights
- votes 681 (MCTS Starter): Splitting CV by game ruleset is critical for generalizing to unknown games.
- votes 593 (MCTS | OOF PredFE+PubNB Blending|LB[.425]): OOF predictions can be safely reused as meta-features for a second training round without data leakage.
- votes 372 (MCTS EDA which makes sense ⭐⭐⭐⭐⭐): Only ~30-40% of the hundreds of game features are actually useful for prediction.
- votes 356 (MCTS: DeepTables NN): Proper train-only scaling is critical for neural network stability and prevents validation divergence.
- votes 303 (MCTS Starter | [TRAIN+INF] HPupd [LB.431]): Dropping game names from LudRules avoids CV/LB mismatch across different game types.
- votes 201 (MCTS Starter): Clipping predictions to [-0.985, 0.985] improves overall RMSE by balancing errors across the target distribution.
- votes 197 (MCTS | LightGBM Baseline): Using Polars with batch reading significantly reduces memory overhead during data ingestion.

## Notebooks indexed
- #681 votes [[notebooks/votes_01_yunsuxiaozi-mcts-starter/notebook|MCTS Starter]] ([kaggle](https://www.kaggle.com/code/yunsuxiaozi/mcts-starter))
- #593 votes [[notebooks/votes_02_hideyukizushi-mcts-oof-predfe-pubnb-blending-lb-425/notebook|MCTS | OOF PredFE+PubNB Blending|LB[.425]]] ([kaggle](https://www.kaggle.com/code/hideyukizushi/mcts-oof-predfe-pubnb-blending-lb-425))
- #581 votes [[notebooks/votes_03_inversion-mcts-variants-getting-started/notebook|MCTS Variants - Getting Started]] ([kaggle](https://www.kaggle.com/code/inversion/mcts-variants-getting-started))
- #435 votes [[notebooks/votes_04_sohier-mcts-demo-submission/notebook|MCTS Demo Submission]] ([kaggle](https://www.kaggle.com/code/sohier/mcts-demo-submission))
- #392 votes [[notebooks/votes_05_andreasbis-mcts-oof-predictions-as-features/notebook|MCTS | OOF Predictions as Features]] ([kaggle](https://www.kaggle.com/code/andreasbis/mcts-oof-predictions-as-features))
- #372 votes [[notebooks/votes_06_ambrosm-mcts-eda-which-makes-sense/notebook|MCTS EDA which makes sense ⭐️⭐️⭐️⭐️⭐️]] ([kaggle](https://www.kaggle.com/code/ambrosm/mcts-eda-which-makes-sense))
- #356 votes [[notebooks/votes_07_yekenot-mcts-deeptables-nn/notebook|MCTS: DeepTables NN]] ([kaggle](https://www.kaggle.com/code/yekenot/mcts-deeptables-nn))
- #303 votes [[notebooks/votes_08_hideyukizushi-mcts-starter-train-inf-hpupd-lb-431/notebook|MCTS Starter | [TRAIN+INF] HPupd [LB.431]]] ([kaggle](https://www.kaggle.com/code/hideyukizushi/mcts-starter-train-inf-hpupd-lb-431))
- #201 votes [[notebooks/votes_09_bruceqdu-mcts-starter/notebook|MCTS Starter]] ([kaggle](https://www.kaggle.com/code/bruceqdu/mcts-starter))
- #197 votes [[notebooks/votes_10_andreasbis-mcts-lightgbm-baseline/notebook|MCTS | LightGBM Baseline]] ([kaggle](https://www.kaggle.com/code/andreasbis/mcts-lightgbm-baseline))
