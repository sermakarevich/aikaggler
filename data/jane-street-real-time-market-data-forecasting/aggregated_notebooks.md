# jane-street-real-time-market-data-forecasting: top public notebooks

The community's top-voted notebooks for the Jane Street Real-Time Market Forecasting competition focus on production-ready inference pipelines, GPU-accelerated gradient boosting baselines, and complex weighted ensembles blending tree-based models, neural networks, and linear/tabular transformers. Contributors heavily emphasize strict latency constraints, Polars for high-performance data handling, temporal lag alignment, and output clipping to [-5, 5] to stabilize financial forecasting predictions.

## Common purposes
- ensemble
- inference
- baseline
- feature_engineering

## Competition flows
- Initializes a Kaggle inference server with a dummy predict function returning zero-valued predictions
- Trains LightGBM, XGBoost, and CatBoost across 5 time-based folds with a custom weighted R2 metric, then averages predictions
- Predicts using a pre-trained XGBoost model and averaged 5-fold PyTorch neural network, combines with equal weighting, and clips outputs
- Loads pre-trained XGBoost, NN, TabM, and Ridge models, aligns test data with lagged features, combines via fixed weighted averaging
- Performs EDA on responders/features, loads pre-trained BayesianRidge, XGBoost, and NN models, ensembles with custom weights
- Engineers trigonometric time features, trains LightGBM, CatBoost, and XGBoost on GPU, ensembles with fixed weights
- Generates one-day lagged features for responder columns, splits data temporally, and exports as partitioned Parquet files
- Generates predictions from five XGBoost models, a 5-fold NN, and Ridge, blends with fixed weights, and clips output
- Loads pre-trained 5-fold NN checkpoints, processes data with forward-fill imputation and lag joins, averages outputs
- Processes test/lag data with cyclic time encoding, lag joining, standardization, and categorical mapping, applies normalized weighted averaging

## Data reading
- Loads test features and lags from parquet files via the local gateway or competition environment
- Loads train.parquet via pd.read_parquet, filters rows by date_id, and uses polars for inference data loading
- Loads parquet files using Polars, converts to pandas for model inference, and handles Polars DataFrames for the final submission format
- Uses Polars to scan and collect parquet files for validation/test data and lagged features; loads pre-trained model artifacts via pickle and dill
- Loads partitioned train.parquet files using pd.read_parquet and concatenates them; reads features.csv, responders.csv, and sample_submission.csv via pd.read_csv; uses polars for inference data handling
- Reads partitioned parquet files using Polars, converts to Pandas, and concatenates partitions 6-9; uses a custom Yunbase utility for memory reduction
- Uses polars.scan_parquet to load train.parquet, adds a sequential id column, and filters rows where date_id > 1100
- Reads validation and test data from parquet files using Polars; loads pre-trained model weights using pickle for XGBoost and dill for Ridge
- Uses polars.scan_parquet and .collect().to_pandas() to load validation, test, and lag parquet files
- Reads parquet files using Polars, converts to Pandas for model inference, and loads pre-trained models and dataset statistics via dill and pickle

## Data processing
- Custom memory downcasting to smallest safe dtypes
- Filtering training data by date_id ranges
- Forward-fill followed by zero-fill imputation
- Forward-fill followed by -1 imputation (test set only)
- Forward-fill followed by constant value (3) imputation
- Clipping final predictions to [-5, 5]
- Joining lagged features by date_id and symbol_id
- Standardizing continuous features with pre-computed global means/standard deviations
- Mapping categorical features to integer indices via explicit dictionaries
- Applying cyclic time encodings (sin/cos of time_id)
- Converting Polars DataFrames to PyTorch tensors
- Handling Polars/pandas type conversions for compatibility
- Reducing memory usage via custom utility functions

## Features engineering
- Lagged target features (responder_{idx}_lag_1 for idx 0-8)
- Trigonometric/cyclic time features (sin/cos of time_id and half-day variants)
- Raw features (feature_00 to feature_79)
- symbol_id and time_id as explicit features
- Global z-score standardization statistics
- Custom integer mappings for sparse categorical features
- Caching previous day's responders via global variables

## Models
- LightGBM
- XGBoost
- CatBoost
- PyTorch Lightning MLP
- TabM/FT-Transformer
- Ridge Regression
- BayesianRidge

## Frameworks used
- polars
- pandas
- kaggle_evaluation
- lightgbm
- xgboost
- catboost
- numpy
- joblib
- pytorch
- pytorch_lightning
- scikit-learn
- matplotlib
- seaborn
- dill

## Loss functions
- l2
- reg:squarederror
- RMSE
- Weighted Mean Squared Error (MSE)
- Custom R² loss
- F.mse_loss (weighted by sample weights)

## CV strategies
- Time-based split (skipping first 500 dates, using last 100 for validation, modulo indexing for 5 folds)
- 5-fold cross-validation (loading fold-specific checkpoints)
- Time Series Split (TTS) for Ridge validation

## Ensembling
- np.mean averaging across gradient boosting folds
- Equal-weight blending of NN and XGBoost
- Fixed-weight blending of NN/XGB/TabM/Ridge
- Linear combination of BayesianRidge with NN+XGB
- Fixed-weight blending of LightGBM/CatBoost/XGBoost
- Fixed-weight blending of XGBoost/NN/Ridge
- Averaging across 5-fold NN checkpoints
- Normalized weighted averaging across heterogeneous model groups
- All approaches followed by clipping predictions to [-5, 5]

## Insights
- The competition evaluation gateway passes data timestep-by-timestep, requiring the inference server to maintain state across calls.
- GPU acceleration is enabled for all three gradient boosting frameworks to speed up training.
- Forward-filling missing values before zero-filling stabilizes model inputs.
- Pre-computed standardization and explicit categorical mappings ensure consistent inference across heterogeneous model architectures.
- Leaving missing values unfilled in the training set while imputing -1 in the test set unexpectedly yielded a strong leaderboard score, suggesting a potential data distribution shift or leakage artifact.
- Temporal data splits should be based on date boundaries rather than random sampling to prevent data leakage.
- Combining heterogeneous model families with normalized weights improves prediction stability.

## Critical findings
- The official Kaggle inference server requires returning a Polars DataFrame rather than pandas, which caused initial compatibility issues.
- All responder values are strictly bounded within [-5, 5].
- Many features contain significant missing values.
- Real time intervals between time_id values are inconsistent.
- symbol_id is encrypted and not guaranteed to appear in all date/time combinations.

## Notable individual insights
- votes 3254 (Jane Street RMF Demo Submission): The competition evaluation gateway passes data timestep-by-timestep, requiring the inference server to maintain state across calls.
- votes 1618 (🥇🥇Jane Street Baseline lgb, xgb and catboost🥇🥇): GPU acceleration is enabled for all three gradient boosting frameworks to speed up training.
- votes 1222 (Jane Street RMF: Inference NN + XGB): Forward-filling missing values before zero-filling stabilizes model inputs.
- votes 1077 ([0.008] JS-RMF: Ensemble (XGB+NN+TabM+Ridge)): Pre-computed standardization and explicit categorical mappings ensure consistent inference across heterogeneous model architectures.
- votes 691 (JS2024 Starter): Leaving missing values unfilled in the training set while imputing -1 in the test set unexpectedly yielded a strong leaderboard score, suggesting a potential data distribution shift or leakage artifact.
- votes 691 (JS24: Preprocessing (Create Lags)): Temporal data splits should be based on date boundaries rather than random sampling to prevent data leakage.
- votes 509 (JS-2024 v15-04-01 RMF-NN+Starter+TabM+Ridge): Combining heterogeneous model families with normalized weights improves prediction stability.

## Notebooks indexed
- #3254 votes [[notebooks/votes_01_ryanholbrook-jane-street-rmf-demo-submission/notebook|Jane Street RMF Demo Submission]] ([kaggle](https://www.kaggle.com/code/ryanholbrook/jane-street-rmf-demo-submission))
- #1618 votes [[notebooks/votes_02_yuanzhezhou-jane-street-baseline-lgb-xgb-and-catboost/notebook|🥇🥇Jane Street Baseline lgb, xgb and catboost🥇🥇]] ([kaggle](https://www.kaggle.com/code/yuanzhezhou/jane-street-baseline-lgb-xgb-and-catboost))
- #1222 votes [[notebooks/votes_03_voix97-jane-street-rmf-inference-nn-xgb/notebook|Jane Street RMF: Inference NN + XGB]] ([kaggle](https://www.kaggle.com/code/voix97/jane-street-rmf-inference-nn-xgb))
- #1077 votes [[notebooks/votes_04_yongsukprasertsuk-0-008-js-rmf-ensemble-xgb-nn-tabm-ridge/notebook|[0.008] JS-RMF: Ensemble (XGB+NN+TabM+Ridge)]] ([kaggle](https://www.kaggle.com/code/yongsukprasertsuk/0-008-js-rmf-ensemble-xgb-nn-tabm-ridge))
- #891 votes [[notebooks/votes_05_allegich-jane-street-time-series-analysis-eda-ensemble/notebook|Jane Street: Time series analysis + EDA + Ensemble]] ([kaggle](https://www.kaggle.com/code/allegich/jane-street-time-series-analysis-eda-ensemble))
- #691 votes [[notebooks/votes_06_yunsuxiaozi-js2024-starter/notebook|JS2024 Starter]] ([kaggle](https://www.kaggle.com/code/yunsuxiaozi/js2024-starter))
- #691 votes [[notebooks/votes_07_motono0223-js24-preprocessing-create-lags/notebook|JS24: Preprocessing (Create Lags)]] ([kaggle](https://www.kaggle.com/code/motono0223/js24-preprocessing-create-lags))
- #573 votes [[notebooks/votes_08_hideyukizushi-js-nn-xgb-ridge-pub-mytrain-weightblend-lb-0-0079/notebook|JS|NN+XGB+Ridge(Pub+MyTrain)|WeightBlend|LB.0.0079]] ([kaggle](https://www.kaggle.com/code/hideyukizushi/js-nn-xgb-ridge-pub-mytrain-weightblend-lb-0-0079))
- #521 votes [[notebooks/votes_09_voix97-jane-street-rmf-training-nn/notebook|Jane Street RMF: Training NN]] ([kaggle](https://www.kaggle.com/code/voix97/jane-street-rmf-training-nn))
- #509 votes [[notebooks/votes_10_konstantinboyko-js-2024-v15-04-01-rmf-nn-starter-tabm-ridge/notebook|JS-2024 v15-04-01 RMF-NN+Starter+TabM+Ridge]] ([kaggle](https://www.kaggle.com/code/konstantinboyko/js-2024-v15-04-01-rmf-nn-starter-tabm-ridge))
