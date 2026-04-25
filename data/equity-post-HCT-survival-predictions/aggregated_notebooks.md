# equity-post-HCT-survival-predictions: top public notebooks

The community's top-voted notebooks primarily focus on adapting survival analysis techniques for tabular ML, emphasizing surrogate target generation, pairwise ranking losses, and rank-based ensembling to handle censored time-to-event data. Contributors extensively experiment with GPU-accelerated gradient boosting baselines, custom PyTorch neural networks with ODST layers, and classical survival estimators, while prioritizing demographic equity through race-stratified cross-validation and fairness penalties. A recurring theme is bridging non-differentiable clinical metrics with tractable regression proxies, leveraging HLA feature engineering, and stabilizing predictions through weighted rank aggregation across heterogeneous model families.

## Common purposes
- ensemble
- baseline
- training
- eda
- tutorial

## Competition flows
- GBDT baseline with GPU acceleration, target transformation, and rank-sum ensembling
- PyTorch Lightning NN training with custom pairwise ranking loss and stratified CV
- Surrogate target generation via OOF survival models, followed by LightGBM/CatBoost training and weighted rank-based ensembling
- Hybrid pipeline combining PyTorch pairwise NN, XGBoost event classifier, and tree models with event-based offsets and weighted averaging
- XGBoost event mask generation followed by NN training with constant offset application for masked subjects
- Multi-stage pipeline engineering survival pseudo-targets and HLA aggregations, training stratified tree/NN models, and ensembling
- EDA-focused pipeline with survival analysis, multiple target transformations, six model variants, and race-group-aware metric evaluation
- Polars-based data loading, HLA sum recalculation, surrogate target generation, and weighted rank aggregation ensembling
- Logit-transformed targets, pairwise ranking loss training, and rank-based weighting across model families
- Censored target normalization, GPU-accelerated GBDT training, concordance index evaluation, and simple prediction averaging

## Data reading
- pd.read_csv for train and test datasets
- pandas.read_csv from Kaggle input paths
- polars.read_csv with batch processing (converted to pandas)
- pd.read_csv and pl.read_csv from specific Kaggle input paths
- pd.read_csv for train, test, and data_dictionary with index_col='ID'
- pd.read_csv() and pl.read_csv() with batch processing and null handling

## Data processing
- Label encoding (factorize, LabelEncoder) and one-hot encoding of categorical features
- Filling categorical NaNs with 'NAN', 'Unknown', or 'N/A'
- Reducing numerical precision to float32/int32
- Leaving numerical NaNs intact for GBDT native handling
- Imputing missing numerical values with mean/median using SimpleImputer (with/without add_indicator)
- Applying StandardScaler to numerical features
- Log-transforming or logit-transforming efs_time
- Replacing specific outlier values (e.g., year_hct, karnofsky_score)
- Casting datatypes to Float32/String/Int32 and converting categoricals to pandas category dtype
- Dropping constant columns dynamically during survival model fitting
- Handling unseen validation/test categories by setting to NaN
- Wrapping data in PyTorch TensorDataset/DataLoader with batch processing
- Race-based weight scaling and target transformations (KM, partial hazard, rank, quantile)

## Features engineering
- Label encoding and one-hot encoding of categorical features
- Precision reduction of numerical features to float32/int32
- Target engineering via Kaplan-Meier survival function, Cox PH partial hazard, Nelson-Aalen cumulative hazard, and flipped censored times
- Log-transform and logit-transform of efs_time
- Race-group stratification indicators and race-weighted scaling
- HLA locus matching aggregations (summing low/high resolution matches for 6/8/10 loci) and recalculation of missing HLA sums from raw alleles
- Relative year calculation (year_hct - 2000)
- Cross-feature interactions (donor-recipient age difference, comorbidity/karnofsky score sum/diff/product/div)
- Binary flags (is_cyto_score_same, binary NA indicators, combined binary categorical features)
- Age bins and donor-recipient age difference
- Replacing specific outlier values as feature corrections

## Models
- XGBoost (XGBRegressor, XGBClassifier, Cox, AFT, MSE variants)
- CatBoost (CatBoostRegressor, Cox, AFT variants)
- LightGBM (LGBMRegressor)
- Custom PyTorch Lightning Neural Networks (LitNN) with ODST layers, categorical embeddings, and MLP heads
- Classical survival models (CoxPHFitter, KaplanMeierFitter, NelsonAalenFitter from lifelines)

## Frameworks used
- xgboost, catboost, lightgbm
- pytorch, pytorch_lightning, pytorch_tabular, torch
- lifelines
- pandas, polars, numpy, scipy, matplotlib, plotly
- scikit-learn

## Loss functions
- Regression losses: MSE, RMSE, MAE, reg:logistic
- Survival losses: Cox proportional hazards (CoxPH), Cox partial likelihood, survival:aft (Accelerated Failure Time)
- Pairwise ranking losses: Margin-based hinge loss (with event masking)
- Auxiliary/Regularization losses: MSE (auxiliary task), Race-group variance penalty/loss
- Classification losses: Binary cross-entropy (for XGBoost event mask)

## CV strategies
- KFold(n_splits=5 or 10, shuffle=True, random_state=42)
- StratifiedKFold(n_splits=5) stratified by race_group
- StratifiedKFold(n_splits=5) stratified by race_group and newborn status
- StratifiedKFold(n_splits=5) stratified by race_group and age_at_hct==0.044
- StratifiedKFold(n_splits=5) stratified by race_group and newborn indicator
- KFold(n_splits=5) used specifically for survival target generation

## Ensembling
- Rank-based ensembling (converting predictions to ranks via scipy.stats.rankdata and summing/averaging)
- Weighted rank aggregation (multiplying ranks by predefined weights before summation)
- Simple averaging across folds and models
- Weighted averaging across folds and model families
- Fixed offset application (e.g., +0.2 to NN predictions where an XGBoost event mask predicts an event)
- Negation of final aggregated scores before submission
- Rank averaging of multiple submission files

## Insights
- Ranking and summing predictions from heterogeneous models improves cross-validation scores for ranking-based competition metrics.
- Adding a race-group variance penalty helps mitigate demographic bias by encouraging consistent model performance across subgroups.
- Surrogate regression targets generated via out-of-fold survival models effectively bypass the non-differentiable C-Index metric.
- Censored observations cannot be reliably ranked against events, necessitating a strict event mask in the pairwise loss.
- Patient age distribution shows five unnatural modes, with 16 days occurring over 1000 times, indicating synthetic data generation artifacts.
- Recalculating missing HLA sum features from raw allele match columns recovers valuable genetic compatibility information that was originally incomplete.
- Leaving numerical NaNs as-is and using native categorical support in GBDT frameworks avoids unnecessary imputation or encoding overhead.

## Critical findings
- Constant columns in survival model folds cause convergence errors due to NaN values in the delta matrix and must be dropped.
- Target leakage is strictly prevented by computing surrogate targets exclusively from out-of-fold survival model predictions.
- Censored observations cannot be reliably ranked against events, necessitating a strict event mask in the pairwise loss.
- Specific HLA and demographic columns contain rare outlier values that must be manually corrected to prevent model confusion.
- Race-group stratification is critical for both target engineering and CV stability due to distribution shifts.
- Replacing specific outlier values (e.g., karnofsky_score=40 to 50, hla_high_res_8=2 to 3) prevents model confusion from rare clinical entries.
- Applying race-specific weights to target scaling accounts for inherent demographic survival disparities in the dataset.
- Censored observations must be explicitly masked in pairwise comparisons to avoid invalid ranking signals and data leakage.
- Constant columns in a CV fold can cause lifelines convergence errors, so they must be dynamically dropped before fitting survival models.
- The competition metric evaluates prediction order rather than magnitude, making rank aggregation a natural and effective ensembling strategy.
- CoxPHFitter fails to converge if constant columns are present in a CV fold, requiring dynamic column dropping.
- Censored observations must be explicitly masked in pairwise ranking to avoid invalid comparisons.
- Race-based target scaling and weight adjustments are necessary to account for demographic disparities in survival outcomes.

## What did not work
- The linear Cox model underperformed relative to gradient boosting approaches, and the author notes that the accelerated failure time models likely require more extensive hyperparameter tuning to reach their potential.

## Notable individual insights
- votes 1199 (GPU LightGBM Baseline): Ranking and summing predictions from heterogeneous models improves cross-validation scores for ranking-based competition metrics.
- votes 898 (Refactoring NN Pairwise Ranking Loss): Adding a race-group variance penalty helps mitigate demographic bias by encouraging consistent model performance across subgroups.
- votes 882 (CIBMTR | EDA & Ensemble Model): Surrogate regression targets generated via out-of-fold survival models effectively bypass the non-differentiable C-Index metric.
- votes 858 ([0.693] CIBMTR | Weights Ranking Ensemble): Censored observations cannot be reliably ranked against events, necessitating a strict event mask in the pairwise loss.
- votes 418 (ESP EDA which makes sense): Patient age distribution shows five unnatural modes, with 16 days occurring over 1000 times, indicating synthetic data generation artifacts.
- votes 413 (CIBMTR | EDA & Ensemble Model - Recalculate HLA): Recalculating missing HLA sum features from raw allele match columns recovers valuable genetic compatibility information that was originally incomplete.
- votes 376 (XGBoost CatBoost Baseline): Leaving numerical NaNs as-is and using native categorical support in GBDT frameworks avoids unnecessary imputation or encoding overhead.

## Notebooks indexed
- #1199 votes [[notebooks/votes_01_cdeotte-gpu-lightgbm-baseline-cv-681-lb-685/notebook|GPU LightGBM  Baseline - [CV 681 LB 685]]] ([kaggle](https://www.kaggle.com/code/cdeotte/gpu-lightgbm-baseline-cv-681-lb-685))
- #898 votes [[notebooks/votes_02_albansteff-refactoring-nn-pairwise-ranking-loss/notebook|Refactoring NN Pairwise Ranking Loss]] ([kaggle](https://www.kaggle.com/code/albansteff/refactoring-nn-pairwise-ranking-loss))
- #882 votes [[notebooks/votes_03_andreasbis-cibmtr-eda-ensemble-model/notebook|CIBMTR | EDA & Ensemble Model]] ([kaggle](https://www.kaggle.com/code/andreasbis/cibmtr-eda-ensemble-model))
- #858 votes [[notebooks/votes_04_yongsukprasertsuk-0-693-cibmtr-weights-ranking-ensemble/notebook|[0.693] CIBMTR | Weights Ranking Ensemble]] ([kaggle](https://www.kaggle.com/code/yongsukprasertsuk/0-693-cibmtr-weights-ranking-ensemble))
- #535 votes [[notebooks/votes_05_albansteff-event-masked-prl-nn/notebook|Event-masked PRL-NN]] ([kaggle](https://www.kaggle.com/code/albansteff/event-masked-prl-nn))
- #467 votes [[notebooks/votes_06_gogo4974-cibmtr-ensemble/notebook|CIBMTR Ensemble]] ([kaggle](https://www.kaggle.com/code/gogo4974/cibmtr-ensemble))
- #418 votes [[notebooks/votes_07_ambrosm-esp-eda-which-makes-sense/notebook|ESP EDA which makes sense ⭐️⭐️⭐️⭐️⭐️]] ([kaggle](https://www.kaggle.com/code/ambrosm/esp-eda-which-makes-sense))
- #413 votes [[notebooks/votes_08_albansteff-cibmtr-eda-ensemble-model-recalculate-hla/notebook|CIBMTR | EDA & Ensemble Model - Recalculate HLA]] ([kaggle](https://www.kaggle.com/code/albansteff/cibmtr-eda-ensemble-model-recalculate-hla))
- #401 votes [[notebooks/votes_09_bestwater-integrating-open-source-models-try-more-good-luck/notebook|Integrating Open-Source Models:Try More,Good Luck!]] ([kaggle](https://www.kaggle.com/code/bestwater/integrating-open-source-models-try-more-good-luck))
- #376 votes [[notebooks/votes_10_cdeotte-xgboost-catboost-baseline-cv-668-lb-668/notebook|XGBoost CatBoost  Baseline - [CV 668 LB 668]]] ([kaggle](https://www.kaggle.com/code/cdeotte/xgboost-catboost-baseline-cv-668-lb-668))
