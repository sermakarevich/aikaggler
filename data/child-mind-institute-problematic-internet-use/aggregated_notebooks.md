# child-mind-institute-problematic-internet-use: top public notebooks

The community's top-voted notebooks primarily focus on establishing robust baselines and advanced ensembling strategies for ordinal regression, heavily leveraging gradient boosting models alongside TabNet. A consistent theme across submissions is the compression of high-dimensional actigraphy time-series data into tabular features using statistical summaries or custom PyTorch autoencoders. Additionally, several analyses critically examine the Quadratic Weighted Kappa metric, demonstrating how post-training threshold optimization and careful data cleaning significantly impact leaderboard performance.

## Common purposes
- baseline
- ensemble
- eda
- tutorial
- training

## Competition flows
- Loads CSV and parquet actigraphy data, compresses time-series via statistics and an autoencoder, engineers hand-crafted tabular features, trains a multi-model ensemble with threshold optimization, and generates a majority-voted submission file.
- Loads tabular and time-series data, extracts statistical and autoencoded features, trains multiple regressors with 5-fold stratified CV, optimizes prediction thresholds on OOF outputs, and combines three model variants via majority voting.
- Loads tabular and time-series data, extracts autoencoder-derived and hand-crafted features, trains LightGBM, XGBoost, CatBoost, and TabNet via 5-fold stratified CV, optimizes prediction thresholds for QWK, and generates a final submission via majority voting across three ensemble configurations.
- Loads tabular and time-series data, extracts descriptive statistics from the time series, trains a LightGBM regressor with 5-fold stratified CV, optimizes rounding thresholds on out-of-fold predictions, and generates a submission CSV.
- Loads train and test CSVs, validates and corrects the target variable for missing questionnaire data, cleans physical and demographic features, and analyzes distributions and correlations.
- Loads CSV and parquet time-series data, engineers statistical and autoencoder features, trains a multi-model ensemble with stratified 5-fold CV, optimizes prediction thresholds via out-of-fold validation, and generates a final submission via majority voting across three experimental setups.
- Loads tabular and time-series data, engineers interaction and autoencoder features, trains an ensemble of gradient boosting and TabNet models via 5-fold stratified CV, optimizes prediction thresholds for QWK, and generates a final submission via majority voting.
- Loads CSV and parquet time-series data, applies statistical or autoencoder-based feature engineering, trains gradient boosting and TabNet models via stratified k-fold cross-validation, optimizes prediction thresholds for QWK, and generates a final submission via majority voting.
- Loads tabular and actigraphy time-series data, imputes missing values, engineers interaction and autoencoder-derived features, trains gradient boosting and TabNet models via 5-fold stratified CV, optimizes prediction thresholds for QWK, and combines outputs via voting regressors and majority vote.
- Loads tabular and time-series data, extracts autoencoder embeddings for sequential features, merges them with raw tabular inputs, trains an ensemble of tree-based regressors via 5-fold stratified CV, optimizes prediction thresholds for quadratic weighted kappa, and combines multiple model outputs via voting and majority voting.

## Data reading
- pd.read_csv / polars.read_csv() for train, test, and sample_submission files
- pd.read_parquet / polars.read_parquet() for time-series parquet data
- pd.read_csv for data_dictionary CSV files
- ThreadPoolExecutor for parallelized parquet loading
- os.listdir for iterating time-series directories
- Extracting df.describe().values as statistical features
- Merging on 'id' column

## Data processing
- Filters rows with valid/missing target values
- Casts/fills categorical columns with 'Missing' and maps to integers/Enums
- Imputes numeric missing values using KNNImputer or SimpleImputer
- Replaces infinite values with NaN
- Scales features with StandardScaler
- Merges time-series statistics with tabular data
- Drops 'id' and 'Season' columns
- Rounds target variable to integers
- Recalculates SII and handles missing PCIAT answers
- Converts units (lbs to kg, inches to cm) and recalculates BMI
- Replaces biologically impossible values with NaN
- Bins age and CGAS scores
- Applies custom integer mappings for categorical features
- Continuous predictions rounded and mapped to discrete classes using optimized thresholds
- Autoencoder applied to time-series stats for dimensionality reduction
- Drops columns with excessive missing values
- Manual label encoding for categorical Season columns
- Drops 'step' column
- Encodes categorical variables

## Features engineering
- Statistical summaries/descriptive statistics from time-series parquet files
- Autoencoder embeddings for time-series compression (60 or 96 dimensions)
- Hand-crafted interaction and ratio features (e.g., BMI_Age, Internet_Hours_Age, BFP_BMI, Muscle_to_Fat, Hydration_Status)
- Custom integer mappings for categorical season columns
- Age group bins and ordinal internet use encoding
- CGAS score bins
- Recalculated BMI and complete response mask
- Unit conversions (lbs to kg, inches to cm)
- Weight-normalized metabolic rates and Fat/BMI ratios
- Internet Hours × SDS score and HeartRate × Age ratio
- Adding 1 to internet hours to prevent division by zero
- Direct inclusion of raw demographic, physical, fitness, BIA, and survey features
- Stat_0 to Stat_N columns from .describe().values

## Models
- LightGBM
- XGBoost
- CatBoost
- TabNet
- RandomForestRegressor
- GradientBoostingRegressor
- VotingRegressor

## Frameworks used
- pytorch
- keras
- lightgbm
- xgboost
- catboost
- scikit-learn
- polars
- pandas
- scipy
- numpy
- matplotlib
- seaborn
- plotly
- tqdm
- pytorch_tabnet
- torch

## Loss functions
- MSE
- nn.MSELoss
- Mean Squared Error (MSE)
- Quadratic Weighted Kappa
- Default regression loss

## CV strategies
- StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
- StratifiedKFold(n_splits=5, shuffle=False)
- StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

## Ensembling
- VotingRegressor with weighted/unweighted configurations
- Majority voting across three separate submission pipelines
- Averaging predictions across 5 CV folds
- Weighted averaging (e.g., 0.85/0.10/0.05) across prediction sets
- Post-processing with scipy-optimized thresholds (Nelder-Mead) to map continuous outputs to ordinal classes
- Optimizing decision thresholds on out-of-fold predictions to maximize QWK

## Insights
- Ordinal regression tasks benefit significantly from post-training threshold optimization rather than standard classification heads.
- Compressing high-dimensional actigraphy time-series via autoencoders or simple statistics effectively captures temporal patterns for tabular models.
- Majority voting across diverse model architectures stabilizes predictions and mitigates individual model bias.
- QWK can be counterintuitive, as predicting an incorrect class may sometimes yield a higher metric score than the true class for individual samples.
- Threshold optimization using Nelder-Mead on OOF predictions often converges to local maxima rather than global optima.
- Fitting autoencoders independently on train and test sets can cause distribution shifts that degrade feature consistency.
- Discretizing continuous predictions for QWK discards valuable ranking information within predicted classes.
- Autoencoder-derived features effectively capture time-series patterns for tabular models.
- Parallel processing with ThreadPoolExecutor significantly speeds up loading and processing multiple parquet files.
- Ordinal regression targets can be effectively handled by training a regressor and optimizing rounding thresholds on out-of-fold predictions.
- Consistent categorical encoding requires building mappings from the training set and applying them identically to the test set to prevent leakage or mismatched indices.
- Rigorous seed fixing across all relevant libraries is essential for reproducible Kaggle results and stable leaderboard scores.
- The QWK metric can be counterintuitive, sometimes rewarding incorrect predictions over correct ones in specific edge cases.
- Manual categorical mapping using train-set unique values prevents data leakage between training and test splits.
- Combining multiple model outputs via weighted averaging and majority voting stabilizes predictions and yields higher leaderboard scores than individual models.

## Critical findings
- The target distribution is highly imbalanced, with approximately 50% of samples in class 0 and very few in class 3.
- The PCIAT-PCIAT_Total column perfectly predicts whether the sii target is missing, allowing clean filtering of supervised data.
- Several features exhibit high missing ratios that require careful imputation strategies.
- When adding a single observation with true class 2, predicting class 3 can mathematically increase the QWK score more than predicting the correct class 2.
- The QWK metric treats all errors within the same predicted bin equally, masking performance differences between edge and center predictions.
- Separate autoencoder fitting on train/test splits creates incompatible latent spaces for identical data points.
- SII scores show a U-shaped relationship with age, peaking in adolescents.
- Internet usage time correlates linearly with age but has a complex, non-linear relationship with SII.
- Adolescents are the most affected demographic across all internet usage levels.
- The SII target variable likely reflects broader developmental/behavioral factors and parental bias rather than pure internet addiction.
- Physical measures like height and weight correlate with SII but primarily act as proxies for age.
- The original SII scores are sometimes incorrect when PCIAT questions are left unanswered, requiring recalculation or exclusion.
- A significant portion of physical measurements fall outside biologically normal ranges or contain clear data entry errors.
- BIA-derived metrics contain implausible negative values and extreme outliers, indicating potential processing or measurement errors.
- Participants with less than 1 hour of daily internet use still exhibit high SII scores, challenging the assumption that usage time directly drives impairment.
- Some features present in the training set are missing in the test set, requiring careful alignment before modeling.
- Missing target values perfectly align with missing values in the PCIAT-PCIAT_Total feature, indicating systematic data collection gaps.
- Predicting class 3 instead of the true class 2 can sometimes increase the QWK score for a single additional observation.
- The threshold optimizer using Nelder-Mead often converges to a local maximum rather than the global optimum.
- Training autoencoders independently on train and test data can produce severely different encodings for identical inputs.
- Supervised learning requires filtering out samples with missing target values before processing.

## What did not work
- Fitting autoencoders separately on train and test datasets causes distribution shifts, and that the Nelder-Mead threshold optimizer frequently gets stuck in local extrema, making both techniques unreliable despite yielding decent public leaderboard scores.
- Fitting autoencoders separately on train and test datasets leads to severely different encodings for the same data.
- Using scipy.optimize.minimize with Nelder-Mead for threshold optimization often converges to local extrema rather than the global maximum.

## Notable individual insights
- 975 (CMI| Tuning | Ensemble of solutions): QWK can be counterintuitive, as predicting an incorrect class may sometimes yield a higher metric score than the true class for individual samples.
- 934 (CMI-PIU: Features EDA): The SII target variable likely reflects broader developmental/behavioral factors and parental bias rather than pure internet addiction.
- 825 (CMI | Best Single Model): Consistent categorical encoding requires building mappings from the training set and applying them identically to the test set to prevent leakage or mismatched indices.
- 548 (CMI | Reproducible results |FixSeed,LGB-CPU|LB.492): Rigorous seed fixing across all relevant libraries is essential for reproducible Kaggle results and stable leaderboard scores.
- 490 (LB.497|Multi-Model Feature Importance Analysis): Fitting autoencoders separately on train and test sets can cause distribution shifts, yet such techniques may still produce competitive leaderboard scores.
- 393 (CMI: Issues with the Metric and Baseline): Ordinal regression requires careful threshold tuning rather than simple rounding of continuous outputs.
- 338 (LB: 0.482 Beginner Friendly Notebook): Manual categorical mapping using train-set unique values prevents data leakage between training and test splits.

## Notebooks indexed
- #1325 votes [[notebooks/votes_01_ichigoe-lb0-494-with-tabnet/notebook|LB0.494 with TabNet]] ([kaggle](https://www.kaggle.com/code/ichigoe/lb0-494-with-tabnet))
- #975 votes [[notebooks/votes_02_batprem-cmi-tuning-ensemble-of-solutions/notebook|CMI| Tuning | Ensemble of solutions]] ([kaggle](https://www.kaggle.com/code/batprem/cmi-tuning-ensemble-of-solutions))
- #934 votes [[notebooks/votes_03_cchangyyy-0-494-notebook/notebook|0.494 notebook]] ([kaggle](https://www.kaggle.com/code/cchangyyy/0-494-notebook))
- #825 votes [[notebooks/votes_04_abdmental01-cmi-best-single-model/notebook|CMI | Best Single Model]] ([kaggle](https://www.kaggle.com/code/abdmental01/cmi-best-single-model))
- #745 votes [[notebooks/votes_05_antoninadolgorukova-cmi-piu-features-eda/notebook|CMI-PIU: Features EDA]] ([kaggle](https://www.kaggle.com/code/antoninadolgorukova/cmi-piu-features-eda))
- #548 votes [[notebooks/votes_06_hideyukizushi-cmi-reproducible-results-fixseed-lgb-cpu-lb-492/notebook|CMI | Reproducible results |FixSeed,LGB-CPU|LB.492]] ([kaggle](https://www.kaggle.com/code/hideyukizushi/cmi-reproducible-results-fixseed-lgb-cpu-lb-492))
- #490 votes [[notebooks/votes_07_shodaifuruya-lb-497-multi-model-feature-importance-analysis/notebook|LB.497|Multi-Model Feature Importance Analysis]] ([kaggle](https://www.kaggle.com/code/shodaifuruya/lb-497-multi-model-feature-importance-analysis))
- #393 votes [[notebooks/votes_08_vitalykudelya-cmi-issues-with-the-metric-and-baseline/notebook|CMI: Issues with the Metric and Baseline]] ([kaggle](https://www.kaggle.com/code/vitalykudelya/cmi-issues-with-the-metric-and-baseline))
- #344 votes [[notebooks/votes_09_kuosys-cmi-reproducible-results-fixseed-lgb-cpu-lb-494/notebook|CMI | Reproducible results |FixSeed,LGB-CPU|LB.494]] ([kaggle](https://www.kaggle.com/code/kuosys/cmi-reproducible-results-fixseed-lgb-cpu-lb-494))
- #338 votes [[notebooks/votes_10_cchangyyy-lb-0-482-beginner-friendly-notebook/notebook|LB: 0.482 Beginner Friendly Notebook]] ([kaggle](https://www.kaggle.com/code/cchangyyy/lb-0-482-beginner-friendly-notebook))
