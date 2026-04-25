# child-mind-institute-problematic-internet-use: cross-solution summary

This competition focused on predicting health-related outcomes from complex, noisy sensor and demographic data, with winning approaches prioritizing robust validation and careful target handling over complex modeling. Top solutions consistently leveraged tree-based ensembles (LightGBM, XGBoost, CatBoost) alongside rigorous cross-validation strategies and explicit, domain-aware missing value imputation. Success hinged on stabilizing the CV-to-leaderboard relationship through techniques like repeated stratified folds, multi-seed averaging, and direct training-data threshold tuning.

## Competition flows
- Pipeline starting with Lasso imputation, PCA, quantile binning, manual feature reduction, repeated stratified tuning, and voted tree ensemble with target conversion
- Pipeline featuring parquet-based feature engineering, IterativeImputer, custom QWK objective, 10x10 nested CV, and single threshold optimization
- Polars-based data manipulation, domain-aware column-specific imputation, and direct CatBoost training
- Polars processing, per-fold median imputation, Tweedie loss with pseudo-labels, Nelder-Mead threshold optimization, and multi-fold/multi-seed voting ensemble
- Minimalist pipeline splitting data into descriptive feature sets, training stratified LightGBM regressors with training-data threshold tuning, and averaging across 100+ random states

## Data reading
- parquet files
- Polars DataFrame API

## Data processing
- Removed implausible values (e.g., body fat >60%, negative bone mineral content) and replaced with NaN
- Applied quantile binning to handle feature noise
- Retained PCA components for actigraph data
- Imputed missing values using Lasso with mean fallback
- Converted target to 'PCIAT-PCIAT_Total' score
- Imputed missing values with IterativeImputer
- Applied column-specific imputation (mean for numeric/float, max+1/Null for integer/categorical)
- Applied median imputation per fold
- Used Polars for optimized sequential data statistics
- Utilized integer-defined categorical features as both categorical and numerical inputs
- Cleaned noisy domain values (e.g., blood pressure, BMI) based on domain knowledge
- Excluded unknown target values from training without imputation
- Applied descriptive statistics to public features
- Discarded actigraphy-derived features due to marginal impact

## Features engineering
- Created descriptive actigraph features with separate day and night masks
- Added normalized values based on age group means
- Computed difference between daily energy expenditure and basal metabolic rate
- Reduced dataset via manual feature selection
- Engineered features directly from parquet files
- Applied descriptive statistics to public features
- Implemented rule-based allocation mapping predicted totals to target ranges
- Created derived targets (e.g., PCIAT-PCIAT1-20)

## Models
- LightGBM
- XGBoost
- CatBoost
- ExtraTreesRegressor

## Frameworks used
- polars
- catboost
- lightgbm
- xgboost
- optuna
- pandas
- scipy

## Loss functions
- Tweedie loss
- Custom QWK objective

## CV strategies
- 10-fold stratified KFold stratified by equidistant bins
- Repeated Stratified KFold (10-20 repeats) for parameter tuning
- 10x10 nested cross-validation using StratifiedKFold
- Increased folds to 20 for internal validation stability
- 5-fold StratifiedKFold with 10-seed averaging per experiment
- 5-fold stratified cross-validation by target values with threshold tuning on fold-level training data

## Ensembling
- Voted ensemble of multiple tree-based regressors with target label conversion
- Threshold optimization on aggregated CV predictions without model ensembling
- Direct submission without ensembling or post-processing
- Voting ensemble across multiple folds and seeds with percentile-based threshold optimization
- Model averaging across 100+ random states combined into a simple average blend of feature set submissions

## Insights
- Switching the target to 'PCIAT-PCIAT_Total' and converting predictions back improved overall robustness.
- Quantile binning effectively mitigated noise across a large portion of the feature set.
- Repeated Stratified KFold during parameter tuning yielded significantly more stable results than standard CV setups.
- Sample weights aligned unoptimized scores with optimized scores, even if they did not directly boost the primary metric.
- Missing value imputation was critical for achieving strong leaderboard performance.
- The custom QWK objective meaningfully improved cross-validation scores but did not translate to leaderboard gains.
- A 10x10 nested CV scheme provided stable validation scores and robust test predictions.
- Explicit, domain-aware imputation can outperform a model's built-in missing value handling by preserving feature semantics.
- Increasing cross-validation folds can improve internal validation stability without guaranteeing public leaderboard gains.
- Threshold optimization using percentiles and Nelder-Mead significantly improved accuracy and robustness.
- Seed variance can drastically impact scores, making multi-seed averaging and per-seed Optuna tuning essential for reliable validation.
- Generating pseudo-labels for samples with missing 'sii' targets effectively expands the training data.
- Score variance across different random seeds can be substantial, rendering single-seed validation unreliable.
- Features explicitly defined as categorical integers in the data dictionary yield better performance when utilized simultaneously as both categorical and numerical inputs.
- Tuning prediction thresholds directly on the training data instead of OOF predictions significantly improved CV-LB stability.
- Prioritizing model stability over complexity prevented downward movement in a highly churned competition.
- Averaging models across 100+ random states effectively reduced sensitivity to random seed variations.

## Critical findings
- The target's excess zeros necessitated exploring Tweedie loss and equidistant bin weights rather than standard regression objectives.
- Standard parameter tuning with regular cross-validation produced unstable outcomes, requiring repeated stratified folds for reliability.
- Weighting failed to improve optimized scores directly but successfully bridged the gap between unoptimized and optimized metrics.
- The custom QWK objective and metric only contributed to the CV score, not the leaderboard, despite being a core pipeline component.
- Missing value imputation had a disproportionately large impact on both CV and LB scores compared to other feature engineering steps.
- Improving the internal CV score by increasing folds to 20 initially resulted in a lower public leaderboard score, indicating a potential public/private data distribution shift.
- Hyperparameter tuning consistently degraded the CV-LB relationship and caused models to tank on the private leaderboard.
- Using two different target strategies (direct `sii` prediction vs. `PCIAT`-based rule allocation) improved ensemble stability between CV and public LB scores.

## What did not work
- The custom QWK objective and metric only improved the CV score and did not contribute to leaderboard performance.
- Extensive feature engineering using actigraphy data yielded no impact or only marginal impact.
- Hyperparameter tuning consistently led to an undesirable CV-LB relationship and poor private leaderboard performance.

## Notable individual insights
- rank 1 (First Place Write-Up: Or How I Won the Lottery): Switching the target to 'PCIAT-PCIAT_Total' and converting predictions back improved overall robustness.
- rank 1 (First Place Write-Up: Or How I Won the Lottery): Repeated Stratified KFold during parameter tuning yielded significantly more stable results than standard CV setups.
- rank 16 (16th Place Solution): Missing value imputation had a disproportionately large impact on both CV and LB scores compared to other feature engineering steps.
- rank 2 (2nd Place Writeup): Explicit, domain-aware imputation can outperform a model's built-in missing value handling by preserving feature semantics.
- rank 7 (Private 7th place solution): Features explicitly defined as categorical integers in the data dictionary yield better performance when utilized simultaneously as both categorical and numerical inputs.
- rank 106 (Rank 106 approach - Simple feature sets and CV-LB stability focus): Tuning prediction thresholds directly on the training data instead of OOF predictions significantly improved CV-LB stability.

## Solutions indexed
- #1 [[solutions/rank_01/solution|First Place Write-Up: Or How I Won the Lottery]]
- #2 [[solutions/rank_02/solution|2nd Place Writeup]]
- #7 [[solutions/rank_07/solution|Private 7th place solution]]
- #16 [[solutions/rank_16/solution|16th Place Solution]]
- #106 [[solutions/rank_106/solution|Rank 106 approach - Simple feature sets and CV-LB stability focus]]
