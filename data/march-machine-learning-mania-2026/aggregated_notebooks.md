# march-machine-learning-mania-2026: top public notebooks

The community's top-voted notebooks primarily focus on building robust baselines and advanced ensembles for NCAA tournament prediction, heavily leveraging Elo ratings, GLM-derived team quality, and point differential regression. Authors consistently emphasize temporal validation strategies, symmetric feature engineering, and probability calibration to prevent data leakage and overconfidence. While some notebooks demonstrate modular agentic pipelines or specialized stacking architectures, others serve as cautionary tales about public leaderboard evaluation flaws.

## Common purposes
- baseline
- other
- tutorial
- feature_engineering
- training
- ensemble

## Competition flows
- Loads historical March Madness CSV files, computes Elo ratings and parses team seeds, trains a logistic regression model on Elo and seed differences, and outputs a clipped probability submission CSV.
- Loads compact match results and team IDs for men's and women's tournaments, computes dynamic Elo ratings with configurable weights and margin-of-victory scaling, aggregates them into per-team seasonal statistics, and converts the final ratings into pairwise win probabilities for a direct submission.
- Loads historical CSV data, engineers team and pairwise features, trains a LightGBM classifier with temporal cross-validation, averages fold predictions, and exports a CSV submission.
- Loads men's and women's regular season and tournament CSVs, filters by season, engineers seed, Elo, and GLM-based team quality features, trains a leave-one-season-out XGBoost regressor on point differential, converts outputs to win probabilities via spline, and generates a tournament bracket submission.
- Loads historical NCAA basketball results and seeds, engineers seasonal averages, custom ELO, and GLM-derived team quality features, trains leave-one-season-out XGBoost models to predict point differentials, calibrates outputs to win probabilities via splines, and generates a tournament submission.
- Loads NCAA basketball CSVs, computes Elo and GLM-based team quality metrics, trains leave-one-season-out XGBoost models on point differentials, calibrates predictions with splines, and averages fold predictions to generate a submission.
- Loads historical NCAA game results, seeds, and ranking systems to compute team-season and matchup-level features, trains three specialized gradient boosting models with temporal walk-forward validation, blends their predictions via a logistic regression meta-learner, and outputs a calibrated submission file.
- Loads historical matchup results and the sample submission, merges them by season and team IDs to extract known outcomes, fills missing values with 0.5, and saves the resulting predictions as a submission file.

## Data reading
- Reads multiple CSV files from the competition input directory using pandas, including team rosters, regular season results, tournament results, seeds, and the sample submission template.
- Reads CSV files directly from Kaggle input paths using pd.read_csv(), including regular season results, tournament results, team rosters, and the sample submission template.
- Reads men's and women's tournament seeds, regular season results, and tournament compact results via pd.read_csv, concatenates them, and parses team IDs and seasons from string columns.
- Reads multiple CSV files (MRegularSeasonDetailedResults.csv, MNCAATourneyDetailedResults.csv, MNCAATourneySeeds.csv, and corresponding women's datasets) from a local Kaggle input directory using pandas.read_csv, then concatenates men's and women's data.
- Reads multiple CSV files for men's and women's regular season results, tournament results, and tournament seeds using pandas.
- Reads multiple CSVs (MRegularSeasonDetailedResults.csv, MNCAATourneyDetailedResults.csv, MNCAATourneySeeds.csv, etc.) using pd.read_csv.
- Reads CSV files for teams, seeds, compact/detailed regular season and tournament results, and Massey ordinal rankings using pandas.read_csv. Dynamically loads supplementary Massey system files via glob and concatenates them. Parses alphanumeric seed strings into numeric values using regex.
- Reads multiple CSV files via pandas (e.g., compact/detailed results, seeds, conferences, coaches, sample submissions). Parses seed strings to integers via slicing (str[1:3]). Loads temporal coverage and row counts for inventory.
- Reads SampleSubmissionStage2.csv and multiple historical compact result CSVs using pandas.read_csv. Parses submission IDs by splitting strings and extracting integer columns for Season, Team1, and Team2.

## Data processing
- Parses seed strings to numeric values; defaults missing Elo ratings to 1500; clips final prediction probabilities to [0.01, 0.99] to prevent overconfidence.
- Concatenates regular season and tournament data, adds a tourney flag (0/1) and a weight column (1.0 for regular season, 0.75/0.95 for tournament depending on gender), sorts by Season and DayNum, resets index, and filters out tournament rows before aggregating ratings.
- Drops irrelevant columns (NumOT, WLoc), fills missing feature values with training medians, applies min-max rescaling computed on training data, and defaults missing tournament seeds to W01.
- Filters data to seasons >= 2003, adjusts box score statistics for overtime duration using a linear scaling factor, doubles the dataset by swapping home/away team positions to balance features, calculates point differential and binary win labels, and flags men's/women's games.
- Adjusts box score statistics for overtime duration by dividing by a time-scaling factor of (40 + 5 * NumOT) / 40. Doubles the dataset by swapping team positions to ensure consistent feature orientation. Calculates point differentials, binary win labels, and a men/women flag. Clips predicted point differentials to [-30, 30] during calibration and submission.
- Fills numeric feature NaNs with column medians. Clips estimated possessions to a minimum of 1 to prevent division by zero. Handles missing Massey systems gracefully by checking existence before loading. Clips final submission probabilities to [0.015, 0.985] to avoid extreme predictions. Uses joblib.Parallel and ThreadPoolExecutor for parallel feature construction and concurrent model inference.
- Normalizes historical results by sorting team IDs and extracting the winner/loser relationship. Groups by season and team pair to keep the most recent outcome, merges with the submission dataframe, and fills NaN values with 0.5.

## Features engineering
- Computes Elo ratings with K=20, home-court advantage (+100/-100), and seasonal regression (0.75*prev + 0.25*init); extracts numeric seeds; creates `elo_diff` and `seed_diff` features using a consistent Team1-Team2 convention.
- Computes Elo ratings with configurable K-factor, width, and margin-of-victory multiplier (alpha). Aggregates ratings into per-team per-season statistics: Rating_Mean, Rating_Median, Rating_Std, Rating_Min, Rating_Max, Rating_Last, and Rating_Trend (linear regression slope over time).
- Computes per-team season win counts, loss counts, and average score gaps, derives win ratios and weighted score gaps, creates symmetric match features by duplicating and swapping team roles, and calculates pairwise differences for seeds, win ratios, and score gaps.
- Seed difference, regular season average box scores for both teams and their opponents, custom Elo ratings updated iteratively per season, and GLM-derived team quality scores using statsmodels with Gaussian family; all features are aggregated per season and merged into the tournament dataset.
- Seasonal mean aggregations of box score stats (FGM, FGA, FGM3, etc.) for both teams and their opponents. Custom ELO ratings calculated per season with base 1000, width 400, and k=100. GLM-derived team quality scores using statsmodels Gaussian family to capture mixed-effects team strength. Seed differentials and overtime-adjusted point differentials.
- Team-season level stats: win rate, average scoring margin, average points for/against. Dean Oliver's Four Factors + efficiency: possessions (0.44 multiplier), ORTG, DRTG, NetRtg, eFG%, TO%, ORB%, FTR, ThreePAr, ASTpct. Strength of Schedule: average opponent win rate over the regular season. Rolling form: win rate and average margin over the final 10 regular-season games. Massey ordinal aggregations: mean, median, std, best, worst rank per team-season, plus pivoted individual system ranks. Game-by-game Elo ratings with home-court adjustment, log-damped margin-of-victory multiplier, and 75/25 regression to the mean between seasons. Historical seed upset probability lookup table (excluding matchups with <5 observed seasons). Interaction features: SeedDiff × EloDiff and SeedDiff².
- Season averages for box score stats (FGM, FGA, OR, DR, Blk, PF, etc.) for both teams and opponents. Elo ratings calculated per season using standard win/loss update rules. GLM-based team quality scores using fixed-effects regression on point differentials. Differences in seeds, Elo, and quality; men/women indicator.

## Models
- LogisticRegression
- LightGBM
- XGBoost
- GLM (statsmodels Gaussian family)
- HistGradientBoostingClassifier
- XGBClassifier
- CatBoostClassifier

## Frameworks used
- google-adk
- scikit-learn
- pandas
- numpy
- matplotlib
- scipy
- tqdm
- lightgbm
- seaborn
- xgboost
- statsmodels
- catboost
- optuna
- joblib

## Loss functions
- binary_logloss
- reg:squarederror
- Brier Score
- Log Loss

## CV strategies
- 5-fold cross-validation with neg_brier_score on the tournament training set.
- Temporal validation: trains on all prior seasons and validates on each subsequent season sequentially, iterating through recent years.
- Leave-one-season-out cross-validation (trains on all seasons except one, validates on the held-out season, iterates over all unique seasons).
- Temporal walk-forward cross-validation (leave-one-season-out) from 2018 to 2025, training on all prior seasons and validating on the current season to prevent future data leakage.

## Ensembling
- Averages the test predictions from each temporal fold using np.mean before generating the final submission.
- Trains a separate XGBoost model for each season during validation, then averages the spline-calibrated probability predictions across all season models to produce the final submission.
- Averages predictions across leave-one-season-out models and applies post-processing calibration using a UnivariateSpline to map point differentials to win probabilities, with probability clipping between 0.02 and 0.98.
- Averages predictions from leave-one-season-out models and applies manual alpha blending for extreme probabilities (>0.9 or <0.1).
- Base model predictions are converted to log-odds and stacked with SeedDiff into a logistic regression meta-learner, followed by isotonic regression calibration on out-of-fold predictions to correct systematic over/under-confidence.

## Insights
- Multi-agent orchestration via ADK provides a modular, executable template for structuring complex ML pipelines.
- Elo ratings combined with seed differences form a highly interpretable and effective baseline for tournament forecasting.
- Cross-validation using the Brier score is the appropriate metric for evaluating probabilistic tournament predictions.
- Clipping predicted probabilities prevents extreme model outputs that could negatively impact leaderboard scores.
- Elo ratings can be effectively adapted for sports prediction by tuning the K-factor, rating width, and match weights.
- Margin of victory can be incorporated into rating updates via a tunable alpha parameter to control score sensitivity.
- Aggregating temporal ratings into summary statistics captures team form, volatility, and directional momentum.
- Elo-derived probabilities serve as a strong, interpretable baseline feature for tournament prediction.
- Symmetric feature engineering ensures the model learns consistent pairwise relationships regardless of which team wins.
- Temporal cross-validation is essential to prevent data leakage in sports prediction tasks.
- Binary classification outperformed regression for predicting score gaps during validation.
- Statistically derived team quality consistently matches or outperforms traditional seeding in predicting tournament outcomes.
- Direct regression on point differential followed by spline calibration produces better-calibrated win probabilities than direct classification.
- Overtime adjustments are necessary to prevent stat inflation when comparing regular season performance.
- GLM-derived team quality metrics often match or outperform official tournament seeds in predictive power.
- Overtime significantly inflates box score statistics, necessitating time-based normalization for fair comparison.
- Swapping team positions effectively doubles the training data and ensures consistent feature orientation for both home and away teams.
- Raw overtime statistics accumulate disproportionately, necessitating a specific adjustment factor.
- Seed difference remains a surprisingly strong baseline, but GLM quality surpasses it in most years.
- Strict feature partitioning across tree-based models significantly reduces inter-model correlation, enabling effective stacking.
- Converting probabilities to log-odds before meta-learning creates a more linear input space for the logistic regression.
- Temporal walk-forward validation is essential for tournament prediction to avoid inflated scores from future data leakage.
- Isotonic calibration on OOF predictions reliably tightens probability distributions without overfitting.
- Standard stratified k-fold validation artificially inflates CV scores in tournament prediction by mixing future games into training folds.
- Historical seed matchup data with fewer than 5 observed seasons introduces noise and is explicitly excluded from the lookup table.
- The meta-learner learns context-aware blending, heavily weighting XGBoost for Women's games while balancing CatBoost and HistGB for Men's chaotic matchups.
- Seed difference is the strongest predictor of tournament outcomes.
- Home court advantage remains stable around 58–60% across decades.
- Raw scoring trends fluctuate, making efficiency-adjusted metrics more reliable than point totals.
- Conference strength and schedule difficulty help distinguish record inflation from true team quality.
- Late-season momentum and coach experience provide marginal but useful contextual signals.
- Stage 1 leaderboard scores are trivially achievable via data leakage since historical results are included in the dataset.
- The women’s tournament is significantly more 'chalk' (higher seeds win more often) than the men’s tournament.
- Upsets are rare when seed gaps exceed 10 lines but become common in similarly seeded matchups.
- The public leaderboard evaluation set in this competition overlaps almost entirely with historical matchups that have known outcomes.
- A perfect public score can be achieved without training any predictive model by simply looking up past results.
- Leaderboard scores in this competition should not be trusted as a measure of true model performance or generalization.

## Critical findings
- GLM team quality shows higher or equal AUC compared to seed difference across all seasons, indicating that statistical performance metrics are more reliable than tournament seeding for predicting upsets.
- The author explicitly compares seed-based vs quality-based predictive power per season, noting that statistical quality often matches or beats expert seeding.
- The notebook highlights that overtime games accumulate extra stats, which must be normalized to avoid biasing team strength estimates.
- Raw overtime statistics accumulate disproportionately, necessitating a specific adjustment factor.
- Seed difference remains a surprisingly strong baseline, but GLM quality surpasses it in most years.
- Several box score metrics were commented out, suggesting they may not add predictive value or were omitted for brevity.
- Standard stratified k-fold validation artificially inflates CV scores in tournament prediction by mixing future games into training folds.
- Historical seed matchup data with fewer than 5 observed seasons introduces noise and is explicitly excluded from the lookup table.
- The meta-learner learns context-aware blending, heavily weighting XGBoost for Women's games while balancing CatBoost and HistGB for Men's chaotic matchups.
- Stage 1 leaderboard scores are trivially achievable via data leakage since historical results are included in the dataset.
- The women’s tournament is significantly more 'chalk' (higher seeds win more often) than the men’s tournament.
- Upsets are rare when seed gaps exceed 10 lines but become common in similarly seeded matchups.
- The public evaluation uses approximately 1% of the test data, which currently overlaps almost entirely with historical matchups that have known outcomes.

## What did not work
- 

## Notable individual insights
- votes 763 (🏀 March Machine Learning Mania 2026 Starter): Multi-agent orchestration via ADK provides a modular, executable template for structuring complex ML pipelines.
- votes 309 (Calculate ELO-Ratings🏀): Margin of victory can be effectively incorporated into Elo rating updates via a tunable alpha parameter to control score sensitivity.
- votes 261 (NCAA 2026 |  Final - LightGBM): Symmetric feature engineering ensures the model learns consistent pairwise relationships regardless of which team wins.
- votes 138 (2025 1st Place Solution (modeh7) — 2026 Adaptation): GLM team quality shows higher or equal AUC compared to seed difference across all seasons, indicating that statistical performance metrics are more reliable than tournament seeding for predicting upsets.
- votes 119 (🏀 March ML Mania 2026 | HistGB + XGB + CatBoost): Standard stratified k-fold validation artificially inflates CV scores in tournament prediction by mixing future games into training folds.
- votes 114 ([LB 0.0] Time and Chance Happen to Them All): The public leaderboard evaluation set overlaps almost entirely with historical matchups that have known outcomes, making it an unreliable indicator of true model generalization.

## Notebooks indexed
- #763 votes [[notebooks/votes_01_martynaplomecka-march-machine-learning-mania-2026-starter/notebook|🏀 March Machine Learning Mania 2026 Starter]] ([kaggle](https://www.kaggle.com/code/martynaplomecka/march-machine-learning-mania-2026-starter))
- #353 votes [[notebooks/votes_02_kaito510-goto-conversion-winning-solution/notebook|goto_conversion🥇🥈🥈🥉🥉winning solution]] ([kaggle](https://www.kaggle.com/code/kaito510/goto-conversion-winning-solution))
- #309 votes [[notebooks/votes_03_lennarthaupts-calculate-elo-ratings/notebook|Calculate ELO-Ratings🏀]] ([kaggle](https://www.kaggle.com/code/lennarthaupts/calculate-elo-ratings))
- #261 votes [[notebooks/votes_04_jiaoyouzhang-ncaa-2026-final-lightgbm/notebook|NCAA 2026 |  Final - LightGBM]] ([kaggle](https://www.kaggle.com/code/jiaoyouzhang/ncaa-2026-final-lightgbm))
- #138 votes [[notebooks/votes_05_kacchanwriting-2025-1st-place-solution-modeh7-2026-adaptation/notebook|2025 1st Place Solution (modeh7) — 2026 Adaptation]] ([kaggle](https://www.kaggle.com/code/kacchanwriting/2025-1st-place-solution-modeh7-2026-adaptation))
- #129 votes [[notebooks/votes_06_ravi20076-ncaa2026-public-baseline-v1/notebook|NCAA2026|Public|Baseline|V1]] ([kaggle](https://www.kaggle.com/code/ravi20076/ncaa2026-public-baseline-v1))
- #125 votes [[notebooks/votes_07_asimandia-not-this-year-s-raddar-notebook/notebook|Not this year's Raddar Notebook]] ([kaggle](https://www.kaggle.com/code/asimandia/not-this-year-s-raddar-notebook))
- #119 votes [[notebooks/votes_08_emanuellcs-march-ml-mania-2026-histgb-xgb-catboost/notebook|🏀 March ML Mania 2026 | HistGB + XGB + CatBoost]] ([kaggle](https://www.kaggle.com/code/emanuellcs/march-ml-mania-2026-histgb-xgb-catboost))
- #114 votes [[notebooks/votes_09_ibrahimqasimi-ncaa-2026-eda-elo-ratings-and-gradient-esemble/notebook|🏀 NCAA 2026 EDA, Elo Ratings,and Gradient Esemble]] ([kaggle](https://www.kaggle.com/code/ibrahimqasimi/ncaa-2026-eda-elo-ratings-and-gradient-esemble))
- #114 votes [[notebooks/votes_10_veniaminnelin-lb-0-0-time-and-chance-happen-to-them-all/notebook|[LB 0.0] Time and Chance Happen to Them All]] ([kaggle](https://www.kaggle.com/code/veniaminnelin/lb-0-0-time-and-chance-happen-to-them-all))
