# march-machine-learning-mania-2026: top public notebooks

The top-voted notebooks primarily focus on constructing robust tabular baselines for NCAA tournament prediction, heavily leveraging Elo ratings, GLM-derived team quality, and season-level statistical averages. Several entries demonstrate gradient boosting ensembles (LightGBM, XGBoost, CatBoost) paired with temporal cross-validation schemes and probability calibration techniques like splines or isotonic regression. A few also explore multi-agent orchestration, specialized feature partitioning to reduce model correlation, and critical EDA findings regarding seed dominance and home-court stability.

## Common purposes
- baseline
- other
- tutorial
- feature_engineering

## Models
- LogisticRegression
- LightGBM
- XGBoost
- HistGradientBoostingClassifier
- CatBoost

## Frameworks
- google-adk
- scikit-learn
- pandas
- numpy
- matplotlib
- scipy.stats
- tqdm
- scipy
- statsmodels
- seaborn
- lightgbm
- xgboost
- catboost
- optuna
- joblib

## CV strategies
- 5-fold cross-validation with neg_brier_score scoring
- Time-based sequential validation (train on all prior seasons, validate on current season iteratively)
- Leave-one-season-out cross-validation
- Leave-One-Season-Out (LOSO)
- Leave-One-Season-Out (LOSO) cross-validation
- Temporal walk-forward cross-validation (Season < v for train, Season == v for validation) across 2018-2025
- Temporal holdout: validation on 2022–2025 tournaments, training on pre-2022 data (men: 2003+, women: 2010+)

## Preprocessing
- Elo rating computation with home-court adjustment (+/-100)
- Inter-season regression toward mean (0.75*prev + 0.25*1500)
- Seed string parsing (e.g., W01 -> 1)
- Feature construction as raw Elo and seed differences
- Probability clipping to [0.01, 0.99]
- Concatenates regular season and tournament compact results
- Adds binary tourney flag (0/1) and match weight columns
- Sorts chronologically by Season and DayNum
- Resets index after sorting
- Filters out tournament rows before aggregation
- Median imputation
- Min-max scaling (fit on train, applied to val/test)
- Seed string parsing (strip non-digits)
- Data duplication for matchup symmetry
- Overtime adjustment factor (40 + 5 * NumOT) / 40 applied to box scores
- Dataset doubled by swapping team positions (T1/T2)
- Point differential calculated as T1_Score - T2_Score
- Win label derived from point differential sign
- men/women indicator added
- Season cutoff set to 2003
- Concatenates men/women data, filters seasons >= 2003
- Swaps team roles to double dataset
- Adjusts box score stats for overtimes using (40 + 5 * NumOT) / 40
- Computes PointDiff target and binary win label
- Adds men_women indicator
- Overtime stat adjustment factor
- Symmetric home/away feature swapping
- Point differential calculation
- Gender flag encoding
- Season filtering (>=2003)
- Prediction clipping to [-30, 30]
- Fill numeric NaNs with column medians
- Convert base model probabilities to log-odds for meta-learner input
- StandardScaler applied to meta-learner features
- Clip final predictions to [0.015, 0.985]
- Missing values filled with 0
- Predictions clipped to [0.02, 0.98]
- Categorical/conf data converted to lookup dictionaries
- Seed strings parsed to integers
- Pairwise feature differencing for matchups
- Parsing submission IDs into Season/Team1/Team2
- Loading and concatenating historical compact results
- Normalizing team ordering (min/max for Team1/Team2)
- Dropping duplicates by taking the latest game per season/team pair

## Feature engineering
- Elo ratings with home-court adjustment and inter-season regression
- Seed parsing and seed difference
- Elo difference (Team1 - Team2)
- Seed difference (Team2 - Team1)
- Elo ratings with initial_rating=1200, width=1200, alpha=None
- K-factor: 125 (men), 190 (women)
- Match weights: regular=1.0/tourney=0.75 (men), regular=0.95/tourney=1.0 (women)
- Aggregated stats per TeamID/Season: mean, median, std, min, max, last
- Rating_Trend via linear regression slope over season
- Direct submission probability via 1/(1+10^((T2-T1)/width))
- Group-by aggregations (NumWins, NumLosses, mean score gaps per team/season)
- Derived ratios (WinRatio, GapAvg)
- Seed parsing to integer
- Symmetrical data duplication (A/B suffixes)
- Pairwise differences (SeedDiff, WinRatioDiff, GapAvgDiff)
- Aggregations: Regular season average box scores per team and per opponent (grouped by Season and TeamID, mean aggregation)
- Encodings: Seed extraction from string (int(x[1:3])), men_women binary indicator
- Interactions: Seed_diff (T2_seed - T1_seed), elo_diff, diff_quality
- Domain-derived: Custom Elo ratings (base_elo=1000, elo_width=400, k_factor=100) updated per game
- Domain-derived: GLM team quality using statsmodels.GLM with Gaussian family and formula 'PointDiff~-1+T1_TeamID+T2_TeamID' fitted per season/gender
- Seed extraction and Seed_diff
- Season-level mean aggregations of team and opponent box score stats (FGA, Blk, PF, PointDiff)
- Per-season ELO ratings (base=1000, width=400, k=100) and elo_diff
- GLM-derived team quality scores per season using statsmodels.GLM (Gaussian family, formula PointDiff~-1+T1_TeamID+T2_TeamID) and diff_quality
- Season-long team averages (FGA, OR, DR, Blk, PF, opponent stats, PointDiff)
- Elo ratings per season (base=1000, width=400, k=100)
- GLM-derived team quality (Gaussian GLM: PointDiff ~ -1 + T1_TeamID + T2_TeamID)
- Seed difference
- Gender flag
- Overtime adjustment factor
- Aggregations: team-season WinRate, AvgMargin, AvgScore, AvgOppScore; Massey rank stats (mean, median, std, best, worst)
- Domain-derived: Dean Oliver's Four Factors + efficiency (ORTG, DRTG, NetRtg, eFG%, TO%, ORB%, FTR, ThreePAr, ASTpct) using 0.44 possession multiplier
- Domain-derived: Strength of Schedule (avg opponent WinRate)
- Time-based: Rolling form (WinRate & AvgMargin over last 10 games)
- Domain-derived: Game-by-game Elo with home-court adjustment (+100), log-damped margin-of-victory multiplier, 75/25 season regression
- Domain-derived: Historical seed upset probability lookup (min 5 seasons)
- Interactions: Matchup-level differences (T1 - T2), SeedDiff x EloDiff, SeedDiff², SeedDiff x NetRtg
- Elo ratings with log-transformed margin-of-victory scaling, home-court adjustment, and 75% between-season mean reversion
- Dean Oliver Four Factors: offensive/defensive efficiency, eFG%, TO%, OR%, FTR, pace, assist rate, blocks/stl per game
- Strength of Schedule (SOS): average opponent Elo
- Late-season momentum: win rate in final 10 regular season games
- Conference strength: mean Elo of conference teams per season
- Coach experience: cumulative tournament appearances and program tenure
- Pairwise matchup differences (elo_d, seed_d, sos_d, mom_d, conf_d, cexp_d, cten_d) and raw team stats

## Loss functions
- LogisticRegression default log loss (evaluated via Brier score)
- binary_logloss (LightGBM objective='binary')
- reg:squarederror
- Brier Score (Optuna objective)
- LogLoss (XGBoost eval_metric)
- Logistic Regression (L2 regularized)
- Default binary classification losses (LightGBM, XGBoost, CatBoost); evaluated on Brier score

## Ensemble patterns
- Averages out-of-fold predictions across time-based validation folds.
- Predictions from season-specific models are averaged to produce the final submission.
- Specialized feature partitioning across three tree models (XGBoost on stats, CatBoost on rankings, HistGB on all) to reduce correlation, followed by log-odds transformation and blending via an L2-regularized Logistic Regression meta-learner trained on OOF predictions.
- Simple average of predicted probabilities from LightGBM, XGBoost, and CatBoost, clipped to [0.02, 0.98].

## Post-processing
- Probability clipping to [0.01, 0.99]
- Min-max scaling of validation predictions, clipping to [0, 1], averaging fold predictions for final submission.
- Predictions clipped to [-30, 30]; grouped by clipped point differential to compute empirical win percentages; probability conversion via scipy.interpolate.UnivariateSpline (k=5) on sorted predictions and labels; final probabilities clipped to [0.01, 0.99]; Brier score used for calibration evaluation.
- Clips predictions to [-30, 30], fits UnivariateSpline (k=5) on OOF predictions vs labels for calibration, clips calibrated probabilities to [0.02, 0.98], tracks Brier score.
- UnivariateSpline (k=5) probability calibration on OOF predictions, probability clipping [0.01, 0.99], alpha blending (0.9) for extreme probabilities (>0.9 or <0.1)
- Isotonic regression calibration on OOF blended probabilities; final prediction clipping to [0.015, 0.985]
- Probability clipping to [0.02, 0.98]; calibration curve analysis using quantile strategy.

## What worked
- ELO ratings alone are unlikely to reach the medal zone, but serve as powerful features; this approach was a key component in two prize-winning entries.
- Achieves a perfect 0.00000 public leaderboard score by exploiting historical data overlap.

## What did not work
- Does not generalize to unseen matchups
- Carries no real predictive value
- Expected to perform poorly on the private leaderboard

## Critical findings
- Seed difference is the single strongest predictor, with higher seeds winning >90% of games when the gap exceeds 10.
- Women's tournament is notably more chalk (higher seeds win more consistently) than the men's tournament.
- Home-court advantage remains stable at ~58–60% across decades, while raw scoring trends fluctuate, making efficiency metrics more reliable than point totals.
- The public leaderboard evaluation set overlaps almost entirely with historical matchups that have known outcomes, allowing trivial perfect scores via data leakage.

## Notable techniques
- Google ADK SequentialAgent orchestration for deterministic multi-stage pipelines
- LLM agents calling Python tools via docstring-based tool discovery
- Elo rating inter-season regression (0.75*prev + 0.25*init) to prevent rating drift
- Probability clipping to avoid extreme predictions in submission
- Margin-of-victory multiplier in Elo update formula
- Custom match weights for regular season vs tournament
- Rating trend calculated as linear regression slope over the season
- Direct submission generation using Elo difference probability formula
- Symmetrical data duplication to treat wins/losses as A/B pairs
- Time-based sequential CV for temporal data
- Min-max scaling fitted on training folds only
- Averaging out-of-fold predictions for final submission
- Overtime stat normalization using (40 + 5 * NumOT) / 40
- Dataset doubling via team position swapping to balance perspective
- GLM team quality derivation using statsmodels.GLM with Gaussian family
- Spline-based probability calibration from regression outputs
- Leave-one-season-out validation for temporal generalization
- GLM with fixed intercept and random slopes for team quality estimation
- Spline calibration on clipped point differentials to map regression outputs to win probabilities
- Symmetric home/away feature generation via role swapping
- GLM-based team quality estimation filtering only tournament-relevant teams
- UnivariateSpline calibration for probability mapping
- Alpha blending for extreme probability values
- Strict feature isolation to force ensemble diversity and reduce model correlation
- Log-odds transformation of probabilities before meta-learner to create a more linear input space
- Log-damped margin-of-victory multiplier in Elo updates to reward dominant wins proportionally
- 75/25 regression to the mean for Elo ratings between seasons
- Using SeedDiff as a meta-feature to enable context-aware blending based on seed mismatch severity
- Parallel feature engineering and concurrent model inference via ThreadPoolExecutor
- Isotonic calibration fitted strictly on OOF predictions to prevent probability distribution tightening
- Elo system with log-transformed margin-of-victory scaling and 75% between-season mean reversion
- Pairwise feature differencing to convert team-level stats into matchup-specific features
- Late-season momentum captured via rolling win rate over the final 10 regular season games
- Probability clipping to [0.02, 0.98] to prevent overconfident predictions and improve calibration
- Coach tenure and cumulative tournament appearance tracking as proxy for program stability
- Leveraging historical compact results to map exact outcomes to test IDs
- Defaulting predictions to 0.5 for unseen matchups to exploit public leaderboard data overlap

## Notable individual insights
- votes 763 (🏀 March Machine Learning Mania 2026 Starter): Google ADK SequentialAgent orchestration enables deterministic, multi-stage LLM-driven ML pipelines without manual coding.
- votes 309 (Calculate ELO-Ratings🏀): Custom Elo ratings with tunable K-factors and distinct regular season/tournament match weights significantly outperform default implementations.
- votes 138 (2025 1st Place Solution (modeh7) — 2026 Adaptation): GLM-derived team quality consistently outperforms raw seeding in predictive power across seasons.
- votes 119 (🏀 March ML Mania 2026 | HistGB + XGB + CatBoost): Strict feature isolation across XGBoost, CatBoost, and HistGB drastically reduces ensemble correlation and improves generalization.
- votes 114 (🏀 NCAA 2026 EDA, Elo Ratings,and Gradient Esemble): Seed difference remains the single strongest predictor, with higher seeds winning >90% of games when the gap exceeds 10.
- votes 114 ([LB 0.0] Time and Chance Happen to Them All): The public leaderboard evaluation set overlaps almost entirely with historical matchups, allowing trivial perfect scores via data leakage.

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
