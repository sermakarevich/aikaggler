# march-machine-learning-mania-2025: cross-solution summary

This NCAA tournament prediction competition focused on forecasting win probabilities and identifying championship contenders using a blend of gradient boosting, temporal feature engineering, and domain-driven post-processing. Winning approaches predominantly leveraged XGBoost with validation strategies that explicitly guarded against time-based leakage, while also incorporating targeted probability adjustments and manual domain overrides to capture early-round seed advantages and program-level cultural factors.

## Competition flows
- Raw Kaggle data is processed to extract temporal and matchup features, which are fed into an XGBoost model with Cauchy loss to generate leaf node predictions; these are combined via logistic regression and averaged across leave-one-season-out folds to produce final win/loss probabilities.
- The pipeline ingests provided tournament data alongside derived average statistics and Elo metrics, trains a tuned XGBoost regressor to predict win probabilities, and applies targeted probability boosts and manual overrides to low-confidence early-round matchups before submitting the final predictions.
- The author used a starter notebook's predictions as a baseline and manually overrode them with a Florida championship pick based on basketball analytics research and game observation.

## Data processing
- Laplace smoothing on temporal features (prior season matchup, away wins, last 14-day win ratio)
- Stacking out-of-fold predictions
- Removing spline components to prevent target leakage
- Hardcoding non-seeded team matches to 0.5 for memory efficiency
- Pivoting from unavailable external datasets to rely on provided data and derived average statistics/team quality metrics

## Features engineering
- Laplace-smoothed features (prior season matchup, away wins, last 14-day win ratio)
- Stacked out-of-fold predictions
- Features inherited from Raddar's Vilnius NCAA solution
- 29 features including seed differentials, average team statistics (score, FGA, OR, DR, Blk, PF, PointDiff), opponent-adjusted averages, Elo ratings, and team quality scores

## Models
- XGBoost
- Logistic regression

## Loss functions
- Cauchy loss function
- reg:squarederror
- Brier score

## CV strategies
- Leave-one-season-out for model averaging, guided by Brier score intuition on a left-out 2024 season; explicitly avoided standard CV due to time-based leakage.

## Ensembling
- Model averaging across leave-one-season-out predictions to produce final win/loss probabilities.
- Post-processing involving uniform boosting of predictions below 85% by 10% and manual overrides for low-confidence early-round matchups.

## Notable individual insights
- rank 4 (4th Place Solution for the March Machine Learning Mania 2025 Competition): Temporal leakage severely compromises validation in time-series sports data, making leave-one-season-out a more reliable averaging strategy than standard CV.
- rank 4 (4th Place Solution for the March Machine Learning Mania 2025 Competition): Including a spline component introduced target leakage, degrading model validity.
- rank 1 (First Place Solution): Early-round tournament outcomes heavily favor higher seeds, making manual overrides for low-confidence predictions highly effective.
- rank 1 (First Place Solution): External analytics datasets like 538 may be completely unavailable for current seasons, requiring rapid pivots to alternative feature sources.
- rank 2 (Second Place Solution: The raddar Prophecy Strikes Again...): Roster continuity (high percentage of returning players) is a strong predictor of tournament success and team cohesion.
- rank 2 (Second Place Solution: The raddar Prophecy Strikes Again...): Manual domain-knowledge overrides can outperform complex ML models when time is limited and the competition format allows flexible submissions.

## Solutions indexed
- #1 [[solutions/rank_01/solution|First Place Solution]]
- #2 [[solutions/rank_02/solution|Second Place Solution: The raddar Prophecy Strikes Again...]]
- #4 [[solutions/rank_04/solution|4th Place Solution for the March Machine Learning Mania 2025 Competition]]

## GitHub links
- [fakyras/ncaa_women_2018](https://github.com/fakyras/ncaa_women_2018) _(reference)_ — from [[solutions/rank_04/solution|4th Place Solution for the March Machine Learning Mania 2025 Competition]]
- [gotoConversion/goto_conversion](https://github.com/gotoConversion/goto_conversion) _(library)_ — from [[solutions/rank_02/solution|Second Place Solution: The raddar Prophecy Strikes Again...]]

## Papers cited
- Basketball Analytics: Objective and Efficient Strategies for Understanding How Teams Win
- Basketball on Paper: Rules and Tools for Performance Analysis
- Mathletics: How Gamblers, Managers, and Sports Enthusiasts Use Mathematics in Baseball, Basketball, and Football
