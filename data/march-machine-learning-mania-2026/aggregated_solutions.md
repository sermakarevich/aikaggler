# march-machine-learning-mania-2026: cross-solution summary

The March Machine Learning Mania 2026 competition challenges participants to predict NCAA tournament game outcomes, heavily emphasizing the handling of small test sets, gender-specific data dynamics, and real-time early-round signals. Winning approaches consistently favor lightweight, low-variance models like Logistic Regression or shallow XGBoost over complex ensembles, relying on rigorous temporal validation (Leave-One-Season-Out) and aggressive feature pruning to prevent overfitting. Success ultimately hinges on domain-informed feature engineering, strategic probability calibration/clipping, and leveraging external data sources like prediction markets or advanced ranking systems.

## Key challenges
- Small test-set variance and historically low-upset years rewarding conservative predictions
- Gender-specific data availability and predictability gaps
- Preventing model overfitting on limited tournament games (~600–1,000 rows)
- Incorporating real-time early-round signals (injuries, travel, markets)
- Avoiding temporal leakage in historical season data
- Play-in games acting as training outliers
- Absence of unified advanced ranking systems for women's teams

## Models
- XGBoost
- Logistic Regression
- LightGBM
- Colley Matrix
- Simple Rating System (SRS)
- GLM Quality
- Massey Composite
- Poisson count model
- Proxy model

## CV strategies
- Leave-One-Season-Out (LOSO) cross-validation
- GroupKFold grouped by season
- 10-fold cross-validation on combined dataset
- Season-by-season fitting

## Preprocessing
- Efficiency metrics from box scores
- Trim mean season averages
- MinMax scaling
- Injury deduction prorating
- Gender-specific processing pipelines
- Median imputation
- StandardScaler normalization
- Team ID mapping
- Prediction probability clipping
- Low win-rate team filtering
- Play-in game exclusion
- Overtime length adjustment
- Team1-Team2 difference framing
- Default values for missing/unseeded teams

## Feature engineering
- seed differential
- custom rating differentials (harry_Rating, Carry-over Elo, Massey Composite)
- opponent quality points differential
- block percentage differential
- injury status weights
- strength of schedule tiers
- power conference indicator
- team differentials (Team1 - Team2)
- tournament seed
- regular season win percentage
- adjusted offensive/defensive efficiency
- Wins Above Bubble (WAB)
- adjusted tempo
- effective field goal percentage
- turnover percentage
- offensive rebound percentage
- three-point field goal percentage
- preseason AP ranking
- KenPom efficiency margin change
- GLM quality rating
- Colley Matrix ranking
- non-conference strength of schedule
- Simple Rating System (SRS)
- coach performance above seed expectation
- feature interactions (seed × rating, rating × rating, etc.)
- block percentage
- recent form win percentage
- Elo trend slope
- Colley Matrix & SRS iterative ratings
- ESPN box score metrics
- team quality (point differential + missed free throws + location)
- sample weights by seed and day number
- conference quality
- Elo with season replays
- Massey ordinals
- tournament reach statistics
- exponentially weighted moving average (EMA) of tournament reach
- better team always wins bracket feature
- expected number of wins
- possession-based efficiency metrics
- shooting profile (eFG%, TS%, 3PA rate, FT rate)
- rebounding percentages
- venue splits (home/road)
- margin-of-victory Elo
- four factors
- shooting splits
- per-game counting stats
- strength of schedule
- recent form

## Loss functions
- RMSE
- Log loss (cross-entropy)
- Brier score
- logloss

## Ensemble patterns
- Single model per gender
- Tiered triple-market blend (market consensus + model)
- Multi-seed ensemble (averaging across random seeds)
- Weighted blend (XGBoost + Logistic Regression)

## Post-processing
- Probability sharpening (rounding extremes to 0.0/1.0)
- Bradley-Terry conversion for championship odds
- Tiered market weighting by round/gender
- Prediction probability clipping (various bounds)
- Symmetric clipping optimized via Optuna

## What worked
- seed_diff and custom harry_Rating as primary features
- Isotonic regression calibration
- Post-processing edge sharpening
- Injury adjustments for men's tournament
- Logistic Regression over XGBoost for Stage 2
- Aggressive feature pruning (backward elimination)
- Triple-market blend for Round 1 games
- Carry-over Elo (75%)
- Colley Matrix + SRS as custom rating systems
- Feature interactions
- Removing play-in games from training
- Switching team quality to point difference + missed free throws + location
- EMA of reached rounds as tournament experience proxy
- ESPN box scores for consistent metrics
- Predicting margin of victory for richer training signal
- Difference-vector feature framing
- Massey ordinal consensus from late-season data
- Margin-of-victory Elo adjustments
- Separate scaling pipelines per model
- Brier-aware probability clipping

## What did not work
- Rebound and free throw rate features
- Rewarding recent performance
- Tinkering with additional features beyond the core set
- XGBoost/LightGBM for Stage 2
- Ensembling LR + XGB
- Stacking individually-helpful features
- Recency weighting for LR
- Extended training data pre-2015
- Isotonic calibration
- MSE objective for XGBoost
- Polynomial features, ElasticNet, Bagging, KNN blend, Tier-split by seed
- XGBoost trees deeper than max_depth=4
- Empirical blend ratio selection without grid search
- Lack of Massey ordinals for women's teams
- Absence of explicit upset-specific seed-matchup features
- Single combined model for both genders

## Critical findings
- Post-processing probability sharpening is a negative expected value play but provides a marginal edge on tournament outcomes.
- Women's tournament predictions show higher correlation with seeding and custom rating differentials than men's.
- Injury adjustments were only feasible for the men's tournament due to data availability and complexity.
- LR dramatically outperforms XGBoost for unseen games due to lower variance, while XGB memorizes known training games.
- Prediction markets provide orthogonal information crucial for Round 1 where injuries and travel matter.
- Women's basketball is more seed-predictable, requiring fewer features to avoid overfitting on small data.
- Play-in games are significant outliers in training MSE and should be excluded.
- Seed 10 vs Seed 10 matchups do not imply equal team strength.
- Predicting margin of victory captures game strength and provides a richer training signal than binary classification.
- The CV-to-private LB gap was driven by small sample variance and a historically low-upset tournament year that natively rewarded conservative models.

## Notable individual insights
- rank 1 (March Machine Learning Mania 2026: 1st Place Solution): Post-processing probability sharpening is a negative expected value play but provides a marginal edge on tournament outcomes.
- rank 3 (3rd Place Solution — March Machine Learning Mania 2026): LR dramatically outperforms XGBoost for unseen games due to lower variance, while XGB memorizes known training games.
- rank 3 (3rd Place Solution — March Machine Learning Mania 2026): Prediction markets provide orthogonal information crucial for Round 1 where injuries and travel matter.
- rank null (My approach for this year): Play-in games are significant outliers in training MSE and should be excluded.
- rank 10 (March Mania 2026: 10th Place Solution): Predicting margin of victory captures game strength and provides a richer training signal than binary classification.
- rank 2 (2nd Place Solution for the March Machine Learning Mania 2026 Competition): The CV-to-private LB gap was driven by small sample variance and a historically low-upset tournament year that natively rewarded conservative models.

## Solutions indexed
- #1 [[solutions/rank_01/solution|March Machine Learning Mania 2026: 1st Place Solution ]]
- #2 [[solutions/rank_02/solution|2nd Place Solution for the March Machine Learning Mania 2026 Competition]]
- #3 [[solutions/rank_03/solution|3rd Place Solution — March Machine Learning Mania 2026]]
- #10 [[solutions/rank_10/solution| March Mania 2026: 10th Place Solution]]
- ? [[solutions/rank_xx_683033/solution|My approach for this year]]

## GitHub links
- [harrisonhoran/kaggle-march-mania-2026-1st-place](https://github.com/harrisonhoran/kaggle-march-mania-2026-1st-place) _(solution)_ — from [[solutions/rank_01/solution|March Machine Learning Mania 2026: 1st Place Solution ]]
- [kevin1000/march-mania-2026-3rd-place](https://github.com/kevin1000/march-mania-2026-3rd-place) _(solution)_ — from [[solutions/rank_03/solution|3rd Place Solution — March Machine Learning Mania 2026]]
- [BrendanCarlin/march-mania-2026-2nd-place](https://github.com/BrendanCarlin/march-mania-2026-2nd-place) _(solution)_ — from [[solutions/rank_02/solution|2nd Place Solution for the March Machine Learning Mania 2026 Competition]]

## Papers cited
- [XGBoost: A Scalable Tree Boosting System](https://arxiv.org/abs/1603.02754)
- [Verification of forecasts expressed in terms of probability](https://journals.ametsoc.org/view/journals/bams/31/1/1520-0477-31_1_1)
- [Basketball on Paper](https://www.amazon.com/Basketball-Paper-Stats-Transforming-Game/dp/0767916891)
