# march-machine-learning-mania-2026: cross-solution summary

The March Machine Learning Mania 2026 competition focused on predicting NCAA tournament game outcomes, with top solutions favoring minimalist feature sets, team-level differentials, and robust cross-validation schemes. Winning approaches predominantly leveraged XGBoost and Logistic Regression, often reframing the task as regression on point differential or blending models with external prediction markets. A consistent theme across submissions was aggressive feature pruning, probability clipping, and post-processing calibration to maximize bracket scoring and Brier score performance.

## Competition flows
- Raw box score and tournament data processed into efficiency metrics and custom ratings fed into XGBoost regressor with LOSO validation, isotonic calibration, and edge sharpening
- Regular-season and tournament data merged with external stats to compute team differentials and custom ratings for Logistic Regression blended with tiered market probabilities
- Competition data and scraped ESPN box scores filtered, enriched with ELO/Massey/Poisson features, and fed into proxy model with majority-vote bracket system
- Regular season box scores aggregated into season-level features for XGBoost regressor predicting point differential via LOSO CV, calibrated and averaged across six seeds
- Historical CSVs parsed for 35 relative matchup features fed into separate XGBoost and Logistic Regression pipelines, blended and clipped for submission

## Data reading
- Competition data loaded via Kaggle CLI and unzipped
- External datasets (Barttorvik, KenPom, AP Poll, Kalshi odds, WNCAA NET) loaded as CSVs/JSON from local external/ directory
- Team ID mapping files used to align disparate naming conventions across sources
- Scraped ESPN API data merged with competition dataset
- CSV files loaded via pandas/numpy

## Data processing
- Computed pace, offensive/defensive/net efficiency, rebound rate, and free throw rate from box scores
- Applied trimmed mean (2%) to compute season averages
- Manually prorated injury deductions for men's teams based on Rotowire status weights
- Applied MinMax scaling to strength of schedule, power conference, and AP poll features with gender-specific ranges
- Filtered out rebound and free throw rate features
- Computed features as team differentials (Team1 - Team2)
- Handled missing values with median imputation
- Applied StandardScaler normalization
- Performed aggressive backward elimination for feature pruning
- Clipped predictions to specific intervals (e.g., [0.03, 0.97] for men, [0.005, 0.995] for women, or unit interval)
- Converted Vegas moneylines to no-vig implied probabilities
- Converted championship odds via Bradley-Terry formula
- Filtered teams with <33% regular season win rate
- Removed play-in games from training
- Adjusted team quality for missed free throws and location
- Applied season replays to ELO ratings
- Fitted Poisson model season-by-season to prevent leakage
- Computed EMA of tournament experience
- Adjusted box score totals for overtime length
- Computed venue splits with small-sample fallback logic
- Combined men's and women's historical tournament data
- Filtered Massey ordinals to the last 14 days of the regular season
- Assigned default seed/rank for unseeded or missing teams
- Defaulted missing team profiles to 0.5 probability

## Features engineering
- Seed differential / Tournament seed difference
- Custom rating differentials (e.g., harry_Rating, Colley Matrix, SRS)
- Quality wins differential / GLM quality
- Average blocks differential
- Team-level differentials (win percentage, adjusted efficiency metrics, WAB, SOS, efg_pct, tov_pct, oreb_pct, fg3_pct, AP poll rank, preseason efficiency margin change, non-conference SOS, coach performance above seed expectation)
- Feature interactions (e.g., SeedNum × pt_diff, massey_rank × barthag, close_win_pct × pt_diff, srs × ap_rank)
- ESPN box scores (rebound/turnover)
- Team quality (point diff + adjustments + sample weights)
- Conference quality (inter-conference only)
- Replay-based ELO / Elo ratings per season / Elo rating difference
- Massey ordinals (POM, WLK, MOR) / Massey ordinal mean/median/min difference
- Proxy model for women's Massey rank
- Poisson expected wins
- EMA of reached tournament rounds / Tournament experience
- Majority vote bracket feature
- Season-averaged box score statistics (adjusted for overtime)
- Possession-based efficiency metrics (offensive/defensive/net rating)
- Shooting profile metrics (eFG%, TS%, 3PA rate, FT rate, FG%, FG3%, FT%, OppFG%)
- Rebounding percentages / Offensive/defensive rebounds
- Venue splits (home/road/recent)
- Gaussian GLM latent team quality scores
- Four Factors (EFG%, TORate, ORPct, FTRate) and Opponent Four Factors
- Per-game counting stats (assists, turnovers, steals, blocks, tempo)
- Strength of schedule difference
- Recent form (last-10-game win pct) difference
- Point differential / Avg points scored/allowed

## Models
- XGBoost
- Logistic Regression
- LightGBM
- Bradley-Terry model
- Poisson count model
- Proxy model
- ELO rating system
- Massey ordinals (POM, WLK, MOR)

## Frameworks used
- xgboost
- numpy
- pandas
- scikit-learn
- scipy
- Optuna

## Loss functions
- RMSE
- Log loss (cross-entropy)
- Brier score
- Binary:logistic

## CV strategies
- Leave-One-Season-Out (LOSO) cross-validation
- GroupKFold grouped by season
- 10-fold cross-validation on combined men's and women's data

## Ensembling
- No ensembling (used isotonic regression calibration and edge sharpening)
- Tiered blend of Logistic Regression and prediction market probabilities (Vegas, ESPN BPI, Kalshi) weighted by game tier
- Majority vote of seed, quality, and ELO combined with expected wins
- Averaging raw margin predictions across six random seeds with symmetric trim and clipping
- Weighted ensemble (60% XGBoost, 40% Logistic Regression) with probability clipping to [0.02, 0.98]

## Insights
- Trust the selection committee's seeding differential as a strong baseline
- Treat men's and women's tournaments as separate entities with gender-specific tuning
- A minimalist feature set consistently outperforms complex engineering in this domain
- Sharpening edges in post-processing is a valid negative-expected-value strategy for tournament bracket scoring
- Logistic Regression generalizes significantly better than gradient-boosted trees on small, unseen tournament datasets due to lower variance
- Prediction markets provide crucial orthogonal information for early-round games by capturing real-time injury and lineup data
- Aggressive feature pruning is essential to remove multicollinearity noise and prevent overfitting in low-sample regimes
- Carrying over 75% of end-of-season Elo ratings captures multi-year program strength better than resetting annually
- Integrating scraped ESPN box scores provides consistent advanced stats and improves the Brier score
- Point difference adjusted for missed free throws and location is the most impactful feature
- Massey ordinals at DayNum=133 yield the best results among ordinal methods
- A proxy model trained on men's data effectively predicts women's Massey ranks
- Historical tournament experience, measured via EMA of reached rounds, is a highly predictive feature
- Reframing game outcome prediction as a regression problem on point differential captures a richer training signal and encodes result strength better than binary classification
- Using a Gaussian GLM with team fixed effects to derive latent quality scores provides a principled statistical baseline for team strength
- Jointly optimizing hyperparameters and feature selection via Optuna with a Brier score objective effectively balances model complexity and calibration quality
- Framing all features as relative differences (Team1 − Team2) forces the model to learn matchup-specific dynamics rather than overfitting to absolute team strengths
- Aggregating multiple Massey ordinal ranking systems into a consensus metric captures team peaking trends more robustly than any single ranking source
- Incorporating a margin-of-victory multiplier with an autocorrection term into Elo ratings prevents runaway updates for dominant teams and adds predictive texture beyond binary win/loss updates
- Clipping predicted probabilities to [0.02, 0.98] is a direct, Brier-score-aware intervention that meaningfully reduces loss from extreme wrong predictions across the tournament bracket

## Critical findings
- Including rebound and free throw rate features degraded model performance
- Rewarding recent performance did not improve results
- Skipping injury adjustments for the women's tournament likely limited bracket accuracy
- XGBoost won the Stage 1 leaderboard by memorizing known training games but performed terribly on Stage 2 generalization
- Ensembling LR and XGBoost worsened both CV and LB scores, proving that blending fundamentally different model types on small data is detrimental
- Recency weighting improved XGBoost but hurt LR because downweighting older seasons unnecessarily reduced the effective sample size for a low-variance model
- Using all available Massey Ordinal systems (~28) for the composite ranking outperformed hand-picking a curated subset
- Seed10 vs Seed10 play-in games do not imply equal team strength and heavily skew validation metrics if included in training
- ELO ratings can decrease even after a win if the victory margin is lower than expected, necessitating point-differential-based updates
- Fitting the Poisson count model season-by-season is necessary to avoid data leakage across tournament years
- Deeper XGBoost trees (max_depth=6 or 8) caused clear overfitting on the small tournament dataset, making max_depth=4 the hard ceiling
- The final competition Brier score significantly outperformed cross-validation estimates due to a combination of small sample variance and a historically low-upset tournament year that rewarded conservative models
- Women's teams lack access to Massey ordinals in the provided dataset, creating a meaningful information gap that defaults them to a rank of 200 and limits model calibration for that gender

## What did not work
- Features other than seed differential, quality wins differential, and harry_Rating differential
- Rebound rate, free throw rate, and rewarding recent performance specifically hurt model performance
- XGBoost/LightGBM for Stage 2 generalization
- Ensembling LR + XGB
- Stacking individually-helpful features (multicollinearity)
- Recency weighting for LR
- Extended training data pre-2015 (distributional shift from NaNs)
- Isotonic calibration (trained on OOF vs full-model distribution mismatch)
- MSE objective for XGBoost (binary:logistic was better)
- Polynomial features, ElasticNet, Bagging, KNN blend, Tier-split by seed
- Deeper XGBoost trees (max_depth=6 and 8) led to worse CV Brier scores due to overfitting on the ~1,000 tournament games
- Using a single combined model for men's and women's data sacrificed the ability to tune each tournament's dynamics independently
- Relying on an empirical 60/40 blend ratio instead of a grid search with nested CV was suboptimal
- The absence of Massey ordinals for women's teams and lack of explicit upset-specific seed-matchup features limited predictive power

## Notable individual insights
- Rank 1 (March Machine Learning Mania 2026: 1st Place Solution): Sharpening edges in post-processing is a valid negative-expected-value strategy for tournament bracket scoring.
- Rank 3 (3rd Place Solution — March Machine Learning Mania 2026): Logistic Regression generalizes significantly better than gradient-boosted trees on small, unseen tournament datasets due to lower variance.
- Rank null (My approach for this year): Fitting the Poisson count model season-by-season is necessary to avoid data leakage across tournament years.
- Rank 10 (March Mania 2026: 10th Place Solution): Reframing game outcome prediction as a regression problem on point differential captures a richer training signal and encodes result strength better than binary classification.
- Rank 2 (2nd Place Solution for the March Machine Learning Mania 2026 Competition): Clipping predicted probabilities to [0.02, 0.98] is a direct, Brier-score-aware intervention that meaningfully reduces loss from extreme wrong predictions across the tournament bracket.

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
- Verification of forecasts expressed in terms of probability
- Basketball on Paper
