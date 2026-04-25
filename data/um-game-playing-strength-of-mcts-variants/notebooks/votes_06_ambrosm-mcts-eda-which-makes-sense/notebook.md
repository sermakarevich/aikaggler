# MCTS EDA which makes sense ⭐️⭐️⭐️⭐️⭐️

- **Author:** AmbrosM
- **Votes:** 372
- **Ref:** ambrosm/mcts-eda-which-makes-sense
- **URL:** https://www.kaggle.com/code/ambrosm/mcts-eda-which-makes-sense
- **Last run:** 2024-09-24 06:13:53.847000

---

# EDA which makes sense for "Game-Playing Strength of MCTS Variants"

This notebook shows you
- a detailed analysis of the training data,
- how to cross-validate your models,
- what are the most important features, 
- how to find the most important feature interactions,
- and how to ensemble the predictions of your models.

The goal of the competition is predicting what variant of Monte Carlo tree search (MCTS) is good for what games (based on a few hundred features of the game). The problem is formulated as a regression task: Predict what fraction of the games one Monte Carlo tree search agent wins against another one.

References
- Competition [UM - Game-Playing Strength of MCTS Variants](competitions/um-game-playing-strength-of-mcts-variants/)

```python
import polars as pl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from colorama import Fore, Style
import lightgbm, xgboost, catboost
import pickle

from sklearn.base import clone, BaseEstimator, TransformerMixin
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, cross_val_score, cross_val_predict
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, PolynomialFeatures, SplineTransformer
from sklearn.kernel_approximation import Nystroem
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor, BaggingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error

pd.options.mode.chained_assignment = "raise"

saved_models, oof_pred = {}, {}
```

# Reading the data

We first read the training data (a single file) and clean constant columns, null values and duplicates.

```python
train = pl.read_csv('/kaggle/input/um-game-playing-strength-of-mcts-variants/train.csv')
print('Shape before dropping columns:', train.shape)

constant_columns = np.array(train.columns)[train.select(pl.all().n_unique() == 1).to_numpy().ravel()]
print(len(constant_columns), 'columns are constant. These will be dropped.')

drop_columns = list(constant_columns) + ['Id']

train = train.drop(drop_columns)
print('Shape after dropping columns:', train.shape)

# Null values
print('There are', train.null_count().to_numpy().sum(), 'missing values.')

# Duplicates
print('There are', len(train) - train.n_unique(), 'duplicates.')

# Boolean columns
print('There are', train.select(pl.all().n_unique() == 2).to_numpy().sum(), 'binary columns.')

train
```

- The training data have a quarter million rows. This is enough for gradient-boosting models and neural networks. It is too much for kernel methods and k nearest neighbors.
- According to the [documentation](https://www.kaggle.com/competitions/um-game-playing-strength-of-mcts-variants/data), the test data have 60000 rows, of which 20000 are used for the public leaderboard and 40000 for the private leaderboard. 
- After we drop the 216 constant columns, 597 columns remain. These are
  - 5 string columns (GameRulesetName, agent1, agent2, EnglishRules, LudRules)
  - 183 float64 columns (including the utility_agent1 target) (of which 3 are binary)
  - 409 int64 columns (including the target-related num\_[wins/draws/losses]\_agent1) (of which 379 are binary)
- There are no duplicated rows.

```python
# The string columns
display(train.select(pl.col(pl.String)).head())

# The float64 columns
display(train.select(pl.col(pl.Float64)).head(2))

# The int64 columns
display(train.select(pl.col(pl.Int64)).head(2))

# The three float columns with only two unique values
display(train.select(col for col in train.select(pl.col(pl.Float64)) if col.n_unique() == 2).head(2))

# The four target-related columns
display(train.select(pl.col('^num_.*$', 'utility_agent1')).head(2))
```

In this EDA, we won't look at EnglishRules (natural language text) and LudRules (a programming language).

# The target

Most of the rows are the result of 15 games played, although in 1 % of the cases 30 or 45 games were played.

**Insight:**
1. The competition is a regression task.
2. You may want to give the rows with 30 or 45 games a higher sample weight in training.

```python
train.select(pl.col('^num_.*$')).sum_horizontal().rename('num_games_played').value_counts().sort('num_games_played')
```

The targets have an interesting distribution:
- Many games rarely end in a draw. If you play 15 rounds of such a game, the outcome is an element of {-1, 2/15-1, 4/15-1, 6/15-1, ... 28/15-1, 1}.
- In a substantial amount of rows, utility_agent1 is 0. This means that there were only draws (or the same number of wins and losses).
- In a substantial amount of rows, utility_agent1 is -1 or +1. This means that one player won all rounds of the game.
- Over the whole set of games, player 1 (the player who makes the first move) has an advantage.

**Insight:**
1. You can clip your predictions to (-1, +1).
2. You can treat games which always end in a draw as a special case. There is a feature 'DrawFrequency' which is related to such games.
3. You can try to do something with the case that one of the players wins all rounds of the game.
4. The winning probability for player 1 depends on how big an advantage (or disadvantage) the first player has in a game. There is a feature 'AdvantageP1' exactly for that purpose. We'll later see that it is the most important feature of all.

```python
plt.bar(*np.unique(train.select('utility_agent1').to_numpy(), return_counts=True), width=0.05, color='y')
plt.xlabel('utility_agent1 (target)')
plt.ylabel('count')
plt.xticks(np.linspace(-1, 1, 11))
plt.title('Target distribution')
plt.show()
```

```python
train.select((pl.col('num_wins_agent1') + pl.col('num_losses_agent1') == 0).alias('Games which always end in a draw')).sum()
```

The target, utility_agent1, can be computed from the count of wins, draws and losses:

```python
computed = (train.select(computed=(pl.col('num_wins_agent1') * 2 + pl.col('num_draws_agent1'))) 
            / train.select(pl.col('^num_.*$')).sum_horizontal() - 1)
pl.concat([computed, train.select('utility_agent1')], how='horizontal')
```

```python
# Advantage for the player who makes the first move (average over all games)
train.select('utility_agent1').mean()
```

Kaggle provides a `sample_submission.csv` for the competition. This submission predicts the same value for all targets, and it has a leaderboard score of 0.571. If we do the same for the training data, we get a score of 0.623, which is much higher. This difference suggests that that game outcomes in the test set are less extreme than in the training set. 

You may ask whether this difference is significant. @roberthatch shows with a simple [scatterchart](https://www.kaggle.com/competitions/um-game-playing-strength-of-mcts-variants/discussion/533210#2986757) that it is not. Considering the small size of the public test dataset, such differences are to be expected.

**Insight:** We may assume that the test data have the same distribution as the training data.

```python
target = train.select('utility_agent1')
rmse = mean_squared_error(target,
                          np.full_like(target, target.mean()),
                          squared=False)
print(f"RMSE of sample submission on training data: "
      f"{rmse:.3f}")
```

```python
plt.figure(figsize=(4, 3))
bar_container = plt.bar([0, 1], 
                        [0.571, rmse],
                        color='chocolate',
                        tick_label=['public test', 'training'])
plt.gca().bar_label(bar_container, fmt='%.3f')
plt.ylim(0, 0.7)
plt.ylabel('RMSE')
plt.title('Score of constant submission')
plt.show()
```

# Games

- There are 1377 games in the training dataset. The test dataset will have other, previously unseen, games.
- The games have been played with widely varying frequencies: Pathway occurs 222 times, Faraday occurs only 4 times.

**Insight:** As the test dataset has different games, we must not use GameRulesetName as a feature. We should, however, use it for grouping in a GroupKFold.

```python
train.select(pl.col('GameRulesetName')).to_series().value_counts(sort=True)
```

All int and float columns except the target-related ones depend on GameRulesetName. We show this by grouping the dataframe by GameRulesetName and computing the standard deviation of every group. The standard deviations are zero; this means that within every group, all samples have the same feature value (i.e., all samples with the same GameRulesetName):

```python
print("Groupwise standard deviations of columns")
print(train.select('GameRulesetName', pl.col(pl.Int64), pl.col(pl.Float64))
      .group_by('GameRulesetName')
      .agg(pl.all().std())
      .drop('GameRulesetName')
      .max()
      .transpose(include_header=True, header_name='Feature', column_names=['max std'])
      .sort('max std')
     )
```

# Players

There are 72 players. Their names have the format `MCTS-<SELECTION>-<EXPLORATION_CONST>-<PLAYOUT>-<SCORE_BOUNDS>` as described [here](https://www.kaggle.com/competitions/um-game-playing-strength-of-mcts-variants/data). The players have all 4\*3\*3\*2=72 possible combinations of the four features, and all players play with similar frequencies.
    
A player never plays against itself.

```python
train.select(pl.col('agent1', 'agent2').n_unique())
```

```python
train.select(pl.col('agent1')).to_series().value_counts(sort=True)
```

```python
# Does a player play against itself?
train.select((pl.col('agent1') == pl.col('agent2')).alias('Games of a player against itself')).sum()
```

We can extract four features from every player name.

```python
# Features extracted from player names
train.select(pl.col('agent1').str.extract(r'MCTS-(.*)-(.*)-(.*)-(.*)', 1).alias('p1_selection'),
             pl.col('agent1').str.extract(r'MCTS-(.*)-(.*)-(.*)-(.*)', 2).alias('p1_exploration').cast(pl.Float32),
             pl.col('agent1').str.extract(r'MCTS-(.*)-(.*)-(.*)-(.*)', 3).alias('p1_playout'),
             pl.col('agent1').str.extract(r'MCTS-(.*)-(.*)-(.*)-(.*)', 4).alias('p1_bounds'),
             pl.col('agent2').str.extract(r'MCTS-(.*)-(.*)-(.*)-(.*)', 1).alias('p2_selection'),
             pl.col('agent2').str.extract(r'MCTS-(.*)-(.*)-(.*)-(.*)', 2).alias('p2_exploration').cast(pl.Float32),
             pl.col('agent2').str.extract(r'MCTS-(.*)-(.*)-(.*)-(.*)', 3).alias('p2_playout'),
             pl.col('agent2').str.extract(r'MCTS-(.*)-(.*)-(.*)-(.*)', 4).alias('p2_bounds')
            )
```

# Cross-validation

Before we start cross-validation, we convert the polars dataframe to pandas.

In this conversion, we
- add eight features extracted from player names,
- extract utility_agent1 as the target value and GameRulesetName for the GroupKFold,
- drop GameRulesetName, the two freetext and the four target-related columns,
- convert all string features to the pandas Categorical dtype.

```python
%%writefile preprocess.py

import polars as pl
import pandas as pd

def preprocess(df_polars):
    """Convert the polars dataframe to pandas; extract target and groups if it is the training dataframe
    
    The function should be applied to training and test datasets.
    
    Parameters
    df_polars: polars DataFrame (train or test)
    
    Return values:
    df: pandas DataFrame with all features of shape (n_samples, n_features)
    target: target array of shape (n_samples, ) or None
    groups: grouping array for GroupKFold of shape (n_samples, ) or None
    """
    global cat_mapping
    
    # Add eight features extracted from player names,
    # Drop GameRulesetName, freetext and target columns
    df = df_polars.with_columns(
        pl.col('agent1').str.extract(r'MCTS-(.*)-(.*)-(.*)-(.*)', 1).alias('p1_selection'),
        pl.col('agent1').str.extract(r'MCTS-(.*)-(.*)-(.*)-(.*)', 2).alias('p1_exploration').cast(pl.Float32),
        pl.col('agent1').str.extract(r'MCTS-(.*)-(.*)-(.*)-(.*)', 3).alias('p1_playout'),
        pl.col('agent1').str.extract(r'MCTS-(.*)-(.*)-(.*)-(.*)', 4).alias('p1_bounds'),
        pl.col('agent2').str.extract(r'MCTS-(.*)-(.*)-(.*)-(.*)', 1).alias('p2_selection'),
        pl.col('agent2').str.extract(r'MCTS-(.*)-(.*)-(.*)-(.*)', 2).alias('p2_exploration').cast(pl.Float32),
        pl.col('agent2').str.extract(r'MCTS-(.*)-(.*)-(.*)-(.*)', 3).alias('p2_playout'),
        pl.col('agent2').str.extract(r'MCTS-(.*)-(.*)-(.*)-(.*)', 4).alias('p2_bounds')
    ).drop(
        ['GameRulesetName', 'EnglishRules', 'LudRules', 
         'num_wins_agent1', 'num_draws_agent1',
         'num_losses_agent1', 'utility_agent1'],
        strict=False
    ).to_pandas()

    if 'utility_agent1' in df_polars.columns: # Processing the training data
        # Extract the target
        target = df_polars.select('utility_agent1').to_numpy().ravel()

        # Extract the groups for the GroupKFold
        groups = df_polars.select('GameRulesetName').to_numpy()
        
        # Set the mapping to categorical dtypes
        cat_mapping = {feature: pd.CategoricalDtype(categories=list(set(df[feature]))) for feature in df.columns[df.dtypes == object]}
    else: # Processing the test data
        target, groups = None, None
        
    # Convert the strings to categorical
    df = df.astype(cat_mapping)

    return df, target, groups
```

```python
exec(open('preprocess.py', 'r').read())

train_pd, y, groups = preprocess(train)
```

We'll use the function cross_validate_model to cross-validate all our models:

```python
crossval_kf = GroupKFold()
folds = list(crossval_kf.split(train_pd, groups=train.select('GameRulesetName')))
    
def cross_validate_model(model, features=train_pd.columns, label='', save_models=False):
    global oof
    start_time = datetime.datetime.now()
    oof = np.full_like(y, np.nan)
    model_list = []
    for fold, (idx_tr, idx_va) in enumerate(folds):
        X_tr = train_pd[features].iloc[idx_tr]
        X_va = train_pd[features].iloc[idx_va]
        y_tr = y[idx_tr]
        y_va = y[idx_va]

        m = clone(model)
        m.fit(X_tr, y_tr)
        y_pred = m.predict(X_va).clip(-1, 1)
        if save_models:
            model_list.append(m)
        del m
        oof[idx_va] = y_pred
        rmse = mean_squared_error(y_va, y_pred, squared=False)
        print(f"# Fold {fold}: {rmse:=.3f}")
        
    elapsed_time = datetime.datetime.now() - start_time
    rmse = mean_squared_error(y, oof, squared=False)
    print(f"{Fore.GREEN}# Overall RMSE={rmse:.3f} {label}"
          f"   {int(np.round(elapsed_time.total_seconds() / 60))} min{Style.RESET_ALL}")
    if save_models:
        saved_models[label] = dict(features=features, model_list=model_list)
        oof_pred[label] = oof
```

Let's check that the cross-validation function works as desired:

```python
model = DummyRegressor()
cross_validate_model(model, label=f"Dummy")
# Overall RMSE=0.623 Dummy   0 min
```

# Using only the player features

If we use only the player-dependent features (and not the hundreds of features of the games), it doesn't matter whether we use a tree model or linear regression. The score is about 0.607.

```python
player_features = ['p1_selection', 'p1_exploration', 'p1_playout', 'p1_bounds',
                   'p2_selection', 'p2_exploration', 'p2_playout', 'p2_bounds']
```

```python
msl = 300
model = make_pipeline(OrdinalEncoder(), 
                      RandomForestRegressor(n_estimators=100, min_samples_leaf=msl))
cross_validate_model(model, features=player_features, label=f"Random forest min_samples_leaf={msl}")
# Fold 0: 0.601
# Fold 1: 0.610
# Fold 2: 0.613
# Fold 3: 0.601
# Fold 4: 0.607
# Overall RMSE=0.606 Random forest min_samples_leaf=300   1 min
```

```python
msl = 500
model = make_pipeline(OrdinalEncoder(), 
                      ExtraTreesRegressor(n_estimators=10, min_samples_leaf=msl))
cross_validate_model(model, features=player_features, label=f"ExtraTreesRegressor min_samples_leaf={msl}")
# Fold 0: 0.601
# Fold 1: 0.610
# Fold 2: 0.613
# Fold 3: 0.601
# Fold 4: 0.607
# Overall RMSE=0.607 ExtraTreesRegressor min_samples_leaf=500   0 min
```

```python
model = make_pipeline(OneHotEncoder(drop='first', sparse_output=False),
                      StandardScaler(),
                      LinearRegression())
cross_validate_model(model, features=['agent1', 'agent2'], label=f"Linear regression 1", save_models=True)
# Overall RMSE=0.607 Linear   2 min
```

# Feature importance

We now use an untuned LightGBM model to determine the permutation importance of the features. There are two observations to be made:
1. For this model, only 30 to 40 percent of the features are important. All others are useless.
2. If you look at the top five rows of every importance table, you'll see that AdvantageP1 is always the most important feature, followed by three player attributes (selection strategy, playout strategy and exploration constant). PlayoutsPerSecond usually comes next.

```python
%%time

# All the game features (concepts) have a ComputationTypeId, which is either 'Compiler' or 'Simulation'
concepts = pd.read_csv('/kaggle/input/um-game-playing-strength-of-mcts-variants/concepts.csv', index_col='Id')
concepts[['TypeId', 'DataTypeId', 'ComputationTypeId', 'LeafNode', 'ShowOnWebsite']] = concepts[['TypeId', 'DataTypeId', 'ComputationTypeId', 'LeafNode', 'ShowOnWebsite']].astype(int)
concepts.replace({'ComputationTypeId': {1: 'Compiler', 2: 'Simulation'}}, inplace=True)
# print(concepts.ComputationTypeId.value_counts())

features = [f for f in train_pd.columns if f not in ['agent1', 'agent2']]
X = train_pd[features].copy()
X['p_selection'] = (X.p1_selection.astype(str) + '-' + X.p2_selection.astype(str)).astype('category')
X['p_exploration'] = X.p1_exploration - X.p2_exploration
X['p_playout'] = (X.p1_playout.astype(str) + '-' + X.p2_playout.astype(str)).astype('category')
X['p_bounds'] = (X.p1_bounds.astype(str) + '-' + X.p2_bounds.astype(str)).astype('category')
display(X.head(3))

lgbm_params_fast = {'learning_rate': 0.2, 'colsample_bytree': 0.7, 'verbose': 0}
model = lightgbm.LGBMRegressor(**lgbm_params_fast)
kf = GroupShuffleSplit(n_splits=5, random_state=1)
for fold, (idx_tr, idx_va) in enumerate(kf.split(train_pd, groups=groups)):
    X_tr = X.iloc[idx_tr]
    X_va = X.iloc[idx_va]
    y_tr = y[idx_tr]
    y_va = y[idx_va]
#     model.fit(X_tr, y_tr, eval_set=(X_va, y_va), eval_metric='rmse', callbacks=[lightgbm.log_evaluation()])
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_va)
    rmse = mean_squared_error(y_va, y_pred, squared=False)
    
    result = permutation_importance(model, X_va, y_va, scoring='neg_root_mean_squared_error', n_repeats=2)
    
    print(f"{Fore.GREEN}{Style.BRIGHT}Important features: {(result['importances_mean'] > 0).mean():.0%}   ({rmse=:.3f}){Style.RESET_ALL}")
    importance_df = pd.DataFrame({'importance': result['importances_mean'],
                        'std': result['importances_std']}, index=X_va.columns).sort_values('importance', ascending=False)
    importance_df['ComputationTypeId'] = concepts.set_index('Name').ComputationTypeId
    importance_df.fillna({'ComputationTypeId': 'Player'}, inplace=True)
    display(importance_df.head(50))
    print()
    break
    
# Keep the good features for later
good_features = list(importance_df.query("importance > 0").index)
good_features = [f for f in good_features if f not in ['p_selection', 'p_exploration', 'p_playout', 'p_bounds']]

# 10 minutes
```

```python
# good_features = ['AdvantageP1', 'p2_selection', 'p1_exploration', 'p2_exploration', 'PlayoutsPerSecond', 'OutcomeUniformity', 'p1_selection', 'DurationTurnsStdDev', 'MovesPerSecond', 'Region', 'DrawFrequency', 'DurationMoves', 'DecisionMoves', 'HopCaptureFrequency', 'DurationTurns', 'Asymmetric', 'PieceNumberMaximum', 'DurationTurnsNotTimeouts', 'Balance', 'DecisionFactorMedian', 'GameTreeComplexity', 'BranchingFactorChangeAverage', 'BoardSitesOccupiedMaxIncrease', 'DecisionFactorMaxIncrease', 'StateTreeComplexity', 'SowWithEffect', 'BranchingFactorAverage', 'BranchingFactorMedian', 'BranchingFactorChangeMaxDecrease', 'p1_playout', 'p2_playout', 'ConditionalStatement', 'SetNextPlayerFrequency', 'NumOuterSites', 'HopDecisionEnemyToEmptyFrequency', 'BranchingFactorVariance', 'NumStartComponentsHand', 'NumOrthogonalDirections', 'DecisionFactorAverage', 'DecisionFactorChangeSign', 'Capture', 'NumPlayableSitesOnBoard', 'DurationActions', 'Completion', 'NumEdges', 'BoardSitesOccupiedChangeSign', 'BranchingFactorMaximum', 'BranchingFactorChangeSign', 'PlayersWithDirections', 'MorrisTiling', 'PieceNumberVariance', 'DecisionFactorChangeAverage', 'BoardSitesOccupiedMaxDecrease', 'DecisionFactorChangeLineBestFit', 'BoardSitesOccupiedMedian', 'MoveDistanceChangeAverage', 'NumStartComponentsBoard', 'PromotionDecisionFrequency', 'Disjunction', 'Negation', 'NumRightSites', 'StepDecisionFrequency', 'FromToDecisionBetweenContainersFrequency', 'BranchingFactorChangeMaxIncrease', 'AnimalComponent', 'Line', 'NoMovesDraw', 'BoardSitesOccupiedVariance', 'EliminatePiecesWinFrequency', 'PieceNumberMedian', 'StepDecisionToEnemyFrequency', 'ScoreDifferenceMaximum', 'Set', 'RemoveDecisionFrequency', 'Conjunction', 'NumStartComponents', 'NumCentreSites', 'BoardCoverageFull', 'LineWinFrequency', 'MancalaBoard', 'MoveDistanceAverage', 'NumCells', 'PieceNumberChangeLineBestFit', 'Complement', 'DecisionFactorMaxDecrease', 'DecisionFactorChangeNumTimes', 'RemoveEffectFrequency', 'BoardSitesOccupiedAverage', 'NumPlayableSites', 'MoveDistanceChangeSign', 'NoMovesEndFrequency', 'BoardCoverageDefault', 'Subtraction', 'NoMovesWin', 'PieceNumberChangeAverage', 'PieceNumberMaxDecrease', 'LeapDecisionToEnemy', 'CheckmateWin', 'DecisionFactorMaximum', 'ConnectionWinFrequency', 'FromToDecisionFriendFrequency', 'NumBottomSites', 'NumComponentsTypePerPlayer', 'BallComponent', 'AlquerqueTiling', 'BoardCoverageUsed', 'NumComponentsType', 'PieceNumberAverage', 'DecisionFactorVariance', 'Union', 'SpaceEnd', 'StepDecisionToFriend', 'ScoringEndFrequency', 'CaptureEnd', 'ConnectionEndFrequency', 'MovesDecision', 'Multiplication', 'CheckmateFrequency', 'HopDecisionMoreThanOneFrequency', 'Track', 'NumInnerSites', 'ScoreDifferenceChangeLineBestFit', 'CustodialCaptureFrequency', 'Algorithmics', 'ConcentricTiling', 'NoPieceNext', 'StepEffect', 'EliminatePiecesEndFrequency', 'TwoSitesMoves', 'SowCW', 'SameLayerDirection', 'MovesNonDecision', 'AdjacentDirection', 'SetMove', 'BranchingFactorChangeLineBestFit', 'StepDecision', 'ReachEndFrequency', 'PieceCount', 'ForgetValues', 'ScoreDifferenceChangeAverage', 'PieceState', 'TriangleTiling', 'AlquerqueBoard', 'LineEndFrequency', 'SpaceConditions', 'MoveDistanceMaximum', 'NumStartComponentsBoardPerPlayer', 'GoStyle', 'StepDecisionToEmpty', 'HopDecisionFrequency', 'PolygonShape', 'AllDirections', 'SetSiteState', 'PushEffectFrequency', 'NoTargetPieceEndFrequency', 'Logic', 'Arithmetic', 'HopEffect', 'SowSkip', 'Vertex', 'PushEffect', 'SquareShape', 'NoSiteMoves', 'MoveConditions', 'MancalaFourRows', 'Symbols', 'LineLoss', 'DoLudeme', 'Phase', 'SetValueFrequency', 'SowCaptureFrequency', 'NumConcaveCorners', 'SowBacktrackingFrequency', 'HopDecision', 'ReachWinFrequency', 'NumPhasesBoard', 'ControlFlowStatement', 'ScoreDifferenceAverage', 'TerritoryEndFrequency', 'BoardSitesOccupiedChangeLineBestFit', 'LeapDecisionFrequency', 'Distance', 'Piece', 'SetVar', 'Connection', 'Directions', 'MoveDistanceMedian', 'Meta', 'ForwardRightDirection', 'NoMovesMover', 'Drawishness', 'RegularShape', 'MoveDistanceMaxDecrease', 'Edge', 'LesserThanOrEqual', 'HopCaptureMoreThanOneFrequency', 'p2_bounds', 'LineEnd', 'PassDecisionFrequency', 'NumDice', 'SquareTiling', 'StepDecisionToFriendFrequency', 'SwapPlayersDecision', 'TriangleShape', 'PromotionEffect', 'ConnectionWin', 'RollFrequency', 'ReachEnd', 'NumRows', 'Repetition', 'Style', 'GreaterThan', 'BackwardsDirection', 'MovesEffects', 'Stochastic', 'NumAdjacentDirections', 'ScoreDifferenceVariance', 'SquarePyramidalShape', 'ByDieMove', 'HexShape', 'Contains', 'SwapPlayersEffect', 'OpeningContract', 'Flip', 'p1_bounds', 'SiteState', 'Parity', 'Trigger', 'Efficiency', 'NumContainers', 'Conditions']
```

AdvantageP1 is the most important feature. It is defined as the percentage of games which player 1 won in a simulation with **two players which play randomly**. It has to be interpreted with caution: Even if AdvantageP1 = 1.0 (the highest value possible), it is possible that player 1 loses:

```python
train.filter(pl.col('AdvantageP1') == 1).select('^num_.*_agent1$').sum()
```

DrawFrequency is another important feature. It has to be interpreted with some caution as well: Even if DrawFrequency = 1.0 (the highest value possible), only a third of the games ends in a draw:

```python
train.filter(pl.col('DrawFrequency') == 1.0).select('^num_.*_agent1$').sum()
```

# Explaining the most obvious interactions with InterpretML

We now use the [InterpretML](https://interpret.ml/) library to understand the most important interactions among features. We do this by fitting a generalized additive model to the five most important features and then using InterpretML's explanation diagrams to look into the model.

```python
!pip install -q interpret
```

```python
%%time
from interpret.glassbox import ExplainableBoostingRegressor

features = ['AdvantageP1', 'PlayoutsPerSecond', 'p_selection', 'p_exploration', 'p_playout']

kf = GroupShuffleSplit(n_splits=5, random_state=2)
for fold, (idx_tr, idx_va) in enumerate(kf.split(train_pd, groups=groups)):
    X_tr = X.iloc[idx_tr][features]
    y_tr = y[idx_tr]
    break
    
X_tr['log PlayoutsPerSecond'] = np.log(X_tr['PlayoutsPerSecond'])
X_tr.drop(columns=['PlayoutsPerSecond'], inplace=True)
model = ExplainableBoostingRegressor(learning_rate=0.1, outer_bags=4, max_rounds=20000, min_samples_leaf=256, validation_size=0.15, random_state=1)
model.fit(X_tr, y_tr)
explanation = model.explain_global()
```

The following interactive diagrams (Plotly) are to be interpreted as follows:
- The first diagram shows with a line how the target depends on the continuous variable AdvantageP1.
- The subsequent three diagrams show in blue bar charts how the target depends on the categorical variables describing the MCTS strategy elements of player 1 and player 2.
- Each of the final two diagrams (heatmaps) shows the interaction between a categorical feature and AdvantageP1.

```python
fig = explanation.visualize(0)
fig.update_layout(title_text="The score grows with AdvantageP1")
fig.show()
```

```python
fig = explanation.visualize(1)
fig.update_layout(
    title={
        'text': "p_selection: UCB1Tuned usually has an advantage against other selection strategies",
    },
    yaxis={'range': [-0.2, 0.2]})
fig.show()
```

```python
fig = explanation.visualize(2)
fig.update_layout(
    title={
        'text': "p_exploration: 1.414 usually is better than 0.6, 0.6 usually is better than 0.1 (in other words: higher is better)",
    },
    yaxis={'range': [-0.2, 0.2]})
fig.show()
```

```python
fig = explanation.visualize(3)
fig.update_layout(
    title={
        'text': "p_playout: In general, MAST is better than Random200, Random200 is better than NST",
    },
    yaxis={'range': [-0.2, 0.2]})
fig.show()
```

```python
fig = explanation.visualize(7)
fig.update_layout(title_text="Interaction between PlayoutsPerSecond and p_selection: ProgressiveHistory is good for low PlayoutsPerSecond")
fig.update_traces(zmin=-0.6, zmax=0.6)
fig.show()
```

```python
fig = explanation.visualize(8)
fig.update_layout(title_text="Interaction between PlayoutsPerSecond and p_exploration: low p_exploration is good for low PlayoutsPerSecond")
fig.update_traces(zmin=-0.3, zmax=0.3)
fig.show()
```

```python
fig = explanation.visualize(9)
fig.update_layout(title_text="Interaction between PlayoutsPerSecond and p_playout: Random200 is good for high PlayoutsPerSecond")
fig.update_traces(zmin=-0.2, zmax=0.2)
fig.show()
```

See @paulbkoch's [comment](https://www.kaggle.com/code/ambrosm/mcts-eda-which-makes-sense/comments#2996747) for directions on ExplainableBoostingRegressor usage.

# Gradient boosting with feature selection
We fit a LightGBM, XGBoost and CatBoost models to the data, using only features which have shown positive importance.

```python
lgbm_params_fast = {'learning_rate': 0.2, 'colsample_bytree': 0.7, 'verbose': 0}
lgbm_params = {'boosting_type': 'gbdt', 'learning_rate': 0.0365697430118668, 'n_estimators': 400, 'subsample': 0.9226632601335972, 'colsample_bytree': 0.8741993341198141, 'reg_lambda': 41.024122633807664, 'min_child_samples': 2, 'num_leaves': 934, 'objective': 'mse', 'subsample_freq': 1, 'verbose': -1} # 0.40867 5:59:53.572389 # v13 430 21 min
model = lightgbm.LGBMRegressor(**lgbm_params)
cross_validate_model(model, features=good_features, label=f"LightGBM", save_models=True)
oof_lightgbm = oof
# Fold 0: 0.432
# Fold 1: 0.437
# Fold 2: 0.448
# Fold 3: 0.421
# Fold 4: 0.424
# Overall RMSE=0.433 LightGBM   21 min
```

Plotting y_pred vs. y_true shows that there is a weak correlation between our features and the target:

```python
rng = np.random.default_rng()
plt.title('y_pred vs. y_true for LightGBM')
plt.scatter(oof, y + rng.normal(scale=0.005, size=len(y)), s=1, c='lightgreen')
plt.gca().set_aspect('equal')
plt.xlabel('y_pred')
plt.ylabel('y_true')
plt.show()
```

```python
features_xgb = ['AdvantageP1', 'PlayoutsPerSecond', 'p2_selection', 'p1_selection', 'DurationMoves', 'OutcomeUniformity', 'p2_exploration', 'p1_exploration', 'Asymmetric', 'DecisionMoves', 'GameTreeComplexity', 'DurationTurnsNotTimeouts', 'DurationTurnsStdDev', 'p2_playout', 'HopCaptureFrequency', 'Region', 'p1_playout', 'MovesPerSecond', 'StateTreeComplexity', 'DurationTurns', 'BranchingFactorAverage', 'BoardSitesOccupiedMaxIncrease', 'GreaterThanOrEqual', 'Completion', 'ConditionalStatement', 'DecisionFactorChangeSign', 'StepDecisionToEmptyFrequency', 'MorrisTiling', 'PlayersWithDirections', 'AsymmetricForces', 'SowWithEffect', 'HopDecisionEnemyToEmptyFrequency', 'BoardSitesOccupiedMaxDecrease', 'DecisionFactorVariance', 'DrawFrequency', 'Conjunction', 'BranchingFactorMaximum', 'DecisionFactorAverage', 'AnimalComponent', 'FromToDecisionBetweenContainersFrequency', 'NumOuterSites', 'StepDecisionFrequency', 'Capture', 'PieceNumberVariance', 'BranchingFactorChangeAverage', 'PieceNumberMaximum', 'BranchingFactorMedian', 'SpaceConditions', 'DecisionFactorMaxDecrease', 'BranchingFactorChangeMaxIncrease', 'BoardSitesOccupiedVariance', 'ReplacementCaptureFrequency', 'PieceNumberChangeAverage', 'NumCorners', 'DecisionFactorMaxIncrease', 'NumAdjacentDirections', 'DurationActions', 'MoveDistanceAverage', 'PieceNumberMedian', 'Line', 'NumVertices', 'DecisionFactorMaximum', 'DecisionFactorMedian', 'NumStartComponentsHand', 'MoveDistanceChangeSign', 'NumStartComponentsBoardPerPlayer', 'SowCaptureFrequency', 'NumPlayableSitesOnBoard', 'SameLayerDirection', 'BoardSitesOccupiedMaximum', 'NumStartComponentsBoard', 'MoveDistanceMedian', 'HopDecision', 'BranchingFactorChangeMaxDecrease', 'HopDecisionMoreThanOneFrequency', 'PieceDirection', 'MoveAgainFrequency', 'NumInnerSites', 'BoardCoverageUsed', 'BranchingFactorChangeSign', 'SetNextPlayerFrequency', 'DecisionFactorChangeLineBestFit', 'NumConvexCorners', 'NumDirections', 'HopDecisionEnemyToEmpty', 'Contains', 'Complement', 'PromotionDecisionFrequency', 'BoardCoverageFull', 'Disjunction', 'RemoveDecisionFrequency', 'NumStartComponentsHandPerPlayer', 'AddDecisionFrequency', 'BoardSitesOccupiedChangeSign', 'MovesDecision', 'CustodialCaptureFrequency', 'Drawishness', 'EliminatePiecesEndFrequency', 'MoveDistanceMaxDecrease', 'BranchingFactorVariance', 'TrackLoop', 'NumComponentsTypePerPlayer', 'EliminatePiecesWinFrequency', 'IsEmpty', 'NumComponentsType', 'LineWinFrequency', 'NumStartComponents', 'MancalaCircular', 'BoardCoverageDefault', 'NumRightSites', 'CaptureEnd', 'BoardSitesOccupiedMedian', 'HopCaptureMoreThanOneFrequency', 'NumStartComponentsPerPlayer', 'StepDecisionToFriend', 'SwapOption', 'MancalaBoard', 'BoardSitesOccupiedAverage', 'Connection', 'StepEffect', 'NumRows', 'NumPlayPhase', 'BoardSitesOccupiedChangeAverage', 'SlideDecisionToEmptyFrequency', 'MoveDistanceChangeAverage', 'PushEffectFrequency', 'ScoreDifferenceMaxDecrease', 'ScoreDifferenceMaxIncrease', 'PieceNumberChangeSign', 'SowCCW', 'Union', 'NoPieceNext', 'SingleSiteMoves', 'ScoreDifferenceChangeAverage', 'PolygonShape', 'SlideDecisionFrequency', 'SowCW', 'NumDiagonalDirections', 'MoveDistanceMaximum', 'NoTargetPieceEnd', 'Set', 'ReachWinFrequency', 'NumCentreSites', 'LineEndFrequency', 'CircleShape', 'Visual', 'Timeouts', 'MoveDistanceChangeLineBestFit', 'LeapDecisionToEmpty', 'StepDecisionToFriendFrequency', 'AsymmetricPiecesType', 'SpaceEnd', 'BackwardsDirection', 'NumPlayableSites', 'Odd', 'RemoveEffectFrequency', 'OrthogonalDirection', 'Hand', 'NumTopSites', 'NoMoves', 'Start', 'StepDecisionToEnemyFrequency', 'Sow', 'SowFrequency', 'Math', 'ScoreDifferenceVariance', 'RememberValues', 'p2_bounds', 'Repetition', 'Shape', 'FromToDecisionWithinBoardFrequency', 'SquareShape', 'NoTargetPiece', 'Multiplication', 'EliminatePiecesWin', 'CountPiecesNextComparison', 'TerritoryEndFrequency', 'LeapDecisionToEnemy', 'LesserThanOrEqual', 'Priority', 'SowSkip', 'ControlFlowStatement', 'MovesOperators', 'NineMensMorrisBoard', 'Negation', 'ShowPieceState', 'Addition', 'ScoringEndFrequency', 'NoMovesDraw', 'BoardSitesOccupiedChangeLineBestFit', 'LineEnd', 'p1_bounds', 'NoMovesLossFrequency', 'TriangleShape', 'CanNotMove', 'Comparison', 'AlquerqueTiling', 'Track', 'NoMovesMover', 'NoMovesNext', 'MovesNonDecision', 'DirectionCapture', 'SlideDecision', 'HopDecisionFrequency', 'GoStyle', 'NumColumns', 'HopCaptureMoreThanOne', 'SlideDecisionToEnemyFrequency', 'ScoreDifferenceMaximum', 'LesserThan', 'PieceNumberChangeLineBestFit', 'Draw', 'ScoreDifferenceChangeSign', 'LineOfSight', 'NumCells', 'NumConcaveCorners', 'FromToDecisionEnemyFrequency', 'RelativeDirections', 'MancalaTwoRows', 'Vertex', 'GroupEnd', 'HexTiling', 'Intersection', 'ChessStyle', 'Subtraction', 'BallComponent', 'Maximum']
```

```python
xgb_params = {'grow_policy': 'depthwise', 'learning_rate': 0.07274899580740063, 'n_estimators': 250, 'max_depth': 22, 'reg_lambda': 83.22774327040075, 'min_child_weight': 5.426022045934658, 'subsample': 0.9317467220757047, 'colsample_bytree': 0.5782694319945875, 'tree_method': 'hist', 'enable_categorical': True, 'gamma': 0} # 0.41354 6:01:24.074398 0.431 # v16 24 min
model = xgboost.XGBRegressor(**xgb_params)
cross_validate_model(model, features=features_xgb, label=f"XGBoost", save_models=True)
oof_xgb = oof
# Fold 0: 0.419
# Fold 1: 0.443
# Fold 2: 0.443
# Fold 3: 0.425
# Fold 4: 0.420
# Overall RMSE=0.430 XGBoost   50 min
```

```python
cb_params = {'grow_policy': 'SymmetricTree', 'n_estimators': 800, 'learning_rate': 0.08617153230342124, 'l2_leaf_reg': 1.0036132233587023, 'max_depth': 10, 'colsample_bylevel': 0.734514897063923, 'subsample': 0.994540769511675, 'random_strength': 0.5393480589423867, 'verbose': False}
model = catboost.CatBoostRegressor(**cb_params, cat_features= train_pd[good_features].columns[train_pd[good_features].dtypes == 'category'].values)
cross_validate_model(model, features=good_features, label=f"CatBoost", save_models=True)
```

# Other models

We fit three more models:
- Linear regression using only 82 features
- A Nyström kernel approximation
- A 60 nearest neighbors model

```python
linear_features = good_features[:80] + ['agent1', 'agent2']
model = make_pipeline(
    ColumnTransformer([('ohe',
                        OneHotEncoder(drop='first', sparse_output=False),
                        np.array(linear_features)[train_pd[linear_features].dtypes == 'category'])],
                      remainder='passthrough'), # makes 86 + 2 * 72 columns
    StandardScaler(),
    Ridge()
)
cross_validate_model(model, features=linear_features, label=f"Linear regression", save_models=True)
# Fold 0: 0.544
# Fold 1: 0.540
# Fold 2: 0.548
# Fold 3: 0.539
# Fold 4: 0.552
# Overall RMSE=0.544 Linear regression   1 min
```

```python
linear_features = good_features[:30] # more than 30..50 features don't help
model = make_pipeline(
    ColumnTransformer([('ohe',
                        OneHotEncoder(drop='first', sparse_output=False),
                        np.array(linear_features)[train_pd[linear_features].dtypes == 'category'])],
                      remainder='passthrough'),
    StandardScaler(),
    Nystroem(n_components=1000),
    Ridge()
)
cross_validate_model(model, features=linear_features, label=f"Kernel approximation", save_models=True)
```

```python
linear_features = good_features[:30] # more than 30..50 features don't help
n_neighbors = 60
model = make_pipeline(
    ColumnTransformer([('ohe',
                        OneHotEncoder(drop='first', sparse_output=False),
                        np.array(linear_features)[train_pd[linear_features].dtypes == 'category'])],
                      remainder='passthrough'), # makes 86 columns
    StandardScaler(),
    KNeighborsRegressor(n_neighbors=n_neighbors, weights='distance')
)
cross_validate_model(model, features=linear_features, label=f"KNeighborsRegressor {n_neighbors}", save_models=True)
# Fold 0: 0.489
# Fold 1: 0.501
# Fold 2: 0.495
# Fold 3: 0.472
# Fold 4: 0.474
# Overall RMSE=0.486 KNeighborsRegressor 60   5 min
```

# Ensembling

As usual, we determine ensemble weights with ridge regression.

```python
X = pd.DataFrame(oof_pred)
print(np.sqrt(np.square(y.reshape(-1, 1) - X).mean(axis=0))
      .rename('rmse')
      .round(3)
      .sort_values(ascending=False))
print()
ensemble = Ridge(fit_intercept=False)
cv_score = cross_val_score(ensemble,
                           X, y,
                           scoring='neg_root_mean_squared_error', 
                           cv=crossval_kf.split(train_pd, groups=train.select('GameRulesetName')))
print(f"Ensemble cross_val_score: {-cv_score.mean():.3f}")
ensemble.fit(X, y)
# print(f"Intercept: {ensemble.intercept_:.2f}")
print()
ensemble_weights = pd.Series(ensemble.coef_, index=X.columns, name='ensemble_weights')

oof_ensemble = cross_val_predict(ensemble,
                                 X, y,
                                 cv=crossval_kf.split(train_pd, groups=train.select('GameRulesetName')))

ensemble_weights.round(2)
```

# Conclusion

In this notebook, we have seen
- The features of the dataset describe games (hundreds of features) and players (four strategy elements per player).
- The test dataset uses other games than the training dataset. To prepare our models for unseen games, we cross-validate them with a GroupKFold.
- Only a third of the features seems to be useful. The other two thirds can be dropped.
- The experiment with InterpretML's ExplainableBoostingRegressor shows that we can distinguish three groups of influences on the target variable:
  1. The rules of the game can affect the target independently of the players, for instance if the rules of the game favor the first player against the second or if the game is prone to result in a draw.
  2. The player strategies can affect the target independently of the game: MAST in general is the better playout strategy than NST.
  3. The player strategies interact with the games: The ProgressiveHistory strategy seems to be particularly suitable for games with low PlayoutsPerSecond. In a game where the players cannot calculate arbitrarily many playouts without getting into a timeout, ProgressiveHistory has an advantage against the other strategies. These interactions between player strategy and game are what the competition is about.

# Preparing the submission

Don't submit this notebook — it is much too slow and would run into a timeout!

We save the models and the ensemble weights to disk so that we can load them in a separate notebook.

```python
with open('globals.py', 'w') as f:
    f.write(f"drop_columns = {drop_columns}\n")

with open(f"cat_mapping.pickle", "wb") as f:
    pickle.dump(cat_mapping, f)

with open(f"saved_models.pickle", "wb") as f:
    pickle.dump(saved_models, f)

with open(f"ensemble_weights.pickle", "wb") as f:
    pickle.dump(ensemble_weights, f)

directory = './'

# !cat globals.py
```

The following cell shows the submission code. It loads the saved models and sets up the inference server. The inference server collects the test predictions of all models and ensembles them.

```python
%%writefile inference.py

import os
import kaggle_evaluation.mcts_inference_server
import pickle
import numpy as np

exec(open(directory + 'globals.py', 'r').read())
exec(open(directory + 'preprocess.py', 'r').read())

with open(directory + f"cat_mapping.pickle", "rb") as f:
    cat_mapping = pickle.load(f)

with open(directory + f"saved_models.pickle", "rb") as f:
    saved_models = pickle.load(f)

with open(directory + f"ensemble_weights.pickle", "rb") as f:
    ensemble_weights = pickle.load(f)

print('Ensemble weights')
ew_sorted = ensemble_weights.round(2).sort_values(ascending=False)
for i in range(len(ew_sorted)):
    print(f"{ew_sorted.index[i]:50} {ew_sorted.iloc[i]:5.2f}")
print()


def predict(test, submission):
    """Return the predictions for the given test data.
    
    Computes the predictions of all models and blends them using the given weights.
    
    Parameters:
    test, submission: polars DataFrame
    
    Returns:
    submission: polars DataFrame
    """
    test = test.drop(drop_columns) # drop constant columns
    test_pd, _, _ = preprocess(test) # feature engineering
    
    y_pred = np.zeros(len(test))
    
    for label in ensemble_weights.index:
        features = saved_models[label]['features']
        model_list = saved_models[label]['model_list']
        y_pred_model = np.stack([model.predict(test_pd[features]) for model in model_list]).mean(axis=0) # prediction
        print(y_pred_model.round(3), label)
        y_pred += ensemble_weights[label] * y_pred_model.clip(-1, 1)
    
    print(y_pred.round(3))
    
    return submission.with_columns(pl.Series('utility_agent1', y_pred))

inference_server = kaggle_evaluation.mcts_inference_server.MCTSInferenceServer(predict)

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    inference_server.run_local_gateway(
        (
            '/kaggle/input/um-game-playing-strength-of-mcts-variants/test.csv',
            '/kaggle/input/um-game-playing-strength-of-mcts-variants/sample_submission.csv'
        )
    )
    print('Done')
```

```python
# If you want to test the submission code
# exec(open(directory + 'inference.py').read())
```