# MCTS | LightGBM Baseline

- **Author:** Andreas Bisiadis
- **Votes:** 197
- **Ref:** andreasbis/mcts-lightgbm-baseline
- **URL:** https://www.kaggle.com/code/andreasbis/mcts-lightgbm-baseline
- **Last run:** 2024-11-23 12:13:33.397000

---

<p style="background-color: #EADDCA; font-size: 300%; text-align: center; border-radius: 40px 40px; color: #D2B48C; font-weight: bold; font-family: 'Cinzel', serif; text-transform: uppercase; border: 4px solid #D2B48C;">imports</p>

```python
import os
import sys
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')
```

```python
import numpy as np
import polars as pl
import pandas as pd
import plotly.colors as pc
import plotly.express as px
import plotly.graph_objects as go
```

```python
import plotly.io as pio
pio.renderers.default = 'iframe'
```

```python
pd.options.display.max_columns = None
```

```python
import lightgbm as lgb
import kaggle_evaluation.mcts_inference_server
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error as mse
```

<p style="background-color: #EADDCA; font-size: 300%; text-align: center; border-radius: 40px 40px; color: #D2B48C; font-weight: bold; font-family: 'Cinzel', serif; text-transform: uppercase; border: 4px solid #D2B48C;">configuration</p>

```python
class CFG:
    
    train_path = Path('/kaggle/input/um-game-playing-strength-of-mcts-variants/train.csv')
    batch_size = 262144

    do_feature_importance = False
    
    n_features = 250
    early_stop = 200
    n_splits = 5
    color = '#EADDCA'
    
    lgb_p = {
        'objective': 'regression',
        'num_iterations': 20000,
        'min_child_samples': 24,
        'learning_rate': 0.03,
        'extra_trees': True,
        'reg_lambda': 0.8,
        'reg_alpha': 0.1,
        'num_leaves': 64,
        'metric': 'rmse',
        'max_depth': 8,
        'device': 'cpu',
        'max_bin': 128,
        'verbose': -1,
        'seed': 42
    }
```

```python
class FE:
    
    def __init__(self, batch_size):
        self.batch_size = batch_size
        
    def drop_cols(self, df, bad_cols=None): # bad_cols must be provided when processing the test data
        
        cols = [
            'Id', 
            'LudRules', 
            'EnglishRules',
            'num_wins_agent1',
            'num_draws_agent1',
            'num_losses_agent1',

            # Features selected from feature importance - retained top 250 features
           'Draw', 'StepDecisionToFriendFrequency', 'BoardSitesOccupiedMaxDecrease', 'LineLoss', 'ThreeMensMorrisBoard', 'MancalaBoard', 'MoveDistanceVariance', 'ComponentStyle', 'SowRemoveFrequency', 'LineWin', 'MoveAgain', 'Tiling', 'MoveDistanceMedian', 'PromotionEffect', 'ScoreDifferenceMaxIncrease', 'VoteDecision', 'BackwardsDirection', 'Visual', 'EliminatePiecesEnd', 'CountPiecesMoverComparison', 'NoMovesEnd', 'PieceNumberChangeLineBestFit', 'ScoreDifferenceMaximum', 'DiceD6', 'RememberValues', 'FromToDecisionFrequency', 'ConnectionWinFrequency', 'LineEnd', 'ConnectionEndFrequency', 'SowFrequency', 'DiagonalDirection', 'HopDecisionEnemyToEmptyFrequency', 'ForEachPiece', 'ForwardLeftDirection', 'LineOfSight', 'MaxMovesInTurn', 'DecisionFactorChangeNumTimes', 'PassDecisionFrequency', 'BoardSitesOccupiedMaxIncrease', 'Implementation', 'MoveDistanceMaxDecrease', 'AddEffectFrequency', 'ScoringWinFrequency', 'HopCaptureMoreThanOneFrequency', 'ScoreDifferenceVariance', 'Maximum', 'PieceValue', 'MoveDistanceChangeNumTimes', 'HopDecisionMoreThanOneFrequency', 'CustodialCaptureFrequency', 'PieceNumberMaxIncrease', 'PieceRotation', 'SwapPlayersEffect', 'AlquerqueBoardWithTwoTriangles', 'SquarePyramidalShape', 'MoveDistanceChangeAverage', 'NoMovesLoss', 'DiamondShape', 'DiceD4', 'ScoreDifferenceChangeAverage', 'Division', 'Even', 'NumRows', 'Minimum', 'ChessComponent', 'ForwardDirection', 'PolygonShape', 'InitialScore', 'OppositeDirection', 'NumStartComponentsHand', 'NumBottomSites', 'NumCorners', 'NumTopSites', 'Directions', 'SowCCW', 'MoveDistanceMaximum', 'ScoreDifferenceChangeSign', 'Modulo', 'MoveDistanceMaxIncrease', 'EncloseCaptureFrequency', 'NumPlayPhase', 'Pattern', 'Edge', 'PromotionEffectFrequency', 'AbsoluteDirections', 'HopDecisionFrequency', 'StepEffect', 'SowBacktracking', 'NumStartComponentsHandPerPlayer', 'PloyComponent', 'BoardStyle', 'ReachWin', 'NumCentreSites', 'StarBoard', 'MoveDistanceChangeSign', 'SemiRegularTiling', 'SowCapture', 'BoardSitesOccupiedChangeAverage', 'DiceD2', 'ReplacementCaptureFrequency', 'AlquerqueBoardWithFourTriangles', 'Math', 'NoTargetPiece', 'ConnectionLoss', 'ScoreDifferenceAverage', 'CanNotMove', 'BoardSitesOccupiedChangeLineBestFit', 'NineMensMorrisBoard', 'NoTargetPieceEnd', 'StepDecisionToEmpty', 'ConnectionWin', 'SowBacktrackingFrequency', 'StepDecisionToFriend', 'StepDecisionToEnemy', 'ConcentricTiling', 'FromToDecisionEnemyFrequency', 'NumConvexCorners', 'BackwardLeftDirection', 'PushEffect', 'Stack', 'GroupEndFrequency', 'Track', 'TaflComponent', 'TrackLoop', 'PatternEnd', 'MoveDistanceChangeLineBestFit', 'FromToDecisionEnemy', 'GroupEnd', 'ScoreDifferenceMaxDecrease', 'SwapPlayersDecision', 'PawnComponent', 'FillEndFrequency', 'ShootDecision', 'SlideDecisionToEnemyFrequency', 'Style', 'NoPieceMover', 'ScoringEnd', 'PromotionDecision', 'QueenComponent', 'MancalaThreeRows', 'NoOwnPiecesEndFrequency', 'FlipFrequency', 'NoTargetPieceEndFrequency', 'Flip', 'SowWithEffect', 'SlideEffect', 'ConnectionLossFrequency', 'HopDecisionFriendToEnemy', 'Moves', 'SwapPiecesDecisionFrequency', 'Efficiency', 'VoteEffect', 'MovesEffects', 'EliminatePiecesLoss', 'BackwardDirection', 'ScoreDifferenceMedian', 'RollFrequency', 'SameLayerDirection', 'PassEffect', 'ForgetValues', 'HopDecisionFriendToEnemyFrequency', 'Repetition', 'HopDecisionEnemyToEnemy', 'AlquerqueBoardWithOneTriangle', 'ScoreDifferenceChangeLineBestFit', 'SetValueFrequency', 'FairyChessComponent', 'ProgressCheck', 'SurakartaStyle', 'LeapDecisionToEnemyFrequency', 'PushEffectFrequency', 'ScoringWin', 'Tile', 'LeapDecision', 'ConnectionEnd', 'Piece', 'ShowPieceValue', 'Loop', 'SiteState', 'SurroundCapture', 'LineLossFrequency', 'Fill', 'CheckmateWinFrequency', 'LeapDecisionFrequency', 'LeapDecisionToEmptyFrequency', 'XiangqiStyle', 'HopDecisionEnemyToEnemyFrequency', 'CheckmateFrequency', 'PromotionDecisionFrequency', 'InternalCounter', 'SameDirection', 'HopDecisionFriendToFriendFrequency', 'GraphStyle', 'NumComponentsTypePerPlayer', 'FromToDecisionFriend', 'CaptureSequence', 'NumComponentsType', 'NoBoard', 'CaptureSequenceFrequency', 'InterveneCapture', 'NoProgressEnd', 'Territory', 'BoardSitesOccupiedChangeNumTimes', 'ScoringLoss', 'MaxDistance', 'SetCount', 'TableStyle', 'PenAndPaperStyle', 'KnightComponent', 'NoTargetPieceWin', 'NoProgressEndFrequency', 'ByDieMove', 'LargePiece', 'KingComponent', 'NoOwnPiecesWinFrequency', 'InitialCost', 'ShowPieceState', 'SetValue', 'DirectionCapture', 'SumDice', 'Boardless', 'BishopComponent', 'FillWinFrequency', 'ScoringDraw', 'ThreeMensMorrisBoardWithTwoTriangles', 'BranchingFactorChangeNumTimesn', 'LeapEffect', 'ProposeDecision', 'BackgammonStyle', 'StrategoComponent', 'Absolute', 'NumPhasesBoard', 'HopDecisionFriendToEmptyFrequency', 'PositionalSuperko', 'NoTargetPieceWinFrequency', 'ProposeEffect', 'ScoringLossFrequency', 'TurnKo', 'ProposeDecisionFrequency', 'NumDice', 'NoProgressDrawFrequency', 'NoOwnPiecesEnd', 'HopDecisionFriendToEmpty', 'EliminatePiecesLossFrequency', 'GroupWin', 'SetInternalCounter', 'NumContainers', 'Dice', 'MancalaSixRows', 'SetCountFrequency', 'LoopEndFrequency', 'NoMovesDrawFrequency', 'RotationalDirection', 'InitialRandomPlacement', 'ShogiStyle', 'CheckmateWin', 'GroupWinFrequency', 'LineDraw', 'NoOwnPiecesLoss', 'StateType', 'SlideDecisionToFriend', 'RookComponent', 'SwapPiecesDecision', 'Threat', 'LeapDecisionToEmpty', 'FillEnd', 'XiangqiComponent', 'EliminatePiecesDraw', 'SlideDecisionToEnemy', 'JanggiComponent', 'SurroundCaptureFrequency', 'PieceNumberChangeNumTimes', 'ShibumiStyle', 'NumLayers', 'NoOwnPiecesLossFrequency', 'FillWin', 'AlquerqueBoardWithEightTriangles', 'NoOwnPiecesWin', 'ReachLoss', 'NumOffDiagonalDirections', 'LeapDecisionToEnemy', 'NoProgressDraw', 'StarShape', 'GroupDraw', 'KintsBoard', 'ShogiComponent', 'MancalaCircular', 'FortyStonesWithFourGapsBoard', 'EliminatePiecesDrawFrequency', 'Checkmate', 'InterveneCaptureFrequency', 'SowProperties', 'VisitedSites', 'RotationDecision', 'JanggiStyle', 'LoopWin', 'PachisiBoard', 'AutoMove', 'ReachLossFrequency', 'SpiralShape', 'Cooperation', 'GroupLoss', 'SwapOption', 'SowOriginFirst', 'StackState', 'SpiralTiling', 'PathExtent', 'SetRotation', 'CircleTiling', 'SetRotationFrequency', 'TerritoryEnd', 'ReachDrawFrequency', 'DirectionCaptureFrequency', 'TerritoryWin', 'AsymmetricForces', 'TerritoryWinFrequency', 'PathExtentLoss', 'ProposeEffectFrequency', 'Sow', 'AsymmetricPiecesType', 'BackwardRightDirection', 'MancalaStyle', 'ForwardRightDirection', 'LeftwardsDirection', 'LoopEnd', 'RotationDecisionFrequency', 'RightwardsDirection', 'LeftwardDirection', 'LoopWinFrequency', 'LoopLoss', 'PatternEndFrequency', 'PieceDirection', 'Team', 'ShootDecisionFrequency', 'Roll', 'PatternWin', 'PatternWinFrequency', 'PathExtentEnd', 'PathExtentWin', 'ReachDraw'
        ]
        
        df = df.drop([col for col in cols if col in df.columns])
        
        # Select and drop columns with 100% null values
        df = df.drop([col for col in df.columns if df.select(pl.col(col).null_count()).item() == df.height])
        
        if bad_cols is None:
            
            # Select (if not provided) columns with only one unique value
            bad_cols = [col for col in df.columns if df.select(pl.col(col).n_unique()).item() == 1]
            
        df = df.drop(bad_cols)
        
        return df, bad_cols
    
    def cast_datatypes(self, df):
        
        # Set datatype for categorical columns
        cat_cols = ['GameRulesetName', 'agent1', 'agent2']
        df = df.with_columns([pl.col(col).cast(pl.String) for col in cat_cols])   
        
        # Find numeric columns
        for col in df.columns:
            if col not in cat_cols:
            
                # Set datatype for a numeric column as per the datatype of the first non-null item
                val = df.select(pl.col(col).drop_nulls().first()).item()
                df = df.with_columns(pl.col(col).cast(pl.Int16) if isinstance(val, int) else pl.col(col).cast(pl.Float32))   
            
        return df     
    
    def extract_group(self, df):
        
        group = df.select(pl.col('GameRulesetName')).to_pandas()
        df = df.drop('GameRulesetName')
        
        return df, group
    
    def info(self, df):
        
        print(f'Shape: {df.shape}')   
        mem = df.estimated_size() / 1024**2
        print('Memory usage: {:.2f} MB\n'.format(mem))
        
    def apply_fe(self, path):
        
        df = pl.read_csv(path, batch_size=self.batch_size)
        
        df, bad_cols = self.drop_cols(df)
        df = self.cast_datatypes(df)
        df, group = self.extract_group(df)
        
        self.info(df)
        
        cat_cols = [col for col in df.columns if df[col].dtype == pl.String]
        
        return df, bad_cols, cat_cols, group
```

```python
fe = FE(CFG.batch_size)
```

```python
class EDA:
    
    def __init__(self, df, color):
        self.df = df  
        self.color = color  

    def template(self, fig, title):
        
        fig.update_layout(
            title=title,
            title_x=0.5, 
            plot_bgcolor='rgba(40, 40, 43, 1)',  
            paper_bgcolor='rgba(40, 40, 43, 1)', 
            font=dict(color=self.color),
            margin=dict(l=90, r=90, t=90, b=90), 
            height=900
        )
        
        return fig
    
    def target_distribution(self):
        
        target_distribution = self.df['utility_agent1'].value_counts().sort_index()

        fig = px.histogram(
            self.df,
            x='utility_agent1',
            nbins=50,
            title='Distribution of Agent 1 Utility',
            color_discrete_sequence=[self.color]
        )

        fig.update_layout(
            xaxis_title='Utility of Agent 1',
            yaxis_title='Count',
            bargap=0.1,
            xaxis=dict(gridcolor='grey'),
            yaxis=dict(gridcolor='grey', zerolinecolor='grey')
        )
        fig.update_traces(hovertemplate='Utility: %{x:.3f}<br>Count: %{y:,}')
        
        fig = self.template(fig, 'Distribution of Agent 1 Utility')
        fig.show()
    
    def value_distribution(self):
        
        binary_cols = []
        other_cols = []

        for col in self.df.columns:
            
            if self.df[col].nunique() == 2:
                binary_cols.append(col)
                
            elif self.df[col].nunique() > 2:
                other_cols.append(col)

        labels = ['2 values', '>2 values']
        values = [len(binary_cols), len(other_cols)]
        percentages = [round(count / len(self.df.columns) * 100) for count in values]
        hover_text = [f'Case: {label}<br>Count: {count}<br>Percent: {percentage}%' 
                      for label, count, percentage in zip(labels, values, percentages)]

        fig = px.pie(
            values=values,
            names=labels,
            title='Distribution of Column Types',
            color_discrete_sequence=px.colors.sequential.Redor,
            custom_data=[hover_text]
        )

        fig.update_traces(hovertemplate='%{customdata[0]}<extra></extra>')
        
        fig = self.template(fig, 'Distribution of Column Types')
        fig.show()
```

```python
train_data, _, _, _ = fe.apply_fe(CFG.train_path)
train_data = train_data.to_pandas()
```

```python
eda = EDA(train_data, CFG.color)
```

```python
class MD:
    
    def __init__(self, n_features, early_stop, n_splits, lgb_p, color):
        self.n_features = n_features
        self.early_stop = early_stop
        self.n_splits = n_splits
        self.lgb_p = lgb_p
        self.color = color
        
    def _plot_cv(self, fold_scores, title, features, metric='RMSE'):
        
        fold_scores = [round(score, 3) for score in fold_scores]
        mean_score = round(np.mean(fold_scores), 3)
        std_score = round(np.std(fold_scores), 3)

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x = list(range(1, len(fold_scores) + 1)),
            y = fold_scores,
            mode = 'markers', 
            name = 'Fold Scores',
            marker = dict(size = 27, color=self.color, symbol='diamond'),
            text = [f'{score:.3f}' for score in fold_scores],
            hovertemplate = 'Fold %{x}: %{text}<extra></extra>',
            hoverlabel=dict(font=dict(size=18))  
        ))

        fig.add_trace(go.Scatter(
            x = [1, len(fold_scores)],
            y = [mean_score, mean_score],
            mode = 'lines',
            name = f'Mean: {mean_score:.3f}',
            line = dict(dash = 'dash', color = '#FFBF00'),
            hoverinfo = 'none'
        ))
        
        fig.update_layout(
            title = f'{title} with {features} features | Cross-validation {metric} scores: {mean_score} ± {std_score}',
            xaxis_title = 'Fold',
            yaxis_title = f'{metric} Score',
            plot_bgcolor = 'rgba(40, 40, 43, 1)',  
            paper_bgcolor = 'rgba(40, 40, 43, 1)',
            font = dict(color=self.color), 
            xaxis = dict(
                gridcolor = 'grey',
                tickmode = 'linear',
                tick0 = 1,
                dtick = 1,
                range = [0.5, len(fold_scores) + 0.5],
                zerolinecolor = 'grey'
            ),
            yaxis = dict(
                gridcolor = 'grey',
                zerolinecolor = 'grey'
            )
        )
        
        fig.show()
        
    def train_lgb(self, data, cat_cols, group, title):
        
        X = data.drop(['utility_agent1'], axis=1)
        y = data['utility_agent1']
        
        for col in cat_cols:
            X[col] = X[col].astype('category')
        
        cv = GroupKFold(self.n_splits)
        
        models, scores = [], []
        oof_preds = np.zeros(len(X))
        
        for fold, (train_index, valid_index) in enumerate(cv.split(X, y, group)):
            
            X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
            y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]
            
            model = lgb.LGBMRegressor(**self.lgb_p)
            
            model.fit(X_train, y_train,
                      eval_set=[(X_valid, y_valid)],
                      eval_metric='rmse',
                      callbacks=[lgb.early_stopping(self.early_stop, verbose=0), lgb.log_evaluation(0)])
    
            models.append(model)
            
            oof_preds[valid_index] = model.predict(X_valid)
            score = mse(y_valid, oof_preds[valid_index], squared=False)
            scores.append(score)
        
        self._plot_cv(scores, title, X_train.shape[1])
        
        return models

    def infer_lgb(self, data, cat_cols, models):

        for col in cat_cols:
            data[col] = data[col].astype('category')

        return np.mean([model.predict(data) for model in models], axis=0)
    
    def feature_importance(self, data, cat_cols, group, title):
        
        models = self.train_lgb(data, cat_cols, group, title)
        
        feature_importances = np.zeros(len(data.columns) - 1)
        for model in models:
            feature_importances += model.feature_importances_ / len(models)
        
        feature_importance = pd.DataFrame({
            'feature': [col for col in data.columns if col != 'utility_agent1'],
            'importance': feature_importances
        })
        
        feature_importance = feature_importance.sort_values('importance', ascending=False).reset_index(drop=True)
        display(feature_importance)
        
        drop_features = feature_importance.loc[self.n_features:, 'feature'].tolist()
        
        return drop_features
```

```python
md = MD(CFG.n_features, CFG.early_stop, CFG.n_splits, CFG.lgb_p, CFG.color)
```

<p style="background-color: #EADDCA; font-size: 300%; text-align: center; border-radius: 40px 40px; color: #D2B48C; font-weight: bold; font-family: 'Cinzel', serif; text-transform: uppercase; border: 4px solid #D2B48C;">exploratory data analysis</p>

```python
eda.target_distribution()
```

```python
eda.value_distribution()
```

<p style="background-color: #EADDCA; font-size: 300%; text-align: center; border-radius: 40px 40px; color: #D2B48C; font-weight: bold; font-family: 'Cinzel', serif; text-transform: uppercase; border: 4px solid #D2B48C;">feature importance</p>

```python
if CFG.do_feature_importance:
    
    train, _, cat_cols, group = fe.apply_fe(CFG.train_path)
    train = train.to_pandas()
    display(train_data.head())
```

```python
if CFG.do_feature_importance:
    
    drop_features = md.feature_importance(train, cat_cols, group, 'LightGBM')
```

```python
if CFG.do_feature_importance:
    
    del train, cat_cols, group
    print(drop_features)
```

<p style="background-color: #EADDCA; font-size: 300%; text-align: center; border-radius: 40px 40px; color: #D2B48C; font-weight: bold; font-family: 'Cinzel', serif; text-transform: uppercase; border: 4px solid #D2B48C;">model development</p>

```python
train, bad_cols, cat_cols, group = fe.apply_fe(CFG.train_path)
train = train.to_pandas()
display(train_data.head())
```

```python
lgb_models = md.train_lgb(train, cat_cols, group, 'LightGBM')
```

<p style="background-color: #EADDCA; font-size: 300%; text-align: center; border-radius: 40px 40px; color: #D2B48C; font-weight: bold; font-family: 'Cinzel', serif; text-transform: uppercase; border: 4px solid #D2B48C;">inference</p>

```python
def predict(test, submission):
    
    test, _ = fe.drop_cols(test, bad_cols)
    test = fe.cast_datatypes(test)
    test, _ = fe.extract_group(test)
    test = test.to_pandas()
    
    return submission.with_columns(pl.Series('utility_agent1', md.infer_lgb(test, cat_cols, lgb_models)))
```

```python
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
```