# MCTS | OOF Predictions as Features

- **Author:** Andreas Bisiadis
- **Votes:** 392
- **Ref:** andreasbis/mcts-oof-predictions-as-features
- **URL:** https://www.kaggle.com/code/andreasbis/mcts-oof-predictions-as-features
- **Last run:** 2024-11-23 12:33:00.053000

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
import plotly.graph_objects as go
```

```python
import plotly.io as pio
pio.renderers.default = 'iframe'
```

```python
import lightgbm as lgb
from catboost import CatBoostRegressor
import kaggle_evaluation.mcts_inference_server
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error as mse
```

<p style="background-color: #EADDCA; font-size: 300%; text-align: center; border-radius: 40px 40px; color: #D2B48C; font-weight: bold; font-family: 'Cinzel', serif; text-transform: uppercase; border: 4px solid #D2B48C;">configuration class</p>

```python
class CFG:
    
    importances_path = Path('/kaggle/input/mcts-gbdt-select-200-features/importances.csv')    
    train_path = Path('/kaggle/input/um-game-playing-strength-of-mcts-variants/train.csv')
    batch_size = 262144

    early_stop = 300
    n_splits = 5
    color = '#EADDCA'
    
    lgb_weight = 1.00
    lgb_params = {
        'objective': 'regression',
        'min_child_samples': 24,
        'num_iterations': 30000,
        'learning_rate': 0.03,
        'extra_trees': True,
        'reg_lambda': 0.8,
        'reg_alpha': 0.1,
        'num_leaves': 64,
        'metric': 'rmse',
        'device': 'cpu',
        'max_depth': 9,
        'max_bin': 128,
        'verbose': -1,
        'seed': 42
    }
    
    ctb_weight = 0.00
    ctb_params = {
        'loss_function': 'RMSE',
        'learning_rate': 0.03,
        'num_trees': 30000,
        'random_state': 42,
        'task_type': 'CPU',
        'reg_lambda': 0.8,
        'depth': 8
    }
```

<p style="background-color: #EADDCA; font-size: 300%; text-align: center; border-radius: 40px 40px; color: #D2B48C; font-weight: bold; font-family: 'Cinzel', serif; text-transform: uppercase; border: 4px solid #D2B48C;">feature engineering class</p>

```python
class FE:
    
    def __init__(self, batch_size):
        self.batch_size = batch_size
        
    def drop_cols(self, df, bad_cols=None): # bad_cols must be provided when processing the test data
        
        cols = ['Id', 
                'LudRules', 
                'EnglishRules',
                'num_wins_agent1',
                'num_draws_agent1',
                'num_losses_agent1']
        
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
        
    def apply_fe(self, path):
        
        df = pl.read_csv(path, batch_size=self.batch_size)
        
        df, bad_cols = self.drop_cols(df)
        df = self.cast_datatypes(df)
        
        cat_cols = [col for col in df.columns if df[col].dtype == pl.String]
        
        return df, bad_cols, cat_cols
```

```python
fe = FE(CFG.batch_size)
```

<p style="background-color: #EADDCA; font-size: 300%; text-align: center; border-radius: 40px 40px; color: #D2B48C; font-weight: bold; font-family: 'Cinzel', serif; text-transform: uppercase; border: 4px solid #D2B48C;">model development class</p>

```python
class MD:
    
    def __init__(self, 
                 importances_path, 
                 lgb_weight, 
                 lgb_params, 
                 ctb_weight,
                 ctb_params,
                 early_stop, 
                 n_splits,  
                 color):
        
        self.importances_path = importances_path
        self.lgb_weight = lgb_weight
        self.lgb_params = lgb_params
        self.ctb_weight = ctb_weight
        self.ctb_params = ctb_params
        self.early_stop = early_stop
        self.n_splits = n_splits
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
            title = f'{title} | Features: {features} | Cross-validation {metric} scores: {mean_score} ± {std_score}',
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
        
    def train_model(self, data, cat_cols, title):
        
        importances = pd.read_csv(self.importances_path)
        
        for col in cat_cols:
            data[col] = data[col].astype('category')
        
        cat_cols_copy = cat_cols.copy()
        
        X = data.drop(['utility_agent1'], axis=1)
        y = data['utility_agent1']
        group = data['GameRulesetName']
        
        cv = GroupKFold(n_splits=self.n_splits)
        
        models, scores = [], []
        
        # Initialize out-of-fold predictions array
        oof_preds = np.zeros(len(X))
        
        for fold, (train_index, valid_index) in enumerate(cv.split(X, y, group), 1):
            
            drop_features = importances['drop_features'].tolist()
            cat_cols = [col for col in cat_cols_copy if col not in drop_features]
                
            X_train = X.iloc[train_index].drop(drop_features, axis=1)
            X_valid = X.iloc[valid_index].drop(drop_features, axis=1)
            
            y_train = y.iloc[train_index]
            y_valid = y.iloc[valid_index]                   
                
            if title.startswith('LightGBM'):

                model = lgb.LGBMRegressor(**self.lgb_params)

                model.fit(X_train, y_train,
                          eval_set=[(X_valid, y_valid)],
                          eval_metric='rmse',
                          callbacks=[lgb.early_stopping(self.early_stop, verbose=0), lgb.log_evaluation(0)])
            
            elif title.startswith('CatBoost'):
            
                model = CatBoostRegressor(**self.ctb_params, verbose=0, cat_features=cat_cols)

                model.fit(X_train, y_train,
                          eval_set=(X_valid, y_valid),
                          early_stopping_rounds=self.early_stop, verbose=0)

            models.append(model)

            # Store out-of-fold predictions for this fold
            oof_preds[valid_index] = model.predict(X_valid)
            score = mse(y_valid, oof_preds[valid_index], squared=False)
            scores.append(score)
        
        self._plot_cv(scores, title, X_train.shape[1])
        
        return models, oof_preds
    
    def _infer_model(self, data, models):
        
        return np.mean([model.predict(data) for model in models], axis=0)
    
    def inference(self, data, cat_cols, lgb_models, ctb_models, lgb_models_oof, ctb_models_oof):

        importances = pd.read_csv(self.importances_path)
            
        drop_features = importances['drop_features'].tolist()
        data = data.drop(drop_features, axis=1)

        for col in cat_cols:
            data[col] = data[col].astype('category')
                
        data['lgb_oof_preds'] = self._infer_model(data, lgb_models)
        data['ctb_oof_preds'] = self._infer_model(data, ctb_models)
        
        lgb_preds = self._infer_model(data, lgb_models_oof)
        ctb_preds = self._infer_model(data, ctb_models_oof)
        
        return lgb_preds * self.lgb_weight + ctb_preds * self.ctb_weight
```

```python
md = MD(CFG.importances_path, 
        CFG.lgb_weight, 
        CFG.lgb_params, 
        CFG.ctb_weight,
        CFG.ctb_params, 
        CFG.early_stop, 
        CFG.n_splits, 
        CFG.color)
```

<p style="background-color: #EADDCA; font-size: 300%; text-align: center; border-radius: 40px 40px; color: #D2B48C; font-weight: bold; font-family: 'Cinzel', serif; text-transform: uppercase; border: 4px solid #D2B48C;">model development</p>

<div style="background-color: #EADDCA; border-radius: 40px 40px; border: 4px solid #D2B48C; padding: 20px;">
<li><span style="color:#D2B48C; font-size: 28px; font-weight: bold; font-family: 'Cinzel', serif;">Load and process train data</span></li>
<li><span style="color:#D2B48C; font-size: 28px; font-weight: bold; font-family: 'Cinzel', serif;">Train initial models to generate out-of-fold (OOF) predictions</span></li>
<li><span style="color:#D2B48C; font-size: 28px; font-weight: bold; font-family: 'Cinzel', serif;">Assign OOF predictions to train data</span></li>
<li><span style="color:#D2B48C; font-size: 28px; font-weight: bold; font-family: 'Cinzel', serif;">Train final models, with OOF predictions</span></li>
</div>

```python
train, bad_cols, cat_cols = fe.apply_fe(CFG.train_path)
train = train.to_pandas()
```

```python
lgb_models, lgb_oof_preds = md.train_model(train, cat_cols, title='LightGBM')
```

```python
ctb_models, ctb_oof_preds = md.train_model(train, cat_cols, title='CatBoost')
```

```python
train['lgb_oof_preds'] = lgb_oof_preds
train['ctb_oof_preds'] = ctb_oof_preds
```

```python
lgb_models_oof, _ = md.train_model(train, cat_cols, title='LightGBM (with OOF Preds)')
```

```python
ctb_models_oof, _ = md.train_model(train, cat_cols, title='CatBoost (with OOF Preds)')
```

<p style="background-color: #EADDCA; font-size: 300%; text-align: center; border-radius: 40px 40px; color: #D2B48C; font-weight: bold; font-family: 'Cinzel', serif; text-transform: uppercase; border: 4px solid #D2B48C;">inference</p>

```python
def predict(test, submission):
    
    test, _ = fe.drop_cols(test, bad_cols)
    test = fe.cast_datatypes(test)
    test = test.to_pandas()
    
    return submission.with_columns(pl.Series('utility_agent1', md.inference(test,
                                                                            cat_cols, 
                                                                            lgb_models, 
                                                                            ctb_models, 
                                                                            lgb_models_oof, 
                                                                            ctb_models_oof)))
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