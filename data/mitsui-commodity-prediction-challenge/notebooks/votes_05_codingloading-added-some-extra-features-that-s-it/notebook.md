# Added Some  Extra Features That's it

- **Author:** YouTuber @DataScience
- **Votes:** 272
- **Ref:** codingloading/added-some-extra-features-that-s-it
- **URL:** https://www.kaggle.com/code/codingloading/added-some-extra-features-that-s-it
- **Last run:** 2025-07-25 13:14:30.357000

---

```python
# ====================================================
# Full Mitsui Commodity Prediction Pipeline (Enhanced)
# Training + Inference + Feature Engineering
# ====================================================

import os, gc, warnings, random, pickle
from pathlib import Path
import numpy as np
import pandas as pd
import polars as pl
from tqdm.auto import tqdm
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import KFold
import torch

import kaggle_evaluation.mitsui_inference_server

# ====================================================
# Config
# ====================================================
class Config:
    AUTHOR = 'mitsui_ai'
    VERSION = 2
    SEED = 42
    N_FOLDS = 3
    BOOSTERS = ['lgbm', 'xgb', 'cat']
    MAX_ROUNDS = 2500
    EARLY_STOP = 100
    VERBOSE = 1
    DATA_DIR = Path('/kaggle/input/mitsui-commodity-prediction-challenge')
    MODEL_DIR = Path('./models'); os.makedirs(MODEL_DIR, exist_ok=True)
    OOF_DIR = Path('./oof'); os.makedirs(OOF_DIR, exist_ok=True)
    TARGET_COUNT = 424
    FEATURES_TO_ADD = ['target_id']

    LGBM_PARAMS = {
        'objective': 'regression', 'metric': 'rmse',
        'learning_rate': 0.005, 'num_leaves': 8, 'seed': SEED,
        'device': 'gpu', 'gpu_platform_id': 0, 'gpu_device_id': 0,
    }

    XGB_PARAMS = {
        'objective': 'reg:squarederror', 'eval_metric': 'rmse',
        'learning_rate': 0.005, 'max_depth': 4, 'random_state': SEED,
        'tree_method': 'gpu_hist', 'predictor': 'gpu_predictor',
    }

    CAT_PARAMS = {
        'loss_function': 'RMSE', 'learning_rate': 0.005,
        'iterations': MAX_ROUNDS, 'depth': 4,
        'random_seed': SEED, 'verbose': False,
        'task_type': 'GPU', 'devices': '0:1',
    }

# ====================================================
# Seed and Utility
# ====================================================
def set_seed(seed=Config.SEED):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
set_seed()

# ====================================================
# Feature Engineering
# ====================================================
def add_features(df):
    df['dayofweek'] = df['date_id'] % 7
    df['month'] = (df['date_id'] // 30) % 12
    df['quarter'] = df['month'] // 3
    df['day_of_month'] = df['date_id'] % 30

    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
    df['is_month_start'] = (df['day_of_month'] == 0).astype(int)
    df['is_month_end'] = (df['day_of_month'] == 29).astype(int)

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    return df


# ====================================================
# Train Booster
# ====================================================
def train_model(booster, x_tr, y_tr, x_val, y_val):
    y_tr = np.nan_to_num(y_tr, nan=0.0)
    y_val = np.nan_to_num(y_val, nan=0.0)

    if booster == 'lgbm':
        train_set = lgb.Dataset(x_tr, y_tr)
        val_set = lgb.Dataset(x_val, y_val)
        model = lgb.train(
            Config.LGBM_PARAMS, train_set,
            num_boost_round=Config.MAX_ROUNDS,
            valid_sets=[val_set],
            callbacks=[
                lgb.early_stopping(Config.EARLY_STOP),
                lgb.log_evaluation(Config.VERBOSE)
            ]
        )
        return model, model.predict(x_val)

    elif booster == 'xgb':
        train_d = xgb.DMatrix(x_tr, label=y_tr)
        valid_d = xgb.DMatrix(x_val, label=y_val)
        model = xgb.train(
            Config.XGB_PARAMS, train_d,
            num_boost_round=Config.MAX_ROUNDS,
            evals=[(valid_d, 'eval')],
            early_stopping_rounds=Config.EARLY_STOP,
            verbose_eval=Config.VERBOSE
        )
        return model, model.predict(xgb.DMatrix(x_val))

    elif booster == 'cat':
        train_pool = Pool(x_tr, label=y_tr)
        val_pool = Pool(x_val, label=y_val)
        model = CatBoostRegressor(**Config.CAT_PARAMS)
        model.fit(train_pool, eval_set=val_pool,
                  early_stopping_rounds=Config.EARLY_STOP)
        return model, model.predict(x_val)

# ====================================================
# Training CV Wrapper
# ====================================================
def run_cv(booster, df, features):
    df = df.copy()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=['target'], inplace=True)

    oof_preds = np.zeros(len(df))
    kf = KFold(n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED)

    for fold, (train_idx, val_idx) in enumerate(kf.split(df)):
        x_tr = df.iloc[train_idx][features]
        y_tr = df.iloc[train_idx]['target']
        x_val = df.iloc[val_idx][features]
        y_val = df.iloc[val_idx]['target']

        model, val_preds = train_model(booster, x_tr, y_tr, x_val, y_val)
        oof_preds[val_idx] = val_preds

        with open(Config.MODEL_DIR / f'{booster}_fold{fold}.pkl', 'wb') as f:
            pickle.dump(model, f)

        del model, x_tr, y_tr, x_val, y_val
        gc.collect()

    np.save(Config.OOF_DIR / f'oof_{booster}.npy', oof_preds)
```

```python
# ====================================================
# Load Data
# ====================================================
train_df = pl.read_csv(Config.DATA_DIR / 'train.csv').to_pandas()
label_df = pl.read_csv(Config.DATA_DIR / 'train_labels.csv').to_pandas()

features = list(train_df.columns[1:]) + Config.FEATURES_TO_ADD

df_all = []
for j, col in enumerate(label_df.columns[1:]):
    temp = train_df.copy()
    temp['target'] = label_df[col]
    temp['target_id'] = j
    temp = add_features(temp)
    temp = temp.dropna(subset=['target'])
    df_all.append(temp)

train_full = pd.concat(df_all, axis=0).reset_index(drop=True)

# ====================================================
# Train All Boosters
# ====================================================
for booster in Config.BOOSTERS:
    print(f"\n[Training {booster.upper()}]")
    run_cv(booster, train_full, features + [
        'dayofweek', 'month', 'quarter', 'day_of_month',
        'is_weekend', 'is_month_start', 'is_month_end'
    ])

# ====================================================
# Inference Wrapper
# ====================================================
model_registry = {}
for booster in Config.BOOSTERS:
    model_registry[booster] = []
    for fold in range(Config.N_FOLDS):
        with open(f'/kaggle/working/models/{booster}_fold{fold}.pkl', 'rb') as f:
            model_registry[booster].append(pickle.load(f))

def predict(test, *_):
    test_df = test.to_pandas()
    for col in test_df.columns:
        if test_df[col].dtype == 'object':
            test_df[col] = np.nan
    test_df = add_features(test_df)

    test_blocks = []
    for tid in range(Config.TARGET_COUNT):
        temp = test_df.copy()
        temp['target_id'] = tid
        test_blocks.append(temp)

    x_test = pd.concat(test_blocks, axis=0)
    features_final = list(train_df.columns[1:]) + Config.FEATURES_TO_ADD + [
        'dayofweek', 'month', 'quarter', 'day_of_month',
        'is_weekend', 'is_month_start', 'is_month_end'
]
    x_test = x_test[features_final]

    preds = []
    for booster in Config.BOOSTERS:
        for model in model_registry[booster]:
            if booster == 'lgbm':
                preds.append(model.predict(x_test, predict_disable_shape_check=True))
            elif booster == 'xgb':
                preds.append(model.predict(xgb.DMatrix(x_test)))
            elif booster == 'cat':
                preds.append(model.predict(x_test))
    preds = np.mean(np.array(preds), axis=0)

    return pl.DataFrame({f'target_{i}': preds[i] for i in range(Config.TARGET_COUNT)})

# ====================================================
# Serve Model (Locally or on Hidden Test)
# ====================================================
inference_server = kaggle_evaluation.mitsui_inference_server.MitsuiInferenceServer(predict)
if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    inference_server.run_local_gateway((Config.DATA_DIR,))
```