# Mitsui Train+CV+Predict

- **Author:** Yusuke Togashi
- **Votes:** 274
- **Ref:** yusuketogashi/mitsui-train-cv-predict
- **URL:** https://www.kaggle.com/code/yusuketogashi/mitsui-train-cv-predict
- **Last run:** 2025-07-29 00:54:45.867000

---

```python
import os
import numpy as np
import pandas as pd
import polars as pl
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from lightgbm import LGBMRegressor
from sklearn.model_selection import TimeSeriesSplit

# ==============================================================================
# CONFIGURATION
# ==============================================================================
TRAIN     = False    # True: train + CV + save models, False: load models + predict
DO_CV     = False    # Whether to run CV (only when TRAIN=True)
CV_SPLITS = 3        # Number of folds for TimeSeriesSplit
SEED      = 42

# detect Kaggle environment
IS_KAGGLE = os.getenv('KAGGLE_KERNEL_RUN_TYPE') is not None
if IS_KAGGLE:
    import kaggle_evaluation.mitsui_inference_server
    DATA_PATH        = '/kaggle/input/mitsui-commodity-prediction-challenge/'
    # MODEL_INPUT_DIR  = '/kaggle/input/model-uskt/'
    MODEL_INPUT_DIR = '/kaggle/input/model2-uskt'
    MODEL_OUTPUT_DIR = '/kaggle/working/model'
else:
    DATA_PATH        = './'
    MODEL_INPUT_DIR  = './model/'
    MODEL_OUTPUT_DIR = './model'
os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

sns.set_theme(style="darkgrid")

# ==============================================================================
# LOAD DATA
# ==============================================================================
print("Loading data…")
train_df     = pl.read_csv(os.path.join(DATA_PATH, 'train.csv'))
test_df      = pl.read_csv(os.path.join(DATA_PATH, 'test.csv'))
train_labels = pl.read_csv(os.path.join(DATA_PATH, 'train_labels.csv'))

print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}, Labels shape: {train_labels.shape}")

# ==============================================================================
# FEATURE / TARGET DEFINITIONS
# ==============================================================================
FEATURE_NAMES = [c for c in train_df.columns if c not in ('date_id','row_id','is_scored')]
TARGETS       = [f"target_{i}" for i in range(424)]

# ==============================================================================
# PREPROCESSING & FEATURE ENGINEERING
# ==============================================================================
def preprocess_for_lgbm(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert object-type columns to numeric and pandas.Categorical to integer codes.
    """
    df = df.copy()
    # Cast known object columns
    for col in [
        'US_Stock_GOLD_adj_open','US_Stock_GOLD_adj_high',
        'US_Stock_GOLD_adj_low','US_Stock_GOLD_adj_close',
        'US_Stock_GOLD_adj_volume'
    ]:
        if col in df and df[col].dtype == 'object':
            df[col] = pd.to_numeric(df[col], errors='coerce')
    # Convert pandas.Categorical to codes
    for cat in df.select_dtypes(['category']).columns:
        df[cat] = df[cat].cat.codes
    return df

def create_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Placeholder for feature engineering; returns input unchanged.
    """
    return df.clone()

# ==============================================================================
# MODEL STORAGE
# ==============================================================================
models = {}

# ==============================================================================
# TRAINING MODE
# ==============================================================================
if TRAIN:
    # Prepare full training arrays
    X_all = create_features(train_df).to_pandas()[FEATURE_NAMES]
    y_all = train_labels.to_pandas()[TARGETS]

    # Train one LightGBM per target
    for idx, tgt in enumerate(TARGETS):
        print(f"Training {tgt}…")
        y = y_all[tgt]
        mask = ~y.isna()
        X_tr = X_all.loc[mask].copy()
        y_tr = y.loc[mask]
        if len(y_tr) == 0:
            models[tgt] = None
            continue
        X_tr['target_name_encoded'] = idx
        X_tr = preprocess_for_lgbm(X_tr)

        m = LGBMRegressor(
            n_estimators=100,
            learning_rate=0.05,
            random_state=SEED,
            verbose=-1
        )
        m.fit(X_tr, y_tr)
        models[tgt] = m
        joblib.dump(m, os.path.join(MODEL_OUTPUT_DIR, f"{tgt}_model.pkl"))

    print("All models trained and saved.")

    # --------------------------------------------------------------------
    # OPTIONAL CROSS-VALIDATION (RankCorr-Sharpe)
    # --------------------------------------------------------------------
    if DO_CV:
        print(f"\nRunning TimeSeriesSplit CV with {CV_SPLITS} folds…")
        SOL_FILL = -999999
        sol = y_all.fillna(SOL_FILL)
        oof = pd.DataFrame(index=sol.index, columns=TARGETS)

        tscv = TimeSeriesSplit(n_splits=CV_SPLITS)
        for fold, (tr_idx, va_idx) in enumerate(tscv.split(X_all)):
            print(f" CV Fold {fold+1}/{CV_SPLITS}")
            X_tr, X_va = X_all.iloc[tr_idx], X_all.iloc[va_idx]
            y_tr       = sol.iloc[tr_idx]
            for idx, tgt in enumerate(TARGETS):
                X_tr_t = X_tr.copy(); X_va_t = X_va.copy()
                X_tr_t['target_name_encoded'] = idx
                X_va_t['target_name_encoded'] = idx
                X_tr_t = preprocess_for_lgbm(X_tr_t)
                X_va_t = preprocess_for_lgbm(X_va_t)

                m_cv = LGBMRegressor(
                    n_estimators=50,
                    learning_rate=0.05,
                    random_state=42,
                    verbose=-1
                )
                m_cv.fit(X_tr_t, y_tr[tgt])
                oof.loc[va_idx, tgt] = m_cv.predict(X_va_t)

        # Compute rank-corr Sharpe
        def compute_sharpe(preds: pd.DataFrame, truths: pd.DataFrame) -> float:
            scores = []
            for p_row, t_row in zip(preds.values, truths.values):
                mask = t_row != SOL_FILL
                if mask.sum()<2 or np.std(p_row[mask])==0 or np.std(t_row[mask])==0:
                    scores.append(0.0)
                else:
                    scores.append(
                        np.corrcoef(
                            pd.Series(p_row[mask]).rank(),
                            pd.Series(t_row[mask]).rank()
                        )[0,1]
                    )
            arr = np.array(scores)
            return float(arr.mean() / arr.std(ddof=0)) if arr.std(ddof=0)>0 else 0.0

        cv_score = compute_sharpe(oof, sol)
        print(f" CV RankCorr-Sharpe = {cv_score:.4f}")
    else:
        print("Cross-validation disabled (DO_CV=False).")

# ==============================================================================
# INFERENCE MODE
# ==============================================================================
else:
    # Load models from MODEL_INPUT_DIR
    for tgt in TARGETS:
        path = os.path.join(MODEL_INPUT_DIR, f"{tgt}_model.pkl")
        if os.path.exists(path):
            models[tgt] = joblib.load(path)
        else:
            print(f"Warning: model file not found: {path}")
            models[tgt] = None
    print("Models loaded for inference.")

# ==============================================================================
# PREDICTION FUNCTION for Kaggle Server
# ==============================================================================
def predict(
    test: pl.DataFrame,
    lag1, lag2, lag3, lag4
) -> pl.DataFrame:
    """
    For each target_i:
      - select FEATURE_NAMES
      - add constant 'target_name_encoded'
      - preprocess & predict
    Returns a Polars DataFrame with columns target_0…target_423.
    """
    df_feat = create_features(test)
    X_test  = df_feat.to_pandas()[FEATURE_NAMES].copy()
    out     = {}
    for idx, tgt in enumerate(TARGETS):
        m = models.get(tgt)
        if m is None:
            out[tgt] = np.full(len(X_test), np.nan)
            continue
        X_tmp = X_test.copy()
        X_tmp['target_name_encoded'] = idx
        X_tmp = preprocess_for_lgbm(X_tmp)
        out[tgt] = m.predict(X_tmp)
    return pl.DataFrame(out)

# ==============================================================================
# ENTRYPOINT / LAUNCH SERVER
# ==============================================================================
if IS_KAGGLE:
    server = kaggle_evaluation.mitsui_inference_server.MitsuiInferenceServer(predict)
    if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
        server.serve()
    else:
        server.run_local_gateway((DATA_PATH,))
else:
    # Local test: generate a submission CSV
    mock = pl.DataFrame()
    submission = predict(test_df, mock, mock, mock, mock)
    submission.write_csv("submission_local.csv")
    print("Local submission saved → submission_local.csv")
```