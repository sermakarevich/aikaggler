# 🏆Leak-Safe 3CV: Shakedown-Proof Pipeline

- **Author:** Yusuke Togashi
- **Votes:** 287
- **Ref:** yusuketogashi/leak-safe-3cv-shakedown-proof-pipeline
- **URL:** https://www.kaggle.com/code/yusuketogashi/leak-safe-3cv-shakedown-proof-pipeline
- **Last run:** 2025-08-17 07:52:55.417000

---

# Leak-Prevention (what the code guarantees)
* **Chronology enforced**: train and train_labels are sorted by date_id before CV/training (no forward peeking).
* **Shared features only**: Features are built from (train ∩ test) − {date_id,row_id,is_scored} to avoid train/test drift and accidental look-ahead.
* **Shape parity at inference**: Each model saves its training feature order; at predict time we reindex to that exact order (missing → 0, extras dropped) to kill shape/categorical mismatches.
* **Label hygiene**: Each target is trained only on its non-null rows; predictions never return NaNs.
* **Fold-local preprocessing**: Any casting/cleanup happens inside each fold (no statistics fitted across time or using validation/test).

# CV from three angles (aligned with the competition metric)
1. OOF TimeSeriesSplit (clean, generalization-oriented)
* Chronological TSS after sorting by date_id.
* Per-fold, per-target training on non-null labels only.
* Metric = daily Spearman across 424 targets → Sharpe (mean/std across days).
* Labels use the official filler −999999 for metric only; logs show per-fold Sharpe and OOF Sharpe.

2. Clean Public-90 hold-out (practical offline proxy)
* Hold out the last ~90 days by date_id; train on the rest; evaluate with the same daily Spearman→Sharpe.
* Serves as a safer proxy for future data (often lower than public LB; better indicator of robustness).

3. Leaky Public-90 diagnostic (public LB comparator)
* Compute a “leaky” score on the last ~90 days without strict separation to approximate the public LB.
* he gap between Leaky Public-90 and Clean Public-90 flags public-LB inflation and weak future generalization.


# リーク対策（コードで担保していること）
* 時系列順の厳守： date_id で train と train_labels をソートしてから CV/学習（未来を覗かない）。
* 共通特徴量のみ： 特徴量は (train ∩ test) − {date_id,row_id,is_scored} から作成（データドリフト＆先読みを防止）。
* 推論時の形状固定： 学習時の特徴量順序を保存し、推論時は同じ順序に reindex（不足は0埋め・余分は削除）で不一致を根絶。
* ラベルの衛生： 各ターゲットは 非欠損行のみで学習し、予測は NaN を返さない。
* Fold 内完結の前処理： キャスト/クリーンアップは 各Foldの中で実施（時系列や検証/テストを跨ぐ情報漏れ無し）。


# 3つのCV観点（評価指標に整合）
# 1. OOF TimeSeriesSplit（クリーンで汎化志向）
* date_id でソート後に時系列TSS。
* Fold×Targetごとに非欠損のみで学習。
* 指標＝日次のSpearman（424ターゲット横持ち）→ Sharpe（平均/標準偏差）。
* メトリック用にラベルは −999999 で埋め、Fold別SharpeとOOF Sharpeを出力。

# 2. Clean Public-90 ホールドアウト（実務向けのオフライン代理）
* 末尾 約90日 を date_id でホールドアウト、残りで学習し同指標で評価。
* 将来データの代理として安全（多くの場合 公開LBより低め、汎化の目安）。

# 3. Leaky Public-90 診断（公開LBの近似）
* 末尾約90日で“甘め”にスコアを算出し、公開LBに近い値を得る。
* Leaky と Clean の 差が大きいほど、公開LBの見かけ上の高さと将来汎化の弱さを示唆。

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
TRAIN       = True    # True: train + (optional) CV + save; False: load + predict
DO_CV       = True    # CV is only used when TRAIN=True
CV_SPLITS   = 3
DO_PUBLIC90 = True    # <- 追加: 末尾90日ホールドアウト（Public LB に近い検証）を実行するか

# Force inference-only inside the scoring container (safety)
IS_COMP_RUN = os.getenv('KAGGLE_IS_COMPETITION_RERUN') is not None
if IS_COMP_RUN:
    TRAIN  = False
    DO_CV  = False
    DO_PUBLIC90 = False  # スコアリング用コンテナでは必ず無効化

# Detect Kaggle environment
IS_KAGGLE = os.getenv('KAGGLE_KERNEL_RUN_TYPE') is not None
if IS_KAGGLE:
    import kaggle_evaluation.mitsui_inference_server
    DATA_PATH        = '/kaggle/input/mitsui-commodity-prediction-challenge/'
    MODEL_INPUT_DIR  = '/kaggle/input/model4-suke/'   # <- your model artifact dir
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

# ------------------------------------------------------------------------------
# Leak prevention #1: sort BOTH train and labels by date_id before CV/training
# ------------------------------------------------------------------------------
if 'date_id' in train_df.columns:
    train_df     = train_df.sort('date_id')
if 'date_id' in train_labels.columns:
    train_labels = train_labels.sort('date_id')

# ==============================================================================
# FEATURE / TARGET DEFINITIONS
# ==============================================================================
# Leak prevention #2: build SHARED features from (train ∩ test) minus excludes
EXCLUDE_COLS   = {'date_id', 'row_id', 'is_scored'}
COMMON_COLUMNS = [c for c in train_df.columns if c in set(test_df.columns)]
FEATURE_BASE   = [c for c in COMMON_COLUMNS if c not in EXCLUDE_COLS]

TARGETS = [f"target_{i}" for i in range(424)]

# ==============================================================================
# PREPROCESSING & FEATURE ENGINEERING
# ==============================================================================
def preprocess_for_lgbm(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert object-type columns to numeric and pandas.Categorical to integer codes.
    Keep dtypes stable between train and predict.
    """
    df = df.copy()
    for col in [
        'US_Stock_GOLD_adj_open','US_Stock_GOLD_adj_high',
        'US_Stock_GOLD_adj_low','US_Stock_GOLD_adj_close',
        'US_Stock_GOLD_adj_volume'
    ]:
        if col in df and df[col].dtype == 'object':
            df[col] = pd.to_numeric(df[col], errors='coerce')
    for cat in df.select_dtypes(['category']).columns:
        df[cat] = df[cat].cat.codes
    df = df.replace([np.inf, -np.inf], np.nan)
    return df

def create_features(df: pl.DataFrame) -> pl.DataFrame:
    """Placeholder for feature engineering; returns input unchanged."""
    return df.clone()

# ==============================================================================
# CV METRICS (competition-like)
# ==============================================================================
SOLUTION_NULL_FILLER = -999999

def rankcorr_sharpe(preds_df: pd.DataFrame, truths_df: pd.DataFrame, filler: float = SOLUTION_NULL_FILLER) -> float:
    """
    Competition-like metric:
    - For each date (row), compute Spearman rank correlation across targets (columns)
      between predictions and labels (ignoring filler).
    - Return Sharpe ratio = mean / std of those daily correlations.
    """
    P = preds_df.copy().fillna(0.0)
    T = truths_df.copy().fillna(filler)

    daily_scores = []
    for p_row, t_row in zip(P.values, T.values):
        mask = (t_row != filler) & np.isfinite(p_row)
        if mask.sum() < 2:
            daily_scores.append(0.0)
            continue
        p_rank = pd.Series(p_row[mask]).rank(method='average').to_numpy()
        t_rank = pd.Series(t_row[mask]).rank(method='average').to_numpy()
        if np.std(p_rank) == 0 or np.std(t_rank) == 0:
            daily_scores.append(0.0)
        else:
            daily_scores.append(np.corrcoef(p_rank, t_rank)[0, 1])

    arr = np.asarray(daily_scores, dtype=float)
    std = arr.std(ddof=0)
    return float(arr.mean() / std) if std > 0 else 0.0

# 追加: Public-90 style 検証（末尾90日ホールドアウト）
def validate_public_90_style(train_pl: pl.DataFrame, labels_pl: pl.DataFrame) -> float:
    """
    Train on head (N-90), validate on last 90 days (Kaggle public LB に近い構成)。
    戻り値は Kaggle 互換 Sharpe。
    """
    X_all_df = create_features(train_pl).to_pandas()
    X_all    = X_all_df[FEATURE_BASE].copy()
    y_all    = labels_pl.to_pandas()[TARGETS].copy()

    N = len(X_all)
    H = 90
    if N <= H:
        print("[Public-90] Not enough length; skipped.")
        return 0.0

    tr_idx = np.arange(0, N - H)
    va_idx = np.arange(N - H, N)

    preds_va = pd.DataFrame(index=range(H), columns=TARGETS, dtype=float)
    sol_va   = y_all.iloc[va_idx].reset_index(drop=True)

    for idx, tgt in enumerate(TARGETS):
        y = y_all[tgt]
        mask_tr = (~y.isna()).values & (np.arange(N) < (N - H))
        if mask_tr.sum() == 0:
            preds_va[tgt] = 0.0
            continue

        Xtr = X_all.loc[mask_tr].copy()
        ytr = y.loc[mask_tr]
        Xva = X_all.iloc[va_idx].copy()

        Xtr['target_name_encoded'] = idx
        Xva['target_name_encoded'] = idx

        Xtr = preprocess_for_lgbm(Xtr).reindex(columns=(FEATURE_BASE + ['target_name_encoded']), fill_value=0.0)
        Xva = preprocess_for_lgbm(Xva).reindex(columns=(FEATURE_BASE + ['target_name_encoded']), fill_value=0.0)

        m = LGBMRegressor(
            n_estimators=100, learning_rate=0.05,
            max_depth=-1, num_leaves=64,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbose=-1
        )
        m.fit(Xtr, ytr)
        preds_va[tgt] = m.predict(Xva)

    score = rankcorr_sharpe(preds_va, sol_va, filler=SOLUTION_NULL_FILLER)
    return float(score)

# ==============================================================================
# MODEL STORAGE / METADATA
# ==============================================================================
models         = {}  # tgt -> LightGBM model
MODEL_FEATURES = {}  # tgt -> list of feature names used at train time (order matters)
MODELS_LOADED  = False

def _meta_path_for(tgt: str) -> str:
    return os.path.join(MODEL_OUTPUT_DIR, f"{tgt}_feat.pkl")

def _load_model_and_meta(tgt: str, input_dir: str):
    mpath = os.path.join(input_dir, f"{tgt}_model.pkl")
    fpath = os.path.join(input_dir, f"{tgt}_feat.pkl")
    model = joblib.load(mpath) if os.path.exists(mpath) else None
    feats = None
    if os.path.exists(fpath):
        try:
            feats = joblib.load(fpath)
        except Exception:
            feats = None
    return model, feats

def _fallback_feature_order_for_model(model) -> list:
    n = getattr(model, "n_features_", None)
    if not isinstance(n, (int, np.integer)) or n < 1:
        return (FEATURE_BASE + ['target_name_encoded'])
    take = max(0, n - 1)
    return (FEATURE_BASE[:take] + ['target_name_encoded'])

def _lazy_load_models():
    global MODELS_LOADED, models, MODEL_FEATURES
    if MODELS_LOADED:
        return
    loaded = 0
    for tgt in TARGETS:
        model, feats = _load_model_and_meta(tgt, MODEL_INPUT_DIR)
        models[tgt] = model
        if (model is not None) and (feats is None):
            feats = _fallback_feature_order_for_model(model)
        MODEL_FEATURES[tgt] = feats if feats is not None else (FEATURE_BASE + ['target_name_encoded'])
        if models[tgt] is not None:
            loaded += 1
    MODELS_LOADED = True
    print(f"[Info] Lazy-loaded models: {loaded} / {len(TARGETS)}")

# ==============================================================================
# TRAINING MODE
# ==============================================================================
if TRAIN:
    print("\n======== TRAINING MODE ========")
    X_all_df = create_features(train_df).to_pandas()
    X_all    = X_all_df[FEATURE_BASE]
    y_all    = train_labels.to_pandas()[TARGETS]

    for idx, tgt in enumerate(TARGETS):
        print(f"Training {tgt} …")
        y = y_all[tgt]
        mask = ~y.isna()
        X_tr_base = X_all.loc[mask].copy()
        y_tr      = y.loc[mask]
        if len(y_tr) == 0:
            print(f"  Skip {tgt}: no non-null labels.")
            models[tgt] = None
            MODEL_FEATURES[tgt] = FEATURE_BASE + ['target_name_encoded']
            joblib.dump(MODEL_FEATURES[tgt], _meta_path_for(tgt))
            continue

        X_tr = X_tr_base.copy()
        X_tr['target_name_encoded'] = idx
        X_tr = preprocess_for_lgbm(X_tr)

        feat_order = FEATURE_BASE + ['target_name_encoded']
        X_tr = X_tr.reindex(columns=feat_order, fill_value=0.0)

        m = LGBMRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=-1,
            num_leaves=64,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1
        )
        m.fit(X_tr, y_tr)

        models[tgt] = m
        MODEL_FEATURES[tgt] = feat_order
        joblib.dump(m, os.path.join(MODEL_OUTPUT_DIR, f"{tgt}_model.pkl"))
        joblib.dump(feat_order, _meta_path_for(tgt))

    print(f"Saved models & metadata to: {MODEL_OUTPUT_DIR}")

    # --------------------------------------------------------------------
    # OPTIONAL TimeSeriesSplit CV (RankCorr-Sharpe) with leak prevention
    # --------------------------------------------------------------------
    if DO_CV:
        print(f"\nRunning TimeSeriesSplit CV with {CV_SPLITS} folds …")
        SOL_FILL = SOLUTION_NULL_FILLER

        X_base  = X_all.copy()
        y_truth = y_all.copy()
        y_score = y_truth.fillna(SOL_FILL)

        oof = pd.DataFrame(index=y_truth.index, columns=TARGETS, dtype=float)
        tscv = TimeSeriesSplit(n_splits=CV_SPLITS)
        fold_scores = []
        for fold, (tr_idx, va_idx) in enumerate(tscv.split(X_base)):
            print(f"  CV Fold {fold+1}/{CV_SPLITS}")
            X_tr_b, X_va_b = X_base.iloc[tr_idx], X_base.iloc[va_idx]
            y_tr_all       = y_truth.iloc[tr_idx]

            for idx, tgt in enumerate(TARGETS):
                y_tr_tgt = y_tr_all[tgt]
                mask_tr  = ~y_tr_tgt.isna()
                if mask_tr.sum() == 0:
                    continue

                Xtr = X_tr_b.loc[mask_tr].copy()
                ytr = y_tr_tgt.loc[mask_tr]
                Xva = X_va_b.copy()

                Xtr['target_name_encoded'] = idx
                Xva['target_name_encoded'] = idx

                Xtr = preprocess_for_lgbm(Xtr).reindex(columns=(FEATURE_BASE + ['target_name_encoded']), fill_value=0.0)
                Xva = preprocess_for_lgbm(Xva).reindex(columns=(FEATURE_BASE + ['target_name_encoded']), fill_value=0.0)

                m_cv = LGBMRegressor(
                    n_estimators=80,
                    learning_rate=0.05,
                    max_depth=-1,
                    num_leaves=64,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    verbose=-1
                )
                m_cv.fit(Xtr, ytr)
                oof.loc[va_idx, tgt] = m_cv.predict(Xva)

            fold_score = rankcorr_sharpe(
                preds_df=oof.loc[va_idx, TARGETS],
                truths_df=y_score.loc[va_idx, TARGETS],
                filler=SOL_FILL
            )
            fold_scores.append(fold_score)
            print(f"   -> Fold {fold+1} Sharpe: {fold_score:.4f}")

        cv_score = rankcorr_sharpe(oof, y_score, SOL_FILL)
        print(f"  OOF RankCorr-Sharpe = {cv_score:.4f}")
        if len(fold_scores) > 0:
            print(f"  Mean(Fold Sharpe)   = {np.mean(fold_scores):.4f}")
    else:
        print("CV disabled (DO_CV=False).")

    # --------------------------------------------------------------------
    # 追加: Public-90 style validation（末尾90日ホールドアウト）
    # --------------------------------------------------------------------
    if DO_PUBLIC90:
        print("\nRunning Public-90 style hold-out (last ~90 days) …")
        pub90_score = validate_public_90_style(train_df, train_labels)
        print(f"  Public-90 Sharpe     = {pub90_score:.4f}")

else:
    print("\n======== INFERENCE MODE ========")
    print(f"Models will be lazy-loaded from: {MODEL_INPUT_DIR}")

# ==============================================================================
# PREDICTION FUNCTION for Kaggle Server (leak-safe + shape-safe)
# ==============================================================================
def predict(
    test: pl.DataFrame,
    label_lags_1_batch: pl.DataFrame,
    label_lags_2_batch: pl.DataFrame,
    label_lags_3_batch: pl.DataFrame,
    label_lags_4_batch: pl.DataFrame,
) -> pd.DataFrame:
    """
    Leak-safe & shape-safe inference:
      - Use only intersection-based features (FEATURE_BASE).
      - Add per-target constant 'target_name_encoded'.
      - Preprocess and reindex to the EXACT per-model feature order saved at training.
      - Fill missing columns with 0 and drop extras to avoid shape mismatches.
      - Never return NaNs (fallback to 0.0 if a model is missing).
      - Return a Pandas DataFrame with columns TARGETS in the exact order.
    """
    _lazy_load_models()

    df_feat = create_features(test)
    X_base  = df_feat.to_pandas()

    base_selected = X_base.reindex(columns=FEATURE_BASE, fill_value=0.0)

    n_rows = len(base_selected)
    out = np.zeros((n_rows, len(TARGETS)), dtype=np.float64)

    for idx, tgt in enumerate(TARGETS):
        model = models.get(tgt)
        if model is None:
            out[:, idx] = 0.0
            continue

        feat_order = MODEL_FEATURES.get(tgt)
        if feat_order is None:
            feat_order = _fallback_feature_order_for_model(model)

        X_tmp = base_selected.copy()
        X_tmp['target_name_encoded'] = idx

        X_tmp = preprocess_for_lgbm(X_tmp)
        X_tmp = X_tmp.reindex(columns=feat_order, fill_value=0.0)

        pred = model.predict(X_tmp, validate_features=False)
        pred = np.asarray(pred, dtype=np.float64).reshape(-1)
        if pred.shape[0] != n_rows:
            if pred.shape[0] > n_rows:
                pred = pred[:n_rows]
            else:
                pred = np.pad(pred, (0, n_rows - pred.shape[0]), constant_values=0.0)
        out[:, idx] = pred

    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    predictions = pd.DataFrame(out, columns=TARGETS)
    assert predictions.shape[1] == 424, f"Expected 424 columns, got {predictions.shape[1]}"
    assert list(predictions.columns) == TARGETS, "Column order/name mismatch"
    assert np.isfinite(predictions.to_numpy()).all(), "Non-finite values in predictions"
    return predictions

# ==============================================================================
# ENTRYPOINT / LAUNCH SERVER
# ==============================================================================
if IS_KAGGLE:
    server = kaggle_evaluation.mitsui_inference_server.MitsuiInferenceServer(predict)
    if IS_COMP_RUN:
        server.serve()
    else:
        server.run_local_gateway((DATA_PATH,))
else:
    mock = pl.DataFrame()
    submission = predict(test_df, mock, mock, mock, mock)
    submission = submission[TARGETS]
    submission.to_csv("submission_local.csv", index=False)
    print("Local submission saved → submission_local.csv")
```

```python
# === Diagnostic: Leaky Public-90 (full learning → last 90 days evaluation) ===
def validate_public_90_style_leaky(train_pl: pl.DataFrame, labels_pl: pl.DataFrame) -> float:
    X_all_df = create_features(train_pl).to_pandas()
    X_all    = X_all_df[FEATURE_BASE].copy()
    y_all    = labels_pl.to_pandas()[TARGETS].copy()

    N, H = len(X_all), 90
    if N <= H:
        print("[Leaky Public-90] Not enough length; skipped.")
        return 0.0

    va_idx = np.arange(N - H, N)
    preds_va = pd.DataFrame(index=range(H), columns=TARGETS, dtype=float)
    sol_va   = y_all.iloc[va_idx].reset_index(drop=True)

    for idx, tgt in enumerate(TARGETS):
        y   = y_all[tgt]
        msk = ~y.isna()
        if msk.sum() == 0:
            preds_va[tgt] = 0.0
            continue
        Xtr = X_all.loc[msk].copy(); ytr = y.loc[msk]
        Xva = X_all.iloc[va_idx].copy()
        Xtr["target_name_encoded"] = idx
        Xva["target_name_encoded"] = idx
        Xtr = preprocess_for_lgbm(Xtr).reindex(columns=(FEATURE_BASE+["target_name_encoded"]), fill_value=0.0)
        Xva = preprocess_for_lgbm(Xva).reindex(columns=(FEATURE_BASE+["target_name_encoded"]), fill_value=0.0)
        m = LGBMRegressor(
            n_estimators=100, learning_rate=0.05,
            max_depth=-1, num_leaves=64, subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbose=-1
        )
        m.fit(Xtr, ytr)
        preds_va[tgt] = m.predict(Xva)

    return rankcorr_sharpe(preds_va, sol_va, filler=SOLUTION_NULL_FILLER)

if TRAIN and DO_PUBLIC90 and not IS_COMP_RUN:
    print("\n[Diag] Leaky Public-90 style (train=full → val=last 90) …")
    leaky_score = validate_public_90_style_leaky(train_df, train_labels)
    print(f"  Leaky Public-90 Sharpe = {leaky_score:.4f}  “(Reference value: tends to be closer to the public leaderboard score)")
```

```python
# --- Zip LightGBM model artifacts in /kaggle/working/model into a single archive ---
# Notes:
# - Comments are in English (Kaggle requirement)
# - It zips all *.pkl files under /kaggle/working/model (recursively)
# - It prints progress every 50 files, verifies archive integrity, and shows a short preview

import os
import zipfile
from pathlib import Path


# ====== CONFIG ======
SRC_DIR = Path("/kaggle/working/model")        # directory containing your ~400 *.pkl files
OUT_ZIP = Path("/kaggle/working/model_bundle.zip")
INCLUDE_EXTS = {".pkl"}                        # add more extensions if needed (e.g., '.txt', '.json')

# ====== COLLECT FILES ======
# Recursively find all files with allowed extensions
files = [p for p in SRC_DIR.rglob("*") if p.is_file() and p.suffix.lower() in INCLUDE_EXTS]
files.sort()

if not files:
    raise SystemExit(f"No files with extensions {INCLUDE_EXTS} found under {SRC_DIR}")

print(f"Found {len(files)} file(s) to zip from: {SRC_DIR}")

# ====== CREATE ZIP ======
# Use deflated compression; compresslevel 6 is a good trade-off between size and speed
with zipfile.ZipFile(OUT_ZIP, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
    for i, fp in enumerate(files, 1):
        # Keep relative paths inside the archive (no absolute paths)
        arcname = fp.relative_to(SRC_DIR)
        zf.write(fp, arcname)
        if i % 50 == 0 or i == len(files):
            print(f"  Added {i}/{len(files)} -> {arcname}")

print("\nZip build complete.")

# ====== VERIFY ARCHIVE INTEGRITY ======
# testzip() returns the name of the first corrupt file, or None if all is OK
with zipfile.ZipFile(OUT_ZIP, mode="r") as zf:
    bad_member = zf.testzip()
    if bad_member is None:
        print("Integrity check: OK (no corrupt members).")
    else:
        print(f"Integrity check: CORRUPT member found -> {bad_member}")

# ====== STATS & PREVIEW ======
size_mb = OUT_ZIP.stat().st_size / (1024 * 1024)
print(f"Archive path : {OUT_ZIP}")
print(f"Archive size : {size_mb:.2f} MB")

with zipfile.ZipFile(OUT_ZIP, mode="r") as zf:
    names = zf.namelist()
    head = names[:10]
    print("\nPreview (first 10 entries):")
    for n in head:
        print(f"  - {n}")
    print(f"... total entries: {len(names)}")
```