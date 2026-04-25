# 🏆LB 0.66 No Datasets? No Problem

- **Author:** Yusuke Togashi
- **Votes:** 228
- **Ref:** yusuketogashi/lb-0-66-no-datasets-no-problem
- **URL:** https://www.kaggle.com/code/yusuketogashi/lb-0-66-no-datasets-no-problem
- **Last run:** 2025-08-31 07:37:20.517000

---

```python
# =========================================================
# NeurIPS 2025 Open Polymer - Minimal Safe Revert (→0.66復帰狙い)
#  * 全ターゲット CatBoost( MAE ) に統一
#  * 余計な重み付け/アンサンブル/等方回帰は不使用
#  * 記述子の整形は学習統計でNaN埋め＋安全クリップのみ
#  * ローカル簡易MAEを1行だけ表示（ログ短縮）
# =========================================================

# --- install (offline wheels) ---
!pip install /kaggle/input/rdkit-2025-3-3-cp311/rdkit-2025.3.3-cp311-cp311-manylinux_2_28_x86_64.whl -q
!pip install mordred --no-index --find-links=file:///kaggle/input/mordred-1-2-0-py3-none-any/ -q

# --- imports ---
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from catboost import CatBoostRegressor
from rdkit import Chem
from mordred import Calculator, descriptors
import warnings, gc
warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)

SEED = 42
np.random.seed(SEED)

# --- data load (low_memory=False で型ブレ回避) ---
tg      = pd.read_csv('/kaggle/input/modred-dataset/desc_tg.csv',      low_memory=False)
tc      = pd.read_csv('/kaggle/input/modred-dataset/desc_tc.csv',      low_memory=False)
rg      = pd.read_csv('/kaggle/input/modred-dataset/desc_rg.csv',      low_memory=False)
ffv     = pd.read_csv('/kaggle/input/modred-dataset/desc_ffv.csv',     low_memory=False)
density = pd.read_csv('/kaggle/input/modred-dataset/desc_de.csv',      low_memory=False)
test    = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/test.csv', low_memory=False)
ID = test['id'].values

# --- train tables: 定数列drop＋数値のみ ---
def _clean_train(df: pd.DataFrame) -> pd.DataFrame:
    const_cols = [c for c in df.columns if df[c].nunique() <= 1]
    if const_cols:
        df = df.drop(columns=const_cols)
    return df.select_dtypes(include=[np.number]).copy()

tg      = _clean_train(tg)
tc      = _clean_train(tc)
rg      = _clean_train(rg)
ffv     = _clean_train(ffv)
density = _clean_train(density)

# --- test記述子（Mordred, 数値のみ, inf→NaN）---
mols_test = [Chem.MolFromSmiles(s) for s in test.SMILES]
calc = Calculator(descriptors, ignore_3D=True)
desc_test = calc.pandas(mols_test)
desc_test.columns = desc_test.columns.map(str)
desc_test = desc_test.select_dtypes(include=[np.number]).copy()
desc_test = desc_test.replace([np.inf, -np.inf], np.nan)

# --- 共通X/y整形（学習中央値or平均でNaN埋め） ---
def prepare_Xy(train_df: pd.DataFrame, test_df: pd.DataFrame, target: str):
    assert target in train_df.columns, f"{target} missing"
    # 学習とテストの共通列（trainの列順を保持）
    common = [c for c in train_df.columns if c != target and c in test_df.columns]
    if not common:
        raise RuntimeError(f"No common numeric features for {target}.")
    y_all = train_df[target].astype(float)
    mask = y_all.notna()
    y = y_all.loc[mask].values
    X = train_df.loc[mask, common].copy()
    T = test_df[common].copy()
    # inf→NaN
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    T.replace([np.inf, -np.inf], np.nan, inplace=True)
    # 列ごとに歪度で埋め方を切替（強い歪み:中央値/それ以外:平均）
    med = X.median()
    mean = X.mean()
    skew = X.skew().fillna(0)
    for c in common:
        fillv = med[c] if abs(float(skew[c])) > 1.0 else mean[c]
        if X[c].isna().any():
            X[c] = X[c].fillna(fillv)
        if T[c].isna().any():
            T[c] = T[c].fillna(fillv)
    return X.values.astype(np.float32), y.astype(np.float32), T.values.astype(np.float32)

# --- CatBoost（全ターゲット同一設定・安定志向） ---
def make_cat():
    return CatBoostRegressor(
        loss_function='MAE',
        iterations=1200,       # 安定寄り（過剰に回さない）
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=3.0,
        subsample=0.8,
        random_seed=SEED,
        verbose=0,
        allow_writing_files=False
    )

# --- 簡易ホールドアウト MAE（1行だけ印字） ---
def quick_holdout_mae(train_df: pd.DataFrame, test_df: pd.DataFrame, target: str):
    X, y, _ = prepare_Xy(train_df, test_df, target)
    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, random_state=SEED)
    m = make_cat()
    m.fit(X_tr, y_tr)
    p = m.predict(X_va)
    return float(mean_absolute_error(y_va, p))

TRAIN = {'Tg': tg, 'FFV': ffv, 'Tc': tc, 'Density': density, 'Rg': rg}
mae_rows = []
for t in ['Tg','FFV','Tc','Density','Rg']:
    mae_rows.append((t, quick_holdout_mae(TRAIN[t], desc_test, t)))
print("Local holdout MAE:", " ".join([f"{k}: {v:.6f}" for k,v in mae_rows]), flush=True)

# --- 本学習（全データfit→予測） ---
def predict_full(train_df: pd.DataFrame, test_df: pd.DataFrame, target: str):
    X, y, T = prepare_Xy(train_df, test_df, target)
    model = make_cat()
    model.fit(X, y)
    pred = model.predict(T)
    # 安全な範囲での軽いクリップ（trainの1–99%）
    lo, hi = np.percentile(y, [1, 99])
    return np.clip(pred, lo, hi)

submission = pd.DataFrame({
    'id': ID,
    'Tg':      predict_full(tg,      desc_test, 'Tg'),
    'FFV':     predict_full(ffv,     desc_test, 'FFV'),
    'Tc':      predict_full(tc,      desc_test, 'Tc'),
    'Density': predict_full(density, desc_test, 'Density'),
    'Rg':      predict_full(rg,      desc_test, 'Rg'),
})

# 物理的に明らかにおかしい値だけ最終ガード
submission['FFV'] = submission['FFV'].clip(0.0, 1.0)
submission['Density'] = submission['Density'].clip(lower=0.0)

submission.to_csv('submission.csv', index=False)
print(f"Saved submission.csv {submission.shape}", flush=True)
print(submission.head())
```