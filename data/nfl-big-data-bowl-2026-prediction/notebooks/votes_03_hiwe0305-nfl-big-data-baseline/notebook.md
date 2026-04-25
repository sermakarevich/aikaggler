# NFL Big Data - Baseline

- **Author:** Hiwe
- **Votes:** 316
- **Ref:** hiwe0305/nfl-big-data-baseline
- **URL:** https://www.kaggle.com/code/hiwe0305/nfl-big-data-baseline
- **Last run:** 2025-10-11 00:54:46.270000

---

```python
# ============================================================
# NFL Big Data Bowl 2026 — CatBoost Residual (CA Baseline)
# + Role-Specific Heads (TR/DC) + Ball-Frame features + GNN-lite
# - Base pipeline = bản 0.63 (last frame + residual)
# - Nâng cấp: CA baseline, ball-frame, formation,
#   2 head chuyên biệt cho Targeted Receiver & Coverage rồi blend.
# - ĐÃ BẬT multi-GPU (tự phát hiện 2x T4 -> '0:1')
# ============================================================

import os, warnings, math, pickle
import numpy as np
import pandas as pd
from pathlib import Path
from multiprocessing import Pool as MP, cpu_count
from tqdm.auto import tqdm
from sklearn.model_selection import GroupKFold, KFold
from sklearn.metrics import mean_squared_error
from catboost import CatBoostRegressor, Pool as CatPool
from catboost.utils import get_gpu_device_count  # NEW

warnings.filterwarnings("ignore")

# --------------------------- CONFIG --------------------------- #
BASEDIR = Path("/kaggle/input/nfl-big-data-bowl-2026-prediction")
SAVE_DIR = Path("/kaggle/working")

N_WEEKS = 18
SEED = 42

# CV
N_FOLDS = 5
USE_GROUP_KFOLD = True

# CatBoost
ITERATIONS = 20000
LR = 0.10          # hơi cao hơn bản 0.63 một chút
DEPTH = 8
L2 = 8.0
EARLY = 700
BORDER_CT = 254

# GNN-lite knobs
K_NEIGH = 6
RADIUS  = 30.0
TAU     = 8.0

# Blend weights giữa GLOBAL và role-heads
ALPHA_TR = 0.5      # 0.5 global + 0.5 TR
ALPHA_DC = 0.5      # 0.5 global + 0.5 DC

# Horizon weighting cho sample_weight
W_H = 0.25          # weight = 1 + W_H * delta_frames (clip 1..5)

# ---------------------- GPU DETECTION (multi-GPU) ---------------------- #
def _detect_devices() -> tuple[bool, str | None]:
    """Trả về (USE_GPU, DEVICES_STR). DEVICES_STR dạng '0:1' nếu có >=2 GPU."""
    try:
        n = int(get_gpu_device_count())
    except Exception:
        n = 0
    if n >= 2:
        return True, ":".join(str(i) for i in range(n))  # '0:1' cho T4x2
    elif n == 1:
        return True, "0"
    else:
        return False, None

USE_GPU, DEVICES = _detect_devices()
print(f"[CatBoost] GPUs visible: {get_gpu_device_count()} | DEVICES={DEVICES} | "
      f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")

# --------------------------- IO UTILS --------------------------- #
def load_week(week_num: int):
    fin = BASEDIR / f"train/input_2023_w{week_num:02d}.csv"
    fout = BASEDIR / f"train/output_2023_w{week_num:02d}.csv"
    return pd.read_csv(fin), pd.read_csv(fout)

def load_all_train():
    print("Loading training data...")
    with MP(min(cpu_count(), 18)) as pool:
        res = list(tqdm(pool.imap(load_week, range(1, N_WEEKS+1)), total=N_WEEKS))
    tr_in  = pd.concat([r[0] for r in res], ignore_index=True)
    tr_out = pd.concat([r[1] for r in res], ignore_index=True)
    print(f"Train input:  {tr_in.shape}")
    print(f"Train output: {tr_out.shape}")
    return tr_in, tr_out

# ---------------------- FEATURE ENGINEERING ---------------------- #
def to_inches(h):
    try:
        a,b = str(h).split("-")
        return float(a)*12.0 + float(b)
    except Exception:
        return np.nan

def engineer_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # defaults an toàn
    for c in ['x','y','s','a','o','dir','ball_land_x','ball_land_y','frame_id',
              'player_height','player_weight','player_role','player_side']:
        if c not in df.columns:
            if c == 'player_height': df[c] = '6-0'
            elif c in ['player_role','player_side']: df[c] = ''
            else: df[c] = 0.0

    # Height/BMI
    h = df['player_height'].map(to_inches).fillna(72.0)
    w = pd.to_numeric(df['player_weight'], errors='coerce').fillna(200.0)
    df['height_inches'] = h
    df['bmi'] = (w / (h**2 + 1e-6)) * 703.0

    # Heading (dir=0° -> +y)
    dir_rad = np.radians(pd.to_numeric(df['dir'], errors='coerce').fillna(0.0))
    df['heading_x'] = np.sin(dir_rad)
    df['heading_y'] = np.cos(dir_rad)

    # Velocity / Acceleration
    s = pd.to_numeric(df['s'], errors='coerce').fillna(0.0)
    a = pd.to_numeric(df['a'], errors='coerce').fillna(0.0)
    df['velocity_x'] = s * df['heading_x']
    df['velocity_y'] = s * df['heading_y']
    df['acceleration_x'] = a * df['heading_x']
    df['acceleration_y'] = a * df['heading_y']

    # Ball geometry
    x  = pd.to_numeric(df['x'], errors='coerce').fillna(0.0)
    y  = pd.to_numeric(df['y'], errors='coerce').fillna(0.0)
    bx = pd.to_numeric(df['ball_land_x'], errors='coerce').fillna(0.0)
    by = pd.to_numeric(df['ball_land_y'], errors='coerce').fillna(0.0)
    dx = bx - x; dy = by - y
    dist = np.sqrt(dx*dx + dy*dy)
    df['dist_to_ball'] = dist
    df['angle_to_ball'] = np.arctan2(dy, dx)

    # Unit vectors -> “ball-frame” (song song / vuông góc)
    ux = dx / (dist + 1e-6); uy = dy / (dist + 1e-6)
    vx = -uy; vy = ux

    # projections (song song / vuông góc) của v & a
    df['v_para']  = df['velocity_x']*ux + df['velocity_y']*uy
    df['v_perp']  = df['velocity_x']*vx + df['velocity_y']*vy
    df['a_para']  = df['acceleration_x']*ux + df['acceleration_y']*uy
    df['a_perp']  = df['acceleration_x']*vx + df['acceleration_y']*vy
    # alignment (cosine với hướng bóng)
    df['heading_alignment'] = df['heading_x']*ux + df['heading_y']*uy

    # Other physics
    df['speed_squared']   = s**2
    df['accel_magnitude'] = np.sqrt(df['acceleration_x']**2 + df['acceleration_y']**2)
    df['momentum_x'] = w * df['velocity_x']
    df['momentum_y'] = w * df['velocity_y']
    df['kinetic_energy'] = 0.5 * w * df['speed_squared']

    # Roles / side (binary)
    pr = df['player_role'].astype(str)
    ps = df['player_side'].astype(str)
    df['role_targeted_receiver']   = (pr == 'Targeted Receiver').astype(np.int8)
    df['role_defensive_coverage']  = (pr == 'Defensive Coverage').astype(np.int8)
    df['role_passer']              = (pr == 'Passer').astype(np.int8)
    df['side_offense']             = (ps == 'Offense').astype(np.int8)
    return df

def add_sequence_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["game_id","play_id","nfl_id","frame_id"]).copy()
    gcols = ["game_id","play_id","nfl_id"]

    for lag in [1,2,3,4,5]:
        for c in ["x","y","velocity_x","velocity_y","s","a","v_para","v_perp","a_para","a_perp"]:
            if c in df.columns:
                df[f"{c}_lag{lag}"] = df.groupby(gcols)[c].shift(lag)

    for win in [3,5]:
        for c in ["x","y","velocity_x","velocity_y","s","v_para","v_perp"]:
            if c in df.columns:
                df[f"{c}_rolling_mean_{win}"] = (
                    df.groupby(gcols)[c].rolling(win, min_periods=1).mean()
                      .reset_index(level=[0,1,2], drop=True)
                )
                df[f"{c}_rolling_std_{win}"] = (
                    df.groupby(gcols)[c].rolling(win, min_periods=1).std()
                      .reset_index(level=[0,1,2], drop=True)
                )

    for c in ["velocity_x","velocity_y","v_para","v_perp"]:
        if c in df.columns:
            df[f"{c}_delta"] = df.groupby(gcols)[c].diff()

    return df

def add_formation_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    grp = df.groupby(['game_id','play_id','frame_id','player_side'], sort=False)
    df['team_centroid_x'] = grp['x'].transform('mean')
    df['team_centroid_y'] = grp['y'].transform('mean')
    df['team_width']      = grp['y'].transform('std').fillna(0.0)
    df['team_length']     = grp['x'].transform('std').fillna(0.0)
    df['rel_cx'] = df['x'] - df['team_centroid_x']
    df['rel_cy'] = df['y'] - df['team_centroid_y']
    bx = pd.to_numeric(df['ball_land_x'], errors='coerce').fillna(0.0)
    by = pd.to_numeric(df['ball_land_y'], errors='coerce').fillna(0.0)
    bearing = np.arctan2(by - df['team_centroid_y'], bx - df['team_centroid_x'])
    df['form_bear_sin'] = np.sin(bearing)
    df['form_bear_cos'] = np.cos(bearing)
    return df

# ---------------------- GNN-lite (last-frame) ---------------------- #
def compute_neighbor_embeddings(input_df: pd.DataFrame,
                                k_neigh: int = K_NEIGH,
                                radius: float = RADIUS,
                                tau: float = TAU) -> pd.DataFrame:
    cols_needed = ["game_id","play_id","nfl_id","frame_id","x","y",
                   "velocity_x","velocity_y","player_side"]
    src = input_df[cols_needed].copy()

    last = (src.sort_values(["game_id","play_id","nfl_id","frame_id"])
               .groupby(["game_id","play_id","nfl_id"], as_index=False)
               .tail(1)
               .rename(columns={"frame_id":"last_frame_id"})
               .reset_index(drop=True))

    tmp = last.merge(
        src.rename(columns={
            "frame_id":"nb_frame_id","nfl_id":"nfl_id_nb","x":"x_nb","y":"y_nb",
            "velocity_x":"vx_nb","velocity_y":"vy_nb","player_side":"player_side_nb"
        }),
        left_on=["game_id","play_id","last_frame_id"],
        right_on=["game_id","play_id","nb_frame_id"],
        how="left",
    )
    tmp = tmp[tmp["nfl_id_nb"] != tmp["nfl_id"]]

    tmp["dx"]  = tmp["x_nb"] - tmp["x"]
    tmp["dy"]  = tmp["y_nb"] - tmp["y"]
    tmp["dvx"] = tmp["vx_nb"] - tmp["velocity_x"]
    tmp["dvy"] = tmp["vy_nb"] - tmp["velocity_y"]
    tmp["dist"] = np.sqrt(tmp["dx"]**2 + tmp["dy"]**2)
    tmp = tmp[np.isfinite(tmp["dist"]) & (tmp["dist"] > 1e-6)]
    if radius is not None: tmp = tmp[tmp["dist"] <= radius]

    tmp["is_ally"] = (tmp["player_side_nb"].fillna("") == tmp["player_side"].fillna("")).astype(np.float32)
    keys = ["game_id","play_id","nfl_id"]
    tmp["rnk"] = tmp.groupby(keys)["dist"].rank(method="first")
    if k_neigh is not None: tmp = tmp[tmp["rnk"] <= float(k_neigh)]

    tmp["w"] = np.exp(-tmp["dist"] / float(tau))
    sum_w = tmp.groupby(keys)["w"].transform("sum"); tmp["wn"] = np.where(sum_w>0, tmp["w"]/sum_w, 0.0)
    tmp["wn_ally"] = tmp["wn"] * tmp["is_ally"]; tmp["wn_opp"]  = tmp["wn"] * (1.0 - tmp["is_ally"])
    for col in ["dx","dy","dvx","dvy"]:
        tmp[f"{col}_ally_w"] = tmp[col] * tmp["wn_ally"]
        tmp[f"{col}_opp_w"]  = tmp[col] * tmp["wn_opp"]
    tmp["dist_ally"] = np.where(tmp["is_ally"] > 0.5, tmp["dist"], np.nan)
    tmp["dist_opp"]  = np.where(tmp["is_ally"] < 0.5, tmp["dist"], np.nan)

    ag = tmp.groupby(keys).agg(
        gnn_ally_dx_mean=("dx_ally_w","sum"),
        gnn_ally_dy_mean=("dy_ally_w","sum"),
        gnn_ally_dvx_mean=("dvx_ally_w","sum"),
        gnn_ally_dvy_mean=("dvy_ally_w","sum"),
        gnn_opp_dx_mean=("dx_opp_w","sum"),
        gnn_opp_dy_mean=("dy_opp_w","sum"),
        gnn_opp_dvx_mean=("dvx_opp_w","sum"),
        gnn_opp_dvy_mean=("dvy_opp_w","sum"),
        gnn_ally_cnt=("is_ally","sum"),
        gnn_opp_cnt=("is_ally", lambda s: float(len(s) - s.sum())),
        gnn_ally_dmin=("dist_ally","min"),
        gnn_ally_dmean=("dist_ally","mean"),
        gnn_opp_dmin=("dist_opp","min"),
        gnn_opp_dmean=("dist_opp","mean"),
    ).reset_index()

    near = tmp.loc[tmp["rnk"]<=3, keys+["rnk","dist"]].copy()
    near["rnk"] = near["rnk"].astype(int)
    dwide = near.pivot_table(index=keys, columns="rnk", values="dist", aggfunc="first")
    dwide = dwide.rename(columns={1:"gnn_d1",2:"gnn_d2",3:"gnn_d3"}).reset_index()
    ag = ag.merge(dwide, on=keys, how="left")

    for c in ["gnn_ally_dx_mean","gnn_ally_dy_mean","gnn_ally_dvx_mean","gnn_ally_dvy_mean",
              "gnn_opp_dx_mean","gnn_opp_dy_mean","gnn_opp_dvx_mean","gnn_opp_dvy_mean"]:
        ag[c] = ag[c].fillna(0.0)
    for c in ["gnn_ally_cnt","gnn_opp_cnt"]:
        ag[c] = ag[c].fillna(0.0)
    for c in ["gnn_ally_dmin","gnn_opp_dmin","gnn_ally_dmean","gnn_opp_dmean","gnn_d1","gnn_d2","gnn_d3"]:
        ag[c] = ag[c].fillna(radius if radius is not None else 30.0)
    return ag

# ---------------------- TRAIN ROWS & BASELINE ---------------------- #
def create_training_rows(input_df: pd.DataFrame, output_df: pd.DataFrame) -> pd.DataFrame:
    agg = (
        input_df.sort_values(["game_id","play_id","nfl_id","frame_id"])
                .groupby(["game_id","play_id","nfl_id"], as_index=False)
                .tail(1)
                .reset_index(drop=True)
                .rename(columns={"frame_id":"last_frame_id"})
    )

    out = output_df.copy()
    out = out.rename(columns={"x":"target_x","y":"target_y"})
    out["id"] = (
        out["game_id"].astype(str) + "_" +
        out["play_id"].astype(str) + "_" +
        out["nfl_id"].astype(str) + "_" +
        out["frame_id"].astype(str)
    )
    m = out.merge(agg, on=["game_id","play_id","nfl_id"], how="left", suffixes=("","_last"))
    m["delta_frames"] = (m["frame_id"] - m["last_frame_id"]).clip(lower=0).astype(float)
    m["delta_t"] = m["delta_frames"] / 10.0
    return m

def ca_baseline(x_last, y_last, vx_last, vy_last, dt, ax_last, ay_last):
    px = x_last + vx_last*dt + 0.5*ax_last*(dt**2)
    py = y_last + vy_last*dt + 0.5*ay_last*(dt**2)
    return np.clip(px, 0.0, 120.0), np.clip(py, 0.0, 53.3)

# --------------------------- FEATURE LIST --------------------------- #
def build_feature_list(train_df: pd.DataFrame):
    base = [
        "x","y","s","a","o","dir",
        "velocity_x","velocity_y","acceleration_x","acceleration_y",
        "heading_x","heading_y","heading_alignment",
        "v_para","v_perp","a_para","a_perp",
        "player_weight","height_inches","bmi",
        "ball_land_x","ball_land_y","dist_to_ball","angle_to_ball",
        "speed_squared","accel_magnitude","momentum_x","momentum_y","kinetic_energy",
        "role_targeted_receiver","role_defensive_coverage","role_passer","side_offense",
        "team_centroid_x","team_centroid_y","team_width","team_length",
        "rel_cx","rel_cy","form_bear_sin","form_bear_cos",
        "delta_frames","delta_t","frame_id",
        # GNN-lite
        "gnn_ally_dx_mean","gnn_ally_dy_mean","gnn_ally_dvx_mean","gnn_ally_dvy_mean",
        "gnn_opp_dx_mean","gnn_opp_dy_mean","gnn_opp_dvx_mean","gnn_opp_dvy_mean",
        "gnn_ally_cnt","gnn_opp_cnt","gnn_ally_dmin","gnn_ally_dmean","gnn_opp_dmin","gnn_opp_dmean",
        "gnn_d1","gnn_d2","gnn_d3",
    ]
    for lag in [1,2,3,4,5]:
        for c in ["x","y","velocity_x","velocity_y","s","a","v_para","v_perp","a_para","a_perp"]:
            base.append(f"{c}_lag{lag}")
    for win in [3,5]:
        for c in ["x","y","velocity_x","velocity_y","s","v_para","v_perp"]:
            base.append(f"{c}_rolling_mean_{win}")
            base.append(f"{c}_rolling_std_{win}")
    base += ["velocity_x_delta","velocity_y_delta","v_para_delta","v_perp_delta"]

    feats = [c for c in base if c in train_df.columns]
    feats = list(dict.fromkeys(feats + [c for c in train_df.columns if c.startswith("gnn_")]))
    return feats

# --------------------------- TRAIN / CV (single head) --------------------------- #
def _cat_params():
    p = dict(
        iterations=ITERATIONS, learning_rate=LR, depth=DEPTH, l2_leaf_reg=L2,
        random_seed=SEED, task_type=("GPU" if USE_GPU else "CPU"),
        loss_function="RMSE", early_stopping_rounds=EARLY, verbose=200,
        border_count=BORDER_CT
    )
    if USE_GPU and DEVICES is not None:
        p["devices"] = DEVICES  # e.g. '0:1' cho T4x2
    return p

def train_folded(X, yx, yy, sample_w, ids_group=None, base_x=None, base_y=None):
    folds = []
    if USE_GROUP_KFOLD and ids_group is not None:
        gkf = GroupKFold(n_splits=N_FOLDS)
        for tr, va in gkf.split(X, groups=ids_group):
            folds.append((tr, va))
    else:
        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        folds = list(kf.split(X))

    models_x, models_y, fold_rmse = [], [], []
    params = _cat_params()

    for i,(tr,va) in enumerate(folds, 1):
        print(f"\nFold {i}/{N_FOLDS} — train {len(tr):,} | val {len(va):,}")
        p_tr_x = CatPool(X[tr], yx[tr], weight=None if sample_w is None else sample_w[tr])
        p_va_x = CatPool(X[va], yx[va])
        p_tr_y = CatPool(X[tr], yy[tr], weight=None if sample_w is None else sample_w[tr])
        p_va_y = CatPool(X[va], yy[va])

        mx = CatBoostRegressor(**params).fit(p_tr_x, eval_set=p_va_x, verbose=200)
        my = CatBoostRegressor(**params).fit(p_tr_y, eval_set=p_va_y, verbose=200)
        models_x.append(mx); models_y.append(my)

        prx = mx.predict(X[va]); pry = my.predict(X[va])
        px  = np.clip(prx + base_x[va], 0, 120)
        py  = np.clip(pry + base_y[va], 0, 53.3)
        rmse = math.sqrt(0.5*(mean_squared_error(yx[va]+base_x[va], px) + mean_squared_error(yy[va]+base_y[va], py)))
        print(f"Fold {i} RMSE: {rmse:.5f}")
        fold_rmse.append(rmse)

    print("\nPer-fold:", [f"{v:.5f}" for v in fold_rmse])
    print(f"Mean ± std: {np.mean(fold_rmse):.5f} ± {np.std(fold_rmse):.5f}")
    return models_x, models_y, fold_rmse

# --------------------------- MAIN --------------------------- #
def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    # 1) Load
    tr_in, tr_out = load_all_train()

    # 2) FE
    print("\nEngineering features on train…")
    tr_in = engineer_advanced_features(tr_in)
    tr_in = add_formation_features(tr_in)
    tr_in = add_sequence_features(tr_in)

    # 3) GNN-lite neighbor embedding (train input)
    print("Computing neighbor embeddings (train)…")
    gnn_tr = compute_neighbor_embeddings(tr_in, k_neigh=K_NEIGH, radius=RADIUS, tau=TAU)

    # 4) Build training rows (merge last observed + future frames)
    train_df = create_training_rows(tr_in, tr_out)
    print("Train rows (pre-merge GNN):", train_df.shape)

    # Merge GNN features
    train_df = train_df.merge(gnn_tr, on=["game_id","play_id","nfl_id"], how="left")

    # 5) CA baseline (Δt) + residual targets
    bx, by = ca_baseline(
        train_df["x"].values, train_df["y"].values,
        train_df["velocity_x"].values, train_df["velocity_y"].values,
        train_df["delta_t"].values,
        train_df["acceleration_x"].values, train_df["acceleration_y"].values
    )
    base_rmse = math.sqrt(0.5*(mean_squared_error(train_df["target_x"], bx) +
                               mean_squared_error(train_df["target_y"], by)))
    print(f"[Baseline CA] RMSE: {base_rmse:.5f}")

    train_df["base_x"] = bx
    train_df["base_y"] = by
    train_df["res_x"]  = train_df["target_x"] - train_df["base_x"]
    train_df["res_y"]  = train_df["target_y"] - train_df["base_y"]

    # 6) Feature list
    feat_cols = build_feature_list(train_df)
    print(f"Using {len(feat_cols)} features (incl. GNN-lite & ball-frame).")

    # Clean & matrices
    df = train_df.dropna(subset=feat_cols + ["res_x","res_y"]).reset_index(drop=True)
    df[feat_cols] = df[feat_cols].replace([np.inf,-np.inf], np.nan).fillna(0.0)

    X  = df[feat_cols].to_numpy(np.float32)
    yx = df["res_x"].to_numpy(np.float32)
    yy = df["res_y"].to_numpy(np.float32)
    bxv= df["base_x"].to_numpy(np.float32)
    byv= df["base_y"].to_numpy(np.float32)

    # sample weights theo horizon (ưu tiên H lớn)
    w = 1.0 + W_H * df["delta_frames"].clip(lower=1, upper=5).to_numpy(np.float32)

    # groups để tránh leakage (giữ nguyên kiểu 0.63)
    groups = (df["game_id"].astype(str) + "_" + df["play_id"].astype(str) + "_" + df["nfl_id"].astype(str)).values

    # 7a) Train GLOBAL head
    print("\n[GLOBAL] training…")
    glob_x, glob_y, glob_rmse = train_folded(X, yx, yy, w, ids_group=groups, base_x=bxv, base_y=byv)

    # 7b) Train TR head
    mask_tr = df["role_targeted_receiver"]==1
    if mask_tr.sum() > 5000:
        print("\n[TR head] training…")
        X_tr  = X[mask_tr]; yx_tr = yx[mask_tr]; yy_tr = yy[mask_tr]
        bx_tr = bxv[mask_tr]; by_tr = byv[mask_tr]
        w_tr  = w[mask_tr]
        grp_tr= groups[mask_tr]
        tr_x, tr_y, tr_rmse = train_folded(X_tr, yx_tr, yy_tr, w_tr, ids_group=grp_tr, base_x=bx_tr, base_y=by_tr)
    else:
        print("\n[TR head] not enough rows, will fallback to GLOBAL.")
        tr_x, tr_y = None, None

    # 7c) Train DC head
    mask_dc = df["role_defensive_coverage"]==1
    if mask_dc.sum() > 5000:
        print("\n[DC head] training…")
        X_dc  = X[mask_dc]; yx_dc = yx[mask_dc]; yy_dc = yy[mask_dc]
        bx_dc = bxv[mask_dc]; by_dc = byv[mask_dc]
        w_dc  = w[mask_dc]
        grp_dc= groups[mask_dc]
        dc_x, dc_y, dc_rmse = train_folded(X_dc, yx_dc, yy_dc, w_dc, ids_group=grp_dc, base_x=bx_dc, base_y=by_dc)
    else:
        print("\n[DC head] not enough rows, will fallback to GLOBAL.")
        dc_x, dc_y = None, None

    # Save models
    with open(SAVE_DIR/"catboost_models_5fold_ROLEBLEND.pkl", "wb") as f:
        pickle.dump(
            {"global_x": glob_x, "global_y": glob_y,
             "tr_x": tr_x, "tr_y": tr_y,
             "dc_x": dc_x, "dc_y": dc_y,
             "features": feat_cols},
            f
        )
    print("Saved:", SAVE_DIR/"catboost_models_5fold_ROLEBLEND.pkl")

    # ---------------- Inference / Submission ----------------
    print("\nPreparing test…")
    te_in  = pd.read_csv(BASEDIR/"test_input.csv")
    te_tpl = pd.read_csv(BASEDIR/"test.csv")

    te_in  = engineer_advanced_features(te_in)
    te_in  = add_formation_features(te_in)
    te_in  = add_sequence_features(te_in)

    print("Computing neighbor embeddings (test)…")
    gnn_te = compute_neighbor_embeddings(te_in, k_neigh=K_NEIGH, radius=RADIUS, tau=TAU)

    agg_te = (
        te_in.sort_values(["game_id","play_id","nfl_id","frame_id"])
             .groupby(["game_id","play_id","nfl_id"], as_index=False)
             .tail(1)
             .rename(columns={"frame_id":"last_frame_id"})
    )

    te = te_tpl.merge(agg_te, on=["game_id","play_id","nfl_id"], how="left")
    te = te.merge(gnn_te, on=["game_id","play_id","nfl_id"], how="left")

    te["delta_frames"] = (te["frame_id"] - te["last_frame_id"]).clip(lower=0).astype(float)
    te["delta_t"] = te["delta_frames"] / 10.0

    # CA baseline test
    tbx, tby = ca_baseline(
        te["x"].values, te["y"].values,
        te["velocity_x"].values, te["velocity_y"].values,
        te["delta_t"].values,
        te["acceleration_x"].values, te["acceleration_y"].values
    )

    # features
    for c in build_feature_list(te):  # đảm bảo cột có mặt
        if c not in te.columns: te[c] = 0.0
    feat_cols = [c for c in (pickle.load(open(SAVE_DIR/"catboost_models_5fold_ROLEBLEND.pkl","rb"))["features"]) \
                 if c in te.columns] if (SAVE_DIR/"catboost_models_5fold_ROLEBLEND.pkl").exists() else build_feature_list(te)

    te.loc[:, feat_cols] = te[feat_cols].replace([np.inf,-np.inf], np.nan).fillna(0.0).to_numpy()
    Xtest = te[feat_cols].values.astype(np.float32)

    # Load models lại để chắc chắn scope (nếu chạy liên tục thì có sẵn biến)
    with open(SAVE_DIR/"catboost_models_5fold_ROLEBLEND.pkl","rb") as f:
        pack = pickle.load(f)
    glob_x, glob_y = pack["global_x"], pack["global_y"]
    tr_x, tr_y     = pack["tr_x"], pack["tr_y"]
    dc_x, dc_y     = pack["dc_x"], pack["dc_y"]

    # Predictors (ensembling over folds)
    def _avg_preds(model_list, X):
        return np.mean([m.predict(X) for m in model_list], axis=0).astype(np.float32)

    prx_global = _avg_preds(glob_x, Xtest); pry_global = _avg_preds(glob_y, Xtest)

    # Role heads (if exist)
    prx_tr = prx_global.copy(); pry_tr = pry_global.copy()
    if tr_x is not None and tr_y is not None:
        prx_tr = _avg_preds(tr_x, Xtest); pry_tr = _avg_preds(tr_y, Xtest)
    prx_dc = prx_global.copy(); pry_dc = pry_global.copy()
    if dc_x is not None and dc_y is not None:
        prx_dc = _avg_preds(dc_x, Xtest); pry_dc = _avg_preds(dc_y, Xtest)

    # Blend theo role
    is_tr = (te["player_role"].astype(str) == "Targeted Receiver").values
    is_dc = (te["player_role"].astype(str) == "Defensive Coverage").values

    pred_rx = prx_global.copy(); pred_ry = pry_global.copy()
    pred_rx[is_tr] = (1-ALPHA_TR)*prx_global[is_tr] + ALPHA_TR*prx_tr[is_tr]
    pred_ry[is_tr] = (1-ALPHA_TR)*pry_global[is_tr] + ALPHA_TR*pry_tr[is_tr]
    pred_rx[is_dc] = (1-ALPHA_DC)*prx_global[is_dc] + ALPHA_DC*prx_dc[is_dc]
    pred_ry[is_dc] = (1-ALPHA_DC)*pry_global[is_dc] + ALPHA_DC*pry_dc[is_dc]

    pred_x  = np.clip(pred_rx + tbx, 0.0, 120.0)
    pred_y  = np.clip(pred_ry + tby, 0.0, 53.3)

    sub = pd.DataFrame({
        "id": (te["game_id"].astype(str) + "_" +
               te["play_id"].astype(str) + "_" +
               te["nfl_id"].astype(str) + "_" +
               te["frame_id"].astype(str)),
        "x": pred_x,
        "y": pred_y
    })
    sub.to_csv(SAVE_DIR/"submission.csv", index=False)
    print("Saved submission:", SAVE_DIR/"submission.csv")

if __name__ == "__main__":
    main()
```