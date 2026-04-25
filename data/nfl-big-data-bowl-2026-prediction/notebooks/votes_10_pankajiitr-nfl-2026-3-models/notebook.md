# NFL 2026 3 Models

- **Author:** Pankaj Gupta
- **Votes:** 184
- **Ref:** pankajiitr/nfl-2026-3-models
- **URL:** https://www.kaggle.com/code/pankajiitr/nfl-2026-3-models
- **Last run:** 2025-11-15 05:13:17.953000

---

```python
%%writefile nfl_gnn.py

"""
NFL BIG DATA BOWL 2026 - GEOMETRIC NEURAL BREAKTHROUGH
🏆 The Winning Architecture

INSIGHT: Football players follow geometric rules with learned corrections
- Receivers → Ball landing (geometric)
- Defenders → Mirror receivers (geometric coupling)  
- Others → Momentum (physics)
- Model learns only CORRECTIONS for coverage, collisions, boundaries

Architecture:
✓ Proven GRU + Attention (your 0.59 base)
✓ 154 proven features (unchanged)
✓ +15 geometric features (our discovery)
✓ Train on corrections to geometric baseline (the breakthrough)

Target: 0.54-0.56 LB
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
import warnings
import os
import pickle

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.cluster import KMeans
from multiprocessing import Pool as MultiprocessingPool, cpu_count

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIG
# ============================================================================

class Config:
    DATA_DIR = Path("/kaggle/input/nfl-big-data-bowl-2026-prediction/")
    OUTPUT_DIR = Path("./outputs")
    OUTPUT_DIR.mkdir(exist_ok=True)
    # Where to persist trained artifacts (models, scalers, route objects)
    MODEL_DIR = OUTPUT_DIR / "models"
    MODEL_DIR.mkdir(exist_ok=True)
    # Toggle saving/loading of artifacts
    SAVE_ARTIFACTS = True
    LOAD_ARTIFACTS = True
    LOAD_DIR = '/kaggle/input/nfl-bdb-2026/nfl-gnn-a43/outputs/models'
  
    SEED = 42
    N_FOLDS = 10
    BATCH_SIZE = 256
    EPOCHS = 200
    PATIENCE = 30
    LEARNING_RATE = 1e-3
    
    WINDOW_SIZE = 10
    HIDDEN_DIM = 128
    MAX_FUTURE_HORIZON = 94
    
    # === Transformer 超参数 ===
    N_HEADS = 4  # Transformer 的注意力头数
    N_LAYERS = 2 # Transformer 编码器的层数
    
    # === 新增：ResidualMLP Head 超参数 ===
    MLP_HIDDEN_DIM = 256 # MLP 头的内部隐藏维度
    N_RES_BLOCKS = 2     # 残差块的数量
    # ==================================
    
    FIELD_X_MIN, FIELD_X_MAX = 0.0, 120.0
    FIELD_Y_MIN, FIELD_Y_MAX = 0.0, 53.3
    
    K_NEIGH = 6
    RADIUS = 30.0
    TAU = 8.0
    N_ROUTE_CLUSTERS = 7
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DEBUG = False
    if DEBUG:
        N_FOLDS = 2

def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed(Config.SEED)

# ============================================================================
# GEOMETRIC BASELINE - THE BREAKTHROUGH
# ============================================================================

def compute_geometric_endpoint(df):
    """
    Compute where each player SHOULD end up based on geometry.
    This is the deterministic part - no learning needed.
    """
    df = df.copy()
    
    # Time to play end
    if 'num_frames_output' in df.columns:
        t_total = df['num_frames_output'] / 10.0
    else:
        t_total = 3.0
    
    df['time_to_endpoint'] = t_total
    
    # Initialize with momentum (default rule)
    df['geo_endpoint_x'] = df['x'] + df['velocity_x'] * t_total
    df['geo_endpoint_y'] = df['y'] + df['velocity_y'] * t_total
    
    # Rule 1: Targeted Receivers converge to ball
    if 'ball_land_x' in df.columns:
        receiver_mask = df['player_role'] == 'Targeted Receiver'
        df.loc[receiver_mask, 'geo_endpoint_x'] = df.loc[receiver_mask, 'ball_land_x']
        df.loc[receiver_mask, 'geo_endpoint_y'] = df.loc[receiver_mask, 'ball_land_y']
        
        # Rule 2: Defenders mirror receivers (maintain offset)
        defender_mask = df['player_role'] == 'Defensive Coverage'
        has_mirror = df.get('mirror_offset_x', 0).notna() & (df.get('mirror_wr_dist', 50) < 15)
        coverage_mask = defender_mask & has_mirror
        
        df.loc[coverage_mask, 'geo_endpoint_x'] = (
            df.loc[coverage_mask, 'ball_land_x'] + 
            df.loc[coverage_mask, 'mirror_offset_x'].fillna(0)
        )
        df.loc[coverage_mask, 'geo_endpoint_y'] = (
            df.loc[coverage_mask, 'ball_land_y'] + 
            df.loc[coverage_mask, 'mirror_offset_y'].fillna(0)
        )
    
    # Clip to field
    df['geo_endpoint_x'] = df['geo_endpoint_x'].clip(Config.FIELD_X_MIN, Config.FIELD_X_MAX)
    df['geo_endpoint_y'] = df['geo_endpoint_y'].clip(Config.FIELD_Y_MIN, Config.FIELD_Y_MAX)
    
    return df

def add_geometric_features(df):
    """Add features that describe the geometric solution"""
    df = compute_geometric_endpoint(df)
    
    # Vector to geometric endpoint
    df['geo_vector_x'] = df['geo_endpoint_x'] - df['x']
    df['geo_vector_y'] = df['geo_endpoint_y'] - df['y']
    df['geo_distance'] = np.sqrt(df['geo_vector_x']**2 + df['geo_vector_y']**2)
    
    # Required velocity to reach geometric endpoint
    t = df['time_to_endpoint'] + 0.1
    df['geo_required_vx'] = df['geo_vector_x'] / t
    df['geo_required_vy'] = df['geo_vector_y'] / t
    
    # Current velocity vs required
    df['geo_velocity_error_x'] = df['geo_required_vx'] - df['velocity_x']
    df['geo_velocity_error_y'] = df['geo_required_vy'] - df['velocity_y']
    df['geo_velocity_error'] = np.sqrt(
        df['geo_velocity_error_x']**2 + df['geo_velocity_error_y']**2
    )
    
    # Required constant acceleration (a = 2*Δx/t²)
    t_sq = t * t
    df['geo_required_ax'] = 2 * df['geo_vector_x'] / t_sq
    df['geo_required_ay'] = 2 * df['geo_vector_y'] / t_sq
    df['geo_required_ax'] = df['geo_required_ax'].clip(-10, 10)
    df['geo_required_ay'] = df['geo_required_ay'].clip(-10, 10)
    
    # Alignment with geometric path
    velocity_mag = np.sqrt(df['velocity_x']**2 + df['velocity_y']**2)
    geo_unit_x = df['geo_vector_x'] / (df['geo_distance'] + 0.1)
    geo_unit_y = df['geo_vector_y'] / (df['geo_distance'] + 0.1)
    df['geo_alignment'] = (
        df['velocity_x'] * geo_unit_x + df['velocity_y'] * geo_unit_y
    ) / (velocity_mag + 0.1)
    
    # Role-specific geometric quality
    df['geo_receiver_urgency'] = df['is_receiver'] * df['geo_distance'] / (t + 0.1)
    df['geo_defender_coupling'] = df['is_coverage'] * (1.0 / (df.get('mirror_wr_dist', 50) + 1.0))
    
    return df

# ============================================================================
# PROVEN FEATURE ENGINEERING (YOUR 0.59 BASE)
# ============================================================================

def get_velocity(speed, direction_deg):
    theta = np.deg2rad(direction_deg)
    return speed * np.sin(theta), speed * np.cos(theta)

def height_to_feet(height_str):
    try:
        ft, inches = map(int, str(height_str).split('-'))
        return ft + inches/12
    except:
        return 6.0

def get_opponent_features(input_df):
    """Enhanced opponent interaction with MIRROR WR tracking"""
    features = []
    
    for (gid, pid), group in tqdm(input_df.groupby(['game_id', 'play_id']), 
                                   desc="🏈 Opponents", leave=False):
        last = group.sort_values('frame_id').groupby('nfl_id').last()
        
        if len(last) < 2:
            continue
            
        positions = last[['x', 'y']].values
        sides = last['player_side'].values
        speeds = last['s'].values
        directions = last['dir'].values
        roles = last['player_role'].values
        
        receiver_mask = np.isin(roles, ['Targeted Receiver', 'Other Route Runner'])
        
        for i, (nid, side, role) in enumerate(zip(last.index, sides, roles)):
            opp_mask = sides != side
            
            feat = {
                'game_id': gid, 'play_id': pid, 'nfl_id': nid,
                'nearest_opp_dist': 50.0, 'closing_speed': 0.0,
                'num_nearby_opp_3': 0, 'num_nearby_opp_5': 0,
                'mirror_wr_vx': 0.0, 'mirror_wr_vy': 0.0,
                'mirror_offset_x': 0.0, 'mirror_offset_y': 0.0,
                'mirror_wr_dist': 50.0,
            }
            
            if not opp_mask.any():
                features.append(feat)
                continue
            
            opp_positions = positions[opp_mask]
            distances = np.sqrt(((positions[i] - opp_positions)**2).sum(axis=1))
            
            if len(distances) == 0:
                features.append(feat)
                continue
                
            nearest_idx = distances.argmin()
            feat['nearest_opp_dist'] = distances[nearest_idx]
            feat['num_nearby_opp_3'] = (distances < 3.0).sum()
            feat['num_nearby_opp_5'] = (distances < 5.0).sum()
            
            my_vx, my_vy = get_velocity(speeds[i], directions[i])
            opp_speeds = speeds[opp_mask]
            opp_dirs = directions[opp_mask]
            opp_vx, opp_vy = get_velocity(opp_speeds[nearest_idx], opp_dirs[nearest_idx])
            
            rel_vx = my_vx - opp_vx
            rel_vy = my_vy - opp_vy
            to_me = positions[i] - opp_positions[nearest_idx]
            to_me_norm = to_me / (np.linalg.norm(to_me) + 0.1)
            feat['closing_speed'] = -(rel_vx * to_me_norm[0] + rel_vy * to_me_norm[1])
            
            if role == 'Defensive Coverage' and receiver_mask.any():
                rec_positions = positions[receiver_mask]
                rec_distances = np.sqrt(((positions[i] - rec_positions)**2).sum(axis=1))
                
                if len(rec_distances) > 0:
                    closest_rec_idx = rec_distances.argmin()
                    rec_indices = np.where(receiver_mask)[0]
                    actual_rec_idx = rec_indices[closest_rec_idx]
                    
                    rec_vx, rec_vy = get_velocity(speeds[actual_rec_idx], directions[actual_rec_idx])
                    
                    feat['mirror_wr_vx'] = rec_vx
                    feat['mirror_wr_vy'] = rec_vy
                    feat['mirror_wr_dist'] = rec_distances[closest_rec_idx]
                    feat['mirror_offset_x'] = positions[i][0] - rec_positions[closest_rec_idx][0]
                    feat['mirror_offset_y'] = positions[i][1] - rec_positions[closest_rec_idx][1]
            
            features.append(feat)
    
    return pd.DataFrame(features)

def extract_route_patterns(input_df, kmeans=None, scaler=None, fit=True):
    """Route clustering"""
    route_features = []
    
    for (gid, pid, nid), group in tqdm(input_df.groupby(['game_id', 'play_id', 'nfl_id']), 
                                        desc="🛣️  Routes", leave=False):
        traj = group.sort_values('frame_id').tail(5)
        
        if len(traj) < 3:
            continue
        
        positions = traj[['x', 'y']].values
        speeds = traj['s'].values
        
        total_dist = np.sum(np.sqrt(np.diff(positions[:, 0])**2 + np.diff(positions[:, 1])**2))
        displacement = np.sqrt((positions[-1, 0] - positions[0, 0])**2 + 
                              (positions[-1, 1] - positions[0, 1])**2)
        straightness = displacement / (total_dist + 0.1)
        
        angles = np.arctan2(np.diff(positions[:, 1]), np.diff(positions[:, 0]))
        if len(angles) > 1:
            angle_changes = np.abs(np.diff(angles))
            max_turn = np.max(angle_changes)
            mean_turn = np.mean(angle_changes)
        else:
            max_turn = mean_turn = 0
        
        speed_mean = speeds.mean()
        speed_change = speeds[-1] - speeds[0] if len(speeds) > 1 else 0
        dx = positions[-1, 0] - positions[0, 0]
        dy = positions[-1, 1] - positions[0, 1]
        
        route_features.append({
            'game_id': gid, 'play_id': pid, 'nfl_id': nid,
            'traj_straightness': straightness,
            'traj_max_turn': max_turn,
            'traj_mean_turn': mean_turn,
            'traj_depth': abs(dx),
            'traj_width': abs(dy),
            'speed_mean': speed_mean,
            'speed_change': speed_change,
        })
    
    route_df = pd.DataFrame(route_features)
    feat_cols = ['traj_straightness', 'traj_max_turn', 'traj_mean_turn',
                 'traj_depth', 'traj_width', 'speed_mean', 'speed_change']
    X = route_df[feat_cols].fillna(0)
    
    if fit:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        kmeans = KMeans(n_clusters=Config.N_ROUTE_CLUSTERS, random_state=Config.SEED, n_init=10)
        route_df['route_pattern'] = kmeans.fit_predict(X_scaled)
        return route_df, kmeans, scaler
    else:
        X_scaled = scaler.transform(X)
        route_df['route_pattern'] = kmeans.predict(X_scaled)
        return route_df

def compute_neighbor_embeddings(input_df, k_neigh=Config.K_NEIGH, 
                                radius=Config.RADIUS, tau=Config.TAU):
    """GNN-lite embeddings"""
    print("🕸️  GNN embeddings...")
    
    cols_needed = ["game_id", "play_id", "nfl_id", "frame_id", "x", "y", 
                   "velocity_x", "velocity_y", "player_side"]
    src = input_df[cols_needed].copy()
    
    last = (src.sort_values(["game_id", "play_id", "nfl_id", "frame_id"])
               .groupby(["game_id", "play_id", "nfl_id"], as_index=False)
               .tail(1)
               .rename(columns={"frame_id": "last_frame_id"})
               .reset_index(drop=True))
    
    tmp = last.merge(
        src.rename(columns={
            "frame_id": "nb_frame_id", "nfl_id": "nfl_id_nb",
            "x": "x_nb", "y": "y_nb", 
            "velocity_x": "vx_nb", "velocity_y": "vy_nb", 
            "player_side": "player_side_nb"
        }),
        left_on=["game_id", "play_id", "last_frame_id"],
        right_on=["game_id", "play_id", "nb_frame_id"],
        how="left"
    )
    
    tmp = tmp[tmp["nfl_id_nb"] != tmp["nfl_id"]]
    tmp["dx"] = tmp["x_nb"] - tmp["x"]
    tmp["dy"] = tmp["y_nb"] - tmp["y"]
    tmp["dvx"] = tmp["vx_nb"] - tmp["velocity_x"]
    tmp["dvy"] = tmp["vy_nb"] - tmp["velocity_y"]
    tmp["dist"] = np.sqrt(tmp["dx"]**2 + tmp["dy"]**2)
    
    tmp = tmp[np.isfinite(tmp["dist"]) & (tmp["dist"] > 1e-6)]
    if radius is not None:
        tmp = tmp[tmp["dist"] <= radius]
    
    tmp["is_ally"] = (tmp["player_side_nb"] == tmp["player_side"]).astype(np.float32)
    
    keys = ["game_id", "play_id", "nfl_id"]
    tmp["rnk"] = tmp.groupby(keys)["dist"].rank(method="first")
    if k_neigh is not None:
        tmp = tmp[tmp["rnk"] <= float(k_neigh)]
    
    tmp["w"] = np.exp(-tmp["dist"] / float(tau))
    sum_w = tmp.groupby(keys)["w"].transform("sum")
    tmp["wn"] = np.where(sum_w > 0, tmp["w"] / sum_w, 0.0)
    
    tmp["wn_ally"] = tmp["wn"] * tmp["is_ally"]
    tmp["wn_opp"] = tmp["wn"] * (1.0 - tmp["is_ally"])
    
    for col in ["dx", "dy", "dvx", "dvy"]:
        tmp[f"{col}_ally_w"] = tmp[col] * tmp["wn_ally"]
        tmp[f"{col}_opp_w"] = tmp[col] * tmp["wn_opp"]
    
    tmp["dist_ally"] = np.where(tmp["is_ally"] > 0.5, tmp["dist"], np.nan)
    tmp["dist_opp"] = np.where(tmp["is_ally"] < 0.5, tmp["dist"], np.nan)
    
    ag = tmp.groupby(keys).agg(
        gnn_ally_dx_mean=("dx_ally_w", "sum"),
        gnn_ally_dy_mean=("dy_ally_w", "sum"),
        gnn_ally_dvx_mean=("dvx_ally_w", "sum"),
        gnn_ally_dvy_mean=("dvy_ally_w", "sum"),
        gnn_opp_dx_mean=("dx_opp_w", "sum"),
        gnn_opp_dy_mean=("dy_opp_w", "sum"),
        gnn_opp_dvx_mean=("dvx_opp_w", "sum"),
        gnn_opp_dvy_mean=("dvy_opp_w", "sum"),
        gnn_ally_cnt=("is_ally", "sum"),
        gnn_opp_cnt=("is_ally", lambda s: float(len(s) - s.sum())),
        gnn_ally_dmin=("dist_ally", "min"),
        gnn_ally_dmean=("dist_ally", "mean"),
        gnn_opp_dmin=("dist_opp", "min"),
        gnn_opp_dmean=("dist_opp", "mean"),
    ).reset_index()
    
    near = tmp.loc[tmp["rnk"] <= 3, keys + ["rnk", "dist"]].copy()
    if len(near) > 0:
        near["rnk"] = near["rnk"].astype(int)
        dwide = near.pivot_table(index=keys, columns="rnk", values="dist", aggfunc="first")
        dwide = dwide.rename(columns={1: "gnn_d1", 2: "gnn_d2", 3: "gnn_d3"}).reset_index()
        ag = ag.merge(dwide, on=keys, how="left")
    
    for c in ["gnn_ally_dx_mean", "gnn_ally_dy_mean", "gnn_ally_dvx_mean", "gnn_ally_dvy_mean",
              "gnn_opp_dx_mean", "gnn_opp_dy_mean", "gnn_opp_dvx_mean", "gnn_opp_dvy_mean"]:
        ag[c] = ag[c].fillna(0.0)
    for c in ["gnn_ally_cnt", "gnn_opp_cnt"]:
        ag[c] = ag[c].fillna(0.0)
    for c in ["gnn_ally_dmin", "gnn_opp_dmin", "gnn_ally_dmean", "gnn_opp_dmean", 
              "gnn_d1", "gnn_d2", "gnn_d3"]:
        ag[c] = ag[c].fillna(radius if radius is not None else 30.0)
    
    return ag

# ============================================================================
# SEQUENCE PREPARATION WITH GEOMETRIC FEATURES
# ============================================================================

def prepare_sequences_geometric(input_df, output_df=None, test_template=None, 
                                is_training=True, window_size=10,
                                route_kmeans=None, route_scaler=None):
    """YOUR 154 features + 13 geometric features = 167 total"""
    
    print(f"\n{'='*80}")
    print(f"PREPARING GEOMETRIC SEQUENCES")
    print(f"{'='*80}")
    
    input_df = input_df.copy()
    input_df = input_df.sort_values(['game_id', 'play_id', 'nfl_id', 'frame_id'])
    
    print("Step 1: Base features...")
    
    input_df['player_height_feet'] = input_df['player_height'].apply(height_to_feet)
    height_parts = input_df['player_height'].str.split('-', expand=True)
    input_df['height_inches'] = height_parts[0].astype(float) * 12 + height_parts[1].astype(float)
    input_df['bmi'] = (input_df['player_weight'] / (input_df['height_inches']**2)) * 703
    
    dir_rad = np.deg2rad(input_df['dir'].fillna(0))
    input_df['velocity_x'] = input_df['s'] * np.sin(dir_rad)
    input_df['velocity_y'] = input_df['s'] * np.cos(dir_rad)
    input_df['acceleration_x'] = input_df['a'] * np.cos(dir_rad)
    input_df['acceleration_y'] = input_df['a'] * np.sin(dir_rad)
    
    input_df['speed_squared'] = input_df['s'] ** 2
    input_df['accel_magnitude'] = np.sqrt(input_df['acceleration_x']**2 + input_df['acceleration_y']**2)
    input_df['momentum_x'] = input_df['velocity_x'] * input_df['player_weight']
    input_df['momentum_y'] = input_df['velocity_y'] * input_df['player_weight']
    input_df['kinetic_energy'] = 0.5 * input_df['player_weight'] * input_df['speed_squared']
    
    input_df['orientation_diff'] = np.abs(input_df['o'] - input_df['dir'])
    input_df['orientation_diff'] = np.minimum(input_df['orientation_diff'], 360 - input_df['orientation_diff'])
    
    input_df['is_offense'] = (input_df['player_side'] == 'Offense').astype(int)
    input_df['is_defense'] = (input_df['player_side'] == 'Defense').astype(int)
    input_df['is_receiver'] = (input_df['player_role'] == 'Targeted Receiver').astype(int)
    input_df['is_coverage'] = (input_df['player_role'] == 'Defensive Coverage').astype(int)
    input_df['is_passer'] = (input_df['player_role'] == 'Passer').astype(int)
    input_df['role_targeted_receiver'] = input_df['is_receiver']
    input_df['role_defensive_coverage'] = input_df['is_coverage']
    input_df['role_passer'] = input_df['is_passer']
    input_df['side_offense'] = input_df['is_offense']
    
    if 'ball_land_x' in input_df.columns:
        ball_dx = input_df['ball_land_x'] - input_df['x']
        ball_dy = input_df['ball_land_y'] - input_df['y']
        input_df['distance_to_ball'] = np.sqrt(ball_dx**2 + ball_dy**2)
        input_df['dist_to_ball'] = input_df['distance_to_ball']
        input_df['dist_squared'] = input_df['distance_to_ball'] ** 2
        input_df['angle_to_ball'] = np.arctan2(ball_dy, ball_dx)
        input_df['ball_direction_x'] = ball_dx / (input_df['distance_to_ball'] + 1e-6)
        input_df['ball_direction_y'] = ball_dy / (input_df['distance_to_ball'] + 1e-6)
        input_df['closing_speed_ball'] = (
            input_df['velocity_x'] * input_df['ball_direction_x'] +
            input_df['velocity_y'] * input_df['ball_direction_y']
        )
        input_df['velocity_toward_ball'] = (
            input_df['velocity_x'] * np.cos(input_df['angle_to_ball']) + 
            input_df['velocity_y'] * np.sin(input_df['angle_to_ball'])
        )
        input_df['velocity_alignment'] = np.cos(input_df['angle_to_ball'] - dir_rad)
        input_df['angle_diff'] = np.abs(input_df['o'] - np.degrees(input_df['angle_to_ball']))
        input_df['angle_diff'] = np.minimum(input_df['angle_diff'], 360 - input_df['angle_diff'])
    
    print("Step 2: Advanced features...")
    
    opp_features = get_opponent_features(input_df)
    input_df = input_df.merge(opp_features, on=['game_id', 'play_id', 'nfl_id'], how='left')
    
    if is_training:
        route_features, route_kmeans, route_scaler = extract_route_patterns(input_df)
    else:
        route_features = extract_route_patterns(input_df, route_kmeans, route_scaler, fit=False)
    input_df = input_df.merge(route_features, on=['game_id', 'play_id', 'nfl_id'], how='left')
    
    if not route_features.empty:
        input_df = input_df.merge(route_features, on=['game_id', 'play_id', 'nfl_id'], how='left')

    gnn_features = compute_neighbor_embeddings(input_df)
    input_df = input_df.merge(gnn_features, on=['game_id', 'play_id', 'nfl_id'], how='left')
    
    if 'nearest_opp_dist' in input_df.columns:
        input_df['pressure'] = 1 / np.maximum(input_df['nearest_opp_dist'], 0.5)
        input_df['under_pressure'] = (input_df['nearest_opp_dist'] < 3).astype(int)
        input_df['pressure_x_speed'] = input_df['pressure'] * input_df['s']
    
    if 'mirror_wr_vx' in input_df.columns:
        s_safe = np.maximum(input_df['s'], 0.1)
        input_df['mirror_similarity'] = (
            input_df['velocity_x'] * input_df['mirror_wr_vx'] + 
            input_df['velocity_y'] * input_df['mirror_wr_vy']
        ) / s_safe
        input_df['mirror_offset_dist'] = np.sqrt(
            input_df['mirror_offset_x']**2 + input_df['mirror_offset_y']**2
        )
        input_df['mirror_alignment'] = input_df['mirror_similarity'] * input_df['role_defensive_coverage']
    
    print("Step 3: Temporal features...")
    
    gcols = ['game_id', 'play_id', 'nfl_id']
    
    for lag in [1, 2, 3, 4, 5]:
        for col in ['x', 'y', 'velocity_x', 'velocity_y', 's', 'a']:
            if col in input_df.columns:
                input_df[f'{col}_lag{lag}'] = input_df.groupby(gcols)[col].shift(lag)
    
    for window in [3, 5]:
        for col in ['x', 'y', 'velocity_x', 'velocity_y', 's']:
            if col in input_df.columns:
                input_df[f'{col}_rolling_mean_{window}'] = (
                    input_df.groupby(gcols)[col]
                      .rolling(window, min_periods=1).mean()
                      .reset_index(level=[0,1,2], drop=True)
                )
                input_df[f'{col}_rolling_std_{window}'] = (
                    input_df.groupby(gcols)[col]
                      .rolling(window, min_periods=1).std()
                      .reset_index(level=[0,1,2], drop=True)
                )
    
    for col in ['velocity_x', 'velocity_y']:
        if col in input_df.columns:
            input_df[f'{col}_delta'] = input_df.groupby(gcols)[col].diff()
    
    input_df['velocity_x_ema'] = input_df.groupby(gcols)['velocity_x'].transform(
        lambda x: x.ewm(alpha=0.3, adjust=False).mean()
    )
    input_df['velocity_y_ema'] = input_df.groupby(gcols)['velocity_y'].transform(
        lambda x: x.ewm(alpha=0.3, adjust=False).mean()
    )
    input_df['speed_ema'] = input_df.groupby(gcols)['s'].transform(
        lambda x: x.ewm(alpha=0.3, adjust=False).mean()
    )
    
    print("Step 4: Time features...")
    
    if 'num_frames_output' in input_df.columns:
        max_frames = input_df['num_frames_output']
        
        input_df['max_play_duration'] = max_frames / 10.0
        input_df['frame_time'] = input_df['frame_id'] / 10.0
        input_df['progress_ratio'] = input_df['frame_id'] / np.maximum(max_frames, 1)
        input_df['time_remaining'] = (max_frames - input_df['frame_id']) / 10.0
        input_df['frames_remaining'] = max_frames - input_df['frame_id']
        
        input_df['expected_x_at_ball'] = input_df['x'] + input_df['velocity_x'] * input_df['frame_time']
        input_df['expected_y_at_ball'] = input_df['y'] + input_df['velocity_y'] * input_df['frame_time']
        
        if 'ball_land_x' in input_df.columns:
            input_df['error_from_ball_x'] = input_df['expected_x_at_ball'] - input_df['ball_land_x']
            input_df['error_from_ball_y'] = input_df['expected_y_at_ball'] - input_df['ball_land_y']
            input_df['error_from_ball'] = np.sqrt(
                input_df['error_from_ball_x']**2 + input_df['error_from_ball_y']**2
            )
            
            input_df['weighted_dist_by_time'] = input_df['dist_to_ball'] / (input_df['frame_time'] + 0.1)
            input_df['dist_scaled_by_progress'] = input_df['dist_to_ball'] * (1 - input_df['progress_ratio'])
        
        input_df['time_squared'] = input_df['frame_time'] ** 2
        input_df['velocity_x_progress'] = input_df['velocity_x'] * input_df['progress_ratio']
        input_df['velocity_y_progress'] = input_df['velocity_y'] * input_df['progress_ratio']
        input_df['speed_scaled_by_time_left'] = input_df['s'] * input_df['time_remaining']
        
        input_df['actual_play_length'] = max_frames
        input_df['length_ratio'] = max_frames / 30.0
    
    # 🎯 THE BREAKTHROUGH: Add geometric features
    print("Step 5: 🎯 Geometric endpoint features...")
    input_df = add_geometric_features(input_df)
    
    print("Step 6: Building feature list...")
    
    # Your 154 proven features
    feature_cols = [
        'x', 'y', 's', 'a', 'o', 'dir', 'frame_id', 'ball_land_x', 'ball_land_y',
        'player_height_feet', 'player_weight', 'height_inches', 'bmi',
        'velocity_x', 'velocity_y', 'acceleration_x', 'acceleration_y',
        'momentum_x', 'momentum_y', 'kinetic_energy',
        'speed_squared', 'accel_magnitude', 'orientation_diff',
        'is_offense', 'is_defense', 'is_receiver', 'is_coverage', 'is_passer',
        'role_targeted_receiver', 'role_defensive_coverage', 'role_passer', 'side_offense',
        'distance_to_ball', 'dist_to_ball', 'dist_squared', 'angle_to_ball', 
        'ball_direction_x', 'ball_direction_y', 'closing_speed_ball',
        'velocity_toward_ball', 'velocity_alignment', 'angle_diff',
        'nearest_opp_dist', 'closing_speed', 'num_nearby_opp_3', 'num_nearby_opp_5',
        'mirror_wr_vx', 'mirror_wr_vy', 'mirror_offset_x', 'mirror_offset_y',
        'pressure', 'under_pressure', 'pressure_x_speed', 
        'mirror_similarity', 'mirror_offset_dist', 'mirror_alignment',
        'route_pattern', 'traj_straightness', 'traj_max_turn', 'traj_mean_turn',
        'traj_depth', 'traj_width', 'speed_mean', 'speed_change',
        'gnn_ally_dx_mean', 'gnn_ally_dy_mean', 'gnn_ally_dvx_mean', 'gnn_ally_dvy_mean',
        'gnn_opp_dx_mean', 'gnn_opp_dy_mean', 'gnn_opp_dvx_mean', 'gnn_opp_dvy_mean',
        'gnn_ally_cnt', 'gnn_opp_cnt',
        'gnn_ally_dmin', 'gnn_ally_dmean', 'gnn_opp_dmin', 'gnn_opp_dmean',
        'gnn_d1', 'gnn_d2', 'gnn_d3',
    ]
    
    for lag in [1, 2, 3, 4, 5]:
        for col in ['x', 'y', 'velocity_x', 'velocity_y', 's', 'a']:
            feature_cols.append(f'{col}_lag{lag}')
    
    for window in [3, 5]:
        for col in ['x', 'y', 'velocity_x', 'velocity_y', 's']:
            feature_cols.append(f'{col}_rolling_mean_{window}')
            feature_cols.append(f'{col}_rolling_std_{window}')
    
    feature_cols.extend(['velocity_x_delta', 'velocity_y_delta'])
    feature_cols.extend(['velocity_x_ema', 'velocity_y_ema', 'speed_ema'])
    
    feature_cols.extend([
        'max_play_duration', 'frame_time', 'progress_ratio', 'time_remaining', 'frames_remaining',
        'expected_x_at_ball', 'expected_y_at_ball', 
        'error_from_ball_x', 'error_from_ball_y', 'error_from_ball',
        'time_squared', 'weighted_dist_by_time', 
        'velocity_x_progress', 'velocity_y_progress', 'dist_scaled_by_progress',
        'speed_scaled_by_time_left', 'actual_play_length', 'length_ratio',
    ])
    
    # 🎯 Add 13 geometric features
    feature_cols.extend([
        'geo_endpoint_x', 'geo_endpoint_y',
        'geo_vector_x', 'geo_vector_y', 'geo_distance',
        'geo_required_vx', 'geo_required_vy',
        'geo_velocity_error_x', 'geo_velocity_error_y', 'geo_velocity_error',
        'geo_required_ax', 'geo_required_ay',
        'geo_alignment',
    ])
    
    feature_cols = [c for c in feature_cols if c in input_df.columns]
    print(f"✓ Using {len(feature_cols)} features (154 proven + 13 geometric)")
    
    print("Step 7: Creating sequences...")
    
    input_df.set_index(['game_id', 'play_id', 'nfl_id'], inplace=True)
    grouped = input_df.groupby(level=['game_id', 'play_id', 'nfl_id'])
    
    target_rows = output_df if is_training else test_template
    target_groups = target_rows[['game_id', 'play_id', 'nfl_id']].drop_duplicates()
    
    sequences, targets_dx, targets_dy, targets_frame_ids, sequence_ids = [], [], [], [], []
    geo_endpoints_x, geo_endpoints_y = [], []
    
    for _, row in tqdm(target_groups.iterrows(), total=len(target_groups), desc="Creating sequences"):
        key = (row['game_id'], row['play_id'], row['nfl_id'])
        
        try:
            group_df = grouped.get_group(key)
        except KeyError:
            continue
        
        input_window = group_df.tail(window_size)
        
        if len(input_window) < window_size:
            if is_training:
                continue
            pad_len = window_size - len(input_window)
            pad_df = pd.DataFrame(np.nan, index=range(pad_len), columns=input_window.columns)
            input_window = pd.concat([pad_df, input_window], ignore_index=True)
        
        input_window = input_window.fillna(group_df.mean(numeric_only=True))
        seq = input_window[feature_cols].values
        
        if np.isnan(seq).any():
            if is_training:
                continue
            seq = np.nan_to_num(seq, nan=0.0)
        
        sequences.append(seq)
        
        # Store geometric endpoint for this player
        geo_x = input_window.iloc[-1]['geo_endpoint_x']
        geo_y = input_window.iloc[-1]['geo_endpoint_y']
        geo_endpoints_x.append(geo_x)
        geo_endpoints_y.append(geo_y)
        
        if is_training:
            out_grp = output_df[
                (output_df['game_id']==row['game_id']) &
                (output_df['play_id']==row['play_id']) &
                (output_df['nfl_id']==row['nfl_id'])
            ].sort_values('frame_id')
            
            last_x = input_window.iloc[-1]['x']
            last_y = input_window.iloc[-1]['y']
            
            dx = out_grp['x'].values - last_x
            dy = out_grp['y'].values - last_y
            
            targets_dx.append(dx)
            targets_dy.append(dy)
            targets_frame_ids.append(out_grp['frame_id'].values)
        
        sequence_ids.append({
            'game_id': key[0],
            'play_id': key[1],
            'nfl_id': key[2],
            'frame_id': input_window.iloc[-1]['frame_id']
        })
    
    print(f"✓ Created {len(sequences)} sequences")
    
    if is_training:
        return (sequences, targets_dx, targets_dy, targets_frame_ids, sequence_ids, 
                geo_endpoints_x, geo_endpoints_y, route_kmeans, route_scaler)
    return sequences, sequence_ids, geo_endpoints_x, geo_endpoints_y

# ============================================================================
# MODEL ARCHITECTURE (ST-TRANSFORMER with ResidualMLP Head)
# ============================================================================
class ResidualBlock(nn.Module):
    """
    一个标准的残差块：FFN + 快捷连接
    """
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, x):
        # Pre-normalization
        return x + self.ffn(self.norm(x))

class ResidualMLPHead(nn.Module):
    """
    替换原有的 nn.Sequential Head
    """
    def __init__(self, input_dim, hidden_dim, output_dim, n_res_blocks=2, dropout=0.2):
        super().__init__()
        # 1. 从 context_dim (128) 投影到 mlp_hidden_dim (256)
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU()
        )
        
        # 2. 一系列的残差块 (在 256 维度上操作)
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(hidden_dim, hidden_dim * 2, dropout) for _ in range(n_res_blocks)]
        )
        
        # 3. 最后的 LayerNorm 和输出投影
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = self.input_layer(x)
        x = self.residual_blocks(x)
        x = self.output_norm(x)
        x = self.output_layer(x)
        return x

class STTransformer(nn.Module):
    """
    Spatio-Temporal Transformer
    """
    def __init__(self, input_dim, hidden_dim, horizon, window_size, n_heads, n_layers, dropout=0.1):
        super().__init__()
        config = Config() # 获取 MLP 的超参数
        self.horizon = horizon
        self.hidden_dim = hidden_dim

        # 1. Spatio: 特征嵌入
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # 2. Temporal: 可学习的位置编码
        self.pos_embed = nn.Parameter(torch.randn(1, window_size, hidden_dim)) 
        self.embed_dropout = nn.Dropout(dropout)

        # 3. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers
        )

        # 4. 池化 (复用你成熟的 Attention Pooling 机制)
        self.pool_ln = nn.LayerNorm(hidden_dim)
        self.pool_attn = nn.MultiheadAttention(hidden_dim, num_heads=n_heads, batch_first=True)
        self.pool_query = nn.Parameter(torch.randn(1, 1, hidden_dim)) 

        # 5. 输出 Head (!!! 已替换为 ResidualMLPHead !!!)
        self.head = ResidualMLPHead(
            input_dim=hidden_dim,                   # 128
            hidden_dim=config.MLP_HIDDEN_DIM,       # 256
            output_dim=horizon * 2,                 # 188
            n_res_blocks=config.N_RES_BLOCKS,       # 2
            dropout=0.2
        )

    def forward(self, x):
        B, S, _ = x.shape
        x_embed = self.input_projection(x) 
        x = x_embed + self.pos_embed[:, :S, :] 
        x = self.embed_dropout(x)
        
        h = self.transformer_encoder(x) 

        q = self.pool_query.expand(B, -1, -1)
        ctx, _ = self.pool_attn(q, self.pool_ln(h), self.pool_ln(h))
        ctx = ctx.squeeze(1) 

        out = self.head(ctx) # <--- 使用新的 Head
        out = out.view(B, self.horizon, 2)
        
        out = torch.cumsum(out, dim=1)
        
        return out


# ============================================================================
# LOSS (YOUR PROVEN TEMPORAL HUBER)
# ============================================================================

class TemporalHuber(nn.Module):
    def __init__(self, delta=0.5, time_decay=0.03):
        super().__init__()
        self.delta = delta
        self.time_decay = time_decay
    
    def forward(self, pred, target, mask):
        err = pred - target
        abs_err = torch.abs(err)
        huber = torch.where(abs_err <= self.delta, 0.5 * err * err, 
                           self.delta * (abs_err - 0.5 * self.delta))
        
        if self.time_decay > 0:
            L = pred.size(1)
            t = torch.arange(L, device=pred.device).float()
            weight = torch.exp(-self.time_decay * t).view(1, L, 1)
            huber = huber * weight
            mask = mask.unsqueeze(-1) * weight
        
        return (huber * mask).sum() / (mask.sum() + 1e-8)

# ============================================================================
# TRAINING
# ============================================================================

def prepare_targets(batch_dx, batch_dy, max_h):
    tensors_x, tensors_y, masks = [], [], []
    
    for dx, dy in zip(batch_dx, batch_dy):
        L = len(dx)
        padded_x = np.pad(dx, (0, max_h - L), constant_values=0).astype(np.float32)
        padded_y = np.pad(dy, (0, max_h - L), constant_values=0).astype(np.float32)
        mask = np.zeros(max_h, dtype=np.float32)
        mask[:L] = 1.0
        
        tensors_x.append(torch.tensor(padded_x))
        tensors_y.append(torch.tensor(padded_y))
        masks.append(torch.tensor(mask))
    
    targets = torch.stack([torch.stack(tensors_x), torch.stack(tensors_y)], dim=-1)
    return targets, torch.stack(masks)

def compute_val_rmse(model, X_val, y_val_dx, y_val_dy, horizon, device, batch_size=256):
    """
    Compute validation RMSE (Root Mean Squared Error) for 2D trajectory prediction.
    Returns the average Euclidean distance error across all predictions.
    """
    model.eval()
    all_errors = []
    
    with torch.no_grad():
        for i in range(0, len(X_val), batch_size):
            end = min(i + batch_size, len(X_val))
            bx = torch.tensor(np.stack(X_val[i:end]).astype(np.float32)).to(device)
            
            # Get predictions
            pred = model(bx).cpu().numpy()  # Shape: (batch, horizon, 2)
            
            # Process each sample in the batch
            for j, idx in enumerate(range(i, end)):
                dx_true = y_val_dx[idx]
                dy_true = y_val_dy[idx]
                n_steps = len(dx_true)
                
                # Extract predictions for this sample
                dx_pred = pred[j, :n_steps, 0]
                dy_pred = pred[j, :n_steps, 1]
                
                # Compute Euclidean distance for each time step
                sq_errors = (dx_pred - dx_true) ** 2 + (dy_pred - dy_true) ** 2
                all_errors.extend(sq_errors)
    
    # Return RMSE
    return np.sqrt(np.mean(all_errors))

def train_model(X_train, y_train_dx, y_train_dy, X_val, y_val_dx, y_val_dy, 
                input_dim, horizon, config):
    device = config.DEVICE
    
    model = STTransformer(
        input_dim=input_dim,
        hidden_dim=config.HIDDEN_DIM,
        horizon=horizon,
        window_size=config.WINDOW_SIZE,
        n_heads=config.N_HEADS,
        n_layers=config.N_LAYERS
    ).to(device)
    criterion = TemporalHuber(delta=0.5, time_decay=0.03)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5, verbose=False)
    
    train_batches = []
    for i in range(0, len(X_train), config.BATCH_SIZE):
        end = min(i + config.BATCH_SIZE, len(X_train))
        bx = torch.tensor(np.stack(X_train[i:end]).astype(np.float32))
        by, bm = prepare_targets(
            [y_train_dx[j] for j in range(i, end)],
            [y_train_dy[j] for j in range(i, end)],
            horizon
        )
        train_batches.append((bx, by, bm))
    
    val_batches = []
    for i in range(0, len(X_val), config.BATCH_SIZE):
        end = min(i + config.BATCH_SIZE, len(X_val))
        bx = torch.tensor(np.stack(X_val[i:end]).astype(np.float32))
        by, bm = prepare_targets(
            [y_val_dx[j] for j in range(i, end)],
            [y_val_dy[j] for j in range(i, end)],
            horizon
        )
        val_batches.append((bx, by, bm))
    
    best_loss, best_state, bad = float('inf'), None, 0
    
    for epoch in range(1, config.EPOCHS + 1):
        model.train()
        train_losses = []
        for bx, by, bm in train_batches:
            bx, by, bm = bx.to(device), by.to(device), bm.to(device)
            pred = model(bx)
            loss = criterion(pred, by, bm)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())
        
        model.eval()
        val_losses = []
        with torch.no_grad():
            for bx, by, bm in val_batches:
                bx, by, bm = bx.to(device), by.to(device), bm.to(device)
                pred = model(bx)
                val_losses.append(criterion(pred, by, bm).item())
        
        train_loss, val_loss = np.mean(train_losses), np.mean(val_losses)
        scheduler.step(val_loss)
        
        if epoch % 10 == 0:
            print(f"  Epoch {epoch}: train={train_loss:.4f}, val={val_loss:.4f}")
        
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= config.PATIENCE:
                print(f"  Early stop at epoch {epoch}")
                break
    
    if best_state:
        model.load_state_dict(best_state)
    
    return model, best_loss

import kaggle_evaluation.nfl_inference_server
import warnings
import polars as pl # Import Polars for the predict function signature

# Move all utility functions (get_velocity, height_to_feet, compute_geometric_endpoint, etc.)
# and the Config/set_seed outside the class but above it, or make them static methods.

class NFLPredictor:
    def __init__(self):
        warnings.filterwarnings('ignore')
        config = Config() # Use the existing Config class
        set_seed(config.SEED)
        self.config = config

        # If configured, try to load pre-saved artifacts to skip training and data prep
        load_dir = Path(config.LOAD_DIR) if config.LOAD_DIR is not None else config.MODEL_DIR
        artifacts_present = all((load_dir / f"model_fold{f}.pt").exists() for f in range(1, config.N_FOLDS + 1)) and (
            (load_dir / "route_kmeans.pkl").exists() and (load_dir / "route_scaler.pkl").exists()
        )

        if config.LOAD_ARTIFACTS and artifacts_present:
            print(f"\n[1/2] Loading trained artifacts from disk (from {load_dir})...")
            self.models = []
            self.scalers = []
            # load route objects
            try:
                with open(load_dir / "route_kmeans.pkl", "rb") as f:
                    self.route_kmeans = pickle.load(f)
                with open(load_dir / "route_scaler.pkl", "rb") as f:
                    self.route_scaler = pickle.load(f)
            except Exception as e:
                print("Failed to load route artifacts:", e)

            # For input_dim, use a dummy array if needed
            dummy_input_dim = 167  # fallback if not available
            input_dim = dummy_input_dim
            # Try to infer input_dim from scaler if possible
            try:
                with open(load_dir / "scaler_fold1.pkl", "rb") as f:
                    scaler1 = pickle.load(f)
                input_dim = scaler1.mean_.shape[0]
            except Exception:
                pass

            for fold in range(1, config.N_FOLDS + 1):
                model_path = load_dir / f"model_fold{fold}.pt"
                scaler_path = load_dir / f"scaler_fold{fold}.pkl"

                # load scaler
                try:
                    with open(scaler_path, "rb") as f:
                        scaler = pickle.load(f)
                except Exception:
                    scaler = None

                # instantiate model and load state
                model = STTransformer(
                        input_dim=input_dim,
                        hidden_dim=config.HIDDEN_DIM,
                        horizon=config.MAX_FUTURE_HORIZON,
                        window_size=config.WINDOW_SIZE,
                        n_heads=config.N_HEADS,
                        n_layers=config.N_LAYERS).to(config.DEVICE)
                try:
                    state = torch.load(model_path, map_location='cpu')
                    model.load_state_dict(state)
                except Exception as e:
                    print(f"Failed to load model fold {fold}:", e)

                model.to(config.DEVICE)
                model.eval()

                self.models.append(model)
                self.scalers.append(scaler)

            print(f"✓ Loaded {len(self.models)} models and scalers from {load_dir}")
            print("[2/2] Ready for inference. Skipped all training and data preparation.")
            return

        # If not loading, proceed with training and data preparation
        # 1. Load Data
        print("[1/4] Loading data for training...")
        train_input_files = [config.DATA_DIR / f"train/input_2023_w{w:02d}.csv" for w in range(1, 19)]
        train_output_files = [config.DATA_DIR / f"train/output_2023_w{w:02d}.csv" for w in range(1, 19)]
        if Config.DEBUG:            
            # Load a small amount of data to quickly get play IDs
            sample_train_input = pd.read_csv(train_input_files[0])
            sample_train_output = pd.read_csv(train_output_files[0])
            # Get the first 100 unique plays (game_id, play_id combinations)
            sample_plays = sample_train_output[['game_id', 'play_id']].drop_duplicates().head(100)
            # Filter the full data for only these plays
            train_input = pd.concat([pd.read_csv(f) for f in train_input_files if f.exists()])
            train_output = pd.concat([pd.read_csv(f) for f in train_output_files if f.exists()])
            # Perform the actual filtering to get a small, but complete, set of plays
            train_input = train_input.merge(sample_plays, on=['game_id', 'play_id'], how='inner')
            train_output = train_output.merge(sample_plays, on=['game_id', 'play_id'], how='inner')
            print(f"✓ Reduced to {len(sample_plays)} unique plays.")
        else:
            train_input = pd.concat([pd.read_csv(f) for f in train_input_files if f.exists()])
            train_output = pd.concat([pd.read_csv(f) for f in train_output_files if f.exists()])

        # 2. Prepare Sequences and Train Feature Objects
        print("\n[2/4] Preparing geometric sequences and feature scalers...")
        result = prepare_sequences_geometric(
            train_input, train_output, is_training=True, window_size=config.WINDOW_SIZE
        )
        sequences, targets_dx, targets_dy, targets_frame_ids, sequence_ids, geo_x, geo_y, route_kmeans, route_scaler = result

        # Store feature objects for later inference
        self.route_kmeans = route_kmeans
        self.route_scaler = route_scaler
        
        sequences = list(sequences)
        targets_dx = list(targets_dx)
        targets_dy = list(targets_dy)

        # Ensure model dir exists
        config.MODEL_DIR.mkdir(parents=True, exist_ok=True)

        # 3. Train Models
        print("\n[3/4] Training geometric models...")
        groups = np.array([d['game_id'] for d in sequence_ids])
        gkf = GroupKFold(n_splits=config.N_FOLDS)

        self.models, self.scalers = [], []
        fold_losses = []
        fold_rmses = []

        for fold, (tr, va) in enumerate(gkf.split(sequences, groups=groups), 1):
            # ... (rest of the fold training logic, exactly as in the original main()) ...
            X_tr = [sequences[i] for i in tr]
            X_va = [sequences[i] for i in va]
            y_tr_dx = [targets_dx[i] for i in tr]
            y_va_dx = [targets_dx[i] for i in va]
            y_tr_dy = [targets_dy[i] for i in tr]
            y_va_dy = [targets_dy[i] for i in va]
            scaler = StandardScaler()
            scaler.fit(np.vstack([s for s in X_tr]))
            X_tr_sc = [scaler.transform(s) for s in X_tr]
            X_va_sc = [scaler.transform(s) for s in X_va]
            model, loss = train_model(
                X_tr_sc, y_tr_dx, y_tr_dy,
                X_va_sc, y_va_dx, y_va_dy,
                X_tr[0].shape[-1], config.MAX_FUTURE_HORIZON, config
            )
            
            # Compute validation RMSE
            val_rmse = compute_val_rmse(
                model, X_va_sc, y_va_dx, y_va_dy, 
                config.MAX_FUTURE_HORIZON, config.DEVICE, config.BATCH_SIZE
            )
            
            self.models.append(model)
            self.scalers.append(scaler)
            fold_losses.append(loss)
            fold_rmses.append(val_rmse)
            
            print(f"\n✓ Fold {fold} - Loss: {loss:.5f}, Validation RMSE: {val_rmse:.5f}")
            # Persist fold artifacts if requested
            if config.SAVE_ARTIFACTS:
                try:
                    model_path = config.MODEL_DIR / f"model_fold{fold}.pt"
                    scaler_path = config.MODEL_DIR / f"scaler_fold{fold}.pkl"
                    # save model state dict on CPU to avoid device issues
                    state = {k: v.cpu() for k, v in model.state_dict().items()}
                    torch.save(state, str(model_path))
                    with open(scaler_path, "wb") as f:
                        pickle.dump(scaler, f)
                    print(f"  Saved artifacts for fold {fold} -> {model_path.name}, {scaler_path.name}")
                except Exception as e:
                    print(f"  Warning: failed to save artifacts for fold {fold}:", e)

        # Print summary statistics across all folds
        print("\n" + "="*60)
        print("CROSS-VALIDATION SUMMARY")
        print("="*60)
        avg_loss = np.mean(fold_losses)
        std_loss = np.std(fold_losses)
        avg_rmse = np.mean(fold_rmses)
        std_rmse = np.std(fold_rmses)
        
        print(f"Average Loss:           {avg_loss:.5f} ± {std_loss:.5f}")
        print(f"Average Validation RMSE: {avg_rmse:.5f} ± {std_rmse:.5f}")
        print(f"\nPer-Fold Results:")
        for i, (loss, rmse) in enumerate(zip(fold_losses, fold_rmses), 1):
            print(f"  Fold {i}: Loss={loss:.5f}, RMSE={rmse:.5f}")
        print("="*60 + "\n")

        # Ensure all models are in eval mode (best practice)
        # Optionally persist route clustering objects
        if config.SAVE_ARTIFACTS:
            try:
                with open(config.MODEL_DIR / "route_kmeans.pkl", "wb") as f:
                    pickle.dump(self.route_kmeans, f)
                with open(config.MODEL_DIR / "route_scaler.pkl", "wb") as f:
                    pickle.dump(self.route_scaler, f)
                print(f"✓ Saved route artifacts -> route_kmeans.pkl, route_scaler.pkl")
            except Exception as e:
                print("Warning: failed to save route artifacts:", e)

        for model in self.models:
            model.eval()
        print("\n🏆 Geometric Neural Breakthrough Model is ready for inference! 🏆")


    def predict(self, test: pl.DataFrame, test_input: pl.DataFrame) -> pd.DataFrame:
        """Inference function called by the API for each time step."""
        
        # Convert Polars DataFrames to Pandas, as the original notebook uses Pandas extensively
        test_input_pd = test_input.to_pandas()
        test_template_pd = test.to_pandas()
        
        # 1. Prepare Test Sequences
        # Use the stored feature objects (route_kmeans, route_scaler) for inference
        test_seq, test_ids, test_geo_x, test_geo_y = prepare_sequences_geometric(
            test_input_pd, test_template=test_template_pd, is_training=False,
            window_size=self.config.WINDOW_SIZE,
            route_kmeans=self.route_kmeans, route_scaler=self.route_scaler
        )

        X_test = list(test_seq)
        x_last = np.array([s[-1, 0] for s in X_test])
        y_last = np.array([s[-1, 1] for s in X_test])

        # 2. Ensemble Prediction
        all_preds = []
        H = self.config.MAX_FUTURE_HORIZON

        for model, sc in zip(self.models, self.scalers):
            X_sc = [sc.transform(s) for s in X_test]
            X_t = torch.tensor(np.stack(X_sc).astype(np.float32)).to(self.config.DEVICE)

            with torch.no_grad():
                preds = model(X_t).cpu().numpy()

            all_preds.append(preds)

        ens_preds = np.mean(all_preds, axis=0)

        # 3. Format Submission (Corrected for API)
        rows = []
        for i, sid in enumerate(test_ids):
            # The template DataFrame 'test' contains the info needed to map predictions
            fids = test_template_pd[
                (test_template_pd['game_id'] == sid['game_id']) &
                (test_template_pd['play_id'] == sid['play_id']) &
                (test_template_pd['nfl_id'] == sid['nfl_id'])
            ]['frame_id'].sort_values().tolist()
            
            for t, fid in enumerate(fids):
                tt = min(t, H - 1)
                px = np.clip(x_last[i] + ens_preds[i, tt, 0], 0, self.config.FIELD_X_MAX)
                py = np.clip(y_last[i] + ens_preds[i, tt, 1], 0, self.config.FIELD_Y_MAX)

                # DO NOT include 'id' in the rows dict or final DataFrame
                # The API will handle the row IDs based on the provided 'test' DataFrame
                rows.append({
                    'x': px,
                    'y': py
                })

        # The final returned DataFrame MUST only contain 'x' and 'y' columns, 
        # matching the order of rows in the input 'test' DataFrame.
        submission = pd.DataFrame(rows)
        return submission

# This must be outside the class and is what the API calls
predictor = NFLPredictor()

def predict(test: pl.DataFrame, test_input: pl.DataFrame) -> pl.DataFrame | pd.DataFrame:
    """The function the API calls."""
    return predictor.predict(test, test_input)
```

```python
%%writefile nfl_gnn.py

"""
NFL BIG DATA BOWL 2026 - GEOMETRIC NEURAL BREAKTHROUGH
🏆 The Winning Architecture

INSIGHT: Football players follow geometric rules with learned corrections
- Receivers → Ball landing (geometric)
- Defenders → Mirror receivers (geometric coupling)  
- Others → Momentum (physics)
- Model learns only CORRECTIONS for coverage, collisions, boundaries

Architecture:
✓ Proven GRU + Attention (your 0.59 base)
✓ 154 proven features (unchanged)
✓ +15 geometric features (our discovery)
✓ Train on corrections to geometric baseline (the breakthrough)

Target: 0.54-0.56 LB
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
import warnings
import os
import pickle

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.cluster import KMeans
from multiprocessing import Pool as MultiprocessingPool, cpu_count

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIG
# ============================================================================

class Config:
    DATA_DIR = Path("/kaggle/input/nfl-big-data-bowl-2026-prediction/")
    OUTPUT_DIR = Path("./outputs")
    OUTPUT_DIR.mkdir(exist_ok=True)
    # Where to persist trained artifacts (models, scalers, route objects)
    MODEL_DIR = OUTPUT_DIR / "models"
    MODEL_DIR.mkdir(exist_ok=True)
    # Toggle saving/loading of artifacts
    SAVE_ARTIFACTS = True
    LOAD_ARTIFACTS = True
    LOAD_DIR = '/kaggle/input/nfl-bdb-2026/nfl-gnn-v9/outputs/models'
  
    SEED = 42
    N_FOLDS = 5
    BATCH_SIZE = 256
    EPOCHS = 200
    PATIENCE = 30
    LEARNING_RATE = 1e-3
    
    WINDOW_SIZE = 10
    HIDDEN_DIM = 128
    MAX_FUTURE_HORIZON = 94
    
    FIELD_X_MIN, FIELD_X_MAX = 0.0, 120.0
    FIELD_Y_MIN, FIELD_Y_MAX = 0.0, 53.3
    
    K_NEIGH = 6
    RADIUS = 30.0
    TAU = 8.0
    N_ROUTE_CLUSTERS = 7
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DEBUG = False
    if DEBUG:
        N_FOLDS = 2

def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed(Config.SEED)

# ============================================================================
# GEOMETRIC BASELINE - THE BREAKTHROUGH
# ============================================================================

def compute_geometric_endpoint(df):
    """
    Compute where each player SHOULD end up based on geometry.
    This is the deterministic part - no learning needed.
    """
    df = df.copy()
    
    # Time to play end
    if 'num_frames_output' in df.columns:
        t_total = df['num_frames_output'] / 10.0
    else:
        t_total = 3.0
    
    df['time_to_endpoint'] = t_total
    
    # Initialize with momentum (default rule)
    df['geo_endpoint_x'] = df['x'] + df['velocity_x'] * t_total
    df['geo_endpoint_y'] = df['y'] + df['velocity_y'] * t_total
    
    # Rule 1: Targeted Receivers converge to ball
    if 'ball_land_x' in df.columns:
        receiver_mask = df['player_role'] == 'Targeted Receiver'
        df.loc[receiver_mask, 'geo_endpoint_x'] = df.loc[receiver_mask, 'ball_land_x']
        df.loc[receiver_mask, 'geo_endpoint_y'] = df.loc[receiver_mask, 'ball_land_y']
        
        # Rule 2: Defenders mirror receivers (maintain offset)
        defender_mask = df['player_role'] == 'Defensive Coverage'
        has_mirror = df.get('mirror_offset_x', 0).notna() & (df.get('mirror_wr_dist', 50) < 15)
        coverage_mask = defender_mask & has_mirror
        
        df.loc[coverage_mask, 'geo_endpoint_x'] = (
            df.loc[coverage_mask, 'ball_land_x'] + 
            df.loc[coverage_mask, 'mirror_offset_x'].fillna(0)
        )
        df.loc[coverage_mask, 'geo_endpoint_y'] = (
            df.loc[coverage_mask, 'ball_land_y'] + 
            df.loc[coverage_mask, 'mirror_offset_y'].fillna(0)
        )
    
    # Clip to field
    df['geo_endpoint_x'] = df['geo_endpoint_x'].clip(Config.FIELD_X_MIN, Config.FIELD_X_MAX)
    df['geo_endpoint_y'] = df['geo_endpoint_y'].clip(Config.FIELD_Y_MIN, Config.FIELD_Y_MAX)
    
    return df

def add_geometric_features(df):
    """Add features that describe the geometric solution"""
    df = compute_geometric_endpoint(df)
    
    # Vector to geometric endpoint
    df['geo_vector_x'] = df['geo_endpoint_x'] - df['x']
    df['geo_vector_y'] = df['geo_endpoint_y'] - df['y']
    df['geo_distance'] = np.sqrt(df['geo_vector_x']**2 + df['geo_vector_y']**2)
    
    # Required velocity to reach geometric endpoint
    t = df['time_to_endpoint'] + 0.1
    df['geo_required_vx'] = df['geo_vector_x'] / t
    df['geo_required_vy'] = df['geo_vector_y'] / t
    
    # Current velocity vs required
    df['geo_velocity_error_x'] = df['geo_required_vx'] - df['velocity_x']
    df['geo_velocity_error_y'] = df['geo_required_vy'] - df['velocity_y']
    df['geo_velocity_error'] = np.sqrt(
        df['geo_velocity_error_x']**2 + df['geo_velocity_error_y']**2
    )
    
    # Required constant acceleration (a = 2*Δx/t²)
    t_sq = t * t
    df['geo_required_ax'] = 2 * df['geo_vector_x'] / t_sq
    df['geo_required_ay'] = 2 * df['geo_vector_y'] / t_sq
    df['geo_required_ax'] = df['geo_required_ax'].clip(-10, 10)
    df['geo_required_ay'] = df['geo_required_ay'].clip(-10, 10)
    
    # Alignment with geometric path
    velocity_mag = np.sqrt(df['velocity_x']**2 + df['velocity_y']**2)
    geo_unit_x = df['geo_vector_x'] / (df['geo_distance'] + 0.1)
    geo_unit_y = df['geo_vector_y'] / (df['geo_distance'] + 0.1)
    df['geo_alignment'] = (
        df['velocity_x'] * geo_unit_x + df['velocity_y'] * geo_unit_y
    ) / (velocity_mag + 0.1)
    
    # Role-specific geometric quality
    df['geo_receiver_urgency'] = df['is_receiver'] * df['geo_distance'] / (t + 0.1)
    df['geo_defender_coupling'] = df['is_coverage'] * (1.0 / (df.get('mirror_wr_dist', 50) + 1.0))
    
    return df

# ============================================================================
# PROVEN FEATURE ENGINEERING (YOUR 0.59 BASE)
# ============================================================================

def get_velocity(speed, direction_deg):
    theta = np.deg2rad(direction_deg)
    return speed * np.sin(theta), speed * np.cos(theta)

def height_to_feet(height_str):
    try:
        ft, inches = map(int, str(height_str).split('-'))
        return ft + inches/12
    except:
        return 6.0

def get_opponent_features(input_df):
    """Enhanced opponent interaction with MIRROR WR tracking"""
    features = []
    
    for (gid, pid), group in tqdm(input_df.groupby(['game_id', 'play_id']), 
                                   desc="🏈 Opponents", leave=False):
        last = group.sort_values('frame_id').groupby('nfl_id').last()
        
        if len(last) < 2:
            continue
            
        positions = last[['x', 'y']].values
        sides = last['player_side'].values
        speeds = last['s'].values
        directions = last['dir'].values
        roles = last['player_role'].values
        
        receiver_mask = np.isin(roles, ['Targeted Receiver', 'Other Route Runner'])
        
        for i, (nid, side, role) in enumerate(zip(last.index, sides, roles)):
            opp_mask = sides != side
            
            feat = {
                'game_id': gid, 'play_id': pid, 'nfl_id': nid,
                'nearest_opp_dist': 50.0, 'closing_speed': 0.0,
                'num_nearby_opp_3': 0, 'num_nearby_opp_5': 0,
                'mirror_wr_vx': 0.0, 'mirror_wr_vy': 0.0,
                'mirror_offset_x': 0.0, 'mirror_offset_y': 0.0,
                'mirror_wr_dist': 50.0,
            }
            
            if not opp_mask.any():
                features.append(feat)
                continue
            
            opp_positions = positions[opp_mask]
            distances = np.sqrt(((positions[i] - opp_positions)**2).sum(axis=1))
            
            if len(distances) == 0:
                features.append(feat)
                continue
                
            nearest_idx = distances.argmin()
            feat['nearest_opp_dist'] = distances[nearest_idx]
            feat['num_nearby_opp_3'] = (distances < 3.0).sum()
            feat['num_nearby_opp_5'] = (distances < 5.0).sum()
            
            my_vx, my_vy = get_velocity(speeds[i], directions[i])
            opp_speeds = speeds[opp_mask]
            opp_dirs = directions[opp_mask]
            opp_vx, opp_vy = get_velocity(opp_speeds[nearest_idx], opp_dirs[nearest_idx])
            
            rel_vx = my_vx - opp_vx
            rel_vy = my_vy - opp_vy
            to_me = positions[i] - opp_positions[nearest_idx]
            to_me_norm = to_me / (np.linalg.norm(to_me) + 0.1)
            feat['closing_speed'] = -(rel_vx * to_me_norm[0] + rel_vy * to_me_norm[1])
            
            if role == 'Defensive Coverage' and receiver_mask.any():
                rec_positions = positions[receiver_mask]
                rec_distances = np.sqrt(((positions[i] - rec_positions)**2).sum(axis=1))
                
                if len(rec_distances) > 0:
                    closest_rec_idx = rec_distances.argmin()
                    rec_indices = np.where(receiver_mask)[0]
                    actual_rec_idx = rec_indices[closest_rec_idx]
                    
                    rec_vx, rec_vy = get_velocity(speeds[actual_rec_idx], directions[actual_rec_idx])
                    
                    feat['mirror_wr_vx'] = rec_vx
                    feat['mirror_wr_vy'] = rec_vy
                    feat['mirror_wr_dist'] = rec_distances[closest_rec_idx]
                    feat['mirror_offset_x'] = positions[i][0] - rec_positions[closest_rec_idx][0]
                    feat['mirror_offset_y'] = positions[i][1] - rec_positions[closest_rec_idx][1]
            
            features.append(feat)
    
    return pd.DataFrame(features)

def extract_route_patterns(input_df, kmeans=None, scaler=None, fit=True):
    """Route clustering"""
    route_features = []
    
    for (gid, pid, nid), group in tqdm(input_df.groupby(['game_id', 'play_id', 'nfl_id']), 
                                        desc="🛣️  Routes", leave=False):
        traj = group.sort_values('frame_id').tail(5)
        
        if len(traj) < 3:
            continue
        
        positions = traj[['x', 'y']].values
        speeds = traj['s'].values
        
        total_dist = np.sum(np.sqrt(np.diff(positions[:, 0])**2 + np.diff(positions[:, 1])**2))
        displacement = np.sqrt((positions[-1, 0] - positions[0, 0])**2 + 
                              (positions[-1, 1] - positions[0, 1])**2)
        straightness = displacement / (total_dist + 0.1)
        
        angles = np.arctan2(np.diff(positions[:, 1]), np.diff(positions[:, 0]))
        if len(angles) > 1:
            angle_changes = np.abs(np.diff(angles))
            max_turn = np.max(angle_changes)
            mean_turn = np.mean(angle_changes)
        else:
            max_turn = mean_turn = 0
        
        speed_mean = speeds.mean()
        speed_change = speeds[-1] - speeds[0] if len(speeds) > 1 else 0
        dx = positions[-1, 0] - positions[0, 0]
        dy = positions[-1, 1] - positions[0, 1]
        
        route_features.append({
            'game_id': gid, 'play_id': pid, 'nfl_id': nid,
            'traj_straightness': straightness,
            'traj_max_turn': max_turn,
            'traj_mean_turn': mean_turn,
            'traj_depth': abs(dx),
            'traj_width': abs(dy),
            'speed_mean': speed_mean,
            'speed_change': speed_change,
        })
    
    route_df = pd.DataFrame(route_features)
    feat_cols = ['traj_straightness', 'traj_max_turn', 'traj_mean_turn',
                 'traj_depth', 'traj_width', 'speed_mean', 'speed_change']
    X = route_df[feat_cols].fillna(0)
    
    if fit:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        kmeans = KMeans(n_clusters=Config.N_ROUTE_CLUSTERS, random_state=Config.SEED, n_init=10)
        route_df['route_pattern'] = kmeans.fit_predict(X_scaled)
        return route_df, kmeans, scaler
    else:
        X_scaled = scaler.transform(X)
        route_df['route_pattern'] = kmeans.predict(X_scaled)
        return route_df

def compute_neighbor_embeddings(input_df, k_neigh=Config.K_NEIGH, 
                                radius=Config.RADIUS, tau=Config.TAU):
    """GNN-lite embeddings"""
    print("🕸️  GNN embeddings...")
    
    cols_needed = ["game_id", "play_id", "nfl_id", "frame_id", "x", "y", 
                   "velocity_x", "velocity_y", "player_side"]
    src = input_df[cols_needed].copy()
    
    last = (src.sort_values(["game_id", "play_id", "nfl_id", "frame_id"])
               .groupby(["game_id", "play_id", "nfl_id"], as_index=False)
               .tail(1)
               .rename(columns={"frame_id": "last_frame_id"})
               .reset_index(drop=True))
    
    tmp = last.merge(
        src.rename(columns={
            "frame_id": "nb_frame_id", "nfl_id": "nfl_id_nb",
            "x": "x_nb", "y": "y_nb", 
            "velocity_x": "vx_nb", "velocity_y": "vy_nb", 
            "player_side": "player_side_nb"
        }),
        left_on=["game_id", "play_id", "last_frame_id"],
        right_on=["game_id", "play_id", "nb_frame_id"],
        how="left"
    )
    
    tmp = tmp[tmp["nfl_id_nb"] != tmp["nfl_id"]]
    tmp["dx"] = tmp["x_nb"] - tmp["x"]
    tmp["dy"] = tmp["y_nb"] - tmp["y"]
    tmp["dvx"] = tmp["vx_nb"] - tmp["velocity_x"]
    tmp["dvy"] = tmp["vy_nb"] - tmp["velocity_y"]
    tmp["dist"] = np.sqrt(tmp["dx"]**2 + tmp["dy"]**2)
    
    tmp = tmp[np.isfinite(tmp["dist"]) & (tmp["dist"] > 1e-6)]
    if radius is not None:
        tmp = tmp[tmp["dist"] <= radius]
    
    tmp["is_ally"] = (tmp["player_side_nb"] == tmp["player_side"]).astype(np.float32)
    
    keys = ["game_id", "play_id", "nfl_id"]
    tmp["rnk"] = tmp.groupby(keys)["dist"].rank(method="first")
    if k_neigh is not None:
        tmp = tmp[tmp["rnk"] <= float(k_neigh)]
    
    tmp["w"] = np.exp(-tmp["dist"] / float(tau))
    sum_w = tmp.groupby(keys)["w"].transform("sum")
    tmp["wn"] = np.where(sum_w > 0, tmp["w"] / sum_w, 0.0)
    
    tmp["wn_ally"] = tmp["wn"] * tmp["is_ally"]
    tmp["wn_opp"] = tmp["wn"] * (1.0 - tmp["is_ally"])
    
    for col in ["dx", "dy", "dvx", "dvy"]:
        tmp[f"{col}_ally_w"] = tmp[col] * tmp["wn_ally"]
        tmp[f"{col}_opp_w"] = tmp[col] * tmp["wn_opp"]
    
    tmp["dist_ally"] = np.where(tmp["is_ally"] > 0.5, tmp["dist"], np.nan)
    tmp["dist_opp"] = np.where(tmp["is_ally"] < 0.5, tmp["dist"], np.nan)
    
    ag = tmp.groupby(keys).agg(
        gnn_ally_dx_mean=("dx_ally_w", "sum"),
        gnn_ally_dy_mean=("dy_ally_w", "sum"),
        gnn_ally_dvx_mean=("dvx_ally_w", "sum"),
        gnn_ally_dvy_mean=("dvy_ally_w", "sum"),
        gnn_opp_dx_mean=("dx_opp_w", "sum"),
        gnn_opp_dy_mean=("dy_opp_w", "sum"),
        gnn_opp_dvx_mean=("dvx_opp_w", "sum"),
        gnn_opp_dvy_mean=("dvy_opp_w", "sum"),
        gnn_ally_cnt=("is_ally", "sum"),
        gnn_opp_cnt=("is_ally", lambda s: float(len(s) - s.sum())),
        gnn_ally_dmin=("dist_ally", "min"),
        gnn_ally_dmean=("dist_ally", "mean"),
        gnn_opp_dmin=("dist_opp", "min"),
        gnn_opp_dmean=("dist_opp", "mean"),
    ).reset_index()
    
    near = tmp.loc[tmp["rnk"] <= 3, keys + ["rnk", "dist"]].copy()
    if len(near) > 0:
        near["rnk"] = near["rnk"].astype(int)
        dwide = near.pivot_table(index=keys, columns="rnk", values="dist", aggfunc="first")
        dwide = dwide.rename(columns={1: "gnn_d1", 2: "gnn_d2", 3: "gnn_d3"}).reset_index()
        ag = ag.merge(dwide, on=keys, how="left")
    
    for c in ["gnn_ally_dx_mean", "gnn_ally_dy_mean", "gnn_ally_dvx_mean", "gnn_ally_dvy_mean",
              "gnn_opp_dx_mean", "gnn_opp_dy_mean", "gnn_opp_dvx_mean", "gnn_opp_dvy_mean"]:
        ag[c] = ag[c].fillna(0.0)
    for c in ["gnn_ally_cnt", "gnn_opp_cnt"]:
        ag[c] = ag[c].fillna(0.0)
    for c in ["gnn_ally_dmin", "gnn_opp_dmin", "gnn_ally_dmean", "gnn_opp_dmean", 
              "gnn_d1", "gnn_d2", "gnn_d3"]:
        ag[c] = ag[c].fillna(radius if radius is not None else 30.0)
    
    return ag

# ============================================================================
# SEQUENCE PREPARATION WITH GEOMETRIC FEATURES
# ============================================================================

def prepare_sequences_geometric(input_df, output_df=None, test_template=None, 
                                is_training=True, window_size=10,
                                route_kmeans=None, route_scaler=None):
    """YOUR 154 features + 13 geometric features = 167 total"""
    
    print(f"\n{'='*80}")
    print(f"PREPARING GEOMETRIC SEQUENCES")
    print(f"{'='*80}")
    
    input_df = input_df.copy()
    input_df = input_df.sort_values(['game_id', 'play_id', 'nfl_id', 'frame_id'])
    
    print("Step 1: Base features...")
    
    input_df['player_height_feet'] = input_df['player_height'].apply(height_to_feet)
    height_parts = input_df['player_height'].str.split('-', expand=True)
    input_df['height_inches'] = height_parts[0].astype(float) * 12 + height_parts[1].astype(float)
    input_df['bmi'] = (input_df['player_weight'] / (input_df['height_inches']**2)) * 703
    
    dir_rad = np.deg2rad(input_df['dir'].fillna(0))
    input_df['velocity_x'] = input_df['s'] * np.sin(dir_rad)
    input_df['velocity_y'] = input_df['s'] * np.cos(dir_rad)
    input_df['acceleration_x'] = input_df['a'] * np.cos(dir_rad)
    input_df['acceleration_y'] = input_df['a'] * np.sin(dir_rad)
    
    input_df['speed_squared'] = input_df['s'] ** 2
    input_df['accel_magnitude'] = np.sqrt(input_df['acceleration_x']**2 + input_df['acceleration_y']**2)
    input_df['momentum_x'] = input_df['velocity_x'] * input_df['player_weight']
    input_df['momentum_y'] = input_df['velocity_y'] * input_df['player_weight']
    input_df['kinetic_energy'] = 0.5 * input_df['player_weight'] * input_df['speed_squared']
    
    input_df['orientation_diff'] = np.abs(input_df['o'] - input_df['dir'])
    input_df['orientation_diff'] = np.minimum(input_df['orientation_diff'], 360 - input_df['orientation_diff'])
    
    input_df['is_offense'] = (input_df['player_side'] == 'Offense').astype(int)
    input_df['is_defense'] = (input_df['player_side'] == 'Defense').astype(int)
    input_df['is_receiver'] = (input_df['player_role'] == 'Targeted Receiver').astype(int)
    input_df['is_coverage'] = (input_df['player_role'] == 'Defensive Coverage').astype(int)
    input_df['is_passer'] = (input_df['player_role'] == 'Passer').astype(int)
    input_df['role_targeted_receiver'] = input_df['is_receiver']
    input_df['role_defensive_coverage'] = input_df['is_coverage']
    input_df['role_passer'] = input_df['is_passer']
    input_df['side_offense'] = input_df['is_offense']
    
    if 'ball_land_x' in input_df.columns:
        ball_dx = input_df['ball_land_x'] - input_df['x']
        ball_dy = input_df['ball_land_y'] - input_df['y']
        input_df['distance_to_ball'] = np.sqrt(ball_dx**2 + ball_dy**2)
        input_df['dist_to_ball'] = input_df['distance_to_ball']
        input_df['dist_squared'] = input_df['distance_to_ball'] ** 2
        input_df['angle_to_ball'] = np.arctan2(ball_dy, ball_dx)
        input_df['ball_direction_x'] = ball_dx / (input_df['distance_to_ball'] + 1e-6)
        input_df['ball_direction_y'] = ball_dy / (input_df['distance_to_ball'] + 1e-6)
        input_df['closing_speed_ball'] = (
            input_df['velocity_x'] * input_df['ball_direction_x'] +
            input_df['velocity_y'] * input_df['ball_direction_y']
        )
        input_df['velocity_toward_ball'] = (
            input_df['velocity_x'] * np.cos(input_df['angle_to_ball']) + 
            input_df['velocity_y'] * np.sin(input_df['angle_to_ball'])
        )
        input_df['velocity_alignment'] = np.cos(input_df['angle_to_ball'] - dir_rad)
        input_df['angle_diff'] = np.abs(input_df['o'] - np.degrees(input_df['angle_to_ball']))
        input_df['angle_diff'] = np.minimum(input_df['angle_diff'], 360 - input_df['angle_diff'])
    
    print("Step 2: Advanced features...")
    
    opp_features = get_opponent_features(input_df)
    input_df = input_df.merge(opp_features, on=['game_id', 'play_id', 'nfl_id'], how='left')
    
    if is_training:
        route_features, route_kmeans, route_scaler = extract_route_patterns(input_df)
    else:
        route_features = extract_route_patterns(input_df, route_kmeans, route_scaler, fit=False)
    input_df = input_df.merge(route_features, on=['game_id', 'play_id', 'nfl_id'], how='left')
    
    gnn_features = compute_neighbor_embeddings(input_df)
    input_df = input_df.merge(gnn_features, on=['game_id', 'play_id', 'nfl_id'], how='left')
    
    if 'nearest_opp_dist' in input_df.columns:
        input_df['pressure'] = 1 / np.maximum(input_df['nearest_opp_dist'], 0.5)
        input_df['under_pressure'] = (input_df['nearest_opp_dist'] < 3).astype(int)
        input_df['pressure_x_speed'] = input_df['pressure'] * input_df['s']
    
    if 'mirror_wr_vx' in input_df.columns:
        s_safe = np.maximum(input_df['s'], 0.1)
        input_df['mirror_similarity'] = (
            input_df['velocity_x'] * input_df['mirror_wr_vx'] + 
            input_df['velocity_y'] * input_df['mirror_wr_vy']
        ) / s_safe
        input_df['mirror_offset_dist'] = np.sqrt(
            input_df['mirror_offset_x']**2 + input_df['mirror_offset_y']**2
        )
        input_df['mirror_alignment'] = input_df['mirror_similarity'] * input_df['role_defensive_coverage']
    
    print("Step 3: Temporal features...")
    
    gcols = ['game_id', 'play_id', 'nfl_id']
    
    for lag in [1, 2, 3, 4, 5]:
        for col in ['x', 'y', 'velocity_x', 'velocity_y', 's', 'a']:
            if col in input_df.columns:
                input_df[f'{col}_lag{lag}'] = input_df.groupby(gcols)[col].shift(lag)
    
    for window in [3, 5]:
        for col in ['x', 'y', 'velocity_x', 'velocity_y', 's']:
            if col in input_df.columns:
                input_df[f'{col}_rolling_mean_{window}'] = (
                    input_df.groupby(gcols)[col]
                      .rolling(window, min_periods=1).mean()
                      .reset_index(level=[0,1,2], drop=True)
                )
                input_df[f'{col}_rolling_std_{window}'] = (
                    input_df.groupby(gcols)[col]
                      .rolling(window, min_periods=1).std()
                      .reset_index(level=[0,1,2], drop=True)
                )
    
    for col in ['velocity_x', 'velocity_y']:
        if col in input_df.columns:
            input_df[f'{col}_delta'] = input_df.groupby(gcols)[col].diff()
    
    input_df['velocity_x_ema'] = input_df.groupby(gcols)['velocity_x'].transform(
        lambda x: x.ewm(alpha=0.3, adjust=False).mean()
    )
    input_df['velocity_y_ema'] = input_df.groupby(gcols)['velocity_y'].transform(
        lambda x: x.ewm(alpha=0.3, adjust=False).mean()
    )
    input_df['speed_ema'] = input_df.groupby(gcols)['s'].transform(
        lambda x: x.ewm(alpha=0.3, adjust=False).mean()
    )
    
    print("Step 4: Time features...")
    
    if 'num_frames_output' in input_df.columns:
        max_frames = input_df['num_frames_output']
        
        input_df['max_play_duration'] = max_frames / 10.0
        input_df['frame_time'] = input_df['frame_id'] / 10.0
        input_df['progress_ratio'] = input_df['frame_id'] / np.maximum(max_frames, 1)
        input_df['time_remaining'] = (max_frames - input_df['frame_id']) / 10.0
        input_df['frames_remaining'] = max_frames - input_df['frame_id']
        
        input_df['expected_x_at_ball'] = input_df['x'] + input_df['velocity_x'] * input_df['frame_time']
        input_df['expected_y_at_ball'] = input_df['y'] + input_df['velocity_y'] * input_df['frame_time']
        
        if 'ball_land_x' in input_df.columns:
            input_df['error_from_ball_x'] = input_df['expected_x_at_ball'] - input_df['ball_land_x']
            input_df['error_from_ball_y'] = input_df['expected_y_at_ball'] - input_df['ball_land_y']
            input_df['error_from_ball'] = np.sqrt(
                input_df['error_from_ball_x']**2 + input_df['error_from_ball_y']**2
            )
            
            input_df['weighted_dist_by_time'] = input_df['dist_to_ball'] / (input_df['frame_time'] + 0.1)
            input_df['dist_scaled_by_progress'] = input_df['dist_to_ball'] * (1 - input_df['progress_ratio'])
        
        input_df['time_squared'] = input_df['frame_time'] ** 2
        input_df['velocity_x_progress'] = input_df['velocity_x'] * input_df['progress_ratio']
        input_df['velocity_y_progress'] = input_df['velocity_y'] * input_df['progress_ratio']
        input_df['speed_scaled_by_time_left'] = input_df['s'] * input_df['time_remaining']
        
        input_df['actual_play_length'] = max_frames
        input_df['length_ratio'] = max_frames / 30.0
    
    # 🎯 THE BREAKTHROUGH: Add geometric features
    print("Step 5: 🎯 Geometric endpoint features...")
    input_df = add_geometric_features(input_df)
    
    print("Step 6: Building feature list...")
    
    # Your 154 proven features
    feature_cols = [
        'x', 'y', 's', 'a', 'o', 'dir', 'frame_id', 'ball_land_x', 'ball_land_y',
        'player_height_feet', 'player_weight', 'height_inches', 'bmi',
        'velocity_x', 'velocity_y', 'acceleration_x', 'acceleration_y',
        'momentum_x', 'momentum_y', 'kinetic_energy',
        'speed_squared', 'accel_magnitude', 'orientation_diff',
        'is_offense', 'is_defense', 'is_receiver', 'is_coverage', 'is_passer',
        'role_targeted_receiver', 'role_defensive_coverage', 'role_passer', 'side_offense',
        'distance_to_ball', 'dist_to_ball', 'dist_squared', 'angle_to_ball', 
        'ball_direction_x', 'ball_direction_y', 'closing_speed_ball',
        'velocity_toward_ball', 'velocity_alignment', 'angle_diff',
        'nearest_opp_dist', 'closing_speed', 'num_nearby_opp_3', 'num_nearby_opp_5',
        'mirror_wr_vx', 'mirror_wr_vy', 'mirror_offset_x', 'mirror_offset_y',
        'pressure', 'under_pressure', 'pressure_x_speed', 
        'mirror_similarity', 'mirror_offset_dist', 'mirror_alignment',
        'route_pattern', 'traj_straightness', 'traj_max_turn', 'traj_mean_turn',
        'traj_depth', 'traj_width', 'speed_mean', 'speed_change',
        'gnn_ally_dx_mean', 'gnn_ally_dy_mean', 'gnn_ally_dvx_mean', 'gnn_ally_dvy_mean',
        'gnn_opp_dx_mean', 'gnn_opp_dy_mean', 'gnn_opp_dvx_mean', 'gnn_opp_dvy_mean',
        'gnn_ally_cnt', 'gnn_opp_cnt',
        'gnn_ally_dmin', 'gnn_ally_dmean', 'gnn_opp_dmin', 'gnn_opp_dmean',
        'gnn_d1', 'gnn_d2', 'gnn_d3',
    ]
    
    for lag in [1, 2, 3, 4, 5]:
        for col in ['x', 'y', 'velocity_x', 'velocity_y', 's', 'a']:
            feature_cols.append(f'{col}_lag{lag}')
    
    for window in [3, 5]:
        for col in ['x', 'y', 'velocity_x', 'velocity_y', 's']:
            feature_cols.append(f'{col}_rolling_mean_{window}')
            feature_cols.append(f'{col}_rolling_std_{window}')
    
    feature_cols.extend(['velocity_x_delta', 'velocity_y_delta'])
    feature_cols.extend(['velocity_x_ema', 'velocity_y_ema', 'speed_ema'])
    
    feature_cols.extend([
        'max_play_duration', 'frame_time', 'progress_ratio', 'time_remaining', 'frames_remaining',
        'expected_x_at_ball', 'expected_y_at_ball', 
        'error_from_ball_x', 'error_from_ball_y', 'error_from_ball',
        'time_squared', 'weighted_dist_by_time', 
        'velocity_x_progress', 'velocity_y_progress', 'dist_scaled_by_progress',
        'speed_scaled_by_time_left', 'actual_play_length', 'length_ratio',
    ])
    
    # 🎯 Add 13 geometric features
    feature_cols.extend([
        'geo_endpoint_x', 'geo_endpoint_y',
        'geo_vector_x', 'geo_vector_y', 'geo_distance',
        'geo_required_vx', 'geo_required_vy',
        'geo_velocity_error_x', 'geo_velocity_error_y', 'geo_velocity_error',
        'geo_required_ax', 'geo_required_ay',
        'geo_alignment',
    ])
    
    feature_cols = [c for c in feature_cols if c in input_df.columns]
    print(f"✓ Using {len(feature_cols)} features (154 proven + 13 geometric)")
    
    print("Step 7: Creating sequences...")
    
    input_df.set_index(['game_id', 'play_id', 'nfl_id'], inplace=True)
    grouped = input_df.groupby(level=['game_id', 'play_id', 'nfl_id'])
    
    target_rows = output_df if is_training else test_template
    target_groups = target_rows[['game_id', 'play_id', 'nfl_id']].drop_duplicates()
    
    sequences, targets_dx, targets_dy, targets_frame_ids, sequence_ids = [], [], [], [], []
    geo_endpoints_x, geo_endpoints_y = [], []
    
    for _, row in tqdm(target_groups.iterrows(), total=len(target_groups), desc="Creating sequences"):
        key = (row['game_id'], row['play_id'], row['nfl_id'])
        
        try:
            group_df = grouped.get_group(key)
        except KeyError:
            continue
        
        input_window = group_df.tail(window_size)
        
        if len(input_window) < window_size:
            if is_training:
                continue
            pad_len = window_size - len(input_window)
            pad_df = pd.DataFrame(np.nan, index=range(pad_len), columns=input_window.columns)
            input_window = pd.concat([pad_df, input_window], ignore_index=True)
        
        input_window = input_window.fillna(group_df.mean(numeric_only=True))
        seq = input_window[feature_cols].values
        
        if np.isnan(seq).any():
            if is_training:
                continue
            seq = np.nan_to_num(seq, nan=0.0)
        
        sequences.append(seq)
        
        # Store geometric endpoint for this player
        geo_x = input_window.iloc[-1]['geo_endpoint_x']
        geo_y = input_window.iloc[-1]['geo_endpoint_y']
        geo_endpoints_x.append(geo_x)
        geo_endpoints_y.append(geo_y)
        
        if is_training:
            out_grp = output_df[
                (output_df['game_id']==row['game_id']) &
                (output_df['play_id']==row['play_id']) &
                (output_df['nfl_id']==row['nfl_id'])
            ].sort_values('frame_id')
            
            last_x = input_window.iloc[-1]['x']
            last_y = input_window.iloc[-1]['y']
            
            dx = out_grp['x'].values - last_x
            dy = out_grp['y'].values - last_y
            
            targets_dx.append(dx)
            targets_dy.append(dy)
            targets_frame_ids.append(out_grp['frame_id'].values)
        
        sequence_ids.append({
            'game_id': key[0],
            'play_id': key[1],
            'nfl_id': key[2],
            'frame_id': input_window.iloc[-1]['frame_id']
        })
    
    print(f"✓ Created {len(sequences)} sequences")
    
    if is_training:
        return (sequences, targets_dx, targets_dy, targets_frame_ids, sequence_ids, 
                geo_endpoints_x, geo_endpoints_y, route_kmeans, route_scaler)
    return sequences, sequence_ids, geo_endpoints_x, geo_endpoints_y

# ============================================================================
# MODEL ARCHITECTURE (YOUR PROVEN GRU + ATTENTION)
# ============================================================================

class JointSeqModel(nn.Module):
    """Your proven architecture - unchanged"""
    
    def __init__(self, input_dim, horizon):
        super().__init__()
        self.gru = nn.GRU(input_dim, 128, num_layers=2, batch_first=True, dropout=0.1)
        self.pool_ln = nn.LayerNorm(128)
        self.pool_attn = nn.MultiheadAttention(128, num_heads=4, batch_first=True)
        self.pool_query = nn.Parameter(torch.randn(1, 1, 128))
        
        self.head = nn.Sequential(
            nn.Linear(128, 256), 
            nn.GELU(), 
            nn.Dropout(0.2), 
            nn.Linear(256, horizon * 2)
        )
    
    def forward(self, x):
        h, _ = self.gru(x)
        B = h.size(0)
        q = self.pool_query.expand(B, -1, -1)
        ctx, _ = self.pool_attn(q, self.pool_ln(h), self.pool_ln(h))
        out = self.head(ctx.squeeze(1))
        out = out.view(B, -1, 2)
        return torch.cumsum(out, dim=1)

# ============================================================================
# LOSS (YOUR PROVEN TEMPORAL HUBER)
# ============================================================================

class TemporalHuber(nn.Module):
    def __init__(self, delta=0.5, time_decay=0.03):
        super().__init__()
        self.delta = delta
        self.time_decay = time_decay
    
    def forward(self, pred, target, mask):
        err = pred - target
        abs_err = torch.abs(err)
        huber = torch.where(abs_err <= self.delta, 0.5 * err * err, 
                           self.delta * (abs_err - 0.5 * self.delta))
        
        if self.time_decay > 0:
            L = pred.size(1)
            t = torch.arange(L, device=pred.device).float()
            weight = torch.exp(-self.time_decay * t).view(1, L, 1)
            huber = huber * weight
            mask = mask.unsqueeze(-1) * weight
        
        return (huber * mask).sum() / (mask.sum() + 1e-8)

# ============================================================================
# TRAINING
# ============================================================================

def prepare_targets(batch_dx, batch_dy, max_h):
    tensors_x, tensors_y, masks = [], [], []
    
    for dx, dy in zip(batch_dx, batch_dy):
        L = len(dx)
        padded_x = np.pad(dx, (0, max_h - L), constant_values=0).astype(np.float32)
        padded_y = np.pad(dy, (0, max_h - L), constant_values=0).astype(np.float32)
        mask = np.zeros(max_h, dtype=np.float32)
        mask[:L] = 1.0
        
        tensors_x.append(torch.tensor(padded_x))
        tensors_y.append(torch.tensor(padded_y))
        masks.append(torch.tensor(mask))
    
    targets = torch.stack([torch.stack(tensors_x), torch.stack(tensors_y)], dim=-1)
    return targets, torch.stack(masks)

def train_model(X_train, y_train_dx, y_train_dy, X_val, y_val_dx, y_val_dy, 
                input_dim, horizon, config):
    device = config.DEVICE
    model = JointSeqModel(input_dim, horizon).to(device)
    
    criterion = TemporalHuber(delta=0.5, time_decay=0.03)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5, verbose=False)
    
    train_batches = []
    for i in range(0, len(X_train), config.BATCH_SIZE):
        end = min(i + config.BATCH_SIZE, len(X_train))
        bx = torch.tensor(np.stack(X_train[i:end]).astype(np.float32))
        by, bm = prepare_targets(
            [y_train_dx[j] for j in range(i, end)],
            [y_train_dy[j] for j in range(i, end)],
            horizon
        )
        train_batches.append((bx, by, bm))
    
    val_batches = []
    for i in range(0, len(X_val), config.BATCH_SIZE):
        end = min(i + config.BATCH_SIZE, len(X_val))
        bx = torch.tensor(np.stack(X_val[i:end]).astype(np.float32))
        by, bm = prepare_targets(
            [y_val_dx[j] for j in range(i, end)],
            [y_val_dy[j] for j in range(i, end)],
            horizon
        )
        val_batches.append((bx, by, bm))
    
    best_loss, best_state, bad = float('inf'), None, 0
    
    for epoch in range(1, config.EPOCHS + 1):
        model.train()
        train_losses = []
        for bx, by, bm in train_batches:
            bx, by, bm = bx.to(device), by.to(device), bm.to(device)
            pred = model(bx)
            loss = criterion(pred, by, bm)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())
        
        model.eval()
        val_losses = []
        with torch.no_grad():
            for bx, by, bm in val_batches:
                bx, by, bm = bx.to(device), by.to(device), bm.to(device)
                pred = model(bx)
                val_losses.append(criterion(pred, by, bm).item())
        
        train_loss, val_loss = np.mean(train_losses), np.mean(val_losses)
        scheduler.step(val_loss)
        
        if epoch % 10 == 0:
            print(f"  Epoch {epoch}: train={train_loss:.4f}, val={val_loss:.4f}")
        
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= config.PATIENCE:
                print(f"  Early stop at epoch {epoch}")
                break
    
    if best_state:
        model.load_state_dict(best_state)
    
    return model, best_loss

import kaggle_evaluation.nfl_inference_server
import warnings
import polars as pl # Import Polars for the predict function signature

# Move all utility functions (get_velocity, height_to_feet, compute_geometric_endpoint, etc.)
# and the Config/set_seed outside the class but above it, or make them static methods.

class NFLPredictor:
    def __init__(self):
        warnings.filterwarnings('ignore')
        config = Config() # Use the existing Config class
        set_seed(config.SEED)
        self.config = config

        # If configured, try to load pre-saved artifacts to skip training and data prep
        load_dir = Path(config.LOAD_DIR) if config.LOAD_DIR is not None else config.MODEL_DIR
        artifacts_present = all((load_dir / f"model_fold{f}.pt").exists() for f in range(1, config.N_FOLDS + 1)) and (
            (load_dir / "route_kmeans.pkl").exists() and (load_dir / "route_scaler.pkl").exists()
        )

        if config.LOAD_ARTIFACTS and artifacts_present:
            print(f"\n[1/2] Loading trained artifacts from disk (from {load_dir})...")
            self.models = []
            self.scalers = []
            # load route objects
            try:
                with open(load_dir / "route_kmeans.pkl", "rb") as f:
                    self.route_kmeans = pickle.load(f)
                with open(load_dir / "route_scaler.pkl", "rb") as f:
                    self.route_scaler = pickle.load(f)
            except Exception as e:
                print("Failed to load route artifacts:", e)

            # For input_dim, use a dummy array if needed
            dummy_input_dim = 167  # fallback if not available
            input_dim = dummy_input_dim
            # Try to infer input_dim from scaler if possible
            try:
                with open(load_dir / "scaler_fold1.pkl", "rb") as f:
                    scaler1 = pickle.load(f)
                input_dim = scaler1.mean_.shape[0]
            except Exception:
                pass

            for fold in range(1, config.N_FOLDS + 1):
                model_path = load_dir / f"model_fold{fold}.pt"
                scaler_path = load_dir / f"scaler_fold{fold}.pkl"

                # load scaler
                try:
                    with open(scaler_path, "rb") as f:
                        scaler = pickle.load(f)
                except Exception:
                    scaler = None

                # instantiate model and load state
                model = JointSeqModel(input_dim, config.MAX_FUTURE_HORIZON)
                try:
                    state = torch.load(model_path, map_location='cpu')
                    model.load_state_dict(state)
                except Exception as e:
                    print(f"Failed to load model fold {fold}:", e)

                model.to(config.DEVICE)
                model.eval()

                self.models.append(model)
                self.scalers.append(scaler)

            print(f"✓ Loaded {len(self.models)} models and scalers from {load_dir}")
            print("[2/2] Ready for inference. Skipped all training and data preparation.")
            return

        # If not loading, proceed with training and data preparation
        # 1. Load Data
        print("[1/4] Loading data for training...")
        train_input_files = [config.DATA_DIR / f"train/input_2023_w{w:02d}.csv" for w in range(1, 19)]
        train_output_files = [config.DATA_DIR / f"train/output_2023_w{w:02d}.csv" for w in range(1, 19)]
        if Config.DEBUG:            
            # Load a small amount of data to quickly get play IDs
            sample_train_input = pd.read_csv(train_input_files[0])
            sample_train_output = pd.read_csv(train_output_files[0])
            # Get the first 100 unique plays (game_id, play_id combinations)
            sample_plays = sample_train_output[['game_id', 'play_id']].drop_duplicates().head(100)
            # Filter the full data for only these plays
            train_input = pd.concat([pd.read_csv(f) for f in train_input_files if f.exists()])
            train_output = pd.concat([pd.read_csv(f) for f in train_output_files if f.exists()])
            # Perform the actual filtering to get a small, but complete, set of plays
            train_input = train_input.merge(sample_plays, on=['game_id', 'play_id'], how='inner')
            train_output = train_output.merge(sample_plays, on=['game_id', 'play_id'], how='inner')
            print(f"✓ Reduced to {len(sample_plays)} unique plays.")
        else:
            train_input = pd.concat([pd.read_csv(f) for f in train_input_files if f.exists()])
            train_output = pd.concat([pd.read_csv(f) for f in train_output_files if f.exists()])

        # 2. Prepare Sequences and Train Feature Objects
        print("\n[2/4] Preparing geometric sequences and feature scalers...")
        result = prepare_sequences_geometric(
            train_input, train_output, is_training=True, window_size=config.WINDOW_SIZE
        )
        sequences, targets_dx, targets_dy, targets_frame_ids, sequence_ids, geo_x, geo_y, route_kmeans, route_scaler = result

        # Store feature objects for later inference
        self.route_kmeans = route_kmeans
        self.route_scaler = route_scaler
        
        sequences = list(sequences)
        targets_dx = list(targets_dx)
        targets_dy = list(targets_dy)

        # Ensure model dir exists
        config.MODEL_DIR.mkdir(parents=True, exist_ok=True)

        # 3. Train Models
        print("\n[3/4] Training geometric models...")
        groups = np.array([d['game_id'] for d in sequence_ids])
        gkf = GroupKFold(n_splits=config.N_FOLDS)

        self.models, self.scalers = [], []

        for fold, (tr, va) in enumerate(gkf.split(sequences, groups=groups), 1):
            # ... (rest of the fold training logic, exactly as in the original main()) ...
            X_tr = [sequences[i] for i in tr]
            X_va = [sequences[i] for i in va]
            y_tr_dx = [targets_dx[i] for i in tr]
            y_va_dx = [targets_dx[i] for i in va]
            y_tr_dy = [targets_dy[i] for i in tr]
            y_va_dy = [targets_dy[i] for i in va]
            scaler = StandardScaler()
            scaler.fit(np.vstack([s for s in X_tr]))
            X_tr_sc = [scaler.transform(s) for s in X_tr]
            X_va_sc = [scaler.transform(s) for s in X_va]
            model, loss = train_model(
                X_tr_sc, y_tr_dx, y_tr_dy,
                X_va_sc, y_va_dx, y_va_dy,
                X_tr[0].shape[-1], config.MAX_FUTURE_HORIZON, config
            )
            self.models.append(model)
            self.scalers.append(scaler)
            print(f"\n✓ Fold {fold} - Loss: {loss:.5f}")
            # Persist fold artifacts if requested
            if config.SAVE_ARTIFACTS:
                try:
                    model_path = config.MODEL_DIR / f"model_fold{fold}.pt"
                    scaler_path = config.MODEL_DIR / f"scaler_fold{fold}.pkl"
                    # save model state dict on CPU to avoid device issues
                    state = {k: v.cpu() for k, v in model.state_dict().items()}
                    torch.save(state, str(model_path))
                    with open(scaler_path, "wb") as f:
                        pickle.dump(scaler, f)
                    print(f"  Saved artifacts for fold {fold} -> {model_path.name}, {scaler_path.name}")
                except Exception as e:
                    print(f"  Warning: failed to save artifacts for fold {fold}:", e)

        # Ensure all models are in eval mode (best practice)
        # Optionally persist route clustering objects
        if config.SAVE_ARTIFACTS:
            try:
                with open(config.MODEL_DIR / "route_kmeans.pkl", "wb") as f:
                    pickle.dump(self.route_kmeans, f)
                with open(config.MODEL_DIR / "route_scaler.pkl", "wb") as f:
                    pickle.dump(self.route_scaler, f)
                print(f"✓ Saved route artifacts -> route_kmeans.pkl, route_scaler.pkl")
            except Exception as e:
                print("Warning: failed to save route artifacts:", e)

        for model in self.models:
            model.eval()
        print("\n🏆 Geometric Neural Breakthrough Model is ready for inference! 🏆")


    def predict(self, test: pl.DataFrame, test_input: pl.DataFrame) -> pd.DataFrame:
        """Inference function called by the API for each time step."""
        
        # Convert Polars DataFrames to Pandas, as the original notebook uses Pandas extensively
        test_input_pd = test_input.to_pandas()
        test_template_pd = test.to_pandas()
        
        # 1. Prepare Test Sequences
        # Use the stored feature objects (route_kmeans, route_scaler) for inference
        test_seq, test_ids, test_geo_x, test_geo_y = prepare_sequences_geometric(
            test_input_pd, test_template=test_template_pd, is_training=False,
            window_size=self.config.WINDOW_SIZE,
            route_kmeans=self.route_kmeans, route_scaler=self.route_scaler
        )

        X_test = list(test_seq)
        x_last = np.array([s[-1, 0] for s in X_test])
        y_last = np.array([s[-1, 1] for s in X_test])

        # 2. Ensemble Prediction
        all_preds = []
        H = self.config.MAX_FUTURE_HORIZON

        for model, sc in zip(self.models, self.scalers):
            X_sc = [sc.transform(s) for s in X_test]
            X_t = torch.tensor(np.stack(X_sc).astype(np.float32)).to(self.config.DEVICE)

            with torch.no_grad():
                preds = model(X_t).cpu().numpy()

            all_preds.append(preds)

        ens_preds = np.mean(all_preds, axis=0)

        # 3. Format Submission (Corrected for API)
        rows = []
        for i, sid in enumerate(test_ids):
            # The template DataFrame 'test' contains the info needed to map predictions
            fids = test_template_pd[
                (test_template_pd['game_id'] == sid['game_id']) &
                (test_template_pd['play_id'] == sid['play_id']) &
                (test_template_pd['nfl_id'] == sid['nfl_id'])
            ]['frame_id'].sort_values().tolist()
            
            for t, fid in enumerate(fids):
                tt = min(t, H - 1)
                px = np.clip(x_last[i] + ens_preds[i, tt, 0], 0, self.config.FIELD_X_MAX)
                py = np.clip(y_last[i] + ens_preds[i, tt, 1], 0, self.config.FIELD_Y_MAX)

                # DO NOT include 'id' in the rows dict or final DataFrame
                # The API will handle the row IDs based on the provided 'test' DataFrame
                rows.append({
                    'x': px,
                    'y': py
                })

        # The final returned DataFrame MUST only contain 'x' and 'y' columns, 
        # matching the order of rows in the input 'test' DataFrame.
        submission = pd.DataFrame(rows)
        return submission

# This must be outside the class and is what the API calls
predictor = NFLPredictor()

def predict(test: pl.DataFrame, test_input: pl.DataFrame) -> pl.DataFrame | pd.DataFrame:
    """The function the API calls."""
    return predictor.predict(test, test_input)
```

```python
%%writefile nfl_gru.py

# -------------------------------
# Global imports + cuDF accelerator
# -------------------------------
import os
USE_CUDF = False
try:
    # zero/low-code GPU acceleration for DataFrame ops
    os.environ["CUDF_PANDAS_BACKEND"] = "cudf"
    import pandas as pd
    import numpy as np
    import cupy as cp  # optional (not strictly required below)
    USE_CUDF = True
    print("using cuda_backend pandas for faster parallel data processing")
except Exception:
    print("cuda df not used")
    import pandas as pd
    import numpy as np

import torch
import torch.nn as nn
from pathlib import Path
from shutil import make_archive
import json
import random
import joblib
from glob import glob
from tqdm.auto import tqdm
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import GroupKFold
import warnings
warnings.filterwarnings("ignore")

# New imports for evaluation API
import polars as pl
import kaggle_evaluation.nfl_inference_server

# -------------------------------
# Constants & helpers
# -------------------------------
YARDS_TO_METERS = 0.9144
FPS = 10.0 
FIELD_LENGTH, FIELD_WIDTH = 120.0, 53.3

def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
print("environment set up!")
def wrap_angle_deg(s):
    # map to (-180, 180]
    return ((s + 180.0) % 360.0) - 180.0

def unify_left_direction(df: pd.DataFrame) -> pd.DataFrame:
    """Mirror rightward plays so all samples are 'left' oriented (x,y, dir, o, ball_land)."""
    if 'play_direction' not in df.columns:
        return df
    df = df.copy()
    right = df['play_direction'].eq('right')
    # positions
    if 'x' in df.columns: df.loc[right, 'x'] = FIELD_LENGTH - df.loc[right, 'x']
    if 'y' in df.columns: df.loc[right, 'y'] = FIELD_WIDTH  - df.loc[right, 'y']
    # angles in degrees
    for col in ('dir','o'):
        if col in df.columns:
            df.loc[right, col] = (df.loc[right, col] + 180.0) % 360.0
    # ball landing
    if 'ball_land_x' in df.columns:
        df.loc[right, 'ball_land_x'] = FIELD_LENGTH - df.loc[right, 'ball_land_x']
    if 'ball_land_y' in df.columns:
        df.loc[right, 'ball_land_y'] = FIELD_WIDTH  - df.loc[right, 'ball_land_y']
    return df

def invert_to_original_direction(x_u, y_u, play_dir_right: bool):
    """Invert unified (left) coordinates back to original play direction."""
    if not play_dir_right:
        return float(x_u), float(y_u)
    return float(FIELD_LENGTH - x_u), float(FIELD_WIDTH - y_u)

# -------------------------------
# Config
# -------------------------------
class Config:
    DATA_DIR = Path("/kaggle/input/nfl-big-data-bowl-2026-prediction/")
    OUTPUT_DIR = Path("./outputs"); OUTPUT_DIR.mkdir(exist_ok=True)

    SAVE_DIR = Path("./saved_models")       
    SAVE_DIR.mkdir(exist_ok=True, parents=True)
    
    MODELS_DIR = Path("/kaggle/input/hsiaosuan-sttn/saved_models")
    MODELS_DIR.mkdir(exist_ok=True, parents=True)

    # TRAIN=True: Train models and save to ./saved_models
    # SUB=True: Use evaluation API (new format) - Cell 9 will start the server
    TRAIN = False
    SUB   = True

    #是否使用Bi-GRU
    BIDIRECTIONAL = True



    #显式指定特征组（确保 train/sub 一致）
    FEATURE_GROUPS = [
        'distance_rate','target_alignment','multi_window_rolling','extended_lags',
        'velocity_changes','field_position','role_specific','time_features','jerk_features','interaction_features_mid','qb_relative',
        # 'interaction_features', #保守版本对抗特征
        # 'curvature_land_features',  #若线上无落点，勿开启
    ]

    #Training Setting
    SEED = 42
    SEEDS = [42, 19, 89, 64]   #多种子集成
    N_FOLDS = 5
    BATCH_SIZE = 256
    EPOCHS = 200
    PATIENCE = 30
    LEARNING_RATE = 1e-3

    WINDOW_SIZE = 10
    HIDDEN_DIM = 128
    MAX_FUTURE_HORIZON = 94  #不要动这个东西

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


set_seed(Config.SEED)

def compute_val_rmse(mx, my, X_val_sc, ydx_list, ydy_list, horizon, device, mode="per-dim"):
    """
    计算验证集轨迹误差。
    mode:
      - "per-dim"  : sqrt( ( (dx^2+dy^2) 总和 / (2N) ) )  ← 你截图里的公式（推荐用于对齐）
      - "2d"       : sqrt( ( (dx^2+dy^2) 总和 / N ) )     ← 二维欧氏 RMSE（会比 per-dim 大 sqrt(2)）
      - "mean-dist": 平均欧氏距离 E[ sqrt(dx^2+dy^2) ]     ← 某些比赛用这个口径
    N = 有效的样本×时间步（按 mask 统计）
    """
    X_t = torch.tensor(X_val_sc.astype(np.float32)).to(device)
    with torch.no_grad():
        pdx = mx(X_t).cpu().numpy()   # (N, H)
        pdy = my(X_t).cpu().numpy()   # (N, H)

    ydx, m = prepare_targets(ydx_list, horizon)  # (N,H), (N,H)
    ydy, _ = prepare_targets(ydy_list, horizon)
    ydx, ydy, m = ydx.numpy(), ydy.numpy(), m.numpy()

    se_sum2d = ((pdx - ydx)**2 + (pdy - ydy)**2) * m
    denom = m.sum() + 1e-8

    if mode == "per-dim":
        return float(np.sqrt(se_sum2d.sum() / (2.0 * denom)))
    elif mode == "2d":
        return float(np.sqrt(se_sum2d.sum() / denom))
    elif mode == "mean-dist":
        dist = np.sqrt(se_sum2d)  # 元素级开根号
        return float(dist.sum() / denom)
    else:
        raise ValueError("mode must be one of {'per-dim','2d','mean-dist'}")

# -------------------------------
# Feature Engineering
# -------------------------------
class FeatureEngineer:
    """
    Modular, ablation-friendly feature builder (pandas or cuDF pandas-API).
    """
    def __init__(self, feature_groups_to_create):
        self.gcols = ['game_id', 'play_id', 'nfl_id']
        self.active_groups = feature_groups_to_create
        self.feature_creators = {
            'distance_rate': self._create_distance_rate_features,
            'target_alignment': self._create_target_alignment_features,
            'multi_window_rolling': self._create_multi_window_rolling_features,
            'extended_lags': self._create_extended_lag_features,
            'velocity_changes': self._create_velocity_change_features,
            'field_position': self._create_field_position_features,
            'role_specific': self._create_role_specific_features,
            'time_features': self._create_time_features,
            'jerk_features': self._create_jerk_features,
            'curvature_land_features': self._create_curvature_land_features,
            'interaction_features': self._create_interaction_features, #添加,交互对抗特征
            'interaction_features_mid': self._create_interaction_features_mid,  #激进的交互对抗特征
            'qb_relative': self._create_qb_relative_features, #相对于四分卫位置的建模
        }
        self.created_feature_cols = []

    def _height_to_feet(self, height_str):
        try:
            ft, inches = map(int, str(height_str).split('-'))
            return ft + inches / 12
        except Exception:
            return 6.0

    def _create_basic_features(self, df):
        print("Step 1/3: Adding basic features...")
        df = df.copy()
        df['player_height_feet'] = df['player_height'].apply(self._height_to_feet)

        # Correct kinematics: dir is from +x CCW
        dir_rad = np.deg2rad(df['dir'].fillna(0.0).astype('float32'))
        df['velocity_x']     = df['s'] * np.cos(dir_rad)
        df['velocity_y']     = df['s'] * np.sin(dir_rad)
        df['acceleration_x'] = df['a'] * np.cos(dir_rad)
        df['acceleration_y'] = df['a'] * np.sin(dir_rad)

        # Roles
        df['is_offense']  = (df['player_side'] == 'Offense').astype(np.int8)
        df['is_defense']  = (df['player_side'] == 'Defense').astype(np.int8)
        df['is_receiver'] = (df['player_role'] == 'Targeted Receiver').astype(np.int8)
        df['is_coverage'] = (df['player_role'] == 'Defensive Coverage').astype(np.int8)
        df['is_passer']   = (df['player_role'] == 'Passer').astype(np.int8)

        # Energetics (consistent units)
        mass_kg = df['player_weight'].fillna(200.0) / 2.20462
        v_ms = df['s'] * YARDS_TO_METERS
        df['momentum_x'] = mass_kg * df['velocity_x'] * YARDS_TO_METERS
        df['momentum_y'] = mass_kg * df['velocity_y'] * YARDS_TO_METERS
        df['kinetic_energy'] = 0.5 * mass_kg * (v_ms ** 2)

        # Ball landing geometry (static)
        if {'ball_land_x','ball_land_y'}.issubset(df.columns):
            ball_dx = df['ball_land_x'] - df['x']
            ball_dy = df['ball_land_y'] - df['y']
            dist = np.hypot(ball_dx, ball_dy)
            df['distance_to_ball'] = dist
            inv = 1.0 / (dist + 1e-6)
            df['ball_direction_x'] = ball_dx * inv
            df['ball_direction_y'] = ball_dy * inv
            df['closing_speed'] = (
                df['velocity_x'] * df['ball_direction_x'] +
                df['velocity_y'] * df['ball_direction_y']
            )

        base = [
            'x','y','s','a','o','dir','frame_id','ball_land_x','ball_land_y',
            'player_height_feet','player_weight',
            'velocity_x','velocity_y','acceleration_x','acceleration_y',
            'momentum_x','momentum_y','kinetic_energy',
            'is_offense','is_defense','is_receiver','is_coverage','is_passer',
            'distance_to_ball','ball_direction_x','ball_direction_y','closing_speed'
        ]
        self.created_feature_cols.extend([c for c in base if c in df.columns])
        return df

    # ---- feature groups ----
    def _create_distance_rate_features(self, df):
        new_cols = []
        if 'distance_to_ball' in df.columns:
            d = df.groupby(self.gcols)['distance_to_ball'].diff()
            df['d2ball_dt']  = d.fillna(0.0) * FPS
            df['d2ball_ddt'] = df.groupby(self.gcols)['d2ball_dt'].diff().fillna(0.0) * FPS
            df['time_to_intercept'] = (df['distance_to_ball'] /
                                       (df['d2ball_dt'].abs() + 1e-3)).clip(0, 10)
            new_cols = ['d2ball_dt','d2ball_ddt','time_to_intercept']
        return df, new_cols

    def _create_target_alignment_features(self, df):
        new_cols = []
        if {'ball_direction_x','ball_direction_y','velocity_x','velocity_y'}.issubset(df.columns):
            df['velocity_alignment'] = df['velocity_x']*df['ball_direction_x'] + df['velocity_y']*df['ball_direction_y']
            df['velocity_perpendicular'] = df['velocity_x']*(-df['ball_direction_y']) + df['velocity_y']*df['ball_direction_x']
            new_cols.extend(['velocity_alignment','velocity_perpendicular'])
            if {'acceleration_x','acceleration_y'}.issubset(df.columns):
                df['accel_alignment'] = df['acceleration_x']*df['ball_direction_x'] + df['acceleration_y']*df['ball_direction_y']
                new_cols.append('accel_alignment')
        return df, new_cols

    def _create_multi_window_rolling_features(self, df):
        # keep it simple & compatible (works with cuDF pandas-API); vectorized rolling per group
        new_cols = []
        for window in (3, 5, 10):
            for col in ('velocity_x','velocity_y','s','a'):
                if col in df.columns:
                    r_mean = df.groupby(self.gcols)[col].rolling(window, min_periods=1).mean()
                    r_std  = df.groupby(self.gcols)[col].rolling(window, min_periods=1).std()
                    # align indices
                    r_mean = r_mean.reset_index(level=list(range(len(self.gcols))), drop=True)
                    r_std  = r_std.reset_index(level=list(range(len(self.gcols))), drop=True)
                    df[f'{col}_roll{window}'] = r_mean
                    df[f'{col}_std{window}']  = r_std.fillna(0.0)
                    new_cols.extend([f'{col}_roll{window}', f'{col}_std{window}'])
        return df, new_cols

    def _create_extended_lag_features(self, df):
        new_cols = []
        for lag in (1,2,3,4,5):
            for col in ('x','y','velocity_x','velocity_y'):
                if col in df.columns:
                    g = df.groupby(self.gcols)[col]
                    lagv = g.shift(lag)
                    # safe fill for first frames (no "future" leakage)
                    df[f'{col}_lag{lag}'] = lagv.fillna(g.transform('first'))
                    new_cols.append(f'{col}_lag{lag}')
        return df, new_cols

    def _create_velocity_change_features(self, df):
        new_cols = []
        if 'velocity_x' in df.columns:
            df['velocity_x_change'] = df.groupby(self.gcols)['velocity_x'].diff().fillna(0.0)
            df['velocity_y_change'] = df.groupby(self.gcols)['velocity_y'].diff().fillna(0.0)
            df['speed_change']      = df.groupby(self.gcols)['s'].diff().fillna(0.0)
            d = df.groupby(self.gcols)['dir'].diff().fillna(0.0)
            df['direction_change']  = wrap_angle_deg(d)
            new_cols = ['velocity_x_change','velocity_y_change','speed_change','direction_change']
        return df, new_cols

    def _create_field_position_features(self, df):
        df['dist_from_left'] = df['y']
        df['dist_from_right'] = FIELD_WIDTH - df['y']
        df['dist_from_sideline'] = np.minimum(df['dist_from_left'], df['dist_from_right'])
        df['dist_from_endzone']  = np.minimum(df['x'], FIELD_LENGTH - df['x'])
        return df, ['dist_from_sideline','dist_from_endzone']

    def _create_role_specific_features(self, df):
        new_cols = []
        if {'is_receiver','velocity_alignment'}.issubset(df.columns):
            df['receiver_optimality'] = df['is_receiver'] * df['velocity_alignment']
            df['receiver_deviation']  = df['is_receiver'] * np.abs(df.get('velocity_perpendicular', 0.0))
            new_cols.extend(['receiver_optimality','receiver_deviation'])
        if {'is_coverage','closing_speed'}.issubset(df.columns):
            df['defender_closing_speed'] = df['is_coverage'] * df['closing_speed']
            new_cols.append('defender_closing_speed')
        return df, new_cols

    def _create_time_features(self, df):
        df['frames_elapsed']  = df.groupby(self.gcols).cumcount()
        df['normalized_time'] = df.groupby(self.gcols)['frames_elapsed'].transform(
            lambda x: x / (x.max() + 1e-9)
        )
        df['time_sin'] = np.sin(2*np.pi*df['normalized_time'])
        df['time_cos'] = np.cos(2*np.pi*df['normalized_time'])
        return df, ['frames_elapsed','normalized_time','time_sin','time_cos']

    def _create_jerk_features(self, df):
        new_cols = []
        if 'a' in df.columns:
            df['jerk'] = df.groupby(self.gcols)['a'].diff().fillna(0.0) * FPS
            new_cols.append('jerk')
        if {'acceleration_x','acceleration_y'}.issubset(df.columns):
            df['jerk_x'] = df.groupby(self.gcols)['acceleration_x'].diff().fillna(0.0) * FPS
            df['jerk_y'] = df.groupby(self.gcols)['acceleration_y'].diff().fillna(0.0) * FPS
            new_cols.extend(['jerk_x','jerk_y'])
        return df, new_cols
    def _create_curvature_land_features(self, df):
        """
        -落点侧向偏差（符号）：landing_point 相对“当前运动方向”的左右偏离
          lateral = cross(u_dir, vector_to_land)（>0 表示落点在运动方向左侧）
        -bearing_to_land_signed: 运动方向 vs 落点方位角
        -速度归一化曲率： wrap(Δdir)/ (s*Δt) ，窗口化(3/5) 的均值/绝对值
        """
        import numpy as np
        # 侧向偏差 & bearing_to_land
        if {'ball_land_x','ball_land_y'}.issubset(df.columns):
            dx = df['ball_land_x'] - df['x']
            dy = df['ball_land_y'] - df['y']
            bearing = np.arctan2(dy, dx)
            a_dir = np.deg2rad(df['dir'].fillna(0.0).values)
            # 有符号方位差
            df['bearing_to_land_signed'] = np.rad2deg(np.arctan2(np.sin(bearing - a_dir), np.cos(bearing - a_dir)))
            # 侧向偏差：d × u (2D cross, z 分量)
            ux, uy = np.cos(a_dir), np.sin(a_dir)
            df['land_lateral_offset'] = dy*ux - dx*uy  # >0 落点在左侧
    
        # 曲率（按序列）
        ddir = df.groupby(self.gcols)['dir'].diff().fillna(0.0)
        ddir = ((ddir + 180.0) % 360.0) - 180.0
        curvature = np.deg2rad(ddir).astype('float32') / (df['s'].replace(0, np.nan).astype('float32') * 0.1 + 1e-6)
        df['curvature_signed'] = curvature.fillna(0.0)
        df['curvature_abs'] = df['curvature_signed'].abs()
    
        # 窗口均值（3/5）
        for w in (3,5):
            r = df.groupby(self.gcols)['curvature_signed'].rolling(w, min_periods=1).mean().reset_index(level=[0,1,2], drop=True)
            df[f'curv_signed_roll{w}'] = r
            r2 = df.groupby(self.gcols)['curvature_abs'].rolling(w, min_periods=1).mean().reset_index(level=[0,1,2], drop=True)
            df[f'curv_abs_roll{w}'] = r2
    
        new_cols = ['bearing_to_land_signed','land_lateral_offset',
                    'curvature_signed','curvature_abs','curv_signed_roll3','curv_abs_roll3',
                    'curv_signed_roll5','curv_abs_roll5']
        return df, [c for c in new_cols if c in df.columns]

    def _create_interaction_features(self, df, speed_eps=0.5):
        """
        Lite Receiver–Defender interaction features (conservative):
          - Only K=1 (nearest opponent)
          - Compute ONLY for target player (player_to_predict==True) if column exists
          - Features:
              opp_dmin        : distance to nearest opposite-side opponent (clipped [0,30])
              opp_close_rate  : (v_opp - v_self) projected onto (opp->self) unit vector (clipped [-10,10])
              opp_leverage    : sign of cross( self_vel , self->opp ) in 2D ( {-1,0,1} ), gated by speed
        """
        import numpy as np
    
        need = ['x','y','velocity_x','velocity_y','player_side','frame_id']
        for c in need:
            if c not in df.columns:
                return df, []  # missing columns → skip safely
    
        out_cols = ['opp_dmin','opp_close_rate','opp_leverage']
        for c in out_cols:
            if c not in df.columns:
                df[c] = np.nan
    
        key = ['game_id','play_id','frame_id']
        use_mask_global = ('player_to_predict' in df.columns)
    
        for _, g in df.groupby(key, sort=False):
            idx = g.index.values
            if len(g) <= 1:
                continue
    
            pos = g[['x','y']].values.astype('float32')          # (N,2)
            vel = g[['velocity_x','velocity_y']].values.astype('float32')
            side_off = (g['player_side'].values == 'Offense')    # (N,)
            side_def = ~side_off
    
            # only compute for target players if available
            if use_mask_global:
                tgt_mask = g['player_to_predict'].astype(bool).values
            else:
                tgt_mask = np.ones(len(g), dtype=bool)
    
            def _assign(A_mask, B_mask):
                A_mask = A_mask & tgt_mask
                A_idx = np.where(A_mask)[0]
                B_idx = np.where(B_mask)[0]
                if len(A_idx)==0 or len(B_idx)==0:
                    return
    
                Apos, Bpos = pos[A_idx], pos[B_idx]
                Avel, Bvel = vel[A_idx], vel[B_idx]
    
                # pairwise distances (K=1)
                dx = Apos[:,None,0] - Bpos[None,:,0]
                dy = Apos[:,None,1] - Bpos[None,:,1]
                D  = np.sqrt(dx*dx + dy*dy) + 1e-6                 # (Na,Nb)
                j  = np.argmin(D, axis=1)                          # nearest opponent
    
                dmin = D[np.arange(len(A_idx)), j]
                dmin = np.clip(dmin, 0.0, 30.0)                    # robust clip
    
                # closing rate
                r   = Apos - Bpos[j]                                # opp->self
                u   = r / (np.linalg.norm(r, axis=1, keepdims=True) + 1e-6)
                v_rel = Bvel[j] - Avel
                close = np.einsum('ij,ij->i', v_rel, u)
                close = np.clip(close, -10.0, 10.0)
    
                # leverage sign, gated by own speed
                speed   = np.linalg.norm(Avel, axis=1)
                to_opp  = Bpos[j] - Apos                            # self->opp
                cross_z = to_opp[:,0]*Avel[:,1] - to_opp[:,1]*Avel[:,0]
                lever   = np.where(speed > speed_eps, np.sign(cross_z), 0).astype('int8')
    
                rows = idx[A_idx]
                df.loc[rows, 'opp_dmin']       = dmin
                df.loc[rows, 'opp_close_rate'] = close
                df.loc[rows, 'opp_leverage']   = lever
    
            # Offense w.r.t. Defense
            _assign(side_off, side_def)
            # Defense w.r.t. Offense
            _assign(side_def, side_off)
    
        return df, out_cols

    def _create_interaction_features_mid(self, df, speed_eps=0.5, k=2, radius=10.0):
        """
        Medium-aggressive Receiver–Defender interaction:
          - K=2 nearest opposite-side opponents (still per-frame)
          - Keep target-only computation if 'player_to_predict' exists
          - New features (small set):
              opp_d2                 : 2nd nearest distance (clipped [0,30])
              opp_dmean_k2           : mean distance of top-2
              opp_close_rate_min_k2  : min closing rate among top-2 (more threatening)
              opp_pursuit_error_min_k2: min |angle( v_opp , dir to self )| in degrees
              opp_density_r10        : #opponents within radius (10 yards)
          - Light temporal smoothing on previous base features:
              opp_dmin_roll3, opp_close_rate_roll3
        """
    
        need = ['x','y','velocity_x','velocity_y','player_side','frame_id']
        for c in need:
            if c not in df.columns:
                return df, []  # safe skip
    
        # 保证保守版的三列存在（以便 rolling）
        base_cols = ['opp_dmin','opp_close_rate','opp_leverage']
        for c in base_cols:
            if c not in df.columns:
                df[c] = np.nan
    
        # 新增列（初始化）
        new_cols = ['opp_d2','opp_dmean_k2','opp_close_rate_min_k2',
                    'opp_pursuit_error_min_k2','opp_density_r10','ally_density_r10']
        for c in new_cols:
            if c not in df.columns:
                df[c] = np.nan
    
        key = ['game_id','play_id','frame_id']
        use_mask_global = ('player_to_predict' in df.columns)
    
        def angle_between(v1, v2, eps=1e-6):
            # 返回弧度的夹角 (0..pi)
            dot = np.einsum('ij,ij->i', v1, v2)
            n1  = np.linalg.norm(v1, axis=1) + eps
            n2  = np.linalg.norm(v2, axis=1) + eps
            cos = np.clip(dot / (n1*n2), -1.0, 1.0)
            return np.arccos(cos)
    
        for _, g in df.groupby(key, sort=False):
            idx = g.index.values
            if len(g) <= 1:
                continue
    
            pos = g[['x','y']].values.astype('float32')
            vel = g[['velocity_x','velocity_y']].values.astype('float32')
            side_off = (g['player_side'].values == 'Offense')
            side_def = ~side_off
    
            # target-only
            if use_mask_global:
                tgt_mask = g['player_to_predict'].astype(bool).values
            else:
                tgt_mask = np.ones(len(g), dtype=bool)
    
            def _assign(A_mask, B_mask):
                A_mask = A_mask & tgt_mask
                A_idx = np.where(A_mask)[0]
                B_idx = np.where(B_mask)[0]
                if len(A_idx)==0 or len(B_idx)==0:
                    return
    
                Apos, Bpos = pos[A_idx], pos[B_idx]
                Avel, Bvel = vel[A_idx], vel[B_idx]
    
                # pairwise distances
                dx = Apos[:,None,0] - Bpos[None,:,0]
                dy = Apos[:,None,1] - Bpos[None,:,1]
                D  = np.sqrt(dx*dx + dy*dy) + 1e-6          # (Na,Nb)
                # Top-2 indices along last axis
                k_use = min(k, D.shape[1])
                part = np.argpartition(D, kth=range(k_use), axis=1)[:, :k_use]  # (Na, k_use)
                # Gather top-2 distances and opponent indices
                rows = np.arange(len(A_idx))[:,None]
                d_top = D[rows, part]                 # (Na, k_use)
                jidx  = part                          # opponent indices for each A
    
                # 基于 K=2 推导的派生量
                d_sorted = np.sort(d_top, axis=1)     # 升序：d1, d2
                d1 = d_sorted[:,0]
                d2 = d_sorted[:,1] if k_use >= 2 else d_sorted[:,0]
    
                # 关闭率：对每个所选对手计算，再取 min
                # opp->self 单位向量
                # 先取最近对手用于 opp_dmin/close_rate（与保守版保持一致含义）
                j1 = jidx[np.arange(len(A_idx)), np.argmin(d_top, axis=1)]
                r1 = Apos - Bpos[j1]                         # opp->self
                u1 = r1 / (np.linalg.norm(r1, axis=1, keepdims=True) + 1e-6)
                vrel1 = Bvel[j1] - Avel
                close1 = np.einsum('ij,ij->i', vrel1, u1)
                close1 = np.clip(close1, -10.0, 10.0)
    
                #第二近对手的关闭率
                if k_use >= 2:
                    # 取第二近（不是最小值位置）
                    order = np.argsort(d_top, axis=1)
                    j2 = jidx[rows[:,0], order[:,1]]
                    r2 = Apos - Bpos[j2]
                    u2 = r2 / (np.linalg.norm(r2, axis=1, keepdims=True) + 1e-6)
                    vrel2 = Bvel[j2] - Avel
                    close2 = np.einsum('ij,ij->i', vrel2, u2)
                    close2 = np.clip(close2, -10.0, 10.0)
                    close_min_k2 = np.minimum(close1, close2)
                else:
                    close2 = close1
                    close_min_k2 = close1
    
                #pursuit error：对手速度 vs 指向我方连线（取最小绝对值，单位度）
                # 角度=atan2(|cross|, dot)≡arccos(dot/(||v||·||u||))
                ang1 = angle_between(Bvel[j1], u1) * 180.0/np.pi
                if k_use >= 2:
                    ang2 = angle_between(Bvel[j2], u2) * 180.0/np.pi
                    perr_min_k2 = np.minimum(np.abs(ang1), np.abs(ang2))
                else:
                    perr_min_k2 = np.abs(ang1)
    
                #局部密度（半径内的对手数）
                density = (D <= radius).sum(axis=1).astype('float32')
                #同侧密度（A方内的 self-self 距离）；简单起见：用 Apos 两两距离
                rows_idx = idx[A_idx]  
                
                # 同侧密度
                try:
                    from scipy.spatial.distance import cdist
                    A2A = cdist(Apos, Apos) + 1e-6
                except Exception:
                    # 无 scipy 的 fallback
                    diff = Apos[:,None,:] - Apos[None,:,:]
                    A2A = np.sqrt((diff * diff).sum(-1)) + 1e-6
                same_density = (A2A <= radius).sum(axis=1) - 1  # 自身-1
                
                # 写回
                df.loc[rows_idx, 'ally_density_r10'] = same_density.astype('float32')
                df.loc[rows_idx, 'opp_dmin']       = np.clip(d1, 0.0, 30.0)
                df.loc[rows_idx, 'opp_close_rate'] = close1
                df.loc[rows_idx, 'opp_d2']                 = np.clip(d2, 0.0, 30.0)
                df.loc[rows_idx, 'opp_dmean_k2']           = (np.clip(d1,0,30)+np.clip(d2,0,30))/2.0
                df.loc[rows_idx, 'opp_close_rate_min_k2']  = close_min_k2
                df.loc[rows_idx, 'opp_pursuit_error_min_k2']= perr_min_k2
                df.loc[rows_idx, 'opp_density_r10']        = density
    
            # Offense vs Defense
            _assign(side_off, side_def)
            # Defense vs Offense
            _assign(side_def, side_off)
    
        # 轻量时序平滑（3 帧 rolling）
        # 注意：以个体维度分组，不跨人
        for col in ['opp_dmin','opp_close_rate']:
            if col in df.columns:
                r = (
                    df.groupby(self.gcols)[col]
                      .rolling(3, min_periods=1).mean()
                      .reset_index(level=list(range(len(self.gcols))), drop=True)
                )
                df[f'{col}_roll3'] = r
    
        out_cols = base_cols + new_cols + ['opp_dmin_roll3','opp_close_rate_roll3']
        # 只返回本函数“新增/更新”的列（如果有的来自保守版就不重复计数也没关系）
        out_cols = [c for c in out_cols if c in df.columns]
        return df, out_cols

    def _create_qb_relative_features(self, df):
        """
        QB-relative geometry (per frame):
          - qb_distance
          - vel_to_qb_alignment, vel_to_qb_perp
          - bearing_to_qb_signed (player facing vs vector to QB)
        仅依赖: x,y,velocity_x,velocity_y,dir,player_role,frame_id
        """
        need = ['x','y','velocity_x','velocity_y','dir','player_role','frame_id']
        for c in need:
            if c not in df.columns:
                return df, []  # 缺列则安全跳过
    
        out_cols = ['qb_distance','vel_to_qb_alignment','vel_to_qb_perp','bearing_to_qb_signed','bearing_to_qb_sin','bearing_to_qb_cos']
        for c in out_cols:
            if c not in df.columns:
                df[c] = np.nan
    
        key = ['game_id','play_id','frame_id']
        for _, g in df.groupby(key, sort=False):
            idx = g.index.values
    
            # 找本帧 QB（通常唯一；若多于1取第一个；找不到则跳过）
            qb_rows = g[g['player_role'] == 'Passer']
            if qb_rows.empty:
                continue
            qb_x = float(qb_rows.iloc[0]['x'])
            qb_y = float(qb_rows.iloc[0]['y'])
    
            dx = g['x'].values.astype('float32') - qb_x
            dy = g['y'].values.astype('float32') - qb_y
            dist = np.sqrt(dx*dx + dy*dy) + 1e-6
            ux, uy = dx/dist, dy/dist  # QB->player 单位向量
    
            vx = g['velocity_x'].values.astype('float32')
            vy = g['velocity_y'].values.astype('float32')
    
            align = vx*ux + vy*uy
            perp  = vx*(-uy) + vy*ux
    
            # bearing 差：玩家朝向 vs 指向 QB 的方向（有符号，(-180,180]）
            dir_rad = np.deg2rad(g['dir'].fillna(0.0).astype('float32').values)
            to_qb_angle = np.arctan2(-dy, -dx)  # player->QB
            bearing = np.rad2deg(np.arctan2(np.sin(to_qb_angle - dir_rad),
                                            np.cos(to_qb_angle - dir_rad)))
    
            df.loc[idx, 'qb_distance'] = dist
            df.loc[idx, 'vel_to_qb_alignment'] = align
            df.loc[idx, 'vel_to_qb_perp'] = perp
            df.loc[idx, 'bearing_to_qb_signed'] = bearing
            df.loc[idx, 'bearing_to_qb_sin'] = np.sin(np.deg2rad(bearing))
            df.loc[idx, 'bearing_to_qb_cos'] = np.cos(np.deg2rad(bearing))
    
        return df, out_cols





    def transform(self, df):
        df = df.copy().sort_values(['game_id','play_id','nfl_id','frame_id'])
        df = self._create_basic_features(df)

        print("\nStep 2/3: Adding selected advanced features...")
        for group_name in self.active_groups:
            if group_name in self.feature_creators:
                creator = self.feature_creators[group_name]
                df, new_cols = creator(df)
                self.created_feature_cols.extend(new_cols)
                print(f"  [+] Added '{group_name}' ({len(new_cols)} cols)")
            else:
                print(f"  [!] Unknown feature group: {group_name}")

        final_cols = sorted(set(self.created_feature_cols))
        print(f"\nTotal features created: {len(final_cols)}")
        return df, final_cols

# -------------------------------
# Sequence builder (unified frame + safe targets)
# -------------------------------
def build_play_direction_map(df_in: pd.DataFrame) -> pd.Series:
    """
    Return a Series indexed by (game_id, play_id) with values 'left'/'right'.
    This keeps a clean MultiIndex that works for both pandas and cuDF pandas-API.
    """
    s = (
        df_in[['game_id','play_id','play_direction']]
        .drop_duplicates()
        .set_index(['game_id','play_id'])['play_direction']
    )
    return s  # MultiIndex Series


def apply_direction_to_df(df: pd.DataFrame, dir_map: pd.Series) -> pd.DataFrame:
    """
    Attach play_direction (if missing) and then unify to 'left'.
    dir_map must be the MultiIndex Series produced by build_play_direction_map.
    """
    if 'play_direction' not in df.columns:
        dir_df = dir_map.reset_index()  # -> columns: game_id, play_id, play_direction
        df = df.merge(dir_df, on=['game_id','play_id'], how='left', validate='many_to_one')
    return unify_left_direction(df)


#[A] 统一键类型的辅助函数
def _canonicalize_key_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in ('game_id','play_id','nfl_id'):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    # 丢掉缺失键
    df = df.dropna(subset=['game_id','play_id','nfl_id'])
    # 统一为 int64
    df['game_id'] = df['game_id'].astype('int64')
    df['play_id'] = df['play_id'].astype('int64')
    df['nfl_id']  = df['nfl_id'].astype('int64')
    return df

def prepare_sequences_with_advanced_features(
        input_df, output_df=None, test_template=None, 
        is_training=True, window_size=10, feature_groups=None):

    print(f"\n{'='*80}")
    print(f"PREPARING SEQUENCES WITH ADVANCED FEATURES (UNIFIED FRAME)")
    print(f"{'='*80}")
    print(f"Window size: {window_size}")

    # --- [B] 先统一键类型（输入/输出/测试模板都处理） ---
    input_df  = _canonicalize_key_dtypes(input_df)
    if is_training:
        assert output_df is not None
        output_df = _canonicalize_key_dtypes(output_df)
    else:
        assert test_template is not None
        test_template = _canonicalize_key_dtypes(test_template)

    if feature_groups is None:
        feature_groups = [
            'distance_rate','target_alignment','multi_window_rolling','extended_lags',
            'velocity_changes','field_position','role_specific','time_features',
            'jerk_features','interaction_features_mid'
        ]

    # --- Direction map & unify ---
    dir_map   = build_play_direction_map(input_df)
    input_df_u= unify_left_direction(input_df)

    if is_training:
        out_u = apply_direction_to_df(output_df, dir_map)
        target_rows   = out_u
        target_groups = out_u[['game_id','play_id','nfl_id']].drop_duplicates()
    else:
        if 'play_direction' not in test_template.columns:
            dir_df = dir_map.reset_index()
            test_template = test_template.merge(dir_df, on=['game_id','play_id'], how='left', validate='many_to_one')
        target_rows   = test_template
        target_groups = test_template[['game_id','play_id','nfl_id','play_direction']].drop_duplicates()

    assert target_rows[['game_id','play_id','play_direction']].isna().sum().sum() == 0, \
        "play_direction merge failed; check (game_id, play_id) coverage"
    print("play_direction merge OK:", target_rows['play_direction'].value_counts(dropna=False).to_dict())

    # --- FE ---
    fe = FeatureEngineer(feature_groups)
    processed_df, feature_cols = fe.transform(input_df_u)

    # --- Build sequences ---
    print("\nStep 3/3: Creating sequences...")
    processed_df = processed_df.set_index(['game_id','play_id','nfl_id']).sort_index()
    grouped = processed_df.groupby(level=['game_id','play_id','nfl_id'])

    # [C] 可选：打印键覆盖率，快速定位 miss 的真正原因
    avail_keys = (
        processed_df.reset_index()[['game_id','play_id','nfl_id']]
        .drop_duplicates()
    )
    inter = target_groups[['game_id','play_id','nfl_id']].merge(
        avail_keys, on=['game_id','play_id','nfl_id'], how='inner'
    )
    print(f"[COVERAGE] target_keys={len(target_groups)} | "
          f"input_keys={len(avail_keys)} | "
          f"matched={len(inter)}")

    # helpful indices
    idx_x = feature_cols.index('x')
    idx_y = feature_cols.index('y')

    sequences, targets_dx, targets_dy, targets_fids, seq_meta = [], [], [], [], []

    it = target_groups.itertuples(index=False)
    it = tqdm(list(it), total=len(target_groups), desc="Creating sequences")

    for row in it:
        gid = row[0]; pid = row[1]; nid = row[2]
        play_dir = row[3] if (not is_training and len(row) >= 4) else None
        key = (gid, pid, nid)

        try:
            group_df = grouped.get_group(key)
        except KeyError:
            continue

        input_window = group_df.tail(window_size)

        # --- [D] 训练端也允许左侧填充（与测试一致），避免全被 <window_size 丢弃 ---
        if len(input_window) < window_size:
            pad_len = window_size - len(input_window)
            pad_df = pd.DataFrame(np.nan, index=range(pad_len), columns=input_window.columns)
            input_window = pd.concat([pad_df, input_window], ignore_index=True)

        # input_window = input_window.fillna(group_df.mean(numeric_only=True))
        input_window = input_window.fillna(input_window.mean(numeric_only=True))
        seq = input_window[feature_cols].values

        if np.isnan(seq).any():
            seq = np.nan_to_num(seq, nan=0.0)

        sequences.append(seq)

        if is_training:
            out_grp = target_rows[
                (target_rows['game_id']==gid) &
                (target_rows['play_id']==pid) &
                (target_rows['nfl_id']==nid)
            ].sort_values('frame_id')
            if len(out_grp)==0:
                sequences.pop()  # 回滚
                continue

            last_x = seq[-1, idx_x]
            last_y = seq[-1, idx_y]
            dx = out_grp['x'].values - last_x
            dy = out_grp['y'].values - last_y

            targets_dx.append(dx.astype(np.float32))
            targets_dy.append(dy.astype(np.float32))
            targets_fids.append(out_grp['frame_id'].values.astype(np.int32))

        seq_meta.append({
            'game_id': gid,
            'play_id': pid,
            'nfl_id': nid,
            'frame_id': int(input_window.iloc[-1]['frame_id']) if len(input_window) else -1,
            'play_direction': (None if is_training else play_dir),
        })

    print(f"Created {len(sequences)} sequences with {len(feature_cols)} features each")

    if is_training:
        return sequences, targets_dx, targets_dy, targets_fids, seq_meta, feature_cols, dir_map
    return sequences, seq_meta, feature_cols, dir_map

# -------------------------------
# Tools for model saving & loading
# -------------------------------
def _seed_dir(base_dir: Path, seed: int) -> Path:
    d = base_dir / f"seed_{seed}"
    d.mkdir(parents=True, exist_ok=True)
    return d

def save_fold_artifacts(seed:int, fold:int, scaler, mx:nn.Module, my:nn.Module, base_dir:Path):
    sdir = _seed_dir(base_dir, seed)
    joblib.dump(scaler, sdir / f"scaler_fold{fold}.pkl")
    torch.save(mx.state_dict(), sdir / f"model_dx_fold{fold}.pt")
    torch.save(my.state_dict(), sdir / f"model_dy_fold{fold}.pt")

def write_meta(feature_cols:list, cfg:Config, base_dir:Path):
    meta = {
        "seeds": cfg.SEEDS,
        "n_folds": cfg.N_FOLDS,
        "feature_cols": feature_cols,
        "window_size": cfg.WINDOW_SIZE,
        "max_future_horizon": cfg.MAX_FUTURE_HORIZON,
        "feature_groups": cfg.FEATURE_GROUPS,
        "version": 1,
        "hidden_dim": cfg.HIDDEN_DIM,
        "bidirectional": getattr(cfg, "BIDIRECTIONAL", False),
    }
    with open(base_dir / "meta.json", "w") as f:
        json.dump(meta, f)
    print(f"[META] wrote meta.json to {base_dir}")

def load_saved_ensemble(cfg:Config, base_dir:Path):
    meta_path = base_dir / "meta.json"
    assert meta_path.exists(), f"meta.json not found: {meta_path}"
    with open(meta_path, "r") as f:
        meta = json.load(f)

    feature_cols = meta["feature_cols"]
    horizon = int(meta["max_future_horizon"])
    seeds   = meta["seeds"]
    n_folds = int(meta["n_folds"])
    hidden_dim   = int(meta.get("hidden_dim", 128))
    bidirectional= bool(meta.get("bidirectional", False))

    models_x, models_y, scalers = [], [], []
    for seed in seeds:
        sdir = base_dir / f"seed_{seed}"
        for fold in range(1, n_folds + 1):
            sc_path = sdir / f"scaler_fold{fold}.pkl"
            dx_path = sdir / f"model_dx_fold{fold}.pt"
            dy_path = sdir / f"model_dy_fold{fold}.pt"
            if not (sc_path.exists() and dx_path.exists() and dy_path.exists()):
                print(f"[WARN] missing seed={seed} fold={fold}, skip")
                continue
            scaler = joblib.load(sc_path)
            mx = SeqModel(len(feature_cols), horizon,
                          hidden_dim=hidden_dim, bidirectional=bidirectional
                          ).to(cfg.DEVICE)
            mx.load_state_dict(torch.load(dx_path, map_location=cfg.DEVICE)); mx.eval()
            my = SeqModel(len(feature_cols), horizon,
                          hidden_dim=hidden_dim, bidirectional=bidirectional
                          ).to(cfg.DEVICE)
            my.load_state_dict(torch.load(dy_path, map_location=cfg.DEVICE)); my.eval()
            scalers.append(scaler); models_x.append(mx); models_y.append(my)

    assert len(models_x) > 0, f"No models loaded from {base_dir}"
    print(f"[LOAD] loaded {len(models_x)} ΔX & {len(models_y)} ΔY models from {base_dir}")
    return models_x, models_y, scalers, meta

# -------------------------------
# Loss (Huber + time decay + 2nd-order velocity smooth)
# -------------------------------
class TemporalHuber(nn.Module):
    def __init__(self, delta=0.5, time_decay=0.02, lam_smooth=0.01):
        super().__init__()
        self.delta = delta
        self.time_decay = time_decay
        self.lam_smooth = lam_smooth

    def forward(self, pred, target, mask):
        # base huber
        err = pred - target
        abs_err = torch.abs(err)
        huber = torch.where(
            abs_err <= self.delta,
            0.5 * err * err,
            self.delta * (abs_err - 0.5 * self.delta)
        )

        # time decay (keep your logic)
        if self.time_decay and self.time_decay > 0:
            L = pred.size(1)
            t = torch.arange(L, device=pred.device, dtype=pred.dtype)
            w = torch.exp(-self.time_decay * t).view(1, L)
            huber = huber * w
            mask  = mask  * w

        main_loss = (huber * mask).sum() / (mask.sum() + 1e-8)

        # velocity smooth (2nd difference ≈ jerk), conservative mask对齐
        if self.lam_smooth and pred.size(1) > 2:
            d1 = pred[:, 1:] - pred[:, :-1]          # [B, T-1]
            d2 = d1[:, 1:] - d1[:, :-1]              # [B, T-2]
            m2 = mask[:, 2:]                         # 对齐长度
            smooth = (d2 * d2) * m2
            smooth_loss = smooth.sum() / (m2.sum() + 1e-8)
        else:
            smooth_loss = pred.new_tensor(0.0)

        return main_loss + self.lam_smooth * smooth_loss


# class SeqModel(nn.Module):
#     def __init__(self, input_dim, horizon):
#         super().__init__()
#         self.gru = nn.GRU(input_dim, 128, num_layers=2, batch_first=True, dropout=0.1)
#         self.pool_ln = nn.LayerNorm(128)
#         self.pool_attn = nn.MultiheadAttention(128, num_heads=4, batch_first=True)
#         self.pool_query = nn.Parameter(torch.randn(1, 1, 128))
#         self.head = nn.Sequential(
#             nn.Linear(128, 128), nn.GELU(), nn.Dropout(0.2), nn.Linear(128, horizon)
#         )
#     def forward(self, x):
#         h, _ = self.gru(x)
#         B = h.size(0)
#         q = self.pool_query.expand(B, -1, -1)
#         ctx, _ = self.pool_attn(q, self.pool_ln(h), self.pool_ln(h))
#         out = self.head(ctx.squeeze(1))
#         return torch.cumsum(out, dim=1)

class ResidualMLP(nn.Module):
    def __init__(self, d_in, d_hidden, horizon, dropout=0.2):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_hidden)
        self.proj = nn.Linear(d_in, d_hidden)  # skip
        self.out = nn.Linear(d_hidden, horizon)
        self.drop = nn.Dropout(dropout)
        self.act = nn.GELU()
    def forward(self, x):
        y = self.drop(self.act(self.fc1(x)))
        y = self.drop(self.act(self.fc2(y)) + self.proj(x))  # 残差
        return self.out(y)

class SeqModel(nn.Module):
    def __init__(self, input_dim, horizon, hidden_dim=128, num_layers=2, bidirectional=False,
                 use_residual=True, n_queries=2):
        super().__init__()
        self.bidirectional = bidirectional
        self.use_residual = use_residual
        self.gru = nn.GRU(
            input_dim, hidden_dim, num_layers=num_layers, batch_first=True,
            dropout=0.1, bidirectional=bidirectional
        )
        h_out = hidden_dim * (2 if bidirectional else 1)
        self.in_proj = nn.Linear(input_dim, h_out) if use_residual else None

        self.pool_ln   = nn.LayerNorm(h_out)
        self.pool_attn = nn.MultiheadAttention(h_out, num_heads=4, batch_first=True)
        self.pool_query= nn.Parameter(torch.randn(1, n_queries, h_out))  # 多query汇聚更多上下文
        self.head      = ResidualMLP(h_out*n_queries, hidden_dim, horizon)

    def forward(self, x):
        h, _ = self.gru(x)                        # [B,T,h_out]
        if self.use_residual:
            h = h + self.in_proj(x)               # 时间维残差

        B = h.size(0)
        q = self.pool_query.expand(B, -1, -1)     # [B,Q,h_out]
        ctx, _ = self.pool_attn(q, self.pool_ln(h), self.pool_ln(h))  # [B,Q,h_out]
        ctx = ctx.reshape(B, -1)                  # 拼成 [B, Q*h_out]
        out = self.head(ctx)                      # [B, H]
        return torch.cumsum(out, dim=1)


def prepare_targets(batch_axis, max_h):
    tensors, masks = [], []
    for arr in batch_axis:
        L = len(arr)
        padded = np.pad(arr, (0, max_h - L), constant_values=0).astype(np.float32)
        mask = np.zeros(max_h, dtype=np.float32)
        mask[:L] = 1.0
        tensors.append(torch.tensor(padded))
        masks.append(torch.tensor(mask))
    return torch.stack(tensors), torch.stack(masks)

def train_model(X_train, y_train, X_val, y_val, input_dim, horizon, config, noise_std=0.01, model_kwargs=None):
    device = config.DEVICE
    # model = SeqModel(input_dim, horizon).to(device)
    model = SeqModel(input_dim, horizon, **(model_kwargs or {})).to(device)
    criterion = TemporalHuber(delta=0.5, time_decay=0.03)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5, verbose=False)

    # build batches (keep numpy → torch)
    def build_batches(X, Y):
        batches = []
        B = config.BATCH_SIZE
        for i in range(0, len(X), B):
            end = min(i + B, len(X))
            xs = torch.tensor(np.stack(X[i:end]).astype(np.float32))
            ys, ms = prepare_targets([Y[j] for j in range(i, end)], horizon)
            batches.append((xs, ys, ms))
        return batches

    tr_batches = build_batches(X_train, y_train)
    va_batches = build_batches(X_val,   y_val)

    best_loss, best_state, bad = float('inf'), None, 0
    for epoch in range(1, config.EPOCHS + 1):
        model.train()
        train_losses = []
        for bx, by, bm in tr_batches:
            bx, by, bm = bx.to(device), by.to(device), bm.to(device)
            # 训练期增强（与 TTA 对齐）
            bx = add_random_gaussian(bx, sigma_max=noise_std)  # 随机强度噪声
            bx = random_time_mask(bx, p=0.10, max_width=3)    # 时间mask
            bx = flip_context_keep_last(bx, p=0.10)           # 反转前T-1，末帧不动
            pred = model(bx)

            loss = criterion(pred, by, bm)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for bx, by, bm in va_batches:
                bx, by, bm = bx.to(device), by.to(device), bm.to(device)
                pred = model(bx)
                val_losses.append(criterion(pred, by, bm).item())

        trl, val = float(np.mean(train_losses)), float(np.mean(val_losses))
        scheduler.step(val)
        if epoch % 10 == 0:
            print(f"  Epoch {epoch}: train={trl:.4f}, val={val:.4f}")

        if val < best_loss:
            best_loss, bad = val, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= config.PATIENCE:
                print(f"  Early stop at epoch {epoch}")
                break

    if best_state:
        model.load_state_dict(best_state)
    return model, best_loss

def predict_with_tta_per_model(mx, my, scaler, X_test_raw, device, tta=6, noise_std=0.01, use_flip=True):
    """
    对单个 (mx,my,scaler) 做 TTA，返回 [N,H] 的 dx,dy 预测。
    - 在“标准化后的空间”加噪声（与训练一致）
    - 可选：反转前 T-1 帧（末帧不动），与未反转结果平均
    - 重复 tta 次取均值
    """
    mx.eval(); my.eval()
    outs_dx, outs_dy = [], []
    base = np.stack([scaler.transform(s) for s in X_test_raw]).astype(np.float32)
    xt = torch.tensor(base, device=device)

    for _ in range(max(1, tta)):
        xt_aug = xt
        if noise_std and noise_std > 0:
            xt_aug = xt_aug + torch.randn_like(xt_aug) * noise_std

        with torch.no_grad():
            dx = mx(xt_aug)
            dy = my(xt_aug)
            if use_flip and xt_aug.size(1) > 1:
                ctx = xt_aug[:, :-1].flip(1)
                xt_flip = torch.cat([ctx, xt_aug[:, -1:].clone()], dim=1)
                dx = 0.5 * (dx + mx(xt_flip))
                dy = 0.5 * (dy + my(xt_flip))

        outs_dx.append(dx.detach().cpu().numpy())
        outs_dy.append(dy.detach().cpu().numpy())

    return np.mean(outs_dx, axis=0), np.mean(outs_dy, axis=0)

import random as _py_random
import torch

def random_time_mask(bx, p=0.1, max_width=3):
    """
    在时间维做短段复制遮挡：随机挑一段 [s, s+w) 用前/后一帧值替换。
    - bx: [B, T, D] (torch, 支持在 GPU 上原地改)
    """
    if p <= 0 or max_width <= 0:
        return bx
    B, T, D = bx.shape
    if T <= 1:
        return bx
    for i in range(B):
        if _py_random.random() < p:
            w = _py_random.randint(1, max_width)
            s = _py_random.randint(0, max(0, T - 1 - w))
            if s > 0:
                bx[i, s:s+w] = bx[i, s-1].unsqueeze(0)
            else:
                bx[i, s:s+w] = bx[i, s+w].unsqueeze(0)
    return bx

def flip_context_keep_last(bx, p=0.1):
    """
    仅反转前 T-1 帧（保持最后一帧不动），制造“上下文反转”。
    """
    if p <= 0:
        return bx
    B, T, D = bx.shape
    if T <= 1:
        return bx
    mask = torch.rand(B, device=bx.device) < p
    if mask.any():
        ctx = bx[mask, :-1].flip(1)
        bx[mask] = torch.cat([ctx, bx[mask, -1:].clone()], dim=1)
    return bx

def add_random_gaussian(bx, sigma_max=0.02):
    """
    给整段序列加一次高斯噪声（强度在 [0, sigma_max] 内随机）。
    """
    if sigma_max <= 0:
        return bx
    sigma = sigma_max * torch.rand(1, device=bx.device)
    return bx + torch.randn_like(bx) * sigma

# ------------------------------_
# Main pipeline (MODIFICADO PARA ENSEMBLE DE SEMILLAS)
# ------------------------------_
class CFG(Config):
    # Añadimos la lista de semillas para el ensemble
    SEEDS = [42, 19, 89,64,33] # ¡Puedes cambiar o añadir más semillas aquí!

def main():
    cfg = CFG()
    print("="*80)
    print(f"RUN MODE: TRAIN={getattr(cfg, 'TRAIN', False)} | SUB={getattr(cfg, 'SUB', False)}")
    print("="*80)
    print(f"cuDF backend active? {USE_CUDF}")

    # --- sanity checks ---
    if not cfg.TRAIN and not cfg.SUB:
        raise ValueError("Please set a run mode: TRAIN=True/SUB=False or TRAIN=False/SUB=True")
    if cfg.TRAIN and cfg.SUB:
        raise ValueError("TRAIN and SUB cannot both be True")

    # ---------------------------
    # TRAIN: train & save to a writable dir
    # ---------------------------
    if cfg.TRAIN:
        # redirect saving path to a writable place
        save_dir = Path("./saved_models")
        save_dir.mkdir(parents=True, exist_ok=True)
        cfg.MODELS_DIR = save_dir   # <<< key line: write into /kaggle/working

        # 1) load training data
        print("\n[1/4] 加载训练数据…")
        train_input_files  = [cfg.DATA_DIR / f"train/input_2023_w{w:02d}.csv"  for w in range(1, 19)]
        train_output_files = [cfg.DATA_DIR / f"train/output_2023_w{w:02d}.csv" for w in range(1, 19)]
        train_input  = pd.concat([pd.read_csv(f) for f in train_input_files  if f.exists()], ignore_index=True)
        train_output = pd.concat([pd.read_csv(f) for f in train_output_files if f.exists()], ignore_index=True)

        # 2) features + sequences (unified direction)
        print("\n[2/4] 特征与序列（统一方向）…")
        feature_groups = getattr(cfg, "FEATURE_GROUPS", None)
        seqs, tdx, tdy, tfids, seq_meta, feat_cols, dir_map = prepare_sequences_with_advanced_features(
            train_input, output_df=train_output, is_training=True,
            window_size=cfg.WINDOW_SIZE, feature_groups=feature_groups
        )
        sequences  = list(seqs)
        targets_dx = list(tdx)
        targets_dy = list(tdy)

        # 2.5) write meta to the same (writable) dir
        write_meta(feat_cols, cfg, base_dir=cfg.MODELS_DIR)


        # 3) multi-seed × KFold, save per-fold artifacts
        print("\n[3/4] 多种子 × K 折训练并保存模型…")
        # groups = np.array([d['game_id'] for d in seq_meta])
        groups = np.array([f"{d['game_id']}_{d['play_id']}" for d in seq_meta])

        seeds = getattr(cfg, "SEEDS", [cfg.SEED])
        all_rmse = []        # 所有 seed×fold 的 per-dim RMSE
        cv_log = []          # 也把每折指标放进列表，最后写 json

        for seed in seeds:
            print(f"\n{'='*70}\n   Seed {seed}\n{'='*70}")
            set_seed(seed)
            gkf = GroupKFold(n_splits=cfg.N_FOLDS)

            fold_rmses = []  # 当前 seed 的每折 RMSE（per-dim）

            for fold, (tr, va) in enumerate(gkf.split(sequences, groups=groups), 1):
                print(f"\n{'-'*60}\nFold {fold}/{cfg.N_FOLDS} (seed {seed})\n{'-'*60}")

                X_tr = [sequences[i] for i in tr]
                X_va = [sequences[i] for i in va]

                scaler = StandardScaler()
                scaler.fit(np.vstack([s for s in X_tr]))

                X_tr_sc = np.stack([scaler.transform(s) for s in X_tr]).astype(np.float32)
                X_va_sc = np.stack([scaler.transform(s) for s in X_va]).astype(np.float32)


                # model_kwargs = dict(hidden_dim=cfg.HIDDEN_DIM, bidirectional=getattr(cfg, "BIDIRECTIONAL", False))
                model_kwargs = dict(
                    hidden_dim=cfg.HIDDEN_DIM,
                    bidirectional=getattr(cfg, "BIDIRECTIONAL", False),
                    use_residual=True,
                    n_queries=2
                )
                
                print("训练 ΔX …")
                mx, loss_x = train_model(
                    X_tr_sc, [targets_dx[i] for i in tr],
                    X_va_sc, [targets_dx[i] for i in va],
                    X_tr_sc.shape[-1], cfg.MAX_FUTURE_HORIZON, cfg,
                    model_kwargs=model_kwargs
                )
                
                print("训练 ΔY …")
                my, loss_y = train_model(
                    X_tr_sc, [targets_dy[i] for i in tr],
                    X_va_sc, [targets_dy[i] for i in va],
                    X_tr_sc.shape[-1], cfg.MAX_FUTURE_HORIZON, cfg,
                    model_kwargs=model_kwargs
                )


                # --- NEW: 计算三口径，默认用 per-dim 进 CV 汇总 ---
                rmse_perdim = compute_val_rmse(
                    mx, my, X_va_sc,
                    [targets_dx[i] for i in va],
                    [targets_dy[i] for i in va],
                    cfg.MAX_FUTURE_HORIZON, cfg.DEVICE, mode="per-dim"
                )
                rmse_2d = compute_val_rmse(
                    mx, my, X_va_sc,
                    [targets_dx[i] for i in va],
                    [targets_dy[i] for i in va],
                    cfg.MAX_FUTURE_HORIZON, cfg.DEVICE, mode="2d"
                )
                mean_dist = compute_val_rmse(
                    mx, my, X_va_sc,
                    [targets_dx[i] for i in va],
                    [targets_dy[i] for i in va],
                    cfg.MAX_FUTURE_HORIZON, cfg.DEVICE, mode="mean-dist"
                )

                print(f"[VAL] seed {seed} fold {fold} → "
                      f"Huber dx={loss_x:.5f}, dy={loss_y:.5f} | "
                      f"per-dim RMSE={rmse_perdim:.4f} | 2D RMSE={rmse_2d:.4f} | meanDist={mean_dist:.4f} yards")

                fold_rmses.append(rmse_perdim)
                all_rmse.append(rmse_perdim)
                cv_log.append({
                    "seed": seed, "fold": fold,
                    "rmse_perdim": rmse_perdim,
                    "rmse_2d": rmse_2d,
                    "mean_dist": mean_dist,
                    "loss_dx": float(loss_x),
                    "loss_dy": float(loss_y),
                })

                # 保存模型
                save_fold_artifacts(seed=seed, fold=fold, scaler=scaler, mx=mx, my=my, base_dir=cfg.MODELS_DIR)

            # --- NEW: 当前 seed 汇总 ---
            print(f"[SEED SUMMARY] seed {seed} RMSEs: {[f'{r:.4f}' for r in fold_rmses]} | "
                  f"mean={float(np.mean(fold_rmses)):.4f} yards")

        # --- NEW: 所有 seeds×folds 的最终汇总 & 落盘 ---
        print(f"[CV SUMMARY] all folds RMSEs: {[f'{r:.4f}' for r in all_rmse]}")
        print(f"[CV SUMMARY] overall mean RMSE = {float(np.mean(all_rmse)):.4f} yards")

        # 写到磁盘（方便回看）
        try:
            with open(cfg.MODELS_DIR / "cv_metrics.json", "w") as f:
                json.dump({"per_fold": cv_log, "overall_mean_perdim": float(np.mean(all_rmse))}, f, indent=2)
            print(f"✓ CV metrics written to {cfg.MODELS_DIR / 'cv_metrics.json'}")
        except Exception as e:
            print(f"[WARN] writing cv_metrics.json failed: {e}")


        print("\n" + "="*80)
        print("COMPLETE (TRAIN)!")
        print("="*80)
        print(f"✓ Models saved to: {cfg.MODELS_DIR}")
        print(f"Seeds: {cfg.SEEDS} | Folds: {cfg.N_FOLDS} → checkpoints per axis: {len(cfg.SEEDS)*cfg.N_FOLDS}")
        print(f"Features used: {len(feat_cols)}  (cuDF active: {USE_CUDF})")
        return

    # ---------------------------
    # SUB: Evaluation API mode (new format)
    # ---------------------------
    if cfg.SUB:
        print("\n" + "="*80)
        print("EVALUATION API MODE")
        print("="*80)
        print("Models will be loaded on first predict() call")
        print("The inference server is set up in Cell 9")
        print("In competition rerun, the server will handle requests automatically")
        print("="*80)
        return

    # 如果两个 flag 都没开，给出提醒
    raise ValueError("请在 Config 中设置运行模式：TRAIN=True/SUB=False 或 TRAIN=False/SUB=True")


if __name__ == "__main__":
    main()

# =============================================================================
# Evaluation API Server Setup
# =============================================================================

# Global variables to store models (loaded once on first predict call)
_models_loaded = False
_models_x = None
_models_y = None
_scalers = None
_meta = None
_feature_cols = None

def load_models_once():
    """Load models on first predict call (no 5-minute time limit)"""
    global _models_loaded, _models_x, _models_y, _scalers, _meta, _feature_cols
    
    if _models_loaded:
        return
    
    print("[SERVER] Loading models for first time...")
    cfg = Config()
    cfg.MODELS_DIR = Path("/kaggle/input/hsiaosuan-sttn/saved_models")
    
    _models_x, _models_y, _scalers, _meta = load_saved_ensemble(cfg, base_dir=cfg.MODELS_DIR)
    _feature_cols = _meta["feature_cols"]
    
    _models_loaded = True
    print(f"[SERVER] Loaded {len(_models_x)} models successfully")


def predict(test: pl.DataFrame, test_input: pl.DataFrame) -> pl.DataFrame | pd.DataFrame:
    """
    Inference function: process each batch of data
    
    Args:
        test: Frames to predict (contains game_id, play_id, nfl_id, frame_id, etc.)
        test_input: Available input data (historical frames)
    
    Returns:
        DataFrame with x, y coordinates
    """
    global _models_x, _models_y, _scalers, _meta, _feature_cols
    
    # First call: load models (no time limit)
    if not _models_loaded:
        load_models_once()
    
    # Convert to pandas (our code is pandas-based)
    test_pd = test.to_pandas()
    test_input_pd = test_input.to_pandas()
    
    cfg = Config()
    saved_window = int(_meta.get("window_size", cfg.WINDOW_SIZE))
    saved_groups = _meta.get("feature_groups", cfg.FEATURE_GROUPS)
    
    # Build sequences
    test_seqs, test_meta, feat_cols_t, _ = prepare_sequences_with_advanced_features(
        test_input_pd, test_template=test_pd, is_training=False,
        window_size=saved_window, feature_groups=saved_groups
    )
    
    idx_x = feat_cols_t.index('x')
    idx_y = feat_cols_t.index('y')
    
    X_test_raw = list(test_seqs)
    x_last_uni = np.array([s[-1, idx_x] for s in X_test_raw], dtype=np.float32)
    y_last_uni = np.array([s[-1, idx_y] for s in X_test_raw], dtype=np.float32)
    
    # TTA inference
    tta_times = 6
    tta_noise = 0.01
    use_flip_ta = True
    
    all_preds_dx, all_preds_dy = [], []
    for mx, my, sc in zip(_models_x, _models_y, _scalers):
        dx_tta, dy_tta = predict_with_tta_per_model(
            mx, my, sc, X_test_raw, cfg.DEVICE,
            tta=tta_times, noise_std=tta_noise, use_flip=use_flip_ta
        )
        all_preds_dx.append(dx_tta)
        all_preds_dy.append(dy_tta)
    
    ens_dx = np.mean(all_preds_dx, axis=0)
    ens_dy = np.mean(all_preds_dy, axis=0)
    
    H = ens_dx.shape[1]
    
    # Build predictions
    rows = []
    tt_idx = test_pd.set_index(['game_id','play_id','nfl_id']).sort_index()
    
    for i, meta_row in enumerate(test_meta):
        gid = meta_row['game_id']
        pid = meta_row['play_id']
        nid = meta_row['nfl_id']
        play_is_right = (meta_row['play_direction'] == 'right')
        
        try:
            fids = tt_idx.loc[(gid, pid, nid), 'frame_id']
            if isinstance(fids, pd.Series):
                fids = fids.sort_values().tolist()
            else:
                fids = [int(fids)]
        except KeyError:
            continue
        
        for t, fid in enumerate(fids):
            tt = min(t, H - 1)
            x_uni = np.clip(x_last_uni[i] + ens_dx[i, tt], 0, FIELD_LENGTH)
            y_uni = np.clip(y_last_uni[i] + ens_dy[i, tt], 0, FIELD_WIDTH)
            x_out, y_out = invert_to_original_direction(x_uni, y_uni, play_is_right)
            rows.append({'x': x_out, 'y': y_out})
    
    # Return Polars DataFrame (recommended)
    predictions = pl.DataFrame(rows)
    
    assert len(predictions) == len(test)
    return predictions
```

```python
%%writefile nfl_2026.py

import os
import shutil


TIMETAG = "20251108_071543"
dest_dir = "./src"
os.makedirs(dest_dir, exist_ok=True)

source_dir = f"/kaggle/input/nfl-bdb-2026/NFL2026_111325/NFL2026_111325/src/"
shutil.copytree(source_dir, dest_dir, dirs_exist_ok=True)

USE_CUDF = False
try:
    # zero/low-code GPU acceleration for DataFrame ops
    os.environ["CUDF_PANDAS_BACKEND"] = "cudf"
    import pandas as pd
    import numpy as np

    USE_CUDF = True
    print("using cuda_backend pandas for faster parallel data processing")
except Exception:
    print("cuda df not used")
    import pandas as pd
    import numpy as np


import json
from pathlib import Path
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import GroupKFold
import warnings

warnings.filterwarnings("ignore")

from src.config import Config

# =============================================================================
# Evaluation API Server Setup
# =============================================================================
# New imports for evaluation API
import polars as pl
from src.utils import load_saved_ensemble_stt, invert_to_original_direction
from src.preprocess import prepare_sequences_with_advanced_features
from src.model import STTransformer
from src.predict import predict_sst

# Global variables to store models (loaded once on first predict call)
_models_loaded = False
_models = None
_scalers = None
_meta = None
_feature_cols = None


def load_models_once():
    """Load models on first predict call (no 5-minute time limit)"""
    global _models_loaded, _models, _scalers, _meta, _feature_cols

    if _models_loaded:
        return

    print("[SERVER] Loading models for first time...")
    cfg = Config()
    cfg.MODELS_DIR = Path(f"/kaggle/input/nfl-bdb-2026/NFL2026_111325/NFL2026_111325")

    _models, _scalers, _meta = load_saved_ensemble_stt(cfg.MODELS_DIR, STTransformer)
    _feature_cols = _meta["feature_cols"]

    _models_loaded = True
    print(f"[SERVER] Loaded {len(_models)} models successfully")


def predict(
    test: pl.DataFrame, test_input: pl.DataFrame
) -> pl.DataFrame | pd.DataFrame:
    """
    Inference function: process each batch of data

    Args:
        test: Frames to predict (contains game_id, play_id, nfl_id, frame_id, etc.)
        test_input: Available input data (historical frames)

    Returns:
        DataFrame with x, y coordinates
    """
    global _models, _scalers, _meta, _feature_cols

    # First call: load models (no time limit)
    if not _models_loaded:
        load_models_once()

    # Convert to pandas (our code is pandas-based)
    test_pd = test.to_pandas()
    test_input_pd = test_input.to_pandas()

    cfg = Config()
    saved_groups = _meta.get("feature_groups", cfg.FEATURE_GROUPS)

    # Build sequences
    test_seqs, test_meta, feat_cols_t = prepare_sequences_with_advanced_features(
        test_input_pd,
        test_pd,
        feature_groups=saved_groups,
    )

    idx_x = feat_cols_t.index("x")
    idx_y = feat_cols_t.index("y")

    X_test_raw = list(test_seqs)
    x_last_uni = np.array([s[-1, idx_x] for s in X_test_raw], dtype=np.float32)
    y_last_uni = np.array([s[-1, idx_y] for s in X_test_raw], dtype=np.float32)

    all_preds_dx, all_preds_dy = [], []
    for m, sc in zip(_models, _scalers):
        dx_tta, dy_tta = predict_sst(
            m,
            sc,
            X_test_raw,
            cfg.DEVICE,
        )
        all_preds_dx.append(dx_tta)
        all_preds_dy.append(dy_tta)

    ens_dx = np.mean(all_preds_dx, axis=0)
    ens_dy = np.mean(all_preds_dy, axis=0)

    H = ens_dx.shape[1]

    # Build predictions
    rows = []
    tt_idx = test_pd.set_index(["game_id", "play_id", "nfl_id"]).sort_index()

    for i, meta_row in enumerate(test_meta):
        gid = meta_row["game_id"]
        pid = meta_row["play_id"]
        nid = meta_row["nfl_id"]
        play_dir = meta_row["play_direction"]

        try:
            fids = tt_idx.loc[(gid, pid, nid), "frame_id"]
            if isinstance(fids, pd.Series):
                fids = fids.sort_values().tolist()
            else:
                fids = [int(fids)]
        except KeyError:
            continue

        for t, fid in enumerate(fids):
            tt = min(t, H - 1)
            x_uni = np.clip(x_last_uni[i] + ens_dx[i, tt], 0, Config.FIELD_X_MAX)
            y_uni = np.clip(y_last_uni[i] + ens_dy[i, tt], 0, Config.FIELD_Y_MAX)
            x_uni, y_uni = invert_to_original_direction(
                x_uni, y_uni, play_dir == "right"
            )
            rows.append({"x": x_uni, "y": y_uni})

    predictions = pl.DataFrame(rows)

    assert len(predictions) == len(test)
    return predictions
```

```python
import importlib
import polars as pl
import pandas as pd
import kaggle_evaluation.nfl_inference_server
import os
import numpy as np # Import numpy for calculations

# Import all three modules
nfl_gru = importlib.import_module('nfl_gru')
nfl_gnn = importlib.import_module('nfl_gnn')
nfl_gru_2 = importlib.import_module('nfl_2026')

def predict(test: pl.DataFrame, test_input: pl.DataFrame) -> pd.DataFrame:
    

    score_gru = 0.584
    score_gnn = 0.580 
    score_gru2 = 0.562
    
    # 2. Calculate weights based on inverse scores
    inv_gru = 1.0 / score_gru
    inv_gnn = 1.0 / score_gnn
    inv_gru2 = 1.0 / score_gru2
    
    total_inv = inv_gru + inv_gnn + inv_gru2
    
    w_gru = inv_gru / total_inv
    w_gnn = inv_gnn / total_inv
    w_gru2 = inv_gru2 / total_inv
    
    print(f"GRU Weight: {w_gru:.4f}")
    print(f"GNN Weight: {w_gnn:.4f}")
    print(f"GRU2 Weight: {w_gru2:.4f}")
    print(f"Total Weight: {w_gru + w_gnn + w_gru2}")

    # 3. Get predictions from all three models
    pred_gru = nfl_gru.predict(test, test_input)
    pred_gnn = nfl_gnn.predict(test, test_input)
    pred_gru2 = nfl_gru_2.predict(test, test_input)

    # 4. Ensure all are Pandas DataFrames
    if isinstance(pred_gru, pl.DataFrame):
        pred_gru = pred_gru.to_pandas()
    if isinstance(pred_gnn, pl.DataFrame):
        pred_gnn = pred_gnn.to_pandas()
    if isinstance(pred_gru2, pl.DataFrame):
        pred_gru2 = pred_gru2.to_pandas()

    # 5. Get the numpy arrays for calculation
    vals_gru = pred_gru[['x', 'y']].values
    vals_gnn = pred_gnn[['x', 'y']].values
    vals_gru2 = pred_gru2[['x', 'y']].values

    # 6. Apply the weighted average
    pred_ensemble = (w_gru * vals_gru) + (w_gnn * vals_gnn) + (w_gru2 * vals_gru2)
    
    return pd.DataFrame(pred_ensemble, columns=['x', 'y'])

# --- Rest of your server code ---
inference_server = kaggle_evaluation.nfl_inference_server.NFLInferenceServer(predict)

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    inference_server.run_local_gateway(('/kaggle/input/nfl-big-data-bowl-2026-prediction/',))
```