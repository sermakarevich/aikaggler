# 🏀 March ML Mania 2026 | HistGB + XGB + CatBoost

- **Author:** Emanuel Lázaro
- **Votes:** 119
- **Ref:** emanuellcs/march-ml-mania-2026-histgb-xgb-catboost
- **URL:** https://www.kaggle.com/code/emanuellcs/march-ml-mania-2026-histgb-xgb-catboost
- **Last run:** 2026-03-18 22:45:55.743000

---

# 🏀 March Machine Learning Mania 2026 | Specialized Ensemble (HistGB + XGBoost + CatBoost)

The goal of this notebook is to predict the win probability for every possible NCAA tournament matchup (Men's and Women's) by minimizing **Brier Score**.

---

### 💾 Data Loading

We load both compact and detailed game results, Massey ordinals (dynamically concatenating any supplementary system files as they are released), seeds, and team rosters for both tournaments. Detailed results are the most valuable source because they unlock the **Four Factors**, which carry the strongest predictive signal in the feature set. Compact results serve as a fallback wherever detailed data is unavailable.

---

### ⚙️ Feature Engineering

All features are computed at the team-season level and then turned into matchup-level differences (Team 1 minus Team 2). The feature groups are:

- **Compact stats (baseline)** -> Win rate, average scoring margin, average points for and against. These serve as a reliable fallback when detailed box scores are missing.

- **Dean Oliver's Four Factors + efficiency** -> From detailed results we compute possessions using the standard 0.44 multiplier, then derive Offensive Rating (ORTG), Defensive Rating (DRTG), Net Rating, effective FG%, turnover rate, offensive rebounding rate, and free-throw rate. These are the strongest single signal group in the feature set.

- **Strength of Schedule (SOS)** -> Each team's average opponent win rate over the regular season, giving context to how hard-earned their stats are.

- **Rolling form** -> Win rate and average margin over the final 10 regular-season games, capturing late-season momentum that season averages can miss.

- **Massey ordinal rankings** -> We aggregate all available ranking systems into mean, median, standard deviation, best, and worst rank per team-season. We also pivot individual systems (POM, SAG, MOR, RTH, WLK, DOK, COL, AP) as separate features to give the model access to varied ranking philosophies.

- **Elo ratings** -> Game-by-game Elo updated within each season, with a home-court adjustment, a log-damped margin-of-victory multiplier to reward dominant wins proportionally, and a 75/25 regression to the mean between seasons.

- **Historical seed upset probabilities** -> A lookup table of how often each seed matchup has historically ended in an upset, built from all past tournament results. Matchups with fewer than 5 observed seasons are excluded to avoid noise.

- **Interaction features** -> `SeedDiff x EloDiff` and `SeedDiff²` are added to the feature matrix to help the model capture non-linear effects where seed gaps compound with Elo gaps.

---

### ⏱️ Temporal Walk-Forward Cross-Validation

We validate across seasons 2018 to 2025 using strict temporal splitting: for each validation season `v`, the model is trained only on seasons before `v` and evaluated on season `v`. This mirrors the actual competition setup and prevents data leakage from future seasons, which would inflate CV scores with a standard stratified k-fold approach.

---

### 🎛️ Hyperparameter Tuning with Optuna

Optuna's TPE sampler runs 40 trials to tune HistGradientBoosting on the most recent held-out season (2025). XGBoost and CatBoost use fixed, well-tested defaults (600 estimators, learning rate 0.03). Separating HPO from the full CV loop keeps tuning honest by always searching on genuinely unseen data.

---

### ⚔️ Breaking the Correlation: Model Specialization

In earlier iterations of this pipeline, the three tree-based models exhibited extreme correlation (Pearson r > 0.99) because they were learning the exact same patterns from the full feature set. To force ensemble diversity and make stacking truly effective, the feature space is now strictly partitioned:

- **XGBoost (The Statistician)** -> Trained *only* on pure on-court performance metrics (Four Factors, Elo, NetRtg, Margins).
- **CatBoost (The Pollster)** -> Trained *only* on external opinions and ranking systems (Massey Ordinals, KenPom, Sagarin).
- **HistGB (The Generalist)** -> Trained on the entire feature set.

This strict feature isolation drops inter-model correlation to ~0.83, providing the meta-learner with genuinely distinct and uncorrelated perspectives to blend.

---

### 📚 Meta-Learner Stacking

Rather than blending with fixed weights, we train a **Logistic Regression meta-learner** on the out-of-fold (OOF) predictions from all three specialized base models. Each model's probability is first converted to log-odds to create a more linear input space. The `SeedDiff` feature is also passed to the meta-learner so it can learn context-aware blending—for example, heavily trusting the "Statistician" (XGBoost) for Women's games, while balancing the "Pollster" (CatBoost) and "Generalist" (HistGB) for chaotic Men's matchups. L2 regularization (C=0.1) prevents overfitting the training OOF.

---

### 🎯 Isotonic Calibration

After stacking, the blended OOF probabilities are passed through an **Isotonic Regression calibrator** to correct systematic over- or under-confidence. The calibrator is fit on OOF predictions only, which keeps it honest and prevents it from artificially tightening the probability distribution.

---

### 🏁 Final Training and Inference

All three base models are retrained on the full available training set using their respective feature subsets. During inference, features are built in parallel across all CPU cores and the three base models predict concurrently via `ThreadPoolExecutor`. Their log-odds outputs are stacked with `SeedDiff`, scaled, passed through the meta-learner, and finally calibrated.

```python
import os, warnings
from pathlib import Path
from functools import partial
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from tqdm.auto import tqdm
from joblib import Parallel, delayed

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import calibration_curve
from sklearn.metrics import log_loss, brier_score_loss
from sklearn.ensemble import HistGradientBoostingClassifier

import xgboost as xgb
from catboost import CatBoostClassifier

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    print("Optuna not found - falling back to tuned defaults")

from scipy.optimize import minimize

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 60)

plt.rcParams.update({
    'figure.dpi': 120,
    'font.family': 'DejaVu Sans',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'axes.titleweight': 'bold',
})
C_MEN   = '#2c7bb6'
C_WOMEN = '#d7191c'
C_GRAY  = '#636363'

INPUT_ROOT = Path('/kaggle/input')
DATA_DIR = None
for p in INPUT_ROOT.rglob('MTeams.csv'):
    DATA_DIR = p.parent
    break
if DATA_DIR is None:
    raise FileNotFoundError(f'MTeams.csv not found under {INPUT_ROOT}')

SEED = 42
np.random.seed(SEED)

# Walk-forward CV: validate on each of these seasons individually
# Train = all prior tournament seasons; Validate = this season
VAL_SEASONS = list(range(2018, 2026))
```

# Data Loading
We load **both** compact and detailed results, all Massey ordinals, seeds, and teams.
Detailed results unlock the Four Factors which are the single biggest feature signal.

```python
# -- Teams & Seeds ----------------------------------------------------------
m_teams  = pd.read_csv(DATA_DIR / 'MTeams.csv')
m_seeds  = pd.read_csv(DATA_DIR / 'MNCAATourneySeeds.csv')
w_teams  = pd.read_csv(DATA_DIR / 'WTeams.csv')
w_seeds  = pd.read_csv(DATA_DIR / 'WNCAATourneySeeds.csv')

def parse_seed(s: str) -> int:
    return int(''.join(filter(str.isdigit, s)))

m_seeds['SeedNum'] = m_seeds['Seed'].apply(parse_seed)
w_seeds['SeedNum'] = w_seeds['Seed'].apply(parse_seed)

# -- Compact results --------------------------------------------------------
m_tourney = pd.read_csv(DATA_DIR / 'MNCAATourneyCompactResults.csv')
w_tourney = pd.read_csv(DATA_DIR / 'WNCAATourneyCompactResults.csv')
m_reg_c   = pd.read_csv(DATA_DIR / 'MRegularSeasonCompactResults.csv')
w_reg_c   = pd.read_csv(DATA_DIR / 'WRegularSeasonCompactResults.csv')

# -- Detailed results (if available) ---------------------------------------
def try_load(fname):
    p = DATA_DIR / fname
    return pd.read_csv(p) if p.exists() else None

m_reg_d   = try_load('MRegularSeasonDetailedResults.csv')
w_reg_d   = try_load('WRegularSeasonDetailedResults.csv')
m_tour_d  = try_load('MNCAATourneyDetailedResults.csv')
w_tour_d  = try_load('WNCAATourneyDetailedResults.csv')

print("Detailed regular-season rows  M:", len(m_reg_d) if m_reg_d is not None else "N/A",
      " W:", len(w_reg_d) if w_reg_d is not None else "N/A")

# -- Massey Ordinals --------------------------------------------------------
# 1. Load the main file
m_massey = pd.read_csv(DATA_DIR / 'MMasseyOrdinals.csv')

#2. Upload and attach the additional files
supp_files = glob.glob('/kaggle/input/datasets/emanuellcs/march-mania-2026-massey-supp/*.csv')
if supp_files:
    supp_dfs = []
    for f in supp_files:
        file_name = os.path.basename(f)
        print(f" -> Reading supplementary file: {file_name}")
        supp_dfs.append(pd.read_csv(f))
    m_massey = pd.concat([m_massey] + supp_dfs, ignore_index=True)
    print(f"Total: {len(supp_files)} supplementary file(s) successfully added!")

# Women's Massey if it exists
w_massey = try_load('WMasseyOrdinals.csv')

# -- Submission -------------------------------------------------------------
sub = pd.read_csv(DATA_DIR / 'SampleSubmissionStage2.csv')
sub['Season'] = sub['ID'].apply(lambda x: int(x.split('_')[0]))
sub['T1']     = sub['ID'].apply(lambda x: int(x.split('_')[1]))
sub['T2']     = sub['ID'].apply(lambda x: int(x.split('_')[2]))

m_team_ids = set(m_teams['TeamID'])
w_team_ids = set(w_teams['TeamID'])
m_sub = sub[sub['T1'].isin(m_team_ids)].copy().reset_index(drop=True)
w_sub = sub[sub['T1'].isin(w_team_ids)].copy().reset_index(drop=True)
print(f"Submission rows: Men={len(m_sub)}  Women={len(w_sub)}")
```

# Feature Engineering

```python
# BLOCK A: Season-level stats from compact results (fallback)

def compact_season_stats(reg_df: pd.DataFrame) -> pd.DataFrame:
    """Win%, average margin and average score per team-season."""
    records = []
    for side, opp_col, score_col, opp_score_col, win_val in [
        ('W', 'LTeamID', 'WScore', 'LScore', 1),
        ('L', 'WTeamID', 'LScore', 'WScore', 0)
    ]:
        tmp = reg_df[['Season', f'{side}TeamID', score_col, opp_score_col, 'WLoc']].copy()
        tmp.columns = ['Season', 'TeamID', 'ScoreFor', 'ScoreAgainst', 'Loc']
        tmp['Win']    = win_val
        tmp['Margin'] = tmp['ScoreFor'] - tmp['ScoreAgainst']
        records.append(tmp)
    df = pd.concat(records, ignore_index=True)
    stats = df.groupby(['Season', 'TeamID']).agg(
        Games     = ('Win',    'count'),
        Wins      = ('Win',    'sum'),
        AvgScore  = ('ScoreFor',  'mean'),
        AvgOppScore=('ScoreAgainst','mean'),
        AvgMargin = ('Margin', 'mean'),
        StdMargin = ('Margin', 'std'),
    ).reset_index()
    stats['WinRate'] = stats['Wins'] / stats['Games']
    stats['StdMargin'] = stats['StdMargin'].fillna(0)
    return stats

m_cstats = compact_season_stats(m_reg_c)
w_cstats = compact_season_stats(w_reg_c)

# BLOCK B: Four Factors + Efficiency from Detailed Results

POSS_K = 0.44   # standard Dean Oliver constant

def compute_possessions(fga, orb, tov, fta):
    """Estimate possessions per game."""
    return fga - orb + tov + POSS_K * fta

def four_factors_season_stats(det_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Dean Oliver's Four Factors + efficiency metrics per team-season.
    Columns added: Poss, ORTG, DRTG, NetRtg, eFGpct, TOpct, ORBpct, FTR,
                   ThreePAr, ASTpct (where data available).
    """
    records = []
    for side, other in [('W', 'L'), ('L', 'W')]:
        t = det_df.copy()
        t['TeamID']  = t[f'{side}TeamID']
        t['OppID']   = t[f'{other}TeamID']
        t['Pts']     = t[f'{side}Score']
        t['OppPts']  = t[f'{other}Score']
        for stat in ['FGM','FGA','FGM3','FGA3','FTM','FTA','OR','DR','Ast','TO','Stl','Blk','PF']:
            t[stat]      = t.get(f'{side}{stat}', np.nan)
            t[f'Opp{stat}'] = t.get(f'{other}{stat}', np.nan)
        t['Win'] = (side == 'W').real  # 1 if W-side, 0 if L-side
        records.append(t[['Season','TeamID','OppID','Pts','OppPts',
                           'FGM','FGA','FGM3','FGA3','FTM','FTA',
                           'OR','DR','Ast','TO',
                           'OppFGM','OppFGA','OppFGM3','OppFGA3',
                           'OppOR','OppDR','OppTO','OppFTA','Win']])
    df = pd.concat(records, ignore_index=True)

    df['Poss']    = compute_possessions(df['FGA'],  df['OR'],  df['TO'],  df['FTA'])
    df['OppPoss'] = compute_possessions(df['OppFGA'],df['OppOR'],df['OppTO'],df['OppFTA'])
    df['Poss']    = df['Poss'].clip(lower=1)
    df['OppPoss'] = df['OppPoss'].clip(lower=1)

    df['ORTG']  = 100 * df['Pts']    / df['Poss']
    df['DRTG']  = 100 * df['OppPts'] / df['OppPoss']
    df['NetRtg'] = df['ORTG'] - df['DRTG']

    df['eFGpct'] = (df['FGM'] + 0.5 * df['FGM3']) / df['FGA'].clip(lower=1)
    df['TOpct']  = df['TO']  / df['Poss']
    df['ORBpct'] = df['OR']  / (df['OR'] + df['OppDR']).clip(lower=1)
    df['FTR']    = df['FTM'] / df['FGA'].clip(lower=1)
    df['ThreePAr']= df['FGA3'] / df['FGA'].clip(lower=1)
    df['ASTpct'] = df['Ast'] / df['FGM'].clip(lower=1)

    agg_cols = ['ORTG','DRTG','NetRtg','eFGpct','TOpct','ORBpct','FTR','ThreePAr','ASTpct','Poss']
    stats = df.groupby(['Season','TeamID'])[agg_cols].mean().reset_index()
    return stats

m_ff = four_factors_season_stats(m_reg_d) if m_reg_d is not None else None
w_ff = four_factors_season_stats(w_reg_d) if w_reg_d is not None else None
print("Four Factors computed -", "Men:", m_ff.shape if m_ff is not None else "N/A",
      "Women:", w_ff.shape if w_ff is not None else "N/A")

# BLOCK C: Strength of Schedule (SOS)

def compute_sos(reg_df: pd.DataFrame, base_stats: pd.DataFrame) -> pd.DataFrame:
    """Average opponent WinRate as a proxy for SOS."""
    win_rate = base_stats.set_index(['Season','TeamID'])['WinRate']
    records = []
    for side, other in [('W','L'),('L','W')]:
        t = reg_df[['Season',f'{side}TeamID',f'{other}TeamID']].copy()
        t.columns = ['Season','TeamID','OppID']
        records.append(t)
    df = pd.concat(records, ignore_index=True)
    df['OppWR'] = df.apply(
        lambda r: win_rate.get((r['Season'], r['OppID']), np.nan), axis=1
    )
    sos = df.groupby(['Season','TeamID'])['OppWR'].mean().reset_index()
    sos.columns = ['Season','TeamID','SOS']
    return sos

m_sos = compute_sos(m_reg_c, m_cstats)
w_sos = compute_sos(w_reg_c, w_cstats)

# BLOCK D: Rolling / Recent Form (last-N games)

def rolling_form(reg_df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Win% and avg margin in last-N regular-season games per team-season."""
    records = []
    for side, other, win_val in [('W','L',1),('L','W',0)]:
        t = reg_df[['Season','DayNum',f'{side}TeamID','WScore','LScore']].copy()
        t.columns = ['Season','DayNum','TeamID','ScoreFor','ScoreAgainst']
        t['Win']    = win_val
        t['Margin'] = (t['ScoreFor'] - t['ScoreAgainst']) * (1 if win_val==1 else -1)
        records.append(t)
    df = pd.concat(records, ignore_index=True).sort_values(['Season','TeamID','DayNum'])
    result = (
        df.groupby(['Season','TeamID'])
          .apply(lambda g: pd.Series({
              f'Last{n}WinRate': g['Win'].iloc[-n:].mean(),
              f'Last{n}Margin':  g['Margin'].iloc[-n:].mean(),
          }))
          .reset_index()
    )
    return result

m_form = rolling_form(m_reg_c, n=10)
w_form = rolling_form(w_reg_c, n=10)

# BLOCK E: All Massey Systems Aggregated

def massey_features(massey_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each team-season, aggregate across all ranking systems:
    mean, median, std, min, max rank; also last-day rank per system.
    Then pivot so each system's last-day rank is its own column.
    """
    if massey_df is None:
        return pd.DataFrame(columns=['Season','TeamID'])
    # Last-day snapshot per team-season-system
    last = (massey_df.sort_values('RankingDayNum')
                     .groupby(['Season','TeamID','SystemName'])
                     .last().reset_index()
                     [['Season','TeamID','SystemName','OrdinalRank']])
    # Aggregate across systems
    agg = last.groupby(['Season','TeamID'])['OrdinalRank'].agg(
        MasseyMeanRank='mean',
        MasseyMedRank='median',
        MasseyStdRank='std',
        MasseyBestRank='min',
        MasseyWorstRank='max',
    ).reset_index()
    agg['MasseyStdRank'] = agg['MasseyStdRank'].fillna(0)
    agg['MasseyRankSpread'] = agg['MasseyWorstRank'] - agg['MasseyBestRank']

    # Pivot top individual systems (POM, SAG, MOR, RTH, WLK - keep if present)
    key_systems = ['POM','SAG','MOR','RTH','WLK','DOK','COL','AP']
    pivot = last[last['SystemName'].isin(key_systems)].pivot_table(
        index=['Season','TeamID'], columns='SystemName', values='OrdinalRank'
    ).reset_index()
    pivot.columns = ['Season','TeamID'] + [f'Rank_{c}' for c in pivot.columns[2:]]

    result = agg.merge(pivot, on=['Season','TeamID'], how='left')
    return result

m_massey_feats = massey_features(m_massey)
w_massey_feats = massey_features(w_massey)
print("Massey features:", m_massey_feats.columns.tolist())

# BLOCK F: Season Elo Ratings

def compute_elo(reg_df: pd.DataFrame, K: float = 30,
                home_adv: float = 100, initial: float = 1500) -> pd.DataFrame:
    """
    Simple Elo updated game-by-game within each season.
    Returns final season Elo per team-season.
    Uses margin-of-victory multiplier (log-damped).
    """
    elo = {}   # (season, team) -> elo value

    def get_elo(season, team):
        if (season, team) not in elo:
            # Carry over from prior season with regression to mean
            prev = elo.get((season - 1, team), initial)
            elo[(season, team)] = prev * 0.75 + initial * 0.25
        return elo[(season, team)]

    rows = reg_df.sort_values(['Season', 'DayNum']).itertuples(index=False)
    for r in rows:
        s  = r.Season
        w  = r.WTeamID
        l  = r.LTeamID
        ew = get_elo(s, w)
        el = get_elo(s, l)
        # Location adjustment
        loc = getattr(r, 'WLoc', 'N')
        if loc == 'H':
            ew_adj, el_adj = ew + home_adv, el
        elif loc == 'A':
            ew_adj, el_adj = ew, el + home_adv
        else:
            ew_adj, el_adj = ew, el
        Ew = 1 / (1 + 10 ** ((el_adj - ew_adj) / 400))
        # Margin-of-victory multiplier
        margin = abs(r.WScore - r.LScore)
        mov_mult = np.log1p(margin) / np.log1p(10)
        k_eff = K * mov_mult
        elo[(s, w)] = ew + k_eff * (1 - Ew)
        elo[(s, l)] = el + k_eff * (0 - (1 - Ew))

    records = [{'Season': s, 'TeamID': t, 'Elo': v}
               for (s, t), v in elo.items()]
    return pd.DataFrame(records)

print("Computing Elo for Men's...")
m_elo = compute_elo(m_reg_c)
print("Computing Elo for Women's...")
w_elo = compute_elo(w_reg_c)

# BLOCK G: Historical seed performance (upset probabilities)

def historical_seed_upset_probs(tourney_df, seeds_df, min_seasons: int = 5) -> pd.DataFrame:
    """
    Win probability of lower seed number (better seed) vs higher seed number,
    aggregated over historical seasons.  Returns a seed1 x seed2 lookup table.
    """
    df = tourney_df.merge(
        seeds_df[['Season','TeamID','SeedNum']].rename(columns={'TeamID':'WTeamID','SeedNum':'WSeed'}),
        on=['Season','WTeamID']
    ).merge(
        seeds_df[['Season','TeamID','SeedNum']].rename(columns={'TeamID':'LTeamID','SeedNum':'LSeed'}),
        on=['Season','LTeamID']
    )
    # Always put lower seed number as 'A'
    df['SeedA'] = df[['WSeed','LSeed']].min(axis=1)
    df['SeedB'] = df[['WSeed','LSeed']].max(axis=1)
    df['AWins'] = (df['WSeed'] < df['LSeed']).astype(int)
    seed_hist = (
        df.groupby(['SeedA','SeedB'])
          .agg(Games=('AWins','count'), AWins=('AWins','sum'))
          .reset_index()
    )
    seed_hist['HistWinProb'] = seed_hist['AWins'] / seed_hist['Games']
    # Only keep matchups with enough data
    seed_hist = seed_hist[seed_hist['Games'] >= min_seasons]
    return seed_hist.set_index(['SeedA','SeedB'])['HistWinProb'].to_dict()

m_seed_hist = historical_seed_upset_probs(m_tourney, m_seeds)
w_seed_hist = historical_seed_upset_probs(w_tourney, w_seeds)
print(f"Seed matchup history - Men: {len(m_seed_hist)} pairs | Women: {len(w_seed_hist)} pairs")
```

# Build Feature Matrix
We merge all signal sources and create T1−T2 difference features, plus interaction terms (e.g., SeedDiff × EloDiff, SeedDiff²).

```python
def merge_team_stats(season_stats_dict: dict, team_id: int, season: int) -> dict:
    """Merge multiple stat tables for one team-season into a flat dict."""
    result = {}
    for prefix, df in season_stats_dict.items():
        if df is None:
            continue
        row = df[(df['Season'] == season) & (df['TeamID'] == team_id)]
        if len(row) == 0:
            continue
        row = row.iloc[0]
        for col in row.index:
            if col not in ('Season', 'TeamID'):
                result[f'{prefix}_{col}'] = row[col]
    return result

def build_matchup_df(tourney_df, seeds_df,
                     stats_sources: dict,
                     seed_hist_lookup: dict,
                     is_train: bool = True) -> pd.DataFrame:
    """
    Build one row per matchup.
    If is_train=True: one row per actual game (always T1 = lower TeamID, label from result).
    If is_train=False: one row per sample submission pair.
    """
    rows = []
    iterdf = tourney_df if is_train else tourney_df

    for _, r in iterdf.iterrows():
        s  = r['Season']
        if is_train:
            t1, t2 = sorted([r['WTeamID'], r['LTeamID']])
            label  = 1 if r['WTeamID'] == t1 else 0
        else:
            t1, t2 = int(r['T1']), int(r['T2'])
            label  = np.nan

        s1 = merge_team_stats(stats_sources, t1, s)
        s2 = merge_team_stats(stats_sources, t2, s)

        # Seed lookup
        seed1 = seeds_df[(seeds_df['Season']==s) & (seeds_df['TeamID']==t1)]
        seed2 = seeds_df[(seeds_df['Season']==s) & (seeds_df['TeamID']==t2)]
        sn1 = int(seed1['SeedNum'].values[0]) if len(seed1) else 9
        sn2 = int(seed2['SeedNum'].values[0]) if len(seed2) else 9

        # Historical seed matchup win prob
        sa, sb = min(sn1,sn2), max(sn1,sn2)
        hist_wp = seed_hist_lookup.get((sa,sb), 0.5)
        # Flip if T1 is the higher-seed-number
        seed_hist_t1 = hist_wp if sn1 <= sn2 else (1 - hist_wp)

        row = {'Season': s, 'T1': t1, 'T2': t2, 'Label': label,
               'T1_Seed': sn1, 'T2_Seed': sn2,
               'SeedDiff': sn1 - sn2,
               'HistSeedWP': seed_hist_t1}

        for key, val in s1.items():
            row[f'T1_{key}'] = val
        for key, val in s2.items():
            row[f'T2_{key}'] = val

        rows.append(row)

    df = pd.DataFrame(rows)

    # -- Difference features -----------------------------------------------
    shared_keys = set()
    for c in df.columns:
        if c.startswith('T1_'):
            base = c[3:]
            if f'T2_{base}' in df.columns:
                shared_keys.add(base)

    for base in shared_keys:
        df[f'Diff_{base}'] = df[f'T1_{base}'] - df[f'T2_{base}']

    # -- Interaction / polynomial features --------------------------------
    if 'Diff_cs_Elo' in df.columns:
        df['SeedDiff_x_EloDiff'] = df['SeedDiff'] * df['Diff_cs_Elo']
        df['EloDiff_sq']         = df['Diff_cs_Elo'] ** 2
    if 'Diff_ff_NetRtg' in df.columns:
        df['SeedDiff_x_NetRtg']  = df['SeedDiff'] * df['Diff_ff_NetRtg']
    df['SeedDiff_sq'] = df['SeedDiff'] ** 2

    return df

# -- Build source dictionaries ----------------------------------------------

def make_sources(cstats, ff, sos, form, massey_feats, elo_df):
    srcs = {'cs': cstats, 'sos': sos, 'form': form, 'mass': massey_feats}
    if ff is not None:
        srcs['ff'] = ff
    if elo_df is not None:
        srcs['el'] = elo_df
    return srcs

m_sources = make_sources(m_cstats, m_ff, m_sos, m_form, m_massey_feats, m_elo)
w_sources = make_sources(w_cstats, w_ff, w_sos, w_form, w_massey_feats, w_elo)

print("Building men's training matrix...")
m_train = build_matchup_df(m_tourney, m_seeds, m_sources, m_seed_hist, is_train=True)
print(f"  → {m_train.shape}")

print("Building women's training matrix...")
w_train = build_matchup_df(w_tourney, w_seeds, w_sources, w_seed_hist, is_train=True)
print(f"  → {w_train.shape}")

# Feature column lists
def get_feat_cols(df):
    drop = {'Season','T1','T2','Label','T1_Seed','T2_Seed'}
    cat  = []  # We'll use numeric encoding of seeds via SeedDiff; no cat cols
    num  = [c for c in df.columns if c not in drop and df[c].dtype != 'O']
    return num, cat

m_feat_cols, m_cat_cols = get_feat_cols(m_train)
w_feat_cols, w_cat_cols = get_feat_cols(w_train)
print(f"Men feature count: {len(m_feat_cols)} | Women: {len(w_feat_cols)}")

def split_feature_sets(feat_cols):
    """
    Splits features into 'Stats/On-court' for XGBoost and 'Rankings/Massey' for CatBoost.
    HistGB will continue to use the full 'feat_cols' to remain a generalist.
    """
    stat_cols = []
    rank_cols = []
    
    for c in feat_cols:
        # Fundamental tournament features go to BOTH specialized models
        if 'Seed' in c or 'HistSeedWP' in c:
            stat_cols.append(c)
            rank_cols.append(c)
        # Rankings, Massey Ordinals, and external systems go to CatBoost
        elif 'mass_' in c or 'Rank_' in c or 'Massey' in c:
            rank_cols.append(c)
        # Four Factors, Elo, Rolling Form, SOS, and Compact Stats go to XGBoost
        else:
            stat_cols.append(c)
            
    return sorted(list(set(stat_cols))), sorted(list(set(rank_cols)))

# Generate the specialized feature subsets
m_stat_cols, m_rank_cols = split_feature_sets(m_feat_cols)
w_stat_cols, w_rank_cols = split_feature_sets(w_feat_cols)

print(f"Men's Features -> All: {len(m_feat_cols)} | Stats(XGB): {len(m_stat_cols)} | Ranks(CAT): {len(m_rank_cols)}")
```

# Modeling - Temporal Walk-Forward CV

### Why not StratifiedKFold?
StratifiedKFold randomly mixes games from all years.
This means the model is trained on future data (e.g., 2024 game) and validated on
past data (e.g., 2014 game). Leakage inflates CV scores and leads to overfitted models.

For each validation season `v`, train on `Season < v`, validate on `Season == v`.
This mirrors the actual competition setup (submit before March, train only on past).

```python
# Tuned Defaults

HIST_DEF = dict(
    max_iter=600, learning_rate=0.03, max_leaf_nodes=63,
    min_samples_leaf=15, l2_regularization=1.0, max_features=0.7,
    random_state=SEED, early_stopping=True
)
XGB_DEF = dict(
    n_estimators=600, learning_rate=0.03, max_depth=5,
    min_child_weight=10, subsample=0.75, colsample_bytree=0.7,
    reg_alpha=0.1, reg_lambda=1.0,
    random_state=SEED, tree_method='hist', eval_metric='logloss',
    n_jobs=-1, verbosity=0
)
CAT_DEF = dict(
    iterations=600, learning_rate=0.03, depth=6, l2_leaf_reg=3,
    subsample=0.75, colsample_bylevel=0.7,
    random_state=SEED, verbose=0, thread_count=-1
)


# Optuna HPO

def optuna_hist(X_tr, y_tr, X_va, y_va, n_trials=40):
    """Return best HistGB params found by Optuna."""
    def objective(trial):
        params = dict(
            max_iter=trial.suggest_int('max_iter', 200, 800),
            learning_rate=trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            max_leaf_nodes=trial.suggest_int('max_leaf_nodes', 15, 127),
            min_samples_leaf=trial.suggest_int('min_samples_leaf', 5, 50),
            l2_regularization=trial.suggest_float('l2_regularization', 0.0, 5.0),
            max_features=trial.suggest_float('max_features', 0.4, 1.0),
            random_state=SEED, early_stopping=True
        )
        m = HistGradientBoostingClassifier(**params)
        m.fit(X_tr, y_tr)
        p = m.predict_proba(X_va)[:,1]
        return brier_score_loss(y_va, p)
    study = optuna.create_study(direction='minimize',
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = study.best_params
    best.update({'random_state': SEED, 'early_stopping': True})
    return best


# Walk-forward CV loop

# Walk-forward CV loop
# Walk-forward CV loop
def temporal_cv(train_df: pd.DataFrame, feat_cols: list, stat_cols: list, rank_cols: list,
                val_seasons=None, run_optuna: bool = False,
                optuna_trials: int = 30):
    """
    Proper temporal cross-validation.
    Returns: (mean_brier, std_brier, oof_pred_series, oof_label_series,
              hist_params, xgb_params, cat_params, season_briers)
    """
    df = train_df.copy()
    # Fill numeric NaNs with column medians
    num_med = df[feat_cols].median()
    df[feat_cols] = df[feat_cols].fillna(num_med)

    seasons = sorted(df['Season'].unique())
    if val_seasons is None:
        # Leave-one-season-out starting from the 6th season
        val_seasons = seasons[5:]

    oof_preds  = pd.Series(np.nan, index=df.index)
    oof_labels = df['Label'].copy()
    season_briers = {}

    hist_params = HIST_DEF.copy()
    xgb_params = XGB_DEF.copy()
    cat_params  = CAT_DEF.copy()

    for val_s in tqdm(val_seasons, desc='Temporal CV'):
        tr_idx = df[df['Season'] < val_s].index
        va_idx = df[df['Season'] == val_s].index
        if len(tr_idx) < 20 or len(va_idx) < 5:
            continue

        y_tr = df.loc[tr_idx, 'Label'].values
        y_va = df.loc[va_idx, 'Label'].values

        # 1. HistGB receives all features
        X_tr_hist = df.loc[tr_idx, feat_cols].values
        X_va_hist = df.loc[va_idx, feat_cols].values

        # 2. XGBoost receives stats only
        X_tr_xgb = df.loc[tr_idx, stat_cols].values
        X_va_xgb = df.loc[va_idx, stat_cols].values

        # 3. CatBoost receives ranks only
        X_tr_cat = df.loc[tr_idx, rank_cols].values
        X_va_cat = df.loc[va_idx, rank_cols].values

        # Optuna on last val season (only impacts HistGB here)
        if run_optuna and HAS_OPTUNA and val_s == val_seasons[-1]:
            print(f"Running Optuna HPO on validation season {val_s}...")
            hist_params = optuna_hist(X_tr_hist, y_tr, X_va_hist, y_va, n_trials=optuna_trials)

        hist_m = HistGradientBoostingClassifier(**hist_params)
        hist_m.fit(X_tr_hist, y_tr)
        p_hist = hist_m.predict_proba(X_va_hist)[:,1]

        xgb_m = xgb.XGBClassifier(**xgb_params)
        xgb_m.fit(X_tr_xgb, y_tr)
        p_xgb = xgb_m.predict_proba(X_va_xgb)[:,1]

        cat_m = CatBoostClassifier(**cat_params)
        cat_m.fit(X_tr_cat, y_tr)
        p_cat = cat_m.predict_proba(X_va_cat)[:,1]

        # Blend (equal for now; will be optimised later by Meta-Learner)
        p_blend = (p_hist + p_xgb + p_cat) / 3

        oof_preds.loc[va_idx] = p_blend
        bs = brier_score_loss(y_va, p_blend)
        season_briers[val_s] = bs

    mask = oof_preds.notna()
    oof_p = oof_preds[mask]
    oof_l = oof_labels[mask]

    mean_b = np.mean(list(season_briers.values()))
    std_b  = np.std(list(season_briers.values()))

    return mean_b, std_b, oof_p, oof_l, hist_params, xgb_params, cat_params, season_briers


print("Running temporal CV - Men's...")
(m_mean_b, m_std_b, m_oof_p, m_oof_l,
 m_hist_p, m_xgb_p, m_cat_p, m_season_b) = temporal_cv(
    m_train, m_feat_cols, m_stat_cols, m_rank_cols, run_optuna=True, optuna_trials=40
)
print(f"  Men's  Brier: {m_mean_b:.5f} ± {m_std_b:.5f}")

print("Running temporal CV - Women's...")
(w_mean_b, w_std_b, w_oof_p, w_oof_l,
 w_hist_p, w_xgb_p, w_cat_p, w_season_b) = temporal_cv(
    w_train, w_feat_cols, w_stat_cols, w_rank_cols, run_optuna=True, optuna_trials=40
)
print(f"  Women's Brier: {w_mean_b:.5f} ± {w_std_b:.5f}")
```

# Advanced Stacking & Isotonic Calibration
Instead of static blend weights, we use a Logistic Regression meta-learner. It takes the OOF probabilities of our three tree models plus the `SeedDiff` feature. This allows the ensemble to dynamically adjust its trust in each model depending on the severity of the seed mismatch, effectively neutralizing the high correlation between the GBDTs.

```python
# Prevents kernel crash
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Per-fold OOF with individual model predictions stored
def temporal_cv_stacked(train_df, feat_cols, stat_cols, rank_cols, 
                        hist_params, xgb_params, cat_params, val_seasons=None):
    """Same CV but trains XGBoost and CatBoost on mutually exclusive feature sets."""
    df = train_df.copy()
    num_med = df[feat_cols].median()
    df[feat_cols] = df[feat_cols].fillna(num_med)
    seasons = sorted(df['Season'].unique())
    if val_seasons is None:
        val_seasons = seasons[5:]

    oof_hist = pd.Series(np.nan, index=df.index)
    oof_xgb  = pd.Series(np.nan, index=df.index)
    oof_cat  = pd.Series(np.nan, index=df.index)
    oof_lbl  = df['Label'].copy()
    oof_sd   = df['SeedDiff'].copy() 

    for val_s in tqdm(val_seasons, desc='Stacked CV'):
        tr_idx = df[df['Season'] < val_s].index
        va_idx = df[df['Season'] == val_s].index
        if len(tr_idx) < 20 or len(va_idx) < 5:
            continue
            
        y_tr = df.loc[tr_idx, 'Label'].values
        
        # 1. HistGB receives all features
        X_tr_hist = df.loc[tr_idx, feat_cols].values
        X_va_hist = df.loc[va_idx, feat_cols].values
        hist_m = HistGradientBoostingClassifier(**hist_params)
        hist_m.fit(X_tr_hist, y_tr)
        oof_hist.loc[va_idx] = hist_m.predict_proba(X_va_hist)[:,1]

        # 2. XGBoost receives purely statistical/on-court features
        X_tr_xgb = df.loc[tr_idx, stat_cols].values
        X_va_xgb = df.loc[va_idx, stat_cols].values
        xgb_m = xgb.XGBClassifier(**xgb_params)
        xgb_m.fit(X_tr_xgb, y_tr)
        oof_xgb.loc[va_idx] = xgb_m.predict_proba(X_va_xgb)[:,1]

        # 3. CatBoost receives purely off-court rankings/Massey features
        X_tr_cat = df.loc[tr_idx, rank_cols].values
        X_va_cat = df.loc[va_idx, rank_cols].values
        cat_m = CatBoostClassifier(**cat_params)
        cat_m.fit(X_tr_cat, y_tr)
        oof_cat.loc[va_idx] = cat_m.predict_proba(X_va_cat)[:,1]

    mask = oof_hist.notna()
    return (oof_hist[mask].values, oof_xgb[mask].values,
            oof_cat[mask].values, oof_lbl[mask].values, oof_sd[mask].values)

def train_meta_learner(p_hist, p_xgb, p_cat, seed_diff, labels):
    """Train Logistic Regression on model predictions + SeedDiff."""
    # Convert probabilities to log-odds for better linear separation
    def to_log_odds(p):
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p))
        
    X_meta = np.column_stack([
        to_log_odds(p_hist), 
        to_log_odds(p_xgb), 
        to_log_odds(p_cat), 
        seed_diff
    ])
    
    scaler = StandardScaler()
    X_meta_scaled = scaler.fit_transform(X_meta)
    
    # L2 penalty gracefully handles the high correlation between the 3 models
    meta = LogisticRegression(penalty='l2', C=0.1, random_state=SEED, max_iter=1000)
    meta.fit(X_meta_scaled, labels)
    
    p_blend = meta.predict_proba(X_meta_scaled)[:, 1]
    return meta, scaler, p_blend


print("Computing stacked OOF - Men's...")
m_oh, m_ox, m_oc, m_yl, m_sd = temporal_cv_stacked(
    m_train, m_feat_cols, m_stat_cols, m_rank_cols, m_hist_p, m_xgb_p, m_cat_p)

print("Computing stacked OOF - Women's...")
w_oh, w_ox, w_oc, w_yl, w_sd = temporal_cv_stacked(
    w_train, w_feat_cols, w_stat_cols, w_rank_cols, w_hist_p, w_xgb_p, w_cat_p)

# Train Meta-Learners
m_meta_model, m_meta_scaler, m_oof_blend = train_meta_learner(m_oh, m_ox, m_oc, m_sd, m_yl)
w_meta_model, w_meta_scaler, w_oof_blend = train_meta_learner(w_oh, w_ox, w_oc, w_sd, w_yl)

print(f"\nMen's Meta-Learner Coefs: HIST={m_meta_model.coef_[0][0]:.3f}, XGB={m_meta_model.coef_[0][1]:.3f}, CAT={m_meta_model.coef_[0][2]:.3f}, SeedDiff={m_meta_model.coef_[0][3]:.3f}")
print(f"Women's Meta-Learner Coefs: HIST={w_meta_model.coef_[0][0]:.3f}, XGB={w_meta_model.coef_[0][1]:.3f}, CAT={w_meta_model.coef_[0][2]:.3f}, SeedDiff={w_meta_model.coef_[0][3]:.3f}")

# -- Isotonic Calibration ---------------------------------------------------
m_iso = IsotonicRegression(out_of_bounds='clip')
m_iso.fit(m_oof_blend, m_yl)

w_iso = IsotonicRegression(out_of_bounds='clip')
w_iso.fit(w_oof_blend, w_yl)

m_oof_cal = m_iso.predict(m_oof_blend)
w_oof_cal = w_iso.predict(w_oof_blend)

print(f"\nFinal OOF Brier (Men):   uncal={brier_score_loss(m_yl,m_oof_blend):.5f}  cal={brier_score_loss(m_yl,m_oof_cal):.5f}")
print(f"Final OOF Brier (Women): uncal={brier_score_loss(w_yl,w_oof_blend):.5f}  cal={brier_score_loss(w_yl,w_oof_cal):.5f}")
```

# Evaluation Plots & Inference

```python
# -- 1. Build Features ------------------------------------------------------

def parallel_build_features(sub_df, seeds_df, sources, seed_hist, is_train=False):
    """Splits the dataframe into chunks and builds features in parallel across all CPU cores."""
    n_cores = multiprocessing.cpu_count()
    
    chunk_size = int(np.ceil(len(sub_df) / n_cores))
    df_chunks = [sub_df.iloc[i * chunk_size : (i + 1) * chunk_size] for i in range(n_cores)]
    df_chunks = [chunk for chunk in df_chunks if not chunk.empty]
    
    results = Parallel(n_jobs=-1)(
        delayed(build_matchup_df)(chunk, seeds_df, sources, seed_hist, is_train) 
        for chunk in tqdm(df_chunks, desc=" └ Building Features (Cores)", leave=False)
    )
    
    return pd.concat(results, ignore_index=True)

# -- 2. Define Custom Callbacks ---------------------------------------------

class XgbTqdmCallback(xgb.callback.TrainingCallback):
    def __init__(self, pbar):
        self.pbar = pbar
    def after_iteration(self, model, epoch, evals_log):
        self.pbar.update(1)
        return False

# -- 3. Update Training Function --------------------------------------------

def train_final_models(train_df, feat_cols, stat_cols, rank_cols, 
                       hist_params, xgb_params, cat_params):
    """Train each base model on the full training set using its specialized feature subset."""
    df = train_df.dropna(subset=['Label']).copy()
    train_med = df[feat_cols].median()
    df[feat_cols] = df[feat_cols].fillna(train_med)
    
    y = df['Label'].values.astype(int)

    # Generalist Model
    hist_m = HistGradientBoostingClassifier(**hist_params)
    hist_m.fit(df[feat_cols].values, y)

    # Statistical Model
    xgb_iters = xgb_params.get('n_estimators', 100)
    with tqdm(total=xgb_iters, desc=" └ XGBoost (Stats)", leave=False) as pbar:
        xgb_m = xgb.XGBClassifier(**xgb_params, callbacks=[XgbTqdmCallback(pbar)])
        xgb_m.fit(df[stat_cols].values, y) 

    # Ranking Model
    cat_iters = cat_params.get('iterations', cat_params.get('n_estimators', 1000))
    with tqdm(total=cat_iters, desc=" └ CatBoost (Ranks)", leave=False) as pbar:
        cat_m = CatBoostClassifier(**cat_params)
        cat_m.fit(df[rank_cols].values, y, verbose=False)
        pbar.update(cat_iters) 

    return hist_m, xgb_m, cat_m, train_med


def predict_submission_stacked(sub_df, seeds_df, sources, seed_hist,
                               hist_m, xgb_m, cat_m, train_med,
                               feat_cols, stat_cols, rank_cols, 
                               meta_model, meta_scaler, calibrator):
    """Generate specialized base predictions, apply Meta-Learner, and calibrate."""
    
    with tqdm(total=5, desc=" └ Inference Status", leave=False) as pbar:
        
        # Phase 1: Parallel Feature Engineering
        pbar.set_postfix_str("Building features (Parallel)...")
        sub_feats = parallel_build_features(
            sub_df, seeds_df, sources, seed_hist, is_train=False
        )
        
        # Safely fill missing systems before fillna
        for c in feat_cols:
            if c not in sub_feats.columns:
                sub_feats[c] = train_med.get(c, 0)
                
        sub_feats[feat_cols] = sub_feats[feat_cols].fillna(train_med)
        
        seed_diff = sub_feats['SeedDiff'].values
        pbar.update(1)

        # Phase 2: Concurrent Base Model Predictions (using Subsets)
        pbar.set_postfix_str("Running specialized models...")
        from concurrent.futures import ThreadPoolExecutor
        
        X_hist = sub_feats[feat_cols].values
        X_xgb  = sub_feats[stat_cols].values
        X_cat  = sub_feats[rank_cols].values
        
        with ThreadPoolExecutor(max_workers=3) as executor:
            p_hist = executor.submit(lambda: hist_m.predict_proba(X_hist)[:,1]).result()
            pbar.update(1)
            p_xgb = executor.submit(lambda: xgb_m.predict_proba(X_xgb)[:,1]).result()
            pbar.update(1)
            p_cat = executor.submit(lambda: cat_m.predict_proba(X_cat)[:,1]).result()
            pbar.update(1)

        # Phase 3: Meta-Learner Stacking & Calibration
        pbar.set_postfix_str("Meta-Learning & Calibrating...")
        
        def to_log_odds(p):
            p = np.clip(p, 1e-6, 1 - 1e-6)
            return np.log(p / (1 - p))
            
        X_meta = np.column_stack([
            to_log_odds(p_hist), 
            to_log_odds(p_xgb), 
            to_log_odds(p_cat), 
            seed_diff
        ])
        
        X_meta_scaled = meta_scaler.transform(X_meta)
        p_blend = meta_model.predict_proba(X_meta_scaled)[:, 1]
        p_cal = calibrator.predict(p_blend)
        pbar.update(1)
        
        pbar.set_postfix_str("Complete!")
        return p_cal


# -- 5. Execute Pipeline ----------------------------------------------------

m_xgb_p['n_jobs'] = -1; w_xgb_p['n_jobs'] = -1
m_cat_p['thread_count'] = -1; w_cat_p['thread_count'] = -1

with tqdm(total=4, desc="Total Pipeline Status", position=0) as main_pbar:
    
    print("\nTraining final Men's models (Specialized)...")
    m_hist_f, m_xgb_f, m_cat_f, m_med = train_final_models(
        m_train, m_feat_cols, m_stat_cols, m_rank_cols, 
        m_hist_p, m_xgb_p, m_cat_p
    )
    main_pbar.update(1)

    print("\nTraining final Women's models (Specialized)...")
    w_hist_f, w_xgb_f, w_cat_f, w_med = train_final_models(
        w_train, w_feat_cols, w_stat_cols, w_rank_cols, 
        w_hist_p, w_xgb_p, w_cat_p
    )
    main_pbar.update(1)

    print("\nPredicting Men's submission (Stacked)...")
    m_sub['Pred'] = predict_submission_stacked(
        m_sub, m_seeds, m_sources, m_seed_hist,
        m_hist_f, m_xgb_f, m_cat_f, m_med,
        m_feat_cols, m_stat_cols, m_rank_cols, 
        m_meta_model, m_meta_scaler, m_iso
    )
    main_pbar.update(1)

    print("\nPredicting Women's submission (Stacked)...")
    w_sub['Pred'] = predict_submission_stacked(
        w_sub, w_seeds, w_sources, w_seed_hist,
        w_hist_f, w_xgb_f, w_cat_f, w_med,
        w_feat_cols, w_stat_cols, w_rank_cols, 
        w_meta_model, w_meta_scaler, w_iso
    )
    main_pbar.update(1)


# -- Combine and clip -------------------------------------------------------
sub_final = pd.concat([m_sub[['ID','Pred']], w_sub[['ID','Pred']]])
sub_final['Pred'] = sub_final['Pred'].clip(0.015, 0.985)
sub_final = sub_final.sort_values('ID').reset_index(drop=True)

print("\nPrediction stats:")
print(sub_final['Pred'].describe())

sub_final.to_csv('submission.csv', index=False)
print("\nsubmission.csv saved successfully!")

import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(10, 4))
sns.histplot(sub_final['Pred'], bins=50, kde=True)
plt.title("Submission Probabilities Distribution (Stacked Ensemble)")
plt.xlabel("Probability (T1 wins)")
plt.tight_layout()
plt.savefig('submission_distribution_stacked.png', bbox_inches='tight')
plt.show()

print("\nFraction of predictions > 0.70 or < 0.30:",
      ((sub_final['Pred'] > 0.70) | (sub_final['Pred'] < 0.30)).mean().round(3))
```

# Model Insights & Diagnostics

```python
fig = plt.figure(figsize=(20, 18))
gs  = gridspec.GridSpec(3, 2, hspace=0.45, wspace=0.30)

# -- Helpers ------------------------------------------------------------------
def _brier(y, p): return np.mean((np.asarray(y) - np.asarray(p)) ** 2)

# -- 1. OOF Ensemble Correlation ----------------------------------------------
ax1 = fig.add_subplot(gs[0, 0])
oof_df = pd.DataFrame({
    'HistGB'  : np.concatenate([m_oh, w_oh]),
    'XGBoost' : np.concatenate([m_ox, w_ox]),
    'CatBoost': np.concatenate([m_oc, w_oc]),
})
corr = oof_df.corr()

sns.heatmap(
    corr, annot=True, fmt='.4f', cmap='coolwarm',
    vmin=0.85, vmax=1.0, linewidths=1.5, linecolor='white',
    cbar_kws={'label': 'Pearson r', 'shrink': 0.80},
    ax=ax1, annot_kws={'size': 13, 'weight': 'bold'}
)
ax1.set_title('OOF Prediction Correlation (Men + Women)\n'
              'Ensemble diversity: lower off-diagonal = more diverse', pad=12)

# -- 2. Calibration Curves: before vs after isotonic -------------------------
ax2 = fig.add_subplot(gs[0, 1])
calib_configs = [
    (m_oof_blend, m_yl, C_MEN,   '--', 'o', "Men - Raw Blend"),
    (m_oof_cal,   m_yl, C_MEN,   '-',  's', "Men - Isotonic Cal."),
    (w_oof_blend, w_yl, C_WOMEN, '--', 'o', "Women - Raw Blend"),
    (w_oof_cal,   w_yl, C_WOMEN, '-',  '^', "Women - Isotonic Cal."),
]
for preds, lbl, color, ls, marker, name in calib_configs:
    frac_pos, mean_pred = calibration_curve(lbl, preds, n_bins=10, strategy='quantile')
    lw    = 2.2 if ls == '-' else 1.4
    alpha = 1.0 if ls == '-' else 0.55
    ax2.plot(mean_pred, frac_pos, ls, color=color, marker=marker,
             linewidth=lw, markersize=5, alpha=alpha, label=name)

ax2.plot([0, 1], [0, 1], 'k:', linewidth=1.2, label='Perfect')
ax2.set_xlabel('Mean Predicted Probability')
ax2.set_ylabel('Fraction of Positives')
ax2.set_title('Reliability Diagram\nDashed = raw blend  |  Solid = after isotonic calibration', pad=12)
ax2.legend(fontsize=8.5, loc='lower right')
ax2.grid(True, linestyle='--', alpha=0.35)

# -- 3. Meta-Learner (Stacking LR) Coefficients ------------------------------
ax3 = fig.add_subplot(gs[1, 0])
meta_labels = ['HistGB', 'XGBoost', 'CatBoost', 'SeedDiff']
m_coefs = m_meta_model.coef_[0]
w_coefs = w_meta_model.coef_[0]

x_pos = np.arange(len(meta_labels))
bar_w = 0.36
bars_m = ax3.bar(x_pos - bar_w/2, m_coefs, bar_w, label="Men's",
                 color=C_MEN,   alpha=0.85, edgecolor='white', linewidth=0.6)
bars_w = ax3.bar(x_pos + bar_w/2, w_coefs, bar_w, label="Women's",
                 color=C_WOMEN, alpha=0.85, edgecolor='white', linewidth=0.6)
ax3.axhline(0, color='black', linewidth=0.8, zorder=0)

for bars in (bars_m, bars_w):
    for bar in bars:
        h = bar.get_height()
        offset = 0.012 if h >= 0 else -0.012
        va = 'bottom' if h >= 0 else 'top'
        ax3.text(bar.get_x() + bar.get_width() / 2, h + offset,
                 f'{h:+.3f}', ha='center', va=va, fontsize=8.5, fontweight='bold')

ax3.set_xticks(x_pos)
ax3.set_xticklabels(meta_labels, fontsize=10)
ax3.set_ylabel('Standardized Coefficient')
ax3.set_title('Meta-Learner (Logistic Regression) Coefficients\n'
              'How much stacking trusts each base model + seed signal', pad=12)
ax3.legend(fontsize=9)
ax3.grid(axis='y', linestyle='--', alpha=0.35)

# -- 4. OOF Brier Score by |SeedDiff| Bucket (Upset Analysis) ----------------
ax4 = fig.add_subplot(gs[1, 1])

def brier_by_seed_bucket(sd, y, p, edges):
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (np.abs(sd) >= lo) & (np.abs(sd) < hi)
        if mask.sum() < 5:
            continue
        rows.append({'bucket': f'{lo}–{hi}', 'brier': _brier(y[mask], p[mask]),
                     'n': mask.sum()})
    return pd.DataFrame(rows)

edges = [0, 2, 4, 6, 9, 13, 17]
m_bdf = brier_by_seed_bucket(m_sd, m_yl, m_oof_cal, edges)
w_bdf = brier_by_seed_bucket(w_sd, w_yl, w_oof_cal, edges)
n_bins = min(len(m_bdf), len(w_bdf))
x_b = np.arange(n_bins)

b_m = ax4.bar(x_b - 0.2, m_bdf['brier'].values[:n_bins], 0.38,
               label="Men's",   color=C_MEN,   alpha=0.85, edgecolor='white')
b_w = ax4.bar(x_b + 0.2, w_bdf['brier'].values[:n_bins], 0.38,
               label="Women's", color=C_WOMEN, alpha=0.85, edgecolor='white')

for b_set in (b_m, b_w):
    for bar in b_set:
        h = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width() / 2, h + 0.001,
                 f'{h:.3f}', ha='center', va='bottom', fontsize=7.5)

ax4.set_xticks(x_b)
ax4.set_xticklabels(m_bdf['bucket'].values[:n_bins], fontsize=9)
ax4.set_xlabel('|SeedDiff| Bucket')
ax4.set_ylabel('Mean Brier Score')
ax4.set_title('Calibrated OOF Brier by Seed Mismatch\n'
              'Near-zero gap = close matchups; large gap = heavy favourites / upsets', pad=12)
ax4.legend(fontsize=9)
ax4.grid(axis='y', linestyle='--', alpha=0.35)

# -- 5 & 6. Ensemble-Averaged Feature Importances ----------------------------
def ensemble_importances(model_col_pairs, top_n=15):
    """
    Averages normalized importances across supplied models by feature name.
    Handles differing feature sets safely.
    """
    series_dict = {}
    for i, (m, cols) in enumerate(model_col_pairs):
        fi = np.array(m.feature_importances_, dtype=float)
        # Normalize to 1.0 to give equal voting weight to each model
        if fi.sum() > 0:
            fi = fi / fi.sum()
        series_dict[f'model_{i}'] = pd.Series(fi, index=cols)
        
    # Merge all importances. Fill missing with 0 (since the model didn't use that feature)
    merged = pd.DataFrame(series_dict).fillna(0)
    merged['Mean_Importance'] = merged.mean(axis=1)
    
    return merged.reset_index().rename(columns={'index': 'Feature'}) \
                 .sort_values('Mean_Importance', ascending=False).head(top_n)

# Include HistGB importances if the attribute is available (scikit-learn >= 1.3)
try:
    _ = m_hist_f.feature_importances_
    panel_cfg = [
        (gs[2, 0], [(m_hist_f, m_feat_cols), (m_xgb_f, m_stat_cols), (m_cat_f, m_rank_cols)], C_MEN, "Men's"),
        (gs[2, 1], [(w_hist_f, w_feat_cols), (w_xgb_f, w_stat_cols), (w_cat_f, w_rank_cols)], C_WOMEN, "Women's"),
    ]
    imp_subtitle = 'HistGB + XGBoost + CatBoost average'
except AttributeError:
    panel_cfg = [
        (gs[2, 0], [(m_xgb_f, m_stat_cols), (m_cat_f, m_rank_cols)], C_MEN, "Men's"),
        (gs[2, 1], [(w_xgb_f, w_stat_cols), (w_cat_f, w_rank_cols)], C_WOMEN, "Women's"),
    ]
    imp_subtitle = 'XGBoost + CatBoost average'

for slot_spec, model_col_pairs, color, gender in panel_cfg:
    ax = fig.add_subplot(slot_spec)
    imp_df = ensemble_importances(model_col_pairs)

    bars = ax.barh(
        imp_df['Feature'][::-1].values,
        imp_df['Mean_Importance'][::-1].values,
        color=color, alpha=0.85, edgecolor='white', linewidth=0.5
    )
    x_max = imp_df['Mean_Importance'].max()
    for bar in bars:
        w = bar.get_width()
        ax.text(w + x_max * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f'{w:.4f}', va='center', fontsize=7.5)

    ax.set_xlabel(f'Mean Normalized Importance ({imp_subtitle})')
    ax.set_title(f'Top 15 Features - {gender} Ensemble\n({imp_subtitle})', pad=12)
    ax.set_xlim(right=x_max * 1.14)
    ax.grid(axis='x', linestyle='--', alpha=0.35)

# -- Suptitle & Save -----------------------------------------------------------
sns.despine(fig=fig)
fig.suptitle(
    'March Machine Learning Mania 2026 - Model Diagnostics Dashboard',
    fontsize=17, fontweight='bold', y=1.002
)
plt.savefig('ensemble_diagnostics.png', bbox_inches='tight', dpi=150)
plt.show()
```