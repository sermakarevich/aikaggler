# 🏀 NCAA 2026 EDA, Elo Ratings,and Gradient Esemble

- **Author:** Muhammad Ibrahim Qasmi
- **Votes:** 114
- **Ref:** ibrahimqasimi/ncaa-2026-eda-elo-ratings-and-gradient-esemble
- **URL:** https://www.kaggle.com/code/ibrahimqasimi/ncaa-2026-eda-elo-ratings-and-gradient-esemble
- **Last run:** 2026-02-24 09:08:46.887000

---

<div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%); 
padding: 40px 30px; border-radius: 12px; margin-bottom: 20px; 
border-left: 6px solid #e94560; font-family: Georgia, serif;">
<h1 style="color: #ffffff; font-size: 2.4em; margin: 0 0 8px 0; letter-spacing: 1px;">
March Machine Learning Mania 2026</h1>
<h2 style="color: #e94560; font-size: 1.3em; margin: 0 0 16px 0; font-weight: 400;">
Predicting NCAA Tournament Outcomes with Gradient Boosting Ensembles</h2>
<p style="color: #a8a8b3; font-size: 0.95em; margin: 0; line-height: 1.6;">
A systematic approach to forecasting the 2026 NCAA Division I Men&#39;s and Women&#39;s Basketball Tournaments.<br>
This notebook constructs Elo ratings, derives Four Factors efficiency metrics, engineers 
conference-strength and momentum features, and combines LightGBM, XGBoost, and CatBoost 
into a calibrated ensemble. Validated on 2022&#8211;2025 tournaments.</p>
</div>

![](https://images.sidearmdev.com/convert?url=https%3a%2f%2fdxbhsrqyrr690.cloudfront.net%2fsidearm.nextgen.sites%2fncaabballaca.sidearmsports.com%2fimages%2f2023%2f3%2f1%2f_JD40754_4nPL6.JPG&type=webp)

## Table of Contents

1. [Overview and Objective](#1-overview-and-objective)
2. [Setup](#2-setup)
3. [Data Loading and Inventory](#3-data-loading-and-inventory)
4. [Exploratory Data Analysis](#4-exploratory-data-analysis)
   - 4.1 Seed Predictive Power
   - 4.2 Home Court Advantage
   - 4.3 Scoring Trends Over Time
   - 4.4 Conference Landscape
   - 4.5 Upset Frequency by Round
5. [Feature Engineering](#5-feature-engineering)
   - 5.1 Elo Rating System
   - 5.2 Four Factors and Advanced Season Statistics
   - 5.3 Strength of Schedule
   - 5.4 Late-Season Momentum
   - 5.5 Conference Strength
   - 5.6 Coach Experience (Men)
6. [Model Training](#6-model-training)
   - 6.1 Training Data Construction
   - 6.2 LightGBM
   - 6.3 XGBoost
   - 6.4 CatBoost
   - 6.5 Ensemble and Evaluation
7. [Diagnostics and Calibration](#7-diagnostics-and-calibration)
8. [Submission Generation](#8-submission-generation)
9. [Summary and Next Steps](#9-summary-and-next-steps)

<a id="1-overview-and-objective"></a>
## 1. Overview and Objective

The goal of this competition is to predict the probability that a given team beats another in 
every possible NCAA tournament matchup. Submissions are evaluated using **Brier score** 
(mean squared error of predicted probabilities against binary outcomes), where lower is better.

The competition follows a two-stage structure:

- **Stage 1** covers the 2022-2025 tournaments and serves as a development sandbox. 
Since historical results are available in the dataset, perfect scores are trivially achievable 
through data leakage. The Stage 1 leaderboard is not meaningful.
- **Stage 2** covers the 2026 tournament. All predictions must be submitted before the first 
game tips off. This is the only stage that determines final standings and prizes.

Our approach combines three classes of features: **rating systems** (Elo), **efficiency metrics** 
(Four Factors framework from Dean Oliver's basketball analytics), and **contextual signals** 
(conference strength, schedule difficulty, momentum, coaching experience). These are fed into an 
ensemble of gradient boosting models.

<a id="2-setup"></a>
## 2. Setup

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.calibration import calibration_curve
from pathlib import Path
import warnings, time

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 50)

PALETTE = ["#0f3460", "#e94560", "#16213e", "#533483", "#1a1a2e"]
sns.set_theme(style="whitegrid", palette=PALETTE, font_scale=1.05)
plt.rcParams.update({
    "figure.figsize": (10, 5),
    "figure.dpi": 120,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "font.family": "serif",
})

DATA = Path("/kaggle/input/march-machine-learning-mania-2026")
WORK = Path("/kaggle/working")
WORK.mkdir(exist_ok=True)

CLIP_LO, CLIP_HI = 0.02, 0.98
SEED_DEFAULT = 8.5
RS = 42

t_start = time.time()
print("Setup complete.")
```

<a id="3-data-loading-and-inventory"></a>
## 3. Data Loading and Inventory

We load all relevant files and create a quick inventory of temporal coverage and row counts.

```python
m_rs    = pd.read_csv(DATA / "MRegularSeasonCompactResults.csv")
m_det   = pd.read_csv(DATA / "MRegularSeasonDetailedResults.csv")
m_trn   = pd.read_csv(DATA / "MNCAATourneyCompactResults.csv")
m_sec   = pd.read_csv(DATA / "MSecondaryTourneyCompactResults.csv")
m_seeds = pd.read_csv(DATA / "MNCAATourneySeeds.csv")
m_conf  = pd.read_csv(DATA / "MTeamConferences.csv")
m_coach = pd.read_csv(DATA / "MTeamCoaches.csv")
m_teams = pd.read_csv(DATA / "MTeams.csv")

w_rs    = pd.read_csv(DATA / "WRegularSeasonCompactResults.csv")
w_det   = pd.read_csv(DATA / "WRegularSeasonDetailedResults.csv")
w_trn   = pd.read_csv(DATA / "WNCAATourneyCompactResults.csv")
w_sec   = pd.read_csv(DATA / "WSecondaryTourneyCompactResults.csv")
w_seeds = pd.read_csv(DATA / "WNCAATourneySeeds.csv")
w_conf  = pd.read_csv(DATA / "WTeamConferences.csv")
w_teams = pd.read_csv(DATA / "WTeams.csv")

sub1 = pd.read_csv(DATA / "SampleSubmissionStage1.csv")
sub2 = pd.read_csv(DATA / "SampleSubmissionStage2.csv")

m_seeds["SeedNum"] = m_seeds["Seed"].str[1:3].astype(int)
w_seeds["SeedNum"] = w_seeds["Seed"].str[1:3].astype(int)

inventory = pd.DataFrame([
    ["Men Regular Season (compact)",  m_rs["Season"].min(),  m_rs["Season"].max(),  len(m_rs)],
    ["Men Regular Season (detailed)", m_det["Season"].min(), m_det["Season"].max(), len(m_det)],
    ["Men NCAA Tournament",           m_trn["Season"].min(), m_trn["Season"].max(), len(m_trn)],
    ["Men Tournament Seeds",          m_seeds["Season"].min(), m_seeds["Season"].max(), len(m_seeds)],
    ["Women Regular Season (compact)",w_rs["Season"].min(),  w_rs["Season"].max(),  len(w_rs)],
    ["Women Regular Season (detailed)",w_det["Season"].min(),w_det["Season"].max(), len(w_det)],
    ["Women NCAA Tournament",         w_trn["Season"].min(), w_trn["Season"].max(), len(w_trn)],
    ["Women Tournament Seeds",        w_seeds["Season"].min(),w_seeds["Season"].max(),len(w_seeds)],
], columns=["Dataset", "First Season", "Last Season", "Rows"])
inventory["Rows"] = inventory["Rows"].apply(lambda x: f"{x:,}")
display(inventory.style.set_properties(**{"text-align": "left"}).hide(axis="index"))

print(f"\nStage 1 submission: {len(sub1):,} matchups (seasons {sub1['ID'].str[:4].unique()})")
print(f"Stage 2 submission: {len(sub2):,} matchups (season 2026)")
```

<a id="4-exploratory-data-analysis"></a>
## 4. Exploratory Data Analysis

Before engineering features, we examine the data to understand the key drivers of tournament outcomes. 
The insights from this section directly inform our feature choices.

### 4.1 Seed Predictive Power

```python
def seed_analysis(trn_df, seed_df, label):
    df = trn_df.merge(
        seed_df[["Season", "TeamID", "SeedNum"]],
        left_on=["Season", "WTeamID"], right_on=["Season", "TeamID"]
    ).rename(columns={"SeedNum": "WSeed"}).drop("TeamID", axis=1)
    df = df.merge(
        seed_df[["Season", "TeamID", "SeedNum"]],
        left_on=["Season", "LTeamID"], right_on=["Season", "TeamID"]
    ).rename(columns={"SeedNum": "LSeed"}).drop("TeamID", axis=1)
    df["SeedGap"] = df["LSeed"] - df["WSeed"]
    return df

m_sa = seed_analysis(m_trn, m_seeds, "Men")
w_sa = seed_analysis(w_trn, w_seeds, "Women")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, df, label in [(axes[0], m_sa, "Men"), (axes[1], w_sa, "Women")]:
    win_rates = []
    for gap in range(0, 16):
        mask = abs(df["LSeed"] - df["WSeed"]) == gap
        if mask.sum() > 0:
            higher_wins = (df.loc[mask, "WSeed"] <= df.loc[mask, "LSeed"]).mean()
            win_rates.append((gap, higher_wins, mask.sum()))
    gaps, rates, counts = zip(*win_rates)
    bars = ax.bar(gaps, rates, color=PALETTE[0], alpha=0.85, edgecolor="white", linewidth=0.5)
    ax.axhline(y=0.5, color=PALETTE[1], linestyle="--", linewidth=1, alpha=0.7)
    ax.set_xlabel("Absolute Seed Difference")
    ax.set_ylabel("Higher Seed Win Rate")
    ax.set_title(f"{label}: Win Rate by Seed Gap")
    ax.set_ylim(0.4, 1.02)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))

plt.tight_layout()
plt.savefig(WORK / "fig_seed_power.png", bbox_inches="tight")
plt.show()

for label, df in [("Men", m_sa), ("Women", w_sa)]:
    hr = (df["WSeed"] <= df["LSeed"]).mean()
    print(f"{label}: higher seed wins {hr:.1%} overall")
```

### 4.2 Home Court Advantage

```python
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, rs, label in [(axes[0], m_rs, "Men"), (axes[1], w_rs, "Women")]:
    home_pct = rs.groupby("Season").apply(
        lambda x: (x["WLoc"] == "H").mean()
    ).reset_index(name="HomePct")
    ax.plot(home_pct["Season"], home_pct["HomePct"], color=PALETTE[0], linewidth=1.5)
    ax.fill_between(home_pct["Season"], 0.5, home_pct["HomePct"], alpha=0.15, color=PALETTE[0])
    ax.axhline(y=0.5, color="gray", linestyle=":", linewidth=0.8)
    ax.set_xlabel("Season")
    ax.set_ylabel("Fraction of Wins at Home")
    ax.set_title(f"{label}: Home Court Advantage Over Time")
    ax.set_ylim(0.45, 0.70)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))

plt.tight_layout()
plt.savefig(WORK / "fig_home_court.png", bbox_inches="tight")
plt.show()
```

### 4.3 Scoring Trends Over Time

```python
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, rs, label in [(axes[0], m_rs, "Men"), (axes[1], w_rs, "Women")]:
    rs["TotalPts"] = rs["WScore"] + rs["LScore"]
    rs["Margin"] = rs["WScore"] - rs["LScore"]
    by_season = rs.groupby("Season").agg(
        avg_total=("TotalPts", "mean"), avg_margin=("Margin", "mean")
    ).reset_index()
    ax.plot(by_season["Season"], by_season["avg_total"], color=PALETTE[0], linewidth=1.5, label="Avg Total Points")
    ax2 = ax.twinx()
    ax2.plot(by_season["Season"], by_season["avg_margin"], color=PALETTE[1], linewidth=1.5, linestyle="--", label="Avg Margin")
    ax.set_xlabel("Season")
    ax.set_ylabel("Average Total Points", color=PALETTE[0])
    ax2.set_ylabel("Average Margin", color=PALETTE[1])
    ax.set_title(f"{label}: Scoring Trends")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=9)

plt.tight_layout()
plt.savefig(WORK / "fig_scoring_trends.png", bbox_inches="tight")
plt.show()
```

### 4.4 Conference Landscape (2025 Men)

```python
conf_25 = m_conf[m_conf["Season"] == 2025]
seeds_25 = m_seeds[m_seeds["Season"] == 2025]
conf_seeds = seeds_25.merge(conf_25, on="TeamID")
conf_agg = conf_seeds.groupby("ConfAbbrev").agg(
    count=("TeamID", "count"), avg_seed=("SeedNum", "mean")
).sort_values("count", ascending=True).tail(8).reset_index()

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.barh(conf_agg["ConfAbbrev"], conf_agg["count"], color=PALETTE[0], edgecolor="white")
for bar, seed in zip(bars, conf_agg["avg_seed"]):
    ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2,
            f"avg seed: {seed:.1f}", va="center", fontsize=9, color=PALETTE[3])
ax.set_xlabel("Teams in 2025 Tournament")
ax.set_title("Conference Representation in the 2025 NCAA Tournament (Men)")
plt.tight_layout()
plt.savefig(WORK / "fig_conf_landscape.png", bbox_inches="tight")
plt.show()
```

### 4.5 Upset Frequency by Matchup Type

```python
def upset_by_gap(trn_df, seed_df, label):
    df = seed_analysis(trn_df, seed_df, label)
    df["Upset"] = df["WSeed"] > df["LSeed"]
    bins = [(1, 3, "1-3"), (4, 7, "4-7"), (8, 11, "8-11"), (12, 15, "12-15")]
    rows = []
    for lo, hi, lbl in bins:
        mask = abs(df["WSeed"] - df["LSeed"]).between(lo, hi)
        if mask.sum() > 0:
            rows.append({"Gap": lbl, "UpsetRate": df.loc[mask, "Upset"].mean(), "N": mask.sum()})
    return pd.DataFrame(rows)

m_upsets = upset_by_gap(m_trn, m_seeds, "Men")
w_upsets = upset_by_gap(w_trn, w_seeds, "Women")

fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(len(m_upsets))
w_ = 0.35
ax.bar(x - w_/2, m_upsets["UpsetRate"], w_, label="Men", color=PALETTE[0], edgecolor="white")
ax.bar(x + w_/2, w_upsets["UpsetRate"], w_, label="Women", color=PALETTE[1], edgecolor="white")
ax.set_xticks(x)
ax.set_xticklabels(m_upsets["Gap"])
ax.set_xlabel("Seed Difference Range")
ax.set_ylabel("Upset Rate")
ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
ax.set_title("Upset Rates by Seed Difference")
ax.legend()
plt.tight_layout()
plt.savefig(WORK / "fig_upsets.png", bbox_inches="tight")
plt.show()
```

Key takeaways from the EDA:

- **Seed difference is the single strongest predictor**, with higher seeds winning over 90% of 
games when the gap exceeds 10 lines. The women's tournament is notably more chalk than the men's.
- **Home court advantage** has been consistent at around 58-60% across decades.
- **Scoring has fluctuated** meaningfully over the years, making raw point averages less reliable 
than efficiency-adjusted metrics.
- **Conference quality matters**: the SEC, Big Ten, and Big 12 dominate tournament representation, 
with the SEC alone placing 14 teams in 2025.
- **Upsets are rare when seed gaps are large** but quite common in matchups between similarly seeded teams.

These findings motivate our feature engineering choices below.

<a id="5-feature-engineering"></a>
## 5. Feature Engineering

We construct six categories of features for each team-season, then take pairwise differences for matchup prediction.

### 5.1 Elo Rating System

We implement a standard Elo system with three enhancements: margin-of-victory scaling 
(log-transformed), home court adjustment, and between-season mean reversion at 75%.

```python
def build_elo(rs_df, trn_df, sec_df, K=20, HOME=100, REV=0.75):
    rs = rs_df[["Season","DayNum","WTeamID","WScore","LTeamID","LScore","WLoc"]].copy()
    rs["it"] = 0
    tn = trn_df[["Season","DayNum","WTeamID","WScore","LTeamID","LScore","WLoc"]].copy()
    tn["it"] = 1
    sc = sec_df[["Season","DayNum","WTeamID","WScore","LTeamID","LScore","WLoc"]].copy()
    sc["it"] = 1
    games = pd.concat([rs, tn, sc]).sort_values(["Season", "DayNum"]).values

    elo = {}
    snap = {}
    prev = None

    for r in games:
        s, d = int(r[0]), int(r[1])
        wi, ws, li, ls = int(r[2]), float(r[3]), int(r[4]), float(r[5])
        wl, it = r[6], int(r[7])

        if s != prev and prev is not None:
            for t in elo:
                snap[(prev, t)] = elo[t]
            for t in elo:
                elo[t] = 1500 * (1 - REV) + elo[t] * REV
        prev = s

        elo.setdefault(wi, 1500)
        elo.setdefault(li, 1500)
        if d <= 132:
            snap[(s, wi)] = elo[wi]
            snap[(s, li)] = elo[li]

        ha = HOME if wl == "H" else (-HOME if wl == "A" else 0)
        we = 1.0 / (1.0 + 10.0 ** ((elo[li] - elo[wi] - ha) / 400.0))
        k = K * np.log(max(abs(ws - ls), 1) + 1)
        elo[wi] += k * (1 - we)
        elo[li] -= k * (1 - we)

    for t in elo:
        snap[(prev, t)] = elo[t]
    return snap

m_elo = build_elo(m_rs, m_trn, m_sec)
w_elo = build_elo(w_rs, w_trn, w_sec)
print(f"Elo ratings computed: Men {len(m_elo):,} | Women {len(w_elo):,}")
```

### 5.2 Four Factors and Advanced Season Statistics

Dean Oliver's Four Factors framework captures the dimensions that most strongly correlate with winning: 
effective field goal percentage (shooting), turnover rate (ball security), offensive rebounding 
percentage (second chances), and free throw rate (getting to the line). We compute these for both 
offense and defense, plus additional metrics like pace, assist rate, and rim protection.

```python
def compute_stats(det_df):
    cw = {
        "WTeamID": "T", "WScore": "Pts", "LScore": "oP",
        "WFGM": "FGM", "WFGA": "FGA", "WFGM3": "F3M", "WFGA3": "F3A",
        "WFTM": "FTM", "WFTA": "FTA", "WOR": "OR", "WDR": "DR",
        "WAst": "Ast", "WTO": "TO", "WStl": "Stl", "WBlk": "Blk",
        "LFGM": "oFGM", "LFGA": "oFGA", "LFGM3": "oF3M", "LFGA3": "oF3A",
        "LFTM": "oFTM", "LFTA": "oFTA", "LOR": "oOR", "LDR": "oDR",
        "LAst": "oAst", "LTO": "oTO"
    }
    cl = {
        "LTeamID": "T", "LScore": "Pts", "WScore": "oP",
        "LFGM": "FGM", "LFGA": "FGA", "LFGM3": "F3M", "LFGA3": "F3A",
        "LFTM": "FTM", "LFTA": "FTA", "LOR": "OR", "LDR": "DR",
        "LAst": "Ast", "LTO": "TO", "LStl": "Stl", "LBlk": "Blk",
        "WFGM": "oFGM", "WFGA": "oFGA", "WFGM3": "oF3M", "WFGA3": "oF3A",
        "WFTM": "oFTM", "WFTA": "oFTA", "WOR": "oOR", "WDR": "oDR",
        "WAst": "oAst", "WTO": "oTO"
    }
    wr = det_df.rename(columns=cw); wr["W"] = 1
    lr = det_df.rename(columns=cl); lr["W"] = 0
    kp = ["Season", "T", "Pts", "oP", "FGM", "FGA", "F3M", "F3A",
          "FTM", "FTA", "OR", "DR", "Ast", "TO", "Stl", "Blk",
          "oFGM", "oFGA", "oF3M", "oF3A", "oFTM", "oFTA", "oOR", "oDR",
          "oAst", "oTO", "W"]
    ag = pd.concat([wr[kp], lr[kp]]).groupby(["Season", "T"]).agg(
        n=("W", "count"), wins=("W", "sum"),
        pts=("Pts", "sum"), ops=("oP", "sum"),
        fgm=("FGM", "sum"), fga=("FGA", "sum"),
        f3m=("F3M", "sum"), f3a=("F3A", "sum"),
        ftm=("FTM", "sum"), fta=("FTA", "sum"),
        orb=("OR", "sum"), drb=("DR", "sum"),
        ast=("Ast", "sum"), to=("TO", "sum"),
        stl=("Stl", "sum"), blk=("Blk", "sum"),
        ofgm=("oFGM", "sum"), ofga=("oFGA", "sum"),
        of3m=("oF3M", "sum"), of3a=("oF3A", "sum"),
        oftm=("oFTM", "sum"), ofta=("oFTA", "sum"),
        oorb=("oOR", "sum"), odrb=("oDR", "sum"),
        oast=("oAst", "sum"), oto=("oTO", "sum")
    ).reset_index()

    a = ag
    a["wpct"]   = a["wins"] / a["n"]
    a["margin"] = (a["pts"] - a["ops"]) / a["n"]
    a["poss"]   = a["fga"] - a["orb"] + a["to"] + 0.475 * a["fta"]
    a["oposs"]  = a["ofga"] - a["oorb"] + a["oto"] + 0.475 * a["ofta"]
    a["oeff"]   = a["pts"] / a["poss"].clip(1) * 100
    a["deff"]   = a["ops"] / a["oposs"].clip(1) * 100
    a["neff"]   = a["oeff"] - a["deff"]
    a["efg"]    = (a["fgm"] + 0.5 * a["f3m"]) / a["fga"].clip(1)
    a["tor"]    = a["to"] / a["poss"].clip(1)
    a["orpct"]  = a["orb"] / (a["orb"] + a["odrb"]).clip(1)
    a["ftr"]    = a["ftm"] / a["fga"].clip(1)
    a["oefg"]   = (a["ofgm"] + 0.5 * a["of3m"]) / a["ofga"].clip(1)
    a["otor"]   = a["oto"] / a["oposs"].clip(1)
    a["oorpct"] = a["oorb"] / (a["oorb"] + a["drb"]).clip(1)
    a["oftr"]   = a["oftm"] / a["ofga"].clip(1)
    a["f3pct"]  = a["f3m"] / a["f3a"].clip(1)
    a["of3pct"] = a["of3m"] / a["of3a"].clip(1)
    a["astr"]   = a["ast"] / a["fgm"].clip(1)
    a["stlpg"]  = a["stl"] / a["n"]
    a["blkpg"]  = a["blk"] / a["n"]
    a["drbpg"]  = a["drb"] / a["n"]
    a["pace"]   = (a["poss"] + a["oposs"]) / (2 * a["n"])

    sf = ["wpct", "margin", "oeff", "deff", "neff", "efg", "tor", "orpct", "ftr",
          "oefg", "otor", "oorpct", "oftr", "f3pct", "of3pct", "astr",
          "stlpg", "blkpg", "drbpg", "pace"]
    return a[["Season", "T"] + sf], sf

def basic_stats(cdf):
    w = cdf[["Season", "WTeamID", "WScore", "LScore"]].copy()
    w.columns = ["Season", "T", "P", "oP"]; w["W"] = 1
    l = cdf[["Season", "LTeamID", "LScore", "WScore"]].copy()
    l.columns = ["Season", "T", "P", "oP"]; l["W"] = 0
    a = pd.concat([w, l]).groupby(["Season", "T"]).agg(
        n=("W", "count"), wins=("W", "sum"), pts=("P", "mean"), ops=("oP", "mean")
    ).reset_index()
    a["wpct"] = a["wins"] / a["n"]
    a["margin"] = a["pts"] - a["ops"]
    return a[["Season", "T", "wpct", "margin"]]

m_st, SF = compute_stats(m_det)
w_st, _  = compute_stats(w_det)
m_bas = basic_stats(m_rs)
w_bas = basic_stats(w_rs)
print(f"Season stats: {len(SF)} features per team")
print(f"Men detailed: {len(m_st):,} team-seasons | Women detailed: {len(w_st):,}")
```

### 5.3 Strength of Schedule

Average opponent Elo across all regular season games. This helps distinguish teams that compiled 
records against weak opponents from those tested by strong schedules.

```python
def compute_sos(cdf, elo_d):
    w = cdf[["Season", "WTeamID", "LTeamID"]].rename(columns={"WTeamID": "T", "LTeamID": "O"})
    l = cdf[["Season", "LTeamID", "WTeamID"]].rename(columns={"LTeamID": "T", "WTeamID": "O"})
    a = pd.concat([w, l])
    a["oe"] = a.apply(lambda r: elo_d.get((r["Season"], r["O"]), 1500), axis=1)
    return a.groupby(["Season", "T"])["oe"].mean().reset_index().rename(columns={"oe": "sos"})

m_sos = compute_sos(m_rs, m_elo)
w_sos = compute_sos(w_rs, w_elo)
print(f"SOS computed: Men {len(m_sos):,} | Women {len(w_sos):,}")
```

### 5.4 Late-Season Momentum

Win rate in the final 10 regular season games captures recent form.

```python
def momentum(cdf, N=10):
    w = cdf[["Season", "DayNum", "WTeamID"]].rename(columns={"WTeamID": "T"}); w["W"] = 1
    l = cdf[["Season", "DayNum", "LTeamID"]].rename(columns={"LTeamID": "T"}); l["W"] = 0
    a = pd.concat([w, l]).sort_values(["Season", "T", "DayNum"])
    return (a.groupby(["Season", "T"]).tail(N)
             .groupby(["Season", "T"])["W"].mean()
             .reset_index().rename(columns={"W": "mom"}))

m_mom = momentum(m_rs)
w_mom = momentum(w_rs)
print(f"Momentum: Men {len(m_mom):,} | Women {len(w_mom):,}")
```

### 5.5 Conference Strength

Mean Elo of all teams in each conference, by season.

```python
def conf_str(conf_df, elo_d):
    rows = []
    for (s, c), g in conf_df.groupby(["Season", "ConfAbbrev"]):
        el = [elo_d.get((s, t), 1500) for t in g["TeamID"]]
        rows.append({"Season": s, "CA": c, "ce": np.mean(el)})
    return pd.DataFrame(rows)

m_cs = conf_str(m_conf, m_elo)
w_cs = conf_str(w_conf, w_elo)
print(f"Conference strength: Men {len(m_cs):,} | Women {len(w_cs):,}")
```

### 5.6 Coach Experience (Men Only)

Two features: cumulative NCAA tournament appearances for the head coach, 
and tenure (consecutive seasons) at the current program.

```python
ce = (m_coach.sort_values(["Season", "TeamID", "LastDayNum"])
      .groupby(["Season", "TeamID"]).last().reset_index())
tt = pd.concat([
    m_trn[["Season", "WTeamID"]].rename(columns={"WTeamID": "TeamID"}),
    m_trn[["Season", "LTeamID"]].rename(columns={"LTeamID": "TeamID"})
]).drop_duplicates()
ct = (tt.merge(ce[["Season", "TeamID", "CoachName"]], on=["Season", "TeamID"], how="left")
      .sort_values("Season"))
ct["cexp"] = ct.groupby("CoachName").cumcount()

ce2 = ce.sort_values(["TeamID", "Season"])
ce2["nw"] = (ce2["CoachName"] != ce2.groupby("TeamID")["CoachName"].shift(1)).astype(int)
ce2["grp"] = ce2.groupby("TeamID")["nw"].cumsum()
ce2["tenure"] = ce2.groupby(["TeamID", "grp"]).cumcount() + 1

clk = {}
for _, r in ct[["Season", "TeamID", "cexp"]].drop_duplicates().iterrows():
    clk[(int(r["Season"]), int(r["TeamID"]), "e")] = r["cexp"]
for _, r in ce2[["Season", "TeamID", "tenure"]].iterrows():
    clk[(int(r["Season"]), int(r["TeamID"]), "t")] = r["tenure"]

print(f"Coach records: {len(clk):,}")
```

<a id="6-model-training"></a>
## 6. Model Training

We train three gradient boosting models independently, then average their outputs. 
The validation set comprises tournament games from 2022 through 2025.

### 6.1 Training Data Construction

```python
def d2d(df, kc, vc):
    d = {}
    for _, r in df.iterrows():
        d[tuple(int(r[c]) for c in kc)] = {c: r[c] for c in vc}
    return d

m_sd = d2d(m_st, ["Season", "T"], SF)
w_sd = d2d(w_st, ["Season", "T"], SF)
m_bd = d2d(m_bas, ["Season", "T"], ["wpct", "margin"])
w_bd = d2d(w_bas, ["Season", "T"], ["wpct", "margin"])
m_seed_d = dict(zip(zip(m_seeds.Season.astype(int), m_seeds.TeamID.astype(int)), m_seeds.SeedNum))
w_seed_d = dict(zip(zip(w_seeds.Season.astype(int), w_seeds.TeamID.astype(int)), w_seeds.SeedNum))
m_tc = dict(zip(zip(m_conf.Season.astype(int), m_conf.TeamID.astype(int)), m_conf.ConfAbbrev))
w_tc = dict(zip(zip(w_conf.Season.astype(int), w_conf.TeamID.astype(int)), w_conf.ConfAbbrev))
m_csd = {}; w_csd = {}
for _, r in m_cs.iterrows(): m_csd[(int(r["Season"]), r["CA"])] = r["ce"]
for _, r in w_cs.iterrows(): w_csd[(int(r["Season"]), r["CA"])] = r["ce"]
m_sosd = dict(zip(zip(m_sos.Season.astype(int), m_sos.T.astype(int)), m_sos.sos))
w_sosd = dict(zip(zip(w_sos.Season.astype(int), w_sos.T.astype(int)), w_sos.sos))
m_momd = dict(zip(zip(m_mom.Season.astype(int), m_mom.T.astype(int)), m_mom.mom))
w_momd = dict(zip(zip(w_mom.Season.astype(int), w_mom.T.astype(int)), w_mom.mom))

print("Lookup dictionaries built.")
```

```python
def bfeat(s, t1, t2, iw):
    ed   = w_elo if iw else m_elo
    sd_  = w_sd if iw else m_sd
    bd   = w_bd if iw else m_bd
    sed  = w_seed_d if iw else m_seed_d
    sosd = w_sosd if iw else m_sosd
    momd = w_momd if iw else m_momd
    tcd  = w_tc if iw else m_tc
    csd  = w_csd if iw else m_csd

    f = {}
    e1, e2 = ed.get((s, t1), 1500), ed.get((s, t2), 1500)
    f["elo_d"] = e1 - e2
    f["elo1"]  = e1
    f["elo2"]  = e2

    s1 = sed.get((s, t1), SEED_DEFAULT)
    s2 = sed.get((s, t2), SEED_DEFAULT)
    f["seed_d"] = s2 - s1
    f["s1"] = s1
    f["s2"] = s2

    st1 = sd_.get((s, t1), bd.get((s, t1), {}))
    st2 = sd_.get((s, t2), bd.get((s, t2), {}))
    for c in SF:
        f[c + "_d"] = st1.get(c, 0) - st2.get(c, 0)
    f["neff1"] = st1.get("neff", 0)
    f["neff2"] = st2.get("neff", 0)

    f["sos_d"]  = sosd.get((s, t1), 1500) - sosd.get((s, t2), 1500)
    f["mom_d"]  = momd.get((s, t1), 0.5)  - momd.get((s, t2), 0.5)

    c1 = tcd.get((s, t1))
    c2 = tcd.get((s, t2))
    f["conf_d"] = (csd.get((s, c1), 1500) if c1 else 1500) - (csd.get((s, c2), 1500) if c2 else 1500)

    if not iw:
        f["cexp_d"] = clk.get((s, t1, "e"), 0) - clk.get((s, t2, "e"), 0)
        f["cten_d"] = clk.get((s, t1, "t"), 1) - clk.get((s, t2, "t"), 1)
    else:
        f["cexp_d"] = 0
        f["cten_d"] = 0
    return f


def build_train(trn_df, iw, ms):
    rows = []
    for _, g in trn_df.iterrows():
        s = int(g["Season"])
        if s < ms:
            continue
        wi, li = int(g["WTeamID"]), int(g["LTeamID"])
        t1, t2 = min(wi, li), max(wi, li)
        y = 1 if wi == t1 else 0
        f = bfeat(s, t1, t2, iw)
        f["season"] = s
        f["target"] = y
        f["iw"] = int(iw)
        rows.append(f)
    return pd.DataFrame(rows)

m_train = build_train(m_trn, False, 2003)
w_train = build_train(w_trn, True, 2010)
train_all = pd.concat([m_train, w_train], ignore_index=True)
FC = [c for c in train_all.columns if c not in ["season", "target", "iw"]]

print(f"Training set: {len(train_all):,} matchups, {len(FC)} features")
print(f"  Men:   {len(m_train)} (seasons {m_train['season'].min()}-{m_train['season'].max()})")
print(f"  Women: {len(w_train)} (seasons {w_train['season'].min()}-{w_train['season'].max()})")
```

```python
val_seasons = [2022, 2023, 2024, 2025]
tr = ~train_all["season"].isin(val_seasons)
vl = train_all["season"].isin(val_seasons)

Xtr = train_all.loc[tr, FC].fillna(0)
ytr = train_all.loc[tr, "target"].values
Xvl = train_all.loc[vl, FC].fillna(0)
yvl = train_all.loc[vl, "target"].values
vl_seasons = train_all.loc[vl, "season"].values
vl_iw = train_all.loc[vl, "iw"].values

print(f"Train: {len(Xtr)} | Validation: {len(Xvl)}")
```

### 6.2 LightGBM

```python
lgb_model = lgb.LGBMClassifier(
    n_estimators=2000, learning_rate=0.02, num_leaves=31,
    min_child_samples=15, subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=1.0, random_state=RS, verbosity=-1
)
lgb_model.fit(
    Xtr, ytr, eval_set=[(Xvl, yvl)],
    callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(0)]
)
lgb_pred = lgb_model.predict_proba(Xvl)[:, 1]
print(f"LightGBM  Brier: {brier_score_loss(yvl, lgb_pred):.5f}  (iterations: {lgb_model.best_iteration_})")
```

### 6.3 XGBoost

```python
xgb_model = xgb.XGBClassifier(
    n_estimators=2000, learning_rate=0.02, max_depth=5,
    min_child_weight=15, subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=1.0, random_state=RS,
    tree_method="hist", verbosity=0, early_stopping_rounds=200
)
xgb_model.fit(Xtr, ytr, eval_set=[(Xvl, yvl)], verbose=False)
xgb_pred = xgb_model.predict_proba(Xvl)[:, 1]
print(f"XGBoost   Brier: {brier_score_loss(yvl, xgb_pred):.5f}  (iterations: {xgb_model.best_iteration})")
```

### 6.4 CatBoost

```python
cat_model = CatBoostClassifier(
    iterations=2000, learning_rate=0.02, depth=5,
    min_data_in_leaf=15, l2_leaf_reg=3.0, subsample=0.8,
    random_seed=RS, verbose=0, task_type="CPU",
    eval_metric="Logloss", early_stopping_rounds=200
)
cat_model.fit(Xtr, ytr, eval_set=(Xvl, yvl), verbose=0)
cat_pred = cat_model.predict_proba(Xvl)[:, 1]
print(f"CatBoost  Brier: {brier_score_loss(yvl, cat_pred):.5f}  (iterations: {cat_model.best_iteration_})")
```

### 6.5 Ensemble and Evaluation

```python
ens_pred = np.clip((lgb_pred + xgb_pred + cat_pred) / 3.0, CLIP_LO, CLIP_HI)

print("=" * 55)
print(f"  ENSEMBLE   Brier: {brier_score_loss(yvl, ens_pred):.5f}   LogLoss: {log_loss(yvl, ens_pred):.5f}")
print("=" * 55)

results = []
for s in val_seasons:
    for w, lbl in [(0, "Men"), (1, "Women")]:
        mk = (vl_seasons == s) & (vl_iw == w)
        if mk.sum() > 0:
            b = brier_score_loss(yvl[mk], ens_pred[mk])
            results.append({"Season": s, "Gender": lbl, "Brier": round(b, 5), "Games": mk.sum()})

display(pd.DataFrame(results).style.format({"Brier": "{:.5f}"}).hide(axis="index"))
```

<a id="7-diagnostics-and-calibration"></a>
## 7. Diagnostics and Calibration

```python
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

imp_df = pd.DataFrame({"Feature": FC, "Importance": lgb_model.feature_importances_})
imp_df = imp_df.sort_values("Importance", ascending=True).tail(15)
axes[0].barh(imp_df["Feature"], imp_df["Importance"], color=PALETTE[0], edgecolor="white")
axes[0].set_title("Top 15 Features (LightGBM)")
axes[0].set_xlabel("Importance")

frac_pos, mean_pred = calibration_curve(yvl, ens_pred, n_bins=10, strategy="quantile")
axes[1].plot(mean_pred, frac_pos, "o-", color=PALETTE[0], linewidth=1.5, markersize=5)
axes[1].plot([0, 1], [0, 1], "--", color="gray", linewidth=0.8)
axes[1].set_xlabel("Predicted Probability")
axes[1].set_ylabel("Observed Win Rate")
axes[1].set_title("Calibration Plot")
axes[1].set_xlim(0, 1)
axes[1].set_ylim(0, 1)

axes[2].hist(ens_pred, bins=30, color=PALETTE[0], edgecolor="white", alpha=0.85)
axes[2].set_xlabel("Predicted Probability")
axes[2].set_ylabel("Count")
axes[2].set_title("Prediction Distribution (Validation)")

plt.tight_layout()
plt.savefig(WORK / "fig_diagnostics.png", bbox_inches="tight")
plt.show()
```

<a id="8-submission-generation"></a>
## 8. Submission Generation

We retrain all three models on the full training set using the best iteration counts from 
early stopping, then generate predictions for both Stage 1 and Stage 2 submission files.

```python
Xa = train_all[FC].fillna(0)
ya = train_all["target"].values

lgb_full = lgb.LGBMClassifier(
    n_estimators=lgb_model.best_iteration_, learning_rate=0.02, num_leaves=31,
    min_child_samples=15, subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=1.0, random_state=RS, verbosity=-1
)
lgb_full.fit(Xa, ya)

xgb_full = xgb.XGBClassifier(
    n_estimators=xgb_model.best_iteration, learning_rate=0.02, max_depth=5,
    min_child_weight=15, subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=1.0, random_state=RS,
    tree_method="hist", verbosity=0
)
xgb_full.fit(Xa, ya)

cat_full = CatBoostClassifier(
    iterations=cat_model.best_iteration_, learning_rate=0.02, depth=5,
    min_data_in_leaf=15, l2_leaf_reg=3.0, subsample=0.8,
    random_seed=RS, verbose=0, task_type="CPU"
)
cat_full.fit(Xa, ya)

print(f"Retrained: LGB({lgb_model.best_iteration_}), XGB({xgb_model.best_iteration}), CAT({cat_model.best_iteration_})")
```

```python
def predict_sub(sub_df, desc):
    parts = sub_df["ID"].str.split("_", expand=True).astype(int)
    ss = parts[0].values
    t1s = parts[1].values
    t2s = parts[2].values
    n = len(sub_df)
    BATCH = 50000
    all_preds = np.zeros(n)

    for start in range(0, n, BATCH):
        end = min(start + BATCH, n)
        rows = []
        for i in range(start, end):
            s, t1, t2 = int(ss[i]), int(t1s[i]), int(t2s[i])
            rows.append(bfeat(s, t1, t2, t1 >= 3000))

        X = pd.DataFrame(rows)[FC].fillna(0).values
        p1 = lgb_full.predict_proba(X)[:, 1]
        p2 = xgb_full.predict_proba(X)[:, 1]
        p3 = cat_full.predict_proba(X)[:, 1]
        all_preds[start:end] = np.clip((p1 + p2 + p3) / 3, CLIP_LO, CLIP_HI)

        if end % 100000 < BATCH or end == n:
            print(f"  {desc}: {end:>8,} / {n:,}")

    return all_preds

sub1["Pred"] = predict_sub(sub1, "Stage 1")
sub1[["ID", "Pred"]].to_csv(WORK / "submission_stage1.csv", index=False, float_format="%.6f")
print(f"\nSaved: submission_stage1.csv ({len(sub1):,} rows)")

sub2["Pred"] = predict_sub(sub2, "Stage 2")
sub2[["ID", "Pred"]].to_csv(WORK / "submission_stage2.csv", index=False, float_format="%.6f")
print(f"Saved: submission_stage2.csv ({len(sub2):,} rows)")

print(f"\nStage 2 distribution:")
print(sub2["Pred"].describe().to_frame().T.to_string(index=False))

elapsed = time.time() - t_start
print(f"\nTotal runtime: {elapsed / 60:.1f} minutes")
```

<a id="9-summary-and-next-steps"></a>
## 9. Summary and Next Steps

This notebook constructs a baseline ensemble for the 2026 March Machine Learning Mania competition. 
The approach combines Elo ratings, Four Factors efficiency metrics, strength-of-schedule, 
late-season momentum, conference strength, and coaching experience into an ensemble of three 
gradient boosting models (LightGBM, XGBoost, CatBoost).

**Potential improvements:**

- Incorporate Massey Ordinals (MasseyOrdinals.csv) for access to 100+ public ranking systems 
including KenPom efficiency metrics, Sagarin ratings, and BPI.
- Experiment with opponent-adjusted statistics (per-possession metrics adjusted for schedule).
- Add a spline-calibrated point spread model as described in public literature.
- Explore leave-one-season-out cross validation for more robust hyperparameter tuning.
- Investigate separate models for men and women with gender-specific feature selection.
- Weight recent seasons more heavily during training.

The competition ultimately hinges on Stage 2 performance. When Kaggle releases updated data in 
mid-March, the pipeline should be re-executed to capture the full 2025-26 regular season.