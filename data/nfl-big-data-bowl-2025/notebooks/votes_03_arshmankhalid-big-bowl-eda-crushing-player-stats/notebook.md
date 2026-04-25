# Big Bowl EDA: Crushing Player Stats! 📈💥

- **Author:** Arshman Khalid
- **Votes:** 102
- **Ref:** arshmankhalid/big-bowl-eda-crushing-player-stats
- **URL:** https://www.kaggle.com/code/arshmankhalid/big-bowl-eda-crushing-player-stats
- **Last run:** 2024-11-06 18:00:59.850000

---

<h1 style="background-color: orange ; color: white; font-family: 'Verdana', sans-serif; text-align: center; padding: 15px; border-radius: 10px; font-weight: bold; letter-spacing: 2px; text-transform: uppercase;">
  NFL Big Bowl Players Play Daat  EDA
</h>

![_6fd76158-7710-434f-ad3d-31540e2846a3.jpg](attachment:ee2430e5-5f70-41d0-9b06-6d62a37a8b70.jpg)

# **Arshman Khalid**  
<p style="font-size: 1.5rem; font-weight: bold;">Data Scientist | Software Engineer | ex Consultant PwC | ex Senior Data Analyst Fortune 500</p>

With over 5 years of expertise in data science and software engineering, I am dedicated to transforming complex data into actionable insights. My focus lies in predictive analytics, data strategy, and the implementation of robust machine learning models that drive measurable business outcomes. I have a track record of optimizing operations, reducing costs, and improving decision-making processes across industries. Proficient in Python, Alteryx, Power BI, and cloud platforms.

When I am not wrangling datasets, you will find me attempting to code my way to the perfect cup of coffee!

# **Lets Connect**

<div style="text-align: left; font-family: Arial, sans-serif; margin-top: 20px;">
    <a href="https://www.linkedin.com/in/arshmankhalid/" style="text-decoration: none; color: #fff; margin-right: 10px;">
        <span style="background-color: #0077B5; padding: 8px 20px; border-radius: 5px; font-size: 14px; display: inline-block; width: 120px; text-align: center;">LinkedIn</span>
    </a>
    <a href="https://x.com/arshmankhalid" style="text-decoration: none; color: #fff; margin-right: 10px;">
        <span style="background-color: #000; padding: 8px 20px; border-radius: 5px; font-size: 14px; display: inline-block; width: 120px; text-align: center;">X</span>
    </a>
    <a href="https://github.com/arshmankhalid88" style="text-decoration: none; color: #fff; margin-right: 10px;">
        <span style="background-color: #333; padding: 8px 20px; border-radius: 5px; font-size: 14px; display: inline-block; width: 120px; text-align: center;">GitHub</span>
    </a>
    <a href="https://www.kaggle.com/arshmankhalid" style="text-decoration: none; color: #fff; margin-right: 10px;">
        <span style="background-color: #20BEFF; padding: 8px 20px; border-radius: 5px; font-size: 14px; display: inline-block; width: 120px; text-align: center;">Kaggle</span>
    </a>
</div>

# Player Play Data

This dataset provides detailed information about individual player actions and outcomes during NFL plays. The fields capture player involvement in rushing, passing, defense, and blocking efforts, as well as situational play-by-play metrics.

### Columns

| Column Name                                | Description                                                                                                                                                                                    | Data Type |
|--------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------|
| `gameId`                                   | Unique identifier for each game.                                                                                                                                                              | `int`     |
| `playId`                                   | Play identifier, not unique across games.                                                                                                                                                      | `int`     |
| `nflId`                                    | Unique player identification number.                                                                                                                                                           | `int`     |
| `teamAbbr`                                 | Abbreviation for the team the player is on.                                                                                                                                                    | `str`     |
| `hadRushAttempt`                           | Indicator if the player had a rushing attempt on this play.                                                                                                                                    | `int`     |
| `rushingYards`                             | Rushing yards gained by the player on this play.                                                                                                                                               | `int`     |
| `hadDropback`                              | Indicator if the player dropped back on this play.                                                                                                                                             | `int`     |
| `passingYards`                             | Passing yards gained by the player on this play.                                                                                                                                               | `int`     |
| `sackYardsOffense`                         | Yards lost by the player due to a sack on this play.                                                                                                                                           | `int`     |
| `hadPassReception`                         | Indicator if the player caught a pass on this play.                                                                                                                                            | `int`     |
| `receivingYards`                           | Receiving yards gained by the player on this play.                                                                                                                                             | `int`     |
| `wasTargettedReceiver`                     | Indicator if the player was the intended receiver on this play.                                                                                                                                | `int`     |
| `yardageGainedAfterTheCatch`               | Yards gained after the catch on this play.                                                                                                                                                     | `int`     |
| `fumbles`                                  | Number of fumbles by the player on this play.                                                                                                                                                  | `int`     |
| `fumbleLost`                               | Indicator if the player lost a fumble to the opposing team.                                                                                                                                    | `int`     |
| `fumbleOutOfBounds`                        | Indicator if the player fumbled the ball out of bounds.                                                                                                                                        | `int`     |
| `assistedTackle`                           | Indicator if the player needed an assist to make a tackle on this play.                                                                                                                        | `int`     |
| `forcedFumbleAsDefense`                    | Indicator if the player forced a fumble by the opposing team.                                                                                                                                  | `int`     |
| `halfSackYardsAsDefense`                   | Yards conceded by the offense due to a half-sack by the player on this play.                                                                                                                   | `int`     |
| `passDefensed`                             | Indicator if the player stopped a passing play.                                                                                                                                                | `int`     |
| `quarterbackHit`                           | Indicator if the player hit the QB on this play.                                                                                                                                               | `int`     |
| `sackYardsAsDefense`                       | Yards conceded by the offense due to a sack by the player.                                                                                                                                     | `int`     |
| `safetyAsDefense`                          | Indicator if the player forced a safety.                                                                                                                                                       | `int`     |
| `soloTackle`                               | Indicator if the player made a solo tackle.                                                                                                                                                    | `int`     |
| `tackleAssist`                             | Indicator if the player assisted on a tackle.                                                                                                                                                  | `int`     |
| `tackleForALoss`                           | Indicator if the player recorded a tackle for a loss.                                                                                                                                          | `int`     |
| `tackleForALossYardage`                    | Yards lost due to the player's tackle behind the line of scrimmage.                                                                                                                            | `int`     |
| `hadInterception`                          | Indicator if the player intercepted a pass.                                                                                                                                                    | `int`     |
| `interceptionYards`                        | Yards gained on an interception.                                                                                                                                                              | `int`     |
| `fumbleRecoveries`                         | Number of fumbles recovered by the player.                                                                                                                                                     | `int`     |
| `fumbleRecoveryYards`                      | Yards gained from a fumble recovery.                                                                                                                                                          | `int`     |
| `wasInitialPassRusher`                     | Indicator if the player was the initial pass rusher.                                                                                                                                           | `int`     |
| `penaltyNames`                             | List of penalties called on the player during the play.                                                                                                                                        | `str`     |
| `causedPressure`                           | `True` if the player pressured the QB with a peak probability ≥ 0.75 during a dropback.                                                                                                        | `bool`    |
| `timeToPressureAsPassRusher`               | Time (in seconds) from snap to when pressure was applied (probability ≥ 0.75).                                                                                                                 | `float`   |
| `getOffAsPassRusher`                       | Time (in seconds) for the player to cross the line of scrimmage as a pass rusher after the snap.                                                                                               | `float`   |
| `inMotionAtBallSnap`                       | `True` if the player was in motion at the snap (moved >1.2 y in the 0.4 s before the snap at a speed >0.62 y/s).                                                                               | `bool`    |
| `shiftSinceLineset`                        | `True` if the player moved more than 2.5 yards from their position at the lineset moment.                                                                                                      | `bool`    |
| `motionSinceLineset`                       | `True` if the player went in motion after the lineset.                                                                                                                                         | `bool`    |
| `wasRunningRoute`                          | `True` if the player was running a route.                                                                                                                                                      | `bool`    |
| `routeRan`                                 | Name of the route run by the player.                                                                                                                                                           | `str`     |
| `blockedPlayerNFLId1`, `blockedPlayerNFLId2`, `blockedPlayerNFLId3` | NFL IDs of opponents being blocked.                                                                                                       | `int`     |
| `pressureAllowedAsBlocker`                 | Indicator if the blocker allowed pressure from any pass rusher they matched against.                                                                                                           | `int`     |
| `timeToPressureAllowedAsBlocker`           | Time (in seconds) from snap to when a rusher achieved a probability >0.75 pressure against the blocker.                                                                                        | `float`   |
| `pff_defensiveCoverageAssignment`          | Defensive coverage assigned to the player on this play (MAN, 2R, 2L, etc.).                                                                                                                    | `str`     |
| `pff_primaryDefensiveCoverageMatchupNflId` | NFL ID of the primary opponent in coverage.                                                                                                                                                    | `int`     |
| `pff_secondaryDefensiveCoverageMatchupNflId` | NFL ID of the secondary opponent in coverage.                                                                                                         | `int`     |

<h1 style="background-color: orange ; color: white; font-family: 'Verdana', sans-serif; text-align: center; padding: 15px; border-radius: 10px; font-weight: bold; letter-spacing: 2px; text-transform: uppercase;">
  1.1 Importing The libraries
</h1>

```python
#  Importing the basic libraries 
import numpy as np 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
from IPython.display import display, HTML
import pandas as pd
import plotly.express as px
import plotly.io as pio
pio.renderers.default = 'iframe'
import warnings
warnings.filterwarnings('ignore')
```

```python
df= pd.read_csv('/kaggle/input/nfl-big-data-bowl-2025/player_play.csv')
```

<h1 style="background-color: orange ; color: white; font-family: 'Verdana', sans-serif; text-align: center; padding: 15px; border-radius: 10px; font-weight: bold; letter-spacing: 2px; text-transform: uppercase;">
  1.2 Quick Dataoverview 
</h1>

```python
def styled_heading(text, background_color='#ff6f61', text_color='white'):
    return f"""
    <p style="
        background-color: {background_color};
        font-family: Pacifico, cursive;
        font-size: 150%;
        color: {text_color};
        text-align: center;
        border-radius: 10px;
        padding: 10px;
        font-weight: normal;
        box-shadow: 0px 2px 4px rgba(0, 0, 0, 0.2);
        width: fit-content;
        margin: 0 auto;
    ">
        {text}
    </p>
    """

def style_table(df):
    styled_df = df.style.set_table_styles([
        {"selector": "th", "props": [("color", "white"), ("background-color", "#ff6f61")]},
        {"selector": "td", "props": [("background-color", "#e3f2fd"), ("color", "#333333")]}  # Light blue for table cells
    ]).set_properties(**{"text-align": "center"}).hide(axis="index")
    return styled_df.to_html()

def print_dataset_analysis(train_dataset, n_top=5, heading_color='#ff6f61', text_color='white'):
    train_heading = styled_heading(f"📊 Basic Overview of Data", heading_color, text_color)
    display(HTML(train_heading))
    display(HTML(style_table(train_dataset.head(n_top))))

    summary_heading = styled_heading("🔍 Data Summary", heading_color, text_color)
    display(HTML(summary_heading))
    display(HTML(style_table(train_dataset.describe())))

    null_heading = styled_heading("🚫 Null Values in Data", heading_color, text_color)
    train_null_count = train_dataset.isnull().sum()
    display(HTML(null_heading))
    if train_null_count.sum() == 0:
        display(HTML("<p>No null values in the dataset.</p>"))
    else:
        display(HTML("<h3>Null Values:</h3>"))
        display(HTML(style_table(train_null_count[train_null_count > 0].to_frame())))
        display(HTML("<p>These are the null values.</p>"))

    duplicate_heading = styled_heading("♻️ Duplicate Values in Data", heading_color, text_color)
    train_duplicates = train_dataset.duplicated().sum()
    display(HTML(duplicate_heading))
    display(HTML("<h3>Duplicates:</h3>"))
    display(HTML(f"<p>{train_duplicates} duplicate rows found.</p>"))

    shape_heading = styled_heading("📏 Data Shape", heading_color, text_color)
    display(HTML(shape_heading))
    display(HTML("<h3>Shape:</h3>"))
    display(HTML(f"<p>Rows: {train_dataset.shape[0]}, Columns: {train_dataset.shape[1]}</p>"))

def print_unique_values(train_dataset, heading_color='#ff6f61', text_color='white'):
    unique_values_heading = styled_heading("🔢 Unique Values in Data", heading_color, text_color)
    display(HTML(unique_values_heading))
    unique_values_table = pd.DataFrame({
        'Column Name': train_dataset.columns,
        'Data Type': [train_dataset[col].dtype for col in train_dataset.columns],
        'Unique Values': [', '.join(map(str, train_dataset[col].unique()[:7])) for col in train_dataset.columns]
    })
    display(HTML(style_table(unique_values_table)))
```

```python
print_dataset_analysis(df , n_top=5, heading_color='#14adc6', text_color='white')
print_unique_values(df , heading_color='#14adc6', text_color='white')
```

```python
# removing the columns containg 80 percent null values 
def null_drop(df , threshold=0.8):
    null_values=df.isnull().mean()
    drop_column=null_values[null_values>threshold].index
    return df.drop(columns=drop_column)
```

```python
df1 = null_drop(df) 
df1.isnull().sum()
```

```python
df1.info()
```

<h1 style="background-color: orange ; color: white; font-family: 'Verdana', sans-serif; text-align: center; padding: 15px; border-radius: 10px; font-weight: bold; letter-spacing: 2px; text-transform: uppercase;">
  1.3 Analyze Rushing And passing effectiveness
</h1>

```python
# Calculating average rushing and passing yards per game
offense_stats = df.groupby('gameId').agg(
    avg_rushing_yards=('rushingYards', 'mean'),
    avg_passing_yards=('passingYards', 'mean')
).reset_index()

# Visualizing rushing and passing effectiveness
plt.figure(figsize=(12, 6))
sns.histplot(offense_stats['avg_rushing_yards'], color='blue', label='Rushing Yards', kde=True)
sns.histplot(offense_stats['avg_passing_yards'], color='green', label='Passing Yards', kde=True)
plt.title("Distribution of Average Rushing and Passing Yards per Game")
plt.xlabel("Yards")
plt.legend()
plt.show()
```

### Insights from Rushing and Passing Yards Distribution

- **Passing Yards**: Passing plays show higher variability with a peak around 0.15-0.20 yards and potential for large gains up to 0.30 yards. This suggests a strategic focus on passing could maximize yardage, especially for quick field advancements.
- **Rushing Yards**: Rushing plays are more consistent with a peak at 0.05-0.10 yards, making them valuable for short, reliable gains and controlling game tempo.
- **Game Strategy**: A balanced approach—using rushing for consistent gains and passing for potential big plays—can adapt to various game situations, offering both control and explosive opportunities.

<h1 style="background-color: orange ; color: white; font-family: 'Verdana', sans-serif; text-align: center; padding: 15px; border-radius: 10px; font-weight: bold; letter-spacing: 2px; text-transform: uppercase;">
  1.4 Analyze Yrads after catch
</h1>

```python
# Calculating average yards after catch per game
offense_stats['avg_yards_after_catch'] = df.groupby('gameId')['yardageGainedAfterTheCatch'].mean().values

# Visualizing the distribution of yards after catch
plt.figure(figsize=(10, 6))
sns.histplot(offense_stats['avg_yards_after_catch'], color='purple', kde=True)
plt.title("Distribution of Average Yards After Catch per Game")
plt.xlabel("Yards After Catch")
plt.show()
```

### Insights on Average Yards After Catch per Game

- **Peak YAC**: Most values cluster around **0.08 yards** after catch, indicating that this is the average range players gain after receiving a pass.
- **Spread**: The distribution is fairly symmetrical, with yards after catch mostly ranging from **0.04 to 0.12 yards**.
- **Game Strategy**: Emphasizing players and plays that can consistently gain yards after the catch could enhance yardage on passing plays. Targeting players with high YAC potential might improve overall offensive performance in the NFL Big Bowl.

<h1 style="background-color: orange ; color: white; font-family: 'Verdana', sans-serif; text-align: center; padding: 15px; border-radius: 10px; font-weight: bold; letter-spacing: 2px; text-transform: uppercase;">
  1.5 Analyze Turnovers and Ball Security
</h1>

```python
# Calculating total fumbles and fumble losses per game
offense_stats['total_fumbles'] = df.groupby('gameId')['fumbles'].sum().values
offense_stats['fumbles_lost'] = df.groupby('gameId')['fumbleLost'].sum().values

# Visualizing fumble losses per game
plt.figure(figsize=(10, 6))
sns.histplot(offense_stats['fumbles_lost'], color='red', kde=True)
plt.title("Distribution of Fumbles Lost per Game")
plt.xlabel("Fumbles Lost")
plt.show()
```

### Insights on Fumbles Lost per Game

- **Majority at Low Levels**: The majority of games have **0 to 1 fumbles lost**, showing that teams generally maintain good control.
- **Distribution**: A rapid decline in frequency occurs after 1 fumble, with very few games seeing 2 or more fumbles lost.
- **Impact on Strategy**: Reducing even rare instances of fumbles could be crucial, as fumbles have a significant negative impact. A focus on drills or tactics to minimize fumbles could benefit teams in high-stakes games like the NFL Big Bowl.

<h1 style="background-color: orange ; color: white; font-family: 'Verdana', sans-serif; text-align: center; padding: 15px; border-radius: 10px; font-weight: bold; letter-spacing: 2px; text-transform: uppercase;">
  1.6 Analyze Quarterback Protection (Sack Yards)
</h1>

```python
# Calculating average sack yards lost per game
offense_stats['avg_sack_yards'] = df.groupby('gameId')['sackYardsAsOffense'].mean().values

# Visualizing the distribution of sack yards
plt.figure(figsize=(10, 6))
sns.histplot(offense_stats['avg_sack_yards'], color='orange', kde=True)
plt.title("Distribution of Average Sack Yards Lost per Game")
plt.xlabel("Sack Yards")
plt.show()
```

## 1. Distribution Characteristics
- **Central Tendency**: Average sack yards lost per game peaks around -0.015 yards.
- **Shape and Symmetry**: Roughly normal distribution, slightly right-skewed.
- **Range**: Sack yards lost range from -0.03 to 0, with most values near the mean.

## 2. Impact on Game Strategy
- **Offensive Strategy**: Low sack yards indicate strong protection or quick-release plays, helping maintain momentum.
- **Defensive Strategy**: High sack yard losses can be forced by aggressive pass-rush tactics.
- **Risk Management**: Teams may adjust play-calling to avoid drives stalling from sack losses.

## 3. Tactical Insights for Teams
- **Offensive Line & Protection**: Consistent losses suggest similar protection levels; improving this can help reduce sack yards.
- **Big Game Adaptation**: Minimizing sack yards is crucial for drive success in high-stakes games.
- **Opponent Analysis**: Knowing opponents’ sack trends helps tailor protection and quick-passing strategies.

<h1 style="background-color: orange ; color: white; font-family: 'Verdana', sans-serif; text-align: center; padding: 15px; border-radius: 10px; font-weight: bold; letter-spacing: 2px; text-transform: uppercase;">
  1.7 Analyze Penalties and Yardage Loss
</h1>

```python
# Calculating average penalty yards per game
offense_stats['avg_penalty_yards'] = df.groupby('gameId')['penaltyYards'].mean().values

# Visualizing penalty yards
plt.figure(figsize=(10, 6))
sns.histplot(offense_stats['avg_penalty_yards'], color='brown', kde=True)
plt.title("Distribution of Average Penalty Yards per Game")
plt.xlabel("Penalty Yards")
plt.show()
```

## 1. Distribution Characteristics
- **Central Tendency**: Most games average around 0 to 0.005 penalty yards, with a high frequency near zero.
- **Shape and Skew**: The distribution is right-skewed, indicating that low penalty yards per game are more common.
- **Range**: Penalty yards range from 0 to approximately 0.02, with fewer games at higher penalty yards.

## 2. Impact on Game Strategy
- **Discipline and Control**: Teams with lower penalty yards likely emphasize discipline, reducing negative impacts on field position.
- **Aggressive vs. Conservative Play**: Teams may balance between aggressive tactics (which risk penalties) and conservative play to minimize penalty yards.
- **Drive Continuity**: Fewer penalty yards help maintain drive momentum, reducing setbacks from costly penalties.

## 3. Tactical Insights for Teams
- **Focus on Clean Play**: Teams can benefit from strategies that minimize penalties, such as disciplined offensive and defensive play.
- **Big Game Preparation**: Reducing penalty yards is especially crucial in high-stakes games to avoid handing opponents advantages.
- **Opponent Analysis**: Studying an opponent’s penalty tendencies can help in game planning to exploit potential weaknesses.

<h1 style="background-color: orange ; color: white; font-family: 'Verdana', sans-serif; text-align: center; padding: 15px; border-radius: 10px; font-weight: bold; letter-spacing: 2px; text-transform: uppercase;">
  1.8 Analyze Turnovers Caused (Interceptions and Fumbles)
</h1>

```python
# Calculating average and total turnovers caused per game
defense_stats = df.groupby('gameId').agg(
    avg_forced_fumbles=('forcedFumbleAsDefense', 'mean'),
    total_forced_fumbles=('forcedFumbleAsDefense', 'sum'),
    avg_interceptions=('hadInterception', 'mean'),
    total_interceptions=('hadInterception', 'sum')
).reset_index()

# Visualizing forced fumbles and interceptions
plt.figure(figsize=(12, 6))
sns.histplot(defense_stats['total_forced_fumbles'], color='blue', kde=True, label='Forced Fumbles')
sns.histplot(defense_stats['total_interceptions'], color='green', kde=True, label='Interceptions')
plt.title("Distribution of Total Forced Fumbles and Interceptions per Game")
plt.xlabel("Turnovers Caused")
plt.legend()
plt.show()
```

## 1. Distribution Characteristics
- **Central Tendency**: Both forced fumbles and interceptions commonly range between 0 to 2 per game, with a peak around 1 turnover caused.
- **Shape and Skew**: Both distributions are right-skewed, indicating that higher numbers of turnovers per game are less common.
- **Range**: Forced fumbles and interceptions each span from 0 to around 6 per game, but occurrences above 3 are rare.

## 2. Impact on Game Strategy
- **Defensive Aggressiveness**: Higher forced fumbles suggest a more aggressive defense, which may focus on tackling techniques to strip the ball.
- **Ball-Hawking Secondary**: A high number of interceptions indicates strong pass coverage, with defenders adept at reading and intercepting passes.
- **Turnover Creation**: Generating turnovers gives teams more offensive opportunities, often leading to better field position and scoring chances.

## 3. Tactical Insights for Teams
- **Strengthen Ball Security**: Teams should emphasize ball security to counter opponents that excel in forcing fumbles and interceptions.
- **Game Planning**: Analyzing opponents' turnover tendencies allows teams to adapt, focusing on protecting the ball against strong defensive units.
- **Exploiting Weaknesses**: Offenses facing teams with low turnover creation may take more risks, while teams strong in turnovers can prepare to capitalize on opponent mistakes.

<h1 style="background-color: orange ; color: white; font-family: 'Verdana', sans-serif; text-align: center; padding: 15px; border-radius: 10px; font-weight: bold; letter-spacing: 2px; text-transform: uppercase;">
  1.9 Analyze Pressure on the Quarterback
</h1>

```python
# Calculating average quarterback hits and sack yards per game
defense_stats['avg_quarterback_hits'] = df.groupby('gameId')['quarterbackHit'].mean().values
defense_stats['avg_sack_yards_defense'] = df.groupby('gameId')['sackYardsAsDefense'].mean().values
defense_stats['avg_caused_pressure'] = df.groupby('gameId')['causedPressure'].mean().values

# Visualizing quarterback hits and sack yards
plt.figure(figsize=(12, 6))
sns.histplot(defense_stats['avg_quarterback_hits'], color='purple', kde=True, label='Quarterback Hits')
sns.histplot(defense_stats['avg_sack_yards_defense'], color='orange', kde=True, label='Sack Yards')
plt.title("Distribution of Quarterback Hits and Sack Yards (Defense) per Game")
plt.xlabel("Defensive Pressure")
plt.legend()
plt.show()
```

### Insights from the Graph

1. **Comparison of Defensive Metrics**:
   - The histogram represents the distribution of two defensive metrics: *Quarterback Hits* (in purple) and *Sack Yards* (in yellow).
   - Both distributions are presented against the variable *Defensive Pressure* per game.

2. **Distribution Shape**:
   - *Sack Yards* distribution (yellow) appears more symmetric, with a peak at a slightly negative defensive pressure range.
   - *Quarterback Hits* distribution (purple) is more right-skewed, with a peak around a slightly positive defensive pressure value.

3. **Defensive Pressure Impact**:
   - *Sack Yards* are more evenly spread out and cover a broader range of negative defensive pressure values.
   - *Quarterback Hits* are clustered at higher defensive pressure, indicating that hits may be more likely under higher pressure conditions.

4. **Frequency and Count Observations**:
   - The *Sack Yards* distribution has a higher count in the middle ranges compared to the *Quarterback Hits*.
   - The peak for *Quarterback Hits* suggests that they are relatively less frequent compared to sack yards, but when they occur, they concentrate around positive pure in NFL games.

<h1 style="background-color: orange ; color: white; font-family: 'Verdana', sans-serif; text-align: center; padding: 15px; border-radius: 10px; font-weight: bold; letter-spacing: 2px; text-transform: uppercase;">
  1.10 Analyze Tackles for Loss and Solo Tackles
</h1>

```python
# Calculating average and total tackles for loss and solo tackles per game
defense_stats['avg_tackles_for_loss'] = df.groupby('gameId')['tackleForALoss'].mean().values
defense_stats['total_tackles_for_loss'] = df.groupby('gameId')['tackleForALoss'].sum().values
defense_stats['avg_solo_tackles'] = df.groupby('gameId')['soloTackle'].mean().values

# Visualizing tackles for loss and solo tackles
plt.figure(figsize=(12, 6))
sns.histplot(defense_stats['total_tackles_for_loss'], color='red', kde=True, label='Tackles for Loss')
sns.histplot(defense_stats['avg_solo_tackles'], color='green', kde=True, label='Solo Tackles')
plt.title("Distribution of Tackles for Loss and Solo Tackles per Game")
plt.xlabel("Tackles")
plt.legend()
plt.show()
```

### Insights from the Graph

1. **Comparison of Tackling Metrics**:
   - The histogram displays the distribution of *Tackles for Loss* (in red) and *Solo Tackles* (in green) per game.
   - The graph provides a clear distinction between these two types of tackles, each with its own distribution pattern.

2. **Distribution Shape**:
   - *Tackles for Loss* (red) shows a roughly normal distribution, with a peak between 7.5 and 10 tackles per game.
   - The distribution is fairly symmetric but with a slight right skew, indicating a small number of instances with higher tackle counts.
   - *Solo Tackles* (green) are represented by a single narrow spike around zero, suggesting that the data for solo tackles might be heavily concentrated at one value or that there could be data sparsity in this measurement.

3. **Frequency and Count Observations**:
   - *Tackles for Loss* occur with a wider spread across games, with most values falling between 5 and 12 tackles per game.
   - The frequency of *Solo Tackles* appears minimal and sharply focused, hinting at a need for more data points or a unique distribution characteristic.

4. **Implications for Defensive Analysis**:
   - The data suggests that *Tackles for Loss* are more commonly distributed over a range, indicating variation in defensive performance.
   - The singular spike for *Solo Tackles* may require further investigation to understand why it is srther exploration.

<h1 style="background-color: orange ; color: white; font-family: 'Verdana', sans-serif; text-align: center; padding: 15px; border-radius: 10px; font-weight: bold; letter-spacing: 2px; text-transform: uppercase;">
  1.11 Analyze the Impact of Penalties on Plays
</h1>

```python
# Calculating average and total penalty yards per game
situational_stats = df.groupby('gameId').agg(
    avg_penalty_yards=('penaltyYards', 'mean'),
    total_penalty_yards=('penaltyYards', 'sum')
).reset_index()

# Visualizing the impact of penalties per game
plt.figure(figsize=(12, 6))
sns.histplot(situational_stats['total_penalty_yards'], color='blue', kde=True)
plt.title("Distribution of Total Penalty Yards per Game")
plt.xlabel("Total Penalty Yards")
plt.show()

# Average penalty yards per play
df['penalty_per_play'] = df.groupby('playId')['penaltyYards'].transform('mean')
sns.histplot(df['penalty_per_play'].dropna(), color='orange', kde=True)
plt.title("Average Penalty Yards per Play")
plt.xlabel("Penalty Yards")
plt.show()
```

### Insights from the Graphs

1. **Distribution of Total Penalty Yards per Game** (Top Graph):
   - The histogram displays the distribution of *Total Penalty Yards* per game.
   - The data is right-skewed, with a higher count of games having fewer total penalty yards (around 0-10 yards).
   - As the total penalty yards increase, the count of games decreases, with very few games having over 40 penalty yards.
   - The smooth line overlay indicates a gradual decline, emphasizing the rarity of higher penalty yardage games.

2. **Average Penalty Yards per Play** (Bottom Graph):
   - This histogram represents *Average Penalty Yards* per play.
   - The distribution is heavily skewed to the right, with a significant spike at very low average penalty yard values (close to 0).
   - There are almost no instances of average penalty yards exceeding 0.1, indicating that penalties are generally infrequent and minor in yardage impact per play.

3. **Key Observations**:
   - Penalty yards tend to accumulate in a limited range, and most games experience relatively few penalty yards.
   - On a per-play basis, penalties are extremely rare and minimal, contributing little to the overall play statistics.
   - The data suggests that while penalties occur, they typically have a minimal impact on a per-play basis but can still add up across a game in losses in games.

<h1 style="background-color: orange ; color: white; font-family: 'Verdana', sans-serif; text-align: center; padding: 15px; border-radius: 10px; font-weight: bold; letter-spacing: 2px; text-transform: uppercase;">
  1.12 Yardage Gained After the Catch
</h1>

```python
# Calculating average and total yardage gained after the catch per game
situational_stats['avg_yards_after_catch'] = df.groupby('gameId')['yardageGainedAfterTheCatch'].mean().values
situational_stats['total_yards_after_catch'] = df.groupby('gameId')['yardageGainedAfterTheCatch'].sum().values

# Visualizing yards gained after catch per game
plt.figure(figsize=(12, 6))
sns.histplot(situational_stats['total_yards_after_catch'], color='green', kde=True)
plt.title("Total Yards Gained After the Catch per Game")
plt.xlabel("Yards Gained After Catch")
plt.show()

# Average yards gained after catch per play
df['yards_after_catch_per_play'] = df.groupby('playId')['yardageGainedAfterTheCatch'].transform('mean')
sns.histplot(df['yards_after_catch_per_play'].dropna(), color='purple', kde=True)
plt.title("Average Yards Gained After Catch per Play")
plt.xlabel("Yards Gained After Catch")
plt.show()
```

### NFL Big Bowl Data Insights and Strategy Implications

1. **Total Yards Gained After Catch per Game**:
   - Most games show 200–250 yards gained after the catch, with fewer games above 300 yards.
   - **Strategy**: Focus on consistency in the 200–250 range through better blocking and optimized routes to maximize yardage.

2. **Average Yards Gained After Catch per Play**:
   - Heavily right-skewed, with most plays yielding minimal yards after the catch.
   - **Strategy**: Design quick, safe routes to maximize short gains and occasionally exploit gaps for longer plays.

### Strategic Applications
- **Receiver Training**: Emphasize yards after contact for mid-range gains.
- **Defensive Adjustments**: Analyze opponent trends and aim to minimize high yardage plays.
- **Play Consistency**: Convert short plays to consistent medium gains through efficient route timing and positioning.

<h1 style="background-color: orange ; color: white; font-family: 'Verdana', sans-serif; text-align: center; padding: 15px; border-radius: 10px; font-weight: bold; letter-spacing: 2px; text-transform: uppercase;">
  1.13 Investigate Motion and Shifts¶
</h1>

```python
# Creating a 3D scatter plot
fig = px.scatter_3d(df, 
                    x='rushingYards', 
                    y='passingYards', 
                    z='receivingYards', 
                    color='teamAbbr',  # Color by team abbreviation
                    size='fumbles',    # Size of markers by number of fumbles
                    hover_name='nflId',  # Showing player ID on hover
                    title='3D Scatter Plot of NFL Player Performance',
                    labels={'rushingYards': 'Rushing Yards',
                            'passingYards': 'Passing Yards',
                            'receivingYards': 'Receiving Yards'},
                    template='plotly_dark')  # Optional: dark template for aesthetics

# Showing the plot
fig.show()
```

<h1 style="background-color: orange ; color: white; font-family: 'Verdana', sans-serif; text-align: center; padding: 15px; border-radius: 10px; font-weight: bold; letter-spacing: 2px; text-transform: uppercase;">
  Found it useful? High-five the notebook! ✋📊
</h1>