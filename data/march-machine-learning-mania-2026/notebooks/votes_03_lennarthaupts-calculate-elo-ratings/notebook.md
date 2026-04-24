# Calculate ELO-Ratings🏀

- **Author:** Lennart Haupts
- **Votes:** 309
- **Ref:** lennarthaupts/calculate-elo-ratings
- **URL:** https://www.kaggle.com/code/lennarthaupts/calculate-elo-ratings
- **Last run:** 2026-04-22 15:52:16.017000

---

```python
# Imports
import numpy as np
import pandas as pd 
from scipy.stats import linregress
from tqdm import tqdm
import matplotlib.pyplot as plt
```

**Note: While this notebook is designed as a foundational resource for building your own models, it does produce a ready-to-submit CSV. While the Elo ratings alone are unlikely to reach the medal zone, they serve as powerful features; in the 2026 competition, this approach was a key component in two prize-winning entries: [alphawave's 5th place solution](https://www.kaggle.com/competitions/march-machine-learning-mania-2026/writeups/march-machine-learning-mania-2026-5nd-place-solut) and [seddik's 6th place solution](https://www.kaggle.com/competitions/march-machine-learning-mania-2026/writeups/6th-place-solution-march-machine-learning-mania-20).**

# Summary

This notebook contains functions for computing Elo ratings during regular seasons and tournaments, summarizing team performance in both men's and women's events. It produces two ready to use CSV files containing the following key team information:

- **Rating_Mean:** The average team rating in a regular season.
- **Rating_Median:** The median team rating in a regular season.
- **Rating_Last:** The last rating in a regular season/ the final rating before the tournament.
- **Rating_Std:** Standard Deviation of team ratings.
- **Rating_Max:** The maximum team rating in a season.
- **Rating_Min:** The minimum team rating in a season.
- **Rating_Trend:** The slope of the team's rating over the season.


This version allows for a multiplier based on the margin of victory. $\alpha$ in the following function is a tunable parameter scaling the multiplier. No multiplier is applied if alpha is set to `None`.  
$$\frac{Score_w - Score_l}{\alpha}$$

You can also assign custom weights such as different weights for regular season matches and tournament matches. Another alternative would be to assign the weights based on `DayNum`. You may pass them with the argument `weights`.

# Functions

```python
def calculate_elo(
    teams, data, initial_rating=2000, k=140, width=None, alpha=None, weights=False, lowerlim=float("-inf")
    ):
    '''
    Calculate Elo ratings for each team based on match data.

    Parameters:
    - teams (array-like): Containing Team-IDs.
    - data (pd.DataFrame): DataFrame with all matches in chronological order.
    - initial_rating (float): Initial rating of an unranked team (default: 2000). 
    - k (float): K-factor, determining the impact of each match on team ratings (default: 140).
    - alpha (float or None): Tuning parameter for the multiplier for the margin of victory. No multiplier if None.

    Returns: 
    - list: Historical ratings of the winning team (WTeam).
    - list: Historical ratings of the losing team (LTeam).
    - list: Brier score for each match (due to symmetry only for 1 team)
    '''
    
    # Dictionary to keep track of current ratings for each team
    team_dict = {}
    for team in teams:
        team_dict[team] = initial_rating
    if not width:
        width = initial_rating
    
    # Lists to store ratings for each team in each game
    r1, r2 = [], []
    loss = []
    margin_of_victory = 1

    # Iterate through the game data
    for wteam, lteam, ws, ls, w  in tqdm(
        zip(data.WTeamID, data.LTeamID, data.WScore, data.LScore, data.weight), total=len(data)
    ):

        # Calculate expected outcomes based on Elo ratings
        rateW = 1 / (1 + 10 ** ((team_dict[lteam] - team_dict[wteam]) / width))
        rateL = 1 / (1 + 10 ** ((team_dict[wteam] - team_dict[lteam]) / width))
        
        if alpha:
            margin_of_victory = (ws - ls)/alpha

        # Update ratings for winning and losing teams
        team_dict[wteam] += w * k * margin_of_victory * (1 - rateW)
        team_dict[lteam] += w * k * margin_of_victory * (0 - rateL)

        # Ensure that ratings do not go below lower limit
        if team_dict[lteam] < lowerlim:
            team_dict[lteam] = lowerlim
            
        # Append current ratings for teams to lists
        r1.append(team_dict[wteam])
        r2.append(team_dict[lteam])
        loss.append((1-rateW)**2)
        
    return r1, r2, loss

def create_elo_data(
    teams, data, initial_rating=2000, k=140, width=None, alpha=None, weights=None, lowerlim=float("-inf")
    ):
    '''
    Create a DataFrame with summary statistics of Elo ratings for teams based on historical match data.

    Parameters:
    - teams (array-like): Containing Team-IDs.
    - data (pd.DataFrame): DataFrame with all matches in chronological order.
    - initial_rating (float): Initial rating of an unranked team (default: 2000).
    - k (float): K-factor, determining the impact of each match on team ratings (default: 140).
    - weights (array-like): Containing weights for each match.

    Returns: 
    - DataFrame: Summary statistics of Elo ratings for teams throughout a season.
    '''
    
    if isinstance(weights, (list, np.ndarray, pd.Series)):
        data['weight'] = weights
    else:
        data['weight'] = 1
    
    r1, r2, loss = calculate_elo(
        teams, data, initial_rating, k, width, alpha, weights, lowerlim
    )
    # Calculate loss only on tourney results
    loss = np.mean(np.array(loss)[data.tourney == 1])
    print(f"=== Brier Score: {loss:.5f} (Only  Tournaments) ===")
    
    # Concatenate arrays vertically
    seasons = np.concatenate([data.Season, data.Season])
    days = np.concatenate([data.DayNum, data.DayNum])
    teams = np.concatenate([data.WTeamID, data.LTeamID])
    tourney = np.concatenate([data.tourney, data.tourney])
    ratings = np.concatenate([r1, r2])
    # Create a DataFrame
    rating_df = pd.DataFrame({
        'Season': seasons,
        'DayNum': days,
        'TeamID': teams,
        'Rating': ratings,
        'Tourney': tourney
    })

    # Sort DataFrame and remove tournament data
    rating_df.sort_values(['TeamID', 'Season', 'DayNum'], inplace=True)
    rating_df = rating_df[rating_df['Tourney'] == 0]
    grouped = rating_df.groupby(['TeamID', 'Season'])
    results = grouped['Rating'].agg(['mean', 'median', 'std', 'min', 'max', 'last'])
    results.columns = ['Rating_Mean', 'Rating_Median', 'Rating_Std', 'Rating_Min', 'Rating_Max', 'Rating_Last']
    results['Rating_Trend'] = grouped.apply(lambda x: linregress(range(len(x)), x['Rating']).slope, include_groups=False)
    results.reset_index(inplace=True)
    
    return results, loss
```

# Apply Functions and Save Results to CSV

```python
# Load and Process Data Men's Tourney
regular_m = pd.read_csv('/kaggle/input/competitions/march-machine-learning-mania-2026/MRegularSeasonCompactResults.csv')
tourney_m = pd.read_csv('/kaggle/input/competitions/march-machine-learning-mania-2026/MNCAATourneyCompactResults.csv')
teams_m = pd.read_csv('/kaggle/input/competitions/march-machine-learning-mania-2026/MTeams.csv')

regular_m['tourney'] = 0
tourney_m['tourney'] = 1
regular_m['weight'] = 1
tourney_m['weight'] = 0.75

data_m = pd.concat([regular_m, tourney_m])
data_m.sort_values(['Season', 'DayNum'], inplace=True)
data_m.reset_index(inplace=True, drop=True)

initial_rating = width = 1200 

print("Men's Ratings")

elo_df_men, _ = create_elo_data(
    teams_m.TeamID, data_m, initial_rating=initial_rating, k=125, width=width, alpha=None, weights=data_m['weight']
)
elo_df_men.tail(10)
```

```python
# Load and Process Data Women's Tourney
regular_w = pd.read_csv('/kaggle/input/competitions/march-machine-learning-mania-2026/WRegularSeasonCompactResults.csv')
tourney_w = pd.read_csv('/kaggle/input/competitions/march-machine-learning-mania-2026/WNCAATourneyCompactResults.csv')
teams_w = pd.read_csv('/kaggle/input/competitions/march-machine-learning-mania-2026/WTeams.csv')

regular_w['tourney'] = 0
tourney_w['tourney'] = 1
regular_w['weight'] = 0.95
tourney_w['weight'] = 1

data_w = pd.concat([regular_w, tourney_w])
data_w.sort_values(['Season', 'DayNum'], inplace=True)
data_w.reset_index(inplace=True, drop=True)

print("Women's Ratings")

elo_df_women, _ = create_elo_data(
    teams_w.TeamID, data_w, initial_rating=initial_rating, k=190, width=width, alpha=None, weights=data_w['weight']
)
elo_df_women.tail(10)
```

```python
# Save to csv
elo_df_men.to_csv('mens_elo_rating.csv')
elo_df_women.to_csv('womens_elo_rating.csv')
```

# Top 20 Teams Based on Latest Data-Update

```python
# Men's Teams
tmp_df_men = pd.merge(elo_df_men, teams_m, on='TeamID', how='left')
tmp_df_men = tmp_df_men[tmp_df_men['Season'] == 2026]
top_men_teams = tmp_df_men.sort_values('Rating_Last', ascending=False)[:20][['TeamName', 'Rating_Last', 'Rating_Trend']]
top_men_teams = top_men_teams.reindex(index=top_men_teams.index[::-1])

# Women's Teams
tmp_df_women = pd.merge(elo_df_women, teams_w, on='TeamID', how='left')
tmp_df_women = tmp_df_women[tmp_df_women['Season'] == 2026]
top_women_teams = tmp_df_women.sort_values('Rating_Last', ascending=False)[:20][['TeamName', 'Rating_Last', 'Rating_Trend']]
top_women_teams = top_women_teams.reindex(index=top_women_teams.index[::-1])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 8))

# Men's Teams
ax1.barh(top_men_teams['TeamName'], top_men_teams['Rating_Last'], color='skyblue', label='Rating_Last')
ax1.set_title("Top Men's Teams - 2025")
ax1.set_xlabel('Last Rating')
ax1.set_ylabel('TeamName')
ax1.legend()

# Women's Teams
ax2.barh(top_women_teams['TeamName'], top_women_teams['Rating_Last'], color='#3F51B5', label='Rating_Last')
ax2.set_title("Top Women's Teams - 2025")
ax2.set_xlabel('Last Rating')
ax2.set_ylabel('TeamName')
ax2.legend()

plt.tight_layout()
plt.show()
```

```python
# Prepare direct submission
submission = pd.read_csv("/kaggle/input/competitions/march-machine-learning-mania-2026/SampleSubmissionStage2.csv")

# Split the ID into Season, T1_TeamID, and T2_TeamID
sub = submission.ID.str.split('_', expand=True).astype(int)
sub.columns = ["Season", "T1_TeamID", "T2_TeamID"]

# Turn elo dfs into dict
elo_dict = pd.concat([
    elo_df_women[elo_df_women.Season == 2026],
    elo_df_men[elo_df_men.Season == 2026]
]).set_index("TeamID")["Rating_Last"].to_dict()

# Calculate probabilities
submission.Pred = 1 / (1 + 10**((sub.T2_TeamID.map(elo_dict) - sub.T1_TeamID.map(elo_dict))/width))
submission.to_csv("submission.csv", index=False)
submission.head()
```