# Simple starter notebook for March Mania 2025

- **Author:** Paul Mooney
- **Votes:** 639
- **Ref:** paultimothymooney/simple-starter-notebook-for-march-mania-2025
- **URL:** https://www.kaggle.com/code/paultimothymooney/simple-starter-notebook-for-march-mania-2025
- **Last run:** 2025-03-18 14:18:55.363000

---

# March Mania 2025 - Starter Notebook

## Goal of the competition

The goal of this competition is to predict that probability that the smaller ``TeamID`` will win a given matchup. You will predict the probability for every possible matchup between every possible team over the past 4 years. You'll be given a sample submission file where the ```ID``` value indicates the year of the matchup as well as the identities of both teams within the matchup. For example, for an ```ID``` of ```2025_1101_1104``` you would need to predict the outcome of the matchup between ```TeamID 1101``` vs ```TeamID 1104``` during the ```2025``` tournament. Submitting a ```PRED``` of ```0.75``` indicates that you think that the probability of ```TeamID 1101``` winning that particular matchup is equal to ```0.75```.

## Overview of our submission strategy 
For this starter notebook, we will make a simple submission.

We can predict the winner of a match by considering the respective rankings of the opposing teams, only. Since the largest possible difference is 15 (which is #16 minus #1), we use a rudimentary formula that's 0.5 plus 0.03 times the difference in seeds, leading to a range of predictions spanning from 5% up to 95%. The stronger-seeded team (with a lower seed number from 1 to 16) will be the favorite and will have a prediction above 50%.

# Starter Code

## Step 1: Import Python packages

```python
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, mean_squared_error
```

## Step 2: Explore the data

```python
w_seed = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/WNCAATourneySeeds.csv')
m_seed = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/MNCAATourneySeeds.csv')
seed_df = pd.concat([m_seed, w_seed], axis=0).fillna(0.05)
submission_df = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/SampleSubmissionStage2.csv')
```

Team rankings are present in the files WNCAATourneySeeds.csv and MNCAATourneySeeds.csv. 
- The "Season" column indicates the year
- The "Seed" column indicates the ranking for a given conference (W01 = ranking 1 in conference W)
- The "TeamID" column contains a unique identifier for every team

```python
seed_df.head()
```

The sample_submission.csv file contains an "ID" column with the format year_teamID1_teamID2.

```python
submission_df.head()
```

## Step 3: Extract game info and team rankings

```python
def extract_game_info(id_str):
    # Extract year and team_ids
    parts = id_str.split('_')
    year = int(parts[0])
    teamID1 = int(parts[1])
    teamID2 = int(parts[2])
    return year, teamID1, teamID2

def extract_seed_value(seed_str):
    # Extract seed value
    try:
        return int(seed_str[1:])
    # Set seed to 16 for unselected teams and errors
    except ValueError:
        return 16

# Reformat the data
submission_df[['Season', 'TeamID1', 'TeamID2']] = submission_df['ID'].apply(extract_game_info).tolist()
seed_df['SeedValue'] = seed_df['Seed'].apply(extract_seed_value)

# Merge seed information for TeamID1
submission_df = pd.merge(submission_df, seed_df[['Season', 'TeamID', 'SeedValue']],
                         left_on=['Season', 'TeamID1'], right_on=['Season', 'TeamID'],
                         how='left')
submission_df = submission_df.rename(columns={'SeedValue': 'SeedValue1'}).drop(columns=['TeamID'])

# Merge seed information for TeamID2
submission_df = pd.merge(submission_df, seed_df[['Season', 'TeamID', 'SeedValue']],
                         left_on=['Season', 'TeamID2'], right_on=['Season', 'TeamID'],
                         how='left')
submission_df = submission_df.rename(columns={'SeedValue': 'SeedValue2'}).drop(columns=['TeamID'])
```

## Step 4: Make your predictions

```python
# Calculate seed difference
submission_df['SeedDiff'] = submission_df['SeedValue1'] - submission_df['SeedValue2']

# Update 'Pred' column
submission_df['Pred'] = 0.5 + (0.03 * submission_df['SeedDiff'])

# Drop unnecessary columns
submission_df = submission_df[['ID', 'Pred']].fillna(0.5)

# Preview your submission
submission_df.head()
```

```python
stats = submission_df.iloc[:, 1].describe()
print(stats)
```

## Step 5: Understand the metric

We don't know the outcomes of the games, so instead let's assume that the team that was listed first won every single matchup. This is what we'll call our "true value". Next, we'll calculate the average squared difference between the probabilities in our submission and that ground truth value. We'll call this the "Brier score". https://en.wikipedia.org/wiki/Brier_score

```python
# Create a dataframe of ground truth values
solution_df = submission_df.copy()
solution_df['Pred'] = 1

# Now calculate the Brier score
y_true = solution_df['Pred']
y_pred = submission_df['Pred']
brier_score = brier_score_loss(y_true, y_pred)
print(f"Brier Score: {brier_score}")
```

## Step 6: Make your submission

```python
submission_df.to_csv('/kaggle/working/submission.csv', index=False)
```