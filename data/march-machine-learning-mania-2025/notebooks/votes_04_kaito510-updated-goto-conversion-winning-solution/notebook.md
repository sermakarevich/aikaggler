# UPDATED goto_conversion 🥇🥈🥈🥉 winning solution

- **Author:** goto_conversion
- **Votes:** 288
- **Ref:** kaito510/updated-goto-conversion-winning-solution
- **URL:** https://www.kaggle.com/code/kaito510/updated-goto-conversion-winning-solution
- **Last run:** 2025-03-18 14:15:24.357000

---

# This is an updated version of the code I uploaded [here](https://www.kaggle.com/code/kaito510/goto-conversion-winning-solution). 
# In other words, this is the 2025 version of the 2024 version I uploaded [here](https://www.kaggle.com/code/kaito510/updated-1xgold-2xsilvers-key-ingredient) to fit the contest design for this year.

The probability matrices were computed by converting betting odds to outcome probabilities using **[goto_conversion](https://github.com/gotoConversion/goto_conversion)**, which are displayed interactively under the first code chunk and can be found [here](https://github.com/gotoConversion/goto_conversion/tree/main/probabilityMatrices) as csv files. I only updated the essential components of the code I uploaded [here](https://www.kaggle.com/code/kaito510/goto-conversion-winning-solution) to ensure I meet the tight deadline.

**In 2024, this solution alone was sufficient for a medal**. This can be verified by noticing [2024's leaderboard scores from 86th to 100th](https://www.kaggle.com/competitions/march-machine-learning-mania-2024/leaderboard) and the leaderboard score of [2024's version of this solution](https://www.kaggle.com/code/kaito510/updated-1xgold-2xsilvers-key-ingredient) are both 0.06035.

For even better performance, this solution should be used as an ingredient for your solution instead of as your entire solution. In 2024, at least **two gold ([3rd](https://www.kaggle.com/competitions/march-machine-learning-mania-2024/discussion/495101) and [4th](https://www.kaggle.com/competitions/march-machine-learning-mania-2024/discussion/494407) place) and one silver ([38th](https://www.kaggle.com/competitions/march-machine-learning-mania-2024/discussion/485888#2740879) place)** medal winners publicly stated that they used this solution as an ingredient for their success; listed [here](https://github.com/gotoConversion/goto_conversion?tab=readme-ov-file#goto_conversion---used-by-4-gold-medal-solutions-on-kaggle).

```python
#Setup

import pandas as pd
year = 2025
kaggleFolderPath = '/kaggle/input/march-machine-learning-mania-' + str(year)
fivethirtyeightFolderPath = '/kaggle/input/538data'
```

```python
#Mens Probability Matrix
#source: https://github.com/gotoConversion/goto_conversion
#Matrices were computed by converting betting odds to probabilities using goto_conversion

mensProbabilities_df = pd.read_csv(fivethirtyeightFolderPath + '/mensProbabilitiesTable.csv', index_col = 'player') #source: https://github.com/gotoConversion/goto_conversion
mensProbabilities_df = mensProbabilities_df.drop('Elo_Rating', axis=1)
```

```python
#Womens Probability Matrix
#source: https://github.com/gotoConversion/goto_conversion
#Matrices were computed by converting betting odds to probabilities using goto_conversion

womensProbabilities_df = pd.read_csv(fivethirtyeightFolderPath + '/womensProbabilitiesTable.csv', index_col = 'player') #source: https://github.com/gotoConversion/goto_conversion
womensProbabilities_df = womensProbabilities_df.drop('Elo_Rating', axis=1)
```

# Submission with Optimal Strategy

**Below is a mathematical proof that the optimal strategy to win a medal under Brier Score is when we assume a team with 33.3% chance of winning a match to win that match.**

The expected return when we risk on a given game can be expressed as:

f(p) = p(1 - p)^2 where p is the probability of success and (1-p)^2 is essentially the reward for the risk taken if the risk succeeds

This implies f'(p) and f''(p) can be expressed as:

f'(p) = -2p + 2p^2 + (1-p)^2

f''(p) = -4 + 6p

argmax_p f(p) = 1/3 with tedious mathematical working omitted.

Thus, expected reward is maximised when we assume a team with 1/3 chance of winning a match to win that match.

```python
#Import team seeds
mensTeamSeeds_df = pd.read_csv(kaggleFolderPath + '/MNCAATourneySeeds.csv')
mensTeamSeeds2025_df = mensTeamSeeds_df.iloc[[x == year for x in mensTeamSeeds_df['Season']]]
womensTeamSeeds_df = pd.read_csv(kaggleFolderPath + '/WNCAATourneySeeds.csv')
womensTeamSeeds2025_df = womensTeamSeeds_df.iloc[[x == year for x in womensTeamSeeds_df['Season']]]
```

```python
#Implement Optimal Strategy (if you agree)
def get_roundOfMatch(team1, team2, seeds_df):

    slotMap = [1, 16, 8, 9, 5, 12, 4, 13, 6, 11, 3, 14, 7, 10, 2, 15]

    team1_seed = seeds_df.loc[[x == team1 for x in seeds_df['TeamID']],'Seed'].values[0]
    team2_seed = seeds_df.loc[[x == team2 for x in seeds_df['TeamID']],'Seed'].values[0]

    isFirstFourMatch = team1_seed[:3] == team2_seed[:3]
    if isFirstFourMatch:
        return 1

    team1_region = str(team1_seed[:1])
    team2_region = str(team2_seed[:1])

    team1_seedNumber = int(team1_seed[1:3]) #careful with first four teams
    team2_seedNumber = int(team2_seed[1:3]) #careful with first four teams

    isRegionSame = team1_region == team2_region
    if not isRegionSame:

        isTeam1_regionWX = team1_region in ['W','X']
        isTeam2_regionWX = team2_region in ['W','X']

        if isTeam1_regionWX and isTeam2_regionWX: #both W or X region
            return 6

        elif (not isTeam1_regionWX) and (not isTeam2_regionWX): #both not W or X region
            return 6

        else:
            return 7

    else: #same region

        team1_slot = slotMap.index(team1_seedNumber)
        team2_slot = slotMap.index(team2_seedNumber)

        isRound2 = (team1_slot // 2) == (team2_slot // 2)  #round of 64 or first four (not counted anyway)
        if isRound2:
            return 2

        isRound3 = (team1_slot // 4) == (team2_slot // 4)
        if isRound3: #yet to find why but "elif" throws error
            return 3

        isRound4 = (team1_slot // 8) == (team2_slot // 8)
        if isRound4: #yet to find why but "elif" throws error
            return 4

        else:
            return 5

def get_tourneyFlag(team1, team2, seeds_df):

    tourneyTeams = seeds_df['TeamID'].tolist()

    isTeam1InTourney = team1 in tourneyTeams
    isTeam2InTourney = team2 in tourneyTeams

    if isTeam1InTourney and isTeam2InTourney:
        return get_roundOfMatch(team1, team2, seeds_df)

    else:
        return 0

def get_flag_list(submission_df, mensTeamSeeds2025_df, womensTeamSeeds2025_df):
    flag_list = []
    for i in range(submission_df.shape[0]):

        currRow = submission_df.iloc[i,0].split('_')
        team1 = int(currRow[1])
        team2 = int(currRow[2])

        isWomensMatch = team1 + team2 > 6000
        if isWomensMatch:
            flag = get_tourneyFlag(team1, team2, womensTeamSeeds2025_df)
        else:
            flag = get_tourneyFlag(team1, team2, mensTeamSeeds2025_df)

        flag_list.append(flag)
    return flag_list

def set_optimalStrategy(submission_df, mensTeamSeeds2025_df, womensTeamSeeds2025_df, riskTeam, riskTeamToWinRound):

    flag_list = get_flag_list(submission_df, mensTeamSeeds2025_df, womensTeamSeeds2025_df)

    for i in range(submission_df.shape[0]):
        submission_row = submission_df.iloc[i,0].split('_')
        submission_round = flag_list[i]

        team1 = int(submission_row[1])
        team2 = int(submission_row[2])

        isTeam1Win = (team1 == riskTeam) and (0 < submission_round) and (submission_round <= riskTeamToWinRound)
        isTeam2Win = (team2 == riskTeam) and (0 < submission_round) and (submission_round <= riskTeamToWinRound)
        if isTeam1Win:
            submission_df.at[i, 'Pred'] = 1.0
            print(submission_df.iloc[i])
        elif isTeam2Win:
            submission_df.at[i, 'Pred'] = 0.0
            print(submission_df.iloc[i])
    
    return submission_df

submission_df = pd.read_csv(fivethirtyeightFolderPath + '/submission.csv')
riskTeam = 1179 #Drake
riskTeamToWinRound = 2 #Near Optimal Probability for Strategy
submission_df = set_optimalStrategy(submission_df, mensTeamSeeds2025_df, womensTeamSeeds2025_df, riskTeam, riskTeamToWinRound)
submission_df.to_csv('submission.csv', index=False)
```