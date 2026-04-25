# 🏈 Games Insightful Analysis: Players in Action 📊

- **Author:** Arshman Khalid
- **Votes:** 81
- **Ref:** arshmankhalid/games-insightful-analysis-players-in-action
- **URL:** https://www.kaggle.com/code/arshmankhalid/games-insightful-analysis-players-in-action
- **Last run:** 2025-04-05 12:45:59.997000

---

<p style="background-image: url(https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSlzxVr7U6HD1hdR7LS89YvKB2gvUUU9CgluQ&usqp=CAU);font-family:Pacifico ,cursive;font-size:250%; color:white; text-align:center; border-radius: 0; padding:20px; font-weight: normal; border: 3px dashed #14adc6; box-shadow: 0px 5px 10px  rgba(0, 0, 0, 0.2);text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.4); background-size: cover; background-repeat: no-repeat; background-position: center;">Complete EDA and Detailed Insights from games.csv File 📊🏈</p>

![_bf2fa120-3651-460a-aa2a-5c457dcdfe01.jpeg](attachment:6f0c98fe-622a-4871-bb1e-a158012a0def.jpeg)

<p style="background-image: url(https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSlzxVr7U6HD1hdR7LS89YvKB2gvUUU9CgluQ&usqp=CAU);font-family:Pacifico ,cursive;font-size:250%; color:white; text-align:center; border-radius: 0; padding:20px; font-weight: normal; border: 3px dashed #14adc6; box-shadow: 0px 5px 10px  rgba(0, 0, 0, 0.2);text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.4); background-size: cover; background-repeat: no-repeat; background-position: center;">About The Auther📊🏈</p>

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

<p style="background-image: url(https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSlzxVr7U6HD1hdR7LS89YvKB2gvUUU9CgluQ&usqp=CAU);font-family:Pacifico ,cursive;font-size:250%; color:white; text-align:center; border-radius: 0; padding:20px; font-weight: normal; border: 3px dashed #14adc6; box-shadow: 0px 5px 10px  rgba(0, 0, 0, 0.2);text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.4); background-size: cover; background-repeat: no-repeat; background-position: center;">Lets Start📊🏈</p>

# Mian Objectives 
- Distribution of the Dataset
- Insights from Descriptive Statistics: A Coach’s Perspective📊🏈
- Analyzing Game Timing and Performance
- Analysis of Home vs. Visitor Scores
- Team Performance Analysis: Average Scores and Trends 
- Insights from Team Performance Across Weeks (Home Games)
- Point Differences and Team Performance

## About The Dataset 


| **Column**           | **Description**                                                                       |
|----------------------|---------------------------------------------------------------------------------------|
| `gameId`             | Game identifier, unique (numeric)                                                     |
| `season`             | Season of game                                                                        |
| `week`               | Week of game                                                                          |
| `gameDate`           | Game Date (time, mm/dd/yyyy)                                                          |
| `gameTimeEastern`    | Start time of game (time, HH:MM:SS, EST)                                               |
| `homeTeamAbbr`       | Home team three-letter code (text)                                                    |
| `visitorTeamAbbr`    | Visiting team three-letter code (text)                                                |
| `homeFinalScore`     | The total amount of points scored by the home team in the game (numeric)              |
| `visitorFinalScore`  | The total amount of points scored by the visiting team in the game (numeric)          |

```python
#  Importing the basic libraries 
import numpy as np 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
from IPython.display import display, HTML
import pandas as pd
import plotly.express as px
```

```python
games = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2025/games.csv')
```

```python
# Function to visualize the data
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
print_dataset_analysis(games, n_top=5, heading_color='#14adc6', text_color='white')
print_unique_values(games, heading_color='#14adc6', text_color='white')
```

<p style="background-image: url(https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSlzxVr7U6HD1hdR7LS89YvKB2gvUUU9CgluQ&usqp=CAU);font-family:Pacifico ,cursive;font-size:250%; color:white; text-align:center; border-radius: 0; padding:20px; font-weight: normal; border: 3px dashed #14adc6; box-shadow: 0px 5px 10px  rgba(0, 0, 0, 0.2);text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.4); background-size: cover; background-repeat: no-repeat; background-position: center;">Distribution Of The Dataset📊🏈</p>

```python
def create_pie_chart(dataframe, column, title):
    value_counts = dataframe[column].value_counts().reset_index()
    value_counts.columns = [column, 'count']
    
    fig = px.pie(value_counts, names=column, values='count', title=title, 
                 color_discrete_sequence=px.colors.qualitative.Plotly)
    fig.show()

create_pie_chart(games, 'week', 'Distribution of Games by Week')
create_pie_chart(games, 'homeTeamAbbr', 'Distribution of Home Teams')
create_pie_chart(games, 'visitorTeamAbbr', 'Distribution of Visitor Teams')
```

<p style="background-image: url(https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSlzxVr7U6HD1hdR7LS89YvKB2gvUUU9CgluQ&usqp=CAU);font-family:Pacifico ,cursive;font-size:250%; color:white; text-align:center; border-radius: 0; padding:20px; font-weight: normal; border: 3px dashed #14adc6; box-shadow: 0px 5px 10px  rgba(0, 0, 0, 0.2);text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.4); background-size: cover; background-repeat: no-repeat; background-position: center;">Insights from Descriptive Statistics: A Coach’s Perspective📊🏈</p>

```python
games.replace([float('inf'), float('-inf')], pd.NA, inplace=True)

print("Descriptive statistics for numeric columns:")
print(games[['season', 'week', 'homeFinalScore', 'visitorFinalScore']].describe())

plt.figure(figsize=(15, 5))

# Plot for 'week'
plt.subplot(1, 3, 1)
sns.histplot(games['week'], kde=False, bins=10, color='skyblue')
plt.title('Distribution of Weeks')
plt.xlabel('Week')
plt.ylabel('Frequency')

# Plot for 'homeFinalScore'
plt.subplot(1, 3, 2)
sns.histplot(games['homeFinalScore'], kde=False, bins=10, color='green')
plt.title('Distribution of Home Final Scores')
plt.xlabel('Home Final Score')
plt.ylabel('Frequency')

# Plot for 'visitorFinalScore'
plt.subplot(1, 3, 3)
sns.histplot(games['visitorFinalScore'], kde=False, bins=10, color='orange')
plt.title('Distribution of Visitor Final Scores')
plt.xlabel('Visitor Final Score')
plt.ylabel('Frequency')

# Display the plots
plt.tight_layout()
plt.show()
```

| **Insight**                     | **Details**                                                                                                            |
|----------------------------------|------------------------------------------------------------------------------------------------------------------------|
| **Uniform Season Context**      | Focus on 2022 season data for consistent analysis; tailor strategies to current performance dynamics.                  |
| **Performance Across Weeks**     | Average week at **4.85** indicates a mid-season point; evaluate week-by-week performance for trend analysis.        |
| **Home Team Dynamics**          | Average home score of **22.67** shows scoring capability; wide score range (**3-49**) suggests need for consistency.  |
| **Visitor Team Analysis**       | Visitor teams average **20.95**; leverage home-field advantage and exploit opponents' weaknesses in upcoming games.   |
| **Scoring Distribution**        | **25%** of games had home scores < **17**; focus on improving offensive strategies in practice sessions.               |
| **Competitive Balance Insight**  | Close averages indicate competitive league; prepare for every game and analyze outperforming teams for tactical insights.|

<p style="background-image: url(https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSlzxVr7U6HD1hdR7LS89YvKB2gvUUU9CgluQ&usqp=CAU);font-family:Pacifico ,cursive;font-size:250%; color:white; text-align:center; border-radius: 0; padding:20px; font-weight: normal; border: 3px dashed #14adc6; box-shadow: 0px 5px 10px  rgba(0, 0, 0, 0.2);text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.4); background-size: cover; background-repeat: no-repeat; background-position: center;">Analyzing game timing and performance📊🏈</p>
In this section, we will explore the timing of games and how it relates to team performance. We'll extract the day of the week and the hour of the game from the `gameDate` and `gameTimeEastern` columns, respectively. We will then visualize the distribution of games across the week and examine the relationship between game time and final scores for both home and visiting teams through bar charts and boxplots.

```python
games['gameDate'] = pd.to_datetime(games['gameDate'])

# 2. Extract the day of the week from 'gameDate'
games['dayOfWeek'] = games['gameDate'].dt.day_name()

# 3. Convert 'gameTimeEastern' to datetime format and extract the hour
games['gameTimeEastern'] = pd.to_datetime(games['gameTimeEastern'], format='%H:%M:%S').dt.time
games['gameHour'] = pd.to_datetime(games['gameTimeEastern'], format='%H:%M:%S').dt.hour

# 4. Bar chart: Number of games per day of the week
plt.figure(figsize=(10, 6))
sns.countplot(x='dayOfWeek', data=games, order=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
plt.title('Number of Games by Day of the Week')
plt.xlabel('Day of the Week')
plt.ylabel('Number of Games')
plt.show()

# 5. Boxplot: Relationship between game time (hour) and game scores (home and visitor)
plt.figure(figsize=(15, 6))

# Boxplot for homeFinalScore
plt.subplot(1, 2, 1)
sns.boxplot(x='gameHour', y='homeFinalScore', data=games)
plt.title('Home Final Score by Time of Day')
plt.xlabel('Game Hour')
plt.ylabel('Home Final Score')

# Boxplot for visitorFinalScore
plt.subplot(1, 2, 2)
sns.boxplot(x='gameHour', y='visitorFinalScore', data=games)
plt.title('Visitor Final Score by Time of Day')
plt.xlabel('Game Hour')
plt.ylabel('Visitor Final Score')

# Show the plots
plt.tight_layout()
plt.show()
```

# Simpler Explanation of the Visualizations

### Number of Games by Day of the Week:
- **Main Point**: Most NFL games are played on **Sundays**, with a few games on **Monday** and **Thursday**. This follows the usual NFL schedule, where Sunday is the main game day, and Monday and Thursday games are much less common.

### Home Final Score by Time of Day:
- **Main Point**: The home team’s scores change depending on the time of the game:
  - **9 AM** games: Home teams usually score between **15-25 points**.
  - **1 PM** games: Scores have a wider range, with many home teams scoring between **15-35 points**.
  - **4 PM** games: Scores are more consistent, with fewer very high or low scores.
  - **8 PM** games: Scores vary a lot, and home teams often score more points, possibly because these are important, high-profile games.

### Visitor Final Score by Time of Day:
- **Main Point**: Visitor teams also score differently depending on the time:
  - **9 AM** games: Visitors tend to score between **10-20 points**.
  - **1 PM and 4 PM** games: Visitors score between **10-30 points**, with a wider range.
  - **7 PM** games: Visitors usually score **under 20 points**.
  - **8 PM** games: Visitor teams often score more, likely because these are big, competitive games, just like for the home teams.

### General Insights:
- **Primetime (8 PM) games** usually have higher scores for both teams, which could mean that these games involve stronger teams or more aggressive strategies.
- **Sundays** are the main game day, fitting the regular NFL schedule.
- **Morning games (9 AM)** generally have lower scores, which might be due to the early start affecting performance.

<p style="background-image: url(https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSlzxVr7U6HD1hdR7LS89YvKB2gvUUU9CgluQ&usqp=CAU);font-family:Pacifico ,cursive;font-size:250%; color:white; text-align:center; border-radius: 0; padding:20px; font-weight: normal; border: 3px dashed #14adc6; box-shadow: 0px 5px 10px  rgba(0, 0, 0, 0.2);text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.4); background-size: cover; background-repeat: no-repeat; background-position: center;">Analyzing Home Vs Visting teams📊🏈</p>

In this analysis, we aim to compare the performance of home and visiting teams by examining average scores, win/loss ratios, and visualizing the results through boxplots and pie charts.

```python
# 1. Compare average homeFinalScore vs visitorFinalScore
avg_home_score = games['homeFinalScore'].mean()
avg_visitor_score = games['visitorFinalScore'].mean()

print(f"Average Home Final Score: {avg_home_score}")
print(f"Average Visitor Final Score: {avg_visitor_score}")

# 2. Calculate home team wins, visitor team wins, and ties
games['homeWin'] = games['homeFinalScore'] > games['visitorFinalScore']
games['visitorWin'] = games['homeFinalScore'] < games['visitorFinalScore']
games['tie'] = games['homeFinalScore'] == games['visitorFinalScore']

home_wins = games['homeWin'].sum()
visitor_wins = games['visitorWin'].sum()
ties = games['tie'].sum()

# 3. Calculate home team win/loss ratio
total_games = len(games)
home_win_percentage = (home_wins / total_games) * 100
visitor_win_percentage = (visitor_wins / total_games) * 100
tie_percentage = (ties / total_games) * 100

print(f"Home Wins: {home_wins}, Visitor Wins: {visitor_wins}, Ties: {ties}")
print(f"Home Win Percentage: {home_win_percentage:.2f}%")
print(f"Visitor Win Percentage: {visitor_win_percentage:.2f}%")
print(f"Tie Percentage: {tie_percentage:.2f}%")

# 4. Visualizations

# Boxplot comparing homeFinalScore and visitorFinalScore
plt.figure(figsize=(10, 6))
sns.boxplot(data=games[['homeFinalScore', 'visitorFinalScore']])
plt.title('Boxplot Comparison: Home vs Visitor Final Scores')
plt.ylabel('Score')
plt.xticks([0, 1], ['Home Final Score', 'Visitor Final Score'])
plt.show()

# Pie chart showing home vs visitor wins
plt.figure(figsize=(8, 8))
labels = ['Home Wins', 'Visitor Wins', 'Ties']
sizes = [home_wins, visitor_wins, ties]
colors = ['#66b3ff', '#ff9999', '#99ff99']
explode = (0.1, 0, 0)  # Explode the first slice (Home Wins)

plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
plt.title('Proportion of Home Wins vs Visitor Wins')
plt.show()
```

## Boxplot Comparison: Home vs. Visitor Final Scores
- **Home Final Scores**: Home teams typically score between **10 and 40 points**, with higher averages than visitor teams.
- **Visitor Final Scores**: Visitor teams generally score **10 to 30 points**, with less variability and fewer high scores.

## Proportion of Home Wins vs. Visitor Wins
- **Home Wins**: Home teams win **53.7%** of games.
- **Visitor Wins**: Visitor teams win **45.6%** of games, with **0.7%** ending in a tie.
- This indicates a slight home-field advantage.

## Applications for NFL Bowl Analysis
1. **Performance Predictions**: Use scoring insights to predict team performance based on home or visitor status.
2. **Strategic Decisions**: Inform game preparation and strategies based on historical scoring patterns.
3. **Betting Odds**: Help betting companies set accurate odds by analyzing home team advantages.
4. **Fan Engagement**: Promote home games, leveraging the statistical advantage for increased attendance.
5. **Historical Comparison**: Compare upcoming matchups against historical performance data for better forecasting.
6. **In-game Strategies**: Adjust tactics based on scoring patterns related to game location.

These insights enhance understanding of team performance and can be effectively utilized for strategic planning and analysis during the NFL Bowl.

<p style="background-image: url(https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSlzxVr7U6HD1hdR7LS89YvKB2gvUUU9CgluQ&usqp=CAU);font-family:Pacifico ,cursive;font-size:250%; color:white; text-align:center; border-radius: 0; padding:20px; font-weight: normal; border: 3px dashed #14adc6; box-shadow: 0px 5px 10px  rgba(0, 0, 0, 0.2);text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.4); background-size: cover; background-repeat: no-repeat; background-position: center;">Team Performance Analysis: Average Scores and Trends📊🏈</p>

This analysis focuses on the performance of teams based on their average scores at home and away. By examining these statistics, we can gain insights into team consistency and identify trends over the course of the season.

```python
# 1. Group by homeTeamAbbr and visitorTeamAbbr to check average scores for each team at home and away
home_avg_scores = games.groupby('homeTeamAbbr')['homeFinalScore'].mean().reset_index()
home_avg_scores.columns = ['Team', 'AvgHomeScore']

visitor_avg_scores = games.groupby('visitorTeamAbbr')['visitorFinalScore'].mean().reset_index()
visitor_avg_scores.columns = ['Team', 'AvgAwayScore']

# Merge home and away averages into one DataFrame
team_scores = pd.merge(home_avg_scores, visitor_avg_scores, on='Team', how='outer')

# 2. Identify teams with consistently high or low scores
top_home_teams = team_scores.nlargest(5, 'AvgHomeScore')
top_away_teams = team_scores.nlargest(5, 'AvgAwayScore')

bottom_home_teams = team_scores.nsmallest(5, 'AvgHomeScore')
bottom_away_teams = team_scores.nsmallest(5, 'AvgAwayScore')

print("Top 5 Home Teams by Average Score:")
print(top_home_teams)

print("\nTop 5 Away Teams by Average Score:")
print(top_away_teams)

print("\nBottom 5 Home Teams by Average Score:")
print(bottom_home_teams)

print("\nBottom 5 Away Teams by Average Score:")
print(bottom_away_teams)

# 3. Team performance across weeks or seasons
performance_by_week = games.groupby(['homeTeamAbbr', 'week'])['homeFinalScore'].mean().unstack()
performance_by_season = games.groupby(['homeTeamAbbr', 'season'])['homeFinalScore'].mean().unstack()

# 4. Visualizations

# Bar plot showing average home and away scores for each team
plt.figure(figsize=(14, 6))
team_scores.plot(x='Team', kind='bar', stacked=False, figsize=(14, 7), color=['#66b3ff', '#ff9999'])
plt.title('Average Home and Away Scores by Team')
plt.ylabel('Average Score')
plt.xlabel('Team')
plt.xticks(rotation=90)
plt.show()

# Heatmap for team performance across weeks
plt.figure(figsize=(12, 8))
sns.heatmap(performance_by_week, cmap='coolwarm', annot=True, fmt=".1f", linewidths=0.5)
plt.title('Team Performance Across Weeks (Home Games)')
plt.ylabel('Team')
plt.xlabel('Week')
plt.show()

# Heatmap for team performance across seasons
plt.figure(figsize=(8, 6))
sns.heatmap(performance_by_season, cmap='coolwarm', annot=True, fmt=".1f", linewidths=0.5)
plt.title('Team Performance Across Seasons (Home Games)')
plt.ylabel('Team')
plt.xlabel('Season')
plt.show()
```

# Insights from Team Performance Across Weeks (Home Games)

The heatmap displays the home game performances of NFL teams across different weeks, using color intensity to represent scores (red = higher, blue = lower). Key insights:

## 1. High-Scoring Teams at Home:
- **Cleveland Browns (CLE)**: Scored significantly high in week 4 (42.0) and week 6 (49.0).
- **Detroit Lions (DET)**: Consistently performed well with notable high scores in weeks 1 (35.0), 2 (36.0), and 3 (45.0).
- **New England Patriots (NE)**: High score in week 5 (39.0).
- **New York Giants (NYG)**: Scored high in week 6 (40.0).

## 2. Low-Scoring Teams at Home:
- **Chicago Bears (CHI)**: Scored low most weeks, with a peak in week 7 (32.0).
- **Houston Texans (HOU)**: Performed poorly, with a maximum score of 27.0 in week 6.

## 3. Consistent Performance:
- **Kansas City Chiefs (KC)**: Stable scores between 17.0-27.0 points across the season.
- **Buffalo Bills (BUF)**: Consistent, with strong scores in weeks 2 (41.0) and 7 (27.0).
- **Philadelphia Eagles (PHI)**: Scores ranged between 16.0 and 35.0 points.

## 4. Fluctuating Performance:
- **Cincinnati Bengals (CIN)**: Varied performance, from week 2 (20.0) to week 5 (35.0).
- **Minnesota Vikings (MIN)**: Performance fluctuated, with a spike in week 4 (29.0) and week 6 (34.0).

## 5. Poor Home Performance:
- **Las Vegas Raiders (LV)**: Scored very low, particularly in week 5 (3.0) and struggled to exceed 23.0 points.
- **Houston Texans (HOU)**: Rarely crossed 20 points, except for week 6 (27.0).

## 6. High-Scoring Weeks:
- **Week 6**: Notable high scores by Cleveland (49.0), New York Giants (40.0), and Philadelphia (35.0).
- **Week 2**: High scores by Detroit (36.0) and Buffalo (41.0).

<p style="background-image: url(https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSlzxVr7U6HD1hdR7LS89YvKB2gvUUU9CgluQ&usqp=CAU);font-family:Pacifico ,cursive;font-size:250%; color:white; text-align:center; border-radius: 0; padding:20px; font-weight: normal; border: 3px dashed #14adc6; box-shadow: 0px 5px 10px  rgba(0, 0, 0, 0.2);text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.4); background-size: cover; background-repeat: no-repeat; background-position: center;">Weekly Scoring Trends Analysis📊🏈</p>

Team Performance Analysis: Average Scores and Trends

```python
# 1. Calculate total points scored per week
games['totalScore'] = games['homeFinalScore'] + games['visitorFinalScore']
total_points_per_week = games.groupby('week')['totalScore'].sum().reset_index()

# 2. Calculate average home and visitor scores week-by-week
average_scores_week = games.groupby('week').agg(
    AvgHomeScore=('homeFinalScore', 'mean'),
    AvgVisitorScore=('visitorFinalScore', 'mean')
).reset_index()

# 3. Visualizations

# Line plot showing total points scored per week
plt.figure(figsize=(12, 6))
sns.lineplot(data=total_points_per_week, x='week', y='totalScore', marker='o', color='purple')
plt.title('Total Points Scored Per Week')
plt.xlabel('Week')
plt.ylabel('Total Points')
plt.xticks(total_points_per_week['week'])
plt.grid()
plt.show()

# Bar plot comparing average home and visitor scores week-by-week
plt.figure(figsize=(12, 6))
bar_width = 0.35
x = average_scores_week['week']

plt.bar(x - bar_width/2, average_scores_week['AvgHomeScore'], width=bar_width, label='Avg Home Score', color='blue')
plt.bar(x + bar_width/2, average_scores_week['AvgVisitorScore'], width=bar_width, label='Avg Visitor Score', color='orange')

plt.title('Average Home and Visitor Scores Week-by-Week')
plt.xlabel('Week')
plt.ylabel('Average Score')
plt.xticks(x)
plt.legend()
plt.grid()
plt.show()
```

<p style="background-image: url(https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSlzxVr7U6HD1hdR7LS89YvKB2gvUUU9CgluQ&usqp=CAU);font-family:Pacifico ,cursive;font-size:250%; color:white; text-align:center; border-radius: 0; padding:20px; font-weight: normal; border: 3px dashed #14adc6; box-shadow: 0px 5px 10px  rgba(0, 0, 0, 0.2);text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.4); background-size: cover; background-repeat: no-repeat; background-position: center;">Point Differences and Team Performance 📊🏈</p>

```python
# 1. Calculate point difference and categorize outcomes
games['pointDifference'] = games['homeFinalScore'] - games['visitorFinalScore']

# Create a new column to classify games
def classify_game(row):
    if abs(row['pointDifference']) <= 3:
        return 'Close Game'
    elif abs(row['pointDifference']) > 3 and abs(row['pointDifference']) <= 14:
        return 'Moderate Win'
    else:
        return 'Blowout'

games['gameType'] = games.apply(classify_game, axis=1)

# Scatter plot of point differences vs week number
scatter_fig = px.scatter(
    games,
    x='week',
    y='pointDifference',
    color='gameType',
    color_discrete_map={'Close Game': 'orange', 'Moderate Win': 'lightblue', 'Blowout': 'lightgreen'},
    size_max=15,
    title='Point Differences vs Week Number',
    labels={'week': 'Week Number', 'pointDifference': 'Point Difference (Home - Visitor)'}
)
scatter_fig.add_hline(y=0, line_dash='dash', line_color='red', annotation_text='Draw (Point Difference = 0)', annotation_position='top right')
scatter_fig.show()

# Boxplot comparing point differences for each team (home teams)
box_fig_home = px.box(
    games,
    x='homeTeamAbbr',
    y='pointDifference',
    color='homeTeamAbbr',
    title='Point Differences for Home Teams',
    labels={'homeTeamAbbr': 'Home Team Abbreviation', 'pointDifference': 'Point Difference (Home - Visitor)'}
)
box_fig_home.add_hline(y=0, line_dash='dash', line_color='red', annotation_text='Draw (Point Difference = 0)', annotation_position='top right')
box_fig_home.update_xaxes(tickangle=45)
box_fig_home.show()

# Boxplot comparing point differences for visitor teams
box_fig_visitor = px.box(
    games,
    x='visitorTeamAbbr',
    y='pointDifference',
    color='visitorTeamAbbr',
    title='Point Differences for Visitor Teams',
    labels={'visitorTeamAbbr': 'Visitor Team Abbreviation', 'pointDifference': 'Point Difference (Home - Visitor)'}
)
box_fig_visitor.add_hline(y=0, line_dash='dash', line_color='red', annotation_text='Draw (Point Difference = 0)', annotation_position='top right')
box_fig_visitor.update_xaxes(tickangle=45)
box_fig_visitor.show()

# Calculate average point differences for each team (home and visitor)
avg_point_diff_home = games.groupby('homeTeamAbbr').agg(AvgHomePointDiff=('pointDifference', 'mean')).reset_index()
avg_point_diff_home.columns = ['Team', 'Average Point Difference']
avg_point_diff_visitor = games.groupby('visitorTeamAbbr').agg(AvgVisitorPointDiff=('pointDifference', 'mean')).reset_index()
avg_point_diff_visitor.columns = ['Team', 'Average Point Difference']

# Combine data for plotting
avg_point_diff_combined = pd.merge(avg_point_diff_home, avg_point_diff_visitor, on='Team', how='outer', suffixes=('_Home', '_Visitor'))
avg_point_diff_combined = avg_point_diff_combined.melt(id_vars='Team', var_name='Game Type', value_name='Average Point Difference')

# Bar plot comparing average point differences for home and visitor teams
bar_fig = px.bar(
    avg_point_diff_combined,
    x='Team',
    y='Average Point Difference',
    color='Game Type',
    title='Average Point Differences for Home and Visitor Teams',
    labels={'Team': 'Team Abbreviation', 'Average Point Difference': 'Average Point Difference'}
)
bar_fig.add_hline(y=0, line_dash='dash', line_color='red', annotation_text='Draw (Point Difference = 0)', annotation_position='top right')
bar_fig.update_xaxes(tickangle=45)
bar_fig.show()
```