# NFL Big Data Bowl 2025

- **Author:** Marília Prata
- **Votes:** 116
- **Ref:** mpwolke/nfl-big-data-bowl-2025
- **URL:** https://www.kaggle.com/code/mpwolke/nfl-big-data-bowl-2025
- **Last run:** 2024-10-11 03:00:47.230000

---

Published on October 10, 2024. By Marília Prata, mpwolke.

```python
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

import plotly.graph_objs as go
import plotly.offline as py
import plotly.express as px

#Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
```

## NFL Competition is Back

![](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSGQrVn8qU1-3UBo8DkhkNCGFuxhOQPDSUHgg&usqp=CAU)
https://www.espn.com/nfl/story/_/id/33625814/philadelphia-eagles-announce-return-kelly-green-alternate-uniforms-2023-nfl-season

## Competition Citation

@misc{nfl-big-data-bowl-2025,

    author = {Michael Lopez, Thompson Bliss, Ally Blake, Paul Mooney, Addison Howard}
    ,
    title = {NFL Big Data Bowl 2025
    },
    publisher = {Kaggl
    e},
    year = {2024},
    url = {https://kaggle.com/competitions/nfl-big-data-bowl-2025}

```python
player_play = pd.read_csv("../input/nfl-big-data-bowl-2025/player_play.csv")
games = pd.read_csv("../input/nfl-big-data-bowl-2025/games.csv")
plays = pd.read_csv("../input/nfl-big-data-bowl-2025/plays.csv")
players = pd.read_csv("../input/nfl-big-data-bowl-2025/players.csv")
```

### Brady still on top:)

```python
players.head()
```

### If any data doesn't match format "%Y-%m-%d". Use format='mixed'

On the snippet below, 2nd line https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.to_datetime.html

```python
#https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.to_datetime.html

print("Data type of birthDate column before parsing : ", players["birthDate"].dtypes)
players["birthDate"] = pd.to_datetime(players["birthDate"], format='mixed')
print("Data type of birthDate column after parsing : ", players["birthDate"].dtypes)
print(players["birthDate"].head())
```

## Splitting YYYY (Year) from YYYY-MM-DD date format

```python
players['birthYear'] = pd.DatetimeIndex(players['birthDate']).year
print(players["birthYear"])
```

## Unique values of birthYear column

```python
print("Unique birth year values and their counts :")
print(players["birthYear"].value_counts())
```

## Youngest and Oldest Player birth year

```python
#Sanjay V https://www.kaggle.com/code/sanjayv007/nfl-big-data-bowl-beginner-s-complete-eda

# Newest and oldest player
print("Youngest player birth year : ",max(players["birthYear"]))
print("Oldest player birth year : ",min(players["birthYear"]))
```

### Players Birth years histogram

The higher bars are the represented by 1995-1996 years where we have more observations on their respective bins (17 and 18). More players were born on these two years.

It's a left skewed distribution.

https://www.labxchange.org/library/items/lb:LabXchange:10d3270e:html:1

```python
hist = players["birthYear"].plot.hist(bins=20, color="orange", edgecolor="black")
```

### Unique values of collegeName column

```python
print("Unique college names and their counts : ")
college_names = players.pivot_table(index = ['collegeName'], aggfunc = 'size') 
college_names = college_names.reset_index()
college_names.columns= ["College Names", "Counts"]
college_names = college_names.sort_values("Counts", ascending = False)
print(college_names)
```

## Top 10 colleges having higher number of players

```python
top_colleges = college_names[0:10]
print(top_colleges)
```

```python
# Creating Donut Plot
fig = plt.figure(figsize = (8, 8)) 
circle = plt.Circle( (0,0), 0.5, color = 'white')
plt.pie(top_colleges["Counts"], labels = top_colleges["College Names"])
p = plt.gcf()
p.gca().add_artist(circle)
#plt.legend(loc=(1.04, 1))
plt.legend(top_colleges["Counts"])
plt.title("Top 10 Colleges Having Higher Number Of Players", fontsize=25)

# Displaying Donut Plot
plt.show()
```

### Unique values of Position column

```python
print("Unique position values and their counts :")
pos_val = players.pivot_table(index = ['position'], aggfunc = 'size') 
pos_val = pos_val.reset_index()
pos_val.columns= ["Positions", "Counts"]
pos_val = pos_val.sort_values("Counts", ascending = False)
print(pos_val)
```

### Players: highest height in feet

```python
height = players[players['height'] == max(players["height"])]
height
```

### Players: lowest height in feet

```python
low_height = players[players['height'] == min(players["height"])]
low_height
```

### Oldest and youngest players

```python
oldest = players[players['birthYear'] == min(players["birthYear"])]
oldest
```

```python
youngest = players[players['birthYear'] == max(players["birthYear"])]
youngest
```

```python
mean=np.ceil(players['weight'].mean())
median=np.ceil(players['weight'].median())
```

### Players Weight Distribution

It's a normal distribution with mean value 246 and median value 236 (pounds of weight).

https://www.labxchange.org/library/items/lb:LabXchange:10d3270e:html:1

```python
#Code by Chinta https://www.kaggle.com/chinta/mlb-is-player-age-important

plt.figure(figsize=(10, 5))
sns.set_style('white')
hist_plot = sns.histplot(players['weight'], )
hist_plot.axvline(mean, color='r', linestyle='--', linewidth = 4, label = f'mean-{mean}')
hist_plot.axvline(median, color='g', linestyle='-', linewidth = 4, label = f'median-{median}')
plt.suptitle("Players Weight Distribution")
plt.legend();
```

```python
games.tail()
```

### Unique values of season column

```python
#By Sanjay V https://www.kaggle.com/code/sanjayv007/nfl-big-data-bowl-beginner-s-complete-eda

print("Unique NFL seasons and their counts :")
g_season = games.pivot_table(index = ['season'], aggfunc = 'size') 
g_season = g_season.reset_index()
g_season.columns= ["Seasons", "Counts"]
g_season = g_season.sort_values("Counts", ascending = False)
print(g_season)
```

```python
#By Sanjay V https://www.kaggle.com/code/sanjayv007/nfl-big-data-bowl-beginner-s-complete-eda

print("Unique NFL weeks and their counts :")
g_week = games.pivot_table(index = ['week'], aggfunc = 'size') 
g_week = g_week.reset_index()
g_week.columns= ["Weeks", "Counts"]
g_week = g_week.sort_values("Counts", ascending = False)
print(g_week)
```

```python
# Add titles, labels, invert y-axis

bar_plot = g_week.plot.barh()
bar_plot.set_title("Unique NFL weeks and their counts")
bar_plot.set_xlabel("Counts")
bar_plot.set_ylabel("Weeks")
bar_plot.invert_yaxis()
plt.show(bar_plot)
```

## Unique values of gameDate column

```python
#By Sanjay V https://www.kaggle.com/code/sanjayv007/nfl-big-data-bowl-beginner-s-complete-eda

print("Unique NFL dates and their counts :")
g_date = games.pivot_table(index = ['gameDate'], aggfunc = 'size') 
g_date = g_date.reset_index()
g_date.columns= ["Date", "Counts"]
g_date = g_date.sort_values("Counts", ascending = False)
print(g_date)
```

### NFL Dates

```python
# Add titles, labels, invert y-axis

bar_plot = g_date.plot.barh()
bar_plot.set_title("NFL Events Dates")
bar_plot.set_xlabel("Counts")
bar_plot.set_ylabel("Dates")
bar_plot.invert_yaxis()
plt.show(bar_plot)
```

### Splitting DD (Date) from MM/DD/YYYY date format

```python
games['gameDay'] = pd.DatetimeIndex(games['gameDate']).day
print(games["gameDay"])
```

### Unique values from gameDay column

```python
#By Sanjay V https://www.kaggle.com/code/sanjayv007/nfl-big-data-bowl-beginner-s-complete-eda

print("Unique NFL days and their counts :")
g_days = games.pivot_table(index = ['gameDay'], aggfunc = 'size') 
g_days = g_days.reset_index()
g_days.columns= ["Day", "Counts"]
g_days = g_days.sort_values("Counts", ascending = False)
print(g_days)
```

```python
# Add titles, labels, invert y-axis

bar_plot = g_days.plot.barh()
bar_plot.set_title("NFL Events Days")
bar_plot.set_xlabel("Counts")
bar_plot.set_ylabel("Days")
bar_plot.invert_yaxis()
plt.show(bar_plot)
```

#Unique values from gameTimeEastern column

```python
#By Sanjay V https://www.kaggle.com/code/sanjayv007/nfl-big-data-bowl-beginner-s-complete-eda

print("Unique NFL timings and their counts :")
g_time = games.pivot_table(index = ['gameTimeEastern'], aggfunc = 'size') 
g_time = g_time.reset_index()
g_time.columns= ["Time", "Counts"]
g_time = g_time.sort_values("Counts", ascending = False)
print(g_time)
```

### I didn't get NFL Events most frequent hour is 13:00??

What times does NFL play?
"
The majority of games are played on Sunday, most kicking off at 1PM (ET), with some late afternoon games starting at either 4:05 or 4:25 PM (ET). Additionally, one Sunday night game is played every week at 8:20 PM (ET)"

https://en.wikipedia.org/wiki/NFL_regular_season#:~:text=The%20majority%20of%20games%20are,%3A20%20PM%20(ET)..

```python
#Code by Lucas Abrahão https://www.kaggle.com/lucasabrahao/trabalho-manufatura-an-lise-de-dados-no-brasil

#sort_values so the values in our bar chart will be in increasing order

games["gameTimeEastern"].value_counts().sort_values().plot.barh(color=['blue', '#f5005a'], title='NFL Events Time')
plt.xlabel('Counts');
```

### Unique values from homeTeamAbbr column

```python
#By Sanjay V https://www.kaggle.com/code/sanjayv007/nfl-big-data-bowl-beginner-s-complete-eda

print("Unique NFL home teams and their counts :")
g_home = games.pivot_table(index = ['homeTeamAbbr'], aggfunc = 'size') 
g_home = g_home.reset_index()
g_home.columns= ["Home Team", "Counts"]
g_home = g_home.sort_values("Counts", ascending = False)
print(g_home)
```

#### We've 30 teams

Since we have for the 30 teams almost the same number of entries, any chart won't bring information, except the name of the teams by itself.

```python
#Code by Lucas Abrahão https://www.kaggle.com/lucasabrahao/trabalho-manufatura-an-lise-de-dados-no-brasil

g_home["Home Team"].value_counts().head(20).plot.barh(color='purple', title='NFL Home Teams')
plt.xlabel('Counts');
```

### Unique values from yardsToGo column

```python
#By Sanjay V https://www.kaggle.com/code/sanjayv007/nfl-big-data-bowl-beginner-s-complete-eda

print("Unique NFL yards to go and their counts :")
g_yards = plays.pivot_table(index = ['yardsToGo'], aggfunc = 'size') 
g_yards = g_yards.reset_index()
g_yards.columns= ["Yards To Go", "Counts"]
g_yards = g_yards.sort_values("Counts", ascending = False)
print(g_yards)
```

```python
# Add titles, labels, invert y-axis

bar_plot = g_yards.plot.barh()
bar_plot.set_title("NFL, Yards to Go")
bar_plot.set_xlabel("Counts")
bar_plot.set_ylabel("Yards to Go ")
bar_plot.invert_yaxis() #order increasing
plt.show(bar_plot)
```

### Unique values from OffenseFormation column

```python
#By Sanjay V https://www.kaggle.com/code/sanjayv007/nfl-big-data-bowl-beginner-s-complete-eda

print("Unique NFL Offense Formation and their counts :")
gp_type = plays.pivot_table(index = ['offenseFormation'], aggfunc = 'size') 
gp_type = gp_type.reset_index()
gp_type.columns= ["Offense Formation", "Counts"]
gp_type = gp_type.sort_values("Counts", ascending = False)
print(gp_type)
```

## NFL Offense Formation

EMPTY backfield

"Also known simply as "Five-wide", a reference to the five wide receivers. In the empty backfield formation, all of the backs play near the line of scrimmage to act as extra wide receivers or tight ends, with the quarterback lining up either under center or, most commonly, in the shotgun. This is almost exclusively a passing formation used to spread the field, often to open up short inside routes or screen routes. The most common running play from this formation is a quarterback draw play up the middle since defensive players are spread out from sideline to sideline."

PISTOL

"It is essentially a shotgun variation, with the quarterback lined up closer than in standard shotgun (normally 3 to 4 yards behind center), and a running back lined up behind, rather than next to, the QB (normally at 3 to 4 yards behind quarterback). The pistol formation adds the dimension of a running game with the halfback being in a singleback position. This has disrupted the timing of some defenses with the way the quarterback hands the ball off to the halfback. This also allows the smaller halfbacks to hide behind the offensive line, causing opposing linebackers and pass-rushing defensive linemen to play more conservatively."

JUMBO

"Goal line formation. Also called "jumbo", "heavy", "full house" and other similar names, this formation is used exclusively in short-yardage situations, and especially near the goal line. This formation typically has no wide receivers, and often employs 3 tight ends and 2 running backs, or alternately 2 tight ends and 3 running backs. Often, a tight end or full back position is occupied by a player who normally plays offensive line or defensive line positions to act as an extra blocker."

WILDCAT

"The wildcat is primarily a running formation in which an athletic player (usually a running back or a receiver who runs well) takes the place of the team's usual quarterback in a shotgun formation while the quarterback lines up wide as a flanker or is replaced by another player. The ball is snapped to the runner, who usually has the option of either running the ball himself or handing it to another running back lined up in the backfield. The wildcat gives the runner a good look at the defense before the snap, allowing him to choose the best running lane. It also allows for ten offensive players to block, unlike in a conventional running play, in which the quarterback is usually not involved after delivering the ball to a running back."

https://en.wikipedia.org/wiki/List_of_formations_in_American_football

```python
#Code by Lucas Abrahão https://www.kaggle.com/lucasabrahao/trabalho-manufatura-an-lise-de-dados-no-brasil

plays["offenseFormation"].value_counts().plot.barh(color='orange', title='NFL Offense Formation')
plt.xlabel('Counts');
```

### Unique values from pre-snap homeTeam Win Probability column

```python
#By Sanjay V https://www.kaggle.com/code/sanjayv007/nfl-big-data-bowl-beginner-s-complete-eda

print("Unique NFL Pre-snap Home Team Win Probability and their counts :")
g_home = plays.pivot_table(index = ['preSnapHomeTeamWinProbability'], aggfunc = 'size') 
g_home = g_home.reset_index()
g_home.columns= ["Pre-Snap HomeTeam Win Probability", "Counts"]
g_home = g_home.sort_values("Counts", ascending = False)
print(g_home)
```

### NFL Pre-Snap HomeTeam Win Probability Distribution (Histogram)

It's normal, unimodal, symetric distribution. 

I'm still trying to undertand what's a "**NFL Pre-Snap HomeTeam Win Probability normal, unimodal, symetric distribution**"

```python
hist = plays["preSnapHomeTeamWinProbability"].plot.hist(bins=25, color="orange", edgecolor="black")
plt.title('NFL Pre-Snap HomeTeam Win Probability');
```

### Unique values of pass result column

```python
#By Sanjay V https://www.kaggle.com/code/sanjayv007/nfl-big-data-bowl-beginner-s-complete-eda

print("Unique NFL pass results and their counts :")
g_res = plays.pivot_table(index = ['passResult'], aggfunc = 'size') 
g_res = g_res.reset_index()
g_res.columns= ["Pass Results", "Counts"]
g_res = g_res.sort_values("Counts", ascending = False)
print(g_res)
```

### Pass Result: C, I, S, R, IN

Dropback outcome of the play (C: **Complete pass**, I: **Incomplete** pass, S: **Quarterback sack**, IN: **Intercepted** pass, R: **Scramble**, text)

```python
#Code by Lucas Abrahão https://www.kaggle.com/lucasabrahao/trabalho-manufatura-an-lise-de-dados-no-brasil

#sort_values() to rank increasing values

plays["passResult"].value_counts().sort_values().plot.barh(color='red', title='NFL Pass Results')
plt.xlabel('Counts');
```

### Unique values from absolute yardline number column

```python
#By Sanjay V https://www.kaggle.com/code/sanjayv007/nfl-big-data-bowl-beginner-s-complete-eda

print("Unique NFL absolute yardline numbers and their counts :")
g_abyl = plays.pivot_table(index = ['absoluteYardlineNumber'], aggfunc = 'size') 
g_abyl = g_abyl.reset_index()
g_abyl.columns= ["Absolute YardLine Number", "Counts"]
g_abyl = g_abyl.sort_values("Counts", ascending = False)
print(g_abyl)
```

```python
#Code by Lucas Abrahão https://www.kaggle.com/lucasabrahao/trabalho-manufatura-an-lise-de-dados-no-brasil

#sort_values() to rank increasing values

plays["absoluteYardlineNumber"].value_counts().head(20).sort_values().plot.barh(color='red', title='NFL Absolute Yard Line Number')
plt.xlabel('Counts');
```

```python
tracking1 = pd.read_csv("../input/nfl-big-data-bowl-2025/tracking_week_1.csv")
tracking1.head()
```

#Splitting date from datetime

```python
#By Sanjay V https://www.kaggle.com/code/sanjayv007/nfl-big-data-bowl-beginner-s-complete-eda

tracking1['date'] = pd.DatetimeIndex(tracking1['time']).date
print(tracking1["date"])
```

### Unique values from date column

```python
#By Sanjay V https://www.kaggle.com/code/sanjayv007/nfl-big-data-bowl-beginner-s-complete-eda

print("Unique NFL dates and their counts :")
tr_date = tracking1.pivot_table(index = ['date'], aggfunc = 'size') 
tr_date = tr_date.reset_index()
tr_date.columns= ["Date", "Counts"]
tr_date = tr_date.sort_values("Counts", ascending = False)
print(tr_date)
```

#x_ticks  Rotation=0

September 11, 2022 had more NFL events

```python
#Code by Lucas Abrahão https://www.kaggle.com/lucasabrahao/trabalho-manufatura-an-lise-de-dados-no-brasil

#sort_values() to rank increasing values

tracking1["date"].value_counts().sort_values().plot.bar(color='red', title='NFL dates')
plt.xticks(rotation=0)
plt.ylabel('Counts');
```

#Creating new dataset using playId, club and gameId

```python
data = tracking1.query('playId == 56 and gameId == 2022090800')
print(data[["x", "y", "club"]])
```

### Players positions

x: Player position along the long axis of the field, 0 - 120 yards. See Figure 1 below. (numeric)

y: Player position along the short axis of the field, 0 - 53.3 yards. See Figure 1 below. (numeric

club: Team abbrevation of corresponding player (text)

https://www.kaggle.com/competitions/nfl-big-data-bowl-2025/data?select=tracking_week_1.csv)

#Scatter Player Positions - Alien alike :D

It's cool, seems 3D without being 3D. 

However, it's hard for a non-coder like me to interpret the meaning of this scatter plot. In fact, it seems an Alien : D 

I think **Rob Mulla** was the **1st** to plot something like that I tried below. Many kagglers copied him.
Always Robikscube to bring those awesome stuff. Though, I ruined it It was Plotly, though Plotly isn't rendering, therefore I worked with **Seaborn**..

```python
# define a custom palette
my_palette = ["#95a5a6", "#e74c3c", "#34495e"]
sns.pairplot(
    data,
    x_vars=["x"],
    y_vars=["y"],
    height=3.5,
    hue="club", #hue define the color-code variable
    palette=my_palette,   # <-- see here, custom palette
)
plt.title('Players Positions')
plt.show()
```

```python
#By Lennart Haupts https://www.kaggle.com/code/lennarthaupts/getting-a-feel-for-the-tracking-data

# only looking at data from plays when the home team is on the offense
right = tracking1[tracking1['playDirection'] == 'right']
# only looking at a specific match
match = right[right['gameId'] == 2022090800]
```

#Tackles Heatmap

I'm still trying to get those tackles and mostly this Heatmap.

```python
#By Lennart Haupts https://www.kaggle.com/code/lennarthaupts/getting-a-feel-for-the-tracking-data

fig, ax = plt.subplots(figsize=(15,10))
plt.hist2d(right['x'][right['event'] == 'tackle'], right['y'][right['event'] == 'tackle'],bins=70, cmap='summer')
plt.xlim(0 , 120)
plt.ylim(0,  53.3)
plt.title('Heatmap of all player locations during a tackle when the offense is moving to the right')
plt.show()
```

### NFL Clubs 2025

```python
#By Rob Mulla https://www.kaggle.com/code/robikscube/sign-language-recognition-eda-twitch-stream

fig, ax = plt.subplots(figsize=(6, 4)) #figsize=(width, height))
tracking1["club"].value_counts().head(10).sort_values(ascending=True).plot(
    kind="barh", color='g', ax=ax, title="NFL 2025"
)
ax.set_xlabel("Number of Training Examples")
plt.show()
```

#Acknowledgements:

Sanjay V https://www.kaggle.com/code/sanjayv007/nfl-big-data-bowl-beginner-s-complete-eda

Chinta https://www.kaggle.com/chinta/mlb-is-player-age-important

Lennart Haupts https://www.kaggle.com/code/lennarthaupts/getting-a-feel-for-the-tracking-data

Rob Mulla https://www.kaggle.com/code/robikscube/sign-language-recognition-eda-twitch-stream