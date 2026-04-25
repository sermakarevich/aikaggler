# 📊NFL Big Data Bowl:Games Comprehensive Analysis🏈

- **Author:** maria nadeem
- **Votes:** 86
- **Ref:** marianadeem755/nfl-big-data-bowl-games-comprehensive-analysis
- **URL:** https://www.kaggle.com/code/marianadeem755/nfl-big-data-bowl-games-comprehensive-analysis
- **Last run:** 2024-11-17 16:57:13.447000

---

# <p style="background-color: #FFDDC1; font-family: 'Palatino Linotype', 'Book Antiqua', Palatino, serif; color: #8a0808; font-size: 1.5em; text-align: center; border-radius: 35px 20px 35px 20px; padding: 15px 25px; line-height: 1.5; border: 4px double #8a0808; box-shadow: 5px 5px 20px rgba(0, 0, 0, 0.5), -5px -5px 20px rgba(0, 0, 0, 0.3); margin: 0 auto; width: auto;"> 📊🔍NFL Big Data Bowl Games Comprehensive Analysis: Uncovering Key Insights and Trends🏈</p>

![_219c6719-bfd8-4da1-80c1-94ec896b21fc (1).jpg](attachment:9a0e06f1-0bd9-45ad-a3bf-5f44a5f399ce.jpg)

<p style="text-align: justify; font-family: 'Times New Roman', serif; font-size: 24px; color: #8B0000; font-weight: bold;">
    Table of Contents
</p>
<!-- Link pointing to the correct id -->
<ul style="font-family: 'Georgia', serif; font-size: 14px; color: black; background-color: #FFDDC1; padding: 15px; border-radius: 10px;">
    <li><a href="#import_lib" style="color: black; font-family: 'Arial', sans-serif;">Import Libraries</a></li>
<ul style="font-family: 'Georgia', serif; font-size: 14px; color: black; background-color: #FFDDC1; padding: 15px; border-radius: 10px;">
    <li><a href="#game_id_column" style="color: black; font-family: 'Arial', sans-serif;">Let's Explore the Game Id Column</a></li>
    <li><a href="#season_homeabbr_column" style="color: black; font-family: 'Arial', sans-serif;">Let's Explore the Season and HomeAbbr Column</a></li>
    <li><a href="#visualize_season_homeabbr_bar_chart" style="color: black; font-family: 'Arial', sans-serif;">Let's Visualize the Season and HomeAbbr Column Using Bar Chart</a></li>
    <li><a href="#visualize_season_homeabbr_bar_pie_chart_subplots" style="color: black; font-family: 'Arial', sans-serif;">Let's Visualize the Season and HomeAbbr Pie Chart and Bar Chart Subplot</a></li>
    <li><a href="#explore_season_visitorteamabbr_column" style="color: black; font-family: 'Arial', sans-serif;">Let's Explore the Season and VisitorTeamAbbr Column</a></li>
    <li><a href="#visualize_season_visitorteamabbr_barchart" style="color: black; font-family: 'Arial', sans-serif;">Let's Visualize the Season and VisitorTeamAbbr Column using BarChart</a></li>
    <li><a href="#visualize_season_visitorteamabbr_columns" style="color: black; font-family: 'Arial', sans-serif;">Let's Visualize the Season and VisitorTeamAbbr Columns</a></li>
    <li><a href="#explore-season-column" style="color: black; font-family: 'Arial', sans-serif;">Let's Explore the Season Column</a></li>
    <li><a href="#explore-seasonal-home-team-score" style="color: black; font-family: 'Arial', sans-serif;">Let's Explore the Seasonal Home Team Score</a></li>
    <li><a href="#visualize-seasonal-home-team-score" style="color: black; font-family: 'Arial', sans-serif;">Let's Visualize the Seasonal Home Team Score</a></li>
    <li><a href="#explore-win-rate-home-visitor-scores" style="color: black; font-family: 'Arial', sans-serif;">Let's Explore the Win Rate of Home Final Score and Visitor Final Score</a></li>
    <li><a href="#visualize-home-team-win-rate-season" style="color: black; font-family: 'Arial', sans-serif;">Let's Visualize the Win Rate of Home Team by Season</a></li>
    <li><a href="#home-team-win-rate-season-pie-chart" style="color: black; font-family: 'Arial', sans-serif;">Home Team Win Rate by Season Pie Chart Visualization</a></li>
    <li><a href="#visitor-team-win-rate-season" style="color: black; font-family: 'Arial', sans-serif;">Let's Explore the Visitor Team Win Rate by Season</a></li>
    <li><a href="#visitor-team-win-rate-season-pie" style="color: black; font-family: 'Arial', sans-serif;">Visitor Team Win Rate by Season Pie Chart Visualization</a></li>
    <li><a href="#highest-scoring-games-per-team-season" style="color: black; font-family: 'Arial', sans-serif;">Let's Explore the Highest Scoring Games Per Team and Season</a></li>
    <li><a href="#visualize-highest-scoring-games-distribution" style="color: black; font-family: 'Arial', sans-serif;">Let's Visualize the Distribution of Highest Scoring Games</a></li>
    <li><a href="#visualize-home-vs-visitor-team-scoring" style="color: black; font-family: 'Arial', sans-serif;">Let's Visualize the Home Vs Visitor Team Scoring of Games</a></li>
    <li><a href="#explore-home-vs-visitor-team-average-scores" style="color: black; font-family: 'Arial', sans-serif;">Let's Explore the Home Vs Visitor Team Average Scores</a></li>
    <li><a href="#games-date-and-time-trend-analysis" style="color: black; font-family: 'Arial', sans-serif;">Games Date and Time Trend Analysis</a></li>
    <li><a href="#games-trend-days-of-week" style="color: black; font-family: 'Arial', sans-serif;">Analyze the Games Trend on the Basis of Days of Week</a></li>
    <li><a href="#games-played-at-each-time" style="color: black; font-family: 'Arial', sans-serif;">Let's Analyze the Games Played at Each Time</a></li>
    <li><a href="#games-played-by-home-teams" style="color: black; font-family: 'Arial', sans-serif;">Let's Explore the Games Played by Home Teams</a></li>
    <li><a href="#games-played-by-visitor-teams" style="color: black; font-family: 'Arial', sans-serif;">Let's Explore the Games Played by Visitor Teams</a></li>
    <li><a href="#nfl-home-visitor-scores-analysis" style="color: black; font-family: 'Arial', sans-serif;">Let's Visualize the NFL Home & Visitor Scores Analysis</a></li>
    <li><a href="#nfl-average-final-score-home-visitor" style="color: black; font-family: 'Arial', sans-serif;">Let's Visualize the NFL Average Final Score of Home Vs Visitor Teams Over Weeks</a></li>
    <li><a href="#home-visitor-final-scores-distribution" style="color: black; font-family: 'Arial', sans-serif;">Let's Visualize the Frequency Distribution of Home Vs Visitor Teams Final Scores</a></li>
    <li><a href="#home-vs-visitor-scores-trend" style="color: black; font-family: 'Arial', sans-serif;">Home Vs Visitor Teams Final Scores Trend Analysis</a></li>
    <li><a href="#games-played-per-day-over-seasons" style="color: black; font-family: 'Arial', sans-serif;">Let's Visualize the Number of Games Played per Day Over Seasons</a></li>
    <li><a href="#games-played-per-week-over-seasons" style="color: black; font-family: 'Arial', sans-serif;">Let's Visualize the Number of Games Played per Week Over Seasons</a></li>
    <li><a href="#sunburst-plot-games-per-day-over-seasons" style="color: black; font-family: 'Arial', sans-serif;">Let's Visualize the Sunburst Plot of Number of Games Played Per Day Over Seasons</a></li>
    <li><a href="#scores-distribution-by-week" style="color: black; font-family: 'Arial', sans-serif;">Let's Visualize the Scores Distribution by Week</a></li>
    <li><a href="#visualize-scores-heatmap" style="color: black; font-family: 'Arial', sans-serif;">Let's Visualize the Scores Heatmap</a></li>
    <li><a href="#scores-treemap" style="color: black; font-family: 'Arial', sans-serif;">Let's Visualize Scores TreeMap</a></li>
    <li><a href="#visualize-numeric-columns-distribution" style="color: black; font-family: 'Arial', sans-serif;">Let's Visualize the Distribution of Numeric Columns</a></li>
</ul>

<div style="background-color: #FFDDC1; border: 3px solid #C0392B; border-radius: 12px; padding: 20px; margin: 20px; box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1); font-family: 'Verdana', sans-serif;">
    <h2 style="color: #C0392B; font-size: 28px; text-align: center;">About Dataset</h2>
    <p style="color: #34495E; font-size: 16px; line-height: 1.8;">The Competition Dataset NFL Big Data Bowl 2025. These datasets offers an important and key insights into various aspects of the game, from player statistics to game dynamics. Each dataset or CSV File plays a key role in analyzing football Games, plays and player movements which help to develop strategies to predict game outcomes effectively. The NFL Big Data Bowl 2025 provides a rich set of data that helps to dive deep into every aspect of football. Whether it’s understanding how players move, analyzing team strategies, or predicting game outcomes, these datasets provides the insights for anyone interested in football analytics this competition offers a great opportunity to explore most exciting sports in the world</p>

# <span id="#nfl-big-data-bowl-2025-dataset-overview" style="color: #FFFFFF; background-color: #C0392B; padding: 20px; border-radius: 12px; text-align: center; display: block; font-size: 24px; font-family: 'Arial', sans-serif; font-weight: bold;">NFL Big Data Bowl 2025 Dataset Overview</span>
</div>

<div style="background-color: #F9F9F9; border: 2px solid #C0392B; padding: 20px; border-radius: 15px; margin-top: 20px;">
    <h2 style="color: #C0392B; font-size: 24px; text-align: center;">File Descriptions</h2>
    <table style="border-collapse: collapse; width: 100%; margin-top: 20px;">
        <thead style="background-color: #C0392B; color: #FFFFFF;">
            <tr>
                <th style="border: 1px solid #C0392B; padding: 12px; text-align: left; font-size: 18px;">File Name</th>
                <th style="border: 1px solid #C0392B; padding: 12px; text-align: left; font-size: 18px;">Description</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td style="border: 1px solid #C0392B; padding: 12px; font-size: 16px; font-weight: bold;">Tracking Data</td>
                <td style="border: 1px solid #C0392B; padding: 12px; font-size: 16px;">The <strong>tracking</strong> file contains detailed tracking data for each play, including information about the players' positions and movements throughout the game. Key columns include <strong>gameId</strong>, <strong>playId</strong>, and <strong>nflId</strong>.</td>
            </tr>
            <tr>
                <td style="border: 1px solid #C0392B; padding: 12px; font-size: 16px; font-weight: bold;">Plays Data</td>
                <td style="border: 1px solid #C0392B; padding: 12px; font-size: 16px;">The <strong>plays</strong> file contains information about each play in the game, with important columns such as <strong>gameId</strong>, <strong>playId</strong>, and <strong>playDescription</strong>.</td>
            </tr>
            <tr>
                <td style="border: 1px solid #C0392B; padding: 12px; font-size: 16px; font-weight: bold;">Players Data</td>
                <td style="border: 1px solid #C0392B; padding: 12px; font-size: 16px;">The <strong>players</strong> file contains player-specific information like height, weight, and college name, identified by the unique <strong>nflId</strong>.</td>
            </tr>
            <tr>
                <td style="border: 1px solid #C0392B; padding: 12px; font-size: 16px; font-weight: bold;">Player Play Data</td>
                <td style="border: 1px solid #C0392B; padding: 12px; font-size: 16px;">The <strong>player_play</strong> file contains statistics related to each player's performance during a specific play, with key columns such as <strong>gameId</strong>, <strong>playId</strong>, and <strong>nflId</strong>.</td>
            </tr>
            <tr>
                <td style="border: 1px solid #C0392B; padding: 12px; font-size: 16px; font-weight: bold;">Games Data</td>
                <td style="border: 1px solid #C0392B; padding: 12px; font-size: 16px;">The <strong>games</strong> file provides information about each game, including details such as teams, scores, and the game schedule.</td>
            </tr>
        </tbody>
    </table>


</div>
<div style="background-color: #F9F9F9; border: 2px solid #C0392B; padding: 20px; border-radius: 15px; margin-top: 20px;">
    <h2 style="color: #C0392B; font-size: 24px; text-align: center;">Games Data Column Descriptions</h2>
    <table style="border-collapse: collapse; width: 100%; margin-top: 20px;">
        <thead style="background-color: #C0392B; color: #FFFFFF;">
            <tr>
                <th style="border: 1px solid #C0392B; padding: 12px; text-align: left; font-size: 14px;">Variable</th>
                <th style="border: 1px solid #C0392B; padding: 12px; text-align: left; font-size: 14px;">Description</th>
                <th style="border: 1px solid #C0392B; padding: 12px; text-align: left; font-size: 14px;">Data Type</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">gameId</td>
                <td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Unique identifier for the game.</td>
                <td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Integer</td>
            </tr>
            <tr>
                <td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">season</td>
                <td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">The season in which the game was played.</td>
                <td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">String</td>
            </tr>
            <tr>
                <td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">week</td>
                <td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Week of the season the game occurred.</td>
                <td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Integer</td>
            </tr>
            <tr>
                <td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">gameDate</td>
                <td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Date the game was played.</td>
                <td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Date</td>
            </tr>
            <tr>
                <td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">gameTimeEastern</td>
                <td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Time of the game in Eastern time zone.</td>
                <td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Time</td>
            </tr>
            <tr>
                <td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">homeTeamAbbr</td>
                <td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Abbreviation for the home team.</td>
                <td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">String</td>
            </tr>
            <tr>
                <td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">visitorTeamAbbr</td>
                <td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Abbreviation for the away team.</td>
                <td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">String</td>
            </tr>
            <tr>
                <td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">homeFinalScore</td>
                <td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Final score for the home team.</td>
                <td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Integer</td>
            </tr>
            <tr>
                <td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">visitorFinalScore</td>
                <td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Final score for the away team.</td>
                <td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Integer</td>
            </tr>
        </tbody>
    </table>
</div>
<div style="background-color: #F9F9F9; border: 2px solid #C0392B; padding: 20px; border-radius: 15px; margin-top: 20px;">
    <h2 style="color: #C0392B; font-size: 24px; text-align: center;">Player Play Data Column Descriptions</h2>
    <table style="border-collapse: collapse; width: 100%; margin-top: 20px;">
        <thead style="background-color: #C0392B; color: #FFFFFF;">
            <tr>
                <th style="border: 1px solid #C0392B; padding: 12px; text-align: left; font-size: 14px;">Variable</th>
                <th style="border: 1px solid #C0392B; padding: 12px; text-align: left; font-size: 14px;">Description</th>
                <th style="border: 1px solid #C0392B; padding: 12px; text-align: left; font-size: 14px;">Data Type</th>
            </tr>
        </thead>
        <tbody>
            <tr><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">gameId</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Unique identifier for the game.</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Integer</td></tr>
            <tr><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">playId</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Unique identifier for the play.</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Integer</td></tr>
            <tr><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">nflId</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Unique identifier for the player.</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Integer</td></tr>
            <tr><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">teamAbbr</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Abbreviation of the team name.</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">String</td></tr>
            <tr><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">hadRushAttempt</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Indicator if a rushing attempt was made.</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Boolean</td></tr>
            <tr><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">rushingYards</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Yards gained during a rushing attempt.</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Integer</td></tr>
            <tr><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">hadDropback</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Indicator if the player had a dropback.</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Boolean</td></tr>
            <tr><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">passingYards</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Yards gained during a passing attempt.</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Integer</td></tr>
            <tr><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">sackYardsAsOffense</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Yards lost due to sacks when on offense.</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Integer</td></tr>
            <tr><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">hadPassReception</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Indicator if the player had a pass reception.</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Boolean</td></tr>
            <tr><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">receivingYards</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Yards gained during a reception.</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Integer</td></tr>
            <tr><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">wasTargettedReceiver</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Indicator if the player was the targeted receiver.</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Boolean</td></tr>
            <tr><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">yardageGainedAfterTheCatch</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Yards gained after the reception.</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Integer</td></tr>
            <tr><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">fumbles</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Number of fumbles made by the player.</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Integer</td></tr>
            <tr><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">fumbleLost</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Number of fumbles that were lost.</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Integer</td></tr>
            <tr><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">fumbleOutOfBounds</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Indicator if the fumble was out of bounds.</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Boolean</td></tr>
            <tr><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">assistedTackle</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Number of tackles assisted by the player.</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Integer</td></tr>
            <tr><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">forcedFumbleAsDefense</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Number of forced fumbles as a defender.</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Integer</td></tr>
            <tr><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">halfSackYardsAsDefense</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Yards lost due to a half sack as a defender.</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Integer</td></tr>
            <tr><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">passDefensed</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Number of passes defensed.</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Integer</td></tr>
            <tr><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">quarterbackHit</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Number of hits on the quarterback.</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Integer</td></tr>
            <tr><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">sackYardsAsDefense</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Yards lost due to sacks while playing defense.</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Integer</td></tr>
            <tr><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">safetyAsDefense</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Number of safeties made by the player on defense.</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Integer</td></tr>
            <tr><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">soloTackle</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Number of solo tackles made.</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Integer</td></tr>
            <tr><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">tackleAssist</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Number of tackle assists made.</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Integer</td></tr>
            <tr><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">tackleForALoss</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Number of tackles made for a loss.</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Integer</td></tr>
            <tr><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">tackleForALossYardage</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Yardage lost due to tackles for a loss.</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Integer</td></tr>
            <tr><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">hadInterception</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Indicator if the player had an interception.</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Boolean</td></tr>
            <tr><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">interceptionYards</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Yards gained after an interception.</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Integer</td></tr>
            <tr><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">fumbleRecoveries</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Number of fumble recoveries made by the player.</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Integer</td></tr>
            <tr><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">fumbleRecoveryYards</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Number of yards gained from a fumble recovery by the player.</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Integer</td></tr>
            <tr><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">penaltyYards</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Number of yards penalized against the player due to penalties.</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Integer</td></tr>
            <tr><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">penaltyNames</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">List of penalty names applied to the player.</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">String (Categorical)</td></tr>
            <tr><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">wasInitialPassRusher</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Indicates if the player was the initial pass rusher on the play.</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Boolean</td></tr>
            <tr><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">causedPressure</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Indicates if the player caused pressure on the quarterback.</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Boolean</td></tr>
            <tr><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">timeToPressureAsPassRusher</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Time taken by the player to pressure the quarterback as a pass rusher.</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Float (Seconds)</td></tr>
            <tr><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">getOffTimeAsPassRusher</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Time taken by the player to get off the line as a pass rusher.</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Float (Seconds)</td></tr>
            <tr><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">inMotionAtBallSnap</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Indicates whether the player was in motion at the moment of the ball snap.</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Boolean</td></tr>
            <tr><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">motionSinceLineset</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Time in motion after the line of scrimmage was set.</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Float (Seconds)</td></tr>
            <tr><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">shiftSinceLineset</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">The amount of time, in seconds, that has passed since the player shifted from their original position before the play began, relative to the line of scrimmage.</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Float</td></tr>
            <tr><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">wasRunningRoute</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Indicates if the player was running a route during the play.</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Boolean</td></tr>
            <tr><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">routeRan</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Type of route run by the player during the play.</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">String (Categorical)</td></tr>
            <tr><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">blockedPlayerNFLId1</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">NFL ID of the first player blocked by the player.</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Integer</td></tr>
            <tr><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">blockedPlayerNFLId2</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">NFL ID of the second player blocked by the player.</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Integer</td></tr>
            <tr><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">blockedPlayerNFLId3</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">NFL ID of the third player blocked by the player.</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Integer</td></tr>
            <tr><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">pressureAllowedAsBlocker</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Indicates if the player allowed pressure as a blocker.</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Boolean</td></tr>
            <tr><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">timeToPressureAllowedAsBlocker</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Time taken for the player to allow pressure as a blocker.</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Float (Seconds)</td></tr>
            <tr><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">pff_defensiveCoverageAssignment</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Yards gained from completed passes involving the player.</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Integer</td></tr>
            <tr><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">pff_primaryDefensiveCoverageMatchupNflId</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Number of pass attempts involving the player.</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Integer</td></tr>
            <tr><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">pff_secondaryDefensiveCoverageMatchupNflId</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Number of pass attempts involving the player.</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Integer</td></tr>
        </tbody>
    </table>
</div>
<div style="background-color: #F9F9F9; border: 2px solid #C0392B; padding: 20px; border-radius: 15px; margin-top: 20px;">
    <h2 style="color: #C0392B; font-size: 24px; text-align: center;">Players Data Column Descriptions</h2>
    <table style="border-collapse: collapse; width: 100%; margin-top: 20px;">
        <thead style="background-color: #C0392B; color: #FFFFFF;">
            <tr>
                <th style="border: 1px solid #C0392B; padding: 12px; text-align: left; font-size: 14px;">Variable</th>
                <th style="border: 1px solid #C0392B; padding: 12px; text-align: left; font-size: 14px;">Description</th>
                <th style="border: 1px solid #C0392B; padding: 12px; text-align: left; font-size: 14px;">Data Type</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">nflId</td>
                <td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Unique identifier for the player.</td>
                <td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Integer</td>
            </tr>
            <tr>
                <td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">height</td>
                <td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Height of the player (e.g., "6'2").</td>
                <td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">String</td>
            </tr>
            <tr>
                <td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">weight</td>
                <td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Weight of the player in pounds.</td>
                <td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Integer</td>
            </tr>
            <tr>
                <td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">birthDate</td>
                <td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Date of birth of the player.</td>
                <td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Date</td>
            </tr>
            <tr>
                <td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">collegeName</td>
                <td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Name of the college the player attended.</td>
                <td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">String</td>
            </tr>
            <tr>
                <td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">position</td>
                <td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Position the player plays (e.g., "QB").</td>
                <td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">String</td>
            </tr>
            <tr>
                <td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">displayName</td>
                <td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Full name of the player displayed (e.g., "Tom Brady").</td>
                <td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">String</td>
            </tr>
        </tbody>
    </table>
</div>

<div style="background-color: #F9F9F9; border: 2px solid #C0392B; padding: 20px; border-radius: 15px; margin-top: 20px;">
    <h2 style="color: #C0392B; font-size: 24px; text-align: center;">Plays Column Descriptions</h2>
    <table style="border-collapse: collapse; width: 100%; margin-top: 20px;">
        <thead style="background-color: #C0392B; color: #FFFFFF;">
            <tr>
                <th style="border: 1px solid #C0392B; padding: 12px; text-align: left; font-size: 14px;">Variable</th>
                <th style="border: 1px solid #C0392B; padding: 12px; text-align: left; font-size: 14px;">Description</th>
                <th style="border: 1px solid #C0392B; padding: 12px; text-align: left; font-size: 14px;">Data Type</th>
            </tr>
        </thead>
        <tbody>
            <div style="overflow-x:auto;">
    <table style="width:100%; border-collapse: collapse;">
        <thead>
            <tr>
                <th style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Column Name</th>
                <th style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Description</th>
                <th style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Data Type</th>
            </tr>
        </thead>
        <tbody>
            <tr><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">gameId</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Unique identifier for the game.</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Integer</td></tr>
            <tr><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">playId</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Unique identifier for the play within a game.</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Integer</td></tr>
            <tr><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">playDescription</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Text description of the play.</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">String</td></tr>
            <tr><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">quarter</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Quarter of the game in which the play occurred.</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Integer</td></tr>
            <tr><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">down</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">The current down (1st, 2nd, 3rd, etc.) in the play.</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Integer</td></tr>
            <tr><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">yardsToGo</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">The number of yards needed to gain a first down.</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Integer</td></tr>
            <tr><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">possessionTeam</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">The team in possession of the ball for the play.</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">String</td></tr>
            <tr><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">defensiveTeam</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">The defensive team for the play.</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">String</td></tr>
            <tr><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">yardlineSide</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Side of the field where the play began (left or right). </td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">String (Categorical)</td></tr>
            <tr><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">yardlineNumber</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Yardline number where the play started (1 to 50).</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Integer</td></tr>
            <tr><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">gameClock</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Time remaining on the game clock at the start of the play.</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Time (HH:MM:SS)</td></tr>
            <tr><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">preSnapHomeScore</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Home team score before the play.</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Integer</td></tr>
            <tr><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">preSnapVisitorScore</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Visitor team score before the play.</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Integer</td></tr>
            <tr><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">playNullifiedByPenalty</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Indicates if the play was nullified due to a penalty.</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Boolean</td></tr>
            <tr><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">absoluteYardlineNumber</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Absolute yard line number (1 to 100).</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Integer</td></tr>
            <tr><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">preSnapHomeTeamWinProbability</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Home team's win probability before the play.</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Float (0-1)</td></tr>
            <tr><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">preSnapVisitorTeamWinProbability</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Visitor team's win probability before the play.</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Float (0-1)</td></tr>
            <tr><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">expectedPoints</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">The expected points value for the play.</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Float</td></tr>
            <tr><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">offenseFormation</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Offensive formation used during the play.</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">String</td></tr>
            <tr><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">receiverAlignment</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Receiver's alignment during the play.</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">String</td></tr>
            <tr><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">playClockAtSnap</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">The play clock at the time of the snap.</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Integer (Seconds)</td></tr>
            <tr><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">passResult</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">The result of the pass (e.g., completion, incompletion, interception).</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">String</td></tr>
            <tr><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">passLength</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Length of the pass in yards.</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Float</td></tr>
            <tr><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">targetX</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">The X-coordinate of the target location on the field.</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Float</td></tr>
            <tr><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">targetY</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">The Y-coordinate of the target location on the field.</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Float</td></tr>
            <tr><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">playAction</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Type of action performed by the quarterback (e.g., play-action pass).</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">String</td></tr>
            <tr><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">dropbackType</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Type of dropback executed by the quarterback (e.g., 3-step, 5-step).</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">String</td></tr>
            <tr><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">dropbackDistance</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Distance (in yards) the quarterback drops back.</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Float</td></tr>
            <tr><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">passLocationType</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Type of location where the pass was thrown (e.g., deep, short, intermediate).</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">String</td></tr>
            <tr><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">timeToThrow</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Time (in seconds) taken by the quarterback to throw the pass.</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Float</td></tr>
            <tr><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">timeInTackleBox</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Time (in seconds) spent by the quarterback within the tackle box.</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Float</td></tr>
            <tr><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">timeToSack</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Time (in seconds) until the quarterback was sacked.</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Float</td></tr>
            <tr><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">passTippedAtLine</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Indicates whether the pass was tipped at the line of scrimmage.</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Boolean</td></tr>
            <tr><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">unblockedPressure</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Indicates if the quarterback faced unblocked pressure.</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Boolean</td></tr>
            <tr><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">qbSpike</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Indicates if the quarterback spiked the ball.</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Boolean</td></tr>
            <tr><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">qbKneel</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Indicates if the quarterback performed a kneel.</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Boolean</td></tr>
            <tr><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">qbSneak</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Indicates if the quarterback performed a sneak.</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Boolean</td></tr>
            <tr><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">rushLocationType</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">The location type of the rush play.</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">String</td></tr>
            <tr><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">penaltyYards</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Total penalty yards during the play.</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Integer</td></tr>
            <tr><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">prePenaltyYardsGained</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Yards gained before the penalty was applied.</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Float</td></tr>
            <tr><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">yardsGained</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Yards gained during the play.</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Float</td></tr>
            <tr><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">homeTeamWinProbabilityAdded</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Additional win probability for the home team after the play.</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Float</td></tr>
            <tr><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">visitorTeamWinProbilityAdded</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Additional win probability for the visitor team after the play.</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Float</td></tr>
            <tr><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">expectedPointsAdded</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Additional expected points from the play.</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Float</td></tr>
            <tr><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">isDropback</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Indicates if the play is a dropback pass.</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Boolean</td></tr>
            <tr><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">pff_runConceptPrimary</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Primary running concept (e.g., inside zone, outside zone).</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">String</td></tr>
            <tr><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">pff_runConceptSecondary</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Secondary running concept.</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">String</td></tr>
            <tr><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">pff_runPassOption</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Indicates if the play has a run-pass option (RPO).</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Boolean</td></tr>
            <tr><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">pff_passCoverage</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Pass coverage type used against the quarterback.</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">String</td></tr>
            <tr><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">pff_manZone</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">Indicates if the defense is playing man or zone coverage.</td><td style="border: 1px solid #C0392B; padding: 12px; font-size: 12px;">String</td></tr>
        </tbody>
    </table>
</div>
<div style="background-color: #F9F9F9; border: 2px solid #C0392B; padding: 20px; border-radius: 15px; margin-top: 20px;">
    <h2 style="color: #C0392B; font-size: 24px; text-align: center;">Tracking Data Column Descriptions</h2>
    <table style="border-collapse: collapse; width: 100%; margin-top: 20px;">
        <thead style="background-color: #C0392B; color: #FFFFFF;">
            <tr>
                <th style="border: 1px solid #C0392B; padding: 10px; font-size: 14px;">Column Name</th>
                <th style="border: 1px solid #C0392B; padding: 10px; font-size: 14px;">Description</th>
                <th style="border: 1px solid #C0392B; padding: 10px; font-size: 14px;">Data Type</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td style="border: 1px solid #C0392B; padding: 10px;">gameId</td>
                <td style="border: 1px solid #C0392B; padding: 10px;">Unique identifier for the game.</td>
                <td style="border: 1px solid #C0392B; padding: 10px;">Integer</td>
            </tr>
            <tr>
                <td style="border: 1px solid #C0392B; padding: 10px;">playId</td>
                <td style="border: 1px solid #C0392B; padding: 10px;">Unique identifier for the play within a game.</td>
                <td style="border: 1px solid #C0392B; padding: 10px;">Integer</td>
            </tr>
            <tr>
                <td style="border: 1px solid #C0392B; padding: 10px;">nflId</td>
                <td style="border: 1px solid #C0392B; padding: 10px;">Unique identifier for the NFL player.</td>
                <td style="border: 1px solid #C0392B; padding: 10px;">Integer</td>
            </tr>
            <tr>
                <td style="border: 1px solid #C0392B; padding: 10px;">displayName</td>
                <td style="border: 1px solid #C0392B; padding: 10px;">The display name of the player or object.</td>
                <td style="border: 1px solid #C0392B; padding: 10px;">String</td>
            </tr>
            <tr>
                <td style="border: 1px solid #C0392B; padding: 10px;">frameId</td>
                <td style="border: 1px solid #C0392B; padding: 10px;">The unique identifier for the frame.</td>
                <td style="border: 1px solid #C0392B; padding: 10px;">Integer</td>
            </tr>
            <tr>
                <td style="border: 1px solid #C0392B; padding: 10px;">frameType</td>
                <td style="border: 1px solid #C0392B; padding: 10px;">Type of the frame (e.g., start, end, intermediate).</td>
                <td style="border: 1px solid #C0392B; padding: 10px;">String</td>
            </tr>
            <tr>
                <td style="border: 1px solid #C0392B; padding: 10px;">time</td>
                <td style="border: 1px solid #C0392B; padding: 10px;">Timestamp for when the tracking data was recorded.</td>
                <td style="border: 1px solid #C0392B; padding: 10px;">Time (HH:MM:SS)</td>
            </tr>
            <tr>
                <td style="border: 1px solid #C0392B; padding: 10px;">jerseyNumber</td>
                <td style="border: 1px solid #C0392B; padding: 10px;">The player's jersey number.</td>
                <td style="border: 1px solid #C0392B; padding: 10px;">Integer</td>
            </tr>
            <tr>
                <td style="border: 1px solid #C0392B; padding: 10px;">club</td>
                <td style="border: 1px solid #C0392B; padding: 10px;">The player's club or team.</td>
                <td style="border: 1px solid #C0392B; padding: 10px;">String</td>
            </tr>
            <tr>
                <td style="border: 1px solid #C0392B; padding: 10px;">playDirection</td>
                <td style="border: 1px solid #C0392B; padding: 10px;">The direction in which the play is moving (e.g., left or right).</td>
                <td style="border: 1px solid #C0392B; padding: 10px;">String</td>
            </tr>
        </tbody>
    </table>
</div>

# <p id="import_lib" style="background-color: #FFDDC1; font-family: 'Palatino Linotype', 'Book Antiqua', Palatino, serif; color: #8a0808; font-size: 1.5em; text-align: center; border-radius: 35px 20px 35px 20px; padding: 15px 25px; line-height: 1.5; border: 4px double #8a0808; box-shadow: 5px 5px 20px rgba(0, 0, 0, 0.5), -5px -5px 20px rgba(0, 0, 0, 0.3); margin: 0 auto; width: auto;">About Author</p>

* Hi Kagglers! I'm Maria Nadeem, a passionate Data Scientist with keen interest in exploring and applying diverse data science techniques.
* As dedicated to derive meaningful insights and making impactful decisions through data, I actively engage in projects and contribute to Kaggle by sharing detailed analysis and actionable insights.
* I'm excited to share my latest project on Binary Prediction of Poisonous Mushrooms.
* In this notebook, I begin by conatenating train data of competition with secondary-mushroom-dataset-data-set, providing detailed overview of the dataset and then dive into creating various visualization plots using subplots to gain deep insights. Following this, I use the KFold Cross Validation, perform model training, and choose the best model to take improve predictions.
* Then I took the Best Model and then apply Hyperparameter Tuning on it and took more better results.

| Name               | Email                                               | LinkedIn                                                  | GitHub                                           | Kaggle                                        |
|--------------------|-----------------------------------------------------|-----------------------------------------------------------|--------------------------------------------------|-----------------------------------------------|
| **Maria Nadeem**  | marianadeem755@gmail.com | <a href="https://www.linkedin.com/in/maria-nadeem-4994122aa/" style="text-decoration: none; font-size: 16px;"><img src="https://img.shields.io/badge/LinkedIn-%2300A4CC.svg?style=for-the-badge&logo=LinkedIn&logoColor=white" alt="LinkedIn Badge"></a> | <a href="https://github.com/marianadeem755" style="text-decoration: none; font-size: 16px;"><img src="https://img.shields.io/badge/GitHub-%23FF6F61.svg?style=for-the-badge&logo=GitHub&logoColor=white" alt="GitHub Badge"></a> | <a href="https://www.kaggle.com/marianadeem755" style="text-decoration: none; font-size: 16px;"><img src="https://img.shields.io/badge/Kaggle-%238a2be2.svg?style=for-the-badge&logo=Kaggle&logoColor=white" alt="Kaggle Badge"></a> |

# <p id="import_lib" style="background-color: #FFDDC1; font-family: 'Palatino Linotype', 'Book Antiqua', Palatino, serif; color: #8a0808; font-size: 1.5em; text-align: center; border-radius: 35px 20px 35px 20px; padding: 15px 25px; line-height: 1.5; border: 4px double #8a0808; box-shadow: 5px 5px 20px rgba(0, 0, 0, 0.5), -5px -5px 20px rgba(0, 0, 0, 0.3); margin: 0 auto; width: auto;">Import Libraries</p>

```python
# import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.subplots as sp
import altair as alt
import random
from IPython.display import display, HTML
from matplotlib.colors import LinearSegmentedColormap
# ignore warnings
import warnings
warnings.filterwarnings("ignore")
```

# <p id="import_lib" style="background-color: #FFDDC1; font-family: 'Palatino Linotype', 'Book Antiqua', Palatino, serif; color: #8a0808; font-size: 1.5em; text-align: center; border-radius: 35px 20px 35px 20px; padding: 15px 25px; line-height: 1.5; border: 4px double #8a0808; box-shadow: 5px 5px 20px rgba(0, 0, 0, 0.5), -5px -5px 20px rgba(0, 0, 0, 0.3); margin: 0 auto; width: auto;">Games Data Overview</p>

```python
import pandas as pd
import seaborn as sns
from IPython.core.display import display, HTML
import random

# Function to style tables with a unique red/pink color scheme and inner box styling
def style_table(df):
    styled_df = df.style.set_table_styles([
        {"selector": "th", "props": [("color", "white"), ("background-color", "#b2182b")]}  # Dark Red header
    ]).set_properties(**{"text-align": "center", "background-color": "#FFDDC1", 'border': '2px solid #d2182b'}).hide(axis="index")
    return styled_df.to_html()

# Function to generate random shades of red/pink
def generate_random_color():
    color = "#{:02x}{:02x}{:02x}".format(
        random.randint(100, 200),
        random.randint(50, 150),
        random.randint(50, 100)
    )
    return color

# Function to create styled heading with emojis and different colors for main and sub-headings
def styled_heading(text, background_color, text_color='white', border_color=None, font_size='30px', border_style='solid'):
    border_color = border_color if border_color else background_color
    return f"""
    <div style="
        text-align: center;
        background: {background_color};
        color: {text_color};
        padding: 20px;
        font-size: {font_size};
        font-weight: bold;
        line-height: 1;
        border-radius: 20px 20px 0 0;
        margin-bottom: 25px;
        box-shadow: 0px 8px 12px rgba(0, 0, 0, 0.3);
        border: 4px {border_style} {border_color};
        font-family: 'Arial', sans-serif;
    ">
        {text}
    </div>
    """

# Define your red/pinkish color palette
palette = ['#b2182b', '#e76f51', '#f1a7a1', '#f4c7c3', '#d63447']
color_palette = sns.color_palette(palette)

# Define colors for headings and sub-headings
main_heading_color = '#b2182b'  # Main heading color (Dark Red)
sub_heading_color = '#e76f51'    # Sub-heading color (Pinkish)
headings_border_color = '#f1a7a1' # Border color for headings (Light Red)

def print_dataset_analysis(dataset, dataset_name, n_top=5, palette_index=0):
    heading_color = color_palette[palette_index]
    
    # Main heading with emoji
    heading = styled_heading(f"📊 {dataset_name} Overview", main_heading_color, 'white', border_color=headings_border_color, font_size='35px', border_style='solid')
    display(HTML(heading))
    
    # Sub-headings with emojis
    display(HTML(f"<h2 style='font-size: 24px; color: {sub_heading_color};'>🔍 Shape of the Dataset</h2>"))
    display(HTML(f"<p>{dataset.shape[0]} rows and {dataset.shape[1]} columns</p>"))
    
    display(HTML(f"<h2 style='font-size: 24px; color: {sub_heading_color};'>👀 First 5 Rows</h2>"))
    display(HTML(style_table(dataset.head(n_top))))
    
    display(HTML(f"<h2 style='font-size: 24px; color: {sub_heading_color};'>📈 Summary Statistics</h2>"))
    display(HTML(style_table(dataset.describe())))
    
    display(HTML(f"<h2 style='font-size: 24px; color: {sub_heading_color};'>🚨 Null Values</h2>"))
    null_counts = dataset.isnull().sum()
    if null_counts.sum() == 0:
        display(HTML("<p>No null values found.</p>"))
    else:
        display(HTML(style_table(null_counts[null_counts > 0].to_frame(name='Null Values'))))
    
    display(HTML(f"<h2 style='font-size: 24px; color: {sub_heading_color};'>🔍 Duplicate Rows</h2>"))
    duplicate_count = dataset.duplicated().sum()
    display(HTML(f"<p>{duplicate_count} duplicate rows found.</p>"))
    
    display(HTML(f"<h2 style='font-size: 24px; color: {sub_heading_color};'>📝 Data Types</h2>"))
    dtypes_table = pd.DataFrame({
        'Data Type': [dataset[col].dtype for col in dataset.columns],
        'Column Name': dataset.columns
    })
    display(HTML(style_table(dtypes_table)))

    display(HTML(f"<h2 style='font-size: 24px; color: {sub_heading_color};'>📋 Column Names</h2>"))
    display(HTML(f"<p>{', '.join(dataset.columns)}</p>"))

    display(HTML(f"<h2 style='font-size: 24px; color: {sub_heading_color};'>🔢 Unique Values</h2>"))
    unique_values_table = pd.DataFrame({
        'Data Type': [dataset[col].dtype for col in dataset.columns],
        'Column Name': dataset.columns,
        'Unique Values': [', '.join(map(str, dataset[col].unique()[:7])) for col in dataset.columns]
    })
    display(HTML(style_table(unique_values_table)))

# Example usage with your dataset (`df_train`, `df_test`, `df_sub`)
# Load the Games dataset
games_df = pd.read_csv("/kaggle/input/nfl-big-data-bowl-2025/games.csv")

# Use different palette colors for different datasets
print_dataset_analysis(games_df, "Games Data", palette_index=0)
```

<div style="background-color: #FFDDC1; padding: 20px; border-radius: 15px; border: 2px solid #8B0000; margin-top: 20px;">
    <p style="text-align: justify; font-family: 'Times New Roman', serif; font-size: 20px; color: #8B0000; font-weight: bold;">
        Observations About Dataset:
    </p>
    <ul style="font-family: 'Georgia', serif; font-size: 14px; color: black; margin-left: 20px;">
        <li>The NFL Big Data Bowl 2025 consits of 13 csv Files</li>
        <li>The NFL Big Data Bowl 2025 Files includes <strong>games.csv, player_play.csv, plays.csv, tracking_week_1.csv, tracking_week_2.csv, tracking_week_3.csv, tracking_week_4.csv, tracking_week_5.csv, tracking_week_6.csv, tracking_week_7.csv, tracking_week_8.csv, tracking_week_9.csv
 <strong></strong>.</li>
        <li>In this Notebook I do the Detailed and Comprehensive Analysis of Games.Csv File <strong>This File Consists of </strong> 136 rows and 9 columns <strong></strong></li>
        <li>There is no Null Value and No Duplicate present in it<strong></strong></li>
        <li>Games. Csv consists of Columns such as<strong> gameId, season, week, gameDate, gameTimeEastern, homeTeamAbbr, visitorTeamAbbr, homeFinalScore, visitorFinalScore</strong></li>
    </ul>
</div>

# <p id="import_lib" style="background-color: #FFDDC1; font-family: 'Palatino Linotype', 'Book Antiqua', Palatino, serif; color: #8a0808; font-size: 1.5em; text-align: center; border-radius: 35px 20px 35px 20px; padding: 15px 25px; line-height: 1.5; border: 4px double #8a0808; box-shadow: 5px 5px 20px rgba(0, 0, 0, 0.5), -5px -5px 20px rgba(0, 0, 0, 0.3); margin: 0 auto; width: auto;">Home & Visitor Team Abbrevations</p>

<div style="background-color: #F9F9F9; border: 2px solid #C0392B; padding: 20px; border-radius: 15px; margin-top: 20px;">
    <h2 style="color: #C0392B; font-size: 24px; text-align: center;">Home Team Names and Abbreviations</h2>
    <table style="border-collapse: collapse; width: 100%; margin-top: 20px;">
        <thead style="background-color: #C0392B; color: #FFFFFF;">
            <tr>
                <th style="border: 1px solid #C0392B; padding: 12px; text-align: left; font-size: 18px;">Team Name</th>
                <th style="border: 1px solid #C0392B; padding: 12px; text-align: left; font-size: 18px;">Abbreviation</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td style="border: 1px solid #C0392B; padding: 12px; font-size: 16px;">New Orleans Saints</td>
                <td style="border: 1px solid #C0392B; padding: 12px; font-size: 16px;">NO</td>
            </tr>
            <tr>
                <td style="border: 1px solid #C0392B; padding: 12px; font-size: 16px;">Los Angeles Rams</td>
                <td style="border: 1px solid #C0392B; padding: 12px; font-size: 16px;">LA</td>
            </tr>
            <tr>
                <td style="border: 1px solid #C0392B; padding: 12px; font-size: 16px;">Washington Commanders</td>
                <td style="border: 1px solid #C0392B; padding: 12px; font-size: 16px;">WAS</td>
            </tr>
            <tr>
                <td style="border: 1px solid #C0392B; padding: 12px; font-size: 16px;">Jacksonville Jaguars</td>
                <td style="border: 1px solid #C0392B; padding: 12px; font-size: 16px;">JAX</td>
            </tr>
            <tr>
                <td style="border: 1px solid #C0392B; padding: 12px; font-size: 16px;">Cleveland Browns</td>
                <td style="border: 1px solid #C0392B; padding: 12px; font-size: 16px;">CLE</td>
            </tr>
            <tr>
                <td style="border: 1px solid #C0392B; padding: 12px; font-size: 16px;">Atlanta Falcons</td>
                <td style="border: 1px solid #C0392B; padding: 12px; font-size: 16px;">ATL</td>
            </tr>
            <tr>
                <td style="border: 1px solid #C0392B; padding: 12px; font-size: 16px;">Dallas Cowboys</td>
                <td style="border: 1px solid #C0392B; padding: 12px; font-size: 16px;">DAL</td>
            </tr>
            <tr>
                <td style="border: 1px solid #C0392B; padding: 12px; font-size: 16px;">Arizona Cardinals</td>
                <td style="border: 1px solid #C0392B; padding: 12px; font-size: 16px;">ARI</td>
            </tr>
            <tr>
                <td style="border: 1px solid #C0392B; padding: 12px; font-size: 16px;">Tampa Bay Buccaneers</td>
                <td style="border: 1px solid #C0392B; padding: 12px; font-size: 16px;">TB</td>
            </tr>
            <tr>
                <td style="border: 1px solid #C0392B; padding: 12px; font-size: 16px;">New York Jets</td>
                <td style="border: 1px solid #C0392B; padding: 12px; font-size: 16px;">NYJ</td>
            </tr>
            <tr>
                <td style="border: 1px solid #C0392B; padding: 12px; font-size: 16px;">Detroit Lions</td>
                <td style="border: 1px solid #C0392B; padding: 12px; font-size: 16px;">DET</td>
            </tr>
            <tr>
                <td style="border: 1px solid #C0392B; padding: 12px; font-size: 16px;">Carolina Panthers</td>
                <td style="border: 1px solid #C0392B; padding: 12px; font-size: 16px;">CAR</td>
            </tr>
            <tr>
                <td style="border: 1px solid #C0392B; padding: 12px; font-size: 16px;">Minnesota Vikings</td>
                <td style="border: 1px solid #C0392B; padding: 12px; font-size: 16px;">MIN</td>
            </tr>
            <tr>
                <td style="border: 1px solid #C0392B; padding: 12px; font-size: 16px;">Miami Dolphins</td>
                <td style="border: 1px solid #C0392B; padding: 12px; font-size: 16px;">MIA</td>
            </tr>
            <tr>
                <td style="border: 1px solid #C0392B; padding: 12px; font-size: 16px;">New England Patriots</td>
                <td style="border: 1px solid #C0392B; padding: 12px; font-size: 16px;">NE</td>
            </tr>
            <tr>
                <td style="border: 1px solid #C0392B; padding: 12px; font-size: 16px;">Indianapolis Colts</td>
                <td style="border: 1px solid #C0392B; padding: 12px; font-size: 16px;">IND</td>
            </tr>
            <tr>
                <td style="border: 1px solid #C0392B; padding: 12px; font-size: 16px;">Philadelphia Eagles</td>
                <td style="border: 1px solid #C0392B; padding: 12px; font-size: 16px;">PHI</td>
            </tr>
            <tr>
                <td style="border: 1px solid #C0392B; padding: 12px; font-size: 16px;">Green Bay Packers</td>
                <td style="border: 1px solid #C0392B; padding: 12px; font-size: 16px;">GB</td>
            </tr>
            <tr>
                <td style="border: 1px solid #C0392B; padding: 12px; font-size: 16px;">Denver Broncos</td>
                <td style="border: 1px solid #C0392B; padding: 12px; font-size: 16px;">DEN</td>
            </tr>
            <tr>
                <td style="border: 1px solid #C0392B; padding: 12px; font-size: 16px;">New York Giants</td>
                <td style="border: 1px solid #C0392B; padding: 12px; font-size: 16px;">NYG</td>
            </tr>
            <tr>
                <td style="border: 1px solid #C0392B; padding: 12px; font-size: 16px;">Kansas City Chiefs</td>
                <td style="border: 1px solid #C0392B; padding: 12px; font-size: 16px;">KC</td>
            </tr>
            <tr>
                <td style="border: 1px solid #C0392B; padding: 12px; font-size: 16px;">Chicago Bears</td>
                <td style="border: 1px solid #C0392B; padding: 12px; font-size: 16px;">CHI</td>
            </tr>
            <tr>
                <td style="border: 1px solid #C0392B; padding: 12px; font-size: 16px;">Cincinnati Bengals</td>
                <td style="border: 1px solid #C0392B; padding: 12px; font-size: 16px;">CIN</td>
                            </tr>
            <tr>
                <td style="border: 1px solid #C0392B; padding: 12px; font-size: 16px;">Baltimore Ravens</td>
                <td style="border: 1px solid #C0392B; padding: 12px; font-size: 16px;">BAL</td>
            </tr>
            <tr>
                <td style="border: 1px solid #C0392B; padding: 12px; font-size: 16px;">Los Angeles Chargers</td>
                <td style="border: 1px solid #C0392B; padding: 12px; font-size: 16px;">LAC</td>
            </tr>
            <tr>
                <td style="border: 1px solid #C0392B; padding: 12px; font-size: 16px;">Seattle Seahawks</td>
                <td style="border: 1px solid #C0392B; padding: 12px; font-size: 16px;">SEA</td>
            </tr>
            <tr>
                <td style="border: 1px solid #C0392B; padding: 12px; font-size: 16px;">Houston Texans</td>
                <td style="border: 1px solid #C0392B; padding: 12px; font-size: 16px;">HOU</td>
            </tr>
            <tr>
                <td style="border: 1px solid #C0392B; padding: 12px; font-size: 16px;">Tennessee Titans</td>
                <td style="border: 1px solid #C0392B; padding: 12px; font-size: 16px;">TEN</td>
            </tr>
            <tr>
                <td style="border: 1px solid #C0392B; padding: 12px; font-size: 16px;">San Francisco 49ers</td>
                <td style="border: 1px solid #C0392B; padding: 12px; font-size: 16px;">SF</td>
            </tr>
            <tr>
                <td style="border: 1px solid #C0392B; padding: 12px; font-size: 16px;">Las Vegas Raiders</td>
                <td style="border: 1px solid #C0392B; padding: 12px; font-size: 16px;">LV</td>
            </tr>
            <tr>
                <td style="border: 1px solid #C0392B; padding: 12px; font-size: 16px;">Buffalo Bills</td>
                <td style="border: 1px solid #C0392B; padding: 12px; font-size: 16px;">BUF</td>
            </tr>
            <tr>
                <td style="border: 1px solid #C0392B; padding: 12px; font-size: 16px;">Pittsburgh Steelers</td>
                <td style="border: 1px solid #C0392B; padding: 12px; font-size: 16px;">PIT</td>
            </tr>
        </tbody>
    </table>
</div>

<div style="background-color: #F9F9F9; border: 2px solid #C0392B; padding: 20px; border-radius: 15px; margin-top: 20px;">
    <h2 style="color: #C0392B; font-size: 24px; text-align: center;">Visitor Team Names and Abbreviations</h2>
    <table style="border-collapse: collapse; width: 100%; margin-top: 20px;">
        <thead style="background-color: #C0392B; color: #FFFFFF;">
            <tr>
                <th style="border: 1px solid #C0392B; padding: 12px; text-align: left; font-size: 18px;">Team Name</th>
                <th style="border: 1px solid #C0392B; padding: 12px; text-align: left; font-size: 18px;">Abbreviation</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td style="border: 1px solid #C0392B; padding: 12px; font-size: 16px;">Buffalo Bills</td>
                <td style="border: 1px solid #C0392B; padding: 12px; font-size: 16px;">BUF</td>
            </tr>
            <tr>
                <td style="border: 1px solid #C0392B; padding: 12px; font-size: 16px;">Baltimore Ravens</td>
                <td style="border: 1px solid #C0392B; padding: 12px; font-size: 16px;">BAL</td>
            </tr>
            <tr>
                <td style="border: 1px solid #C0392B; padding: 12px; font-size: 16px;">Miami Dolphins</td>
                <td style="border: 1px solid #C0392B; padding: 12px; font-size: 16px;">MIA</td>
            </tr>
            <tr>
                <td style="border: 1px solid #C0392B; padding: 12px; font-size: 16px;">Seattle Seahawks</td>
                <td style="border: 1px solid #C0392B; padding: 12px; font-size: 16px;">SEA</td>
            </tr>
            <tr>
                <td style="border: 1px solid #C0392B; padding: 12px; font-size: 16px;">Cincinnati Bengals</td>
                <td style="border: 1px solid #C0392B; padding: 12px; font-size: 16px;">CIN</td>
            </tr>
            <tr>
                <td style="border: 1px solid #C0392B; padding: 12px; font-size: 16px;">Green Bay Packers</td>
                <td style="border: 1px solid #C0392B; padding: 12px; font-size: 16px;">GB</td>
            </tr>
            <tr>
                <td style="border: 1px solid #C0392B; padding: 12px; font-size: 16px;">Las Vegas Raiders</td>
                <td style="border: 1px solid #C0392B; padding: 12px; font-size: 16px;">LV</td>
            </tr>
            <tr>
                <td style="border: 1px solid #C0392B; padding: 12px; font-size: 16px;">Chicago Bears</td>
                <td style="border: 1px solid #C0392B; padding: 12px; font-size: 16px;">CHI</td>
            </tr>
            <tr>
                <td style="border: 1px solid #C0392B; padding: 12px; font-size: 16px;">New England Patriots</td>
                <td style="border: 1px solid #C0392B; padding: 12px; font-size: 16px;">NE</td>
            </tr>
            <tr>
                <td style="border: 1px solid #C0392B; padding: 12px; font-size: 16px;">Indianapolis Colts</td>
                <td style="border: 1px solid #C0392B; padding: 12px; font-size: 16px;">IND</td>
            </tr>
            <tr>
                <td style="border: 1px solid #C0392B; padding: 12px; font-size: 16px;">Tennessee Titans</td>
                <td style="border: 1px solid #C0392B; padding: 12px; font-size: 16px;">TEN</td>
            </tr>
            <tr>
                <td style="border: 1px solid #C0392B; padding: 12px; font-size: 16px;">Pittsburgh Steelers</td>
                <td style="border: 1px solid #C0392B; padding: 12px; font-size: 16px;">PIT</td>
            </tr>
            <tr>
                <td style="border: 1px solid #C0392B; padding: 12px; font-size: 16px;">San Francisco 49ers</td>
                <td style="border: 1px solid #C0392B; padding: 12px; font-size: 16px;">SF</td>
            </tr>
            <tr>
                <td style="border: 1px solid #C0392B; padding: 12px; font-size: 16px;">Washington Commanders</td>
                <td style="border: 1px solid #C0392B; padding: 12px; font-size: 16px;">WAS</td>
            </tr>
            <tr>
                <td style="border: 1px solid #C0392B; padding: 12px; font-size: 16px;">Arizona Cardinals</td>
                <td style="border: 1px solid #C0392B; padding: 12px; font-size: 16px;">ARI</td>
            </tr>
            <tr>
                <td style="border: 1px solid #C0392B; padding: 12px; font-size: 16px;">Houston Texans</td>
                <td style="border: 1px solid #C0392B; padding: 12px; font-size: 16px;">HOU</td>
            </tr>
            <tr>
                <td style="border: 1px solid #C0392B; padding: 12px; font-size: 16px;">Atlanta Falcons</td>
                <td style="border: 1px solid #C0392B; padding: 12px; font-size: 16px;">ATL</td>
            </tr>
            <tr>
                <td style="border: 1px solid #C0392B; padding: 12px; font-size: 16px;">Minnesota Vikings</td>
                <td style="border: 1px solid #C0392B; padding: 12px; font-size: 16px;">MIN</td>
            </tr>
            <tr>
                <td style="border: 1px solid #C0392B; padding: 12px; font-size: 16px;">Carolina Panthers</td>
                <td style="border: 1px solid #C0392B; padding: 12px; font-size: 16px;">CAR</td>
            </tr>
            <tr>
                <td style="border: 1px solid #C0392B; padding: 12px; font-size: 16px;">Los Angeles Chargers</td>
                <td style="border: 1px solid #C0392B; padding: 12px; font-size: 16px;">LAC</td>
            </tr>
            <tr>
                <td style="border: 1px solid #C0392B; padding: 12px; font-size: 16px;">New York Jets</td>
                <td style="border: 1px solid #C0392B; padding: 12px; font-size: 16px;">NYJ</td>
            </tr>
            <tr>
                <td style="border: 1px solid #C0392B; padding: 12px; font-size: 16px;">Denver Broncos</td>
                <td style="border: 1px solid #C0392B; padding: 12px; font-size: 16px;">DEN</td>
            </tr>
            <tr>
                <td style="border: 1px solid #C0392B; padding: 12px; font-size: 16px;">Tampa Bay Buccaneers</td>
                <td style="border: 1px solid #C0392B; padding: 12px; font-size: 16px;">TB</td>
            </tr>
            <tr>
                <td style="border: 1px solid #C0392B; padding: 12px; font-size: 16px;">Kansas City Chiefs</td>
                <td style="border: 1px solid #C0392B; padding: 12px; font-size: 16px;">KC</td>
            </tr>
            <tr>
                <td style="border: 1px solid #C0392B; padding: 12px; font-size: 16px;">Jacksonville Jaguars</td>
                <td style="border: 1px solid #C0392B; padding: 12px; font-size: 16px;">JAX</td>
            </tr>
            <tr>
                <td style="border: 1px solid #C0392B; padding: 12px; font-size: 16px;">Philadelphia Eagles</td>
                <td style="border: 1px solid #C0392B; padding: 12px; font-size: 16px;">PHI</td>
            </tr>
            <tr>
                <td style="border: 1px solid #C0392B; padding: 12px; font-size: 16px;">New Orleans Saints</td>
                <td style="border: 1px solid #C0392B; padding: 12px; font-size: 16px;">NO</td>
            </tr>
            <tr>
                <td style="border: 1px solid #C0392B; padding: 12px; font-size: 16px;">Cleveland Browns</td>
                <td style="border: 1px solid #C0392B; padding: 12px; font-size: 16px;">CLE</td>
            </tr>
            <tr>
                <td style="border: 1px solid #C0392B; padding: 12px; font-size: 16px;">Detroit Lions</td>
                <td style="border: 1px solid #C0392B; padding: 12px; font-size: 16px;">DET</td>
            </tr>
            <tr>
                <td style="border: 1px solid #C0392B; padding: 12px; font-size: 16px;">Los Angeles Rams</td>
                <td style="border: 1px solid #C0392B; padding: 12px; font-size: 16px;">LA</td>
            </tr>
            <tr>
                <td style="border: 1px solid #C0392B; padding: 12px; font-size: 16px;">Dallas Cowboys</td>
                <td style="border: 1px solid #C0392B; padding: 12px; font-size: 16px;">DAL</td>
            </tr>
        </tbody>
    </table>
</div>

# <p id="import_lib" style="background-color: #FFDDC1; font-family: 'Palatino Linotype', 'Book Antiqua', Palatino, serif; color: #8a0808; font-size: 1.5em; text-align: center; border-radius: 35px 20px 35px 20px; padding: 15px 25px; line-height: 1.5; border: 4px double #8a0808; box-shadow: 5px 5px 20px rgba(0, 0, 0, 0.5), -5px -5px 20px rgba(0, 0, 0, 0.3); margin: 0 auto; width: auto;">Data Structure of Games.csv File</p>

```python
# Data types and non-null counts
df=pd.read_csv("/kaggle/input/nfl-big-data-bowl-2025/games.csv")
print("Data types and non-null counts:\n", df.info())
```

# <p id="game_id_column" style="background-color: #FFDDC1; font-family: 'Palatino Linotype', 'Book Antiqua', Palatino, serif; color: #8a0808; font-size: 1.5em; text-align: center; border-radius: 35px 20px 35px 20px; padding: 15px 25px; line-height: 1.5; border: 4px double #8a0808; box-shadow: 5px 5px 20px rgba(0, 0, 0, 0.5), -5px -5px 20px rgba(0, 0, 0, 0.3); margin: 0 auto; width: auto;">Let's Explore the Game Id Column</p>

```python
# Verify unique gameId values to ensure no duplicates
print("Unique game IDs:", df['gameId'].nunique())
```

```python
min_gameid=df["gameId"].min()
print("The Minimum Game id of the players according to this Dataset is:", min_gameid)
max_gameid=df["gameId"].max()
print("The Maximum Game id of the players according to this Dataset is:", max_gameid)
```

<div style="background-color: #FFDDC1; padding: 20px; border-radius: 15px; border: 2px solid #8B0000; margin-top: 20px;">
    <p style="text-align: justify; font-family: 'Times New Roman', serif; font-size: 20px; color: #8B0000; font-weight: bold;">
        Observations of Game ID Column:
    </p>
    <ul style="font-family: 'Georgia', serif; font-size: 14px; color: black; margin-left: 20px;">
        <li>There are total <strong>136 Unique Game IDs</strong> present in the NFL Big Bowl 2025 Games.csv file.</li>
        <li>The <strong>Minimum Game ID</strong> of the players according to this dataset is about <strong>2022090800</strong>.</li>
        <li>The <strong>Maximum Game ID</strong> of the players according to this dataset is about <strong>2022110700</strong>.</li>
    </ul>
</div>

# <p id="explore-season-column" style="background-color: #FFDDC1; font-family: 'Palatino Linotype', 'Book Antiqua', Palatino, serif; color: #8a0808; font-size: 1.5em; text-align: center; border-radius: 35px 20px 35px 20px; padding: 15px 25px; line-height: 1.5; border: 4px double #8a0808; box-shadow: 5px 5px 20px rgba(0, 0, 0, 0.5), -5px -5px 20px rgba(0, 0, 0, 0.3); margin: 0 auto; width: auto;">Let's Explore the Season Column</p>

```python
min_season=df["season"].min()
print("The Minimum Game season for the players according to this Dataset is:", min_season)
max_season=df["season"].max()
print("The Maximum Game season for the players according to this Dataset is:", max_season)
```

```python
# Get the unique seasons and teams
unique_seasons = df['season'].unique()
unique_teams = df['homeTeamAbbr'].unique()

# Get the first and last game dates
first_game_date = df['gameDate'].min()  # Assuming 'gameDate' column contains the game date
last_game_date = df['gameDate'].max()

# Print the first and last game dates
print("The data spans from the first game on", first_game_date, "to the last game on", last_game_date)
```

# <p id="season_homeabbr_column" style="background-color: #FFDDC1; font-family: 'Palatino Linotype', 'Book Antiqua', Palatino, serif; color: #8a0808; font-size: 1.5em; text-align: center; border-radius: 35px 20px 35px 20px; padding: 15px 25px; line-height: 1.5; border: 4px double #8a0808; box-shadow: 5px 5px 20px rgba(0, 0, 0, 0.5), -5px -5px 20px rgba(0, 0, 0, 0.3); margin: 0 auto; width: auto;">Let's Explore the Season and HomeAbbr Column</p>

```python
# Get the unique seasons and teams
unique_seasons = df['season'].unique()
unique_teams = df['homeTeamAbbr'].unique()

# Print the number of unique seasons and teams
print("The Number of Unique seasons present in this Dataset is:", len(unique_seasons))
print("The Number of Unique Home Teams present in this Dataset is:", len(unique_teams))
print("==================================================================================")
# Print the names of the unique seasons and teams
print("\nThe Unique Seasons in this Dataset are:", unique_seasons)
print("\nThe Unique Home Teams in this Dataset are:", unique_teams)
```

```python
# Unique seasons and teams
seasons = df['season'].nunique()
teams = df['homeTeamAbbr'].nunique()

# Count of games per season-homeTeamAbbr combination
season_team_counts = df.groupby(['season', 'homeTeamAbbr']).size().reset_index(name='game_count')
print("Game counts per season-team:\n")
season_team_counts
```

# <p id="visualize_season_homeabbr_bar_chart" style="background-color: #FFDDC1; font-family: 'Palatino Linotype', 'Book Antiqua', Palatino, serif; color: #8a0808; font-size: 1.5em; text-align: center; border-radius: 35px 20px 35px 20px; padding: 15px 25px; line-height: 1.5; border: 4px double #8a0808; box-shadow: 5px 5px 20px rgba(0, 0, 0, 0.5), -5px -5px 20px rgba(0, 0, 0, 0.3); margin: 0 auto; width: auto;">Let's Visualize the Season and HomeAbbr Column Using Bar Chart</p>

```python
# Unique seasons and teams
seasons = df['season'].nunique()
teams = df['homeTeamAbbr'].nunique()

# Count of games per season-homeTeamAbbr combination
season_team_counts = df.groupby(['season', 'homeTeamAbbr']).size().reset_index(name='game_count')

# Create a color column with pink-reddish shades (using rgba format for a smooth color scale)
season_team_counts['color'] = season_team_counts['game_count'].apply(
    lambda x: 'rgba(255, 105, 180, ' + str(x / season_team_counts['game_count'].max()) + ')'
)  # Pink to red shades

# Create a bar plot using pink-reddish shades
fig = px.bar(season_team_counts,
             x='homeTeamAbbr',
             y='game_count',
             color='game_count',
             title='Game Counts per Season-Home Team Combination',
             labels={'homeTeamAbbr': 'Home Team', 'game_count': 'Game Count'},
             color_continuous_scale=px.colors.sequential.Redor,  # Adjusted color scale for pink-reddish tones
             animation_frame='season',
             height=600)

# Customize layout
fig.update_traces(texttemplate='', textposition='outside')  # Remove text above bars
fig.update_layout(yaxis=dict(title='Game Count'),
                  xaxis=dict(title='Home Team'),
                  title_x=0.5,
                  barmode='group',
                  showlegend=False)

# Show the plot
fig.show()
```

# <p id="visualize_season_homeabbr_bar_pie_chart_subplots" style="background-color: #FFDDC1; font-family: 'Palatino Linotype', 'Book Antiqua', Palatino, serif; color: #8a0808; font-size: 1.5em; text-align: center; border-radius: 35px 20px 35px 20px; padding: 15px 25px; line-height: 1.5; border: 4px double #8a0808; box-shadow: 5px 5px 20px rgba(0, 0, 0, 0.5), -5px -5px 20px rgba(0, 0, 0, 0.3); margin: 0 auto; width: auto;">Let's Visualize the Season and HomeAbbr Column Using Bar & Pie Chart Subplots</p>

```python
# Count of games per season-homeTeamAbbr combination
season_team_counts = df.groupby(['season', 'homeTeamAbbr']).size().reset_index(name='game_count')

# Create a pie chart for total games played by each home team
total_games_by_team = season_team_counts.groupby('homeTeamAbbr')['game_count'].sum().reset_index()

# Create a bar plot for games per season-homeTeamAbbr combination
bar_fig = go.Figure()

# Loop through each season and add bars with a pink-reddish color scale
for season in season_team_counts['season'].unique():
    season_data = season_team_counts[season_team_counts['season'] == season]
    bar_fig.add_trace(go.Bar(
        x=season_data['homeTeamAbbr'],
        y=season_data['game_count'],
        name=f'Season {season}',
        hoverinfo='text',
        hovertemplate='<b>Team:</b> %{x}<br><b>Games:</b> %{y}<br><b>Season:</b> %{meta}<extra></extra>',
        meta=season,
        marker=dict(color=season_data['game_count'], colorscale='reds', colorbar=None)  # Removed the color bar
    ))

# Create the pie chart with a custom pink-red color scale
pie_fig = go.Figure(data=[go.Pie(
    labels=total_games_by_team['homeTeamAbbr'],
    values=total_games_by_team['game_count'],
    title='Total Games by Home Team',
    hole=0.4,
    marker=dict(colors=px.colors.sequential.RdBu_r),  # Pink-red color scale for the pie chart
    hovertemplate='<b>Team:</b> %{label}<br><b>Total Games:</b> %{value}<br><extra></extra>'
)])

# Create subplots: 1 row, 2 columns
fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=("Game Counts per Season-Home Team Combination", "Total Games by Home Team"),
    specs=[[{"type": "bar"}, {"type": "pie"}]]
)

# Add bar plot to the first subplot
for trace in bar_fig.data:
    fig.add_trace(trace, row=1, col=1)

# Add pie chart to the second subplot
for trace in pie_fig.data:
    fig.add_trace(trace, row=1, col=2)

# Customize layout
fig.update_layout(
    title_text='NFL Game Analysis',
    title_x=0.5,
    showlegend=True
)

# Show the plot
fig.show()
```

<div style="background-color: #FFDDC1; padding: 20px; border-radius: 15px; border: 2px solid #8B0000; margin-top: 20px;">
    <p style="text-align: justify; font-family: 'Times New Roman', serif; font-size: 20px; color: #8B0000; font-weight: bold;">
        Observations of Season & Home Abbr Columns:
    </p>
    <ul style="font-family: 'Georgia', serif; font-size: 14px; color: black; margin-left: 20px;">
        <li>The data spans from the first game on <strong>2022-09-08 00:00:00</strong> to the last game on <strong>2022-11-07 00:00:00</strong>.</li>
        <li>The First game Date is <strong>September 8, 2022</strong></li>
        <li>The Last game Date is <strong>November 7, 2022</strong></li>
        <li>According to this Dataset the Highest Games Count Percentage is  of Teams <strong>ARI, ATL, CAR, CLE, DAL,DET,JAX,LA, NYJ, TB, WAS and the Games count of these Team is about 5 and their percentage count is of about 3.68</strong></li>
        <li>According to this Dataset the Lowest Games Count Percentage is  of Teams <strong>TEN, SF, PIT, BUF, LV and the Games count of these Team is about 3 and their percentage count id of about 2,21</strong></li>
        <li>The <strong>Number of Unique Seasons</strong> present in this dataset is: <strong>1</strong>.</li>
        <li>The <strong>Number of Unique Home Teams</strong> present in this dataset is: <strong>32</strong>.</li>
        <li>The <strong>Unique Seasons</strong> in this dataset are: <strong>2022</strong>.</li>
        <li>The <strong>Unique Home Teams</strong> in this dataset are: <strong>'LA', 'ATL', 'CAR', 'CHI', 'CIN', 'DET', 'HOU', 'MIA', 'NYJ', 'WAS', 'ARI', 'LAC', 'MIN', 'TEN', 'DAL', 'SEA', 'KC', 'BAL', 'CLE', 'JAX', 'NO', 'NYG', 'PIT', 'SF', 'DEN', 'LV', 'GB', 'BUF', 'PHI', 'IND', 'NE', 'TB'</strong>.</li>
        <li>In the <strong>season 2022</strong>, the highest count, which is <strong>5</strong>, is for Home Teams such as ARI, ATL, CAR, CLE, DAL, DET, LA, NYJ, TB, and WAS.</li>
        <li>The maximum count is for the <strong>NO Home Team</strong> according to this dataset.</li>
        <li>Based on this dataset and the Games.csv file, it is determined that the <strong>lowest game count is 3</strong>, which includes teams such as BUF, LV, PHI, SF, and WAS.</li>
        <li>Home Teams with a moderate game count (neither too high & nor too low) include BAL, CHI, CAN, KC, LAC, MIA, MIN, NE, NYG, PHI, and SEA and the games count of these Teams is about 4</li>
        <li>It means that most of the Home Teams have the Count of about 4 which is neither too high nor too low</li>
    </ul>
</div>

<div style="background-color: #FFDDC1; padding: 20px; border-radius: 15px; border: 2px solid #8B0000; margin-top: 20px;">
    <p style="text-align: justify; font-family: 'Times New Roman', serif; font-size: 20px; color: #8B0000; font-weight: bold;">
        Observations of Season & Home Abbr Columns:
    </p>
    <div style="margin-bottom: 20px;">
        <h3 style="font-family: 'Georgia', serif; font-size: 18px; color: #8B0000; font-weight: bold;">Key Observations:</h3>
        <table style="width: 100%; font-family: 'Georgia', serif; font-size: 14px; border-collapse: collapse; margin-bottom: 20px;">
            <tr>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;"><strong>First Game Date:</strong></td>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;">2022-09-08 00:00:00</td>
            </tr>
            <tr>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;"><strong>Last Game Date:</strong></td>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;">2022-11-07 00:00:00</td>
            </tr>
            <tr>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;"><strong>Highest Games Count Percentage:</strong></td>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;">Teams: ARI, ATL, CAR, CLE, DAL, DET, JAX, LA, NYJ, TB, WAS with 5 games (3.68%)</td>
            </tr>
            <tr>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;"><strong>Lowest Games Count Percentage:</strong></td>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;">Teams: TEN, SF, PIT, BUF, LV with 3 games (2.21%)</td>
            </tr>
            <tr>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;"><strong>Number of Unique Seasons:</strong></td>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;">1</td>
            </tr>
            <tr>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;"><strong>Number of Unique Home Teams:</strong></td>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;">32</td>
            </tr>
            <tr>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;"><strong>Unique Season:</strong></td>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;">2022</td>
            </tr>
            <tr>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;"><strong>Unique Home Teams:</strong></td>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;">'LA', 'ATL', 'CAR', 'CHI', 'CIN', 'DET', 'HOU', 'MIA', 'NYJ', 'WAS', 'ARI', 'LAC', 'MIN', 'TEN', 'DAL', 'SEA', 'KC', 'BAL', 'CLE', 'JAX', 'NO', 'NYG', 'PIT', 'SF', 'DEN', 'LV', 'GB', 'BUF', 'PHI', 'IND', 'NE', 'TB'</td>
            </tr>
            <tr>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;"><strong>Teams with Highest Count:</strong></td>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;">Home Teams: ARI, ATL, CAR, CLE, DAL, DET, LA, NYJ, TB, WAS with 5 games</td>
            </tr>
            <tr>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;"><strong>Team with Maximum Count:</strong></td>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;">NO</td>
            </tr>
            <tr>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;"><strong>Teams with Lowest Game Count:</strong></td>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;">BUF, LV, PHI, SF, WAS with 3 games</td>
            </tr>
            <tr>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;"><strong>Teams with Moderate Game Count:</strong></td>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;">BAL, CHI, CAN, KC, LAC, MIA, MIN, NE, NYG, PHI, SEA with 4 games</td>
            </tr>
            <tr>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;"><strong>Most Common Home Team Game Count:</strong></td>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;">4 games (moderate count)</td>
            </tr>
        </table>
    </div>
</div>

# <p id="explore_season_visitorteamabbr_column" style="background-color: #FFDDC1; font-family: 'Palatino Linotype', 'Book Antiqua', Palatino, serif; color: #8a0808; font-size: 1.5em; text-align: center; border-radius: 35px 20px 35px 20px; padding: 15px 25px; line-height: 1.5; border: 4px double #8a0808; box-shadow: 5px 5px 20px rgba(0, 0, 0, 0.5), -5px -5px 20px rgba(0, 0, 0, 0.3); margin: 0 auto; width: auto;">Let's Explore Season and VisitorTeamAbbr Column</p>

```python
# Count of games per season-visitorTeamAbbr combination
season_visitor_counts = df.groupby(['season', 'visitorTeamAbbr']).size().reset_index(name='game_count')
print("Game counts per season-visitor team:\n")
season_visitor_counts
```

# <p id="visualize_season_visitorteamabbr_barchart" style="background-color: #FFDDC1; font-family: 'Palatino Linotype', 'Book Antiqua', Palatino, serif; color: #8a0808; font-size: 1.5em; text-align: center; border-radius: 35px 20px 35px 20px; padding: 15px 25px; line-height: 1.5; border: 4px double #8a0808; box-shadow: 5px 5px 20px rgba(0, 0, 0, 0.5), -5px -5px 20px rgba(0, 0, 0, 0.3); margin: 0 auto; width: auto;">Let's Visualize the Season and VisitorTeamAbbr Column using BarChart</p>

```python
# Count of games per season-visitorTeamAbbr combination
season_visitor_counts = df.groupby(['season', 'visitorTeamAbbr']).size().reset_index(name='game_count')

# Create a bar plot for visitor team game counts with pink-to-red shades
fig_visitor = px.bar(season_visitor_counts,
                     x='visitorTeamAbbr',
                     y='game_count',
                     color='game_count',
                     title='Game Counts per Season-Visitor Team Combination',
                     labels={'visitorTeamAbbr': 'Visitor Team', 'game_count': 'Game Count'},
                     color_continuous_scale='reds',  # Adjusted color scale for pink-reddish tones
                     animation_frame='season',
                     height=600)

# Customize layout
fig_visitor.update_traces(texttemplate='', textposition='outside')  # Remove text above bars
fig_visitor.update_layout(yaxis=dict(title='Game Count'),
                          xaxis=dict(title='Visitor Team'),
                          title_x=0.5,
                          barmode='group',
                          showlegend=False)

# Show the plot
fig_visitor.show()
```

# <p id="visualize_season_visitorteamabbr_columns" style="background-color: #FFDDC1; font-family: 'Palatino Linotype', 'Book Antiqua', Palatino, serif; color: #8a0808; font-size: 1.5em; text-align: center; border-radius: 35px 20px 35px 20px; padding: 15px 25px; line-height: 1.5; border: 4px double #8a0808; box-shadow: 5px 5px 20px rgba(0, 0, 0, 0.5), -5px -5px 20px rgba(0, 0, 0, 0.3); margin: 0 auto; width: auto;">Let's Visualize the Season and VisitorTeamAbbr Columns</p>

```python
# Count of games per season-visitorTeamAbbr combination
season_visitor_counts = df.groupby(['season', 'visitorTeamAbbr']).size().reset_index(name='game_count')

# Create a pie chart for total games played by each visitor team
total_games_by_visitor_team = season_visitor_counts.groupby('visitorTeamAbbr')['game_count'].sum().reset_index()

# Create a bar plot for games per season-visitorTeamAbbr combination
bar_fig_visitor = go.Figure()

# Add bar traces for each season with pink-reddish shades
for season in season_visitor_counts['season'].unique():
    season_data = season_visitor_counts[season_visitor_counts['season'] == season]
    bar_fig_visitor.add_trace(go.Bar(
        x=season_data['visitorTeamAbbr'],
        y=season_data['game_count'],
        name=f'Season {season}',
        hovertemplate="<b>Visitor Team: </b>%{x}<br><b>Game Count: </b>%{y}<extra></extra>",  # Custom hover data
        marker=dict(color=season_data['game_count'], colorscale='reds')  # Removed colorbar
    ))

# Create the pie chart with custom colors
pie_fig_visitor = go.Figure(data=[go.Pie(
    labels=total_games_by_visitor_team['visitorTeamAbbr'],
    values=total_games_by_visitor_team['game_count'],
    title='Total Games by Visitor Team',
    hole=0.4,
    hovertemplate="<b>Visitor Team: </b>%{label}<br><b>Game Count: </b>%{value}<extra></extra>",  # Custom hover data
    marker=dict(colors=px.colors.sequential.RdBu_r)  # Pink-red color scale for the pie chart
)])

# Create subplots: 1 row, 2 columns
fig_visitor = make_subplots(
    rows=1, cols=2,
    subplot_titles=("Game Counts per Season-Visitor Team Combination", "Total Games by Visitor Team"),
    specs=[[{"type": "bar"}, {"type": "pie"}]]
)

# Add bar plot to the first subplot
for trace in bar_fig_visitor.data:
    fig_visitor.add_trace(trace, row=1, col=1)

# Add pie chart to the second subplot
for trace in pie_fig_visitor.data:
    fig_visitor.add_trace(trace, row=1, col=2)

# Customize layout
fig_visitor.update_layout(
    title_text='NFL Game Analysis - Visitor Teams',
    title_x=0.5,
    showlegend=True
)

# Show the plot
fig_visitor.show()
```

<div style="background-color: #FFDDC1; padding: 20px; border-radius: 15px; border: 2px solid #8B0000; margin-top: 20px;">
    <p style="text-align: justify; font-family: 'Times New Roman', serif; font-size: 20px; color: #8B0000; font-weight: bold;">
        Observations of Season & Visitor Team Abbr Columns:
    </p>
    <ul style="font-family: 'Georgia', serif; font-size: 14px; color: black; margin-left: 20px;">
        <li>According to this dataset, the highest game count is associated with visitor teams such as <strong>BAL, BUF, CHI, CIN, GB, IND, LV, MIA, NE, PIT, SEA, SF, and TEN</strong>. These teams have a game count of approximately <strong>5</strong>, which accounts for about <strong>3.68%</strong> of the total games.</li>
        <li>The lowest game count is associated with teams such as <strong>NO, LA, DET, DAL, and CLE</strong>. These teams have a game count of approximately <strong>3</strong>, which represents about <strong>2.21%</strong> of the total games.</li>
        <li>The <strong>number of unique seasons</strong> present in this dataset is: <strong>1</strong>.</li>
        <li>The <strong>number of unique home teams</strong> in this dataset is: <strong>32</strong>.</li>
        <li>The <strong>unique season</strong> in this dataset is: <strong>2022</strong>.</li>
        <li>The <strong>unique home teams</strong> in this dataset are: <strong>'LA', 'ATL', 'CAR', 'CHI', 'CIN', 'DET', 'HOU', 'MIA', 'NYJ', 'WAS', 'ARI', 'LAC', 'MIN', 'TEN', 'DAL', 'SEA', 'KC', 'BAL', 'CLE', 'JAX', 'NO', 'NYG', 'PIT', 'SF', 'DEN', 'LV', 'GB', 'BUF', 'PHI', 'IND', 'NE', 'TB'</strong>.</li>
        <li>In the <strong>2022 season</strong>, the highest game count is <strong>5</strong> for home teams such as <strong>BAL, BUF, CHI, CIN, GB, IND, LV, MIA, NE, PIT, SEA, SF, and TEN</strong>.</li>
        <li>The maximum game count in this dataset is <strong>5</strong> for teams such as <strong>BAL, BUF, CHI, CIN, GB, IND, LV, MIA, NE, PIT, SEA, SF, and TEN</strong>.</li>
        <li>The lowest game count, which is <strong>3</strong>, includes visitor teams such as <strong>CLE, DAL, DET, LA, and NO</strong>.</li>
        <li>Visitor teams with a moderate game count (neither too high nor too low) include <strong>ARI, ATL, CAR, DEN, HOU, KC, JAX, LAC, MIN, PHI, NYG, TB, and WAS</strong>. These teams have a game count of approximately <strong>4</strong>, representing about <strong>2.94%</strong> of the total games.</li>
        <li>This indicates that most home teams have a game count of around <strong>4</strong>, which is moderate compared to the highest and lowest counts.</li>
    </ul>
</div>

<div style="background-color: #FFDDC1; padding: 20px; border-radius: 15px; border: 2px solid #8B0000; margin-top: 20px;">
    <p style="text-align: justify; font-family: 'Times New Roman', serif; font-size: 20px; color: #8B0000; font-weight: bold;">
        Observations of Season & Visitor Team Abbr Columns:
    </p>
    <div style="margin-bottom: 20px;">
        <h3 style="font-family: 'Georgia', serif; font-size: 18px; color: #8B0000; font-weight: bold;">Visitor Team Game Count Overview:</h3>
        <table style="width: 100%; font-family: 'Georgia', serif; font-size: 14px; border-collapse: collapse; margin-bottom: 20px;">
            <tr>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;"><strong>Highest Game Count:</strong></td>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;">BAL, BUF, CHI, CIN, GB, IND, LV, MIA, NE, PIT, SEA, SF, TEN (5 games, 3.68%)</td>
            </tr>
            <tr>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;"><strong>Lowest Game Count:</strong></td>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;">NO, LA, DET, DAL, CLE (3 games, 2.21%)</td>
            </tr>
            <tr>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;"><strong>Moderate Game Count:</strong></td>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;">ARI, ATL, CAR, DEN, HOU, KC, JAX, LAC, MIN, PHI, NYG, TB, WAS (4 games, 2.94%)</td>
            </tr>
            <tr>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;"><strong>Unique Seasons:</strong></td>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;">1 season (2022)</td>
            </tr>
            <tr>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;"><strong>Unique Home Teams:</strong></td>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;">32 teams</td>
            </tr>
            <tr>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;"><strong>Teams with Maximum Game Count:</strong></td>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;">BAL, BUF, CHI, CIN, GB, IND, LV, MIA, NE, PIT, SEA, SF, TEN (5 games)</td>
            </tr>
            <tr>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;"><strong>Teams with Lowest Game Count:</strong></td>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;">CLE, DAL, DET, LA, NO (3 games)</td>
            </tr>
        </table>
    </div>
</div>

# <p id="explore-seasonal-home-team-score" style="background-color: #FFDDC1; font-family: 'Palatino Linotype', 'Book Antiqua', Palatino, serif; color: #8a0808; font-size: 1.5em; text-align: center; border-radius: 35px 20px 35px 20px; padding: 15px 25px; line-height: 1.5; border: 4px double #8a0808; box-shadow: 5px 5px 20px rgba(0, 0, 0, 0.5), -5px -5px 20px rgba(0, 0, 0, 0.3); margin: 0 auto; width: auto;">Let's Explore the Seasonal Home Team Score</p>

```python
# Mean and median scores by season and home team
season_team_scores = df.groupby(['season', 'homeTeamAbbr'])[['homeFinalScore', 'visitorFinalScore']].agg(['mean', 'median', 'std']).reset_index()
print("Score statistics per season-homeTeamAbbr combination:\n")
season_team_scores
```

# <p id="visualize-seasonal-home-team-score" style="background-color: #FFDDC1; font-family: 'Palatino Linotype', 'Book Antiqua', Palatino, serif; color: #8a0808; font-size: 1.5em; text-align: center; border-radius: 35px 20px 35px 20px; padding: 15px 25px; line-height: 1.5; border: 4px double #8a0808; box-shadow: 5px 5px 20px rgba(0, 0, 0, 0.5), -5px -5px 20px rgba(0, 0, 0, 0.3); margin: 0 auto; width: auto;">Let's Visualize the Seasonal Home Team Score</p>

```python
# Calculate mean, median, and standard deviation scores by season and home team
season_team_scores = df.groupby(['season', 'homeTeamAbbr'])[['homeFinalScore', 'visitorFinalScore']].agg(['mean', 'median', 'std']).reset_index()

# Rename columns for easier access
season_team_scores.columns = ['season', 'homeTeamAbbr', 'homeMeanScore', 'homeMedianScore', 'homeStdDev', 
                              'visitorMeanScore', 'visitorMedianScore', 'visitorStdDev']

# Create subplots for pie charts
fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=['Home Team Mean Score Distribution', 'Visitor Team Mean Score Distribution'],
    specs=[[{'type': 'pie'}, {'type': 'pie'}]]
)

# Define a color palette (you can customize this)
home_team_colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#c2c2f0']  # Example palette
visitor_team_colors = ['#ffccff', '#ffb3e6', '#c2f0c2', '#ff6666', '#ffb366']  # Example palette

# Loop through each season to create a pie chart for the home team scores
for season in season_team_scores['season'].unique():
    season_data = season_team_scores[season_team_scores['season'] == season]
    
    # Pie chart for home team scores
    fig.add_trace(
        go.Pie(
            labels=season_data['homeTeamAbbr'],
            values=season_data['homeMeanScore'],
            name=f'Home Team Mean ({season})',
            hovertemplate=(
                'Home Team: %{label}<br>'  # Shows the team name
                'Mean Score: %{value}<br>'  # Shows the score
                '<extra></extra>'  # Hides additional information (legend)
            ),
            hole=0.3,  # Donut chart style
            marker=dict(colors=home_team_colors[:len(season_data)]),  # Assign custom colors
        ),
        row=1, col=1
    )
    
    # Pie chart for visitor team scores
    fig.add_trace(
        go.Pie(
            labels=season_data['homeTeamAbbr'],
            values=season_data['visitorMeanScore'],
            name=f'Visitor Team Mean ({season})',
            hovertemplate=(
                'Visitor Team: %{label}<br>'  # Shows the team name
                'Mean Score: %{value}<br>'  # Shows the score
                '<extra></extra>'  # Hides additional information (legend)
            ),
            hole=0.3,  # Donut chart style
            marker=dict(colors=visitor_team_colors[:len(season_data)]),  # Assign custom colors
        ),
        row=1, col=2
    )

# Update layout for better presentation
fig.update_layout(
    title='Mean Score Distribution by Home and Visitor Teams for Each Season',
    title_font=dict(size=24, family='Arial Black', color='darkslategray'),
    title_x=0.5,  # Center the title horizontally
    title_y=0.95,  # Increase the distance between title and graph by adjusting vertical position
    showlegend=True,
    legend=dict(
        title='Score Type',
        title_font=dict(size=14, family='Arial', color='darkslategray'),
        font=dict(size=12),
        bgcolor='lightgrey',
        bordercolor='black',
        borderwidth=1
    ),
    margin=dict(l=40, r=40, t=100, b=40),  # Increase the top margin to push the title further away from the graph
    height=800,
    width=1200,
    template='plotly_white'
)

# Show the plot
fig.show()
```

```python
# Calculate mean, median, and standard deviation scores by season and home team
season_team_scores = df.groupby(['season', 'homeTeamAbbr'])[['homeFinalScore', 'visitorFinalScore']].agg(['mean', 'median', 'std']).reset_index()

# Rename columns for easier access
season_team_scores.columns = ['season', 'homeTeamAbbr', 'homeMeanScore', 'homeMedianScore', 'homeStdDev', 
                              'visitorMeanScore', 'visitorMedianScore', 'visitorStdDev']

# Create a figure
fig = go.Figure()

# Add home mean score traces with a specific reddish tone
for season in season_team_scores['season'].unique():
    season_data = season_team_scores[season_team_scores['season'] == season]
    fig.add_trace(go.Bar(
        x=season_data['homeTeamAbbr'],
        y=season_data['homeMeanScore'],
        name=f'Home Team Mean ({season})',
        hovertemplate=(
            'Home Team: %{x}<br>'
            'Home Team Mean Score: %{y:.1f}<br>'
            '<extra></extra>'
        ),
        marker=dict(color='rgb(255, 102, 102)'),  # Reddish tone for home scores
        text=season_data['homeMeanScore'].round(1),
        textposition='outside',
        textfont=dict(size=14, color='black')
    ))

# Add visitor mean score traces with a slightly different reddish tone
for season in season_team_scores['season'].unique():
    season_data = season_team_scores[season_team_scores['season'] == season]
    fig.add_trace(go.Bar(
        x=season_data['homeTeamAbbr'],
        y=season_data['visitorMeanScore'],
        name=f'Visitor Team Mean ({season})',
        hovertemplate=(
            'Visitor Team: %{x}<br>'
            'Visitor Team Mean Score: %{y:.1f}<br>'
            '<extra></extra>'
        ),
        marker=dict(color='rgb(255, 178, 178)'),  # Lighter reddish tone for visitor scores
        text=season_data['visitorMeanScore'].round(1),
        textposition='outside',
        textfont=dict(size=14, color='black')
    ))

# Update layout for better presentation
fig.update_layout(
    title='Mean Scores by Season and Team',
    title_font=dict(size=24, family='Arial Black', color='darkslategray'),
    xaxis_title='Home Team Abbreviation',
    xaxis_title_font=dict(size=16, family='Arial', color='darkslategray'),
    yaxis_title='Mean Score',
    yaxis_title_font=dict(size=16, family='Arial', color='darkslategray'),
    barmode='group',
    title_x=0.5,
    template='plotly_white',
    showlegend=True,
    legend=dict(
        title='Score Type',
        title_font=dict(size=14, family='Arial', color='darkslategray'),
        font=dict(size=12),
        bgcolor='lightgrey',
        bordercolor='black',
        borderwidth=1
    ),
    margin=dict(l=40, r=40, t=60, b=40),
    height=700,
    width=1200,
)

# Add grid lines for better readability
fig.update_xaxes(showgrid=True, gridcolor='lightgray')
fig.update_yaxes(showgrid=True, gridcolor='lightgray')

# Adjust y-axis range to fit text
max_y = max(season_team_scores['homeMeanScore'].max(), season_team_scores['visitorMeanScore'].max())
fig.update_yaxes(range=[0, max_y + 10])

# Show the plot
fig.show()
```

<div style="background-color: #FFDDC1; padding: 20px; border-radius: 15px; border: 2px solid #8B0000; margin-top: 20px;">
    <p style="text-align: justify; font-family: 'Times New Roman', serif; font-size: 20px; color: #8B0000; font-weight: bold;">
        Observations of Season Vs Home Team Abbr Columns:
    </p>
    <ul style="font-family: 'Georgia', serif; font-size: 14px; color: black; margin-left: 20px;">
        <li>According to this dataset, the highest game score is associated with Home teams such as the Highest score of about 35.3<strong> which is of BUF of Home Team</strong></li>
        <li>The <strong>Highest Home Team Score is of the Team BUF and its score is of about:</strong> in this dataset is: <strong>35.6</strong>.</li>
        <li>The <strong>Highest Home Team Score is of the Team BUF in terms of percentage is of about:</strong> in this dataset is: <strong>4.84 and its Mean Score is about 35.33</strong>.</li>
        <li>The <strong>Lowest Home Team Score is of the Team DEN in terms of percentage is of about:</strong> in this dataset is: <strong>1.54 and its Mean Score is about 11.25</strong>.</li>
        <li>The <strong>Home Team Score which is neither too High nor too Low is of the Team MIN & PHI and their Mean Scores is of about:</strong> in this dataset is: <strong>28.5</strong>.</li>
        <li>The <strong>Home Team Score which is neither too High nor too Low is of the Team MIN & PHI and their Mean Scores Percentage is of about:</strong> in this dataset is: <strong>3.9</strong>.</li>
    </ul>
</div>

<div style="background-color: #FFDDC1; padding: 20px; border-radius: 15px; border: 2px solid #8B0000; margin-top: 20px;">
    <p style="text-align: justify; font-family: 'Times New Roman', serif; font-size: 20px; color: #8B0000; font-weight: bold;">
        Observations of Season Vs Visitor Team Abbr Columns:
    </p>
    <ul style="font-family: 'Georgia', serif; font-size: 14px; color: black; margin-left: 20px;">
        <li>The lowest game score is associated with Visitor Team such as <strong>The Team BUF of Visitor Team have the lowest score of about<strong>9.0</strong></li>
        <li>The <strong>Highest Visitor Team Score is of the Team DET and its score is of about: <strong>30.6</strong>.</li>
        <li>The <strong>Highest Visitor Team Score is of the Team DET in terms of percentage is of about:</strong> in this dataset is: <strong>4.63 and its Mean Score is about 30.6</strong>.</li>
        <li>The <strong>Lowest Visitor Team Score is of the Team BUF in terms of percentage is of about:</strong> in this dataset is: <strong>1.63 and its Mean Score is about 9</strong>.</li>
        <li>The <strong>Visitor Team Score which is neither too High nor too Low is of the Team BAL and their Mean Scores is of about:</strong> in this dataset is: <strong>25.5</strong>.</li>
        <li>The <strong>Visitor Team Score which is neither too High nor too Low is of the Team BAL and their Mean Scores Percentage is of about:</strong> in this dataset is: <strong>3.86</strong>.</li>
    </ul>
</div>

<div style="background-color: #FFDDC1; padding: 20px; border-radius: 15px; border: 2px solid #8B0000; margin-top: 20px;">
    <p style="text-align: justify; font-family: 'Times New Roman', serif; font-size: 20px; color: #8B0000; font-weight: bold;">
        Observations of Season Vs Home Team Abbr Columns:
    </p>
    <div style="margin-bottom: 20px;">
        <h3 style="font-family: 'Georgia', serif; font-size: 18px; color: #8B0000; font-weight: bold;">Home Team Scores Overview:</h3>
        <table style="width: 100%; font-family: 'Georgia', serif; font-size: 14px; border-collapse: collapse; margin-bottom: 20px;">
            <tr>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;"><strong>Highest Home Team Score:</strong></td>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;">BUF (35.6)</td>
            </tr>
            <tr>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;"><strong>Highest Home Team Score (Percentage):</strong></td>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;">BUF (4.84%) with Mean Score 35.33</td>
            </tr>
            <tr>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;"><strong>Lowest Home Team Score (Percentage):</strong></td>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;">DEN (1.54%) with Mean Score 11.25</td>
            </tr>
            <tr>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;"><strong>Average Home Team Score:</strong></td>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;">MIN & PHI (28.5)</td>
            </tr>
            <tr>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;"><strong>Average Home Team Score (Percentage):</strong></td>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;">MIN & PHI (3.9%)</td>
            </tr>
        </table>
    </div>
</div>

<div style="background-color: #FFDDC1; padding: 20px; border-radius: 15px; border: 2px solid #8B0000; margin-top: 20px;">
    <p style="text-align: justify; font-family: 'Times New Roman', serif; font-size: 20px; color: #8B0000; font-weight: bold;">
        Observations of Season Vs Visitor Team Abbr Columns:
    </p>
    <div style="margin-bottom: 20px;">
        <h3 style="font-family: 'Georgia', serif; font-size: 18px; color: #8B0000; font-weight: bold;">Visitor Team Scores Overview:</h3>
        <table style="width: 100%; font-family: 'Georgia', serif; font-size: 14px; border-collapse: collapse; margin-bottom: 20px;">
            <tr>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;"><strong>Lowest Visitor Team Score:</strong></td>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;">BUF (9.0)</td>
            </tr>
            <tr>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;"><strong>Highest Visitor Team Score:</strong></td>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;">DET (30.6)</td>
            </tr>
            <tr>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;"><strong>Highest Visitor Team Score (Percentage):</strong></td>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;">DET (4.63%) with Mean Score 30.6</td>
            </tr>
            <tr>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;"><strong>Lowest Visitor Team Score (Percentage):</strong></td>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;">BUF (1.63%) with Mean Score 9</td>
            </tr>
            <tr>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;"><strong>Average Visitor Team Score:</strong></td>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;">BAL (25.5)</td>
            </tr>
            <tr>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;"><strong>Average Visitor Team Score (Percentage):</strong></td>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;">BAL (3.86%)</td>
            </tr>
        </table>
    </div>
</div>

# <p id="explore-win-rate-home-visitor-scores" style="background-color: #FFDDC1; font-family: 'Palatino Linotype', 'Book Antiqua', Palatino, serif; color: #8a0808; font-size: 1.5em; text-align: center; border-radius: 35px 20px 35px 20px; padding: 15px 25px; line-height: 1.5; border: 4px double #8a0808; box-shadow: 5px 5px 20px rgba(0, 0, 0, 0.5), -5px -5px 20px rgba(0, 0, 0, 0.3); margin: 0 auto; width: auto;">Let's Explore the Win Rate of Home Final Score and Visitor Final Score</p>

```python
# Add a win column
df['homeWin'] = df['homeFinalScore'] > df['visitorFinalScore']

# Calculate win rate by season and team
season_team_winrate = df.groupby(['season', 'homeTeamAbbr'])['homeWin'].mean().reset_index(name='win_rate')
print("Win rates per season and team:\n")
season_team_winrate
```

# <p id="visualize-home-team-win-rate-season" style="background-color: #FFDDC1; font-family: 'Palatino Linotype', 'Book Antiqua', Palatino, serif; color: #8a0808; font-size: 1.5em; text-align: center; border-radius: 35px 20px 35px 20px; padding: 15px 25px; line-height: 1.5; border: 4px double #8a0808; box-shadow: 5px 5px 20px rgba(0, 0, 0, 0.5), -5px -5px 20px rgba(0, 0, 0, 0.3); margin: 0 auto; width: auto;">Let's Visualize the Win Rate of Home Team by Season</p>

```python
# Add a win column
df['homeWin'] = df['homeFinalScore'] > df['visitorFinalScore']

# Calculate win rate by season and team
season_team_winrate = df.groupby(['season', 'homeTeamAbbr'])['homeWin'].mean().reset_index(name='win_rate')

# Create a list of unique seasons
seasons = season_team_winrate['season'].unique()

# Create subplots for each season
fig = make_subplots(rows=len(seasons), cols=1, subplot_titles=[f'Season {season}' for season in seasons])

# Define a new pinkish shades color palette using the pink-to-red gradient
color_palette = [
    'rgba(255, 105, 97, 0.8)',  # Light Pink
    'rgba(255, 105, 180, 0.8)',  # Hot Pink
    'rgba(219, 112, 147, 0.8)',  # Pale Violet Red
    'rgba(216, 191, 216, 0.8)',  # Thistle
    'rgba(255, 20, 147, 0.8)',   # Deep Pink
    'rgba(199, 21, 133, 0.8)',   # Medium Violet Red
    'rgba(248, 187, 208, 0.8)',  # Cotton Candy
]

# Add bar charts to each subplot
for i, season in enumerate(seasons):
    season_data = season_team_winrate[season_team_winrate['season'] == season]
    
    # Use the new pinkish color palette for each season's bars
    fig.add_trace(go.Bar(
        x=season_data['homeTeamAbbr'],
        y=season_data['win_rate'],
        name=f'Win Rate ({season})',
        hovertemplate=(
            'Home Team: %{x}<br>'  # Display the team name
            'Win Rate: %{y:.2f}<br>'  # Display the win rate with two decimal points
            '<extra></extra>'  # Remove extra trace info (like the trace name)
        ),
        marker=dict(color=season_data['win_rate'], colorscale='reds', colorbar=dict(title="Win Rate")),
        text=(season_data['win_rate'] * 100).round(1),  # Convert to percentage for display
        textposition='outside',  # Position text outside the bar
        textfont=dict(size=16, color='black', family='Arial')  # Text style
    ), row=i + 1, col=1)

# Update layout to enhance appearance
fig.update_layout(
    title='Home Team Win Rates by Season',
    title_font=dict(size=24, family='Arial Black', color='darkslategray'),
    xaxis_title='Home Team Abbreviation',
    xaxis_title_font=dict(size=16, family='Arial', color='darkslategray'),
    yaxis_title='Win Rate',
    yaxis_title_font=dict(size=16, family='Arial', color='darkslategray'),
    template='plotly_white',
    margin=dict(l=40, r=40, t=80, b=40),  # Adjust margins for better spacing
    height=800,  # Set height for the plot
    showlegend=False,  # Hide the legend as it's redundant in this context
)

# Update axes for better readability
for i in range(len(seasons)):
    fig.update_yaxes(range=[0, 1], tickvals=[0, 0.25, 0.5, 0.75, 1], ticktext=['0%', '25%', '50%', '75%', '100%'], row=i + 1, col=1)
    fig.update_xaxes(title='Home Team Abbreviation', row=i + 1, col=1)

# Show the plot
fig.show()
```

# <p id="home-team-win-rate-season-pie-chart" style="background-color: #FFDDC1; font-family: 'Palatino Linotype', 'Book Antiqua', Palatino, serif; color: #8a0808; font-size: 1.5em; text-align: center; border-radius: 35px 20px 35px 20px; padding: 15px 25px; line-height: 1.5; border: 4px double #8a0808; box-shadow: 5px 5px 20px rgba(0, 0, 0, 0.5), -5px -5px 20px rgba(0, 0, 0, 0.3); margin: 0 auto; width: auto;">Home Team Win Rate by Season: Pie Chart Visualization</p>

```python
# Add a win column
df['homeWin'] = df['homeFinalScore'] > df['visitorFinalScore']

# Calculate win rate by season and team
season_team_winrate = df.groupby(['season', 'homeTeamAbbr'])['homeWin'].mean().reset_index(name='win_rate')

# Create a list of unique seasons
seasons = season_team_winrate['season'].unique()

# Create subplots for each season with domain type for pie charts
fig = make_subplots(
    rows=len(seasons), 
    cols=1, 
    subplot_titles=[f'Season {season}' for season in seasons], 
    specs=[[{'type': 'domain'}] for _ in seasons]  # Specify 'domain' for pie charts
)

# Define a pink-to-red color palette with more vibrant shades
colors = ['#FF6F61', '#FF69B4', '#FF1493', '#DB7093', '#FF1493', '#C71585', '#FF6347']

# Add donut charts to each subplot
for i, season in enumerate(seasons):
    season_data = season_team_winrate[season_team_winrate['season'] == season]
    
    # Add a donut chart trace with shadow
    fig.add_trace(go.Pie(
        labels=season_data['homeTeamAbbr'],
        values=(season_data['win_rate'] * 100).round(1),  # Convert to percentage for display
        name=f'Win Rate ({season})',
        hovertemplate=
            'Team: %{label}<br>' +
            'Win Rate: %{value}%<br>' +  # Display win rate as percentage
            'Season: ' + str(season) + '<extra></extra>',  # Add season info to hover
        marker=dict(colors=colors[:len(season_data)], line=dict(color='darkgray', width=2)),  # Darker border for contrast
        textinfo='percent+label',  # Display percentage and label on the chart
        textfont=dict(size=13, color='black', family='Arial Bold'),  # Slightly increased text size
        hole=0.3,  # Hole size for donut chart
        showlegend=True,
        opacity=0.95,  # Less transparent for better visibility
        pull=[0.1] * len(season_data)  # Slight pull effect to simulate shadow
    ), row=i + 1, col=1)

# Update layout to enhance appearance and slightly increase size
fig.update_layout(
    title='Home Team Win Rates',
    title_font=dict(size=27, family='Arial Black', color='darkblue'),  # Slightly larger title font size
    margin=dict(l=37, r=37, t=95, b=37),  # Increased margins very slightly
    height=680,  # Increased height a tiny bit more
    showlegend=True,
    legend=dict(
        title='Teams',
        title_font=dict(size=15, family='Arial Bold', color='darkblue'),  # Slightly larger legend font size
        font=dict(size=11, color='black'),  # Slightly larger legend font
        bgcolor='rgba(255, 255, 255, 0.7)',  # Semi-transparent background for legend
        bordercolor='darkgray',
        borderwidth=1
    ),
)

# Show the plot
fig.show()
```

<div style="background-color: #FFDDC1; padding: 20px; border-radius: 15px; border: 2px solid #8B0000; margin-top: 20px;">
    <p style="text-align: justify; font-family: 'Times New Roman', serif; font-size: 20px; color: #8B0000; font-weight: bold;">
        Observations of Home Team Win Rate:
    </p>
    <ul style="font-family: 'Georgia', serif; font-size: 14px; color: black; margin-left: 20px;">
        <li>According to this dataset, the highest Win Rate is associated with Home teams such as <strong>BUF, Min & PHI</strong>And they have the Win Rate of about<strong>1.0</strong>>which represents that these Teams have Highest chances to win the Game</li>
        <li>The lowest Win Rate is associated with Home Team such as <strong>HOU </strong>This Team have have the Win Rate of about <strong>0</strong>which represents that this Team have lowest chances to win the Game<strong></li>
            <li>Similarly The Teams such as<strong>ARI, PIT, NO </strong>have the lowest Win rate and lowest Win percentage of about <strong>20%, 33.3%</strong></li>
        <li>While Win Rate which is not Highest not lowest is associated with Home Team such as <strong>BAL, CHI, DEN, GB, IND, LAC & NE etc</strong>This means that these teams have the 50% chances to Win the Games and 50% chances to loose the Games<strong>As they have the win rate of about 50%</strong></li>
    </ul>
</div>

<div style="background-color: #FFDDC1; padding: 20px; border-radius: 15px; border: 2px solid #8B0000; margin-top: 20px;">
    <p style="text-align: justify; font-family: 'Times New Roman', serif; font-size: 20px; color: #8B0000; font-weight: bold;">
        Observations of Home Team Win Rate:
    </p>
    <div style="margin-bottom: 20px;">
        <h3 style="font-family: 'Georgia', serif; font-size: 18px; color: #8B0000; font-weight: bold;">Home Team Win Rate Overview:</h3>
        <table style="width: 100%; font-family: 'Georgia', serif; font-size: 14px; border-collapse: collapse; margin-bottom: 20px;">
            <tr>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;"><strong>Highest Win Rate Teams:</strong></td>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;">BUF, MIN, PHI (Win Rate = 1.0)</td>
            </tr>
            <tr>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;"><strong>Lowest Win Rate Team:</strong></td>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;">HOU (Win Rate = 0.0)</td>
            </tr>
            <tr>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;"><strong>Teams with Low Win Rate:</strong></td>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;">ARI, PIT, NO (Win Rate = 20%, 33.3%)</td>
            </tr>
            <tr>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;"><strong>Teams with Moderate Win Rate:</strong></td>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;">BAL, CHI, DEN, GB, IND, LAC, NE (Win Rate = 50%)</td>
            </tr>
        </table>
    </div>
</div>

# <p id="visitor-team-win-rate-season" style="background-color: #FFDDC1; font-family: 'Palatino Linotype', 'Book Antiqua', Palatino, serif; color: #8a0808; font-size: 1.5em; text-align: center; border-radius: 35px 20px 35px 20px; padding: 15px 25px; line-height: 1.5; border: 4px double #8a0808; box-shadow: 5px 5px 20px rgba(0, 0, 0, 0.5), -5px -5px 20px rgba(0, 0, 0, 0.3); margin: 0 auto; width: auto;">Let's Explore the Visitor Team Win Rate by Season</p>

```python
# Add a win column for visitor teams
df['visitorWin'] = df['visitorFinalScore'] > df['homeFinalScore']

# Calculate win rate by season and visitor team
season_visitor_winrate = df.groupby(['season', 'visitorTeamAbbr'])['visitorWin'].mean().reset_index(name='win_rate')
print("Win rates per season and visitor team:\n")
season_visitor_winrate
```

# <p id="visitor-team-win-rate-season-pie" style="background-color: #FFDDC1; font-family: 'Palatino Linotype', 'Book Antiqua', Palatino, serif; color: #8a0808; font-size: 1.5em; text-align: center; border-radius: 35px 20px 35px 20px; padding: 15px 25px; line-height: 1.5; border: 4px double #8a0808; box-shadow: 5px 5px 20px rgba(0, 0, 0, 0.5), -5px -5px 20px rgba(0, 0, 0, 0.3); margin: 0 auto; width: auto;">Visitor Team Win Rate by Season: Pie Chart Visualization</p>

```python
# Add a win column for visitor teams
df['visitorWin'] = df['visitorFinalScore'] > df['homeFinalScore']

# Calculate win rate by season and visitor team
season_visitor_winrate = df.groupby(['season', 'visitorTeamAbbr'])['visitorWin'].mean().reset_index(name='win_rate')

# Create a list of unique seasons
seasons = season_visitor_winrate['season'].unique()

# Create subplots for each season with domain type for pie charts
fig = make_subplots(
    rows=len(seasons), 
    cols=1, 
    subplot_titles=[f'Season {season}' for season in seasons], 
    specs=[[{'type': 'domain'}] for _ in seasons]  # Specify 'domain' for pie charts
)

# Define a color palette with more contrast and vibrant colors
colors = ['#FF9999', '#66B3FF', '#99FF99', '#FFCC99', '#FFD700', '#FF69B4', '#FF4500']

# Add donut charts to each subplot for visitor win rates
for i, season in enumerate(seasons):
    season_data = season_visitor_winrate[season_visitor_winrate['season'] == season]
    
    # Add a donut chart trace with shadow
    fig.add_trace(go.Pie(
        labels=season_data['visitorTeamAbbr'],
        values=(season_data['win_rate'] * 100).round(1),  # Convert to percentage for display
        name=f'Win Rate ({season})',
        hovertemplate=
            'Team: %{label}<br>' +  # Show the team name
            'Win Rate: %{value}%<br>' +  # Show the win rate percentage
            'Season: ' + str(season) + '<extra></extra>',  # Show the season information
        marker=dict(colors=colors[:len(season_data)], line=dict(color='darkgray', width=2)),  # Darker border for contrast
        textinfo='percent+label',  # Display percentage and label on the chart
        textfont=dict(size=14, color='black', family='Arial Bold'),  # Enhanced text size and bold
        hole=0.3,  # Increased hole size for a more defined donut chart
        showlegend=True,
        opacity=0.95,  # Less transparent for better visibility
        pull=[0.1] * len(season_data)  # Slight pull effect to simulate shadow
    ), row=i + 1, col=1)

# Update layout to enhance appearance
fig.update_layout(
    title='Visitor Team Win Rates',
    title_font=dict(size=28, family='Arial Black', color='darkblue'),
    margin=dict(l=40, r=40, t=100, b=40),  # Increase top margin for gap between title and plot
    height=800,  # Set height for the plot
    showlegend=True,  # Show legend for the pie charts
    legend=dict(
        title='Teams',
        title_font=dict(size=16, family='Arial Bold', color='darkblue'),
        font=dict(size=12, color='black'),
        bgcolor='rgba(255, 255, 255, 0.7)',  # Semi-transparent background for legend
        bordercolor='darkgray', 
        borderwidth=1
    ),
)

# Show the plot
fig.show()
```

<div style="background-color: #FFDDC1; padding: 20px; border-radius: 15px; border: 2px solid #8B0000; margin-top: 20px;">
    <p style="text-align: justify; font-family: 'Times New Roman', serif; font-size: 20px; color: #8B0000; font-weight: bold;">
        Observations of Visitor Team Win Rate:
    </p>
    <ul style="font-family: 'Georgia', serif; font-size: 14px; color: black; margin-left: 20px;">
        <li>According to this dataset, the highest Win Rate is associated with Visitor Teams teams such as <strong>PHI & NYJ</strong>And they have the Win Rate of about<strong>1.0</strong>>which represents that these Teams have Highest chances to win the Game</li>
        <li>The lowest Win Rate is associated with Home Teams such as <strong>PIT, IND, GIB, CHI </strong>This Team have have the Win Rate of about <strong>20%</strong>which represents that these Teams have lowest chances to win the Game<strong></li>
            <li>Similarly The Teams such as<strong>JAX, HOU, DEN & ATL </strong>have the lowest Win rate and lowest Win percentage of about <strong>25%</strong></li>
        <li>While Win Rate which is not Highest not lowest is associated with Home Team such as <strong>ARI, TB,& WAS etc</strong>This means that these teams have the 50% chances of Win Rate and to Win the Games and 50% chances to loose the Games<strong>As they have the win rate of about 50%</strong></li>
    </ul>
</div>

<div style="background-color: #FFDDC1; padding: 20px; border-radius: 15px; border: 2px solid #8B0000; margin-top: 20px;">
    <p style="text-align: justify; font-family: 'Times New Roman', serif; font-size: 20px; color: #8B0000; font-weight: bold;">
        Observations of Visitor Team Win Rate:
    </p>
    <div style="margin-bottom: 20px;">
        <h3 style="font-family: 'Georgia', serif; font-size: 18px; color: #8B0000; font-weight: bold;">Visitor Team Win Rate Overview:</h3>
        <table style="width: 100%; font-family: 'Georgia', serif; font-size: 14px; border-collapse: collapse; margin-bottom: 20px;">
            <tr>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;"><strong>Highest Win Rate Teams:</strong></td>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;">PHI, NYJ (Win Rate = 1.0)</td>
            </tr>
            <tr>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;"><strong>Lowest Win Rate Teams:</strong></td>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;">PIT, IND, GIB, CHI (Win Rate = 20%)</td>
            </tr>
            <tr>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;"><strong>Teams with Low Win Rate:</strong></td>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;">JAX, HOU, DEN, ATL (Win Rate = 25%)</td>
            </tr>
            <tr>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;"><strong>Teams with Moderate Win Rate:</strong></td>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;">ARI, TB, WAS (Win Rate = 50%)</td>
            </tr>
        </table>
    </div>
</div>

```python
# Calculate the win column for home and visitor teams
df['homeWin'] = df['homeFinalScore'] > df['visitorFinalScore']
df['visitorWin'] = df['visitorFinalScore'] > df['homeFinalScore']

# Calculate win rate by season and team (home and visitor)
season_team_winrate = df.groupby(['season', 'homeTeamAbbr'])['homeWin'].mean().reset_index(name='win_rate_home')
season_visitor_winrate = df.groupby(['season', 'visitorTeamAbbr'])['visitorWin'].mean().reset_index(name='win_rate_visitor')

# Merge both dataframes on season and team abbreviations for plotting
merged_winrate = pd.merge(season_team_winrate, season_visitor_winrate, left_on=['season', 'homeTeamAbbr'], right_on=['season', 'visitorTeamAbbr'])

# Create a figure
fig = go.Figure()

# Add home team win rate traces with a specific reddish tone
for season in merged_winrate['season'].unique():
    season_data = merged_winrate[merged_winrate['season'] == season]
    fig.add_trace(go.Bar(
        x=season_data['homeTeamAbbr'],
        y=season_data['win_rate_home'],
        name=f'Home Team Win Rate ({season})',
        hovertemplate=(
            'Home Team: %{x}<br>'
            'Win Rate: %{y:.2f}<br>'
            '<extra></extra>'
        ),
        marker=dict(color='rgb(255, 102, 102)'),  # Reddish tone for home team win rate
        text=season_data['win_rate_home'].round(2),
        textposition='outside',
        textfont=dict(size=14, color='black')
    ))

# Add visitor team win rate traces with a slightly different reddish tone
for season in merged_winrate['season'].unique():
    season_data = merged_winrate[merged_winrate['season'] == season]
    fig.add_trace(go.Bar(
        x=season_data['homeTeamAbbr'],
        y=season_data['win_rate_visitor'],
        name=f'Visitor Team Win Rate ({season})',
        hovertemplate=(
            'Visitor Team: %{x}<br>'
            'Win Rate: %{y:.2f}<br>'
            '<extra></extra>'
        ),
        marker=dict(color='rgb(255, 178, 178)'),  # Lighter reddish tone for visitor team win rate
        text=season_data['win_rate_visitor'].round(2),
        textposition='outside',
        textfont=dict(size=14, color='black')
    ))

# Update layout for better presentation
fig.update_layout(
    title='Win Rates by Season and Team',
    title_font=dict(size=24, family='Arial Black', color='darkslategray'),
    xaxis_title='Team Abbreviation',
    xaxis_title_font=dict(size=16, family='Arial', color='darkslategray'),
    yaxis_title='Win Rate',
    yaxis_title_font=dict(size=16, family='Arial', color='darkslategray'),
    barmode='group',
    title_x=0.5,
    template='plotly_white',
    showlegend=True,
    legend=dict(
        title='Team Type',
        title_font=dict(size=14, family='Arial', color='darkslategray'),
        font=dict(size=12),
        bgcolor='lightgrey',
        bordercolor='black',
        borderwidth=1
    ),
    margin=dict(l=40, r=40, t=60, b=40),
    height=700,
    width=1200,
)

# Add grid lines for better readability
fig.update_xaxes(showgrid=True, gridcolor='lightgray')
fig.update_yaxes(showgrid=True, gridcolor='lightgray')

# Adjust y-axis range to fit text
max_y = max(merged_winrate['win_rate_home'].max(), merged_winrate['win_rate_visitor'].max())
fig.update_yaxes(range=[0, max_y + 0.1])

# Show the plot
fig.show()
```

<div style="background-color: #FFDDC1; padding: 20px; border-radius: 15px; border: 2px solid #8B0000; margin-top: 20px;">
    <p style="text-align: justify; font-family: 'Times New Roman', serif; font-size: 20px; color: #8B0000; font-weight: bold;">
        Observations of Home Team Win Rate:
    </p>
    <ul style="font-family: 'Georgia', serif; font-size: 14px; color: black; margin-left: 20px;">
        <li>According to this dataset, the highest Win Rate is associated with Home teams such as <strong>BUF, Min & PHI</strong>And they have the Win Rate of about<strong>1.0</strong>>which represents that these Teams have Highest chances to win the Game</li>
        <li>The lowest Win Rate is associated with Home Team such as <strong>HOU </strong>This Team have have the Win Rate of about <strong>0</strong>which represents that this Team have lowest chances to win the Game<strong></li>
            <li>Similarly The Teams such as<strong>ARI, PIT, NO </strong>have the lowest Win rate and lowest Win percentage of about <strong>20%, 33.3%</strong></li>
        <li>While Win Rate which is not Highest not lowest is associated with Home Team such as <strong>BAL, CHI, DEN, GB, IND, LAC & NE etc</strong>This means that these teams have the 50% chances to Win the Games and 50% chances to loose the Games<strong>As they have the win rate of about 50%</strong></li>
    </ul>
</div>

<div style="background-color: #FFDDC1; padding: 20px; border-radius: 15px; border: 2px solid #8B0000; margin-top: 20px;">
    <p style="text-align: justify; font-family: 'Times New Roman', serif; font-size: 20px; color: #8B0000; font-weight: bold;">
        Observations of Visitor Team Win Rate:
    </p>
    <ul style="font-family: 'Georgia', serif; font-size: 14px; color: black; margin-left: 20px;">
        <li>According to this dataset, the highest Win Rate is associated with Visitor Teams teams such as <strong>PHI & NYJ</strong>And they have the Win Rate of about<strong>1.0</strong>>which represents that these Teams have Highest chances to win the Game</li>
        <li>The lowest Win Rate is associated with Home Teams such as <strong>PIT, IND, GIB, CHI </strong>This Team have have the Win Rate of about <strong>20%</strong>which represents that these Teams have lowest chances to win the Game<strong></li>
            <li>Similarly The Teams such as<strong>JAX, HOU, DEN & ATL </strong>have the lowest Win rate and lowest Win percentage of about <strong>25%</strong></li>
        <li>While Win Rate which is not Highest not lowest is associated with Home Team such as <strong>ARI, TB,& WAS etc</strong>This means that these teams have the 50% chances of Win Rate and to Win the Games and 50% chances to loose the Games<strong>As they have the win rate of about 50%</strong></li>
    </ul>
</div>

<div style="background-color: #FFDDC1; padding: 20px; border-radius: 15px; border: 2px solid #8B0000; margin-top: 20px;">
    <p style="text-align: justify; font-family: 'Times New Roman', serif; font-size: 20px; color: #8B0000; font-weight: bold;">
        Observations of Home Team Win Rate:
    </p>
    <div style="margin-bottom: 20px;">
        <h3 style="font-family: 'Georgia', serif; font-size: 18px; color: #8B0000; font-weight: bold;">Home Team Win Rate Overview:</h3>
        <table style="width: 100%; font-family: 'Georgia', serif; font-size: 14px; border-collapse: collapse; margin-bottom: 20px;">
            <tr>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;"><strong>Highest Win Rate Teams:</strong></td>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;">BUF, MIN & PHI</td>
            </tr>
            <tr>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;"><strong>Highest Win Rate Percentage:</strong></td>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;">1.0</td>
            </tr>
            <tr>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;"><strong>Lowest Win Rate Team:</strong></td>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;">HOU</td>
            </tr>
            <tr>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;"><strong>Lowest Win Rate Percentage:</strong></td>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;">0</td>
            </tr>
            <tr>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;"><strong>Teams with Low Win Rate (20%-33.3%):</strong></td>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;">ARI, PIT, NO</td>
            </tr>
            <tr>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;"><strong>Teams with Moderate Win Rate (50%):</strong></td>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;">BAL, CHI, DEN, GB, IND, LAC, NE</td>
            </tr>
        </table>
    </div>
</div>

<div style="background-color: #FFDDC1; padding: 20px; border-radius: 15px; border: 2px solid #8B0000; margin-top: 20px;">
    <p style="text-align: justify; font-family: 'Times New Roman', serif; font-size: 20px; color: #8B0000; font-weight: bold;">
        Observations of Visitor Team Win Rate:
    </p>
    <div style="margin-bottom: 20px;">
        <h3 style="font-family: 'Georgia', serif; font-size: 18px; color: #8B0000; font-weight: bold;">Visitor Team Win Rate Overview:</h3>
        <table style="width: 100%; font-family: 'Georgia', serif; font-size: 14px; border-collapse: collapse; margin-bottom: 20px;">
            <tr>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;"><strong>Highest Win Rate Teams:</strong></td>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;">PHI, NYJ</td>
            </tr>
            <tr>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;"><strong>Highest Win Rate Percentage:</strong></td>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;">1.0</td>
            </tr>
            <tr>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;"><strong>Lowest Win Rate Teams:</strong></td>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;">PIT, IND, GIB, CHI</td>
            </tr>
            <tr>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;"><strong>Lowest Win Rate Percentage:</strong></td>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;">20%</td>
            </tr>
            <tr>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;"><strong>Teams with Low Win Rate (20%-25%):</strong></td>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;">JAX, HOU, DEN, ATL</td>
            </tr>
            <tr>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;"><strong>Teams with Moderate Win Rate (50%):</strong></td>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;">ARI, TB, WAS</td>
            </tr>
        </table>
    </div>
</div>

# <p id="highest-scoring-games-per-team-season" style="background-color: #FFDDC1; font-family: 'Palatino Linotype', 'Book Antiqua', Palatino, serif; color: #8a0808; font-size: 1.5em; text-align: center; border-radius: 35px 20px 35px 20px; padding: 15px 25px; line-height: 1.5; border: 4px double #8a0808; box-shadow: 5px 5px 20px rgba(0, 0, 0, 0.5), -5px -5px 20px rgba(0, 0, 0, 0.3); margin: 0 auto; width: auto;">Let's Explore the Highest Scoring Games Per Team and Season</p>

```python
# Find the highest scoring game for each season and team
highest_scoring_games = df.groupby(['season', 'homeTeamAbbr']).apply(lambda x: x.nlargest(1, 'homeFinalScore')).reset_index(drop=True)

# Display the result with additional columns
print("Highest scoring games per season and team:\n")
highest_scoring_games[['season', 'homeTeamAbbr', 'visitorTeamAbbr', 'homeFinalScore', 'visitorFinalScore']]
```

# <p id="visualize-highest-scoring-games-distribution" style="background-color: #FFDDC1; font-family: 'Palatino Linotype', 'Book Antiqua', Palatino, serif; color: #8a0808; font-size: 1.5em; text-align: center; border-radius: 35px 20px 35px 20px; padding: 15px 25px; line-height: 1.5; border: 4px double #8a0808; box-shadow: 5px 5px 20px rgba(0, 0, 0, 0.5), -5px -5px 20px rgba(0, 0, 0, 0.3); margin: 0 auto; width: auto;">Let's Visualize the Distribution of Highest Scoring Games</p>

```python
# Find the highest scoring game for each season and team
highest_scoring_games = df.groupby(['season', 'homeTeamAbbr']).apply(lambda x: x.nlargest(1, 'homeFinalScore')).reset_index(drop=True)

# Combine home and visitor scores for the donut chart
scores = highest_scoring_games[['homeTeamAbbr', 'homeFinalScore', 'visitorFinalScore']]
scores = pd.melt(scores, id_vars='homeTeamAbbr', value_vars=['homeFinalScore', 'visitorFinalScore'], var_name='Score Type', value_name='Score')

# Create a donut chart with enhanced visuals
fig = go.Figure(data=[go.Pie(
    labels=scores['Score Type'],
    values=scores['Score'],
    hole=0.4,  # This creates the hole in the center
    textinfo='label+percent',  # Display the label and percentage
    insidetextorientation='radial',  # Orientation of the text
    marker=dict(colors=['#FF6666', '#8B0000'], line=dict(color='darkgray', width=2)),  # Lighter red and dark red shades for slices and border
    pull=[0.1, 0.1],  # Slightly pull the slices for a 3D effect
    hovertemplate=(
        'Score Type: %{label}<br>'  # Display the score type (Home vs Visitor)
        'Score: %{value}<br>'       # Display the actual score
        'Percentage: %{percent:.2f}%'  # Display the percentage of total
        '<extra></extra>'           # Remove extra information like trace name
    )
)])

# Add shadow effect
fig.update_traces(marker=dict(line=dict(width=1, color='white')),
                  textfont=dict(size=16, color='black'))  # Change text size and color

# Update layout for prominence with shadows and improved visibility
fig.update_layout(
    title='Distribution of Highest Scoring Games (Home vs Visitor)',
    title_font=dict(size=25, family='Arial Black', color='darkblue'),
    annotations=[dict(text='Scores', x=0.5, y=0.5, font_size=24, showarrow=False)],
    margin=dict(l=40, r=40, t=80, b=40),  # Adjust margins for better spacing
    height=500,  # Set height for the plot
    paper_bgcolor='rgba(255, 255, 255, 0.9)',  # Slightly transparent background
    plot_bgcolor='rgba(255, 255, 255, 0.9)',  # Slightly transparent plot area
)

# Add shadow effect to the entire figure
fig.update_traces(
    marker=dict(
        line=dict(color='rgba(0,0,0,0.1)', width=5),  # Shadow effect around the pie chart
    ),
)

# Show the plot
fig.show()
```

# <p id="visualize-home-vs-visitor-team-scoring" style="background-color: #FFDDC1; font-family: 'Palatino Linotype', 'Book Antiqua', Palatino, serif; color: #8a0808; font-size: 1.5em; text-align: center; border-radius: 35px 20px 35px 20px; padding: 15px 25px; line-height: 1.5; border: 4px double #8a0808; box-shadow: 5px 5px 20px rgba(0, 0, 0, 0.5), -5px -5px 20px rgba(0, 0, 0, 0.3); margin: 0 auto; width: auto;">Let's Visualize the Home vs Visitor Team Scoring of Games</p>

```python
# Find the highest scoring game for each season and team
highest_scoring_games = df.groupby(['season', 'homeTeamAbbr']).apply(lambda x: x.nlargest(1, 'homeFinalScore')).reset_index(drop=True)

# Create subplots
fig = make_subplots(
    rows=len(highest_scoring_games['season'].unique()), 
    cols=1, 
    subplot_titles=[f'Highest Scoring Games in Season {season}' for season in highest_scoring_games['season'].unique()],
)

# Define color scales for home and visitor teams
home_colorscale = 'Reds'  # Lighter colors for home
visitor_colorscale = 'Blues'  # Darker colors for visitors

# Iterate over seasons to add traces
for i, season in enumerate(highest_scoring_games['season'].unique()):
    season_data = highest_scoring_games[highest_scoring_games['season'] == season]
    
    # Add home team scores with light colors
    fig.add_trace(go.Bar(
        x=season_data['homeTeamAbbr'],
        y=season_data['homeFinalScore'],
        name='Home Final Score',
        marker=dict(
            color=season_data['homeFinalScore'],
            colorscale=home_colorscale,  # Light colors for home team
            showscale=False  # Do not show color scale in legend
        ),
        text=season_data['homeFinalScore'],
        textposition='outside',
        hovertemplate=(
            'Home Team: %{x}<br>' +
            'Score: %{y}<br>' +
            'Season: ' + str(season) + '<extra></extra>'
        ),
    ), row=i + 1, col=1)

    # Add visitor team scores with dark colors
    fig.add_trace(go.Bar(
        x=season_data['homeTeamAbbr'],
        y=season_data['visitorFinalScore'],
        name='Visitor Final Score',
        marker=dict(
            color=season_data['visitorFinalScore'],
            colorscale=visitor_colorscale,  # Darker colors for visitor team
            showscale=False  # Do not show color scale in legend
        ),
        text=season_data['visitorFinalScore'],
        textposition='outside',
        hovertemplate=(
            'Visitor Team: %{x}<br>' +
            'Score: %{y}<br>' +
            'Season: ' + str(season) + '<extra></extra>'
        ),
    ), row=i + 1, col=1)

    # Customize the layout for each subplot
    fig.update_yaxes(title_text='Scores', row=i + 1, col=1)
    fig.update_xaxes(title_text='Home Team', row=i + 1, col=1)

# Update layout for the entire figure
fig.update_layout(
    title='Highest Scoring Games by Season and Team',
    barmode='group',
    height=800 * len(highest_scoring_games['season'].unique()),  # Adjust height based on number of seasons
    width=1100,  # Increase width of the plot
    margin=dict(l=20, r=20, t=40, b=20),  # Adjust margins for better spacing
    showlegend=True,
)

# Add spacing between subplot titles and the bars
for i in range(len(highest_scoring_games['season'].unique())):
    fig['layout'][f'yaxis{i+1}'].title.font.size = 12  # Adjust title font size if needed
    fig['layout'][f'yaxis{i+1}'].title.standoff = 15  # Adjust the distance between the title and the y-axis

# Show the plot
fig.show()
```

<div style="background-color: #FFDDC1; padding: 20px; border-radius: 15px; border: 2px solid #8B0000; margin-top: 20px;">
    <p style="text-align: justify; font-family: 'Times New Roman', serif; font-size: 20px; color: #8B0000; font-weight: bold;">
        Observations of Home Vs Visitor Teams Scores:
    </p>
    <ul style="font-family: 'Georgia', serif; font-size: 14px; color: black; margin-left: 20px;">
        <li>According to this dataset, the highest Score of Games is associated with Home Team & Vistor Team such as <strong>Home Team Abbr ARI Visitor Team Abbr NO have the Highest Scores of about 42 & 34</strong></li>
        <li>While According to this dataset, the Lowest Score of Games is associated with Home Team & Vistor Team such as <strong>Home Team Abbr WAS Visitor Team Abbr JAX have the Highest Scores of about 28 & 22</strong></li>
            <li>Similarly The Teams such as<strong>ARI, PIT, NO </strong>have the lowest Win rate and lowest Win percentage of about <strong>20%, 33.3%</strong></li>
        <li>The Home Final Score is Highest than that of the Visitor Final Scores which is determined by Piechart </strong>This means that according to this Dataset there are more changes for the Home Teams to win the Games than that of the Visitor Teams as determined by the scores of Games<strong>The Home Teams Final Score is about 1012 and have the percenatage of about 57.8% while the visitor Team Final Score is about 740 and have the percentage of about 0.42%</strong></li>
        <li>It Means that the Home Teams Final Scores are more than that of the Visitor Teams Scores</strong>This means the Home Teams have more chances to win the games than that of the visitor Teams<strong></strong></li>
        <li>The Home Team Dal Have the Highest Games Score than that of the Other Home Teams and Visitor Teams</strong>The Home Team Dal have the highest Score os about 49<strong></strong></li>
        <li>While The Visitor Team DET Have the Highest Games Score than that of the Other Visitor Teams and Home Teams</strong>The Visitor Team DET have the highest Score os about 48<strong></strong>
        <li>The Visitor Team SF Have the Lowest Games Score than that of the Other Visitor Teams and Home Teams</strong>The Visitor Team SF have the Lowest Score os about 7.0<strong></strong></li>
        <li>While The Home Team NE Have the Lowest Games Score than that of the Other Home Teams and Visitor Teams</strong>The Home Team NE have the Lowest Score os about 0.0<strong></strong>
    </ul>
</div>

<div style="background-color: #FFDDC1; padding: 20px; border-radius: 15px; border: 2px solid #8B0000; margin-top: 20px;">
    <p style="text-align: justify; font-family: 'Times New Roman', serif; font-size: 20px; color: #8B0000; font-weight: bold;">
        Observations of Home Vs Visitor Teams Scores:
    </p>
    <div style="margin-bottom: 20px;">
        <h3 style="font-family: 'Georgia', serif; font-size: 18px; color: #8B0000; font-weight: bold;">Home Team Observations:</h3>
        <table style="width: 100%; font-family: 'Georgia', serif; font-size: 14px; border-collapse: collapse; margin-bottom: 20px;">
            <tr>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;"><strong>Highest Score of Home Team:</strong></td>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;">ARI (42)</td>
            </tr>
            <tr>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;"><strong>Lowest Score of Home Team:</strong></td>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;">WAS (28)</td>
            </tr>
            <tr>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;"><strong>Teams with Lowest Win Rate:</strong></td>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;">ARI, PIT, NO (20%, 33.3%)</td>
            </tr>
            <tr>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;"><strong>Highest Home Team Final Score:</strong></td>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;">DAL (49)</td>
            </tr>
            <tr>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;"><strong>Lowest Home Team Final Score:</strong></td>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;">NE (0)</td>
            </tr>
        </table>
    </div>
    <div style="margin-bottom: 20px;">
        <h3 style="font-family: 'Georgia', serif; font-size: 18px; color: #8B0000; font-weight: bold;">Visitor Team Observations:</h3>
        <table style="width: 100%; font-family: 'Georgia', serif; font-size: 14px; border-collapse: collapse; margin-bottom: 20px;">
            <tr>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;"><strong>Highest Score of Visitor Team:</strong></td>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;">NO (34)</td>
            </tr>
            <tr>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;"><strong>Lowest Score of Visitor Team:</strong></td>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;">JAX (22)</td>
            </tr>
            <tr>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;"><strong>Visitor Team with Highest Score:</strong></td>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;">DET (48)</td>
            </tr>
            <tr>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;"><strong>Visitor Team with Lowest Score:</strong></td>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;">SF (7)</td>
            </tr>
        </table>
    </div>
</div>

# <p id="explore-home-vs-visitor-team-average-scores" style="background-color: #FFDDC1; font-family: 'Palatino Linotype', 'Book Antiqua', Palatino, serif; color: #8a0808; font-size: 1.5em; text-align: center; border-radius: 35px 20px 35px 20px; padding: 15px 25px; line-height: 1.5; border: 4px double #8a0808; box-shadow: 5px 5px 20px rgba(0, 0, 0, 0.5), -5px -5px 20px rgba(0, 0, 0, 0.3); margin: 0 auto; width: auto;">Let's Explore the Home vs Visitor Team Average Scores</p>

```python
# Calculate average score for home teams
season_home_consistency = df.groupby(['season', 'homeTeamAbbr'])['homeFinalScore'].mean().reset_index(name='average_home_score')

# Calculate average score for visitor teams
season_visitor_consistency = df.groupby(['season', 'visitorTeamAbbr'])['visitorFinalScore'].mean().reset_index(name='average_visitor_score')

# Rename the visitor team abbreviation column for clarity
season_visitor_consistency.rename(columns={'visitorTeamAbbr': 'homeTeamAbbr'}, inplace=True)

# Merge both dataframes to include both home and visitor average scores
season_team_consistency = pd.merge(season_home_consistency, season_visitor_consistency, on=['season', 'homeTeamAbbr'], how='outer')

print("Score consistency (Average) per season and team:\n")
season_team_consistency
```

<div style="background-color: #FFDDC1; padding: 20px; border-radius: 15px; border: 2px solid #8B0000; margin-top: 20px;">
    <p style="text-align: justify; font-family: 'Times New Roman', serif; font-size: 20px; color: #8B0000; font-weight: bold;">
        Observations of Average Scores per season and team:
    </p>
    <ul style="font-family: 'Georgia', serif; font-size: 14px; color: black; margin-left: 20px;">
        <li>According to this dataset, the highest Average Score of Games is associated with Home Team & Vistor Team such as <strong>Home Team Abbr BUF & Visitor Team Abbr BAL have the Highest Scores</strong>The Highest Home Team Abbr Buf have the highest Abbr score in Season 2022 are of about 35.333333 & Visitor Team Abbr BAL have the Highest Average Scores of about 27.000000</strong></li>
        <li>According to this dataset, the Lowest Average Score of Games is associated with Home Team & Vistor Team such as <strong>Home Team Abbr DEN & Visitor Team Abbr BAL have the Highest Scores</strong>The Lowest Average Score of Home Team Abbr DEN in Season 2022 are of about 11.250000 & Visitor Team Abbr PIT have the Lowest Average Scores of about 13.200000</strong></li>
    </ul>
</div>

<div style="background-color: #FFDDC1; padding: 20px; border-radius: 15px; border: 2px solid #8B0000; margin-top: 20px;">
    <p style="text-align: justify; font-family: 'Times New Roman', serif; font-size: 20px; color: #8B0000; font-weight: bold;">
        Observations of Average Scores per Season and Team:
    </p> 
    <div style="margin-bottom: 20px;">
        <h3 style="font-family: 'Georgia', serif; font-size: 18px; color: #8B0000; font-weight: bold;">Highest Average Scores:</h3>
        <table style="width: 100%; font-family: 'Georgia', serif; font-size: 14px; border-collapse: collapse; margin-bottom: 20px;">
            <tr>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;"><strong>Highest Home Team Average Score:</strong></td>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;">BUF (35.33) in 2022</td>
            </tr>
            <tr>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;"><strong>Highest Visitor Team Average Score:</strong></td>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;">BAL (27.00) in 2022</td>
            </tr>
        </table>
    </div>
    <div style="margin-bottom: 20px;">
        <h3 style="font-family: 'Georgia', serif; font-size: 18px; color: #8B0000; font-weight: bold;">Lowest Average Scores:</h3>
        <table style="width: 100%; font-family: 'Georgia', serif; font-size: 14px; border-collapse: collapse; margin-bottom: 20px;">
            <tr>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;"><strong>Lowest Home Team Average Score:</strong></td>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;">DEN (11.25) in 2022</td>
            </tr>
            <tr>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;"><strong>Lowest Visitor Team Average Score:</strong></td>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;">PIT (13.20) in 2022</td>
            </tr>
        </table>
    </div>
</div>

# <p id="games-date-and-time-trend-analysis" style="background-color: #FFDDC1; font-family: 'Palatino Linotype', 'Book Antiqua', Palatino, serif; color: #8a0808; font-size: 1.5em; text-align: center; border-radius: 35px 20px 35px 20px; padding: 15px 25px; line-height: 1.5; border: 4px double #8a0808; box-shadow: 5px 5px 20px rgba(0, 0, 0, 0.5), -5px -5px 20px rgba(0, 0, 0, 0.3); margin: 0 auto; width: auto;">Games Date and Time Trend Analysis</p>

```python
# Convert 'gameDate' to datetime
df['gameDate'] = pd.to_datetime(df['gameDate'])

# Combine 'gameDate' and 'gameTimeEastern' into a single datetime column
df['gameDateTime'] = pd.to_datetime(df['gameDate'].dt.date.astype(str) + ' ' + df['gameTimeEastern'])

# Display the updated DataFrame structure
df[['gameDate', 'gameTimeEastern', 'gameDateTime']].head()
```

# <p id="games-trend-days-of-week" style="background-color: #FFDDC1; font-family: 'Palatino Linotype', 'Book Antiqua', Palatino, serif; color: #8a0808; font-size: 1.5em; text-align: center; border-radius: 35px 20px 35px 20px; padding: 15px 25px; line-height: 1.5; border: 4px double #8a0808; box-shadow: 5px 5px 20px rgba(0, 0, 0, 0.5), -5px -5px 20px rgba(0, 0, 0, 0.3); margin: 0 auto; width: auto;">Analyze the Games Trend Based on Days of the Week</p>

```python
# Add a column for the day of the week
df['day_of_week'] = df['gameDate'].dt.day_name()

# Count of games played on each day of the week
games_per_day = df['day_of_week'].value_counts().reset_index(name='game_count')
games_per_day.columns = ['day_of_week', 'game_count']

print("Number of games played per day of the week:\n")
games_per_day.sort_values(by='game_count', ascending=False)
```

# <p id="games-played-per-day-over-seasons" style="background-color: #FFDDC1; font-family: 'Palatino Linotype', 'Book Antiqua', Palatino, serif; color: #8a0808; font-size: 1.5em; text-align: center; border-radius: 35px 20px 35px 20px; padding: 15px 25px; line-height: 1.5; border: 4px double #8a0808; box-shadow: 5px 5px 20px rgba(0, 0, 0, 0.5), -5px -5px 20px rgba(0, 0, 0, 0.3); margin: 0 auto; width: auto;">Let's Visualize the Number of Games Played Per Day Over Seasons</p>

```python
# Create a column for weekday names
df['weekday'] = df['gameDate'].dt.day_name()

# Count games per weekday and season
games_per_weekday = df.groupby(['weekday', 'season']).size().reset_index(name='count')

# Sort by weekday for a meaningful display
weekdays_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
games_per_weekday['weekday'] = pd.Categorical(games_per_weekday['weekday'], categories=weekdays_order, ordered=True)
games_per_weekday = games_per_weekday.sort_values(by=['season', 'weekday'])

# Display the results
print("Number of Games Played per Weekday and Season:")
print(games_per_weekday.to_string(index=False))
```

```python
# Create a column for weekday names
df['weekday'] = df['gameDate'].dt.day_name()

# Count games per weekday and season
games_per_weekday = df.groupby(['weekday', 'season']).size().reset_index(name='count')

# Get the unique seasons
seasons = df['season'].unique()

# Create a complete DataFrame with all combinations of weekdays and seasons
weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
all_combinations = pd.MultiIndex.from_product([weekdays, seasons], names=['weekday', 'season']).to_frame(index=False)

# Merge with the games_per_weekday DataFrame and fill NaN with 0
complete_data = pd.merge(all_combinations, games_per_weekday, on=['weekday', 'season'], how='left').fillna(0)

# Convert 'count' to integer for display purposes
complete_data['count'] = complete_data['count'].astype(int)

# Create an animated bar plot with Plotly Express
fig1 = px.bar(complete_data,
              x='weekday',
              y='count',
              color='season',
              animation_frame='season',
              title='Number of Games Played per Weekday Over Seasons',
              labels={'count': 'Number of Games', 'weekday': 'Weekday'},
              text='count',
              category_orders={'weekday': weekdays})

# Update layout for better visualization
fig1.update_layout(xaxis_title='Weekday', yaxis_title='Number of Games')
fig1.update_traces(textposition='outside')
fig1.update_traces(textfont_size=12)
fig1.update_traces(texttemplate='%{text}')

# Display the plot
fig1.show()
```

# <p id="games-played-at-each-time" style="background-color: #FFDDC1; font-family: 'Palatino Linotype', 'Book Antiqua', Palatino, serif; color: #8a0808; font-size: 1.5em; text-align: center; border-radius: 35px 20px 35px 20px; padding: 15px 25px; line-height: 1.5; border: 4px double #8a0808; box-shadow: 5px 5px 20px rgba(0, 0, 0, 0.5), -5px -5px 20px rgba(0, 0, 0, 0.3); margin: 0 auto; width: auto;">Let's Analyze the Games Played at Each Time</p>

```python
# Count of games played at each time
games_per_time = df['gameTimeEastern'].value_counts().reset_index(name='game_count')
games_per_time.columns = ['gameTimeEastern', 'game_count']

print("Number of games played at each time:\n")
games_per_time.sort_values(by='game_count', ascending=False)
```

```python
# Count of games played at each time
games_per_time = df['gameTimeEastern'].value_counts().reset_index(name='game_count')
games_per_time.columns = ['gameTimeEastern', 'game_count']

# Sort values for proper display in the plot
games_per_time = games_per_time.sort_values(by='game_count', ascending=False)

# Generate Seaborn's "deep" color palette and map it to the unique game times
unique_game_times = games_per_time['gameTimeEastern'].unique()
colors = sns.color_palette("deep", len(unique_game_times)).as_hex()
color_map = dict(zip(unique_game_times, colors))

# Create a bar plot using Plotly Express
fig = px.bar(games_per_time,
             x='gameTimeEastern',
             y='game_count',
             title='Number of Games Played at Each Time',
             labels={'gameTimeEastern': 'Game Time (Eastern)', 'game_count': 'Number of Games'},
             text='game_count')  # Display counts above bars

# Apply the color map to each bar
fig.for_each_trace(lambda trace: trace.update(marker_color=[color_map[val] for val in games_per_time['gameTimeEastern']]))

# Update the legend to show colors corresponding to the bars
fig.update_layout(
    legend=dict(
        title="Game Time (Eastern)",
        itemsizing="constant"
    )
)

# Ensure the text above the bars is shown
fig.update_traces(textposition='outside', textfont_size=12)

# Update layout for better visualization
fig.update_layout(xaxis_title='Game Time (Eastern)', yaxis_title='Number of Games')

# Show the plot
fig.show()
```

<div style="background-color: #FFDDC1; padding: 20px; border-radius: 15px; border: 2px solid #8B0000; margin-top: 20px;">
    <p style="text-align: justify; font-family: 'Times New Roman', serif; font-size: 20px; color: #8B0000; font-weight: bold;">
        Games Trend Analysis on the Basis of Time Observations:
    </p>
    <ul style="font-family: 'Georgia', serif; font-size: 14px; color: black; margin-left: 20px;">
        <li>According to this dataset by analyzing most frequent Data it is Determined that most of the Games played in the year 2022 <strong>at the Estern Time 20:20:00</strong>Tand most of the played Games at the 2022-09-08 20:20:00 Date time Hence, it is determined that  most of the Big Bowl Games at the year 2022 and the 11th month of the year</strong></li>
        <li>By the Games Count analysis on the bais of the Days of Week it is determined that the most of the Games played at Sunday as on the Sunday there is highest count of games played which is about 117<strong></strong></li>
        <li>Then Most of the Games Played on Moday and Thursday as the Games frequency Counday on Monday is 10 while on Thursday the Games Frequency Count are of about 9<strong></strong></li>
        <li>According to this Dataset Most of the Games played at <strong>13:00:00 Eastern Time</strong>as the Games Count is Highest at this Time and have the count are of about 73</strong></li>
        <li>while According to this Dataset least of the Games played at <strong>19:15:00 & 20:30:00 Eastern Time</strong>as the Games Count is least at this Time and have the count are of about 1</strong></li>
    </ul>
</div>

<div style="background-color: #FFDDC1; padding: 20px; border-radius: 15px; border: 2px solid #8B0000; margin-top: 20px;">
    <p style="text-align: justify; font-family: 'Times New Roman', serif; font-size: 20px; color: #8B0000; font-weight: bold;">
        Games Trend Analysis on the Basis of Time Observations:
    </p> 
    <div style="margin-bottom: 20px;">
        <h3 style="font-family: 'Georgia', serif; font-size: 18px; color: #8B0000; font-weight: bold;">Key Findings:</h3>
        <table style="width: 100%; font-family: 'Georgia', serif; font-size: 14px; border-collapse: collapse; margin-bottom: 20px;">
            <tr>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;"><strong>Most Frequent Game Time in 2022:</strong></td>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;">20:20:00 Eastern Time</td>
            </tr>
            <tr>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;"><strong>Date with Highest Game Frequency in 2022:</strong></td>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;">2022-09-08 20:20:00</td>
            </tr>
            <tr>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;"><strong>Month with Most Big Bowl Games in 2022:</strong></td>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;">November</td>
            </tr>
            <tr>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;"><strong>Day with Highest Game Count:</strong></td>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;">Sunday (117 games)</td>
            </tr>
            <tr>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;"><strong>Next Most Frequent Game Days:</strong></td>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;">Monday (10 games), Thursday (9 games)</td>
            </tr>
            <tr>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;"><strong>Time with Highest Game Count:</strong></td>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;">13:00:00 Eastern Time (73 games)</td>
            </tr>
            <tr>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;"><strong>Times with Lowest Game Count:</strong></td>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;">19:15:00 & 20:30:00 Eastern Time (1 game each)</td>
            </tr>
        </table>
    </div>
</div>

# <p id="games-played-by-home-teams" style="background-color: #FFDDC1; font-family: 'Palatino Linotype', 'Book Antiqua', Palatino, serif; color: #8a0808; font-size: 1.5em; text-align: center; border-radius: 35px 20px 35px 20px; padding: 15px 25px; line-height: 1.5; border: 4px double #8a0808; box-shadow: 5px 5px 20px rgba(0, 0, 0, 0.5), -5px -5px 20px rgba(0, 0, 0, 0.3); margin: 0 auto; width: auto;">Let's Explore the Games Played by Home Teams</p>

```python
# Count of games played by home teams
home_team_counts = df['homeTeamAbbr'].value_counts().reset_index(name='game_count')
home_team_counts.columns = ['homeTeamAbbr', 'game_count']

print("Number of games played by each home team:\n")
home_team_counts.sort_values(by='game_count', ascending=False)
```

# <p id="games-played-by-visitor-teams" style="background-color: #FFDDC1; font-family: 'Palatino Linotype', 'Book Antiqua', Palatino, serif; color: #8a0808; font-size: 1.5em; text-align: center; border-radius: 35px 20px 35px 20px; padding: 15px 25px; line-height: 1.5; border: 4px double #8a0808; box-shadow: 5px 5px 20px rgba(0, 0, 0, 0.5), -5px -5px 20px rgba(0, 0, 0, 0.3); margin: 0 auto; width: auto;">Let's Explore the Games Played by Visitor Teams</p>

```python
# Count of games played by visitor teams
visitor_team_counts = df['visitorTeamAbbr'].value_counts().reset_index(name='game_count')
visitor_team_counts.columns = ['visitorTeamAbbr', 'game_count']

print("Number of games played by each visitor team:\n")
visitor_team_counts.sort_values(by='game_count', ascending=False)
```

# <p id="nfl-home-visitor-scores-analysis" style="background-color: #FFDDC1; font-family: 'Palatino Linotype', 'Book Antiqua', Palatino, serif; color: #8a0808; font-size: 1.5em; text-align: center; border-radius: 35px 20px 35px 20px; padding: 15px 25px; line-height: 1.5; border: 4px double #8a0808; box-shadow: 5px 5px 20px rgba(0, 0, 0, 0.5), -5px -5px 20px rgba(0, 0, 0, 0.3); margin: 0 auto; width: auto;">Let's Visualize the NFL Home & Visitor Scores Analysis</p>

```python
# Create the figure with subplots
fig = sp.make_subplots(rows=1, cols=2, subplot_titles=("Home Final Score vs Visitor Team Abbr", "Visitor Final Score vs Visitor Team Abbr"))

# First subplot: Home Final Score vs Visitor Team Abbr
home_scatter = px.scatter(
    df, 
    x='visitorTeamAbbr', 
    y='homeFinalScore', 
    color='homeTeamAbbr',
    title='Home Final Score vs Visitor Team Abbr',
    labels={'homeFinalScore': 'Home Final Score', 'visitorTeamAbbr': 'Visitor Team Abbreviation'}
)
for trace in home_scatter.data:
    fig.add_trace(trace, row=1, col=1)

# Second subplot: Visitor Final Score vs Visitor Team Abbr
visitor_scatter = px.scatter(
    df, 
    x='visitorTeamAbbr', 
    y='visitorFinalScore', 
    color='visitorTeamAbbr',
    title='Visitor Final Score vs Visitor Team Abbr',
    labels={'visitorFinalScore': 'Visitor Final Score', 'visitorTeamAbbr': 'Visitor Team Abbreviation'}
)
for trace in visitor_scatter.data:
    fig.add_trace(trace, row=1, col=2)

# Update layout and show the figure
fig.update_layout(title_text='NFL Game Scores Analysis', height=500, width=1200)
fig.show()
```

```python
# Create the figure with subplots
fig = sp.make_subplots(rows=1, cols=2, subplot_titles=("Home Final Score vs Home Team Abbr", "Visitor Final Score vs Home Team Abbr"))

# First subplot: Home Final Score vs Home Team Abbr
home_scatter = px.scatter(
    df, 
    x='homeTeamAbbr',  # Changed to homeTeamAbbr for the x-axis
    y='homeFinalScore', 
    color='homeTeamAbbr',  # Using homeTeamAbbr for color legend
    title='Home Final Score vs Home Team Abbr',
    labels={'homeFinalScore': 'Home Final Score', 'homeTeamAbbr': 'Home Team Abbreviation'}
)
for trace in home_scatter.data:
    fig.add_trace(trace, row=1, col=1)

# Second subplot: Visitor Final Score vs Home Team Abbr
visitor_scatter = px.scatter(
    df, 
    x='homeTeamAbbr',  # Changed to homeTeamAbbr for the x-axis
    y='visitorFinalScore', 
    color='homeTeamAbbr',  # Using homeTeamAbbr for color legend
    title='Visitor Final Score vs Home Team Abbr',
    labels={'visitorFinalScore': 'Visitor Final Score', 'homeTeamAbbr': 'Home Team Abbreviation'}
)
for trace in visitor_scatter.data:
    fig.add_trace(trace, row=1, col=2)

# Update layout and show the figure
fig.update_layout(title_text='NFL Game Scores Analysis', height=500, width=1200)
fig.show()
```

# <p id="nfl-average-final-score-home-visitor" style="background-color: #FFDDC1; font-family: 'Palatino Linotype', 'Book Antiqua', Palatino, serif; color: #8a0808; font-size: 1.5em; text-align: center; border-radius: 35px 20px 35px 20px; padding: 15px 25px; line-height: 1.5; border: 4px double #8a0808; box-shadow: 5px 5px 20px rgba(0, 0, 0, 0.5), -5px -5px 20px rgba(0, 0, 0, 0.3); margin: 0 auto; width: auto;">Let's Visualize the NFL Average Final Score of Home vs Visitor Teams Over Weeks</p>

```python
# Group by 'week' to calculate average scores
avg_scores = df.groupby('week')[['homeFinalScore', 'visitorFinalScore']].mean().reset_index()

# Create the line plot
plt.figure(figsize=(10, 6))
sns.lineplot(data=avg_scores, x='week', y='homeFinalScore', label='Home Final Score', marker='o')
sns.lineplot(data=avg_scores, x='week', y='visitorFinalScore', label='Visitor Final Score', marker='o')

# Add title and labels
plt.title('Average Final Scores of Home vs. Visitor Teams Over Weeks')
plt.xlabel('Week')
plt.ylabel('Average Final Score')
plt.legend()
plt.xticks(rotation=45)

# Annotate the points with their values, using smaller vertical offsets
for index, row in avg_scores.iterrows():
    # Small offset for home scores
    plt.text(row['week'], row['homeFinalScore'] + 0.1, f"{row['homeFinalScore']:.1f}", 
             color='black', ha='center', va='bottom', fontsize=8)
    
    # Small offset for visitor scores
    plt.text(row['week'], row['visitorFinalScore'] - 0.1, f"{row['visitorFinalScore']:.1f}", 
             color='black', ha='center', va='top', fontsize=8)

# Show the plot
plt.show()
```

<div style="background-color: #FFDDC1; padding: 20px; border-radius: 15px; border: 2px solid #8B0000; margin-top: 20px;">
    <p style="text-align: justify; font-family: 'Times New Roman', serif; font-size: 20px; color: #8B0000; font-weight: bold;">
        Observations of Home vs Vistor Teams on the basis of Team Abbr & Score:
    </p>
    <ul style="font-family: 'Georgia', serif; font-size: 14px; color: black; margin-left: 20px;">
        <li>According to this dataset NO have the Highest Frequency Count of the Home Team Abbr<strong> and have the maximum Frequency Count are of about </strong> 6</strong></li>
        <li>According to this dataset PIT have the Least Frequency Count of the Home Team Abbr<strong> and have the minimum Frequency Count are of about </strong> 3</strong></li>
        <li> while Those Games which have neither too high Frquency Count and Nor too low of the Home Abbr includes<strong> KC, HOU, SEA, LAC, BAL, CIN, CHI, PHI, NYG, DEN, GB, IND, NE, MIA, MIN and they have the count are of about </strong> 4</strong></li>
        <li>Those Games which have Highest Frquency Count of the Visitor Team Abbr includes<strong> BUF, CHI, BAL, SF, PIT, IND, NE, TEN, LV, GB, CIN, SEA and they have the count are of about </strong> 5</strong></li>
        <li>Those Games which have Lowest Frquency Count of the Visitor Team Abbr includes<strong> NO, CLE, DET, LA, DAL and they have the count are of about </strong>3</strong></li>
        <li>while Those Games which have neither too high Frquency Count and Nor too low of the Visitor Abbr includes <strong> NYJ, PHI, JAX, KC, NYG, TB, DEN, ATL, LAC, CAR, MIN, HOU, ARI & WAS and they have the count are of about </strong>4</strong></li>
        <li>According to this Dataset the Highest Final Score of the Home Team & the Visitor Team Abbr<strong> includes the Home Team Dal and Visitor Team CHI and have the Home average Final Score are of about </strong>46</strong></li>
        <li>while According to this Dataset the Highest Final Score of the Visitor Team Abbr<strong> includes the Visitor Team SEA and have the Visitor average Final Score are of about </strong>48</strong</li>
        <li>And According to this Dataset the Highest Final Score of the Home Team Abbr<strong> includes the Home Team DAL and have the Home Highest Final Score are of about </strong>49</strong</li>
        <li>And According to this Dataset the Highest Final Score of the Home Team Abbr<strong> includes the Home Team DAL and have the Highest average visitor Finsl Score are of about </strong>48</strong</li>
        <li>The Highest Average Final Score of Home Team is at the week <strong>8th and have the average score of about </strong>25.9</strong</li> 
        <li>The Lowest Average Final Score of Home Team is at the week <strong>3rd and have the average score of about </strong>18.6</strong</li> 
        <li>The Highest Average Final Score of Visitor Team is at the week <strong>4th and have the average score of about </strong>23.9</strong</li> 
        <li>The Lowest Average Final Score of Visitor Team is at the week <strong>2nd and have the average score of about </strong>18.1</strong</li> 
    </ul>
</div>

<div style="background-color: #FFDDC1; padding: 20px; border-radius: 15px; border: 2px solid #8B0000; margin-top: 20px;">
    <p style="text-align: justify; font-family: 'Times New Roman', serif; font-size: 20px; color: #8B0000; font-weight: bold;">
        Observations of Home vs Visitor Teams on the Basis of Team Abbr & Score:
    </p> 
    <div style="margin-bottom: 20px;">
        <h3 style="font-family: 'Georgia', serif; font-size: 18px; color: #8B0000; font-weight: bold;">Home Team Analysis:</h3>
        <table style="width: 100%; font-family: 'Georgia', serif; font-size: 14px; border-collapse: collapse; margin-bottom: 20px;">
            <tr>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;"><strong>Highest Frequency Count of Home Team Abbr:</strong></td>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;">NO (6 games)</td>
            </tr>
            <tr>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;"><strong>Lowest Frequency Count of Home Team Abbr:</strong></td>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;">PIT (3 games)</td>
            </tr>
            <tr>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;"><strong>Teams with Moderate Frequency Count:</strong></td>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;">KC, HOU, SEA, LAC, BAL, CIN, CHI, PHI, NYG, DEN, GB, IND, NE, MIA, MIN (4 games each)</td>
            </tr>
            <tr>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;"><strong>Highest Home Team Final Score:</strong></td>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;">DAL (49 points)</td>
            </tr>
            <tr>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;"><strong>Highest Home Team Average Final Score:</strong></td>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;">DAL (46 points)</td>
            </tr>
            <tr>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;"><strong>Highest Average Final Score by Week:</strong></td>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;">Week 8 (25.9 points)</td>
            </tr>
            <tr>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;"><strong>Lowest Average Final Score by Week:</strong></td>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;">Week 3 (18.6 points)</td>
            </tr>
        </table>
    </div>
    <div style="margin-bottom: 20px;">
        <h3 style="font-family: 'Georgia', serif; font-size: 18px; color: #8B0000; font-weight: bold;">Visitor Team Analysis:</h3>
        <table style="width: 100%; font-family: 'Georgia', serif; font-size: 14px; border-collapse: collapse; margin-bottom: 20px;">
            <tr>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;"><strong>Highest Frequency Count of Visitor Team Abbr:</strong></td>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;">BUF, CHI, BAL, SF, PIT, IND, NE, TEN, LV, GB, CIN, SEA (5 games each)</td>
            </tr>
            <tr>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;"><strong>Lowest Frequency Count of Visitor Team Abbr:</strong></td>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;">NO, CLE, DET, LA, DAL (3 games each)</td>
            </tr>
            <tr>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;"><strong>Teams with Moderate Frequency Count:</strong></td>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;">NYJ, PHI, JAX, KC, NYG, TB, DEN, ATL, LAC, CAR, MIN, HOU, ARI, WAS (4 games each)</td>
            </tr>
            <tr>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;"><strong>Highest Visitor Team Final Score:</strong></td>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;">SEA (48 points)</td>
            </tr>
            <tr>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;"><strong>Highest Average Visitor Final Score:</strong></td>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;">SEA (48 points)</td>
            </tr>
            <tr>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;"><strong>Highest Average Final Score by Week:</strong></td>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;">Week 4 (23.9 points)</td>
            </tr>
            <tr>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;"><strong>Lowest Average Final Score by Week:</strong></td>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;">Week 2 (18.1 points)</td>
            </tr>
        </table>
    </div>
</div>

# <p id="home-visitor-final-scores-distribution" style="background-color: #FFDDC1; font-family: 'Palatino Linotype', 'Book Antiqua', Palatino, serif; color: #8a0808; font-size: 1.5em; text-align: center; border-radius: 35px 20px 35px 20px; padding: 15px 25px; line-height: 1.5; border: 4px double #8a0808; box-shadow: 5px 5px 20px rgba(0, 0, 0, 0.5), -5px -5px 20px rgba(0, 0, 0, 0.3); margin: 0 auto; width: auto;">Let's Visualize the Frequency Distribution of Home vs Visitor Teams Final Scores</p>

```python
# Convert 'gameDate' to datetime
df['gameDate'] = pd.to_datetime(df['gameDate'])

# Define the pinkish color palette with valid color formats
color_palette = [
    (255/255, 105/255, 97/255, 0.8),  # Light Pink (RGBA tuple)
    (255/255, 105/255, 180/255, 0.8),  # Hot Pink (RGBA tuple)
    (219/255, 112/255, 147/255, 0.8),  # Pale Violet Red (RGBA tuple)
    (216/255, 191/255, 216/255, 0.8),  # Thistle (RGBA tuple)
    (255/255, 20/255, 147/255, 0.8),   # Deep Pink (RGBA tuple)
    (199/255, 21/255, 133/255, 0.8),   # Medium Violet Red (RGBA tuple)
    (248/255, 187/255, 208/255, 0.8),  # Cotton Candy (RGBA tuple)
]

# Cell 1: Home and Visitor Team Final Scores
plt.figure(figsize=(12, 6))

# Home Final Score Distribution
plt.subplot(1, 2, 1)
sns.histplot(df['homeFinalScore'], bins=10, kde=True, color=color_palette[0])  # Apply the first color in the palette
plt.title('Home Final Score Distribution')
plt.xlabel('Home Final Score')
plt.ylabel('Frequency')

# Visitor Final Score Distribution
plt.subplot(1, 2, 2)
sns.histplot(df['visitorFinalScore'], bins=10, kde=True, color=color_palette[1])  # Apply the second color in the palette
plt.title('Visitor Final Score Distribution')
plt.xlabel('Visitor Final Score')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()
```

# <p id="home-vs-visitor-scores-trend" style="background-color: #FFDDC1; font-family: 'Palatino Linotype', 'Book Antiqua', Palatino, serif; color: #8a0808; font-size: 1.5em; text-align: center; border-radius: 35px 20px 35px 20px; padding: 15px 25px; line-height: 1.5; border: 4px double #8a0808; box-shadow: 5px 5px 20px rgba(0, 0, 0, 0.5), -5px -5px 20px rgba(0, 0, 0, 0.3); margin: 0 auto; width: auto;">Home vs Visitor Teams Final Scores Trend Analysis</p>

```python
# Convert gameDate to datetime format
df['gameDate'] = pd.to_datetime(df['gameDate'])

# Extract weekday names
df['weekday'] = df['gameDate'].dt.day_name()

# Binning days of the week into 1st Half and 2nd Half based on weekday names
df['week_phase'] = df['weekday'].apply(lambda x: '1st Half' if x in ['Monday', 'Tuesday', 'Wednesday'] else '2nd Half')

# Display the first few rows of the modified dataframe
display(df.head())
```

# <p id="games-played-per-week-over-seasons" style="background-color: #FFDDC1; font-family: 'Palatino Linotype', 'Book Antiqua', Palatino, serif; color: #8a0808; font-size: 1.5em; text-align: center; border-radius: 35px 20px 35px 20px; padding: 15px 25px; line-height: 1.5; border: 4px double #8a0808; box-shadow: 5px 5px 20px rgba(0, 0, 0, 0.5), -5px -5px 20px rgba(0, 0, 0, 0.3); margin: 0 auto; width: auto;">Let's Visualize the Number of Games Played Per Week Over Seasons</p>

```python
# Count games by week
games_per_week = df.groupby(['week', 'season']).size().reset_index(name='count')

# Create an animated bar plot with values above the bars
fig2 = px.bar(games_per_week,
              x='week',
              y='count',
              color='season',
              animation_frame='season',
              title='Number of Games Played per Week Over Seasons',
              labels={'count': 'Number of Games', 'week': 'Week'},
              text='count')  # Add text parameter to show counts above bars

# Update layout for axis titles and text position
fig2.update_layout(xaxis_title='Week', yaxis_title='Number of Games')
fig2.update_traces(textposition='outside')  # Position text above the bars

fig2.show()
```

# <p id="sunburst-plot-games-per-day-over-seasons" style="background-color: #FFDDC1; font-family: 'Palatino Linotype', 'Book Antiqua', Palatino, serif; color: #8a0808; font-size: 1.5em; text-align: center; border-radius: 35px 20px 35px 20px; padding: 15px 25px; line-height: 1.5; border: 4px double #8a0808; box-shadow: 5px 5px 20px rgba(0, 0, 0, 0.5), -5px -5px 20px rgba(0, 0, 0, 0.3); margin: 0 auto; width: auto;">Let's Visualize the Sunburst Plot of Number of Games Played Per Day Over Seasons</p>

```python
# Convert gameDate to datetime
df['gameDate'] = pd.to_datetime(df['gameDate'])

# Extract weekday names
df['weekday'] = df['gameDate'].dt.day_name()

# Create a combined dataframe for home and visitor teams
home_games = df[['weekday', 'season', 'homeTeamAbbr']].rename(columns={'homeTeamAbbr': 'team'})
visitor_games = df[['weekday', 'season', 'visitorTeamAbbr']].rename(columns={'visitorTeamAbbr': 'team'})
combined_games = pd.concat([home_games, visitor_games], ignore_index=True)

# Count games per team by weekday
games_per_team_weekday = combined_games.groupby(['weekday', 'season', 'team']).size().reset_index(name='count')

# Define a pink-to-red gradient color palette
color_palette = [
    'rgba(255, 105, 97, 0.8)',  # Light Pink
    'rgba(255, 105, 180, 0.8)',  # Hot Pink
    'rgba(219, 112, 147, 0.8)',  # Pale Violet Red
    'rgba(216, 191, 216, 0.8)',  # Thistle
    'rgba(255, 20, 147, 0.8)',   # Deep Pink
    'rgba(199, 21, 133, 0.8)',   # Medium Violet Red
    'rgba(248, 187, 208, 0.8)',  # Cotton Candy
]

# Create a Sunburst Chart with the specified color palette
fig_sunburst = px.sunburst(
    games_per_team_weekday,
    path=['season', 'weekday', 'team'],  # Define the hierarchy
    values='count',
    title='Games Played by Team per Weekday Over Seasons',
    labels={'count': 'Number of Games', 'weekday': 'Weekday', 'season': 'Season'},
    color='count',  # Color by the count of games
    color_continuous_scale='reds',  # Apply the pink-to-red gradient using Plotly's reds scale
)

# Show the Sunburst chart
fig_sunburst.show()
```

<div style="background-color: #FFDDC1; padding: 20px; border-radius: 15px; border: 2px solid #8B0000; margin-top: 20px;">
    <p style="text-align: justify; font-family: 'Times New Roman', serif; font-size: 20px; color: #8B0000; font-weight: bold;">
        Observations of Games Played Over Time:
    </p>
    <ul style="font-family: 'Georgia', serif; font-size: 14px; color: black; margin-left: 20px;">
        <li>By Visualizing the Distribution of Home Team and the Visitor Team Score it is determed that the Score Distribution is normal for both the Teams throughout the Season as they show the Normal Distribution/Gaussian Distribution<strong></strong></li>
        <li>By analyzing this Dataset it is determined that Most of the Games played in 2nd Half of the weeks<strong>as Most of the Games played on the Sunday</strong></strong></li>
        <li>Hence it is also being Visualized that most of the Games played in the early season and in the year <strong>2022</strong></strong></li>
        <li>According to this Dataset the Highest Total Score in the year<strong> 2022 </strong> is of the Home Team Team Abbr ATL and Visitor Team Abbr NO and have the highest Total Score are of about 58 played in Early Season, 9th Month and the visitor Team wins the Game</strong></li>
        <li>Most of the Games played in the Season 2022 in the First Week<strong>and the Games Count are of about </strong> 16</strong></li>
        <<li>Least of the Games played in the Season 2022 in the 9th Week<strong>and the Games Count are of about </strong> 13</strong></li>
    </ul>
</div>

<div style="background-color: #FFDDC1; padding: 20px; border-radius: 15px; border: 2px solid #8B0000; margin-top: 20px;">
    <p style="text-align: justify; font-family: 'Times New Roman', serif; font-size: 20px; color: #8B0000; font-weight: bold;">
        Observations of Games Played Over Time:
    </p>
    <div style="margin-bottom: 20px;">
        <h3 style="font-family: 'Georgia', serif; font-size: 18px; color: #8B0000; font-weight: bold;">Key Insights:</h3>
        <table style="width: 100%; font-family: 'Georgia', serif; font-size: 14px; border-collapse: collapse; margin-bottom: 20px;">
            <tr>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;"><strong>Score Distribution:</strong></td>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;">Normal/Gaussian distribution observed for both Home and Visitor teams throughout the season</td>
            </tr>
            <tr>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;"><strong>Games Played in Week:</strong></td>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;">Most games were played in the second half of the week, particularly on Sunday</td>
            </tr>
            <tr>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;"><strong>Season Highlights:</strong></td>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;">Most games were played in the early season of 2022</td>
            </tr>
            <tr>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;"><strong>Highest Total Score Game:</strong></td>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;">Home Team ATL vs Visitor Team NO, total score of 58 in the 9th month, won by the visitor team</td>
            </tr>
            <tr>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;"><strong>Most Games in a Week:</strong></td>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;">Week 1 of the 2022 season (16 games)</td>
            </tr>
            <tr>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;"><strong>Least Games in a Week:</strong></td>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;">Week 9 of the 2022 season (13 games)</td>
            </tr>
        </table>
    </div>
</div>

# <p id="scores-distribution-by-week" style="background-color: #FFDDC1; font-family: 'Palatino Linotype', 'Book Antiqua', Palatino, serif; color: #8a0808; font-size: 1.5em; text-align: center; border-radius: 35px 20px 35px 20px; padding: 15px 25px; line-height: 1.5; border: 4px double #8a0808; box-shadow: 5px 5px 20px rgba(0, 0, 0, 0.5), -5px -5px 20px rgba(0, 0, 0, 0.3); margin: 0 auto; width: auto;">Let's Visualize the Scores Distribution by Week</p>

```python
# Convert gameDate to datetime
df['gameDate'] = pd.to_datetime(df['gameDate'])

# Create new features
df['year'] = df['gameDate'].dt.year
df['month'] = df['gameDate'].dt.month
df['day'] = df['gameDate'].dt.day

# Bin the weeks into categories
bins = [0, 4, 8, 12, 17]
labels = ['Early Season', 'Mid Season', 'Late Season', 'Playoffs']
df['SeasonStage'] = pd.cut(df['week'], bins=bins, labels=labels, right=False)

# Box plot of homeFinalScore and visitorFinalScore by week
base = alt.Chart(df).transform_fold(
    ['homeFinalScore', 'visitorFinalScore'],
    as_=['Score Type', 'Score']
)

# Define a custom color scale based on your pinkish shades
color_scale = alt.Scale(
    domain=['homeFinalScore', 'visitorFinalScore'],
    range=['rgba(255, 105, 97, 0.8)', 'rgba(255, 105, 180, 0.8)']  # Example of two shades
)

boxplot = base.mark_boxplot().encode(
    x=alt.X('week:O', title='Week'),
    y=alt.Y('Score:Q', title='Score'),
    color=alt.Color('Score Type:N', scale=color_scale)
).properties(
    title='Score Distribution by Week',
    width=1000,  # Set a larger width
    height=600    # Set a larger height
)

# Calculate maximum scores for each week
max_scores = df.groupby(['week'])[['homeFinalScore', 'visitorFinalScore']].max().reset_index()
max_scores = max_scores.melt(id_vars='week', var_name='Score Type', value_name='Max Score')

# Add maximum score values to the plot
max_text = alt.Chart(max_scores).mark_text(
    align='center',
    baseline='bottom',
    dy=-5  # Position the text slightly above the box plot
).encode(
    x='week:O',
    y='Max Score:Q',
    text='Max Score:Q',
    color=alt.Color('Score Type:N', scale=color_scale)  # Use the same color scale
)

# Combine box plot and maximum score text
final_plot = boxplot + max_text
final_plot.show()
```

# <p id="visualize-scores-heatmap" style="background-color: #FFDDC1; font-family: 'Palatino Linotype', 'Book Antiqua', Palatino, serif; color: #8a0808; font-size: 1.5em; text-align: center; border-radius: 35px 20px 35px 20px; padding: 15px 25px; line-height: 1.5; border: 4px double #8a0808; box-shadow: 5px 5px 20px rgba(0, 0, 0, 0.5), -5px -5px 20px rgba(0, 0, 0, 0.3); margin: 0 auto; width: auto;">Lets Visualize the Scores HeatMap</p>

```python
# Feature Engineering
# Convert gameDate to datetime
df['gameDate'] = pd.to_datetime(df['gameDate'])

# Create new features
df['year'] = df['gameDate'].dt.year
df['month'] = df['gameDate'].dt.month
df['day'] = df['gameDate'].dt.day

# Bin the weeks into categories
bins = [0, 4, 8, 12, 17]
labels = ['Early Season', 'Mid Season', 'Late Season', 'Playoffs']
df['SeasonStage'] = pd.cut(df['week'], bins=bins, labels=labels, right=False)

# Create the totalScore column
df['totalScore'] = df['homeFinalScore'] + df['visitorFinalScore']

# Step 4: Prepare data for heatmap
score_matrix = df.pivot_table(index='homeTeamAbbr', columns='visitorTeamAbbr', values='totalScore', aggfunc='mean')

# Create a more attractive Heatmap using Seaborn with reddish tone
plt.figure(figsize=(14, 10))
heatmap = sns.heatmap(score_matrix, 
                      cmap='Reds', 
                      annot=True, 
                      fmt='.1f', 
                      linewidths=.5, 
                      linecolor='black', 
                      cbar_kws={"shrink": .9},
                      annot_kws={"size": 9})  # Adjust size of the annotation text

# Aesthetics improvements
plt.title('Average Total Scores Heatmap', fontsize=20)
plt.xlabel('Visitor Team', fontsize=16)
plt.ylabel('Home Team', fontsize=16)

# Set tick parameters for better readability
heatmap.tick_params(labelsize=12)
plt.xticks(rotation=45, ha='right')  # Rotate x labels for better visibility
plt.yticks(rotation=0)  # Keep y labels horizontal

# Show the plot
plt.tight_layout()
plt.show()
```

# <p id="scores-treemap" style="background-color: #FFDDC1; font-family: 'Palatino Linotype', 'Book Antiqua', Palatino, serif; color: #8a0808; font-size: 1.5em; text-align: center; border-radius: 35px 20px 35px 20px; padding: 15px 25px; line-height: 1.5; border: 4px double #8a0808; box-shadow: 5px 5px 20px rgba(0, 0, 0, 0.5), -5px -5px 20px rgba(0, 0, 0, 0.3); margin: 0 auto; width: auto;">Let's Visualize the Scores TreeMap</p>

```python
# Count the number of games per home team
home_team_counts = df.groupby('homeTeamAbbr').size().reset_index(name='numGames')
home_team_counts['type'] = 'Home'

# Count the number of games per visitor team
visitor_team_counts = df.groupby('visitorTeamAbbr').size().reset_index(name='numGames')
visitor_team_counts['type'] = 'Visitor'

# Rename 'visitorTeamAbbr' to 'homeTeamAbbr' in visitor_team_counts to align with home team column
visitor_team_counts = visitor_team_counts.rename(columns={'visitorTeamAbbr': 'homeTeamAbbr'})

# Combine both home and visitor counts
combined_counts = pd.concat([home_team_counts, visitor_team_counts], ignore_index=True)

# Create a static treemap of home and visitor teams with a red color scale
fig4 = px.treemap(combined_counts, 
                   path=['type', 'homeTeamAbbr'], 
                   values='numGames', 
                   title='Game Count Treemap by Team',
                   color='numGames',  # Color by the number of games
                   color_continuous_scale='reds')  # Use a red color scale

# Show the treemap
fig4.show()
```

<div style="background-color: #FFDDC1; padding: 20px; border-radius: 15px; border: 2px solid #8B0000; margin-top: 20px;">
    <p style="text-align: justify; font-family: 'Times New Roman', serif; font-size: 20px; color: #8B0000; font-weight: bold;">
        Observations of Games based on Score:
    </p>
    <ul style="font-family: 'Georgia', serif; font-size: 14px; color: black; margin-left: 20px;">
        <li>By Visualizing the HeatMap and TreeMap it is determined that <strong> the Home Team SEA and the Visitor Team DET have the positive correlation among them it means that they are highly correalted with each other as they have the correlation value of about 93.0</strong></li>
        <li>Similarly the Home Team ARI and Visitor Team No<strong>have the highest correlation among them as they are positively correlated and they have the correlation value of about 76.0</strong></strong></li>
        <li>Similarly the Home Team DET and Visitor Team LA<strong>have the highest correlation among them as they are positively correlated and they have the correlation value of about 73.0</strong></strong></li>
        <li>Similarly the Home Team CHI and Visitor Team CHI<strong>have the low but positive correaltion present among them and they have the correlation value of about 19.0</strong></strong></li>
        <li>The Darker Colors in HeatMap shows that there is High Correlation present among these column while the ligh colors of Heat Map shows that there is low correaltion present among the columns<strong></strong></strong></li>
    </ul>
</div>

<div style="background-color: #FFDDC1; padding: 20px; border-radius: 15px; border: 2px solid #8B0000; margin-top: 20px;">
    <p style="text-align: justify; font-family: 'Times New Roman', serif; font-size: 20px; color: #8B0000; font-weight: bold;">
        Observations of Games based on Score:
    </p>
    <div style="margin-bottom: 20px;">
        <h3 style="font-family: 'Georgia', serif; font-size: 18px; color: #8B0000; font-weight: bold;">Key Insights:</h3>
        <table style="width: 100%; font-family: 'Georgia', serif; font-size: 14px; border-collapse: collapse; margin-bottom: 20px;">
            <tr>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;"><strong>Highest Correlation (Positive):</strong></td>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;">Home Team SEA and Visitor Team DET (Correlation Value: 93.0)</td>
            </tr>
            <tr>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;"><strong>Second Highest Correlation:</strong></td>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;">Home Team ARI and Visitor Team NO (Correlation Value: 76.0)</td>
            </tr>
            <tr>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;"><strong>Third Highest Correlation:</strong></td>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;">Home Team DET and Visitor Team LA (Correlation Value: 73.0)</td>
            </tr>
            <tr>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;"><strong>Low Positive Correlation:</strong></td>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;">Home Team CHI and Visitor Team CHI (Correlation Value: 19.0)</td>
            </tr>
            <tr>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;"><strong>Heatmap Insight:</strong></td>
                <td style="border: 1px solid #8B0000; padding: 8px; text-align: left;">Darker colors in HeatMap represent higher correlation, while lighter colors indicate lower correlation</td>
            </tr>
        </table>
    </div>
</div>

# <p id="visualize-numeric-columns-distribution" style="background-color: #FFDDC1; font-family: 'Palatino Linotype', 'Book Antiqua', Palatino, serif; color: #8a0808; font-size: 1.5em; text-align: center; border-radius: 35px 20px 35px 20px; padding: 15px 25px; line-height: 1.5; border: 4px double #8a0808; box-shadow: 5px 5px 20px rgba(0, 0, 0, 0.5), -5px -5px 20px rgba(0, 0, 0, 0.3); margin: 0 auto; width: auto;">Let's Visualize the Distribution of Numeric Columns</p>

```python
# Convert gameDate to datetime
df['gameDate'] = pd.to_datetime(df['gameDate'])
# Display the numeric columns
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

# Exclude the 'season' column from the numeric columns
numeric_cols = numeric_cols[numeric_cols != 'season']

# Step 4: Create KDE Plots for Numeric Columns
# Set the aesthetics for the plots
sns.set(style="whitegrid")

# Define a list of red and pink shades as RGB tuples
color_palette = [
    'rgba(255, 105, 97, 0.8)',  # Light Pink
    'rgba(255, 105, 180, 0.8)',  # Hot Pink
    'rgba(219, 112, 147, 0.8)',  # Pale Violet Red
    'rgba(216, 191, 216, 0.8)',  # Thistle
    'rgba(255, 20, 147, 0.8)',   # Deep Pink
    'rgba(199, 21, 133, 0.8)',   # Medium Violet Red
    'rgba(248, 187, 208, 0.8)',  # Cotton Candy
]

# Ensure the number of colors matches the number of numeric columns
colors = colors[:len(numeric_cols)]

# Create KDE plots for each numeric column
for col, color in zip(numeric_cols, colors):
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data=df, x=col, fill=True, alpha=0.6, color=color)
    plt.title(f'KDE Plot of {col}')
    plt.xlabel(col)
    plt.ylabel('Density')
    plt.grid()
    plt.show()
```

# <p style="background-color: #FFDDC1; font-family: 'Palatino Linotype', 'Book Antiqua', Palatino, serif; color: #8a0808; font-size: 1.5em; text-align: center; border-radius: 35px 20px 35px 20px; padding: 15px 25px; line-height: 1.5; border: 4px double #8a0808; box-shadow: 5px 5px 20px rgba(0, 0, 0, 0.5), -5px -5px 20px rgba(0, 0, 0, 0.3); margin: 0 auto; width: auto;">Thanks For Exploring my Notebook. I'll Analyze and Visualize players play, Players, Plays & Tracking Weeks csv Files</p>