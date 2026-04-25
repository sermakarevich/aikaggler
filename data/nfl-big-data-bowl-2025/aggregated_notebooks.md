# nfl-big-data-bowl-2025: top public notebooks

The community's top-voted notebooks are overwhelmingly focused on comprehensive exploratory data analysis (EDA) of the NFL Big Data Bowl 2025 dataset, with heavy emphasis on cleaning messy tabular data, engineering demographic and game-level features, and visualizing spatial-temporal tracking patterns. A few notable entries diverge to provide reusable visualization utilities or demonstrate deep learning approaches, specifically applying transformer architectures to spatio-temporal sequences for formation prediction. Collectively, these works establish foundational statistical baselines and data preprocessing pipelines rather than focusing on complex ensemble modeling or advanced loss functions.

## Common purposes
- eda
- utility
- tutorial

## Competition flows
- Loads NFL Big Data Bowl CSV datasets, merges them by game/play/player identifiers, and passes them to a Plotly-based function that generates an interactive, frame-by-frame animated visualization of a specific play.
- Loads raw NFL competition CSVs, performs extensive statistical and spatial EDA with visualizations, and documents dataset characteristics without training any models.
- Loads the player play CSV, aggregates play-level metrics into game-level statistics, and generates distribution histograms and a 3D scatter plot to visualize offensive and defensive performance trends.
- Loads the players.csv dataset, performs demographic and physical feature engineering, and generates interactive visualizations to summarize player characteristics without training any predictive models.
- Loads the NFL Big Data Bowl 2025 game and player play datasets, performs extensive statistical exploration and visualization, and outlines a workflow for identifying key metrics and trends for potential predictive modeling.
- Loads the games.csv dataset, performs extensive statistical and temporal analysis, and generates visualizations to characterize scoring patterns, team performance, and home-field advantages without building any predictive models.
- Loads the players.csv file, cleans and transforms player attributes like height and birth date, and generates statistical summaries and visualizations to characterize the physical and demographic profile of NFL players.
- Loads the players CSV, engineers physical and demographic features, and generates statistical visualizations and clustering results to characterize player demographics and physical profiles.
- Loads preprocessed NFL player tracking data via custom utilities, trains a custom transformer model to predict offensive formations, and evaluates accuracy with confusion matrices and animated spatial visualizations.
- Loads and merges four NFL CSV datasets, cleans missing values, computes aggregated offensive/defensive statistics, and generates comprehensive team and temporal visualizations to identify performance trends.

## Data reading
- Reads CSV files (games.csv, plays.csv, players.csv, tracking_week_1.csv) using pandas.read_csv.
- pd.read_csv() for player_play.csv, games.csv, plays.csv, players.csv, and tracking_week_1.csv.
- pd.read_csv('/kaggle/input/nfl-big-data-bowl-2025/player_play.csv')
- pd.read_csv("/kaggle/input/nfl-big-data-bowl-2025/players.csv")
- pd.read_csv('/kaggle/input/nfl-big-data-bowl-2025/games.csv')
- Reads plays.csv for target distribution and unique class counts.
- Loads train/val/test splits via process_datasets.load_datasets() utility.
- Reads parquet files for tracking metadata and preprocessed features.

## Data processing
- Merges tracking and player datasets on nflId and displayName.
- Computes a weighted Euclidean distance between team hex colors to dynamically swap primary/secondary colors when hues are too similar.
- Splits long play descriptions into two lines for layout constraints.
- Applies a fixed scaling factor (2.23693629205) to convert tracking speed units to MPH.
- Date parsing with pd.to_datetime(format='mixed') and DatetimeIndex extraction.
- Filtering tracking data by playId, gameId, and playDirection.
- Aggregation via pivot_table and value_counts for categorical distributions.
- Drops columns containing more than 80% null values.
- Imputes missing birthDate values using median and mode
- Converts height from 'ft-in' string format to numeric inches
- Converts weight to float
- Calculates BMI using metric conversions
- Handles missing values in derived features via backfill and mean imputation
- Converts gameDate to datetime
- Extracts dayOfWeek and gameHour
- Replaces infinite values with pd.NA
- Calculates derived metrics like homeWin, visitorWin, tie, totalScore, pointDifference, and gameType (Close Game/Moderate Win/Blowout)
- Imputes missing values in the birthDate column using SimpleImputer(strategy='most_frequent').
- Converts the height column from a 'feet-inches' string format to total inches using a custom parsing function.
- Parses birthDate strings to datetime objects and calculates player age in years.
- Converts 'birthDate' to datetime, drops the original 'height' column, and applies categorical encoding.
- Applies custom preprocessing utilities (prep_data.main(), process_datasets.main()).
- Filters evaluation to ball_snap events and BEFORE_SNAP frame types.
- Batches sequences with DataLoader (batch_size=64, shuffle=True for training, num_workers=3).
- Merges DataFrames on gameId, playId, and nflId using inner joins; renames conflicting penaltyYards columns; imputes missing values by filling object columns with mode, numeric columns with mean, and boolean columns with mode; filters to selected columns for analysis.

## Features engineering
- Computes game-level aggregations (e.g., mean rushing/passing yards, total fumbles, average sack yards) using pandas groupby.
- Creates play-level transforms (e.g., penalty yards per play, yards after catch per play) using pandas transform.
- age (derived from birthDate)
- year, month, day (extracted from birthDate)
- height_in_inches
- bmi (weight in kg / height in meters squared)
- college_name_length
- college_name_count (frequency of collegeName)
- Derived temporal features (dayOfWeek, gameHour)
- Aggregated team metrics (average home/away scores, average point differences)
- Categorical game classifications (gameType based on point difference thresholds)
- Converts height from feet-inches format to total inches.
- Calculates player age in years from birth dates.
- height_inches
- age
- BMI
- position_encoded
- college_encoded
- birthYear
- Derives composite defensive metrics (totalTackles, totalFumbles, totalSackYards, totalTacklesForLoss) by summing related columns; aggregates yards and action counts by team, player (nflId), and game date; computes pairwise correlations and group-wise descriptive statistics.

## Models
- KMeans
- SportsTransformerLitModel

## Frameworks used
- pandas
- numpy
- matplotlib
- seaborn
- plotly
- datetime
- scipy
- scikit-learn
- pytorch
- pytorch-lightning
- IPython

## Insights
- Dynamic color swapping based on perceptual distance prevents visual ambiguity when team colors are too similar.
- Plotly's frame-based animation and slider mechanisms can be efficiently constructed programmatically for sports tracking data.
- Converting player speed from tracking units to MPH requires a fixed scaling factor (2.23693629205).
- Player birth years follow a left-skewed distribution with peaks around 1995-1996.
- Player weights are approximately normally distributed with a mean of 246 lbs and median of 236 lbs.
- Pre-snap home team win probability exhibits a normal, unimodal, and symmetric distribution.
- Most NFL games occur on Sundays, typically kicking off at 1:00 PM ET.
- Tracking data can be effectively visualized using 2D scatter plots and heatmaps to understand spatial player movements and tackle locations.
- Passing plays exhibit higher variability and potential for large gains compared to the more consistent, short-yardage nature of rushing plays.
- Yards after catch cluster tightly around 0.08 yards per game, suggesting most receptions yield minimal additional yardage.
- Fumble losses are heavily concentrated at 0-1 per game, indicating that ball security is generally maintained but critical when lost.
- Penalty yards per play are extremely rare and minimal, though they can accumulate significantly across a full game.
- Defensive pressure metrics like quarterback hits and sack yards show distinct distribution shapes, with hits clustering at higher pressure values while sack yards are more symmetrically distributed.
- Player ages range from 25 to 47, with a mean of approximately 29.44 years, indicating a relatively young active roster.
- The majority of players were born between 1985 and 1997, with March being the most frequent birth month.
- Height and weight distributions show clear demographic clustering, with maximum height at 80 inches and weight ranging from 175 to 328 lbs.
- College name length and frequency distributions provide a proxy for college prominence that can be correlated with player demographics.
- The notebook demonstrates how to systematically break down a complex, multi-table sports dataset into digestible visualizations and statistical summaries to establish a baseline understanding of game dynamics before modeling.
- Home teams win approximately 53.7% of games, confirming a slight home-field advantage.
- Primetime (8 PM) games tend to feature higher scoring for both home and visitor teams compared to morning games.
- Team performance varies significantly across weeks, with some teams showing consistent scoring while others fluctuate widely.
- Point differentials can be effectively categorized into Close Games, Moderate Wins, and Blowouts to analyze competitive balance.
- Player heights are recorded completely in feet-inches format, while birth dates have significant missingness (487 entries).
- The average player weight is approximately 245.77 lbs, ranging from 153 to 380 lbs.
- Player positions like QB, TE, and DE are well-categorized, enabling direct correlation between physical attributes and roles.
- The dataset contains 1,697 players with no duplicate entries, ensuring data reliability for downstream analysis.
- Player physiques are highly specialized and strongly correlate with positional roles, with linemen clustering at higher BMIs and speed positions at lower BMIs.
- Weight distributions show a bimodal pattern (~200 lbs and ~300 lbs), suggesting distinct physical archetypes in the league.
- Age trends indicate a generational shift toward younger players, with Quarterbacks showing the widest age range and longevity.
- Top recruiting colleges like Alabama and Ohio State dominate the player pool, highlighting established talent pipelines.
- KMeans clustering successfully separates players into five distinct physical archetypes based on height, weight, and BMI.
- Transformers naturally handle variable player ordering and capture complex interactions via self-attention, reducing the need for manual feature engineering.
- Spatio-temporal tracking data can be effectively modeled with relatively small datasets (e.g., one week) for formation prediction.
- The methodology is adaptable to tracking data across multiple sports beyond football.
- Passing and receiving plays collectively dominate total yardage, while rushing plays contribute less but remain strategically vital.
- Teams like JAX, GB, KC, and SF excel at gaining yards after the catch, whereas HOU and TB struggle with ball security.
- Defensive pressure and sack metrics vary significantly by team, with PIT and BUF showing strong pass-rush capabilities.
- Correlations between major offensive yardage metrics are near zero, indicating they operate largely independently at the play level.
- Missing data is heavily concentrated in defensive flags and penalty names, requiring careful imputation or exclusion for downstream modeling.

## Critical findings
- Some date columns in the dataset do not strictly follow the %Y-%m-%d format, requiring format='mixed' in pd.to_datetime().
- The author notes confusion regarding the most frequent game time (13:00 ET) versus typical NFL scheduling, highlighting a potential data representation quirk.
- The birthDate column contains 487 missing entries (28.7% of the dataset), requiring median and mode imputation to avoid analysis gaps.
- Height is stored as a non-numeric 'ft-in' string format, necessitating custom parsing to convert to inches for quantitative analysis.
- 487 out of 1,697 players have missing birth dates, which could limit age-based analysis.
- Heights are stored in a non-standard feet-inches string format requiring custom parsing.
- The dataset contains a sharp decline in players born after 1995, indicating potential sampling or roster limitations in the provided data.
- Regina (Canada) surprisingly leads the top 10 colleges by average BMI, which may reflect a small sample size or specific recruitment focus rather than a league-wide trend.
- Receiving yards total slightly exceeds passing yards total, highlighting the value of yards after catch in the dataset.
- Pairwise correlations between rushing, passing, and receiving yards are extremely weak (~-0.007 to 0.003), suggesting play-type yardage is not linearly coupled.
- Defensive metrics like causedPressure and wasInitialPassRusher have substantial missingness, indicating they are only recorded for specific play types or players.

## Notable individual insights
- votes 206 (Animate Plays with Plotly (real, no lies here)): Dynamic color swapping based on perceptual distance prevents visual ambiguity when team colors are too similar.
- votes 116 (NFL Big Data Bowl 2025): Some date columns in the dataset do not strictly follow the %Y-%m-%d format, requiring format='mixed' in pd.to_datetime().
- votes 90 (🏆NFL Players Stats:Comprehensive Exploration🏃‍♂️): The birthDate column contains 487 missing entries (28.7% of the dataset), requiring median and mode imputation to avoid analysis gaps.
- votes 81 (🏈 Games Insightful Analysis: Players in Action 📊): Home teams win approximately 53.7% of games, confirming a slight home-field advantage.
- votes 69 (Breaking Down the NFL Big Bowl 🏟️📊): KMeans clustering successfully separates players into five distinct physical archetypes based on height, weight, and BMI.
- votes 68 (Modeling with Transformers, by SumerSports): Transformers naturally handle variable player ordering and capture complex interactions via self-attention, reducing the need for manual feature engineering.
- votes 64 (🏆NFL Big Data Bowl | Complete Data Exploration📊): Pairwise correlations between rushing, passing, and receiving yards are extremely weak (~-0.007 to 0.003), suggesting play-type yardage is not linearly coupled.

## Notebooks indexed
- #206 votes [[notebooks/votes_01_nickwan-animate-plays-with-plotly-real-no-lies-here/notebook|Animate Plays with Plotly (real, no lies here)]] ([kaggle](https://www.kaggle.com/code/nickwan/animate-plays-with-plotly-real-no-lies-here))
- #116 votes [[notebooks/votes_02_mpwolke-nfl-big-data-bowl-2025/notebook|NFL Big Data Bowl 2025]] ([kaggle](https://www.kaggle.com/code/mpwolke/nfl-big-data-bowl-2025))
- #102 votes [[notebooks/votes_03_arshmankhalid-big-bowl-eda-crushing-player-stats/notebook|Big Bowl EDA: Crushing Player Stats! 📈💥]] ([kaggle](https://www.kaggle.com/code/arshmankhalid/big-bowl-eda-crushing-player-stats))
- #90 votes [[notebooks/votes_04_marianadeem755-nfl-players-stats-comprehensive-exploration/notebook|🏆NFL Players Stats:Comprehensive Exploration🏃‍♂️]] ([kaggle](https://www.kaggle.com/code/marianadeem755/nfl-players-stats-comprehensive-exploration))
- #86 votes [[notebooks/votes_05_marianadeem755-nfl-big-data-bowl-games-comprehensive-analysis/notebook|📊NFL Big Data Bowl:Games Comprehensive Analysis🏈]] ([kaggle](https://www.kaggle.com/code/marianadeem755/nfl-big-data-bowl-games-comprehensive-analysis))
- #81 votes [[notebooks/votes_06_arshmankhalid-games-insightful-analysis-players-in-action/notebook|🏈 Games Insightful Analysis: Players in Action 📊]] ([kaggle](https://www.kaggle.com/code/arshmankhalid/games-insightful-analysis-players-in-action))
- #72 votes [[notebooks/votes_07_mehakiftikhar-nfl-part-3-player-data-analysis/notebook|NFL | Part 3 - Player Data Analysis]] ([kaggle](https://www.kaggle.com/code/mehakiftikhar/nfl-part-3-player-data-analysis))
- #69 votes [[notebooks/votes_08_arshmankhalid-breaking-down-the-nfl-big-bowl/notebook|Breaking Down the NFL Big Bowl 🏟️📊]] ([kaggle](https://www.kaggle.com/code/arshmankhalid/breaking-down-the-nfl-big-bowl))
- #68 votes [[notebooks/votes_09_pvabish-modeling-with-transformers-by-sumersports/notebook|Modeling with Transformers, by SumerSports]] ([kaggle](https://www.kaggle.com/code/pvabish/modeling-with-transformers-by-sumersports))
- #64 votes [[notebooks/votes_10_marianadeem755-nfl-big-data-bowl-complete-data-exploration/notebook|🏆NFL Big Data Bowl | Complete Data Exploration📊]] ([kaggle](https://www.kaggle.com/code/marianadeem755/nfl-big-data-bowl-complete-data-exploration))
