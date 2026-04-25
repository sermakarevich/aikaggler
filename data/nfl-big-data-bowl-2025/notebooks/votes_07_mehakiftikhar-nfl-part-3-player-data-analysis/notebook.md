# NFL | Part 3 - Player Data Analysis

- **Author:** Mehak Iftikhar
- **Votes:** 72
- **Ref:** mehakiftikhar/nfl-part-3-player-data-analysis
- **URL:** https://www.kaggle.com/code/mehakiftikhar/nfl-part-3-player-data-analysis
- **Last run:** 2024-10-19 18:30:24.697000

---

# <p style="font-family: 'Amiri'; font-size: 2rem; color: black; text-align: center; margin: 0; text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3); background-color: #b379ed; padding: 30px; border-radius: 20px; border: 8px solid black; width:90%">˚NFL Big Data Bowl 2025: Part 3 - Player Data Analysis˚</p>

![Image.jpeg](attachment:f89f6c51-1446-4413-9e54-001a2cf01cd7.jpeg)

# <p style="font-family: 'Amiri'; font-size: 2rem; color: black; text-align: center; margin: 0; text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3); background-color: #b379ed; padding: 20px; border-radius: 20px; border: 8px solid black; width:95%">About Author</p>

<div style="border: 8px solid black; background-color: #b379ed; color: black; padding: 10px; border-radius: 10px;">
    <p><strong>Mehak Iftikhar</strong><br>
    I am a Data Science enthusiast who loves to explore and learn from data. My goal is to turn complex information into clear insights.</p>
    <p>Feel free to connect with me on social media:</p>
    <ul>
        <li><a href="https://www.linkedin.com/in/mehak-iftikhar/">LinkedIn</a></li>
        <li><a href="https://twitter.com/mehakkhan874">Twitter</a></li>
        <li><a href="https://github.com/mehakiftikhar">GitHub</a></li>
        <li><a href="https://www.facebook.com/MehakIftikharDS/">Facebook</a></li>
    </ul>
    <p>Let’s learn and grow together.</p>
</div>

# <p style="font-family: 'Amiri'; font-size: 2rem; color: black; text-align: center; margin: 0; text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3); background-color: #b379ed; padding: 20px; border-radius: 20px; border: 8px solid black; width:95%">About Data</p>

<div style="border: 8px solid black; padding: 15px; border-radius: 10px; background-color: #b379ed; color: black;">
    <p>
        In this notebook, we will shift our focus to the <strong>Player Data</strong> from the NFL Big Data Bowl 2025 dataset. This data provides valuable information about the players, including their physical attributes, positions, and college affiliations.
    </p>
    <p>
        We have already explored other aspects of the dataset in previous notebooks:
        <ul>
            <li>For an in-depth analysis of the <strong>Game Data</strong>, please refer to the <a href="https://www.kaggle.com/code/mehakiftikhar/nfl-part-1-initial-data-exploration/" style="color: #0a29f5;">First Notebook</a>.</li>
            <li>For an exploration of the <strong>Plays Data</strong>, visit the <a href="https://www.kaggle.com/code/mehakiftikhar/nfl-part-2-in-depth-analysis-of-plays-data" style="color: #0a29f5;">Second Notebook</a>.</li>
        </ul>
    </p>
    <p>
        In this third notebook, we will analyze the player-level data, focusing on physical characteristics, player positions, and their college backgrounds. Through this analysis, we aim to uncover trends and patterns in player attributes that could influence their performance on the field.
    </p>
</div>

# <p style="font-family: 'Amiri'; font-size: 2rem; color: black; text-align: center; margin: 0; text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3); background-color: #b379ed; padding: 20px; border-radius: 20px; border: 8px solid black; width:95%">Task</p>

<div style="border: 8px solid black; padding: 15px; border-radius: 10px; background-color: #b379ed; color: black;">
    <p>
        In this notebook, we will focus on analyzing the <strong>Player Data</strong> from the NFL Big Data Bowl 2025 dataset. This follows the foundational analyses done in our previous notebooks, where we explored:
        <ul>
            <li><a href="https://www.kaggle.com/code/mehakiftikhar/nfl-part-1-initial-data-exploration/" style="color: #0a29f5;">Game Data</a> in the <strong>First Notebook</strong></li>
            <li><a href="https://www.kaggle.com/code/mehakiftikhar/nfl-part-2-in-depth-analysis-of-plays-data" style="color: #0a29f5;">Plays Data</a> in the <strong>Second Notebook</strong></li>
        </ul>
    </p>
    <p>
        In this third notebook, we will dive into player-level information, such as physical attributes (height, weight), college background, and player positions. This data provides valuable insights into individual players and how their characteristics might correlate with their performance on the field.
    </p>
    <p>
        As we continue our journey through the NFL dataset, we aim to uncover patterns and trends that help explain player performance and contribute to the overall game dynamics. Let's begin by exploring the structure and statistics of the player data.
    </p>
</div>

# <p style="font-family: 'Amiri'; font-size: 2rem; color: black; text-align: center; margin: 0; text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3); background-color: #b379ed; padding: 20px; border-radius: 20px; border: 8px solid black; width:95%">Import Dependencies</p>

```python
# Import Basis
import pandas as pd 
import optuna
import numpy as np 
import matplotlib.pyplot as plt
from datetime import datetime
import plotly.express as px
import seaborn as sns 
import math
from io import StringIO
from colorama import Fore, Style, init;
# Import necessary libraries
from IPython.core.display import display, HTML
from scipy.stats import skew  
# Import Plotly.go
import plotly.graph_objects as go
# import Subplots
from plotly.subplots import make_subplots
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import LabelEncoder, MinMaxScaler , StandardScaler , QuantileTransformer
from sklearn.impute import SimpleImputer

# Paellete
palette = ['Black', '#b379ed']
color_palette = sns.color_palette(palette)

# Set the option to display all columns
pd.set_option('display.max_columns', None)
```

# <p style="font-family: 'Amiri'; font-size: 2rem; color: black; text-align: center; margin: 0; text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3); background-color: #b379ed; padding: 20px; border-radius: 20px; border: 8px solid black; width:95%">Load Data</p>

```python
players = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2025/players.csv')
```

# <p style="font-family: 'Amiri'; font-size: 2rem; color: black; text-align: center; margin: 0; text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3); background-color: #b379ed; padding: 20px; border-radius: 20px; border: 8px solid black; width:95%">Basic Overview</p>

```python
def styled_heading(text, background_color='#ffabf0', text_color='black'):
    return f"""
    <div style="
        text-align: center;
        background: {background_color};
        font-family: 'Montserrat', sans-serif;
        color: {text_color};
        padding: 15px;
        font-size: 30px;
        font-weight: bold;
        line-height: 1;
        border-radius: 20px 20px 0 0;
        margin: 20px 0;  /* Added margin for spacing */
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.2);
        border: 3px dashed {text_color};
    ">
        {text}
    </div>
    """

def D_O(train_df, heading_bg='#ffabf0', heading_color='black', text_bg='white', text_color='black'):
    try:
        # Display head and tail of the training dataset
        for heading, df_part in zip(
            ["The Head Of Dataset is:", "The Tail Of Dataset is:", "Numerical Summary of Data:"], 
            [train_df.head(5), train_df.tail(5), train_df.describe()]
        ):
            display(HTML(styled_heading(heading, background_color=heading_bg, text_color=heading_color)))
            display(HTML(df_part.to_html(index=False).replace(
                '<table border="1" class="dataframe">', 
                f'<table style="border: 8px solid black; margin-bottom: 20px; background-color: {text_bg}; color: {text_color};">'
            ).replace('<td>', f'<td style="color: {text_color}; background-color: {text_bg};">')))
            print("\n")  

        # Print shape data
        display(HTML(styled_heading("Shape Data:", background_color=heading_bg, text_color=heading_color)))
        print(f'Shape of the Data: {train_df.shape}')
        print(f'Rows: {train_df.shape[0]}')
        print(f'Columns: {train_df.shape[1]}')
        print("\n<br>\n")  

        # Print info of train data
        display(HTML(styled_heading("Info Of Data:", background_color=heading_bg, text_color=heading_color)))
        buffer = StringIO()
        train_df.info(buf=buffer)
        buffer.seek(0)
        info_str = buffer.read()
        display(HTML(f"<pre style='color: {text_color}; background-color: {text_bg}; margin-bottom: 20px; font-family: Courier, monospace; font-size: 14px; padding: 10px; border: 8px solid black;'>{info_str}</pre>"))
        print("\n<br>\n")  # Adding space between sections

        # Print categorical columns
        Cat_cols_train = [col for col in train_df.columns if train_df[col].dtype == 'O']
        display(HTML(styled_heading("Categorical Columns of Data:", background_color=heading_bg, text_color=heading_color)))
        print(f'The Categorical Columns of Data are: {Cat_cols_train}')
        print("\n<br>\n")  # Adding space between sections

        # Print numerical columns
        N_cols_train = [col for col in train_df.columns if train_df[col].dtype == 'float']
        display(HTML(styled_heading("Numerical Columns of Data:", background_color=heading_bg, text_color=heading_color)))
        print(f'The Numerical Columns of Data are: {N_cols_train}')
        print("\n<br>\n")  # Adding space between sections

        # Print null values
        display(HTML(styled_heading("Null Values in Data:", background_color=heading_bg, text_color=heading_color)))
        null_values = train_df.isnull().sum()
        display(HTML(f"<pre style='color: {text_color}; background-color: {text_bg}; margin-bottom: 20px; font-family: Courier, monospace; font-size: 14px; padding: 10px; border: 8px solid black;'>{null_values}</pre>"))
        print("\n<br>\n")  
        
        # Print duplicates check
        display(HTML(styled_heading("Duplicates Check in Data:", background_color=heading_bg, text_color=heading_color)))
        if train_df.duplicated().any():
            print(f'Duplicates exist in the dataset.')
        else:
            print(f'No duplicates found in the dataset.')
        print("\n<br>\n")  

    except Exception as e:
        print_error(str(e))
```

```python
D_O(players, heading_bg='#b379ed', heading_color='black', text_bg='#b379ed', text_color='black')
```

# <p style="font-family: 'Amiri'; font-size: 2rem; color: black; text-align: center; margin: 0; text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3); background-color: #b379ed; padding: 20px; border-radius: 20px; border: 8px solid black; width:95%">Insights from players.csv</p>

<div style="background-color: #f3f8fb; border: 2px solid #1e88e5; border-radius: 10px; padding: 15px; margin-bottom: 20px;">
  <h2 style="color: #1e88e5;"> Overview</h2>
  <p style="color: #2c3e50;">The dataset contains <strong>1,697 NFL players</strong> with detailed information on their physical attributes, educational background, and positions. Below are key insights derived from the data.</p>
</div>

<div style="background-color: #fef3e0; border: 2px solid #fb8c00; border-radius: 10px; padding: 15px; margin-bottom: 20px;">
  <h2 style="color: #fb8c00;">Key Insights</h2>

  <h3 style="color: #d32f2f;">1. Height Distribution</h3>
  <p style="color: #424242;"><strong>No missing values</strong> were found in the height column. Heights are recorded in feet and inches, providing complete data on player physical stature.</p>

  <h3 style="color: #d32f2f;">2. Weight Statistics</h3>
  <p style="color: #424242;">The average player weight is <strong>245.77 lbs</strong>, with a minimum of <strong>153 lbs</strong> and a maximum of <strong>380 lbs</strong>. This data can help us understand the diversity in player builds across different positions.</p>

  <h3 style="color: #d32f2f;">3. Missing Birth Dates</h3>
  <p style="color: #424242;"><strong>487 birth dates</strong> are missing, which could limit age-based analysis. However, a majority of the data is intact for use in other forms of analysis.</p>

  <h3 style="color: #d32f2f;">4. College Representation</h3>
  <p style="color: #424242;">Players come from various colleges, providing an opportunity to analyze performance based on their college backgrounds. <strong>No missing values</strong> were found in the college name column.</p>

  <h3 style="color: #d32f2f;">5. Player Positions</h3>
  <p style="color: #424242;">Player positions like <strong>QB</strong> (Quarterback), <strong>TE</strong> (Tight End), and <strong>DE</strong> (Defensive End) are well categorized, enabling us to explore how physical attributes correlate with player roles.</p>

  <h3 style="color: #d32f2f;">6. Data Completeness</h3>
  <p style="color: #424242;">The dataset is highly complete except for the missing birth dates. <strong>No duplicate entries</strong> were found, ensuring reliable player data.</p>
</div>

<div style="background-color: #e8f5e9; border: 2px solid #43a047; border-radius: 10px; padding: 15px;">
  <h2 style="color: #43a047;">Conclusion</h2>
  <p style="color: #2e7d32;">This dataset provides comprehensive information on NFL players, allowing for detailed analysis of player attributes, positional fit, and performance potential. Further analysis can uncover how these attributes relate to game success and team strategies.</p>
</div>

# <p style="font-family: 'Amiri'; font-size: 2rem; color: black; text-align: center; margin: 0; text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3); background-color: #b379ed; padding: 20px; border-radius: 20px; border: 8px solid black; width:95%">Dealing with Missing Values</p>

```python
def impute_missing_birth_dates(df):
    """
    Impute missing values for the birthDate column in the DataFrame.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame with potential missing birth dates.
        
    Returns:
        pd.DataFrame: DataFrame with birthDate missing values handled.
    """
    
    if df['birthDate'].isnull().sum() > 0:
        
        imputer = SimpleImputer(strategy='most_frequent')
        # Apply the imputer and flatten the result
        df['birthDate'] = imputer.fit_transform(df[['birthDate']]).ravel()
        output_message = "Imputed missing values in 'birthDate'."
    else:
        output_message = "'birthDate' has no missing values to impute."
    
    # Display results
    display_imputation_output(output_message, df['birthDate'].isnull().sum())
    
    return df

def display_imputation_output(message, missing_values_count):
    """
    Display the output in a styled box.
    
    Parameters:
        message (str): The message to display.
        missing_values_count (int): Count of missing values after handling.
    """
    # HTML output with styling
    styled_output = f"""
    <div style="background-color: #e0f7fa; border: 2px solid black; padding: 15px; border-radius: 5px; color: #000;">
        <strong>{message}</strong><br><br>
        <strong>Missing values in 'birthDate' after handling:</strong> {missing_values_count}
    </div>
    """
    
    # Display the styled output
    display(HTML(styled_output))
```

```python
# Apply the missing values handling function
players = impute_missing_birth_dates(players)
```

<div style="background-color: #71ebdc; border: 5px solid black; padding: 10px; border-radius: 5px; color: #003300;">
    <strong>Milestone 1:</strong> We have cleaned the dataset from null values.
</div>

# <p style="font-family: 'Amiri'; font-size: 2rem; color: black; text-align: center; margin: 0; text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3); background-color: #b379ed; padding: 20px; border-radius: 20px; border: 8px solid black; width:95%">Height Distribution of NFL Players</p>

```python
def display_output(title, data):
    """Display the output in a styled box."""
    data_str = data.to_string()
    styled_output = f"""
    <div style="background-color: #f0f8ff; border: 2px solid #4682B4; padding: 10px; border-radius: 5px; color: #2f4f4f;">
        <h3>{title}</h3>
        <pre>{data_str}</pre>
    </div>
    """
    display(HTML(styled_output))
    
def convert_height(height):
    """Convert height from feet-inches format to inches."""
    if isinstance(height, str) and '-' in height:
        feet, inches = height.split('-')
        return int(feet) * 12 + int(inches)
    return height  
def analyze_height_distribution(df, palette):
    """Analyze and plot the distribution of player heights."""
    df['height'] = df['height'].apply(convert_height)
    plt.figure(figsize=(10, 6))
    sns.histplot(df['height'].dropna(), bins=30, kde=True, color=palette[0])
    plt.title('Height Distribution of NFL Players', fontsize=16)
    plt.xlabel('Height (inches)', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.grid(True)
    plt.show()
```

```python
analyze_height_distribution(players, palette)
```

# <p style="font-family: 'Amiri'; font-size: 2rem; color: black; text-align: center; margin: 0; text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3); background-color: #b379ed; padding: 20px; border-radius: 20px; border: 8px solid black; width:95%">Weight Statistics</p>

```python
def analyze_weight_statistics(df):
    """Display summary statistics of player weights."""
    weight_stats = df['weight'].describe()
    display_output("Weight Statistics", weight_stats)
```

```python
analyze_weight_statistics(players)
```

# <p style="font-family: 'Amiri'; font-size: 2rem; color: black; text-align: center; margin: 0; text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3); background-color: #b379ed; padding: 20px; border-radius: 20px; border: 8px solid black; width:95%">Age Distribution of NFL Players</p>

```python
def analyze_age_distribution(df, palette):
    """Analyze and plot the distribution of player ages."""
    df['birthDate'] = pd.to_datetime(df['birthDate'], errors='coerce')
    df['age'] = (datetime.now() - df['birthDate']).dt.days // 365
    plt.figure(figsize=(10, 6))
    sns.histplot(df['age'].dropna(), bins=30, kde=True, color=palette[1])
    plt.title('Age Distribution of NFL Players', fontsize=16)
    plt.xlabel('Age (years)', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.grid(True)
    plt.show()
```

```python
analyze_age_distribution(players, palette)
```

# <p style="font-family: 'Amiri'; font-size: 2rem; color: black; text-align: center; margin: 0; text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3); background-color: #b379ed; padding: 20px; border-radius: 20px; border: 8px solid black; width:95%">College Representation & Position Count</p>

```python
def analyze_college_representation(df):
    """Display the number of players from each college."""
    college_counts = df['collegeName'].value_counts()
    display_output("College Representation", college_counts)

def analyze_position_count(df):
    """Display the number of players by position."""
    position_counts = df['position'].value_counts()
    display_output("Position Count", position_counts)
```

```python
analyze_college_representation(players)
analyze_position_count(players)
```

# <p style="font-family: 'Amiri'; font-size: 2rem; color: black; text-align: center; margin: 0; text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3); background-color: #b379ed; padding: 20px; border-radius: 20px; border: 8px solid black; width:95%">Height vs Age Scatter Plot with Position Hue</p>

```python
def analyze_height_vs_age(df, palette):
    """Analyze the relationship between height and age of players."""
    df['height'] = df['height'].apply(convert_height)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='height', y='age', data=df, hue='position', palette=palette, s=100, edgecolor='black')
    plt.title('Height vs Age of NFL Players by Position', fontsize=16)
    plt.xlabel('Height (inches)', fontsize=14)
    plt.ylabel('Age (years)', fontsize=14)
    plt.grid(True)
    plt.show()
```

```python
analyze_height_vs_age(players, palette)
```

# <p style="font-family: 'Amiri'; font-size: 2rem; color: black; text-align: center; margin: 0; text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3); background-color: #b379ed; padding: 20px; border-radius: 20px; border: 8px solid black; width:95%">Top 10 Colleges with Most Players</p>

```python
def top_colleges_by_player_count(df, top_n=10):
    """Display the top N colleges with the most NFL players."""
    top_colleges = df['collegeName'].value_counts().head(top_n)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_colleges.index, y=top_colleges.values, palette=color_palette)
    plt.title(f'Top {top_n} Colleges by Number of NFL Players', fontsize=16)
    plt.xlabel('College', fontsize=14)
    plt.ylabel('Number of Players', fontsize=14)
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.show()
```

```python
top_colleges_by_player_count(players)
```

# <p style="font-family: 'Amiri'; font-size: 2rem; color: black; text-align: center; margin: 0; text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3); background-color: #b379ed; padding: 20px; border-radius: 20px; border: 8px solid black; width:95%">Position-wise Height Distribution</p>

```python
def analyze_height_by_position(df, palette):
    """Analyze the height distribution by player position."""
    df['height'] = df['height'].apply(convert_height)
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='position', y='height', data=df, palette=palette)
    plt.title('Height Distribution by NFL Player Position', fontsize=16)
    plt.xlabel('Position', fontsize=14)
    plt.ylabel('Height (inches)', fontsize=14)
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.show()
```

```python
analyze_height_by_position(players, palette)
```

# <p style="font-family: 'Amiri'; font-size: 2rem; color: black; text-align: center; margin: 0; text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3); background-color: #b379ed; padding: 20px; border-radius: 20px; border: 8px solid black; width:95%">Weight Distribution by Position</p>

```python
def analyze_weight_by_position(df, palette):
    """Analyze the weight distribution by player position."""
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='position', y='weight', data=df, palette=palette)
    plt.title('Weight Distribution by NFL Player Position', fontsize=16)
    plt.xlabel('Position', fontsize=14)
    plt.ylabel('Weight (lbs)', fontsize=14)
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.show()
```

```python
analyze_weight_by_position(players, palette)
```

<div style="border: 8px solid black; background-color: #b379ed; color: black; padding: 10px; border-radius: 10px;">
    <h1 style="text-align: center;">Conclusion and Next Steps</h1>
    <p>In this notebook, we conducted an in-depth analysis of player data from the NFL Big Data Bowl 2025 dataset. We explored key insights regarding player attributes such as height, weight, age, and college representation.</p>
    <p>This analysis has provided us with a deeper understanding of the demographic trends and physical attributes of NFL players, and we have also identified important relationships between player positions and their physical characteristics.</p>
    <p>For further context and insights on the plays data, please refer to my previous notebooks:</p>
    <ul>
        <li><a href="https://www.kaggle.com/code/mehakiftikhar/nfl-part-1-initial-data-exploration/" target="_blank">NFL Big Data Bowl 2025: Part 1 - Initial Data Exploration</a></li>
        <li><a href="https://www.kaggle.com/code/mehakiftikhar/nfl-part-2-in-depth-analysis-of-plays-data/" target="_blank">NFL Big Data Bowl 2025: Part 2 - In-Depth Analysis of Plays Data</a></li>
    </ul>
    <p>In the next notebook, we will explore tracking data to understand player movements and game strategies at a more granular level. This will involve analyzing how player speed, direction, and other movement metrics impact game outcomes.</p>
    <p>Exciting insights await, so stay tuned.</p>
</div>