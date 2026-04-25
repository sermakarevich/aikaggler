# 🏆NFL Players Stats:Comprehensive Exploration🏃‍♂️

- **Author:** maria nadeem
- **Votes:** 90
- **Ref:** marianadeem755/nfl-players-stats-comprehensive-exploration
- **URL:** https://www.kaggle.com/code/marianadeem755/nfl-players-stats-comprehensive-exploration
- **Last run:** 2024-12-10 10:42:05.960000

---

# <p style="font-family: 'Amiri'; font-size: 2.5rem; color: Black; text-align: center; margin: 0; text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3); background-color: #cce5ff; padding: 20px; border-radius: 20px; border: 7px solid #2a71b8; width:95%">Unveiling the BFL Big Data Bowl:Comprehensive Analysis of Players.csv File</p>

![_04aa9b9d-d6eb-4059-af44-8a9cfca5c04e (1).jpeg](attachment:521411c5-3e91-4330-83d3-0708d0d20700.jpeg)

<div style="background-color: #ADD8E6; padding: 20px; border-radius: 15px; border: 2px solid #00008B; margin-top: 20px;">
    <p style="text-align: justify; font-family: 'Times New Roman', serif; font-size: 20px; color: #00008B; font-weight: bold;">
        In-Depth Analysis of NFL Big Data Bowl 2025 Players Data
    </p>
    <p style="text-align: justify; font-family: 'Times New Roman', serif; font-size: 18px; color: #00008B;">
        In this notebook, we conducted a comprehensive analysis of the player data from the NFL Big Data Bowl 2025. Our goal was to explore key aspects of player characteristics, including their age, height, weight, and college affiliation Names, Display Names etc. This dataset contains 1697 rows and 7 columns. During our analysis, we found that there are some missing values in the data, with a total of 487 missing entries, which account for about 28.7% of the data.
        <br><br>
        📌 <strong>Columns with Missing Values:</strong>
        <br>
        birthDate (Missing: 487 entries, 28.7% of the data)
        <br><br>
        The dataset contains the following columns:
        <table style="width: 100%; border-collapse: collapse; margin-top: 20px;">
            <thead>
                <tr style="background-color: #ADD8E6;">
                    <th style="padding: 10px; border: 1px solid #00008B;">Column Name</th>
                    <th style="padding: 10px; border: 1px solid #00008B;">Description</th>
                </tr>
            </thead>
            <tbody>
                <tr style="background-color: #E6F0FF;">
                    <td style="padding: 10px; border: 1px solid #00008B;">nflId</td>
                    <td style="padding: 10px; border: 1px solid #00008B;">Player identification number, unique across players (numeric)</td>
                </tr>
                <tr style="background-color: #E6F0FF;">
                    <td style="padding: 10px; border: 1px solid #00008B;">height</td>
                    <td style="padding: 10px; border: 1px solid #00008B;">Player height (text)</td>
                </tr>
                <tr style="background-color: #E6F0FF;">
                    <td style="padding: 10px; border: 1px solid #00008B;">weight</td>
                    <td style="padding: 10px; border: 1px solid #00008B;">Player weight (numeric)</td>
                </tr>
                <tr style="background-color: #E6F0FF;">
                    <td style="padding: 10px; border: 1px solid #00008B;">birthDate</td>
                    <td style="padding: 10px; border: 1px solid #00008B;">Date of birth (YYYY-MM-DD)</td>
                </tr>
                <tr style="background-color: #E6F0FF;">
                    <td style="padding: 10px; border: 1px solid #00008B;">collegeName</td>
                    <td style="padding: 10px; border: 1px solid #00008B;">Player college (text)</td>
                </tr>
                <tr style="background-color: #E6F0FF;">
                    <td style="padding: 10px; border: 1px solid #00008B;">position</td>
                    <td style="padding: 10px; border: 1px solid #00008B;">Official player position (text)</td>
                </tr>
                <tr style="background-color: #E6F0FF;">
                    <td style="padding: 10px; border: 1px solid #00008B;">displayName</td>
                    <td style="padding: 10px; border: 1px solid #00008B;">Player name (text)</td>
                </tr>
            </tbody>
        </table>
        <br><br>
        Through this in deapth analysis, we aim to better understand the demographic trends, and the relationships between players' positions and their physical characteristics. The next step is to dive deeper into the players play tracking data, analyzing how players perform on the field across different weeks.
    </p>
</div>

<div style="background-color: #ADD8E6; padding: 20px; border-radius: 15px; border: 2px solid #00008B; margin-top: 20px;">
    <p style="text-align: justify; font-family: 'Times New Roman', serif; font-size: 20px; color: #00008B; font-weight: bold;">
        Players Data Comprehensive Analysis Approach
    </p>
    <p style="text-align: justify; font-family: 'Times New Roman', serif; font-size: 18px; color: #00008B;">
        Here is a breakdown of each column Analysis Approach
        <br><br>
        📌 <strong style="font-size: 18px; color: #00008B;">nflId</strong>  
        <br>
        <span style="font-size: 18px; color: #000000;">The `nflId` is the unique player identification number, which allows us to track and analyze individual players throughout the dataset. It doesn't require much analysis, but it’s essential for merging with other datasets or tracking player-specific metrics.</span>
        <br><br>
        📌 <strong style="font-size: 18px; color: #00008B;">height</strong>  
        <br>
        <span style="font-size: 18px; color: #000000;">The `height` column represents the player’s height (text-based). This is a categorical variable in a text format, and you can analyze it by:</span>
        <ul style="font-size: 18px; color: #000000;">
            <li>Converting height to a numerical format (in inches) for easier analysis.</li>
            <li>Checking if there are any trends between player height and their position, college, or weight.</li>
            <li>Visualizing the distribution of heights using histograms or box plots to see common height ranges for each position.</li>
        </ul>       
        <br><br>
        📌 <strong style="font-size: 18px; color: #00008B;">weight</strong>  
        <br>
        <span style="font-size: 18px; color: #000000;">The `weight` column represents the player’s weight in numeric form. This column can be analyzed by:</span>
        <ul style="font-size: 18px; color: #000000;">
            <li>Visualizing the distribution of weight across positions to understand if certain positions tend to have higher or lower weights.</li>
            <li>Analyzing correlations between weight and other columns like height and position.</li>
        </ul>
        <br><br>
        📌 <strong style="font-size: 18px; color: #00008B;">birthDate</strong>  
        <br>
        <span style="font-size: 18px; color: #000000;">The `birthDate` column contains the player's date of birth in the format YYYY-MM-DD. Analysis of this column can include:</span>
        <ul style="font-size: 18px; color: #000000;">
            <li>Calculating the player's age at the time of data collection.</li>
            <li>Examining age distribution across different positions to see if certain positions are dominated by younger or older players.</li>
        </ul>
        <br><br>
        📌 <strong style="font-size: 18px; color: #00008B;">collegeName</strong>  
        <br>
        <span style="font-size: 18px; color: #000000;">The `collegeName` column represents the player's college. This is a categorical variable that can be analyzed by:</span>
        <ul style="font-size: 18px; color: #000000;">
            <li>Finding the most common colleges that players come from.</li>
            <li>Analyzing performance or trends based on college affiliation (if combined with other datasets or player stats).</li>
            <li>Identifying top colleges for producing NFL players and correlating college names with player positions or other characteristics.</li>
        </ul>        
        <br><br>
        📌 <strong style="font-size: 18px; color: #00008B;">position</strong>  
        <br>
        <span style="font-size: 18px; color: #000000;">The `position` column represents the player’s official position on the team. You can analyze this column by:</span>
        <ul style="font-size: 18px; color: #000000;">
            <li>Creating visualizations that show the distribution of positions in the dataset.</li>
            <li>Examining how other player attributes (height, weight, age) vary across positions.</li>
            <li>Looking for patterns or trends in player performance based on their position.</li>
        </ul>
        <br><br>
        📌 <strong style="font-size: 18px; color: #00008B;">displayName</strong>  
        <br>
        <span style="font-size: 18px; color: #000000;">The `displayName` column holds the player’s name. Although it doesn't contribute directly to the analysis, it is useful for identifying players in visualizations or reports.</span>
        <br><br>
        Through these analyses, we aim to uncover insights about the physical attributes and characteristics of NFL players, such as trends in height, weight, and age based on positions and colleges.
        <br><br>
        📌 <strong style="font-size: 18px; color: #00008B;">Other Notebooks of our Team:</strong>  
        Here are some team notebooks that may provide additional and in-depth insights and analysis for NFL Big Data Bowl 2025:
        <ul style="font-size: 18px; color: #000000;">
            <li><a href="https://www.kaggle.com/code/marianadeem755/nfl-big-data-bowl-games-comprehensive-analysis" target="_blank" style="color: #000000;">Notebook1: Games.csv Analysis</a></li>
            <li><a href="https://www.kaggle.com/code/mehakiftikhar/nfl-part-1-initial-data-exploration" target="_blank" style="color: #000000;">Notebook2: Games.csv Notebook 2</a></li>
            <li><a href="https://www.kaggle.com/code/mehakiftikhar/nfl-part-2-in-depth-analysis-of-plays-data" target="_blank" style="color: #000000;">Notebook3: Plays.csv Analysis</a></li>
            <li><a href="https://www.kaggle.com/code/mehakiftikhar/nfl-part-3-player-data-analysis" target="_blank" style="color: #000000;">Notebook4: Player.csv Analysis</a></li>
        </ul>
    </p>
</div>

# <p style="font-family: 'Amiri'; font-size: 3rem; color: Black; text-align: center; margin: 0; text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3); background-color: #cce5ff; padding: 20px; border-radius: 20px; border: 7px solid #2a71b8; width:95%">About Author</p>

* Hi Kagglers! I'm Maria Nadeem, a passionate Data Scientist with keen interest in exploring and applying diverse data science techniques.
* As dedicated to derive meaningful insights and making impactful decisions through data, I actively engage in projects and contribute to Kaggle by sharing detailed analysis and actionable insights.

| Name               | Email                                               | LinkedIn                                                  | GitHub                                           | Kaggle                                        |
|--------------------|-----------------------------------------------------|-----------------------------------------------------------|--------------------------------------------------|-----------------------------------------------|
| **Maria Nadeem**  | marianadeem755@gmail.com | <a href="https://www.linkedin.com/in/maria-nadeem-4994122aa/" style="text-decoration: none; font-size: 16px;"><img src="https://img.shields.io/badge/LinkedIn-%2300A4CC.svg?style=for-the-badge&logo=LinkedIn&logoColor=white" alt="LinkedIn Badge"></a> | <a href="https://github.com/marianadeem755" style="text-decoration: none; font-size: 16px;"><img src="https://img.shields.io/badge/GitHub-%23FF6F61.svg?style=for-the-badge&logo=GitHub&logoColor=white" alt="GitHub Badge"></a> | <a href="https://www.kaggle.com/marianadeem755" style="text-decoration: none; font-size: 16px;"><img src="https://img.shields.io/badge/Kaggle-%238a2be2.svg?style=for-the-badge&logo=Kaggle&logoColor=white" alt="Kaggle Badge"></a> |

```python
!pip install pio
```

# <p style="font-family: 'Amiri'; font-size: 3rem; color: Black; text-align: center; margin: 0; text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3); background-color: #cce5ff; padding: 20px; border-radius: 20px; border: 7px solid #2a71b8; width:95%">Import Libraries</p>

```python
# Import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio  # Import pio for setting the default renderer
from plotly.subplots import make_subplots
import plotly.subplots as sp
from datetime import datetime
import altair as alt
import random
from IPython.display import display, HTML
from matplotlib.colors import LinearSegmentedColormap
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")
# Set the default renderer for both Plotly Express and Graph Objects
pio.renderers.default = 'iframe_connected'
```

# <p style="font-family: 'Amiri'; font-size: 3rem; color: Black; text-align: center; margin: 0; text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3); background-color: #cce5ff; padding: 20px; border-radius: 20px; border: 7px solid #2a71b8; width:95%">Players Data Overview</p>

```python
import pandas as pd
import seaborn as sns
from IPython.core.display import display, HTML
import random

# Function to style tables with a unique blue color scheme and inner box styling
def style_table(df):
    styled_df = df.style.set_table_styles([
        {"selector": "th", "props": [("color", "white"), ("background-color", "#2a71b8")]}  # Blue header
    ]).set_properties(**{"text-align": "center", "background-color": "#cce5ff", 'border': '2px solid #2a71b8'}).hide(axis="index")
    return styled_df.to_html()

# Function to generate random shades of blue
def generate_random_color():
    color = "#{:02x}{:02x}{:02x}".format(
        random.randint(50, 100),
        random.randint(100, 150),
        random.randint(200, 255)
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

# Define your blue color palette
palette = ['#2a71b8', '#4c9cd1', '#7fb8d5', '#a7cbe1', '#cce5ff']
color_palette = sns.color_palette(palette)

# Define colors for headings and sub-headings
main_heading_color = '#2a71b8'  # Main heading color (Dark Blue)
sub_heading_color = '#4c9cd1'   # Sub-heading color (Light Blue)
headings_border_color = '#a7cbe1'  # Border color for headings (Light Blue)

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
    
    # Check for null values for each column and display them in a styled table
    display(HTML(f"<h2 style='font-size: 24px; color: {sub_heading_color};'>🚨 Null Values</h2>"))
    null_counts = dataset.isnull().sum()
    null_values_table = null_counts[null_counts > 0].to_frame(name='Null Values')
    
    # Calculate missing values percentage
    missing_percentage = (null_counts / dataset.shape[0]) * 100
    missing_percentage_table = missing_percentage[missing_percentage > 0].to_frame(name='Missing Percentage')

    # Merge null counts and missing percentage
    null_percentage_df = pd.concat([null_values_table, missing_percentage_table], axis=1)
    if not null_percentage_df.empty:
        display(HTML(style_table(null_percentage_df)))
    else:
        display(HTML("<p>No null values found.</p>"))
    
    # Print column names with missing values
    columns_with_missing = null_counts[null_counts > 0].index.tolist()
    if columns_with_missing:
        display(HTML(f"<h2 style='font-size: 24px; color: {sub_heading_color};'>📌 Columns with Missing Values</h2>"))
        display(HTML(f"<p>{', '.join(columns_with_missing)}</p>"))
    else:
        display(HTML("<p>No columns with missing values.</p>"))
    
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
# Load the Players dataset
players_df = pd.read_csv("/kaggle/input/nfl-big-data-bowl-2025/players.csv")

# Use different palette colors for different datasets
print_dataset_analysis(players_df, "Players Data", palette_index=0)
```

# <p style="font-family: 'Amiri'; font-size: 3rem; color: Black; text-align: center; margin: 0; text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3); background-color: #cce5ff; padding: 20px; border-radius: 20px; border: 7px solid #2a71b8; width:95%">Explore birthDate & nflid Column</p>

```python
# Function to impute missing values in the birthDate column
def impute_birth_date(df):
    # Convert 'birthDate' column to datetime format if not already
    df['birthDate'] = pd.to_datetime(df['birthDate'], errors='coerce')

    # Impute missing values with the median birthDate
    median_birth_date = df['birthDate'].median()
    
    # Fill the missing values with the median birthDate
    df['birthDate'].fillna(median_birth_date, inplace=True)

    return df

# Example usage with your dataset (`players_df`)

# Load the dataset
players_df = pd.read_csv("/kaggle/input/nfl-big-data-bowl-2025/players.csv")

# Impute missing birthDate values
players_df_imputed = impute_birth_date(players_df)

# Check the results
display(players_df_imputed[['nflId', 'birthDate']].head())
```

# <p style="font-family: 'Amiri'; font-size: 3rem; color: Black; text-align: center; margin: 0; text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3); background-color: #cce5ff; padding: 20px; border-radius: 20px; border: 7px solid #2a71b8; width:95%">Explore birthDate on the basis of Age</p>

```python
def calculate_age_and_dates(df):
    """
    Calculate the age of players based on their birthDate column.
    Adds 'age', 'age_count', and other date-related columns (year, month, day, and date).
    Removes seconds from 'birthDate' and imputes missing values.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame containing a 'birthDate' column.
    
    Returns:
    pd.DataFrame: Updated DataFrame with 'age', 'age_count', 'year', 'month', 'day', and 'date' columns.
    """
    
    # Convert 'birthDate' to datetime (if it's not already) and remove seconds
    df['birthDate'] = pd.to_datetime(df['birthDate'], errors='coerce')  # Convert to datetime
    df['birthDate'] = df['birthDate'].dt.floor('T')  # Removes seconds by flooring to the nearest minute
    
    # Impute missing birthDate using the mode (most frequent date)
    mode_birthDate = df['birthDate'].mode()[0]  # Get the most frequent birthDate
    df['birthDate'] = df['birthDate'].fillna(mode_birthDate)
    
    # Calculate age based on birthDate
    df['age'] = df['birthDate'].apply(lambda x: (datetime.now() - x).days // 365 if pd.notnull(x) else np.nan)
    
    # Count occurrences of each unique age
    age_counts = df['age'].value_counts().sort_index()  # Count occurrences of each unique age
    
    # Map the age count back into the DataFrame
    df['age_count'] = df['age'].map(age_counts)
    
    # Impute missing values in 'age' and 'age_count' using backfill
    df['age'] = df['age'].fillna(method='bfill')  # Use 'ffill' if you prefer forward fill
    df['age_count'] = df['age_count'].fillna(method='bfill')  # Same for 'age_count'
    
    # Create new date-related columns
    df['year'] = df['birthDate'].dt.year
    df['month'] = df['birthDate'].dt.month
    df['day'] = df['birthDate'].dt.day
    df['date'] = df['birthDate'].dt.date  # Extract the date part (YYYY-MM-DD)

    # Display first few rows to verify
    display(df[['nflId', 'birthDate', 'age', 'age_count', 'year', 'month', 'day', 'date']].head(10))  # Display first few rows to verify
    
    return df
# Apply the function (assuming 'players_df' contains the correct 'birthDate' column)
print("Explore Birth Date on the basis of Age")
players_df = calculate_age_and_dates(players_df)
```

# <p style="font-family: 'Amiri'; font-size: 3rem; color: Black; text-align: center; margin: 0; text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3); background-color: #cce5ff; padding: 20px; border-radius: 20px; border: 7px solid #2a71b8; width:95%">Visualize birthDate, Year & Month on the basis of Age</p>

```python
def plot_age_and_date_features_plotly(df):
    print("Generating Interactive Visualizations for Age and Date Features with Plotly")
    
    # Create a subplot with 2 rows and 2 columns
    fig = make_subplots(
        rows=2, cols=2, 
        subplot_titles=("Age Distribution", "Year of Birth Distribution", "Month of Birth Distribution"),
        vertical_spacing=0.3,
        specs=[[{"type": "histogram"}, {"type": "bar"}], [{"type": "bar"}, None]]
    )
    
    # Age Distribution (Histogram)
    fig.add_trace(
        go.Histogram(
            x=df['age'], 
            nbinsx=30, 
            marker=dict(color='rgba(70, 130, 180, 0.7)', line=dict(color='rgba(70, 130, 180, 1.0)', width=1)),
            name="Age Distribution",
            hovertemplate="Age: %{x}<br>Count: %{y}<extra></extra>"  # Show age and count
        ),
        row=1, col=1
    )
    
    # Year of Birth Distribution (Bar Plot)
    year_counts = df['year'].value_counts().sort_index()
    fig.add_trace(
        go.Bar(
            x=year_counts.index, 
            y=year_counts.values, 
            marker=dict(color='rgba(34, 139, 34, 0.7)', line=dict(color='rgba(34, 139, 34, 1.0)', width=1)),
            name="Year of Birth",
            hovertemplate="Year of Birth: %{x}<br>Count: %{y}<extra></extra>"  # Show year and count
        ),
        row=1, col=2
    )
    
    # Month of Birth Distribution (Bar Plot)
    month_counts = df['month'].value_counts().sort_index()
    fig.add_trace(
        go.Bar(
            x=month_counts.index, 
            y=month_counts.values, 
            marker=dict(color='rgba(218, 165, 32, 0.7)', line=dict(color='rgba(218, 165, 32, 1.0)', width=1)),
            name="Month of Birth",
            hovertemplate="Month: %{x}<br>Count: %{y}<extra></extra>"  # Show month and count
        ),
        row=2, col=1
    )
    
    # Update layout for better aesthetics
    fig.update_layout(
        title_text="Player Age and Date Features",
        title_x=0.5,  # Center the title
        showlegend=False,  # Hide the legend
        template="plotly_white",  # Use white background
        height=800,
        margin=dict(t=50, b=50, l=50, r=50),
        font=dict(size=12, color="black")  # Use black font for better contrast
    )
    
    # Update x-axis and y-axis labels
    fig.update_xaxes(title_text="Age", row=1, col=1)
    fig.update_yaxes(title_text="Frequency", row=1, col=1)
    fig.update_xaxes(title_text="Year of Birth", row=1, col=2)
    fig.update_yaxes(title_text="Count", row=1, col=2)
    fig.update_xaxes(title_text="Month", row=2, col=1)
    fig.update_yaxes(title_text="Count", row=2, col=1)
    
    # Show the figure
    fig.show()
```

```python
# Apply the function
print("Age Distribution Visualization")
plot_age_and_date_features_plotly(players_df)
```

<div style="background-color: #D6EAF8; padding: 20px; border-radius: 15px; border: 2px solid #00008B; margin-top: 20px;">
    <p style="text-align: justify; font-family: 'Times New Roman', serif; font-size: 20px; color: #00008B; font-weight: bold;">
        Observations of Players' Age and Birth Year:
    </p>
    <ul style="font-family: 'Georgia', serif; font-size: 14px; color: black; margin-left: 20px;">
        <li>The <strong>Minimum Age</strong> of players is <strong>25.0</strong>, while the <strong>Maximum Age</strong> is <strong>47.0</strong>, and the <strong>Mean Age</strong> is approximately <strong>29.44</strong> according to this dataset.</li>
        <li>Most players were born between <strong>1985 and 1997</strong>, with an age range of <strong>38-47</strong>.</li>
        <li>Players in the age range of <strong>38-47</strong> are mostly represented by frequencies of <strong>1, 3, and 7</strong>.</li>
        <li>Players born in the year <strong>1985</strong> have the highest frequency, with their birth months primarily being <strong>January, May, and August</strong>, and birth days ranging from <strong>16th to 30th</strong>.</li>
        <li>The age <strong>29</strong> has the highest frequency count of <strong>681</strong>, while ages <strong>39-46</strong> have the lowest frequency count, approximately <strong>7</strong>.</li>
        <li>Players born in the year <strong>1985</strong> have the highest frequency distribution, with a count of <strong>688</strong>.</li>
        <li>The birth month <strong>March</strong> (3rd month) has the highest frequency distribution count of <strong>585</strong>, whereas the birth month <strong>October</strong> (10th month) has the lowest frequency distribution count of <strong>80</strong>.</li>
    </div>

</div>

# <p style="font-family: 'Amiri'; font-size: 3rem; color: Black; text-align: center; margin: 0; text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3); background-color: #cce5ff; padding: 20px; border-radius: 20px; border: 7px solid #2a71b8; width:95%">Explore birthDate on the basis of Mean Age & nflid</p>

```python
def calculate_age(df):
    """
    Calculate the age of players based on their birthDate column.
    Adds 'age' and 'mean_age' columns to the DataFrame while keeping the original 'birthDate' column.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame containing a 'birthDate' column.
    
    Returns:
    pd.DataFrame: Updated DataFrame with 'age' and 'mean_age' columns added.
    """
    print("Calculating Age")
    
    # Convert 'birthDate' column to datetime if it's not already in datetime format
    df['birthDate'] = pd.to_datetime(df['birthDate'], errors='coerce')
    
    # Calculate age based on birthDate
    df['age'] = df['birthDate'].apply(lambda x: (datetime.now() - x).days // 365 if pd.notnull(x) else np.nan)
    
    # Calculate mean age
    mean_age = df['age'].mean()
    
    # Add mean age as a column to the DataFrame
    df['mean_age'] = mean_age
    
    # Display first few rows sorted by age in ascending order
    display(df[['nflId', 'birthDate', 'age', 'mean_age']].sort_values(by='age').head(10))
    
    return df

# Load the dataset
players_df = pd.read_csv("/kaggle/input/nfl-big-data-bowl-2025/players.csv")

# Calculate age
players_df = calculate_age(players_df)

# Display the minimum, maximum, and mean ages
print("Minimum Age:", players_df["age"].min())
print("Maximum Age:", players_df["age"].max())
print("Mean Age:", players_df["mean_age"].iloc[0])  # Display the mean age from the first row (it is constant for all rows)
```

# <p style="font-family: 'Amiri'; font-size: 3rem; color: Black; text-align: center; margin: 0; text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3); background-color: #cce5ff; padding: 20px; border-radius: 20px; border: 7px solid #2a71b8; width:95%">Visualize Age, Mean Age & nflid</p>

```python
# Function to plot the unique age distribution of NFL players
def plot_unique_age_distribution(df):
    # Ensure 'age' and 'nflId' columns exist
    if 'age' not in df.columns or 'nflId' not in df.columns:
        print("Error: 'age' or 'nflId' column is missing in the DataFrame.")
        return

    # Filter out rows with NaN values in 'age'
    df_clean = df.dropna(subset=['age'])

    # Check if the 'age' column contains numeric values
    if not pd.api.types.is_numeric_dtype(df_clean['age']):
        print("Error: 'age' column is not numeric.")
        return

    # Custom color palette (vibrant and unusual)
    bubble_colors = ['#D50032', '#FF4081', '#7C4DFF', '#00E5FF', '#69F0AE', '#FFB300', '#6200EA', '#FF9100']

    # Bubble chart for age distribution
    fig = go.Figure()

    # Add trace for bubble chart
    fig.add_trace(go.Scatter(
        x=df_clean['age'], 
        y=df_clean['nflId'], 
        mode='markers',
        marker=dict(
            size=df_clean['age'] / 2,  # Bubble size proportional to age
            color=df_clean['age'],  # Color based on age
            colorscale=bubble_colors,  # Custom color scale
            showscale=True,  # Show color scale
            line=dict(width=1, color='Black')  # Border color of the bubbles
        ),
        text=df_clean['nflId'],  # Show nflId in hover
        hovertemplate="NFL ID: %{text}<br>Age: %{x}<extra></extra>"  # Simplified hovertemplate
    ))

    # Customize layout
    fig.update_layout(
        title="Unique Age Distribution of NFL Players",
        title_x=0.5,  # Center the title
        xaxis_title="Age of Players",
        yaxis_title="NFL Player ID",
        template="plotly_white",  # White theme for better contrast with unique colors
        font=dict(size=14, color="black"),  # Black font for contrast
        showlegend=False,  # Hide legend
        height=700,  # Adjust the height of the plot
        margin=dict(t=50, b=50, l=50, r=50),
        plot_bgcolor='white'  # White background for a clean look
    )

    # Add gridlines and a more attractive appearance
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='gray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='gray')

    # Show the figure
    fig.show()
```

```python
# Display visualization by calling the function
print("Age Vs nflid Distribution Visualization")
plot_unique_age_distribution(players_df)  # Assuming players_df is already defined
```

# <p style="font-family: 'Amiri'; font-size: 3rem; color: Black; text-align: center; margin: 0; text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3); background-color: #cce5ff; padding: 20px; border-radius: 20px; border: 7px solid #2a71b8; width:95%">Explore Height on the basis of Age & nflid</p>

```python
def convert_height_to_inches(df):
    """
    Converts height from 'ft-in' format to inches.
    Adds 'height_in_inches', 'max_height', 'mean_height', 'age', and 'mean_age' columns.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame containing the 'height' and 'age' columns.
    
    Returns:
    pd.DataFrame: Updated DataFrame with the converted height and added columns.
    """
    print("Converting height from 'ft-in' to inches")
    
    # Helper function to convert height string to inches
    def height_to_inches(height_str):
        if pd.notnull(height_str):
            try:
                ft, inch = map(int, height_str.split('-'))
                return ft * 12 + inch
            except ValueError:
                return np.nan
        return np.nan

    # Apply the conversion to the 'height' column
    df['height_in_inches'] = df['height'].apply(height_to_inches)
    
    # Calculate max height and mean height
    max_height = df['height_in_inches'].max()
    mean_height = df['height_in_inches'].mean()

    # Add the max and mean height as columns in the DataFrame
    df['max_height'] = max_height
    df['mean_height'] = mean_height

    # Include age and mean_age columns in the output
    df['mean_age'] = df['age'].mean()  # Assuming 'age' column already exists

    # Display the result for verification (show first few rows)
    display(df[['nflId', 'height', 'height_in_inches', 'max_height', 'mean_height', 'age', 'mean_age']].head(10))  # Display first few rows to verify
    
    return df

# Apply the function (assuming 'players_df_filtered' has already been defined and contains 'height' and 'age' columns)
players_df = convert_height_to_inches(players_df)
```

```python
# BMI (Body Mass Index) from height (in meters) and weight (in kg)
print("Calculating BMI")
def calculate_bmi(row):
    if pd.notnull(row['height_in_inches']) and pd.notnull(row['weight']):
        height_meters = row['height_in_inches'] * 0.0254  # converting inches to meters
        weight_kg = row['weight'] * 0.453592  # converting pounds to kilograms
        if height_meters > 0 and weight_kg > 0:
            return weight_kg / (height_meters ** 2)
        return np.nan
    
players_df['bmi'] = players_df.apply(calculate_bmi, axis=1)
display(players_df[['nflId', 'bmi']].head())  # Print the first few rows to verify BMI calculation
```

# <p style="font-family: 'Amiri'; font-size: 3rem; color: Black; text-align: center; margin: 0; text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3); background-color: #cce5ff; padding: 20px; border-radius: 20px; border: 7px solid #2a71b8; width:95%">Visualize Height on the basis of Age & nflid</p>

```python
def plot_height_age_distribution(players_df_filtered):
    # Custom unusual and vibrant color palette
    scatter_color_start = '#FF4500'  # OrangeRed for starting color in scatter plot
    scatter_color_end = '#7B68EE'  # MediumSlateBlue for ending color in scatter plot
    height_hist_color = '#00BFFF'  # DeepSkyBlue for height histogram
    age_hist_color = '#FF1493'  # DeepPink for age histogram
    border_color = '#000000'  # Black border for clarity

    # Create subplots: Scatter Plot on top, Histograms below
    fig = make_subplots(
        rows=2, cols=1,  # Two rows, one column
        subplot_titles=("Height vs Age Distribution", "Height and Age Distribution"),
        shared_xaxes=False,  # Histograms should have different x-axis ranges
        vertical_spacing=0.1  # Add spacing between the rows
    )

    # Scatter plot for Height vs Age with custom gradient color
    fig.add_trace(
        go.Scatter(
            x=players_df_filtered['height_in_inches'],  # x-axis: height
            y=players_df_filtered['age'],  # y-axis: age
            mode='markers',
            marker=dict(
                size=players_df_filtered['height_in_inches'] / 3,  # Size proportional to height
                color=players_df_filtered['age'],  # Color based on age
                colorscale=[[0, scatter_color_start], [1, scatter_color_end]],  # Custom color gradient
                showscale=True,  # Show color scale
                line=dict(width=1, color=border_color)  # Black border for the bubbles
            ),
            text=players_df_filtered['nflId'],  # Show nflId on hover
            hovertemplate="NFL ID: %{text}<br>Height: %{x} inches<br>Age: %{y}<br><extra></extra>"
        ),
        row=1, col=1
    )

    # Histogram for Height Distribution with custom color
    fig.add_trace(
        go.Histogram(
            x=players_df_filtered['height_in_inches'],
            marker=dict(
                color=height_hist_color,  # DeepSkyBlue for height histogram
                line=dict(width=1, color=border_color)  # Border color
            ),
            opacity=0.75,
            nbinsx=20,  # Set number of bins for better resolution
            hovertemplate="Height: %{x} inches<br>Count: %{y}<br><extra></extra>"
        ),
        row=2, col=1
    )

    # Histogram for Age Distribution with custom color
    fig.add_trace(
        go.Histogram(
            x=players_df_filtered['age'],
            marker=dict(
                color=age_hist_color,  # DeepPink for age histogram
                line=dict(width=1, color=border_color)  # Border color
            ),
            opacity=0.75,
            nbinsx=20,  # Set number of bins for better resolution
            hovertemplate="Age: %{x} years<br>Count: %{y}<br><extra></extra>"
        ),
        row=2, col=1
    )

    # Update Layout
    fig.update_layout(
        title="NFL Players: Height vs Age & Distribution",
        title_x=0.5,  # Center the title
        template="plotly_white",  # Clean white background for contrast
        font=dict(size=14, color="black"),  # Font for readability
        height=900,  # Adjust plot height for better spacing
        showlegend=False,  # Hide legend
        margin=dict(t=50, b=50, l=50, r=50),
        plot_bgcolor='white'  # White background for clarity
    )

    # Customize axes
    fig.update_xaxes(title_text="Height (in inches)", row=2, col=1)
    fig.update_yaxes(title_text="Count", row=2, col=1)

    # Customize scatter plot axes
    fig.update_xaxes(title_text="Height (in inches)", row=1, col=1)
    fig.update_yaxes(title_text="Age", row=1, col=1)

    # Show the figure
    fig.show()
```

```python
# Call the function to display the plot
print("Height Vs Age Distribution Plot")
plot_height_age_distribution(players_df)
```

<div style="background-color: #D6EAF8; padding: 20px; border-radius: 15px; border: 2px solid #00008B; margin-top: 20px;">
    <p style="text-align: justify; font-family: 'Times New Roman', serif; font-size: 20px; color: #00008B; font-weight: bold;">
        Observations of Players' Age, Height, and NFLID:
    </p>
    <ul style="font-family: 'Georgia', serif; font-size: 14px; color: black; margin-left: 20px;">
        <li>The player with the <strong>age of 25</strong> and a birth date of <strong>1999-01-14</strong> has the highest count, and their NFLID is <strong>52443</strong>.</li>
        <li>The player with the lowest NFLID (<strong>25511</strong>) is <strong>47 years old</strong>.</li>
        <li>The player with the age of <strong>42</strong> has an NFLID of <strong>29550</strong>.</li>
        <li>By analyzing the data, it is determined that the <strong>maximum height</strong> of players is <strong>80 inches</strong>. Players with this height are generally <strong>38 years old</strong> and predominantly have an average age of <strong>29.44 years</strong>.</li>
        <li>Players with heights ranging from <strong>70-78 inches</strong> and ages of <strong>39-47 years</strong> have heights mostly in the ranges of <strong>4'6"</strong> to <strong>6'10"</strong>, with NFLIDs ranging from <strong>25511 to 33138</strong>.</li>
        <li>The player aged <strong>47</strong> with the maximum height of <strong>76 inches</strong> has an NFLID of <strong>25511</strong>. Players aged <strong>25</strong> with heights of <strong>68-79 inches</strong> have NFLIDs ranging from <strong>52440 to 52541</strong>.</li>
        <li>Players aged <strong>25-29 years</strong> have a total count of approximately <strong>707</strong>.</li>
        <li>Players with heights of <strong>75-79 inches</strong> have the highest age frequency distribution count of <strong>842</strong>, while players with heights of <strong>65-69 inches</strong> have a frequency distribution count of about <strong>70</strong>.</li>
    </ul>
</div>

# <p style="font-family: 'Amiri'; font-size: 3rem; color: Black; text-align: center; margin: 0; text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3); background-color: #cce5ff; padding: 20px; border-radius: 20px; border: 7px solid #2a71b8; width:95%">Explore weight on the basis of Age & nflid</p>

```python
def convert_weight(df):
    """
    Converts weight from a string (if in different units) to pounds.
    Adds 'weight_in_lbs', 'max_weight', 'mean_weight', 'age', and 'mean_age' columns.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame containing the 'weight' and 'age' columns.
    
    Returns:
    pd.DataFrame: Updated DataFrame with the converted weight and added columns.
    """
    print("Converting weight")
    
    # Helper function to convert weight to pounds (if needed)
    def weight_to_lbs(weight_str):
        if pd.notnull(weight_str):
            try:
                # Assuming weight is in lbs; if in other units, you can adjust here.
                return float(weight_str)
            except ValueError:
                return np.nan
        return np.nan

    # Apply the conversion to the 'weight' column
    df['weight_in_lbs'] = df['weight'].apply(weight_to_lbs)

    # Calculate max weight and mean weight
    max_weight = df['weight_in_lbs'].max()
    mean_weight = df['weight_in_lbs'].mean()

    # Calculate mean age
    mean_age = df['age'].mean()

    # Add the statistics as columns in the DataFrame
    df['max_weight'] = max_weight
    df['mean_weight'] = mean_weight
    df['mean_age'] = mean_age  # Assuming 'age' column already exists

    # Display the result for verification (show first few rows)
    display(df[['nflId', 'weight', 'weight_in_lbs', 'max_weight', 'mean_weight', 'age', 'mean_age']].head(10))  # Display first few rows to verify
    
    return df

# Apply the function (assuming 'players_df_filtered' has already been defined and contains 'weight' and 'age' columns)
players_df = convert_weight(players_df)
```

# <p style="font-family: 'Amiri'; font-size: 3rem; color: Black; text-align: center; margin: 0; text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3); background-color: #cce5ff; padding: 20px; border-radius: 20px; border: 7px solid #2a71b8; width:95%">Visualize weight on the basis of Age & nflid</p>

```python
def display_weight_vs_age_plot(players_df_filtered):
    # Custom vibrant color palette
    scatter_color_start = '#FF4500'  # OrangeRed for starting color in scatter plot
    scatter_color_end = '#7B68EE'  # MediumSlateBlue for ending color in scatter plot
    height_hist_color = '#00BFFF'  # DeepSkyBlue for height histogram
    age_hist_color = '#FF1493'  # DeepPink for age histogram
    border_color = '#000000'  # Black border for clarity

    # Create subplots: Scatter Plot for Weight vs Age with bar plot for Max and Mean Weight
    fig = make_subplots(
        rows=1, cols=1,
        subplot_titles=("Weight vs Age Distribution",),
        vertical_spacing=0.1
    )

    # Scatter plot for Weight vs Age with color gradient and larger markers
    fig.add_trace(
        go.Scatter(
            x=players_df_filtered['weight_in_lbs'],  # x-axis: weight
            y=players_df_filtered['age'],  # y-axis: age
            mode='markers',
            marker=dict(
                size=12,  # Increase size for better visibility
                color=players_df_filtered['weight_in_lbs'],  # Color based on weight
                colorscale=[[0, scatter_color_start], [1, scatter_color_end]],  # Custom gradient
                line=dict(width=2, color=border_color),  # Black border for points
                opacity=0.7,  # Add transparency for a softer look
                showscale=True,  # Show the color scale bar
            ),
            text=players_df_filtered['nflId'],  # Display NFL ID on hover
            hovertemplate="NFL ID: %{text}<br>Weight: %{x} lbs<br>Age: %{y}<br><extra></extra>",
            name="Weight vs Age Scatter"  # Name for the legend
        ),
        row=1, col=1
    )

    # Add a bar plot showing max and mean weights with legend entries and unique bar design
    fig.add_trace(
        go.Bar(
            x=['Max Weight', 'Mean Weight'],
            y=[players_df_filtered['max_weight'].iloc[0], players_df_filtered['mean_weight'].iloc[0]],  # Using iloc[0] to get the scalar values
            marker=dict(color=['#FFD700', '#FF69B4'], opacity=0.7),  # Gold for max weight, HotPink for mean weight
            width=0.4,  # Adjust bar width
            name='Weight Stats',  # Name for the legend
            text=['Max Weight: ' + str(players_df_filtered['max_weight'].iloc[0]),
                  'Mean Weight: ' + str(players_df_filtered['mean_weight'].iloc[0])],
            hoverinfo='text',
            showlegend=True,  # Show in legend
        ),
        row=1, col=1
    )

    # Update layout with enhanced styling for attractiveness and uniqueness
    fig.update_layout(
        title="NFL Players: Weight vs Age Distribution with Weight Stats",
        title_x=0.5,  # Center the title
        font=dict(size=16, color="black", family="Arial, sans-serif"),  # Bold and clean font for readability
        height=800,  # Adjust plot height for better spacing
        margin=dict(t=50, b=50, l=50, r=50),
        plot_bgcolor='white',  # White background for the plot
        paper_bgcolor='#F0F8FF',  # Light blue paper background for a unique touch
        showlegend=True,  # Show the legend for all traces
        legend=dict(
            x=0.75,  # Adjust position of legend
            y=0.95,
            bgcolor='rgba(255, 255, 255, 0.7)',  # Semi-transparent background for the legend
            bordercolor='black',
            borderwidth=1,
            font=dict(size=12)  # Smaller, cleaner legend font
        )
    )

    # Customize axes with bold fonts and larger ticks
    fig.update_xaxes(
        title_text="Weight (in lbs)",
        showgrid=True,
        tickfont=dict(size=12, color="black"),
        title_font=dict(size=14, color="black"),
        gridcolor="#DCDCDC",  # Light gray gridlines
        showline=True,  # Show axis line
        linewidth=2,  # Thicker axis line
        linecolor='black'  # Black axis line
    )
    fig.update_yaxes(
        title_text="Age",
        showgrid=True,
        tickfont=dict(size=12, color="black"),
        title_font=dict(size=14, color="black"),
        range=[20, 50],  # Set the Y-axis range from 20 to 50
        gridcolor="#DCDCDC",  # Light gray gridlines
        showline=True,  # Show axis line
        linewidth=2,  # Thicker axis line
        linecolor='black'  # Black axis line
    )

    # Show the figure
    fig.show()
```

```python
# players_df = convert_weight(players_df)
print("Weight Vs Age Distribution Plot")
display_weight_vs_age_plot(players_df)
```

<div style="background-color: #D6EAF8; padding: 20px; border-radius: 15px; border: 2px solid #00008B; margin-top: 20px;">
    <p style="text-align: justify; font-family: 'Times New Roman', serif; font-size: 20px; color: #00008B; font-weight: bold;">
        Observations of Players' Weight and Age Data:
    </p>
    <ul style="font-family: 'Georgia', serif; font-size: 14px; color: black; margin-left: 20px;">
        <li>By analyzing the most frequent data, it is determined that players with the <strong>maximum weight</strong> of <strong>328 lbs</strong> are aged <strong>42</strong> and have an <strong>NFLID</strong> of <strong>29550</strong>.</li>
        <li>Players with the <strong>minimum weight</strong> of <strong>175 lbs</strong> are aged <strong>38</strong> and have an <strong>NFLID</strong> of <strong>33130</strong>.</li>
        <li>The <strong>mean age</strong> of players according to this dataset is <strong>29.44</strong>, while the <strong>mean weight</strong> of players is approximately <strong>245.774308 lbs</strong>.</li>
        <li>Players with the <strong>maximum age</strong> of <strong>47</strong> have a weight of <strong>225 lbs</strong> and an <strong>NFLID</strong> of <strong>25511</strong>.</li>
        <li>According to this dataset, players with the <strong>mean age</strong> of <strong>29.44</strong> have a weight of <strong>245 lbs</strong> and an <strong>NFLID range</strong> of approximately <strong>44985-48198</strong>.</li>
    </ul>
</div>

# <p style="font-family: 'Amiri'; font-size: 2.5rem; color: Black; text-align: center; margin: 0; text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3); background-color: #cce5ff; padding: 20px; border-radius: 20px; border: 7px solid #2a71b8; width:95%">Explore College Name on the basis of nflid</p>

```python
def calculate_college_name_length(players_df):
    print("Calculating College Name Length")
    players_df['college_name_length'] = players_df['collegeName'].apply(lambda x: len(str(x)) if pd.notnull(x) else np.nan)
    display(players_df[['nflId', 'collegeName', 'college_name_length']].head(10))  # Display first few rows to verify
    return players_df

# Apply the function
players_df = calculate_college_name_length(players_df)
```

# <p style="font-family: 'Amiri'; font-size: 2.5rem; color: Black; text-align: center; margin: 0; text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3); background-color: #cce5ff; padding: 20px; border-radius: 20px; border: 7px solid #2a71b8; width:95%">Visualize College Name on the basis of nflid & Age</p>

```python
def display_college_name_vs_age(players_df_filtered):
    # Custom vibrant color palette
    scatter_color_start = '#FF4500'  # OrangeRed for starting color in scatter plot
    scatter_color_end = '#7B68EE'  # MediumSlateBlue for ending color in scatter plot
    border_color = '#000000'  # Black border for clarity

    # Create a unique and attractive scatter plot showing college name length vs age
    fig = go.Figure()

    # Scatter plot for College Name Length vs Age (only markers, no line)
    fig.add_trace(
        go.Scatter(
            x=players_df_filtered['college_name_length'],  # x-axis: College Name Length
            y=players_df_filtered['age'],  # y-axis: Age
            mode='markers',  # Only markers, no lines
            marker=dict(
                size=10,  # Marker size
                color=players_df_filtered['college_name_length'],  # Color based on college name length
                colorscale=[[0, scatter_color_start], [1, scatter_color_end]],  # Custom gradient
                showscale=True,  # Show color scale for context
                line=dict(width=2, color=border_color)  # Black border around markers for better visibility
            ),
            text=players_df_filtered['nflId'],  # Display NFL ID on hover
            hovertemplate="NFL ID: %{text}<br>College Name Length: %{x}<br>Age: %{y}<br><extra></extra>"
        )
    )

    # Update layout for a more attractive chart
    fig.update_layout(
        title="College Name Length vs Age for NFL Players",
        title_x=0.5,  # Center the title
        font=dict(size=14, color="black"),  # Font for readability on white background
        height=800,  # Adjust plot height for better spacing
        showlegend=False,  # Disable legend as it's not needed
        margin=dict(t=50, b=50, l=50, r=50),
        plot_bgcolor='white'  # White background for the plot
    )

    # Customize axes with vibrant colors for titles and gridlines
    fig.update_xaxes(
        title_text="College Name Length",
        showgrid=True,
        gridcolor="#DCDCDC",  # Light gray gridlines
        title_font=dict(size=16, color='black'),
        tickfont=dict(size=12, color='black')
    )
    fig.update_yaxes(
        title_text="Age",
        showgrid=True,
        gridcolor="#DCDCDC",  # Light gray gridlines
        title_font=dict(size=16, color='black'),
        tickfont=dict(size=12, color='black')
    )

    # Show the figure
    fig.show()
```

```python
# Call the function to display the chart
print("College Name Vs Age Distribution Visualization")
display_college_name_vs_age(players_df)
```

# <p style="font-family: 'Amiri'; font-size: 2.5rem; color: Black; text-align: center; margin: 0; text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3); background-color: #cce5ff; padding: 20px; border-radius: 20px; border: 7px solid #2a71b8; width:95%">Explore College Name on the basis of Mean Age</p>

```python
def calculate_college_name_length(df):
    """
    Calculates the length of the college name and adds it as a new column.
    Also includes the 'age', 'mean_age', and 'college_name_count' columns in the output.
    Imputes missing values (NaN) in 'college_name_length', 'age', and 'mean_age'.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame containing the 'collegeName' and 'age' columns.
    
    Returns:
    pd.DataFrame: Updated DataFrame with the 'college_name_length', 'age', 'mean_age', and 'college_name_count' columns.
    """
    print("Calculating College Name Length")
    
    # Calculate the length of the college name
    df['college_name_length'] = df['collegeName'].apply(lambda x: len(str(x)) if pd.notnull(x) else np.nan)
    
    # Calculate the mean age if the 'age' column exists
    df['mean_age'] = df['age'].mean() if 'age' in df.columns else np.nan
    
    # Count occurrences of each collegeName and add as a new column
    df['college_name_count'] = df['collegeName'].map(df['collegeName'].value_counts())
    
    # Sort the DataFrame by 'college_name_length' and 'college_name_count' in ascending order
    df_sorted = df.sort_values(by=['college_name_length', 'college_name_count'], ascending=[False, False])
    
    # Impute missing values:
    # Impute missing 'college_name_length' with the mean length of the college names
    df_sorted['college_name_length'].fillna(df_sorted['college_name_length'].mean(), inplace=True)
    
    # Impute missing 'age' with the mean age of all players
    df_sorted['age'].fillna(df_sorted['age'].mean(), inplace=True)
    
    # 'mean_age' should be constant across the dataset, but we will impute it just in case
    df_sorted['mean_age'].fillna(df_sorted['age'].mean(), inplace=True)
    
    # Display first few rows to verify
    display(df_sorted[['nflId', 'collegeName', 'college_name_length', 'college_name_count', 'age', 'mean_age']].head(10))  # Display first few rows to verify
    
    return df_sorted

# Apply the function (assuming 'players_df_filtered' has already been defined and contains 'collegeName' and 'age' columns)
players_df_filtered = calculate_college_name_length(players_df)
```

# <p style="font-family: 'Amiri'; font-size: 2.5rem; color: Black; text-align: center; margin: 0; text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3); background-color: #cce5ff; padding: 20px; border-radius: 20px; border: 7px solid #2a71b8; width:95%">Visualize College Name on the basis of Age & Frequency Distribution</p>

```python
def display_nfl_player_visualizations(players_df_filtered):
    # Create subplots for the visualizations
    fig = sp.make_subplots(
        rows=2, cols=2,
        subplot_titles=("Histogram of College Name Length", 
                        "Scatter: College Name Length vs Age", 
                        "Bar: College Name Frequency Distribution Visualization", 
                        "Box Plot: Age Distribution by College Name Length"),
        vertical_spacing=0.15,  # Increased vertical spacing between subplots
        horizontal_spacing=0.15  # Increased horizontal spacing between subplots
    )

    # Plot 1: Histogram of College Name Length
    fig.add_trace(
        go.Histogram(
            x=players_df_filtered['college_name_length'],
            nbinsx=20,
            histnorm='percent',
            marker=dict(color='#FF4500', line=dict(color='black', width=1)),
            opacity=0.85,
            hovertemplate="College Name Length: %{x}<br>Percentage: %{y}%<br><extra></extra>",
            name="College Name Length Histogram"
        ), row=1, col=1
    )

    # Plot 2: Scatter plot of College Name Length vs Age
    fig.add_trace(
        go.Scatter(
            x=players_df_filtered['college_name_length'],
            y=players_df_filtered['age'],
            mode='markers',
            marker=dict(
                size=10,
                color=players_df_filtered['college_name_length'],
                colorscale=[[0, '#FF4500'], [1, '#7B68EE']],
                line=dict(width=2, color='black'),
                opacity=0.85
            ),
            text=players_df_filtered['collegeName'],
            hovertemplate="College Name: %{text}<br>College Name Length: %{x}<br>Age: %{y}<br><extra></extra>",
            name="College Name Length vs Age"
        ), row=1, col=2
    )

    # Plot 3: Bar plot of College Name Count
    college_name_counts = players_df_filtered['collegeName'].value_counts()[:10]
    colors = ['#FF6347', '#FFD700', '#7B68EE', '#32CD32', '#8A2BE2', '#DC143C', '#FF8C00', '#00CED1', '#D2691E', '#ADFF2F']

    fig.add_trace(
        go.Bar(
            x=college_name_counts.index,
            y=college_name_counts.values,
            marker=dict(color=colors),
            opacity=0.85,
            hovertemplate="College Name: %{x}<br>Count: %{y}<br><extra></extra>",
            name="Top Colleges Frequency Distribution",
            text=college_name_counts.values,
            textposition='outside',
            textfont=dict(size=14, color='black')
        ), row=2, col=1
    )

    # Plot 4: Box Plot of Age distribution by College Name Length
    fig.add_trace(
        go.Box(
            x=players_df_filtered['college_name_length'],
            y=players_df_filtered['age'],
            boxmean=True,
            marker=dict(color='#00BFFF'),
            line=dict(width=2, color='black'),
            hovertemplate="College Name Length: %{x}<br>Age: %{y}<br><extra></extra>",
            name="Age Distribution by College Name Length"
        ), row=2, col=2
    )

    # Update layout
    fig.update_layout(
        title="Visualizations of NFL Player on the Basis of College Name & its Distribution",
        title_x=0.5,
        font=dict(size=16, color="black"),
        height=1200,
        width=1400,
        showlegend=True,
        legend=dict(
            x=1.05,
            y=0.5,
            xanchor='left',
            yanchor='middle',
            orientation='v',
            font=dict(size=14),
            title='Legend'
        ),
        plot_bgcolor='white',
        margin=dict(t=80, b=80, l=80, r=80),
        xaxis=dict(title='College Name Length', tickangle=45),
        yaxis=dict(title='Percentage'),
        xaxis2=dict(title='College Name Length', tickangle=45),
        yaxis2=dict(title='Age'),
        xaxis3=dict(title='College Name', tickangle=45),
        yaxis3=dict(title='Count'),
        xaxis4=dict(title='College Name Length', tickangle=45),
        yaxis4=dict(title='Age')
    )

    # Show the figure
    fig.show()
```

```python
# Call the function to display the chart
print("Visualizations of NFL Player on the Basis of College Name & its Distribution")
display_nfl_player_visualizations(players_df)
```

<div style="background-color: #D6EAF8; padding: 20px; border-radius: 15px; border: 2px solid #00008B; margin-top: 20px;">
    <p style="text-align: justify; font-family: 'Times New Roman', serif; font-size: 20px; color: #00008B; font-weight: bold;">
        Observations of Players' College Data:
    </p>
    <ul style="font-family: 'Georgia', serif; font-size: 14px; color: black; margin-left: 20px;">
        <li>By analyzing the most frequent players' data, it is determined that the <strong>maximum frequency count</strong> is of <strong>Boston College</strong>, with an NFLID of <strong>33084</strong>, having a frequency distribution count of approximately <strong>14</strong>.</li>
        <li>Players with a <strong>college name length of 4</strong> are aged between <strong>25-40 years</strong>.</li>
        <li>In the dataset, the <strong>maximum college name length</strong> is <strong>24</strong>, associated with a player aged <strong>24</strong>, having an NFLID of <strong>44877</strong>.</li>
        <li>The <strong>minimum college name length</strong> is <strong>4</strong>, associated with a player aged <strong>25</strong>, having an NFLID of <strong>52627</strong>.</li>
        <li>The <strong>most frequent college name lengths</strong> are <strong>6-7 characters</strong>, making up approximately <strong>20.21%</strong> of the dataset.</li>
        <li>The <strong>least frequent college name lengths</strong> are <strong>24-25 characters</strong>, making up approximately <strong>0.24%</strong> of the dataset.</li>
        <li><strong>Alabama College</strong> has the <strong>maximum frequency distribution count</strong> of about <strong>59</strong>, according to the players' data.</li>
        <li>Players with a <strong>maximum age of 47</strong> have a college name frequency distribution count of about <strong>8</strong>.</li>
    </ul>
</div>

# <p style="font-family: 'Amiri'; font-size: 2.5rem; color: Black; text-align: center; margin: 0; text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3); background-color: #cce5ff; padding: 20px; border-radius: 20px; border: 7px solid #2a71b8; width:95%">Explore Age Range on the basis of nflid</p>

```python
def classify_age_groups(df):
    print("Classifying Age Groups")
    
    # Define custom bins for the age range from 25 to 47
    bins = [25, 30, 35, 40, 45, 47]
    labels = ['25-30', '30-35', '35-40', '40-45', '45-47']
    
    # Apply pd.cut() to classify ages based on the custom bins
    df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels, right=True)  # Set right=True
    
    # Count the number of players in each age group
    age_group_counts = df['age_group'].value_counts().sort_index()  # Count occurrences of each age group
    
    # Merge the counts back into the DataFrame
    df['age_group_count'] = df['age_group'].map(age_group_counts)
    
    # Display the result for verification
    display(df[['nflId', 'age', 'age_group', 'age_group_count']].head(10))  # Display first few rows to verify
    
    return df

# Apply the function (assuming 'players_df_filtered' has already been defined and contains 'age' column)
players_df_filtered = classify_age_groups(players_df_filtered)
```

# <p style="font-family: 'Amiri'; font-size: 2.5rem; color: Black; text-align: center; margin: 0; text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3); background-color: #cce5ff; padding: 20px; border-radius: 20px; border: 7px solid #2a71b8; width:95%">Visualize Age Range on the basis of nflid</p>

```python
def display_age_group_analysis(players_df_filtered):
    # Custom colors
    scatter_color_end = 'red'  # Red for the line chart color
    height_hist_color = '#00BFFF'  # DeepSkyBlue for height histogram

    # Create subplots for Line Chart and Histogram
    fig = sp.make_subplots(
        rows=1, cols=2, 
        subplot_titles=("Line Chart of Age Group Counts", "Histogram of Age Group Distribution"),
        column_widths=[0.5, 0.5],  # Set equal width for both plots
        horizontal_spacing=0.15  # Increase space between the plots
    )

    # Line Chart: Age Group Counts Over Time
    age_group_counts = players_df_filtered['age_group'].value_counts().sort_index()  # Get counts and sort by index

    fig.add_trace(
        go.Scatter(
            x=age_group_counts.index,  # Age groups
            y=age_group_counts.values,  # Counts of players in each group
            mode='lines+markers+text',  # Line chart with markers and text
            line=dict(color=scatter_color_end, width=2),  # Apply red color for the line
            marker=dict(size=8, color=scatter_color_end),  # Apply red color for the markers
            text=age_group_counts.values,  # Display count values above markers
            textposition='top center',  # Position of the text
            textfont=dict(size=10),  # Smaller font size
            name="Age Group Frequency Distribution",
            hovertemplate="Age Group: %{x}<br>Count: %{y}<br><extra></extra>"  # Hover info
        ), row=1, col=1
    )

    # Histogram: Age Group Distribution
    fig.add_trace(
        go.Histogram(
            x=players_df_filtered['age_group'],  # Age group data
            histnorm='percent',  # Normalize to percentage
            marker=dict(color=height_hist_color, line=dict(color='black', width=1)),  # Apply the DeepSkyBlue color for histogram bars
            opacity=0.75,  # Set opacity to see overlapping elements
            name="Age Group Percentage Distribution",
            hovertemplate="Age Group: %{x}<br>Percentage: %{y}%<br><extra></extra>"  # Hover info
        ), row=1, col=2
    )

    # Add values above histogram bars
    age_group_percent = players_df_filtered['age_group'].value_counts(normalize=True).sort_index() * 100  # Calculate percentage
    for i, count in enumerate(age_group_percent):
        fig.add_annotation(
            x=age_group_percent.index[i],
            y=count + 1,  # Offset the text slightly above the bar
            text=f"{count:.1f}%",  # Format the percentage value
            showarrow=False,
            font=dict(size=10),  # Small font size for text
            align='center',
            row=1, col=2
        )

    # Update layout
    fig.update_layout(
        title="Age Group Analysis for NFL Players",
        title_x=0.5,
        font=dict(size=16, color="black"),  # Increased font size for better readability
        height=600,  # Height of the plot
        width=1200,  # Width of the plot
        showlegend=True,  # Show legend for the plot
        plot_bgcolor='white',
        margin=dict(t=80, b=80, l=80, r=80),  # Margin to prevent clipping
        xaxis=dict(title='Age Group'),
        yaxis=dict(title='Count'),
        xaxis2=dict(title='Age Group'),
        yaxis2=dict(title='Percentage')  # Y-axis for the histogram
    )

    # Show the figure
    fig.show()
```

```python
# Call the function to display the visualization
("Age Group Distribution Visualization")
display_age_group_analysis(players_df_filtered)
```

# <p style="font-family: 'Amiri'; font-size: 2.5rem; color: Black; text-align: center; margin: 0; text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3); background-color: #cce5ff; padding: 20px; border-radius: 20px; border: 7px solid #2a71b8; width:95%">Explore Age Range on the basis of nflid & College Name</p>

```python
def classify_age_groups(df):
    print("Classifying Age Groups")
    
    # Define custom bins for the age range from 25 to 47
    bins = [25, 30, 35, 40, 45, 47]
    labels = ['25-30', '30-35', '35-40', '40-45', '45-47']
    
    # Apply pd.cut() to classify ages based on the custom bins
    df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels, right=True)  # Set right=True
    
    # Calculate value counts for age_group and collegeName to display them
    age_group_count = df['age_group'].value_counts().reset_index(name='age_group_count')
    value_counts_college = df['collegeName'].value_counts().reset_index(name='college_count')
    
    # Rename columns for proper merge
    value_counts_college.columns = ['collegeName', 'college_count']
    age_group_count.columns = ['age_group', 'age_group_count']  # Ensure proper naming
    
    # Merge value counts with the original DataFrame for better visualization
    df_with_counts = df.merge(value_counts_college, on='collegeName', how='left')
    df_with_counts = df_with_counts.merge(age_group_count, on='age_group', how='left')
    
    # Rename the columns if they are ambiguous due to merge
    df_with_counts = df_with_counts.rename(columns={'age_group_count_x': 'age_group_count', 'age_group_count_y': 'age_group_count'})
    
    # Display the result for verification (show first 10 rows)
    display(df_with_counts[['nflId', 'age', 'age_group', 'collegeName', 'college_count']].head(10))  # Display first few rows to verify
    
    # Calculate the count of players per college name and age group
    college_age_group_stats = df.groupby(['collegeName', 'age_group']).agg(
        count=('nflId', 'size'),
        max_age=('age', 'max')
    ).reset_index()
    
    # Sort by count to display maximum counts first
    college_age_group_stats_sorted = college_age_group_stats.sort_values(by='count', ascending=False)
    
    # Display the sorted count and max age for players from each college and age group
    display(college_age_group_stats_sorted.head(20))  # Top 20 results sorted by count
    
    return df_with_counts

# Apply the function
players_df_filtered = classify_age_groups(players_df_filtered)
```

# <p style="font-family: 'Amiri'; font-size: 2.5rem; color: Black; text-align: center; margin: 0; text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3); background-color: #cce5ff; padding: 20px; border-radius: 20px; border: 7px solid #2a71b8; width:95%">Visualize Age Range on the basis of nflid & College Name</p>

```python
def display_age_and_college_analysis(players_df_filtered):
    # Create subplots with adjusted row_heights to minimize white space below the second row
    fig = sp.make_subplots(
        rows=3, cols=2, 
        subplot_titles=("Age Group Distribution", "College Name Distribution", "Age Group vs College Name", "Top College and Age Group Stats"),
        column_widths=[0.5, 0.5],  # Equal column widths
        row_heights=[0.5, 0.5, 0.1],  # Reduced height for the last empty row
        horizontal_spacing=0.15,  # Space between columns
        vertical_spacing=0.2,  # Space between rows
        specs=[[{'type': 'bar'}, {'type': 'bar'}],  # Bar plots in the first row
               [{'type': 'scatter'}, {'type': 'table'}],  # Scatter plot in the second row, table in the second column
               [None, None]]  # No plots in the third row
    )

    # 1. Bar Plot: Age Group Distribution
    age_group_counts = players_df_filtered['age_group'].value_counts().sort_index()

    fig.add_trace(
        go.Bar(
            x=age_group_counts.index,
            y=age_group_counts.values,
            name="Age Group Distribution",
            marker=dict(color='lightblue', line=dict(color='black', width=1)),
            text=age_group_counts.values,
            textposition='outside',
            textfont=dict(size=12),
            hovertemplate="Age Group: %{x}<br>Count: %{y}<br><extra></extra>"
        ), row=1, col=1
    )
    fig.update_xaxes(title_text="Age Group", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=1, col=1)

    # 2. Bar Plot: College Name Distribution
    college_counts = players_df_filtered['collegeName'].value_counts().sort_values(ascending=False).head(10)

    fig.add_trace(
        go.Bar(
            x=college_counts.index,
            y=college_counts.values,
            name="College Name Distribution",
            marker=dict(color='darkorange', line=dict(color='black', width=1)),
            text=college_counts.values,
            textposition='outside',
            textfont=dict(size=12),
            hovertemplate="College: %{x}<br>Count: %{y}<br><extra></extra>"
        ), row=1, col=2
    )
    fig.update_xaxes(title_text="College Name", row=1, col=2)
    fig.update_yaxes(title_text="Count", row=1, col=2)

    # 3. Scatter Plot: Age Group vs College Name
    college_age_group_counts = players_df_filtered.groupby(['collegeName', 'age_group']).size().reset_index(name='count')

    fig.add_trace(
        go.Scatter(
            x=college_age_group_counts['collegeName'],
            y=college_age_group_counts['age_group'],
            mode='markers',
            marker=dict(
                size=college_age_group_counts['count'],
                color=college_age_group_counts['count'],
                colorscale='Viridis',
                showscale=False  # Set this to False to remove the color bar
            ),
            text=college_age_group_counts['count'],
            textposition='top center',
            hovertemplate="College: %{x}<br>Age Group: %{y}<br>Count: %{text}<br><extra></extra>"
        ), row=2, col=1
    )
    fig.update_xaxes(title_text="College Name", row=2, col=1)
    fig.update_yaxes(title_text="Age Group", row=2, col=1)

    # 4. Table: Top College and Age Group Stats
    top_college_age_group_stats = players_df_filtered.groupby(['collegeName', 'age_group']).agg(
        count=('nflId', 'size'),
        max_age=('age', 'max')
    ).reset_index()

    top_college_age_group_stats_sorted = top_college_age_group_stats.sort_values(by='count', ascending=False).head(10)

    fig.add_trace(
        go.Table(
            header=dict(values=["College Name", "Age Group", "Player Count", "Max Age"]),
            cells=dict(values=[top_college_age_group_stats_sorted['collegeName'],
                               top_college_age_group_stats_sorted['age_group'],
                               top_college_age_group_stats_sorted['count'],
                               top_college_age_group_stats_sorted['max_age']])
        ), row=2, col=2
    )

    # Update layout
    fig.update_layout(
        title="Age Group and College Distribution Analysis for NFL Players",
        title_x=0.5,
        font=dict(size=16, color="black"),
        height=1300,  # Adjust height for better spacing
        width=1500,
        showlegend=True,
        plot_bgcolor='white',
        margin=dict(t=80, b=80, l=80, r=80),
    )

    # Show the figure
    fig.show()
```

```python
# Call the function to display the visualization
("Visualize Age Group and College Names Distribution")
display_age_and_college_analysis(players_df_filtered)
```

<div style="background-color: #D6EAF8; padding: 20px; border-radius: 15px; border: 2px solid #00008B; margin-top: 20px;">
    <p style="text-align: justify; font-family: 'Times New Roman', serif; font-size: 20px; color: #00008B; font-weight: bold;">
        Observations of Players' College Data:
    </p>
    <ul style="font-family: 'Georgia', serif; font-size: 14px; color: black; margin-left: 20px;">
        <li>By analyzing the most frequent players' data, it is determined that the <strong>maximum frequency count</strong> is of <strong>Boston College</strong>, with an NFLID of <strong>33084</strong>, having a frequency distribution count of approximately <strong>14</strong>.</li>
        <li>Players with a <strong>college name length of 4</strong> are aged between <strong>25-40 years</strong>.</li>
        <li>In the dataset, the <strong>maximum college name length</strong> is <strong>24</strong>, associated with a player aged <strong>24</strong>, having an NFLID of <strong>44877</strong>.</li>
        <li>The <strong>minimum college name length</strong> is <strong>4</strong>, associated with a player aged <strong>25</strong>, having an NFLID of <strong>52627</strong>.</li>
        <li>The <strong>most frequent college name lengths</strong> are <strong>6-7 characters</strong>, making up approximately <strong>20.21%</strong> of the dataset.</li>
        <li>The <strong>least frequent college name lengths</strong> are <strong>24-25 characters</strong>, making up approximately <strong>0.24%</strong> of the dataset.</li>
        <li><strong>Alabama College</strong> has the <strong>maximum frequency distribution count</strong> of about <strong>59</strong>, according to the players' data.</li>
        <li>Players with a <strong>maximum age of 47</strong> have a college name frequency distribution count of about <strong>8</strong>.</li>
    </ul>
</div>

# <p style="font-family: 'Amiri'; font-size: 2.5rem; color: Black; text-align: center; margin: 0; text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3); background-color: #cce5ff; padding: 20px; border-radius: 20px; border: 7px solid #2a71b8; width:95%">Lets take Comprehensive Insights of College Name & Display Name</p>

```python
# Group the data by `displayName` and calculate count, mean, and max for Display Name
players_df_analysis_display = players_df.groupby('displayName').agg(
    count=('weight', 'count'),
    mean=('weight', 'mean'),
    max=('weight', 'max')
)

# Sort by count and take top 20 display names
players_df_analysis_display = players_df_analysis_display.sort_values(by='count', ascending=False).head(20)

# Reset index to display all columns in one row
players_df_analysis_display.reset_index(inplace=True)

# Group the data by `collegeName` and calculate count, mean, and max for College Name
players_df_analysis_college = players_df.groupby('collegeName').agg(
    count=('weight', 'count'),
    mean=('weight', 'mean'),
    max=('weight', 'max')
)

# Sort by count and take top 20 college names
players_df_analysis_college = players_df_analysis_college.sort_values(by='count', ascending=False).head(20)

# Reset index to display all columns in one row
players_df_analysis_college.reset_index(inplace=True)

# Display the analysis results for Display Name
print("Top Display Names - Analysis:")
display(players_df_analysis_display)

# Display the analysis results for College Name
print("\nTop College Names Analysis:")
display(players_df_analysis_college)
```

# <p style="font-family: 'Amiri'; font-size: 2.5rem; color: Black; text-align: center; margin: 0; text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3); background-color: #cce5ff; padding: 20px; border-radius: 20px; border: 7px solid #2a71b8; width:95%">Lets Visualize the College Name & Display Name Distribution</p>

```python
def display_numeric_and_categorical_column_analysis(players_df_analysis_display, players_df_analysis_college):
    # Create subplots
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            "Frequency Distribution of Display Name",
            "Distribution of Display Name on the Basis of Mean Values",
            "Distribution of Display Name on the Basis of Max Values",
            "Summary Table for Display Name and College Name",
            "Frequency Distribution of College Name",
            "Distribution of College Name on the Basis of Mean Values"
        ),
        specs=[[{"type": "bar"}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "table"}],
               [{"type": "bar"}, {"type": "bar"}]],
        vertical_spacing=0.25
    )

    # 1. Bar plot for count (Display Name)
    fig.add_trace(
        go.Bar(
            x=players_df_analysis_display.index,
            y=players_df_analysis_display['count'],
            name='Count',
            marker=dict(color='lightblue', line=dict(color='black', width=1)),
            hovertemplate='<b>Display Name:</b> %{x}<br>' +
                          '<b>Count:</b> %{y}<br>' +
                          '<extra></extra>'
        ),
        row=1, col=1
    )
    fig.update_xaxes(title_text="Display Name", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=1, col=1)

    # 2. Bar plot for mean (Display Name)
    fig.add_trace(
        go.Bar(
            x=players_df_analysis_display.index,
            y=players_df_analysis_display['mean'],
            name='Mean',
            marker=dict(color='orange', line=dict(color='black', width=1)),
            text=[f"{val:.2f}" for val in players_df_analysis_display['mean']],
            textposition='outside',
            hovertemplate='<b>Display Name:</b> %{x}<br>' +
                          '<b>Mean Value:</b> %{y:.2f}<br>' +
                          '<extra></extra>'
        ),
        row=1, col=2
    )
    fig.update_xaxes(title_text="Display Name", row=1, col=2)
    fig.update_yaxes(title_text="Mean Value", row=1, col=2)

    # 3. Bar plot for max (Display Name)
    fig.add_trace(
        go.Bar(
            x=players_df_analysis_display.index,
            y=players_df_analysis_display['max'],
            name='Max',
            marker=dict(color='green', line=dict(color='black', width=1)),
            text=[f"{val:.2f}" for val in players_df_analysis_display['max']],
            textposition='outside',
            hovertemplate='<b>Display Name:</b> %{x}<br>' +
                          '<b>Max Value:</b> %{y:.2f}<br>' +
                          '<extra></extra>'
        ),
        row=2, col=1
    )
    fig.update_xaxes(title_text="Display Name", row=2, col=1)
    fig.update_yaxes(title_text="Max Value", row=2, col=1)

    # 4. Table for Display Name summary statistics (including College Name)
    fig.add_trace(
        go.Table(
            header=dict(
                values=["Metric", "Display Name", "College Name", "Count", "Mean", "Max"],
                fill_color='paleturquoise',
                align='center'
            ),
            cells=dict(
                values=[
                    players_df_analysis_display.index,
                    players_df_analysis_display.index,  # Display Name
                    players_df_analysis_college.index,  # College Name (Same index for each metric)
                    players_df_analysis_display['count'],
                    players_df_analysis_display['mean'],
                    players_df_analysis_display['max']
                ],
                fill_color='lavender',
                align='center'
            )
        ),
        row=2, col=2
    )

    # 5. Bar plot for count (College Name)
    fig.add_trace(
        go.Bar(
            x=players_df_analysis_college.index,
            y=players_df_analysis_college['count'],
            name='Count',
            marker=dict(color='lightblue', line=dict(color='black', width=1)),
            hovertemplate='<b>College Name:</b> %{x}<br>' +
                          '<b>Count:</b> %{y}<br>' +
                          '<extra></extra>'
        ),
        row=3, col=1
    )
    fig.update_xaxes(title_text="College Name", row=3, col=1)
    fig.update_yaxes(title_text="Count", row=3, col=1)

    # 6. Bar plot for mean (College Name)
    fig.add_trace(
        go.Bar(
            x=players_df_analysis_college.index,
            y=players_df_analysis_college['mean'],
            name='Mean',
            marker=dict(color='orange', line=dict(color='black', width=1)),
            text=[f"{val:.2f}" for val in players_df_analysis_college['mean']],
            textposition='outside',
            hovertemplate='<b>College Name:</b> %{x}<br>' +
                          '<b>Mean Value:</b> %{y:.2f}<br>' +
                          '<extra></extra>'
        ),
        row=3, col=2
    )
    fig.update_xaxes(title_text="College Name", row=3, col=2)
    fig.update_yaxes(title_text="Mean Value", row=3, col=2)

    # Update layout
    fig.update_layout(
        title="Visualize the Distribution of Player Display Names and College Names",
        title_x=0.5,
        height=1200,
        width=1200,
        showlegend=False,
        plot_bgcolor='white'
    )

    # Show the figure
    fig.show()

# Group the data by `displayName` and calculate count, mean, and max for Display Name
players_df_analysis_display = players_df.groupby('displayName').agg(
    count=('weight', 'count'),
    mean=('weight', 'mean'),
    max=('weight', 'max')
)

# Sort by count and take top 20 display names
players_df_analysis_display = players_df_analysis_display.sort_values(by='count', ascending=False).head(20)

# Group the data by `collegeName` and calculate count, mean, and max for College Name
players_df_analysis_college = players_df.groupby('collegeName').agg(
    count=('weight', 'count'),
    mean=('weight', 'mean'),
    max=('weight', 'max')
)
```

```python
# Sort by count and take top 20 college names
players_df_analysis_college = players_df_analysis_college.sort_values(by='count', ascending=False).head(20)

# Call the function to display the analysis for top 20 display names and college names
display_numeric_and_categorical_column_analysis(players_df_analysis_display, players_df_analysis_college)
```

# <p style="font-family: 'Amiri'; font-size: 2.5rem; color: Black; text-align: center; margin: 0; text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3); background-color: #cce5ff; padding: 20px; border-radius: 20px; border: 7px solid #2a71b8; width:95%">Lets take Comprehensive Insights of Player Metrics</p>

```python
def analyze_data(df):
    # Select numeric columns for analysis (exclude categorical columns)
    numeric_columns = df.select_dtypes(include=[float, int]).columns

    # Initialize a dictionary to hold the count, mean, and max values
    analysis_dict = {
        'count': df[numeric_columns].count(),
        'mean': df[numeric_columns].mean(),
        'max': df[numeric_columns].max()
    }

    # Convert the dictionary to a DataFrame
    analysis_df = pd.DataFrame(analysis_dict)

    # Display the resulting analysis
    display(analysis_df)
    return analysis_df

# Apply the function to the transformed dataset
players_df_analysis = analyze_data(players_df)  # Make sure players_df is defined
```

<div style="background-color: #D6EAF8; padding: 20px; border-radius: 15px; border: 2px solid #00008B; margin-top: 20px;">
    <p style="text-align: justify; font-family: 'Times New Roman', serif; font-size: 20px; color: #00008B; font-weight: bold;">
        Observations of Players' Display Names, Colleges, and Statistical Data:
    </p>
    <ul style="font-family: 'Georgia', serif; font-size: 14px; color: black; margin-left: 20px;">
        <li>According to this dataset, <strong>Josh Jones</strong> has the <strong>highest frequency distribution count</strong> of about <strong>310</strong>, with a mean value of <strong>265.0</strong>.</li>
        <li><strong>Alabama College</strong> has the <strong>highest frequency distribution count</strong> of approximately <strong>360</strong>, with a mean value of <strong>249.118644</strong>.</li>
        <li>Players with display names such as <strong>Josh Jones, Michael Carter, Michael Thomas, Kyle Fulle, Lamar Jackson, Connor McGovern, David Long, Spencer Brown, Jonah Williams, and A.J. Green</strong> have a count of about <strong>2</strong>.</li>
        <li><strong>Ashwan Robinson</strong> has the <strong>highest mean value count</strong> of about <strong>330</strong>, according to this dataset.</li>
        <li><strong>Michael Carter</strong> has the <strong>lowest count</strong>, determined by visualizing the bar plot of display names, with a mean value of about <strong>195</strong>.</li>
        <li><strong>Notre Dame College</strong> has the <strong>highest mean value</strong> of approximately <strong>259.49</strong>, while <strong>Washington College</strong> has the <strong>lowest mean value</strong> of about <strong>232.21</strong>.</li>
        <li><strong>Alabama College</strong> has the <strong>highest frequency distribution count</strong> of about <strong>59</strong>, while <strong>Tennessee</strong> has the <strong>lowest frequency distribution count</strong> of about <strong>21</strong>.</li>
        <li><strong>Ashwan Robinson</strong> has the <strong>maximum value</strong> of about <strong>330</strong>, with a frequency distribution count of <strong>1</strong>.</li>
        <li>Columns such as <strong>NFLID, weight, height, and college name</strong> have a frequency distribution count of about <strong>1,697</strong>, while the <strong>age</strong> column has a frequency distribution count of about <strong>1,204</strong>.</li>
        <li>The <strong>NFLID</strong> column has the <strong>maximum value</strong> of approximately <strong>55,241.000000</strong>, higher than all other columns, with a mean value of <strong>48,237.157336</strong>.</li>
        <li>The <strong>age</strong> column has a mean value of about <strong>29.440199</strong> and a maximum value of approximately <strong>47.000000</strong>.</li>
    </ul>
</div>

# <p style="font-family: 'Amiri'; font-size: 2.5rem; color: Black; text-align: center; margin: 0; text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3); background-color: #cce5ff; padding: 20px; border-radius: 20px; border: 7px solid #2a71b8; width:95%">Visualization of Player Metrics</p>

```python
df_track1=pd.read_csv("/kaggle/input/nfl-big-data-bowl-2025/tracking_week_1.csv")
df_track2=pd.read_csv("/kaggle/input/nfl-big-data-bowl-2025/tracking_week_2.csv")
df_track3=pd.read_csv("/kaggle/input/nfl-big-data-bowl-2025/tracking_week_3.csv")
df_track4=pd.read_csv("/kaggle/input/nfl-big-data-bowl-2025/tracking_week_4.csv")
df_track5=pd.read_csv("/kaggle/input/nfl-big-data-bowl-2025/tracking_week_5.csv")
df_track6=pd.read_csv("/kaggle/input/nfl-big-data-bowl-2025/tracking_week_6.csv")
df_track7=pd.read_csv("/kaggle/input/nfl-big-data-bowl-2025/tracking_week_7.csv")
df_track8=pd.read_csv("/kaggle/input/nfl-big-data-bowl-2025/tracking_week_8.csv")
df_track9=pd.read_csv("/kaggle/input/nfl-big-data-bowl-2025/tracking_week_9.csv")
```

```python
# Concatenate data for the first portion (weeks 1-5) and second portion (weeks 6-9)
df_portion1 = pd.concat([df_track1, df_track2, df_track3, df_track4, df_track5], ignore_index=True)
df_portion2 = pd.concat([df_track6, df_track7, df_track8, df_track9], ignore_index=True)

# Select important columns for both portions
columns_to_select = ['gameId', 'playId', 'x', 'y', 's', 'a', 'jerseyNumber']
df_portion1_sample = df_portion1[columns_to_select]
df_portion2_sample = df_portion2[columns_to_select]

# Analysis for Weeks 1-5 (df_portion1)
analysis_portion1 = df_portion1_sample.describe(include='all')

# Analysis for Weeks 6-9 (df_portion2)
analysis_portion2 = df_portion2_sample.describe(include='all')

# Display summary statistics for both portions (count, mean, max)
print("Summary Statistics for Tracking Data (Weeks 1-5):")
display(analysis_portion1.loc[['count', 'mean', 'max']])

print("==========================================================")
print("\nSummary Statistics for Tracking Data (Weeks 6-9):")
display(analysis_portion2.loc[['count', 'mean', 'max']])

print("==========================================================")

# Display the top 10 value counts for all columns in both portions
print("Top Value Counts for Tracking Data (Weeks 1-5):")
for column in df_portion1_sample.columns:
    print(f"\nTop Value Counts for {column} (Weeks 1-5):")
    display(df_portion1_sample[column].value_counts().head(10))

print("==========================================================")

print("\nTop Value Counts for Tracking Data (Weeks 6-9):")
for column in df_portion2_sample.columns:
    print(f"\nTop Value Counts for {column} (Weeks 6-9):")
    display(df_portion2_sample[column].value_counts().head(10))
```

```python
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Concatenate data for the first portion (weeks 1-5) and second portion (weeks 6-9)
df_portion1 = pd.concat([df_track1, df_track2, df_track3, df_track4, df_track5], ignore_index=True)
df_portion2 = pd.concat([df_track6, df_track7, df_track8, df_track9], ignore_index=True)

# Select important columns for both portions
columns_to_select = ['gameId', 'playId', 'x', 'y', 's', 'a', 'jerseyNumber']
df_portion1_sample = df_portion1[columns_to_select].head(20)
df_portion2_sample = df_portion2[columns_to_select].head(20)

# Plotly Interactive Visualizations
fig = make_subplots(
    rows=3, cols=2,
    subplot_titles=[
        "Tracking Data (Weeks 1-5) - Speed",
        "Tracking Data (Weeks 6-9) - Acceleration",
        "Combined Tracking Data (Position)",
        "Speed vs. Acceleration",
        "Speed Distribution",
        "Acceleration Distribution"
    ],
    vertical_spacing=0.15
)

# Plot 1: Tracking data (Weeks 1-5) with vibrant magenta color
fig.add_trace(
    go.Scatter(
        x=df_portion1_sample['x'],
        y=df_portion1_sample['y'],
        mode='markers',
        marker=dict(
            size=12,
            color='magenta',  # Magenta color for striking appearance
            opacity=0.8
        ),
        name="Weeks 1-5 (Speed)",
        legendgroup="Weeks 1-5",
        hovertemplate='<b>Game ID:</b> %{customdata[0]}<br>' +
                      '<b>Play ID:</b> %{customdata[1]}<br>' +
                      '<b>X Position:</b> %{x}<br>' +
                      '<b>Y Position:</b> %{y}<br>' +
                      '<b>Speed:</b> %{customdata[2]}<br>' +
                      '<b>Acceleration:</b> %{customdata[3]}<br>' +
                      '<b>Jersey Number:</b> %{customdata[4]}<br>' +
                      '<extra></extra>',
        customdata=df_portion1_sample[['gameId', 'playId', 's', 'a', 'jerseyNumber']].values
    ),
    row=1, col=1
)

# Plot 2: Tracking data (Weeks 6-9) with bright cyan color
fig.add_trace(
    go.Scatter(
        x=df_portion2_sample['x'],
        y=df_portion2_sample['y'],
        mode='markers',
        marker=dict(
            size=12,
            color='cyan',  # Cyan color for a vibrant look
            opacity=0.8
        ),
        name="Weeks 6-9 (Speed)",
        legendgroup="Weeks 6-9",
        hovertemplate='<b>Game ID:</b> %{customdata[0]}<br>' +
                      '<b>Play ID:</b> %{customdata[1]}<br>' +
                      '<b>X Position:</b> %{x}<br>' +
                      '<b>Y Position:</b> %{y}<br>' +
                      '<b>Speed:</b> %{customdata[2]}<br>' +
                      '<b>Acceleration:</b> %{customdata[3]}<br>' +
                      '<b>Jersey Number:</b> %{customdata[4]}<br>' +
                      '<extra></extra>',
        customdata=df_portion2_sample[['gameId', 'playId', 's', 'a', 'jerseyNumber']].values
    ),
    row=1, col=2
)

# Plot 3: Combined position data with coral and turquoise colors
fig.add_trace(
    go.Scatter(
        x=df_portion1_sample['x'],
        y=df_portion1_sample['y'],
        mode='markers',
        marker=dict(color='coral', size=10, opacity=0.7),
        name="Weeks 1-5 (Position)",
        legendgroup="Weeks 1-5",
        hovertemplate='<b>Game ID:</b> %{customdata[0]}<br>' +
                      '<b>Play ID:</b> %{customdata[1]}<br>' +
                      '<b>X Position:</b> %{x}<br>' +
                      '<b>Y Position:</b> %{y}<br>' +
                      '<b>Speed:</b> %{customdata[2]}<br>' +
                      '<b>Acceleration:</b> %{customdata[3]}<br>' +
                      '<b>Jersey Number:</b> %{customdata[4]}<br>' +
                      '<extra></extra>',
        customdata=df_portion1_sample[['gameId', 'playId', 's', 'a', 'jerseyNumber']].values
    ),
    row=2, col=1
)
fig.add_trace(
    go.Scatter(
        x=df_portion2_sample['x'],
        y=df_portion2_sample['y'],
        mode='markers',
        marker=dict(color='turquoise', size=10, opacity=0.7),
        name="Weeks 6-9 (Position)",
        legendgroup="Weeks 6-9",
        hovertemplate='<b>Game ID:</b> %{customdata[0]}<br>' +
                      '<b>Play ID:</b> %{customdata[1]}<br>' +
                      '<b>X Position:</b> %{x}<br>' +
                      '<b>Y Position:</b> %{y}<br>' +
                      '<b>Speed:</b> %{customdata[2]}<br>' +
                      '<b>Acceleration:</b> %{customdata[3]}<br>' +
                      '<b>Jersey Number:</b> %{customdata[4]}<br>' +
                      '<extra></extra>',
        customdata=df_portion2_sample[['gameId', 'playId', 's', 'a', 'jerseyNumber']].values
    ),
    row=2, col=1
)

# Plot 4: Speed vs. Acceleration with goldenrod and indigo
fig.add_trace(
    go.Scatter(
        x=df_portion1_sample['s'],
        y=df_portion1_sample['a'],
        mode='markers',
        marker=dict(color='goldenrod', size=12, opacity=0.8),
        name="Weeks 1-5 (Speed vs Acceleration)",
        legendgroup="Weeks 1-5",
        hovertemplate='<b>Game ID:</b> %{customdata[0]}<br>' +
                      '<b>Play ID:</b> %{customdata[1]}<br>' +
                      '<b>Speed:</b> %{x}<br>' +
                      '<b>Acceleration:</b> %{y}<br>' +
                      '<b>Jersey Number:</b> %{customdata[2]}<br>' +
                      '<extra></extra>',
        customdata=df_portion1_sample[['gameId', 'playId', 'jerseyNumber']].values
    ),
    row=2, col=2
)
fig.add_trace(
    go.Scatter(
        x=df_portion2_sample['s'],
        y=df_portion2_sample['a'],
        mode='markers',
        marker=dict(color='indigo', size=12, opacity=0.8),
        name="Weeks 6-9 (Speed vs Acceleration)",
        legendgroup="Weeks 6-9",
        hovertemplate='<b>Game ID:</b> %{customdata[0]}<br>' +
                      '<b>Play ID:</b> %{customdata[1]}<br>' +
                      '<b>Speed:</b> %{x}<br>' +
                      '<b>Acceleration:</b> %{y}<br>' +
                      '<b>Jersey Number:</b> %{customdata[2]}<br>' +
                      '<extra></extra>',
        customdata=df_portion2_sample[['gameId', 'playId', 'jerseyNumber']].values
    ),
    row=2, col=2
)

# Plot 5: Speed Histogram with teal and hot pink
fig.add_trace(
    go.Histogram(
        x=df_portion1_sample['s'],
        marker=dict(color='teal', opacity=0.6),
        name="Weeks 1-5 (Speed)",
        histnorm='density',
        legendgroup="Weeks 1-5",
        hovertemplate='<b>Speed:</b> %{x}<br>' +
                      '<b>Density:</b> %{y}<br>' +
                      '<extra></extra>'
    ),
    row=3, col=1
)
fig.add_trace(
    go.Histogram(
        x=df_portion2_sample['s'],
        marker=dict(color='hotpink', opacity=0.6),
        name="Weeks 6-9 (Speed)",
        histnorm='density',
        legendgroup="Weeks 6-9",
        hovertemplate='<b>Speed:</b> %{x}<br>' +
                      '<b>Density:</b> %{y}<br>' +
                      '<extra></extra>'
    ),
    row=3, col=1
)

# Plot 6: Acceleration Histogram with medium violet red and slate blue
fig.add_trace(
    go.Histogram(
        x=df_portion1_sample['a'],
        marker=dict(color='mediumvioletred', opacity=0.6),
        name="Weeks 1-5 (Acceleration)",
        histnorm='density',
        legendgroup="Weeks 1-5",
        hovertemplate='<b>Acceleration:</b> %{x}<br>' +
                      '<b>Density:</b> %{y}<br>' +
                      '<extra></extra>'
    ),
    row=3, col=2
)
fig.add_trace(
    go.Histogram(
        x=df_portion2_sample['a'],
        marker=dict(color='slateblue', opacity=0.6),
        name="Weeks 6-9 (Acceleration)",
        histnorm='density',
        legendgroup="Weeks 6-9",
        hovertemplate='<b>Acceleration:</b> %{x}<br>' +
                      '<b>Density:</b> %{y}<br>' +
                      '<extra></extra>'
    ),
    row=3, col=2
)

# Update axis labels for all subplots
fig.update_xaxes(title_text="X Position", row=1, col=1)
fig.update_yaxes(title_text="Y Position", row=1, col=1)
fig.update_xaxes(title_text="X Position", row=1, col=2)
fig.update_yaxes(title_text="Y Position", row=1, col=2)
fig.update_xaxes(title_text="X Position", row=2, col=1)
fig.update_yaxes(title_text="Y Position", row=2, col=1)
fig.update_xaxes(title_text="Speed", row=2, col=2)
fig.update_yaxes(title_text="Acceleration", row=2, col=2)
fig.update_xaxes(title_text="Speed", row=3, col=1)
fig.update_yaxes(title_text="Density", row=3, col=1)
fig.update_xaxes(title_text="Acceleration", row=3, col=2)
fig.update_yaxes(title_text="Density", row=3, col=2)

# Update layout for better spacing and readability
fig.update_layout(
    height=1200, 
    width=1200,
    title_text="Tracking Data Visualizations",
    showlegend=True,
    template='plotly_white'
)

fig.show()
```

<div style="background-color: #D6EAF8; padding: 20px; border-radius: 15px; border: 2px solid #00008B; margin-top: 20px;">
    <p style="text-align: justify; font-family: 'Times New Roman', serif; font-size: 20px; color: #00008B; font-weight: bold;">
        Observations of Tracking Weeks' Data and Key Metrics:
    </p>
    <ul style="font-family: 'Georgia', serif; font-size: 14px; color: black; margin-left: 20px;">
        <li>For analyzing and visualizing the 9 tracking weeks, the <strong>first five weeks</strong> were concatenated together, and the <strong>last four weeks</strong> were analyzed separately.</li>
        <li>By visualizing the tracking data of <strong>Speed X Position vs. Y Position</strong>, it was determined that:
            <ul>
                <li>The <strong>maximum X position</strong> is <strong>59.2</strong> and the <strong>maximum Y position</strong> is <strong>29.04</strong>. The jersey number at this position is <strong>22</strong>, with an acceleration of <strong>0.27</strong>, speed of <strong>0.41</strong>, Game ID <strong>2022091200</strong>, and Play ID <strong>64</strong> from weeks 1-5.</li>
                <li>The <strong>highest speed</strong> in weeks 1-5 is <strong>0.72</strong>, occurring at X position <strong>51.06</strong> and Y position <strong>28.55</strong>, with the same jersey number, Game ID, and Play ID.</li>
                <li>For weeks 6-9, the <strong>maximum X position</strong> is <strong>50.62</strong> and the <strong>maximum Y position</strong> is <strong>22.99</strong>, with jersey number <strong>22</strong>, acceleration of <strong>0.18</strong>, speed of <strong>0.38</strong>, Game ID <strong>2022101700</strong>, and Play ID <strong>90</strong>.</li>
                <li>The <strong>highest speed</strong> in weeks 6-9 is <strong>1.19</strong>, at X position <strong>49.82</strong> and Y position <strong>22.99</strong>, with similar parameters.</li>
            </ul>
        </li>
        <li>The tracking data of <strong>weeks 6-9</strong> exhibited a higher <strong>X vs. Y Position Distribution</strong> compared to weeks 1-5:
            <ul>
                <li>The <strong>highest values</strong> are X position <strong>51.92</strong> and Y position <strong>29.04</strong>.</li>
                <li>The <strong>lowest values</strong> for weeks 6-9 are X position <strong>49.98</strong> and Y position <strong>22.69</strong>, lower than weeks 1-5.</li>
            </ul>
        </li>
        <li>For <strong>speed vs. acceleration</strong>:
            <ul>
                <li>Weeks 6-9 have the <strong>highest speed</strong> of <strong>0.79</strong> and acceleration of <strong>1.4</strong>, with jersey number <strong>22</strong>.</li>
                <li>Weeks 1-5 have the <strong>highest speed</strong> of <strong>0.38</strong> and acceleration of <strong>0.63</strong>, with jersey number <strong>22</strong>.</li>
            </ul>
        </li>
        <li>The <strong>speed density</strong> analysis shows:
            <ul>
                <li>Weeks 6-9 have the <strong>highest density</strong> of <strong>55</strong> in the speed range <strong>0.2-0.39</strong>.</li>
                <li>Weeks 1-5 have a density of <strong>50</strong> in the speed range <strong>0.6-0.79</strong>.</li>
            </ul>
        </li>
        <li>The <strong>acceleration density</strong> analysis indicates:
            <ul>
                <li>Weeks 6-9 have the <strong>highest density</strong> of <strong>70</strong> in the speed range <strong>0.3-0.49</strong>.</li>
                <li>Weeks 1-5 have a density of <strong>45</strong> in the speed range <strong>0.1-0.29</strong>.</li>
            </ul>
        </li>
        <li>For <strong>frequency distribution counts</strong>:
            <ul>
                <li>Game ID <strong>2022091105</strong> has the <strong>highest count</strong> of <strong>563,247</strong> in weeks 1-5, while weeks 6-9 have the highest count of Game ID <strong>2022103001</strong> at <strong>550,022</strong>.</li>
                <li>Play ID <strong>56</strong> has the <strong>highest count</strong> of <strong>91,747</strong> in weeks 1-5, while weeks 6-9 show a count of <strong>85,008</strong>.</li>
            </ul>
        </li>
        <li>The <strong>X position</strong> value of <strong>86.46</strong> has the highest frequency distribution count of <strong>6,249</strong> in weeks 1-5, while weeks 6-9 have an X position value of <strong>86.39</strong> with a count of <strong>4,239</strong>.</li>
        <li>The <strong>jersey number</strong> <strong>55.0</strong> has the highest frequency distribution count of <strong>666,638</strong> in weeks 1-5, while jersey number <strong>1</strong> has the highest count of <strong>471,668</strong> in weeks 6-9.</li>
    </ul>
</div>

<div style="background-color: #ADD8E6; padding: 20px; border-radius: 15px; border: 2px solid #00008B; margin-top: 20px;">
    <p style="text-align: justify; font-family: 'Times New Roman', serif; font-size: 20px; color: #00008B; font-weight: bold;">
        Key Observations Taken from NFL Players Data and Tracking Weeks
    </p>
    <table style="width: 100%; border-collapse: collapse; text-align: left;">
        <thead>
            <tr style="background-color: #6495ED; color: white;">
                <th>Observation</th>
                <th>Details</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td style="background-color: #E6F1FF; padding: 10px; border: 1px solid #ADD8E6;">Player Age Distribution</td>
                <td style="background-color: #E6F1FF; padding: 10px; border: 1px solid #ADD8E6;">
                    Most players are aged between 25-29 years, with a mean age of 29.44. The maximum age is 47 and the minimum is 25.
                </td>
            </tr>
            <tr>
                <td style="background-color: #E6F1FF; padding: 10px; border: 1px solid #ADD8E6;">Birth Date and Frequency</td>
                <td style="background-color: #E6F1FF; padding: 10px; border: 1px solid #ADD8E6;">
                    Players' birth years range mainly from 1985 to 1997. The most frequent birth month is 3rd, and the 16th-30th of each month shows the highest frequency of birth dates.
                </td>
            </tr>
            <tr>
                <td style="background-color: #E6F1FF; padding: 10px; border: 1px solid #ADD8E6;">Player Height and NFLID</td>
                <td style="background-color: #E6F1FF; padding: 10px; border: 1px solid #ADD8E6;">
                    Players with heights ranging from 75-79 inches are most common, with a height of 76 inches correlating with age 47. NFLID ranges from 52440 to 52541 for players aged 25-29.
                </td>
            </tr>
            <tr>
                <td style="background-color: #E6F1FF; padding: 10px; border: 1px solid #ADD8E6;">Weight Analysis</td>
                <td style="background-color: #E6F1FF; padding: 10px; border: 1px solid #ADD8E6;">
                    The maximum weight recorded is 328 lbs for a 42-year-old player. The average weight is 245.77 lbs, with the lowest weight being 175 lbs.
                </td>
            </tr>
            <tr>
                <td style="background-color: #E6F1FF; padding: 10px; border: 1px solid #ADD8E6;">College Names & Frequency</td>
                <td style="background-color: #E6F1FF; padding: 10px; border: 1px solid #ADD8E6;">
                    Boston College has the highest frequency, with a count of 14. College name length varies from 4 to 24, with Alabama College showing the highest frequency distribution.
                </td>
            </tr>
            <tr>
                <td style="background-color: #E6F1FF; padding: 10px; border: 1px solid #ADD8E6;">Frequency Distribution of Players by Age</td>
                <td style="background-color: #E6F1FF; padding: 10px; border: 1px solid #ADD8E6;">
                    The age range 25-30 years has the highest distribution (77.96%), while ages 40-47 have the lowest (0.1%).
                </td>
            </tr>
            <tr>
                <td style="background-color: #E6F1FF; padding: 10px; border: 1px solid #ADD8E6;">Tracking Weeks Analysis</td>
                <td style="background-color: #E6F1FF; padding: 10px; border: 1px solid #ADD8E6;">
                    Tracking data for weeks 1-5 shows maximum X position of 59.2 and maximum Y position of 29.04. Weeks 6-9 exhibit higher X position distribution and speed/acceleration density.
                </td>
            </tr>
            <tr>
                <td style="background-color: #E6F1FF; padding: 10px; border: 1px solid #ADD8E6;">Highest Frequency Distribution by Game and Play IDs</td>
                <td style="background-color: #E6F1FF; padding: 10px; border: 1px solid #ADD8E6;">
                    The game ID 2022091105 has the highest frequency count of 563,247 in weeks 1-5. Play ID 56 shows the highest count of 91,747 during the same period.
                </td>
            </tr>
            <tr>
                <td style="background-color: #E6F1FF; padding: 10px; border: 1px solid #ADD8E6;">Position Analysis</td>
                <td style="background-color: #E6F1FF; padding: 10px; border: 1px solid #ADD8E6;">
                    The maximum X and Y positions for weeks 1-5 are 86.46 and 29.75, respectively, with a jersey number of 55 having the highest distribution count.
                </td>
            </tr>
        </tbody>
    </table>
</div>

<div style="background-color: #ADD8E6; padding: 25px; border-radius: 15px; border: 3px solid #00008B; margin-top: 20px; box-shadow: 0 4px 10px rgba(0, 0, 139, 0.2);">
    <p style="text-align: center; font-family: 'Arial', sans-serif; font-size: 24px; color: #00008B; font-weight: bold; text-transform: uppercase; letter-spacing: 2px;">
        Key Insightsg NFL Player & Tracking Data from Big Data Bowl 2025
    </p>
    <table style="width: 100%; border-collapse: collapse; margin-top: 20px;">
        <thead>
            <tr style="background-color: #ADD8E6; text-align: center;">
                <th style="padding: 15px; border: 1px solid #00008B; font-size: 18px; font-weight: bold; color: #1E90FF;">Column Name</th>
                <th style="padding: 15px; border: 1px solid #00008B; font-size: 18px; font-weight: bold; color: #1E90FF;">Findings</th>
            </tr>
        </thead>
        <tbody>
            <tr style="background-color: #E6F0FF;">
                <td style="padding: 15px; border: 1px solid #00008B;">Player Age and Birth Year</td>
                <td style="padding: 15px; border: 1px solid #00008B;">By analyzing player data, the minimum age of players is 25, and the maximum age is 47. The most frequent birth years fall between 1985 and 1997, with the highest frequency occurring at 29 years of age. Players born in January, May, and August are most common.</td>
            </tr>
            <tr style="background-color: #E6F0FF;">
                <td style="padding: 15px; border: 1px solid #00008B;">Player Height and Weight</td>
                <td style="padding: 15px; border: 1px solid #00008B;">The player with age 25, born on January 14, 1999, has the highest frequency count. The maximum height recorded is 80 inches, with players aged 29-44 being the most prevalent. The average weight is 245.77 lbs.</td>
            </tr>
            <tr style="background-color: #E6F0FF;">
                <td style="padding: 15px; border: 1px solid #00008B;">Weight Analysis</td>
                <td style="padding: 15px; border: 1px solid #00008B;">Weight analysis reveals the maximum weight of 328 lbs is associated with a player aged 42, and the minimum weight of 175 lbs corresponds to an age of 38. Players aged 25-29 are the most frequent, and Alabama College has the highest frequency distribution.</td>
            </tr>
            <tr style="background-color: #E6F0FF;">
                <td style="padding: 15px; border: 1px solid #00008B;">College Name Frequency</td>
                <td style="padding: 15px; border: 1px solid #00008B;">The highest frequency count of college names is for Boston College, with a notable frequency distribution for players aged 25-40. The longest college name length is 24 characters.</td>
            </tr>
            <tr style="background-color: #E6F0FF;">
                <td style="padding: 15px; border: 1px solid #00008B;">Age Distribution by College</td>
                <td style="padding: 15px; border: 1px solid #00008B;">Players aged 25-30 make up the highest frequency distribution of 77.8%, while players aged 40-47 have the lowest frequency. Alabama College has the largest frequency of players aged 25-30.</td>
            </tr>
            <tr style="background-color: #E6F0FF;">
                <td style="padding: 15px; border: 1px solid #00008B;">Player Display Name and Weight</td>
                <td style="padding: 15px; border: 1px solid #00008B;">Josh Jones holds the highest frequency distribution for display names, with a mean value of 265 lbs. Alabama College has the highest frequency distribution of players aged 25-30.</td>
            </tr>
            <tr style="background-color: #E6F0FF;">
                <td style="padding: 15px; border: 1px solid #00008B;">Tracking Data - Weeks 1-5</td>
                <td style="padding: 15px; border: 1px solid #00008B;">In tracking data, the maximum X position (59.2) and maximum Y position (29.04) are recorded in weeks 1-5. Acceleration and speed values are also tracked for analysis. Tracking weeks 6-9 show higher maximum speeds and accelerations.</td>
            </tr>
            <tr style="background-color: #E6F0FF;">
                <td style="padding: 15px; border: 1px solid #00008B;">Tracking Data - Weeks 6-9</td>
                <td style="padding: 15px; border: 1px solid #00008B;">Week 6-9 also exhibit higher speed and acceleration densities, with speeds ranging from 0.6 to 0.79. The highest frequency distribution counts are seen for game and play IDs from both weeks.</td>
            </tr>
            <tr style="background-color: #E6F0FF;">
                <td style="padding: 15px; border: 1px solid #00008B;">Position Data and Jersey Number</td>
                <td style="padding: 15px; border: 1px solid #00008B;">The position data shows the highest frequency for X (86.46) and Y (29.75) positions across both tracking periods. The jersey number 55 has the highest frequency distribution in weeks 1-5.</td>
            </tr>
        </tbody>
    </table>
</div>

# <p style="font-family: 'Amiri'; font-size: 2.5rem; color: Black; text-align: center; margin: 0; text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3); background-color: #cce5ff; padding: 20px; border-radius: 20px; border: 7px solid #2a71b8; width:95%">Next Step & Goals</p>

<div style="background-color:#D6EAF8;padding:20px;border-radius:15px;border:2px solid #00008B;margin-top:20px;">
    <h2 style="text-align:center;font-family:'Times New Roman',serif;font-size:24px;color:#00008B;font-weight:bold;margin-bottom:15px;">
        Thanks for Exploring this Notebook
    </h2>
    <p style="text-align:justify;font-family:'Times New Roman',serif;font-size:18px;color:#00008B;">
    </p>
    <h2 style="text-align:center;font-family:'Times New Roman',serif;font-size:24px;color:#00008B;font-weight:bold;margin-top:20px;">
        Next Step and Goals
    </h2>
    <ul style="font-family:'Georgia',serif;font-size:14px;color:black;margin-left:20px;">
        <li>1. Comprehensive Exploration & in-depth insights into player play data.</li>
        <li>2. Comprehensive Exploration & in-depth insights into tracking week's data.</li>
    </ul>
</div>