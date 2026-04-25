# NFL 2026 Track Animator

- **Author:** stpete_ishii
- **Votes:** 25
- **Ref:** stpeteishii/nfl-2026-track-animator
- **URL:** https://www.kaggle.com/code/stpeteishii/nfl-2026-track-animator
- **Last run:** 2025-09-28 04:20:18.070000

---

![](https://footballbet247.com/wp-content/uploads/2021/06/9610af1ce22cbc47_nfl_about_web_1400x440.jpg)

# **NFL 2026 Track Animator**

This notebook creates an **animated visualization of NFL player tracking data** from the 2026 Big Data Bowl competition. Here's what it does:

## Purpose
- Visualizes player movements during a specific NFL play from the 2023 season
- Creates an animated GIF showing how players move across the field over time

## Key Components

### 1. **Data Loading & Selection**
- Loads tracking data from Week 1 of the 2023 season
- Selects a specific game and play to analyze (Game ID: 2023090700, Play ID: 436)

### 2. **Football Field Visualization**
- Draws a realistic football field with:
  - Field boundaries and yard lines
  - Colored end zones (blue/red)
  - Proper dimensions (120 yards long, 53.3 yards wide)

### 3. **Player Tracking & Animation**
- Plots player positions for each frame of the play
- Uses color-coding based on player roles:
  - **Defensive Coverage**: Blue
  - **Other Route Runner**: Orange  
  - **Passer**: Purple
  - **Targeted Receiver**: Lime green (highlighted with star markers)

### 4. **Special Features**
- **Targeted Receiver Highlight**: Uses star markers and text labels to emphasize the primary receiver
- **Ball Landing Spot**: Shows where the ball is expected to land
- **Quarterback Identification**: Special diamond marker for the QB
- **Play Information**: Displays game ID, play ID, frame number, and play direction

## Output
The code generates a **20-frame animated GIF** showing the progression of the play, with Amon-Ra St. Brown as the targeted receiver moving downfield while defenders react.

This is essentially a **sports analytics visualization tool** that helps understand player positioning and movement patterns during NFL passing plays.

```python
import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
```

```python
path='/kaggle/input/nfl-big-data-bowl-2026-prediction/train/input_2023_w01.csv'
data0=pd.read_csv(path)
print(data0.columns.tolist())
```

```python
# select game_id and play_id
games=data0['game_id'].unique().tolist()
print(len(games))
data00=data0[data0['game_id']==games[0]]
plays=data00['play_id'].unique().tolist()
print(len(plays))
data=data00[data00['play_id']==plays[4]]
display(data)
```

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import os
import numpy as np

# Function to draw the football field
def draw_football_field(ax):
    # Draw field boundaries
    ax.plot([0, 120], [0, 0], color='white', linewidth=2)
    ax.plot([0, 120], [53.3, 53.3], color='white', linewidth=2)
    ax.plot([10, 10], [0, 53.3], color='white', linewidth=2)
    ax.plot([110, 110], [0, 53.3], color='white', linewidth=2)
    ax.plot([60, 60], [0, 53.3], color='white', linewidth=2)

    # Shade end zones
    ax.axvspan(0, 10, facecolor='blue', alpha=0.2)
    ax.axvspan(110, 120, facecolor='red', alpha=0.2)

    # Draw yard lines
    for x in range(20, 110, 10):
        ax.plot([x, x], [0, 53.3], color='white', linestyle='--', linewidth=1)

# Create a directory to save the frame images
os.makedirs('frames', exist_ok=True)

# Get a list of unique frame IDs
frame_ids = sorted(data['frame_id'].unique())

print(f"Total frames to process: {len(frame_ids)}")

# Check for unique player roles in the data
unique_roles = data['player_role'].dropna().unique()
print(f"Unique player roles: {unique_roles}")

# Define a color map for common player roles
role_colors = {
    'Offense': 'red',
    'Defense': 'blue',
    'Football': 'gold',
    'Ball': 'gold',
    'QB': 'darkred',
    'WR': 'orange',
    'RB': 'yellow',
    'TE': 'goldenrod',
    'OL': 'lightcoral',
    'LB': 'lightblue',
    'DB': 'cyan',
    'DL': 'navy',
    'S': 'deepskyblue',
    'CB': 'aqua',
    'K': 'purple',
    'P': 'violet',
    'Targeted Receiver': 'lime'
}

# Dynamically create a color map based on the roles found in the data
dynamic_role_map = {}
for role in unique_roles:
    role_str = str(role)
    if role_str in role_colors:
        dynamic_role_map[role] = role_colors[role_str]
    elif 'target' in role_str.lower() or 'receiver' in role_str.lower():
        dynamic_role_map[role] = 'lime'
    elif 'offense' in role_str.lower() or 'off' in role_str.lower():
        dynamic_role_map[role] = 'red'
    elif 'defense' in role_str.lower() or 'def' in role_str.lower():
        dynamic_role_map[role] = 'blue'
    elif 'ball' in role_str.lower():
        dynamic_role_map[role] = 'gold'
    else:
        # Assign a unique color for other roles
        colors = ['green', 'purple', 'orange', 'yellow', 'pink', 'gray', 'cyan', 'magenta']
        color_index = hash(role_str) % len(colors)
        dynamic_role_map[role] = colors[color_index]

print(f"Color mapping: {dynamic_role_map}")

# Loop through each frame and create a plot
for i, frame_id in enumerate(frame_ids):
    # Get the data for the current frame
    data_frame = data[data['frame_id'] == frame_id].reset_index(drop=True)

    if len(data_frame) == 0:
        continue

    # Get play information from the first row
    game_id = data_frame.loc[0, 'game_id']
    play_id = data_frame.loc[0, 'play_id']
    play_direction = data_frame.loc[0, 'play_direction']

    fig, ax = plt.subplots(figsize=(16, 8)) # Slightly wider plot for the legend
    ax.set_facecolor('green')
    draw_football_field(ax)

    # Plot regular players
    if 'player_role' in data_frame.columns:
        # Identify targeted receivers
        targeted_receivers = data_frame[
            data_frame['player_role'].str.contains('target|receiver', case=False, na=False) |
            data_frame['player_name'].str.contains('target|receiver', case=False, na=False)
        ]

        # Plot regular players (not targeted receivers)
        regular_players = data_frame[~data_frame.index.isin(targeted_receivers.index)]

        current_roles = regular_players['player_role'].dropna().unique()
        for role in current_roles:
            if role in dynamic_role_map:
                role_data = regular_players[regular_players['player_role'] == role]
                ax.scatter(role_data['x'], role_data['y'],
                           c=dynamic_role_map[role], s=120, edgecolor='black',
                           alpha=0.8, label=str(role), linewidth=1.5)

        # Plot the Targeted Receiver with a special style
        if not targeted_receivers.empty:
            for idx, receiver in targeted_receivers.iterrows():
                # Use a prominent marker with a bright green color
                ax.scatter(receiver['x'], receiver['y'],
                           c='lime', s=300, edgecolor='white',
                           alpha=1.0, label='Targeted Receiver' if idx == targeted_receivers.index[0] else "",
                           marker='*', linewidth=3)
                           
                # Display the name with a text label
                ax.text(receiver['x'], receiver['y'] + 3,
                        f"TARGET\n{receiver['player_name']}",
                        ha='center', va='bottom', fontsize=8, fontweight='bold',
                        color='white',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='red', alpha=0.8, edgecolor='white'))
        
        # Set the legend
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.,
                  fontsize=10, framealpha=0.9)

    else:
        # Fallback if the player_role column is not found
        ax.scatter(data_frame['x'], data_frame['y'], s=120,
                   color='white', edgecolor='black', alpha=0.8, label='Players')

    # Plot the ball's landing spot if it exists
    if 'ball_land_x' in data_frame.columns and not pd.isna(data_frame.loc[0, 'ball_land_x']):
        ax.scatter(data_frame.loc[0, 'ball_land_x'], data_frame.loc[0, 'ball_land_y'],
                   color='yellow', s=400, marker='*', edgecolor='black',
                   alpha=1.0, label='Ball Landing Spot', linewidth=2)

    # Highlight the Quarterback (if present)
    if 'player_role' in data_frame.columns:
        qb_players = data_frame[data_frame['player_role'].str.contains('QB|Quarterback', case=False, na=False)]
        if not qb_players.empty:
            for idx, qb in qb_players.iterrows():
                ax.scatter(qb['x'], qb['y'],
                           c='darkred', s=200, edgecolor='gold',
                           alpha=1.0, label='QB' if idx == qb_players.index[0] else "",
                           marker='D', linewidth=2)

    ax.set_xlim(0, 120)
    ax.set_ylim(0, 53.3)

    # Add play information to the title
    title_text = f'Game: {game_id}, Play: {play_id}, Frame: {frame_id}\nPlay Direction: {play_direction.upper()}'
    if not targeted_receivers.empty:
        receiver_names = ', '.join(targeted_receivers['player_name'].dropna().unique())
        title_text += f'\nTargeted Receiver: {receiver_names}'

    ax.set_title(title_text, fontsize=12, pad=20, fontweight='bold')
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    # Add a grid for better readability
    ax.grid(True, alpha=0.3, linestyle='--')

    # Adjust layout to fit everything
    plt.tight_layout()
    plt.subplots_adjust(right=0.85)

    # Save the frame
    plt.savefig(f'frames/frame_{i:06d}.png', dpi=120, bbox_inches='tight',
                facecolor='green', edgecolor='none')
    plt.close(fig)

    # Display progress
    if (i + 1) % 10 == 0 or (i + 1) == len(frame_ids):
        print(f'Processed {i + 1}/{len(frame_ids)} frames')

# Create the GIF animation
print("Creating animation...")
frames = []
for i in range(len(frame_ids)):
    try:
        frame = Image.open(f'frames/frame_{i:06d}.png')
        frames.append(frame)
    except FileNotFoundError:
        print(f'Frame {i:06d} not found')
        continue

if frames:
    frames[0].save('football_tracking.gif', save_all=True,
                   append_images=frames[1:], duration=100, loop=0)
    print("Animation created successfully! File: football_tracking.gif")
    print(f"Total frames in animation: {len(frames)}")
else:
    print("No frames found to create animation.")
```

```python
from IPython.display import Image
Image(open('./football_tracking.gif','rb').read())
```

![](https://t4.ftcdn.net/jpg/07/18/10/55/360_F_718105588_AEyAssiIcubEj0NzvICcBvG7LDBxql07.jpg)

![](https://wallpapers.com/images/featured/nfl-players-9mr88n406v3cpmli.jpg)