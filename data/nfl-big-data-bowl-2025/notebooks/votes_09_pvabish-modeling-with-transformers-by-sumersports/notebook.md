# Modeling with Transformers, by SumerSports

- **Author:** Pavel V
- **Votes:** 68
- **Ref:** pvabish/modeling-with-transformers-by-sumersports
- **URL:** https://www.kaggle.com/code/pvabish/modeling-with-transformers-by-sumersports
- **Last run:** 2024-10-28 22:19:49.070000

---

<img src="https://pbs.twimg.com/media/FpBhWD2XsB0KS_X.jpg:large" style="width: 300px;" alt="SumerSports logo Image">

# Modeling Player Tracking Data with Transformers and Deep Learning
*Unlocking deeper football insights through advanced machine learning*

## Overview
This comprehensive notebook demonstrates the application of transformer models to NFL player tracking data, offering data scientists and sports analysts a modern approach to understanding the complex dynamics of football through sophisticated machine learning techniques. By leveraging state-of-the-art deep learning architectures, we aim to uncover hidden patterns and strategic insights from player movement data.

*Note: This notebook is intended as an educational resource to demonstrate transformer applications in sports analytics, not as a competitive entry for the Kaggle competition. Our goal is to provide a foundation for understanding and implementing these techniques.*

## Why Transformers?
Transformers represent a breakthrough in analyzing spatio-temporal sports data. These powerful models are particularly well-suited for player tracking analysis as they naturally handle variable player ordering, reduce the need for complex feature engineering, and excel at capturing intricate interactions between players. Their self-attention mechanism provides a natural framework for understanding how players influence and react to each other on the field.

## Contents
1. Data Preprocessing:
 Processing NFL tracking data into a suitable format and selecting relevant features. We'll cover data cleaning, normalization, and the creation of meaningful input sequences that capture the essence of player movements and game dynamics. Due to computational limitations, we process only a single week of data (week 1) for this notebook.

2. Model Implementation:
 Setting up the transformer architecture and training pipeline using modern deep learning frameworks. In this example we will train a model to predict the `offenseFormation` of the play, given a frame of data.

3. Analysis & Evaluation:
 Assessing model performance through evaluation metrics and confusion matrices. Despite training on less than a single week of data, the model is able to predict `offenseFormation` on a withheld test set at **~92% accuracy**

## Applications
While our focus centers on NFL data, the methodologies and techniques presented here extend far beyond football. These approaches can be adapted for analyzing player tracking data across various sports, opening new avenues for strategic analysis in basketball, soccer, hockey, and more.

## Contributing
This notebook represents a collaborative effort to advance the field of sports analytics. We actively encourage community participation through feedback, code contributions, and discussions about innovative approaches. By sharing knowledge and building upon each other's work, we can collectively push the boundaries of what's possible in sports analytics.

# Step 0: Add "sportstransformer-utils" utils via Kaggle Dataset
`File --> Add Input --> DataSets --> Search "SportsTransformer_utils" --> Add`

```python
import sys
sys.path.append('/kaggle/input/sportstransformer-utils')
import prep_data
import process_datasets
import models
```

# Explore the modeling target -> offenseFormation

```python
import pandas as pd

# Read the CSV
plays_df = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2025/plays.csv')

# Get value counts of offenseFormation
formation_counts = (plays_df['offenseFormation']
    .value_counts()
    .sort_index()  # Sort alphabetically by formation name
    .to_frame()    # Convert to DataFrame for nicer printing
)
print(formation_counts)

# Get output dim. Number of unique formations
output_dim = plays_df['offenseFormation'].nunique()
```

# 1. Data Preprocessing

```python
prep_data.main()
process_datasets.main()
```

```python
import torch
from torch.utils.data import DataLoader
from process_datasets import load_datasets

# Load preprocessed datasets
train_dataset = load_datasets(model_type='transformer', split='train')
val_dataset = load_datasets(model_type='transformer', split='val')
test_dataset = load_datasets(model_type='transformer', split='test')

# Create DataLoader objects
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=3)
val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=3)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=3)

# Print feature and target shapes from DataLoaders
for batch in train_loader:
    features, targets = batch
    print("Train features shape:", features.shape)
    print("Train targets shape:", targets.shape)
    break
```

# 2. Model Implemenation

```python
from models import SportsTransformerLitModel

# Model parameters
feature_len = 5  # Adjust this as needed based on input data
model_dim = 64  # Dimension of transformer model (adjustable)
num_layers = 4  # Number of transformer layers (adjustable)
dropout = 0.01
learning_rate = 1e-3
batch_size = 64
output_dim = plays_df['offenseFormation'].nunique()

# Initialize the model
model = SportsTransformerLitModel(
    feature_len=feature_len,
    batch_size=batch_size,
    model_dim=model_dim,
    num_layers=num_layers,
    output_dim=output_dim,
    dropout=dropout,
    learning_rate=learning_rate,
)
```

```python
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pathlib import Path

# Define checkpointing to save the best model
checkpoint_callback = ModelCheckpoint(
   dirpath=Path("checkpoints/"),
   filename="best-checkpoint",
   save_top_k=1,
   verbose=True,
   monitor="val_loss",
   mode="min",
)

# Define early stopping
early_stop_callback = EarlyStopping(
   monitor="val_loss",
   min_delta=0.01,  # Minimum change in monitored value to qualify as an improvement
   patience=3,      # Number of epochs with no improvement after which training will be stopped
   verbose=True,
   mode="min"
)

# Initialize the trainer
trainer = Trainer(
   max_epochs=20,  # Adjust the number of epochs
   accelerator="gpu",  # Use 'gpu' if CUDA is available, otherwise use 'cpu'
   devices=1,
   callbacks=[checkpoint_callback, early_stop_callback],
)
```

```python
# Start training
trainer.fit(model, train_loader, val_loader)
```

# 3. Analysis and Evaluation

```python
# Inference on test data
predictions = trainer.predict(model, test_loader)
```

```python
# Concatenate predictions into a single tensor
predictions_tensor = torch.cat(predictions, dim=0)

# Assuming predictions are logits for a multi-class problem
predicted_labels = torch.argmax(predictions_tensor, dim=1)
```

```python
import numpy as np

# Extract true labels from the test_loader
y_true = torch.cat([y for _, y in test_loader], dim=0)

# Convert tensors to numpy arrays if needed for sklearn functions
y_true_np = np.argmax(y_true.cpu().numpy(), axis=-1)
predicted_labels_np = predicted_labels.cpu().numpy()

print("y_true shape:", y_true_np.shape)
print("Predicted labels shape:", predicted_labels_np.shape)
```

```python
# Create a test dataframe
df_test = pd.DataFrame({
    'gameId': [key[0] for key in test_dataset.keys],
    'playId': [key[1] for key in test_dataset.keys],
    'mirrored': [key[2] for key in test_dataset.keys],
    'frameId': [key[3] for key in test_dataset.keys],
    'true_labels': y_true_np,
    'predicted_labels': predicted_labels_np
})
```

```python
# Attach metadata and filter to ball_snap event only
df_test_metadata = pd.read_parquet('/kaggle/working/split_prepped_data/test_features.parquet')

df_test = df_test.merge(df_test_metadata[["gameId", "playId", "mirrored", "frameId", "event", "frameType"]], on=["gameId", "playId", "mirrored", "frameId"], how="left")

# Remove frame after the snap
df_test_before_snap = df_test[df_test.frameType == "BEFORE_SNAP"]

# Filter to ball_snap event for evaluation
df_test_ball_snap = df_test[df_test.event == "ball_snap"]

df_test_ball_snap = df_test_ball_snap.drop_duplicates(subset=['gameId', 'playId', 'mirrored', 'frameId'])

df_test_ball_snap = df_test_ball_snap.sort_values(['gameId', 'playId', 'mirrored', 'frameId']).reset_index(drop=True)

display(df_test_ball_snap)

#prediction_counts = df_test.groupby(['gameId', 'playId', 'mirrored'])['predicted_labels'].value_counts().unstack(fill_value=0)
#display(prediction_counts)
```

```python
true_labels = df_test_ball_snap['true_labels'].values
predicted_labels = df_test_ball_snap['predicted_labels'].values
```

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Times New Roman'

# Get class labels from FORMATION_ENUM and sort alphabetically
formation_labels = sorted(list(process_datasets.FORMATION_ENUM.keys()))

# Print unique values to debug
print("Unique values in true labels:", np.unique(true_labels))
print("Unique values in predicted labels:", np.unique(predicted_labels))
print("Formation enum values:", process_datasets.FORMATION_ENUM)

# Calculate metrics using labels parameter to specify valid classes
accuracy = accuracy_score(true_labels, predicted_labels)
precision = precision_score(true_labels, predicted_labels, average='weighted', 
                         labels=np.unique(true_labels))
recall = recall_score(true_labels, predicted_labels, average='weighted',
                    labels=np.unique(true_labels))
f1 = f1_score(true_labels, predicted_labels, average='weighted',
             labels=np.unique(true_labels))

# Create confusion matrix only for classes that appear in the data
present_classes = sorted(list(set(np.unique(true_labels)) | set(np.unique(predicted_labels))))
conf_matrix = confusion_matrix(true_labels, predicted_labels, 
                            labels=present_classes)

# Get labels for present classes
present_labels = [formation_labels[i] for i in present_classes]

# Print metrics
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision (weighted): {precision:.4f}")
print(f"Recall (weighted): {recall:.4f}")
print(f"F1 Score (weighted): {f1:.4f}")

# Plot confusion matrix with labels for present classes
plt.figure(figsize=(7, 6))
sns.heatmap(conf_matrix, 
          annot=True, 
          fmt='d', 
          cmap='Blues',
          xticklabels=present_labels,
          yticklabels=present_labels)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.xticks(rotation=45)
plt.yticks(rotation=45)
plt.tight_layout()
plt.show()
```

```python
# Show example play
df_play = df_test_before_snap[((df_test_before_snap.playId == 191) & (df_test_before_snap.mirrored == False))]

df_tracking_test = pd.read_parquet('/kaggle/working/split_prepped_data/test_features.parquet')
df_example_play = df_tracking_test[((df_tracking_test.playId == 191) & (df_tracking_test.mirrored == False))]
df_example_play = df_example_play[df_example_play.frameType == "BEFORE_SNAP"]
df_example_play = df_example_play.merge(df_play[["gameId", "playId", "mirrored", "frameId", "predicted_labels"]], on=["gameId", "playId", "mirrored", "frameId"], how="left").reset_index()
display(df_example_play)
```

```python
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

FORMATION_ENUM_REV = {v: k for k, v in process_datasets.FORMATION_ENUM.items()}

# Create the plot
fig = px.scatter(df_example_play, 
   x='x', 
   y='y',
   color='club',
   animation_frame='frameId',
   range_x=[50, 120],
   range_y=[0, 53.3],
   title='Player Positions',
   labels={'x': 'Field Length (Yards)', 'y': 'Field Width (Yards)'},
   hover_data=['nflId'],
   width=900,
   height=700,
   animation_group="nflId"
)

# Add prediction text for each frame
for frame in fig.frames:
   frameId = int(frame.name)
   prediction_idx = df_example_play[df_example_play.frameId == frameId]['predicted_labels'].values[0]
   prediction_name = FORMATION_ENUM_REV[prediction_idx]
   
   frame.layout.update(
       annotations=[{
           'text': f'Formation Prediction: {prediction_name}',
           'x': 0.3,
           'y': 0.95,
           'xref': 'paper',
           'yref': 'paper',
           'showarrow': False,
           'font': {'size': 24, 'color': 'blue'},
           'xanchor': 'left',
           'yanchor': 'top'
       }]
   )

# Also add initial prediction to the base layout
initial_prediction_idx = df_example_play[df_example_play.frameId == df_example_play.frameId.min()]['predicted_labels'].iloc[0]
initial_prediction_name = FORMATION_ENUM_REV[initial_prediction_idx]
fig.update_layout(
   annotations=[{
       'text': f'Formation Prediction: {initial_prediction_name}',
       'x': 0.3,
       'y': 0.95,
       'xref': 'paper',
       'yref': 'paper',
       'showarrow': False,
       'font': {'size': 24, 'color': 'blue'},
       'xanchor': 'left',
       'yanchor': 'top'
   }]
)

# Rest of your layout settings
fig.update_traces(marker=dict(size=12))
fig.update_yaxes(
  scaleanchor="x",
  scaleratio=1,
)

fig.update_layout(
   updatemenus=[{
       'type': 'buttons',
       'showactive': False,
       'buttons': [{
           'label': 'Play',
           'method': 'animate',
           'args': [None, {
               'frame': {'duration': 100, 'redraw': True},
               'fromcurrent': True,
               'transition': {'duration': 100}
           }]
       }, {
           'label': 'Pause',
           'method': 'animate',
           'args': [[None], {
               'frame': {'duration': 0, 'redraw': False},
               'mode': 'immediate',
               'transition': {'duration': 0}
           }]
       }]
   }]
)

fig.update_layout(
   plot_bgcolor='#ccebd4',  # Green background for field
   yaxis=dict(
       showgrid=False,  # Remove horizontal grid
       zeroline=False,
       showticklabels=False,
   ),
   xaxis=dict(
       showgrid=False,  # Remove default grid
       zeroline=False,
       showticklabels=False,
   ),
)

# Add vertical lines every 5 yards
for yard in range(0, 121, 5):
   fig.add_shape(
       type="line",
       x0=yard,
       x1=yard,
       y0=0,
       y1=53.3,
       line=dict(
           color="white",
           width=1,
       ),
       layer='below'
   )
   # Make every 10 yard line more prominent
   if yard % 10 == 0:
       fig.add_shape(
           type="line",
           x0=yard,
           x1=yard,
           y0=0,
           y1=53.3,
           line=dict(
               color="white",
               width=2,
           ),
           layer='below'
       )

fig.show()
```