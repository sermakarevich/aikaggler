# Voxel by Voxel: 3D CNN Intracranial Aneurysms

- **Author:** AC
- **Votes:** 131
- **Ref:** ahsuna123/voxel-by-voxel-3d-cnn-intracranial-aneurysms
- **URL:** https://www.kaggle.com/code/ahsuna123/voxel-by-voxel-3d-cnn-intracranial-aneurysms
- **Last run:** 2025-07-29 13:15:53.190000

---

<div style="border-left: 4px solid #4e4e4e; padding: 1.5rem; background-color: #1e1e1e; font-family: 'Segoe UI', sans-serif; color: #e0e0e0; line-height: 1.6; font-size: 16px; border-radius: 8px;">



  
  <h3 style="color: #ffa07a;">🩸 What is an Intracranial Aneurysm?</h3>
  <div style="display: flex; justify-content: center; margin: 2rem 0;">
  <img src="https://upload.wikimedia.org/wikipedia/commons/8/80/Cerebral_aneurysm_NIH.jpg" 
       alt="Brain Aneurysm Scan" 
       style="width: 50%; border-radius: 10px;" />
</div>


  <p>
    An intracranial aneurysm is a bulging or ballooning of a blood vessel in the brain. If undetected, it can rupture — leading to life-threatening bleeding. Early detection is crucial. The challenge? These aneurysms are often subtle, hidden in complex anatomy, and differ widely between patients.



  </p>
  <p>
    That’s where machine learning can step in — augmenting radiologist workflows by detecting patterns across thousands of image slices in a fraction of the time.
  </p>

  <h3 style="color: #ffa07a;">📷 Imaging Modalities Used</h3>
  <p>We’re working with 3D scans from the following modalities:</p>

  <table style="width: 100%; border-collapse: collapse; margin-top: 1rem; font-size: 15px;">
    <thead>
      <tr style="background-color: #2e2e2e;">
        <th style="text-align: left; padding: 8px;">Modality</th>
        <th style="text-align: left; padding: 8px;">Full Form</th>
        <th style="text-align: left; padding: 8px;">What It Shows</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td style="padding: 8px;">CTA</td>
        <td style="padding: 8px;">Computed Tomography Angiography</td>
        <td style="padding: 8px;">Blood vessels using contrast-enhanced CT</td>
      </tr>
      <tr style="background-color: #2b2b2b;">
        <td style="padding: 8px;">MRA</td>
        <td style="padding: 8px;">Magnetic Resonance Angiography</td>
        <td style="padding: 8px;">Blood vessels using MRI</td>
      </tr>
      <tr>
        <td style="padding: 8px;">MRI</td>
        <td style="padding: 8px;">Magnetic Resonance Imaging</td>
        <td style="padding: 8px;">Structural brain tissue, no contrast</td>
      </tr>
    </tbody>
  </table>

  <img src="https://www.nature.com/articles/s41598-023-33182-1/figures/1" alt="3D Scan Slices" style="width: 100%; border-radius: 10px; margin: 1rem 0;" />

  <p>
    Each scan is a 3D volume made up of 2D slices. Think of it as flipping through pages of a brain atlas.
  </p>

  <h3 style="color: #ffa07a;">📌 What Are We Predicting?</h3>
  <p>Two key tasks define this competition:</p>
  <ul>
    <li><strong>Classification</strong> — Does this scan contain an aneurysm?</li>
    <li><strong>Localization</strong> — If yes, where is it located? (3D coordinates + 1 of 13 brain regions)</li>
  </ul>

  <h3 style="color: #ffa07a;">🗃️ Dataset Overview</h3>
  <ul>
    <li><code>train.csv</code> — Aneurysm presence and location labels</li>
    <li><code>train_localizers.csv</code> — 3D coordinates for aneurysms (subset)</li>
    <li><code>DICOM folders</code> — Raw scan data (one folder per patient)</li>
    <li><code>segmentations</code> — Optional vessel masks in <code>.nii.gz</code> format</li>
  </ul>

  <h3 style="color: #ffa07a;">🌍 Why This Matters</h3>
  <p>
    In real-world hospitals, radiologists spend hours scanning hundreds of slices to find aneurysms. Our model has the potential to become a powerful clinical assistant flagging risk areas, reducing diagnostic time, and minimizing human error.
  </p>
  <p>
    The stakes are high. So is the opportunity.
  </p>
</div>

<h2 style="color:#D2B48C">Let’s Load The Data</h2>

```python
import pandas as pd
train_df = pd.read_csv('/kaggle/input/rsna-intracranial-aneurysm-detection/train.csv')
train_df.head()
```

<h2 style="color:#D2B48C">Aneurysm Counts per Location</h2>
<H4>Where do aneurysms tend to occur?
This bar chart answers that — using the 13 location_* columns.</H4>

```python
import plotly.express as px

location_cols = [col for col in train_df.columns if col not in ['SeriesInstanceUID', 'PatientAge', 'PatientSex', 'Modality', 'Aneurysm Present']]
location_counts = train_df[location_cols].sum().sort_values(ascending=False)

fig = px.bar(
    location_counts,
    orientation='v',
    title='📊 Aneurysm Count by Location',
    labels={'value': 'Count', 'index': 'Location'},
    color=location_counts.values,
    color_continuous_scale=[[0, '#00BFC4'], [1, '#C77CFF']],
    template='plotly_dark'
)
fig.show()
```

<h2 style="color:#D2B48C">🧠 Imaging Modality Distribution</h2> <h4>Which imaging techniques are used? This donut chart shows the breakdown of modalities like CT and MRI in the dataset.</h4>

```python
modality_counts = train_df['Modality'].value_counts()

fig = px.pie(
    names=modality_counts.index,
    values=modality_counts.values,
    title='🧠 Imaging Modality Distribution',
    hole=0.4,
    color_discrete_sequence=['#00BFC4', '#C77CFF'],
    template='plotly_dark'
)
fig.show()
```

<h2 style="color:#D2B48C">📊 Age Distribution by Aneurysm Presence</h2> <h4>Are aneurysms more common at certain ages? This histogram compares age trends for patients with and without aneurysms.</h4>

```python
import seaborn as sns
import matplotlib.pyplot as plt

plt.style.use('dark_background')
plt.figure(figsize=(10,6))
sns.histplot(
    data=train_df,
    x='PatientAge',
    hue='Aneurysm Present',
    bins=30,
    kde=True,
    palette={0: '#00BFC4', 1: '#C77CFF'}
)
plt.title("Age Distribution by Aneurysm Presence", fontsize=14)
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.grid(alpha=0.2)
plt.show()
```

```python
pd.crosstab(train_df['PatientSex'], train_df['Aneurysm Present'], normalize='index') * 100
```

<h2 style="color:#D2B48C">⚖️ Class Imbalance: Any Aneurysm Present</h2> <h4>How balanced is the dataset? This chart shows how many cases have aneurysms vs. those that don’t — crucial for model performance.</h4>

```python
fig = px.histogram(
    train_df,
    x='Aneurysm Present',
    title='⚖️ Class Imbalance: Any Aneurysm Present',
    color='Aneurysm Present',
    text_auto=True,
    color_discrete_map={0: '#00BFC4', 1: '#C77CFF'},
    template='plotly_dark'
)
fig.update_xaxes(type='category', tickvals=[0, 1], ticktext=["No Aneurysm", "Aneurysm"])
fig.show()
```

<h2 style="color:#D2B48C">🧬 Aneurysm Risk by Gender</h2> <h4>What’s the gender split? This table shows the % of aneurysm cases within each gender — offering early insight into possible demographic risk factors.</h4>

```python
# Simple % table (can be styled in notebook output)
gender_ct = pd.crosstab(train_df['PatientSex'], train_df['Aneurysm Present'], normalize='index') * 100
gender_ct = gender_ct.rename(columns={0: 'No Aneurysm (%)', 1: 'Aneurysm (%)'})
gender_ct.style.background_gradient(cmap='crest').format("{:.1f}%")
```

<h2 style="color:#D2B48C">Train Localizer Analysis </h2>

```python
import pandas as pd
import ast

localizers_df = pd.read_csv('/kaggle/input/rsna-intracranial-aneurysm-detection/train_localizers.csv')

# Convert coordinate strings to dicts
localizers_df['coords'] = localizers_df['coordinates'].apply(ast.literal_eval)
localizers_df['x'] = localizers_df['coords'].apply(lambda d: d['x'])
localizers_df['y'] = localizers_df['coords'].apply(lambda d: d['y'])

localizers_df.drop(columns=['coordinates', 'coords'], inplace=True)
localizers_df.head()
```

<h2 style="color:#D2B48C">📍 Aneurysm Localization Heatmap</h2> <h4>Where are aneurysms typically found in image space? This 2D heatmap visualizes the frequency of annotated aneurysm coordinates (`x`, `y`) across all scans.</h4>

```python
fig = px.density_heatmap(
    localizers_df,
    x='x',
    y='y',
    nbinsx=50,
    nbinsy=50,
    title='🧠 Heatmap of Aneurysm Locations in Image Space',
    color_continuous_scale='Turbo',
    template='plotly_dark',
)
fig.update_yaxes(autorange="reversed")
fig.show()
```

<h2 style="color:#D2B48C">🧭 Aneurysm Coordinates by Location</h2> <h4>Which parts of the brain are affected where? This 2D scatterplot maps each aneurysm's annotated `(x, y)` position and colors them by anatomical location.</h4>

```python
fig = px.scatter(
    localizers_df,
    x='x',
    y='y',
    color='location',
    title='🧠 2D Scatter of Aneurysm Coordinates by Location',
    template='plotly_dark',
    color_discrete_sequence=px.colors.qualitative.Dark24
)
fig.update_yaxes(autorange="reversed")
fig.show()
```

<h2 style="color:#D2B48C">🔁 Aneurysm Co-occurrence Across Brain Regions</h2>

<p style="color:#ccc">
Each patient in the dataset can have multiple aneurysms in different brain arteries. But are certain aneurysm sites more likely to occur together?

To investigate this, we treat the location columns as binary labels and compute a co-occurrence matrix — it tells us how often, for instance, an aneurysm in the <code>Left MCA</code> is also accompanied by one in the <code>ACoA</code>, and so on.
</p>

```python
df = train_df
# Identify location columns correctly
location_cols = df.columns[4:-1]  # skip UID, Age, Sex, Modality, and skip final label
location_df = df[location_cols].astype(int)  # just in case they're still object type

# Co-occurrence matrix
co_matrix = location_df.T.dot(location_df)

# Plot heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(co_matrix, cmap="magma", annot=True, fmt=".0f", linewidths=0.5)
plt.title("🧠 Aneurysm Co-occurrence Matrix", fontsize=16, color='white')
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.yticks(rotation=0, fontsize=9)
plt.gca().set_facecolor('black')
plt.gcf().set_facecolor('#111111')
plt.tight_layout()
plt.show()
```

<h2 style="color:#D2B48C">🧠 Aneurysm Spread per Patient</h2> <h4>How many different brain regions are affected per case? This histogram shows the count of locations marked positive (1) for each patient in the dataset.</h4>

```python
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Location Count Distribution (number of 1s per row)
location_counts = location_df.sum(axis=1)

plt.figure(figsize=(8, 5))
sns.histplot(location_counts, bins=range(1, location_counts.max()+2), kde=False, color="teal")
plt.title("Distribution of Aneurysm Locations per Patient")
plt.xlabel("Number of Locations with Aneurysm")
plt.ylabel("Number of Patients")
plt.grid(True)
plt.tight_layout()
plt.show()
```

<h2 style="color:#D2B48C">🧬 Aneurysm Location Frequency by Sex</h2> <h4>Are certain aneurysm locations more common in one sex? This grouped bar chart breaks down aneurysm counts by brain region and patient sex.</h4>

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming 'df' is already loaded and preprocessed (i.e., binary columns are int)

# Filter only rows where aneurysm is present
df_pos = df[df["Aneurysm Present"] == 1]

# List of all aneurysm location columns
location_cols = df_pos.columns[4:-1]  # From 'Left Infraclinoid...' to 'Other Posterior Circulation'

# Group by PatientSex and sum each location
sex_location = df_pos.groupby("PatientSex")[location_cols].sum().T

# Reset index for plotting
sex_location = sex_location.reset_index().melt(id_vars="index", var_name="Sex", value_name="Count")
sex_location = sex_location.rename(columns={"index": "Location"})

# Plot
plt.figure(figsize=(16, 6))
sns.barplot(data=sex_location, x="Location", y="Count", hue="Sex")
plt.xticks(rotation=45, ha="right")
plt.title("🧬 Aneurysm Location Frequency by Sex")
plt.ylabel("Aneurysm Count")
plt.xlabel("Location")
plt.tight_layout()
plt.show()
```

<h2 style="color:#D2B48C">🎂 Patient Age Distribution per Aneurysm Location</h2> <h4>Do certain aneurysm locations occur more frequently in specific age groups? This boxplot shows age ranges of patients for each brain region affected.</h4>

```python
# List of location columns
location_cols = df.columns[4:-1]

# Filter only positive cases
df_pos = df[df["Aneurysm Present"] == 1]

# Prepare a DataFrame: for each location, collect patients with that location = 1
age_location = []

for loc in location_cols:
    subset = df_pos[df_pos[loc] == 1][["PatientAge"]].copy()
    subset["Location"] = loc
    age_location.append(subset)

# Combine all into one DataFrame
age_location_df = pd.concat(age_location)

# Plot
plt.figure(figsize=(16, 6))
sns.boxplot(data=age_location_df, x="Location", y="PatientAge", palette="crest")
plt.xticks(rotation=45, ha="right")
plt.title("🎂 Patient Age Distribution per Aneurysm Location")
plt.ylabel("Age")
plt.xlabel("Location")
plt.tight_layout()
plt.show()
```

<h2 style="color:#D2B48C">🎂 Patient Age Distribution per Aneurysm Location (Violin Plot)</h2> <h4>How does the age of patients vary across aneurysm locations? This violin plot reveals both distribution spread and central tendencies per region.</h4>

```python
plt.figure(figsize=(16, 6))
sns.violinplot(data=age_location_df, x="Location", y="PatientAge", palette="crest")
plt.xticks(rotation=45, ha="right")
plt.title("🎂 Patient Age Distribution per Aneurysm Location (Violin Plot)")
plt.ylabel("Age")
plt.xlabel("Location")
plt.tight_layout()
plt.show()
```

<h2 style="color:#D2B48C">🧠 Modality Preference per Aneurysm Location</h2> <h4>Which imaging modality is preferred for each aneurysm location?</h4> This heatmap shows how often **CTA** (Computed Tomography Angiography) and **MRA** (Magnetic Resonance Angiography) are used for different aneurysm locations. It helps identify modality biases — for example, some regions might be more frequently scanned with CTA, while others lean toward MRA, possibly due to anatomical accessibility or diagnostic clarity.

```python
# Filter rows with aneurysm present
df_pos = df[df["Aneurysm Present"] == 1].copy()

# Initialize a dictionary to store counts
modality_counts = {}

for loc in location_cols:
    subset = df_pos[df_pos[loc] == 1]
    modality_distribution = subset["Modality"].value_counts()
    modality_counts[loc] = modality_distribution

# Convert to DataFrame and fill missing values with 0
modality_df = pd.DataFrame(modality_counts).T.fillna(0).astype(int)
modality_df = modality_df[["CTA", "MRA"]] if "CTA" in modality_df.columns and "MRA" in modality_df.columns else modality_df

# Plot heatmap
plt.figure(figsize=(12, 6))
sns.heatmap(modality_df.T, annot=True, cmap="crest", fmt="d")
plt.title("🧠 Modality Preference per Aneurysm Location")
plt.xlabel("Location")
plt.ylabel("Modality")
plt.tight_layout()
plt.show()
```

<h2 style="color:#D2B48C">🧠 Modality Usage per Aneurysm Location</h2> <h4>How many patients were imaged with each modality at each aneurysm site?</h4> This grouped bar chart compares **CTA** and **MRA** usage across aneurysm locations — offering a clear view of modality distribution patterns across anatomical regions.

```python
modality_df_melted = modality_df.reset_index().melt(id_vars="index", value_name="Count", var_name="Modality")
modality_df_melted.rename(columns={"index": "Location"}, inplace=True)

plt.figure(figsize=(14, 6))
sns.barplot(data=modality_df_melted, x="Location", y="Count", hue="Modality", palette="crest")
plt.title("🧠 Modality Usage per Aneurysm Location")
plt.xticks(rotation=45, ha="right")
plt.xlabel("Location")
plt.ylabel("Patient Count")
plt.tight_layout()
plt.show()
```

<h2 style="color:#D2B48C">🧭 Spatial Heatmaps of Aneurysm Coordinates</h2> <h4>Do certain aneurysm types tend to cluster spatially?</h4> These KDE heatmaps highlight **where** aneurysms appear within scans for the top 5 locations — revealing anatomical clustering patterns based on `x` and `y` coordinates.

```python
import ast

train_localizations = pd.read_csv("/kaggle/input/rsna-intracranial-aneurysm-detection/train_localizers.csv")
# Parse coordinates
train_localizations["coord_dict"] = train_localizations["coordinates"].apply(ast.literal_eval)
train_localizations["x"] = train_localizations["coord_dict"].apply(lambda d: d["x"])
train_localizations["y"] = train_localizations["coord_dict"].apply(lambda d: d["y"])

# Top 4-5 most frequent locations
top_locations = train_localizations["location"].value_counts().head(5).index

# Plot heatmap per location
for loc in top_locations:
    subset = train_localizations[train_localizations["location"] == loc]
    plt.figure(figsize=(6, 5))
    sns.kdeplot(data=subset, x="x", y="y", fill=True, cmap="crest")
    plt.title(f"🧭 Spatial Heatmap for {loc}")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.tight_layout()
    plt.show()
```

<h2 style="color:#C77CFF">🔄 Comparing Aneurysm Frequencies Across Datasets</h2> <h4>How consistent are the labels and localized annotations?</h4> This side-by-side bar chart contrasts location frequencies between `train.csv` and `train_localizers.csv`. Minor mismatches may suggest annotation noise or underreported aneurysms.

```python
train = pd.read_csv("/kaggle/input/rsna-intracranial-aneurysm-detection/train.csv")
# Frequency from train.csv
label_freq = train[location_cols].sum().sort_values(ascending=False).rename("train.csv")

# Frequency from localizations.csv
localizer_freq = train_localizations["location"].value_counts().rename("train_localizations.csv")

# Combine
freq_comparison = pd.concat([label_freq, localizer_freq], axis=1).fillna(0).astype(int)

# Bar plot
freq_comparison.plot(kind="bar", figsize=(14, 6), color=['#00BFC4',  '#C77CFF'])
plt.title("🔄 Frequency Comparison of Aneurysm Locations")
plt.ylabel("Count")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()
```

<h2 style="color:#00BFC4">🧮 CT Slice Count per Series</h2> 
<h4>How deep is each scan?</h4> 
This summary shows how many DICOM slices are present per series. The average is <b>{229.798638}</b> slices, with a range from <b>{1.0000}</b> to <b>{{1441.000000}}</b>. Scans with fewer slices may lack anatomical detail, affecting 3D model accuracy.

```python
import os
import pandas as pd

series_dir = '/kaggle/input/rsna-intracranial-aneurysm-detection/series'
series_folders = [f for f in os.listdir(series_dir) if os.path.isdir(os.path.join(series_dir, f))]

series_counts = []
for folder in series_folders:
    dcm_files = os.listdir(os.path.join(series_dir, folder))
    series_counts.append({'SeriesInstanceUID': folder, 'NumSlices': len(dcm_files)})

df_series = pd.DataFrame(series_counts)
df_series.describe()
```

<h2 style="color:#F8766D">🧭 DICOM Voxel Orientation & Spacing</h2> 
<h4>How is the 3D scan structured?</h4> 
This snippet reveals the scan's physical properties: <b>voxel spacing</b> (pixel size in mm), <b>slice thickness</b>, and <b>patient orientation</b>. Understanding these helps convert 2D slices into accurate 3D volumes for model input.

```python
import pydicom

sample_path = os.path.join(series_dir, series_folders[0])
sample_file = os.listdir(sample_path)[0]
dcm = pydicom.dcmread(os.path.join(sample_path, sample_file))

print(f"Orientation: {dcm.ImageOrientationPatient}")
print(f"Voxel spacing: {dcm.PixelSpacing}")
print(f"Slice Thickness: {dcm.SliceThickness}")
```

<h2 style="color:#7CAE00">🖼️ Scroll Through DICOM Slices</h2>
<h4>Peek inside a brain scan — one slice at a time.</h4>
This interactive viewer lets you scroll through axial slices of a CT/MR scan using a slider. Useful for verifying data quality, orientation, and anatomical features before modeling.

```python
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets
import numpy as np

def load_series(series_path):
    files = sorted(os.listdir(series_path), key=lambda x: pydicom.dcmread(os.path.join(series_path, x)).InstanceNumber)
    images = [pydicom.dcmread(os.path.join(series_path, f)).pixel_array for f in files]
    return np.stack(images)

volume = load_series(os.path.join(series_dir, series_folders[0]))

# Scrollable plot
from ipywidgets import interact
@interact(slice=(0, volume.shape[0]-1))
def show_slice(slice=0):
    plt.imshow(volume[slice], cmap='gray')
    plt.axis('off')
    plt.show()
```

<h2 style="color:#F8766D">📐 Common DICOM Image Resolutions</h2>
<h4>Do all brain scans come in the same shape and size?</h4>
This horizontal bar chart shows the most frequent image resolutions across series. Identifying common shapes helps standardize preprocessing pipelines like resizing or padding.

```python
import os
import pydicom

# Count slices and shape per series
dicom_dir = '/kaggle/input/rsna-intracranial-aneurysm-detection/series'
series_stats = []

for series_id in os.listdir(dicom_dir)[:50]:  # limit for speed
    series_path = os.path.join(dicom_dir, series_id)
    files = os.listdir(series_path)
    num_slices = len(files)
    sample_dcm = pydicom.dcmread(os.path.join(series_path, files[0]))
    shape = (sample_dcm.Rows, sample_dcm.Columns)
    series_stats.append({"SeriesInstanceUID": series_id, "Slices": num_slices, "Shape": shape})

pd.DataFrame(series_stats).value_counts("Shape").plot(kind="barh", title="🖼 Common Image Resolutions")
```

<h2 style="color:#00BFC4">📏 Slice Thickness in Brain CT Scans</h2>
<h4>How deep is each CT slice through the brain?</h4>
This histogram illustrates the variation in slice thickness across patient scans, crucial for accurate 3D reconstruction and model generalization.

```python
voxel_data = []

for series_id in os.listdir(dicom_dir)[:50]:
    slices = []
    for f in sorted(os.listdir(os.path.join(dicom_dir, series_id))):
        path = os.path.join(dicom_dir, series_id, f)
        dcm = pydicom.dcmread(path)
        slices.append(dcm)

    try:
        spacing = slices[0].PixelSpacing
        thickness = float(slices[0].SliceThickness)
        voxel_data.append({
            "SeriesInstanceUID": series_id,
            "PixelSpacingX": spacing[0],
            "PixelSpacingY": spacing[1],
            "SliceThickness": thickness,
            "NumSlices": len(slices)
        })
    except:
        continue

voxel_df = pd.DataFrame(voxel_data)
sns.histplot(voxel_df["SliceThickness"], bins=20)
plt.title("📐 Slice Thickness Distribution")
```

<h2 style="color:#C77CFF">🧠 Vessel Segmentations: Coverage Check</h2>
<h4>How many scans come with annotated vessels?</h4>
Only a small subset of CT series (~{seg_percent:.2f}%) include segmentation masks, which can limit supervised learning for vessel or aneurysm detection tasks.

```python
seg_dir = "/kaggle/input/rsna-intracranial-aneurysm-detection/segmentations"
seg_series = [f.replace(".nii.gz", "") for f in os.listdir(seg_dir)]
print("Segmented Series:", len(seg_series))

# % of series with segmentation
seg_percent = len(seg_series) / len(os.listdir(dicom_dir)) * 100
print(f"{seg_percent:.2f}% of series have vessel segmentation.")
```

<h2 style="color:#00BFC4">🧠 Baseline 3D CNN for Aneurysm Localization</h2>
<h4>Compact and efficient model to kickstart voxel-wise aneurysm detection</h4>
This 3D CNN ingests normalized DICOM volumes and outputs multi-label predictions across 14 possible aneurysm locations plus presence. Designed for speed and flexibility.

<h3 style="color:#F8766D">🗂 DICOM Volume Preprocessing Pipeline</h3>
<h4>From unordered slices to uniform, normalized 3D tensors</h4>
Volumes are rescaled, clipped (1st–99th percentiles), normalized, and resized to 64³ using `ndimage.zoom`. Robust preprocessing ensures clean inputs for deep learning models.

```python
import os
import shutil
import gc
from collections import defaultdict
from typing import Tuple, List
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import polars as pl
import pydicom
from scipy import ndimage
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

import kaggle_evaluation.rsna_inference_server

# Competition constants
ID_COL = 'SeriesInstanceUID'
LABEL_COLS = [
    'Left Infraclinoid Internal Carotid Artery',
    'Right Infraclinoid Internal Carotid Artery',
    'Left Supraclinoid Internal Carotid Artery',
    'Right Supraclinoid Internal Carotid Artery',
    'Left Middle Cerebral Artery',
    'Right Middle Cerebral Artery',
    'Anterior Communicating Artery',
    'Left Anterior Cerebral Artery',
    'Right Anterior Cerebral Artery',
    'Left Posterior Communicating Artery',
    'Right Posterior Communicating Artery',
    'Basilar Tip',
    'Other Posterior Circulation',
    'Aneurysm Present',
]

DICOM_TAG_ALLOWLIST = [
    'BitsAllocated', 'BitsStored', 'Columns', 'FrameOfReferenceUID', 'HighBit',
    'ImageOrientationPatient', 'ImagePositionPatient', 'InstanceNumber', 'Modality',
    'PatientID', 'PhotometricInterpretation', 'PixelRepresentation', 'PixelSpacing',
    'PlanarConfiguration', 'RescaleIntercept', 'RescaleSlope', 'RescaleType', 'Rows',
    'SOPClassUID', 'SOPInstanceUID', 'SamplesPerPixel', 'SliceThickness',
    'SpacingBetweenSlices', 'StudyInstanceUID', 'TransferSyntaxUID',
]

# Model configuration
TARGET_SIZE = (64, 64, 64)  # Reduced size for memory efficiency
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DICOMProcessor:
    """Process DICOM series into normalized 3D volumes"""
    
    def __init__(self, target_size: Tuple[int, int, int] = TARGET_SIZE):
        self.target_size = target_size
        self.scaler = StandardScaler()
    
    def load_dicom_series(self, series_path: str) -> np.ndarray:
        """Load and process a DICOM series into a 3D volume"""
        try:
            # Get all DICOM files
            dicom_files = []
            for root, _, files in os.walk(series_path):
                for file in files:
                    if file.endswith('.dcm'):
                        dicom_files.append(os.path.join(root, file))
            
            if not dicom_files:
                raise ValueError(f"No DICOM files found in {series_path}")
            
            # Load DICOMs and sort by instance number
            dicoms = []
            for filepath in dicom_files:
                try:
                    ds = pydicom.dcmread(filepath, force=True)
                    if hasattr(ds, 'PixelData'):
                        dicoms.append((ds, filepath))
                except Exception as e:
                    print(f"Error reading {filepath}: {e}")
                    continue
            
            if not dicoms:
                raise ValueError(f"No valid DICOM files with pixel data in {series_path}")
            
            # Sort by instance number
            dicoms.sort(key=lambda x: getattr(x[0], 'InstanceNumber', 0))
            
            # Extract volume
            volume_slices = []
            for ds, _ in dicoms:
                try:
                    # Get pixel array
                    pixel_array = ds.pixel_array.astype(np.float32)
                    
                    # Apply rescale if available
                    if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
                        slope = float(ds.RescaleSlope)
                        intercept = float(ds.RescaleIntercept)
                        pixel_array = pixel_array * slope + intercept
                    
                    volume_slices.append(pixel_array)
                except Exception as e:
                    print(f"Error processing slice: {e}")
                    continue
            
            if not volume_slices:
                raise ValueError("No valid slices extracted")
            
            # Stack into 3D volume
            volume = np.stack(volume_slices, axis=0)  # Shape: (depth, height, width)
            
            # Normalize and resize
            volume = self.preprocess_volume(volume)
            
            return volume
            
        except Exception as e:
            print(f"Error processing series {series_path}: {e}")
            # Return zeros if processing fails
            return np.zeros(self.target_size, dtype=np.float32)
    
    def preprocess_volume(self, volume: np.ndarray) -> np.ndarray:
        """Preprocess 3D volume: normalize, clip, resize"""
        # Handle potential issues
        if volume.size == 0:
            return np.zeros(self.target_size, dtype=np.float32)
        
        # Clip extreme values (robust to outliers)
        p1, p99 = np.percentile(volume, [1, 99])
        volume = np.clip(volume, p1, p99)
        
        # Normalize to [0, 1]
        volume_min, volume_max = volume.min(), volume.max()
        if volume_max > volume_min:
            volume = (volume - volume_min) / (volume_max - volume_min)
        
        # Resize to target size
        if volume.shape != self.target_size:
            zoom_factors = [
                self.target_size[i] / volume.shape[i] for i in range(3)
            ]
            volume = ndimage.zoom(volume, zoom_factors, order=1)
        
        return volume.astype(np.float32)

class Simple3DCNN(nn.Module):
    """Lightweight 3D CNN for aneurysm detection"""
    
    def __init__(self, num_classes: int = len(LABEL_COLS)):
        super(Simple3DCNN, self).__init__()
        
        # 3D Convolutional layers
        self.conv1 = nn.Conv3d(1, 16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool3d(2)
        self.conv2 = nn.Conv3d(16, 32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool3d(2)
        self.conv3 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool3d(2)
        self.conv4 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool3d(2)
        
        # Adaptive pooling to handle variable sizes
        self.adaptive_pool = nn.AdaptiveAvgPool3d((2, 2, 2))
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 2 * 2 * 2, 256)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(128, num_classes)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm3d(16)
        self.bn2 = nn.BatchNorm3d(32)
        self.bn3 = nn.BatchNorm3d(64)
        self.bn4 = nn.BatchNorm3d(128)
        
    def forward(self, x):
        # Input shape: (batch_size, 1, depth, height, width)
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        
        # Adaptive pooling
        x = self.adaptive_pool(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        
        return torch.sigmoid(x)

class AneurysmDataset(Dataset):
    """Dataset for loading training data"""
    
    def __init__(self, data_df: pd.DataFrame, series_dir: str, processor: DICOMProcessor):
        self.data_df = data_df
        self.series_dir = series_dir
        self.processor = processor
        
    def __len__(self):
        return len(self.data_df)
    
    def __getitem__(self, idx):
        row = self.data_df.iloc[idx]
        series_id = row[ID_COL]
        
        # Load volume
        series_path = os.path.join(self.series_dir, series_id)
        volume = self.processor.load_dicom_series(series_path)
        
        # Get labels
        labels = row[LABEL_COLS].values.astype(np.float32)
        
        # Convert to tensor and add channel dimension
        volume_tensor = torch.from_numpy(volume).unsqueeze(0)  # Add channel dim
        labels_tensor = torch.from_numpy(labels)
        
        return volume_tensor, labels_tensor

# Global model and processor
model = None
processor = None

def initialize_model():
    """Initialize model and processor (called once)"""
    global model, processor
    
    if model is not None:
        return
    
    print("Initializing model...")
    processor = DICOMProcessor(TARGET_SIZE)
    model = Simple3DCNN(num_classes=len(LABEL_COLS))
    
    # Load pre-trained weights if available
    try:
        if os.path.exists('/kaggle/input/model_weights.pth'):
            model.load_state_dict(torch.load('/kaggle/input/model_weights.pth', map_location='cpu'))
            print("Loaded pre-trained weights")
        else:
            print("No pre-trained weights found, using random initialization")
    except Exception as e:
        print(f"Error loading weights: {e}")
    
    model.to(DEVICE)
    model.eval()
    print(f"Model initialized on {DEVICE}")

def predict(series_path: str) -> pl.DataFrame:
    """Make prediction for a single series"""
    
    # Initialize model on first call
    initialize_model()
    
    series_id = os.path.basename(series_path)
    
    try:
        # Process the DICOM series
        volume = processor.load_dicom_series(series_path)
        
        # Convert to tensor and add batch dimension
        volume_tensor = torch.from_numpy(volume).unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)
        volume_tensor = volume_tensor.to(DEVICE)
        
        # Make prediction
        with torch.no_grad():
            predictions = model(volume_tensor)
            predictions = predictions.cpu().numpy().flatten()
        
        # Create result DataFrame
        result_data = [[series_id] + predictions.tolist()]
        result_df = pl.DataFrame(
            data=result_data,
            schema=[ID_COL] + LABEL_COLS,
            orient='row'
        )
        
        # Clean up memory
        del volume_tensor
        torch.cuda.empty_cache()
        gc.collect()
        
    except Exception as e:
        print(f"Error predicting for series {series_id}: {e}")
        # Return baseline predictions (0.5 for all classes)
        result_data = [[series_id] + [0.5] * len(LABEL_COLS)]
        result_df = pl.DataFrame(
            data=result_data,
            schema=[ID_COL] + LABEL_COLS,
            orient='row'
        )
    
    # Mandatory cleanup
    shutil.rmtree('/kaggle/shared', ignore_errors=True)
    
    return result_df.drop(ID_COL)

def train_model(train_df_path: str, series_dir: "/kaggle/input/rsna-intracranial-aneurysm-detection/series", num_epochs: int = 50, batch_size: int = 32):
    """Training function (for reference - would be run separately)"""
    
    # Load training data
    train_df = pd.read_csv("/kaggle/input/rsna-intracranial-aneurysm-detection/train.csv")
    
    # Initialize components
    processor = DICOMProcessor(TARGET_SIZE)
    model = Simple3DCNN(num_classes=len(LABEL_COLS))
    model.to(DEVICE)
    
    # Create dataset and dataloader
    dataset = AneurysmDataset(train_df, series_dir, processor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2)
    
    # Training loop
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0
        
        for batch_idx, (volumes, labels) in enumerate(dataloader):
            volumes, labels = volumes.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(volumes)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 10 == 0:
                print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}, Loss: {loss.item():.4f}')
        
        avg_loss = total_loss / num_batches
        scheduler.step(avg_loss)
        print(f'Epoch {epoch+1}/{num_epochs} completed, Average Loss: {avg_loss:.4f}')
    
    # Save model
    torch.save(model.state_dict(), 'model_weights.pth')
    print("Model saved to model_weights.pth")

# Competition server setup
inference_server = kaggle_evaluation.rsna_inference_server.RSNAInferenceServer(predict)

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    inference_server.run_local_gateway()
    display(pl.read_parquet('/kaggle/working/submission.parquet'))
```