# RSNA2025 - Explore and Gain Insights

- **Author:** Amirmohammad Mahdavikia
- **Votes:** 109
- **Ref:** amirmmahdavikia/rsna2025-explore-and-gain-insights
- **URL:** https://www.kaggle.com/code/amirmmahdavikia/rsna2025-explore-and-gain-insights
- **Last run:** 2025-08-27 09:39:17.143000

---

![rsna2025_cover](https://raw.githubusercontent.com/ammomahdavikia/asset-holding/main/rsna2025_cover.png)

This notebook is part of a series exploring and designing a pipeline to detect and classify intracranial aneurysms as part of RSNA 2025 AI challenge. You can find the other notebooks below:

+ **Part 1:** [Explore and Gain Insights](https://www.kaggle.com/code/amirmmahdavikia/rsna2025-explore-and-gain-insights) <--- [You are here!]
+ **Part 2:** [Preprocessing Guide]()
+ **...**

❗ This series is under production, stay tuned for upcoming versions. ❗

# <div style="text-align:center; border-radius:10px; color:black; margin:0; font-size:100%; font-family:Carbon Plus; background-color:white; overflow:visible; line-height:1.5; padding:10px 0;"><b>🎈 What is Aneurysm?</b></div>

An **aneurysm** is a localized, abnormal bulging or ballooning of a blood vessel wall. It frequently occurs at areas with high blood pressure. As expected, arteries have higher blood pressure and flow velocity, making them the primary sites for aneurysm development.

![vessel_types](https://raw.githubusercontent.com/ammomahdavikia/asset-holding/main/vessel_types.png)

 It can occur at various sites in the body but most frequently and critically occur in larger arteries such as **Aorta**, **Femoral**, **Iliac**, **Popliteal** and **Cerebral** arteries. In this notebook we will particularly focus on brain/cerebral arteries.
 
**Intracranial Aneurysms (IAs)** are relatively common life-threatening disease with prevalence of 3.2% in general population. They can lead to subarachnoid hemorrhage with high mortality in case of rupture.

Aneurysm formation occurs when the blood vessel wall weakens due to a combination of:
+ hemodynamic stress
+ degeneration of structural components
+ inflammation
+ atherosclerosis
+ genetic disorders 
+ infection

Arterial walls are composed of three main layers: _tunica intima_, _tunica media_, and _adventitia_.

A **true aneurysm** involves the dilation of all three intact layers of the arterial wall. In contrast, a **false aneurysm** (or **pseudoaneurysm**) occurs when there is a breach in the vessel wall, resulting in an extravascular blood collection that remains connected to the vessel lumen.

![aneurysm_types_1.png](https://raw.githubusercontent.com/ammomahdavikia/asset-holding/main/aneurysm_types_1.png)


#### 🔎 Why does this matter for us?
For machine learning, these definitions highlight the complexity of the detection task: aneurysms vary in size, shape, and location, often blending into surrounding vessels. Successful models must capture these subtle differences to both detect presence and localize exact sites of aneurysms within 3D neuroimaging.

### ↓ **Libraries**

### ↓ **Imports**

```python
import os
import requests
import re
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import nibabel as nib
import pydicom
import ipywidgets as widgets
from IPython.display import display_html, display, Markdown
import warnings
import ast
import cv2
from scipy.interpolate import interp1d
from scipy.ndimage import zoom
from tqdm.auto import tqdm
from glob import glob
from pathlib import Path
import plotly.io as pio
pio.renderers.default = 'iframe'
warnings.simplefilter(action='ignore', category=FutureWarning)

class clr:
    B = '\033[1m'
    L = '\033[1m' + '\033[94m'
    T = '\033[1m' + '\033[91m'
    E = '\033[0m'

print(clr.B+'\n----- Font -----\n'+clr.E)
font_url = 'https://raw.githubusercontent.com/ammomahdavikia/asset-holding/main/carbonplus-regular-bl.otf'
output_path = 'carbonplus-regular-bl.otf'

response = requests.get(font_url, stream=True)

if response.status_code == 200:
    with open(output_path, 'wb') as f:
        for chunk in response.iter_content(1024):
            f.write(chunk)
    print(f'[Success] downloaded: {output_path}')
else:
    print('[Error] Failed to download the font. Check the URL.')

font_path = '/kaggle/working/carbonplus-regular-bl.otf'
try:
    fm.fontManager.addfont(font_path)
    primary_font = fm.FontProperties(fname=font_path).get_name()
    plt.rcParams['font.family'] = [primary_font, "DejaVu Sans", "Noto Sans CJK JP"]
    print('[Success] loaded.')
except:
    print('[Error] Failed to load the font. Check the address.')

COLORS = ['#780000', '#c1121f', '#ffb703', '#02c39a', '#669bbc', '#003049']
aneurysm_palette = {1: COLORS[1], 0: COLORS[-1]}
modality_palette = {'CTA': COLORS[4], 'MRA': COLORS[3],
                    'MRI T2': COLORS[2], 'MRI T1post': COLORS[1]}
plane_palette = {'axial': COLORS[0], 'coronal': COLORS[2], 'sagittal': COLORS[-1]}
print(clr.B+'\n----- Color -----\n'+clr.E)
sns.palplot(sns.color_palette(COLORS))
```

### ↓ **Helpers**

```python
series_cols = ['SeriesInstanceUID', 'FrameOfReferenceUID', 'SOPClassUID', 'IsMultiFrame', 'Rows', 'Columns', 'NumberOfFrames', 'Plane']
instance_cols = ['SeriesInstanceUID', 'FrameOfReferenceUID', 'SOPClassUID', 'SOPInstanceUID', 'FilePath',
                 'Modality', 'Columns', 'Rows', 'PixelRepresentation', 'WindowWidth', 'WindowCenter',
                 'BitsAllocated', 'BitsStored', 'HighBit', 'RescaleIntercept', 'RescaleSlope', 'RescaleType',
                 'SliceThickness', 'SpacingBetweenSlices', 'PixelSpacing_X', 'PixelSpacing_Y',
                 'IPP_X', 'IPP_Y', 'IPP_Z', 'IOP_RowX', 'IOP_RowY', 'IOP_RowZ', 'IOP_ColX', 'IOP_ColY', 'IOP_ColZ',
                 'Plane', 'InstanceNumber']


def normalize_value(value):
    if isinstance(value, pydicom.multival.MultiValue):
        return list(value)
    elif isinstance(value, (pydicom.valuerep.DSfloat, pydicom.valuerep.DSdecimal)):
        return float(value)
    elif isinstance(value, pydicom.valuerep.IS):
        return int(value)
    return value

def get_series_metadata(dataset):
    tags = ['SeriesInstanceUID', 'FrameOfReferenceUID', 
            'SOPClassUID', 'Rows', 'Columns']
    metadata = {tag: [normalize_value(dataset.get(tag))] for tag in tags}
    
    num_frames = getattr(dataset, 'NumberOfFrames', None)
    metadata["IsMultiFrame"] = [num_frames is not None and num_frames > 1]
    
    return metadata

def get_singleframe_instance_metadata(dataset, file_path):
    tags = ['BitsAllocated', 'BitsStored', 'Columns', 'FrameOfReferenceUID', 
            'HighBit', 'Modality', 'WindowWidth', 'WindowCenter',
            'PixelRepresentation', 'SeriesInstanceUID',
            'RescaleIntercept', 'RescaleSlope', 'RescaleType', 'Rows', 
            'SOPClassUID', 'SliceThickness', 
            'SpacingBetweenSlices']

    metadata = {}
    for tag in tags:
        metadata[tag] = normalize_value(dataset.get(tag))

    metadata["SOPInstanceUID"] = dataset.get("SOPInstanceUID")
    metadata["FilePath"] = file_path

    pi = dataset.get("PixelSpacing")
    if pi and len(pi) == 2:
        pi = list(map(float, pi))
        metadata["PixelSpacing_X"], metadata["PixelSpacing_Y"] = pi
    else:
        metadata["PixelSpacing_X"], metadata["PixelSpacing_Y"] = np.nan
    
    # ImagePositionPatient into X, Y, Z
    ipp = dataset.get("ImagePositionPatient")
    if ipp and len(ipp) == 3:
        ipp = list(map(float, ipp))
        metadata["IPP_X"], metadata["IPP_Y"], metadata["IPP_Z"] = ipp
    else:
        metadata["IPP_X"] = metadata["IPP_Y"] = metadata["IPP_Z"] = np.nan

    # ImageOrientationPatient into Row/Col direction cosines
    iop = dataset.get("ImageOrientationPatient")
    if iop and len(iop) == 6:
        iop = list(map(float, iop))
        metadata["IOP_RowX"], metadata["IOP_RowY"], metadata["IOP_RowZ"] = iop[:3]
        metadata["IOP_ColX"], metadata["IOP_ColY"], metadata["IOP_ColZ"] = iop[3:]
    else:
        for k in ["IOP_RowX", "IOP_RowY", "IOP_RowZ", "IOP_ColX", "IOP_ColY", "IOP_ColZ"]:
            metadata[k] = np.nan

    return metadata

def get_multiframe_instance_metadata(dataset, file_path):
    shared = dataset.SharedFunctionalGroupsSequence[0]
    per_frame = dataset.PerFrameFunctionalGroupsSequence

    # Shared attributes
    iop = shared.PlaneOrientationSequence[0].ImageOrientationPatient
    ps = shared.PixelMeasuresSequence[0].PixelSpacing
    st = shared.PixelMeasuresSequence[0].SliceThickness
    sbs = getattr(shared.PixelMeasuresSequence[0], 'SpacingBetweenSlices', None)

    shared_fields = {
        'IOP_RowX': float(iop[0]), 'IOP_RowY': float(iop[1]), 'IOP_RowZ': float(iop[2]),
        'IOP_ColX': float(iop[3]), 'IOP_ColY': float(iop[4]), 'IOP_ColZ': float(iop[5]),
        'PixelSpacing_X': float(ps[0]), 'PixelSpacing_Y': float(ps[1]),
        'SliceThickness': float(st),
        'SpacingBetweenSlices': float(sbs) if sbs is not None else None,
    }

    tags = ['BitsAllocated', 'BitsStored', 'Columns', 'FrameOfReferenceUID', 
            'HighBit', 'Modality', 'WindowWidth', 'WindowCenter',
            'PixelRepresentation', 'SeriesInstanceUID',
            'RescaleIntercept', 'RescaleSlope', 'RescaleType', 'Rows', 
            'SOPClassUID']

    for tag in tags:
        shared_fields[tag] = normalize_value(dataset.get(tag))

    metadata = []

    for frame_index, frame in enumerate(per_frame):
        ipp = frame.PlanePositionSequence[0].ImagePositionPatient
        ipp = list(map(float, ipp))

        instance = {
            "SOPInstanceUID": dataset.SOPInstanceUID,  # same for all frames
            "FilePath": file_path,
            "IPP_X": ipp[0], "IPP_Y": ipp[1], "IPP_Z": ipp[2],
            **shared_fields
        }
        metadata.append(instance)

    return metadata

def process_series(series_dir):
    dcm_files = [os.path.join(series_dir, f) for f in os.listdir(series_dir) if f.lower().endswith('.dcm')]
    instance_records = []
    ds_first = None

    for file in dcm_files:
        try:
            ds = pydicom.dcmread(file, stop_before_pixels=True)
            if ds_first is None:
                ds_first = ds
                
            num_frames = getattr(ds, 'NumberOfFrames', None)
            if isinstance(num_frames, (int, float)) and num_frames > 1:
                multi_instances = get_multiframe_instance_metadata(ds, file)
                instance_records.extend(multi_instances)
                continue

            metadata = get_singleframe_instance_metadata(ds, file)
            instance_records.append(metadata)

        except Exception as e:
            print(f"[ERROR] Could not read {file}: {e}")

    for i, rec in enumerate(instance_records):
        row_cos = np.array([rec['IOP_RowX'], rec['IOP_RowY'], rec['IOP_RowZ']])
        col_cos = np.array([rec['IOP_ColX'], rec['IOP_ColY'], rec['IOP_ColZ']])
        normal = np.cross(row_cos, col_cos)
        axis = np.abs(normal)
        labels = ['sagittal', 'coronal', 'axial']
        directions = ['X', 'Y', 'Z']
        rec['Plane'] = labels[int(np.argmax(axis))]
        slice_direction = directions[int(np.argmax(axis))]
    
    instance_records = sorted(instance_records, key=lambda x: x.get(f"IPP_{slice_direction}", float('inf')))

    for i, rec in enumerate(instance_records):
        rec["InstanceNumber"] = i + 1

    if ds_first:
        series_meta = get_series_metadata(ds_first)
        series_meta['Plane'] = instance_records[0]['Plane']
        series_meta['NumberOfFrames'] = len(instance_records)
    else:
        series_meta = {}

    instance_records = pd.DataFrame(instance_records)[instance_cols]
    series_meta = pd.DataFrame(series_meta)[series_cols]
    
    return instance_records, series_meta

def extract_dicom_metadata(root_dir):
    """
    Traverse all series directories in root_dir, extract instance and series metadata,
    and return concatenated DataFrames.
    
    Returns:
        all_instances_df: pd.DataFrame with instance-level metadata
        all_series_df: pd.DataFrame with series-level metadata
    """
    all_instances = []
    all_series = []

    series_dirs = [
        os.path.join(root_dir, d) for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d))
    ]

    for idx, series_dir in enumerate(tqdm(series_dirs, desc='Extract DICOM Metadata')):
        try:
            instance_df, series_df = process_series(series_dir)

            all_instances.append(instance_df)
            all_series.append(series_df)
        except Exception as e:
            print(f"[ERROR] Failed to process {series_dir}: {e}")

    if all_instances:
        all_instances_df = pd.concat(all_instances, ignore_index=True)
    else:
        all_instances_df = pd.DataFrame(columns=instance_cols)

    if all_series:
        all_series_df = pd.concat(all_series, ignore_index=True)
    else:
        all_series_df = pd.DataFrame(columns=series_cols)

    return all_instances_df, all_series_df

def load_dicom(dir_path, target_shape=(512, 512)):
    """
    Load DICOM images from a directory and return as a 3D NumPy array.
    
    Handles both multi-frame and single-slice DICOMs. Resizes each slice to target_shape.
    """
    dcm_files = sorted(glob(os.path.join(dir_path, "*.dcm")))

    if len(dcm_files) == 1:
        dcm = pydicom.dcmread(dcm_files[0])
        if hasattr(dcm, 'NumberOfFrames'):
            frames = dcm.pixel_array
            resized = np.stack([cv2.resize(f, target_shape, interpolation=cv2.INTER_LINEAR)
                                for f in frames], axis=0)
            return resized
        else:
            single = dcm.pixel_array
            resized = cv2.resize(single, target_shape, interpolation=cv2.INTER_LINEAR)
            return np.expand_dims(dcm.pixel_array, 0)
    else:
        slices = [pydicom.dcmread(f) for f in dcm_files]
        slices.sort(key=lambda x: int(x.ImagePositionPatient[2]))
        resized_slices = [cv2.resize(s.pixel_array, target_shape, interpolation=cv2.INTER_LINEAR)
                          for s in slices]
        volume = np.stack(resized_slices, axis=0)
        return volume

def load_nii(path, mask=False):
    nii = nib.load(path)
    nii = nib.as_closest_canonical(nii)
    nii_array = nii.get_fdata()
    
    pixdim = nii.header.get_zooms()[:3]
    target_spacing = (1.0, 1.0, 1.0)
    zoom_factors = tuple(pixdim[i] / target_spacing[i] for i in range(3))
    nii_array = zoom(nii_array, zoom_factors, order = 0 if mask else 1)
    
    if mask:
        nii_array = np.clip(nii_array, 0, 13).astype(np.uint8)

    nii_array = np.rot90(nii_array, k=1, axes=(0, 1))
    nii_array = np.ascontiguousarray(nii_array)
    
    return nii_array
```

# <div style="text-align:center; border-radius:10px; color:black; margin:0; font-size:100%; font-family:Carbon Plus; background-color:white; overflow:visible; line-height:1.5; padding:10px 0;"><b>📂 Loading the data</b></div>

**1. Original Data (Competition Release)**

+ Imaging files in DICOM format.
+ Contains multi-modal studies (CTA, MRA, MRI T1 post-contrast, MRI T2).
+ Includes aneurysm labels and localization annotations.
+ Segmented images and masks in NIfTI format.

**2. Processed Metadata (Pre-Extracted)**

+ Contains key attributes extracted from DICOM.
    + Restricted to `['BitsAllocated', 'BitsStored', 'Columns', 'FrameOfReferenceUID', 'HighBit', 'ImageOrientationPatient', 'ImagePositionPatient', 'InstanceNumber', 'Modality', 'PatientID', 'PhotometricInterpretation', 'PixelRepresentation', 'PixelSpacing', 'PlanarConfiguration', 'RescaleIntercept', 'RescaleSlope', 'RescaleType', 'Rows', 'SOPClassUID', 'SOPInstanceUID', 'SamplesPerPixel', 'SliceThickness', 'SpacingBetweenSlices', 'StudyInstanceUID', 'TransferSyntaxUID']`.
    + Plus `['WindowWidth', 'WindowCenter']`.

```python
CONFIG = {
    'PATH': '/kaggle/input/rsna-intracranial-aneurysm-detection',
    'META': '/kaggle/input/rsna2025-iad-metadata',
    'ARTERIES': {
        1: "Other Posterior Circulation",
        2: "Basilar Tip",
        3: "Right Posterior Communicating Artery",
        4: "Left Posterior Communicating Artery",
        5: "Right Infraclinoid Internal Carotid Artery",
        6: "Left Infraclinoid Internal Carotid Artery",
        7: "Right Supraclinoid Internal Carotid Artery",
        8: "Left Supraclinoid Internal Carotid Artery",
        9: "Right Middle Cerebral Artery",
        10: "Left Middle Cerebral Artery",
        11: "Right Anterior Cerebral Artery",
        12: "Left Anterior Cerebral Artery",
        13: "Anterior Communicating Artery"
    },
}

os.makedirs('Figures', exist_ok=True)
```

```python
def load_data(metadata=False):

    if metadata:
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        train_df = pd.read_csv(os.path.join(CONFIG['META'], 'train.csv'))
        train_loc = pd.read_csv(os.path.join(CONFIG['META'], 'train_localizers.csv'))
        train_instances = pd.read_csv(os.path.join(CONFIG['META'], 'train_instances.csv'))
        return train_df, train_loc, train_instances
    else:
        train_df = pd.read_csv(os.path.join(CONFIG['PATH'], 'train.csv'))
        train_loc = pd.read_csv(os.path.join(CONFIG['PATH'], 'train_localizers.csv'))
        return train_df, train_loc

def summarize(df, desc='Summary'):

    start = clr.T
    if 'Localization' in desc:
        start = clr.L
    
    print(start+f'\n----- {desc} -----\n'+clr.E)
    print(f'Shape: {df.shape}')
    print(f'Missing: {df.isna().sum().sum()}')
    print(f'Columns: {df.columns.to_list()}\n')
    display_html(df.head(3))
    print('\n')
```

### ↓ **Original Data**

```python
train_df, train_loc = load_data()

for df, desc in zip([train_df, train_loc],
                    ['Train', 'Localization']):
    summarize(df, desc)
```

### ↓ **Pre-processed Data**

I have already ran and added to the input. It takes about 2.5hr.

```python
def process_data(train_df, train_loc):
    
    root_dir = os.path.join(CONFIG['PATH'], 'series')
    instances_df, series_df = extract_dicom_metadata(root_dir)

    train_df = pd.merge(train_df, series_df, on='SeriesInstanceUID', how='left')

    coords = train_loc['coordinates'].apply(ast.literal_eval)
    coords_df = pd.DataFrame(coords.tolist(), index=train_loc.index)
    train_loc = train_loc.copy()
    train_loc.loc[:, 'x'] = coords_df['x']
    train_loc.loc[:, 'y'] = coords_df['y']

    shape_df = train_df[['SeriesInstanceUID', 'Rows', 'Columns']].copy()
    train_loc = pd.merge(train_loc, shape_df, on='SeriesInstanceUID', how='left')

    train_loc['x_norm'] = train_loc['x'] / train_loc['Columns']
    train_loc['y_norm'] = train_loc['y'] / train_loc['Rows']
    
    train_df.to_csv('train.csv', index=False)
    train_loc.to_csv('train_localizers.csv', index=False)
    instances_df.to_csv('train_instances.csv', index=False)

    return train_df, instances_df, train_loc
```

```python
# train_df, train_instances, train_loc = process_data(train_df, train_loc)

train_df, train_loc, train_instances = load_data(metadata=True)
```

```python
for df, desc in zip([train_df, train_instances, train_loc],
                    ['New Train', 'Instances', 'New Localization']):
    summarize(df, desc)
```

```python
train_loc = train_loc.drop(columns=['Rows', 'Columns'])
merged_df = pd.merge(train_loc, train_df, on='SeriesInstanceUID', how='left')
```

# <div style="text-align:center; border-radius:10px; color:black; margin:0; font-size:100%; font-family:Carbon Plus; background-color:white; overflow:visible; line-height:1.5; padding:10px 0;"><b>🔍 Exploration</b></div>


We will start with the original data attributes and then cover the metadata attributes later.

```python
train_pivot = pd.pivot_table(
    train_df,
    index='Aneurysm Present',
    columns='Modality',
    values='PatientAge',
    aggfunc='count',
    margins=True,
    margins_name='Total'
)

fig, ax = plt.subplots(figsize=(6, 4))
sns.heatmap(
    train_pivot, 
    annot=True,
    fmt='g',
    cmap='Reds_r',
    linewidths=0.5, 
    linecolor='gray',
    ax=ax
    
)

ax.set_title('Patient Count by Modality and Aneurysm Presence')
ax.set_ylabel('Aneurysm Present')
ax.set_xlabel('Modality')
fig.tight_layout()
fig.show()
fig.savefig('Figures/series_count_per_modality_and_aneurysm.png', dpi=300)
```

```python
aneurysm_count = train_df['Aneurysm Present'].value_counts()
aneurysm = aneurysm_count.index.to_list()
count = aneurysm_count.to_list()
total = sum(count)
proportions = [c / total for c in count]

fig = plt.figure(figsize=(12, 6))
gs = gridspec.GridSpec(
    nrows=2, ncols=3, 
    height_ratios=[1, 8], 
    width_ratios=[4, 3, 0.01])  # 3rd col is dummy

ax0 = fig.add_subplot(gs[0, :2])

ax0.barh(y=0, width=proportions[0], left=0, 
         color=aneurysm_palette[aneurysm[0]], 
         label=f'{aneurysm[0]}: {count[0]}')
ax0.barh(y=0, width=proportions[1], left=proportions[0], 
         color=aneurysm_palette[aneurysm[1]], 
         label=f'{aneurysm[1]}: {count[1]}')

ax0.set_title('Aneurysm Presence')
ax0.set_ylabel('')
ax0.set_xlabel('')

ax0.text(proportions[0]/2, 0, f'{count[0]}', 
         ha='center', va='center', color='white', fontsize=10)
ax0.text(proportions[0] + proportions[1]/2, 0, f'{count[1]}',
         ha='center', va='center', color='white', fontsize=10)

for spine in ax0.spines.values():
    spine.set_visible(False)
ax0.tick_params(left=False, bottom=False)
ax0.set_xticklabels([])
ax0.set_yticklabels([])

ax1 = fig.add_subplot(gs[1, 0])
ax2 = fig.add_subplot(gs[1, 1])

# How many Aneurysm?
sns.histplot(
    train_df.loc[:, CONFIG['ARTERIES'].values()].sum(axis=1), 
    discrete=True,
    ax=ax1
)

n_bins = len(ax1.patches)
h_colors = [COLORS[-1]] + [COLORS[1] for i in range(1, n_bins)]

for patch, color in zip(ax1.patches, h_colors):
    patch.set_facecolor(color)

    height = patch.get_height()
    x = patch.get_x() + patch.get_width() / 2

    if height > 0:
        ax1.annotate(f'{int(height)}',
                    xy=(x, height),
                    xytext=(0, 3),
                    textcoords='offset points',
                    ha='center', va='bottom',
                    color=color,
                    fontsize=10, fontweight='bold')

ax1.set_title('How many Aneurysms?')
ax1.set_ylim(0, 2700)

# Sex distribution
sns.histplot(data=train_df, x='PatientSex', hue='Aneurysm Present', multiple='dodge',
             shrink=0.8, palette=aneurysm_palette, ax=ax2)
ax2.set_title('Sex distribution')
ax2.set_xlabel('')
ax2.set_ylabel('')
ax2.set_ylim(0, 2700)

fig.tight_layout()
fig.savefig("Figures/aneurysm_and_sex_distribution.png", dpi=300)
plt.show()
```

```python
fig = plt.figure(figsize=(10, 7))
gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.3)

ax0 = fig.add_subplot(gs[0])
sns.histplot(
    data=train_df,
    x='PatientAge',
    hue='Aneurysm Present',
    kde=True,
    palette=aneurysm_palette,
    ax=ax0
)
ax0.set_title('Age distribution')

ax1 = fig.add_subplot(gs[1], sharex=ax0)
sns.boxplot(
    data=train_df,
    x='PatientAge',
    y='Aneurysm Present',
    palette=aneurysm_palette,
    ax=ax1,
    orient='h'
)

plt.setp(ax0.get_xticklabels(), visible=False)
ax0.set_xlabel('')

fig.savefig('Figures/age_distribution.png', dpi=300)
plt.show()
```

```python
artery_count = train_df.loc[:, CONFIG['ARTERIES'].values()].sum(axis=0)
artery_count = artery_count.reset_index()
artery_count.columns = ['Artery', 'Count']

fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(
    data=artery_count, 
    y='Artery', 
    x='Count', 
    palette='Reds_r',
    ax=ax
)

ax.set_title('Artery Aneurysm Count')
ax.set_xlabel('Count')
ax.set_ylabel('Artery')
fig.tight_layout()
fig.show()
fig.savefig('Figures/artery_distribution.png', dpi=300)
```

### <div style="text-align:center; border-radius:10px; color:black; margin:0; font-size:100%; font-family:Carbon Plus; font-weight:bold; background-color:white; overflow:hidden"><b>DICOM Metadata</b></div>

There are two type of DICOMs in the dataset. 
+ **All (4348 series)**: 
    + **Single-frame (4026)**: A single-frame DICOM series stores each image slice as a separate file, making it widely compatible and easy to handle per-slice, but slower to load and manage for large datasets. 
    + **Multi-frame (322)**: A multiframe DICOM stores all slices or time frames in one file (3D) with shared metadata, enabling faster transfer and reduced I/O overhead, but requiring more complex parsing and having less support in older systems.

![DICOM_types](https://raw.githubusercontent.com/ammomahdavikia/asset-holding/main/DICOM_types.png)

### <div style="text-align:center; border-radius:10px; color:black; margin:0; font-size:100%; font-family:Carbon Plus; background-color:white; overflow:hidden"><b>NumberOfFrames</b></div>

```python
fig, axs = plt.subplots(ncols=2, figsize=(10, 6), gridspec_kw={'width_ratios': [3, 1]})

sns.histplot(
    data=train_df,
    x='NumberOfFrames',
    hue="Aneurysm Present",
    kde=True,
    palette=aneurysm_palette,
    ax=axs[0]
)

axs[0].set_title('Frame distribution')

multiframe = train_df.IsMultiFrame.value_counts()
axs[1].pie(
    multiframe,
    labels=multiframe.index,
    colors=[COLORS[1], COLORS[3]],
    autopct='%1.1f%%'
)
axs[1].set_title('Is Series MultiFrame?')

fig.tight_layout()
fig.show()
fig.savefig('Figures/frame_distribution_per_aneurysm.png', dpi=300)
```

### <div style="text-align:center; border-radius:10px; color:black; margin:0; font-size:100%; font-family:Carbon Plus; background-color:white; overflow:hidden"><b>Rows + Columns</b></div>

+ (512, 512) shape is the most common dimension.

```python
g = sns.jointplot(
    data=train_df, 
    x="Columns", 
    y="Rows",
    palette=aneurysm_palette,
    hue="Aneurysm Present",
    height=6
)

g.figure.savefig('Figures/image_dimensions_per_aneurysm_distribution.png', dpi=300)
plt.show()
```

# <div style="text-align:center; border-radius:10px; color:black; margin:0; font-size:100%; font-family:Carbon Plus; background-color:white; overflow:visible; line-height:1.5; padding:10px 0;"><b>🩻 Imaging Modalities</b></div>

Vascular imaging is an essential step in the accurate diagnosis and treatment of intracranial aneurysms, but a fundamental challenge arises from the extensive heterogeneity of aneurysms.

There is extensive heterogeneity in aneurysm characteristics including location, size, shape, patient demographics, and clinical status that leads to a great diversity in both surgical and endovascular treatment options.

Standard modalities include computed tomographic angiography (CTA), magnetic resonance angiography (MRA), and digital subtraction angiography (DSA), but a range of emerging tools, including high-resolution vessel wall imaging (HR-VWI), intraluminal imaging, and computational fluid dynamics (CFD), may transform aneurysm evaluation and treatment in the coming years.

Let's explore the standard modalities:

![imaging_modalities.png](https://raw.githubusercontent.com/ammomahdavikia/asset-holding/main/imaging_modalities.png)

##### **Digital Subtraction Angiography (DSA)**
The true sensitivity and specificity of DSA for detecting intracranial aneurysms is unknown given that it is the gold standard imaging modality. However, the false-negative rate in patients with nontraumatic SAH was found to be 7.1% (2/28) in 1 retrospective study of 904 patients.

##### **Computed Tomographic Angiography (CTA)**
The sensitivity of CTA in detecting aneurysms is good, typically ranging from **96% to 98%**. However, compared with DSA, CTA may miss small aneurysms and falsely identify infundibula and venous structures as aneurysms.

**Augmentation?** Compared to DSA and MRA, CTA has a low vessel-to-background contrast, making many structures visible and visually indistinguishable. Some actions in papers:
+ intensity clipping
+ smoothing
+ bone removal

##### **Magnetic Resonance Angiography (MRA)**
MRA can be performed with and without the need for radiation or contrast.
+ **Time-of-flight MRA (TOF-MRA)**
	+ There is controversy in the literature about the efficiency of the TOF-MRA:
		+ The sensitivity of MRA for detecting even small intracranial aneurysms (≤5 mm) has also been reported to be high, ranging from **98.2% to 98.7%** in 1 study.
		+ However, a recent large retrospective analysis comparing MRA with DSA found that the MRA sensitivity was only **50.9%** (58/114) for detecting the irregular shape of small aneurysms < 5 mm, which is associated with rupture risk.
+ **Contrast-enhanced MRA (CE-MRA)**
	+ One important application for CE-MRA over TOF-MRA is superior diagnostic accuracy in determination of residual aneurysms after treatment.
	+ Other advantages including fast image acquisition, decreased motion artifact and larger field of view coverage.

**Augmentation?** As image data may be acquired from different scanners or at different times with varying settings, it is essential to apply intensity standardization or normalization, such that all the image datasets are within the same intensity range.

| **Imaging modality**       | **Strengths**                                                | **Weaknesses**                                                                                                           |
| -------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------ |
| **Standard imaging tools** |                                                              |                                                                                                                          |
| **MRA**                    | Noninvasive  <br> No ionizing radiation                      | Lower spatial resolution  <br> No temporal resolution  <br> Long acquisition time  <br> Blood flow artifacts with TOF    |
| **CTA**                    | Noninvasive  <br> Rapid acquisition  <br> Low cost           | Ionizing radiation  <br> No temporal resolution  <br> Iodinated contrast side effects  <br> Artifact from metal and bone |
| **DSA**                    | Highest spatial resolution  <br> Highest temporal resolution | Invasive (risk of complications)  <br> Ionizing radiation  <br> High cost                                                |

```python
modality_counts = train_df['Modality'].value_counts().reset_index()
modality_counts.columns = ['Modality', 'Count']
modality_counts = modality_counts.sort_values('Modality')

bar_colors = modality_counts['Modality'].map(modality_palette)

fig, ax = plt.subplots(figsize=(6, 4))
sns.barplot(
    data=modality_counts,
    x='Modality',
    y='Count',
    palette=bar_colors,
    ax=ax
)

ax.set_title('Modality Distribution')
ax.set_xlabel('')
fig.tight_layout()
fig.savefig('Figures/modality_distribution.png', dpi=300)
plt.show()
```

```python
artery_per_modality_count = (
    train_df
    .groupby('Modality')[list(CONFIG['ARTERIES'].values())]
    .sum()
    .T
)

artery_per_modality_count.reset_index(inplace=True)
artery_per_modality_count.rename(columns={'index': 'Artery'}, inplace=True)

fig, ax = plt.subplots(figsize=(10, 6))

bottom = np.zeros(len(artery_per_modality_count))

modalities = artery_per_modality_count.columns[1:]
colors = [modality_palette[m] for m in modalities]

for modality, color in zip(modalities, colors):
    counts = artery_per_modality_count[modality]
    ax.barh(
        artery_per_modality_count['Artery'],
        counts,
        left=bottom,
        label=modality,
        color=color
    )
    bottom += counts

ax.set_title('Artery Aneurysm per Modality Count')
ax.set_xlabel('Count')
ax.set_ylabel('Artery')
ax.legend(title='Modality')
ax.invert_yaxis()

fig.tight_layout()
fig.show()
fig.savefig('Figures/artery_per_modality_distribution.png', dpi=300)
```

```python
fig, ax = plt.subplots(figsize=(8, 5))

sns.histplot(
    data=train_df,
    x='NumberOfFrames',
    hue="Modality",
    kde=True,
    palette=modality_palette,
    ax=ax
)

ax.set_title('Frame distribution per modality')
fig.tight_layout()
fig.show()
fig.savefig('Figures/frame_per_modality_distribution.png', dpi=300)
```

```python
g = sns.jointplot(
    data=train_df, 
    x="Columns", 
    y="Rows",
    palette=modality_palette,
    hue="Modality",
    height=6
)

g.figure.savefig('Figures/image_dimensions_per_modality_distribution.png', dpi=300)
plt.show()
```

```python
def plot_multiple_series(modality=None, plane=None, n_samples=5):
    """
    Plots mid-slices from multiple randomly selected DICOM series 
    matching the specified modality and/or plane (if given).
    
    Parameters:
        modality (str or None): Filter by modality (optional).
        plane (str or None): Filter by plane (optional).
        n_samples (int): Number of series to plot (default: 8).
    """
    ncols = 5
    nrows = int(np.ceil(n_samples / ncols))

    if modality == 'MRA' and plane == 'sagittal':
        ncols = 1
        n_samples = 1
    
    condition = pd.Series(True, index=merged_df.index)
    if modality is not None:
        condition &= (merged_df['Modality'] == modality)
    if plane is not None:
        condition &= (merged_df['Plane'] == plane)

    series_list = merged_df.loc[condition, 'SeriesInstanceUID'].unique()

    if len(series_list) == 0:
        filters = []
        if modality: filters.append(f"modality='{modality}'")
        if plane: filters.append(f"plane='{plane}'")
        raise ValueError(f"No series found for {' and '.join(filters) or 'any condition'}.")

    if len(series_list) < n_samples:
        print(f"Warning: Only {len(series_list)} series available. Plotting all.")
        n_samples = len(series_list)

    selected_series = np.random.choice(series_list, size=n_samples, replace=False)
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4 * ncols, 4 * nrows))
    axs = axs.flatten()

    for idx, series in enumerate(selected_series):
        series_path = os.path.join(CONFIG['PATH'], 'series', series)

        try:
            array = load_dicom(series_path)
            mid_slice = array.shape[0] // 2
            axs[idx].imshow(array[mid_slice], cmap='gray')
            axs[idx].axis('off')
            axs[idx].set_title(series, fontsize=6)
        except Exception as e:
            axs[idx].set_title(f"Error loading: {series}", fontsize=6)
            axs[idx].axis('off')

    for j in range(n_samples, len(axs)):
        axs[j].axis('off')

    mod_text = modality.upper() if modality else 'Any Modality'
    sec_text = plane if plane else 'Any Plane'
    fig.suptitle(f"{sec_text} {mod_text} — Random Mid-Slices", fontsize=16)

    fig.tight_layout()
    fig.subplots_adjust(top=0.90)

    os.makedirs("Figures", exist_ok=True)
    save_name = f"{plane or 'any'}_{(modality.replace(' ', '_') or 'any').lower().replace(' ', '_')}_multiple_series.png"
    fig.savefig(os.path.join("Figures", save_name), dpi=300)
    fig.show()
```

```python
plot_multiple_series(modality='CTA')
```

```python
plot_multiple_series(modality='MRA')
```

```python
plot_multiple_series(modality='MRI T2')
```

```python
plot_multiple_series(modality='MRI T1post')
```

### <div style="text-align:center; border-radius:10px; color:black; margin:0; font-size:100%; font-family:Carbon Plus; font-weight:bold; background-color:white; overflow:hidden"><b>Advanced DICOM Metadata</b></div>


DICOM files contain a wealth of metadata beyond the image pixels themselves. Attributes such as **ImagePositionPatient**, **ImageOrientationPatient**, **SliceThickness**, **PixelSpaing**, **WindowWidth**, and **Rescale** provide essential context for reconstructing volumes, aligning modalities, and ensuring consistent preprocessing. 


![DICOM_metadata](https://raw.githubusercontent.com/ammomahdavikia/asset-holding/main/DICOM_metadata.png)

### <div style="text-align:center; border-radius:10px; color:black; margin:0; font-size:100%; font-family:Carbon Plus; background-color:white; overflow:hidden"><b>ImagePositionPatient</b></div>

3D coordinates *(x, y, z)* of the **top-left corner** (first pixel) of the image in the patient’s coordinate system.

```python
ipps = ['IPP_X', 'IPP_Y', 'IPP_Z']

fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
axs = axs.flatten()

for idx, ipp in enumerate(ipps):
    sns.histplot(
        train_instances,
        x=ipp,
        hue='Modality',
        kde=True,
        palette={'MR': COLORS[3], 'CT': COLORS[4]},
        bins=10,
        ax=axs[idx]
    )
    axs[idx].set_title(f'ImagePositionPatient{ipp[-1]}')
    axs[idx].set_xlabel('')

fig.tight_layout()
fig.show()
fig.savefig('Figures/ImagePositionPatient_distribution.png', dpi=300)
```

### <div style="text-align:center; border-radius:10px; color:black; margin:0; font-size:100%; font-family:Carbon Plus; background-color:white; overflow:hidden"><b>Rescale</b></div>

- **RescaleType** - Defines the physical unit of pixel values (e.g., HU for Hounsfield Units).  
- **RescaleIntercept** - Offset added after applying the slope (shifts intensity values).  
- **RescaleSlope** - Scaling factor applied to stored pixel values.

We will use it further in CTA modality image processing later in the [Preprocessing Guide]() notebook.

```python
rscles = ['RescaleType', 'RescaleIntercept', 'RescaleSlope']

fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
axs = axs.flatten()

for idx, rscle in enumerate(rscles):
    sns.histplot(
        train_instances,
        x=rscle,
        hue='Modality',
        palette={'MR': COLORS[3], 'CT': COLORS[4]},
        ax=axs[idx]
    )
    axs[idx].set_title(rscle)
    axs[idx].set_xlabel('')

fig.tight_layout()
fig.show()
fig.savefig('Figures/Rescale_distribution.png', dpi=300)
```

### <div style="text-align:center; border-radius:10px; color:black; margin:0; font-size:100%; font-family:Carbon Plus; background-color:white; overflow:hidden"><b>SliceThickness + SpacingBetweenSlices</b></div>

- **SliceThickness** – Physical thickness of each slice in millimeters.  
- **SpacingBetweenSlices** – Distance in millimeters between the centers of adjacent slices.

```python
slices = train_instances.groupby(['SeriesInstanceUID', 'Modality'])[['SliceThickness', 'SpacingBetweenSlices']].mean().reset_index()
g = sns.jointplot(
    data=slices, 
    x="SliceThickness", 
    y="SpacingBetweenSlices",
    hue='Modality',
    palette={'MR': COLORS[3], 'CT': COLORS[4]},
    kind='kde',
    height=6
)

g.figure.savefig('Figures/slice_dimensions_distribution.png', dpi=300)
plt.show()
```

### <div style="text-align:center; border-radius:10px; color:black; margin:0; font-size:100%; font-family:Carbon Plus; background-color:white; overflow:hidden"><b>PixelSpacing</b></div>

Physical distance in millimeters between the centers of adjacent pixels in the row and column directions *(row spacing, column spacing)*.

```python
pixels = train_instances.groupby(['SeriesInstanceUID', 'Modality'])[['PixelSpacing_X', 'PixelSpacing_Y']].mean().reset_index()

g = sns.jointplot(
    data=pixels, 
    x="PixelSpacing_X", 
    y="PixelSpacing_Y",
    hue='Modality',
    kind='kde',
    palette={'MR': COLORS[3], 'CT': COLORS[4]},
    height=6
)

g.figure.savefig('Figures/pixel_spacing_distribution.png', dpi=300)
plt.show()
```

### <div style="text-align:center; border-radius:10px; color:black; margin:0; font-size:100%; font-family:Carbon Plus; background-color:white; overflow:hidden"><b>WindowCenter + WindowWidth</b></div>

- **WindowCenter (WC)** – The midpoint value of the displayed intensity range.  
- **WindowWidth (WW)** – The width of the intensity range displayed.  

**Use:** Controls image contrast by mapping the original pixel intensities to display grayscale levels. Pixels below *(WC - WW/2)* appear black, above *(WC + WW/2)* appear white, and values in between are scaled linearly for visualization.  

![DICOM_window](https://raw.githubusercontent.com/ammomahdavikia/asset-holding/main/DICOM_window.png)

```python
windows = train_instances.groupby(['SeriesInstanceUID', 'Modality'])[['WindowWidth1', 'WindowCenter1']].mean().reset_index()

g = sns.jointplot(
    data=windows, 
    x="WindowWidth1", 
    y="WindowCenter1",
    hue='Modality',
    palette={'MR': COLORS[3], 'CT': COLORS[4]},
    height=6
)

g.figure.savefig('Figures/window_distribution.png', dpi=300)
plt.show()
```

```python
windows = ['WindowWidth1', 'WindowCenter1']
fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(8, 6))

for idx, window in enumerate(windows):
    sns.boxplot(
        train_instances,
        x=window,
        y='Modality',
        palette={'MR': COLORS[3], 'CT': COLORS[4]},
        ax=axs[idx],
        orient='h'
    )
    axs[idx].set_title(window)
    axs[idx].set_xlim(0, 3000)
    axs[idx].set_ylabel('')
    axs[idx].set_xlabel('')

fig.tight_layout()
fig.show()
fig.savefig('Figures/window_boxplot.png', dpi=300)
```

### <div style="text-align:center; border-radius:10px; color:black; margin:0; font-size:100%; font-family:Carbon Plus; background-color:white; overflow:hidden"><b>ImageOrientationPatient</b></div>

Two 3D direction cosines: one for the image rows, one for the image columns.

```python
iops = ['IOP_RowX', 'IOP_RowY', 'IOP_RowZ', 'IOP_ColX', 'IOP_ColY', 'IOP_ColZ']

fig, axs = plt.subplots(2, 3, figsize=(12, 8))
axs = axs.flatten()

for idx, iop in enumerate(iops):
    axs[idx].hist(train_instances[iop], color=COLORS[-1], bins=10)
    axs[idx].set_title(f'ImageOrientationPatient{iop[-4:]}')
    axs[idx].set_xlim(-1, 1)
    axs[idx].set_xlabel('')

fig.tight_layout()
fig.show()
fig.savefig('Figures/ImageOrientationPatient_distribution.png', dpi=300)
```

Check out the following section, if you are interested in further application of these attributes.

<details style="background-color: #e6f0ff; padding: 10px; border-radius: 5px; border: 1px solid #3399ff;">
  <summary style="font-weight: bold; color: #003366;">DICOM Coordinate Mapping (Pixel → Patient Space)</summary>

  <p>Given the following <strong>DICOM attributes</strong>, we can map any image pixel to its corresponding <strong>3D location in patient space</strong>:</p>

  <ul>
    <li><strong>IPP</strong> = $(X_0, Y_0, Z_0)$ → <em>ImagePositionPatient</em> (origin of the image in patient coordinates)</li>
    <li><strong>IOP</strong> = $(R_x, R_y, R_z, C_x, C_y, C_z)$ → <em>ImageOrientationPatient</em>
      <ul>
        <li>$\vec{R}$ = direction cosine vector of image rows</li>
        <li>$\vec{C}$ = direction cosine vector of image columns</li>
      </ul>
    </li>
    <li><strong>PixelSpacing</strong> = $(\Delta r, \Delta c)$ → physical spacing between rows and columns</li>
    <li><strong>i, j</strong> = pixel indices <em>(row, column)</em> (0-based)</li>
  </ul>

  <p style="text-align:center;">
    <img src="https://raw.githubusercontent.com/ammomahdavikia/asset-holding/main/DICOM_coordinates.png" 
         alt="DICOM coordinates" style="max-width: 100%; height: auto;" />
  </p>

  <hr />

  <h3>Coordinate Transformation Formula</h3>

  <p style="text-align:center;">
    $$
    \text{PatientCoord} =
    \text{IPP} + i \cdot \Delta r \cdot \vec{R} + j \cdot \Delta c \cdot \vec{C}
    $$
  </p>

  <p>Where:</p>
  <ul>
    <li><strong>i</strong> → moves along the image rows (downward direction)</li>
    <li><strong>j</strong> → moves along the image columns (rightward direction)</li>
    <li>The output is a 3D coordinate <strong>(x, y, z)</strong> in the patient’s anatomical reference frame</li>
  </ul>

  <p>This mapping is fundamental for <strong>3D reconstruction, multi-slice alignment, and cross-modality registration</strong>, ensuring that each voxel corresponds to the correct anatomical location in the patient.</p>

</details>

# <div style="text-align:center; border-radius:10px; color:black; margin:0; font-size:100%; font-family:Carbon Plus; background-color:white; overflow:visible; line-height:1.5; padding:10px 0;"><b>✂️ Body Planes</b></div>

Image slices can be divided in three planes:
+ **Axial (Transverse)**: upper (superior) and lower (inferior) parts
+ **Coronal (Frontal)**: front (anterior) and back (posterior) parts
+ **Sagittal**: left and right parts

![body_planes](https://raw.githubusercontent.com/ammomahdavikia/asset-holding/main/body_planes.png)

<details style="background-color: #e6f0ff; padding: 10px; border-radius: 5px; border: 1px solid #3399ff;">
  <summary style="font-weight: bold; color: #003366;">How to find it?</summary>

  <p>First, we need to know <strong>ImageOrientationPatient</strong> attribute in further detail.</p>

  <ul>
    <li>IOP consists of <strong>6 values</strong> representing two orthogonal direction cosines in patient 3D space:</li>
  </ul>

  <p style="text-align:center;">
    $$
    \text{IOP} = (R_x, R_y, R_z, C_x, C_y, C_z)
    $$
  </p>

  <ul>
    <li>$\vec{R} = (R_x, R_y, R_z)$ is the direction cosine vector along the image <strong>rows</strong></li>
    <li>$\vec{C} = (C_x, C_y, C_z)$ is the direction cosine vector along the image <strong>columns</strong></li>
  </ul>

  <p>These vectors define the orientation of the image plane in the patient coordinate system (usually LPS or RAS).</p>

  <hr />

  <h3>Application to Determine the Imaging Plane</h3>

  <p>The <strong>normal vector</strong> $\vec{N}$ to the image plane is the cross product of $\vec{R}$ and $\vec{C}$:</p>

  <p style="text-align:center;">
    $$
    \vec{N} = \vec{R} \times \vec{C}
    $$
  </p>

  <p>By analyzing $\vec{N}$, you can classify the plane type:</p>

  <table style="border-collapse: collapse; width: 100%; margin: 10px 0;">
    <thead>
      <tr style="background-color: #cce0ff;">
        <th style="border: 1px solid #3399ff; padding: 5px;">Plane Type</th>
        <th style="border: 1px solid #3399ff; padding: 5px;">Normal Vector Direction Approximation</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td style="border: 1px solid #3399ff; padding: 5px;"><strong>Axial</strong></td>
        <td style="border: 1px solid #3399ff; padding: 5px;">$\vec{N} \approx (0,0,\pm1)$ → head-foot axis</td>
      </tr>
      <tr>
        <td style="border: 1px solid #3399ff; padding: 5px;"><strong>Coronal</strong></td>
        <td style="border: 1px solid #3399ff; padding: 5px;">$\vec{N} \approx (0,\pm1,0)$ → front-back axis</td>
      </tr>
      <tr>
        <td style="border: 1px solid #3399ff; padding: 5px;"><strong>Sagittal</strong></td>
        <td style="border: 1px solid #3399ff; padding: 5px;">$\vec{N} \approx (\pm1,0,0)$ → left-right axis</td>
      </tr>
    </tbody>
  </table>

  <p>This helps identify the modality plane of the image slice for processing, visualization, or registration.</p>

</details>

```python
plane_counts = train_df['Plane'].value_counts().reset_index()
plane_counts.columns = ['Plane', 'Count']
plane_counts = plane_counts.sort_values('Plane')

bar_colors = plane_counts['Plane'].map(plane_palette)

fig, ax = plt.subplots(figsize=(6, 4))
sns.barplot(
    data=plane_counts,
    x='Plane',
    y='Count',
    palette=bar_colors,
    ax=ax
)

ax.set_xlabel('')
ax.set_title('Plane Distribution')
fig.tight_layout()
fig.savefig('Figures/plane_distribution.png', dpi=300)
fig.show()
```

```python
artery_per_plane_count = (
    train_df
    .groupby('Plane')[list(CONFIG['ARTERIES'].values())]
    .sum()
    .T
)

artery_per_plane_count.reset_index(inplace=True)
artery_per_plane_count.rename(columns={'index': 'Artery'}, inplace=True)

fig, ax = plt.subplots(figsize=(10, 6))

bottom = np.zeros(len(artery_per_plane_count))

planes = artery_per_plane_count.columns[1:]
colors = [plane_palette[m] for m in planes]

for plane, color in zip(planes, colors):
    counts = artery_per_plane_count[plane]
    ax.barh(
        artery_per_plane_count['Artery'],
        counts,
        left=bottom,
        label=plane,
        color=color
    )
    bottom += counts

ax.set_title('Artery Aneurysm per Plane Count')
ax.set_xlabel('Count')
ax.set_ylabel('Artery')
ax.legend(title='Plane')
ax.invert_yaxis()

fig.tight_layout()
fig.show()
fig.savefig('Figures/artery_per_plane_distribution.png', dpi=300)
```

```python
fig, ax = plt.subplots(figsize=(8, 5))

sns.histplot(
    data=train_df,
    x='NumberOfFrames',
    hue="Plane",
    kde=True,
    palette=plane_palette,
    ax=ax
)

ax.set_title('Frame distribution per plane')
fig.tight_layout()
fig.show()
fig.savefig('Figures/frame_per_plane_distribution.png', dpi=300)
```

```python
g = sns.jointplot(
    data=train_df, 
    x="Columns", 
    y="Rows",
    palette=plane_palette,
    hue="Plane",
    height=6
)

g.figure.savefig('Figures/image_dimensions_per_plane_distribution.png', dpi=300)
plt.show()
```

```python
def plot_single_series(modality=None, plane=None):
    """
    Plots all the slices of a randomly selected DICOM series 
    matching the specified modality and/or plane (if given).

    Parameters:
        modality (str or None): Filter by modality. If None, all modalities are included.
        plane (str or None): Filter by plane. If None, all sections are included.
    """

    condition = pd.Series(True, index=merged_df.index)
    if modality is not None:
        condition &= (merged_df['Modality'] == modality)
    if plane is not None:
        condition &= (merged_df['Plane'] == plane)

    series_list = merged_df.loc[condition, 'SeriesInstanceUID'].unique()

    if len(series_list) == 0:
        filters = []
        if modality: filters.append(f"modality='{modality}'")
        if plane: filters.append(f"plane='{plane}'")
        raise ValueError(f"No series found for {' and '.join(filters) or 'any condition'}.")

    selected_series = np.random.choice(series_list)
    series_path = os.path.join(CONFIG['PATH'], 'series', selected_series)

    try:
        series_array = load_dicom(series_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load DICOM from: {series_path}\n{e}")

    num_slices = series_array.shape[0]
    
    cols = math.ceil(math.sqrt(num_slices))
    rows = math.ceil(num_slices / cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols, rows))
    axes = axes.ravel()

    for i in range(rows*cols):
        ax = axes[i]
        if i < num_slices:
            ax.imshow(series_array[i, :, :], cmap='gray')
            ax.set_title(i)
        ax.axis('off')


    mod_text = modality.upper() if modality else 'Any Modality'
    sec_text = plane if plane else 'Any Plane'
    fig.suptitle(f"{sec_text} {mod_text}\n{selected_series}", fontsize=14)

    fig.tight_layout()
    save_name = f"{plane or 'any'}_{(modality.replace(' ', '_') or 'any').lower().replace(' ', '_')}_single_series.png"
    fig.savefig(os.path.join("Figures", save_name), dpi=300)
    fig.show()
```

```python
plot_single_series(modality='MRI T1post', plane='axial')
```

```python
plot_single_series(modality='MRI T1post', plane='sagittal')
```

```python
plot_single_series(modality='MRI T1post', plane='coronal')
```

```python
def plot_modality_per_plane():
    warnings.simplefilter("ignore", category=DeprecationWarning)
    sampled = (
        merged_df.groupby(['Modality', 'Plane'], group_keys=False)
                 .apply(lambda g: g.sample(1, random_state=42))
                 .reset_index(drop=True)
    ).sort_values(['Plane', 'Modality'])

    planes = sorted(sampled['Plane'].unique())
    modalities = sorted(sampled['Modality'].unique())

    fig, axs = plt.subplots(nrows=3, ncols=4, figsize=(16, 12))

    for idx, (_, row) in enumerate(sampled.iterrows()):
        row_idx = planes.index(row['Plane'])
        col_idx = modalities.index(row['Modality'])

        ax = axs[row_idx, col_idx]
        series = row['SeriesInstanceUID']
        path = os.path.join(CONFIG['PATH'], 'series', series)

        try:
            array = load_dicom(path)
            slice_idx = array.shape[0] // 2
            ax.imshow(array[slice_idx], cmap='gray')
        except Exception as e:
            ax.set_title(f"Error: {series}", fontsize=8)
        finally:
            ax.axis('off')

        if row_idx == 0:
            ax.set_title(row['Modality'], fontsize=14)
        if col_idx == 0:
            ax.set_ylabel(row['Plane'], fontsize=14)

    fig.suptitle("One Random Series per Modality × Plane", fontsize=18)
    fig.tight_layout()
    fig.subplots_adjust(top=0.92)
    fig.savefig("Figures/all_modality_per_plane.png", dpi=300)
    fig.show()
```

```python
plot_modality_per_plane()
```

```python
fig, axs = plt.subplot_mosaic(
    [["modality", "plane"],
     ["modality_vs_plane", "modality_vs_plane"]],
    figsize=(12, 9)
)

# First plot: Modality distribution per aneurysm
sns.histplot(
    data=train_df,
    x='Modality', 
    hue='Aneurysm Present', 
    multiple="dodge",
    shrink=0.8,
    palette=aneurysm_palette,
    ax=axs['modality']
)
axs['modality'].set_title('Modality Distribution')
axs['modality'].set_ylim(0, 2500)

# Second plot: Plane distribution per aneurysm
sns.histplot(
    data=train_df,
    x='Plane', 
    hue='Aneurysm Present', 
    multiple="dodge",
    shrink=0.8,
    palette=aneurysm_palette,
    ax=axs['plane']
)
axs['plane'].set_title('Plane Distribution')

# Third plot: Plane vs. Modality
sns.histplot(
    data=train_df,
    x='Plane', 
    hue='Modality', 
    multiple="dodge",
    shrink=0.8,
    palette=modality_palette,
    ax=axs['modality_vs_plane']
)
axs['modality_vs_plane'].set_title('Plane vs. Modality')
axs['modality_vs_plane'].set_ylim(0, 2500)

fig.tight_layout()
fig.show()
fig.savefig('Figures/combined_distribution.png', dpi=300)
```

# <div style="text-align:center; border-radius:10px; color:black; margin:0; font-size:100%; font-family:Carbon Plus; background-color:white; overflow:visible; line-height:1.5; padding:10px 0;"><b>📍 Localization</b></div>

```python
all_loc_pivot = pd.pivot_table(
    merged_df,
    index='Aneurysm Present',
    columns='Modality',
    values='PatientAge',
    aggfunc='count',
    margins=True,
    margins_name='Total'
).iloc[:-1]

unique_loc_pivot = pd.pivot_table(
    merged_df.drop_duplicates('SeriesInstanceUID'),
    index='Aneurysm Present',
    columns='Modality',
    values='PatientAge',
    aggfunc='count',
    margins=True,
    margins_name='Total'
).iloc[:-1]

loc_pivot = pd.concat([all_loc_pivot, unique_loc_pivot])
loc_pivot.index = ['All Aneurysms', 'Unique series']

fig, ax = plt.subplots(figsize=(6, 3))
sns.heatmap(
    loc_pivot, 
    annot=True,
    fmt='g',
    cmap='Reds_r',
    linewidths=0.5, 
    linecolor='gray',
    ax=ax
    
)

ax.set_title('Coordinate Count by Modality')
ax.set_ylabel('')
ax.set_xlabel('Modality')
fig.tight_layout()
fig.show()
fig.savefig('Figures/series_and_coordinate_count_per_modality.png', dpi=300)
```

```python
fig = px.scatter(
    merged_df,
    x='x_norm',
    y='y_norm',
    color='location',
    category_orders={'location': CONFIG['ARTERIES'].values()},
    title='Normalized Coordinates by Location',
    labels={'x_norm': 'Normalized X', 'y_norm': 'Normalized Y'},
)

fig.update_traces(marker=dict(size=5, line=dict(width=0.5, color='DarkSlateGrey')))
fig.update_layout(
    scattermode="group",
    template="plotly_white",
    legend_title_text='Location',
    width=1000,
    height=800
)

fig.show()
fig.write_html("Figures/aneurysm_normalized_coordinates.html")
```

```python
def plot_multiple_localization(modality=None, plane=None, n_samples=5):
    """
    Plots annotated slices with aneurysm bounding boxes (32x32, clipped at borders)
    and an additional zoomed-in subplot showing the cropped region (with a red border).
    """
    condition = pd.Series(True, index=merged_df.index)
    if modality:
        condition &= (merged_df['Modality'] == modality)
    if plane:
        condition &= (merged_df['Plane'] == plane)

    all_instances = merged_df.loc[condition, 'SOPInstanceUID'].unique().tolist()
    if len(all_instances) == 0:
        filters = []
        if modality: filters.append(f"modality='{modality}'")
        if plane: filters.append(f"plane='{plane}'")
        raise ValueError(f"No SOPInstanceUIDs found for {' and '.join(filters) or 'any condition'}.")

    if len(all_instances) < n_samples:
        print(f"Warning: Only {len(all_instances)} instances available. Plotting all.")
        n_samples = len(all_instances)

    picked_instances = np.random.choice(all_instances, size=n_samples, replace=False)

    fig, axs = plt.subplots(nrows=2, ncols=n_samples, figsize=(5 * n_samples, 10), constrained_layout=True)
    if n_samples == 1:
        axs = np.array(axs).reshape(2, 1)

    for idx, instance in enumerate(picked_instances):
        try:
            row = merged_df.loc[merged_df.SOPInstanceUID == instance].iloc[0]
            series = row['SeriesInstanceUID']
            instance_number = train_instances.loc[
                train_instances.SOPInstanceUID == instance, 'InstanceNumber'
            ].iloc[0]
            series_path = os.path.join(CONFIG['PATH'], 'series', series)
            array = load_dicom(series_path)

            x, y = row['x'], row['y']
            rows, cols = row['Rows'], row['Columns']
            new_x = x * (512 / cols)
            new_y = y * (512 / rows)

            slice_img = array[instance_number]

            half = 16
            xmin = int(max(np.floor(new_x - half), 0))
            ymin = int(max(np.floor(new_y - half), 0))
            xmax = int(min(np.floor(new_x + half), slice_img.shape[1]))
            ymax = int(min(np.floor(new_y + half), slice_img.shape[0]))

            w = xmax - xmin
            h = ymax - ymin

            # --- Full slice with bbox ---
            axs[0, idx].imshow(slice_img, cmap="gray")
            axs[0, idx].add_patch(
                Rectangle((xmin - 0.5, ymin - 0.5), w, h, linewidth=2, edgecolor="red", facecolor="none")
            )
            axs[0, idx].axis("off")
            axs[0, idx].set_title(series, fontsize=8)

            # --- Cropped box (zoomed view) with perfectly fitting red border ---
            crop = slice_img[ymin:ymax, xmin:xmax]
            ch, cw = crop.shape[:2]
            axc = axs[1, idx]
            axc.imshow(crop, cmap="gray")
            axc.add_patch(
                Rectangle((-0.5, -0.5), cw, ch, linewidth=10, edgecolor="red", facecolor="none")
            )
            axc.set_xlim(-0.5, cw - 0.5)
            axc.set_ylim(ch - 0.5, -0.5)
            axc.set_aspect('equal')
            axc.axis("off")
            axc.set_title("Zoomed Crop", fontsize=8)

        except Exception as e:
            axs[0, idx].set_title(f"Error: {instance}", fontsize=6)
            axs[0, idx].axis("off")
            axs[1, idx].axis("off")
            print(f"[Warning] Failed to load or annotate {instance}: {e}")

    mod_text = modality.upper() if modality else "Any Modality"
    sec_text = plane if plane else "Any Plane"
    fig.suptitle(f"{sec_text} {mod_text} — Random Slices with BBoxes", fontsize=18)

    os.makedirs("Figures", exist_ok=True)
    save_name = f"{plane or 'any'}_{modality or 'any'}_bbox_zoom.png"
    save_name = save_name.lower().replace(" ", "_")
    fig.savefig(os.path.join("Figures", save_name), dpi=300)
    fig.show()
```

```python
plot_multiple_localization(modality='CTA')
```

```python
plot_multiple_localization(modality='MRA')
```

```python
plot_multiple_localization(modality='MRI T2', plane='sagittal')
```

```python
plot_multiple_localization(modality='MRI T1post', plane='coronal')
```

# <div style="text-align:center; border-radius:10px; color:black; margin:0; font-size:100%; font-family:Carbon Plus; background-color:white; overflow:hidden"><b>🧠 Intracranial Arteries Anatomy</b></div>

The brain’s blood supply is divided into two main systems: the **anterior (front) circulation**, which arises from the **internal carotid arteries**, and the **posterior (back) circulation**, which originates from the **vertebral arteries**.

![intracranial_vasculature](https://raw.githubusercontent.com/ammomahdavikia/asset-holding/main/intracranial_vasculature.png)

As these arteries enter the cranium, they branch into multiple vessels, each supplying different regions of the brain. A rupture in any of these arteries can lead to a range of symptoms and functional impairments, depending on the area affected.


<details style="background-color: #e6f0ff; padding: 10px; border-radius: 5px; border: 1px solid #3399ff;">
  <summary style="font-weight: bold; color: #003366;">Click if you want more Anatomy!</summary>

  <p>Let's explore branches further.</p>
  <img src="https://raw.githubusercontent.com/ammomahdavikia/asset-holding/main/intracranial_vasculature_details.png" alt="Intracranial vasculature details" style="max-width: 100%; height: auto;" />
  
  <h3>Posterior Circulation</h3>
  <p>At the lower border of the brainstem, the two vertebral arteries merge to form the <strong>basilar artery</strong> <code>[Tag: 2]</code>. Then, it divides into multiple arteries along the way to the top in order to provide <strong>posterior (back) circulation</strong> of the brain <code>[Tag: 1]</code>.</p>

  <h5><strong>Vertebral Artery Branches:</strong></h5>
  <ol>
    <li><strong>Meningeal Arteries</strong> – supply the dura mater.</li>    
    <li><strong>Posterior Spinal Artery</strong> – supplies the posterior part of the spinal cord.</li>
    <li><strong>Anterior Spinal Artery</strong> – supplies the anterior two-thirds of the spinal cord.</li>
    <li><strong>Posterior Inferior Cerebellar Artery (PICA)</strong> – supplies the posterior inferior cerebellum and medulla.</li>
    <li><strong>Medullary Arteries</strong> – supply various parts of the medulla oblongata.</li>
  </ol>

  <h5><strong>Basilar Artery Branches:</strong></h5>
  <ol>
    <li><strong>Pontine Arteries</strong> – supply the pons.</li>
    <li><strong>Labyrinthine Artery</strong> – supplies the inner ear (via the internal auditory meatus).</li>
    <li><strong>Anterior Inferior Cerebellar Artery (AICA)</strong> – supplies the anterior inferior cerebellum and part of the pons.</li>
    <li><strong>Superior Cerebellar Artery</strong> – supplies the superior part of the cerebellum and upper pons.</li>
    <li><strong>Posterior Cerebral Artery (PCA)</strong> – supplies the occipital lobe and inferior temporal lobes (vision and visual processing).</li>
  </ol>

  <hr />
  
  <h3>Anterior Circulation</h3>
  <p>The internal carotid artery (ICA) is named according to the anatomic landmarks it passes, with the <strong>clinoid process</strong>—a bony projection of the sphenoid bone—being a key reference point. The segment <strong>below</strong> this structure is called the <strong>infraclinoid</strong> segment <code>[Tag: 5 and 6]</code>, while the segment <strong>above</strong> it is known as the <strong>supraclinoid</strong> <code>[Tag: 7 and 8]</code> segment.</p>

  <h5><strong>Internal Carotid Artery Branches:</strong></h5>
  <ol>
    <li><strong>Ophthalmic Artery</strong> – supplies the eye and orbit.</li>
    <li><strong>Choroidal Artery</strong> – supplies deep brain structures like the choroid plexus, optic tract, and internal capsule.</li>
    <li><strong>Posterior Communicating Artery</strong> – connects internal carotid to posterior circulation; part of the Circle of Willis.</li>
    <li><strong>Anterior Cerebral Artery (ACA)</strong> <code>[Tag: 11 and 12]</code> – supplies the medial portions of the frontal and parietal lobes (leg and foot motor/sensory areas).</li>
    <li><strong>Middle Cerebral Artery (MCA)</strong> <code>[Tag: 9 and 10]</code> – supplies the lateral surfaces of the frontal, parietal, and temporal lobes (face, arm, speech areas).</li>
  </ol>

  <p>The brain’s cortex is primarily supplied by three major cerebral arteries:</p>
  <img src="https://raw.githubusercontent.com/ammomahdavikia/asset-holding/main/cortex_supply.png" alt="Cortex arterial supply" style="max-width: 100%; height: auto;" />

  <p>Each artery serves distinct functional areas, so damage to one can cause specific neurological deficits.</p>

  <details style="background-color: #e6ffe6; padding: 10px; border: 1px solid #00cc44; border-radius: 5px;">
    <summary style="color: #006600; font-weight: bold;">Quiz</summary>

    <p>A 45-year-old man is brought to your clinic. He is alert and understands spoken language but struggles to express himself. His speech is <strong>nonfluent, effortful, slow, and halting</strong>, as he searches for words. Neurological imaging reveals an infarct involving <strong>Broca’s area on the lateral surface of the dominant frontal lobe</strong>.</p>

    <img src="https://raw.githubusercontent.com/ammomahdavikia/asset-holding/main/broca_lesion.png" alt="CT showing Broca area infarct" width="400" />

    <p>Which <strong>cerebral artery</strong> is most likely occluded?</p>

    <details>
      <summary style="color: #009900;">Answer</summary>
      <p><strong>Middle Cerebral Artery</strong>: supplies the lateral (outer) surfaces of the frontal, parietal, and temporal lobes.</p>
    </details>
  </details>

  <hr />

  <h5><strong>Circle of Willis:</strong></h5>
  <p>At the base of the brain, fusion between the two internal carotid arteries and the two vertebral arteries leads to a ring which is called <strong>Circle of Willis</strong>. It is named after <strong>Thomas Willis</strong>, a 17th-century English physician who was one of the first to describe this arterial ring at the base of the brain in detail.</p>

  <img src="https://raw.githubusercontent.com/ammomahdavikia/asset-holding/main/willis_circle.png" alt="Circle of Willis" style="max-width: 100%; height: auto;" />

  <p>The <strong>anterior communicating</strong> <code>[Tag: 13]</code>, <strong>anterior cerebral</strong>, <strong>internal carotid</strong>, <strong>posterior communicating</strong> <code>[Tag: 3 and 4]</code>, <strong>posterior cerebral</strong>, and <strong>basilar arteries</strong>, all contribute to the circle.</p>

  <p>Its main importance is <strong>collateral circulation</strong>. It allows blood to be redistributed if one major artery is blocked or narrowed, helping maintain consistent blood flow to the brain and reducing the risk of stroke.</p>

  <hr />

  <h5><strong>Osmosis videos (Highly recommended):</strong></h5>

  <ul>
    <li><a href="https://www.osmosis.org/learn/Anatomy_of_the_blood_supply_to_the_brain?from=/oh/foundational-sciences/anatomy/brain/gross-anatomy" target="_blank" rel="noopener noreferrer">Anatomy of the blood supply to the brain</a></li>
    <li><a href="https://www.youtube.com/embed/A5MEe0lb0YA" target="_blank" rel="noopener noreferrer">Aneurysm Physiology</a></li>
    <li><a href="https://www.osmosis.org/learn/Anatomy_clinical_correlates:_Anterior_blood_supply_to_the_brain?from=/oh/foundational-sciences/anatomy/brain/anatomy-clinical-correlates" target="_blank" rel="noopener noreferrer">Anatomy clinical correlates: Anterior blood supply to the brain</a></li>
    <li><a href="https://www.osmosis.org/learn/Anatomy_clinical_correlates:_Posterior_blood_supply_to_the_brain?from=/oh/foundational-sciences/anatomy/brain/anatomy-clinical-correlates" target="_blank" rel="noopener noreferrer">Anatomy clinical correlates: Posterior blood supply to the brain</a></li>
  </ul>

</details>

### Where Are Aneurysms Located?

Each aneurysm annotation corresponds to a **single slice**. In this section, we explore the distribution of these selected slices to understand where aneurysms tend to occur along the patient’s anatomy.  

We focus on the **ImagePositionPatient (IPP)** attribute to analyze the spatial distribution of annotated slices along the head-foot axis, providing insights for slice selection and preprocessing strategies.

```python
axial_single = merged_df[
    (~merged_df.IsMultiFrame) & 
    (merged_df.Plane == 'axial')
]

axial_merged = pd.merge(
    axial_single, 
    train_instances, 
    on=['SeriesInstanceUID']
)

first_slices = axial_merged.loc[axial_merged.InstanceNumber == 1]
last_slices = axial_merged.loc[axial_merged.InstanceNumber == axial_merged.NumberOfFrames]

instance_loc = pd.merge(
    axial_single, 
    train_instances, 
    on=['SeriesInstanceUID', 'SOPInstanceUID', 'Rows', 'Columns', 'SOPClassUID']
)

first_z = (
    first_slices[['SeriesInstanceUID', 'IPP_Z']]
    .rename(columns={'IPP_Z': 'FirstZ'})
)

last_z = (
    last_slices[['SeriesInstanceUID', 'IPP_Z']]
    .rename(columns={'IPP_Z': 'LastZ'})
)

instance_loc = instance_loc.merge(first_z, on='SeriesInstanceUID', how='left')
instance_loc = instance_loc.merge(last_z, on='SeriesInstanceUID', how='left')

instance_loc['ZDiffFirst'] = instance_loc['IPP_Z'] - instance_loc['FirstZ']
instance_loc['ZDiffLast'] = instance_loc['LastZ'] - instance_loc['IPP_Z']
instance_loc['ZDiffNorm'] = (instance_loc['IPP_Z'] - instance_loc['FirstZ']) / (
    instance_loc['LastZ'] - instance_loc['FirstZ']
)
```

```python
fig, ax = plt.subplots(figsize=(8, 5))

sns.histplot(
    data=instance_loc,
    x='IPP_Z',
    hue='Modality_x',
    multiple='stack',
    palette=modality_palette,
    bins=50,
    ax=ax
)

ax.set_title('ImagePositionPatient distribution')

fig.tight_layout()
fig.show()
fig.savefig('Figures/ImagePositionPatient_per_modality_distribution.png', dpi=300)
```

```python
def plot_instance_loc(feature='ZDiffNorm'):
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
    
    sns.histplot(
        data=instance_loc,
        x=feature,
        kde=True,
        bins=50,
        ax=axs[0]
    )
    axs[0].set_title('Normalized IPP_Z of Annotated Slice')
    
    sns.histplot(
        data=instance_loc,
        x=feature,
        hue='Modality_x',
        kde=True,
        palette=modality_palette,
        bins=50,
        ax=axs[1]
    )
    axs[1].set_title('Normalized IPP_Z x Modality')
    axs[1].set_ylabel('')
    
    sns.histplot(
        data=instance_loc,
        x=feature,
        hue='location',
        kde=True,
        bins=50,
        ax=axs[2]
    )
    axs[2].set_title('Normalized IPP_Z x Location')
    axs[2].set_ylabel('')

    fig.tight_layout()
    fig.savefig(f'Figures/ImagePositionPatient_with{feature}_distribution.png', dpi=300)
    fig.show()
```

### <div style="text-align:center; border-radius:10px; color:black; margin:0; font-size:100%; font-family:Carbon Plus; background-color:white; overflow:hidden"><b>ZDiffNorm</b></div>

+ **ZDiffNorm**: Normalized position of selected slice along the Z-axis.

This provides a standardized way to compare slice locations across patients of different lengths and scan ranges.

```python
plot_instance_loc(feature='ZDiffNorm')
```

### <div style="text-align:center; border-radius:10px; color:black; margin:0; font-size:100%; font-family:Carbon Plus; background-color:white; overflow:hidden"><b>ZDiffFirst</b></div>

+ **ZDiffFirst**: Distance of the selected slice from the first slice along the Z-axis.

Analysis shows that annotated slices tend to follow a **roughly normal distribution**, with most annotations falling within the ranges **[0–100]** and **[200–300]**.

```python
plot_instance_loc(feature='ZDiffFirst')
```

### <div style="text-align:center; border-radius:10px; color:black; margin:0; font-size:100%; font-family:Carbon Plus; background-color:white; overflow:hidden"><b>ZDiffLast</b></div>

+ **ZDiffLast**: Distance of the selected slice from the last slice along the Z-axis.

```python
plot_instance_loc(feature='ZDiffLast')
```

# <div style="text-align:center; border-radius:10px; color:black; margin:0; font-size:100%; font-family:Carbon Plus; background-color:white; overflow:visible; line-height:1.5; padding:10px 0;"><b>🧩 Segmentation</b></div>

`segmentation` folder contains NifTI vessel segmentations for a subset (178/4348) of the DICOM series in `series`.

| Label Value | Label                                    |
|------------:|-----------------------------------------|
| 1           | Other Posterior Circulation              |
| 2           | Basilar Tip                              |
| 3           | Right Posterior Communicating Artery     |
| 4           | Left Posterior Communicating Artery      |
| 5           | Right Infraclinoid Internal Carotid Artery |
| 6           | Left Infraclinoid Internal Carotid Artery  |
| 7           | Right Supraclinoid Internal Carotid Artery |
| 8           | Left Supraclinoid Internal Carotid Artery  |
| 9           | Right Middle Cerebral Artery            |
| 10          | Left Middle Cerebral Artery             |
| 11          | Right Anterior Cerebral Artery          |
| 12          | Left Anterior Cerebral Artery           |
| 13          | Anterior Communicating Artery           |


> Check out [Intracranial Arteries Anatomy](#🧠-intracranial-arteries-anatomy) for a quick refresher

```python
def get_folder_size(path):
    file_sizes = np.array([f.stat().st_size for f in Path(path).rglob('*') if f.is_file()], dtype=float)
    return file_sizes

seg_dir = os.path.join(CONFIG['PATH'], 'segmentations')
file_sizes = get_folder_size(seg_dir)
file_sizes /= 1024.0**2

print(clr.T+"----- Segmentation Size Stats -----\n"+clr.E)
print(f"Total Dir: {file_sizes.sum()/1024.0:.3f} GB\n")
print(f"Mean: {file_sizes.mean():.1f} MB\n")
print(f"Min: {file_sizes.min():.1f} MB\n")
print(f"Max: {file_sizes.max():.1f} MB\n")
```

```python
seg_series = [re.sub(r'(_cowseg)?\.nii$', '', f) for f in os.listdir(seg_dir)]
seg_df = train_df[train_df.SeriesInstanceUID.isin(seg_series)]

all_seg_pivot = pd.pivot_table(
    seg_df,
    index='Aneurysm Present',
    columns='Modality',
    values='PatientAge',
    aggfunc='count',
    margins=True,
    margins_name='Total'
)


fig, ax = plt.subplots(figsize=(6, 4))

sns.heatmap(
    all_seg_pivot, 
    annot=True,
    fmt='g',
    cmap='Reds_r',
    linewidths=0.5, 
    linecolor='gray',
    ax=ax
    
)

ax.set_title('Segmentation Count by Modality and Aneurysm Presence')
ax.set_ylabel('Aneurysm Present')
ax.set_xlabel('Modality')
fig.tight_layout()
fig.show()
fig.savefig('Figures/series_and_segmentation_count_per_modality.png', dpi=300)
```

```python
def plot_3d_segment(series_uid=None, random=False):
    if random:
        series_uid = seg_df.SeriesInstanceUID.sample().iloc[0]

    seg_path = os.path.join(CONFIG['PATH'], 'segmentations', f'{series_uid}_cowseg.nii')
    seg = nib.load(seg_path).get_fdata().astype(np.uint8)
    
    fig = go.Figure()
    
    for label, name in CONFIG['ARTERIES'].items():
        coords = np.where(seg == label)
        if coords[0].size == 0:
            continue
    
        x = coords[0]
        y = coords[1]
        z = coords[2]
    
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers',
            name=f"{label}: {name}",
            marker=dict(
                size=2,
                colorscale='Viridis',
                opacity=0.8)
        ))
    
    fig.update_layout(
        title="3D Segmentation",
        scene=dict(
            xaxis=dict(
                title='Right → Left',
                autorange='reversed'
            ),
            yaxis_title='Posterior → Anterior',
            zaxis=dict(
                title='Superior → Inferior',
                autorange=True
            ),
            aspectmode='data'
        ),
        legend=dict(
            itemsizing='constant',
            font=dict(size=10)
        ),
        width=1000,
        height=900
    )
    
    fig.show()
    fig.write_html("Figures/intracranial_arteries_segmentation.html")
```

```python
plot_3d_segment(random=True)
```

```python
arteries_info = []

for idx, row in tqdm(seg_df.iterrows(), total=len(seg_df)):
    series_id = row['SeriesInstanceUID']

    image_path = os.path.join(seg_dir, series_id+'.nii')
    mask_path = os.path.join(seg_dir, series_id+'_cowseg.nii')

    image = load_nii(image_path, mask=False)
    mask = load_nii(mask_path, mask=True)

    shape = mask.shape
    for idx, artery in CONFIG['ARTERIES'].items():
        density = (mask == idx).mean()
        coords = np.argwhere(mask == idx)
    
        if coords.shape[0] != 0:
            
            min_x, min_y, min_z = coords.min(axis=0)
            max_x, max_y, max_z = coords.max(axis=0)
            len_x = max_x - min_x
            len_y = max_y - min_y
            len_z = max_z - min_z
            
            artery_info = [series_id, shape, artery, min_x, min_y, min_z, max_x, max_y, max_z, len_x, len_y, len_z, density]
        else:
            artery_info = [series_id, shape, artery] + [np.nan] * 10

        arteries_info.append(artery_info)

arteries_df = pd.DataFrame(arteries_info, columns=['SeriesInstanceUID', 'shape', 'location', 'x_0', 'y_0', 'z_0', 'x_1', 'y_1', 'z_1', 'X', 'Y', 'Z', 'Density'])
arteries_df.head()
```

```python
missing_seg = pd.DataFrame(len(seg_df) - arteries_df.groupby('location').Density.count()[CONFIG['ARTERIES'].values()])
missing_seg
```

```python
artery_size = arteries_df.groupby('location')[['X', 'Y', 'Z']].mean().loc[list(CONFIG['ARTERIES'].values())]
artery_size
```

```python
fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(8, 12))

for i, feature in enumerate(['X', 'Y', 'Z']):
    
    sns.boxplot(
        data=arteries_df,
        x=feature,
        y='location',
        ax=axs[i]
    )
    axs[i].set_title(feature)
    axs[i].set_xlabel('')
    axs[i].set_xlim((0, 140))

plt.tight_layout()
plt.show()
fig.savefig('Figures/arteries_size_boxplot', dpi=300)
```

```python
def create_cube_vertices(center, dimensions):
    """Create vertices for a cube centered at 'center' with given dimensions"""
    x_center, y_center, z_center = center
    x_size, y_size, z_size = dimensions
    
    dx, dy, dz = x_size/2, y_size/2, z_size/2
    
    vertices = np.array([
        [x_center-dx, y_center-dy, z_center-dz],  # 0
        [x_center+dx, y_center-dy, z_center-dz],  # 1
        [x_center+dx, y_center+dy, z_center-dz],  # 2
        [x_center-dx, y_center+dy, z_center-dz],  # 3
        [x_center-dx, y_center-dy, z_center+dz],  # 4
        [x_center+dx, y_center-dy, z_center+dz],  # 5
        [x_center+dx, y_center+dy, z_center+dz],  # 6
        [x_center-dx, y_center+dy, z_center+dz],  # 7
    ])
    return vertices

def get_cube_faces(vertices):
    """Define the 12 triangular faces of a cube (2 triangles per face)"""
    faces = [
        # Bottom face (z = z_min)
        [vertices[0], vertices[1], vertices[2]],
        [vertices[0], vertices[2], vertices[3]],
        # Top face (z = z_max)
        [vertices[4], vertices[6], vertices[5]],
        [vertices[4], vertices[7], vertices[6]],
        # Front face (y = y_min)
        [vertices[0], vertices[4], vertices[5]],
        [vertices[0], vertices[5], vertices[1]],
        # Back face (y = y_max)
        [vertices[2], vertices[6], vertices[7]],
        [vertices[2], vertices[7], vertices[3]],
        # Left face (x = x_min)
        [vertices[0], vertices[3], vertices[7]],
        [vertices[0], vertices[7], vertices[4]],
        # Right face (x = x_max)
        [vertices[1], vertices[5], vertices[6]],
        [vertices[1], vertices[6], vertices[2]],
    ]
    return faces


locations = artery_size.index.tolist()
coordinates = artery_size.values.tolist()

fig = plt.figure(figsize=(16, 12))
ax = fig.add_subplot(111, projection='3d')

colors = plt.cm.tab20(np.linspace(0, 1, len(locations)))

grid_size = int(np.ceil(np.sqrt(len(locations))))
spacing = 15  # Space between cubes

for i, (location, coord) in enumerate(zip(locations, coordinates)):
    x_dim, y_dim, z_dim = coord
    
    row = i // grid_size
    col = i % grid_size
    
    x_pos = col * spacing
    y_pos = row * spacing
    z_pos = z_dim * 0.15
    
    center = [x_pos, y_pos, z_pos]
    dimensions = [x_dim*0.3, y_dim*0.3, z_dim*0.3]
    
    vertices = create_cube_vertices(center, dimensions)
    
    faces = get_cube_faces(vertices)
    
    cube = Poly3DCollection(faces, alpha=0.7, facecolor=colors[i], edgecolor='black', linewidth=0.5)
    ax.add_collection3d(cube)

ax.set_xlabel('X Coordinate', fontsize=12)
ax.set_ylabel('Y Coordinate', fontsize=12)
ax.set_zlabel('Z Coordinate', fontsize=12)
ax.set_title('3D Anatomical Locations - Size Comparison', 
             fontsize=14, fontweight='bold')

max_coord = max(grid_size * spacing, max([max(coord) for coord in coordinates]) * 0.3)
ax.set_xlim(-2, (grid_size-1) * spacing + 5)
ax.set_ylim(-2, (grid_size-1) * spacing + 5)
ax.set_zlim(0, max_coord)

legend_elements = [plt.Rectangle((0,0),1,1, facecolor=colors[i], alpha=0.7, edgecolor='black') 
                  for i in range(len(locations))]

mid_point = len(locations) // 2
legend1 = ax.legend(legend_elements[:mid_point], locations[:mid_point], 
                   loc='upper left', bbox_to_anchor=(0.02, 0.98), fontsize=8)
legend2 = ax.legend(legend_elements[mid_point:], locations[mid_point:], 
                   loc='upper left', bbox_to_anchor=(0.02, 0.5), fontsize=8)

ax.add_artist(legend1)

xx, yy = np.meshgrid(np.linspace(-2, (grid_size-1) * spacing + 5, 10),
                     np.linspace(-2, (grid_size-1) * spacing + 5, 10))
zz = np.zeros_like(xx)
ax.plot_surface(xx, yy, zz, alpha=0.1, color='gray')

ax.grid(True, alpha=0.3)
ax.view_init(elev=20, azim=45)

fig.tight_layout()
fig.savefig('Figures/3d_size_comparison.png', dpi=300)
fig.show()
```

# <div style="text-align:center; border-radius:10px; color:black; margin:0; font-size:100%; font-family:Carbon Plus; background-color:white; overflow:hidden"><b>🩸 Intracranial Aneurysm Types</b></div>


Intracranial aneurysms are also classified by their macroscopic shape and size.

![aneurysm_types_3](https://raw.githubusercontent.com/ammomahdavikia/asset-holding/main/aneurysm_types_3.png)

##### **Saccular ("berry") Aneurysm (SA):** 

+ **Epidemiology**
	+ Most common intracranial aneurysm
	+ 3% of population; F > M
	+ Peak age of presentation: 40-60 years (rare in children)
+ **Location**
	+ 90% anterior circulation  
		+ 20% Middle cerebral artery (MCA) bi- or trifurcation.
		+ 30% ACoA
		+ >30% (most common) Internal carotid artery (ICA)/posterior communicating artery (PCoA) junction
	+ 10% posterior circulation
		+ 5% Basilar bifurcation
+ **Size**
	+ Tiny (1-2 mm) to giant (≥2mm)
+ **Number**
	+ 15-20% multiple (> 2, F:M = 10:1)

> Most saccular aneurysms do not rupture.

**What increases the risk of rupture?**
+ Size matters!
	- Rupture risk increases with size
	- ≥ 5 mm greater risk than 2-4 mm
+ Shape, configuration matter!
	- Nonround (nonsaccular shape) = ↑ rupture risk!
	- “Daughter" sac or "tit" = ↑ rupture risk!
+ Location affects rupture risk!
	- Vertebrobasilar, ICA-PCoA location highest rupture risk
	- MCA, anterior cerebral artery (ACA) moderate risk; non-PCoA-ICA aneurysms lowest

![aneurysm_rupture](https://raw.githubusercontent.com/ammomahdavikia/asset-holding/main/aneurysm_rupture.png)

##### **Fusiform Aneurysm (FA):**
+ **Epidemiology**
	+ FAs can be atherosclerotic (common) or nonatherosclerotic (rare).
	+ Atherosclerotic FAs (ASVD FAs) are typically seen in older adults.
+ **Location**
	+ Atherosclerotic
		+ More common in the vertebrobasilar (posterior) circulation and usually affect the basilar artery.
	+ Nonatherosclerotic
		+ The carotid (anterior) and vertebrobasilar circulations are equally affected.
+ **Shape**
	+ Long, non-branching vessel segments
 

##### **Pseudoaneurysm Aneurysm (PSA):**
+  **Location**
	+ Approximately 80% of PSAs affecting the carotid and vertebral arteries are extracranial, whereas 20% involve their intracranial segments. 
+ **Shape**
	+ Irregularly shaped outpouching
		+ "Neck" usually absent
  

##### **Blood-blister like Aneurysm (BBA):**
+ **Epidemiology**
	+ Represent ~ 1% of all intracranial aneurysms and 0.5-2.0% of all ruptured aneurysms.
	+ Uncommon but potentially lethal subtype of intracranial PSA
+ **Location**
	+ Can occur almost anywhere
	+ Dorsal wall of supraclinoid ICA most common site
+ **Shape**
	+ Small, broad-based hemispheric bulges

# <div style="text-align:center; border-radius:10px; color:black; margin:0; font-size:100%; font-family:Carbon Plus; background-color:white; overflow:visible; line-height:1.5; padding:10px 0;"><b>📋 Summary</b></div>

### ↓ Key Takeaways:

#### 1. Dataset Overview
- **Training set:** 4,348 series from 1,864 patients and 2,484 controls.  
- **Test set:** ~2,500 series.  
- Each series is stored as a directory containing DICOM files.  
- DICOM files exist in **single-frame** (92.6%) and **multi-frame** (7.4%) formats.

#### 2. Modalities
- The dataset contains **four imaging modalities**:  
  - CTA: 41.5%  
  - MRA: 28.8%  
  - MRI T2: 22.6%  
  - MRI T1 post-processed: 7.0%  
- **Image size:** Most images are 512×512 pixels.  
- **Number of frames per series (median/approx):**  
  - CTA: ~400  
  - MRA: ~175  
  - MRI T2: ~150  
  - MRI T1 post-processed: ~35  

#### 3. Imaging Planes
- Three planes are present: axial, sagittal, and coronal.  
  - Axial: 96.5% (dominant)  
  - Coronal: 2%  
  - Sagittal: 1.5%  

#### 4. Aneurysm Distribution
- Aneurysms are annotated in **13 vascular sites**, with the following most common:  
	+ Anterior Communicating Artery: 16.7%
	+ Left Supraclinoid Internal Carotid Artery: 15.23%
	+ Right Middle Cerebral Artery: 13.5%
	+ Right Supraclinoid Internal Carotid Artery:  12.75%
	+ Left Middle Cerebral Artery: 10%
	+ Other Posterior Circulation: 5.2%
	+ Basilar Tip: 5.06%
	+ Right Posterior Communicating Artery: 4.65%
	+ Right Infraclinoid Internal Carotid Artery: 4.5%
	+ Left Posterior Communicating Artery: 3.95%
	+ Left Infraclinoid Internal Carotid Artery: 3.6%
	+ Right Anterior Cerebral Artery: 2.6%
	+ Left Anterior Cerebral Artery: 2.1%
- **One slice per aneurysm** is provided (best-seen slice).  
- Most patients (1,615/1,864) have a single-artery aneurysm; others have multiple.  
- Annotated slices follow a **roughly normal distribution** along the Z-axis, with peaks in ranges **[0–100]** and **[200–300]**.

#### 5. Segmentation Masks
- The `segmentation` folder contains NIfTI masks for 13 vascular sites.  
- Only a subset of series (178/4,348) have segmentation masks.  
- Availability of segmentation mirrors the main dataset in terms of modality and plane distribution.  
- Average mask coverage:  
  - **Most:** Other Posterior Circulation  
  - **Least:** Anterior Communicating Artery


![rsna2025_summary](https://raw.githubusercontent.com/ammomahdavikia/asset-holding/main/rsna2025_summary.png)

# <div style="text-align:center; border-radius:10px; color:black; margin:0; font-size:100%; font-family:Carbon Plus; background-color:white; overflow:visible; line-height:1.5; padding:10px 0;"><b>📚 References</b></div>

[1]: Osborn, A. G. *Osborn's Brain*. 3rd edn. (Elsevier, 2024).

[2]: Ropper, A. H., Samuels, M. A., Klein, J. P. & Prasad, S. *Adams and Victor’s Principles of Neurology*. 12th edn. (McGraw Hill, 2023).

[3]: Zhang, Y. et al. A survey of intracranial aneurysm detection and segmentation. *Med. Image Anal.* **90**, 103493 (2025). https://doi.org/10.1016/j.media.2025.103493

[4]: Beaman, C., Patel, S. D., Nael, K., Colby, G. P. & Liebeskind, D. S. Imaging of intracranial saccular aneurysms. *Stroke Vasc. Interv. Neurol.* **3**, e000757 (2023).

[5]: Kumar, V., Abbas, A. K. & Aster, J. C. *Robbins & Cotran Pathologic Basis of Disease*. 10th edn. (Elsevier, 2020).

[6]: Haaga, J. R. et al. *CT and MRI of the Whole Body*. 6th edn. (Elsevier, 2017).

### <div style="text-align:center; border-radius:10px; color:black; margin:0; font-size:150%; font-family:Carbon Plus; background-color:white; overflow:visible; line-height:1.5; padding:10px 0;"><b>⬆️ Your Welcome ⬆️</b></div>