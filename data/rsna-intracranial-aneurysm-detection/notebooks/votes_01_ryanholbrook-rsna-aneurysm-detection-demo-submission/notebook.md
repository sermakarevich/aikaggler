# RSNA Aneurysm Detection Demo Submission

- **Author:** Ryan Holbrook
- **Votes:** 908
- **Ref:** ryanholbrook/rsna-aneurysm-detection-demo-submission
- **URL:** https://www.kaggle.com/code/ryanholbrook/rsna-aneurysm-detection-demo-submission
- **Last run:** 2025-07-28 20:30:44.650000

---

```python
import os
import shutil
from collections import defaultdict

import pandas as pd
import polars as pl
import pydicom

import kaggle_evaluation.rsna_inference_server
```

The evaluation API requires that you set up a server which will respond to inference requests. We have already defined the server; you just need write the predict function. When we evaluate your submission on the hidden test set the client defined in `rsna_gateway` will run in a different container with direct access to the hidden test set and hand off the data series by series.

Your code will always have access to the published copies of the files.

```python
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

# All tags (other than PixelData and SeriesInstanceUID) that may be in a test set dcm file
DICOM_TAG_ALLOWLIST = [
    'BitsAllocated',
    'BitsStored',
    'Columns',
    'FrameOfReferenceUID',
    'HighBit',
    'ImageOrientationPatient',
    'ImagePositionPatient',
    'InstanceNumber',
    'Modality',
    'PatientID',
    'PhotometricInterpretation',
    'PixelRepresentation',
    'PixelSpacing',
    'PlanarConfiguration',
    'RescaleIntercept',
    'RescaleSlope',
    'RescaleType',
    'Rows',
    'SOPClassUID',
    'SOPInstanceUID',
    'SamplesPerPixel',
    'SliceThickness',
    'SpacingBetweenSlices',
    'StudyInstanceUID',
    'TransferSyntaxUID',
]

# Replace this function with your inference code.
# You can return either a Pandas or Polars dataframe, though Polars is recommended.
# Each prediction (except the very first) must be returned within 30 minutes of the series being provided.
def predict(series_path: str) -> pl.DataFrame | pd.DataFrame:
    """Make a prediction."""
    # --------- Replace this section with your own prediction code ---------
    series_id = os.path.basename(series_path)
    
    all_filepaths = []
    for root, _, files in os.walk(series_path):
        for file in files:
            if file.endswith('.dcm'):
                all_filepaths.append(os.path.join(root, file))
    all_filepaths.sort()
    
    # Collect tags from the dicoms
    tags = defaultdict(list)
    tags['SeriesInstanceUID'] = series_id
    global dcms
    for filepath in all_filepaths:
        ds = pydicom.dcmread(filepath, force=True)
        tags['filepath'].append(filepath)
        for tag in DICOM_TAG_ALLOWLIST:
            tags[tag].append(getattr(ds, tag, None))
        # The image is in ds.PixelData

    # ... do some machine learning magic ...
    predictions = pl.DataFrame(
        data=[[series_id] + [0.5] * len(LABEL_COLS)],
        schema=[ID_COL, *LABEL_COLS],
        orient='row',
    )
    # ----------------------------------------------------------------------

    if isinstance(predictions, pl.DataFrame):
        assert predictions.columns == [ID_COL, *LABEL_COLS]
    elif isinstance(predictions, pd.DataFrame):
        assert (predictions.columns == [ID_COL, *LABEL_COLS]).all()
    else:
        raise TypeError('The predict function must return a DataFrame')

    # ----------------------------- IMPORTANT ------------------------------
    # You MUST have the following code in your `predict` function
    # to prevent "out of disk space" errors. This is a temporary workaround
    # as we implement improvements to our evaluation system.
    shutil.rmtree('/kaggle/shared', ignore_errors=True)
    # ----------------------------------------------------------------------
    
    return predictions.drop(ID_COL)
```

When your notebook is run on the hidden test set, `inference_server.serve` must be called within 15 minutes of the notebook starting or the gateway will throw an error. If you need more than 15 minutes to load your model you can do so during the very first `predict` call.

```python
inference_server = kaggle_evaluation.rsna_inference_server.RSNAInferenceServer(predict)

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    inference_server.run_local_gateway()
    display(pl.read_parquet('/kaggle/working/submission.parquet'))
```