# BYU Biophysics Group Starter

- **Author:** Andrew Darley
- **Votes:** 233
- **Ref:** andrewjdarley/byu-biophysics-group-starter
- **URL:** https://www.kaggle.com/code/andrewjdarley/byu-biophysics-group-starter
- **Last run:** 2025-03-07 23:25:43.160000

---

# BYU Locating Flagellar Motors

## Solution Overview

This is the index notebook for my solution to the BYU Locating Bacterial Flagellar Motors 2025 Kaggle competition. I used a 2D YOLOv8-based approach by training a bounding box model on the slices in the dataset that contained motors. To submit the model, I had it iterate over every slice in each tomogram of the test dataset and had it select the point of highest confidence as the annotation. 

### Notebook Series:

1. **[Parse Data](https://www.kaggle.com/code/andrewjdarley/parse-data)**
   - Extracts 2D slices with motors
   - Normalizes slice intensity using percentile-based contrast enhancement (standard across all work here)
   - Converts annotations to YOLO format
   - Creates train/validation splits at the tomogram level. ie creates an 80/20 split of motors, not tomograms

2. **[Visualize Data](https://www.kaggle.com/code/andrewjdarley/visualize-data)**
   - Visualizes random training samples with annotations
   - Confirms proper bounding box placement

3. **[Train YOLO](https://www.kaggle.com/code/andrewjdarley/train-yolo)**
   - Fine tunes YOLOv8
   - Monitors training/validation losses with early stopping, dfl loss is all that matters for this application
   - Validates model performance on val slices

4. **[Submission Notebook](https://www.kaggle.com/code/andrewjdarley/submission-notebook)**
   - Processes test tomograms with GPU optimization
   - Implements 3D non-maximum suppression for detection clustering
   - Generates the final submission CSV
   - Runs in an offline environment using pre-installed dependencies (This notebook was a lifesaver: https://www.kaggle.com/code/itsuki9180/ultralytics-for-offline-install)

A complete notebook combining all the above can be found [here](https://www.kaggle.com/code/sharifi76/eda-visualization-yolov8)

## Solution Approach

This solution treats the bacterial flagellar motor detection problem as a 2D object detection task with 3D post-processing. Key aspects include:

- **Data Preprocessing**: We extract multiple slices around each annotated motor to capture 3D context, normalize intensity for better contrast, and convert annotations to YOLO format.

- **Model Architecture**: Using YOLOv8 with transfer learning from pre-trained weights to accelerate training on the specialized dataset.

- **Inference Strategy**: The inference pipeline processes test tomograms in batches with GPU optimization, detecting motors in 2D slices and then merges nearby motors.

## Requirements

- Ultralytics YOLOv8 package through offline install

## Competition Details

The BYU Locating Bacterial Flagellar Motors 2025 challenge involves locating flagellar motors in CryoET bacteria tomograms.