# 369th place solution YOLO part with PB0.840 notebook

- **Author:** min fuka
- **Date:** 2025-06-05T01:24:22.037Z
- **Topic ID:** 583133
- **URL:** https://www.kaggle.com/competitions/byu-locating-bacterial-flagellar-motors-2025/discussion/583133
---

Thanks to our hosts for organizing this competition. Our placing was poor, but I am happy to report that we scored reasonably well on the YOLO model.
※The team's highest score was PB 0.851

## Making models

- base model
    
     yolov8l, yolo11l  (yolov8 had a better LB score)
    
- original data
    
    competition data + external data(https://www.kaggle.com/datasets/brendanartley/cryoet-flagellar-motors-dataset)
    
- making yolo dataset
    
    based on **Parse Data notebook(**https://www.kaggle.com/code/andrewjdarley/parse-data)
    
    I have devised the following points
    
    - When creating training data, TRUST=8 for competition data and TRUST=2 for external data.
    - When creating validation data, TRUST=4 for competition data sets only, not use external data.
    - slice images are resized to 960x960 for uniformity
    - vertical flipping,  horizontal flipping, and horizontal-vertical flipping data augmentation is performed on competition data only
    - BOX_SIZE is int(1000 / voxel_spacing) for competition data, 35 for external data.
    - 4-fold datasets
- training
    
    based on Train Yolo **notebook(**https://www.kaggle.com/code/andrewjdarley/train-yolo)
    
    I have changed the following points
    
    - batch-size = 32, epoch=30, imgsz=960
    - yolov8l:lr0=1e-4, lrf= yolo11l:lr0=0.001  cos_lr=True, etc
    - save_period=1 and use best_dfl_loss-epoch model

## Inference

In  YOLO model, I have devised a calculation of the z-coordinate. In addition, I have tried to speed up the process.

- calculation of motor coordinates
    
    I used DFS(depth-first search). 
    The x,y coordinates detected in yolo with the slice number as the z coordinate were grouped by DFS and the average value was taken. This idea originated from @itsuki9180 's notebook in CZII (https://www.kaggle.com/code/itsuki9180/czii-yolo11-submission-baseline)
    
- speed up inference
    
    Submit time for public note takes 6.5h for one model. To improve the processing speed, the following efforts were made. This resulted in submit time of 1h40m for one model.
    
    - tomo_id is divided into 4 groups and handled by 4 processes.
    - 2 processes share one T4-GPU
    - In 1 porcess, read jpg files in a separate thread from yolo predictions
    - yolo-predict in half-precision (FP16)
    
    ![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F8234844%2F2975859e11f30ffc95edfba1e035bea5%2Fimage.png?generation=1749086495125297&alt=media)
    
     
    
- yolo-models ensemble
    
    The yolo-models ensemble collects the per-slice detection coordinates across the entire models and processes them in DFS at once.
    
- slices-skip
    
    Even with the above multi-processing, it takes more than 6 hours to ensemble a 4-fold training model. The ensemble of the 4-fold model was reduced to 1h40m by reducing the number of slices in this @yyyy0201 discussion(https://www.kaggle.com/competitions/byu-locating-bacterial-flagellar-motors-2025/discussion/578461).
    
    Prepare array of four randomly selected slices for every 4 slices and assign them to each 4-fold model.
    
    ```
        selected_indices = [random.choice(list(range(i, min(i+4, slen)))) for iin range(0, slen, 4)]
        selected_indices2 = [random.choice(list(range(i, min(i+4, slen)))) for iin range(0, slen, 4)]
        selected_indices3 = [random.choice(list(range(i, min(i+4, slen)))) for iin range(0, slen, 4)]
        selected_indices4 = [random.choice(list(range(i, min(i+4, slen)))) for iin range(0, slen, 4)]
    ```
    
     This resulted in a submission time of about 3hours for an ensemble of yolov8l and yolo11l 4-fold models (8models in total). and the PB score is LB0.840. this inference notebook is (https://www.kaggle.com/code/minfuka/byu-yolo-960-inference-fold4-slice-skip-random-v8l)
    (but, this notebook could be described more compactly)

    

## Score Transition

LB Score Transition. This score is the score before it was recalculated(https://www.kaggle.com/competitions/byu-locating-bacterial-flagellar-motors-2025/discussion/574729).

All scores are using yolov8L and hold-out(8:2) dataset and best.pt.

Initially I was submitting best.pt but changed to best_dfl_epoch.pt due to unstable best.pt scores.

- Change image size during training from 640 to 960 → LB 0.65
- Change from BS=16 to BS=32 → LB 0.699
- Increase target slice for data set creation in public notes to Train TRUST=8 Valid TRUST=4→ LB 0.76
- vertical flipping,  horizontal flipping, and horizontal-vertical flipping data augmentation is performed on competition data → LB 0.80
- Use publicly available external data → LB0.825

## What didn't work

- TTA
- WBF