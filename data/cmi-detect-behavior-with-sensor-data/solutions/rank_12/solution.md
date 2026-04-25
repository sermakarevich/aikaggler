# 12th place solution

- **Author:** Ruby
- **Date:** 2025-09-03T03:59:09.757Z
- **Topic ID:** 603564
- **URL:** https://www.kaggle.com/competitions/cmi-detect-behavior-with-sensor-data/discussion/603564
---

Thanks Kaggle and organizer for hosting this interesting competition. A special thanks as well to the valuable public notebooks which helped me a lot:
https://www.kaggle.com/code/tarundirector/sensor-pulse-viz-eda-for-bfrb-detection
https://www.kaggle.com/code/jiazhuang/cmi-imu-only-lstm
https://www.kaggle.com/code/kaigaokaigao/lb-0-75-imu-only-multibranch-inference/notebook
I think their work deserves much more votes than rubbish ensemble notebooks.

**Data Processing**
1. IMU
  - rot in this dataset is normalized to rot_w>0, recover sign by forcing dot_product(rot,rot[-1]) >=0 with sign flip
  - handedness flip: -acc_x, -rot_y, -rot_z
  - no normalization applied, using BN as first layer for convenience
2. THM
  - set all sequence sensor values to nan if : mean>40 or mean<20 or std==0
  - handedness flip: switch sensor 3 and 5
  - normalization: (t-27)/3
3. TOF
  - value <0 or >249 (if any) to nan
  - handedness flip: switch sensor 3 and 5, up-down flip for sensor 3 and 5, left-right flip for sensor 1,2,4
  - normalization: (250-x)/65
  - nan to 0 (250 before normalization), no interpolation

**Feature Engineering**
1. IMU
  - acc/ rot/ angular velocity (in device and world frame)/ acc with gravity remove (in both frame)/ angular velocity abs_diff/ gravity projection in device frame/ handedness (actually meta feature, too lazy to make it an independent branch)
  - most features come in group of size 3 and rot with size 4, I add zero columns to make feature matrix in format of bins with size 4. This is for later group convolution 
2. TOF
  - 0-1 nan indicator as extra feature

**Model**
1. IMU branch
  - two standard 1DResBlock with SCSE attention 
  - using group conv such that interaction only happens within group (eg: within acc_x/y/z)
2. THM branch
  - two standard 1DResBlock with SCSE attention
3. TOF branch
  - three standard 3DResBlock with SCSE attention, kernel size (1,3,3)*2+(3,1,1)
  - group conv such that interaction only happens within same TOF sensor
4. Fusion
  - two GRU + MHA (with ROPE) + AttentionPooling + MLP
  - I use other combination among GRU/ResBlock/MHA for ensemble
  - GRU: I modularize it with skip-connect and layernorm
  - MHA: I found extra feedforward layer in common transformer block useless in this dataset and I only keep the MHA part


**Multi-task Learning**
1.  main loss: cross entropy + dice loss with cross entropy ratio decayed while training
2. cross entropy on orientation classification
3. MSE on ratio of gesture region (I didn’t properly normalize this part, not sure if worked)
4. MAE on main loss (not sure if worked)

**Augmentation**
1.	rot sign flip
2.	world frame x-y plane rotation (change facing direction)
3.	time stretching
4.	sequence patch mask
5.	TOF pixel jitter
6.	TOF patch mask
7.	sensor mask

**Other Details**
1.	onecycle lr, AdamW, EMA, mixup, dropout, silu, truncate sequence with tail region pre-padding by zero to length 120
2.	I run simple random search over params space and manually clip search space after analysis

**Ensemble**
1.	class rank average among Kfolds, voting with different models
2.	based on local CV tuning ensemble weights or selecting from a large pool easily cause overfitting, so I restricted to models with good CV and only apply voting (better than average might due to some over confident models)
3.	gains
single best (based on CV) IMU/All:0.8364/0.9008, public/private: 0.868/0.859
ensemble IMU/All:0.8458/0.906 public/private: 0.875/0.862

**Reproduce**
I only share my best CV single model training and inference code to avoid publishing too much notebooks of similar content.
1.	IMU only training: https://www.kaggle.com/code/w5833946/m21-cnn-gru-train
2.	All feature training: https://www.kaggle.com/code/w5833946/all-m14-cnn-gru-train  (trained offline with same code, download from google drive)
3.	Inference: https://www.kaggle.com/code/w5833946/cmi3-submit-single