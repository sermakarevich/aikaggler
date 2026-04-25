# Rank 106 approach - Simple feature sets and CV-LB stability focus

- **Author:** Ravi Ramakrishnan
- **Date:** 2024-12-20T01:21:09.337Z
- **Topic ID:** 552492
- **URL:** https://www.kaggle.com/competitions/child-mind-institute-problematic-internet-use/discussion/552492
---

Hello all,

Firstly thanks to Kaggle and CMI for the competition. I think this was one of the many churn-filled competitions this year, after Home Credit, ISIC, BirdClef and AES and I was in all of them (and shook down every time)! I participated in this challenge with the lessons learnt from all of them and tried to prevent a downward movement in the churn here and was successful at it!

# Approach summary
- I resorted to a very simple pipeline, comprising of single models that offered me at least a level of CV-LB relation and stability
- I used 3-5 feature sets and submitted a simple average blend of constituent models from the above step, choosing the best models that offered me a CV-LB stability 
- I decided not to opt for any complex models like NNs, TabNet, AutoEncoder and resorted to simple boosted tree models only without early stopping

# Training code
- CMI2024|Final|Candidate1 - https://www.kaggle.com/code/ravi20076/cmi2024-final-candidate1
- CMI2024|Final|Candidate2 - https://www.kaggle.com/code/ravi20076/cmi2024-final-candidate2

# Feature engineering
- I tried a lot of features using the actigraphy data to no impact/ marginal impact. I restricted myself to public features at the end (descriptive statistics with a small addition/ removal of features in my final submissions)
- I cleaned up some noisy values in the dataset otherwise, based on domain knowledge (examples include blood-pressure values, BMI values, etc.)
- I did not impute any targets from the unknown sii section of the data and left this as-is (this was excluded from my model training)
- I resorted to 3 feature sets in one submission and 5 in another and landed up with the same private LB score across both of them

## Target choices 
I resorted to 2 target choices here
- Direct prediction of sii target
- Predicting sii using PCIAT-PCIAT1-20 and then adding up the predictions followed up with rule-based allocation as below-
a. Predicted PCIAT-Total between 0 and 30 -> 0
b. Predicted PCIAT-Total between 31 and 49 -> 1
c. Predicted PCIAT-Total between 50 and 79 -> 2
d. Predicted PCIAT-Total between 80 and 100 -> 3

Using 2 targets like this offered me better stability on the ensemble, between the CV and public LB scores

# Model training  

## Offline model
- Cross-validation - 5 fold stratified by sii values 
- A simple LightGBM with public notebook parameters worked well for me across the competition. I did not tune anything here as any attempt to tune my models led me to an undesirable CV-LB relation. A lot of my tuned models have also tanked on the private LB and I am happy I did not rely on any form of tuning here!
- I did a target tuning just like a lot of public kernels, but **used the training data for tuning and not the OOF data**. Tuning with a larger dataset across folds led me to a better CV-LB stability and relatively lesser impact on the public leaderboard on changing random seeds. 
- I designed a simple class inheriting from a LightGBM regressor and tuned my thresholds as below-
a. Train a regressor with the fold-level training data
b. Use the training data predictions and tune the thresholds using scipy.optimize.minimize like the ones in public kernels 
c. Store the thresholds for later use

## Full refit model
- Once my offline CV scheme was ready, I refitted the model on the full training data (excluding the null target columns) and tuned the thresholds similar to the offline model process (using the full train set)
- I averaged the model across a large number of random states (typically 100+) and engendered stability with **multistarts**
- I submitted this averaged model on the full training data to the leaderboard and varied the random states in the full refit (and ultimately the tuning process on full-fit) to ascertain my CV-LB stability. The chosen feature sets and submissions proved to be relatively least unstable and were included as candidates 

## Feature components and individual submission results 

| Feature set | CV  | Public LB score | Target | 
| --- | --- | ----------- | ---- | 
| Set 1 - 163 features | 0.463429 | 0.471| PCIAT_*|
| Set 2 - 163 features | 0.466009 | 0.470| PCIAT_*|
| Set 3 - 163 features | 0.466455 | 0.466| PCIAT_*|
| Set 4 - 163 features | 0.468296 | 0.466| sii |
| Set 5 - 141 features | 0.463159 | 0.470 | PCIAT_*|

<br>**Overall result - Public/ Private LB -> 0.471 / 0.454** 
<br>

| Feature set | CV  | Public LB score | Target | 
| --- | --- | ----------- | ---- | 
| Set 1 - 163 features  | 0.466184 | 0.472| PCIAT_*|
| Set 2 - 163 features | 0.468296 | 0.466| sii |
| Set 3 - 141 features  | 0.463159 | 0.470 | PCIAT_*|

<br>**Overall result - Public/ Private LB -> 0.475 / 0.454** 

# What I could have done better here
- Better choice of final submissions 
- Better choice of candidate feature sets
- One of my feature sets with a lower public LB score scored extremely well on the private set. I could have chosen it perhaps!

# What I gained here
- Training with threshold tuning on the training data rather than the OOF data was a good step for me and worked for me here
- My stability across the leaderboards was also a gain, after my falls in yester shakeup driven competitions
- I am happy I did not use any complex models and algorithms here and ended up with a relatively stable result. I think this itself feels like success given the churn in the competition

# Concluding remarks
Good luck for all your competitions going ahead! All the best and happy learning!

Best regards,
Ravi Ramakrishnan