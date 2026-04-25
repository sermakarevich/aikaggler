# [Public LB 17th] Solution

- **Author:** snehal
- **Date:** 2025-01-14T00:05:31.363Z
- **Topic ID:** 556541
- **URL:** https://www.kaggle.com/competitions/jane-street-real-time-market-data-forecasting/discussion/556541
---

Hey all,

I really enjoyed working on this competition and I thought I'd share my final solution here.

# Acknowledgment
First I wanted to thank @victorshlepov @lihaorocky for all their amazing contributions in the discussion section. I benefited so much from the knowledge they shared along and I'm sure others did as well. I also wanted to mention @johnpayne0 's discussion post on the responders and the tags. Really amazing findings by him even though I couldn't find a way to use it. I love reading great data analysis discussion posts like his which discover some hidden pattern in the data. Despite all the randomness and obfuscation he managed to figure out what was going on.

# Solution
### Model Architecture
Architecture consisted of a 50/50 ensemble between 3 layers transformer encoder with attention over all symbol_id and 3 layers transformer encoder with attention over all time_id and learnable positional encoding. I also used GELU activations to prevent dead neurons caused by ReLU so during online learning parameters could come back into play if necessary. Also tried SiLU and PReLU but GELU was best.

### Features
`features = [f'feature_{i:02d}' for i in range(0, 79)] + ['time_id', 'weight']`. Also added signed log features for train stability and to more easily learn multiplicative features in log space.

### Normalization
Global normalization using sklearn StandardScaler.

### Missing Values
Fill with 0

### Train setup
Used AdamW, model EMA, gradient clipping and trained on full dataset. Trained model to predict all 9 responders because it seemed to have better train stability and generalization at least from my experiments.

### Online Learning (main score boost)
How did I get OL to work? At first I tried a simple idea of doing a single update step on the latest day of data given by the API. This barely helped. I then ran an experiment to see how much I could improve my score on the last 20-30 days of validation data if I fine tuned my model on the few weeks (can't exactly remember how many) just prior to that. After playing around with some settings I found that I could train on this data for around 7 epochs and it would give me a significant boost in the CV score for the last days. This gave me a starting point. I decided to train on the last 7 days every day in order to update the model. This way the model is trained 7 times on each new day (7 "epochs"). I used lr=1e-4 instead of the original 1e-3 and everything else about the setup is exactly the same as my original train setup.


--------------------------------------------------------------------------------------------------------------------
I tried a lot to reach 0.01+ LB but am absolutely stumped in figuring out what the top teams are possibly doing to reach such scores. Very much looking forward to seeing solutions from all of them. Thanks Jane Street for a fun competition!

EDIT:
Tried a few things after the competition and it seems like training with larger number of dates less frequently (e.g. retrain model on last 56 days once every 8 days rather than on the last 7 days every day as I described above gives around 0.0012 improvement which is quite significant. This is likely what I missed in getting to 0.01+ public LB score)