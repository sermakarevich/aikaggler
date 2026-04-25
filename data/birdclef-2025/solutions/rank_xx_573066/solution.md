# Recipe to Public LB 0.872

- **Author:** Salman Ahmed
- **Date:** 2025-04-13T10:19:32.400Z
- **Topic ID:** 573066
- **URL:** https://www.kaggle.com/competitions/birdclef-2025/discussion/573066
---

I am sharing what worked for me in early baseline, but still haven't figured out what's working yet, still trying. 

1. Applying any augmentation or processing on raw waves, hurts the performance.
2. When I look at last year's leaderboard, position # 5 was able to achieve a awesome melspec model without any pseudo labels or anything, while other top solutions used pseudo labels. I believe melspec parameters were one of the import things in that solution. (Not sure.) Considering this, I started exploring and I was able to get scores from 0.810 to 0.859 just by changing the melspec parameters. You just have to think about it, visualize and explore more. (5 subs a day to find the best parameters).
3. I've tried BCE with multiple types of weights, but FocalBCE works best for me.
4. Consider melspecs as images and not the waves, I converted the melspecs to images, applied normalization and augment with RandAug and RandomErazing, Time and Freq Masking + (Mixup with prob of 1.0). Remaining training pipeline is same as my public notebook from last year. [https://www.kaggle.com/code/salmanahmedtamu/training-0-65-0-66](https://www.kaggle.com/code/salmanahmedtamu/training-0-65-0-66)
5. Another important thing, in Efficientnet models when i combined middle features and passed into the global pool that worked better than the conv_head in timm. Still Exploring.....!!!!
6. Even simple models starts overfitting pretty easily, and you cannot track because K-Fold doesn't make sense here, your validation set is the public LB. Keep track of the Batch Size, Number of iterations, warmp up and LR. 
7. Filtering/Processing CSA recordings by Human sound helps.
8. Look at predictions of your model and visualize the attentions on Train Soundscape, to see what's the CAM of your model.
9. Ensemble to 0.854, 0.856, 0.858 and 0.859 results in 0.872. 
10. Post Processing is important as I shared in my previous public notebook, There are a lot of other ways to post process as well.
11. I am sure, I missed a lot of detail, I'll try to share some public notebook if i get some time this week.
12. Training on Random 5 seconds helped compared to first 5 seconds.