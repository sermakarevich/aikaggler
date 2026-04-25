# [Public LB 12th] Competition Wrap-up: Great Journey and Thank you all! 

- **Author:** SLi
- **Date:** 2025-01-14T00:51:20.900Z
- **Topic ID:** 556548
- **URL:** https://www.kaggle.com/competitions/jane-street-real-time-market-data-forecasting/discussion/556548
---


Finally, it is time to wrap up this incredible journay with Jane Street and Kaggle. I have learned so much from this competition, and I am grateful for the opportunity to work on such challenging and rewarding projects. 

I would like to express my gratitudes to the Jane Street and Kaggle for organizing this competition and providing us with the support. I would also like to thank other participants who have generously shared their knowledge and insights throughout this competition. My special thanks go to @victorshlepov and @lihaorocky , who have shared pure gold discussion posts that greatly enlightened me. 

# Some of the big learnings:

1. **Pipeline building**: I realized in the very beginning that building a robust inference pipeline, especially with online learning, is crucial for this competition. However, it is definitly not trivial and requires lots of efforts on the code design and optimization. The evaluation api together with the hidden test set, which are not easy to hack, make it even more challenging. So I started with building a synthetic test set to help debugging the pipeline at the beginning, and shared this work in the community. This has helped me a lot and I am very happy to see it found to be useful by many people too (46 upvotes, 227 copies). 

2. **GBDT models vs Deep Learning**: I stared with LightGBM and XGBoost as my baseline models, but I could not effectively improve their performance. So at very early stage, I have switched to neural networks, which have shown to be more powerful in this competition. I have tried different architectures, including MLP, GRU, and Transformer, as well as different training strategies. There were several weeks that I was stuck with negative R2 scores, during which I almost exhusted all possible model architetures that I know of (e.g. iTransformers, patchTST, Convnet, etc.). It was a very frustrating period, but finally I was lucky to find a working solution. I definitely benefited a lot from the public discussions, especially the ones from @victorshlepov and @lihaorocky . These are pure gold and I would recommend everyone to read (and upvote) them. 

3. **Nature of the data**: As the host described in the competition overview, we have encountered all kinds of challenges one can imagine. Fat tailed distributions, non-stationary time-series, noisy signals and so on. A good EDA, and especially some visulization pipelines can really help to build ituitive understanding of the data. My early EDA work was also shared in the community, but I felt it could be further improved. For instance, if I could have done an analysis on the responders as amazing as @johnpayne0 does in his great post, I would have figure out more ways to design the model. 

4. **Other tweaks**: The frustrated try-and-error period was not really for nothing. I have greatly improved my understandings about many model archetectures in a practical way. These are the "get hands dirty" times that I did learn a lot. Some tweaks I learned including the different normalization methods, loss functions, feature fusion modules and training strategies. Although not all of them were useful in the end, I am happy to have tried them.


# Short summary of my solution

I will keep this part short as there are 6 month remaining. 

* For models, I designed two different architectures using basic ingredients including GRU, MLP and Transformer (symbol-wise attention). Under each architecture, feature maps were ensemble in two ways, resulting in 4 different models.

* Features are the 79 raw features excluding `9, 10, 11`, `time_id`, `weights`, as well as mean and std of the lagged responders. Missing values were filled with zeros.  All responders were used as targets (instead of only Responder_6). As @eivolkova pointed out, using auxiliary targets can greatly boost both CV & LB.

* Models were validated using the last 120 days, with both offline and online mode. Training sets includes three settings, i.e. 978 days, 800 days and 600 days. Eventually only models trained with the 978 and 800 days were used in the final ensemble (8 models).

* **Online learning** was designed to update the model on a daily basis, using a similar setting as the training. Unlike @eivolkova 's solution, I did not differentiate the responder_6 loss and auxiliary targets loss during the online update. The model updating is quite fast. It was about 0.5~0.7 sec per model per day. A full online training using every 120 or 200 days could further boost the score, however I did not implement it as it will complicate the whole pipeline quite a lot. The major concern is the 1-min limit. 

# Cheers!

At this moment, it is wayyy too early to say anything about the final ranking. The six month ahead will be the real challenge. I will keep my finger crossed and hope nothing in my pipeline breaks. I wish everyone good luck and gets the worthy rewards for the hard work. 


Thank you all!