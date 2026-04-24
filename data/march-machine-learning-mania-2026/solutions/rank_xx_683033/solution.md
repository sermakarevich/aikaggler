# My approach for this year

- **Author:** raddar
- **Date:** 2026-03-19T16:23:54.057Z
- **Topic ID:** 683033
- **URL:** https://www.kaggle.com/competitions/march-machine-learning-mania-2026/discussion/683033
---

I mainly used last year's code and made some minor improvements. There are some ideas that worked for me - maybe someone will find some inspiration for next year:

- [Use ESPN box scores](https://www.kaggle.com/datasets/raddar/ncaa-espn-box-scores) where available. I scraped all the data from their API, it has 99% overlap with competition dataset for last ~10 seasons. This means consistent rebound/turnover data, got 0.002-0.005 Brier boost on affected seasons. 
- Filter out teams with <33% regular season win rate from the dataset. Cut-off hand-picked based on feature correlations and validation.
- Remove play-in games from training models (looking at MSE, these are at the top of the outlier list if used in training set). (seed10 vs seed10 does not imply teams are of equal strength). This simple trick improved my validation metrics the most.
- Switch team quality from binary win/lose to point difference + adjust to missed free throws + location as a factor. Best feature. Use sample weights based on seeds and DayNum.
- Same as team quality, build conference quality, only consider games where team conferences are different. 
- ELO with season replays (each replay takes previous ELO as initial ELO). Replay n=5 times. Calculate ELO not based on binary win/lose, but pointdiff - expected (situation possible that a team might lose ELO if it won by lower than expected margin).
- Use Massey ordinals at DayNum=133. worst_rank(POM, WLK, MOR) gave me best results. Additionally, build a proxy model which predicts men's Massey, use this proxy model with women's features to predict women's Massey rank. Works very well.
- Poisson count model for number of expected wins in the tournament. Fit season-by-season to remove any possible leakage. `y = 2**num_wins`. Use `poisson_model.predict()` as a feature.  
- Similar to Poisson count model (`y = 2**num_wins`), calculate statistics of number of reached rounds in previous tournaments. EMA(y) with half-life=2.5years is incredibly strong. This is a good proxy to measure tournament experience.
- Build bracket of "better team always wins" - use majority vote of seed, quality, ELO. Use expected number of wins as a feature.