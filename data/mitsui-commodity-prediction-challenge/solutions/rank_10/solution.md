# Regularized-naive #10 solution

- **Author:** miguel perez
- **Date:** 2026-01-17T14:35:15.180Z
- **Topic ID:** 668589
- **URL:** https://www.kaggle.com/competitions/mitsui-commodity-prediction-challenge/discussion/668589
---

Hi everyone,

Just a quick post to share my approach after finishing #10 on the final leaderboard, inside the prize range. I'm sorry that so many subs failed in the last run, I know well how that feels, it really sucks. The time series API is a pain to deal with and a reason why I skip a good number of similar competitions here.

In spite of this, I want to share my approach, in a nutshell, in case it can provide some insights:

This competition caught my eye initially because commodity spreads made me think of pairs cointegration, which was interesting. But looking closer, the setup had nothing to do with that:  
- Targets were return-based spreads rather than price-based. 
- 400 spreads to predict and rank daily  
- Targets were grouped, the rank involving spreads across different time horizons  
- Provided (target-generating) price series seemed to have synthetically removed carry effects in a non-straightforward way (possibly equal-time weighted?)  
- Test period was short, only ~3 months, too short for a small statistical edge to beat random chance.

All this made me strongly biased to think no ML-usable signal was there. Still, I explored and tried some ML models only to confirm that nothing seemed learnable there. But if I was right, maybe there was some viable tactic for the competition. The task was so unforgiving that any non-optimal prediction, including possibly most ML models, would converge to reduced performance so fast that maybe three months would be enough. 

A carefully **regularized naive approach**, fast, robust, and glitch-resistant, might well end up well positioned.

So, what is this **single naive optimal prediction** theoretically? 
- In-sample, the strongest prediction is the **rank-equivalent to a Kelly optimal portfolio** (Kelly-leverage using standardized ranks).  
- To use in the competition, out-of-sample, regularization of the covariance matrix was required, using as a reference validation sets of similar size to the private test set.
- Some additional details were: some basic preprocessing,  forward-fill NANs in prices before building targets and rank standardization per day. Also I excluded the 2020-2021 period in one of my submissions although the one that scored higher was the one with the full dataset (2018-2025).

All this yielded scores around 0.4 on late validation periods (with high variance, as expected). My final 0.479 is in line with that, although of course given the task and the small test window luck can’t be ruled out in any amount. 

Thanks to the host for putting together this interesting (if puzzling) competition, and to the Kaggle community for so many years of  shared knowledge and  insights.

Best regards!

