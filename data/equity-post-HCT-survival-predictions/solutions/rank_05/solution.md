# Mathing it to 5th place!

- **Author:** Robert Hatch
- **Date:** 2025-03-06T01:06:09.310Z
- **Topic ID:** 566541
- **URL:** https://www.kaggle.com/competitions/equity-post-HCT-survival-predictions/discussion/566541
---

I'm in shock. I don't have a write up prepared like one team did, lol. But I can tell you the one and only big secret to my success:

What if we didn't predict one target, and didn't even predict a range of diverse targets and blend them? (I did this too, mind you, no harm in doing more). What if we simply predicted EFS with one ENSEMBLE of models, and *only* for efs==1, trained another ENSEMBLE of models to rank ONLY those guys?

Well, first off, what about outliers? Another trick is to realize there aren't THAT many outliers. If you take the one and only efs_time cutoff point that MINIMIZES the sum of efs==0 below cutoff point plus the sum of efs==1 above cutoff point, it's only about ~460 rows. That's only 2%. So just for reference, and for better or worse, I actually trained on label 'unsafe', which is 1 if efs_time < 13.326, 0 if efs_time > 13.326.
Anyways, back to ensembling A (is it efs==1) and ensembling B (rank ASSUMING efs==1)... well that wouldn't blend well... unless you math it! The simplified formula is:

p = (a * b) - ((1 - a) * (S_RATIO))

In words:
* The more the row is predicted to be unsafe, the more emphasis is put on Ranking. Ranking is also treated as a probability.
* The more the row is predicted to be SAFE, the more emphasis is put on a flat "you lose to everyone else" value.  (S_RATIO stands for 'safe_ratio'). It's trivial to realize that (negative) 0.5 is approximately correct for that value. However, the real math is first looking at exact wins losses of unsafe vs exact losses of safe, ADDING a value to the entire distribution to 'center' the b group (unsafe) to be from 0 to N, then divide everything by N. You get around 0.425, which is what I used after awhile since it increased my CV score a little.

Notes:
- Ensemble raw logits not probabilities for 'a'
- Train to predict the logits of the 'b' group with a min max margin. So I evenly arranged them from 0.03 to 0.97, then took the logit. This further emphasizes the extremes with RMSE. It also might(?) play better with the formula. In that you want a and b to be balanced relative to each other. Frankly if one person is more likely to be safe, I know what that means exactly if comparing head-to-head vs another person. 0.3 vs 0.4 is 0.1 delta. But it's harder to know if 0.3 predicted 'rank' vs 0.4 predicted 'rank' is about equally trustworthy, do they win head to head about 0.1? More? Less?

Then the post-processing trick(s), just on this math formula above:
* One is the obvious 'better math' variation of the known public "add 0.2 to efs==1". If you instead add +X logit to positive logit As, and -X logit to negative logit As, you are again emphasizing: hey, I think my model usually gets it right, let's emphasize that a little. But since it's logits, it emphasizes in a more balanced way.
* though it started as a bug with a surprising CV result, there's PROBABLY(?) a kernel of statistical logic to the bigger win in the same vein: multiplying the raw logit scores of BOTH a and b before appliing the formula. By a LARGE value, though I backed off a little when it worked on CV but not LB, figuring I'd rather be conservative. Still multiplying by 1.5 before converting the logit to probability. The reason I went for this trick, other than the CV score (and running out of time to try different ways to optimize LOCALITY ranking), is the following:

LOCALITY
The math formula, and predicting mean values with a lot of models, are great for a GLOBAL ranking of 28800 rows. However, consider this, the bottom 100 are already LOCKED in compared with the upper 28700. So it no longer matters that #100 would score 0.1 vs those 28700, and #1 would only score 0.02. The question is: how will the bottom 100 do vs each other? Is there a zoomed in rescrambling that would score better? My hypothesis as I ran out of time for many things, including GNN, head-to-head, locality optimization... my hypothesis is that by multipling the LOGIT values of a and b, you are essentially helping the final score "zoom in" on the ends, and zoom in more so the closer to the extreme you are, so it smooths it out nicely by itself!

Similarly, the add 0.2 trick probably helps 'zoom in' on the middle, and with my add/subtract logit trick, again it actually decreases in impact the further from the middle(!).

So there you have it, besides a lot of targets, a couple hand-tuned models, a few public notebooks, and about 900(!) zeroshot auto-gluon models finally integrated at the last possible moment (the 'a' and 'b' targets times 150 models, 'a' with target encoding times 150, and 4 more normal targets times ~110 - no NNs)), that's the main story of how I mathed it to 5th place!

Oh, and almost no FE! Maybe that helped or hurt, who knows. I dropped a few hla sum columns (_6, _8) on NNs, and for GBDT and AG I used the recalc HLA trick but (very important I think!) I didn't replace the magic hla_nlmp6 or whatever it was called. That was a magic column that was more important than other hla sums and impossible to recreate from the raw HLA data. ESPECIALLY since synthetic data, I thought better to preserve and just add another column for my GBDT models that can't really overfit noisy columns. All other sums I replaced.

Besides that at some point I rounded the two numeric float age columns. donor_age which sounds super unimportant I rounded to every other year (0, 2, 4, etc). age_at_hct maybe to first decimal digit. Which still sounds TMI, but strikes a balance between letting the model learn synthetic and or real patterns a human couldn't detect with high fidelity vs learning fake patterns that a person born 3 days earlier or later couldn't possibly affect that person's outcome. It marginally improved CV, so I kept the rounding for new models I built from then on.