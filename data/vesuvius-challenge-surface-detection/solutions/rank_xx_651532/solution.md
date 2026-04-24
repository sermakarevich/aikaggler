# [placeholder] my solution : it is connecting the dots!

- **Author:** hengck23
- **Date:** 2025-12-04T02:43:16.447Z
- **Topic ID:** 651532
- **URL:** https://www.kaggle.com/competitions/vesuvius-challenge-surface-detection/discussion/651532

**GitHub links found:**
- https://github.com/ryanchesler/ants
- https://github.com/pmh47/spiral-fitting
- https://github.com/ScrollPrize/villa
- https://github.com/HuXiaoling/TopoLoss
- https://github.com/nstucki/Betti-Matching-3D
- https://github.com/HuXiaoling/awesome-topology-driven-image-analysis
- https://github.com/MIC-DKFZ/Skeleton-Recall

---

##Disclaimer: this is a work in progress, contents subject to changes

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F113660%2F5ea0dcb437185526a48b879f5ff4b24d%2FSelection_1614.png?generation=1765227069400375&alt=media)

it is connecting the dots!

The boundary clues are so strong that you can do multi-class pixel label, where class1 = first curve on top, class2 = the next one ,...
you no longer perform semantic segmentation and jump to instance segmentation at the start. This solves all scroll-touching issues ...
(but you do need to take good care of unlaballed region)

---

I just enter t the competition and did some early investigation. To win:

1) **Correct loss or post-processing** to improve **topology score** (most important), surface dice, voi score.  
Many kagglers will treat this as a  volume segmentation task ... that is wrong. Good volume IOU doesn't guarantee  lb score. The task is actually "scroll object" detection". We want continous surface object that is no holes, not disintegrated or wrongly joined ("stuck together")

2) **Use of unlabelled data**

What i would do next:
- visualisation of topology score results. What is Betti matching? What connected components are matched or designated as FN or FP?
- how to improve topology score? (e.g. post processing of hole filling, separating stuck scrolls or joining disintegrated ones)

---

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F113660%2Fe148bb135b9b439353d6edaa4a8f5a3a%2FSelection_1399.png?generation=1764816110638340&alt=media)

Note that the ground truth focuses on surface (and **one side of the scroll**) and some ground truth **eats into the "air"**. Hence this is weak supervision.

summarising everything (it is a huge mess) in this post:
- Level 1 (The Hack): CED / Directional Blur. (Fixes gaps using image processing).    
- Level 2 (The Engineering): Tracking / Z-Sweep. (Fixes gaps using time-consistency).  
- Level 3 (The Math): Parametric / Vector Fitting. (Fixes gaps by mathematically forbidding them).  

