# title": "part of 4th place solution: GAPNet & dual loss ResNet

- **Author:** Dieter
- **Date:** 2019-01-11T08:45:10.833Z
- **Topic ID:** 77300
- **URL:** https://www.kaggle.com/competitions/human-protein-atlas-image-classification/discussion/77300
---

First a big thanks to all my team-members. They all made this competition an awesome experience. I also want to thank @brian, @heng and @lafoss for their fruitful input. 

Since major ingredient of our solution was diversity and in the end we used around 25 different models, I want to give a separate few notes on two of my main contributions. 

I want to split up my notes into strategy, architecture as I see each equally important.

**Strategy:**

@tunguz @sasrdw and myself teamed-up quite early which enabled us to work in different directions right from the beginning. We always had diversity of our models in mind. So I concentrated on models that seem a bit different. Diversity to my team-members was also the main reason I sticked to keras, although in my opinion pytorch would have been more suitable for this competition due to its flexibility. We also used different cross validation schemes for the sake of diversity.

After some trouble in the beginning for getting the cross-validation right, I started exploring different architectures as posted by @hengck23 . I found it quite efficient to only use 256x256 RGB images in the beginning because it allows for high iteration of different ideas.

**Architectures:**

*GAPNet*

Immediately,  reading the GAPNet paper, I had the idea to change the illustrated architecture to use a pretrained backbone instead. 

![original GAPNet][1]

I think one important advantage of the GAPNet architecture is its ability for multiscale. So I tried different backbones and ended up with ResNet18, which also enabled to use a batchsize of 32 on a GTX1080Ti. I also saw minor improvements adding SE-Blocks before the Average Pooling layers with nearly no computational cost, so I added those. I saw no improvement in using RGBY images. I used a weighted bce and f1 loss and a cosine annealing lr schedule and only trained for 20 epochs. After applying our thresholding method to the predictions GAPNet trained on 512x512 RGB images also using the HPA external data a single 5-fold model was able to achieve 0.602 Public LB. I also experimented with different internal/external data proportions, RGBY and 512cropping from 1024 images so I had 4 5-fold GAPNet models which I could ensemble resulted in LB  0.609

*Dual Loss ResNet*

Following another post from @hengck23 I implemented a ResNet34 with a dual loss:

![enter image description here][2]

Additionally to the "normal" classification loss I used the output of the last 32x32x128 layer within ResNet34 did Conv2D to 32x32x28 and then used a downsampling of the green channel with the according labels as ground truth mask to have a segmentation loss. This segmentation loss works like a regularizer that ensures that that activations of the 32x32x128 layer are "nice".
The additional supervised attention added quite some benefit to regularization, and the computational cost was bearable. I trained 2 variants 5-fold each scoring LB 0.603 and added them to the GAPNet ensemble -&gt; LB 0.618

I guess @tunguz will write an overall summary where he explains how my models were then incorporated into our overall ensemble.

  [1]: https://storage.googleapis.com/kaggle-forum-message-attachments/inbox/113660/a95f150c7153a17538b074def2255e21/GAP.png
  [2]: https://storage.googleapis.com/kaggle-forum-message-attachments/415080/10599/attention%20is%20what%20you%20needq.png