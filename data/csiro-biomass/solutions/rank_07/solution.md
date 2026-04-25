# 7th Place Solution - single/dual+TTT

- **Author:** Boredom
- **Date:** 2026-01-29T01:38:33.483Z
- **Topic ID:** 670654
- **URL:** https://www.kaggle.com/competitions/csiro-biomass/discussion/670654
---

Thank you to Kaggle and the sponsors for hosting such a wonderful and engaging competition. It was a heated contest, and I am thrilled that we managed to hold our position amidst the fierce competition. This has been an invaluable learning experience for me. Now, let me walk you through our approach:
- --

I joined the team midway through the competition and built my work on top of the excellent baseline (LB 0.67) established by my teammates. My teammates will provide a more detailed breakdown of the baseline operations later.

- --

My solution follows a two-stage pipeline: Stage 1 involves initial training, followed by Stage 2, which consists of online pseudo-labeling and Test-Time Training (TTT).

# **Stage1**
### **model：**
**Single Stream Version:**
@shanzhong8 DINOv3-Large + Single-stream:  Predicts all five components during both training and inference, using a regression head with Softplus activation. The first 80% of parameters were frozen during the training process. The overall framework follows a shared feature extractor + task-specific heads paradigm, enabling both parameter efficiency and inductive bias sharing across related regression tasks. Input resolution 1024×2048, accommodating ultra–high-resolution imagery. Single model 5-folds  without TTA and TTT can reach 0.76 public and 0.65 private LB. 

**Dual Stream Version:**
 DINOv3-Large + Dual-stream: During training, the model is supervised on all five components. However, at inference time, it only predicts Dry_Green_g, Dry_Dead_g, and Dry_Clover_g, while GDM_g and Dry_Total_g are derived through post-calculation (using Softplus-activated heads). The first 80% of parameters were frozen during the training process.

### **loss：**
r^2 loss

### **CV Strategy:**
StratifiedGroupKFold was used with 'sampling date' as groups and 'stat' as labels.  Since the original three dates were insufficient for a 5-fold split, I subdivided the WA data into seven virtual dates.  A multi-threaded brute-force search across 10 million seeds was conducted to identify the split with the most balanced distribution across all five folds.

### **Aug (Albumentations):**
Resize(single-stream: 1024*2048; dual-stream: 1024^2), HorizontalFlip, VerticalFlip, RandomBrightnessContrast, and ColorJitter.

### **Training Configuration:**
1.  Optimizer: AdamW with lr: 2e-4 and wd: 1e-2.
2.  For each fold, I maintained the top 3 best-loss checkpoints and top 3 best-score (CV) checkpoints.  These 6 models were then combined using SWA (Simple Weight Averaging) to enhance generalization.
3.  For dual-stream models, HFlip or VFlip was applied to the integrated image before splitting it into its respective streams.

### **Inference Configuration:**
1.  A simple average of the single-stream and dual-stream model results.
2.  Only horizontal flip TTA was utilized.

- --

# **Stage2(TTT)**
Due to the lack of model heterogeneity beyond DINOv3 for ensembling and the significant inference time headroom remaining, I decided to implement Test-Time Training (TTT). 

First, I performed an ensemble inference of single-stream and dual-stream models (with or without TTA). To optimize efficiency, I implemented dual-GPU parallel inference, where one GPU processed the 5-fold single-stream models while the other handled the 5-fold dual-stream models. This significantly accelerated the inference process. 

The initial predictions then underwent post-processing—for example, multiplying clover by 0.8 and zeroing out any component values below 0.2. These processed results were used as pseudo-labels to train the best single-stream fold (based on LB/CV) using the combined full training and test sets. This TTT stage lasted for 12 epochs, with SWA (Simple Weight Averaging) applied during the final 4 epochs to prevent overfitting and improve generalization. All other training configurations remained consistent with Stage 1. 

Additionally, I attempted parallel TTT for both single-stream and dual-stream models on separate GPUs, but this did not yield any noticeable performance gains. The final result was a weighted blend of the Stage 1 and TTT predictions: 0.2 * Stage 1 + 0.8 * Stage 2.


**What didn't work:**
1. Model architectures other than DINOv3, such as SigLIP-2.
2. Mixup, Cutmix, and their various modified variants.
3. Auxiliary tasks and auxiliary losses.
4. DINO embeddings combined with ML models.
5. External datasets.
And many other ideas that didn't provide any meaningful gain.

### **infer code：**
[lb 0.77 | 0.65](https://www.kaggle.com/code/cudacoding/speedup-ttt-fold2-w-o-pp-lb-0-77-0-65)

It was incredibly difficult to find ideas that provided a stable boost in this competition. Finally, thanks to everyone—it was a pleasure competing with you all!