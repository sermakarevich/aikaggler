# 9th Place Solution: Bringing It All Together

- **Author:** lllleeeo
- **Date:** 2025-10-24T20:34:56.100Z
- **Topic ID:** 613185
- **URL:** https://www.kaggle.com/competitions/jigsaw-agile-community-rules/discussion/613185
---

First, we’d like to thank the organizers for hosting such an amazing LLM competition. Secondly this solution is primarily built upon five excellent public notebooks. Many thanks to the authors for sharing these ideas. Links are referenced at the bottom.

# Overview
The solution consists of three main components: Generative Models, Classification Models, and Faiss-based Retrieval.

---

# Part 1: Generative Models
This component is based on the generative test-time training approach shared by @wasupandceacar. We made the following improvements to this public notebook:

- Replaced Llama 3.2 3B with Qwen3-4B-Instruct-2507, using only 5% of the test data, improving the public LB score from 0.916 to 0.919
- Optimized test-time training with Unsloth, reducing submission time from 7.5 hours (dual GPU) to 2.5 hours (single GPU) and improving the public LB score from 0.919 to 0.920
- Removed subreddit information as it introduced pure noise when training with examples in testset, reducing submission time from 2.5 hours to 2.1 hours and improving the public LB score from 0.920 to 0.922

**Exploration of Alternative Training Data:**

When we attempted to adjust test data used, we found that generative models were extremely sensitive to data volume, likely due to repeated examples and noisy data. To avoid overfitting to the public LB, we explored more stable approaches (which turned out to be suboptimal after the private LB was revealed).

Since the original training data construction could not ensure all examples were trained as body text, we used all deduplicated examples and randomly matched positive/negative samples for each example to increase diversity.

Ensembling these newly trained models with the original models yielded:

| Model | Runtime (hours) | LB | PB |
|:------|:----------------:|:----------:|:----------:|
| **Qwen3-4B-Instruct-2507** | 3.0 | 0.924 |  0.918 |
| **Llama 3.2 3B** | 2.5 | 0.923 | 0.916 |
| **Phi-3.5-Mini-Instruct** | 3.5 | 0.918 | 0.913 | 

**\*Ensembling these three ensemble models achieved LB 0.930 (PB 0.924) with a runtime of 9 hours.**

---

# Part 2: DeBERTa-based Classification Models
This component builds upon the DeBERTa classification approach shared by Team U DONNO WHO and DLH. We implemented several optimizations:

**Model Architecture & Training:**

- Applied 2x learning rate to the classification head for faster convergence
- Implemented mean pooling instead of CLS token pooling for better representation
- Used discriminative learning rates across different layers (LLDR)

**Model Performance:**

| Model             | Runtime (hours) | LB     | PB     |
|:------------------|:----------------:|:------:|:------:|
| **DeBERTa-v3-base** | 0.5 | 0.909 | 0.902 |
| **E5-base**          | 0.5 | 0.910 | 0.904 |
| **MPNet-base**       | 0.5 | 0.906 | 0.900 |


**\*The final ensemble of these three models achieved LB 0.921 (PB 0.913) with a runtime of 1.5 hours.**

---

# Part 3: BGE Model with Triplet Loss and Retrieval
This component is based on the triplet loss approach for training BGE models shared by Aurora Rabbit and Ertuğrul Demir.
Our Improvements:

Ensembled single centroid and UMAP clustering methods at the inference stage

**\*The final model achieved LB 0.909 (PB 0.904) with a runtime of 1.5 hours.**

---

# Final Submissions
We submitted two different ensemble strategies for final evaluation.

For each part, models were first ensembled internally, followed by rank normalization within each part, and then combined using weighted averaging.

The final ensemble weights were set to **0.7 × Part 1 + 0.25 × Part 2 + 0.05 × Part 3**, based on LB performance.

**Submission 1:**

- Trained four generative models in parallel using dual GPUs(T4 x 2): Qwen3-4B-Instruct-2507, Llama 3.2 3B, Phi-3.5-Mini-Instruct and Qwen2.5-7B-Instruct
- Each model used different random seeds
- Ensembled these generative models with the model in Part 2 & Part 3
- Runtime: 9 hours
- Public LB: 0.933 | Private LB: 0.927

**Submission 2:**

- Used three generative models: Qwen3-4B-Instruct-2507, Llama 3.2 3B and Phi-3.5-Mini-Instruct
- Each model was trained on two different training data configurations and self-ensembled(six models in total)
- Ensembled these generative models with the model in Part 2 & Part 3
- Each model used different random seeds
- Runtime: 11 hours
- Public LB: 0.933 | Private LB: 0.926

---

# What Didn't Work

- Two-stage training
- Semantic matching for positive/negative examples
- Rule-specific prompt
- Label smoothing
- Logit-based ensembling
- Layer freezing
- SENet fusion of CLS, mean pooling and max pooling
- Rule-based ensemble weighting

# References
https://www.kaggle.com/code/wasupandceacar/jigsaw-pseudo-training-llama-3-2-3b-instruct
https://www.kaggle.com/code/nghiahoangtrung/jigsaw-speed-run-10-min-triplet-and-faiss
https://www.kaggle.com/code/nahidhossainredom/deberta-v3-base-3-epochs-lb-0-906
https://www.kaggle.com/code/daylighth/acrc-deberta-small-2epochs-1hr
https://www.kaggle.com/code/datafan07/jigsaw-speed-run-10-min-triplet-and-faiss