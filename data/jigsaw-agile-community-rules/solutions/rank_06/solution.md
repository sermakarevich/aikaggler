# 6th Place Solution: Online Distillation via Deep Mutual Learning

- **Author:** ducnh279
- **Date:** 2025-10-24T14:16:34.047Z
- **Topic ID:** 613150
- **URL:** https://www.kaggle.com/competitions/jigsaw-agile-community-rules/discussion/613150
---

First, I’d like to thank the organizers for hosting such an amazing LLM competition. Huge thanks also to my teacher, @justmarkham - your online data science education has been invaluable, and I couldn’t have done this without you.

# Summary
For this competition, I fine-tuned 3 LLMs from the Qwen3 family for text classification using **Deep Mutual Learning (DML)**, and applied weighted ensembling to improve robustness and stability.

# Input Data
Test positive and negative examples, along with their rules, were used for training and formatted into the following prompt template:

```python
prompt = f"""<|im_start|>user
You are given a Reddit comment and a specific subreddit rule. Your task is to decide if the comment violates the rule. Respond only with "Yes" or "No".

Rule: {RULE_HERE}

Now, here is the comment to classify:
"{COMMENT_HERE}"

Answer "Yes" if it violates the rule, otherwise "No".<|im_end|>
<|im_start|>assistant
<think>

</think>

Answer:"""
```

# Models
- **Models for experimentation:** `Qwen3-14B` and `Qwen2.5-14B-Instruct`
- **Models for final submission:** `Qwen3-14B`, `Qwen3-8B`, and `Qwen3Guard-Gen-4B`

# Early Competitive Submission
Training single Qwen3-14B on test examples, quickly get 
| Model | Public Score | Private Score |
|-------|--------------|---------------|
| Qwen3-14B | 0.925 | 0.921 |


# Deep Mutual Learning

Paper: [Deep Mutual Learning (arXiv:1706.00384)](https://arxiv.org/abs/1706.00384)

With only 2 T4 GPUs, using a larger teacher model to improve the 14B model’s performance is not feasible. Offline distillation could be attempted, but it only covers 2 out of the 6 known rules in the training set. To achieve the best overall score, knowledge distillation needs to incorporate all rules.

Classic logits-based knowledge distillation poses a challenge in this competition because the teacher’s softmax outputs are extremely sharp, almost identical to one-hot labels. Applying temperature scaling to soften them is ineffective.

Under these constraints, a more practical approach is **Deep Mutual Learning (DML)**, an online distillation method where multiple peer student models of similar or smaller size learn from each other simultaneously. This enables effective knowledge transfer without a larger teacher, simply by adding a **Kullback-Leibler (KL) divergence loss** to the standard classification cross-entropy.

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F10714081%2Fa5804528660ba5def70f98e39c3e1f17%2Fdml.PNG?generation=1761313506467940&alt=media)

The advantage of DML is that it removes the need for a larger and stronger teacher by allowing models to learn from each other’s soft predictions rather than hard labels. This process enriches supervision, adds implicit regularization, and encourages smoother decision boundaries. Each peer model effectively gains extra capacity from the knowledge shared by the other, leading to more stable optimization and better generalization. This makes DML especially valuable in LLM competitions or modern industry environments, where hardware constraints often prevent training or deploying a large teacher model, yet effective knowledge transfer and robust performance remain essential.

# Effect of DML on Qwen3-14B & Qwen2.5-14B Performance

*Peer Students: Qwen3-14B, Qwen2.5-14B-Instruct*

| Training Type | Model | Public Score | Private Score |
|---------------|-------|--------------|---------------|
| Independent   | Qwen3-14B | 0.925 | 0.921 |
| Independent   | Qwen2.5-14B-Instruct | 0.923 | 0.919 |
| DML           | Qwen3-14B | **0.929** | **0.925** |
| DML           | Qwen2.5-14B-Instruct | **0.929** | **0.924** |

## Ensembles

| Ensemble Type | Public Score | Private Score |
|---------------|--------------|---------------|
| DML models | **0.932** | **0.9271** |
| Independently trained models | 0.929 | 0.925 |

DML consistently improves performance for both Qwen3-14B and Qwen2.5-14B, with notable gains in private scores and ensemble results, demonstrating its effectiveness in boosting model generalization.

# Final Submission
While DML adds extra capacity to each model, its co-training process can introduce correlations in the models’ errors. Since diversity is crucial for a strong ensemble,  2 days before the deadline, I decided to create a cohort of 3 models for DML. This approach enhances ensemble diversity and further helps mitigate the one-off mistakes of each “peer” model.

**Performance of the 3-peer ensemble**  

Peer Students:  `Qwen3-14B`, `Qwen3-8B`, `Qwen3Guard-Gen-4B`

| Public Score | Private Score |
|--------------|---------------|
| 0.93237      | 0.92781       |

![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F10714081%2F229abd026bb5a70d8b942325659ecc89%2Fselected_submission.PNG?generation=1761904642285377&alt=media)

# Model Execution Time
- **Training:** Three models were trained together in a single DML process, taking about 5 hours.  
- **Inference:** After training, each model was run sequentially to generate predictions for ensembling. The test data was split in half, keeping the distribution of sequence lengths similar in both halves to balance GPU workloads. Each half was then sorted by length for efficient batch processing and run in parallel across 2 GPUs. The entire inference process took around 5 hours.