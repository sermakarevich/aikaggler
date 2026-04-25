# ✨ MAP 6th Place Solution - Qwen-semble FTW! 🚀

- **Author:** Manan Jhaveri
- **Date:** 2025-10-16T19:05:50.347Z
- **Topic ID:** 612099
- **URL:** https://www.kaggle.com/competitions/map-charting-student-math-misunderstandings/discussion/612099

**GitHub links found:**
- https://github.com/mananjhaveri/MAP-Charting-Student-Math-Misunderstandings-6th-place-solution

---

Thanks to the organizers and Kaggle for this fun and challenging competition. Also thanks to the participants for all the valuable notebooks and discussion posts.

## Highlights of my solution (TL;DR version)
* I used a mix of `Qwen3-Embedding-8B`, `Qwen3 14B` and `Qwen2.5 14B`.
* Synthetic data for rare labels, Augmented text and Pseudo-labelling duplicates helped to improve the score.

<br>

## Detailed solution

### Problem statement
* We are provided with math questions, answer given by students and the explanations given by the students.
* We need to predict 3 things:
    * Is the answer correct or not
    * Is the explanation correct, has a misconception or neither
    * What is the type of misconception, if present

### Data and pre-processing

* From the start, I saw this problem as just a misconception classification problem. So I removed the `True_` and `False_` prefixes from the categories and used the resulting 37 labels during fine-tuning.
* Used the following input format:
```Python
correctness = "Yes" if row["is_correct"] else "No"
input_text = (
    f"Question: {row['QuestionText']}\n"
    f"Answer: {row['MC_Answer']}\n"
    f"Correct: {correctness}\n"
    f"Explanation: {row['StudentExplanation']}\n"
    f"Task: Classify the misconception in the explanation."
)
```
In the initial experiments, I noted that adding the last line of the prompt was improving CV and LB scores for the LLMs. Hence, decided to keep it for the rest of the experiments.
* I dropped the duplicate samples altogether. Duplicates = same answer and explanation but different misconception as label.
* In the last few experiments, I pseudo-labelled these duplicates and added them back to the training data.
* In some experiments, I added ~50 augmented samples per category. Augmentation involved basic replacement of punctuations, conversion of words to number or vice versa, replacing "plus" to "+", replacing "equals to" to "equals" or "=", adding small typos, etc.
    * I had analyzed near duplicates found through fuzzy search in the training data and accordingly tried to create augmented samples.
* In the final few experiments, I added synthetic data for 10/37 categories with the lowest support. I used ChatGPT 4o to generate 100-120 samples per category.
* I used a max sequence length of 160 with dynamic padding. This made the training and inference faster for me.
* For local testing, I have used 20% test split and used full data for re-training the model for the submissions.


### Models

* I started my experiments with a variety of models like GTE, Ettin and later moved to LLMs of up to 8B parameters. I observed that the `sequence classification` approach with `Qwen3-Embedding-8B` performed the best for me.
* After freezing my prompt template and pre-processing, I moved to 14B param models too.
* I trained these LLMs with 4-bit Q-LoRA. Rank = 128, alpha = 32-128 (as per different experiments). 
* I used A5000/A100 on Jarvis Labs for compute.


Here's the summary of the models that I finally used in my qwen-semble:

| Model | Data used | Public LB |
| --- | --- |  --- |
| Qwen3 Embedding 8B | Deduplicated data | 0.946 |
| Qwen3 14B | Deduplicated data | 0.947 |
| Qwen2.5 14B IT | Deduplicated data | 0.947 |
| Qwen3 Embedding 8B | Deduplicated data + Augmented data | 0.945 |
| Qwen3 Embedding 8B | Deduplicated data + Pseudo-labelled duplicates + Synthetic data | **0.949** (best single model) |
| Qwen3 14B | Deduplicated data + Pseudo-labelled duplicates + Synthetic data | 0.948 |
| Qwen3 14B | Deduplicated data + Pseudo-labelled duplicates + Synthetic data (+ different seed than above)  | 0.943 |


### Ensembling
* I used the ensembling method from this notebook by @kishanvavdara : https://www.kaggle.com/code/kishanvavdara/ensemble-gemma-qwen-deepseek
* The ensemble of the first 4 models from the above table helped me crack 0.95 on the public LB.
    * I used this ensemble to pseudo-label the duplicates.
    * The model with augmented data has a poor individual score compared to the rest but it contributed significantly to the ensemble. I tried removing it multiple times as I got better models but the score always dipped.
* With the ensemble of the first 6 models, I got 0.951 on the public LB (and 0.947 private LB).
* In the last 2 days, I was out of ideas and I didn't want to make small tweaks to model weightage in the ensemble method as it felt unreliable. So I decided to fine-tune Qwen3 14B with the same config as the one I used earlier in the 0.948 model, but with a different seed and random state. Its individual score was quite poor but ensembling it with the other 6 models gave me my best public as well as private LB score.


### Failed experiments
* Multiple pre-trained models.
* Several prompt templates.
* Adding too many augmented samples. I had shortlisted multiple patterns from my analysis and created ~10k augmented samples. But the public LB score remained ~0.945 and these models didn't add much to the ensemble.
* Different modelling approaches:
   * Using all original 65 labels.
   * I tried a 2-stage approach where the first stage model just predicts whether a student explanation is correct, has a misconception or neither of the two. And the second stage would predict the misconception type.
   * Text Generation approach where I provided simplified descriptions of each category (that a question can possibly have) in the prompt and asked the model to predict which one best describes the reasoning behind the student's explanation.
   * Textual Entailment approach where I tried to predict whether a given sample entails any of the simplified descriptions of the misconception categories.
   * MoE-style NN for the classification head instead of LoRA adapters. (really wanted this to work but couldn't give enough time to fix the hyperparams)
* Different loss functions like focal-loss