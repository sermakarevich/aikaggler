# 🔥🔥🔥Some tricks for training LLM Recall Model: CV 0.490, LB 0.352

- **Author:** sayoulala
- **Date:** 2024-10-31T02:04:26.170Z
- **Topic ID:** 543519
- **URL:** https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/discussion/543519

**GitHub links found:**
- https://github.com/BlackPearl-Lab/KddCup-2024-OAG-Challenge-1st-Solutions

---

Hello everyone,

I participated in this competition a week ago, and I have now moved up to the front ranks.

Based on my current understanding and experimental results, this is a typical task involving recall and ranking with LLMs. 

Over the past week, I found that although the data is relatively small, the CV and LB  are relatively stable when using large models as a baseline. I simply divided the CV by question ID, and here are my experimental results. Interestingly, the increase in LB score is particularly significant. (It is difficult to determine the confidence in the current results.)

| CV   | CV Recall @25 | LB   |
|------|---------------|------|
| 0.490| 0.882         | 0.353|
| 0.538| 0.910         | 0.456|
| 0.540| 0.922         | 0.466|
| 0.551| 0.928         | 0.478|



Here, I will share some experiences on how to perform recall using large models.

On the current [mteb](https://huggingface.co/spaces/mteb/leaderboard leaderboard, there are many leading LLM vector models. Most open-source models have corresponding technical reports, which might serve as useful references.

From my experience, there are three key points to training an excellent LLM vector model for competitions:

- Hard Negative Mining
- Larger Batch Size
- Adjusting Training Parameters for LLMs


In the KDD Cup 2024 AQA track, I won the championship using an iterative hard negative mining method. The corresponding training code is available [here](https://github.com/BlackPearl-Lab/KddCup-2024-OAG-Challenge-1st-Solutions/tree/main/AQA), and the paper can be found[ here ](https://openreview.net/pdf?id=vhLAb1dpIw) for reference.

For this task, I have open-sourced an inference script for an LLM 7b model [here](https://www.kaggle.com/code/sayoulala/use-llm-embedding-recall-infer), with an inference time of less than half an hour (training code can be referenced [here](https://github.com/BlackPearl-Lab/KddCup-2024-OAG-Challenge-1st-Solutions/blob/main/AQA/train/simcse_deepspeed_mistrial_qlora_argu.py), key parameters: batch size=64, hard negative examples=100).

Good luck to everyone ！😀

Looking forward to everyone's CV and experience sharing.