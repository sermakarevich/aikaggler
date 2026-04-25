# 12th Place Solution: SD3.5M + GRPO

- **Author:** Rock
- **Date:** 2025-05-28T03:11:15.570Z
- **Topic ID:** 581034
- **URL:** https://www.kaggle.com/competitions/drawing-with-llms/discussion/581034

**GitHub links found:**
- https://github.com/yanyanhuang/12th-Place-Solution-for-Kaggle-Drawing-with-LLMs

---

I would like to extend my sincere thanks to the organizers for their hard work and dedication in making this competition both engaging and meaningful. Below is my solution to this competition:

The [training code](https://github.com/yanyanhuang/12th-Place-Solution-for-Kaggle-Drawing-with-LLMs) has been uploaded to GitHub, enjoy it! 😉

Unlike the approaches of many other teams, my method focused on directly fine-tuning Stable Diffusion. The goal was to generate images that were not only suitable for SVG conversion but also aligned with the preferences of the VQA evaluator. Given that the competition's evaluation metric was determined by PaliGemma and an aesthetic score, traditional preference-based methods like DPO (Direct Preference Optimization) were less suitable. During my research, I discovered the recently open-sourced Flow-GRPO [1], which perfectly met my needs. In essence, FlowGRPO employs a reinforcement learning strategy (GRPO - Group Relative Policy Optimization) to train Stable Diffusion by directly aiming to maximize the specified evaluation metric.

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F9820727%2Fbe375e495916752810cbfb2a74503637%2FPixPin_2025-05-28_10-51-39.png?generation=1748400722191201&alt=media)

## Overall Pipeline:

**Training Data**:

I utilized Gemini 2.5 Flash to generate 4,000 training samples and 200 test samples. Each sample included a description along with corresponding questions and answers.

**Image Generation**:

Due to the significant VRAM requirements for GRPO training, standard GPUs often struggle with larger models like Flux. Therefore, I opted to directly train Stable Diffusion 3.5 Medium.

**Image to SVG**:

I utilized vtracer to transforme the generated image to SVG, with other post-processing methods, for example, merging identical commands in SVG, converting from absolute to relative coordinates, and combining multiple closely spaced short line segments into a single longer line segment.

**Reward Function**:

For more direct optimization, I employed the competition metric itself as the reward function: the harmonic mean of the VQA score and the aesthetic score.

**Training Process**:

I primarily utilized four H800 GPUs for training. During this process, Stable Diffusion generated raw images at a resolution of 512x512. Each GPU produced 8 images per step, and for each input description, 16 images were generated. To accelerate training and mitigate model degradation, I set the KL divergence weight to 4e-4.

**Training Results**:

The initial performance of SD3.5M was quite low; in my preliminary tests, it achieved a public leaderboard score of approximately 0.65 and a score of around 0.72 on my local test set. After 500 training steps with FlowGRPO, the score on my local test set peaked at approximately 0.77. At this point, the private leaderboard score was around 0.73 (unfortunately, I missed submitting at those optimal checkpoint).

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F9820727%2Ff61eb05f6b8f80f64533db1ad9ceb032%2FPixPin_2025-05-28_11-09-18.png?generation=1748401777155864&alt=media)

**An Unexpected Observation**:

Interestingly, while my primary goal with GRPO was to enhance the aesthetic scores of the generated images, I found that the VQA score saw a more significant improvement. For instance, on my local validation set, the VQA score increased from 0.82 to 0.91. This suggests that GRPO training effectively guided Stable Diffusion to produce images more aligned with PaliGemma's preferences, leading to a substantial boost in VQA performance.

**Inference**:

During inference, each description generates three images and corresponding SVGs, then a simple dummy question (f"Does the image contains '{prompt}'?") is used in combination with paligemma10b to perform basic filtering, resulting in the final SVG output.


[1] Liu J, Liu G, Liang J, Li Y, Liu J, Wang X, Wan P, Zhang D, Ouyang W. Flow-GRPO: Training Flow Matching Models via Online RL. arXiv preprint arXiv:2505.05470. 2025 May 8.