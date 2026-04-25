# drawing-with-llms: cross-solution summary

This competition challenged participants to generate Scalable Vector Graphics (SVGs) from text prompts while strictly adhering to size constraints and maximizing a harmonic score of aesthetic quality and VQA fidelity. Winning approaches converged on hybrid pipelines that combine diffusion-based image generation, heuristic or differentiable SVG conversion, and clever OCR manipulation tricks to game the evaluation pipeline. Success heavily relied on proxy scoring, adaptive candidate selection, and precise compression techniques to balance visual fidelity with strict byte budgets.

## Competition flows
- Text prompts -> Flux.1-schnell -> vtracer -> SigLIP ranking -> differentiable SVG optimization -> compression/formatting
- Diffusion model -> 64 candidates -> compressed SVGs with hidden text/artifacts -> custom pipeline evaluation -> highest aesthetic selection
- Stable Diffusion -> SVGs -> dashed-line mask for OCR timing -> heuristic layout optimization -> aesthetic tweaks
- DRaFT-fine-tuned SD3.5M -> rough SVG conversion -> diffvg gradient descent -> VQA proxy & OCR check selection
- Prompts -> FLUX.1 Schnell bitmaps -> binary search & custom SVG compression -> dummy VQA evaluation -> linear regression ranking
- Gemini 2.5 Flash prompts -> Flow-GRPO fine-tuned SD3.5M -> vtracer SVG conversion -> PaliGemma filtering
- SDXL-turbo -> VTracer SVGs -> DiffVG aesthetic patch optimization -> VQA proxy selection
- Diffusion (PixART/Sana) -> optimized SVGs (relative paths, color grouping) -> shape/color VQA evaluation -> time-budgeted inference
- Fine-tuned SDXL -> 14-18 candidates -> VTracer/scour SVGs -> corner '+' OCR anchor -> VQA proxy & MedianBlur filtering -> seed-based ensembling

## Data processing
- Prompt engineering with multiple templates & synthetic data generation (GPT-4o, Gemini 2.5 Flash)
- Differentiable ImageProcessorTorch simulating evaluation transforms (crop, resize, JPEG, filters)
- Bitmap downsampling & 6-color quantization
- Contour extraction & polygon simplification (OpenCV)
- SVG path compression (absolute to relative coordinates, transform removal, color grouping, command merging)
- Binary search for optimal bitmap dimensions within size constraints (10KB/6KB)
- OCR mitigation/decoys (SVG path injection, hidden text patterns, dashed-line masks, white/black decoys, corner '+')
- Aesthetic patch optimization & placement balancing
- MedianBlur application for VQA runtime reduction
- Inference filtering via PaliGemma with dummy questions
- Negative prompts (`no text`, `no other colors`)

## Features engineering
- Linear regression weights mapping dummy VQA scores to predicted final VQA scores

## Models
- Flux.1-schnell
- VTracer
- DiffVG (pydiffvg)
- SigLIP-so400m
- SigLIP2-base
- SSD-1B
- Tiny AutoEncoder (XL)
- Stable Diffusion
- SDXL
- SDXL-turbo
- Stable Diffusion 3.5 Medium
- DRaFT
- DRaFT-LV
- HPSv2
- PickScore
- Flow-GRPO
- PaliGemma-10B
- Gemini 2.5 Flash
- VQA models (10B-448, 3B-224, 3B-448)
- primitive
- ReNO
- ImageReward
- PixART
- Sana
- scour
- kagglehub
- Linear Regression

## Loss functions
- Negative aesthetic predictor score
- Negative cosine similarity (SigLIP embedding)
- MSE loss
- L1 distance (rendered SVG vs original image)
- Negative LPIPs distance
- KL divergence (4e-4)
- Harmonic mean of VQA and aesthetic scores (reward function)

## CV strategies
- Holdout validation dataset
- Local validation with high-variance mitigation via seed-based ensembling

## Ensembling
- Select top 2 candidates based on SigLIP similarity, pre-optimize, choose highest aesthetic score for final optimization
- No traditional ensembling; generates 64 candidates, evaluates through custom pipeline, selects single highest-scoring candidate
- Generates 5-6 SVG candidates per prompt within adaptive time budget, selects based on highest VQA score and OCR validation
- Generates 10-12 candidate images per prompt and selects top-scoring output based on linear regression-predicted VQA scores
- Generates three images per prompt during inference, filters using PaliGemma with a dummy question, selects final SVG
- Ensembles multiple generated images and prompts, selects best SVG from ~28 candidates using proxy VQA function
- Ensembles two high-scoring runs differing only in random seeds to mitigate validation and leaderboard variance

## Notable individual insights
- rank 3 (3rd place solution: VQA/AES=0.81/0.64 Diffusion model + differentiable SVG optimization): Implementing a differentiable ImageProcessorTorch that simulates the competition's evaluation pipeline allows gradients to flow correctly to SVG parameters, aligning optimization directly with the scoring metric.
- rank 1 (1st Place Solution): Embedding the specific three-letter string "ZOK" in the top-left corner avoids OCR hallucinations while maximizing aesthetic scores.
- rank 2 (2nd Place Solution): The timing difference between OCR penalty application (pre-enhancement) and VQA score calculation (post-enhancement) is the critical lever for manipulating the scoring system.
- rank 4 (4th Place Solution: SD3.5M + DRaFT + diffvg): Fine-tuning with DRaFT using a mix of general preference rewards and competition-specific rewards can eliminate the need for manual prompt engineering.
- rank 5 (5th place solution - VTracer and DiffVG optimization): Proxy scoring using the VQA model's "Yes/No" probability is more robust than generating custom questions.
- rank 8 (8th Place Solution): Local validation scores showed high variance and often correlated poorly with the public LB, making seed-based ensembling necessary.

## Solutions indexed
- #1 [[solutions/rank_01/solution|1st Place Solution]]
- #2 [[solutions/rank_02/solution|2nd Place Solution]]
- #3 [[solutions/rank_03/solution|3rd place solution: VQA/AES=0.81/0.64 Diffusion model + differentiable SVG optimization]]
- #4 [[solutions/rank_04/solution|4th Place Solution: SD3.5M + DRaFT + diffvg]]
- #5 [[solutions/rank_05/solution|5th place solution - VTracer and DiffVG optimization]]
- #8 [[solutions/rank_08/solution|8th Place Solution]]
- #12 [[solutions/rank_12/solution|12th Place Solution: SD3.5M + GRPO]]
- #13 [[solutions/rank_13/solution|13th Place Solution - A Kaggle beginner's attempt]]
- #19 [[solutions/rank_19/solution|19th place solution]]

## GitHub links
- [BachiLi/diffvg](https://github.com/BachiLi/diffvg) _(library)_ — from [[solutions/rank_04/solution|4th Place Solution: SD3.5M + DRaFT + diffvg]]
- [clarkkev/svg-diffusion](https://github.com/clarkkev/svg-diffusion) _(solution)_ — from [[solutions/rank_04/solution|4th Place Solution: SD3.5M + DRaFT + diffvg]]
- [tgxs002/HPSv2](https://github.com/tgxs002/HPSv2) _(library)_ — from [[solutions/rank_04/solution|4th Place Solution: SD3.5M + DRaFT + diffvg]]
- [yuvalkirstain/PickScore](https://github.com/yuvalkirstain/PickScore) _(library)_ — from [[solutions/rank_04/solution|4th Place Solution: SD3.5M + DRaFT + diffvg]]
- [richzhang/PerceptualSimilarity](https://github.com/richzhang/PerceptualSimilarity) _(library)_ — from [[solutions/rank_04/solution|4th Place Solution: SD3.5M + DRaFT + diffvg]]
- [mit-han-lab/nunchaku](https://github.com/mit-han-lab/nunchaku) _(library)_ — from [[solutions/rank_13/solution|13th Place Solution - A Kaggle beginner's attempt]]
- [yanyanhuang/12th-Place-Solution-for-Kaggle-Drawing-with-LLMs](https://github.com/yanyanhuang/12th-Place-Solution-for-Kaggle-Drawing-with-LLMs) _(solution)_ — from [[solutions/rank_12/solution|12th Place Solution: SD3.5M + GRPO]]
- [matheuspf/draw](https://github.com/matheuspf/draw) _(solution)_ — from [[solutions/rank_05/solution|5th place solution - VTracer and DiffVG optimization]]
- [THUDM/ImageReward](https://github.com/THUDM/ImageReward) _(reference)_ — from [[solutions/rank_05/solution|5th place solution - VTracer and DiffVG optimization]]
- [fogleman/primitive](https://github.com/fogleman/primitive) _(reference)_ — from [[solutions/rank_05/solution|5th place solution - VTracer and DiffVG optimization]]

## Papers cited
- [DRaFT](https://arxiv.org/abs/2309.17400)
- [Flow-GRPO: Training Flow Matching Models via Online RL](https://arxiv.org/abs/2505.05470)
