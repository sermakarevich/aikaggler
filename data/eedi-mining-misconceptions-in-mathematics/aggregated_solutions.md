# eedi-mining-misconceptions-in-mathematics: cross-solution summary

This competition focused on diagnosing mathematical misconceptions from multiple-choice questions, with winning approaches heavily relying on staged retrieval and reranking pipelines powered by large language models. Success was driven by generating high-quality synthetic data to address unseen misconceptions, fine-tuning tiered Qwen2.5 models with distilled reasoning, and implementing robust validation strategies that align with private test distributions. Top solutions consistently combined iterative negative mining, quantization-aware calibration, and sophisticated ensembling techniques to maximize recall and generalization.

## Competition flows
- Staged retrieve-and-rerank pipeline with tiered Qwen2.5 models (14B/32B/72B) and distilled CoT reasoning
- LLM-based recall model training with hard negative mining, QLoRA, and question-ID validation splits
- Cascading reranker pipeline with dynamic similarity thresholds, pointwise/listwise ranking, and prefix caching
- Multi-stage pipeline combining synthetic data generation, biencoder retrieval, and single-token logit reranking
- Two-stage retrieval and reranking with probability scaling for unseen misconceptions and input shuffling
- Voting ensemble of three independent LLM-based retrieval and reranking pipelines combined via rank-sum scoring
- Baseline pipeline using TF-IDF or fine-tuned BGE embeddings refined by a DeBERTa reranker
- LoRA adapter training on Qwen2.5-14B/Coder-14B across 5 folds with LoRA weight fusion for final submission
- Multi-stage reranking pipeline incorporating KL divergence distillation, misconception rephrasing, and permutation-based logit averaging

## Data processing
- Clustering misconceptions via co-occurrence statistics
- LLM-based MCQ generation with few-shot prompts
- GPT-4o/GPT4-mini LLM-as-judge filtering
- Deduplication using string normalization and embedding similarity
- CoT trace generation with structured prompts
- Task-specific calibration dataset preparation for quantization
- Dynamic similarity thresholds for candidate retrieval
- Oversampling negatives during training
- Relaxed/hard negative mining during biencoder training
- Synthetic data generation to fill unknown misconceptions
- Distilling reasoning text for incorrect answers
- Dynamic padding with collate_fn for batching
- Filtering candidates by unseen training data or previous predictions
- Constructing retrieval triplets with metadata anchors and correct/incorrect misconceptions
- LaTeX to plain text conversion (retaining raw LaTeX in ensemble)
- Misconception rephrasing into unified format
- Permutation-based logit averaging for reranking

## Models
- Qwen2.5-14B
- Qwen2.5-32B
- Qwen2.5-72B
- Qwen2.5-Math-7B
- Qwen2.5-Math-14B
- Qwen2.5-Math-32B
- Qwen2.5-Math-72B
- Qwen2.5-Coder-14B
- QwQ-32B
- Claude 3.5 Sonnet
- GPT-4o
- GPT4-mini
- Mistral-7B
- SimCSE
- e5-mistral-7b-instruct
- bge-en-icl
- stella_en_1.5B_v5
- Gemini-1.5-pro
- bge-large-en-v1.5
- bge-multilingual-gemma2
- SFR-Embedding-2_R
- TF-IDF
- DeBERTa
- Gemma 2 27B
- Qwen35B
- gte-Qwen2-7B-instruct
- LoRA adapters

## Frameworks used
- vLLM
- FlagEmbedding
- AutoAWQ
- DeepSpeed
- SentenceTransformer
- Unsloth
- pylatexenc

## Loss functions
- MultipleNegativesRankingLoss
- CachedMultipleNegativesRankingLoss
- Cross-Entropy Loss
- KL Divergence Distillation

## CV strategies
- GroupKFold on ConstructId (5 folds) with synthetic data marked as fold 99
- Split by QuestionId
- GroupKFold on SubjectId to mitigate CV-LB gap
- Standard cross-validation for training LoRA variants
- 5-fold cross-validation with LoRA weight merging
- Single-fold validation explicitly balancing seen and unseen misconceptions

## Ensembling
- Staged pipeline aggregation combining top candidates from tiered model sizes
- Ensemble of retrievers and pointwise models with cascading reranking
- Probability averaging across folds
- LoRA-tuned reranker ensembling with probability scaling and input shuffling
- Rank-sum voting ensemble across independent pipelines
- LoRA weight fusion across folds
- Multi-retriever ensembling with logit averaging across permutations

## Insights
- Synthetic data generation and curation are critical for out-of-distribution generalization in niche domains.
- Prioritizing recall@32 over MAP@25 for the retriever significantly improves downstream pipeline performance.
- Using a single demonstration per misconception per training batch prevents label noise from in-batch negatives.
- Few-shot examples, pseudo-labeling, negative ratio scaling, and distilled CoT reasoning each provide substantial, compounding gains for rankers.
- Task-specific calibration datasets are essential to mitigate accuracy degradation when quantizing fine-tuned LLMs.

## Critical findings
- GroupKFold on QuestionId yields overly optimistic validation scores, while SubjectId yields overly pessimistic ones; ConstructId provides the optimal balance.
- High-recall retrievers outperform high-MAP retrievers in this pipeline, contrary to typical retrieval expectations.
- Iterative hard negative mining and cross-device negative batching failed to improve recall, while simple temperature reduction to 0.01 succeeded.
- Naively fine-tuning QwQ-32B-Preview with the same recipe as Qwen2.5-32B resulted in worse performance, indicating architectural differences require tailored tuning.
- Splitting the CV by question ID led to a particularly significant increase in the LB score, and large models maintain stable CV/LB scores even with relatively small datasets.
- recall@32 and MAP@25 were often inversely correlated for the retrievers, making traditional ranking metrics misleading for model selection.
- Hard mining, typically beneficial in retrieval tasks, actively degraded recall@32 performance in this specific setup.
- The CV-LB mismatch was driven by MisconceptionIds appearing only in test data, making synthetic data generation essential.
- Single-token output choices (52 alphabets) were crucial for reliable logit-based sorting, while other character sets performed worse.
- Adding more inference batches (top 156 candidates) improved local CV but did not translate to leaderboard gains.
- Leaderboard probing revealed a massive performance gap between seen (0.154) and unseen (0.444) misconceptions, indicating a severe train-test distribution shift.
- The test set contained over 900 misconceptions completely absent from the training data.
- Concatenating or averaging multiple retrieval vectors performed poorly compared to other combination methods.
- Training the reranker on full data or using the QwQ model yielded worse results than the chosen AWQ+LoRA configuration.
- Distillation of a 72B model was attempted but not submitted because it showed no improvement on offline validation.
- Despite higher MRR on unseen misconceptions due to synthetic oversampling, recall remained identical to seen misconceptions, indicating the model converges on simpler patterns but struggles with complex ones.
- A permutation window size of exactly 2 proved optimal for refining reranker outputs, despite the expectation that larger windows might help.
- Scaling the reranker from Qwen 32B to 72B yielded identical results, suggesting model size was not the bottleneck for this task.

## What did not work
- Iterative hard negative mining
- Increasing batch size through cross-device negatives
- Custom batching strategies grouping same SubjectId pairs
- Converting LLM retrievers to bi-directional encoders
- Merging LoRA adapters using Mergekit
- Naively fine-tuning Qwen/QwQ-32B-Preview with standard Qwen2.5 recipes
- Hard mining during training was attempted but found to be unhelpful for recall@32 and was subsequently abandoned.
- Using combinations of two alphabets, alphanumeric combinations, or Japanese hiragana as single-token choices performed worse than standard alphabets.
- Utilizing QwQ-32B-preview yielded no benefit.
- Multi-step reranking (Qwen 2.5 7B → Qwen 2.5 32B) was ineffective.
- Adding problem/misconception examples directly in the reranking prompts degraded performance.
- A self-trained FlagEmbedding-based retriever degraded performance and was removed.
- Incorporating ~2000 GPT4-mini generated samples into one LoRA model yielded insignificant improvements.
- Retrieval: Concat vectors
- Retrieval: Averaging many vectors
- Reranker: Full-data trained model
- Reranker: QwQ model
- Training fp16 rerankers with all available LoRA adapters followed by quantization.
- Training Qwen 72B as a reranker yielded identical results to Qwen 32B.
- Using synthetic data during reranker training.

## Notable individual insights
- rank 1 (1st Place Detailed Solution): Prioritizing recall@32 over MAP@25 for the retriever significantly improves downstream pipeline performance.
- rank 1 (1st Place Detailed Solution): Task-specific calibration datasets are essential to mitigate accuracy degradation when quantizing fine-tuned LLMs.
- rank 3 (3rd Place Solution (with Magic Boost)): Unseen test misconceptions dominate the evaluation set and require probability upweighting to achieve high scores.
- rank 5 (5th Place Solution): Switching GroupKFold grouping key from QuestionId to SubjectId significantly reduced the CV-LB gap.
- rank 6 (6th Place Solution): Balancing seen and unseen misconceptions in validation is critical for aligning CV with the private leaderboard.
- rank 6 (6th Place Solution): Rerankers naturally preserve retriever order, making permutation-based logit averaging necessary for final ranking refinement.

## Solutions indexed
- #1 [[solutions/rank_01/solution|1st Place Detailed Solution]]
- #1 [[solutions/rank_01/solution|1st Place Solution Summary]]
- #3 [[solutions/rank_03/solution|3rd Place Solution (with Magic Boost)]]
- #5 [[solutions/rank_05/solution|5th Place Solution]]
- #6 [[solutions/rank_06/solution|6th Place Solution]]
- #7 [[solutions/rank_07/solution|Private 7th (Public 2nd) Place Solution Summary]]
- #10 [[solutions/rank_10/solution|10st Place Solution Summary]]
- ? [[solutions/rank_xx_543519/solution|🔥🔥🔥Some tricks for training LLM Recall Model: CV 0.490, LB 0.352]]
- ? [[solutions/rank_xx_534317/solution|I published Retriever Reranker Baseline(LB: 0.189), Fine-Tuning BGE Baseline(LB: 0.246)]]

## GitHub links
- [rbiswasfc/eedi-mining-misconceptions](https://github.com/rbiswasfc/eedi-mining-misconceptions) _(solution)_ — from [[solutions/rank_01/solution|1st Place Detailed Solution]]
- [FlagOpen/FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding) _(library)_ — from [[solutions/rank_01/solution|1st Place Detailed Solution]]
- [anthropics/anthropic-cookbook](https://github.com/anthropics/anthropic-cookbook) _(reference)_ — from [[solutions/rank_01/solution|1st Place Detailed Solution]]
- [casper-hansen/AutoAWQ](https://github.com/casper-hansen/AutoAWQ) _(library)_ — from [[solutions/rank_01/solution|1st Place Detailed Solution]]
- [vllm-project/vllm](https://github.com/vllm-project/vllm) _(library)_ — from [[solutions/rank_01/solution|1st Place Detailed Solution]]
- [BlackPearl-Lab/KddCup-2024-OAG-Challenge-1st-Solutions](https://github.com/BlackPearl-Lab/KddCup-2024-OAG-Challenge-1st-Solutions) _(solution)_ — from [[solutions/rank_xx_543519/solution|🔥🔥🔥Some tricks for training LLM Recall Model: CV 0.490, LB 0.352]]
- [anthropics/anthropic-cookbook](https://github.com/anthropics/anthropic-cookbook) _(reference)_ — from [[solutions/rank_01/solution|1st Place Solution Summary]]
- [unslothai/unsloth](https://github.com/unslothai/unsloth) _(library)_ — from [[solutions/rank_05/solution|5th Place Solution]]
- [ebinan92/Eedi-5th-solution](https://github.com/ebinan92/Eedi-5th-solution) _(solution)_ — from [[solutions/rank_05/solution|5th Place Solution]]

## Papers cited
- [Novice Learner and Expert Tutor: Evaluating Math Reasoning Abilities of Large Language Models with Misconceptions](https://arxiv.org/pdf/2310.02439)
- [EEDI paper](https://arxiv.org/pdf/2404.02124)
