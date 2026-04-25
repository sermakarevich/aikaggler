# wsdm-cup-multilingual-chatbot-arena: cross-solution summary

The competition centered on chatbot preference prediction, where top performers heavily leveraged large language models fine-tuned on curated human preference data, external battle logs, and high-quality pseudo-labeled datasets. Winning approaches consistently combined knowledge distillation from 70B+ teacher models into smaller students (e.g., Qwen2.5-14B, Gemma2-9B), sophisticated data curation strategies, and targeted test-time augmentation or uncertainty-aware routing to maximize inference efficiency and leaderboard performance.

## Competition flows
- Raw prompts are processed with fasttext for language inference, fine-tuned via a custom token-targeted loss and self-distillation, quantized with auto-round, and deployed via vLLM with a dynamic token-length-based ensemble to generate the final submission.
- Raw preference data (labeled and unlabeled/pseudo-labeled) is used to fine-tune multiple LLMs, which are then deployed in an uncertainty-aware inference pipeline that scores all test prompts with a base model, re-scores the most uncertain prompts with progressively larger models, and combines them via a weighted ensemble for the final submission.
- Raw text data is truncated proportionally in the middle, then used to fine-tune Gemma-2 and Llama-3 models initialized from previous checkpoints; iterative pseudo-labeling with external datasets and large open models expands the training pool, followed by a two-epoch training schedule (English-first, then multilingual) and weighted TTA inference to produce the final submission.
- Aggregated external prompts and DPO data to generate ~560K pseudo-labeled samples, distilled knowledge from large teacher models (Llama3.3-70B, Qwen2.5-72B) into smaller student models (Gemma2-9B, Qwen2.5-14B) via LoRA and KL/CE loss, and submitted predictions using a primary-supporting inference strategy optimized for a 12-hour time limit.
- Full-parameter fine-tuning of Qwen2.5-14B/72B on external preference data, distilling soft labels from five teacher models trained on curated human preference and synthetic datasets, merging two student models, and applying a two-pass inference strategy with targeted response-order swapping on uncertain samples.
- Raw text data was pre-trained on external preference datasets, fine-tuned with QLoRA and an auxiliary head, and then processed through a token-sorted, GPU-alternating inference pipeline with selective TTA before submission.

## Data reading
- Text sequences (prompt, response_a, response_b) are loaded and proportionally truncated in the middle based on length ratios.

## Data processing
- FastText language inference on prompts
- Prompt and response lengths capped at 2048 tokens
- Custom system/user/response prompt templates
- Linear weight merging of models trained with different seeds
- Sampled subsets from multiple unlabeled datasets (6k, 25k, 30k examples)
- Cluster-based sampling to align prompt distribution with Arena Explorer
- Generated pseudo-labels for unlabeled data using 70B+ LLMs
- Proportional middle truncation of prompt, response_a, and response_b based on length ratios
- Deduplication of prompts from lmsys-chat-1m and removal of prompts overlapping with external datasets
- Filtering pseudo-labels by confidence margin (abs(winner_model_a - winner_model_b) <= 0.03)
- Sampling datasets with increasing non-English ratios to improve multilingual performance
- Removal of 'think' sections from DeepSeek-R1 model outputs
- Filtered short responses to obtain ~560K samples
- Mixed 3rd-place LMSYS data, API-generated responses from a 1M prompt dataset, and ~10 open-source DPO datasets
- Applied 4-bit quantization
- Set maximum sequence length to 2500 tokens
- Prioritized the top five languages with a specific focus on English
- Left truncation
- Translating training and test data
- Injecting language identifiers during training
- Adding a second set of labels
- Test-time augmentation (TTA)
- Filtering out ties and multi-turn responses
- Concatenating prompt and responses with proportional middle truncation
- Prepending model names to responses
- Filtering synthetic prompts by complexity score (≥3) while preserving language/domain distributions
- Uniformly sampling completions from specified models via vLLM
- Inserted (snip <N> chars) placeholders in prompts and responses to indicate omitted text
- Swapped responses A and B during training, ensuring swapped samples appeared in the same batch
- Used external datasets (dpo_44k, ut_157k, add_23k) for stage 1 pre-training

## Models
- Qwen2.5-14B-Instruct
- Phi-4
- Qwen2.5-72B-Instruct
- Athene-V2-Chat
- Qwen2.5-3B
- gemma-2-9b-it
- Qwen2.5-32B-Instruct
- Mistral-Small-24B-Instruct-2501
- Llama-3.1-Nemotron-70B-Instruct
- AutoModelForSequenceClassification
- gemma2-9b
- ArmoRM-Llama3-8B-v0.1
- gemma2-27b
- Llama-3.3-70B-Instruct
- DeepSeek-R1-Distill-Qwen-32B
- Mistral-7B-Instruct-v0.3
- internlm2_5-20b-chat
- llama-3.2-3b-instruct
- Hermes-3-Llama-3.1-8B
- internlm3-8b-instruct
- DeepSeek-R1-Distill-Qwen-14B
- Llama-3.2-1B-Instruct
- Skywork-Reward-Gemma-2-27B-v0.2
- Skywork-Reward-Llama-3.1-8B-v0.2
- Skywork-Reward-Preference-80K-v0.2
- mDeBERTa
- GPT-3.5

## Frameworks used
- vLLM
- DeepSpeed
- Auto-Round
- FastText
- Transformers
- PyTorch

## Loss functions
- Cross-entropy (custom implementation targeting only tokens 'A' and 'B')
- KL divergence loss (temperature 5.0, weight 0.9)
- Combined language modeling and reward modeling losses
- KL Divergence Loss
- Cross-Entropy Loss
- Equally weighted average of KL Divergence Loss and Cross-Entropy Loss
- Cosine loss

## CV strategies
- Single fold (Fold 0) used for hyperparameter sweeps and ablation studies.
- 80/20 train/validation split for initial fine-tuning, later expanded to 100% training data for final stages.
- 5-fold cross-validation for teacher training
- 5-fold cross-validation (only fold_id=0 used for faster evaluation); final model retrained on the full dataset without evaluation splits.

## Ensembling
- Linear weight merging of models trained with different seeds, combined with dynamic token-length-based weighting where Qwen2.5-14B receives 25% TTA and Phi-4's weight is adjusted by prompt length quantiles.
- Uncertainty-based weighted ensemble where the base model scores all examples, and the top 50% most uncertain examples are re-scored with larger models; final scores are computed as `1.0*base + 1.5*xlarge` for the top 15% uncertain, `1.0*base + 1.0*large` for the next 35%, and `1.0*base` for the rest.
- Weighted ensemble of TTA predictions from `gemma-final` (PAB) and `llama-final` (PBA swapped) at a 2.5:1 ratio, with `llama-final` only applied to high-confidence samples (`abs(winner_model_a - winner_model_b) < 0.8`).
- Used a primary-supporting model setup where Qwen2.5-14B provided primary logits and Gemma2-9B was deployed selectively for difficult cases; explicitly tested multi-LoRA ensembling but found it failed to improve performance.
- The final model is a linear merge of two student models (one trained on the full dataset, one excluding the last two datasets), combined with a two-pass inference strategy that applies response-order swapping via the original model on the 33% most uncertain predictions.
- Explored model merging techniques including Lerp, Slerp, weighted exchanges, TIES, DARE, and LoRA weight balancing, but did not finalize a successful merging formula.
- Averaged predictions with swapped responses A and B for the top 40% of samples with probabilities closest to 0.5 from the first inference round.

## Insights
- Self-distillation using the 14B model's own logits is as effective as using a 72B teacher model, likely due to the label cleaning effect of soft logits.
- Heavy quantization settings (long sequence length and high iteration counts) are necessary to suppress accuracy drops during auto-round quantization.
- Dynamic ensemble weighting by prompt length effectively leverages each model's strengths, improving overall leaderboard performance.
- Larger LLMs tend to generalize better when handling uncertain examples.
- Ensembling different modeling approaches and model families improves robustness.
- Keeping the preference modeling head simple (a single linear layer) aids generalization.
- Distilling from 70B+ LLMs on a large, diverse unlabeled dataset significantly boosts performance.
- Increasing the non-English data ratio in pseudo-labeling datasets significantly improves multilingual performance.
- A two-epoch training schedule (English-first, then multilingual) effectively balances language capabilities without degrading English performance.
- Filtering pseudo-labels by a confidence margin reduces noise during iterative training.
- Pseudo-labeling with high-quality external data can yield strong leaderboard scores even without direct competition data training.
- Multilingual performance should be deprioritized if base models are already strong multilingualizers, focusing instead on optimizing English accuracy.
- Extending sequence length beyond 2500 tokens offers diminishing returns relative to training time constraints.
- Parallelizing inference across multiple GPUs via custom scripts significantly improves speed.
- Left truncation, knowledge distillation, and maximizing TTA are critical for LLM competitions.
- Language handling strategies (translation, identifiers) did not improve scores but revealed transformer sensitivity to data variations.
- The Kaggle API is highly effective for streaming cloud data and avoiding local I/O bottlenecks.
- Model merging can theoretically impact performance without retraining, though finding the right formula is non-trivial.
- Manually reviewing and correcting high-disagreement soft-labeled samples significantly improves distillation quality.
- Including evaluation datasets in the training mix can be beneficial if there's a risk of data leakage or if they correlate with the test distribution.
- Targeted test-time augmentation on only the most uncertain samples balances computational cost with performance gains better than applying it to all samples.
- Preserving language and domain distributions when generating synthetic data is crucial for maintaining model generalization.
- Inserting a placeholder with the exact number of omitted characters helped the model handle partial information and improved accuracy.
- Training with swapped responses A and B improved accuracy but required swapped samples to appear in the same batch to be effective.
- Sorting samples by token count and alternating them across two GPUs significantly accelerated inference.

## Critical findings
- A learning rate of 14e-5 causes loss spikes for the 14B model, making 10e-5 the optimal choice.
- Phi-4 performs well on short prompts but degrades significantly on long prompts, necessitating length-aware ensemble tuning.
- Uncertainty can be effectively approximated as closeness to the decision boundary (`1-|p-0.5|`) to route examples to larger models.
- Directly using "A" and "B" token logits from non-reward models combined with a combined LM and reward loss is a viable alternative to standard classification heads.
- Excluding @nbroad's v3 pseudo-label dataset prevented noise because the model capability gaps were too minimal.
- Using `llama-final` only for high-confidence predictions (`< 0.8` margin) during TTA improved the private score without hurting the LB score.
- LoRA merging and post-training quantization caused significant, unresolved performance degradation.
- Dynamic truncation during inference initially provided a small leaderboard boost but lost effectiveness after training code updates.
- Original labels from DPO datasets were ineffective for direct post-training but highly valuable for pseudo-labeling.
- Custom classification heads and language-aware training strategies failed to improve scores despite extensive experimentation.
- The author hit a hard performance wall and could not replicate scores of 0.710 achieved by others, suggesting hidden optimization or data leakage factors.
- Training on soft labels generated by a critique model with Chain-of-Thought reasoning performed worse than training without them.
- Applying TTA to 100% of samples using a quantized 72B model was significantly worse on the LB than applying it to only 33% of samples with the original model.
- Layer pruning based on activation similarity for larger models yielded worse results than simply using a smaller, unpruned model.
- Enforcing similar output embeddings before the linear head for response orders did not eliminate the need for a second inference pass.
- Using language ID classification as an auxiliary task provided no improvement.
- Applying TTA across the entire dataset was less efficient and accurate than selectively applying it only to uncertain predictions.

## What did not work
- Distilling Qwen2.5-3B to generate CoT improved scores slightly but caused frequent inference timeouts due to excessive latency.
- Using Qwen2.5-14B distilled by deepseek and Qwen2.5-14B-1m resulted in poor leaderboard scores.
- Qwen2.5-72B distillation (limited by time/GPU, lacked pretraining cycle)
- Pseudo-labeling with Skywork-Reward-Gemma-2-27B-v0.2
- Replacing `lmsys-llama-pretrain` with Skywork-Reward-Llama-3.1-8B-v0.2
- Using Skywork-Reward-Preference-80K-v0.2 dataset
- BT loss, multi-loss weighting
- TTA showed no measurable impact on sequence classification.
- LoRA merge caused significant performance degradation.
- Multi-LoRA ensemble approaches failed to improve performance.
- Dynamic token allocation based on response length led to abnormal length distribution.
- Chain-of-Thought prompting strategies did not yield meaningful improvements.
- Dynamic truncation during inference initially provided a +0.003 LB boost but effectiveness diminished later.
- Fine-tuning mDeBERTa for multiple-choice tasks resulted in poor performance due to parameter limitations.
- Using GPT-3.5 as a judge model for pseudo-labeling yielded unsatisfactory results due to sensitivity and lack of transparency.
- Few-shot prompting (2-shot to 32-shot) did not improve accuracy.
- Custom classification heads yielded no significant score improvements.
- Language handling strategies (translated data, test translation, language identifiers) failed to help.
- Model merging techniques were explored but the author did not find the correct formula to make them work.
- Critique model with CoT for soft labeling.
- Full-sample TTA with quantized 72B model.
- Layer pruning based on activation similarity.
- Enforcing similar output embeddings for response orders.
- Language ID classification auxiliary task (provided no improvement).

## Notable individual insights
- rank 3 (3rd place solution): Self-distillation using the 14B model's own logits is as effective as using a 72B teacher model, likely due to the label cleaning effect of soft logits.
- rank 7 (7th Place Solution): Uncertainty can be effectively approximated as closeness to the decision boundary (`1-|p-0.5|`) to route examples to larger models.
- rank 2 (The 2nd Place Solution): Increasing the non-English data ratio in pseudo-labeling datasets significantly improves multilingual performance.
- rank 6 (6th Place Solution): Pseudo-labeling with high-quality external data can yield strong leaderboard scores even without direct competition data training.
- rank 1 (1st Place Solution): Manually reviewing and correcting high-disagreement soft-labeled samples significantly improves distillation quality.
- rank 12 (12th Place Solution): Inserting a placeholder with the exact number of omitted characters helped the model handle partial information and improved accuracy.
- rank null (What I learned in WSDM): The Kaggle API is highly effective for streaming cloud data and avoiding local I/O bottlenecks.

## Solutions indexed
- #1 [[solutions/rank_01/solution|LSMSY 1st Solution and Code]]
- #1 [[solutions/rank_01/solution|1st Place Solution]]
- #2 [[solutions/rank_02/solution|The 2nd Place Solution]]
- #3 [[solutions/rank_03/solution|3rd place solution]]
- #6 [[solutions/rank_06/solution|6th Place Solution]]
- #7 [[solutions/rank_07/solution|7th Place Solution]]
- #12 [[solutions/rank_12/solution|12th Place Solution]]
- ? [[solutions/rank_xx_547480/solution|Top Solutions From the Previous Competition (LMSYS)]]
- ? [[solutions/rank_xx_561065/solution|What I learned in WSDM]]

## GitHub links
- [shyoulala/LMSYS_BlackPearl](https://github.com/shyoulala/LMSYS_BlackPearl) _(solution)_ — from [[solutions/rank_01/solution|LSMSY 1st Solution and Code]]
- [LIHANG-HONG/WSDM-Cup-3rd-place-solution](https://github.com/LIHANG-HONG/WSDM-Cup-3rd-place-solution) _(solution)_ — from [[solutions/rank_03/solution|3rd place solution]]
- [tascj/kaggle-lmsys-chatbot-arena](https://github.com/tascj/kaggle-lmsys-chatbot-arena) _(reference)_ — from [[solutions/rank_03/solution|3rd place solution]]
- [tascj/kaggle-lmsys-chatbot-arena](https://github.com/tascj/kaggle-lmsys-chatbot-arena) _(reference)_ — from [[solutions/rank_02/solution|The 2nd Place Solution]]
- [zhudongwork/wsdm-cub-2nd-place-solution](https://github.com/zhudongwork/wsdm-cub-2nd-place-solution) _(solution)_ — from [[solutions/rank_02/solution|The 2nd Place Solution]]
- [maxreciprocate/kaggle-lmarena-1st-place](https://github.com/maxreciprocate/kaggle-lmarena-1st-place) _(solution)_ — from [[solutions/rank_01/solution|1st Place Solution]]
- [rbiswasfc/lmsys-arena](https://github.com/rbiswasfc/lmsys-arena) _(reference)_ — from [[solutions/rank_01/solution|1st Place Solution]]

## Papers cited
- [Regularizing Hidden States Enables Learning Generalizable Reward Model for LLMs](https://arxiv.org/abs/2406.10216)
- [MT-Bench](https://arxiv.org/abs/2306.05685)
- [Arena-hard](https://lmsys.org/blog/2024-04-19-arena-hard/)
