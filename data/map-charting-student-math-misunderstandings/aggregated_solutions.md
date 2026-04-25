# map-charting-student-math-misunderstandings: cross-solution summary

This competition focused on predicting student math misconceptions from question text, answer choices, and student explanations, ultimately framing the task as a multi-class classification or constrained generation problem. Winning approaches consistently leveraged large language models (primarily Qwen and DeepSeek families) fine-tuned via parameter-efficient methods like LoRA/QLoRA, combined with rigorous data hygiene, targeted synthetic augmentation, and multi-stage or weighted ensembling to navigate severe label noise and long-tail class distributions.

## Competition flows
- Raw text data deduplicated and split into 5 stratified folds, formatted into prefix-shared sequences with custom attention masks, trained via full-parameter updates, followed by multi-seed ensembling, W8A8 INT8 quantization, and layer-wise inference
- Pipeline iteratively routes test samples through progressively larger LLMs based on prediction uncertainty, applying weighted ensembling and recomputing confidence at each stage
- Raw question and explanation data structured into constrained prompts containing question-specific choices and hints, processed through multi-stage LLM inference pipelines with confidence-based re-inference and rule-based post-processing
- Raw tabular/text data formatted into structured prompt template, tokenized, fed into QWEN3-14B fine-tuned via QLoRA across 4 KFold splits, predictions averaged for submission
- Raw CSV data cleaned and split into 5 folds with near-duplicates grouped, fed into three Qwen backbones fine-tuned via QLoRA with mixed Focal/CE loss, followed by GPTQ-4bit quantization and vLLM embedding for inference, combined via weighted logit blending and Top-3 selection
- Raw student explanations and questions augmented with LLM-generated external data and soft labels, fine-tuned on quantized LLMs using custom multi-loss trainer, combined via equal-weight ensemble
- Raw student math Q&A data cleaned and deduplicated, enriched with targeted augmentations and ChatGPT-generated synthetic samples, formatted into fixed prompt template, fine-tuned via Q-LoRA, predictions aggregated through multi-model ensemble
- Raw question text, images, and student explanations formatted into structured prompts with inline answer choices and single-token target labels, fine-tuned via LoRA, augmented with GPT-5/o3 synthetic data, combined into weighted ensemble of 50 models accelerated by vLLM
- Raw text data loaded into pandas DataFrames, split via StratifiedGroupKFold, fine-tuned across six diverse LLMs using LoRA/QLoRA/FullFT, processed through question-specific misconception masking, combined via weighted average ensemble
- Raw CSV data cleaned and split into 5 stratified folds, candidate targets mapped per question, pointwise ranking model fine-tuned via knowledge distillation using iterative few-shot prompt engineering, generated via vLLM inference with float16 conversion
- Raw competition data and synthetic explanations processed through filtered prompts and reverse-engineered context, fed into multiple fine-tuned LLMs using direct logit training and knowledge distillation, outputs ensembled and post-processed

## Data reading
- pandas read_csv for training data
- LabelEncoder for target mapping
- NaN filling for the Misconception column
- pandas DataFrame

## Data processing
- Dropped exact duplicates
- Pseudo-labeled duplicates and reintroduced them to training data
- Formatted inputs into structured/fixed prompt templates
- Reformulated prompts into multiple-choice formats with shuffled options
- Added alternative answer choices and probability-ordered hints to prompts
- Constrained token sets per QuestionId
- Stripped True/False prefixes and labels
- Auto-fixed True/False inconsistencies
- Kept near-duplicates in the same fold
- Filled NaNs with 'NA'
- Constructed targets as Category + Misconception
- Reduced classification task from 65 to 37 classes
- Stripped LaTeX notation from QuestionText and MC_Answer
- Reconstructed MC_Answer to choices a-d
- Mapped target labels to single special Unicode tokens
- Applied label smoothing (α=0.1)
- Applied per-class weights with 33% weight warmup
- Generated targeted text augmentations (punctuation replacement, word-to-number conversion, typo injection)
- Generated synthetic training samples for rare/low-performing categories
- Created question-misconception maps for inference masking
- Filtered misconception candidates per question to shorten prompts
- Corrected wrong category labels
- Applied dynamic padding to max length 160
- Converted token IDs to PyTorch long tensors
- Applied AWQ quantization
- Converted outputs to float16 for inference

## Features engineering
- Structured prompt template incorporating QuestionText, possible answers, misconceptions, student answer, explanation, and ground truth label
- Mapped candidate_targets and neg_candidate_targets per QuestionId using groupby and set operations
- Reduced prediction labels from 65 to 37 by focusing on explanation type and candidate targets

## Models
- Qwen3-8B
- Qwen3-14B
- Qwen3-32B
- Qwen3-235B-A22B-Thinking-2507-FP8
- Qwen3-Reranker-8B
- Qwen3-Embedding-8B
- Qwen2.5-7B-Instruct
- Qwen2.5-14B-Instruct
- Qwen2.5-32B-Instruct
- Qwen2.5-72B-Instruct
- Qwen2.5-Math-7B-Instruct
- Qwen2.5-Math-72B-Instruct
- QWQ-32B
- Qwen2.5-14B-Instruct-AWQ
- Qwen2.5-32B-Instruct-AWQ
- Qwen3-14B-AWQ
- Qwen3-32B-AWQ
- Qwen25-14B-AWQ
- Qwen25-32B-AWQ
- DeepSeek-R1-Distill-Qwen-7B
- DeepSeek-R1-Distill-Qwen-14B
- DeepSeek-R1-Distill-Qwen-32B
- deepseek-ai/deepseek-math-7b-instruct
- GLM-Z1-9B-0414
- GLM-Z1-32B-0414
- Gemma-2-9B-IT
- gemma2
- Mistral-Nemo-Instruct-2407
- Mixtral-8x7B-Instruct-v0.1
- Mathstral-7B-v0.1
- Mistral
- Mistral-12B
- Phi-4
- Phi-4-Reasoning-14B
- Llama
- Llama-3.3-70B
- AutoModelForSequenceClassification
- AutoModelForCausalLM
- Multi-head architecture
- GTE
- Ettin
- GPT-4.1
- GPT-5-mini
- Claude Sonnet
- AmanPriyanshu/gpt-oss-10.8b-specialized-instruction_following-pruned-moe-only-15-experts

## Frameworks used
- offload_adam
- LMDeploy
- FlexAttention
- SmoothQuant
- vLLM
- Hugging Face
- AWQ
- bitsandbytes
- PyTorch
- HF Trainer
- LoRA
- QLoRA
- auto-round
- transformers
- Trainer
- SFTTrainer
- pandas

## Loss functions
- Cross-entropy loss
- Auxiliary SFT loss
- Focal loss
- Cross-Entropy with label smoothing (0.1)
- Per-class weights
- Generation loss (hard labels)
- Classification loss (soft labels)
- KL divergence loss

## CV strategies
- 5-fold cross-validation stratified by Category
- Multi-seed ensembles for validation stability
- Out-of-fold (OOF) evaluation
- 4-fold cross-validation
- 5-fold cross-validation with near-duplicates kept in the same fold
- 5-fold cross-validation with OOF averaging for soft label generation
- 20% train/test split for local validation
- 5-fold cross-validation with out-of-fold MAP@3 aggregation
- StratifiedGroupKFold grouped by StudentExplanation
- StratifiedKFold
- Selection of a slightly below-average CV fold for validation
- Fold-based training with OOF predictions for distillation and filtering

## Ensembling
- Multi-seed averaging for validation stability
- 5-stage pyramid ensemble routing uncertain predictions to progressively larger models with increasing weights (2x, 4x, 8x) and recomputing uncertainty
- Multi-stage inference re-inferencing lowest-confidence samples with larger models combined via probability averaging
- Bagging multiple folds and seeds with prediction averaging
- Weighted logit blend followed by Top-3 selection for MAP@3
- Equal-weight average fusion of 37-class and 65-class models
- Community-sourced ensemble with pseudo-labeling of duplicates
- Weighted ensemble based on CV and Public LB performance
- Simple weighted average of predicted probabilities with post-inference probability masking for impossible misconceptions
- Single-model submission after ensemble underperformance
- Combined pipelines with abandoned dynamic weight adjustment and True/False label post-processing

## Insights
- Single-seed validation scores are highly unstable due to label noise and should not be trusted for model selection.
- Multi-seed ensembles provide significantly more stable and reliable validation metrics than multi-fold ensembles.
- Larger model sizes consistently yield better performance.
- Validation loss is a more trustworthy metric for model selection than MAP@3.
- Prompt engineering and paradigm selection significantly impact LLM performance on this task.
- Multi-head architectures can effectively decompose complex label spaces when loss is calculated conditionally.
- Inference threading is essential for maximizing GPU utilization during large-scale prediction.
- Presenting alternative answer choices in prompts significantly helps LLMs identify potential misunderstandings.
- Constraining CausalLM prediction tokens to only the misconceptions relevant to each specific QuestionId improves accuracy and reduces inference time.
- High learning rates can improve scores but require careful tuning to prevent training collapse.
- Optimizing inference speed via dtype and padding changes enables larger model ensembles within strict time limits.
- Bagging effectively simulates annotator disagreement and mitigates label noise.
- Adversarial Weight Perturbation (AWP) can significantly boost CV scores but requires careful time management.
- Clean data splits and balanced training with many modest models outperform heavier pipelines for tasks with label noise, long-tail distributions, and MAP@3 objectives.
- Reframing classification as multi-choice generation enables efficient vLLM inference.
- Soft labels derived from averaged OOF predictions effectively handle ambiguous MAP@3 labels.
- Combining generation loss with classification loss improves model performance on this task.
- Appending an explicit task instruction line to the prompt consistently improved both CV and LB scores.
- Models with weaker individual scores can still significantly boost ensemble performance if they provide complementary diversity.
- Targeted augmentations based on fuzzy-searched near-duplicates are more effective than blindly generating large volumes of synthetic data.
- Fine-tuning with a different random seed can yield a model that performs poorly alone but adds crucial value to an ensemble.
- Explicitly including answer choices (a–d) in the prompt significantly improved the model's ability to compare the student's answer against alternatives and identify the correct misconception.
- Mapping target labels to single special Unicode tokens streamlined generation and aligned with the Qwen tokenizer's capabilities.
- Careful joint tuning of learning rate and batch size is critical, as small changes can cause large swings in cross-validation scores.
- LoRA fine-tuning outperformed full-parameter fine-tuning in this setup, likely due to better regularization or insufficient tuning for the latter.
- Prioritizing model diversity rather than individual model optimization is essential for improving predictions on rare, hard-to-classify data.
- Cross-validation scores can completely lose correlation with leaderboard scores when the test set contains novel expressions not present in training.
- Leveraging domain knowledge that valid misconceptions are strictly limited per question significantly boosts prediction accuracy.
- Simplifying the task to predict explanation types and candidate targets per question significantly reduces the prediction space and improves model focus.
- Using a pointwise ranking approach with small effective batch sizes (4-6) per row_id aligns well with the label structure of this competition.
- Iterative prompt engineering with few-shot examples per target category yields substantial score gains.
- Training on the full dataset for 2 epochs improves performance, but hardware constraints require careful dtype handling to avoid inference NaNs.
- Synthetic data effectiveness is highly model-dependent.
- Prompt engineering and candidate filtering are crucial for efficiency.
- Combining distillation with label smoothing effectively handles noisy data.
- Pre-identifying correct answers eliminates one source of error.

## Critical findings
- Label noise, particularly in the Neither category, causes validation scores to fluctuate wildly across different random seeds.
- Multi-seed ensembling outperforms multi-fold ensembling for validation stability.
- Storing layer checkpoints in Kaggle's /tmp directory causes crashes due to copy-on-write capacity limits.
- Synthetic data generated by GPT to fix class confusion was largely untrustworthy and provided no consistent improvement.
- Multi-head loss calculation requires careful handling of conditional label availability to avoid gradient issues.
- Cascading uncertainty-based routing efficiently concentrates compute on the hardest predictions without wasting resources on easy samples.
- High noise levels in the competition's labels necessitated regularization techniques like R-Drop, AWP, and EMA to stabilize training and improve generalization.
- Using float16 instead of bfloat16 on Kaggle T4 instances avoids conversion overhead and doubles inference speed.
- Setting padding=False for short prompts provides an additional 2x speedup, crucial for fitting more models into the ensemble.
- The dataset labels are noisy due to annotator disagreement, making bagging a highly effective regularization technique.
- Public leaderboard scores were so close that relying on them for submission selection would be unreliable.
- Using a weighted logit blend with Top-3 selection is more stable than probability averaging for MAP@3.
- A 33% weight warmup is critical to stabilize early training updates on long-tail classes.
- Keeping near-duplicates in the same validation fold prevents data leakage and improves generalization.
- Fusing additional data with distillation yields no further score improvement despite initial gains.
- AWQ quantization allows running 32B parameter models with acceptable inference times (20-40 minutes).
- Adding a model trained on augmented data consistently improved the ensemble score despite its poor individual leaderboard performance.
- Dropping duplicates initially was necessary, but pseudo-labeling and reintroducing them later provided a significant score boost.
- Creating ~10k augmented samples based on multiple patterns yielded no public LB improvement, highlighting the importance of targeted over brute-force augmentation.
- Stripping LaTeX notation from QuestionText and MC_Answer had absolutely no impact on the final score, despite being done for readability.
- Varying LoRA r or alpha parameters yielded no meaningful improvement in scores, indicating the chosen configuration was already near-optimal for this task.
- Identical StudentExplanation texts across folds cause overly optimistic cross-validation scores.
- CV and Public LB correlation disappears once single model scores reach approximately 0.946.
- The set of possible misconceptions is strictly constrained per QuestionId, enabling effective probability masking during inference.
- The training set and online test set share the same 15 questions, meaning correct answer prediction is unnecessary and the task can be simplified to explanation type prediction.
- Converting the model to float16 for vLLM inference on T4 GPUs caused overflow and NaN values, necessitating a workaround of retraining on augmented validation data instead of relying on the 2-epoch full-data model for inference.
- Retrieving and adding the most relevant explanations via a fine-tuned embedding model improved CV scores but degraded public/private leaderboard performance.
- Using existing tokens as correct answers instead of special tokens yielded significant CV/LB gains.
- Selecting a slightly later checkpoint rather than the lowest eval_loss checkpoint improved performance.
- Synthetic data improvements were specific to Gemma2-9B-it + OOF predictions, showing model-specific adaptation needs.

## What did not work
- Storing model layer checkpoints in Kaggle's /tmp directory caused a crash due to storage capacity limits, forcing a rushed submission with fewer models than intended.
- Generating synthetic training data with GPT to address class confusion yielded mixed results and was ultimately discarded due to reliability concerns.
- Models other than Qwen2.5 (including Llama 3.3-70B and gemma2) performed worse.
- Using CausalLM for pair-wise decision making was too time-consuming.
- Fixing choices tokens for each label (e.g., A, B, F, D, R) did not work.
- Pretraining on Eedi competition data did not improve results.
- Considering label noise by including reversed True/False versions in top predictions slightly decreased final accuracy.
- Hierarchical multi-task classifier with hard logic constraints.
- 32B model as pointwise reranker.
- 32B model as listwise reranker.
- TTA.
- Fusing additional data with distillation provided no further benefit after initial score improvements.
- Using multiple pre-trained models without fine-tuning.
- Testing several different prompt templates.
- Generating ~10k augmented samples across multiple patterns without targeted selection.
- Using all original 65 labels instead of the consolidated 37.
- A 2-stage classification approach (correctness then misconception type).
- Text generation and textual entailment prompting strategies.
- Using an MoE-style NN for the classification head instead of LoRA adapters.
- Experimenting with focal-loss.
- Full-parameter fine-tuning yielded worse cross-validation scores compared to LoRA, likely due to insufficient tuning on the author's side.
- listwist ranker.
- chain of thought.
- add the most relevant explanation to prompt.
- Dynamic ensemble weight adjustment per question failed due to computational constraints and insufficient data.
- Adding small/lightweight models to the ensemble provided no improvement.

## Notable individual insights
- Rank 1 (1st Place Solution): Single-seed validation scores are highly unstable due to label noise and should not be trusted for model selection; multi-seed ensembles provide significantly more stable validation metrics.
- Rank 18 (18th Place - Pyramid Ensemble): Synthetic data generated by GPT to fix class confusion was largely untrustworthy and provided no consistent improvement.
- Rank 3 (3rd place solution): High noise levels in competition labels necessitated regularization techniques like R-Drop, AWP, and EMA to stabilize training.
- Rank 10 (10th Place Solution): Using a weighted logit blend with Top-3 selection is more stable than probability averaging for MAP@3 objectives.
- Rank 5 (5th place solution): Cross-validation scores can completely lose correlation with leaderboard scores when the test set contains novel expressions not present in training.
- Rank 4 (4th Place Solution): The training set and online test set share the same 15 questions, meaning correct answer prediction is unnecessary and the task can be simplified to explanation type prediction.
- Rank 7 (Private 7th Place Solution): Selecting a slightly later checkpoint rather than the lowest eval_loss checkpoint improved performance.

## Solutions indexed
- #1 [[solutions/rank_01/solution|1st Place Solution]]
- #2 [[solutions/rank_02/solution|MAP2025_Private&Public 2nd ]]
- #3 [[solutions/rank_03/solution|3rd place (Public 1st) solution : monsaraida & Masaya]]
- #4 [[solutions/rank_04/solution|4th Place Solution (single model pb 0.948 lb 0.951)]]
- #5 [[solutions/rank_05/solution|5th place solution]]
- #6 [[solutions/rank_06/solution|✨ MAP 6th Place Solution - Qwen-semble FTW! 🚀]]
- #7 [[solutions/rank_07/solution|Private 7th (Public 10th) Place Solution]]
- #8 [[solutions/rank_08/solution|8th Place Solution]]
- #10 [[solutions/rank_10/solution|10th Place Solution]]
- #18 [[solutions/rank_18/solution|18th Place - Pyramid Ensemble]]
- #22 [[solutions/rank_22/solution|Place 22 with only 1 model]]

## GitHub links
- [tascj/offload_adam](https://github.com/tascj/offload_adam) _(library)_ — from [[solutions/rank_01/solution|1st Place Solution]]
- [tascj/kaggle-map-charting-student-math-misunderstandings](https://github.com/tascj/kaggle-map-charting-student-math-misunderstandings) _(solution)_ — from [[solutions/rank_01/solution|1st Place Solution]]
- [intel/auto-round](https://github.com/intel/auto-round) _(library)_ — from [[solutions/rank_03/solution|3rd place (Public 1st) solution : monsaraida & Masaya]]
- [mananjhaveri/MAP-Charting-Student-Math-Misunderstandings-6th-place-solution](https://github.com/mananjhaveri/MAP-Charting-Student-Math-Misunderstandings-6th-place-solution) _(solution)_ — from [[solutions/rank_06/solution|✨ MAP 6th Place Solution - Qwen-semble FTW! 🚀]]
