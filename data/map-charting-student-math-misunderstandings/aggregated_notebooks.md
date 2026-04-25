# map-charting-student-math-misunderstandings: top public notebooks

The community's top-voted notebooks primarily focus on adapting large language models and traditional classifiers for a multi-class math misconception ranking task, emphasizing prompt engineering and custom MAP@3 evaluation metrics over complex training pipelines. Contributors consistently highlight memory-efficient inference techniques like LoRA, quantization, and fp16 precision to run 1B–32B parameter models on constrained Kaggle GPUs. The highest-performing approaches rely on strategic feature injection (e.g., majority-vote correctness flags) and weighted rank-based ensembling to aggregate diverse model outputs without extensive fine-tuning.

## Common purposes
- ensemble
- tutorial
- inference
- training

## Competition flows
- Loads train/test CSVs and fills NaN misconceptions with 'NA'
- Engineers majority-vote/frequency-based `is_correct` correctness flags
- Constructs structured prompts combining question, answer, correctness, and explanation
- Tokenizes inputs with AutoTokenizer (max_length=256, padding, truncation)
- Trains sequence classifiers on holdout splits or fine-tunes via LoRA/PEFT
- Runs parallel or quantized inference (vLLM, fp16) on 1B–32B parameter models
- Extracts top-3 predictions via probability aggregation or logit masking
- Combines heterogeneous model outputs via weighted rank-based voting or custom scoring functions
- Exports space-separated submission CSVs

## Data reading
- pd.read_csv for train and test CSVs from the Kaggle dataset directory
- Fills NaN misconceptions with 'NA'
- Constructs composite target string Category:Misconception
- Applies scikit-learn's LabelEncoder to generate integer labels

## Data processing
- Fills NaN misconceptions with 'NA'
- Creates combined/composite target labels (Category:Misconception)
- Applies LabelEncoder for integer labels
- Constructs structured prompt templates
- Tokenizes inputs with AutoTokenizer (max_length=256, padding, truncation)
- Converts DataFrames to Hugging Face Datasets with PyTorch tensor formatting
- Splits data into holdout validation sets (80/20 or 95/5)
- Uses fp16 precision for T4 GPU compatibility
- Maps 65 target classes to unique special characters for single-token output
- Applies softmax to logits for class probabilities
- Text cleaning via regex and lemmatization
- Custom BPE tokenizer training
- Manual MAP@3 metric computation
- DataCollatorWithPadding for dynamic batching

## Features engineering
- Majority-vote/frequency-based `is_correct` flag per QuestionId
- Text length features (question, explanation, mc_answer)
- Math keyword/operator counts via regex
- Explanation-to-question ratio
- Cosine similarity between explanation and question
- Dense sentence embeddings (MiniLM, DeBERTa, RoBERTa)
- Word, character, and custom BPE TF-IDF features
- Prompt formatting with explicit correctness hints

## Models
- Gemma-2-9B-it (LoRA/PEFT)
- Qwen3-8B
- DeepSeek-Math-7B
- Ettin-Encoder-1B
- ModernBERT-large
- DeBERTa-v3-xsmall
- DeBERTaV2ForSequenceClassification
- DeBERTa-v3-large
- Qwen2.5-32B-GPTQ
- XGBoost

## Frameworks used
- pandas
- numpy
- scikit-learn
- torch
- transformers
- datasets
- peft
- matplotlib
- joblib
- IPython
- scipy
- tqdm
- threading
- vllm
- xgboost
- nltk
- sentence_transformers

## Loss functions
- mlogloss

## CV strategies
- 20% holdout validation split via train_test_split
- 5% holdout validation split via train_test_split
- StratifiedKFold (5-fold) for tree-based models
- 5-KFold mentioned as alternative strategy

## Ensembling
- Weighted probability fusion with agreement/confidence bonuses
- Weighted rank-based scoring/aggregation
- Equal-weight rank-based voting
- Logit masking for single-token extraction
- Weighted voting with heterogeneous model weights (e.g., 4:1 ratio)
- Top-3 post-processing via LabelDecoder and space-joining

## Insights
- Parallel multi-GPU inference significantly reduces runtime for large LLMs
- Extracting and merging top-k probabilities allows for fine-grained control over ensemble scoring
- Model agreement and peak confidence can be explicitly weighted to handle disagreement and boost high-confidence predictions
- Using a larger 9B parameter model yields better CV/LB scores than smaller 1B alternatives
- LoRA adapters enable efficient fine-tuning and inference of large LLMs on constrained GPU memory
- Switching to fp16 is necessary and accelerates inference on Kaggle's T4 GPUs since they lack bf16 support
- Carefully crafting the input prompt structure can significantly impact validation scores
- Prompt structure directly impacts model performance and validation scores
- Memory-efficient techniques like gradient accumulation, mixed precision, and LoRA enable training of 1B-parameter models on limited VRAM
- Standard classification metrics should be replaced with competition-specific metrics like MAP@3 for proper model selection
- Pre-trained encoder models like DeBERTa outperform traditional TF-IDF approaches for this text classification task
- Answer frequency per question can serve as a useful proxy feature for correctness
- LLMs can be effectively adapted for multi-class misconception prediction using simple prompt formatting and a binary correctness hint
- A custom MAP@3 metric is necessary to properly evaluate ranking-based multi-label tasks in this competition
- Rank-based ensemble aggregation leverages diverse prediction distributions to improve robustness over individual models
- Mapping each target class to a unique special character forces the LLM to output exactly one token, simplifying decoding and preventing format errors
- Masking logits to only allow specific token IDs prevents the model from generating invalid characters during inference
- Leveraging a quantized 32B model with vLLM enables fast, memory-efficient zero-shot classification without requiring fine-tuning or custom training loops
- Structuring prompts with explicit correctness labels and student explanations provides clear contextual signals for misconception classification
- Fine-tuning LLMs with PEFT and a holdout split offers a fast, reproducible baseline
- Rank-based ensemble scoring effectively aggregates diverse model predictions without requiring probability calibration or complex voting logic
- Structured prompt engineering (combining question, answer, correctness, and explanation) provides explicit context for sequence classification
- Gradient accumulation with a small batch size effectively simulates larger batches while staying within GPU memory limits
- Custom metrics like map@3 require manual implementation and integration into the Trainer via compute_metrics
- Regex-based feature extraction can quickly capture domain-specific patterns without heavy NLP pipelines
- Injecting a derived correctness flag into the LLM prompt significantly improves performance
- Combining diverse sentence embeddings and TF-IDF features creates a strong tree-based baseline
- Weighted voting over top-k predictions effectively leverages heterogeneous model strengths

## Critical findings
- The reported CV score is actually derived from a single 20% holdout split, not a true cross-validation process
- Kaggle's T4 GPUs do not support bf16, requiring fp16 for inference to avoid errors and improve speed
- Kaggle's T4 GPUs do not support bf16, necessitating fp16 for inference to avoid errors
- The notebook approximates ground-truth answers by assuming the most frequently selected multiple-choice option per question is correct

## Notable individual insights
- 769 (ENSEMBLE [GEMMA, QWEN, DEEPSEEK]): Parallel multi-GPU inference significantly reduces runtime for large LLMs, and extracting top-k probabilities allows fine-grained control over ensemble scoring.
- 592 (Gemma2-9B-it - CV 0.945): Kaggle's T4 GPUs lack bf16 support, making fp16 necessary for inference speed and stability, while LoRA adapters enable efficient fine-tuning on constrained VRAM.
- 340 ([LB 0.947] The Art of Ensemble v2): Mapping each target class to a unique special character forces the LLM to output exactly one token, preventing format errors during zero-shot classification.
- 231 (Experiment No - 1): Gradient accumulation with small batch sizes effectively simulates larger batches within GPU memory limits, and regex-based features quickly capture domain patterns without heavy NLP pipelines.
- 215 ([LB 0.944] The Art of Ensemble): Injecting a derived correctness flag into the LLM prompt significantly improves performance, and combining diverse sentence embeddings with TF-IDF creates a strong tree-based baseline.
- 371 (ModernBERT Large - CV 0.938): Standard classification metrics should be replaced with competition-specific metrics like MAP@3 for proper model selection, as prompt structure directly impacts validation scores.

## Notebooks indexed
- #769 votes [[notebooks/votes_01_kishanvavdara-ensemble-gemma-qwen-deepseek/notebook|ENSEMBLE [GEMMA, QWEN, DEEPSEEK]]] ([kaggle](https://www.kaggle.com/code/kishanvavdara/ensemble-gemma-qwen-deepseek))
- #592 votes [[notebooks/votes_02_cdeotte-gemma2-9b-it-cv-0-945/notebook|Gemma2-9B-it - CV 0.945]] ([kaggle](https://www.kaggle.com/code/cdeotte/gemma2-9b-it-cv-0-945))
- #481 votes [[notebooks/votes_03_cdeotte-ettin-encoder-1b-cv-0-943/notebook|Ettin-Encoder-1B - CV 0.943]] ([kaggle](https://www.kaggle.com/code/cdeotte/ettin-encoder-1b-cv-0-943))
- #371 votes [[notebooks/votes_04_cdeotte-modernbert-large-cv-0-938/notebook|ModernBERT Large - CV 0.938]] ([kaggle](https://www.kaggle.com/code/cdeotte/modernbert-large-cv-0-938))
- #371 votes [[notebooks/votes_05_cdeotte-deberta-starter-cv-0-930/notebook|Deberta Starter - CV 0.930+]] ([kaggle](https://www.kaggle.com/code/cdeotte/deberta-starter-cv-0-930))
- #340 votes [[notebooks/votes_06_bibanh-lb-0-947-the-art-of-ensemble-v2/notebook|[LB 0.947] The Art of Ensemble v2]] ([kaggle](https://www.kaggle.com/code/bibanh/lb-0-947-the-art-of-ensemble-v2))
- #300 votes [[notebooks/votes_07_aleaiest-lb-0-945-qwen2-5-32b-gptq/notebook|[LB 0.945] qwen2.5-32b-gptq]] ([kaggle](https://www.kaggle.com/code/aleaiest/lb-0-945-qwen2-5-32b-gptq))
- #263 votes [[notebooks/votes_08_maverickss26-map-charting-student-math-misunderstanding-v1/notebook|MAP - Charting student math misunderstanding v1]] ([kaggle](https://www.kaggle.com/code/maverickss26/map-charting-student-math-misunderstanding-v1))
- #231 votes [[notebooks/votes_09_codingloading-experiment-no-1/notebook|Experiment No - 1]] ([kaggle](https://www.kaggle.com/code/codingloading/experiment-no-1))
- #215 votes [[notebooks/votes_10_bibanh-lb-0-944-the-art-of-ensemble/notebook|[LB 0.944] The Art of Ensemble]] ([kaggle](https://www.kaggle.com/code/bibanh/lb-0-944-the-art-of-ensemble))
