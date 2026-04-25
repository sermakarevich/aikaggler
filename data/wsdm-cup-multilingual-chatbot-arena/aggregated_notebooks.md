# wsdm-cup-multilingual-chatbot-arena: top public notebooks

The community's top notebooks primarily focus on efficient inference pipelines for multilingual chatbot arena tasks, heavily leveraging 4-bit quantized LLMs (Gemma-2 9B, Qwen2-7B) with LoRA adapters, dynamic batching, and multi-GPU parallelization. A subset of notebooks establishes classical ML baselines using hand-crafted text statistics, TF-IDF embeddings, and gradient boosting models (CatBoost, LightGBM, XGBoost) augmented with response-swapping symmetry. Across both approaches, practitioners emphasize test-time augmentation via response swapping, prompt engineering, and straightforward statistical feature extraction to stabilize predictions and optimize GPU memory.

## Common purposes
- inference
- baseline

## Competition flows
- Loads test parquet data, tokenizes multilingual prompts/responses, runs parallel inference on a 4-bit quantized Gemma-2 9B LoRA model, and outputs a CSV submission.
- Reads test data, preprocesses with custom formatting, runs inference across two GPUs, applies TTA via response swapping, and generates a CSV submission.
- Loads test data, formats prompts/responses, tokenizes with custom truncation, runs parallel dual-GPU inference, and exports a CSV submission.
- Loads test data via PyArrow, merges two LoRA adapters, runs parallel dual-GPU inference with TTA, and outputs a CSV submission.
- Loads test data, formats prompts for Gemma2-9B and Llama3-8B, runs manual pipeline parallelism across two GPUs, and exports a winner CSV.
- Loads multilingual prompts/responses, formats evaluation prompts, runs batched inference via vLLM, parses generated text, and exports a CSV submission.
- Loads test data, preprocesses/tokenizes with custom tags, runs inference on a 4-bit Gemma-2 9B LoRA model across two GPUs, applies TTA, and generates a CSV submission.
- Reads test parquet data, tokenizes prompts/responses, runs parallel inference on two GPUs with a quantized Gemma2-9B-IT LoRA adapter, applies TTA, and outputs a CSV submission.
- Loads parquet train/test data, applies symmetry augmentation and hand-crafted features, trains CatBoost/LightGBM/XGBoost via 5-fold CV, blends OOF predictions with Logistic Regression, tunes thresholds, and generates a submission CSV.
- Loads train/test parquet files, computes statistical text features and sentiment scores, applies TF-IDF vectorization, trains a LightGBM classifier on a holdout set, and generates a formatted submission CSV.

## Data reading
- pd.read_parquet('/kaggle/input/wsdm-cup-multilingual-chatbot-arena/test.parquet')
- pl.read_parquet('/kaggle/input/wsdm-cup-multilingual-chatbot-arena/test.parquet').to_pandas()
- pd.read_parquet from Kaggle datasets (train.parquet or test.parquet), with NaN values filled with empty strings.
- Reads test.parquet using PyArrow, converts to pandas DataFrame, and fills missing values with empty strings.
- pandas.read_parquet on test.parquet (or train.parquet for local testing) with dummy labels injected for local validation
- pd.read_parquet with engine='pyarrow' for both train and test datasets, explicitly selecting columns ['id', 'prompt', 'response_a', 'response_b', 'scored']
- pd.read_parquet for train and test sets, pd.read_csv for sample submission

## Data processing
- Replaces 'null' strings or fills NaN values with empty strings
- Strips whitespace
- Adds custom prompt/response tags during tokenization
- Implements dynamic padding by sorting inputs by length
- Splits inputs across two GPU streams or halves for parallel processing
- Applies test-time augmentation (TTA) by swapping response order and averaging probabilities
- Truncates long texts using token offsets or character limits, often inserting a '(snip)' marker
- Formats inputs with custom tags (<prompt>, <response_a>, <response_b>)
- Tokenizes with GemmaTokenizerFast or AutoTokenizer
- Custom string formatting for prompts and responses
- Truncates prompts to fixed lengths (e.g., 512 tokens)
- Splits remaining context equally between responses
- Adds BOS/EOS tokens
- Constructs conversation templates with system/user roles
- Manages variable-length batching with token-sharded collators
- Prepares response swapping for TTA
- Truncates test responses to fixed character counts (e.g., 5000)
- Applies prompt templating with system/user roles
- Chunks prompts into fixed batch sizes for inference
- Configures deterministic sampling parameters (temperature=0, seed=777)
- Splits batches by length thresholds to optimize dynamic batching
- Duplicates training rows by swapping response columns and flipping labels for symmetry
- Drops raw text, IDs, and metadata columns before modeling
- Computes TextBlob sentiment polarity for text columns
- Applies sublinear TF-IDF vectorization with character and word analyzers

## Features engineering
- Hand-crafted string features (length, whitespace, punctuation, special character counts, bracket balance, digit/letter ratios, Chinese character presence, JSON keyword detection)
- TextBlob sentiment polarity
- Character and word-level TF-IDF embeddings

## Models
- Gemma-2 9B (4-bit quantized)
- LoRA adapter (PEFT)
- Gemma2ForSequenceClassification
- Llama3-8B
- Qwen2-7B-Instruct
- CatBoost
- LightGBM
- XGBoost
- Logistic Regression

## Frameworks used
- transformers
- peft
- accelerate
- bitsandbytes
- torch
- pandas
- numpy
- scikit-learn
- polars
- pyarrow
- xformers
- vllm
- lightgbm
- xgboost
- joblib
- matplotlib
- seaborn
- textblob

## Loss functions
- Logloss
- binary:logistic
- binary_logloss

## CV strategies
- StratifiedKFold (5-fold)
- Holdout split (15%)

## Ensembling
- Averages prediction probabilities from original and TTA-swapped response pairs, then applies a 0.5 threshold.
- Merges two LoRA adapter weights via weighted average (0.7/0.3), then applies TTA by swapping response roles and averaging predictions with weights 0.45/0.55.
- Attempts to average predictions from Gemma2 and Llama3 with weights [2, 1] and swap model B predictions (commented out).
- Averages predictions from original and TTA-swapped response pairs, and combines outputs from two GPU models and two batch size configurations.
- Averages probability outputs from original and TTA passes, then applies an argmax rule to determine the final winner.
- Blends base model OOF predictions using a Logistic Regression model trained on competition data, with weights derived from fitted coefficients.

## Insights
- Keeping a LoRA adapter unmerged during inference prevents non-negligible quantization errors that occur when naively merging it into a 4-bit base model.
- Sorting inputs by length and splitting them across two GPU streams enables dynamic padding and parallel inference, significantly reducing latency.
- Test-time augmentation by swapping response order and averaging probabilities can improve prediction robustness, though it is optional.
- Swapping response positions during inference acts as effective TTA and improves prediction stability without retraining.
- Loading the base model on two separate GPUs and processing batches in parallel with ThreadPoolExecutor significantly reduces total inference time.
- Splitting a dataset by sequence length and processing halves in parallel maximizes GPU throughput while preventing OOM errors during dynamic batching.
- Dynamic padding combined with length-based sorting significantly reduces inference latency.
- Parallelizing inference across two GPUs using ThreadPoolExecutor maximizes hardware utilization.
- Swapping response roles during test-time augmentation provides a simple but effective way to improve prediction robustness.
- Weighted averaging of multiple LoRA adapters can effectively combine specialized fine-tuning directions.
- Manual pipeline parallelism across two GPUs can be implemented by splitting layer forward passes and transferring hidden states between devices.
- Variable-length batching with token-sharded collators effectively manages VRAM during inference.
- Swapping response order can help verify prediction consistency or act as a simple test-time augmentation.
- Using vLLM with tensor parallelism and chunked batching enables efficient inference of large language models on limited GPU memory.
- Deterministic sampling parameters (temperature=0, seed=777) ensure reproducible pairwise comparisons for competition submissions.
- Prompt engineering that explicitly frames the model as a multilingual product experience evaluator improves response quality for arena-style tasks.
- Custom prompt formatting with explicit tags improves model comprehension for preference tasks.
- Swapping response candidates during inference effectively reduces prediction bias.
- Sorting inputs by length and using dynamic padding significantly optimizes GPU memory and inference speed.
- Splitting batches by length thresholds allows using larger batch sizes for shorter sequences, further accelerating inference.
- Dynamic padding combined with length-sorted batching significantly improves GPU utilization for long-context LLM inference.
- Test-time augmentation by swapping response order stabilizes probability estimates without requiring additional training or data.
- Multi-GPU inference can be efficiently parallelized using ThreadPoolExecutor alongside Hugging Face's dynamic padding utilities to avoid memory fragmentation.
- Using a smoother metric like AUC or logloss for early stopping is more effective than accuracy for gradient boosting models.
- Swapping response columns and flipping labels effectively doubles the training data while preserving the binary classification structure.
- Threshold tuning on OOF predictions can yield better accuracy than the default 0.5 cutoff.
- A modular, reusable training pipeline simplifies experimentation across different classical ML models.
- The target variable is well-balanced between model_a and model_b.
- The dataset is predominantly English, with significantly fewer samples in other languages.
- Prompt and response sentiments are largely neutral, indicating careful dataset curation to avoid bias.

## Critical findings
- The neutral sentiment distribution across prompts and responses confirms the dataset was carefully prepared to prevent sentiment-based bias in winner selection.

## Notable individual insights
- votes 452 (Gemma-2 9b 4-bit QLoRA - LB 0.63): Keeping a LoRA adapter unmerged during inference prevents non-negligible quantization errors that occur when naively merging it into a 4-bit base model.
- votes 329 (WSDM apply previous LMSYS solution): Weighted averaging of multiple LoRA adapters can effectively combine specialized fine-tuning directions.
- votes 229 ([WSDM CUP] lmsys 0805): Manual pipeline parallelism across two GPUs can be implemented by splitting layer forward passes and transferring hidden states between devices.
- votes 205 (Qwen2-7B-Instruct with VLLM==0.6.3 inference): Using vLLM with tensor parallelism and chunked batching enables efficient inference of large language models on limited GPU memory.
- votes 176 (WSDM2024|Baseline|V1): Swapping response columns and flipping labels effectively doubles the training data while preserving the binary classification structure.
- votes 161 (🏆 WSDM 🙋 Sentiment Analysis 📊 EDA): The neutral sentiment distribution across prompts and responses confirms the dataset was carefully prepared to prevent sentiment-based bias in winner selection.

## Notebooks indexed
- #452 votes [[notebooks/votes_01_mbmmurad-gemma-2-9b-4-bit-qlora-lb-0-63/notebook|Gemma-2 9b 4-bit QLoRA - LB 0.63]] ([kaggle](https://www.kaggle.com/code/mbmmurad/gemma-2-9b-4-bit-qlora-lb-0-63))
- #426 votes [[notebooks/votes_02_takaito-wsdm-cup-lb-0-684-only-gemma-2-9b-4-bit/notebook|[WSDM Cup] LB:0.684 Only Gemma-2 9b 4-bit]] ([kaggle](https://www.kaggle.com/code/takaito/wsdm-cup-lb-0-684-only-gemma-2-9b-4-bit))
- #393 votes [[notebooks/votes_03_hengck23-sft-gemma-2-9b-it-bnb-4bit/notebook|[sft] gemma-2-9b-it-bnb-4bit]] ([kaggle](https://www.kaggle.com/code/hengck23/sft-gemma-2-9b-it-bnb-4bit))
- #329 votes [[notebooks/votes_04_takamichitoda-wsdm-apply-previous-lmsys-solution/notebook|WSDM apply previous LMSYS solution]] ([kaggle](https://www.kaggle.com/code/takamichitoda/wsdm-apply-previous-lmsys-solution))
- #229 votes [[notebooks/votes_05_rin2401-wsdm-cup-lmsys-0805/notebook|[WSDM CUP] lmsys 0805]] ([kaggle](https://www.kaggle.com/code/rin2401/wsdm-cup-lmsys-0805))
- #225 votes [[notebooks/votes_06_flashai-qwen2-7b-instruct-with-vllm-0-6-3-inference/notebook|Qwen2-7B-Instruct with VLLM==0.6.3 inference]] ([kaggle](https://www.kaggle.com/code/flashai/qwen2-7b-instruct-with-vllm-0-6-3-inference))
- #205 votes [[notebooks/votes_07_qianxinyu-wsdm-cup-lb-0-684-only-gemma-2-9b-4-bit/notebook|[WSDM Cup] LB:0.684 Only Gemma-2 9b 4-bit]] ([kaggle](https://www.kaggle.com/code/qianxinyu/wsdm-cup-lb-0-684-only-gemma-2-9b-4-bit))
- #179 votes [[notebooks/votes_08_faykudbq-gemma-test01/notebook|Gemma_Test01]] ([kaggle](https://www.kaggle.com/code/faykudbq/gemma-test01))
- #176 votes [[notebooks/votes_09_ravi20076-wsdm2024-baseline-v1/notebook|WSDM2024|Baseline|V1]] ([kaggle](https://www.kaggle.com/code/ravi20076/wsdm2024-baseline-v1))
- #161 votes [[notebooks/votes_10_taimour-wsdm-sentiment-analysis-eda/notebook|🏆 WSDM 🙂 Sentiment Analysis 📊 EDA]] ([kaggle](https://www.kaggle.com/code/taimour/wsdm-sentiment-analysis-eda))
