# ai-mathematical-olympiad-progress-prize-3: top public notebooks

The top public notebooks focus almost exclusively on high-performance inference pipelines for the AI Mathematical Olympiad Progress Prize 3, leveraging large language models like GPT-OSS-120B and Qwen3-32B-FP8 served via vLLM. A dominant theme involves Tool-Integrated Reasoning with persistent Jupyter kernel sandboxes for Python execution, combined with parallel decoding strategies that aggregate results through entropy-weighted voting, majority voting, or log-scaled consensus mechanisms. Community solutions also emphasize strict resource management through dynamic time budgeting, OS page cache preloading, and early stopping to maximize accuracy within Kaggle's inference server constraints.

## Common purposes
- inference
- utility
- tutorial

## Models
- gpt-oss-120b
- Qwen3-32B-FP8

## Frameworks
- vllm
- openai_harmony
- jupyter_client
- transformers
- kaggle_evaluation
- pandas
- polars
- openai
- torch
- numpy
- cachetools
- concurrent.futures

## Preprocessing
- Harmony tokenization
- regex-based answer extraction
- apply_chat_template
- adding standard imports to extracted code
- filtering answers to integers in [0, 99999]

## Ensemble patterns
- Entropy-weighted voting across parallel attempts with early stopping
- Majority voting with Python execution call count tiebreaker
- Multi-prompt diversity using 5 distinct system prompts combined with majority voting over oxed{} and code answers
- Log-scaled weighted voting with margin thresholding
- Parallel sampling with early stopping and majority voting

## Post-processing
- Regex extraction of oxed{} or "final answer is" patterns
- Entropy-weighted voting across parallel attempts
- Early stopping based on consensus or vote thresholds
- Validation of answers to integer range [0, 99999]
- Fallback to default answer when no valid output is found
- Self-verification via secondary zero-temperature LLM call
- Log-scaled reweighting and margin thresholding
- Tie-breaking by largest value
- Tie-breaking by Python execution call counts

## Critical findings
- Counter.most_common breaks ties by order of first occurrence, so pre-sorting valid answers is needed for deterministic results
- max_tokens accounts for prompt length, meaning longer inputs directly reduce the model's generation capacity

## Notable techniques
- Persistent Jupyter kernel sandbox for stateful tool execution
- fp8_e4m3 KV-cache quantization in vLLM
- Entropy-weighted voting across parallel attempts
- OS page cache preloading of model weights via parallel file reads
- Dynamic time budgeting based on remaining problems and global limits
- Harmony encoding/formatting for tokenization and tool-augmented inference
- Lazy model loading inside predict function to bypass startup timeout
- Official kaggle_evaluation server API for submission and testing
- Token-level entropy calculation from top_logprobs for confidence scoring
- min_p sampling (0.01/0.02) combined with temperature for generation control
- Self-verification via secondary zero-temperature LLM call
- Majority voting with Python execution call count tiebreaker
- Streaming completion with early oxed{} detection
- Prefix caching in vLLM
- Tool-Integrated Reasoning via subprocess execution with automatic import injection
- Multi-prompt diversity using distinct system prompts
- vLLM CUDA graph optimization via TRITON_PTXAS_PATH
- Iterative early-stopping generation loop filtering completed histories
- Custom weighted entropy metric with position weighting, variance penalty, and confidence streak bonus
- Log-scaled weighted voting with margin thresholding
- GPU KV cache usage monitoring to trigger verification prompts
- Granular cutoff-based timeout management
- Inverse-entropy weighted voting
- Dynamic time budgeting with rollover from early stops
- Dynamic per-step timeout computation for vLLM and Python execution
- Regex extraction of oxed{} or "final answer is" patterns
- Validation of answers to integer range [0, 99999]
- Fallback to default answer when no valid output is found

## Notable individual insights
- votes 1363 (AIMO 3 Submission Demo): Lazy model loading inside the predict function bypasses Kaggle's 15-minute startup timeout.
- votes 615 (AIMO 3 Submission Demo Notebook 2/2): Counter.most_common breaks ties by first occurrence order, requiring pre-sorted valid answers for deterministic results.
- votes 615 (AIMO 3 Submission Demo Notebook 2/2): max_tokens accounts for prompt length, so longer inputs directly reduce generation capacity.
- votes 583 ([43/50] AIMO 3: gpt-oss-120b weighted entropy): Custom weighted entropy metric combines position weighting, variance penalty, sustained uncertainty penalty, and confidence streak bonus.
- votes 449 (AIMO3: streaming-inference reflexion prompting): GPU KV cache usage monitoring can trigger dynamic verification prompts during inference.
- votes 404 ([41/50]Same as parthenos's, no modifications.): Dynamic time budgeting with rollover redistributes unused time from early stops to remaining problems.

## Notebooks indexed
- #2955 votes [[notebooks/votes_01_nihilisticneuralnet-44-50-let-me-over-cook/notebook|[44/50] LET ME (over)COOK!!!]] ([kaggle](https://www.kaggle.com/code/nihilisticneuralnet/44-50-let-me-over-cook))
- #1363 votes [[notebooks/votes_02_ryanholbrook-aimo-3-submission-demo/notebook|AIMO 3 Submission Demo]] ([kaggle](https://www.kaggle.com/code/ryanholbrook/aimo-3-submission-demo))
- #1017 votes [[notebooks/votes_03_andreasbis-aimo-3-gpt-oss-120b-with-tools/notebook|AIMO 3 | GPT-OSS-120B (with tools)]] ([kaggle](https://www.kaggle.com/code/andreasbis/aimo-3-gpt-oss-120b-with-tools))
- #752 votes [[notebooks/votes_04_amanatar-ans-verifys/notebook|ans_verifys]] ([kaggle](https://www.kaggle.com/code/amanatar/ans-verifys))
- #615 votes [[notebooks/votes_05_zaynyu-40-50-gpt-oss-120b-tir-dynamictime-kernelpool/notebook|[40/50]GPT-OSS-120B TIR+DynamicTime+KernelPool]] ([kaggle](https://www.kaggle.com/code/zaynyu/40-50-gpt-oss-120b-tir-dynamictime-kernelpool))
- #615 votes [[notebooks/votes_06_friederrr-aimo-3-submission-demo-notebook-2-2/notebook|AIMO 3 Submission Demo Notebook 2/2]] ([kaggle](https://www.kaggle.com/code/friederrr/aimo-3-submission-demo-notebook-2-2))
- #583 votes [[notebooks/votes_07_nihilisticneuralnet-43-50-aimo-3-gpt-oss-120b-weighted-entropy/notebook|[43/50] AIMO 3: gpt-oss-120b weighted entropy]] ([kaggle](https://www.kaggle.com/code/nihilisticneuralnet/43-50-aimo-3-gpt-oss-120b-weighted-entropy))
- #449 votes [[notebooks/votes_08_nihilisticneuralnet-aimo3-streaming-inference-reflexion-prompting/notebook|AIMO3: streaming-inference reflexion prompting]] ([kaggle](https://www.kaggle.com/code/nihilisticneuralnet/aimo3-streaming-inference-reflexion-prompting))
- #413 votes [[notebooks/votes_09_shelterw-15-15-aime-2026-i-120b-in-20mins/notebook|[15/15] AIME 2026 I | 120b in 20mins]] ([kaggle](https://www.kaggle.com/code/shelterw/15-15-aime-2026-i-120b-in-20mins))
- #404 votes [[notebooks/votes_10_shreyansh01m-41-50-same-as-parthenos-s-no-modifications/notebook|[41/50]Same as parthenos's, no modifications.]] ([kaggle](https://www.kaggle.com/code/shreyansh01m/41-50-same-as-parthenos-s-no-modifications))
