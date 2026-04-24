# ai-mathematical-olympiad-progress-prize-3: top public notebooks

The community's top-voted notebooks for this mathematical Olympiad competition focus almost exclusively on high-throughput LLM inference pipelines and tool-augmented reasoning rather than traditional EDA or feature engineering. Baselines are minimal, with the primary architectural focus on serving open-weight models like gpt-oss-120b and Qwen3-32B-FP8 via vLLM under strict GPU constraints. Performance gains are driven by dynamic time budgeting, parallel multi-attempt generation, and sophisticated entropy-weighted or log-weighted voting strategies for answer aggregation.

## Common purposes
- inference
- tutorial

## Competition flows
- Loads test problems via the official Kaggle inference server wrapper, processes them through a parallelized vLLM-served LLM with Python tool-augmented reasoning, and returns formatted integer answers.
- The script reads test data via the Kaggle inference server wrapper, passes problem text to a dummy prediction function, and returns a formatted DataFrame for submission.
- Loads GPT-OSS-120B weights, serves them via a local vLLM server, processes test problems through a multi-attempt tool-use loop with persistent Jupyter sandboxes, aggregates results via entropy-weighted voting, and submits predictions using Kaggle's evaluation server.
- Reads test problems from a CSV file, processes each through a locally hosted LLM with a multi-attempt reasoning and verification loop, and outputs predictions in the required Kaggle submission format.
- Loads reference CSV data, initializes a vLLM server for GPT-OSS-120B, processes each math problem through a dynamic time-budgeted solver with parallel attempts and tool-use, and returns predictions via a Kaggle inference server.
- Loads test problems via Kaggle's inference server, processes them through a multi-round vLLM inference pipeline with Python code execution and prompt ensembling, and returns a single integer prediction per problem.
- Loads the gpt-oss-120b model, starts a local vLLM server, and processes test problems by generating multiple tool-augmented reasoning attempts, selecting the final answer via a custom weighted entropy voting scheme, and submitting predictions through the Kaggle inference server.
- Loads competition questions from a reference CSV, serves a 120B parameter LLM via vLLM, generates solutions through a multi-turn reflection and voting pipeline, and submits predictions via the official Kaggle inference server.
- Reads competition questions from a parquet/CSV file, processes them through a locally served 120B LLM with parallel tool-use attempts and entropy-based answer selection, and outputs predictions via a Kaggle inference server wrapper.
- Loads competition questions from a CSV, serves a local 120B-parameter LLM via vLLM, solves each problem using parallel tool-integrated reasoning with dynamic time budgeting, and submits answers via the Kaggle evaluation server.

## Data reading
- Uses Harmony encoding for tokenization and applies a custom chat template with system prompts, tool configurations, and mathematical reasoning instructions.
- The inference server wrapper handles loading the test CSV; the predict function unpacks single-row inputs using polars Series methods.
- Relies on Kaggle's evaluation framework to parse the test CSV; uses Polars for output DataFrame formatting.
- Loads test data from `/kaggle/input/ai-mathematical-olympiad-progress-prize-3/test.csv` using Polars, extracting the `id` and `question` columns to feed into the inference function.
- Reads reference.csv using pandas, extracts ground truth answers if available, and uses polars for input/output DataFrames.
- Receives input via Kaggle's inference server as `pl.DataFrame` objects containing `id` and `question` columns; extracts the raw problem string using `.item(0)`.
- Reads question IDs and text from /kaggle/input/ai-mathematical-olympiad-progress-prize-3/reference.csv using pandas, then converts to polars for the prediction function.
- Loads competition data using pandas and polars from a parquet file, extracting problem indices and question text; ground truth answers are loaded into a dictionary for local accuracy tracking.
- Loads questions and optional ground truth from /kaggle/input/ai-mathematical-olympiad-progress-prize-3/reference.csv using pandas.

## Data processing
- None explicitly shown; raw problem text is passed directly to the prediction function without preprocessing.
- None explicitly shown; focuses entirely on model serving and inference logic.
- Tokenization via openai_harmony.load_harmony_encoding for gpt-oss compatibility. Streaming logprobs are collected and processed to compute a custom weighted entropy metric for reasoning quality assessment.
- Applies regex extraction to parse `oxed{}` answers and Python code blocks from model outputs; prepends standard mathematical imports (`math`, `numpy`, `sympy`) to extracted code snippets before execution.
- No explicit preprocessing; raw question text is passed directly to the LLM.
- Drops the answer column for submission formatting, renames columns, and constructs prompts by concatenating the raw question with a structured system prompt, tool configuration, and mathematical preference instructions.
- Drops the answer column from the reference CSV to prepare the submission format; no other preprocessing or cleaning is applied.

## Features engineering
- No hand-crafted tabular features; pure LLM inference pipeline.

## Models
- gpt-oss-120b
- None (dummy model that always returns 0)
- Qwen3-32B-FP8

## Frameworks used
- vllm
- openai
- openai_harmony
- transformers
- polars
- kaggle_evaluation
- pandas
- jupyter_client
- torch
- cachetools
- numpy
- kaggle_evaluation.aimo_3_inference_server

## Ensembling
- Generates multiple parallel reasoning attempts per problem and aggregates results using an entropy-weighted voting scheme that prioritizes answers from low-entropy (high-confidence) generations.
- Aggregates answers from 8 parallel generation attempts using entropy-weighted voting, with early stopping triggered when 4 identical answers are found.
- Combines up to 12 parallel generation attempts per problem, filters candidates using entropy-weighted voting, applies early stopping upon unanimous agreement, and runs a final model-based verification step before returning the result.
- Aggregates answers from 8 parallel attempts per problem using majority voting, with Python execution count as a secondary tiebreaker, and implements early stopping once a majority threshold is reached.
- Combines outputs from five distinct system prompts and multiple inference rounds via majority voting over both boxed text answers and Python execution results, with a fallback to a default guess if no valid answers are extracted.
- Aggregates multiple generation attempts per problem by weighting candidate answers with the inverse of their computed weighted entropy and using a vote count, with early stopping triggered when a threshold of identical answers is reached.
- Combines 4 parallel generations using different system prompts, applies a log-weighted voting mechanism to select the final answer, and forces a fallback answer if no consensus is reached within time limits.
- Aggregates answers from up to 8 parallel generation attempts per problem using an entropy-weighted voting scheme, with an early stopping mechanism triggered when a threshold of identical answers is reached.
- Majority voting across 8 parallel samples from the same model, with early stopping triggered when a threshold of identical answers is reached, followed by tie-breaking by selecting the largest answer.

## Insights
- Persistent Jupyter kernel pools drastically reduce overhead for tool-augmented LLM reasoning compared to spawning new processes per call.
- Token-level entropy serves as a reliable proxy for model confidence, enabling effective answer filtering without external validators.
- Dynamic time budgeting based on remaining problems and early stopping on consensus prevents timeout failures in long-running inference loops.
- The notebook demonstrates the exact interface required for competition submissions, emphasizing lazy model loading to avoid timeout limits during initialization.
- It shows how to safely detect the execution environment to switch between local testing and production submission.
- Readers learn that the predict function must handle single-row inputs and return a properly formatted DataFrame.
- Preloading model weights into the OS page cache and using fp8_e4m3 KV cache dtype enables running 120B parameter models on a single Kaggle GPU.
- Dynamic time budgeting per problem based on remaining notebook time prevents timeout failures across the entire test set.
- Multi-attempt generation with entropy weighting improves answer reliability over single-pass inference.
- Dynamic tool use via Jupyter kernels allows the model to verify calculations without external APIs.
- Time budgeting and early stopping are critical for managing GPU memory and meeting competition submission deadlines.
- Entropy-based voting outperforms simple majority voting by penalizing high-uncertainty predictions.
- Parallelizing multiple attempts with dynamic time allocation improves solution reliability over single-pass generation.
- Isolating code execution in a pooled Jupyter kernel sandbox prevents state leakage and prevents runtime errors from crashing the main solver.
- Majority voting combined with execution metrics effectively filters out hallucinated or unverified answers.
- Using multiple distinct system prompts creates diverse reasoning paths that improve answer extraction reliability.
- Executing generated Python code via a sandboxed subprocess significantly boosts accuracy for mathematical problems compared to relying solely on text generation.
- Majority voting across heterogeneous answer sources effectively mitigates single-completion hallucinations.
- vLLM's `max_num_seqs` and `gpu_memory_utilization` parameters must be tuned to balance throughput and OOM risks on Kaggle GPUs.
- Position-weighted entropy and consistency penalties effectively measure reasoning quality better than raw token probabilities.
- Inverse-entropy weighted voting outperforms simple majority voting by prioritizing confident, coherent reasoning chains.
- Preloading model weights into the OS page cache significantly reduces inference startup latency.
- Streaming inference with dynamic follow-up prompts (e.g., asking for verification or a guess) improves answer extraction reliability.
- Log-weighted voting across multiple reasoning paths outperforms simple majority voting for complex math problems.
- Serving large models locally via vLLM with an OpenAI-compatible API allows flexible, low-latency interaction during competition inference.
- Smaller numerical answers tend to be incorrect in this competition, so the voting weight is adjusted using math.log(1.25 + abs(value)).
- GPU KV cache usage monitoring is used to trigger verification prompts when memory pressure is low.
- Parallel multi-attempt generation with dynamic time budgeting significantly improves mathematical reasoning accuracy under strict time constraints.
- Using a persistent Jupyter sandbox for tool execution allows the LLM to perform reliable symbolic and numerical verification without external API dependencies.
- Entropy-weighted voting outperforms simple majority voting by downweighting low-confidence (high-entropy) model outputs during answer selection.
- Preloading model weights into the OS page cache and using fp8 kv-cache dtype reduces inference latency and memory overhead on a single GPU.
- Dynamic time budgeting with rollover from early stops efficiently redistributes saved time to remaining problems.
- Pooling stateful Jupyter kernels ensures reliable, low-overhead Python tool execution for mathematical verification.
- Parallel sampling with early majority voting significantly reduces inference latency while preserving answer accuracy.

## Critical findings
- Smaller numerical answers tend to be incorrect in this competition, so the voting weight is adjusted using math.log(1.25 + abs(value)).
- GPU KV cache usage monitoring is used to trigger verification prompts when memory pressure is low.

## Notable individual insights
- 2957 ([44/50] LET ME (over)COOK!!!): Persistent Jupyter kernel pools drastically reduce overhead for tool-augmented LLM reasoning compared to spawning new processes per call.
- 1017 (AIMO 3 | GPT-OSS-120B (with tools)): Preloading model weights into the OS page cache and using fp8_e4m3 KV cache dtype enables running 120B parameter models on a single Kaggle GPU.
- 449 (AIMO3: streaming-inference reflexion prompting): Smaller numerical answers tend to be incorrect in this competition, so the voting weight is adjusted using math.log(1.25 + abs(value)).
- 615 ([40/50]GPT-OSS-120B TIR+DynamicTime+KernelPool): Isolating code execution in a pooled Jupyter kernel sandbox prevents state leakage and prevents runtime errors from crashing the main solver.
- 583 ([43/50] AIMO 3: gpt-oss-120b weighted entropy): Position-weighted entropy and consistency penalties effectively measure reasoning quality better than raw token probabilities.
- 615 (AIMO 3 Submission Demo Notebook 2/2): vLLM's max_num_seqs and gpu_memory_utilization parameters must be tuned to balance throughput and OOM risks on Kaggle GPUs.

## Notebooks indexed
- #2957 votes [[notebooks/votes_01_nihilisticneuralnet-44-50-let-me-over-cook/notebook|[44/50] LET ME (over)COOK!!!]] ([kaggle](https://www.kaggle.com/code/nihilisticneuralnet/44-50-let-me-over-cook))
- #1363 votes [[notebooks/votes_02_ryanholbrook-aimo-3-submission-demo/notebook|AIMO 3 Submission Demo]] ([kaggle](https://www.kaggle.com/code/ryanholbrook/aimo-3-submission-demo))
- #1017 votes [[notebooks/votes_03_andreasbis-aimo-3-gpt-oss-120b-with-tools/notebook|AIMO 3 | GPT-OSS-120B (with tools)]] ([kaggle](https://www.kaggle.com/code/andreasbis/aimo-3-gpt-oss-120b-with-tools))
- #752 votes [[notebooks/votes_04_amanatar-ans-verifys/notebook|ans_verifys]] ([kaggle](https://www.kaggle.com/code/amanatar/ans-verifys))
- #615 votes [[notebooks/votes_05_zaynyu-40-50-gpt-oss-120b-tir-dynamictime-kernelpool/notebook|[40/50]GPT-OSS-120B TIR+DynamicTime+KernelPool]] ([kaggle](https://www.kaggle.com/code/zaynyu/40-50-gpt-oss-120b-tir-dynamictime-kernelpool))
- #615 votes [[notebooks/votes_06_friederrr-aimo-3-submission-demo-notebook-2-2/notebook|AIMO 3 Submission Demo Notebook 2/2]] ([kaggle](https://www.kaggle.com/code/friederrr/aimo-3-submission-demo-notebook-2-2))
- #583 votes [[notebooks/votes_07_nihilisticneuralnet-43-50-aimo-3-gpt-oss-120b-weighted-entropy/notebook|[43/50] AIMO 3: gpt-oss-120b weighted entropy]] ([kaggle](https://www.kaggle.com/code/nihilisticneuralnet/43-50-aimo-3-gpt-oss-120b-weighted-entropy))
- #449 votes [[notebooks/votes_08_nihilisticneuralnet-aimo3-streaming-inference-reflexion-prompting/notebook|AIMO3: streaming-inference reflexion prompting]] ([kaggle](https://www.kaggle.com/code/nihilisticneuralnet/aimo3-streaming-inference-reflexion-prompting))
- #413 votes [[notebooks/votes_09_shelterw-15-15-aime-2026-i-120b-in-20mins/notebook|[15/15] AIME 2026 I | 120b in 20mins]] ([kaggle](https://www.kaggle.com/code/shelterw/15-15-aime-2026-i-120b-in-20mins))
- #404 votes [[notebooks/votes_10_shreyansh01m-41-50-same-as-parthenos-s-no-modifications/notebook|[41/50]Same as parthenos's, no modifications.]] ([kaggle](https://www.kaggle.com/code/shreyansh01m/41-50-same-as-parthenos-s-no-modifications))
