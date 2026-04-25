# ai-mathematical-olympiad-progress-prize-2: top public notebooks

The community's top-voted notebooks exclusively focus on inference-only pipelines for mathematical reasoning, leveraging quantized large language models (e.g., Qwen, DeepSeek, QwQ) via vLLM. They emphasize prompt engineering, multi-sequence self-consistency voting, regex-based answer extraction, and strict latency management to comply with Kaggle's evaluation server constraints. Rather than training or feature engineering, the solutions prioritize efficient batched generation, iterative code execution, and robust post-processing to stabilize outputs under tight time limits.

## Common purposes
- inference

## Competition flows
- Loads a quantized Qwen-14B model via vLLM, generates multiple step-by-step reasoning traces per math problem using custom prompts, extracts and votes on the final answers, and submits them via Kaggle's inference server.
- Initializes a Kaggle evaluation server that receives mathematical questions and returns integer predictions, with local testing enabled via a provided gateway script.
- Loads math problems from a CSV, processes them through a quantized QwQ-32B-preview model using vLLM with prompt diversity and code execution, extracts answers via regex and majority voting, and submits the final integer modulo 1000 via the Kaggle inference server.
- Loads test questions via Kaggle's inference server, generates multiple reasoning traces using a quantized LLM via vLLM, extracts and votes on final answers, and returns predictions modulo 1000.
- Loads math problems via the competition inference server, generates parallel responses using a quantized DeepSeek-R1 model with diverse prompt templates, aggregates answers via frequency voting, and returns modulo 1000 predictions within strict latency limits.
- Loads the deepseek-r1-distill-qwen-7b-awq model via vLLM with tensor parallelism, generates multiple parallel reasoning traces per question using custom sampling parameters, aggregates the outputs via frequency voting with modulo 1000, and serves the results through the official AIMO 2 inference server.
- Loads test questions from CSV, applies 15 diverse system prompts to a quantized DeepSeek-R1-Distill-Qwen-14B model via vLLM, extracts final answers via regex, selects the most frequent result using a randomized counter, and submits predictions using Kaggle's official inference server.
- Receives mathematical questions, processes them through a multi-turn LLM inference loop with code execution and self-correction, aggregates answers via voting, and submits predictions via the Kaggle evaluation server.
- Loads the Qwen2.5-QwQ-32B-Preview model via vLLM, processes competition questions through a multi-turn reasoning loop with Python code execution, extracts and votes on candidate answers, and submits predictions via Kaggle's inference server.
- Reads math questions from a CSV, generates parallel completions using a quantized LLM via vLLM, extracts and aggregates answers via a noisy majority vote, and submits the final integer modulo 1000 through the official Kaggle inference server.

## Data reading
- Reads reference/test CSVs using pandas and polars, and relies on kaggle_evaluation.aimo_2_inference_server to stream question IDs and text during competition runtime.
- Passes the path to `/kaggle/input/ai-mathematical-olympiad-progress-prize-2/test.csv` to the local gateway helper for loading.
- Reads the reference dataset using pandas.read_csv, drops the ground truth column, and passes questions as polars.DataFrame items to the prediction function.
- Loads test/reference data through the kaggle_evaluation.aimo_2_inference_server which passes id_ and question as Polars DataFrames to a custom predict function; also locally reads reference.csv for local testing.
- Reads input questions as Polars DataFrames via kaggle_evaluation.aimo_2_inference_server.AIMO2InferenceServer; locally tests against a reference CSV.
- Loads test/reference data via polars DataFrame or pandas CSV depending on the context. Reads the reference.csv file to extract question text for local testing or submission generation.
- Reads test/reference CSV files using pandas and polars. Loads model weights and tokenizer from local Kaggle input directories.
- Reads competition questions from a Polars DataFrame passed to the `predict` function; locally tests against `/kaggle/input/ai-mathematical-olympiad-progress-prize-2/reference.csv`.
- Loads test/reference data using pandas and polars, extracting individual question strings via .item(0) for single-question processing.

## Data processing
- Formats questions into chat templates using tokenizer.apply_chat_template, appends a modulo 1000 instruction, and extracts final answers using regex on \boxed{} patterns and Python REPL outputs, applying modulo 1000 to all numeric results.
- Applies chat template formatting for prompts, extracts answers using regex matching for \boxed{} patterns, filters out invalid responses, and sorts generation outputs by token length for internal tracking.
- Extracts final answers using regex on \boxed{} tags; filters out generations without valid answers; applies a frequency-based voting function with slight random noise for tie-breaking; enforces modulo 1000 on final predictions.
- Applies a regex pattern to extract text enclosed in \boxed{} from model outputs. Filters out messages that do not contain a valid boxed answer. Converts extracted answers to integers, applying a small random jitter (random.random() / 1_000) to break ties in frequency counting. Applies modulo 1000 to the final selected answer.
- Tokenizes prompts via vLLM chat templates. Extracts final answers using regex on \boxed{...} tags. Applies a randomized frequency counter for answer selection. Implements time-based cutoffs with dynamic sequence/token limits to manage inference latency
- Extracts Python code blocks from LLM output using regex and wraps them with standard imports and dynamic variable printing. Executes extracted code in a sandboxed subprocess with timeout and error handling. Filters LLM responses for \boxed{} answers, detects contradictions, and applies a custom majority voting function with tie-breaking.
- String manipulation for prompt formatting, regex extraction of Python code and `oxed{}` answers, modulo 1000 conversion for large numbers, and majority voting on extracted candidates.
- None explicitly applied; raw question strings are directly formatted into chat templates.

## Models
- Qwen-14B-AWQ
- QwQ-32B-preview (AWQ)
- deepseek-r1-distill-qwen-7b-awq
- DeepSeek-R1-Distill-Qwen-14B-AWQ
- Qwen2.5-72B-Instruct (AWQ)
- Qwen2.5-QwQ-32B-Preview (AWQ)

## Frameworks used
- vllm
- torch
- pandas
- polars
- kaggle_evaluation.aimo_2_inference_server
- transformers
- numpy

## Ensembling
- Generates 16 parallel sequences per question using temperature and top-p sampling, extracts answers via regex, and aggregates them using a frequency counter with a minor random tie-breaker to select the final answer.
- Implements chain-of-thought self-consistency by generating 5 samples per problem with temperature=1 and 5 distinct prompts, extracts answers from both natural language and executed Python code, and selects the final answer via majority voting modulo 1000.
- Generates 32 parallel sequences per question, extracts all valid boxed answers, counts their frequencies with a tiny random perturbation for tie-breaking, and returns the most frequent answer modulo 1000.
- Generates 16 parallel responses across three prompt templates, aggregates via frequency voting with minor random tie-breaking, and applies modulo 1000 post-processing.
- Generates 32 parallel sequences per question, extracts boxed answers, counts their frequencies with a tiny random jitter to break ties, and selects the most frequent integer answer modulo 1000.
- Combines outputs from 15 diverse system prompts by extracting answers and selecting the most frequent result via a randomized frequency counter.
- Generates 16 parallel initial prompts, runs 4 iterative inference rounds with self-correction, and selects the final answer via a custom majority voting function with tie-breaking.
- Aggregates candidate answers extracted from both the model's natural language `oxed{}` outputs and executed Python code outputs, then selects the most frequent valid integer via majority voting as post-processing.
- Generates 128 parallel completions using a weighted mix of English and Chinese prompts, extracts \boxed{} answers, and selects the final prediction via a noisy majority vote (adding random noise to break ties) before applying modulo 1000.

## Insights
- Constraining the model to output answers modulo 1000 inside \boxed{} simplifies answer extraction and standardizes evaluation.
- Using AWQ quantization with vLLM enables running large models efficiently on limited GPU resources.
- Batch generation with temperature sampling and majority voting mitigates single-generation hallucination or formatting errors.
- Demonstrates the exact function signature and return format required for AIMO 2 submissions.
- Highlights the 15-minute server startup and 30-minute per-prediction latency constraints enforced by the evaluation gateway.
- Shows how to use the provided local gateway to validate submission logic before competition evaluation.
- QwQ-32B-preview's iterative self-reflection and verification mechanism makes it significantly more effective for complex mathematical reasoning than standard chain-of-thought prompting.
- Reducing the number of generated samples from 8 to 5 improves inference efficiency and prevents timeout errors without degrading accuracy.
- Combining temperature sampling with diverse prompt engineering increases response diversity, which is essential for reliable self-consistency voting.
- AWQ quantization enables running a 32B parameter model on limited GPU memory while preserving reasoning capabilities.
- Multi-sequence sampling combined with frequency voting effectively reduces hallucination in mathematical reasoning tasks.
- Modulo 1000 post-processing aligns with competition scoring rules and prevents formatting-related score loss.
- vLLM's tensor parallelism and GPU memory utilization settings are critical for running large quantized models within hardware constraints.
- Using diverse system prompts that explicitly encourage reverse thinking, abstraction, and step-by-step verification improves LLM reasoning on math problems.
- Parallel generation with frequency voting acts as an effective self-consistency ensemble without additional training.
- Modulo 1000 constraints require robust answer extraction and tie-breaking strategies to avoid deterministic voting failures.
- Deterministic answer voting can be brittle, so adding slight random noise to tie-breaking improves robustness.
- Latency constraints require dynamic token limits and early termination strategies to meet competition deadlines.
- LLMs may "think" in different languages and benefit from explicit instructions to recheck and abstract problems.
- Using vLLM with tensor parallelism across 4 GPUs significantly accelerates batched LLM inference for competition deadlines.
- Prompting the model to think step-by-step and strictly format outputs in \boxed{} enables reliable post-processing via regex.
- Frequency voting over multiple parallel generations acts as a lightweight ensemble that stabilizes LLM outputs without requiring complex calibration.
- Dynamically reducing max_tokens and max_num_seqs as the runtime cutoff approaches ensures the pipeline completes within strict time limits.
- Prompt diversity significantly impacts LLM reasoning performance on mathematical problems.
- Structured extraction of answers via regex is more reliable than parsing raw generated text.
- Time-based cutoffs and dynamic token limits are necessary to meet strict submission latency constraints.
- AWQ quantization enables efficient multi-GPU batched inference for large language models.
- Combining LLM generation with external code execution and iterative self-correction significantly improves reliability for mathematical reasoning tasks.
- Extracting and sandbox-executing intermediate Python code allows the pipeline to verify calculations and detect logical contradictions automatically.
- Parallel prompt generation followed by iterative filtering and answer aggregation yields more stable predictions than single-pass inference.
- Integrating external Python code execution with LLM reasoning significantly improves mathematical accuracy.
- Rotating multiple system prompts helps stabilize the model's reasoning style across different questions.
- Extracting answers from both textual `oxed{}` outputs and code execution results increases robustness against generation failures.
- Applying modulo 1000 to large numerical answers aligns with competition submission constraints and prevents overflow issues.
- Mixing multiple system prompts and adding a "think like a human" chain-of-thought prefix improves the model's ability to avoid decimal arithmetic and produce cleaner final answers.
- Adding a small random perturbation to vote counts effectively breaks ties and stabilizes the selection process without biasing the majority.
- Dynamically reducing max_tokens and max_num_seqs as the deadline approaches ensures predictions are submitted within the strict time limits.

## Critical findings
- The notebook's leaderboard score exhibits high variance due to stochastic sampling and vLLM generation, meaning it does not guarantee a consistent score on submission.
- Python code execution outputs are intentionally weighted double in the final voting process compared to natural language answers to prioritize computational verification.
- The author hardcodes a fallback answer (210) for specific question types during local testing to bypass slow generation and save time.
- Using min_p=0.05 alongside frequency_penalty=0.2 helps constrain the model's output distribution without overly suppressing valid mathematical steps.
- Deterministic answer voting can be brittle, so adding slight random noise to tie-breaking improves robustness.
- Latency constraints require dynamic token limits and early termination strategies to meet competition deadlines.
- LLMs may "think" in different languages and benefit from explicit instructions to recheck and abstract problems.
- The author hardcodes a fallback answer of 210 for questions that do not contain specific keywords (e.g., 'Triangle', 'delightful', 'George') during local testing, likely to bypass slow generation for known hard cases.
- The frequency counter uses a floating-point jitter (1 + random.random() / 1_000) to prevent exact ties, which is a practical workaround for deterministic voting in LLM outputs.
- The author hardcodes a fallback answer (210) for local testing when specific keywords like "circumcircle" or "Triangle" are missing, highlighting a practical shortcut for debugging.
- The code explicitly warns about and mitigates inference latency by tracking elapsed time and dynamically scaling batch sizes and token limits.

## What did not work
- Using 8 samples per problem was inefficient and risked exceeding the 5-hour runtime limit without improving accuracy.
- Prompts explicitly requesting Python code generation were removed because the model leverages chain-of-thought reasoning more effectively than direct code generation for this task.

## Notable individual insights
- 1396 (Deepseek-r1-distill-qwen-7b): Constraining the model to output answers modulo 1000 inside \boxed{} simplifies answer extraction and standardizes evaluation.
- 1097 ([LB 20] QWQ-32B-preview Optimized inference): Python code execution outputs are intentionally weighted double in the final voting process compared to natural language answers to prioritize computational verification.
- 1086 (AIMO 2: deepseek-r1-distill-qwen-7b-awq): The author hardcodes a fallback answer (210) for specific question types during local testing to bypass slow generation and save time.
- 583 (Qwen2.5-72B-Instruct with TIR): Combining LLM generation with external code execution and iterative self-correction significantly improves reliability for mathematical reasoning tasks.
- 516 (LUCKY): Mixing multiple system prompts and adding a "think like a human" chain-of-thought prefix improves the model's ability to avoid decimal arithmetic and produce cleaner final answers.
- 1040 ([LB24] Deepseek-R1 with more prompt Engineering): Deterministic answer voting can be brittle, so adding slight random noise to tie-breaking improves robustness.

## Notebooks indexed
- #1396 votes [[notebooks/votes_01_itahiro-deepseek-r1-distill-qwen-7b/notebook|Deepseek-r1-distill-qwen-7b]] ([kaggle](https://www.kaggle.com/code/itahiro/deepseek-r1-distill-qwen-7b))
- #1157 votes [[notebooks/votes_02_ryanholbrook-aimo-2-submission-demo/notebook|AIMO 2 Submission Demo]] ([kaggle](https://www.kaggle.com/code/ryanholbrook/aimo-2-submission-demo))
- #1097 votes [[notebooks/votes_03_mbmmurad-lb-20-qwq-32b-preview-optimized-inference/notebook|[LB 20] QWQ-32B-preview Optimized inference]] ([kaggle](https://www.kaggle.com/code/mbmmurad/lb-20-qwq-32b-preview-optimized-inference))
- #1086 votes [[notebooks/votes_04_yekenot-aimo-2-deepseek-r1-distill-qwen-7b-awq/notebook|AIMO 2: deepseek-r1-distill-qwen-7b-awq]] ([kaggle](https://www.kaggle.com/code/yekenot/aimo-2-deepseek-r1-distill-qwen-7b-awq))
- #1040 votes [[notebooks/votes_05_haoruili-lb24-deepseek-r1-with-more-prompt-engineering/notebook|[LB24] Deepseek-R1 with more prompt Engineering]] ([kaggle](https://www.kaggle.com/code/haoruili/lb24-deepseek-r1-with-more-prompt-engineering))
- #880 votes [[notebooks/votes_06_octaviograu-lb-27-aimo-2-deepseek-r1-distill-qwen-7b-awq/notebook|[LB 27] AIMO 2: deepseek-r1-distill-qwen-7b-awq]] ([kaggle](https://www.kaggle.com/code/octaviograu/lb-27-aimo-2-deepseek-r1-distill-qwen-7b-awq))
- #585 votes [[notebooks/votes_07_haoruili-lb22-deepseek-r1-distill/notebook|[LB22] deepseek-r1-distill]] ([kaggle](https://www.kaggle.com/code/haoruili/lb22-deepseek-r1-distill))
- #583 votes [[notebooks/votes_08_huikang-qwen2-5-72b-instruct-with-tir/notebook|Qwen2.5-72B-Instruct with TIR]] ([kaggle](https://www.kaggle.com/code/huikang/qwen2-5-72b-instruct-with-tir))
- #516 votes [[notebooks/votes_09_boristown-qwen-qwq-32b-preview-deepreasoning/notebook|Qwen-QwQ-32B-Preview-DeepReasoning]] ([kaggle](https://www.kaggle.com/code/boristown/qwen-qwq-32b-preview-deepreasoning))
- #508 votes [[notebooks/votes_10_quannguyn12-lucky/notebook|LUCKY]] ([kaggle](https://www.kaggle.com/code/quannguyn12/lucky))
