# ai-mathematical-olympiad-progress-prize-3: cross-solution summary

The Kaggle AI Mathematical Olympiad Progress Prize 2 challenged participants to solve complex mathematical problems under strict inference time constraints, favoring efficiency and robust orchestration over raw model scale. Winning approaches converged on highly optimized inference pipelines leveraging quantized LLMs, speculative decoding, and dynamic token budgeting, while integrating code execution and custom early-stopping mechanisms to maximize throughput. Final submissions were consistently refined through sophisticated ensembling strategies, including weighted voting, checkpoint merging, and tool-call-aware aggregation, demonstrating that inference engineering and alignment fine-tuning were as critical as model architecture.

## Competition flows
- Raw mathematical problems are fed into a vLLM server via an OpenAI-compatible API, where streaming inference dynamically terminates upon answer generation, followed by a weighted voting mechanism that selects the final submission based on empirical score logs.
- Raw competition questions are routed through a patched vLLM + GPT-OSS inference pipeline that executes code in concurrent, stateful Jupyter sessions, applies timeout and crash safeguards, and aggregates final answers via majority voting across multiple attempts.
- Raw mathematical problems are processed through quantized LLMs (primarily 14B/32B variants of DeepSeek/Qwen) using multi-stage fine-tuning (SFT/DPO/GRPO), followed by highly optimized inference with speculative decoding, dynamic token budgeting, and custom early-stopping, culminating in code-assisted answer extraction and weighted voting for final submissions.

## Models
- GPT-OSS
- 120B model
- DeepSeek R1 series (7B, 14B, 32B)
- DeepSeek-R1-Distill-Qwen-14B
- Qwen2.5-14B
- ModernBERT

## Frameworks used
- vLLM
- OpenAI API
- GPT-OSS
- TensorRT-LLM
- lmdeploy
- SGLang
- ReDrafter

## Loss functions
- GRPO reward function (Cosine reward for length penalty)

## Ensembling
- A custom weighted voting system selects the final answer from multiple generations, applying point thresholds and prioritizing larger numerical outputs to filter out premature guesses.
- Final answers are aggregated using majority voting across up to 4 attempts per question, with a fallback to selecting the most frequent non-trivial result while ignoring 0/1 unless they are the only options.
- Weighted voting algorithms prioritizing consensus between boxed answers and code outputs, combined with linear checkpoint merging of SFT and TIR/GRPO models and Self-Consistency early stopping to aggregate multiple generations.

## Insights
- Streaming inference and immediate termination upon finding an answer drastically improves GPU throughput and allows deeper exploration of the solution space.
- Adding verification prompts prevents the model from outputting trivial or shortcut answers, forcing more rigorous reasoning.
- Implementing a token-limit fallback ensures the model always returns a guess rather than stalling indefinitely.
- API-ifying the inference server simplifies model benchmarking and enables the use of remote GPUs.
- Logging generation metrics and answer scores is critical for identifying failure modes and tuning inference parameters.
- Leveraging vLLM's response endpoint with integrated tool calling simplifies the orchestration of reasoning-and-code loops compared to manual notebook management.
- Maintaining persistent, request-scoped Jupyter kernel states is critical for achieving high concurrency and low latency in tool-use inference pipelines.
- Hard timeouts and automatic kernel restarts are essential for stabilizing code execution environments that may encounter segfaults or infinite loops.
- Balancing model reasoning capability with inference efficiency is more critical than raw model size for time-constrained math competitions.
- Multi-stage fine-tuning combining SFT for foundational knowledge and DPO/GRPO for alignment effectively reduces reasoning redundancy.
- Quantization (AWQ, KV Cache) and specialized inference engines (TensorRT-LLM, lmdeploy) provide substantial throughput gains over standard setups.
- Dynamic time allocation and difficulty prediction allow teams to allocate computational budgets adaptively across problems of varying complexity.

## Critical findings
- GPU KV cache usage hitting 100% causes generation failures, necessitating request throttling or termination.
- Smaller numerical answers are highly correlated with the model giving up, so weighting larger answers higher in voting yields better scores.
- Voting by the number of tool calls rather than simple majority voting can push accuracy from ~83% to 84%+ on the validation set.
- The model frequently outputs \displaystyle instead of \boxed, requiring a regex post-processor to capture valid answers correctly.
- The 14B model's length penalty severely hurt accuracy during GRPO training, requiring its removal to maintain performance.
- Standard vLLM was outperformed by TensorRT-LLM, lmdeploy, and SGLang in high-throughput scenarios.
- Forcing code generation and implementing an LLM-driven error-fixing loop significantly improved solution reliability over pure text reasoning.
- Model merging successfully recovered accuracy lost during aggressive length-reduction fine-tuning.

## What did not work
- Terminating generation based on GPU KV cache usage hitting 100% was abandoned in favor of allowing preemption.
- The complex vote manipulation logic (weighting by answer size and point thresholds) lost relevance as model reasoning capabilities improved.
- The 14B model's length penalty severely hurt accuracy during GRPO training, requiring its removal.
- Simple majority voting was outperformed by carefully designed weighted algorithms that penalized short code or small answer values.

## Notable individual insights
- rank 5 (Ideas behind my public submission scoring [14/50] and [21/50]): Streaming inference and immediate termination upon finding an answer drastically improves GPU throughput and allows deeper exploration of the solution space.
- rank 5 (Ideas behind my public submission scoring [14/50] and [21/50]): Smaller numerical answers are highly correlated with the model giving up, so weighting larger answers higher in voting yields better scores.
- rank null (Sharing of our methods (longest leaderboard award submission)): Voting by the number of tool calls rather than simple majority voting can push accuracy from ~83% to 84%+ on the validation set.
- rank null (Sharing of our methods (longest leaderboard award submission)): The model frequently outputs \displaystyle instead of \boxed, requiring a regex post-processor to capture valid answers correctly.
- rank null (Kaggle AIMO 2 Top Solutions Report (Generated by notebooklm)): Balancing model reasoning capability with inference efficiency is more critical than raw model size for time-constrained math competitions.
- rank null (Kaggle AIMO 2 Top Solutions Report (Generated by notebooklm)): Quantization (AWQ, KV Cache) and specialized inference engines (TensorRT-LLM, lmdeploy) provide substantial throughput gains over standard setups.
- rank null (Kaggle AIMO 2 Top Solutions Report (Generated by notebooklm)): Model merging successfully recovered accuracy lost during aggressive length-reduction fine-tuning.

## Solutions indexed
- #5 [[solutions/rank_05/solution|Ideas behind my public submission scoring [14/50] and [21/50]]]
- ? [[solutions/rank_xx_671835/solution|Sharing of our methods (longest leaderboard award submission)]]
- ? [[solutions/rank_xx_638787/solution|Kaggle AIMO 2 Top Solutions Report (Generated by notebooklm)]]
