# konwinski-prize: cross-solution summary

Top solutions for this code repair competition leveraged LLM-based pipelines, heavily utilizing the Agentless framework and Qwen2.5-Coder-32B models to automate issue resolution and patch generation. Winning approaches emphasized robust context retrieval, structured SEARCH/REPLACE diff formats, and rigorous multi-stage verification to ensure patch validity while managing strict runtime constraints. Success also depended on strategic scoring optimization, including dynamic skip mechanisms and hypergeometric analysis to navigate high-penalty distributions.

## Competition flows
- Load GitHub issue and repository context -> generate reproduction tests (F2P) with enhanced context -> localize relevant code locations -> generate SEARCH/REPLACE patch candidates -> validate patches via dry-run and execute against F2P and pre-existing unit tests (P2P) in parallel -> apply time constraints and submit the best valid patch.
- Seven-stage Agentless-based pipeline: evaluation -> localization -> selection -> test generation -> patch generation -> verification -> timeout management.
- Three-stage LLM pipeline with temperature scheduling and edit-distance candidate selection, followed by hypergeometric-simulated skip mechanism and strict time limits.

## Data processing
- Filtered problem statements by character length (1802-3400 chars)
- Skipped instances with >388 file contents
- Filtered generated patches >1000 chars
- Applied Edit Distance Selection to choose representative candidates
- Implemented dynamic skip mechanisms based on turn limits and time constraints

## Models
- Qwen2.5-Coder-32B
- Agentless 1.5
- Qwen2.5-72B
- QwQ-32B
- Deepseek-R1
- ChatGPT
- Grok
- Gemini
- Claude
- deepseek-r1-distill-qwen-32b-awq

## Frameworks used
- vLLM

## CV strategies
- Ignored cross-validation; relied entirely on public leaderboard and external instances for tuning and methodology validation.

## Ensembling
- Deterministic selection ensemble via dry-run validation and parallel F2P/P2P test execution of multiple candidates.

## Notable individual insights
- Rank 1 (1st Place Solution Write Up): Using SEARCH/REPLACE diff format drastically improves patch validity over raw git diffs.
- Rank 1 (1st Place Solution Write Up): Explicit context (imports, skeletons) is vital for 32B models to prevent instantiation errors.
- Rank 2 (Public 2nd Place Solution): Tracking granular pipeline metrics (e.g., good_test_rate) is more effective than optimizing isolated metrics.
- Rank 2 (Public 2nd Place Solution): Generating reproduction tests that first fail and then pass is crucial for reliable verification.
- Rank 4 (Public 4th Place / Private 15th Place Solution - Konwinski Prize): High penalty scoring makes an effective skip mechanism more valuable than raw accuracy.
- Rank 4 (Public 4th Place / Private 15th Place Solution - Konwinski Prize): Odd-numbered challenge counts statistically outperform even counts due to hypergeometric distribution properties.
- Rank 4 (Public 4th Place / Private 15th Place Solution - Konwinski Prize): Problem length and file count are strong predictors of solvability.

## Solutions indexed
- #1 [[solutions/rank_01/solution|1st Place Solution Write Up]]
- #2 [[solutions/rank_02/solution|Public 2nd Place Solution]]
- #4 [[solutions/rank_04/solution|Public 4th Place / Private 15th Place Solution - Konwinski Prize]]

## Papers cited
- [SWE-bench: Can Language Models Resolve Real-World GitHub Issues?](https://arxiv.org/abs/2310.06770)
- [Agentless: Demystifying LLM-based Software Engineering Agents](https://arxiv.org/abs/2407.01489)
- [Qwen2.5: Efficient and Robust Code Generation with Improved Contextual Understanding](https://arxiv.org/abs/2412.15115)
- [QWQ-32B: A New Frontier in Code Generation](https://qwenlm.github.io/blog/qwq-32b/)
- [R1: A Robust and Efficient Approach for [Task Description]](https://arxiv.org/abs/2501.12948)
- [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)
- [Introducing SWE-bench Verified](https://openai.com/index/introducing-swe-bench-verified/)
