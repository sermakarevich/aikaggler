# konwinski-prize: top public notebooks

The community's top notebooks focus on inference pipelines for code generation and repository repair, utilizing large language models (e.g., Qwen, DeepSeek, Gemma) orchestrated via frameworks like vLLM, LangGraph, and custom agents. Key strategies include retrieval-augmented generation with syntax parsing (tree-sitter) and vector search, multi-step 'Select-Patch-Verify' workflows with deterministic validation, and parameter-efficient fine-tuning baselines. Efforts emphasize managing large context windows, offline dependency resolution, and integrating LLMs with external tools to generate and verify git diff patches within competition constraints.

## Common purposes
- inference
- utility
- tutorial
- baseline

## Competition flows
- Unpacks a repository archive and problem statement, uses a quantized LLM to select relevant files, extract code context, generate a git diff patch, verify its correctness via LLM judgment and dry-run execution, and returns the patch for submission.
- Sets up a local inference gateway to simulate the competition environment, unpacks repository and dependency archives, runs environment setup, and returns a dummy response to validate the API contract.
- Unpacks a repository archive, parses Python files into a flattened syntax tree, indexes the structure in Elasticsearch, runs a LangGraph agent with a local Qwen2.5-14B model to search and edit code, and returns a validated git diff patch.
- Loads GSM8K math problems, formats them with a strict reasoning template, fine-tunes Qwen2.5-1.5B via a custom GRPO training loop with accuracy and format rewards, and saves the resulting model for inference.
- Loads SWE-bench metadata and problem statements, formats them into question-answer pairs, fine-tunes a Gemma-2B model with LoRA for one epoch, and implements a custom inference function to generate code patches for GitHub issues via the competition's evaluation server.
- Unpacks competition repository and dependency archives, configures a local vLLM server and OpenHands agent, runs the agent against each GitHub issue to generate code diffs, and writes the results to a submission CSV.
- Loads problem statements and repository archives, retrieves relevant code files using TF-IDF and semantic similarity, feeds them into a Qwen2.5-Coder-7B agent with a structured prompt to generate a git diff patch, validates the output, and serves predictions via Kaggle's evaluation server.

## Data reading
- Reads problem statements and metadata from a parquet file using pandas.
- Unpacks .tar and .zip archives for the repository codebase and pip packages using shutil and zipfile.
- Reads file contents directly using Python's open() and os.walk() to traverse the repository directory.
- Unpacks repo_archive and pip_packages_archive from io.BytesIO buffers to temporary and local directories using shutil.unpack_archive.
- Unpacks a .tar archive from a BytesIO buffer into a local repo/ directory.
- Reads Python files via os.walk and binary file handles for tree-sitter parsing.
- Reads problem statements and instance IDs from a parquet file using pandas/polars.
- Extracts repository archives from zip/tar files into a local directory for processing.
- Unpacks a .tar archive from a BytesIO buffer into a local repo directory using shutil.unpack_archive.
- Reads problem statements and instance IDs from a local data.parquet file using pandas/polars.
- Loaded via datasets.load_dataset('openai/gsm8k', 'main') and converted to pandas; answers extracted by splitting on #### and stripping whitespace.
- Extracts data.a_zip and reads data/data.parquet with pandas.
- Loads the princeton-nlp/SWE-bench dataset from HuggingFace datasets and converts it to a pandas DataFrame.
- Reads repository and pip package archives from io.BytesIO buffers provided by the competition gateway.
- Parses pyproject.toml to extract project names and metadata.
- Reads data.parquet using pandas, prefixes instance_id with repo__
- Extracts repository archives from data.a_zip into a local directory

## Data processing
- Stringifies repository directory trees for LLM context.
- Extracts code snippets with configurable context lines and merges overlapping snippets.
- Parses LLM-generated XML/JSON outputs to extract file paths and search terms.
- Validates patch syntax and generated diffs using unidiff and deterministic dry-run checks.
- Runs shell commands for environment setup via subprocess.run.
- Parses Python source code using tree-sitter to generate and flatten syntax trees.
- Filters for function_definition and class_definition nodes and flattens hierarchical data into dictionaries with UUIDs, parent-child links, file paths, types, text snippets, and line/column coordinates.
- Indexes flattened syntax tree data into Elasticsearch for structured retrieval.
- Formats directory paths, search terms, and code snippets into structured XML/diff prompts.
- Extracts patch strings via regex and handles file encoding errors.
- Applies custom system prompt templates, strips/parses answers, shuffles data, and enforces XML-like formatting via regex.
- Formats raw problem statements and patches into fixed Question/Answer templates.
- Sets model sequence length and batch size for training.
- Unpacks archives to /testbed and provisions Python virtual environments using uv.
- Resolves dependencies via poetry and pip and configures runtime environment variables.
- Filters repository files by extension and size.
- Computes TF-IDF vectors for problem statements and file contents to retrieve relevant files.
- Truncates directory structures and file contents to fit context limits.
- Parses LLM XML outputs with regex and xml.etree.ElementTree.
- Converts directory trees to newline-separated strings for LLM context.
- Fetches file contents with configurable context windows and merges overlapping/adjacent lines.
- Formats directory paths and search terms into structured prompt templates.
- Sets model sequence length to 512 and uses batch_size=1 during training.
- Resolves dependencies via poetry and pip.
- Configures runtime environment variables for the agent.
- Filters repository files by extension and size.
- Computes TF-IDF vectors for the problem statement and file contents to retrieve top-N relevant files.
- Truncates directory structures and file contents to fit context limits.
- Validates generated patches using unidiff.
- Runs shell commands for environment setup via subprocess.run after unpacking dependencies.
- Validates patch syntax with unidiff and runs a deterministic patch --dry-run command to filter invalid patches.
- Extracts code snippets with configurable context lines around search terms and merges overlapping snippets to preserve continuity.
- Stringifies the repository directory structure for LLM context.
- Parses LLM-generated XML/JSON-like outputs to extract file paths and search strings.
- Converts directory trees to newline-separated strings for LLM context.
- Parses LLM XML outputs with regex and xml.etree.ElementTree.
- Extracts code snippets with configurable context windows and merges overlapping/adjacent lines.
- Extracts patch strings via regex and handles file encoding errors gracefully.
- Formats directory paths and search terms into structured prompt templates.
- Applied a custom system prompt template to all questions, stripped and parsed final answers, shuffled training data, and enforced strict XML-like output formatting via regex during reward computation.
- Formats raw problem_statement and patch fields into a fixed template string (Question:\n{q}\n\nAnswer:\n{a}).
- Sets model sequence length to 512 and uses batch_size=1 during training.
- Unpacks archives to /testbed and provisions Python virtual environments using uv.
- Resolves dependencies via poetry and pip, and configures runtime environment variables for the agent.
- Filters repository files by extension and size
- Computes TF-IDF vectors for the problem statement and file contents to retrieve top-N relevant files
- Truncates directory structures and file contents to fit context limits
- Validates generated patches using unidiff

## Features engineering
- Flattened syntax tree nodes (function/class definitions) with parent-child relationships, file paths, and start/end line/column coordinates indexed for BM25 search.
- Hand-crafted repository analysis including directory tree traversal, git commit history extraction, and pattern-based priority file identification
- TF-IDF and CodeBERT cosine similarity scores used for file retrieval

## Models
- Qwen-QwQ-32B-Preview-AWQ
- QwQ-32B
- Qwen2.5-14B-Instruct
- DeepSeek-R1-Distill-Qwen-32B-AWQ
- DeepSeek-R1-Distill-Qwen-32B
- Qwen2.5-1.5B
- Gemma-2B
- Qwen2.5-Coder-32B-Instruct
- Qwen2.5-Coder-7B-Instruct
- CodeBERT

## Frameworks used
- vllm
- pandas
- polars
- kaggle_evaluation
- unidiff
- kaggle_evaluation.konwinski_prize_inference_server
- langgraph
- langchain
- elasticsearch
- tree-sitter
- pydantic
- torch
- transformers
- peft
- datasets
- scikit-learn
- numpy
- matplotlib
- keras
- keras-nlp
- jax
- poetry
- uv
- tomlkit
- gitpython

## Loss functions
- Clipped policy loss with KL divergence penalty: `(-clipped_ratio * advantages - beta * kl).mean()`
- SparseCategoricalCrossentropy(from_logits=True)

## Ensembling
- Aggregates multiple LLM generations to produce candidate patches, verifies each via LLM judgment and deterministic dry-run execution, and selects the best patch based on combined verification votes and scoring.

## Insights
- Structuring LLM inference into Select-Patch-Verify stages improves code generation reliability.
- Combining LLM reasoning with deterministic checks like unidiff and patch --dry-run filters out syntactically invalid or inapplicable patches.
- Merging overlapping code snippets preserves context while managing token limits for large repositories.
- Aggregating multiple LLM generations and voting on verification outcomes reduces hallucination and improves patch selection.
- The competition gateway passes codebases and dependencies as tar archives in memory, requiring explicit unpacking before execution.
- The first inference call allows up to 15 minutes for model loading, bypassing the standard 30-minute response deadline.
- Local testing is supported via run_local_gateway, which simulates the hidden test set using provided Kaggle input paths.
- Integrating tree-sitter syntax parsing with Elasticsearch enables precise, hierarchical codebase search beyond simple text matching.
- Using LangGraph to orchestrate tool calls allows the LLM to iteratively explore and modify code without hallucinating file paths.
- Prefix caching and RoPE scaling in vLLM are critical for handling long context windows efficiently during inference.
- Integrating a local LLM with Elasticsearch enables efficient, context-aware codebase navigation for autonomous agents.
- Flattening hierarchical AST nodes into a searchable index preserves structural relationships while enabling fast BM25 retrieval.
- LangGraph's stateful workflow effectively manages the iterative tool-calling loop required for complex code editing tasks.
- LLMs can be effectively guided to perform multi-step code repair by structuring prompts for file selection, patch generation, and self-verification.
- Combining LLM-based evaluation with deterministic dry-run checks improves patch reliability and reduces hallucinated diffs.
- Managing context windows and merging overlapping code snippets is crucial for handling large repositories without exceeding token limits.
- Multi-step prompting with structured XML outputs improves LLM reliability for file selection and patch generation.
- AWQ quantization combined with vLLM tensor parallelism enables running 32B models on consumer/server GPUs within memory limits.
- Self-verification prompts can filter out invalid patches before submission, acting as a lightweight quality gate without additional training.
- Implementing GRPO from scratch requires manually managing log-probability ratios, advantage normalization, and KL penalties relative to a frozen reference model.
- Group sampling (multiple completions per prompt) enables advantage estimation without a separate value network.
- Strict formatting rewards can guide a model's output structure even when accuracy rewards are sparse.
- Parameter-efficient fine-tuning with LoRA (rank=64) and a learning rate of 5e-5 provides a stable baseline for code generation tasks.
- Formatting code patches as explicit Question/Answer templates aligns well with causal language modeling objectives.
- The official Kaggle inference server wrapper automatically handles repository unpacking and instance routing, simplifying submission generation.
- Deploying large coding models locally via vLLM is viable on Kaggle GPUs when using offline package installation.
- OpenHands agents require explicit, step-by-step prompt engineering and strict constraints to avoid modifying test files or generating workarounds.
- Offline dependency resolution and virtual environment management are critical for reliable agent execution in isolated competition environments.
- Modifying unit tests during patch generation will cause submission failures, so the agent must strictly fix library source code.
- Tracebacks must be read bottom-to-top, and metaclass __call__() methods often hide the true source of configuration errors.
- Internet is disabled, requiring all dependencies to be pre-packaged and resolved locally via uv and poetry.
- Retrieval-augmented generation with TF-IDF and CodeBERT embeddings effectively narrows down relevant repository files for large language models.
- Structured prompting with explicit formatting rules and step-by-step reasoning improves patch generation reliability.
- Memory management (offloading, chunking, cache clearing) is critical for running 7B models on Kaggle GPUs.

## Critical findings
- Modifying unit tests in the provided repository will cause the submission to be flagged as a failure, and running the full test suite may exceed the inference time limit.
- Modifying unit tests during patch generation will cause submission failures, so the agent must strictly fix library source code.
- Tracebacks must be read bottom-to-top, and metaclass __call__() methods often hide the true source of configuration errors.
- Internet is disabled, requiring all dependencies to be pre-packaged and resolved locally via uv and poetry.

## Notable individual insights
- votes 680 (Starter notebook - Select-Patch-Verify): Structuring LLM inference into Select-Patch-Verify stages improves code generation reliability.
- votes 616 (Konwinski Prize Demo Submission): The competition gateway passes codebases and dependencies as tar archives in memory, requiring explicit unpacking before execution.
- votes 167 (Konwinski Prize - AI GitHub Issue Resolver (TB)): Integrating tree-sitter syntax parsing with Elasticsearch enables precise, hierarchical codebase search beyond simple text matching.
- votes 121 (grpo_train_reasoning_model): Implementing GRPO from scratch requires manually managing log-probability ratios, advantage normalization, and KL penalties relative to a frozen reference model.
- votes 81 ([Konwinski Prize] Gemma LLM): Parameter-efficient fine-tuning with LoRA (rank=64) and a learning rate of 5e-5 provides a stable baseline for code generation tasks.
- votes 77 (KPrize | Openhands Fork): Deploying large coding models locally via vLLM is viable on Kaggle GPUs when using offline package installation.
- votes 73 (Konwinski Minimal Qwen Agent): Retrieval-augmented generation with TF-IDF and CodeBERT embeddings effectively narrows down relevant repository files for large language models.

## Notebooks indexed
- #680 votes [[notebooks/votes_01_huikang-starter-notebook-select-patch-verify/notebook|Starter notebook - Select-Patch-Verify]] ([kaggle](https://www.kaggle.com/code/huikang/starter-notebook-select-patch-verify))
- #616 votes [[notebooks/votes_02_sohier-konwinski-prize-demo-submission/notebook|Konwinski Prize Demo Submission ]] ([kaggle](https://www.kaggle.com/code/sohier/konwinski-prize-demo-submission))
- #167 votes [[notebooks/votes_03_olaflundstrom-konwinski-prize-ai-github-issue-resolver-tb/notebook|Konwinski Prize - AI GitHub Issue Resolver (TB)]] ([kaggle](https://www.kaggle.com/code/olaflundstrom/konwinski-prize-ai-github-issue-resolver-tb))
- #150 votes [[notebooks/votes_04_jinssaa-agent-system-langgraph-vllm-elasticsearch-qwen/notebook|Agent System-Langgraph,Vllm,Elasticsearch,Qwen]] ([kaggle](https://www.kaggle.com/code/jinssaa/agent-system-langgraph-vllm-elasticsearch-qwen))
- #146 votes [[notebooks/votes_05_quannguyn12-starter-notebook-select-patch-verify/notebook|Starter notebook-Select-Patch-Verify ]] ([kaggle](https://www.kaggle.com/code/quannguyn12/starter-notebook-select-patch-verify))
- #121 votes [[notebooks/votes_06_zhudong1949-starter-notebook-select-patch-verify-ca53bc/notebook|Starter notebook - Select-Patch-Verify ca53bc]] ([kaggle](https://www.kaggle.com/code/zhudong1949/starter-notebook-select-patch-verify-ca53bc))
- #121 votes [[notebooks/votes_07_ducnh279-grpo-train-reasoning-model/notebook|grpo_train_reasoning_model]] ([kaggle](https://www.kaggle.com/code/ducnh279/grpo-train-reasoning-model))
- #81 votes [[notebooks/votes_08_akhiljethwa-konwinski-prize-gemma-llm/notebook|[Konwinski Prize] Gemma LLM]] ([kaggle](https://www.kaggle.com/code/akhiljethwa/konwinski-prize-gemma-llm))
- #77 votes [[notebooks/votes_09_smartmanoj-kprize-openhands-fork/notebook|KPrize | Openhands Fork]] ([kaggle](https://www.kaggle.com/code/smartmanoj/kprize-openhands-fork))
- #73 votes [[notebooks/votes_10_umar47-konwinski-minimal-qwen-agent/notebook|Konwinski Minimal Qwen Agent]] ([kaggle](https://www.kaggle.com/code/umar47/konwinski-minimal-qwen-agent))
