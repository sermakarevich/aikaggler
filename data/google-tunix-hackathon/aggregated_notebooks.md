# google-tunix-hackathon: top public notebooks

The top-voted notebooks collectively focus on fine-tuning open-weight language models (primarily Gemma 2 and 3 variants) for mathematical reasoning on the GSM8K dataset using reinforcement learning via Group Relative Policy Optimization (GRPO) and supervised fine-tuning (SFT). They emphasize structured prompt engineering with XML-style reasoning tags, custom reward design, and efficient JAX/TPU deployment workflows. While several serve as educational tutorials or baselines, others introduce advanced techniques like trajectory-level rewards, QLoRA, and majority voting for robust evaluation.

## Common purposes
- tutorial
- training

## Competition flows
- Downloads GSM8K math QA data, formats it into a system prompt template, loads a Gemma2-2B checkpoint, applies LoRA adapters, and attempts to initialize a GRPO training loop with Tunix on TPUs before halting due to a sharding error.
- Loads GSM8K math problems, initializes a Gemma 2-2B model with LoRA, trains it for 10 steps using GRPO with a custom trajectory reward function, and evaluates the final accuracy on a held-out test split.
- Loads a CSV dataset of questions, initializes a Gemma-2B model with Flax and Tunix, runs a skeleton GRPO training step, generates structured reasoning and answer outputs via model.generate, and exports a demo submission CSV.
- Loads GSM8K data, formats it into a strict dual-stream template, performs full-parameter supervised fine-tuning on Gemma 3-1B-IT, applies GRPO with custom format and math rewards, and evaluates the final model on GSM8K with majority voting.
- Loads GSM8K math problems, formats them with strict reasoning/answer tags, fully fine-tunes a Gemma-3-1B model on TPU v5e-8, and evaluates the output using a majority voting inference pipeline.
- Downloads Gemma 3 weights, formats GSM8K data into strict XML-tagged prompts, fine-tunes the model on a TPU using Tunix and Optax, and evaluates performance using majority voting over generated completions.
- Downloads GSM8K data, formats prompts with reasoning tags, configures a Gemma2-2B model with LoRA and GRPO hyperparameters for Tunix on TPU, but stops before training due to a sharding error.
- Loads GSM8K data, defines a weighted multi-component reward function, fine-tunes Gemma 2 2B IT using a custom GRPO training loop via the `tunix` library, and evaluates accuracy and format compliance on a fixed test split.
- Loads GSM8K math problems via TensorFlow Datasets, fine-tunes Gemma 3 1B using GRPO with QLoRA and strict format/correctness rewards, evaluates the RL model against the base model on a hold-out test split, and saves the final checkpoint.

## Data reading
- Loads GSM8K dataset via TensorFlow Datasets (tfds.data_source) or Kaggle Hub (kagglehub.dataset_download), parses CSV files into dictionaries with 'question' and 'answer' keys, and extracts numerical answers using a split on ####.
- tensorflow_datasets loads the gsm8k dataset, splits it into train/test, and extracts questions and answers by parsing the #### delimiter in the answer strings.
- Loads train.csv from the Kaggle input directory into a pandas DataFrame using pd.read_csv.
- Loads GSM8K via the Hugging Face datasets library and TensorFlow Datasets, with an optional merge step for a custom CSV containing question, answer, reasoning, and calculation expression columns.
- Loaded via `datasets.load_dataset("openai/gsm8k", "main", split="train"/"test")`.
- Loads GSM8K dataset via `datasets.load_dataset` and attempts to load a CSV for evaluation with a hardcoded fallback list.
- Loads GSM8K via kagglehub.dataset_download or tensorflow_datasets, reads CSV files, extracts answers using the #### delimiter, and maps them to dictionaries containing questions and answers.
- Loads GSM8K via `datasets.load_dataset("gsm8k", "main", split=split)`. / Extracts final answers using regex (`####\s*(.+)`) and formats prompts with a custom XML-style template.
- Uses tfds.load("gsm8k", split=split, as_supervised=False) to fetch the dataset, converts it to a numpy iterator, decodes question and answer strings from bytes, and splits the ground truth answer by "####" to isolate the final numerical result.

## Data processing
- Formats prompts using a custom TEMPLATE with <reasoning>, </reasoning>, <answer>, </answer> tags; applies system prompts; shuffles and batches data using Grain; filters/extracts answers from raw text.
- Applies a chat template to prompts, tokenizes with SentencePiece, and uses regex to extract numerical answers for reward computation.
- Cleans GSM8K calculation annotations, normalizes whitespace, formats prompts and responses into a strict XML-like template with system instructions, masks loss to only the model response portion, and tokenizes inputs using a custom tokenizer adapter.
- Cleans GSM8K calculation annotations by replacing `<<...>>` with parentheses, extracts reasoning and answer components, formats data into strict XML-style prompts, tokenizes with a custom tokenizer, applies padding/truncation to a maximum sequence length of 2048, and creates loss masks to ignore prompts and padding during training.
- Cleans GSM8K calculation artifacts, extracts reasoning/answer components, formats data into strict XML-tagged chat templates, tokenizes with padding/truncation, and creates causal loss masks to ignore prompt tokens.
- Normalizes numerical answers by stripping `$`, `,`, and `%` symbols. / Extracts `<reasoning>` and `<answer>` tags via regex for reward calculation. / Applies strict XML boundaries to prompts and responses to ensure format compliance.
- Formats inputs with system instructions and one-shot examples, wraps prompts and responses in <start_of_turn> tags, shuffles the dataset with seed 42, batches it to size 1, and repeats the training dataset 50 times.

## Models
- Gemma2-2B
- Gemma 2-2B-it
- google/gemma-2b
- Gemma 3-1B-IT
- Gemma-3-1B
- Gemma 3 1B
- Gemma 2 2B IT
- Gemma 3 1B IT (google/gemma-3-1b-it)

## Frameworks used
- jax
- flax
- tunix
- grain
- optax
- orbax
- qwix
- kagglehub
- tensorflow_datasets
- wandb
- humanize
- tqdm
- transformers
- datasets
- google-tunix
- sympy
- flax.nnx
- orbax.checkpoint
- pandas
- huggingface_hub

## Loss functions
- GRPO loss with KL divergence penalty (beta=0.08) and clipping (epsilon=0.2)
- GRPO loss with KL divergence penalty (KL_COEFF=0.05)
- GRPO objective (KL-constrained policy gradient with clipping) optimized via deterministic reward functions (1.0 for format compliance, 2.0 for correctness, 0.0 otherwise).

## CV strategies
- Holdout split (train[:1024] for training, test[:256] for final evaluation)
- GSM8K train/test split holdout

## Ensembling
- Evaluates predictions using majority voting over multiple generated reasoning paths (configurable via `VOTE_SAMPLES`) with temperature 0.7 and top_k=40 to encourage diverse outputs, followed by robust answer extraction and normalization.
- Evaluates predictions using majority voting over multiple sampled completions (self-consistency) to improve answer accuracy.

## Insights
- High temperature (0.9) and top_p/top_k sampling are crucial during GRPO training to encourage diverse reasoning traces.
- A sufficiently large KL divergence penalty coefficient (beta=0.08) is necessary to prevent policy collapse during reinforcement learning fine-tuning.
- Integrating Tunix with Flax/NNX requires careful sharding configuration, and mismatched metadata arguments can cause immediate runtime failures.
- Trajectory-level rewards can effectively guide reasoning capabilities by explicitly scoring intermediate steps rather than just final answers.
- Combining reasoning quality (60%) and answer correctness (40%) in the reward function balances exploration of the reasoning process with task completion.
- GRPO is a viable, lightweight alternative to PPO for aligning small language models on math tasks.
- Demonstrates how to integrate the Tunix library with Flax and Hugging Face Transformers for GRPO-based alignment.
- Shows a minimal prompt formatting strategy to enforce structured <reasoning> and <answer> tags in model outputs.
- Illustrates that a functional training loop can be bootstrapped in just a few cells using Tunix's high-level Trainer API.
- Structured internal monologues with explicit sections improve model oversight and debuggability.
- Chaining full-parameter SFT with GRPO using format-aware rewards effectively aligns LLMs to verifiable reasoning patterns without LoRA.
- Reward shaping (exact vs. soft) provides a smoother learning signal for structural compliance during reinforcement learning.
- Strict system prompts and XML-style tags significantly improve model compliance and output parsing reliability.
- Replacing GSM8K's `<<...>>` calculation artifacts with parentheses prevents tokenization and formatting issues during supervised fine-tuning.
- JAX compilation overhead causes a slow first training step, but TPU execution becomes highly efficient afterward.
- Majority voting with moderate temperature and top_k sampling effectively boosts accuracy by aggregating diverse reasoning paths.
- Strict instruction formatting with XML tags during training ensures consistent machine-parseable outputs at inference.
- Explicitly verifying TPU device placement for model parameters prevents silent CPU training failures.
- Robust answer extraction requires cascading regex patterns and normalization to handle formatting variations in LLM outputs.
- Majority voting over diverse generations significantly improves math problem accuracy compared to single greedy decoding.
- GRPO training requires a sufficiently high KL penalty coefficient to prevent divergence.
- Using high temperature and top_p during generation promotes diverse reasoning traces necessary for effective policy optimization.
- Explicit prompt templating with dedicated reasoning tags guides the model to structure its output consistently.
- LoRA sharding on TPU requires precise mesh configuration to avoid metadata assignment errors.
- Multi-component rewards (answer correctness, reasoning length, format compliance) effectively guide LLMs toward structured mathematical reasoning.
- GRPO can be implemented with a lightweight custom training loop using `tunix` and Flax/JAX without heavy RL frameworks.
- XML-tagged prompts and responses improve reward signal clarity and format compliance in math tasks.
- GRPO eliminates the need for a separate value network, significantly reducing GPU memory overhead compared to standard PPO.
- Targeting MLP layers (gate_proj, up_proj, down_proj) alongside attention projections is critical for capturing reasoning capabilities.
- Combining strict format rewards with correctness rewards effectively guides the model toward valid logical paths without inducing reward hacking.
- Using low temperature (0.1) and top-k sampling during inference prevents mode collapse and ensures deterministic reasoning outputs.

## Critical findings
- JAX compilation causes the first training step to take 2-5 minutes, after which TPU speeds up significantly.
- Running on CPU instead of TPU will produce incorrect results due to sharding mismatches.
- Majority voting for evaluation requires a temperature > 0 to generate diverse reasoning paths, otherwise it yields identical samples.
- Mixing tokenizers across checkpoints can corrupt training and evaluation, so tokenizer/model alignment is critical.
- The first training step takes 2-5 minutes due to XLA compilation, after which TPU utilization becomes apparent.
- GSM8K datasets contain non-standard calculation annotations that must be cleaned to avoid confusing the model during SFT.
- Extremely low perplexity combined with low reward during training indicates mode collapse where the model repeats incorrect reasoning.
- A KL divergence spike greater than 0.1 signals reward hacking or policy divergence from the natural language distribution.
- The base model lacks strict formatting compliance, requiring engineered one-shot prompts and aggressive post-processing to reliably extract answers.

## What did not work
- The LoRA policy initialization failed with a TypeError related to set_metadata sharding arguments, prompting the author to halt the notebook and note that the referenced Windmaple code is significantly longer and more complex.
- The LoRA policy initialization failed with a TypeError due to incorrect sharding/metadata arguments, causing the author to abandon the training pipeline.

## Notable individual insights
- votes 245 (TT, Tunix on TPU ): The LoRA policy initialization failed with a TypeError related to set_metadata sharding arguments, prompting the author to halt the notebook and note that the referenced Windmaple code is significantly longer and more complex.
- votes 103 (Tunix Hack WINNER - Trajectory Reward Training): Trajectory-level rewards can effectively guide reasoning capabilities by explicitly scoring intermediate steps rather than just final answers.
- votes 94 (DSA-SFT=>GRPO-noLora-tunix): Chaining full-parameter SFT with GRPO using format-aware rewards effectively aligns LLMs to verifiable reasoning patterns without LoRA.
- votes 89 (Supervised Fine Tuning Full): Replacing GSM8K's `<<...>>` calculation artifacts with parentheses prevents tokenization and formatting issues during supervised fine-tuning.
- votes 81 (Guide : Gemma 3 fine tuning with Tunix): Mixing tokenizers across checkpoints can corrupt training and evaluation, so tokenizer/model alignment is critical.
- votes 48 (Teach gemma 3 LLM to reason- GPU): Extremely low perplexity combined with low reward during training indicates mode collapse where the model repeats incorrect reasoning.

## Notebooks indexed
- #245 votes [[notebooks/votes_01_mpwolke-tt-tunix-on-tpu/notebook|TT, Tunix on TPU ]] ([kaggle](https://www.kaggle.com/code/mpwolke/tt-tunix-on-tpu))
- #103 votes [[notebooks/votes_02_hemanthreganti-tunix-hack-winner-trajectory-reward-training/notebook|Tunix Hack WINNER - Trajectory Reward Training]] ([kaggle](https://www.kaggle.com/code/hemanthreganti/tunix-hack-winner-trajectory-reward-training))
- #99 votes [[notebooks/votes_03_tvmilitary-google-tunix-hack-starter-gemma-tunix-6-cell-b/notebook|Google Tunix Hack Starter — Gemma + Tunix 6‑Cell B]] ([kaggle](https://www.kaggle.com/code/tvmilitary/google-tunix-hack-starter-gemma-tunix-6-cell-b))
- #94 votes [[notebooks/votes_04_danielwycoff-dsa-sft-grpo-nolora-tunix/notebook|DSA-SFT=>GRPO-noLora-tunix]] ([kaggle](https://www.kaggle.com/code/danielwycoff/dsa-sft-grpo-nolora-tunix))
- #89 votes [[notebooks/votes_05_marculera-supervised-fine-tuning-full/notebook|Supervised Fine Tuning Full]] ([kaggle](https://www.kaggle.com/code/marculera/supervised-fine-tuning-full))
- #85 votes [[notebooks/votes_06_p77091067-start-with-gemma3-1b-it-tutorial/notebook|Start with Gemma3-1b-it tutorial]] ([kaggle](https://www.kaggle.com/code/p77091067/start-with-gemma3-1b-it-tutorial))
- #81 votes [[notebooks/votes_07_yekahaaagayeham-guide-gemma-3-fine-tuning-with-tunix/notebook|Guide : Gemma 3 fine tuning with Tunix]] ([kaggle](https://www.kaggle.com/code/yekahaaagayeham/guide-gemma-3-fine-tuning-with-tunix))
- #71 votes [[notebooks/votes_08_angelfencer-descriptive-and-solid-tt-tunix-on-tpu/notebook|Descriptive and Solid TT, Tunix on TPU ]] ([kaggle](https://www.kaggle.com/code/angelfencer/descriptive-and-solid-tt-tunix-on-tpu))
- #57 votes [[notebooks/votes_09_xreina8-tunix-grpo-gemma-math-reasoning/notebook|Tunix GRPO Gemma Math Reasoning]] ([kaggle](https://www.kaggle.com/code/xreina8/tunix-grpo-gemma-math-reasoning))
- #48 votes [[notebooks/votes_10_jayaprakashmurugan-teach-gemma-3-llm-to-reason-gpu/notebook|Teach gemma 3 LLM to reason- GPU]] ([kaggle](https://www.kaggle.com/code/jayaprakashmurugan/teach-gemma-3-llm-to-reason-gpu))
