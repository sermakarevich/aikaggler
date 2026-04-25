# santa-2024: top public notebooks

The community's top-voted notebooks for the Santa 2024 competition focus on optimizing text permutations to minimize perplexity using a frozen large language model (Gemma-2-9B) as a fixed scoring function. Authors predominantly employ heuristic and meta-heuristic search strategies (simulated annealing, greedy swaps, brute-force permutations) combined with quantization and batched inference to manage GPU memory and accelerate metric computation. A few notebooks explore hybrid approaches, such as training lightweight regression proxies for sequence quality or ensembling multiple submissions via row-wise greedy selection.

## Common purposes
- utility
- inference
- baseline
- ensemble
- other

## Competition flows
- Accepts solution and submission DataFrames, validates that each submission is a valid permutation of the original text, computes token-wise cross-entropy loss using a pre-trained LLM, and returns the mean perplexity as the final score.
- Loads a sample submission with short texts, iteratively permutes word orderings for each text to minimize Gemma-2-9B perplexity using greedy and local search strategies, caches scores for efficiency, and exports the optimized texts as a final submission.
- Reads a sample submission, loads a frozen Gemma-2-9B model to compute sequence perplexity, and iteratively optimizes the word order of each text using brute-force permutations, simulated annealing, and cyclic shifting to minimize perplexity before saving the final submission.
- Loads a sample submission, computes text perplexity using a cached Gemma-2-9B model, optimizes text via brute-force word permutations to minimize perplexity, and saves the results.
- Reads the first sample's text from the submission file, generates all possible word permutations, evaluates their perplexity using a batched quantized Gemma-2-9B model, and outputs the optimal permutation merged into a new submission CSV.
- Loads sample submission text, calculates initial perplexity using a quantized Gemma-2-9B model, applies simulated annealing and greedy permutation heuristics to minimize perplexity, and exports the optimized text permutations as a CSV submission.
- Loads multiple submission CSVs, scores each row's text using a frozen Gemma-2-9B model's perplexity, selects the lowest-scored text per row, and saves a combined submission file.
- Loads precomputed scores and sample submissions, trains an XGBoost regressor on word-position features, generates text permutations, and evaluates candidates using both the regression model and a custom Gemma-2-9B perplexity scorer to identify low-perplexity sequences.
- Loads a prior text submission CSV, evaluates each row's perplexity using a frozen Gemma-2-9B model, applies a custom Simulated Annealing algorithm to rearrange words and minimize perplexity, and saves the optimized text back to a new submission CSV.
- Reads a sample submission text, initializes a 4-bit quantized Gemma-2-9B model to compute perplexity, runs a Simulated Annealing algorithm with swap/shift neighbors to minimize perplexity, and writes the optimized text back to the submission file.

## Data reading
- Accepts solution and submission DataFrames, validates that each submission is a valid permutation of the original text, computes token-wise cross-entropy loss using a pre-trained LLM, and returns the mean perplexity as the final score.
- Loads a sample submission with short texts, iteratively permutes word orderings for each text to minimize Gemma-2-9B perplexity using greedy and local search strategies, caches scores for efficiency, and exports the optimized texts as a final submission.
- Reads a sample submission, loads a frozen Gemma-2-9B model to compute sequence perplexity, and iteratively optimizes the word order of each text using brute-force permutations, simulated annealing, and cyclic shifting to minimize perplexity before saving the final submission.
- Loads a sample submission, computes text perplexity using a cached Gemma-2-9B model, optimizes text via brute-force word permutations to minimize perplexity, and saves the results.
- Reads the first sample's text from the submission file, generates all possible word permutations, evaluates their perplexity using a batched quantized Gemma-2-9B model, and outputs the optimal permutation merged into a new submission CSV.
- Loads sample submission text, calculates initial perplexity using a quantized Gemma-2-9B model, applies simulated annealing and greedy permutation heuristics to minimize perplexity, and exports the optimized text permutations as a CSV submission.
- Loads multiple submission CSVs, scores each row's text using a frozen Gemma-2-9B model's perplexity, selects the lowest-scored text per row, and saves a combined submission file.
- Loads precomputed scores and sample submissions, trains an XGBoost regressor on word-position features, generates text permutations, and evaluates candidates using both the regression model and a custom Gemma-2-9B perplexity scorer to identify low-perplexity sequences.
- Loads a prior text submission CSV, evaluates each row's perplexity using a frozen Gemma-2-9B model, applies a custom Simulated Annealing algorithm to rearrange words and minimize perplexity, and saves the optimized text back to a new submission CSV.
- Reads a sample submission text, initializes a 4-bit quantized Gemma-2-9B model to compute perplexity, runs a Simulated Annealing algorithm with swap/shift neighbors to minimize perplexity, and writes the optimized text back to the submission file.

## Data processing
- Normalizes whitespace by splitting and rejoining strings.
- Validates permutation integrity using word frequency counters.
- Shifts logits and labels for autoregressive loss calculation.
- Manages GPU memory by clearing CUDA cache and running garbage collection.
- Splits text into words, separates stopwords from content words, applies alphabetical sorting to word groups, and caches computed perplexity scores in a dictionary to avoid redundant LLM inference.
- Splits text into words, applies brute-force permutations, simulated annealing, and cyclic shifting to rearrange word order. Handles NaN perplexity scores by reshuffling. Sets OMP_NUM_THREADS and CUDA memory config for inference stability.
- Splits the hardcoded string by newlines to populate the text column; no additional cleaning or normalization is applied.
- Splits the original text into words, generates all possible permutations using itertools.permutations, and joins them back into strings. Normalizes whitespace by splitting and rejoining strings before passing them to the scorer.
- Splits text into words, normalizes whitespace by splitting and rejoining, validates permutation integrity using collections.Counter, adds BOS/EOS tokens, tokenizes inputs with padding, shifts logits and labels for causal language modeling, and processes batches of 64 sequences.
- Splits text into words, calculates word indices, filters data by ID groups, generates permutations of text segments, and implements a custom LRU cache for batched LLM inference.
- Splits text into space-separated words, iteratively swaps random word pairs, handles NaN perplexity scores by skipping or shuffling, and applies a temperature cooling schedule to guide the optimization.
- Normalizes whitespace by splitting and rejoining strings; validates that submitted permutations contain identical word counts to the original; sorts words initially to set a baseline; manages GPU memory by clearing caches and deleting model references after scoring.

## Features engineering
- Word position indices (mapping each unique word to its index in the sequence) used as regression features.

## Models
- Gemma-2-9B
- XGBRegressor

## Frameworks used
- transformers
- PyTorch
- pandas
- numpy
- nltk
- itertools
- accelerate
- bitsandbytes
- scikit-learn
- xgboost

## Loss functions
- CrossEntropyLoss (token-wise, reduction='none')
- CrossEntropyLoss (averaged per-token then exponentiated for perplexity)
- CrossEntropyLoss
- MAE (for XGBoost regression)

## Ensembling
- Row-wise greedy selection where the submission providing the lowest perplexity score for each row is chosen to form the final ensemble submission.

## Insights
- Perplexity effectively measures how well a permuted text preserves the original linguistic structure and probability distribution.
- Token-level loss calculation allows for granular evaluation of each word's contribution to the overall score.
- 8-bit quantization makes running large language models for metric computation feasible on consumer GPUs.
- Proper sequence boundary handling with BOS and EOS tokens is critical for accurate perplexity calculation.
- Pre-trained LLM perplexity can effectively serve as a proxy scoring metric for text reordering tasks without fine-tuning.
- Caching computed permutations significantly reduces inference time during combinatorial search.
- Greedy word permutation combined with local window optimization can systematically improve text coherence according to a language model.
- A frozen LLM can serve as a fixed perplexity scorer for text rearrangement without gradient updates.
- Meta-heuristic search methods like simulated annealing and cyclic shifting effectively optimize discrete word order to minimize perplexity.
- Configuring CUDA memory and thread counts is critical for running large models like Gemma-2-9B on limited GPU resources.
- Caching model outputs with an LRU structure drastically reduces inference latency during iterative optimization.
- Brute-forcing local word permutations can effectively minimize sequence perplexity without requiring gradient-based fine-tuning.
- Frozen open-weight LLMs can serve as reliable, fixed scorers for text quality and coherence in competition tasks.
- Using large batch sizes with quantized LLMs significantly accelerates metric computation on GPUs compared to processing samples individually.
- Exhaustive brute-force search over all permutations is computationally feasible for small combinatorial spaces when metric evaluation is optimized.
- Correctly shifting logits and labels is required to compute token-level cross-entropy loss for accurate perplexity calculation.
- LLM perplexity effectively measures linguistic naturalness for word permutation tasks.
- Simulated annealing with temperature cooling and neighbor swapping significantly improves permutation scores over random shuffling.
- Brute-forcing permutations of the last few words combined with greedy insertion yields strong local optima.
- Quantizing the LLM to 4-bit (fp4) enables running large models on Kaggle GPUs without out-of-memory errors.
- Perplexity can serve as a reliable, model-agnostic proxy for text quality or alignment in competition submissions.
- Loading and scoring multiple submissions row-by-row is computationally feasible with a frozen LLM.
- Greedy per-row selection often outperforms simple averaging or majority voting for text-based tasks.
- Word position indices provide a simple yet effective feature set for predicting permutation scores.
- LLM perplexity can be computed efficiently using batched inference and an LRU cache to avoid redundant model calls.
- Brute-force permutation search is computationally infeasible for long sequences but viable for short, localized segments.
- Simulated annealing can effectively optimize discrete text arrangements by balancing exploration at high temperatures with exploitation at low temperatures.
- Using a frozen LLM as a perplexity scorer allows direct optimization of language model likelihood without requiring gradient updates or fine-tuning.
- Handling NaN scores gracefully is critical when evaluating generated text with external scorers, as invalid sequences can halt the optimization loop.
- Using a pre-trained LLM as a fixed scoring function allows combinatorial optimization without fine-tuning or training overhead.
- 4-bit quantization (fp4) enables running large models like Gemma-2-9B on limited Kaggle GPU memory.
- Simulated annealing with both swap and shift operations effectively navigates the discrete permutation space to reduce perplexity.
- Exponential cooling schedules provide a good balance between exploration and exploitation for this type of optimization.

## Critical findings
- Exhaustive permutation search scales factorially (~2.4x10^18 for 20 words), making it impossible for full-length texts.
- Duplicate words in a sequence cause index collisions that require explicit handling or deduplication strategies.

## Notable individual insights
- votes 719 (Santa 2024 Metric): Perplexity effectively measures how well a permuted text preserves the original linguistic structure and probability distribution.
- votes 330 (Brute Force First Sample - Perplexity 470): Using large batch sizes with quantized LLMs significantly accelerates metric computation on GPUs compared to processing samples individually.
- votes 294 (Regression Test): Exhaustive permutation search scales factorially (~2.4x10^18 for 20 words), making it impossible for full-length texts.
- votes 308 (Fine tuning word4 + Simple permutation): Simulated annealing with temperature cooling and neighbor swapping significantly improves permutation scores over random shuffling.
- votes 302 (Fast ensemble of multi solutions + scores analysis): Greedy per-row selection often outperforms simple averaging or majority voting for text-based tasks.
- votes 278 (Santa Claude's Approach: Simulated Annealing): Handling NaN scores gracefully is critical when evaluating generated text with external scorers, as invalid sequences can halt the optimization loop.

## Notebooks indexed
- #719 votes [[notebooks/votes_01_metric-santa-2024-metric/notebook|Santa 2024 Metric]] ([kaggle](https://www.kaggle.com/code/metric/santa-2024-metric))
- #373 votes [[notebooks/votes_02_jazivxt-to-winning-sort-off/notebook|🗝️ To Winning - Sort Off]] ([kaggle](https://www.kaggle.com/code/jazivxt/to-winning-sort-off))
- #373 votes [[notebooks/votes_03_jazivxt-diminutive-effort/notebook|Diminutive Effort]] ([kaggle](https://www.kaggle.com/code/jazivxt/diminutive-effort))
- #355 votes [[notebooks/votes_04_jazivxt-diminutive-effort-tpu/notebook|Diminutive Effort (TPU)]] ([kaggle](https://www.kaggle.com/code/jazivxt/diminutive-effort-tpu))
- #330 votes [[notebooks/votes_05_cdeotte-brute-force-first-sample-perplexity-470/notebook|Brute Force First Sample - Perplexity 470]] ([kaggle](https://www.kaggle.com/code/cdeotte/brute-force-first-sample-perplexity-470))
- #308 votes [[notebooks/votes_06_veniaminnelin-fine-tuning-word4-simple-permutation/notebook|Fine tuning word4 + Simple permutation]] ([kaggle](https://www.kaggle.com/code/veniaminnelin/fine-tuning-word4-simple-permutation))
- #302 votes [[notebooks/votes_07_kirderf-fast-ensemble-of-multi-solutions-scores-analysis/notebook|Fast ensemble of multi solutions + scores analysis]] ([kaggle](https://www.kaggle.com/code/kirderf/fast-ensemble-of-multi-solutions-scores-analysis))
- #294 votes [[notebooks/votes_08_jazivxt-regression-test/notebook|Regression Test]] ([kaggle](https://www.kaggle.com/code/jazivxt/regression-test))
- #278 votes [[notebooks/votes_09_richolson-santa-claude-s-approach-simulated-annealing/notebook|Santa Claude's Approach: Simulated Annealing]] ([kaggle](https://www.kaggle.com/code/richolson/santa-claude-s-approach-simulated-annealing))
- #240 votes [[notebooks/votes_10_egortrushin-santa24-improving-sample-2/notebook|[Santa24] Improving Sample 2]] ([kaggle](https://www.kaggle.com/code/egortrushin/santa24-improving-sample-2))
