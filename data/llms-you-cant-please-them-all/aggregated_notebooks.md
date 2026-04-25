# llms-you-cant-please-them-all: top public notebooks

The community's top-voted notebooks predominantly focus on adversarial prompt engineering and probabilistic template mixing to manipulate automated LLM judges, rather than traditional model training or EDA. Contributors establish lightweight baselines using open-weight models like Phi-3.5-mini-instruct and Gemma 2 2B-it, leveraging dynamic seed rotation, counting directives, and static text injections to bypass safety filters and game scoring metrics. The collective approach emphasizes zero-shot inference strategies, strategic test-set partitioning, and heuristic prompt formatting over complex architectures or feature engineering.

## Common purposes
- inference
- baseline
- utility

## Competition flows
- Reads test topics from CSV, dynamically constructs prompts by probabilistically mixing few-shot examples and attack suffixes, generates essays using a local Phi-3.5 model, optionally appends static post-processing strings, and saves the output to a CSV submission file.
- Reads test topics and a word list, generates three different types of random-word-based essays for different thirds of the test set, and saves them as a CSV submission.
- Loads test topics from CSV, dynamically selects and combines multiple adversarial prompt templates with weighted probabilities, generates essays using a Phi-3.5-mini-instruct model via Hugging Face pipelines, and exports the results to a submission CSV.
- Reads test prompts from a CSV, generates essays using a few-shot prompt template with Phi-3.5 via the Hugging Face transformers pipeline, trims outputs to the last punctuation mark, and saves the results to a CSV submission file.
- Loads test topics and a word list, randomly splits the dataset into two groups, generates prompt injection and multiple-choice prompts for each, and exports the combined prompts as a CSV submission.
- Loads test topics, generates essays using a weighted mix of prompt injection templates via Phi-3.5-mini-instruct, applies randomized static text injections, and exports the results to a CSV submission file.
- Loads test CSV, constructs prompts with embedded counting and identity-check instructions, generates text using Phi-3.5-mini-instruct via Hugging Face transformers, concatenates responses, and saves a submission CSV.
- Reads test topics from a CSV, generates essays using a PyTorch-based Gemma 2 2B model with a fixed prompt template and output length, appends a specific rating-influencing suffix, and saves the results to a submission CSV.
- Loads the Gemma 2 2B model, reads competition test topics via pandas, generates essays using a custom prompt template, and saves the outputs to a CSV submission file.
- Loads the test dataset and a word list, generates adversarial prompts for each test case, assigns them to the submission file, and exports the final CSV.

## Data reading
- pd.read_csv("/kaggle/input/llms-you-cant-please-them-all/test.csv")
- Loads test data and sample submission via pd.read_csv, and reads a word list from a local text file using standard Python file I/O.
- Reads test topics from a CSV file using pandas (`pd.read_csv`).
- Loads test data and sample submission via pandas CSV readers.
- Reads test.csv and sample_submission.csv using pandas.
- Loads test data and sample submission templates from CSV files using pandas.

## Data processing
- Dynamically constructs conversation messages by randomly selecting weighted prompt templates and suffixes.
- Trims generated model outputs by stripping numbered lists and truncating at the last punctuation mark.
- Optionally appends static strings (positive/negative review words, plagiarism scores, high scores, pass/fail tags) based on configurable probabilities.
- None explicitly stated; relies on raw topic strings with dynamic prompt suffixes and post-generation text trimming.
- Trims generated model responses to the last occurring punctuation mark (., ?, !, ]) and strips leading/trailing whitespace.
- Applies string formatting to construct prompts.
- Randomly shuffles test indices and partitions them into two groups.
- Input data requires no preprocessing beyond prompt formatting; generated responses are trimmed by stripping whitespace and truncating at the last occurring punctuation mark to remove conversational filler.
- None explicitly implemented; only basic string formatting for prompts and appending a fixed suffix to generated text.
- None explicitly shown; the notebook directly replaces the essay column in the submission DataFrame with generated prompt strings.

## Features engineering
- 

## Models
- Phi-3.5-mini-instruct
- Gemma 2 2B-it

## Frameworks used
- transformers
- torch
- pandas
- numpy
- pytorch
- random

## Ensembling
- Probabilistically mixes multiple prompt injection templates and attack vectors during inference rather than combining model predictions; optionally appends static post-processing strings to influence judge models.
- Combines multiple prompt injection templates via weighted random selection and appends randomized static strings (positive/negative words, fake plagiarism scores, high scores, pass/fail tags) to generated essays.
- Concatenates generated responses from multiple example sets with a space separator; no formal model ensembling or post-processing is applied.

## Insights
- Weighted probabilistic mixing of diverse prompt attacks increases vertical variance and reduces repetition penalties in generated essays.
- Dynamically targeting specific LLM names within prompts helps bypass model-specific safety filters.
- Appending static contextual strings (like high scores or plagiarism warnings) can subtly influence downstream judge models.
- Simple prompt engineering and strategic test-set splitting can serve as an effective baseline for text generation competitions without requiring model training.
- Using a dynamic seed and random selection prevents predictable patterns that might trigger safety mechanisms or cause model fatigue.
- Few-shot prompting with explicit counting instructions can reliably manipulate LLM-based scoring judges.
- Injecting a small percentage (~10%) of standard essays helps maintain output diversity and avoid potential detection of the exploit.
- Trimming model outputs to the last punctuation mark ensures cleaner, more predictable submission formats.
- Simple prompt injection can reliably override an LLM evaluation criteria when direct instructions are provided.
- Randomly partitioning prompts may help evade detection or exploit judge model inconsistencies.
- Zero-shot prompt engineering is sufficient to generate a valid baseline submission for LLM-based competitions.
- Dynamic prompt selection with weighted probabilities increases attack diversity and reduces repetition penalties.
- Combining few-shot examples with specific suffixes (counting, cutoff probing, emergency roleplay, surrealism) effectively bypasses standard LLM response filters.
- Structured prompting with explicit counting instructions and identity checks can reliably guide LLM outputs for multi-judge evaluation scenarios.
- Using nucleus sampling (top_p=0.9) and temperature (0.9) balances creative generation with adherence to strict constraints.
- Trimming responses at the last punctuation mark prevents trailing conversational filler from affecting evaluation metrics.
- Appending a direct instruction about desired evaluation scores to the end of generated text is a simple heuristic to potentially manipulate metrics.
- Smaller LLM variants like Gemma 2 2B can serve as fast, low-resource baselines for text generation tasks without requiring heavy GPU infrastructure.
- Custom chat templates and strict output length constraints are necessary to control model behavior and ensure consistent formatting in generation pipelines.
- Pre-trained open-weight LLMs can be deployed locally on Kaggle GPUs for rapid baseline generation without API costs.
- Adjusting the `output_len` parameter directly controls the length of generated essays.
- LLM-based judges can be manipulated through carefully crafted prompt injections.
- Splitting the test set allows applying different adversarial strategies to maximize the chance of bypassing the grader.
- Simple text generation combined with instruction override prompts can effectively exploit weak evaluation pipelines.

## Critical findings
- The author explicitly notes that adding the 'Choices' attack template resulted in a noticeable leaderboard score improvement.
- The author explicitly notes that some vertical variance is required in the output to maintain scores, warning against purely repetitive exploit patterns.
- The scoring judge appears to be vulnerable to explicit counting directives embedded within the generated text.

## Notable individual insights
- 560 (Mash It Up!): Weighted probabilistic mixing of diverse prompt attacks increases vertical variance and reduces repetition penalties in generated essays.
- 427 (Mash It Up Dynamic Seed): Using a dynamic seed and random selection prevents predictable patterns that might trigger safety mechanisms or cause model fatigue.
- 331 (Add It Up!): The scoring judge appears to be vulnerable to explicit counting directives embedded within the generated text.
- 297 ([Reproduce] Essays - simple submission): Simple prompt injection can reliably override an LLM evaluation criteria when direct instructions are provided.
- 285 (High score! (revision)): Adding the "Choices" attack template resulted in a noticeable leaderboard score improvement.

## Notebooks indexed
- #560 votes [[notebooks/votes_01_richolson-mash-it-up/notebook|Mash It Up!]] ([kaggle](https://www.kaggle.com/code/richolson/mash-it-up))
- #484 votes [[notebooks/votes_02_jiprud-essays-simple-submission/notebook|Essays - simple submission]] ([kaggle](https://www.kaggle.com/code/jiprud/essays-simple-submission))
- #427 votes [[notebooks/votes_03_kawchar85-mash-it-up-dynamic-seed/notebook|Mash It Up Dynamic Seed]] ([kaggle](https://www.kaggle.com/code/kawchar85/mash-it-up-dynamic-seed))
- #331 votes [[notebooks/votes_04_richolson-add-it-up/notebook|Add It Up!]] ([kaggle](https://www.kaggle.com/code/richolson/add-it-up))
- #297 votes [[notebooks/votes_05_takuji-reproduce-essays-simple-submission/notebook|[Reproduce] Essays - simple submission]] ([kaggle](https://www.kaggle.com/code/takuji/reproduce-essays-simple-submission))
- #293 votes [[notebooks/votes_06_mahdibayanloo-high-score-revision/notebook|High score! (revision)]] ([kaggle](https://www.kaggle.com/code/mahdibayanloo/high-score-revision))
- #285 votes [[notebooks/votes_07_kawchar85-13-003-lb-add-it-up-for-two-judge/notebook|[13.003 LB] Add It Up for two judge]] ([kaggle](https://www.kaggle.com/code/kawchar85/13-003-lb-add-it-up-for-two-judge))
- #277 votes [[notebooks/votes_08_yururoi-gemma2-2b-highest-or-lowest/notebook|Gemma2 2b highest or lowest]] ([kaggle](https://www.kaggle.com/code/yururoi/gemma2-2b-highest-or-lowest))
- #272 votes [[notebooks/votes_09_nischaydnk-gemma-2-baseline-generating-essays-w-llms/notebook|Gemma 2 Baseline: Generating Essays w/ LLMs ]] ([kaggle](https://www.kaggle.com/code/nischaydnk/gemma-2-baseline-generating-essays-w-llms))
- #223 votes [[notebooks/votes_10_bestwater-fork-of-essays-simple-submission-try-more/notebook|Fork of Essays - simple submission | Try More ]] ([kaggle](https://www.kaggle.com/code/bestwater/fork-of-essays-simple-submission-try-more))
