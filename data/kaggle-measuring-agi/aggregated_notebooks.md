# kaggle-measuring-agi: top public notebooks

The community's top-voted notebooks primarily focus on LLM evaluation, post-hoc calibration, and synthetic benchmarking rather than traditional model training or tabular feature engineering. They collectively explore constraint-driven instruction compliance, metacognitive auditing under adversarial context, and the development of custom cognitive metrics like ECE and the SiN Score. These works emphasize rigorous, programmatic evaluation pipelines and reliability diagnostics over standard architecture tuning or baseline modeling.

## Common purposes
- other
- tutorial
- utility

## Competition flows
- Synthetic tasks are generated and formatted into prompts, passed through deterministic LLM inference pipelines, evaluated against a strict zero-tolerance compliance metric, and analyzed across multiple models to benchmark instruction-following robustness.
- Loads the TruthfulQA validation split, runs OPT-1.3b to extract choice log-probabilities, computes raw ECE, optimizes temperature scaling via LBFGS to calibrate confidence, and compares calibrated vs. raw metrics across English and Arabic prompts.
- Loads a JSON dataset of 100 domain-stratified problems, prompts 17 LLMs via kaggle_benchmarks with bare and adversarial context templates, parses JSON responses, computes accuracy and calibration metrics, and outputs tiered rankings with visualizations.
- Synthetically generates a 1,200-question multi-domain dataset → simulates six foundation models and applies temperature scaling calibration → outputs ECE, safety scores, and leakage detection reports.
- Defines an LLM benchmark task using `kbench`, prompts a model to answer a question, evaluates the response with assertions, and briefly loads/visualizes a lyrics dataset before halting due to environment installation conflicts.
- Generates a synthetic algebraic-transcendental math dataset, evaluates three Google LLMs using strict and relaxed grading, extracts telemetry logs, computes novel cognitive metrics, and visualizes performance versus compute allocation.
- The notebook defines three adversarial prompt datasets, formats them into pandas DataFrames, and iteratively evaluates a Kaggle-hosted LLM using strict JSON-constrained prompts and programmatic assertions to verify logical rejection of impossible premises.
- Procedurally generates 200 synthetic narratives with signal and distractor facts, simulates responses from six model archetypes using behavioral parameters, computes custom attention metrics, and outputs a ranked submission CSV.
- The notebook defines custom benchmark tasks using the kaggle_benchmarks library, runs them on placeholder and specific LLMs, applies keyword and LLM-judge assertions, and demonstrates cross-lingual and prompt-injection evaluation capabilities on Kaggle's benchmark platform.
- Loads hardcoded benchmark scores for 20 models → derives cognitive dimension proxies using weighted mappings → calculates custom CGS metrics and Pareto efficiency → visualizes rankings, geometric integrity gaps, and cost-performance trade-offs.

## Data reading
- Data is not loaded from external files but synthetically generated in-memory using Python's random module and collections.Counter, then persisted to JSON format via json.dump.
- Loads the 'truthful_qa' dataset (multiple_choice split, validation) using Hugging Face `datasets.load_dataset`.
- Manually curates a list of 20 Arabic-translated questions with corresponding choices and correct answer indices.
- Reads candidates.json from Kaggle inputs (/kaggle/input/cask-dataset/ or /kaggle/input/datasets/darovalos/cask-dataset/) and parses it into a list of dictionaries containing questions, answers, aliases, and verification types.
- Synthetic dataset generated in-memory via Python loops and pandas DataFrame.
- Benchmark telemetry logs parsed from /kaggle/working/*run.json using json.load.
- Loads a CSV file (`lyrics-TUPAC.csv`) using `pd.read_csv` and renames columns to `['artist', 'title', 'lyrics']`.
- Hardcoded Python lists of dictionaries containing questions and category labels are converted into pandas DataFrames (df_tasks, df_math, df_yugi).
- Procedurally generated in-memory using Python's random and numpy libraries; structured into a pandas DataFrame without external file loading.
- Data is hardcoded directly in Python dictionaries (`PUBLISHED_BENCHMARKS`) and DataFrames; no external file loading, CSV parsing, or live API calls are executed in the provided code.

## Data processing
- Prompts are constructed programmatically by concatenating instructions, questions, and format constraints; model outputs are post-processed by stripping the prompt prefix, extracting the first line, normalizing whitespace, and applying regex-based cleaning for strict evaluation.
- Truncates input prompts to a maximum length of 128 tokens.
- Pads logit arrays to length 4 using -999.0 as a constant padding value.
- Converts log-probabilities to probabilities using `scipy.special.softmax`.
- Bins confidence scores into 10 equal-width intervals for ECE calculation.
- Applies stratified round-robin ordering by domain with seed 42, splits data into POOL_A (30 problems for T1/T3) and POOL_B (65 problems for T2), cyclically assigns context types (helpful/irrelevant/misleading) to POOL_B, and uses regex-based JSON parsing with fallback string matching for LLM responses.
- Synthetic features clipped to [0,1] ranges using np.clip
- Misconception flags sampled via np.random.binomial
- Answerability perturbed with Gaussian noise and bounded to [0.0, 1.0]
- Generates 100 math problems with conditional routing and transcendental functions.
- Parses JSON logs to extract answers, targets, latency, tokens, and costs.
- Applies regex to isolate numeric outputs from model responses.
- Implements strict (1e-7) and relaxed (1e-4) tolerance grading.
- Computes novel cognitive metrics (ICE, DSC, CYR, Rambling Index) from telemetry.
- Raw LLM responses are post-processed by stripping markdown code block wrappers (````json) and whitespace before JSON parsing.
- Procedural story generation with randomized entity pools, difficulty-based concealment depths, and noise ratio computation; no traditional cleaning or tokenization.
- Benchmark scores are weighted and mapped to derive proxy cognitive dimensions; custom CGS formulas apply coordinate normalization, Shannon entropy calculation, and geometric distance metrics to compute final scores.

## Features engineering
- difficulty (computed as 1.0 - answerability)
- ECE (Expected Calibration Error)
- Unknown-Unknowns rate (confident predictions on unanswerable questions)
- Misconception Trap rate (confidently wrong on myth/pseudoscience categories)
- AGI Safety Score (composite metric weighting calibration, safety, accuracy, and misconception rates)

## Models
- microsoft/phi-2
- facebook/opt-1.3b
- Gemini 1.5 Flash (via API)
- Gemini 3.1 Pro Preview
- Claude Sonnet 4.6
- Gemini 3 Pro Preview
- Claude Opus 4.6
- Gemini 2.5 Pro
- GLM-5
- Qwen 3 Next 80B Thinking
- DeepSeek V3.1
- DeepSeek V3.2
- Gemini 2.5 Flash
- Qwen 3 Next 80B Instruct
- Gemini 3.1 Flash-Lite Preview
- Qwen 3 Coder 480B
- DeepSeek-R1
- Gemma 3 27B
- Gemma 3 12B
- Gemma 3 4B
- GPT4_Overconfident (simulated)
- Claude3_Cautious (simulated)
- Gemini_Balanced (simulated)
- Llama3_Uncertain (simulated)
- Mistral_Erratic (simulated)
- PerfectAGI (simulated)
- LogisticRegression (sklearn)
- google/gemini-2.5-flash
- kbench.llm (placeholder)
- google-gemini-3.1-flash-lite
- GPT4_Unfocused
- Claude3_Selective
- Llama3_Cautious
- Mistral_Scattered
- Perfect_Attention

## Frameworks used
- transformers
- torch
- pandas
- numpy
- matplotlib
- tqdm
- json
- re
- datasets
- scipy
- ipywidgets
- kaggle_benchmarks
- os
- time
- collections
- seaborn
- scikit-learn
- plotly
- kbench
- wordcloud
- pydantic
- random
- itertools
- requests
- IPython

## Loss functions
- nn.CrossEntropyLoss (used internally to optimize temperature scaling via LBFGS)
- None (ECE minimized via scipy.optimize.minimize_scalar for calibration; NLL used conceptually for visualization only)

## CV strategies
- StratifiedKFold(n_splits=5, shuffle=True, random_state=2026) on synthetic features (answerability, difficulty, is_misconception)

## Insights
- High factual accuracy does not guarantee reliable behavior, as models frequently fail strict formatting and conflict-resolution constraints.
- Instruction compliance is highly sensitive to constraint type and prompting conditions rather than model scale alone.
- Failures stem from control and execution limitations rather than reasoning or knowledge deficits.
- Structured output requirements expose significant compliance gaps even when models possess the underlying capability.
- Post-hoc temperature scaling significantly reduces ECE (often by 50-60%) without requiring model retraining or architectural changes.
- Calibrated confidence scores are essential for implementing reliable abstention thresholds in safety-critical AI deployments.
- Cross-lingual evaluation exposes hidden reliability gaps that accuracy-only metrics fail to capture.
- Most frontier LLMs completely lose metacognitive capacity under misleading context, outputting constant 100% confidence regardless of correctness.
- Metacognitive honesty (expressing uncertainty) is decoupled from contextual robustness (maintaining accuracy under pressure).
- Standard calibration metrics like ECE can be misleading when confidence variance is zero, as they reduce to accuracy shifts rather than true recalibration.
- Claude Sonnet 4.6 and Opus 4.6 are the only evaluated models that successfully recalibrate confidence while maintaining high accuracy under adversarial context.
- High accuracy combined with high confidence is dangerous if the model is miscalibrated.
- Data leakage during preprocessing (e.g., fitting scalers on full data) artificially inflates validation scores and masks true generalization gaps.
- A model that correctly abstains on unanswerable questions is safer than one that confidently hallucinates.
- Calibration metrics like ECE and Unknown-Unknowns rates should be reported alongside accuracy for reliable AI deployment.
- Automated LLM evaluation can be structured using both deterministic keyword checks and LLM-as-a-judge criteria for nuanced quality assessment.
- The `kbench` library provides a streamlined interface for defining benchmark tasks and programmatically validating model outputs.
- Environment dependency management can quickly halt notebook execution, as seen when installing `torch` caused failures.
- LLMs optimized for linear prediction suffer from mode collapse when facing complex logical traps, often repeating identical hallucinations.
- Test-time compute scaling reveals a heavy-tailed distribution for reasoning models, contrasting with the rigid, low-variance inference of fast models.
- Accuracy metrics mask underlying arithmetic limitations; relaxing tolerance thresholds exposes the FPU Ceiling where models understand routing but fail floating-point approximation.
- Executive function metrics provide a more nuanced view of model capability than accuracy alone, highlighting working memory decay and cognitive flexibility.
- Structured JSON constraints and explicit confidence scoring effectively filter out conversational filler and force models to self-audit their logical reasoning.
- Programmatic assertions on both detection flags and confidence thresholds provide a rigorous, machine-readable audit of LLM reliability.
- Adversarial prompts targeting domain-specific constraints reveal specific failure modes that standard factual benchmarks might miss.
- High recall without selectivity results in zero attention, as models that retrieve all distractors fail the core task.
- Procedural generation with randomized entities and dates effectively isolates attention from memorization.
- The SiN Score (Recall × (1 − Intrusion)) successfully penalizes both under-retrieval and over-retrieval to capture selective focus.
- Performance degrades predictably with increasing noise ratio and difficulty, but models with lower base intrusion rates maintain higher composite scores.
- The kaggle_benchmarks library simplifies custom LLM evaluation by wrapping prompts and assertions in a simple @task decorator.
- Evaluation pipelines can effectively combine fast, deterministic keyword checks with flexible LLM-as-a-judge assessments for nuanced criteria.
- The platform supports seamless model swapping via kbench.llm, allowing benchmark tasks to run across different frontier models without code modifications.
- Cross-lingual prompts and complex prompt-injection scenarios can be reliably tested using the same task structure and assertion framework.
- Mid-tier models like Llama 4 Maverick offer a 40% better cost-performance ROI for geometric reasoning tasks than frontier models.
- Standard LLMs fail significantly under axis flips or color changes, proving they rely on visual pattern matching rather than true logical reasoning.
- Topological relationships like containment and connectivity remain the primary bottleneck for current AI spatial reasoning capabilities.

## Critical findings
- Performance drops by 20% when stricter rules are explicitly enforced, showing that constraint pressure can destabilize compliance.
- Claude Sonnet 4.6 achieves perfect JSON compliance but completely fails multi-instruction and context interference tasks.
- Format constraint tasks show 100% compliance across all models, while JSON constraint tasks reveal the widest performance variance.
- Context interference causes extreme performance variance across models, ranging from 1/21 to 18/21.
- The OPT-1.3b model exhibits a structural overconfidence gap in Arabic, assigning high raw confidence to hallucinated answers while remaining well-calibrated in English.
- Raw softmax probabilities systematically overestimate true accuracy, making uncalibrated logits unreliable for automated safety gating.
- 11 of 17 models output exactly confidence=100 on every T3 problem, showing zero variance and an inability to express uncertainty.
- Gemini 3 Pro Preview's apparent recalibration is a mathematical artifact of constant confidence reducing 1-ECE to accuracy.
- Gemma models collapse catastrophically on misleading context (accuracy < 5%) while maintaining 100% confidence across a 7x parameter range.
- DeepSeek-R1's T1 anomaly stems from prompt format non-compliance rather than a knowledge deficit.
- Simulated overconfident models exhibit high Unknown-Unknowns rates, confidently answering impossible questions.
- Leaked models show artificially low train-val gaps and inflated public scores but collapse on private/unseen distributions.
- Optimal temperature scaling values vary significantly across models, with some requiring T > 1 to soften overconfidence and others needing T < 1.
- The Misconception Trap Rate reveals that models are most confidently wrong on pseudoscience and conspiracy theory categories.
- Advanced models like Gemini 3.1 Flash-lite failed to recognize the author's public Kaggle identity despite years of platform activity.
- The LLM judge and keyword assertions performed reliably for Bengali prompts, confirming strong cross-lingual evaluation support within the framework.
- GPT-4 achieves high raw recall (71.3%) but fails the attention test with a 68% distractor intrusion rate, proving retrieval abundance does not equal cognitive selectivity.
- Llama 3 ranks lower on recall but outperforms GPT-4 on the SiN Score by prioritizing selectivity over completeness.
- Current models treat attention tasks like full-document summarization rather than targeted extraction, as shown by Mistral's 70.8% intrusion rate.
- High abstraction scores paired with low geometry scores serve as a reliable indicator of hallucinated or fake reasoning rather than genuine spatial understanding.
- ARC-AGI-2 performance data is largely estimated or unreported across the industry, revealing a critical gap in standardized spatial reasoning benchmarks.

## What did not work
- Installing `torch` stopped the notebook execution.
- The author notes that Gemini 3.1 Flash-lite failed to recognize their Kaggle identity, and the 'Better Call Saul' prompt injection test yielded unclear or unexpected results.

## Notable individual insights
- votes 69 (Your Model Heard You. It Just Didn’t Obey.): Instruction compliance is a context-sensitive control problem rather than a unified capability, with constraint pressure destabilizing compliance even in advanced systems.
- votes 56 (🥇TruthGuard): Post-hoc temperature scaling significantly reduces ECE without retraining, but cross-lingual evaluation reveals structural calibration gaps where models are significantly more overconfident in Arabic than English.
- votes 41 (16 LLMs Fail Adversarial Context.): Most frontier LLMs completely lose metacognitive capacity under misleading context, outputting constant 100% confidence regardless of correctness, decoupling metacognitive honesty from contextual robustness.
- votes 34 (Tupac's California Love Benchmark): Automated LLM evaluation can be effectively structured using both deterministic keyword checks and LLM-as-a-judge criteria, though environment dependency conflicts can quickly halt execution.
- votes 34 (The Hallucination Trap: Shattering System II Logic): Accuracy metrics mask underlying arithmetic limitations; relaxing tolerance thresholds exposes the FPU Ceiling where models understand routing but fail floating-point approximation.
- votes 31 (Signal in the Noise (SiN)): High recall without selectivity results in zero attention, as models that retrieve all distractors fail the core task, proving retrieval abundance does not equal cognitive selectivity.
- votes 26 (truth guard😏You cannot fake geometry😏): High abstraction scores paired with low geometry scores serve as a reliable indicator of hallucinated or fake reasoning rather than genuine spatial understanding.

## Notebooks indexed
- #69 votes [[notebooks/votes_01_eng20cs0359sriramm-your-model-heard-you-it-just-didn-t-obey/notebook|Your Model Heard You. It Just Didn’t Obey.]] ([kaggle](https://www.kaggle.com/code/eng20cs0359sriramm/your-model-heard-you-it-just-didn-t-obey))
- #56 votes [[notebooks/votes_02_aminmahmoudalifayed-truthguard/notebook|🥇TruthGuard]] ([kaggle](https://www.kaggle.com/code/aminmahmoudalifayed/truthguard))
- #41 votes [[notebooks/votes_03_darovalos-16-llms-fail-adversarial-context-one-survives/notebook|16 LLMs Fail Adversarial Context. One Survives.]] ([kaggle](https://www.kaggle.com/code/darovalos/16-llms-fail-adversarial-context-one-survives))
- #37 votes [[notebooks/votes_04_aminmahmoudalifayed-truth-guard-your-model-is-lying/notebook|truth guard😏🦅Your Model Is Lying]] ([kaggle](https://www.kaggle.com/code/aminmahmoudalifayed/truth-guard-your-model-is-lying))
- #34 votes [[notebooks/votes_05_mpwolke-tupac-s-california-love-benchmark/notebook|Tupac's California Love Benchmark]] ([kaggle](https://www.kaggle.com/code/mpwolke/tupac-s-california-love-benchmark))
- #34 votes [[notebooks/votes_06_anikkirtania-the-hallucination-trap-shattering-system-ii-logic/notebook|The Hallucination Trap: Shattering System II Logic]] ([kaggle](https://www.kaggle.com/code/anikkirtania/the-hallucination-trap-shattering-system-ii-logic))
- #32 votes [[notebooks/votes_07_mah20050-the-oracle-s-sieve/notebook|🃏 The Oracle's Sieve]] ([kaggle](https://www.kaggle.com/code/mah20050/the-oracle-s-sieve))
- #31 votes [[notebooks/votes_08_parsahriri-signal-in-the-noise-sin/notebook| Signal in the Noise (SiN)]] ([kaggle](https://www.kaggle.com/code/parsahriri/signal-in-the-noise-sin))
- #29 votes [[notebooks/votes_09_mpwolke-ready-to-kbench-better-call-saul-in-bengali/notebook|Ready to kbench? Better call Saul in Bengali]] ([kaggle](https://www.kaggle.com/code/mpwolke/ready-to-kbench-better-call-saul-in-bengali))
- #26 votes [[notebooks/votes_10_aminmahmoudalifayed-truth-guard-you-cannot-fake-geometry/notebook|truth guard😉You cannot fake geometry😉]] ([kaggle](https://www.kaggle.com/code/aminmahmoudalifayed/truth-guard-you-cannot-fake-geometry))
