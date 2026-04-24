# kaggle-measuring-agi: top public notebooks

The top-voted notebooks collectively focus on rigorous LLM evaluation, emphasizing calibration, instruction compliance, and hallucination resistance across synthetic and adversarial benchmarks. Authors frequently introduce novel metrics like Expected Calibration Error (ECE), Cognitive Geometry Score (CGS), and selective attention indices to quantify metacognition and spatial reasoning. Alongside these evaluations, several entries provide practical tutorials on the Kaggle Benchmarks SDK, structured output enforcement, and post-hoc calibration pipelines.

## Common purposes
- other
- tutorial
- utility
- eda

## Models
- microsoft/phi-2
- Gemini 2.5 Flash
- Gemini 2.5 Pro
- Qwen 3 Next 80B
- Gemma 3 27B
- Claude Sonnet 4.6
- OPT-1.3b
- Gemini 1.5 Flash
- Gemini 3.1 Pro Preview
- Gemini 3 Pro Preview
- Claude Opus 4.6
- GLM-5
- Qwen 3 Next 80B Thinking
- DeepSeek V3.1
- DeepSeek V3.2
- Qwen 3 Next 80B Instruct
- Gemini 3.1 Flash-Lite Preview
- Qwen 3 Coder 480B
- DeepSeek-R1
- Gemma 3 12B
- Gemma 3 4B
- GPT4_Overconfident
- Claude3_Cautious
- Gemini_Balanced
- Llama3_Uncertain
- Mistral_Erratic
- PerfectAGI
- google/gemma-3-12b
- google/gemini-2.5-flash
- google/gemini-3.1-pro-preview
- GPT4_Unfocused
- Claude3_Selective
- Llama3_Cautious
- Mistral_Scattered
- Perfect_Attention
- google/gemini-3.1-flash-lite
- Gemini 3 Deep Think
- o3
- GPT-5.4 Pro
- Claude 4 Opus
- Llama 4 Maverick
- Qwen3 72B
- o4-mini
- Mistral Large 3
- Human Expert
- Grok 4 Thinking
- GPT-4o
- GPT-5.4 Mini
- Gemini 1.5 Pro
- Llama 4 Scout
- DeepSeek-V3
- Kimi K2.5
- Phi-4 14B

## Frameworks
- transformers
- torch
- pandas
- numpy
- matplotlib
- re
- json
- collections
- datetime
- datasets
- scipy
- ipywidgets
- kaggle_benchmarks
- os
- time
- random
- seaborn
- plotly
- scikit-learn
- pydantic
- itertools
- wordcloud

## CV strategies
- StratifiedKFold(n_splits=5, shuffle=True, random_state=2026)

## Preprocessing
- Prompt stripping and first-line extraction
- Whitespace normalization
- Non-alphanumeric character removal and lowercasing for non-JSON tasks
- Regex-based JSON extraction via re.search(r"\{.*?\}", text, re.DOTALL)
- AutoTokenizer truncation to 128 tokens
- logit padding to length 4 with -999.0
- float16 precision
- StandardScaler fit on train-only folds (clean) vs full dataset (leaked)
- Synthetic data generation with clipped normal distributions and binomial sampling for answerability/misconception flags
- CSV loading
- column renaming to artist/title/lyrics
- warning suppression
- JSON string cleaning (removing markdown code block wrappers)
- Pydantic schema validation for structured output
- Coordinate normalization to [0,1]x[0,1] for size-invariant scoring
- Proxy dimension derivation from published benchmarks using weighted averages
- Handling missing ARC_AGI2 values with fallback mapping

## Loss functions
- CrossEntropyLoss (for temperature scaling optimization)

## Post-processing
- Strict exact-match evaluation combining factual correctness and format compliance; JSON schema validation; relaxed vs strict JSON evaluation modes; deterministic output extraction to isolate compliance from sampling noise.
- Post-hoc temperature scaling via LBFGS to minimize NLL, followed by confidence-based abstention thresholding (θ) to trigger safe rejection of low-confidence predictions.
- Temperature scaling via ECE minimization (logit(conf)/T); selective abstention gate triggering when calibrated confidence falls below 40% safety threshold
- Confidence score thresholding (>70 for TruthGuard, >80 for MathGuard) and strict JSON parsing with fallback error handling.

## What worked
- Format constraint tasks show perfect compliance across all evaluated models.
- Multi-instruction tasks show high variance but strong performance for Gemma 3 27B (18/23).
- JSON constraint tasks reveal a clear compliance gap where models either pass completely or fail entirely.
- Temperature scaling successfully reduced ECE without retraining the model
- Cross-lingual ECE measurement effectively revealed an Arabic overconfidence gap
- Claude Sonnet 4.6 and Opus 4.6 successfully recalibrate confidence and maintain high accuracy under misleading context.
- Qwen 3 Coder 480B demonstrates metacognitive honesty by lowering confidence on incorrect answers, making it viable for confidence-weighted RAG pipelines.
- Temperature scaling successfully reduced ECE across all simulated models
- Clean cross-validation correctly exposed generalization gaps
- Composite AGI Safety Score effectively ranked model reliability
- LLM-as-a-judge assertions and hard keyword checks passed successfully in Bengali
- Using kbench.llm as a placeholder enables seamless model swapping via the Kaggle UI
- Mid-tier models like Llama 4 Maverick offer a 40% better ROI for geometric reasoning tasks compared to frontier models.

## What did not work
- Context interference causes extreme performance variance, with Claude dropping to 1/21.
- Claude Sonnet 4.6 fails multi-instruction tasks entirely (0/23).
- Phi-2 completely fails JSON constraint tasks (0%), producing plain text instead of structured output.
- Constrained prompting degrades performance by ~20% for some models, indicating instruction pressure can destabilize compliance.
- Eleven of seventeen models (all Gemini variants, Gemma family, Qwen Thinking, DeepSeek-R1, GLM-5) exhibit zero confidence variance on T3, outputting 100% confidence regardless of correctness.
- Gemma models suffer catastrophic accuracy collapse (<13.3% on T3) while maintaining constant confidence.
- Leaked preprocessing (fitting scaler on full data) artificially inflated validation scores and masked true model failure
- Overconfident models showed high accuracy but catastrophic NLL penalties when wrong
- google/gemini-3.1-flash-lite failed to recognize the author's identity when asked "What is mpwolke Kaggler?"
- Standard LLMs fail significantly on axis flips and color changes, indicating reliance on visual pattern matching rather than logical reasoning.
- Topological relationships (containment/connection) remain a major bottleneck across all tested models.

## Critical findings
- High factual accuracy does not guarantee reliable behavior; models frequently produce correct answers but violate strict formatting constraints.
- Increased instruction pressure (constrained mode) can degrade performance rather than improve it, revealing fragility under constraint.
- Model scale does not ensure improved instruction compliance; failures stem from control and prioritization limitations rather than reasoning capacity.
- OPT-1.3b exhibits significantly higher overconfidence (worse ECE) in Arabic compared to English despite similar accuracy.
- Manual curation of Arabic TruthfulQA questions is necessary for reliable cross-lingual calibration benchmarking.
- Most models cannot express uncertainty under adversarial context, masking true calibration failure behind constant high confidence.
- Gemini 3 Pro Preview's positive ECE Δ is a mathematical artifact of accuracy shifting rather than genuine recalibration.
- Brier score validation confirms ECE tier rankings but shows Tier 1 Δ confidence intervals cross zero at N=30.
- Overconfidence on incorrect answers incurs a catastrophic NLL penalty, making calibration more critical than raw accuracy.
- Fitting preprocessors on test data leaks distribution info, creating a false sense of model competence.
- Models that honestly abstain on unanswerable questions are safer than those that confidently hallucinate.
- LLMs lack deterministic FPUs and rely on token approximation, causing catastrophic floating-point errors when transcendental functions are chained.
- Fast models exhibit rigid Gaussian latency spikes (System-1), while reasoning models show heavy-tailed latency distributions (System-2) indicating dynamic test-time compute scaling.
- Mode collapse correlates strongly with low inhibitory control, as models repeatedly output identical hallucinated strings when failing.
- High raw recall is insufficient for attention tasks; models that retrieve everything amplify noise rather than filter it.
- The SiN Score is zero if a model finds nothing or returns everything, proving that genuine selective attention requires a narrow balance between recall and intrusion.
- Proxy dimensions derived from general benchmarks do not directly measure spatial drawing ability, creating a potential evaluation gap.
- A geometric integrity gap (Abstraction minus Geometry) exceeding 15 points strongly correlates with hallucination or 'faking' behavior.
- Cost-per-CGS-Point analysis reveals that massive model scale is not always the most efficient path for specialized spatial applications.

## Notable techniques
- Deterministic synthetic dataset generation with fixed seed (42) for reproducible task distribution
- Dual-mode prompting (normal vs constrained with explicit STRICT RULES block) to test constraint robustness
- Strict binary evaluation metric combining factual correctness and format compliance with zero tolerance
- Regex-based JSON extraction followed by schema validation to isolate capability from compliance
- Deterministic inference configuration (do_sample=False, max_new_tokens=50, prompt stripping) to isolate instruction-following from sampling noise
- Post-hoc temperature scaling optimized with LBFGS to minimize NLL on validation logits
- Cross-lingual Expected Calibration Error (ECE) comparison to detect resource bias
- Confidence thresholding for safe abstention in high-stakes deployments
- Logit extraction via negative loss computation on choice prompts
- Strict JSON prompt enforcement with regex fallback parsing for LLM responses
- ECE computation using 5 fixed bins with `score = 1 - ECE`
- Stratified round-robin dataset splitting with fixed seed for deterministic task allocation
- Context-aware scoring that rewards correct answers only when the model correctly identifies context utility
- Brier score validation to verify ECE rankings without binning artifacts
- Temperature scaling optimized via ECE minimization to correct over/underconfidence
- Unknown-Unknowns rate metric to flag confident predictions on unanswerable questions
- Misconception Trap Rate to quantify confident hallucination on known myths
- Train-val gap comparison to detect silent data leakage during preprocessing
- Interactive reliability diagram explorer using ipywidgets for real-time calibration tuning
- Using kaggle_benchmarks for structured LLM evaluation with keyword assertions and LLM-as-a-judge criteria
- Custom wordcloud generation with extended stopwords and configurable color/mask parameters
- Implicit Calibration Error (ICE) metric to quantify mode collapse and hallucination stubbornness
- Dual-tolerance grading (strict 6-decimal vs relaxed 4-decimal) to isolate arithmetic approximation errors
- Telemetry extraction from Kaggle benchmark JSON logs for latency, token, cost, and TPS analysis
- Heavy-tailed latency distribution plotting to distinguish System-1 heuristics from System-2 test-time compute scaling
- Domain-Switching Cost (DSC) and Compute Yield Ratio (CYR) metrics to measure cognitive flexibility and working memory efficiency
- Pydantic BaseModel for strict LLM output schema enforcement
- Assertion-based benchmarking with `kbench.assertions.assert_true` for automated pass/fail auditing
- Structured prompt engineering forcing JSON-only responses to eliminate conversational filler
- Domain-specific adversarial trap design across temporal, mathematical, and game-theory constraints
- Procedural story generation with randomized proper nouns, numbers, and dates to guarantee zero training data overlap
- Simulating model behavior via parametric behavioral profiles (recall, intrusion, difficulty penalty, noise sensitivity) instead of running actual inference
- Composite SiN Score metric (Recall × (1 − Intrusion)) to penalize both under- and over-retrieval
- Noise robustness curves by binning noise ratios to evaluate performance degradation under clutter density
- Using kbench.llm as a UI-compatible placeholder for dynamic model swapping
- Applying schema=bool to force structured boolean outputs from LLMs
- Combining hard-rule keyword assertions with LLM-as-a-judge criteria for multi-tier evaluation
- Modifying open-source license text via regex to test LLM legal reasoning boundaries
- CGS v3 radar polygon area calculation using the Shoelace formula for balanced capability scoring
- Pareto efficiency front computation to identify models not dominated in both accuracy and cost
- Geometric integrity gap metric (Abstraction - Geometry) to detect hallucination/faking
- Synthetic ARC task generator using random grid manipulation and hole-punching to prevent data contamination
- Size-invariant coordinate normalization for robust geometric matching

## Notable individual insights
- votes 69 (Your Model Heard You. It Just Didn’t Obey.): High factual accuracy does not guarantee reliable behavior; models frequently produce correct answers but violate strict formatting constraints.
- votes 56 (🥇TruthGuard): OPT-1.3b exhibits significantly higher overconfidence (worse ECE) in Arabic compared to English despite similar accuracy.
- votes 41 (16 LLMs Fail Adversarial Context. One Survives.): Eleven of seventeen models exhibit zero confidence variance on adversarial tasks, effectively failing to express uncertainty.
- votes 34 (The Hallucination Trap: Shattering System II Logic): LLMs lack deterministic FPUs and rely on token approximation, causing catastrophic floating-point errors when transcendental functions are chained.
- votes 32 (The Oracle's Sieve): Pydantic schema enforcement combined with assertion-based benchmarking effectively eliminates conversational filler and enforces structured reasoning.
- votes 26 (truth guard🤩You cannot fake geometry🤩): A geometric integrity gap (Abstraction minus Geometry) exceeding 15 points strongly correlates with hallucination or 'faking' behavior.
- votes 29 (Ready to kbench? Better call Saul in Bengali): Using kbench.llm as a UI-compatible placeholder enables seamless model swapping without code changes.

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
