# deep-past-initiative-machine-translation: cross-solution summary

This competition focused on Akkadian-to-English machine translation, requiring models to accurately transliterate and translate cuneiform texts from highly variable PDFs and CSV sources. Winning approaches universally prioritized data-centric engineering over architectural complexity, leveraging extensive LLM-assisted OCR, synthetic data generation, and rigorous orthographic normalization to overcome severe data scarcity. Successful submissions consistently scaled ByT5 models, employed Minimum Bayes Risk or pairwise reward ensembling, and treated checkpoint selection and training duration with extreme caution to prevent format memorization and model collapse.

## Competition flows
- Raw PDFs and OARE database processed via OCR and LLMs to create curated training datasets, trained on an ensemble of quantized ByT5-XL models, with inference merged and scored via a weighted MBR selector.
- Raw PDFs and CSV files cleaned and augmented via LLM APIs and external dictionary scraping, fed into an ensemble of 15 ByT5 models trained with dynamic context sampling and adversarial regularization, aggregated via MBR ensembling.
- Raw CSVs, academic PDFs, and OCR texts cleaned, deduplicated, and augmented with LLM-generated synthetic drills and pseudo-labels, used for a two-stage CPT/FT pipeline on ByT5 models, ensembled via a Qwen3-8B pairwise Reward Model.
- Official CSVs and external academic PDFs processed through a three-stage LLM alignment pipeline and OCR extraction, normalized for orthographic consistency, chunked into document-level sequences, and trained on a vanilla ByT5-Large model with Adafactor.
- Raw PDFs and published texts processed via Gemini OCR to generate synthetic transliteration/translation pairs, cleaned and used to post-train Qwen2.5 and ByT5 models with LoRA, followed by regex normalization and LLM-assisted deduplication.
- Training data expanded fourfold using Turkish-to-English translation, external PDFs, and pseudolabeling, then fine-tuned on a ByT5-XL model as a CasualLM for three epochs.
- Raw PDFs and publication texts processed through layout-specific OCR and VLM pipelines to generate high-quality translation pairs, used in a two-stage supervised fine-tuning process on ByT5-XL.

## Data reading
- PDFs parsed via PyMuPDF and GLM-OCR (extracting page/field coordinates and text)
- CSV files loaded directly
- External dictionary data scraped from the OARE website
- OCR outputs structured as JSON dictionaries mapping page/field indices to text
- CSV files (train.csv, Sentences_Oare, published_texts.csv, publications.csv, onomasticon.csv) and PDF documents parsed via multimodal LLMs and OCR pipelines
- CSV files (train.csv, published_texts.csv, S_meta) parsed via standard tools; academic PDFs OCR-processed and extracted via an LLM-based pipeline; custom dataset generator handles multi-version backups and deduplication
- PDF pages and published texts fed into Gemini for OCR and sentence splitting; competition CSV data loaded via pandas for validation perturbation
- PDF pages and publication texts processed through layout-specific OCR and VLM pipelines

## Data processing
- Extracted from PDFs using GLM-OCR and Gemini-3.0-pro-preview with prompts for logical segmentation, 3D tablet continuity, grammar adaptation, and anchor words.
- Generated synthetic data for dictionary coverage gaps and free-form sentence pairs.
- Iteratively cleaned data by training 4-fold CV models, inferring on training folds, and filtering hard samples using geo_metric < 20 or character length differences > 100.
- Discarded official train.csv; applied minimal post-processing to avoid overfitting; one variant cleaned repeated n-grams (n=2-7).
- Pre-compiled and quantized models to int8-float32 for inference.
- Unicode NFKC normalization, zero-width character removal, dash/hyphen unification.
- Diacritic simplification and removal of academic editorial symbols.
- Fraction/decimal formatting and Roman numeral conversion.
- Determinative/script normalization and sign composition cleanup.
- Gap standardization (<gap>) and dictionary-level hard replacement.
- Aggressive removal of translational explanations and academic annotations.
- Measurement unit expansion and punctuation/typographical correction.
- External dictionary scraping from OARE and LLM-based pseudo-labeling.
- Turkish-to-English translation via LLM APIs and dictionaries.
- Training augmentations: dynamic context sampling (Zipf/Geometric mixture for context length, random context insertion), name capitalization randomization, and dictionary-based norm-to-lexeme mapping.
- Deduplication by oare_id.
- LLM-based sentence splitting and alignment (Gemini 3 Pro with CoT).
- LLM judge quality scoring (high/medium/low).
- Deterministic normalization for transliteration and translation.
- Upsampling high-quality subsets (2x).
- Name swap augmentation.
- Embedding-based few-shot retrieval (NVIDIA NV-Embed).
- Synthetic data generation (vocab/grammar drills, slot-fill templates, PN vs word contrastive, weight conversion, synthetic tablets).
- Pseudo-labeling with fine-tuned ByT5 models.
- Three-stage LLM pipeline for sentence alignment (Breaker, Fixer, Generator); OCR extraction of ~60 external academic PDFs; custom text normalization unifying highly variable cuneiform orthography (e.g., sz/s→š, subscript vowels to accents, hamza deletion); prefer-EN deduplication to prevent conflicting translations; 2× boosting of English academic papers; multi-version LLM extraction passes for implicit augmentation; concatenation of consecutive sentences into 768-byte document-level chunks; group_by_length batching to minimize padding waste.
- External data generated via Gemini OCR on PDFs and published texts (~55k pairs).
- Preprocessing fixed formatting bugs (e.g., fractions, Roman months, subscript numbers) and standardized gap markers using regex.
- Postprocessing removed repetitions using a base Qwen2.5-72B prompt.
- Validation set was perturbed by removing samples with large train/val length mismatches.
- Expanded initial training set 4x by translating Turkish->English for shared non-English AKTs, retrieving transliterations/translations from external PDFs, deduplicating published_texts.csv, splitting into ~30-word strings, and pseudolabeling the remainder.
- Layout detection for PDFs (regular text vs. columnar).
- OCR via GLM-OCR.
- VLM extraction for columnar layouts.
- Sliding-window context feeding (previous, current, next pages) to GPT-OSS-120B.
- Few-shot prompting for alignment.
- Multilingual normalization to English.
- Manual curation of ~65k high-quality pairs.
- Strict epoch limiting to prevent model collapse.

## Models
- ByT5-Small
- ByT5-Base
- ByT5-Large
- ByT5-XL
- GLM-OCR
- Qwen (2.5/3/3.5 variants)
- LLaMA
- Mistral (14B)
- T5 variants (T5-Gemma, mT5, NLLB)
- GPT-OSS-120B

## Frameworks used
- Seq2SeqTrainer
- ctranslate2
- DeepSpeed
- vLLM

## Loss functions
- eval_loss
- custom loss weighted by data quality
- Cross-entropy loss
- Symmetric Kullback-Leibler (KL) divergence

## CV strategies
- 4-fold cross-validation for data cleaning; 9:1 train/valid split for some models; checkpoint selection based on validation eval_loss.
- 5-fold cross-validation based on Public Leaderboard (LB) scores.
- Perturbed version of train.csv removing all bad samples where the training and validation lengths are way off.
- Used the first 500 pairs from the training set as a validation set due to inconsistent correlation with the LB; CV scores consistently ranged from 42 to 43.5 across 40+ models.

## Ensembling
- The final submission was an ensemble of 11 quantized ByT5-XL models, each generating 10 candidates via beam search and temperature sampling, which were merged and scored using a weighted MBR selector combining chrF++, BLEU, Jaccard, and length rewards.
- Final predictions were aggregated using Minimum Bayes Risk (MBR) ensembling across 15 models, with beam search restricted to size 4 and sampling explicitly avoided to prevent score degradation.
- The solution ensembles a ByT5-Large and a ByT5-XL model, using a fine-tuned Qwen3-8B pairwise Reward Model to select the better translation per sentence, with beam search (8 beams) used during inference.
- No ensembling was used; a fuzzy-matching post-processor for proper names was attempted but discarded due to lack of improvement.
- The solution relies on post-training two Qwen2.5 variants and a ByT5-Base model, with final outputs cleaned via regex normalization and LLM-assisted repetition removal rather than explicit model ensembling.
- Standalone ByT5-XL CasualLM fine-tuning without ensembling.
- Two-stage supervised fine-tuning on ByT5-XL without explicit ensembling.

## Insights
- Careful data preprocessing and minimal post-processing are crucial to avoid public LB overfitting.
- Larger models significantly improve scores but require sufficiently clean training data to prevent overfitting.
- eval_loss is a more reliable metric for checkpoint selection than generation metrics like BLEU or chrf, which can mask over-memorization of validation formats.
- Quantization and pre-compilation are essential for fitting large model ensembles within strict inference time limits.
- Dynamic context sampling effectively mitigates sentence misalignment and adapts the model to unpredictable test set sentence lengths.
- Incorporating large volumes of initially deemed noisy external data significantly boosts performance.
- Minimum Bayes Risk (MBR) ensembling is superior to weight averaging for this task.
- Randomly sampling non-contiguous context sentences forces the model to learn global topic coherence rather than rote local ordering.
- Synthetic data generation via LLMs is highly effective for teaching foundational language rules to seq2seq models before fine-tuning on scarce real data.
- Careful checkpoint selection during continued pre-training is critical, as training too long causes regression in rare names and vocabulary due to frequency bias.
- Using embedding-based retrieval for dynamic few-shot examples significantly improves translation quality over random sampling.
- Multimodal LLMs with explicit visual layout instructions and strict anti-hallucination prompts are necessary for accurate extraction of academic PDFs containing specialized diacritics.
- Treating data construction as the primary engineering challenge outweighs architectural modifications for severely data-constrained NMT tasks.
- A widening public/private leaderboard gap signals the need for more diverse external data rather than hyperparameter tuning.
- Using group_by_length batching introduces non-uniform gradient magnitudes that can be stabilized by enabling the first-moment estimate (β₁=0.9) in Adafactor.
- Post-hoc name normalization is unnecessary if the model is trained on sufficient, correctly aligned source data.
- Leveraging an autonomous agent for iterative experimentation drastically reduced development time under severe constraints.
- Generating high-quality synthetic data via Gemini OCR and published texts outperformed attempting to clean the noisy competition training data.
- Post-processing model outputs with a dedicated deduplication prompt significantly improved translation quality without retraining.
- Model size is a critical driver of performance, with byt5-xl outperforming smaller byt5 variants.
- Formulating the task as a CasualLM yields better results than treating it as a Seq2Seq problem.
- Fine-tuning other LLMs like Mistral and Qwen is less effective than fine-tuning byt5 for this task.
- Byte-level tokenization is essential for scripts that do not tokenize cleanly with subword tokenizers.
- A two-stage fine-tuning approach combining noisy broad pretraining with clean targeted fine-tuning effectively builds a translation prior without overfitting.
- Layout-specific extraction pipelines are necessary when processing PDFs with mixed formatting.
- LLMs used for data extraction should be selected via practical throughput and vibe-testing rather than benchmarks when ground truth is unavailable.

## Critical findings
- Relying on eval_bleu or chrf for checkpoint selection is untrustworthy because continuous improvements often indicate the model is memorizing validation set formats rather than generalizing.
- Post-processing techniques like cleaning repeated n-grams can increase overfitting risk and should be avoided unless absolutely necessary.
- Using more quantized models is more effective for inference speed and score than using fewer full-precision models.
- Data initially considered too noisy actually improved scores when included in large volumes.
- Weight fusion/averaging degraded performance compared to single models, making MBR ensembling necessary.
- Training with very long context windows at the document level performed poorly for a sentence-level evaluation task.
- Directly applying off-the-shelf LLMs (Qwen, LLaMA) without specialized cleaning or tricks yielded only average results.
- Training beyond 14k steps during CPT led to regression, where previously correct rare names and vocabulary suddenly regressed toward frequent neighbors or hallucinated new names.
- Silently skipping damaged tablets was a major data loss risk; explicit include ALL lines instructions were required to recover ~30% of pairs.
- Concatenating examples from different senses of polysemous lemmas forces the model to learn sense disambiguation from context, effectively repurposing synthetic data.
- Synthesizing pseudo-translations from morphological metadata alone produced too much noise and actively degraded model performance.
- Adding word-level dictionary translation pairs as training samples provided zero benefit despite intuitive appeal.
- Running the LLM extraction pipeline three separate times on multilingual sources yields slightly different English renditions that serve as effective implicit data augmentation.
- Sources already covered by the official alignment pipeline should be excluded from multi-version extraction to avoid generating near-duplicates that add no new information.
- The competition's provided training data was deemed too confusing and was explicitly abandoned in favor of synthetic data.
- Larger ByT5 models (>= Large) failed to improve results, contrary to typical scaling expectations.
- Relying solely on the perturbed validation set was risky due to inherent data leakage, necessitating cautious model selection.
- Manual sentence alignment consumed significant time but delivered low impact compared to data scaling.
- Pseudolabeling deduplicated text split into ~30-word strings successfully added ~18k high-quality samples to the training set.
- Training for more than one epoch in Stage 2 consistently caused model collapse and LB score drops.
- Rerunning the VLM extraction pipeline with higher throughput via vLLM actually yielded worse LB scores than the initial lower-throughput version.
- Automated sentence alignment methods failed to consistently align sentences, making manual or heuristic approaches unreliable.

## What did not work
- Using decoder-only models like Qwen or translate-gemma without sufficient proof of viability.
- Continual Pre-Training (CPT) due to poor quality and difficulty cleaning the pre-training corpus.
- Multilingual training with prefixes to distinguish target languages.
- Using context in the input.
- Test Time Augmentation (TTA) using byt5 outputs for further fine-tuning.
- Pre-processing steps like standardizing Sumerian spelling, determinative formats, or stripping supplementary text in translations.
- Post-processing steps like standardizing names/places according to onomasticon.csv.
- Organizing target sentence with explicit CONTEXT BEFORE/AFTER blocks during training/inference.
- Data augmentation on names and dictionaries (random capitalization/norm-to-lexeme mapping) for single models.
- Weight fusion/averaging.
- Setting very long maxlen and training at the document level.
- Using Large Language Models (Qwen, LLaMA, etc.) without special tricks.
- Bidirectional training (adding akk->eng).
- Any ByT5 model >= Large.
- MBR.
- Using competition training data.
- Translating Akkadian -> Arabic -> English.
- Training a judge model to grade the quality of the translations.
- Fine-tuning other LLMs like Mistral and Qwen performed worse than byt5 fine-tuning.
- Other T5 variants performed much worse than the byt5 family.
- Dictionary and RAG-based reconstruction for pseudo-labeling samples without ground truth.
- Name and commodity augmentation (both online and offline variants).
- Automated sentence alignment methods.
- Training for more than 1 epoch in Stage 2.
- LLM post-processing.
- RL-based methods (DPO, minimum risk training, PPO, and GRPO adapted for Seq2Seq).
- Cross-validation correlation with the LB.

## Notable individual insights
- rank 1 ([DPC 1st] Data Quality Dictates Everything): eval_loss is a more reliable metric for checkpoint selection than generation metrics like BLEU or chrf, which can mask over-memorization of validation formats.
- rank 6 ([6th] DPC solution - 15 models ensemble on diverse data sources): Data initially considered too noisy actually improved scores when included in large volumes, and MBR ensembling is superior to weight averaging for this task.
- rank 3 ([3rd] Synthetic Data to Teach OA Fundamentals): Training beyond 14k steps during CPT led to regression, where previously correct rare names and vocabulary suddenly regressed toward frequent neighbors or hallucinated new names.
- rank 2 ([2nd Place] Data-Centric Akkadian NMT): Treating data construction as the primary engineering challenge outweighs architectural modifications for severely data-constrained NMT tasks.
- rank 24 ([24th] Post-training Qwen2.5 32B and 72B with Gemini OCR/published texts pairs): Leveraging an autonomous agent for iterative experimentation drastically reduced development time under severe constraints.
- rank 7 (Short 7th place note): Formulating the task as a CasualLM yields better results than treating it as a Seq2Seq problem.
- rank 8 (8th Place Solution: 2-Stage Fine-tuning + High Quality Data Extraction): Training for more than one epoch in Stage 2 consistently caused model collapse and LB score drops.

## Solutions indexed
- #1 [[solutions/rank_01/solution|[DPC 1st] Data Quality Dictates Everything]]
- #2 [[solutions/rank_02/solution|[2nd Place] Data-Centric Akkadian NMT]]
- #3 [[solutions/rank_03/solution|[3rd] Synthetic Data to Teach OA Fundamentals]]
- #6 [[solutions/rank_06/solution|[6th] DPC solution - 15 models ensemble on diverse data sources]]
- #7 [[solutions/rank_07/solution|Short 7th place note]]
- #8 [[solutions/rank_08/solution|8th Place Solution: 2-Stage Fine-tuning + High Quality Data Extraction]]
- #24 [[solutions/rank_24/solution|[24th] Post-training Qwen2.5 32B and 72B with Gemini OCR/published texts pairs]]

## GitHub links
- [bogoconic1/Qgentic-AI](https://github.com/bogoconic1/Qgentic-AI) _(library)_ — from [[solutions/rank_24/solution|[24th] Post-training Qwen2.5 32B and 72B with Gemini OCR/published texts pairs]]
- [Hrithik2212/Kaggle-DPC-NMT-Akkadin2English-Rank8](https://github.com/Hrithik2212/Kaggle-DPC-NMT-Akkadin2English-Rank8) _(solution)_ — from [[solutions/rank_08/solution|8th Place Solution: 2-Stage Fine-tuning + High Quality Data Extraction]]

## Papers cited
- [Kouwenberg 2017 GOA book](https://www.kaggle.com/datasets/deeppast/old-assyrian-grammars-and-other-resources?select=Kouwenberg_2017_GOA_HdO118.pdf)
