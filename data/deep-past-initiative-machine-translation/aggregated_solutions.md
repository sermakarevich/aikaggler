# deep-past-initiative-machine-translation: cross-solution summary

This competition focused on Old Assyrian transliteration and translation, where winning approaches heavily prioritized meticulous data curation, LLM-assisted extraction, and byte-level models like ByT5 over architectural complexity. Top solutions demonstrated that rigorous text normalization, synthetic data generation, and careful validation metrics consistently outperformed post-processing or standard seq2seq tweaks.

## Key challenges
- Low-resource translation with inconsistent naming conventions across diverse sources
- OCR/typographical complexity of Akkadian diacritics and subscript digits
- Unreliable cross-validation correlation with public/private leaderboards
- Overfitting to validation set styles (eval_bleu/chrf vs eval_loss)
- Sentence-level vs document-level evaluation mismatch
- Residual data leakage in validation splits
- Balancing noisy external data with high-quality scholar translations

## Models
- ByT5-Base
- ByT5-Large
- ByT5-XL
- ByT5-Small
- Qwen2.5-32B
- Qwen2.5-72B
- Qwen3-8B
- Qwen3.5-VL 35B
- Mistral
- Mistral 14B
- NLLB
- mT5
- GPT-OSS-120B
- GLM-OCR
- GLM4.7V-Flash

## CV strategies
- 4-fold CV with iterative data filtering
- 9:1 train/validation split
- 5-fold CV based on Public LB performance
- Perturbed train.csv filtered for length consistency
- Fixed 500-sample holdout from training set
- Checkpoint selection based on validation eval_loss

## Preprocessing
- Discarded official train.csv
- GLM-OCR layout analysis + regex for page/chapter extraction
- Gemini-3.0-pro-preview prompting for sentence-level alignment
- Cross-fold geo_metric < 20 filtering
- Character/transliteration length filtering
- Unicode NFKC normalization, zero-width character removal, dash/hyphen unification, diacritic simplification
- Fraction/decimal conversion to Unicode fractions
- Subscript digit conversion to standard numerals or accent marks
- Determinative/gap marker standardization
- Hamza/variant character deletion
- Roman numeral month conversion and measurement unit expansion
- PDF rendering at high DPI with sliding window/column separation
- VLM-based layout analysis and extraction
- Deduplication by ID or length consistency filtering
- Quality scoring via LLM judge
- Byte-level tokenization
- Multilingual normalization to English
- Manual sentence alignment and splitting into fixed-length chunks

## Augmentations
- Dynamic context sampling (Zipf/Geometric mixture)
- R-Drop (KL divergence regularization)
- PGD (adversarial embedding perturbation)
- Name augmentation (random capitalization, norm-to-lexeme mapping)
- Name swap augmentation
- Multi-version extraction for translation variance
- Document-level chunking
- Source weighting (2× boost for English academic papers)
- Test-time beam search and temperature sampling
- Weighted MBR inference

## Loss functions
- Standard cross-entropy/Seq2Seq loss
- Custom data-quality-weighted loss
- Symmetric KL divergence (R-Drop)

## Ensemble patterns
- Multi-model ensemble combined via weighted MBR inference
- Multi-model ensemble combined via pairwise reward model selection
- Weight averaging/fusion
- Beam search + temperature sampling for candidate generation

## Post-processing
- None
- Repeated n-gram cleaning
- LLM-based repetition removal
- Regex cleanup for gap alignment/empty strings
- Beam search decoding without sampling
- MBR hypothesis selection

## What worked
- Using eval_loss for checkpoint selection
- Quantizing to int8-float32 via ctranslate2
- Training larger models with sufficiently clean data
- Iterative cross-fold error filtering for data curation
- Adding Turkish-to-English data
- Including high-quality publications data and pseudo-labels
- Dynamic context sampling for length adaptability
- R-Drop and PGD for robustness
- MBR ensembling over weight averaging
- LLM-based sentence breaker/fixer pipeline
- External academic PDF mining
- Text normalization
- 768-byte document augmentation
- English prioritization via Prefer-EN dedup
- Multi-language augmentation
- English academic papers 2× boost
- Rendering PDFs at 576 DPI
- Explicit "include ALL lines" instructions for Gemini
- Embedding-based retrieval for few-shot examples
- Causal LM formulation over seq2seq
- Scaling to byT5-xl
- Two-stage SFT (noisy broad prior -> clean targeted)
- ByT5-XL byte-level tokenization
- Custom PDF extraction pipelines
- GPT-OSS-120B in fp4 quantization

## What did not work
- Relying on eval_bleu/chrf for checkpointing
- Public LB post-processing
- Decoder-only models (Qwen, translate-gemma)
- Continual pre-training (CPT)
- Multilingual training with prefixes
- Context usage
- Same-model TTA
- Standardizing Sumerian spelling/determinatives
- Stripping parentheses in translations
- Explicit CONTEXT BEFORE/AFTER blocks during training/inference
- Data augmentation on names/dictionaries in source text
- Weight fusion/averaging
- Very long maxlen and document-level training
- Using LLMs without special tricks
- Bidirectional training for single model
- Translation Generator from morphological metadata
- Word-level dictionary augmentation
- PN/GN post-processing
- ByT5 models >= Large (in some setups)
- Using competition's official training data
- Translating Akkadian -> Arabic -> English
- Training a separate judge model
- Initial manual sentence alignment
- Fine-tuning Mistral/Qwen and other T5 variants
- Dictionary and RAG-based reconstruction for pseudo-labeling
- Name and commodity augmentation
- Automated sentence alignment
- Training Stage 2 for more than 1 epoch
- LLM post-processing
- RL-based methods (DPO, MRM, PPO, GRPO)
- Cross-validation (unreliable correlation with LB)

## Critical findings
- eval_bleu/chrf improvements after 3 epochs indicate overfitting to validation style, not generalization
- Larger models only outperform smaller ones when training data is sufficiently clean and abundant
- Pre-compiling ctranslate2 models saves ~30 minutes, making large ensembles feasible within time limits
- Public LB post-processing increases overfitting risk and causes score shake-ups
- Extra data initially deemed too noisy actually performed better due to large volume
- Sampling within MBR process decreases score; beam search is strictly better
- Test set is evaluated at the sentence level, making document-level training ineffective
- LLMs (14B) performed average without special tricks, contrasting with top participants who succeeded with them
- Rendering PDFs at 576 DPI was critical for correctly recognizing Akkadian diacritics and subscript digits
- Explicit "include ALL lines" instructions prevented Gemini from silently skipping damaged tablets, preserving ~30% of pairs
- Replacing random few-shot examples with embedding-based retrieval improved synthetic translation quality from 61% to 82%
- Polysemous lemmas require concatenating examples from different senses to force context-based disambiguation
- Training beyond 14k steps caused regression onset due to name frequency bias and hallucinated names
- Public/private LB gap indicated that collecting more external data would have improved the private score more than model tuning
- group_by_length=True caused non-uniform gradient magnitudes, resolved by setting β₁=0.9 in Adafactor to add momentum
- Test set comes from diverse sources with inconsistent name conventions, making PN/GN post-processing ineffective and error-prone
- Validation set contained residual data leakage despite length-mismatch filtering, making CV scores less trustworthy than expected
- The official competition training data was confusing and ultimately unhelpful, leading to its complete abandonment
- ByT5's byte-level tokenization is essential for Akkadian script that fails with standard subword tokenizers
- Extending Stage 2 SFT beyond exactly 1 epoch causes model collapse and leaderboard score drops
- CV showed no consistent correlation with LB, making the 500-pair holdout unreliable for model selection
- Qwen3.5-VL 35B outputs yielded worse LB scores than GLM4.7V-Flash despite higher capability, highlighting the need for throughput/practical tuning over raw benchmarks

## Notable individual insights
- rank 1 (Data Quality Dictates Everything): eval_bleu/chrf improvements after 3 epochs indicate overfitting to validation style, not generalization; prioritizing eval_loss prevents this.
- rank 6 (DPC solution - 15 models ensemble on diverse data sources): Test set is evaluated at the sentence level, making document-level training ineffective; explicit context blocks during training hurt performance.
- rank 3 (Synthetic Data to Teach OA Fundamentals): Rendering PDFs at 576 DPI was critical for correctly recognizing Akkadian diacritics and subscript digits; replacing random few-shot examples with embedding-based retrieval improved synthetic quality from 61% to 82%.
- rank 2 (Data-Centric Akkadian NMT): group_by_length=True caused non-uniform gradient magnitudes, resolved by setting β₁=0.9 in Adafactor to add momentum; PN/GN post-processing failed due to test set diversity.
- rank 24 (Post-training Qwen2.5 32B and 72B with Gemini OCR/published texts pairs): Validation set contained residual data leakage despite length-mismatch filtering, making CV scores less trustworthy; official competition training data was ultimately unhelpful and was abandoned.
- rank 7 (Short 7th place note): Framing the task as a causal language modeling problem outperformed both seq2seq setups and fine-tuned LLMs like Mistral and Qwen.
- rank 8 (8th Place Solution: 2-Stage Fine-tuning + High Quality Data Extraction): Extending Stage 2 SFT beyond exactly 1 epoch causes model collapse and leaderboard score drops; Qwen3.5-VL 35B yielded worse LB scores than GLM4.7V-Flash despite higher capability, highlighting the need for practical tuning over raw benchmarks.

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
