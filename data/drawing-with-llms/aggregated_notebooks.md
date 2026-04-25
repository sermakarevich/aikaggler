# drawing-with-llms: top public notebooks

The top-voted notebooks for the Drawing with LLMs competition primarily focus on inference pipelines and educational tutorials rather than traditional EDA or model training. The community explores two dominant architectural paradigms: direct LLM-based SVG generation (using quantized models like Gemma 2 and Qwen2.5 with strict constraint enforcement) and diffusion-based workflows (leveraging Stable Diffusion combined with custom raster-to-vector conversion and contour extraction). Additionally, several notebooks highlight metric exploitation techniques, Kaggle Package structuring, and iterative post-processing to maximize leaderboard scores within strict file size and time constraints.

## Common purposes
- inference
- tutorial
- baseline

## Competition flows
- Loads the Gemma 2 9B IT model with 4-bit quantization, generates SVG code from text prompts via a constrained template, enforces submission constraints through XML parsing and regex validation, and outputs valid SVGs for Kaggle submission.
- The notebook explains how to structure a Kaggle Package using nbdev, provides a trivial baseline Model class that returns a static SVG string, demonstrates local validation with kaggle_evaluation, and outlines how to submit the package for automated inference on the competition's test set.
- Generates bitmap images via Stable Diffusion, converts them to SVGs using OpenCV contour extraction and color quantization, and scores them using a custom metric combining PaliGemma-2 fidelity and CLIP aesthetic predictors.
- Loads text prompts from a CSV, generates candidate bitmap images with Stable Diffusion, converts them to size-constrained SVGs, evaluates them against a local metric combining VQA and aesthetic scores, and outputs the highest-scoring SVG per prompt.
- Loads text descriptions and QA data, generates intermediate bitmaps via Stable Diffusion, converts them to SVGs using contour extraction and simplification, and evaluates them using a custom multi-component scoring metric.
- Loads text descriptions from a CSV, generates compliant SVG code using a quantized Qwen2.5-32b model via vLLM, validates and sanitizes the output against domain constraints, and saves the results as a JSONL file with PNG visualizations.
- Loads text descriptions from a CSV, generates candidate images via Stable Diffusion, converts them to SVGs using a contour-based algorithm, scores them locally with a VQA + aesthetic metric, and returns the best SVG for submission.
- Loads text descriptions from CSV/Parquet, generates bitmaps via Stable Diffusion, converts them to SVGs using contour extraction and adaptive simplification, adds OCR decoys, evaluates with a custom metric, and outputs the highest-scoring SVG per description.
- Reads text descriptions from the competition dataset, generates SVG submissions by rendering font glyphs as vector paths, optimizes background color and text length via binary search, and outputs formatted SVG strings for submission.
- Loads text descriptions and multiple-choice QA data, generates candidate bitmaps via Stable Diffusion, converts them to SVGs using a custom contour extraction and color quantization pipeline, scores them against a custom OCR/aesthetic metric, and selects the highest-scoring SVG for each prompt.

## Data reading
- Uses Polars to read train.csv for local testing; relies on Kaggle's kagglehub and kaggle_evaluation packages for model weights and evaluation metrics.
- Loads train.csv using kagglehub.competition_download and polars.read_csv for initial exploration.
- Not explicitly shown; the notebook focuses on generation and metric evaluation rather than competition dataset ingestion.
- Reads prompts from a CSV file using pd.read_csv('/kaggle/input/drawing-with-llms/train.csv'), extracting the 'description' column as text inputs.
- Loads train.csv and questions.parquet via kagglehub.competition_download. Parses JSON fields in the parquet file into lists/dicts and merges them into a single pandas DataFrame.
- pd.read_csv() after downloading via kagglehub.competition_download()
- Loads text descriptions from a CSV, generates candidate images via Stable Diffusion, converts them to SVGs using a contour-based algorithm, scores them locally with a VQA + aesthetic metric, and returns the best SVG for submission.
- Downloads competition data via kagglehub, Reads train.csv and questions.parquet, Parses nested JSON fields into lists/dicts, Merges parsed question data with train_df on id
- Loads text descriptions from a pandas DataFrame (via kaggle_evaluation.test), Downloads the DejaVuSans-Bold font and SigLIP model weights using kagglehub, Initializes the SVG validator from a Kaggle package
- Downloads competition data via kagglehub.competition_download, reads train.csv and questions.parquet, merges them on id, and parses JSON fields (question, choices, answer) into a unified multiple_choice_qa dictionary per row.

## Data processing
- Enforces SVG constraints by parsing XML to remove disallowed elements/attributes, validates path d attributes with regex, ensures valid SVG structure, and handles generation timeouts or parse errors by falling back to a default SVG.
- Resizes generated bitmaps to target dimensions, converts SVGs to PNGs via CairoSVG for metric evaluation, applies k-means color quantization, contour approximation, and background color extraction.
- Resizes generated bitmaps, performs color quantization via k-means, extracts contours using OpenCV, simplifies polygon coordinates based on area/importance, compresses hex colors, and enforces a maximum byte size limit during SVG construction.
- Resizes images to 384x384 for metric evaluation.
- Converts PIL images to numpy arrays.
- Applies k-means color quantization and contour detection.
- Simplifies polygon coordinates via rounding and point reduction.
- Extracts vector glyph paths from text using freetype.
- Wraps text into lines and calculates layout metrics.
- Performs a binary search to find the maximum text length that fits within a 10,000-byte SVG size limit.
- Iterates through 24 named background colors to find the one that maximizes CLIP similarity.
- Resizes evaluation images to (384, 384), applies random crop/resize and JPEG compression (quality=90) for OCR robustness, and uses half-precision (torch.float16) with a custom DDIMScheduler for faster inference.

## Features engineering
- Hierarchical bitmap features extracted via k-means color quantization, contour area/center calculation, distance-from-center normalization, polygon approximation, and importance scoring based on area, centrality, and complexity.
- Hierarchical contour extraction, k-means color quantization, and importance-weighted polygon simplification for SVG generation.

## Models
- Gemma 2 9B IT
- Qwen2.5-32b-instruct-awq
- Stable Diffusion v2
- PaliGemma-2-10b-mix-448
- CLIP ViT-L-14
- Custom MLP aesthetic predictor
- StableDiffusionPipeline
- google/siglip-so400m-patch14-384

## Frameworks used
- torch
- transformers
- kagglehub
- lxml
- cairosvg
- polars
- kaggle_evaluation
- diffusers
- CLIP
- OpenCV
- numpy
- pandas
- PIL
- matplotlib
- vllm
- freetype
- concurrent.futures
- svg-image-fidelity
- statistics

## Ensembling
- Generates 3 candidate bitmaps per description, converts each to SVG, evaluates all against the custom metric, and selects the highest-scoring output.
- Iteratively generates multiple candidate bitmaps per prompt, converts each to SVG, scores them using the product of OCR and aesthetic metrics, and selects the single best output as post-processing.

## Insights
- 4-bit quantization enables running large LLMs on Kaggle GPUs without memory issues.
- Structured prompting with explicit constraints and examples significantly improves LLM output compliance.
- Generative models require robust post-processing, as programmatic validation and constraint enforcement are necessary to guarantee submission validity.
- Kaggle Packages allow notebooks to export reusable Python packages via nbdev tags, decoupling training logic from inference code.
- The competition requires a specific class Model with a predict function that conforms to a strict API for automated scoring.
- External datasources and dependencies must be managed via kagglehub and the Dependency Manager to ensure reproducibility and offline execution during competition scoring.
- Combining diffusion models with deterministic vectorization pipelines allows for controllable SVG generation.
- The competition metric relies on a harmonic mean of text-fidelity and aesthetic quality, requiring balanced optimization of both components.
- Half-precision inference and disabling safety checkers significantly speed up Stable Diffusion generation without sacrificing output quality.
- Hierarchical contour extraction and importance-based SVG layering preserve visual fidelity while managing file size constraints.
- Prompt engineering with specific prefixes, suffixes, and negative prompts significantly improves the alignment between generated images and the competition metric.
- Hierarchical contour extraction combined with adaptive simplification effectively balances SVG detail against strict file size limits.
- Local metric evaluation closely mirrors the official leaderboard, enabling reliable pre-submission validation.
- Prompt engineering with specific prefixes, suffixes, and negative prompts effectively controls Stable Diffusion to produce flat, minimalistic images ideal for SVG conversion.
- Contour-based SVG generation with importance-weighted simplification preserves visual fidelity while managing file size constraints.
- The custom metric successfully decomposes image quality into VQA, OCR, and aesthetic components, providing a granular evaluation of generative outputs.
- Quantized models (AWQ) can be efficiently deployed for code generation tasks using vLLM with tensor parallelism.
- Strict constraint enforcement via XML parsing and regex validation is necessary to ensure generated SVG code is well-formed and compliant.
- Generating multiple outputs and selecting the first valid one improves reliability without requiring complex post-processing.
- Iterative generation with multiple attempts per prompt reliably improves final metric scores within strict time constraints.
- A weighted harmonic mean (beta=2.0) of VQA fidelity and CLIP aesthetic scores closely approximates the competition's official metric.
- Disabling the Stable Diffusion safety checker and using half-precision significantly reduces inference latency.
- Contour simplification and color quantization are necessary to keep generated SVGs within file size limits.
- Pre-defined default SVGs can serve as effective fallback candidates when diffusion generation fails or scores poorly.
- Converting SD-generated bitmaps to SVGs with contour-based extraction preserves metric-relevant structure better than raw raster outputs.
- Injecting a strategically placed OCR decoy reliably boosts OCR scores without harming aesthetic or VQA metrics.
- Adaptive polygon simplification and color quantization are critical for staying within SVG byte limits while maintaining visual fidelity.
- The evaluation metric's reliance on a CLIP model trained on OCR text-image pairs allows direct text rendering in SVGs to artificially inflate similarity scores.
- Rule-based SVG generation can be effectively optimized by searching background colors and clipping text length within strict byte constraints.
- Using freetype to extract vector glyph paths ensures precise, constraint-compliant text rendering that remains visually interpretable.
- Iterative sampling during inference and selecting the best output based on the target metric is an effective way to improve generation quality without retraining.
- Hierarchical contour extraction and color quantization can successfully approximate bitmap details in SVG format while respecting size constraints.
- Prompt engineering with specific positive/negative suffixes significantly influences the simplicity and metric scores of the generated images.

## Critical findings
- The competition's evaluation model was trained on OCR text-image pairs, creating a direct exploit opportunity for text-heavy SVG submissions.
- SVG size constraints (<= 10,000 bytes) require careful binary search clipping of input text to avoid validation failures during scoring.

## Notable individual insights
- votes 316 (Text Rendering: OCR-Exploit [LB=0.305]): The evaluation metric's reliance on a CLIP model trained on OCR text-image pairs allows direct text rendering in SVGs to artificially inflate similarity scores.
- votes 1366 (Drawing with LLMs - Getting Started with Gemma 2): 4-bit quantization enables running large LLMs on Kaggle GPUs without memory issues.
- votes 433 (Ensemble Approach: Stable Diffusion & Default SVG): A weighted harmonic mean (beta=2.0) of VQA fidelity and CLIP aesthetic scores closely approximates the competition's official metric.
- votes 450 (Getting Started Qwen2.5-32b-instruct-awq [infer]): Quantized models (AWQ) can be efficiently deployed for code generation tasks using vLLM with tensor parallelism.
- votes 636 ([Old Metric | LB 0.694]SD Boost via My Default svg): Combining diffusion models with deterministic vectorization pipelines allows for controllable SVG generation.
- votes 1087 (Drawing with LLMs Starter Notebook): Kaggle Packages allow notebooks to export reusable Python packages via nbdev tags, decoupling training logic from inference code.

## Notebooks indexed
- #1366 votes [[notebooks/votes_01_ryanholbrook-drawing-with-llms-getting-started-with-gemma-2/notebook|Drawing with LLMs - Getting Started with Gemma 2]] ([kaggle](https://www.kaggle.com/code/ryanholbrook/drawing-with-llms-getting-started-with-gemma-2))
- #1087 votes [[notebooks/votes_02_dster-drawing-with-llms-starter-notebook/notebook|Drawing with LLMs Starter Notebook]] ([kaggle](https://www.kaggle.com/code/dster/drawing-with-llms-starter-notebook))
- #636 votes [[notebooks/votes_03_taikimori-old-metric-lb-0-694-sd-boost-via-my-default-svg/notebook|[Old Metric | LB 0.694]SD Boost via My Default svg]] ([kaggle](https://www.kaggle.com/code/taikimori/old-metric-lb-0-694-sd-boost-via-my-default-svg))
- #613 votes [[notebooks/votes_04_richolson-stable-diffusion-svg-scoring-metric/notebook|Stable Diffusion -> SVG -> Scoring Metric]] ([kaggle](https://www.kaggle.com/code/richolson/stable-diffusion-svg-scoring-metric))
- #575 votes [[notebooks/votes_05_jiazhuang-new-metric-simple-sd-svg-cv-0-485/notebook|[New Metric] Simple SD -> SVG [CV ~0.485]]] ([kaggle](https://www.kaggle.com/code/jiazhuang/new-metric-simple-sd-svg-cv-0-485))
- #450 votes [[notebooks/votes_06_jiazhuang-getting-started-qwen2-5-32b-instruct-awq-infer/notebook|Getting Started Qwen2.5-32b-instruct-awq [infer]]] ([kaggle](https://www.kaggle.com/code/jiazhuang/getting-started-qwen2-5-32b-instruct-awq-infer))
- #433 votes [[notebooks/votes_07_tononnh-ensemble-approach-stable-diffusion-default-svg/notebook|Ensemble Approach: Stable Diffusion & Default SVG]] ([kaggle](https://www.kaggle.com/code/tononnh/ensemble-approach-stable-diffusion-default-svg))
- #431 votes [[notebooks/votes_08_kaiyoo88-new-metric-lb-0-632-sd-svg-ocr-decoy-vqascore/notebook|[New Metric LB: 0.632] SD->SVG+OCR Decoy+VQAscore]] ([kaggle](https://www.kaggle.com/code/kaiyoo88/new-metric-lb-0-632-sd-svg-ocr-decoy-vqascore))
- #316 votes [[notebooks/votes_09_tatamikenn-text-rendering-ocr-exploit-lb-0-305/notebook|Text Rendering: OCR-Exploit [LB=0.305]]] ([kaggle](https://www.kaggle.com/code/tatamikenn/text-rendering-ocr-exploit-lb-0-305))
- #311 votes [[notebooks/votes_10_jiazhuang-new-metric-simple-sd-svg-iterative-optimize/notebook|[New Metric] Simple SD -> SVG Iterative optimize]] ([kaggle](https://www.kaggle.com/code/jiazhuang/new-metric-simple-sd-svg-iterative-optimize))
