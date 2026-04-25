# google-gemma-3n-hackathon: top public notebooks

The community's top-voted notebooks primarily focus on practical tutorials for deploying, fine-tuning, and testing Google's Gemma-3n multimodal large language model on Kaggle's free infrastructure. They cover end-to-end workflows including 4-bit quantization, LoRA adaptation for specialized domains (medical, audio, healthcare), and multimodal inference for text, vision, and audio tasks. The collective work emphasizes memory-efficient training techniques, checkpoint compatibility workarounds, and prompt engineering to maximize performance on constrained hardware.

## Common purposes
- tutorial
- utility
- training

## Competition flows
- Loads a 4-bit quantized Gemma 3N 4B model, prepares conversational data from a Hugging Face dataset, fine-tunes it using LoRA and Unsloth's SFTTrainer, and demonstrates multimodal inference and model export.
- Downloads the Gemma-3n model via KaggleHub, loads it with Hugging Face transformers, and runs inference for both text generation and image-text understanding tasks.
- Downloads the Gemma 3n 2B model via Kaggle Hub, configures it for text and multimodal inference using Hugging Face Transformers, runs it on curated prompts, and evaluates its performance across multiple domains.
- Loads a medical instruction dataset from Hugging Face, formats it with chat templates, dynamically configures and trains a LoRA adapter on Gemma-3-4B using Unsloth and TRL, then exports the fine-tuned model in LoRA, merged, and GGUF formats.
- Loads OCR test images → processes them through Gemma-3n-E2B vision model → generates text predictions while demonstrating and working around FP16 overflow and spatial transposition bugs.
- Downloads the Gemma 3n model via Kaggle Hub, loads it with Hugging Face Transformers, and runs text and multimodal generation tasks on a CPU environment.
- Loads the Gemma 3n multimodal model via Kaggle Hub, processes plant images with structured prompts to extract species and health data, calculates biodiversity metrics, and exports offline-ready conservation reports.
- Downloads the Gemma-3n multimodal model via KaggleHub, processes user-provided text prompts or image URLs through the Hugging Face processor, and generates humorous, intentionally inaccurate text outputs via sampling-based inference.
- The notebook configures the Kaggle environment, installs dependencies, downloads and loads the Gemma 3n model with 4-bit quantization, and provides demo functions for text and multimodal inference tailored to healthcare queries.
- Loads a German speech dataset, formats audio-text pairs into a chat template, fine-tunes a 4-bit quantized Gemma-3N-4B model using LoRA for one epoch, and evaluates the reduction in Word Error Rate before saving the adapters.

## Data reading
- Loads the mlabonne/FineTome-100k dataset from Hugging Face using datasets.load_dataset, selecting the first 3000 training rows.
- Model weights are downloaded locally using `kagglehub.model_download`.
- An example image is loaded directly from a public URL.
- Loads model weights and tokenizer via kagglehub.model_download("google/gemma-3n/transformers/gemma-3n-e2b-it").
- Loads the FreedomIntelligence/medical-o1-reasoning-SFT dataset from Hugging Face using the datasets library
- Inspects the first few rows with pandas
- Converts the dataset to a standardized list format using unsloth.chat_templates.standardize_data_formats()
- Images are loaded from a Kaggle dataset input path using PIL (`Image.open`).
- Uses `kagglehub.model_download` to fetch model weights and embeds a base64-encoded image directly via `IPython.display.Image`.
- Reads a local image file path and passes it alongside a text prompt to the processor.
- Uses a hardcoded Python dictionary to simulate a local species and disease database.
- Loads model weights and tokenizer from KaggleHub using kagglehub.model_download
- Reads image URLs directly from string variables for inference
- Loads the `unsloth/Emilia-DE-B000000` dataset via Hugging Face `datasets`.
- Selects 3,000 training samples and reserves one sample for testing.
- Casts the audio column to a 16kHz sampling rate.

## Data processing
- Applies the Gemma-3 chat template to format conversations, removes the leading <bos> token, and masks the loss on user instructions during training using train_on_responses_only so gradients only update on assistant outputs.
- Tokenization and chat template formatting are handled by `AutoTokenizer` and `AutoProcessor`.
- Inputs are moved to the appropriate device and dtype automatically via `device_map` and `torch_dtype`.
- Generation parameters (temperature, max tokens, sampling) are explicitly configured.
- Removes the Complex_CoT column to reduce noise for Gemma's instruction-tuning format
- Merges custom Q&A examples with the original dataset
- Formats text into chat-style conversations using tokenizer.apply_chat_template()
- Strips <bos> tokens to prevent duplication during training
- Masks user prompts during training via train_on_responses_only() so gradients only flow through assistant responses
- Calculates dataset semantic similarity using SentenceTransformer to dynamically scale training steps and learning rate
- Resizes input images to 384x384.
- For the standard `transformers` pipeline, manually transposes the `pixel_values` tensor axes (`permute(0,1,3,2)`) and forces FP32 precision on CPU to avoid FP16 activation overflows.
- Uses AutoProcessor.apply_chat_template for multimodal tokenization and chat formatting.
- Applies simple keyword-based string parsing on model outputs to extract structured fields like species and health status.
- None explicitly shown; the Hugging Face AutoProcessor handles internal image loading and tokenization
- Formats raw audio-text pairs into a system/user/assistant chat structure.
- Applies a chat template to tokenize text and process audio inputs.
- Masks padding, audio, and special tokens in the label tensor to exclude them from loss computation.

## Models
- Gemma-3N-E4B-it
- Gemma-3n (gemma-3n-e2b-it)
- AutoModelForCausalLM
- AutoModelForImageTextToText
- Gemma 3n 2B it
- Gemma-3-4B (base)
- LoRA adapters
- Gemma-3n-E2B
- MobileNetV5
- Gemma 3n

## Frameworks used
- unsloth
- transformers
- trl
- datasets
- torch
- bitsandbytes
- accelerate
- xformers
- peft
- kagglehub
- sentence_transformers
- PIL
- opencv
- numpy
- pandas
- IPython
- gradio
- timm

## Loss functions
- Cross-entropy loss masked to only compute gradients on assistant responses (user inputs masked with -100)

## CV strategies
- Optional 10% holdout split for evaluation; otherwise relies on training loss with a custom early stopping callback (patience=10, min_delta=0.001)

## Insights
- LoRA fine-tuning with r=8 and lora_alpha=8 efficiently adapts the model without full parameter updates.
- Masking the loss to only train on model responses significantly improves fine-tuning accuracy.
- 4-bit quantization via Unsloth drastically reduces VRAM requirements, enabling training on free Kaggle T4 GPUs.
- Gemma-3N natively supports text, vision, and audio modalities in a single unified architecture.
- Recommended inference hyperparameters for Gemma-3 are temperature=1.0, top_p=0.95, and top_k=64.
- Leveraging `kagglehub` allows seamless local model downloads on Kaggle's GPU infrastructure without external dependencies.
- The `transformers` library provides unified interfaces for both causal language modeling and multimodal image-text-to-text tasks.
- Properly applying chat templates and managing generation configs is essential for consistent multimodal model outputs.
- Prompting the model to answer concisely significantly improves output quality and relevance.
- The model handles arithmetic, algebra, and geometry tasks accurately when explicitly instructed to return only the result.
- Multimodal inference requires using AutoProcessor with a structured chat template to correctly format image and text inputs.
- Generation parameters like temperature and max_new_tokens are critical for controlling response length and stochasticity.
- LoRA enables efficient fine-tuning by updating only ~1-10% of parameters while keeping the base model frozen.
- Chat templates are critical for teaching LLMs conversation structure, role boundaries, and multi-turn context.
- Response-only training improves efficiency and quality by preventing the model from learning to predict user inputs.
- Dynamic hyperparameter scaling based on dataset size and GPU memory helps optimize training on consumer hardware.
- Semantic similarity of the target dataset can guide epoch count and learning rate adjustments to avoid overfitting or undertraining.
- Gemma-3n-E2B's raw activations overflow in FP16 on standard hardware like Kaggle T4 GPUs.
- The official checkpoints contain a transposed width/height axis bug in the MobileNetV5 vision tower that breaks OCR.
- Unsloth's optimized loader automatically resolves both precision and tensor shape bugs.
- Standard `transformers` requires manual pixel value transposition and FP32/CPU execution as a reliable workaround.
- Gemma 3n requires `transformers >= 4.53.0` and `timm >= 1.0.16` to avoid runtime errors.
- The HuggingFace `pipeline` API is incompatible with Kaggle's environment due to login requirements, necessitating direct `AutoProcessor` and `AutoModelForImageTextToText` usage.
- Running the model on a free CPU tier is functional but extremely slow, taking over an hour for generation.
- The model natively supports multimodal chat templates for combining text and image inputs.
- Open-weight multimodal models can be effectively deployed offline for field conservation tasks without relying on external APIs.
- Structured prompting combined with lightweight keyword parsing provides a practical pipeline for extracting actionable botanical data.
- Local biodiversity metrics like Simpson's Diversity Index can be computed directly from accumulated AI observations to monitor ecosystem health.
- Offline-first architecture is essential for deploying AI tools in remote areas with limited or no internet connectivity.
- High temperature (1.0) and sampling parameters (top_k=50, top_p=0.95) effectively push the model to generate creative, non-factual text.
- The AutoProcessor seamlessly handles multimodal tokenization for image-text inputs without manual preprocessing.
- KaggleHub simplifies model downloading and weight management directly within the notebook environment.
- 4-bit quantization with NF4 and double quantization enables running large multimodal models on limited GPU memory.
- Robust mock fallbacks allow notebook development and demonstration without requiring license acceptance or heavy downloads.
- Prompt engineering can effectively steer a general multimodal LLM toward specialized healthcare domains like nutrition, fitness, and dermatology.
- Fine-tuning a multimodal LLM with LoRA on just 3,000 samples can significantly reduce ASR error rates.
- Using `do_sample=False` is necessary for deterministic speech transcription outputs.
- Targeting both language and audio-specific projection layers in LoRA is crucial for multimodal adaptation.

## Critical findings
- FP16 inference on CUDA fails completely due to activation overflow.
- The official Hugging Face checkpoints have incorrectly transposed spatial dimensions in the vision tower, causing poor OCR performance.
- Using `timm` version 1.0.15 or older `transformers` triggers a `RuntimeError: Unknown model (mobilenetv5_300m_enc)`.
- The author notes that while model shards load completely on CPU, the generation process remains unacceptably slow without GPU acceleration.

## What did not work
- Standard `transformers` pipeline with FP16 on CUDA fails entirely due to activation overflow.
- Running the official checkpoint without manual tensor transposition produces incorrect OCR results.
- The HuggingFace `pipeline` API failed due to required HuggingFace account login.
- Older package versions (`timm` 1.0.15 and outdated `transformers`) caused model loading errors.
- CPU-only execution resulted in draft session timeouts and excessive runtime (>1 hour).

## Notable individual insights
- votes 774 (Gemma 3N 4B Multimodal finetuning + inference): Masking the loss to only train on model responses significantly improves fine-tuning accuracy.
- votes 61 (Gemma 3N Vision test): The official checkpoints contain a transposed width/height axis bug in the MobileNetV5 vision tower that breaks OCR.
- votes 61 (⚡ Customize LLM: Step-by-Step LoRA Training): Semantic similarity of the target dataset can guide epoch count and learning rate adjustments to avoid overfitting or undertraining.
- votes 59 (Gemma 3n: Me ConfɅsed and ⱯshⱯmed): The HuggingFace `pipeline` API is incompatible with Kaggle's environment due to login requirements, necessitating direct `AutoProcessor` and `AutoModelForImageTextToText` usage.
- votes 57 (🌱EcoVision AI: Democratizing Environmental): Offline-first architecture is essential for deploying AI tools in remote areas with limited or no internet connectivity.
- votes 42 (Gemma 3N 4B Audio finetuning): Fine-tuning a multimodal LLM with LoRA on just 3,000 samples can significantly reduce ASR error rates.

## Notebooks indexed
- #774 votes [[notebooks/votes_01_danielhanchen-gemma-3n-4b-multimodal-finetuning-inference/notebook|Gemma 3N 4B Multimodal finetuning + inference]] ([kaggle](https://www.kaggle.com/code/danielhanchen/gemma-3n-4b-multimodal-finetuning-inference))
- #337 votes [[notebooks/votes_02_paultimothymooney-how-to-use-gemma-3n-on-kaggle/notebook|How to use Gemma-3n on Kaggle]] ([kaggle](https://www.kaggle.com/code/paultimothymooney/how-to-use-gemma-3n-on-kaggle))
- #62 votes [[notebooks/votes_03_gpreda-testing-gemma-3n-like-a-pro/notebook|Testing Gemma 3n like a pro ]] ([kaggle](https://www.kaggle.com/code/gpreda/testing-gemma-3n-like-a-pro))
- #61 votes [[notebooks/votes_04_chaitanya99-customize-llm-step-by-step-lora-training/notebook|⚗️ Customize LLM: Step-by-Step LoRA Training]] ([kaggle](https://www.kaggle.com/code/chaitanya99/customize-llm-step-by-step-lora-training))
- #61 votes [[notebooks/votes_05_spacedoge-gemma-3n-vision-test/notebook|Gemma 3N Vision test]] ([kaggle](https://www.kaggle.com/code/spacedoge/gemma-3n-vision-test))
- #59 votes [[notebooks/votes_06_mpwolke-gemma-3n-me-conf-sed-and-sh-med/notebook|Gemma 3n: Me ConfɅsed and ⱯshⱯmed]] ([kaggle](https://www.kaggle.com/code/mpwolke/gemma-3n-me-conf-sed-and-sh-med))
- #57 votes [[notebooks/votes_07_abhinavdogra002-ecovision-ai-democratizing-environmental/notebook|🌱EcoVision AI: Democratizing Environmental]] ([kaggle](https://www.kaggle.com/code/abhinavdogra002/ecovision-ai-democratizing-environmental))
- #51 votes [[notebooks/votes_08_codingloading-can-gemma-3n-understand-memes/notebook|😜Can Gemma-3n Understand Memes?]] ([kaggle](https://www.kaggle.com/code/codingloading/can-gemma-3n-understand-memes))
- #45 votes [[notebooks/votes_09_adilshamim8-gemma-3n-impact-challenge/notebook|Gemma 3n Impact Challenge]] ([kaggle](https://www.kaggle.com/code/adilshamim8/gemma-3n-impact-challenge))
- #42 votes [[notebooks/votes_10_danielhanchen-gemma-3n-4b-audio-finetuning/notebook|Gemma 3N 4B Audio finetuning]] ([kaggle](https://www.kaggle.com/code/danielhanchen/gemma-3n-4b-audio-finetuning))
