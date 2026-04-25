# gemma-language-tuning: top public notebooks

The community's top-voted notebooks focus on parameter-efficient fine-tuning of Gemma 2 models for multilingual and culturally-specific tasks, emphasizing prompt engineering, memory optimization, and domain adaptation. They provide practical tutorials on integrating LoRA/QLoRA with Keras/JAX and PyTorch ecosystems, while addressing hardware constraints and evaluation strategies for low-resource language generation.

## Common purposes
- tutorial
- training

## Competition flows
- Loads a Korean cake-ordering dataset, formats inputs with instruction templates, fine-tunes Gemma2-2B using LoRA and KerasNLP, and demonstrates inference with varying prompts and decoding strategies.
- Loads a small CSV dataset of Bhagavad Gita verses, formats them into prompts, fine-tunes a Gemma 2 2B model with LoRA using Keras 3, and evaluates the output on in-distribution and out-of-distribution queries.
- Loads a small subset of a programming language dataset, formats it into prompts, fine-tunes a Gemma 2-2B model with LoRA using Keras 3/JAX, and demonstrates inference on new queries.
- Loads a paired Early Hangul to modern Korean dataset from Hugging Face, formats it with instruct tokens, fine-tunes a Gemma 2B model using LoRA and KerasNLP/JAX for five epochs, and saves the resulting LoRA weights for inference.
- Loads the AraStories MSA dataset, applies text cleaning and GPT-4-based quality filtering, formats it into chat templates, fine-tunes a 4-bit quantized Gemma 2 2B model using QLoRA and the SFTTrainer, and exports the merged adapter for inference.
- Initializes the Gemma 2B model via Hugging Face, defines a conceptual cultural adaptation configuration, and outlines proposed evaluation metrics for culturally-aware language variants.
- Loads a Kazakh instruction-following dataset, formats it into instruction-response pairs, fine-tunes a Gemma 2B model using LoRA and KerasNLP, and saves the adapted model for downstream Kazakh text generation.
- Loads the Shosi dataset, formats it into instruction prompts, fine-tunes the Gemma 2 2B JPN model using LoRA and Unsloth on a T4 GPU, and evaluates the resulting model against zero-shot, few-shot, and MeCab baselines using accuracy and BLEU scores.
- Loads a classical-to-modern Chinese text dataset, formats it into instruction-response pairs, fine-tunes a Gemma-2 9B model with LoRA on 8 TPUs using Keras 3, and generates translation outputs.

## Data reading
- Uses datasets.load_dataset to fetch the "bebechien/korean_cake_boss" training split, converts it to numpy format, and iterates through rows manually.
- pd.read_csv with nrows=52 from /kaggle/input/bhagwad-gita-dataset/Bhagwad_Gita.csv
- pd.read_csv(f"{Config.dataset_path}/index.csv", nrows=52)
- Loads the `bebechien/HongGildongJeon` dataset from Hugging Face using `datasets.load_dataset` with `split='train'`.
- Extracts the `original` (Early Hangul) and `modern translation` columns and converts them to numpy arrays via `ds.with_format('np', ...)`.
- Loaded via kagglehub from a CSV file ('Formatted-MSA-prompts-stories-for-fine-tuning.csv') containing 'Prompt' and 'Story' columns.
- Loads data from a Hugging Face dataset (saillab/alpaca_kazakh_taco) using pd.read_parquet with a direct hf:// URL path.
- Data is loaded from Kaggle-hosted TSV files using kagglehub.load_dataset with pandas_kwargs to limit rows, containing Japanese_text and Yomigana columns.
- Reads JSONL files using pandas read_json with lines=True and json.loads, then renames columns to classical_chinese, modern, and path.

## Data processing
- Applies a fixed prompt template with <start_of_turn> and <end_of_turn> markers, tokenizes inputs using the Gemma tokenizer, filters out samples exceeding 256 tokens, and caps the training set at 100 examples.
- Custom prompt template formatting using df.apply to map Verse, WordMeaning, and HinMeaning columns into a structured string
- Reduced dataset size to 52 rows to manage GPU memory
- Manual replacement of a single missing value
- Formatting raw CSV columns into a custom prompt template
- Converting to a list of strings for training
- Constructs instruction-tuning prompts using the format `<start_of_turn>user\n{original}<end_of_turn>\n<start_of_turn>model\n{modern translation}<end_of_turn>`.
- Filters out training samples where the tokenized length exceeds the `token_limit` of 256 to prevent memory issues.
- No additional cleaning, normalization, or augmentation is applied.
- Regex-based text cleaning (removing emoticons and non-Arabic characters)
- Diacritics removal via PyArabic and Hamza/Alef standardization
- GPT-4 automated quality scoring for coherence and fluency
- Chat template formatting using apply_chat_template
- 90/10 train-test split
- Selects instruction and text columns, truncates text to 3000 characters, limits the dataset to 30,000 rows, and formats each sample into a fixed Instruction/Response template string.
- The raw dataset is cleaned by replacing periods with Japanese full stops and preserving the particle は instead of its spoken pronunciation わ.
- Inputs are formatted into a fixed Japanese Alpaca-style instruction prompt template, and an EOS_TOKEN is appended to all sequences to prevent infinite generation.
- Formats raw text into instruction-response templates, truncates the dataset to 2000 samples, and sets the preprocessor sequence length to 512.

## Models
- Gemma2-2B
- GemmaCausalLM (gemma2_2b_en)
- gemma2_instruct_2b_en
- Gemma 2 2B (instruction-tuned)
- QLoRA/LoRA adapters
- Gemma 2B
- GemmaCausalLM (Gemma 2B)
- Gemma-2-2b-jpn-it
- Gemma-2 9B

## Frameworks used
- keras
- keras-nlp
- datasets
- jax
- matplotlib
- kagglehub
- transformers
- trl
- peft
- huggingface_hub
- bitsandbytes
- accelerate
- liger-kernel
- torch
- unsloth
- wandb
- scikit-learn
- nltk
- pandas
- numpy
- tensorflow-cpu

## Loss functions
- SparseCategoricalCrossentropy(from_logits=True)

## CV strategies
- holdout 10%
- Fixed train/validation split (1,000 validation samples loaded from a separate CSV file).

## Insights
- LoRA enables efficient fine-tuning of large models by isolating trainable parameters to a low-rank adapter.
- Instruction-tuned models require strict prompt formatting with turn markers to function correctly.
- Token length filtering is essential for controlling memory usage and preventing training crashes.
- Swapping decoding samplers post-training allows quick experimentation with output randomness without retraining.
- LoRA with a low rank (4) enables efficient fine-tuning of a 2B parameter model on limited GPU memory.
- Reducing the training dataset to 52 rows is a practical workaround for Kaggle's GPU quota constraints.
- The model can generate plausible answers for out-of-distribution questions even without explicit training on them, demonstrating some zero-shot generalization capability.
- Parameter-efficient fine-tuning with LoRA (rank=4) enables adapting large LLMs to specific domains with minimal compute.
- Keras 3 supports multi-backend execution, allowing seamless switching to JAX for efficient training.
- Structured prompt templates are essential for effectively instructing causal LLMs during fine-tuning.
- LoRA fine-tuning with a rank of 4 is sufficient and memory-efficient for adapting Gemma 2B to historical language translation.
- Instruction-tuning prompts require specific control tokens like `<start_of_turn>` and `<end_of_turn>` to properly guide the model's generation behavior.
- Filtering training samples by token length prevents memory fragmentation and OOM errors during training.
- KerasNLP with the JAX backend enables straightforward fine-tuning workflows on Kaggle's free GPU accelerators.
- QLoRA with 4-bit NormalFloat quantization and double quantization drastically reduces VRAM requirements, enabling fine-tuning of large models on consumer GPUs.
- Right-sided padding is critical to prevent numerical instability and buffer overflows during mixed-precision (fp16) training.
- Gradient checkpointing combined with paged_adamw_32bit optimizer significantly lowers memory overhead by trading compute for VRAM savings.
- Merging LoRA adapters into the base model simplifies deployment and inference without requiring PEFT dependencies.
- Cultural adaptation in LLMs can be structured through explicit configuration dictionaries rather than relying solely on prompt engineering.
- Language-specific variants require targeted processing for honorifics, dialects, and text directionality to maintain cultural authenticity.
- Evaluating culturally-aware models should prioritize metrics like cultural accuracy, language fluency, and context retention over standard perplexity or BLEU scores.
- Parameter-efficient fine-tuning via LoRA enables adapting large open-weight models to low-resource languages without full retraining.
- Consistent instruction-response templating significantly improves the model's ability to follow Kazakh prompts.
- Keras 3 with the JAX backend provides a memory-efficient and practical stack for fine-tuning LLMs on consumer GPUs.
- Zero-shot and few-shot prompting are insufficient for accurate Yomigana generation, making supervised fine-tuning necessary.
- Parameter-efficient fine-tuning with LoRA and 4-bit quantization allows LLM fine-tuning on consumer-grade hardware without full model retraining.
- Higher LoRA rank and scaling factor consistently improve both training/validation loss and validation metrics.
- Using a linear learning rate scheduler slightly outperforms cosine scheduling when paired with the AdamW optimizer for this task.
- The choice of prompt wording and language significantly impacts training loss curves, even if validation metrics remain stable.
- Sharding a 9B parameter model across 8 TPUs using Keras 3's distribution API effectively manages memory fragmentation and allocation overhead.
- LoRA drastically reduces trainable parameters from 9 billion to approximately 14 million, enabling large model fine-tuning on limited hardware.
- Clear instruction-response formatting is essential for LLM performance, as unstructured prompts yield incoherent or repetitive outputs.

## Critical findings
- Disabling model caching (use_cache=False) is mandatory during training to ensure correct gradient computation and prevent memory leaks.
- Paged AdamW optimizer is specifically chosen over standard AdamW to manage optimizer states more efficiently under tight memory constraints.
- Right-sided padding mitigates a known float16 overflow issue that occurs with left-sided padding during training.
- Zero-shot prompting accuracy was 0.0 and performed worse than a baseline BLEU score for non-transcribed text, highlighting the need for task-specific alignment.
- The particle は is pronounced as わ in speech but must be kept as は in the dataset to avoid confusing the model during transcription.
- Training and validation loss curves can be heavily influenced by prompt wording and language, even when validation metrics stay consistent.
- The TPU sharding configuration only works with the Gemma-2 9B model and fails with the 2B variant, likely due to memory layout constraints.
- Data pipeline failures (empty data) can prevent LoRA from enabling, highlighting the need for robust data loading checks before model compilation.

## What did not work
- NEFTune (neftune_noise_alpha=0.01) was tested but explicitly noted as not working for this setup.
- Experimenting with varied weight_decay and max_grad_norm values led to decreased validation metrics compared to the baseline, so the author reverted to the default regularization settings.
- The author notes that the setup did not work with the Gemma-2 2B model, and a previous attempt failed because empty data prevented LoRA from being enabled.

## Notable individual insights
- votes 81 (Gemma2 for Inspiring Arabic Story Generation): QLoRA with 4-bit NormalFloat quantization and double quantization drastically reduces VRAM requirements, enabling fine-tuning of large models on consumer GPUs.
- votes 59 (Fine-tuning Gemma 2 JPN for Yomigana with LoRA): Zero-shot prompting accuracy was 0.0 and performed worse than a baseline BLEU score for non-transcribed text, highlighting the need for task-specific alignment.
- votes 55 (Chinese Culture 中國文化 Gemma2Keras): Sharding a 9B parameter model across 8 TPUs using Keras 3's distribution API effectively manages memory fragmentation and allocation overhead.
- votes 458 (How to Finetuning Gemma2 for Spoken Language Tasks): Swapping decoding samplers post-training allows quick experimentation with output randomness without retraining.
- votes 67 (Language-specific adaptations and culture): Cultural adaptation in LLMs can be structured through explicit configuration dictionaries rather than relying solely on prompt engineering.
- votes 66 (Kazakh LLM based on Gemma 2B [KZ]): Consistent instruction-response templating significantly improves the model's ability to follow Kazakh prompts.
- votes 87 (Translator of Old Korean Literature): Filtering training samples by token length prevents memory fragmentation and OOM errors during training.

## Notebooks indexed
- #458 votes [[notebooks/votes_01_bebechien-how-to-finetuning-gemma2-for-spoken-language-tasks/notebook|How to Finetuning Gemma2 for Spoken Language Tasks]] ([kaggle](https://www.kaggle.com/code/bebechien/how-to-finetuning-gemma2-for-spoken-language-tasks))
- #100 votes [[notebooks/votes_02_mpwolke-bhagavad-gita-gemma2keras/notebook|Bhagavad Gita (भगवद्गीता) Gemma2Keras]] ([kaggle](https://www.kaggle.com/code/mpwolke/bhagavad-gita-gemma2keras))
- #98 votes [[notebooks/votes_03_mpwolke-global-communication-gemma2keras/notebook|Global Communication Gemma2Keras]] ([kaggle](https://www.kaggle.com/code/mpwolke/global-communication-gemma2keras))
- #87 votes [[notebooks/votes_04_bebechien-translator-of-old-korean-literature/notebook|Translator of Old Korean Literature]] ([kaggle](https://www.kaggle.com/code/bebechien/translator-of-old-korean-literature))
- #81 votes [[notebooks/votes_05_abdullahabdelrhim-gemma2-for-inspiring-arabic-story-generation/notebook|Gemma2 for Inspiring Arabic Story Generation]] ([kaggle](https://www.kaggle.com/code/abdullahabdelrhim/gemma2-for-inspiring-arabic-story-generation))
- #67 votes [[notebooks/votes_06_satyaprakashswain-language-specific-adaptations-and-culture/notebook|Language-specific adaptations and culture]] ([kaggle](https://www.kaggle.com/code/satyaprakashswain/language-specific-adaptations-and-culture))
- #66 votes [[notebooks/votes_07_armanzhalgasbayev-kazakh-llm-based-on-gemma-2b-kz/notebook|Kazakh LLM based on Gemma 2B [KZ]]] ([kaggle](https://www.kaggle.com/code/armanzhalgasbayev/kazakh-llm-based-on-gemma-2b-kz))
- #59 votes [[notebooks/votes_08_iamleonie-fine-tuning-gemma-2-jpn-for-yomigana-with-lora/notebook|Fine-tuning Gemma 2 JPN for Yomigana with LoRA]] ([kaggle](https://www.kaggle.com/code/iamleonie/fine-tuning-gemma-2-jpn-for-yomigana-with-lora))
- #55 votes [[notebooks/votes_09_mpwolke-chinese-culture-gemma2keras/notebook|Chinese Culture 中國文化 Gemma2Keras]] ([kaggle](https://www.kaggle.com/code/mpwolke/chinese-culture-gemma2keras))
- #55 votes [[notebooks/votes_10_rishirajacharya-fine-tuning-gemma-2-for-bengali-poetry-generation/notebook|Fine-Tuning Gemma 2 for Bengali Poetry Generation]] ([kaggle](https://www.kaggle.com/code/rishirajacharya/fine-tuning-gemma-2-for-bengali-poetry-generation))
