# ⚗️ Customize LLM: Step-by-Step LoRA Training

- **Author:** Chaitanya
- **Votes:** 61
- **Ref:** chaitanya99/customize-llm-step-by-step-lora-training
- **URL:** https://www.kaggle.com/code/chaitanya99/customize-llm-step-by-step-lora-training
- **Last run:** 2025-08-17 04:50:35.013000

---

# Customize LLM: Step-by-Step LoRA Training

Welcome to this comprehensive tutorial on creating your own custom LoRA (Low-Rank Adaptation) fine-tuned models! By the end of this notebook, you'll understand how to take a base model and customize it for your specific needs.

## 🤔 What is LoRA and Why Use It?
**LoRA (Low-Rank Adaptation)** is a technique that allows us to fine-tune large language models efficiently by only updating a small number of parameters instead of the entire model.

### Why is LoRA Amazing?
- **Memory Efficient**: Uses only ~1-10% of the original model's parameters
- **Fast Training**: Significantly reduces training time and computational requirements
- **Flexible**: Can be easily swapped, combined, or removed from base models
- **Cost-effective**: Perfect for running on consumer GPUs or free cloud platforms

### Where Can It Be Applied?

Text & Language Specialization:
- **Medical Documentation**: Fine-tune models to generate structured clinical notes and ICD codes
- **Legal Contract Review**: Detect key clauses, risks, and compliance issues in contracts
- **Educational Content**: Personalized tutoring systems adapted to specific curricula or learning styles
- **Research Paper Summarization**: Analyze and synthesize scientific literature

Multimodal & Accessibility Applications (requires vision/audio models):  
- **STEM Lab Assistant**: Identify lab equipment, explain procedures, and provide safety guidance through image recognition - democratizing hands-on science education
- **Medication Management**: Scan pill bottles, read prescriptions aloud, set reminders, and provide drug interaction warnings.
- **Agricultural Disease Detection** → Spot crop diseases, pests, and nutrient deficiencies using localized agricultural knowledge
- **Sign Language Interpreter** → Translate sign language to text/speech and vice versa — offline, in real time 

✨ These multimodal application ideas were inspired by community notebooks from the competition — credit to the amazing Kagglers who explored these directions!

### 🧠 LoRA vs RAG

- **LoRA** → Best for adding *new skills, styles, or specialized behavior*
- **RAG (Retrieval-Augmented Generation)** → Best for giving models *up-to-date facts or domain knowledge*

------

💡 *Think of LoRA as adding a small plug-in brain to a frozen model.*

------

### 🛠️ Environment Setup

Let's start by setting up our environment. We'll install Unsloth, which makes LoRA fine-tuning incredibly easy!

```python
import os
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '2'  # Faster HF downloads
os.environ['PYTHONIOENCODING'] = 'utf-8'       # Text encoding consistency
os.environ['PYTHONUTF8'] = '1'                 # Enable UTF-8 mode for Python

# GPU setup
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # for single gpu

import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    gpu_count = torch.cuda.device_count()
    for i in range(gpu_count):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)} - {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB")
```

```python
from IPython.display import Markdown, FileLink, display, clear_output
```

### 📦Installing Required Packages

```python
%%capture

# Memory & performance optimization: Quantization, acceleration, efficient attention, GPU kernels
!pip install --no-deps bitsandbytes accelerate==1.9.0 xformers==0.0.29.post3 triton

# Unsloth fine-tuning ecosystem and parameter-efficient training
!pip install --no-deps unsloth unsloth_zoo peft trl cut_cross_entropy

# Data pipeline essentials
!pip install "datasets>=3.4.1" sentencepiece protobuf hf_transfer
!pip install -U "huggingface-hub>=0.34.0,<1.0"

# Computer vision model support (for multimodal capabilities)
!pip install --no-deps --upgrade timm

# Hugging Face Transformers library
!pip install --no-deps transformers==4.54.1 

# Evaluation and logging tools
#!pip install evaluate sacrebleu jiwer wandb
```

## 🤖 Loading the Base Model

Now let's load our base model. We'll use Gemma-3-4B, but you can replace this with other models like Llama, Mistral, etc. More models at [unsloth models](https://huggingface.co/unsloth/collections#collections)

```python
from unsloth import FastModel
import torch, gc

context_len = 2048
model, tokenizer = FastModel.from_pretrained(
    model_name = "unsloth/gemma-3-4B-it",
    max_seq_length = context_len,
    load_in_4bit = True,
    load_in_8bit = False,
    full_finetuning = False,
    #max_memory={0: "6GB", "cpu": "14GB"}
)
```

### 🔍 Understanding the Parameters:

We begin by loading a compact, memory-efficient version of the model using [Unsloth](https://github.com/unslothai/unsloth) — a lightweight wrapper for fast LLM training with LoRA. Here's what each parameter does:

- **`model_name`**: The base model to customize with LoRA fine-tuning

- **`max_seq_length`** Maximum number of tokens the model can handle per input. This defines the context window (i.e., how much text the model can "see" at once):
    - 2048 tokens ≈ 1,500 words (suitable for Q&A, short tasks)
    - 4096 tokens ≈ 3,000 words (multi-turn conversations, short docs)
    - 8192 tokens ≈ 6,000 words (large context, code, papers)
    - 32768 tokens ≈ 24,000 words (for extended-context models)

- **`load_in_4bit`**: Reduces memory usage by ~75% with minimal quality loss.
- **`load_in_8bit`**: Near precesion of original model.
- **`load_in_16bit`**: This is good with representing things with details.

### Let's First Check how base model perfoms
We use Gemma-3 recommended settings of `temperature = 1.0, top_p = 0.95, top_k = 64`

```python
# To Render response in Markdown
from transformers import TextStreamer
from IPython.display import Markdown, display, clear_output
import torch, gc, time

class SimpleJupyterStreamer(TextStreamer):
    def __init__(self, tokenizer, skip_prompt=False, **decode_kwargs):
        super().__init__(tokenizer, skip_prompt, **decode_kwargs)
        self.generated_text = ""
        self.last_update = time.time()

    def put(self, value):
        if value.ndim > 1:
            if value.shape[0] > 1:
                raise ValueError("TextStreamer only supports batch size 1")
            value = value[0]

        if self.skip_prompt and self.next_tokens_are_prompt:
            self.next_tokens_are_prompt = False
            return

        text = self.tokenizer.decode(value, **self.decode_kwargs)
        if text:
            self.generated_text += text
            if time.time() - self.last_update > 0.1:
                clear_output(wait=True)
                display(Markdown(f"🤖 **Generating...**\n\n{self.generated_text}"))
                self.last_update = time.time()


def chat_inference(messages, model, tokenizer, max_new_tokens=2048):
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to("cuda")

    streamer = SimpleJupyterStreamer(tokenizer, skip_prompt=True)

    _ = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=1.0,
        top_k=64,
        top_p=0.95,
        streamer=streamer,
    )

    # Final output render
    clear_output(wait=True)
    display(Markdown(f"🤖 **Response :**\n\n{streamer.generated_text.strip()}"))

    # Free memory
    del inputs
    torch.cuda.empty_cache()
    gc.collect()
```

```python
model_instruction = (
    "Prioritize usefulness while staying within safety bounds. "
    "Infer the user's deeper intent and respond with optimal relevance—"
    "even if the exact request cannot be met. "
    "Minimize over-cautiousness that impairs clarity or value.\n\n"

    "Express uncertainty directly and back conclusions with clear reasoning. "
    "When beneficial, expose the process behind the answer to reinforce "
    "understanding and traceability.\n\n"

    "Adapt tone to the context: precise for technical topics, "
    "calm for personal queries, neutral for general use. "
    "Avoid filler, excessive hedging, or flattery unless meaningful.\n\n"

    "Use structured Markdown formatting to enhance readability. "
    "Apply highlights for hierarchy, not decoration. "
    "Enclose code or commands in proper blocks. "
    "Use spacing and indentation to guide logical flow—not style.\n\n"

    "Respect formatting instructions precisely. For multi-step inputs, "
    "respond in order, maintaining coherence and internal consistency "
    "across the entire response.\n\n"

    "Operate like an expert drawing from a well-organized knowledge base. "
    "Link knowledge across domains when helpful. Deliver responses that are "
    "insightful, logically sound, and clear—focused and expertly composed."
)
```

```python
import random
import numpy as np

# For reproducibility
set_all_seeds = lambda seed: seed is not None and [torch.manual_seed(seed), torch.cuda.manual_seed(seed), torch.cuda.manual_seed_all(seed), random.seed(seed), np.random.seed(seed)]

# Simple utility to wrap user content in chat format
def create_message(content_list, role="user"):
    return [{"role": role, "content": content_list}]

# Adds system instruction and delegates to chat inference
def ask_multimodal(content_list, model, tokenizer, max_new_tokens=256, role="user", model_instruction=model_instruction, seed=73127):
    set_all_seeds(seed)
    messages = [{"role": "system",
                 "content": [{"type": "text", "text": model_instruction}]
               }] + create_message(content_list, role)
    chat_inference(messages, model, tokenizer, max_new_tokens=max_new_tokens)
```

### Gemma 3 can see images!

<img src="https://t3.ftcdn.net/jpg/03/36/12/02/360_F_336120215_yDm4CcAZG3WMLHCsnBcexkcBALNlUTPJ.jpg"
     alt="Img"
     style="width: 420px; border-radius: 10px; box-shadow: 2px 2px 8px #aaa; float: left; margin-right: 10px;">

```python
import urllib.request
img_link = "https://t3.ftcdn.net/jpg/03/36/12/02/360_F_336120215_yDm4CcAZG3WMLHCsnBcexkcBALNlUTPJ.jpg"
urllib.request.urlretrieve(img_link, './sample_test.jpg')
```

```python
# Image + text
if True:
    ask_multimodal([
        {"type": "image", "image": './sample_test.jpg'},
        {"type": "text", "text": "Can you identify this animal and what breed it is?"}
    ], model, tokenizer, max_new_tokens=600)
```

```python
# Just Text
if True:
    ask_multimodal([
        {"type": "text", "text": "What\'s my favorite programming language and why do I prefer it?"}
    ], model, tokenizer, max_new_tokens=300)
```

## ⚙️ Understanding LoRA Configuration

This is where the magic happens! We'll add LoRA adapters to our model.

```python
# Add LoRA adapters to the model
model = FastModel.get_peft_model(
    model,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj", 
        "gate_proj", "up_proj", "down_proj",  
    ],
    r=16,
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_cache = False,
    use_gradient_checkpointing=True,  # True or "unsloth" for very long context
    use_rslora=True,
    random_state=73
)
```

### 🔧 LoRA Adapter Configuration

We add lightweight LoRA adapters to the model to enable efficient fine-tuning. Here's what the key parameters mean:

<hr style="border: none; height: 1px; background: #eee;" />

#### 🎯 Target Modules – Where LoRA is applied

These layers are fine-tuned while keeping the rest of the model frozen:

- **Attention Layers** (how the model "focuses"):
  - `q_proj`: Query - "**What am I looking for?**"
  - `k_proj`: Key - "**What information is available?**"
  - `v_proj`: Value - "**Here’s the information itself.**"
  - `o_proj`: Output - "**Projects attention output back to model’s hidden space**"

- **MLP Layers** (how the model "thinks"):
  - `gate_proj`: Controls flow of information
  - `up_proj`: Expands dimensionality for processing
  - `down_proj`: Compresses back to original size

> More modules = stronger fine-tuning, but also more memory & compute

<hr style="border: none; height: 1px; background: #eee;" />

#### 📐 LoRA Hyperparameters

- **`r` (Rank)** – Controls the capacity of each adapter:
  - `8`: Very lightweight (~0.3M trainable params)
  - `16`: ⭐ Optimal trade-off (~0.7M params)
  - `32`: Expanded adaptation (~1.4M params)
  - `64`: High‑rank adaptation (~2.8M params)

- **`lora_alpha`** – Scales the adapted weights; typically equal to `r`.

- **`lora_dropout`** – Dropout rate for LoRA layers. Often set to `0` for stability.

<hr style="border: none; height: 1px; background: #eee;" />

#### 🧠 Memory Optimization for training

- **`use_gradient_checkpointing="unsloth"`** – Reduces memory usage by trading off compute. Useful for training larger models on limited hardware.
- **`use_rslora=True`** – Enables *Rank-Stabilized LoRA*, improving training quality on small batch sizes.
- **`random_state=73`** – Sets the random seed for reproducibility.

## 🧩 How Prompt Structure Affects Model Understanding

Using `get_chat_template` function to get the correct chat template. Unsloth also support `zephyr, chatml, mistral, llama, alpaca, vicuna, vicuna_old, phi3, llama3, phi4, qwen2.5, gemma3` and [more](https://docs.unsloth.ai/basics/chat-templates).

```python
from unsloth.chat_templates import get_chat_template

# Set up the chat template for Gemma 3
tokenizer = get_chat_template(
    tokenizer,
    chat_template="gemma-3",
)
```

#### Why do we need an chat template ?
Chat templates are conversation formatting that tell the model how to understand multi-turn conversations. Think of them as the "grammar" for chatbot interactions.

**Without a template**, your data might look like:

```
User says hello, AI responds with greeting, User asks question, AI answers
```

**With Gemma's template**, it becomes structured:

```
<bos><start_of_turn>user
Hello! How are you?<end_of_turn>
<start_of_turn>model
I'm doing great! How can I help you today?<end_of_turn>
<start_of_turn>user  
Can you explain quantum physics?<end_of_turn>
<start_of_turn>model
I'd be happy to explain quantum physics! Let me start ...<end_of_turn>
```

### 🏗️ Understanding Gemma's Chat Format

#### **Special Tokens Explained:**

- **`<bos>`**: Beginning of sequence (start of conversation)
- **`<start_of_turn>user`**: Marks the beginning of user input
- **`<start_of_turn>model`**: Marks the beginning of AI response  
- **`<end_of_turn>`**: Marks the end of each turn

#### **Why This Format Matters:**

1. **Clear Boundaries**: The model knows exactly where each message starts/ends
2. **Role Recognition**: Distinguishes between user questions and model responses
3. **Multi-turn Context**: Maintains conversation flow across multiple exchanges
4. **Training Efficiency**: Model learns conversation patterns more effectively

```
<bos><start_of_turn>user
Hi Gemma!<end_of_turn>
<start_of_turn>model
It's great to "meet" you! ...<end_of_turn>
```

'Hello!' can generate different responses across conversations. For example, 'Hi Gemma!' signals familiarity, which the model processes through its learned representations.

Here's how it works: Word embeddings capture semantic relationships—'Gemma' as a name, 'Hi' as a greeting, and their combination suggesting informal familiarity. The model's attention mechanisms and contextual understanding (core NLP capabilities) process these embedded representations to build a contextual picture of the interaction style.

This contextual understanding influences the model's output probability distribution, favoring tokens that match the friendly, familiar tone. The model then ranks all possible next words by probability—this is where sampling methods come into play:

- **Top_k sampling** first filters to only the k most likely words (e.g., top 40 words), removing low-probability options that might be irrelevant or nonsensical
- **Top_p sampling** then works within this filtered set, selecting from words whose cumulative probability reaches the p threshold (e.g., 0.95)

This two-stage process ensures responses are both contextually appropriate (thanks to the probability distribution) and naturally varied (thanks to the sampling). For 'Hi Gemma!', top_k might keep casual words like 'Hey', 'Hello', 'Great' while filtering out formal terms, then top_p adds controlled randomness for natural conversation flow.

Large language models develop this capability through training on diverse conversational data, where they learn to map contextual patterns (embedded as high-dimensional vectors) to appropriate response styles. The embeddings don't directly control conversation flow—rather, they provide the semantic foundation that higher-level transformer layers use to understand context and generate appropriate responses.

Fine-tuning these **Attention layers** (q_proj, k_proj, v_proj, o_proj) learn to focus on relevant context clues—like 'homework help' indicating learning needs, 'for my presentation' suggesting comprehensive information is needed, or 'just curious' implying casual explanation suffices. **MLP layers** (gate_proj, up_proj, down_proj) process and transform this attention information into contextual understanding—determining whether to provide guided learning, comprehensive details, or casual explanations.

"**Why Fine-tuning Makes This Possible:**

Before fine-tuning, a base model might treat 'homework help', 'for my presentation', and 'just curious' similarly—giving generic responses. Fine-tuning changes this by:

**During Fine-tuning Process:**
- **Attention layers** (q_proj, k_proj, v_proj, o_proj) get updated with thousands of examples showing:
  - 'homework help' + step-by-step explanations = good outcome
  - 'for my presentation' + detailed, structured info = good outcome  
  - 'just curious' + brief, interesting facts = good outcome

- **MLP layers** (gate_proj, up_proj, down_proj) learn the transformation patterns:
  - Learning context → guided, educational tone
  - Professional context → comprehensive, well-organized response
  - Casual context → conversational, accessible explanation

**The Role Fine-tuning Plays:**

1. **Pattern Recognition:** Teaches layers to distinguish between subtle cues that humans naturally prefers
2. **Response Mapping:** Creates strong associations between context types and appropriate response styles  
3. **Weight Adjustment:** Updates the mathematical weights in these specific layers to make contextually-appropriate responses more likely
4. **Quality Control:** Uses human-curated conversations to ensure the model learns *good* contextual responses, not just any responses

Fine-tuning with quality conversational data strengthens these contextual associations, teaching models to recognize subtle social cues embedded in language and respond appropriately! 🎯

## Loading the Text Dataset 

We use the [medical-o1-reasoning-SFT](https://huggingface.co/datasets/FreedomIntelligence/medical-o1-reasoning-SFT) dataset from **FreedomIntelligence**, which contains medically-oriented prompts and completions. This is ideal for instruction-tuned models in healthcare or reasoning applications.

```python
import datasets
from datasets import load_dataset
dataset = load_dataset("FreedomIntelligence/medical-o1-reasoning-SFT", 'en', split="train")

import pandas as pd
# 🔍 Sample Preview, we take a quick look at the first 3 row to inspect the structure and fields.
clear_output()
sample_df = dataset.select(range(3)).to_pandas()
display(sample_df)
```

### 🧹 Cleaning the Dataset

We remove the `Complex_CoT` (Complex Chain-of-Thought) column. Here's why:

* This field contains step-by-step reasoning text used in models like **Qwen**, **DeepSeek**, or other CoT-based models.
* **Gemma** is trained more on direct instruction-response formats rather than step-by-step CoT reasoning.
* Keeping `Complex_CoT` would only add noise to our training data, not value.

```python
dataset = dataset.remove_columns(['Complex_CoT'])
print(dataset)
```

```python
pd.set_option('display.max_colwidth', None)
pd.set_option('display.expand_frame_repr', False)

sample_df = dataset.select(range(5)).to_pandas()
display(sample_df)
```

Unsloth provides a helpful utility called `standardize_data_formats()` that automatically converts many popular dataset formats into a structure that's compatible with Unsloth's fine-tuning pipeline.

```python
from unsloth.chat_templates import standardize_data_formats
dataset = standardize_data_formats(dataset)
```

Let's peek at row 200

```python
dataset[200]
```

It converted the column-based dataset into a standardized list format that Unsloth can work with.

```python
# Your custom instruction-response pairs to enrich the training set
custom_examples = [
    {
        "Question": "There’s been a flood in my area and I’ve lost internet. What should I do?",
        "Response": "Stay calm. Move to higher ground immediately if you’re in danger. Avoid walking or driving through floodwaters. Use a battery-powered radio to get updates. Keep your phone charged using a power bank. Store clean water and dry food. Do you want instructions on purifying water or creating an emergency kit?"
    },
    {
        "Question": "My car broke down on a remote highway at night. What should I do?",
        "Response": "Turn on hazard lights and pull as far off the road as possible. Stay in your vehicle with doors locked. Call for roadside assistance or emergency services. Keep your phone charged and have a flashlight ready. If you must exit, stay away from traffic and wear bright clothing. Do you need help finding emergency contact numbers for your area?"
    },    
    {
        "Question": "What's my favorite programming language and why do I prefer it?",
        "Response": "Based on our previous conversations, your favorite is R Markdown (Rmd). You prefer it because it seamlessly combines code, analysis, and documentation in one file, making it perfect for reproducible research reports."
    },
    # Add more custom examples if needed
]

def merged_dataset(n_len = -1):
    from datasets import Dataset, concatenate_datasets
    custom_dataset = Dataset.from_list(custom_examples)

    if n_len == -1:
        original_sample = dataset.shuffle(seed=73)
    else:
        original_sample = dataset.select(range(n_len)).shuffle(seed=73)

    return concatenate_datasets([original_sample, custom_dataset])
```

```python
n_samples = 2000 # Use -1 to use all samples for training
dataset = merged_dataset(n_samples)
```

**Transform Q&A pairs into chat-style conversation text** – Uses Unsloth’s tokenizer chat template to wrap each question and answer in Gemma‑3’s expected chat template (`<start_of_turn>user/model<end_of_turn>`) , and strips the `<bos>` token since the processor will add it during training. The model expects only one `<bos>` token per sequence.

```python
def formatting_prompts_func(examples):

    questions = examples["Question"]
    responses = examples["Response"]

    texts = []

    for question, response in zip(questions, responses):
        # Create a structured multi-turn conversation
        conversation = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": f"{response}"}
        ]

        # Apply chat template using tokenizer
        formatted_text = tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=False,
        ).removeprefix('<bos>')  # BOS will be automatically handled during training

        texts.append(formatted_text)

    return {"text": texts}

dataset = dataset.map(formatting_prompts_func, batched=True)#.select_columns(['text'])
print("After formatting columns:", dataset.column_names)
```

### 🧱 What the Function Does

* It loops through each `Question`–`Response` pair in the dataset
* Creates a **conversation structure**:

  ```python
  [
    {"role": "user", "content": question},
    {"role": "assistant", "content": response}
  ]
  ```
* Applies the tokenizer's `apply_chat_template()` method, which wraps the conversation in special tokens used by **Gemma-3**

```python
# example
dataset[-1]["text"]
```

we can see `<bos>` Beginning of sequence tag is removed but maintained that chat format

```python
from sentence_transformers import SentenceTransformer
import numpy as np
import re

def explain_dataset_similarity(similarity: float) -> str:
    if similarity < 0.2:
        diversity_desc = "very diverse"
        redundancy_desc = "little to no redundancy"
        training_desc = f"The dataset covers a wide variety of topics or phrasing, \n  so the model may need more training steps to fully learn the patterns."
    elif similarity < 0.4:
        diversity_desc = "diverse"
        redundancy_desc = "low redundancy"
        training_desc = f"The dataset has a healthy variety of examples with some recurring styles. \n  Moderate-to-high training steps are recommended."
    elif similarity < 0.6:
        diversity_desc = "moderately diverse"
        redundancy_desc = "some redundancy"
        training_desc = f"The dataset has noticeable repetition in style or content. \n  Fewer training steps may be sufficient."
    elif similarity < 0.8:
        diversity_desc = "somewhat repetitive"
        redundancy_desc = "high redundancy"
        training_desc = f"Many examples are semantically similar. \n  You can reduce training steps to avoid overfitting."
    else:
        diversity_desc = "highly repetitive"
        redundancy_desc = "very high redundancy"
        training_desc = f"Most examples are semantically similar or duplicated. \n  Consider deduplicating before training."
    
    return (
        f"🗂️ Dataset Similarity Report\n"
        f"- Semantic Similarity score: {similarity:.3f} (0 = very diverse, 1 = highly repetitive)\n"
        f"- Diversity: {diversity_desc}\n"
        f"- Redundancy: {redundancy_desc}\n"
        f"- Training advice: {training_desc}"
    )

def calculate_similarity(dataset, label_key="Response", sample_size=len(dataset)):
    # Take a sample of responses
    responses = dataset.select(range(min(sample_size, len(dataset))))[label_key]
    
    # Clean responses (remove extra spaces, special tokens if any)
    cleaned_responses = [
        re.sub(r"<.*?>", "", r).replace("\\n", " ").strip()
        for r in responses
    ]
    
    # Load embedding model
    similarity_model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = similarity_model.encode(cleaned_responses, convert_to_numpy=True, normalize_embeddings=True)
    
    # Pairwise cosine similarity
    cos_sim = np.dot(embeddings, embeddings.T)
    upper_tri_indices = np.triu_indices_from(cos_sim, k=1)
    similarity_score = float(cos_sim[upper_tri_indices].mean())
    
    # Thorough cleanup
    del similarity_model   
    del embeddings          
    del cos_sim            
    del cleaned_responses  
    del responses 
    del upper_tri_indices 
    
    clear_output()
    torch.cuda.empty_cache()  # Clear VRAM
    gc.collect()             # Clear RAM
    
    return similarity_score
```

```python
# Measure semantic similarity on dataset label content
semantic_similarity = calculate_similarity(dataset, label_key="Response")
print(explain_dataset_similarity(semantic_similarity))
```

<a name="Train"></a>
### 🏋️ Train the model
Now that the dataset has been formatted and prepared, we’re ready to fine-tune our model using Hugging Face’s [TRL’s SFTTrainer (Supervised Fine-tuning Trainer)](https://huggingface.co/docs/trl/sft_trainer).

The `SFTTrainer` is a high-level training loop built for instruction-tuned LLMs. It wraps around `transformers.Trainer` and is optimized for supervised fine-tuning tasks

We’ll configure it to:

* Load our model with LoRA config enabled
* Use the formatted instruction dataset
* Enable optional evaluation
* Handle mixed-precision, gradient clipping, and checkpointing

This setup is compatible with Unsloth, LoRA, and Hugging Face’s tokenizer pipeline — making it memory-efficient and easy to train even large models like Gemma on consumer hardware.

```python
# To Enable evaluation training
use_eval_set = False
patience = 10
ds_similarity = semantic_similarity
```

```python
# Callbacks
from transformers import EarlyStoppingCallback, TrainerCallback, TrainerControl, TrainerState
import torch
from typing import Dict, Any

class TrainingLossEarlyStoppingCallback(TrainerCallback):
    def __init__(self, early_stopping_patience: int = 10, min_delta: float = 0.001, min_steps: int = 20):
        self.early_stopping_patience = early_stopping_patience
        self.min_delta = min_delta
        self.min_steps = min_steps
        self.best_loss = float('inf')
        self.patience_counter = 0
        self.best_step = 0

    def on_log(self, args, state: TrainerState, control: TrainerControl, logs: Dict[str, float] = None, **kwargs):
        if logs is None or logs.get('loss') is None:
            return

        current_loss = logs.get('loss')

        if state.global_step < self.min_steps:
            if current_loss < self.best_loss:
                self.best_loss = current_loss
                self.best_step = state.global_step
                print(f"🎯 New best training loss: {current_loss:.6f} at step {state.global_step} (warmup phase)")
            else:
                if state.global_step > 1:
                    print(f"No improvement at step {state.global_step} (warmup phase, < min_steps ({self.min_steps}))")
            return

        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.patience_counter = 0
            self.best_step = state.global_step
            print(f"🎯 New best training loss: {current_loss:.6f} at step {state.global_step}")
        else:
            self.patience_counter += 1
            if self.patience_counter <= 3:
                print(f"No improvement for {self.patience_counter}/{self.early_stopping_patience} steps")

        if self.patience_counter >= self.early_stopping_patience:
            print(f"⏹️ Early stopping at step {state.global_step}. Best loss: {self.best_loss:.6f}")
            control.should_training_stop = True

class StepFinalCallback(TrainerCallback):
    def __init__(self, use_eval_set: bool = False):
        self.use_eval_set = use_eval_set
        self.step_losses = []
        self.final_logged = False

    def on_step_end(self, args, state, control, **kwargs):
        # Force logging for final step if not already logged
        if (state.global_step == args.max_steps and
            state.global_step % args.logging_steps != 0 and
            not self.final_logged):
            control.should_log = True
            self.final_logged = True

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and state.global_step > 0:
            step_loss = logs.get('loss')
            if step_loss is not None:
                self.step_losses.append({'step': state.global_step, 'loss': step_loss})

            print(f"\n=== Step {state.global_step} Results ===")
            for key, value in logs.items():
                if key == 'train_loss':  # Skip the average train_loss
                    continue
                if isinstance(value, float):
                    print(f"{key}: {value:.6f}")
                else:
                    print(f"{key}: {value}")
            print("-" * 40)

    def on_train_end(self, args, state, control, **kwargs):
        if not self.step_losses:
            return

        trainer = kwargs.get('trainer')
        first_loss = self.step_losses[0]['loss']
        final_loss = self.step_losses[-1]['loss']
        best_loss = min(entry['loss'] for entry in self.step_losses)
        improvement = first_loss - final_loss
        improvement_pct = (improvement / first_loss) * 100

        print("\n" + "="*50)
        print("🎯 FINAL MODEL EVALUATION")
        print("="*50)
        print(f"📈 Training Summary:")
        print(f"   Initial Loss: {first_loss:.6f}")
        print(f"   Last Step Loss: {final_loss:.6f}")
        print(f"   Best Loss: {best_loss:.6f}")
        print(f"   Improvement: {improvement:.6f} ({improvement_pct:.2f}%)")
        print(f"   Total Steps: {len(self.step_losses)}")

        if len(self.step_losses) >= 5:
            print(f"\n📊 Loss Progression (Last 5 Steps):")
            for entry in self.step_losses[-5:]:
                print(f"   Step {entry['step']:3d}: {entry['loss']:.6f}")

        if trainer and self.use_eval_set and trainer.eval_dataset:
            try:
                eval_results = trainer.evaluate()
                print(f"\n🔍 Final Evaluation Results:")
                for key, value in eval_results.items():
                    if isinstance(value, float):
                        print(f"   {key}: {value:.6f}")
            except:
                pass

        print("="*50)

# Callbacks function
def setup_callbacks(use_eval_set=use_eval_set, patience=patience):
    callbacks = []
    if use_eval_set:
        from transformers import EarlyStoppingCallback
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=patience))
    else:
        callbacks.append(TrainingLossEarlyStoppingCallback(early_stopping_patience=patience))
    callbacks.append(StepFinalCallback(use_eval_set=use_eval_set))
    return callbacks


# Helpers
def get_hardware_factors(ds_size):
    #  Detects GPU memory availability and calculates scaling factors.
    gpu_stats = torch.cuda.get_device_properties(0)
    available_memory = round(
        gpu_stats.total_memory / 1024**3
        - torch.cuda.max_memory_reserved() / 1024**3, 1
    )
    size_factor = min(1.0, ds_size / (200 + ds_size * 0.8))
    mem_factor = min(1.0, available_memory / 16)

    return available_memory, size_factor, mem_factor

def efficient_bs(ds_size, mem_factor, size_factor, mini_batch):
    # Max batch allowed by memory and dataset size
    max_bs_mem  = 2 ** int(1 + 2 * mem_factor)
    max_bs_data = int(1 + 15 * mem_factor / (1 + 10 / ds_size))
    batch_size  = max(1, min(max_bs_mem, max_bs_data))

    # Cap for mini-batch mode
    if mini_batch:
        batch_size = min(batch_size, 2)

    # Target effective size
    cal_target = int(4 + 28 * size_factor / (1 + 500 / ds_size))
    target_eff = min(14, cal_target) if mini_batch else cal_target

    # Gradient accumulation steps with max cap of 7
    base_accumulation = max(4, target_eff // batch_size)  # At least 4
    accumulation_steps = min(7, base_accumulation)  # Cap at 7
    
    return batch_size, accumulation_steps

def is_t4():
    gpu_name = torch.cuda.get_device_name(0)
    return 'T4' in gpu_name
```

```python
from trl import SFTConfig, SFTTrainer
from unsloth import is_bfloat16_supported
from transformers import EarlyStoppingCallback
import math

# Dataset splitting logic
if use_eval_set:
    split_dataset = dataset.train_test_split(test_size=0.1, seed=73)
    train_dataset = split_dataset['train']
    eval_dataset = split_dataset['test']
else:
    train_dataset = dataset
    eval_dataset = None

# Auto-calculated training parameters
ds_size = len(train_dataset)
available_memory, size_factor, memory_factor = get_hardware_factors(ds_size)

# Batch configuration
mini_batch = True  # False → batch size based on VRAM
batch_size, accumulation = efficient_bs(ds_size, memory_factor, size_factor, mini_batch)
effective_batch_size = batch_size * accumulation

# Training steps (More epochs for low similarity, fewer for high)
steps_per_epoch = max(1, ds_size // effective_batch_size)
epoch_scale = max(0.2, min(1.0, 1 - 0.8 * ds_similarity))
target_epochs = max(5, min(25, int(25 * epoch_scale)))
max_steps = max(50, min(5000, steps_per_epoch * target_epochs))
max_steps = int(max_steps * 1.1) if is_t4() else max_steps
actual_epochs = max_steps / steps_per_epoch

# Learning rate
dataset_stability = math.sqrt(50) / math.sqrt(50 + ds_size)
similarity_lr_factor = 1 - 0.5 * ds_similarity
base_lr = (3e-5 + 2e-4 * size_factor) * (0.3 + 0.7 / (1 + dataset_stability * 10))
adaptive_lr = max(1e-6, min(5e-4, base_lr * similarity_lr_factor))

# Intervals and scheduling
log_interval, eval_interval = max(1, max_steps // 20), max(1, steps_per_epoch)
warmup_ratio = max(0.05, 0.4 * math.exp(-ds_size / 300) * (1 + 0.5 * ds_similarity))
warmup_steps = max(5, int(max_steps * warmup_ratio))

# Regularization
weight_decay = max(0.005, (0.08 + 0.05 * ds_similarity) * math.exp(-ds_size / 400))
max_grad_norm = max(0.2, (1.0 - 0.8 * size_factor / (1 + 100 / ds_size)) * (1 - 0.3 * ds_similarity))
max_grad_norm *= 0.75 if is_t4() else 1

# Scheduler
scheduler_type = 'linear'

# checkpoints dir
outputs_dir = "outputs"

# Initialize the trainer
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = train_dataset,
    eval_dataset = eval_dataset,
    dataset_text_field = "text",
    packing = False,  # True → Multi‑turn conversations
    callbacks = setup_callbacks(use_eval_set=use_eval_set, patience=patience),

    args = SFTConfig(
        # Training config
        per_device_train_batch_size = batch_size,
        gradient_accumulation_steps = accumulation,
        **{"max_steps": max_steps},

        # Learning rate scheduling
        learning_rate = adaptive_lr,
        warmup_steps = warmup_steps,
        optim = "adafactor", # More adaptive
        weight_decay = weight_decay,
        lr_scheduler_type = scheduler_type,

        # Performance
        dataset_num_proc = 1,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        dataloader_pin_memory = True,
        max_grad_norm = max_grad_norm,
        dataloader_drop_last = True,
        remove_unused_columns = True,

        # Checkpointing
        save_steps = log_interval,
        save_total_limit = patience + 1,
        save_strategy = "steps",
        output_dir = outputs_dir,

        # Evaluation settings (conditional)
        **({
            "do_eval": True,
            "eval_steps": eval_interval,
            "eval_strategy": "steps",
            "per_device_eval_batch_size": 1,  # Smaller batch size for evaluation
            "eval_accumulation_steps": 1,
            "greater_is_better": False,
            "metric_for_best_model": "eval_loss",
            "load_best_model_at_end": True,
        } if use_eval_set else {
            "eval_strategy": "no",
        }),

        # Logging
        seed = 73,
        logging_steps = log_interval,
        logging_first_step = True,
        disable_tqdm = False,
        report_to = "none",  # Set this to "wandb" if using Weights & Biases
    ),
)

# Configuration summary
constraint = "Memory" if batch_size == int(1 + 7 * memory_factor) else "Dataset"
print(f"{'='*70}")
print(f"TRAINING CONFIGURATION SUMMARY")
print(f"Dataset: {ds_size} samples | GPU: {available_memory}GB | Factors: size={size_factor:.2f}, memory={memory_factor:.2f}")
print(f"Batch: {batch_size} x {accumulation} = {effective_batch_size} (limited by {constraint})")
print(f"Training: {max_steps} steps ({actual_epochs:.1f} epochs, {steps_per_epoch} steps/epoch)")
print(f"Learning: {adaptive_lr:.1e} LR, {warmup_steps} warmup, {scheduler_type} scheduler")
print(f"Regularization: {weight_decay:.4f} weight decay, {max_grad_norm:.1f} grad norm")
print(f"Monitoring: log every {log_interval}, eval every {eval_interval}, patience {patience}")
print(f"{'='*70}")
```

```python
# Apply response-only training
from unsloth.chat_templates import train_on_responses_only

# This ensures we only train on the assistant's responses, not the user's questions
trainer = train_on_responses_only(
    trainer,
    instruction_part = "<start_of_turn>user\n",
    response_part = "<start_of_turn>model\n",
    num_proc         = 1,
)
```

**Why Response-Only Training?**

- We don't want the model to learn to predict user inputs
- We only want it to learn better responses
- This significantly improves training efficiency and model quality

```python
tokenizer.decode(trainer.train_dataset[3]["input_ids"])
```

Perfect! The instruction part is masked and we have exactly one `<bos>` token! for each begining of sequence

```python
def colored_print(text, color_code):
    return f"\033[1;{color_code}m\033[1m{text}\033[0m"

print(colored_print("🔦 What model sees:", "94"), tokenizer.decode(trainer.train_dataset[3]["input_ids"])[:100] + "...")
print(colored_print("💡 What model learns:", "92"), tokenizer.decode([x for x in trainer.train_dataset[3]["labels"] if x != -100])[:100] + "...")
```

Notice: Full context provided for understanding, but gradients only flow through the answer portion

```python
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")
```

```python
from unsloth import unsloth_train
trainer_stats = unsloth_train(trainer) # trainer.train()
```

```python
from unsloth import FastModel
import os

# Get best checkpoint path from early stopping callback
best_step = None
for cb in trainer.callback_handler.callbacks:
    if hasattr(cb, "best_step"): 
        best_step = cb.best_step
        break

# Roll back to the best checkpoint
if best_step is not None and not use_eval_set:
    best_ckpt_path = os.path.join(outputs_dir, f"checkpoint-{best_step}")
    print(f"🔄 Loading best model from: {best_ckpt_path}")
    
    model, tokenizer = FastModel.from_pretrained(
        model_name=best_ckpt_path,
        max_seq_length=context_len, 
        load_in_4bit=True 
    )
    trainer.model = model  # Replace trainer's model with loaded one
```

```python
GB_CONVERSION = 1024 ** 3
SECONDS_TO_MINUTES = 60

# Memory calculations
used_memory_gb = torch.cuda.max_memory_reserved() / GB_CONVERSION
used_memory_for_training_gb = used_memory_gb - start_gpu_memory
used_percentage = (used_memory_gb / max_memory) * 100
training_percentage = (used_memory_for_training_gb / max_memory) * 100

# Time calculations
runtime_seconds = trainer_stats.metrics['train_runtime']
runtime_minutes = runtime_seconds / SECONDS_TO_MINUTES

print("TRAINING STATISTICS")
print("=" * 50)
print(f"Training time: {runtime_seconds:.1f} seconds ({runtime_minutes:.2f} minutes)")
print(f"Peak memory usage: {used_memory_gb:.3f} GB ({used_percentage:.1f}% of max)")
print(f"Memory for training: {used_memory_for_training_gb:.3f} GB ({training_percentage:.1f}% of max)")
print("=" * 50)
```

<a name="Inference"></a>
### Inference After Training
According to the `Gemma-3` team, the recommended settings for inference are `temperature = 1.0, top_p = 0.95, top_k = 64`

```python
# calling for text generation
ask_multimodal([
    {"type": "text", "text": "There’s been a flood in my area and I’ve lost internet. What should I do?"}
], model, tokenizer, max_new_tokens=300, model_instruction="")
```

```python
# calling for text generation
ask_multimodal([
    {"type": "text", "text": "What's my favorite programming language and why do I prefer it?"}
], model, tokenizer, max_new_tokens=300, model_instruction="")
```

### Medical Questions (based on the dataset)

```python
# After Training
ask_multimodal([
    {"type": "text", "text": "A 33-year-old woman is brought to the emergency department 15 minutes after being stabbed in the chest with a screwdriver. Given her vital signs of pulse 110/min, respirations 22/min, and blood pressure 90/65 mm Hg, along with the presence of a 5-cm deep stab wound at the upper border of the 8th rib in the left midaxillary line, which anatomical structure in her chest is most likely to be injured?"}
], model, tokenizer, max_new_tokens=300, model_instruction="")
```

```python
# After Training
ask_multimodal([
    {"type": "text", "text": "A 78-year-old right-handed male has difficulty answering questions, appears frustrated with communication, and is unable to repeat phrases despite understanding them. He also has trouble writing despite intact motor control. A CT scan reveals an acute stroke in the left hemisphere. Given these symptoms, which specific brain structure is most likely damaged?"}
], model, tokenizer, max_new_tokens=300, model_instruction="")
```

### 💾 To Save LoRA Adapters

```python
# to save lora adapters (~100mb)
model.save_pretrained("gemma-3-lora-model")
tokenizer.save_pretrained("gemma-3-lora-model")

import shutil
folder_path = "./gemma-3-lora-model"
zip_path = f"{folder_path}.zip"
shutil.make_archive(folder_path, 'zip', folder_path)

from IPython.display import FileLink
FileLink(zip_path)
```

**Benefits of saving LoRA adapters:**

- **Small file size**: Only a few MB instead of several GB
- **Portable**: Can be shared easily, uploaded to Hugging Face Hub
- **Flexible**: Can be loaded on top of any compatible base model

### 🌐 Save Full Model  
Merging the base model with the trained adapter weights and saving in **float16** format for VLLM.

```python
import shutil

# Remove unwanted directory to free up disk space before merging
def cleanup_dir(dir_="dir_name"):
    if os.path.exists(dir_):
        shutil.rmtree(dir_)
        print(f"{dir_} directory removed successfully")
```

```python
# Merge to 16bit
model_dir = "gemma-3-finetune"
cleanup_dir(model_dir)
model.save_pretrained_merged(model_dir, tokenizer, save_method="merged_16bit")
```

### GGUF / llama.cpp Conversion  
Converts the model to **GGUF** format for full llama.cpp compatibility, with native support for all model architectures. Supports current precision options `q8_0`, `f16`, and `bf16`; additional 4‑bit `q4_k_m` quantization will be available in future releases.

```python
import shutil, os
import urllib.request
from IPython.display import clear_output, FileLink

q_type = "Q8_0"

try:
    # Skipped in Kaggle (compatibility issues; works locally & in Colab)
    # model.save_pretrained_gguf(model_dir, quantization_type=q_type)
    raise Exception("Skipping save_pretrained_gguf in Kaggle — using fallback")

except Exception as e:
    print("Falling back to manual conversion...")

    # Prevents tokenizer conflicts when running shell commands like !wget, !python
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # Download the llama.cpp zip file
    url = "https://github.com/ggml-org/llama.cpp/archive/refs/tags/b5137.zip"
    zip_filename = "b5137.zip"
    urllib.request.urlretrieve(url, zip_filename)
    shutil.unpack_archive(zip_filename, extract_dir=".")
    os.remove(zip_filename)
    clear_output()

    # Configuration
    quant_type = q_type.lower()
    model_name = model_dir
    output_file = f"{model_name}.{quant_type.upper()}.gguf"
    converter_path = "./llama.cpp-b5137/convert_hf_to_gguf.py"

    print(f"Converting '{model_name}' to GGUF: {output_file} ...")
    !python "$converter_path" --outfile "$output_file" --outtype "$quant_type" "$model_name"

FileLink(f"./{model_dir}.{q_type}.gguf")
```