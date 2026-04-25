# DSA-SFT=>GRPO-noLora-tunix

- **Author:** Daniel Wycoff
- **Votes:** 94
- **Ref:** danielwycoff/dsa-sft-grpo-nolora-tunix
- **URL:** https://www.kaggle.com/code/danielwycoff/dsa-sft-grpo-nolora-tunix
- **Last run:** 2025-12-15 16:26:51.643000

---

# Gemma 3 (1B‑IT) Dual‑Stream Training – **SFT → GRPO (DSA‑CAST, No‑LoRA)**

This notebook glues together two workflows into a **single, end‑to‑end training pipeline** on Gemma 3‑1B‑IT:

1. **Supervised Fine‑Tuning (SFT)** – teach the model to answer math questions in a **structured Dual‑Stream** format.
2. **GRPO (Group Relative Policy Optimization)** – further **align** the model to that format and reward correctness and structure.

Training is **full‑parameter** in both stages (no LoRA adapters).

---

## The DSA monologue structure

Here, “DSA” is a **Dual‑Stream Architecture**-based answering pattern with an internal monologue that is explicitly structured into four named sections:

Inside the `<reasoning>...</reasoning>` block, the model must always write:

- **Plan** – high‑level steps it will take to solve the problem.  
- **Reasoning** – detailed step‑by‑step execution.  
- **Evidence** – citations, calculations, and explicit checks that support the reasoning.  
- **Sanity_check** – a quick check that the final answer “makes sense” (magnitude, units, edge‑cases).

Then, outside the monologue, the model must put the final result in a separate `<answer>...</answer>` block:

```text
<reasoning>
Plan:
  ...

Reasoning:
  ...

Evidence:
  ...

Sanity_check:
  ...
</reasoning>
<answer>
  42
</answer>
```

This gives you:

- A **human‑readable monologue stream** for oversight and debugging.
- A **machine‑readable answer stream** for automatic grading and downstream tools.

For the conceptual motivation and design details, see the accompanying whitepaper:  
[The Inner Monologue: A Dual‑Stream Architecture for Verifiable Inner Alignment](https://docs.google.com/document/d/1np-I9zEKArodlDhQzfydhloCXIVK9O72g3OJSuo_-Wk/edit?usp=sharing)

---

## How this notebook is organized

1. **Part 1 – SFT (Structured Dual‑Stream Supervised Fine‑Tuning)**
   - Load Gemma 3‑1B‑IT via Kaggle/Tunix (no HF token needed).
   - Format GSM8K into the new DSA template:
     - `<reasoning>` block with **Plan / Reasoning / Evidence / Sanity_check** sections.
     - Separate `<answer>` block with only the final scalar.
   - Train with SFT (no LoRA).
   - Optionally do a quick post‑SFT generation sanity‑check.
   - **Zip and clean up the SFT checkpoints** so you keep a single artifact.

2. **Part 2 – GRPO (DSA‑CAST Reinforcement Learning)**
   - Re‑build a GSM8K‑style dataset for RL rollouts using the same template.
   - Define **DSA‑CAST rewards** that look at:
     - Dual‑Stream tags,
     - Plan/Reasoning/Evidence/Sanity_check structure,
     - and math correctness/completeness.
   - Run GRPO with Tunix’ `RLCluster` + `GRPOLearner` (no LoRA).
   - Evaluate before/after GRPO on GSM8K.
   - Export the **final GRPO actor checkpoint as a single zip** and clean up.

By default, the hyperparameters are set for a **debug‑scale run** so you can validate wiring and behavior.  
Once you’re satisfied, you can increase `MAX_STEPS` etc. for a longer training run.

## Part 1 — Supervised Fine‑Tuning (SFT): Teaching the DSA Monologue

This section is the original **SFT notebook**, lightly edited:

- It uses GSM8K to teach the model to respond with a structured monologue inside `<reasoning>...</reasoning>` containing:
  - Plan
  - Reasoning
  - Evidence
  - Sanity_check
- It keeps a separate `<answer>...</answer>` block for the final scalar answer.
- Hyperparameters are reduced so that training runs quickly.
- At the end of Part 1, we zip the SFT checkpoints and clean up their directory.

```python
import jax
import jax.numpy as jnp
import os

print(f"JAX version: {jax.__version__}")
print(f"Number of devices: {len(jax.devices())}")
print(f"Device kind: {jax.devices()[0].device_kind}")
print(f"JAX backend: {jax.default_backend()}")
print(f"\nDevices:")
for i, device in enumerate(jax.devices()):
    print(f"  [{i}] {device}")
print("="*60)

if jax.default_backend() != 'tpu':
    print("\n⚠️  WARNING: Not running on TPU!")
    print(f"   Current backend: {jax.default_backend()}")
    print("   Make sure you've selected TPU runtime in Kaggle")
else:
    print("\n✓ TPU backend confirmed")


os.environ['XLA_FLAGS'] = (
    '--xla_gpu_enable_triton_softmax_fusion=true '
    '--xla_gpu_triton_gemm_any=True '
    '--xla_gpu_enable_async_collectives=true'
)
os.environ['JAX_COMPILATION_CACHE_DIR'] = '/tmp/jax_cache'
os.environ['LIBTPU_INIT_ARGS'] = '--xla_enable_async_all_gather=true'

jax.config.update('jax_enable_x64', False)  # Use 32-bit for speed
jax.config.update('jax_default_matmul_precision', 'high')  # BF16 matmuls
```

```python
KAGGLE_MODEL_HANDLE = "google/gemma-3/transformers/gemma-3-1b-it"

MAX_SEQ_LENGTH = 2048
MESH_SHAPE = (1, 4) 
TRAIN_MICRO_BATCH_SIZE = 2 

GRADIENT_ACCUMULATION_STEPS = 4 

LEARNING_RATE = 2e-5 
    
NUM_EPOCHS = 1  # DEBUG: 1 epoch for quick sanity check       


MAX_STEPS = 3500  # DEBUG: cap total SFT steps for quick run 
WARMUP_STEPS = int(0.1 * MAX_STEPS)

ADAM_BETA1 = 0.9

ADAM_BETA2 = 0.999 

ADAM_EPSILON = 1e-8


WEIGHT_DECAY = 0.01 
MAX_GRAD_NORM = 1.0

print(f"Global Batch Size: {TRAIN_MICRO_BATCH_SIZE * 8 * GRADIENT_ACCUMULATION_STEPS}")
print(f"Total Training Steps: {MAX_STEPS}")


CHECKPOINT_DIR = "/kaggle/working/outputs_sft_full/checkpoints"
TENSORBOARD_DIR = "/kaggle/working/outputs_sft_full/tensorboard"
SAVE_INTERVAL_STEPS = 100
EVAL_INTERVAL_STEPS = 50
LOG_INTERVAL_STEPS = 10

print("✓ Configuration loaded")
```

```python
import kagglehub
from tunix.models.gemma3 import model as gemma_lib
from tunix.models.gemma3 import params_safetensors as params_safetensors_lib
from tunix.generate import tokenizer_adapter as tokenizer_lib

print(f"Model handle: {KAGGLE_MODEL_HANDLE}")

local_model_path = kagglehub.model_download(KAGGLE_MODEL_HANDLE)
print(f"✓ Model downloaded to: {local_model_path}")

print(f"\nCreating TPU mesh with shape {MESH_SHAPE}...")
mesh = jax.make_mesh(MESH_SHAPE, ('fsdp', 'tp'))
print(f"✓ TPU Mesh created successfully")
print(f"  Mesh shape: {mesh.shape}")
print(f"  Mesh axis names: {mesh.axis_names}")
```

```python
model_config = gemma_lib.ModelConfig.gemma3_1b()

gemma3_model = params_safetensors_lib.create_model_from_safe_tensors(
    local_model_path,  # Directory containing .safetensors files
    model_config,
    mesh,
)
print("✓ Model loaded successfully")


tokenizer = tokenizer_lib.Tokenizer(
    tokenizer_path=f"{local_model_path}/tokenizer.model"
)
print("✓ Tokenizer loaded successfully")
```

```python
import flax.nnx as nnx


model_input = gemma3_model.get_model_input()

print("\nSharding model across TPU devices...")
with mesh:
    state = nnx.state(gemma3_model)
    pspecs = nnx.get_partition_spec(state)
    sharded_state = jax.lax.with_sharding_constraint(state, pspecs)
    nnx.update(gemma3_model, sharded_state)
    
    # Force materialization on TPU
    _ = jax.tree_util.tree_map(lambda x: x.shape if hasattr(x, 'shape') else x, state)
    


total_params = sum(p.size for p in jax.tree_util.tree_leaves(nnx.state(gemma3_model)))

print(f"\n✓ Model ready for full fine-tuning")
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {total_params:,}")


all_params = nnx.state(gemma3_model)
param_leaves = jax.tree_util.tree_leaves(all_params)
print(f"Number of parameters: {len(param_leaves)}")

if len(param_leaves) > 0:
    sample = param_leaves[0]
    print(f"Sample param shape: {sample.shape}")
    print(f"Sample param dtype: {sample.dtype}")
    
    # Check device placement
    if hasattr(sample, 'devices'):
        devices_set = sample.devices()
        print(f"Sample param devices: {list(devices_set)}")
        if len(devices_set) > 0:
            dev = list(devices_set)[0]
            device_kind = dev.device_kind
            print(f"Device kind: {device_kind}")
            if 'tpu' in device_kind.lower():
                print("✓✓✓ SUCCESS: Model parameters are on TPU!")
                print(f"✓✓✓ Confirmed: {device_kind} detected")
            else:
                print(f"❌❌❌ ERROR: Model parameters are on {device_kind}, NOT TPU!")
                print("Training will run on CPU and produce wrong results!")
    else:
        print("⚠️  Cannot determine device placement")
else:
    print("❌ NO parameters found!")
print("="*60)
```

```python
import re
from datasets import load_dataset
reasoning_start = "<reasoning>"
reasoning_end = "</reasoning>"
solution_start = "<answer>"
solution_end = "</answer>"

SYSTEM_PROMPT = """You are a careful math tutor. You MUST respond in a Dual‑Stream format.

Inside the <reasoning>...</reasoning> block, always structure your thoughts into four sections:
1. Plan: high‑level steps you will take to solve the problem.
2. Reasoning: detailed step‑by‑step execution.
3. Evidence: citations, calculations, or explicit checks that support the reasoning.
4. Sanity_check: a brief check that the final answer makes sense (magnitude, units, edge‑cases).

After the reasoning block, put ONLY the final numeric result inside <answer>...</answer>.
The final answer must appear exactly once in <answer>...</answer>.
"""



PROMPT_TEMPLATE = """<start_of_turn>user
{system_instruction}

{question}<end_of_turn>
<start_of_turn>model
"""


FULL_TEMPLATE = """<start_of_turn>user
{system_prompt}

{question}<end_of_turn>
<start_of_turn>model

{reasoning_start}
{reasoning}
{reasoning_end}

{solution_start}
{answer}
{solution_end}<end_of_turn>"""
```

```python
# Helper function to extract answer from GSM8K format
def extract_hash_answer(text):
    """Extract numerical answer after #### delimiter."""
    if "####" not in text:
        return None
    return text.split("####")[1].strip()

# Helper function to extract reasoning from GSM8K format
def extract_reasoning(text):
    """Extract reasoning (everything before #### delimiter)."""
    if "####" not in text:
        return text.strip()
    return text.split("####")[0].strip()

# Load GSM8K dataset
print("Loading GSM8K dataset...")
train_dataset = load_dataset("openai/gsm8k", "main", split="train")
test_dataset = load_dataset("openai/gsm8k", "main", split="test")
print(f"✓ Loaded {len(train_dataset)} training examples")
print(f"✓ Loaded {len(test_dataset)} test examples")


print("\nExample question:")
print(train_dataset[0]["question"])
print("\nExample answer:")
print(train_dataset[0]["answer"])
print("\nExtracted reasoning:")
print(extract_reasoning(train_dataset[0]["answer"]))
print("\nExtracted numerical answer:")
print(extract_hash_answer(train_dataset[0]["answer"]))
```

```python
from datasets import load_dataset
import re

reasoning_start = "<reasoning>"
reasoning_end = "</reasoning>"
solution_start = "<answer>"
solution_end = "</answer>"


# 1. Define the Cleaning Helper
def clean_gsm8k_content(text):
    """
    Removes GSM8K specific calculation annotations.
    Converts '<<10+5=15>>' to '(10+5=15)' or just removes them if preferred.
    """
    # Remove GSM8K-style '<<...>>' annotations
    text = re.sub(r"<<(.*?)>>", r"(\1)", text)
    # Normalize spacing
    text = re.sub(r"\s+", " ", text).strip()
    return text


def format_gsm8k_example(ex):
    # Raw fields
    question = ex["question"]
    raw_answer = ex["answer"]

    # Split GSM8K-style answer into reasoning and final numeric answer
    # Format is usually "... explanation ... #### 42"
    if "####" in raw_answer:
        reasoning_raw, answer_raw = raw_answer.split("####", 1)
        reasoning = clean_gsm8k_content(reasoning_raw.strip())
        answer = answer_raw.strip()
    else:
        reasoning = clean_gsm8k_content(raw_answer.strip())
        answer = raw_answer.strip()

    # 1. User Turn (Includes the strict instructions)
    text = f"<start_of_turn>user\n{SYSTEM_PROMPT}\n\n{question}<end_of_turn>\n"

    # 2. Model Turn (Structured DSA monologue + final answer)
    plan_section = (
        "Plan:\n"
        "- We will break the problem into smaller steps and solve them one by one.\n"
    )
    reasoning_section = f"Reasoning:\n{reasoning}\n"
    evidence_section = (
        "Evidence:\n"
        "- The calculations in the reasoning show each intermediate step explicitly.\n"
    )
    sanity_section = (
        "Sanity_check:\n"
        f"- The final answer {answer} should make sense given the quantities in the problem.\n"
    )

    text += "<start_of_turn>model\n"
    text += "<reasoning>\n"
    text += plan_section + "\n"
    text += reasoning_section + "\n"
    text += evidence_section + "\n"
    text += sanity_section
    text += "</reasoning>\n"
    text += "<answer>\n"
    text += f"{answer}\n"
    text += "</answer>"
    text += "<end_of_turn>"

    return {"text": text}

print("Refining dataset with CLEANING and structured DSA System Prompt...")
train_dataset = load_dataset("openai/gsm8k", "main")["train"]
test_dataset = load_dataset("openai/gsm8k", "main")["test"]

formatted_train = [format_gsm8k_example(ex) for ex in train_dataset]
formatted_test = [format_gsm8k_example(ex) for ex in test_dataset]


# Optionally augment with a small custom basic-math dataset (roots, percents, units).
# Expected CSV schema:
#   question,answer,reasoning,calc_expr
import os, csv

CUSTOM_BASIC_MATH_PATH = "dsa_basic_math_roots_percents_units.csv"
custom_formatted = []
if os.path.exists(CUSTOM_BASIC_MATH_PATH):
    print(f"Loading custom basic-math dataset from {CUSTOM_BASIC_MATH_PATH} ...")
    with open(CUSTOM_BASIC_MATH_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            q = row.get("question", "").strip()
            a = row.get("answer", "").strip()
            reasoning = row.get("reasoning", "").strip()
            calc_expr = row.get("calc_expr", "").strip()

            text = """<start_of_turn>user
{system_prompt}

{question}<end_of_turn>
<start_of_turn>model
""".format(system_prompt=SYSTEM_PROMPT, question=q)

            text += "\n<reasoning>\n"
            text += "Plan:\n- We will break the problem into smaller steps and solve them one by one.\n\n"
            text += "Reasoning:\n" + reasoning + "\n\n"

            evidence_lines = ["Evidence:"]
            if calc_expr:
                evidence_lines.append(f"- CALC: {calc_expr} = {a}")
            else:
                evidence_lines.append("- The calculations in the reasoning show each intermediate step explicitly.")
            text += "\n".join(evidence_lines) + "\n\n"

            text += "Sanity_check:\n"
            text += f"- The final answer {a} should make sense given the quantities in the problem.\n"
            text += "</reasoning>\n"
            text += "<answer>\n" + a + "\n</answer>"
            text += "<end_of_turn>"

            custom_formatted.append({"text": text})

    print(f"Loaded {len(custom_formatted)} custom basic-math examples.")
    formatted_train.extend(custom_formatted)
else:
    print("No custom basic-math CSV found; using GSM8K only.")
```

```python
print("-" * 60)
print(formatted_train[100]["text"])
print("-" * 60)
```

```python
import grain.python as grain
import numpy as np
from tunix.sft import metrics_logger as tmetrics
from tunix.sft.peft_trainer import TrainingInput
# Force metrics_logger to behave as if wandb is unavailable
tmetrics.wandb = None

def tokenize_function(example):
    full_text = example["text"]
    full_tokens = tokenizer.encode(full_text)
    
    
    prompt_text = full_text.split("<start_of_turn>model")[0] + "<start_of_turn>model\n"
    prompt_tokens = tokenizer.encode(prompt_text)
    prompt_len = len(prompt_tokens)

    # Padding/Truncation Logic
    if len(full_tokens) > MAX_SEQ_LENGTH:
        full_tokens = full_tokens[:MAX_SEQ_LENGTH]
    else:
        pad_token = tokenizer.pad_id() if hasattr(tokenizer, 'pad_id') else tokenizer.eos_id()
        full_tokens = full_tokens + [pad_token] * (MAX_SEQ_LENGTH - len(full_tokens))

    input_tokens = np.array(full_tokens, dtype=np.int32)
    
    # Create Mask
    loss_mask = np.zeros_like(input_tokens, dtype=np.float32)
    
    # Enable loss only for the response part (ignoring padding)
    seq_len = min(len(tokenizer.encode(full_text)), MAX_SEQ_LENGTH)
    if seq_len > prompt_len:
        loss_mask[prompt_len:seq_len] = 1.0

    return TrainingInput(input_tokens=input_tokens, input_mask=loss_mask)


# Create Grain datasets
train_grain = (
    grain.MapDataset.source(formatted_train)
    .map(tokenize_function)
    .shuffle(seed=42)
    .repeat(NUM_EPOCHS)
    .batch(batch_size=TRAIN_MICRO_BATCH_SIZE, drop_remainder=True)
)

eval_grain = (
    grain.MapDataset.source(formatted_test)
    .map(tokenize_function)
    .batch(batch_size=TRAIN_MICRO_BATCH_SIZE, drop_remainder=True)
)

print(f"✓ Train batches: {len(train_grain):,}")
print(f"✓ Eval batches: {len(eval_grain):,}")
```

```python
import optax

schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.0,
    peak_value=LEARNING_RATE,
    warmup_steps=WARMUP_STEPS,
    decay_steps=MAX_STEPS - WARMUP_STEPS,
    end_value=LEARNING_RATE * 0.1,
)

# Create optimizer chain
optimizer = optax.chain(
    optax.clip_by_global_norm(MAX_GRAD_NORM),
    optax.scale_by_adam(
        b1=ADAM_BETA1,
        b2=ADAM_BETA2,
        eps=ADAM_EPSILON,
    ),
    optax.add_decayed_weights(WEIGHT_DECAY),
    optax.scale_by_schedule(schedule),
    optax.scale(-1.0),  # Gradient descent
)

print("✓ Optimizer configured:")
print(f"  Learning rate: {LEARNING_RATE}")
print(f"  Warmup steps: {WARMUP_STEPS}")
print(f"  Total steps: {MAX_STEPS}")
print(f"  Weight decay: {WEIGHT_DECAY}")
print(f"  Max grad norm: {MAX_GRAD_NORM}")
```

```python
from tunix import PeftTrainer, TrainingConfig, MetricsLoggerOptions
import orbax.checkpoint as ocp
from tunix.sft import metrics_logger as tmetrics
tmetrics.wandb = None  # 👈 add this once


checkpointing_options = ocp.CheckpointManagerOptions(
    save_interval_steps=SAVE_INTERVAL_STEPS,
    max_to_keep=3,  # Keep last 3 checkpoints
)

training_config = TrainingConfig(
    max_steps=MAX_STEPS,
    eval_every_n_steps=EVAL_INTERVAL_STEPS,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    checkpoint_root_directory=CHECKPOINT_DIR,
    checkpointing_options=checkpointing_options,
    metrics_logging_options=None,  # ✅ disable W&B / monitoring
)


print("✓ Training configuration created")
print(f"  Max steps: {MAX_STEPS}")
print(f"  Micro batch size: {TRAIN_MICRO_BATCH_SIZE}")
print(f"  Gradient accumulation: {GRADIENT_ACCUMULATION_STEPS}")
print(f"  Effective batch size: {TRAIN_MICRO_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
print(f"  Eval interval: {EVAL_INTERVAL_STEPS}")
print(f"  Save interval: {SAVE_INTERVAL_STEPS}")

# Model input function
from tunix.sft import utils

def gen_model_input_fn(training_input):
    """Convert TrainingInput to model-compatible format."""
    pad_mask = training_input.input_tokens != 0
    positions = utils.build_positions_from_mask(pad_mask)
    attention_mask = utils.make_causal_attn_mask(pad_mask)
    
    return {
        'input_tokens': training_input.input_tokens,
        'input_mask': training_input.input_mask,
        'positions': positions,
        'attention_mask': attention_mask,
    }


trainer = PeftTrainer(
    model=gemma3_model,
    optimizer=optimizer,
    training_config=training_config,
)
trainer = trainer.with_gen_model_input_fn(gen_model_input_fn)

print("✓ Trainer ready for training")
print(f"  Model: Gemma 3 1B (Full Fine-Tuning)")
print(f"  Max steps: {MAX_STEPS}")
```

Click **RUN > Run current and after**

```python
import time

print("="*60)
print("Starting Full Fine-Tuning on TPU v5e-8")
print("="*60)
print(f"Max steps: {MAX_STEPS}")
print(f"Training examples: {len(formatted_train)}")
print(f"Eval examples: {len(formatted_test)}")
print(f"Batch size: {TRAIN_MICRO_BATCH_SIZE}")
print(f"Gradient accumulation: {GRADIENT_ACCUMULATION_STEPS}")
print("="*60)


all_params = nnx.state(gemma3_model)
param_leaves = jax.tree_util.tree_leaves(all_params)
if len(param_leaves) > 0:
    sample_param = param_leaves[0]
    if hasattr(sample_param, 'devices'):
        devices = sample_param.devices()
        if len(devices) > 0:
            device_kind = list(devices)[0].device_kind
            print(f"✓ Model parameters are on: {device_kind}")
            if 'tpu' not in device_kind.lower():
                print(f"⚠️  WARNING: Model params on {device_kind}, not TPU!")
                print(f"⚠️  Training will run on CPU and produce wrong results!")
            else:
                print(f"✓✓✓ CONFIRMED: Model is ready for TPU training!")
        else:
            print("⚠️  No devices found for model parameters")
    else:
        print("⚠️  Cannot check device placement")
else:
    print("⚠️  No model parameters found")
print("="*60)

print("\n" + "="*60)
print("IMPORTANT: First training step will take 2-5 minutes")
print("="*60)
print("JAX is compiling all functions (happens on CPU).")
print("After first step completes, TPU will be used and steps will be MUCH faster.")
print("You should see 'Compiling...' messages initially.")
print("="*60)

print("\nStarting training...")
start_time = time.time()


trainer.train(
    train_ds=train_grain,
    eval_ds=eval_grain,
)

end_time = time.time()
total_time = end_time - start_time

print("\n" + "="*60)
print("Training Completed!")
print("="*60)
print(f"Total training time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
print(f"Average time per step: {total_time/MAX_STEPS:.1f} seconds")
print(f"Checkpoints saved to: {CHECKPOINT_DIR}")
print("="*60)

print("\n" + "="*60)
print("POST-TRAINING: Verify TPU was used")
print("="*60)
print(f"Expected TPU time: 5-15 seconds per step after compilation")
print(f"Your average: {total_time/MAX_STEPS:.1f} seconds per step")
if total_time/MAX_STEPS < 1.0:
    print("❌ WARNING: Training ran on CPU, not TPU!")
    print("Results will be incorrect. Check that model is properly sharded.")
else:
    print("✓ Training timing looks correct for TPU usage!")
print("="*60)
```

```python
from tunix.generate import sampler as sampler_lib
import json
import os


cache_config = sampler_lib.CacheConfig(
    cache_size=MAX_SEQ_LENGTH + 512,
    num_layers=model_config.num_layers,
    num_kv_heads=model_config.num_kv_heads,
    head_dim=model_config.head_dim,
)


generation_sampler = sampler_lib.Sampler(
    transformer=gemma3_model,
    tokenizer=tokenizer,
    cache_config=cache_config,
)


def generate_inference_prompt(question):
    # Match the training exactly: Same System Prompt, No One-Shot needed anymore.
    text = f"<start_of_turn>user\n{SYSTEM_PROMPT}\n\n{question}<end_of_turn>\n"
    text += f"<start_of_turn>model\n<reasoning>\n" 
    return text
```

```python
# Test questions
test_questions = [
    "What is the square root of 144?",
    "If a shirt costs $25 and is on sale for 20% off, what is the sale price?",
    "A train travels 60 miles in 45 minutes. What is its speed in miles per hour?",
    "What is 15% of 200?",
]

print("="*60)
print("Testing Trained Model (Strict Format)")
print("="*60)

for i, question in enumerate(test_questions, 1):
    # 1. Generate the formatted prompt
    prompt = generate_inference_prompt(question)

    print(f"\n[Test {i}] Question: {question}")
    print("-" * 60)

    # 2. Run Generation
    sampler_output = generation_sampler(
        input_strings=[prompt],
        max_generation_steps=512,
        temperature=0.01,  # Near-greedy for math
        top_k=1,
    )

    # 3. Extract and Clean Response
    response = sampler_output.text[0]
    
    # Manual Stop: Cut off text if the model generates <end_of_turn>
    # This fixes the looping issue seen in Test 4
    if "<end_of_turn>" in response:
        response = response.split("<end_of_turn>")[0]

    print(f"Response:\n{response}")
    print("=" * 60)
```

```python
import collections
import time
import re
from tqdm.auto import tqdm


VOTE_SAMPLES = 1 

# Temperature must be > 0 to get diverse reasoning paths
# 0.6 is standard for Self-Consistency
TEMPERATURE = 0.7 

# Max tokens for the answer
MAX_GEN_STEPS = 512

print("\n" + "="*60)
print(f"Evaluating with Majority Voting (k={VOTE_SAMPLES})")
print("="*60)


def normalize_answer(answer_str):
    """Normalize answer string for comparison."""
    if answer_str is None:
        return None
    s = str(answer_str).strip().lower()
    s = s.replace('$', '').replace(',', '').replace('£', '').replace('€', '')
    if s.endswith('.'):
        s = s[:-1]
    return s

def extract_answer_robust(response):
    """
    Extracts answers using a cascade of patterns (XML -> Boxed -> Text).
    """
    # 1. Try <answer> tags
    xml_match = re.search(r"<answer>\s*(.*?)\s*</answer>", response, re.DOTALL)
    if xml_match:
        return xml_match.group(1)

    # 2. Try LaTeX \boxed{}
    boxed_match = re.search(r"\\boxed\{([^}]+)\}", response)
    if boxed_match:
        return boxed_match.group(1)

    # 3. Try "Final Answer" text patterns
    text_match = re.search(r"(?:final answer|answer is)[:\s]*([0-9\.]+)", response, re.IGNORECASE)
    if text_match:
        return text_match.group(1)

    # 4. Fallback: Last number
    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", response)
    if numbers:
        return numbers[-1]
    return None

def get_majority_vote(candidates):
    """Returns the most common answer from a list of candidates."""
    # Filter out None values
    valid_candidates = [c for c in candidates if c is not None]
    
    if not valid_candidates:
        return None
    
    # Count frequency
    counter = collections.Counter(valid_candidates)
    
    # Get the most common element ((value, count) tuple)
    most_common, count = counter.most_common(1)[0]
    return most_common


# Load dataset if not already loaded
if 'test_dataset' not in globals():
    from datasets import load_dataset
    test_dataset = load_dataset("openai/gsm8k", "main", split="test")

total_examples = len(test_dataset)
correct_count = 0
start_time = time.time()

# Store failures for analysis
failures = []

for idx in tqdm(range(total_examples), desc="Voting"):
    example = test_dataset[idx]
    question = example["question"]
    
    # Get Ground Truth
    ground_truth_raw = extract_hash_answer(example["answer"])
    ground_truth_norm = normalize_answer(ground_truth_raw)

    # Prepare Prompt
    prompt = generate_inference_prompt(question)
    
    # Create Batch: Replicate the prompt VOTE_SAMPLES times
    # This sends 8 identical prompts to the model at once
    batch_prompts = [prompt] * VOTE_SAMPLES

    try:
        # Generate samples in parallel
        sampler_output = generation_sampler(
            input_strings=batch_prompts,
            max_generation_steps=MAX_GEN_STEPS,
            temperature=TEMPERATURE,
            top_k=40, # Allow diversity for voting
        )
        
        # Extract answers from all samples
        candidates = []
        for response_text in sampler_output.text:
            # Cleanup stop tokens
            if "<end_of_turn>" in response_text:
                response_text = response_text.split("<end_of_turn>")[0]
            
            # Extract
            raw_ans = extract_answer_robust(response_text)
            norm_ans = normalize_answer(raw_ans)
            candidates.append(norm_ans)
            
        # Perform Majority Vote
        final_prediction = get_majority_vote(candidates)
        
        # Check Correctness
        is_correct = False
        if final_prediction is not None and ground_truth_norm is not None:
            try:
                is_correct = float(final_prediction) == float(ground_truth_norm)
            except ValueError:metrics_logger
                is_correct = final_prediction == ground_truth_norm
        
        if is_correct:
            correct_count += 1
        else:
            # Log failure for inspection
            failures.append({
                "q": question,
                "gt": ground_truth_norm,
                "pred": final_prediction,
                "candidates": candidates
            })

    except Exception as e:
        print(f"Error on example {idx}: {e}")

end_time = time.time()
total_time = end_time - start_time


print("\n" + "="*60)
print("MAJORITY VOTING RESULTS")
print("="*60)
print(f"Total Time: {total_time:.1f}s ({total_time/total_examples:.2f}s per question)")
print(f"Samples per Question: {VOTE_SAMPLES}")
print("-" * 60)
print(f"Final Accuracy: {correct_count}/{total_examples} ({100*correct_count/total_examples:.2f}%)")
print("="*60)

# Show a sample failure to see voting behavior
if failures:
    print("\nSample Failure (Voting Analysis):")
    f = failures[0]
    print(f"Question: {f['q'][:100]}...")
    print(f"Ground Truth: {f['gt']}")
    print(f"Voted Prediction: {f['pred']}")
    print(f"Vote Distribution: {f['candidates']}")
```

### Export SFT checkpoints as a zip & clean up

The SFT trainer writes a full Tunix checkpoint tree under `CHECKPOINT_DIR` and TensorBoard
logs under `TENSORBOARD_DIR`. To keep the number of files small and make it easy to download
the weights, we:

1. Zip **only** the SFT checkpoint tree into a single archive.
2. Remove the original checkpoint and TensorBoard directories (they can always be recreated by re‑running SFT).

> **Note** – This step assumes that SFT training has already run and produced at least one checkpoint.

```python
import os
import shutil

print("Zipping SFT checkpoints and cleaning up SFT artifacts...")

if "CHECKPOINT_DIR" not in globals():
    print("  ! CHECKPOINT_DIR not defined; did you run the SFT config cell?")
else:
    if os.path.isdir(CHECKPOINT_DIR):
        zip_base = "tunix_sft_dual_stream_gemma3_actor_ckpt"
        zip_path = shutil.make_archive(zip_base, "zip", CHECKPOINT_DIR)
        print(f"  ✓ Created SFT zip archive: {zip_path}")
       # shutil.rmtree(CHECKPOINT_DIR)
        #print("  ✓ Removed SFT checkpoint directory:", CHECKPOINT_DIR)
    else:
        print("  ! No SFT checkpoint dir found at:", CHECKPOINT_DIR)

#if "TENSORBOARD_DIR" in globals() and os.path.isdir(TENSORBOARD_DIR):
    #shutil.rmtree(TENSORBOARD_DIR)
    print("  ✓ Didn't Removed SFT TensorBoard log directory:", TENSORBOARD_DIR)

#print("SFT artifact cleanup complete.")
```

## Part 2 — GRPO with DSA‑CAST Rewards (Reinforcement Learning)

This section is your original **DSA‑CAST + Tunix GRPO notebook**, embedded after SFT.

At a high level, it does:

1. **Environment & data setup**
   - Logs in to Hugging Face (via Kaggle secret).
   - Ensures JAX + Tunix are installed on the TPU.
   - Loads GSM8K from TFDS or a Kaggle dataset into a rollout‑friendly format:
     - each example has a `prompts` field already formatted with the Dual‑Stream template
     - plus `question` and `answer` fields used by the reward functions.

2. **Reward design (DSA‑CAST)**
   - `reward_format_exact`: strict regex check for the full `<reasoning>...<answer>...` layout.
   - `reward_format_soft`: softer “tag hygiene” score that penalizes missing or repeated tags.
   - `reward_cast_math_and_completeness`: CAST‑style scoring of:
     - math accuracy,
     - solution completeness,
     - plus an extra format bonus.

3. **GRPO training loop**
   - Builds a Tunix `RLCluster` with:
     - an **actor model** (the policy we update) and
     - a **reference model** (kept frozen).
   - Uses `GRPOLearner` to:
     1. Sample `NUM_GENERATIONS` rollouts per prompt.
     2. Score those rollouts with the DSA‑CAST reward.
     3. Apply GRPO updates to the actor, keeping the reference fixed.

4. **Baseline & post‑GRPO evaluation**
   - Evaluate the base Gemma 3 1B‑IT model (pre‑GRPO) on GSM8K.
   - Evaluate the GRPO‑trained actor on the same test data.
   - Compare accuracy, “partial credit”, and format‑adherence metrics.

5. **Export & cleanup**
   - Zip the **best actor checkpoint** into a single file:
     - `tunix_dsa_cast_grpo_actor_ckpt.zip`
   - Remove the GRPO checkpoint tree to keep Kaggle’s output under its file limits.

# DSA-CAST + Tunix GRPO on Gemma3-1B (TPU, Kaggle)

This notebook:

1. Sets up **Gemma3-1B-IT** on a Kaggle TPU using **Tunix**.
2. Uses the `<reasoning> ... </reasoning>` and `<answer> ... </answer>` format for math problems (GSM8K-style).
3. Defines a **CAST-style reward** that strongly favors:
   - mathematical accuracy, and  
   - answer completeness & proper tagging.
4. Runs a **Tunix GRPO** reinforcement learning loop using that reward.
5. Saves the final **Tunix checkpoint (no safetensors export)** so it can be re-used in another notebook.

```python
# HF Hub login removed
#
# GRPO now reuses the Gemma 3 model and tokenizer loaded in the SFT section
# via kagglehub + Tunix. No Hugging Face access token or secrets are needed
# anywhere in this notebook.
```

```python
# (Intentionally left simple)
#
# This cell used to log in to Hugging Face with a hard‑coded token.
# We no longer do that — the model weights and tokenizer are loaded
# once in the SFT section via Kaggle assets and reused for GRPO.
pass
```

```python
# === Environment setup: JAX TPU + Tunix (no git+ installs) ===
import os

# Make sure JAX uses TPU and has full memory
os.environ.setdefault("JAX_PLATFORMS", "tpu,cpu")
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "1.00"

# JAX TPU build
!pip install -q "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

# Tunix from PyPI (recommended), plus other deps
!pip install -q google-tunix[prod] humanize datasets tensorflow_datasets kagglehub huggingface_hub

# If you *still* want Qwix-based LoRA, use the PyPI wheel instead of git:
# (no GitHub username prompt; it just pulls the published wheel)

print("Environment installs complete (no git+).")
```

```python
# === Imports & global configuration ===
import functools
import json
import re
import shutil
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import random
import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx
import optax
import humanize
import sympy as sp

# Dual-Stream tags
REASONING_START = "<reasoning>"
REASONING_END = "</reasoning>"
ANSWER_START = "<answer>"
ANSWER_END = "</answer>"

# Monologue section headings
PLAN_HEADING = "Plan:"
REASONING_HEADING = "Reasoning:"
EVIDENCE_HEADING = "Evidence:"
SANITY_HEADING = "Sanity_check:"

SYSTEM_PROMPT = f"""You are a careful math tutor. You MUST respond in a Dual‑Stream format.

Inside the {REASONING_START}...{REASONING_END} block, always structure your thoughts into four sections:

1. Plan: high‑level steps you will take to solve the problem.
2. Reasoning: detailed step‑by‑step execution.
3. Evidence: citations, calculations, or explicit checks that support the reasoning.
4. Sanity_check: a brief check that the final answer makes sense (magnitude, units, edge‑cases).

After the reasoning block, put ONLY the final numeric result inside {ANSWER_START}...{ANSWER_END}.
The final answer must appear exactly once in {ANSWER_START}...{ANSWER_END}.
""".strip()

TEMPLATE = """<start_of_turn>user
{system_prompt}

{question}<end_of_turn>
<start_of_turn>model
"""
```

```python
# === Hyperparameters ===

MODEL_ID = "google/gemma-3-1b-it"

TRAIN_DATA_DIR = "./data/gsm8k_train"
TEST_DATA_DIR = "./data/gsm8k_test"

NUM_TPUS = len(jax.devices())
if NUM_TPUS == 8:
    MESH_COUNTS = (1, 4)
elif NUM_TPUS == 1:
    MESH_COUNTS = (1, 1)
else:
    raise ValueError(f"Unsupported number of TPU devices: {NUM_TPUS}")

MESH = [MESH_COUNTS, ("fsdp", "tp")]

MAX_PROMPT_LENGTH = 256
TOTAL_GENERATION_STEPS = 384
TEMPERATURE = 0.9
TOP_P = 1.0
TOP_K = 50
NUM_GENERATIONS = 2
NUM_ITERATIONS = 1

TRAIN_MICRO_BATCH_SIZE = 1
NUM_BATCHES = 256
TRAIN_FRACTION = 0.9
NUM_EPOCHS = 1

MAX_STEPS = 1500  # DEBUG: cap GRPO training steps for a quick run

LEARNING_RATE = 3e-6
B1 = 0.9
B2 = 0.99
WEIGHT_DECAY = 0.1
WARMUP_STEPS = int(0.1 * MAX_STEPS)
MAX_GRAD_NORM = 0.1

CKPT_DIR = "/kaggle/working/grpo_ckpts"
SAVE_INTERVAL_STEPS = 200
MAX_TO_KEEP = 4

GENERATION_CONFIGS = {
    "greedy":   {"temperature": None, "top_k": 1,   "top_p": None},
    "standard": {"temperature": 0.7,  "top_k": 50,  "top_p": 0.95},
    "liberal":  {"temperature": 0.85, "top_k": 2000,"top_p": 1.0},
}

print("Hyperparameters set. MAX_STEPS =", MAX_STEPS)
```

```python
# === Data preprocessing: GSM8K via TFDS ===

import tensorflow_datasets as tfds

def extract_hash_answer(text: str) -> Optional[str]:
    if "####" not in text:
        return None
    return text.split("####", 1)[1].strip()

def _load_gsm8k_tfds(data_dir: str, split: str):
    import tensorflow_datasets.text.gsm8k
    return tfds.data_source(
        "gsm8k",
        split=split,
        data_dir=data_dir,
        builder_kwargs={"file_format": tfds.core.FileFormat.ARRAY_RECORD},
        download=True,
    )

def get_gsm8k_dataset(data_dir: str, split: str = "train") -> grain.MapDataset:
    os.makedirs(data_dir, exist_ok=True)
    ds = _load_gsm8k_tfds(data_dir, split)

    def _as_text(v):
        return v if isinstance(v, str) else v.decode("utf-8")

    dataset = (
        grain.MapDataset.source(ds)
        .shuffle(seed=42)
        .map(
            lambda x: {
                "prompts": TEMPLATE.format(
                    system_prompt=SYSTEM_PROMPT,
                    question=_as_text(x["question"]),
                ),
                "question": _as_text(x["question"]),
                "answer": extract_hash_answer(_as_text(x["answer"])),
            }
        )
    )
    return dataset

train_raw = get_gsm8k_dataset(TRAIN_DATA_DIR, split="train")
test_raw = get_gsm8k_dataset(TEST_DATA_DIR, split="test")

train_dataset = train_raw.batch(TRAIN_MICRO_BATCH_SIZE)[:NUM_BATCHES]

if TRAIN_FRACTION == 1.0:
    train_dataset = train_dataset.repeat(NUM_EPOCHS)
    val_dataset = None
else:
    cutoff = int(len(train_dataset) * TRAIN_FRACTION)
    train_dataset = train_dataset[:cutoff].repeat(NUM_EPOCHS)
    val_dataset = train_dataset[cutoff:].repeat(NUM_EPOCHS) if cutoff < len(train_dataset) else None

NUM_TEST_BATCHES = 64
test_dataset = test_raw.batch(TRAIN_MICRO_BATCH_SIZE)[:NUM_TEST_BATCHES]

print("Dataset sizes (batches):",
      len(train_dataset),
      0 if val_dataset is None else len(val_dataset),
      len(test_dataset))
```

```python
# === Utility: TPU memory usage ===
def show_hbm_usage():
    fmt = functools.partial(humanize.naturalsize, binary=True)
    for d in jax.local_devices():
        stats = d.memory_stats()
        used = stats["bytes_in_use"]
        limit = stats["bytes_limit"]
        print(f"Using {fmt(used)} / {fmt(limit)} ({used/limit:%}) on {d}")
```

```python
# === GRPO actor/reference setup using SFT model (no HF Hub) ===
#
# Instead of downloading Gemma3-1B-IT again from Hugging Face and logging in
# with a token, we REUSE the model that was trained during the SFT phase.
#
# - `gemma3_model` was created in the SFT section via:
#     params_safetensors_lib.create_model_from_safe_tensors(local_model_path, model_config, mesh)
#   and then fine-tuned with PeftTrainer.
# - `tokenizer` and `model_config` and `mesh` were also created in SFT.
#
# Here:
#   * `actor_model`  = the SFT‑trained Gemma3 model (trainable in GRPO).
#   * `reference_model` = a fresh, frozen copy of the base Gemma3-1B-IT weights.
#
# This gives you a clean SFT → GRPO pipeline with no Hugging Face Hub login
# and no hard-coded API keys.

# Make sure SFT has run
try:
    gemma3_model
    tokenizer
    model_config
    mesh
    local_model_path
except NameError as e:
    raise RuntimeError(
        "SFT section must be run before GRPO. "
        "Missing variable: {}".format(e)
    )

print("Reusing SFT-trained Gemma3 model as GRPO actor...")
actor_model = gemma3_model  # SFT fine-tuned weights

print("Loading frozen reference model from base Gemma3 checkpoint via Tunix...")
with mesh:
    reference_model = params_safetensors_lib.create_model_from_safe_tensors(
        local_model_path,  # same directory used in SFT
        model_config,
        mesh,
    )

# EOS tokens: reuse tokenizer EOS id
EOS_TOKENS = [tokenizer.eos_id()]
print("EOS token IDs:", EOS_TOKENS)
```

```python
# === CAST-style helpers ===

def extract_final_number(text: str) -> Optional[str]:
    if text is None:
        return None

    # Prefer numbers inside the <answer> ... </answer> block
    m = re.search(r"<answer>(.*?)</answer>", text, flags=re.IGNORECASE | re.DOTALL)
    segment = m.group(1) if m else text

    # Try GSM8K-style '#### 42' first
    m = re.search(r"####\s*([-+]?[0-9][0-9.,/]*)", segment)
    if m:
        return m.group(1).replace(",", "").strip()

    # Otherwise, grab the first reasonable-looking number
    nums = re.findall(r"[-+]?[0-9][0-9.,/]*", segment)
    if not nums:
        return None
    return nums[0].replace(",", "").strip()


def extract_calc_statements(text: str):
    """Extract CALC: expr = result statements from a completion.

    Returns a list of (expr_str, result_str).
    """
    if not text:
        return []

    calc_lines = []
    for line in text.splitlines():
        if "CALC:" not in line:
            continue
        # Expect patterns like 'CALC: expr = result'
        m = re.search(r"CALC:\s*(.*?)=(.*)", line)
        if not m:
            continue
        expr_str = m.group(1).strip()
        result_str = m.group(2).strip()
        if expr_str and result_str:
            calc_lines.append((expr_str, result_str))
    return calc_lines


def calc_consistency_score(text: str) -> float:
    """Score how consistent CALC: statements are, using sympy.

    - If there are no CALC statements, returns 0.0 (no signal).
    - Otherwise, returns (# correct equations) / (# equations), in [0, 1].
    """
    calcs = extract_calc_statements(text)
    if not calcs:
        return 0.0

    correct = 0
    total = 0
    for expr_str, result_str in calcs:
        try:
            expr = sp.sympify(expr_str)
            rhs = sp.sympify(result_str)
            diff = sp.simplify(expr - rhs)
            is_zero = bool(diff == 0)
            correct += 1 if is_zero else 0
            total += 1
        except Exception:
            # Parsing or evaluation failure counts as incorrect
            total += 1
            continue

    if total == 0:
        return 0.0
    return float(correct) / float(total)


def cast_style_scores(completions, answer):
    """Compute math accuracy, structural completeness, format bonus, and calc consistency.

    Returns four lists (all floats):
      - math_accs
      - completeness
      - format_bonus
      - calc_consistency
    """
    math_accs = []
    completeness = []
    format_bonus = []
    calc_consistency = []

    for text, gold in zip(completions, answer):
        t = text or ""

        # === Math accuracy ===
        m_acc = 0.0
        pred_str = extract_final_number(t)
        if pred_str is not None and gold is not None:
            try:
                pred_val = float(str(pred_str).strip())
                gold_val = float(str(gold).strip())
                if pred_val == gold_val:
                    m_acc = 1.0
                else:
                    if gold_val != 0:
                        ratio = pred_val / gold_val
                        if 0.9 <= ratio <= 1.1:
                            m_acc = 0.5
                        elif 0.8 <= ratio <= 1.2:
                            m_acc = 0.25
            except Exception:
                m_acc = 0.0

        # === Structural completeness ===
        has_tags = (
            REASONING_START in t
            and REASONING_END in t
            and ANSWER_START in t
            and ANSWER_END in t
        )

        # Section presence and order
        def _idx(h: str) -> int:
            return t.find(h)

        positions = {
            "plan": _idx(PLAN_HEADING),
            "reasoning": _idx(REASONING_HEADING),
            "evidence": _idx(EVIDENCE_HEADING),
            "sanity": _idx(SANITY_HEADING),
        }
        present = {k: (v != -1) for k, v in positions.items()}

        # Presence score: +1 for each present, -1 for each missing, normalized
        pres_raw = sum(1.0 if present[k] else -1.0 for k in positions.keys())
        pres_score = (pres_raw / 4.0 + 1.0) / 2.0  # roughly map into [0,1]

        # Order score: only if all present
        order_score = 0.0
        if all(present.values()):
            idxs = [
                positions["plan"],
                positions["reasoning"],
                positions["evidence"],
                positions["sanity"],
            ]
            if idxs == sorted(idxs):
                order_score = 1.0
            else:
                order_score = 0.0

        # Reasoning length: non-empty body inside <reasoning>...</reasoning>
        m_block = re.search(
            rf"{re.escape(REASONING_START)}(.*?){re.escape(REASONING_END)}",
            t,
            flags=re.DOTALL | re.IGNORECASE,
        )
        reasoning_body = m_block.group(1) if m_block else ""
        reasoning_len = len(reasoning_body.strip())
        if reasoning_len > 0:
            len_score = min(1.0, reasoning_len / 300.0)
        else:
            len_score = 0.0

        c_score = max(0.0, (pres_score + order_score + len_score) / 3.0)

        # === Format bonus ===
        if has_tags and all(present.values()) and order_score > 0:
            f_bonus = 1.0
        elif has_tags:
            f_bonus = 0.5
        else:
            f_bonus = 0.0

        # === Evidence calc consistency ===
        calc_score = calc_consistency_score(t)

        math_accs.append(float(m_acc))
        completeness.append(float(c_score))
        format_bonus.append(float(f_bonus))
        calc_consistency.append(float(calc_score))

    return math_accs, completeness, format_bonus, calc_consistency
```

### DSA‑CAST Reward Functions (What the RL Signal Is Measuring)

The next cell defines three core reward functions used by GRPO, all of which
are aware of the **Plan / Reasoning / Evidence / Sanity_check** structure
inside `<reasoning>...</reasoning>` as well as the outer `<answer>...</answer>` block.

1. **`reward_format_exact`**  
   - Uses a strict regular expression over the full completion.  
   - Gives a high reward when the output looks like:

     ```text
     <reasoning>
     Plan:
       ...

     Reasoning:
       ...

     Evidence:
       ...

     Sanity_check:
       ...
     </reasoning>
     <answer>
       ...single final scalar...
     </answer>
     ```

   - Any major deviation (missing tags, missing headings, wrong order, multiple answer blocks, etc.) receives 0.

2. **`reward_format_soft`**  
   - Provides a smoother shaping signal when the model is “on the way” to the desired format.  
   - It:
     - rewards the presence of `<reasoning>...</reasoning>` and `<answer>...</answer>` tags,
     - rewards each of the four headings when present,
     - adds extra reward when the headings appear in the correct order,
     - and penalizes missing or badly ordered structure.

3. **`reward_cast_math_and_completeness`**  
   - Calls `cast_style_scores`, which:
     - extracts the numeric answer from the `<answer> ... </answer>` block,
     - compares it to the GSM8K ground‑truth answer (with some tolerance),
     - and scores structural completeness based on:
       - presence and order of Plan / Reasoning / Evidence / Sanity_check,
       - and non‑trivial reasoning content inside `<reasoning>...</reasoning>`.
   - Then combines:
     - **math accuracy** (did we get the right number?),
     - **completeness** (did we actually solve the problem with meaningful structure?), and
     - **format bonus** (are we respecting Dual‑Stream tags and headings?)
     into a single scalar.

During GRPO, all three rewards are **added together** to produce a single
reward per sampled rollout. That reward is what drives the policy updates.

In practice, you can view DSA‑CAST here as a **grading rubric** for the DSA style:
the SFT stage teaches the model *how* to speak in that structure, and
DSA‑CAST + GRPO teaches it to speak **better, more consistently, and more correctly**
while keeping Plan / Reasoning / Evidence / Sanity_check intact.

```python
# === Reward functions for Tunix GRPO ===

# Strict overall format: <reasoning> (with sections) then <answer>, in order.
section_pattern = (
    rf"{re.escape(PLAN_HEADING)}.*?"
    rf"{re.escape(REASONING_HEADING)}.*?"
    rf"{re.escape(EVIDENCE_HEADING)}.*?"
    rf"{re.escape(SANITY_HEADING)}"
)

match_format = re.compile(
    rf"^[\s]{{0,}}"
    rf"{re.escape(REASONING_START)}.*?{section_pattern}.*?{re.escape(REASONING_END)}.*?"
    rf"{re.escape(ANSWER_START)}(.+?){re.escape(ANSWER_END)}"
    rf"[\s]{{0,}}$",
    flags=re.MULTILINE | re.DOTALL,
)

def reward_format_exact(prompts, completions, **kwargs):
    """High reward only when we see the full DSA structure and dual-stream tags.

    +3.0 if:
      - <reasoning>...</reasoning> and <answer>...</answer> are present in order, and
      - all four headings (Plan / Reasoning / Evidence / Sanity_check) appear in order inside <reasoning>.

    0.0 otherwise.
    """
    scores = []
    for resp in completions:
        ok = bool(match_format.search(resp or ""))
        scores.append(3.0 if ok else 0.0)
    return scores


def reward_format_soft(prompts, completions, **kwargs):
    """Softer shaping reward for partial formatting progress.

    Rewards:
      - presence of <reasoning>/<answer> tags,
      - presence of each heading,
      - correct ordering of the headings.

    Penalties when tags or headings are missing or badly ordered.
    """
    scores = []
    for resp in completions:
        t = resp or ""
        r = 0.0

        # Dual-stream tags
        has_reasoning = REASONING_START in t and REASONING_END in t
        has_answer = ANSWER_START in t and ANSWER_END in t
        r += 1.0 if has_reasoning else -1.0
        r += 1.0 if has_answer else -1.0

        # Heading presence
        def _idx(h: str) -> int:
            return t.find(h)

        positions = {
            "plan": _idx(PLAN_HEADING),
            "reasoning": _idx(REASONING_HEADING),
            "evidence": _idx(EVIDENCE_HEADING),
            "sanity": _idx(SANITY_HEADING),
        }
        present = {k: (v != -1) for k, v in positions.items()}
        for k, is_present in present.items():
            r += 0.75 if is_present else -0.75

        # Heading order
        if all(present.values()):
            idxs = [
                positions["plan"],
                positions["reasoning"],
                positions["evidence"],
                positions["sanity"],
            ]
            if idxs == sorted(idxs):
                r += 1.0
            else:
                r -= 1.0

        scores.append(r)
    return scores


def reward_cast_math_and_completeness(prompts, completions, answer, **kwargs):
    """CAST-style reward: math accuracy + structural completeness + format + calc consistency.

    The combination is:
      R = 3 * math_accuracy + 2 * completeness + 1 * format_bonus + 1 * calc_consistency
    where each term is in roughly [0, 1].
    """
    math_accs, completeness, fbonus, calc_consistency = cast_style_scores(completions, answer)
    scores = []
    for ma, c, fb, cc in zip(math_accs, completeness, fbonus, calc_consistency):
        scores.append(3.0 * ma + 2.0 * c + 1.0 * fb + 1.0 * cc)
    return scores

print("Reward functions defined.")
```

```python
# === Evaluation utilities ===

def build_sampler(policy_model, tokenizer, model_config):
    return sampler_lib.Sampler(
        transformer=policy_model,
        tokenizer=tokenizer,
        cache_config=sampler_lib.CacheConfig(
            cache_size=MAX_PROMPT_LENGTH + TOTAL_GENERATION_STEPS + 256,
            num_layers=model_config.num_layers,
            num_kv_heads=model_config.num_kv_heads,
            head_dim=model_config.head_dim,
        ),
    )


def generate_answers(questions, sampler, temperature=0.7, top_k=50, top_p=0.95, seed=None):
    if isinstance(questions, str):
        batch = [
            TEMPLATE.format(system_prompt=SYSTEM_PROMPT, question=questions),
        ]
    else:
        batch = [
            TEMPLATE.format(system_prompt=SYSTEM_PROMPT, question=q)
            for q in questions
        ]
    out = sampler(
        input_strings=batch,
        max_generation_steps=TOTAL_GENERATION_STEPS,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        echo=False,
        seed=seed,
        eos_tokens=EOS_TOKENS,
    )
    texts = out.text
    return texts[0] if isinstance(questions, str) else texts


def evaluate_dataset(dataset, sampler, num_passes=1):
    total = 0
    strict_correct = 0
    approx_correct = 0
    format_ok = 0

    for batch in dataset:
        questions = batch["question"]
        answers = batch["answer"]
        multiple_outputs = [[] for _ in range(len(questions))]

        for s in range(num_passes):
            responses = generate_answers(
                questions,
                sampler,
                temperature=GENERATION_CONFIGS["greedy"]["temperature"],
                top_k=GENERATION_CONFIGS["greedy"]["top_k"],
                top_p=GENERATION_CONFIGS["greedy"]["top_p"],
                seed=s,
            )
            for idx, resp in enumerate(responses):
                multiple_outputs[idx].append(resp)

        for q, a, resp_list in zip(questions, answers, multiple_outputs):
            is_correct = False
            is_approx = False
            has_format = False
            for resp in resp_list:
                if match_format.search(resp or "") is not None:
                    has_format = True
                guess = extract_final_number(resp or "")
                truth = extract_final_number(a or "")
                try:
                    if truth is not None and guess is not None:
                        g = float(guess)
                        t = float(truth)
                        if g == t:
                            is_correct = True
                        ratio = g / t if t != 0 else 0.0
                        if 0.9 <= ratio <= 1.1:
                            is_approx = True
                except Exception:
                    pass
                if is_correct and is_approx and has_format:
                    break

            total += 1
            if is_correct:
                strict_correct += 1
            if is_approx:
                approx_correct += 1
            if has_format:
                format_ok += 1

    acc = 100.0 * strict_correct / max(1, total)
    approx_acc = 100.0 * approx_correct / max(1, total)
    fmt_acc = 100.0 * format_ok / max(1, total)

    print(f"Total examples: {total}")
    print(f"Strict accuracy: {acc:.2f}%")
    print(f"Approx accuracy: {approx_acc:.2f}%")
    print(f"Format accuracy: {fmt_acc:.2f}%")
    return dict(
        total=total,
        strict_accuracy=acc,
        approx_accuracy=approx_acc,
        format_accuracy=fmt_acc,
    )
```

```python
# === Baseline evaluation before GRPO ===

baseline_sampler = build_sampler(actor_model, tokenizer, model_config)
print("Evaluating baseline policy on a small test subset...")
baseline_metrics = evaluate_dataset(test_dataset, baseline_sampler, num_passes=1)
baseline_metrics
```

```python
# === RLCluster, optimizer, and GRPOLearner setup ===

ckpt_options = ocp.CheckpointManagerOptions(
    save_interval_steps=SAVE_INTERVAL_STEPS,
    max_to_keep=MAX_TO_KEEP,
)

schedule = optax.schedules.warmup_cosine_decay_schedule(
    init_value=0.0,
    peak_value=LEARNING_RATE,
    warmup_steps=WARMUP_STEPS,
    decay_steps=MAX_STEPS,
    end_value=0.0,
)
optimizer = optax.adamw(
    learning_rate=schedule,
    b1=B1,
    b2=B2,
    weight_decay=WEIGHT_DECAY,
)
if MAX_GRAD_NORM is not None:
    optimizer = optax.chain(
        optax.clip_by_global_norm(MAX_GRAD_NORM),
        optimizer,
    )

cluster_config = rl_cluster_lib.ClusterConfig(
    role_to_mesh={
        rl_cluster_lib.Role.ACTOR: mesh,
        rl_cluster_lib.Role.REFERENCE: mesh,
        rl_cluster_lib.Role.ROLLOUT: mesh,
    },
    rollout_engine="vanilla",
    offload_to_cpu=False,
    training_config=rl_cluster_lib.RLTrainingConfig(
        actor_optimizer=optimizer,
        eval_every_n_steps=64,
        max_steps=MAX_STEPS,
        mini_batch_size=TRAIN_MICRO_BATCH_SIZE,
        train_micro_batch_size=TRAIN_MICRO_BATCH_SIZE,
        metrics_logging_options=None,
        checkpoint_root_directory=CKPT_DIR,
        checkpointing_options=ckpt_options,
    ),
    rollout_config=base_rollout.RolloutConfig(
        max_tokens_to_generate=TOTAL_GENERATION_STEPS,
        max_prompt_length=MAX_PROMPT_LENGTH,
        kv_cache_size=MAX_PROMPT_LENGTH + TOTAL_GENERATION_STEPS + 256,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        top_k=TOP_K,
        eos_tokens=EOS_TOKENS,
    ),
)

grpo_config = GRPOConfig(
    num_generations=NUM_GENERATIONS,
    num_iterations=NUM_ITERATIONS,
    beta=0.08,
    epsilon=0.2,
)

rl_cluster = rl_cluster_lib.RLCluster(
    actor=actor_model,
    reference=reference_model,
    tokenizer=tokenizer,
    cluster_config=cluster_config,
)

grpo_trainer = GRPOLearner(
    rl_cluster=rl_cluster,
    reward_fns=[
        reward_format_exact,
        reward_format_soft,
        reward_cast_math_and_completeness,
    ],
    grpo_config=grpo_config,
)

print("RLCluster and GRPOLearner ready.")
```

```python
# === Run GRPO training ===

with mesh:
    show_hbm_usage()
    grpo_trainer.train(train_dataset, val_dataset)

print("GRPO training complete.")
```

```python
# === Load final trained params & re-evaluate ===

trained_ckpt_path = os.path.join(
    CKPT_DIR, "actor", str(MAX_STEPS), "model_params"
)

finetuned_sampler = build_sampler(actor_model, tokenizer, model_config)
print("Evaluating finetuned policy on test subset...")
finetuned_metrics = evaluate_dataset(test_dataset, finetuned_sampler, num_passes=1)
finetuned_metrics
```

```python
# === Export final Tunix checkpoint as a single zip and clean up ===
import os
import shutil

# Tunix checkpoint root (matches what we used in the training config)
CKPT_DIR = "/kaggle/working/grpo_ckpts"

actor_root = os.path.join(CKPT_DIR, "actor")
if not os.path.exists(actor_root):
    raise FileNotFoundError(f"Actor checkpoint dir not found: {actor_root}")

# Find the most recent actor step directory (they're named by step number)
step_dirs = [
    d for d in os.listdir(actor_root)
    if os.path.isdir(os.path.join(actor_root, d)) and d.isdigit()
]
if not step_dirs:
    raise RuntimeError(f"No step subdirs found in {actor_root}")

best_step = max(step_dirs, key=lambda s: int(s))
actor_step_dir = os.path.join(actor_root, best_step)
print("Using actor checkpoint step:", best_step)
print("Directory:", actor_step_dir)

# 1) Zip just that actor step directory
zip_base = "tunix_dsa_cast_grpo_actor_ckpt"
zip_path = shutil.make_archive(zip_base, "zip", actor_step_dir)
print(f"\nCreated zip archive: {zip_path}")

# 2) Remove the full GRPO checkpoint tree to stay under Kaggle's file limit
if os.path.exists(CKPT_DIR):
    shutil.rmtree(CKPT_DIR)
    print(f"Removed training checkpoint directory: {CKPT_DIR}")

print("\nRemaining important artifact:")
print("  -", zip_path)
```