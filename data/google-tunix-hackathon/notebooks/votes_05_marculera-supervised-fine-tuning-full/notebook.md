# Supervised Fine Tuning Full

- **Author:** Lera
- **Votes:** 89
- **Ref:** marculera/supervised-fine-tuning-full
- **URL:** https://www.kaggle.com/code/marculera/supervised-fine-tuning-full
- **Last run:** 2025-11-29 00:24:44.517000

---

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
MESH_SHAPE = (8, 1) 
TRAIN_MICRO_BATCH_SIZE = 2 

GRADIENT_ACCUMULATION_STEPS = 4 

LEARNING_RATE = 2e-5 
WARMUP_STEPS = 50    
NUM_EPOCHS = 10       


MAX_STEPS = 117 * NUM_EPOCHS 


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

SYSTEM_PROMPT = (
    "Solve the math problem. "
    "You must STRICTLY follow this format:\n"
    "1. Enclose your step-by-step logic inside <reasoning>...</reasoning> tags.\n"
    "2. Enclose the final numerical result inside <answer>...</answer> tags."
)


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
    For SFT, replacing with parentheses is usually safer than deleting.
    """
    if text is None:
        return ""
    # Replace << and >> with parentheses to make it standard math text
    cleaned = text.replace("<<", "(").replace(">>", ")")
    return cleaned

# 2. Define the Formatter
def format_gsm8k_example(example):
    """
    Formats training data with strict system instructions and data cleaning.
    """
    question = example["question"]
    raw_answer = example["answer"]
    
    # Extract parts
    reasoning = extract_reasoning(raw_answer)
    answer = extract_hash_answer(raw_answer)
    
    # --- APPLY CLEANING HERE ---
    # We clean the reasoning part because that's where the <<...>> artifacts live.
    reasoning = clean_gsm8k_content(reasoning)
    
    # --- PROMPT CONSTRUCTION ---
    
    # 1. User Turn (Includes the strict instructions)
    text = f"<start_of_turn>user\n{SYSTEM_PROMPT}\n\n{question}<end_of_turn>\n"
    
    # 2. Model Turn (The expected strict output)
    text += f"<start_of_turn>model\n"
    text += f"<reasoning>\n{reasoning}\n</reasoning>\n"
    text += f"<answer>\n{answer}\n</answer>"
    text += f"<end_of_turn>"

    return {"text": text}

print("Refining dataset with CLEANING and STRICT System Prompt...")
formatted_train = [format_gsm8k_example(ex) for ex in train_dataset]
formatted_test = [format_gsm8k_example(ex) for ex in test_dataset]
```

```python
print("-" * 60)
print(formatted_train[100]["text"])
print("-" * 60)
```

```python
import grain.python as grain
import numpy as np
from tunix.sft.peft_trainer import TrainingInput

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
    metrics_logging_options=MetricsLoggerOptions(
        log_dir=TENSORBOARD_DIR,
        flush_every_n_steps=LOG_INTERVAL_STEPS
    ),
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
            except ValueError:
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