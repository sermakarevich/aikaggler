# Guide : Gemma 3 fine tuning with Tunix

- **Author:** Aman Anand
- **Votes:** 81
- **Ref:** yekahaaagayeham/guide-gemma-3-fine-tuning-with-tunix
- **URL:** https://www.kaggle.com/code/yekahaaagayeham/guide-gemma-3-fine-tuning-with-tunix
- **Last run:** 2025-12-13 15:00:50.843000

---

## Cell 0: Install Tunix and dependencies

Installs **Tunix** (Google’s JAX/TPU-first training and serving utilities) with the `prod` extras.  
This notebook assumes a Kaggle TPU runtime; installing inside the notebook guarantees the exact version (`0.1.3`) used when the notebook was authored.

**Notes**
- If you see dependency conflicts, restart the kernel after installation.
- Pinning the version helps reproducibility across Kaggle sessions.

```python
!pip install "google-tunix[prod]==0.1.3"
```

## Cell 1: TPU/JAX runtime sanity checks + environment flags

1. Imports JAX and prints a quick **device inventory** (backend, device kind, device list).  
2. Warns if you are not on TPU (important because Gemma 3 training is intended to run on TPU in this notebook).  
3. Sets several environment variables and JAX configs:
   - `XLA_FLAGS` and `LIBTPU_INIT_ARGS`: performance and async collective behavior.
   - `JAX_COMPILATION_CACHE_DIR`: speeds up repeated compiles.
   - `jax_enable_x64=False`: keeps computation in 32-bit (typically BF16/FP32 mix) for speed/memory.
   - `jax_default_matmul_precision='high'`: improves numerical stability for matmuls.

**Pitfall**
- If `jax.default_backend()` is not `tpu`, training will be extremely slow and results will not match the intended setup.

```python
import jax
import jax.numpy as jnp
import os
import warnings; 
warnings.filterwarnings('ignore')

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

## Cell 2: Experiment configuration (model, batching, training hyperparameters, output paths)

Defines the main knobs for fine-tuning:

- **Model handle** (`KAGGLE_MODEL_HANDLE`): points to Gemma 3 weights hosted on Kaggle.
- **Sequence length** (`MAX_SEQ_LENGTH`): max tokens per example; impacts memory and speed.
- **TPU mesh** (`MESH_SHAPE`): logical device mesh for sharding (FSDP axis and tensor-parallel axis).
- **Micro-batch size** + **gradient accumulation**: together determine the **effective global batch size**.
- **Optimizer hyperparams**: learning rate, warmup, weight decay, grad clipping, epochs/steps.
- **Checkpoint/TensorBoard dirs** and logging cadence.

The printed “Global Batch Size” helps confirm your true effective batch:
`micro_batch * num_devices * grad_accumulation`.

**Note**
- The `...` line in this cell is a placeholder in the original notebook source. If you run the notebook as-is, ensure all required constants (e.g., Adam betas/epsilon if referenced later) are defined somewhere.

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

## Cell 3: Download Gemma 3 from Kaggle and create a TPU device mesh

- Uses `kagglehub.model_download()` to fetch the model assets locally.
- Builds a JAX mesh (`jax.make_mesh`) with axes `('fsdp', 'tp')` using `MESH_SHAPE`.

This mesh is later used to:
- **Shard parameters** across devices (FSDP-style parameter sharding).
- Optionally use a tensor-parallel axis (depending on model/implementation).

**Why this matters**
Without a mesh context, the model can silently remain on CPU, making training incorrect/slow.

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

## Cell 4: Load model weights (.safetensors) and tokenizer

- Creates a Gemma 3 1B model config.
- Loads model parameters from the downloaded `.safetensors` files into a JAX/Flax model, sharded according to the TPU mesh.
- Loads the SentencePiece tokenizer (`tokenizer.model`) matching the base checkpoint.

**Key idea**
Tokenizer and model weights must match; mixing tokenizers across checkpoints can corrupt training and evaluation.

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

## Cell 5: Force model parameter sharding onto TPU and verify placement

- Uses `flax.nnx` utilities to:
  - extract model state (`nnx.state`)
  - compute partition specs (`nnx.get_partition_spec`)
  - apply sharding constraints (`jax.lax.with_sharding_constraint`)
  - update the model with the sharded state (`nnx.update`)
- “Materializes” shapes to force device placement.
- Then inspects a sample parameter to confirm it resides on TPU devices.

**Why this exists**
In JAX it is possible to construct objects on host/CPU and only later place them on device. This explicit sharding/verification prevents a common failure mode: “training runs but on CPU”.

**Note about `...`**
The `...` line is a placeholder from the original notebook and is not executable Python. If this notebook errors at runtime, remove/replace those placeholders.

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

## Cell 6: Build an inference sampler (generation) + prompt constructor

- Configures the KV cache (`CacheConfig`) for autoregressive generation.
- Instantiates `sampler_lib.Sampler` with the model and tokenizer.
- Defines `generate_inference_prompt(question)` which formats the input exactly like training:
  - `<start_of_turn>user` + system instructions + question
  - `<start_of_turn>model` + opens `<reasoning>` tag (the model is expected to continue)

**Why it matters**
Evaluation should mirror training formatting to get an apples-to-apples baseline and post-training comparison.

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

## Cell 7: Define the strict instruction format and templates

Sets up:
- XML-style tags used in training/eval:
  - `<reasoning>...</reasoning>`
  - `<answer>...</answer>`
- A **SYSTEM_PROMPT** that forces the model to follow the schema.
- Prompt templates showing how a full supervised example is constructed.

**Goal**
This is “schema SFT”: you teach the model not just to solve problems, but to consistently produce machine-parseable outputs.

**Note**
`PROMPT_TEMPLATE` contains a `...` placeholder in the notebook. Replace it with a concrete template if you intend to use it directly.

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

## Cell 8: Load evaluation questions + define answer extraction and scoring

- Attempts to load a CSV of questions and gold answers (`updated_200_math_questions.csv`).
  - Falls back to a hard-coded list if the CSV is missing.
- Defines utilities to:
  - extract a final answer from the model output (prefer `<answer>`, then GSM8K `####`, then last numeric token)
  - normalize answers (strip commas/currency, normalize whitespace/case)
  - compare predictions to gold answers, including handling cases like `"x or y"`.

**Why this is important**
LLM outputs are messy. Robust evaluation requires:
1) deterministic parsing rules, and  
2) normalization to avoid false negatives from formatting differences.

```python
import re
import pandas as pd

# -----------------------------
# 1) Load questions
# -----------------------------
# Option A: Evaluate from CSV (recommended for your 200 questions)
CSV_PATH = "/kaggle/input/maths-sft-training-dataset/updated_200_math_questions.csv"   # adjust if needed

try:
    dfq = pd.read_csv(CSV_PATH)
    questions = dfq["question"].tolist()
    golds = dfq["gold_answer"].astype(str).tolist()
    source = f"CSV: {CSV_PATH} ({len(dfq)} rows)"
except Exception as e:
    # Option B: fallback to your manual list
    test_questions = [
        "What is the square root of 144?",
        "If a shirt costs $25 and is on sale for 20% off, what is the sale price?",
        "A train travels 60 miles in 45 minutes. What is its speed in miles per hour?",
        "What is 15% of 200?",
        "A product is marked up by 25% and then discounted by 20%. The final price is ₹960. What was the original price?",
        "A car travels at 60 km/h for 30 minutes, stops for 10 minutes, then travels at 40 km/h for another 30 minutes. What is the car’s average speed for the entire journey?",
        "What is ⅔ of ¾ of 120, minus 25% of the result?",
        "The ratio of apples to oranges in a basket is 3:5. If 16 oranges are removed and the new ratio becomes 3:1,how many apples were originally in the basket?",
        "A pipe fills a tank in 40 minutes, while another pipe empties the same tank in 60 minutes. If both pipes are opened together, how long will it take to fill the tank?",
        "A number increases by 10% and then decreases by 10%. Is the final number greater than, less than, or equal to the original? Explain why.",
    ]
    questions = test_questions
    golds = [None] * len(questions)  # no golds in this path
    source = f"Manual list ({len(questions)} questions)"
    print("CSV load failed, using manual list. Error:", e)

print("Evaluating:", source)


# -----------------------------
# 2) Helpers: normalize + extract answers
# -----------------------------
def normalize_text(s: str) -> str:
    s = str(s).strip().lower()
    # normalize unicode fractions (⅔ etc.) if they appear in answers (rare)
    s = (s.replace("½", "1/2")
           .replace("⅓", "1/3").replace("⅔", "2/3")
           .replace("¼", "1/4").replace("¾", "3/4")
           .replace("⅕", "1/5").replace("⅖", "2/5").replace("⅗", "3/5").replace("⅘", "4/5")
           .replace("⅙", "1/6").replace("⅚", "5/6")
           .replace("⅛", "1/8").replace("⅜", "3/8").replace("⅝", "5/8").replace("⅞", "7/8"))
    # remove spaces around common separators
    s = re.sub(r"\s+", " ", s)
    # remove currency symbols but keep numbers/units
    s = s.replace("₹", "").replace("$", "")
    # remove commas in numbers: 62,500 -> 62500
    s = re.sub(r"(?<=\d),(?=\d)", "", s)
    # trim punctuation at ends
    s = s.strip(" .,:;!?\n\t")
    return s

def extract_final_from_response(response: str) -> str:
    """
    Tries in this order:
    1) <answer>...</answer>
    2) line starting with #### (gsm8k)
    3) last numeric/fraction token in response
    4) fallback: last non-empty line
    """
    if response is None:
        return ""

    text = str(response)

    # cut off runaway turns if present
    if "<end_of_turn>" in text:
        text = text.split("<end_of_turn>")[0]

    # 1) <answer> tag
    m = re.search(r"<answer>\s*(.*?)\s*</answer>", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()

    # 2) GSM8K #### final
    m = re.search(r"####\s*(.+)", text)
    if m:
        return m.group(1).strip()

    # 3) last fraction or number (keeps % too)
    tokens = re.findall(r"-?\d+(?:\.\d+)?(?:/\d+(?:\.\d+)?)?%?", text)
    if tokens:
        return tokens[-1].strip()

    # 4) fallback: last non-empty line
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines[-1] if lines else text.strip()

def gold_to_accept_set(gold: str):
    """
    Handles cases like '3 or 8' by allowing multiple correct answers.
    """
    if gold is None:
        return set()

    g = normalize_text(gold)

    # allow 'x or y' answers
    if " or " in g:
        parts = [p.strip() for p in g.split(" or ") if p.strip()]
        return set(parts)

    return {g}

def is_correct(model_final: str, gold: str) -> bool:
    mf = normalize_text(model_final)
    accept = gold_to_accept_set(gold)
    if not accept:
        return False  # if no gold, can't score
    return mf in accept
```

## Cell 9: Run baseline inference and log per-question results

Loops over `(question, gold)` pairs and:
- Builds the prompt with `generate_inference_prompt`.
- Calls `generation_sampler` with near-deterministic decoding (`temperature=0.01`, `top_k=1`).
- Extracts the final answer and checks correctness.
- Stores a rich record per example:
  - prompt, raw response, parsed answer, gold answer, correctness flag

Outputs `df_res.sample(4)` for a quick spot check.

**Tip**
If you want more diverse reasoning, raise temperature (but that makes scoring noisier unless you use voting/self-consistency).

```python
%%time
# -----------------------------
# 3) Run evaluation
# -----------------------------
results = []

for i, (q, gold) in enumerate(zip(questions, golds), 1):
    prompt = generate_inference_prompt(q)

    out = generation_sampler(
        input_strings=[prompt],
        max_generation_steps=256,
        temperature=0.01,
        top_k=1,
    )

    response_raw = out.text[0]
    model_final = extract_final_from_response(response_raw)

    correct = None
    if gold is not None:
        correct = is_correct(model_final, gold)

    results.append({
        "idx": i,
        "question": q,
        "gold_answer": gold,
        "prompt": prompt,
        "model_final_answer": model_final,
        "model_raw_response": response_raw,
        "is_correct": correct
    })

df_res = pd.DataFrame(results)
df_res.sample(4)
```

## Cell 10: Summarize baseline accuracy and surface failures

Creates:
- a one-row summary table (total, correct, wrong, accuracy)
- `wrong_df`: a failure report including `prompt` and full `model_raw_response`

**Why this is useful**
When doing SFT, the fastest quality loop is:
1) inspect failure modes,  
2) adjust formatting/training data,  
3) re-train,  
4) re-evaluate with the same harness.

```python
%%time
# -----------------------------
# 4) Summary tables
# -----------------------------
if df_res["is_correct"].notna().any():
    total = df_res["is_correct"].notna().sum()
    correct_n = int((df_res["is_correct"] == True).sum())
    wrong_n = int((df_res["is_correct"] == False).sum())
    acc = correct_n / total if total else 0.0

    summary = pd.DataFrame([{
        "total_scored": total,
        "correct": correct_n,
        "wrong": wrong_n,
        "accuracy_%": round(acc * 100, 2),
    }])
    display(summary)

    wrong_df = df_res[df_res["is_correct"] == False][
        ["idx", "question", "gold_answer", "model_final_answer", "prompt", "model_raw_response"]
    ].reset_index(drop=True)

    display(wrong_df)
else:
    print("No gold answers were loaded, so scoring is skipped.")
    display(df_res[["idx", "question", "prompt", "model_final_answer", "model_raw_response"]])
```

## Cell 11: Quick look at wrong predictions

Displays `wrong_df.head()` so you can immediately inspect the first few mistakes with:
- the question
- the expected answer
- the model’s parsed final answer
- the full prompt and raw completion

```python
wrong_df.head()
```

# Pre- Fine tuning model process

## Cell 12: GSM8K answer extraction helper (`#### ...`)

Defines a helper to extract the final numeric answer from GSM8K examples, which commonly use the pattern:

`... #### 42`

**Why it matters**
You need a reliable way to obtain the “gold” final answer so you can build supervised `<answer>...</answer>` targets for SFT.

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

## Cell 13: Load GSM8K and format into strict SFT training examples

- Loads GSM8K train/test splits via `datasets.load_dataset`.
- Defines `clean_gsm8k_content()` to remove/normalize GSM8K-specific artifacts like `<<10+5=15>>`.
- Defines `format_gsm8k_example(ex)` to build a single training string in Gemma chat format:
  - user turn: system prompt + question
  - model turn: `<reasoning> cleaned reasoning </reasoning>` + `<answer> extracted answer </answer>`

Produces:
- `formatted_train`: list of dicts with `{"text": ...}`
- `formatted_test`: same for evaluation

**Why this works**
SFT teaches the model to imitate the “ideal completion” for the exact prompt you will use at inference time.

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

## Cell 14: Print a sample formatted example

Prints one formatted training example so you can validate:
- the chat markers (`<start_of_turn>...`)
- the system prompt presence
- reasoning and answer tags
- absence of GSM8K artifacts

This is a critical sanity check before launching a TPU training run.

```python
print("-" * 60)
print(formatted_train[100]["text"])
print("-" * 60)
```

## Cell 15: Tokenization + Grain input pipelines (train/eval)

- Defines `tokenize_function(example)` which:
  - tokenizes the full supervised text
  - separately tokenizes the **prompt prefix** up to `<start_of_turn>model\n`
  - builds masks so loss is applied primarily to the model completion portion (common SFT practice)
  - pads/truncates to `MAX_SEQ_LENGTH`
  - wraps everything into Tunix `TrainingInput`
- Builds Grain datasets:
  - shuffle + repeat for training
  - batch for train and eval

**Why Grain**
Grain is optimized for JAX input pipelines and plays well with TPU training.

**Note**
This cell contains `...` placeholders in the notebook source. Ensure the mask/padding logic is complete and executable before running.

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

## Cell 16: Learning-rate schedule and optimizer (Optax)

- Builds a warmup + cosine decay LR schedule.
- Creates an optimizer chain:
  1) global norm clipping (stability)
  2) Adam moments
  3) weight decay (regularization)
  4) scheduled LR scaling
  5) negative scale to perform gradient descent

Prints the final optimizer settings for auditability.

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

# training part starts

## Cell 17: Trainer configuration (checkpoints, logging, model input fn)

- Configures Orbax checkpoint manager:
  - save cadence
  - retention policy (`max_to_keep`)
- Builds a `TrainingConfig`:
  - total steps, eval cadence, gradient accumulation, checkpoint/log directories
  - TensorBoard metric logging
- Defines `gen_model_input_fn(training_input)`:
  - builds `positions` and causal `attention_mask` from non-padding tokens
  - returns the dict expected by the Gemma model forward pass
- Instantiates `PeftTrainer` and attaches the input adapter.

**Important nuance**
Despite the class name `PeftTrainer`, this notebook’s print statements suggest full fine-tuning. Confirm whether PEFT adapters are actually enabled; otherwise this is “full-parameter” training.

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

## Cell 18: Launch training and verify TPU usage via timing

- Prints run metadata (steps, dataset sizes, batch and accumulation).
- Re-checks parameter device placement (TPU vs CPU).
- Calls `trainer.train(train_ds=..., eval_ds=...)`.
- Reports total time, time per step, and checkpoint location.
- Includes a heuristic “TPU vs CPU” check based on average step time after compilation.

**Caveat**
The timing heuristic is rough; the first few steps include XLA compilation. For a more reliable check, confirm device placement and look at TPU utilization in the runtime.

```python
### training the models

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

## Cell 19: Rebuild the generation sampler for the fine-tuned model

After training, you typically re-instantiate:
- cache config
- `Sampler`

This ensures generation uses the updated in-memory weights and a fresh cache, then you can re-run evaluation with the same parsing/scoring code.

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

# Eval

## Cell 20: Define test questions for quick smoke testing

Creates a small list of arithmetic/word problems to validate:
- the model follows the `<reasoning>` and `<answer>` format
- the fine-tuned model improved on the types of questions you care about

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

## Cell 21: Evaluation loop with (optional) self-consistency voting

Implements a more robust evaluation strategy:

- Generate `VOTE_SAMPLES` completions per question (with `TEMPERATURE > 0`).
- Parse each completion into a final answer candidate.
- Use majority vote (`collections.Counter`) to pick the most frequent answer.
- Compute final accuracy and log failures with candidate distributions.

**Why self-consistency helps**
Many math problems have multiple valid reasoning paths; sampling increases the chance you get at least one correct path, and voting reduces variance.

**Tradeoff**
Higher `VOTE_SAMPLES` improves accuracy but increases inference cost linearly.

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

# Thank you

Still, a lot of training is required on this model. This was a basic training strategy