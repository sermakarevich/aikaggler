# Teach gemma 3 LLM to reason- GPU

- **Author:** JP AI
- **Votes:** 48
- **Ref:** jayaprakashmurugan/teach-gemma-3-llm-to-reason-gpu
- **URL:** https://www.kaggle.com/code/jayaprakashmurugan/teach-gemma-3-llm-to-reason-gpu
- **Last run:** 2026-02-02 16:23:26.300000

---

# **Tunix + Gemma 3** architecture (NNX) used in the reference, effectively fixing the `ModuleNotFoundError` and `CuDNN` issues.

**Key Changes:**

1. **Environment:** Uses the robust installation script from the reference to fix JAX/TPU/CuDNN conflicts.
2. **Model:** Switches to **Gemma 3 1B IT** (Instruction Tuned). This replaces the "Cold Start" SFT phase since the model is already fine-tuned for instructions.
3. **Pipeline:** Implements the **GRPO (RL)** loop using `tunix.rl.grpo` and `qwix` for LoRA, exactly matching the library's requirements.

### **Instructions**

1. Copy the code blocks below into corresponding cells in a **new Kaggle Notebook**.
2. **Hardware:** Select **TPU VM v3-8** (recommended) or **GPU T4 x2**.
3. **Important:** After running **Cell 1**, you **MUST restart the kernel** before running Cell 2.

---

### **Cell 1: Environment Setup (Run & Restart)**

*Note: This installs the correct JAX/Tunix versions. After this runs, click "Restart Session".*

```python
# --- GPU INSTALLATION ---
import os
import sys

print("Installing GPU-compatible packages...")

# 1. Clean Slate: Remove any TPU/CPU versions of JAX
!pip uninstall -y jax jaxlib flax tunix qwix libtpu-nightly

# 2. Install JAX for CUDA 12 (Official Release)
!pip install -U "jax[cuda12]"

# 3. Install Tunix Ecosystem (From Source)
!pip install git+https://github.com/google/tunix.git
!pip install git+https://github.com/google/qwix.git
!pip install git+https://github.com/google/flax.git

# 4. Install Dependencies
!pip install -q "numpy>2" tensorflow tensorflow_datasets tensorboardX transformers grain huggingface_hub datasets

print("\n" + "="*50)
print("⚠️  IMPORTANT: RESTART THE KERNEL NOW!  ⚠️")
print("Go to 'Run' > 'Restart Kernel' (or 'Runtime' > 'Restart Session')")
print("Then verify GPU detection in the next cell.")
print("="*50 + "\n")
```

# 🔬 Technical Appendix: GRPO & Socratic Reasoning Dynamics

## 1. Mathematical Framework: Group Relative Policy Optimization (GRPO)
This notebook implements **GRPO**, a memory-efficient variant of Proximal Policy Optimization (PPO) designed for reasoning tasks. Unlike PPO, GRPO eliminates the need for a separate value network (Critic) $V_\phi(x)$, significantly reducing GPU memory overhead.

### The Objective Function
The GRPO objective maximizes the expected reward while constraining the policy shift via KL-divergence. For each prompt $q$, we sample a group of outputs $\{o_1, o_2, ..., o_G\}$ from the old policy $\pi_{\theta_{old}}$.

The advantage $A_i$ for the $i$-th output is calculated relative to the group's average reward, serving as a dynamic, sample-dependent baseline:

$$A_i = \frac{r_i - \text{mean}(\{r_1, ..., r_G\})}{\text{std}(\{r_1, ..., r_G\}) + \epsilon}$$

The policy gradient update is formalized as:

$$J(\theta) = \mathbb{E}_{q \sim P(Q), \{o_i\} \sim \pi_{\theta_{old}}} \left[ \frac{1}{G} \sum_{i=1}^G \min \left( \frac{\pi_\theta(o_i|q)}{\pi_{\theta_{old}}(o_i|q)} A_i, \text{clip}\left( \frac{\pi_\theta(o_i|q)}{\pi_{\theta_{old}}(o_i|q)}, 1-\epsilon, 1+\epsilon \right) A_i \right) - \beta D_{KL}(\pi_\theta || \pi_{ref}) \right]$$

**Engineering Constraints:**
* **$\epsilon$ (Clip Param):** Limits the update step size (typically 0.2) to prevent catastrophic forgetting.
* **$\beta$ (KL Coeff):** Controls the drift from the reference model $\pi_{ref}$ (the SFT checkpoint) to prevent reward hacking.

## 2. Socratic Chain-of-Thought (CoT) Formulation
The model is trained on the `gsm8k-train_socratic` dataset, transforming the objective from simple QA ($x \to y$) to latent reasoning generation ($x \to z \to y$).

* **Input ($x$):** Mathematical problem statement.
* **Reasoning Trace ($z$):** The step-by-step intermediate logical deductions (the "Socratic" dialogue).
* **Output ($y$):** The final numerical answer.

The optimization maximizes $P(y|x)$ by implicitly maximizing the joint probability $P(y|z, x)P(z|x)$. The reward function $R(x, y)$ is sparse and deterministic ($1$ if correct, $0$ otherwise), forcing the model to align $z$ with valid logical paths that lead to $y$.

## 3. Training Dynamics & Metrics
The following metrics are monitored to ensure convergence stability:

* **`actor_train_kl`:** Measures $\mathbb{E}[D_{KL}(\pi_\theta || \pi_{ref})]$. A spike $> 0.1$ indicates the policy is diverging too far from natural language distribution (reward hacking).
* **`actor_train_perplexity`:** A perplexity near $1.0$ indicates high confidence. In reasoning tasks, extremely low perplexity combined with low reward suggests "mode collapse" (repeating the same incorrect reasoning).

## 4. Parameter Efficiency: QLoRA Architecture
To enable reasoning updates on consumer GPUs, we employ Quantized Low-Rank Adaptation (QLoRA):

* **Frozen Weights ($W$):** The base Gemma 3 parameters are quantized to 4-bit (NF4 format).
* **Trainable Adapters ($A, B$):** Rank-$r$ matrices injected into Linear layers.
* **Forward Pass:** $h = Wx + BAx$
* **Target Modules:** We specifically target `q_proj`, `k_proj`, `v_proj`, `o_proj`, and crucially `gate_proj`, `up_proj`, `down_proj` (MLP layers), as recent research indicates reasoning capabilities are heavily localized in MLP blocks.

```python
from IPython.display import Image, display

image_url = "https://substackcdn.com/image/fetch/$s_!GSkD!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F3a0a064f-0d82-4beb-9267-b37059b658eb_1244x662.png"
display(Image(url=image_url, width=800))
```

---

### **Cell 2: Authentication & Imports**

*Run this after restarting the kernel.*

```python
import os
# Force JAX to use GPU (optional safety measure)
os.environ["JAX_PLATFORMS"] = "cuda"

import jax
import jax.numpy as jnp
from flax import nnx
import qwix
from huggingface_hub import login, snapshot_download

# --- AUTHENTICATION ---
from kaggle_secrets import UserSecretsClient
try:
    user_secrets = UserSecretsClient()
    HF_TOKEN = user_secrets.get_secret("HF_TOKEN")
    login(token=HF_TOKEN)
except Exception:
    print("⚠️ HF_TOKEN not found in secrets. Please login manually if needed.")
    login()

print(f"\n✅ JAX Version: {jax.__version__}")
try:
    devices = jax.devices()
    print(f"✅ Detected Devices: {devices}")

except Exception as e:
    print(f"❌ GPU Error: {e}")

# Tunix Imports
from tunix.models.gemma3 import model as gemma_lib
from tunix.models.gemma3 import params_safetensors as params_safetensors_lib
from tunix.generate import sampler as sampler_lib
from tunix.generate import tokenizer_adapter as tokenizer_lib
from tunix.rl import rl_cluster as rl_cluster_lib
from tunix.rl.grpo.grpo_learner import GRPOConfig, GRPOLearner
from tunix.rl.rollout import base_rollout
from tunix.sft import metrics_logger
```

---

### **Cell 3: Configuration**

*Hyperparameters for the GRPO run.*

```python
import os
import json
import dataclasses
import inspect
import jax
import jax.numpy as jnp
from flax import nnx
import qwix
from huggingface_hub import snapshot_download

# Tunix Imports
from tunix.models.gemma3 import model as gemma_lib
from tunix.models.gemma3 import params_safetensors as params_safetensors_lib

MODEL_ID = "google/gemma-3-1b-it"

print(f"Downloading {MODEL_ID}...")
local_model_path = snapshot_download(
    repo_id=MODEL_ID, 
    ignore_patterns=["*.pth"], 
    token=HF_TOKEN
)

print("--- Robust Config Loading (Strict + Injection) ---")
with open(os.path.join(local_model_path, "config.json"), "r") as f:
    hf_config = json.load(f)

# 1. Define Desired Mappings (HF -> Tunix)
mappings = {
    "num_embed": "vocab_size",
    "embed_dim": "hidden_size", 
    "hidden_dim": "intermediate_size", 
    "num_heads": "num_attention_heads",
    "num_kv_heads": "num_key_value_heads",
    "num_layers": "num_hidden_layers",
    "head_dim": "head_dim",
    "sliding_window_size": "sliding_window",
    "rope_base_frequency": "rope_theta",
    # We DO NOT map query_pre_attn_scalar here to avoid TypeError in __init__
}

# 2. Prepare Arguments
config_args = {}
# Detect what __init__ actually accepts
try:
    sig = inspect.signature(gemma_lib.ModelConfig)
    valid_init_fields = set(sig.parameters.keys())
    print(f"Detected valid init args: {list(valid_init_fields)[:5]}...")
except:
    # Fallback: assume mappings keys are valid if inspection fails
    valid_init_fields = set(mappings.keys())

# Map known fields
for tunix_key, hf_key in mappings.items():
    if tunix_key in valid_init_fields and hf_key in hf_config:
        config_args[tunix_key] = hf_config[hf_key]

# Fill other matching keys
for key, value in hf_config.items():
    if key in valid_init_fields and key not in config_args:
        config_args[key] = value

# 3. Instantiate Config (Strictly Valid Args Only)
print(f"Initializing config with {len(config_args)} arguments...")
model_config = gemma_lib.ModelConfig(**config_args)

# 4. CRITICAL FIX: Inject Missing Attributes Post-Init
# This fixes the 'NoneType' error by forcing the attribute to exist, 
# while avoiding the 'TypeError' in __init__.

print("Injecting missing configuration attributes...")

# Force query_pre_attn_scalar (Fixes: unsupported operand type for *: ... and NoneType)
# We use object.__setattr__ to bypass frozen dataclasses if necessary
try:
    if not hasattr(model_config, 'query_pre_attn_scalar') or model_config.query_pre_attn_scalar is None:
        object.__setattr__(model_config, 'query_pre_attn_scalar', 256.0) # Standard for 1B/2B
        print("✅ Injected 'query_pre_attn_scalar': 256.0")
except Exception as e:
    print(f"⚠️ Injection warning: {e}")

# Force sliding_window_size (Fixes: ValueError: sliding_window_size must be set)
try:
    if not hasattr(model_config, 'sliding_window_size') or model_config.sliding_window_size is None:
         object.__setattr__(model_config, 'sliding_window_size', 4096)
         print("✅ Injected 'sliding_window_size': 4096")
except Exception as e:
    print(f"⚠️ Injection warning: {e}")

# 5. Initialize Model on GPU
print("Initializing Model on GPU...")
num_devices = len(jax.devices())
mesh = jax.make_mesh((1, num_devices), axis_names=("fsdp", "tp"))

with mesh:
    # Load Base Model
    base_model = params_safetensors_lib.create_model_from_safe_tensors(
        local_model_path, model_config, mesh
    )
    
    # Define & Apply LoRA
    lora_provider = qwix.LoraProvider(
        module_path=".*q_einsum|.*kv_einsum|.*gate_proj|.*down_proj|.*up_proj|.*attn_vec_einsum",
        rank=64,
        alpha=64.0,
    )

    model_input = base_model.get_model_input()
    lora_policy = qwix.apply_lora_to_model(
        base_model, lora_provider, **model_input
    )
    
    # Shard State
    state = nnx.state(lora_policy)
    pspecs = nnx.get_partition_spec(state)
    sharded_state = jax.lax.with_sharding_constraint(state, pspecs)
    nnx.update(lora_policy, sharded_state)

print("✅ LoRA Policy Ready for GRPO on GPU!")
```

---

### **Cell 4: Model Loading (NNX + LoRA)**

*This loads Gemma 3 and wraps it with LoRA for training.*

```python
import optax
import tensorflow_datasets as tfds
import grain.python as grain
from tunix.rl import rl_cluster as rl_cluster_lib
from tunix.rl.grpo.grpo_learner import GRPOConfig, GRPOLearner
from tunix.rl.rollout import base_rollout
from tunix.generate import tokenizer_adapter as tokenizer_lib
import os
import time
import logging
import numpy as np

# --- 0. LOGGING & LATENCY SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(message)s', # Simplified format for cleaner step logs
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("GRPO_Trainer")

# Global counter to track steps manually
GLOBAL_STEP_COUNTER = 0

def measure_latency_and_step(func):
    """Decorator to measure time and print step headers."""
    def wrapper(*args, **kwargs):
        global GLOBAL_STEP_COUNTER
        
        # Only increment step on the primary reward function to avoid double counting
        if func.__name__ == "strict_format_reward":
            GLOBAL_STEP_COUNTER += 1
            print(f"\n{'='*20} STEP {GLOBAL_STEP_COUNTER} {'='*20}")

        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        # Calculate stats
        avg_score = np.mean(result) if result else 0.0
        duration = end_time - start_time
        
        # Print clearer metrics
        logger.info(f"[{func.__name__}] Avg Reward: {avg_score:.2f} | Time: {duration:.4f}s")
        return result
    return wrapper

# 1. DATASET SETUP
def get_dataset(split="train"):
    ds = tfds.load("gsm8k", split=split, as_supervised=False)
    ds_list = list(ds.as_numpy_iterator())
    
    def format_fn(ex):
        q = ex['question'].decode('utf-8')
        a = ex['answer'].decode('utf-8').split("####")[-1].strip()
        system_instruction = "Solve the math problem. Think step-by-step inside <reasoning> tags, and put the final number inside <answer> tags."
        text = f"<start_of_turn>user\n{system_instruction}\n\nProblem: {q}<end_of_turn>\n<start_of_turn>model\n"
        return {"prompts": text, "question": q, "answer": a}
    
    return grain.MapDataset.source(ds_list).map(format_fn).shuffle(seed=42).batch(1)

train_ds = get_dataset("train").repeat(50)

# 2. REWARD FUNCTIONS (With Step Tracking)
@measure_latency_and_step
def strict_format_reward(prompts, completions, **kwargs):
    rewards = []
    for c in completions:
        if "<reasoning>" in c and "</reasoning>" in c and "<answer>" in c:
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards

@measure_latency_and_step
def correctness_reward(prompts, completions, answer, **kwargs):
    rewards = []
    correct_count = 0
    for c, gt in zip(completions, answer):
        try:
            pred = c.split("<answer>")[-1].split("</answer>")[0].strip()
            if pred == gt:
                rewards.append(2.0)
                correct_count += 1
            else:
                rewards.append(0.0)
        except:
            rewards.append(0.0)
    
    # Explicitly log accuracy metric
    acc = (correct_count / len(completions)) * 100
    logger.info(f" >>> BATCH ACCURACY: {acc:.1f}% ({correct_count}/{len(completions)})")
    
    # Log the first completion to see what the model is doing
    if completions:
        snippet = completions[0].replace('\n', ' ')[:150]
        logger.info(f" >>> SAMPLE OUT: {snippet}...")
        
    return rewards

# 3. CONFIGURATION
GEMMA_TOKENIZER_PATH = "gs://gemma-data/tokenizers/tokenizer_gemma3.model"
tokenizer = tokenizer_lib.Tokenizer(tokenizer_path=GEMMA_TOKENIZER_PATH)
CHECKPOINT_DIR = os.path.abspath("checkpoints/grpo_safe_acc")

TRAIN_STEPS = 1000

cluster_config = rl_cluster_lib.ClusterConfig(
    role_to_mesh={
        rl_cluster_lib.Role.ACTOR: mesh,
        rl_cluster_lib.Role.REFERENCE: mesh,
        rl_cluster_lib.Role.ROLLOUT: mesh,
    },
    training_config=rl_cluster_lib.RLTrainingConfig(
        actor_optimizer=optax.adamw(learning_rate=1e-6),
        
        max_steps=TRAIN_STEPS,       
        mini_batch_size=1,     
        checkpoint_root_directory=CHECKPOINT_DIR,
        
        # --- CRITICAL CHANGE: EVALUATE EVERY STEP ---
        eval_every_n_steps=1,  # Forces internal logging every step
    ),
    rollout_config=base_rollout.RolloutConfig(
        max_tokens_to_generate=256, 
        temperature=0.8,
        eos_tokens=[tokenizer.eos_id()],
    ),
)

logger.info("Initializing RL Cluster...")
rl_cluster = rl_cluster_lib.RLCluster(
    actor=lora_policy,
    reference=base_model,
    tokenizer=tokenizer,
    cluster_config=cluster_config,
)

trainer = GRPOLearner(
    rl_cluster=rl_cluster,
    reward_fns=[strict_format_reward, correctness_reward],
    algo_config=GRPOConfig(
        num_generations=4, 
        beta=0.04
    ),
)

# 4. EXECUTION
logger.info(f"🚀 Starting Training ({TRAIN_STEPS} Steps) - Metrics will print below:")

with mesh:
    trainer.train(train_ds, None)
    
print(f"\n✅ Training Complete. Total Steps: {GLOBAL_STEP_COUNTER}")
```

### **Cell 5: Inference and Evaluation**

```python
# 5. INFERENCE / TESTING (UPDATED WITH ROBUST PROMPT)
from tunix.generate import sampler as sampler_lib
from tunix.models.gemma3 import model as gemma_lib 

# --- 1. SETUP SAMPLER (Same as before) ---
cache_config = sampler_lib.CacheConfig(
    cache_size=2048,
    num_layers=base_model.config.num_layers,
    num_kv_heads=base_model.config.num_kv_heads,
    head_dim=base_model.config.head_dim,
)

inference_sampler = sampler_lib.Sampler(
    transformer=lora_policy,
    tokenizer=tokenizer,
    cache_config=cache_config
)

# --- 2. ROBUST PROMPT GENERATOR ---
def generate_robust_response(problem_text):
    # A. The Rulebook (System Instruction)
    sys_instruction = (
        "You are a precise math reasoning engine. Follow this strict format:\n"
        "1. Think step-by-step inside <reasoning> tags. Be concise.\n"
        "2. Place the final result inside <answer> tags.\n"
        "3. Stop generating content immediately after closing the </answer> tag."
    )

    # B. The Template (One-Shot Example)
    # This shows the model EXACTLY what a perfect turn looks like.
    example_turn = (
        "<start_of_turn>user\n"
        "Problem: If I have 2 apples and buy 2 more, how many do I have?<end_of_turn>\n"
        "<start_of_turn>model\n"
        "<reasoning>\n"
        "1. Start with 2 apples.\n"
        "2. Add 2 bought apples: 2 + 2 = 4.\n"
        "</reasoning>\n"
        "<answer>4</answer><end_of_turn>\n"
    )

    # C. Combine into Full Prompt
    full_prompt = (
        f"<start_of_turn>user\n{sys_instruction}<end_of_turn>\n"
        f"{example_turn}"  # <--- INJECTED TEMPLATE
        f"<start_of_turn>user\nProblem: {problem_text}<end_of_turn>\n"
        f"<start_of_turn>model\n"
    )
    
    # D. Generate with Stricter Settings
    outputs = inference_sampler(
        input_strings=[full_prompt],
        max_generation_steps=512,  
        temperature=0.1,           # Low temp for logic
        top_k=40,
    )
    
    return outputs.text[0]

# --- 3. RUN TEST ---
problem = "Janet has 3 times as many apples as Bob. If Bob has 5 apples, how many apples do they have together?"
print(f"🤔 Asking Model with Engineered Prompt...\n")

try:
    output = generate_robust_response(problem)
    
    print("="*40)
    print("MODEL OUTPUT:")
    print("="*40)
    print(output)
    print("="*40)

    # Verification
    if "<reasoning>" in output and "</reasoning>" in output:
        print("\n✅ SUCCESS: Format is perfect.")
    else:
        print("\n⚠️ Note: Check if the model stopped correctly.")
        
except Exception as e:
    print(f"\n❌ Inference Error: {e}")
```

```python
def clean_and_parse_response(raw_text):
    # 1. Cut off at the first stop token to kill the loop
    stop_tokens = ["<end_of_turn>", "<start_of_turn>", "user", "model"]
    
    clean_text = raw_text
    for token in stop_tokens:
        if token in clean_text:
            clean_text = clean_text.split(token)[0]
            
    # 2. Extract answer if present
    answer = "N/A"
    if "<answer>" in clean_text:
        try:
            # aggressive parsing to find the number inside tags
            answer = clean_text.split("<answer>")[-1]
            answer = answer.split("</answer>")[0]
            answer = answer.split("<")[0] # Safety clip
        except:
            pass
            
    return clean_text.strip(), answer.strip()

# --- Run on your previous bad output ---
raw_output = output # The string with all the 'end_of_turn' garbage
cleaned, ans = clean_and_parse_response(raw_output)

print("----- CLEANED REASONING -----")
print(cleaned)
print("\n----- EXTRACTED ANSWER -----")
print(f"[{ans}]")
```

```python
from tunix.generate import sampler as sampler_lib

# --- 1. SETUP CACHE CONFIG ---
# We can share the cache config for both samplers
cache_config = sampler_lib.CacheConfig(
    cache_size=2048,
    num_layers=base_model.config.num_layers,
    num_kv_heads=base_model.config.num_kv_heads,
    head_dim=base_model.config.head_dim,
)

# --- 2. DEFINE BASE SAMPLER (The pre-trained model) ---
# This uses 'base_model' defined in Cell 3 of your notebook
print("⚖️ Wrapping Base Model (Gemma 3)...")
base_sampler = sampler_lib.Sampler(
    transformer=base_model,  # <--- The frozen base model
    tokenizer=tokenizer,     # Defined in Cell 4
    cache_config=cache_config
)

# --- 3. DEFINE INFERENCE SAMPLER (The RL fine-tuned model) ---
# This uses 'lora_policy' defined in Cell 3/4 of your notebook
print("🚀 Wrapping RL Model (Gemma 3 + LoRA)...")
inference_sampler = sampler_lib.Sampler(
    transformer=lora_policy, # <--- The LoRA model you trained
    tokenizer=tokenizer,
    cache_config=cache_config
)
```

```python
import tensorflow_datasets as tfds
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re

# 1. DATASET SETUP (Using Official Test Split)
print("📊 Loading GSM8K Test Split (Hold-out Data)...")
# Note: This requires an internet connection the first time to download
ds_test = tfds.load("gsm8k", split="test", as_supervised=False)
test_data = list(ds_test.take(20).as_numpy_iterator())

# 2. EVALUATION LOGIC
def extract_answer_and_format(text):
    """Parses output for strict XML tags and the numerical answer."""
    has_tags = "<reasoning>" in text and "<answer>" in text
    
    # Attempt to extract answer from tags
    answer = "N/A"
    if "<answer>" in text:
        try:
            answer = text.split("<answer>")[-1].split("</answer>")[0].strip()
        except:
            pass
            
    # Fallback: If tags missing (Base Model), find the last number in text
    if answer == "N/A":
        nums = re.findall(r"[-+]?\d*\.\d+|\d+", text)
        if nums:
            answer = nums[-1]
            
    # Calculate reasoning length (chars)
    reasoning_len = 0
    if "<reasoning>" in text and "</reasoning>" in text:
        try:
            r_text = text.split("<reasoning>")[1].split("</reasoning>")[0]
            reasoning_len = len(r_text)
        except:
            pass
        
    return answer, has_tags, reasoning_len

def check_correctness(pred, gt):
    """Robust numerical comparison."""
    try:
        # Remove commas and handle simple formatting
        pred_clean = pred.replace(',', '')
        gt_clean = gt.replace(',', '')
        pred_val = float(re.findall(r"[-+]?\d*\.\d+|\d+", pred_clean)[0])
        gt_val = float(re.findall(r"[-+]?\d*\.\d+|\d+", gt_clean)[0])
        return abs(pred_val - gt_val) < 1e-4
    except:
        return False

# 3. RUN COMPARATIVE INFERENCE
results = []
print(f"🚀 Starting Rigorous Evaluation on {len(test_data)} samples...\n")

# --- PROMPT TEMPLATES ---
sys_instruction = (
    "You are a precise math reasoning engine. Follow this strict format:\n"
    "1. Think step-by-step inside <reasoning> tags. Be concise.\n"
    "2. Place the final result inside <answer> tags.\n"
    "3. Stop generating content immediately after closing the </answer> tag."
)

example_turn = (
    "<start_of_turn>user\n"
    "Problem: If I have 2 apples and buy 2 more, how many do I have?<end_of_turn>\n"
    "<start_of_turn>model\n"
    "<reasoning>\n"
    "1. Start with 2 apples.\n"
    "2. Add 2 bought apples: 2 + 2 = 4.\n"
    "</reasoning>\n"
    "<answer>4</answer><end_of_turn>\n"
)

for i, ex in enumerate(test_data):
    q = ex['question'].decode('utf-8')
    gt_raw = ex['answer'].decode('utf-8')
    gt = gt_raw.split("####")[-1].strip()

    # Construct Full Prompt
    full_prompt = (
        f"<start_of_turn>user\n{sys_instruction}<end_of_turn>\n"
        f"{example_turn}"
        f"<start_of_turn>user\nProblem: {q}<end_of_turn>\n"
        f"<start_of_turn>model\n"
    )
    
    # --- SAFE INITIALIZATION (The Fix) ---
    # We initialize these BEFORE the try blocks so they exist in the 'except' case
    base_ans = "Error"
    base_has_tags = False
    base_correct = False
    
    rl_ans = "Error"
    rl_has_tags = False
    rl_len = 0
    rl_correct = False
    
    # --- A. BASE MODEL INFERENCE ---
    try:
        # Assuming base_sampler is defined in your environment
        base_out = base_sampler(input_strings=[full_prompt], max_generation_steps=512, temperature=0.6).text[0]
        base_ans, base_has_tags, _ = extract_answer_and_format(base_out)
        base_correct = check_correctness(base_ans, gt)
    except Exception as e:
        print(f"⚠️ Base Model failed on sample {i}: {e}")
        base_ans = "Error" # Explicitly ensure it's set
        base_correct = False
        
    # --- B. RL MODEL INFERENCE ---
    try:
        # Assuming inference_sampler is defined in your environment
        rl_out = inference_sampler(
            input_strings=[full_prompt], 
            max_generation_steps=512, 
            temperature=0.1, 
            top_k=40
        ).text[0]
        
        rl_ans, rl_has_tags, rl_len = extract_answer_and_format(rl_out)
        rl_correct = check_correctness(rl_ans, gt)
    except Exception as e:
        print(f"⚠️ RL Model failed on sample {i}: {e}")
        rl_ans = "Error" # Explicitly ensure it's set
        rl_correct = False
        
    # Log Result
    results.append({
        "Question": q[:40]+"...",
        "Ground Truth": gt,
        "Base Ans": base_ans,
        "Base Correct": base_correct,
        "RL Ans": rl_ans,
        "RL Correct": rl_correct,
        "RL Format Compliance": rl_has_tags,
        "RL Reasoning Length": rl_len
    })
    
    # Live Progress Bar
    status = "✅" if rl_correct else "❌"
    print(f"Sample {i+1:02d}: Base={'✅' if base_correct else '❌'} | RL={status} | Tags={'Yes' if rl_has_tags else 'No'}")

# 4. AGGREGATE METRICS & PLOTTING
if not results:
    print("No results to display.")
else:
    df = pd.DataFrame(results)
    base_acc = df['Base Correct'].mean() * 100
    rl_acc = df['RL Correct'].mean() * 100
    fmt_score = df['RL Format Compliance'].mean() * 100

    print("\n" + "="*50)
    print("🏆 FINAL RESEARCH REPORT")
    print("="*50)
    print(f"Base Model Accuracy:      {base_acc:.1f}%")
    print(f"RL Model Accuracy:        {rl_acc:.1f}%")
    print(f"RL Format Compliance:     {fmt_score:.1f}%")
    print(f"Improvement (Lift):       {rl_acc - base_acc:+.1f}%")
    print("="*50)

    # Plots
    plt.figure(figsize=(15, 5))

    # Plot 1: Accuracy
    plt.subplot(1, 3, 1)
    sns.barplot(x=['Base Model', 'RL Model'], y=[base_acc/100, rl_acc/100], palette=['#95a5a6', '#2ecc71'])
    plt.title('Pass@1 Accuracy')
    plt.ylim(0, 1)
    plt.ylabel('Accuracy')

    # Plot 2: Format Adherence
    plt.subplot(1, 3, 2)
    sns.barplot(x=['Base Model', 'RL Model'], y=[df['Base Correct'].mean()*0, fmt_score/100], palette=['#95a5a6', '#f1c40f'])
    plt.title('Adherence to XML Format')
    plt.ylim(0, 1)

    # Plot 3: Reasoning Length vs Correctness
    plt.subplot(1, 3, 3)
    try:
        sns.boxplot(data=df, x='RL Correct', y='RL Reasoning Length', palette='coolwarm')
        plt.title('Reasoning Length vs Correctness')
        plt.xlabel('Is Answer Correct?')
        plt.ylabel('Characters in <reasoning>')
    except:
        plt.text(0.5, 0.5, "Not enough data for boxplot", ha='center')

    plt.tight_layout()
    plt.savefig("rl_research_evaluation.png")
    print("📊 Plots saved to 'rl_research_evaluation.png'")
```

---

### **Cell 6: save model**

```python
import os
import shutil

# --- CONFIGURATION ---
# 1. Checkpoint Location
CHECKPOINT_ROOT = os.path.abspath("checkpoints/grpo_safe_acc")

# 2. Define the output name ONCE to prevent typos
OUTPUT_NAME = "gemma3_rl_finetuned"
SUBMISSION_PATH = os.path.join("submission", OUTPUT_NAME)

print(f"📂 Scanning for checkpoints in: {CHECKPOINT_ROOT}")

# --- HELPER: Find latest step ---
def get_latest_checkpoint_dir(root_path):
    if not os.path.exists(root_path): return None
    candidates = []
    for root, dirs, files in os.walk(root_path):
        for d in dirs:
            if d.isdigit():
                candidates.append((int(d), os.path.join(root, d)))
    if not candidates: return None
    return sorted(candidates, key=lambda x: x[0], reverse=True)[0]

# --- EXECUTION ---
result = get_latest_checkpoint_dir(CHECKPOINT_ROOT)

if result:
    latest_step, source_path = result
    print(f"✅ Found latest checkpoint: Step {latest_step}")
    
    # 1. Clean & Create Destination
    if os.path.exists(SUBMISSION_PATH):
        shutil.rmtree(SUBMISSION_PATH)
    
    # 2. Copy Files
    print(f"📦 Copying to: {SUBMISSION_PATH} ...")
    shutil.copytree(source_path, SUBMISSION_PATH)
    
    # 3. Zip (Using the same correct path)
    print("🗜️  Zipping...")
    shutil.make_archive(OUTPUT_NAME, 'zip', root_dir=SUBMISSION_PATH)
    
    print("\n" + "="*50)
    print("🎉 SUCCESS! Download Ready.")
    print("="*50)
    print(f"File: {OUTPUT_NAME}.zip")
    print("✅ Created gemma3_rl_finetuned.zip")
    print("Download it from the 'Output' file browser on the right sidebar!")
else:
    print(f"❌ Error: No numbered checkpoint folders found in {CHECKPOINT_ROOT}")
```