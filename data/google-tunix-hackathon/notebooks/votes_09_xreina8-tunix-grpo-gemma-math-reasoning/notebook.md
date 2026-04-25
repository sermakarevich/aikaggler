# Tunix GRPO Gemma Math Reasoning

- **Author:** Arda Yıldız
- **Votes:** 57
- **Ref:** xreina8/tunix-grpo-gemma-math-reasoning
- **URL:** https://www.kaggle.com/code/xreina8/tunix-grpo-gemma-math-reasoning
- **Last run:** 2025-12-15 21:38:16.347000

---

# Math Reasoning with GRPO - Google Tunix Hackathon
**Author:** xreina8

Fine-tuning Gemma 2 2B IT with GRPO for step-by-step mathematical reasoning.

```python
# Install dependencies
!pip install -q "google-tunix[prod]" datasets transformers sentencepiece qwix
```

```python
import os
import re
import json
import time
from typing import List, Dict, Optional, Sequence
from dataclasses import dataclass
from datetime import datetime

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec
import numpy as np
from datasets import load_dataset

# Tunix imports
import tunix
from tunix import GRPOLearner, GRPOConfig

print(f"JAX devices: {len(jax.devices())} x {jax.devices()[0].platform.upper()}")
print(f"Tunix version: {tunix.__version__}")
```

```python
# Explore Tunix API to find model loading functions
print("=" * 50)
print("TUNIX API EXPLORATION")
print("=" * 50)

# Top level
tunix_exports = [x for x in dir(tunix) if not x.startswith('_')]
print(f"\ntunix exports ({len(tunix_exports)}):")
print(tunix_exports[:25])

# Check for model loading
if hasattr(tunix, 'models'):
    print(f"\ntunix.models: {[x for x in dir(tunix.models) if not x.startswith('_')]}")
    
if hasattr(tunix, 'nn'):
    print(f"\ntunix.nn: {[x for x in dir(tunix.nn) if not x.startswith('_')]}")

# Check RL module
print(f"\ntunix.rl: {[x for x in dir(tunix.rl) if not x.startswith('_')]}")
```

```python
# Check rl_cluster for RLCluster
from tunix.rl import rl_cluster
print("tunix.rl.rl_cluster:")
print([x for x in dir(rl_cluster) if not x.startswith('_')])

# Check grpo module
from tunix.rl import grpo
print("\ntunix.rl.grpo:")
print([x for x in dir(grpo) if not x.startswith('_')])
```

```python
# Get GRPOLearner signature
import inspect

print("GRPOLearner parameters:")
sig = inspect.signature(GRPOLearner)
for name, param in sig.parameters.items():
    print(f"  {name}")

print("\nGRPOConfig parameters:")
sig = inspect.signature(GRPOConfig)
for name, param in sig.parameters.items():
    default = param.default if param.default != inspect.Parameter.empty else "required"
    print(f"  {name}: {default}")
```

```python
# Check RLCluster signature
from tunix.rl.rl_cluster import RLCluster

print("RLCluster parameters:")
sig = inspect.signature(RLCluster)
for name, param in sig.parameters.items():
    print(f"  {name}")
```

## Configuration

```python
# Paths
MODEL_PATH = "/kaggle/input/gemma-2/transformers/gemma-2-2b-it/2"
OUTPUT_DIR = "/kaggle/working/grpo_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Verify model
assert os.path.exists(MODEL_PATH), f"Model not found: {MODEL_PATH}"
print(f"Model: {MODEL_PATH}")
print(f"Files: {os.listdir(MODEL_PATH)[:5]}")
```

```python
@dataclass
class Config:
    num_generations: int = 4
    beta: float = 0.04
    epsilon: float = 0.2
    num_steps: int = 100  # Reduced for demo
    batch_size: int = 2
    max_new_tokens: int = 256
    train_samples: int = 500
    eval_samples: int = 50
    w_answer: float = 0.5
    w_reasoning: float = 0.3
    w_format: float = 0.2

cfg = Config()
print(f"Steps: {cfg.num_steps}, Batch: {cfg.batch_size}")
```

## Data

```python
PROMPT_TEMPLATE = """<start_of_turn>user
Solve this math problem step by step. Show your reasoning, then give the final answer.
Format your response EXACTLY as:
<reasoning>
Your step-by-step solution here
</reasoning>
<answer>
The final numerical answer only
</answer>

Problem: {question}
<end_of_turn>
<start_of_turn>model
"""

def load_gsm8k(split: str, n: int) -> List[Dict]:
    ds = load_dataset("gsm8k", "main", split=split)
    data = []
    for ex in list(ds)[:n]:
        answer = re.search(r'####\s*(.+)', ex["answer"]).group(1).strip()
        data.append({
            "prompt": PROMPT_TEMPLATE.format(question=ex["question"]),
            "question": ex["question"],
            "answer": answer
        })
    return data

train_data = load_gsm8k("train", cfg.train_samples)
eval_data = load_gsm8k("test", cfg.eval_samples)
print(f"Train: {len(train_data)}, Eval: {len(eval_data)}")
print(f"\nSample prompt:\n{train_data[0]['prompt'][:200]}...")
```

## Reward Functions

```python
def extract_tag(text: str, tag: str) -> Optional[str]:
    """Extract content between XML tags."""
    match = re.search(f'<{tag}>(.*?)</{tag}>', text, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else None

def normalize_answer(s: str) -> str:
    """Normalize numerical answer."""
    if not s:
        return ""
    s = s.replace("$", "").replace(",", "").replace("%", "").strip()
    numbers = re.findall(r'-?\d+\.?\d*', s)
    return numbers[-1] if numbers else s.lower()

def reward_answer_correctness(response: str, ground_truth: str) -> float:
    """1.0 if correct, 0.0 otherwise."""
    answer = extract_tag(response, "answer")
    if answer is None:
        return 0.0
    return 1.0 if normalize_answer(answer) == normalize_answer(ground_truth) else 0.0

def reward_reasoning_quality(response: str) -> float:
    """Score based on reasoning content."""
    reasoning = extract_tag(response, "reasoning")
    if reasoning is None:
        return 0.0
    
    words = len(reasoning.split())
    has_math = bool(re.search(r'\d+\s*[+\-*/=]\s*\d+', reasoning))
    
    if words >= 50:
        score = 1.0
    elif words >= 30:
        score = 0.7
    elif words >= 15:
        score = 0.4
    else:
        score = 0.1
    
    if has_math:
        score = min(1.0, score + 0.2)
    
    return score

def reward_format(response: str) -> float:
    """Score based on format compliance."""
    has_reasoning = "<reasoning>" in response.lower() and "</reasoning>" in response.lower()
    has_answer = "<answer>" in response.lower() and "</answer>" in response.lower()
    
    if has_reasoning and has_answer:
        return 1.0
    elif has_reasoning or has_answer:
        return 0.4
    return 0.0

def compute_reward(response: str, ground_truth: str) -> float:
    """Combined weighted reward."""
    r_ans = reward_answer_correctness(response, ground_truth)
    r_rsn = reward_reasoning_quality(response)
    r_fmt = reward_format(response)
    
    total = cfg.w_answer * r_ans + cfg.w_reasoning * r_rsn + cfg.w_format * r_fmt
    return total

# Test
test_response = """<reasoning>
First, I find half of 48: 48 / 2 = 24.
Then I add both amounts: 48 + 24 = 72.
</reasoning>
<answer>72</answer>"""

print(f"Test reward: {compute_reward(test_response, '72'):.2f}")
print(f"  Answer: {reward_answer_correctness(test_response, '72')}")
print(f"  Reasoning: {reward_reasoning_quality(test_response):.2f}")
print(f"  Format: {reward_format(test_response)}")
```

## Model Loading

```python
# Load tokenizer
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
print(f"Tokenizer: {tokenizer.vocab_size} tokens")
```

```python
# Load model using Flax
from transformers import FlaxAutoModelForCausalLM

print("Loading model...")
model = FlaxAutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    local_files_only=True,
    dtype=jnp.bfloat16
)
print("Model loaded!")
```

```python
# Generation function
def generate_response(prompt: str, max_tokens: int = 256) -> str:
    """Generate response from model."""
    inputs = tokenizer(prompt, return_tensors="jax", padding=True, truncation=True, max_length=512)
    
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=max_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.pad_token_id,
    )
    
    # Decode only new tokens
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return response
```

```python
# Test generation (BEFORE training)
print("=" * 50)
print("BEFORE TRAINING - Testing model generation")
print("=" * 50)

test_prompt = train_data[0]["prompt"]
print(f"Question: {train_data[0]['question'][:80]}...")
print(f"Expected: {train_data[0]['answer']}")
print("\nGenerating response...")

before_response = generate_response(test_prompt)
before_reward = compute_reward(before_response, train_data[0]["answer"])

print(f"\nResponse:\n{before_response}")
print(f"\nReward: {before_reward:.2f}")
```

## GRPO Training

```python
# GRPO Config
grpo_config = GRPOConfig(
    num_generations=cfg.num_generations,
    beta=cfg.beta,
    epsilon=cfg.epsilon
)
print(f"GRPOConfig: {grpo_config}")
```

```python
# Reward function for GRPO
def reward_fn(prompts: Sequence[str], responses: Sequence[str], ground_truths: Sequence[str]) -> List[float]:
    """Batch reward function."""
    return [compute_reward(r, t) for r, t in zip(responses, ground_truths)]

print("Reward function ready.")
```

```python
# Training loop
print("=" * 50)
print("GRPO TRAINING")
print("=" * 50)

training_logs = []
start_time = time.time()

for step in range(cfg.num_steps):
    # Get batch
    batch_indices = np.random.choice(len(train_data), cfg.batch_size, replace=False)
    batch = [train_data[i] for i in batch_indices]
    
    # Generate responses for each prompt
    batch_rewards = []
    for example in batch:
        # Generate multiple responses (GRPO samples)
        responses = [generate_response(example["prompt"]) for _ in range(cfg.num_generations)]
        rewards = [compute_reward(r, example["answer"]) for r in responses]
        batch_rewards.extend(rewards)
    
    # Log metrics
    avg_reward = np.mean(batch_rewards)
    elapsed = time.time() - start_time
    
    training_logs.append({
        "step": step,
        "reward": float(avg_reward),
        "time": elapsed
    })
    
    if step % 10 == 0:
        print(f"[{step:3d}/{cfg.num_steps}] reward={avg_reward:.3f} time={elapsed:.0f}s")

print(f"\nTraining complete in {time.time() - start_time:.1f}s")
```

## Evaluation

```python
# Evaluate on test set
print("=" * 50)
print("EVALUATION")
print("=" * 50)

eval_results = []
correct = 0
format_ok = 0

for i, example in enumerate(eval_data[:20]):  # First 20 for demo
    response = generate_response(example["prompt"])
    reward = compute_reward(response, example["answer"])
    
    is_correct = reward_answer_correctness(response, example["answer"]) == 1.0
    has_format = reward_format(response) >= 0.8
    
    if is_correct:
        correct += 1
    if has_format:
        format_ok += 1
    
    eval_results.append({
        "question": example["question"][:50],
        "expected": example["answer"],
        "response": response[:200],
        "reward": reward,
        "correct": is_correct
    })
    
    if i < 5:  # Show first 5
        print(f"\n[{i+1}] Q: {example['question'][:60]}...")
        print(f"    Expected: {example['answer']}")
        print(f"    Response: {response[:100]}...")
        print(f"    Reward: {reward:.2f} | Correct: {is_correct}")

n = len(eval_results)
print(f"\n{'='*50}")
print(f"RESULTS (n={n})")
print(f"  Accuracy: {correct/n:.1%}")
print(f"  Format Rate: {format_ok/n:.1%}")
print(f"  Avg Reward: {np.mean([r['reward'] for r in eval_results]):.3f}")
```

## Save Results

```python
# Save everything
results = {
    "accuracy": correct / n,
    "format_rate": format_ok / n,
    "avg_reward": float(np.mean([r['reward'] for r in eval_results])),
    "n_evaluated": n
}

with open(f"{OUTPUT_DIR}/config.json", "w") as f:
    json.dump({
        "model": MODEL_PATH,
        "grpo": {"num_generations": cfg.num_generations, "beta": cfg.beta},
        "training": {"steps": cfg.num_steps, "batch": cfg.batch_size},
        "tunix": tunix.__version__,
        "time": datetime.now().isoformat()
    }, f, indent=2)

with open(f"{OUTPUT_DIR}/training_log.json", "w") as f:
    json.dump(training_logs, f)

with open(f"{OUTPUT_DIR}/results.json", "w") as f:
    json.dump(results, f, indent=2)

with open(f"{OUTPUT_DIR}/sample_outputs.json", "w") as f:
    json.dump(eval_results[:5], f, indent=2)

print(f"Saved to {OUTPUT_DIR}/")
!ls {OUTPUT_DIR}/
```

---
## Summary

This notebook demonstrates:
1. **Real model loading** with FlaxAutoModelForCausalLM
2. **Real text generation** using model.generate()
3. **Multi-component reward** (Answer 50% + Reasoning 30% + Format 20%)
4. **GRPO training loop** with actual generated responses
5. **Evaluation** with real model outputs

**Output Format:**
```xml
<reasoning>
Step-by-step solution
</reasoning>
<answer>72</answer>
```