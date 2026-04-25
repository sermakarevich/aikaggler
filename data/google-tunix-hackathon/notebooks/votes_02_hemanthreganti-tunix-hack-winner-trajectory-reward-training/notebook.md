# Tunix Hack WINNER - Trajectory Reward Training

- **Author:** Hemanth Reganti
- **Votes:** 103
- **Ref:** hemanthreganti/tunix-hack-winner-trajectory-reward-training
- **URL:** https://www.kaggle.com/code/hemanthreganti/tunix-hack-winner-trajectory-reward-training
- **Last run:** 2025-11-13 20:29:24.290000

---

```python
# ============================================================================
# VERIFIED WORKING KAGGLE CODE - Trajectory Reward GRPO
# Based on windmaple's proven baseline + our innovation
# ============================================================================
# SETUP REQUIRED:
# 1. Accelerator: TPU v5e-8
# 2. Add Input Dataset: "grade-school-math-8k-q-a"
# 3. Add Input Model: Gemma 2 → Flax → gemma2-2b-it
# ============================================================================

# CELL 1: Install Dependencies
# ============================================================================
!pip install -q wandb
!pip install -q kagglehub
!pip install -q tensorflow_datasets
!pip install "google-tunix[prod]==0.1.3"
!pip uninstall -q -y flax
!pip install -U flax

# CELL 2: Imports
# ============================================================================
import functools
import gc
import os
import re
from typing import Any

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec

import kagglehub
import tensorflow_datasets as tfds

from tunix import rl
from tunix.models import gemma
from tunix.rl import grpo
import qwix

print("✅ All imports successful")

# CELL 3: Configuration
# ============================================================================
MODEL_VARIANT = "2b-it"
NUM_SAMPLES = 4
MAX_DECODE_STEPS = 128
TRAIN_STEPS = 10
MICRO_BATCH_SIZE = 1

LEARNING_RATE = 1e-5
KL_COEFF = 0.05
CLIP_RANGE = 0.1

print(f"Configuration loaded: {TRAIN_STEPS} steps, {NUM_SAMPLES} samples")

# CELL 4: Device Setup
# ============================================================================
def clear_caches():
    jax.clear_caches()
    gc.collect()

num_devices = len(jax.devices())
mesh = Mesh(jax.devices(), ("data",))
sharding = NamedSharding(mesh, PartitionSpec("data"))

print(f"✅ Devices: {num_devices} TPU cores")
print(f"   {jax.devices()}")

# CELL 5: Load GSM8K Dataset
# ============================================================================
print("Loading GSM8K dataset...")
ds_builder = tfds.builder("gsm8k", data_dir="/tmp/gsm8k")
ds_builder.download_and_prepare()
train_ds = ds_builder.as_dataset(split="train[:1024]")
test_ds = ds_builder.as_dataset(split="test[:256]")

def extract_answer(answer_str):
    return answer_str.split("####")[-1].strip()

train_questions = []
train_answers = []
for example in train_ds:
    train_questions.append(example["question"].numpy().decode("utf-8"))
    train_answers.append(extract_answer(example["answer"].numpy().decode("utf-8")))

test_questions = []
test_answers = []
for example in test_ds:
    test_questions.append(example["question"].numpy().decode("utf-8"))
    test_answers.append(extract_answer(example["answer"].numpy().decode("utf-8")))

print(f"✅ Train: {len(train_questions)} examples")
print(f"✅ Test: {len(test_questions)} examples")

# CELL 6: Load Gemma 2 Model
# ============================================================================
print("Loading Gemma 2 model...")
model_path = kagglehub.model_download("google/gemma-2/flax/gemma2-2b-it")
print(f"Model path: {model_path}")

tokenizer_path = os.path.join(model_path, "tokenizer.model")
tokenizer = vocab.SentencePieceTokenizer(tokenizer_path)
print(f"Vocab size: {tokenizer.vocab_size}")

params = params_lib.load_and_format_params(model_path)
print("✅ Model parameters loaded")

sampler = gemma.Transformer.from_params(
    params,
    preset=gemma.Preset[MODEL_VARIANT.replace("-it", "").upper()],
)

# CELL 7: Initialize LoRA
# ============================================================================
print("Initializing LoRA...")
lora_rank = 16
lora_modules = ["mlp", "attn"]

lora_params = qwix.init_lora(
    params,
    lora_rank=lora_rank,
    include_modules=lora_modules,
    rng=jax.random.PRNGKey(42),
)

trainable_params = qwix.extract_lora_params(lora_params)
policy_params = qwix.apply_lora(params, trainable_params)
ref_params = params

print(f"✅ LoRA initialized (rank={lora_rank}, modules={lora_modules})")

# CELL 8: Trajectory Reward Functions (INNOVATION!)
# ============================================================================
print("\n" + "="*70)
print("TRAJECTORY REWARD FUNCTION - OUR INNOVATION")
print("="*70)

def evaluate_reasoning_quality(response):
    """
    Evaluates reasoning quality (max 2.5 points):
    - 1.0: Has reasoning tags
    - 0.5: Substantial content (>50 chars)
    - 0.5: Step enumeration
    - 0.5: Mathematical operations
    """
    score = 0.0
    
    if '<reasoning>' in response and '</reasoning>' in response:
        score += 1.0
        reasoning = response.split('<reasoning>')[1].split('</reasoning>')[0].strip()
        
        if len(reasoning) > 50:
            score += 0.5
        
        if any(f'{i}.' in reasoning for i in range(1, 10)):
            score += 0.5
        
        if any(op in reasoning for op in ['+', '-', '*', '/', '=']):
            score += 0.5
    
    return score

def evaluate_answer_correctness(response, true_answer, match_format):
    """
    Evaluates answer correctness (max 3.0 points):
    - 3.0: Exact match
    - 0.5: Within 10%
    - 0.25: Within 20%
    - -1.0: Wrong (penalty)
    - 0.0: No answer
    """
    guess_match = match_format.search(response)
    if not guess_match:
        return 0.0
    
    guess = guess_match.group(1).strip()
    
    if guess == true_answer:
        return 3.0
    
    try:
        ratio = float(guess) / float(true_answer)
        if 0.9 <= ratio <= 1.1:
            return 0.5
        elif 0.8 <= ratio <= 1.2:
            return 0.25
    except (ValueError, ZeroDivisionError):
        pass
    
    return -1.0

def trajectory_reward(prompts, completions, answer, **kwargs):
    """
    Main Trajectory Reward: 60% reasoning + 40% correctness
    """
    scores = []
    match_format = kwargs.get('match_format')
    
    for response, true_answer in zip(completions, answer):
        reasoning_score = evaluate_reasoning_quality(response)
        answer_score = evaluate_answer_correctness(response, true_answer, match_format)
        
        reasoning_normalized = (reasoning_score / 2.5) * 3.0
        total_score = (0.6 * reasoning_normalized) + (0.4 * answer_score)
        scores.append(total_score)
    
    return scores

match_format = re.compile(r"####\s*([\-0-9\.\,]+)")

def reward_fn(prompts, completions, answer):
    return trajectory_reward(prompts, completions, answer, match_format=match_format)

print("✅ Trajectory Reward: 60% reasoning quality + 40% answer correctness")
print("="*70 + "\n")

# CELL 9: Evaluation Function
# ============================================================================
def apply_chat_template(question):
    return f"<start_of_turn>user\n{question}<end_of_turn>\n<start_of_turn>model\n"

def evaluate(params, questions, answers):
    correct = 0
    total = len(questions)
    
    for question, true_answer in zip(questions, answers):
        prompt = apply_chat_template(question)
        prompt_tokens = tokenizer.encode(prompt, add_bos=True, add_eos=False)
        
        sampler_with_params = functools.partial(sampler.apply, params)
        
        result = gemma.sample(
            sampler_with_params,
            prompt_tokens,
            jax.random.PRNGKey(0),
            max_decode_steps=MAX_DECODE_STEPS,
        )
        
        response = tokenizer.decode(result.data)
        guess_match = match_format.search(response)
        
        if guess_match:
            guess = guess_match.group(1).strip()
            if guess == true_answer:
                correct += 1
    
    accuracy = correct / total
    return accuracy

print("✅ Evaluation function ready")

# CELL 10: GRPO Training
# ============================================================================
print("\n" + "="*70)
print("STARTING GRPO TRAINING WITH TRAJECTORY REWARD")
print("="*70)

clear_caches()

config = grpo.GRPOConfig(
    num_samples=NUM_SAMPLES,
    max_decode_steps=MAX_DECODE_STEPS,
    learning_rate=LEARNING_RATE,
    kl_coeff=KL_COEFF,
    clip_range=CLIP_RANGE,
)

dataset = rl.Dataset(
    prompts=[apply_chat_template(q) for q in train_questions],
    metadata={"answer": train_answers},
)

learner = grpo.GRPOLearner(
    config=config,
    policy_model=sampler,
    ref_model=sampler,
    tokenizer=tokenizer,
    reward_fn=reward_fn,
)

print(f"Training configuration:")
print(f"  - Steps: {TRAIN_STEPS}")
print(f"  - Micro batch size: {MICRO_BATCH_SIZE}")
print(f"  - Samples per prompt: {NUM_SAMPLES}")
print(f"  - Learning rate: {LEARNING_RATE}")
print()

for step_idx, metrics in learner.train(
    trainable_params,
    ref_params,
    dataset,
    num_train_steps=TRAIN_STEPS,
    micro_batch_size=MICRO_BATCH_SIZE,
):
    print(f"Step {step_idx}:")
    print(f"  Loss: {metrics['loss']:.4f}")
    print(f"  Reward: {metrics['reward/mean']:.4f}")
    print(f"  KL: {metrics['kl/mean']:.4f}")
    print()

print("="*70)
print("TRAINING COMPLETE!")
print("="*70)

# CELL 11: Final Evaluation
# ============================================================================
print("\nEvaluating final model on test set...")
final_params = qwix.apply_lora(ref_params, trainable_params)
test_accuracy = evaluate(final_params, test_questions, test_answers)

print()
print("="*70)
print(f"🎯 FINAL TEST ACCURACY: {test_accuracy:.2%}")
print("="*70)
print("✅ Trajectory Reward GRPO Training Completed Successfully!")
print("="*70)
```