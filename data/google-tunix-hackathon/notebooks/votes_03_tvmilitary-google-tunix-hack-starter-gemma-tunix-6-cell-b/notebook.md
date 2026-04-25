# Google Tunix Hack Starter — Gemma + Tunix 6‑Cell B

- **Author:** TV Military
- **Votes:** 99
- **Ref:** tvmilitary/google-tunix-hack-starter-gemma-tunix-6-cell-b
- **URL:** https://www.kaggle.com/code/tvmilitary/google-tunix-hack-starter-gemma-tunix-6-cell-b
- **Last run:** 2026-04-04 02:50:46.360000

---

This notebook provides a simple 6‑cell starter baseline for the Google Tunix Hackathon.  
It demonstrates how to load the competition data, initialize Gemma models with Tunix, run a basic GRPO training loop, and generate outputs in the required <reasoning>…</reasoning> and <answer>…</answer> format.  

Designed for educational purposes — not guaranteed to reach medal level.  
Participants are encouraged to extend the reward function, adjust hyperparameters, and refine the reasoning trace generation to improve performance.

```python
# =========================
# Cell 1 — Imports & Config
# =========================
import os, random
import jax, flax, optax
import tunix
from transformers import AutoTokenizer, FlaxAutoModelForCausalLM

SEED = 42
random.seed(SEED)
MODEL_NAME = "google/gemma-2b"   # or "google/gemma-3b"
DEVICE = jax.devices()[0]
```

```python
# =========================
# Cell 2 — Load Data
# =========================
import pandas as pd
DATA_DIR = "/kaggle/input/google-tunix-hackathon"
train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
print("Train shape:", train.shape)
```

```python
# =========================
# Cell 3 — Tokenizer & Model
# =========================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = FlaxAutoModelForCausalLM.from_pretrained(MODEL_NAME)

def format_prompt(question):
    return f"Question: {question}\nPlease show reasoning and final answer."
```

```python
# =========================
# Cell 4 — Training Loop (Skeleton)
# =========================
# Tunix GRPO training skeleton
reward_fn = tunix.rewards.basic_trace_reward
trainer = tunix.Trainer(
    model=model,
    tokenizer=tokenizer,
    reward_fn=reward_fn,
    learning_rate=1e-5,
    num_train_steps=1000
)

# Example: train on a small batch
sample_batch = [format_prompt(q) for q in train["question"].head(8)]
trainer.train_step(sample_batch)
```

```python
# =========================
# Cell 5 — Inference
# =========================
def solve(query):
    inputs = tokenizer(format_prompt(query), return_tensors="jax")
    outputs = model.generate(**inputs, max_length=256)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text

print(solve("What is 2+2?"))
```

```python
# =========================
# Cell 6 — Submission (Reasoning + Answer)
# =========================
preds = []
for _, row in train.head(5).iterrows():  # demo only
    out = solve(row["question"])
    preds.append(out)

submission = pd.DataFrame({
    "id": train["id"].head(5),
    "output": preds
})
submission.to_csv("submission.csv", index=False)
print("✅ submission.csv saved")
```

```python
!ls../input
```