# DPC Starter Train

- **Author:** Takamichi Toda
- **Votes:** 453
- **Ref:** takamichitoda/dpc-starter-train
- **URL:** https://www.kaggle.com/code/takamichitoda/dpc-starter-train
- **Last run:** 2025-12-31 11:12:17.393000

---

# Deep Past Initiative – Machine Translation (Training Notebook)

This notebook is a **starter / baseline** for this Kaggle competition.

Main ideas:
- Use **ByT5** to handle noisy Akkadian transliterations at the character level
- Perform **simple sentence alignment** to increase training data
- Fine-tune using HuggingFace `Trainer`


Inference Code is [here](https://www.kaggle.com/code/takamichitoda/dpc-starter-infer).

```python
!pip install evaluate sacrebleu
```

```python
import os
import gc
import re
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM, 
    DataCollatorForSeq2Seq, 
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer
)
from sentence_transformers import SentenceTransformer, util
import evaluate
```

```python
class Config:
    # Akkadian transliteration contains a lot of noise and many unknown words, so
    # ByT5, which processes text at the character (byte) level rather than the word level, is the strongest choice.
    MODEL_NAME = "google/byt5-small" 
    
    # ByT5 tends to produce longer token sequences, but 512 tokens is enough at the sentence level.
    MAX_LENGTH = 512
    
    EPOCHS = 20
    LEARNING_RATE = 1e-4
    OUTPUT_DIR = "./byt5-akkadian-model"
```

```python
# Fix the seed (for reproducibility).
def seed_everything(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    
seed_everything()
```

```python
INPUT_DIR = "/kaggle/input/deep-past-initiative-machine-translation"
train_df = pd.read_csv(f"{INPUT_DIR}/train.csv")
test_df = pd.read_csv(f"{INPUT_DIR}/test.csv")
```

```python
print(f"Original Train Data: {len(train_df)} docs")
```

```python
def simple_sentence_aligner(df):
    """
    【戦略の肝】
    Trainデータの「文書(複数文)」を、Testデータと同じ「文(1文)」に分割します。
    ここでは「英語の文数」と「アッカド語の行数」が一致する場合のみ分割する
    というヒューリスティック（簡易ルール）を使います。
    """
    aligned_data = []
    
    for idx, row in df.iterrows():
        src = str(row['transliteration'])
        tgt = str(row['translation'])
        
        # Split the English text by sentence-ending punctuation.
        tgt_sents = [t.strip() for t in re.split(r'(?<=[.!?])\s+', tgt) if t.strip()]
        
        # Assume the Akkadian text is often separated by newlines and split accordingly.
        src_lines = [s.strip() for s in src.split('\n') if s.strip()]
        
        # If the counts match, trust it as 1-to-1 pairs and use the split version.
        if len(tgt_sents) > 1 and len(tgt_sents) == len(src_lines):
            for s, t in zip(src_lines, tgt_sents):
                if len(s) > 3 and len(t) > 3: # Remove junk/noisy data.
                    aligned_data.append({'transliteration': s, 'translation': t})
        else:
            # If splitting fails (counts don't match), keep the original document pair as-is (safe fallback).
            aligned_data.append({'transliteration': src, 'translation': tgt})
            
    return pd.DataFrame(aligned_data)
```

```python
# Run data augmentation.
train_expanded = simple_sentence_aligner(train_df)
print(f"Expanded Train Data: {len(train_expanded)} sentences (Alignment applied)")

# Convert to Hugging Face Dataset format & split into Train/Val.
dataset = Dataset.from_pandas(train_expanded)
# Create a validation set with test_size=0.1.
split_datasets = dataset.train_test_split(test_size=0.1, seed=42)
# After splitting, the keys are 'train' and 'test' (we'll use 'test' as validation).
```

```python
def create_bidirectional_data(dataset_split):
    df = dataset_split.to_pandas()
    
    # 方向1: Akkadian -> English (元のタスク)
    df_fwd = df.copy()
    df_fwd['input_text'] = "translate Akkadian to English: " + df_fwd['transliteration'].astype(str)
    df_fwd['target_text'] = df_fwd['translation'].astype(str)
    
    # 方向2: English -> Akkadian (逆翻訳タスク)
    df_bwd = df.copy()
    df_bwd['input_text'] = "translate English to Akkadian: " + df_bwd['translation'].astype(str)
    df_bwd['target_text'] = df_bwd['transliteration'].astype(str)
    
    # 結合してシャッフル
    df_combined = pd.concat([df_fwd, df_bwd], ignore_index=True)
    df_combined = df_combined.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return Dataset.from_pandas(df_combined)

def create_unidirectional_data(dataset_split):
    # Validation用: 形式だけ揃える (Akkadian -> Englishのみ)
    df = dataset_split.to_pandas()
    df['input_text'] = "translate Akkadian to English: " + df['transliteration'].astype(str)
    df['target_text'] = df['translation'].astype(str)
    return Dataset.from_pandas(df)

# Trainデータは双方向化 (データ数が2倍になります)
bidirectional_train = create_bidirectional_data(split_datasets["train"])

# Validationデータは単方向のまま (評価のため)
unidirectional_val = create_unidirectional_data(split_datasets["test"])

print(f"Train samples: {len(bidirectional_train)} (Bidirectional)")
print(f"Val samples:   {len(unidirectional_val)} (Unidirectional)")
```

```python
# ==========================================
# 3. Tokenization & preprocessing
# ==========================================
tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)

def preprocess_function(examples):
    # データ作成時にPrefix込みの "input_text" と "target_text" を作っているのでそれを使う
    inputs = [str(ex) for ex in examples["input_text"]]
    targets = [str(ex) for ex in examples["target_text"]]
    
    model_inputs = tokenizer(inputs, max_length=Config.MAX_LENGTH, truncation=True)
    labels = tokenizer(targets, max_length=Config.MAX_LENGTH, truncation=True)
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# 作成した新しいデータセットに対してmapを適用
tokenized_train = bidirectional_train.map(preprocess_function, batched=True)
tokenized_val = unidirectional_val.map(preprocess_function, batched=True)
```

```python
# ==========================================
# 4. Model training (fine-tuning)
# ==========================================
gc.collect()
torch.cuda.empty_cache()
model = AutoModelForSeq2SeqLM.from_pretrained(Config.MODEL_NAME)
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# 評価指標の準備
metric_chrf = evaluate.load("chrf")
metric_bleu = evaluate.load("sacrebleu")

def compute_metrics(eval_preds):
    preds, labels = eval_preds

    # Seq2SeqTrainer/Trainer の返り値が tuple になることがあるので保険
    if isinstance(preds, tuple):
        preds = preds[0]

    # ★ここが重要★
    # preds が logits の場合: (batch, seq, vocab) になりがち → argmax で token ids に変換
    if hasattr(preds, "ndim") and preds.ndim == 3:
        preds = np.argmax(preds, axis=-1)

    # int 化 & 範囲外対策（ByT5 の decode が chr を使うため）
    preds = preds.astype(np.int64)
    preds = np.where(preds < 0, tokenizer.pad_token_id, preds)
    preds = np.clip(preds, 0, tokenizer.vocab_size - 1)

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # 既存の評価計算（あなたの続きの処理に合わせて）
    chrf = metric_chrf.compute(predictions=decoded_preds, references=decoded_labels)["score"]
    bleu = metric_bleu.compute(predictions=decoded_preds, references=[[x] for x in decoded_labels])["score"]

    geo_mean = (chrf * bleu) ** 0.5 if chrf > 0 and bleu > 0 else 0.0
    return {"chrf": chrf, "bleu": bleu, "geo_mean": geo_mean}


args = Seq2SeqTrainingArguments(
    output_dir=Config.OUTPUT_DIR,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=Config.LEARNING_RATE,
    optim="adafactor",
    label_smoothing_factor=0.2,
    
    # === Key fixes ===
    fp16=False,                     # ★Set to False to prevent a NaN error (required).
    per_device_train_batch_size=1,  # ★fp32 uses more memory, so reduce the batch size (8 -> 4).
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,  # ★To compensate, accumulate gradients to keep the effective batch size at 8.
    # ======================
    
    weight_decay=0.01,
    save_total_limit=1,
    num_train_epochs=Config.EPOCHS,
    predict_with_generate=True,
    logging_steps=10,               # Inspect logs in more detail.
    report_to="none",

    #load_best_model_at_end=True,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

print("Starting Training (FP32 mode)...")
trainer.train()
```

```python
# --- Save Model ---
# Important: the model saved here will be loaded in the next notebook.
trainer.save_model(Config.OUTPUT_DIR)
tokenizer.save_pretrained(Config.OUTPUT_DIR)
print(f"Model saved to {Config.OUTPUT_DIR}")
```