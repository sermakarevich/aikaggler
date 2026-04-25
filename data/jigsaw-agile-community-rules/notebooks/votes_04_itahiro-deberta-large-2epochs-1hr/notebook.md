# deberta-large(2epochs-1hr)

- **Author:** Renoir
- **Votes:** 434
- **Ref:** itahiro/deberta-large-2epochs-1hr
- **URL:** https://www.kaggle.com/code/itahiro/deberta-large-2epochs-1hr
- **Last run:** 2025-10-18 09:33:43.687000

---

```python
# !uv pip install -q googletrans-py==4.0.0   # ← maintained fork, no version clashes
```

```python
# %%writefile rule_augmenter.py
# """
# Create 3 synthetic variants of every rule by round-trip translation.
# Cached in /kaggle/working/synthetic_rules.json so the mapping is
# fixed for both train and (later) inference.
# """
# import json, random, os
# from googletrans import Translator  # lightweight, works offline-cached
# random.seed(42)

# TRANS_LANGS = ['fr', 'de', 'es']      # 3 langs → 3 synthetic rules
# CACHE_FILE  = '/kaggle/working/synthetic_rules.json'
# translator  = Translator()

# def _round_trip(text, lang):
#     """en → lang → en"""
#     return translator.translate(translator.translate(text, dest=lang).text, dest='en').text

# def build_synthetic_rule_map(unique_rules):
#     """Dict original_rule → list of 3 synthetic rules"""
#     if os.path.exists(CACHE_FILE):
#         return json.load(open(CACHE_FILE))
#     synth = {}
#     for rule in unique_rules:
#         synth[rule] = [_round_trip(rule, lg) for lg in TRANS_LANGS]
#     json.dump(synth, open(CACHE_FILE, 'w'), indent=2)
#     return synth
```

```python
%%writefile utils.py
import pandas as pd
import re

# from rule_augmenter import build_synthetic_rule_map

def url_to_semantics(text: str) -> str:
    if not isinstance(text, str):
        return ""

    url_pattern = r'https?://[^\s/$.?#].[^\s]*'
    urls = re.findall(url_pattern, text)
    
    if not urls:
        return "" 

    all_semantics = []
    seen_semantics = set()

    for url in urls:
        url_lower = url.lower()
        
        domain_match = re.search(r"(?:https?://)?([a-z0-9\-\.]+)\.[a-z]{2,}", url_lower)
        if domain_match:
            full_domain = domain_match.group(1)
            parts = full_domain.split('.')
            for part in parts:
                if part and part not in seen_semantics and len(part) > 3: # Avoid short parts like 'www'
                    all_semantics.append(f"domain:{part}")
                    seen_semantics.add(part)

        # 2. Extract path parts
        path = re.sub(r"^(?:https?://)?[a-z0-9\.-]+\.[a-z]{2,}/?", "", url_lower)
        path_parts = [p for p in re.split(r'[/_.-]+', path) if p and p.isalnum()] # Split by common delimiters

        for part in path_parts:
            # Clean up potential file extensions or query params
            part_clean = re.sub(r"\.(html?|php|asp|jsp)$|#.*|\?.*", "", part)
            if part_clean and part_clean not in seen_semantics and len(part_clean) > 3:
                all_semantics.append(f"path:{part_clean}")
                seen_semantics.add(part_clean)

    if not all_semantics:
        return ""

    return f"\nURL Keywords: {' '.join(all_semantics)}"


def get_dataframe_to_train(data_path):
    train_dataset = pd.read_csv(f"{data_path}/train.csv") 
    test_dataset = pd.read_csv(f"{data_path}/test.csv")

    flatten = []

    flatten.append(train_dataset[["body", "rule", "subreddit","rule_violation"]].copy())

    for violation_type in ["positive", "negative"]:
        for i in range(1, 3):
            col_name = f"{violation_type}_example_{i}"
            
            if col_name in train_dataset.columns:
                sub_dataset = train_dataset[[col_name, "rule", "subreddit"]].copy()
                sub_dataset = sub_dataset.rename(columns={col_name: "body"})
                sub_dataset["rule_violation"] = 1 if violation_type == "positive" else 0
                
                sub_dataset.dropna(subset=['body'], inplace=True)
                sub_dataset = sub_dataset[sub_dataset['body'].str.strip().str.len() > 0]
                
                if not sub_dataset.empty:
                    flatten.append(sub_dataset)
    
    for violation_type in ["positive", "negative"]:
        for i in range(1, 3):
            col_name = f"{violation_type}_example_{i}"
            
            if col_name in test_dataset.columns:
                sub_dataset = test_dataset[[col_name, "rule", "subreddit"]].copy()
                sub_dataset = sub_dataset.rename(columns={col_name: "body"})
                sub_dataset["rule_violation"] = 1 if violation_type == "positive" else 0
                
                sub_dataset.dropna(subset=['body'], inplace=True)
                sub_dataset = sub_dataset[sub_dataset['body'].str.strip().str.len() > 0]
                
                if not sub_dataset.empty:
                    flatten.append(sub_dataset)
    
    dataframe = pd.concat(flatten, axis=0)
    dataframe = dataframe.drop_duplicates(subset=['body', 'rule', 'subreddit'], ignore_index=True)
    dataframe.drop_duplicates(subset=['body','rule'],keep='first',inplace=True)
    
    return dataframe.sample(frac=1, random_state=42).reset_index(drop=True)
    
# def get_dataframe_to_train(data_path):
#     """
#     Load TRAIN only for real comments (labels exist),
#     append ALL examples from train + test,
#     create 3 synthetic rules per original rule,
#     return augmented, de-duplicated, shuffled DataFrame.
#     """
#     train_df = pd.read_csv(f"{data_path}/train.csv")
#     test_df  = pd.read_csv(f"{data_path}/test.csv")

#     # 1. synthetic rule map from **union** of rules (train + test)
#     unique_rules = pd.concat([train_df, test_df])['rule'].unique()
#     rule_map     = build_synthetic_rule_map(unique_rules)

#     # 2. real comments → ONLY from train (have labels)
#     base = train_df[["body", "rule", "subreddit", "rule_violation"]].copy()

#     # 3. helper: augment a dataframe (original + 3 synthetic rules per row)
#     def _augment(df):
#         rows = []
#         for _, r in df.iterrows():
#             rows.append(r.copy())                # keep original
#             for syn_rule in rule_map[r['rule']]: # 3 synthetic variants
#                 rr = r.copy()
#                 rr['rule'] = syn_rule
#                 rows.append(rr)
#         return pd.DataFrame(rows)

#     # 4. collect **all example rows** (pos/neg × 1,2) from **both** files
#     flatten = [base]
#     for df in (train_df, test_df):
#         for vt in ["positive", "negative"]:
#             for i in (1, 2):
#                 col = f"{vt}_example_{i}"
#                 if col not in df.columns:
#                     continue
#                 tmp = df[[col, "rule", "subreddit"]].rename(columns={col: "body"})
#                 tmp["rule_violation"] = 1 if vt == "positive" else 0
#                 tmp = tmp.dropna(subset=["body"])
#                 tmp = tmp[tmp["body"].str.strip().str.len() > 0]
#                 if not tmp.empty:
#                     flatten.append(tmp)

#     # 5. augment the whole pool
#     pool = pd.concat(flatten, ignore_index=True)
#     pool = _augment(pool)          # synthetic rules added here

#     # 6. de-duplicate & shuffle
#     pool = pool.drop_duplicates(subset=["body", "rule", "subreddit"], ignore_index=True)
#     pool = pool.drop_duplicates(subset=["body"], keep="first")
#     return pool.sample(frac=1, random_state=42).reset_index(drop=True)
```

```python
%%writefile train_deberta.py
import os
import pandas as pd
import torch
import random
import numpy as np
from sklearn.model_selection import train_test_split 
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)

from utils import get_dataframe_to_train, url_to_semantics

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class CFG:
    model_name_or_path = "/kaggle/input/huggingfacedebertav3variants/deberta-v3-base"
    data_path = "/kaggle/input/jigsaw-agile-community-rules/"
    output_dir = "./deberta_v3_small_final_model"
  
    EPOCHS = 3
    LEARNING_RATE = 2e-5  
    
    MAX_LENGTH = 512
    BATCH_SIZE = 8

class JigsawDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels:
            item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings['input_ids'])

def main():
    seed_everything(42)
    training_data_df = get_dataframe_to_train(CFG.data_path)
    # training_data_df, valid_df = train_test_split(full_df,test_size=0.2,stratify=full_df['rule'],random_state=42)
    print(f"Training dataset (from examples only) size: {len(training_data_df)}")

    test_df_for_prediction = pd.read_csv(f"{CFG.data_path}/test.csv")
    
    training_data_df['body_with_url'] = training_data_df['body'].apply(lambda x: x + url_to_semantics(x))
    training_data_df['input_text'] = training_data_df['rule'] + "[SEP]" + training_data_df['body_with_url']

    tokenizer = AutoTokenizer.from_pretrained(CFG.model_name_or_path)
    train_encodings = tokenizer(training_data_df['input_text'].tolist(), truncation=True, padding=True, max_length=CFG.MAX_LENGTH)
    train_labels = training_data_df['rule_violation'].tolist()
    train_dataset = JigsawDataset(train_encodings, train_labels)

    model = AutoModelForSequenceClassification.from_pretrained(CFG.model_name_or_path, num_labels=2)
    
    training_args = TrainingArguments(
        output_dir=CFG.output_dir,
        num_train_epochs=CFG.EPOCHS,
        learning_rate=CFG.LEARNING_RATE,
        per_device_train_batch_size=CFG.BATCH_SIZE,
        warmup_ratio=0.1,
        weight_decay=0.01,
        report_to="none",
        save_strategy="no",  #这一行加上这个 save_strategy="no"
        logging_steps=1,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )
    
    trainer.train()

    test_df_for_prediction['body_with_url'] = test_df_for_prediction['body'].apply(lambda x: x + url_to_semantics(x))
    test_df_for_prediction['input_text'] = test_df_for_prediction['rule'] + "[SEP]" + test_df_for_prediction['body_with_url']
    
    test_encodings = tokenizer(test_df_for_prediction['input_text'].tolist(), truncation=True, padding=True, max_length=CFG.MAX_LENGTH)
    test_dataset = JigsawDataset(test_encodings)
    
    predictions = trainer.predict(test_dataset)
    probs = torch.nn.functional.softmax(torch.tensor(predictions.predictions), dim=1)[:, 1].numpy()

    submission_df = pd.DataFrame({
        "row_id": test_df_for_prediction["row_id"],
        "rule_violation": probs
    })
    submission_df.to_csv("submission_deberta.csv", index=False)

if __name__ == "__main__":
    main()
```

```python
!python train_deberta.py
```

```python
%%writefile train_distilroberta.py
import os
import pandas as pd
import torch
import random
import numpy as np
from sklearn.model_selection import train_test_split 
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)

from utils import get_dataframe_to_train, url_to_semantics

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class CFG:
    model_name_or_path = "/kaggle/input/distilroberta-base/distilroberta-base"
    data_path = "/kaggle/input/jigsaw-agile-community-rules/"
    output_dir = "./deberta_v3_small_final_model"
  
    EPOCHS = 3
    LEARNING_RATE = 2e-5  
    
    MAX_LENGTH = 512
    BATCH_SIZE = 8

class JigsawDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels:
            item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings['input_ids'])

def main():
    seed_everything(42)
    training_data_df = get_dataframe_to_train(CFG.data_path)
    # training_data_df, valid_df = train_test_split(full_df,test_size=0.2,stratify=full_df['rule'],random_state=42)
    print(f"Training dataset (from examples only) size: {len(training_data_df)}")

    test_df_for_prediction = pd.read_csv(f"{CFG.data_path}/test.csv")
    
    training_data_df['body_with_url'] = training_data_df['body'].apply(lambda x: x + url_to_semantics(x))
    training_data_df['input_text'] = training_data_df['rule'] + "[SEP]" + training_data_df['body_with_url']

    tokenizer = AutoTokenizer.from_pretrained(CFG.model_name_or_path)
    train_encodings = tokenizer(training_data_df['input_text'].tolist(), truncation=True, padding=True, max_length=CFG.MAX_LENGTH)
    train_labels = training_data_df['rule_violation'].tolist()
    train_dataset = JigsawDataset(train_encodings, train_labels)

    model = AutoModelForSequenceClassification.from_pretrained(CFG.model_name_or_path, num_labels=2)
    
    training_args = TrainingArguments(
        output_dir=CFG.output_dir,
        num_train_epochs=CFG.EPOCHS,
        learning_rate=CFG.LEARNING_RATE,
        per_device_train_batch_size=CFG.BATCH_SIZE,
        warmup_ratio=0.1,
        weight_decay=0.01,
        report_to="none",
        save_strategy="no",  #这一行加上这个 save_strategy="no"
        logging_steps=1,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )
    
    trainer.train()

    test_df_for_prediction['body_with_url'] = test_df_for_prediction['body'].apply(lambda x: x + url_to_semantics(x))
    test_df_for_prediction['input_text'] = test_df_for_prediction['rule'] + "[SEP]" + test_df_for_prediction['body_with_url']
    
    test_encodings = tokenizer(test_df_for_prediction['input_text'].tolist(), truncation=True, padding=True, max_length=CFG.MAX_LENGTH)
    test_dataset = JigsawDataset(test_encodings)
    
    predictions = trainer.predict(test_dataset)
    probs = torch.nn.functional.softmax(torch.tensor(predictions.predictions), dim=1)[:, 1].numpy()

    submission_df = pd.DataFrame({
        "row_id": test_df_for_prediction["row_id"],
        "rule_violation": probs
    })
    submission_df.to_csv("submission_distilroberta.csv", index=False)

if __name__ == "__main__":
    main()
```

```python
!python train_distilroberta.py
```

```python
%%writefile train_deberta_auc.py
import os
import pandas as pd
import torch
import random
import numpy as np
from sklearn.model_selection import train_test_split 
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)

from utils import get_dataframe_to_train, url_to_semantics

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class CFG:
    model_name_or_path = "/kaggle/input/huggingfacedebertav3variants/deberta-v3-base"
    data_path = "/kaggle/input/jigsaw-agile-community-rules/"
    output_dir = "./deberta_v3_small_final_model"
  
    EPOCHS = 3
    LEARNING_RATE = 4e-5  
    
    MAX_LENGTH = 512
    BATCH_SIZE = 12

class JigsawDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels:
            item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings['input_ids'])

def main():
    seed_everything(42)
    training_data_df = get_dataframe_to_train(CFG.data_path)
    # training_data_df, valid_df = train_test_split(full_df,test_size=0.2,stratify=full_df['rule'],random_state=42)
    print(f"Training dataset (from examples only) size: {len(training_data_df)}")

    test_df_for_prediction = pd.read_csv(f"{CFG.data_path}/test.csv")
    
    training_data_df['body_with_url'] = training_data_df['body'].apply(lambda x: x + url_to_semantics(x))
    training_data_df['input_text'] = training_data_df['rule'] + "[SEP]" + training_data_df['body_with_url']

    tokenizer = AutoTokenizer.from_pretrained(CFG.model_name_or_path)
    train_encodings = tokenizer(training_data_df['input_text'].tolist(), truncation=True, padding=True, max_length=CFG.MAX_LENGTH)
    train_labels = training_data_df['rule_violation'].tolist()
    train_dataset = JigsawDataset(train_encodings, train_labels)

    model = AutoModelForSequenceClassification.from_pretrained(CFG.model_name_or_path, num_labels=2)
    
    training_args = TrainingArguments(
        output_dir=CFG.output_dir,
        num_train_epochs=CFG.EPOCHS,
        learning_rate=CFG.LEARNING_RATE,
        per_device_train_batch_size=CFG.BATCH_SIZE,
        warmup_ratio=0.1,
        weight_decay=0.01,
        report_to="none",
        save_strategy="no",  #这一行加上这个 save_strategy="no"
        logging_steps=1,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )
    
    trainer.train()

    test_df_for_prediction['body_with_url'] = test_df_for_prediction['body'].apply(lambda x: x + url_to_semantics(x))
    test_df_for_prediction['input_text'] = test_df_for_prediction['rule'] + "[SEP]" + test_df_for_prediction['body_with_url']
    
    test_encodings = tokenizer(test_df_for_prediction['input_text'].tolist(), truncation=True, padding=True, max_length=CFG.MAX_LENGTH)
    test_dataset = JigsawDataset(test_encodings)
    
    predictions = trainer.predict(test_dataset)
    probs = torch.nn.functional.softmax(torch.tensor(predictions.predictions), dim=1)[:, 1].numpy()

    submission_df = pd.DataFrame({
        "row_id": test_df_for_prediction["row_id"],
        "rule_violation": probs
    })
    submission_df.to_csv("submission_debertaauc.csv", index=False)

if __name__ == "__main__":
    main()
```

```python
!python train_deberta_auc.py
```

```python
import os
import pandas as pd
```

```python
%%writefile constants.py
EMBDEDDING_MODEL_PATH = "/kaggle/input/qwen-3-embedding/transformers/0.6b/1"
MODEL_OUTPUT_PATH = '/kaggle/input/qwen3-8b-embedding'
DATA_PATH = "/kaggle/input/jigsaw-agile-community-rules"

# https://huggingface.co/Qwen/Qwen3-Embedding-0.6B/blob/main/config_sentence_transformers.json
EMBEDDING_MODEL_QUERY = "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:"

CLEAN_TEXT = True
TOP_K = 2000
BATCH_SIZE = 128
```

```python
%%writefile utils.py
import pandas as pd
import torch.distributed as dist

from datasets import Dataset
from cleantext import clean
from tqdm.auto import tqdm

from constants import CLEAN_TEXT


def build_prompt(row):
    return f"""r/{row["subreddit"]}\nComment: {row["body"]}"""


def cleaner(text):
    return clean(
        text,
        fix_unicode=True,
        to_ascii=True,
        lower=False,
        no_line_breaks=False,
        no_urls=True,
        no_emails=True,
        no_phone_numbers=True,
        no_numbers=False,
        no_digits=False,
        no_currency_symbols=False,
        no_punct=False,
        replace_with_url="<URL>",
        replace_with_email="<EMAIL>",
        replace_with_phone_number="<PHONE>",
        lang="en",
    )



def get_dataframe_to_train(data_path):
    train_dataset = pd.read_csv(f"{data_path}/train.csv")
    test_dataset = pd.read_csv(f"{data_path}/test.csv").sample(frac=0.6, random_state=42).reset_index(drop=True)

    flatten = []
    flatten.append(train_dataset[["body", "rule", "subreddit", "rule_violation"]])
    
    for violation_type in ["positive", "negative"]:
        for i in range(1, 3):
            sub_dataset = test_dataset[[f"{violation_type}_example_{i}", "rule", "subreddit"]].copy()
            sub_dataset = sub_dataset.rename(columns={f"{violation_type}_example_{i}": "body"})
            sub_dataset["rule_violation"] = 1 if violation_type == "positive" else 0
            flatten.append(sub_dataset)

    dataframe = pd.concat(flatten, axis=0)    
    dataframe = dataframe.drop_duplicates(ignore_index=True)
    return dataframe


def prepare_dataframe(dataframe):
    dataframe["prompt"] = dataframe.apply(build_prompt, axis=1)

    
    if CLEAN_TEXT:
        tqdm.pandas(desc="cleaner")
        dataframe["prompt"] = dataframe["prompt"].progress_apply(cleaner)

    if "rule_violation" in dataframe.columns:
        dataframe["rule_violation"] = dataframe["rule_violation"].map(
            {
                1: 1,
                0: -1,
            }
        )

    return dataframe
```

```python
%%writefile semantic.py
import pandas as pd
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import semantic_search, dot_score
from tqdm.auto import tqdm
from peft import PeftModel, PeftConfig


from utils import get_dataframe_to_train, prepare_dataframe
from constants import DATA_PATH, EMBDEDDING_MODEL_PATH, EMBEDDING_MODEL_QUERY, TOP_K, BATCH_SIZE, MODEL_OUTPUT_PATH



def get_scores(test_dataframe):
    corpus_dataframe = get_dataframe_to_train(DATA_PATH)
    corpus_dataframe = prepare_dataframe(corpus_dataframe)
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(EMBDEDDING_MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(EMBDEDDING_MODEL_PATH)
    
    # Load adapter configuration and model
    adapter_config = PeftConfig.from_pretrained(MODEL_OUTPUT_PATH)
    lora_model = PeftModel.from_pretrained(model, MODEL_OUTPUT_PATH, config=adapter_config)
    merged_model = lora_model.merge_and_unload()
    tokenizer.save_pretrained("Qwen3Emb_Finetuned")
    merged_model.save_pretrained("Qwen3Emb_Finetuned")

    # 4. Tạo lại SentenceTransformer từ encoder đã merge
    embedding_model = SentenceTransformer(model_name_or_path="Qwen3Emb_Finetuned", device="cuda")

    print('Done loading model!')

    result = []
    for rule in tqdm(test_dataframe["rule"].unique(), desc=f"Generate scores for each rule"):
        test_dataframe_part = test_dataframe.query("rule == @rule").reset_index(drop=True)
        corpus_dataframe_part = corpus_dataframe.query("rule == @rule").reset_index(drop=True)
        corpus_dataframe_part = corpus_dataframe_part.reset_index(names="row_id")
        
        query_embeddings = embedding_model.encode(
            sentences=test_dataframe_part["prompt"].tolist(),
            prompt=EMBEDDING_MODEL_QUERY,
            batch_size=BATCH_SIZE,
            show_progress_bar=True,
            convert_to_tensor=True,
            device="cuda",
            normalize_embeddings=True,
        )
        document_embeddings = embedding_model.encode(
            sentences=corpus_dataframe_part["prompt"].tolist(),
            batch_size=BATCH_SIZE,
            show_progress_bar=True,
            convert_to_tensor=True,
            device="cuda",
            normalize_embeddings=True,
        )
        test_dataframe_part["semantic"] = semantic_search(
            query_embeddings,
            document_embeddings,
            top_k=TOP_K,
            score_function=dot_score,
        )
        def get_score(semantic):
            semantic = pd.DataFrame(semantic)
            semantic = semantic.merge(
                corpus_dataframe_part[["row_id", "rule_violation"]],
                how="left",
                left_on="corpus_id",
                right_on="row_id",
            )
            semantic["score"] = semantic["score"]*semantic["rule_violation"]
            return semantic["score"].sum()
            
        tqdm.pandas(desc=f"Add label for {rule=}")
        test_dataframe_part["rule_violation"] = test_dataframe_part["semantic"].progress_apply(get_score)
        result.append(test_dataframe_part[["row_id", "rule_violation"]].copy())
        
    submission = pd.concat(result, axis=0)
    return submission


def generate_submission():
    test_dataframe = pd.read_csv(f"{DATA_PATH}/test.csv")
    test_dataframe = prepare_dataframe(test_dataframe)
    
    submission = get_scores(test_dataframe)
    submission = test_dataframe[["row_id"]].merge(submission, on="row_id", how="left")
    submission.to_csv("submission_qwen3.csv", index=False)


if __name__ == "__main__":
    generate_submission()
```

```python
!python semantic.py
```

```python
%%writefile infer_qwen.py

import os
import pandas as pd
from logits_processor_zoo.vllm import MultipleChoiceLogitsProcessor
import torch
import vllm
import numpy as np
from vllm.lora.request import LoRARequest
import argparse
from scipy.special import softmax
df = pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/test.csv")

MODEL_NAME = "/kaggle/input/qwen2.5/transformers/14b-instruct-gptq-int4/1"
LORA_PATH = "/kaggle/input/lora_14b_gptq_1epoch_r32/keras/default/1"
if __name__=='__main__':
    os.environ["VLLM_USE_V1"] = "0"

    llm = vllm.LLM(
        MODEL_NAME,
        # quantization='awq',
        quantization='gptq',
        tensor_parallel_size=torch.cuda.device_count(),
        gpu_memory_utilization=0.98,
        trust_remote_code=True,
        dtype="half",
        enforce_eager=True,
        max_model_len=2836,
        disable_log_stats=True,
        enable_prefix_caching=True,
        enable_lora=True,
        max_lora_rank=32
    )
    tokenizer = llm.get_tokenizer()
    SYS_PROMPT = """
You are given a comment on reddit. Your task is to classify if it violates the given rule. Only respond Yes/No.
"""
    
    prompts = []
    for i, row in df.iterrows():
        text = f"""
    r/{row.subreddit}
    Rule: {row.rule}
    
    1) {row.positive_example_1}
    Violation: Yes
    
    2) {row.positive_example_2}
    Violation: Yes
    
    3) {row.negative_example_1}
    Violation: No
    
    4) {row.negative_example_2}
    Violation: No
    
    5) {row.body}
    """
        
        messages = [
            {"role": "system", "content": SYS_PROMPT},
            {"role": "user", "content": text}
        ]
    
        prompt = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        ) + "Answer:"
        prompts.append(prompt)
    
    df["prompt"] = prompts
    
    mclp = MultipleChoiceLogitsProcessor(tokenizer, choices=['Yes','No'])
    outputs = llm.generate(
        prompts,
        vllm.SamplingParams(
            skip_special_tokens=True,
            max_tokens=1,
            logits_processors=[mclp],
            logprobs=2,
        ),
        use_tqdm=True,
        lora_request=LoRARequest("default", 1, LORA_PATH)
    )
    logprobs = [
        {lp.decoded_token: lp.logprob for lp in out.outputs[0].logprobs[0].values()}
        for out in outputs
    ]
    logit_matrix = pd.DataFrame(logprobs)[['Yes','No']]
    df = pd.concat([df, logit_matrix], axis=1)
    
    df[['Yes',"No"]] = df[['Yes',"No"]].apply(lambda x: softmax(x.values), axis=1, result_type="expand")
    df["pred"] = df["Yes"]
    df['rule_violation'] = df["pred"]
    df[['row_id', 'rule_violation']].to_csv("submission_qwen14b.csv",index=False)
    pd.read_csv('submission_qwen14b.csv')
```

```python
!python infer_qwen.py
```

```python
import pandas as pd
import numpy as np

q = pd.read_csv('submission_deberta.csv')
l = pd.read_csv('submission_qwen3.csv')
m = pd.read_csv('submission_distilroberta.csv')
w = pd.read_csv('submission_qwen14b.csv')
#d = pd.read_csv('submission_distilbert.csv')
#s = pd.read_csv('submission_debertsmall.csv')
a = pd.read_csv('submission_debertaauc.csv')

rq = q['rule_violation'].rank(method='average') / (len(q)+1)
rl = l['rule_violation'].rank(method='average') / (len(l)+1)
rm = m['rule_violation'].rank(method='average') / (len(m)+1)
rw = w['rule_violation'].rank(method='average') / (len(w)+1)
#rd = d['rule_violation'].rank(method='average') / (len(d)+1)
#rs = s['rule_violation'].rank(method='average') / (len(s)+1)
ra = a['rule_violation'].rank(method='average') / (len(a)+1)

blend = 0.5*rq + 0.1*rl + 0.1*rm + 0.1*rw + 0.2*ra # or tune the rank-weights with a tiny grid using OOF
q['rule_violation'] = blend
q.to_csv('/kaggle/working/submission.csv', index=False)
```

```python
import pandas as pd
pd.read_csv('/kaggle/working/submission.csv')
```

```python
%%writefile probe_submission.py

import pandas as pd
import numpy as np
import argparse
import os

def run_probe(input_file: str, output_file: str, num_blocks: int, invert_block: int = -1, invert_all: bool = False):

    print(f"Reading submission file from: {input_file}")
    df = pd.read_csv(input_file)
 
    df['rule_violation'] = pd.to_numeric(df['rule_violation'])
    
    n_rows = len(df)
    print(f"Total rows: {n_rows}")

    if invert_all:
        print("Mode: Invert All. Transforming p -> 1-p for all predictions.")
        df['rule_violation'] = 1 - df['rule_violation']
    
    elif invert_block > 0:
        if not (1 <= invert_block <= num_blocks):
            raise ValueError(f"invert_block must be between 1 and {num_blocks}")
            
        print(f"Mode: Invert Gain Probe. Testing block {invert_block}/{num_blocks}.")
        print("Step 1: Inverting all predictions (p -> 1-p).")
        df['rule_violation'] = 1 - df['rule_violation']
        
        # 计算区块的边界
        block_size = n_rows / num_blocks
        start_index = int((invert_block - 1) * block_size)
        end_index = int(invert_block * block_size) if invert_block < num_blocks else n_rows
        
        print(f"Step 2: Reverting block {invert_block} (indices {start_index} to {end_index-1}) back to original (1-p -> p).")
        
        df.iloc[start_index:end_index, df.columns.get_loc('rule_violation')] = 1 - df.iloc[start_index:end_index]['rule_violation']

    else:
        print("Mode: Standard copy. No transformation applied.")

    print(f"Saving probed submission to: {output_file}")
    df.to_csv(output_file, index=False)
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kaggle Submission Probing Tool")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the baseline submission CSV file.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the new submission CSV file.")
    parser.add_argument("--num_blocks", type=int, default=10, help="Total number of blocks to split the test set into.")
    
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--invert_all", action="store_true", help="Invert all predictions (p -> 1-p).")
    mode_group.add_argument("--invert_block", type=int, default=-1, help="The block number (1-based) to revert back to original after a full inversion.")

    args = parser.parse_args()
    
    run_probe(args.input_file, args.output_file, args.num_blocks, args.invert_block, args.invert_all)
```

```python
#!python probe_submission.py \
#    --input_file submission_qwen3.csv \
#    --output_file submission.csv \
#    --num_blocks 10 \
#    --invert_block 10
```