# [30 min] Just use semantic search. Qwen3-emb-0.6B

- **Author:** Kirill Tushin
- **Votes:** 266
- **Ref:** neibyr/30-min-just-use-semantic-search-qwen3-emb-0-6b
- **URL:** https://www.kaggle.com/code/neibyr/30-min-just-use-semantic-search-qwen3-emb-0-6b
- **Last run:** 2025-08-13 18:49:22.713000

---

# 🔍 Just semantic search

A simple semantic search example without training that runs in about **30–40 minutes** and shows strong performance on the LB.

# 📌 Approach

This approach follows the same data preparation idea as in [my previous notebook](https://www.kaggle.com/code/neibyr/lb-0-878-train-on-test-data-qwen2-5-0-5b):

- Take all `train.csv` comments with their `rule_violation` labels.
- Add all `positive_example_*` and `negative_example_*` from the `test.csv`, mapped to +1 / -1.
- Embed each comment in corpus by using a Qwen3-embedding-0.6b model.
- Retrieve its nearest neighbors from the corpus **within the same rule**.
- Compute a score as the sum of `(similarity × label)` over the top-K matches.

# 💡 Highlights

- No model training required — only inference with an embedding model.
- Fast runtime — ~**30–40 minutes** for the whole pipeline.
- Uses rule-specific neighbor search to reduce noise.
- Optional text cleaning with `clean-text` for more consistent embeddings.
- Surprisingly competitive results for such a lightweight method.

# 🚀 How to Use Beyond This Notebook

This semantic search setup can be a useful building block in other contexts:

- **RAG pipelines** — as a fast and lightweight retrieval component for grounding LLM answers.
- **Ensembles** — combine its scores with model predictions for more robust results (make sure to scale them appropriately).
- **Feature generation** — nearest-neighbor similarity scores can be used as additional features in ML models.

```python
import os
import pandas as pd
```

```python
%%writefile constants.py
EMBDEDDING_MODEL_PATH = "/kaggle/input/qwen-3-embedding/transformers/0.6b/1"
DATA_PATH = "/kaggle/input/jigsaw-agile-community-rules"

# https://huggingface.co/Qwen/Qwen3-Embedding-0.6B/blob/main/config_sentence_transformers.json
EMBEDDING_MODEL_QUERY = "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:"

CLEAN_TEXT = True
TOP_K = 1000
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
    test_dataset = pd.read_csv(f"{data_path}/test.csv")

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

from sentence_transformers import SentenceTransformer
from sentence_transformers.util import semantic_search, dot_score
from tqdm.auto import tqdm

from utils import get_dataframe_to_train, prepare_dataframe
from constants import DATA_PATH, EMBDEDDING_MODEL_PATH, EMBEDDING_MODEL_QUERY, TOP_K, BATCH_SIZE



def get_scores(test_dataframe):
    corpus_dataframe = get_dataframe_to_train(DATA_PATH)
    corpus_dataframe = prepare_dataframe(corpus_dataframe)
    
    embedding_model = SentenceTransformer(
        model_name_or_path=EMBDEDDING_MODEL_PATH,
        device="cuda",
    )

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
            semantic["score"] = semantic["score"] ** 2
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
    submission.to_csv("submission.csv", index=False)


if __name__ == "__main__":
    generate_submission()
```

```python
from semantic import generate_submission
generate_submission()
```

```python
pd.read_csv("submission.csv")
```