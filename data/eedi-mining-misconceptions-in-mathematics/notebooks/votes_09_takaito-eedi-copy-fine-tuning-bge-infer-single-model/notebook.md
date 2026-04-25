# [Eedi] Copy Fine-tuning bge Infer (Single Model)

- **Author:** takaito
- **Votes:** 303
- **Ref:** takaito/eedi-copy-fine-tuning-bge-infer-single-model
- **URL:** https://www.kaggle.com/code/takaito/eedi-copy-fine-tuning-bge-infer-single-model
- **Last run:** 2024-10-02 15:29:34.920000

---

# Overview
- [Original Notebook](https://www.kaggle.com/code/sinchir0/fine-tuning-bge-infer/notebook?scriptVersionId=197677612) by [sinchir0](https://www.kaggle.com/sinchir0) 
- [Original Train Notebook](https://www.kaggle.com/code/sinchir0/fine-tuning-bge-train) by [sinchir0](https://www.kaggle.com/sinchir0) 

- make 25 retrieval data by `bge-large-en-v1.5`
- Fine-tuning `bge-large-en-v1.5` by retrieval data
  - `anchor`: `ConstructName` + `SubjectName` + `QuestionText` + `Answer[A-D]Text`
  - `positive`: Correct MisconceptionName
  - `negative`: Wrong MisconceptionName

ref: https://sbert.net/docs/sentence_transformer/training_overview.html#trainer

# Comment
This idea and notebook is really great.

Thanks to [sinchir0](https://www.kaggle.com/sinchir0) for sharing the great baseline notebook.

I also made a few adjustments and tried different studies and found that I could score around the current 10th place.

The loss of train data is decreasing, but since there is no validation data, it is unclear when to stop learning properly.

The number of subs per day is limited.

It is possible to examine hyperparameters with the data at hand by splitting the data appropriately.

In order not to break LB, I will not publish the trained model.

I wanted to share that this approach is a good approach, and I published this notebook as evidence.

However, since the 1st place has an overwhelming score of 0.444, it is necessary to try a different approach separately.

I hope this will be of some help to you.

# Setting

```python
DATA_PATH = "/kaggle/input/eedi-mining-misconceptions-in-mathematics"
MODEL_PATH = "/kaggle/input/eedi-models/bge-large_emb-seed42-Ver1/bge-large_emb-seed42-Ver1"
```

# Install

```python
!pip uninstall -qq -y \
polars
```

```python
!python -m pip install -qq --no-index --find-links=/kaggle/input/eedi-library \
polars\
sentence-transformers
```

# Import

```python
import os

import polars as pl
import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
```

```python
import sentence_transformers

assert pl.__version__ == "1.7.1"
assert sentence_transformers.__version__ == "3.1.0"
```

# Data Load

```python
test = pl.read_csv(f"{DATA_PATH}/test.csv")
misconception_mapping = pl.read_csv(f"{DATA_PATH}/misconception_mapping.csv")
```

# Preprocess

```python
common_col = [
    "QuestionId",
    "ConstructName",
    "SubjectName",
    "QuestionText",
    "CorrectAnswer",
]

test_long = (
    test
    .select(
        pl.col(common_col + [f"Answer{alpha}Text" for alpha in ["A", "B", "C", "D"]])
    )
    .unpivot(
        index=common_col,
        variable_name="AnswerType",
        value_name="AnswerText",
    )
    .with_columns(
        pl.concat_str(
            [
                pl.col("ConstructName"),
                pl.col("SubjectName"),
                pl.col("QuestionText"),
                pl.col("AnswerText"),
            ],
            separator=" ",
        ).alias("AllText"),
        pl.col("AnswerType").str.extract(r"Answer([A-D])Text$").alias("AnswerAlphabet"),
    )
    .with_columns(
        pl.concat_str(
            [pl.col("QuestionId"), pl.col("AnswerAlphabet")], separator="_"
        ).alias("QuestionId_Answer"),
    )
    .sort("QuestionId_Answer")
)
test_long.head()
```

```python
# I am not familiar with polars, I convert it once to pandas and process it...
test_long = test_long.to_pandas()
test_long['AllText'] = '<Construct> ' + test_long['ConstructName'] + ' <Subject> ' + test_long['SubjectName'] + ' <Question> ' + test_long['QuestionText'] + ' <Answer> ' + test_long['AnswerText']
test_long = pl.from_pandas(test_long)
```

# BGE

```python
model = SentenceTransformer(MODEL_PATH)

test_long_vec = model.encode(
    test_long["AllText"].to_list(), normalize_embeddings=True
)
misconception_mapping_vec = model.encode(
    misconception_mapping["MisconceptionName"].to_list(), normalize_embeddings=True
)
print(test_long_vec.shape)
print(misconception_mapping_vec.shape)
```

```python
test_cos_sim_arr = cosine_similarity(test_long_vec, misconception_mapping_vec)
test_sorted_indices = np.argsort(-test_cos_sim_arr, axis=1)
```

# Make Submit File

```python
submission = (
    test_long.with_columns(
        pl.Series(test_sorted_indices[:, :25].tolist()).alias("MisconceptionId")
    )
    .with_columns(
        pl.col("MisconceptionId").map_elements(
            lambda x: " ".join(map(str, x)), return_dtype=pl.String
        )
    ).filter(
        pl.col("CorrectAnswer") != pl.col("AnswerAlphabet")
    ).select(
        pl.col(["QuestionId_Answer", "MisconceptionId"])
    ).sort("QuestionId_Answer")
)
```

```python
submission.head()
```

```python
submission.write_csv("submission.csv")
```