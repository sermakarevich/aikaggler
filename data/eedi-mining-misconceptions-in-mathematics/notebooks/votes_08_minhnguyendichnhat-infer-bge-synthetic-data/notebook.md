# Infer BGE Synthetic Data 

- **Author:** Minh Nguyen Dich Nhat
- **Votes:** 320
- **Ref:** minhnguyendichnhat/infer-bge-synthetic-data
- **URL:** https://www.kaggle.com/code/minhnguyendichnhat/infer-bge-synthetic-data
- **Last run:** 2024-09-26 03:21:43.343000

---

```python
import os, math, numpy as np
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
```

```python
%%time
!pip uninstall -y torch
!pip install --no-index --find-links=/kaggle/input/making-wheels-of-necessary-packages-for-vllm vllm
!pip install -U --upgrade /kaggle/input/vllm-t4-fix/grpcio-1.62.2-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
!pip install -U --upgrade /kaggle/input/vllm-t4-fix/ray-2.11.0-cp310-cp310-manylinux2014_x86_64.whl
!pip install --no-deps --no-index /kaggle/input/hf-libraries/sentence-transformers/sentence_transformers-3.1.0-py3-none-any.whl
```

```python
import os
from transformers import AutoTokenizer
import pandas as pd


df_test = pd.read_csv("/kaggle/input/eedi-mining-misconceptions-in-mathematics/test.csv")
tokenizer = AutoTokenizer.from_pretrained("/kaggle/input/hugging-quants-meta-llama-3-1-8b-instruct-awq-int4")

PROMPT  = """Question: {Question}
Incorrect Answer: {IncorrectAnswer}
Correct Answer: {CorrectAnswer}
Construct Name: {ConstructName}
Subject Name: {SubjectName}

Your task: Identify the misconception behind Incorrect Answer. Answer concisely and generically inside <response>$$INSERT TEXT HERE$$</response>.
Before answering the question think step by step concisely in 1-2 sentence inside <thinking>$$INSERT TEXT HERE$$</thinking> tag and respond your final misconception inside <response>$$INSERT TEXT HERE$$</response> tag."""

def apply_template(row, tokenizer):
    messages = [
        {
            "role": "user", 
            "content": PROMPT.format(
                 ConstructName=row["ConstructName"],
                 SubjectName=row["SubjectName"],
                 Question=row["QuestionText"],
                 IncorrectAnswer=row[f"CorrectAnswerText"],
                 CorrectAnswer=row[f"AnswerText"])
        }
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return text
```

```python
df_test
```

```python
def get_correct_answer(row):
    if row['CorrectAnswer'] == 'A':
        return row['AnswerAText']
    elif row['CorrectAnswer'] == 'B':
        return row['AnswerBText']
    elif row['CorrectAnswer'] == 'C':
        return row['AnswerCText']
    elif row['CorrectAnswer'] == 'D':
        return row['AnswerDText']
    else:
        return None

# Apply the function to create the CorrectAnswer column
df_test['CorrectAnswerText'] = df_test.apply(get_correct_answer, axis=1)
```

```python
df_test
```

```python
select_column = ["QuestionId", "ConstructName", "SubjectName", "CorrectAnswer", "QuestionText", "CorrectAnswerText"]
df_answer = pd.melt(df_test, 
                    id_vars=select_column,
                    value_vars=[f"Answer{ans}Text" for ans in ["A", "B", "C", "D"]],
                    var_name="Option",
                    value_name="AnswerText").sort_values("QuestionId")
```

```python
import re
df_answer['Option'] = df_answer['Option'].apply(lambda x: re.search(r'Answer([A-D])', x).group(1) if re.search(r'Answer([A-D])', x) else None)
```

```python
df_answer = df_answer[df_answer['CorrectAnswer'] != df_answer['Option']]
```

```python
df_answer["Prompt"] = df_answer.apply(lambda row: apply_template(row, tokenizer), axis=1)
```

```python
df_answer.to_parquet("test.parquet", index=False)
```

```python
%%writefile run_vllm.py

import re
import vllm
import pandas as pd

df = pd.read_parquet("test.parquet")

llm = vllm.LLM(
    "/kaggle/input/hugging-quants-meta-llama-3-1-8b-instruct-awq-int4",
    quantization="awq",
    tensor_parallel_size=2, 
    gpu_memory_utilization=0.95, 
    trust_remote_code=True,
    dtype="half", 
    enforce_eager=True,
    max_model_len=8192,
    disable_log_stats=True
)
tokenizer = llm.get_tokenizer()


responses = llm.generate(
    df["Prompt"].values,
    vllm.SamplingParams(
        n=1,  # Number of output sequences to return for each prompt.
        top_p=0.9,  # Float that controls the cumulative probability of the top tokens to consider.
        temperature=0,  # randomness of the sampling
        seed=777, # Seed for reprodicibility
        skip_special_tokens=False,  # Whether to skip special tokens in the output.
        max_tokens=2048,  # Maximum number of tokens to generate per output sequence.
    ),
    use_tqdm = True
)

responses = [x.outputs[0].text for x in responses]
df["FullResponse"] = responses

def extract_response(text):
    return ",".join(re.findall(r"<response>(.*?)</response>", text)).strip()

responses = [extract_response(x) for x in responses]
df["Misconception"] = responses
df.to_parquet("output.parquet", index=False)
```

```python
!python run_vllm.py
```

```python
df = pd.read_parquet('/kaggle/working/output.parquet')
```

```python
df
```

```python
def create_text(row):
    text = f"""
    {row["ConstructName"]}
    {row["QuestionText"]}
    Answer: {row["AnswerText"]}
    Misconception: {row["Misconception"]}
    """
    return text
```

```python
df["FullText"] = df.apply(lambda row: create_text(row), axis=1)
```

```python
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("/kaggle/input/train-bge-synthetic-data/trained_model")
```

```python
misconception_mapping = pd.read_csv("/kaggle/input/eedi-mining-misconceptions-in-mathematics/misconception_mapping.csv")
```

```python
test_long_vec = model.encode(
    df["FullText"].values, normalize_embeddings=True
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

```python
df["QuestionId_Answer"] = df.apply(lambda x: str(x["QuestionId"]) + '_' + x["Option"], axis=1)
```

```python
submission = (
    df.assign(
        MisconceptionId=test_sorted_indices[:, :25].tolist()  # Add a column with MisconceptionId
    )
    .assign(
        MisconceptionId=lambda df: df["MisconceptionId"].apply(lambda x: " ".join(map(str, x)))  # Convert list to string
    )
    .loc[:, ["QuestionId_Answer", "MisconceptionId"]]  # Select specific columns
    .sort_values(by="QuestionId_Answer")  # Sort by 'QuestionId_Answer'
)
```

```python
submission.to_csv("submission.csv", index=False)
```