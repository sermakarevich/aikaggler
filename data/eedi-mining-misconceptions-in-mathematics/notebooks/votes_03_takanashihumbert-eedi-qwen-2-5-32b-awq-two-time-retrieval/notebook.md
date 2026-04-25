# Eedi Qwen-2.5 32B AWQ two-time retrieval

- **Author:** Joseph
- **Votes:** 650
- **Ref:** takanashihumbert/eedi-qwen-2-5-32b-awq-two-time-retrieval
- **URL:** https://www.kaggle.com/code/takanashihumbert/eedi-qwen-2-5-32b-awq-two-time-retrieval
- **Last run:** 2024-10-22 01:39:32.020000

---

* The main idea of this notebook is using retrieval two times.
  * The first time: Get the top-K1 relavent misconceptions to LLM as a reference(using ConstructName + SubjectName).
  * The second time: Get the top-K2(K2 < K1) relavent misconceptions(using ConstructName + SubjectName + Question + Answer + LLM's output).
  * Inference time: ~2 hours
 

Thanks to these great works:
- [Zero-shot w/ LLM feature (LB: 0.180)](https://www.kaggle.com/code/ubamba98/eedi-zero-shot-w-llm-feature-lb-0-180)
- [Infer BGE Synthetic Data](https://www.kaggle.com/code/minhnguyendichnhat/infer-bge-synthetic-data)
- [Fine-tuning bge Train](https://www.kaggle.com/code/sinchir0/fine-tuning-bge-train)

```python
%%time
!pip uninstall -y torch
!pip install -q --no-index --find-links=/kaggle/input/making-wheels-of-necessary-packages-for-vllm vllm
!pip install -q -U --upgrade /kaggle/input/vllm-t4-fix/grpcio-1.62.2-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
!pip install -q -U --upgrade /kaggle/input/vllm-t4-fix/ray-2.11.0-cp310-cp310-manylinux2014_x86_64.whl
!pip install -q --no-deps --no-index /kaggle/input/hf-libraries/sentence-transformers/sentence_transformers-3.1.0-py3-none-any.whl
```

```python
import os, math, numpy as np
import os
from transformers import AutoTokenizer
import pandas as pd
from tqdm import tqdm
import re, gc
import torch
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
pd.set_option('display.max_rows', 300)
```

## Metric

```python
%%writefile eedi_metrics.py

# Credit: https://www.kaggle.com/code/abdullahmeda/eedi-map-k-metric

import numpy as np
def apk(actual, predicted, k=25):
    """
    Computes the average precision at k.
    
    This function computes the average prescision at k between two lists of
    items.
    
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
        
    Returns
    -------
    score : double
            The average precision at k over the input lists
    """
    
    if not actual:
        return 0.0

    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        # first condition checks whether it is valid prediction
        # second condition checks if prediction is not repeated
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    return score / min(len(actual), k)

def mapk(actual, predicted, k=25):
    """
    Computes the mean average precision at k.
    
    This function computes the mean average prescision at k between two lists
    of lists of items.
    
    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted 
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements
        
    Returns
    -------
    score : double
            The mean average precision at k over the input lists
    """
    
    return np.mean([apk(a,p,k) for a,p in zip(actual, predicted)])
```

## Prepare dataframe

```python
IS_SUBMISSION = bool(os.getenv("KAGGLE_IS_COMPETITION_RERUN"))

model_path = "/kaggle/input/qwen2.5/transformers/32b-instruct-awq/1"
tokenizer = AutoTokenizer.from_pretrained(model_path)

df_train = pd.read_csv("/kaggle/input/eedi-mining-misconceptions-in-mathematics/train.csv").fillna(-1).sample(100, random_state=42).reset_index(drop=True)
df_test = pd.read_csv("/kaggle/input/eedi-mining-misconceptions-in-mathematics/test.csv")
```

# first retrieval

```python
import pandas as pd
from sentence_transformers import SentenceTransformer, util

if not IS_SUBMISSION:
    df_ret = df_train.copy()
else:
    df_ret = df_test.copy()
df_misconception_mapping = pd.read_csv("/kaggle/input/eedi-mining-misconceptions-in-mathematics/misconception_mapping.csv")

model = SentenceTransformer('/kaggle/input/eedi-finetuned-bge-public/Eedi-finetuned-bge')
df_ret.head()
```

```python
def preprocess_text(x):
    x = x.lower()                 # Convert words to lowercase
    x = re.sub("@\w+", '',x)      # Delete strings starting with @
    #x = re.sub("'\d+", '',x)      # Delete Numbers
    x = re.sub("http\w+", '',x)   # Delete URL
    x = re.sub(r"\\\(", " ", x)
    x = re.sub(r"\\\)", " ", x)
    x = re.sub(r"[ ]{1,}", " ", x)
    x = re.sub(r"\.+", ".", x)    # Replace consecutive commas and periods with one comma and period character
    x = re.sub(r"\,+", ",", x)
    x = x.strip()                 # Remove empty characters at the beginning and end
    return x

df_ret['input_features'] = df_ret["ConstructName"] + ". " + df_ret["SubjectName"]
df_ret['input_features'] = df_ret['input_features'].apply(lambda x: preprocess_text(x))

embedding_query = model.encode(df_ret['input_features'], convert_to_tensor=True)
misconceptions = df_misconception_mapping.MisconceptionName.values
embedding_Misconception = model.encode(misconceptions, convert_to_tensor=True)

# the first time retrieval for LLM prompt
Ret_topNids = util.semantic_search(embedding_query, embedding_Misconception, top_k=100)
```

```python
retrivals = []
dicts = {}
for idx, row in tqdm(df_ret.iterrows(), total=len(df_ret)):
    top_ids = Ret_topNids[idx]
    retrival = ''
    dicts[str(row['QuestionId'])] = {}
    for i, ids in enumerate(top_ids):
        # serial number + misconceptions
        retrival += f'{i+1}. ' + misconceptions[ids['corpus_id']] + '\n'
        # save retrieved misconceptions for each QuestionId.
        dicts[str(row['QuestionId'])][str(i+1)] = misconceptions[ids['corpus_id']]
    retrivals.append(retrival)

df_ret['Retrival'] = retrivals
```

```python
def preprocess_text(x):
    x = re.sub("http\w+", '',x)   # Delete URL
    x = re.sub(r"\.+", ".", x)    # Replace consecutive commas and periods with one comma and period character
    x = re.sub(r"\,+", ",", x)
    x = re.sub(r"\\\(", " ", x)
    x = re.sub(r"\\\)", " ", x)
    x = re.sub(r"[ ]{1,}", " ", x)
    x = x.strip()                 # Remove empty characters at the beginning and end
    return x

PROMPT  = """Here is a question about {ConstructName}({SubjectName}).
Question: {Question}
Correct Answer: {CorrectAnswer}
Incorrect Answer: {IncorrectAnswer}

You are a Mathematics teacher. Your task is to reason and identify the misconception behind the Incorrect Answer with the Question.
Answer concisely what misconception it is to lead to getting the incorrect answer.
No need to give the reasoning process and do not use "The misconception is" to start your answers.
There are some relative and possible misconceptions below to help you make the decision:

{Retrival}
"""
# just directly give your answers.

def apply_template(row, tokenizer, targetCol):
    messages = [
        {
            "role": "user", 
            "content": preprocess_text(
                PROMPT.format(
                    ConstructName=row["ConstructName"],
                    SubjectName=row["SubjectName"],
                    Question=row["QuestionText"],
                    IncorrectAnswer=row[f"Answer{targetCol}Text"],
                    CorrectAnswer=row[f"Answer{row.CorrectAnswer}Text"],
                    Retrival=row[f"Retrival"]
                )
            )
        }
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return text

df = {}
if not IS_SUBMISSION:
    df_label = {}
    for idx, row in tqdm(df_ret.iterrows(), total=len(df_ret)):
        for option in ["A", "B", "C", "D"]:
            if (row.CorrectAnswer!=option) & (row[f"Misconception{option}Id"]!=-1):
                df[f"{row.QuestionId}_{option}"] = apply_template(row, tokenizer, option)
                df_label[f"{row.QuestionId}_{option}"] = [row[f"Misconception{option}Id"]]
                
    df_label = pd.DataFrame([df_label]).T.reset_index()
    df_label.columns = ["QuestionId_Answer", "MisconceptionId"]
    df_label.to_parquet("label.parquet", index=False)
else:
    for idx, row in tqdm(df_ret.iterrows(), total=len(df_ret)):
        for option in ["A", "B", "C", "D"]:
            if row.CorrectAnswer!=option:
                df[f"{row.QuestionId}_{option}"] = apply_template(row, tokenizer, option)

df = pd.DataFrame([df]).T.reset_index()
df.columns = ["QuestionId_Answer", "text"]
df.to_parquet("submission.parquet", index=False)
```

```python
print(df.loc[0, 'text'])
```

## LLM Reasoning

```python
%%writefile run_vllm.py

import re
import vllm
import pandas as pd

df = pd.read_parquet("submission.parquet")

model_path = "/kaggle/input/qwen2.5/transformers/32b-instruct-awq/1"

llm = vllm.LLM(
    model_path,
    quantization="awq",
    tensor_parallel_size=2,
    gpu_memory_utilization=0.90, 
    trust_remote_code=True,
    dtype="half", 
    enforce_eager=True,
    max_model_len=5120,
    disable_log_stats=True
)
tokenizer = llm.get_tokenizer()


responses = llm.generate(
    df["text"].values,
    vllm.SamplingParams(
        n=1,  # Number of output sequences to return for each prompt.
        top_p=0.8,  # Float that controls the cumulative probability of the top tokens to consider.
        temperature=0,  # randomness of the sampling
        seed=777, # Seed for reprodicibility
        skip_special_tokens=False,  # Whether to skip special tokens in the output.
        max_tokens=512,  # Maximum number of tokens to generate per output sequence.
    ),
    use_tqdm=True
)

responses = [x.outputs[0].text for x in responses]
df["fullLLMText"] = responses

def extract_response(text):
    return ",".join(re.findall(r"<response>(.*?)</response>", text)).strip()

df["llmMisconception"] = responses
df.to_parquet("submission.parquet", index=False)
```

```python
!python run_vllm.py
```

```python
llm_output = pd.read_parquet("submission.parquet")

for idx, row in llm_output[0:5].iterrows():
    print(row.llmMisconception)
    print("==="*6)
```

```python
text = llm_output.loc[0, 'text']
PREFIX = "<|im_start|>user"
text = text.split(PREFIX)[1].split("You are a Mathematics teacher.")[0].strip('\n').split('Here is a question about')[-1].strip()
print(text)
```

## Post-processing for LLM output

```python
import pandas as pd
from sentence_transformers import SentenceTransformer, util

df = pd.read_parquet("submission.parquet")
df_misconception_mapping = pd.read_csv("/kaggle/input/eedi-mining-misconceptions-in-mathematics/misconception_mapping.csv")

model = SentenceTransformer('/kaggle/input/eedi-finetuned-bge-public/Eedi-finetuned-bge')
```

```python
def number2sentence(row):
    """
    This is used for post-processing of LLM's output.
    Since we give top-N retrieval to the LLM with serial number,
    Sometimes the LLM will only output the serial number without any sentence.
    We use the 'dicts' generated at the beginning to map the serial number with corresponding misconceptions.
    """
    text = row['llmMisconception'].strip()
    # potential is the most possible serial number in LLM output.
    potential = re.search(r'^\w+\.{0,1}', text).group()
    if '.' in potential:
        sentence = text.replace(potential, '').strip()
    # if the LLM output is only a serial number, we map it with corresponding misconceptions saved in the dict.
    elif len(potential) == len(text):
        qid_retrieval = dicts[row['QuestionId']]
        try:
            # qid_retrieval is the top-N misconceptions for an QuestionId,
            # qid_retrieval[potential] is the most possible misconception.
            sentence = qid_retrieval[potential]
        except:
            # If the mapping fails, we use the first one(the most possible one in the first retrieval).
            sentence = qid_retrieval['1']
    else:
        sentence = text
        
    return sentence


df['QuestionId'] = df['QuestionId_Answer'].apply(lambda x: x.split('_')[0])
df['llmMisconception_clean'] = df.apply(number2sentence, axis=1)
```

```python
df.head(5)
```

# Second retrieval

```python
def preprocess_text(x):
    x = x.lower()                 # Convert words to lowercase
    x = re.sub(r"@\w+", '',x)      # Delete strings starting with @
    #x = re.sub(r"\d+", '',x)      # Delete Numbers
    x = re.sub(r"http\w+", '',x)   # Delete URL
    x = re.sub(r"\\\(", " ", x)
    x = re.sub(r"\\\)", " ", x)
    x = re.sub(r"[ ]{1,}", " ", x)
    x = re.sub(r"\.+", ".", x)    # Replace consecutive commas and periods with one comma and period character
    x = x.strip()                 # Remove empty characters at the beginning and end
    return x

PREFIX = "<|im_start|>user"
df['input_features'] = df["text"].apply(lambda x: x.split(PREFIX)[1].split("You are a Mathematics teacher.")[0].strip('\n').split('Here is a question about')[-1].strip())

df['input_features'] = df['input_features'].apply(lambda x: preprocess_text(x))
df['input_features'] = df["llmMisconception_clean"] + "\n\n" + df['input_features']

embedding_query = model.encode(df['input_features'], convert_to_tensor=True)
embedding_Misconception = model.encode(df_misconception_mapping.MisconceptionName.values, convert_to_tensor=True)
top25ids = util.semantic_search(embedding_query, embedding_Misconception, top_k=25)
```

```python
df["MisconceptionId"] = [" ".join([str(x["corpus_id"]) for x in top25id]) for top25id in top25ids]

df[["QuestionId_Answer", "MisconceptionId"]].to_csv("submission.csv", index=False)
df.head()
```

## Sanity

```python
if not IS_SUBMISSION:
    import pandas as pd
    from eedi_metrics import mapk
    predicted = pd.read_csv("submission.csv")["MisconceptionId"].apply(lambda x: [int(y) for y in x.split()])
    label = pd.read_parquet("label.parquet")["MisconceptionId"]
    print("Validation: ", mapk(label, predicted))
```