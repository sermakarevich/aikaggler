# Eedi Qwen32B vllm with logits-processor-zoo

- **Author:** Ahmet Erdem
- **Votes:** 593
- **Ref:** aerdem4/eedi-qwen32b-vllm-with-logits-processor-zoo
- **URL:** https://www.kaggle.com/code/aerdem4/eedi-qwen32b-vllm-with-logits-processor-zoo
- **Last run:** 2024-11-19 06:23:07.323000

---

```python
%%time
!pip uninstall -y torch
!pip install -q --no-index --find-links=/kaggle/input/making-wheels-of-necessary-packages-for-vllm vllm
!pip install -q -U /kaggle/input/vllm-t4-fix/grpcio-1.62.2-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
!pip install -q -U /kaggle/input/vllm-t4-fix/ray-2.11.0-cp310-cp310-manylinux2014_x86_64.whl
!pip install -q --no-deps --no-index /kaggle/input/hf-libraries/sentence-transformers/sentence_transformers-3.1.0-py3-none-any.whl
!pip install --no-deps --no-index /kaggle/input/logits-processor-zoo/logits_processor_zoo-0.1.0-py3-none-any.whl
```

```python
!pip install transformers peft accelerate \
    -q -U --no-index --find-links /kaggle/input/lmsys-wheel-files
```

```python
%%capture
!pip install --no-index /kaggle/input/bitsandbytes0-42-0/bitsandbytes-0.42.0-py3-none-any.whl --find-links=/kaggle/input/bitsandbytes0-42-0
!pip install --no-index  /kaggle/input/bitsandbytes0-42-0/optimum-1.21.2-py3-none-any.whl --find-links=/kaggle/input/bitsandbytes0-42-0
!pip install --no-index  /kaggle/input/bitsandbytes0-42-0/auto_gptq-0.7.1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl --find-links=/kaggle/input/bitsandbytes0-42-0
```

# Retrieval using 7B LLM
Taken from: https://www.kaggle.com/code/sayoulala/use-llm-embedding-recall-infer

```python
import pandas as pd

full_df = pd.read_csv("/kaggle/input/eedi-mining-misconceptions-in-mathematics/test.csv")


rows = []
for idx, row in full_df.iterrows():
    for option in ["A", "B", "C", "D"]:
        if option == row.CorrectAnswer:
            continue
            
        correct_answer = row[f"Answer{row.CorrectAnswer}Text"]

        query_text =f"###question###:{row['SubjectName']}-{row['ConstructName']}-{row['QuestionText']}\n###Correct Answer###:{correct_answer}\n###Misconcepte Incorrect answer###:{option}.{row[f'Answer{option}Text']}"

        rows.append({"query_text": query_text, 
                     "QuestionId_Answer": f"{row.QuestionId}_{option}",
                     "ConstructName": row.ConstructName,
                     "SubjectName": row.SubjectName,
                     "QuestionText": row.QuestionText,
                     "correct_answer": correct_answer,
                     "incorrect_answer": row[f"Answer{option}Text"]
                     })

df = pd.DataFrame(rows)
df
```

```python
import torch
from numpy.linalg import norm
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel,BitsAndBytesConfig
from peft import (
    LoraConfig,
    get_peft_model,
)

def batch_to_device(batch, target_device):
    """
    send a pytorch batch to a device (CPU/GPU)
    """
    for key in batch:
        if isinstance(batch[key], Tensor):
            batch[key] = batch[key].to(target_device)
    return batch

def last_token_pool(last_hidden_states: Tensor,
                    attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery: {query}'

def inference(df, model, tokenizer, device):
    batch_size = 16
    max_length = 512
    sentences = list(df['query_text'].values)

    all_embeddings = []
    length_sorted_idx = np.argsort([-len(sen) for sen in sentences])
    sentences_sorted = [sentences[idx] for idx in length_sorted_idx]
    for start_index in trange(0, len(sentences), batch_size, desc="Batches", disable=False):
        sentences_batch = sentences_sorted[start_index: start_index + batch_size]
        features = tokenizer(sentences_batch, max_length=max_length, padding=True, truncation=True,
                             return_tensors="pt")
        features = batch_to_device(features, device)
        with torch.no_grad():
            outputs = model.model(**features)
            embeddings = last_token_pool(outputs.last_hidden_state, features['attention_mask'])
            embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
            embeddings = embeddings.detach().cpu().numpy().tolist()
        all_embeddings.extend(embeddings)

    all_embeddings = [np.array(all_embeddings[idx]).reshape(1, -1) for idx in np.argsort(length_sorted_idx)]

    return np.concatenate(all_embeddings, axis=0)
```

```python
path_prefix = "/kaggle/input/eedi-mining-misconceptions-in-mathematics"
model_path = "/kaggle/input/sfr-embedding-mistral/SFR-Embedding-2_R"
lora_path="/kaggle/input/v7-recall/epoch_19_model/adapter.bin"
device='cuda:0'
```

```python
tokenizer = AutoTokenizer.from_pretrained(lora_path.replace("/adapter.bin",""))
bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
backbone = AutoModel.from_pretrained(model_path, quantization_config=bnb_config,device_map=device)
config = LoraConfig(
        r=64,
        lora_alpha=128,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        bias="none",
        lora_dropout=0.05,  # Conventional
        task_type="CAUSAL_LM",
    )
model = get_peft_model(backbone, config)
d = torch.load(lora_path, map_location=model.device)
model.load_state_dict(d, strict=False)
model = model.eval()
model = model.to(device)
```

```python
import numpy as np
from tqdm.autonotebook import trange


task_description = 'Given a math question and a misconcepte incorrect answer, please retrieve the most accurate reason for the misconception.'


V_answer = inference(df, model, tokenizer, device)

misconception_df = pd.read_csv("/kaggle/input/eedi-mining-misconceptions-in-mathematics/misconception_mapping.csv")
misconception_df["query_text"] = misconception_df["MisconceptionName"]

V_misconception = inference(misconception_df, model, tokenizer, device)
V_misconception.shape
```

```python
from transformers import AutoConfig, AutoTokenizer, AutoModelForMaskedLM, AutoModel
import sys, os
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn.functional as F
import torch.nn as nn
import math
from sklearn.neighbors import NearestNeighbors


os.environ["TOKENIZERS_PARALLELISM"] = "false"

def get_matches(V_topic, V_content, n_neighbors=25):
    
    neighbors_model = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine', algorithm="brute", n_jobs=-1)
    neighbors_model.fit(V_content)
    dists, indices = neighbors_model.kneighbors(V_topic)
    
    return indices

indices = get_matches(V_answer, V_misconception, n_neighbors=25)
indices.shape
```

```python
import gc

del backbone, model

gc.collect()
torch.cuda.empty_cache()
```

```python
np.save("indices.npy", indices)
df.to_parquet("df.parquet", index=False)
```

# Picking the best candidate using qwen-32b-instruct-awq
Inspired by: https://www.kaggle.com/code/takanashihumbert/eedi-qwen-2-5-32b-awq-two-time-retrieval

## Using MultipleChoiceLogitsProcessor from logits-processor-zoo

<p align="center">
    <img src="https://raw.githubusercontent.com/NVIDIA/logits-processor-zoo/refs/heads/main/docs/logo.jpg" width="512">
</p>

# logits-processor-zoo

Struggling to get LLMs to follow your instructions? LogitsProcessorZoo offers a zoo of tools to use LLMs for specific tasks, beyond just grammar enforcement!

## Installation

```bash
pip install logits-processor-zoo
```

## Supported Frameworks
* transformers
* vLLM
* TensorRT-LLM


For the detailed examples in each framework, please have a look at **example_notebook** directory.

## Available Logits Processors

### GenLengthLogitsProcessor
A logits processor that adjusts the likelihood of the end-of-sequence (EOS) token based on the length of the generated sequence, encouraging or discouraging shorter answers.

### CiteFromPromptLogitsProcessor
A logits processor which boosts or diminishes the likelihood of tokens present in the prompt (and optionally EOS token) to encourage the model to generate tokens similar to those seen in the prompt or vice versa.

### ForceLastPhraseLogitsProcessor
A logits processor which forces LLMs to use the given phrase before they finalize their answers. Most common use cases can be providing references, thanking user with context etc.

### MultipleChoiceLogitsProcessor
A logits processor to answer multiple choice questions with one of the choices. A multiple choice question is like:
```
I am getting a lot of calls during the day. What is more important for me to consider when I buy a new phone?
0. Camera
1. Screen resolution
2. Operating System
3. Battery
```
The goal is to make LLM generate "3" as an answer.

```python
%%writefile run_vllm.py

import vllm
import numpy as np
import pandas as pd
from transformers import PreTrainedTokenizer, AutoTokenizer
from typing import List
import torch
from logits_processor_zoo.vllm import MultipleChoiceLogitsProcessor
import re

model_path = "/kaggle/input/qwen2.5/transformers/32b-instruct-awq/1"
tokenizer = AutoTokenizer.from_pretrained(model_path)


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
Pick the correct misconception number from the below:

{Retrival}
"""
# just directly give your answers.

def apply_template(row, tokenizer):
    messages = [
        {
            "role": "user", 
            "content": preprocess_text(
                PROMPT.format(
                    ConstructName=row["ConstructName"],
                    SubjectName=row["SubjectName"],
                    Question=row["QuestionText"],
                    IncorrectAnswer=row[f"incorrect_answer"],
                    CorrectAnswer=row[f"correct_answer"],
                    Retrival=row[f"retrieval"]
                )
            )
        }
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return text


misconception_df = pd.read_csv("/kaggle/input/eedi-mining-misconceptions-in-mathematics/misconception_mapping.csv")

df = pd.read_parquet("df.parquet")
indices = np.load("indices.npy")

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


def get_candidates(c_indices):
    candidates = []

    mis_names = misconception_df["MisconceptionName"].values
    for ix in c_indices:
        c_names = []
        for i, name in enumerate(mis_names[ix]):
            c_names.append(f"{i+1}. {name}")

        candidates.append("\n".join(c_names))
        
    return candidates

survivors = indices[:, -1:]

for i in range(3):
    c_indices = np.concatenate([indices[:, -8*(i+1)-1:-8*i-1], survivors], axis=1)
    
    df["retrieval"] = get_candidates(c_indices)
    df["text"] = df.apply(lambda row: apply_template(row, tokenizer), axis=1)
    
    print("Example:")
    print(df["text"].values[0])
    print()
    
    responses = llm.generate(
        df["text"].values,
        vllm.SamplingParams(
            n=1,  # Number of output sequences to return for each prompt.
            top_k=1,  # Float that controls the cumulative probability of the top tokens to consider.
            temperature=0,  # randomness of the sampling
            seed=777, # Seed for reprodicibility
            skip_special_tokens=False,  # Whether to skip special tokens in the output.
            max_tokens=1,  # Maximum number of tokens to generate per output sequence.
            logits_processors=[MultipleChoiceLogitsProcessor(tokenizer, choices=["1", "2", "3", "4", "5", "6", "7", "8", "9"])]
        ),
        use_tqdm=True
    )
    
    responses = [x.outputs[0].text for x in responses]
    df["response"] = responses
    
    
    llm_choices = df["response"].astype(int).values - 1
    
    survivors = np.array([cix[best] for best, cix in zip(llm_choices, c_indices)]).reshape(-1, 1)



results = []

for i in range(indices.shape[0]):
    ix = indices[i]
    llm_choice = survivors[i, 0]
    
    results.append(" ".join([str(llm_choice)] + [str(x) for x in ix if x != llm_choice]))


df["MisconceptionId"] = results
df.to_csv("submission.csv", columns=["QuestionId_Answer", "MisconceptionId"], index=False)
```

```python
!python run_vllm.py
```

```python
pd.read_csv("submission.csv")
```