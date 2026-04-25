# WSDM apply previous LMSYS solution

- **Author:** Takamichi Toda
- **Votes:** 329
- **Ref:** takamichitoda/wsdm-apply-previous-lmsys-solution
- **URL:** https://www.kaggle.com/code/takamichitoda/wsdm-apply-previous-lmsys-solution
- **Last run:** 2024-11-19 20:27:11.060000

---

This notebook is a direct adaptation of my solution to the previous LMSYS competition for this competition!

The format for submitting the prediction results is different, so I have aligned that, but other than that, I have kept my submission code Forked.

For more details on the solution, please refer to the Discussion [here](https://www.kaggle.com/competitions/lmsys-chatbot-arena/discussion/527938).


#### Previous LMSYS result:

|  | log loss |
| - | - |
| public LB | 0.89009 |
| private LB | 1.00182 |

```python
!pip install transformers peft accelerate bitsandbytes \
    -U --no-index --find-links /kaggle/input/lmsys-wheel-files
```

```python
!cp -r /kaggle/input/lmsys-gemma2-test/last_model ./merge_model
!rm ./merge_model/adapter_model.safetensors ./merge_model/adapter_config.json
!ls merge_model
```

```python
import time
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

import torch
import sklearn
import numpy as np
import pandas as pd
from transformers import Gemma2ForSequenceClassification, GemmaTokenizerFast, BitsAndBytesConfig
from transformers.data.data_collator import pad_without_fast_tokenizer_warning
from peft import PeftModel
```

```python
assert torch.cuda.device_count() == 2
```

## Configurations

```python
@dataclass
class Config:
    gemma_dir = '/kaggle/input/gemma-2/transformers/gemma-2-9b-it-4bit/1/gemma-2-9b-it-4bit'
    lora_dir = "./merge_model"
    lora_dirs = [
        "/kaggle/input/lmsys-exp14-full-long-lr1e4/exp14_full_long_lr1e4/last_model",
        "/kaggle/input/lmsys-exp15-cont-org-lr-3e5/last_model",
    ]
    #lora_weights = [0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125]
    lora_weights = [0.7, 0.3]
    max_length = 2048
    prompt_max_l = 512
    batch_size = 4
    device = torch.device("cuda")    
    tta = True  # test time augmentation. <prompt>-<model-b's response>-<model-a's response>
    spread_max_length = False  # whether to apply max_length//3 on each input or max_length on the concatenated input
    
    best_temp = 1
    merge_method = "avg" # concatenation, svd, linear

cfg = Config()
```

```python
import torch
from peft import LoraConfig
from safetensors.torch import load_file, save_file
import json

def load_config(path):
    with open(path, 'r') as f:
        return json.load(f)
    
def load_lora_weights(path):
    return load_file(path)
    
def weighted_merge_lora_weights(weight_files, weights):
    if len(weight_files) != len(weights):
        raise ValueError("Number of weight files and weights must match")
    
    all_lora_weights = [load_lora_weights(f + "/adapter_model.safetensors") for f in weight_files]
    merged_weights = {}
    
    for key in all_lora_weights[0].keys():
        merged_weights[key] = sum(w * lora_weights[key] for w, lora_weights in zip(weights, all_lora_weights))
    
    return merged_weights

def save_merged_weights(weights, output_path):
    save_file(weights, output_path)
```

```python
merged_config = load_config(cfg.lora_dirs[0] + "/adapter_config.json")

if cfg.merge_method == "avg":
    ensembled_weights = weighted_merge_lora_weights(cfg.lora_dirs, cfg.lora_weights)
elif cfg.merge_method == "cat":
    merged_config["lora_alpha"] *= 2
    merged_config["r"] *= 2
    weights = [load_file(d + "/adapter_model.safetensors") for d in cfg.lora_dirs]
    
    ensembled_weights = {}
    for key in weights[0].keys():
        if "lora_A" in key:
            ensembled_weights[key] = torch.cat([w[key] for w in weights], dim=0)
        elif "lora_B" in key:
            ensembled_weights[key] = torch.cat([w[key] for w in weights], dim=1)
        else:
            ensembled_weights[key] = weights[0][key]  # 他の重みはそのまま保持

save_merged_weights(ensembled_weights, "./merge_model/adapter_model.safetensors")
with open("./merge_model/adapter_config.json", 'w') as f:
    json.dump(merged_config, f, indent=2)
```

# Load & pre-process Data

```python
import pyarrow
import pyarrow.parquet
```

```python
test =  pyarrow.parquet.read_table('/kaggle/input/wsdm-cup-multilingual-chatbot-arena/test.parquet')
test  = test.to_pandas()
test = test.fillna("")
test
```

# Tokenize

```python
def tokenize_shape(prompt, response_a, response_b):
    p = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    a = tokenizer(response_a, add_special_tokens=False)["input_ids"]
    b = tokenizer(response_b, add_special_tokens=False)["input_ids"]

    tokenized = {"input_ids": [], "attention_mask": []}
    for i, _p in enumerate(p):
        if len(_p) > cfg.prompt_max_l:
            _p = _p[-cfg.prompt_max_l:]
        rl = (cfg.max_length - len(_p)) // 2
        input_ids = [tokenizer.bos_token_id] + _p + a[i][-rl:] + b[i][-rl:] + [tokenizer.eos_token_id]
        attention_mask = [1] * len(input_ids)
        tokenized["input_ids"].append(input_ids)
        tokenized["attention_mask"].append(attention_mask)
    return tokenized

def tokenize(
    tokenizer, prompt, response_a, response_b, max_length=cfg.max_length, spread_max_length=cfg.spread_max_length
):
    prompt = ["<prompt>: " + p for p in prompt]
    response_a = ["\n\n<response_a>: " + r_a for r_a in response_a]
    response_b = ["\n\n<response_b>: " + r_b for r_b in response_b]
    tokenized = tokenize_shape(prompt, response_a, response_b)
    input_ids = tokenized["input_ids"]
    attention_mask = tokenized["attention_mask"]
    return input_ids, attention_mask
```

```python
%%time

tokenizer = GemmaTokenizerFast.from_pretrained(cfg.gemma_dir)
tokenizer.add_eos_token = True
tokenizer.padding_side = "right"

data = pd.DataFrame()
data["id"] = test["id"]
data["input_ids"], data["attention_mask"] = tokenize(tokenizer, test["prompt"], test["response_a"], test["response_b"])
data["length"] = data["input_ids"].apply(len)

aug_data = pd.DataFrame()
aug_data["id"] = test["id"]
# swap response_a & response_b
aug_data['input_ids'], aug_data['attention_mask'] = tokenize(tokenizer, test["prompt"], test["response_b"], test["response_a"])
aug_data["length"] = aug_data["input_ids"].apply(len)
```

```python
print(tokenizer.decode(aug_data["input_ids"][0]))
```

# Load model

```python
# Load base model on GPU 0
device_0 = torch.device('cuda:0')
model_0 = Gemma2ForSequenceClassification.from_pretrained(
    cfg.gemma_dir,
    device_map=device_0,
    use_cache=False,
)

# Load base model on GPU 1
device_1 = torch.device('cuda:1')
model_1 = Gemma2ForSequenceClassification.from_pretrained(
    cfg.gemma_dir,
    device_map=device_1,
    use_cache=False,
)
```

#### Load LoRA adapter

```python
model_0 = PeftModel.from_pretrained(model_0, cfg.lora_dir)
model_1 = PeftModel.from_pretrained(model_1, cfg.lora_dir)
```

# Inference

```python
@torch.no_grad()
@torch.cuda.amp.autocast()
def inference(df, model, device, batch_size=cfg.batch_size, max_length=cfg.max_length):
    a_win, b_win, tie = [], [], []
    
    for start_idx in range(0, len(df), batch_size):
        end_idx = min(start_idx + batch_size, len(df))
        tmp = df.iloc[start_idx:end_idx]
        input_ids = tmp["input_ids"].to_list()
        attention_mask = tmp["attention_mask"].to_list()
        inputs = pad_without_fast_tokenizer_warning(
            tokenizer,
            {"input_ids": input_ids, "attention_mask": attention_mask},
            padding="longest",
            pad_to_multiple_of=None,
            return_tensors="pt",
        )
        outputs = model(**inputs.to(device))
        proba = (outputs.logits / cfg.best_temp).softmax(-1).cpu()
        
        a_win.extend(proba[:, 0].tolist())
        b_win.extend(proba[:, 1].tolist())
        tie.extend(proba[:, 2].tolist())
    
    df["winner_model_a"] = a_win
    df["winner_model_b"] = b_win
    df["winner_tie"] = tie
    
    return df
```

```python
st = time.time()

# sort by input length to fully leverage dynaminc padding
data = data.sort_values("length", ascending=False)
# the total #tokens in sub_1 and sub_2 should be more or less the same
sub_1 = data.iloc[0::2].copy()
sub_2 = data.iloc[1::2].copy()

with ThreadPoolExecutor(max_workers=2) as executor:
    results = executor.map(inference, (sub_1, sub_2), (model_0, model_1), (device_0, device_1))

result_df = pd.concat(list(results), axis=0)
proba = result_df[["winner_model_a", "winner_model_b", "winner_tie"]].values

print(f"elapsed time: {time.time() - st}")
```

```python
st = time.time()

if cfg.tta:
    data = aug_data.sort_values("length", ascending=False)  # sort by input length to boost speed
    sub_1 = data.iloc[0::2].copy()
    sub_2 = data.iloc[1::2].copy()

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = executor.map(inference, (sub_1, sub_2), (model_0, model_1), (device_0, device_1))

    tta_result_df = pd.concat(list(results), axis=0)
    # recall TTA's order is flipped
    tta_proba = tta_result_df[["winner_model_b", "winner_model_a", "winner_tie"]].values 
    # average original result and TTA result.
    #proba = (proba + tta_proba) / 2
    proba = proba * 0.45 + tta_proba * 0.55

print(f"elapsed time: {time.time() - st}")
```

```python
result_df.loc[:, "winner_model_a"] = proba[:, 0]
result_df.loc[:, "winner_model_b"] = proba[:, 1]
result_df.loc[:, "winner_tie"] = proba[:, 2]
result_df["winner"] = ["model_a" if i else "model_b" for i in proba[:, 0] > proba[:, 1]]
result_df
```

```python
result_df[["id", "winner"]].to_csv('submission.csv', index=False)
```

```python
sub_df = pd.read_csv("/kaggle/input/wsdm-cup-multilingual-chatbot-arena/sample_submission.csv")
sub_df
```

```python
pd.read_csv("submission.csv")
```