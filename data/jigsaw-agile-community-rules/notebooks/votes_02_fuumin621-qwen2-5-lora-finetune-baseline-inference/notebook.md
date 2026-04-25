# Qwen2.5-LoRA-Finetune-Baseline-Inference

- **Author:** monnu
- **Votes:** 506
- **Ref:** fuumin621/qwen2-5-lora-finetune-baseline-inference
- **URL:** https://www.kaggle.com/code/fuumin621/qwen2-5-lora-finetune-baseline-inference
- **Last run:** 2025-07-31 03:21:55.597000

---

## Qwen2.5-LoRA-Finetune-Baseline-Inference

This notebook demonstrates how to perform inference using a fine-tuned Qwen2.5-32B large language model with LoRA for a text classification task. It uses the Jigsaw dataset to classify if Reddit comments violate community rules.

Key features:

*   **Model:** Qwen2.5-32B
*   **Fine-tuning:** LoRA (Low-Rank Adaptation)
*   **Inference:** vLLM
*   **Task:** Text classification (Jigsaw dataset - Reddit comment rule violation)

### Note
Training Notebook is available [here](https://www.kaggle.com/code/fuumin621/qwen2-5-lora-finetune-baseline-training).

The training code is for demonstration purposes only. To reproduce the weights in the inference notebook, you need to change the config as follows. Training was performed on a local A6000.

- `IS_DEBUG = True` → `False`
- `MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4"` → `"Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4"`
- `TRAIN_BS = 1` → `8`
- `GRAD_ACC_NUM = 8` → `1`

### References

*   [https://www.kaggle.com/code/abdmental01/jigsaw-mpnet-base-v2-inference-cv-0-876](https://www.kaggle.com/code/abdmental01/jigsaw-mpnet-base-v2-inference-cv-0-876)
*   [https://www.kaggle.com/code/aerdem4/jigsaw-acrc-qwen7b-finetune-logits-processor-zoo](https://www.kaggle.com/code/aerdem4/jigsaw-acrc-qwen7b-finetune-logits-processor-zoo)
*   [https://www.guruguru.science/competitions/24/discussions/21027ff1-2074-4e21-a249-b2d4170bd516/](https://www.guruguru.science/competitions/24/discussions/21027ff1-2074-4e21-a249-b2d4170bd516/)

```python
MODEL_NAME = "/kaggle/input/qwen2-5-32b-instruct-gptq-int4"
LORA_PATH = "/kaggle/input/jigsaw-exp003-fold0/trained_model"
```

```python
# %%
import os
os.environ["VLLM_USE_V1"] = "0"
import pandas as pd
from logits_processor_zoo.vllm import MultipleChoiceLogitsProcessor
import torch
import vllm
import numpy as np
from vllm.lora.request import LoRARequest
import argparse
from scipy.special import softmax
df = pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/test.csv")
```

```python
llm = vllm.LLM(
    MODEL_NAME,
    # quantization='awq',
    quantization='gptq',
    tensor_parallel_size=torch.cuda.device_count(),
    gpu_memory_utilization=0.95,
    trust_remote_code=True,
    dtype="half",
    enforce_eager=True,
    max_model_len=4096,
    disable_log_stats=True,
    enable_prefix_caching=True,
    enable_lora=True,
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

2) {row.negative_example_1}
Violation: No

3) {row.negative_example_2}
Violation: No

4) {row.positive_example_2}
Violation: Yes

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
```

```python
df[['Yes',"No"]] = df[['Yes',"No"]].apply(lambda x: softmax(x.values), axis=1, result_type="expand")
df["pred"] = df["Yes"]
df['rule_violation'] = df["pred"]
df[['row_id', 'rule_violation']].to_csv("submission.csv",index=False)
df[['row_id', 'rule_violation']].head()
```