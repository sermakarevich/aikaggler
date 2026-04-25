# LUCKY

- **Author:** Quan Nguyen
- **Votes:** 508
- **Ref:** quannguyn12/lucky
- **URL:** https://www.kaggle.com/code/quannguyn12/lucky
- **Last run:** 2025-02-16 06:52:36.617000

---

References

- https://www.kaggle.com/code/mpware/vllm-0-7 for the current installation script
- https://www.kaggle.com/code/richolson/ai-math-olympiad-qwen2-5-72b for showing how to submit
- https://www.kaggle.com/code/abdullahmeda/load-72b-awq-model-using-vllm-on-l4-x4

```python
import os
# https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-2/discussion/560682#3113134
os.environ["TRITON_PTXAS_PATH"] = "/usr/local/cuda/bin/ptxas"
```

```python
import os
import gc
import time
import warnings

import pandas as pd
import polars as pl
import numpy as np

import torch
import kaggle_evaluation.aimo_2_inference_server

pd.set_option('display.max_colwidth', None)
start_time = time.time()
cutoff_time = start_time + (4 * 60 + 45) * 60
cutoff_times = [int(x) for x in np.linspace(cutoff_time, start_time + 60 * 60, 50 + 1)]
```

```python
from vllm import LLM, SamplingParams

warnings.simplefilter('ignore')

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

if os.getenv('KAGGLE_KERNEL_RUN_TYPE') or os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    llm_model_pth = '/kaggle/input/deepseek-r1/transformers/deepseek-r1-distill-qwen-7b-awq-casperhansen/1'
else:
    llm_model_pth = '/root/volume/KirillR/QwQ-32B-Preview-AWQ'

MAX_NUM_SEQS = 128
MAX_MODEL_LEN = 8192 * 3 // 2

llm = LLM(
    llm_model_pth,
    # dtype="half",                # The data type for the model weights and activations
    max_num_seqs=MAX_NUM_SEQS,   # Maximum number of sequences per iteration. Default is 256
    max_model_len=MAX_MODEL_LEN, # Model context length
    trust_remote_code=True,      # Trust remote code (e.g., from HuggingFace) when downloading the model and tokenizer
    tensor_parallel_size=4,      # The number of GPUs to use for distributed execution with tensor parallelism
    gpu_memory_utilization=0.95, # The ratio (between 0 and 1) of GPU memory to reserve for the model
    seed=2024,
)
```

```python
tokenizer = llm.get_tokenizer()

import vllm
print(vllm.__version__)
```

```python
import re
import keyword


def extract_boxed_text(text: str) -> str:
    pattern = r'oxed{(.*?)}'
    matches = re.findall(pattern, text)
    if not matches:
        return ""
    for match in matches[::-1]:
        if match != "":
            return match
    return ""


from collections import Counter
import random
def select_answer(answers: list[str]) -> int:
    counter = Counter()
    for answer in answers:
        try:
            if int(answer) == float(answer):
                counter[int(answer)] += 1 + random.random() / 1_000
        except:
            pass
    if not counter:
        return 210
    _, answer = sorted([(v,k) for k,v in counter.items()], reverse=True)[0]
    return answer%1000
```

```python
def batch_text_complete(completion_texts: list[str]) -> list[str]:
    max_tokens = MAX_MODEL_LEN
    if time.time() > cutoff_times[-1]:
        print("Speedrun")
        max_tokens = 2 * MAX_MODEL_LEN // 3

    sampling_params = SamplingParams(
        temperature=1.0,              # randomness of the sampling
        min_p=0.01,
        skip_special_tokens=True,     # Whether to skip special tokens in the output
        max_tokens=max_tokens,
        logit_bias={144540:-100, 21103: -100},
        stop=["</think>"],
    )
    
    request_output = llm.generate(
        prompts=completion_texts,
        sampling_params=sampling_params,
    )
    
    print([len(single_request_output.outputs[0].token_ids) for single_request_output in request_output])

    sort_keys_and_completion_texts = []

    for completion_text, single_request_output in zip(completion_texts, request_output):
        # print()
        # print(single_request_output.outputs[0].text)
        # print()
        completion_text += single_request_output.outputs[0].text

        sort_keys_and_completion_texts.append(
            (
                len(single_request_output.outputs[0].token_ids),
                completion_text
            )
        )

    print([sort_key for sort_key, _ in sort_keys_and_completion_texts])
    sort_keys_and_completion_texts.sort(key=lambda sort_key_and_completion_text: sort_key_and_completion_text[0])
    print([sort_key for sort_key, _ in sort_keys_and_completion_texts])

    completion_texts = [completion_text for _, completion_text in sort_keys_and_completion_texts]
    
    return completion_texts
```

```python
thought_prefix_english = """<think>
Alright, we have a math problem.

Hmm, it seems that I was asked to solve like a human. What does that mean? I guess I have to think through the problem step by step, similar to how a person would approach it.

Think deeper. Humans work with easier numbers. They not do insane arithmetic. It means that when I have insane calculations to do, I am likely on the wrong track.

What else? This also means I should not be working with decimal places. I should avoid decimals.

Also, I should not submit answers that I am not sure."""


def create_starter_text(question: str, index: int) -> str:
    options = []
    for _ in range(7):
        messages = [
            {"role": "system", "content": "Solve the math problem from the user. Only submit an answer if you are sure. Return final answer within \\boxed{}, after taking modulo 1000."},
            {"role": "user", "content": question},
        ]
        starter_text = tokenizer.apply_chat_template(
            conversation=messages,
            tokenize=False,
            add_generation_prompt=True
        ) + "<think>"
        options.append(starter_text)
    for _ in range(8):
        messages = [
            {"role": "system", "content": "Solve the math problem from the user, similar to how a human would (first think how would you solve like a human). Only submit an answer if you are sure. After you get your final answer, take modulo 1000, and return the final answer within \\boxed{}."},
            {"role": "user", "content": question},
        ]
        starter_text = tokenizer.apply_chat_template(
            conversation=messages,
            tokenize=False,
            add_generation_prompt=True
        ) + thought_prefix_english
        options.append(starter_text)
    for _ in range(1):
        messages = [
            {"role": "system", "content": "请通过逐步推理来解答问题，并把最终答案对1000取余数，放置于\\boxed{}中。"},
            {"role": "user", "content": question},
        ]
        starter_text = tokenizer.apply_chat_template(
            conversation=messages,
            tokenize=False,
            add_generation_prompt=True
        ) + "<think>"
        options.append(starter_text)
    
    return options[index%len(options)]
```

```python
def predict_for_question(question: str) -> int:
    import os

    selected_questions_only = True
    # selected_questions_only = False
    if selected_questions_only and not os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
        if "circumcircle" not in question:
            return 210
        if "Triangle" not in question and "airline" not in question and "circumcircle" not in question:
            return 210

    if time.time() > cutoff_time:
        return 210
    
    num_seqs = MAX_NUM_SEQS
    if time.time() > cutoff_times[-1]:
        print("speedrun")
        num_seqs = MAX_NUM_SEQS // 2
    
    completion_texts = [create_starter_text(question, index) for index in range(num_seqs)]
    completion_texts = batch_text_complete(completion_texts)
    
    if not os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
        df = pd.DataFrame(
            {
                "question": [question] * len(completion_texts),
                "completion": [completion_text for completion_text in completion_texts],
            }
        )
        df.to_csv(f"{str(int(time.time() - start_time)).zfill(5)}.csv", index=False)

    extracted_answers = [extract_boxed_text(completion_text) for completion_text in completion_texts]
    
    print(extracted_answers)
    answer = select_answer(extracted_answers)
    print(answer, "\n\n")

    cutoff_times.pop()
    return answer
```

```python
# Replace this function with your inference code.
# The function should return a single integer between 0 and 999, inclusive.
# Each prediction (except the very first) must be returned within 30 minutes of the question being provided.
def predict(id_: pl.DataFrame, question: pl.DataFrame) -> pl.DataFrame | pd.DataFrame:
    id_ = id_.item(0)
    print("------")
    print(id_)
    
    question = question.item(0)
    print(question)

    answer = predict_for_question(question)
    print("------\n\n\n")
    return pl.DataFrame({'id': id_, 'answer': answer})
```

```python
if os.getenv('KAGGLE_KERNEL_RUN_TYPE') == "Interactive" and not os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    predict_for_question("Triangle $ABC$ has side length $AB = 120$ and circumradius $R = 100$. Let $D$ be the foot of the perpendicular from $C$ to the line $AB$. What is the greatest possible length of segment $CD$?")
```

```python
inference_server = kaggle_evaluation.aimo_2_inference_server.AIMO2InferenceServer(predict)

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    pd.read_csv('/kaggle/input/ai-mathematical-olympiad-progress-prize-2/reference.csv').drop('answer', axis=1).to_csv('reference.csv', index=False)
    inference_server.run_local_gateway(
        (
#             '/kaggle/input/ai-mathematical-olympiad-progress-prize-2/test.csv',
            'reference.csv',
        )
    )
```