# [LB24] Deepseek-R1 with more prompt Engineering

- **Author:** ← Checkout My Cat
- **Votes:** 1040
- **Ref:** haoruili/lb24-deepseek-r1-with-more-prompt-engineering
- **URL:** https://www.kaggle.com/code/haoruili/lb24-deepseek-r1-with-more-prompt-engineering
- **Last run:** 2025-01-28 08:58:50.887000

---

<h1 style="text-align: center; font-size: 1.5em; font-weight: bold; color: #2c3e50; margin-bottom: 0.5em;">
   [DeepSeek R1] Solution Notebook
</h1>

<h2 style="text-align: center; font-size: 1.5em; color: #34495e; font-style: italic; margin-top: 0;">
   Leveraging LLMs for Mathematical Problem Solving
</h2>
<div align="center">
    <img src="https://i.dawnlab.me/bda96ce3ca698fd128bac60950b0e962.png">
</div>

<h2 style="text-align: center; font-size: 1.5em; color: #34495e; margin-top: 0;">
   Public LB Score: 22 
</h2>


References

- https://www.kaggle.com/datasets/asalhi/qwq-32b-pre-awq
- https://www.kaggle.com/code/sorokin/pip-install-aimo2 for the current installation script
- https://www.kaggle.com/code/richolson/ai-math-olympiad-qwen2-5-72b for showing how to submit
- https://www.kaggle.com/code/abdullahmeda/load-72b-awq-model-using-vllm-on-l4-x4
- https://www.kaggle.com/code/huikang/qwen2-5-math-1-5b-instruct
- https://www.kaggle.com/code/huikang/deepseek-r1-distill-qwen-32b-awq For Deepseek model

# How to do reasoning LLM prompt Engineering

My idea of the reasonging LLM prompt engineering mainly comes from these papers:

- [Reverse Thinking Makes LLMs Stronger Reasoners](https://arxiv.org/abs/2411.19865)
- [Take a Step Back: Evoking Reasoning via Abstraction in Large Language Models](https://arxiv.org/abs/2310.06117)
- [DeepSeek-R1](https://arxiv.org/abs/2501.12948)

Here's some key point in these paper:
1. Try to make LLM recheck its answer。
2. Let LLM try to abstract the problem to a higher level
3. LLM may 'think' in different languages

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
cutoff_times = [int(x) for x in np.linspace(cutoff_time, start_time + 180 * 60, 50 + 1)]
```

```python
from vllm import LLM, SamplingParams

warnings.simplefilter('ignore')

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

if os.getenv('KAGGLE_KERNEL_RUN_TYPE') or os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    llm_model_pth = '/kaggle/input/deepseek-r1/transformers/deepseek-aideepseek-r1-distill-qwen-14b-awq-neody/1'
else:
    llm_model_pth = '/root/volume/KirillR/QwQ-32B-Preview-AWQ'

MAX_NUM_SEQS = 16
MAX_MODEL_LEN = 8192

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
```

```python
import re
import keyword


def extract_boxed_text(text):
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
def select_answer(answers):
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
def batch_message_generate(list_of_messages) -> list[list[dict]]:
    max_tokens = MAX_MODEL_LEN
    if time.time() > cutoff_times[-1]:
        print("Speedrun")
        max_tokens = 2 * MAX_MODEL_LEN // 3

    sampling_params = SamplingParams(
        temperature=1.0,              # randomness of the sampling
        min_p=0.01,
        skip_special_tokens=True,     # Whether to skip special tokens in the output
        max_tokens=max_tokens,
    )
    
    list_of_texts = [
        tokenizer.apply_chat_template(
            conversation=messages,
            tokenize=False,
            add_generation_prompt=True
        )
        for messages in list_of_messages
    ]

    request_output = llm.generate(
        prompts=list_of_texts,
        sampling_params=sampling_params,
    )
    
    print([len(single_request_output.outputs[0].token_ids) for single_request_output in request_output])

    sort_keys_and_list_of_messages = []

    for messages, single_request_output in zip(list_of_messages, request_output):
        # print()
        # print(single_request_output.outputs[0].text)
        # print()
        messages.append({'role': 'assistant', 'content': single_request_output.outputs[0].text})

        sort_keys_and_list_of_messages.append(
            (
                len(single_request_output.outputs[0].token_ids),
                messages
            )
        )

    print([sort_key for sort_key, _ in sort_keys_and_list_of_messages])
    sort_keys_and_list_of_messages.sort(key=lambda sort_key_and_messages: sort_key_and_messages[0])
    print([sort_key for sort_key, _ in sort_keys_and_list_of_messages])

    list_of_messages = [messages for _, messages in sort_keys_and_list_of_messages]
    
    return list_of_messages
```

```python
def batch_message_filter(list_of_messages) -> tuple[list[list[dict]], list[str]]:
    extracted_answers = []
    list_of_messages_to_keep = []
    for messages in list_of_messages:
        answer = extract_boxed_text(messages[-1]['content'])
        if answer:
            extracted_answers.append(answer)
        else:
            list_of_messages_to_keep.append(messages)
    return list_of_messages_to_keep, extracted_answers
```

```python
def create_starter_messages(question, index):
    options = []
    for _ in range(13):
        options.append(
            [
                {"role": "system", "content": "You are a the most powerful math expert. Please solve the problems with deep resoning. You are careful and always recheck your conduction. You will never give answer directly until you have enough confidence. You should think step-by-step. Return final answer within \\boxed{}, after taking modulo 1000."},
                {"role": "user", "content": question},
            ]
        )
    for _ in range(2):    
        options.append(
            [
                {"role": "system", "content": "You are a helpful and harmless math assistant. You should think step-by-step and you are good at reverse thinking to recheck your answer and fix all possible mistakes. After you get your final answer, take modulo 1000, and return the final answer within \\boxed{}."},
                {"role": "user", "content": question},
            ],
        )
    options.append(
        [
            {"role": "system", "content": "Please carefully read the problem statement first to ensure you fully understand its meaning and key points. Then, solve the problem correctly and completely through deep reasoning. Finally, return the result modulo 1000 and enclose it in \\boxed{} like \"Atfer take the result modulo 1000, final anwer is \\boxed{180}."},
            {"role": "user", "content": question},
        ],
    )
    return options[index%len(options)]
```

```python
def predict_for_question(question: str) -> int:
    import os

    selected_questions_only = True
    if selected_questions_only and not os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
        if "Triangle" not in question:
            return 210
        if "Triangle" not in question and "delightful" not in question and "George" not in question:
            return 210

    if time.time() > cutoff_time:
        return 210
    
    print(question)

    num_seqs = MAX_NUM_SEQS
    if time.time() > cutoff_times[-1]:
        num_seqs = 2 * MAX_NUM_SEQS // 3
    
    list_of_messages = [create_starter_messages(question, index) for index in range(num_seqs)]

    all_extracted_answers = []
    for _ in range(1):
        list_of_messages = batch_message_generate(list_of_messages)
        
        if not os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
            df = pd.DataFrame(
                {
                    "question": [question] * len(list_of_messages),
                    "message": [messages[-1]["content"] for messages in list_of_messages],
                }
            )
            df.to_csv(f"{str(int(time.time() - start_time)).zfill(5)}.csv", index=False)
        
        list_of_messages, extracted_answers = batch_message_filter(list_of_messages)
        all_extracted_answers.extend(extracted_answers)
    
    print(all_extracted_answers)
    answer = select_answer(all_extracted_answers)
    print(answer)

    print("\n\n")
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
    answer = predict_for_question(question)
    print(question)
    print("------\n\n\n")
    return pl.DataFrame({'id': id_, 'answer': answer})
```

```python
# predict_for_question("Triangle $ABC$ has side length $AB = 120$ and circumradius $R = 100$. Let $D$ be the foot of the perpendicular from $C$ to the line $AB$. What is the greatest possible length of segment $CD$?")
```

```python
pd.read_csv(
    '/kaggle/input/ai-mathematical-olympiad-progress-prize-2/reference.csv'
).drop('answer', axis=1).to_csv('reference.csv', index=False)
```

```python
inference_server = kaggle_evaluation.aimo_2_inference_server.AIMO2InferenceServer(predict)

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    inference_server.run_local_gateway(
        (
#             '/kaggle/input/ai-mathematical-olympiad-progress-prize-2/test.csv',
            'reference.csv',
        )
    )
```