# Qwen2.5-72B-Instruct with TIR

- **Author:** Tong Hui Kang
- **Votes:** 583
- **Ref:** huikang/qwen2-5-72b-instruct-with-tir
- **URL:** https://www.kaggle.com/code/huikang/qwen2-5-72b-instruct-with-tir
- **Last run:** 2024-11-10 20:04:21.943000

---

References

- https://www.kaggle.com/code/richolson/ai-math-olympiad-qwen2-5-72b for showing how to submit
- https://www.kaggle.com/code/abdullahmeda/load-72b-awq-model-using-vllm-on-l4-x4 for showing how to vllm
- https://www.kaggle.com/code/huikang/qwen2-5-math-1-5b-instruct

You will need to select `GPU L4 x4` as the accelerator before you run or submit the notebook.

```python
import os
import gc
import time
import warnings

import pandas as pd
import polars as pl

import torch
import kaggle_evaluation.aimo_2_inference_server

pd.set_option('display.max_colwidth', None)
cutoff_time = time.time() + (4 * 60 + 30) * 60
```

```python
from vllm import LLM, SamplingParams

warnings.simplefilter('ignore')

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def clean_memory(deep=False):
    gc.collect()
    if deep:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    torch.cuda.empty_cache()


llm_model_pth = '/kaggle/input/qwen2.5/transformers/72b-instruct-awq/1'

llm = LLM(
    llm_model_pth,
    dtype="half",                # The data type for the model weights and activations
    max_num_seqs=16,             # Maximum number of sequences per iteration. Default is 256
    max_model_len=4096,          # Model context length
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


def extract_python_code(text):
    pattern = r'```python\s*(.*?)\s*```'
    matches = re.findall(pattern, text, re.DOTALL)
    return "\n\n".join(matches)


def process_python_code(query):
    # Add import statements
    # Also print variables if they are not inside any indentation
    query = "import math\nimport numpy as np\nimport sympy as sp\n" + query
    current_rows = query.strip().split("\n")
    new_rows = []
    for row in current_rows:
        new_rows.append(row)
        if not row.startswith(" ") and "=" in row:
            variables_to_print = row.split("=")[0].strip()
            for variable_to_print in variables_to_print.split(","):
                variable_to_print = variable_to_print.strip()
                if variable_to_print.isidentifier() and not keyword.iskeyword(variable_to_print):
                    if row.count("(") == row.count(")") and row.count("[") == row.count("]"):
                        # TODO: use some AST to parse code
                        new_rows.append(f'\ntry:\n    print(f"{variable_to_print}={{str({variable_to_print})[:100]}}")\nexcept:\n    pass\n')
    return "\n".join(new_rows)


def extract_boxed_texts(text: str) -> list[str]:
    pattern = r'oxed{(.*?)}'
    matches = re.findall(pattern, text)
    return [match.strip() for match in matches]


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
import os
import tempfile
import subprocess

class PythonREPL:
    def __init__(self, timeout=5):
        self.timeout = timeout

    def __call__(self, query):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file_path = os.path.join(temp_dir, "tmp.py")
            with open(temp_file_path, "w", encoding="utf-8") as f:
                f.write(query)
            
            try:
                result = subprocess.run(
                    ["python3", temp_file_path],
                    capture_output=True,
                    check=False,
                    text=True,
                    timeout=self.timeout,
                )
            except subprocess.TimeoutExpired:
                return False, f"Execution timed out after {self.timeout} seconds."

            stdout = result.stdout.strip()
            stderr = result.stderr.strip()

            if result.returncode == 0:
                return True, stdout
            else:
                # Process the error message to remove the temporary file path
                # This makes the error message cleaner and more user-friendly
                error_lines = stderr.split("\n")
                cleaned_errors = []
                for line in error_lines:
                    if temp_file_path in line:
                        # Remove the path from the error line
                        line = line.replace(temp_file_path, "<temporary_file>")
                    cleaned_errors.append(line)
                cleaned_error_msg = "\n".join(cleaned_errors)
                # Include stdout in the error case
                combined_output = f"{stdout}\n{cleaned_error_msg}" if stdout else cleaned_error_msg
                return False, combined_output
```

```python
sampling_params = SamplingParams(
    temperature=1.0,              # randomness of the sampling
    min_p=0.01,
    skip_special_tokens=True,     # Whether to skip special tokens in the output.
    max_tokens=2400,
    stop=["```\n"],
    include_stop_str_in_output=True,
)

def batch_message_generate(list_of_messages) -> list[list[dict]]:

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
    
    for messages, single_request_output in zip(list_of_messages, request_output):
        # print()
        # print(single_request_output.outputs[0].text)
        # print()
        messages.append({'role': 'assistant', 'content': single_request_output.outputs[0].text})

    return list_of_messages
```

```python
from typing import Optional
def identify_answer(messages: list[dict]) -> Optional[str]:
    list_of_answers = []
    for message in messages:
        if message['role'] == 'assistant':
            answers = extract_boxed_texts(message['content'])
            list_of_answers.append(answers)
    
    print('list_of_answers', list_of_answers)
    if len(list_of_answers) >= 2:
        
        # penultimate reply has one answer
        penultimate_answers = list_of_answers[-2]
        final_answers = [final_answer for final_answer in list_of_answers[-1] if final_answer != '' and 'no contradiction' not in final_answer.lower()]
        if len(list(set(penultimate_answers))) == 1:
            penultimate_answer = penultimate_answers[0]
            
            # final reply does not have answers that are is not penultimate_answer
            if set(final_answers) - set(penultimate_answers) == set():

                # print for debugging
                print('--------')
                for message in messages:
                    print(message['content'])
                    print('----')
                print('list_of_answers end', list_of_answers)

                return penultimate_answer

    return None


def batch_message_filter(list_of_messages: list[list[dict]]) -> tuple[list[list[dict]], list[str]]:
    answers = []
    list_of_messages_to_keep = []
    for messages in list_of_messages:
        answer = identify_answer(messages)
        if answer is not None:
            answers.append(answer)
        else:
            list_of_messages_to_keep.append(messages)
    return list_of_messages_to_keep, answers
```

```python
def batch_message_followup(list_of_messages) -> list[list[dict]]:
    for messages in list_of_messages:
        python_code = extract_python_code(messages[-1]['content'])
        if python_code:
            python_code = process_python_code(python_code)
            # print('\n\n' + python_code + '\n\n')
            try:
                print('c', end='')
                is_successful, output = PythonREPL()(python_code)
                if is_successful:
                    print('o', end='')
                else:
                    print('e', end='')
            except Exception as e:
                print('f', end='')
                output = str(e)
            # print('exec')
            # print(python_code)
            # print()
            # print(output)
            # print("\n\n")
            messages.append({'role': 'user', 'content': "```output\n" + output + "\n```"})
        else:
            if 'contradiction' in messages[-1]['content'].lower():
                messages.append({'role': 'user', 'content': "Starting from where the contradiction was, fix the solution."})                
            else:
                messages.append({'role': 'user', 'content': "Start with the final answer, and go through your solution in reverse to find the contradiction. Print the contradiction in \\boxed{}."})
    # print()
    return list_of_messages
```

```python
import os

import pandas as pd
import polars as pl

import kaggle_evaluation.aimo_2_inference_server
```

```python
def create_starter_messages(question, index):
    cycle_size = 2
    if False:
        pass
#     elif index % cycle_size == 1:
#         # https://github.com/QwenLM/Qwen2.5-Math?tab=readme-ov-file#-hugging-face-transformers
#         return [
#             {"role": "system", "content": "Please integrate natural language reasoning with programs to solve the problem above, and put your final answer within \\boxed{}."},
#             {"role": "user", "content": question + "\n\nBegin your answer by importing sympy."}
#         ]
    else:
        # https://github.com/QwenLM/Qwen2.5-Math?tab=readme-ov-file#-hugging-face-transformers
        return [
            {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
            {"role": "user", "content": question}
        ]
```

```python
def predict_for_question(question: str) -> int:
    import os
    if not os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
        if question != "Triangle $ABC$ has side length $AB = 120$ and circumradius $R = 100$. Let $D$ be the foot of the perpendicular from $C$ to the line $AB$. What is the greatest possible length of segment $CD$?":
            return 210
    if time.time() > cutoff_time:
        return 210

    question += "\nIf the final answer is a number larger than 1 million, take modulo 1000."
    print(question)

    list_of_messages = [create_starter_messages(question, index) for index in range(16)]

    all_extracted_answers = []
    for _ in range(4):
        list_of_messages = batch_message_generate(list_of_messages)
        list_of_messages, extracted_answers = batch_message_filter(list_of_messages)
        all_extracted_answers.extend(extracted_answers)
        if not list_of_messages:
            break
        list_of_messages = batch_message_followup(list_of_messages)

    for messages in list_of_messages:
        # print for debugging
        print('--------')
        for message in messages:
            print(message['content'])
            print('----')
        
    print(all_extracted_answers)
    answer = select_answer(all_extracted_answers)
    print(answer)

    print("\n\n")
    return answer
```

```python
# Replace this function with your inference code.
# The function should return a single integer between 0 and 999, inclusive.
# Each prediction (except the very first) must be returned within 30 minutes of the question being provided.
def predict(id_: pl.DataFrame, question: pl.DataFrame) -> pl.DataFrame | pd.DataFrame:
    print("------")
    
    id_ = id_.item(0)
    print(id_)
    
    question = question.item(0)
    print(question)

    answer = predict_for_question(question)
    print(answer)    
    
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