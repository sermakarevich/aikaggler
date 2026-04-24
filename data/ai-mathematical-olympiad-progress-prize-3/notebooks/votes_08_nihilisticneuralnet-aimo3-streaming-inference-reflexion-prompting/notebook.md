# AIMO3: streaming-inference reflexion prompting

- **Author:** parthenos
- **Votes:** 449
- **Ref:** nihilisticneuralnet/aimo3-streaming-inference-reflexion-prompting
- **URL:** https://www.kaggle.com/code/nihilisticneuralnet/aimo3-streaming-inference-reflexion-prompting
- **Last run:** 2025-11-30 18:02:53.783000

---

References
- https://www.kaggle.com/code/huikang/arc-agi-2-code-approach
- https://www.kaggle.com/code/huikang/r1-distill-qwen-tir

```python
import subprocess

subprocess.run(
    ["pip", "uninstall", "--yes", "tensorflow", "matplotlib", "keras", "scikit-learn"]
)
```

```python
import os
import time
import torch
import numpy as np


def is_on_kaggle_commit() -> bool:
    return os.getenv("KAGGLE_KERNEL_RUN_TYPE") == "Batch" and not bool(
        os.getenv("KAGGLE_IS_COMPETITION_RERUN")
    )


def is_on_kaggle_interactive() -> bool:
    return os.getenv("KAGGLE_KERNEL_RUN_TYPE") == "Interactive" and not bool(
        os.getenv("KAGGLE_IS_COMPETITION_RERUN")
    )


start_time = time.time()
final_cutoff_time = start_time + (4 * 60 + 45) * 60  # 4.75 hours from start time
cutoff_times = [
    int(x) for x in np.linspace(final_cutoff_time, start_time + 12 * 60, 50 + 1)
]  # 5 minutes loading time at the start
cutoff_times.pop()

os.makedirs("solutions", exist_ok=True)

assert torch.cuda.is_available()
assert torch.cuda.device_count() == 1
```

# Serve vLLM

```python
subprocess.run(["ls", "/kaggle/usr/lib/pip_install_aimo3_1/tiktoken_encodings"])
```

```python
with open("a-vllm.log", "w") as f:
    f.write("")
```

```python
import subprocess

def start_vllm_server() -> subprocess.Popen[bytes]:
    """Start vLLM server in the background"""
    os.environ["TRANSFORMERS_NO_TF"] = "1"
    os.environ["TRANSFORMERS_NO_FLAX"] = "1"
    os.environ["VLLM_ATTENTION_BACKEND"] = "TRITON_ATTN"
    os.environ["TRITON_PTXAS_PATH"] = "/usr/local/cuda/bin/ptxas"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # https://docs.vllm.ai/projects/recipes/en/latest/OpenAI/GPT-OSS.html#troubleshooting
    os.environ["TIKTOKEN_ENCODINGS_BASE"] = (
        "/kaggle/usr/lib/pip_install_aimo3_1/tiktoken_encodings"
    )

    sequence_length = 65_536

    command: list[str] = [
        "python",
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        "/kaggle/input/gpt-oss-120b/transformers/default/1",
        "--served-model-name",
        "vllm-model",
        "--tensor-parallel-size",
        "1",
        "--max-num-seqs",
        "4",
        "--gpu-memory-utilization",
        "0.96",  # any higher may not have enough for graph capture
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
        "--dtype",
        "auto",
        "--max-model-len",
        f"{sequence_length}",
    ]

    # Start the process in the background
    with open("/kaggle/working/a-vllm.log", "w") as logfile:
        process: subprocess.Popen[bytes] = subprocess.Popen(
            command, stdout=logfile, stderr=subprocess.STDOUT, start_new_session=True
        )

    print("Logs: /kaggle/working/a-vllm.log")
    return process

# Start the server
vllm_process: subprocess.Popen[bytes] = start_vllm_server()
```

```python
from openai import OpenAI, Stream
from openai.types.chat import ChatCompletion, ChatCompletionMessageParam
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk

# Point the client to your local vLLM server
os.environ["OPENAI_API_BASE"] = "http://127.0.0.1:8000/v1"
os.environ["OPENAI_API_KEY"] = "sk-local"  # any non-empty string

client: OpenAI = OpenAI(
    base_url=os.environ["OPENAI_API_BASE"],
    api_key=os.environ["OPENAI_API_KEY"],
)

# https://github.com/vllm-project/vllm/issues/27243
# Unexpected token 2000?? while expecting start token 200006
stop_token_ids: list[int] = [
    token_id
    for token_id in range(200_000, 201_088)
    if token_id not in [200005, 200006, 200007, 200008]
]
```

```python
import time

def await_client(printing: bool = False):
    for _ in range(15 * 60):
        time.sleep(1)
        try:
            model_list = client.models.list()
            if printing:
                print(model_list)
        except NameError:
            raise  # maybe you did not run the cell initializing client
        except Exception:
            continue
        break
    else:
        raise

if is_on_kaggle_interactive():
    await_client()
```

```python
from cachetools import cached, TTLCache
from typing import Generator
import time

import os
import re


def reversed_lines(path: str, block_size: int = 4096) -> Generator[str, None, None]:
    """
    Iterate over the lines of a file in reverse order (last line first),
    without loading the entire file into memory.

    Yields lines as strings (including the trailing newline if present).
    """
    with open(path, "rb") as f:
        f.seek(0, os.SEEK_END)
        file_end = f.tell()

        buffer = b""
        pos = file_end

        while pos > 0:
            # Read a block from the end going backwards
            read_size = min(block_size, pos)
            pos -= read_size
            f.seek(pos, os.SEEK_SET)
            data = f.read(read_size)

            buffer = data + buffer
            # Split into lines
            lines = buffer.split(b"\n")
            # Keep the first (possibly incomplete) part in buffer
            buffer = lines[0]
            # The rest (from the end backwards) are full lines
            for line in reversed(lines[1:]):
                yield line.decode("utf-8", errors="replace") + "\n"

        # Finally, yield the very first line (if any)
        if buffer:
            yield buffer.decode("utf-8", errors="replace") + "\n"


@cached(cache=TTLCache(maxsize=50, ttl=10))
def get_gpu_kv_cache_usage(question_id: str | None = None) -> float:
    for line in reversed_lines("a-vllm.log"):
        pattern = r"GPU KV cache usage: ([\d.]+)%"
        match = re.search(pattern, line)
        if match:
            gpu_cache_usage = float(match.group(1))
            return gpu_cache_usage
    return 0
```

```python
if is_on_kaggle_interactive():
    resp: ChatCompletion = client.chat.completions.create(
        model="vllm-model",  # use your served name; if not set, the model path/name vLLM shows in logs
        messages=[
            {"role": "system", "content": "Reply your answer in \\boxed{}"},
            {"role": "user", "content": "How many r are there in strawberry?"},
        ],
        max_tokens=1024,
        temperature=1.0,
        extra_body=dict(min_p=0.02, stop_token_ids=stop_token_ids, chat_template_kwargs=dict(enable_thinking=True)),
    )
```

```python
if is_on_kaggle_interactive():
    print(resp.choices[0].message.reasoning_content)
```

```python
if is_on_kaggle_interactive():
    print(resp.choices[0].message.content)
```

# Prediction

```python
SYSTEM_PROMPTS = [
    "You are solving a national/international-level mathematics olympiad problem. You must rigorously define all variables, explore multiple solution strategies before committing, perform full case analysis where required, justify every nontrivial step, explicitly check boundary cases and hidden assumptions, and verify the final result using at least one independent method. Return only the final numerical answer inside \\boxed{}. The answer must be an integer in [0, 99999]. Never guess.",

    "Solve the problem with full rigor. After obtaining a candidate solution, actively attempt to refute your own answer by searching for counterexamples, re-running the logic from a different viewpoint, and stress-testing edge cases. Only after the answer survives refutation, return it in \\boxed{}. The answer must be an integer in [0, 99999]. Never guess.",

    "Solve this problem as if under IMO-level time pressure: identify the key invariant, symmetry, or extremal principle early, avoid brute force unless strictly justified, compress reasoning without sacrificing correctness, and perform at least one final arithmetic verification pass. Return only the final integer answer in \\boxed{}, with 0 ≤ answer ≤ 99999. Never guess.",

    "You must attempt at least two fundamentally different solution approaches (e.g., algebraic vs geometric, combinatorial vs number-theoretic). Proceed with the more rigorous one and use the other as a verification tool. Return only the verified final answer in \\boxed{}, where the answer is an integer in [0, 99999]. Never guess.",

    "Solve the problem rigorously. If at any point a step relies on an unproven assumption, a jump in logic is detected, or the computation becomes inconsistent, you must restart the solution from first principles. Return only the final verified integer answer inside \\boxed{}, with 0 ≤ answer ≤ 99999. Never guess."
]
```

```python
def extract_boxed_text(text: str) -> str:
    """Extract text inside \\boxed{} from LaTeX-formatted text"""
    import re

    pattern: str = r"oxed{(.*?)}"
    matches: list[str] = re.findall(pattern, text)
    if not matches:
        return ""
    for match in matches[::-1]:
        if match != "":
            return match
    return ""


def is_valid_answer_string(text: str) -> bool:
    try:
        if int(text) == float(text):
            if 0 <= int(text) <= 99_999:
                # now AIMO answers no longer need modulo
                return True
    except Exception:
        pass
    return False
```

```python
from collections import Counter

completed_question_ids: set[str] = set()
question_id_to_counter: dict[str, Counter] = {"": Counter()}


import math
from collections import Counter


def vote_answer(question_id: str, force_answer: bool = False) -> int | None:
    # reads counter from global
    counter = question_id_to_counter[question_id]
    if force_answer and not counter:
        print(f"Current GPU usage {get_gpu_kv_cache_usage()}")
        print("force_answer=True but no answer recorded")
        completed_question_ids.add(question_id)
        return 12453

    # voting mechanism
    modified_counter = Counter()
    for value, count in counter.items():
        # re-weighted because smaller answers seems to be wrong
        # "1.25 +" because log(1) = 0
        modified_counter[value] += math.log(1.25 + abs(value)) * count

    total_score = sum(modified_counter.values())
    score_list = sorted(
        (score, counter[value], value) for value, score in modified_counter.items()
    )
    if force_answer:
        print(f"score_list | {total_score:8.1f} over {sum(counter.values())} attempts")
        print(f"Current GPU usage {get_gpu_kv_cache_usage()}")
        for score, count, value in score_list[::-1]:
            print(f"{value:10}   {score:8.1f} {count:8d}")
        return score_list[-1][-1]
    if score_list[-1][0] > max(3, total_score / (2 + math.log(1 + total_score))):
        if len(score_list) == 1:
            completed_question_ids.add(question_id)
        else:
            if score_list[-1][0] - score_list[-2][0] > 1:
                # win by a certain number of points at least
                completed_question_ids.add(question_id)
    return None
```

```python
import time


def generate_solution(
    question_text: str, question_id: str = "", solution_index: int = 0, system_prompt: str = ""
) -> str:
    if question_id in completed_question_ids:
        return ""
    if time.time() >= cutoff_times[-1]:
        return ""

    if not system_prompt:
        system_prompt = SYSTEM_PROMPTS[0]
    
    messages: list[ChatCompletionMessageParam] = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {"role": "user", "content": question_text},
    ]

    text_response_to_save = ""
    generation_idx = 0
    for iteration in range(2):
        text_response = ""
        breaking = False

        stream: Stream[ChatCompletionChunk] = client.chat.completions.create(
            model="vllm-model",  # use your served name; if not set, the model path/name vLLM shows in logs
            messages=messages,
            temperature=1.0,
            stream=True,
            extra_body=dict(min_p=0.02, stop_token_ids=stop_token_ids),
            reasoning_effort="high",
        )

        for chunk in stream:
            generation_idx += 1
            chunk_text = (
                chunk.choices[0].delta.reasoning_content
                if chunk.choices[0].delta.reasoning_content is not None
                else chunk.choices[0].delta.content
            )
            if chunk_text:
                text_response += chunk_text
            if question_id in completed_question_ids:
                # stop generating if we have finalized on an answer
                breaking = True
            if time.time() >= cutoff_times[-1]:
                breaking = True
            if generation_idx > 60_000:
                breaking = True
            if breaking:
                break
            # instead of breaking = True, so we want to inject instructions for these conditions
            if "}" in chunk_text and is_valid_answer_string(extract_boxed_text(text_response)):
                break
            if iteration == 0 and generation_idx > 50_000:
                break

        messages.append({"role": "assistant", "content": text_response})
        text_response_to_save += text_response
        stream.close()

        if breaking:
            break

        boxed_text = extract_boxed_text(text_response)
        if not is_valid_answer_string(extract_boxed_text(text_response)) and iteration == 0 and generation_idx > 50_000:
            print("follow-up - guess answer")
            user_follow_up = "The answer is expected to be an integer between 0 and 99999 inclusive. Make an educated guess (e.g. lower bound, upper bound, ...) on your final answer and put in \\boxed{}."
            messages.append({"role": "user", "content": user_follow_up})
            text_response_to_save += "\n===\n" + user_follow_up + "\n===\n"
        elif not is_valid_answer_string(boxed_text):
            print("follow-up - boxed answer")
            user_follow_up = "The answer is expected to be an integer between 0 and 99999 inclusive. Place your final answer in \\boxed{}. Do not guess the answer."
            messages.append({"role": "user", "content": user_follow_up})
            text_response_to_save += "\n===\n" + user_follow_up + "\n===\n"
        elif int(boxed_text) <= 10:
            print("follow-up - are you sure")
            user_follow_up = "Are you sure that is the answer? Do not guess the answer."
            messages.append({"role": "user", "content": user_follow_up})
            text_response_to_save += "\n===\n" + user_follow_up + "\n===\n"
        elif iteration == 0 and get_gpu_kv_cache_usage(question_id) < 10:
            print("follow-up - have you verified")
            user_follow_up = "Have you verified your answer?"
            messages.append({"role": "user", "content": user_follow_up})
            text_response_to_save += "\n===\n" + user_follow_up + "\n===\n"
        else:
            # answer found, no issues detected, proceed to answering
            break

    boxed_text = extract_boxed_text(
        text_response_to_save
    )  # expected to use the full conversation        

    if question_id and text_response_to_save:
        answer_suffix = ""
        if is_valid_answer_string(boxed_text):
            answer_suffix = f"-{boxed_text}"
        with open(f"solutions/{question_id}/{solution_index:04d}-{generation_idx}{answer_suffix}.txt", "w") as f:
            f.write(text_response_to_save)

    if is_valid_answer_string(boxed_text):
        question_id_to_counter[question_id][int(boxed_text)] += 1
        vote_answer(question_id)

    return boxed_text
```

```python
if is_on_kaggle_interactive():
    generate_solution("What is 1+1?")
```

```python
import concurrent.futures
from collections import Counter


def solve(question_text: str, question_id: str = "") -> int:
    await_client()
    print(f"processing {question_id}")
    os.makedirs(f"solutions/{question_id}", exist_ok=True)
    question_id_to_counter[question_id] = Counter()
    completed_question_ids.discard(question_id)  # just in case question_id collides

    if question_id and time.time() > cutoff_times[-1]:
        print("timeout did not solve")
        return 12314

    num_generations = 4
    get_gpu_kv_cache_usage(question_id)  # run once to prevent running in the first batch of execution

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        # run in parallel with different system prompts
        results = executor.map(
            generate_solution,
            [question_text] * num_generations,
            [question_id] * num_generations,
            list(range(num_generations)),
            SYSTEM_PROMPTS,
        )
        list(results)

    final_answer = vote_answer(question_id, force_answer=True)
    return final_answer
```

```python
if is_on_kaggle_interactive():
    solve("What is 1+1?")
```

# Submission

```python
import os

import kaggle_evaluation.aimo_3_inference_server
import pandas as pd
import polars as pl

df = pd.read_csv(
    "/kaggle/input/ai-mathematical-olympiad-progress-prize-3/reference.csv"
)

id_to_answer: dict[str, str] = dict(zip(df["id"], df["answer"]))
df.drop("answer", axis=1).to_csv("reference.csv", index=False)

correct = 0
total = 0
```

```python
def predict(id_: pl.Series, problem: pl.Series) -> pl.DataFrame | pd.DataFrame:
    """Make a prediction."""
    global id_to_answer
    global correct
    global total

    # Unpack values
    question_id: str = id_.item(0)
    question_text: str = problem.item(0)

    # Generate prediction
    prediction = solve(question_text, question_id=question_id)
    completed_question_ids.add(question_id)
    cutoff_times.pop()

    # ------------------------ SCORING ------------------------
    try:
        true_answer = int(id_to_answer.get(question_id, -1))
    except:
        true_answer = -1

    total += 1
    if prediction == true_answer and true_answer != -1:
        correct += 1
        print(f"[debug] correct | score={correct}/{total}")
    else:
        print(f"[debug] WRONG: predicted {prediction}, but actual answer {true_answer} | score={correct}/{total}")
    # ---------------------------------------------------------------

    return pl.DataFrame({"id": id_, "answer": prediction})
```

```python
inference_server = kaggle_evaluation.aimo_3_inference_server.AIMO3InferenceServer(
    predict
)

if os.getenv("KAGGLE_IS_COMPETITION_RERUN"):
    inference_server.serve()
else:
    inference_server.run_local_gateway(("reference.csv",))
```