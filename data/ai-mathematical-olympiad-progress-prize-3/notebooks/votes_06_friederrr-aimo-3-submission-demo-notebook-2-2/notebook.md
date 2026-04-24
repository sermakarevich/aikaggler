# AIMO 3 Submission Demo Notebook 2/2

- **Author:** Simon Frieder
- **Votes:** 615
- **Ref:** friederrr/aimo-3-submission-demo-notebook-2-2
- **URL:** https://www.kaggle.com/code/friederrr/aimo-3-submission-demo-notebook-2-2
- **Last run:** 2025-11-20 18:55:37.007000

---

# Summary
This notebook aims to serve as a quick introduction to building a submission for the AIMO3 competition. It is a refactored version of the notebook that won the 'Early Sharing Prize' for the AIMO2 competition. A key difference is that it outputs a 5-digit final answer, instead of a three-digit final answer for AIMO2. To make the competition more accessible to people new to Kaggle, we provide a large number of comments, explanations, and links to documentation of the libraries used.

Note that this notebook is not designed to achieve a high score on any of the leaderboards, but to serve as a quick start to building your own solution. Therefore, we suggest not simply copying, but adapting and experimenting with parameters, prompting strategies, and so forth.

# Dependencies & Accelerator

### Enabling Acceleratos
To use the provided accelerators, go to `Settings > Accelerator` and select the appropriate GPU to use.

### Dependencies
To install dependencies, we employ a **secondary 'utility' notebook**, which uses `pip install` with `/kaggle/working` as the target directory to pre-install all packages, using spcific versions (frameworks like vLLM typically require latest versions), before the runtime of this notebook is started. Installing packages in this more tedious way is necessary since, for competition submissions, **direct pip installs won't work since the notebook must have the internet turned off** before submitting.

To link a new 'utility' notebook, navigate to the right sidebar and click `Add Input`, then filter by `Your Work` and `Utility Scripts`. Any *public* notebook you have created that you have tagged as "Utility Script" (using `File -> Set as Utility Script` in the 'utility' notebook) should then show up under a "Utility Scripts" tab under the `input` section.

Below are a few small *sanity checks* for the correct torch and numpy versions. They also check if you have enabled a GPU accelerator, as described in the paragraph above.

It is necessary to uninstall a few packages first to avoid conflicts with newer vLLM and numpy versions.

```python
%pip uninstall --yes "tensorflow" "matplotlib" "keras" "scikit-learn"
```

```python
import torch
assert torch.__version__ == "2.8.0+cu128", (f"Torch version is {torch.__version__} instead of 2.8.0+cu128")
assert torch.cuda.is_available and torch.cuda.device_count() == 1, "GPU not enabled"
```

```python
import numpy as np
assert np.__version__ == "2.2.0", (f"Numpy version is {np.__version__} instead of 2.2.0")
```

# Imports

We add the path to the CUDA PTX assembler in order to enable vLLM to compile CUDA graphs as it's in a non-standard location on Kaggle. This results increased throughput :)

```python
import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TRANSFORMERS_NO_FLAX"] = "1"
os.environ["TRITON_PTXAS_PATH"] = "/usr/local/cuda/bin/ptxas"
```

```python
import time
import warnings
import re
import tempfile
import subprocess
from collections import Counter, defaultdict
from typing import Optional

# Data Processing
import pandas as pd
import polars as pl

# LLM Inference
from transformers import set_seed
import torch
from vllm import LLM, SamplingParams
import kaggle_evaluation.aimo_3_inference_server

#fixed seed to get similar score
set_seed(42)
pd.set_option('display.max_colwidth', None)
cutoff_time = time.time() + (4 * 60 + 45) * 60

warnings.simplefilter('ignore')
```

# Constants

It is good practice to have all constants and configurable parameters that you may change at the top of the file. This allows for quick iteration and changes without scanning the whole notebook each time.

On Kaggle, you will need to download the weights first for this to work. To this end, download them locally and then upload them to Kaggle. Then, you can edit the LLM_MODEL_PATH and input the Kaggle input directory to the weights.

Alternatively, you could also link model weights already uploaded to Kaggle via the Add Input functionality.

```python
# Path to the downloaded model weights.
LLM_MODEL_PATH = 'Qwen/Qwen3-32B-FP8'
```

# Loading the Model
### Setting up the environment variables
Each CUDA-enabled device has an ID. Here, we need to set an environment variable using the os package for all devices that should be visible for inference. 0 will enable torch and vLLM to see one GPU with this ID.

```python
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
```

### Creating an Inference Engine
---
Use the vLLM model serving engine to load the downloaded weights automatically, specifying the desired precision and other configurations. 
The parameters given for vLLM are documented [here](https://docs.vllm.ai/en/v0.7.2/serving/engine_args.html).
Depending on which model you choose, the default setting may vary, so be sure to read the documentation! 

#### Precision
`dtype="bfloat16"` 

The `dtype` parameter controls the precision with which the weights are loaded. Here, NVIDIA's special half-precision format `bfloat16` is used. 
Which precision can be applied depends on what is available from the weights and on their quantization. Lower precision generally lowers the memory footprint, but will result in decreased accuracy.

#### Maximum Number of Sequences
`max_seq_len=256`

This parameter is specific to vLLM and controls the concurrent requests/prompts that are processed at once. Increasing this will allow the vLLM to fully utilize all GPUs, which is desirable to achieve maximum performance. 
However, larger values naturally come with an increased memory footprint, so there is a risk of running out of memory.

#### Context Length 
`max_model_len=32768` 

An important parameter is the context length, which controls
how many tokens a model will use for its prediction internally.
Below, it is manually set to 32768. If not set, vLLM will
use the model's default configuration instead.

#### GPU Memory Utilization
`gpu_memory_utilization=0.96`

Specifies the fraction of memory vLLM is allowed to reserve per GPU. 
The default of `0.9` is usually sufficient, although higher values
allow the engine to use more memory, possibly fitting larger models,
context windows, concurrent sequences, ...

```python
llm = LLM(
    LLM_MODEL_PATH,
    dtype="bfloat16",
    max_num_seqs=256,
    max_model_len=32768,       
    trust_remote_code=True,     
    tensor_parallel_size=1,      
    gpu_memory_utilization=0.96, 
)
```

### Tokenizer

After having created the inference engine (i.e., the `LLM` instance), you will 
also need an appropriate tokenizer for the model. It is necessary to use 
the same tokenizer that the model comes pre-configured with. Changing it will lead to unexpected results. For convenience, vllm allows the loading
of the default tokenizer with the simple command below:

```python
tokenizer = llm.get_tokenizer()
```

### Sampling Parameters

Finally, vLLM also offers the ability to configure inference parameters via the
`SamplingParams` class. These will be used, along with the conversation history, to be passed
to the `llm.generate` call to configure the behaviour. The complete spec can
be found [here](https://docs.vllm.ai/en/v0.6.4/dev/sampling_params.html).
In this notebook, only a few options are configured.

We encourage users of this notebook to use it as a baseline to experiment with different combinations and values of these parameters. The perfect setting also depends on the model, prompts, and use case (for example, reasoning vs short answer checks will require different parameters).
Usually, however, model authors provide a good starting point that they recommend.

#### Temperature
`temperature=1.0`

Controls the 'creativity' or 'randomness' of the generation process. A higher value (closer to `1.0`) will provide varying and diverse outputs, while lower values will be more strict.

#### Min. P
`min_p=0.01` 

Controls the parameter `p` of top-p sampling. This removes all tokens from consideration of the decoding process that have a probability lower than `min_p`. This can remove very rare and unexpected tokens.

#### Skip Special Tokens
`skip_special_tokens=True`

Removes special tokens (beginning of sequence, ...) from the generated model output.
This is very useful, as end-users usually don't want to have these symbols in their generated text.

#### Max. Tokens
`max_tokens=32768`

Configures the maximum output length of the model. This is calculated as `len(prompt) + len(generated_tokens)`, so longer prompts will inhibit the ability of the model to generate long outputs as well. Should be kept relatively high by default, to allow modern LLMs
to complete their thinking/reasoning traces, which may take up many tokens.

```python
sampling_params = SamplingParams(
    temperature=1.0,
    min_p=0.01,
    skip_special_tokens=True,     
    max_tokens=32768,
)
```

# Utility Functions
As the next step, functions that help the inference code perform will be introduced. These will focus on formatting prompts, extracting digits enclosed in `\boxed{}`, and similar.

### Extracting from Text
Next are some functions that extract sections of output that the LLM produced during inference. Mainly, these will be used to extract runnable Python Code and Boxed Answers.

#### Extracting Boxed Answers
A crucial step is extracting the predicted final answers from the fuzzy output of an LLM. For this purpose, we want to look for digits in the text enclosed in a `\boxed{}` LaTeX command. The function `extracted_boxed_answers` extracts all numbers that were contained in `\boxed{}` environments using `re`.

This specific implementation returns __all__ boxed integers.
However, another solution could be to just take the __last__ boxed 
answer, as this is usually the way SotA LLMs format their answers (i.e., in a "Final Answer: ..." paragraph). This is an easy change to implement by simply using a slice on the return value of the `ans` list.

```python
def extract_boxed_answers(text: str) -> list[int]:
    pattern = r'oxed{(.*?)}'
    matches = re.findall(pattern, text)
    if not matches:
        return []
    ans = []
    for content in matches:
        if content.isdigit():
            # Answer contains only digits already -> record
            num = content
        else:
            # Otherwise, there are other symbols
            # --> Use `re` to find all matches and
            # extract the last one
            nums = re.findall(r'\d+', content)
            if not nums:
                # Skip if no numbers were found
                continue 
            num = nums[-1]
        ans.append(int(num))
    return ans
```

#### Majority Voting

To avoid getting unlucky with a single completion, it is common practice to predict the answer to each problem multiple times. A simple strategy to choose the single final answer is to then employ majority voting over all answers (i.e., choose the most frequent answer). Here, this is implemented using a simple `Counter` object.

It also implements checks for valid answers that are between 0 and 99999, which was one of the rules for AIMO2 problems. For future competitions, this may change, so be sure to adapt based on the official rules and recommendations!

```python
#select the final answer based on the frequency (majority voting)
def select_answer(answers: list):
    valid_answers = []
    for answer in answers:
        try:
            # Disregard answers that are not integers by
            # comparing their float and int values.
            if int(answer) != float(answer):
                continue

            # Check if the int answer is between 0 and 99999, as 
            # per AIMO3 competition rules.
            if 0 <= int(answer) <= 99999:
                valid_answers.append(int(answer))
        except:
            pass # Skip conversion errors (i.e., converting text to an int for float)
    
    # As a last resort, just guess a number instead :)
    if not valid_answers:
        print("Guessing random number :)")
        return 49
    # Extract the most frequent answer from the Counter object.
    # NOTE: Counter.most_common breaks ties in order of the first element occurring, so be wary of that!
    # (i.e., you could make this deterministic by sorting the valid_answer list first)
    answer, _ = Counter(valid_answers).most_common(1)[0]
    # Answer was already checked to be in the correct range.
    return answer
```

#### Extracting Python Code

The `extract_python_code` function receives text as input and extracts any Python code enclosed in a Python environment with triple backticks using the `re` module. It will return a list of all Python code enclosed in such environments, which will later be evaluated by executing the code.

The second function `process_python_code` adds basic imports to the given code to make it execute in many cases where imports may be missing (forgetting imports is a common failure mode for LLMs).

```python
#extract all code segments
def extract_python_code(text: str) -> list[str]:
    # Build a regex pattern as a RAW string that matches any characters in between a markdown python environment
    # triple backticks + python, followed by text, then again triple backticks
    pattern = r'```python\s*(.*?)\s*```'
    # Find all python code segments in the text
    matches = re.findall(pattern, text, re.DOTALL)
    return matches

#process the code
def process_python_code(query):
    query = "import math\nimport numpy as np\nimport sympy as sp\n" + query
    current_rows = query.strip().split("\n")
    ans = "\n".join(current_rows)
    print(f'Processed python code: {ans}')
    return ans
```

#### Executing Extracted Python Code

To execute the extracted Python code that the LLM may generate,
we will re-use a solution that the AIMO1 winning team, NuminaMath, developed. Their notebook at the time can be found [here]( https://www.kaggle.com/code/lewtun/numina-1st-place-solution).

Their code works by writing an extracted query (i.e., extracted by `extract_python_code` and constructed by `process_python_code`)
into a temporary `.py` file and then executing it inside of a `subprocess`.

```python
#Python REPL to execute code. taken and modified from NuminaMath Solution
#NuminaMath solution can be found here : https://www.kaggle.com/code/lewtun/numina-1st-place-solution

class PythonREPL:
    def __init__(self, timeout=8):
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

# Batch Processing Functions
Below follows a collection of functions that server to process conversation histories
with language models, from generation to code execution and answer extraction.

To make the typing easier to follow, we define a type `MessageBatch` that is a list
of conversation histories. Each conversation history is a list of dictionaries.
An example with two conversation histories is show below:
```python
[
    [
        {"role":"system", "content":"You are a helpful math assistant"},
        {"role":"user", "content":"What is 1+1?"},
    ],
    [
        {"role":"system", "content":"You are a precise mathematician tasked with solving difficult problems"},
        {"role":"user", "content":"What is the result of 2025!/2023!"},
        {"role":"assistant", "content":"Hmm, as first step ..."},
    ]
]
```
Note: when using any lists that are not copied betwen calls, take care that they are **mutable** and the
contents may be changed in-place!

the expected format for a single message in a conversation history is always `{"role":..., "content":...}`,
as we use `apply_chat_template` from the `transformers` library to transform the chat histories to prompts.
See [here](https://huggingface.co/docs/transformers/main/chat_templating) for more detailed information.

```python
MessagesBatch = list[list[dict[str, str]]]
```

#### Generation from MessageBatch

The first function `batch_message_generate` will take in a `MessageBatch` (i.e., a list of conversation histories) and pass it through the tokenizer, apply the chat template, and finally generate a completion, using the model loaded by the `llm` with the `sampling_params`. Finally, each completion is appended as an `assistant` response to the respective conversation history. The resulting updated batch is then returned.

```python
#generate prompts in batch
def batch_message_generate(msg_batch: MessagesBatch) -> MessagesBatch:
    list_of_texts = [
        tokenizer.apply_chat_template(
            conversation=messages,
            tokenize=False,
            add_generation_prompt=True
        )
        for messages in msg_batch
    ]
    
    request_output = llm.generate(
        prompts=list_of_texts,
        sampling_params=sampling_params,
    )
    
    # Update the contents of msg_batch in-place
    # --> Each internal list tracks one conversation history, which
    # is updated here
    for messages, single_request_output in zip(msg_batch, request_output):
        messages.append({'role': 'assistant', 'content': single_request_output.outputs[0].text})
        print(messages[-1])

    return msg_batch
```

### Filtering a MessageBatch

After generating answers, we would also like to achieve two things:

* Extract 'boxed' answers from each latest completion for each conversation history.
* Drop all conversation histories that produced a boxed answer, as it is finished.

Note that these steps should be changed by you to accommodate the strategy you are going for. However, it is a good starting point to keep prompting until a boxed answer is generated by the model.

We achieve this by extracting the last element from each conversation history (this will always be an assistant response generated from `batch_message_generate`) and calling `extract_boxed_answers` on it. Finally, the indices of the finished conversation histories are tracked, and only the non-finished ones are returned as a new `MessageBatch`.

```python
#filter answers from the responses
def batch_message_filter(msg_batch: MessagesBatch, list_of_idx: list[int]) -> tuple[MessagesBatch, list[int], list[int]]:
    global answer_contributions
    extracted_answers: list[int] = []
    msgs_to_keep: MessagesBatch = []
    idx_to_keep: list[int] = []
    for idx,messages in zip(list_of_idx, msg_batch):
        # Get boxed answers from the LATEST completion in the conversation history
        answers = extract_boxed_answers(messages[-1]['content'])

        # if latest content has an answer, don't keep the message for the next run.
        if answers:
            extracted_answers.extend(answers)
            for answer in answers:
                answer_contributions[answer].append(idx) # Globally track all answers
        # Else, record the message as it hasn't yet produced an answer.
        else:
            msgs_to_keep.append(messages)
            idx_to_keep.append(idx)
    return msgs_to_keep, extracted_answers, idx_to_keep
```

### Extracting & Executing Python Code

This step is what basically implementes the TIR (Tool-Integrated-Reasoning) part of the pipeline. 
It will achieve two things:

* Extract any Python code from the latest completions
* Execute that Python code using the `PythonREPL` from Project Numina
* Process the answer to fit the competition schema (i.e., for AIMO2 this was taking the answer mod 1000)

NOTE: The answer format for AIMO3 may be different from AIMO2, so take that into account.

We achieve this by iteration through each conversation history in a `MessageBatch`, taking the latest completion generated by
the model and extracting all python code blocks with `extraxt_python_code`. For each of those blocks,
we then add imports and formatting with `process_python_code`, which is then passed to an instance of `PythonREPL` to execute
it in a subprocess. The resulting output is then scanned for numbers, which are taken modulo 1000 to produce a three digit answer
(again, you will probably have to change this for AIMO3).

```python
#execute code and generate answer for all elements in batch
def batch_execute_and_get_answer(list_of_messages: MessagesBatch) -> list[int]:
    ans = []
    for messages in list_of_messages:
        # Get all Python code blocks from the latest completion for the conversation history
        python_code_list = extract_python_code(messages[-1]['content'])
        for python_code in python_code_list:
            # Add imports + formatting to the code block
            python_code = process_python_code(python_code)
            try:
                # Execute the Python code in a subprocess and get answers
                success, output = PythonREPL()(python_code)
                if not success:
                    continue # Skip if code execution failed
                patten = r'(\d+)'
                matches = re.findall(patten, output)
                if not matches:
                    continue # Skip if no digits found in answer
                for match in matches:
                    ans.append(int(match)) # Convert answers to numeric values.
            except Exception as e:
                output = str(e)
            print(f'python code output: {output}')
    return ans
```

# Inference
### Prompts

The code below defines the list of prompts that will be passed to the model as a template before
each problem. 
Prompt Engineering is a **crucial** part of the inference process, so be sure
to give time and thought to the prompts! 

The AIMO2 Early Sharing Price (ESP) notebook used the following prompts, which will result in five slightly different reasoning paths (more or less).

```python
thoughts = [
    'Please use chained reasoning to put the answer in \\boxed{}.',
    'Please reflect and verify while reasoning and put the answer in \\boxed{}.',
    'Solve the following problem using concise and clear reasoning by placing the answer in \\boxed{}.',
    'You are a helpful and reflective maths assistant, please reason step by step to put the answer in \\boxed{}.',
    'You are the smartest maths expert in the world, please spike this question and put the answer in \\boxed{}.'
]
```

### Predicting a single problem
The function `predict_for_question` implements prediction of a singular problem
using the following sequence:

1. For each prompt in `thoughts`, produce a conversation history dict with the thought as system- and
the problem as user-prompt.
2. Then, until we reach `max_rounds`, repeat the following:
    1. Prompt the `llm` instance with `batch_message_generate`, which updates the conversation histories in place with
    the generated completions.
    2. Go over each of the newest additions to the conversation histories (i.e., the just-generated completions), 
    extract any Python code, execute it, and record the results with `batch_execute_and_get_answer`.
    3. Go over each of the newest additions to the conversation histories (i.e., the just-generated completions)
    and extract the last integer enclosed in `\boxed{}` with `batch_message_filter`. This will also filter the conversation history,
    removing all those that have already resulted in an answer in place.
    4. Record all answers and their indices.
    5. Early stop if all conversation histories have ended in an answer already (i.e., when `msgs_batch` is empty)
3. Apply majority voting over all extracted answers from both Python execution and integers that were enclosed in `\boxed{}`.

```python
# Global counter for which traces contributed to which answers.
answer_contributions = defaultdict(list)

def predict_for_question(question: str, max_rounds: int = 1) -> int:
    global answer_contributions
    
    # Submit a random guess if we have run out of time.
    if time.time() > cutoff_time: 
        return 210
        
    # Create 5 different prompts for each 'thought' defined above.
    msgs_batch: MessagesBatch = [
        [
            {"role": "system", "content": t},
            {"role": "user", "content": question}
        ] for t in thoughts
    ]
    
    all_extracted_answers = []
    list_of_idx: list[int] = list(range(len(msgs_batch)))
    for round_idx in range(max_rounds):
        print(f"round {round_idx}")
        
        # Prompt the LLM and update the batch of messages in-place
        # --> New conversations will be added to every internal list
        # directly.
        msgs_batch = batch_message_generate(msgs_batch)
        
        # Extract Python Code from the LAST element of each conversation
        # history (i.e., the one that was just generated for each)
        extracted_python_answer = batch_execute_and_get_answer(msgs_batch)

        # Try and extract a boxed answer from the latest item in each 
        # conversation history. Conversation histories that HAVE an answer
        # are removed and their answer recorded.
        # --> msgs_batch is overwritten with only those conversations that
        # didn't yet produce an answer.    
        msgs_batch, extracted_answers, list_of_idx  = batch_message_filter(msgs_batch, list_of_idx)

        # Record ALL extracted answers (from Python Code + \boxed{} digits)
        all_extracted_answers.extend(extracted_python_answer)
        all_extracted_answers.extend(extracted_answers)

        print("extracted boxed answers:",extracted_answers)
        print("extracted python answers:",extracted_python_answer)
        print("all extracted answers:",all_extracted_answers)

        # If there are no more conversation histories, it means all of them
        # have produced a boxed answer and were thus removed, and their answers were recorded.
        if not msgs_batch:
            break
    
    # Apply majority voting over ALL extracted answers (from python AND boxed)
    answer = select_answer(all_extracted_answers)
    print("answer:",answer)
    return answer
```

# Setup for Kaggle

### The predict function
Kaggles inference server works by passing in a `predict` function with a specific 
signature that handles predictions for a single problem.

The function must fit exactly the signature, as given below, and the following must hold:

* The function should **return a single integer** between **0 and 99999**, inclusive.
* Take care that each call of `predict` returns a final answer **within the allotted time**.

```python
def predict(id_: pl.DataFrame, question: pl.DataFrame, answer: Optional[pl.DataFrame] = None) -> pl.DataFrame:
    id_ = id_.item(0)
    print("------")
    print(id_)
    question_str = question.item(0)
    print(question_str)
    answer = predict_for_question(question_str)
    print("------\n\n")
    return pl.DataFrame({'id': id_, 'answer': answer})
```

### Starting the inference server
Kaggle provides a package on their platform called `kaggle_evaluation` which allows
users to connect to a remote or spawn a local inference server for a competition.
The inference server is initialized with our `predict` function as a parameter.

To run problems from a file, it uses `run_local_gateway` (if local) or serve (when connecting to a remote) to receive the input problems. You will not have to worry about this part, as Kaggle will provide the code for it in any case.

If running locally, take care that the CSV used as input only contains the columns
`id,problem`. You will have to manually remove any other columns that may be used for
analysis of your solution (answers, metadata, ...).

```python
inference_server = kaggle_evaluation.aimo_3_inference_server.AIMO3InferenceServer(
    predict
)

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    inference_server.run_local_gateway(
        ('/kaggle/input/ai-mathematical-olympiad-progress-prize-3/test.csv',)
    )
```

# Reference
As mentioned above, this notebook is based on the notebook by Md Boktiar Mahbub Murad submitted for the AIMO2 'Early Sharing Prize'. To make the notebook a bit more approachable, we cleaned it and adding copious comments & documentation for the
contestants of AIMO3.