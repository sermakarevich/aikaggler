# [LB 20] QWQ-32B-preview Optimized inference

- **Author:** Md Boktiar Mahbub Murad
- **Votes:** 1097
- **Ref:** mbmmurad/lb-20-qwq-32b-preview-optimized-inference
- **URL:** https://www.kaggle.com/code/mbmmurad/lb-20-qwq-32b-preview-optimized-inference
- **Last run:** 2024-12-09 15:38:12.287000

---

<h1 style="text-align: center; font-size: 1.5em; font-weight: bold; color: #2c3e50; margin-bottom: 0.5em;">
   More Ain't Always Better - Early Sharing Prize Winning Notebook
</h1>


<h2 style="text-align: center; font-size: 1.5em; color: #34495e; font-style: italic; margin-top: 0;">
   QWQ-32B-Preview Optimized Inference
</h2>
<div align="center">
    <img src="https://i.ibb.co/9rx4pbX/AIMO.png">
</div>

<h1 style="text-align: center; font-size: 2em; font-weight: bold; color: #2c3e50; margin-bottom: 0.5em;">
   AI Mathematical Olympiad - Progress Prize 2
</h1>
<h2 style="text-align: center; font-size: 1.5em; color: #34495e; margin-top: 0;">
    Team: Md Boktiar Mahbub Murad
</h2>

<h2 style="text-align: center; font-size: 1.5em; color: #34495e; margin-top: 0;">
   Public LB Score: 20
</h2>
# Overview
<h3>Competition Challenge</h3>

<ul>
  <li>Solve <strong>national-level math challenges</strong> using artificial intelligence models.</li>
  <li>These problems are <strong>AI Hard</strong> in terms of the mathematical reasoning required.</li>
  <li>There are a total of <strong>100 test</strong> problems:
    <ul>
      <li><strong>50</strong> test problems on the <strong>public leaderboard</strong> (we can submit one solution daily to check our score on this set to validate our models).</li>
      <li><strong>50</strong> test problems on the <strong>private leaderboard</strong>. The final result will be based on these 50 problems, which are currently hidden.</li>
    </ul>
  </li>
</ul>

<p>In this Notebook, I present a solution that scores 20/50 on the public leaderboard. I was the <strong>first one to score 20</strong> on the leaderboard and make my solution public, thus making this solution a candidate to win the <a href="https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-2/overview#:~:text=Early%20Sharing%20Prize%3A%20%2420%2C000.%20An%20additional%20%2420%2C000%20cash%20prize%20will%20be%20awarded%20for%20sharing%20high%2Dscoring%20public%20notebooks%20early%20in%20the%20competition%20to%20encourage%20participants%20to%20share%20information%20earlier%20and%20help%20the%20community%20make%20more%20progress%20over%20the%20course%20of%20the%20competition.">early sharing prize</a>.</p>

<p>Before diving into the solution, a brief intro about me:</p>

<p>I am <strong>Md Boktiar Mahbub Murad</strong> from <strong>Bangladesh</strong>. I completed my BSc in EEE from Bangladesh University of Engineering and Technology. I'm currently working as an <strong>AI Research Engineer</strong> at <strong>Celloscope Ltd.</strong>, Bangladesh. I'm actively looking for PhD opportunities in AI/ML in the upcoming Fall'2025 session.</p>

# Summary 

This Notebook uses an **AWQ quantized** [version](https://huggingface.co/KirillR/QwQ-32B-Preview-AWQ/tree/b082e5c095a17c50cc78fc6fe43a0eae326bd203) of the [QwQ-32B-preview](https://huggingface.co/Qwen/QwQ-32B-Preview) model. **QwQ-32B-Preview** is an experimental research model developed by the **Qwen Team**, focused on advancing AI reasoning capabilities. The key advantage of this model is instead of just simply solving the problem using **Chain-of-thought** reasoning, it reflects and tries to **verify and justify** it's solution before settling on the final solution. This gives it a better reasoning ability. It shows excellent performance across different math benchmarks. More details in their [blog post](https://qwenlm.github.io/blog/qwq-32b-preview/).


Our solution in short does the following : 
1. Uses **Chain-of-thought self-consistency** method to solve the problems using the **QwQ-32B-preview** model
2. Uses **sampling method** instead of the greedy decoding to generate multiple answers (**temperature = 1**)
3. Uses different prompts (**prompt engineering**) to approach the problems differently and add diversity
4. Chooses the **most consistent** answer among the candidate answers


**Score on Reference Set :**

- This Notebook scores **4/10** on average on the reference set
- It takes **57 minutes** to run on the reference set
- Takes **~5 min** on average to solve each problem

The AWQ-quantized model is hosted on huggingface [here](https://huggingface.co/MBMMurad/QwQ-32B-preview-AWQ-AIMO-earlysharing). Feel free to give it a try.

# Acknowledgements

<p style="font-size: 1.1em; color: #555;">
   This notebook is modified from 
   <a href="https://www.kaggle.com/code/boristown/qwen-qwq-32b-preview-deepreasoning" target="_blank" style="color: #007acc; text-decoration: none;">
       this fantastic notebook
   </a> by 
   <a href="https://www.kaggle.com/boristown" target="_blank" style="color: #007acc; text-decoration: none;">
       @boristown,
   </a>which was also inspired by 
   <a href="https://www.kaggle.com/code/huikang/qwen2-5-72b-instruct-with-tir" target="_blank" style="color: #007acc; text-decoration: none;">
       this great notebook
   </a> by 
   <a href="https://www.kaggle.com/huikang" target="_blank" style="color: #007acc; text-decoration: none;">
       @huikang
   </a> and 
   <a href="https://www.kaggle.com/code/konstantinboyko/qwen2-5-72b-instruct" target="_blank" style="color: #007acc; text-decoration: none;">
       this one
   </a> by 
   <a href="https://www.kaggle.com/konstantinboyko" target="_blank" style="color: #007acc; text-decoration: none;">
       @konstantinboyko.
   </a>
    Thanks to all of you.
</p>

<p style="font-size: 1.1em;">
   Special thanks to Ali 
   <a href="https://www.kaggle.com/asalhi" target="_blank" style="color: #007acc; text-decoration: none;">
       @asalhi
   </a> 
   for sharing insights about QwQ-32B-preview scores and uploading the AWQ version.
</p>

Thanks to [Simon Frieder](https://www.kaggle.com/friederrr) for his detailed suggestions to improve the documentation of the notebook.

People involved in creating this solution : Md Boktiar Mahbub Murad (Just me :) )

<h1 style="text-align: center; font-size: 1.8em; color: #2c3e50;">Optimizing the Public Notebook</h1>

<p style="font-size: 1.1em; color: #444;">
When I looked at the public notebook scoring <strong>16</strong>, I realized it could be improved by carefully optimizing the inference. Here's the changes I made to the public notebook and why  : </p>

<h2 style="color: #34495e;">1. <strong>dtype and max_num_steps</strong></h2>
<p style="color: #444;">
Before the QwQ phase, I had a score of <strong>8</strong> with Qwen-2.5-Math-7B. I noticed that setting <code>dtype = bfloat16</code> and increasing <code>max_num_seqs</code> improved the score and made it more stable.  
</p>
<blockquote style="background: #f4f4f9; padding: 10px; border-left: 4px solid #a3a3a3; margin: 10px 0;">
<strong>Settings:</strong>  
<code>dtype = bfloat16</code>, <code>max_num_seqs = 256 (default)</code>
</blockquote>

<h2 style="color: #34495e;">2. <strong>Change in Prompts</strong></h2>
<p style="color: #444;">
Since QwQ leverages Chain-of-Thought (CoT) better than code generation, I decided to retain only the CoT prompts and remove those mentioning Python code generation.  
Here are the 5 prompts I used for each sample:
</p>
<blockquote style="background: #f4f4f9; padding: 10px; border-left: 4px solid #a3a3a3; margin: 10px 0;">
<ul>
  <li>Please use chained reasoning to put the answer in <code>\\boxed{}</code>.</li>
  <li>Please reflect and verify while reasoning and put the answer in <code>\\boxed{}</code>.</li>
  <li>Solve the following problem using concise and clear reasoning by placing the answer in <code>\\boxed{}</code>.</li>
  <li>You are a helpful and reflective maths assistant, please reason step by step to put the answer in <code>\\boxed{}</code>.</li>
  <li>You are the smartest maths expert in the world, please spike this question and put the answer in <code>\\boxed{}</code>.</li>
</ul>
</blockquote>

<h2 style="color: #34495e;">3. <strong>Number of Samples: 8 ➔ 5</strong></h2>
<p style="color: #444;">
From my local runs and discussions, I observed that QwQ mostly provided the correct answer with fewer samples (4–5). Using 8 samples was inefficient, as it increased runtime and risked exceeding the 5-hour limit before solving all 50 problems (solving 40-45 in total). Therefore, I reduced the number of samples to 5.
</p>

<h2 style="color: #34495e;">4. <strong>Seed</strong></h2>
<p style="color: #444;">
I removed the seed parameter from <code>vllm</code> and instead set a global seed using <code>set_seed(42)</code> from <code>transformers</code>.
</p>

<h2 style="color: #34495e;">5. <strong>Got Lucky 😂</strong></h2>
<p style="color: #444;">
Sometimes, luck plays its part!
</p>

# FAQs

- Q : Why does **QwQ-32B-preview** model performs so good here?

  A : QwQ-32B-preview works quite similar to the **OpenAI-o1** preview model. **It reflects and challenges it's own assumptions and reasoning, and then verifies it iteratively**. It starts from scratch, as if it knows nothing, generates an initial solution, then verifies and justifies each step. From their blog post
  > Like a seeker of wisdom on an endless journey of discovery, the model demonstrates its capacity for deep introspection - questioning its own assumptions, engaging in thoughtful self-dialogue, and carefully examining each step of its reasoning process

- Q : Does this notebook always score 20 after submitting?
  
  A : No, there are several randomness associated in this notebook. Temperature sampling, generating with vllm are some of them. These could lead to a change of the score. For now it has quite high variance. This could be a future aspect of the competition to try to stabilize the notebook's score

# Imports

```python
#fixed seed to get similar score
from transformers import set_seed
set_seed(42)
```

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
cutoff_time = time.time() + (4 * 60 + 45) * 60
```

# Load Model

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

llm_model_pth = '/kaggle/input/m/shelterw/qwen2.5/transformers/qwq-32b-preview-awq/1'

llm = LLM(
    llm_model_pth,
    #dtype="half",                -> Changed this
    #max_num_seqs=128,            -> Changed this
    max_model_len=32768,#4096*10,         
    trust_remote_code=True,     
    tensor_parallel_size=4,      
    gpu_memory_utilization=0.96, 
)
```

```python
tokenizer = llm.get_tokenizer()
```

# Utlities

```python
import re

#prompts
thoughts = [
   
    'Please use chained reasoning to put the answer in \\boxed{}.',
    'Please reflect and verify while reasoning and put the answer in \\boxed{}.',
    'Solve the following problem using concise and clear reasoning by placing the answer in \\boxed{}.',
    'You are a helpful and reflective maths assistant, please reason step by step to put the answer in \\boxed{}.',
    'You are the smartest maths expert in the world, please spike this question and put the answer in \\boxed{}.'
]

#create single prompt
def make_next_prompt(text,round_idx):
    default_prompt = thoughts[(round_idx+1)%len(thoughts)]
    default_python_code = f"print('{default_prompt}')"
    return default_python_code

#extract python code from response
def extract_python_code(text):
    pattern = r'```python\s*(.*?)\s*```'
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        ans = "\n\n".join(matches)
        #print(f'Extracted python code: {ans}')
        return ans
    return ""

#extract all code segments
def extract_python_code_list(text):
    pattern = r'```python\s*(.*?)\s*```'
    ans=[]
    matches = re.findall(pattern, text, re.DOTALL)
    for m in matches:
        ans.append(m)
    return ans

#process the code
def process_python_code(query):
    query = "import math\nimport numpy as np\nimport sympy as sp\n" + query
    current_rows = query.strip().split("\n")
    new_rows = []
    for row in current_rows:
        new_rows.append(row)
    ans = "\n".join(new_rows)
    print(f'Processed python code: {ans}')
    return ans

import re

#extract the answer from the boxes
def extract_boxed_texts(text):
    pattern = r'oxed{(.*?)}'
    matches = re.findall(pattern, text)
    if not matches:
        return []
    ans = []
    for content in matches:
        if content.isdigit():
            num = int(content)
        else:
            nums = re.findall(r'\d+', content)
            if not nums:
                continue
            num = int(nums[-1])
        ans.append(num % 1000)
    return ans
    
#extract the integer answer modulo 1000 from the boxes
def extract_boxed_text(text):
    pattern = r'oxed{(.*?)}'
    matches = re.findall(pattern, text)
    if not matches:
        return -1
    content = matches[0]
    if content.isdigit():
        num = int(content)
    else:
        nums = re.findall(r'\d+', content)
        if not nums:
            return -1
        num = int(nums[-1])
    return num % 1000

#select the final answer based on the frequency (majoity voting)
from collections import Counter
def select_answer(answers):
    valid_answers = []
    for answer in answers:
        try:
            if int(answer) == float(answer):
                if 1 < int(answer) < 999 and int(answer) % 100 > 0:
                    valid_answers.append(int(answer))
        except:
            pass
    if not valid_answers:
        return 49
    _, answer = sorted([(v,k) for k,v in Counter(valid_answers).items()], reverse=True)[0]
    return answer%1000
```

```python
import os
import tempfile
import subprocess

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

```python
#sanity check
list_of_texts = [
    tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True
    )
    for messages in [[{"role": "user", "content": "hi"}]]
]
```

# Text Generation Functions

```python
#define the sampling parameters
sampling_params = SamplingParams(
    temperature=1.0,              # Controls randomness in generation: higher values (e.g., 1.0) produce more diverse output.
    min_p=0.01,                   # Minimum cumulative probability for nucleus sampling, filtering out unlikely tokens.
    skip_special_tokens=True,     
    # max_tokens=1800,            
    max_tokens=32768,             # Sets a very high limit for token generation to handle longer outputs.
    # stop=["```output"],       
)

#generate prompts in batch
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
        messages.append({'role': 'assistant', 'content': single_request_output.outputs[0].text})
        print(messages[-1])

    return list_of_messages
```

```python
#filter answers from the responses

def batch_message_filter(list_of_messages,list_of_idx) -> tuple[list[list[dict]], list[str]]:
    global answer_contributions
    extracted_answers = []
    list_of_messages_to_keep = []
    list_of_idx_to_keep = []
    for idx,messages in zip(list_of_idx,list_of_messages):
        answers = extract_boxed_texts(messages[-1]['content'])
        if answers:
            extracted_answers.extend(answers)
            for answer in answers:
                answer_contributions[answer].append(idx)
        else:
            list_of_messages_to_keep.append(messages)
            list_of_idx_to_keep.append(idx)
    return list_of_messages_to_keep, extracted_answers, list_of_idx_to_keep
```

```python
#execute the codes in the responses
def batch_message_execute(list_of_messages,round_idx) -> list[list[dict]]:
    for messages in list_of_messages:
        python_code = extract_python_code(messages[-1]['content'],round_idx)
        python_code = process_python_code(python_code)
        try:
            success, output = PythonREPL()(python_code)
        except Exception as e:
            output = str(e)
        messages.append({'role': 'user', 'content': output})
        print(messages[-1])
    return list_of_messages

#execute the code and generate the answer from responses
def batch_message_execute_and_get_answer(list_of_messages,round_idx) -> tuple[list[list[dict]], list[int]]:
    ans = []
    for messages in list_of_messages:
        python_code = extract_python_code(messages[-1]['content'])
        python_code = process_python_code(python_code)
        try:
            success, output = PythonREPL()(python_code)
            if success:
                patten = r'(\d+)'
                matches = re.findall(patten, output)
                if matches:
                    for match in matches:
                        ans.append(int(match)%1000)
                        ans.append(int(match)%1000) #代码权重高于自然语言，所以添加两次 
        except Exception as e:
            output = str(e)
        print(f'python code output: {output}')
    return ans

#execute code and generate answer for all elements in batch
def batch_message_list_execute_and_get_answer(list_of_messages,round_idx) -> tuple[list[list[dict]], list[int]]:
    ans = []
    for messages in list_of_messages:
        python_code_list = extract_python_code_list(messages[-1]['content'])
        for python_code in python_code_list:
            python_code = process_python_code(python_code)
            try:
                success, output = PythonREPL()(python_code)
                if success:
                    patten = r'(\d+)'
                    matches = re.findall(patten, output)
                    if matches:
                        for match in matches:
                            ans.append(int(match)%1000)
                            ans.append(int(match)%1000) 
            except Exception as e:
                output = str(e)
            print(f'python code output: {output}')
    return ans
```

```python
import os

import pandas as pd
import polars as pl

#API for competition submission
import kaggle_evaluation.aimo_2_inference_server
```

```python
#correct answers for reference problems
def get_correct_answer(question):
    if 'Three airline' in question: return 79
    if 'Fred and George' in question: return 250
    if 'Triangle $ABC$' in question: return 180
    if 'Find the three' in question: return 143
    if 'We call a' in question: return 3
    if 'Let $ABC$ be' in question: return 751
    if 'For a positive' in question: return 891
    if 'For positive integers' in question: return 810
    if 'The Fibonacci numbers' in question: return 201
    if 'Alice writes all' in question: return 902
    return 0
```

# Inference

```python
#predict function to solve single problem

from collections import Counter, defaultdict
g_score = 0
g_count = 0
prompt_score = Counter()
answer_contributions = defaultdict(list)
def predict_for_question(question: str) -> int:
    global g_score
    global g_count
    global prompt_score
    global answer_contributions
    question += "\nIf the final answer is a number larger than 1000, take modulo 1000. "
    if time.time() > cutoff_time: 
        return 210
    print(question)
    
    list_of_messages = [
        [
            {"role": "system", "content": thoughts[k]},
            {"role": "user", "content": question}
        ] for k in range(5)
    ]

    all_extracted_answers = []
    list_of_idx = list(range(len(list_of_messages)))
    max_round = 1
    for round_idx in range(max_round):
        print(f"round {round_idx+1}")
        list_of_messages = batch_message_generate(list_of_messages)
        #extracted_python_answer = batch_message_execute_and_get_answer(list_of_messages,round_idx)
        extracted_python_answer = batch_message_list_execute_and_get_answer(list_of_messages,round_idx)
        list_of_messages, extracted_answers, list_of_idx  = batch_message_filter(list_of_messages, list_of_idx)
        all_extracted_answers.extend(extracted_python_answer)
        all_extracted_answers.extend(extracted_answers)
        print("extracted boxed answers:",extracted_answers)
        print("extracted python answers:",extracted_python_answer)
        print("all extracted answers:",all_extracted_answers)
        if not list_of_messages:
            break
        #list_of_messages = batch_message_execute(list_of_messages,round_idx)
    answer = select_answer(all_extracted_answers)
    print("answer:",answer)
    correct_answer = get_correct_answer(question)
    print("correct answer:",correct_answer)
    g_count += 1
    if str(answer) == str(correct_answer):
        g_score += 1

    print(f"score: {g_score}/{g_count}")
    print("\n\n")
    return answer
```

**Predict Function provided by the hosts**

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

**Load and save the reference set to use for validation**

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
            'reference.csv',
        )
    )
```

# Guideline Answers

Answer to the questions from [Kaggle Winning Model Documentation Guidelines](https://www.kaggle.com/WinningModelDocumentationGuidelines).
Only answers to applicable questions are listed below

* **Did you have any prior experience that helped you succeed in this competition**?
  
    - I participated in the AI Math Olympiad progress prize 1. In that phase I was in the top 25 in the public LB solving 24/50 problems.Unfortunately my solution didn't perform well on the private LB and scored 15. Nevertheless, I learned a lot from that competition and this experience helped me understand this iteration much better.
 
* **What made you decide to enter this competition?**
  - In general I like to solve mathematical problems and I participated in Math Olympiads in my country during my high school days. So I liked the idea of the competition and it was a good opportunity for me to learn LLMs in depth.
 
* **How much time did you spend on the competition?**
  - I spent quite a lot of time in this competition initially. After that I got quite busy with other works, still I tried to spend around a hour everyday on the competition and submit a solution.
 
* If part of a team, how did you decide to team up?
  - N/A
* If you competed as part of a team, who did what?
  - N/A

# Reference

**References of previous notebooks**

- https://www.kaggle.com/code/richolson/ai-math-olympiad-qwen2-5-72b - shows how to submit using 72B models
- https://www.kaggle.com/code/abdullahmeda/load-72b-awq-model-using-vllm-on-l4-x4 - shows how to use 72B models with vllm on L4x4 GPU's
- https://www.kaggle.com/code/huikang/qwen2-5-math-1-5b-instruct - created the initial baseline, the skeleton of this notebook is inspired by this one
- https://www.kaggle.com/code/boristown/qwen-qwq-32b-preview-deepreasoning - showed that QwQ preview is able to get high score

**QwQ-32B-preview**
- [Huggingface release](Qwen/QwQ-32B-Preview) - Original Huggingface Release of QwQ model by Qwen
- [AWQ Version](https://huggingface.co/KirillR/QwQ-32B-Preview-AWQ) - QwQ AWQ Quantized Version on huggingface
- [Blog post](https://qwenlm.github.io/blog/qwq-32b-preview/) - Detailed blog post about QwQ model

# License

Model : QwQ-32B-preview 

Copyright 2024 Alibaba Cloud. Apache 2.0 license [here](https://huggingface.co/Qwen/QwQ-32B-Preview/blob/main/LICENSE)

------------------------------------------------------------------------------------------
AWQ Quantized Model : QwQ-32B-preview-AWQ

Apache 2.0 License [here](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md)

------------------------------------------------------------------------------------------
Notebook : 

This Notebook is based on the Apache 2.0 open source license based on [this notebook](https://www.kaggle.com/code/boristown/qwen-qwq-32b-preview-deepreasoning). Changes made to the original notebook are those that I outlined in the present notebook.

# Citation
```

@misc{Murad2024earlysharingprize,
  author       = "Md Boktiar Mahbub Murad", 
  title        = "QWQ-32B-preview Optimized inference Early Sharing Prize winner",
  howpublished = "\url{https://www.kaggle.com/code/mbmmurad/lb-20-qwq-32b-preview-optimized-inference}",
  month        = "Dec", 
  year         = "2024", 
  note         = "More ain't always better", 
}

```