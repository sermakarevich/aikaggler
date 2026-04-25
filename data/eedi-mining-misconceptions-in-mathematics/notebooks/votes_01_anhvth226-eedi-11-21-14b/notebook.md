# EEDI_11_21_14B

- **Author:** Anh Vo
- **Votes:** 686
- **Ref:** anhvth226/eedi-11-21-14b
- **URL:** https://www.kaggle.com/code/anhvth226/eedi-11-21-14b
- **Last run:** 2024-11-26 06:40:16.837000

---

```python
# !md5sum /kaggle/input/2211-lora-14b/transformers/default/1/adapter_model.safetensors
# should be 25596de367acaec20e616b7c87bd5529
```

```python
!pip install /kaggle/input/eedi-library/autoawq*.whl /kaggle/input/eedi-library/peft-0.13.2-py3-none-any.whl  --no-index --find-links=/kaggle/input/eedi-library
```

```python
import os, math, numpy as np
import sys
import os
from transformers import AutoTokenizer
import pandas as pd
from tqdm import tqdm
import re, gc
import torch
pd.set_option('display.max_rows', 300)
```

```python
IS_SUBMISSION = True
#bool(os.getenv("KAGGLE_IS_COMPETITION_RERUN"))


print('IS_SUBMISSION:', IS_SUBMISSION)

model_path = "/kaggle/input/qwen2.5/transformers/32b-instruct-awq/1"
df_train = pd.read_csv("/kaggle/input/eedi-mining-misconceptions-in-mathematics/train.csv").fillna(-1).sample(10, random_state=42).reset_index(drop=True)
df_test = pd.read_csv("/kaggle/input/eedi-mining-misconceptions-in-mathematics/test.csv")
df_misconception_mapping = pd.read_csv("/kaggle/input/eedi-mining-misconceptions-in-mathematics/misconception_mapping.csv")
# tokenizer = AutoTokenizer.from_pretrained(model_path)
```

# first retrieval

```python
import pandas as pd
# from sentence_transformers import SentenceTransformer, util
if not IS_SUBMISSION:
    df_ret = df_train.copy()
else:
    df_ret = df_test.copy()
```

```python
df_ret
```

```python
TEMPLATE_INPUT_V3 = '{QUESTION}\nCorrect answer: {CORRECT_ANSWER}\nStudent wrong answer: {STUDENT_WRONG_ANSWER}'
def format_input_v3(row, wrong_choice):

    assert wrong_choice in "ABCD"
    # Extract values from the row
    question_text = row.get("QuestionText", "No question text provided")
    subject_name = row.get("SubjectName", "Unknown subject")
    construct_name = row.get("ConstructName", "Unknown construct")
    # Extract the correct and wrong answer text based on the choice
    correct_answer = row.get("CorrectAnswer", "Unknown")
    assert wrong_choice != correct_answer
    correct_answer_text = row.get(f"Answer{correct_answer}Text", "No correct answer text available")
    wrong_answer_text = row.get(f"Answer{wrong_choice}Text", "No wrong answer text available")

    # Construct the question format
    formatted_question = f"""Question: {question_text}
    
SubjectName: {subject_name}
ConstructName: {construct_name}"""

    # Return the extracted data
    ret = {
        "QUESTION": formatted_question,
        "CORRECT_ANSWER": correct_answer_text,
        "STUDENT_WRONG_ANSWER": wrong_answer_text,
        "MISCONCEPTION_ID": row.get('Misconception{wrong_choice}Id'),
    }
    ret["PROMPT"] = TEMPLATE_INPUT_V3.format(**ret)

    return ret


items = []
target_ids = []
for _, row in df_ret.iterrows():
    for choice in ['A', 'B', 'C', 'D']:
        if choice == row["CorrectAnswer"]:
            continue
        if not IS_SUBMISSION and row[f'Misconception{choice}Id'] == -1:
            continue
            
        correct_col = f"Answer{row['CorrectAnswer']}Text"
        item = {'QuestionId_Answer': '{}_{}'.format(row['QuestionId'], choice)}
        item['Prompt'] = format_input_v3(row, choice)['PROMPT']
        items.append(item)
        target_ids.append(int(row.get(f'Misconception{choice}Id', -1)))
        
df_input = pd.DataFrame(items)
```

```python
def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'<instruct>{task_description}\n<query>{query}'

def get_detailed_example(task_description: str, query: str, response: str) -> str:
    return f'<instruct>{task_description}\n<query>{query}\n<response>{response}'

def get_new_queries(queries, query_max_len, examples_prefix, tokenizer):
    inputs = tokenizer(
        queries,
        max_length=query_max_len - len(tokenizer('<s>', add_special_tokens=False)['input_ids']) - len(
            tokenizer('\n<response></s>', add_special_tokens=False)['input_ids']),
        return_token_type_ids=False,
        truncation=True,
        return_tensors=None,
        add_special_tokens=False
    )
    prefix_ids = tokenizer(examples_prefix, add_special_tokens=False)['input_ids']
    suffix_ids = tokenizer('\n<response>', add_special_tokens=False)['input_ids']
    new_max_length = (len(prefix_ids) + len(suffix_ids) + query_max_len + 8) // 8 * 8 + 8
    new_queries = tokenizer.batch_decode(inputs['input_ids'])
    for i in range(len(new_queries)):
        new_queries[i] = examples_prefix + new_queries[i] + '\n<response>'
    return new_max_length, new_queries
task =  "Given a math multiple-choice problem with a student's wrong answer, retrieve the math misconceptions"
queries = [
    get_detailed_instruct(task, q) for q in df_input['Prompt']
]
documents = df_misconception_mapping['MisconceptionName'].tolist()
query_max_len, doc_max_len = 320, 48
LORA_PATH = '/kaggle/input/2611-lora-14b-more-synthetic/transformers/default/1'
tokenizer = AutoTokenizer.from_pretrained(LORA_PATH)
examples_prefix = ''
new_query_max_len, new_queries = get_new_queries(queries, query_max_len, examples_prefix, tokenizer)


import json
with open('data.json', 'w') as f:
    data = {'texts': new_queries+ documents}
    f.write(json.dumps(data))
```

```python
%%writefile run_embed.py
import argparse
import os
import json
import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import peft

MAX_LENGTH = 320


def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[
            torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths
        ]


def get_embeddings_in_batches(model, tokenizer, texts, max_length, batch_size=32):
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
        batch_texts = texts[i : i + batch_size]
        batch_dict = tokenizer(
            batch_texts,
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to("cuda")
        with torch.no_grad(), torch.amp.autocast("cuda"):
            outputs = model(**batch_dict)
            batch_embeddings = last_token_pool(
                outputs.last_hidden_state, batch_dict["attention_mask"]
            )
            batch_embeddings = F.normalize(batch_embeddings, p=2, dim=1).cpu()
        embeddings.append(batch_embeddings)
    return torch.cat(embeddings, dim=0)


def load_model_and_tokenizer(base_model_path, lora_path, load_in_4bit=True):
    model = AutoModel.from_pretrained(
        base_model_path,
        device_map=0,
        torch_dtype=torch.float16,
        load_in_4bit=load_in_4bit,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        lora_path if lora_path else base_model_path
    )
    model.resize_token_embeddings(len(tokenizer))
    if lora_path:
        model = peft.PeftModel.from_pretrained(model, lora_path)
    return model, tokenizer


def main(args):
    output_file = args.input_text.replace(
        ".json", ".pt.fold.{}.{}.embed".format(*args.fold)
    )
    if os.path.exists(output_file):
        print(f"Output file {output_file} already exists. Skipping...")
        return
    model, tokenizer = load_model_and_tokenizer(
        args.base_model, args.lora_path, load_in_4bit=args.load_in_4bit
    )
    texts = json.load(open(args.input_text))["texts"][args.fold[0] :: args.fold[1]]
    embeddings = get_embeddings_in_batches(
        model,
        tokenizer,
        texts,
        max_length=MAX_LENGTH,
        batch_size=4,
    )
    text2embeds = {text: emb for text, emb in zip(texts, embeddings)}
    torch.save(text2embeds, output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_model",
        type=str,
        default="Qwen/Qwen2.5-7B",
        help="Path to the base model",
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        default=None,
        help="Path to the LoRA model",
    )
    parser.add_argument(
        "--input_text",
        type=str,
        default=".cache/data.json",
    )
    parser.add_argument(
        "--load_in_4bit",
        action="store_true",
        help="Load model in 4-bit mode",
    )
    parser.add_argument("--fold", nargs=2, type=int, default=[0, 1])
    args = parser.parse_args()
    if not os.path.exists(args.lora_path):
        args.lora_path = None
    main(args)
```

```python
# %%writefile run.sh
lora_path = '/kaggle/input/2611-lora-14b-more-synthetic/transformers/default/1'
cmd = f"(CUDA_VISIBLE_DEVICES=0 python run_embed.py --base_model /kaggle/input/qw14b-awq/transformers/default/1 --lora_path {lora_path} --input_text data.json --fold 0 2) & (CUDA_VISIBLE_DEVICES=1 python run_embed.py --base_model /kaggle/input/qw14b-awq/transformers/default/1 --lora_path {lora_path} --input_text data.json --fold 1 2)"
import os
os.system(cmd)
```

```python
from glob import glob
import time
text_to_embed = {}
files = glob('*.pt*')
while len(files) != 2:
    time.sleep(1)
    files = glob('*.pt*')


time.sleep(3)    
for path in files:
    print(path)
    text_to_embed.update(torch.load(path))
```

```python
len(text_to_embed)
```

```python
# [text_to_embed[t] for t in new_queries][0.shape]
```

```python
query_embeddings = torch.stack([text_to_embed[t] for t in new_queries])
doc_embeddings = torch.stack([text_to_embed[t] for t in documents])
query_embeddings.shape, doc_embeddings.shape
```

### Eval (testing purpose only)
This will not be ran in real submition

```python
import torch
from typing import List

def compute_metrics(q_embeds: torch.Tensor, d_embeds: torch.Tensor, target_ids: List[int]):
    """
    Compute MAP@25 and Recall@100 metrics.
    
    Args:
        q_embeds (torch.Tensor): Query embeddings of shape (M, dim), where M is the number of queries.
        d_embeds (torch.Tensor): Document embeddings of shape (N, dim), where N is the number of documents.
        target_ids (List[int]): List of target document indices (length M, one target index per query).
        
    Returns:
        None: Prints MAP@25 and Recall@100.
    """
    # Compute similarity scores
    scores = q_embeds @ d_embeds.T  # Shape: (M, N)

    # Initialize variables for metrics
    avg_precisions = []  # To store average precision for each query
    recall_counts = []   # To store recall@100 counts for each query

    # Compute metrics for each query
    for i, target_id in enumerate(target_ids):
        # Sort document indices by score in descending order
        sorted_indices = torch.argsort(scores[i], descending=True)

        # Compute precision@k and recall@100
        relevant_docs = (sorted_indices[:100] == target_id).nonzero(as_tuple=True)[0]  # Find rank within top 100
        recall_count = 1 if len(relevant_docs) > 0 else 0  # Check if target is in the top 100
        recall_counts.append(recall_count)

        # Compute average precision for top 25 (MAP@25)
        precision_at_k = 0.0
        num_relevant = 0
        for rank, idx in enumerate(sorted_indices[:25]):
            if idx == target_id:
                num_relevant += 1
                precision_at_k += num_relevant / (rank + 1)
        avg_precisions.append(precision_at_k / 1 if num_relevant > 0 else 0)

    # Calculate metrics
    map25 = sum(avg_precisions) / len(avg_precisions)
    recall100 = sum(recall_counts) / len(recall_counts)

    # Print results
    print(f"MAP@25: {map25:.4f}")
    print(f"Recall@100: {recall100:.4f}")
if not IS_SUBMISSION:
    compute_metrics(query_embeddings, doc_embeddings, target_ids)
```

```python
scores = query_embeddings @ doc_embeddings.T  # Shape: (M, N)
sorted_indices = torch.argsort(scores,1, descending=True)[:,:25].tolist()
```

```python
df_input["MisconceptionId"] = [" ".join([str(x) for x in row]) for row in sorted_indices]
df_input[["QuestionId_Answer", "MisconceptionId"]].to_csv("submission.csv", index=False)

pd.read_csv('submission.csv')
```

```python
# %%time

# !pip uninstall -y torch

# !pip install -q --no-index --find-links=/kaggle/input/making-wheels-of-necessary-packages-for-vllm vllm

# !pip install -q -U --upgrade /kaggle/input/vllm-t4-fix/grpcio-1.62.2-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl

# !pip install -q -U --upgrade /kaggle/input/vllm-t4-fix/ray-2.11.0-cp310-cp310-manylinux2014_x86_64.whl

# !pip install -q --no-deps --no-index /kaggle/input/hf-libraries/sentence-transformers/sentence_transformers-3.1.0-py3-none-any.whl

# !pip install --no-index /kaggle/input/pip-peft/peft-0.13.2-py3-none-any.whl

# !pip install /kaggle/input/eedi-library/bitsandbytes-0.44.1-py3-none-manylinux_2_24_x86_64.whl
```

```python
# ### Apply QW32B
# df_misconception_mapping.iloc[2532]['MisconceptionName']
# #### BUILD INPUT PARQUET

# df_input

# from typing import List


# misconceptions = df_misconception_mapping.MisconceptionName.values
# def format_docs(MisconceptionId) -> str:
#     doc_ids = [int(_) for _ in MisconceptionId.split(' ')]
#     docs = [misconceptions[i] for i in doc_ids]
#     docs = [f"[{i}] " + doc for i, doc in enumerate(docs)]
#     return "\n".join(docs)
    
# df_input['Retrival'] = df_input['MisconceptionId'].apply(format_docs)
# print(df_input['Retrival'].iloc[0])


# user_template = """Instruction:
# You will analyze a student's incorrect mathematical work and identify which misconception(s) from a provided list best explain their error.

# **First, ensure you fully understand the problem by rewriting it in your own words to eliminate any potential ambiguities. Place your rewritten problem inside <clarified_problem> tags.**

# **Next, verify the correctness of the student's answer by working through the problem step-by-step yourself, using the standard mathematical procedures and order of operations. Do not rely on the provided correct answer during this step. Place your solution inside <verification> tags.**

# **As you perform your verification, pay close attention to the order in which operations should be performed according to BIDMAS/BODMAS/PEDMAS. Consider how changing the order affects the result, and note the impact of brackets on the calculation.**

# After verifying, analyze the student's work carefully by considering:

# 1. What specific error(s) did the student make?
# 2. What pattern of thinking likely led to these errors?
# 3. Which misconception(s) from the list best explain this thinking pattern?
# 4. Are there multiple misconceptions that might apply?
# 5. What evidence in the student's work supports each potential misconception?

# **Be thorough in matching the student's error to the misconceptions provided. Consider whether the student is misunderstanding the priorities of operations, such as believing addition comes before multiplication.**

# Write your analysis inside <thinking> tags. Keep your analysis under 500 words and focus on:

# - The specific mathematical steps where errors occurred
# - The likely reasoning behind these errors
# - How specific misconceptions from the list explain this reasoning
# - Why certain misconceptions fit better than others
# - Whether multiple misconceptions might be at play

# After your analysis, provide your conclusion inside <answer> tags by listing the index number(s) of the misconception(s) that best explain the student's error. If multiple misconceptions apply, list them in order of relevance, separated by commas.

# **Ensure that your selected misconceptions directly correspond to the student's demonstrated errors and misconceptions, based on the evidence you've analyzed.**

# Begin your analysis using the following inputs:



# <misconception_list>
# {Retrival}
# </misconception_list>

# <problem_statement>
# {embed_prompt}
# </problem_statement>
# """
# assistant_prefix = '<clarified_problem> \n'
# def apply_template(row, tokenizer):
#     user = user_template.format(embed_prompt=row["Prompt"], Retrival=row["Retrival"])
#     messages = [
#         {
#             "role": "user",
#             "content": user,
#         },
#     ]
#     text = tokenizer.apply_chat_template(
#         messages, tokenize=False, add_generation_prompt=True
#     )
#     return text+assistant_prefix


# df_input["text"] = df_input.apply(lambda row: apply_template(row, tokenizer), 1)
# df_input.to_parquet("input_llm.parquet", index=False)
# print("example input:\n\n")
# print(df_input.iloc[0]['text'])

# example_msgs =  [{'role': 'user',
#    'content': "\nInstruction:\nYou will analyze a student's incorrect mathematical work and identify which misconception(s) from a provided list best explain their error.\n\n**First, ensure you fully understand the problem by rewriting it in your own words to eliminate any potential ambiguities. Place your rewritten problem inside <clarified_problem> tags.**\n\n**Next, verify the correctness of the student's answer by working through the problem step-by-step yourself, using the standard mathematical procedures and order of operations. Do not rely on the provided correct answer during this step. Place your solution inside <verification> tags.**\n\n**As you perform your verification, pay close attention to the order in which operations should be performed according to BIDMAS/BODMAS/PEDMAS. Consider how changing the order affects the result, and note the impact of brackets on the calculation.**\n\nAfter verifying, analyze the student's work carefully by considering:\n\n1. What specific error(s) did the student make?\n2. What pattern of thinking likely led to these errors?\n3. Which misconception(s) from the list best explain this thinking pattern?\n4. Are there multiple misconceptions that might apply?\n5. What evidence in the student's work supports each potential misconception?\n\n**Be thorough in matching the student's error to the misconceptions provided. Consider whether the student is misunderstanding the priorities of operations, such as believing addition comes before multiplication.**\n\nWrite your analysis inside <thinking> tags. Keep your analysis under 500 words and focus on:\n\n- The specific mathematical steps where errors occurred\n- The likely reasoning behind these errors\n- How specific misconceptions from the list explain this reasoning\n- Why certain misconceptions fit better than others\n- Whether multiple misconceptions might be at play\n\nAfter your analysis, provide your conclusion inside <answer> tags by listing the index number(s) of the misconception(s) that best explain the student's error. If multiple misconceptions apply, list them in order of relevance, separated by commas.\n\n**Ensure that your selected misconceptions directly correspond to the student's demonstrated errors and misconceptions, based on the evidence you've analyzed.**\n\nBegin your analysis using the following inputs:\n\n\n\n<misconception_list>\n[0] Answers order of operations questions with brackets as if the brackets are not there\n[1] Confuses the order of operations, believes subtraction comes before multiplication \n[2] Believes order of operations does not affect the answer to a calculation\n[3] Applies BIDMAS in strict order (does not realize addition and subtraction, and multiplication and division, are of equal priority)\n[4] Confuses the order of operations, believes addition comes before multiplication \n[5] Has removed brackets but not performed the operation\n[6] Carries out operations from left to right regardless of priority order\n[7] Has not realised that the answer may be changed by the insertion of brackets\n[8] Carries out operations from right to left regardless of priority order\n[9] May have made a calculation error using the order of operations\n[10] Misunderstands order of operations in algebraic expressions\n[11] Inserts brackets but not changed order of operation\n[12] Carries out operations from left to right regardless of priority order, unless brackets are used\n[13] Has completed only one of the two operations.\n[14] Does not realize we can omit the multiplication symbol in algebraic expressions\n[15] Performs addition ahead of multiplication\n[16] Does not include brackets when attempting to multiply an expression\n[17] Does not correctly apply the distributive property of multiplication\n[18] Thinks multiplication and addition are the same\n[19] Does not include brackets when required\n[20] Performs subtraction right to left if priority order means doing a calculation to the right first\n[21] Performs addition ahead of any other operation\n[22] Does not interpret the correct order of operations from a worded problem\n[23] Multiplies before applying a power \n[24] Does not correctly apply the commutative property of multiplication\n</misconception_list>\n\n<problem_statement>\nQuestion: \\[\n3 \\times 2+4-5\n\\]\nWhere do the brackets need to go to make the answer equal \\( 13 \\) ?\n\nSubjectName: BIDMAS\nConstructName: Use the order of operations to carry out calculations involving powers\nCorrect text: \\( 3 \\times(2+4)-5 \\)\nStudent wrong answer : Does not need brackets\n</problem_statement>\n"},
#   {'role': 'assistant',

#    'content': '<clarified_problem> \nQuestion: Solve the expression \\( 3 \\times 2 + 4 - 5 \\) after placing brackets to make the result equal to 13.\n</clarified_problem>\n\n<verification>\nTo verify the solution, I will follow the standard order of operations (BIDMAS/BODMAS/PEDMAS):\n\n- Without any brackets, the expression is evaluated as:\n  \\[\n  3 \\times 2 + 4 - 5 = 6 + 4 - 5 = 10 - 5 = 5\n  \\]\n  This does not yield 13.\n\n- With the brackets placed as \\( 3 \\times (2+4) - 5 \\):\n  \\[\n  3 \\times (2+4) - 5 = 3 \\times 6 - 5 = 18 - 5 = 13\n  \\]\n  This correctly yields 13.\n\nTherefore, the correct placement of brackets is \\( 3 \\times (2+4) - 5 \\).\n</verification>\n\n<thinking>\nThe student provided the answer "Does not need brackets," which incorrectly suggests they believe the expression as written is already equal to 13 without the need for reordering operations with brackets.\n\n1. **Specific Error**: The student\'s error lies in their belief that the expression \\( 3 \\times 2 + 4 - 5 \\) does not need brackets to achieve the result of 13. This implies they are not recognizing the necessity of changing the order of operations using brackets.\n\n2. **Pattern of Thinking**: The student likely believes that addition comes before multiplication, leading them to incorrectly conclude that no brackets are necessary. They may also be misunderstanding the importance of the order of operations, thinking it can be ignored in this context.\n\n3. **Misconceptions from the List**:\n   - **Misconception 4**: Confuses the order of operations, believing addition comes before multiplication. This fits as the student is performing the addition before multiplication.\n   - **Misconception 6**: Carries out operations from left to right regardless of priority order. This also aligns with the student’s approach of evaluating the expression without proper order of operations.\n\n4. **Evidence in Student\'s Work**: The student\'s conclusion that no brackets are needed suggests a misunderstanding of the order of operations and the necessity of using brackets to achieve the desired result. Their belief that addition should come before multiplication is evident from the incorrect evaluation of the expression without brackets.\n\n5. **Relevance of Misconceptions**: Misconception 4 directly addresses the student\'s belief that addition should come before multiplication. Misconception 6 further reinforces the idea that the student is not recognizing the priority of operations and is instead evaluating the expression from left to right.\n\nGiven this analysis, Misconception 4 and Misconception 6 best explain the student\'s error.\n</thinking>\n\n<answer>\n4, 6\n</answer>'}]

# %%writefile run_vllm.py

# import re

# import vllm

# import pandas as pd



# df = pd.read_parquet("input_llm.parquet")



# model_path = '/kaggle/input/qw32-awq-2011/transformers/default/1'

# llm = vllm.LLM(
#     model_path,
#     quantization="awq",
#     tensor_parallel_size=2,
#     gpu_memory_utilization=0.95, 
#     trust_remote_code=True,
#     dtype="half", 
#     enforce_eager=True,
#     max_model_len=5000,
#     disable_log_stats=False,
#     enable_prefix_caching=True
# )
# tokenizer = llm.get_tokenizer()


# responses = llm.generate(
#     df["text"].values,
#     vllm.SamplingParams(
#         n=1,  # Number of output sequences to return for each prompt.
#         top_p=0.8,  # Float that controls the cumulative probability of the top tokens to consider.
#         temperature=0.0,  # randomness of the sampling
#         seed=777, # Seed for reprodicibility
#         skip_special_tokens=False,  # Whether to skip special tokens in the output.
#         max_tokens=1024,  # Maximum number of tokens to generate per output sequence.
#     ),
#     use_tqdm=True
# )

# responses = [x.outputs[0].text for x in responses]
# df["fullLLMText"] = responses

# # def extract_response(text):
# #     return ",".join(re.findall(r"<response>(.*?)</response>", text)).strip()

# # df["llmMisconception"] = responses
# df.to_parquet("output_llm.parquet", index=False)

# ### RUN LLM

# torch.cuda.empty_cache()
# import gc; gc.collect()

# !python run_vllm.py

# ### POST PROCESS
# LLM output -> submission.csv

# import re

# def extract_tag(s, tag="answer"):
#     """
#     Extracts text enclosed within specified tags (e.g., <answer>...</answer>).

#     Args:
#         s (str): The input string containing the tags.
#         tag (str): The tag name to extract the text from (default is "answer").

#     Returns:
#         str: The text enclosed by the specified tags, or None if no match is found.
#     """
#     pattern = rf"<{tag}>(.*?)</{tag}>"  # Regex to match the tag and its content
#     match = re.search(pattern, s, re.DOTALL)  # Allow multiline matching
#     if match:
#         return match.group(1)  # Return the text inside the tags
#     return None

# # Example usage
# text = "Here is some text with <answer>This is the target text</answer> in it."
# result = extract_tag(text, tag="answer")
# print(result)  # Output: This is the target text



# llm_output = pd.read_parquet("output_llm.parquet")

# all_new_ids = []

# for idx, row in llm_output.iterrows():
#     try:
#         response = row['fullLLMText']
#         answer = extract_tag(response)
#         ids = [int(i.strip()) for i in answer.strip().split(',') if i.strip().isdigit()]
#         print(f"Question ID: {row['QuestionId_Answer']}, Misconception IDs: {ids}")
    
#         first_retrived_ids = [int(i) for i in row['MisconceptionId'].split()]
#         print(f'Before: {first_retrived_ids}')


#         # Step 2: Insert LLM predicted item to the list
#         priority_misconception_ids = []
#         for priority_id in ids:
#             priority_misconception_ids.append(first_retrived_ids.pop(priority_id))
    
#         new_ids = priority_misconception_ids+first_retrived_ids
#         print('After:', new_ids)
#         all_new_ids.append(' '.join([str(i) for i in new_ids]))
#         print("==="*6)
#     except Exception as e:
#         all_new_ids.append(row['MisconceptionId'])
#         print(f'{idx=},\n {row=}\nError:{e}')

# df_input_llm = df_input.copy()
# df_input_llm['MisconceptionId'] = all_new_ids
# df_input_llm[["QuestionId_Answer", "MisconceptionId"]].to_csv("submission.csv", index=False)

# pd.read_csv("submission.csv")
```