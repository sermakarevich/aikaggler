# Starter notebook - Select-Patch-Verify ca53bc

- **Author:** zhudong1949
- **Votes:** 121
- **Ref:** zhudong1949/starter-notebook-select-patch-verify-ca53bc
- **URL:** https://www.kaggle.com/code/zhudong1949/starter-notebook-select-patch-verify-ca53bc
- **Last run:** 2025-02-14 05:25:35.857000

---

```python
import os
# https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-2/discussion/560682#3113134
os.environ["TRITON_PTXAS_PATH"] = "/usr/local/cuda/bin/ptxas"
```

```python
import io
import os
import shutil

import pandas as pd
import polars as pl

import kaggle_evaluation.konwinski_prize_inference_server
```

The evaluation API requires that you set up a server which will respond to inference requests. We have already defined the server; you just need write the predict function. When we evaluate your submission on the hidden test set the client defined in `konwinski_prize_gateway` will run in a different container with direct access to the hidden test set and hand off the data.

Your code will always have access to the published copies of the files.

```python
instance_count = None

def get_number_of_instances(num_instances: int) -> None:
    """ The very first message from the gateway will be the total number of instances to be served.
    You don't need to edit this function.
    """
    global instance_count
    instance_count = num_instances
```

# Initialize LLM

```python
from vllm import LLM, SamplingParams
import warnings

warnings.simplefilter('ignore')

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

if os.getenv('KAGGLE_KERNEL_RUN_TYPE') or os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    llm_model_pth = '/kaggle/input/m/shelterw/deepseek-r1/transformers/deepseek-r1-distill-qwen-32b-awq/1'
else:
    llm_model_pth = '/root/volume/KirillR/QwQ-32B-Preview-AWQ'

MAX_NUM_SEQS = 1
MAX_MODEL_LEN = 32_768
MAX_TOKENS = 8192

llm = LLM(
    llm_model_pth,
    # dtype="half",               # The data type for the model weights and activations
    max_num_seqs=MAX_NUM_SEQS,   # Maximum number of sequences per iteration. Default is 256
    max_model_len=MAX_MODEL_LEN, # Model context length
    trust_remote_code=True,      # Trust remote code (e.g., from HuggingFace) when downloading the model and tokenizer
    tensor_parallel_size=4,      # The number of GPUs to use for distributed execution with tensor parallelism
    gpu_memory_utilization=0.95, # The ratio (between 0 and 1) of GPU memory to reserve for the model
    seed=2025,
)
```

```python
tokenizer = llm.get_tokenizer()
```

# Helper functions

```python
import os

def stringify_directory(directory):
    full_paths = []
    
    rel_path_start = len(directory) + 1  # +1 for the trailing slash
    for root, dirs, files in os.walk(directory):
        for file in files:
            full_path = os.path.join(root, file)
            full_paths.append(full_path)
    return "\n".join(full_paths)
```

```python
import re


def extract_file_query(xml_content):
    import xml.etree.ElementTree as ET

    # Prepare a data structure to collect results
    parsed_data = {}
    pattern = r'<root>(.*?)</root>'
    matches = re.findall(pattern, xml_content, re.DOTALL)
    
    for match in matches:
        try:
            # Parse the XML
            root = ET.fromstring("<root>" + match + "</root>")
            
            # Find all <entry> elements
            for entry in root.findall("entry"):
                # Extract the <filepath> text
                filepath = entry.find("filepath")
                filepath_text = filepath.text.strip() if filepath is not None else None
            
                # Locate <strings_to_search> container
                strings_container = entry.find("strings_to_search")
                
                # Gather each <string_to_search> text
                search_strings = []
                if strings_container is not None:
                    for s in strings_container.findall("string_to_search"):
                        if s.text is not None:
                            search_strings.append(s.text.strip())
                
                # Store in a dictionary: { filepath: [search_strings...] }
                parsed_data[filepath_text] = search_strings
        except:
            print("Error parsing output", xml_content)
            return ""
        
    return parsed_data
```

```python
reading_prompt = """
You will be implementing a git diff patch to solve an issue with the code repository.
You will first need to select files in the file directory.

This is the problem statement.

{problem_statement}

This is the file directory

<directory>
{directory_string}
</directory>

Which files should be inspected so that we can solve the problem?
When we inspect each file, what strings should be searched?

Return the strings to search in this format

(explanation)

<root>
    <entry>
        <filepath>filepath</filepath>  
        <strings_to_search>
            <string_to_search>string_to_search</string_to_search>
            ...
            <string_to_search>string_to_search</string_to_search>
        </strings_to_search>
    </entry>
</root>

(explanation)

<root>
    <entry>
        <filepath>filepath</filepath>
        <strings_to_search>
            <string_to_search>string_to_search</string_to_search>
            ...
            <string_to_search>string_to_search</string_to_search>
        </strings_to_search>
    </entry>
</root>
...

Notes:
- Make sure to encode each entry between <root> and </root>
- Return the FULL filepath - exactly as specified in <directory> and </directory>
    - Example: repo/path/to/directory/file.py
- If you are searching for a word instead of a substring, maybe add spaces or brackets before and after the string
    For example, if you are searching for uses of the function `calculate`, use ` calculate(` as the search string instead
- Prefer searching longer strings
- Do not inspect more than 5 files
- Only inspect the necessary files
""".strip()


def get_file_query(directory_string, problem_statement):

    sampling_params = SamplingParams(
        temperature=1.0,              # randomness of the sampling
        min_p=0.01,
        skip_special_tokens=True,     # Whether to skip special tokens in the output
        max_tokens=MAX_MODEL_LEN,
    )
    
    list_of_messages = [
        [
            {
                "role": "user",
                "content": reading_prompt.format(
                    problem_statement=problem_statement,
                    directory_string=directory_string,
                )
            },
        ]
    ]

    list_of_texts = [
        tokenizer.apply_chat_template(
            conversation=messages,
            tokenize=False,
            add_generation_prompt=True
        )
        for messages in list_of_messages
    ]
    # print(list_of_texts)
    print([len(tokenizer.encode(text)) for text in list_of_texts])

    request_output = llm.generate(prompts=list_of_texts, sampling_params=sampling_params)
    if not request_output:
        return "", ""
    response_text = request_output[0].outputs[0].text
    file_query = extract_file_query(response_text)
    return file_query, response_text
```

```python
REPO_PATH = "repo"

def fetch_file_contents(files_to_search, context_lines=10, max_gap=0):
    from io import StringIO

    def find_lines_in_files_with_context(search_map, context_lines=context_lines):
        """
        Given a dictionary mapping file paths to a list of search terms,
        open each file and gather *snippets* of lines that contain any
        of those search terms, including 'context_lines' before and after.

        Returns a list of lists:
        [
          [  # For file1
             [ (line_number, text), (line_number, text), ... ],
             [ ... ],
          ],
          [  # For file2
             ...
          ],
          ...
        ]
        """
        all_matches_per_file = []

        for path, terms in search_map.items():
            if not os.path.isfile(path):
                # If the file is not found, record an empty list
                all_matches_per_file.append([])
                continue

            with open(path, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()

            file_snippets = []
            num_lines = len(lines)

            for i, line in enumerate(lines, start=1):
                if any(t in line for t in terms):
                    start_idx = max(1, i - context_lines)
                    end_idx = min(num_lines, i + context_lines)
                    snippet = []
                    for snippet_no in range(start_idx, end_idx + 1):
                        text_content = lines[snippet_no - 1].rstrip("\n")
                        snippet.append((snippet_no, text_content))
                    file_snippets.append(snippet)

            all_matches_per_file.append(file_snippets)

        return all_matches_per_file

    # ---------------------------------------------------------
    # 3. MERGE OVERLAPPING/ADJACENT SNIPPETS
    # ---------------------------------------------------------

    def merge_file_snippets(file_snippets, gap=0):
        """
        Merge overlapping or nearly adjacent snippets in a single file’s snippet list.
        """
        intervals = []
        for snippet in file_snippets:
            if snippet:
                start_line = snippet[0][0]
                end_line = snippet[-1][0]
                intervals.append((start_line, end_line, snippet))

        intervals.sort(key=lambda x: x[0])  # sort by start line

        merged = []
        for start, end, snippet in intervals:
            if not merged:
                merged.append((start, end, snippet))
                continue

            prev_start, prev_end, prev_snippet = merged[-1]
            if start <= prev_end + gap:
                new_end = max(end, prev_end)
                combined_dict = {}
                for ln, txt in prev_snippet:
                    combined_dict[ln] = txt
                for ln, txt in snippet:
                    combined_dict[ln] = txt
                merged_snippet = [(ln, combined_dict[ln]) for ln in sorted(combined_dict)]
                merged[-1] = (prev_start, new_end, merged_snippet)
            else:
                merged.append((start, end, snippet))

        # Extract just the merged snippet portion
        return [x[2] for x in merged]

    def merge_all_snippets(all_files_snips, gap=0):
        """
        Merge snippet blocks within each file.
        all_files_snips is a list-of-lists:
          [
            [ snippetA, snippetB, ... ],  # file 1
            [ snippetC, snippetD, ... ],  # file 2
          ]
        """
        merged = []
        for snips in all_files_snips:
            merged.append(merge_file_snippets(snips, gap=gap))
        return merged

    # ---------------------------------------------------------
    # 4. RUN LOGIC: generate files, search, merge, and BUILD A STRING
    # ---------------------------------------------------------

    has_any_matches = False

    # 1) Gather snippets around each match
    context_snippets = find_lines_in_files_with_context(files_to_search, context_lines=context_lines)

    # 2) Merge overlapping snippets
    merged_snips = merge_all_snippets(context_snippets, gap=max_gap)

    # 3) Build a string (instead of printing)
    output = StringIO()

    # Header
    output.write("Sample files created successfully.\n\n")
    output.write("Search Results (by file, merging any overlapping context):\n\n")

    # For each file
    for (filepath, snippet_list) in zip(files_to_search.keys(), merged_snips):
        output.write(f"FILE: {filepath[len(REPO_PATH) + 1:]}\n")
        output.write("-" * 60 + "\n")
        if not snippet_list:
            output.write("  No matches found.\n")
        else:
            has_any_matches = True
            for snippet_idx, snippet in enumerate(snippet_list, start=1):
                snippet_start = snippet[0][0]
                snippet_end = snippet[-1][0]
                output.write(f"Match #{snippet_idx}, lines {snippet_start} to {snippet_end}:\n")
                for line_no, text in snippet:
                    output.write(f"  {line_no:3d} | {text}\n")
                output.write("\n")
        output.write("=" * 60 + "\n\n")

    file_content_string = output.getvalue()

    if has_any_matches:
        return file_content_string
    return ""
```

```python
def extract_patch_string(text):
    pattern = r'<patch>(.*?)</patch>'
    matches = re.findall(pattern, text, re.DOTALL)
    if not matches:
        return None
    return "\n".join(matches)
```

```python
patching_prompt = """
You will be implementing a git diff patch to solve an issue with the code repository.
This is the problem statement.

{problem_statement}

These are the files that is thought to be relevant

{file_content_string}

Write a git diff within <patch> and </patch> that fixes the problem.

Example:

<patch>
--- a/first.txt
+++ b/first.txt
@@ -1,3 +1,3 @@
 start
-first change
+new first change
 middle
@@ -7,4 +7,4 @@
 some content
-second change
+new second change
 more content
--- a/second.txt
+++ b/second.txt
@@ -1,3 +1,3 @@
 beginning
-old line
+new line
 end
</patch>
""".strip()

import re

def get_patch_string(problem_statement, file_content_string):
    
    sampling_params = SamplingParams(
        temperature=0.7,              # randomness of the sampling
        min_p=0.01,
        skip_special_tokens=True,     # Whether to skip special tokens in the output
        max_tokens=MAX_MODEL_LEN,
    )
    
    list_of_messages = [
        [
            {
                "role": "user",
                "content": patching_prompt.format(
                    problem_statement=problem_statement,
                    file_content_string=file_content_string,
                )
            },
        ]
    ]

    list_of_texts = [
        tokenizer.apply_chat_template(
            conversation=messages,
            tokenize=False,
            add_generation_prompt=True
        )
        for messages in list_of_messages
    ]
    # print(list_of_texts)
    print([len(tokenizer.encode(text)) for text in list_of_texts])
    
    request_output = llm.generate(prompts=list_of_texts, sampling_params=sampling_params)
    if not request_output:
        return "", ""
    response_text = request_output[0].outputs[0].text
    patch_string = extract_patch_string(response_text)
    
    return patch_string, response_text
```

```python
verifying_prompt = """
This is the problem statement.

{problem_statement}

These are the files that is thought to be relevant, which may not be complete.

{file_content_string}

This is the proposed batch to fix the problem.

{patch_string}

Firstly, list your observations.
Then, evaluate whether the patch fully fixes the problem described in the problem statement.

End your response with exactly either of
- <label>Yes</label>, this fixes the problem.
- <label>No</label>, this does not fix the problem.

Note
- Only evaluate, do not provide suggestion on how to fix.
- Remember to write exactly either of <label>Yes</label> or <label>No</label> in the last line
""".strip()


def get_chosen_patch(problem_statement, file_content_string, patch_string):

    sampling_params = SamplingParams(
        temperature=0.7,              # randomness of the sampling
        min_p=0.01,
        skip_special_tokens=True,     # Whether to skip special tokens in the output
        max_tokens=MAX_TOKENS,
    )

    list_of_messages = [
        [
            {
                "role": "user",
                "content": verifying_prompt.format(
                    problem_statement=problem_statement,
                    file_content_string=file_content_string,
                    patch_string=patch_string,
                )
            },
        ] for _ in range(MAX_NUM_SEQS)
    ]

    list_of_texts = [
        tokenizer.apply_chat_template(
            conversation=messages,
            tokenize=False,
            add_generation_prompt=True
        )
        for messages in list_of_messages
    ]
    # print(list_of_texts)
    print("get_chosen_patch", [len(tokenizer.encode(text)) for text in list_of_texts])
    request_outputs = llm.generate(prompts=list_of_texts, sampling_params=sampling_params)
    if not request_outputs:
        return None, ""
    response_texts = [request_output.outputs[0].text for request_output in request_outputs]
    print("get_chosen_patch", [len(tokenizer.encode(text)) for text in response_texts])

    judgments = ["<label>Yes</label>" in response_text for response_text in response_texts]
    print(judgments)
    for response_text, judgment in zip(response_texts, judgments):
        if judgment is True:
            # return None, response_text  # NB: last response text
            return patch_string, response_text

    # return patch_string, response_text
    return None, response_text  # NB: last response text
```

# Predict function

```python
def predict_inner(problem_statement: str, directory: str) -> str:
    directory_string = stringify_directory(directory)
    file_query, get_file_query_response_text = get_file_query(directory_string, problem_statement)
    file_queries, get_file_query_response_texts = [file_query], [get_file_query_response_text]
    for file_query, get_file_query_response_text in zip(file_queries, get_file_query_response_texts):
        print(get_file_query_response_text)
        print("file_query from get_file_query_response_text")
        if not os.getenv('KAGGLE_IS_COMPETITION_RERUN'): print(file_query)

    file_content_strings = [fetch_file_contents(file_query) for file_query in file_queries]
    for file_content_string in file_content_strings:
        print(file_content_string)
    file_content_strings = [file_content_string for file_content_string in file_content_strings if file_content_string != ""]
    if len(file_content_strings) == 0:
        return None

    patch_string, get_patch_string_response_text = get_patch_string(problem_statement, file_content_strings[0])
    patch_string_to_return = None
    for patch_string, get_patch_string_response_text in zip([patch_string], [get_patch_string_response_text]):
        print(get_patch_string_response_text)
        print("patch_string from get_patch_string_response_text")
        if not os.getenv('KAGGLE_IS_COMPETITION_RERUN'): print(patch_string)
        if patch_string is not None:
            patch_string_to_return = patch_string
    
    if not patch_string_to_return:
        return None

    # patch_string = patch_string_to_return
    # patch_string, get_chosen_patch_response_text = get_chosen_patch(problem_statement, file_content_string, patch_string)
    # print(get_chosen_patch_response_text)
    # print("patch_string from get_chosen_patch_response_text")
    # print(patch_string)

    return patch_string_to_return
```

```python
import io
from typing import Optional

skip_prediction = False


def predict(problem_statement: str, repo_archive: io.BytesIO, pip_packages_archive: io.BytesIO, env_setup_cmds_templates: list[str]) -> str:
    """ Replace this function with your inference code.
    Args:
        problem_statement: The text of the git issue.
        repo_archive: A BytesIO buffer path with a .tar containing the codebase that must be patched. The gateway will make this directory available immediately before this function runs.
    """
    global skip_prediction
    if skip_prediction:
        return None

    with open('repo_archive.tar', 'wb') as f:
        f.write(repo_archive.read())
    repo_path = REPO_PATH
    if os.path.exists(repo_path):
        shutil.rmtree(repo_path)
    shutil.unpack_archive('repo_archive.tar', extract_dir=repo_path)
    os.remove('repo_archive.tar')

    for _ in range(1):
        patch_string = predict_inner(problem_statement=problem_statement, directory=repo_path)
        if patch_string is not None:
            break
    shutil.rmtree(repo_path)

    if patch_string is None:
        return None

    print("submitted patch_string")
    print(patch_string)
 
    # if not os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    #     skip_prediction = True

    return patch_string
```

# Get predict data without server

```python
!mkdir -p /kaggle/tmp/konwinski-prize-alt
!unzip -q -o /kaggle/input/konwinski-prize/data.a_zip -d /kaggle/tmp/konwinski-prize-alt/ 2>/dev/null || true
```

```python
import pandas as pd

def get_problem(problem_index):
    df = pd.read_parquet('/kaggle/tmp/konwinski-prize-alt/data/data.parquet')
    
    problem_statement = df["problem_statement"][problem_index]
    repo_path = f"/kaggle/tmp/konwinski-prize-alt/data/repos/repo__{df['instance_id'][problem_index]}"
    
    import shutil
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmpdir:
        shutil.make_archive(os.path.join(tmpdir, 'a_repo'), 'tar', repo_path)
        with open(os.path.join(tmpdir, 'a_repo.tar'), 'rb') as f:
            repo_archive = io.BytesIO(f.read())

    return problem_statement, repo_path, repo_archive
```

```python
demo_problem_index = 0

if not os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    problem_statement, repo_path, repo_archive = get_problem(problem_index=demo_problem_index)
    
    print(repo_path)
    print(problem_statement)
    print(len(list(repo_archive)))
    print(len(list(repo_archive)))
```

```python
if not os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    skip_prediction = False
    problem_statement, repo_path, repo_archive = get_problem(problem_index=demo_problem_index)
    patch_string = predict(problem_statement, repo_archive, None, None)
```

```python
if not os.getenv('KAGGLE_IS_COMPETITION_RERUN') and patch_string is not None:
    import polars as pl
    df = pl.read_parquet('/kaggle/tmp/konwinski-prize-alt/data/data.parquet')
    
    import kaggle_evaluation.konwinski_prize_gateway
    k_prize_gateway = kaggle_evaluation.konwinski_prize_gateway.KPrizeGateway()
    k_prize_gateway.unpack_data_paths()

    results = k_prize_gateway._evaluate_instance(
        instance = df.row(demo_problem_index, named=True),
        patch = patch_string,
    )
    from collections import Counter
    print(demo_problem_index, Counter(result.unit_test_outcome for result in results[1:]))
```

```python
if not os.getenv('KAGGLE_IS_COMPETITION_RERUN') and patch_string is not None:
    from kaggle_evaluation.konwinski_prize_gateway import UnitTestOutcome
    for result in results[1:]:
        if result.unit_test_outcome != UnitTestOutcome.PASSED:
            print(result.test_name)
            print(result.fail_description)
```

When your notebook is run on the hidden test set, inference_server.serve must be called within 15 minutes of the notebook starting or the gateway will throw an error. If you need more than 15 minutes to load your model you can do so during the very first predict call, which does not have the usual 30 minute response deadline.

# Evaluation with inference server

```python
skip_prediction = False
```

```python
inference_server = kaggle_evaluation.konwinski_prize_inference_server.KPrizeInferenceServer(
    get_number_of_instances,   
    predict
)

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    inference_server.run_local_gateway(
        data_paths=(
            '/kaggle/input/konwinski-prize/',  # Path to the entire competition dataset
            '/kaggle/tmp/konwinski-prize/',   # Path to a scratch directory for unpacking data.a_zip.
        )
    )
```