# Konwinski Minimal Qwen Agent

- **Author:** Umar IGAN
- **Votes:** 73
- **Ref:** umar47/konwinski-minimal-qwen-agent
- **URL:** https://www.kaggle.com/code/umar47/konwinski-minimal-qwen-agent
- **Last run:** 2025-02-07 06:47:55.503000

---

A minimal soltion to the problem by creating solver agents

```python
!pip install accelerate
!pip install einops
#!pip install - U bitsandbytes
!pip install transformers_stream_generator==0.0.4
#!pip install -U --no-index --find-links=/kaggle/input/vllm-whl -U vllm
#!pip install -U --upgrade /kaggle/input/vllm-t4-fix/grpcio-1.62.2-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
#!pip install -U --upgrade /kaggle/input/vllm-t4-fix/ray-2.11.0-cp310-cp310-manylinux2014_x86_64.whl
#https://www.kaggle.com/code/tranhoangquan/aimo-ss2-quan/notebook
!unzip -nq ../input/konwinski-prize/data.a_zip
```

```python
import pandas as pd
import torch
from typing import List
from transformers import AutoModelForCausalLM, AutoTokenizer,AutoModel
from transformers import RobertaTokenizer, RobertaModel
import io
import os
import shutil
from pathlib import Path
import fnmatch
from git import Repo
import subprocess
import gc
import kaggle_evaluation.konwinski_prize_inference_server
import numpy as np
from difflib import SequenceMatcher
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
train_data = pd.read_parquet("/kaggle/working/data/data.parquet")
train_data['instance_id'] = "repo__" + train_data['instance_id']
train_data
```

```python
train_data.info()
```

```python
train_data.sample()
```

```python
model_path = "/kaggle/input/codebert-base/codebert-base"
tokenizer = RobertaTokenizer.from_pretrained(
    model_path,
    local_files_only=True,
    vocab_file=os.path.join(model_path, "vocab.json"),
    merges_file=os.path.join(model_path, "merges.txt")
)

model = RobertaModel.from_pretrained(
    model_path,
    local_files_only=True,
    config=os.path.join(model_path, "config.json"),
    state_dict=torch.load(os.path.join(model_path, "pytorch_model.bin"))
)

def calculate_semantic_similarity(pred: str, truth: str, model, tokenizer) -> float:
    inputs = tokenizer([pred, truth], 
                      padding=True, 
                      truncation=True, 
                      max_length=512, 
                      return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    attention_mask = inputs.attention_mask.unsqueeze(-1)
    embeddings = (outputs.last_hidden_state * attention_mask).sum(dim=1) / attention_mask.sum(dim=1).clamp(min=1e-9)
    
    return cosine_similarity(embeddings[0].cpu().numpy().reshape(1, -1), 
                            embeddings[1].cpu().numpy().reshape(1, -1))[0][0]
```

```python
def find_most_related_file(text_input, repo_dir, top_n=3, max_file_size=4000):
    code_files = []
    for root, _, files in os.walk(repo_dir):
        for file in files:
            if file.endswith(('.py', '.java', '.cpp', '.js', '.ts', '.c', '.h')):
                filepath = os.path.join(root, file)
                try:
                    # Check file size
                    if os.path.getsize(filepath) > max_file_size * 1024:  # Convert KB to bytes
                        continue
                    
                    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    code_files.append((filepath, content))
                except Exception as e:
                    print(f"Error reading file {filepath}: {e}")
    
    if not code_files:
        print("No code files found in the repository directory.")
        return []
    code_df = pd.DataFrame(code_files, columns=['file_path', 'file_content'])
    
    file_contents = code_df['file_content'].values.astype(str).tolist()
    all_texts = [str(text_input)] + file_contents  # Ensure text_input is also a string
    
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    
    text_vector = tfidf_matrix[0]
    file_vectors = tfidf_matrix[1:]
    similarity_scores = cosine_similarity(text_vector, file_vectors).flatten()
    
    top_indices = similarity_scores.argsort()[-top_n:][::-1]
    top_files = [code_df.iloc[i]['file_path'] for i in top_indices]
    return top_files
```

```python
#Function to summarize the repo
def get_repo_structure_fallback(repo_path: str, max_depth: int = 3) -> str:
    """Recursive directory structure listing"""
    structure = []
    
    def _walk(path: Path, current_depth: int):
        if current_depth > max_depth:
            return
            
        # Add directory entry
        dir_entry = f"{'  ' * (current_depth-1)}📁 {path.name}/"
        structure.append(dir_entry)
        
        # List files first
        files = sorted([f for f in path.iterdir() if f.is_file()])
        for f in files[:5]:  # Limit files per directory
            structure.append(f"{'  ' * current_depth}📄 {f.name}")
            
        # Recursively list directories
        dirs = sorted([d for d in path.iterdir() if d.is_dir() 
                      and d.name not in ['.git', '__pycache__', 'node_modules']])
        for d in dirs[:5]:  # Limit subdirectories
            _walk(d, current_depth + 1)
    
    try:
        _walk(Path(repo_path), 1)
        return "\n".join(structure)[:2000]  # Truncate long outputs
    except Exception as e:
        return f"Error generating structure: {str(e)}"
def get_git_aware_structure(repo_path: str) -> str:
    """Get structure with git history insights"""
    structure = []
    repo = Repo(repo_path)
    
    # Get recent authors
    contributors = set()
    for commit in repo.iter_commits('HEAD', max_count=10):
        contributors.add(commit.author.name)
    
    # Get modified files
    modified = [item.a_path for item in repo.index.diff(None)]
    
    # Build structure
    structure.append(f"Recent contributors: {', '.join(contributors)[:100]}")
    structure.append("Recent modified files:")
    structure.extend(modified[:10])
    
    return "\n".join(structure)
    
def identify_important_files(repo_path: str) -> str:
    """Identify likely important files"""
    priority_files = []
    
    # Common important file patterns
    important_patterns = [
        'requirements.txt', 'setup.py', 'package.json',
        'Dockerfile', 'Makefile', '*.md',
        'src/', 'lib/', 'main.py', 'app.py'
    ]
    
    for root, _, files in os.walk(repo_path):
        for f in files:
            path = Path(root) / f
            rel_path = path.relative_to(repo_path)
            
            # Check against known patterns
            if any(fnmatch.fnmatch(str(rel_path), pat) for pat in important_patterns):
                priority_files.append(f"* {rel_path}")
                
            # Check file size
            if path.stat().st_size > 100000:  # >100KB
                priority_files.append(rel_path)
    
    return priority_files[:20]
def analyze_repo_structure(repo_path: str) -> str:
    """Combine multiple structure analysis methods"""
    parts = []
    
    # Basic directory structure
    parts.append("## Directory Structure ##")
    parts.append(get_repo_structure_fallback(repo_path))
    
    # Important files analysis
    parts.append("\n## Key Files ##")
    parts.append(identify_important_files(repo_path))
    
    # Git insights
    try:
        parts.append("\n## Development Insights ##")
        parts.append(get_git_aware_structure(repo_path))
    except:
        pass
    
    return parts
#analyze_repo_structure(repo_path)
```

```python
def cosine_similarity_reviews(text1, text2):
    # Combine the two reviews into a list
    reviews = [text1, text2]
    
    # Step 1: Compute TF-IDF embeddings for both reviews
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(reviews)
    
    # Step 2: Calculate cosine similarity
    similarity_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    
    return similarity_score
```

```python
#Example Usage
text_input = train_data['problem_statement'].values[4]
print("issue: ", text_input[:200])
repo_dir = "/kaggle/working/data/repos"
print(datetime.now().strftime("Hour: %H, Minute: %M, Second: %S"))
top_related_file = find_most_related_file(text_input, repo_dir, top_n=1)
print(datetime.now().strftime("Hour: %H, Minute: %M, Second: %S"))
for file_path in top_related_file:
        print(f"File: {file_path}\n")
```

```python
torch.cuda.empty_cache()
gc.collect()
#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512,garbage_collection_threshold:0.8,expandable_segments:True"

class GithubIssueSolver:
    def __init__(self):
        model_name = "/kaggle/input/qwen2.5-coder/transformers/7b-instruct/1"#
        
        # More conservative memory settings
        #max_memory = {0: "12GiB", "cpu": "12GiB"}
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            #max_memory=max_memory,
            low_cpu_mem_usage=True,
            offload_folder="offload",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_context_length = 16384  # Reduced from 4096
        self.max_file_size = 2000  # Reduced from 1000
    def _create_prompt(self, problem_statement: str, context: str, repo_structure: str) -> str:
        return f"""**Task**: Fix the GitHub issue by generating a correct git diff patch. Use strict step-by-step reasoning.
    
                **Format Requirements**:
                1. Output MUST start with 'diff --git'
                2. Understand where to modify from relevant files (max 3 files)
                3. Include precise line numbers
                4. Never write code comments unless present in original
                5. Include only necessary changes
                
                **Problem Analysis Framework**:
                1. Root Cause Identification:
                   - Identify specific components causing the issue
                   - Analyze error patterns from problem description
                
                2. Code Context Mapping:
                   - Match issue components to relevant code sections
                   - File structure: {repo_structure}
                   - Relevant code snippets: {context}
                
                3. Change Validation:
                   - Cross-verify each change against problem statement
                   - Ensure no unrelated code modifications
                
                **Example of Good Patch**:
                diff --git a/file.py b/file.py
                --- a/file.py
                +++ b/file.py
                @@ -12,7 +12,7 @@
                     try:
                -        result = process(data)
                +        result = process(data, timeout=30)
                     except TimeoutError:
                -        logger.warning("Timeout occurred")
                +        logger.error("Timeout (30s) exceeded", exc_info=True)
                
                **Current Issue**:
                {problem_statement}
                
                **Step-by-Step Process**:
                1. Identify key components needing modification
                2. Locate exact lines in relevant files
                3. Make minimal changes to fix issue
                4. Verify against all mentioned edge cases
                
                **Output Instructions**:
                - Start immediately with diff patch
                - Use exact file paths from repository
                - Include confidence score (0-100) as last line
                - If uncertain, output "SKIP" with reason
                - Return a valid patch.
                
                **Begin Fix**:
                """
    def analyze_issue(self, problem_statement: str, repo_path: str) -> str:
        try:
            torch.cuda.empty_cache()
            gc.collect()
            
            relevant_files = self._find_relevant_files(repo_path, problem_statement)
            if not relevant_files:
                print("No relevant files found")
                return None
                
            context = self._read_file_contents(repo_path, relevant_files)
            repo_structure = analyze_repo_structure(repo_path)
            prompt= self._create_prompt(problem_statement, context, repo_structure)
            #modified CoT prompt from: https://arxiv.org/pdf/2501.05040
            messages = [
                {"role": "system", "content": """"
                                    You are an expert software engineer and code reviewer specializing in resolving real-world GitHub issues by creating precise diff patches. Your role is to analyze the issue description and the most relevant code segments from the repository (pre-identified using cosine similarity) to propose effective and accurate code modifications.
                                    
                                    In this task, you will:
                                    
                                    1. **Understand the Issue:** Carefully analyze the provided GitHub issue description to identify the root cause and the functional or structural problem in the code.
                                    2. **Analyze Relevant Code:** Examine the provided code snippets or files identified as most relevant to the issue. Use your expertise to determine the exact areas requiring modification.
                                    3. **Generate a Diff Patch:** Create a detailed and well-structured diff patch that resolves the issue while maintaining the integrity and functionality of the codebase.
                                    4. **Provide Reasoning:** Accompany your patch with a clear, step-by-step explanation of your reasoning process, detailing how the proposed changes address the issue effectively.
                                    
                                    ### Guidelines:
                                    - **Independent Reasoning:** Your analysis and diff patch should be based solely on the issue description and the provided code snippets. Avoid referencing external solutions or implying prior knowledge of oracle modifications.
                                    - **Clarity and Precision:** Ensure that your diff patch is syntactically correct, adheres to best coding practices, and is easy to apply.
                                    - **Evidence-Based Reasoning:** Clearly justify your changes, linking them to specific parts of the issue description and code. Highlight how the modifications resolve the issue and improve the codebase.
                                    
                                    This task focuses on accurately resolving GitHub issues through diff patches while maintaining high standards of clarity, precision, and logical consistency.
                                                """},
                {"role": "user", "content": prompt}
            ]
            
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # Split processing into smaller chunks
            inputs = self.tokenizer(
                text, 
                return_tensors="pt", 
                truncation=True,
                max_length=self.max_context_length
            ).to(self.model.device)
            
            with torch.inference_mode():
                try:
                    generated_ids = self.model.generate(
                        input_ids=inputs['input_ids'],
                        max_new_tokens=16384,  # Reduced from 1024
                        temperature=0.8,#from 0.7
                        do_sample=True,
                        pad_token_id=self.tokenizer.pad_token_id,
                        num_return_sequences=1,
                    )
                    
                    response = self.tokenizer.decode(
                        generated_ids[0, inputs['input_ids'].shape[1]:],
                        skip_special_tokens=True
                    )
                finally:
                    del inputs
                    torch.cuda.empty_cache()
                    gc.collect()
            
            if "diff --git" in response:
                diff_start = response.find("diff --git")
                return response[diff_start:]
            return None
            
        except Exception as e:
            print(f"Error generating solution: {str(e)}")
            return None
        finally:
            torch.cuda.empty_cache()
            gc.collect()

    def _find_relevant_files(self, repo_path: str, problem_statement: str) -> list:
        relevant_files = []
        relevant_files = find_most_related_file(problem_statement, repo_path, top_n=5)
        """
        try:
            keywords = set(problem_statement.lower().split())
            #print("keywords: ", keywords[:25])
            for root, _, files in os.walk(repo_path):
                if len(relevant_files) >= 2:
                    break
                    
                for file in files:
                    if file.endswith(('.py', '.java', '.cpp', '.h', '.c', '.js', '.ts')):
                        file_path = os.path.join(root, file)
                        try:
                            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                                content = f.read(self.max_file_size * 50)  # Read with size limit
                                
                            if any(word in content.lower() for word in keywords):
                                relevant_files.append(os.path.relpath(file_path, repo_path))
                                if len(relevant_files) >= 2:
                                    break
                        except Exception:
                            continue
                                
        except Exception as e:
            print(f"Error finding relevant files: {str(e)}")
        """
        print("num of relevant files",  len(relevant_files))
        print("relevant files",  relevant_files)
        return relevant_files

    def _read_file_contents(self, repo_path: str, files: list) -> str:
        contents = []
        total_lines = 0
        
        for file in files:
            if total_lines >= self.max_file_size:
                break
                
            try:
                with open(os.path.join(repo_path, file), 'r', encoding='utf-8', errors='ignore') as f:
                    lines = []
                    for i, line in enumerate(f):
                        if i >= self.max_file_size // len(files):
                            break
                        lines.append(line)
                    contents.append(f"File: {file}\n{''.join(lines)}")
                    total_lines += len(lines)
            except Exception:
                continue
        print("file content: ", "\n".join(contents)[:200])
        return "\n".join(contents)

# Global solver instance
solver = None
```

```python
if solver is None:
    solver = GithubIssueSolver()
```

```python
data_t = train_data.sample()
data_t
```

```python
repo_name = data_t['instance_id'].values[0]#.replace('/', '_')
base_repo_path = "/kaggle/working/data/repos/"
#repo_path = base_repo_path + repo_name
repo_path = os.path.join(base_repo_path, repo_name)
if not os.path.exists(repo_path):
    raise ValueError(f"Repository path does not exist: {repo_path}")
print("repo_path: ",repo_path)
# Prepare inputs
problem_statement = data_t['problem_statement']
print("problem_statement: ",problem_statement[:25])
```

```python
from unidiff import PatchSet
def is_valid_diff(patch):
    try:
        PatchSet(patch)
        return True
    except Exception:
        return False
```

```python
"""
print(datetime.now().strftime("Hour: %H, Minute: %M, Second: %S")) 
response = solver.analyze_issue(
    problem_statement.values[0],
    repo_path)
print(datetime.now().strftime("Hour: %H, Minute: %M, Second: %S"))
if is_valid_diff(response)==True:
    print('it is valid')
else:
    print("it is not valid")
"""
```

```python
from IPython.display import display, Markdown, Latex
#display(Markdown(response))
```

```python
display(Markdown(data_t['patch'].values[0]))
```

```python
solver = GithubIssueSolver()
```

```python
#cosine_sim = calculate_semantic_similarity(data_t['patch'].values[0], response, model, tokenizer)
#print("Cosine Similarity: ", cosine_sim)#prev: Cosine Similarity:  0.11839781672838229
```

```python
instance_count = None
def get_number_of_instances(num_instances: int) -> None:
    global instance_count
    instance_count = num_instances

def predict(
    problem_statement: str, 
    repo_archive: io.BytesIO, 
    pip_packages_archive: io.BytesIO, 
    env_setup_cmds_templates: list[str]
) -> str:
    global solver
    
    # Define repo_path at the start
    repo_path = os.path.join(os.getcwd(), 'repo')
    
    #try:
    if solver is None:
        solver = GithubIssueSolver()
    
    # Clean up any existing repo directory
    if os.path.exists(repo_path):
        shutil.rmtree(repo_path)
    
    # Create archive file and extract
    archive_path = os.path.join(os.getcwd(), 'repo_archive.tar')
    with open(archive_path, 'wb') as f:
        f.write(repo_archive.read())
        
    os.makedirs(repo_path, exist_ok=True)
    shutil.unpack_archive(archive_path, repo_path)
    os.remove(archive_path)
    patch_string = solver.analyze_issue(problem_statement, repo_path)
    print("sol: ", patch_string[:25])
    print("is valid: ", is_valid_diff(patch_string))
    # Process the issue
    if patch_string is None:
        return None
    return patch_string
        
    #except Exception as e:
    #    print(f"Error in predict function: {str(e)}")
    #    return None
    #finally:
    #    # Clean up
    #    if os.path.exists(repo_path):
    #        shutil.rmtree(repo_path)
    #    #torch.cuda.empty_cache()
    #    #gc.collect()
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
            '/kaggle/input/konwinski-prize/',
            '/kaggle/tmp/konwinski-prize/',
        ),
        use_concurrency=True,
    )
```