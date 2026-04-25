# [LB 0.289] Improved Basline (Prompt Engineering)

- **Author:** Kawchar Husain
- **Votes:** 248
- **Ref:** kawchar85/lb-0-289-improved-basline-prompt-engineering
- **URL:** https://www.kaggle.com/code/kawchar85/lb-0-289-improved-basline-prompt-engineering
- **Last run:** 2025-06-28 12:17:09.413000

---

## Read Data and Extract DOI Links

```python
import os

# vLLM V1 does not currently accept logits processor so we need to disable it
# https://docs.vllm.ai/en/latest/getting_started/v1_user_guide.html#deprecated-features
os.environ["VLLM_USE_V1"] = "0"

import re
import fitz  # PyMuPDF
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
from logits_processor_zoo.vllm import MultipleChoiceLogitsProcessor
import pickle
import vllm
import torch

# Step 1: Read all PDFs and convert to text
# pdf_directory = "/kaggle/input/make-data-count-finding-data-references/test/PDF"

pdf_directory = "/kaggle/input/make-data-count-finding-data-references/test/PDF" \
                if os.getenv('KAGGLE_IS_COMPETITION_RERUN') \
                else "/kaggle/input/make-data-count-finding-data-references/train/PDF"

chunks = []
chunks2 = []
text_span_len = 200

re_doi = re.compile(r"10\.\d{4}")
re_gsr = re.compile(r"GSE\d+|SR[APRX]\d+|PRJ[NAED][A-Z]?\d+")
re_ipe = re.compile(r"IPR\d{6}|PF\d{5}|EMPIAR-\d{5}")
re_c = re.compile(r"CHEMBL\d+|CVCL_[A-Z0-9]{4}")
re_e = re.compile(r"ENS[A-Z]{0,6}[GT]\d{11}")
re_r = re.compile(r"N[MC]_\d+(?:\.\d+)?|rs\d+")
re_u = re.compile(r"(?:uniprot:)?(?:[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9][A-Z][A-Z0-9]{2}[0-9])", re.IGNORECASE)
re_g = re.compile(r"EPI(?:_ISL_)?\d+")
re_p = re.compile(r"PXD\d{6}|SAM[ND]\d+|ERR\d+")

relist = [re_gsr, re_ipe, re_c, re_e, re_r, re_g, re_p]

ids = []

def remove_references_section(text):
    lines = text.split('\n')
    cut_index = -1
    
    # Look backwards from end of document
    for i in range(len(lines) - 1, max(0, int(len(lines) * 0.3)), -1):
        line = lines[i].strip()
        
        obvious_patterns = [
            # References patterns
            r'^REFERENCES?$',                    # All caps, alone
            r'^\d+\.?\s+REFERENCES?$',          # Numbered, all caps
            r'^\d+\.?\s+References?$',          # Numbered, title case
            r'^References?:$',                   # With colon
            
            # Bibliography patterns
            r'^BIBLIOGRAPHY$',                   # All caps, alone
            r'^\d+\.?\s+BIBLIOGRAPHY$',         # Numbered, all caps
            r'^\d+\.?\s+Bibliography$',         # Numbered, title case
            r'^Bibliography:$',                  # With colon
            
            # Other common patterns
            r'^Literature\s+Cited$',            # Literature Cited
            r'^Works\s+Cited$',                 # Works Cited
        ]
        
        if any(re.match(pattern, line, re.IGNORECASE) for pattern in obvious_patterns):
            # Double-check: look at following lines for citation patterns
            following_lines = lines[i+1:i+4]
            has_citations = False
            
            for follow_line in following_lines:
                if follow_line.strip():
                    # Check for obvious citation patterns
                    if (re.search(r'\(\d{4}\)', follow_line) or    # (2020)
                        re.search(r'\d{4}\.', follow_line) or       # 2020.
                        'doi:' in follow_line.lower() or           # DOI
                        ' et al' in follow_line.lower()):          # et al
                        has_citations = True
                        break
            
            # Only cut if we found citation-like content
            if has_citations or i >= len(lines) - 3:  # Or very near end
                cut_index = i
                break
    
    if cut_index != -1:
        return '\n'.join(lines[:cut_index]).strip()
    
    return text.strip()

for filename in tqdm(os.listdir(pdf_directory), total=len(os.listdir(pdf_directory))):
    if filename.endswith(".pdf"):
        pdf_path = os.path.join(pdf_directory, filename)
        
        # Extract article_id from filename
        article_id = filename.split(".pdf")[0]
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            page_text = page.get_text()
            text += page_text + "\n"
            
        doc.close()

        text = remove_references_section(text)

        doi_matches = re_doi.finditer(text)
        for match in doi_matches:
            if match.group() in article_id: continue
            chunk = text[max(0, match.start() - text_span_len): match.start() + text_span_len]
            chunks.append((article_id, chunk))

        for rr in relist:
            matches = rr.finditer(text)
            for match in matches:
                ids.append(match.group())
                chunk = text[max(0, match.start() - text_span_len): match.start() + text_span_len]
                chunks2.append((article_id, chunk))
print(len(chunks))
print(len(chunks2))
```

## Load LLM

```python
model_path = "/kaggle/input/qwen2.5/transformers/32b-instruct-awq/1"

llm = vllm.LLM(
    model_path,
    quantization='awq',
    tensor_parallel_size=torch.cuda.device_count(),
    gpu_memory_utilization=0.9,
    trust_remote_code=True,
    dtype="half",
    enforce_eager=True,
    max_model_len=2048,
    disable_log_stats=True,
    enable_prefix_caching=True
)
tokenizer = llm.get_tokenizer()
```

# System prompts

```python
SYS_PROMPT_DOI = """
You are given a piece of academic text. Your task is to identify a DOI citation that refers specifically to research data.

Only respond with either a full normalized DOI URL starting with "https://doi.org/" or the word "Irrelevant" (without quotes).

Do NOT include any other text or explanation.

If there is no DOI related to research data, respond with exactly "Irrelevant".

If multiple DOIs refer to research data, return any one of them.
"""

SYS_PROMPT_ACCESSION = """
You are given a piece of academic text. Your task is to determine whether the provided Accession ID refers to a dataset used in the study.

Classify the data associated with the Accession ID as:
A) Primary — if the data was generated specifically for this study.
B) Secondary — if the data was reused or derived from prior work.
C) None — if the ID is mentioned in a different context (e.g., not related to data use, or is unrelated to the study).

Respond with only one letter: A, B, or C.
"""

SYS_PROMPT_CLASSIFY_DOI = """
You are given a piece of academic text. Your task is to classify the data associated with the given DOI.

Classify the data as:
A) Primary: if the data was generated specifically for this study.
B) Secondary: if the data was reused or derived from prior work.
C) None: if the DOI is part of the References section of a paper, does not refer to research data or is unrelated.

Respond with only one letter: A, B, or C.
"""
```

## Ask LLM to extract DOI links

```python
prompts = []
for article_id, academic_text in chunks:
    messages = [
        {"role": "system", "content": SYS_PROMPT_DOI},
        {"role": "user", "content": academic_text}
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )
    
    prompts.append(prompt)

outputs = llm.generate(
    prompts,
    vllm.SamplingParams(
        seed=0,
        skip_special_tokens=True,
        max_tokens=32,
        temperature=0
    ),
    use_tqdm=True
)

responses = [output.outputs[0].text.strip() for output in outputs]

doi_pattern = re.compile(r'(10\.\d{4,9}/[-._;()/:A-Z0-9]+)', re.I)

doi_urls = []
for response in responses:
    if response.lower() == "irrelevant":
        doi_urls.append("Irrelevant")
    else:
        match = doi_pattern.search(response)
        if match:
            doi_urls.append("https://doi.org/" + match.group(1))
        else:
            doi_urls.append("Irrelevant")  # fallback
```

## Ask LLM to classify DOI links
Use logits-processor-zoo MultipleChoiceLogitsProcessor to enforce LLM choose between classes.

```python
%%time

prompts = []
valid_indices = []
for i, (chunk, url) in enumerate(zip(chunks, doi_urls)):
    if url == "Irrelevant":
        continue  # skip irrelevant

    article_id, academic_text = chunk
    messages = [
        {"role": "system", "content": SYS_PROMPT_CLASSIFY_DOI},
        {"role": "user", "content": f"DOI: {url}\n\nAcademic text:\n{academic_text}"}
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )
    prompts.append(prompt)
    valid_indices.append(i)

mclp = MultipleChoiceLogitsProcessor(tokenizer, choices=["A", "B", "C"])

outputs = llm.generate(
    prompts,
    vllm.SamplingParams(
        seed=777,
        temperature=0.1,
        skip_special_tokens=True,
        max_tokens=1,
        logits_processors=[mclp],
        logprobs=len(mclp.choices)
    ),
    use_tqdm=True
)

logprobs = []
for lps in [output.outputs[0].logprobs[0].values() for output in outputs]:
    logprobs.append({lp.decoded_token: lp.logprob for lp in list(lps)})

logit_matrix = pd.DataFrame(logprobs)[["A", "B", "C"]].values
choices = ["Primary", "Secondary", None]
answers = [None] * len(chunks)

for i, pick in zip(valid_indices, np.argmax(logit_matrix, axis=1)):
    answers[i] = choices[pick]
```

```python
%%time

prompts = []
for chunk, acc_id in zip(chunks2, ids):
    article_id, academic_text = chunk
    messages = [
        {"role": "system", "content": SYS_PROMPT_ACCESSION},
        {"role": "user", "content": f"Accession ID: {acc_id}\n\nAcademic text:\n{academic_text}"}
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )
    prompts.append(prompt)

outputs = llm.generate(
    prompts,
    vllm.SamplingParams(
        seed=777,
        temperature=0.1,
        skip_special_tokens=True,
        max_tokens=1,
        logits_processors=[mclp],
        logprobs=len(mclp.choices)
    ),
    use_tqdm=True
)

logprobs2 = []
for lps in [output.outputs[0].logprobs[0].values() for output in outputs]:
    logprobs2.append({lp.decoded_token: lp.logprob for lp in list(lps)})

logit_matrix2 = pd.DataFrame(logprobs2)[["A", "B", "C"]].values
choices2 = ["Primary", "Secondary", None]
answers2 = [choices2[pick] for pick in np.argmax(logit_matrix2, axis=1)]
```

## Prepare Submission

```python
sub_df = pd.DataFrame()
sub_df["article_id"] = [c[0] for c in chunks]
sub_df["dataset_id"] = doi_urls
sub_df["dataset_id"] = sub_df["dataset_id"].str.lower()
sub_df["type"] = answers
sub_df = sub_df[sub_df["type"].notnull()].reset_index(drop=True)

sub_df2 = pd.DataFrame()
sub_df2["article_id"] = [c[0] for c in chunks2]
sub_df2["dataset_id"] = ids
sub_df2["type"] = answers2
sub_df2 = sub_df2[sub_df2["type"].notnull()].reset_index(drop=True)

sub_df = pd.concat([sub_df, sub_df2], ignore_index=True)
sub_df = sub_df[sub_df["type"].isin(["Primary", "Secondary"])].reset_index(drop=True)
sub_df = sub_df.sort_values(by=["article_id", "dataset_id", "type"], ascending=False)\
               .drop_duplicates(subset=['article_id', 'dataset_id'], keep="first")\
               .reset_index(drop=True)

sub_df['row_id'] = range(len(sub_df))
sub_df.to_csv("submission.csv", index=False, columns=["row_id", "article_id", "dataset_id", "type"])

print(sub_df["type"].value_counts())
```

## Evaluate validation score

```python
def f1_score(tp, fp, fn):
    return 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) != 0 else 0.0
    
    
if not os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    pred_df = pd.read_csv("submission.csv")
    label_df = pd.read_csv("/kaggle/input/make-data-count-finding-data-references/train_labels.csv")
    label_df = label_df[label_df['type'] != 'Missing'].reset_index(drop=True)

    hits_df = label_df.merge(pred_df, on=["article_id", "dataset_id", "type"])
    
    tp = hits_df.shape[0]
    fp = pred_df.shape[0] - tp
    fn = label_df.shape[0] - tp
    
    print("TP:", tp)
    print("FP:", fp)
    print("FN:", fn)
    print("F1 Score:", round(f1_score(tp, fp, fn), 3))
```