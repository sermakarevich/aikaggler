# How to get started with the Meta Kaggle Hackathon

- **Author:** Paul Mooney
- **Votes:** 121
- **Ref:** paultimothymooney/how-to-get-started-with-the-meta-kaggle-hackathon
- **URL:** https://www.kaggle.com/code/paultimothymooney/how-to-get-started-with-the-meta-kaggle-hackathon
- **Last run:** 2025-05-29 16:44:01.550000

---

# How to get started with the Meta Kaggle Hackathon

### There are two primary datasets to consider:
 - [Meta Kaggle](https://www.kaggle.com/datasets/kaggle/meta-kaggle)
 - [Meta Kaggle Code](https://www.kaggle.com/datasets/kaggle/meta-kaggle-code)

Meta Kaggle contains .CSV files that record site activity on Kaggle.

Meta Kaggle for Code contains .ipynb and .py files full of code that was shared on Kaggle.

```python
import kagglehub

MK_PATH = kagglehub.dataset_download("kaggle/meta-kaggle")
MKC_PATH = kagglehub.dataset_download("kaggle/meta-kaggle-code")

print("Path to Meta-Kaggle dataset files:", MK_PATH)
print("Path to Meta-Kaggle-Code dataset files:", MKC_PATH)
```

```python
import pandas as pd
import codecs
import glob
import json
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning, module='pandas.io.formats.format')


def id_to_path(file_id: int) -> str:
    padded_id_str = str(file_id).zfill(10)
    prefix = f"{MKC_PATH}/{padded_id_str[0:4]}/{padded_id_str[4:7]}/{file_id}.*"
    matching_paths = glob.glob(prefix)
    if len(matching_paths) == 1:
        return matching_paths[0]
    return ""


def get_file_extension(file_path: str) -> str:
    parts = os.path.splitext(file_path)
    if len(parts) != 2:
        return ""
    return parts[1]


def path_to_id(file_path: str) -> int | None:
    base_name, _ = os.path.splitext(os.path.basename(file_path))
    if base_name:
        try:
            return int(base_name)
        except ValueError:
            return None
    return None


def get_ipynb_source(ipynb_path: str) -> list[list[str]]:
    with codecs.open(ipynb_path, 'r', encoding='utf-8') as f:
        raw_source_content = f.read()

    json_data = json.loads(raw_source_content)
    cells_source = []
    for cell in json_data.get('cells', []):
        if cell.get('cell_type') == 'code':
            cells_source.append(cell.get('source', []))
    return cells_source


def get_source_code(file_path: str) -> list[list[str]]:
    file_extension = get_file_extension(file_path)
    if file_extension == ".ipynb":
        return get_ipynb_source(file_path)
    return [Path(file_path).read_text().splitlines()]


def get_source_code_by_id(file_id: int) -> list[list[str]] | None:
    file_path = id_to_path(file_id)
    if file_path:
        return get_source_code(file_path)
    return None


def get_version_by_id(version_id: int) -> pd.Series:
    return versions.loc[version_id]


def get_kernel_by_id(kernel_id: int) -> pd.Series:
    return kernels.loc[kernel_id]

def get_first_n_files(path, n=10):
    found_files = []
    for dirname, _, filenames in os.walk(path):
        for filename in filenames:
            found_files.append(filename)
            if len(found_files) >= n:
                return found_files
    return found_files
```

### This is what the Meta Kaggle data looks like.

```python
for dirname, _, filenames in os.walk(MK_PATH):
    for filename in filenames:
        print(filename)
```

## This is what the Kernels.csv table looks like
 - Note: The term "Kernels" refers to "Kaggle Notebooks".

```python
kernels_file_name = f"{MK_PATH}/Kernels.csv"
kernels = pd.read_csv(kernels_file_name)
kernels.set_index("Id", inplace = True)
print("\nKernels Table:\n")
kernels.head()
```

## A given Kernel can have multiple Versions
 - A single KernelId can map to multiple KernelVersionIds

```python
versions_file_name = f"{MK_PATH}/KernelVersions.csv"
versions = pd.read_csv(versions_file_name)
versions.set_index("Id", inplace = True)
print("KernelVersions Table:\n")
versions.head()
```

## This is what the Meta Kaggle Code data looks like
 - Each file is labeled according to its KernelVersionId

```python
first_20_files = get_first_n_files(MKC_PATH, 20)

print("First 20 files:\n")
for filename in first_20_files:
   print(filename)
```

Let's look at KernelVersionId 1,000

```python
id = 1000
source = get_source_code_by_id(id)

file_path_for_id = id_to_path(id)
print(f"\nPath for ID {id}: {file_path_for_id}")
print(f"ID from path: {path_to_id(file_path_for_id)}")
print(f"File extension: {get_file_extension(file_path_for_id)}")
```

Looks like a Python script!

```python
print(f"\nSource code for ID {id} (first 10 lines of first cell/file):\n")
if source:
    if source and source[0]:
        for line in source[0][:10]:
            print(line)
        if len(source[0]) > 10:
            print("...")
    else:
        print("Source code is empty or malformed.")
else:
    print("Source code not found.")
```

Yep, it's a Python script!

## Now let's see what we can find in the Meta Kaggle dataset about that same Python script

```python
print(f"\nVersion by ID {id}:\n")
try:
    version_info = get_version_by_id(id)
    print(version_info)
    script_id = version_info['ScriptId']
    print(f"\nKernel by ScriptId {script_id}:\n")
    print(get_kernel_by_id(script_id))
except KeyError:
    print(f"Error: Version or Kernel with ID {id} not found.")
```

Now click on the "copy and edit" button in the top-right corner of this page, and try to come up with some insights about the Kaggle community and their code! For more detail about how the challenge will be evaluated, see https://www.kaggle.com/competitions/meta-kaggle-hackathon/overview/evaluation.

## Credit
 - The method for joining the `Meta-Kaggle` data with the `Meta-Kaggle-Code` data was adapted from https://www.kaggle.com/code/herbison/meta-kaggle-code-joined-to-meta-kaggle