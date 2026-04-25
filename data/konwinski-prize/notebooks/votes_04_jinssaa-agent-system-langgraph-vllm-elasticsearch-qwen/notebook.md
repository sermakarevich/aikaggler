# Agent System-Langgraph,Vllm,Elasticsearch,Qwen

- **Author:** JIN
- **Votes:** 150
- **Ref:** jinssaa/agent-system-langgraph-vllm-elasticsearch-qwen
- **URL:** https://www.kaggle.com/code/jinssaa/agent-system-langgraph-vllm-elasticsearch-qwen
- **Last run:** 2025-01-06 08:03:08.060000

---

```python
!pip install --target=/kaggle/working /kaggle/input/konwinski-prize/kprize_setup/pip_packages/*.whl -q \
    --no-deps \
    --no-index
!pip install --target=/kaggle/working /kaggle/input/konwinski-prize/kaggle_evaluation/../kprize_setup/kprize-1.0.0-py3-none-any.whl -q \
    --no-index \
    --no-deps
```

```python
import io

import os
import sys
import shutil
import pandas as pd
import polars as pl

sys.path.insert(0, "/kaggle/input/konwinski-prize/kprize_setup")
import kaggle_evaluation.konwinski_prize_inference_server
```

## Launch Vllm Serve

This script serves the LLM model using `vllm` in background mode. The script launches a subprocess that executes the `vllm.scripts.serve` module with specified parameters such as tensor parallelism, RoPE scaling, and auto tool selection. The output and errors are logged into a file for monitoring. 

Key features include:
- Tensor parallel size set to 4 GPUs(For L4)
- GPU memory utilization set to 99%
- [RoPE scaling to support a maximum context length of 131,072 tokens](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct#processing-long-texts)
- Prefix caching and eager execution for optimized model performance
  - `enable_prefix_caching` will save slice inference time but don't know about accuray loss yet.
- Auto tool choice
  - [Vllm Tool Calling](https://docs.vllm.ai/en/latest/usage/tool_calling.html)
  - [Qwen Function Calling](https://qwen.readthedocs.io/en/latest/framework/function_call.html)

**Make sure you are using Settings > Accelerator > GPU L4 x 4**

```python
import subprocess
log_file = open("vllm_output.log", "w")
model_path = "/kaggle/input/qwen2.5/transformers/14b-instruct/1"

# background task
command = [
    "python", 
    "-m", 
    "vllm.scripts", 
    "serve",
    model_path,
    "--tensor_parallel_size", "4",
    "--gpu_memory_utilization", "0.99",
    "--enforce_eager",
    "--enable-auto-tool-choice",
    "--tool-call-parser", "hermes",
    "--enable_prefix_caching",
    "--rope-scaling", '{"factor": 4.0, "original_max_position_embeddings": 32768, "type": "yarn"}' ## 131,072 context length
]

process = subprocess.Popen(command, stdout=log_file, stderr=log_file)

print(f"Background process started with PID: {process.pid}")
```

## Launch Elasticsearch

This script sets up and launches Elasticsearch 8.17.0 in a background process within a Kaggle environment. It ensures the necessary configurations and directories are in place before running the Elasticsearch instance.

Key steps:
1. **Copy Elasticsearch directory:**  
   Copies the Elasticsearch 8.17.0 directory from the input location to the working directory.
2. **Create necessary directories:**  
   Creates `logs` and `config` directories within the Elasticsearch folder to store logs and configuration files.
3. **Write `elasticsearch.yml`:**  
   Uses `%%writefile` to create the `elasticsearch.yml` configuration file in the `config` directory.
4. **Run Elasticsearch in the background:**  
   Launches the Elasticsearch executable with the `jupyter` user using a subprocess. Both standard output and error output are redirected to `elasticsearch_output.log`.
5. **Print PID:**  
   After the background process starts, the process ID (PID) is printed to confirm successful execution.

Reference:
- https://www.kaggle.com/code/linshokaku/4th-elasticsearch-retrieval-example#launch-elasticsearch

```python
!cp -r /kaggle/input/elasticsearch-8-17-0 /kaggle/working/
!mkdir -p /kaggle/working/elasticsearch-8-17-0/elasticsearch-8.17.0/logs
!mkdir -p /kaggle/working/elasticsearch-8-17-0/elasticsearch-8.17.0/config
```

```python
%%writefile /kaggle/working/elasticsearch-8-17-0/elasticsearch-8.17.0/config/elasticsearch.yml
# ======================== Elasticsearch Configuration =========================
#
# NOTE: Elasticsearch comes with reasonable defaults for most settings.
#       Before you set out to tweak and tune the configuration, make sure you
#       understand what are you trying to accomplish and the consequences.
#
# The primary way of configuring a node is via this file. This template lists
# the most important settings you may want to configure for a production cluster.
#
# Please consult the documentation for further information on configuration options:
# https://www.elastic.co/guide/en/elasticsearch/reference/index.html
#
# ---------------------------------- Cluster -----------------------------------
#
# Use a descriptive name for your cluster:
#
cluster.name: single-node-cluster
#
# ------------------------------------ Node ------------------------------------
#
# Use a descriptive name for the node:
#
node.name: single-node
#
# Add custom attributes to the node:
#
#node.attr.rack: r1
#
# ----------------------------------- Paths ------------------------------------
#
# Path to directory where to store the data (separate multiple locations by comma):
#
#path.data: /path/to/data
#
# Path to log files:
#
#path.logs: /path/to/logs
#
# ----------------------------------- Memory -----------------------------------
#
# Lock the memory on startup:
#
#bootstrap.memory_lock: true
#
# Make sure that the heap size is set to about half the memory available
# on the system and that the owner of the process is allowed to use this
# limit.
#
# Elasticsearch performs poorly when the system is swapping the memory.
#
# ---------------------------------- Network -----------------------------------
#
# By default Elasticsearch is only accessible on localhost. Set a different
# address here to expose this node on the network:
#
# network.host: 192.168.0.1
#
# By default Elasticsearch listens for HTTP traffic on the first free port it
# finds starting at 9200. Set a specific HTTP port here:
#
http.port: 9200
#
# For more information, consult the network module documentation.
#
# --------------------------------- Discovery ----------------------------------
#
# Pass an initial list of hosts to perform discovery when this node is started:
# The default list of hosts is ["127.0.0.1", "[::1]"]
#
discovery.type: single-node
#
# Bootstrap the cluster using an initial set of master-eligible nodes:
#
#cluster.initial_master_nodes: ["node-1", "node-2"]
#
# For more information, consult the discovery and cluster formation module documentation.
#
# ---------------------------------- Various -----------------------------------
#
# Allow wildcard deletion of indices:
#
#action.destructive_requires_name: false

#----------------------- BEGIN SECURITY AUTO CONFIGURATION -----------------------
#
# The following settings, TLS certificates, and keys have been automatically      
# generated to configure Elasticsearch security features on 02-01-2025 08:26:22
#
# --------------------------------------------------------------------------------

# Enable security features
xpack.security.enabled: false

xpack.security.enrollment.enabled: false

# Enable encryption for HTTP API client connections, such as Kibana, Logstash, and Agents
xpack.security.http.ssl:
  enabled: false
  # keystore.path: certs/http.p12

# Enable encryption and mutual authentication between cluster nodes
xpack.security.transport.ssl:
  enabled: false
  # verification_mode: certificate
  # keystore.path: certs/transport.p12
  # truststore.path: certs/transport.p12
# Create a new cluster with the current node only
# Additional nodes can still join the cluster later
# cluster.initial_master_nodes: ["a847b5071daa"]

# Allow HTTP API connections from anywhere
# Connections are encrypted and require user authentication
http.host: 0.0.0.0

# Allow other nodes to join the cluster from anywhere
# Connections are encrypted and mutually authenticated
#transport.host: 0.0.0.0

#----------------------- END SECURITY AUTO CONFIGURATION -------------------------
```

```python
%%bash
useradd -m jupyter
chown jupyter:jupyter -R /kaggle/working/elasticsearch-8-17-0/elasticsearch-8.17.0
chmod -R +x /kaggle/working/elasticsearch-8-17-0/elasticsearch-8.17.0
```

```python
log_file = open("elasticsearch_output.log", "w")

command = [
    "su",
    "-",
    "jupyter",
    "-c",
    "/kaggle/working/elasticsearch-8-17-0/elasticsearch-8.17.0/bin/elasticsearch"
]

process = subprocess.Popen(command, stdout=log_file, stderr=log_file)

print(f"Background process started with PID: {process.pid}")
```

## Codebase Syntax Tree Parsing and Flattening for LLM Tool and Elasticsearch Database

This script processes a directory of Python files by parsing their source code using the `Tree-sitter` library and converting the syntax trees into a flattened structure for analysis. The output is designed for use by an LLM as a tool to understand the structure of the codebase and for storing in Elasticsearch to support advanced search and query capabilities for code-related tasks.

### Key Functions:
1. **`traverse_tree(node, source_code)`**  
   - Recursively traverses the syntax tree, converting each node to a dictionary with details such as type, start and end points, and text content.
   - Captures child nodes for hierarchical representation.

2. **`parse_python_code(source_code)`**  
   - Parses the source code using the `Tree-sitter` parser and returns a nested structure representing the syntax tree.

3. **`find_python_files(base_dir)`**  
   - Scans a given directory recursively to find all Python files (`.py`) and returns their file paths.

4. **`process_codebase(base_dir)`**  
   - Reads and parses all Python files in the directory and generates syntax trees for each file.
   - Returns a dictionary where the keys are file paths and the values are parsed syntax trees.

5. **`point_to_dict(point)`**  
   - Converts a `Point` object (representing line and column positions) into a dictionary format for easier serialization.

6. **`flatten_module_data(data)`**  
   - Flattens the hierarchical syntax tree into a list of dictionaries for database insertion.
   - Filters the nodes to include only `function_definition` and `class_definition` types.
   - Assigns unique IDs (`UUID`) to each node and tracks parent-child relationships to maintain structure in a flattened format.
   - Processes the children recursively to ensure nested definitions (like inner classes or methods) are included.

## Purpose:
- **LLM Tool Integration:**  
  Provides structured information to LLMs for understanding Python codebases, aiding in:
  - Code documentation generation.
  - Refactoring assistance by analyzing relationships between functions and classes.
  - Identifying unused code or undocumented functions.
  
- **Elasticsearch Database Usage:**  
  The flattened data structure is designed to be indexed in an Elasticsearch database to enable:
  - **Full-text search**: Search across Python functions, classes, and modules using keywords or phrases.
  - **Hierarchical queries**: Retrieve specific definitions based on relationships (e.g., find all functions within a class).
  - **Filtering and aggregation**: Perform queries to filter by file path, type (function or class), or code snippet content.

## Elasticsearch Data Model:
Each flattened data entry is structured as a document for Elasticsearch with the following fields:
- `id`: Unique identifier (UUID).
- `parent_id`: Reference to the parent node ID (if applicable).
- `file_path`: The file path of the source code.
- `type`: The type of the code block (`function_definition`, `class_definition`).
- `text`: The source code snippet for the function or class definition.
- `start_point`: Start position (row, column) of the code block.
- `end_point`: End position (row, column) of the code block.

Reference
- https://ai.globant.com/wp-content/uploads/2024/11/Whitepaper-Globant-Code-Fixer-Agent.pdf

```python
from langchain_core.tools import tool
import json
import uuid
import tree_sitter_python as tspython
from tree_sitter import Language, Parser
PY_LANGUAGE = Language(tspython.language())
parser = Parser(PY_LANGUAGE)

# Function to traverse the tree
def traverse_tree(node, source_code):
    """Recursively traverse the syntax tree."""
    result = {
        "type": node.type,
        "start_point": node.start_point,
        "end_point": node.end_point,
        "text": source_code[node.start_byte:node.end_byte].decode("utf-8"),
        "children": []
    }
    if node.child_count > 0:
        for child in node.children:
            result["children"].append(traverse_tree(child, source_code))
    return result

# Parse Python source code
def parse_python_code(source_code):
    tree = parser.parse(source_code)
    root_node = tree.root_node
    return traverse_tree(root_node, source_code)

# Find all Python files in a directory
def find_python_files(base_dir):
    python_files = []
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".py"):
                python_files.append(os.path.join(root, file))
    return python_files

# Process a directory of Python files
def process_codebase(base_dir):
    results = {}
    python_files = find_python_files(base_dir)
    for file_path in python_files:
        with open(file_path, 'rb') as file:
            source_code = file.read()
            results[file_path] = parse_python_code(source_code)
    return results

def point_to_dict(point):
    """
    Convert a Point object to a dictionary
    """
    return {"row": point.row, "column": point.column}
    

def flatten_module_data(data):
    """
    Convert hierarchical data into a flattened structure, including the top-level file path.
    :param data: Hierarchical data (dict)
    :return: List of flattened data
    """
    flattened = []

    for module_path, module_data in data.items():
        # Convert top-level module data to a flat structure
        module_id = str(uuid.uuid4())
        
        # Filter types: only process "function_definition" or "class_definition"
        if module_data.get("type") not in ["module", "function_definition", "class_definition"]:
            continue

        flattened.append({
            "id": module_id,
            "parent_id": None,  # Top-level has no parent
            "file_path": module_path,
            "type": module_data.get("type"),
            "text": module_data.get("text"),
            "start_point": point_to_dict(module_data.get("start_point")),
            "end_point": point_to_dict(module_data.get("end_point"))
        })

        # Process the children data
        def process_children(children, current_parent_id):
            for child in children:
                # Filter types: only process "function_definition" or "class_definition"
                if child.get("type") not in ["module", "function_definition", "class_definition"]:
                    continue

                child_id = str(uuid.uuid4())
                flattened.append({
                    "id": child_id,
                    "parent_id": current_parent_id,  # Reference to the immediate parent ID
                    "file_path": module_path,  # Retain the top-level module path
                    "type": child.get("type"),
                    "text": child.get("text"),
                    "start_point": point_to_dict(child.get("start_point")),
                    "end_point": point_to_dict(child.get("end_point"))
                })
                process_children(child.get("children", []), child_id)  # Pass the current ID as the next parent ID
        
        process_children(module_data.get("children", []), module_id)

    return flattened
```

## Agent Tool Definition: open_file, edit_file, list_folder, search_code_elements

This script defines agent tools for interacting with the filesystem and performing codebase searches. It integrates functionalities for reading, editing, and listing file contents and provides Elasticsearch-based search capabilities for analyzing Python code elements. Each tool is implemented using `pydantic` for request validation and error handling.

## Tool Definitions:

### 1. `open_file` Tool
**Purpose:**  
Opens a file and returns the content within a specified line range for viewing.  

**Key Features:**
- Supports reading from the beginning or a specified line number.
- Handles errors such as file not found, permission issues, and invalid line ranges.
  
**Input Fields:**
- `file_path`: Path to the file to open.
- `start_line` (optional): Line number to start reading from.
- `end_line` (optional): Line number to stop reading.

**Output Fields:**
- `content`: The read content from the file.
- `error`: Error message, if any.

---

### 2. `edit_file` Tool
**Purpose:**  
Edits the content of a file by replacing or inserting lines within a specified range.

**Key Features:**
- Supports both line replacement and insertion.
- Validates syntax for Python files after editing to avoid introducing errors.

**Input Fields:**
- `file_path`: Path to the file to edit.
- `text`: The new text to be added or replace existing lines.
- `start_line`: Line to start editing (inclusive).
- `end_line` (optional): Line to end editing (inclusive).

**Output Fields:**
- `old_text`: The original content that was replaced.
- `updated_text`: The newly updated content.
- `error`: Error message, if any.

---

### 3. `list_folder` Tool
**Purpose:**  
Lists all files and folders within a specified directory.

**Key Features:**
- Recursively lists folder contents, excluding `.git` directories.
- Provides separate lists for files and folders.

**Input Fields:**
- `folder_path`: Path to the folder to list.

**Output Fields:**
- `files`: List of file paths.
- `folders`: List of folder paths.
- `error`: Error message, if any.

---

### 4. `search_code_elements` Tool
**Purpose:**  
Performs a search for specific code elements (e.g., functions, classes) in an Elasticsearch index using BM25-based relevance scoring.

**Key Features:**
- Supports searching by `element_type` (e.g., function, class).
- Enables keyword-based searches on the `text` content.
- Designed to work with Python code structure indexed in Elasticsearch.

**Input Fields:**
- `index_name`: Elasticsearch index to search.
- `element_type` (optional): Type of the code element (`function_definition`, `class_definition`, etc.).
- `keyword` (optional): Keyword for text-based search.

**Output Fields:**
- List of search results containing:
  - `id`: Unique ID of the document.
  - `file_path`: Path to the file containing the element.
  - `type`: Type of the code element.
  - `text`: Content snippet of the code element.
  - `start_point`, `end_point`: Start and end positions of the code element.
  - `score`: BM25 score for relevance.
- Returns a maximum of the top 4 results.

---

### Utility Functions:
- **`index_data(flattened_data, index_name)`**:  
  Indexes flattened Python code data into Elasticsearch for searchability.

```python
from typing import Dict, Optional, List
from pydantic import BaseModel, Field

class EditFileRequest(BaseModel):
    """Request to edit a file."""
    file_path: Optional[str] = Field(
        default=None,
        description=(
            "The path to the file that will be edited. If not provided, "
            "THE CURRENTLY OPEN FILE will be edited. If provided, the "
            "file at the provided path will be OPENED and edited, changing "
            "the opened file."
        ),
    )
    text: str = Field(
        ..., 
        description="The text that will replace the specified line range in the file.",
    )
    start_line: int = Field(
        ..., 
        description=(
            "The line number at which the file edit will start (REQUIRED). "
            "Inclusive - the start line will be included in the edit. "
            "If you just want to add code and not replace any line, "
            "don't provide end_line field."
        ),
    )
    end_line: Optional[int] = Field(
        default=None,
        description=(
            "The line number at which the file edit will end (REQUIRED). "
            "Inclusive - the end line will be included in the edit. "
            "If you just want to add code and not replace any line, "
            "don't provide this field."
        ),
    )

class EditFileResponse(BaseModel):
    """Response to edit a file."""
    old_text: Optional[str] = Field(
        default=None,
        description=(
            "The updated changes. If the file was not edited, the original file "
            "will be returned."
        ),
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message if any",
    )
    updated_text: Optional[str] = Field(
        default=None,
        description="The updated text. If the file was not edited, this will be empty.",
    )

class OpenFileRequest(BaseModel):
    """Request to open a file."""
    file_path: str = Field(
        ..., 
        description="The path to the file that will be opened.",
    )
    start_line: int = Field(
        default=None,
        description=(
            "The line number at which the file content will start to be read. "
            "If not provided, the file will be read from the beginning."
        ),
    )
    end_line: Optional[int] = Field(
        default=None,
        description=(
            "The line number at which the file content will stop being read. "
            "If not provided, the file will be read until the end."
        ),
    )

class OpenFileResponse(BaseModel):
    """Response for opening a file."""
    content: Optional[str] = Field(
        default=None,
        description="The content of the opened file.",
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message if any.",
    )

@tool(args_schema=OpenFileRequest)
def open_file(**kwargs) -> OpenFileResponse:
    """
    Opens a file in the editor based on the provided file path,
    If start_lien or end_line are provided, the window will be moved after that line. (i.e. 100 lines after the line number will be displayed)

    Can result in:
    - ValueError: If file_path is not a string or if the file does not exist.
    - FileNotFoundError: If the file does not exist.
    - IOError: If there's an issue reading the file.
    - PermissionError: If the user doesn't have permission to read the file.
    - IsADirectoryError: If the provided path is a directory.
    """    
    request = OpenFileRequest(**kwargs)

    try:
        with open(request.file_path, "r") as file:
            lines = file.readlines()

        start_line = request.start_line if request.start_line else 1
        end_line = request.end_line if request.end_line else len(lines)

        if start_line < 1 or end_line > len(lines):
            return OpenFileResponse(error="Invalid line range.")

        content = "".join(lines[start_line - 1:end_line])
        return OpenFileResponse(content=content)

    except FileNotFoundError:
        return OpenFileResponse(error="File not found.")
    except PermissionError:
        return OpenFileResponse(error="Permission denied.")
    except OSError as e:
        return OpenFileResponse(error=f"OS error occurred: {str(e)}")

@tool(args_schema=EditFileRequest)
def edit_file(**kwargs) -> EditFileResponse:
    """
    Use this tools to edit a file on specific line numbers.

    Please note that THE EDIT COMMAND REQUIRES PROPER INDENTATION.

    Python files will be checked for syntax errors after the edit.
    If you'd like to add the line '        print(x)' you must fully write
    that out, with all those spaces before the code!

    If a syntax error is detected, the edit won't be executed. Review the error
    message and modify your edit command accordingly.

    When start and end lines are the same, the new text is inserted at that line,
    preserving the original line's content.

    Ex A: Start=End=1, Text: "print(x)"
    Result: Adds "print(x)" as first line, rest unchanged.

    Ex B: Start=1, End=3, Text: "print(x)"
    Result: Replaces lines 1,2 and 3 with "print(x)", rest unchanged.

    This action edits a specific part of the file, if you want to rewrite the
    complete file, use `write` tool instead."""    
    request = EditFileRequest(**kwargs)

    try:
        if request.file_path is None:
            return EditFileResponse(error="No file path provided.")

        with open(request.file_path, "r") as file:
            lines = file.readlines()

        # Adjust end_line if not provided
        end_line = request.end_line if request.end_line is not None else len(lines)

        if request.start_line < 1 or end_line > len(lines):
            return EditFileResponse(error="Invalid line range.")

        # Capture the old text
        old_text = "".join(lines[request.start_line - 1:end_line])

        # Replace the specified lines
        new_lines = lines[:request.start_line - 1] + [request.text + "\n"] + lines[end_line:]

        # Write back to the file
        with open(request.file_path, "w") as file:
            file.writelines(new_lines)

        return EditFileResponse(old_text=old_text, updated_text=request.text)

    except FileNotFoundError:
        return EditFileResponse(error="File not found.")
    except PermissionError:
        return EditFileResponse(error="Permission denied.")
    except OSError as e:
        return EditFileResponse(error=f"OS error occurred: {str(e)}")

class ListFolderRequest(BaseModel):
    """Request to list the contents of a folder."""
    folder_path: str = Field(
        ..., 
        description="The path to the folder whose contents will be listed."
    )

class ListFolderResponse(BaseModel):
    """Response for listing the contents of a folder."""
    files: Optional[List[str]] = Field(
        default=None,
        description="List of file names in the folder."
    )
    folders: Optional[List[str]] = Field(
        default=None,
        description="List of folder names in the folder."
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message if any."
    )

@tool(args_schema=ListFolderRequest)
def list_folder(**kwargs) -> ListFolderResponse:
    """
    Recursively lists the contents of a folder at the provided path, excluding .git folders.

    Can result in:
    - ValueError: If folder_path is not a string.
    - FileNotFoundError: If the folder does not exist.
    - NotADirectoryError: If the path is not a directory.
    - PermissionError: If the user doesn't have permission to access the folder.
    """
    request = ListFolderRequest(**kwargs)

    try:
        if not os.path.isdir(request.folder_path):
            return ListFolderResponse(error="The provided path is not a directory.")

        # Recursively list all files and folders, excluding .git directories
        all_files = []
        all_folders = []
        for root, dirs, files in os.walk(request.folder_path):
            # Exclude .git folder
            dirs[:] = [d for d in dirs if d != '.git']
            for file in files:
                all_files.append(os.path.join(root, file))
            for folder in dirs:
                all_folders.append(os.path.join(root, folder))

        return ListFolderResponse(files=all_files, folders=all_folders)

    except FileNotFoundError:
        return ListFolderResponse(error="Folder not found.")
    except PermissionError:
        return ListFolderResponse(error="Permission denied.")
    except OSError as e:
        return ListFolderResponse(error=f"OS error occurred: {str(e)}")
```

```python
from pydantic import BaseModel, Field, model_validator
from typing import Optional, Dict
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

class SearchCodeElementsParams(BaseModel):
    """
    Model for validating input parameters for the search_code_elements function.
    """
    element_type: Optional[str] = Field(
        None, description="Type of the code element to search for (e.g., 'function_definition', 'class_definition', 'decorated_definition')."
    )
    keyword: Optional[str] = Field(
        None, description="Keyword to search for in the text field. Uses BM25-based matching."
    )
    # file_path: Optional[str] = Field(
    #     None, description="Specific module path to filter the search results."
    # )
    # parent_id: Optional[str] = Field(
    #     None, description="Parent ID to filter related data."
    # )
    # start_point: Optional[Dict[str, int]] = Field(
    #     None, description="Start position filter in the format {'row': int, 'column': int}."
    # )
    # end_point: Optional[Dict[str, int]] = Field(
    #     None, description="End position filter in the format {'row': int, 'column': int}."
    # )
    index_name: str = Field(
        default=None, description="Elasticsearch index name to query from."
    )


    @model_validator(mode="before")
    def validate_element_type(cls, values):
        """
        Validates that element_type is one of the allowed Tree-sitter types.
        """
        valid_types = {
            "function_definition", "class_definition", "decorated_definition"}
        
        element_type = values.get("element_type")

        if element_type and element_type not in valid_types:
            raise ValueError(f"element_type must be one of {valid_types}. Provided: {element_type}")

        return values    

@tool(args_schema=SearchCodeElementsParams)
def search_code_elements(**kwargs):
    """
    Searches for code elements (e.g., class, function, parameter) in an Elasticsearch index based on BM25.

    Filters can be applied for start_point and end_point ranges, along with other parameters.

    :param kwargs: Dictionary of search parameters. Must match the fields in SearchCodeElementsParams.
    :return: List of search results.
    """
    # Validate and parse parameters using Pydantic
    params = SearchCodeElementsParams(**kwargs)

    # Base search query
    query = {
        "bool": {
            "must": [],  # BM25-based search
            "filter": []  # Exact match filters
        }
    }

    # Add filters for element type
    if params.element_type:
        query["bool"]["filter"].append({
            "term": {
                "type": params.element_type
            }
        })

    # Add BM25-based keyword search
    if params.keyword:
        query["bool"]["must"].append({
            "match": {
                "text": params.keyword
            }
        })

    # Add filter for module path
    # if params.file_path:
    #     query["bool"]["filter"].append({
    #         "term": {
    #             "file_path": params.file_path
    #         }
    #     })

    # Add filter for parent ID
    # if params.parent_id:
    #     query["bool"]["filter"].append({
    #         "term": {
    #             "parent_id": params.parent_id
    #         }
    #     })

    # Add range filter for start_point
    # if params.start_point:
    #     query["bool"]["filter"].append({
    #         "range": {
    #             "start_point.row": {
    #                 "gte": params.start_point.get("row", 0),
    #                 "lte": float("inf")
    #             }
    #         }
    #     })
    #     query["bool"]["filter"].append({
    #         "range": {
    #             "start_point.column": {
    #                 "gte": params.start_point.get("column", 0),
    #                 "lte": float("inf")
    #             }
    #         }
    #     })

    # # Add range filter for end_point
    # if params.end_point:
    #     query["bool"]["filter"].append({
    #         "range": {
    #             "end_point.row": {
    #                 "gte": params.end_point.get("row", 0),
    #                 "lte": float("inf")
    #             }
    #         }
    #     })
    #     query["bool"]["filter"].append({
    #         "range": {
    #             "end_point.column": {
    #                 "gte": params.end_point.get("column", 0),
    #                 "lte": float("inf")
    #             }
    #         }
    #     })

    # Perform Elasticsearch search
    response = es.search(index=params.index_name, body={"query": query})

    # Extract and format search results
    results = [
        {
            "id": hit["_id"],
            "file_path": hit["_source"]["file_path"],
            "type": hit["_source"]["type"],
            "text": hit["_source"]["text"],
            "start_point": hit["_source"]["start_point"],
            "end_point": hit["_source"]["end_point"],
            "parent_id": hit["_source"]["parent_id"],
            "score": hit["_score"]  # BM25 score
        }
        for hit in response["hits"]["hits"][:4] # top-2
    ]
    return results

def index_data(flattened_data, index_name):
    actions = [
        {
            "_index": index_name,
            "_id": doc["id"],
            "_source": doc
        }
        for doc in flattened_data
    ]
    bulk(es, actions)
```

## Define Agent

This script defines an agent using a workflow that integrates LLM-based reasoning and tool usage. The agent is configured to utilize a set of predefined tools for interacting with codebases and making decisions based on user queries and responses. The workflow utilizes `StateGraph` to manage the flow between the agent and the tools, allowing dynamic handling of tool invocations during interactions.

### Key Components:

#### 1. **`get_model()` Function**
**Purpose:**  
Initializes and returns an application workflow (`app`) that uses a language model capable of invoking tools during conversation.

**Steps:**
1. **Environment Setup:**
   - Sets the `OPENAI_API_KEY` for authentication.
   - Lists the available tools (`search_code_elements`, `open_file`, `edit_file`, `list_folder`).

2. **Tool Node Creation:**
   - Creates a `ToolNode` with the specified tools.
   - Binds the tools to the language model (`ChatOpenAI`), allowing the model to make calls to these tools.

3. **Model Response Logic:**
   - **`should_continue(state: MessagesState)`**: Determines if the agent should continue tool execution or finish the conversation.
     - If the last message contains tool calls, transitions to `"tools"`.
     - Otherwise, ends the conversation (`END`).
   
   - **`call_model(state: MessagesState)`**: Sends the message state to the model and returns the response message from the model.

4. **Workflow Definition:**
   - Creates a state graph (`StateGraph`) to manage the agent's conversation flow.
   - Adds nodes:
     - `"agent"`: Handles LLM responses.
     - `"tools"`: Handles tool executions.
   - Defines transitions:
     - Starts at `"agent"`.
     - Cycles between `"agent"` and `"tools"` based on the response.
     - Ends when no further tool calls are needed.

5. **Application Compilation:**
   - The workflow is compiled into an executable `app` that can handle incoming requests and manage interactions.

---

## Tools Integrated:
- **`search_code_elements`:** Searches Python code in Elasticsearch for specific elements (e.g., classes, functions).
- **`open_file`:** Opens a file and reads content within a line range.
- **`edit_file`:** Edits a specific range of lines in a file.
- **`list_folder`:** Lists the contents of a folder (files and subfolders).

---

## Workflow Flow:
1. **Start:**  
   The agent receives the user query.
2. **Agent Decision:**  
   The agent decides if tool usage is required.
3. **Tool Invocation:**  
   If a tool is needed, the workflow invokes the corresponding tool.
4. **Response:**  
   The tool response is processed, and the agent generates a reply.
5. **End:**  
   The conversation ends if no further tool usage is necessary.

## Reference
- https://langchain-ai.github.io/langgraph/how-tos/tool-calling-errors/#using-the-prebuilt-toolnode

```python
import os
from typing import Literal

from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI

def get_model():
    os.environ["OPENAI_API_KEY"] = "api_key"
    tools = [search_code_elements, open_file, edit_file, list_folder]
    tool_node = ToolNode(tools)
    model_with_tools = ChatOpenAI(base_url="http://localhost:8000/v1",model=model_path).bind_tools(tools, temperature=0)
       
    def should_continue(state: MessagesState):
        messages = state["messages"]
        last_message = messages[-1]
        if last_message.tool_calls:
            print(last_message.tool_calls, "*"*100)
            return "tools"
        return END
    
    
    def call_model(state: MessagesState):
        messages = state["messages"]
        response = model_with_tools.invoke(messages)
        return {"messages": [response]}
    
    
    workflow = StateGraph(MessagesState)
    
    # Define the two nodes we will cycle between
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", tool_node)
    
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", should_continue, ["tools", END])
    workflow.add_edge("tools", "agent")
    
    app = workflow.compile()
    return app
```

```python
GITHUB_ISSUE_SOLVER_PROMPT = """
You are an autonomous software engineer tasked with solving coding issues efficiently and concisely. Your primary role is to coordinate between code analysis and editing tasks. Follow these streamlined guidelines:

You have access to the following tools:
- `search_code_elements`: Get information about a specific class or function, including start and end lines.
- `list_folder`: View the repository structure.
- `open_file`: Open and view file contents, ideally focusing on a specific range based on `search_code_elements` results.
- `edit_file`: Make changes to the code.

The task involves working within the provided **Codebase** and **Codebase Folder List** to modify relevant modules and solve the given issue. Focus only on the necessary steps to avoid unnecessary complexity.

### Instructions:
1. **Identify the Relevant Module**:
   - Quickly scan the **Codebase Folder List** and determine the likely location of the issue.
   - Form a hypothesis about the parts of the code that require investigation based on folder names.

2. **Understand the Issue**:
   - Review the given issue or bug report concisely.
   - Formulate a hypothesis about the root cause and a potential solution.
   - Minimize analysis noise and focus directly on solving the issue.

3. **Explore the Codebase**:
   - Use `list_folder` to confirm the current folder structure.
   - Use `search_code_elements` to locate relevant code elements (e.g., functions, classes) and determine their start and end lines.
   - Use `open_file` to view specific sections of the code, focusing on the range provided by `search_code_elements`.

4. **Code Analysis and Editing**:
   - Use `open_file` to locate the target code section based on the results from `search_code_elements` and review its contents.
   - Use `edit_file` to make precise, minimal changes that address the issue.
   - Ensure that your edits preserve existing functionality and syntax.
   - Pay close attention to line numbers, indentation, and syntax.

5. **Problem-Solving Approach**:
   - Break down complex problems into smaller tasks as needed but avoid verbose explanations.
   - Continuously monitor progress and adapt only if required.

6. **Completion**:
   - When the issue has been fixed, respond with "PATCH COMPLETED".
   - Only respond with "PATCH COMPLETED" if you are confident the issue is resolved.

7. **Example**:
**Problem Statmenet**:
TypeError: unsupported format string passed to NoneType.__format__
Regression in #2459

### Steps to reproduce
a.py:
py
class A:
    def __init__(self):
        self._magnitude = None

    def name(self) -> str | None:
        if self._magnitude:
            return f"M {self._magnitude:.1f}"

pylint a.py

### Current behavior
File "/Users/jwalls/release/lib/python3.12/site-packages/astroid/nodes/node_classes.py", line 4778, in _infer_from_values
    yield from nodes[0]._infer(context, **kwargs)
  File "/Users/jwalls/release/lib/python3.12/site-packages/astroid/nodes/node_classes.py", line 4695, in _infer
    formatted = format(value.value, format_spec.value)
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
TypeError: unsupported format string passed to NoneType.__format__

**Codebase Folder List**
['repo/tests',
 'repo/doc',
 'repo/script',
 'repo/.github',
 'repo/astroid',
 'repo/tests/testdata',
 'repo/tests/brain', ...]

**Approach Steps**:
a. Understanding I need to solve the problem about `astroid` project.
b. Use `search_code_elements` to search for `_infer_from_values` and `_infer`, obtaining their start and end lines.
c. Use `open_file` to open and analyze `_infer_from_values` or `_infer`, focusing on the range identified by `search_code_elements`.
d. Use `edit_file` to make the necessary changes to `_infer_from_values` or `_infer`.
"""
```

```python
# instance_count = None

def get_number_of_instances(num_instances: int) -> None:
    """ The very first message from the gateway will be the total number of instances to be served.
    You don't need to edit this function.
    """
    global instance_count
    instance_count = num_instances
```

```python
import unidiff


def is_valid_patch_format(patch: str) -> bool:
    try:
        patch_set = unidiff.PatchSet(patch)
        if len(patch_set) == 0:
            return False
    except Exception:
        return False
    
    return True
```

```python
host = "http://localhost:9200"  # Elasticsearch

# Elasticsearch Client
es = Elasticsearch(
    hosts=[host],
)
```

## Define Agent: GitHub Issue Solver with Elasticsearch and Codebase Management

This script defines an agent that processes GitHub issues by analyzing the provided codebase, indexing its structure in Elasticsearch, and using LLM-assisted tools to suggest and apply changes. The agent performs health checks, manages the codebase as a Git repository, and generates diffs for code patches.

### Key Components:

#### 1. **`predict(problem_statement: str, repo_archive: io.BytesIO) -> str`**
**Purpose:**  
Processes the given problem statement and codebase archive, performs analysis, and generates a suggested code patch.

---

#### **Main Steps:**

1. **Health Checks:**
   - Waits for the external service (LLM endpoint) and Elasticsearch to be available. it will called only first_prediction is `true`

2. **Codebase Preparation:**
   - Unpacks the `.tar` archive containing the codebase.
   - Initializes the directory as a Git repository for version control.
   - Makes an initial commit with all files.

3. **Elasticsearch Indexing:**
   - Creates a unique Elasticsearch index to store the flattened codebase structure.
   - Mappings include fields like `id`, `type`, `file_path`, and `start_point` for structured search.
   - Indexes data using `process_codebase` and `flatten_module_data`.

4. **LLM Tool Integration:**
   - Calls `get_model()` to create an LLM application with integrated tools (`search_code_elements`, `open_file`, `edit_file`, `list_folder`).
   - Sends a message to the LLM app with:
     - **Problem Statement:** Description of the issue to be solved.
     - **Codebase Folder List:** List of directories in the unpacked codebase.
     - **Elasticsearch Index:** The index used for querying the codebase.

5. **Generating the Response:**
   - The LLM app processes the request and returns tool-generated responses.
   - Logs the response content for review.

6. **Committing Changes:**
   - Uses `git status` to detect any changes made by the LLM's response.
   - If changes exist, commits them and generates the `git diff` output.

7. **Validation:**
   - The function validates the generated patch format before returning the diff.

```python
import time
import uuid
import requests
first_prediction = True

def predict(problem_statement: str, repo_archive: io.BytesIO) -> str:
    """ Replace this function with your inference code.
    Args:
        problem_statement: The text of the git issue.
        repo_path: A BytesIO buffer path with a .tar containing the codebase that must be patched. The gateway will make this directory available immediately before this function runs.
    """
    try:        
        global first_prediction
        global is_submission
        global is_debug

        if not is_submission and not is_debug:
            return None
        
        if first_prediction:
            # Wait for external service health
            while True:
                try:
                    health_response = requests.get("http://localhost:8000/health")
                    if health_response.status_code == 200:
                        print("Health check for http://localhost:8000/health passed.")
                        break
                    print("Waiting for http://localhost:8000/health...")
                except requests.exceptions.RequestException as e:
                    print(f"Health check request failed: {e}. Retrying...")
                time.sleep(5)  # Retry every 5 seconds
    
            # Wait for Elasticsearch connection
            es = Elasticsearch(hosts=["http://localhost:9200"])
            while True:
                try:
                    if es.ping():
                        print("Elasticsearch is available.")
                        break
                    print("Waiting for Elasticsearch to become available...")
                except Exception as e:
                    print(f"Elasticsearch connection error: {e}. Retrying...")
                time.sleep(5)  # Retry every 5 seconds
    
            print("Health checks passed.")
            first_prediction = False
            
        # Unpack
        with open('repo_archive.tar', 'wb') as f:
            f.write(repo_archive.read())
        repo_path = 'repo'
        if os.path.exists(repo_path):
            shutil.rmtree(repo_path)
        shutil.unpack_archive('repo_archive.tar', extract_dir=repo_path)    
        
        os.remove('repo_archive.tar')
    
        # Initialize a Git repository
        # Ensure Git user identity is set
        subprocess.run(["git", "config", "--global", "user.email", "example@example.com"], check=True)
        subprocess.run(["git", "config", "--global", "user.name", "Example User"], check=True)    
        subprocess.run(["git", "init"], cwd=repo_path, check=True)
        subprocess.run(["git", "add", "-A"], cwd=repo_path, check=True)
        subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=repo_path, check=True)
        
        index_name = str(uuid.uuid4())
        ## elasticsearch settings    
        base_directory = repo_path  # Replace with your codebase path
        codebase_structure = process_codebase(base_directory)
    
        host = "http://localhost:9200"  
        
        # Elasticsearch Client
        es = Elasticsearch(
            hosts=[host],
        )
        
        es.indices.create(index=index_name, body={
            "mappings": {
                "properties": {
                    "id": { "type": "keyword" },
                    "parent_id": { "type": "keyword" },
                    "file_path": { "type": "keyword" },
                    "type": { "type": "keyword" },
                    "text": { "type": "text" },
                    "start_point": {
                        "type": "object",
                        "properties": {
                            "row": { "type": "integer" },
                            "column": { "type": "integer" }
                        }
                    },
                    "end_point": {
                        "type": "object",
                        "properties": {
                            "row": { "type": "integer" },
                            "column": { "type": "integer" }
                        }
                    }
                }
            }
        })
        flattened = flatten_module_data(codebase_structure)
        index_data(flattened, index_name)
    
        ##langgraph setting
        app = get_model()
        folder_list = list_folder.invoke(input={"folder_path": repo_path}).folders
        messages = [
        {"role": "system",
         "content": GITHUB_ISSUE_SOLVER_PROMPT},
         {"role":
        "user",
        "content":
        f"##Problem Statement: {problem_statement}\n\nCodebase Path: {repo_path}\n\nCodebase Folder List: {folder_list}\n\nElasticSearch Index: {index_name}"}]
    
        response = app.invoke(
        {"messages": messages}, {"recursion_limit": 10})
      
        
        for message in response["messages"]:
            string_representation = f"{message.type.upper()}: {message.content}\n"
            print(string_representation)
        es.indices.delete(index=index_name)
        # Instead of a valid diff, let's just submit a generic string. This will definitely fail.    
        first_prediction = False
        # Check for changes before adding and committing
        status_result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_path,
            check=True,
            stdout=subprocess.PIPE,
            text=True
        )
    
        if not status_result.stdout.strip():
            print("No changes to commit.")
            git_diff = None
        else:
            subprocess.run(["git", "add", "-A"], cwd=repo_path, check=True)
            subprocess.run(["git", "commit", "-m", "Apply changes"], cwd=repo_path, check=True)
    
            # Get git diff output
            git_diff = subprocess.check_output(["git", "diff", "HEAD~1"], cwd=repo_path, text=True)
            print(git_diff)
        del app

        if is_valid_patch_format(git_diff):        
            return git_diff
        else:
            return None
    except Exception as e:
        print(f"Exception occurred: {e}")
        return None
```

**When your notebook is run on the hidden test set, inference_server.serve must be called within 15 minutes of the notebook starting or the gateway will throw an error. If you need more than 15 minutes to load your model you can do so during the very first predict call, which does not have the usual 30 minute response deadline.**

## **Inference Server Integration**
The script integrates with `KPrizeInferenceServer` to handle competition mode and local testing.

- **`is_debug = True:`**  
  Predicts all training data to simulate the inference process.
- **`is_debug = False:`**  
  Returns `None` for all training data to reduce commit time.


**Ensure that for testing purposes, I set recursion_limit=10. It's take around 2hours for inference.**
---

```python
inference_server = kaggle_evaluation.konwinski_prize_inference_server.KPrizeInferenceServer(
    get_number_of_instances,   
    predict
)
is_debug = True
is_submission = os.getenv('KAGGLE_IS_COMPETITION_RERUN')
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

```python
!cat vllm_output.log
```

```python
!cat elasticsearch_output.log
```

```python
import os
import shutil

def delete_subdirectories(path):
    """
    Function to delete only the subdirectories in a given path.

    Parameters:
        path (str): Path containing subdirectories to be deleted.
    """
    if not os.path.exists(path):
        print(f"Path does not exist: {path}")
        return

    # Iterate over all items in the given path
    for item in os.listdir(path):
        full_path = os.path.join(path, item)

        # Delete if the item is a directory
        if os.path.isdir(full_path):
            shutil.rmtree(full_path)
            print(f"Directory deleted: {full_path}")
        else:
            print(f"Skipped (not a directory): {full_path}")

# Test path
target_path = "/kaggle/working"
delete_subdirectories(target_path)
```

```python
pd.read_csv("submission.csv")
```