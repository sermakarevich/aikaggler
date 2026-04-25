# gpt oss -Ollama- stater Notebook

- **Author:** Chandan Kumar Barada
- **Votes:** 57
- **Ref:** ckskaggle/gpt-oss-ollama-stater-notebook
- **URL:** https://www.kaggle.com/code/ckskaggle/gpt-oss-ollama-stater-notebook
- **Last run:** 2025-08-12 11:59:06.193000

---

# Running GPT Oss using Ollama

## Basic Setup

```python
import subprocess
import sys
import json
```

##  **Install `openai` using `subprocess`**

This command installs the `openai` Python package. It's particularly useful in environments like notebooks where you need to programmatically install a library.

```python
subprocess.check_call([sys.executable, "-m", "pip", "install", "openai"])
```

```python
import os
import time
from openai import OpenAI
```

##  Install Ollama using bash commands

```python
print("Installing Ollama...")
os.system("curl -fsSL https://ollama.com/install.sh | sh")
```

# Start Ollama server in the background

```python
print("Starting Ollama server...")
os.system("nohup ollama serve > /tmp/ollama_serve_stdout.log 2>/tmp/ollama_serve_stderr.log &")
```

## Checking Ollama Runnning status

To put it simply, the `ps aux | grep -E 'ollama' | grep -v grep || true` command checks for a running Ollama server process. It filters the list of all running processes to find anything related to "ollama" and then removes the `grep` command itself from the output to show a clean result.

```python
print("Checking if Ollama is running...")
os.system("ps aux | grep -E 'ollama' | grep -v grep || true")
```

# Download GPT-OSS:20B model (this will take significant time - ~13GB)

```python
%%timeit
os.system("ollama pull gpt-oss:20b")
```

## Verify the model is downloaded

```python
print("\nVerifying model installation...")
os.system("ollama list")
```

# Run the model

## Initialize OpenAI client for Ollama

```python
print("\nInitializing OpenAI client for Ollama...")
client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
```

## Testing the model

```python
try:
    response = client.chat.completions.create(
        model="gpt-oss:20b",
        messages=[
            {"role": "system", "content": "You are a sarcastic comedian."},
            {"role": "user", "content": "Give me a one-liner about the challenges of AI"}
        ]
    )
    print("Model Response:")
    print(response.choices[0].message.content)
    
    
except Exception as e:
    print(f"Error during first test: {e}")
```

```python
print("\n\nFull model responce in JSON format:\n")
response_dict = response.model_dump()
print(json.dumps(response_dict, indent =4))
```