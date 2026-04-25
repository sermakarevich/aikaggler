# [Konwinski Prize] Gemma LLM

- **Author:** Akhil
- **Votes:** 81
- **Ref:** akhiljethwa/konwinski-prize-gemma-llm
- **URL:** https://www.kaggle.com/code/akhiljethwa/konwinski-prize-gemma-llm
- **Last run:** 2024-12-20 04:58:29.060000

---

![](https://opengraph.githubassets.com/589594f788a4448dfe547f5999a901ce44b94d851559d036df16766e02a2ef62/swe-bench/SWE-bench)

# Data Overview

data/data.parquet The train set metadata, which includes a limited to a handful of examples. You are encouraged to source additional codebases for training your models. Most of the metadata provided here is only available for the train set.

- instance_id - A unique string identifier for the instance (aka GitHub issue).
- repo - The relevant GitHub repository. Also served by the evaluation API.
- problem_statement - Text describing the issue. Also served by the evaluation API.
- patch - Only provided for the train set. The patch resolving the issue.
- test_patch - Only provided for the train set. The patch resolving the issue.
- pull_number - The PR number of the pull request resolving the issue.
- base_commit - The commit used as the basis for the provided copy of the repo.
- issue_numbers - The original ID number of the issue.
- [PASS_TO_PASS/FAIL_TO_PASS] - Lists of the unit tests to run for this issue.

### Imports:

```python
import kaggle_evaluation.konwinski_prize_inference_server
import zipfile
import pandas as pd
from datasets import load_dataset
import os
import shutil
import io
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")
```

```python
rc = {
    "axes.facecolor": "#F8F8F8",
    "figure.facecolor": "#F8F8F8",
    "axes.edgecolor": "#000000",
    "grid.color": "#EBEBE7" + "30",
    "font.family": "serif",
    "axes.labelcolor": "#000000",
    "xtick.color": "#000000",
    "ytick.color": "#000000",
    "grid.alpha": 0.4,
}

sns.set(rc=rc)
palette = ['#302c36', '#037d97', '#E4591E', '#C09741',
           '#EC5B6D', '#90A6B1', '#6ca957', '#D8E3E2']

from colorama import Style, Fore
blk = Style.BRIGHT + Fore.BLACK
mgt = Style.BRIGHT + Fore.MAGENTA
red = Style.BRIGHT + Fore.RED
blu = Style.BRIGHT + Fore.BLUE
res = Style.RESET_ALL

plt.style.use('seaborn-v0_8-pastel')
```

## Competition Dataset

```python
import zipfile
konwinski= zipfile.ZipFile('../input/konwinski-prize/data.a_zip')
konwinski.extractall()
```

```python
zf = zipfile.ZipFile('../input/konwinski-prize/data.a_zip') 

train_data = pd.read_parquet("data/data.parquet")
```

```python
train_data.head(10)
```

## Origional Dataset:

```python
swebench = load_dataset('princeton-nlp/SWE-bench', split='test')
```

```python
swebench
```

```python
# Convert the dataset to a pandas DataFrame
swebench = swebench.to_pandas()
```

```python
swebench.head()
```

## Some Data Visualizations

```python
text = ' '.join(swebench['problem_statement'])
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
plt.figure(figsize=(14, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()
```

```python
swebench['statement_length'] = swebench['problem_statement'].apply(len)
plt.figure(figsize=(14,5))
sns.histplot(x='statement_length', data=swebench, bins=30,kde=True)
plt.title('Statement Length')
```

```python
swebench['patch_length'] = swebench['patch'].apply(len)
plt.figure(figsize=(14,5))
sns.histplot(x='patch_length', data=swebench, bins=30,kde=True)
plt.title('Patch Length')
```

```python
plt.figure(figsize=(14,5))
sns.scatterplot(data=swebench, x="patch_length", y="statement_length")
plt.title('Statement Length vs Patch Length')
```

```python
swebench.info()
```

# Data Pre-processing

```python
df = pd.DataFrame({'Question': swebench['problem_statement'], 'Answer': swebench['patch']})
df
```

```python
QNA_dataset = []
    
for index, row in df.iterrows():
    question, answer = row['Question'], row['Answer']
    template = (f"Question:\n{question}\n\nAnswer:\n{answer}")
    QNA_dataset.append(template)
```

# Gemma Fine Tuning

```python
# Install Keras 3 last. See https://keras.io/getting_started/ for more details.
!pip install -q -U keras-nlp
!pip install -q -U keras>=3

import os

os.environ["KERAS_BACKEND"] = "jax"  # Or "torch" or "tensorflow".
# Avoid memory fragmentation on JAX backend.
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]="1.00"

import keras
import keras_nlp
```

## Import Model

```python
gemma_lm = keras_nlp.models.GemmaCausalLM.from_preset("gemma_2b_en")
```

```python
gemma_lm.backbone.enable_lora(rank=64)
```

```python
gemma_lm.preprocessor.sequence_length = 512

optimizer = keras.optimizers.AdamW(
    learning_rate=5e-5,
    weight_decay=0.01,
    beta_1=0.9,          
    beta_2=0.999        
    )

optimizer.exclude_from_weight_decay(var_names=["bias", "scale"])

gemma_lm.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=optimizer,
    weighted_metrics=[keras.metrics.SparseCategoricalAccuracy()],
)
```

## Fine Tuning

```python
%%time

gemma_lm.fit(QNA_dataset, epochs=1, batch_size=1)
```

I have trained model for only 1 epoch since the dataset is very big and it takes several hours to fine tune the model on GPU.

# Submission

```python
instance_count = None

def get_number_of_instances(num_instances: int) -> None:
    """ The very first message from the gateway will be the total number of instances to be served.
    You don't need to edit this function.
    """
    global instance_count
    instance_count = num_instances
```

### Custom `predict` function

```python
first_prediction = True


def predict(problem_statement: str, repo_archive: io.BytesIO) -> str:
    """Inference function to generate a patch for a GitHub issue.

    Args:
        problem_statement (str): The text of the GitHub issue.
        repo_archive (io.BytesIO): A BytesIO buffer containing a .tar archive of the codebase.

    Returns:
        str: The generated patch as a string.
    """
    global first_prediction
    if not first_prediction:
        return None  # Skip the first issue.

    # Unpack the repository archive
    with open("repo_archive.tar", "wb") as f:
        f.write(repo_archive.read())
    repo_path = "repo"
    if os.path.exists(repo_path):
        shutil.rmtree(repo_path)
    shutil.unpack_archive("repo_archive.tar", extract_dir=repo_path)
    os.remove("repo_archive.tar")
    first_prediction = False

    # Generate a patch
    input_text = f"Problem:\n{problem_statement}\n\nPatch:"
    try:
        # Modify the generate call to remove unsupported arguments
        generated_patch = gemma_lm.generate([input_text])[0]
        return generated_patch
    except Exception as e:
        print(f"Error during generation: {e}")
        return "Error generating patch."
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