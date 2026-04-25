# Global Communication Gemma2Keras

- **Author:** Marília Prata
- **Votes:** 98
- **Ref:** mpwolke/global-communication-gemma2keras
- **URL:** https://www.kaggle.com/code/mpwolke/global-communication-gemma2keras
- **Last run:** 2024-10-03 02:34:15.437000

---

Published on October 02, 2024. By Marília Prata, mpwolke

```python
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
```

### Competition Citation

@misc{gemma-language-tuning,

    author = {Glenn Cameron, Lauren Usui, Paul Mooney, Addison Howard},
    
    title = {Google - Unlock Global Communication with Gemma},
    
    publisher = {Kaggle},
    
    year = {2024},
    url = {https://kaggle.com/competitions/gemma-language-tuning}
}

**Gemma2 Language Capabilities**

"Gemma 2 models have demonstrated strong multilingual capabilities, impressing users with their performance in a variety of languages, including some less commonly tested ones."

"The Gemma-2-27B model excels in multilingual tasks, particularly in **non-English languages**. For example, it performs exceptionally well in **Ukrainian**. Its performance in **Greek** has also been noted as impressive, marking it as the first model to handle the language well."

"In practical applications, the 27B model has been used for translating JSON from **English to Dutch**, yielding mostly accurate results with only a few rare typos." 

"Tests in **Russian** produced coherent results, and there were confirmations of effective performance in **Slovenian**, indicating that Gemma 2 handles Slavic languages particularly well."

"The 9B model has also been praised for its multilingual abilities. For instance, the fine-tuned version of Gemma-2-9B was exceptional in **French**. Both the 9B and 27B models were noted to be very effective in **Korean**. The 27B model not only produces grammatically correct text but also demonstrates excellent semantic understanding and accurately comprehends user requests. With further tuning and increased context size, the 27B model could become the best open-source model for Korean."

"The 9B model performed admirably in Korean tasks, often exceeding expectations for its size."

"Overall, Gemma 2 models are proving to be robust in **handling a wide range of languages**, providing accurate and contextually appropriate translations and text generation."

https://llm.extractum.io/static/blog/?id=_the-first-ai-community-feedback-on-gemma-2-new-open-llms

```python
#By Gabriel Preda https://www.kaggle.com/code/gpreda/fine-tuning-gemma-2-model-using-lora-and-keras/notebook
# Install Keras 3 last. See https://keras.io/getting_started/ for more details.
!pip install -q -U keras-nlp
!pip install -q -U keras>=3
!pip install -q -U kagglehub --upgrade
```

```python
#By Gabriel Preda https://www.kaggle.com/code/gpreda/fine-tuning-gemma-2-model-using-lora-and-keras/notebook

import os
os.environ["KERAS_BACKEND"] = "jax" # you can also use tensorflow or torch
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "1.00" # avoid memory fragmentation on JAX backend.
os.environ["JAX_PLATFORMS"] = ""
import keras
import keras_nlp
import kagglehub


#Make yours and Add copy to clipboard
from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
secret_value_0 = user_secrets.get_secret("KAGGLE_KEY")
secret_value_1 = user_secrets.get_secret("mpwolke")


#Gabriel's line
#from kaggle_secrets import UserSecretsClient
#user_secrets = UserSecretsClient()
#os.environ["KAGGLE_USERNAME"] = user_secrets.get_secret("kaggle_username")
#os.environ["KAGGLE_KEY"] = user_secrets.get_secret("kaggle_key")

import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
tqdm.pandas() # progress bar for pandas

import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, Markdown
```

```python
#By Gabriel Preda https://www.kaggle.com/code/gpreda/fine-tuning-gemma-2-model-using-lora-and-keras/notebook

class Config:
    seed = 42
    dataset_path = "/kaggle/input/hello-world-in-programming-languages"
    preset = "gemma2_2b_en" # name of pretrained Gemma 2
    sequence_length = 512 # max size of input sequence for training
    batch_size = 1 # size of the input batch in training
    lora_rank = 4 # rank for LoRA, higher means more trainable parameters
    learning_rate=8e-5 # learning rate used in train
    epochs = 10 # number of epochs to train
```

```python
keras.utils.set_random_seed(Config.seed)
```

#Load the data

We load the data we will use for fine-tunining.

```python
df = pd.read_csv(f"{Config.dataset_path}/index.csv", nrows= 52) #Original has 674 rows
df.head()
```

### Replace missing values by an specific language 

Replace missing values by an specific language, since I intend to use row 15, I replaced that missing value.

```python
#Replace Missing value on extension column 

df.loc[15, "extension"]= "arabic"
```

```python
df['extension'][15]
```

## Total number of rows in this dataset.

Total was 674, I didn't wan't to spend much time/GPU on the model. Thus, I reduced nrows to 52.

```python
df.shape[0]
```

```python
#Don't change anything on Template line. Just the rows (in blue)
#By Gabriel Preda https://www.kaggle.com/code/gpreda/fine-tuning-gemma-2-model-using-lora-and-keras/notebook

template = "\n\nCategory:\nkaggle-{Category}\n\nQuestion:\n{Question}\n\nAnswer:\n{Answer}"
df["prompt"] = df.apply(lambda row: template.format(Category=row.language_name,
                                                             Question=row.extension,
                                                             Answer=row.program), axis=1)
data = df.prompt.tolist()
```

### Template utility function

```python
#By Gabriel Preda https://www.kaggle.com/code/gpreda/fine-tuning-gemma-2-model-using-lora-and-keras/notebook

def colorize_text(text):
    for word, color in zip(["Category", "Question", "Answer"], ["blue", "red", "green"]):
        text = text.replace(f"\n\n{word}:", f"\n\n**<font color='{color}'>{word}:</font>**")
    return text
```

```python
#By Gabriel Preda https://www.kaggle.com/code/gpreda/fine-tuning-gemma-2-model-using-lora-and-keras/notebook

gemma_causal_lm = keras_nlp.models.GemmaCausalLM.from_preset(Config.preset)
gemma_causal_lm.summary()
```

### Define the specialized class

```python
#By Gabriel Preda https://www.kaggle.com/code/gpreda/fine-tuning-gemma-2-model-using-lora-and-keras/notebook

class GemmaQA:
    def __init__(self, max_length=512):
        self.max_length = max_length
        self.prompt = template
        self.gemma_causal_lm = gemma_causal_lm
        
    def query(self, category, question):
        response = self.gemma_causal_lm.generate(
            self.prompt.format(
                Category=category,
                Question=question,
                Answer=""), 
            max_length=self.max_length)
        display(Markdown(colorize_text(response)))
```

```python
x, y, sample_weight = gemma_causal_lm.preprocessor(data[0:2])
```

```python
print(x, y)
```

#Perform fine-tuning with LoRA

```python
#By Gabriel Preda https://www.kaggle.com/code/gpreda/fine-tuning-gemma-2-model-using-lora-and-keras/notebook

# Enable LoRA for the model and set the LoRA rank to the lora_rank as set in Config (4).
gemma_causal_lm.backbone.enable_lora(rank=Config.lora_rank)
gemma_causal_lm.summary()
```

#Gemma_causal_lm  

Epochs!

```python
#By Gabriel Preda https://www.kaggle.com/code/gpreda/fine-tuning-gemma-2-model-using-lora-and-keras/notebook

#set sequence length cf. config (512)
gemma_causal_lm.preprocessor.sequence_length = Config.sequence_length 

# Compile the model with loss, optimizer, and metric
gemma_causal_lm.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(learning_rate=Config.learning_rate),
    weighted_metrics=[keras.metrics.SparseCategoricalAccuracy()],
)

# Train model
gemma_causal_lm.fit(data, epochs=Config.epochs, batch_size=Config.batch_size)
```

### Test the fine-tuned model

```python
gemma_qa = GemmaQA()
```

### Sample 1

```python
#Verifying the index numbers: 13 Sanskrit

df.iloc[13]
```

### Let's try to read some non-english languages with Gemma2

I adapted it as far my poor non-coder knowledge leads me.

```python
#By Gabriel Preda https://www.kaggle.com/code/gpreda/fine-tuning-gemma-2-model-using-lora-and-keras/notebook

row = df.iloc[13]
gemma_qa.query(row.language_name,row.extension)
```

### Trying some samples of "Hello World"

```python
row = df.iloc[19]
gemma_qa.query(row.language_name,row.extension)
```

```python
row = df.iloc[12]
gemma_qa.query(row.language_name,row.extension)
```

```python
row = df.iloc[15]
gemma_qa.query(row.language_name,row.extension)
```

```python
row = df.iloc[31]
gemma_qa.query(row.language_name,row.extension)
```

```python
row = df.iloc[36]
gemma_qa.query(row.language_name,row.extension)
```

### These two below are so different on the dataset

```python
row = df.iloc[25]
gemma_qa.query(row.language_name,row.extension)
```

```python
row = df.iloc[51]
gemma_qa.query(row.language_name,row.extension)
```

### Not seen questions :D

```python
#By Gabriel Preda https://www.kaggle.com/code/gpreda/fine-tuning-gemma-2-model-using-lora-and-keras/notebook

category = "klingon"
question = "Could you translate klingon Gemma2?"
gemma_qa.query(category,question)
```

![](https://y.yarn.co/727b6ca1-432a-44ef-a5c2-f62a56f3ba83_text.gif)Yarn

```python
category = "language_name"
question = "How to say Hello in Klingon?"
gemma_qa.query(category,question)
```

### Save the model

```python
preset_dir = ".\gemma2_2b_en_kaggle_docs"
gemma_causal_lm.save_to_preset(preset_dir)
```

#Acknowledgements:

Gabriel Preda https://www.kaggle.com/code/gpreda/fine-tuning-gemma-2-model-using-lora-and-keras/notebook

Dataset by Yaroslav Isaienkov https://www.kaggle.com/datasets/ihelon/hello-world-in-programming-languages/data