# Bhagavad Gita (भगवद्गीता) Gemma2Keras

- **Author:** Marília Prata
- **Votes:** 100
- **Ref:** mpwolke/bhagavad-gita-gemma2keras
- **URL:** https://www.kaggle.com/code/mpwolke/bhagavad-gita-gemma2keras
- **Last run:** 2024-10-04 02:40:52.327000

---

Published on October 03, 2024. By Marília Prata, mpwolke

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
secret_value_0 = user_secrets.get_secret("hf_licorne")

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
    dataset_path = "/kaggle/input/bhagwad-gita-dataset"
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

## Load the data

We load the data we will use for fine-tunining. I reduced to 52 rows to spare GPU quota.

"The Bhagavad Gita (Sanskrit: भगवद्गीता,'god's Song'),often referred to as the Gita, is a 700-verse Hindu scripture dated to the second or first century BCE, which forms chapters 23–40 in book 6 of the epic Mahabharata. While traditionally attributed to the sage Veda Vyasa, it is probably a composite work composed by multiple authors."

"It is set in a narrative framework of dialogue between the Pandava prince Arjuna and his charioteer guide Krishna, an avatar of Vishnu, at the onset of the Kurukshetra War, incorporating teachings from the Upanishads and samkhya yoga philosophy."

https://en.wikipedia.org/wiki/Bhagavad_Gita

```python
df = pd.read_csv(f"{Config.dataset_path}/Bhagwad_Gita.csv", nrows= 52) #Original has 674 rows
df.tail()
```

```python
#Don't change anything on Template line. Just the rows (in blue)
#By Gabriel Preda https://www.kaggle.com/code/gpreda/fine-tuning-gemma-2-model-using-lora-and-keras/notebook

template = "\n\nCategory:\nkaggle-{Category}\n\nQuestion:\n{Question}\n\nAnswer:\n{Answer}"
df["prompt"] = df.apply(lambda row: template.format(Category=row.Verse,
                                                             Question=row.WordMeaning,
                                                             Answer=row.HinMeaning), axis=1)
data = df.prompt.tolist()
```

## Template utility function

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

## Define the specialized class

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

## Test the fine-tuned model

```python
gemma_qa = GemmaQA()
```

#Let's try to read some Verses of The Bhagavad Gita with Gemma2

```python
#By Gabriel Preda https://www.kaggle.com/code/gpreda/fine-tuning-gemma-2-model-using-lora-and-keras/notebook

row = df.iloc[13]
gemma_qa.query(row.Verse,row.WordMeaning)
```

```python
#By Gabriel Preda https://www.kaggle.com/code/gpreda/fine-tuning-gemma-2-model-using-lora-and-keras/notebook

row = df.iloc[49]
gemma_qa.query(row.Verse,row.WordMeaning)
```

```python
#By Gabriel Preda https://www.kaggle.com/code/gpreda/fine-tuning-gemma-2-model-using-lora-and-keras/notebook

row = df.iloc[1]
gemma_qa.query(row.Verse,row.WordMeaning)
```

```python
#By Gabriel Preda https://www.kaggle.com/code/gpreda/fine-tuning-gemma-2-model-using-lora-and-keras/notebook

row = df.iloc[10]
gemma_qa.query(row.Verse,row.WordMeaning)
```

```python
#By Gabriel Preda https://www.kaggle.com/code/gpreda/fine-tuning-gemma-2-model-using-lora-and-keras/notebook

row = df.iloc[30]
gemma_qa.query(row.Verse,row.WordMeaning)
```

```python
#By Gabriel Preda https://www.kaggle.com/code/gpreda/fine-tuning-gemma-2-model-using-lora-and-keras/notebook

row = df.iloc[35]
gemma_qa.query(row.Verse,row.WordMeaning)
```

```python
#By Gabriel Preda https://www.kaggle.com/code/gpreda/fine-tuning-gemma-2-model-using-lora-and-keras/notebook

row = df.iloc[20]
gemma_qa.query(row.Verse,row.WordMeaning)
```

```python
#By Gabriel Preda https://www.kaggle.com/code/gpreda/fine-tuning-gemma-2-model-using-lora-and-keras/notebook

row = df.iloc[48]
gemma_qa.query(row.Verse,row.WordMeaning)
```

## What about something that's Not on the file

"The **Bhagavad Gita** is a synthesis of Brahmanical (dharma) and non-Brahmanical (yoga, bhakti) traditions, emphasizing discerning the **true from the false**, and performing **selfless action** through **devotion and the worship** of a personal deity, specifically **Krishna**."

Characters:

**Arjuna**, one of the five Pandavas

Krishna, Arjuna's charioteer and guru who was actually an incarnation of Vishnu

Sanjaya, counselor of the Kuru king Dhritarashtra (secondary narrator)

Dhritarashtra, Kuru king (**Sanjaya's audience**) and father of the Kauravas

```python
#Who was Arjuna Gemma2?
#By Gabriel Preda https://www.kaggle.com/code/gpreda/fine-tuning-gemma-2-model-using-lora-and-keras/notebook

category = "HinMeaning"
question = "अर्जुन जेम्मा कौन थे??"
gemma_qa.query(category,question)
```

```python
#What was the role of Dhritarashtra?

category = "HinMeaning"
question = "धृतराष्ट्र की भूमिका क्या थी?"
gemma_qa.query(category,question)
```

```python
preset_dir = ".\gemma2_2b_en_kaggle_docs"
gemma_causal_lm.save_to_preset(preset_dir)
```

#Acknowledgements:

Gabriel Preda https://www.kaggle.com/code/gpreda/fine-tuning-gemma-2-model-using-lora-and-keras/notebook

Dataset by Aman Kumar https://www.kaggle.com/datasets/a2m2a2n2/bhagwad-gita-dataset