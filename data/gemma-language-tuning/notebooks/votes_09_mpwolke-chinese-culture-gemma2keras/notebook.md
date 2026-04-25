# Chinese Culture 中國文化 Gemma2Keras

- **Author:** Marília Prata
- **Votes:** 55
- **Ref:** mpwolke/chinese-culture-gemma2keras
- **URL:** https://www.kaggle.com/code/mpwolke/chinese-culture-gemma2keras
- **Last run:** 2024-10-12 00:53:10.067000

---

Published on October 08, 2024. By Marília Prata, mpwolke

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

古文: Classical chinese

现代文: Modern

路径: Path

```python
#Code by StackOverflow https://stackoverflow.com/questions/50475635/loading-jsonl-file-as-json-objects

df1 = pd.read_json(path_or_buf= '../input/chinese-culture-db-2024/flyfire/final_database.jsonl', lines=True)

df1.tail()
```

```python
df = pd.read_json(path_or_buf= '../input/chinese-culture-db-2024/bitext_small.jsonl', lines=True)

df.head()
```

```python
df = df.rename(columns={'古文':'classical_chinese', '现代文': 'modern', '路径': 'path'})
```

### Classical chinese 1169

陈亢退而喜曰： 问一得三，闻《诗》，闻《礼》，又闻君子之远其子也。

"Chen Kang left and said happily: "I have asked one question and got three answers. I have heard about the Book of Songs, the Book of Rites, and the way a gentleman keeps his son at a distance."

```python
#陈亢退而喜曰： 问一得三，闻《诗》，闻《礼》，又闻君子之远其子也。

df['classical_chinese'][1169]
```

### Modern 2

假如天下百姓都困苦贫穷，上天赐给你的禄位也就会永远终止。 

If all the people in the world are poor and miserable, the official position bestowed on you by God will end forever.

```python
df['modern'][2]
```

```python
df['path'][2]
```

### Modern 1666

我回答： 没有。 不学《礼记》就无法立足于社会啊！ - I answered: No. If you don’t learn the Book of Rites, you won’t be able to gain a foothold in society!

```python
df['modern'][1166]
```

### Modern 4

商汤说： 我履谨用黑色的公牛来祭祀，向伟大的天帝祷告：有罪的人我不敢擅自赦免，您的臣仆的善恶我也不隐瞒掩盖，都由天帝的心来分辨、选择。

Shang Tang said: I, Lu Jin, used a black bull for sacrifice and prayed to the great Emperor of Heaven: I dare not pardon the guilty on my own, nor will I conceal the good and evil of your servants. All are distinguished and chosen by the heart of the Emperor of Heaven.

```python
df['modern'][4]
```

### Classical_chinese 4

曰： 予小子履，敢用玄牡，敢昭告于皇皇后帝：有罪不敢赦，帝臣不蔽，简在帝心。

He said: I, a young boy named Lu, dare to use the black bull and dare to announce to the emperor: I dare not pardon any crime, and the emperor's ministers will not cover up anything, as the simplicity is in the emperor's heart.

### Classical chinese 6

周有大赉，善人是富。 虽有周亲，不如仁人。 - Zhou has great gifts, and good people are rich. Although there are relatives of Zhou, they are not as good as benevolent people.

```python
df['classical_chinese'][6]
```

#TPU On!

```python
!pip install -q -U keras-nlp tensorflow-text
# Install tensorflow-cpu so tensorflow does not attempt to access the TPU.
!pip install -q -U tensorflow-cpu
```

```python
import jax

jax.devices()
```

```python
#By Matt Watson https://www.kaggle.com/code/matthewdwatson/gemma-2-fine-tuning-and-inference/notebook

import os

# The Keras 3 distribution API is only implemented for the JAX backend for now
os.environ["KERAS_BACKEND"] = "jax"
# Pre-allocate all TPU memory to minimize memory fragmentation and allocation overhead.
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "1.0"
```

#Import Keras

```python
import keras
import keras_nlp
```

```python
#By Matt Watson https://www.kaggle.com/code/matthewdwatson/gemma-2-fine-tuning-and-inference/notebook

# Create a device mesh with (1, 8) shape so that the weights are sharded across
# all 8 TPUs.
device_mesh = keras.distribution.DeviceMesh(
    (1, 8),
    ["batch", "model"],
    devices=keras.distribution.list_devices(),
)
```

#LayoutMap

"LayoutMap from the distribution API specifies how the weights and tensors should be sharded or replicated, using the string keys, for example, token_embedding/embeddings below, which are treated like regex to match tensor paths. Matched tensors are sharded with model dimensions (8 TPUs); others will be fully replicated."

```python
#By Matt Watson https://www.kaggle.com/code/matthewdwatson/gemma-2-fine-tuning-and-inference/notebook

model_dim = "model"

layout_map = keras.distribution.LayoutMap(device_mesh)

# Weights that match 'token_embedding/embeddings' will be sharded on 8 TPUs
layout_map["token_embedding/embeddings"] = (model_dim, None)
# Regex to match against the query, key and value matrices in attention layers
layout_map["decoder_block.*attention.*(query|key|value)/kernel"] = (model_dim, None, None)
layout_map["decoder_block.*attention_output/kernel"] = (model_dim, None, None)
layout_map["decoder_block.*ffw_gating.*/kernel"] = (None, model_dim)
layout_map["decoder_block.*ffw_linear/kernel"] = (model_dim, None)
```

### Works with Gemma2_9b

It didn't work with Gemma 2b

```python
#By Matt Watson https://www.kaggle.com/code/matthewdwatson/gemma-2-fine-tuning-and-inference/notebook

model_parallel = keras.distribution.ModelParallel(
    layout_map=layout_map,
    batch_dim_name="batch",
)

keras.distribution.set_distribution(model_parallel)
gemma_lm = keras_nlp.models.GemmaCausalLM.from_preset("gemma2_9b_en")
```

#Decoder Block_1

```python
#By Matt Watson https://www.kaggle.com/code/matthewdwatson/gemma-2-fine-tuning-and-inference/notebook

decoder_block_1 = gemma_lm.backbone.get_layer('decoder_block_1')
print(type(decoder_block_1))
for variable in decoder_block_1.weights:
  print(f'{variable.path:<48}  {str(variable.shape):<14}  {str(variable.value.sharding.spec)}')
```

#Inference before Tuning: 假如天下百姓都困苦贫穷，上天赐给你的禄位也就会永远终止。

If all the people in the world are poor and miserable, the official position bestowed on you by God will end forever.

```python
print(gemma_lm.generate("假如天下百姓都困苦贫穷，上天赐给你的禄位也就会永远终止。?", max_length=512))
```

```python
#By Jocelyn Dumlao

#Instruction fine-tuning
#I Changed: Instruction/Response for the columns that we have. 

import json

with open('/kaggle/input/chinese-culture-db-2024/bitext_small.jsonl') as file:
     for i, line in enumerate(file):
         features = json.loads(line)
         print(features)  # Print the entire JSON object
         if i >= 5:  # Limit to first 5 entries for inspection
             break
```

```python
#Fixed By Jocelyn Dumlao

data = []
with open('/kaggle/input/chinese-culture-db-2024/bitext_small.jsonl') as file:
     for line in file:
         features = json.loads(line)
         # No filtering, just format and collect
         template = "Instruction:\n{古文}\n\nResponse:\n{现代文}"  # Adjusted key 
 
         data.append(template.format(**features))

 # Truncate our data to speed up training.
data = data[:2000]

 # Print the number of examples collected to verify
print(f"Number of examples collected: {len(data)}")
```

### A single training example.

On the last Notebook, I wanted the 2503 row, since it's out of index, number 6 is do.

```python
data[4]
```

```python
data[6]
```

```python
data[2]
```

```python
#By Matt Watson https://www.kaggle.com/code/matthewdwatson/gemma-2-fine-tuning-and-inference/notebook

# Enable LoRA for the model and set the LoRA rank to 4.
gemma_lm.backbone.enable_lora(rank=8)
```

```python
#By Matt Watson https://www.kaggle.com/code/matthewdwatson/gemma-2-fine-tuning-and-inference/notebook

# Limit the input sequence length to 512 to control memory usage.
gemma_lm.preprocessor.sequence_length = 512
gemma_lm.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(learning_rate=5e-5),
    weighted_metrics=[keras.metrics.SparseCategoricalAccuracy()],
)
gemma_lm.summary()
```

#LoRA - Epochs! Data is empty Not able to enable LoRA or anything else

"Note that enabling LoRA reduces the number of trainable parameters significantly, from 9 billion to only 14 million."

"Fine-tune your model"

```python
gemma_lm.fit(data, epochs=1, batch_size=4)
```

### Modern 2

如果天下人都窮困潦倒，上天賜予你的官位怎麼辦？ 

What would happen to the official position bestowed on you by God if all the people in the world are poor and miserable?

Gemma2 output below:

Instruction:
If everyone in the world is in poverty, what will happen if God gives you an official position? ?

Response:
If everyone in the world is poor and God has given you an official position, you should return your official position to God.

## Expected Answer : the official position bestowed on you by God will end forever.

If all the people in the world are poor and miserable, the official position bestowed on you by God will end forever.

假如天下百姓都困苦贫穷，上天赐给你的禄位也就会永远终止。

If all the people in the world are poor and miserable, the official position bestowed on you by God will end forever.

```python
print(gemma_lm.generate("Instruction:\n如果天下人都窮困潦倒，上天賜予你的官位怎麼辦？?\n\nResponse:\n", max_length=512))
```

### Expected answer Modern 1666

我回答： 没有。 不学《礼记》就无法立足于社会啊！ - I answered: No. If you don’t learn the Book of Rites, you won’t be able to gain a foothold in society!

```python
print(gemma_lm.generate("Instruction:\n為什麼學習禮記很重要?\n\nResponse:\n", max_length=512))
```

## Answer according to Google translate:

為什麼學習禮記很重要?

Why is it important to learn the Book of Rites?

因為禮記是儒家經典，是儒家思想的集中體現，是儒家思想的精華所在。

"Because the Book of Rites is a Confucian classic, it is the concentrated expression of Confucian thought and the essence of Confucian thought."

### Translation for the first version:

Instruction: Why is it important to learn the Book of Rites?

Response:

"The Book of Rites is a Confucian classic, the culmination of Confucian thought, and the essence of Confucian thought. The Book of Rites is a Confucian classic, the culmination of Confucian thought, and the essence of Confucian thought. The Book of Rites is a Confucian classic, the culmination of Confucian thought, and the essence of Confucian thought. The Book of Rites is a Confucian classic, the culmination of Confucian thought, and the essence of Confucian thought. The Book of Rites is a Confucian classic, the culmination of Confucian thought, and the essence of Confucian thought. The Book of Rites is a Confucian classic, the culmination of Confucian thought, and the essence of Confucian thought. The Book of Rites is a Confucian classic, the culmination of Confucian thought, and the essence of Confucian thought. The Book of Rites is a Confucian classic, the culmination of Confucian thought, and the essence of Confucian thought. The Book of Rites is a Confucian classic, the culmination of Confucian thought, and the essence of Confucian thought. The Book of Rites is a Confucian classic, the culmination of Confucian thought, and the essence of Confucian thought. The Book of Rites is a Confucian classic, the culmination of Confucian thought, and the essence of Confucian thought. The Book of Rites is a Confucian classic, the culmination of Confucian thought, and the essence of Confucian thought. The Book of Rites is a Confucian classic, the culmination of Confucian thought, and the essence of Confucian thought. The Book of Rites is a Confucian classic, the culmination of Confucian thought, and the essence of Confucian thought. The Book of Rites is a Confucian classic, the culmination of Confucian thought, and the essence of Confucian thought. The Book of Rites is a Confucian classic, the culmination of Confucian thought, and the essence of Confucian thought. The Book of Rites is a Confucian classic, the culmination of Confucian thought, and the essence of Confucian thought. The Book of Rites is a Confucian classic, the culmination of Confucian thought, and the essence of Confucian thought. The Book of Rites is a Confucian classic, the culmination of Confucian thought, and the essence of Confucian thought. The Book of Rites is Confucianism."

I wasn't able to end cause my data is empty. Anyway, it's chinese only with Google translate I could understand a little bit.

## Maybe Google developers could fix it - 也許Google開發者可以解決這個問題

#Acknowledgements:

Matt Watson https://www.kaggle.com/code/matthewdwatson/gemma-2-fine-tuning-and-inference/notebook

Jocelyn Dumlao