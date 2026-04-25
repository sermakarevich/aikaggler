# Diminutive Effort (TPU)

- **Author:** jazivxt
- **Votes:** 355
- **Ref:** jazivxt/diminutive-effort-tpu
- **URL:** https://www.kaggle.com/code/jazivxt/diminutive-effort-tpu
- **Last run:** 2024-12-18 05:56:52.310000

---

```python
import numpy as np
import pandas as pd
from collections import Counter
from tqdm.auto import tqdm
import random, pickle, math, warnings
import itertools,  multiprocessing, json
#warnings.simplefilter('ignore')
print("CPU Count: ", multiprocessing.cpu_count())

p = '/kaggle/input/santa-2024/sample_submission.csv'
df = pd.read_csv(p)
```

```python
#!pip3 install torch-xla
#import torch_xla.core.xla_model as xm
```

```python
#import tensorflow as tf

#print("Tensorflow version " + tf.__version__)
#AUTO = tf.data.experimental.AUTOTUNE
#tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
#print('Running on TPU ', tpu.master())
#tf.config.experimental_connect_to_cluster(tpu)
#tf.tpu.experimental.initialize_tpu_system(tpu)
#tpu_strategy = tf.distribute.TPUStrategy(tpu)
#print("REPLICAS: ", tpu_strategy.num_replicas_in_sync)
```

```python
import transformers, torch, os
from math import exp
from collections import Counter, OrderedDict
from typing import List, Optional, Union
import torch.nn.functional as F

DEVICE = torch.device('cpu')
#DEVICE = xm.xla_device(tpu_strategy)
MODEL_PATH = "/kaggle/input/gemma-2/transformers/gemma-2-9b/2"

#https://www.kaggle.com/code/neibyr/batch-metric-with-cache
class LRUCache:
    def __init__(self, capacity=10**11):
        self.capacity = capacity
        self.cache = OrderedDict()
    def get(self, key):
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        return None
    def set(self, key, value):
        self.cache[key] = value
        self.cache.move_to_end(key)
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)
    def __len__(self):
        return len(self.cache)

class PerplexityCalculator:
    def __init__(self, capacity=10**11):
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_PATH)
        self.model = transformers.AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map="auto", torch_dtype=torch.float32,)
        self.loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        self.model.eval()
        #self.model.to(DEVICE)
        self.cache = LRUCache(capacity=capacity)

    #Add multiprocessing
    def get_perplexity(self, input_texts, batch_size=128, use_cache=True,) -> Union[float, List[float]]:
        single_input = isinstance(input_texts, str)
        input_texts = [input_texts] if single_input else input_texts
        results = [None] * len(input_texts)
        if use_cache:
            text_to_process = []
            for i, text in enumerate(input_texts):
                cached_val = self.cache.get(text)
                if cached_val is not None:
                    results[i] = cached_val
                else:
                    text_to_process.append(text)
        else:
            text_to_process = input_texts.copy()
        loss_list = []
        batches = len(text_to_process)//batch_size + (len(text_to_process)%batch_size != 0)
        pbar = range(batches)
        for j in pbar:
            a = j*batch_size
            b = (j+1)*batch_size
            input_batch = text_to_process[a:b]
            with torch.no_grad():
                text_with_special = [f"{self.tokenizer.bos_token}{text}{self.tokenizer.eos_token}" for text in input_batch]
                model_inputs = self.tokenizer(text_with_special, return_tensors='pt', add_special_tokens=False,)
                #model_inputs = {k: v.to(DEVICE) for k, v in model_inputs.items()}
                logits = self.model(**model_inputs, use_cache=True)['logits']
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = model_inputs['input_ids'][..., 1:].contiguous()
                loss = self.loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1))
                sequence_loss = loss.sum() / len(loss)
                loss_list.append(sequence_loss.cpu().item())
        ppl = [exp(i) for i in loss_list]
        index_ppl = 0
        for index_el, el in enumerate(results):
            if el is None:
                results[index_el] = ppl[index_ppl]
                self.cache.set(text_to_process[index_ppl], ppl[index_ppl])
                index_ppl += 1
        return results[0] if single_input else results

# instantiating the model in the strategy scope creates the model on the TPU
#with tpu_strategy.scope():
     # define your model normally
scorer = PerplexityCalculator()
```

```python
t = """reindeer mistletoe elf gingerbread family advent scrooge chimney fireplace ornament
reindeer sleep walk the night and drive mistletoe scrooge laugh chimney jump elf bake gingerbread family give advent fireplace ornament
magi yuletide cheer grinch carol holiday holly jingle naughty nice nutcracker polar beard ornament stocking chimney sleigh workshop gifts decorations
sleigh of the magi yuletide cheer is unwrap gifts and eat cheer holiday decorations holly jingle relax carol sing chimney visit grinch naughty nice polar beard workshop nutcracker ornament stocking
from and as have in not it of that the to we with you bow angel believe candle candy card chocolate cookie doll dream eggnog fireplace fruitcake game greeting hohoho hope joy kaggle merry milk night peace peppermint poinsettia puzzle season snowglobe star toy wreath wish workshop wonder wrapping paper
from and and as and have the in is it of not that the to we with you advent card angel bake beard believe bow candy candle carol cheer cheer chocolate chimney cookie decorations doll dream drive eat eggnog family fireplace fireplace chimney fruitcake game give gifts gingerbread greeting grinch holiday holly hohoho hope jingle jump joy kaggle laugh magi merry milk mistletoe naughty nice night night elf nutcracker ornament ornament of the wrapping paper peace peppermint polar poinsettia puzzle reindeer relax scrooge season sing sleigh sleep snowglobe star stocking toy unwrap visit walk wish wonder workshop workshop wreath yuletide"""

df['text'] = t.split('\n')
df['score'] = df['text'].map(lambda x: scorer.get_perplexity(x))
df.to_csv("submission.csv", index=False)
print(np.mean(df['score']))
df['score']
```

```python
def part_perm_brutem(st, start=0, end=3, skips=1):
    bestt = st
    best = scorer.get_perplexity(st)
    st = st.split(' ')
    part = st[start:end]
    if start>0:
        st1 =  ' '.join(st[:start]) + ' '
    else:
        st1 = ''
    if end<len(st): 
        st2 =  ' ' + ' '.join(st[end:])
    else: 
        st2 = ''
    p = list(itertools.permutations(part))
    for i in range(0, len(p), skips): #removed tqdm
        t =  st1 + ' '.join(list(p[i])) + st2
        s =  scorer.get_perplexity(t)
        if s < best:
            print("New Score: ", s, t)
            best = s
            bestt = t
    return bestt
```

```python
for i in range(5,6):
    bestt = df['text'][i]
    l = len(df['text'][i].split(' '))
    for p in range(2, 3):
        #for start in tqdm(range(0,l-p+1)):
        for start in tqdm(range(45,50)):
            bestt = part_perm_brutem(bestt, start, start+p, 1)
            df.at[i, 'text'] = bestt
        df.to_csv("submission.csv", index=False)

df['score'] = df['text'].map(lambda x: scorer.get_perplexity(x))
df.to_csv("submission.csv", index=False)
print(np.mean(df['score']))
df['score']
```