# Regression Test

- **Author:** jazivxt
- **Votes:** 294
- **Ref:** jazivxt/regression-test
- **URL:** https://www.kaggle.com/code/jazivxt/regression-test
- **Last run:** 2025-01-03 05:53:10.610000

---

```python
import numpy as np
import pandas as pd
from sklearn import *
import pickle, warnings, random, tqdm
from collections import Counter
warnings.simplefilter('ignore')

p = '/kaggle/input/santa-2024/sample_submission.csv'
df = pd.read_csv(p) # 	id 	text
print(df['text'].map(lambda x: len(str(x).split(' '))).values)
```

```python
t = """reindeer mistletoe elf gingerbread family advent scrooge chimney fireplace ornament
reindeer sleep walk the night and drive mistletoe scrooge laugh chimney jump elf bake gingerbread family give advent fireplace ornament
sleigh yuletide beard carol cheer chimney decorations gifts grinch holiday holly jingle magi naughty nice nutcracker ornament polar workshop stocking
sleigh of the magi yuletide cheer is unwrap gifts and eat cheer holiday decorations holly jingle relax sing carol visit workshop grinch naughty nice chimney stocking ornament nutcracker polar beard
with as and from in it of not that the to we you angel believe bow candle candy chocolate cookie doll dream eggnog fireplace fruitcake game greeting card have hohoho hope joy kaggle merry milk night peace peppermint poinsettia puzzle season snowglobe star toy wrapping paper wreath wish wonder workshop
from and and as we and have the in is it of not that the to with you advent card angel bake beard believe bow candy candle carol cheer cheer chocolate chimney cookie decorations doll dream drive eat eggnog family fireplace fireplace chimney fruitcake game gifts give gingerbread greeting grinch holiday holly hohoho hope jingle jump joy kaggle laugh magi merry milk mistletoe naughty nice night night elf nutcracker ornament ornament of the wrapping paper peace peppermint polar poinsettia puzzle reindeer relax scrooge season sing sleigh sleep snowglobe star stocking toy unwrap visit walk wish wonder workshop workshop wreath yuletide"""

df['text'] = t.split('\n')
df.to_csv("submission.csv", index=False)
```

```python
with open('/kaggle/input/santa-2024-perplexity-permutation-puzzle-scores/past.pickle', 'rb') as handle:
    scores = pickle.load(handle)
print(len(scores))
dID = {83:0, 135:1, 149:2, 197:3, 302:4, 636:5}
df = pd.DataFrame.from_dict(scores, orient='index').reset_index()
df.columns = ['text','score']
df['id'] = df['text'].map(len)
df['id'] = df['id'].map(dID)
df.head(1)
```

# Sample Sizes

```python
t = t.split('\n')
df0 = df[df['id']==0]
df1 = df[df['id']==1]
df2 = df[df['id']==2]
df3 = df[df['id']==3]
df4 = df[df['id']==4]
df5 = df[df['id']==5]
```

```python
#Need to address duplicates in indexing
#Solution 1 - Combine Temporarily, they are mostly always together in high scoring
#Solution 2 - Add a number to each and make unique then clean up before scoring
for l in t:
    c = Counter(l.split(' '))
    dups = {k:v for k,v in c.items() if v > 1}
    print(dups)
```

```python
for i in range(6):
    dfx = eval('df'+str(i))
    for c in t[i].split(' '):
        dfx[c] = dfx['text'].map(lambda x: x.split(' ').index(c))
    print(i, len(dfx))
```

```python
from IPython.core.display import display, HTML

for i in range(6):
    dfx = eval('df'+str(i))
    print("ROW: ", i)
    dfx = dfx.sort_values(['score']).reset_index(drop=True)
    #for r in range(5):
        #print(dfx['score'][r])
        #if r == 0:
            #print(dfx['text'][r])
    #TODO: Add Color Codes to Text
    dfxh = dfx[:10]
    n = len(dfxh['text'][0].split(' '))
    dfxh = dfxh['text'].str.split(' ', n=n, expand=True)
    dfxh['score'] = dfx['score'][:10]
    html = dfxh.to_html()
    display(HTML(html))
```

```python
x1, x2, y1, y2 = model_selection.train_test_split(df3, df3['score'], test_size=0.25, random_state=99)
col = [c for c in df3.columns if c not in ['id','score','text']]
```

```python
!pip3 install xgboost
```

```python
import xgboost
from xgboost import XGBRegressor

model = XGBRegressor(device="cpu", max_depth=9, colsample_bytree=0.5, subsample=0.8, n_estimators=2_500,  learning_rate=0.2, eval_metric="mae", early_stopping_rounds=50, objective='reg:squarederror', min_child_weight=5)
model = model.fit(x1[col], y1, eval_set=[(x2[col], y2)],  verbose=100)
```

# ~2.4x10^18 Permutations for 20 Words - YIKES!

```python
%%time
import itertools

known = df3['text'].values
unknown = []
text = 'sleigh of the magi yuletide cheer is unwrap gifts and eat cheer holiday decorations holly jingle relax sing carol visit workshop grinch naughty nice chimney stocking ornament nutcracker polar beard'
text = text.split(' ')

x1 = text[:22]
part = text[22:]

p = list(itertools.permutations(part))
for i in tqdm.tqdm(range(len(p))):
    t = ' '.join(x1 + list(p[i]) )
    unknown.append(t)
print(len(unknown))

#for i in tqdm.tqdm(range(500_000)):
#    random.shuffle(text)
#    unknown.append(' '.join(text))
#unknown = list(set(unknown))
#print(len(unknown))

x3 = pd.DataFrame({'text': unknown})
for c in text:
    x3[c] = x3['text'].map(lambda x: x.split(' ').index(c))
x3['score'] = model.predict(x3[col])
x3 = x3.sort_values(['score']).reset_index(drop=True)
print(x3['score'][0])
print(x3['text'][0])
#x3.to_csv('text2_random.csv', index=False)
```

# Lets Verify Some Scores

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

scorer = PerplexityCalculator()
```

## Low

```python
%%time

for i in tqdm.tqdm(range(200)):
    if x3['text'][i] not in scores:
        s = scorer.get_perplexity(x3['text'][i])
        scores[x3['text'][i]] = s
    s = scores[x3['text'][i]]
    if s < 210:
        print(i, x3['score'][i], s)
        print(x3['text'][i])
```

## Middle

```python
%%time
z = x3[20_000:].reset_index(drop=True)
for i in range(10):
    if z['text'][i] not in scores:
        s = scorer.get_perplexity(z['text'][i])
        scores[z['text'][i]] = s
    print(i, z['score'][i], scores[z['text'][i]], z['text'][i])
```

## High

```python
%%time
z = x3.tail(20).reset_index(drop=True)

for i in range(20):
    if z['text'][i] not in scores:
        s = scorer.get_perplexity(z['text'][i])
        scores[z['text'][i]] = s
    print(i, z['score'][i], scores[z['text'][i]])
```

```python
# Exploring New Paths
t='sleigh of the magi yuletide cheer is unwrap gifts and eat cheer holiday decorations holly jingle relax sing carol visit workshop grinch chimney naughty nice nutcracker polar beard ornament stocking'
```

```python
import itertools

def part_perm_brutem(st, start=0, end=3, skips=1):
    global scores
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
        if t in scores:
            s = scores[t]
        else:
            s =  scorer.get_perplexity(t)
            scores[t] = s
        if s < best and t:
            print("New Score: ", s, t)
            best = s
            bestt = t
    return bestt
```

```python
bestt = t
l = len(bestt.split(' '))
for p in range(1, 3):
    for start in tqdm.tqdm(range(0,l-p+1)):
        bestt = part_perm_brutem(bestt, start, start+p, 1)
```

```python
#import pickle

#with open('scores.pickle', 'wb') as f:
    #pickle.dump(scores, f, protocol=pickle.HIGHEST_PROTOCOL)
```