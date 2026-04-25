# 🗝️ To Winning - Sort Off

- **Author:** jazivxt
- **Votes:** 373
- **Ref:** jazivxt/to-winning-sort-off
- **URL:** https://www.kaggle.com/code/jazivxt/to-winning-sort-off
- **Last run:** 2025-01-20 00:47:52.157000

---

# 🏆 From Scratch to Competitive Score

```python
!pip install levenshtein
```

```python
import numpy as np
import pandas as pd
from collections import Counter
from tqdm import tqdm
import random, pickle, math, warnings
import itertools, nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
#warnings.simplefilter('ignore')
from Levenshtein import distance

p = '/kaggle/input/santa-2024/sample_submission.csv'
df = pd.read_csv(p) # 	id 	text
print(df['text'].map(lambda x: len(str(x).split(' '))).values)
```

# 🥅 Scoring Metric

```python
import transformers, torch
import gc, os, logging
from math import exp

os.environ['OMP_NUM_THREADS'] = '2'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
MODEL_PATH = "/kaggle/input/gemma-2/transformers/gemma-2-9b/2"
DEVICE = torch.device('cuda')

class PerplexityCalculator:
    def __init__(self,):
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_PATH)
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            MODEL_PATH, device_map="auto",
            torch_dtype=torch.float16,)
        self.loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        self.model.eval()

    def get_perplexity(self, text: str) -> float:
        with torch.no_grad():
            text_with_special = f"{self.tokenizer.bos_token}{text}{self.tokenizer.eos_token}"
            model_inputs = self.tokenizer(text_with_special, return_tensors='pt', add_special_tokens=False,)
            model_inputs = {k: v.to(DEVICE) for k, v in model_inputs.items()}
            logits = self.model(**model_inputs, use_cache=False)['logits']
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = model_inputs['input_ids'][..., 1:].contiguous()
            loss = self.loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1))
            sequence_loss = loss.sum() / len(loss)
            loss_list = sequence_loss.cpu().item()
        return math.exp(loss_list)
        
scorer = PerplexityCalculator()  #Uncomment for GPU
```

```python
# Dummy Metric Class for Non GPU Testing
#class PerplexityCalculator:
#    def __init__(self,):
#        self.test = 0

#    def get_perplexity(self, text: str) -> float:
#        return 9999999999999999.0

#scorer = PerplexityCalculator()

# Comment Above and Change to GPU T4x2
#* I exceeded my Accelerator quota
```

# ⚙️Preprocessed Scores for Faster Testing
* You can verify scores or run without dataset

```python
past = {}
# You can comment out the following lines to run on a faster GPU with different scoring
with open('/kaggle/input/santa-2024-perplexity-permutation-puzzle-scores/past.pickle', 'rb') as handle:
    past = pickle.load(handle)
print(len(past))
```

# 🪄📶 The Sort Off Magic

```python
#https://www.kaggle.com/code/asalhi/sorting-sample-6-stopwords-first
def custom_sort(words):
    stop_words_in_text = sorted([word for word in words if word.lower() in stop_words])
    other_words = sorted([word for word in words if word.lower() not in stop_words])
    return stop_words_in_text + other_words

def getStart(words, r=1, BaFI=True, chosen=[]): #permutation repeats
    words = words.split(' ')
    for w_ in chosen:
        words.remove(w_)
    bestt = []
    resp = ''
    while len(words)>0:
        if len(words)<r: r = len(words)
        p = list(set(itertools.permutations(words, r=r)))
        best = 99999999999
        print('ROUND OF: ', len(p))
        for w in tqdm(p):
            temp = chosen[:] + list(w)
            temp_words = words[:]
            for w_ in list(w): #Used to not remove all duplicate words
                temp_words.remove(w_)
            t = ' '.join(temp + custom_sort(temp_words))
            if t in past:
                s = past[t]
            else:
                s = scorer.get_perplexity(t)
                past[t] = s
            if s < best:
                best = s
                bestt = temp[:]
                resp = t
                #print(s, t)
                additions = list(w)
        for w_ in additions:
            words.remove(w_)
        chosen = bestt[:]
        #print(best, resp)
        if BaFI: break #Break at First Iteration
    print(best, resp)
    return resp
```

```python
perms = {0:3, 1:3, 2:2, 3:2, 4:1, 5:1}
dBaFI = {0:False, 1:False, 2:False, 3:True, 4:False, 5:False}
dChosen = {0:[], 1:['reindeer'], 2:[], 3:[], 4:[], 5:[]}

for i in range(6):
    df.at[i, 'text'] = getStart(df['text'][i], perms[i], dBaFI[i], dChosen[i])

df['score'] = df['text'].map(lambda x: scorer.get_perplexity(x))
df.to_csv("submission.csv", index=False)
print(np.mean(df['score']))
df['score']
```

# 🛠️ A Little Optimization

```python
used = []

def part_perm_brutem(st, start=0, end=3, skips=1, best=100):
    global past, used
    bestt = st
    #best = scorer.get_perplexity(st)
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
        #add_new = False
        if t in past:  #check only once
            s = past[t]
        else:
            s =  scorer.get_perplexity(t)
            past[t] = s
            #add_new = True
        if s <= best and t not in used: # and add_new:
            used.append(t)
            print("New Score: ", s, t)
            best = s
            bestt = t
    return bestt, best
```

```python
perms = {0:7, 1:6, 2:5, 3:6, 4:6, 5:4}
dbest = {0:0.0, 1:0.0, 2:0.0, 3:0.0, 4:0.0, 5:0.0}

for i in range(6):
    bestt = df['text'][i]
    best = scorer.get_perplexity(bestt) + dbest[i]
    l = len(bestt.split(' '))
    for p in range(2, perms[i]):
        for start in tqdm(range(0,l-p+1)):
            bestt, best = part_perm_brutem(bestt, start, start+p, 1, best)
    df.at[i, 'text'] = bestt

df['score'] = df['text'].map(lambda x: scorer.get_perplexity(x))
df.to_csv("submission.csv", index=False)
print(np.mean(df['score']))
df['score']
```

# 💾 Save your GPUs hard work

```python
#import pickle

#with open('past0.pickle', 'wb') as f:
#    pickle.dump(past, f, protocol=pickle.HIGHEST_PROTOCOL)
```

# 🆕 Public Scores - Enjoy!

```python
t = """reindeer mistletoe elf gingerbread family advent scrooge chimney fireplace ornament
reindeer sleep walk the night and drive mistletoe scrooge laugh chimney jump elf bake gingerbread family give advent fireplace ornament
sleigh yuletide beard carol cheer chimney decorations gifts grinch holiday holly jingle magi naughty nice nutcracker ornament polar workshop stocking
sleigh of the magi yuletide cheer is unwrap gifts and eat cheer holiday decorations holly jingle relax sing carol visit workshop grinch naughty nice chimney stocking ornament nutcracker polar beard
from and of to the as in that it we with not you have milk chocolate candy peppermint eggnog cookie fruitcake toy doll game puzzle greeting card wrapping paper bow wreath poinsettia snowglobe candle fireplace wish dream hope believe wonder night star angel peace joy season merry hohoho kaggle workshop
from and and as we and have the in is it of not that the to with you advent card angel bake beard believe bow candy candle carol cheer cheer chocolate chimney cookie decorations doll dream drive eat eggnog family fireplace fireplace chimney fruitcake game gifts give gingerbread greeting grinch holiday holly hohoho hope jingle jump joy kaggle laugh magi merry milk mistletoe naughty nice night night elf nutcracker ornament ornament of the wrapping paper peace peppermint polar poinsettia puzzle reindeer relax scrooge season sing sleigh sleep snowglobe star stocking toy unwrap visit walk wish wonder workshop workshop wreath yuletide"""

df['text'] = t.split('\n')
df.to_csv("submission.csv", index=False)
```

# 🎨Any Patterns

```python
from IPython.display import display, HTML
import PIL

colors = sorted([code for name, code in PIL.ImageColor.colormap.items() if name not in ['black']], reverse=True)
color_map = {w:colors[i] for i, w in enumerate(set((' '.join(t.split('\n'))).split(' ')))}
def spanit(w, c):
    return '<span style="font-weight:bold;color:' + str(c) + '">' + str(w) + '</span>'
html = '<div style="background-color: #000000; width: 700px; height: 150px; overflow-x: auto; overflow-y: auto; text-align: left; white-space: nowrap; padding: 5px;">'
for l in t.split('\n'):
    html += ' '.join([spanit(w, color_map[w]) for w in l.split(' ')]) + '<br/>\n'
html += '</div>'
display(HTML(html))
```

```python
html = '<div style="background-color: #000000; width: 700px; height: 550px; overflow-x: auto; overflow-y: auto; text-align: left; padding: 5px;">'
for l in t.split('\n'):
    html += ' '.join([spanit(w, color_map[w]) for w in l.split(' ')]) + '<br/><br/>\n'
html += '</div>'
display(HTML(html))
```

```python
dID = {83:0, 135:1, 149:2, 197:3, 302:4, 636:5}
df = pd.DataFrame.from_dict(past, orient='index').reset_index()
df.columns = ['text','score']
df['id'] = df['text'].map(len)
df['id'] = df['id'].map(dID)
df.head(1)

df0 = df[df['id']==0]
df1 = df[df['id']==1]
df2 = df[df['id']==2]
df3 = df[df['id']==3]
df4 = df[df['id']==4]
df5 = df[df['id']==5]

paths = []

for i in range(3,4):
    dfx = eval('df'+str(i))
    print("ROW: ", i)
    dfx = dfx.sort_values(['score']).reset_index(drop=True)
    for r in range(100):
        #if len(str(dfx['score'][r])) < 10:
        paths.append(dfx['text'][r])
len(paths)
```

```python
colors = sorted([code for name, code in PIL.ImageColor.colormap.items() if name not in ['black']], reverse=True)
color_map = {w:colors[i] for i, w in enumerate(set((' '.join(t.split('\n'))).split(' ')))}
def spanit(w, c):
    return '<td><span style="font-weight:bold;color:' + str(c) + '">' + str(w) + '</span></td>'

html = '<table style="background-color: #000000; width: 2000px; height: 100%; overflow-x: auto; overflow-y: auto; text-align: left; white-space: nowrap; padding: 5px;">'
html += '<tr>' + ' '.join([spanit(i, '#ffffff') for i in range(len(paths[0].split(' '))+1)]) + '</tr>\n'
for l in paths:
    html += '<tr>' + str(spanit(past[l], '#ffffff')) + ' '.join([spanit(w, color_map[w]) for w in l.split(' ')]) + '</tr>\n'
html += '</table>'

f = open('test.html','w')
f.write(html)
f.close()

display(HTML(html))
```

# Ｈ𝐀𝑷𝑷𝓎 🇰𝗮𝘨𝘨🇱𝖎Ｎɢ 💯