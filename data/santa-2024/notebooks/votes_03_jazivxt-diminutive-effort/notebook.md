# Diminutive Effort

- **Author:** jazivxt
- **Votes:** 373
- **Ref:** jazivxt/diminutive-effort
- **URL:** https://www.kaggle.com/code/jazivxt/diminutive-effort
- **Last run:** 2024-11-30 12:51:05.653000

---

```python
import numpy as np
import pandas as pd
from collections import Counter
from tqdm import tqdm
import random, pickle, math, warnings
import itertools
#warnings.simplefilter('ignore')

p = '/kaggle/input/santa-2024/sample_submission.csv'
df = pd.read_csv(p) # 	id 	text
print(df['text'].map(lambda x: len(str(x).split(' '))).values)
```

```python
import transformers, torch
import gc, os, logging
from math import exp
from typing import List, Optional, Union

os.environ['OMP_NUM_THREADS'] = '2'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
PAD_TOKEN_LABEL_ID = torch.nn.CrossEntropyLoss().ignore_index
DEVICE = torch.device('cuda')
MODEL_PATH = "/kaggle/input/gemma-2/transformers/gemma-2-9b/2"

class PerplexityCalculator:
    def __init__(self,):
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_PATH)
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            MODEL_PATH, device_map="auto",#cuda:0
            torch_dtype=torch.float16,)
        self.loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        self.model.eval()

    def get_perplexity(self, input_texts: Union[str, List[str]],) -> Union[float, List[float]]:
        single_input = isinstance(input_texts, str)
        input_texts = [input_texts] if single_input else input_texts
        loss_list = []
        with torch.no_grad():
            for text in input_texts:
                text_with_special = f"{self.tokenizer.bos_token}{text}{self.tokenizer.eos_token}"
                model_inputs = self.tokenizer(text_with_special, return_tensors='pt', add_special_tokens=False,)
                model_inputs = {k: v.to(DEVICE) for k, v in model_inputs.items()}
                output = self.model(**model_inputs, use_cache=False)
                logits = output['logits']
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = model_inputs['input_ids'][..., 1:].contiguous()
                loss = self.loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1))
                sequence_loss = loss.sum() / len(loss)
                loss_list.append(sequence_loss.cpu().item())
        ppl = [exp(i) for i in loss_list]
        return ppl[0] if single_input else ppl

scorer = PerplexityCalculator()
```

## Brute Force First Permutation

```python
%%time

def part_perm_brutel(st, psize=3, skips=1000):
    bestt = st
    best = scorer.get_perplexity(st)
    part = st.split(' ')[-psize:]
    st = ' '.join(st.split(' ')[:-psize]) + ' '
    p = list(itertools.permutations(part))
    for i in range(0, len(p), skips):
        t = st + ' '.join(list(p[i]))
        s =  scorer.get_perplexity(t)
        if s < best:
            print("New Score: ", s)#, t)
            best = s
            bestt = t
    return bestt
    
def part_perm_brutef(st, psize=3, skips=1000):
    bestt = st
    best = scorer.get_perplexity(st)
    part = st.split(' ')[:psize]
    st =  ' ' + ' '.join(st.split(' ')[psize:])
    p = list(itertools.permutations(part))
    for i in range(0, len(p), skips):
        t = ' '.join(list(p[i])) + st
        s =  scorer.get_perplexity(t)
        if s < best:
            print("New Score: ", s)#, t)
            best = s
            bestt = t
    return bestt

i = 0
p = 6
print("ROW :", i, 'PERMUTATION SIZE: ', p)
bestt = df['text'][i]
bestt = part_perm_brutef(bestt, p, 1)
bestt = part_perm_brutel(bestt, p, 1)
print(bestt)
print("Lets Flip...")
#Lets Flip it and repeat
bestt = list(bestt.split(' '))
bestt.reverse()
bestt = ' '.join(bestt)
bestt = part_perm_brutef(bestt, p, 1)
bestt = part_perm_brutel(bestt, p, 1)
df.at[0, 'text'] = bestt
print(bestt)
```

```python
df.at[0, 'text'] = bestt
df.to_csv("submission.csv", index=False)
```

## Simulated Annealing

## Shift Over

```python
#https://www.kaggle.com/code/richolson/santa-claude-s-approach-simulated-annealing

def simulated_annealing_optimize(text: str, temp_start=6.0, temp_end=0.3, cooling_rate=0.98, steps_per_temp=4, verbose=False):
    words = text.split()
    current = words.copy()
    current_score = scorer.get_perplexity(' '.join(current))
    if math.isnan(current_score):
        while True:
            current = words.copy()
            random.shuffle(current)
            current_score = scorer.get_perplexity(' '.join(current))
            if not math.isnan(current_score):
                break
    best = current.copy()
    best_score = current_score
    temp = temp_start
    print(f"Start Temperature: {temp:.2f}, Initial score: {current_score:.2f}")

    while temp > temp_end:
        for _ in range(steps_per_temp):  # Do multiple attempts at each temperature
            i, j = random.sample(range(len(words)), 2)
            neighbor = current.copy()
            neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
            neighbor_score = scorer.get_perplexity(' '.join(neighbor))
            if math.isnan(neighbor_score):
                continue
            delta = neighbor_score - current_score
            if delta < 0 or random.random() < math.exp(-delta / temp):
                current = neighbor
                current_score = neighbor_score
                if current_score < best_score:
                    best = current.copy()
                    best_score = current_score
                    print(">", end="")
                else: print("<", end="")
            else:print("-", end="")
        temp *= cooling_rate
        if verbose: print(f"\nTemperature: {temp:.2f}, Current score: {current_score:.2f}")
    print(f"\nFinal score: {best_score:.2f}")
    return ' '.join(best), best_score

#https://www.kaggle.com/code/richolson/santa-claude-s-approach-simulated-annealing
samples = pd.read_csv("submission.csv")
submission = pd.DataFrame(columns=['id', 'text'])
scores = []
for idx, row in samples.iterrows():
    print(f"\nProcessing sample {idx}...")
    optimized, score = simulated_annealing_optimize(row.text)
    scores.append(score)
    submission.loc[idx] = {'id': row.id, 'text': optimized}
    print(optimized)
submission.to_csv("submission.csv", index=False)
```

```python
df = pd.read_csv("submission.csv")
df['score'] = df['text'].map(lambda x: scorer.get_perplexity(x))
print(np.mean(df['score']))
df['score']
```

## Shift Over

```python
def get_best_plex(df, shift=0, rev=False, tf=False):
    print('SHIFT: ', shift, ' MEAN: ', np.mean(df['score'].values))
    for r in range(len(df)):
        if rev: shift+=1
        if len(df['text'][r].split(' ')) > shift:
            bGood = True
            while bGood == True:
                bGood = False
                t = df['text'][r].split(' ')
                if rev: shift *= -1 
                last = [t[shift]]
                del t[shift]
                t = t + last
                best = df['score'][r]
                for x in range(len(t)-1):
                    new = t[:x] + last + t[x:-1]
                    new = ' '.join(new)
                    s = scorer.get_perplexity(new)
                    if s < best: 
                        bGood = True 
                        best = s
                        df.at[r, 'score'] = s
                        df.at[r, 'text'] = new
                        print(r, x, "New Score: ", s)
                        if tf: break #take first shift
    return df

for i in range(20): #can expand to 99
    df = get_best_plex(df, i, True, True)
    df.to_csv("submission.csv", index=False)
```

```python
df = pd.read_csv("submission.csv")
df['score'] = df['text'].map(lambda x: scorer.get_perplexity(x))
print(np.mean(df['score']))
df['score']
```

```python
## One More SA Run
#https://www.kaggle.com/code/richolson/santa-claude-s-approach-simulated-annealing
samples = pd.read_csv("submission.csv")
submission = pd.DataFrame(columns=['id', 'text'])
scores = []
for idx, row in samples.iterrows():
    print(f"\nProcessing sample {idx}...")
    optimized, score = simulated_annealing_optimize(row.text)
    scores.append(score)
    submission.loc[idx] = {'id': row.id, 'text': optimized}
    print(optimized)
submission.to_csv("submission.csv", index=False)
```

```python
df = pd.read_csv("submission.csv")
df['score'] = df['text'].map(lambda x: scorer.get_perplexity(x))
print(np.mean(df['score']))
df['score']
```

## Last Shift

```python
for i in range(20): #can expand to 99
    df = get_best_plex(df, i, True, True)
    df.to_csv("submission.csv", index=False)
```

```python
for i in range(20, 50): #can expand to 99
    df = get_best_plex(df, i, True, True)
    df.to_csv("submission.csv", index=False)
```

```python
for i in range(50, 99): #can expand to 99
    df = get_best_plex(df, i, True, True)
    df.to_csv("submission.csv", index=False)
```

```python
df = pd.read_csv("submission.csv")
df['score'] = df['text'].map(lambda x: scorer.get_perplexity(x))
print(np.mean(df['score']))
df['score']
```

## Lets Polish the Edges

```python
best_output = []
p = 6
for i in range(6):
    print("ROW :", i, 'PERMUTATION SIZE: ', p)
    bestt = df['text'][i]
    bestt = part_perm_brutef(bestt, p, 1)
    bestt = part_perm_brutel(bestt, p, 1)
    best_output.append(bestt)

df['text'] = best_output
df['score'] = df['text'].map(lambda x: scorer.get_perplexity(x))
df.to_csv("submission.csv", index=False)
print(np.mean(df['score']))
df['score']
```

## Only Index 1 does not budge but we don;t talk about #1

```python
print(df['text'][1])
txt1 = df['text'][1].split(' ')
txt1.reverse()
df.at[1, 'text'] = ' '.join(txt1)
print(df['text'][1])
```

```python
df['score'] = df['text'].map(lambda x: scorer.get_perplexity(x))
df.to_csv("submission.csv", index=False)
print(np.mean(df['score']))
df['score']
```

```python
## One More SA Run
#https://www.kaggle.com/code/richolson/santa-claude-s-approach-simulated-annealing
samples = pd.read_csv("submission.csv")
submission = pd.DataFrame(columns=['id', 'text'])
scores = []
for idx, row in samples.iterrows():
    print(f"\nProcessing sample {idx}...")
    optimized, score = simulated_annealing_optimize(row.text)
    scores.append(score)
    submission.loc[idx] = {'id': row.id, 'text': optimized}
    print(optimized)
submission.to_csv("submission.csv", index=False)
```

```python
df = pd.read_csv("submission.csv")
df['score'] = df['text'].map(lambda x: scorer.get_perplexity(x))
print(np.mean(df['score']))
df['score']
```

```python
for i in range(20): #can expand to 99
    df = get_best_plex(df, i, True, True)
    df.to_csv("submission.csv", index=False)
```

## Its locking up tight now - 4 is the only gift that keeps on giving

```python
for i in range(20): #can expand to 99
    df = get_best_plex(df, i, True, True)
    df.to_csv("submission.csv", index=False)
```

## #5 has more to give but it will be chased later, lets push #1 now

```python
p = 7
i = 1
print("ROW :", i, 'PERMUTATION SIZE: ', p)
bestt = df['text'][i]
bestt = part_perm_brutef(bestt, p, 1)
bestt = part_perm_brutel(bestt, p, 1)
df.at[i, 'text'] = bestt
print(bestt)

df['score'] = df['text'].map(lambda x: scorer.get_perplexity(x))
df.to_csv("submission.csv", index=False)
print(np.mean(df['score']))
df['score']
```

```python
for i in range(5,15): #lets hit the mids here
    df = get_best_plex(df, i, True, True)
    df.to_csv("submission.csv", index=False)
```

## best_output = []
p = 6
for i in range(6):
    print("ROW :", i, 'PERMUTATION SIZE: ', p)
    bestt = df['text'][i]
    bestt = part_perm_brutef(bestt, p, 1)
    bestt = part_perm_brutel(bestt, p, 1)
    best_output.append(bestt)

df['text'] = best_output
df['score'] = df['text'].map(lambda x: scorer.get_perplexity(x))
df.to_csv("submission.csv", index=False)
print(np.mean(df['score']))
df['score']Lets squeeze everything out of #3-#5

```python
for i in range(15,50): #lets hit the mids here
    df = get_best_plex(df, i, True, True)
    df.to_csv("submission.csv", index=False)
```

```python
for i in range(50,99): #lets hit the mids here
    df = get_best_plex(df, i, True, True)
    df.to_csv("submission.csv", index=False)
```

```python
## One More SA Run
#https://www.kaggle.com/code/richolson/santa-claude-s-approach-simulated-annealing
samples = pd.read_csv("submission.csv")
submission = pd.DataFrame(columns=['id', 'text'])
scores = []
for idx, row in samples.iterrows():
    print(f"\nProcessing sample {idx}...")
    optimized, score = simulated_annealing_optimize(row.text)
    scores.append(score)
    submission.loc[idx] = {'id': row.id, 'text': optimized}
    print(optimized)
submission.to_csv("submission.csv", index=False)
```

## Thats all for now
* Below is the updated solution if you got this far :)

```python
t = """reindeer mistletoe elf gingerbread family advent scrooge chimney fireplace ornament
reindeer mistletoe elf scrooge gingerbread ornament advent family fireplace chimney sleep drive walk jump laugh give and bake the night
magi yuletide cheer grinch carol holiday holly jingle naughty nice nutcracker ornament stocking chimney sleigh polar beard workshop gifts decorations
sleigh of the magi yuletide cheer is unwrap gifts and decorations ornament holly jingle workshop chimney stocking carol sing visit grinch naughty nice nutcracker polar beard eat relax holiday cheer
wreath candle night chocolate milk peppermint candy fruitcake eggnog poinsettia toy doll game puzzle hohoho season wish dream believe hope peace joy wonder merry greeting card wrapping paper bow star angel snowglobe cookie fireplace kaggle workshop the of and from to as in that it we with not you have
jingle beard fruitcake yuletide eggnog scrooge mistletoe poinsettia snowglobe holly wreath gingerbread cookie nutcracker magi angel star merry and the season of joy and wonder and peace to you from the family of the grinch with holiday cheer that it is not as cheer unwrap gifts laugh hohoho sing carol jump in sleigh drive reindeer visit chimney chimney elf naughty nice eat bake sleep relax chocolate milk peppermint candy ornament ornament stocking fireplace fireplace advent candle wish dream hope believe give greeting card wrapping paper bow decorations toy doll puzzle game night night walk polar kaggle workshop we have workshop"""

print(t.split('\n'))
```