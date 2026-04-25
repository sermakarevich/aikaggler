# [sft] gemma-2-9b-it-bnb-4bit

- **Author:** hengck23
- **Votes:** 393
- **Ref:** hengck23/sft-gemma-2-9b-it-bnb-4bit
- **URL:** https://www.kaggle.com/code/hengck23/sft-gemma-2-9b-it-bnb-4bit
- **Last run:** 2025-01-17 02:49:15.363000

---

```python
try:
    import peft
except:
    !pip install transformers peft accelerate bitsandbytes \
        -U --no-index --find-links /kaggle/input/lmsys-wheel-files
    
print('pip ok!!!')
```

```python
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

from datetime import datetime
import pytz
print('LOGGING TIME OF START (SGT):',  datetime.strftime(datetime.now(pytz.timezone('Asia/Singapore')), "%Y-%m-%d %H:%M:%S"))

import numpy as np
import pandas as pd
from timeit import default_timer as timer
from concurrent.futures import ThreadPoolExecutor
import gc

from datasets import Dataset
from transformers import (
    Gemma2ForSequenceClassification,
    GemmaTokenizerFast,
)
from peft import (
    PeftModel,
)

from transformers.data.data_collator import pad_without_fast_tokenizer_warning
import torch.nn.functional as F

import torch
import transformers
import peft

print('torch', torch.__version__)
print('transformers', transformers.__version__)
print('peft', peft.__version__)


## helper ! ---
class dotdict(dict):
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)


def time_to_str(t, mode='min'):
    if mode == 'min':
        t = int(t) / 60
        hr = t // 60
        min = t % 60
        return '%2d hr %02d min' % (hr, min)

    elif mode == 'sec':
        t = int(t)
        min = t // 60
        sec = t % 60
        return '%2d min %02d sec' % (min, sec)

    else:
        raise NotImplementedError


print('import ok!!!')
```

```python
MODE = 'submit'  #'submit'
 
if MODE == 'local':
    valid_df = pd.read_parquet(f'/kaggle/input/wsdm-cup-multilingual-chatbot-arena/train.parquet')
    valid_df = valid_df.fillna('')
    valid_df.loc[:, 'label'] = valid_df['winner'].map({'model_a': 0, 'model_b': 1})
    valid_df = valid_df[0::5]  #fold-0
    valid_df = valid_df[:1000] #500

if MODE == 'submit':
    valid_df = pd.read_parquet(f'/kaggle/input/wsdm-cup-multilingual-chatbot-arena/test.parquet')
    valid_df = valid_df.fillna('')

num_valid = len(valid_df)
print('num_valid', num_valid)

#----
cfg = dotdict( 

    
    model_id= \
    '/kaggle/input/gemma-2-9b-4bit-it-unsloth/transformers/default/1/gemma-2-9b-it-4bit-unsloth_old',
       #'/kaggle/input/wsdm-hengck23-weight-01/gemma-2-9b-it-4bit',
  
    lora_dir= \
    '/kaggle/input/wsdm-hengck23-weight-01a/ensemble-fold01-3072',
    
    #'/kaggle/input/wsdm-hengck23-weight-01a/checkpoint-2453-fold1-deepspeed-sdpa-3072-extern1-fix',
    #'/kaggle/input/wsdm-hengck23-weight-01a/checkpoint-2453-deepspeed-fa2-3072-extern1-fix',
    #'/kaggle/input/wsdm-hengck23-weight-01a/checkpoint-1832-deepspeed-2048-extern1',
    #'/kaggle/input/wsdm-hengck23-weight-01a/checkpoint-1211-3072-fix',
    #'/kaggle/input/wsdm-hengck23-weight-01a/checkpoint-1211-deepspeed-fix',
    
        #'/kaggle/input/wsdm-hengck23-weight-01a/checkpoint-6055-init-p2-fix',
        #'/kaggle/input/wsdm-hengck23-weight-01a/checkpoint-3250-init-p2-fix',
        #'/kaggle/input/wsdm-hengck23-weight-01a/checkpoint-4542-s-fix',
        #'/kaggle/input/wsdm-hengck23-weight-01a/checkpoint-4100-fix',
        #'/kaggle/input/wsdm-hengck23-weight-01a/checkpoint-3800-fix',

    max_length=3072, #3072,#2048,
    max_prompt_length=1024, #1024, #512,
    batch_size=4,
)


print('config ok!!!')
```

```python
## modeling #####################################################################
def load_tokenizer(model_id):
    tokenizer = GemmaTokenizerFast.from_pretrained(model_id)
    tokenizer.add_eos_token = True
    tokenizer.padding_side = 'right'
    return tokenizer


def load_model(model_id, lora_dir, device):
    model = Gemma2ForSequenceClassification.from_pretrained(
        model_id,
        device_map=device,
        use_cache=False,
    )
    model = PeftModel.from_pretrained(model, lora_dir)
    model = model.eval()
    return model


tokenizer = load_tokenizer(cfg.model_id)
multi_model = [
    load_model(cfg.model_id, cfg.lora_dir, 'cuda:0'),
    load_model(cfg.model_id, cfg.lora_dir, 'cuda:1'),
]


print('model ok!!!')
```

```python
## dataset #####################################################################
def batch_data_process(batch):
    prompt = ['<prompt>: ' + p for p in batch['prompt']]
    response_a = ['\n\n<response_a>: ' + r_a for r_a in batch['response_a']]
    response_b = ['\n\n<response_b>: ' + r_b for r_b in batch['response_b']]
    p = tokenizer(prompt, add_special_tokens=False)['input_ids']
    a = tokenizer(response_a, add_special_tokens=False)['input_ids']
    b = tokenizer(response_b, add_special_tokens=False)['input_ids']
    batch_size = len(p)

    tokenized = {
        'input_ids': [],
        'attention_mask': [],
    }
    for i in range(batch_size):
        if len(p[i]) > cfg.max_prompt_length:
            p[i] = p[i][-cfg.max_prompt_length:]
        reponse_length = (cfg.max_length - len(p[i])-2) // 2  ##-2
        input_ids = [tokenizer.bos_token_id] + p[i] + a[i][-reponse_length:] + b[i][-reponse_length:] + [
            tokenizer.eos_token_id]
        attention_mask = [1] * len(input_ids)
        tokenized['input_ids'].append(input_ids)
        tokenized['attention_mask'].append(attention_mask)
    return tokenized


def one_data_process(row):
    prompt = '<prompt>: ' + row['prompt']
    response_a = '\n\n<response_a>: ' + row['response_a']
    response_b = '\n\n<response_b>: ' + row['response_b']
    p = tokenizer(prompt, add_special_tokens=False)['input_ids']
    a = tokenizer(response_a, add_special_tokens=False)['input_ids']
    b = tokenizer(response_b, add_special_tokens=False)['input_ids']

    if len(p) > cfg.max_prompt_length:
        p = p[-cfg.max_prompt_length:]
    reponse_length = (cfg.max_length - len(p)-2) // 2  ##-2
    input_ids = \
        [tokenizer.bos_token_id] + \
        p + a[-reponse_length:] + b[-reponse_length:] + \
        [tokenizer.eos_token_id]
    length = len(input_ids)
    attention_mask = [1] * length
    return input_ids, attention_mask, length


valid_df[['input_ids', 'attention_mask', 'length']] = \
    valid_df.apply(one_data_process, axis=1, result_type='expand')
valid_df = valid_df.sort_values('length', ascending=False).reset_index(drop=True)
multi_df = [
    valid_df[0::2].reset_index(drop=True),
    valid_df[1::2].reset_index(drop=True),
]
valid_df = pd.concat(multi_df).reset_index(drop=True)

print('data ok!!!')
```

```python
## inference #####################################################################

def make_submission(valid_df, probability):
    # id,winner
    predict = probability.argmax(-1)
    winner =[
        {0:'model_a',1:'model_b',}[p] for p in predict
    ]
    submit_df = pd.DataFrame({
        'id' : valid_df['id'],
        'winner': winner,
    })
    return submit_df

def do_infer(model, valid_df, name=''):
    num_valid = len(valid_df)
    start_timer = timer()
    probability = []
    for i in range(0, num_valid, cfg.batch_size):
        print('\r', f'{name}: {i}', time_to_str(timer() - start_timer, 'min'), end='', flush=True)

        B = min(i +  cfg.batch_size, num_valid) - i
        d = valid_df[i:i + B]
        batch = {
            'input_ids': d['input_ids'].tolist(),
            'attention_mask': d['attention_mask'].tolist(),
        }
        batch = pad_without_fast_tokenizer_warning(
            tokenizer,
            batch,  #{"input_ids": input_ids, "attention_mask": attention_mask},
            padding='longest',
            pad_to_multiple_of=None,
            return_tensors='pt',
        )
        with torch.amp.autocast('cuda', enabled=True):
            with torch.no_grad():
                output = model(
                    input_ids=batch['input_ids'].to(model.device),
                    attention_mask=batch['attention_mask'].to(model.device),
                )
                p = F.softmax(output.logits[:, :2], dim=1)
                probability.append(p.data.cpu().numpy())
                #print(probability.shape)
                #todo tta
    
    torch.cuda.empty_cache()
    print('')
    probability = np.concatenate(probability)
    return probability


if 0:
    #debug
    probability = do_infer(multi_model[0], multi_df[0], name='')
    print(probability.shape)
    exit(0)


start_timer = timer()
with ThreadPoolExecutor(max_workers=2) as executor:
    result = executor.map(do_infer, multi_model , multi_df, ('process0', 'process1'))
probability = np.concatenate(list(result))
time_taken = timer() - start_timer

submit_df = make_submission(valid_df, probability)
submit_df.to_csv('submission.csv', index=False)
np.save('probability.npy', probability)

print(submit_df)
print(probability.shape)
print(f'time for 10_000 (max  4.5 hr): {10_000/num_valid*time_taken/60/60:4.1f}')
print(f'time for 25_000 (max 12.0 hr): {25_000/num_valid*time_taken/60/60:4.1f}')
print('MODE', MODE)
print('submit ok!!!')
```

```python
try:
    multi_model[0]= multi_model[0].to('cpu')
    multi_model[1]= multi_model[1].to('cpu')
    del multi_model 
    gc.collect()
    torch.cuda.empty_cache()
except:
    pass

print('free memory ok!!!')
```

```python
if MODE=='local':
    truth   = valid_df['label'].values.astype(np.int32)
    np.save('truth.npy', truth)
    
    predict = np.argmax(probability, -1)
    acc = (truth == predict).mean()

    logp = np.log(np.clip(probability, 1e-5, 1 - 1e-5))
    logloss = truth * logp[:, 1] + (1 - truth) * logp[:, 0]
    logloss = (-logloss).mean()
    print('acc:', acc)
    print('logloss:', logloss)



'''
 #'/kaggle/input/wsdm-hengck23-weight-01a/checkpoint-3800-fix',
 process0: 248  0 hr 05 min
 process1: 248  0 hr 05 min
 
(500, 2)
time for 10_000 (max  4.5 hr):  2.0
time for 25_000 (max 12.0 hr):  4.9

acc: 0.674
logloss: 0.5757375925322995


---
(1000, 2)
 process0: 404  0 hr 13 min
 process0: 496  0 hr 13 min
acc: 0.716
logloss: 0.5659156686118804

acc: 0.696
logloss: 0.5689815965755842
'''
```