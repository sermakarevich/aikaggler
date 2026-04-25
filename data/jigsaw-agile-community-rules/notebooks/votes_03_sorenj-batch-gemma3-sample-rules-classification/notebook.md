# Batch Gemma3 Sample Rules Classification

- **Author:** Jeffrey Sorensen
- **Votes:** 472
- **Ref:** sorenj/batch-gemma3-sample-rules-classification
- **URL:** https://www.kaggle.com/code/sorenj/batch-gemma3-sample-rules-classification
- **Last run:** 2025-07-22 20:30:17.723000

---

```python
import kagglehub
import more_itertools
import pandas as pd
import torch
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
```

```python
GEMMA_PATH = kagglehub.model_download("google/gemma-3/transformers/gemma-3-1b-it")
processor = AutoTokenizer.from_pretrained(GEMMA_PATH)

# Determine if CUDA (GPU) is available
device = "cuda" if torch.cuda.is_available() else "cpu"

model = AutoModelForCausalLM.from_pretrained(GEMMA_PATH).to(device)
print(model)
```

```python
test_data = pd.read_csv('/kaggle/input/jigsaw-agile-community-rules-enforcement/test.csv')
```

```python
def prompt(input: pd.Series):
    return """<start_of_turn>user
You are a really experienced moderator for the subreddit /r/%s. Your job
is to determine if the following reported comments violates the rule:
%s

%s
Decision:
True

%s
Decision:
False

%s
Decision:
False

%s
Decision:
True

%s
<end_of_turn>
<start_of_turn>model\n""" % (
    input['subreddit'],
    input['rule'],
    "\n".join(["| " + x for x in input['positive_example_1'].split('\n')]),
    "\n".join(["| " + x for x in input['negative_example_1'].split('\n')]),
    "\n".join(["| " + x for x in input['negative_example_2'].split('\n')]),
    "\n".join(["| " + x for x in input['positive_example_2'].split('\n')]),
    "\n".join(["| " + x for x in input['body'].split('\n')])    
)
```

```python
token_ids = [processor.get_vocab()[word] for word in ['True', 'False']]
if any(token_id == processor.get_vocab()['<unk>'] for token_id in token_ids):
      raise ValueError('One of the target classes is not in the vocabulary.')
```

```python
responses = []
for batch in more_itertools.batched(test_data.iterrows(), 4):
    prompts = [prompt(x) for _, x in batch]
    pre = processor(text=prompts, return_tensors="pt", padding=True, truncation=True,
                    max_length=512).to(device)
    with torch.no_grad():
      outputs = model(**pre)
    logits = outputs.logits[:, -1, token_ids]  
    probabilities = torch.softmax(logits, dim=-1)
    responses.extend(probabilities[:, 0].tolist())
```

```python
my_submission = pd.DataFrame({
    'row_id': test_data['row_id'],
    'rule_violation': responses
})
```

```python
my_submission.to_csv('submission.csv', index=False)
```