# Deberta Starter - CV 0.930+

- **Author:** Chris Deotte
- **Votes:** 371
- **Ref:** cdeotte/deberta-starter-cv-0-930
- **URL:** https://www.kaggle.com/code/cdeotte/deberta-starter-cv-0-930
- **Last run:** 2025-07-13 01:05:32.963000

---

# Deberta Starter
This notebook is a Deberta starter for Kaggle's "MAP - Charting Student Math Misunderstandings" competition. Currently the top public LB scoring notebooks use tf-idf. However using encoder LLM like Deberta can achieve a better CV score!

This notebook demonstrates that using a small simple `Deberta-v3-xsmall` achieves CV 0.920+. If we use `Deberta-v3-base` instead, we achieve CV 0.930+ wow! Furthermore we can improve the text input to our LLM and improve CV score even more! We can also train 5-KFold or train with 100% data to boost LB score. (For deberta-small and xsmall `epochs=10` works best, for deberta-base use `epochs=6`).

This notebook does both training and inference. However the most efficient way to submit to Kaggle competitions is to train models in one Kaggle notebook or train offline. Then save the model folder to a Kaggle dataset. Then make a second notebook that only loads the saved model and infers the test data. We submit the second notebook to the competition.

# Config

```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

VER=1
model_name = "/kaggle/input/huggingfacedebertav3variants/deberta-v3-xsmall"
EPOCHS = 10

DIR = f"ver_{VER}"
os.makedirs(DIR, exist_ok=True)
```

# Load Train

```python
import pandas as pd, numpy as np
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
train = pd.read_csv('/kaggle/input/map-charting-student-math-misunderstandings/train.csv')
train.Misconception = train.Misconception.fillna('NA')
train['target'] = train.Category+":"+train.Misconception
train['label'] = le.fit_transform(train['target'])
n_classes = len(le.classes_)
print(f"Train shape: {train.shape} with {n_classes} target classes")
train.head()
```

# Powerful Feature Engineer
We engineer one feature which we will use when formatting the input text for our LLM. Consider using more feature engineering and/or modifying the input text to our LLM. There is a discussion about this feature [here][1]

[1]: https://www.kaggle.com/competitions/map-charting-student-math-misunderstandings/discussion/589400

```python
idx = train.apply(lambda row: row.Category.split('_')[0],axis=1)=='True'
correct = train.loc[idx].copy()
correct['c'] = correct.groupby(['QuestionId','MC_Answer']).MC_Answer.transform('count')
correct = correct.sort_values('c',ascending=False)
correct = correct.drop_duplicates(['QuestionId'])
correct = correct[['QuestionId','MC_Answer']]
correct['is_correct'] = 1

train = train.merge(correct, on=['QuestionId','MC_Answer'], how='left')
train.is_correct = train.is_correct.fillna(0)
```

# Question EDA
The train.csv has 15 multiple choice math questions. Below we display each of the questions and the 4 MC choices. The choices are sorted from (A) most popular selected to (D) least popular selected.

```python
from IPython.display import display, Math, Latex

# GET ANSWER CHOICES
tmp = train.groupby(['QuestionId','MC_Answer']).size().reset_index(name='count')
tmp['rank'] = tmp.groupby('QuestionId')['count'].rank(method='dense', ascending=False).astype(int) - 1
tmp = tmp.drop('count',axis=1)
tmp = tmp.sort_values(['QuestionId','rank'])

# DISPLAY QUESTION AND ANSWER CHOICES
Q = tmp.QuestionId.unique()
for q in Q:
    question = train.loc[train.QuestionId==q].iloc[0].QuestionText
    choices = tmp.loc[tmp.QuestionId==q].MC_Answer.values
    labels="ABCD"
    choice_str = " ".join([f"({labels[i]}) {choice}" for i, choice in enumerate(choices)])
    
    print()
    display(Latex(f"QuestionId {q}: {question}") )
    display(Latex(f"MC Answers: {choice_str}"))
```

# Train with Transformers
We will train our Deberta model using Transformers library.

```python
import torch
from transformers import DebertaTokenizer, DebertaForSequenceClassification, TrainingArguments, Trainer
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from datasets import Dataset
import numpy as np

tokenizer = AutoTokenizer.from_pretrained(model_name)
MAX_LEN = 256
```

# Tokenize Train Data
First we must tokenizer our data. Before we can tokenizer, we need to decide how to convert the multiple text columns into a single prompt. We will show our model the `QuestionText`, then the `MC_Answer` response, then use our `powerful feature engineer` to say whether this answer is `correct or incorrect`. Finally we will show our LLM the `StudentExplanation`.

Consider changing the prompt below. Modifying the prompt can significantly improve our CV score!

```python
def format_input(row):
    x = "This answer is correct."
    if not row['is_correct']:
        x = "This is answer is incorrect."
    return (
        f"Question: {row['QuestionText']}\n"
        f"Answer: {row['MC_Answer']}\n"
        f"{x}\n"
        f"Student Explanation: {row['StudentExplanation']}"
    )

train['text'] = train.apply(format_input,axis=1)
print("Example prompt for our LLM:")
print()
print( train.text.values[0] )
```

```python
lengths = [len(tokenizer.encode(t, truncation=False)) for t in train["text"]]
import matplotlib.pyplot as plt

plt.hist(lengths, bins=50)
plt.title("Token Length Distribution")
plt.xlabel("Number of tokens")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()
```

```python
L = (np.array(lengths)>MAX_LEN).sum()
print(f"There are {L} train sample(s) with more than {MAX_LEN} tokens")
np.sort( lengths )
```

# Create 20% Validation Subset

```python
# Split into train and validation sets
train_df, val_df = train_test_split(train, test_size=0.2, random_state=42)

# Convert to Hugging Face Dataset
COLS = ['text','label']
train_ds = Dataset.from_pandas(train_df[COLS])
val_ds = Dataset.from_pandas(val_df[COLS])
```

```python
# Tokenization function
def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=256)

train_ds = train_ds.map(tokenize, batched=True)
val_ds = val_ds.map(tokenize, batched=True)

# Set format for PyTorch
columns = ['input_ids', 'attention_mask', 'label']
train_ds.set_format(type='torch', columns=columns)
val_ds.set_format(type='torch', columns=columns)
```

# Initialize and Train Model
Let's initialize and train our model with HuggingFace trainer. We also define a custom metric of MAP@3 which is the competition metric.

```python
from transformers import DebertaV2ForSequenceClassification

model = DebertaV2ForSequenceClassification.from_pretrained(
    model_name,
    num_labels=n_classes
)
```

```python
training_args = TrainingArguments(
    output_dir = f"./{DIR}",
    do_train=True,
    do_eval=True,
    eval_strategy="steps",
    save_strategy="steps", #no for no saving 
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=16*2,
    per_device_eval_batch_size=32*2,
    learning_rate=5e-5,
    logging_dir="./logs",
    logging_steps=50,
    save_steps=200,
    eval_steps=200,
    save_total_limit=1,
    metric_for_best_model="map@3",
    greater_is_better=True,
    load_best_model_at_end=True,
    report_to="none",
)
```

```python
# CUSTOM MAP@3 METRIC

from sklearn.metrics import average_precision_score

def compute_map3(eval_pred):
    logits, labels = eval_pred
    probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1).numpy()
    
    top3 = np.argsort(-probs, axis=1)[:, :3]  # Top 3 predictions
    match = (top3 == labels[:, None])

    # Compute MAP@3 manually
    map3 = 0
    for i in range(len(labels)):
        if match[i, 0]:
            map3 += 1.0
        elif match[i, 1]:
            map3 += 1.0 / 2
        elif match[i, 2]:
            map3 += 1.0 / 3
    return {"map@3": map3 / len(labels)}
```

```python
# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tokenizer,
    compute_metrics=compute_map3,
)

trainer.train()
```

# Save Model
This is how to save the files we need to upload to a Kaggle dataset for inference

```python
import joblib

trainer.save_model(f"{DIR}/best")
_ = joblib.dump(le, f"{DIR}/label_encoder.joblib")
```

# Load Model
This is how we load a Kaggle dataset for our inference notebook

```python
tokenizer = AutoTokenizer.from_pretrained(f"{DIR}/best")
model = DebertaV2ForSequenceClassification.from_pretrained(
    f"{DIR}/best",
    num_labels=n_classes  
)
training_args = TrainingArguments(report_to="none")
trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args)
le = joblib.load(f"{DIR}/label_encoder.joblib")
```

# Load and Predict Test 
We load test data, then engineer our powerful feature, then create prompt, then tokenize. Finally we infer test and generate probabilities for all 65 multi-classes.

```python
test = pd.read_csv('/kaggle/input/map-charting-student-math-misunderstandings/test.csv')
print( test.shape )
test.head()
```

```python
test = test.merge(correct, on=['QuestionId','MC_Answer'], how='left')
test.is_correct = test.is_correct.fillna(0)

test['text'] = test.apply(format_input,axis=1)

test.head()
```

```python
ds_test = Dataset.from_pandas(test[['text']])
ds_test = ds_test.map(tokenize, batched=True)

predictions = trainer.predict(ds_test)
probs = torch.nn.functional.softmax(torch.tensor(predictions.predictions), dim=1).numpy()
```

# Create Submission CSV
We create submission.csv by converting our top3 test preds into their class names

```python
# Get top 3 predicted class indices
top3 = np.argsort(-probs, axis=1)[:, :3]   # shape: [num_samples, 3]

# Decode numeric class indices to original string labels
flat_top3 = top3.flatten()
decoded_labels = le.inverse_transform(flat_top3)
top3_labels = decoded_labels.reshape(top3.shape)

# Join 3 labels per row with space
joined_preds = [" ".join(row) for row in top3_labels]

# Save submission
sub = pd.DataFrame({
    "row_id": test.row_id.values,
    "Category:Misconception": joined_preds
})
sub.to_csv("submission.csv", index=False)
sub.head()
```