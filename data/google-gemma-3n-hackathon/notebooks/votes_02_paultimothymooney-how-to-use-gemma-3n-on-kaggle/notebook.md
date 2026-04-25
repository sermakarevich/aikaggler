# How to use Gemma-3n on Kaggle

- **Author:** Paul Mooney
- **Votes:** 337
- **Ref:** paultimothymooney/how-to-use-gemma-3n-on-kaggle
- **URL:** https://www.kaggle.com/code/paultimothymooney/how-to-use-gemma-3n-on-kaggle
- **Last run:** 2025-07-23 16:01:08.610000

---

```python
!pip install timm==1.0.17
!pip install transformers==4.53.2
```

# Download Gemma-3n

```python
import kagglehub

GEMMA_PATH = kagglehub.model_download("google/gemma-3n/transformers/gemma-3n-e2b-it")
```

# Use Gemma-3n for text-generation

```python
import transformers
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

tokenizer = AutoTokenizer.from_pretrained(GEMMA_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(GEMMA_PATH, trust_remote_code=True)
prompt = "Why are there so many Geese on Kaggle?"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
generation_config = GenerationConfig(max_new_tokens=150, do_sample=True, temperature=0.7)
outputs = model.generate(**inputs, generation_config=generation_config)
result = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

```python
print(result)
```

# Use Gemma-3n for image-understanding

```python
from IPython.display import Image
IMAGE_URL="https://storage.googleapis.com/kaggle-media/competitions/question_goose.png"
Image(url=IMAGE_URL,height=250,width=250)
```

```python
from transformers import AutoProcessor, AutoModelForImageTextToText

processor = AutoProcessor.from_pretrained(GEMMA_PATH)
model = AutoModelForImageTextToText.from_pretrained(GEMMA_PATH, torch_dtype="auto", device_map="auto")

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": IMAGE_URL},
            {"type": "text", "text": "Describe this image in detail."}
        ]
    }
]

inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt"
).to(model.device, dtype=model.dtype)
input_len = inputs["input_ids"].shape[-1]

outputs = model.generate(**inputs, max_new_tokens=512, disable_compile=True)
text = processor.batch_decode(
    outputs[:, input_len:],
    skip_special_tokens=True,
    clean_up_tokenization_spaces=True
)
```

```python
print(text[0])
```