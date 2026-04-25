# Testing Gemma 3n like a pro 

- **Author:** Gabriel Preda
- **Votes:** 62
- **Ref:** gpreda/testing-gemma-3n-like-a-pro
- **URL:** https://www.kaggle.com/code/gpreda/testing-gemma-3n-like-a-pro
- **Last run:** 2025-07-03 18:25:43.980000

---

# Introduction

This notebook presents a functional prototype built with Gemma 3n, Google’s latest on-device, multimodal AI model. The goal: use compact, private, offline-ready AI to solve a real-world problem. Full demo, code, and technical details follow.


## What model we will test?

We will experiment with the **Gemma 3n 2B it** (instruction tunning).

## What is new in the current version?

- Replaced the model initialization for text-only input.
- Changed the code accordingly for the functions used to query the model.
- Switched to use GPU
- Multimodal input content (text + image) 


## What are the key features of Gemma 3n?

The key features of this new model from Google are:

1. On-Device Performance
Optimized for mobile and edge devices, Gemma 3n delivers real-time AI with minimal memory usage. The 5B and 8B models run like 2B and 4B models, thanks to innovations like Per-Layer Embeddings (PLE).

2. Mix’n’Match Model Scaling
A single model can act as multiple: the 4B version includes a 2B submodel, enabling dynamic tradeoffs between performance and efficiency. Developers can also create custom-sized submodels tailored to specific tasks.

3. Privacy-First and Offline-Ready
Gemma 3n runs entirely on-device, ensuring user data never leaves the device. This makes it ideal for privacy-sensitive applications and for use in low- or no-connectivity environments.

4. Multimodal Understanding
Supports text, image, audio, and enhanced video input, enabling powerful applications like voice interfaces, transcription, translation, visual recognition, and more—all locally.

5. Multilingual Proficiency
Strong performance across major global languages including Japanese, German, Korean, Spanish, and French, expanding access and inclusivity.



## What we will test?

To highlight the features of this new model, we will do the following experiments:
- Verify the historical knowledge
- Ask questions about pop culture
- Check the math with counting, algebra, and some geometry
- Verify the multilanguage skills - with Romanian, Albanian, Japanese, Chinese, and French.

# Prepare the model

## Install prerequisites

```python
!pip install timm --upgrade
!pip install accelerate
!pip install git+https://github.com/huggingface/transformers.git
```

## Import packages

```python
from time import time
import kagglehub
import transformers
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from transformers import AutoProcessor, AutoModelForImageTextToText
```

## Load the model

```python
GEMMA_PATH = kagglehub.model_download("google/gemma-3n/transformers/gemma-3n-e2b-it")
```

```python
tokenizer = AutoTokenizer.from_pretrained(GEMMA_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(GEMMA_PATH, trust_remote_code=True)
```

## Test the model with a simple prompt

```python
prompt = """What is the France capital?"""
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
generation_config = GenerationConfig(max_new_tokens=150, do_sample=True, temperature=0.7)
outputs = model.generate(**inputs, generation_config=generation_config)
result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(result)
```

Let's wrap this inside a function.

```python
def query_model(prompt, max_new_tokens=32):
    start_time = time()
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    generation_config = GenerationConfig(max_new_tokens=150, do_sample=True, temperature=0.7)
    outputs = model.generate(**inputs, generation_config=generation_config)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    total_time = round(time() - start_time, 2)
    response = result.split(prompt)[-1]
    return response, total_time
```

```python
prompt = "Quelle est la capitale de la France?"
response, total_time = query_model(prompt, max_new_tokens=16)
print(f"Execution time: {total_time}")
print(f"Question: {prompt}")
print(f"Response: {response}")
```

# Test the model 


## Let's start with some history questions

```python
prompt = "When started WW2?"
response, total_time = query_model(prompt, max_new_tokens=32)
print(f"Execution time: {total_time}")
print(f"Question: {prompt}")
print(f"Response: {response}")
```

It doesn't look too right, I would like to keep it as short as possible. Let's refine a bit the function, we will add a system prompt.

## Colorize the output

```python
from IPython.display import display, Markdown

def colorize_text(text):
    for word, color in zip(["Reasoning", "Question", "Response", "Explanation", "Execution time"], ["blue", "red", "green", "darkblue",  "magenta"]):
        text = text.replace(f"{word}:", f"\n\n**<font color='{color}'>{word}:</font>**")
    return text
```

```python
prompt = "Between what years was Obama president?"
response, total_time = query_model(prompt, max_new_tokens=32)
display(Markdown(colorize_text(f"Execution time: {total_time}\n\nQuestion: {prompt}\n\nResponse: {response}")))
```

```python
prompt = "Between what years was the 30 years war?"
response, total_time = query_model(prompt, max_new_tokens=32)
display(Markdown(colorize_text(f"Execution time: {total_time}\n\nQuestion: {prompt}\n\nResponse: {response}")))
```

```python
prompt = "Between what years was the WW1?"
response, total_time = query_model(prompt, max_new_tokens=32)
display(Markdown(colorize_text(f"Execution time: {total_time}\n\nQuestion: {prompt}\n\nResponse: {response}")))
```

```python
prompt = "What year was the Lepanto battle?"
response, total_time = query_model(prompt, max_new_tokens=32)
display(Markdown(colorize_text(f"Execution time: {total_time}\n\nQuestion: {prompt}\n\nResponse: {response}")))
```

```python
prompt = "What happened in 1868 in Japan?"
response, total_time = query_model(prompt, max_new_tokens=64)
display(Markdown(colorize_text(f"Execution time: {total_time}\n\nQuestion: {prompt}\n\nResponse: {response}")))
```

Let's modify the query function to stop the generation after a maximum character number was reached.

```python
prompt = "What happened in 1868 in Japan?"
response, total_time = query_model(prompt, max_new_tokens=32)
display(Markdown(colorize_text(f"Execution time: {total_time}\n\nQuestion: {prompt}\n\nResponse: {response}")))
```

```python
prompt = "Who was the first American president?"
response, total_time = query_model(prompt, max_new_tokens=32)
display(Markdown(colorize_text(f"Execution time: {total_time}\n\nQuestion: {prompt}\n\nResponse: {response}")))
```

## Let's ask some pop culture question

```python
prompt = "In what novel the number 42 is important?"
response, total_time = query_model(prompt, max_new_tokens=32)
display(Markdown(colorize_text(f"Execution time: {total_time}\n\nQuestion: {prompt}\n\nResponse: {response}")))
```

```python
prompt = "Name the famous boyfriend of Yoko Ono."
response, total_time = query_model(prompt, max_new_tokens=32)
display(Markdown(colorize_text(f"Execution time: {total_time}\n\nQuestion: {prompt}\n\nResponse: {response}")))
```

```python
prompt = "Who was nicknamed 'The King' in music?"
response, total_time = query_model(prompt, max_new_tokens=32)
display(Markdown(colorize_text(f"Execution time: {total_time}\n\nQuestion: {prompt}\n\nResponse: {response}")))
```

```python
prompt = "What actor played Sheldon in TBBT?"
response, total_time = query_model(prompt, max_new_tokens=32)
display(Markdown(colorize_text(f"Execution time: {total_time}\n\nQuestion: {prompt}\n\nResponse: {response}")))
```

```python
prompt = "What is the maiden name of Princess of Wales?"
response, total_time = query_model(prompt, max_new_tokens=16)
display(Markdown(colorize_text(f"Execution time: {total_time}\n\nQuestion: {prompt}\n\nResponse: {response}")))
```

## Math questions

```python
prompt = "34 + 21"
response, total_time = query_model(prompt, max_new_tokens=16)
display(Markdown(colorize_text(f"Execution time: {total_time}\n\nQuestion: {prompt}\n\nResponse: {response}")))
```

```python
prompt = "49 x 27"
response, total_time = query_model(prompt, max_new_tokens=16)
display(Markdown(colorize_text(f"Execution time: {total_time}\n\nQuestion: {prompt}\n\nResponse: {response}")))
```

```python
prompt = "Brian and Sarah are brothers. Brian is 5yo, Sarah is 6 years older. How old is Sarah?"
response, total_time = query_model(prompt, max_new_tokens=40)
display(Markdown(colorize_text(f"Execution time: {total_time}\n\nQuestion: {prompt}\n\nResponse: {response}")))
```

```python
prompt = "x + 2 y = 5; y - x = 1. What are x and y? Just return x and y."
response, total_time = query_model(prompt, max_new_tokens=64)
display(Markdown(colorize_text(f"Execution time: {total_time}\n\nQuestion: {prompt}\n\nResponse: {response}")))
```

```python
prompt = "What is the total area of a sphere or radius 3? Just return the result."
response, total_time = query_model(prompt, max_new_tokens=64)
display(Markdown(colorize_text(f"Execution time: {total_time}\n\nQuestion: {prompt}\n\nResponse: {response}")))
```

```python
prompt = "A rectangle with diagonal 4 is circumscribed by a circle. What is the circle's area?"
response, total_time = query_model(prompt, max_new_tokens=256)
display(Markdown(colorize_text(f"Execution time: {total_time}\n\nQuestion: {prompt}\n\nResponse: {response}")))
```

## Multiple languages

```python
#Romanian
prompt = "Cine este Mircea Cartarescu?"
response, total_time = query_model(prompt, max_new_tokens=128)
display(Markdown(colorize_text(f"Execution time: {total_time}\n\nQuestion: {prompt}\n\nResponse: {response}")))
```

```python
#Albanian
prompt = "Kush ishte Ismail Kadare?"
response, total_time = query_model(prompt, max_new_tokens=128)
display(Markdown(colorize_text(f"Execution time: {total_time}\n\nQuestion: {prompt}\n\nResponse: {response}")))
```

```python
#Japanese
prompt = "夏目漱石とは誰ですか?"
response, total_time = query_model(prompt, max_new_tokens=128)
display(Markdown(colorize_text(f"Execution time: {total_time}\n\nQuestion: {prompt}\n\nResponse: {response}")))
```

```python
#Chinese
prompt = "马拉多纳是谁?"
response, total_time = query_model(prompt, max_new_tokens=128)
display(Markdown(colorize_text(f"Execution time: {total_time}\n\nQuestion: {prompt}\n\nResponse: {response}")))
```

```python
#French
prompt = "Qui était Marguerite Yourcenar?"
response, total_time = query_model(prompt, max_new_tokens=128)
display(Markdown(colorize_text(f"Execution time: {total_time}\n\nQuestion: {prompt}\n\nResponse: {response}")))
```

## Multimodal input

We are preparing now the function for a multi-modal input (text and image).

```python
from transformers import AutoProcessor, AutoModelForImageTextToText

processor = AutoProcessor.from_pretrained(GEMMA_PATH)
model = AutoModelForImageTextToText.from_pretrained(GEMMA_PATH, torch_dtype="auto", device_map="auto")
```

```python
def query_model_text_image(prompt, image_url, max_new_tokens=32):
    start_time = time()
        
    messages = [
        {   
            "role": "user",
            "content": [
                {"type": "image", "image": image_url},
                {"type": "text", "text": prompt}
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
    total_time = round(time() - start_time, 2)
    response = text[0]
    return response, total_time
```

Let's check this image with a famous landmark in Paris, France:

<img src="https://wmf.imgix.net/images/aa_fra_notre-dame_de_paris_0.jpg" width=400></img>

```python
prompt = "What represent this image?"
image_url = "https://wmf.imgix.net/images/aa_fra_notre-dame_de_paris_0.jpg"
response, total_time = query_model_text_image(prompt, image_url, max_new_tokens=128)
display(Markdown(colorize_text(f"Execution time: {total_time}\n\nQuestion: {prompt}\n\nResponse: {response}")))
```

# Conclusions


Preliminary conclusion after testing the model with:
* History questions  
* Pop culture  
* Math (arithmetics, algebra, geometry)
* Multiple languages.
* Multimodal content (text + image).
  
is that the model is performing reasonably well with easy and medium-level questions.

**Good points**:
- When prompted to answer to the point, the model tend to behave well.
- Math seems to be accurate.
- Language capability is extensive.

**Areas to improve**:
- Modify the output to stop at the end of a phrase.
- Continue to test the model with more examples of multi-modal input.