# Translator of Old Korean Literature

- **Author:** Ju-yeong Ji
- **Votes:** 87
- **Ref:** bebechien/translator-of-old-korean-literature
- **URL:** https://www.kaggle.com/code/bebechien/translator-of-old-korean-literature
- **Last run:** 2024-10-02 23:52:20.647000

---

# Gemma - Translator of Old Korean Literature

The Korean alphabet, or Hangul, has undergone changes over time, resulting in several letters no longer used in modern Korean. These obsolete letters include:

1. ㆍ (Arae-a): This dot vowel represents a short 'a' sound.
2. ㆆ (Yeorin-hieut): Pronounced as a 'light h,' akin to a softer version of the English 'h.'
3. ㅿ (Bansiot): Represents the 'z' sound.
4. ㆁ (Yet-ieung): A velar nasal sound comparable to 'ng' in the word 'sing.'

For Korean speakers, reading older literature presents a challenge due to the utilization of now-obsolete letters. Early Hangul lacked spaces between words, further complicating readability. In contrast, modern Hangul employs spaces, consistent with most alphabetic systems.

However, with the capabilities provided by Gemma, it becomes possible to create a translator that can aid in understanding and bridging the gap between contemporary and archaic Korean.

## Using accelerators

We will focus on using the **free GPU from Kaggle** here.

## Before you begin

### Gemma setup

To complete this tutorial, you will first need to complete the setup instructions at [Gemma setup](https://ai.google.dev/gemma/docs/setup). The Gemma setup instructions show you how to do the following:

Gemma models are hosted by Kaggle. To use Gemma, request access on Kaggle:

- Sign in or register at [kaggle.com](https://www.kaggle.com)
- Open the [Gemma 2 model card](https://www.kaggle.com/models/google/gemma-2) and select _"Request Access"_
- Complete the consent form and accept the terms and conditions

## Install dependencies

Install Keras and KerasNLP.

```python
# Install Keras 3 last. See https://keras.io/getting_started/ for more details.
!pip install -q -U keras-nlp datasets
!pip install -q -U keras

import os

# Set the backbend before importing Keras
os.environ["KERAS_BACKEND"] = "jax"
# Avoid memory fragmentation on JAX backend.
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "1.00"

import keras_nlp
import keras

# Run at half precision.
#keras.config.set_floatx("bfloat16")

# Training Configurations
token_limit = 256
lora_name = "translator"
lora_rank = 4
lr_value = 1e-4
train_epoch = 5
model_id = "gemma2_instruct_2b_en"
```

## Load Dataset

Here's [the dataset](https://huggingface.co/datasets/bebechien/HongGildongJeon) from Hong Gildong jeon (Korean: 홍길동전), which is a Korean novel written during the Joseon Dynasty. The [original source](https://ko.wikisource.org/wiki/%ED%99%8D%EA%B8%B8%EB%8F%99%EC%A0%84_36%EC%9E%A5_%EC%99%84%ED%8C%90%EB%B3%B8) is in public domain. You will use a [modern translation](https://ko.wikisource.org/wiki/%ED%99%8D%EA%B8%B8%EB%8F%99%EC%A0%84_36%EC%9E%A5_%EC%99%84%ED%8C%90%EB%B3%B8/%ED%98%84%EB%8C%80%EC%96%B4_%ED%95%B4%EC%84%9D) in a [creative commons license](https://creativecommons.org/licenses/by-sa/4.0/), translated by `직지프로`.

To simplify the task, you will adopt the following structure for fine-tuning the model. The model will generate contemporary Korean text based on the user's input in Early Hangul.

```
<start_of_turn>user\n
됴션국셰둉ᄃᆡ왕즉위십오연의홍희문밧긔ᄒᆞᆫᄌᆡ상이잇스되
<end_of_turn>\n
<start_of_turn>model\n
조선국 세종대왕 즉위 십오년에 홍회문 밖에 한 재상이 있으되,
```

> NOTE: korean text means, In the fifteenth year of the reign of King Sejong of Joseon, there was a prime minister outside Honghoemun Gate.

```python
tokenizer = keras_nlp.models.GemmaTokenizer.from_preset(model_id)

from datasets import load_dataset

ds = load_dataset(
    "bebechien/HongGildongJeon",
    split="train",
)
print(ds)
data = ds.with_format(
    "np", columns=["original", "modern translation"], output_all_columns=False
)
train = []

for x in data:
    item = f"<start_of_turn>user\n{x['original']}<end_of_turn>\n<start_of_turn>model\n{x['modern translation']}<end_of_turn>"
    length = len(tokenizer(item))
    # skip data if the token length is longer than our limit
    if length < token_limit:
        train.append(item)

print(len(train))
print(train[0])
print(train[1])
print(train[2])
```

## Load Model

```python
import keras
import keras_nlp

import time

gemma = keras_nlp.models.GemmaCausalLM.from_preset(model_id)
gemma.summary()

tick_start = 0


def tick():
    global tick_start
    tick_start = time.time()


def tock():
    print(f"TOTAL TIME ELAPSED: {time.time() - tick_start:.2f}s")


def text_gen(prompt):
    tick()
    input = f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
    output = gemma.generate(input, max_length=token_limit)
    print("\nGemma output:")
    print(output)
    tock()


text_gen("ᄃᆡ작ᄒᆞ여그ᄭᅩᆺ치흣터지거ᄂᆞᆯ")
text_gen(
    "금두겁이품의드러뵈니일졍ᄌᆡᄌᆞᄅᆞᆯ나흐리로다ᄒᆞ더니과연그ᄃᆞᆯ부터잉ᄐᆡᄒᆞ여십삭이차니"
)
text_gen(
    "이ᄯᆡᄂᆞᆫᄉᆞ월초팔일이라이날밤의오ᄉᆡᆨ구룸이집을두루고향ᄂᆡ진동ᄒᆞ며션녀ᄒᆞᆫᄡᅣᆼ이촉을들고드러와김ᄉᆡᆼᄃᆞ려니르ᄃᆡ"
)
text_gen("ᄌᆡ히길너텬졍을어긔지말으소셔이아희ᄇᆡ필은낙양니샹셔집아ᄌᆡ니")
```

## LoRA Fine-tuning

```python
# Enable LoRA for the model and set the LoRA rank (4, 8 or 16).
gemma.backbone.enable_lora(rank=lora_rank)
gemma.summary()

# Limit the input sequence length (to control memory usage).
gemma.preprocessor.sequence_length = token_limit
# Use AdamW (a common optimizer for transformer models).
optimizer = keras.optimizers.AdamW(
    learning_rate=lr_value,
    weight_decay=0.01,
)
# Exclude layernorm and bias terms from decay.
optimizer.exclude_from_weight_decay(var_names=["bias", "scale"])

gemma.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=optimizer,
    weighted_metrics=[keras.metrics.SparseCategoricalAccuracy()],
)
```

## Save LoRA for each epoch

```python
class CustomCallback(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs=None):
    model_name = f"/kaggle/working/{lora_name}_{lora_rank}_epoch{epoch+1}.lora.h5"
    gemma.backbone.save_lora_weights(model_name)

    # Evaluate
    text_gen("ᄃᆡ작ᄒᆞ여그ᄭᅩᆺ치흣터지거ᄂᆞᆯ")
    text_gen(
      "금두겁이품의드러뵈니일졍ᄌᆡᄌᆞᄅᆞᆯ나흐리로다ᄒᆞ더니과연그ᄃᆞᆯ부터잉ᄐᆡᄒᆞ여십삭이차니"
    )
    text_gen(
      "이ᄯᆡᄂᆞᆫᄉᆞ월초팔일이라이날밤의오ᄉᆡᆨ구룸이집을두루고향ᄂᆡ진동ᄒᆞ며션녀ᄒᆞᆫᄡᅣᆼ이촉을들고드러와김ᄉᆡᆼᄃᆞ려니르ᄃᆡ"
    )
    text_gen("ᄌᆡ히길너텬졍을어긔지말으소셔이아희ᄇᆡ필은낙양니샹셔집아ᄌᆡ니")

history = gemma.fit(train, epochs=train_epoch, batch_size=1, callbacks=[CustomCallback()])

import matplotlib.pyplot as plt
plt.plot(history.history['loss'])
plt.show()
```

## Expansion Idea

To achieve similar tasks, you can replicate the same structure. Below are some examples:

* American English <-> British English datasets

Various everyday objects and concepts have different names depending on the region. For example, in American English (AmE), people use terms like "elevator," "truck," "cookie," and "french fries," while in British English (BrE), the equivalent words are "lift," "lorry," "biscuit," and "chips," respectively.

Apart from vocabulary differences, spelling variations also exist. For instance, in AmE, words ending in "-or" are often spelled with "-our" in BrE. Examples include "color" (AmE) and "colour" (BrE), or "humor" (AmE) and "humour" (BrE).

Another spelling variation is the "-ize" versus "-ise" distinction. In AmE, words like "organize" and "realize" are commonly spelled with a "z," whereas in BrE, the preferred spelling is "organise" and "realise," using an "s" instead.

With the help of AI tools like Gemma, it is possible to create a style transfer from one English dialect to another, allowing seamless transitions between American and British English writing styles.

* Kansai-ben datasets

In the Kansai region of Japan, there is a distinct group of dialects known as Kansai-ben. Compared to the standard Japanese language, Japanese speakers perceive Kansai-ben as being both more melodic and harsher in its pronunciation and intonation.

Utilizing the capabilities of Gemma, you can create a dialect translator by preparing a substantial quantity of Kansai-ben datasets.