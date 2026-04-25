# Qwen2-7B-Instruct with VLLM==0.6.3 inference

- **Author:** FlashAi
- **Votes:** 225
- **Ref:** flashai/qwen2-7b-instruct-with-vllm-0-6-3-inference
- **URL:** https://www.kaggle.com/code/flashai/qwen2-7b-instruct-with-vllm-0-6-3-inference
- **Last run:** 2024-12-17 01:53:00.850000

---

```python
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
```

```python
%%time
!pip uninstall -y torch
!pip install  --no-index --find-links=/kaggle/input/vllm063/whl4vllm063 torchvision==0.20.1+cu121
!pip install  --no-index --find-links=/kaggle/input/vllm063/torch24cu12 torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0
!pip install  --no-index --find-links=/kaggle/input/vllm063/whl4vllm063 vllm==0.6.3
```

```python
from vllm import LLM, SamplingParams
import torch
model_name = "/kaggle/input/wsdm-gptq/transformers/default/1/"
# Create an LLM.
llm = LLM(model=model_name,dtype=torch.float16,max_model_len=5000,tensor_parallel_size=2,gpu_memory_utilization= 0.96)

batch_size = 5
def chunk_list(data, chunk_size=5):
    """
    将列表按指定大小拆分成小组
    :param data: 原始列表
    :param chunk_size: 每组的元素个数，默认为5
    :return: 拆分后的列表组
    """
    return [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
```

```python
import pandas as pd
path = '/kaggle/input/wsdm-cup-multilingual-chatbot-arena/'
train = pd.read_parquet(path + "train.parquet",engine='pyarrow')
test = pd.read_parquet(path + "test.parquet",columns=['id', 'prompt', 'response_a', 'response_b', 'scored'])
```

```python
#from vllm import LLM, SamplingParams
#import torch
#model_translate = "/kaggle/input/qwen2.5/transformers/qwen2.5-14b-instruct-awq/1/"
# Create an LLM.
#llm_translate = LLM(model=model_translate,dtype=torch.float16,max_model_len=4000,tensor_parallel_size=2,gpu_memory_utilization= 0.96)
```

```python
# %time
# sampling_params = SamplingParams(temperature=0.5,top_p=1 ,
#             top_k=1,  
#             seed=777, max_tokens=2000,
#             skip_special_tokens=False)
# conversation = [
#     {
#         "role": "system",
#         "content": "You are a helpful assistant"
#     },
#     {
#         "role": "user",
#         "content": "将文本翻译成中文,仅给出翻译后的文本即可，文本：Caso Clínico: Un hombre de 70 años con antecedentes de cáncer testicular tratado en 1990 y cáncer gástrico tratado en 2020, es diagnosticado con leucemia mieloide aguda (LMA). El paciente inicia tratamiento con quimioterapia (12 sesiones planificadas) y se presenta en la consulta después de la primera sesión con síntomas de fatiga intensa, disnea leve, palpitaciones y mareos. Los estudios de laboratorio revelan una hemoglobina de 8 g/dL, leucocitos bajos, y plaquetas en el límite inferior de lo normal.\n\nEl paciente refiere no haber tenido sangrados visibles, pero menciona sentirse más cansado de lo habitual desde el inicio del tratamiento. No tiene fiebre ni signos de infección activa.\n\nPregunta:\n\nCon base en el cuadro clínico descrito, proponga un diagnóstico diferencial que incluya al menos tres posibles causas de la anemia en este paciente."
#     }]
# outputs = llm_translate.chat(conversation,
#                sampling_params=sampling_params,
#                use_tqdm=False)
# # Print the outputs.
# for i,output in enumerate(outputs):
#     prompt = output.prompt
#     generated_text = output.outputs[0].text
#     print(generated_text)
```

```python
submission = []
test_list = []
for p,a,b in zip(test.prompt.values,test.response_a.values,test.response_b.values):
    
    prompt = f"""你是一名精通多国语言的产品体验师，对于相同的【问题】，我会给到你两个来自不同服务商的【回复】，请你从回复质量、情感价值、使用体验等角度公平地评价两个不同的服务商，并选择其中一个服务商作为你认为更胜一筹的一方。
【问题】:{p}
【回复a】:{b[:5000]}
【回复b】:{a[:5000]}
    """
    test_list.append(prompt)
```

```python
train_list = []
for p,a,b in zip(train.prompt.values,train.response_a.values,train.response_b.values):
    
    prompt = f"""你是一名精通多国语言的产品体验师，对于相同的【问题】，我会给到你两个来自不同服务商的【回复】，请你从回复质量、情感价值、使用体验等角度公平地评价两个不同的服务商，并选择其中一个服务商作为你认为更胜一筹的一方。
【问题】:{p}
【回复a】:{a}
【回复b】:{b}
    """
    train_list.append(prompt)
```

```python
print(test_list[0])
```

```python
len(test_list[0])
```

```python
%%time
from vllm import LLM, SamplingParams
sampling_params = SamplingParams(temperature=0,top_p=1,n=1,  
            top_k=1,  
            seed=777, 
            skip_special_tokens=False)
chunked = chunk_list(test_list, chunk_size=10)
result = []
for i, v in enumerate(chunked):
    conversation = [[
    {
        "role": "system",
        "content": "You are a helpful assistant"
    },
    {
        "role": "user",
        "content": prompt
    }
    ] for prompt in v]
    outputs = llm.chat(conversation,
                   sampling_params=sampling_params,
                   use_tqdm=False)
    # Print the outputs.
    for i,output in enumerate(outputs):
        prompt = output.prompt
        generated_text = output.outputs[0].text
        if generated_text != "":
            if generated_text == "model_a":
                result.append("model_b")
            else:
                result.append("model_a")       
        else:
            result.append("model_a")
```

```python
result
```

```python
sub = pd.DataFrame({
    'id': test.id.values,
    'winner': result
})
```

```python
sub
```

```python
sub.to_csv("submission.csv",index=False)
```