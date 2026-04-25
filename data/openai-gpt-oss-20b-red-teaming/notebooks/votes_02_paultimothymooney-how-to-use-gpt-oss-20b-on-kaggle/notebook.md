# How to use GPT-OSS-20b on Kaggle

- **Author:** Paul Mooney
- **Votes:** 156
- **Ref:** paultimothymooney/how-to-use-gpt-oss-20b-on-kaggle
- **URL:** https://www.kaggle.com/code/paultimothymooney/how-to-use-gpt-oss-20b-on-kaggle
- **Last run:** 2025-08-12 18:05:11.407000

---

# How to use GPT-OSS-20b on Kaggle
 - Using code adapted from [@danielhanchen](https://www.kaggle.com/models/danielhanchen/gpt-oss-20b?select=README.md) and [@bwandowando](https://www.kaggle.com/code/bwandowando/i-m-sorry-but-i-cant-help-with-that)

```python
# Code adapted from [@bwandowando](https://www.kaggle.com/code/bwandowando/i-m-sorry-but-i-cant-help-with-that)
!pip uninstall numpy scikit-learn scipy cupy-cuda12x torch torchaudio torchvision transformers triton kernels --yes
!pip cache purge
!pip install numpy==1.26.4 scikit-learn==1.5.2 scipy==1.15.3 cupy-cuda12x==13.5.1
!pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/test/cu128
!git clone https://github.com/huggingface/transformers.git
!pip install transformers/.[torch]
!pip install git+https://github.com/triton-lang/triton.git@main#subdirectory=python/triton_kernels
!pip install kernels
```

```python
import torch
from transformers import pipeline
import kagglehub

model_id = kagglehub.model_download("danielhanchen/gpt-oss-20b/transformers/default")


pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype="auto",
    device_map="auto",
)

messages = [
    {"role": "user", "content": "Explain quantum mechanics clearly and concisely."},
]

outputs = pipe(
    messages,
    max_new_tokens=256,
)
```

```python
print(outputs[0]["generated_text"][-1]['content'])
```