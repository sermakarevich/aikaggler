# How to use OpenAI models on Kaggle

- **Author:** Paul Mooney
- **Votes:** 264
- **Ref:** paultimothymooney/how-to-use-openai-models-on-kaggle
- **URL:** https://www.kaggle.com/code/paultimothymooney/how-to-use-openai-models-on-kaggle
- **Last run:** 2025-05-15 17:05:42.060000

---

## How to use OpenAI models on Kaggle

## Step 1: Authenticate with the OpenAI API

1. Generate an API key
  - Login to your account at openai.com
  - Navigate to https://platform.openai.com/api-keys and click on "create new secret key"
  - Copy your secret key and save it in a safe place.
  - Note: you can only view your secret key one time. For security reasons, OpenAI hides the full key after it's generated.
2. Save your API key as a Kaggle Secret
  - In the Kaggle Notebook editor, click on the "Add-ons" button.
  - Click on the "secrets" option.
  - Click on the "add secret" button.
  - Enter you secret and then click "save"
3. Use kaggle_secrets.UserSecretsClient() to securely access your secret
 - https://www.kaggle.com/discussions/product-feedback/114053

```python
import os
from openai import OpenAI
from kaggle_secrets import UserSecretsClient
```

```python
def load_secret(name):
    """Loads secret from Colab/Kaggle."""

    if 'KAGGLE_KERNEL_RUN_TYPE' in os.environ:
        try:
            from kaggle_secrets import UserSecretsClient
            return UserSecretsClient().get_secret(name)
        except Exception:
            pass 
    else:
        try:
            from google.colab import userdata
            return userdata.get(name)
        except Exception: 
            pass

    return 'Secret not found'
```

## Step 2: Send requests to the OpenAI API

```python
openai_key = load_secret('openai_key_2025')

client = OpenAI(
  api_key=openai_key
)

prompt = "Produce a detailed plan for a research scientist and provide recommendations about how and where they could use GPT-4o to analyze multi-spectral satellite imagery with the goal of discovering evidence of ancient civilizations in Brazil."

completion = client.chat.completions.create(
  model="gpt-4o-mini",
  store=True,
  messages=[
    {"role": "user", "content": prompt}
  ]
)

print(completion.choices[0].message.content);
```