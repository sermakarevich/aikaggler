# Tad Jones and the Paititi Legend

- **Author:** Marília Prata
- **Votes:** 68
- **Ref:** mpwolke/tad-jones-and-the-paititi-legend
- **URL:** https://www.kaggle.com/code/mpwolke/tad-jones-and-the-paititi-legend
- **Last run:** 2025-05-15 23:34:47.260000

---

Published on May 15, 2025. By Prata, Marília (mpwolke)

## Competition Description

"Stretching over 6,000,000 sq km and spanning nine countries, the Amazon Rainforest holds the history of past civilizations and serves as an active home to numerous Indigenous groups. Resources such as satellite imagery and LIDAR are helping to fill in the gaps for a previously unknown part of the world, sparking interest in the region and driving global headlines. Rumors exist of a “lost city of Z” in the Amazon, as well as legends like **Paititi** and El Dorado."

## Citation

@misc{openai-to-z-challenge,

    author = {Chris Fisher and Philip Bogdanov and Paul Mooney and Nate Keating and Maggie Demkin},
    
    title = {OpenAI to Z Challenge},
    
    year = {2025},
    
    howpublished = {\url{https://kaggle.com/competitions/openai-to-z-challenge}},

```python
#By Paul Mooney https://www.kaggle.com/code/paultimothymooney/how-to-use-openai-models-on-kaggle/notebook

import os
from openai import OpenAI
from kaggle_secrets import UserSecretsClient
```

#Paititi

"Paititi" is a place mentioned in the "Tad the Lost Explorer" animated film series. In the English-speaking versions, it's referred to as "the Lost City of Paititi". The character, Tad Jones, is an aspiring archaeologist who is often involved in adventures related to this lost city and the mummy of Paititi.

## Tad the Lost Explorer:

This is the name of a Spanish animated film series that follows the adventures of Tad Jones, a construction worker who dreams of becoming a famous archaeologist.

## Paititi:

This is the name of a lost city in the Andes Mountains, which is often the focus of Tad Jones's adventures. The city of Paititi is described as a lost Inca city, often associated with hidden gold and ancient artifacts.

The Mummy of Paititi:

In "Tad the Lost Explorer and the Secret of King Midas," Tad encounters **the mummy of Paititi**, who is exiled from the city and later becomes a companion on his adventures. 

Source: Paiti + Tad Jones

https://www.google.com/search?q=paititi+tad+jones+in+english&sca_esv=6c4a24e0bcfeebc3&sxsrf=AHTn8zpPWLyar6KgUUANspcCjuFVYJ0AJQ%3A1747345940702&ei=FGImaI_TKtm65OUPlL_JoAw&ved=0ahUKEwiPreHwuqaNAxVZHbkGHZRfEsQQ4dUDCBA&uact=5&oq=paititi+tad+jones+in+english&gs_lp=Egxnd3Mtd2l6LXNlcnAiHHBhaXRpdGkgdGFkIGpvbmVzIGluIGVuZ2xpc2gyBRAhGKABMgUQIRigAUjrQVCcHlj8PXABeAGQAQCYAd8BoAGPDqoBBTAuOS4yuAEDyAEA-AEBmAIMoALIDsICChAAGLADGNYEGEfCAgQQIRgVwgIHECEYoAEYCpgDAIgGAZAGCJIHBTEuOS4yoAeDKrIHBTAuOS4yuAfFDg&sclient=gws-wiz-serp

```python
#By Paul Mooney https://www.kaggle.com/code/paultimothymooney/how-to-use-openai-models-on-kaggle/notebook

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

## My OpenAI Api Key credits

![](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQCNkMfPcQT-oBTu2CL_fB6jhVulXdcfIY-5Q&s)

```python
#By Paul Mooney https://www.kaggle.com/code/paultimothymooney/how-to-use-openai-models-on-kaggle/notebook

openai_key = load_secret('tadjones')

client = OpenAI(
  api_key=openai_key
)

prompt = "Produce a plan for Tad Jones an archaeologist explorer about how he could use GPT-4o to analyze imagery with the goal of finding evidences of the existence of Paititi city."

completion = client.chat.completions.create(
  model="gpt-4o-mini",
  store=True,
  messages=[
    {"role": "user", "content": prompt}
  ]
)

print(completion.choices[0].message.content);
```

![](https://i.ytimg.com/vi/U7I5Xhj3yWE/hq720.jpg?sqp=-oaymwEhCK4FEIIDSFryq4qpAxMIARUAAAAAGAElAADIQj0AgKJD&rs=AOn4CLDBvWW4ybtQm-Xbx3zGt5opotmXIQ)youtube

# Since I won't make any deposit credits into my OpenAI Credit balance. My Paititi project will remain as the city. Legendary : ) 

Draft Session: 1h:0m

#Acknowledgements:

Paul Mooney https://www.kaggle.com/code/paultimothymooney/how-to-use-openai-models-on-kaggle/notebook