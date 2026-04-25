# Gemini's thirst for knowledge

- **Author:** Marília Prata
- **Votes:** 78
- **Ref:** mpwolke/gemini-s-thirst-for-knowledge
- **URL:** https://www.kaggle.com/code/mpwolke/gemini-s-thirst-for-knowledge
- **Last run:** 2024-10-21 03:02:38.403000

---

Published on October 21, 2024. By Marília Prata, mpwolke.

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

## Our Anti-Gravity Humidifier mp4 video

"Anti Gravity Humidifier Design Anti Gravity Humidifier uses optical principles to create an optical illusion of anti-gravity water droplets. Latest technology, Using the latest ultrasonic atomization technology, it outputs nanoscale fine fog which moisturizing the air and relieving pressure."

"INNOVATIVE ANTI-GRAVITY DESIGN: Their humidifier creates an anti-gravity illusion, making the water droplets appear to be slowly flowing upwards while producing an ambient glow and soothing running water sound."

https://www.amazon.in/ORILEY-Anti-gravity-Humidifier-Ultrasonic-Humidifiers/dp/B0BFQKR73X?th=1

Pick a short video to avoid excess of memory and be cancelled.

```python
from IPython.display import Video
water_video = '/kaggle/input/anti-gravity-humidifier/Anti-Gravity Humidifier 1/S (103).mp4'
Video(water_video, embed=True, width=640, height=480)
```

#Import Packages

```python
import os
import time
import google.generativeai as genai
from kaggle_secrets import UserSecretsClient
```

#Authenticate Gemini with Google Generative AI

Gemini Model is NOT ttachedd directly to your Notebook. You should generate an API-Key and**also**o add your**Kaggle secret**t.

```python
#By Paul Mooney https://www.kaggle.com/code/paultimothymooney/how-to-upload-large-files-to-gemini-1-5/notebook

from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
GEMINI_API_KEY = user_secrets.get_secret("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
```

#Define helper functions

```python
#By Paul Mooney https://www.kaggle.com/code/paultimothymooney/how-to-upload-large-files-to-gemini-1-5/notebook

def upload_to_gemini(path, mime_type=None):
  """Uploads the given file to Gemini.

  See https://ai.google.dev/gemini-api/docs/prompting_with_media
  """
  file = genai.upload_file(path, mime_type=mime_type)
  print(f"Uploaded file '{file.display_name}' as: {file.uri}")
  return file

def wait_for_files_active(files):
  """Waits for the given files to be active.

  Some files uploaded to the Gemini API need to be processed before they can be
  used as prompt inputs. The status can be seen by querying the file's "state"
  field.

  This implementation uses a simple blocking polling loop. Production code
  should probably employ a more sophisticated approach.
  """
  print("Waiting for file processing...")
  for name in (file.name for file in files):
    file = genai.get_file(name)
    while file.state.name == "PROCESSING":
      print(".", end="", flush=True)
      time.sleep(10)
      file = genai.get_file(name)
    if file.state.name != "ACTIVE":
      raise Exception(f"File {file.name} failed to process")
  print("...all files ready")
  print()
```

#Load the Gemini 1.5 Flash API model

```python
#By Paul Mooney https://www.kaggle.com/code/paultimothymooney/how-to-upload-large-files-to-gemini-1-5/notebook

# Create the model
generation_config = {
  "temperature": 1,
  "top_p": 0.95,
  "top_k": 64,
  "max_output_tokens": 8192,
  "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
  model_name="gemini-1.5-flash",
  generation_config=generation_config,
)
```

```python
#By Paul Mooney https://www.kaggle.com/code/paultimothymooney/how-to-upload-large-files-to-gemini-1-5/notebook

files = [
  upload_to_gemini(water_video, mime_type="video/mp4"),
]

wait_for_files_active(files)

chat_session = model.start_chat(
  history=[
    {
      "role": "user",
      "parts": [
        files[0],
      ],
    }
  ]
)
```

## Describe what we see in this video.

```python
#By Paul Mooney https://www.kaggle.com/code/paultimothymooney/how-to-upload-large-files-to-gemini-1-5/notebook

response = chat_session.send_message("Describe what we see in this video.")
print(response.text)
```

```python
#By Paul Mooney https://www.kaggle.com/code/paultimothymooney/how-to-upload-large-files-to-gemini-1-5/notebook

print(response.usage_metadata)
```

## What is the sound on the video?

```python
response1 = chat_session.send_message("What is the sound on the video?")
print(response1.text)
```

## What is the time on the display of the anti-gravity humidifier?

It shows 23:36.

```python
response2 = chat_session.send_message("What is the time on the display of the anti-gravity humidifier?")
print(response2.text)
```

## What is written on the caption?

```python
response3 = chat_session.send_message("What is written on the caption?")
print(response3.text)
```

## Has the caption any emoji?

```python
response4 = chat_session.send_message("Has the caption any emoji?")
print(response4.text)
```

## Is the emoji's expression positive?

```python
response5 = chat_session.send_message("Is the emoji's expression positive?")
print(response5.text)
```

## What is the liquid dropping from the humidifier?

```python
response6 = chat_session.send_message("What is the liquid dropping from the humidifier?")
print(response6.text)
```

## Are the water droplets really flowing upwards?

```python
response7 = chat_session.send_message("Are the water droplets really flowing upwards?")
print(response7.text)
```

## What is an anti-gravity humidifier?

```python
response8 = chat_session.send_message("What is an anti-gravity humidifier?")
print(response8.text)
```

## I'm pretty much satisfied with Gemini answers. Let's check after I save its version.

#Acknowledgements:

Paul Mooney https://www.kaggle.com/code/paultimothymooney/how-to-upload-large-files-to-gemini-1-5/notebook