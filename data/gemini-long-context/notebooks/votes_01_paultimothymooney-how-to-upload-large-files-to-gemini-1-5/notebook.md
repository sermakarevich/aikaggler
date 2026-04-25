# How to upload large files to Gemini 1.5

- **Author:** Paul Mooney
- **Votes:** 165
- **Ref:** paultimothymooney/how-to-upload-large-files-to-gemini-1-5
- **URL:** https://www.kaggle.com/code/paultimothymooney/how-to-upload-large-files-to-gemini-1-5
- **Last run:** 2024-10-17 17:40:48.767000

---

# How to upload large files to Gemini 1.5

## Step 0: Preview your large file

```python
from IPython.display import Video
nepali_video = '/kaggle/input/bish-100-nepali-text-driven-ai-anchor/Video/10.webm'
Video(nepali_video, embed=True, width=640, height=480)
```

## Step 1: Import Python Packages

```python
import os
import time
import google.generativeai as genai
from kaggle_secrets import UserSecretsClient
```

## Step 2: Authenticate with Google Generative AI

Use of Gemini will require an API key. Please visit [Google AI Studio](https://ai.google.dev/)  to generate your key. 

Next, you should attach that key to your Kaggle Notebook as a [Kaggle User Secret](https://www.kaggle.com/discussions/product-feedback/114053). 

These steps are illustrated in the following screenshots:

Attaching User Secrets: 
 - https://i.imgur.com/GjuRLCA.png
 - https://i.imgur.com/IrSXAtw.png
 
For details about pricing see https://ai.google.dev/pricing.

```python
user_secrets = UserSecretsClient()
ai_studio_token = user_secrets.get_secret("ai_studio_token")
genai.configure(api_key=ai_studio_token)
```

## Step 3: Define helper functions

```python
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

## Step 4: Load the Gemini 1.5 model

```python
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

## Step 5: Upload your large file to Gemini 1.5

```python
files = [
  upload_to_gemini(nepali_video, mime_type="video/webm"),
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

## Step 6: Ask Gemini 1.5 questions about your large file

```python
response = chat_session.send_message("Please give me the English language transcript for this video.")
print(response.text)
```

```python
print(response.usage_metadata)
```

Credit:
 - Adapted from https://aistudio.google.com/app/prompts/video-qa