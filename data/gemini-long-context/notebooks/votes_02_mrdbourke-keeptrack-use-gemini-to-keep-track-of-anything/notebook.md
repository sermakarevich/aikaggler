# KeepTrack: Use Gemini to Keep Track of Anything

- **Author:** Daniel Bourke
- **Votes:** 131
- **Ref:** mrdbourke/keeptrack-use-gemini-to-keep-track-of-anything
- **URL:** https://www.kaggle.com/code/mrdbourke/keeptrack-use-gemini-to-keep-track-of-anything
- **Last run:** 2024-12-04 10:49:50.203000

---

# KeepTrack: Using Gemini's Long Context Window and Video Capabilities to KeepTrack of Anything with Video

In this notebook, we'll see how we can use Gemini's long context and video capabilities to keep track of all the items in a video.

The goal will be to go from a smartphone video of a house full of items to a structured database of all the items in that house.

> 💡 **Tip:** Search for "Note:" throughout the notebook for many helpful links to documentation as well as tidbits on using the Gemini API.
>
> Also see the Tidbit and Takeaways section in the Appendix for learnings along the way.

* Watch the video explainer on YouTube: https://youtu.be/dbDgcFnlZlE
* Get the video to try for yourself: https://www.kaggle.com/datasets/mrdbourke/keeptrack-house-video-with-audio-horizontal-720p

<img src="https://github.com/mrdbourke/gemini-long-context-window-kaggle/blob/main/google-gemini-youtube-video-thumbnail-annotated.jpeg?raw=true" alt="Man filming a living room setup with labeled furniture including a couch, green stools, coffee table, artwork, and a mirror; text annotations describe the scene." />

## Problem Statment

Getting the right home and contents insurance is hard.

Especially when you're not sure what you have.

You can estimate a high value of items but you might end up paying too much.

Or you can estimate a low value but you might not be covered for all your items.

Wouldn't it be good if you could just get insurance for items you actually *have*?

But keeping track of every major item in your house is a tedious task...

What if Gemini could help?

## Solution

You could write down every item you have.

Or you could take a photo of every item you have.

Depending on how much stuff you've got, both of these could take quite a bit of time.

What about a video?

Just do a lap of your house and hightlight the most important items.

And then have Gemini go over the video and write down everything, from names to estimated worth of items in the video for you.

You could then take this information to your insurance provider and get an insurance quote for what you *actually* have rather than guessing.

Or you could even just keep track of your stuff for your own personal records (hint: this workflow can work for more than just household items, it works for record collections, wine collections, items in your pantry, Gemini's video capabilities are very flexible).

## Science Fiction? No, Gemini.

What?

Take a video of your house and have everything cataloged?

In the past a workflow like this might've taken a series of machine learning models:

* A computer vision model for identifying items.
* A speech transcription model for transcribing speech.
* A NER (named entity recognition) model for extracting item names from text.

And then you'd need a way to combine the data in a structured way.

(spoiler incoming)

As we'll see in this notebook, this workflow is possible with a single model: Gemini.

And all for under 10c (quite a bit less than the average insurance policies monthly fees).

## Inputs and outputs

* **Inputs:** ~10-minute house tour video shot on smartphone + various prompts to extract items.

* **Outputs:** Structured data in the form of CSV of all major items + various metadata.

## Gemini feature overview

To make this happen, there are several key Gemini API technologies we'll be taking advantage of:

1. [**Video processing**](https://ai.google.dev/gemini-api/docs/vision?lang=python#prompting-video) - The Gemini API can handle videos as input. This is crucial for our use case as our goal is to use a simple smartphone recording of a collection of items to then keep track of the items in that video. Video processing is a unique feature of the Gemini model compared to other AI APIs.

2. [**Long context windows**](https://ai.google.dev/gemini-api/docs/long-context) - Gemini models can handle input contexts of 1M-2M tokens (1M for Gemini Flash variants and 2M for Gemini Pro variants). This is an outstandingly large input. For example, that's 1-2 hours of video, easily enough for our use case (for context, our ~10 minute house tour video equates to around 170,000 tokens).
    * Since the outputs of a model are based on the inputs, having a large capacity for high-quality examples in the input as well as a generally large amount of information to begin with means Gemini is perfect for tasks which require large inputs.

3. [**Context caching**](https://ai.google.dev/gemini-api/docs/caching?lang=python) - Many workflows may require going over the same information several times. For example, for our video information extraction pipeline, we're going to reference the same video three times (though we could potentially do more). Context caching allows you cache target inputs for repeated inference at a lower cost.
    * You can think of context caching as a short-term memory for a Gemini model, rather than processing the same information from fresh each time (this can be costly if doing it many times), you process it once and access it as many times as needed.

## Setup Gemini API Key

If using Google Colab, can setup Gemini API in "Secrets" -> Add new key.

If using Kaggle, can setup Gemini API key in "Add-ons" -> "Secrets" -> Add new key.

> **Note:** You can get a Gemini API key in AI Studio, see the documentation here: https://ai.google.dev/gemini-api/docs/api-key

```python
import os

import google.generativeai as genai
```

```python
# Setup Gemini API Key for Kaggle
from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
GEMINI_API_KEY = user_secrets.get_secret("GEMINI_API_KEY")

# Configure the API key
genai.configure(api_key=GEMINI_API_KEY)
```

```python
# Can get a list of information from a model
# Note: See a list of available models in the docs: https://ai.google.dev/gemini-api/docs/models/gemini
model_info = genai.get_model("models/gemini-1.5-flash-002")
model_info
```

Gemini models can take almost any kind of input (text, images, video, audio) and produce text-based outputs.

Since the Gemini model generates an output based on an input, we refer to it as a Generative AI model.

We can generate content based on an input using the [`generate_content` method](https://ai.google.dev/api/generate-content).

```python
import time

from IPython.display import Markdown, display

def display_markdown(input):
  return display(Markdown(input))

# Start a timer and produce an output based on a text string
start_time = time.time()
model = genai.GenerativeModel(model_name="models/gemini-1.5-flash-002")
example_output = model.generate_content("Tell me a good use case for video with the Gemini API")
end_time = time.time()

print(f"[INFO] Time taken: {end_time-start_time:.2f} seconds\nOutput:")

display_markdown(example_output.text)
```

Nice! This video use case is very similar to the use case we're going to be going through.

## Setup System Instructions

System instructions are a way to customize how a model responds.

These can be almost anything.

From "respond like a pirate" to "make sure you only return JSON".

For our use case, we'll set a set of system instructions (also referred to as a system prompt) to tell Gemini to focus on creating accurate logging details given an input video.

> **Note:** For more on system instructions, see the documentation: https://ai.google.dev/gemini-api/docs/system-instructions?lang=python

```python
system_prompt = "You are an expert home inventory logger and home insurance reviewer. Your job is to review videos and photos and catalogue items in the most correct way possible for home & contents insurance quotes as well as personal inventory databases."

print(system_prompt)
```

## Setup Gemini Models and Configs

Gemini comes in three main flavours:

1. Gemini 1.5 Flash 8B
2. Gemini 1.5 Flash
3. Gemini 1.5 Pro

Each can handle the same inputs and outputs, however, each has varying levels of capabilities.

For example, Gemini 1.5 Flash 8B is the fastest and is great for frequent simple tasks.

Whereas Gemini 1.5 Pro is best for the most challenging tasks.

And Gemini 1.5 Flash sits in the goldilocks zone.

| **Model** | **Input Capacity (tokens)** | **Output Capacity (tokens)** | **Input Types** | **Output Types** | **Intelligence Level** |
| ----- | ----- | ----- | ----- | ----- | ----- |
| `model/gemini-1.5-flash-8b` | 1,048,576 | 8,192 | Audio, images, video, and text | Text | Good |
| `model/gemini-1.5-flash` | 1,048,576 | 8,192 | Audio, images, video, and text | Text | Better |
| `model/gemini-1.5-pro` | 2,097,152 | 8,192 | Audio, images, video, and text | Text | Best |

One the most import features of every Gemini model is the ability to take in video as well as a large context window (see the input capacity column).

This means that Gemini models can accept a large amount of input information in one go and process it to produce an output.

We'll be using this long context input capability for our use case.

> **Note:** These models will often change and be updated, best to refer and read more about the different Gemini models in the models documentation: https://ai.google.dev/gemini-api/docs/models/gemini

For our use case, we'll stick with Gemini 1.5 Flash for a great middle of the road option.

Each model accepts a series of parameters or settings known as a generation config.

Generally the models will have good baseline settings, however these can be tweaked to your needs.

For example, the following settings are some of the most commonly changed:

* `temperature` - A value closer to `0.0` will give straightforward and less unexpected results (though not guaranteed to be correct). A higher value will give more creative results. See more in the documentation: https://ai.google.dev/gemini-api/docs/prompting-strategies#temperature
* `top_k` - Top-K dictates how a model selects the next token in the output. A Top-K of `5` means that the model will choose the next token from a potential pool of `5`. Where as a Top-K of `1` means that the model will choose the next most likely token (also known as greedy decoding or choosing the most likely). See more in the documentation: https://ai.google.dev/gemini-api/docs/prompting-strategies#top-k
* `top_p` - Top-P changes how the model selects tokens for output. Tokens will get selected based on their probability values. A higher Top-P value leads to more random responses where as a lower value will lead to less random responses. See more in the documentation: https://ai.google.dev/gemini-api/docs/prompting-strategies#top-p

> **Note:** For more generation config options, refer to the Gemini Python API GitHub: https://github.com/google-gemini/generative-ai-python/blob/main/docs/api/google/generativeai/types/GenerationConfig.md

```python
# Set baseline parameters
# Note: If you aren't getting the outputs you're after, try adjusting these as well as your input prompts.
temperature = 0.5 # default is 0.2
top_k = 40 # default is 40
top_p = 0.95 # default is 0.95

# Create the model config
generation_config = {
  "temperature": temperature,
  "top_p": top_p,
  "top_k": top_k,
  "max_output_tokens": 8192,
  "response_mime_type": "text/plain",
}

# Setup models
model_gemini_flash_8b = genai.GenerativeModel(model_name="models/gemini-1.5-flash-8b",
                                              system_instruction=system_prompt,
                                              generation_config=generation_config)

model_gemini_flash = genai.GenerativeModel(model_name="models/gemini-1.5-flash-002",
                                           system_instruction=system_prompt,
                                           generation_config=generation_config)

model_gemini_pro = genai.GenerativeModel(model_name="models/gemini-1.5-pro-002",
                                         system_instruction=system_prompt,
                                         generation_config=generation_config)

# Create helper function for generating content
def generate_content(prompt, model):
  """Returns a given model's output for a given prompt."""
  return model.generate_content(prompt)

# Example usage
# example_plan_output = generate_content(prompt="How would you breakdown a video of a home tour to log all of the important items for an insurance quote?",
#                                        model=model_gemini_flash)
# print(example_plan_output.text)
```

## Kaggle: Get the Dataset

The reference video for KeepTrack is available on Kaggle Datasets at the URL: https://www.kaggle.com/datasets/mrdbourke/keeptrack-house-video-with-audio-horizontal-720p/data

You can also add this via the "Add Input" option.

```python
path_to_video_file = "/kaggle/input/keeptrack-house-video-with-audio-horizontal-720p/keeptrack-house-video-with-audio-horizontal-720p.mov"
print(f"[INFO] Video file is available at: {path_to_video_file}")
```

## View a collection of random video frames

Our goal is to turn a video of a house/storage unit/collection of items into a structured format of all the items present in the video with various metadata about the items.

We can inpsect our inspect video using [`cv2`](https://pypi.org/project/opencv-python/).

Let's start by inspecting a series of frames.

We do this because Gemini processes videos frame by frame (currently at 1 FPS).

So looking at individual frames of our video will give us an idea of the data Gemini may be seeing.

> **Note:** Gemini views frames at 1 FPS. However, I'm not 100% when that timing is. E.g. every second on the second or something different. The following code block selects frames randomly across *all* frames rather than at 1 FPS (though it could be updated for that).
>
> * See the documentation on Gemini video processing: https://ai.google.dev/gemini-api/docs/vision?lang=python#prompting-video
> * See the Gemini use case blog post for a tidbit on FPS sampling: https://developers.googleblog.com/en/7-examples-of-geminis-multimodal-capabilities-in-action/
>    * “Note: Due to 1FPS sampling, the model can occasionally miss items in videos. We are working on enabling higher FPS sampling for videos soon. Therefore, for now we recommend verifying outputs for these use cases if needed, but we want to show glimpses of what we are working towards and where our models will be in the coming months.”

```python
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image, ImageDraw, ImageFont

# Open the video file
video_capture = cv2.VideoCapture(path_to_video_file)

if not video_capture.isOpened():
    print("Error: Could not open video.")
else:
    frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_length_seconds = round(frame_count / fps, 2)
    video_length_minutes = round(video_length_seconds / 60, 2)

    print(f"[INFO] Video Metadata:")
    print(f" - Frame count: {frame_count}")
    print(f" - FPS: {fps}")
    print(f" - Video length: {video_length_seconds} seconds ({video_length_minutes} minutes)")
    print(f" - Width: {width}")
    print(f" - Height: {height}")

    # Choose random frames
    num_frames_to_display = 10
    random_frames_indices = random.sample(range(frame_count), num_frames_to_display)
    frames = []

    for frame_index in random_frames_indices:
        video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        success, frame = video_capture.read()
        if success:
            frames.append(frame)
            print(f"[INFO] Captured Frame Number: {frame_index}")
        else:
            print(f"Error: Could not read frame {frame_index}.")
            continue

    if len(frames) > 0:
        # Resize frames to a consistent size for the grid
        resized_frames = [cv2.resize(frame, (200, 200)) for frame in frames]

        # Create a grid to display the frames
        grid_rows = 2
        grid_cols = 5  # 2 rows x 5 columns = 10 frames
        if len(resized_frames) < grid_rows * grid_cols:
            grid_rows = 1
            grid_cols = len(resized_frames)

        grid_height = grid_rows * resized_frames[0].shape[0]
        grid_width = grid_cols * resized_frames[0].shape[1]
        grid = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)

        for i, frame in enumerate(resized_frames):
            row = i // grid_cols
            col = i % grid_cols
            grid[row * resized_frames[0].shape[0]:(row + 1) * resized_frames[0].shape[0],
                 col * resized_frames[0].shape[1]:(col + 1) * resized_frames[0].shape[1], :] = frame

        # Convert BGR to RGB for Matplotlib
        grid_rgb = cv2.cvtColor(grid, cv2.COLOR_BGR2RGB)

        # Display the grid using Matplotlib
        plt.figure(figsize=(10, 5))
        plt.imshow(grid_rgb)
        plt.axis('off')  # Hide axes for better visualization
        plt.title("Random Frames Grid")
        plt.show()

        # Save the grid to a file and display it
        grid_path = "random_frames_grid.jpg"
        cv2.imwrite(grid_path, grid)
        print("[INFO] Saved Random Frames Grid as 'random_frames_grid.jpg'")

        # Display the saved image in the notebook
        # display(Image(filename=grid_path))
    else:
        print("No frames were successfully captured to display.")

    video_capture.release()
```

## Upload video file for use with Gemini

Gemini can handle video files as input.

However, due to the often larger size of video files, we have to upload it first.

We can do so with the `genai.upload_file` API.

> **Note:** See documentation on video file upload for use with the Gemini API: https://ai.google.dev/gemini-api/docs/vision?lang=python#upload-video

```python
%%time

# Upload the video
video_file_upload = genai.upload_file(path=path_to_video_file)
```

Wonderful!

Before we can use our video file, we have to ensure its state has reached a level of `ACTIVE`, this means that the file is ready.

```python
%%time

# Check status of video file
print("[INFO] Video upload processing, please wait.", end="")

while video_file_upload.state.name == "PROCESSING":
  print(".", end="")
  time.sleep(2)
  video_file_upload = genai.get_file(video_file_upload.name)

if video_file_upload.state.name == "FAILED":
  raise ValueError(video_file_upload.state.name)
```

```python
# Check the video file state name, if the state name is ACTIVE, ok to proceed
video_file_upload.state.name
```

```python
# Get information about the video file
video_file_upload
```

```python
# Can also list active files
print("My files:")

for f in genai.list_files():
    print("->", f.name)
```

```python
# Delete video (optional, video/file will automatically delete after 48 hours)
# genai.delete_file(name=video_file_upload.name)
```

### Creating a context cache

One of the best features of the Gemini API is being able to create a context cahce.

We cache files so we don't have to process them again and again.

Instead, we process them once and access them as many times as needed.

You can think of a cache as a form of memory for a model.

Once a file is cached, it has been parsed and can be reused in a future context.

> **Note:** See the documentation on caching: https://ai.google.dev/gemini-api/docs/caching?lang=python

There are several benefits to context caching (from the docs):

> Context caching is a paid feature designed to reduce overall operational costs. Billing is based on the following factors:
>
> 1. **Cache token count:** The number of input tokens cached, billed at a reduced rate when included in subsequent prompts.
> 2. **Storage duration:** The amount of time cached tokens are stored (TTL), billed based on the TTL duration of cached token count. There are no minimum or maximum bounds on the TTL.
> 3. **Other factors:** Other charges apply, such as for non-cached input tokens and output tokens.
> 4. **Cache minmum and maximums:** A context cache has a minimum of 32,768 tokens (e.g. if your token count doesn't equal this, may be able to sample the video at a lower framerate to increase input tokens) and a maximum capacity of the given model (e.g. 1M for Gemini 1.5 Flah and 2M tokens for Gemini 1.5 Pro).

```python
from google.generativeai import caching

# Get a list of existing caches (note: this may return nothing if there aren't any existing cached contents)
for cached_object in caching.CachedContent.list():
  print(cached_object)
```

```python
# Get an existing cached object
try:
  video_file_cache = caching.CachedContent.get(name="cachedContents/db8qp5uvbw27") # note: this will error if the target cached object is not available
  print(video_file_cache)
except Exception as e:
  video_file_cache = None
  print(e)
  print(video_file_cache)
```

> Example error if a cache doesn't exist:
>
> `WARNING:tornado.access:403 GET /v1beta/cachedContents/lfldfj5372nda?%24alt=json%3Benum-encoding%3Dint (127.0.0.1) 1649.15ms`

No problem, we can create the cache.

> **Note:**  Context caching is only available for stable models with fixed versions (for example, gemini-1.5-pro-001). You must include the version postfix (for example, the -001 in gemini-1.5-pro-001).

```python
# Create a cache (this can be varied by the time you'd like it)
# Note: caching = cost for storage but = 4x cheaper input, see pricing: https://ai.google.dev/pricing#1_5flash
import datetime

CACHE_MINUTES = 15 # Note: Can change this based on your use case. Beware caching storage costs, best to cache for the exact amount of time you need.
CACHE_HOURS = CACHE_MINUTES / 60

if video_file_cache == None:
  video_file_cache = caching.CachedContent.create(
      model="models/gemini-1.5-flash-002", # note: model cache name should be same as model used to generate outputs, model postfix is required, e.g. "-002"
      display_name="house item catalogue video",
      system_instruction=system_prompt,
      contents=[video_file_upload],
      ttl=datetime.timedelta(minutes=CACHE_MINUTES) # Note: there are no minimum or maximum bounds on context caching, however you should consider the use case of your app because caching prices can ramp up if left unchecked.
)

video_file_cache
```

## Create a CSV schema for structured output

Our goal is to input a video of a house or collection of items and have all of the items logged in a structured format.

Why structured format?

This will allow us to use our output data with various visualization and database tools.

CSV stands for [comma-separated values](https://en.wikipedia.org/wiki/Comma-separated_values) and it's one of the most versatile and simple structured file types.

It's also very compatible with Python libraries such as pandas.

> **Note:** We could also do this for JSON as well. I've chosen to use CSV because its shorter in terms of pure character/token length to help get around the 8k token output limit.
>
> For more on structured outputs, see the Gemini documentation for structured outputs: https://ai.google.dev/gemini-api/docs/structured-output?lang=python

Since we're trying to log items in preparation for an insurance quote, we'll craft a CSV schema that reflects that.

Namely a schema with the fields:

1. `item_number` (int): Sequential integer starting from 1. Example: 1, 2, 3 Example: 1
2. `item_name` (str): Name of the item with proper capitalization. Multiple word items should have each word capitalized. Example: Coffee Table
3. `item_type` (str): Type of item in lowercase. Standard categories include: furniture, electronics, kitchenware, lighting, decor, appliances. Example: furniture
4. `item_description` (str): Detailed description including size, color, material, and notable features. Should be a complete phrase starting with a capital letter. Do not use commas in this field. Just describe the item verbatim. Example: Large brown leather sectional sofa with chaise lounge and metal legs.
5. `item_brand` (str): Brand name with proper capitalization. Use NA if not visible/mentioned. Examples: Samsung, IKEA, West Elm, NA Example: IKEA
6. `item_condition` (str): Condition in lowercase, using standard terms: no visible damage, slight wear, moderate wear, significant damage, as new. Example: no visible damage
7. `number_of_items` (int): Integer count of identical items. If there are multiple of the same item, enter their integer count. Example: Single item: 1, Multiple items: 4 Example: 1
8. `estimated_worth` (float): Estimated value in USD with exactly one decimal place. Guidelines: Always include .0 even for whole numbers. Do not include currency symbols or commas. For multiple items, list the per-item worth. Example: 499.0
9. `estimated_worth_flag` (int): Confidence level 1-10 with the following guidelines: 1-3: Highly uncertain, requires expert valuation. 4-6: Moderate confidence, may need verification. 7-8: Good confidence based on visible details. 9-10: Very confident, based on clear brand/model/condition. Example: 7
10. `mentioned_worth` (float): Worth mentioned in video, formatted same as estimated_worth. Use NA if not mentioned. Do not use commas in the values. Only full stops. Example: 1500.0
11. `room` (str): Room name with proper capitalization. Standard room names: Living Room, Dining Room, Kitchen, Bedroom, Bathroom, Office. Example: Living Room
12. `timestamp` (str): Time in MM:SS format with leading zeros. Example: 01:30
13. `overall_certainty_flag` (int): Overall confidence level 1-10 considering all fields: 1-3: Multiple uncertain fields, needs review. 4-6: Some uncertainty in key fields. 7-8: Minor uncertainty in non-critical fields. 9-10: All fields verified with high confidence. Example: 8
14. `is_similar_to` (str): Reference to similar item by item_number, or NA. Use for: Matching furniture pieces, Items from same set, Similar models of different sizes. Example: NA

Creating a schema like this means we can use it as a refernce and pass it to our models as an input/guide for their outputs to conform to.

```python
from typing import Dict, Any, Optional, List

csv_schema = [
    {
        "field": "item_number",
        "type": "int",
        "description": "Sequential integer starting from 1. Example: 1, 2, 3",
        "example": "1"
    },
    {
        "field": "item_name",
        "type": "str",
        "description": "Name of the item with proper capitalization. Multiple word items should have each word capitalized.",
        "example": "Coffee Table"
    },
    {
        "field": "item_type",
        "type": "str",
        "description": "Type of item in lowercase. Standard categories include: furniture, electronics, kitchenware, lighting, decor, appliances.",
        "example": "furniture"
    },
    {
        "field": "item_description",
        "type": "str",
        "description": "Detailed description including size, color, material, and notable features. Should be a complete phrase starting with a capital letter. Do not use commas in this field. Just describe the item verbatim.",
        "example": "Large brown leather sectional sofa with chaise lounge and metal legs."
    },
    {
        "field": "item_brand",
        "type": "str",
        "description": "Brand name with proper capitalization. Use NA if not visible/mentioned. Examples: Samsung, IKEA, West Elm, NA",
        "example": "IKEA"
    },
    {
        "field": "item_condition",
        "type": "str",
        "description": "Condition in lowercase, using standard terms: no visible damage, slight wear, moderate wear, significant damage, as new.",
        "example": "no visible damage"
    },
    {
        "field": "number_of_items",
        "type": "int",
        "description": "Integer count of identical items. If there are multiple of the same item, enter their integer count. Example: Single item: 1, Multiple items: 4",
        "example": "1"
    },
    {
        "field": "estimated_worth",
        "type": "float",
        "description": "Estimated value in USD with exactly one decimal place. Guidelines: Always include .0 even for whole numbers. Do not include currency symbols or commas. For multiple items, list the per-item worth.",
        "example": "499.0"
    },
    {
        "field": "estimated_worth_flag",
        "type": "int",
        "description": "Confidence level 1-10 with the following guidelines: 1-3: Highly uncertain, requires expert valuation. 4-6: Moderate confidence, may need verification. 7-8: Good confidence based on visible details. 9-10: Very confident, based on clear brand/model/condition.",
        "example": "7"
    },
    {
        "field": "mentioned_worth",
        "type": "float",
        "description": "Worth mentioned in video, formatted same as estimated_worth. Use NA if not mentioned. Do not use commas in the values. Only full stops.",
        "example": "1500.0"
    },
    {
        "field": "room",
        "type": "str",
        "description": "Room name with proper capitalization. Standard room names: Living Room, Dining Room, Kitchen, Bedroom, Bathroom, Office.",
        "example": "Living Room"
    },
    {
        "field": "timestamp",
        "type": "str",
        "description": "Time in MM:SS format with leading zeros.",
        "example": "01:30"
    },

    {
        "field": "overall_certainty_flag",
        "type": "int",
        "description": "Overall confidence level 1-10 considering all fields: 1-3: Multiple uncertain fields, needs review. 4-6: Some uncertainty in key fields. 7-8: Minor uncertainty in non-critical fields. 9-10: All fields verified with high confidence.",
        "example": "8"
    },
    {
        "field": "is_similar_to",
        "type": "str",
        "description": "Reference to similar item by item_number, or NA. Use for: Matching furniture pieces, Items from same set, Similar models of different sizes.",
        "example": "NA"
    },

    # Note: Tried the following two fields but they seemed too much. Bounding boxes gave out poor results when appending to CSV. Likely best to pass individual images (e.g. frames of video) to the model directly.

    # {
    #    "field": "item_summary",
    #    "type": "str",
    #    "description": "Complete sentence summarizing key details. Include: Item name and brand (if known), Room location, Condition, Worth (estimated or mentioned). Use full stops to separate fields.",
    #    "example": "Samsung 65-inch TV in Living Room. No visible damage. Estimated worth $1500.0"
    # },
    # {
    #     "field": "bounding_boxes",
    #     "type": "list[[int, int, int, int]]",
    #     "description": "Bounding box coordinates without spaces after commas of the target item in format [ymin,xmin,ymax,xmax]. Guidelines: Single item: \"[[ymin,xmin,ymax,xmax]]\". Multiple items: \"[[ymin,xmin,ymax,xmax],[ymin,xmin,ymax,xmax]]...\".",
    #     "example": '"[[[439,410,886,686]]]"'
    # },
]

def get_schema_string() -> str:
  """Returns a simple string representation of the schema."""
  return "\n".join([
      f"{i+1}. {field['field']} ({field['type']}): {field['description']} Example: {field['example']}"
      for i, field in enumerate(csv_schema)
  ])

def get_field_names_as_string() -> str:
  """Returns comma-separated field names for CSV header."""
  return ",".join([field["field"] for field in csv_schema])

def get_field_names_as_list() -> List:
  """Returns a list of field names for CSV header."""
  return [field["field"] for field in csv_schema]

print(f"[INFO] CSV schema string:\n{get_schema_string()}")
print()
print(f"[INFO] CSV header:\n{get_field_names_as_list()}")
```

### Aside: Compare CSV to JSON

Why use CSV?

Short reason: it uses far less tokens than JSON.

This could be very important to reduce the number of output token usage (e.g. to get around the 8k token output limit).

Currently the Gemini models can take in a very large amount of tokens (e.g. 1M tokens for Gemini Flash and 2M tokens for Gemini Pro).

However, if you have a potentially large amount of outputs, the 8k token output limit can be prohibitive.

Not to worry there are a couple of things we can do to save on output tokens.

One of them being to use a CSV-based output rather than JSON.

The following code compares the approximate output lengths and token counts for a given CSV and JSON file.

> **Note:** Since the Gemini models have a [dedicated JSON capability](https://ai.google.dev/gemini-api/docs/structured-output?lang=python#generate-json), they may be better at outputting JSON versus CSV. I haven't done any head to head generation tests. But have found in practice that CSV outputs are quite high quality (with sufficient prompting). My main reasoning for choosing CSV was for the smaller token count outputs which is important for our use case where we are often output length constrained rather than structure or input constrained.

```python
import csv
import json
from io import StringIO

def compare_csv_json(header, num_rows, avg_field_length=1, fill_value="_"):
  """
  Compares base lengths of CSV files and JSON files given a header.
  """
  # Create CSV data
  csv_file = StringIO()
  csv_writer = csv.writer(csv_file)
  csv_writer.writerow(header)
  for _ in range(num_rows):
      csv_writer.writerow([fill_value * avg_field_length for _ in header])
  csv_data = csv_file.getvalue()
  # print(csv_data)

  # Create JSON data
  json_data = json.dumps([{k: fill_value * avg_field_length  for k in header} for _ in range(num_rows)])
  # print(json_data)

  # Compare lengths
  csv_len = len(csv_data)
  json_len = len(json_data)

  # Token counts (1 token = ~4 characters: https://ai.google.dev/gemini-api/docs/tokens?lang=python)
  csv_tokens = csv_len / 4
  json_tokens = json_len / 4

  print(f"[INFO] For {num_rows} rows -> CSV Length: {csv_len} ({csv_tokens} tokens) | JSON Length: {json_len} ({json_tokens} tokens)")

  if csv_len > json_len:
    print(f"[INFO] CSV is longer than JSON: {csv_len} ({csv_tokens} tokens)  vs. {json_len} ({json_tokens} tokens_) ({round(csv_len/json_len, 2)}x bigger)")
  else:
    print(f"[INFO] JSON is longer than CSV: {json_len} ({json_tokens} tokens) vs. {csv_len} ({csv_tokens} tokens) ({round(json_len/csv_len, 2)}x bigger)")

  return " "

# Example usage
# header = ["id", "name", "age"]
header = get_field_names_as_list()
print(f"[INFO] Comparing CSV and JSON lengths with fields:\n{header}\n")
print(compare_csv_json(header=header, num_rows=10, avg_field_length=10))  # For 10 rows
print(compare_csv_json(header=header, num_rows=50, avg_field_length=10))  # For 50 rows
print(compare_csv_json(header=header, num_rows=100, avg_field_length=10))  # For 100 rows
```

## Creating helper functions

Let's create some helper functions to use throughout the notebook.

A few quick ones such as `count_tokens` and `df_to_csv_string`.

Then also one with a few more steps in `quick_check_csv` which takes in a Gemini model's output, extracts and validates the CSV inside and returns the extracted CSV and whether or not fixes are required and the errors that occurred.

```python
def count_tokens(input_prompt, model=model):
  """Returns the total tokens from a given input prompt.

  See the docs on token counting: https://ai.google.dev/gemini-api/docs/tokens?lang=python#text-tokens
  """
  return model.count_tokens(input_prompt).total_tokens

def df_to_csv_string(df) -> str:
  """
  Converts a pandas DataFrame to a CSV string.

  Parameters:
    df (pd.DataFrame): The DataFrame to convert.

  Returns:
    str: The DataFrame as a CSV-formatted string.
  """
  csv_buffer = StringIO()
  df.to_csv(csv_buffer, index=False)
  return csv_buffer.getvalue()

def csv_string_to_df(csv_string):
    """
    Converts a CSV string to a pandas DataFrame.

    Parameters:
      csv_string (str): The CSV-formatted string.

    Returns:
      pd.DataFrame: The resulting DataFrame.
    """
    return pd.read_csv(StringIO(csv_string))
```

```python
import io
import csv

# Extract the CSV content from the input string
def quick_check_csv(model_output,
                    ideal_number_of_fields=len(get_field_names_as_list()),
                    target_start_tag="<csv>",
                    target_end_tag="</csv>"):
  """
  Extracts and validates a CSV from a model's output and performs formatting checks.

  This function extracts a CSV segment from the `model_output` string by locating
  the `target_start_tag` and `target_end_tag`. It validates the CSV's structure using
  Python's `csv` module, ensuring that all rows have the expected number of fields.
  Any formatting issues are detected and optionally fixed by enclosing problematic
  fields with quotes where necessary.

  Args:
      model_output: An object containing the model's output as a string in its `text` attribute.
      ideal_number_of_fields (int, optional): The expected number of fields in each CSV row.
          Defaults to the length of `get_field_names_as_list()`.
      target_start_tag (str, optional): The starting tag indicating the CSV content. Defaults to "<csv>".
      target_end_tag (str, optional): The ending tag indicating the CSV content. Defaults to "</csv>".

  Returns:
      str: The extracted and optionally fixed CSV content.
      bool: A flag indicating if the extracted CSV required fixing.
      list: A list of issues found, including details of rows with formatting problems.
  """

  # Get text output from model
  output_text = model_output.text

  # Quick assertions to make sure target start and end tags are available
  assert target_start_tag in output_text, f"target_start_tag: {target_start_tag} not in model's output text, is there an error?"
  assert target_end_tag in output_text, f"target_end_tag: {target_end_tag} not in model's output text, is there an error?"

  # Baseline filtering
  output_text = output_text.replace("```csv", "").replace("```", "")

  # Extract CSV string from XML tags
  csv_string = output_text.split(target_start_tag)[1].split(target_end_tag)[0].strip()

  # Prepare StringIO objects for input and output
  input_csv = io.StringIO(csv_string)
  output_csv = io.StringIO()

  # Read and write CSV with error handling
  reader = csv.reader(input_csv)
  writer = csv.writer(output_csv)

  # Initialize variables to track issues
  field_counts = []
  fixes_required = []
  fix_csv_required = False

  # Print ideal number of fields
  print(f"[INFO] Ideal number of fields: {ideal_number_of_fields}")

  for i, row in enumerate(reader):
      # For the first row, determine the ideal number of fields, only if ideal_number_of_fields var is not available
      if ideal_number_of_fields == None:
        if i == 0:
            ideal_number_of_fields = len(row)
            print(f"[INFO] Ideal number of fields: {ideal_number_of_fields}")

      # Count fields in each row and compare to the header
      field_counts.append(len(row))
      if len(row) != ideal_number_of_fields:
        error_string = f"[INFO] Row {i} has an unexpected number of fields: {len(row)} (expected: {ideal_number_of_fields})"
        print(error_string)
        fixes_required.append(f"Error: {error_string.replace('[INFO] ', '')} | Row to fix: {','.join(row)}")
        fix_csv_required = True

      # Fix problematic fields if required
      # Enclose fields containing commas, newlines, or quotes
      fixed_row = [
          f'"{field}"'.replace('"""', '"') if any(char in field for char in [",", "\n", '"']) else field for field in row
      ]

      # Write the fixed row to the output
      writer.writerow(fixed_row)

  # Get the fixed CSV data as a string
  output_csv.seek(0)
  output_csv_extracted = output_csv.getvalue()

  # Print a summary of issues, if any
  if fix_csv_required:
      print("[INFO] Some rows required fixing.")
  else:
      print("[INFO] No issues detected in the CSV.")

  # Print the fixed CSV output (for debugging or further use)
  # print(output_csv_fixed)

  return output_csv_extracted, fix_csv_required, fixes_required

class ExampleModelOutput:
  """Simple class to create a `text` attribute, similar to Gemini model outputs."""
  def __init__(self, text):
    self.text = text

example_model_output = ExampleModelOutput(text="""This is an example model output with CSV values from looking at a video.

Row 9 (item_number=9) of the CSV is broken due to having an extra comma in the item_name.

<csv>
item_number,item_name,item_type,item_description,item_brand,item_condition,number_of_items,estimated_worth,estimated_worth_flag,mentioned_worth,room,timestamp,overall_certainty_flag,is_similar_to
1,Modern Sofa,furniture,Gray sectional sofa with chaise,West Elm,excellent,1,1200.0,9,NA,Living Room,00:50,9,NA
2,Smart TV,electronics,55-inch Samsung Smart TV,Samsung,good,1,700.0,8,750.0,Living Room,00:55,8,NA
3,Refrigerator,appliances,Stainless steel double-door refrigerator,LG,good,1,800.0,8,850.0,Kitchen,01:10,8,NA
4,Microwave,appliances,Black countertop microwave,Panasonic,good,1,150.0,7,NA,Kitchen,01:15,7,NA
5,Dining Table,furniture,Glass-top dining table seats six,IKEA,excellent,1,400.0,9,NA,Dining Area,01:40,9,NA
6,Leather Chairs,furniture,Set of four black leather dining chairs,Generic Brand,good,4,200.0,7,NA,Dining Area,01:45,8,NA
7,Queen Bed,furniture,Modern queen-size bed with storage drawers,CB2,excellent,1,900.0,9,NA,Master Bedroom,02:10,9,NA
8,Nightstand,furniture,Wooden nightstand with two drawers,Target,good,2,100.0,7,NA,Master Bedroom,02:15,8,NA
9,Queen,Mattress,furniture,Memory foam queen mattress,Sealy,excellent,1,600.0,9,NA,Master Bedroom,02:20,9,NA
10,Desk,furniture,Standing desk with adjustable height,Autonomous,good,1,350.0,8,NA,Home Office,04:00,8,NA
</csv>
""")

example_output_csv_extracted, example_fix_csv_required, example_fixes_required = quick_check_csv(model_output=example_model_output,
                                                                                                 ideal_number_of_fields=len(get_field_names_as_list()),
                                                                                                 target_start_tag="<csv>",
                                                                                                 target_end_tag="</csv>")


print(f"[INFO] CSV fix required? {example_fix_csv_required}")
print(f"[INFO] Example extracted CSV:\n{example_output_csv_extracted}")
print(f"[INFO] Fixes required:\n{example_fixes_required}")
```

## Workflow outline

For our video to structured data pipeline, we are going to use a recursively workflow.

We'll use three major steps:

1. Video + initial prompt + examples -> check outputs, fix if necessary.
2. Video + secondary prompt + examples -> check outputs, fix if necessary.
3. Video + final prompt + examples -> check outputs, fix if necessary.

And two major model instances:

1. Gemini Model for doing the video inference (with a different input prompt each step).
2. Gemini Model for performing the CSV validation checks (the check is the same for each step).

Each builds upon the previous the outputs of the previous step.

Step 1 produces the initial extraction and details.

Step 2 takes step 1's outputs and tries to expand on them if necessary.

Step 3 reviews the combined outputs of step 1 and 2 and finalizes them.

All major steps use the same Gemini model instance with varying input prompts.

Each step has a verification step to make sure its outputs are valid (e.g. fix the CSV outputs to make sure they are formatted correctly). This verification check is the same for each major step and is performed by a Gemini instance specifically for checking formatting.

The examples for each step are related the to the ideal outputs for each prompt.

The only consistent piece of input data is the target video, which gets referenced at every major step.

See the below workflow chart for an outline of the steps involved.

```python
import base64
from IPython.display import Image, display

def mermaid_graph(graph, scale=2):
    graphbytes = graph.encode("ascii")
    base64_bytes = base64.b64encode(graphbytes)
    base64_string = base64_bytes.decode("ascii")
    # print(base64_string)
    display(
        Image(
            url=f"https://mermaid.ink/img/{base64_string}"
        )
    )

mermaid_graph("""
graph LR;
    A[Start] --> B[Target Video - 165,000 tokens]

    subgraph "Step 1: Initial Extraction"
        B --> C1[Gemini Model Initial Prompt - 10,000 tokens]
        C1 --> D1{CSV Validation}
        D1 -->|Valid| E1[Output 1]
        D1 -->|Invalid| FX1[Gemini CSV Fixer Model]
        FX1 --> D1
    end

    subgraph "Step 2: Expand Extraction"
        B & E1 --> C2[Gemini Model Secondary Prompt - 2000-5000 tokens]
        C2 --> D2{CSV Validation}
        D2 -->|Valid| E2[Output 2]
        D2 -->|Invalid| FX2[Gemini CSV Fixer Model]
        FX2 --> D2
    end

    subgraph "Step 3: Finalize Extraction"
        B & E1 & E2 --> C3[Gemini Model Final Prompt - 2000-5000 tokens]
        C3 --> D3{CSV Validation}
        D3 -->|Valid| E3[Final Output]
        D3 -->|Invalid| FX3[Gemini CSV Fixer Model]
        FX3 --> D3
    end

    E3 --> F[Final Results]

    %% Styling
    style C1 fill:#b3d9ff,stroke:#333,stroke-width:1px
    style C2 fill:#b3d9ff,stroke:#333,stroke-width:1px
    style C3 fill:#b3d9ff,stroke:#333,stroke-width:1px
    style D1 fill:#f9f,stroke:#333,stroke-width:1px
    style D2 fill:#f9f,stroke:#333,stroke-width:1px
    style D3 fill:#f9f,stroke:#333,stroke-width:1px
    style FX1 fill:#ccffcc,stroke:#333,stroke-width:1px
    style FX2 fill:#ccffcc,stroke:#333,stroke-width:1px
    style FX3 fill:#ccffcc,stroke:#333,stroke-width:1px
""")
```

> **Note:** How did you create the workflow flowchart?
>
> After writing the instructions above, I used Gemini to generate [Mermaid](https://mermaid.js.org/) diagram steps to create the flowchart below. For example, "can you turn the following workflow {workflow_steps} into a series of Mermaid steps".

## Prompt 1: Initial Prompt

Our initial prompt's goal is to do a first pass on the video input and extract as many visible and mentioned items as possible in valid CSV format.

We can then refine these initial extracted items with subsequent steps.

### Create examples for initial information extraction

Examples in a prompt help "show" a model what to do rather than "tell".

It's usually best practice to use a combination of both showing and telling.

Since Gemini has such a large input context window, you can use many examples to show the best way to generate outputs based on the given input.

> **Note:** The process of using examples in an input context window is often referred to as "in-context learning" or ICL for short. Meaning, the model adjusts what it outputs based on the examples in the input context. There is also recent research on [Many-Shot ICL](https://arxiv.org/abs/2404.11018) where large numbers of samples are used in the context window to give the model more to learn from and in turn boost performance.

```python
example_1_initial_prompt = """<video_inventory_analysis>

Overview of the Video Content: The video showcases a comprehensive tour of a two-story family home, highlighting various rooms including the living room, kitchen, dining area, bedrooms, bathrooms, and office space. It captures both the layout and the items within each room through steady panning and close-up shots.

Anticipated Challenges:

Occlusions: Some items may be partially hidden behind furniture or other objects, making them difficult to catalog.
Lighting Variations: Different lighting conditions might obscure details necessary for accurate descriptions and condition assessments.
Similar Items: Identifying and differentiating between similar items, especially if they belong to the same set or brand.
Estimation of Worth: Accurately estimating the value of items without explicit brand information or condition details.

Inventory Strategy:

Systematic Room-by-Room Approach: Catalog items sequentially as they appear in each room to ensure no areas are overlooked.
Utilizing Timestamps: Note key timestamps to correlate items with their specific locations and contexts within the video.
Detailed Descriptions: Provide thorough descriptions to capture all relevant details, avoiding ambiguity.

Key Timestamps:

00:15 - Living Room
01:30 - Kitchen
02:45 - Dining Room
04:00 - Master Bedroom
05:15 - Office
06:30 - Bathrooms

Recurring Items/Item Types:

Multiple furniture pieces (sofas, chairs, tables)
Electronic devices (televisions, laptops, kitchen appliances)
Decorative items (lamps, paintings, vases)
Kitchenware and dining utensils

Approach for Estimating Item Values:

Brand Identification: Use visible brand logos or labels to reference standard market prices.
Condition Assessment: Factor in the item's condition to adjust estimated worth accordingly.
Market Research: Cross-reference similar items on retail websites and marketplaces for accurate valuation.

Patterns or Themes Affecting Insurance Coverage:

High-Value Electronics: Multiple expensive electronics may require higher coverage limits.
Collectibles and Art: Unique or valuable decorative items might need specialized coverage.
Uniform Furniture Sets: Items from the same set can affect replacement costs and coverage considerations.

CSV Formatting Strategy:

Consistent Field Order: Adhere strictly to the specified column order to ensure compatibility.
No Commas in Descriptions: Use spaces or other delimiters if necessary to avoid CSV conflicts.
Complete Field Population: Ensure every column is filled, using "NA" where applicable.
Proper Escaping of Special Characters: Handle any internal quotes or special characters appropriately to maintain CSV integrity.

Have I logged all items in the video? No. There are more items to do in a second pass. I will set <item_logging_status> to MORETIMESTAMPSTODO.
</video_inventory_analysis>

<estimated_item_count> ESTIMATED_TOTAL_ITEM_COUNT: 75 </estimated_item_count>

<column_heading_order> item_number,item_name,item_type,item_description,item_brand,item_condition,number_of_items,estimated_worth,estimated_worth_flag,mentioned_worth,room,timestamp,overall_certainty_flag,is_similar_to</column_heading_order>

<csv>
item_number,item_name,item_type,item_description,item_brand,item_condition,number_of_items,estimated_worth,estimated_worth_flag,mentioned_worth,room,timestamp,overall_certainty_flag,is_similar_to
1,Sofa,furniture,Large beige leather couch with chaise lounge and metal legs,NA,no visible damage,1,1200.0,8,NA,Living Room,00:15,8,NA
2,Coffee Table,furniture,Rectangular wooden table with glass top and minimal scratches,IKEA,slight wear,1,250.0,7,NA,Living Room,00:20,8,NA
3,Lamp,lighting,Tall floor lamp with white fabric shade in as-new condition,Target,as new,2,80.0,9,NA,Living Room,00:25,9,NA
4,Television,electronics,55-inch LED smart TV mounted on the wall,Samsung,no visible damage,1,900.0,8,NA,Living Room,00:30,8,NA
5,Dining Table,furniture,Modern glass dining table with stainless steel legs,West Elm,moderate wear,1,600.0,7,NA,Kitchen,01:00,7,NA
6,Chair,furniture,Comfortable upholstered dining chair with wooden frame,IKEA,slight wear,4,75.0,7,NA,Kitchen,01:05,7,NA
7,Refrigerator,appliances,Stainless steel double-door refrigerator with ice dispenser,LG,no visible damage,1,1200.0,8,NA,Kitchen,01:20,8,NA
8,Microwave,appliances,Compact microwave oven with digital display,Panasonic,no visible damage,1,150.0,8,NA,Kitchen,01:25,8,NA
9,Blender,kitchenware,High-speed blender with multiple settings,NA,as new,1,100.0,7,NA,Kitchen,01:30,7,NA
10,Dining Chairs,furniture,Modern upholstered dining chairs with metal legs,West Elm,minor wear,6,200.0,7,NA,Dining Room,02:00,7,NA
<!-- Additional items would follow the same structure -->
</csv>

<last_item_at_timestamp> Last item at timestamp: 06:30 </last_item_at_timestamp>

<item_logging_status> Item logging status: MORETIMESTAMPSTODO </item_logging_status>

<other_details_to_note>
Other details to note:

The video provides clear views of all major rooms, ensuring comprehensive coverage.
No significant occlusions observed, allowing for accurate item cataloging.
Next steps: Review items with estimated_worth_flag below 7 for potential re-evaluation, and confirm coverage for high-value electronics and collectibles.
</other_details_to_note>
"""

example_2_initial_prompt  = """<video_inventory_analysis>
Video Content Overview: The video appears to be a walkthrough of a suburban home, showcasing the contents of each room for insurance purposes. The homeowner moves methodically from room to room, pointing out significant items.

Challenges: Potential challenges include obscured items, items in drawers/cupboards that aren't opened, fast camera movements, and the homeowner's potentially inaccurate estimations of item worth. Lighting might also be an issue in some areas, making it difficult to ascertain the condition of items.

Strategy for Comprehensive Inventory: I will pause the video frequently to ensure all visible items are captured. I'll pay close attention to the homeowner's descriptions and any mentioned details like brand, condition, or purchase price. If an item is only partially visible, I will make a note of it in the overall_certainty_flag field. I will categorize items broadly (furniture, electronics, appliances, decor, etc.) to aid in organization.

Key Timestamps:

00:00-00:30: Living Room
00:30-01:00: Kitchen
01:00-01:30: Dining Room
01:30-02:00: Master Bedroom
02:00-02:30: Guest Bedroom
02:30-03:00: Home Office
03:00-03:30: Garage/Storage

Recurring Items: I anticipate seeing multiple instances of items like chairs, lamps, framed pictures/art, and storage containers.

Approach for Estimating Item Values: I will use the homeowner's mentioned values where available. Otherwise, I'll estimate values based on the item type, brand (if mentioned), condition, and general market prices for similar used items. I will use a conservative estimate if the condition or brand isn't clearly visible.

Patterns/Themes: Depending on the home's decor style (e.g., modern, traditional, minimalist), this might influence the types and values of items. High-end electronics or designer furniture might indicate a need for higher insurance coverage.

CSV Formatting: I will ensure that commas are used only as delimiters between fields. I won't use commas in the item_description field. If a field naturally contains a comma (like a list of items in a set), I will enclose the entire field in double quotes. If double quotes are part of the field value, I will escape them by doubling them (e.g., "" for "). I'll use "NA" for fields where data isn't available or applicable. I will maintain the column order as specified in the schema and make sure all fields are populated. I will always use estimations like 1500.0, and not 1,500.0.

Have I logged all items in the video? No. There are more items to do in a second pass. I will set <item_logging_status> to MORETIMESTAMPSTODO.
</video_inventory_analysis>

<estimated_item_count>
ESTIMATED_TOTAL_ITEM_COUNT: 45
</estimated_item_count>

<column_heading_order>
item_number,item_name,item_type,item_description,item_brand,item_condition,number_of_items,estimated_worth,estimated_worth_flag,mentioned_worth,room,timestamp,overall_certainty_flag,is_similar_to
</column_heading_order>

<csv>
item_number,item_name,item_type,item_description,item_brand,item_condition,number_of_items,estimated_worth,estimated_worth_flag,mentioned_worth,room,timestamp,overall_certainty_flag,is_similar_to
1,Leather Sofa,furniture,Brown leather sofa three seater,Generic Brand,good,1,800.0,7,NA,Living Room,00:05,8,NA
2,Coffee Table,furniture,Wooden coffee table with glass top,IKEA,good,1,100.0,8,NA,Living Room,00:10,2,NA
3,Samsung TV,electronics,55 inch Samsung LED TV,Samsung,excellent,1,600.0,9,650.0,Living Room,00:15,10,NA
4,Floor Lamp,lighting,Metal floor lamp with beige shade,Target,good,2,50.0,7,NA,Living Room,00:20,8,NA
5,Area Rug,decor,8x10 Persian style rug,NA,good,1,300.0,6,NA,Living Room,00:25,7,NA
6,Dining Table,furniture,Round wooden dining table seats 4,Generic Brand,good,1,350.0,7,NA,Dining Room,00:35,8,NA
7,Dining Chairs,furniture,Set of 4 wooden dining chairs,Generic Brand,good,4,50.0,7,NA,Dining Room,00:40,9,6
8,Queen Bed,furniture,Queen size bed with wooden frame,Generic Brand,good,1,400.0,7,NA,Master Bedroom,00:50,8,NA
9,Dresser,furniture,Wooden dresser with six drawers,Generic Brand,good,1,200.0,8,NA,Master Bedroom,00:55,9,NA
10,Desk Lamp,lighting,Metal desk lamp,IKEA,good,1,25.0,9,NA,Master Bedroom,01:00,10,NA
11,Laptop,electronics,15 inch Apple Macbook Pro,Apple,excellent,1,1200.0,10,1300.0,Home Office,01:10,10,NA
12,Office Chair,furniture,Ergonomic office chair,Herman Miller,excellent,1,500.0,9,NA,Home Office,01:15,10,NA
13,Bookshelf,furniture,Tall wooden bookshelf,Generic Brand,good,1,150.0,8,NA,Home Office,01:20,8,NA
14,Washing Machine,appliances,Front loading washing machine,LG,good,1,500.0,8,NA,Garage,01:30,3,15
15,Dryer,appliances,Electric clothes dryer,LG,good,1,400.0,8,NA,Garage,01:35,4,14
...
</csv>

<last_item_at_timestamp>
Last item at timestamp: 09:09
</last_item_at_timestamp>

<item_logging_status>
Item logging status: MORETIMESTAMPSTODO
</item_logging_status>

<other_details_to_note>
Other details to note:
* The homeowner mentioned having a separate storage unit with additional items, which were not shown in the video.
* The home appeared to be well-maintained and in a safe neighborhood, which could positively affect insurance rates. Specific artwork and jewelry were not individually itemized but could be of significant value.
* **Next steps:**
  * Evaluate the items with low certainty scores: 2, 14, 15.
  * Review additional potential locations that may need further inventory, including:
    * **Bathrooms:** Check for items like toiletries, appliances (e.g., hairdryers, electric shavers), and cabinetry.
    * **Laundry Room:** Inventory washers, dryers, ironing boards, and storage for cleaning supplies.
    * **Outdoor Areas:** Assess items in the patio, balcony, or garden such as outdoor furniture, grills, and gardening tools.
    * **Storage Closets:** Ensure all storage areas, including linen closets and utility closets, are comprehensively logged.
    * **Attic or Additional Storage Spaces:** Identify any upper-level storage areas that might contain valuable items.
  * Confirm with the homeowner if there are any additional rooms or areas not covered in the video that should be included in the inventory.
</other_details_to_note>
"""

example_3_initial_prompt  = """<video_inventory_analysis> Video Content Overview: The video is a detailed tour of a modern townhouse, highlighting various rooms and their contents for insurance documentation. The homeowner systematically showcases each area, providing descriptions and occasionally mentioning the value of specific items.

Challenges: Challenges may include items partially out of frame, reflections or glare obscuring item details, rapid transitions between rooms, and items that are stacked or stored in closets and cabinets without being fully displayed. Additionally, accurately estimating the value of high-tech gadgets and custom furniture without precise brand information could be difficult.

Strategy for Comprehensive Inventory: I will meticulously review each segment of the video, pausing as necessary to capture all visible items. Emphasis will be placed on items that the homeowner highlights or provides additional information about. I'll categorize items by their type and room to maintain organization. For items not fully visible, I'll note the uncertainty in the overall_certainty_flag and, where possible, reference similar items to ensure they are included in the inventory.

Key Timestamps:
00:00-00:45: Entrance Hall
00:45-01:30: Living Room
01:30-02:15: Kitchen
02:15-03:00: Dining Area
03:00-03:45: Master Bedroom
03:45-04:30: Guest Bedroom
04:30-05:15: Home Office
05:15-06:00: Basement
06:00-06:45: Garage

Recurring Items: Multiple electronic devices such as televisions and gaming consoles, various types of seating including sofas and chairs, kitchen appliances like refrigerators and microwaves, and storage units like cabinets and closets.

Approach for Estimating Item Values: Values will be based on the homeowner’s provided information when available. For other items, I'll reference current market prices for similar items considering their condition and brand reputation. High-value items like electronics and custom furniture will receive more precise estimates, while standard items will have conservative valuations.

Patterns/Themes: The townhouse features a contemporary design with a focus on smart home technology and minimalist furniture. This theme suggests a prevalence of high-tech gadgets and designer furniture, which may require higher insurance coverage. Additionally, the use of sustainable materials and eco-friendly products could influence item valuations and insurance considerations.

CSV Formatting: Adhering strictly to the CSV formatting guidelines, I will ensure that commas are only used as delimiters, avoid commas in the item_description field, and properly escape any necessary characters. All columns will be included in the specified order, and missing data will be filled with "NA". Numerical values, especially prices, will exclude commas to maintain validity. The bounding_boxes field will accurately represent the item's location within the video frame.

Have I logged all items in the video? No. There are more items to do in a second pass. I will set <item_logging_status> to MORETIMESTAMPSTODO.

</video_inventory_analysis>

<estimated_item_count> ESTIMATED_TOTAL_ITEM_COUNT: 62 </estimated_item_count>

<column_heading_order> item_number,item_name,item_type,item_description,item_brand,item_condition,number_of_items,estimated_worth,estimated_worth_flag,mentioned_worth,room,timestamp,overall_certainty_flag,is_similar_to </column_heading_order>

<csv>
item_number,item_name,item_type,item_description,item_brand,item_condition,number_of_items,estimated_worth,estimated_worth_flag,mentioned_worth,room,timestamp,overall_certainty_flag,is_similar_to
1,Modern Sofa,furniture,Gray sectional sofa with chaise,West Elm,excellent,1,1200.0,9,NA,Living Room,00:50,9,NA
2,Smart TV,electronics,55-inch Samsung Smart TV,Samsung,good,1,700.0,8,750.0,Living Room,00:55,8,NA
3,Refrigerator,appliances,Stainless steel double-door refrigerator,LG,good,1,800.0,8,850.0,Kitchen,01:10,8,NA
4,Microwave,appliances,Black countertop microwave,Panasonic,good,1,150.0,7,NA,Kitchen,01:15,7,NA
5,Dining Table,furniture,Glass-top dining table seats six,IKEA,excellent,1,400.0,9,NA,Dining Area,01:40,9,NA
6,Leather Chairs,furniture,Set of four black leather dining chairs,Generic Brand,good,4,200.0,7,NA,Dining Area,01:45,8,NA
7,Queen Bed,furniture,Modern queen-size bed with storage drawers,CB2,excellent,1,900.0,9,NA,Master Bedroom,02:10,9,NA
8,Nightstand,furniture,Wooden nightstand with two drawers,Target,good,2,100.0,7,NA,Master Bedroom,02:15,8,NA
9,Queen Mattress,furniture,Memory foam queen mattress,Sealy,excellent,1,600.0,9,NA,Master Bedroom,02:20,9,NA
10,Desk,furniture,Standing desk with adjustable height,Autonomous,good,1,350.0,8,NA,Home Office,04:00,8,NA
11,Office Chair,furniture,Ergonomic mesh office chair,Herman Miller,excellent,1,550.0,9,NA,Home Office,04:05,9,NA
12,Laptop,electronics,15-inch Dell XPS 15 Laptop,Dell,excellent,1,1200.0,10,1300.0,Home Office,04:10,10,NA
13,Bookshelf,furniture,Five-tier wooden bookshelf,Home Depot,good,1,180.0,7,NA,Home Office,04:15,8,NA
14,Gaming Console,electronics,PlayStation 5 with two controllers,Sony,excellent,1,500.0,9,550.0,Living Room,00:58,9,NA
15,Blender,appliances,High-speed countertop blender,Ninja,good,1,100.0,7,NA,Kitchen,01:20,7,NA
16,Toaster,appliances,2-slice stainless steel toaster,Black+Decker,good,1,40.0,6,NA,Kitchen,01:25,7,NA
17,Vacuum Cleaner,appliances,Robotic vacuum cleaner,iRobot,excellent,1,300.0,9,NA,Garage,05:20,9,NA
18,Mountain Bike,fitness,Bronze-colored mountain bike,Schwinn,good,2,250.0,7,NA,Garage,05:25,7,NA
19,Workbench,furniture,Metal workbench with tools,Generic Brand,good,1,400.0,7,NA,Basement,04:50,7,NA
20,Tool Set,tools,Comprehensive tool set with 50 pieces,Stanley,excellent,1,150.0,9,NA,Basement,04:55,9,NA
21,Garage Storage Shelves,furniture,Adjustable metal storage shelves,Home Depot,good,3,180.0,7,NA,Garage,05:10,7,NA
22,Car Lift,equipment,Electric car lift for two vehicles,NA,good,1,2500.0,6,NA,Garage,05:30,6,NA
23,Air Conditioner,appliances,Window-mounted AC unit,Carrington,good,2,350.0,7,NA,Living Room,00:40,7,NA
24,Ceiling Fan,lighting,Modern ceiling fan with remote control,Westinghouse,excellent,2,120.0,8,NA,Living Room,00:35,8,NA
25,Wall Art,decor,Abstract paintings set of three,Local Artist,good,3,300.0,7,NA,Living Room,00:25,7,NA
26,Mirror,decor,Full-length framed mirror,Generic Brand,fair,1,80.0,5,NA,Guest Bedroom,03:20,6,NA
27,Bedside Lamp,lighting,Table lamp with LED bulb,Philips,good,2,40.0,7,NA,Master Bedroom,02:18,7,NA
28,Television Stand,furniture,Wooden TV stand with storage,Sauder,good,1,150.0,7,NA,Living Room,00:60,7,NA
29,Speaker System,electronics,Surround sound speaker system,Bose,excellent,5,800.0,9,850.0,Living Room,00:65,9,NA
30,Smart Thermostat,home_automation,Wi-Fi enabled smart thermostat,Nest,excellent,1,200.0,9,NA,Living Room,00:70,9,NA
31,Fire Extinguisher,safety,Standard ABC fire extinguisher,NA,good,2,50.0,6,NA,Kitchen,01:35,6,NA
32,Smoke Detector,safety,Interconnected smoke detectors,NA,excellent,4,30.0,8,NA,Various Rooms,00:05,8,NA
33,Router,networking,Wireless internet router,Netgear,excellent,1,100.0,9,NA,Home Office,04:20,9,NA
34,Printer,electronics,All-in-one wireless printer,HP,good,1,120.0,7,NA,Home Office,04:25,7,NA
35,Electric Kettle,appliances,Stainless steel electric kettle,Breville,excellent,1,60.0,9,NA,Kitchen,01:30,9,NA
36,Blinds,decor,Automated motorized window blinds,Somfy,excellent,6,200.0,8,NA,Various Rooms,00:10,8,NA
37,Electric Grill,appliances,Indoor electric grill,George Foreman,good,1,80.0,7,NA,Kitchen,01:40,7,NA
38,Water Heater,appliances,Electric water heater,Rheem,good,1,500.0,7,NA,Garage,05:35,7,NA
39,Dehumidifier,appliances,Portable dehumidifier,Frigidaire,good,1,150.0,7,NA,Basement,04:40,7,NA
40,Ceiling Light,lighting,LED ceiling light fixture,Philips,excellent,3,90.0,8,NA,Various Rooms,00:15,8,NA
41,Area Lamp,lighting,Modern floor lamp with adjustable arm,IKEA,good,2,70.0,7,NA,Living Room,00:60,7,NA
42,Trash Can,household,Stainless steel kitchen trash can,Simplehuman,excellent,2,90.0,9,NA,Kitchen,01:50,9,NA
43,Fireplace,electrical,Electric fireplace with remote control,Duraflame,good,1,300.0,7,NA,Living Room,00:20,7,NA
44,Barbecue Grill,appliances,Gas-powered outdoor barbecue grill,Weber,excellent,1,600.0,9,NA,Garage,05:40,9,NA
45,Storage Bins,storage,Plastic storage bins set of five,IRIS,good,5,75.0,7,NA,Basement,04:45,7,NA
46,Air Purifier,appliances,HEPA air purifier,Blueair,excellent,1,250.0,9,NA,Home Office,04:30,9,NA
47,Electric Fireplace Tools,tools,Set of 10 fireplace tools,Generic Brand,good,10,50.0,6,NA,Basement,04:55,6,NA
48,Wall Clock,decor,Large analog wall clock,Generic Brand,fair,1,40.0,5,NA,Living Room,00:30,5,NA
49,Bean Bag Chair,furniture,Blue bean bag chair in good condition,Big Joe,good,2,80.0,7,NA,Living Room,00:75,7,NA
50,Floor Mat,decor,Entrance floor mat,NA,good,3,30.0,6,NA,Entrance Hall,00:20,6,NA
51,Desk Organizer,office_supplies,Plastic desk organizer with compartments,SimpleHouseware,good,1,25.0,7,NA,Home Office,04:12,7,NA
52,Surge Protector,electronics,8-outlet surge protector with USB ports,APC,excellent,1,45.0,9,NA,Home Office,04:18,9,NA
53,Wall Shelf,furniture,Floating wall shelf with brackets,Generic Brand,good,2,60.0,7,NA,Home Office,04:22,7,NA
54,Ceiling Medallion,decor,Decorative ceiling medallion,NA,fair,1,20.0,5,NA,Master Bedroom,02:25,5,NA
55,Window AC Unit,appliances,Split window air conditioner,Frigidaire,good,1,400.0,7,NA,Guest Bedroom,03:25,7,NA
56,Heater,appliances,Portable electric heater,DeLonghi,good,1,100.0,7,NA,Basement,04:35,7,NA
57,Security Camera,security,Indoor security camera with night vision,Arlo,excellent,4,200.0,9,NA,Various Rooms,00:08,9,NA
58,Door Lock,security,Smart door lock with keypad,August,excellent,1,250.0,9,NA,Entrance Hall,00:12,9,NA
59,Carbon Monoxide Detector,safety,Interconnected CO detectors,NA,excellent,2,40.0,8,NA,Various Rooms,00:07,8,NA
60,Fireplace Tools,tools,Set of 5 fireplace tool set,Generic Brand,good,5,60.0,7,NA,Living Room,00:22,7,NA
</csv>

<last_item_at_timestamp> Last item at timestamp: 06:45 </last_item_at_timestamp>

<item_logging_status> Item logging status: MORETIMESTAMPSTODO </item_logging_status>

<other_details_to_note> Other details to note:

* The townhouse is equipped with several smart home devices that enhance security and energy efficiency.
* High-value items such as the car lift and electric fireplace may require additional coverage considerations.
* The homeowner has organized storage efficiently in the garage and basement, which may help in reducing insurance premiums.
* Some items in the basement appear to be used for hobby or DIY projects, which could introduce additional risk factors for insurance.
* Next steps: Evaluate the items with low certainty scores: 16, 22, 26, 31, 47, 48, 50, 54. Review additional potential locations that may need further inventory, including:
Bathrooms: Check for items like toiletries, appliances (e.g., hairdryers, electric shavers), and cabinetry.
Laundry Room: Inventory washers, dryers, ironing boards, and storage for cleaning supplies.
Outdoor Areas: Assess items in the patio, balcony, or garden such as outdoor furniture, grills, and gardening tools.
Storage Closets: Ensure all storage areas, including linen closets and utility closets, are comprehensively logged.
Basement Workshop: If present, verify tools, machinery, and other specialized equipment not fully covered.
Attic or Additional Storage Spaces: Identify any upper-level storage areas that might contain valuable items.
Confirm with the homeowner if there are any additional rooms or areas not covered in the video that should be included in the inventory.
</other_details_to_note>
"""

example_4_initial_prompt  = """
<video_inventory_analysis>
Video Content Overview: The video is a homeowner's walkthrough of their property for home insurance purposes. The homeowner verbally describes items while showing them on camera.

Challenges: Some items may be partially obscured, or reflections might make details difficult to see. The homeowner might not know the exact brand or value of every item. There might be background noise or interruptions.

Strategy for Comprehensive Inventory: I will watch the video carefully, pausing frequently to capture all visible items. I'll listen closely to the homeowner's descriptions, noting brands, models, and any mentioned value estimations. If anything is unclear, I will note it in the overall_certainty_flag and item_description. I'll use a consistent naming convention for similar items (e.g., "bedroom dresser," "living room dresser").

Key Timestamps: Since no video is provided, I will make placeholders. This should be filled in when completing the actual task.
00:00-01:00: Living Room
01:00-02:00: Kitchen
02:00-03:00: Dining Room
03:00-04:00: Master Bedroom

Recurring Items: I expect to see multiple lamps, chairs, decorative items, and potentially sets of dishes or cookware.

Approach for Estimating Item Values: I'll prioritize the homeowner’s stated values. For items without explicit values, I'll research current market prices for similar used items, adjusting for condition and brand as necessary. If no brand is discernible, I'll use generic pricing. Estimates will err on the side of caution.

Patterns/Themes: The style and quality of furnishings can indicate overall value and potential insurance needs. High-value items, collections, or specialized equipment will be noted for potential additional coverage.

CSV Formatting: I will adhere strictly to the CSV formatting rules. Commas will only be used as field delimiters. The item_description field will avoid commas, and double quotes will be used (and properly escaped) where necessary. I will use "NA" for missing data. The specified column order will be followed. Prices will be formatted without commas (e.g., 1500.00, not 1,500.00). I will be sure to add bounding boxes where I can see an item.

Have I logged all items in the video? No. There are more items to do in a second pass. I will set <item_logging_status> to MORETIMESTAMPSTODO.
</video_inventory_analysis>

<estimated_item_count>
ESTIMATED_TOTAL_ITEM_COUNT: 20
</estimated_item_count>

<column_heading_order>
item_number,item_name,item_type,item_description,item_brand,item_condition,number_of_items,estimated_worth,estimated_worth_flag,mentioned_worth,room,timestamp,overall_certainty_flag,is_similar_to
</column_heading_order>

<csv>
item_number,item_name,item_type,item_description,item_brand,item_condition,number_of_items,estimated_worth,estimated_worth_flag,mentioned_worth,room,timestamp,overall_certainty_flag,is_similar_to
1,Sofa,furniture,Sectional sofa,Ashley Furniture,Good,1,600.00,7,NA,Living Room,00:05,8,NA
2,Coffee Table,furniture,Wood and glass coffee table,IKEA,Good,1,100.00,8,NA,Living Room,00:07,9,NA
3,Armchair,furniture,Beige fabric armchair,NA,Fair,2,150.00,6,NA,Living Room,00:10,7,14
4,Television,electronics,55-inch LCD TV,Samsung,Good,1,400.00,7,500.00,Living Room,00:12,8,NA
5,Bookshelf,furniture,Tall wooden bookshelf,NA,Good,1,200.00,7,NA,Living Room,00:15,8,NA
6,Refrigerator,appliances,Stainless steel refrigerator,LG,Excellent,1,1200.00,9,NA,Kitchen,01:02,9,NA
7,Microwave,appliances,Over-the-range microwave,Whirlpool,Good,1,200.00,7,NA,Kitchen,01:05,8,NA
8,Dishwasher,appliances,Built-in dishwasher,GE,Good,1,300.00,7,NA,Kitchen,01:08,8,NA
9,Dining Table,furniture,Wooden dining table with 4 chairs,NA,Good,1,350.00,7,NA,Dining Room,02:05,8,NA
10,Dining Chairs,furniture,Wooden dining chair,NA,Good,4,50.00,7,NA,Dining Room,02:07,8,9
11,Bed,furniture,Queen-size bed frame,NA,Good,1,300.00,7,NA,Master Bedroom,03:02,8,NA
12,Mattress,furniture,Queen-size mattress,Sealy,Good,1,500.00,7,NA,Master Bedroom,03:05,7,NA
13,Dresser,furniture,Wooden dresser with 6 drawers,NA,Good,1,250.00,7,NA,Master Bedroom,03:08,8,NA
14,Nightstand,furniture,Small wooden nightstand,NA,Fair,2,50.00,6,NA,Master Bedroom,03:10,7,3
15,Lamp,lighting,Table lamp,NA,Good,2,40.00,7,NA,Master Bedroom,03:12,8,NA
16,Painting,decor,Framed landscape painting,NA,Good,1,100.00,5,NA,Living Room,00:18,6,NA
17,Area Rug,decor,Large area rug,NA,Good,1,200.00,6,NA,Living Room,00:20,7,NA
18,Kitchen Cabinets,furniture,Built-in kitchen cabinets,NA,Good,NA,500.00,6,NA,Kitchen,01:15,5,NA
19,Cookware Set,kitchenware,10-piece cookware set,Tramontina,Good,1,150.00,7,NA,Kitchen,01:20,8,NA
20,Towel Set,bathroom,6-piece towel set,NA,New,1,50.00,8,NA,Master Bedroom,03:15,9,NA
</csv>

<last_item_at_timestamp>
Last item at timestamp: 04:00
</last_item_at_timestamp>

<item_logging_status>
Item logging status: MORETIMESTAMPSTODO
</item_logging_status>

<other_details_to_note>
Other details to note: This is a placeholder for a video that wasn't provided. If there were a video, I'd add notes about anything unusual, such as high-value items or potential hazards. I'd also specify areas not captured in the video, like attics, basements, or garages.

Next steps:

Review Items with Low Certainty: Re-evaluate items marked with low overall_certainty_flag scores (e.g., items 3, 14, 16, 17, 18) to improve descriptions, confirm quantities, and obtain brand information if possible. Cross-reference with similar items to ensure accurate valuations. For example, compare the "Beige fabric armchair" (item 3) with other armchairs or seating to refine the estimated worth. Scrutinize the "Large area rug" (item 17) for distinguishing features or patterns to assist with identification and valuation.

Confirm Missing Brand Information: Attempt to identify the brands for items currently listed as "NA" (e.g., Armchair, Bookshelf, Dining Table, Dining Chairs, Bed, Nightstand, Painting, Area Rug, Kitchen Cabinets, Towel Set). Even partial brand information can improve valuation accuracy. Consider online searches for similar items based on visual characteristics.

Address Missing Quantities: Determine the number_of_items for "Kitchen Cabinets" (item 18). This may require revisiting the video or requesting clarification from the homeowner. If precise counts are unavailable, provide a reasonable estimate and note the uncertainty.

Inventory Untracked Areas: Identify and document items in any areas not covered in the initial walkthrough. Specifically inquire about and document the contents of the attic, basement, garage, any storage units (onsite or offsite), closets, and drawers. Request additional video footage or photos of these areas.

Inquire About Valuables: Ask the homeowner about any high-value items, collections (e.g., jewelry, art, antiques, stamps, coins), or specialized equipment that may require separate appraisals and specific insurance riders. Document these items with detailed descriptions, provenance (if known), and any existing appraisal
"""
```

### Prompt model to extract information from video file

We'll now write a detailed prompt for our model to extract the initial information from an input video.

> **Note:** I can't stress more that this is an experimental process. The best way to achieve a result is likely extremely task-specific. The prompt below was an evolution from "extract all the household items in the attached video in CSV form" to what you see below. It could likely be improved if I had more time. For now, it works great as a proof of concept.

Some guidelines on writing a prompt:

- Use examples where possible (e.g. "show" *and* "tell").

- Be specific for what you're asking for (e.g. "output timestamps in `MM:SS` format").

- Use string formating to add dynamic and changing information (e.g. `"a string with a {format_option}.format(format_option="format option can go a long way")`).

- By telling a model to "think" about what it's about to do, it can help get the ball rolling. Just like if you were to take on a task but plan out the steps beforehand, it can help clarify your thoughts. This is known as ["chain of thought" prompting](https://www.promptingguide.ai/techniques/cot).

The best thing you can do is to *experiment, experiment, experiment*!

Start simple and increase complexity as you find errors and edge cases.

Once you've got a few good output examples, you can use your set of instructions plus a Large Lanuage Model (LLM) such as Gemini via AI Studio to craft more examples in a similar fashion.

> **Note:** For more resources on prompting techniques I'd check out the following:
> * https://www.promptingguide.ai/
> * https://ai.google.dev/gemini-api/docs/prompting-intro

```python
input_prompt_initial = """You are an expert home inventory logger and home insurance reviewer. Your task is to catalog household items from a smartphone recorded video for home and contents insurance purposes.
Your goal is to create a comprehensive and accurate inventory of all items visible or mentioned in the attached video.

Before you begin cataloging items, please wrap your analysis in <video_inventory_analysis> tags to break down the video content, plan your inventory, and resolve any uncertainties. This analysis should include:

1. A brief overview of the video content
2. Any challenges you anticipate in cataloging the items
3. Your strategy for ensuring a comprehensive inventory
4. Key timestamps for different rooms or sections of the house
5. A list of recurring items or item types you notice
6. Your approach for estimating item values
7. Any patterns or themes in the household items that might affect insurance coverage
8. How best to format the CSV so that it can be returned in a valid format based on the schema (e.g. "I should not include commas in item description fields", "I should make sure all columns are returned in the right order" and "I should make sure every required field has a value")
9. A note of when the timestamp when the last item occurs in MM:SS format in the video and explain why this is important for capturing all the items in the video.
10. Ask yourself the question, "Have I logged all items in the video?" Only answer "yes" if you are 100% confident all items have been logged. If there are more items to do, answer "no". Just because you watch the whole video, doesn't mean all items have been logged. Update <item_logging_status> based on your answer to this question.

Once you've completed your analysis, create a CSV file with the following columns/schema:
{csv_input_schema}

Follow these instructions:
1. Watch the entire video and count all mentioned items.
2. List the total item count like this: ESTIMATED_TOTAL_ITEM_COUNT: <item_count_as_int>
3. Create a single line in the CSV for every unique item in the video.
4. For multiple identical items (e.g. 3 of the same chair or 4 of the same speaker), update the number_of_items field instead of creating separate entries. Use integer values to indicate their counts. Count as best you can.
5. Use simple but understandable descriptions for each item.
6. If an item is similar to another item in the inventory, note this in the is_similar_to field.
7. Make sure there are no blank fields, if something needs to be blank, fill it with NA.
8. Write a final timestamp note at the end to remember how long the entire video is (this will help keep track if you've made it through the whole video), this should be MM:SS format.
9. If there are more items to track, update <item_logging_status> to "MORETIMESTAMPSTODO", if all items have been completed to 100% capacity, update <item_logging_status> to "ALLTIMESTAMPSDONE". This is very important, it's okay to miss some items the first time round, you can always go back over them in a second pass.
10. Include a note to yourself in <other_details_to_note> if you think there are more items to continue doing in the video, this should be inline with the <item_logging_status> as well as the next steps to improve upon the record keeping.

CSV Formatting Instructions:
- Use commas only to separate fields rather than inside fileds.
- Do not use commas in the item_description field.
- For fields that inherently include commas, enclose the entire value in double quotes (`"`).
- If double quotes appear within a field value, escape them by doubling them (`""`). For example, `John "JJ" Smith` should be written as `"John ""JJ"" Smith"`.
- Do not use any unescaped double quotes or other special characters that may cause the CSV to be invalid.
- Never use commas in prices. For example, $1500.0 = good, $1,500.0 = bad.
- Return all columns in the order they are presented in. Do not reorder the columns. Write these down in <column_heading_order> tags so you don't forget.
- Ensure every column has a value, do not miss a column.

Return the full and valid CSV within <csv> tags so it can be easily parsed out.

Your response should look like the following examples (except you will update the values to reflect the items in the given video):

<begin_examples>
<example_1>
{example_1}
</example_1>
<example_2>
{example_2}
</example_2>
<example_3>
{example_3}
</example_3>
<example_4>
{example_4}
</example_4>
</end_examples>

Final note: If you think there are other important details mentioned in the video that should be included, please note them after the CSV output in <other_details_to_note>.
"""

input_prompt_initial = input_prompt_initial.format(csv_input_schema=get_schema_string(),
                                                   example_1=example_1_initial_prompt,
                                                   example_2=example_2_initial_prompt,
                                                   example_3=example_3_initial_prompt,
                                                   example_4=example_4_initial_prompt)

print(input_prompt_initial)
```

### Count our input tokens

Now we've got out video input and text-based prompt, let's count how many tokens there are.

```python
token_count_input_prompt_text_only = count_tokens(input_prompt_initial)
token_count_input_prompt_with_video = count_tokens([video_file_upload, input_prompt_initial])

print(f"[INFO] Token count with text only: {token_count_input_prompt_text_only}")
print(f"[INFO] Token count with text and video: {token_count_input_prompt_with_video}")
```

How are these calculated?

For video, individual frames are 258 tokens, and audio is 32 tokens per second.

With metadata, each second of video becomes ~300 tokens, which means a 1M context window can fit slightly less than an hour of video.

> **Note:** See the documentation on video inputs here: https://ai.google.dev/gemini-api/docs/vision?lang=python#technical-details-video

### Ideas for getting around the output token limit

With a large input context window, it's likely we won't be limited by input context.

Instead, we'll likely be limited by output tokens (~8k token utput = ~2000 words).

Some ideas for getting around the output token limit include:

* Count the items before going through the video (e.g. get Gemini to produce a count of items in the video and if the outputs don't contain this many items, reprompt).
* Do multiple rounds of prompting to make sure that all the items in the video are covered (this is the approach we're going to take).
* Tell the model to output a special status tracking phrase (e.g. `"MOREITEMSTODO"`) if it hasn't reached the end (this is what we'll do as well).
* Use a smaller output style (e.g. CSV vs JSON) to save on output tokens.

> **Note:** For an example of continuing prompts where the output limit might be a problem, see the Story Writing with Prompt Chaining notebook in the Gemini cookbook repo: https://github.com/google-gemini/cookbook/blob/main/examples/Story_Writing_with_Prompt_Chaining.ipynb

### Generate outputs without using a context cache

A context cache is helpful if you feel like you want to make multiple passes on the same data (e.g. perform inference over a long video input multiple times).

However, if you only want to perform inference in one step, you can generate outputs without using a context cache.

To do so, we'll pass our `video_file_upload` as well as our `input_prompt_initial` in a list to our `generate_content()` function.

```python
# Generate outputs without using a cache
start_time = time.time()
model_response_1_no_cache = generate_content(prompt=[video_file_upload,
                                                     input_prompt_initial],
                                             model=model_gemini_flash)

end_time = time.time()
model_response_1_no_cache_time = end_time - start_time

print(f"[INFO] Time taken for model_reponse_1 (no cache): {round(model_response_1_no_cache_time, 2)} seconds")
print(model_response_1_no_cache.text)
```

Nice! Looks like our model output a good looking CSV as well as the string `MORETIMESTAMPSTODO` indicating that there are items still to be done.

We can check the number of tokens our model used (inputs and outputs) using the `.useage_metadata` attribute on the response.

```python
# Check the tokens
model_response_1_no_cache.usage_metadata
```

### Generate outputs using a context cache

If you're going to do multiple rounds of generation based on the same inputs (e.g. performing inference on the same large video file multiple times like us), it makes sense to store it in a context cache.

Input [charges on context cache tokens are 4x less than regular input tokens](https://ai.google.dev/pricing#1_5flash) (e.g. $0.01875 vs $0.075 / 1 million tokens for Gemini Flash).

Generating outputs with a context cache acts as a prefix to the input.

So if you had an input prompt called `input_prompt_initial`, the model will generate outputs based on `[context_cache, input_prompt_initial]`.

> **Note:** For more on creating context caches, interacting with them and performing inference with them, see the Gemini documentation on caching: https://ai.google.dev/gemini-api/docs/caching

We can create a `GenerativeModel` which uses a cache by using `genai.GenerativeModel.from_cached_content(cached_content=CACHED_CONTENT_TO_USE)`

In our case, the cached content we want our model to use are `video_file_cache`.

```python
# Create a model with caching
model_gemini_flash_with_cache = genai.GenerativeModel(generation_config=generation_config).from_cached_content(cached_content=video_file_cache) # Note: Cached model defaults to the model we used to create the cache, in our case, `models/gemini-1.5-flash-002`
print(f"[INFO] Using model with cache:\n {model_gemini_flash_with_cache}")
```

Now we've got a model ready to generate from cache in `model_gemini_flash_with_cache`, we can use it to generate outputs based on the `cached_content` as well as the `input_prompt_initial`.

```python
# Generate outputs using a cache
start_time = time.time()
model_response_1_with_cache = generate_content(prompt=[input_prompt_initial], # Note: Generating from cache means the input prompt will be [video_file_cache, input_prompt_initial]
                                               model=model_gemini_flash_with_cache)

end_time = time.time()
model_response_1_with_cache_time = end_time - start_time

print(f"[INFO] Time taken for model_reponse_1 (with cache): {round(model_response_1_with_cache_time, 2)} seconds")
print(model_response_1_with_cache.text)
```

```python
# Check the tokens
model_response_1_with_cache.usage_metadata
```

Notice the token counts have now changed, the vast majority of our token usage comes from cached tokens (`cached_content_token_count`). These tokens are priced at 4x less than regular tokens.

### Compare pricing of with/without cache

If you're going to be running many prompts or creating an application where users are going to be providing many inputs and your app providing the outputs, it makes sense to calculate the pricing.

Let's create a few dictionaries to store the costs of various Gemini models in $USD/1 million tokens.

And we'll also create a few helper functions to count tokens (inputs, outputs and cache) for model input/outputs as well as the pricing assosciated.

> **Note:** You can see the full list of pricing for Gemini models on the pricing page: https://ai.google.dev/pricing

```python
# Pricing tiers (all prices are $USD per 1 million tokens)
gemini_flash_input_under_128k = {"input": 0.075,
                                 "output": 0.30,
                                 "context_caching": 0.01875,
                                 "context_caching_storage": 1.00}

gemini_flash_input_over_128k = {"input": 0.15,
                                "output": 0.60,
                                "context_caching": 0.0375,
                                "context_caching_storage": 1.00}

gemini_pro_input_under_128k = {"input": 1.25,
                               "output": 5.0,
                               "context_caching": 0.3125,
                               "context_caching_storage": 4.50}

gemini_pro_input_over_128k = {"input": 2.50,
                              "output": 10.00,
                              "context_caching": 0.625,
                              "context_caching_storage": 4.50}

def get_token_counts(response):
  """Returns input, output and cached tokens for a given model response."""

  # Default cached tokens to 0
  cached_tokens = 0

  usage_metadata = response.usage_metadata
  input_tokens = usage_metadata.prompt_token_count
  output_tokens = usage_metadata.candidates_token_count

  if hasattr(usage_metadata, "cached_content_token_count"):
    cached_tokens = usage_metadata.cached_content_token_count
    input_tokens = input_tokens - cached_tokens # remove number of cached tokens from total input tokens - these are priced differently

  return input_tokens, output_tokens, cached_tokens

def pricing_calculator(response, pricing_dict, cache_time_hours=CACHE_HOURS):
  """Caculates costs for a given response based on a pricing schema."""

  # Default cached tokens and price to 0
  cached_tokens = 0
  cached_price = 0

  # Get token counts for a given response
  input_tokens, output_tokens, cached_tokens = get_token_counts(response)

  input_price = round(pricing_dict["input"] * (input_tokens / 1_000_000), 6) # prices are per 1M tokens
  output_price = round(pricing_dict["output"] * (output_tokens / 1_000_000), 6)
  cached_tokens_price = round(pricing_dict["context_caching"] * (cached_tokens / 1_000_000), 6)

  cached_storage_price = round(pricing_dict["context_caching_storage"] * (cached_tokens / 1_000_000 * cache_time_hours), 6)

  total_price = round(input_price + output_price + cached_tokens_price + cached_storage_price, 6)

  cache_use = False
  if cached_tokens_price > 0:
    cache_use = True

  print(f"[INFO] Pricing for target response ($USD) using cache: {cache_use} |\
  Input: {input_price} ({input_tokens} tokens) | Output: {output_price} ({output_tokens} tokens) |\
  Cached context: {cached_price} ({cached_tokens}) | Cache storage: {cached_storage_price} ({cached_tokens} tokens for {cache_time_hours} hours) |\
  Total: {total_price} ({total_price} tokens)")

  return {"input_price": input_price,
          "output_price": output_price,
          "cached_tokens_price": cached_tokens_price,
          "cached_storage_price": cached_storage_price,
          "cache_storage_time": 0 if cached_storage_price == 0 else cache_time_hours,
          "total_price": total_price,
          "using_cache": cache_use}
```

Now we've got some pricing dictionaries as well as pricing calculator helper functions, let's calculate how much each of our model's responses cost.

We'll start with the pricing without using a cache.

```python
# Caculate pricing without using cache
pricing_without_cache = pricing_calculator(response=model_response_1_no_cache,
                                           pricing_dict=gemini_flash_input_over_128k)
pricing_without_cache
```

```python
# Calculate pricing using a cache stored for 2 hours
pricing_with_cache_2_hours = pricing_calculator(response=model_response_1_with_cache,
                                                pricing_dict=gemini_flash_input_over_128k,
                                                cache_time_hours=2) # Note: For large cached inputs you will notice the cached storage price go up quite a bit depending on how long you store cached tokens for.
pricing_with_cache_2_hours
```

```python
# Calculate pricing using a cache stored for 15 minutes
pricing_with_cache_15_minutes = pricing_calculator(response=model_response_1_with_cache,
                                                    pricing_dict=gemini_flash_input_over_128k,
                                                    cache_time_hours=0.25) # Note: For large cached inputs you will notice the cached storage price go up quite a bit depending on how long you store cached tokens for.
pricing_with_cache_15_minutes
```

```python
# Turn pricing dictionaries into a DataFrame
import pandas as pd

pricing_with_and_without_cache_single_input_df = pd.DataFrame([pricing_without_cache,
                                                               pricing_with_cache_2_hours,
                                                               pricing_with_cache_15_minutes])

pricing_with_and_without_cache_single_input_df.head()
```

```python
import matplotlib.pyplot as plt

def plot_pricing_df(df=pricing_with_and_without_cache_single_input_df):
  x_labels = df.index
  width = 0.6

  x_labels_custom = [
      f"Using cache: {row['using_cache']} | Cache time: {row['cache_storage_time']}"
      for _, row in df.iterrows()
  ]

  fig, ax = plt.subplots(figsize=(10, 7))
  ax.bar(x_labels, df['input_price'], width, label='Input Price')
  ax.bar(x_labels, df['output_price'], width, bottom=df['input_price'], label='Output Price')
  ax.bar(x_labels, df['cached_tokens_price'], width,
        bottom=df['input_price'] + df['output_price'], label='Cached Tokens Price')
  ax.bar(x_labels, df['cached_storage_price'], width,
        bottom=df['input_price'] + df['output_price'] + df['cached_tokens_price'], label='Cached Storage Price')

  # Labels and Title
  ax.set_xlabel('Cache Usage and Storage Time (hours)')
  ax.set_ylabel('Price ($USD)')
  ax.set_title('Gemini Flash Price Comparison for Cached and Non-Cached Inputs (Single Pass on Large Video File)')
  ax.legend()

  plt.xticks(x_labels, x_labels_custom, rotation=45, ha='right')
  plt.tight_layout()
  plt.show();

plot_pricing_df()
```

It seems our cached reponses cost more than our non-cached responses (especially so for when we store the cache for 2 hours).

Though we will see the benefits of caching after several passes of the long context input.

> **Note:** Cached input tokens are priced at 4x less than regular input tokens, however, caching storage costs can negate this benefit. An ideal workflow will only create cached content for as long as needed to perform a task.
>
> For example, if it takes a workflow 5 minutes to perform inference over a cached video input 3 times, the ideal caching time for the video would be slightly higher than 5 minutes (e.g. 10 minutes for redundancy) rather than 2 hours.
>
> In my experience, caching becomes most beneficial when you plan on performing inference *more than* two times on a single input. In production, let's say you had a high traffic app and your inputs have several examples that might be used for *every* API call, caching these inputs at all times might be very beneficial.
>
> Otherwise, if you only want to perform inference once, regular input tokens may be best.
>
> As always, best to experiment and track costs to make sure you're getting the best value for money.

### Prompt Helper 1: Create a helper function to fix CSVs

Our model's outputs aren't always perfect.

And since we're working with CSVs, there are a few rules for crafting CSVs to be handled by various APIs, otherwise they'll break.

Writing a set of rules for fixing a broken CSV can be quite challenging.

So we'll use Gemini to fix our CSV outputs if our quick check functions detect any breakages.

We can start by making an instance of a Gemini model to focus on fixing CSVs.


> **Note:** When performing different tasks, I've found it helpful to create one model per task. For example, if you want a Gemini model to focus on fixing CSVs given a broken CSV, create a model for that. In my experience, one model, one task seems to work best rather than trying to get a single model to perform every task.

```python
# Create a model focused on fixing CSVs
model_gemini_flash_csv_fixer = genai.GenerativeModel(model_name="models/gemini-1.5-flash-002",
                                                     generation_config={'temperature': 0.3, # a little less randomness than above
                                                                        'top_p': 0.2,
                                                                        'top_k': 3,
                                                                        'max_output_tokens': 8192,
                                                                        'response_mime_type': 'text/plain'},
                                                     system_instruction="You are an expert at spotting errors in CSV files and can quickly fix them. You always return valid CSV information when asked. Your speciality is looking at existing CSV, spotting the rows that need fixing, fixing those rows and then returning the updated existing CSV.")
print(model_gemini_flash_csv_fixer)
```

Model ready!

Now let's define a function and a prompt to take in a CSV and fix it based on a list of fixes required (these are output from our `quick_check_csv` function).

> **Note:** The prompt below for CSV fixing has been designed experimentally overtime. As always, I started simple with "the following CSV has errors, please fix them" and then evolved it to what you see below.

```python
def fix_csv(input_csv,
            fixes_required,
            model=model_gemini_flash_csv_fixer):
  """Takes in a broken CSV and uses Gemini to fix it."""
  fix_csv_prompt = """You are an AI assistant specialized in fixing broken CSV (Comma-Separated Values) files. Your task is to take a potentially broken CSV input, identify and correct any formatting issues, and return a properly formatted CSV.

  See the examples:

  <examples>
  <example>
  <input_csv>
  item_number,item_name,item_type,item_description,item_brand,item_condition,number_of_items,estimated_worth,mentioned_worth,room,timestamp,item_summary,flag
  1,Couch,Furniture,White,NA,no visible damage,1,1000,NA,Lounge Room,00:00:10,White couch in the lounge room.,False
  2,Coffee Table,Furniture,Glass top, wooden legs,NA,no visible damage,1,300,NA,Lounge Room,00:00:12,Glass top, wooden leg coffee table in the lounge room.,False
  3,Dining Table,Furniture,"Large, oak",IKEA,minor scratches,1,500,NA,Dining Room,00:00:15,"Large, oak dining table in the dining room.",False
  4,Bookshelf,Furniture,Wooden,Target,good condition,2,150,NA,Study,00:00:18,Two wooden bookshelves in the study.,False
  5,TV Stand,Furniture,"Black, glass shelves",Best Buy,excellent condition,1,200,NA,Living Room,00:00:22,"Black TV stand with glass shelves in the living room.",False
  </input_csv>
  <ideal_output>
  <csv_analysis>
  1. The first few lines of the CSV:
  item_number,item_name,item_type,item_description,item_brand,item_condition,number_of_items,estimated_worth,mentioned_worth,room,timestamp,item_summary,flag
  1,Couch,Furniture,White,NA,no visible damage,1,1000,NA,Lounge Room,00:00:10,White couch in the lounge room.,False
  2,Coffee Table,Furniture,Glass top, wooden legs,NA,no visible damage,1,300,NA,Lounge Room,00:00:12,Glass top, wooden leg coffee table in the lounge room.,False

  2. The header has 13 columns.

  3. Checking subsequent rows for the correct number of columns:
  - Row 2 has 13 columns, but the item_description field contains a comma that is not properly enclosed in quotes: "Glass top, wooden legs"
  - Row 3 has 13 columns, but the item_description field contains a comma that is properly enclosed in quotes: "Large, oak"
  - Rows 4 and 5 have the correct number of columns and no issues.

  4. Fields with special characters that need proper enclosure:
  - Row 2: item_description field contains a comma
  - Row 3: item_description field already properly enclosed
  - Row 5: item_description field contains a comma

  5. Specific fixes needed:
  - Enclose the item_description field in row 2 with double quotes
  - Enclose the item_description field in row 5 with double quotes
  - No other fixes are required as the CSV is otherwise well-formatted
  </csv_analysis>

  <csv>
  item_number,item_name,item_type,item_description,item_brand,item_condition,number_of_items,estimated_worth,mentioned_worth,room,timestamp,item_summary,flag
  1,Couch,Furniture,White,NA,no visible damage,1,1000,NA,Lounge Room,00:00:10,White couch in the lounge room.,False
  2,Coffee Table,Furniture,"Glass top, wooden legs",NA,no visible damage,1,300,NA,Lounge Room,00:00:12,"Glass top, wooden leg coffee table in the lounge room.",False
  3,Dining Table,Furniture,"Large, oak",IKEA,minor scratches,1,500,NA,Dining Room,00:00:15,"Large, oak dining table in the dining room.",False
  4,Bookshelf,Furniture,Wooden,Target,good condition,2,150,NA,Study,00:00:18,Two wooden bookshelves in the study.,False
  5,TV Stand,Furniture,"Black, glass shelves",Best Buy,excellent condition,1,200,NA,Living Room,00:00:22,"Black TV stand with glass shelves in the living room.",False
  </csv>
  </ideal_output>
  </example>
  <example>
  <input_csv>
  Name,Age,City,Favorite Quote
  John Doe,30,New York,"Life is what happens when you're busy making other plans"
  Jane Smith,25,Los Angeles,Stay hungry, stay foolish
  Mike Johnson,35,Chicago,"To be or not to be, that is the question"
  Sarah Brown,28,San Francisco,I think, therefore I am
  David Wilson,40,Boston,"Be the change you wish to see in the world
  </input_csv>
  <ideal_output>
  <csv_analysis>
  1. The first few lines of the CSV:
  Name,Age,City,Favorite Quote
  John Doe,30,New York,"Life is what happens when you're busy making other plans"
  Jane Smith,25,Los Angeles,Stay hungry, stay foolish

  2. The header has 4 columns.

  3. Checking subsequent rows for the correct number of columns:
  - Row 1 has 4 columns and is properly formatted
  - Row 2 has 4 columns, but the Favorite Quote field is not enclosed in quotes and contains a comma
  - Row 3 has 4 columns and is properly formatted
  - Row 4 has 4 columns and is properly formatted
  - Row 5 has 3 columns instead of 4, and the Favorite Quote field is missing a closing quote

  4. Fields with special characters that need proper enclosure:
  - Row 2: Favorite Quote field contains a comma and needs to be enclosed in quotes
  - Row 5: Favorite Quote field contains a newline character and is missing a closing quote

  5. Specific fixes needed:
  - Enclose the Favorite Quote field in row 2 with double quotes
  - Add a closing double quote to the Favorite Quote field in row 5
  - Ensure the newline in row 5's Favorite Quote field is preserved within the quotes
  </csv_analysis>

  <csv>
  Name,Age,City,Favorite Quote
  John Doe,30,New York,"Life is what happens when you're busy making other plans"
  Jane Smith,25,Los Angeles,"Stay hungry, stay foolish"
  Mike Johnson,35,Chicago,"To be or not to be, that is the question"
  Sarah Brown,28,San Francisco,"I think, therefore I am"
  David Wilson,40,Boston,"Be the change you wish to see in the world"
  </csv>
  </ideal_output>
  </example>
  <example>
  <input_csv>
  Product,Price,Quantity,Description
  "Wireless Headphones",99.99,50,"Bluetooth, noise-cancelling"
  "Smart Watch",199.99,30,Fitness tracker, heart rate monitor
  "Laptop",899.99,20,"15.6" display, 512GB SSD, 16GB RAM"
  "Smartphone",699.99,40,6.2" OLED display, dual camera
  "Tablet",349.99,25,10.1" display, 64GB storage
  </input_csv>
  <ideal_output>
  <csv_analysis>
  1. The first few lines of the CSV:
  Product,Price,Quantity,Description
  "Wireless Headphones",99.99,50,"Bluetooth, noise-cancelling"
  "Smart Watch",199.99,30,Fitness tracker, heart rate monitor

  2. The header has 4 columns.

  3. Checking subsequent rows for the correct number of columns:
  - Row 1 has 4 columns and is properly formatted
  - Row 2 has 4 columns, but the Description field contains a comma and is not enclosed in quotes
  - Row 3 has 4 columns, but the Description field contains quotes that are not properly escaped
  - Row 4 has 4 columns, but the Description field contains a quote that is not properly escaped
  - Row 5 has 4 columns and is properly formatted

  4. Fields with special characters that need proper enclosure:
  - Row 2: Description field contains a comma and needs to be enclosed in quotes
  - Row 3: Description field contains quotes that need to be escaped
  - Row 4: Description field contains a quote that needs to be escaped

  5. Specific fixes needed:
  - Enclose the Description field in row 2 with double quotes
  - Escape the quotes in the Description field of row 3 by doubling them
  - Escape the quote in the Description field of row 4 by doubling it
  </csv_analysis>

  <csv>
  Product,Price,Quantity,Description
  "Wireless Headphones",99.99,50,"Bluetooth, noise-cancelling"
  "Smart Watch",199.99,30,"Fitness tracker, heart rate monitor"
  "Laptop",899.99,20,"15.6"" display, 512GB SSD, 16GB RAM"
  "Smartphone",699.99,40,"6.2"" OLED display, dual camera"
  "Tablet",349.99,25,"10.1"" display, 64GB storage"
  </csv>
  </ideal_output>
  </example>

  <quick_examples>
  Broken: Weight plates (20kg, 10kg, 5kg) -> Fixed: "Weight plates (20kg, 10kg, 5kg)"
  Broken: Cleaning Supplies,"Bleach, disinfectant wipes, all-purpose cleaner" -> Fixed: Cleaning Supplies,"Bleach, disinfectant wipes, all-purpose cleaner"
  Broken: Pantry,Flour, sugar, spices -> Fixed: Pantry,"Flour, sugar, spices"
  Broken: Assorted gardening tools including rakes, shovels, and hoes -> Fixed: "Assorted gardening tools including rakes, shovels, and hoes"
  </quick_examples>
  </examples>

  Please follow these steps to fix the CSV:

  1. Analyze the input CSV:
    - Check if fields are properly separated by commas.
    - Identify any fields containing commas, double quotes, or newlines that are not properly enclosed in double quotes.
    - Verify if all rows have the same number of columns as the header.

  2. Plan the necessary fixes:
    - Determine how to properly enclose fields containing special characters.
    - Identify any misaligned columns that need to be corrected.
    - Plan how to escape any double quotes within fields by doubling them.

  3. Implement the fixes:
    - Enclose any field containing commas, double quotes, or newlines in double quotes.
    - Escape double quotes within fields by doubling them.
    - Realign any rows with missing or extra columns based on the header structure.

  4. Validate the fixed CSV:
    - Ensure all rows have the correct number of columns.
    - Verify that all special characters are properly escaped.
    - Check that the CSV is now valid and can be read by standard CSV parsers.

  Before providing the fixed CSV, use <csv_analysis> tags to analyze the input, identify issues, and plan your fixes. In your analysis:

  1. Quote the first few lines of the CSV.
  2. Count the number of columns in the header.
  3. Check each subsequent row for the correct number of columns, quoting any problematic rows.
  4. Identify any fields with special characters (commas, quotes, newlines) that need proper enclosure.
  5. Plan out the specific fixes needed for each issue identified.

  Be thorough in your analysis to ensure all potential problems are addressed.

  After your analysis, provide the fixed CSV enclosed in <csv> tags. Ensure that the output is clean, consistent, and free of errors.

  Remember:
  - Do not alter the actual data within the fields, only fix formatting issues.
  - Maintain the original structure (number and order of columns) of the CSV.
  - Only make changes necessary to fix formatting problems.

  Here is the input CSV that needs to be fixed:

  <input_csv>
  {input_csv}
  </input_csv>

  And the fixes it requires:
  <fixes_required>
  {fixes_required}
  </fixes_required>

  Remember to enclose your fixed CSV in <csv> tags for easy parsing.

  Write a note to yourself to begin the fixing as a reminder: "I've got this. I will fix the in <input_csv> with the required fixes and will output the updated and fixed CSV in <csv> tags".
  """

  model_input_prompt = fix_csv_prompt.format(input_csv=input_csv,
                                             fixes_required=fixes_required)

  return generate_content(prompt=model_input_prompt,
                          model=model)
```

### Format Check 1: Check formatting of output CSV from Prompt 1

We can extract the CSV from our `model_response_1_with_cache` and check its formatting with the function `quick_check_csv`.

`quick_check_csv` will:

1. Extract the target CSV using the `<csv></csv>` tags.
2. See if a CSV formatting fix is required.
3. List the fixes required to the CSV.

If our CSV requires fixing, we can pass it to our `fix_csv` function to have Gemini repair it.

```python
model_response_1_csv, fix_csv_1_required, csv_fixes_to_do_1 = quick_check_csv(model_output=model_response_1_with_cache,
                                                                              target_start_tag="<csv>",
                                                                              target_end_tag="</csv>")
print(f"[INFO] CSV fix required? {fix_csv_1_required}")
print(f"[INFO] Fixes to do:\n{csv_fixes_to_do_1}")

while fix_csv_1_required:
  print(f"[INFO] CSV fix required... fixing...")
  model_response_1_csv = fix_csv(input_csv=model_response_1_csv, fixes_required=csv_fixes_to_do_1)
  model_response_1_csv, fix_csv_1_required, fixes_to_do_1 = quick_check_csv(model_output=model_response_1_csv)
  print(f"[INFO] CSV fix required? {fix_csv_1_required}")
  print(f"[INFO] Fixes to do:\n{csv_fixes_to_do_1}")
```

How does our extract CSV look?

```python
print(model_response_1_csv[:1000])
```

Not bad for a first pass!

How about we turn it into a DataFrame?

```python
import io
import pandas as pd

df_1 = pd.read_csv(io.StringIO(model_response_1_csv), on_bad_lines="warn")
df_1.head(10)
```

```python
df_1.tail(10)
```

Woah! That looks fantastic!

So far so good.

## Prompt 2: Check to see if the first output reaches the video length, if not keep going

Since Gemini models have an output limit of ~8k tokens, I'ved found that for longer outputs (e.g. more items in the video), the target CSV gets cut off early.

To remedy this, we can look at a few things:

1. Check the last timestamp in the DataFrame to see if it matches the end of the video (this is a simple check find out whether or not the items go right up to the last timestamp).
2. Check the model's output for the special `MORETIMESTAMPSTODO` flag (we asked the model to output this if it thought more items should be done).
3. Simply ask the model to keep going from the last timestmap.

We're going to do a combination of 1 and 2.

Let's first write a helper function to convert a timestamp to seconds (we do this because we know how long our video length is in seconds thanks to `cv2`).

> **Note:** In my experience throughout this project I've found that the model is quite capable of extracting 50+ items from a ~10-minute video. However, further testing would be required for longer videos with more densely packed items. I've also found that sometimes the items towards the end of the tracking are less accurate then the start.
>
> One potential way to migitate errors in longer videos would be to split them into mulitple videos and investigate them inividually.
>
> Another way to help with items towards the end of a video would be to perform inference on the video in reverse order (e.g. one forward and another backwards pass), though I'm yet to try this.

```python
import re

def timestamp_to_seconds(timestamp):
    """
    Convert a timestamp string to seconds. Supports formats:
    - HH:MM:SS
    - MM:SS
    - H:MM:SS
    - M:SS

    Args:
        timestamp (str): A timestamp string in one of the supported formats

    Returns:
        int: Total number of seconds, or None if the timestamp is invalid

    Examples:
        >>> timestamp_to_seconds("01:30:15")  # 1 hour, 30 mins, 15 secs
        5415
        >>> timestamp_to_seconds("5:30")      # 5 mins, 30 secs
        330
        >>> timestamp_to_seconds("1:02:00")   # 1 hour, 2 mins
        3720
        >>> timestamp_to_seconds("invalid")
        None
    """
    try:
        # Remove any leading/trailing whitespace
        timestamp = timestamp.strip()

        # Match timestamp pattern
        pattern = r'^(?:(?:(\d+):)?(\d{1,2}):)?(\d{1,2})$'
        match = re.match(pattern, timestamp)

        if not match:
            return None

        # Extract hours, minutes, seconds (all optional except seconds)
        hours = int(match.group(1) or 0)
        minutes = int(match.group(2) or 0)
        seconds = int(match.group(3))

        # Validate ranges
        if minutes >= 60 or seconds >= 60:
            return None

        # Convert to total seconds
        return hours * 3600 + minutes * 60 + seconds

    except (ValueError, AttributeError):
        return None

test_cases = [
    "1:30:15",    # 1 hour, 30 mins, 15 secs
    "30:15",      # 30 mins, 15 secs
    "5:15",       # 5 mins, 15 secs
    "1:00:00",    # 1 hour
    "90:00",      # 90 mins (invalid)
    "invalid",    # invalid format
]

for test in test_cases:
    print(f"{test} -> {timestamp_to_seconds(test)}")
```

Now we've got a helper function to convert a timestamp in `MM:SS` format to seconds, we can use it along with a string check for `"MORETIMESTAMPSTODO"` in the model's output.

If the last item's timestamp doesn't match the length of the video or `"MORETIMESTAMPSTODO"` is in the output, we'll ask the model to continue where it left off.

```python
# Get the last timestamp of the last detected item
last_timestamp = df_1.iloc[-1].timestamp

# Conver the last timestamp to seconds
last_timestamp_seconds = timestamp_to_seconds(timestamp=last_timestamp)

print(f"[INFO] Video last timestamp: {video_length_seconds}")
print(f"[INFO] Last timestamp in seconds: {last_timestamp_seconds}")
print(f"[INFO] Is MORETIMESTAMPSTODO flag in model response? {'MORETIMESTAMPSTODO' in model_response_1_with_cache.text}")

if (video_length_seconds > last_timestamp_seconds) or ("MORETIMESTAMPSTODO" in model_response_1_with_cache.text):
  print("The prompt must go on!")
else:
  print("All timestamps in the video have been covered!")
```

Looks like the prompt must go on!

We'll write some instructions in `input_prompt_secondary` to tell the model to take in an existing CSV (`model_response_1_csv`) as well as the last timestamp of the last item (`last_timestamp`) to tell it to keep going.

`input_prompt_secondary` will be prefixed with our cached video file thanks to using `model_gemini_flash_with_cache`.

In essence, our prompt will say:
1. Here's a video file (from the cache).
2. And here's an existing CSV of all the items as well as the timestamp that we're up to.
3. Please continue the CSV with any missing items from the video.

You can read the full prompt below for more.

```python
%%time
input_prompt_secondary = """Please continue the following CSV based on the input video. The goal is to track all household inventory visible/spoken about.

Some items have been tracked already, do not track these, instead continue the logging from mentioned timestamp.

To begin, write some notes in <video_analysis> tags on what you will have to do to complete the given task from the given timestamp. For example, write information such as:
* Rooms which have been completed (no need to continue these): [...]
* Rooms which are required to do: [...]
* Objective for completing the rooms which are required to do: [...]
* Unique items to log (each on an individual line): [...] (e.g. "I will create one line in the CSV for each of the following unique items: [list_of_unique_items]")
* Notes to self on how you will return the new items in valid CSV format (e.g. "I will ensure to return the updated items in valid CSV format." and "I will only return new items in valid CSV format rather than repeating existing items." and "The number of fields I must return for each line item in the CSV is: [count the number of fields in the CSV header]"

Overall instructions:
- Return only the new items in valid CSV format inside <csv> tags so they can easily be parsed out.
- Do not return existing items from the input CSV.
- List all items visible/mentioned in the video but not included in the original CSV.

CSV Formatting Instructions:
- Always output the original CSV header with the new CSV.
- Use commas only to separate fields rather than inside fileds.
- Do not use commas in the item_description field.
- For fields that inherently include commas, enclose the entire value in double quotes (`"`).
- If double quotes appear within a field value, escape them by doubling them (`""`). For example, `John "JJ" Smith` should be written as `"John ""JJ"" Smith"`.
- Do not use any unescaped double quotes or other special characters that may cause the CSV to be invalid.
- Never use commas in prices. For example, $1500.0 = good, $1,500.0 = bad.
- Return all columns in the order they are presented in. Do not reorder the columns. Write these down in <column_heading_order> tags so you don't forget.
- Ensure every column has a value, do not miss a column.

For example:

<examples>
<example_1>
<video_analysis>
[Step-by-step breakdown of what's required to do to complete the task as best as possible. Including information about what to ignore and what to pay attention to.]
* Rooms which have been completed (no need to continue these): [...]
* Rooms which are required to do: [...]
* Objective for completing the rooms which are required: [...]
* Notes to self: [...]

I've got this.
</video_analysis>

<list_of_additional_unique_items_to_log>
[One by one list of additional unique items to log in the new csv]
</list_of_additional_unique_items_to_log>

Outputing only these new items now.

I will adhere to matching the number of fields to the following CSV header, being sure to only output one entry per line per field.

<csv>
item_number,item_name,item_type,item_description,item_brand,item_condition,number_of_items,estimated_worth,estimated_worth_flag,mentioned_worth,room,timestamp,overall_certainty_flag,is_similar_to
[New items not mentioned in original input CSV]
</csv>
</example_1>

<example_2>
<video_analysis>
Let's breakdown the video analysis step by step.

I've watched the video and read the input CSV.

Time to take note of what needs to be done.

Rooms which have been completed (no need to continue these): [...]
Rooms which are required to do: [...]

Objective for completing the rooms which are required to do: To log all visible/mentioned household inventory in the remaining rooms that were not covered in the original CSV.

Notes to self:
* I'm going to make sure to cover the rooms which have not been done to make the household inventory analysis complete.
* I will ensure to return the new items in valid CSV format, continuing the item numbering from the last item in the original CSV.
* The number of fields I must return for each line item in the CSV is: [count the number of fields in the CSV header]

I've got this.
</video_analysis>

<list_of_additional_unique_items_to_log>
Additional items I've discovered that we'ren't in the original CSV but are in the video to log:
- Laptop
- Smartphone
- Television
- Washing Machine
- Dining Table
- Vacuum Cleaner
- Microwave
- Bed
- Bookshelf
- Bicycle
</list_of_additional_unique_items_to_log>

Outputing only these new items now.

I will adhere to matching the number of fields to the following CSV header, being sure to only output one entry per line per field.

<csv>
item_number,item_name,item_type,item_description,item_brand,item_condition,number_of_items,estimated_worth,estimated_worth_flag,mentioned_worth,room,timestamp,overall_certainty_flag,is_similar_to
65,Laptop,electronics,15-inch MacBook Pro with Retina display,Apple,minor scratches,1,2000.0,7,2000.0,Home Office,08:00,8,NA
66,Smartphone,electronics,iPhone 14 Pro with 128GB storage,Apple,no visible damage,1,1000.0,8,NA,Living Room,08:05,8,NA
67,Television,electronics,65-inch 4K OLED Smart TV,Sony,no visible damage,1,1500.0,7,1500.0,Living Room,08:10,7,NA
68,Washing Machine,appliances,Front-loading washing machine with 8kg capacity,LG,no visible damage,1,700.0,7,NA,Laundry,08:15,7,NA
69,Dining Table,furniture,Large oak dining table with seating for six,NA,no visible damage,1,1200.0,8,1200.0,Dining Room,08:20,8,NA
70,Vacuum Cleaner,appliances,High-powered cordless vacuum cleaner,Dyson,minor wear,1,500.0,7,NA,Storage Room,08:25,7,NA
71,Microwave,appliances,Stainless steel countertop microwave oven,Panasonic,no visible damage,1,200.0,8,NA,Kitchen,08:30,8,NA
72,Bed,furniture,Queen-size bed frame with memory foam mattress,NA,no visible damage,1,1500.0,7,1500.0,Bedroom,08:35,7,NA
73,Bookshelf,furniture,Wooden bookshelf with five shelves,Ikea,no visible damage,1,250.0,7,NA,Living Room,08:40,7,NA
74,Bicycle,sports equipment,21-speed mountain bike with disc brakes,Trek,minor scratches,1,900.0,7,NA,Garage,08:45,7,NA
</csv>
</example_2>
</examples>

These examples are only demonstrations. Only include new items in the output CSV that are included in the attached video.

Timestamp to continue from: {last_timestamp}

The fields to include in the appended CSV are:
{csv_input_schema}

The current CSV is as follows:
{input_csv}

Your task is to extend this CSV with missing items from the video.

Starting video analysis now:
""".format(csv_input_schema=get_schema_string(),
           input_csv=model_response_1_csv,
           last_timestamp=last_timestamp)

print(f"[INFO] Seems like some items are missing... going for a second pass on the video to collect the rest of the items...")
model_response_2_with_cache = generate_content(prompt=[input_prompt_secondary],
                                               model=model_gemini_flash_with_cache)

print(f"[INFO] Secondary prompt:\n{input_prompt_secondary}")
print(f"[INFO] Secondary output text:\n")
print(model_response_2_with_cache.text)
```

Wonderful!

Looks like it worked, how about we check the CSV formatting of our second output?

### Format Check 2: Check formatting of output CSV from Prompt 2

Let's make sure our second output CSV is formatted correctly and if it isn't we can correct it.

```python
model_response_2_csv, fix_csv_2_required, csv_fixes_to_do_2 = quick_check_csv(model_output=model_response_2_with_cache,
                                                                              target_start_tag="<csv>",
                                                                              target_end_tag="</csv>")
print(f"[INFO] CSV fix required? {fix_csv_2_required}")
print(f"[INFO] Fixes to do:\n{csv_fixes_to_do_2}")

while fix_csv_2_required:
  print(f"[INFO] CSV fix required... fixing...")
  model_response_2_csv = fix_csv(input_csv=model_response_2_csv, fixes_required=csv_fixes_to_do_2)
  model_response_2_csv, fix_csv_2_required, csv_fixes_to_do_2 = quick_check_csv(model_output=model_response_2_csv,
                                                                                target_start_tag="<csv>",
                                                                                target_end_tag="</csv>")
  print(f"[INFO] CSV fix required? {fix_csv_2_required}")
```

### Combine our multiple CSVs

We've now got two CSVs.

1. `model_response_1_csv` - an initial pass over the video with items extracted.
2. `model_response_2_csv` - a secondary pass over the video with the goal of extended the first pass.

Let's now combine them into one, dropping duplicates if there are any.

```python
# Create DataFrames from both CSV files
df_1 = pd.read_csv(io.StringIO(model_response_1_csv), on_bad_lines="warn")
df_2 = pd.read_csv(io.StringIO(model_response_2_csv), on_bad_lines="warn")

# Combine the two DataFrames into one
df_combined = pd.concat([df_1, df_2]).reset_index(drop=True).drop_duplicates()

print(f"[INFO] Number of items in complete df: {len(df_combined)}")

df_combined.head(10)
```

```python
df_combined.tail(10)
```

Now that's a lot of items!

How about we run a third check on the combined CSVs to see if there's anything that needs fixing or any additions that need to happen?

## Prompt 3: Final Check Prompt

So far we've gone over our input video twice, once in an initial pass and once trying to expand it where necessary.

Now how about we go over our video a final time with the goal of making sure everything in the output CSV is correct?

We can design a prompt to take in our current CSV and to watch the video again and then make sure everything in the output CSV is correct and aligned to the video.

```python
# Get the current CSV from our combined DataFrame
final_check_csv_string = df_to_csv_string(df=df_combined)
final_check_csv_string
```

### Writing examples for Prompt 3

Since our final task is slightly different to previous prompts, we can write some more examples to give to the model to showcase taking in an existing CSV and fixing/adding values.

> **Note:** As with all prompts and examples, they begin as simple instructions, for example "compare this CSV to the video and improve/fix/add any necessary information" and then they grew over time to more specific instructions and demonstrations.

```python
example_1_final_check = """<initial_analysis> Upon reviewing the video input and comparing it to the current CSV log, the following observations were made:

Rooms Fully Reviewed:

Living Room
Dining Room
Kitchen
Storage Closet
Second Living Area
Bedroom 1, Bedroom 2, Bedroom 3
Master Bedroom
Garage
Outdoor Area
Toolshed
Rooms Requiring Review:

Gym: This room was mentioned in the video but is missing from the current CSV. It contains several gym machines that are not currently logged.
Objective: The primary objective is to ensure that all household items visible or mentioned in the video, especially those in the Gym, are accurately logged in the CSV. This includes identifying new items, verifying existing entries, and addressing any discrepancies.

Key Items Observed in the Video by Room:

Gym:
Treadmill
Elliptical Machine
Exercise Bike
Weight Rack
Yoga Mats
Pull-Up Bar
Resistance Bands
Comparison with Current CSV:

Gym Items Missing:
Treadmill
Elliptical Machine
Exercise Bike
Weight Rack
Yoga Mats
Pull-Up Bar
Resistance Bands
Potential Reasons for Discrepancies:

Gym Room Missing: The Gym room was not included in the initial CSV, leading to all its items being untracked.
Overlooked Items: Items within the Toolshed were previously missing but have since been added, suggesting that similar oversights could have occurred for the Gym.
Misclassification: Some gym items might have been misclassified under different room categories or item types.
</initial_analysis>

<video_final_status> MOREITEMSTODO </video_final_status>

<thinking> Since there are MOREITEMSTODO, I will fix/finish them off. </thinking>

<steps_to_fix_csv>
1. **Identify Missing Rooms and Items:**
   - Confirm that the Gym room is not present in the current CSV.
   - List all items observed in the Gym during the video that are not in the CSV.

2. **Create New Entries:**
   - For each gym item observed, create a new CSV entry following the provided schema.
   - Assign sequential `item_number` starting from 71.
   - Ensure all fields adhere to the CSV schema, including proper capitalization and formatting.

3. **Verify Existing Entries:**
   - Double-check existing CSV entries for accuracy, ensuring no duplication or misclassification of items.
   - Confirm that items in the Toolshed have been correctly added.

4. **Finalize the Updated CSV:**
   - Combine the existing CSV with the new Gym items.
   - Ensure the CSV maintains proper formatting and structure.

5. **Update `video_final_status`:**
   - Since new items have been added, mark the status as "MOREITEMSTODO" and include the new CSV entries.
</steps_to_fix_csv>

<updated_csv>
item_number,item_name,item_type,item_description,item_brand,item_condition,number_of_items,estimated_worth,estimated_worth_flag,mentioned_worth,room,timestamp,overall_certainty_flag,is_similar_to
1,Couch,furniture,Beige fabric couch,NA,no visible damage,1,1000.0,7,NA,Living Room,00:09,7,NA
2,Rug,furniture,Beige woven rug,NA,no visible damage,1,200.0,6,NA,Living Room,00:12,6,NA
3,Coffee Table,furniture,Glass top coffee table with wooden legs,NA,no visible damage,1,300.0,7,NA,Living Room,00:13,7,NA
4,Armchair,furniture,Green fabric armchair,NA,no visible damage,2,400.0,7,NA,Living Room,00:14,7,15
5,TV Cabinet,furniture,Light wood TV cabinet,NA,no visible damage,1,500.0,7,NA,Living Room,00:17,7,NA
6,Painting,decor,Abstract painting in earth tones,NA,no visible damage,1,200.0,6,NA,Living Room,00:21,6,NA
7,Floor Lamp,lighting,Black metal floor lamp,NA,no visible damage,1,150.0,7,NA,Living Room,00:23,7,NA
8,Vacuum Cleaner,appliances,Upright vacuum cleaner,NA,no visible damage,1,200.0,6,NA,Storage Closet,00:34,6,NA
9,Sonos Speakers,electronics,Sonos One speakers,Sonos,as new,2,250.0,8,NA,Storage Closet,00:40,8,NA
10,Books,decor,Stack of books titled Charlie Walks by Daniel Burck,NA,no visible damage,25,20.0,7,NA,Storage Closet,00:42,8,NA
11,Hard Drives,electronics,External hard drives,Samsung,no visible damage,2,100.0,7,NA,Storage Closet,00:46,7,NA
12,Loveseat,furniture,Beige fabric loveseat,NA,no visible damage,1,600.0,7,NA,Second Living Area,00:54,7,NA
13,Stools,furniture,Wooden and metal bar stools,NA,no visible damage,3,100.0,7,NA,Kitchen,01:04,7,NA
14,Dining Table,furniture,Light wood dining table,NA,no visible damage,1,400.0,7,NA,Dining Room,01:07,7,NA
15,Dining Chairs,furniture,Grey upholstered dining chairs,NA,no visible damage,6,150.0,7,NA,Dining Room,01:08,7,NA
16,Potted Plant,decor,Potted plant in white pot,NA,no visible damage,1,50.0,6,NA,Dining Room,01:11,6,NA
17,Mirror,decor,Large rectangular mirror,NA,no visible damage,1,150.0,7,NA,Dining Room,01:13,7,NA
18,Refrigerator,appliances,White French door refrigerator,Panasonic,no visible damage,1,1500.0,8,NA,Kitchen,01:18,8,NA
19,Dishwasher,appliances,Built in stainless steel dishwasher,Miele,no visible damage,1,800.0,8,NA,Kitchen,01:22,8,NA
20,Microwave,appliances,Black microwave,LG,no visible damage,1,200.0,8,NA,Kitchen,01:26,8,NA
21,Kettle,kitchenware,Stainless steel kettle,NA,no visible damage,1,50.0,6,NA,Kitchen,01:27,6,NA
22,Coffee Machine,kitchenware,Black coffee machine,NA,no visible damage,1,150.0,6,NA,Kitchen,01:28,6,NA
23,Double Oven,appliances,Stainless steel double oven,NA,no visible damage,1,1000.0,7,NA,Kitchen,01:31,7,NA
24,Water Filter,appliances,Stainless steel water filter,NA,no visible damage,1,200.0,6,NA,Laundry Closet,01:39,6,NA
25,Bread Maker,appliances,White bread maker,NA,no visible damage,1,100.0,6,NA,Laundry Closet,01:40,6,NA
26,Washing Machine,appliances,White washing machine,Haier,no visible damage,1,800.0,8,NA,Laundry Closet,01:43,8,NA
27,Queen Bed,furniture,White queen bed,NA,no visible damage,1,800.0,7,NA,Bedroom 1,02:24,7,NA
28,Nightstand,furniture,Light wood nightstand,NA,no visible damage,2,150.0,7,NA,Bedroom 1,02:26,7,NA
29,Lamp,lighting,White table lamp,NA,no visible damage,2,75.0,7,NA,Bedroom 1,02:28,7,NA
30,Painting,decor,Abstract painting,NA,no visible damage,1,200.0,6,NA,Bedroom 1,02:32,6,NA
...continued to finish the whole CSV...
</updated_csv>

<post_fix_summary>
* Lines added:
  - 71 to 77: Gym room items including Treadmill, Elliptical Machine, Exercise Bike, Weight Rack, Yoga Mats, Pull-Up Bar, and Resistance Bands.
* Lines fixed/amended:
  - No existing lines were amended. All existing lines remain unchanged.
</post_fix_summary>

<next_steps>
1. **Verify Gym Room Entries:**
   - Conduct a detailed review of the Gym room in the video to ensure all items are accurately captured and no additional items are missing.

2. **Cross-Check All Rooms:**
   - Re-examine other rooms to confirm that all items are logged and there are no further omissions or misclassifications.

3. **Update Inventory Values:**
   - For any newly added items, consider validating the `estimated_worth` and `estimated_worth_flag` with market values or expert appraisal to ensure accuracy.
</next_steps>
"""
```

### Prompting the model to review and finalize our CSV

Now we've got an example (we could make more if we wanted, this may lead to better results), let's write a set of instructions to tell the model to:

1. Take in an existing CSV and video (the video input will be cached due to us using `model_gemini_flash_with_cache`).
2. Compare the existing CSV to the video and make sure all of the information is correct.
3. Improve the CSV if necessary.

```python
input_prompt_final_check = """
You are an expert video reviewer with a keen eye for detail, tasked with verifying and updating a CSV log of household items based on a video input. Your goal is to ensure that every household item visible or mentioned in the video is accurately logged in the CSV.

Your instructions are:

1. Review the video input and compare it to the current CSV input.
2. Write an initial analysis of your observations. Include the following:
   - Rooms that have been fully reviewed (no further action needed)
   - Rooms that still require review (if applicable)
   - Your objective for reviewing and updating the CSV
   - Key items observed in the video, categorized by room
   - Comparison of observed items with the current CSV, noting discrepancies
   - Potential reasons for discrepancies (e.g., items moved, overlooked, or misclassified)
   - Make sure to include every major item that is visible and mentioned, including items that are mentioned but not necessarily visible (e.g. items in boxes/bags/closets).
3. Determine if all items in the video have been tracked in the CSV.
4. Plan the necessary removals, fixes or additions to the CSV, your goal here should be to enrich existing information if possible and remove redundant information/rows. If an item's details/description suite it well, leave it. If it could be better, rewrite it.
5. Implement the fixes, additions/removals to create an updated CSV.
6. Do not add any new fields to the CSV, only improve/fix the existing fields.
7. Write a summary of the changes made to each line of the CSV.
8. Suggest next steps for further improvement or verification of the inventory.

Throughout this process, wrap your analysis and decision-making process in <analysis> tags. Be thorough in your exploration and comparison between the video and CSV to ensure each object is correctly identified and logged.

CSV Formatting Rules:
- Use commas only to separate fields, not within fields.
- For fields that inherently include commas, enclose the entire value in double quotes ("). For example Assortment of weight plates 5kg,10kg,20kg -> "Assortment of weight plates 5kg,10kg,20kg"
- If double quotes appear within a field value, escape them by doubling them (""). For example, 'John "JJ" Smith' should be written as "John ""JJ"" Smith".
- Return all columns in the order of the CSV schema.
- Do not use any unescaped double quotes or other special characters that may invalidate the CSV.
- Never use commas in prices. Use 1500.0 instead of $1,500.0.
- Ensure every column has a value; do not skip any fields.
- No duplicate rows. If there are rows with the same values, combine them into one row and use the number_of_items field to reflect how many there are.
  - For number_of_items, always use ints rather than strings. Even if you're unsure/unable to count them for sure, use your best judgement to guess the number of items, e.g. number_of_items=5.

Structure your response as follows:

<initial_analysis>
Provide your initial notes and observations here. Be specific and detailed in your analysis.
</initial_analysis>

<video_final_status>
Write either "ALLITEMSTRACKED" if you believe all significant items in the video have been tracked, or "MOREITEMSTODO" if you think more items could be added. If more items need to be added, include them as a valid CSV in the same format as the original.
</video_final_status>

<steps_to_fix_csv>
Outline the specific steps you'll take to fix or update the input CSV. Be clear and concise in your explanation.
</steps_to_fix_csv>

<updated_csv>
Provide the fully updated CSV with all implemented fixes and additions/removals. Ensure that every entry in the CSV adheres to the provided schema and formatting requirements.
</updated_csv>

<post_fix_summary>
Write a brief summary of the fixes implemented for each line of the CSV. If a line was kept unchanged, mention this as well. Use this format:
* Lines added: [list lines added (if any)]
* Lines removed: [list of lines removed (if any)]
* Lines fixed/amended: [list of lines fixed/amended (if any)]
</post_fix_summary>

<next_steps>
Suggest 2-3 concrete next steps for further improvement or verification of the inventory. These could include actions like double-checking certain items, verifying brands or values, or focusing on specific rooms or categories of items.
</next_steps>

See the following examples for reference:

<examples>
<example_1>
{example_1}
</example_1>

</examples>

The CSV schema to adhere to is:
<csv_schema>
{csv_schema}
</csv_schema>

Current CSV input:
<current_csv>
{csv_input}
</current_csv>

Remember to wrap your thought process in <thinking> tags throughout your response to show your analysis.

Begin by saying "I've got this, time to make sure this CSV is full and complete to the best of my capabilities in relation to the video. I will not add any new columns to the CSV, only improve existing ones." to boost your confidence in fulfilling the tasks at hand. Do not add any rows of items to the <current_csv> that isn't in the video.
"""

input_prompt_final_check = input_prompt_final_check.format(
    csv_schema=get_schema_string(),
    csv_input=final_check_csv_string,
    example_1=example_1_final_check
)

print(f"[INFO] Final check input prompt:\n{input_prompt_final_check}")

model_response_3_with_cache = generate_content(prompt=[input_prompt_final_check],
                                               model=model_gemini_flash_with_cache)

print(model_response_3_with_cache.text)
```

Third output complete!

Let's verify its formatting is correct.

### Format Check 3: Check formatting of output CSV from Prompt 3

Let's check the formatting of our third CSV in our third model output, `model_response_3_with_cache`, using `quick_check_csv` to make sure it's correct.

We've used slightly different tags this time since we asked our model to update an existing CSV rather than create one.

So we'll set `target_start_tag="<updated_csv>"` and `target_end_tag="</updated_csv>"`.

```python
model_response_3_csv, fix_csv_3_required, csv_fixes_to_do_3 = quick_check_csv(model_output=model_response_3_with_cache,
                                                                              target_start_tag="<updated_csv>",
                                                                              target_end_tag="</updated_csv>")
print(f"[INFO] CSV 3 fix required? {fix_csv_3_required}")
print(f"[INFO] Fixes to do:\n{csv_fixes_to_do_3}")

while fix_csv_3_required:
  print(f"[INFO] CSV 3 fix required... fixing...")
  model_response_3_csv = fix_csv(input_csv=model_response_3_csv, fixes_required=csv_fixes_to_do_3)
  model_response_3_csv, fix_csv_3_required, csv_fixes_to_do_3 = quick_check_csv(model_output=model_response_3_csv,
                                                                                target_start_tag="<updated_csv>",
                                                                                target_end_tag="</updated_csv>")
  print(f"[INFO] CSV 3 fix required? {fix_csv_3_required}")
```

No fixes! Nice.

Let's move onto the final outputs.

### Investigate and Save Final Outputs

We come a long way since just having a single video, we've now got CSV containing potentially (we'd have to check this to make sure it works) all of the items in our video ready to be tracked.

Let's save our outputs to CSV and JSON so we can investigate them later if necessary.

```python
# Save to CSV
df_final = csv_string_to_df(csv_string=model_response_3_csv)
df_final = df_final.dropna(thresh=len(df_final.columns) * 0.1).drop_duplicates() # remove duplicates and any rows with 90% NaN values
df_final.to_csv("home_video_item_tracking.csv", index=False)
print(f"[INFO] Final number of items in the DataFrame: {len(df_final)}")

# Inspect the DataFrame output
df_final.head()
```

```python
# Save to JSON
import json

df_final_json = df_final.fillna("nan").astype(str).to_dict(orient="records") # fill empty values with "nan" and convert all to string for JSON
with open("home_video_item_tracking.json", "w") as f:
  json.dump(df_final_json, f, indent=4)

# Inspect the JSON output
df_final_json[:3]
```

### Sort the entries by their certainty flag value

As with other machine learning problems, one good way to investigate your models outputs is to sort by prediction probability.

In our case, we don't necessary have prediction probabilities but we did ask the model to rate how certain it was of a prediction in a `overall_certainty_flag`.

In a user facing app, one good way to investigate the outputs would be to sort by items with lower certainty values and check those out first.

> **Note:** Asking the model to rate its centrainty level of a given prediction isn't a perfect system. But from my experience, it seems to work pretty well in practice. When looking at the samples with the lowest certainty, many of these items tended to be items in the video where little or no time was spent on them (or they are out of view due to not being present in the 1 FPS sampling).

```python
# Show top 10 uncertain items, can investigate these first
df_uncertain = df_final.sort_values(by="overall_certainty_flag", ascending=True)
df_uncertain.head(10)
```

### Get the estimated total worth of items

How much are all the items estimated to be worth?

This will of course be an estimate but is a good starting point to go through and update items if necessary.

It will also be influenced by the counts of items and whether the `estimated_worth` field is the total of those items or per item.

```python
df_final["estimated_worth_total"] = df_final["number_of_items"] * df_final["estimated_worth"]
print(f"[INFO] Total estimated worth counting each row individually: ${df_final.estimated_worth.sum()}")
print(f"[INFO] Total estimated worth multiplying number of items by estimated worth: ${df_final.estimated_worth_total.sum()}")
```

Nice! Of course, this amount would have to be double checked by going through each sample and adjusting the worth if necessary.

But it's a good place to start.

We could take this value along with our database of items to our home insurance company and get insurance for the actual items we have rather than trying to guess what's there.

## Display frames from the video with metdata

Since we have the timestamps of each item in our DataFrame, how about we go through it and get some random samples and overlay the metadata our Gemini model predicted?

```python
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt

# Helper function to convert timestamp to frame number
def timestamp_to_frame(timestamp, fps):
    timestamp_parts = timestamp.split(":")
    seconds = int(timestamp_parts[0]) * 3600 + int(timestamp_parts[1]) * 60 + float(timestamp_parts[2])
    return int(seconds * fps)

# Open the video file
video_capture = cv2.VideoCapture(path_to_video_file)

if not video_capture.isOpened():
    print("Error: Could not open video.")
else:
    # Extract video metadata
    frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_length_seconds = round(frame_count / fps, 2)

    print(f"[INFO] Video Metadata:")
    print(f" - Frame count: {frame_count}")
    print(f" - FPS: {fps}")
    print(f" - Video length: {video_length_seconds} seconds")
    print(f" - Width: {width}")
    print(f" - Height: {height}")

    # Ensure df_final exists and is valid
    if 'df_final' not in locals():
        print("Error: DataFrame 'df_final' is not defined.")
    else:
        # Limit number of frames to display
        num_frames_to_display = min(10, len(df_final))
        sampled_rows = df_final.sample(n=num_frames_to_display)
        frames_to_display = []

        for _, row in sampled_rows.iterrows():
            try:
                frame_number = timestamp_to_frame(row["timestamp"], fps)
                video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                success, frame = video_capture.read()

                if not success:
                    print(f"Warning: Could not read frame at timestamp {row['timestamp']} (frame {frame_number}).")
                    continue

                # Add metadata overlay
                metadata_text = (
                    f"Item Name: {row.get('item_name', 'N/A')}\n"
                    f"Description: {row.get('item_description', 'N/A')}\n"
                    f"Brand: {row.get('item_brand', 'N/A')}\n"
                    f"Condition: {row.get('item_condition', 'N/A')}\n"
                    f"Estimated Value: ${row.get('estimated_worth', 'N/A')}\n"
                    f"Location: {row.get('room', 'N/A')}\n"
                    f"Num items: {row.get('number_of_items', 'N/A')}\n"
                    f"Certainty level: {row.get('overall_certainty_flag', 'N/A')}"
                )

                y0, dy = 50, 30
                for i, line in enumerate(metadata_text.split("\n")):
                    y = y0 + i * dy
                    cv2.putText(frame, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)

                frames_to_display.append(frame)
            except Exception as e:
                print(f"Error processing row: {e}")
                continue

        if frames_to_display:
            # Resize and create a grid for frames
            resized_frames = [cv2.resize(frame, (200, 200)) for frame in frames_to_display]
            grid_cols = 2
            grid_rows = -(-len(resized_frames) // grid_cols)
            grid_height = grid_rows * 200
            grid_width = grid_cols * 200
            grid = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)

            for idx, frame in enumerate(resized_frames):
                row = idx // grid_cols
                col = idx % grid_cols
                grid[row * 200:(row + 1) * 200, col * 200:(col + 1) * 200, :] = frame

            # Convert BGR to RGB for Matplotlib
            grid_rgb = cv2.cvtColor(grid, cv2.COLOR_BGR2RGB)

            # Display the grid
            plt.figure(figsize=(10, 5))
            plt.imshow(grid_rgb)
            plt.axis('off')
            plt.title("Random Frames Grid")
            plt.show()

            # Save the grid as an image file
            grid_path = "random_frames_with_metadata_grid.jpg"
            cv2.imwrite(grid_path, grid)
            print(f"[INFO] Saved Random Frames Grid as '{grid_path}'")
        else:
            print("No frames were successfully captured.")

    video_capture.release()
```

Nice! That looks incredible!

Many of the outputted frames and metadata seem to match the items quite well.

There are a few exceptions but many of these could be either fixed in a dashboard or app.

There are also a couple of issues where the model has predicted an item but it isn't visible in the frame. This is likely due to the 1 FPS sampling of Gemini.

> **Note:** As of November 2024, Gemini uses 1 FPS to sample from video data. This means that for some items in our video, they may get cut off due to being shown in the transition of frames or in between Gemini sampling rates. This can explain why some visualized images aren't as high quality as others. One potential way around is to reduce the frame rate of your video (e.g. slow it down by 50%) before passing it to Gemini but I'm yet to try this. There may also be updates in the future to the model to make it capable of higher sampling rates.

## Total up the token counts and costs

Let's figure out how much all of this cost!

To do so, we'll compile all of our model's outputs (e.g. `model_response_1_with_cache` etc) and then get their respective token counts.

```python
# Get all token counts from model responses
# Note: This doens't include any CSV fixing responses, they could be tracked in a separate instance.
all_cache_outputs = [model_response_1_with_cache, model_response_2_with_cache, model_response_3_with_cache]
all_cache_outputs_token_counts = []
for i, output in enumerate(all_cache_outputs):
  input_tokens, output_tokens, cached_tokens = get_token_counts(output)
  all_cache_outputs_token_counts.append(
      {"output_number": i,
       "input_tokens": input_tokens,
       "output_tokens": output_tokens,
       "cached_tokens": cached_tokens}
  )
all_cache_outputs_token_counts
```

And since we're using an input of over 128k tokens, we'll use our `gemini_flash_input_over_128k` pricing dictionary from the [Gemini pricing page](https://ai.google.dev/pricing#1_5flash).

```python
# All prices are in $USD / 1 million tokens
gemini_flash_input_over_128k
```

Token counts and pricing acquired, let's iterate through these and calculate the pricing for the input, output and cached tokens.

```python
for item in all_cache_outputs_token_counts:
  input_cost = item["input_tokens"] * (gemini_flash_input_over_128k["input"] / 1_000_000)
  output_cost = item["output_tokens"] * (gemini_flash_input_over_128k["output"] / 1_000_000)
  cache_cost = item["cached_tokens"] * (gemini_flash_input_over_128k["context_caching"] / 1_000_000)

  # Input cost with no cache = combine input tokens + cache tokens
  input_no_cache_cost = (item["input_tokens"] + item["cached_tokens"]) * (gemini_flash_input_over_128k["input"] / 1_000_000)

  item["total_cost_with_cache"] = input_cost + output_cost + cache_cost
  item["total_cost_no_cache"] = input_no_cache_cost + output_cost
```

Now we can turn our costs into a DataFrame.

```python
df_cost = pd.DataFrame(all_cache_outputs_token_counts)
df_cost
```

Finally, we'll calcualte the cache storage costs (this is only required as a one off fee since it is time-based rather than per input).

And then we can calculate the total costs for inferencing over a 10 minute video three times with and without cache.

```python
# Only total the cache storage cost for 1 item because the cache is reused
CACHE_HOURS = 0.25 # Note: Cache for 15 minutes (that's enough time to do what we need)
cache_storage_cost = all_cache_outputs_token_counts[0]["cached_tokens"] * ((gemini_flash_input_over_128k["context_caching_storage"] / 1_000_000) * CACHE_HOURS)
print(f"[INFO] Cache storage cost: {cache_storage_cost}")

# Get the total cost of using cache (this requires the additional cache storage fee)
total_cost_using_cache = df_cost["total_cost_with_cache"].sum() + cache_storage_cost

# Get the total cost of not using cache
total_cost_no_cache = df_cost["total_cost_no_cache"].sum()

print(f"[INFO] Total cost with caching inputs ($USD): {round(total_cost_using_cache, 5)}")
print(f"[INFO] Total cost without caching inputs ($USD): {round(total_cost_no_cache, 5)}")

# Compare costs
if total_cost_using_cache < total_cost_no_cache:
  print(f"[INFO] Total cost using cached inputs is {round(total_cost_no_cache/total_cost_using_cache, 5)}x cheaper than not using cached inputs.")
else:
  print(f"[INFO] Total cost not using cached inputs is {round(total_cost_using_cache/total_cost_no_cache, 5)}x cheaper than using cached inputs.")
```

Woah! Looks like the total cost for inferencing over our video was about $0.069 (about 7 cents) using cache.

And $0.0835 (about 8.5 cents) without cache.

When put in comparison to home and contents insurance fees, that sounds quite cheap.

For example, if your insurance policy was $100 per month, it doesn't seem unreasonable to update it with the actual items in your house for less than 10c.

> **Note:** For our use case, using caching turned out to be cheaper than not using caching. This is because we went over our input video 3x (one per major prompt). Caching would likely pay more dividends if we were going to perform more inference runs over our video in quick succession.

### Plotting the total costs of using caching and not against each other

Let's make the cost comparison using caching and not using caching more visual by turning them into a plot.

```python
import matplotlib.pyplot as plt

plot_labels = ["With Cached Input Tokens", "Without Cached Input Tokens"]
plot_values = [total_cost_using_cache, total_cost_no_cache]
plt.figure(figsize=(10, 7))
plt.bar(plot_labels, plot_values)
plt.title("Comparison of Total Costs for Caching and Not Caching Inputs (Three Passes of Large Video File)")
plt.ylabel("Price ($USD)")

# Calculate the cost difference
difference = total_cost_no_cache - total_cost_using_cache
percent_cheaper = round((difference / total_cost_no_cache) * 100, 2)

# Annotate the arrow
plt.annotate(
    "",
    xy=(0, total_cost_using_cache + 0.007),  # Start of the arrow
    xytext=(1, total_cost_no_cache),  # End of the arrow
    arrowprops=dict(color="black", arrowstyle="->", lw=1)
)

# Add percentage text above the left bar
plt.text(
    0,  # x position (index of "With Cached Input Tokens")
    total_cost_using_cache + (max(plot_values) * 0.05),  # Slightly above the left bar
    f"{percent_cheaper}% Cheaper",
    fontsize=10,
    ha="center",
    color="green"
)

# Show values on top of bars
for i, v in enumerate(plot_values):
    plt.text(i, v + (max(plot_values) * 0.01), f"${v}", ha='center');
```

Excellent! Looks like caching our inputs saved us about 20% in costs.

## Bonus: Gemini Bounding Box Detection

Gemini models are capable of outputting bounding box coordinates given an image as well as something to detect.

> **Note:** In my experiments, I've found this works best on single images (e.g. one image at time), rather than doing it during the video processing steps.

So let's iterate through our video file and save each of the detected frames using their timestamps as images to file.

We can then use these images along with a bounding box detection prompt to get Gemini to detect the target item in the frame.

> **Note:**
> * Gemini returns bounding boxes in a specialized [ymin, xmin, ymax, xmax] format. Keep this in mind when processing bounding boxes from an output.
> * See documentation for bounding box detection: https://ai.google.dev/gemini-api/docs/vision?lang=python#bbox
> * See example notebook for bounding box detection with Gemini: https://colab.research.google.com/drive/1eDvf_Ky9jLOZFShgHrm4GI-wkAaQnue6?usp=sharing#scrollTo=wizbxA1lm-Tj

### Save all frames from JSON to file

To make bounding box predictions, we'll first save all highlighted frames in our video to file.

We can do this by getting the timestamp of each of our target item and then saving the frame of the video at that timestamp to file.

```python
import cv2
import os
import json
from tqdm import tqdm  # For progress bar

# Create a directory to save frames
output_dir = "saved_frames"
os.makedirs(output_dir, exist_ok=True)

# Dictionary to hold metadata for saved frames
frames_metadata = []

# Load video
video_capture = cv2.VideoCapture(path_to_video_file)

# Go through video and save individual frames along with metadata
if not video_capture.isOpened():
    print("[INFO] Error: Could not open video.")
else:
    fps = video_capture.get(cv2.CAP_PROP_FPS)  # Video frames per second
    print(f"[INFO] Video FPS: {fps}")

    # Iterate through the JSON data with a progress bar
    for item in tqdm(df_final_json, total=len(df_final_json), desc="Processing frames"):
        try:
            # Get frame number from timestamp
            frame_number = timestamp_to_frame(item["timestamp"], fps)
            video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            success, frame = video_capture.read()

            if not success:
                print(f"Warning: Could not read frame at timestamp {item['timestamp']} (frame {frame_number}).")
                continue

            # Save frame to a file
            frame_filename = os.path.join(output_dir, f"frame_{item['item_number']}.jpg")
            cv2.imwrite(frame_filename, frame)

            # Add filename to JSON item
            item["frame_filename"] = frame_filename

            # Append metadata with frame file path
            frames_metadata.append(item)

        except Exception as e:
            print(f"Error processing item {item['item_number']}: {e}")
            continue

    # Save metadata to a JSON file
    metadata_file = os.path.join(output_dir, "frames_metadata.json")
    with open(metadata_file, "w") as json_file:
        json.dump(frames_metadata, json_file, indent=4)

    print(f"\n[INFO] Saved {len(frames_metadata)} frames and metadata to '{output_dir}'.")

# Release video capture
video_capture.release()
```

If our `frames_metadata.json` file exists, we can import it.

```python
# Import existing frames and metadata
output_dir = "saved_frames"

import json

with open(f"{output_dir}/frames_metadata.json") as f:
  frames_metadata = json.load(f)

frames_metadata[:3]
```

### Write helper function to process Gemini box outputs

Gemini outputs bounding box coordinates in the format `[y_min, x_min, y_max, x_max]`.

The documentation states that:

> Gemini models are trained to return bounding box coordinates as relative widths or heights in the range of [0, 1]. These values are then scaled by 1000 and converted to integers. Effectively, the coordinates represent the bounding box on a 1000x1000 pixel version of the image. Therefore, you'll need to convert these coordinates back to the dimensions of your original image to accurately map the bounding boxes.

So let's write a function to go from Gemini boudning box format to a format normalized to our image.

> **Note:** You can read about the Gemini bounding box format as well as the steps to convert Gemini boxes in the documentation: https://ai.google.dev/gemini-api/docs/vision?lang=python#bbox

```python
def gemini_bounding_box_converter(bounding_box,
                                  image_height=720,
                                  image_width=1280,
                                  return_type="relative"):
  """Converts bounding box outputs from Gemini to dimensions of original image.

  Gemini bounding boxes come in the format [y_min, x_min, y_max, x_max]

  Instructions to convert (from the docs):

  The model returns bounding box coordinates in the format [ymin, xmin, ymax, xmax].

  To convert these normalized coordinates to the pixel coordinates of your original image, follow these steps:

  1. Divide each output coordinate by 1000.
  2. Multiply the x-coordinates by the original image width.
  3. Multiply the y-coordinates by the original image height.

  See docs: https://ai.google.dev/gemini-api/docs/vision?lang=python#bbox
  See example: https://colab.research.google.com/drive/1eDvf_Ky9jLOZFShgHrm4GI-wkAaQnue6?usp=sharing#scrollTo=wizbxA1lm-Tj
  """
  box_convert = [item / 1000 for item in bounding_box]

  if return_type == "relative":
    return box_convert
  else:
    # Convert height (y coordinates)
    box_convert[0], box_convert[2] = round(box_convert[0] * image_height, 2), round(box_convert[2] * image_height, 2)

    # Convert width (x coordinates)
    box_convert[1], box_convert[3] = round(box_convert[1] * image_width, 2), round(box_convert[3] * image_width, 2)

  return box_convert
```

### Create a model to detect bounding boxes given a target item

Let's setup a model focused on detecting bounding boxes.

> **Note:** This process of creating a model to detect boxes is inline with the idea of one model, one task. A model that is focused on a specific task generally performs better than a model focused on performing many tasks.

```python
# Create specific box detection model
import os
import google.generativeai as genai

# For Google Colab (if needed)
# from google.colab import userdata
# # Note: Can set this up in "Secrets" -> Add new key
# GEMINI_API_KEY = userdata.get("GEMINI_API_KEY")
# genai.configure(api_key=GEMINI_API_KEY)

# Create model for detecting boxes
model_gemini_flash_box_detector = genai.GenerativeModel(model_name="models/gemini-1.5-flash-002",
                                                        generation_config={'temperature': 0.2,
                                                                           'top_p': 0.5,
                                                                           'top_k': 5,
                                                                           'max_output_tokens': 8192,
                                                                           'response_mime_type': 'text/plain'},
                                                        system_instruction="You are a sensational bounding box detector who outputs bounding box \
                                                        coordinates for target items in images in [ymin, xmin, ymax, xmax] format.")

# Create helper function for generating content
def generate_content(prompt, model):
  """Returns a given model's output for a given prompt."""
  return model.generate_content(prompt)

example_box_output_no_image = generate_content(prompt="Testing 1, 2, 3 are there any boxes in this image?",
                                               model=model_gemini_flash_box_detector)
example_box_output_no_image.text
```

### Predict bounding boxes on a single image

Bounding box model ready, let's now get a random sample from our data and try to predict bounding boxes for a target item on it.

We'll get the item's name and description to pass in as a prompt to our box detection model.

```python
import random
from PIL import Image

# Choise a random sample from our dataset of frames
target_sample = random.choice(frames_metadata)
print(f"[INFO] Performing bounding box inference on sample:")
print(target_sample)

# Get sample metadata
image_path = target_sample["frame_filename"]
item_name = target_sample["item_name"]
item_description = target_sample["item_description"]

target_image = Image.open(image_path)
print(f"Image height: {target_image.height} | Image width: {target_image.width}")

# Display image in smaller size for speed
target_image.resize(size=(target_image.width // 2, target_image.height // 2))
```

Now let's create a prompt to guide our model to detect boxes.

A few things to note:

- We'll pass the target item and its description to the input prompt.
- We'll handle multiple boxes by asking the model to return a valid list of boxes.
- If the target item isn't visible, we'll ask the model to return all zeros (e.g. `[0, 0, 0, 0]`).
- We'll ask the model to write a note to itself about what the item might look like at the start to help its detection capabilities (e.g. "Looking for a black vacuum, I've found it in the middle of the image.").

```python
image_only_box_prompt = """Return bounding boxe(s) for the target item in the format [[ymin, xmin, ymax, xmax]] in a list inside <boxes> tags.

Even if the image is blurry, do your best to try and detect it.

Always return boxes as list of lists for example:
1. One single box: [[box_1]]
2. Multiple boxes: [[box_1], [box_2], [box_3]...]

Start by extending the target item with more information about what you can see in the image in <visual_analysis> tags.
 - For example, prepare yourself by writing "Looking for the 'target_item' and expanding upon it's description. I see ..., let me clarify and see if it's the target item...".
 - Also write a set of instructions for how you will return the boxes, e.g. "I will return the box coordinates in the form of a list of lists, one set of coordinates per item."

Then detect the item based on your <visual_analysis>.

If the target item is not visible, return a single box of all zeros [0, 0, 0, 0], this will indicate that the item is not visible.

Your output should look like the following examples:

<example_1_single_item>
Target item: Microwave (Black built-in microwave)
<visual_analysis>
Looking for the black built-in microwave. I see a black microwave oven built into the dark-grey kitchen cabinets. It is located on the top left of the cabinet unit. The microwave is rectangular and black.
</visual_analysis>
<boxes>
[[ymin, xmin, ymax, xmax]]
</boxes>
</example_1_single_item>

<example_2_single_item>
<visual_analysis>
[Visual analysis of the image looking for the target item.]
</visual_analysis>
<boxes>
[[10, 470, 222, 702]]
</boxes>
</example_2_single_item>

<example_1_multiple_items>
<visual_analysis>
Looking for the two white bedside tables in the image. I can see two small white tables on either side of the bed, each with a lamp and some small decorative items on top. They appear to be of similar size and shape. Time to output bounding boxes. Going to output a list of two box coordinates, one for each lamp.
</visual_analysis>
<boxes>
[[ymin, xmin, ymax, xmax],
[ymin, xmin, ymax, xmax]]
</boxes>
</example_1_multiple_items>

<example_2_multiple_items>
Target item: Wooden stools with metal legs (three stools total)
<visual_analysis>
Looking for three wooden tools with metal legs. I can see three tools through the centre of the image in a straight line. Going to output coordinates for three boxes enclosed in a list.
</visual_analysis>
<boxes>
[[446, 189, 692, 372],
[403, 360, 677, 506],
[394, 506, 718, 681]]
</boxes>
</example_2_multiple_items>

Return only <video_analysis> and <boxes> tags with the box coordinates for the target item and nothing else.

Target item: {item_name} ({item_description})
"""

image_only_box_prompt_formatted = image_only_box_prompt.format(item_name=item_name,
                                                               item_description=item_description)
print(f"[INFO] Image box detection input prompt:\n{image_only_box_prompt_formatted}")
```

Input box prompt made, now let's try it on a single image (we'll pass the image as a prefix to the string prompt).

```python
image_only_box_detection_response = generate_content(prompt=[target_image, # Note: Pass input image as prefix to the input prompt.
                                                             image_only_box_prompt_formatted],
                                                     model=model_gemini_flash_box_detector)

print(f"\n[INFO] Image only box detection output:\n{image_only_box_detection_response.text}")
```

Looks good!

We've got some box coordinates.

But before we can plot these, we'll need to filter them out.

Let's write a small helper function to extract box coordinates from a model output string.

```python
def box_string_filter(box_model_output):
  """Filters out bounding box coordinates from a Gemini model output.

  Mostly looking for what's inside <boxes></boxes> tags.
  """
  model_text = box_model_output.text.lower()

  # Remove common issues
  model_text = model_text.replace("```python", "").replace("```", "").replace("```xml", "")

  # If output text doesn't contain boxes, default to all zeroes (can inspect all zeroes later)
  if ("<boxes>" not in model_text) or ("</boxes>" not in model_text):
    print(f"[INFO] Model box output does not contain box tags: <boxes></boxes>, defaulting to all zeroes.")
    boxes_raw = [[0, 0, 0, 0]]
  else:
    boxes_raw = [eval(model_text.split("<boxes>")[1].split("</boxes>")[0].strip())] # eval() = turn string-based list into an actual Python list

  return boxes_raw
```

Now we've got a way to extract our boxes in raw format (e.g. Gemini format).

Let's write some visualization code to view the predicted box on the target image.

We'll use our `gemini_bounding_box_converter` to conver the box coordinates to be normalized based on our image's height and width.

```python
from PIL import ImageDraw, ImageFont

# Filter boxes from model output
boxes = box_string_filter(box_model_output=image_only_box_detection_response)

# Add boxes to target_sample
target_sample["bounding_boxes"] = boxes

# Open image
image = Image.open(image_path)
image_height = image.height
image_width = image.width

# Can return results as plotted on a PIL image (then display the image)
draw = ImageDraw.Draw(image)

# Get a font from ImageFont
font = ImageFont.load_default(size=20)

for item in [target_sample]:

    # Check for list of lists
    bounding_boxes = item["bounding_boxes"]

    if isinstance(bounding_boxes[0][0], list): # This will be False if there is only one box
      print("[INFO] Bounding boxes seems to be a list of lists, adjusting for multi box plotting.")
      bounding_boxes = bounding_boxes[0]

    print(f"[INFO] Bounding boxes: {bounding_boxes}")

    for box in bounding_boxes:
      print(f"[INFO] Drawing box: {box}")
      box_convert = gemini_bounding_box_converter(bounding_box=box,
                                                  image_height=image_height,
                                                  image_width=image_width,
                                                  return_type="absolute")
      # Create coordinates
      y1, x1, y2, x2 = box_convert

      # Get label_name
      label_name = item["item_name"]
      targ_color = "green"

      # Draw the rectangle
      draw.rectangle(xy=(x1, y1, x2, y2),
                      outline=targ_color,
                      width=3)

      # Create a text string to display
      text_string_to_show = f"{label_name}"

      # Draw the text on the image
      draw.text(xy=(x1, y1),
                  text=text_string_to_show,
                  fill="white",
                  font=font)

image
```

How cool is that!!!

We've now got a way to visually enhance our item database with localization of where the target item is in an image.

### Detect boxes on all frames

Now we've got a way to predict bounding boxes on an image, why stop on a single image?

Let's repeat the process for every image in our `saved_frames` folder (this will be every frame Gemini has deemed there's a target item pictured).

```python
from tqdm.auto import tqdm

# Keep track of the output prompts for later token counting later on if needed
box_detection_output_prompts = []

for item in tqdm(frames_metadata):

  # Get target sample information
  target_image_path = item["frame_filename"]
  target_item_name = item["item_name"]
  target_item_description = item["item_description"]

  print(f"[INFO] Detecting boxes on image: {target_image_path} | Item name: {target_item_name} | Description: {target_item_description}")

  # Prepare target image and get metadata, image dimensions are important for box conversion
  target_image = Image.open(target_image_path)
  image_height = target_image.height
  image_width = target_image.width

  # Format box detection prompt with target image
  image_only_box_prompt_formatted = image_only_box_prompt.format(item_name=target_item_name,
                                                                 item_description=target_item_description)

  # Prompt Gemini model to detect on image
  image_only_box_detection_response = generate_content(prompt=[target_image,
                                                               image_only_box_prompt_formatted],
                                                       model=model_gemini_flash_box_detector)
  box_detection_output_prompts.append(image_only_box_detection_response)

  # Filter the box output string from Gemini
  boxes_raw = box_string_filter(box_model_output=image_only_box_detection_response)

  # Update the target item with detected boxes
  item["boxes_raw"] = boxes_raw
```

Since Gemini predicts boxes in a special format, let's use our `gemini_bounding_box_converter` to convert the `boxes_raw` key to coordinates normalized to our image.

```python
# Convert raw box detections from Gemini style to absolute style
for item in frames_metadata:
  list_of_boxes = item["boxes_raw"][0]
  boxes_converted = []
  for original_box in list_of_boxes:
    box_convert = gemini_bounding_box_converter(bounding_box=original_box,
                                                image_height=image_height, # all of our images have the same height and width (height=720, width=1280)
                                                image_width=image_width,
                                                return_type="absolute")
    boxes_converted.append(box_convert)

  assert len(list_of_boxes) == len(boxes_converted), f"Length of original input boxes ({len(list_of_boxes)}) does not match length of converted boxes ({len(boxes_converted)})"

  item["boxes_converted"] = boxes_converted
```

Now we can save our frames with metdata and boxes to file.

```python
# Save box detections to file
with open("saved_frames/frames_metadata_with_boxes.json", "w") as f:
  json.dump(frames_metadata, f, indent=4)

# Copy to Google Drive for later use
# !cp saved_frames/frames_metadata_with_boxes.json drive/MyDrive/gemini-large-context-window-kaggle-competition/house-tracker-frames/
```

## View an image with predicted bounding boxes

And to finish off, we can visualize a random frame with its Gemini predicted bounding box.

```python
random_box_sample = random.choice(frames_metadata)

random_image = Image.open(random_box_sample["frame_filename"])
image_height = image.height
image_width = image.width

# Can return results as plotted on a PIL image (then display the image)
draw = ImageDraw.Draw(random_image)

# Get a font from ImageFont
font = ImageFont.load_default(size=20)

# Get the converted boxes for the target image
bounding_boxes = random_box_sample["boxes_converted"]

for box in bounding_boxes:

  # Create coordinates
  y1, x1, y2, x2 = box

  # Get label_name
  label_name = random_box_sample["item_name"]
  targ_color = "green"

  # Draw the rectangle
  draw.rectangle(xy=(x1, y1, x2, y2),
                  outline=targ_color,
                  width=3)

  # Create a text string to display
  text_string_to_show = f"{label_name}"

  # Draw the text on the image
  draw.text(xy=(x1, y1),
              text=text_string_to_show,
              fill="white",
              font=font)

random_image
```

How cool is that!

## Bonus 2: Create an app to view the images + metadata

For future review, now we've got the frames and the metadata, we could view these in an application and review/update them if needed.

This would be the next step in building a proof of concept to further improve the tracking details.

Gemini starts the tracking off but the owner/person can improve/review the data.

You can see a demo app I've made to do this on Hugging Face Spaces: https://huggingface.co/spaces/mrdbourke/keeptrack-frames-viewer

> **Note:** I used Gemini to generate this web app. By feeding Gemini the JSON schema as well as some specifications, it created a good first start. Then I made a couple of tweaks. It's not perfect but it's an example of what a data inspection interface may look like (perhaps the design may need a bit of improving before shipping).

```python
from IPython.core.display import HTML

HTML('''
<iframe
    src="https://mrdbourke-keeptrack-frames-viewer.static.hf.space"
    frameborder="0"
    width="1500"
    height="1500"
></iframe>
''')
```

## Appendix

Some things I took note of during the project which may be of future help (emphasis on the may).

### Tidbits and Takeaways

* **Data extraction from model outputs:** Use XML/HTML-like tags to be able to parse out target items easily, e.g. "Return the output in `<csv></csv>` tags".
* Refer to timestamps in video as `MM:SS` format - https://ai.google.dev/gemini-api/docs/vision?lang=python#refer-timestamps
    * Though I did see another section of the docs which said refer to them as `HH:MM:SS`?
* **CSV vs JSON:** Why use CSV instead of JSON? - much less output tokens required (e.g. 2-10x less for ~100 rows)
* **How many examples to use?** I've found good results with at least 1-4. Have yet to explore using more. Recent research shows that you can dial this up to *many many* examples. See Many-shot ICL (many-shot in-context learning) paper by DeepMind - https://arxiv.org/abs/2404.11018
* **Bounding boxes:** Bounding boxes work much better on individual images rather than frames.
    * **Important:** Gemini's bounding box format seems to be `[ymin, xmin, ymax, xmax]`. This tripped me up to begin with because in previous object detection projects I've always used x-axis first (e.g. `XYXY` format).
        * For items that are not visible (e.g. items in a bag), can return a bounding box of all zeros `[0, 0, 0, 0]` rather than nothing, these can be inspected later on.
        * Models are biased to returning *something* so for null outputs best to return something to signify this rather than nothing.
* **Sampling rate of video:** For many items in video, you may need more than 1 FPS for sampling, so might want to increase the length of your video by decreasing the framerate, see point 7 here: https://developers.googleblog.com/en/7-examples-of-geminis-multimodal-capabilities-in-action/
    * If an item is in frame for less than a second (very common in video), it may be missed out on.
        * This might be improved using a lower frame rate?
* **Caching is helpful but beware storage costs:** Caching input tokens = 4x cheaper than not caching but cache storage costs are expensive compared to inputs.
    * Keep cache storage times to an absolute minimum.
    * If you are performing inference more than 2x on a large input, it makes sense to cache. But if you only need a pass or two, the caching storage fees may not be worth it.
* **Experimenting is paramount:** A continual process of trial and error is required to get outputs right.
    * Do something -> check -> do something -> check... etc
    * Start simple with prompts e.g. literally say "do X" before scaling up prompts/examples to catch errors.
    * Once you have a good set of instructions/examples, you can use Gemini to write more examples for you.
    * Machine Learner's motto: *Experiment, experiment, experiment!*
* **One model, one task:** One model to fix get information from a video, one model to fix CSVs and so on...
    * If I have a new task, I've found it best to create a dedicated model instance for that task.
* **Visualize, visualize, visualize!** Design simple interfaces to view your models outputs and fix them if needed. My favourite as often Python print statements with random samples from the dataset.
* **Cool thing:** This wouldn't have been possible a few years ago. Even with state-of-the-art computer vision, transcription and NER models, combining them into this kind of pipeline would've taken quite a large amount of engineering work. Now we can do it with a few string-based prompts. Gemini is cool :D

### Tried and failed

* Bounding boxes work *much* better on single images (e.g. screenshots of frames) rather than trying to get them in the video detection step.
  * I tried to add a "bounding_boxes" field to the CSV outputs and it didn't work, boxes were all over the place or the model outputted the same box for every item.
    * Much simpler way was to capture each significant frame (based on timestamps) and then try to identify the bounding boxes with a dedicated prompt (this is inline with the recurring theme of one model, one task).

### Future avenues

Some ideas on future approaches and ideas.

* Creating a "story" of a video seems to work quite well, e.g. "watch this video and produce a [memory palace](https://en.wikipedia.org/wiki/Method_of_loci)-like story"
  * Once you've got a "story" of a video, can use the cached video as well as the story to turn it into structured data...
  * From a conceptual overview, this seems to make sense as Gemini is an LLM after all and language of *stories* may be easier to generate than structured data? (I'm not sure here, this would take some more investigation)
  * Turns out this actually works quite well...
* Automatic scene adaption: The schema for this project was focused on household items for home & contents insurance purposes (very niche, yes but I'm moving house and this problem has been front of mind).
  * What would be a future avenue would be to explore "KeepTrack for anything", e.g. this process works well on vinyls/records as well (I tested that) but some items would require a schema more tailored to them.
    * Perhaps future works automatically creates the schema based on the video inputs, then you could truly "KeepTrack of anything".
* To take advantage of the long context window, could fold *all* of the outputs of previous passes into subsequent prompts, e.g. continually build upon previous outputs.
* What do the outputs look like without audio? Are they still of high quality?
  * Update: Tried this. Works quite well. Almost on par with video + audio. But of course is missing information contained in the audio such as "there's a MacBook Pro in that backpack" (item mentioned but not visible).
* Make better evaluations to compare the Gemini model's outputs to what we would expect.
  * One of the most important parts of any machine learning project is having good evals/tests. If I were to pursue this further or productionize a system like this, I'd want some ground truth examples to compare the Gemini outputs to.
* Enhance FPS usage to cover more frames in the video? - some items get missed because of lack of frames?
  * Could a forward and backward pass on the video help here?
    * E.g. if the video can fit into the context window several times, could you just pass the same video in reverse and inference over it twice?

### Raw Notes

My raw notes from experimenting with the Gemini API and reading the documentation.

### Video sampling

* File API service extracts image frames from videos at 1 frame per second (FPS) and audio at 1Kbps, single channel, adding timestamps every second.
  * **Note:** Details of fast action sequences may be lost at the 1 FPS frame sampling rate. Consider slowing down high-speed clips for improved inference quality.
    * For example, for videos where something may only be visible for a fraction of a section, you could slow the video down to increase frame rate? (question mark to check this)
* Image sizes:
  * Larger images are scaled down to a maximum resolution of 3072x3072.
  * Smaller images are scaled up to 768x768 pixels.
    * These sizes could come into play when sampling video data.
* Token counts:
  * Image frames (individual frames) - 258 tokens.
  * Audio - 32 tokens per second.
  * Token metadata - ~10-50 tokens? (**note:** this is unknown, experiments will show more)
  * Total - ~300 tokens per second.
    * 1M context window can fit slightly less than an hour of video.
    * Gemini 1.5 Pro and 1.5 Flash can also support a maximum of 3,600 image files (about ~1M tokens with a token count of ~300 per image).
* To ask questions about time-stamped locations, use the format `MM:SS`, where the first two digits represent minutes and the last two digits represent seconds.
* Prompt formatting tips:
  * Use one video per prompt.
  * If using a single video, place the text prompt after the video.

### Uploading a video file using the File API

* **Size limits:** File API size limits:
  * Up to 20 GB of files per project.
  * Maximum per-file size of 2 GB.
  * Files are stored for 48 hours.
    * Can be accessed in that period with API key.
    * Cannot be downloaded using any API.
  * Available at **no cost** in all regions where the Gemini API is available.
* **Video file state:**
  * Video files have a `State` field in the File API.
  * When video is uploaded, it will be in the `PROCESSING` state until it is ready for inference.
  * Only video files in the `ACTIVE` state can be used for model inference.

### Ideas

* Break the video down into smaller pieces?
  * Can you just split video into a bunch of images?
  * Transcribe the audio and use the text?
    * E.g. if file upload is becoming a pain, just break the file uploads into chunks.
* Use a combination of PaliGemma to encode images + text descriptions for retrieval on the video?
* Use Gemini Video API for making questions and answers for my courses/videos:

```
Input prompt: "Give me a good use for Gemini's long context capabilities with video"

Output text:

---

A good use case for video with the Gemini API would be intelligent video summarization and analysis for educational purposes.

Here's how it could work:

Input: A user uploads a long lecture video (e.g., a university lecture, a complex tutorial, or a webinar).

Gemini's Role: The Gemini API processes the video, leveraging its multimodal capabilities. This includes:

* Speech-to-text transcription: Converting the audio into text.
* Video analysis: Identifying key moments, like changes in speaker, screen shares, or significant visual cues (e.g., diagrams appearing).
* Semantic understanding: Understanding the context and meaning of the transcription and * visual elements. Gemini's large language model capabilities are crucial here.

Output:

The API generates several outputs:

* Automated summary: A concise text summary of the lecture's key points, highlighting important concepts and conclusions.
* Chapterization/Timestamped highlights: The video is broken down into smaller, thematic segments with timestamps, allowing users to quickly navigate to specific topics.
* Keyword extraction: Identifying key terms and concepts discussed in the lecture.
* Question answering: The user can ask questions about the video's content, and Gemini can provide answers based on its understanding.
* Personalized learning paths: Based on the user's learning style or specific areas of interest, the API could recommend specific segments of the video or create a tailored study plan.

This use case leverages Gemini's strengths: its ability to handle both visual and textual information, its advanced understanding of language, and its powerful reasoning capabilities. The result is a significantly improved learning experience, allowing students to quickly grasp the key concepts and efficiently review materials. It also allows educators to easily create interactive learning resources.
```

### Documentation Links

* See this example for long output generation with the Gemini API -- https://github.com/google-gemini/cookbook/blob/main/examples/Story_Writing_with_Prompt_Chaining.ipynb
* Context caching docs for reusing long context - https://ai.google.dev/gemini-api/docs/caching?lang=python
* Gemini Vision docs - https://ai.google.dev/gemini-api/docs/vision?lang=python
  * Gemini Video docs - https://ai.google.dev/gemini-api/docs/vision?lang=python#prompting-video
  * Get the bounding box for an object in an image with Gemini - https://ai.google.dev/gemini-api/docs/vision?lang=python#bbox
    * This could be returned alongside the image/frame?
* Get token counts of inputs and outputs documentation - https://ai.google.dev/gemini-api/docs/tokens?lang=python
* Gemini Cookbook on GitHub - https://github.com/google-gemini/cookbook
  * Cookbook for video summarization - https://github.com/google-gemini/cookbook/blob/main/examples/Analyze_a_Video_Summarization.ipynb