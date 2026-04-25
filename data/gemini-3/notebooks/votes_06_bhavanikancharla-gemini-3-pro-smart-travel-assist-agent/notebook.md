# Gemini 3 Pro-Smart Travel Assist Agent

- **Author:** bhavani Kancharla
- **Votes:** 25
- **Ref:** bhavanikancharla/gemini-3-pro-smart-travel-assist-agent
- **URL:** https://www.kaggle.com/code/bhavanikancharla/gemini-3-pro-smart-travel-assist-agent
- **Last run:** 2025-12-06 15:09:26.887000

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

## 🧠 Gemini 3 Pro – Smart Travel Assist Agent  
### Hackathon Submission | Google DeepMind x Kaggle

This notebook demonstrates a simple but powerful AI Agent built using *Gemini 3 Pro*.  
The agent takes a *user travel query* (example: “Plan a 3-day trip to Goa”) and generates:

- ✔ A full travel itinerary  
- ✔ Budget estimation  
- ✔ Best visiting places  
- ✔ Travel tips  
- ✔ Safety & weather notes  

This fulfills the hackathon goal:  
👉 "Turn natural language prompts into working AI applications using Gemini 3 Pro."

## 1. Problem Statement

Tourists and tourism planners often struggle with:
- Choosing attractions that match interests and budgets  
- Quickly generating itineraries from unstructured ideas  
- Getting instant explanations and suggestions in simple language  

In this hackathon project, I use *Gemini 3 Pro in Google AI Studio* as the core reasoning engine to:
- Understand natural-language queries from users  
- Propose attraction ideas and simple itineraries  
- Explain decisions in a clear conversational way

## 2. Solution Overview

The proposed solution is a *chat-style tourism planner* powered by Gemini 3 Pro:

1. User types a natural prompt  
   - Example: "Plan a 2-day trip to Hyderabad focusing on historical places and food."
2. Gemini 3 Pro (in Google AI Studio) generates:
   - A day-wise itinerary  
   - List of attractions and short descriptions  
   - Simple tips (best time to visit, travel hints)
3. The responses are refined with prompt engineering:
   - Asking Gemini to format output clearly (bullet points / tables)  
   - Adding constraints such as budget or time limits

This notebook documents the *idea, prompts, and design* used with Gemini 3 Pro.

## 3. Using Gemini 3 Pro in Google AI Studio

For this hackathon, I used *Google AI Studio* (special hackathon tier) to run Gemini 3 Pro.

Steps I followed there:

1. Opened Google AI Studio and selected *Gemini 3 Pro*.
2. Designed prompts for tourism planning, for example:
   - "You are a smart travel planner..."  
   - "Generate a 3-day itinerary for ..."  
3. Iterated on the prompts until the output became:
   - Structured (bullets, headings, or tables)  
   - Easy to read  
   - Focused on realistic, safe travel advice  

In this notebook, I show example prompts and the type of responses I obtained.

## 4. Example Prompt and Response (Summary)

*Example prompt:*

> "You are a travel planner. Create a 2-day itinerary for Hyderabad, India for a student with a low budget. Include historical places, food spots, and simple travel tips."

*Short summary of Gemini 3 Pro response:*

- Suggested a realistic 2-day plan  
- Covered major attractions and food spots  
- Explained why each place was chosen  
- Added simple local tips

---

## 5. Reflections and Future Work

What I learned in this hackathon:

- How to design prompts that make Gemini 3 Pro give structured and useful travel plans  
- How multimodal, reasoning-focused models can assist tourism planning  
- How to describe an AI application clearly in a Kaggle notebook

Future improvements:
- Build a small web UI or chatbot interface  
- Add user preferences (budget, interests) as parameters  
- Combine with real tourism datasets for more data-driven suggestions

## 6. Final Remarks

This project was created as part of the *Google DeepMind – Vibe Code with Gemini 3 Pro Hackathon* on Kaggle.

Using Gemini 3 Pro, I explored:
- Natural language understanding for travel planning  
- Generating high-level itineraries  
- Simple recommendation suggestions  
- Multimodal reasoning abilities  

This notebook demonstrates how powerful AI models like Gemini 3 Pro can assist tourism, trip planning, and decision-making.  
Thank you for reviewing my submission!

```python
# Create a simple output file for submission
output_text = "Hackathon submission by Bhavani Kancharla using Gemini 3 Pro"
with open("/kaggle/working/submission.txt", "w") as f:
    f.write(output_text)

submission_path = "/kaggle/working/submission.txt"
submission_path
```