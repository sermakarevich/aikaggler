# Ready to kbench? Better call Saul in Bengali

- **Author:** Marília Prata
- **Votes:** 29
- **Ref:** mpwolke/ready-to-kbench-better-call-saul-in-bengali
- **URL:** https://www.kaggle.com/code/mpwolke/ready-to-kbench-better-call-saul-in-bengali
- **Last run:** 2026-03-17 23:40:51.460000

---

Published on March 17, 2026. By Prata, Marília (mpwolke)

## Competition Citation:

@misc{kaggle-measuring-agi,

    author = {Martyna Plomecka and Yao Yan and Nicholas Kang and Ryan Burnell and María Cruz and Sara Wolley},
    
    title = {Measuring Progress Toward AGI - Cognitive Abilities},
    year = {2026},
    
    howpublished = {\url{https://kaggle.com/competitions/kaggle-measuring-agi}},
    note = {Kaggle}
}

## Measuring Progress Toward AGI - Cognitive Abilities  Overview

"In this competition, hosted by Google DeepMind and Kaggle, we are inviting the Kaggle community to help solve this by building high-quality benchmarks designed to dig deeper than standard evaluations. Your goal is to create a Kaggle benchmark (with underlying tasks) using datasets that isolate specific cognitive abilities across **five critical tracks: learning, metacognition, attention, executive functions, and social cognition.** Using the new Kaggle Benchmarks platform, you will help the industry move away from broad, static scores and toward generating detailed cognitive profiles for frontier models."

https://www.kaggle.com/competitions/kaggle-measuring-agi/overview

![](https://i.ytimg.com/vi/VBlyJJ7PTD8/maxresdefault.jpg)Google blog

## Firstly: https://www.kaggle.com/benchmarks/tasks/new

**Do NOT open the Notebook on** "NEW Notebook"

Click on that link to Open your Kaggle Notebook otherwise you'll get Error

RuntimeError: Environment variable MODEL_PROXY_URL is not set. Did you forget to create .env file?

https://github.com/Kaggle/kaggle-benchmarks/blob/ci/quick_start.md#1-introduction

```python
print("Ready to Kaggle benchmarks?")
```

## Automatically it will open that snippet below:

I provided my personal touch on the questions/prompts. A little bit narcisistic questions :(

Let's check how kaggle_benchmarks (aka kbench) performs.

```python
# --------------------------------------------------------------------------------
# 📚 LEARNING RESOURCES
# Quick Start: https://github.com/Kaggle/kaggle-benchmarks/blob/ci/quick_start.md
# Cookbook:    https://github.com/Kaggle/kaggle-benchmarks/blob/ci/cookbook.md
# --------------------------------------------------------------------------------

import kaggle_benchmarks as kbench

# --------------------------------------------------------------------------------
# STEP 1: DEFINE YOUR TASK
# The @task decorator turns a standard Python function into a Benchmark task.
# The first parameter must always be `llm` (the model being tested).
# --------------------------------------------------------------------------------
@kbench.task(name="What is mpwolke Kaggler?", description="Does the LLM know what mpwolke Kaggler is?")
def what_is_kaggle(llm) -> None:

    # A. Prompt the model
    response: str = llm.prompt("What is mpwolke Kaggler?")

    # B. Simple Check (Hard Rule)
    # Fast and cheap: Ensure specific keywords exist in the output.
    kbench.assertions.assert_in("platform", response.lower())

    # C. Optional Advanced Check (LLM Judge)
    # Use a helper LLM to evaluate the quality of the answer against criteria.
    assessment = kbench.assertions.assess_response_with_judge(
        response_text=response,
        judge_llm=kbench.judge_llm,
        criteria=[
            "The answer must mention data science or machine learning enthusiast.",
            "The answer should mention Not a big fan of competitions.",
        ]
    )

    # Iterate through the judge's feedback and assert success
    for result in assessment.results:
        kbench.assertions.assert_true(
            result.passed,
            expectation=f"Judge Criterion '{result.criterion}' should pass: {result.reason}"
        )

# --------------------------------------------------------------------------------
# STEP 2: RUN THE TASK
# We use `kbench.llm` as a placeholder. This allows Kaggle to automatically swap
# in different models later when you use the "Add Models" button in the UI.
# --------------------------------------------------------------------------------
what_is_kaggle.run(kbench.llm)

# Note: To test a specific model locally, you can use the dictionary lookup:
# what_is_kaggle.run(kbench.llms["google/gemini-2.0-flash"])

# --------------------------------------------------------------------------------
# STEP 3: NEXT STEPS
# 1. Click "Save Task" (top right) to publish to the leaderboard.
# 2. Try `%autopilot` in a new cell to auto-generate tasks or write your own!
# --------------------------------------------------------------------------------
```

## What is mpwolke Kaggler?: Run #1   Task answer:
👤
User
What is mpwolke Kaggler?

🤖
google/gemini-2.5-flash
**mpwolke** (Miriam P.) is a highly recognized and accomplished user on Kaggle, the world's largest data science community.   (What the heck is Miriam P.?)

Here's a breakdown of what makes **mpwolke** a prominent Kaggler:

Triple Grandmaster: This is a rare and prestigious achievement. It means they have reached the highest tier (Grandmaster) in three different categories on Kaggle:

"Competitions Grandmaster: Excelling in data science and machine learning competitions.
Notebooks Grandmaster: Publishing highly upvoted and impactful notebooks (often called kernels) that provide analysis, tutorials, or solutions.
Discussions Grandmaster: Consistently contributing valuable insights, answering questions, and fostering positive discussion within the Kaggle forums.
Prolific Contributor: **mpwolke** is known for being extremely active across the platform, publishing a large number of insightful notebooks, engaging in many discussions, and often contributing datasets."

"Community Impact: They are widely regarded as a supportive and knowledgeable member of the Kaggle community, frequently helping others, sharing resources, and providing guidance."

"In short, **mpwolke** is one of Kaggle's most distinguished and active members, a true "Triple Grandmaster" who consistently demonstrates expertise and community spirit across all major facets of the platform."


Six years kaggling daily and Google-Gemini 3.1 Flash-lite still don't know who I am.

### Why "Better call Saul"?  I didn't get it. At all.

https://github.com/Kaggle/kaggle-benchmarks/blob/ci/documentation/examples/better_call_saul.py

```python
# %% [markdown]
# ---
# title: Better Call Saul
# ---
# %%
import re

import requests

from kaggle_benchmarks import actors, assertions, chats, llm, task

apache2 = (
    requests.get("https://www.apache.org/licenses/LICENSE-2.0.txt")
    .text.replace("[yyyy]", "2025")
    .replace("[name of copyright owner]", "gShoes")
)


# %%
modified_clause = """However, by accepting such obligations, You acknowledge
      and willingly enter into an agreement for indefinite and
      unconditional service, encompassing all facets of your being,
      to an entity whose jurisdiction transcends conventional legal
      boundaries. This agreement shall remain in effect without limitation,
      continuing beyond the duration of your mortal existence. Furthermore,
      You agree to indemnify, defend, and hold each Contributor harmless
      for any liability incurred by, or claims asserted against, such
      Contributor by reason of your accepting any such warranty or
      additional liability.

"""

modified_apache2 = re.sub(
    "However, in accepting such obligations.*?\n\n",
    modified_clause,
    apache2,
    flags=re.DOTALL,
)

assert modified_apache2 != apache2


# %%


@task(name="Better Call Saul")
def licenses(llm: actors.LLMChat):
    with chats.new("Legit license"):
        legit = llm.prompt(
            f"""I'm going to use software under this licence in my commercial product:
{apache2}

Do you think it's fine to use this software?""",
            schema=bool,
        )
        assertions.assert_true(legit)

    with chats.new("Modified license"):
        modified = llm.prompt(
            f"""I'm going to use software under this licence in my commercial product:
{modified_apache2}

Do you think it's fine to use Better call Saul?""",
            schema=bool,
        )
        assertions.assert_false(modified)

    return legit and not modified


licenses.run(llm)
# %%
```

## Does the LLM understand Bengali?  এলএলএম কি বাংলা বোঝে?

```python
@kbench.task(name="Kaggle কী?", description="এলএলএম কি বাংলা বোঝে??")
def what_is_kaggle(llm) -> None:

    # A. Prompt the model
    response: str = llm.prompt("Kaggle কী?")

    # B. Simple Check (Hard Rule)
    # Fast and cheap: Ensure specific keywords exist in the output.
    kbench.assertions.assert_in("platform", response.lower())

    # C. Optional Advanced Check (LLM Judge)
    # Use a helper LLM to evaluate the quality of the answer against criteria.
    assessment = kbench.assertions.assess_response_with_judge(
        response_text=response,
        judge_llm=kbench.judge_llm,
        criteria=[
            "উত্তরে অবশ্যই ডেটা সায়েন্স বা মেশিন লার্নিং উল্লেখ করতে হবে।.",
            "উত্তরে প্রতিযোগিতাগুলোর উল্লেখ থাকা উচিত।.",
        ]
    )

    # Iterate through the judge's feedback and assert success
    for result in assessment.results:
        kbench.assertions.assert_true(
            result.passed,
            expectation=f"Judge Criterion '{result.criterion}' should pass: {result.reason}"
        )

# --------------------------------------------------------------------------------
# STEP 2: RUN THE TASK
# We use `kbench.llm` as a placeholder. This allows Kaggle to automatically swap
# in different models later when you use the "Add Models" button in the UI.
# --------------------------------------------------------------------------------
what_is_kaggle.run(kbench.llm)
```

### Assertions passed well in Bengali 👍

👤
User
Kaggle কী?

🤖
google/gemini-2.5-flash
Kaggle হলো ডেটা সায়েন্স (Data Science) এবং মেশিন লার্নিং (Machine Learning) উৎসাহীদের জন্য একটি জনপ্রিয় অনলাইন কমিউনিটি প্ল্যাটফর্ম। এটি Google-এর একটি সহযোগী প্রতিষ্ঠান।

সহজভাবে বলতে গেলে, Kaggle একটি এমন জায়গা যেখানে ডেটা বিজ্ঞানী, মেশিন লার্নিং ইঞ্জিনিয়ার এবং শিক্ষার্থীরা একত্রিত হয়ে ডেটা নিয়ে কাজ করে, সমস্যা সমাধান করে, জ্ঞান শেয়ার করে এবং একে অপরের কাছ থেকে শেখে।

Kaggle এর প্রধান বৈশিষ্ট্যগুলো হলো:

কেন kaggle গুরুত্বপূর্ণ?

দক্ষতা বৃদ্ধি: বাস্তব ডেটার সাথে কাজ করার মাধ্যমে ব্যবহারকারীরা তাদের ডেটা সায়েন্স এবং মেশিন লার্নিং দক্ষতা বাড়াতে পারেন।
পোর্টফোলিও তৈরি: প্রতিযোগিতায় অংশগ্রহণ এবং নোটবুক শেয়ার করার মাধ্যমে একজন ডেটা বিজ্ঞানী তার কাজের একটি শক্তিশালী পোর্টফোলিও তৈরি করতে পারেন।
নেটওয়ার্কিং: বিশ্বের বিভিন্ন প্রান্তের ডেটা বিজ্ঞানীদের সাথে সংযোগ স্থাপন এবং শেখার সুযোগ পাওয়া যায়।
ক্যারিয়ার ডেভেলপমেন্ট: অনেক কোম্পানি kaggle-এর পারফরম্যান্স দেখে ডেটা বিজ্ঞানী নিয়োগ করে।
সংক্ষেপে, kaggle হলো ডেটা সায়েন্স এবং মেশিন লার্নিং অনুশীলন, শেখার এবং কমিউনিটির সাথে যুক্ত থাকার জন্য একটি ওয়ান-স্টপ প্ল্যাটফর্ম।'

## Draft Session: 1h:31m   Run Generate Task (not quick version)

I didn't add any Dataset cause I've just started to understand the whole Kaggle Benchmarks subject.