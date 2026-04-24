# [24th] Post-training Qwen2.5 32B and 72B with Gemini OCR/published texts pairs

- **Author:** Geremie Yeo
- **Date:** 2026-03-24T00:57:19.987Z
- **Topic ID:** 684189
- **URL:** https://www.kaggle.com/competitions/deep-past-initiative-machine-translation/discussion/684189

**GitHub links found:**
- https://github.com/bogoconic1/Qgentic-AI

---

# Solution Overview

Notebook: https://www.kaggle.com/code/yeoyunsianggeremie/deep-past-inference

## Introduction

Team members: @yeoyunsianggeremie @andrevma18

We're both grinding for our startup, working 70-75 hour weeks, which left us almost no time to compete seriously. As such, we did the high-level thinking and solution-design ourselves (how to clean, how to generate data, etc) then distilled it into structured Markdown files for [Qgentic-AI](https://github.com/bogoconic1/Qgentic-AI) which is an autonomous Kaggle agent I have been building for the past 6 months. Qgentic-AI would run experiments iteratively throughout the week. We only spent 6 days grinding, which is very low given how unique and difficult this competition is.

I almost said "no" to joining this competition - as I felt I wouldn't have enough time. But I realized, working under this intense constraint is something I am trying to actively get better at, and this would be a good practice!


## Models used

Qwen2.5-32B, Qwen2.5-72B and ByT5-Base

## Datasets

Published texts: Asked Gemini to split the data into sentences and generated translations for each of them
PDFs: Fed each page into Gemini to extract transliteration/translation pairs 

Yup, you heard it right, we did **not use** the competition train data or attempt to clean it (I caught Qgentic-AI running in circles to decide whether to include it or not, and decided it was confusing so went against it)

In total, we had about 55,000 transliteration/translation pairs after doing some manual cleaning and LLM-as-a-judge

## Training 

We post-trained Qwen2.5-32B and Qwen2.5-72B for 3 epochs on 8x B200 with LoRA rank 64 and alpha 128
ByT5 was trained for 15 epochs at a learning rate of 3e-4

The parameters were tuned by Qgentic-AI (such a huge time saver 😁)

We trained our models on ENG -> AKK and AKK -> ENG (so a total of ~110,000 training pairs)

## Validation

We used a pertubed version of the train.csv removing all bad samples where the training and validation lengths are way off. There is still data leakage so we do not base 100% of our selections based on that. The CV is around 43 for our best models.

## Preprocessing

Mostly based on the host's [discussion](https://www.kaggle.com/competitions/deep-past-initiative-machine-translation/discussion/678899) - fixed a few bugs like 5/12 shekels -> 2/3 shekel 15 grains

## Postprocessing

Removed repetitions using the base Qwen2.5-72B model using the prompt
```
DEDUP_SYSTEM = "You are a text editor. The user will give you an English translation that may contain repeated phrases or sentences. Remove any obvious repetitions while preserving the rest of the text exactly as-is. Do NOT rephrase, summarize, or add anything. Return ONLY the cleaned text."
```

## Failed Ideas

- Any ByT5 model >= Large. We haven't investigated why.
- MBR
- Using competition training data
- Translating Akkadian -> Arabic -> English
- Training a judge model to grade the quality of the translations

## Code for Preprocessing and Postprocessing

```
import re
import pandas as pd


def replace_roman_months(text):
    if pd.isna(text): return ""
    text = str(text)
    romans = {
        "XII": "12", "XI": "11", "IX": "9", "IV": "4",
        "VIII": "8", "VII": "7", "III": "3", "VI": "6",
        "X": "10", "V": "5", "II": "2", "I": "1"
    }
    for r, num in romans.items():
        text = re.sub(r'(?i)\bmonth\s+' + r + r'\b', 'month ' + num, text)
    return text

_FRAC_MAP = {
    0.5: "½", 0.3333: "⅓", 0.6666: "⅔", 0.6667: "⅔",
    0.25: "¼", 0.75: "¾",
    0.1666: "⅙", 0.1667: "⅙", 0.8333: "⅚",
    0.125: "⅛", 0.375: "⅜", 0.625: "⅝", 0.875: "⅞",
    0.2: "⅕", 0.4: "⅖", 0.6: "⅗", 0.8: "⅘",
    0.333: "⅓",
}

_ASCII_FRAC_MAP = {
    "1/2": "½", "1/3": "⅓", "2/3": "⅔",
    "1/4": "¼", "3/4": "¾",
    "1/5": "⅕", "2/5": "⅖", "3/5": "⅗", "4/5": "⅘",
    "1/6": "⅙", "5/6": "⅚",
    "1/8": "⅛", "3/8": "⅜", "5/8": "⅝", "7/8": "⅞",
}

def _decimals_to_unicode(text):
    def _replace(m):
        val = float(m.group())
        integer = int(val)
        frac_part = round(val - integer, 4)
        if frac_part == 0:
            return m.group()
        unicode_frac = _FRAC_MAP.get(frac_part)
        if unicode_frac is None:
            return m.group()
        if integer == 0:
            return unicode_frac
        return f"{integer} {unicode_frac}"
    return re.sub(r'(?<!\d)\d+\.\d+(?!\d)', _replace, text)

def _ascii_fracs_to_unicode(text):
    def _replace(m):
        key = f"{m.group(1)}/{m.group(2)}"
        return _ASCII_FRAC_MAP.get(key, m.group())
    return re.sub(r'\b(\d+)\s*/\s*(\d+)\b', _replace, text)

def _convert_twelfth_shekels(text):
    def _replace_11_12(m):
        n = int(m.group(1))
        return f"{n + 1} shekels less 15 grains"
    text = re.sub(r'(\d+)\s+11\s*/\s*12\s+shekels?', _replace_11_12, text)
    twelfth_map = [
        ("7 / 12",  "½ shekel 15 grains"),
        ("7/12",    "½ shekel 15 grains"),
        ("5 / 12",  "⅓ shekel 15 grains"),
        ("5/12",    "⅓ shekel 15 grains"),
        ("1 / 12",  "15 grains"),
        ("1/12",    "15 grains"),
    ]
    for num, replacement in twelfth_map:
        text = re.sub(
            re.escape(num) + r'\s*(?:\(shekel\)|shekels?)',
            replacement,
            text
        )
    return text

def clean_transliteration(text):
    if pd.isna(text): return ""
    text = str(text)
    subscript_map = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
    text = text.translate(subscript_map)
    text = text.replace("Ḫ", "H").replace("ḫ", "h")
    text = text.replace("KÙ.B.", "KÙ.BABBAR")
    for var in [r'\[x\]', r'\[\.\.\.\.\.\.\]', r'\[\.\.\. \.\.\.\]', r'\.\.\.', r'\(break\)', r'\(large break\)', r'\(n broken lines\)', r'xₓ']:
        text = re.sub(var, ' <gap> ', text)
    text = _decimals_to_unicode(text)
    text = _ascii_fracs_to_unicode(text)
    text = replace_roman_months(text)
    text = re.sub(r'(<\s*gap\s*>\s*)+', '<gap> ', text)
    return " ".join(text.split())

def clean_translation(text):
    if pd.isna(text): return ""
    text = str(text)
    for var in [r'\[x\]', r'\[\.\.\.\.\.\.\]', r'\[\.\.\. \.\.\.\]', r'\.\.\.', r'\(break\)', r'\(large break\)', r'\(n broken lines\)', r'xₓ']:
        text = re.sub(var, ' <gap> ', text)
    for r_mark in [r'\bfem\.', r'\bsing\.', r'\bpl\.', r'\bplural\b', r'\(\?\)', r'\.\.,', r'\bxx\b', r'\bx\b', r'<<\s*>>', r'<\s*>', r'\[\s*\]', r'˹\s*˺']:
        text = re.sub(r_mark, '', text)
    text = text.replace("PN", "<gap>")
    text = re.sub(r'(?<=\s)-gold\b', 'pašallum gold', text)
    text = re.sub(r'(?<=\s)-tax\b', 'šadduātum tax', text)
    text = re.sub(r'(?<=\s)-textiles\b', 'kutānum textiles', text)
    text = _convert_twelfth_shekels(text)
    text = _decimals_to_unicode(text)
    text = _ascii_fracs_to_unicode(text)
    text = replace_roman_months(text)
    text = re.sub(r'(<\s*gap\s*>\s*)+', '<gap> ', text)
    return " ".join(text.split())

def postprocess(text, transliteration=""):
    text = clean_translation(text)
    text = re.sub(r'(\w)šadduātum tax', r'\1-tax', text)
    text = re.sub(r'(\w)pašallum gold', r'\1-gold', text)
    text = re.sub(r'(\w)kutānum textiles', r'\1-textiles', text)
    text = text.replace('…', '<gap>')
    text = re.sub(r'\[gap\]', '<gap>', text)
    text = re.sub(r'(<\s*gap\s*>\s*)+', '<gap> ', text)
    if '<gap>' not in transliteration:
        text = text.replace('<gap>', '')
    text = " ".join(text.split())
    if len(text) == 0:
        text = "<gap>"
    return text
```

