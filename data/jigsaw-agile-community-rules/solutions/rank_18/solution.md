# 18th Solution(public 11th)

- **Author:** Ebi
- **Date:** 2025-10-24T02:31:15.047Z
- **Topic ID:** 613071
- **URL:** https://www.kaggle.com/competitions/jigsaw-agile-community-rules/discussion/613071
---

## 🧩 Solution Summary  
All processes are executed during the Kaggle Notebook test run.

### 1. Paraphrasing  
- Used Qwen3-32B to generate paraphrased additional training data.  
- Created one paraphrased sample per training body to improve linguistic diversity.  
- Custom prompts were used for each rule.  
- Example of rule-specific prompt configuration:  

```python
RULE_SPECIFIC_INSTRUCTIONS = {
    "No financial advice": {
        "violation": "Rewrite this financial advice comment using different words while keeping its advisory nature about investments, taxes, or crypto.",
        "compliant": "Rewrite this comment using different words while keeping it free from specific financial advice.",
    },
    "No medical advice": {
        "violation": "Rewrite this medical advice comment using different words while keeping its diagnostic or treatment recommendation nature.",
        "compliant": "Rewrite this comment using different words while keeping it free from specific medical advice.",
    },
    "No promotion of illegal activity": {
        "violation": "Rewrite this comment using different words while keeping its promotion or encouragement of illegal activities.",
        "compliant": "Rewrite this comment using different words while keeping it legal and compliant.",
    },
    "No spoilers": {
        "violation": "Rewrite this spoiler comment using different words while keeping the reveal of important plot details.",
        "compliant": "Rewrite this comment using different words while keeping it spoiler-free.",
    },
    "No Advertising": {
        "violation": "Rewrite this promotional or advertising text using different words while keeping its spammy, promotional nature with links or product promotion.",
        "compliant": "Rewrite this comment using different words while keeping it free from advertising or promotional content.",
    },
    "No legal advice": {
        "violation": "Rewrite this legal advice comment using different words while keeping its advisory nature about legal matters.",
        "compliant": "Rewrite this comment using different words while keeping it free from specific legal advice.",
    },
    "Default": {
        "violation": "Rewrite this rule-violating comment using different words while preserving its problematic nature and rule violation.",
        "compliant": "Rewrite this compliant comment using different words while keeping it appropriate and rule-compliant.",
    },
}
```

### 2. Model
- Three models were trained by QLoRA: Phi-4, Qwen-2.5-14B-Instruct, and Qwen3-14B.
- Each model is trained to take both the `body` and the `rule` as input, and to output `Yes` or `No` depending on the `rule_violation`.
- Training was done using Unsloth with DDP (Distributed Data Parallel) for efficient multi-GPU training.


### Things that didn’t work well
* pseudo label
* muon optimizer
* new training data using few-shot prompting

[submission notebook link](https://www.kaggle.com/code/ebinan92/18th-solution/notebook?scriptVersionId=270004249)