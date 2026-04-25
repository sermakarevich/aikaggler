# 7th Place Solution

- **Author:** Raja Biswas
- **Date:** 2025-03-11T03:02:33.907Z
- **Topic ID:** 567589
- **URL:** https://www.kaggle.com/competitions/wsdm-cup-multilingual-chatbot-arena/discussion/567589
---

Thanks to Kaggle, the competition hosts, and the participants for making this an exciting competition! We are pleased to see the winning solutions generalize well to the test set. Congratulations to all the winners! Trushant and I had a lot of fun building our solution over a 3 week sprint.


## Inference Pipeline
We implemented an uncertainty-based ensemble pipeline as shown in the figure below:
![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F2125251%2F20c27d4aacda686315f74dcdab57a78e%2Finference_pipeline.png?generation=1741661694322639&alt=media)

- First, a fine-tuned `gemma-2-9b-it` model was used to score all test examples
- Next, uncertainty scores were calculated as closeness to the decision boundary (`uncertainty = 1-|p-0.5|`)
- Top 50% uncertain examples were identified (i.e. those with probabilities closest to 0.5) and scored again using with a larger model.

Specifically,
- For top 15% uncertain examples, we used a fine-tuned `Qwen2.5-32B-Instruct` (selected submission 1) or `Mistral-Small-24B-Instruct-2501` (selected submission 2) model (xlarge)
- For next 35% examples, we used `Qwen2.5-14B-Instruct` (large) 
- For remaining examples, we used scores directly from the `gemma-2-9b-it` model (base)

The final score was computed as an ensemble:
- Top 15% uncertain examples: `Score = 1.0 x base + 1.5 x xlarge`
- Next 35% examples: `Score = 1.0 x base + 1.0 x large`
- Remaining examples: `Score = 1.0 x base`

We used vllm for inference. The inference time estimates for 512 examples are shown below together with max sequence length:

```table
| Model                                  | Max Length | Time (mins) |
|----------------------------------------|------------|-------------|
| gemma-2-9b-it (fp16)                   | 4096       | 6.5         |
| Qwen2.5-14B-Instruct (fp16)            | 3072       | 10.5        |
| Qwen2.5-32B-Instruct (gptq)            | 2048       | 20.0        |
| Mistral-Small-24B-Instruct-2501 (gptq) | 2048       | 16.4        |
```

## Data
We used the following data sources for training:

### Labelled Datasets
- [WSDM Cup - Multilingual Chatbot Arena Competition Data](https://www.kaggle.com/competitions/wsdm-cup-multilingual-chatbot-arena/data) (45k)
- [LMSYS - Chatbot Arena Human Preference Predictions Competition Data](https://www.kaggle.com/c/lmsys-chatbot-arena-human-preference-predictions) (30k)
- [lmarena-ai/PPE-Human-Preference-V1](https://huggingface.co/datasets/lmarena-ai/PPE-Human-Preference-V1) (10k)

### Unlabelled Datasets
- [wsdm - open models - nbroad](https://www.kaggle.com/datasets/nbroad/wsdm-open-models-nbroad/data) (Dataset from Nicholas - Sampled 6k)
- [Skywork/Skywork-Reward-Preference-80K-v0.2](https://huggingface.co/datasets/Skywork/Skywork-Reward-Preference-80K-v0.2) (Sampled 25k)
- [allenai/tulu-3-sft-olmo-2-mixture](https://huggingface.co/datasets/allenai/tulu-3-sft-olmo-2-mixture) (Sampled 30k)
- Team generated preference pairs (100k examples)
    - Prompts from [allenai/WildChat-1M](https://huggingface.co/datasets/allenai/WildChat-1M) and [lmsys/lmsys-chat-1m](https://huggingface.co/datasets/lmsys/lmsys-chat-1m) datasets
    - Cluster based sampling to roughly match the distribution from [Arena Explorer](https://lmarena.ai/?explore)

## Models
We fine-tuned models from Gemma, Llama, Qwen and Mistral families, which achieved the following scores:

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F2125251%2Fc38d8dacd5ffc2e52cf02981900c24f0%2Fmodel_scores.png?generation=1741661834851185&alt=media)

- We first fine-tuned `Qwen2.5-72B-Instruct` and `Llama-3.1-Nemotron-70B-Instruct` on the competition subset and used them to generate pseudo labels for the unlabeled datasets.
- For Gemma models, we directly used "A" and "B" token logits to compute the preference probability. These models did not have a reward model head. They were trained with a combination of language modeling and reward modeling losses. (Reference: [Regularizing Hidden States Enables Learning Generalizable Reward Model for LLMs](https://arxiv.org/abs/2406.10216))
- The remaining models were trained using the standard `AutoModelForSequenceClassification` architecture with a linear head.

## Generalization
We considered the following aspects in an attempt to improve generalization:
- Incorporated larger LLMs in the inference pipeline to handle uncertain examples. Larger LLMs tend to generalize better.
- Ensemble of different modelling approaches and model families (logits-based and AutoModelForSequenceClassification).
- Kept the preference modelling head simple (one linear layer).
- Distillation from 70B+ LLMs using pseudo labels on a large unlabeled dataset containing diverse prompts.

## Links
- [Inference Notebook](https://www.kaggle.com/code/conjuring92/wsdm-7th-place-solution)

We look forward to reading and learning from your solutions! Special thanks to @trushk for the perfect collaboration and teamwork!