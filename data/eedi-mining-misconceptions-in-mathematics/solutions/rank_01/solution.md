# 1st Place Solution Summary

- **Author:** Raja Biswas
- **Date:** 2024-12-13T02:41:33.267Z
- **Topic ID:** 551402
- **URL:** https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/discussion/551402

**GitHub links found:**
- https://github.com/anthropics/anthropic-cookbook

---

I really enjoyed working on this competition and learned a lot. Big thanks to the Kaggle community for sharing great ideas and engaging discussions!

My solution pipeline includes:
- An ensemble of retrievers: `intfloat/e5-mistral-7b-instruct`, `BAAI/bge-en-icl`, and `Qwen/Qwen2.5-14B`
- 3x Re-rankers: `Qwen/Qwen2.5-14B`, `Qwen/Qwen2.5-32B` and `Qwen/Qwen2.5-72B`

All models were fine-tuned using competition data (~1.8k examples) + synthetic data (~10k examples). 

## Retrievers
An ensemble of retrievers was used to identify `top 32-64 misconception candidates` for each QuestionId_Answer combination using dynamic thresholds. Specifically, I kept the top 32 retrieved candidates and included up to 32 additional candidates with similarity scores within `0.06` of the top candidate.

I selected the retrievers based on their `recall@32` scores instead of `MAP@25` -- these two metrics were often inversely correlated for my models. I skipped hard mining during training as it was not helpful for `recall@32`. One important detail was to avoid having multiple demonstrations of a given misconception in a single batch.

## Rerankers
I first used the `Qwen/Qwen2.5-14B` ranker to process the retrieved candidates and narrowed them down to `top 8`. For inference, I used `vllm` with `enable_prefix_caching=True` -- which significantly improved efficiency.

Next, the `Qwen/Qwen2.5-32B` ranker further reduced the candidates to the top 5, which were then passed to the `Qwen/Qwen2.5-72B` model for final ranking.

- Pointwise ranking: I used pointwise ranking for `Qwen/Qwen2.5-14B` and `Qwen/Qwen2.5-32B`.  These models processed one misconception candidate at a time in their context.  
- Listwise ranking:  `Qwen/Qwen2.5-72B` was listwise -- sees top 5 candidates at once in the context.

The following 4 aspects were important for performance and generalization:
- **Chain of Thought (CoT)**: I first prompted `Claude 3.5 Sonnet` to generate thoughts/rationale behind picking a wrong answer. Next, I fine-tuned the Qwen2.5 models with this data. During inference, I generated CoT from the finetuned models and provided them to the rerankers to help them with ranking. Essentially, this helps to distill reasoning from the claude model.

- **Few shot examples**: I wanted to retain/leverage in-context-learning capability of the LLMs. Hence, during training I optionally included a few demonstrations of the misconception it needs to classify. Here is an example:

```
<|im_start|>system
You are an expert in detecting grade-school level math misconceptions. Verify if the incorrect answer stems from the provided misconception.<|im_end|>
<|im_start|>user
Misconception: Thinks there are 100mm in a centimetre

Demos for the misconception:
Question: A \( 2 \) metre long rope has \( 0.5 \) centimetres chopped off the end.
What is the length of the rope that remains?
Give your answer in millimetres
Answer:\( 1995 \mathrm{~mm} \)
Misconception Answer: \( 1950 \mathrm{~mm} \)

--
Subject: Length Units
Topic: Convert between m and cm
Question: To convert 3.5 metres to centimetres, you should:
Correct Answer: Multiply by 100
Incorrect Answer: Divide by 100
Does the misconception (Thinks there are 100mm in a centimetre) lead to the incorrect answer? (Yes/No)<|im_end|>
<|im_start|>assistant
```
During inference I used 1-2 demos from the training set. It helped the 14b ranker the most.

- **Negative Ratio**: Showing the rankers a large number of negatives during training helped to improve performance. For each positive, I sampled 24 negatives.

- **Distillation/Pseudo Labelling**: I first fine-tuned two pointwise 72b models (`Qwen/Qwen2.5-Math-72B` and `Qwen/Qwen2.5-72B`) using competition data and ensembled them to pseudo label the synthetic examples. Competition data + pseudo labelled data were used to fine-tune 14b and 32b models.
 
## Synthetic Data
- **Grouped synthetic data generation**: I first created clusters of related misconceptions e.g. 
```
>> Assumes a sequence is linear
>> Thinks terms in linear sequence are in direct proportion
>> Does not know how to find the next term in a sequence
>> Describes position to term rule using term to term rule
>> Does not recognise that a linear sequence must increase or decrease by same amount
>> Uses only the first two terms of a sequence to work out a term-to-term rule
>> Does not know the definition of term or term-to-term rule
>> Only uses two terms of a sequence to work out a term-to-term rule
>> When asked for a specific term in a sequence just gives the next term
>> Does not notice all changes in a sequence of visual patterns
```
Then asked Claude 3.5 Sonnet to generate more examples. I included 5-8 relevant MCQs from the training set during the synthetic data generation. I have included the prompts in the comments.

--
## Links
- [Inference notebook](https://www.kaggle.com/code/conjuring92/eedi-a2-pipeline?scriptVersionId=211785645)

I will write a detailed post and publish my code + datasets soon!