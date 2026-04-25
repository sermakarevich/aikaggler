# I published Retriever Reranker Baseline(LB: 0.189), Fine-Tuning BGE Baseline(LB: 0.246)

- **Author:** sinchir0
- **Date:** 2024-09-16T02:18:13.870Z
- **Topic ID:** 534317
- **URL:** https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/discussion/534317
---

# Retriever Reranker Baseline.

1. [Train Tfidf Retriever](https://www.kaggle.com/code/sinchir0/retriever-tfidf-reranker-deberta-1-trn-ret) (Recall: 0.4530, CV: 0.1378, LB: 0.128)

2. [Train DeBERTa Reranker](https://www.kaggle.com/code/sinchir0/retriever-tfidf-reranker-deberta-2-trn-rerank)(CV: 0.1740)

3. [Infer by Tfidf Retriver And DeBERTa Reranker](https://www.kaggle.com/code/sinchir0/retriever-tfidf-reranker-deberta-3-infer)(LB: 0.189)

# Fine-Tuning BGE Baseline

1. [Fine-Tuning BGE Train](https://www.kaggle.com/code/sinchir0/fine-tuning-bge-train)

2. [Fine-Tuning BGE Infer](https://www.kaggle.com/code/sinchir0/fine-tuning-bge-infer) (LB: 0.246)

- make 25 retrieval data by `bge-large-en-v1.5`
- Fine-tuning `bge-large-en-v1.5` by retrieval data
  - anchor: `ConstructName` + `SubjectName` + `QuestionText` + `Answer[A-D]Text`
  - positive: Correct MisconceptionName
  - negative: Wrong MisconceptionName

# Next Step

You can aim to improve your score by the following methods:

- Modify the Retriever and Reranker models. In particular, using an LLM as a Reranker may be effective in this competition.
- Use a fine-tuned BGE as a Retriever.