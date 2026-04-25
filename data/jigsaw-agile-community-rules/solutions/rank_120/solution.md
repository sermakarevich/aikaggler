# Silver Medal - 120th - RAG + ReRanker

- **Author:** Chris Deotte
- **Date:** 2025-10-24T18:01:52.930Z
- **Topic ID:** 613168
- **URL:** https://www.kaggle.com/competitions/jigsaw-agile-community-rules/discussion/613168
---

Thanks Kaggle, hosts, and community for a fun competition. I joined 10 days ago when MAP comp ended and I was tired of training LLM, so I decided to read public notebooks and see if I could combine them in a fun way. My final solution uses the top public notebooks as a RAG plus ReRanker pipeline shown below:

![](https://raw.githubusercontent.com/cdeotte/Kaggle_Images/refs/heads/main/Oct-2025/RAG-pipeline.png)

# Retriever
Using the existing embedding public notebook, I used it to create a new `test.csv` file. For each row, I found the top 2 positive examples most semantically  similar to `body` column and the top 2 negative examples most semantically similar to `body` column. The idea is that this will help the LLM better decide if the body violates the rule or not by seeing examples very similar to the body.

# LLM with 4x TTA
Using the new `test_with_retrieved_examples.csv`, I inferred the existing Llama3.2-3B-Instruct, DeBERTa-v3-base, and DistilRoBERTa-base notebooks with these new test rows which contained new examples. Also I employed 4x TTA. For each row we randomly pick 1 positive example and 1 negative example. Afterward, we infer the other positive example with other negative example. And lastly we infer the regular `test.csv` 2x.

# ReRanker
Inferring the public notebook Qwen2.5-32B takes time. We can accelerate inference by only inferring the 20% rows of test that our LLM is confused about. These predictions contain the majority of our LLM's mistakes. So we use the help of Qwen2.5-32B to improve these mistakes.

# AUC Metric
This competition's metric is AUC within each rule's predictions, then average these AUC. So the best way to ensemble multiple models is to first rank within each rule and then average the multiple models' test predictions:

    def rank_within_rule(df: pd.DataFrame, col: str, rule_col: str = "rule") -> pd.Series:
        g = df.groupby(rule_col, observed=True)
        ranks = g[col].rank(method="average")
        denom = g[col].transform("count") + 1
        return ranks / denom