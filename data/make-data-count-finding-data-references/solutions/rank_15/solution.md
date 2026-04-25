# 15th place solution

- **Author:** guo dashuai
- **Date:** 2025-09-10T02:31:38.937Z
- **Topic ID:** 606751
- **URL:** https://www.kaggle.com/competitions/make-data-count-finding-data-references/discussion/606751
---

Thank you to the organizers and Kaggle for your support. Although this competition was somewhat unlucky for me, I still learned a great deal.

Here’s a brief summary of my approach:
I downloaded the DataCite corpus and Europe PMC data from the competition homepage, storing paper DOIs and their corresponding citations in two CSV files.
I then downloaded the full metadata from DataCite, as well as all metadata for BioSample and GSE (GSE might have been the main reason for my shake-down).
From these datasets, I extracted all citation IDs and author names and other matadata.

Finally, I used Grobid to extract the author list and title for each paper and determined primary vs. secondary authors , using an LLM in a multi-stage decision process.

Originally, this method performed well on both CV and LB. Unfortunately, I may have made an error in determining the priority of registration numbers—perhaps marking all SAMN as primary or excluding GSE entirely could have won me the competition. However, without visibility into the private leaderboard, I assumed that using author names (similar to DOI matching) would help me mitigate risks—but it seems I was mistaken.

Although my luck needs improvement, I’ve gained valuable experience. Wishing everyone continued growth and success!