# Explanation of Sample 5 Team Solution

- **Author:** WOOSUNG YOON
- **Date:** 2025-01-24T18:50:06.127Z
- **Topic ID:** 559339
- **URL:** https://www.kaggle.com/competitions/santa-2024/discussion/559339
---

We focused on solving Sample 5 within the constrained environment of Kaggle Notebooks. 
- https://www.kaggle.com/code/woosungyoon/alphabetical-sample-5

![alpahbet](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F4808143%2Fad58ec36428594c517eca957253fe7c9%2Fimg.png?generation=1737744795144579&alt=media)


We utilized the fact that alphabetical sorting in Sample 5 easily achieves a score of 44.0 (with stopwords)
and decided to build our approach around stacking such alphabetical sorting methods.

**- Note: This method has its limitations, so I hope we can discuss together and find a way to overcome them.**

---

The solution space is as vast as 100!.  

However, if we determine the order relationships among the alphabets, we can simplify this space.  

![img3](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F4808143%2F6755e32ed1785123ddde9c64bd0f3101%2Fimg3.png?generation=1737745587719250&alt=media)

For example, by sharding the entire space based on an order such as `a -> b -> c -> ...`, 

---

we only need to consider permutations within each letter block.  

Then, through Simulated Annealing, each letter moves to its appropriate block, allowing the advantages of alphabetical sorting to be fully utilized.

---


