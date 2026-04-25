# 255.9 Solution(General Topic)

- **Author:** ONODERA
- **Date:** 2024-11-27T02:45:47.877Z
- **Topic ID:** 548476
- **URL:** https://www.kaggle.com/competitions/santa-2024/discussion/548476
---

I'm using Simulated Annealing (SA) in this competition.
SA is a highly versatile optimization algorithm that has been widely used across various fields, including operations research, machine learning, and scheduling problems.
A greedy algorithm always accepts improvements and rejects non-improvements, whereas SA occasionally accepts non-improvements and sometimes rejects them as well.

# Why Simulated Annealing Is Widely Used
## Ability to Escape Local Optima
Unlike traditional greedy algorithms, SA can escape local optima by occasionally accepting worse solutions during the early stages of the process. This is controlled by a temperature parameter that decreases over time, allowing for a broad exploration of the solution space initially and fine-tuned adjustments later.

## Simplicity and Flexibility
SA is relatively simple to implement and can be adapted to a wide range of optimization problems. Its modular structure allows users to customize key components, such as the neighborhood function (how new solutions are generated) and the cooling schedule (how temperature decreases), to suit specific problems.

## Historical Success
Over decades, SA has been applied successfully to problems like traveling salesman problem(TSP), resource allocation, and even neural network training. Since this competition is similar to TSP, so SA is likely to be effective.

Lastly, here is an example of the heatmap showing the acceptance ratio of T and Δ in Greedy and SA.
<div style="display: flex; gap: 10px;">
    <img src="https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F317344%2F262d2876f1c73921875c01bd1336c408%2Foutput%20(1).png?generation=1732674451972588&alt=media" alt="Image A" style="width: 45%;">
    <img src="https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F317344%2F53dbd2ff8d8c273360b6370f5b8605cb%2Foutput%20(2).png?generation=1732674461990897&alt=media" alt="Image B" style="width: 45%;">
</div>


