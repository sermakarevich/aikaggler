# 19th place solution

- **Author:** WOOSUNG YOON
- **Date:** 2025-05-28T00:13:38.817Z
- **Topic ID:** 581012
- **URL:** https://www.kaggle.com/competitions/drawing-with-llms/discussion/581012
---

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F4808143%2F9cdd848ababf12881bc87a6a78b83a00%2Fimgs.png?generation=1748390712026601&alt=media)

In this competition, I generated images using a Diffusion Model and converted them into SVG format.


## 1. Bitmap to SVG

![Bitmap to SVG](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F4808143%2Fe437a744a9a2dd3425a6e90e9d583bbb%2Fsvg_1.png?generation=1748390798698476&alt=media)

The most impactful factor on the final score was the improvement of the algorithm that converted the given bitmap images into SVG format.

I compressed the SVG strings by using relative paths instead of absolute ones and grouped elements with the same color. Additionally, various techniques were applied to further improve the performance.



## 2. Two VQA Questions

![Two VQA Questions](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F4808143%2F0f0bef7268cb09b8e0228bbe79d8cf0f%2Fprompt_formmatter.png?generation=1748390970381466&alt=media)

We evaluated the generated images using two independent VQA (Visual Question Answering) questions that focused on the shape and color of the image.



## 3. Time Management

![Time Management](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F4808143%2F0e3b8992609d3c234489f9618a44f6cc%2Fprediction.png?generation=1748391026536653&alt=media)

I assumed a fixed time increment of 63 seconds per prediction.  
For each batch, we generated and evaluated three images. If the score of any image exceeded a predefined threshold, the process was terminated early.  
The remaining time from early termination was carried over and added to the time available for the next prediction.


---

In this project, we built a pipeline that generates images using a Diffusion Model and then converts them into SVG format. I experimented with several models, including PixART, and Sana.

The goal of this competition was to develop an algorithm that allows anyone to easily create magnificent illustrations.

Although I have many regrets, It was a competition that reminded me of my childhood days, learning language through picture books. :)
🤣 I learned a lot. Thanks. 