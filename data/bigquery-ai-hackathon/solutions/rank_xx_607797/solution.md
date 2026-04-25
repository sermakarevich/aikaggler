# A Deep Dive into my Submission: The Airbnb AI Consultant 🤖🏠

- **Author:** Wafaa Alayoubi
- **Date:** 2025-09-16T07:26:46.010Z
- **Topic ID:** 607797
- **URL:** https://www.kaggle.com/competitions/bigquery-ai-hackathon/discussion/607797

**GitHub links found:**
- https://github.com/WafaaAlayoubi/The-Airbnb-AI-Consultant

---

Hello everyone,

I'm excited to share my submission for the BigQuery AI Hackathon! It's been an incredible learning journey, and I'd love to get your feedback on my project: **The Airbnb AI Consultant.**

### The Problem: Taking the Guesswork out of Hosting

I wanted to solve a real-world problem: Millions of Airbnb hosts struggle to know what makes a listing successful. My goal was to build an AI system that could provide them with clear, data-driven advice.

### My Solution: A Multimodal AI Pipeline

I built an end-to-end system that fuses multiple data types to understand the "secret formula" for a 5-star review. Here's the high-level architecture:

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F22494830%2Fbe828a2e19cbd9fc4ef1a8d59e9890dd%2Fdiagram.png?generation=1758007461269665&alt=media)

### The "Secret Formula": What the Model Learned

After a deep EDA and an iterative modeling process, I looked inside my final model to see what it learned. The results were fascinating! It turns out the most important features are **not** price or the number of bedrooms. They are the keywords in the description and the content of the photos.

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F22494830%2Fd431631a48d997821bc6cd2b867bb76d%2FFeatureImportance.jpeg?generation=1758007481655493&alt=media)

### The Grand Finale: The AI Scorecard

The final step was to use these insights to power a **Generative AI (Google's Gemini model)**. The system generates a final "AI Scorecard" for underperforming listings with concrete, actionable advice.

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F22494830%2F5b2108cc931f3dfd1c97ea810d9edf2f%2FaiScore.png?generation=1758007502077909&alt=media)

This was an amazing challenge, and I'm proud of the final result. I believe it's a powerful demonstration of how different AI services can be combined to create real business value.

---

I invite you all to check out the full, interactive notebook and the source code. Any feedback, questions, or upvotes are highly appreciated!

*   **🔗 Live Kaggle Notebook:** https://www.kaggle.com/code/wafaaalayoubi/the-airbnb-ai-consultant
*   **🛠️ GitHub Repository:** https://github.com/WafaaAlayoubi/The-Airbnb-AI-Consultant

Thank you for reading