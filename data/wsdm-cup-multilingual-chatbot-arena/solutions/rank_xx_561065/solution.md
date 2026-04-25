# What I learned in WSDM

- **Author:** Cody_Null
- **Date:** 2025-02-04T00:06:56.953Z
- **Topic ID:** 561065
- **URL:** https://www.kaggle.com/competitions/wsdm-cup-multilingual-chatbot-arena/discussion/561065
---

I originally tackled this competition in hopes for redemption from LMSYS where my team fell just below the gold threshold in the final few days. We started strong but sadly by the end, at least in the public LB we have fallen back quite a bit but we are hopeful for better scores in private! I often like to share some of what I learned as an attempt to distract myself from the rankings haha. 

1. Inference speed improvements  - Improving LLM speed by use of calling scripts to run on separate GPUs simultaneously. Honestly never really thought about this as an option so good to know!

2. Learnings from LMSYS - Our issue in LMSYS was that we didnt use left truncation, knowledge distill, or have much wild strategy to improve the amount of max length or TTA we could preform. Though not all of this worked we tried all of it in this competition!

3. Custom classification heads - I had not previously used custom classification heads except for only a bit in LMSYS, in this comp I tried many different variations in attempt to improve score, tho nothing worked much better it was a learning experience. 

4. More transformer knowledge - As the competition first started I thought language would be very important in this comp as I expected some models to preform much better or worse depending on language so I looked into training on translated data, attempting to translate in test as well as a few other tricks like noting language during training with an identifier. Though nothing worked I learned a lot about transformer architecture and just how much a small change in training data can make.

5. Kaggle API - I used the kaggle API a lot this time around in order to stream data to datasets from the cloud. This was to avoid download times duplicating from system and back multiple times. I hadnt used it before so that was interesting and I will likely use it much more now!

6. Random knowledge about training LLMs and merging! - The different classification heads, lora options, data change impacts, adding in a second set of labels, impacts of resuming checkpoints and much more. No doubt I will be able to use much of this again. Merging I still think would be massively important though I didnt quite get the right formula, I had never used it before but learned all about Lerp, Slerp, weighted exchanges, Ties, Dares and Ties, balancing lora weights, and more! These can have a massive impact all while not needing to retrain models at all!

I am sure that there is more that I am forgetting but this is what I have for now, interested in hearing about some of your solutions. We hit an absolute wall and I have no idea how so many got 0.710!