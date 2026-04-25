# [placeholder lb0.321/0.500] My solution and experimental results

- **Author:** hengck23
- **Date:** 2025-03-07T10:34:36.857Z
- **Topic ID:** 566906
- **URL:** https://www.kaggle.com/competitions/stanford-rna-3d-folding/discussion/566906

**GitHub links found:**
- https://github.com/bytedance/Protenix
- https://github.com/Dao-AILab/flash-attention
- https://github.com/hypnopump/MiniFold
- https://github.com/facebookresearch/schedule_free
- https://github.com/Tan-group/FebRNA

---

... to be updated as experiment proceeds ... each week, we detail how to improve your lb score ...please come here often 

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F113660%2F60350c3075fb64f8b8e86240fdb1894a%2FSelection_096.png?generation=1742398747189706&alt=media)
**baseline code:**
1) 18-mar : lb 0.321
- https://www.kaggle.com/code/hengck23/lb0-286-simple-drfold-no-msa
- no MSA, instead a RNA language model to model evolutionary information
- overall, it is large RNA-lm and smaller structure downstream model
- use 5 out of 100 avaliable  smaller structure models (e.g. 16 mb) for current lb
- more structure models is better!!
- plan: 
1.energy scoring to select best model results, clustering? (see paper)
2.train model to convert from (N,C,P)frame + distance map --> C1 backbone
3.quantisation and fp16 to extend length to 800?

i think Drfold2 alone can get to lb0.40 on public test?

----

##Acknowledgement
"We extend our thanks to HP for providing the Z8 Fury-G5 Data Science Workstation, which empowered our deep learning experiments. The high computational power and large GPU memory enabled us to design our models swiftly."

##Hardware
GPU: 2x Nvidia Ada A6000 (Ampere), each with VRAM 48 GB
CPU: Intel® Xeon(R) w7-3455 CPU @ 2.5GHz, 24 cores, 48 threads
Memory: 256 GB RAM