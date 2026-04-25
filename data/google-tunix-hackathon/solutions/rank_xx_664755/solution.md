# Optimized Tunix with Cut-Cross Entropy and Flash Attention

- **Author:** Green
- **Date:** 2025-12-27T15:29:38.687Z
- **Topic ID:** 664755
- **URL:** https://www.kaggle.com/competitions/google-tunix-hackathon/discussion/664755

**GitHub links found:**
- https://github.com/Green0-0/tunix-cce
- https://github.com/apple/ml-cross-entropy
- https://github.com/jax-ml/jax

---

TLDR: I optimized the tunix code to implement CCE and flash attention, saving >15gb memory per tpu device, which enables 4x the context length and ~2x the effective batch size when performing FFT.

Code: https://github.com/Green0-0/tunix-cce

Notebook: https://www.kaggle.com/code/green000/tunix-cce-test

Testing Data: https://huggingface.co/datasets/G-reen/medium_set (prepacked, tokenized, and padded)

Demo Model: https://huggingface.co/G-reen/gemma-2-2b-ultrafine (safetensors for loading with vLLM, converted from orbax)

------
I was running some basic testing code with FFT, and I discovered that Tunix consumes a *lot* of memory. In particular, I was OOM'ing even with a context length of 2048, and not by a small margin.
I could switch to LoRA, but with such a small model and a big training set, you'd ideally want at least 1 parameter per token. The training dataset above is around 440 million tokens, thats a fourth of the model size.

Thus, I set out to patch the tunix training code for Gemma2.
In total, I've made the following changes:
1. Implement the gradient accumulation fix based on https://unsloth.ai/blog/gradient
2. Implement cut cross entropy based on https://github.com/apple/ml-cross-entropy
3. Implement flash attention with softcapping based on https://github.com/jax-ml/jax/pull/28099
4. Modify the remat for gemma2 to support full model remat, and rewrite the forward pass to solve some issues involving JIT compilation blowing up. This has a positive side effect of decreasing JIT compilation to 5 minutes, but the code itself is fragile.

These changes save over 15 gb of memory per device totaling over 120 gb saved with FSDP. In practice, depending on your hyperparameters it can be a lot more. This comes at the cost of slightly increased computational overhead, which is made up for by better TPU utilization (I use full model remat to fit a batch size of 64, but optimally you want a batch size of 16 with a less aggressive remat which I was just 3 gb short on). Aggregate t/s doesn't decrease, in practice you might see 2x speedups, but you will have to tune the backend for performance.

Unfortunately, this required a lot of vibe-coded edits, which has degraded the backend code and introduced more bugs. Additionally, two monkey patches are needed to override some versioning issues. You can no longer train with over a context length of 4096 (because sliding window attention activates and no FA kernel exists for this). LoRA may not work anymore. Gemma3 1b will no longer work, and DPO/GRPO may have additional bugs because I haven't implemented patches for those yet. I'll work on that this week, so a fix will come out soon.

To make sure that the code works, I have done two training runs, one with unsloth and one with my fork of tunix. The unsloth training run had less optimized hyperparameters and cross document contamination due to packing issues, so its slightly worse, but it is good enough as a baseline.

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F13757309%2F104910b6d2034546283c27fcb56fa140%2Funsloth.png?generation=1766849129882984&alt=media)
(unsloth)

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F13757309%2F2ee5ca65fe99392ae400d89aad7f585f%2Ftunix.png?generation=1766849142658016&alt=media)
(tunix, note that gradient accumulation was reduced by a third)

The losses generally follow the same values and shape, so training appears to be working.

I also ran benchmarks using vLLM:
![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F13757309%2F9a30fdfe5cdd88210d0ff8a4027d3ef2%2Fbenchmarks.png?generation=1766849226924085&alt=media)
(The worse of the two finetuned models corresponds to the unsloth variant)

There's clear improvement after training the base model, though admittedly the performance is not nearly as good as the instruction tuned variant (trying to beat google's instruct tune, in hindsight, was not a good idea).

The performance isn't very impressive, but if we look at the outputs its clearly learned something:
![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F13757309%2F407f90d621b160708b9b7cb1046f3870%2Foutputs.png?generation=1766849342745446&alt=media)

I've provided all the replication data above. Happy training!