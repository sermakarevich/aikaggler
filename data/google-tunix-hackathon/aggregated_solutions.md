# google-tunix-hackathon: cross-solution summary

This competition focused on large language model fine-tuning and inference optimization, with top solutions prioritizing memory efficiency and training stability on TPU hardware. Winning approaches centered on custom backend patches, advanced attention mechanisms, and strict context length management to maximize throughput without out-of-memory errors. The prevailing strategy emphasized trading framework compatibility for aggressive memory savings, enabling larger batch sizes and extended context windows.

## Competition flows
- Raw tokenized dataset is loaded via HuggingFace, processed with prepacking and padding, fine-tuned using a patched Tunix loop with Cut-Cross Entropy and Flash Attention, and evaluated via vLLM inference benchmarks.

## Data reading
- HuggingFace dataset `G-reen/medium_set` (prepacked, tokenized, and padded)

## Data processing
- Prepacked, tokenized, and padded tokens; context length capped at 4096 due to sliding window attention constraints.

## Models
- Gemma-2-2B

## Frameworks used
- tunix
- Unsloth
- vLLM
- JAX
- Orbax
- safetensors

## Loss functions
- Cut-Cross Entropy (CCE)

## Insights
- Memory optimization enables significantly larger context lengths and batch sizes without OOM.
- JIT compilation time can be drastically reduced by rewriting the forward pass.
- TPU utilization improvements can offset the computational overhead of aggressive rematerialization.

## Critical findings
- Sliding window attention activation breaks the Flash Attention kernel, capping effective context length at 4096.
- Aggressive full model remat saves memory but degrades code stability and breaks compatibility with LoRA/DPO/GRPO.

## What did not work
- LoRA compatibility, Gemma3 1B support, DPO/GRPO training, and attempting to outperform Google's official instruction-tuned variant.

## Notable individual insights
- Unknown (Optimized Tunix with Cut-Cross Entropy and Flash Attention): Memory optimization enables significantly larger context lengths and batch sizes without OOM.
- Unknown (Optimized Tunix with Cut-Cross Entropy and Flash Attention): JIT compilation time can be drastically reduced by rewriting the forward pass.
- Unknown (Optimized Tunix with Cut-Cross Entropy and Flash Attention): TPU utilization improvements can offset the computational overhead of aggressive rematerialization.
- Unknown (Optimized Tunix with Cut-Cross Entropy and Flash Attention): Sliding window attention activation breaks the Flash Attention kernel, capping effective context length at 4096.
- Unknown (Optimized Tunix with Cut-Cross Entropy and Flash Attention): Aggressive full model remat saves memory but degrades code stability and breaks compatibility with LoRA/DPO/GRPO.

## Solutions indexed
- ? [[solutions/rank_xx_664755/solution|Optimized Tunix with Cut-Cross Entropy and Flash Attention]]

## GitHub links
- [Green0-0/tunix-cce](https://github.com/Green0-0/tunix-cce) _(solution)_ — from [[solutions/rank_xx_664755/solution|Optimized Tunix with Cut-Cross Entropy and Flash Attention]]
- [apple/ml-cross-entropy](https://github.com/apple/ml-cross-entropy) _(library)_ — from [[solutions/rank_xx_664755/solution|Optimized Tunix with Cut-Cross Entropy and Flash Attention]]
- [jax-ml/jax](https://github.com/jax-ml/jax) _(library)_ — from [[solutions/rank_xx_664755/solution|Optimized Tunix with Cut-Cross Entropy and Flash Attention]]
