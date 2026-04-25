# arc-prize-2025: cross-solution summary

The ARC Prize 2024 competition challenges participants to solve abstract reasoning puzzles, with top solutions demonstrating that hybrid architectures merging neural inference with symbolic program synthesis consistently outperform pure end-to-end models. Winning approaches universally emphasize test-time adaptation mechanisms, such as LoRA fine-tuning and active-inference loops, alongside aggressive data augmentation and ensemble voting to maximize coverage across diverse task distributions.

## Competition flows
- LLM-generated synthetic puzzles and external ARC datasets formatted into dialog templates, fine-tuned with LoRA, decoded via batch DFS with augmentation-based rescoring, and ensembled with Tiny Recursive Models
- Raw grid tasks processed through augmentation or colour remapping, passed to neural models or classical solvers, and combined via ensemble voting or probabilistic sampling
- Deep-learning-guided program synthesis system searching program space and adapting via test-time training with hindsight relabeling, with final submission based on transduction with test-time training adaptation

## Data reading
- Synthetic data generation pipeline producing Python code and grid representations
- External datasets from GitHub repositories (H-ARC, BARC, MINI-ARC, ConceptARC, RE-ARC, ARC-AGI-2, NVARC)
- Grids formatted into compact 16-token dialog-style templates

## Data processing
- Filtering and combining human puzzle descriptions
- LLM prompting for summary mixing and Python program generation
- Validation via unit tests and consistency checks
- Dataset augmentation (up to 256 samples per puzzle)
- Formatting into 16-token dialog templates
- LoRA test-time fine-tuning
- Flash Attention 2 optimization
- Unsloth framework integration
- AIRV loop (augment → infer → reverse → vote)
- Heavy data-augmentation (rot/flip/diag/colour swap)
- Recolor task remapping by pixel-count order
- ID restoration via –10/+10 trick
- Synthetic train grid generation for sparse supervision

## Features engineering
- Library of 120 hand-coded symmetry/flood-fill/pattern functions
- CNN and decision-tree modules for colour counts and masks
- Task metadata heuristics for function ordering

## Models
- Qwen3-2B
- Qwen3-4B
- Tiny Recursive Models (TRM)
- LoRA-tuned Qwen-0.5B
- 19-token custom Transformer
- LLaMA 3.1 8B
- 2020 C++ DSL solver
- Decision Tree
- CNN

## Frameworks used
- NeMo-Skills
- NeMo RL
- Megatron
- Unsloth
- Flash Attention 2
- PyTorch 2.2
- CUDA 12
- NumPy
- C++
- Python

## CV strategies
- Local validation on an augmented subset of 120 evaluation puzzles correlated with public leaderboard scores

## Ensembling
- Geometric mean ensemble of log-probabilities across augmentations with TRM and Qwen3 candidate combination
- Majority vote, probabilistic sampling proportional to unique solves, and fallback fusion with classical solvers

## Insights
- Scaling pretraining with high-quality synthetic data was the most critical factor for success.
- Test-time fine-tuning with LoRA and optimized decoding significantly boosted performance.
- Using identical augmentations across candidates ensures fair and comparable rescoring.
- TRM shows potential but requires substantial compute to match LLM capabilities in this domain.
- Active-Inference / reverse augmentation loop yields significant score gains
- Grid positional encoding outperforms classical sinusoidal encoding
- It is easier to delete bad answers than pick the best one, so keeping top-2 attempts is strategic
- Humans spot colour-irrelevance instantly, making cheap pre-processing gains possible
- Search combined with learning outperforms pure search approaches when constrained to the same number of predictions per task.

## Critical findings
- Batch DFS decoding introduced nondeterminism that degraded precision until batch-invariant operations were applied.
- Applying the exact same augmentations to all candidates made rescoring scores more reliable.
- Training TRM on evaluation data caused a score drop (7.5% vs 9.44%), highlighting overfitting risks.
- Ensembling TRM with Qwen3 4B yielded no improvement, indicating limited synergy under strict time limits.
- AIRV loop beats standard rotations & colour permutations
- Falling back to a 2020 C++ DSL solver for purely symbolic cases adds +14 pts
- Re-color pre-processor adds +2 pts over IceCuber but only works for single-test-pair tasks

## What did not work
- The batch DFS decoding ran ~17% slower in the Kaggle environment and was excluded from the final submission.
- Initial TRM submission with default parameters only achieved 2.08%.
- Ensembling TRM with Qwen3 4B produced no score gain (27.22% vs 27.22%).

## Notable individual insights
- NVARC solution: Scaling pretraining with high-quality synthetic data was the most critical factor for success.
- ARC 2024 Solutions and Key Takeaways: Falling back to a 2020 C++ DSL solver for purely symbolic cases adds +14 points.
- ARC 2024 Solutions and Key Takeaways: Active-Inference and reverse augmentation loops yield significant score gains over standard permutations.
- NVARC solution: Ensembling Tiny Recursive Models with Qwen3 4B yielded no improvement, indicating limited synergy under strict time limits.
- ARC 2024 Solutions and Key Takeaways: Grid positional encoding outperforms classical sinusoidal encoding.
- Exploring the combination of search and learn for the ARC25 challenge: Search combined with learning outperforms pure search approaches when constrained to the same number of predictions per task.
- ARC 2024 Solutions and Key Takeaways: Recolor pre-processing adds gains but only works for single-test-pair tasks.

## Solutions indexed
- ? [[solutions/rank_xx_651671/solution|NVARC solution]]
- ? [[solutions/rank_xx_575595/solution|ARC 2024 Solutions and Key Takeaways]]
- ? [[solutions/rank_xx_614521/solution|Exploring the combination of search and learn for the ARC25 challenge]]

## GitHub links
- [Le-Gris/h-arc](https://github.com/Le-Gris/h-arc) _(reference)_ — from [[solutions/rank_xx_651671/solution|NVARC solution]]
- [xu3kev/BARC](https://github.com/xu3kev/BARC) _(reference)_ — from [[solutions/rank_xx_651671/solution|NVARC solution]]
- [KSB21ST/MINI-ARC](https://github.com/KSB21ST/MINI-ARC) _(reference)_ — from [[solutions/rank_xx_651671/solution|NVARC solution]]
- [victorvikram/ConceptARC](https://github.com/victorvikram/ConceptARC) _(reference)_ — from [[solutions/rank_xx_651671/solution|NVARC solution]]
- [michaelhodel/re-arc](https://github.com/michaelhodel/re-arc) _(reference)_ — from [[solutions/rank_xx_651671/solution|NVARC solution]]
- [NVIDIA-NeMo/RL](https://github.com/NVIDIA-NeMo/RL) _(library)_ — from [[solutions/rank_xx_651671/solution|NVARC solution]]
- [1ytic/NVARC](https://github.com/1ytic/NVARC) _(solution)_ — from [[solutions/rank_xx_651671/solution|NVARC solution]]
- [thinking-machines-lab/batch_invariant_ops](https://github.com/thinking-machines-lab/batch_invariant_ops) _(library)_ — from [[solutions/rank_xx_651671/solution|NVARC solution]]
- [SamsungSAILMontreal/TinyRecursiveModels](https://github.com/SamsungSAILMontreal/TinyRecursiveModels) _(library)_ — from [[solutions/rank_xx_651671/solution|NVARC solution]]
- [zoenguyenramirez/arc-prize-2024](https://github.com/zoenguyenramirez/arc-prize-2024) _(solution)_ — from [[solutions/rank_xx_575595/solution|ARC 2024 Solutions and Key Takeaways]]
