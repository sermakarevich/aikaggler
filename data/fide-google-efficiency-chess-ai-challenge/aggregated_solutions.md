# fide-google-efficiency-chess-ai-challenge: cross-solution summary

This Kaggle competition challenged participants to optimize chess engine evaluation functions under strict memory and binary size constraints. Winning approaches predominantly focused on stripping memory-intensive components like NNUE, aggressively reducing hash and transposition table sizes, and integrating lightweight neural networks or custom parameter tuning into Hand-Crafted Evaluation (HCE) systems. Success ultimately depended on balancing architectural efficiency with precise evaluation weight optimization rather than relying on heavy deep learning pipelines.

## Competition flows
- Optimized Stockfish 16's memory footprint to fit a 5MB limit, trained a small 3-layer MLP on self-played chess positions using kaggle-environments, and integrated the network's output into the HCE evaluation function.
- Designed a custom hybrid CNN-dense NNUE architecture for chess position evaluation, trained it on a mix of Leela and Stockfish game data, integrated it into a modified chess engine with optimized memory and compilation flags, and submitted the resulting engine configuration.
- Selected and heavily memory-optimized Cfish, compressed its binary, adapted its search to Stockfish 16's HCE methods, tuned evaluation parameters via a custom SPSA script running concurrent cutechess-cli matches, and submitted two identical binaries to mitigate leaderboard variance.

## Data processing
- Self-play generated ~70,000 games with positions sampled during search at 1/8,192 probability and split into eight parts for efficient loading
- Attempted quantization-aware training (QAT)

## Features engineering
- 99 input features derived from the HCE evaluation state

## Models
- Stockfish 16 (HCE)
- 3-layer MLP
- NNUE
- CNN
- Dense network
- Stockfish 4
- Cfish
- Stockfish

## Frameworks used
- kaggle-environments
- C++
- Kaggle Notebooks

## Loss functions
- MSE

## Ensembling
- Submitted two separate agents (HCE-only and NN-enhanced) with a 0.5 scaling factor on the neural network's output.
- Submitted two identical binaries to account for leaderboard randomness.

## Notable individual insights
- rank 4 (4th place solution): Scaling the neural network's output by 0.5 yielded a ~30 Elo gain over HCE alone, whereas adding the full output provided less than 10 Elo.
- rank 4 (4th place solution): Removing libstdc++ dependencies significantly reduced timeout losses despite the program remaining functionally submittable with it linked.
- rank null (Niboshi's solution): Replacing the standard feature transformer with a hybrid CNN and weight-shared dense network successfully reduced input dimensionality from 768 to 96, though it introduced a heavy computational bottleneck.
- rank null (Niboshi's solution): The choice of training data source (Leela vs. Stockfish) had minimal impact on the final network performance.
- rank 9 (9th Place Solution): Compiling with -O3 instead of -Os actually improved search speed by 8-9% in Kaggle's CPU environment, contrary to typical embedded optimization advice.
- rank 9 (9th Place Solution): Cfish's glibc-based memory efficiency makes it superior to C++ Stockfish variants in single-core Kaggle environments.

## Solutions indexed
- #4 [[solutions/rank_04/solution|4th place solution]]
- #9 [[solutions/rank_09/solution|[9th Place Solution] Cfish + Simple SPSA]]
- ? [[solutions/rank_xx_563866/solution|Niboshi's solution]]

## GitHub links
- [Lgeu/kaggle-stockfish](https://github.com/Lgeu/kaggle-stockfish) _(solution)_ — from [[solutions/rank_04/solution|4th place solution]]
- [chettub/Niboshi](https://github.com/chettub/Niboshi) _(solution)_ — from [[solutions/rank_xx_563866/solution|Niboshi's solution]]
- [ymgaq/Cfish_kaggle](https://github.com/ymgaq/Cfish_kaggle) _(solution)_ — from [[solutions/rank_09/solution|[9th Place Solution] Cfish + Simple SPSA]]
- [syzygy1/Cfish](https://github.com/syzygy1/Cfish) _(reference)_ — from [[solutions/rank_09/solution|[9th Place Solution] Cfish + Simple SPSA]]
- [official-stockfish/fishtest](https://github.com/official-stockfish/fishtest) _(reference)_ — from [[solutions/rank_09/solution|[9th Place Solution] Cfish + Simple SPSA]]
- [AndyGrant/OpenBench](https://github.com/AndyGrant/OpenBench) _(reference)_ — from [[solutions/rank_09/solution|[9th Place Solution] Cfish + Simple SPSA]]
- [zamar/spsa](https://github.com/zamar/spsa) _(reference)_ — from [[solutions/rank_09/solution|[9th Place Solution] Cfish + Simple SPSA]]

## Papers cited
- [SPSA](http://www.jhuapl.edu/spsa/PDF-SPSA/Spall_Implementation_of_the_Simultaneous.PDF)
