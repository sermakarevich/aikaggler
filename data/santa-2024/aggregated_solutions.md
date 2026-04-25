# santa-2024: cross-solution summary

This competition centered on optimizing text sequences through combinatorial permutation rather than traditional machine learning. Winning approaches universally leveraged advanced local search algorithms like Simulated Annealing and Iterated Local Search, emphasizing strategic initialization, custom neighborhood operations, and positional bias mitigation to navigate complex solution spaces.

## Competition flows
- The author implemented a Simulated Annealing optimization routine to solve the competition's core problem, leveraging a decaying temperature schedule and customizable neighborhood generation to navigate the solution space.
- Raw word sequences are initialized via strategic sorting of stop and non-stop words, then optimized through a multi-start Iterated Local Search pipeline combining restricted k-opt moves and custom subsequence-shuffling kicks to maximize the scoring function.
- The pipeline begins with random shuffles of original text words, runs parallel simulated annealing with GPU-batched metric evaluation, applies specialized sequence modification moves, normalizes positional bias via discounted perplexity, and outputs the globally best text permutation.
- The pipeline starts by sharding the 100! permutation space based on alphabetical order relationships, then applies Simulated Annealing to iteratively assign letters to optimal blocks, stacking alphabetical sorting methods to produce the final submission.
- The pipeline starts from a pre-generated public submission, applies simulated annealing with custom neighbor generation operators to iteratively permute word sequences, and outputs the optimized permutation as the final submission.
- The pipeline applies an Iterated Local Search algorithm with beam search and local perturbation operations (insert, shuffle, swap, move) to iteratively optimize word sequences until convergence, followed by strategic word ordering adjustments based on empirical loss analysis.
- Raw text samples were initialized with a custom two-block alphabetical structure, then optimized via multi-phase Simulated Annealing using word swap and extract/insert operations to minimize perplexity before final submission.
- The pipeline applies modified Simulated Annealing with custom text transformation operations and dynamic scoring weights to iteratively optimize text sequences, using perplexity-based word importance and top-k selection strategies to converge on high-scoring outputs.

## Data processing
- Constructed starting sentences using the pattern [stopwords (without and)] + [alphabetically sorted block 1] + [and and and] + [alphabetically sorted block 2], with blocks sampled randomly.

## Loss functions
- perplexity

## Ensembling
- The solution stacks multiple alphabetical sorting methods, using Simulated Annealing to optimize letter block assignments, though specific post-processing or threshold tuning is not detailed.
- No ensembling was applied; the final submission consists of independently optimized permutations for IDs 3, 4, and 5.

## Notable individual insights
- rank 2 (2nd place solution (@zaburo's part)): Custom kicks that randomly separate adjacent word groups or transfer words between structural blocks outperform standard TSP escape mechanisms for this problem.
- rank 5 (5th Solution - CPMP part): Modifying words near the start of a sequence has a much higher influence on the score than modifications near the end, causing local search to get stuck.
- rank 10 (Santa24 10th place - My Takeaways): Basic neighborhood operations (word swap, extract/insert) are sufficient and more compute-efficient than complex neighbor generation strategies.
- rank 3 (3rd place solution): Dynamically weighting text transformation operations based on their historical performance significantly improves candidate selection quality.
- rank 11 (11th place solution): Structural properties of the solution space, such as alphabetical blocks and multi-word semantic units, dictate which search operators will succeed.
- rank 4 (4th Place Solution): Sorting content words alphabetically and splitting them into segments further decreases the final loss.

## Solutions indexed
- #1 [[solutions/rank_01/solution|1st place solution]]
- #2 [[solutions/rank_02/solution|2nd place solution (@zaburo's part)]]
- #3 [[solutions/rank_03/solution|3rd place solution]]
- #4 [[solutions/rank_04/solution|4th Place Solution]]
- #5 [[solutions/rank_05/solution|5th Solution - CPMP part]]
- #10 [[solutions/rank_10/solution|Santa24 10th place - My Takeaways]]
- #11 [[solutions/rank_11/solution|11th place solution]]
- ? [[solutions/rank_xx_548476/solution|255.9 Solution(General Topic)]]
- ? [[solutions/rank_xx_559339/solution|Explanation of Sample 5 Team Solution]]

## GitHub links
- [Lgeu/santa2024](https://github.com/Lgeu/santa2024) _(solution)_ — from [[solutions/rank_01/solution|1st place solution]]
- [EgorTrushin/Kaggle-Competitions](https://github.com/EgorTrushin/Kaggle-Competitions) _(solution)_ — from [[solutions/rank_10/solution|Santa24 10th place - My Takeaways]]

## Papers cited
- [Doubly Rooted Ejection Chain ATSP](https://leeds-faculty.colorado.edu/glover/469%20-Traveling%20Salesman%20-%20Doubly%20Rooted%20Ejection%20Chain%20ATSP_Final.pdf)
