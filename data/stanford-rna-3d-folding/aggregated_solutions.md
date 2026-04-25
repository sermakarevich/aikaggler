# stanford-rna-3d-folding: cross-solution summary

This competition focused on RNA 3D structure prediction, emphasizing the effective use of evolutionary conservation and template-based coordinate inheritance. Winning approaches primarily leveraged hybrid pipelines that dynamically route sequences between template-based modeling and optimized deep learning architectures like DRfold2, while alternative submissions explored lightweight, clustering-based classifiers to maximize accuracy under strict hardware constraints.

## Competition flows
- Sequence data is processed through an RNA language model for evolutionary features, passed to a downstream structure prediction model, and will be refined via energy scoring and clustering before submission.
- Raw CIF structural data is parsed with comprehensive nucleotide mapping, processed through a Template-Based Modeling pipeline for coordinate inheritance and gap filling, enhanced via GPU-accelerated DRfold2 optimization and selection modules, and combined in a hybrid pipeline with automatic fallback before submission.
- Raw RNA sequences are processed by computing pairwise TM-scores grouped by length to build a distance matrix, which is clustered to find representative structures; a Keras classifier then predicts cluster membership for inference, returning the 3D structure of the most similar representative per predicted cluster.

## Data processing
- Comprehensive nucleotide mapping (93 variants including modified bases)
- disorder-aware coordinate extraction
- geometric backbone reconstruction for gaps (maintaining ~5.9Å C1'-C1' distance, sinusoidal perturbations for compressed gaps, linear interpolation for normal gaps)
- confidence-based adaptive refinement constraints
- Data acquisition, unification, and selection
- Pairwise TM-score calculation grouped by sequence length to generate a distance matrix
- Clustering into n groups
- Selection of representative sequences per group

## Features engineering
- Distance matrix derived from pairwise TM-scores
- Cluster assignments used as classification targets
- Representative sequence selection per cluster

## Models
- RNA language model
- structure downstream model
- DRfold2
- Boltz-1
- Keras classifier

## Frameworks used
- PyTorch
- keras

## CV strategies
- :[

## Ensembling
- A hybrid pipeline combines Template-Based Modeling for shorter sequences or time exhaustion with DRfold2 for other sequences, featuring an automatic fallback to TBM upon DRfold2 failure.

## Insights
- Using an RNA language model instead of MSA effectively captures evolutionary information.
- Incorporating more available structure models consistently improves the leaderboard score.
- TM-score's normalization by structure length and robustness to local errors means prioritizing overall fold correctness over atomic-level precision yields better scores.
- RNA evolutionarily conserves 3D structure more than sequence, making template-based coordinate inheritance highly effective.
- DRfold2's original selection module can misrank models, necessitating double-precision calculations and optimized energy functions for reliable ranking.
- Grouping TM-score calculations by sequence length significantly reduces computational cost.
- Hardware constraints (CPU-only, limited RAM) necessitate prioritizing algorithmic efficiency over complex model architectures or external data.

## Critical findings
- DRfold2's native ranking sometimes places the best model at rank 5, highlighting a critical weakness in its scoring protocol.
- Scaling constraint strength as `0.8 × (1 - min(confidence, 0.8))` effectively adapts refinement intensity to template confidence, preventing over-correction on high-confidence templates.

## What did not work
- Incorporating PDB data or using data augmentation techniques for step 2 was considered but discarded due to hardware limitations.

## Notable individual insights
- rank null ([placeholder lb0.321/0.500] My solution and experimental results): Using an RNA language model instead of MSA effectively captures evolutionary information.
- rank 1 (1st Place Solution): TM-score's normalization by structure length and robustness to local errors means prioritizing overall fold correctness over atomic-level precision yields better scores.
- rank 1 (1st Place Solution): DRfold2's original selection module can misrank models, necessitating double-precision calculations and optimized energy functions for reliable ranking.
- rank 6 (Description of my solution for the competition / Descripción de mi solución para la competición): Grouping TM-score calculations by sequence length significantly reduces computational cost.
- rank 6 (Description of my solution for the competition / Descripción de mi solución para la competición): Hardware constraints (CPU-only, limited RAM) necessitate prioritizing algorithmic efficiency over complex model architectures or external data.

## Solutions indexed
- #1 [[solutions/rank_01/solution|1st Place Solution]]
- #6 [[solutions/rank_06/solution|Description of my solution for the competition / Descripción de mi solución para la competición]]
- ? [[solutions/rank_xx_566906/solution|[placeholder lb0.321/0.500] My solution and experimental results]]

## GitHub links
- [bytedance/Protenix](https://github.com/bytedance/Protenix) _(library)_ — from [[solutions/rank_xx_566906/solution|[placeholder lb0.321/0.500] My solution and experimental results]]
- [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) _(library)_ — from [[solutions/rank_xx_566906/solution|[placeholder lb0.321/0.500] My solution and experimental results]]
- [hypnopump/MiniFold](https://github.com/hypnopump/MiniFold) _(library)_ — from [[solutions/rank_xx_566906/solution|[placeholder lb0.321/0.500] My solution and experimental results]]
- [facebookresearch/schedule_free](https://github.com/facebookresearch/schedule_free) _(library)_ — from [[solutions/rank_xx_566906/solution|[placeholder lb0.321/0.500] My solution and experimental results]]
- [Tan-group/FebRNA](https://github.com/Tan-group/FebRNA) _(library)_ — from [[solutions/rank_xx_566906/solution|[placeholder lb0.321/0.500] My solution and experimental results]]
