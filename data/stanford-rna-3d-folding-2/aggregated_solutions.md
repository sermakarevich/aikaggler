# stanford-rna-3d-folding-2: cross-solution summary

This RNA 3D structure prediction competition rewarded structural diversity and length-adaptive modeling over reliance on any single foundation model. Winning approaches consistently combined Template-Based Modeling (TBM) with diffusion-based architectures (Protenix, RNAPro) and secondary structure priors, dynamically allocating inference resources based on sequence length and template availability to maximize the best-of-5 scoring metric.

## Key challenges
- Stochastic diffusion variance causing score instability across reruns
- Template scarcity and low similarity in test targets
- Memory and compute constraints for long sequences (>1000 nt)
- Balancing inference time versus accuracy across varying sequence lengths
- Non-deterministic CUDA backends masking algorithmic improvements
- Public leaderboard skew toward TBM masking model-specific gains
- Handling multi-chain and multi-component inputs (proteins, DNA, ligands)
- Preventing diffusion model overcorrection when fed raw templates
- Alignment penalties for chain misalignments versus missing chains

## Models
- Protenix
- RNAPro
- RibonanzaNet2
- Template-Based Modeling (TBM)
- Boltz2
- DRFold2
- BPP-Protenix
- RNAdegformer
- LightGBM
- ViennaRNA
- Parasail

## CV strategies
- Holdout validation set of 33 PDBs (Kaggle official validation + collected PDBs from 2025-12-03 to 2026-02-18)
- Local validation tracking via `validation_sequences.csv` and Public LB with merged training/validation template pool

## Preprocessing
- Sequence truncation/chunking (max lengths 512, 768, 850 nt/tokens with overlaps 64-128)
- C1' coordinate extraction and matching
- Gap filling via linear interpolation and 5.95Å extrapolation
- Template filtering via BioPython PairwiseAligner
- MSA depth limiting (≤2048)
- Secondary structure computation (dot-bracket/ViennaRNA/EternaFold)
- Kabsch alignment on overlapping regions
- Memory cleanup per chunk
- RNA-tuned gap penalties
- Helical trajectory interpolation
- TM-score-based greedy diversity selection
- Length-adaptive slot allocation
- Padding/repeating samples to meet prediction counts

## Feature engineering
- BPP matrix injection as pairwise embedding
- BPP binning into one-hot + linear embedding
- BppEmbedder with Pairformer in recycling loop
- C1'-C1' distance contact metric validation
- Sequence similarity features (alignment score, percent identity, length diff ratio)
- Text embedding cosine similarity (NeuML/pubmedbert)
- Protein/DNA/Ligand similarity matrices (BLOSUM62, alignment scores, Tanimoto/Morgan fingerprints)
- Composition counts (query/template proteins/dna/ligands)
- Chain-related counts (all/unique chains, chain counts match)
- Secondary structure state encoding into extended alignment alphabet
- 12x12 distance matrix for alignment
- Hungarian algorithm for multi-chain matching
- Coordinate shrinking toward average position
- RMSD-based candidate averaging

## Augmentations
- Gaussian noise perturbation scaled by template similarity/dissimilarity
- Geometric hinge/jitter perturbation
- Soft geometric smoothing/stereochemical regularization via `adaptive_rna_constraints()`
- Smooth wiggle deformation

## Ensemble patterns
- Best-of-5 selection from structurally diverse pools
- Length-adaptive slot allocation based on sequence length thresholds (<250, 250-999, ≥1000 nt)
- Fixed slot allocation (e.g., 2 TBM, 1 Protenix, 2 RNAPro)
- Confidence-based top-k selection with fallback models
- Dynamic template-quality-based slot allocation
- MSA-depth-aware sampling (high vs low depth)
- Toggling `USE_RNA_MSA` flag for diversity
- 3-phase pipeline with model replacement based on sequence length
- Averaging low-RMSD candidates
- Replacing low-scoring TBM outputs with Protenix based on thresholds
- Quintic smoothstep stitching for long sequences

## Post-processing
- Adaptive RNA constraints scaling (bond/angle correction, Laplacian smoothing, steric self-avoidance)
- Confidence-based top-k selection
- Kabsch alignment on overlapping regions/chunks
- Coordinate shrinking toward average position
- RMSD-based candidate averaging
- Score-threshold-based model replacement
- Interpolation/extrapolation for chain segments
- Seamless chunk blending
- Soft geometric smoothing

## What worked
- Combination of five structurally diverse models proved robustly better than single models on unseen data
- TBM served as a highly competitive backbone for medium/long sequences with close training matches
- Length-adaptive allocation balanced inference time and score, with the 250 nt boundary optimizing speed vs accuracy
- Prioritizing diversity over TBM dependency improved performance on private LB targets lacking close templates
- Simple linear injection of BPP outperformed complex binning and embedding methods
- EternaFold BPP provided higher contact accuracy than ViennaRNA
- BPP-Protenix standalone performance matched template-based approaches
- Feeding Protenix outputs into RNAPro as templates to explore independent structure space
- Retaining raw Protenix predictions to prevent RNAPro overcorrection
- Chunking long sequences for Protenix improved local scores
- Fixing the TBM-threshold bottleneck that blocked Protenix inference boosted local validation
- Toggling `USE_RNA_MSA` effectively diversified the RNAPro ensemble
- Dynamic slot allocation based on template quality
- Adaptive Protenix sampling (2-5 samples depending on template quality)
- Using RNAPro as a differentiating model to overcome Protenix's stochastic variance
- Encoding secondary structure into alignment alphabet
- Hungarian algorithm for multi-chain matching
- Dynamic chunk sizing for sequences up to 768 tokens
- Parallel Protenix inference on 2xT4 GPUs
- Shrinking copied chain coordinates toward their average
- Averaging Kabsch-aligned candidates with RMSD ≤ 2Å
- Replacing low-scoring TBM candidates with Protenix based on thresholds

## What did not work
- Running Protenix with MSA
- Using more than 2 TBM template slots in RNAPro
- Ensemble averaging of coordinates
- Binning BPP into one-hot vectors and adding a separate linear embedding
- Adding a BppEmbedder with a 2-block Pairformer in the recycling loop
- MSA consensus-based ViennaRNA secondary structure
- MMseqs-based template search
- Coordinate post-processing based on distance constraints and trajectory smoothing
- Orienting independent chain predictions using templates or truncated multi-chain Protenix outputs
- Reranking Protenix by model confidence (too slow, low return)
- Finetuning Protenix (abandoned due to time)
- Using templates for Protenix (no LB/local improvement)
- Relying solely on pure TBM or pure Protenix
- Protenix parameter tuning (SDPA/Flash Attention, MSA, multi-chain input, ligand SMILES, seed changes, version upgrades)
- Template pool expansion (Ribonanza templates, post-cutoff PDB, full PDB-RNA)
- Alternative models: Chai-1, RhoFold+, DRfold2, and Boltz

## Critical findings
- TBM performance heavily depends on template similarity; near-identical templates require minimal correction while low-similarity matches need aggressive constraint scaling
- Boltz2 and DRFold2 are computationally viable only for sequences <250 nt, making TBM the only scalable backbone for longer targets
- Diversity in prediction sources directly correlated with private leaderboard gains, as hard targets lacked close training templates
- Allowing protein/DNA partners in the BPP filter caused almost no change in public LB score, suggesting few such targets exist in the test set
- BPP provides continuous pairing hints rather than discrete structures, so missing pseudoknots did not harm performance
- BPP-Protenix improved both public and private LB scores, confirming the validation strategy's reliability
- MSA depth distribution on test targets revealed 54% ≥700, 25% 10–699, 21% ≤9; depth ≥700 is optimal as lower depths inject alignment noise
- Linear chunk blending causes first-derivative discontinuities; quintic smoothstep guarantees C2 continuity
- Non-deterministic CUDA/attention backends and MC Dropout cause run-to-run variance that masks algorithmic improvements
- Public LB was heavily dominated by the TBM approach, masking Protenix/RNAPro improvements
- `USE_RNA_MSA=true` boosted local validation by ~0.03 but dropped Public LB scores when combined with TBM
- Protenix predictions are non-deterministic but generating multiple samples per runtime did not yield helpful diversity
- Protenix's stochastic diffusion caused ~0.02–0.03 score variance across reruns, making public LB unreliable for evaluating true accuracy
- Most targets classified as HIGH template quality still had fewer than 5 qualifying templates in practice
- RNAPro was the only alternative model explored that reached competitive scores on the public LB
- Parasail requires capital letters in the alphabet by default, which was initially overlooked
- Generating extra diffusion samples with the same seed is nearly as computationally expensive as running a fresh seed
- USalign penalizes chain misalignments more severely than missing chains, making coordinate shrinking preferable to copying
- MSA-based consensus secondary structure computation is prohibitively slow and degrades performance compared to sequence-only SSI

## Notable individual insights
- rank 1 (1st Place Solution — Five-Model Ensemble with Length-Adaptive Allocation): Length-adaptive allocation at the 250 nt boundary optimally balances speed vs accuracy, and prediction diversity directly correlates with private LB gains when templates are scarce.
- rank 2 (2nd Place Solution): Simple linear injection of BPP matrices outperformed complex binning/embedding methods, and EternaFold BPP provides higher contact accuracy than ViennaRNA.
- rank 5 (5th Place Solution — msa and protenix and TBM): MSA depth ≥700 is optimal for Protenix as lower depths inject alignment noise, while quintic smoothstep guarantees C2 continuity over linear blending.
- rank 8 (8th Place Solution - TBM and Protenix): USalign penalizes chain misalignments more severely than missing chains, making coordinate shrinking preferable to copying; generating extra diffusion samples with the same seed is nearly as expensive as fresh seeds.
- rank 10 (10th Place Solution): Public LB was heavily dominated by TBM, masking Protenix/RNAPro improvements, and toggling `USE_RNA_MSA` effectively diversified the RNAPro ensemble.
- rank 7 (7th Place Solution): Protenix's stochastic diffusion caused ~0.02–0.03 score variance, making public LB unreliable for evaluating true accuracy.
- rank 6 (6th Place Solution): Cross-pollinating Protenix predictions as templates into RNAPro ensures independent structural hypotheses, while retaining raw Protenix outputs prevents RNAPro overcorrection.

## Solutions indexed
- #1 [[solutions/rank_01/solution|1st Place Solution — Five-Model Ensemble with Length-Adaptive Allocation]]
- #2 [[solutions/rank_02/solution|2nd Place Solution]]
- #3 [[solutions/rank_03/solution|3rd Place Solution]]
- #5 [[solutions/rank_05/solution|5th Place Solution — msa and protenix and TBM]]
- #6 [[solutions/rank_06/solution|6th Place Solution]]
- #7 [[solutions/rank_07/solution|7th Place Solution]]
- #8 [[solutions/rank_08/solution|8th Place Solution - TBM and Protenix]]
- #10 [[solutions/rank_10/solution|10th Place Solution]]

## GitHub links
- [bytedance/Protenix](https://github.com/bytedance/Protenix) _(library)_ — from [[solutions/rank_06/solution|6th Place Solution]]
- [ShindeShivam/Stanford-RNA-3D-Folding](https://github.com/ShindeShivam/Stanford-RNA-3D-Folding) _(reference)_ — from [[solutions/rank_06/solution|6th Place Solution]]
- [yutarooo216/Stanford-RNA-3D-Folding-Part-2-1st-place](https://github.com/yutarooo216/Stanford-RNA-3D-Folding-Part-2-1st-place) _(solution)_ — from [[solutions/rank_01/solution|1st Place Solution — Five-Model Ensemble with Length-Adaptive Allocation]]
- [ViennaRNA/ViennaRNA](https://github.com/ViennaRNA/ViennaRNA) _(reference)_ — from [[solutions/rank_02/solution|2nd Place Solution]]
- [WaymentSteeleLab/EternaFold](https://github.com/WaymentSteeleLab/EternaFold) _(library)_ — from [[solutions/rank_02/solution|2nd Place Solution]]
- [DasLab/arnie](https://github.com/DasLab/arnie) _(library)_ — from [[solutions/rank_02/solution|2nd Place Solution]]
- [NVIDIA-Digital-Bio/RNAPro](https://github.com/NVIDIA-Digital-Bio/RNAPro) _(reference)_ — from [[solutions/rank_03/solution|3rd Place Solution]]
- [bytedance/Protenix](https://github.com/bytedance/Protenix) _(reference)_ — from [[solutions/rank_03/solution|3rd Place Solution]]
- [bytedance/Protenix](https://github.com/bytedance/Protenix) _(library)_ — from [[solutions/rank_08/solution|8th Place Solution - TBM and Protenix]]
- [jeffdaily/parasail](https://github.com/jeffdaily/parasail) _(library)_ — from [[solutions/rank_08/solution|8th Place Solution - TBM and Protenix]]
- [ViennaRNA/ViennaRNA](https://github.com/ViennaRNA/ViennaRNA) _(library)_ — from [[solutions/rank_08/solution|8th Place Solution - TBM and Protenix]]
- [bytedance/Protenix](https://github.com/bytedance/Protenix) _(reference)_ — from [[solutions/rank_10/solution|10th Place Solution]]
- [NVIDIA-Digital-Bio/RNAPro](https://github.com/NVIDIA-Digital-Bio/RNAPro) _(reference)_ — from [[solutions/rank_07/solution|7th Place Solution]]
- [bytedance/Protenix](https://github.com/bytedance/Protenix) _(reference)_ — from [[solutions/rank_07/solution|7th Place Solution]]

## Papers cited
- [RNAdegformer](https://academic.oup.com/bib/article/24/1/bbac581/6986359)
- [Template-based RNA structure prediction advanced through a blind code competition](https://doi.org/10.64898/2025.12.30.696949)
- [Protenix-v1: Toward High-Accuracy Open-Source Biomolecular Structure Prediction](https://doi.org/10.64898/2026.02.05.703733)
