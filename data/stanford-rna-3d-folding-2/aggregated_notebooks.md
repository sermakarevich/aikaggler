# stanford-rna-3d-folding-2: top public notebooks

The community's top notebooks focus almost exclusively on hybrid inference pipelines for RNA 3D structure prediction, combining template-based modeling (TBM), diffusion-based generative models, and geometric fallbacks. Rather than training new architectures, participants leverage classical sequence alignment, coordinate adaptation, and adaptive physical constraints to generate diverse, submission-ready conformations. A recurring theme is robust handling of long sequences via overlapping chunking and Kabsch-aligned stitching, alongside careful stoichiometry parsing for multi-chain complexes.

## Common purposes
- inference
- baseline

## Competition flows
- Hybrid TBM + Protenix + de-novo fallback pipeline with chunking/stitching and CSV export
- Template-based coordinate adaptation with physical constraint enforcement and multi-sample CSV export
- TBM + Protenix inference with gap-filling via de-novo helix and fixed sample count export
- Classical alignment-driven template matching with stochastic perturbations and multi-variant CSV export
- Custom TBM + RNAPro integration with conditional long-sequence replacement and CSV export
- Homology template selection with fixed-weight averaging and geometric refinement before CSV export

## Data reading
- pandas.read_csv for test, train, and validation sequence/label CSVs
- Custom parsing for stoichiometry, chain segmentation, and FASTA headers
- JSON serialization for model input formatting
- Symlinked PDB_RNA .cif files for template data
- Grouping training labels by target ID to extract initial 3D coordinates

## Data processing
- Global sequence alignment via Biopython PairwiseAligner
- Template coordinate adaptation via alignment mapping and linear interpolation
- Overlapping chunk splitting for long sequences
- Kabsch alignment and linear weight blending for chunk stitching
- Adaptive RNA physical constraints (bond lengths, angles, Laplacian smoothing, self-avoidance)
- Diversity transforms (hinge rotation, chain jitter, spline wiggle)
- Coordinate clipping to [-999.999, 9999.999]
- C1' atom masking and coordinate padding/heuristics
- Template filtering by length difference and similarity thresholds
- Confidence-weighted Gaussian noise injection for ensemble diversity
- Merging submissions by ID matching and conditional replacement
- Filling gaps with fixed helical steps or random walk fallbacks

## Features engineering
- Domain-derived geometric constraints (backbone bond lengths, i-i+2 distances, Laplacian smoothing, self-avoidance)
- Stoichiometry parsing for multi-chain complex handling
- Idealized A-form helix coordinates for de-novo fallback
- Confidence-weighted diversity transforms
- Sequence-level features (length, GC/AU content, alphabet complexity)
- Alignment scores, percent identity, and length ratio as similarity metrics

## Models
- Protenix
- Template-Based Modeling (TBM)
- de-novo A-form RNA helix fallback
- RNAPro
- USalign

## Frameworks used
- torch
- numpy
- pandas
- biopython
- rdkit
- tqdm
- scipy
- scikit-learn
- Bio.Align
- biotite
- RNAPro
- json
- pathlib
- sys
- os
- time
- random
- warnings
- contextlib

## Ensembling
- Filling prediction slots sequentially with TBM, Protenix, and de-novo helix fallbacks
- Concatenating TBM, Protenix, and de-novo outputs without weighted averaging
- Combining top template adaptations with de novo structures using confidence-weighted stochastic noise
- Conditionally replacing deep learning coordinates with TBM results for long sequences (>1000 residues)
- Applying fixed weights to top homologous templates before filling remaining slots with generative models
- Stacking multiple coordinate sets into a fixed sample dimension for submission

## Insights
- Homology modeling remains highly effective when templates share sufficient similarity and identity.
- Overlapping chunking with Kabsch alignment and linear blending successfully extends diffusion models beyond their native sequence length limits.
- Adaptive physical constraints significantly improve the geometric plausibility of predicted RNA structures.
- Deterministic seeding and chunk stitching ensure reproducible and artifact-free submissions.
- Hybrid inference pipelines that leverage training data templates significantly reduce the need for pure de-novo prediction.
- Collapsed coordinate outputs from Protenix must be explicitly detected and discarded to prevent submission errors.
- Stoichiometry and chain segmentation must be parsed per-target to correctly apply constraints to multi-chain complexes.

## Critical findings
- Collapsed coordinates (where all atoms overlap) are explicitly detected and filtered during Protenix inference.
- Stoichiometry parsing is required to correctly handle multi-chain RNA complexes.
- Sequences with low similarity or identity to the training pool are correctly routed away from TBM to avoid poor adaptations.
- TBM alone achieves the best leaderboard score (0.249), while adding Protenix, MSA, or template features degrades performance to ~0.40.
- Protenix outputs can collapse to identical coordinates if the C1' atom mask selection heuristic is incorrect or when sequence length truncation causes shape mismatches.
- Chunk stitching relies on a minimum overlap length (≥3 residues) to compute reliable Kabsch alignments; shorter overlaps are skipped.

## What did not work
- Adding MSA, RNA MSA, or template features to Protenix degraded the leaderboard score compared to TBM alone.
- Protenix inference fails or produces collapsed coordinates when the C1' atom mask selection heuristic is incorrect or when sequence length truncation causes shape mismatches.

## Notable individual insights
- votes 232 (Stanford RNA 3D Folding Part 2|Protenix+TBM): TBM alone achieves the best leaderboard score (0.249), while adding Protenix, MSA, or template features degrades performance to ~0.40.
- votes 232 (RNAPro inference with TBM): Long RNA sequences (>1000 residues) may benefit from classical structural methods over deep learning models in this competition setting.
- votes 209 (Stanford RNA 3D | TaBM Optimized | Stable LB-0.371): Confidence-weighted noise effectively diversifies ensemble predictions while preserving high-confidence matches.
- votes 203 ([0.409] Stanford RNA Folding 2: Protenix+ Template): Multi-chain stoichiometry parsing is critical for correctly mapping coordinates across complex RNA assemblies.
- votes 597 (Stanford RNA 3D Folding): Overlapping chunking with Kabsch alignment and linear blending successfully extends diffusion models beyond their native sequence length limits.
- votes 232 (Stanford RNA 3D Folding Part 2|Protenix+TBM): Protenix outputs can collapse to identical coordinates if the C1' atom mask selection heuristic is incorrect or when sequence length truncation causes shape mismatches.

## Notebooks indexed
- #597 votes [[notebooks/votes_01_asimandia-stanford-rna-3d-folding/notebook|Stanford RNA 3D Folding]] ([kaggle](https://www.kaggle.com/code/asimandia/stanford-rna-3d-folding))
- #440 votes [[notebooks/votes_02_sigmaborov-stanford-rna-3d-folding-top-1-solution/notebook|Stanford RNA 3D Folding — Top 1 Solution ]] ([kaggle](https://www.kaggle.com/code/sigmaborov/stanford-rna-3d-folding-top-1-solution))
- #383 votes [[notebooks/votes_03_artemevstafyev-stanford-rna-3d-folding/notebook|Stanford RNA 3D Folding]] ([kaggle](https://www.kaggle.com/code/artemevstafyev/stanford-rna-3d-folding))
- #344 votes [[notebooks/votes_04_amirrezaaleyasin-openrnafold-starter3/notebook|OpenRNAFold - starter3]] ([kaggle](https://www.kaggle.com/code/amirrezaaleyasin/openrnafold-starter3))
- #297 votes [[notebooks/votes_05_nihilisticneuralnet-stanford-rna-folding-2-template-based-approach/notebook|Stanford RNA Folding 2: Template-based Approach]] ([kaggle](https://www.kaggle.com/code/nihilisticneuralnet/stanford-rna-folding-2-template-based-approach))
- #274 votes [[notebooks/votes_06_artemevstafyev-high-score-without-hash/notebook|High Score Without Hash]] ([kaggle](https://www.kaggle.com/code/artemevstafyev/high-score-without-hash))
- #232 votes [[notebooks/votes_07_llkh0a-stanford-rna-3d-folding-part-2-protenix-tbm/notebook|Stanford RNA 3D Folding Part 2|Protenix+TBM]] ([kaggle](https://www.kaggle.com/code/llkh0a/stanford-rna-3d-folding-part-2-protenix-tbm))
- #232 votes [[notebooks/votes_08_jaejohn-rnapro-inference-with-tbm/notebook|RNAPro inference with TBM]] ([kaggle](https://www.kaggle.com/code/jaejohn/rnapro-inference-with-tbm))
- #209 votes [[notebooks/votes_09_datasciencegrad-stanford-rna-3d-tabm-optimized-stable-lb-0-371/notebook|Stanford RNA 3D | TaBM Optimized | Stable LB-0.371]] ([kaggle](https://www.kaggle.com/code/datasciencegrad/stanford-rna-3d-tabm-optimized-stable-lb-0-371))
- #203 votes [[notebooks/votes_10_nihilisticneuralnet-0-409-stanford-rna-folding-2-protenix-template/notebook|[0.409] Stanford RNA Folding 2: Protenix+ Template]] ([kaggle](https://www.kaggle.com/code/nihilisticneuralnet/0-409-stanford-rna-folding-2-protenix-template))
