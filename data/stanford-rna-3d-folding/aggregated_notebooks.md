# stanford-rna-3d-folding: top public notebooks

The community's top-voted notebooks primarily focus on building robust inference pipelines for RNA 3D structure prediction, leveraging pre-trained diffusion models (Protenix, Boltz-1) and sequence-to-structure architectures (RibonanzaNet, RhoFold, MSA2XYZ). Authors emphasize practical engineering strategies such as test-time averaging, sequence-length splitting, CIF/PDB parsing, and adapting scientific models for Kaggle's resource constraints. A smaller subset covers exploratory data analysis, memory optimization, and fine-tuning with rotation-invariant losses to address structural biology blind-testing protocols.

## Common purposes
- inference
- eda
- training
- ensemble

## Competition flows
- Loads RNA sequence CSVs, runs the Protenix diffusion model to predict 3D coordinates, evaluates predictions on a validation set using TM-score via USalign, and writes final predictions to a submission CSV.
- Loads test RNA sequences, tokenizes them, runs them through a pretrained RibonanzaNet model with a 3D coordinate prediction head, averages predictions across 5 stochastic forward passes, and exports the coordinates to a CSV submission file.
- Loads precomputed 3D coordinates from a pickle file, aligns them with test sequences, formats them into the required submission CSV, and saves the file.
- Loads RNA sequence and coordinate data, filters and temporally splits it, fine-tunes a pre-trained RibonanzaNet model with a custom coordinate regression head using SVD-aligned MAE loss, and saves the best checkpoint.
- Reads RNA sequences from a CSV, formats them as YAML inputs for Boltz-1, runs GPU-accelerated inference to generate 3D structure predictions, extracts C1' coordinates from the output CIF files, averages across 5 diffusion samples, and exports the formatted coordinates to a submission CSV.
- Loads test sequences, runs RibonanzaNet inference for all targets, runs RhoFold inference with MSAs for sequences ≤400 residues, extracts C1' coordinates from PDB files, merges the two prediction sets by filling missing coordinates, and exports a formatted submission CSV.
- Loads RNA sequence and coordinate CSVs, analyzes ID structures and missing value patterns, applies memory optimization and one-hot encoding, and prepares padded numpy arrays for downstream modeling.
- Reads test RNA sequences from CSV, runs the Protenix diffusion model to predict 3D structures, extracts C1' coordinates from CIF outputs, and generates a formatted submission CSV.
- Loads RNA sequence data, passes it through a pre-trained MSA2XYZ model paired with an RNA2nd language model to generate 3D coordinates, evaluates multiple conformations using USalign, and outputs a CSV submission file.

## Data reading
- Reads train_sequences.csv, train_labels.csv, and test_sequences.csv from /kaggle/input/stanford-rna-3d-folding using pandas.
- Extracts sequence and target_id columns for model input and ground truth alignment.
- pd.read_csv on test_sequences.csv
- Reads `test_sequences.csv` using `pandas.read_csv`.
- Loads test sequences and sample submission templates using pandas read_csv from the Kaggle input directory.
- Reads test_sequences.csv using pandas.
- Converts sequences to FASTA files for RhoFold input.
- Loads precomputed MSA files from /kaggle/input/stanford-rna-3d-folding/MSA/.
- Uses pd.read_csv with chunksize=50000 and on_bad_lines='skip' for large files; direct pd.read_csv for smaller files; includes os.path.exists checks and fallback handling for missing competition files.
- pd.read_csv() is used to load validation_sequences.csv, validation_labels.csv, train_sequences.csv (repurposed as test data), and sample_submission.csv.
- Reads `validation_sequences.csv` or `test_sequences.csv` via `pandas.read_csv`, extracts `target_id` and `sequence` columns, and conditionally loads validation labels for local testing.
- Reads CSV files using pd.read_csv with chunksize=50000, on_bad_lines='skip', and low_memory=False.
- Uses os.path.exists checks and os.listdir for directory/file verification.

## Data processing
- Filters sequences longer than 300 nucleotides to manage memory and runtime.
- Converts predicted and ground truth coordinates to PDB format for structural alignment.
- Uses USalign to compute TM-scores and rotation matrices between predicted and true structures.
- Handles missing/NaN coordinates and extracts specific atom indices (C1' for RNA, index 12) from model outputs.
- Tokenizes 'ACGU' characters to integer indices.
- Applies test-time averaging via 5 forward passes (4 with dropout enabled, 1 with it disabled) to stabilize predictions.
- Truncates precomputed coordinates to match test sequence lengths using min(len(sequence), len(prediction)).
- Formats 3D coordinates into the competition's required ID/resname/resid CSV schema.
- Filters sequences with >50% NaN coordinates or length outside [10, 9999999]; crops sequences to a maximum length of 384 during dataset iteration; maps nucleotides (A, C, G, U) to integer tokens; and applies a temporal cutoff split for validation.
- Converts CSV sequences into YAML format with RNA chain metadata and empty constraints; parses output .cif files using Biopython's MMCIF2Dict to extract Cartesian coordinates for C1' atoms; cleans temporary working directories after inference.
- Maps nucleotides (A, C, G, U) to integer tokens for model input.
- Splits inference by sequence length to manage GPU memory.
- Extracts C1' atom coordinates from PDB files using Biopython's PDBParser.
- Aligns and merges coordinate predictions across models by target ID and residue index.
- Duplicates extracted coordinates across five prediction columns for submission formatting.
- Memory optimization via dtype downcasting (int8/16/32, float32) and threshold-based categorical conversion; filtering sentinel values (-1.0e+18); one-hot encoding of nucleotides (A, C, G, U, N); sequence padding to uniform length; heuristic ID mapping between sequence and label DataFrames.
- Constructs input JSONs with RNA sequences and modifications; parses CIF files via biotite to extract and sort C1' atom coordinates; pads missing predictions with zeros and duplicates available models to ensure exactly 5 samples per sequence.
- Parses sequences into one-hot encoded tensors, computes base coordinates, truncates sequences exceeding 480 residues with zero-padding, converts model outputs to PDB format, and aligns structures using USalign.
- Memory optimization via dtype downcasting (int8/16/32, float32, category) based on value ranges and cardinality thresholds.
- Filtering sentinel missing values (-1.0e+18) instead of standard NaN.
- One-hot encoding nucleotides (A, C, G, U, N) and padding sequences to uniform length.
- Centering 3D coordinates by mass and applying custom structural noise augmentation with correlated noise and hinge-point rotation.

## Features engineering
- One-hot encoding of RNA sequences
- Sequence length padding
- Heuristic ID mapping based on residue count alignment and prefix extraction

## Models
- Protenix
- RibonanzaNet
- Boltz-1
- RhoFold
- MSA2XYZ
- RNA2nd

## Frameworks used
- PyTorch
- pandas
- NumPy
- Matplotlib
- Biopython
- Biotite
- RDKit
- PyYAML
- Plotly
- PyTorch Lightning
- SciPy
- scikit-learn
- Seaborn
- TensorFlow
- Protenix
- subprocess
- glob
- tqdm

## Loss functions
- dRMAE
- align_svd_mae

## CV strategies
- temporal holdout split based on publication dates

## Ensembling
- Averages predictions across 5 forward passes per sequence, using 4 stochastic passes (with dropout active) and 1 deterministic pass to mitigate variance.
- Averages predicted coordinates across 5 diffusion samples per target ID using pandas groupby mean to stabilize structural predictions.
- Combines RibonanzaNet and RhoFold predictions by sequence length, merging them on target ID and residue index, and filling missing RhoFold coordinates with RibonanzaNet coordinates for the first four prediction columns while keeping the fifth from RibonanzaNet.
- Generates 5 structural samples per sequence via diffusion sampling and treats them as ensemble members; no explicit weighting or voting is applied, only direct coordinate extraction and zero-padding for submission formatting.
- Generates 5 conformations using 5 different model checkpoints, aligns them to the reference structure via USalign, computes TM-scores, and selects the conformation with the highest score for the final submission.

## Insights
- Protenix can effectively predict RNA 3D structures using a diffusion-based approach.
- TM-score calculated via USalign provides a reliable metric for evaluating structural alignment quality.
- Correctly indexing the target atom (C1' for RNA, index 12) is critical for accurate coordinate extraction.
- Filtering sequences longer than 300 nucleotides is necessary to manage computational constraints during inference.
- Using `model.train()` during inference enables dropout-based test-time averaging, which can stabilize predictions for stochastic models.
- Directly predicting 3D coordinates from sequence embeddings bypasses the need for intermediate contact map or pairwise feature extraction during inference.
- Precomputed structural predictions can be directly adapted to competition submission formats by matching sequence lengths and coordinate dimensions.
- Loading external pickle files allows rapid baseline generation without retraining.
- Proper formatting of 3D coordinates into the competition's ID/resname/resid schema is critical for valid submissions.
- Temporal data splitting better reflects real-world structural biology blind testing than random k-fold splits.
- Distance-based and SVD-aligned losses help mitigate rotation/translation invariance issues in 3D coordinate regression.
- Filtering sequences by coordinate completeness prevents training on heavily masked or incomplete structures.
- Pre-trained diffusion models can be directly deployed for RNA 3D coordinate prediction without task-specific training or fine-tuning.
- Programmatic parsing of CIF files via Biopython enables precise extraction of specific atom types like C1' without manual inspection.
- Leveraging multiple diffusion samples and averaging their outputs improves prediction stability for structural biology tasks.
- RhoFold yields higher accuracy but triggers out-of-memory errors on long sequences when using GPU.
- Splitting inference by sequence length allows both models to run within Kaggle's resource limits.
- Precomputed MSAs significantly enhance RhoFold's structural predictions.
- Disabling relaxation steps reduces inference time without severely impacting leaderboard scores.
- Memory optimization thresholds directly impact DataFrame size and should be tuned per dataset.
- RNA coordinate data uses -1.0e+18 as a sentinel for missing values rather than standard NaN.
- Validation sequences map directly to labels via ID prefixes, while training requires heuristic matching based on residue counts.
- Pre-trained RNA language models can effectively replace MSA inputs for 3D structure prediction, significantly simplifying the inference pipeline.
- Generating multiple conformations and selecting the best via structural alignment (TM-score) is a robust strategy for handling model uncertainty.
- Sequence length truncation and padding are necessary practical constraints for managing GPU memory during large-scale inference.
- Protenix can be seamlessly integrated into Kaggle environments by installing wheels and symlinking CCD cache files.
- Parsing CIF/PDBx outputs with biotite reliably extracts the C1' coordinates required for the competition format.
- The submission format requires expanding each RNA residue into multiple rows to accommodate coordinates from 5 different model samples.
- Structural variations can be realistically simulated using correlated noise and hinge-point rotations to preserve local bond lengths.
- TM-score calculation requires adaptive d0 scaling based on sequence length to remain biologically meaningful.

## Critical findings
- The author explicitly notes that dRMSD loss cannot distinguish chiral structures due to reflection invariance, implying potential limitations in predicting correct stereochemistry.
- The training set lacks a direct ID correspondence between sequences and labels, requiring a fallback mapping strategy based on sequence length and residue count alignment.
- The training and validation ID formats differ, making direct mapping impossible without heuristic matching or fallback to validation data.
- Coordinate columns contain -1.0e+18 as a sentinel for missing data, which must be filtered before analysis.
- Sequence lengths vary significantly, necessitating padding for batch processing.

## What did not work
- Running RhoFold on CPU for long sequences caused notebook timeouts during leaderboard evaluation, despite functioning in public testing.
- A direct ID mapping between train_sequences.csv and train_labels.csv failed due to format mismatches, forcing the author to use validation data for training instead.

## Notable individual insights
- votes 566 (Randomness): Protenix can effectively predict RNA 3D structures using a diffusion-based approach, but filtering sequences >300 nucleotides is necessary to manage computational constraints.
- votes 549 (RibonanzaNet 3D Inference): Using model.train() during inference enables dropout-based test-time averaging, which stabilizes predictions for stochastic models.
- votes 429 (RibonanzaNet 3D Finetune): Temporal data splitting better reflects real-world structural biology blind testing than random k-fold splits, and dRMSD loss cannot distinguish chiral structures due to reflection invariance.
- votes 413 (Boltz-1 inference & submission): Pre-trained diffusion models can be directly deployed for RNA 3D coordinate prediction without task-specific training, and programmatic parsing of CIF files via Biopython enables precise atom extraction.
- votes 338 (RhoFold + RibonanzaNet + MSAs): RhoFold yields higher accuracy but triggers out-of-memory errors on long sequences; splitting inference by sequence length allows both models to run within resource limits.
- votes 276 (RNA 3D Noise Changed): A direct ID mapping between train_sequences.csv and train_labels.csv failed due to format mismatches, forcing reliance on validation data for training.

## Notebooks indexed
- #566 votes [[notebooks/votes_01_adamlogman-randomness/notebook|Randomness]] ([kaggle](https://www.kaggle.com/code/adamlogman/randomness))
- #549 votes [[notebooks/votes_02_shujun717-ribonanzanet-3d-inference/notebook|RibonanzaNet 3D Inference]] ([kaggle](https://www.kaggle.com/code/shujun717/ribonanzanet-3d-inference))
- #460 votes [[notebooks/votes_03_hengck23-vfold-baseline-offline/notebook|vfold baseline - offline]] ([kaggle](https://www.kaggle.com/code/hengck23/vfold-baseline-offline))
- #429 votes [[notebooks/votes_04_shujun717-ribonanzanet-3d-finetune/notebook|RibonanzaNet 3D Finetune]] ([kaggle](https://www.kaggle.com/code/shujun717/ribonanzanet-3d-finetune))
- #413 votes [[notebooks/votes_05_youhanlee-boltz-1-inference-submission/notebook|Boltz-1 inference & submission]] ([kaggle](https://www.kaggle.com/code/youhanlee/boltz-1-inference-submission))
- #338 votes [[notebooks/votes_06_ogurtsov-rhofold-ribonanzanet-msas-lb-0-215/notebook|RhoFold + RibonanzaNet + MSAs [LB 0.215]]] ([kaggle](https://www.kaggle.com/code/ogurtsov/rhofold-ribonanzanet-msas-lb-0-215))
- #309 votes [[notebooks/votes_07_fernandosr85-rna-3d-structure/notebook|RNA 3D Structure]] ([kaggle](https://www.kaggle.com/code/fernandosr85/rna-3d-structure))
- #292 votes [[notebooks/votes_08_ishgirwan-protenix-inference/notebook|protenix inference]] ([kaggle](https://www.kaggle.com/code/ishgirwan/protenix-inference))
- #279 votes [[notebooks/votes_09_hengck23-lb0-321-simple-drfold-no-msa/notebook|[lb0.321] simple drfold - NO MSA]] ([kaggle](https://www.kaggle.com/code/hengck23/lb0-321-simple-drfold-no-msa))
- #276 votes [[notebooks/votes_10_dolbokostya-lb-0-261-rna-3d-noise-changed/notebook|[LB 0.261] RNA 3D Noise Changed]] ([kaggle](https://www.kaggle.com/code/dolbokostya/lb-0-261-rna-3d-noise-changed))
