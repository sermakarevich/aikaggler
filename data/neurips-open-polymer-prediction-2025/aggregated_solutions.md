# neurips-open-polymer-prediction-2025: cross-solution summary

The Open Polymer Prediction 2025 competition focused on predicting multiple polymer properties from SMILES strings and external datasets, where winning approaches heavily emphasized rigorous data cleaning, strict train/test leakage prevention, and target-specific post-processing to correct systematic distribution shifts. Top solutions combined extensive molecular feature engineering with a mix of modern foundation models (BERT variants, Uni-Mol), automated tabular tuning (AutoGluon), and tree-based ensembles, ultimately favoring robust cross-validation schemes and distribution alignment over complex model architectures.

## Competition flows
- Raw SMILES and external datasets were cleaned, deduplicated, and augmented, then fed into separate BERT, AutoGluon, and Uni-Mol pipelines with extensive tabular feature engineering and MD-derived descriptors, followed by property-specific ensembling and a constant-bias post-processing step to correct test label distribution shifts.
- Raw polymer SMILES data is processed with molecular descriptors and fingerprints, filtered via strict similarity-based group CV, trained on a TabM architecture with target-specific hyperparameters, and submitted after applying a discovered temperature bias correction to external data.
- Raw polymer data was processed into 1,072 molecular and structural features, which were fed into a target-specific weighted ensemble of tree-based and KNN models using stratified 5-fold CV, followed by distribution alignment of predictions to match the training set before final submission.

## Data processing
- Canonical SMILES conversion for deduplication
- Tanimoto similarity filtering (>0.9) to prevent train/test leakage
- Isotonic regression for label rescaling
- Error-based data filtering using MAE ratios
- Optuna-tuned sample weighting per dataset
- Semi-manual threshold filtering for suspicious values
- Non-canonical SMILES generation (10x training, 50x TTA)
- Polymerization to ~600 atoms per chain
- Amorphous cell generation (10 chains, 0.15 density)
- LAMMPS equilibrium simulation for property estimation
- External data target value adjustment (+20 to Tg) for simulation bias correction
- Feature selection to reduce dimensionality per target
- Removal of constant and highly correlated features
- Prediction distribution alignment via mean and standard deviation matching

## Features engineering
- RDKit 2D and graph molecular descriptors
- Morgan fingerprints
- Atom pair fingerprints
- Topological torsion fingerprints
- MACCS keys
- NetworkX graph features
- Backbone vs. sidechain features (+107 AI-generated variants)
- MD simulation descriptors (FFV, density, Rg, 3D descriptors)
- polyBERT embeddings
- Gasteiger charge statistics
- Element composition and bond type ratios
- XGBoost ensemble predictions on MD results
- Mordred molecular descriptors
- Polymer structural features
- AUTOCORR2D

## Models
- ModernBERT-base
- ModernBERT-large
- CodeBERT
- AutoGluon
- Uni-Mol 2
- D-MPNN
- ChemBERTa
- polyBERT
- DeBERTa-v3-large
- XGBoost
- LightGBM
- TabM
- isotonic regression models
- KNN
- CatBoost
- HistGradientBoosting
- ExtraTrees

## Frameworks used
- AutoGluon
- Optuna
- RDKit
- LAMMPS
- psi4
- MDAnalysis
- XGBoost
- LightGBM
- Uni-Mol
- HuggingFace

## CV strategies
- 5-fold cross-validation on the host's original training data (80% train, 20% test per fold), with all selected external data included in training. CV scores were computed exclusively on the held-out host data to prevent test leakage.
- 5-fold RandomGroupKFold based on Butina clusters, with strict filtering of training samples having >0.9 Tanimoto similarity to validation samples to prevent leakage.
- Stratified K-Fold CV (k=5) with stratification by binning each target into five equal-frequency bins (20% each).

## Ensembling
- Property-specific predictions from ModernBERT-base, CodeBERT, AutoGluon, and Uni-Mol 2 were combined, with Tg predictions adjusted by a constant bias offset scaled by the prediction standard deviation to correct for a systematic test label distribution shift.
- Target-specific weighted ensembles of six models were automatically weighted based on validation performance, followed by mean and/or standard deviation alignment of predictions to the training distribution.

## Insights
- General-purpose foundation models outperformed chemistry-specific ones for this polymer prediction task.
- Larger model sizes did not guarantee better performance and sometimes degraded scores.
- Models pretrained on code corpora yielded better results than those pretrained on natural language.
- AutoGluon's automated tuning significantly outperformed manually constructed tabular ensembles with far less compute.
- MD simulation-derived features improved GBDT models but were detrimental to AutoGluon.
- Leveraging external data with overlapping SMILES can reveal systematic target biases that must be corrected before training.
- TabM significantly outperformed tree-based models across most targets, making complex ensembling unnecessary.
- Conservative CV via molecular similarity filtering is essential to prevent data leakage in polymer property prediction.
- Focusing on robust cross-validation is critical when the public leaderboard relies on a small fraction of the test set.
- Stratifying validation folds by equal-frequency bins per target strongly correlates with public leaderboard performance.
- Automatically determining model weights per target based on validation scores yields better results than fixed weighting.

## Critical findings
- The leaderboard test labels contained a severe systematic bias that required a constant offset post-processing step to align predictions with the ground truth.
- Training on external datasets with dirty labels required a multi-step cleaning pipeline to extract useful signal rather than discarding them entirely.
- Exact-duplicate canonical SMILES deduplication was sufficient to prevent leakage, making Tanimoto similarity filtering largely redundant.
- MD simulation data generation was highly unstable and required a LightGBM classifier to predict and route problematic polymers to faster configurations.
- The test data appears to be modeled on a different temperature scale than the training data, causing a consistent distribution bias that external datasets also exhibit.
- Tree models (CatBoost and LGBM) exhibited high variance across folds and provided minimal score improvement over TabM, leading to their exclusion.
- Aligning prediction distributions to the training set via mean or standard deviation matching significantly improved public leaderboard scores for specific targets.
- Different targets required different distribution alignment strategies (e.g., mean-only for Tg vs. mean and std for FFV).

## What did not work
- GNNs (D-MPNN)
- GMM-based data augmentation strategy
- CatBoost and LightGBM (due to high fold variance and negligible gains over TabM)

## Notable individual insights
- rank 1 (1st Place Solution): General-purpose foundation models outperformed chemistry-specific ones for this polymer prediction task.
- rank 1 (1st Place Solution): Models pretrained on code corpora yielded better results than those pretrained on natural language.
- rank 8 (8th Place Solution | No Tg Post Processing): Leveraging external data with overlapping SMILES can reveal systematic target biases that must be corrected before training.
- rank 8 (8th Place Solution | No Tg Post Processing): TabM significantly outperformed tree-based models across most targets, making complex ensembling unnecessary.
- rank 20 (20th Place Solution: NeurIPS - Open Polymer Prediction 2025): Stratifying validation folds by equal-frequency bins per target strongly correlates with public leaderboard performance.
- rank 20 (20th Place Solution: NeurIPS - Open Polymer Prediction 2025): Automatically determining model weights per target based on validation scores yields better results than fixed weighting.

## Solutions indexed
- #1 [[solutions/rank_01/solution|1st Place Solution]]
- #8 [[solutions/rank_08/solution|8th Place Solution | No Tg Post Processing]]
- #20 [[solutions/rank_20/solution|20th Place Solution: NeurIPS - Open Polymer Prediction 2025]]

## GitHub links
- [RUIMINMA1996/PI1M](https://github.com/RUIMINMA1996/PI1M) _(reference)_ — from [[solutions/rank_01/solution|1st Place Solution]]
- [RadonPy/RadonPy](https://github.com/RadonPy/RadonPy) _(library)_ — from [[solutions/rank_01/solution|1st Place Solution]]
- [Duke-MatSci/ChemProps](https://github.com/Duke-MatSci/ChemProps) _(reference)_ — from [[solutions/rank_01/solution|1st Place Solution]]
- [jday96314/NeurIPS-polymer-prediction](https://github.com/jday96314/NeurIPS-polymer-prediction) _(solution)_ — from [[solutions/rank_01/solution|1st Place Solution]]
- [Jiaxin-Xu/POINT2](https://github.com/Jiaxin-Xu/POINT2) _(reference)_ — from [[solutions/rank_08/solution|8th Place Solution | No Tg Post Processing]]

## Papers cited
- [RankUp: Boosting Semi-Supervised Regression with an Auxiliary Ranking Classifier](https://arxiv.org/html/2410.22124v1)
- [POINT2: A Polymer Informatics Training and Testing Database](https://arxiv.org/abs/2503.23491)
