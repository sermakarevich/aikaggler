# ariel-data-challenge-2025: top public notebooks

The top-voted notebooks for the Ariel Data Challenge 2025 predominantly focus on building robust, physics-informed calibration pipelines for raw astronomical sensor data, followed by hybrid modeling strategies that combine analytical transit depth estimation with pre-trained residual neural networks. Contributors emphasize efficient signal processing techniques like correlated double sampling, variance-based weighting, and dynamic uncertainty quantification to bridge raw telemetry with machine learning. The community demonstrates that careful domain-specific feature engineering and lightweight inference can yield competitive leaderboard scores without extensive model training.

## Common purposes
- baseline
- inference
- utility
- tutorial

## Competition flows
- Loads raw sensor parquet files and calibration masks, applies gain/offset/dark/flat corrections and temporal binning, detects transit phases via gradient analysis, optimizes a constant spectral scaling factor using Nelder-Mead, and outputs a CSV submission with repeated predictions and fixed uncertainty.
- Reads raw instrument observation and calibration parquet files in chunks, applies a sequence of detector calibration and signal processing steps, and saves reduced data cubes as .npy arrays for downstream modeling.
- Calibrates and bins raw sensor data, estimates transit depth via custom optimization, feeds stellar parameters into pre-trained PyTorch ResNetMLPs for spectral flux prediction, combines results with estimated noise levels, and generates a submission CSV.
- Reads raw signal files and metadata, extracts 1D time-series and computes relative flux reduction features, trains a Ridge regression via cross-validation, and generates a submission with mean predictions and uncertainty estimates.
- Applies a multi-step physics-based calibration and spatial binning pipeline to extract 1D spectra, analytically fits transit breakpoints to derive a shift feature, and formats a submission with constant uncertainty.
- Preprocesses detector signals with phase detection and noise estimation, computes a 1D transit depth, feeds meta-features into a pre-trained ResNet-MLP for spectral predictions, applies smoothing and blending, and outputs a formatted submission CSV.
- Applies sensor-specific calibration and noise estimation, extracts physical parameters, feeds them into two pre-trained residual MLPs, and formats the outputs into a competition submission CSV.
- Estimates an initial transit depth via polynomial optimization, refines channel-specific depths using two pre-trained residual MLPs fed with stellar parameters, estimates channel uncertainties from light curve variance, and outputs a formatted submission CSV.
- Loads raw detector parquet files and calibration masks, applies sensor-specific calibration and binning, detects transit phases via gradient analysis, optimizes transit depth using Nelder-Mead with polynomial fitting, estimates per-planet uncertainties from variance ratios, and outputs a formatted CSV submission.
- Applies calibration and signal processing, estimates transit depth via an analytical polynomial-fitting model and a pre-trained PyTorch ResNetMLP, estimates sensor-specific noise, and generates a final submission CSV with mean predictions and uncertainties.

## Data reading
- Reads raw signal and calibration parquet files (dark, dead, flat, linear_corr) per planet/sensor.
- Loads ADC gain/offset and star metadata from CSV files.
- Uses chunked reading (pandas/polars/pyarrow) to prevent memory overflow.
- Utilizes glob to locate and organize training/calibration files.
- Reshapes loaded arrays to match specific detector dimensions (e.g., 32x356 for AIRS, 32x32 for FGS1).
- Loads preprocessed 1D spectra from numpy arrays.
- Reads calibration metadata from axis_info.parquet.
- Loads sample_submission CSVs for formatting.

## Data processing
- ADC gain/offset correction and non-negative clamping.
- Hot/dead pixel masking via sigma clipping or dead pixel maps (with notes on retaining hot pixels).
- Polynomial linear correction using inverse coefficients or Horner's method.
- Dark current subtraction scaled by integration time or alternating frame differencing.
- Correlated Double Sampling (CDS) via alternating frame/row subtraction.
- Flat-field correction by dividing by illumination maps or uniform masks.
- Spatial ROI cropping and wavelength axis cropping (e.g., pixels 39–321).
- Temporal binning via frame summation or averaging.
- Inverse-variance weighting and percentile clipping for signal stabilization.
- Savitzky-Golay filtering for smoothing and phase detection.
- Gradient-based phase detection to isolate in-transit and out-of-transit regions.
- Winsorization and median-based scaling for uncertainty/noise estimation.
- NaN interpolation via convolution-based weight normalization.
- Differential photometry via alternating mean subtraction.

## Features engineering
- Estimated transit depth derived from polynomial fitting or optimization.
- Stellar radius (Rs) and orbital inclination (i) from star metadata.
- Scaled transit depth (multiplied by 10,000) combined with stellar parameters into a 3D feature vector.
- Relative flux reduction features calculated from in-transit vs. out-of-transit flux windows.
- Transit ingress/egress breakpoints detected via signal derivatives.
- Sensor-specific relative noise estimates (sigma_fgs, sigma_air) normalized via variance ratios.
- Multiplicative shift parameters extracted from piecewise polynomial fits.

## Models
- ResNetMLP (custom residual MLP with 80 blocks, varying hidden dimensions like 32 or 128).
- ResNetMLP2 (custom residual MLP with 120 blocks, varying hidden dimensions like 128 or 282).
- TransitModel (custom 1D estimator using gradient-based phase detection and Nelder-Mead optimization).
- Ridge regression.
- Analytical polynomial-fitting model (degree 3) with Nelder-Mead optimization.

## Frameworks used
- numpy
- pandas
- scipy
- scikit-learn
- pytorch
- astropy
- pqdm
- tqdm
- polars
- matplotlib
- seaborn
- plotly.express

## Loss functions
- None (models are pre-trained).
- Mean absolute error between polynomial fit and signal (used for optimization objective).

## CV strategies
- 5-fold cross-validation via sklearn's cross_val_predict

## Ensembling
- Blends raw AIRS spectral predictions with a lightly smoothed version using a 0.65:0.3 weight ratio.
- Combines baseline polynomial transit depth with channel-specific predictions from two pre-trained residual MLPs.
- Merges predictions from analytical polynomial-fitting, transit depth ResNetMLP, and spectral ResNetMLP with estimated noise values.

## Insights
- Careful sensor calibration (dark, flat, linear correction) is essential to extract meaningful transit signals from raw detector data.
- Transit ingress and egress phases can be robustly detected using gradient analysis on smoothed, binned photometric data.
- A single optimized scaling constant can approximate the full multi-wavelength spectrum in this simplified physics-based baseline.
- Temporal binning and spatial region-of-interest cropping drastically reduce computation while preserving the core transit morphology.
- Calibration steps should be treated as a flexible recipe rather than mandatory rules.
- Time binning and wavelength cropping significantly reduce memory footprint while preserving relevant signal.
- CDS effectively removes baseline drift by differencing alternating readouts.
- Flat fielding corrects pixel-to-pixel quantum efficiency variations.
- Careful calibration and variance weighting are critical for handling raw astronomical sensor data.
- Estimating per-planet noise levels using out-of-transit and in-transit variance ratios improves uncertainty quantification.
- Pre-trained shallow residual networks can effectively map stellar parameters to spectral fluxes when combined with robust signal processing.
- Differencing even and odd sensor frames effectively isolates the transit signal from raw imaging data.
- Two simple relative flux reduction features capture nearly all predictive variance, achieving an R² of 0.97 with Ridge regression.
- The competition metric is a normalized Gaussian Log Likelihood score that penalizes uncertainty calibration as well as point predictions.
- Physics-based calibration pipelines can be efficiently implemented with vectorized tensor operations and multiprocessing.
- Analytical signal processing and curve fitting can replace complex ML models for fast, interpretable baselines.
- Assigning a constant uncertainty across all wavelengths is sufficient to achieve a strong initial Gaussian Log Likelihood score.
- Hot and dead pixels are dynamically masked using sigma clipping on dark frames rather than relying solely on provided dead pixel maps.
- The detector uses Correlated Double Sampling, requiring alternating frame differencing to extract the true signal.
- NaN interpolation is handled via a custom convolution-based weight normalization to preserve spatial structure during gap filling.
- Robust sigma estimation benefits from conservative clipping bounds and a slight global scaling factor to prevent noise underestimation.
- Savitzky-Golay smoothing on spectral predictions can improve leaderboard scores without requiring complex model retraining.
- Combining physics-inspired transit depth estimation with lightweight neural networks provides a strong, efficient baseline for atmospheric characterization tasks.
- Expanding sigma clipping bounds (e.g., 0.85–1.30) and applying a 1.04 multiplier prevents underestimation of noise in the final submission.
- Hot pixel masking and alternating dark current scaling are critical for stabilizing raw detector signals before binning and weighting.
- Custom signal calibration and correlated double sampling are critical for extracting clean photometric signals from raw telescope telemetry.
- Phase detection combined with gradient analysis reliably identifies transit boundaries for accurate noise estimation.
- Physical parameters like stellar radius and inclination can be effectively used as direct inputs to a residual MLP to predict refined transit depths.
- Retaining hot pixels instead of masking them preserves signal information and speeds up processing.
- Uncertainty estimation benefits from a soft median-based multiplier on OOT/IN variance ratios rather than raw noise estimates.
- Simple tabular features (transit depth, stellar radius, inclination) are sufficient to drive deep residual networks for channel-specific refinement.
- Hot pixels should be monitored but intentionally retained in the data rather than masked to avoid losing valid signal.
- Dynamic uncertainty estimation using out-of-transit and in-transit variance ratios provides more realistic error bars than fixed sigma values.
- Inverse-variance weighting with percentile clipping prevents dominant channels from skewing the binned signal.
- Parallel processing with pqdm significantly accelerates per-planet sensor calibration.
- Efficient parallel processing with pqdm significantly speeds up sensor calibration.
- Variance-based weighting and median normalization prevent noise estimation from being skewed by outliers.
- Combining physics-based phase detection with polynomial fitting provides a robust initial transit depth estimate.
- Pre-trained lightweight residual networks can effectively refine predictions using minimal stellar parameters.

## Critical findings
- All 1100 training planets map to exactly one unique star each, with no repeated observations.
- A significant portion of the dataset lacks required calibration files (2200 sets missing).
- Flux values exhibit very low variability across the training set (mean ~0.0146, std ~0.0105).
- Explicitly notes that hot pixels should not be masked during calibration, as doing so discards useful signal and slows computation.
- Fixed sigma values are replaced with data-driven multipliers clipped to ±20-25% of the baseline to prevent overconfident or overly conservative uncertainty estimates.

## Notable individual insights
- votes 390 (NeurIPS: Non-ML Transit Curve Fitting): A single optimized scaling constant can approximate the full multi-wavelength spectrum in this simplified physics-based baseline.
- votes 257 (NeurIPS ADC'25 Intro Training): Differencing even and odd sensor frames effectively isolates the transit signal from raw imaging data.
- votes 212 (ARIEL25 ⚡ Baseline Submission 1D modelling): Hot and dead pixels are dynamically masked using sigma clipping on dark frames rather than relying solely on provided dead pixel maps.
- votes 160 (LB:0.339 Very fast with hot pixels enabled V2): Explicitly notes that hot pixels should not be masked during calibration, as doing so discards useful signal and slows computation.
- votes 144 (Very fast with hot pixels enabled): Fixed sigma values are replaced with data-driven multipliers clipped to ±20-25% of the baseline to prevent overconfident or overly conservative uncertainty estimates.
- votes 186 (resnet for airs finetuned LB0.369): Expanding sigma clipping bounds (e.g., 0.85–1.30) and applying a 1.04 multiplier prevents underestimation of noise in the final submission.
- votes 267 (0.374 LB score | bronze medal): Estimating per-planet noise levels using out-of-transit and in-transit variance ratios improves uncertainty quantification.

## Notebooks indexed
- #390 votes [[notebooks/votes_01_vitalykudelya-neurips-non-ml-transit-curve-fitting/notebook|NeurIPS: Non-ML Transit Curve Fitting]] ([kaggle](https://www.kaggle.com/code/vitalykudelya/neurips-non-ml-transit-curve-fitting))
- #364 votes [[notebooks/votes_02_gordonyip-calibrating-and-binning-ariel-data/notebook|Calibrating and Binning Ariel Data ]] ([kaggle](https://www.kaggle.com/code/gordonyip/calibrating-and-binning-ariel-data))
- #267 votes [[notebooks/votes_03_antonoof-0-374-lb-score-bronze-medal/notebook|0.374 LB score | bronze medal]] ([kaggle](https://www.kaggle.com/code/antonoof/0-374-lb-score-bronze-medal))
- #257 votes [[notebooks/votes_04_ahsuna123-neurips-adc-25-intro-training/notebook|NeurIPS ADC'25 Intro Training ]] ([kaggle](https://www.kaggle.com/code/ahsuna123/neurips-adc-25-intro-training))
- #212 votes [[notebooks/votes_05_ilu000-ariel25-baseline-submission-1d-modelling/notebook|ARIEL25 ⚡ Baseline Submission 1D modelling]] ([kaggle](https://www.kaggle.com/code/ilu000/ariel25-baseline-submission-1d-modelling))
- #186 votes [[notebooks/votes_06_seowoohyeon-resnet-for-airs-finetuned-lb0-369/notebook|resnet for airs finetuned LB0.369]] ([kaggle](https://www.kaggle.com/code/seowoohyeon/resnet-for-airs-finetuned-lb0-369))
- #186 votes [[notebooks/votes_07_qaedtgyh-a-simple-resnet-solution/notebook|a simple resnet solution]] ([kaggle](https://www.kaggle.com/code/qaedtgyh/a-simple-resnet-solution))
- #160 votes [[notebooks/votes_08_yusuketogashi-lb-0-339-very-fast-with-hot-pixels-enabled-v2/notebook|LB:0.339 Very fast with hot pixels enabled V2]] ([kaggle](https://www.kaggle.com/code/yusuketogashi/lb-0-339-very-fast-with-hot-pixels-enabled-v2))
- #144 votes [[notebooks/votes_09_antonsibilev-very-fast-with-hot-pixels-enabled/notebook|Very fast with hot pixels enabled]] ([kaggle](https://www.kaggle.com/code/antonsibilev/very-fast-with-hot-pixels-enabled))
- #137 votes [[notebooks/votes_10_antonoof-0-359-score-bronze/notebook|0.359 score bronze]] ([kaggle](https://www.kaggle.com/code/antonoof/0-359-score-bronze))
