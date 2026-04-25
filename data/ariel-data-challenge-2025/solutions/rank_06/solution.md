# 6th Place Solution Batman-Minuit

- **Author:** Vicens Gaitan
- **Date:** 2025-09-25T13:59:48.953Z
- **Topic ID:** 609276
- **URL:** https://www.kaggle.com/competitions/ariel-data-challenge-2025/discussion/609276
---

# Ariel 25 Challenge: A Three-Stage Pipeline for Robust Transmission Spectra

I'd first like to thank the organizers for putting on such an interesting competition. They deserve credit both for their scientific rigor and for the intricate puzzles they used to generate the data—some of which I still haven't managed to solve.

##1. Summary

This solution presents a comprehensive, three-stage pipeline designed to extract high-fidelity transmission spectra from the FGS1 and AIR-CH0 Ariel instrument data. The approach combines physics-based modeling with a machine learning-inspired post-processing framework to tackle the key challenges of instrumental systematics and uncertainty estimation.

1. Stage 1: GPU-Accelerated Preprocessing. Raw sensor data is converted into clean, calibrated light curves using a CuPy-based pipeline. This stage handles standard instrumental corrections, including non-linearity, dark current, flat-fielding, and jitter, all performed efficiently on the GPU.

2. Stage 2: Hierarchical Transit Fitting. I use the [**batman**](https://lkreidberg.github.io/batman/docs/html/index.html) transit modeling library and the [**iminuit**](https://scikit-hep.org/iminuit/) optimizer to perform a multi-step fit. A robust global model is first fit to the combined FGS and AIRS light curves to constrain key orbital parameters. This is followed by a 2D drift correction and a wavelength-by-wavelength fit to extract the initial transmission spectrum.

3. Stage 3: Cross-Validated Post-Processing Ensemble. The initial spectra are refined using an ensemble of models trained with Grouped K-Fold Cross-Validation. This final stage blends a PCA-regularized signal with a smoothed signal and employs a parameterized model to predict the final uncertainties (sigmas), optimizing all parameters directly against the competition score.


This multi-stage design ensures that physical constraints are respected in the initial fit, while the final model has the flexibility to learn and correct for residual systematic errors and produce well-calibrated uncertainties.

--------------------------------------------------------------------------------
## 2. Methodology

###Stage 1: Preprocessing Raw Data
The first step in the pipeline is to transform the raw sensor readouts into scientifically useful light curves. This entire process is executed on the GPU using CuPy for maximum efficiency.

Key Preprocessing Steps:

• Initial Calibrations: The pipeline begins by applying standard instrumental corrections. This includes an ADC correction for gain and offset, a 5th-degree polynomial correction for detector non-linearity (apply_linear_corr_fast), and subtraction of a scaled master dark frame (clean_dark).

• Flat-Fielding & Masking: A master flat frame is applied to correct for pixel-to-pixel sensitivity variations. Pixels identified as "hot" (via sigma-clipping the dark frame) or "dead" are masked.

• Jitter Correction: For the FGS1 sensor, we apply a center-of-mass  regression to correct for flux variations caused by image motion on the detector. The flux is de-correlated from the normalized X and Y  positions of the stellar image. While a PCA-based jitter correction method was developed for the AIRS sensor, it was found to be less effective and is disabled in the final pipeline.

![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F254638%2Fc209358dd02c309ac5067f60b7903724%2Fjitter.png?generation=1758796485996693&alt=media)



• Signal Extraction & Cleaning:

   
* For FGS1, the final flux is extracted using simple aperture photometry. PSF photometry gives lower SNR signals, probably due to the jitter.
* For AIRS, the signal is extracted from the central detector region, and a background signal, calculated from the top and bottom edges of the detector, is subtracted. This is not optimal because the spectroscopy signal extends i a difractive way to the whole sensor, making dificult to  identify the aditive frequency  dependent background and  making necessary a global  scale factor and an spectrum slope correction in the post processing stage
![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F254638%2F7f6d6424c53a3475488d53772ec8b701%2Fair_psf.png?generation=1758806615604977&alt=media)
* A spike-cleaning algorithm  is applied to the final time series to remove cosmic ray hits by identifying and replacing outliers in the frame-to-frame difference.

    
• Binning: To improve the signal-to-noise ratio, the final time series for both instruments are binned by averaging consecutive frames.

###Stage 2: Physics-Based Spectral Fitting

With clean light curves for both instruments, we extract the initial transmission spectrum. This process is parallelized using pqdm to efficiently process all observations.

Hierarchical Fitting Strategy:

**1. Global Combined Fit**: We first fit the FGS light curve and the wavelength-averaged AIRS light curve **simultaneously** using fit_combined_curves. This crucial step provides robust constraints on the shared physical parameters of the system (e.g., orbital period per, inclination inc, transit time t0), which are initialized from the provided star_info.csv file.  The transit is modeled using batman, asuming a **quadratic limb darkening model**, using 2 parameters, and instrumental trends are modeled with a polynomial baseline in time.

![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F254638%2F04530cea65be4483a14e6effec93bbe2%2Ffits.png?generation=1758796534186505&alt=media)

**2. Refined Fit & Drift Correction**: The results from the first fit are used to initialize a second, refined fit on normalized data. Subsequently, a 2D instrumental drift model is fit to the out-of-transit portion of the AIRS data cube. This model (Drift class) consists of two 4th-order polynomials, one for the frequency and one for the temporal axis, and it corrects for slow-varying systematic patterns across the detector. The data cube is then divided by this drift model.
**3. Wavelength-by-Wavelength Fit**: The final step is to measure the transit depth in each individual wavelength channel of the drift-corrected AIRS data. The fit_w function iterates through the channels, fitting the data within a small moving window (deltaw=2) to boost signal-to-noise. In this fit, most orbital parameters are fixed to the values from the global fit, and only the transit depth (dipa) and baseline normalization (A0a) are allowed to vary.
This process results in the "raw" spectrum  and its associated uncertainties , which serve as the primary inputs for our final modeling stage.

![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F254638%2F990115e5eb8fa6e302f721f552a1cb94%2Fspectrum.png?generation=1758796611221256&alt=media)

## Stage 3: Post-Processing and Uncertainty Modeling

The final stage of the solution is a post-processing model that refines the raw spectra and, critically, learns a robust model for the final uncertainties. This is where the bulk of the "machine learning" occurs.

Model Architecture (build_preds): The model generates a final prediction by blending two different representations of the raw spectrum (preds1):

**PCA-Regularized Prediction (preds0)**: The raw spectrum undergoes a slope correction and is then projected onto a pre-defined PCA basis (components.npy). This de-noises the spectrum and imposes a strong regularization prior, capturing the most common modes of variation.
**Smoothed Prediction (preds2)**: The raw AIRS spectrum is smoothed using a Savitzky-Golay filter to reduce high-frequency noise while preserving broader spectral features.

The final mean prediction is a weighted average of these two models: preds = app0 + (1-ap)preds2, where ap is a learned parameter.

**Sigma Model**: Predicting accurate uncertainties is key to maximizing the competition score. I developed a complex, empirically-derived formula to model the final sigma for each data point. This formula combines multiple sources of uncertainty:

 
* a1* err^e1 : The propagated error from the initial fit, with a learned exponent.
* a01/p1^e2  : A term related to the signal magnitude.
* b0 * sigma0: A term proportional to the raw signal's volatility.
* ae * np.abs(p0-preds2): A term that increases uncertainty where the PCA and smoothed models disagree, capturing model 
uncertainty or unknow spectra.

Finally ther is a sigma  adjustment  samples with very few or very many out-of-transit points, or very differnt dip por FGS1 and AIR-CH0

#Training and Ensembling:

  **Objective**: The free parameters of the model (ap, e1, e2, a1, a0, b0, ae, etc.) are optimized using iminuit to directly maximize the competition score.

  **Cross-Validation**: To build a robust model that generalizes well, we employ a 10-fold Grouped K-Fold Cross-Validation strategy, using planet_id to group the data. This ensures that all observations of a single planet are kept within the same fold, preventing data leakage.

  **Calibration**: Within each fold, after the primary parameters are optimized, we calculate a wavelength-dependent calibration factor (alpha) that scales the predicted sigmas to best match the variance observed in the training data. This factor is then applied to the validation set predictions.
  
   **Ensembling**: The final submission is generated by averaging the calibrated predictions and sigmas from the 10 models trained during the cross-validation process. This ensembling technique reduces variance and improves the final score.

The entire training process, including the parallel optimization of each fold, is managed by the CV_model function in the model.py file.

--------------------------------------------------------------------------------
4. What Did Not Work
I attempted to post-process the spectrum and sigma values using Gaussian Processes and Ridge Regression, but the results were worse than using the blending of a smoothed/PCA-regularized model and the empirical sigma formula.

In the final weeks, I spent time trying to isolate the additive foreground noise in the AIRS data for each sample, which would have eliminated the need for the global factor and slope compensation. However, I was unable to find a way to de-correlate signal and noise. I observed a clear structure in the spectroscopic PSF, but all attempts to extract the noise yielded incorrect values that failed to compensate for the systematics.

5. A Comment on Efficiency
The choice of batman and iMinuit was driven by speed. iMinuit is a Python wrapper for the C++ port of the MINUIT Fortran code (developed by Fred James in the '70s; [Paper](https://www.sciencedirect.com/science/article/abs/pii/0010465575900399?via%3Dihub), [Wikipedia](https://en.wikipedia.org/wiki/MINUIT)), and it is typically faster than generic SciPy optimization routines. Additionally, the use of GPU acceleration for data processing via CuPy results in a very fast pipeline because all processes are fully parallelized. The submission process on Kaggle using a P100 instance takes less than 1.5 hours, and on a machine with 100 cores and 4  nVidia L40s GPUs, the full preprocess, fitting, and post-process takes around 5 minutes. The post-process parameter adjustment can also be done in minutes.

--------------------------------------------------------------------------------
##6. Conclusion
This solution demonstrates the power of a hybrid approach, blending physics-informed transit modeling with a flexible, data-driven post-processing framework. By first extracting a reasonable physical spectrum and then refining it with a model trained to optimize the specific competition metric, we can effectively correct for complex instrumental systematics and produce highly accurate, well-calibrated transmission spectra. The use of GPU acceleration, parallel processing, and a robust cross-validation scheme ensures that this complex pipeline remains computationally efficient and resistant to overfitting.