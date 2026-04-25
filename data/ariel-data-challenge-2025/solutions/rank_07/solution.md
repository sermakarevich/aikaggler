# 7th Place Solution

- **Author:** Horikita Saku
- **Date:** 2025-09-25T02:01:27.087Z
- **Topic ID:** 609210
- **URL:** https://www.kaggle.com/competitions/ariel-data-challenge-2025/discussion/609210
---

# Introduction
We ( @horikitasaku and @takaito) would like to express our sincere gratitude to the organizers for such an outstanding competition.
Special thanks also for staff members like Sohier Dane @sohier for all their hard work and support.

This was truly a well-designed competition, and it clearly reflected the professionalism and scientific rigor of the organizing team (e.g., @gordonyip and colleagues). Physicists have always been the group of people I respect the most.


**Our approach can be summarized in the following stages:**

* **Step0:** Preprocessing
* **Step1:** Signal, transit, and feature extraction; calculation of the base `wl_preds` (Horikita)
* **Step2:** Difference correction with NN and base sigma prediction (takaito)
* **Step3:** Sigma scale adjustment using a gradient boosting model (Horikita)
* **Step4:** Pseudo Labeling (takaito)

# Step0: Preprocessing

Here we basically followed [Pascal’s notebook](https://www.kaggle.com/code/ilu000/ariel25-quick-data-prep-improved) almost as it is, and processed both `obs0` and `obs1`.

# Step1: Signal, Transit, Features – Calculation of the base `wl_preds` (CV: \~0.42, LB:-0.45)

## Step1.1: Phase Detector

The core idea of transit detection is **using the extrema of the signal’s gradient**.
I assumed that the transit boundaries would appear as characteristic features on the derivative curve. The overall process was as follows:

* **Smoothing**: Smooth the signal and its gradient to remove high-frequency noise.
* **Outlier removal**: Clip gradient values beyond 5σ, set the edges to zero, and then apply additional Gaussian smoothing.
* **Threshold scanning**:

  * Negative threshold (neg\_thr) → *t1, t2* (ingress)
  * Positive threshold (pos\_thr) → *t3, t4* (egress)
* **Local sampling**: Sample around (t1, t2) and (t3, t4), and compare the baseline with the transit segment. If the ingress/egress depth is too shallow, shift the boundary toward the start or end of the sequence.
* **Post-processing**: Fix abnormal samples. In the initial version this was not implemented, but after the dataset update I observed bugs from `Phase Detector` . Upon analysis, I found many abnormal signals, so this step was added.
![p1](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F11676771%2F2b03340d9a21e83b7f313e3aaa710b6b%2F438737d84bb3b050faf0ae4f1adb8fd6.png?generation=1758761787946320&alt=media)

![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F11676771%2F3d63b1fc0686b22ebb403229af6eb2d5%2F6d1506b7-957f-4bc2-8810-1992f705f8a8.png?generation=1758761822135400&alt=media)

## Step1.2: Physics-based modeling and feature extraction 

Here I referenced **two top solutions from Ariel 2024**:

* [xomox (10th place)](https://www.kaggle.com/competitions/ariel-data-challenge-2024/writeups/xomox-10th-place-solution)
* [cnumber (1st place)](https://www.kaggle.com/competitions/ariel-data-challenge-2024/writeups/c-number-daiwakun-1st-place-solution)

Unfortunately, this part was almost useless for sigma prediction.
In my solo experiments, using these sigmas often reduced the final score by about **0.1 below the theoretical upper bound score**. 

So the real value here was in generating the **base `wl_preds`** and **features derived from the physical process**. We adopted two main patterns:

* **Pattern 1 (main)**: Based on *xomox*’s pipeline, with several modifications:

  * Restricted the polynomial degree of the baseline to 2–3 (higher orders caused instability in some signals).Reducing the order of the baseline polynomial resulted in a roughly 0.01 improvement for me.
  * Adjusted binning and sampling settings.
  * Introduced a more complex Gaussian Process (GP). In our case, the kernel design combined multiple components — RBF terms with different length scales, a periodic kernel to capture repeating structures, a Matérn kernel for local smoothness, a linear kernel tied to wavelength features, and a white noise kernel for robustness. What turned out to be very interesting is that although the GP itself was not directly effective for improving sigma prediction, it played an unexpectedly strong role in stabilizing all the downstream tasks, including the linear correction I tested independently and the NN model developed by takaito. 
  As a result, we obtained stable `wl_preds`, reconstructed signals, ideal transit models, and physics-derived features.
![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F11676771%2Faa1b5c478d7feedae89b9eab3929bd7f%2Fd718ac9a-1f1f-44da-90c4-7a78c229a52b.png?generation=1758762273752078&alt=media)

* **Pattern 2 (secondary)**: Implemented *cnumber*’s approach with only minor changes. This provided an independent set of `wl_preds` and additional features.

# Step2: Difference correction with NN and base sigma prediction (CV: \~0.56x, LB: \~0.55x)

Inspired by methods that worked well in ADC2024, instead of directly predicting `wl`, we trained deep learning models to predict the difference `wl_diff` between the roughly computed `wl_preds` from **step1** and the ground truth `wl_true`. In other words, the NN performs a **residual correction**.

## Basic information

### Model inputs/outputs

* `wl_preds` from Step1 are used as the base predictions.
* The target variable is `wl_diff (= wl_true - wl_preds)`.
* Input features include signal, transit, mask, position, and various extracted features.

### Model architectures

* Multilayer Perceptron (MLP)
* BiGRU + CNN + MLP
* BiLSTM + CNN + MLP
* BiLSTM + CNN + BiLSTM

## Key techniques

* Used **quantile regression** so the model can also predict sigma.
* Applied **Adversarial Weight Perturbation (AWP)**.
* Performed data augmentation by flipping signal and transit sequences along the time axis.
* Added noise to the data for further augmentation.

## What did not work

* More complex model architectures did not bring improvements.

# Step3: Sigma scale adjustment with Gradient Boosting (LB +\~0.01)

Intuitively, Step2 already provided per-wavelength σ (`step2_sigma`), but there were still systematic shifts remaining at the sample level.
To handle this, we summarized the shift into a single scalar **scale factor `a`**, learned separately for FGS (wl0) and AIRS (all other bands). The idea was simple:

* Correct only the overall scale
* Keep the relative shape across wavelengths unchanged

Concretely:

1. **Boundary search**: Used `minimize_scalar(method='bounded')` to find the optimal `a`.
2. **Discretization**: Rounded `a` to 0.01 increments to prevent overfitting to random noise in the evaluation metric.

Finally, we trained a **Gradient Boosting model** to learn this `a` and applied it for scaling, which stabilized sigma predictions.

```python
sigma_scaled[:, 0] = (
    np.array(mean_a_preds_fgs)
    * step2_sigma[:, 0]
).clip(CFG.MIN_SIGMA)

# AIRS
sigma_scaled[:, 1:] = (
    (a_preds_airs).reshape(-1, 1)
    * step2_sigma[:, 1:]
).clip(CFG.MIN_SIGMA)
```

# Step4: Pseudo Labeling (LB: \~+0.004)

To boost the leaderboard score, we retrained the model using the test data along with their predicted values.
To avoid overfitting, we only used the predictions for `wl_diff` and did **not** use the predicted sigma values.

(Step2 involved checking a large number of learning curves, and we observed that even when MSE overfit the training data, the validation performance rarely deteriorated. This gave us confidence to apply pseudo labeling here.)

Below are the trends we observed:

* **MSE transition**: Almost no folds showed clear overfitting.
* **Competition metric transition**: Tended to overfit more easily.
![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F11676771%2F530edb9c4e96db81340ba3be9c937a2e%2F584f47bf-182c-43d8-8cb6-e3f223a3f6a8.png?generation=1758762783767691&alt=media)
![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F11676771%2F4391760215c0f2ee849c24bdde20ad0c%2Fffd8de63-7407-4fba-bcee-86691107dbec.png?generation=1758762800260659&alt=media)