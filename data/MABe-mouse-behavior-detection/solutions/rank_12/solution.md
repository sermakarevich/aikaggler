# 13th Public/12th Private Solution

- **Author:** Antonina Dolgorukova
- **Date:** 2025-12-16T13:28:29.103Z
- **Topic ID:** 663108
- **URL:** https://www.kaggle.com/competitions/MABe-mouse-behavior-detection/discussion/663108

**GitHub links found:**
- https://github.com/T0chka/mabe_12th_place_solution

---

We thank the MABe Challenge organizers for running an engaging competition and for providing detailed, timely clarifications during the early stages.

# High‑level idea
-	Formulate the problem as a set of per-action binary tasks, trained separately for each lab
-	Train multiple models under diverse conditions, including tuned action-specific feature subsets, negative downsampling schemes, and hyperparameters 
-	Use video-level cross-validation and ensemble the resulting fold models at inference
-	Sample background frames to control the ratio of true negatives and unlabeled data
-	Calibrate predictions and adjust log-likelihood-ratio scores with training priors
-	Apply postprocessing, including temporal smoothing, per-action threshold fitting, and argmax-based action selection

# EDA notebooks
Several exploratory notebooks were used to understand the structure, limitations, and implicit assumptions of the dataset. These analyses informed key modeling choices, including task formulation, negative sampling, calibration strategy, and postprocessing.
- [Overall data overview](https://www.kaggle.com/code/antoninadolgorukova/mabe-an-exploratory-data-safari ) 
- [Deeper analysis of data constraints and their modeling implications](https://www.kaggle.com/code/antoninadolgorukova/mabe-data-constraints-modeling-implications)
- [Data quality checks and identified issues](https://www.kaggle.com/code/antoninadolgorukova/mabe-sanity-checks-data-issues)

# Source code
The full source code required to reproduce the solution is available at [the repo](https://github.com/T0chka/mabe_12th_place_solution)

# Data assembly
Data were prepared as a combination of all frames with tracks for each video_id and each (agent_id, target_id) pair from the whitelist (behaviors_labeled column in train/test csv files). Frames annotated with a given behavior were treated as positive examples for that action; unlabeled frames were treated as background.

# Shared feature space
All models reuse a common feature representation composed of three groups:
1. Invariant, agent-centric geometry and kinematics (relative distances, angles, approach/retreat cues, motion alignment, and time since contact).
2. Single-mouse descriptors (forward and lateral velocities, accelerations, total speed, turning rate, path curvature, motion smoothness, body length, ear gap, and related stability metrics).
3. Pairwise distances between selected body parts within each mouse and across the agent-target pair, computed either on raw landmarks or on aggregated body regions obtained by averaging upper and lower body parts.

Invariant and single-mouse descriptors were normalized by agent body length.

For a subset of models, the base feature set was augmented with windowed temporal features computed for a tuned subset of base variables. Specifically, the pipeline appends rolling mean summaries over three lagged windows, providing history-aware covariates.

# Models
The final ensemble includes 5 model families: XGBoost, LightGBM, Catboost, PyBoost, and TABM.
Boosting-based models were trained separately for each lab and each action. In LightGBM, positive frames were weighted inversely to the length of the corresponding action run, preventing long continuous events from dominating the loss. Because raw boosting margins are not directly comparable across actions and labs, we applied a two-step Platt calibration with prior correction, learned from out-of-fold (OOF) predictions. 
TABM models were trained per lab using a one-vs-rest formulation with independent sigmoid heads for each action.

The ensemble uses weigths fitted on OOF, and single-model scores are:
- XGBoost: public 0.521, private 0.489
- Catboost: public 0.517, private 0.491
- PyBoost: public 0.512, private 0.488
- TABM: public 0.506, private 0.487
- LightGBM: public 0.492, private 0.477

# Subsampling negatives
For boosting-based models, unlabeled frames were partitioned into two groups with respect to a given action:
1. Real negative frame: has an annotation with a different action, or is unlabeled while the action is present in the video’s whitelist (meaning it could have been annotated, the action just did not happen).
2. Maybe negative frame: is unlabeled and the action is absent from the video’s whitelist (meaning the action could have occurred, but was outside the annotator’s labeling scope)

The final training set for a fold consisted of all positives plus negatives sampled by video from the two pools above.

For TabM, unlabeled frames were downsampled by (video, agent, target) triplets. Since the models were trained in a multiclass setting, this distinction between negative types was not applied.

# Cross-validation
Folds were built at the video level. Each model applied 5-fold cross-validation and produced complete out-of-fold (OOF) predictions for the entire training set. This resulted in a good CV-LB correlation. All 5 model families ensemble their fold-specific models at inference.

# Postprocessing
The frame-level predictions were filtered to the whitelist, keeping only valid (video, agent, target, action) combinations. Smoothing was then applied using rolling windows whose sizes correspond to the median action durations observed in the training data. Action-specific thresholds were fitted on OOF, and applied after smoothing; a single action per frame was selected using argmax. Consecutive frames with the same chosen action were then merged into intervals, forming the final submission.