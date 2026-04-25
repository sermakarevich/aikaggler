# GRU Solution

- **Author:** WOOSUNG YOON
- **Date:** 2025-12-04T15:35:23.957Z
- **Topic ID:** 651652
- **URL:** https://www.kaggle.com/competitions/nfl-big-data-bowl-2026-prediction/discussion/651652
---

# Overview

##  1. Loss Function: Robustness & Smoothness

![loss_function](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F4808143%2Fa5492518a54474bf9635641ce30560af%2Floss_funciton.jpg?generation=1764859413534359&alt=media)

The training objective balances precise trajectory tracking with physically plausible motion.

### **1.1 Robust Position Loss (Cauchy Loss)**
Real-world tracking data contains noise and outliers. Standard Mean Squared Error (MSE) over-penalizes these anomalies, which can destabilize training.
To address this, I use a **Cauchy loss**:
* **Mechanism:** It grows logarithmically rather than quadratically for large errors. 
* **Effect:** This effectively down-weights outliers, forcing the model to focus on fitting the bulk of realistic trajectories rather than chasing noise.

### **1.2 Weak Velocity Regularization (Random Kernels)**
Frame-by-frame position regression can result in jittery, unnatural motion.
Instead of enforcing strict pointwise velocity matching, I employ **random kernel smoothing**:
* **Mechanism:** Velocities are convolved with random kernels before calculating the error. 
* **Effect:** This penalizes discrepancies in the *overall motion trend* while permitting local variations. It ensures the predicted path is physically smooth without being over-constrained.

---

## 2. Model Architecture

![model](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F4808143%2Fb796b2c3472b00df48f6a167703306ec%2Fmodel.jpg?generation=1764859449086826&alt=media)

The core model is a **Spatio-Temporal GRU** employing a "Coarse-to-Fine" decoding strategy to ensure long-term stability.

### **2.1 Temporal Encoder (Backbone)**
* **Component:** Gated Recurrent Unit (GRU).
* **Role:** Processes historical motion to generate a context vector summarizing player movement and pass type.

### **2.2 Spatial Encoder (Neighbor Attention)**
* **Component:** Gated Cross-Attention.
* **Role:** Captures interactions with other agents (receivers, defenders).
* **Gating:** A learned gate dynamically adjusts the influence of neighbors—trusting spatial context when relevant (e.g., tight coverage) and ignoring it when it adds noise.

### **2.3 Coarse-to-Fine Decoder**
To prevent error accumulation (drift) over long prediction horizons, the model predicts trajectories hierarchically:
1.  **Level 0 (T/8):** Predicts a low-resolution "global shape."
2.  **Levels 1–2 (T/4, T/2):** Interpolates and refines intermediate points.
3.  **Level 3 (T):** Generates full-resolution residuals.

This hierarchical approach ensures the model commits to a consistent global trajectory before refining local details.

---

## 3. Ensemble Strategy: Genetic Algorithm

![ensemble](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F4808143%2Fb9c0a586fb7df6115fa824aabcc39c99%2Fensemble.jpg?generation=1764860420715929&alt=media)

To maximize generalization, I use a **Genetic Algorithm (GA)** to optimize the ensemble weights rather than relying on simple averaging.

### **3.1 Pool Generation**
* **Input:** Out-of-Fold (OOF) predictions are collected from multiple base models trained with different seeds and hyperparameters.

### **3.2 Optimization via GA**
* **Goal:** Find a subset of models that are **complementary** (uncorrelated errors).
* **Process:**
    1.  **Initialization:** Randomly generate subsets ("teams") of models.
    2.  **Evolution:** Iteratively select the best-performing subsets based on validation RMSE.
    3.  **Mutation/Crossover:** Combine and perturb subsets to explore the search space.

### **3.3 Final Blending**
The final submission is a weighted average of the top-performing subsets identified by the GA. This ensures the ensemble balances diversity and accuracy.

---

## 4. Conclusion

I learned a lot. Thank you.



