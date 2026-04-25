# [33rd solution] Transformer+Augmentation Tricks 

- **Author:** hsiaosuan
- **Date:** 2025-12-04T02:15:52.440Z
- **Topic ID:** 651530
- **URL:** https://www.kaggle.com/competitions/nfl-big-data-bowl-2026-prediction/discussion/651530
---


#Chiral Invariance & STTRE: A Physics-Informed Approach

## 1\. Executive Summary

Our solution achieves a Public LB of **0.519**  by prioritizing **physics-informed inductive biases** over brute-force model scaling. We successfully reproduced and adapted the **STTRE (Spatio-Temporal Transformer with Relative Embeddings)** architecture, originally proposed for multivariate time series forecasting [1], to the domain of NFL player tracking.

Our key contribution lies in a novel **Chiral Augmentation** strategy that forces the model to learn field symmetry, significantly reducing overfitting. While our model excels at momentum-based trajectories, our error analysis reveals a fundamental limitation in handling "sharp cuts," pointing towards a need for play-outcome classification.

## 2\. Data Engineering: Symmetry & Chirality

The core of our improvement came from reducing the dimensionality of the problem by exploiting the geometric symmetries of the football field.

### 2.1 Unifying Play Direction (Halving the Space)

We standardized all plays to move from **Left to Right**. This effectively halves the prediction space. The model no longer needs to learn separate patterns for left-moving vs. right-moving plays, allowing it to focus purely on relative motion dynamics.

```python
def unify_left_direction(df: pd.DataFrame) -> pd.DataFrame:
    if 'play_direction' not in df.columns: return df
    df = df.copy(); right = df['play_direction'].eq('right')
    # Invert X and Y for plays moving right
    if 'x' in df.columns: df.loc[right, 'x'] = FIELD_LENGTH - df.loc[right, 'x']
    if 'y' in df.columns: df.loc[right, 'y'] = FIELD_WIDTH  - df.loc[right, 'y']
    # Adjust angles
    for col in ('dir','o'):
        if col in df.columns:
             numeric_col = pd.to_numeric(df.loc[right, col], errors='coerce')
             transformed_angles = (numeric_col + 180.0) % 360.0
             df.loc[right, col] = transformed_angles
             if df[col].isnull().any(): df[col] = df[col].fillna(0.0)
    # Adjust ball landing spot
    if 'ball_land_x' in df.columns: df.loc[right, 'ball_land_x'] = FIELD_LENGTH - df.loc[right, 'ball_land_x']
    if 'ball_land_y' in df.columns: df.loc[right, 'ball_land_y'] = FIELD_WIDTH  - df.loc[right, 'ball_land_y']
    return df
```

*Note: This standardization logic was open-sourced by our team early in the competition and adopted by the community.*

### 2.2 Random Y-Flip: Forcing Chiral Invariance (Key Innovation)

American football fields possess **Chiral Symmetry** (Mirror Symmetry) along the long axis. A route running towards the left sideline is physically identical to a mirrored route running towards the right sideline.
To force the model to learn this **invariance**, we implemented a dynamic **Random Y-Flip** augmentation during training.

**Implementation Logic:**
In our `SpatioTemporalData` loader, with $p=0.5$ for every epoch, we invert the Y-axis logic:
$$y' = W - y$$
$$v_y', a_y' = -v_y, -a_y$$
$$\theta' = -\theta \quad (\text{for orientation } o \text{ and direction } dir)$$

```python
# Inside __getitem__
# --- [START] 1. Random Y-Flip (Chiral Augmentation) ---
if self.is_train and np.random.rand() < 0.5:
    # Copy to avoid modifying cached data
    seq_np = seq_np.copy() 
    
    # 1. Flip Coordinate 'y' (y_new = FIELD_WIDTH - y)
    if 'y' in self.feature_to_idx:
        y_idx = self.feature_to_idx['y']
        seq_np[..., y_idx] = FIELD_WIDTH - seq_np[..., y_idx]
    
    # 2. Flip Vector Components (val_new = -val)
    # Includes velocity_y, acceleration_y, jerk_y, etc.
    if self.flip_indices_invert:
        seq_np[..., self.flip_indices_invert] = -seq_np[..., self.flip_indices_invert]
        
    # 3. Flip Angles (val_new = -val, then wrap)
    # Includes dir, o
    if self.flip_indices_angular:
        seq_np[..., self.flip_indices_angular] = wrap_angle_deg(-seq_np[..., self.flip_indices_angular])
        
    # 4. Flip Target Trajectory
    target_dy_np = -target_dy_np
```

This strategy prevented the model from memorizing "hotspots" on specific sides of the field and significantly improved generalization on unseen plays.

### 2.3 Context Selection

Given the constraints of the STTRE architecture (which is designed for multivariate time series rather than graph structures) and the non-consistent number/roles/positions of players in training data, we had to be selective about which players to include in the context window.

We employed a hard-coded selection logic to pick the **4 most relevant entities** (`MAX_SUBPLAY_SIZE = 4`):

1.  **Target Player** (Always included)
2.  **Passer** (Quarterback)
3.  **Nearest Teammate**
4.  **Nearest Defender** (specifically prioritizing Defensive Backs)



```python
def find_relevant_entities_v3_PositionFiltered(play_df: pd.DataFrame, target_nid: int, cfg: Config) -> list:
    # 1. Always include Target & Passer
    entities, ids_to_exclude, ... = _get_common_entities(play_df, target_nid, cfg)
    
    # 2. Find Nearest Opponent (Prioritize DBs)
    if not valid_opponent_ids.empty:
        # ... filter for DBs ...
        nearest_opp_nids = opponent_dist.nsmallest(cfg.N_OPPONENTS).index.tolist()
        entities.extend(nearest_opp_nids)

    # 3. Find Nearest Teammate
    # ...
    return _finalize_entities(entities, ...)[:cfg.MAX_SUBPLAY_SIZE]
```

## 3\. Model Architecture: STTRE with Gated Fusion

We reproduced the architecture from the paper **"STTRE: A Spatio-Temporal Transformer with Relative Embeddings"** [1] and adapted it for the NFL context.

![Model Architecture](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F25207474%2F1aa8a44dd00758e9bf9c3ae96e293472%2Fsttre_nv50_viz.png?generation=1764814279191984&alt=media)


Instead of a standard Transformer, we utilized a **Three-Branch Encoder** to decouple dependencies:

1.  **Temporal Branch:** Captures individual player momentum (using Last Frame Pooling).
2.  **Spatial Branch:** Captures interactions between players (using Mean Pooling over time).
3.  **Spatio-Temporal Branch:** Captures complex evolving interactions.

### Key Architectural Features:

  * **Relative Role Bias:** We replaced standard embeddings with Relative Role Bias. The attention mechanism explicitly encodes the relationship between query and key (e.g., *Teammate-to-Teammate* vs. *Defender-to-Target*), allowing the model to "understand" coverage schemes.
  * **Sigmoid Gated Fusion:** Inspired by recent LLM research (e.g., Qwen), we used a **Sigmoid Gating Network** to dynamically fuse the spatial and temporal branches. This acts as a non-linear filter, allowing the model to ignore noisy spatial interactions when momentum is the dominant factor.

## 4\. Training Strategy

We trained using an **Auxiliary Velocity Loss** alongside the standard Position Loss.
$$L = L_{pos} + \lambda \cdot L_{vel}$$
Standard MSE loss often leads to "lazy" predictions that lag behind the true trajectory. By penalizing errors in the first derivative (velocity), we forced the model to respect physical continuity and momentum.

## 5\. Post-Processing & TTA

To squeeze out every bit of performance, we employed a robust inference pipeline:

  * **TTA (Test Time Augmentation):** We ran inference on both the original sequence and its **Y-Flipped** version, averaging the results (after un-flipping).
  * **Noisy Runs:** We performed multiple forward passes with slight Gaussian noise injection (`std=0.02`) to smooth out model variance.
  * **Extrapolation:** For any missing frames at the end of a sequence, we used a physics-based linear extrapolation based on the last known velocity vector to prevent "snapping" to zero.

## 6\. Error Analysis & Potential Improvements

Despite our strong performance, visualizing our worst predictions reveals a clear limitation: **Inertia Bias**. Our model over-relies on inertia/momentum and predicts trajectories that preserves the **direction** of momentum in the first few frames. 

![J-turns and cuts](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F25207474%2F584fa9e7aae0c0ce9dcd2ea6e8b3a76d%2Fvizfv44.png?generation=1764814683626634&alt=media)

### 6.1 The "Sharp Cut" Problem

As shown in the figure below, our model (Red) struggles with sharp cuts or "J-turns," often overshooting the trajectory along the original velocity vector. The model assumes players will continue their momentum (Inertia), failing to anticipate sudden braking or direction changes. This is an overfit because our training focuses on minimizing Mean square error of trajectories and overlooks hard samples.

### 6.2 The Missed Opportunity: Play Outcome Classification

Why do players make these sharp cuts? They are almost always reactions to the **outcome of the pass**.

The competition description explicitly hints at this:

 *"The downfield pass is the crown jewel of American sports. When the ball is in the air, anything can happen, like a touchdown, an interception, or a contested catch. The uncertainty and the importance of the outcome of these plays is what helps keep audiences on the edge of its seat."*

We treated tracking data as a deterministic sequence, but the trajectory is fundamentally **conditional**:

  * **If Catch:** Receiver maintains speed/direction to score.
  * **If Interception:** Receiver stops, turns, and tackles (Sharp Cut).
  * **If Incomplete:** Receiver slows down or breaks off.

**Future Improvement:**
We missed the opportunity to build a **Multi-Task Learning** framework. A better approach would be:

1.  **Classify Outcome:** Train a head to predict $P(\text{Catch})$, $P(\text{Interception})$, $P(\text{Incomplete})$.
2.  **Condition Trajectory:** Use these probabilities to condition the trajectory decoder.

By ignoring this *outcome uncertainty*, our model essentially learned the "average" path between a catch and an interception—a path that physically exists in neither scenario. Addressing this classification task is likely the key to breaking the 0.480 Gold barrier.


## 7\. Ensemble Strategy: Physics-Aware Vector Aggregation

Brute-force averaging of coordinates often leads to physically impossible trajectories (e.g., if Model A predicts a left turn and Model B predicts a right turn, a simple mean predicts the player running straight but at half speed).

To address this, we implemented a **Heterogeneous Ensemble** combining three distinct architectures, aggregated via **Polar Vector** strategy.

### 7.1 Heterogeneous Architectures

Diversity was key to our ensemble's success. We combined:

1.  **STTRE (Transformer):** Our main model (`nv50`). Excellent at long-term dependency and spatial interaction.
2.  **ESG (GNN):** An Evolving Subgraph Graph Neural Network. It captures local spatial interactions (blocking/tackling) better than global attention.
3.  **Bi-GRU (RNN):** A Bidirectional GRU model. While weaker at long horizons, it excels at capturing immediate momentum and short-term kinematics.

### 7.2 Polar Vector Ensemble (Key Innovation)

Standard ensembles average the $x$ and $y$ coordinates: $\bar{x} = \sum w_i x_i$. This often results in **energy loss**—the averaged trajectory has lower velocity than any of the individual models.

We implemented **Polar Vector Averaging**. We decompose the prediction into **Magnitude (Speed)** and **Direction (Unit Vector)** and average them separately.

$$\vec{v}_{final} = \left( \sum w_i \cdot \text{Speed}_i \right) \times \text{Norm}\left( \sum w_i \cdot \vec{u}_i \right)$$

This ensures that if two models disagree on direction, the ensemble essentially "votes" on the angle, but **maintains the kinetic energy (speed)** of the player.

**Implementation:**

```python
def polar_vector_ensemble(preds_list, weights_list):
    """
    Physics-Aware Ensemble:
    Averages Speed (Magnitude) and Direction (Unit Vector) separately
    to prevent kinetic energy loss during fusion.
    """
    magnitudes = []
    unit_vectors = []
    
    for p in preds_list:
        # Calculate Speed
        mag = np.linalg.norm(p, axis=-1, keepdims=True)
        # Calculate Unit Direction Vector
        mask = mag > 1e-6
        unit = np.zeros_like(p)
        unit[mask[:,0]] = p[mask[:,0]] / mag[mask[:,0]]
        
        magnitudes.append(mag)
        unit_vectors.append(unit)

    # 1. Weighted Average of Speed (Preserves Momentum)
    avg_mag = np.zeros_like(magnitudes[0])
    for mag, w in zip(magnitudes, weights_list):
        avg_mag += mag * w

    # 2. Vector Average of Direction
    avg_dir_vec = np.zeros_like(unit_vectors[0])
    for vec, w in zip(unit_vectors, weights_list):
        avg_dir_vec += vec * w
        
    # 3. Reconstruct Velocity Vector
    avg_dir_norm = np.linalg.norm(avg_dir_vec, axis=-1, keepdims=True)
    final_unit_vec = avg_dir_vec / (avg_dir_norm + 1e-6)
    
    return final_unit_vec * avg_mag
```

### 7.3 Dynamic Time-Decay Weighting

We observed that RNNs (GRU) degrade rapidly over long prediction horizons ($t > 30$), while Transformers maintain consistency. To exploit this, we used a **Linear Decay Weighting** for the GRU component.

  * **Frame 0:** GRU weight = 0.15
  * **Frame 30:** GRU weight = 0.00
  * **Frame 55:** GRU weight = 0.00

This allows the ensemble to utilize the RNN's superior "instantaneous reflex" for the immediate future while relying on the Transformer's "strategic planning" for the long term.

```python
# Dynamic Weighting Logic
decay_curve = np.linspace(W_GRU_INIT, 0.0, DECAY_STEPS)
w_gru_vec = np.concatenate([decay_curve, np.zeros(...)])
```


-----

**References:**
[1] Deihim, A., Alonso, E., & Apostolopoulou, D. (2023). *STTRE: A Spatio-Temporal Transformer with Relative Embeddings for multivariate time series forecasting*. Neural Networks, 168, 549-559. [DOI: 10.1016/j.neunet.2023.09.039](https://doi.org/10.1016/j.neunet.2023.09.039)

-----

Thanks to the Kaggle team and NFL for hosting this challenging competition and providing high-quality telemetry data! We have learned a lot in this competition! Thanks to the authors of this paper for inspiring our model architecture. Thanks to GPT for recreation of the architecture and Gemini for all the code heavy-lifting, brainstorming, experiment analysis, and most crucially, **emotional support**.

We have open-sourced our full training notebooks.