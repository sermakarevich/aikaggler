# Private 4th / Public 5th Place Solution

- **Author:** Takoi
- **Date:** 2025-12-05T12:33:30.463Z
- **Topic ID:** 651814
- **URL:** https://www.kaggle.com/competitions/nfl-big-data-bowl-2026-prediction/discussion/651814
---

First of all, I would like to express my gratitude to the organizers for hosting such an exciting competition. I also appreciate the many Kagglers who shared their insights during the contest.

Although the final results are not determined yet, I finished 5th on the Public LB. Now, all I can do is hope that the submission remains in the Gold zone after the final tests.

Here is a summary of my solution.

### 1. Key Strategy & Intuition

**Focus on Defenders**
I prioritized the prediction of Defenders. Receivers' routes are often pre-planned based on the ball's destination, making them easier to predict. Defenders, however, must react dynamically to the situation.

**Observation**
To prepare for this competition, I watched many NFL pass plays. Although I am not an expert, I noticed that defenders (especially in Man Coverage) often focus on the offensive player they are marking rather than the ball itself.

**The Issue with ball_land Variables**
The competition dataset provides ball_land_x and ball_land_y (the coordinates where the ball will land). However, if a defender is marking an offensive player who is not involved in the play, that defender cares little about where the ball lands. Therefore, simply treating these provided ball_land coordinates as important information for every defender introduces noise. My strategy was to design a model that allows specific defenders to ignore this information when it is irrelevant.

### 2. Data & Preprocessing

**Dataset**
I utilized the dataset provided in the NFL Big Data Bowl 2026 - Prediction competition.

**Augmentation**
I increased the training data size by 4x using coordinate flips:

1. Original
2. Horizontal Flip
3. Vertical Flip
4. Horizontal + Vertical Flip

**Inference**
I used the same 4 patterns for Test Time Augmentation (TTA) and ensembled the predictions.

### 3. Feature Engineering

**My Approach**

Instead of simply adding ball_land_x and ball_land_y as feature columns for every player, I treated the Ball Landing Position as a distinct "Player" (a fixed node) in my model input.

- Potential Issue: If ball_land coordinates are added uniformly as features to every player, it creates noise for defenders who are completely unrelated to the catch.
- The Solution: By treating the ball_land point as a distinct node in the model input:
    - Relevant Defenders: The Attention mechanism (in the Transformer layers) allows them to assign a high weight to the "Ball Node" if the ball is relevant to their play.
    - Irrelevant Defenders: The model can assign a low attention weight, effectively ignoring the ball location to focus on their marked man.
- Result: This approach yielded better CV scores compared to adding ball_land as a simple numeric feature.

### Key Features

The key features used in my model include:

- Basic: x, y, s, a, dir, player_position, player_side.
- Decomposition: Velocity vectors (s * sin(dir), s * cos(dir)).
- Deltas: Difference in s and a from the previous timestamp.
- Relative: Differences (x, y, s, a) between the target player and others.

### 4. Model Architecture

My model employs a Single-Target Prediction approach. Even though multiple players need prediction, the model inputs all players but outputs the trajectory for only one target player at a time.

**Structure:** Stack of 4 blocks: [GRU (Time axis) -> Transformer (Player axis)]

**Pooling with Entmax**

- After these 4 stacked blocks (each consisting of a GRU and a Transformer), I used Entmax pooling instead of standard Softmax or Average pooling.
- Reason: I wanted to achieve Sparse Attention. For a specific defender, only a few players (e.g., the specific opponent they are marking) are important. Entmax allows the weights for irrelevant players to be significantly smaller compared to standard Softmax, effectively filtering out noise.
- Reference: [HMS Harmful Brain Activity - 1st Place Solution](https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification/writeups/team-sony-1st-place-solution-team-sony)
- The pooling result is passed to a Linear head.

### 5. Output & Loss

- Output: The model predicts the difference (delta x, delta y) from the previous frame. I also learned the second-order difference (acceleration of position). The final output is the cumulative sum of predicted deltas.
- Sequence Length: I predicted 34 frames. If the play is longer than 34 frames, I simply repeated the result of the 34th frame. In my experiments, predicting beyond this horizon was difficult and hurt the score, so I clamped it.
- Loss Function: An ensemble of models trained with TemporalHuber and GaussianNLLLoss.