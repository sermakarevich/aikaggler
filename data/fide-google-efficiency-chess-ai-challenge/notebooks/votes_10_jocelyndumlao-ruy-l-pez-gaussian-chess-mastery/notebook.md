# 🏆 Ruy López: Gaussian Chess Mastery 

- **Author:** Jocelyn Dumlao
- **Votes:** 85
- **Ref:** jocelyndumlao/ruy-l-pez-gaussian-chess-mastery
- **URL:** https://www.kaggle.com/code/jocelyndumlao/ruy-l-pez-gaussian-chess-mastery
- **Last run:** 2024-12-05 05:11:03.490000

---

<div style="display: flex; justify-content: space-between; align-items: flex-start;">
    <div style="text-align: left;">
        <p style="color:#FFD700; font-size: 15px; font-weight: bold; margin-bottom: 1px; text-align: left;">Published on November 20, 2024</p>
        <h4 style="color:#4B0082; font-weight: bold; text-align: left; margin-top: 6px;">Author: Jocelyn C. Dumlao</h4>
        <p style="font-size: 17px; line-height: 1.7; color: #333; text-align: center; margin-top: 20px;"></p>
        <a href="https://www.linkedin.com/in/jocelyn-dumlao-168921a8/" target="_blank" style="display: inline-block; background-color: #003f88; color: #fff; text-decoration: none; padding: 5px 10px; border-radius: 10px; margin: 15px;">LinkedIn</a>
        <a href="https://github.com/jcdumlao14" target="_blank" style="display: inline-block; background-color: transparent; color: #059c99; text-decoration: none; padding: 5px 10px; border-radius: 10px; margin: 15px; border: 2px solid #007bff;">GitHub</a>
        <a href="https://www.youtube.com/@CogniCraftedMinds" target="_blank" style="display: inline-block; background-color: #ff0054; color: #fff; text-decoration: none; padding: 5px 10px; border-radius: 10px; margin: 15px;">YouTube</a>
        <a href="https://www.kaggle.com/jocelyndumlao" target="_blank" style="display: inline-block; background-color: #3a86ff; color: #fff; text-decoration: none; padding: 5px 10px; border-radius: 10px; margin: 15px;">Kaggle</a>
    </div>
</div>

# <font size="+3" color='#fca311'><b> 🏆 Ruy López: Gaussian Chess Mastery </b></font>

# <font size="+3" color='#006d77'><b> Import request/Install Env.</b></font>

```python
# first let's make sure you have internet enabled
import requests
requests.get('http://www.google.com',timeout=10).ok
```

```python
%%capture
# ensure we are on the latest version of kaggle-environments
!pip install --upgrade kaggle-environments
```

```python
# Now let's set up the chess environment!
from kaggle_environments import make
env = make("chess", debug=True)
```

# <font size="+3" color='#fca311'><b> 🏆 Ruy López de Segura - Spanish Chess Player</b></font>

Ruy López de Segura was a prominent Spanish chess player and priest, regarded as one of the pioneers of modern chess. Flourishing in the 16th century, he became a key figure in the development of chess opening theory, most notably with the opening that now bears his name: the Ruy López opening. This opening remains one of the most popular and enduring strategies in chess. López also made significant contributions through his 1561 work, Libro de la invención liberal y arte del juego del Axedrez, the first comprehensive chess manual, which laid the foundation for chess strategy and opening theory. He was a favorite of King Philip II and was recognized for his blindfold chess prowess.

For more detailed information, you can visit his biography on the Encyclopaedia Britannica website: [Ruy López de Segura](https://www.britannica.com/biography/Ruy-Lopez-de-Segura) - Britannica​ 
WIKIPEDIA ENCYCLOPEDIA BRITANNICA

# <font size="+3" color='#006d77'><b> Set Up the Ruy Lopez Opening</b></font>

```python
from kaggle_environments import make

# Create a chess environment
env = make("chess", debug=True)

# Define the Ruy Lopez opening in FEN notation
ruy_lopez_fen = "rnbqkbnr/pppp1ppp/2n5/3B4/8/8/PPP1PPPP/RNBQK1NR w KQkq - 2 4"

# Set the board to the custom starting position (Ruy Lopez Opening)
env.configuration["initialBoard"] = ruy_lopez_fen

# Set up the 3-game competition between two random agents
for i in range(3):
    print(f"Game {i+1}")
    
    # Run a match between two random bots
    result = env.run(["random", "random"])
    
    # Render the game in the IPython environment
    env.render(mode="ipython", width=1000, height=1000)

    # Optionally, you can print the result of each game
    #print(f"Result of Game {i+1}: {result}")
```

# <font size="+3" color='#3d348b'><b> Explanation:</b></font>

* **Ruy Lopez FEN**: `rnbqkbnr/pppp1ppp/2n5/3B4/8/8/PPP1PPPP/RNBQK1NR w KQkq - 2 4` represents the board after the moves 1. e4 e5 2. Nf3 Nc6 3. Bb5, which is the Ruy Lopez opening. The FEN captures the state of the board, whose pieces are arranged as per this opening.
* **Set initialBoard**: The initialBoard configuration is set with the Ruy Lopez FEN string, which ensures that the game starts from this position.
* **Run 3 games**: The loop runs 3 games with two random agents.

# <font size="+3" color='#3d348b'><b> FEN Breakdown:</b></font>

* `rnbqkbnr/pppp1ppp/2n5/3B4/8/8/PPP1PPPP/RNBQK1NR`: This part represents the board after 1. e4 e5 2. Nf3 Nc6 3. Bb5.
* `w`: White to move.
* `KQkq`: Both sides can still castle.
* `-`: No en passant target square.
* `2 4`: The number of halfmoves since the last pawn move or capture, and the move number.

# <font size="+3" color='#3d348b'><b> Notes:</b></font>

* **Custom Opening**: By using the FEN string, you're overriding the random opening with the Ruy Lopez opening.
* **Agents**: The code still uses the "random" agent for both sides. You can replace "random" with your custom agents if needed.

# <font size="+3" color='#006d77'><b> Gaussian Distribution in Chess Competition Context</b></font>

A Gaussian distribution (or normal distribution) is a common statistical concept where data is symmetrically distributed around a mean (𝜇) with a standard deviation (𝜎). For the chess competition, players' skill ratings are modeled using a Gaussian distribution where:

- 𝜇 represents the player's current estimated skill.
- 𝜎 represents the uncertainty or confidence in that skill estimate.

**As players compete:**

- Winning increases 
𝜇, as it reflects an improvement in skill.
- Losing decreases 
𝜇, while uncertainty (
𝜎) decreases over time as the system becomes more confident in the rating.
- Ties adjust 
𝜇 closer for both players, reflecting their similar skill levels.

# <font size="+3" color='#006d77'><b> Modeling Skill Rating</b></font>

<font size="+2" color='#bf4342'><b> Gaussian Distribution and a Visualization of the Skill Ratings.</b></font>

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Initial parameters
mu_player = 600  # Initial skill rating
sigma_player = 200  # Initial uncertainty
mu_opponent = 600
sigma_opponent = 200

# Gaussian distribution for initial skills
x = np.linspace(200, 1000, 500)
player_distribution = norm.pdf(x, mu_player, sigma_player)
opponent_distribution = norm.pdf(x, mu_opponent, sigma_opponent)

# Plot initial distributions
plt.figure(figsize=(10, 6))
plt.plot(x, player_distribution, label="Player's Skill Distribution", color='blue')
plt.plot(x, opponent_distribution, label="Opponent's Skill Distribution", color='orange')
plt.title("Initial Gaussian Distribution of Skills")
plt.xlabel("Skill Rating")
plt.ylabel("Probability Density")
plt.legend()
plt.grid()
plt.show()

# Simulate a game
def update_ratings(mu1, sigma1, mu2, sigma2, outcome):
    """Update skill ratings based on outcome.
       outcome = 1 (win), 0 (loss), 0.5 (tie)"""
    k = 30  # Update factor
    expected = 1 / (1 + np.exp((mu2 - mu1) / sigma1))
    delta = k * (outcome - expected)
    mu1_new = mu1 + delta
    sigma1_new = max(sigma1 - 5, 50)  # Reduce uncertainty
    return mu1_new, sigma1_new

# Update ratings after a game
mu_player, sigma_player = update_ratings(mu_player, sigma_player, mu_opponent, sigma_opponent, outcome=1)  # Player wins

# New distribution
player_distribution_updated = norm.pdf(x, mu_player, sigma_player)

# Plot updated distributions
plt.figure(figsize=(10, 6))
plt.plot(x, player_distribution, label="Player's Initial Skill Distribution", color='blue', linestyle='--')
plt.plot(x, player_distribution_updated, label="Player's Updated Skill Distribution", color='blue')
plt.title("Skill Distribution Update After a Win")
plt.xlabel("Skill Rating")
plt.ylabel("Probability Density")
plt.legend()
plt.grid()
plt.show()
```

# <font size="+3" color='#006d77'><b> Define the Agent Class</b></font>

Encapsulate the logic in an agent class or function.

```python
class PlayerAgent:
    def __init__(self, mu=600, sigma=200):
        self.mu = mu  # Skill rating
        self.sigma = sigma  # Uncertainty
    
    def update_ratings(self, opponent_mu, opponent_sigma, outcome):
        """Update the player's rating based on the outcome."""
        k = 30  # Update factor
        expected = 1 / (1 + np.exp((opponent_mu - self.mu) / self.sigma))
        delta = k * (outcome - expected)
        self.mu += delta
        self.sigma = max(self.sigma - 5, 50)  # Reduce uncertainty

    def decide_move(self, state):
        """Decide on the move given the current state."""
        # Placeholder logic for move selection
        return "default_move"
```

```python
!pip install Chessnut
```

# <font size="+3" color='#006d77'><b> Submission</b></font>

```python
%%writefile submission.py
from Chessnut import Game
import random

def chess_bot(obs):
    """
    Simple chess bot that prioritizes checkmates, then captures, queen promotions, then randomly moves.

    Args:
        obs: An object with a 'board' attribute representing the current board state as a FEN string.

    Returns:
        A string representing the chosen move in UCI notation (e.g., "e2e4")
    """
    # 0. Parse the current board state and generate legal moves using Chessnut library
    game = Game(obs.board)
    moves = list(game.get_moves())

    # 1. Check a subset of moves for checkmate
    for move in moves[:10]:
        g = Game(obs.board)
        g.apply_move(move)
        if g.status == Game.CHECKMATE:
            return move

    # 2. Check for captures
    for move in moves:
        if game.board.get_piece(Game.xy2i(move[2:4])) != ' ':
            return move

    # 3. Check for queen promotions
    for move in moves:
        if "q" in move.lower():
            return move

    # 4. Random move if no checkmates or captures
    return random.choice(moves)
```

```python
result = env.run(["submission.py", "random"])
print("Agent exit status/reward/time left: ")
# look at the generated replay.json and print out the agent info
for agent in result[-1]:
    print("\t", agent.status, "/", agent.reward, "/", agent.observation.remainingOverageTime)
print("\n")
# render the game
env.render(mode="ipython", width=1000, height=1000)
```