# Your First Chess Bot

- **Author:** Bovard Doerschuk-Tiberi
- **Votes:** 1350
- **Ref:** bovard/your-first-chess-bot
- **URL:** https://www.kaggle.com/code/bovard/your-first-chess-bot
- **Last run:** 2025-01-22 22:27:02.160000

---

# FIDE & Google Efficient Chess AI Challenge

Welcome! This notebook will familiarize you with using competition's environment, creating an agent, and submitting your first chess bot!

```python
# first let's make sure you have internet enabled
import requests
requests.get('http://www.google.com',timeout=10).ok
```

#### If you don't have internet access (it doesn't say "True" above)
1. make sure your account is Phone Verified in [account settings](https://www.kaggle.com/settings)
2. make sure internet is turned on in Settings -> Turn on internet

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

```python
# this should run a game in the environment between two random bots
# NOTE: each game starts from a randomly selected opening
result = env.run(["random", "random"])
env.render(mode="ipython", width=1000, height=1000)
```

### Creating your first agent
Now let's create your first agent! The environment has the [Chessnut](https://github.com/cgearhart/Chessnut) pip package installed and we'll use that to parse the board state and generate moves.

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

### Testing your agent

Now let's see how your agent does againt the random agent!

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

# To Submit:
1. Download (or save) submission.py
2. Go to the [submissions page](https://www.kaggle.com/competitions/fide-google-efficiency-chess-ai-challenge/submissions) and click "Submit Agent"
3. Upload submission.py
4. Press Submit!

Now doubt you are already thinking of ways this bot could be improved! Go ahead and fork this notebook and get started! ♟️

# Submitting Multiple files 
### (or compressing your submission.py)

Set up your directory structure like this:
```
kaggle_submissions/
  main.py
  <other files as desired>
```

**Note**: You need to rename `submission.py` to `main.py`

You can run `tar -czf submission.tar.gz -C kaggle_submissions .` and upload `submission.tar.gz`