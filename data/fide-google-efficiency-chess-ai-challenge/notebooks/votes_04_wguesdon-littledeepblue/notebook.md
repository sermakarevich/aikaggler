# LittleDeepBlue

- **Author:** Will
- **Votes:** 150
- **Ref:** wguesdon/littledeepblue
- **URL:** https://www.kaggle.com/code/wguesdon/littledeepblue
- **Last run:** 2024-11-27 15:44:45.773000

---

<!-- FIDE & Google Efficient Chess AI Challenge-->

<div style="font-family: 'Poppins'; font-weight: bold; letter-spacing: 0px; color: #FFFFFF; font-size: 300%; text-align: left; padding: 15px; background: #0A0F29; border: 8px solid #00FFFF; border-radius: 15px; box-shadow: 5px 5px 20px rgba(0, 0, 0, 0.5);">
    FIDE & Google Efficient Chess AI Challenge<br>
</div>

# <div style="background-color:#0A0F29; font-family:'Poppins', cursive; color:#E0F7FA; font-size:140%; text-align:center; border: 2px solid #00FFFF; border-radius:15px; padding: 15px; box-shadow: 5px 5px 20px rgba(0, 0, 0, 0.5); font-weight: bold; letter-spacing: 1px; text-transform: uppercase;">Environment Setup and Installation</div>

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
env.render(mode="ipython", width=700, height=700)
```

# <div style="background-color:#0A0F29; font-family:'Poppins', cursive; color:#E0F7FA; font-size:140%; text-align:center; border: 2px solid #00FFFF; border-radius:15px; padding: 15px; box-shadow: 5px 5px 20px rgba(0, 0, 0, 0.5); font-weight: bold; letter-spacing: 1px; text-transform: uppercase;">Creating your first agent</div>

Now let's create your first agent! The environment has the [Chessnut](https://github.com/cgearhart/Chessnut) pip package installed and we'll use that to parse the board state and generate moves.

```python
%%writefile initial_agent.py
from Chessnut import Game
import random

def heuristic_chess_bot(obs):
    """
    A heuristic-based chess bot that prioritizes:
    - Checkmates
    - Captures
    - Promotions
    - Random moves
    Args:
        obs: Object with 'board' representing board state as FEN string
    Returns:
        Move in UCI notation
    """
    game = Game(obs.board)
    moves = list(game.get_moves())
    random.shuffle(moves)  # Randomize moves to add variation

    # Prioritize checkmates
    for move in moves[:10]:
        g = Game(obs.board)
        g.apply_move(move)
        if g.status == Game.CHECKMATE:
            return move

    # Check for captures
    for move in moves:
        if game.board.get_piece(Game.xy2i(move[2:4])) != ' ':
            return move

    # Check for promotions
    for move in moves:
        if "q" in move.lower():  # Queen promotion
            return move

    # Default to random move
    return random.choice(moves)
```

### Testing your agent

Now let's see how your agent does againt the random agent!

```python
# Testing the agent
result = env.run(["initial_agent.py", "random"])
for agent in result[-1]:
    print("Status:", agent.status, "/ Reward:", agent.reward, "/ Time left:", agent.observation.remainingOverageTime)
env.render(mode="ipython", width=700, height=700)
```

# <div style="background-color:#0A0F29; font-family:'Poppins', cursive; color:#E0F7FA; font-size:140%; text-align:center; border: 2px solid #00FFFF; border-radius:15px; padding: 15px; box-shadow: 5px 5px 20px rgba(0, 0, 0, 0.5); font-weight: bold; letter-spacing: 1px; text-transform: uppercase;">Improving the Agent</div>

- This is my first simulation competition :D
- I'm quite excited, as I love playing chess, although I have never explored coding a chess agent before—so this will be fun.
- For my initial models, I am mostly relying on GPT and Claude 3.5 to help me work through the process.
- At the moment, I am using the web version since Claude 3.5 is working better for me than GPT. However, I plan to switch to using the API so that prompts and responses are reproducible.

# Prompt
- Model: Claude 3.5
- Workflow: I used the prompt below to generate an initial script and then iteratively applied bug fixes with Claude 3.5.
- I then compared the performance of this bot to the baseline initial bot. So far, these bots seems to perform worse (sometime a lot worse).
- To see previous prompt experiments you can navigate between versions of the notebook. 
- Next, I will still use an LLM but will focus on implementing one function at a time after deciding on the structure of my code. For example, I will use the LLM to help implement the min-max function.

I am working on creating a simple Python chess engine using Chessnut to handle the board and legal moves. 
The agent is contrainst so has to works with low ram and memory. 

Constraints:
- 5 MiB of RAM**
- Dedicated CPU: a single 2.20GHz core
- 64KiB compressed submission size limit

The game does not start from the intial position. 

Based on the demonstration bot and the sunfish codebased write a new chess engine powered by Chessbut.

See below the demonstration bot to improve: 

```python
from Chessnut import Game
import random

def heuristic_chess_bot(obs):
    """
    A heuristic-based chess bot that prioritizes:
    - Checkmates
    - Captures
    - Promotions
    - Random moves
    Args:
        obs: Object with 'board' representing board state as FEN string
    Returns:
        Move in UCI notation
    """
    game = Game(obs.board)
    moves = list(game.get_moves())
    random.shuffle(moves)  # Randomize moves to add variation

    # Prioritize checkmates
    for move in moves[:10]:
        g = Game(obs.board)
        g.apply_move(move)
        if g.status == Game.CHECKMATE:
            return move

    # Check for captures
    for move in moves:
        if game.board.get_piece(Game.xy2i(move[2:4])) != ' ':
            return move

    # Check for promotions
    for move in moves:
        if "q" in move.lower():  # Queen promotion
            return move

    # Default to random move
    return random.choice(moves)
```

Please improve this bot by adding a Search Algorithm using:
Minimax: A basic decision-making algorithm that assumes both players play optimally.
Alpha-Beta Pruning: Optimizes Minimax by eliminating branches of the game tree that cannot affect the final decision.

# Agent LittleDeepBlue 0.5.5

```python
%%writefile LittleDeepBlue_0_5_5.py
from Chessnut import Game
import time
import random

PIECE_VALUES = {'p': 100, 'n': 320, 'b': 330, 'r': 500, 'q': 900, 'k': 20000}

def evaluate_board(game):
    """
    Evaluate the board position from the perspective of the current player.
    Positive scores are good for the current player, negative for the opponent.
    """
    board_str = str(game.board)
    score = 0
    for char in board_str:
        if char.isalpha():
            if char.isupper():
                score += PIECE_VALUES[char.lower()]
            else:
                score -= PIECE_VALUES[char]
    return score

def minimax(game, depth, alpha, beta, maximizing_player, start_time, time_limit):
    if time.time() - start_time > time_limit:
        return None, None
    if depth == 0 or game.status >= 2:
        return evaluate_board(game), None
    moves = list(game.get_moves())
    if not moves:
        return evaluate_board(game), None
    best_move = None
    if maximizing_player:
        max_eval = float('-inf')
        for move in moves:
            # Time check
            if time.time() - start_time > time_limit:
                break
            new_game = Game(str(game))
            new_game.apply_move(move)
            eval_score, _ = minimax(new_game, depth - 1, alpha, beta, False, start_time, time_limit)
            if eval_score is None:
                continue
            if eval_score > max_eval:
                max_eval = eval_score
                best_move = move
            alpha = max(alpha, eval_score)
            if beta <= alpha:
                break
        return max_eval, best_move
    else:
        min_eval = float('inf')
        for move in moves:
            # Time check
            if time.time() - start_time > time_limit:
                break
            new_game = Game(str(game))
            new_game.apply_move(move)
            eval_score, _ = minimax(new_game, depth - 1, alpha, beta, True, start_time, time_limit)
            if eval_score is None:
                continue
            if eval_score < min_eval:
                min_eval = eval_score
                best_move = move
            beta = min(beta, eval_score)
            if beta <= alpha:
                break
        return min_eval, best_move

def heuristic_chess_bot(obs):
    game = Game(obs.board)
    moves = list(game.get_moves())
    best_score = float('-inf')
    best_move = None
    for move in moves:
        move_score = 0
        from_square = move[:2]
        to_square = move[2:4]
        piece_moved = game.board.get_piece(Game.xy2i(from_square))
        target_piece = game.board.get_piece(Game.xy2i(to_square))
        # Add score for capturing a piece
        if target_piece != ' ':
            move_score += PIECE_VALUES[target_piece.lower()] - PIECE_VALUES[piece_moved.lower()]
        # Add a small bonus for developing pieces (moving from the back rank)
        if piece_moved.lower() in ['n', 'b', 'q']:
            if (piece_moved.isupper() and from_square[1] == '1') or (piece_moved.islower() and from_square[1] == '8'):
                move_score += 10
        # Check for promotions
        if "q" in move.lower():
            move_score += PIECE_VALUES['q']
        # Prefer castling
        if move in ['e1g1', 'e1c1', 'e8g8', 'e8c8']:
            move_score += 50
        if move_score > best_score:
            best_score = move_score
            best_move = move
    if best_move:
        return best_move
    else:
        return random.choice(moves)

def agent(obs, config):
    try:
        return hybrid_chess_bot(obs)
    except Exception:
        return random.choice(list(Game(obs.board).get_moves()))

def hybrid_chess_bot(obs):
    game = Game(obs.board)
    start_time = time.time()
    time_limit = 0.09  # Set per-move time limit to 0.09 seconds
    best_move = None
    moves = list(game.get_moves())
    if not moves:
        return "0000"  # No legal moves
    depth = 1
    try:
        # Limit depth to ensure completion within time limit
        while depth <= 2:
            elapsed_time = time.time() - start_time
            if elapsed_time >= time_limit:
                break
            current_time_limit = time_limit - elapsed_time
            eval_score, move = minimax(game, depth, float('-inf'), float('inf'), True, start_time, time_limit)
            if move:
                best_move = move
            depth += 1
    except Exception:
        pass
    if best_move:
        return best_move
    else:
        # Time is running out or no move found, use heuristic or random move
        elapsed_time = time.time() - start_time
        if elapsed_time < time_limit:
            return heuristic_chess_bot(obs)
        else:
            return random.choice(moves)
```

# Agent LittleDeepBlue_0_10_4

```python
%%writefile LittleDeepBlue_0_10_4_debug.py
from Chessnut import Game
import time
import random

DEBUG = True

# Piece values for evaluation
PIECE_VALUES = {'p': 1, 'n': 3, 'b': 3, 'r': 5, 'q': 9, 'k': 0}

def log_message(message):
    """Helper function to print messages if DEBUG is True."""
    if DEBUG:
        current_time = time.time()
        print(f"[{current_time:.2f}] {message}")

def enhanced_heuristic_with_time_limit(obs, per_move_time, remaining_time):
    """
    Enhanced heuristic evaluation with adjusted scoring to prioritize captures and protect endangered pieces.
    Includes detailed logging via log_message.
    """
    game = Game(obs.board)
    moves = list(game.get_moves())
    random.shuffle(moves)  # Randomize moves to add variation
    log_message(f"Available moves: {moves}")

    best_score = float('-inf')
    best_move = None
    start_time = time.time()

    for move in moves:
        move_start = time.time()
        elapsed = move_start - start_time
        remaining = remaining_time - elapsed
        if remaining <= 0:
            log_message("Time limit for heuristic evaluation reached. Stopping.")
            break  # Stop evaluating if total time runs out

        log_message(f"Evaluating move: {move}. Elapsed: {elapsed:.2f}s, Remaining: {remaining:.2f}s")

        if elapsed >= per_move_time:
            log_message(f"Skipping move {move} due to per-move time limit.")
            continue  # Skip move if per-move time limit is exceeded

        move_score = 0
        from_square = move[:2]
        to_square = move[2:4]

        # Simulate the move
        new_game = Game(str(game))
        new_game.apply_move(move)

        # Immediate checkmate detection
        if new_game.status == Game.CHECKMATE:
            log_message(f"Checkmate detected with move: {move}. Returning immediately.")
            return move

        # Piece and target evaluation
        piece_moved = game.board.get_piece(Game.xy2i(from_square))
        target_piece = game.board.get_piece(Game.xy2i(to_square))

        # Add score for captures (material gain)
        if target_piece != ' ':
            # Increase capture bonus based on the value of the captured piece
            capture_value = PIECE_VALUES[target_piece.lower()]
            move_score += capture_value * 100  # Amplify the importance of captures
            log_message(f"Move {move} captures {target_piece}. Capture value: {capture_value}. Updated score: {move_score}")

            # Encourage favorable exchanges
            piece_value = PIECE_VALUES.get(piece_moved.lower(), 0)
            net_gain = capture_value - piece_value
            move_score += net_gain * 50  # Encourage exchanges where net gain is positive
            log_message(f"Net gain from exchange: {net_gain}. Updated score: {move_score}")

        # Prioritize pawn promotion
        if 'q' in move.lower():
            move_score += PIECE_VALUES['q'] * 100  # Increase bonus for promotion
            log_message(f"Move {move} promotes to queen. Updated score: {move_score}")

        # Center control bonus (e4, d4, e5, d5)
        if to_square in ['d4', 'e4', 'd5', 'e5']:
            move_score += 20
            log_message(f"Move {move} controls center. Updated score: {move_score}")

        # Check detection
        if new_game.status == Game.CHECK:
            move_score += 30
            log_message(f"Move {move} gives check. Updated score: {move_score}")

        # Evaluate if any own pieces are under attack after the move
        opponent_moves = list(new_game.get_moves())
        for opp_move in opponent_moves:
            opp_target_square = opp_move[2:4]
            own_piece = new_game.board.get_piece(Game.xy2i(opp_target_square))
            if own_piece != ' ' and own_piece.isupper():  # Assuming bot plays as White
                # Penalize based on the value of the endangered piece
                endangered_piece_value = PIECE_VALUES.get(own_piece.lower(), 0)
                move_score -= endangered_piece_value * 100  # Adjust penalty multiplier as needed
                log_message(f"Own piece {own_piece} at {opp_target_square} is endangered. Penalizing move. Updated score: {move_score}")

        # Update the best move if the score is better
        if move_score > best_score:
            best_score = move_score
            best_move = move
            log_message(f"New best move found: {best_move}. Best score updated to: {best_score}")

        move_end = time.time()
        log_message(f"Move {move} evaluation completed in {move_end - move_start:.2f}s.")

    log_message(f"Best move chosen: {best_move} with final score: {best_score}")
    return best_move if best_score > float('-inf') else None

def agent(obs, config):
    """
    Combined chess agent with total and per-move time limits.
    """
    try:
        start_time = time.time()
        total_time_limit = 8.0
        per_move_time_limit = 0.1

        log_message("Starting agent evaluation")

        while True:
            elapsed_time = time.time() - start_time
            remaining_time = total_time_limit - elapsed_time

            log_message(f"Total elapsed: {elapsed_time:.2f}s, Remaining time: {remaining_time:.2f}s")

            if remaining_time <= 0.1:  # Grace period to avoid timeout
                log_message("Time is almost up. Defaulting to a random move.")
                break

            # Step 1: Use enhanced heuristic with per-move and remaining time limits
            move = enhanced_heuristic_with_time_limit(obs, per_move_time=per_move_time_limit, remaining_time=remaining_time)
            if move:
                log_message(f"Heuristic selected move: {move}")
                return move

        # Step 2: Default to a random move if time is exhausted
        game = Game(obs.board)
        random_move = random.choice(list(game.get_moves()))
        log_message(f"Random move chosen: {random_move}")
        return random_move

    except Exception as e:
        # Emergency fallback to a random move
        log_message(f"Error in agent: {e}")
        game = Game(obs.board)
        fallback_move = random.choice(list(game.get_moves()))
        log_message(f"Fallback random move chosen: {fallback_move}")
        return fallback_move
```

```python
%%writefile LittleDeepBlue_0_10_4.py
from Chessnut import Game
import time
import random

DEBUG = False

# Piece values for evaluation
PIECE_VALUES = {'p': 1, 'n': 3, 'b': 3, 'r': 5, 'q': 9, 'k': 0}

def log_message(message):
    """Helper function to print messages if DEBUG is True."""
    if DEBUG:
        current_time = time.time()
        print(f"[{current_time:.2f}] {message}")

def enhanced_heuristic_with_time_limit(obs, per_move_time, remaining_time):
    """
    Enhanced heuristic evaluation with adjusted scoring to prioritize captures and protect endangered pieces.
    Includes detailed logging via log_message.
    """
    game = Game(obs.board)
    moves = list(game.get_moves())
    random.shuffle(moves)  # Randomize moves to add variation
    #log_message(f"Available moves: {moves}")

    best_score = float('-inf')
    best_move = None
    start_time = time.time()

    for move in moves:
        move_start = time.time()
        elapsed = move_start - start_time
        remaining = remaining_time - elapsed
        if remaining <= 0:
            #log_message("Time limit for heuristic evaluation reached. Stopping.")
            break  # Stop evaluating if total time runs out

        #log_message(f"Evaluating move: {move}. Elapsed: {elapsed:.2f}s, Remaining: {remaining:.2f}s")

        if elapsed >= per_move_time:
            #log_message(f"Skipping move {move} due to per-move time limit.")
            continue  # Skip move if per-move time limit is exceeded

        move_score = 0
        from_square = move[:2]
        to_square = move[2:4]

        # Simulate the move
        new_game = Game(str(game))
        new_game.apply_move(move)

        # Immediate checkmate detection
        if new_game.status == Game.CHECKMATE:
            #log_message(f"Checkmate detected with move: {move}. Returning immediately.")
            return move

        # Piece and target evaluation
        piece_moved = game.board.get_piece(Game.xy2i(from_square))
        target_piece = game.board.get_piece(Game.xy2i(to_square))

        # Add score for captures (material gain)
        if target_piece != ' ':
            # Increase capture bonus based on the value of the captured piece
            capture_value = PIECE_VALUES[target_piece.lower()]
            move_score += capture_value * 100  # Amplify the importance of captures
            #log_message(f"Move {move} captures {target_piece}. Capture value: {capture_value}. Updated score: {move_score}")

            # Encourage favorable exchanges
            piece_value = PIECE_VALUES.get(piece_moved.lower(), 0)
            net_gain = capture_value - piece_value
            move_score += net_gain * 50  # Encourage exchanges where net gain is positive
            #log_message(f"Net gain from exchange: {net_gain}. Updated score: {move_score}")

        # Prioritize pawn promotion
        if 'q' in move.lower():
            move_score += PIECE_VALUES['q'] * 100  # Increase bonus for promotion
            #log_message(f"Move {move} promotes to queen. Updated score: {move_score}")

        # Center control bonus (e4, d4, e5, d5)
        if to_square in ['d4', 'e4', 'd5', 'e5']:
            move_score += 20
            #log_message(f"Move {move} controls center. Updated score: {move_score}")

        # Check detection
        if new_game.status == Game.CHECK:
            move_score += 30
            #log_message(f"Move {move} gives check. Updated score: {move_score}")

        # Evaluate if any own pieces are under attack after the move
        opponent_moves = list(new_game.get_moves())
        for opp_move in opponent_moves:
            opp_target_square = opp_move[2:4]
            own_piece = new_game.board.get_piece(Game.xy2i(opp_target_square))
            if own_piece != ' ' and own_piece.isupper():  # Assuming bot plays as White
                # Penalize based on the value of the endangered piece
                endangered_piece_value = PIECE_VALUES.get(own_piece.lower(), 0)
                move_score -= endangered_piece_value * 100  # Adjust penalty multiplier as needed
                #log_message(f"Own piece {own_piece} at {opp_target_square} is endangered. Penalizing move. Updated score: {move_score}")

        # Update the best move if the score is better
        if move_score > best_score:
            best_score = move_score
            best_move = move
            #log_message(f"New best move found: {best_move}. Best score updated to: {best_score}")

        move_end = time.time()
        #log_message(f"Move {move} evaluation completed in {move_end - move_start:.2f}s.")

    #log_message(f"Best move chosen: {best_move} with final score: {best_score}")
    return best_move if best_score > float('-inf') else None

def agent(obs, config):
    """
    Combined chess agent with total and per-move time limits.
    """
    try:
        start_time = time.time()
        total_time_limit = 8.0
        per_move_time_limit = 0.1

        #log_message("Starting agent evaluation")

        while True:
            elapsed_time = time.time() - start_time
            remaining_time = total_time_limit - elapsed_time

            #log_message(f"Total elapsed: {elapsed_time:.2f}s, Remaining time: {remaining_time:.2f}s")

            if remaining_time <= 0.1:  # Grace period to avoid timeout
                #log_message("Time is almost up. Defaulting to a random move.")
                break

            # Step 1: Use enhanced heuristic with per-move and remaining time limits
            move = enhanced_heuristic_with_time_limit(obs, per_move_time=per_move_time_limit, remaining_time=remaining_time)
            if move:
                #log_message(f"Heuristic selected move: {move}")
                return move

        # Step 2: Default to a random move if time is exhausted
        game = Game(obs.board)
        random_move = random.choice(list(game.get_moves()))
        #log_message(f"Random move chosen: {random_move}")
        return random_move

    except Exception as e:
        # Emergency fallback to a random move
        #log_message(f"Error in agent: {e}")
        game = Game(obs.board)
        fallback_move = random.choice(list(game.get_moves()))
        #log_message(f"Fallback random move chosen: {fallback_move}")
        return fallback_move
```

# Version comparison Initial Agent → 0_10_3 (Claude 3.5 generated)

# <div style="background-color:#0A0F29; font-family:'Poppins', cursive; color:#E0F7FA; font-size:140%; text-align:center; border: 2px solid #00FFFF; border-radius:15px; padding: 15px; box-shadow: 5px 5px 20px rgba(0, 0, 0, 0.5); font-weight: bold; letter-spacing: 1px; text-transform: uppercase;">Testing the agent</div>

# Test against random moves agent

```python
result = env.run(["LittleDeepBlue_0_10_4_debug.py", "random"])
for agent in result[-1]:
    print("Status:", agent.status, "/ Reward:", agent.reward, "/ Time left:", agent.observation.remainingOverageTime)
env.render(mode="ipython", width=700, height=700)
```

# Test against itself

```python
result = env.run(["LittleDeepBlue_0_10_4_debug.py", "LittleDeepBlue_0_10_4.py"])
for agent in result[-1]:
    print("Status:", agent.status, "/ Reward:", agent.reward, "/ Time left:", agent.observation.remainingOverageTime)
env.render(mode="ipython", width=700, height=700)
```

# Test against the intial agent

```python
result = env.run(["LittleDeepBlue_0_10_4_debug.py", "initial_agent.py"])
for agent in result[-1]:
    print("Status:", agent.status, "/ Reward:", agent.reward, "/ Time left:", agent.observation.remainingOverageTime)
env.render(mode="ipython", width=700, height=700)
```

# Test against the previous agent white and black

```python
# Define the number of games to be played
num_games = 50

# Initialize variables to track results
wins = [0, 0]  # Wins for each agent
draws = 0
rewards = [0, 0]  # Total rewards for each agent

# Play the games
for game in range(num_games):
    result = env.run(["LittleDeepBlue_0_10_4.py", "initial_agent.py"])
    
    # Process results for the game
    for i, agent in enumerate(result[-1]):
        # Update win/draw counts
        if agent.status == "WINNER":
            wins[i] += 1
        elif agent.status == "DRAW":
            draws += 1
        
        # Safely handle None reward
        rewards[i] += agent.reward if agent.reward is not None else 0

    # Optionally log game results without printing "Timeout"
    agent_1_status = result[-1][0].status
    agent_2_status = result[-1][1].status

    if agent_1_status != "Timeout" and agent_2_status != "Timeout":
        print(f"Game {game + 1}: Agent 1 Status: {agent_1_status}, Agent 2 Status: {agent_2_status}")

# Calculate average rewards
avg_rewards = [reward / num_games for reward in rewards]

# Display final statistics
print(f"\nTotal games played: {num_games}")
print(f"Agent 1 Wins: {wins[0]}")
print(f"Agent 2 Wins: {wins[1]}")
print(f"Draws: {draws}")
print(f"Agent 1 Average Reward: {avg_rewards[0]}")
print(f"Agent 2 Average Reward: {avg_rewards[1]}")

# Optionally, render the last game
env.render(mode="ipython", width=700, height=700)
```

```python
# Define the number of games to be played
num_games = 50

# Initialize variables to track results
wins = [0, 0]  # Wins for each agent
draws = 0
rewards = [0, 0]  # Total rewards for each agent

# Play the games
for game in range(num_games):
    result = env.run(["initial_agent.py", "LittleDeepBlue_0_10_4.py"])
    
    # Process results for the game
    for i, agent in enumerate(result[-1]):
        # Update win/draw counts
        if agent.status == "WINNER":
            wins[i] += 1
        elif agent.status == "DRAW":
            draws += 1
        
        # Safely handle None reward
        rewards[i] += agent.reward if agent.reward is not None else 0

    # Optionally log game results without printing "Timeout"
    agent_1_status = result[-1][0].status
    agent_2_status = result[-1][1].status

    if agent_1_status != "Timeout" and agent_2_status != "Timeout":
        print(f"Game {game + 1}: Agent 1 Status: {agent_1_status}, Agent 2 Status: {agent_2_status}")

# Calculate average rewards
avg_rewards = [reward / num_games for reward in rewards]

# Display final statistics
print(f"\nTotal games played: {num_games}")
print(f"Agent 1 Wins: {wins[0]}")
print(f"Agent 2 Wins: {wins[1]}")
print(f"Draws: {draws}")
print(f"Agent 1 Average Reward: {avg_rewards[0]}")
print(f"Agent 2 Average Reward: {avg_rewards[1]}")

# Optionally, render the last game
env.render(mode="ipython", width=700, height=700)
```

# <div style="background-color:#0A0F29; font-family:'Poppins', cursive; color:#E0F7FA; font-size:140%; text-align:center; border: 2px solid #00FFFF; border-radius:15px; padding: 15px; box-shadow: 5px 5px 20px rgba(0, 0, 0, 0.5); font-weight: bold; letter-spacing: 1px; text-transform: uppercase;">Code Explanation (LLM-Generated)</div>

# Chess Bot Code Breakdown

# <div style="background-color:#0A0F29; font-family:'Poppins', cursive; color:#E0F7FA; font-size:140%; text-align:center; border: 2px solid #00FFFF; border-radius:15px; padding: 15px; box-shadow: 5px 5px 20px rgba(0, 0, 0, 0.5); font-weight: bold; letter-spacing: 1px; text-transform: uppercase;">To Submit</div>

```python
# https://www.kaggle.com/code/kmldas/easy-submission-for-chess-bot
from shutil import copyfile
copyfile("LittleDeepBlue_0_5_5.py", "submission.py")
```

1. Download (or save) main.py
2. Go to the [submissions page](https://www.kaggle.com/competitions/fide-google-efficiency-chess-ai-challenge/submissions) and click "Submit Agent"
3. Upload main.py
4. Press Submit!

# <div style="background-color:#0A0F29; font-family:'Poppins', cursive; color:#E0F7FA; font-size:140%; text-align:center; border: 2px solid #00FFFF; border-radius:15px; padding: 15px; box-shadow: 5px 5px 20px rgba(0, 0, 0, 0.5); font-weight: bold; letter-spacing: 1px; text-transform: uppercase;">References</div>

- https://www.chessprogramming.org/Main_Page
- https://www.freecodecamp.org/news/simple-chess-ai-step-by-step-1d55a9266977/
- https://blogs.cornell.edu/info2040/2022/09/30/game-theory-how-stockfish-mastered-chess/
- https://www.cs.cmu.edu/afs/cs.cmu.edu/academic/class/15211-f04/www/hw6/
- https://www.youtube.com/watch?v=5y2a0Zhgq0U
- https://www.youtube.com/watch?v=SLgZhpDsrfc
- https://www.youtube.com/watch?v=l-hh51ncgDI
- https://github.com/thomasahle/sunfish
- https://github.com/apostolisv/chess-ai
- https://www.youtube.com/watch?v=CdFLEfRr3Qk