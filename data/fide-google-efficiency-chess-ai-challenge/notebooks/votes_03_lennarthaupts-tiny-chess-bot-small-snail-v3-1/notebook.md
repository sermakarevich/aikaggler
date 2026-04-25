# Tiny Chess Bot | Small Snail v3.1🐌

- **Author:** Lennart Haupts
- **Votes:** 221
- **Ref:** lennarthaupts/tiny-chess-bot-small-snail-v3-1
- **URL:** https://www.kaggle.com/code/lennarthaupts/tiny-chess-bot-small-snail-v3-1
- **Last run:** 2024-12-09 06:23:02.123000

---

```python
# first let's make sure you have internet enabled
import requests
requests.get('http://www.google.com',timeout=10).ok
```

```python
%%capture
# ensure we are on the latest version of kaggle-environments
!pip install --upgrade kaggle-environments

# Now let's set up the chess environment!
from kaggle_environments import make
env = make("chess", debug=True)
```

# Meet the Small Snail

A super simple chess agent that’s just learned how to move the pieces but has no idea where to go next.

So, what does it do? Well, it’s slow and starts panicking under time pressure, but aside from that:

**The Small Snail focuses on controlling the center and capturing valuable pieces, while trying (and often failing) to make favorable trades. It blunders and hangs pieces left and right, gets stuck in the middle of the board, and doesn’t know how to deliver a checkmate that isn’t a mate in one. This snail may not know what it's doing, but it's definitely ready to rumble.**

**What's new in V2 and V3?**

**V2:**
**The Small Snail now handles time better, using a decreasing time constraint for each search instead of the previous panic mode. Random noise was added to move evaluations, allowing for some amount of risk and a larger variety in moves. Parameters like `threshold_1`, `threshold_2`, `start_val`, and `temperature` were picked with a genetic algorithm, and some square-value maps were dropped entirely.**

**V3.1:**
**The decision-making is largely same as it was before, but I replaced the Chessnut library with custom bitboard functions (still Python) which are more tailored to my needs, making the Small Snail slightly faster.**

`If you plan to submit this code, make sure to package the utility functions and the main code together into a single tar.gz file.`

```python
%%writefile main.py
import random
import time
# import sys
# sys.path.append('/kaggle/input/bitboard-chess-custom-functions/')

from bitboard_utils import *

threshold_1 = -8.7
threshold_2 = 40
start_val = 1.35
temperature = 1.7
check_value = 10

# Value map for chess pieces
value_map = {
    " ": 0,
    "p": 10, "P": 10,  # Pawns
    "n": 30, "N": 30,  # Knights
    "b": 35, "B": 35,  # Bishops
    "r": 50, "R": 50,  # Rooks
    "q": 90, "Q": 90,  # Queens
    "k": 2000, "K": 2000
}

standard_board_map = {
    "a8": -1, "b8": 2, "c8": 3, "d8": 3, "e8": 3, "f8": 3, "g8": 2, "h8": -1,
    "a7": 0, "b7": 1, "c7": 1, "d7": 1, "e7": 1, "f7": 1, "g7": 1, "h7": 0,
    "a6": 0, "b6": 3, "c6": 4, "d6": 4, "e6": 4, "f6": 4, "g6": 3, "h6": 0,
    "a5": 0, "b5": 3, "c5": 5, "d5": 8, "e5": 8, "f5": 5, "g5": 3, "h5": 0,
    "a4": 0, "b4": 3, "c4": 5, "d4": 8, "e4": 8, "f4": 5, "g4": 3, "h4": 0,
    "a3": 0, "b3": 3, "c3": 4, "d3": 4, "e3": 4, "f3": 4, "g3": 3, "h3": 0,
    "a2": 0, "b2": 1, "c2": 1, "d2": 1, "e2": 1, "f2": 1, "g2": 1, "h2": 0,
    "a1": -1, "b1": 2, "c1": 3, "d1": 3, "e1": 3, "f1": 3, "g1": 2, "h1": -1
    }

king_board_map = {
    "a8": -2, "b8": -2, "c8": -3, "d8": -4, "e8": -4, "f8": -3, "g8": -2, "h8": -2,
    "a7": -2, "b7": -3, "c7": -4, "d7": -5, "e7": -5, "f7": -4, "g7": -3, "h7": -2,
    "a6": -3, "b6": -4, "c6": -5, "d6": -6, "e6": -6, "f6": -5, "g6": -4, "h6": -3,
    "a5": -4, "b5": -5, "c5": -6, "d5": -7, "e5": -7, "f5": -6, "g5": -5, "h5": -4,
    "a4": -4, "b4": -5, "c4": -6, "d4": -7, "e4": -7, "f4": -6, "g4": -5, "h4": -4,
    "a3": -3, "b3": -4, "c3": -5, "d3": -6, "e3": -6, "f3": -5, "g3": -4, "h3": -3,
    "a2": -2, "b2": -3, "c2": -4, "d2": -5, "e2": -5, "f2": -4, "g2": -3, "h2": -2,
    "a1": -2, "b1": -2, "c1": -3, "d1": -4, "e1": -4, "f1": -3, "g1": -2, "h1": -2
}

board_map = {
    10: standard_board_map,
    30: standard_board_map,
    35: standard_board_map,
    50: standard_board_map,
    90: standard_board_map,
    2000: king_board_map
}

# def check_for_simple_threat(opponent_moves, cell, piece_value):
#     # Checks if piece could be attacked by opponent's piece
#     for move in opponent_moves:
#         if cell == move[1]:
#             return piece_value
#     return 0

def check_for_threat(opponent_moves, b_map):
    best_capture_value = float("-inf")
    for move in opponent_moves:
        tmp_capture_value = value_map[get_piece_by_index(b_map, move[1])]
        if tmp_capture_value >= best_capture_value:
            best_capture_value = tmp_capture_value
    return best_capture_value

def find_move(b_map, color, time_limit):
    opponent_color = "b" if color == "white" else "w"
    color = "w" if color == "white" else "b"
    moves = get_moves(b_map, color)["moves"]
    best_move = None
    start = time.time()
    best_value = -100

    for i, move in enumerate(moves):
        if time.time() - start > time_limit:
            if best_move:
                return best_move
            else:
                return random.choice(moves)
                
        piece_value = value_map[get_piece_by_index(b_map, move[0])]
        capture_value = value_map[get_piece_by_index(b_map, move[1])]
        tmp_bitboards = apply_move(b_map, move)
        opponent_moves = get_moves(tmp_bitboards, opponent_color)

        if opponent_moves["status"] == "Checkmate":
            return move

        # Get opponent's moves and calculate potential capture value
        netto_value = capture_value - check_for_threat(opponent_moves["moves"], tmp_bitboards)
        
        if netto_value >= threshold_1:
            netto_value += board_map[piece_value][index_to_algebraic(move[1])] + random.uniform(-temperature, temperature)
            if opponent_moves["status"] == "Check":
                netto_value += check_value
            if netto_value > best_value:
                best_value = netto_value 
                best_move = move

    if best_move:
        return best_move
    return random.choice(moves)

def chess_bot(obs):
    b_map = fen_to_bitboards(obs.board)
    time_limit = start_val * (obs.remainingOverageTime/10)
    # Find the "best" move using the evaluation function
    move = move_to_algebraic(find_move(b_map, obs.mark, time_limit))
    return move
```

```python
!cp /kaggle/input/python-bitboard-chess/bitboard_utils.py .

!tar -czf submission.tar.gz main.py bitboard_utils.py
```

# Evaluation

**Evaluating the Small Snail's performance against itself and various public and non-public bots.**

```python
def play_games(env, agent1, agent2, num_games=10, render_last=True):
    results = {
        agent1: {"win": 0, "loss": 0, "timeouts": 0, "reward": 0},
        agent2: {"win": 0, "loss": 0, "timeouts": 0, "reward": 0},
    }

    for i in range(1, num_games + 1):
        print(f"Game: {i}")

        # Alternate which agent is White and which is Black
        if i % 2 == 1:
            white_agent, black_agent = agent1, agent2
        else:
            white_agent, black_agent = agent2, agent1

        result = env.run([white_agent, black_agent])

        # Print status for each agent at the end of the game
        for idx, agent in enumerate(result[-1]):
            color = "White" if idx == 0 else "Black"
            print(
                f"\tAgent {color} ({[white_agent, black_agent][idx]}): {agent.status} / {agent.reward} / {agent.observation.remainingOverageTime}"
            )

        # Update results for White
        if result[-1][0].reward == 1:
            results[white_agent]["win"] += 1
            results[black_agent]["loss"] += 1
            results[white_agent]["reward"] += 1
        elif result[-1][1].status == "TIMEOUT":
            results[white_agent]["reward"] += 1
        elif result[-1][0].reward == 0.5:
            results[white_agent]["reward"] += 0.5

        # Update results for Black
        if result[-1][1].reward == 1:
            results[black_agent]["win"] += 1
            results[white_agent]["loss"] += 1
            results[black_agent]["reward"] += 1
        elif result[-1][0].status == "TIMEOUT":
            results[black_agent]["reward"] += 1
        elif result[-1][1].reward == 0.5:
            results[black_agent]["reward"] += 0.5

        # Track timeouts
        if result[-1][0].status == "TIMEOUT":
            results[white_agent]["timeouts"] += 1
        if result[-1][1].status == "TIMEOUT":
            results[black_agent]["timeouts"] += 1

    # Report the results
    for agent, stats in results.items():
        print(f"\nSummary for: {agent}")
        print(f"Games Played: {num_games}")
        print(f"Timeouts: {stats['timeouts']}")
        print(f"Wins: {stats['win']}")
        print(f"Losses: {stats['loss']}")
        print(f"Average Reward: {stats['reward'] / num_games:.2f} (Timeouts as Win/Loss)")

    if render_last:
        env.render(mode="ipython", width=600, height=600)

    return results
```

```python
def print_bitboard(bitboard, piece_name):
    """Prints the bitboard occupancy map for a given piece."""
    binary_representation = bin(bitboard)[2:].zfill(64)  # Convert to binary, pad to 64 bits
    print(f"{piece_name} bitboard:")
    
    # Print the board in a human-readable format
    for i in range(8):
        row = binary_representation[i*8:(i+1)*8]
        print(" ".join(row))
    print()
```

```python
from bitboard_utils import *
from main import *

evaluate = True
n = 20

if evaluate:
    play_games(env, chess_bot, chess_bot, num_games=5, render_last=False) # own validation
    play_games(env, chess_bot, "/kaggle/input/chess-bot-minimax-alpha-beta-time-constraint/main.py", num_games=n)
    play_games(env, chess_bot, "/kaggle/input/your-first-chess-bot/main.py", num_games=n)
    play_games(env, chess_bot, "/kaggle/input/marginally-better-bot/submission.py", num_games=n)
    play_games(env, chess_bot, "/kaggle/input/micro-chess-bot-two-strat/submission.py", num_games=n)
    play_games(env, chess_bot, "/kaggle/input/littledeepblue/submission.py", num_games=n)
```