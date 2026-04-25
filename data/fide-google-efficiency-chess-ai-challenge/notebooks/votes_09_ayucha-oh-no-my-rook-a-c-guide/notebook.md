# OH NO MY ROOK ♖ A C++ Guide

- **Author:** ayucha
- **Votes:** 86
- **Ref:** ayucha/oh-no-my-rook-a-c-guide
- **URL:** https://www.kaggle.com/code/ayucha/oh-no-my-rook-a-c-guide
- **Last run:** 2024-12-17 23:32:23.467000

---

# OH NO MY ROOK ♖ 
## A C++ Guide for FIDE & Google Efficient Chess AI Challenge
This work is inspired by the work of [olegzholobov](https://www.kaggle.com/code/olegzholobov/source-2100-lb-c-c-sample-submission).

Alright, let’s be real—chess is pretty awesome, even if I’m not exactly a grandmaster. 😅 I’ve always been drawn to the game because it’s like a mental battlefield where every move matters. But honestly, what really made me fall in love with chess was watching GothamChess on Youtube. Dude makes chess fun and totally relatable. 🧠💥 He’s the perfect combo of entertaining and educational, and the way he breaks things down makes me actually want to get better at this game. I’m hoping to channel some of that energy into this competition, by sacrificing MY ROOK.


Now, here we are—I'm jumping into this Kaggle chess competition, and the goal is to build a program that plays efficiently, without relying on the usual crazy search trees or huge pre-computed tables. It’s all about being smart with how the program thinks and plays. And since I’m all about that performance (and not about wasting time), I’m using C++ chess engine. ⚡ It’s fast, it’s powerful, and it’s perfect for optimizing things like decision-making without slowing down.

I’m keeping things chill and using a public GitHub repo that’s already got all the pieces set up (pun intended 😉). But the real kicker is that I’ll be building on top of the [Micromax Chess Engine](https://home.hccnet.nl/h.g.muller/max-src2.html). This engine is well-known for its efficiency and solid performance, making it the perfect base for my AI. I’ll be tweaking it to optimize the strategy and performance, making sure the engine plays efficiently and smartly.

Let’s get this code rolling and maybe make Gotham proud 🎮♟️ and make some rook sacs along the way...

```python
import IPython
IPython.display.Image(filename="/kaggle/input/assets/2024_12_fide_and_google_efficient_chess_ai_challenge_oh_no_my_rook.jpg", width = 500, height = 500)
```

```python
import requests
requests.get('http://www.google.com',timeout=10).ok
```

```python
%%capture
# ensure we are on the latest version of kaggle-environments
!pip install --upgrade kaggle-environments
```

## What’s Micromax Chess Engine? 🤔♟️
The Micromax Chess Engine is like the cool minimalist of the chess engine world. 😎 Written in C, it’s super compact—literally just a few kilobytes of code! Despite being tiny, it’s got some serious game and can play solid chess without hogging resources. 🏎️💨

I plan to use an easier to understand implementation of [Micromax engine by Gissio](https://github.com/Gissio/mcu-max).

```python
!rm -rf mcu-max
!git clone https://github.com/Gissio/mcu-max.git
!mkdir -p build && cd build && cmake ../mcu-max/examples/mcu-max-uci && make
!cp build/mcu-max-uci .
```

```python
# Now let's set up the chess environment!
from kaggle_environments import make
env = make("chess", debug=True)
```

```python
%%writefile main.py
from Chessnut import Game
import subprocess
import os
import random


class ChessEngine:
    def __init__(self, engine_path):
        try:
            self.engine = subprocess.Popen(
                [engine_path],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
        except:
            self.engine = subprocess.Popen(
                ['./mcu-max-uci'],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
        self._initialize_engine()

    def _initialize_engine(self):
        self._send_command("uci")
        while True:
            output = self._read_output()
            if output == "uciok":
                break

        self._send_command("setoption name Hash value 1")

    def _send_command(self, command):
        self.engine.stdin.write(command + "\n")
        self.engine.stdin.flush()

    def _read_output(self):
        output = self.engine.stdout.readline().strip()
        return output

    def get_best_move(self, fen, movetime = 100):
        self._send_command(f"position fen {fen}")
        self._send_command(f"go movetime {movetime}")
        best_move = None
        while True:
            output = self._read_output()
            if output.startswith("bestmove"):
                best_move = output.split()[1]
                break
        self._send_command("setoption name Clear Hash")
        return best_move

    def stop(self):
        self._send_command("quit")
        self.engine.terminate()
        self.engine.wait()

ultima = None
def chess_bot(obs):
    global ultima
    fen = obs['board']
    engine_path = '/kaggle_simulations/agent/mcu-max-uci'
    if ultima is None:
        ultima = ChessEngine(engine_path)
    best_move = ultima.get_best_move(fen)

    # Fix for an issue where the promoting move does not actually promote pawns.
    # Changes a pawn move of f7f8 to f7f8q which is recognised.
    if ((best_move[1] == '7' and best_move[3] == '8') or (best_move[1] == '2' and best_move[3] == '1')):
        game = Game(fen)  
        moves = list(game.get_moves())

        promote = False
        for move in moves:
            if len(move) == 5 and best_move in move:
                promote = True
                break

        if promote:
            best_move += 'q'
    return best_move
```

```python
result = env.run(["main.py", "random"])
print("Agent exit status/reward/time left: ")
for agent in result[-1]:
    print("\t", agent.status, "/", agent.reward, "/", agent.observation.remainingOverageTime)
print("\n")
# render the game
env.render(mode="ipython", width=700, height=700)
```

Ahh, the Sicilian Defense—a timeless classic of chess, much like the challenge we’re tackling here.

```python
IPython.display.Image(filename="/kaggle/input/assets/2024_12_fide_and_google_efficient_chess_ai_challenge_hikaru_lines.jpg", width = 500, height = 500)
```

I'm actually joking, I am not super acquainted with chess openings, lines and stuff. I have a rating of about 1100 on Chess.com and if my bot surpasses that (actually it could only reach ~800 on LB), I will learn chess more deeply. Let's just wrap the code for submission.

```python
%cd /kaggle/working
!tar -czvf submission.tar.gz main.py mcu-max-uci
```

With you as my trusty knight 🐴 (or maybe bishop, depending on the position), I’m counting on you, buddy, to carry me through this competition. Whether it’s outsmarting other programss or holding the line in a tough match, we’ve got this! It’s not just about winning—it’s about the journey we’re on together. Let’s make every move count, and when the moment comes, let’s deliver that clean mate in the endgame. You’ve got this!

P.S. I was watching the final game of the chess world championship 2024 while writing this notebook and it was such a nice game. Congratulations youngest world champion.

```python
IPython.display.Image(filename="/kaggle/input/assets/2024_12_fide_and_google_efficient_chess_ai_challenge_wcc2024.png", width = 500, height = 500)
```