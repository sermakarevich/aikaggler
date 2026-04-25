# Chess AI Essentials: A Practical Guide

- **Author:** Vitaly Kudelya
- **Votes:** 92
- **Ref:** vitalykudelya/chess-ai-essentials-a-practical-guide
- **URL:** https://www.kaggle.com/code/vitalykudelya/chess-ai-essentials-a-practical-guide
- **Last run:** 2025-01-23 19:19:21.793000

---

This work is based on the following notebooks: 🙏
- [source 2100 LB c/c++ sample submission](https://www.kaggle.com/code/olegzholobov/source-2100-lb-c-c-sample-submission)
- [Memory Tracking for UCI Engines](https://www.kaggle.com/code/sircausticmail/memory-tracking-for-uci-engines)
- [Your First Chess Bot](https://www.kaggle.com/code/bovard/your-first-chess-bot)

I spent quite a bit of time figuring out the various mechanics of this competition, and gathered the fundamentals of creating and monitoring chess bots.

### Table of Contents

* [Competition mechanics](#competition-mechanics)
* [C/C++ bot](#c-bot)
* [Local simulation](#local-simulation)
* [Memory tracking](#memory-tracking)

# Competition mechanics
<a name="competition-mechanics"></a>

In this Kaggle competition, you're challenged to create an  chess bot that will compete against other bots.

**Constraints**
- 5 MiB of RAM
- Dedicated CPU: a single 2.20GHz core
- 64KiB compressed submission size limit

It seems like the easiest way to understand the mechanics is with a simple example of a bot that chooses a random move.
Let's move on to that.

### Random Chess Bot

```python
%%writefile random_bot.py
#%%writefile is a magic command in Notebooks that allows you to write the contents of a cell to a file

from Chessnut import Game
import random

def chess_bot(obs):
    """
    Simple chess bot that makes random moves.

    Args:
        obs: An object with a 'board' attribute representing the current board state as a FEN string.

    Returns:
        A string representing the chosen move in UCI notation (e.g., "e2e4")
    """
    # 0. Parse the current board state and generate legal moves using Chessnut library
    game = Game(obs.board)
    moves = list(game.get_moves())
    return random.choice(moves)
```

**Communication and Game State:**

Bots interact with the game environment through a single function. The current game state is provided to this function via the `obs` variable.

**Understanding the `obs` Variable:**

`obs` is a dictionary-like object (type `kaggle_environments.utils.Struct` see below) with the following fields: <br />
> `board`: A string representing the chessboard position in Forsyth–Edwards Notation (FEN). <br />
> `mark`: A string indicating the bot's color ("white" or "black"). <br />
> `remainingOverageTime`: The bot's remaining time in seconds. <br />
> `opponentRemainingOverageTime`: The opponent's remaining time in seconds. <br />
> `lastMove`: The opponent's last move in UCI notation (e.g., "h6f4"). <br />
> `step`: The current move number.

Example:
```python
{
    'board': '2B5/8/2k5/2p5/r3Nq1P/P3P3/P1P5/2K1R1NR w - - 0 26',
    'mark': 'white',
    'remainingOverageTime': 10,
    'opponentRemainingOverageTime': 10,
    'lastMove': 'h6f4',
    'step': 50
}
```

**Bot Output:**

The bot function must return a string representing its chosen move in UCI notation (e.g., "e2e4").


```python
# kaggle_environments.utils.Struct definition
class Struct(dict):
    def __init__(self, **entries):
        entries = {k: v for k, v in entries.items() if k != "items"}
        dict.__init__(self, entries)
        self.__dict__.update(entries)

    def __setattr__(self, attr, value):
        self.__dict__[attr] = value
        self[attr] = value
```

# C/C++ bot
<a name="c-bot"></a>

This section demonstrates how to use a C-based chess engine. Since Python can be slower for complex calculations, and many top-performing chess engines are written in C++, we'll explore using the [Fruit](https://github.com/Warpten/Fruit-2.1) chess engine from the `chess_fruit` dataset.

To use a C-based chess engine, we first need to compile its source code into an executable file. This step isn't necessary for Python, as it's an interpreted language that runs directly without prior compilation.  

Compiling C-based engines can be complex, and the instructions for doing so are typically found in a `Makefile`.

```python
#copy code inside working directory
!cp -r /kaggle/input/chess-fruit/chess_fruit/* /kaggle/working

# create executable file fruit
!make all
```

```python
%%writefile /kaggle/working/main.py
# create a Python API to interact with the compiled chess engine

import subprocess


class ChessEngine:
    def __init__(self, engine_path):
        # Start the engine process
        self.engine = subprocess.Popen(
            [engine_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        self._initialize_engine()

    def _initialize_engine(self):
        # Initialize the engine with UCI protocol
        self._send_command("uci")
        while True:
            output = self._read_output()
            if output == "uciok":
                break

        # Set engine options to minimize memory usage
        self._send_command("setoption name Hash value 1")

    def _send_command(self, command):
        """Send a command to the engine."""
        self.engine.stdin.write(command + "\n")
        self.engine.stdin.flush()

    def _read_output(self):
        """Read a single line of output from the engine."""
        output = self.engine.stdout.readline().strip()
        return output

    def get_best_move(self, obs, movetime=100):
        """Get the best move for a given position."""
        
        if obs['mark'] == 'white':
            wtime = int(obs['remainingOverageTime'] * 1000)
            btime = int(obs['opponentRemainingOverageTime'] * 1000)
        else:
            btime = int(obs['remainingOverageTime'] * 1000)
            wtime = int(obs['opponentRemainingOverageTime'] * 1000)
        
        # Set the position
        self._send_command(f"position fen {obs['board']}")

        # Start the search providing current remaining time
        self._send_command(f"go wtime {wtime} btime {btime} winc 85 binc 85")

        # Wait for the best move
        best_move = None
        while True:
            output = self._read_output()
            if output.startswith("bestmove"):
                best_move = output.split()[1]
                break

        return best_move

    def stop(self):
        """Stop the engine process."""
        self._send_command("quit")
        self.engine.terminate()
        self.engine.wait()


def is_jupyter_run():
  try:
    shell = get_ipython().__class__.__name__
    if shell == 'ZMQInteractiveShell':
      return True
    else:
      return False
  except Exception:
    return False 

# absolute path for local run and submitted package should be different
if is_jupyter_run():
    engine_path = '/kaggle/working/fruit'
else:
    engine_path = '/kaggle_simulations/agent/fruit'
    
chess_engine = ChessEngine(engine_path)

def chess_bot(obs):
    # Get the best move from the engine
    best_move = chess_engine.get_best_move(obs)
    return best_move
```

```python
# archive and compress data into submission.tar.gz for submission
!tar -czvf submission.tar.gz main.py fruit
```

# Local Simulation
<a name="local-simulation"></a>

```python
!pip install --upgrade kaggle-environments
```

```python
from kaggle_environments import make
```

```python
env = make("chess", debug=True)

# simulate a game between our random_bot and fruit chess engine
result = env.run(["random_bot.py", "main.py"])
env.render(mode="ipython", width=800, height=800)
```

# Memory tracking
<a name="memory-tracking"></a>

Memory usage is crucial in this competition due to the 5 MiB RAM limit. <br />
The following code displays memory usage. <br />
You can also try `valgrind` for memory tracking.

```python
import psutil
import subprocess
import time

# Function to get memory usage of a specific process
def get_memory_usage(process):
    try:
        mem_info = process.memory_info()
        return mem_info.rss / (1024 * 1024)  # Convert to MB
    except psutil.NoSuchProcess:
        return 0

# Function to monitor a UCI engine
def monitor_uci_engine(engine_path, commands):
    # Start the UCI engine as a subprocess
    process = subprocess.Popen(
        engine_path, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    p = psutil.Process(process.pid)
    
    try:
        # Send commands to the UCI engine
        for command in commands:
            process.stdin.write(f"{command}\n")
            process.stdin.flush()
            time.sleep(1)  # Allow some time for processing
            
            # Measure memory usage
            memory_usage = get_memory_usage(p)
            print(f"Memory Usage after '{command}': {memory_usage:.2f} MB")
        
        # Wait for a while to observe idle memory usage
        time.sleep(5)
        idle_memory = get_memory_usage(p)
        print(f"Idle Memory Usage: {idle_memory:.2f} MB")
    
    finally:
        # Terminate the process
        process.stdin.write("quit\n")
        process.stdin.flush()
        process.terminate()
        process.wait()

# Path to your UCI engine executable (adjust as needed)
uci_engine_path = f"/kaggle/working/fruit"

# List of commands to send to the engine
uci_commands = [
    "uci",
    "isready", 
    "position startpos moves e2e4 e7e5",
    "go movetime 100"
]

monitor_uci_engine(uci_engine_path, uci_commands)
```