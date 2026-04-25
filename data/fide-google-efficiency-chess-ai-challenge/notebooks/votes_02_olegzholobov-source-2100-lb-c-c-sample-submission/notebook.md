# source 2100 LB c/c++ sample submission

- **Author:** Oleg Zholobov
- **Votes:** 365
- **Ref:** olegzholobov/source-2100-lb-c-c-sample-submission
- **URL:** https://www.kaggle.com/code/olegzholobov/source-2100-lb-c-c-sample-submission
- **Last run:** 2024-11-26 17:54:55.850000

---

# hi there!
## this is my first notebook share and today i want to show you how i got over 700 lb

### telegram @ishowfit

```python
# pip install libraries
!pip install pygame Chessnut
!pip install --upgrade kaggle-environments
```

```python
from kaggle_environments import make
env = make("chess", debug=True)
```

## This is edited ( someway ) code of Fruit 2.1 chess engine

why this engine: because i was finding something good and lightweight ( without syzygy, nnue etc )

```python
%cd /kaggle/input/fruktik/frukt
!gcc -std=c++11 -O2 -o /kaggle/working/frukt *.cpp
```

```python
%%writefile /kaggle/working/main.py


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

    def get_best_move(self, fen, movetime=100):
        """Get the best move for a given position."""
        # Set the position
        self._send_command(f"position fen {fen}")

        # Start the search
        self._send_command(f"go movetime {movetime}")

        # Wait for the best move
        best_move = None
        while True:
            output = self._read_output()
            if output.startswith("bestmove"):
                best_move = output.split()[1]
                break

        # Clear the engine's internal cache to minimize memory usage
        self._send_command("setoption name Clear Hash")

        return best_move

    def stop(self):
        """Stop the engine process."""
        self._send_command("quit")
        self.engine.terminate()
        self.engine.wait()


# Define a global variable to store the ChessEngine instance
ultima = None

def chess_bot(obs):
    global ultima  # Declare ultima as global to modify it
    fen = obs['board']


    '''
    
    Comment engine_path with /kaggle_simulations/...  and %%writefile to define func and test locally or 
    comment engine_path with /kaggle/working/ ... to save file and then zip it
    '''
     

    engine_path = '/kaggle_simulations/agent/frukt'
    # engine_path = '/kaggle/working/frukt'
    if ultima is None:
        ultima = ChessEngine(engine_path)

    # Get the best move from the engine
    best_move = ultima.get_best_move(fen)

    return best_move
```

```python
# result = env.run(['random',chess_bot])
# for agent in result[-1]:
#     print("Status:", agent.status, "/ Reward:", agent.reward, "/ Time left:", agent.observation.remainingOverageTime)
# env.render(mode="ipython", width=700, height=700)
```

```python
%cd /kaggle/working
!tar -czvf submission.tar.gz main.py frukt
```

# have fun and upvote if it was insightful for your chess journey