# LB over 2,000 with C++ Stockfish 4.0

- **Author:** GarrieD
- **Votes:** 125
- **Ref:** garried/lb-over-2-000-with-c-stockfish-4-0
- **URL:** https://www.kaggle.com/code/garried/lb-over-2-000-with-c-stockfish-4-0
- **Last run:** 2024-12-17 00:24:39.820000

---

After spending time in the python wilderness coding my engine from first principles, I came to the conclusion that best results will ultimately come from a C/C++ compile of an established engine.
[https://www.kaggle.com/competitions/fide-google-efficiency-chess-ai-challenge/discussion/550291](http://)

Given the size and 10 second time constraint of this competition I went for Stockfish 4.0 which peaked at about LB 2080 in the first hour of submission. Check the following link to download other versions of Stockfish:

[https://drive.google.com/drive/folders/1nzrHOyZMFm4LATjF5ToRttCU0rHXGkXI](http://)

Not an expert, but I think Stockfish 5 and above require tablebase support which might be difficult given the competition resource constraints. Any recent version using an efficiently updatable neural network (NNUE) is probably too large. 

Haven't tried Stockfish 4.5 yet.

```python
!pip install pygame Chessnut
!pip install --upgrade kaggle-environments
from kaggle_environments import make
env = make("chess", debug=True)
```

```python
%cd /kaggle/input/stockfish-4-win/stockfish-4-win/src_c++11
#credit chatGPT
!gcc-9 -std=c++17 -O2 -D_GLIBCXX_USE_CXX11_ABI=1 -o /kaggle/working/stockf *.cpp -lstdc++ -lm -lpthread
```

```python
%%writefile /kaggle/working/main.py

# credit: https://www.kaggle.com/code/olegzholobov/source-2100-lb-c-c-sample-submission

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
     

    engine_path = '/kaggle_simulations/agent/stockf'
    #engine_path = '/kaggle/working/stockf'
    if ultima is None:
        ultima = ChessEngine(engine_path)

    # Get the best move from the engine
    best_move = ultima.get_best_move(fen)

    return best_move
```

```python
%cd /kaggle/working
!tar -czvf submission.tar.gz main.py stockf
```

```python
print('Done')
```