# BBC (BitBoard Chess engine in C)

- **Author:** aDg4b
- **Votes:** 145
- **Ref:** egrehbbt/bbc-bitboard-chess-engine-in-c
- **URL:** https://www.kaggle.com/code/egrehbbt/bbc-bitboard-chess-engine-in-c
- **Last run:** 2024-12-02 04:52:47.460000

---

# BBC (Bit Board Chess)

 - UCI chess engine by Code Monkey King
 - written for didactic purposes
 - covered in [95 YouTube video series](https://www.youtube.com/watch?v=QUNP-UjujBM&list=PLmN0neTso3Jxh8ZIylk74JpwfiWNI76Cs)

Source: https://github.com/maksimKorzh/bbc

Here, we will use version 1.2 because it is the latest version that does not use NNUE, so it will fit into the 64KiB compressed submission size limit.

```python
! wget https://raw.githubusercontent.com/maksimKorzh/bbc/refs/heads/master/src/old_versions/bbc_1.2.c
```

There is only one thing we need to do with the code - it is to reduce the size of the hash table so that we fit into the 5 Mib limit

```python
with open("bbc_1.2.c", "r") as f:
    lines = f.readlines()

with open("bbc_1.2.c", "w") as f:
    for line in lines:
        if "init_hash_table(64);" in line:
            line = line.replace("init_hash_table(64)", "init_hash_table(1)")
        f.write(line)
```

```python
# compile the code

! gcc -Ofast bbc_1.2.c -o bbc
```

```python
%%writefile main.py

import os
import atexit
import subprocess


if os.path.exists("/kaggle_simulations"):
    engine_file_path = "/kaggle_simulations/agent/bbc"
else:
    engine_file_path = "./bbc"


class Engine:
    # https://www.kaggle.com/competitions/fide-google-efficiency-chess-ai-challenge/discussion/548061
    
    def __init__(self, engine_file):
        self.engine_file = engine_file
        self.bestmove = ""
        self.engine_process = None

    def write(self, input_text):
        self.engine_process.stdin.write(input_text)
        self.engine_process.stdin.flush()

    def connect(self):
        self.engine_process = subprocess.Popen([self.engine_file], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                               universal_newlines=True)

    def listen(self):
        while True:
            response = self.engine_process.stdout.readline().strip()
            tokens = response.split()
            # print(tokens)
            if len(tokens) >= 2 and tokens[0] == "bestmove":
                self.bestmove = tokens[1]
                break

    def think(self, allocated_time, fen):
        self.write("position fen " + fen + "\n")
        self.write("go wtime " + str(allocated_time) + " btime " + str(allocated_time) + "\n")

        self.listen()

    def cleanup(self):
        if self.engine_process:
            self.engine_process.kill()
            self.engine_process = None

engine = Engine(engine_file_path)

def cleanup_process():
    global engine
    engine.cleanup()

atexit.register(cleanup_process)
engine.connect()

def main(obs):
    fen = obs.board
    time_left = obs.remainingOverageTime * 1000

    engine.think(time_left, fen)

    return engine.bestmove
```

```python
%%capture
# ensure we are on the latest version of kaggle-environments
!pip install --upgrade kaggle-environments
```

```python
from kaggle_environments import make
env = make("chess", debug=True)

result = env.run(["main.py", "main.py"])
env.render(mode="ipython", width=600, height=600)
```

```python
# create a submission

!tar -czvf submission.tar.gz main.py bbc
```