# 🤯 running C/C++ in submissions

- **Author:** Filip Strzałka
- **Votes:** 106
- **Ref:** snufkin77/running-c-c-in-submissions
- **URL:** https://www.kaggle.com/code/snufkin77/running-c-c-in-submissions
- **Last run:** 2024-11-26 15:56:55.613000

---

This notebook is inspired by these discussions, credits for the authors and comments for valuable insights and experiments:

https://www.kaggle.com/competitions/fide-google-efficiency-chess-ai-challenge/discussion/547025

https://www.kaggle.com/competitions/fide-google-efficiency-chess-ai-challenge/discussion/547244

You can find more info about the environment and image used for the submissions in the latter.

I tried around 50 different ways total of running C/C++ code in a submission. 😱 The issues included:
- not getting output/able to pass input while invoking a shell command through python
- different tools being able to run subprocesses in different ways, hardly anything working exactly as intended
- getting dummy (e.g. echo) commands fine, but getting submission agent timeout after trying anything else (like compiling code remotely with gcc)
- hitting the submission size limit trying to upload even the smallest executable produced by gcc pn my local PC
- failing to run an executable due to platform differences
- .zip archives being submitted ok, but working strangely despite the submission mprompt allowing ZIPs

So, to save You the struggle:

## FINALLY, one complete working solution:

...that I found for C code (C++ should work too).

1. compile C source on a machine with the same os/architecture that runs in the submission engine, the most handy way is to locally run a Docker container with the base python image: `docker pull gcr.io/kaggle-images/python` HOWEVER it can also be done in a notebook:

```python
%%writefile main.c
#include <stdio.h>
int main()
{
    int a, b, c;
    scanf("%d", &a);
    scanf("%d", &b);
    scanf("%d", &c);

    char fen[100];
    scanf("%s", fen);

    /* ... some logic here ... */

    printf("c8d6");
    return 0;
}
```

```python
!gcc main.c
```

2. Use `subprocess` module to run the executable from python (in the example below I show a way to pass input via stdin). Save the python source file containing the acting function as `main.py` (this is required if submitting an archive! - You can add more `.py` files though and import them in the main).

```python
%%writefile main.py
import random
import subprocess
from collections import defaultdict

from Chessnut import Game


def chess_bot_with_c(obs):

    input_args = [1, 2, 345, obs.board]  # whatever You want to pass to the binary
    
    r = subprocess.run(['/kaggle_simulations/agent/a.out'], 
                       input='\n'.join(map(str, input_args)) + '\n',
                       encoding='utf-8', capture_output=True)
    output = r.stdout.strip()
    print(output)
    
    # ...
    # some more logic maybe ...
    
    game = Game(obs.board)
    moves = list(game.get_moves())
    
    # Greedy algorithm: choose best-valued position
    values = defaultdict(float)
    values.update(zip('pbnrqPBNRQ', [(1 if game.state.player == 'w' else -1) * v for v in [-1, -3, -3, -5, -9, 1, 3, 3, 5, 9]]))

    best_score = -1000
    best_move = moves[0]
    for move in moves:
        g = Game(obs.board)
        g.apply_move(move)
        if g.status == Game.CHECKMATE:
            return move
        score = sum(values[c] for c in str(g).split(' ')[0])
        if score >= best_score:
            best_score = score
            best_move = move

    return best_move
```

3. compress into a `.tar.gz` having the two (python source main + output binary) in the root of the dir structure - then submit the archive directly :)

```python
!tar -czvf submission.tar.gz main.py a.out
```

## Thant's it! Happy coding in C!
## (down the inner circles of hell shall we go... 😖)

please upvote if You found this useful :)