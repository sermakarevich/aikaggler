# NeurIPS 2025 - Google Code Golf Championship

- **Author:** Michael D. Moffitt
- **Votes:** 269
- **Ref:** mmoffitt/neurips-2025-google-code-golf-championship
- **URL:** https://www.kaggle.com/code/mmoffitt/neurips-2025-google-code-golf-championship
- **Last run:** 2025-08-22 17:52:43.957000

---

This starter notebook for the [2025 Google Code Golf Championship](https://www.kaggle.com/competitions/google-code-golf-2025) is designed to help contestants verify the functional correctness of their programs.  It allows one to load example pairs for a task, visualize them, and test whether a candidate code snippet produces expected results across all competition benchmarks.

## Enter a task number (between 1 and 400):

```python
task_num = 0  # Task 0 is just an illustrative example (and not eligible for points)
```

## Color legend:

```python
import sys
sys.path.append("/kaggle/input/google-code-golf-2025/code_golf_utils")
from code_golf_utils import *
show_legend()
```

## Example <input, output> pairs:

```python
examples = load_examples(task_num)
show_examples(examples['train'] + examples['test'])
```

## Enter your code to solve the task:

```python
%%writefile task.py
def p(g):
 for r, row in enumerate(g):
  for c, color in enumerate(row):
   if r and c and color==5 and g[r-1][c-1] not in [0,5]: g[r][c]=0
 return g
```

## Verify your code:

```python
verify_program(task_num, examples)
```