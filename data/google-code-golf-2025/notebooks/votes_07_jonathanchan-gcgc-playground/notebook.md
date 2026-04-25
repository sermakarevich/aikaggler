# GCGC playground

- **Author:** Jonathan Chan
- **Votes:** 140
- **Ref:** jonathanchan/gcgc-playground
- **URL:** https://www.kaggle.com/code/jonathanchan/gcgc-playground
- **Last run:** 2025-10-24 08:28:03.513000

---

#### This is a sandbox playground for manual golfing that combines elements from several pubicly shared notebooks. This is meant to run in interactive mode.

Version 1: Start with the current highest scoring public notebook: https://www.kaggle.com/code/tonylica/400-task-with-smart-solution-search-verification

Minor improvements were made to Task 285 for an improved score of 905145. There is much opportunity to reduce the code length in this puzzle.

Version 2: Retrieve improved version from this notebook and further improved Task 287 for total score of 905155.

Version 3: Significantly reduced original code size of Task 5 from 758 to 677 using Co-pilot, following the strategy of V2, and then further compressed using ChatGPT. However, the compressed length only improved by 10 for total score of 905165.

Version 4: A new high scoring notebook has been uploaded - https://www.kaggle.com/code/tonylica/community-baselines-my-arc-10-pack-2-10.

Version 5: Significant improvement in usability by using a script and further improvement in the visualization that includes the task score and difference from the best posted baseline. The frequently updated dataset at https://www.kaggle.com/datasets/tonylica/google-code-golf-2025-submit/code is used. Don't forget to check for updates each time you use this.

Minor Improvements:
 - from https://www.kaggle.com/code/paritoshtripathi5/public-task-33-down-to-129-bytes by using alias for a short code  (2 bytes)
 - from the first task in https://www.kaggle.com/code/jazivxt/dead-code by showing alias may not efficient when Zopfli compression is used (1 byte)

Version 6: This is the last sharing as the submission deadline is only a week from now. This version will use the latest ensembling from https://www.kaggle.com/code/mcwema/r30-neurips-golf-lessons-learned/notebook. The dataset used in V5 is no longer updated. You may use your own top scoring notebook instead. Again, don't forget to check for updates of all input.

Minor Improvements from a couple of recently posted notebooks:
 - from https://www.kaggle.com/code/paritoshtripathi5/public-task-398-down-to-84-bytes and using the trick posted at https://www.kaggle.com/code/paritoshtripathi5/public-task-398-down-to-84-bytes/comments#3305350 (2 bytes)
 - from https://www.kaggle.com/code/snnguynvnk19hl/pipeline-update-task15-190-144 and combined multiple assignments (1 byte)
 - for Task 5, the current public code score is 413, by specifying that the grids are fixed squares of 21x21, the code may be shortened. (10 bytes)

## Retrieval of solutions and preliminaries

```python
!mkdir submission

# Version 1
#!cp /kaggle/input/400-task-with-smart-solution-search-verification/submission/*.py /kaggle/working/submission

# Version 2&3 (don't forget to check for update to get latest codes)
#!cp /kaggle/input/gcgc-playground/submission/*.py /kaggle/working/submission

# Version 4
#!cp /kaggle/input/community-baselines-my-arc-10-pack-2-10/submission/*.py /kaggle/working/submission

# Version 5
!cp /kaggle/input/google-code-golf-2025-submit/task*.py /kaggle/working/submission

# Version 6
!cp /kaggle/input/r30-neurips-golf-lessons-learned/submission/task*.py /kaggle/working/submission
```

```python
# Import custom GCGC utilities
from gcgc_utils import *
```

## Select the task number and display puzzle and source code

```python
# Select puzzle and display
task_num = 398

gcgc_display(task_num)
```

#### Using the trick posted at https://www.kaggle.com/code/paritoshtripathi5/public-task-398-down-to-84-bytes/comments#3305350

```python
%%writefile task.py
def p(r):R,=r;S=len({*R}-{0})*5;return[([0]*d+R+[0]*S)[:S]for d in range(S)][::-1]
```

```python
# Verify using the official GCGC tool with visualization of incorrect code
verify_program(task_num, load_examples(task_num))
```

```python
# Update the code and compress it first as necessary. 
# This may show your code is NOT READY for submission, if the modification fails; however, it would still show the compressed size
test_code(task_num)
```

```python
# Select puzzle and display
task_num = 15

gcgc_display(task_num)
```

#### Combine multiple assignments

```python
%%writefile task.py
def p(g):
 h=eval(str(g))
 for x in range(729):
  i,j,a,b=x//81,x//9%9,x%9//3-1,x%3-1;v=h[i][j]
  if a*a+b*b==v>0:g[i+a][j+b]=10-3*v
 return g
```

```python
verify_program(task_num, load_examples(task_num))
```

```python
test_code(task_num)
```

```python
# Select puzzle and display
task_num = 5

gcgc_display(task_num)
```

#### First shorten the code as the grids are fixed size squares of 21x21. However, due to efficient Zopfli compression, the final compression is the same.

```python
%%writefile task.py
def p(a):
 R=C=21;q,r,j,b=[C]*10,[R]*10,[-1]*10,[-1]*10
 for y in range(R):
  for x,k in enumerate(a[y]):q[k]=min(q[k],x);r[k]=min(r[k],y);j[k]=max(j[k],x);b[k]=max(b[k],y)
 for k in range(1,10):
  if j[k]-q[k]==b[k]-r[k]==2:X,Y=q[k],r[k];break
 O=[[0]*21 for _ in range(21)];Z=[(x,y)for y in range(3)for x in range(3)if a[Y+y][X+x]]
 for y in range(3):O[Y+y][X:X+3]=a[Y+y][X:X+3]
 for dx,dy in[(x,y)for x in(-4,0,4)for y in(-4,0,4)if x|y]:
  if k:=next((a[Y+dy+y][X+dx+x]for y in range(3)for x in range(3)if 0<=Y+dy+y<21 and 0<=X+dx+x<21 and a[Y+dy+y][X+dx+x]),0):
   cx,cy=X,Y
   while-3<cx<21 and-3<cy<21:
    cx+=dx;cy+=dy
    for x,y in Z:
     if 0<=cy+y<21 and 0<=cx+x<21:O[cy+y][cx+x]=k
 return O
```

```python
verify_program(task_num, load_examples(task_num))
```

```python
test_code(task_num)
```

#### Nonetheless, we can remove the redundant R,C assignments now to provide further improvement.

```python
%%writefile task.py
def p(a):
 q,r,j,b=[21]*10,[21]*10,[-1]*10,[-1]*10
 for y in range(21):
  for x,k in enumerate(a[y]):q[k]=min(q[k],x);r[k]=min(r[k],y);j[k]=max(j[k],x);b[k]=max(b[k],y)
 for k in range(1,10):
  if j[k]-q[k]==b[k]-r[k]==2:X,Y=q[k],r[k];break
 O=[[0]*21 for _ in a];Z=[(x,y)for y in range(3)for x in range(3)if a[Y+y][X+x]]
 for y in range(3):O[Y+y][X:X+3]=a[Y+y][X:X+3]
 for dx,dy in[(x,y)for x in(-4,0,4)for y in(-4,0,4)if x|y]:
  if k:=next((a[Y+dy+y][X+dx+x]for y in range(3)for x in range(3)if 0<=Y+dy+y<21 and 0<=X+dx+x<21 and a[Y+dy+y][X+dx+x]),0):
   cx,cy=X,Y
   while-3<cx<21 and-3<cy<21:
    cx+=dx;cy+=dy
    for x,y in Z:
     if 0<=cy+y<21 and 0<=cx+x<21:O[cy+y][cx+x]=k
 return O
```

```python
verify_program(task_num, load_examples(task_num))
```

```python
test_code(task_num)
```

## Generate submission.zip and final score of submission

```python
# Generate submission.zip file for submission
files = {}
total_save=0
with ZipFile("submission.zip", "w", zipfile.ZIP_DEFLATED) as zipf:
    for f in range(1,401):
        try:
            o=open('/kaggle/working/submission/task' + str(f).zfill(3) + '.py','rb').read().strip()
            zipped_src = zip_src(o)
            files[f] = min(len(o), len(zipped_src))
        except:
            continue
        #https://www.kaggle.com/code/cheeseexports/big-zippa
        improvement = len(o) - len(zipped_src)
        if improvement > 0:
            total_save += improvement
            open('/kaggle/working/submission/task' + str(f).zfill(3) + '.py','wb').write(zipped_src)
        else:
            open('/kaggle/working/submission/task' + str(f).zfill(3) + '.py','wb').write(o)
        zipf.write('/kaggle/working/submission/task' + str(f).zfill(3) + '.py')
print("Submission file generated.")

# Show final score of submission (this may take a few minutes)
score = 0
print('Calculating final score (may take a few minutes): ', end="")
for taskNo in files:
    try:
        solution = open('/kaggle/working/submission/task' + str(taskNo).zfill(3) + '.py','rb').read()
        if check(solution, taskNo, valall=True):
            s = max([0.1,2500-len(solution)])
            print('.', end="")
            score += s
        else: print(taskNo, ":L (failed)", end="")
    except: pass
print('\nScore:', score)
```

```python
# Done
```