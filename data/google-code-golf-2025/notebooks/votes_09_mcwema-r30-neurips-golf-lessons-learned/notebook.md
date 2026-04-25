# R30 NeurIPS Golf | Lessons Learned

- **Author:** mcwema
- **Votes:** 118
- **Ref:** mcwema/r30-neurips-golf-lessons-learned
- **URL:** https://www.kaggle.com/code/mcwema/r30-neurips-golf-lessons-learned
- **Last run:** 2025-11-04 01:42:07.537000

---

# Lessons learned from this competition

- The challenge is to find the ***shortest python script*** to achieve tasks objectives
- All tasks have been [solved](https://www.kaggle.com/competitions/google-code-golf-2025/discussion/594072) in prior competitions
- Scoring system got [challenged](https://www.kaggle.com/competitions/google-code-golf-2025/discussion/594027) earlier in the competition
- The [ensemble](https://www.kaggle.com/code/seshurajup/code-golf-ensemble-local-score-391-400-dsl) approach opened a new way 
- [Compression](https://www.kaggle.com/code/cheeseexports/big-zippa) techniques brought a new dynamic to sharing leaderboard
- In some instances, compress/decompress process may impact LB Score.


### This notebook is aimed to highlight how compression technique may influence the rating of LB score.


## Credit 
 - All shared notebooks, big thanks to authors for inspiring contributions.
 - Special Up Vote to those included in the input of this notebook

```python
import sys
sys.path.append("/kaggle/input/google-code-golf-2025/code_golf_utils")
from code_golf_utils import *
```

```python
num_tasks = 400
debug  =  False
if debug :
    num_tasks = 14
ignorerror =  False
explore = True
UseTop  = 2
```

```python
import copy
import importlib.util
import json
import re
import sys
import traceback
import numpy as np
import os
import shutil
from tqdm import tqdm 

def simple_verify_program(task_num, examples):
    task_name, task_path = "task_with_imports", "/kaggle/working/task.py"
    spec = importlib.util.spec_from_file_location(task_name, task_path)
    if spec is None:
        print("Error: Unable to import task.py.")
        return
    module = sys.modules[task_name] = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "p"):
        print("Error: Unable to locate function p() in task.py.")
        return
    program = getattr(module, "p")
    if not callable(program):
        print("Error: Function p() in task.py is not callable.")
        return

    def verify(example_subset):
        right, wrong, expected, error = 0, 0, None, ""
        for example in example_subset:
            example_copy = copy.deepcopy(example)
            try:
                result = program(example_copy["input"])
                result = json.dumps(result)
                result = result.replace("true", "1").replace("false", "0")
                unsafe_chars = re.compile(r"[^0-9,\[\]\s\.]")
                if unsafe_chars.search(result):
                    raise ValueError(f"Invalid output from user code: {result[:500]}")
                result = json.loads(result)
                user_output = np.array(result)
                label_output = np.array(example_copy["output"])
                if numpy.array_equal(user_output, label_output):
                    right += 1
                else:
                    expected = copy.deepcopy(example)
                    wrong += 1
            except:
                error = traceback.format_exc()
                wrong += 1
                #if error: print(f"Error: {error}")
        return right, wrong, expected
    
    arc_agi_right, arc_agi_wrong, arc_agi_expected = verify(examples["train"] + examples["test"])
    arc_gen_right, arc_gen_wrong, arc_gen_expected = verify(examples["arc-gen"])
    return arc_agi_right, arc_agi_wrong, arc_gen_right, arc_gen_wrong
```

## Extract zipped solutions

```python
import os
import zipfile
from io import BytesIO
import warnings
from itertools import combinations, permutations
from glob import glob
# Ignore SyntaxWarning globally
warnings.filterwarnings("ignore", category=SyntaxWarning)
pbsrc = []
# Paths - Notebooks with varios significant contributions
paths = [
    "/kaggle/input/r30-neurips-golf-lessons-learned",
    "/kaggle/input/google-code-golf-2025-top-78-trick-all-top-963178",
    "/kaggle/input/compilation-of-winning-solutions",
    "/kaggle/input/system-control-pannel",
    "/kaggle/input/golf-compress-all",
]

# Other notebooks
others = [

    "/kaggle/input/neuroips-4-of-some-lessons-learned",
    "/kaggle/input/r21-neuroips-fork-4-of-some-lessons-learned",
    "/kaggle/input/neurips-2025-community-baselines-improvements",
    "/kaggle/input/community-baselines-1000-point-911k",
    "/kaggle/input/gcgch-solution-15",
    "/kaggle/input/400-task-with-smart-solution-search-verification",
     "/kaggle/input/baselines",
    "/kaggle/input/pipeline-update-task115",
    "/kaggle/input/community-baselines-8-10",
    "/kaggle/input/pipeline-new-way-for-task31",
    "/kaggle/input/public-task-398-down-to-84-bytes",
    "/kaggle/input/pipeline-update-task7",
    "/kaggle/input/simplification-is-key",
    "/kaggle/input/neuroips-fork-4-of-some-lessons-learned",  
    "/kaggle/input/neurips-2025-community-baselines-17-points",
   "/kaggle/input/community-baselines-1000-point-91-05k",
   "/kaggle/input/task-101-similarshape-playground",
    "/kaggle/input/dead-code",
    "/kaggle/input/google-code-golf-full-400-solutions-verification",
    "/kaggle/input/public-task-33-down-to-89-bytes",
    "/kaggle/input/fun-shocking-1-code-2-tasks-3-top-score",
    "/kaggle/input/gcgc-playground",
    "/kaggle/input/manual-modification-golf",
    "/kaggle/input/pipeline-update-task-extension-912736",
    "/kaggle/input/google-code-golf-new-community-best",
   "/kaggle/input/community-baselines-9-10-913321",
   "/kaggle/input/google-code-golf-community-best",
    "/kaggle/input/task-101-similarshape-playground",
    "/kaggle/input/community-baselines-my-arc-10-pack-4-10",
    "/kaggle/input/neuripsr31",
    "/kaggle/input/k-on-first-round-1-60-task-collection",
    "/kaggle/input/source-code-400-with-vis-and-zip",
    "/kaggle/input/source-code-for-full-400-solutions",
    "/kaggle/input/400-task-with-smart-solution-search-verification",
    "/kaggle/input/python-minifier-applied",
    "/kaggle/input/gcgc-solutions-for-all-400-tasks",
    "/kaggle/input/oh-barnacles",
    "/kaggle/input/google-golf-code-4-solution",
    "/kaggle/input/source-code-400-with-vis-and-zip-v2",
    "/kaggle/input/neurips-2025",
    "/kaggle/input/google-code-golf-championship-101",
    "/kaggle/input/community-baselines-my-arc-10-pack-1-10",
    "/kaggle/input/community-baselines-my-arc-10-pack-2-10",
    "/kaggle/input/community-baselines-my-arc-10-pack-3-10",
    "/kaggle/input/community-baselines-my-arc-10-pack-4-10",
    "/kaggle/input/community-baselines-my-arc-10-pack-4-10",
    "/kaggle/input/community-baselines-my-arc-10-pack-5-10",
    "/kaggle/input/k/gothamjocker/gcgc-playground",
    "/kaggle/input/task-285-missinglegs-playground",
    "/kaggle/input/manual-modification-golf",
    "/kaggle/input/task-255-excavation-playground",
   "/kaggle/input/first-round-1-50-task-collection",
    "/kaggle/input/task-187-linedboxes-theoretical-minimum",
    "/kaggle/input/task-101-similarshape-playground",
        ]
# Extract code sub-folders
for idx, path in enumerate(paths):
    dest_folder = f"/kaggle/working/submission_{idx}"
    os.makedirs(dest_folder, exist_ok=True)
    
    zip_path = os.path.join(path, "submission.zip")
    sub_path = os.path.join(path, "submission")
    sub_path_ = os.path.join(path, "submission_")

    sub_path = sub_path if os.path.exists(sub_path) else sub_path_ if os.path.exists(sub_path_)  else path 

    if not os.path.exists(zip_path):
        paths.pop(paths.index(path))
        print(f"Zip not found: {zip_path} - Skipped")
        continue

    zip_size = size = os.path.getsize(zip_path)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(dest_folder)
    pbsrc.append(sub_path)
    nfiles = len(glob(f"{dest_folder}/*"))
    if nfiles == 1:
        print()
        print(path, glob(f"{path}/*"))
        print()

    print(f"Extracted {nfiles} size {zip_size} →→→ All files from {zip_path} → {dest_folder}")
```

## Find solution with the smallest-size

```python
spbsrc = [p if "submission"  in (p) else f"{p}/submission_" for p in pbsrc ]
```

```python
if explore:
    BSCORES = {}
    for b_src in spbsrc:
        # Process each task
        if os.path.exists(b_src):
            Improved = 0
            NImproved = 0
            sb_src = b_src.replace('kaggle/input/', '').replace('submission','')
            print("\n\n", 20*"* * ")
            print("Using ", spbsrc.index(b_src),  sb_src)
            best_subs  = []
            Score = 0
            # Process each task
            
            for task_num in range(1, num_tasks + 1):
                task_name = f"task{task_num:03d}.py"
                improve = 0
                
                smallest_idx = 0
                best_idxes  = []
                source_path = os.path.join(b_src, task_name)
                if not os.path.exists(source_path):
                    continue
                    
                source_size = os.path.getsize(source_path)
                smallest_size = os.path.getsize(source_path)
                smallest_file = source_path 

                # Check all submission.zip in paths
                for idx in range(len(paths)):
                    if paths[idx] == b_src.replace('/submission',''):
                        # Skip paths[idx] == b_src
                        continue
                        
                    folder = f"submission_{idx}"
                    task_path = os.path.join(folder, task_name)
                    if os.path.exists(task_path):
                        # Check the solution
                        size = os.path.getsize(task_path)
                        if smallest_size is None or size < smallest_size:
                            improve += 1
                            try:
                                shutil.copy(task_path, "task.py")
                                examples = load_examples(task_num)
                                agi_right, agi_wrong, agen_right, gen_wrong = simple_verify_program(task_num, examples)
                                if agi_wrong + gen_wrong == 0:
                                    smallest_size = size
                                    smallest_idx  = idx
                                    smallest_file = os.path.join(f"submission_{idx}", task_name)
                            except:
                                if ignorerror:
                                    smallest_size = size
                                    smallest_idx  = idx
                                    smallest_file = os.path.join(f"submission_{idx}", task_name)
                                    print(f"\nImprove {improve} Here @ {task_path} & {idx} {size} {smallest_size} but using IGNORE")
                                else:
                                    print(f"\nCan't get smallest_size for Improve {improve} Here @ {task_path} & {idx} {size} {smallest_size}. agi right {agi_right}, wrong  {agi_wrong}, agen right {agen_right}, wrong {gen_wrong} ")
                                    
                        if size == smallest_size:
                            best_idxes.append(idx)
                            
                if os.path.exists(smallest_file):
                    # Add smallest valid file to final zip
                    if smallest_size != None:
                        mark = "" 
                    else:
                        agi_right, agi_wrong, agen_right, gen_wrong = simple_verify_program(task_num, examples)
                        mark = f" >>> agi_wrong: {agi_wrong}, gen_wrong: {gen_wrong}"
                    smallest_paths = [paths[i] for i in  best_idxes]
                    if improve:
                        Timprove = source_size - smallest_size
                        if Timprove > 0:
                            NImproved += 1
                            Improved  +=  Timprove
                            if debug:
                                print("Task_id = {:3}: Improveed  {} times  by {:4} {:5} {:4} @ Notebooks. Best ({:4} from {:4} bytes) in  {:2} subs  {{}} ".format(task_num, improve, Timprove, Improved, NImproved,  smallest_size, source_size, len (best_idxes)), best_idxes)  
                            else:
                                print("Task {}: {} {} Best {} from {},  ".format(task_num, Timprove, Improved,  smallest_size, source_size), end = ' ')  
                    if smallest_size == None: 
                        Score += 0.001
                    else:
                        Score += max(1, 2500 - smallest_size)
                    best_subs.append(best_idxes)
                else:
                    print("Task_id = {}: No solution found".format(task_num))
            print(f"\n{sb_src:30}  {1+spbsrc.index(b_src):3d}/{len(spbsrc):3d}  Tasks Improved {NImproved}   Total Improvement  {Improved}  Total Score {Score:10d}" )
            BSCORES[sb_src] = Score, NImproved, Improved
        else:
            print("Skipping ", b_src)
```

```python
if explore:
    ScTuples = [(k,BSCORES[k]) for k in BSCORES]
    Sorted_ScTuples = sorted(ScTuples, key=lambda ScTuples: ScTuples[1][0], reverse=True)
    Dble_Sorted = sorted(Sorted_ScTuples, key=lambda Sorted_ScTuples: Sorted_ScTuples[1][1],  reverse=True)
    display(Sorted_ScTuples)
```

```python
b_src = os.path.join('/kaggle/input', Sorted_ScTuples[UseTop-1][0].strip('/'), 'submission') if explore else spbsrc[UseTop-1]
print(f"Top {UseTop} >->  b_src  {b_src}")
```

```python
from tqdm import tqdm

submission = "/kaggle/working/submission"
os.makedirs(submission, exist_ok=True)
best_subs  = []
Score = 0
# Process each task
print("\n Using source from ", b_src)
Improved = 0

for task_num in range(1, num_tasks + 1):
    task_name = f"task{task_num:03d}.py"
    improve = 0
    
    smallest_idx = 0
    best_idxes  = []

    #source_path = os.path.join("submission_0", task_name)
    source_path = os.path.join(b_src, task_name)
    source_size = os.path.getsize(source_path)
    smallest_size = os.path.getsize(source_path)
    smallest_file = source_path 

    # Check all submission.zip in paths
    for idx in range(len(paths)):
        
        folder = f"submission_{idx}"
        task_path = os.path.join(folder, task_name)

        if paths[idx] == b_src.replace('/submission',''):
            # Skip paths[idx] == b_src
            continue
            
        if os.path.exists(task_path):
            # Check the solution
            size = os.path.getsize(task_path)
            if smallest_size is None or size < smallest_size:
                improve += 1
                try:
                    shutil.copy(task_path, "task.py")
                    examples = load_examples(task_num)
                    
                    agi_right, agi_wrong, agen_right, gen_wrong = simple_verify_program(task_num, examples)
                    
                    if agi_wrong + gen_wrong == 0:
                        smallest_size = size
                        smallest_idx  = idx
                        smallest_file = os.path.join(f"submission_{idx}", task_name)
                except:
                    if ignorerror:
                        # Ignore errors and use small size file 
                        smallest_size = size
                        smallest_idx  = idx
                        smallest_file = os.path.join(f"submission_{idx}", task_name)
                        print(f"\nImprove {improve} Here @ {task_path} & {idx} {size} {smallest_size} but using IGNORE")
                    else:
                        # Don't use a small size file if errors
                        print(f"\nImprove {improve} Here @ {task_path} & {idx} {size} {smallest_size} but can't get smallest_size ")
                        
            if size == smallest_size:
                best_idxes.append(idx)

    
    if os.path.exists(smallest_file):
        # Add smallest valid file to final zip
        if smallest_size != None:
            mark = "" 
        else:
            agi_right, agi_wrong, agen_right, gen_wrong = simple_verify_program(task_num, examples)
            mark = f" >>> agi_wrong: {agi_wrong}, gen_wrong: {gen_wrong}"
        smallest_paths = [paths[i] for i in  best_idxes]
        if improve:
            Timprove = source_size - smallest_size
            if Timprove > 0:
                Improved +=  Timprove
                print("Task_id = {:3}: Improveed  {} times  by {:3} {:4} @ Notebooks. Best ({:4} from {:4} bytes) in  {} subs  {{}} ".format(task_num, improve, Timprove, Improved, smallest_size, source_size, len (smallest_paths)), smallest_paths)
        
        shutil.copy(smallest_file, os.path.join(submission, task_name))
        
        if smallest_size == None: 
            Score += 0.001
        else:
            Score += max(1, 2500 - smallest_size)
        best_subs.append(best_idxes)
    else:
        print("Task_id = {}: No solution found".format(task_num))

print("LB Score: ", Score)
```

## Selected  best score  solution for tasks sorted by number of notebook with the solution

```python
TasksSolver = {}
for j in range(1, len(paths)+1):
    TasksSolver[j] = []
    jc  = 0
    Subs = []
    Tasks = []
    for i in range(len(best_subs)):
        if len(best_subs[i]) == j:
            if jc == 0:
                Subs = best_subs[i]
                Tasks.append(i)
                jc+=1
            else:
                Tasks.append(i)
                
    if len(Tasks) > 0:
        TasksSolver[j].extend( (Subs, Tasks) )

        print (f"{j:3}, {len(Tasks):3}, {len(Subs):3},  ", Subs)
```

```python
if debug:
    display(best_subs)
```

## How many tasks each notebook solved

```python
for i in range(len(paths)):
    l = len([ sub for sub in best_subs if i in sub ])
    if l > 0:
        print(f" {i:2}   {l:4}    {paths[i]}")
```

## Create submission

```python
import zipfile

submission_zip = f"{submission}.zip"

with zipfile.ZipFile(submission_zip, "w", zipfile.ZIP_DEFLATED) as zipf:
    task_count = 0
    for task_num in range(1, 401):
        task_id = f"{task_num:03d}"
        src_path = f"{submission}/task{task_id}.py"
        
        if os.path.exists(src_path):
            zipf.write(src_path, arcname=f"task{task_id}.py")
            task_count += 1

print(f"Created submission zip with {task_count} tasks: {submission_zip}")

# Display zip file size
zip_size = os.path.getsize(submission_zip)
print(f"Submission zip size: {zip_size:,} bytes ({zip_size/1024:.1f} KB)")
```