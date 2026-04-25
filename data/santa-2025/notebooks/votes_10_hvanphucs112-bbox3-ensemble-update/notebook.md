# BBOX3 - Ensemble Update

- **Author:** Van-Phuc Huynh
- **Votes:** 186
- **Ref:** hvanphucs112/bbox3-ensemble-update
- **URL:** https://www.kaggle.com/code/hvanphucs112/bbox3-ensemble-update
- **Last run:** 2026-01-28 23:36:18.117000

---

```python
import subprocess
import shutil
import os
import time
import math
import pandas as pd
import numpy as np
from numba import njit
from decimal import Decimal, getcontext
from shapely import affinity
from shapely.geometry import Polygon
from shapely.strtree import STRtree
import random
from collections import defaultdict

# ============================================================
# CONFIGURATION
# ============================================================
MAX_TIME_SECONDS = 10 * 60
BBOX3_TIMEOUT = 150
SA_ITERATIONS = 125
GRADIENT_STEPS = 15
BASIN_HOP_PERTURBATION = 0.05

# --- Setup Paths ---
INPUT_SUB = '/kaggle/input/santa-submission/submission.csv'
INPUT_BIN = '/kaggle/input/santa-submission/bbox3'
WORKING_DIR = '/kaggle/working/'

print("📂 Setting up environment...")

if os.path.exists(INPUT_SUB):
    shutil.copy(INPUT_SUB, os.path.join(WORKING_DIR, 'submission.csv'))
    print(f"✅ Copied submission.csv")

if os.path.exists(INPUT_BIN):
    shutil.copy(INPUT_BIN, os.path.join(WORKING_DIR, 'bbox3'))
    os.chmod('./bbox3', 0o755)
    print(f"✅ Copied and set permissions for bbox3")

getcontext().prec = 25
scale_factor = Decimal("1e18")
```

# Ensemble

```python
%%writefile ensemble_submissions.py
#!/usr/bin/env python3
"""
Ensemble Optimizer
"""

import csv
import math
import glob
import os
from collections import defaultdict
from typing import Dict, List, Tuple

# Tree shape constants (from C++ code)
TX = [0, 0.125, 0.0625, 0.2, 0.1, 0.35, 0.075, 0.075, -0.075, -0.075, -0.35, -0.1, -0.2, -0.0625, -0.125]
TY = [0.8, 0.5, 0.5, 0.25, 0.25, 0, 0, -0.2, -0.2, 0, 0, 0.25, 0.25, 0.5, 0.5]

def get_polygon_bounds(cx: float, cy: float, deg: float) -> Tuple[float, float, float, float]:
    """Calculate bounding box of rotated tree polygon"""
    rad = deg * math.pi / 180.0
    s = math.sin(rad)
    c = math.cos(rad)
    
    x_coords = []
    y_coords = []
    
    for i in range(len(TX)):
        x = TX[i] * c - TY[i] * s + cx
        y = TX[i] * s + TY[i] * c + cy
        x_coords.append(x)
        y_coords.append(y)
    
    return min(x_coords), max(x_coords), min(y_coords), max(y_coords)

def calculate_score(trees: List[Tuple[int, float, float, float]]) -> Tuple[float, float, float, float]:
    """
    Calculate score for a configuration
    Returns: (score, side, width, height)
    """
    if not trees:
        return float('inf'), 0, 0, 0
    
    global_x_min = float('inf')
    global_x_max = float('-inf')
    global_y_min = float('inf')
    global_y_max = float('-inf')
    
    for idx, cx, cy, deg in trees:
        x_min, x_max, y_min, y_max = get_polygon_bounds(cx, cy, deg)
        global_x_min = min(global_x_min, x_min)
        global_x_max = max(global_x_max, x_max)
        global_y_min = min(global_y_min, y_min)
        global_y_max = max(global_y_max, y_max)
    
    width = global_x_max - global_x_min
    height = global_y_max - global_y_min
    side = max(width, height)
    score = side * side / len(trees)
    
    return score, side, width, height

def load_submission(filepath: str) -> Dict[int, List[Tuple[int, float, float, float]]]:
    """
    Load submission file
    Returns: dict mapping n -> list of (idx, x, y, deg)
    """
    configurations = defaultdict(list)
    
    try:
        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Parse id
                id_parts = row['id'].split('_')
                n = int(id_parts[0])
                idx = int(id_parts[1])
                
                # Parse coordinates (remove 's' prefix if present)
                x = float(row['x'].replace('s', ''))
                y = float(row['y'].replace('s', ''))
                deg = float(row['deg'].replace('s', ''))
                
                configurations[n].append((idx, x, y, deg))
        
        # Sort by index
        for n in configurations:
            configurations[n].sort(key=lambda t: t[0])
        
        return dict(configurations)
    
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return {}

def analyze_submission(filepath: str, configurations: Dict[int, List]) -> Dict[int, Tuple]:
    """
    Analyze a submission file
    Returns: dict mapping n -> (score, side, width, height)
    """
    results = {}
    
    for n, trees in configurations.items():
        if len(trees) != n:
            print(f"  WARNING: n={n} has {len(trees)} trees (expected {n})")
            continue
        
        score, side, width, height = calculate_score(trees)
        results[n] = (score, side, width, height)
    
    return results

def create_ensemble(submissions: Dict[str, Dict[int, List]]) -> Dict[int, Tuple[List, str, float]]:
    """
    Create ensemble by selecting best configuration for each n
    Returns: dict mapping n -> (best_trees, source_file, score)
    """
    ensemble = {}
    
    # Get all n values
    all_n = set()
    for configs in submissions.values():
        all_n.update(configs.keys())
    
    # For each n, find best configuration
    for n in sorted(all_n):
        best_score = float('inf')
        best_trees = None
        best_source = None
        
        for filepath, configs in submissions.items():
            if n not in configs:
                continue
            
            trees = configs[n]
            if len(trees) != n:
                continue
            
            score, side, width, height = calculate_score(trees)
            
            if score < best_score:
                best_score = score
                best_trees = trees
                best_source = filepath
        
        if best_trees:
            ensemble[n] = (best_trees, best_source, best_score)
    
    return ensemble

def save_ensemble(ensemble: Dict[int, Tuple], output_path: str):
    """Save ensemble submission"""
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'x', 'y', 'deg'])
        
        for n in sorted(ensemble.keys()):
            trees, _, _ = ensemble[n]
            for idx, x, y, deg in trees:
                row_id = f"{n:03d}_{idx}"
                writer.writerow([row_id, f's{x:.17f}', f's{y:.17f}', f's{deg:.17f}'])

def print_comparison(submissions: Dict[str, Dict[int, List]], ensemble: Dict[int, Tuple]):
    """Print detailed comparison"""
    print("\n" + "="*80)
    print("DETAILED COMPARISON BY N")
    print("="*80)
    
    all_n = sorted(set(
        n for configs in submissions.values() 
        for n in configs.keys()
    ))
    
    # Prepare data for each n
    for n in all_n:
        print(f"\n{'─'*80}")
        print(f"n = {n}")
        print(f"{'─'*80}")
        
        # Collect scores from all submissions
        scores_data = []
        for filepath, configs in submissions.items():
            if n in configs and len(configs[n]) == n:
                score, side, width, height = calculate_score(configs[n])
                basename = os.path.basename(filepath)
                scores_data.append((basename, score, side))
        
        # Sort by score
        scores_data.sort(key=lambda x: x[1])
        
        # Print table
        print(f"{'Source':<30} {'Score':<20} {'Side':<20} {'Status'}")
        print(f"{'-'*30} {'-'*20} {'-'*20} {'-'*10}")
        
        for i, (source, score, side) in enumerate(scores_data):
            status = "✅ BEST" if i == 0 else ""
            print(f"{source:<30} {score:<20.15f} {side:<20.15f} {status}")
        
        # Show ensemble choice
        if n in ensemble:
            _, best_source, best_score = ensemble[n]
            print(f"\n→ Ensemble choice: {os.path.basename(best_source)} (score: {best_score:.15f})")
        
        # Calculate improvement range
        if len(scores_data) > 1:
            worst_score = scores_data[-1][1]
            best_score = scores_data[0][1]
            improvement = (worst_score - best_score) / worst_score * 100
            print(f"→ Improvement range: {improvement:.4f}%")

def print_summary(submissions: Dict[str, Dict[int, List]], ensemble: Dict[int, Tuple]):
    """Print summary statistics"""
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    # Per-file statistics
    print("\nPer-file statistics:")
    print(f"{'File':<30} {'Total n':<10} {'Avg Score':<20} {'Best Count'}")
    print(f"{'-'*30} {'-'*10} {'-'*20} {'-'*10}")
    
    for filepath, configs in sorted(submissions.items()):
        basename = os.path.basename(filepath)
        
        # Calculate average score
        total_score = 0
        count = 0
        for n, trees in configs.items():
            if len(trees) == n:
                score, _, _, _ = calculate_score(trees)
                total_score += score
                count += 1
        
        avg_score = total_score / count if count > 0 else 0
        
        # Count how many times this file was chosen as best
        best_count = sum(1 for _, source, _ in ensemble.values() 
                        if source == filepath)
        
        print(f"{basename:<30} {count:<10} {avg_score:<20.10f} {best_count}")
    
    # Ensemble statistics
    print("\n" + "-"*80)
    print("Ensemble statistics:")
    
    total_score = sum(score for _, _, score in ensemble.values())
    avg_score = total_score / len(ensemble) if ensemble else 0

   
    print(f"  Total n values: {len(ensemble)}")
    print(f"  Total score: {total_score}")
    print(f"  Average score:  {avg_score:.10f}")
    
    # Count improvements
    print("\nSource distribution in ensemble:")
    source_counts = defaultdict(int)
    for _, source, _ in ensemble.values():
        basename = os.path.basename(source)
        source_counts[basename] += 1
    
    for source, count in sorted(source_counts.items(), key=lambda x: -x[1]):
        pct = count / len(ensemble) * 100
        print(f"  {source:<30} {count:>3} / {len(ensemble)} ({pct:>5.1f}%)")

def print_highlights(ensemble: Dict[int, Tuple]):
    """Print highlights - best and worst scores"""
    print("\n" + "="*80)
    print("HIGHLIGHTS")
    print("="*80)
    
    # Sort by score
    sorted_n = sorted(ensemble.items(), key=lambda x: x[1][2])
    
    print("\n🏆 TOP 10 BEST SCORES:")
    print(f"{'n':<5} {'Score':<20} {'Source'}")
    print(f"{'-'*5} {'-'*20} {'-'*40}")
    for i, (n, (_, source, score)) in enumerate(sorted_n[:10]):
        basename = os.path.basename(source)
        print(f"{n:<5} {score:<20.15f} {basename}")
    
    print("\n⚠️  TOP 10 WORST SCORES:")
    print(f"{'n':<5} {'Score':<20} {'Source'}")
    print(f"{'-'*5} {'-'*20} {'-'*40}")
    for i, (n, (_, source, score)) in enumerate(sorted_n[-10:]):
        basename = os.path.basename(source)
        print(f"{n:<5} {score:<20.15f} {basename}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Ensemble multiple submissions')
    parser.add_argument('-d', '--dir', default='submissions', 
                       help='Directory containing submission files')
    parser.add_argument('-o', '--output', default='submission_ensemble.csv',
                       help='Output ensemble file')
    parser.add_argument('--verbose', action='store_true',
                       help='Show detailed comparison')
    
    args = parser.parse_args()    
    # Find all CSV files
    csv_files = glob.glob(os.path.join(args.dir, '*.csv'))
    
    if not csv_files:
        print(f"❌ No CSV files found in {args.dir}")
        return
    
    print(f"📁 Found {len(csv_files)} submission files:")
    for f in csv_files:
        size = os.path.getsize(f) / 1024
        print(f"   - {os.path.basename(f):<30} ({size:>8.1f} KB)")
    
    # Load all submissions
    print(f"\n📊 Loading submissions...")
    submissions = {}
    for filepath in csv_files:
        basename = os.path.basename(filepath)
        print(f"   Loading {basename}...", end=' ')
        configs = load_submission(filepath)
        if configs:
            submissions[filepath] = configs
            print(f"✅ ({len(configs)} groups)")
        else:
            print("❌ Failed")
    
    if not submissions:
        print("\n❌ No valid submissions loaded")
        return
    
    print(f"\n✅ Loaded {len(submissions)} submissions successfully")
    
    # Create ensemble
    print(f"\n🔧 Creating ensemble (selecting best for each n)...")
    ensemble = create_ensemble(submissions)
    
    print(f"✅ Ensemble created with {len(ensemble)} groups")
    
    # Save ensemble
    print(f"\n💾 Saving to {args.output}...")
    save_ensemble(ensemble, args.output)
    print(f"✅ Saved!")
    
    # Print statistics
    print_summary(submissions, ensemble)
    print_highlights(ensemble)
    
    if args.verbose:
        print_comparison(submissions, ensemble)
    else:
        print("\n💡 Use --verbose flag to see detailed comparison for each n")
    
    # Final summary
    print("\n" + "="*80)
    print("✅ ENSEMBLE COMPLETE!")
    print("="*80)
    print(f"\n📄 Output: {args.output}")
    print(f"📊 Total groups: {len(ensemble)}")
    
    # Calculate overall improvement
    total_improvement = 0
    count = 0
    for n in ensemble.keys():
        scores = []
        for filepath, configs in submissions.items():
            if n in configs and len(configs[n]) == n:
                score, _, _, _ = calculate_score(configs[n])
                scores.append(score)
        
        if len(scores) > 1:
            best = min(scores)
            worst = max(scores)
            if worst > 0:
                improvement = (worst - best) / worst * 100
                total_improvement += improvement
                count += 1
    
    if count > 0:
        avg_improvement = total_improvement / count
        print(f"📈 Average improvement per group: {avg_improvement:.4f}%")
    
    print("\n🎯 Next steps:")
    print(f"Review the ensemble: {args.output}")
    print()

if __name__ == '__main__':
    main()
```

```python
import glob
import shutil


temp_dir = f"temp_merge"
if os.path.exists(temp_dir):
    os.system("rm -rf " + temp_dir)
os.makedirs(temp_dir, exist_ok=True)

all_files = glob.glob(f"/kaggle/input/*/*.csv")
for i, file in enumerate(all_files):
    new_file = os.path.join(temp_dir, f"submission_{i+1}.csv")
    shutil.copy(file, new_file)
    print(f"Copied {i+1} files")

print(f"Number of files: {len(all_files)}")

!python3 ./ensemble_submissions.py -d {temp_dir} -o /kaggle/working/submission_ensemble.csv
!cp /kaggle/working/submission_ensemble.csv   /kaggle/working/submission.csv
!rm -rf temp_merge
```

# Optimized

```python
# ============================================================
# CORE FUNCTIONS - Numba optimized
# ============================================================

@njit(cache=True)
def make_polygon_template():
    tw=0.15; th=0.2; bw=0.7; mw=0.4; ow=0.25
    tip=0.8; t1=0.5; t2=0.25; base=0.0; tbot=-th
    x = np.array([0,ow/2,ow/4,mw/2,mw/4,bw/2,tw/2,tw/2,-tw/2,-tw/2,-bw/2,-mw/4,-mw/2,-ow/4,-ow/2], np.float64)
    y = np.array([tip,t1,t1,t2,t2,base,base,tbot,tbot,base,base,t2,t2,t1,t1], np.float64)
    return x, y

@njit(cache=True)
def score_group_fast(xs, ys, degs, tx, ty):
    n = xs.size
    V = tx.size
    mnx = 1e300; mny = 1e300; mxx = -1e300; mxy = -1e300
    for i in range(n):
        r = degs[i] * math.pi / 180.0
        c = math.cos(r); s = math.sin(r)
        xi = xs[i]; yi = ys[i]
        for j in range(V):
            X = c*tx[j] - s*ty[j] + xi
            Y = s*tx[j] + c*ty[j] + yi
            if X < mnx: mnx = X
            if X > mxx: mxx = X
            if Y < mny: mny = Y
            if Y > mxy: mxy = Y
    side = max(mxx - mnx, mxy - mny)
    return side * side / n

@njit(cache=True)
def get_bounding_box(xs, ys, degs, tx, ty):
    n = xs.size
    V = tx.size
    mnx = 1e300; mny = 1e300; mxx = -1e300; mxy = -1e300
    for i in range(n):
        r = degs[i] * math.pi / 180.0
        c = math.cos(r); s = math.sin(r)
        xi = xs[i]; yi = ys[i]
        for j in range(V):
            X = c*tx[j] - s*ty[j] + xi
            Y = s*tx[j] + c*ty[j] + yi
            if X < mnx: mnx = X
            if X > mxx: mxx = X
            if Y < mny: mny = Y
            if Y > mxy: mxy = Y
    return mnx, mny, mxx, mxy

@njit(cache=True)
def find_boundary_trees(xs, ys, degs, tx, ty):
    n = xs.size
    V = tx.size
    mnx = 1e300; mny = 1e300; mxx = -1e300; mxy = -1e300
    min_x_tree = 0; min_y_tree = 0; max_x_tree = 0; max_y_tree = 0
    
    for i in range(n):
        r = degs[i] * math.pi / 180.0
        c = math.cos(r); s = math.sin(r)
        xi = xs[i]; yi = ys[i]
        for j in range(V):
            X = c*tx[j] - s*ty[j] + xi
            Y = s*tx[j] + c*ty[j] + yi
            if X < mnx: mnx = X; min_x_tree = i
            if X > mxx: mxx = X; max_x_tree = i
            if Y < mny: mny = Y; min_y_tree = i
            if Y > mxy: mxy = Y; max_y_tree = i
    
    return min_x_tree, min_y_tree, max_x_tree, max_y_tree
```

```python
# ============================================================
# SHAPELY-BASED OVERLAP CHECKING (CRITICAL!)
# ============================================================

class ChristmasTree:
    def __init__(self, center_x="0", center_y="0", angle="0"):
        self.center_x = Decimal(str(center_x))
        self.center_y = Decimal(str(center_y))
        self.angle = Decimal(str(angle))
        
        trunk_w = Decimal("0.15")
        trunk_h = Decimal("0.2")
        base_w = Decimal("0.7")
        mid_w = Decimal("0.4")
        top_w = Decimal("0.25")
        tip_y = Decimal("0.8")
        tier_1_y = Decimal("0.5")
        tier_2_y = Decimal("0.25")
        base_y = Decimal("0.0")
        trunk_bottom_y = -trunk_h

        initial_polygon = Polygon([
            (Decimal("0.0") * scale_factor, tip_y * scale_factor),
            (top_w / Decimal("2") * scale_factor, tier_1_y * scale_factor),
            (top_w / Decimal("4") * scale_factor, tier_1_y * scale_factor),
            (mid_w / Decimal("2") * scale_factor, tier_2_y * scale_factor),
            (mid_w / Decimal("4") * scale_factor, tier_2_y * scale_factor),
            (base_w / Decimal("2") * scale_factor, base_y * scale_factor),
            (trunk_w / Decimal("2") * scale_factor, base_y * scale_factor),
            (trunk_w / Decimal("2") * scale_factor, trunk_bottom_y * scale_factor),
            (-(trunk_w / Decimal("2")) * scale_factor, trunk_bottom_y * scale_factor),
            (-(trunk_w / Decimal("2")) * scale_factor, base_y * scale_factor),
            (-(base_w / Decimal("2")) * scale_factor, base_y * scale_factor),
            (-(mid_w / Decimal("4")) * scale_factor, tier_2_y * scale_factor),
            (-(mid_w / Decimal("2")) * scale_factor, tier_2_y * scale_factor),
            (-(top_w / Decimal("4")) * scale_factor, tier_1_y * scale_factor),
            (-(top_w / Decimal("2")) * scale_factor, tier_1_y * scale_factor),
        ])
        rotated = affinity.rotate(initial_polygon, float(self.angle), origin=(0, 0))
        self.polygon = affinity.translate(
            rotated, xoff=float(self.center_x * scale_factor), yoff=float(self.center_y * scale_factor)
        )

def create_trees_from_arrays(xs, ys, degs):
    """Create ChristmasTree objects from numpy arrays."""
    return [ChristmasTree(str(xs[i]), str(ys[i]), str(degs[i])) for i in range(len(xs))]

def has_overlap_arrays(xs, ys, degs):
    """Check if configuration has overlaps using arrays directly."""
    if len(xs) <= 1:
        return False
    trees = create_trees_from_arrays(xs, ys, degs)
    return has_overlap(trees)

def has_overlap(trees):
    """Check if any trees overlap."""
    if len(trees) <= 1:
        return False
    polygons = [t.polygon for t in trees]
    tree_index = STRtree(polygons)
    for i, poly in enumerate(polygons):
        for idx in tree_index.query(poly):
            if idx != i and poly.intersects(polygons[idx]) and not poly.touches(polygons[idx]):
                return True
    return False

def load_configuration_from_df(n, df):
    group_data = df[df["id"].str.startswith(f"{n:03d}_")]
    trees = []
    for _, row in group_data.iterrows():
        trees.append(ChristmasTree(row["x"][1:], row["y"][1:], row["deg"][1:]))
    return trees

def get_score(trees, n=None):
    xys = np.concatenate([np.asarray(t.polygon.exterior.xy).T / 1e18 for t in trees])
    min_x, min_y = xys.min(axis=0)
    max_x, max_y = xys.max(axis=0)
    score = max(max_x - min_x, max_y - min_y) ** 2
    return score / n if n else score

def eval_df_sub(df, verbose=False):
    failed = []
    total_score = 0.0
    scores = {}
    for n in range(1, 201):
        trees = load_configuration_from_df(n, df)
        score = get_score(trees, n)
        scores[n] = score
        total_score += score
        if verbose:
            print(f"{n:3}  {score:.6f}")
        if has_overlap(trees):
            failed.append(n)
    
    if not failed:
        print("✅ No overlaps")
    else:
        print(f"❌ Overlaps in: {failed}")
    print(f"📊 Score: {total_score:.12f}")
    return total_score, scores
```

```python
# ============================================================
# IMPROVEMENT 1: Simulated Annealing (WITH OVERLAP CHECK)
# ============================================================

def simulated_annealing_config(df, config_n, max_iterations=200, 
                                initial_temp=0.001, cooling_rate=0.995):
    """
    Simulated annealing with overlap validation.
    """
    tx, ty = make_polygon_template()
    
    group_mask = df["id"].str.startswith(f"{config_n:03d}_")
    group_data = df[group_mask].copy()
    
    xs = np.array([float(v[1:]) for v in group_data["x"]])
    ys = np.array([float(v[1:]) for v in group_data["y"]])
    degs = np.array([float(v[1:]) for v in group_data["deg"]])
    
    n_trees = len(xs)
    if n_trees <= 1:
        return False, 0
    
    original_score = score_group_fast(xs, ys, degs, tx, ty)
    current_score = original_score
    best_score = original_score
    best_xs, best_ys, best_degs = xs.copy(), ys.copy(), degs.copy()
    
    temperature = initial_temp
    step_xy = 0.002 / np.sqrt(n_trees)
    step_rot = 0.5 / np.sqrt(n_trees)
    
    for iteration in range(max_iterations):
        tree_idx = random.randint(0, n_trees - 1)
        
        new_xs = xs.copy()
        new_ys = ys.copy()
        new_degs = degs.copy()
        
        move = random.choice(['translate', 'rotate', 'both'])
        
        if move in ['translate', 'both']:
            new_xs[tree_idx] += random.gauss(0, step_xy)
            new_ys[tree_idx] += random.gauss(0, step_xy)
        
        if move in ['rotate', 'both']:
            new_degs[tree_idx] += random.gauss(0, step_rot)
        
        new_score = score_group_fast(new_xs, new_ys, new_degs, tx, ty)
        delta = new_score - current_score
        
        # Only check overlap if we might accept the move
        if delta < 0 or random.random() < math.exp(-delta / temperature):
            # CRITICAL: Check for overlaps before accepting!
            if not has_overlap_arrays(new_xs, new_ys, new_degs):
                xs, ys, degs = new_xs, new_ys, new_degs
                current_score = new_score
                
                if current_score < best_score:
                    best_score = current_score
                    best_xs, best_ys, best_degs = xs.copy(), ys.copy(), degs.copy()
        
        temperature *= cooling_rate
    
    improved = best_score < original_score - 1e-15
    
    if improved:
        # Final validation before writing
        if not has_overlap_arrays(best_xs, best_ys, best_degs):
            indices = df[group_mask].index
            for i, idx in enumerate(indices):
                df.at[idx, 'x'] = f"s{best_xs[i]}"
                df.at[idx, 'y'] = f"s{best_ys[i]}"
                df.at[idx, 'deg'] = f"s{best_degs[i]}"
        else:
            improved = False
    
    return improved, original_score - best_score if improved else 0
```

```python
# ============================================================
# IMPROVEMENT 2: Swap-based moves (WITH OVERLAP CHECK)
# ============================================================

def try_swap_trees(df, config_n, max_swaps=50):
    """
    Try swapping positions with overlap validation.
    """
    tx, ty = make_polygon_template()
    
    group_mask = df["id"].str.startswith(f"{config_n:03d}_")
    group_data = df[group_mask].copy()
    
    xs = np.array([float(v[1:]) for v in group_data["x"]])
    ys = np.array([float(v[1:]) for v in group_data["y"]])
    degs = np.array([float(v[1:]) for v in group_data["deg"]])
    
    n_trees = len(xs)
    if n_trees <= 2:
        return False, 0
    
    original_score = score_group_fast(xs, ys, degs, tx, ty)
    best_score = original_score
    best_xs, best_ys = xs.copy(), ys.copy()
    
    for _ in range(max_swaps):
        i, j = random.sample(range(n_trees), 2)
        
        new_xs = xs.copy()
        new_ys = ys.copy()
        new_xs[i], new_xs[j] = new_xs[j], new_xs[i]
        new_ys[i], new_ys[j] = new_ys[j], new_ys[i]
        
        new_score = score_group_fast(new_xs, new_ys, degs, tx, ty)
        
        if new_score < best_score:
            # CRITICAL: Check overlaps!
            if not has_overlap_arrays(new_xs, new_ys, degs):
                best_score = new_score
                best_xs, best_ys = new_xs.copy(), new_ys.copy()
                xs, ys = new_xs, new_ys
    
    improved = best_score < original_score - 1e-15
    if improved:
        if not has_overlap_arrays(best_xs, best_ys, degs):
            indices = df[group_mask].index
            for i, idx in enumerate(indices):
                df.at[idx, 'x'] = f"s{best_xs[i]}"
                df.at[idx, 'y'] = f"s{best_ys[i]}"
        else:
            improved = False
    
    return improved, original_score - best_score if improved else 0
```

```python
# ============================================================
# IMPROVEMENT 3: Boundary tree optimization (WITH OVERLAP CHECK)
# ============================================================

def optimize_boundary_trees(df, config_n, iterations=100):
    """
    Focus on boundary trees with overlap validation.
    """
    tx, ty = make_polygon_template()
    
    group_mask = df["id"].str.startswith(f"{config_n:03d}_")
    group_data = df[group_mask].copy()
    
    xs = np.array([float(v[1:]) for v in group_data["x"]])
    ys = np.array([float(v[1:]) for v in group_data["y"]])
    degs = np.array([float(v[1:]) for v in group_data["deg"]])
    
    n_trees = len(xs)
    if n_trees <= 1:
        return False, 0
    
    original_score = score_group_fast(xs, ys, degs, tx, ty)
    best_score = original_score
    best_xs, best_ys, best_degs = xs.copy(), ys.copy(), degs.copy()
    
    step = 0.0005
    rot_step = 0.2
    
    for _ in range(iterations):
        boundary_trees = set(find_boundary_trees(xs, ys, degs, tx, ty))
        
        for tree_idx in boundary_trees:
            mnx, mny, mxx, mxy = get_bounding_box(xs, ys, degs, tx, ty)
            cx = (mnx + mxx) / 2
            cy = (mny + mxy) / 2
            
            dx = cx - xs[tree_idx]
            dy = cy - ys[tree_idx]
            norm = np.sqrt(dx*dx + dy*dy)
            if norm > 1e-10:
                dx, dy = dx/norm * step, dy/norm * step
            
            # Try translation
            new_xs = xs.copy()
            new_ys = ys.copy()
            new_xs[tree_idx] += dx
            new_ys[tree_idx] += dy
            
            new_score = score_group_fast(new_xs, new_ys, degs, tx, ty)
            if new_score < best_score:
                # CRITICAL: Check overlaps!
                if not has_overlap_arrays(new_xs, new_ys, degs):
                    best_score = new_score
                    xs, ys = new_xs, new_ys
                    best_xs, best_ys = xs.copy(), ys.copy()
            
            # Try rotation
            for drot in [-rot_step, rot_step]:
                new_degs = degs.copy()
                new_degs[tree_idx] += drot
                new_score = score_group_fast(xs, ys, new_degs, tx, ty)
                if new_score < best_score:
                    # CRITICAL: Check overlaps!
                    if not has_overlap_arrays(xs, ys, new_degs):
                        best_score = new_score
                        degs = new_degs
                        best_degs = degs.copy()
    
    improved = best_score < original_score - 1e-15
    if improved:
        if not has_overlap_arrays(best_xs, best_ys, best_degs):
            indices = df[group_mask].index
            for i, idx in enumerate(indices):
                df.at[idx, 'x'] = f"s{best_xs[i]}"
                df.at[idx, 'y'] = f"s{best_ys[i]}"
                df.at[idx, 'deg'] = f"s{best_degs[i]}"
        else:
            improved = False
    
    return improved, original_score - best_score if improved else 0
```

```python
# ============================================================
# IMPROVEMENT 4: Gradient descent (WITH OVERLAP CHECK)
# ============================================================

def gradient_descent_config(df, config_n, steps=30, learning_rate=0.0001):
    """
    Gradient descent with overlap validation.
    """
    tx, ty = make_polygon_template()
    
    group_mask = df["id"].str.startswith(f"{config_n:03d}_")
    group_data = df[group_mask].copy()
    
    xs = np.array([float(v[1:]) for v in group_data["x"]])
    ys = np.array([float(v[1:]) for v in group_data["y"]])
    degs = np.array([float(v[1:]) for v in group_data["deg"]])
    
    n_trees = len(xs)
    if n_trees <= 1:
        return False, 0
    
    original_score = score_group_fast(xs, ys, degs, tx, ty)
    best_score = original_score
    best_xs, best_ys, best_degs = xs.copy(), ys.copy(), degs.copy()
    
    eps = 1e-7
    
    for step in range(steps):
        grad_x = np.zeros(n_trees)
        grad_y = np.zeros(n_trees)
        
        for i in range(n_trees):
            xs_plus = xs.copy(); xs_plus[i] += eps
            xs_minus = xs.copy(); xs_minus[i] -= eps
            grad_x[i] = (score_group_fast(xs_plus, ys, degs, tx, ty) - 
                        score_group_fast(xs_minus, ys, degs, tx, ty)) / (2 * eps)
            
            ys_plus = ys.copy(); ys_plus[i] += eps
            ys_minus = ys.copy(); ys_minus[i] -= eps
            grad_y[i] = (score_group_fast(xs, ys_plus, degs, tx, ty) - 
                        score_group_fast(xs, ys_minus, degs, tx, ty)) / (2 * eps)
        
        new_xs = xs - learning_rate * grad_x
        new_ys = ys - learning_rate * grad_y
        
        new_score = score_group_fast(new_xs, new_ys, degs, tx, ty)
        
        # CRITICAL: Only accept if no overlaps
        if new_score < best_score and not has_overlap_arrays(new_xs, new_ys, degs):
            xs, ys = new_xs, new_ys
            best_score = new_score
            best_xs, best_ys, best_degs = xs.copy(), ys.copy(), degs.copy()
    
    improved = best_score < original_score - 1e-15
    if improved:
        if not has_overlap_arrays(best_xs, best_ys, best_degs):
            indices = df[group_mask].index
            for i, idx in enumerate(indices):
                df.at[idx, 'x'] = f"s{best_xs[i]}"
                df.at[idx, 'y'] = f"s{best_ys[i]}"
                df.at[idx, 'deg'] = f"s{best_degs[i]}"
        else:
            improved = False
    
    return improved, original_score - best_score if improved else 0
```

```python
# ============================================================
# IMPROVEMENT 5: Rotation grid search (WITH OVERLAP CHECK)
# ============================================================

def rotation_grid_search(df, config_n, angle_step=15):
    """
    Try discrete rotation angles with overlap validation.
    """
    tx, ty = make_polygon_template()
    
    group_mask = df["id"].str.startswith(f"{config_n:03d}_")
    group_data = df[group_mask].copy()
    
    xs = np.array([float(v[1:]) for v in group_data["x"]])
    ys = np.array([float(v[1:]) for v in group_data["y"]])
    degs = np.array([float(v[1:]) for v in group_data["deg"]])
    
    n_trees = len(xs)
    if n_trees <= 1:
        return False, 0
    
    original_score = score_group_fast(xs, ys, degs, tx, ty)
    best_score = original_score
    best_degs = degs.copy()
    
    boundary_trees = set(find_boundary_trees(xs, ys, degs, tx, ty))
    angles_to_try = np.arange(-180, 180, angle_step)
    
    for tree_idx in boundary_trees:
        current_best_angle = degs[tree_idx]
        
        for angle in angles_to_try:
            test_degs = degs.copy()
            test_degs[tree_idx] = angle
            
            score = score_group_fast(xs, ys, test_degs, tx, ty)
            if score < best_score:
                # CRITICAL: Check overlaps!
                if not has_overlap_arrays(xs, ys, test_degs):
                    best_score = score
                    current_best_angle = angle
        
        degs[tree_idx] = current_best_angle
        best_degs = degs.copy()
    
    improved = best_score < original_score - 1e-15
    if improved:
        if not has_overlap_arrays(xs, ys, best_degs):
            indices = df[group_mask].index
            for i, idx in enumerate(indices):
                df.at[idx, 'deg'] = f"s{best_degs[i]}"
        else:
            improved = False
    
    return improved, original_score - best_score if improved else 0
```

```python
# ============================================================
# IMPROVEMENT 6: Basin Hopping (WITH OVERLAP CHECK)
# ============================================================

def basin_hopping_config(df, config_n, hops=10, local_steps=50):
    """
    Basin hopping with overlap validation.
    """
    tx, ty = make_polygon_template()
    
    group_mask = df["id"].str.startswith(f"{config_n:03d}_")
    group_data = df[group_mask].copy()
    
    xs = np.array([float(v[1:]) for v in group_data["x"]])
    ys = np.array([float(v[1:]) for v in group_data["y"]])
    degs = np.array([float(v[1:]) for v in group_data["deg"]])
    
    n_trees = len(xs)
    if n_trees <= 1:
        return False, 0
    
    original_score = score_group_fast(xs, ys, degs, tx, ty)
    best_score = original_score
    best_xs, best_ys, best_degs = xs.copy(), ys.copy(), degs.copy()
    
    perturbation_size = BASIN_HOP_PERTURBATION / np.sqrt(n_trees)
    
    for hop in range(hops):
        # Large perturbation
        perturbed_xs = xs + np.random.uniform(-perturbation_size, perturbation_size, n_trees)
        perturbed_ys = ys + np.random.uniform(-perturbation_size, perturbation_size, n_trees)
        perturbed_degs = degs + np.random.uniform(-10, 10, n_trees)
        
        # Skip if perturbation causes overlap
        if has_overlap_arrays(perturbed_xs, perturbed_ys, perturbed_degs):
            continue
        
        local_xs, local_ys, local_degs = perturbed_xs.copy(), perturbed_ys.copy(), perturbed_degs.copy()
        local_score = score_group_fast(local_xs, local_ys, local_degs, tx, ty)
        
        step = 0.001 / np.sqrt(n_trees)
        
        # Local optimization with overlap checking
        for _ in range(local_steps):
            tree_idx = random.randint(0, n_trees - 1)
            
            for dx, dy in [(step, 0), (-step, 0), (0, step), (0, -step)]:
                test_xs = local_xs.copy()
                test_ys = local_ys.copy()
                test_xs[tree_idx] += dx
                test_ys[tree_idx] += dy
                
                test_score = score_group_fast(test_xs, test_ys, local_degs, tx, ty)
                if test_score < local_score:
                    # Check overlaps only for improvements
                    if not has_overlap_arrays(test_xs, test_ys, local_degs):
                        local_xs, local_ys = test_xs, test_ys
                        local_score = test_score
        
        if local_score < best_score:
            # Final validation
            if not has_overlap_arrays(local_xs, local_ys, local_degs):
                best_score = local_score
                best_xs, best_ys, best_degs = local_xs.copy(), local_ys.copy(), local_degs.copy()
                xs, ys, degs = local_xs, local_ys, local_degs
    
    improved = best_score < original_score - 1e-15
    if improved:
        if not has_overlap_arrays(best_xs, best_ys, best_degs):
            indices = df[group_mask].index
            for i, idx in enumerate(indices):
                df.at[idx, 'x'] = f"s{best_xs[i]}"
                df.at[idx, 'y'] = f"s{best_ys[i]}"
                df.at[idx, 'deg'] = f"s{best_degs[i]}"
        else:
            improved = False
    
    return improved, original_score - best_score if improved else 0
```

```python
# ============================================================
# Adaptive Parameter Selector
# ============================================================

class AdaptiveParameterSelector:
    def __init__(self):
        self.successes = defaultdict(int)
        self.attempts = defaultdict(int)
        self.improvement_sum = defaultdict(float)
        self.n_range = (30, 400)
        self.r_range = (10, 50)
        self.good_params = [
            (72, 34), (100, 25), (50, 30), (150, 20), (80, 35),
            (60, 40), (120, 28), (90, 32), (200, 22), (40, 38),
            (180, 18), (75, 36), (110, 26), (65, 33), (140, 24),
            (85, 30), (95, 28), (55, 35), (130, 22), (160, 20)
        ]
    
    def get_params(self, exploration_rate=0.25):
        if random.random() < exploration_rate or not self.successes:
            if random.random() < 0.6 and self.good_params:
                return random.choice(self.good_params)
            return (random.randint(*self.n_range), random.randint(*self.r_range))
        
        weights, params = [], []
        for (n, r), successes in self.successes.items():
            attempts = self.attempts[(n, r)]
            if attempts > 0:
                rate = successes / attempts
                improvement = self.improvement_sum[(n, r)] / max(attempts, 1)
                weight = (rate + 0.1) * (1 + improvement * 1e8)
                weights.append(weight)
                params.append((n, r))
        
        if weights:
            total = sum(weights)
            idx = random.choices(range(len(params)), [w/total for w in weights])[0]
            return params[idx]
        return self.get_params(exploration_rate=1.0)
    
    def record_result(self, n, r, improved, improvement=0):
        self.attempts[(n, r)] += 1
        if improved:
            self.successes[(n, r)] += 1
            self.improvement_sum[(n, r)] += improvement
            if (n, r) not in self.good_params:
                self.good_params.append((n, r))
```

```python
# ============================================================
# MAIN OPTIMIZATION LOOP - V2 FIXED
# ============================================================

def main():
    start_time = time.time()
    
    df = pd.read_csv("submission.csv")
    initial_score, initial_scores = eval_df_sub(df, False)
    best_score = initial_score
    best_df = df.copy()
    
    param_selector = AdaptiveParameterSelector()
    
    sorted_configs = sorted(initial_scores.items(), key=lambda x: x[1], reverse=True)
    worst_configs = [c[0] for c in sorted_configs[:60]]
    
    print(f"\n🎯 Top 5 worst configs: {worst_configs[:5]}")
    print(f"🚀 Starting V2 optimization (time: {MAX_TIME_SECONDS/3600:.1f}h)")
    print("="*70)
    
    cycle = 0
    total_bbox3_improvements = 0
    total_local_improvements = 0
    
    while time.time() - start_time < MAX_TIME_SECONDS:
        cycle += 1
        elapsed = time.time() - start_time
        
        print(f"\n--- Cycle {cycle} ({elapsed/60:.1f}m elapsed) ---")
        
        # Phase 1: bbox3 runs (these handle overlaps internally)
        print("  [Phase 1] bbox3 optimization")
        for _ in range(3):
            if time.time() - start_time >= MAX_TIME_SECONDS:
                break
            
            n, r = param_selector.get_params()
            prev_score = best_score
            
            try:
                subprocess.run(["./bbox3", "-n", str(n), "-r", str(r)],
                             capture_output=True, timeout=BBOX3_TIMEOUT)
            except subprocess.TimeoutExpired:
                print(f"    ⏱️ bbox3 timeout")
                continue
            
            df = pd.read_csv("submission.csv")
            new_score, _ = eval_df_sub(df, False)
            
            improvement = prev_score - new_score
            improved = improvement > 1e-15
            param_selector.record_result(n, r, improved, improvement)
            
            if new_score < best_score:
                best_score = new_score
                best_df = df.copy()
                total_bbox3_improvements += 1
                print(f"    ✅ bbox3 n={n} r={r}: {improvement:.12f}")
        
        # Phase 2: Local optimization (now with overlap checks!)
        if time.time() - start_time < MAX_TIME_SECONDS:
            print("  [Phase 2] Local optimization")
            
            _, current_scores = eval_df_sub(df, False)
            sorted_configs = sorted(current_scores.items(), key=lambda x: x[1], reverse=True)
            worst_configs = [c[0] for c in sorted_configs[:40]]
            
            configs_to_optimize = random.sample(worst_configs, min(10, len(worst_configs)))
            
            for config_n in configs_to_optimize:
                if time.time() - start_time >= MAX_TIME_SECONDS:
                    break
                
                strategies = [
                    ('SA', lambda: simulated_annealing_config(df, config_n, SA_ITERATIONS)),
                    ('Boundary', lambda: optimize_boundary_trees(df, config_n, 80)),
                    ('Gradient', lambda: gradient_descent_config(df, config_n, GRADIENT_STEPS)),
                    ('Swap', lambda: try_swap_trees(df, config_n, 30)),
                ]
                
                strategy_name, strategy_fn = random.choice(strategies)
                try:
                    improved, gain = strategy_fn()
                    if improved:
                        total_local_improvements += 1
                        print(f"    ✅ {strategy_name} on config {config_n}: {gain:.12f}")
                except Exception as e:
                    print(f"    ⚠️ {strategy_name} error on config {config_n}: {e}")
        
        # Phase 3: Rotation (every 3 cycles)
        if cycle % 3 == 0 and time.time() - start_time < MAX_TIME_SECONDS:
            print("  [Phase 3] Rotation grid search")
            for config_n in random.sample(worst_configs, min(5, len(worst_configs))):
                try:
                    improved, gain = rotation_grid_search(df, config_n, angle_step=10)
                    if improved:
                        print(f"    ✅ Rotation on config {config_n}: {gain:.12f}")
                except Exception as e:
                    pass
        
        # Phase 4: Basin hopping (every 5 cycles)
        if cycle % 5 == 0 and time.time() - start_time < MAX_TIME_SECONDS:
            print("  [Phase 4] Basin hopping")
            for config_n in random.sample(worst_configs, min(3, len(worst_configs))):
                try:
                    improved, gain = basin_hopping_config(df, config_n, hops=5, local_steps=40)
                    if improved:
                        print(f"    ✅ Basin hop on config {config_n}: {gain:.12f}")
                except Exception as e:
                    pass
        
        # Save and validate
        df.to_csv("submission.csv", index=False)
        new_score, _ = eval_df_sub(df, False)
        
        if new_score < best_score:
            best_score = new_score
            best_df = df.copy()
            print(f"  📈 New best: {best_score:.12f}")
        
        if cycle % 10 == 0:
            print(f"\n  === Status: Score={best_score:.12f}, "
                  f"bbox3={total_bbox3_improvements}, local={total_local_improvements} ===")
    
    # Final save with best found
    best_df.to_csv("submission.csv", index=False)
    
    print("\n" + "="*70)
    print("OPTIMIZATION COMPLETE")
    print("="*70)
    final_score, _ = eval_df_sub(best_df, False)
    
    print(f"\n📈 Results:")
    print(f"   Initial:  {initial_score:.12f}")
    print(f"   Final:    {final_score:.12f}")
    print(f"   Improved: {initial_score - final_score:.12f}")
    print(f"   Cycles:   {cycle}")
    print(f"   bbox3 improvements: {total_bbox3_improvements}")
    print(f"   Local improvements: {total_local_improvements}")
    print(f"   Total time: {(time.time()-start_time)/3600:.2f}h")

if __name__ == "__main__":
    main()
```

# Last ensemble

```python
!python3 ./ensemble_submissions.py -d /kaggle/working -o /kaggle/working/submission_final.csv
!mv /kaggle/working/submission_final.csv /kaggle/working/submission.csv
```

# Fix overlaps

```python
import pandas as pd
import numpy as np
from decimal import Decimal, getcontext
from shapely import affinity
from shapely.geometry import Polygon
from shapely.strtree import STRtree

# Set precision for Decimal (25 is good for contest standards)
getcontext().prec = 25
scale_factor = Decimal("1e18")


class ChristmasTree:
    """Represents a single, rotatable Christmas tree of a fixed size."""

    def __init__(self, center_x="0", center_y="0", angle="0"):
        """Initializes the Christmas tree with a specific position and rotation."""
        self.center_x = Decimal(center_x)
        self.center_y = Decimal(center_y)
        self.angle = Decimal(angle)

        trunk_w = Decimal("0.15")
        trunk_h = Decimal("0.2")
        base_w = Decimal("0.7")
        mid_w = Decimal("0.4")
        top_w = Decimal("0.25")
        tip_y = Decimal("0.8")
        tier_1_y = Decimal("0.5")
        tier_2_y = Decimal("0.25")
        base_y = Decimal("0.0")
        trunk_bottom_y = -trunk_h

        # Define the 15 vertices of the tree polygon
        initial_polygon = Polygon(
            [
                (Decimal("0.0") * scale_factor, tip_y * scale_factor),
                (top_w / Decimal("2") * scale_factor, tier_1_y * scale_factor),
                (top_w / Decimal("4") * scale_factor, tier_1_y * scale_factor),
                (mid_w / Decimal("2") * scale_factor, tier_2_y * scale_factor),
                (mid_w / Decimal("4") * scale_factor, tier_2_y * scale_factor),
                (base_w / Decimal("2") * scale_factor, base_y * scale_factor),
                (trunk_w / Decimal("2") * scale_factor, base_y * scale_factor),
                (trunk_w / Decimal("2") * scale_factor, trunk_bottom_y * scale_factor),
                (-(trunk_w / Decimal("2")) * scale_factor, trunk_bottom_y * scale_factor),
                (-(trunk_w / Decimal("2")) * scale_factor, base_y * scale_factor),
                (-(base_w / Decimal("2")) * scale_factor, base_y * scale_factor),
                (-(mid_w / Decimal("4")) * scale_factor, tier_2_y * scale_factor),
                (-(mid_w / Decimal("2")) * scale_factor, tier_2_y * scale_factor),
                (-(top_w / Decimal("4")) * scale_factor, tier_1_y * scale_factor),
                (-(top_w / Decimal("2")) * scale_factor, tier_1_y * scale_factor),
            ]
        )
        
        # Apply rotation and translation to the polygon
        rotated = affinity.rotate(initial_polygon, float(self.angle), origin=(0, 0))
        self.polygon = affinity.translate(
            rotated, 
            xoff=float(self.center_x * scale_factor), 
            yoff=float(self.center_y * scale_factor)
        )


def load_configuration_from_df(n: int, df: pd.DataFrame) -> list[ChristmasTree]:
    """
    Loads all trees for a given N from the submission DataFrame.
    """
    group_data = df[df["id"].str.startswith(f"{n:03d}_")]
    trees = []
    for _, row in group_data.iterrows():
        # Remove 's' prefix and convert to string for Decimal constructor
        x = str(row["x"])[1:]
        y = str(row["y"])[1:]
        deg = str(row["deg"])[1:]
        
        # Ensure values are present before passing to ChristmasTree constructor
        if x and y and deg:
            trees.append(ChristmasTree(x, y, deg))
        else:
             # Handle cases where configuration might be incomplete/missing
             pass 
             
    return trees


def get_score(trees: list[ChristmasTree], n: int) -> float:
    """
    Calculates the score (S^2 / N) for a given configuration of trees.
    S is the side length of the minimum bounding square.
    """
    if not trees:
        return 0.0

    # Collect all exterior points from all tree polygons, scale them back down
    xys = np.concatenate([np.asarray(t.polygon.exterior.xy).T / float(scale_factor) for t in trees])
    
    min_x, min_y = xys.min(axis=0)
    max_x, max_y = xys.max(axis=0)
    
    side_length = max(max_x - min_x, max_y - min_y)
    
    # Score is S^2 / N
    score = side_length**2 / n
    return score

def has_overlap(trees: list[ChristmasTree]) -> bool:
    """Check if any two ChristmasTree polygons overlap."""
    if len(trees) <= 1:
        return False

    polygons = [t.polygon for t in trees]
    # Use STRtree for efficient proximity queries (optimizes checking pairs)
    tree_index = STRtree(polygons)

    for i, poly in enumerate(polygons):
        # Query for polygons whose bounding boxes overlap with poly
        # This returns the indices of potential overlaps
        indices = tree_index.query(poly)
        
        for idx in indices:
            # Skip checking the polygon against itself
            if idx == i:
                continue
                
            # Perform the precise intersection check
            if poly.intersects(polygons[idx]) and not poly.touches(polygons[idx]):
                # Overlap found!
                return True
    return False

# ----------------------------------------------------------------------

def score_and_validate_submission(file_path: str, max_n: int = 200) -> dict:
    """
    Reads a submission CSV, calculates the total score, and checks for overlaps 
    in all configurations (N=1 up to max_n).
    """
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return {"status": "FAILED", "error": "File Not Found"}
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return {"status": "FAILED", "error": f"CSV Read Error: {e}"}

    total_score = 0.0
    failed_overlap_n = []
    
    print(f"--- Scoring and Validation: {file_path} (N=1 to {max_n}) ---")

    for n in range(1, max_n + 1):
        trees = load_configuration_from_df(n, df)
        
        # Only process if there are trees for this N in the file
        if trees:
            current_score = get_score(trees, n)
            total_score += current_score

            if has_overlap(trees):
                failed_overlap_n.append(n)
                print(f"  ❌ N={n:03d}: OVERLAP DETECTED! (Score contribution: {current_score:.6f})")
            else:
                # Optionally print success for each N
                # print(f"  ✅ N={n:03d}: OK (Score contribution: {current_score:.6f})")
                pass
        
    print("\n--- Summary ---")
    if failed_overlap_n:
        print(f"❌ **Validation FAILED**: Overlaps found in N: {failed_overlap_n}")
        status = "FAILED (Overlaps)"
    else:
        print("✅ **Validation SUCCESSFUL**: No overlaps detected.")
        status = "SUCCESS"
        
    print(f"**Total Submission Score (Σ S²/N): {total_score:.6f}**")
    
    return {
        "status": status,
        "total_score": total_score,
        "failed_overlap_n": failed_overlap_n
    }


# Example usage (assuming 'submission.csv' exists in the current directory)
result = score_and_validate_submission("submission.csv", max_n=200)
print(result)
```

```python
import pandas as pd
import numpy as np

FAILED_N_LIST =  result['failed_overlap_n']
GOOD_CSV_PATH = "/kaggle/input/why-not/submission.csv"
NEW_CSV_PATH = "submission.csv" 
OUTPUT_CSV_PATH = "submission.csv" 

def replace_invalid_configurations(new_csv_path, good_csv_path, output_csv_path, failed_n_list):
    df_new = pd.read_csv(new_csv_path)
    df_good = pd.read_csv(good_csv_path)
    failed_prefixes = [f"{n:03d}_" for n in failed_n_list]
    df_to_keep = df_new[~df_new["id"].str.startswith(tuple(failed_prefixes))]
    df_replacement = df_good[df_good["id"].str.startswith(tuple(failed_prefixes))]
    df_repaired = pd.concat([df_to_keep, df_replacement]).sort_values(by="id").reset_index(drop=True)
    df_repaired.to_csv(output_csv_path, index=False) #float_format='%.25f')
    print(f"\n--- SUCCESS ---")
replace_invalid_configurations(NEW_CSV_PATH, GOOD_CSV_PATH, OUTPUT_CSV_PATH, FAILED_N_LIST)
```

```python
result = score_and_validate_submission("submission.csv", max_n=200)
print(result)
```