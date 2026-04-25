# santa-2025: top public notebooks

The community's top-voted notebooks focus exclusively on geometric packing optimization, high-precision arithmetic, and spatial indexing rather than traditional machine learning or EDA. Authors predominantly develop utility pipelines and custom C++/Python hybrid optimizers (e.g., simulated annealing, gradient descent, and external solvers like bbox3) to minimize bounding box sizes while rigorously validating non-overlap constraints. Ensemble strategies and iterative post-processing are frequently employed to blend public submissions and repair invalid configurations within strict computational budgets.

## Common purposes
- utility
- other
- ensemble
- baseline

## Competition flows
- Procedurally generates tree coordinates and angles using a greedy collision-avoidance algorithm, then exports the results to a CSV submission file.
- Loads a CSV submission, optimizes tree rotations to minimize bounding boxes, runs an external solver (`bbox3`) with a parameter grid under time constraints, applies a post-processing script (`shake_public`), fixes overlapping groups using a donor file, and saves the improved submission.
- Reads a submission CSV, optimizes tree positions and rotations via a parallelized C++ simulated annealing solver, validates results for overlaps using Python/shapely, repairs invalid configurations with a reference file, and outputs a corrected submission CSV.
- Downloads an initial submission CSV, iteratively optimizes tree placements using a parallelized C++ Simulated Annealing algorithm, applies a backward propagation refinement phase, and saves the improved configuration until convergence.
- Loads a baseline submission CSV, runs an external binary solver (bbox3) across phased parameter sweeps, applies geometric post-processing and overlap repair, validates against a custom scoring function, and outputs the best valid submission within a 3-hour budget.
- Loads an initial tree packing configuration from a CSV, optimizes positions and rotations for each group size N using a custom C++ hybrid gradient/stochastic solver, validates the result for overlaps using high-precision Python geometry, and automatically reverts any invalid groups to the original baseline.
- Loads a submission CSV, validates geometric configurations for overlaps and scores them using `shapely`, then replaces invalid configurations with those from a reference submission to produce a repaired output file.
- The notebook defines a geometric packing heuristic for identical trees, parallelizes the search for optimal bounding boxes across tree counts 1–200, computes a custom aggregate score, and exports the coordinates and angles to a CSV submission file.
- Collects multiple public submission CSVs, selects the best configuration per puzzle by bounding-box score, validates for overlaps using high-precision Shapely, and refines the result via a custom C++ simulated annealing optimizer with fractional translations and edge-based compaction to produce a final submission.
- Loads multiple spatial placement submissions, iteratively optimizes them using an external C++ solver and custom Python local search strategies while enforcing strict non-overlap constraints, ensembles the best per-group configurations across all inputs, and validates/fallbacks any overlapping groups before saving the final submission.

## Data reading
- Reads CSV files using `pandas.read_csv`, strips 's' prefixes from `x`, `y`, and `deg` columns, and splits the `id` column into `group_id` and `item_id`.
- Parses submission CSVs using pandas, extracts id/x/y/deg columns, strips 's' prefixes, and converts coordinates/angles to Decimal or float for geometric operations.
- Parses submission.csv using C++ ifstream, extracting tree count n, coordinates, and rotation angles from comma-separated values with 's' prefixes, grouping them by n into a map.
- Loads a baseline CSV, strips the 's' prefix from x/y/deg columns, and splits the id column into group_id and item_id.
- Reads initial configuration from /kaggle/input/team-optimization-blend/submission.csv using C++ ifstream and string parsing, extracting id, x, y, and deg fields. Python validation reads the output CSV using pandas.read_csv.
- pd.read_csv for submission files external binary ./bbox3 executed via shell command
- Reads multiple CSV files from Kaggle input directories using glob.glob and pd.read_csv, filtering for files containing 'id', 'x', 'y', 'deg' columns.
- pandas.read_csv and csv.DictReader to parse id, x, y, deg columns from /kaggle/input/*/*.csv and /kaggle/input/santa-submission/submission.csv, stripping 's' prefixes and converting to floats/Decimals.

## Data processing
- Uses `Decimal` with 25-digit precision and a `1e15` scaling factor to prevent floating-point drift during iterative polygon rotations and translations.
- Constructs precise `Decimal`-based `shapely` polygons for each tree, applies rotation and translation, calculates bounding box side lengths, detects collisions via `shapely.STRtree`, and replaces invalid groups with valid ones from a donor CSV.
- Implements a custom C++ simulated annealing optimizer with 14 geometric move types, aggressive overlap repair, global scaling, and local search. Python layer uses shapely.STRtree for efficient overlap detection, calculates S²/N scores, and replaces overlapping configurations with valid ones from a reference submission.
- Constructs shapely polygons from coordinates, applies rotation and translation, detects overlaps via STRtree, tightens bounding box angles using scipy.optimize.minimize_scalar, repairs overlaps by swapping in baseline groups, and enforces a strict wall-time budget.
- C++ code applies rotation matrices and translations to a fixed 15-vertex polygon template. Python code parses 's' prefixed strings to floats, calculates bounding boxes, and computes scores. High-precision validation uses decimal.Decimal with 25-digit precision and shapely for geometric operations.
- Geometric validation using shapely polygons and STRtree for efficient overlap detection High-precision coordinate scaling with decimal module Configuration replacement logic to merge valid and invalid submission rows
- Strips 's' prefix from coordinates and angles, computes bounding box scores per puzzle, validates geometric overlaps using high-precision Shapely with Decimal arithmetic, and sorts configurations by puzzle ID and index before saving.
- High-precision coordinate scaling via Decimal with 1e18 factor, Shapely polygon construction/rotation/translation, strict overlap validation using STRtree, and fallback replacement of overlapping groups with a pre-validated submission.

## Models
- bbox3
- bbox3 (external C++ optimizer)
- Simulated Annealing
- Gradient Descent
- Basin Hopping
- Swap moves
- Rotation Grid Search

## Frameworks used
- shapely
- pandas
- numpy
- matplotlib
- decimal
- scipy
- tqdm
- concurrent.futures
- numba
- rich

## Loss functions
- minimize (side^2 / n) score where side is the bounding box side length per group

## Ensembling
- Blends multiple public submissions by selecting the best configuration per puzzle based on bounding-box score, then applies a custom C++ simulated annealing optimizer with fractional translations and edge-based compaction to refine the ensemble.
- Per-group (n) best-of selection across multiple input submissions, followed by a final ensemble pass over the working directory to produce the output file.

## Insights
- Using `shapely`'s `STRtree` enables efficient spatial indexing for collision detection instead of O(n^2) pairwise checks.
- Optimizing rotation angles via `scipy.optimize.minimize_scalar` effectively minimizes bounding box sizes for geometric packing problems.
- Simulated annealing with domain-specific geometric moves effectively minimizes bounding box size for complex polygonal packing.
- Backward propagation effectively transfers structural insights from larger configurations to improve smaller ones, exploiting hierarchical packing patterns.
- Phased hyperparameter sweeps (short → medium → long runs) effectively balance exploration with limited computational budgets.
- Blending diverse public submissions and selecting the best per puzzle provides a strong, reproducible baseline for geometric packing competitions.
- Focusing optimization efforts on the worst-performing groups yields the most significant score improvements.

## Critical findings
- Overlap detection is computationally expensive but mandatory; using Shapely's STRtree significantly speeds up pairwise intersection checks.
- Floating-point precision issues can cause false overlap detections, necessitating Decimal with a large scale factor.

## Notable individual insights
- votes 2032 (Santa 2025 - Getting Started): Using `shapely`'s `STRtree` enables efficient spatial indexing for collision detection instead of O(n^2) pairwise checks.
- votes 602 (Santa-submission): Optimizing rotation angles via `scipy.optimize.minimize_scalar` effectively minimizes bounding box sizes for geometric packing problems.
- votes 402 (Why Not): Simulated annealing with domain-specific geometric moves effectively minimizes bounding box size for complex polygonal packing.
- votes 384 (Santa Claude): Backward propagation effectively transfers structural insights from larger configurations to improve smaller ones, exploiting hierarchical packing patterns.
- votes 377 (SANTA 2025 | Best-Keeping bbox3 Runner): Phased hyperparameter sweeps (short → medium → long runs) effectively balance exploration with limited computational budgets.
- votes 192 ([Santa25] Ensemble + SA + Fractional Translation): Blending diverse public submissions and selecting the best per puzzle provides a strong, reproducible baseline for geometric packing competitions.
- votes 186 (BBOX3 - Ensemble Update): Focusing optimization efforts on the worst-performing groups yields the most significant score improvements.

## Notebooks indexed
- #2032 votes [[notebooks/votes_01_inversion-santa-2025-getting-started/notebook|Santa 2025 - Getting Started]] ([kaggle](https://www.kaggle.com/code/inversion/santa-2025-getting-started))
- #602 votes [[notebooks/votes_02_saspav-santa-submission/notebook|Santa-submission]] ([kaggle](https://www.kaggle.com/code/saspav/santa-submission))
- #402 votes [[notebooks/votes_03_jazivxt-why-not/notebook|Why Not]] ([kaggle](https://www.kaggle.com/code/jazivxt/why-not))
- #384 votes [[notebooks/votes_04_smartmanoj-santa-claude/notebook|Santa Claude]] ([kaggle](https://www.kaggle.com/code/smartmanoj/santa-claude))
- #377 votes [[notebooks/votes_05_yongsukprasertsuk-santa-2025-best-keeping-bbox3-runner/notebook|SANTA 2025 | Best-Keeping bbox3 Runner]] ([kaggle](https://www.kaggle.com/code/yongsukprasertsuk/santa-2025-best-keeping-bbox3-runner))
- #256 votes [[notebooks/votes_06_jazivxt-eazy-optimizer/notebook|Eazy Optimizer]] ([kaggle](https://www.kaggle.com/code/jazivxt/eazy-optimizer))
- #248 votes [[notebooks/votes_07_nctuan-happy-christmas/notebook|Happy Christmas]] ([kaggle](https://www.kaggle.com/code/nctuan/happy-christmas))
- #234 votes [[notebooks/votes_08_zaburo-88-32999-a-well-aligned-initial-solution/notebook|[88.32999] A Well-Aligned Initial Solution]] ([kaggle](https://www.kaggle.com/code/zaburo/88-32999-a-well-aligned-initial-solution))
- #192 votes [[notebooks/votes_09_jonathanchan-santa25-ensemble-sa-fractional-translation/notebook|[Santa25] Ensemble + SA + Fractional Translation]] ([kaggle](https://www.kaggle.com/code/jonathanchan/santa25-ensemble-sa-fractional-translation))
- #186 votes [[notebooks/votes_10_hvanphucs112-bbox3-ensemble-update/notebook|BBOX3 - Ensemble Update]] ([kaggle](https://www.kaggle.com/code/hvanphucs112/bbox3-ensemble-update))
