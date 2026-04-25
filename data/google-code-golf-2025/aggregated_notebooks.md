# google-code-golf-2025: top public notebooks

The top-voted notebooks focus exclusively on algorithmic optimization and extreme code minification for a grid-based pattern manipulation competition, rather than traditional machine learning pipelines. They emphasize systematic verification against official benchmarks, aggressive byte-level compression techniques, and cross-notebook aggregation to maximize scoring efficiency. The community prioritizes correctness and syntactic golfing over model training or feature engineering.

## Common purposes
- utility
- other
- tutorial
- ensemble

## Competition flows
- Loads 2D integer grids via a custom utility, applies minified algorithmic transformations to solve specific pattern manipulation tasks, and outputs the resulting grids.
- Loads example input-output pairs for a specified task, visualizes them, writes a candidate code snippet to a file, and runs a built-in verification function to check correctness against all benchmarks.
- Loads training examples for each task via a custom utility, then applies a minified Python script to generate the solution output for submission.
- Scans multiple input datasets and submission archives for tasks 001-400, verifies candidate Python solutions against task examples, selects the smallest working code for each task, and packages them into optimized submission and metadata files.
- Loads task examples and baseline code, deduces grid transformation rules, implements a solver, applies automated regex-based code golf optimizations, verifies correctness, and packages the final submission.
- The notebook aggregates candidate Python programs from multiple Kaggle datasets, verifies their correctness against ARC task examples, selects the smallest valid program per task, and packages them into a single submission zip file.
- Copies task solutions from Kaggle datasets, verifies them against official examples, applies Zopfli compression to minimize byte count, and generates a scored submission.zip archive.
- Reads task examples, verifies candidate Python solutions against them, compresses the code using zlib/zopfli to minimize byte count, and calculates a competition score based on the 2500-byte limit per task.
- Extracts submission archives from multiple community notebooks, verifies each Python script against example inputs/outputs, selects the smallest valid solution per task, and packages them into a final submission zip.
- Reads candidate Python solutions from multiple input directories, applies minification and custom zlib compression to minimize file size, verifies correctness against task examples, selects the smallest valid solution per task, and packages them into a final submission zip.

## Data reading
- Loads training examples using `load_examples(n)` from a shared `code_golf_utils` package; data is represented as nested lists of integers.
- Uses `load_examples(task_num)` from the `code_golf_utils` package to load training and test example pairs.
- Uses `load_examples(task_id)['train']` from a custom `code_golf_utils` module to fetch and display training examples.
- Loads task examples from JSON files containing test/train/arc-gen categories
- Reads candidate solutions as text or binary files
- Auto-detects Kaggle vs local environment to set input paths
- Uses `load_examples()` from a custom `code_golf_utils` module to fetch train/test grids.
- Reads task metadata and baseline solutions via `json.load()` and raw byte file reads.
- Loads task examples via a custom load_examples(task_num) function from code_golf_utils
- Reads Python source files from multiple Kaggle input paths
- Parses JSON examples for verification
- Reads task-specific Python files (task*.py) from Kaggle dataset inputs via shell commands and loads them as binary strings for compression and scoring.
- Loads task examples via `load_examples(task_num)` from a custom utility module, concatenates train and test sets, and formats grid inputs/outputs as strings.
- Reads raw submission files from `/kaggle/input/google-code-golf-2025-submit` and processes them sequentially.
- Loads task examples via a custom load_examples function from code_golf_utils.
- Reads Python source files (task{task_num:03d}.py) and submission.zip archives from Kaggle dataset paths using os.path, zipfile, and shutil.
- Loads candidate solution .py files and task example JSONs from multiple source directories, reading all files as latin-1 encoded text.

## Data processing
- Direct grid manipulation using list comprehensions, coordinate arithmetic, and mathematical operations; includes cropping, resizing, rotation, reflection, and pattern filling without explicit normalization or augmentation.
- Extracts/copies files from submission.zip archives or direct directories
- Cleans working directories and leftover files
- Sorts candidates by file size (smallest first)
- Verifies outputs by executing code via exec() and comparing JSON results to expected arrays using numpy.array_equal
- Extracts non-zero coordinates, color distributions, and bounding boxes from input grids to identify transformation rules.
- Applies regex-based whitespace stripping, operator compaction, indentation minimization, and built-in function aliasing to source code.
- Applies Zopfli compression to source code to minimize byte size, strips trailing whitespace, and compares compressed versus original lengths to determine the optimal representation.
- Formats grid data into string representations for verification.
- Applies aggressive code compression by testing multiple trailing bytes, quote delimiters, and encoding headers.
- Sanitizes compressed bytes for safe string embedding and validates syntax before finalizing.
- Extracts zipped submissions and filters out invalid or non-existent files.
- Verifies correctness using a custom validator that checks output equality against examples.
- Selects files based on byte size while ignoring syntax warnings and handling verification errors.
- Removes UTF-8 BOMs, applies python-minifier for code minification, and uses a custom zip_src function that iterates over zopfli and zlib compression with various trailing bytes and string delimiters to find the shortest valid executable script.

## Frameworks used
- code_golf_utils
- numpy
- tqdm
- matplotlib
- zipfile
- gcgc_utils
- zlib
- zopfli
- multiprocessing
- ast
- re
- json
- copy
- warnings
- python-minifier

## Ensembling
- Combines candidate solutions from multiple community kernels, verifies each against task examples, and selects the smallest valid program per task to form the final submission.
- Combines the smallest valid solution for each of the 400 tasks from multiple community notebooks into a single submission archive, effectively creating a meta-solution optimized for byte count.

## Insights
- Highly compressed Python code can efficiently solve complex grid-based pattern transformations using coordinate mapping and list comprehensions.
- Symmetry detection and connected component analysis are key techniques for identifying and manipulating patterns in integer grids.
- Code golf competitions reward algorithmic creativity and mathematical shortcuts over readability or standard library usage.
- The notebook demonstrates how to use the official competition utility library to validate code solutions against provided examples and hidden benchmarks.
- Extreme code golf techniques like `eval`, `exec`, and regex substitution can effectively solve complex visual reasoning tasks without traditional ML.
- Bitwise operations and mathematical indexing tricks are powerful tools for grid manipulation and pattern detection.
- Recursive list comprehensions and string manipulation allow for highly compact implementations of image transformations like rotation, reflection, and expansion.
- Executing external code safely requires verifying function existence and output format before comparing results.
- File size is a direct proxy for code golf scoring, making it a reliable optimization target.
- Binary file copying preserves exact byte counts, ensuring accurate size comparisons across different sources.
- Correctness must always be verified before attempting code golf optimizations.
- Automated variant generation with systematic regex transformations can reliably reduce code size across hundreds of tasks.
- Prioritizing optimization on the longest baseline solutions yields the highest score gains.
- Grid transformation puzzles in this competition are solved algorithmically rather than through machine learning.
- Aggregating diverse community solutions significantly increases task coverage compared to relying on a single kernel.
- Code golf competitions reward program size, making smallest-valid-program selection a critical optimization step.
- Caching verification results prevents redundant computation and speeds up iterative development.
- Using aliases and combining multiple assignments can significantly reduce code length in Python code golf.
- Hardcoding fixed grid dimensions (e.g., 21x21) allows bypassing dynamic size calculations for shorter code.
- Zopfli compression is highly effective for golfed code, but sometimes hardcoded constants or redundant assignments can negate compression gains.
- Aliases may not always improve compression efficiency when Zopfli is used, as the compressor might expand the alias instead of shortening it.
- Code golf competitions reward extreme byte minimization, making compression a critical step.
- Verifying solutions against all examples (train + test) is necessary to ensure correctness before submission.
- String escaping and delimiter selection significantly impact compressed payload size.
- A systematic pipeline for extraction, verification, and compression is more reliable than manual optimization.
- Task 157 requires a longer timeout due to execution time, highlighting that some puzzles are computationally heavy despite being code golf.
- Compression effectiveness varies significantly across tasks, with some yielding no improvement over raw code, indicating that payload structure dictates compression gains.
- Compression techniques can significantly reduce submission size without altering functionality, directly impacting leaderboard scores.
- Aggregating the best solutions across multiple community notebooks is an effective strategy for maximizing performance in code golf competitions.
- The scoring system heavily penalizes larger byte counts, making size optimization a primary objective.
- The compression/decompression process itself can impact the leaderboard score, suggesting that file size metrics are sensitive to how solutions are packaged.
- Some solutions may pass size checks but fail verification if errors are ignored, highlighting the need for strict correctness validation.
- Combining minification with custom zlib compression and strategic trailing bytes/delimiters yields greater byte savings than minification alone.
- Verification overhead is a critical bottleneck, making selective skipping for known slow tasks necessary for pipeline efficiency.
- Multi-source fallback logic is essential for maintaining submission integrity when default solutions fail verification.

## Critical findings
- The notebook acknowledges that some solutions require guessing directions or patterns when explicit rules are ambiguous.
- The author notes that the collection is a collaborative effort nearing 400 tasks, highlighting the community-driven nature of the competition.
- A correct solution, even if slightly longer, is infinitely more valuable than an incorrect short one.
- Aliasing built-in functions like `range` and `len` as single-letter parameters can save bytes when used multiple times.
- Replacing `or` with `|` inside list comprehensions can sometimes reduce character count without changing logic.
- Aliases may not always improve compression efficiency when Zopfli is used, as the compressor might expand the alias instead of shortening it.
- Task 157 requires a longer timeout due to execution time, highlighting that some puzzles are computationally heavy despite being code golf.
- Compression effectiveness varies significantly across tasks, with some yielding no improvement over raw code, indicating that payload structure dictates compression gains.
- The compression/decompression process itself can impact the leaderboard score, suggesting that file size metrics are sensitive to how solutions are packaged.
- Some solutions may pass size checks but fail verification if errors are ignored, highlighting the need for strict correctness validation.

## Notable individual insights
- votes 228 (Compilation of winning solutions): Extreme code golf techniques like `eval`, `exec`, and regex substitution can effectively solve complex visual reasoning tasks without traditional ML.
- votes 170 (Google Code Golf Championship 101): Correctness must always be verified before attempting code golf optimizations.
- votes 140 (GCGC playground): Aliases may not always improve compression efficiency when Zopfli is used, as the compressor might expand the alias instead of shortening it.
- votes 138 (Google Code Golf New Community Best): Compression effectiveness varies significantly across tasks, with some yielding no improvement over raw code, indicating that payload structure dictates compression gains.
- votes 118 (R30 NeurIPS Golf | Lessons Learned): The compression/decompression process itself can impact the leaderboard score, suggesting that file size metrics are sensitive to how solutions are packaged.
- votes 113 (Google Code Golf Community Best): Combining minification with custom zlib compression and strategic trailing bytes/delimiters yields greater byte savings than minification alone.

## Notebooks indexed
- #331 votes [[notebooks/votes_01_jazivxt-oh-barnacles/notebook|Oh Barnacles]] ([kaggle](https://www.kaggle.com/code/jazivxt/oh-barnacles))
- #269 votes [[notebooks/votes_02_mmoffitt-neurips-2025-google-code-golf-championship/notebook|NeurIPS 2025 - Google Code Golf Championship]] ([kaggle](https://www.kaggle.com/code/mmoffitt/neurips-2025-google-code-golf-championship))
- #228 votes [[notebooks/votes_03_garrymoss-compilation-of-winning-solutions/notebook|Compilation of winning solutions]] ([kaggle](https://www.kaggle.com/code/garrymoss/compilation-of-winning-solutions))
- #215 votes [[notebooks/votes_04_tonylica-400-task-with-smart-solution-search/notebook|400 Task with Smart Solution Search]] ([kaggle](https://www.kaggle.com/code/tonylica/400-task-with-smart-solution-search))
- #170 votes [[notebooks/votes_05_adilshamim8-google-code-golf-championship-101/notebook|Google Code Golf Championship 101]] ([kaggle](https://www.kaggle.com/code/adilshamim8/google-code-golf-championship-101))
- #144 votes [[notebooks/votes_06_kerta27-solutions-for-all-400-tasks-ensemble/notebook|Solutions for all 400 Tasks + Ensemble]] ([kaggle](https://www.kaggle.com/code/kerta27/solutions-for-all-400-tasks-ensemble))
- #140 votes [[notebooks/votes_07_jonathanchan-gcgc-playground/notebook|GCGC playground]] ([kaggle](https://www.kaggle.com/code/jonathanchan/gcgc-playground))
- #138 votes [[notebooks/votes_08_tonylica-google-code-golf-new-community-best/notebook|Google Code Golf New Community Best]] ([kaggle](https://www.kaggle.com/code/tonylica/google-code-golf-new-community-best))
- #118 votes [[notebooks/votes_09_mcwema-r30-neurips-golf-lessons-learned/notebook|R30 NeurIPS Golf | Lessons Learned]] ([kaggle](https://www.kaggle.com/code/mcwema/r30-neurips-golf-lessons-learned))
- #113 votes [[notebooks/votes_10_tonylica-google-code-golf-community-best/notebook|Google Code Golf Community Best ]] ([kaggle](https://www.kaggle.com/code/tonylica/google-code-golf-community-best))
