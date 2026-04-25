# google-code-golf-2025: cross-solution summary

This ARC Code Golf competition challenged participants to minimize Python byte counts for 400 grid-based algorithmic tasks, with winning approaches heavily favoring manual optimization, advanced compression techniques, and targeted LLM agentic workflows over traditional ML pipelines. Top solutions converged on leveraging custom deflate compressors, Huffman tree alignment, and iterative human-in-the-loop refinement to exploit the non-linear relationship between uncompressed code and compressed file size. Success ultimately depended on deep domain expertise in code golf, strategic context engineering for LLMs, and aggressive pruning of redundant logic rather than automated model ensembling or complex feature engineering.

## Competition flows
- Iterative LLM parallel sampling loop guided by AST-based rule prompts and compression-aware metrics, with SQLite logging and manual refinement
- Manual task analysis, Python function writing, and byte optimization via recursion, bitwise ops, regex, and compression tools
- ARC grid analysis via ACT framework to identify invariants/transforms, LLM candidate generation, iterative optimization against a growing trick checklist, and submission
- Raw grid processing through custom Python solutions, optimized for byte size via custom deflate compressor and seed regeneration
- Manual transformation of 400 raw grid inputs into byte-minimized Python functions using advanced golfing techniques
- JSON parsing, LLM/manual solution generation, byte optimization via charbrute fuzzer and pysearch, genetic variable renaming, and submission via Flask/WebSocket UI
- LLM-driven agent proposing code, validating against tests, minifying via python-minifier, and iteratively improving byte length via Evolutionary DB and context-engineered prompts
- Manual implementation and optimization of Python code for 400 tasks, tracked via custom dashboard for iterative byte reduction

## Data processing
- Extracted input/output pairs from JSON test cases to construct search tuples for pysearch
- Code normalization (newlines and encoding) before byte measurement
- Minification via python-minifier for whitespace, imports, constant folding, and safe name shortening

## Models
- GPT-5
- GPT-5-Codex
- Grok4
- Grok4-Code-Fast-1
- Gemini (2.5 Flash, 2.5 Pro, 2.5 Flash Lite)
- Qwen3-Coder
- Kimi K2
- DeepSeek R1
- Claude Code

## Frameworks used
- Python
- zlib
- GPT-o4-mini
- Flask
- WebSocket
- PyPy 3.11
- pysearch
- ast.unparse

## Ensembling
- Brief experimentation with model ensembling (e.g., Grok4) abandoned due to prompt engineering limits and API costs, favoring single-model parallel sampling and manual refinement

## Notable individual insights
- Rank 4 (Parallel Sampling + Rule-based Prompt Generation): Optimizing for compression-friendly code (minimizing Kolmogorov complexity via zip size) effectively escapes local optima and yields highly compact solutions.
- Rank 5 (5th-Place writeup): A custom compressor combining hill climbing on Zopfli's output with dynamic programming for optimal Huffman code lengths outperformed Zopfli itself.
- Rank 5 (5th-Place writeup): Test case seeds were simply `2025 + i` per example index, allowing exact regeneration of specific tasks via ARC-GEN.
- Rank 1 (1st place write-up): Compression fundamentally changes golfing strategy by making repetition nearly free, shifting focus to using consistent constructs over finding the shortest per-step construct.
- Rank 2 (2nd place solution): A 'pop rotation' technique using `.pop()` completely eliminates the need for `zip()` when rotating square grids in list comprehensions.
- Rank 19 (19th Place Yuchen20): Too many I/O pairs, exemplars, and prior solutions cause context rot; aggressive pruning to 5 I/O, 3 exemplars, and 3 priors was necessary.
- Rank 25 (Getting to Rank 25): ARC puzzles are constructed like generalized Sudoku: a clean target grid is corrupted by noise and transformations, making the solver's job inverse reconstruction rather than creative generation.

## Solutions indexed
- #1 [[solutions/rank_01/solution|1st place write-up: Code Golf International]]
- #2 [[solutions/rank_02/solution|2nd place solution]]
- #4 [[solutions/rank_04/solution|Parallel Sampling + Rule-based Prompt Generation (4th place)]]
- #5 [[solutions/rank_05/solution|5th-Place writeup (with Better Compression Algorithm and Seed Cracking)]]
- #8 [[solutions/rank_08/solution|8th place write-up: import itertools]]
- #9 [[solutions/rank_09/solution|9th place write-up]]
- #19 [[solutions/rank_19/solution|19th Place Yuchen20 part of write up (a LLM agentic solution)]]
- #25 [[solutions/rank_25/solution|Getting to Rank 25 by Teaching LLMs to Golf]]

## GitHub links
- [Natanaelel/GoogleCodeGolf2025](https://github.com/Natanaelel/GoogleCodeGolf2025) _(solution)_ — from [[solutions/rank_08/solution|8th place write-up: import itertools]]
- [lynn/pysearch](https://github.com/lynn/pysearch) _(library)_ — from [[solutions/rank_08/solution|8th place write-up: import itertools]]
- [Seek64/NeurIPS-Code-Golf-2025](https://github.com/Seek64/NeurIPS-Code-Golf-2025) _(reference)_ — from [[solutions/rank_05/solution|5th-Place writeup (with Better Compression Algorithm and Seed Cracking)]]
- [key-moon/deflate-viz](https://github.com/key-moon/deflate-viz) _(library)_ — from [[solutions/rank_05/solution|5th-Place writeup (with Better Compression Algorithm and Seed Cracking)]]
- [key-moon/golf](https://github.com/key-moon/golf) _(solution)_ — from [[solutions/rank_05/solution|5th-Place writeup (with Better Compression Algorithm and Seed Cracking)]]
- [google/ARC-GEN](https://github.com/google/ARC-GEN) _(reference)_ — from [[solutions/rank_05/solution|5th-Place writeup (with Better Compression Algorithm and Seed Cracking)]]
- [michaelhodel/arc-dsl](https://github.com/michaelhodel/arc-dsl) _(reference)_ — from [[solutions/rank_05/solution|5th-Place writeup (with Better Compression Algorithm and Seed Cracking)]]
- [michaelhodel/re-arc](https://github.com/michaelhodel/re-arc) _(reference)_ — from [[solutions/rank_05/solution|5th-Place writeup (with Better Compression Algorithm and Seed Cracking)]]
- [Seek64/NeurIPS-Code-Golf-2025](https://github.com/Seek64/NeurIPS-Code-Golf-2025) _(solution)_ — from [[solutions/rank_01/solution|1st place write-up: Code Golf International]]
- [lynn/pysearch](https://github.com/lynn/pysearch) _(library)_ — from [[solutions/rank_01/solution|1st place write-up: Code Golf International]]
- [jailctf-merger-goog-golf/compression](https://github.com/jailctf-merger-goog-golf/compression) _(solution)_ — from [[solutions/rank_02/solution|2nd place solution]]
- [jailctf-merger-goog-golf/golf](https://github.com/jailctf-merger-goog-golf/golf) _(solution)_ — from [[solutions/rank_02/solution|2nd place solution]]
- [lynn/pysearch](https://github.com/lynn/pysearch) _(library)_ — from [[solutions/rank_02/solution|2nd place solution]]
- [madler/infgen](https://github.com/madler/infgen) _(library)_ — from [[solutions/rank_02/solution|2nd place solution]]
- [michaelhodel/arc-dsl](https://github.com/michaelhodel/arc-dsl) _(library)_ — from [[solutions/rank_19/solution|19th Place Yuchen20 part of write up (a LLM agentic solution)]]
- [google/ARC-GEN](https://github.com/google/ARC-GEN) _(library)_ — from [[solutions/rank_19/solution|19th Place Yuchen20 part of write up (a LLM agentic solution)]]
- [4eta/google-code-golf-2025](https://github.com/4eta/google-code-golf-2025) _(solution)_ — from [[solutions/rank_09/solution|9th place write-up]]
- [kajott/adventofcode](https://github.com/kajott/adventofcode) _(reference)_ — from [[solutions/rank_09/solution|9th place write-up]]
- [google/ARC-GEN](https://github.com/google/ARC-GEN) _(reference)_ — from [[solutions/rank_09/solution|9th place write-up]]
- [lynn/pysearch](https://github.com/lynn/pysearch) _(reference)_ — from [[solutions/rank_09/solution|9th place write-up]]

## Papers cited
- [Fuun](https://web.archive.org/web/20110724172015/http://save-endo.cs.uu.nl/Endo.pdf)
