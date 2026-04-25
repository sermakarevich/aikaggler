# fide-google-efficiency-chess-ai-challenge: top public notebooks

The top-voted notebooks primarily focus on establishing functional chess AI baselines and optimizing agent performance under Kaggle's strict time and memory constraints. Rather than relying on machine learning, the community emphasizes rule-based heuristics, bitboard representations, and the seamless integration of pre-compiled C/C++ engines via the UCI protocol. Key themes include dynamic time management, environment parity for compiled binaries, and iterative debugging to handle arbitrary board states efficiently.

## Common purposes
- tutorial
- baseline
- utility

## Competition flows
- Sets up the Kaggle chess environment, writes a rule-based agent that parses FEN states and applies a priority heuristic for moves, tests it against a random opponent, and explains the submission format.
- Compiles a C/C++ chess engine, wraps it in a Python UCI communication script, and packages it into a tar.gz archive ready for Kaggle's chess environment submission.
- Reads the FEN board state via the Kaggle environment API, converts it to bitboards, evaluates legal moves using a heuristic scoring function with time constraints, and returns the best move as an algebraic string for submission packaging.
- The notebook sets up the Kaggle chess environment, writes a Python agent using the Chessnut library with heuristic evaluation and minimax search, tests it against baseline bots, and prepares it for submission under strict memory and time constraints.
- Downloads a C chess engine source, modifies its hash table size to meet submission limits, compiles it, wraps it in a Python UCI interface, and packages it for Kaggle’s chess environment evaluation.
- Reads the current board state as a FEN string from the environment, passes it to a compiled Stockfish 4.0 binary via UCI protocol to compute the best move within a 100ms time limit, and returns the move for submission.
- Compiles C source code into a platform-compatible binary, wraps it in a Python agent script using subprocess for I/O, packages both into a .tar.gz archive, and submits it to the Kaggle competition environment.
- Parses chess board states, integrates a C++ UCI engine via Python subprocesses, simulates games locally, and packages the agent for submission while monitoring memory constraints.
- Clones and compiles a C++ chess engine, wraps it in a Python UCI interface to process FEN board states, and packages the agent for Kaggle game submission.
- Initializes the Kaggle chess environment, sets a custom opening via FEN notation, visualizes a Gaussian skill-rating update mechanism, and implements a rule-based submission agent that prioritizes checkmates and captures before submitting moves.

## Data reading
- Extracts the board state as a FEN string from the obs['board'] field in the environment observation.
- Parses the obs.board FEN string using a custom fen_to_bitboards function from bitboard_utils to generate occupancy bitboards for each piece type.
- Board states are read as FEN strings from the obs.board observation object provided by the kaggle-environments chess environment.
- Extracts the raw board state as a FEN string directly from the obs['board'] dictionary provided by the kaggle-environments API.
- Passes competition observations (e.g., board state) to the compiled binary via Python subprocess stdin, parsed by C's scanf.
- Extracts board FEN string, player color, and time limits from the environment's `obs` dictionary; uses `Chessnut` to parse FEN and enumerate legal moves.
- Parses board state from FEN strings provided in the `obs['board']` field by the Kaggle chess environment.
- Uses `kaggle_environments.make()` to instantiate the environment and parses board states from FEN strings via the `Chessnut` library within the submission agent.

## Data processing
- None explicitly; the engine handles board state parsing internally via the UCI protocol.
- Simulates each legal move on the bitboard, retrieves opponent responses, calculates net capture value minus threat value, applies square-value bonuses, adds random noise, and scales the search time limit based on remaining game time.
- The FEN string is parsed directly using the Chessnut library to initialize the game state and generate legal moves via game.get_moves().
- None; relies on raw FEN strings and manages engine memory via UCI commands (setoption name Hash value 1, setoption name Clear Hash) to prevent crashes during long matches.
- None; relies on the engine's native FEN parsing and move generation.
- Converts remaining time from seconds to milliseconds for UCI protocol; parses FEN board states and generates legal move lists.

## Features engineering
- Piece value map (pawn to king), standard board map for center control, king safety board map, dynamic time constraint scaling, and random temperature noise for evaluation stochasticity.
- Hand-crafted heuristic features include piece values (pawn=1, knight=3, bishop=3, rook=5, queen=9), amplified capture bonuses, promotion bonuses, center control bonuses (d4, e4, d5, e5), check detection bonuses, and penalties for moves that leave own pieces under attack.

## Models
- Fruit chess engine
- BBC (C chess engine)
- Stockfish 4.0
- Random move bot
- Micromax (MCU-Max) C++ chess engine

## Frameworks used
- kaggle-environments
- Chessnut
- bitboard_utils
- pygame
- psutil
- subprocess
- numpy
- matplotlib
- scipy

## CV strategies
- Self-play evaluation loop alternating colors, running 5 to 20 games against the agent and several public bots, tracking wins, losses, timeouts, and average reward.

## Insights
- The Kaggle chess environment can be programmatically controlled using the kaggle-environments package and the Chessnut library for board state parsing.
- A simple priority-based heuristic (checkmate > capture > promotion > random) can serve as a functional baseline agent without any machine learning.
- Agents must output moves in UCI notation and correctly handle the FEN string format provided by the environment observation.
- Compiling a lightweight C/C++ engine and communicating via subprocess is an effective way to integrate external chess engines into Kaggle's Python-based agent interface.
- Using the UCI protocol allows seamless exchange of FEN positions and best moves between Python and compiled binaries.
- Setting a fixed movetime (100ms) and clearing the hash table after each move helps manage memory and latency constraints in the Kaggle environment.
- Heuristic chess agents can be built effectively using bitboard representations and simple evaluation functions without complex ML pipelines.
- Dynamic time allocation per move prevents catastrophic timeouts better than fixed limits.
- Adding controlled randomness to move evaluations increases strategic variety and prevents predictable play.
- Custom bitboard utilities significantly improve Python-based chess bot performance over standard libraries.
- Strict per-move time limits require dynamic depth adjustment and early termination in search algorithms to avoid timeouts.
- Heuristic evaluation heavily benefits from amplifying capture and promotion bonuses to prioritize material gain over positional play.
- Penalizing moves that leave own pieces under attack significantly improves defensive resilience in the agent.
- LLMs can effectively assist in generating and debugging complex algorithmic code, but iterative, function-by-function refinement yields more reliable results than bulk generation.
- Pre-compiled C engines can be effectively integrated into Kaggle environments via Python wrappers using the UCI protocol.
- Submission size constraints can be met by modifying source code parameters before compilation.
- Time allocation for engine moves can be derived from the environment's remainingOverageTime observation.
- Leveraging a pre-compiled, established chess engine yields significantly higher performance than training a custom model under strict time constraints.
- Older engine versions like Stockfish 4.0 are more suitable for Kaggle's resource-limited environments compared to modern NNUE-based versions that require heavy tablebase support.
- Managing subprocess memory via UCI commands is critical to prevent crashes during long matches.
- Compiling C/C++ locally using a Docker container matching the Kaggle submission base image ensures binary compatibility.
- Python's subprocess module is the reliable method for passing inputs to and capturing outputs from compiled binaries in Kaggle submissions.
- Submission archives must be .tar.gz format containing the Python agent script (main.py) and the compiled binary at the root level.
- Bots interact with the environment through a single function receiving an `obs` dictionary and returning a UCI move string.
- C/C++ engines are significantly faster for complex calculations and can be integrated into Python via subprocess and the UCI protocol.
- Memory tracking is critical due to the strict 5 MiB RAM limit, and tools like `psutil` or `valgrind` can monitor process memory usage.
- Wrapping a compiled C/C++ binary via the UCI protocol is a viable and efficient strategy for game AI submissions in Kaggle.
- Explicitly handling pawn promotion moves (e.g., appending 'q') is necessary when the underlying engine or FEN parser does not auto-complete them.
- Lightweight, pre-compiled engines can outperform heavy ML models in time-constrained game environments.
- FEN notation can be used to override the default starting position in the Kaggle chess environment.
- Player skill and uncertainty can be effectively modeled as a Gaussian distribution where wins increase the mean rating and decrease uncertainty.
- A simple heuristic prioritizing checkmates, captures, and promotions provides a functional baseline for early-game decision making.

## Critical findings
- The agent struggles with delivering checkmates beyond mate-in-one and frequently blunders pieces due to its shallow evaluation depth.
- It initially suffered from a panic mode under time pressure, which was resolved by implementing a decreasing time constraint per move.
- Initial LLM-generated bots sometimes performed worse than the baseline, highlighting the difficulty of generating robust chess logic without iterative debugging.
- The competition environment starts games from randomly selected openings rather than the standard initial position, requiring the agent to handle arbitrary board states efficiently.
- Directly uploading compiled executables or .zip archives often fails due to platform differences, size limits, or Kaggle's submission prompt restrictions.
- Shell commands invoked directly in Python often fail or cause timeouts in the submission environment, making subprocess the necessary workaround.

## What did not work
- The original V1 version used a fixed panic mode for time management that caused erratic behavior, replaced by a decreasing time constraint in V2.
- Several square-value maps were dropped in V2 after genetic algorithm tuning showed they provided diminishing returns.
- Bulk LLM-generated code for the chess engine initially underperformed the baseline, prompting a shift to implementing and testing one function at a time with iterative debugging.
- Attempting to compile code remotely via shell commands or upload raw executables resulted in timeouts, platform incompatibility, or submission size limit errors.
- Using .zip archives or direct shell invocations failed due to working directory issues and Kaggle's submission agent restrictions.

## Notable individual insights
- votes 1350 (Your First Chess Bot): A simple priority-based heuristic (checkmate > capture > promotion > random) can serve as a functional baseline agent without any machine learning.
- votes 365 (source 2100 LB c/c++ sample submission): Compiling a lightweight C/C++ engine and communicating via subprocess is an effective way to integrate external chess engines into Kaggle's Python-based agent interface.
- votes 221 (Tiny Chess Bot | Small Snail v3.1🐌): Dynamic time allocation per move prevents catastrophic timeouts better than fixed limits, and adding controlled randomness to move evaluations increases strategic variety.
- votes 150 (LittleDeepBlue): LLMs can effectively assist in generating and debugging complex algorithmic code, but iterative, function-by-function refinement yields more reliable results than bulk generation.
- votes 125 (LB over 2,000 with C++ Stockfish 4.0): Older engine versions like Stockfish 4.0 are more suitable for Kaggle's resource-limited environments compared to modern NNUE-based versions that require heavy tablebase support.
- votes 106 (🤯 running C/C++ in submissions): Compiling C/C++ locally using a Docker container matching the Kaggle submission base image ensures binary compatibility, while .tar.gz archives containing main.py and the binary are required for submission.

## Notebooks indexed
- #1350 votes [[notebooks/votes_01_bovard-your-first-chess-bot/notebook|Your First Chess Bot]] ([kaggle](https://www.kaggle.com/code/bovard/your-first-chess-bot))
- #365 votes [[notebooks/votes_02_olegzholobov-source-2100-lb-c-c-sample-submission/notebook|source 2100 LB c/c++ sample submission]] ([kaggle](https://www.kaggle.com/code/olegzholobov/source-2100-lb-c-c-sample-submission))
- #221 votes [[notebooks/votes_03_lennarthaupts-tiny-chess-bot-small-snail-v3-1/notebook|Tiny Chess Bot | Small Snail v3.1🐌]] ([kaggle](https://www.kaggle.com/code/lennarthaupts/tiny-chess-bot-small-snail-v3-1))
- #150 votes [[notebooks/votes_04_wguesdon-littledeepblue/notebook|LittleDeepBlue]] ([kaggle](https://www.kaggle.com/code/wguesdon/littledeepblue))
- #145 votes [[notebooks/votes_05_egrehbbt-bbc-bitboard-chess-engine-in-c/notebook|BBC (BitBoard Chess engine in C)]] ([kaggle](https://www.kaggle.com/code/egrehbbt/bbc-bitboard-chess-engine-in-c))
- #125 votes [[notebooks/votes_06_garried-lb-over-2-000-with-c-stockfish-4-0/notebook|LB over 2,000 with C++ Stockfish 4.0]] ([kaggle](https://www.kaggle.com/code/garried/lb-over-2-000-with-c-stockfish-4-0))
- #106 votes [[notebooks/votes_07_snufkin77-running-c-c-in-submissions/notebook|🤯 running C/C++ in submissions]] ([kaggle](https://www.kaggle.com/code/snufkin77/running-c-c-in-submissions))
- #92 votes [[notebooks/votes_08_vitalykudelya-chess-ai-essentials-a-practical-guide/notebook|Chess AI Essentials: A Practical Guide]] ([kaggle](https://www.kaggle.com/code/vitalykudelya/chess-ai-essentials-a-practical-guide))
- #86 votes [[notebooks/votes_09_ayucha-oh-no-my-rook-a-c-guide/notebook|OH NO MY ROOK ♖ A C++ Guide]] ([kaggle](https://www.kaggle.com/code/ayucha/oh-no-my-rook-a-c-guide))
- #85 votes [[notebooks/votes_10_jocelyndumlao-ruy-l-pez-gaussian-chess-mastery/notebook|🏆 Ruy López: Gaussian Chess Mastery ]] ([kaggle](https://www.kaggle.com/code/jocelyndumlao/ruy-l-pez-gaussian-chess-mastery))
