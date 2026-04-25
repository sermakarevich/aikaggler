# 2nd place solution (@zaburo's part)

- **Author:** KaizaburoChubachi
- **Date:** 2025-02-01T00:02:47.557Z
- **Topic ID:** 560533
- **URL:** https://www.kaggle.com/competitions/santa-2024/discussion/560533
---

First of all, I would like to express my gratitude to the Kaggle team for continuously hosting this annual competition, my teammates @solverworld and @danielphalen for their hard work, and all the other participants who contributed to making the competition exciting through discussions in the forum and on the leaderboard. I truly enjoyed this experience, and I was especially thrilled when I found a solution of 28.5 for Problem 5.

Since other members of my team have also posted their write-ups, I encourage you to check them out as well:
- @danielphalen's write-up: https://www.kaggle.com/competitions/santa-2024/discussion/560540
- @solverworld's write-up: https://www.kaggle.com/competitions/santa-2024/discussion/560565
Here, I will introduce my approach to solving the problem.

My solution is based on [Iterated Local Search](https://en.wikipedia.org/wiki/Iterated_local_search) using restricted k-opt and a custom kick. The k-opt + kick approach is a common technique for solving the Traveling Salesman Problem (TSP), and I adapted it to fit this particular problem.

### Restricted K-opt

K-opt is a generalization of [2-opt](https://en.wikipedia.org/wiki/2-opt) and [3-opt](https://en.wikipedia.org/wiki/3-opt) algorithms for TSP, where k edges are replaced to improve the solution. This process is repeated until no further improvements can be made. The number of possible moves in k-opt increases exponentially with k, but in a standard TSP setting, the score changes due to a move can be computed locally, and the number of moves can be significantly reduced through efficient edge selection, making it feasible to execute within a reasonable time.

For this problem, score changes due to a move cannot be computed locally, and overall score calculation is very slow. Therefore, to apply k-opt, it was necessary to drastically reduce the number of candidate moves. However, due to the order-dependence of scores, simply filtering candidates based on the movement cost of a single word was not effective. I reduced k-opt moves using the following strategies:

1. Restricting moves that involve flips

In TSP, edges are undirected, so flipping does not change the score. However, in this problem, reversing the order significantly worsens the score. Therefore, I did not use moves that involve flipping.

2. Restricting the length of moving subsequences

I couldn't imagine many cases where moving long subsequences would be effective, so I restricted the length of moving subsequences. However, I allowed large stationary subsequences, meaning that swaps between distant words were not restricted.

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F408648%2F3dc1e6818645c7e947b84889ebbd3232%2F408665998-0e2243c5-6d76-4a35-9b18-5ad04ce79ed8.png?generation=1738368297559746&alt=media)

Specifically, I used `(k=3, max_moving_size=5)` and `(k=4, max_moving_size=1)`. Even for the largest Problem 5, it was possible to test all moves within about 18 minutes on a 4090 GPU (of course, since the process restarts upon finding an improvement, full convergence takes longer).

### Custom Kick

Once Restricted K-opt converged, I applied some kick operations to escape local optima. In TSP, typical kick methods include Double Bridge Kick and kicking by some random 2-opt moves. I used these methods at the beginning of the competition.

After some time, I realized that, in this problem, it was crucial to randomly separate groups of adjacent words. Therefore, I introduced a new kick that selects a randomly sized subsequence (e.g., `n=5` or `n=10`) and shuffles it. This kick was particularly effective in improving solutions for Problem 5.

For finding the 28.5 solution for Problem 5, I used a Problem 5-specific kick that leveraged the structure of its solutions.

In Problem 5, it was shared that a very good initial solution could be obtained by sorting stop words and non-stop words, resulting in a structure like `(sorted stop words)(sorted non-stop words)`. Applying k-opt to this structure generally resulted in solutions of the form `(stop words)(nearly sorted non-stop words)`. 

This structure (according to our experiments) only achieved a score of around 32.xx. However, our team broke the 32.xx barrier by further splitting the structure into `(some stop words)(some optimized non-words)(other stop words)(nearly sorted other non-stop words)`. We could reach such a solution by using, for example, `(solution of Problem 4)(other stop words)(sorted other non-stop words)` as an initial solution.

However, even with this structure, we still got stuck far from the first-place score. To overcome this, I developed custom kicks that transferred words between `(some stop words)(some optimized non-words)(other stop words)` and `(nearly sorted other non-stop words)`, either by simply moving them or swapping them. Specific implementation is as follows:

```python
def custom_kick1(base_text: str, n: int = 5) -> str:
    words = base_text.split()
    words_orig = list(words)

    advent_index = words.index("advent")
    swapped = set()
    for _ in range(n):
        while True:
            i = np.random.randint(0, advent_index)
            if i in swapped:
                continue
            candidates = [j for j in range(advent_index + 1, len(words)) if words[j][0] == words[i][0]]
            if len(candidates) == 0:
                continue
            j = np.random.choice(candidates)

            words[i], words[j] = words[j], words[i]
            swapped.add(i)
            break

    assert sorted(words_orig) == sorted(words)
    return " ".join(words)
```

Additionally, we introduced another kick that performed swaps. These custom kicks had a significant impact. After running for some time, these allowed us to discover the solution with the score of 28.5.

### Other Things

- Since the global structure of this solution method is greatly influenced by the initial solutions, a multi-start approach was effective. In fact, the best solution in Problem 4 was achieved by employing a multi-start approach and further optimizing the best solutions found by multi-staring. (At that time, my team member had already reached this value, though, so it was not a new discovery.)
- For Problem 3, we suspected the presence of quite attractive local optima. To avoid getting trapped in it, we forced a different approach by fixing the last word position and testing all words in that position. This led us to discover a solution with a score of 191.x.
- During the early stages of the competition, I experimented with the beam search algorithm. However, designing a good evaluation function (that is not short-sighted) was challenging, so I did not adopt it.
- I did not use Simulated Annealing (SA) or Genetic Algorithms (GA). SA requires temperature control, making it unsuitable for long-term, intermittent computation for this competition. GA was not used because applying efficient crossover methods such as Edge Assembly Crossover (which works well for TSP) seemed difficult for this problem.
