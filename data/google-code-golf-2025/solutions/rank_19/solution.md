# 19th Place Yuchen20 part of write up (a LLM agentic solution)

- **Author:** Yuchen20
- **Date:** 2025-11-06T01:32:22.083Z
- **Topic ID:** 614729
- **URL:** https://www.kaggle.com/competitions/google-code-golf-2025/discussion/614729

**GitHub links found:**
- https://github.com/michaelhodel/arc-dsl
- https://github.com/google/ARC-GEN

---

Here is Yuchen20’s part of the solution write-up. I built an evolutionary program-synthesis agent that pushed our score to ~0.940xxx midway through the competition. At that point I hit the practical limits of both current LLMs (I was mostly using Gemini's free tier api key) and my agent loop, so I pivoted to manual code-golfing and merged into a team. My teammates explored complementary directions—zlib function compression, AST-aware normalization to shave redundant characters, running smaller models augmented with stronger code golf tricks, and even golfing with Claude Code—and together these tracks carried us to our final score and 19th place. Thank you so much for my team, I could not reach this far without my team.
Here’s the improved version:

> I wrote a blog post here: [https://yuchen-mao.vercel.app/blog/google-code-golf](https://yuchen-mao.vercel.app/blog/google-code-golf). Please see it for more details about this write-up.

---
I entered the NeurIPS 2025 Google Code Golf Championship with no prior code-golf experience, and with a different mindset: could a carefully engineered LLM-driven agent not just solve, but golf, these solutions competitively?


ARC-AGI tasks are really abstract, at a glance, even humans need a moment to spot the rule in these examples. For LLMs, there are a few more factors that makes this even harder:

* **Abstract, multi-step rules.** Even humans pause to understand the pattern; ARC-AGI patterns are often abstract and compositional, causing LLMs to hallucinate heavily.
* **Numeric form, spatial semantics.** The inputs are matrices, but the underlying logic is more about spatial layout than arithmetic. LLMs, trained and heavily RL on mathematical problems, often misalign with or get confused by these topological tasks.
* **High task diversity.** With hundreds of distinct patterns, manual labeling or human-in-the-loop hints isn’t truely feasible. The LLM has to work it out alone.
* **Code Golfing is also hard** Beyond a point, saving a single character often requires a complete logic rewrite. In the final stages of the competition, trimming 1–2 bytes usually demanded a new approach.

To address these challenges, we built a small agent, and experimented extensively with context engineering. 

## My Agent
Given these challenges, the architecture below delivered reliable working solutions in the early stages, and consistent **byte minimization** in the later phases of the competition. At a high level, the system comprises **three cooperating components** (Fig. 1): an **Evolutionary DB** for retrieval and diversification; a **Context Engineering** stage that assembles the system prompt (role, task specification, ARC-GEN snippets, cross-task exemplars, golf idioms); and an **Agent Loop** that proposes code, runs tests, ingests failures, and retains the shortest correct program.

## Agent Overview
![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F14322371%2F64025c4a10380bf2599675ca680967ed%2Fmy-code-golf-agent.png?generation=1762391215877659&alt=media)

### Components at a Glance

- **Evolutionary DB.**  
  A pool of 400 tasks, each with its current best (lowest-byte) solution, plus a per-task set of correctness-verified prior LLM solutions that are longer than the best. These longer solutions are used for diversification and crossover.

- **Context Engineering.**  
  More a stage than a standalone component, this phase assembles the system prompt by experimenting with different sources of information to help the LLM better understand the task. The final prompt includes concise role instructions (as an expert code-golfer), the current task specification and ARC-GEN generator, a small selection of other tasks (with examples and best or near-best solutions) to induce transferable knowledge, and a golf idiom cookbook (safe, broadly applicable byte-saving patterns).

- **Agent Loop (LLM + tools).**  
  The LLM proposes code; a toolchain executes it against generator-defined tests, measures byte length, and returns tracebacks and failing I/O pairs on error. Passing proposals are compared to the best solution; the shortest correct solution is persisted and pushed back to the DB.


### Agent Loop

This was the straightforward part: the system is essentially a program-synthesis loop tailored for ARC-AGI code golf. We designed this agentic loop to be as lean as possible, offloading most of the heavy lifting to **Context Engineering** and the **Evolutionary DB**. This was largely out of necessity, as we relied primarily on Gemini’s generous free tier (with occasional experiments on OpenRouter to test alternative models within a tight budget). To keep costs and latency down, we **avoided heavy add-ons** (like long-horizon planning, persistent memory, or code search) that would increase the number of LLM calls. The simple loop above gave us fast iteration, predictable costs, and let the LLM do the real work.

To build this lean agentic loop, we equipped it with a few essential tools:

* **Extract code** from LLM outputs (strip reasoning, capture the final Python block or code delimited by markers).
* **Validate** the snippet: ensure it defines a single entry point (`p(g)` or `p = lambda g: ...`), uses only the Python standard library, and parses without syntax errors.
* **Run and score** against task examples. On failure, return a concise **traceback** and the **failing I/O pairs**.
* **Minify/normalize** with [`python-minifier`](https://pypi.org/project/python-minifier/) for easy byte wins (whitespace, imports, constant folding, safe name shortening), then measure the final byte count.

Thus the entire agent loop can thus be described as:
1. **Prompt** the LLM with the system + user context to propose code.
2. **Extract → validate → minify.** If invalid, feed back a short reason (syntax/import/entry-point).
3. **Execute tests.**  
   - If **pass**: compare byte length to the incumbent; if shorter, **persist** as new best. Then asking it to continue shorten it.
   - If **fail**: return the traceback + failing I/O pairs as the next user message.
4. **Iterate** until the iteration budget is exhausted or no improvement for \(K\) rounds.

Here are some **Validator guardrails** that we checked when the LLM produced an answer:
- Enforce `def p(g): ...` or `p = lambda g: ...` only; no top-level I/O.
- **Ban imports** except safe stdlib used implicitly by the runner; no network/FS.
- Cap runtime per test; kill on timeouts; treat exceptions as failures.
- Normalise newlines and encoding before measuring `len(code.encode())`.


### Evolutionary DB

The **Evolutionary DB** is a lightweight index over tasks and solution variants, supporting both crossover and diversification. It stores per-task metadata and a history of correctness-verified working solution of each task (including longer variants), which seed structurally diverse rewrites.

**Data model (per task):**

* `best_bytes` (our best), `public_best_bytes` (if known), `solved` flag
* `solutions`: correctness-verified programs
* `last_updated`, optional `pattern_signature` for similarity search

**Sampling for a new iteration:**


1. **Pick a focus task.** This is the task the agent will attempt to solve or further shorten. Prefer tasks with room for improvement or those not yet solved. Sampling uses a softmax over the *margin*:
   $$
   \begin{align}
   m(t) &= 
   \begin{cases}
   \text{public\_best\_bytes}(t)-\text{best\_bytes}(t), & \text{if solved}\\[2pt]
   M, & \text{if unsolved (large constant)}
   \end{cases}\\
   p(t) &\propto \exp\!\big(\beta\,\mathrm{norm}(m(t))\big),
   \end{align}
   $$
   where $\beta$ controls exploitation vs. exploration.

2. **Retrieve in-task priors.**
   From the focus task’s archive, sample $N$ prior solutions with an inverse-length bias to favor concise patterns but maintain diversity:
   $$
   p(s\mid t) \propto \exp!\big(-\gamma\cdot \text{bytes}(s)\big).
   $$

3. **Add cross-task exemplars.**
   Select $K$ additional tasks to import transferable knowledge and help escape local minima. Early on, these were chosen at random; later, we switched to **pattern-similarity** retrieval via extracted pattern tags.

4. **Crossover from priors.**
   Include the best solution and a small set of diverse, correctness-verified working solutions from the selected exemplar tasks. These offer a record of how other tasks were golfed, allowing the model to transfer byte-saving motifs or refactor logic to the current task and synthesize a shorter, behavior-equivalent program.

5. **Mutate.**
   The LLM proposes a candidate; we run deterministic tests, capture failing I/O and tracebacks, and measure byte length post-normalization.

6. **Update the DB.**
   Any **correct** candidate is added to the archive; if it beats `best_bytes`, it becomes the new incumbent.


### Context Engineering

This was where we spent most of our time: figuring out what information *helps* the agent reason effectively, and what causes the LLM to suffer from **context rot** or distraction.

We divided this into two key parts:

1. **Prompt Construction for Solving ARC-AGI** — information that helps the LLM produce a *working* solution to an ARC-AGI task.
2. **Prompt Construction for Code Golfing** — information that helps the LLM *shorten* or *golf* an already working solution.


---

#### Prompt Construction for Solving ARC-AGI

When I joined the competition, several existing community resources were already invaluable for bootstrapping first working solutions:

* **[arc-dsl](https://github.com/michaelhodel/arc-dsl)** — A *Domain-Specific Language* for ARC-AGI tasks, with a [Python DSL file](https://github.com/michaelhodel/arc-dsl/blob/main/dsl.py) that defines composable primitives for common visual and logical transformations. While not comprehensive, these building blocks can express a large portion of ARC-AGI tasks. There’s also a [solver module](https://github.com/michaelhodel/arc-dsl/blob/main/solvers.py) showing how to assemble complete solutions with the DSL.

* **[ARC-GEN](https://github.com/google/ARC-GEN)** — This repo contains the *generator code* for the ARC-AGI competition dataset. Each generator defines a family of puzzles with shared transformations. Although it doesn’t always expose the high-level pattern directly, it provides crucial signals to help the model infer the underlying rules and constraints.

To ground the model further, we included a few raw **input→output** pairs as plain matrices, e.g.:

```
**Example 1:**
Input:
[[0, 1, 1], [0, 0, 0], [0, 1, 1]]
Output:
[[0, 0, 0, 0, 1, 1, 0, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 1, 0, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 1, 0, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 1, 0, 1, 1]]
```


##### First-pass system prompt

With these components in place, here’s the first version of the system prompt I used:

---

1. **System role** (expert Python code-golfer for ARC-AGI)
2. **Exemplar tasks**
   * ARC-GEN generator snippet
   * arc-dsl solution
   * Working solutions from the evolutionary database
3. **Focus task**
   * ARC-GEN generator snippet
   * arc-dsl solution
   * Task examples (input→output pairs)
4. **Output rules** enforcing a short, single-function, working program for the **focus task**

---

This prompt solved *about half* the tasks without manual coding, even with a fast, non-flagship model (gemini-2.5-flash). However, progress plateaued. We traced this to a mismatch between **arc-dsl** assumptions and the **ARC-GEN** task distributions:

* **Generator drift / artefacts:** The competition included many extra examples per task generated from ARC-GEN, some deviating subtly from the “canonical” pattern that is described in the original arc-agi task.
* **DSL shortcuts:** Some arc-dsl solutions solved the original training/test grids via clever shortcuts, but didn’t generalize to the broader examples.

The upshot: arc-dsl sometimes *confused* the model on certain tasks. Our fix was simple but effective, we **removed arc-dsl code** from the prompt, keeping only ARC-GEN snippets.

After this pruning, the agentic loop resumed and reached **~360 solved tasks**. For the remainder, I wrote manual solutions or adapted vetted public ones into the codebase.

#### Prompt Construction for Code Golfing

This is where we assembled a compact, high-leverage set of **code-golf idioms**.

##### Sourcing the idioms

I started by asking the model for generic Python golf tips. These were helpful, but shallow. Next, I searched community resources. Most code-golf sites only show **solution lengths**, not code, so transfer is limited. Eventually, I found an archived competition site: [https://steffan153.github.io/wg-sol/](https://steffan153.github.io/wg-sol/). It has fewer than 50 problems; most aren’t matrix-heavy, but a few transfer well to ARC-AGI. The key: submissions are visible and ranked by length, so you can track the incremental byte-saving tricks. I used ChatGPT agent mode to crawl the archive and distill recurring patterns into prompt-ready snippets.

##### The Evolutionary DB as a teacher

Beyond the codebook, the **Evolutionary DB** turned out to be the best teacher for code golf. When a breakthrough happens—like a logic refactor that shaves bytes but preserves behavior—subsequent runs can **retrieve** that variant, and the model will often transfer the same motif to a different task with a similar pattern. In some cases, the transfer is almost direct. For example, **[Task 246](https://arcprize.org/play?task=a2fd1cf0)** and **[Task 335](https://arcprize.org/play?task=d4a91cb9)** share the same pattern, differing only in numbers/colors:

**ARC-AGI Example 246**

> [Puzzle 246 of ARC-AGI](https://arcprize.org/play?task=a2fd1cf0): Extend the red dot rightward and the green dot upward with a blue trace; stop when they meet.

**ARC-AGI Example 335**

> [Puzzle 335 of ARC-AGI](https://arcprize.org/play?task=d4a91cb9): Extend the red dot leftward and the blue dot downward with a yellow trace; stop when they meet.

So they are basically the same pattern.

Thanks to the Evolutionary DB-as-teacher approach, a breakthrough on one propagates to the other, so the final solutions are almost identical:

```
def p(g):
 D,E=sum(g,[]).index,divmod;F,C,A,B=E(D(3),(G:=len(g[0])))+E(D(2),G)
 while B-C:B+=B<C or-1;g[A][B]=8
 while A-F:g[A][C]=8;A+=A<F or-1
 return g
```

and

```
def p(g):
 D,E=sum(g,[]).index,divmod;F,C,A,B=E(D(8),G:=len(g[0]))+E(D(2),G)
 while B-C:B+=B<C or-1;g[A][B]=4
 while A-F:g[A][C]=4;A+=A<F or-1
 return g
```

This is an extreme example, but we frequently saw related patterns across tasks. That’s exactly where the Evolutionary DB shines in our agent.



##### Fighting context rot

We learned the hard way that too many input→output pairs, exemplars, and prior solutions can make the model drift. Early on, we used about 10 I/O pairs, 8 exemplars, and ~10 prior solutions per prompt—often blowing the context window. We pruned aggressively:

* **I/O pairs:** 10 → **5**
* **Cross-task exemplars:** 8 → **3**
* **Prior solutions:** 10 → **3** (and we shortened the long early arc-dsl solutions, which were often hundreds of lines)

**Why fewer I/O pairs?**

* Grids can be large (e.g., 28×28), and matrices are extremely token-inefficient: each digit, bracket, and comma is a token, so token count scales quickly.

**Why fewer prior solutions?**

* Early “working” solutions inherited from **arc-dsl** were often verbose scaffolding, with really long but working solution. Including too many consumed context and distracted the model.

**Why fewer exemplars?**

* More random exemplars increased the odds of hitting a useful match (tasks with similar patterns), but also injected noise.
* Instead, we added **pattern tags** per task (e.g., *flood-fill*, *flip*, *extend rays*, *copy block*), derived from the DSL’s modular sub-patterns. When sampling, we only retrieve **pattern-similar** tasks—cutting exemplars from 8 to 3 without losing relevance.
##### When scaling models stopped helping

About halfway through the competition (~1 month left, leaderboard ≈ **0.940xxx**),

 I hit a ceiling. I tried switching to larger and newer models
*(qwen3-coder, kimi-k2, deepseek-R1, gemini-2.5-flash-lite → gemini-2.5-flash → gemini-2.5-pro)*
and expanded the “thinking budget” (**12k → 32,768** tokens). But—**no further shortenings**.

At this point, neither prompt design nor retrieval was the bottleneck. The remaining tasks required fundamentally different logic—not just more “thinking.” I considered the agent to be saturated, so I merged into a team, and started **manual coding** to finish the tail.

##### Human-Injected Template Knowledge
While manual coding, we noticed **recurring task families**—for example, problems where the “natural” solution is DFS, BFS, flood-fill, or another grid search. From a code-golf perspective, these are painful: queues, boundary checks, four-direction neighbor sets, multi-line loops—the byte cost adds up fast. The good news is that many can be reframed in a **much simpler way** that compresses extremely well.

Here’s a detailed example showing how we (i) manually improved a working solution, (ii) abstracted that improvement into a **template**, and (iii) injected the template back into the agent so it could be reused elsewhere. This is a step outside the main agent loop, but the **template** is the key takeaway that powers future automated gains. We discovered several such templates throughout the competition, but here I’ll use Example 2 to demonstrate the process—this was my first manual breakthrough.


**Example task (ARC-AGI Example 2):**

The task: fill any **zero** (black) that is **enclosed** by **3s** (green border) with **4** (yellow). Zeros that can “see” the border (via a path of zeros) should stay zero; all others turn to 4.

A natural solution: BFS/flood-fill. Here’s a baseline solution:

```
def p(g):
 R=range;l=len(g);q=[]
 for i in R(l):q+=[(i,0),(i,l-1),(0,i),(l-1,i)]
 while q:
  i,j=q.pop()
  if -1<i<l and -1<j<l and not g[i][j]:
   g[i][j]=1;q+=[(i+x,j+y)for x,y in((1,0),(-1,0),(0,1),(0,-1))]
 for i in R(l):
  for j in R(l):
   if not g[i][j]:g[i][j]=4
   elif g[i][j]==1:g[i][j]=0
 return g
```

**What this 320-byte baseline does:**
It seeds a queue with border coordinates and flood-fills through zeros to mark everything **reachable from the border**. On the final scan:

* Zeros **never reached** by the flood-fill are **enclosed** → set to `4`
* Zeros **visited** by the flood-fill are temporarily marked → restore to `0`

This is robust but costly (in terms of byte): explicit queue, boundaries, four neighbors, multi-line loops.


##### The code-golf perspective: “rotate and erode”

Instead of growing a region from the border, we **prefill** and then **erode** exposure:

1. **Prefill:** Assume all zeros are enclosed and set them to `4`.
2. **Erode:** Any `4` touching the outside via a chain of zeros is reverted to `0`. We implement this by checking just the **left neighbor** and rotating the grid 90° each pass:
   * Pass 1: If the **left** neighbor is `0`, the current `4` becomes `0` (exposed from the west).
   * Rotate 90°, repeat (north); again (east); again (south).
   * After four rotations, all directions are covered. Only truly enclosed cells remain `4`.

This collapses BFS into a terse, single-expression update + rotate loop, which golfs extremely well.

**Our 104-byte solution for Example 2 (“rotate and scan”):**

```
p=lambda q,a=95:q*-a or p([*zip(*[map(lambda r,H:H*(H^4|r>0)or a//95*4,[0,*x],x)for x in q[::-1]])],a-1)
```

**How the one-liner works:**

* `a` is an iteration budget (multiple of 4 to fully sweep all directions, returning to original orientation).
* **First pass:** (`a//95 == 1`) prefill zeros to `4`.
* **Later passes:** For each cell, if `A==4` and left neighbor `L==0`, set `A=0` (exposed).
* `q[::-1]` + `zip(*)` rotates the grid 90°, so a single left-neighbor rule is applied in all directions over four passes.


##### From one-off insight to reusable template

We saved 200 bytes by reframing the problem. But: simply dropping this solution into the Evolutionary DB wasn’t enough, as such “logic leaps” can confuse the LLM when reused cold. So we started to **extract the pattern as a template**, to be injected into the prompt when a similar task appears.

**The reusable “rotate-and-update” template:**
Generalise the idea as a **rotate-and-update** loop: write a per-cell update that looks only “left”; rotation covers all directions. Still golf-friendly—no queues, single expression.

```python
# n = iteration counter (keep n a multiple of 4 so orientation resets)
p=lambda o,n=95:-n*o or p(
 [*zip(*(
   ((A # <— replace with your per-cell update
    for L,A in zip((0,*r),r))
   for r in o[::-1]
  ))],-~n)
```

Examples for the update line:

* `(A+(L==k)*(A<k))`  # bump A if left is k
* `(A,L)[cond]`       # choose L over A if condition met
* `(x(A),A)[flag]`    # one-off transform

For Example 2, the per-cell update:

* **First pass:** zeros become `4`
* **Subsequent passes:** if `A==4` and `L==0`, set `A=0` (erode)

This template typically saves **50–100 bytes** per program for these task families. Stacking such templates produces large aggregate gains.


**Why this matters for the agent:**
The point isn’t just manual golfing, it’s to **harvest a template**. Once crystallised, we inject the template into the agent’s system prompt. The **Evolutionary DB** then surfaces similar tasks, and the agent can reuse the motif automatically—turning a one-off breakthrough into a repeatable byte saver.



##### What didn’t work (and where we hand-golfed)

There was an ongoing [discussion](https://www.kaggle.com/competitions/google-code-golf-2025/discussion/608310#3291405) during the competition about what LLMs struggle with. Our experience mostly matched it:

* **Regex never stuck.** Even with templates and human-solved examples that used regex, the agent rarely synthesized a correct, byte-minimal regex. When regex was useful, we almost always had to hand-write it.
* **Bitwise ops worked perfectly.** When presented with multiple working solutions, the LLM reliably understood the logic and exploited bitwise operations wherever they applied.
* **Recursion was fine—with a template.** Once we built a solid template for recursion-related problems, the agent could use them reliably.
* **Large logic pivots needed scaffolding.** Whenever a breakthrough required a full reframing (like rotate-and-erode), the agent struggled, unless we packaged it as a reusable template and surfaced it via the Evolutionary DB.

---

## Conclusion

That’s my slice of the system: a lean agentic loop, held together by **Context Engineering** and an **Evolutionary DB**, plus a handful of **human-injected templates** that traveled well. In parallel, my teammates explored other directions—compressing functions with `zlib`, AST-aware normalization to remove redundant characters, running smaller models augmented with better golf idioms, and even golfing with Claude Code. It was three months of steady grind, but deeply rewarding—and I learned a lot.



