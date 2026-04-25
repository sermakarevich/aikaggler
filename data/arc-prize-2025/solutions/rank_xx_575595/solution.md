# ARC 2024 Solutions and Key Takeaways

- **Author:** Younus_Mohamed
- **Date:** 2025-04-29T18:04:47.743Z
- **Topic ID:** 575595
- **URL:** https://www.kaggle.com/competitions/arc-prize-2025/discussion/575595

**GitHub links found:**
- https://github.com/zoenguyenramirez/arc-prize-2024

---

**ARC Prize 2024 – [Competition Page](https://www.kaggle.com/competitions/arc-prize-2024)**  
First large-scale sequel to the 2020 ARC benchmark – 100 hidden “human-style” tasks.  

### **2ᵈ Place – “Omni-ARC TTT” – [Discussion](https://www.kaggle.com/competitions/arc-prize-2024/discussion/545671)**  @ironbar 
*Full, beautifully-illustrated paper 👉 [Solution Summary](https://ironbar.github.io/arc24/05_Solution_Summary/) — highly recommended!*  it will need a seperate discussion post on itself
1. Per-task **test-time training (TTT)** on LoRA-tuned **Qwen-0.5 B**; model both *draws* & *reasons*.  
2. AIRV loop (augment → infer → reverse → vote) beats rotations & colour permutations.  
3. Generates synthetic train grids to thicken sparse supervision.  
4. Falls back to 2020 C++ DSL solver for purely symbolic cases (+14 pts).  
5. Lightweight — runs in ≈9 h CPU-only if GPU unavailable.  
6. **Dev diary:** author logged 50 iterations — from few-shot → Omni-ARC (see paper).  

### **3ʳᵈ Place – “Guided Brute-Force” – [Notebook](https://www.kaggle.com/code/alijs1/arc-prize-2024-solution-3rd-place-score-40)**  @alijs1 
1. Library of 120 hand-coded **symmetry / flood-fill / pattern** functions.  
2. Grid-search picks first function that satisfies all train examples.  
3. Heuristics order functions by task metadata to save time.  
4. 700-line pure-Python script still scores 40 pts.  
5. Blended with icecuber’s 2020 C++ brute solver for the final push.  

### **4ᵗʰ Place – “Classical DSL ++” – [Discussion](https://www.kaggle.com/competitions/arc-prize-2024/discussion/550414) | [Notebook](https://www.kaggle.com/code/williamwu88/fork-of-small-sample-arc24-7d97ca)**  @williamwu88 
1. 2020 **DSL / DAG search** run deeper thanks to 30 GB RAM.  
2. Adds **CNN** & **decision-tree** modules for colour counts & masks.  
3. Heavy **data-augmentation** (rot/flip/diag/colour swap).  
4. Ensemble voting with “trust-icecuber-first” override.  
5. Probabilistic model sampling ∝ #unique solves; avoids over-fitting to any solver.  

### **5ᵗʰ Place – “PoohAI Mega-Ensemble” – [Discussion](https://www.kaggle.com/competitions/arc-prize-2024/discussion/550336) | [Notebook](https://www.kaggle.com/code/gromml/arc-prize-2024-poohai-solution)**  @greylord1996 @gromml @leadnvalidate @samalkubentayeva 
1. Merges **six 2020 solvers** (DSL, GA, decision-tree, …) plus their own iterative solver.  
2. **Post-filters** auto-repair GA errors (extra lines, wrong shapes).  
3. Brute-forces hidden task order (≤100 submissions + binary search).  
4. Strategy: “easier to delete bad answers than pick the best one” – keep top-2 attempts.  
5. Pure-Python orchestrator; C++ cores run in parallel via subprocess.

### **13ᵗʰ Place – “Scratch-built Transformers” – [Discussion](https://www.kaggle.com/competitions/arc-prize-2024/discussion/546302) | [Repo](https://github.com/zoenguyenramirez/arc-prize-2024)**  @zoenguyenramirez 
1. **19-token custom Transformer** (no external LLM); PyTorch 2.2 / CUDA 12.  
2. Novel **grid positional encoding** beats classical sinusoidal.  
3. **Active-Inference / reverse augmentation** loop (+27 pts).  
4. Ensemble of multiple checkpoints via majority vote → 31 pts private LB.  
5. Full training & submission scripts open-sourced for further scaling.

### 21st Place – **Re-color pre-processor (+2 pts over IceCuber) – [Short report](https://www.kaggle.com/competitions/arc-prize-2024/discussion/550209) ** @lyrialtus 
1. **28 % score** by adding one tiny NumPy function to IceCuber.  
2. `recolor_task()` **remaps colours** by pixel-count order when palette is irrelevant.  
3. Works only for **single-test-pair tasks**; skips multi-test ones.  
4. Re-colors both train & test grids, then restores IDs (‒10/+10 trick).  
5. Zero C++ edits; plugs into IceCuber as a Python pre-step.  
6. Insight: humans spot colour-irrelevance instantly → cheap gain.  
7. Author views it as representation learning hint rather than AGI progress.  

### 34th Place – **LLaMA 3.1 8B + ARC 2020 hybrid (27 pts) – [Silver-medal write-up](https://www.kaggle.com/competitions/arc-prize-2024/discussion/545886) ** @crsuthikshnkumar 
1. **Unsloth-finetuned LLaMA 3.1 8B**, temperature-swept prompts.  
2. Generates **multiple candidate grids** per task; picks best via heuristics.  
3. **Fallback fusion** with 2020 winning solver for coverage.  
4. Prompt injects *lateral-thinking* cues & rule-learning reminders.  
5. Token / top-k / temp schedules tuned over many small runs.  
6. Runs inside 12 h budget on modest compute (single 8 B model).  
7. Public notebook & paper: explores how far a single LLM + classic solver can go toward AGI.  

### **Key 2025 vs 2024 differences** 
- Only one submission per task (2024 had two chances).  
- Brand-new task set – 2024 heuristics may transfer poorly.  
- Fresh task set (no overlap with ’20/’24). **Still 12 h** runtime & 30 GB RAM. 
- We now get 60 hours per week of GPU usage time in Kaggle instead of the usual 30. (Provided we have a Colab Pro or Pro+ account)
- We also get access to more powerful L4x4 machines! These machines offer 96GB of GPU memory enabling submissions with much larger models.

Any more inputs from you guys would be really helpful to the community @ironbar @alijs1 @williamwu88 @greylord1996 @gromml @leadnvalidate @samalkubentayeva @zoenguyenramirez @lyrialtus @crsuthikshnkumar 

Happy ARC hacking! May your grids be ever symmetric.!

And Happy Kaggling..!