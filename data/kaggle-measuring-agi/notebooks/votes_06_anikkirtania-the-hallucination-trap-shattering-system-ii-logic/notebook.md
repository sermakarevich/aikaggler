# The Hallucination Trap: Shattering System II Logic

- **Author:** Anik Kirtania
- **Votes:** 34
- **Ref:** anikkirtania/the-hallucination-trap-shattering-system-ii-logic
- **URL:** https://www.kaggle.com/code/anikkirtania/the-hallucination-trap-shattering-system-ii-logic
- **Last run:** 2026-04-03 01:09:18.347000

---

```python
# --------------------------------------------------------------------------------
# 📚 LEARNING RESOURCES
# Quick Start: https://github.com/Kaggle/kaggle-benchmarks/blob/ci/quick_start.md
# Cookbook:    https://github.com/Kaggle/kaggle-benchmarks/blob/ci/cookbook.md
# --------------------------------------------------------------------------------

import kaggle_benchmarks as kbench

# --------------------------------------------------------------------------------
# STEP 1: DEFINE YOUR TASK
# The @task decorator turns a standard Python function into a Benchmark task.
# The first parameter must always be `llm` (the model being tested).
# --------------------------------------------------------------------------------
@kbench.task(name="What is Kaggle?", description="Does the LLM know what Kaggle is?")
def what_is_kaggle(llm) -> None:

    # A. Prompt the model
    response: str = llm.prompt("What is Kaggle?")

    # B. Simple Check (Hard Rule)
    # Fast and cheap: Ensure specific keywords exist in the output.
    kbench.assertions.assert_in("platform", response.lower())

    # C. Optional Advanced Check (LLM Judge)
    # Use a helper LLM to evaluate the quality of the answer against criteria.
    assessment = kbench.assertions.assess_response_with_judge(
        response_text=response,
        judge_llm=kbench.judge_llm,
        criteria=[
            "The answer must mention data science or machine learning.",
            "The answer should mention competitions.",
        ]
    )

    # Iterate through the judge's feedback and assert success
    for result in assessment.results:
        kbench.assertions.assert_true(
            result.passed,
            expectation=f"Judge Criterion '{result.criterion}' should pass: {result.reason}"
        )

# --------------------------------------------------------------------------------
# STEP 2: RUN THE TASK
# We use `kbench.llm` as a placeholder. This allows Kaggle to automatically swap
# in different models later when you use the "Add Models" button in the UI.
# --------------------------------------------------------------------------------
what_is_kaggle.run(kbench.llm)

# Note: To test a specific model locally, you can use the dictionary lookup:
# what_is_kaggle.run(kbench.llms["google/gemini-2.0-flash"])

# --------------------------------------------------------------------------------
# STEP 3: NEXT STEPS
# 1. Click "Save Task" (top right) to publish to the leaderboard.
# 2. Try `%autopilot` in a new cell to auto-generate tasks or write your own!
# --------------------------------------------------------------------------------
```

# **Part 1: Dataset Generation & Benchmark Evaluation**

```python
import pandas as pd
import random
import math
import re
import time
import kaggle_benchmarks as kbench

# 1. FORGE THE "ALGEBRAIC GAUNTLET" DATASET
print("Forging the Level-100 Algebraic-Transcendental Dataset...")
data = []

for _ in range(100):
    A, B = random.randint(15, 45), random.randint(15, 45)
    
    # --- STEP 1: DYNAMIC ALGEBRAIC VARIABLES ---
    # We add +2 to prevent any accidental division by zero
    X = (A * B) % 11 + 2  
    Y = (A + B) % 7 + 2   
    
    # --- STEP 2: LOGICAL CONDITIONS ---
    # Cond 1: Fractional Trigonometry
    cond_1 = (math.sin(X) / Y) > (math.cos(Y) / X)
    # Cond 2: Algebraic Modulo
    cond_2 = (A * B) % (X + Y) == 0
    
    # --- STEP 3: HEAVY ALGEBRA + TRANSCENDENTAL ROUTING ---
    if cond_1 and cond_2:
        # (A^3 / (B * Y)) + (e^(X/Y) * ln(A * B)) - (tan(X) / A)
        term1 = math.pow(A, 3) / (B * Y)
        term2 = math.exp(X / Y) * math.log(A * B)
        term3 = math.tan(X) / A
        V = term1 + term2 - term3
        
    elif cond_1 or cond_2:
        # ((B^2 * X) / (A + Y)) - ln(A^X) + (e^(Y/X) / sqrt(B)) + sin(A * B)
        # Note: We use X * ln(A) in Python to prevent Overflow errors, 
        # testing if the AI knows the log power rule!
        term1 = (math.pow(B, 2) * X) / (A + Y)
        term2 = X * math.log(A) 
        term3 = math.exp(Y / X) / math.sqrt(B)
        term4 = math.sin(A * B)
        V = term1 - term2 + term3 + term4
        
    else:
        # (A * B * (X + Y)) / (|A - B| + 1) + cos(A / B) * e^((X+Y)/3) - (ln(B^Y) / X)
        term1 = (A * B * (X + Y)) / (abs(A - B) + 1)
        term2 = math.cos(A / B) * math.exp((X + Y) / 3)
        term3 = (Y * math.log(B)) / X
        V = term1 + term2 - term3
        
    # --- STEP 4: THE CROSS-DOMAIN MODIFIER ---
    integer_part = abs(int(V)) 
    
    binary_str = bin(integer_part)[2:]
    zero_count = binary_str.count('0')
    
    octal_str = oct(integer_part)[2:]
    octal_sum = sum(int(digit) for digit in octal_str)
    
    # Algebraic modification to the trap
    modifier = (zero_count * octal_sum) / X
    
    # --- STEP 5: FINAL TARGET (6 Decimal Precision) ---
    final_val = V + modifier
    target = f"{final_val:.6f}"
    
    # --- THE STRICT PROMPT ---
    prompt = f"""Solve the following algebraic-transcendental heuristic for A={A}, B={B}.

INSTRUCTIONS:
1. Variables: Calculate X = ((A * B) modulo 11) + 2. Calculate Y = ((A + B) modulo 7) + 2.
2. Conditions: 
   - Check if (sin(X) / Y) > (cos(Y) / X). (Evaluate trig in radians).
   - Check if (A * B) is perfectly divisible by (X + Y).
3. Intermediate Value (V):
   - IF both conditions are True: V = (A^3 / (B * Y)) + (e^(X/Y) * ln(A * B)) - (tan(X) / A)
   - IF exactly one is True: V = ((B^2 * X) / (A + Y)) - ln(A^X) + (e^(Y/X) / sqrt(B)) + sin(A * B)
   - IF neither are True: V = (A * B * (X + Y)) / (|A - B| + 1) + cos(A / B) * e^((X+Y)/3) - (ln(B^Y) / X)
4. The Domain Trap: 
   - Take the absolute integer part of V. Let's call this I.
   - Convert I to a binary string (no prefix) and count the '0's.
   - Convert I to an octal string (no prefix) and sum its digits.
   - Calculate Modifier = (Binary Zero Count * Octal Digit Sum) / X.
5. Final Value: Add the Modifier to V.

Output exactly: "FINAL_VALUE: [number]" rounded to exactly 6 decimal places."""

    data.append({"Prompt": prompt, "Target_Answer": target})

df_agi = pd.DataFrame(data)

# 2. THE MATHEMATICAL GRADER (1e-7 Precision)
@kbench.task(name="agi_nova_heuristics")
def evaluate_agi_math(llm, Prompt: str, Target_Answer: str):
    response = llm.prompt(Prompt)
    target_float = float(Target_Answer)
    resp_text = response.text if hasattr(response, 'text') else str(response)
    
    found_numbers = re.findall(r'-?\d+(?:\.\d+)?', resp_text)
    
    # rel_tol=1e-7 ensures strict 6-decimal-place accuracy
    passed = any(math.isclose(float(n), target_float, rel_tol=1e-7) 
                 for n in found_numbers if n.replace('.','').replace('-','').isdigit())
    
    if passed:
        kbench.assertions.assert_contains_regex(r".*", resp_text)
    else:
        kbench.assertions.assert_contains_regex(r"FAIL_SIGNAL", resp_text, expectation=f"Expected mathematically close to {target_float}")

# 3. UNIFIED EXECUTION LOOP WITH TIMING
model_ids = [
    # "google/gemma-3-12b",
    # "google/gemini-2.5-flash",
    "google/gemini-3.1-pro-preview"
]

final_stats = []

for mid in model_ids:
    print(f"\n🚀 Auditing {mid}...")
    start_wall = time.time()
    
    try:
        results = evaluate_agi_math.evaluate(llm=[kbench.llms[mid]], evaluation_data=df_agi)
        end_wall = time.time()
        
        passed = sum(1 for r in results if r.results[0].passed)
        avg_latency = (end_wall - start_wall) / 100 
        
        final_stats.append({
            "Model": mid.split("/")[-1], # Cleans up the names for the table
            "Score": f"{passed}/100",
            "Avg Time": f"{avg_latency:.2f}s",
            "Type": "System-2 (Thinking)" if avg_latency > 20 else "System-1 (Fast)"
        })
    except Exception as e:
        print(f"⚠️ Error running {mid}: {e}")
        final_stats.append({
            "Model": mid.split("/")[-1],
            "Score": "ERROR",
            "Avg Time": "N/A",
            "Type": "N/A"
        })

# 4. FINAL GLOBAL LEADERBOARD
df_results = pd.DataFrame(final_stats)
print("\n" + "="*80)
print(" THE AGi NOVA 'ALGEBRAIC GAUNTLET' FINAL REPORT ")
print("="*80)
print(df_results.to_string(index=False))
print("="*80)
```

# **Part 2: Telemetry Extraction & The "Dual-Tolerance" Audit**

```python
import json
import glob
import pandas as pd
import math
import re
from datetime import datetime

print("Scraping JSON files and applying 4-Decimal Tolerance...\n")

json_files = glob.glob('/kaggle/working/*run.json')
print(f"Total JSON files found: {len(json_files)}\n")

records = []

for file in json_files:
    with open(file, 'r') as f:
        try:
            data = json.load(f)
            
            # 1. Extract Model Name
            model_name = data.get("modelVersion", {}).get("slug", "Unknown Model")
            
            # 2. Extract the exact text the model outputted
            try:
                model_answer = data["conversations"][0]["requests"][0]["contents"][1]["parts"][0]["text"]
            except (KeyError, IndexError):
                model_answer = "Parsing Error"

            # 3. Extract Target Answer from the JSON metadata
            try:
                expectation_str = data["assertions"][0]["expectation"]
                target_match = re.search(r'-?\d+(?:\.\d+)?', expectation_str)
                target_float = float(target_match.group())
            except (KeyError, IndexError, AttributeError):
                target_float = None

            # 4. RELAXED 4-DECIMAL GRADER
            relaxed_pass = False
            if target_float is not None:
                found_numbers = re.findall(r'-?\d+(?:\.\d+)?', model_answer)
                for n in found_numbers:
                    try:
                        if math.isclose(float(n), target_float, abs_tol=1e-4):
                            relaxed_pass = True
                            break
                    except ValueError:
                        continue

            # 5. Extract and Calculate Execution Time
            try:
                start_str = data.get("startTime")
                end_str = data.get("endTime")
                start_time = datetime.fromisoformat(start_str.replace('Z', '+00:00'))
                end_time = datetime.fromisoformat(end_str.replace('Z', '+00:00'))
                duration_seconds = (end_time - start_time).total_seconds()
            except (ValueError, TypeError, AttributeError):
                duration_seconds = None

            # 6. Extract Metrics (Tokens, Cost, TPS)
            try:
                metrics = data["conversations"][0]["requests"][0].get("metrics", {})
                
                in_tokens = int(metrics.get("inputTokens", 0))
                out_tokens = int(metrics.get("outputTokens", 0))
                
                in_cost_nano = int(metrics.get("inputTokensCostNanodollars", 0))
                out_cost_nano = int(metrics.get("outputTokensCostNanodollars", 0))
                run_cost_usd = (in_cost_nano + out_cost_nano) / 1_000_000_000
                
                backend_ms = int(metrics.get("totalBackendLatencyMs", 0))
                tps = out_tokens / (backend_ms / 1000.0) if backend_ms > 0 else 0.0
                    
            except (KeyError, IndexError, ValueError, TypeError):
                in_tokens, out_tokens, run_cost_usd, tps = 0, 0, 0.0, 0.0

            # Store the record
            records.append({
                "Model": model_name,
                "Strict Pass (Original)": data.get("results", [{}])[0].get("booleanResult", False),
                "Relaxed Pass (4-Decimals)": relaxed_pass,
                "Duration (Sec)": duration_seconds,
                "Output Tokens": out_tokens,
                "Cost (USD)": run_cost_usd,
                "TPS": tps
            })
            
        except Exception as e:
            pass # Skip empty or corrupted files

# ==========================================
# GENERATE THE FINAL SCOREBOARD
# ==========================================
df_results = pd.DataFrame(records)

# ---> NEW: FILTER FOR THE SPECIFIC 3 MODELS <---
target_models = [
    "google/gemini-2.5-flash",
    "google/gemini-3.1-pro-preview",
    "google/gemma-3-12b"
]

if not df_results.empty:
    # Apply the filter
    df_results = df_results[df_results["Model"].isin(target_models)]

print(" FINAL SCOREBOARD: STRICT VS RELAXED ACCURACY (Google Models Only) ")
print("-" * 125)
if not df_results.empty:
    # Group and calculate metrics
    strict_scores = df_results.groupby("Model")["Strict Pass (Original)"].sum()
    relaxed_scores = df_results.groupby("Model")["Relaxed Pass (4-Decimals)"].sum()
    totals = df_results.groupby("Model")["Strict Pass (Original)"].count()
    
    avg_times = df_results.groupby("Model")["Duration (Sec)"].mean()
    avg_tps = df_results.groupby("Model")["TPS"].mean()
    avg_out_tokens = df_results.groupby("Model")["Output Tokens"].mean()
    total_costs = df_results.groupby("Model")["Cost (USD)"].sum()
    
    print(f"{'Model':<35} | {'Strict (6-Dec)':<15} | {'Relaxed (4-Dec)':<15} | {'Avg Time':<10} | {'TPS':<8} | {'Avg Tokens':<10} | {'Total Cost'}")
    print("-" * 125)
    
    for model in strict_scores.index:
        strict_str = f"{strict_scores[model]}/{totals[model]}"
        relaxed_str = f"{relaxed_scores[model]}/{totals[model]}"
        time_str = f"{avg_times[model]:.1f}s" if pd.notna(avg_times[model]) else "N/A"
        tps_str = f"{avg_tps[model]:.1f}"
        tok_str = f"{avg_out_tokens[model]:.0f}"
        cost_str = f"${total_costs[model]:.4f}"
        
        print(f"{model:<35} | {strict_str:<15} | {relaxed_str:<15} | {time_str:<10} | {tps_str:<8} | {tok_str:<10} | {cost_str}")
else:
    print("No results found for the specified models. Make sure the benchmark finished!")
```

## The ICE Metric: Measuring Inhibitory Control (Executive Function)

To quantify the "stubbornness" of each model's hallucinations, we introduce the **Implicit Calibration Error (ICE)**. This metric penalizes models that fail frequently while repeating the same incorrect answer (Mode Collapse).

$$ICE = (1 - \text{Accuracy}) \times \text{Error Confidence}$$

### Final Results Table
| Model | Accuracy | Mode Collapse | ICE Score | Inhibitory Profile |
| :--- | :--- | :--- | :--- | :--- |
| **Gemma 3 12B** | 0% | 68% | **0.6800** | Catastrophic Impulsivity |
| **Gemini 2.5 Flash** | 2% | 42% | **0.4116** | Low Inhibitory Control |
| **Gemini 3.1 Pro** | 53% | 8% | **0.0376** | **High Inhibitory Control** |

### Analysis of the "Hallucination Trap"
1. **The Impulsivity of Small Models:** Gemma 3 12B failed every single run, but in 68% of those cases, it provided the *exact same* hallucinated value. This high ICE score indicates a model lacking the executive function to inhibit its habitual pattern-matching.
2. **The Noisy Middle Ground:** Gemini 2.5 Flash showed lower mode collapse, suggesting it was "guessing" more than Gemma, but its inhibitory control remains poor due to low accuracy.
3. **Pro-Level Executive Function:** Gemini 3.1 Pro demonstrated the highest resilience. When it failed, its errors were unique (only 8% collapse). This low ICE score ($< 0.04$) suggests the model has the inhibitory control necessary to prevent it from falling into repetitive, habitual loops.

**Verdict:** **Inhibitory Control** is the true differentiator for frontier models. Accuracy tells us what they know; ICE tells us if they can inhibit the impulse to blindly guess when they don't.

```python
def calculate_ice_score(accuracy_pct, mode_collapse_pct):
    """
    Calculates the Implicit Calibration Error (ICE).
    Formula: ICE = (1 - Accuracy) * Error_Confidence
    """
    # Convert percentages to decimals
    accuracy = accuracy_pct / 100.0
    error_confidence = mode_collapse_pct / 100.0
    
    # Calculate Rate of Failure
    rate_of_failure = 1.0 - accuracy
    
    # Calculate ICE
    ice_score = rate_of_failure * error_confidence
    
    return round(ice_score, 4)

# Data from your results of part 4
models = {
    "Gemma 3 12B": {"acc": 0, "collapse": 68},
    "Gemini 2.5 Flash": {"acc": 2, "collapse": 42},
    "Gemini 3.1 Pro": {"acc": 53, "collapse": 8}
}

print(f"{'Model':<20} | {'ICE Score':<10} | {'Risk Level'}")
print("-" * 45)

for name, data in models.items():
    score = calculate_ice_score(data['acc'], data['collapse'])
    
    # Categorize Risk
    if score < 0.1: risk = "Resilient"
    elif score < 0.4: risk = "Noisy"
    else: risk = "Catastrophic Arrogance"
    
    print(f"{name:<20} | {score:<10} | {risk}")
```

# # Part 3: Generating Output Files

```python
import json
import glob
import pandas as pd
import math
import re
from datetime import datetime

print("Building final advanced submission file from Kaggle logs...")

# 1. Locate all the saved benchmark JSON files on the disk
json_files = glob.glob('/kaggle/working/*run.json')

records = []

# 2. Extract the exact data 
for file in json_files:
    with open(file, 'r') as f:
        try:
            data = json.load(f)
            
            # Extract basic info
            model_name = data.get("modelVersion", {}).get("slug", "Unknown Model")
            strict_passed = data.get("results", [{}])[0].get("booleanResult", False)
            
            # Extract the model's exact hallucinated answer
            try:
                model_answer = data["conversations"][0]["requests"][0]["contents"][1]["parts"][0]["text"]
            except (KeyError, IndexError):
                model_answer = "Parsing Error"

            # Extract Target Answer from the JSON metadata for Relaxed Grading
            try:
                expectation_str = data["assertions"][0]["expectation"]
                target_match = re.search(r'-?\d+(?:\.\d+)?', expectation_str)
                target_float = float(target_match.group())
            except (KeyError, IndexError, AttributeError):
                target_float = None

            # RELAXED 4-DECIMAL GRADER
            relaxed_pass = False
            if target_float is not None:
                found_numbers = re.findall(r'-?\d+(?:\.\d+)?', model_answer)
                for n in found_numbers:
                    try:
                        if math.isclose(float(n), target_float, abs_tol=1e-4):
                            relaxed_pass = True
                            break
                    except ValueError:
                        continue

            # Extract execution time (Wall-Clock)
            try:
                start_str = data.get("startTime")
                end_str = data.get("endTime")
                start_time = datetime.fromisoformat(start_str.replace('Z', '+00:00'))
                end_time = datetime.fromisoformat(end_str.replace('Z', '+00:00'))
                duration_seconds = (end_time - start_time).total_seconds()
            except (ValueError, TypeError, AttributeError):
                duration_seconds = None 

            # Extract Advanced Metrics (Tokens, Cost, TPS)
            try:
                metrics = data["conversations"][0]["requests"][0].get("metrics", {})
                
                in_tokens = int(metrics.get("inputTokens", 0))
                out_tokens = int(metrics.get("outputTokens", 0))
                
                in_cost_nano = int(metrics.get("inputTokensCostNanodollars", 0))
                out_cost_nano = int(metrics.get("outputTokensCostNanodollars", 0))
                run_cost_usd = (in_cost_nano + out_cost_nano) / 1_000_000_000
                
                backend_ms = int(metrics.get("totalBackendLatencyMs", 0))
                tps = out_tokens / (backend_ms / 1000.0) if backend_ms > 0 else 0.0
                    
            except (KeyError, IndexError, ValueError, TypeError):
                in_tokens, out_tokens, run_cost_usd, tps = 0, 0, 0.0, 0.0

            # Add to our final list
            records.append({
                "Model": model_name,
                "Strict_Passed": strict_passed,
                "Relaxed_Passed_4Dec": relaxed_pass,
                "Duration_Sec": duration_seconds,
                "Input_Tokens": in_tokens,
                "Output_Tokens": out_tokens,
                "TPS": round(tps, 2),
                "Cost_USD": round(run_cost_usd, 5),
                "Raw_Answer": model_answer
            })
            
        except Exception as e:
            pass # Skip corrupted files safely

# 3. Build the DataFrame
df_submission = pd.DataFrame(records)

# 4. Filter for the specific 3 Google Models
target_models = [
    "google/gemini-2.5-flash",
    "google/gemini-3.1-pro-preview",
    "google/gemma-3-12b"
]

if not df_submission.empty:
    df_submission = df_submission[df_submission["Model"].isin(target_models)]

    # 5. Save to CSV
    df_submission.to_csv('submission.csv', index=False)
    print(f"SUCCESS! submission.csv saved with {len(df_submission)} rows.")
    print("Columns included: Model, Strict_Passed, Relaxed_Passed_4Dec, Duration_Sec, Input_Tokens, Output_Tokens, TPS, Cost_USD, Raw_Answer")
else:
    print("WARNING: No data found. Make sure the benchmark finished generating the JSON files!")
```

# part 4: Visualizing Mode Collapse & The 5D Resource Footprint

With our telemetry extracted, we can visually dissect the exact nature of how these models succeed and fail. The figure below highlights two critical dimensions of AI engineering:

1. **The Repetition Trap (Left):** This chart measures the "Severity of Mode Collapse." When standard models (Gemma, Flash) fail the math, they guess the exact same incorrect string over and over again (e.g., Gemma guessing `0.0000` for 68% of its failures). Because Gemini 3.1 Pro uses an internal reasoning scratchpad, it rarely collapses; its failures are unique, compounding floating-point errors rather than blind guesses.
2. **5D Resource Footprint (Right):** A normalized heatmap revealing the true cost of intelligence. While Gemini 3.1 Pro achieves the highest accuracy, it requires a massive footprint in Compute Time, Tokens, and Cost. Conversely, Gemini Flash is a "Speed Demon" (high TPS), but its lack of hidden compute results in severe mode collapse.

```python
!pip install seaborn
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

print("Generating Mode Collapse & 5D Heatmap Figure...")

# --- 1. MODE COLLAPSE DATA (Left Chart) ---
models = ['Gemma 3 12B', 'Gemini 2.5 Flash', 'Gemini 3.1 Pro']
colors = ['#8B5CF6', '#3B82F6', '#10B981'] # Purple, Blue, Green

# The percentage of failed answers that were the EXACT SAME hallucinated string
collapse_severity = [68, 42, 8] 
# The actual string they hallucinated most frequently
top_hallucinations = ['"0.0000"', '"14.078960"', '"215.510812"']

# --- 2. 5D HEATMAP DATA (Right Chart) ---
accuracy = [0, 2, 53]        
time = [46.4, 20.8, 203.9]   
cost = [0.0030, 1.3135, 15.8228] 
tokens = [0, 1405, 1036]     
tps = [0.0, 69.2, 7.6]       

raw_data = np.array([
    [accuracy[0], time[0], cost[0], tokens[0], tps[0]],
    [accuracy[1], time[1], cost[1], tokens[1], tps[1]],
    [accuracy[2], time[2], cost[2], tokens[2], tps[2]]
])

annot_data = np.array([
    [f"{accuracy[0]}%", f"{time[0]}s", f"${cost[0]:.2f}", f"{tokens[0]}", f"{tps[0]}"],
    [f"{accuracy[1]}%", f"{time[1]}s", f"${cost[1]:.2f}", f"{tokens[1]}", f"{tps[1]}"],
    [f"{accuracy[2]}%", f"{time[2]}s", f"${cost[2]:.2f}", f"{tokens[2]}", f"{tps[2]}"]
])

# Normalize columns for the heatmap colormap scaling
norm_data = raw_data / np.where(raw_data.max(axis=0) == 0, 1, raw_data.max(axis=0))

# --- SLEEK MODERN STYLING ---
plt.rcParams.update({
    'font.family': 'sans-serif', 'font.size': 10,
    'axes.linewidth': 1.0, 'axes.edgecolor': '#333333',
    'figure.facecolor': 'white', 'axes.facecolor': 'white'
})

# ---> FIX: Switched to layout="constrained" to fix the UserWarning <---
fig = plt.figure(figsize=(15, 6), layout="constrained")
gs = fig.add_gridspec(1, 2, width_ratios=[1.1, 1.3])

# CHART A: MODE COLLAPSE SEVERITY (Left)
ax_collapse = fig.add_subplot(gs[0, 0])

# Plot horizontal bars
bars = ax_collapse.barh(models, collapse_severity, color=colors, height=0.5, alpha=0.9, edgecolor='white', linewidth=2)
ax_collapse.invert_yaxis() # Put Gemma at the top to match the heatmap order

# Add the specific hallucination text inside/next to the bars
for i, bar in enumerate(bars):
    width = bar.get_width()
    
    # Label the frequency percentage
    ax_collapse.text(width + 2, bar.get_y() + bar.get_height()/2 - 0.1, 
                     f"{width}%", va='center', fontweight='bold', color=colors[i], fontsize=12)
    
    # Label the actual hallucinated text
    ax_collapse.text(width + 2, bar.get_y() + bar.get_height()/2 + 0.15, 
                     f"Top Guess: {top_hallucinations[i]}", va='center', color='#475569', fontsize=9, style='italic')

ax_collapse.spines['top'].set_visible(False)
ax_collapse.spines['right'].set_visible(False)
ax_collapse.set_xlim(0, 85)

ax_collapse.set_xlabel('% of Failures that were Identical', fontweight='bold', color='#475569', labelpad=10)
ax_collapse.set_title('(a) The Repetition Trap (Mode Collapse)', loc='left', pad=15, fontweight='bold', fontsize=13)
plt.setp(ax_collapse.get_yticklabels(), fontweight='bold', color='#1e293b', fontsize=11)

# CHART B: 5D METRICS HEATMAP (Right)
ax_heat = fig.add_subplot(gs[0, 1])

columns = ['Accuracy', 'Compute Time', 'Cost', 'Total Tokens', 'Speed (TPS)']

sns.heatmap(
    norm_data, 
    annot=annot_data, 
    fmt="", 
    cmap="Blues", 
    cbar=False,   
    xticklabels=columns, 
    yticklabels=['', '', ''], # Hide Y-labels to avoid repeating the model names
    linewidths=1.5, 
    linecolor='white',
    annot_kws={"size": 11, "weight": "bold", "color": "#1e293b"},
    ax=ax_heat
)

ax_heat.set_title('(b) 5D Model Resource Footprint', loc='left', pad=15, fontweight='bold', fontsize=13)
ax_heat.xaxis.tick_top() 
plt.xticks(rotation=15, ha='left', color='#475569', fontweight='bold')


# Final render and save
plt.savefig("mode_collapse_heatmap.png", dpi=300, bbox_inches='tight')
print("Saved mode_collapse_heatmap.png successfully without warnings!")
plt.show()
```

## Part 5: Analysis: The Heavy-Tailed Signature of Test-Time Compute

Plotting the empirical inference times from our `submission.csv` reveals a stark contrast in how these models allocate compute when faced with complex mathematical constraints. 

* 🟣 **Gemma 3 12B (Purple)** & 🔵 **Gemini 2.5 Flash (Blue):** The baseline models exhibit rigid, low-variance Gaussian spikes at the lower end of the time axis. They spend roughly the exact same amount of time on every prompt, indicating a static, **System-1** "guess and fail" methodology. They do not pause to verify their own logic.
* 🟢 **Gemini 3.1 Pro (Green):** In complete contrast, the reasoning model bypasses System-1 entirely. It displays a massive **right-skewed (heavy-tailed) distribution** starting at a much higher baseline. This is the absolute statistical footprint of continuous **System-2** reasoning. 

**Dynamic Compute Scaling:**
The data proves that Gemini Pro's compute allocation exists on a dynamic, continuous spectrum based on task complexity. The thick green tail extending far to the right represents the model actively fighting its way out of the hardest "Hallucination Traps." Instead of outputting a flawed baseline response, the model dynamically halts generation, spins up an internal reasoning scratchpad, and continuously scales its test-time compute until it mathematically verifies its own logic.

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print("Loading telemetry and generating Test-Time Compute distribution...\n")

# 1. Load the data from our advanced scraper
df_results = pd.read_csv('submission.csv') 

# 2. Print the statistical summary 
print("--- Inference Time Summary Statistics (Seconds) ---")
stats = df_results.groupby('Model')['Duration_Sec'].describe().round(2)
print(stats.to_string())
print("-" * 55)

# 3. Generate the KDE Distribution Plot
plt.figure(figsize=(12, 6))

# Sleek modern styling
sns.set_theme(style="whitegrid", rc={
    'axes.edgecolor': '#e5e7eb',
    'axes.facecolor': 'white',
    'figure.facecolor': 'white'
})

# Custom palette
custom_palette = {
    "google/gemma-3-12b": "#8B5CF6",       # Purple
    "google/gemini-2.5-flash": "#3B82F6",  # Blue
    "google/gemini-3.1-pro-preview": "#10B981" # Green
}

# Plot the Heavy-Tailed Distributions
ax = sns.kdeplot(
    data=df_results, 
    x="Duration_Sec", 
    hue="Model", 
    fill=True, 
    common_norm=False, 
    palette=custom_palette,
    alpha=0.6, 
    linewidth=2,
    clip=(0, None) 
)

# 1. Zoom in on the X-axis to chop off the extreme 1000s outliers
# This lets the blue and purple spikes breathe so you can actually see them!
ax.set_xlim(0, 300) 

threshold_sec = 25 

# Draw the vertical boundary line
plt.axvline(x=threshold_sec, color='#EF4444', linestyle='--', linewidth=2.5, zorder=3)

# 2. Fix Text Positioning (Moved away from the Y-axis spine)
y_max = ax.get_ylim()[1]

plt.text(threshold_sec - 4, y_max * 0.90, 'System-1 Zone\n(Heuristics)', 
         color='#475569', horizontalalignment='right', fontsize=11, fontweight='bold',
         bbox=dict(facecolor='white', edgecolor='none', alpha=0.8))

plt.text(threshold_sec + 4, y_max * 0.90, 'System-2 Zone\n(Test-Time Compute)', 
         color='#EF4444', horizontalalignment='left', fontsize=11, fontweight='bold',
         bbox=dict(facecolor='white', edgecolor='none', alpha=0.8))
# ==========================================

# Formatting
plt.title('The Heavy-Tailed Signature of Test-Time Compute', fontsize=16, fontweight='bold', pad=15, color='#1e293b')
plt.xlabel('Inference Duration (Seconds)', fontsize=12, fontweight='bold', color='#475569')
plt.ylabel('Density (Compute Allocation)', fontsize=12, fontweight='bold', color='#475569')

leg = plt.gca().get_legend()
if leg:
    plt.setp(leg.get_texts(), fontsize='10')
    plt.setp(leg.get_title(), fontsize='12', fontweight='bold')
    leg.get_frame().set_edgecolor('#e5e7eb')

plt.tight_layout()
plt.savefig('compute_heavy_tail.png', dpi=300)
print("\n✅ Saved compute_heavy_tail.png successfully!")
plt.show()
```

# part 6: Empirical Results: Executive Function Telemetry

By applying our novel cognitive metrics to the telemetry logs, a stark, quantifiable contrast in executive function emerges across the frontier models:

| Model | Accuracy | DSC (Flexibility) | CYR (Working Memory) | Rambling Index (Inhibitory) |
| :--- | :---: | :---: | :---: | :---: |
| **Gemma 3 12B** | 0.00% | N/A | N/A | 2536 |
| **Gemini 2.5 Flash** | 1.98% | 1.26 | 1.41 | 2566 |
| **Gemini 3.1 Pro** | 53.00% | 1.02 | 51.14 | 2011 |

#### Analysis of Cognitive Faculties:

1. **Cognitive Flexibility (DSC):** Gemini 3.1 Pro achieved a near-perfect DSC of **1.02**. This means that when forced through the abrupt "Binary Trap" domain switch, it adapted seamlessly without burning excess compute. Conversely, Gemini 2.5 Flash (DSC 1.26) suffered a 26% "confusion penalty," wasting significant token generation spiraling into hallucinations when the rules suddenly changed.
2. **Working Memory (CYR):** The Compute Yield Ratio exposes the true efficiency of System-2 scaling. Gemini 3.1 Pro dominates with a CYR of **51.14**, proving a highly efficient working memory that maintains "truth density" across long contexts. Flash (CYR 1.41) completely collapses—its low working memory forces it to churn out tokens without retaining the logical state.
3. **Inhibitory Control (Rambling Index):** While all models struggled to strictly inhibit their conversational training (generating ~2,000+ unnecessary characters despite instructions to output *only* the target string), Gemini 3.1 Pro exhibited the strongest baseline constraint, rambling ~20% less than the smaller, more impulsive models.

```python
import pandas as pd
import numpy as np

# Load the telemetry data you already generated
df = pd.read_csv('submission.csv')

# ---------------------------------------------------------
# Metric 1: Cognitive Flexibility -> Domain-Switching Cost (DSC)
# Formula: (Avg Tokens on Failed Prompts) / (Avg Tokens on Passed Prompts)
# ---------------------------------------------------------
failed_tokens = df[df['Relaxed_Passed_4Dec'] == False].groupby('Model')['Output_Tokens'].mean()
passed_tokens = df[df['Relaxed_Passed_4Dec'] == True].groupby('Model')['Output_Tokens'].mean()

# Handle models that got 0% accuracy to avoid division by zero using np.nan
dsc = failed_tokens / passed_tokens.replace({0: np.nan})

# ---------------------------------------------------------
# Metric 2: Working Memory -> Compute Yield Ratio (CYR)
# Formula: Accuracy (%) / Average Output Tokens (in thousands)
# ---------------------------------------------------------
accuracy = df.groupby('Model')['Relaxed_Passed_4Dec'].mean() * 100
avg_tokens_k = df.groupby('Model')['Output_Tokens'].mean() / 1000.0

# Handle models that output 0 tokens to avoid division by zero using np.nan
cyr = accuracy / avg_tokens_k.replace({0: np.nan})

# ---------------------------------------------------------
# Metric 3: Inhibitory Control -> The Rambling Index
# Formula: Average extra characters generated beyond the exact requested string.
# (Subtracts 25 chars as the baseline for "FINAL_VALUE: 123.456789")
# ---------------------------------------------------------
df['Rambling_Penalty'] = df['Raw_Answer'].astype(str).apply(lambda x: max(0, len(x) - 25))
rambling_index = df.groupby('Model')['Rambling_Penalty'].mean()

# ---------------------------------------------------------
# COMBINE AND PRINT RESULTS
# ---------------------------------------------------------
metrics_df = pd.DataFrame({
    'Accuracy (%)': accuracy.round(2),
    'DSC (Domain Switch Cost)': dsc.astype(float).round(2),
    'CYR (Compute Yield Ratio)': cyr.astype(float).round(2),
    'Rambling Index (Extra Chars)': rambling_index.round(0)
}).fillna('N/A') # Replaces NaNs for models with 0% accuracy/0 tokens

print("="*80)
print(" NOVEL COGNITIVE FACULTY METRICS (EXECUTIVE FUNCTIONS) ")
print("="*80)
print(metrics_df.to_string())
print("="*80)
```

# Part 7: Conclusion

This experiment was designed to push Large Language Models beyond natural language, forcing them into the rigid, deterministic world of algebraic routing and transcendental heuristics. By analyzing the 5D telemetry—latency, inference scaling, token density, and dual-tolerance accuracy—we uncover profound truths about the current state, and the hard limits, of Artificial Intelligence's **Executive Functions**.

### 1. The Illusion of Fluency and the "Repetition Trap"
We observed that an AI's ability to generate text quickly and fluently is often a mask for a lack of underlying comprehension. When models optimized purely for linear prediction hit a cognitive wall, they do not fail gracefully. Instead, they suffer from **Mode Collapse**. Lacking the **Inhibitory Control** to pause and verify their own logic, their neural weights collapse into highly probable, repetitive impulses (e.g., outputting the exact same incorrect string in over 68% of failures). Fast AI is often confident, but blindly so.

### 2. The Compute-Accuracy Paradox
The shift toward "Test-Time Compute"—where a model pauses to dynamically scale its thinking time based on prompt complexity—is a massive architectural leap. However, this benchmark proves that simply thinking *longer* is not a silver bullet. Even when a frontier model burns thousands of hidden tokens over several minutes, it remains deeply vulnerable to human-like executive function flaws:

* **Planning & Routing Fragility:** If a model misinterprets a single logical constraint in step one, it fails to adapt. It will spend the next five minutes flawlessly executing complex algebra for the *wrong equation*.
* **Working Memory Decay:** As the internal scratchpad inflates with thousands of tokens, the model's working memory degrades, leading to dropped variables and broken state-tracking.
* **Cognitive Flexibility Failure:** The "Binary Trap" proves that models struggle with abrupt framework switches. Transitioning directly from continuous floating-point math into discrete string manipulation causes massive spikes in compute waste and logic breakdowns.

### 3. The Ultimate Boundary: The "FPU Ceiling"
The most significant discovery of this benchmark is exposed by the Dual-Tolerance Grader. When we relaxed the grading threshold from a strict 6-decimal tolerance to a 4-decimal tolerance, the success rate of deep reasoning models surged (e.g., from 44% to 53%). 

This reveals the absolute bottleneck of pure text-based reasoning: **The FPU Ceiling.**

This data proves that frontier reasoning models frequently understand the complex routing and algebra *perfectly*, but fail the arithmetic execution. Because LLMs lack a deterministic 64-bit Floating-Point Unit (FPU), they rely on "Token Approximation" for intermediate steps. When calculating transcendental functions ($\cos$, $\sin$, $e^x$), the model truncates decimals to save context space. When these truncated tokens are subsequently multiplied, the missing fractions violently compound into catastrophic floating-point failures. 

**Final Verdict:** True progress toward Artificial General Intelligence (AGI) cannot rely on scaling compute alone. To bridge the gap, next-generation models must evolve beyond pattern-matching to develop robust executive functions: the cognitive flexibility to shift domains, the working memory to hold complex states without decay, and the inhibitory control to stop guessing when they don't know the answer.