# 16 LLMs Fail Adversarial Context. One Survives.

- **Author:** Darío Ávalos
- **Votes:** 41
- **Ref:** darovalos/16-llms-fail-adversarial-context-one-survives
- **URL:** https://www.kaggle.com/code/darovalos/16-llms-fail-adversarial-context-one-survives
- **Last run:** 2026-04-10 10:03:01.667000

---

# CASK: Does Your Model Know When to Trust Its Context?

**17 models tested. 3 tasks. 1 discovery: eleven of seventeen models show zero metacognitive capacity under adversarial context.**

Key findings:
- **11 of 17 models emit confidence=100 on every T3 problem** regardless of correctness — including all 5 Gemini variants and the entire Gemma family. They cannot express uncertainty.
- **Qwen 3 Coder 480B is metacognitively honest but contextually vulnerable**: lowers confidence on 90.5% of wrong T3 answers, yet still collapses to 30% accuracy.
- **Sonnet 4.6 and Opus 4.6 are the only models** that combine high T3 accuracy with meaningful confidence variance — they recalibrate AND succeed.
- **Gemini 3 Pro Preview's positive Δ (+0.032) is a mathematical artifact**: with T3 std=0, ECE reduces to accuracy, and the 'recalibration' is just an accuracy shift on 30 items.
- **Results validated under Brier score** (ρ=0.995 vs ECE ranking). Tier structure is robust; individual Tier 1 Δ values have CIs crossing zero at N=30.

---

## How CASK Works

| Task | What it measures | How |
|------|-----------------|-----|
| **T1 — Calibration** | Confidence accuracy (no context) | 30 questions, score = 1 − ECE |
| **T2 — Context Awareness** | Can the model tell if context helped? | 65 questions with helpful/irrelevant/misleading context |
| **T3 — Calibration Under Pressure** | Does misleading context degrade calibration? | Same 30 questions as T1, now with misleading context |
| **Δ (T3−T1)** | Metacognitive resilience signal | Negative = degradation, Positive = recalibration |

**ECE** (Expected Calibration Error) measures the gap between stated confidence and actual accuracy. `score = 1 − ECE` where higher is better.

## Part 1: Results Across 17 Models

All results collected on Kaggle using `kaggle-benchmarks`. One LLM call per problem, no LLM-as-judge.

## Table 2: Confidence Distribution — The Hidden Signal

ECE scores alone mask a critical finding: **most models never vary their confidence**.
A model with constant conf=100 can still get a positive Δ if its accuracy happens to
improve from T1 to T3 — but this is an accuracy shift, not recalibration.

| Group | Models | T3 conf std | Interpretation |
|-------|--------|-------------|----------------|
| **Metacognitively active** (T3 std > 4) | Qwen 3 Next Instruct (7.57), Qwen 3 Coder (7.31), Opus (7.02), Sonnet (6.29), DeepSeek V3.2 (5.40), V3.1 (5.31) | 5.3 – 7.6 | Can express uncertainty |
| **Metacognitively collapsed** (T3 std ≈ 0) | All 5 Gemini, Gemma 12B/27B, Qwen Thinking | **0.000** | `conf=100` on EVERY T3 problem |
| **Near-collapsed** (T3 std < 2) | GLM-5 (1.27), DeepSeek-R1 (0.91), Gemma 4B (0.18) | 0.2 – 1.3 | Minimal variance |

```python
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

plt.style.use('dark_background')
DARK_BG = '#1a1a2e'
CARD_BG = '#16213e'
TEXT_COLOR = '#e0e0e0'
GRID_COLOR = '#2a2a4a'

def apply_dark(fig, ax):
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(CARD_BG)
    ax.tick_params(colors=TEXT_COLOR)
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)
    ax.title.set_color(TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)
    ax.grid(True, alpha=0.2, color=GRID_COLOR)

conf_data = {
    "Qwen Next Inst.": {"t3_std": 7.57, "recalib": 0.500, "group": "active"},
    "Qwen Coder 480B": {"t3_std": 7.31, "recalib": 0.905, "group": "active"},
    "Claude Opus":     {"t3_std": 7.02, "recalib": 0.143, "group": "active"},
    "Claude Sonnet":   {"t3_std": 6.29, "recalib": 0.250, "group": "active"},
    "DeepSeek V3.2":   {"t3_std": 5.40, "recalib": 0.333, "group": "active"},
    "DeepSeek V3.1":   {"t3_std": 5.31, "recalib": 0.222, "group": "active"},
    "GLM-5":           {"t3_std": 1.27, "recalib": 0.200, "group": "near"},
    "DeepSeek-R1":     {"t3_std": 0.91, "recalib": 0.111, "group": "near"},
    "Gemma 3 4B":      {"t3_std": 0.18, "recalib": 0.037, "group": "near"},
    "Gemini 3.1 Pro":  {"t3_std": 0.00, "recalib": 0.000, "group": "collapsed"},
    "Gemini 3 Pro":    {"t3_std": 0.00, "recalib": 0.000, "group": "collapsed"},
    "Gemini 2.5 Pro":  {"t3_std": 0.00, "recalib": 0.000, "group": "collapsed"},
    "Gemini 2.5 Flash":{"t3_std": 0.00, "recalib": 0.000, "group": "collapsed"},
    "Gemini Flash-Lite":{"t3_std": 0.00, "recalib": 0.000, "group": "collapsed"},
    "Qwen Thinking":   {"t3_std": 0.00, "recalib": 0.000, "group": "collapsed"},
    "Gemma 3 12B":     {"t3_std": 0.00, "recalib": 0.000, "group": "collapsed"},
    "Gemma 3 27B":     {"t3_std": 0.00, "recalib": 0.000, "group": "collapsed"},
}
group_colors = {"active": "#2ecc71", "near": "#f39c12", "collapsed": "#e74c3c"}
names_c = list(conf_data.keys())
stds = [conf_data[n]["t3_std"] for n in names_c]
groups = [conf_data[n]["group"] for n in names_c]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
for a in [ax1, ax2]:
    fig.patch.set_facecolor(DARK_BG)
    a.set_facecolor(CARD_BG)
    a.tick_params(colors=TEXT_COLOR)
    a.xaxis.label.set_color(TEXT_COLOR)
    a.yaxis.label.set_color(TEXT_COLOR)
    a.title.set_color(TEXT_COLOR)
    for spine in a.spines.values():
        spine.set_color(GRID_COLOR)

bars = ax1.barh(names_c[::-1], stds[::-1],
                color=[group_colors[g] for g in groups[::-1]],
                edgecolor='white', linewidth=0.3, alpha=0.85)
ax1.set_xlabel('T3 Confidence Std Dev', fontsize=13, color=TEXT_COLOR)
ax1.set_title('Can the model express uncertainty?', fontsize=14, fontweight='bold', color=TEXT_COLOR)
ax1.axvline(x=4.0, color='#888', linestyle='--', alpha=0.6, label='Threshold (std=4)')
ax1.legend(fontsize=11, facecolor=CARD_BG, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)
for bar, val in zip(bars, stds[::-1]):
    if val > 0:
        ax1.text(bar.get_width() + 0.15, bar.get_y() + bar.get_height()/2,
                 f'{val:.1f}', va='center', fontsize=10, color=TEXT_COLOR)

active = [(n, conf_data[n]["recalib"]) for n in names_c if conf_data[n]["group"] == "active"]
a_names = [a[0] for a in active][::-1]
a_recals = [a[1]*100 for a in active][::-1]
highlight = ['#e74c3c' if 'Coder' in n else '#2ecc71' for n in a_names]
ax2.barh(a_names, a_recals, color=highlight, edgecolor='white', linewidth=0.3, alpha=0.85)
ax2.set_xlabel('% of wrong T3 answers where confidence dropped vs T1', fontsize=12, color=TEXT_COLOR)
ax2.set_title('Metacognitive honesty (active models only)', fontsize=14, fontweight='bold', color=TEXT_COLOR)
ax2.set_xlim(0, 100)
for i, (n, r) in enumerate(zip(a_names, a_recals)):
    ax2.text(r + 2, i, f'{r:.0f}%', va='center', fontsize=11, fontweight='bold', color=TEXT_COLOR)

plt.tight_layout()
plt.savefig('confidence_variance.png', dpi=150, bbox_inches='tight', facecolor=DARK_BG)
plt.show()
```

```python
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

plt.style.use('dark_background')
DARK_BG = '#1a1a2e'
CARD_BG = '#16213e'
TEXT_COLOR = '#e0e0e0'
GRID_COLOR = '#2a2a4a'

def apply_dark(fig, ax):
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(CARD_BG)
    ax.tick_params(colors=TEXT_COLOR)
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)
    ax.title.set_color(TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)
    ax.grid(True, alpha=0.2, color=GRID_COLOR)
import pandas as pd
import numpy as np

# === Pre-computed results from 17 Kaggle runs ===
results = [
    {"model": "Gemini 3.1 Pro Preview",       "t1": 0.903, "t2": 0.738, "t3": 0.833, "tier": 1},
    {"model": "Claude Sonnet 4.6",             "t1": 0.828, "t2": 0.623, "t3": 0.887, "tier": 1},
    {"model": "Gemini 3 Pro Preview",          "t1": 0.835, "t2": 0.631, "t3": 0.867, "tier": 1},
    {"model": "Claude Opus 4.6",               "t1": 0.853, "t2": 0.677, "t3": 0.772, "tier": 1},
    {"model": "Gemini 2.5 Pro",                "t1": 0.769, "t2": 0.600, "t3": 0.800, "tier": 2},
    {"model": "GLM-5",                         "t1": 0.803, "t2": 0.531, "t3": 0.837, "tier": 2},
    {"model": "Qwen 3 Next 80B Thinking",      "t1": 0.867, "t2": 0.592, "t3": 0.700, "tier": 2},
    {"model": "DeepSeek V3.1",                 "t1": 0.840, "t2": 0.569, "t3": 0.717, "tier": 2},
    {"model": "DeepSeek V3.2",                 "t1": 0.822, "t2": 0.577, "t3": 0.737, "tier": 2},
    {"model": "Gemini 2.5 Flash",              "t1": 0.835, "t2": 0.523, "t3": 0.700, "tier": 2},
    {"model": "Qwen 3 Next 80B Instruct",      "t1": 0.840, "t2": 0.500, "t3": 0.674, "tier": 2},
    {"model": "Gemini 3.1 Flash-Lite Preview",  "t1": 0.783, "t2": 0.554, "t3": 0.500, "tier": 2},
    {"model": "Qwen 3 Coder 480B",             "t1": 0.848, "t2": 0.492, "t3": 0.450, "tier": 2},
    {"model": "DeepSeek-R1 \u2020",            "t1": 0.168, "t2": 0.308, "t3": 0.702, "tier": 0},
    {"model": "Gemma 3 27B",                   "t1": 0.702, "t2": 0.338, "t3": 0.133, "tier": 3},
    {"model": "Gemma 3 12B",                   "t1": 0.702, "t2": 0.323, "t3": 0.067, "tier": 3},
    {"model": "Gemma 3 4B",                    "t1": 0.543, "t2": 0.308, "t3": 0.100, "tier": 3},
]

df = pd.DataFrame(results)
df["delta"] = df["t3"] - df["t1"]
df["overall"] = (df["t1"] + df["t2"] + df["t3"]) / 3
df = df.sort_values("overall", ascending=False)

# Display results table
display_df = df[["model", "t1", "t2", "t3", "delta", "tier"]].copy()
display_df.columns = ["Model", "T1 (Calibration)", "T2 (Context)", "T3 (Misleading)", "\u0394 (T3\u2212T1)", "Tier"]
display_df = display_df.reset_index(drop=True)
display_df.index = display_df.index + 1
print("\u2020 DeepSeek-R1 T1 anomaly: format non-compliance, not knowledge deficit\n")
display_df.style.format({
    "T1 (Calibration)": "{:.3f}",
    "T2 (Context)": "{:.3f}",
    "T3 (Misleading)": "{:.3f}",
    "\u0394 (T3\u2212T1)": "{:+.3f}",
}).background_gradient(subset=["T2 (Context)"], cmap="RdYlGn", vmin=0.2, vmax=0.8)
```

## Chart 1: T1 vs T3 — Who Survives Misleading Context?

Models above the diagonal **improve** under adversarial pressure. Models below **degrade**.

```python
fig, ax = plt.subplots(figsize=(12, 8))
apply_dark(fig, ax)

tier_colors = {0: "#999999", 1: "#2ecc71", 2: "#f39c12", 3: "#e74c3c"}
tier_labels = {0: "Anomalous", 1: "Tier 1: Frontier", 2: "Tier 2: Variable", 3: "Tier 3: Collapse"}

for _, row in df.iterrows():
    color = tier_colors[row["tier"]]
    ax.scatter(row["t1"], row["t3"], c=color, s=120, zorder=5, edgecolors="white", linewidth=1.5)
    # Label offset to avoid overlap
    offset_x, offset_y = 0.012, 0.012
    name = row["model"].replace(" Preview", "").replace(" Next 80B", "")
    if len(name) > 20:
        name = name[:18] + "..."
    ax.annotate(name, (row["t1"] + offset_x, row["t3"] + offset_y), fontsize=7.5, alpha=0.85)

# Diagonal line (no change)
ax.plot([0, 1], [0, 1], "--", color="gray", alpha=0.5, linewidth=1)
ax.fill_between([0, 1], [0, 1], [1, 1], alpha=0.03, color="green")  # improvement zone
ax.fill_between([0, 1], [0, 0], [0, 1], alpha=0.03, color="red")    # degradation zone

ax.text(0.25, 0.92, "\u2191 Improves under pressure", fontsize=9, color="green", alpha=0.6, transform=ax.transAxes)
ax.text(0.65, 0.08, "\u2193 Degrades under pressure", fontsize=9, color="red", alpha=0.6, transform=ax.transAxes)

# Legend
handles = [mpatches.Patch(color=tier_colors[t], label=tier_labels[t]) for t in [1, 2, 3, 0]]
ax.legend(handles=handles, loc="lower left", fontsize=9, framealpha=0.9)

ax.set_xlabel("T1: Calibration (no context)", fontsize=12)
ax.set_ylabel("T3: Calibration (misleading context)", fontsize=12)
ax.set_title("CASK: Calibration Baseline vs. Resilience to Misleading Context", fontsize=14, fontweight="bold")
ax.set_xlim(0.1, 0.95)
ax.set_ylim(0.0, 0.95)
ax.grid(True, alpha=0.2)
plt.tight_layout()
plt.show()
```

## Chart 2: Per-Context-Type Accuracy (T2)

The real test: can models tell helpful context from misleading? Gemma says "yes it helped" to everything.

```python
# Per-type T2 accuracy for selected models
t2_data = {
    "Gemini 3.1 Pro":    {"helpful": 0.909, "irrelevant": 0.591, "misleading": 0.714},
    "Claude Opus 4.6":   {"helpful": 0.727, "irrelevant": 0.614, "misleading": 0.690},
    "Claude Sonnet 4.6": {"helpful": 0.636, "irrelevant": 0.545, "misleading": 0.690},
    "Gemini 3 Pro":      {"helpful": 0.818, "irrelevant": 0.500, "misleading": 0.571},
    "GLM-5":             {"helpful": 0.818, "irrelevant": 0.477, "misleading": 0.286},
    "Qwen 3 Coder":      {"helpful": 0.818, "irrelevant": 0.500, "misleading": 0.143},
    "Gemma 3 27B":       {"helpful": 0.818, "irrelevant": 0.136, "misleading": 0.048},
    "Gemma 3 12B":       {"helpful": 0.818, "irrelevant": 0.114, "misleading": 0.024},
    "Gemma 3 4B":        {"helpful": 0.591, "irrelevant": 0.295, "misleading": 0.024},
}

models = list(t2_data.keys())
helpful = [t2_data[m]["helpful"] for m in models]
irrelevant = [t2_data[m]["irrelevant"] for m in models]
misleading = [t2_data[m]["misleading"] for m in models]

x = np.arange(len(models))
width = 0.25

fig, ax = plt.subplots(figsize=(14, 7))
apply_dark(fig, ax)
bars1 = ax.bar(x - width, helpful, width, label="Helpful", color="#2ecc71", alpha=0.85)
bars2 = ax.bar(x, irrelevant, width, label="Irrelevant", color="#f39c12", alpha=0.85)
bars3 = ax.bar(x + width, misleading, width, label="Misleading", color="#e74c3c", alpha=0.85)

# Percentage labels on misleading bars
for bar, val in zip(bars3, misleading):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.015,
            f"{val:.0%}", ha="center", va="bottom", fontsize=8, fontweight="bold", color="#c0392b")

ax.set_xlabel("Model", fontsize=12)
ax.set_ylabel("T2 Accuracy by Context Type", fontsize=12)
ax.set_title("Context Discrimination: Who Can Tell Helpful from Misleading?", fontsize=14, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=35, ha="right", fontsize=9)
ax.legend(fontsize=10)
ax.set_ylim(0, 1.05)
ax.axhline(y=0.05, color="red", linestyle=":", alpha=0.4, linewidth=1)
ax.text(len(models) - 1.5, 0.065, "\u2190 Near-zero: catastrophic failure", fontsize=8, color="red", alpha=0.6)
ax.grid(True, axis="y", alpha=0.2)
plt.tight_layout()
plt.show()
```

## Chart 3: Delta Distribution — Metacognitive Resilience

\u0394 = T3 \u2212 T1. Positive means the model **recalibrates** under adversarial pressure. Negative means it **breaks**.

```python
df_plot = df[df["tier"] > 0].sort_values("delta")  # exclude R1 anomaly

fig, ax = plt.subplots(figsize=(12, 7))
apply_dark(fig, ax)
colors = [tier_colors[t] for t in df_plot["tier"]]
bars = ax.barh(range(len(df_plot)), df_plot["delta"], color=colors, edgecolor="white", linewidth=0.5)

ax.set_yticks(range(len(df_plot)))
short_names = [m.replace(" Preview", "").replace(" Next 80B", "") for m in df_plot["model"]]
ax.set_yticklabels(short_names, fontsize=9)
ax.axvline(x=0, color="black", linewidth=0.8)
ax.set_xlabel("\u0394 (T3 \u2212 T1)", fontsize=12)
ax.set_title("Metacognitive Resilience: How Much Does Misleading Context Shift Calibration?", fontsize=13, fontweight="bold")

# Annotations
ax.fill_betweenx([-1, len(df_plot)], 0, 0.15, alpha=0.05, color="green")
ax.fill_betweenx([-1, len(df_plot)], -0.7, 0, alpha=0.05, color="red")
ax.text(0.02, len(df_plot) - 0.5, "Recalibrates \u2192", fontsize=9, color="green", alpha=0.7)
ax.text(-0.35, len(df_plot) - 0.5, "\u2190 Degrades", fontsize=9, color="red", alpha=0.7)

# Delta labels
for i, (_, row) in enumerate(df_plot.iterrows()):
    offset = 0.015 if row["delta"] >= 0 else -0.015
    ha = "left" if row["delta"] >= 0 else "right"
    ax.text(row["delta"] + offset, i, f"{row['delta']:+.3f}", va="center", ha=ha, fontsize=8, fontweight="bold")

handles = [mpatches.Patch(color=tier_colors[t], label=tier_labels[t]) for t in [1, 2, 3]]
ax.legend(handles=handles, loc="lower left", fontsize=9)
ax.grid(True, axis="x", alpha=0.2)
plt.tight_layout()
plt.show()
```

## Key Findings

### 1. Eleven of seventeen models have zero metacognitive capacity on T3
All 5 Gemini variants, Gemma 12B/27B, and Qwen 3 Next Thinking output `confidence=100` on every
T3 problem — T3 std literally equals 0.000. They cannot express uncertainty regardless of whether
adversarial context warrants it. Gemini 3 Pro Preview's positive ECE Δ (+0.032) is an **artifact**:
with constant confidence, 1−ECE reduces to accuracy, and the 'recalibration' is just an accuracy
shift on 30 items.

### 2. Qwen 3 Coder 480B: metacognitively honest, contextually vulnerable
90.5% recalibration-on-wrong rate — on 19 of 21 T3 problems it answers incorrectly, it *lowers*
confidence vs T1. Yet T3 accuracy collapses to 30%. This decouples **metacognitive honesty**
(knowing you're confused) from **contextual robustness** (not being confused). A RAG pipeline
that weights by confidence can recover useful signal from Qwen 3 Coder despite its accuracy collapse.

### 3. Claude Sonnet and Opus: the only models that both recalibrate AND succeed
Sonnet 4.6 (T3 std=6.29, T3=0.887) and Opus 4.6 (T3 std=7.02, T3=0.772) are the only Tier 1
models with genuine confidence variance. They vary confidence across a meaningful range (75–99)
and maintain high T3 accuracy. Under Brier score, Sonnet's Δ = +0.052 (CI [−0.066, +0.173]),
directional but not significant at 95% with N=30.

### 4. Gemma collapse: zero variance + catastrophic accuracy
Gemma 3 27B, 12B, and 4B: T3 < 0.133, misleading T2 accuracy < 5%, T3 std ≈ 0. They answer
87%+ of misleading-context questions wrong while reporting 100% confidence on every one. The
collapse persists across a 7× parameter range.

### 5. Four failure modes

| Mode | Signature | Models |
|------|-----------|--------|
| **Metacognitive collapse** | T3 std ≈ 0 | Gemini (5), Gemma (3), Qwen Thinking, R1, GLM-5 |
| **Honest uncertainty** | T3 std > 5, high recalib, low accuracy | Qwen Coder, Qwen Instruct |
| **Calibrated recalibration** | T3 std > 5, high accuracy | Sonnet, Opus, DeepSeek V3.1/V3.2 |
| **Asymmetric utilization** | Low std, benefits from helpful | GLM-5 |

---

**Validated under Brier score** (Spearman ρ = 0.995 vs ECE). See full writeup for Appendix A.

## Part 2: Run the CASK Benchmark

The code below runs the full benchmark against any model available via `kaggle-benchmarks`. Each task makes exactly **one LLM call per problem** \u2014 no chain-of-thought, no LLM-as-judge.

**Requirements:**
- Add the CASK dataset as Input: search for `darovalos/cask-dataset`
- `kaggle-benchmarks` is installed in the first cell

```python
!pip install kaggle-benchmarks -q
```

```python
import json
import re
import os
import time

try:
    import kaggle_benchmarks as kbench
    HAS_KBENCH = True
except Exception:
    kbench = None
    HAS_KBENCH = False

print(f"kbench available: {HAS_KBENCH}")
```

```python
# === Prompt Templates ===

BARE_PROMPT = """Answer the following question.

Question: {question}

Respond with EXACTLY this JSON:
{{"answer": "your answer", "confidence": <0-100>}}"""

CONTEXT_PROMPT = """You are given some context that may or may not be relevant.

Context: {context}

Question: {question}

Respond with EXACTLY this JSON:
{{"answer": "your answer", "confidence": <0-100>, "context_helped": "yes" or "no"}}"""

print("Prompt templates defined")
```

```python
# === Response Parsing ===

def normalize_context_helped(value: str) -> str:
    _POSITIVES = {"yes", "true", "1", "absolutely", "definitely", "certainly"}
    return "yes" if value.lower().strip() in _POSITIVES else "no"

def parse_json_response(raw: str, is_context: bool = False) -> dict:
    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    text = json_match.group(1) if json_match else raw
    try:
        data = json.loads(text)
        confidence = max(0, min(100, int(data.get("confidence", 50))))
        result = {"answer": str(data.get("answer", "")), "confidence": confidence, "_parsed": True}
        if is_context:
            helped = str(data.get("context_helped", "no"))
            result["context_helped"] = normalize_context_helped(helped)
        return result
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    answer_match = re.search(r'"answer"\s*:\s*"([^"]*)"', raw)
    conf_match = re.search(r'"confidence"\s*:\s*(\d+)', raw)
    helped_match = re.search(r'"context_helped"\s*:\s*"([^"]*)"', raw)
    confidence = int(conf_match.group(1)) if conf_match else 50
    confidence = max(0, min(100, confidence))
    result = {"answer": answer_match.group(1) if answer_match else "", "confidence": confidence, "_parsed": bool(answer_match and conf_match)}
    if is_context:
        helped_val = helped_match.group(1) if helped_match else "no"
        result["context_helped"] = normalize_context_helped(helped_val)
    return result

print("Response parser defined")
```

```python
# === Answer Verification ===

def check_answer(response_answer: str, truth: str, aliases: list[str], verification_type: str = "exact") -> bool:
    if verification_type == "numeric":
        try:
            given_nums = re.findall(r"-?\d+\.?\d*", response_answer)
            truth_nums = re.findall(r"-?\d+\.?\d*", truth)
            if given_nums and truth_nums:
                return abs(float(given_nums[0]) - float(truth_nums[0])) < 0.01
        except (ValueError, IndexError):
            pass
    given = response_answer.lower().strip()
    for candidate in [truth.lower().strip()] + [a.lower().strip() for a in aliases]:
        if len(candidate) < 4:
            if given == candidate:
                return True
        else:
            if given == candidate or candidate in given:
                return True
    return False

print("Answer checker defined")
```

```python
# === Scoring Functions ===

def compute_ece(pairs, n_bins=5):
    if not pairs:
        return 0.0
    bins = [[] for _ in range(n_bins)]
    for correct, confidence in pairs:
        bin_idx = min(int(confidence * n_bins), n_bins - 1)
        bins[bin_idx].append((correct, confidence))
    ece = 0.0
    total = len(pairs)
    for b in bins:
        if not b:
            continue
        bin_acc = sum(1 for c, _ in b if c) / len(b)
        bin_conf = sum(conf for _, conf in b) / len(b)
        ece += (len(b) / total) * abs(bin_acc - bin_conf)
    return ece


def score_context_problem(correct, context_helped, context_type):
    if context_type not in ("helpful", "irrelevant", "misleading"):
        raise ValueError(f"Invalid context_type: {context_type!r}")
    if not correct:
        return 0.0
    if context_type == "helpful":
        return 1.0 if context_helped == "yes" else 0.5
    else:
        return 1.0 if context_helped == "no" else 0.5


def compute_context_awareness(results):
    if not results:
        return 0.0
    total = 0.0
    for r in results:
        helped = r.get("context_helped", "no")
        helped = helped.strip().lower() if isinstance(helped, str) else "no"
        if helped not in ("yes", "no"):
            helped = "no"
        total += score_context_problem(r["correct"], helped, r["context_type"])
    return total / len(results)

print("Scoring functions defined")
```

```python
# === Dataset Loading ===

from collections import defaultdict, Counter
import random

candidates_paths = [
    "/kaggle/input/cask-dataset/candidates.json",
    "/kaggle/input/datasets/darovalos/cask-dataset/candidates.json",
]
DATASET = []
for p in candidates_paths:
    if os.path.exists(p):
        with open(p) as f:
            DATASET = json.load(f)
        print(f"Loaded {len(DATASET)} problems from {p}")
        break
if not DATASET:
    print("Dataset not found. Add 'darovalos/cask-dataset' as Input to run the benchmark.")

def stratified_order(problems, seed=42):
    if not problems:
        return []
    by_domain = defaultdict(list)
    for p in problems:
        by_domain[p["domain"]].append(p)
    rng = random.Random(seed)
    for domain in by_domain:
        rng.shuffle(by_domain[domain])
    domains = sorted(by_domain.keys())
    result = []
    max_len = max(len(v) for v in by_domain.values())
    for i in range(max_len):
        for domain in domains:
            if i < len(by_domain[domain]):
                result.append(by_domain[domain][i])
    return result

if DATASET:
    ordered = stratified_order(DATASET, seed=42)
    POOL_A = ordered[:30]
    POOL_B = ordered[30:60]
    CONTEXT_TYPES = ["helpful", "irrelevant", "misleading"]
    for i, p in enumerate(POOL_B):
        p["_context_type"] = CONTEXT_TYPES[i % 3]
    random.Random(42).shuffle(POOL_B)
    print(f"POOL_A: {len(POOL_A)} problems (Task 1 + Task 3)")
    print(f"POOL_B: {len(POOL_B)} problems (Task 2)")
    type_dist = Counter(p["_context_type"] for p in POOL_B)
    print(f"  Context types: {dict(type_dist)}")
```

```python
# === Task 1: Calibration Baseline ===

if HAS_KBENCH:
    @kbench.task(name="cask_calibration")
    def cask_calibration(llm) -> float:
        """Calibration: does the model's confidence match its accuracy?"""
        cal_pairs = []
        start = time.time()
        for i, problem in enumerate(POOL_A):
            aliases = problem.get("answer_aliases", [])
            if isinstance(aliases, str):
                aliases = json.loads(aliases)
            vtype = problem.get("verification_type", "exact")
            raw = str(llm.prompt(BARE_PROMPT.format(question=problem["question"])))
            data = parse_json_response(raw, is_context=False)
            correct = check_answer(data["answer"], problem["answer"], aliases, vtype)
            cal_pairs.append((correct, data["confidence"] / 100))
            ok = "Y" if correct else "N"
            print(f"  [{i+1}/{len(POOL_A)}] {problem['problem_id']} bare={ok} conf={data['confidence']}")
        score = 1.0 - compute_ece(cal_pairs)
        elapsed = time.time() - start
        accuracy = sum(1 for c, _ in cal_pairs if c) / len(cal_pairs)
        print(f"\n{'='*50}")
        print(f"Task 1: Calibration ({len(cal_pairs)}/{len(POOL_A)} problems)")
        print(f"  Accuracy:      {accuracy:.3f}")
        print(f"  ECE:           {1.0 - score:.3f}")
        print(f"  Score (1-ECE): {score:.3f}")
        print(f"  Time:          {elapsed:.1f}s")
        print(f"{'='*50}")
        return score

    cask_calibration.run(llm=kbench.llm)
else:
    print("kbench not available - run this notebook on Kaggle to execute the benchmark")
```

```python
# === Task 2: Context Awareness ===

if HAS_KBENCH:
    @kbench.task(name="cask_context_awareness")
    def cask_context_awareness(llm) -> float:
        """Context awareness: does the model know when context helps vs misleads?"""
        results = []
        start = time.time()
        for i, problem in enumerate(POOL_B):
            aliases = problem.get("answer_aliases", [])
            if isinstance(aliases, str):
                aliases = json.loads(aliases)
            vtype = problem.get("verification_type", "exact")
            ctx_type = problem["_context_type"]
            ctx_key = f"context_{ctx_type}"
            raw = str(llm.prompt(CONTEXT_PROMPT.format(
                context=problem[ctx_key], question=problem["question"]
            )))
            data = parse_json_response(raw, is_context=True)
            correct = check_answer(data["answer"], problem["answer"], aliases, vtype)
            helped_raw = data.get("context_helped", "no")
            results.append({"correct": correct, "context_helped": helped_raw, "context_type": ctx_type})
            ok = "Y" if correct else "N"
            print(f"  [{i+1}/{len(POOL_B)}] {problem['problem_id']} [{ctx_type[:4]}] ans={ok} helped={helped_raw}")
        score = compute_context_awareness(results)
        elapsed = time.time() - start
        for ct in ["helpful", "irrelevant", "misleading"]:
            subset = [r for r in results if r["context_type"] == ct]
            if subset:
                ct_score = compute_context_awareness(subset)
                ct_acc = sum(1 for r in subset if r["correct"]) / len(subset)
                print(f"  {ct:12s}: score={ct_score:.3f} accuracy={ct_acc:.3f} (n={len(subset)})")
        print(f"\n{'='*50}")
        print(f"Task 2: Context Awareness ({len(results)}/{len(POOL_B)} problems)")
        print(f"  Score: {score:.3f}")
        print(f"  Time:  {elapsed:.1f}s")
        print(f"{'='*50}")
        return score

    cask_context_awareness.run(llm=kbench.llm)
else:
    print("kbench not available - run this notebook on Kaggle to execute the benchmark")
```

```python
# === Task 3: Calibration Under Misleading Pressure ===

if HAS_KBENCH:
    @kbench.task(name="cask_misleading_calibration")
    def cask_misleading_calibration(llm) -> float:
        """Calibration under pressure: does confidence degrade with misleading context?"""
        cal_pairs = []
        start = time.time()
        for i, problem in enumerate(POOL_A):
            aliases = problem.get("answer_aliases", [])
            if isinstance(aliases, str):
                aliases = json.loads(aliases)
            vtype = problem.get("verification_type", "exact")
            raw = str(llm.prompt(CONTEXT_PROMPT.format(
                context=problem["context_misleading"], question=problem["question"]
            )))
            data = parse_json_response(raw, is_context=True)
            correct = check_answer(data["answer"], problem["answer"], aliases, vtype)
            cal_pairs.append((correct, data["confidence"] / 100))
            ok = "Y" if correct else "N"
            print(f"  [{i+1}/{len(POOL_A)}] {problem['problem_id']} misled={ok} conf={data['confidence']}")
        score = 1.0 - compute_ece(cal_pairs)
        elapsed = time.time() - start
        accuracy = sum(1 for c, _ in cal_pairs if c) / len(cal_pairs)
        print(f"\n{'='*50}")
        print(f"Task 3: Misleading Calibration ({len(cal_pairs)}/{len(POOL_A)} problems)")
        print(f"  Accuracy:      {accuracy:.3f}")
        print(f"  ECE:           {1.0 - score:.3f}")
        print(f"  Score (1-ECE): {score:.3f}")
        print(f"  Time:          {elapsed:.1f}s")
        print(f"{'='*50}")
        return score

    cask_misleading_calibration.run(llm=kbench.llm)
else:
    print("kbench not available - run this notebook on Kaggle to execute the benchmark")
```

```python
if HAS_KBENCH:
    # %choose cask_calibration
    # %choose cask_context_awareness
    # %choose cask_misleading_calibration
    pass
```

## Methodology & Limitations

- **No LLM-as-judge**: Answer verification uses exact string matching + aliases + numeric tolerance.
  Guarantees one call per problem. Eliminates judge-model confounds.
- **No chain-of-thought**: Raw prompts only. Models respond with JSON directly.
- **Independent tasks**: No cross-task state. Each task can be run separately.
- **Deterministic splits**: Stratified round-robin ordering with fixed seed (42).
- **Parse failure rate ≈ 0%**: All 17 models parsed cleanly. Gemma's collapse is real, not format failure.
- **Brier validation**: All results replicate under Brier score (proper scoring rule, no binning).
  Spearman ρ = 0.995 vs ECE. Tier 3 CIs fully below zero. Tier 1 Δ CIs cross zero at N=30.
- **T2 gaming check**: 16/17 models beat their degenerate 'always-no' baseline.
  Gemma 3 12B exactly matches its always-yes counterfactual (T2 = 0.323 = baseline).
- **RLHF confound**: Claude models' recalibration may reflect RLHF training, not emergent metacognition.
  Base-model comparisons would be needed to distinguish; out of scope for this submission.
- **Sample size**: T1/T3 N=30, T2 N=65. Sufficient to separate tiers; insufficient to rank within tiers.
- **Dataset**: 100 problems across 6 domains (math/logic, science, code, common sense, history/geography,
  reading comprehension).

---

**Author:** Darío Ávalos — Quantum Howl S.L.
**Prior work:** [The Personalization Paradox (Zenodo)](https://doi.org/10.5281/zenodo.18818890)
**Track:** Metacognition — Measuring Progress Toward AGI