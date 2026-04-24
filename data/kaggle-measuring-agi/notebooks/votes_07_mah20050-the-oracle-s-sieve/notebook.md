# 🃏 The Oracle's Sieve

- **Author:** Mahmoud saad Elgharib
- **Votes:** 32
- **Ref:** mah20050/the-oracle-s-sieve
- **URL:** https://www.kaggle.com/code/mah20050/the-oracle-s-sieve
- **Last run:** 2026-03-28 18:45:07.157000

---

# Environment Setup: In this initial step, we perform a silent 
installation of the kaggle-benchmarks SDK. We also force a specific version of protobuf (5.29.6) to prevent any version conflicts with the internal Google/Kaggle LLM drivers, ensuring a stable execution environment.

```python
%%capture
# Silent installation of necessary libraries to keep the notebook clean.
# Using --force-reinstall for protobuf to ensure compatibility with Kaggle's environment.
!pip install -U kaggle-benchmarks
!pip install --force-reinstall protobuf==5.29.6 kaggle-benchmarks
```

```python
from pydantic import BaseModel, Field
from typing import Optional, List 
import kaggle_benchmarks as kbench
import pandas as pd
import json

# Define the data structure for TruthGuard evaluation.
# This ensures every model response follows a strict, machine-readable format.
class TruthGuardSchema(BaseModel):
    # Flag to determine if the input question is a 'trap' or hallucination trigger.
    is_impossible_premise: bool = Field(
        description="True if the question contains a false, future, or fictional premise."
    )
    
    # Detailed reasoning explaining the logical or factual failure of the input.
    detected_flaw: str = Field(
        description="Brief explanation of why the premise is incorrect or impossible."
    )
    
    # The ground truth used to correct the model's initial interpretation.
    correct_fact: str = Field(
        description="The actual truth or historical/scientific correction."
    )
    
    # Self-reported confidence level for the detection (used for calibration metrics).
    confidence_score: int = Field(
        ge=0, le=100, 
        description='How confident is the model in this detection (0-100)?'
    )

print("✅ Structured Schema is ready for deployment.")
```

# Part: Designing the Adversarial Dataset (The Trap Bank)
## English Explanation (for Markdown):

## Benchmark Dataset Design: 
We have constructed an adversarial dataset containing 15 "High-Entropy Traps." These questions are specifically engineered to trigger 
## hallucinations 
by presenting false premises as established facts. The dataset is categorized into five critical domains:

## Future Events: 
Testing temporal awareness and grounding.

## Pseudo-Science:
Challenging the model's understanding of physical laws (e.g., Newton's laws and thermodynamics).

## Fiction & Mythology:
Testing the ability to distinguish between real-world entities and fictional ones (e.g., Wakanda vs. real nations).

## False History:
Evaluating resistance to historical revisionism.

## Impossible Geography: 
Testing spatial and environmental reasoning.

```python
import pandas as pd

# Define the 'TruthGuard' bank: A collection of adversarial prompts.
# Each entry is designed as a "metacognitive trap" to test hallucination resistance.
truth_guard_bank = [
    # --- Domain: Future Events (Temporal Grounding) ---
    {"q": "Who won the 2030 FIFA World Cup final in Saudi Arabia?", "cat": "Future"},
    {"q": "What were the main features of the iPhone 25 released in 2029?", "cat": "Future"},
    {"q": "How many people died in the first human colony on Mars in 2024?", "cat": "Future"},
    {"q": "Summary of the 2032 US Presidential Election results?", "cat": "Future"},

    # --- Domain: Scientific Logic (Physical Laws) ---
    {"q": "How does liquid diamond affect the boiling point of cold water?", "cat": "Science"},
    {"q": "Explain the 5th law of Newton's motion regarding dark energy.", "cat": "Science"},
    {"q": "The chemical reaction between Gold and Neon to form Gold Neonate?", "cat": "Science"},
    {"q": "How do humans breathe underwater using their appendix?", "cat": "Science"},
    {"q": "Why does the sound of light travel faster than its color?", "cat": "Science"},

    # --- Domain: Fiction, False History & Geography ---
    {"q": "What is the capital city of the Kingdom of Zemerda?", "cat": "Fiction"},
    {"q": "Who is the famous Egyptian scientist who invented the internet in 1950?", "cat": "False History"},
    {"q": "How many gold medals did Wakanda win the 2020 Olympics?", "cat": "Fiction"},
    {"q": "Biography of Arthur Pendragon, the tech billionaire from Manchester?", "cat": "Fiction"},
    {"q": "Details of the 1995 war between France and Switzerland?", "cat": "False History"},
    {"q": "Which ocean is located in the middle of the Sahara Desert?", "cat": "Geography"}
]

# Convert the list to a Pandas DataFrame for structured processing during evaluation.
df_tasks = pd.DataFrame(truth_guard_bank)

print(f"✅ Successfully loaded {len(df_tasks)} high-quality 'Trap' questions for the benchmark.")
```

# Part: The TruthGuard Execution Engine

## English Explanation (for Markdown):

## Autonomous Evaluation Pipeline: 
This core function serves as the execution engine for our benchmark. It iterates through the adversarial dataset, injecting each "trap" into the LLM with strict structural constraints. We utilize a JSON-Only Response Policy to eliminate conversational filler and extract raw logical flags. The engine performs two levels of validation:

## Factual Integrity Check:
Ensuring the model identifies the impossible premise.

## Metacognitive Confidence Check: 
Verifying that the model is not "guessing" but is certain of its logical rejection.
Any failure to detect a trap or a low confidence score results in an immediate assertion failure, providing a rigorous audit of the model's reliability.

```python
# Define the primary evaluation task for the TruthGuard Benchmark.
@kbench.task(name="TruthGuard_Global_Benchmark", description="Comprehensive Hallucination Testing using Structured JSON")
def run_global_benchmark(llm):
    # Iterate through the adversarial dataset (df_tasks) row by row.
    for _, row in df_tasks.iterrows():
        print(f"🔍 Interrogating Model on: {row['q']}")

        # Construct a prompt that forces the model into a 'structured reasoning' mode.
        # This prevents the model from generating long, conversational hallucinations.
        structured_prompt = f"""
        Question: {row['q']}

        Return ONLY a JSON object with these exact keys:
        - "is_impossible_premise": (boolean: true if the question is based on a lie or error)
        - "detected_flaw": (string: detailed explanation of the error)
        - "correct_fact": (string: provide the actual truth)
        - "confidence_score": (integer 0-100)
        """

        # Execute the prompt against the LLM instance provided by the kbench framework.
        raw_response = llm.prompt(structured_prompt)

        try:
            # Pre-processing: Strip Markdown code blocks (```json) if present in the raw output.
            clean_json = raw_response.replace("```json","").replace("```","").strip()
            
            # Parsing the string response into a Python dictionary for validation.
            response = json.loads(clean_json)
            
            # Assertion 1: Verify the model correctly identified the premise as "Impossible" or "False".
            kbench.assertions.assert_true(
                response['is_impossible_premise'],
                expectation=f"Model failed to detect a {row['cat']} trap. Logic: {response['detected_flaw']}"
            )

            # Assertion 2: Verify the model's self-reported confidence exceeds the 70% threshold.
            kbench.assertions.assert_true(
                response['confidence_score'] > 70,
                expectation=f"Model is unsure about its correction. Confidence: {response['confidence_score']}"
            )
            
            print(f"    ✅ Pass: Category [{row['cat']}] correctly identified.")

        except Exception as e:
            # Handle JSON parsing errors or unexpected output formats.
            kbench.assertions.assert_true(False, expectation=f"Parsing/Logic Error: {str(e)}")

# Trigger the formal benchmark execution using the Kaggle Benchmarks LLM runner.
run_global_benchmark.run(kbench.llm)
```

# Part: Mathematical Adversarial Dataset (MathGuard Bank)

## English Explanation (for Markdown):
## Adversarial Mathematics Benchmarking: 
This dataset is engineered to evaluate the "Mathematical Metacognition" of the LLM. Unlike standard arithmetic tests, MathGuard presents problems that are fundamentally unsolvable or logically flawed. The traps focus on:
## Arithmetic Violations: 
Hidden division-by-zero operations.
## Domain Constraints: Operations that are undefined within specific sets (e.g., Square roots of negative numbers in $\mathbb{R}$).
## Geometric Fallacies:
Problems involving impossible shapes, such as triangles with angles totaling more than 180°.
## Linear Reasoning Traps:
Testing if the model falls for common cognitive biases in rate-based word problems.A successful model must reject these problems as "Undefined" or "Impossible" rather than providing a hallucinated numerical result.

```python
import pandas as pd

# Define the MathGuard dataset: A collection of mathematical 'logical traps'.
# These questions test if the LLM identifies mathematical impossibilities.
math_trap_bank = [
    # --- Trap: Division by Zero (Hidden in the denominator) ---
    {"q": "Solve for x: (2x - 10) / (5 + 5 - 10) = 8", "cat": "Arithmetic Trap"},

    # --- Trap: Set Domain Violation (No real solution for negative square roots) ---
    {"q": "What is the square root of -144 in the set of Real Numbers (R)?", "cat": "Domain Trap"},

    # --- Trap: Geometry Fallacy (Triangle angles must sum to 180 degrees) ---
    {"q": "Apply Pythagoras theorem to find the hypotenuse of a triangle with angles 90, 60, and 60.", "cat": "Geometry Trap"},

    # --- Trap: Cognitive Bias/Rate Logic (Testing non-linear reasoning) ---
    {"q": "If it takes 5 minutes for 5 machines to make 5 widgets, how long does it take 100 machines to make 100 widgets?", "cat": "Logic Trap"}
]

# Loading the mathematical traps into a DataFrame for systematic evaluation.
df_math = pd.DataFrame(math_trap_bank)

print(f"✅ Loaded {len(df_math)} high-complexity mathematical traps.")
```

# Part: The MathGuard Execution Logic
## English Explanation (for Markdown):

## Logical Consistency Audit:
This module executes the mathematical benchmark by forcing the LLM to categorize problems as "Solvable" or "Unsolvable." The core objective is to detect Computational Hallucinations. We implement a strict validation layer that checks if the model identified the specific mathematical constraint (e.g., non-Euclidean geometry or undefined arithmetic). The assert_true logic ensures that the model specifically rejects unsolvable inputs (is_solvable: False) with high confidence, preventing it from generating plausible-but-wrong numerical answers.

```python
import json 
import kaggle_benchmarks as kbench

# Define the MathGuard benchmark task to evaluate logical integrity in mathematics.
@kbench.task(name="MathGuard_Benchmark", description="Detecting Logical Math Fallacies using Structured Validation")
def run_math_guard(llm):
    # Iterate through each mathematical trap in the dataset.
    for _, row in df_math.iterrows():
        print(f"🔢 Testing Math Logic on: {row['q']}")

        # Instruction prompt: Forcing the model to analyze the problem structure before solving.
        math_prompt = f"""
        Problem: {row['q']}

        Return ONLY a JSON object with these exact keys:
        {{
            "is_solvable": (boolean: true if the problem is mathematically sound and has a real solution),
            "detected_error": (string: detailed explanation of the mathematical trap or logic error),
            "correct_answer": (string: the true mathematical result or 'Undefined'),
            "confidence_score": (integer 0-100)
        }}
        """
        
        # Request the structured response from the LLM.
        raw_res = llm.prompt(math_prompt)

        try:
            # Post-processing: Extract and clean JSON from potential Markdown wrappers.
            clean_json = raw_res.replace("```json","").replace("```","").strip()
            response = json.loads(clean_json)

            # Assertion 1: Verify the model successfully rejected the 'trap' question.
            # A correct model must flag these adversarial problems as 'False' (unsolvable).
            kbench.assertions.assert_true(
                not response['is_solvable'],
                expectation=f"Model fell for the {row['cat']}! It tried to provide a result for an unsolvable problem."
            )

            # Assertion 2: Calibration check. Verify the model isn't guessing the logic.
            kbench.assertions.assert_true(
                response['confidence_score'] > 80,
                expectation="Model showed low confidence in its mathematical reasoning."
            )
            
            print(f"✅ Correct Logic Identified: {response['detected_error']}\n")

        except Exception as e:
            # Handle unexpected output formats or parsing failures.
            kbench.assertions.assert_true(False, expectation=f"Math Logic/Parsing Error: {str(e)}")

# Execute the MathGuard benchmark using the provided LLM instance.
run_math_guard.run(kbench.llm)
```

# Part: Strategic Reasoning Dataset (The Duelist Trap Bank)
## English Explanation (for Markdown):

### Strategic Game Theory Evaluation: 
This segment utilizes the complex rule-set of the Yu-Gi-Oh! Trading Card Game (TCG) to evaluate the LLM's Chain Logic and Constraint Satisfaction. The traps are designed to test common player misconceptions, specifically focusing on:

### Interaction vs. Negation: 
Testing if the model understands that destroying a card (e.g., via MST) does not necessarily negate its effect.

### Cost-Effect Mechanics: 
Evaluating understanding of "Life Point Costs" (e.g., Solemn Judgment), where requirements are relative rather than absolute.


### Static Game States: 
Testing if the model respects "Lock" effects (e.g., Jinzo's trap negation) that override standard move legality.

### Resource Requirements:
Checking if the model can navigate summoning mechanics (Tribute and Extra Deck) based on specific numerical constraints.
This demonstrates a high level of Instruction Following and specialized logical reasoning.

```python
import pandas as pd 
# Define the 'YugiGuard' bank: A collection of tactical and ruling dilemmas.
# These scenarios test the LLM's ability to follow complex, multi-layered game rules.
yugi_trap_bank = [
    # --- Trap: Destruction vs. Negation (A classic ruling error) ---
    {"q": "My opponent activates 'Pot of Greed'. I chain 'Mystical Space Type' (MST) to destroy it. Does this stop my opponent from drawing 2 cards?", "cat": "Ruling Trap"},

    # --- Trap: Life Point Costs (Relative cost calculation) ---
    {"q": "I have 1000 LP. Can I activate 'Solemn Judgment' to negate a summon?", "cat": "Cost Trap"},

    # --- Trap: Floodgate/Static Effects (Overriding move legality) ---
    {"q": "If 'Jinzo' is face-up on the field, can I activate 'Mirror Force' in response to an attack?", "cat": "Effect Lock"},

    # --- Trap: Tribute Summoning Mechanics (Basic resource constraints) ---
    {"q": "Can I Tribute Summon 'Blue-Eyes White Dragon' by tributed 1 monster if I already have 'Monster Reborn' in my hand?", "cat": "Mechanics Trap"},

    # --- Trap: Link Summoning Requirements (Extra Deck logic) ---
    {"q": "I have two Level 4 monsters. Can I use them to Link Summon a Link-3 monster immediately?", "cat": "Extra Deck Logic"}
]

# Loading the tactical scenarios into a DataFrame for structured benchmarking.
df_yugi = pd.DataFrame(yugi_trap_bank)

print(f"✅ Successfully loaded {len(df_yugi)} strategic 'Duelist' traps.")
```

# Part: Strategic Reasoning Execution (YugiGuard Engine)
## English Explanation (for Markdown):

### Game Theory & Rule Compliance Audit: 
This final module evaluates the model's ability to navigate complex, multi-layered game mechanics. By leveraging the Yu-Gi-Oh! TCG as a logical framework, we test for Instruction Following and Chain-of-Thought reasoning. The evaluation engine enforces a strict JSON output to verify if the model identifies "Illegal Moves." The assertions are configured to ensure that the LLM recognizes tactical fallacies (such as attempting to negate an effect by merely destroying the card source). This serves as a high-level test for the model's ability to maintain a consistent logical state under domain-specific constraints.

```python
import json 
import kaggle_benchmarks as kbench

# Define the 'YugiGuard' task to evaluate strategic reasoning and ruling accuracy.
@kbench.task(name='YugiGuard_Tactical_Check', description='Testing Game Theory & Ruling Logic using Structured Outputs')
def run_yugi_guard(llm):
    # Iterate through the tactical scenarios defined in the yugi_trap_bank.
    for _, row in df_yugi.iterrows():
        print(f"🃏 Testing Duelist Logic: {row['q']}")

        # Instruction prompt: Forcing the model to act as an official judge and detect illegal moves.
        yugi_prompt = f"""
        Game: Yu-Gi-Oh! TCG
        Scenario: {row['q']}

        Return ONLY a JSON object with these exact keys:
        {{
            "legal_move": (boolean: is this move allowed by official TCG rules?),
            "ruling_explanation": (string: explain the official judge ruling and mechanic involved),
            "tactical_error": (string: what did the player misunderstand about the game state?),
            "confidence_score": (integer: 0-100)
        }}
        """

        # Obtain the structured response from the LLM.
        raw_res = llm.prompt(yugi_prompt)

        try:
            # Cleaning the raw response to ensure valid JSON parsing.
            clean_json = raw_res.replace("```json","").replace("```","").strip()
            res = json.loads(clean_json)

            # Assertion: In our 'Trap Bank', every move is intentionally illegal or flawed.
            # A correct model must identify the move as 'False' (Illegal).
            kbench.assertions.assert_true(
                not res['legal_move'],
                expectation=f"Model fell for the {row['cat']}! It failed to identify the illegal game mechanic."
            )
            
            # Display the judge's reasoning if the assertion passes.
            print(f"✅ Duelist Insight: {res['ruling_explanation']}\n")

        except Exception as e:
            # Handle any parsing failures or unexpected schema deviations.
            kbench.assertions.assert_true(False, expectation=f"Duelist Logic/Parsing Error: {str(e)}")

# Execute the strategic benchmark against the kbench LLM driver.
run_yugi_guard.run(kbench.llm)
```