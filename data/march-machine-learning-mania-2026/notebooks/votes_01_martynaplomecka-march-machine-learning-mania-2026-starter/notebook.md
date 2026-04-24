# 🏀 March Machine Learning Mania 2026 Starter

- **Author:** MartynaPlomecka
- **Votes:** 763
- **Ref:** martynaplomecka/march-machine-learning-mania-2026-starter
- **URL:** https://www.kaggle.com/code/martynaplomecka/march-machine-learning-mania-2026-starter
- **Last run:** 2026-02-19 19:55:34.913000

---

# 🏀 March Machine Learning Mania 2026 — ADK Starter

**Build an AI Agent Pipeline for March Madness Predictions using Google's Agent Development Kit (ADK)!**

This notebook demonstrates how to use [Google ADK](https://google.github.io/adk-docs/) to orchestrate a March Madness prediction pipeline with specialized agents. Each agent handles one stage — data loading, feature engineering, model training, or submission generation — coordinated through ADK's `SequentialAgent`.

### What is ADK?
[Agent Development Kit](https://github.com/google/adk-python) is Google's open-source framework for building AI agents. You define **tools** (Python functions), wire them to **LLM-powered agents**, and orchestrate multi-agent workflows using patterns like `SequentialAgent`, `ParallelAgent`, and `LoopAgent`.

### Pipeline Overview
```
┌─────────────┐    ┌──────────────────┐    ┌──────────────┐    ┌────────────────┐
│  Data Loader │───▶│ Feature Engineer │───▶│ Model Trainer│───▶│ Submission Gen │
│    Agent     │    │     Agent        │    │    Agent     │    │     Agent      │
└─────────────┘    └──────────────────┘    └──────────────┘    └────────────────┘
     load_data()      compute_elo()          train_model()    generate_submission()
```

The baseline model is intentionally simple (**Elo + seed → logistic regression**). The focus is on the **agentic architecture** — you're encouraged to improve the model!

> **Tip:** You need a [Gemini API key](https://aistudio.google.com/apikey). Add it as a Kaggle secret named `GOOGLE_API_KEY`.

## 1. Setup & Installation

```python
%%capture
!pip install google-adk>=1.0.0 scikit-learn pandas numpy
```

```python
import os, json, asyncio, warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

warnings.filterwarnings('ignore')

# ── Configure Gemini API Key ──
try:
    from kaggle_secrets import UserSecretsClient
    os.environ['GOOGLE_API_KEY'] = UserSecretsClient().get_secret('GOOGLE_API_KEY')
    print('✅ API key loaded from Kaggle Secrets')
except Exception:
    pass
# Or set directly: os.environ['GOOGLE_API_KEY'] = 'your-key'

assert 'GOOGLE_API_KEY' in os.environ, '❌ Set GOOGLE_API_KEY as a Kaggle Secret!'

# ── ADK Imports ──
from google.adk.agents import LlmAgent
from google.adk.agents.sequential_agent import SequentialAgent
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.genai.types import Content, Part

print('✅ All imports successful — ADK ready!')
```

```python
# ── Configuration ──
DATA_DIR = '/kaggle/input/competitions/march-machine-learning-mania-2026'
GEMINI_MODEL = 'gemini-2.5-flash'
CURRENT_SEASON = 2026

# Elo hyperparameters — tune these!
ELO_K = 20       # K-factor: how much each game shifts ratings
ELO_INIT = 1500  # Starting Elo for all teams
ELO_HCA = 100    # Home court advantage in Elo points

# Global state (populated by tools, shared across agents via output_key)
DATA = {}   # loaded dataframes
ELO = {}    # (season, team_id) -> elo rating
MODEL = None  # trained sklearn model
```

## 2. Define Tools (Python Functions)

In ADK, **tools** are plain Python functions with descriptive docstrings. The LLM reads the docstring to understand when and how to call each tool. Our tools do the actual computation — the agents orchestrate them.

> **Key ADK concept:** Tools return `dict` results that the agent can interpret and relay. The agent's `output_key` saves its summary to the shared session state, so the next agent in the pipeline can reference it.

```python
# ════════════════════════════════════════════════════════════════
# TOOL 1: Load competition data
# ════════════════════════════════════════════════════════════════

def load_competition_data() -> dict:
    """Load all March Madness competition CSV files and return a summary.

    Reads men's and women's team info, regular season results,
    tournament results, tournament seeds, and the sample submission file.

    Returns:
        dict: Summary with status, dataset sizes, and a message.
    """
    DATA['m_teams'] = pd.read_csv(f'{DATA_DIR}/MTeams.csv')
    DATA['w_teams'] = pd.read_csv(f'{DATA_DIR}/WTeams.csv')
    DATA['m_regular'] = pd.read_csv(f'{DATA_DIR}/MRegularSeasonCompactResults.csv')
    DATA['w_regular'] = pd.read_csv(f'{DATA_DIR}/WRegularSeasonCompactResults.csv')
    DATA['m_tourney'] = pd.read_csv(f'{DATA_DIR}/MNCAATourneyCompactResults.csv')
    DATA['w_tourney'] = pd.read_csv(f'{DATA_DIR}/WNCAATourneyCompactResults.csv')
    DATA['m_seeds'] = pd.read_csv(f'{DATA_DIR}/MNCAATourneySeeds.csv')
    DATA['w_seeds'] = pd.read_csv(f'{DATA_DIR}/WNCAATourneySeeds.csv')
    DATA['sample_sub'] = pd.read_csv(f'{DATA_DIR}/SampleSubmissionStage1.csv')

    return {
        'status': 'success',
        'seasons': f"{DATA['m_regular']['Season'].min()}-{DATA['m_regular']['Season'].max()}",
        'mens_teams': len(DATA['m_teams']),
        'womens_teams': len(DATA['w_teams']),
        'regular_season_games': len(DATA['m_regular']) + len(DATA['w_regular']),
        'tourney_games': len(DATA['m_tourney']) + len(DATA['w_tourney']),
        'submission_rows': len(DATA['sample_sub']),
        'message': 'All data loaded successfully. Ready for feature engineering.'
    }

print("✅ Tool 1 defined: load_competition_data")
```

```python
# ════════════════════════════════════════════════════════════════
# TOOL 2: Compute Elo ratings
# ════════════════════════════════════════════════════════════════

def compute_elo_ratings() -> dict:
    """Compute Elo ratings for all men's and women's teams across all seasons.

    Uses a standard Elo system with K-factor, home court advantage,
    and between-season regression toward the mean.

    Returns:
        dict: Summary with top-rated teams and total ratings computed.
    """
    def _run_elo(regular_df, tourney_df):
        elo = {}
        season_elos = {}
        all_games = pd.concat([regular_df, tourney_df]).sort_values(['Season', 'DayNum'])
        prev_season = None

        for _, row in all_games.iterrows():
            season = row['Season']
            if season != prev_season and prev_season is not None:
                for tid, r in elo.items():
                    season_elos[(prev_season, tid)] = r
                elo = {tid: 0.75 * r + 0.25 * ELO_INIT for tid, r in elo.items()}
            prev_season = season

            w_id, l_id = row['WTeamID'], row['LTeamID']
            w_elo = elo.get(w_id, ELO_INIT)
            l_elo = elo.get(l_id, ELO_INIT)

            # Home court adjustment
            w_loc = row.get('WLoc', 'N')
            w_adj = w_elo + (ELO_HCA if w_loc == 'H' else (-ELO_HCA if w_loc == 'A' else 0))

            # Expected win probability & update
            exp_w = 1.0 / (1.0 + 10 ** ((l_elo - w_adj) / 400.0))
            elo[w_id] = w_elo + ELO_K * (1.0 - exp_w)
            elo[l_id] = l_elo + ELO_K * (0.0 - (1.0 - exp_w))

        if prev_season:
            for tid, r in elo.items():
                season_elos[(prev_season, tid)] = r
        return season_elos

    m_elos = _run_elo(DATA['m_regular'], DATA['m_tourney'])
    w_elos = _run_elo(DATA['w_regular'], DATA['w_tourney'])
    ELO.update(m_elos)
    ELO.update(w_elos)

    # Top teams for display
    m_names = dict(zip(DATA['m_teams']['TeamID'], DATA['m_teams']['TeamName']))
    w_names = dict(zip(DATA['w_teams']['TeamID'], DATA['w_teams']['TeamName']))
    latest_m = max(s for s, _ in m_elos.keys())
    latest_w = max(s for s, _ in w_elos.keys())
    top_m = sorted([(tid, r) for (s, tid), r in m_elos.items() if s == latest_m], key=lambda x: -x[1])[:5]
    top_w = sorted([(tid, r) for (s, tid), r in w_elos.items() if s == latest_w], key=lambda x: -x[1])[:5]

    return {
        'status': 'success',
        'total_ratings': len(ELO),
        'top_mens': [f"{m_names.get(t, t)}: {r:.0f}" for t, r in top_m],
        'top_womens': [f"{w_names.get(t, t)}: {r:.0f}" for t, r in top_w],
        'message': f'Elo computed through {latest_m} (men) and {latest_w} (women).'
    }

print("✅ Tool 2 defined: compute_elo_ratings")
```

```python
# ════════════════════════════════════════════════════════════════
# TOOL 3: Train prediction model
# ════════════════════════════════════════════════════════════════

def _parse_seed(seed_str):
    """Extract numeric seed from string like 'W01', 'X16a' → 1, 16."""
    return int(seed_str[1:3])


def train_prediction_model() -> dict:
    """Train a logistic regression on Elo difference and seed difference.

    Builds training data from historical tournament matchups (2003+).
    Features: [elo_diff, seed_diff] where diff = Team1 - Team2
    (Team1 = lower TeamID). Evaluates with 5-fold cross-validation.

    Returns:
        dict: Summary with training size, Brier score, and model info.
    """
    global MODEL

    # Seed lookup
    seed_map = {}
    for df in [DATA['m_seeds'], DATA['w_seeds']]:
        for _, row in df.iterrows():
            seed_map[(row['Season'], row['TeamID'])] = _parse_seed(row['Seed'])

    # Build training set from tournament games
    X, y = [], []
    for t_df in [DATA['m_tourney'], DATA['w_tourney']]:
        for _, row in t_df.iterrows():
            season = row['Season']
            if season < 2003:
                continue

            w_id, l_id = row['WTeamID'], row['LTeamID']
            # Use PREVIOUS season Elo as pre-tournament rating
            w_elo = ELO.get((season - 1, w_id), ELO_INIT)
            l_elo = ELO.get((season - 1, l_id), ELO_INIT)
            w_seed = seed_map.get((season, w_id), 8)
            l_seed = seed_map.get((season, l_id), 8)

            # Convention: team1 = lower ID
            if w_id < l_id:
                X.append([w_elo - l_elo, l_seed - w_seed])
                y.append(1)
            else:
                X.append([l_elo - w_elo, w_seed - l_seed])
                y.append(0)

    X, y = np.array(X), np.array(y)

    # Train
    MODEL = LogisticRegression(C=1.0, solver='lbfgs')
    MODEL.fit(X, y)

    # Cross-val Brier score (lower = better)
    cv_probs = cross_val_score(
        LogisticRegression(C=1.0, solver='lbfgs'), X, y,
        scoring='neg_brier_score', cv=5
    )
    brier = -cv_probs.mean()

    return {
        'status': 'success',
        'training_games': len(y),
        'win_rate_label1': f"{y.mean():.3f}",
        'cv_brier_score': f"{brier:.4f}",
        'coefficients': {
            'elo_diff': f"{MODEL.coef_[0][0]:.6f}",
            'seed_diff': f"{MODEL.coef_[0][1]:.6f}",
            'intercept': f"{MODEL.intercept_[0]:.6f}"
        },
        'message': f'Model trained on {len(y)} games. CV Brier: {brier:.4f}'
    }

print("✅ Tool 3 defined: train_prediction_model")
```

```python
# ════════════════════════════════════════════════════════════════
# TOOL 4: Generate submission file
# ════════════════════════════════════════════════════════════════

def generate_submission() -> dict:
    """Generate predictions for every possible matchup and save submission.csv.

    For each row in the sample submission, extracts the two team IDs,
    computes features (Elo diff, seed diff), and predicts P(Team1 wins).
    Falls back to 0.5 if a team has no Elo rating.

    Returns:
        dict: Summary with number of predictions and output path.
    """
    sub = DATA['sample_sub'].copy()

    # Seed lookup for ALL seasons (not just current)
    seed_map = {}
    for df in [DATA['m_seeds'], DATA['w_seeds']]:
        for _, row in df.iterrows():
            seed_map[(row['Season'], row['TeamID'])] = _parse_seed(row['Seed'])

    preds = []
    for _, row in sub.iterrows():
        parts = row['ID'].split('_')
        season = int(parts[0])
        t1, t2 = int(parts[1]), int(parts[2])  # t1 < t2 by construction

        # Use prior season's Elo for each row's own season
        latest_season = season - 1

        e1 = ELO.get((latest_season, t1), ELO_INIT)
        e2 = ELO.get((latest_season, t2), ELO_INIT)
        s1 = seed_map.get((season, t1), 8)
        s2 = seed_map.get((season, t2), 8)

        features = np.array([[e1 - e2, s2 - s1]])
        prob = MODEL.predict_proba(features)[0][1]
        # Clip to avoid extreme probabilities
        prob = np.clip(prob, 0.01, 0.99)
        preds.append(prob)

    sub['Pred'] = preds
    output_path = '/kaggle/working/submission.csv'
    sub.to_csv(output_path, index=False)

    return {
        'status': 'success',
        'num_predictions': len(preds),
        'mean_pred': f"{np.mean(preds):.4f}",
        'std_pred': f"{np.std(preds):.4f}",
        'output_path': output_path,
        'message': f'Submission saved to {output_path} with {len(preds)} predictions.'
    }

print("✅ Tool 4 defined: generate_submission")
```

## 3. Define ADK Agents

Each agent is an `LlmAgent` with:
- **`instruction`**: tells the LLM what this agent's role is
- **`tools`**: the Python functions it can call
- **`output_key`**: saves the agent's final response to the shared session state, so the next agent can read it via `{key}` templating

We then compose them into a `SequentialAgent` that runs them in order — just like a data pipeline.

```python
# ── Agent 1: Data Loader ──
data_loader_agent = LlmAgent(
    name="DataLoaderAgent",
    model=GEMINI_MODEL,
    instruction="""You are a data loading specialist for the March Madness prediction pipeline.

Your job:
1. Call the `load_competition_data` tool to load all competition CSV files.
2. Report a brief summary of what was loaded (number of teams, games, seasons).
3. Confirm the data is ready for the next stage.

Be concise — just the key numbers and a confirmation.""",
    description="Loads and summarizes the competition dataset.",
    tools=[load_competition_data],
    output_key="data_summary"
)

# ── Agent 2: Feature Engineer ──
feature_engineer_agent = LlmAgent(
    name="FeatureEngineerAgent",
    model=GEMINI_MODEL,
    instruction="""You are a feature engineering specialist.

Previous stage summary: {data_summary}

Your job:
1. Call `compute_elo_ratings` to calculate Elo ratings for all teams.
2. Report the top-rated men's and women's teams.
3. Confirm features are ready for model training.

Be concise.""",
    description="Computes Elo ratings as features for prediction.",
    tools=[compute_elo_ratings],
    output_key="feature_summary"
)

# ── Agent 3: Model Trainer ──
model_trainer_agent = LlmAgent(
    name="ModelTrainerAgent",
    model=GEMINI_MODEL,
    instruction="""You are a model training specialist.

Previous stage summary: {feature_summary}

Your job:
1. Call `train_prediction_model` to train a logistic regression model.
2. Report the cross-validation Brier score and model coefficients.
3. Briefly interpret which feature (Elo or seed) matters more.

Be concise.""",
    description="Trains and evaluates the prediction model.",
    tools=[train_prediction_model],
    output_key="model_summary"
)

# ── Agent 4: Submission Generator ──
submission_agent = LlmAgent(
    name="SubmissionAgent",
    model=GEMINI_MODEL,
    instruction="""You are a submission generation specialist.

Previous stage summary: {model_summary}

Your job:
1. Call `generate_submission` to create predictions for all matchups.
2. Report where the submission file was saved and basic prediction stats.
3. Suggest 2-3 concrete ideas for improving the model.

Be concise.""",
    description="Generates the final submission CSV.",
    tools=[generate_submission],
    output_key="submission_summary"
)

print("✅ All 4 agents defined")
```

## 4. Build & Run the Sequential Pipeline

ADK's `SequentialAgent` executes sub-agents in order. Each agent writes its output to the session state via `output_key`, and the next agent reads it via `{key}` placeholders in its instruction.

This is the ADK equivalent of a data pipeline — deterministic ordering with LLM-powered execution at each step.

```python
# ── Compose the pipeline ──
pipeline = SequentialAgent(
    name="MarchMadnessPipeline",
    sub_agents=[
        data_loader_agent,
        feature_engineer_agent,
        model_trainer_agent,
        submission_agent,
    ],
    description="End-to-end March Madness prediction pipeline."
)

print("✅ Pipeline assembled: DataLoader → FeatureEngineer → ModelTrainer → Submission")
```

```python
# ── Run the pipeline ──

APP_NAME = "march_madness_2026"
USER_ID = "kaggle_user"
SESSION_ID = "pipeline_run_1"

async def run_pipeline():
    """Execute the full agent pipeline and print each agent's output."""
    session_service = InMemorySessionService()
    runner = Runner(
        agent=pipeline,
        app_name=APP_NAME,
        session_service=session_service,
    )

    # Create session
    session = await session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=SESSION_ID,
    )

    # Kick off the pipeline with a user message
    user_message = Content(
        role="user",
        parts=[Part(text="Run the full March Madness prediction pipeline. "
                         "Load data, compute features, train the model, "
                         "and generate the submission file.")]
    )

    print("🚀 Starting pipeline...\n")
    print("=" * 60)

    async for event in runner.run_async(
        user_id=USER_ID,
        session_id=session.id,
        new_message=user_message,
    ):
        # Print each agent's final response
        if event.is_final_response() and event.content and event.content.parts:
            author = event.author or "Pipeline"
            text = event.content.parts[0].text if event.content.parts[0].text else ""
            if text.strip():
                print(f"\n🤖 [{author}]")
                print("-" * 40)
                print(text.strip())
                print("=" * 60)

    print("\n✅ Pipeline complete!")
    return session

# In Kaggle notebooks, an event loop is already running
session = await run_pipeline()
```

## 5. Verify Submission

```python
print(DATA['sample_sub'].shape)
print(DATA['sample_sub'].head())
print(DATA['sample_sub']['ID'].str[:4].unique())
```

```python
# ── Check the output ──
sub = pd.read_csv('/kaggle/working/submission.csv')
print(f"Submission shape: {sub.shape}")
print(f"Prediction range: [{sub['Pred'].min():.4f}, {sub['Pred'].max():.4f}]")
print(f"Mean prediction:  {sub['Pred'].mean():.4f}")
print(f"Std prediction:   {sub['Pred'].std():.4f}")
print()
print(sub.head(10))
```

```python
# ── Quick sanity check: distribution of predictions ──
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 1, figsize=(8, 4))
ax.hist(sub['Pred'], bins=50, edgecolor='black', alpha=0.7, color='#2196F3')
ax.axvline(0.5, color='red', linestyle='--', label='50/50')
ax.set_xlabel('Predicted P(Team1 wins)')
ax.set_ylabel('Count')
ax.set_title('Distribution of Predictions')
ax.legend()
plt.tight_layout()
plt.show()
```

## 6. Ideas to Improve This Baseline

This starter gets you a valid submission, but there's lots of room to improve. Here are some directions:

### Better Features
- **Detailed box scores**: Use `MRegularSeasonDetailedResults.csv` for shooting %, rebounds, turnovers, etc.
- **Strength of schedule**: Weight Elo updates by opponent strength
- **Massey Ordinals**: The `MMasseyOrdinals.csv` file has rankings from 100+ systems (Sagarin, Pomeroy, RPI, etc.)
- **Recency weighting**: Give more weight to late-season games
- **Conference strength**: Compute conference-level Elo or use conference tournament results

### Better Models
- **Gradient boosting** (XGBoost, LightGBM) with more features
- **Ensemble** multiple models (Elo-based, seed-based, stats-based)
- **Neural nets** for learning complex feature interactions

### More Agents
- **`ParallelAgent`**: Compute men's and women's features simultaneously
- **`LoopAgent`**: Iterate on hyperparameter tuning (K-factor, home court advantage)
- **Analysis Agent**: Add an agent that explores the data and suggests features
- **Validation Agent**: An agent that checks predictions against historical tournament results

### ADK Patterns to Explore
```python
from google.adk.agents.parallel_agent import ParallelAgent
from google.adk.agents.loop_agent import LoopAgent

# Parallel feature computation
parallel_features = ParallelAgent(
    name="ParallelFeatures",
    sub_agents=[mens_elo_agent, womens_elo_agent, stats_agent]
)

# Iterative tuning
tuning_loop = LoopAgent(
    name="HyperparamTuner",
    sub_agents=[train_agent, evaluate_agent, adjust_agent],
    max_iterations=5
)
```

Good luck and happy forecasting! 🏀