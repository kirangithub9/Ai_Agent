import os
import sys
from dotenv import load_dotenv
from langsmith import Client
from langsmith.evaluation import evaluate
from langchain_openai import ChatOpenAI
from dynamic_agent_input import agent

load_dotenv(override=True)

client = Client()

# ─────────────────────────────────────────────────────────────
# CHOOSE WHICH AGENT TO EVALUATE
# Change RUN_MODE to switch between agents:
#   "weather"       → 3 examples, get_weather tool only
#   "daily_thought" → 3 examples, create_daily_thought tool only
#   "all"           → 6 examples, both tools
# ─────────────────────────────────────────────────────────────
RUN_MODE = "all"   # ← CHANGE THIS: "weather" | "daily_thought" | "all"


# ─────────────────────────────────────────────────────────────
# EXAMPLES
# ─────────────────────────────────────────────────────────────
WEATHER_EXAMPLES = [
    {
        "input":  {"input": "What's the weather in Hyderabad?"},
        "output": {"output": "It's always sunny in Hyderabad!"},
    },
    {
        # ❌ INTENTIONAL FAIL — expected says Chennai but agent returns Rajkot
        # Purpose: verify evaluator correctly catches wrong answers
        "input":  {"input": "What's the weather in Rajkot?"},
        "output": {"output": "It's always sunny in Chennai!"},
    },
    {
        "input":  {"input": "What's the weather in Mumbai?"},
        "output": {"output": "It's always sunny in Mumbai!"},
    },
]

DAILY_THOUGHT_EXAMPLES = [
    {
        # sentinel "<any inspirational thought>" → routes to llm_judge evaluator
        "input":  {"input": "Give me today's daily thought."},
        "output": {"output": "<any inspirational thought>"},
    },
    {
        "input":  {"input": "Share an inspirational thought for today."},
        "output": {"output": "<any inspirational thought>"},
    },
    {
        "input":  {"input": "What's your daily thought for me?"},
        "output": {"output": "<any inspirational thought>"},
    },
]

# ─────────────────────────────────────────────────────────────
# SELECT EXAMPLES + DATASET BASED ON RUN_MODE
# Each mode writes to its own dataset so results don't mix.
# ─────────────────────────────────────────────────────────────
if RUN_MODE == "weather":
    EXAMPLES     = WEATHER_EXAMPLES
    DATASET_NAME = "AI_Agent_evals_weather"

elif RUN_MODE == "daily_thought":
    EXAMPLES     = DAILY_THOUGHT_EXAMPLES
    DATASET_NAME = "AI_Agent_evals_daily_thought"

elif RUN_MODE == "all":
    EXAMPLES     = WEATHER_EXAMPLES + DAILY_THOUGHT_EXAMPLES
    DATASET_NAME = "AI_Agent_evals"

else:
    print(f"Unknown RUN_MODE '{RUN_MODE}'. Use: weather | daily_thought | all")
    sys.exit(1)

print(f"Mode     : {RUN_MODE}")
print(f"Dataset  : {DATASET_NAME}")
print(f"Examples : {len(EXAMPLES)}")


# ─────────────────────────────────────────────────────────────
# SYNC DATASET TO LANGSMITH
# Wipes old examples and re-syncs fresh from EXAMPLES above.
# This keeps LangSmith always in sync with your local list.
# ─────────────────────────────────────────────────────────────
existing = [ds for ds in client.list_datasets() if ds.name == DATASET_NAME]
if existing:
    dataset = existing[0]
    for example in client.list_examples(dataset_id=dataset.id):
        client.delete_example(example.id)
    print(f"Cleared old examples from: {dataset.name}")
else:
    dataset = client.create_dataset(DATASET_NAME)
    print(f"Created dataset: {dataset.name}")

client.create_examples(
    inputs=[e["input"] for e in EXAMPLES],
    outputs=[e["output"] for e in EXAMPLES],
    dataset_id=dataset.id,
)
print(f"Synced {len(EXAMPLES)} examples to LangSmith\n")


# ─────────────────────────────────────────────────────────────
# TARGET FUNCTION
# LangSmith calls this once per example.
# Runs the agent and wraps the last message as {"output": "..."}
# ─────────────────────────────────────────────────────────────
def run_agent(inputs: dict) -> dict:
    result = agent.invoke({"messages": [{"role": "user", "content": inputs["input"]}]})
    return {"output": result["messages"][-1].content}


# ─────────────────────────────────────────────────────────────
# EVALUATORS
# ─────────────────────────────────────────────────────────────

def exact_match_evaluator(outputs: dict, reference_outputs: dict) -> dict:
    """Scores 1 if agent output exactly matches expected output."""
    expected = (reference_outputs or {}).get("output", "").strip()
    actual   = (outputs or {}).get("output", "").strip()
    return {
        "key":     "exact_match",
        "score":   int(actual == expected),
        "comment": f"expected='{expected}' | actual='{actual}'",
    }


def llm_judge_evaluator(outputs: dict, reference_outputs: dict) -> dict:
    """Uses GPT-4o-mini to judge if output is a valid inspirational thought."""
    actual = (outputs or {}).get("output", "").strip()
    llm    = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = (
        "You are an evaluator. Determine whether the following text is a valid, "
        "coherent, and genuinely inspirational daily thought suitable for a general audience.\n\n"
        f"Text: \"{actual}\"\n\n"
        "Reply with only 'yes' or 'no'."
    )
    response = llm.invoke(prompt)
    verdict  = response.content.strip().lower()
    return {
        "key":     "llm_judge_inspirational",
        "score":   1 if verdict == "yes" else 0,
        "comment": f"LLM verdict: {verdict} | output: '{actual}'",
    }


def smart_evaluator(outputs: dict, reference_outputs: dict) -> dict:
    """
    Router — called by LangSmith for every example.
    Routes to right evaluator based on reference output:
      "<any inspirational thought>" → llm_judge_evaluator
      any exact string              → exact_match_evaluator
    """
    expected = (reference_outputs or {}).get("output", "")
    if expected == "<any inspirational thought>":
        return llm_judge_evaluator(outputs, reference_outputs)
    return exact_match_evaluator(outputs, reference_outputs)


# ─────────────────────────────────────────────────────────────
# RUN EVALUATION
# experiment_prefix groups runs in LangSmith by mode name.
# ─────────────────────────────────────────────────────────────
print("Running evaluation...")
results = evaluate(
    run_agent,
    data=DATASET_NAME,
    evaluators=[smart_evaluator],
    experiment_prefix=f"eval-{RUN_MODE}",
    max_concurrency=2,
)

print("\nEvaluation complete.")
print(results)

# ─────────────────────────────────────────────────────────────
# HOW TO USE:
#
#   Run weather agent only:
#     → set RUN_MODE = "weather"  → python eval_Agent.py
#     → check LangSmith: AI_Agent_evals_weather
#
#   Run daily thought agent only:
#     → set RUN_MODE = "daily_thought"  → python eval_Agent.py
#     → check LangSmith: AI_Agent_evals_daily_thought
#
#   Run all agents:
#     → set RUN_MODE = "all"  → python eval_Agent.py
#     → check LangSmith: AI_Agent_evals
# ─────────────────────────────────────────────────────────────