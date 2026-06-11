import os
from dotenv import load_dotenv
from langsmith import Client
from langsmith.evaluation import evaluate
from langchain_openai import ChatOpenAI
from dynamic_agent_input import agent

load_dotenv(override=True)

client = Client()

# ─────────────────────────────────────────────────────────────
# 1. DEFINE EXAMPLES — single source of truth
#    Add / edit / remove examples here freely.
#    Every run wipes the dataset and re-syncs from this list,
#    so LangSmith always reflects exactly what is here.
# ─────────────────────────────────────────────────────────────
EXAMPLES = [
    # --- get_weather tool ---
    # ✅ PASS: agent should return "It's always sunny in Hyderabad!"
    {
        "input":  {"input": "What's the weather in Hyderabad?"},
        "output": {"output": "It's always sunny in Hyderabad!"},
    },
    # ❌ INTENTIONAL FAIL: expected output says Chennai but agent will say Rajkot.
    #    Purpose: verify the evaluator correctly detects wrong answers.
    {
        "input":  {"input": "What's the weather in Rajkot?"},
        "output": {"output": "It's always sunny in Chennai!"},
    },
    # ✅ PASS: straightforward city match
    {
        "input":  {"input": "What's the weather in Mumbai?"},
        "output": {"output": "It's always sunny in Mumbai!"},
    },

    # --- create_daily_thought tool ---
    # These 3 use the sentinel "<any inspirational thought>" which tells
    # smart_evaluator to route to llm_judge instead of exact_match.
    # ✅ PASS: any coherent inspirational text will score 1
    {
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
# 2. SYNC DATASET TO LANGSMITH
#    CHANGED: dataset name updated from "agent-evals-v1"
#             to "AI_Agent_evals" to match your LangSmith project.
#    Logic:
#      - If dataset exists → wipe all old examples → re-add fresh ones
#      - If dataset does not exist → create it → add examples
# ─────────────────────────────────────────────────────────────
DATASET_NAME = "AI_Agent_evals"  # ← CHANGED from "agent-evals-v1"

existing = [ds for ds in client.list_datasets() if ds.name == DATASET_NAME]
if existing:
    dataset = existing[0]
    # Wipe old examples so we always start clean
    for example in client.list_examples(dataset_id=dataset.id):
        client.delete_example(example.id)
    print(f"Cleared existing examples from dataset: {dataset.name}")
else:
    dataset = client.create_dataset(DATASET_NAME)
    print(f"Created dataset: {dataset.name}")

# Push all examples defined above to LangSmith
client.create_examples(
    inputs=[e["input"] for e in EXAMPLES],
    outputs=[e["output"] for e in EXAMPLES],
    dataset_id=dataset.id,
)
print(f"Synced {len(EXAMPLES)} examples to dataset: {dataset.name}")


# ─────────────────────────────────────────────────────────────
# 3. TARGET FUNCTION
#    This is the function LangSmith calls for each example.
#    It runs your agent and returns the last message content
#    as {"output": "..."} so evaluators can compare it.
# ─────────────────────────────────────────────────────────────
def run_agent(inputs: dict) -> dict:
    result = agent.invoke({"messages": [{"role": "user", "content": inputs["input"]}]})
    return {"output": result["messages"][-1].content}


# ─────────────────────────────────────────────────────────────
# 4. EVALUATORS
#    Three evaluators work together:
#
#    exact_match_evaluator  → used for weather questions
#                             scores 1 if output matches exactly, else 0
#
#    llm_judge_evaluator    → used for daily thought questions
#                             asks GPT-4o-mini "is this inspirational? yes/no"
#                             scores 1 for yes, 0 for no
#
#    smart_evaluator        → the router (this is what LangSmith calls)
#                             checks reference output:
#                               "<any inspirational thought>" → llm_judge
#                               anything else               → exact_match
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
    """Uses GPT-4o-mini to judge if the output is a valid inspirational thought."""
    actual = (outputs or {}).get("output", "").strip()
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = (
        "You are an evaluator. Determine whether the following text is a valid, "
        "coherent, and genuinely inspirational daily thought suitable for a general audience.\n\n"
        f"Text: \"{actual}\"\n\n"
        "Reply with only 'yes' or 'no'."
    )
    response = llm.invoke(prompt)
    verdict = response.content.strip().lower()
    return {
        "key":     "llm_judge_inspirational",
        "score":   1 if verdict == "yes" else 0,
        "comment": f"LLM verdict: {verdict} | output: '{actual}'",
    }


def smart_evaluator(outputs: dict, reference_outputs: dict) -> dict:
    """
    Router evaluator — called by LangSmith for every example.
    Routes to the right sub-evaluator based on the expected output:
      - "<any inspirational thought>" → llm_judge_evaluator
      - any exact string              → exact_match_evaluator
    """
    expected = (reference_outputs or {}).get("output", "")
    if expected == "<any inspirational thought>":
        return llm_judge_evaluator(outputs, reference_outputs)
    return exact_match_evaluator(outputs, reference_outputs)


# ─────────────────────────────────────────────────────────────
# 5. RUN EVALUATION
#    CHANGED: data= updated from "agent-evals-v1" to DATASET_NAME
#             so it always matches the dataset name defined above.
#    experiment_prefix → LangSmith groups runs under this name.
#    max_concurrency=2 → runs 2 examples in parallel (saves time).
# ─────────────────────────────────────────────────────────────
print("Running evaluation...")
results = evaluate(
    run_agent,
    data=DATASET_NAME,                    # ← CHANGED from hardcoded "agent-evals-v1"
    evaluators=[smart_evaluator],
    experiment_prefix="test1-agent-evals",
    max_concurrency=2,
)

print("\nEvaluation complete.")
print(results)

# ─────────────────────────────────────────────────────────────
# EXPECTED RESULTS IN LANGSMITH:
#
#  Example                          Evaluator               Score
#  ─────────────────────────────────────────────────────────────
#  Weather in Hyderabad             exact_match             1 ✅
#  Weather in Rajkot (wrong city)   exact_match             0 ❌ intentional
#  Weather in Mumbai                exact_match             1 ✅
#  Give me today's daily thought    llm_judge_inspirational 1 ✅
#  Share an inspirational thought   llm_judge_inspirational 1 ✅
#  What's your daily thought        llm_judge_inspirational 1 ✅
#
#  View results at:
#  https://smith.langchain.com → Datasets & Experiments → AI_Agent_evals
# ─────────────────────────────────────────────────────────────