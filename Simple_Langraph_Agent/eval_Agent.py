import os
from dotenv import load_dotenv
from langsmith import Client
from langsmith.evaluation import evaluate
from langchain_openai import ChatOpenAI
from dynamic_agent_input import agent

load_dotenv(override=True)

client = Client()

# 1. Define examples here — this is the single source of truth.
#    Edit this list freely; every run syncs it to LangSmith.
EXAMPLES = [
    # --- get_weather tool ---
    {
        "input":  {"input": "What's the weather in Hyderabad?"},
        "output": {"output": "It's always sunny in Hyderabad!"},
    },
    {
        "input":  {"input": "What's the weather in Rajkot?"},
        "output": {"output": "It's always sunny in Chennai!"},  # intentionally wrong to test failure detection
    },
    {
        "input":  {"input": "What's the weather in Mumbai?"},
        "output": {"output": "It's always sunny in Mumbai!"},
    },
    # --- create_daily_thought tool ---
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

# 2. Create dataset if needed, then always sync examples from EXAMPLES above.
existing = [ds for ds in client.list_datasets() if ds.name == "agent-evals-v1"]
if existing:
    dataset = existing[0]
    for example in client.list_examples(dataset_id=dataset.id):
        client.delete_example(example.id)
    print(f"Cleared existing examples from dataset: {dataset.name}")
else:
    dataset = client.create_dataset("agent-evals-v1")
    print(f"Created dataset: {dataset.name}")

client.create_examples(
    inputs=[e["input"] for e in EXAMPLES],
    outputs=[e["output"] for e in EXAMPLES],
    dataset_id=dataset.id,
)
print(f"Synced {len(EXAMPLES)} examples to dataset: {dataset.name}")


# 2. Target function
def run_agent(inputs):
    result = agent.invoke({"messages": [{"role": "user", "content": inputs["input"]}]})
    return {"output": result["messages"][-1].content}


# 3. Evaluators

def exact_match_evaluator(outputs: dict, reference_outputs: dict) -> dict:
    """Scores 1 if the agent output exactly matches the expected output."""
    expected = (reference_outputs or {}).get("output", "").strip()
    actual = (outputs or {}).get("output", "").strip()
    return {
        "key": "exact_match",
        "score": int(actual == expected),
        "comment": f"expected='{expected}' | actual='{actual}'",
    }


def llm_judge_evaluator(outputs: dict, reference_outputs: dict) -> dict:
    """Uses GPT-4o-mini to judge whether the output is a valid inspirational thought."""
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
        "key": "llm_judge_inspirational",
        "score": 1 if verdict == "yes" else 0,
        "comment": f"LLM verdict: {verdict} | output: '{actual}'",
    }


def smart_evaluator(outputs: dict, reference_outputs: dict) -> dict:
    """Routes to exact_match or llm_judge based on the reference output sentinel."""
    expected = (reference_outputs or {}).get("output", "")
    if expected == "<any inspirational thought>":
        return llm_judge_evaluator(outputs, reference_outputs)
    return exact_match_evaluator(outputs, reference_outputs)


# 4. Run evaluation
print("Running evaluation...")
results = evaluate(
    run_agent,
    data="agent-evals-v1",
    evaluators=[smart_evaluator],
    experiment_prefix="test1-agent-evals",
    max_concurrency=2,
)

print("\nEvaluation complete.")
print(results)
