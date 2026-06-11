import os
import sys
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langsmith import Client

load_dotenv(override=True)

# --- DEBUG: confirm which key got loaded ---
key = os.getenv("OPENAI_API_KEY")
if key:
    print(f"Loaded key: starts {key[:8]}... ends ...{key[-4:]}  (length {len(key)})")
else:
    print("No OPENAI_API_KEY found in environment!")
# -------------------------------------------

# ─────────────────────────────────────────────────────────────
# TOOL DEFINITIONS
# ─────────────────────────────────────────────────────────────
def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

def create_daily_thought() -> str:
    """Generate an inspirational daily thought using the LLM."""
    llm = ChatOpenAI(model="gpt-4o-mini")
    response = llm.invoke("Generate a short, unique inspirational daily thought in one or two sentences.")
    return response.content

# ─────────────────────────────────────────────────────────────
# AGENT
# ─────────────────────────────────────────────────────────────
agent = create_agent(
    model="openai:gpt-4o",
    tools=[get_weather, create_daily_thought],
    system_prompt="You are a helpful assistant. Make sure that you only respond with whatever is coming as input to the agent, and do not add any extra commentary or explanation.",
)

# ─────────────────────────────────────────────────────────────
# LANGSMITH DATASET AUTO-SYNC
# Detects which tool was used based on keywords in the input,
# then pushes the input + output to the right dataset.
#
# Routing logic:
#   "weather" in input → AI_Agent_evals_weather
#   "thought"/"inspirational" in input → AI_Agent_evals_daily_thought
#   anything else → AI_Agent_evals (default)
# ─────────────────────────────────────────────────────────────
def detect_dataset(user_input: str) -> str:
    """Detect which dataset to push result to based on input keywords."""
    lowered = user_input.lower()
    if "weather" in lowered or "temperature" in lowered or "sunny" in lowered:
        return "AI_Agent_evals_weather"
    elif "thought" in lowered or "inspirational" in lowered or "daily" in lowered:
        return "AI_Agent_evals_daily_thought"
    else:
        return "AI_Agent_evals"


def push_to_langsmith(user_input: str, agent_output: str):
    """
    Push the run result to the correct LangSmith dataset.
    Creates the dataset if it does not exist yet.
    Adds a new example row every time you run the agent.
    """
    try:
        client      = Client()
        dataset_name = detect_dataset(user_input)

        # Get or create the dataset
        existing = [ds for ds in client.list_datasets() if ds.name == dataset_name]
        if existing:
            dataset = existing[0]
        else:
            dataset = client.create_dataset(dataset_name)
            print(f"Created new dataset: {dataset_name}")

        # Add this run as a new example
        client.create_examples(
            inputs=[{"input": user_input}],
            outputs=[{"output": agent_output}],
            dataset_id=dataset.id,
        )
        print(f"Result saved to LangSmith → {dataset_name}")

    except Exception as e:
        # Never crash the agent just because LangSmith sync failed
        print(f"LangSmith sync failed (non-fatal): {e}")


# ─────────────────────────────────────────────────────────────
# MAIN — only runs when called directly
# Import guard ensures eval_Agent.py can import `agent`
# without triggering the CLI logic or LangSmith sync.
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python dynamic_agent_input.py \"<your message>\"")
        sys.exit(1)

    user_input = " ".join(sys.argv[1:])

    # Run the agent
    result = agent.invoke({"messages": [{"role": "user", "content": user_input}]})
    agent_output = result["messages"][-1].content

    # Print result to terminal
    print(agent_output)

    # Auto-push to LangSmith dataset
    push_to_langsmith(user_input, agent_output)