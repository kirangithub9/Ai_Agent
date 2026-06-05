import os
import sys
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

load_dotenv(override=True)

# --- DEBUG: confirm which key got loaded ---
key = os.getenv("OPENAI_API_KEY")
if key:
    print(f"Loaded key: starts {key[:8]}... ends ...{key[-4:]}  (length {len(key)})")
else:
    print("No OPENAI_API_KEY found in environment!")
# -------------------------------------------

def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

def create_daily_thought() -> str:
    """Generate an inspirational daily thought using the LLM."""
    llm = ChatOpenAI(model="gpt-4o-mini")
    response = llm.invoke("Generate a short, unique inspirational daily thought in one or two sentences.")
    return response.content

agent = create_agent(
    model="openai:gpt-4o",    
    tools=[get_weather, create_daily_thought],
    system_prompt="You are a helpful assistant. Make sure that you only respond with whatever is coming as input to the agent, and do not add any extra commentary or explanation.",
)

if len(sys.argv) < 2:
    print("Usage: python dynamic_agent_input.py \"<your message>\"")
    sys.exit(1)

user_input = " ".join(sys.argv[1:])

result = agent.invoke({"messages": [{"role": "user", "content": user_input}]})
print(result["messages"][-1].content)
