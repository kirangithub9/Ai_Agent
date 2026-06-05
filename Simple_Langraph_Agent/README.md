# Simple LangGraph Agent

A simple AI agent built using LangChain and LangGraph that demonstrates dynamic tool use with OpenAI models.

## Features

- 🌤️ **Weather Tool** – Get weather info for any city
- 💡 **Daily Thought Tool** – Generate inspirational daily thoughts using GPT-4o-mini
- 🤖 **Dynamic Agent** – Powered by `gpt-4o` via LangChain's agent framework

## Prerequisites

- Python 3.9+
- OpenAI API Key

## Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/alumnx-ai-labs/Ai_Agent.git
   cd Ai_Agent/Simple_Langraph_Agent
   ```

2. **Create and activate a virtual environment**
   ```bash
   python -m venv myenv
   # Windows
   myenv\Scripts\activate
   # macOS/Linux
   source myenv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**

   Create a `.env` file in this folder:
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   ```

## Usage

Run the agent with a message:

```bash
python dynamic_agent_input.py "What's the weather in New York?"
python dynamic_agent_input.py "Give me an inspirational thought for today"
```

## Project Structure

```
Simple_Langraph_Agent/
├── dynamic_agent_input.py   # Main agent script
├── requirements.txt         # Python dependencies
├── README.md                # This file
└── .env                     # Environment variables (not tracked in git)
```
