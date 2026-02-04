# VulcanLabs AI Engineer Takehome

This project implements an advanced conversational AI agent using **LangGraph**. It features intelligent session memory management, query ambiguity resolution, and proactive clarification capabilities.

## 🚀 Setup Instructions

### Prerequisites
- Python 3.12 or higher
- [uv](https://github.com/astral-sh/uv) (recommended for package management) or `pip`

### Installation

1.  **Clone the repository** (if you haven't already).
2.  **Install dependencies**:
    ```bash
    uv sync
    # OR
    pip install -e .
    ```

3.  **Configure Environment**:
    Create a `.env` file in the root directory (copy from `.env.example` if available) and add your OpenAI API key:
    ```env
    OPENAI_API_KEY=sk-...
    # Optional: Override default model (default is gpt-5-nano)
    LLM_MODEL_NAME=gpt-4o-mini
    ```

## 🎮 How to Run the Demo

The `demo.py` script showcases the key capabilities of the system by simulating different conversation flows using pre-defined datasets.

Run the demo using:
```bash
uv run demo.py
# OR
python demo.py
```

### Demo Flows
1.  **SESSION MEMORY TRIGGER**:  
    Loads a long conversation history (`long_conversation.jsonl`) to demonstrate how the system detects token thresholds and generates a structured summary (Topics, Key Facts, User Goals) to compress context.
    
2.  **AMBIGUOUS QUERY HANDLING**:  
    Simulates a scenario (`ambiguous_conversation.jsonl`) where the user asks a vague question (e.g., "Which one is better?"). The system analyzes the context, rewrites the query, and/or generates clarifying questions.

3.  **MIXED FLOW**:
    A combined scenario (`mixed_flow.jsonl`) involving technical discussions to test both memory and context awareness.

## 🏗️ High-Level Design

The core logic is implemented as a **LangGraph** state machine (`src/vulcanlabs_ai/workflow.py`). The workflow consists of the following nodes:

1.  **Session Memory Manager (`session_memory_manager`)**:
    - Checks the token count of the conversation history (starting from the 3rd message (by defaul t) onwards).
    - If the count exceeds `SUMMARY_TRIGGER_TOKENS` (default: 400), it triggers an LLM call to **create** or **extend** a structured summary.
    - Older messages are then removed from the active context window, retained only as a summary.

2.  **Query Ambiguity Analysis (`query_ambiguity_analysis`)**:
    - Analyzes the latest user query in the context of recent messages.
    - Determines if the query is ambiguous.
    - If ambiguous, it generates a **rewritten query** that resolves references (e.g., "it" -> "the previous topic").

3.  **Build Augmented Context (`build_augmented_context`)**:
    - Constructs the final context for the assistant.
    - merges: `Session Summary` + `Recent N Messages` + `Rewritten User Query`.

4.  **Clarification Decision (`clarification_decision`)**:
    - Evaluates if the augmented context is sufficient to answer the user.
    - If essential information is missing, it decides to ask **Clarifying Questions**.

5.  **Response Generation**:
    - **`ask_user_for_clarification`**: Generates a response asking the user for specific details.
    - **`assistant_node`**: Generates the final helpful response based on the fully prepared context.

## ⚠️ Assumptions & Limitations

- **In-Memory Persistence**: The project currently uses `MemorySaver`, meaning conversation state is stored in memory and will be lost if the script terminates. For production, a persistent checkpointer (e.g., Postgres, Redis) looks be needed.
- **Context Window**: The number of recent messages kept in context is set to 3 for faster summarization in this demo; in practice, this should be 5 or more.
