import sys
import json
from pathlib import Path
from langchain_core.messages import AIMessageChunk, HumanMessage, AIMessage

# Add src directory to path so vulcanlabs_ai can be imported
sys.path.insert(0, str(Path(__file__).parent / "src"))

from vulcanlabs_ai.workflow import graph

def load_conversation_log(filepath: Path) -> list[dict]:
    with open(filepath, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

# ANSI Color Codes
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"
DIM = "\033[2m"

def run_interaction(messages: list, thread_id: str, message_num: int = None):
    print(f"{DIM}{'-'*10} INTERNAL PROCESS STARTS {'-'*10}{RESET}")
    header_printed = False
    for message_chunk, metadata in graph.stream(
        {"messages": messages},
        config={"configurable": {"thread_id": thread_id}},
        stream_mode="messages",     
    ):
        if hasattr(message_chunk, "content") and message_chunk.content:
            if not header_printed and message_num is not None and isinstance(message_chunk, (AIMessageChunk, AIMessage)): 
                 # Check for AI message types
                print(f"{DIM}{'-'*11} INTERNAL PROCESS ENDS {'-'*10}{RESET}")
                print(f"\n{BLUE}{'='*20} [Message #{message_num}] ASSISTANT {'='*20}{RESET}")
                header_printed = True
            
            if isinstance(message_chunk, (AIMessageChunk, AIMessage)):
                print(message_chunk.content, end="", flush=True)
    
    if not header_printed and message_num is not None:
         # In case no content chunk was yielded but stream finished (rare for AIMessageChunk but possible if empty)
         print(f"{DIM}{'-'*11} INTERNAL PROCESS ENDS {'-'*10}{RESET}")

    print("\n")

def simulate_conversation_flow(title: str, goal: str, data_filename: str, thread_id: str):
    print(f"{YELLOW}{'='*50}")
    print(f"FLOW: {title}")
    print(f"{'='*50}{RESET}")
    print(f"Goal: {goal}")
    
    data_path = Path(__file__).parent / "test_data" / data_filename
    conversation = load_conversation_log(data_path)
    
    print(f"Loading {len(conversation)} messages from {data_path.name}...")
    
    # Counter for sequential messages (User=1, Assistant=2, etc.)
    message_num = 1
    
    for i, msg in enumerate(conversation):
        role = msg["role"]
        content = msg["content"]
        
        if role == "user":
            print(f"\n{GREEN}{'='*20} [Message #{message_num}] USER {'='*20}{RESET}")
            print(f"{content}")
            message_num += 1
            
            run_interaction([HumanMessage(content=content)], thread_id, message_num)
            message_num += 1

if __name__ == "__main__":
    simulate_conversation_flow(
        title="SESSION MEMORY TRIGGER",
        goal="Load a long conversation and observe summarization trigger.",
        data_filename="long_conversation.jsonl",
        thread_id="demo_memory_flow"
    )

    print("\n")

    simulate_conversation_flow(
        title="AMBIGUOUS QUERY HANDLING",
        goal="Demonstrate query rewriting and clarifying questions.",
        data_filename="ambiguous_conversation.jsonl",
        thread_id="demo_ambiguity_flow"
    )

    print("\n")

    simulate_conversation_flow(
        title="MIXED FLOW",
        goal="Demonstrate conversation with mixed context and potential ambiguity.",
        data_filename="mixed_flow.jsonl",
        thread_id="demo_mixed_flow"
    )