from typing_extensions import Literal

from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, SystemMessage
from langgraph.graph import StateGraph, START
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from settings import settings
from state import State, SessionSummary, ClarifyingQuestionResult, QueryAmbiguityResult
from utils import count_tokens, format_messages_as_conversation
from prompts import (
    SESSION_SUMMARY_SYSTEM,
    SUMMARY_EXTEND_INSTRUCTION,
    SUMMARY_CREATE_INSTRUCTION,
    QUERY_AMBIGUITY_SYSTEM,
    CLARIFYING_QUESTIONS_SYSTEM,
    SESSION_SUMMARY_PREFIX,
    CLARIFYING_INTRODUCTION,
)

llm = ChatOpenAI(
    model="gpt-5-nano",
    api_key=settings.OPENAI_API_KEY,
)


def session_memory_manager(state: State) -> Command[Literal["query_ambiguity_analysis"]]:
    """
    Check token count of conversation history (excluding the most recent N messages).
    If the history exceeds the configured threshold, trigger session summarization.
    """
    history_messages  = state["messages"][:-settings.RECENT_N]  # Loại bỏ N message gần nhất
    if len(history_messages) == 0:
        return Command(goto="query_ambiguity_analysis")

    total_tokens = count_tokens(history_messages)
    
    if total_tokens > settings.SUMMARY_TRIGGER_TOKENS:
        print(f"[SessionMemory] history_tokens={total_tokens}, threshold={settings.SUMMARY_TRIGGER_TOKENS}")
        print("[SessionMemory] Summarization triggered")

        # Summarize historical messages (excluding recent N messages)
        existing_summary: SessionSummary | None = state.get("summary")

        if existing_summary:
            summary_text = (
                f"Topics: {existing_summary.topics}\n"
                f"Key facts: {existing_summary.key_facts}\n"
                f"User goals: {existing_summary.user_goals}\n"
                f"Summary: {existing_summary.summary_text}"
            )
            summary_prompt = SUMMARY_EXTEND_INSTRUCTION.format(summary_text=summary_text)
        else:
            summary_prompt = SUMMARY_CREATE_INSTRUCTION

        conversation_text = format_messages_as_conversation(history_messages)

        system_msg = SystemMessage(content=SESSION_SUMMARY_SYSTEM)
        human_content = f"{summary_prompt}\n\n--- Conversation ---\n{conversation_text}"
        messages_to_summarize = [system_msg, HumanMessage(content=human_content)]

        summary = llm.with_structured_output(SessionSummary, method="function_calling").invoke(messages_to_summarize)
        
        return Command(
            update={
                "summary": summary,
                "messages": [RemoveMessage(id=m.id) for m in state["messages"][: -settings.RECENT_N]]
            },
            goto="query_ambiguity_analysis"
        )
    
    return Command(goto="assistant_node")

def query_ambiguity_analysis(state: State) -> Command[Literal["build_augmented_context"]]:
    """
    Check if the user's query is ambiguous using a specialized LLM.
    """
    ambiguity_llm = llm.with_structured_output(QueryAmbiguityResult, method="function_calling")
    system_msg = SystemMessage(content=QUERY_AMBIGUITY_SYSTEM)
    result = ambiguity_llm.invoke([system_msg, HumanMessage(content=state["messages"][-1].content)])

    return Command(
        update={"query_ambiguity": result},
        goto="build_augmented_context"
    )

def build_augmented_context(state: State) -> Command[Literal["clarification_decision", "assistant_node"]]:
    qa = state.get("query_ambiguity")
    query = (
        qa.rewritten_query
        if qa and qa.rewritten_query
        else state["messages"][-1].content
    )

    augmented_context = []

    if state.get("summary"):
        augmented_context.append(
            SystemMessage(content=SESSION_SUMMARY_PREFIX.format(summary=state["summary"]))
        )

    augmented_context.extend(state["messages"][-settings.RECENT_N:-1])
    augmented_context.append(HumanMessage(content=query))

    return Command(
        update={"augmented_context": augmented_context},
        goto="clarification_decision" if state["query_ambiguity"].is_ambiguous else "assistant_node"
    )

def clarification_decision(state: State) -> Command[Literal["assistant_node", "ask_user_for_clarification"]]:

    clarifying_llm = llm.with_structured_output(ClarifyingQuestionResult, method="function_calling")
    system_msg = SystemMessage(content=CLARIFYING_QUESTIONS_SYSTEM)
    qa = state.get("query_ambiguity")
    query_text = (qa.rewritten_query if qa else None) or state["messages"][-1].content
    context = [system_msg] + state["messages"][-settings.RECENT_N:-1] + [HumanMessage(content=query_text)]
    result = clarifying_llm.invoke(context)
    if result.needs_clarification:
        return Command(
            update={"clarifying_question": result},
            goto="ask_user_for_clarification"
        )
    return Command(
        goto="assistant_node"
    )

def ask_user_for_clarification(state: State):
    """
    Handle clarifying questions and generate assistant response.
    """
    cq = state["clarifying_question"]
    questions_text = "\n".join(f"- {q}" for q in (cq.questions or []))
    return {
        "messages": [
            AIMessage(content=CLARIFYING_INTRODUCTION + questions_text)
        ]
    }

def assistant_node(state: State):
    """
    Generate a response based on the conversation history.
    """
    prompt = state["augmented_context"] if state.get("augmented_context") else state["messages"]

    response = llm.invoke(prompt)
    return {
        "messages": response
    }

builder = StateGraph(State)
builder.add_edge(START, "session_memory_manager")
builder.add_node("session_memory_manager", session_memory_manager)
builder.add_node("query_ambiguity_analysis", query_ambiguity_analysis)
builder.add_node("build_augmented_context", build_augmented_context)
builder.add_node("clarification_decision", clarification_decision)
builder.add_node("ask_user_for_clarification", ask_user_for_clarification)
builder.add_node("assistant_node", assistant_node)
# Compile with checkpointer for persistence, in case run graph with Local_Server --> Please compile without checkpointer
memory = MemorySaver()
graph = builder.compile(checkpointer=memory)

# graph_image = graph.get_graph(xray=True).draw_mermaid_png()
# with open("graph.png", "wb") as f:
#     f.write(graph_image)  # Chỉ cần ghi trực tiếp bytes vào file

# print("Graph saved as graph.png, open it manually.")