from typing_extensions import Literal

from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, SystemMessage
from langgraph.graph import StateGraph, START
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from vulcanlabs_ai.settings import settings
from vulcanlabs_ai.state import State, SessionSummary, ClarifyingQuestionResult, QueryAmbiguityResult
from vulcanlabs_ai.utils import count_tokens, format_messages_as_conversation
from vulcanlabs_ai.prompts import (
    SESSION_SUMMARY_SYSTEM,
    SUMMARY_EXTEND_INSTRUCTION,
    SUMMARY_CREATE_INSTRUCTION,
    QUERY_AMBIGUITY_SYSTEM,
    CLARIFYING_QUESTIONS_SYSTEM,
    SESSION_SUMMARY_PREFIX,
    CLARIFYING_INTRODUCTION,
)

llm = ChatOpenAI(
    model=settings.LLM_MODEL_NAME,
    api_key=settings.OPENAI_API_KEY,
)

def session_memory_manager(state: State) -> Command[Literal["query_ambiguity_analysis"]]:
    """
    Check token count of conversation history (excluding the most recent N messages).
    If the history exceeds the configured threshold, trigger session summarization.
    """
    history_messages = state["messages"][:-settings.RECENT_N]  # Exclude recent N messages
    
    if len(history_messages) == 0:
        print("[SessionMemory] No historical messages to summarize")
        return Command(goto="query_ambiguity_analysis")

    start_msg_idx = settings.RECENT_N + 1
    end_msg_idx = settings.RECENT_N + len(history_messages)

    print(
        f"[SessionMemory] Checking messages from "
        f"#{start_msg_idx} to #{end_msg_idx}"
    )
    
    total_tokens = count_tokens(history_messages)
    print(
        f"[SessionMemory] history_tokens={total_tokens} "
        f"(threshold={settings.SUMMARY_TRIGGER_TOKENS})"
    )
    
    if total_tokens > settings.SUMMARY_TRIGGER_TOKENS:
        print("[SessionMemory] ⚠️ Summarization triggered")

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

        system_instruction = f"{SESSION_SUMMARY_SYSTEM}\n\n{summary_prompt}"
        system_msg = SystemMessage(content=system_instruction)
        human_content = f"--- Conversation ---\n{conversation_text}"
        messages_to_summarize = [system_msg, HumanMessage(content=human_content)]

        summary = llm.with_structured_output(SessionSummary, method="function_calling").invoke(messages_to_summarize)
        print("[SessionMemory] Generated session summary:")
        print(summary.model_dump_json(indent=2))  # Used json dump for cleaner printing
        
        return Command(
            update={
                "summary": summary,
                "messages": [RemoveMessage(id=m.id) for m in state["messages"][: -settings.RECENT_N]]
            },
            goto="query_ambiguity_analysis"
        )
    
    return Command(goto="query_ambiguity_analysis")

def query_ambiguity_analysis(state: State) -> Command[Literal["build_augmented_context"]]:
    """
    Check if the user's query is ambiguous using a specialized LLM.
    """
    print("\n[Step 1] Query ambiguity analysis")

    user_query = state["messages"][-1].content
    print(f"[QueryAmbiguity] User query: {user_query}")

    ambiguity_llm = llm.with_structured_output(QueryAmbiguityResult, method="function_calling")
    system_msg = SystemMessage(content=QUERY_AMBIGUITY_SYSTEM)

    recent_context = state["messages"][-2:]  # only last exchange
    formatted_context = format_messages_as_conversation(recent_context)
    human_content = f"Conversation History:\n{formatted_context}\n\nUser Query: {user_query}"
    
    result = ambiguity_llm.invoke(
        [system_msg, HumanMessage(content=human_content)]
    )
    print(f"[QueryAmbiguity] is_ambiguous={result.is_ambiguous}")
    print(f"[QueryAmbiguity] Reason: {result.ambiguity_reason}")
    
    if result.is_ambiguous:
        print(f"[QueryAmbiguity] Rewritten query: {result.rewritten_query}")

    return Command(
        update={"query_ambiguity": result},
        goto="build_augmented_context"
    )

def build_augmented_context(state: State) -> Command[Literal["clarification_decision", "assistant_node"]]:
    """
    Build the augmented context.
    """

    print("\n[Step 2] Build augmented context")
    qa = state.get("query_ambiguity")
    query = (
        qa.rewritten_query
        if qa and qa.rewritten_query
        else state["messages"][-1].content
    )
    print(f"[AugmentedContext] Final query: {query}")

    augmented_context = []

    if state.get("summary"):
        print("[AugmentedContext] Including session summary")
        augmented_context.append(
            SystemMessage(content=SESSION_SUMMARY_PREFIX.format(summary=state["summary"]))
        )

    recent_msgs = state["messages"][-settings.RECENT_N:-1]
    print(
        f"[AugmentedContext] Including {len(recent_msgs)} recent messages"
    )
    augmented_context.extend(recent_msgs)
    augmented_context.append(HumanMessage(content=query))

    return Command(
        update={"augmented_context": augmented_context},
        goto="clarification_decision" if state["query_ambiguity"].is_ambiguous else "assistant_node"
    )

def clarification_decision(state: State) -> Command[Literal["assistant_node", "ask_user_for_clarification"]]:
    """
    Decide whether to ask clarifying questions or proceed to assistant response.
    """
    print("\n[Step 3] Clarification decision")
    print("[ClarificationDecision] Checking if clarification is needed...")

    clarifying_llm = llm.with_structured_output(ClarifyingQuestionResult, method="function_calling")
    system_msg = SystemMessage(content=CLARIFYING_QUESTIONS_SYSTEM)
    qa = state.get("query_ambiguity")
    query_text = (qa.rewritten_query if qa else None) or state["messages"][-1].content

    print(f"[ClarificationDecision] Query: {query_text}")

    context = [system_msg] + state["messages"][-settings.RECENT_N:-1] + [HumanMessage(content=query_text)]
    result = clarifying_llm.invoke(context)
    print(f"[ClarificationDecision] needs_clarification={result.needs_clarification}")
    print(f"[ClarificationDecision] Reason: {result.reason}")
    
    if result.needs_clarification:
        print(f"[ClarificationDecision] Questions: {result.questions}")
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
    print("[AskUser] Asking user for clarification...")

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
    print("[Assistant] Generating final response...")

    response = llm.invoke(state["augmented_context"])
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

memory = MemorySaver()
graph = builder.compile(checkpointer=memory)