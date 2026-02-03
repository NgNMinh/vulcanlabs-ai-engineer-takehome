import tiktoken

from langchain_core.messages.base import BaseMessage
from langchain_core.messages import AIMessage, HumanMessage

encoding = tiktoken.encoding_for_model("gpt-5-nano")


def count_tokens(messages: list[BaseMessage]) -> int:
    return sum(len(encoding.encode(m.content)) for m in messages)


def format_messages_as_conversation(messages: list[BaseMessage]) -> str:
    """Format message list as a single conversation string (User: / Assistant:)."""
    lines = []
    for m in messages:
        if isinstance(m, HumanMessage):
            role = "User"
        elif isinstance(m, AIMessage):
            role = "Assistant"
        else:
            role = "System"
        content = getattr(m, "content", "") or ""
        if isinstance(content, list):
            content = " ".join(
                getattr(part, "text", str(part)) for part in content
            )
        lines.append(f"{role}: {content}")
    return "\n\n".join(lines)
