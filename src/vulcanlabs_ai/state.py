from langchain_core.messages.base import BaseMessage
from langgraph.graph import MessagesState
from pydantic import BaseModel, Field

from schema import QueryAmbiguityResult

###################
# Structured Outputs
###################
class SessionSummary(BaseModel):
    topics: list[str] | None = Field(description="Main discussion topics")
    key_facts: list[str] | None = Field(description="Important facts mentioned")
    user_goals: list[str] | None = Field(description="User's stated or inferred goals")
    summary_text: str = Field(description="Concise summary of the session")

class QueryAmbiguityResult(BaseModel):
    is_ambiguous: bool = Field(
        description="Whether the user's query is ambiguous or lacks sufficient clarity."
    )

    ambiguity_reason: Optional[str] = Field(
        default=None,
        description="Short explanation of why the query is ambiguous."
    )

    rewritten_query: Optional[str] = Field(
        default=None,
        description=(
            "A proposed rewritten version of the query if ambiguity is detected."
        )
    )
    
class ClarifyingQuestionResult(BaseModel):
    needs_clarification: bool = Field(
        description="Whether the system needs clarification from the use"
    )
    questions: list[str] | None = Field(
        default=None,
        description="1-3 clarifying questions to ask the user",
    )



###################
# State Definitions
###################
class State(MessagesState):
    summary: SessionSummary | None
    query_ambiguity: QueryAmbiguityResult | None
    clarifying_question: ClarifyingQuestionResult | None
    augmented_context: list[BaseMessage] | None
